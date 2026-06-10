# -*- coding: utf-8 -*-
import os
import csv
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# robust injection 分塊，避免顯存爆
EVAL_INJECTION_CHUNK = 50

INJECTION_SWEEP_LIST = [    # 掃描injection
    0.000,
    0.005,
    0.015,
    0.025,
    0.035,
    0.045,
    0.055,
    0.065,
    0.075,
    0.085,
    0.095,
]

# ================================
# Helpers
# ================================
def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def format_float_for_path(value: float) -> str:
    """
    將 float 轉成適合資料夾名稱的字串。
    例：
        0.125 -> 0p125
        0.5   -> 0p5
        1.0   -> 1p0
        200.0 -> 200
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def build_st_paths(mode: str, penalty: float):
    """
    根據 mode 與 penalty 建立 ST checkpoint 路徑。
    REG : ST_sweep/REG_penalty_XXX/
    ROB : ST_sweep/ROB_penalty_XXX/
    """
    mode = mode.lower()

    if mode not in ("reg", "rob"):
        raise ValueError(f"mode 必須是 'reg' 或 'rob'，收到：{mode}")

    penalty_tag = format_float_for_path(penalty)

    if mode == "reg":
        run_name = f"REG_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm.ckpt"
        radar_ckpt_name = "short_radar.ckpt"
    else:
        run_name = f"ROB_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm_robust.ckpt"
        radar_ckpt_name = "short_radar_robust.ckpt"

    run_dir = os.path.join(ST_SWEEP_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, comm_ckpt_name),
        "radar_ckpt_path": os.path.join(ckpt_dir, radar_ckpt_name),
    }


def build_eval_dir(power_norm: bool):
    """
    建立 evaluate.py 輸出資料夾。
    """
    norm_tag = "powernorm" if power_norm else "raw"

    eval_dir = os.path.join(
        BASE_RUN_DIR,
        "eval_results",
        f"evaluate_setting_penalty_{norm_tag}"
    )

    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


def reset_eval_seed():
    """
    每次 REG / ROB 評估前，將亂數狀態重設回 settings.py 的 RANDOM_SEED。
    目的:
        1. REG 與 ROB 在同一個 test injection variance 下使用同一批 injection noise。
        2. 不同 test injection variance 使用同一批 Gaussian base noise 的縮放版本。
        3. 減少 Monte Carlo randomness 對 REG/ROB 比較的干擾。
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def normalize_injected_channel_power(
    injected: torch.Tensor,
    reference: torch.Tensor,
    norm_dims,
) -> torch.Tensor:
    """
    將 injected channel 的 power normalize 回 reference channel。
    """
    ref_norm = torch.linalg.vector_norm(
        reference,
        ord=2,
        dim=norm_dims,
        keepdim=True,
    )

    inj_norm = torch.linalg.vector_norm(
        injected,
        ord=2,
        dim=norm_dims,
        keepdim=True,
    ).clamp_min(1e-12)

    return injected * (ref_norm / inj_norm)


def apply_power_normalization_to_injected_channels(
    h_dk_inj: torch.Tensor,
    h_rk_inj: torch.Tensor,
    G_inj: torch.Tensor,
    g_dt_inj: torch.Tensor,
    h_dk_ref: torch.Tensor,
    h_rk_ref: torch.Tensor,
    G_ref: torch.Tensor,
    g_dt_ref: torch.Tensor,
    power_norm: bool,
):
    """
    對 injected channels 做 power normalization
    若 power_norm=False，直接回傳原始 injected channels
    """
    if not power_norm:
        return h_dk_inj, h_rk_inj, G_inj, g_dt_inj

    # h_dk / h_rk：對每個 UE 的 channel vector 分別 normalize
    h_dk_inj = normalize_injected_channel_power(injected=h_dk_inj,reference=h_dk_ref,norm_dims=1,)

    h_rk_inj = normalize_injected_channel_power(injected=h_rk_inj,reference=h_rk_ref,norm_dims=1,)

    # G：對整個 BS-RIS matrix 做 Frobenius norm normalize
    G_inj = normalize_injected_channel_power(injected=G_inj,reference=G_ref,norm_dims=(1, 2),)

    # g_dt：target channel vector normalize
    g_dt_inj = normalize_injected_channel_power(injected=g_dt_inj,reference=g_dt_ref,norm_dims=(1, 2),)

    return h_dk_inj, h_rk_inj, G_inj, g_dt_inj


# ================================
# Dataset
# ================================
def load_test_dataset(npz_path: str):
    """
    載入 test dataset。
    回傳:
        dataset : dict
    """
    required_keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "pl_BS_TAR_BS",
        "st_h_dk_hat",
        "st_h_rk_hat",
        "st_G_hat",
        "st_g_dt_hat",
    ]

    with np.load(npz_path) as data:
        dataset = {key: data[key] for key in required_keys}

    print(f"[test] preloaded into RAM: {npz_path}")
    print(f"[test] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[test] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[test] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[test] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[test] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))
    print(f"[test] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")

    return dataset


def extract_shortterm_batch(
    dataset,
    layout_id: int,          # 指定一個layout
    channel_ids: np.ndarray, # 一個array 可以取很多est channels
):
    """
    取出指定 layout 與 channel ids 的 est channels
    並轉為 torch tensor 準備輸入 NN
    """
    return {
        "pl_BS_UE": dataset["pl_BS_UE"][layout_id],
        "pl_BS_RIS_UE": dataset["pl_BS_RIS_UE"][layout_id],
        "pl_BS_TAR_BS": dataset["pl_BS_TAR_BS"][layout_id],

        "h_dk_hat": torch.as_tensor(
            dataset["st_h_dk_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "h_rk_hat": torch.as_tensor(
            dataset["st_h_rk_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "G_hat": torch.as_tensor(
            dataset["st_G_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "g_dt_hat": torch.as_tensor(
            dataset["st_g_dt_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
    }


# ================================
# Model loading
# ================================
def load_longterm_model(longterm_ckpt_path: str):
    """
    載入 shared long-term model。
    """
    if not os.path.exists(longterm_ckpt_path):
        raise FileNotFoundError(f"找不到 long-term checkpoint：{longterm_ckpt_path}")

    print(f"[LOAD] longterm_ckpt_path = {longterm_ckpt_path}")

    longterm_net = LongTermPositionNet(ckpt_kind=None).to(DEVICE)
    longterm_net.load_model(path=longterm_ckpt_path, verbose=True)

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    return longterm_net


def load_shortterm_model(mode: str, penalty: float):
    """
    載入指定 mode / penalty 的 short-term comm/radar model。
    """
    paths = build_st_paths(mode=mode, penalty=penalty)

    if not os.path.exists(paths["comm_ckpt_path"]):
        raise FileNotFoundError(f"找不到 comm checkpoint：{paths['comm_ckpt_path']}")

    if not os.path.exists(paths["radar_ckpt_path"]):
        raise FileNotFoundError(f"找不到 radar checkpoint：{paths['radar_ckpt_path']}")

    print(f"[LOAD] mode = {mode}")
    print(f"[LOAD] penalty = {penalty}")
    print(f"[LOAD] comm_ckpt_path  = {paths['comm_ckpt_path']}")
    print(f"[LOAD] radar_ckpt_path = {paths['radar_ckpt_path']}")

    comm_net = ShortTermCommNet(ckpt_kind=None).to(DEVICE)
    radar_net = ShortTermRadarNet(ckpt_kind=None).to(DEVICE)

    comm_net.load_model(path=paths["comm_ckpt_path"], verbose=True)
    radar_net.load_model(path=paths["radar_ckpt_path"], verbose=True)

    comm_net.eval()
    radar_net.eval()

    for p in comm_net.parameters():
        p.requires_grad_(False)

    for p in radar_net.parameters():
        p.requires_grad_(False)

    return comm_net, radar_net, paths


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout,
):
    """
    給一組 UE layout，從 long-term net 取得固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        theta_lt, _, _ = longterm_net(ue_layout)

    return theta_lt.detach()


# ================================
# Evaluation core
# ================================
@torch.no_grad()
def eval_one_layout_under_injection(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,               # test estimated channels
    test_injection_variance: float,
    test_injection_samples: int,
    power_norm: bool,
):
    """
    對單一 layout 下的全部 estimated channels 做 test injection evaluation。

    流程:
        1. NN input 使用原始 estimated channels + fixed theta_LT
        2. 用原始 estimated channels 產生 W_C, W_R
        3. W_C, W_R 經過 TX power normalization
        4. 複製 estimated channels 並加入 test uncertainty injection
        5. 可選擇是否對 injected channels 做 power normalization
        6. 在 injected channels 上計算 sum-rate 與 sensing SNR

    回傳:
        sumrate_np : shape = (B * L,)
        snr_db_np  : shape = (B * L,)
    """
    h_dk_hat = batch_data["h_dk_hat"] # shape = (B, M, K)
    h_rk_hat = batch_data["h_rk_hat"] # shape = (B, N, K)
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE     = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B, M, K = h_dk_hat.shape
    N = h_rk_hat.shape[1]
    L = int(test_injection_samples)

    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)

    sumrate_chunks = []
    snr_db_chunks = []

    for s0 in range(0, L, EVAL_INJECTION_CHUNK):
        # 將原本 L 次 test injection 分成 EVAL_INJECTION_CHUNK 計算
        s = min(EVAL_INJECTION_CHUNK, L - s0)

        # 複製 estimated channels 成 s 份
        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        # 注入 test uncertainty
        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape, test_injection_variance, DEVICE, h_dk_rep.dtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape, test_injection_variance, DEVICE, h_rk_rep.dtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,    test_injection_variance, DEVICE, G_rep.dtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape, test_injection_variance, DEVICE, g_dt_rep.dtype)

        # 可選：power normalization
        h_dk_inj, h_rk_inj, G_inj, g_dt_inj = apply_power_normalization_to_injected_channels(
            h_dk_inj=h_dk_inj,
            h_rk_inj=h_rk_inj,
            G_inj=G_inj,
            g_dt_inj=g_dt_inj,
            h_dk_ref=h_dk_rep,
            h_rk_ref=h_rk_rep,
            G_ref=G_rep,
            g_dt_ref=g_dt_rep,
            power_norm=power_norm,
        )

        # 複製 theta
        theta_rep = theta_batch.unsqueeze(1).expand(B, s, N).reshape(B * s, N)

        # 複製 W_C & W_R
        W_C_rep = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)

        radar_streams = W_R.shape[2]
        W_R_rep = W_R.unsqueeze(1).expand(
            B,
            s,
            M,
            radar_streams
        ).reshape(B * s, M, radar_streams)

        sinrs = comm_net.compute_comm_sinrs(
            h_dk=h_dk_inj,
            h_rk=h_rk_inj,
            G=G_inj,
            theta=theta_rep,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE,
        )

        rates = comm_net.compute_rates(sinrs)
        sum_rate = rates.sum(dim=1) # shape = (B*s,)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real # shape = (B*s,)

        sense_snr_db = 10.0 * torch.log10(
            sense_snr.clamp_min(1e-12)
        )

        sumrate_chunks.append(sum_rate.detach().cpu().numpy().astype(np.float32))
        snr_db_chunks.append(sense_snr_db.detach().cpu().numpy().astype(np.float32))

    sumrate_np = np.concatenate(sumrate_chunks, axis=0)
    snr_db_np = np.concatenate(snr_db_chunks, axis=0)

    return sumrate_np, snr_db_np


@torch.no_grad()
def evaluate_one_model_at_injection(
    mode: str,
    penalty: float,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    longterm_net: LongTermPositionNet,
    test_dataset,
    test_injection_variance: float,
    test_injection_samples: int,
    power_norm: bool,
    run_dir: str,
):
    """
    評估單一 mode / penalty model 在指定 test injection variance 下的表現。
    使用全部 test layouts 與每個 layout 全部 estimated channels。
    """
    print("\n" + "=" * 80)
    print(
        f"[EVAL] mode={mode.upper()} penalty={penalty} "
        f"test_inj={test_injection_variance}"
    )
    print("=" * 80)

    reset_eval_seed()

    n_test_layouts = test_dataset["ue_layouts"].shape[0]

    sumrate_all = []
    snr_db_all = []

    for layout_id in range(n_test_layouts):
        ue_layout = test_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

        n_channels = test_dataset["st_h_dk_hat"][layout_id].shape[0] # 該layout下est channel數
        channel_ids = np.arange(n_channels)                          # 建立channel ID

        batch_data = extract_shortterm_batch(                        # 取得該layout所有est channels
            test_dataset,
            layout_id,
            channel_ids
        )

        sumrate_np, snr_db_np = eval_one_layout_under_injection(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            test_injection_variance=test_injection_variance,
            test_injection_samples=test_injection_samples,
            power_norm=power_norm,
        )

        sumrate_all.append(sumrate_np)
        snr_db_all.append(snr_db_np)

    sumrate_all = np.concatenate(sumrate_all, axis=0)
    snr_db_all = np.concatenate(snr_db_all, axis=0)

    mean_sumrate = float(np.mean(sumrate_all))
    q_snr_db = float(np.quantile(snr_db_all, OUTAGE_QUANTILE))
    p_out = float(np.mean(snr_db_all < SENSING_SNR_THRESHOLD_dB))
    reliability = float(1.0 - p_out)

    result = {
        "mode": mode.upper(),
        "penalty": float(penalty),
        "test_injection_variance": float(test_injection_variance),
        "mean_sumrate": mean_sumrate,
        "q_snr_db": q_snr_db,
        "p_out": p_out,
        "reliability": reliability,
        "num_test_layouts": int(n_test_layouts),
        "channels_per_layout": int(test_dataset["st_h_dk_hat"].shape[1]),
        "test_injection_samples": int(test_injection_samples),
        "power_norm": bool(power_norm),
        "run_dir": run_dir,
    }

    print(
        f"[RESULT] {mode.upper()} penalty={penalty} "
        f"test_inj={test_injection_variance} | "
        f"Mean SumRate={mean_sumrate:.6f} | "
        f"Q{OUTAGE_QUANTILE:.2f} SNR={q_snr_db:.3f} dB | "
        f"P_out={100.0 * p_out:.3f}% | "
        f"Reliability={100.0 * reliability:.3f}%"
    )

    return result, snr_db_all, sumrate_all


# ================================
# CDF helpers
# ================================
def make_cdf(values: np.ndarray):
    """
    由 samples 建立 CDF。
    """
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


# ================================
# Save helpers
# ================================
def save_csv(rows: list[dict], csv_path: str):
    """
    儲存 list[dict] 成 CSV。
    """
    if len(rows) == 0:
        raise ValueError("rows is empty, cannot save csv.")

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[SAVE] CSV saved: {csv_path}")


def save_npz_metrics(rows: list[dict], npz_path: str):
    """
    儲存 sweep metrics 成 npz。
    """
    np.savez_compressed(
        npz_path,
        rows=np.array(
            [
                (
                    r["mode"],
                    r["penalty"],
                    r["test_injection_variance"],
                    r["mean_sumrate"],
                    r["q_snr_db"],
                    r["p_out"],
                    r["reliability"],
                    r["num_test_layouts"],
                    r["channels_per_layout"],
                    r["test_injection_samples"],
                    r["power_norm"],
                )
                for r in rows
            ],
            dtype=[
                ("mode", "U8"),
                ("penalty", "f8"),
                ("test_injection_variance", "f8"),
                ("mean_sumrate", "f8"),
                ("q_snr_db", "f8"),
                ("p_out", "f8"),
                ("reliability", "f8"),
                ("num_test_layouts", "i4"),
                ("channels_per_layout", "i4"),
                ("test_injection_samples", "i4"),
                ("power_norm", "?"),
            ]
        )
    )

    print(f"[SAVE] NPZ saved: {npz_path}")


# ================================
# Plot functions
# ================================
def plot_fixed_snr_cdf(
    reg_snr_db: np.ndarray,
    rob_snr_db: np.ndarray,
    reg_penalty: float,
    rob_penalty: float,
    fixed_inj: float,
    fig_path: str,
):
    """
    Fixed injection 下 REG / ROB sensing SNR CDF。
    """
    plt.figure(figsize=(9, 5.5))

    reg_x, reg_y = make_cdf(reg_snr_db)
    rob_x, rob_y = make_cdf(rob_snr_db)

    plt.plot(
        reg_x,
        reg_y,
        linewidth=2.2,
        label=f"REG penalty={reg_penalty}"
    )

    plt.plot(
        rob_x,
        rob_y,
        linewidth=2.2,
        label=f"ROB penalty={rob_penalty}"
    )

    plt.axvline(
        x=SENSING_SNR_THRESHOLD_dB,
        linestyle="--",
        linewidth=2.0,
        label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB"
    )

    plt.axhline(
        y=OUTAGE_QUANTILE,
        linestyle=":",
        linewidth=2.0,
        label=f"{100.0 * OUTAGE_QUANTILE:.1f}% outage level"
    )

    plt.xlabel("Sensing SNR (dB)")
    plt.ylabel("CDF")
    plt.title(f"REG vs ROB Sensing SNR CDF (test injection variance = {fixed_inj})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    plt.close()

    print(f"[SAVE] SNR CDF saved: {fig_path}")


def plot_fixed_rate_cdf(
    reg_sumrate: np.ndarray,
    rob_sumrate: np.ndarray,
    reg_penalty: float,
    rob_penalty: float,
    fixed_inj: float,
    fig_path: str,
):
    """
    Fixed injection 下 REG / ROB sum-rate CDF。
    """
    plt.figure(figsize=(9, 5.5))

    reg_x, reg_y = make_cdf(reg_sumrate)
    rob_x, rob_y = make_cdf(rob_sumrate)

    plt.plot(
        reg_x,
        reg_y,
        linewidth=2.2,
        label=f"REG penalty={reg_penalty}"
    )

    plt.plot(
        rob_x,
        rob_y,
        linewidth=2.2,
        label=f"ROB penalty={rob_penalty}"
    )

    plt.xlabel("Sum-Rate (bits/s/Hz)")
    plt.ylabel("CDF")
    plt.title(f"REG vs ROB Sum-Rate CDF (test injection variance = {fixed_inj})")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    plt.close()

    print(f"[SAVE] Rate CDF saved: {fig_path}")


def plot_injection_sweep_metric(
    sweep_rows: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    fig_path: str,
    horizontal_line=None,
    horizontal_label=None,
    scale_percent: bool = False,
):
    """
    畫 injection variance sweep metric。
    """
    plt.figure(figsize=(9, 5.5))

    for mode in ["REG", "ROB"]:
        rows = [
            r for r in sweep_rows
            if r["mode"] == mode
        ]

        rows = sorted(
            rows,
            key=lambda r: r["test_injection_variance"]
        )

        x = np.array(
            [r["test_injection_variance"] for r in rows],
            dtype=np.float64
        )

        y = np.array(
            [r[metric_key] for r in rows],
            dtype=np.float64
        )

        if scale_percent:
            y = 100.0 * y

        penalty = rows[0]["penalty"] if len(rows) > 0 else None

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=f"{mode} penalty={penalty}"
        )

    if horizontal_line is not None:
        h_value = horizontal_line

        if scale_percent:
            h_value = 100.0 * h_value

        plt.axhline(
            y=h_value,
            linestyle="--",
            linewidth=2.0,
            label=horizontal_label
        )

    plt.xlabel("Test Injection Variance")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    plt.close()

    print(f"[SAVE] Injection sweep figure saved: {fig_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate REG/ROB under injection sweep")

    parser.add_argument(
        "--norm",
        action="store_true",
        help="對 injected channels 做 power normalization"
    )

    args = parser.parse_args()

    power_norm = bool(args.norm)
    norm_tag = "powernorm" if power_norm else "raw"

    reg_penalty = float(REG_SENSING_LOSS_WEIGHT)
    rob_penalty = float(ROB_SENSING_LOSS_WEIGHT)

    test_injection_samples = int(INJECTION_SAMPLES)
    fixed_inj = float(INJECTION_VARIANCE)

    injection_sweep_list = list(INJECTION_SWEEP_LIST)

    if fixed_inj not in injection_sweep_list:
        injection_sweep_list = sorted(injection_sweep_list + [fixed_inj])

    test_dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")
    longterm_ckpt_path = os.path.join(LT_CKPT_DIR, "longterm.ckpt")

    eval_dir = build_eval_dir(power_norm=power_norm)

    print("\n" + "=" * 80)
    print("[INFO] evaluate.py started.")
    print("=" * 80)
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] power_norm = {power_norm}")
    print(f"[INFO] reg_penalty = {reg_penalty}")
    print(f"[INFO] rob_penalty = {rob_penalty}")
    print(f"[INFO] test_injection_samples = {test_injection_samples}")
    print(f"[INFO] EVAL_INJECTION_CHUNK = {EVAL_INJECTION_CHUNK}")
    print(f"[INFO] fixed_inj = {fixed_inj}")
    print(f"[INFO] OUTAGE_QUANTILE = {OUTAGE_QUANTILE}")
    print(f"[INFO] injection_sweep_list = {injection_sweep_list}")
    print(f"[INFO] test_dataset_path = {test_dataset_path}")
    print(f"[INFO] longterm_ckpt_path = {longterm_ckpt_path}")
    print(f"[INFO] eval_dir = {eval_dir}")

    # ================================
    # Load dataset / models
    # ================================
    test_dataset = load_test_dataset(test_dataset_path)

    print("\n[INFO] 載入 shared long-term model ...")
    longterm_net = load_longterm_model(longterm_ckpt_path)

    print("\n[INFO] 載入 short-term REG model ...")
    reg_comm_net, reg_radar_net, reg_paths = load_shortterm_model(
        mode="reg",
        penalty=reg_penalty
    )

    print("\n[INFO] 載入 short-term ROB model ...")
    rob_comm_net, rob_radar_net, rob_paths = load_shortterm_model(
        mode="rob",
        penalty=rob_penalty
    )

    # ================================
    # Injection sweep
    # ================================
    sweep_rows = []

    fixed_reg_snr = None
    fixed_reg_rate = None
    fixed_rob_snr = None
    fixed_rob_rate = None

    for test_inj in injection_sweep_list:
        reg_result, reg_snr_db, reg_sumrate = evaluate_one_model_at_injection(
            mode="reg",
            penalty=reg_penalty,
            comm_net=reg_comm_net,
            radar_net=reg_radar_net,
            longterm_net=longterm_net,
            test_dataset=test_dataset,
            test_injection_variance=float(test_inj),
            test_injection_samples=test_injection_samples,
            power_norm=power_norm,
            run_dir=reg_paths["run_dir"],
        )

        rob_result, rob_snr_db, rob_sumrate = evaluate_one_model_at_injection(
            mode="rob",
            penalty=rob_penalty,
            comm_net=rob_comm_net,
            radar_net=rob_radar_net,
            longterm_net=longterm_net,
            test_dataset=test_dataset,
            test_injection_variance=float(test_inj),
            test_injection_samples=test_injection_samples,
            power_norm=power_norm,
            run_dir=rob_paths["run_dir"],
        )

        sweep_rows.append(reg_result)
        sweep_rows.append(rob_result)

        if abs(float(test_inj) - fixed_inj) < 1e-12:
            fixed_reg_snr = reg_snr_db
            fixed_reg_rate = reg_sumrate
            fixed_rob_snr = rob_snr_db
            fixed_rob_rate = rob_sumrate

    # ================================
    # Save sweep results
    # ================================
    sweep_csv_path = os.path.join(
        eval_dir,
        f"injection_sweep_setting_penalty_{norm_tag}.csv"
    )

    sweep_npz_path = os.path.join(
        eval_dir,
        f"injection_sweep_setting_penalty_{norm_tag}.npz"
    )

    save_csv(
        rows=sweep_rows,
        csv_path=sweep_csv_path
    )

    save_npz_metrics(
        rows=sweep_rows,
        npz_path=sweep_npz_path
    )

    # ================================
    # Fixed injection CDF
    # ================================
    fixed_tag = format_float_for_path(fixed_inj)

    if fixed_reg_snr is not None:
        fixed_samples_npz_path = os.path.join(
            eval_dir,
            f"fixed_inj_{fixed_tag}_samples_{norm_tag}.npz"
        )

        np.savez_compressed(
            fixed_samples_npz_path,
            reg_snr_db=fixed_reg_snr.astype(np.float32),
            reg_sumrate=fixed_reg_rate.astype(np.float32),
            rob_snr_db=fixed_rob_snr.astype(np.float32),
            rob_sumrate=fixed_rob_rate.astype(np.float32),
            fixed_injection_variance=np.array(fixed_inj, dtype=np.float32),
            reg_penalty=np.array(reg_penalty, dtype=np.float32),
            rob_penalty=np.array(rob_penalty, dtype=np.float32),
            power_norm=np.array(power_norm),
        )

        print(f"[SAVE] Fixed injection samples saved: {fixed_samples_npz_path}")

        snr_cdf_path = os.path.join(
            eval_dir,
            f"fixed_inj_{fixed_tag}_snr_cdf_{norm_tag}.jpg"
        )

        rate_cdf_path = os.path.join(
            eval_dir,
            f"fixed_inj_{fixed_tag}_rate_cdf_{norm_tag}.jpg"
        )

        plot_fixed_snr_cdf(
            reg_snr_db=fixed_reg_snr,
            rob_snr_db=fixed_rob_snr,
            reg_penalty=reg_penalty,
            rob_penalty=rob_penalty,
            fixed_inj=fixed_inj,
            fig_path=snr_cdf_path,
        )

        plot_fixed_rate_cdf(
            reg_sumrate=fixed_reg_rate,
            rob_sumrate=fixed_rob_rate,
            reg_penalty=reg_penalty,
            rob_penalty=rob_penalty,
            fixed_inj=fixed_inj,
            fig_path=rate_cdf_path,
        )

    # ================================
    # Sweep figures
    # ================================
    pout_fig_path = os.path.join(
        eval_dir,
        f"injection_sweep_pout_{norm_tag}.jpg"
    )

    qsnr_fig_path = os.path.join(
        eval_dir,
        f"injection_sweep_qsnr_{norm_tag}.jpg"
    )

    rate_fig_path = os.path.join(
        eval_dir,
        f"injection_sweep_mean_sumrate_{norm_tag}.jpg"
    )

    plot_injection_sweep_metric(
        sweep_rows=sweep_rows,
        metric_key="p_out",
        ylabel="Sensing Outage Probability (%)",
        title="Injection Variance vs Sensing Outage Probability",
        fig_path=pout_fig_path,
        horizontal_line=OUTAGE_QUANTILE,
        horizontal_label=f"{100.0 * OUTAGE_QUANTILE:.1f}% outage constraint",
        scale_percent=True,
    )

    plot_injection_sweep_metric(
        sweep_rows=sweep_rows,
        metric_key="q_snr_db",
        ylabel=f"Q{OUTAGE_QUANTILE:.2f} Sensing SNR (dB)",
        title=f"Injection Variance vs Q{OUTAGE_QUANTILE:.2f} Sensing SNR",
        fig_path=qsnr_fig_path,
        horizontal_line=SENSING_SNR_THRESHOLD_dB,
        horizontal_label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB",
        scale_percent=False,
    )

    plot_injection_sweep_metric(
        sweep_rows=sweep_rows,
        metric_key="mean_sumrate",
        ylabel="Mean Sum-Rate (bits/s/Hz)",
        title="Injection Variance vs Mean Sum-Rate",
        fig_path=rate_fig_path,
        horizontal_line=None,
        horizontal_label=None,
        scale_percent=False,
    )

    print("\n" + "=" * 80)
    print("[INFO] evaluate.py finished.")
    print("=" * 80)
    print(f"[INFO] Sweep CSV      : {sweep_csv_path}")
    print(f"[INFO] Sweep NPZ      : {sweep_npz_path}")
    print(f"[INFO] Pout figure    : {pout_fig_path}")
    print(f"[INFO] Q-SNR figure   : {qsnr_fig_path}")
    print(f"[INFO] Rate figure    : {rate_fig_path}")