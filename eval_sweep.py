# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ================================
# Eval sweep settings
# ================================
EVAL_INJECTION_CHUNK = 50

REG_PENALTY_LIST = [
    50,
    70,
    100,
    150,
    175,
    200,
    300,
]

ROB_PENALTY_LIST = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.50,
    1.00,
]


# ================================
# Path helpers
# ================================
def format_float_for_path(value: float) -> str:
    """
    將 float 轉成與 main_st.py 相同的資料夾名稱格式。

    例：
        0.10  -> 0p1
        0.15  -> 0p15
        0.5   -> 0p5
        1.0   -> 1
        100.0 -> 100
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")

    return text


def build_st_paths(mode: str, penalty: float) -> dict:
    """
    根據 mode 與 penalty 建立 ST checkpoint 路徑。
    """
    mode = mode.lower()
    penalty_tag = format_float_for_path(penalty)

    if mode == "reg":
        run_name = f"REG_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm.ckpt"
        radar_ckpt_name = "short_radar.ckpt"
    elif mode == "rob":
        run_name = f"ROB_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm_robust.ckpt"
        radar_ckpt_name = "short_radar_robust.ckpt"
    else:
        raise ValueError(f"mode must be 'reg' or 'rob', got {mode}")

    run_dir = os.path.join(ST_SWEEP_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, comm_ckpt_name),
        "radar_ckpt_path": os.path.join(ckpt_dir, radar_ckpt_name),
    }


def build_eval_dir() -> str:
    """
    建立 eval_sweep.py 輸出資料夾。
    """
    inj_tag = format_float_for_path(INJECTION_VARIANCE)

    eval_dir = os.path.join(
        BASE_RUN_DIR,
        "eval_results",
        f"eval_sweep_testinj_{inj_tag}_powernorm"
    )

    os.makedirs(eval_dir, exist_ok=True)

    return eval_dir


# ================================
# Tensor helpers
# ================================
def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    if float(variance) == 0.0:
        return torch.zeros(shape, device=device, dtype=cdtype)

    sigma = math.sqrt(float(variance) / 2.0)

    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma

    return torch.complex(nr, ni).to(dtype=cdtype)


def normalize_channel_power(
    injected: torch.Tensor,
    reference: torch.Tensor,
    norm_dims,
):
    """
    將 injected channel 的 power normalize 回 reference channel。

    injected_norm = injected * ||reference|| / ||injected||
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


def reset_eval_seed():
    """
    每次 REG / ROB model 評估前，將亂數狀態重設回 settings.py 的 RANDOM_SEED。

    目的:
        REG / ROB 在同一個 test injection variance 下使用相同 injection noise。
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ================================
# Dataset
# ================================
def load_test_dataset(npz_path: str):
    """
    載入 test dataset。
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

    print(f"[test] loaded: {npz_path}")
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
    layout_id: int,
    channel_ids: np.ndarray,
):
    """
    取出指定 layout 與 channel ids 的 ST estimated channels
    並轉為 torch tensor 準備輸入 NN。
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
        raise FileNotFoundError(f"找不到 long-term checkpoint: {longterm_ckpt_path}")

    print(f"[LOAD] longterm_ckpt_path = {longterm_ckpt_path}")

    longterm_net = LongTermPositionNet(ckpt_kind=None).to(DEVICE)

    longterm_net.load_model(
        path=longterm_ckpt_path,
        verbose=True
    )

    longterm_net.eval()

    for p in longterm_net.parameters():
        p.requires_grad_(False)

    return longterm_net


def load_shortterm_model(mode: str, penalty: float):
    """
    載入指定 mode / penalty 的 ST comm/radar model。
    """
    paths = build_st_paths(
        mode=mode,
        penalty=penalty
    )

    if not os.path.exists(paths["comm_ckpt_path"]):
        raise FileNotFoundError(f"找不到 comm checkpoint: {paths['comm_ckpt_path']}")

    if not os.path.exists(paths["radar_ckpt_path"]):
        raise FileNotFoundError(f"找不到 radar checkpoint: {paths['radar_ckpt_path']}")

    print(f"[LOAD] mode = {mode.upper()}")
    print(f"[LOAD] penalty = {penalty}")
    print(f"[LOAD] comm_ckpt_path  = {paths['comm_ckpt_path']}")
    print(f"[LOAD] radar_ckpt_path = {paths['radar_ckpt_path']}")

    comm_net = ShortTermCommNet(ckpt_kind=None).to(DEVICE)
    radar_net = ShortTermRadarNet(ckpt_kind=None).to(DEVICE)

    comm_net.load_model(
        path=paths["comm_ckpt_path"],
        verbose=True
    )

    radar_net.load_model(
        path=paths["radar_ckpt_path"],
        verbose=True
    )

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

    ue_layout_t = torch.as_tensor(
        ue_layout,
        dtype=torch.float32,
        device=DEVICE,
    )

    if ue_layout_t.dim() == 2:
        ue_layout_t = ue_layout_t.unsqueeze(0)

    with torch.no_grad():
        theta_lt, _, _ = longterm_net(ue_layout_t)

    if theta_lt.dim() == 1:
        theta_lt = theta_lt.unsqueeze(0)

    return theta_lt.detach()


# ================================
# Evaluation core
# ================================
@torch.no_grad()
def eval_one_layout_under_injection(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
):
    """
    對單一 layout 下全部 estimated channels 做 fixed test injection evaluation。

    流程:
        1. NN input 使用原始 estimated channels + fixed theta_LT
        2. estimated channels -> W_C, W_R
        3. W_C, W_R 做 TX power normalization
        4. estimated channels 複製成 INJECTION_SAMPLES 份
        5. 加入 test uncertainty injection
        6. injected channels normalize 回 estimated channel power
        7. 在 normalized injected channels 上計算 sum-rate 與 sensing SNR

    回傳:
        sumrate_np : shape = (B * INJECTION_SAMPLES,)
        snr_db_np  : shape = (B * INJECTION_SAMPLES,)
    """
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE     = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B, M, K = h_dk_hat.shape
    N = h_rk_hat.shape[1]
    L = int(INJECTION_SAMPLES)

    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    W_C = comm_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )

    W_R = radar_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)

    sumrate_chunks = []
    snr_db_chunks = []

    for s0 in range(0, L, EVAL_INJECTION_CHUNK):
        s = min(EVAL_INJECTION_CHUNK, L - s0)

        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        h_dk_inj = h_dk_rep + complex_awgn(
            h_dk_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            h_dk_rep.dtype
        )

        h_rk_inj = h_rk_rep + complex_awgn(
            h_rk_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            h_rk_rep.dtype
        )

        G_inj = G_rep + complex_awgn(
            G_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            G_rep.dtype
        )

        g_dt_inj = g_dt_rep + complex_awgn(
            g_dt_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            g_dt_rep.dtype
        )

        # normalize injected channel power
        h_dk_inj = normalize_channel_power(h_dk_inj, h_dk_rep, norm_dims=1)
        h_rk_inj = normalize_channel_power(h_rk_inj, h_rk_rep, norm_dims=1)
        G_inj    = normalize_channel_power(G_inj,    G_rep,    norm_dims=(1, 2))
        g_dt_inj = normalize_channel_power(g_dt_inj, g_dt_rep, norm_dims=(1, 2))

        theta_rep = theta_batch.unsqueeze(1).expand(B, s, N).reshape(B * s, N)

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
        sum_rate = rates.sum(dim=1)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real

        sense_snr_db = 10.0 * torch.log10(
            sense_snr.clamp_min(1e-12)
        )

        sumrate_chunks.append(
            sum_rate.detach().cpu().numpy().astype(np.float32)
        )

        snr_db_chunks.append(
            sense_snr_db.detach().cpu().numpy().astype(np.float32)
        )

    sumrate_np = np.concatenate(sumrate_chunks, axis=0)
    snr_db_np = np.concatenate(snr_db_chunks, axis=0)

    return sumrate_np, snr_db_np


@torch.no_grad()
def evaluate_one_model(
    mode: str,
    penalty: float,
    longterm_net: LongTermPositionNet,
    test_dataset,
):
    """
    評估單一 mode / penalty model。
    使用全部 test layouts，且每個 layout 使用全部 ST estimated channels。
    """
    print("\n" + "=" * 80)
    print(f"[EVAL-SWEEP] mode={mode.upper()} penalty={penalty}")
    print("=" * 80)

    reset_eval_seed()

    comm_net, radar_net, paths = load_shortterm_model(
        mode=mode,
        penalty=penalty
    )

    n_test_layouts = test_dataset["ue_layouts"].shape[0]

    sumrate_all = []
    snr_db_all = []

    for layout_id in range(n_test_layouts):
        ue_layout = test_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

        n_channels = test_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = np.arange(n_channels)

        batch_data = extract_shortterm_batch(
            dataset=test_dataset,
            layout_id=layout_id,
            channel_ids=channel_ids,
        )

        sumrate_np, snr_db_np = eval_one_layout_under_injection(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
        )

        sumrate_all.append(sumrate_np)
        snr_db_all.append(snr_db_np)

    sumrate_all = np.concatenate(sumrate_all, axis=0)
    snr_db_all = np.concatenate(snr_db_all, axis=0)

    mean_sumrate = float(np.mean(sumrate_all))
    q05_snr_db = float(np.quantile(snr_db_all, OUTAGE_QUANTILE))
    p_out = float(np.mean(snr_db_all < SENSING_SNR_THRESHOLD_dB))
    reliability = float(1.0 - p_out)
    feasible = bool(p_out <= OUTAGE_QUANTILE)

    result = {
        "mode": mode.upper(),
        "penalty": float(penalty),
        "mean_sumrate": mean_sumrate,
        "q05_snr_db": q05_snr_db,
        "p_out": p_out,
        "reliability": reliability,
        "feasible": feasible,
        "num_test_layouts": int(n_test_layouts),
        "channels_per_layout": int(test_dataset["st_h_dk_hat"].shape[1]),
        "test_injection_variance": float(INJECTION_VARIANCE),
        "test_injection_samples": int(INJECTION_SAMPLES),
        "power_normalized_injection": True,
        "run_dir": paths["run_dir"],
        "comm_ckpt_path": paths["comm_ckpt_path"],
        "radar_ckpt_path": paths["radar_ckpt_path"],
    }

    print(
        f"[RESULT] {mode.upper()} penalty={penalty} | "
        f"Mean SumRate={mean_sumrate:.6f} | "
        f"Q{OUTAGE_QUANTILE:.2f} SNR={q05_snr_db:.3f} dB | "
        f"P_out={100.0 * p_out:.3f}% | "
        f"Reliability={100.0 * reliability:.3f}% | "
        f"Feasible={feasible}"
    )

    return result, snr_db_all, sumrate_all


# ================================
# Selection
# ================================
def select_best_feasible_or_fallback(results: list[dict], mode: str):
    """
    在指定 mode 中：
        1. 若有 feasible model，選 mean_sumrate 最大者。
        2. 若沒有 feasible model，選 p_out 最小者；若 p_out 相同，選 mean_sumrate 最大者。
    """
    mode = mode.upper()

    mode_rows = [
        r for r in results
        if r["mode"] == mode
    ]

    if len(mode_rows) == 0:
        return None

    feasible_rows = [
        r for r in mode_rows
        if r["feasible"]
    ]

    if len(feasible_rows) > 0:
        best = max(
            feasible_rows,
            key=lambda r: r["mean_sumrate"]
        )
        best = dict(best)
        best["selection_type"] = "best_feasible"
        return best

    best = min(
        mode_rows,
        key=lambda r: (r["p_out"], -r["mean_sumrate"])
    )
    best = dict(best)
    best["selection_type"] = "fallback_min_pout"

    return best


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


def save_best_summary(
    best_reg,
    best_rob,
    summary_path: str,
):
    """
    儲存 best feasible REG / ROB summary。
    """
    lines = []

    lines.append("eval_sweep.py best feasible model summary\n")
    lines.append("=" * 80 + "\n\n")

    lines.append("Evaluation setting:\n")
    lines.append(f"  test_injection_variance     = {INJECTION_VARIANCE}\n")
    lines.append(f"  test_injection_samples      = {INJECTION_SAMPLES}\n")
    lines.append(f"  power_normalized_injection  = True\n")
    lines.append(f"  sensing_snr_threshold       = {SENSING_SNR_THRESHOLD_dB} dB\n")
    lines.append(f"  outage_constraint           = {100.0 * OUTAGE_QUANTILE:.3f}%\n")
    lines.append("\n")

    lines.append("Selection rule:\n")
    lines.append("  Among models with P_out <= OUTAGE_QUANTILE, choose the highest mean sum-rate.\n")
    lines.append("  If no feasible model exists, fallback to the minimum P_out model.\n\n")

    def add_best(name, best):
        if best is None:
            lines.append(f"[{name}] No model found.\n\n")
            return

        lines.append(f"[{name}]\n")
        lines.append(f"  selection_type = {best['selection_type']}\n")
        lines.append(f"  penalty        = {best['penalty']}\n")
        lines.append(f"  mean_sumrate   = {best['mean_sumrate']:.6f}\n")
        lines.append(f"  q05_snr_db     = {best['q05_snr_db']:.3f} dB\n")
        lines.append(f"  p_out          = {100.0 * best['p_out']:.3f}%\n")
        lines.append(f"  reliability    = {100.0 * best['reliability']:.3f}%\n")
        lines.append(f"  feasible       = {best['feasible']}\n")
        lines.append(f"  run_dir        = {best['run_dir']}\n\n")

    add_best("Best REG", best_reg)
    add_best("Best ROB", best_rob)

    if best_reg is not None and best_rob is not None:
        lines.append("Performance comparison: ROB - REG\n")
        lines.append(f"  delta_mean_sumrate = {best_rob['mean_sumrate'] - best_reg['mean_sumrate']:.6f}\n")
        lines.append(f"  delta_q05_snr_db   = {best_rob['q05_snr_db'] - best_reg['q05_snr_db']:.3f} dB\n")
        lines.append(f"  delta_p_out        = {100.0 * (best_rob['p_out'] - best_reg['p_out']):.3f}%\n")
        lines.append(f"  delta_reliability  = {100.0 * (best_rob['reliability'] - best_reg['reliability']):.3f}%\n")

    summary_text = "".join(lines)

    print("\n" + summary_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[SAVE] Summary saved: {summary_path}")


def save_npz_results(
    results: list[dict],
    snr_cdf_data: dict,
    sumrate_data: dict,
    npz_path: str,
):
    """
    儲存 metrics 與所有 SNR / sum-rate samples。
    """
    metrics_dtype = [
        ("mode", "U8"),
        ("penalty", "f8"),
        ("mean_sumrate", "f8"),
        ("q05_snr_db", "f8"),
        ("p_out", "f8"),
        ("reliability", "f8"),
        ("feasible", "i4"),
    ]

    metrics_array = np.zeros(
        len(results),
        dtype=metrics_dtype
    )

    for i, r in enumerate(results):
        metrics_array[i]["mode"] = r["mode"]
        metrics_array[i]["penalty"] = r["penalty"]
        metrics_array[i]["mean_sumrate"] = r["mean_sumrate"]
        metrics_array[i]["q05_snr_db"] = r["q05_snr_db"]
        metrics_array[i]["p_out"] = r["p_out"]
        metrics_array[i]["reliability"] = r["reliability"]
        metrics_array[i]["feasible"] = int(r["feasible"])

    save_dict = {
        "metrics": metrics_array,
        "test_injection_variance": np.array(INJECTION_VARIANCE, dtype=np.float32),
        "test_injection_samples": np.array(INJECTION_SAMPLES, dtype=np.int32),
        "power_normalized_injection": np.array(True, dtype=np.bool_),
    }

    for key, value in snr_cdf_data.items():
        save_dict[f"snr_db_{key}"] = value.astype(np.float32)

    for key, value in sumrate_data.items():
        save_dict[f"sumrate_{key}"] = value.astype(np.float32)

    np.savez_compressed(
        npz_path,
        **save_dict
    )

    print(f"[SAVE] NPZ saved: {npz_path}")


# ================================
# Plot helpers
# ================================
def make_cdf(values: np.ndarray):
    """
    由 samples 建立 CDF。
    """
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)

    return x, y


def plot_cdf_group(
    mode: str,
    penalties: list[float],
    data_dict: dict,
    data_prefix: str,
    xlabel: str,
    ylabel: str,
    title: str,
    fig_path: str,
    add_snr_lines: bool = False,
):
    """
    畫同一 mode 下所有 penalty 的 CDF。
    """
    mode = mode.upper()

    plt.figure(figsize=(9, 5.5))

    for penalty in penalties:
        key = f"{mode}_penalty_{format_float_for_path(penalty)}"
        values = data_dict[key]
        x, y = make_cdf(values)

        plt.plot(
            x,
            y,
            linewidth=2.0,
            label=f"{mode} penalty={penalty}"
        )

    if add_snr_lines:
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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] {data_prefix} CDF saved: {fig_path}")


def plot_best_snr_cdf(
    best_reg,
    best_rob,
    snr_cdf_data: dict,
    fig_path: str,
):
    """
    Best REG / ROB SNR CDF。
    """
    reg_key = f"REG_penalty_{format_float_for_path(best_reg['penalty'])}"
    rob_key = f"ROB_penalty_{format_float_for_path(best_rob['penalty'])}"

    reg_x, reg_y = make_cdf(snr_cdf_data[reg_key])
    rob_x, rob_y = make_cdf(snr_cdf_data[rob_key])

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        reg_x,
        reg_y,
        linewidth=2.2,
        label=f"Best REG penalty={best_reg['penalty']}"
    )

    plt.plot(
        rob_x,
        rob_y,
        linewidth=2.2,
        label=f"Best ROB penalty={best_rob['penalty']}"
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
    plt.title(
        f"Best REG vs ROB Sensing SNR CDF "
        f"(test injection variance = {INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Best SNR CDF saved: {fig_path}")


def plot_best_rate_cdf(
    best_reg,
    best_rob,
    sumrate_data: dict,
    fig_path: str,
):
    """
    Best REG / ROB Rate CDF。
    """
    reg_key = f"REG_penalty_{format_float_for_path(best_reg['penalty'])}"
    rob_key = f"ROB_penalty_{format_float_for_path(best_rob['penalty'])}"

    reg_x, reg_y = make_cdf(sumrate_data[reg_key])
    rob_x, rob_y = make_cdf(sumrate_data[rob_key])

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        reg_x,
        reg_y,
        linewidth=2.2,
        label=f"Best REG penalty={best_reg['penalty']}"
    )

    plt.plot(
        rob_x,
        rob_y,
        linewidth=2.2,
        label=f"Best ROB penalty={best_rob['penalty']}"
    )

    plt.xlabel("Sum-Rate (bits/s/Hz)")
    plt.ylabel("CDF")
    plt.title(
        f"Best REG vs ROB Sum-Rate CDF "
        f"(test injection variance = {INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Best Rate CDF saved: {fig_path}")


def plot_tradeoff_rate_reliability(
    results: list[dict],
    fig_path: str,
):
    """
    Rate-Reliability trade-off。
    """
    plt.figure(figsize=(9, 5.5))

    for mode in ["REG", "ROB"]:
        rows = [
            r for r in results
            if r["mode"] == mode
        ]

        rows = sorted(
            rows,
            key=lambda r: r["mean_sumrate"]
        )

        x = np.array([r["mean_sumrate"] for r in rows], dtype=np.float64)
        y = np.array([100.0 * r["reliability"] for r in rows], dtype=np.float64)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=mode
        )

        for r in rows:
            plt.annotate(
                str(r["penalty"]),
                (
                    r["mean_sumrate"],
                    100.0 * r["reliability"]
                ),
                fontsize=8
            )

    plt.axhline(
        y=100.0 * (1.0 - OUTAGE_QUANTILE),
        linestyle="--",
        linewidth=2.0,
        label=f"{100.0 * (1.0 - OUTAGE_QUANTILE):.1f}% reliability"
    )

    plt.xlabel("Mean Sum-Rate (bits/s/Hz)")
    plt.ylabel("Sensing Reliability (%)")
    plt.title(
        f"Rate-Reliability Trade-off "
        f"(test injection variance = {INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Rate-Reliability trade-off saved: {fig_path}")


def plot_tradeoff_rate_q05snr(
    results: list[dict],
    fig_path: str,
):
    """
    Rate-Q0.05 SNR trade-off。
    """
    plt.figure(figsize=(9, 5.5))

    for mode in ["REG", "ROB"]:
        rows = [
            r for r in results
            if r["mode"] == mode
        ]

        rows = sorted(
            rows,
            key=lambda r: r["mean_sumrate"]
        )

        x = np.array([r["mean_sumrate"] for r in rows], dtype=np.float64)
        y = np.array([r["q05_snr_db"] for r in rows], dtype=np.float64)

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=mode
        )

        for r in rows:
            plt.annotate(
                str(r["penalty"]),
                (
                    r["mean_sumrate"],
                    r["q05_snr_db"]
                ),
                fontsize=8
            )

    plt.axhline(
        y=SENSING_SNR_THRESHOLD_dB,
        linestyle="--",
        linewidth=2.0,
        label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB"
    )

    plt.xlabel("Mean Sum-Rate (bits/s/Hz)")
    plt.ylabel(f"Q{OUTAGE_QUANTILE:.2f} Sensing SNR (dB)")
    plt.title(
        f"Rate-Q{OUTAGE_QUANTILE:.2f} Sensing SNR Trade-off "
        f"(test injection variance = {INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Rate-Q0.05 SNR trade-off saved: {fig_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    test_dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")
    longterm_ckpt_path = os.path.join(LT_CKPT_DIR, "longterm.ckpt")

    eval_dir = build_eval_dir()
    inj_tag = format_float_for_path(INJECTION_VARIANCE)

    print("\n" + "=" * 80)
    print("[INFO] eval_sweep.py started.")
    print("=" * 80)
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] BASE_RUN_DIR = {BASE_RUN_DIR}")
    print(f"[INFO] ST_SWEEP_DIR = {ST_SWEEP_DIR}")
    print(f"[INFO] test_dataset_path = {test_dataset_path}")
    print(f"[INFO] longterm_ckpt_path = {longterm_ckpt_path}")
    print(f"[INFO] eval_dir = {eval_dir}")

    print("\n[INFO] Evaluation setting:")
    print(f"[INFO] INJECTION_VARIANCE = {INJECTION_VARIANCE}")
    print(f"[INFO] INJECTION_SAMPLES = {INJECTION_SAMPLES}")
    print(f"[INFO] EVAL_INJECTION_CHUNK = {EVAL_INJECTION_CHUNK}")
    print(f"[INFO] power_normalized_injection = True")
    print(f"[INFO] SENSING_SNR_THRESHOLD_dB = {SENSING_SNR_THRESHOLD_dB}")
    print(f"[INFO] OUTAGE_QUANTILE = {OUTAGE_QUANTILE}")
    print(f"[INFO] REG_PENALTY_LIST = {REG_PENALTY_LIST}")
    print(f"[INFO] ROB_PENALTY_LIST = {ROB_PENALTY_LIST}")

    test_dataset = load_test_dataset(test_dataset_path)
    longterm_net = load_longterm_model(longterm_ckpt_path)

    results = []
    snr_cdf_data = {}
    sumrate_data = {}

    # ================================
    # Evaluate REG models
    # ================================
    for penalty in REG_PENALTY_LIST:
        result, snr_db_all, sumrate_all = evaluate_one_model(
            mode="reg",
            penalty=float(penalty),
            longterm_net=longterm_net,
            test_dataset=test_dataset,
        )

        results.append(result)

        key = f"REG_penalty_{format_float_for_path(penalty)}"
        snr_cdf_data[key] = snr_db_all
        sumrate_data[key] = sumrate_all

    # ================================
    # Evaluate ROB models
    # ================================
    for penalty in ROB_PENALTY_LIST:
        result, snr_db_all, sumrate_all = evaluate_one_model(
            mode="rob",
            penalty=float(penalty),
            longterm_net=longterm_net,
            test_dataset=test_dataset,
        )

        results.append(result)

        key = f"ROB_penalty_{format_float_for_path(penalty)}"
        snr_cdf_data[key] = snr_db_all
        sumrate_data[key] = sumrate_all

    # ================================
    # Selection
    # ================================
    best_reg = select_best_feasible_or_fallback(
        results=results,
        mode="REG"
    )

    best_rob = select_best_feasible_or_fallback(
        results=results,
        mode="ROB"
    )

    # ================================
    # Save tables / summaries
    # ================================
    selection_csv_path = os.path.join(
        eval_dir,
        f"selection_table_testinj_{inj_tag}.csv"
    )

    selection_npz_path = os.path.join(
        eval_dir,
        f"selection_metrics_testinj_{inj_tag}.npz"
    )

    summary_path = os.path.join(
        eval_dir,
        f"best_summary_testinj_{inj_tag}.txt"
    )

    save_csv(
        rows=results,
        csv_path=selection_csv_path
    )

    save_best_summary(
        best_reg=best_reg,
        best_rob=best_rob,
        summary_path=summary_path
    )

    save_npz_results(
        results=results,
        snr_cdf_data=snr_cdf_data,
        sumrate_data=sumrate_data,
        npz_path=selection_npz_path,
    )

    # ================================
    # Plot all penalty CDFs
    # ================================
    reg_snr_cdf_path = os.path.join(
        eval_dir,
        f"snr_cdf_all_REG_testinj_{inj_tag}.jpg"
    )

    rob_snr_cdf_path = os.path.join(
        eval_dir,
        f"snr_cdf_all_ROB_testinj_{inj_tag}.jpg"
    )

    reg_rate_cdf_path = os.path.join(
        eval_dir,
        f"rate_cdf_all_REG_testinj_{inj_tag}.jpg"
    )

    rob_rate_cdf_path = os.path.join(
        eval_dir,
        f"rate_cdf_all_ROB_testinj_{inj_tag}.jpg"
    )

    plot_cdf_group(
        mode="REG",
        penalties=REG_PENALTY_LIST,
        data_dict=snr_cdf_data,
        data_prefix="REG SNR",
        xlabel="Sensing SNR (dB)",
        ylabel="CDF",
        title=f"REG Sensing SNR CDF (test injection variance = {INJECTION_VARIANCE})",
        fig_path=reg_snr_cdf_path,
        add_snr_lines=True,
    )

    plot_cdf_group(
        mode="ROB",
        penalties=ROB_PENALTY_LIST,
        data_dict=snr_cdf_data,
        data_prefix="ROB SNR",
        xlabel="Sensing SNR (dB)",
        ylabel="CDF",
        title=f"ROB Sensing SNR CDF (test injection variance = {INJECTION_VARIANCE})",
        fig_path=rob_snr_cdf_path,
        add_snr_lines=True,
    )

    plot_cdf_group(
        mode="REG",
        penalties=REG_PENALTY_LIST,
        data_dict=sumrate_data,
        data_prefix="REG Rate",
        xlabel="Sum-Rate (bits/s/Hz)",
        ylabel="CDF",
        title=f"REG Sum-Rate CDF (test injection variance = {INJECTION_VARIANCE})",
        fig_path=reg_rate_cdf_path,
        add_snr_lines=False,
    )

    plot_cdf_group(
        mode="ROB",
        penalties=ROB_PENALTY_LIST,
        data_dict=sumrate_data,
        data_prefix="ROB Rate",
        xlabel="Sum-Rate (bits/s/Hz)",
        ylabel="CDF",
        title=f"ROB Sum-Rate CDF (test injection variance = {INJECTION_VARIANCE})",
        fig_path=rob_rate_cdf_path,
        add_snr_lines=False,
    )

    # ================================
    # Plot best REG / ROB CDFs
    # ================================
    best_snr_cdf_path = os.path.join(
        eval_dir,
        f"best_snr_cdf_testinj_{inj_tag}.jpg"
    )

    best_rate_cdf_path = os.path.join(
        eval_dir,
        f"best_rate_cdf_testinj_{inj_tag}.jpg"
    )

    plot_best_snr_cdf(
        best_reg=best_reg,
        best_rob=best_rob,
        snr_cdf_data=snr_cdf_data,
        fig_path=best_snr_cdf_path,
    )

    plot_best_rate_cdf(
        best_reg=best_reg,
        best_rob=best_rob,
        sumrate_data=sumrate_data,
        fig_path=best_rate_cdf_path,
    )

    # ================================
    # Plot trade-off figures
    # ================================
    rate_reliability_fig_path = os.path.join(
        eval_dir,
        f"tradeoff_rate_reliability_testinj_{inj_tag}.jpg"
    )

    rate_q05snr_fig_path = os.path.join(
        eval_dir,
        f"tradeoff_rate_q05snr_testinj_{inj_tag}.jpg"
    )

    plot_tradeoff_rate_reliability(
        results=results,
        fig_path=rate_reliability_fig_path,
    )

    plot_tradeoff_rate_q05snr(
        results=results,
        fig_path=rate_q05snr_fig_path,
    )

    print("\n" + "=" * 80)
    print("[INFO] eval_sweep.py finished.")
    print("=" * 80)

    print(f"[INFO] Selection CSV          : {selection_csv_path}")
    print(f"[INFO] Selection NPZ          : {selection_npz_path}")
    print(f"[INFO] Summary                : {summary_path}")
    print(f"[INFO] REG SNR CDF            : {reg_snr_cdf_path}")
    print(f"[INFO] ROB SNR CDF            : {rob_snr_cdf_path}")
    print(f"[INFO] REG Rate CDF           : {reg_rate_cdf_path}")
    print(f"[INFO] ROB Rate CDF           : {rob_rate_cdf_path}")
    print(f"[INFO] Best SNR CDF           : {best_snr_cdf_path}")
    print(f"[INFO] Best Rate CDF          : {best_rate_cdf_path}")
    print(f"[INFO] Rate-Reliability fig   : {rate_reliability_fig_path}")
    print(f"[INFO] Rate-Q0.05 SNR fig     : {rate_q05snr_fig_path}")