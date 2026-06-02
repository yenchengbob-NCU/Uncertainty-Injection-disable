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
# Evaluation setting
# ================================
EVAL_INJECTION_CHUNK = 50

REG_PENALTY_LIST = [70, 100, 150, 200, 300]
ROB_PENALTY_LIST = [0.025, 0.05, 0.1, 0.5, 1]

# 是否在 fixed-injection evaluation 中，將 injected channels
# 正規化回 estimated channels 的原始 power。
# 目的：避免 additive Gaussian injection 額外放大 channel power，
# 使 rate / SNR 的變化主要反映 channel direction / phase mismatch。
POWER_NORMALIZE_INJECTED_CHANNELS = True
POWER_NORM_EPS = 1e-12


# ================================
# Path helpers
# ================================
def format_float_for_path(value: float) -> str:
    """
    將 float 轉成與 main_st.py 相同的資料夾名稱格式。

    例：
        0.025 -> 0p025
        0.05  -> 0p05
        0.1   -> 0p1
        0.5   -> 0p5
        1.0   -> 1
        200.0 -> 200
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
    根據 mode 與 penalty 建立 ST model checkpoint 路徑。
    需與 main_st.py 的命名一致。
    """
    mode = mode.lower()

    if mode == "reg":
        run_name = f"REG_penalty_{format_float_for_path(penalty)}"
        comm_ckpt_name = "short_comm.ckpt"
        radar_ckpt_name = "short_radar.ckpt"
    elif mode == "rob":
        run_name = f"ROB_penalty_{format_float_for_path(penalty)}"
        comm_ckpt_name = "short_comm_robust.ckpt"
        radar_ckpt_name = "short_radar_robust.ckpt"
    else:
        raise ValueError(f"mode must be 'reg' or 'rob', got {mode}")

    run_dir = os.path.join(ST_SWEEP_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")

    return {
        "run_dir": run_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, comm_ckpt_name),
        "radar_ckpt_path": os.path.join(ckpt_dir, radar_ckpt_name),
    }


def build_eval_dir() -> str:
    """
    建立 fixed-injection selection eval 的輸出資料夾。
    """
    inj_tag = format_float_for_path(INJECTION_VARIANCE)

    eval_dir = os.path.join(
        BASE_RUN_DIR,
        "eval_results",
        f"selection_testinj_{inj_tag}"
    )

    os.makedirs(eval_dir, exist_ok=True)

    return eval_dir


# ================================
# Tensor helpers
# ================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)

    nr = torch.randn(
        shape,
        device=device,
        dtype=torch.float32
    ) * sigma

    ni = torch.randn(
        shape,
        device=device,
        dtype=torch.float32
    ) * sigma

    return torch.complex(nr, ni).to(dtype=cdtype)




def normalize_injected_channel_power(
    injected: torch.Tensor,
    reference: torch.Tensor,
    norm_dims,
    eps: float = POWER_NORM_EPS,
) -> torch.Tensor:
    """
    將 injected channel 的 norm 正規化回 reference channel 的 norm。

    目的：
        1. 保留 estimated channel / large-scale fading 的原始 power。
        2. 讓 test injection 主要代表 channel direction / phase uncertainty。
        3. 避免 additive Gaussian injection 造成平均 channel power 隨 variance 增大。

    公式：
        H_norm = H_inj * ||H_ref|| / (||H_inj|| + eps)

    Args:
        injected  : 已加入 Gaussian perturbation 的 channel。
        reference : 注入前的 estimated channel，shape 必須可與 injected 對應。
        norm_dims : 計算 norm 的維度。
        eps       : 避免除以 0 的小常數。
    """
    if not POWER_NORMALIZE_INJECTED_CHANNELS:
        return injected

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
    )

    scale = ref_norm / inj_norm.clamp_min(eps)

    return injected * scale


def apply_power_normalization_to_injected_channels(
    h_dk_inj: torch.Tensor,
    h_dk_ref: torch.Tensor,
    h_rk_inj: torch.Tensor,
    h_rk_ref: torch.Tensor,
    G_inj: torch.Tensor,
    G_ref: torch.Tensor,
    g_dt_inj: torch.Tensor,
    g_dt_ref: torch.Tensor,
):
    """
    對 fixed-injection evaluation 中使用的四種 channel 做 power normalization。

    Shape convention:
        h_dk : (B*L, M, K)  -> 每個 user k 的 BS-UE direct vector 各自正規化
        h_rk : (B*L, N, K)  -> 每個 user k 的 RIS-UE vector 各自正規化
        G    : (B*L, N, M)  -> 每個 sample 的 BS-RIS matrix 用 Frobenius norm 正規化
        g_dt : (B*L, M, 1)  -> 每個 sample 的 sensing / target vector 正規化
    """
    if not POWER_NORMALIZE_INJECTED_CHANNELS:
        return h_dk_inj, h_rk_inj, G_inj, g_dt_inj

    h_dk_inj = normalize_injected_channel_power(
        injected=h_dk_inj,
        reference=h_dk_ref,
        norm_dims=1,
    )

    h_rk_inj = normalize_injected_channel_power(
        injected=h_rk_inj,
        reference=h_rk_ref,
        norm_dims=1,
    )

    G_inj = normalize_injected_channel_power(
        injected=G_inj,
        reference=G_ref,
        norm_dims=(1, 2),
    )

    g_dt_inj = normalize_injected_channel_power(
        injected=g_dt_inj,
        reference=g_dt_ref,
        norm_dims=(1, 2),
    )

    return h_dk_inj, h_rk_inj, G_inj, g_dt_inj


def reset_eval_seed():
    """
    每個 model eval 前重設 seed。
    讓 REG / ROB 在相同 test injection 下使用相同隨機擾動序列。
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
    keys = [
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
        dataset = {key: data[key] for key in keys}

    print(f"[TEST] loaded: {npz_path}")
    print(f"[TEST] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[TEST] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[TEST] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[TEST] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[TEST] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")

    return dataset


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout_np: np.ndarray
):
    """
    給一組 layout，從 shared LT net 取固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)
        theta_lt, _, _ = longterm_net(layout_t)

    return theta_lt.detach()


def extract_shortterm_batch(
    dataset,
    layout_id: int,
    channel_ids: np.ndarray
):
    """
    從 test dataset 取出指定 layout / channels 的 estimated channels。
    """
    h_dk_hat = np_to_torch_complex(
        dataset["st_h_dk_hat"][layout_id][channel_ids]
    )

    h_rk_hat = np_to_torch_complex(
        dataset["st_h_rk_hat"][layout_id][channel_ids]
    )

    G_hat = np_to_torch_complex(
        dataset["st_G_hat"][layout_id][channel_ids]
    )

    g_dt_hat = np_to_torch_complex(
        dataset["st_g_dt_hat"][layout_id][channel_ids]
    )

    return {
        "pl_BS_UE": dataset["pl_BS_UE"][layout_id],
        "pl_BS_RIS_UE": dataset["pl_BS_RIS_UE"][layout_id],
        "pl_BS_TAR_BS": dataset["pl_BS_TAR_BS"][layout_id],
        "h_dk_hat": h_dk_hat,
        "h_rk_hat": h_rk_hat,
        "G_hat": G_hat,
        "g_dt_hat": g_dt_hat,
    }


# ================================
# Model loading
# ================================
def load_longterm_model():
    """
    載入 shared LT model。
    """
    print(f"[LOAD] LONGTERM_CKPT_PATH = {LONGTERM_CKPT_PATH}")

    longterm_net = LongTermPositionNet(
        ckpt_kind=None
    ).to(DEVICE)

    longterm_net.load_model(
        path=LONGTERM_CKPT_PATH,
        verbose=True
    )

    longterm_net.eval()

    for p in longterm_net.parameters():
        p.requires_grad_(False)

    return longterm_net


def load_shortterm_model(mode: str, penalty: float):
    """
    載入指定 mode / penalty 的 ST comm/radar nets。
    """
    paths = build_st_paths(mode, penalty)

    comm_net = ShortTermCommNet(
        ckpt_kind=None
    ).to(DEVICE)

    radar_net = ShortTermRadarNet(
        ckpt_kind=None
    ).to(DEVICE)

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


# ================================
# Evaluation core
# ================================
@torch.no_grad()
def eval_one_batch_under_injection(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
):
    """
    對一個 estimated-channel batch 做 fixed test injection evaluation。

    回傳：
        sumrate_np : shape = (B * INJECTION_SAMPLES,)
        snr_db_np  : shape = (B * INJECTION_SAMPLES,)
    """
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE = batch_data["pl_BS_UE"]
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

        h_dk_rep = h_dk_hat.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        h_rk_rep = h_rk_hat.unsqueeze(1).expand(
            B, s, N, K
        ).reshape(B * s, N, K)

        G_rep = G_hat.unsqueeze(1).expand(
            B, s, N, M
        ).reshape(B * s, N, M)

        g_dt_rep = g_dt_hat.unsqueeze(1).expand(
            B, s, M, 1
        ).reshape(B * s, M, 1)

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

        h_dk_inj, h_rk_inj, G_inj, g_dt_inj = apply_power_normalization_to_injected_channels(
            h_dk_inj=h_dk_inj,
            h_dk_ref=h_dk_rep,
            h_rk_inj=h_rk_inj,
            h_rk_ref=h_rk_rep,
            G_inj=G_inj,
            G_ref=G_rep,
            g_dt_inj=g_dt_inj,
            g_dt_ref=g_dt_rep,
        )

        theta_rep = theta_batch.unsqueeze(1).expand(
            B, s, N
        ).reshape(B * s, N)

        W_C_rep = W_C.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        W_R_rep = W_R.unsqueeze(1).expand(
            B, s, M, W_R.shape[2]
        ).reshape(B * s, M, W_R.shape[2])

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
        sumrate = rates.sum(dim=1)

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
            sumrate.detach().cpu().numpy().astype(np.float32)
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

    回傳：
        result      : dict
        snr_db_all  : np.ndarray
        sumrate_all : np.ndarray
    """
    print("\n" + "=" * 80)
    print(f"[EVAL] mode={mode.upper()} penalty={penalty}")
    print("=" * 80)

    reset_eval_seed()

    comm_net, radar_net, paths = load_shortterm_model(
        mode=mode,
        penalty=penalty
    )

    n_layouts = test_dataset["ue_layouts"].shape[0]
    n_eval_channels = test_dataset["st_h_dk_hat"].shape[1]
    channel_ids = np.arange(n_eval_channels)

    sumrate_all = []
    snr_db_all = []

    for layout_id in range(n_layouts):
        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net=longterm_net,
            ue_layout_np=test_dataset["ue_layouts"][layout_id]
        )

        batch_data = extract_shortterm_batch(
            dataset=test_dataset,
            layout_id=layout_id,
            channel_ids=channel_ids
        )

        sumrate_np, snr_db_np = eval_one_batch_under_injection(
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
    feasible = bool(p_out <= OUTAGE_QUANTILE)

    result = {
        "mode": mode.upper(),
        "penalty": float(penalty),
        "mean_sumrate": mean_sumrate,
        "q05_snr_db": q05_snr_db,
        "p_out": p_out,
        "feasible": feasible,
        "num_layouts": int(n_layouts),
        "channels_per_layout": int(n_eval_channels),
        "test_injection_variance": float(INJECTION_VARIANCE),
        "test_injection_samples": int(INJECTION_SAMPLES),
        "power_normalized_injection": bool(POWER_NORMALIZE_INJECTED_CHANNELS),
        "run_dir": paths["run_dir"],
        "comm_ckpt_path": paths["comm_ckpt_path"],
        "radar_ckpt_path": paths["radar_ckpt_path"],
    }

    print(
        f"[RESULT] {mode.upper()} penalty={penalty} | "
        f"Mean SumRate={mean_sumrate:.6f} | "
        f"Q{OUTAGE_QUANTILE:.2f} SNR={q05_snr_db:.3f} dB | "
        f"P_out={100.0 * p_out:.3f}% | "
        f"Feasible={feasible}"
    )

    return result, snr_db_all, sumrate_all


# ================================
# Result saving
# ================================
def save_selection_table(results: list[dict], csv_path: str):
    """
    儲存 selection table。
    """
    fieldnames = [
        "mode",
        "penalty",
        "mean_sumrate",
        "q05_snr_db",
        "p_out",
        "feasible",
        "num_layouts",
        "channels_per_layout",
        "test_injection_variance",
        "test_injection_samples",
        "power_normalized_injection",
        "run_dir",
        "comm_ckpt_path",
        "radar_ckpt_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)

    print(f"[SAVE] selection table saved: {csv_path}")


def select_best_feasible(results: list[dict], mode: str):
    """
    在指定 mode 中，從 P_out <= OUTAGE_QUANTILE 的模型選 mean sum-rate 最大者。
    """
    mode = mode.upper()

    candidates = [
        r for r in results
        if r["mode"] == mode and r["feasible"]
    ]

    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: r["mean_sumrate"]
    )


def save_best_summary(
    best_reg,
    best_rob,
    summary_path: str
):
    """
    儲存 best feasible REG / ROB summary。
    """
    lines = [
        "Best feasible model selection\n",
        "=" * 80 + "\n",
        f"Test injection variance = {INJECTION_VARIANCE}\n",
        f"Test injection samples  = {INJECTION_SAMPLES}\n",
        f"Power-normalized inj.   = {POWER_NORMALIZE_INJECTED_CHANNELS}\n",
        f"SNR threshold           = {SENSING_SNR_THRESHOLD_dB} dB\n",
        f"Outage constraint       = {OUTAGE_QUANTILE * 100:.2f}%\n",
        "\nSelection rule:\n",
        "Among models with P_out <= outage constraint, choose the highest mean sum-rate.\n\n",
    ]

    def format_best(name, best):
        if best is None:
            return f"[{name}] No feasible model found.\n"

        return (
            f"[{name}] Best feasible model\n"
            f"  penalty        = {best['penalty']}\n"
            f"  mean_sumrate   = {best['mean_sumrate']:.6f}\n"
            f"  q05_snr_db     = {best['q05_snr_db']:.3f} dB\n"
            f"  p_out          = {100.0 * best['p_out']:.3f}%\n"
            f"  run_dir        = {best['run_dir']}\n"
        )

    lines.append(format_best("REG", best_reg))
    lines.append("\n")
    lines.append(format_best("ROB", best_rob))

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))

    print(f"[SAVE] best feasible summary saved: {summary_path}")


def save_npz_results(
    results: list[dict],
    snr_cdf_data: dict,
    sumrate_data: dict,
    npz_path: str
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
        metrics_array[i]["feasible"] = int(r["feasible"])

    save_dict = {
        "metrics": metrics_array,
        "power_normalized_injection": np.array(
            POWER_NORMALIZE_INJECTED_CHANNELS,
            dtype=np.bool_,
        ),
    }

    for key, value in snr_cdf_data.items():
        save_dict[f"snr_db_{key}"] = value.astype(np.float32)

    for key, value in sumrate_data.items():
        save_dict[f"sumrate_{key}"] = value.astype(np.float32)

    np.savez_compressed(
        npz_path,
        **save_dict
    )

    print(f"[SAVE] npz results saved: {npz_path}")


# ================================
# Plot
# ================================
def make_cdf(values: np.ndarray):
    """
    由 samples 建立 CDF。
    """
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)

    return x, y


def plot_snr_cdf_group(
    mode: str,
    penalties: list[float],
    snr_cdf_data: dict,
    fig_path: str
):
    """
    畫同一 mode 下所有 penalty 的 SNR CDF。
    """
    mode = mode.upper()

    plt.figure(figsize=(9, 5.5))

    for penalty in penalties:
        key = f"{mode}_penalty_{format_float_for_path(penalty)}"
        snr_db = snr_cdf_data[key]
        x, y = make_cdf(snr_db)

        plt.plot(
            x,
            y,
            linewidth=2.0,
            label=f"{mode} penalty={penalty}"
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
        label=f"{OUTAGE_QUANTILE * 100:.0f}% outage level"
    )

    plt.xlabel("Sensing SNR (dB)")
    plt.ylabel("CDF")
    plt.title(
        f"{mode} Sensing SNR CDF "
        f"(test injection variance = {INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fig_path,
        dpi=300,
        format="jpg"
    )

    print(f"[SAVE] SNR CDF figure saved: {fig_path}")

    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    print("[INFO] Fixed-injection selection evaluation started.")
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] TEST_DATASET_PATH = {TEST_DATASET_PATH}")
    print(f"[INFO] LONGTERM_CKPT_PATH = {LONGTERM_CKPT_PATH}")
    print(f"[INFO] ST_SWEEP_DIR = {ST_SWEEP_DIR}")

    print("\n[INFO] Evaluation setting:")
    print(f"[INFO] INJECTION_VARIANCE = {INJECTION_VARIANCE}")
    print(f"[INFO] INJECTION_SAMPLES = {INJECTION_SAMPLES}")
    print(f"[INFO] POWER_NORMALIZE_INJECTED_CHANNELS = {POWER_NORMALIZE_INJECTED_CHANNELS}")
    print(f"[INFO] EVAL_INJECTION_CHUNK = {EVAL_INJECTION_CHUNK}")
    print("[INFO] Evaluation uses all test estimated channels per layout.")
    print(f"[INFO] SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB")
    print(f"[INFO] Outage constraint = {100.0 * OUTAGE_QUANTILE:.2f}%")

    eval_dir = build_eval_dir()
    print(f"\n[INFO] EVAL_DIR = {eval_dir}")

    test_dataset = load_test_dataset(TEST_DATASET_PATH)
    longterm_net = load_longterm_model()

    results = []
    snr_cdf_data = {}
    sumrate_data = {}

    for penalty in REG_PENALTY_LIST:
        result, snr_db_all, sumrate_all = evaluate_one_model(
            mode="reg",
            penalty=penalty,
            longterm_net=longterm_net,
            test_dataset=test_dataset,
        )

        results.append(result)

        key = f"REG_penalty_{format_float_for_path(penalty)}"
        snr_cdf_data[key] = snr_db_all
        sumrate_data[key] = sumrate_all

    for penalty in ROB_PENALTY_LIST:
        result, snr_db_all, sumrate_all = evaluate_one_model(
            mode="rob",
            penalty=penalty,
            longterm_net=longterm_net,
            test_dataset=test_dataset,
        )

        results.append(result)

        key = f"ROB_penalty_{format_float_for_path(penalty)}"
        snr_cdf_data[key] = snr_db_all
        sumrate_data[key] = sumrate_all

    inj_tag = format_float_for_path(INJECTION_VARIANCE)

    csv_path = os.path.join(
        eval_dir,
        f"selection_table_testinj_{inj_tag}.csv"
    )

    npz_path = os.path.join(
        eval_dir,
        f"selection_metrics_testinj_{inj_tag}.npz"
    )

    summary_path = os.path.join(
        eval_dir,
        f"best_feasible_summary_testinj_{inj_tag}.txt"
    )

    save_selection_table(
        results=results,
        csv_path=csv_path
    )

    best_reg = select_best_feasible(
        results=results,
        mode="REG"
    )

    best_rob = select_best_feasible(
        results=results,
        mode="ROB"
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
        npz_path=npz_path
    )

    reg_fig_path = os.path.join(
        eval_dir,
        f"snr_cdf_all_REG_testinj_{inj_tag}.jpg"
    )

    rob_fig_path = os.path.join(
        eval_dir,
        f"snr_cdf_all_ROB_testinj_{inj_tag}.jpg"
    )

    plot_snr_cdf_group(
        mode="REG",
        penalties=REG_PENALTY_LIST,
        snr_cdf_data=snr_cdf_data,
        fig_path=reg_fig_path
    )

    plot_snr_cdf_group(
        mode="ROB",
        penalties=ROB_PENALTY_LIST,
        snr_cdf_data=snr_cdf_data,
        fig_path=rob_fig_path
    )

    print("\n" + "=" * 80)
    print("[INFO] Fixed-injection selection evaluation finished.")
    print("=" * 80)

    print(f"[INFO] CSV table  : {csv_path}")
    print(f"[INFO] NPZ result : {npz_path}")
    print(f"[INFO] Summary    : {summary_path}")
    print(f"[INFO] REG CDF    : {reg_fig_path}")
    print(f"[INFO] ROB CDF    : {rob_fig_path}")