# -*- coding: utf-8 -*-
"""
eval6.py

用途：
    Non-CDF robust-vs-regular analysis.

這版讀取新的 TEST 資料夾結構：

    TEST/
    └── THR_15db_INJ0.075/
        └── punish_(100.0, 0.5, 250.0)/
            ├── REG_100.0/
            │   ├── ckpt/
            │   ├── data/
            │   ├── eval_figures/
            │   └── training_curves/
            └── ROB_0.5/
                ├── ckpt/
                ├── data/
                ├── eval_figures/
                └── training_curves/

每一個 layout-channel pair 會做 EVAL_INJECTION_SAMPLES 次 uncertainty injection，
並計算：
    1. E[SumRate]
    2. Q0.05(SumRate)
    3. min SumRate
    4. Q0.05(SNR)
    5. empirical P(SNR < threshold)

本版輸出 5 種非 CDF 圖：
    fig1_constraint_violation_bar
    fig2_feasible_ratio_bar
    fig3_rate_reliability_scatter
    fig4_tail_rate_bar
    fig5_paired_dominance_ratio_bar
"""

import os
import glob
import csv
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import (
    DEVICE,
    RANDOM_SEED,
    RIS_UNIT,
)

from neural_net import (
    LongTermPositionNet,
    ShortTermCommNet,
    ShortTermRadarNet,
)


# ================================
# User config: TEST folder
# ================================
REG_RUN_DIR = r"C:\WYC_python\Uncertainty Injection\Isac test3\TEST\THR_15db_INJ0.075\punish_(100.0, 0.5, 250.0)\REG_100.0"

ROB_RUN_DIR = r"C:\WYC_python\Uncertainty Injection\Isac test3\TEST\THR_15db_INJ0.075\punish_(100.0, 0.5, 250.0)\ROB_0.5"

# 建議固定用 REG 的 long-term RIS policy
LT_RUN_DIR = REG_RUN_DIR

# test dataset 來源
TEST_DATASET_MIXED_PATH = os.path.join(
    REG_RUN_DIR,
    "data",
    "dataset_test.npz",
)

# eval6 輸出位置
OUTPUT_DIR = r"C:\WYC_python\Uncertainty Injection\Isac test3\TEST\THR_15db_INJ0.075\punish_(100.0, 0.5, 250.0)\eval6_non_cdf_analysis"

EVAL6_TAG = "REG100_ROB0p5_INJ0p075"


# ================================
# Evaluation config
# ================================
TEST_CHANNEL_BATCH = 100
INJECTION_CHUNK = 50

EVAL_INJECTION_VARIANCE = 0.075
EVAL_INJECTION_SAMPLES = 200

EVAL_SENSING_SNR_THRESHOLD_dB = 15.0
EVAL_SENSING_SNR_THRESHOLD = 10.0 ** (EVAL_SENSING_SNR_THRESHOLD_dB / 10.0)

EVAL_OUTAGE_QUANTILE = 0.05

SHOW_FIGURES = True


# ================================
# Helpers
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
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def find_ckpt(run_dir: str, patterns: list[str], tag: str) -> str:
    """
    在 run_dir/ckpt 底下依照 patterns 尋找 checkpoint。

    patterns 有優先順序。
    """
    ckpt_dir = os.path.join(run_dir, "ckpt")

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"[{tag}] 找不到 ckpt 資料夾：\n{ckpt_dir}")

    for pattern in patterns:
        matched = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))

        if len(matched) > 0:
            if len(matched) > 1:
                print("====================================================")
                print(f"[WARN] {tag} 使用 pattern={pattern} 找到多個 checkpoint，使用第一個：")
                for p in matched:
                    print(f"  {p}")
                print("====================================================")

            return matched[0]

    raise FileNotFoundError(
        f"[{tag}] 找不到 checkpoint。\n"
        f"ckpt_dir = {ckpt_dir}\n"
        f"patterns = {patterns}"
    )


def load_model_from_path(model, ckpt_path: str, tag: str):
    """
    強制指定 model.model_path 後載入 checkpoint。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{tag}] checkpoint 不存在：\n{ckpt_path}")

    model.model_path = ckpt_path
    model.load_model(verbose=True)


def load_shortterm_dataset(npz_path: str, split_name: str):
    """
    載入 short-term dataset。
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

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"[{split_name}] 找不到 dataset：\n{npz_path}")

    with np.load(npz_path) as data:
        for key in required_keys:
            if key not in data:
                raise KeyError(f"[{split_name}] dataset 缺少欄位：{key}")

        dataset = {key: data[key] for key in required_keys}

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))

    print(f"[{split_name}] loaded: {npz_path}")
    print(f"[{split_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{split_name}] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[{split_name}] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[{split_name}] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[{split_name}] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")
    print(f"[{split_name}] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")

    return dataset


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout_np: np.ndarray,
):
    """
    給一組 layout，從 long-term net 取固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)
        theta_lt, _, _ = longterm_net(layout_t)

    return theta_lt.detach()


def extract_shortterm_batch(
    dataset,
    layout_id: int,
    channel_ids: np.ndarray,
):
    """
    從 test dataset 中取出指定 layout 的指定 channel batch。
    """
    ue_layout = dataset["ue_layouts"][layout_id]
    pl_BS_UE = dataset["pl_BS_UE"][layout_id]
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]
    pl_BS_TAR_BS = dataset["pl_BS_TAR_BS"][layout_id]

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
        "ue_layout": ue_layout,
        "pl_BS_UE": pl_BS_UE,
        "pl_BS_RIS_UE": pl_BS_RIS_UE,
        "pl_BS_TAR_BS": pl_BS_TAR_BS,
        "h_dk_hat": h_dk_hat,
        "h_rk_hat": h_rk_hat,
        "G_hat": G_hat,
        "g_dt_hat": g_dt_hat,
    }


def save_or_show(fig_path: str):
    plt.tight_layout()
    plt.savefig(fig_path, format="jpg", dpi=300)

    if SHOW_FIGURES:
        plt.show()

    plt.close()

    print(f"[EVAL6] Saved figure: {fig_path}")


def add_bar_labels(ax, bars, fmt="{:.3f}"):
    """
    在 bar 上標數值。
    """
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


# ================================
# Injected evaluation metrics
# ================================
@torch.no_grad()
def eval_metrics_injected(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
    injection_samples: int,
):
    """
    用固定 theta_LT + estimated channels 產生 W_C / W_R，
    再對通道做 injection，計算：
        1. E[SumRate]
        2. Q0.05(SumRate)
        3. min SumRate
        4. Q0.05(SNR) in dB
        5. empirical P(SNR < threshold)
        6. violation count
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
    L = int(injection_samples)
    q = float(EVAL_OUTAGE_QUANTILE)
    eps = 1e-12

    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    # ----------------------------
    # 1) 用 estimated channels 產生 beamformers
    # ----------------------------
    comm_net.eval()
    radar_net.eval()

    W_C = comm_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch,
    )

    W_R = radar_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch,
    )

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)

    # ----------------------------
    # 2) injection 後統計
    # ----------------------------
    sumrate_chunks = []
    snr_chunks = []

    radar_streams = W_R.shape[2]

    for s0 in range(0, L, INJECTION_CHUNK):
        s = min(INJECTION_CHUNK, L - s0)

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
            EVAL_INJECTION_VARIANCE,
            DEVICE,
            h_dk_rep.dtype,
        )

        h_rk_inj = h_rk_rep + complex_awgn(
            h_rk_rep.shape,
            EVAL_INJECTION_VARIANCE,
            DEVICE,
            h_rk_rep.dtype,
        )

        G_inj = G_rep + complex_awgn(
            G_rep.shape,
            EVAL_INJECTION_VARIANCE,
            DEVICE,
            G_rep.dtype,
        )

        g_dt_inj = g_dt_rep + complex_awgn(
            g_dt_rep.shape,
            EVAL_INJECTION_VARIANCE,
            DEVICE,
            g_dt_rep.dtype,
        )

        theta_rep = theta_batch.unsqueeze(1).expand(
            B, s, N
        ).reshape(B * s, N)

        W_C_rep = W_C.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        W_R_rep = W_R.unsqueeze(1).expand(
            B, s, M, radar_streams
        ).reshape(B * s, M, radar_streams)

        sumrate = comm_net.compute_sum_rate(
            h_dk=h_dk_inj,
            h_rk=h_rk_inj,
            G=G_inj,
            theta=theta_rep,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE,
        )

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real

        sumrate_chunks.append(sumrate.reshape(B, s))
        snr_chunks.append(sense_snr.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)     # shape = (B,L)
    snr_samples = torch.cat(snr_chunks, dim=1)             # shape = (B,L)

    # ----------------------------
    # 3) 統計指標
    # ----------------------------
    k_idx = max(1, int(np.ceil(q * L)))
    k_idx = min(k_idx, L)

    sumrate_mean = sumrate_samples.mean(dim=1)

    sumrate_q05 = torch.kthvalue(
        sumrate_samples,
        k=k_idx,
        dim=1,
    ).values

    sumrate_min = sumrate_samples.min(dim=1).values

    snr_q05 = torch.kthvalue(
        snr_samples,
        k=k_idx,
        dim=1,
    ).values

    snr_q05_db = 10.0 * torch.log10(snr_q05.clamp_min(eps))

    snr_violation_bool = snr_samples < EVAL_SENSING_SNR_THRESHOLD
    snr_violation_count = snr_violation_bool.float().sum(dim=1)
    snr_violation_prob = snr_violation_bool.float().mean(dim=1)

    feasible_mask = snr_q05 >= EVAL_SENSING_SNR_THRESHOLD

    return {
        "sumrate_mean": sumrate_mean.detach().cpu().numpy().astype(np.float32),
        "sumrate_q05": sumrate_q05.detach().cpu().numpy().astype(np.float32),
        "sumrate_min": sumrate_min.detach().cpu().numpy().astype(np.float32),
        "snr_q05_db": snr_q05_db.detach().cpu().numpy().astype(np.float32),
        "snr_vprob": snr_violation_prob.detach().cpu().numpy().astype(np.float32),
        "snr_vcount": snr_violation_count.detach().cpu().numpy().astype(np.float32),
        "feasible_mask": feasible_mask.detach().cpu().numpy().astype(bool),
    }


def summarize_method(prefix: str, metrics: dict, total_injections: int) -> dict:
    """
    產生 summary dictionary。
    """
    sample_violation_count = int(np.sum(metrics["snr_vcount"]))
    sample_violation_prob = sample_violation_count / total_injections

    feasible_count = int(np.sum(metrics["feasible_mask"]))
    feasible_ratio = feasible_count / metrics["feasible_mask"].size

    return {
        "method": prefix,
        "num_layout_channel_pairs": int(metrics["sumrate_mean"].size),
        "num_injected_samples": int(total_injections),
        "sample_violation_count": sample_violation_count,
        "sample_violation_prob": sample_violation_prob,
        "mean_per_channel_vprob": float(np.mean(metrics["snr_vprob"])),
        "feasible_count": feasible_count,
        "feasible_ratio": feasible_ratio,
        "mean_sumrate": float(np.mean(metrics["sumrate_mean"])),
        "mean_q05_sumrate": float(np.mean(metrics["sumrate_q05"])),
        "mean_min_sumrate": float(np.mean(metrics["sumrate_min"])),
        "mean_q05_snr_db": float(np.mean(metrics["snr_q05_db"])),
    }


def save_summary_csv(summary_rows: list[dict], csv_path: str):
    fieldnames = [
        "method",
        "num_layout_channel_pairs",
        "num_injected_samples",
        "sample_violation_count",
        "sample_violation_prob",
        "mean_per_channel_vprob",
        "feasible_count",
        "feasible_ratio",
        "mean_sumrate",
        "mean_q05_sumrate",
        "mean_min_sumrate",
        "mean_q05_snr_db",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)

    print(f"[EVAL6] Saved summary CSV: {csv_path}")


def save_summary_txt(reg_summary: dict, rob_summary: dict, dominance: dict, txt_path: str):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("====================================================\n")
        f.write("[EVAL6 Non-CDF Summary]\n")
        f.write(f"EVAL6_TAG = {EVAL6_TAG}\n")
        f.write(f"EVAL_INJECTION_VARIANCE = {EVAL_INJECTION_VARIANCE}\n")
        f.write(f"EVAL_INJECTION_SAMPLES = {EVAL_INJECTION_SAMPLES}\n")
        f.write(f"EVAL_THRESHOLD = {EVAL_SENSING_SNR_THRESHOLD_dB} dB\n")
        f.write("====================================================\n\n")

        for s in [reg_summary, rob_summary]:
            f.write(f"[{s['method']}]\n")
            f.write(f"  sample violation count = {s['sample_violation_count']:,} / {s['num_injected_samples']:,}\n")
            f.write(f"  sample violation prob  = {s['sample_violation_prob'] * 100:.3f} %\n")
            f.write(f"  feasible pairs         = {s['feasible_count']:,} / {s['num_layout_channel_pairs']:,}\n")
            f.write(f"  feasible ratio         = {s['feasible_ratio'] * 100:.3f} %\n")
            f.write(f"  mean SumRate           = {s['mean_sumrate']:.6f} bits/s/Hz\n")
            f.write(f"  mean Q0.05 SumRate     = {s['mean_q05_sumrate']:.6f} bits/s/Hz\n")
            f.write(f"  mean min SumRate       = {s['mean_min_sumrate']:.6f} bits/s/Hz\n")
            f.write(f"  mean Q0.05 SNR         = {s['mean_q05_snr_db']:.3f} dB\n")
            f.write("\n")

        f.write("[Paired dominance ratios]\n")
        for k, v in dominance.items():
            f.write(f"  {k} = {v * 100:.3f} %\n")

    print(f"[EVAL6] Saved summary TXT: {txt_path}")


def compute_dominance(reg_metrics: dict, rob_metrics: dict) -> dict:
    """
    逐 layout-channel pair 比較 ROB 是否勝過 REG。
    """
    eps = 1e-12

    rob_higher_mean_rate = rob_metrics["sumrate_mean"] > reg_metrics["sumrate_mean"] + eps
    rob_higher_q05_rate = rob_metrics["sumrate_q05"] > reg_metrics["sumrate_q05"] + eps
    rob_higher_min_rate = rob_metrics["sumrate_min"] > reg_metrics["sumrate_min"] + eps
    rob_lower_vprob = rob_metrics["snr_vprob"] < reg_metrics["snr_vprob"] - eps
    rob_higher_q05_snr = rob_metrics["snr_q05_db"] > reg_metrics["snr_q05_db"] + eps

    rob_dominates_mean_rate_and_vprob = rob_higher_mean_rate & rob_lower_vprob
    rob_dominates_tail_rate_and_vprob = rob_higher_q05_rate & rob_lower_vprob

    return {
        "ROB higher E[SumRate]": float(np.mean(rob_higher_mean_rate)),
        "ROB higher Q0.05(SumRate)": float(np.mean(rob_higher_q05_rate)),
        "ROB higher min SumRate": float(np.mean(rob_higher_min_rate)),
        "ROB lower violation probability": float(np.mean(rob_lower_vprob)),
        "ROB higher Q0.05(SNR)": float(np.mean(rob_higher_q05_snr)),
        "ROB higher E[SumRate] and lower violation": float(np.mean(rob_dominates_mean_rate_and_vprob)),
        "ROB higher Q0.05(SumRate) and lower violation": float(np.mean(rob_dominates_tail_rate_and_vprob)),
    }


# ================================
# Plot functions: non-CDF
# ================================
def plot_constraint_violation_bar(reg_summary: dict, rob_summary: dict):
    labels = ["REG", "ROB"]
    values = [
        reg_summary["sample_violation_prob"] * 100.0,
        rob_summary["sample_violation_prob"] * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(labels, values)

    ax.axhline(
        EVAL_OUTAGE_QUANTILE * 100.0,
        linestyle="--",
        label=f"Target = {EVAL_OUTAGE_QUANTILE * 100:.1f}%",
    )

    ax.set_ylabel("Sample-level P(SNR < threshold) (%)")
    ax.set_title("Constraint Violation Probability")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    add_bar_labels(ax, bars, fmt="{:.3f}%")

    save_or_show(
        os.path.join(
            OUTPUT_DIR,
            f"fig1_constraint_violation_bar_{EVAL6_TAG}.jpg",
        )
    )


def plot_feasible_ratio_bar(reg_summary: dict, rob_summary: dict):
    labels = ["REG", "ROB"]
    values = [
        reg_summary["feasible_ratio"] * 100.0,
        rob_summary["feasible_ratio"] * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bars = ax.bar(labels, values)

    ax.set_ylabel("Feasible layout-channel ratio (%)")
    ax.set_title(
        f"Feasible Ratio: Q{EVAL_OUTAGE_QUANTILE:.2f}(SNR) ≥ "
        f"{EVAL_SENSING_SNR_THRESHOLD_dB:.0f} dB"
    )
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    add_bar_labels(ax, bars, fmt="{:.3f}%")

    save_or_show(
        os.path.join(
            OUTPUT_DIR,
            f"fig2_feasible_ratio_bar_{EVAL6_TAG}.jpg",
        )
    )


def plot_rate_reliability_scatter(reg_metrics: dict, rob_metrics: dict):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))

    ax.scatter(
        reg_metrics["snr_vprob"] * 100.0,
        reg_metrics["sumrate_mean"],
        s=8,
        alpha=0.35,
        label="REG",
    )

    ax.scatter(
        rob_metrics["snr_vprob"] * 100.0,
        rob_metrics["sumrate_mean"],
        s=8,
        alpha=0.35,
        label="ROB",
    )

    ax.axvline(
        EVAL_OUTAGE_QUANTILE * 100.0,
        linestyle="--",
        label=f"Target = {EVAL_OUTAGE_QUANTILE * 100:.1f}%",
    )

    ax.set_xlabel("Empirical P(SNR < threshold) per layout-channel (%)")
    ax.set_ylabel("E[SumRate] over injections (bits/s/Hz)")
    ax.set_title("Rate-Reliability Trade-off")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    save_or_show(
        os.path.join(
            OUTPUT_DIR,
            f"fig3_rate_reliability_scatter_{EVAL6_TAG}.jpg",
        )
    )


def plot_tail_rate_bar(reg_summary: dict, rob_summary: dict):
    metric_labels = ["Mean Q0.05(SumRate)", "Mean min SumRate"]
    reg_values = [
        reg_summary["mean_q05_sumrate"],
        reg_summary["mean_min_sumrate"],
    ]
    rob_values = [
        rob_summary["mean_q05_sumrate"],
        rob_summary["mean_min_sumrate"],
    ]

    x = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bars_reg = ax.bar(x - width / 2, reg_values, width, label="REG")
    bars_rob = ax.bar(x + width / 2, rob_values, width, label="ROB")

    ax.set_ylabel("SumRate (bits/s/Hz)")
    ax.set_title("Tail-Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    add_bar_labels(ax, bars_reg, fmt="{:.3f}")
    add_bar_labels(ax, bars_rob, fmt="{:.3f}")

    save_or_show(
        os.path.join(
            OUTPUT_DIR,
            f"fig4_tail_rate_bar_{EVAL6_TAG}.jpg",
        )
    )


def plot_paired_dominance_ratio_bar(dominance: dict):
    labels = list(dominance.keys())
    values = [v * 100.0 for v in dominance.values()]

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    bars = ax.bar(np.arange(len(labels)), values)

    ax.set_ylabel("Ratio of layout-channel pairs (%)")
    ax.set_title("Paired Dominance Ratio: ROB vs REG")
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    add_bar_labels(ax, bars, fmt="{:.2f}%")

    save_or_show(
        os.path.join(
            OUTPUT_DIR,
            f"fig5_paired_dominance_ratio_bar_{EVAL6_TAG}.jpg",
        )
    )


# ================================
# Main
# ================================
if __name__ == "__main__":

    # 固定 random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("====================================================")
    print("[EVAL6] Non-CDF robust-vs-regular analysis")
    print("[EVAL6] Explicit TEST folder mode")
    print("----------------------------------------------------")
    print(f"REG_RUN_DIR = {REG_RUN_DIR}")
    print(f"ROB_RUN_DIR = {ROB_RUN_DIR}")
    print(f"LT_RUN_DIR  = {LT_RUN_DIR}")
    print(f"TEST_DATASET_MIXED_PATH = {TEST_DATASET_MIXED_PATH}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print("----------------------------------------------------")
    print(f"EVAL_INJECTION_VARIANCE = {EVAL_INJECTION_VARIANCE}")
    print(f"EVAL_INJECTION_SAMPLES  = {EVAL_INJECTION_SAMPLES}")
    print(f"EVAL_THRESHOLD          = {EVAL_SENSING_SNR_THRESHOLD_dB} dB")
    print(f"EVAL_OUTAGE_QUANTILE    = {EVAL_OUTAGE_QUANTILE}")
    print("====================================================")

    # ================================
    # 1) 載入 test dataset
    # ================================
    test_dataset = load_shortterm_dataset(
        TEST_DATASET_MIXED_PATH,
        "test",
    )

    n_test_layouts = test_dataset["ue_layouts"].shape[0]
    n_channels_per_layout = test_dataset["st_h_dk_hat"].shape[1]

    n_total_layout_channel_pairs = n_test_layouts * n_channels_per_layout
    n_total_injected_samples = n_total_layout_channel_pairs * EVAL_INJECTION_SAMPLES

    print(f"[EVAL6] n_test_layouts = {n_test_layouts}")
    print(f"[EVAL6] channels_per_layout = {n_channels_per_layout}")
    print(f"[EVAL6] total layout-channel pairs = {n_total_layout_channel_pairs:,}")
    print(f"[EVAL6] total injected samples = {n_total_injected_samples:,}")

    # ================================
    # 2) 建立模型
    # ================================
    longterm_net = LongTermPositionNet(
        ckpt_kind="longterm"
    ).to(DEVICE)

    short_comm_reg = ShortTermCommNet(
        ckpt_kind="short_comm"
    ).to(DEVICE)

    short_radar_reg = ShortTermRadarNet(
        ckpt_kind="short_radar"
    ).to(DEVICE)

    short_comm_rob = ShortTermCommNet(
        ckpt_kind="short_comm_robust"
    ).to(DEVICE)

    short_radar_rob = ShortTermRadarNet(
        ckpt_kind="short_radar_robust"
    ).to(DEVICE)

    # ================================
    # 3) checkpoint loading
    # ================================
    longterm_ckpt = find_ckpt(
        LT_RUN_DIR,
        ["longterm_*.ckpt"],
        "LongTerm",
    )

    short_comm_reg_ckpt = find_ckpt(
        REG_RUN_DIR,
        ["short_comm_*.ckpt"],
        "REG-Comm",
    )

    short_radar_reg_ckpt = find_ckpt(
        REG_RUN_DIR,
        ["short_radar_*.ckpt"],
        "REG-Radar",
    )

    short_comm_rob_ckpt = find_ckpt(
        ROB_RUN_DIR,
        ["short_comm_robust_*.ckpt", "short_comm_*.ckpt"],
        "ROB-Comm",
    )

    short_radar_rob_ckpt = find_ckpt(
        ROB_RUN_DIR,
        ["short_radar_robust_*.ckpt", "short_radar_*.ckpt"],
        "ROB-Radar",
    )

    print("====================================================")
    print("[EVAL6] Checkpoint sources")
    print(f"LT ckpt        = {longterm_ckpt}")
    print(f"REG comm ckpt  = {short_comm_reg_ckpt}")
    print(f"REG radar ckpt = {short_radar_reg_ckpt}")
    print(f"ROB comm ckpt  = {short_comm_rob_ckpt}")
    print(f"ROB radar ckpt = {short_radar_rob_ckpt}")
    print("====================================================")

    load_model_from_path(
        longterm_net,
        longterm_ckpt,
        "LongTerm",
    )

    load_model_from_path(
        short_comm_reg,
        short_comm_reg_ckpt,
        "REG-Comm",
    )

    load_model_from_path(
        short_radar_reg,
        short_radar_reg_ckpt,
        "REG-Radar",
    )

    load_model_from_path(
        short_comm_rob,
        short_comm_rob_ckpt,
        "ROB-Comm",
    )

    load_model_from_path(
        short_radar_rob,
        short_radar_rob_ckpt,
        "ROB-Radar",
    )

    # ================================
    # 4) evaluation loop
    # ================================
    reg_lists = {
        "sumrate_mean": [],
        "sumrate_q05": [],
        "sumrate_min": [],
        "snr_q05_db": [],
        "snr_vprob": [],
        "snr_vcount": [],
        "feasible_mask": [],
    }

    rob_lists = {
        "sumrate_mean": [],
        "sumrate_q05": [],
        "sumrate_min": [],
        "snr_q05_db": [],
        "snr_vprob": [],
        "snr_vcount": [],
        "feasible_mask": [],
    }

    for layout_id in range(n_test_layouts):
        ue_layout = test_dataset["ue_layouts"][layout_id]

        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net,
            ue_layout,
        )

        for ch0 in range(0, n_channels_per_layout, TEST_CHANNEL_BATCH):
            ch1 = min(ch0 + TEST_CHANNEL_BATCH, n_channels_per_layout)
            channel_ids = np.arange(ch0, ch1)

            batch_data = extract_shortterm_batch(
                test_dataset,
                layout_id,
                channel_ids,
            )

            # REG
            reg_metrics = eval_metrics_injected(
                comm_net=short_comm_reg,
                radar_net=short_radar_reg,
                theta_fixed=theta_fixed,
                batch_data=batch_data,
                injection_samples=EVAL_INJECTION_SAMPLES,
            )

            for key in reg_lists:
                reg_lists[key].append(reg_metrics[key])

            # ROB
            rob_metrics = eval_metrics_injected(
                comm_net=short_comm_rob,
                radar_net=short_radar_rob,
                theta_fixed=theta_fixed,
                batch_data=batch_data,
                injection_samples=EVAL_INJECTION_SAMPLES,
            )

            for key in rob_lists:
                rob_lists[key].append(rob_metrics[key])

        print(f"[EVAL6] layout {layout_id + 1}/{n_test_layouts} done.")

    # ================================
    # 5) concat
    # ================================
    reg_all = {}
    rob_all = {}

    for key in reg_lists:
        reg_all[key] = np.concatenate(reg_lists[key], axis=0)
        rob_all[key] = np.concatenate(rob_lists[key], axis=0)

    # ================================
    # 6) summary
    # ================================
    reg_summary = summarize_method(
        prefix="REG",
        metrics=reg_all,
        total_injections=n_total_injected_samples,
    )

    rob_summary = summarize_method(
        prefix="ROB",
        metrics=rob_all,
        total_injections=n_total_injected_samples,
    )

    dominance = compute_dominance(
        reg_metrics=reg_all,
        rob_metrics=rob_all,
    )

    print("====================================================")
    print("[EVAL6 Results]")
    print(
        f"Evaluation setting: "
        f"{n_test_layouts} layouts × "
        f"{n_channels_per_layout} estimated channels/layout × "
        f"{EVAL_INJECTION_SAMPLES} injections/channel"
    )
    print()

    for s in [reg_summary, rob_summary]:
        print(f"[{s['method']}]")
        print(f"  sample violation count = {s['sample_violation_count']:,} / {s['num_injected_samples']:,}")
        print(f"  sample violation prob  = {s['sample_violation_prob'] * 100:.3f} %")
        print(f"  feasible pairs         = {s['feasible_count']:,} / {s['num_layout_channel_pairs']:,}")
        print(f"  feasible ratio         = {s['feasible_ratio'] * 100:.3f} %")
        print(f"  mean SumRate           = {s['mean_sumrate']:.6f} bits/s/Hz")
        print(f"  mean Q0.05 SumRate     = {s['mean_q05_sumrate']:.6f} bits/s/Hz")
        print(f"  mean min SumRate       = {s['mean_min_sumrate']:.6f} bits/s/Hz")
        print(f"  mean Q0.05 SNR         = {s['mean_q05_snr_db']:.3f} dB")
        print()

    print("[Paired dominance ratios]")
    for k, v in dominance.items():
        print(f"  {k:<48s} = {v * 100:.3f} %")
    print("====================================================")

    # ================================
    # 7) save metrics
    # ================================
    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"eval6_non_cdf_metrics_{EVAL6_TAG}.npz",
    )

    np.savez_compressed(
        metrics_path,
        reg_sumrate_mean=reg_all["sumrate_mean"],
        reg_sumrate_q05=reg_all["sumrate_q05"],
        reg_sumrate_min=reg_all["sumrate_min"],
        reg_snr_q05_db=reg_all["snr_q05_db"],
        reg_snr_vprob=reg_all["snr_vprob"],
        reg_snr_vcount=reg_all["snr_vcount"],
        reg_feasible_mask=reg_all["feasible_mask"],
        rob_sumrate_mean=rob_all["sumrate_mean"],
        rob_sumrate_q05=rob_all["sumrate_q05"],
        rob_sumrate_min=rob_all["sumrate_min"],
        rob_snr_q05_db=rob_all["snr_q05_db"],
        rob_snr_vprob=rob_all["snr_vprob"],
        rob_snr_vcount=rob_all["snr_vcount"],
        rob_feasible_mask=rob_all["feasible_mask"],
    )

    print(f"[EVAL6] metrics saved to: {metrics_path}")

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"eval6_non_cdf_summary_{EVAL6_TAG}.csv",
    )

    save_summary_csv(
        summary_rows=[reg_summary, rob_summary],
        csv_path=csv_path,
    )

    txt_path = os.path.join(
        OUTPUT_DIR,
        f"eval6_non_cdf_summary_{EVAL6_TAG}.txt",
    )

    save_summary_txt(
        reg_summary=reg_summary,
        rob_summary=rob_summary,
        dominance=dominance,
        txt_path=txt_path,
    )

    # ================================
    # 8) non-CDF figures
    # ================================
    plot_constraint_violation_bar(
        reg_summary=reg_summary,
        rob_summary=rob_summary,
    )

    plot_feasible_ratio_bar(
        reg_summary=reg_summary,
        rob_summary=rob_summary,
    )

    plot_rate_reliability_scatter(
        reg_metrics=reg_all,
        rob_metrics=rob_all,
    )

    plot_tail_rate_bar(
        reg_summary=reg_summary,
        rob_summary=rob_summary,
    )

    plot_paired_dominance_ratio_bar(
        dominance=dominance,
    )

    print("====================================================")
    print(f"[EVAL6] Saved metrics, summary, and 5 non-CDF figures to:")
    print(f"  {OUTPUT_DIR}")
    print("[EVAL6] Finished.")
    print("====================================================")