# -*- coding: utf-8 -*-
"""
eval2.py

用途：
    Feasible minimum injected SumRate evaluation.

這版設計給你新的資料夾結構：

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

功能：
    1. 直接指定 REG_RUN_DIR 與 ROB_RUN_DIR。
    2. REG model 從 REG_RUN_DIR/ckpt 讀取。
    3. ROB model 從 ROB_RUN_DIR/ckpt 讀取。
    4. test dataset 從 REG_RUN_DIR/data/dataset_test.npz 讀取。
    5. 對每個 layout-channel pair 做 200 次 injection。
    6. 若 Q0.05(SNR over injections) >= 15 dB，該 layout-channel pair 視為 feasible。
    7. 對 feasible layout-channel pair 畫 minimum injected SumRate CDF。
"""

import os
import glob
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ================================
# User config
# ================================
REG_RUN_DIR = r"C:\WYC_python\Uncertainty Injection\Isac test3\TEST\THR_15db_INJ0.075\punish_(100.0, 0.5, 250.0)\REG_100.0"

ROB_RUN_DIR = r"C:\WYC_python\Uncertainty Injection\Isac test3\TEST\THR_15db_INJ0.075\punish_(100.0, 0.5, 250.0)\ROB_0.5"

OUTPUT_DIR = r"C:\WYC_python\Uncertainty Injection\Isac test3\TEST\THR_15db_INJ0.075\punish_(100.0, 0.5, 250.0)\eval2_feasible_minrate"

# Long-term RIS policy 來源
# 建議固定使用 REG_RUN_DIR 的 longterm，避免 REG/ROB 比較混入不同 long-term RIS policy
LT_RUN_DIR = REG_RUN_DIR

# Dataset 來源
TEST_DATASET_MIXED_PATH = os.path.join(
    REG_RUN_DIR,
    "data",
    "dataset_test.npz",
)

# Evaluation chunk
EVAL_CHUNK = 50

# Output tag
EVAL_TAG = "REG100_ROB0p5_INJ0p075"


# ================================
# Helpers
# ================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def empirical_cdf(x: np.ndarray):
    x_sorted = np.sort(x)
    y = np.arange(1, x_sorted.size + 1, dtype=np.float64) / x_sorted.size
    return x_sorted, y


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance)
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def find_ckpt(run_dir: str, patterns: list[str], tag: str) -> str:
    """
    在 run_dir/ckpt 底下依照 patterns 尋找 checkpoint。

    patterns 範例：
        ["longterm_*.ckpt"]
        ["short_comm_*.ckpt"]
        ["short_comm_robust_*.ckpt", "short_comm_*.ckpt"]

    若找到多個，使用排序後第一個。
    """
    ckpt_dir = os.path.join(run_dir, "ckpt")

    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"[{tag}] 找不到 ckpt 資料夾：\n{ckpt_dir}")

    matched = []

    for pattern in patterns:
        matched.extend(glob.glob(os.path.join(ckpt_dir, pattern)))

    matched = sorted(set(matched))

    if len(matched) == 0:
        raise FileNotFoundError(
            f"[{tag}] 找不到 checkpoint。\n"
            f"ckpt_dir = {ckpt_dir}\n"
            f"patterns = {patterns}"
        )

    if len(matched) > 1:
        print("====================================================")
        print(f"[WARN] {tag} 找到多個 checkpoint，使用第一個：")
        for p in matched:
            print(f"  {p}")
        print("====================================================")

    return matched[0]


def load_model_from_path(model, ckpt_path: str, tag: str):
    """
    強制指定 model.model_path 後載入 checkpoint。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{tag}] checkpoint 不存在：\n{ckpt_path}")

    model.model_path = ckpt_path
    model.load_model(verbose=True)


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

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到 test dataset：\n{npz_path}")

    with np.load(npz_path) as data:
        for key in required_keys:
            if key not in data:
                raise KeyError(f"test dataset 缺少欄位：{key}")

        dataset = {key: data[key] for key in required_keys}

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))

    print(f"[test] loaded: {npz_path}")
    print(f"[test] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[test] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[test] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[test] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[test] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")
    print(f"[test] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")

    return dataset


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout_np: np.ndarray,
):
    """
    給定一組 layout，從 long-term net 取得固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)
        theta_lt, _, _ = longterm_net(layout_t)

    return theta_lt.detach()


def extract_test_batch(
    dataset,
    layout_id: int,
    channel_ids: np.ndarray,
):
    """
    從 test dataset 中取出指定 layout 與指定 estimated channel ids。
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
        "ue_layout": dataset["ue_layouts"][layout_id],
        "pl_BS_UE": dataset["pl_BS_UE"][layout_id],
        "pl_BS_RIS_UE": dataset["pl_BS_RIS_UE"][layout_id],
        "pl_BS_TAR_BS": dataset["pl_BS_TAR_BS"][layout_id],
        "h_dk_hat": h_dk_hat,
        "h_rk_hat": h_rk_hat,
        "G_hat": G_hat,
        "g_dt_hat": g_dt_hat,
    }


def compute_shortterm_outputs(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
):
    """
    根據 fixed theta_LT 與 estimated channels 產生 short-term beamformers。
    """
    comm_net.eval()
    radar_net.eval()

    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    B = h_dk_hat.shape[0]
    theta_batch = theta_fixed.expand(B, RIS_UNIT)

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

    return W_C, W_R


@torch.no_grad()
def eval_feasible_minrate_one_model(
    comm_net: ShortTermCommNet,
    theta_fixed: torch.Tensor,
    W_C: torch.Tensor,
    W_R: torch.Tensor,
    batch_data: dict,
    injection_samples: int,
    injection_variance: float,
    chunk: int,
):
    """
    對單一 model 計算 feasible minimum injected SumRate。

    每個 layout-channel pair 有 L 個 injected samples。

    指標：
        1. raw violation count:
            count_i[SNR_i < threshold]

        2. per-channel violation probability:
            P_c = count_i[SNR_i < threshold] / L

        3. 5% SNR quantile:
            SNR_q05[c] = Q_0.05(SNR[c,1:L])

        4. feasible condition:
            SNR_q05[c] >= SENSING_SNR_THRESHOLD

        5. min injected SumRate:
            R_min[c] = min_i SumRate[c,i]
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

    theta_batch = theta_fixed.expand(B, N)

    min_injected_sumrate = torch.full(
        (B,),
        float("inf"),
        device=DEVICE,
        dtype=torch.float32,
    )

    violation_count = torch.zeros(
        (B,),
        device=DEVICE,
        dtype=torch.float32,
    )

    snr_chunks = []

    for s0 in range(0, L, chunk):
        s = min(chunk, L - s0)

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
            injection_variance,
            DEVICE,
            h_dk_rep.dtype,
        )

        h_rk_inj = h_rk_rep + complex_awgn(
            h_rk_rep.shape,
            injection_variance,
            DEVICE,
            h_rk_rep.dtype,
        )

        G_inj = G_rep + complex_awgn(
            G_rep.shape,
            injection_variance,
            DEVICE,
            G_rep.dtype,
        )

        g_dt_inj = g_dt_rep + complex_awgn(
            g_dt_rep.shape,
            injection_variance,
            DEVICE,
            g_dt_rep.dtype,
        )

        theta_rep = theta_batch.unsqueeze(1).expand(
            B, s, N
        ).reshape(B * s, N)

        W_C_rep = W_C.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        radar_streams = W_R.shape[2]

        W_R_rep = W_R.unsqueeze(1).expand(
            B, s, M, radar_streams
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
        sum_rate = rates.sum(dim=1).reshape(B, s)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real.reshape(B, s)

        chunk_min_rate = sum_rate.min(dim=1).values
        min_injected_sumrate = torch.minimum(
            min_injected_sumrate,
            chunk_min_rate,
        )

        violation_count += (
            sense_snr < SENSING_SNR_THRESHOLD
        ).float().sum(dim=1)

        snr_chunks.append(sense_snr)

    snr_samples = torch.cat(snr_chunks, dim=1)

    kth = int(math.ceil(OUTAGE_QUANTILE * L))
    kth = max(1, min(kth, L))

    snr_q05 = torch.kthvalue(
        snr_samples,
        k=kth,
        dim=1,
    ).values

    feasible_mask = snr_q05 >= SENSING_SNR_THRESHOLD
    feasible_min_rate = min_injected_sumrate[feasible_mask]

    per_channel_vprob = violation_count / float(L)

    return {
        "feasible_min_rate": feasible_min_rate.detach().cpu().numpy().astype(np.float32),
        "all_min_rate": min_injected_sumrate.detach().cpu().numpy().astype(np.float32),
        "snr_q05": snr_q05.detach().cpu().numpy().astype(np.float32),
        "feasible_mask": feasible_mask.detach().cpu().numpy().astype(bool),
        "violation_count": violation_count.detach().cpu().numpy().astype(np.float32),
        "per_channel_vprob": per_channel_vprob.detach().cpu().numpy().astype(np.float32),
    }


def save_cdf_plot(
    x_reg: np.ndarray,
    x_rob: np.ndarray,
    save_path: str,
    eval_tag: str,
):
    """
    儲存 feasible minimum injected SumRate CDF。
    """
    plt.figure(figsize=(8.5, 5.5))

    if len(x_reg) > 0:
        x_reg_cdf, y_reg_cdf = empirical_cdf(x_reg)
        plt.plot(
            x_reg_cdf,
            y_reg_cdf,
            label=f"REG: feasible min injected SumRate (N={len(x_reg):,})",
        )

    if len(x_rob) > 0:
        x_rob_cdf, y_rob_cdf = empirical_cdf(x_rob)
        plt.plot(
            x_rob_cdf,
            y_rob_cdf,
            label=f"ROB: feasible min injected SumRate (N={len(x_rob):,})",
        )

    plt.xlabel("Minimum injected SumRate per feasible layout-channel pair (bits/s/Hz)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"CDF of feasible minimum injected SumRate — {eval_tag}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_path, format="jpg", dpi=300)
    print(f"[FEASIBLE-MIN-RATE-CDF] Saved figure: {save_path}")

    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    print("====================================================")
    print("[EVAL2] Feasible minimum injected SumRate evaluation")
    print("[EVAL2] Explicit folder mode")
    print("----------------------------------------------------")
    print(f"REG_RUN_DIR = {REG_RUN_DIR}")
    print(f"ROB_RUN_DIR = {ROB_RUN_DIR}")
    print(f"LT_RUN_DIR  = {LT_RUN_DIR}")
    print(f"TEST_DATASET_MIXED_PATH = {TEST_DATASET_MIXED_PATH}")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print("====================================================")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    test_dataset = load_test_dataset(TEST_DATASET_MIXED_PATH)

    n_layouts = test_dataset["ue_layouts"].shape[0]
    eval_layout_ids = np.arange(n_layouts, dtype=np.int32)
    n_eval_layouts = len(eval_layout_ids)

    n_total_layout_channel_pairs = (
        n_eval_layouts
        * SHORTTERM_EST_CHANNELS_PER_LAYOUT
    )

    n_total_injected_samples = (
        n_total_layout_channel_pairs
        * INJECTION_SAMPLES
    )

    print("====================================================")
    print("[EVAL2] Evaluation setting")
    print(f"layouts                         = {n_eval_layouts}")
    print(f"estimated channels/layout       = {SHORTTERM_EST_CHANNELS_PER_LAYOUT}")
    print(f"injections/estimated channel    = {INJECTION_SAMPLES}")
    print(f"total layout-channel pairs      = {n_total_layout_channel_pairs:,}")
    print(f"total injected samples          = {n_total_injected_samples:,}")
    print(f"injection variance              = {INJECTION_VARIANCE}")
    print(f"SNR threshold                   = {SENSING_SNR_THRESHOLD_dB} dB")
    print(f"outage quantile                 = {OUTAGE_QUANTILE}")
    print(f"chunk                           = {EVAL_CHUNK}")
    print("====================================================")

    # ================================
    # Build networks
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
    # Find checkpoints
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
    print("[EVAL2] Checkpoint sources")
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
    # Evaluation loop
    # ================================
    reg_feasible_min_rate_list = []
    rob_feasible_min_rate_list = []

    reg_all_min_rate_list = []
    rob_all_min_rate_list = []

    reg_snr_q05_list = []
    rob_snr_q05_list = []

    reg_feasible_mask_list = []
    rob_feasible_mask_list = []

    reg_violation_count_list = []
    rob_violation_count_list = []

    reg_per_channel_vprob_list = []
    rob_per_channel_vprob_list = []

    channel_ids = np.arange(
        SHORTTERM_EST_CHANNELS_PER_LAYOUT,
        dtype=np.int32,
    )

    for local_idx, layout_id in enumerate(eval_layout_ids):
        layout_id = int(layout_id)

        ue_layout = test_dataset["ue_layouts"][layout_id]

        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net,
            ue_layout,
        )

        batch_data = extract_test_batch(
            test_dataset,
            layout_id,
            channel_ids,
        )

        W_C_reg, W_R_reg = compute_shortterm_outputs(
            short_comm_reg,
            short_radar_reg,
            theta_fixed,
            batch_data,
        )

        W_C_rob, W_R_rob = compute_shortterm_outputs(
            short_comm_rob,
            short_radar_rob,
            theta_fixed,
            batch_data,
        )

        reg_result = eval_feasible_minrate_one_model(
            comm_net=short_comm_reg,
            theta_fixed=theta_fixed,
            W_C=W_C_reg,
            W_R=W_R_reg,
            batch_data=batch_data,
            injection_samples=INJECTION_SAMPLES,
            injection_variance=INJECTION_VARIANCE,
            chunk=EVAL_CHUNK,
        )

        rob_result = eval_feasible_minrate_one_model(
            comm_net=short_comm_rob,
            theta_fixed=theta_fixed,
            W_C=W_C_rob,
            W_R=W_R_rob,
            batch_data=batch_data,
            injection_samples=INJECTION_SAMPLES,
            injection_variance=INJECTION_VARIANCE,
            chunk=EVAL_CHUNK,
        )

        reg_feasible_min_rate_list.append(reg_result["feasible_min_rate"])
        rob_feasible_min_rate_list.append(rob_result["feasible_min_rate"])

        reg_all_min_rate_list.append(reg_result["all_min_rate"])
        rob_all_min_rate_list.append(rob_result["all_min_rate"])

        reg_snr_q05_list.append(reg_result["snr_q05"])
        rob_snr_q05_list.append(rob_result["snr_q05"])

        reg_feasible_mask_list.append(reg_result["feasible_mask"])
        rob_feasible_mask_list.append(rob_result["feasible_mask"])

        reg_violation_count_list.append(reg_result["violation_count"])
        rob_violation_count_list.append(rob_result["violation_count"])

        reg_per_channel_vprob_list.append(reg_result["per_channel_vprob"])
        rob_per_channel_vprob_list.append(rob_result["per_channel_vprob"])

        if (local_idx + 1) % 10 == 0 or (local_idx + 1) == n_eval_layouts:
            print(
                f"[EVAL2] progress: "
                f"{local_idx + 1}/{n_eval_layouts} layouts done"
            )

    print("[EVAL2] Concatenating values ...")

    reg_feasible_min_rate_all = np.concatenate(reg_feasible_min_rate_list, axis=0)
    rob_feasible_min_rate_all = np.concatenate(rob_feasible_min_rate_list, axis=0)

    reg_all_min_rate_all = np.concatenate(reg_all_min_rate_list, axis=0)
    rob_all_min_rate_all = np.concatenate(rob_all_min_rate_list, axis=0)

    reg_snr_q05_all = np.concatenate(reg_snr_q05_list, axis=0)
    rob_snr_q05_all = np.concatenate(rob_snr_q05_list, axis=0)

    reg_feasible_mask_all = np.concatenate(reg_feasible_mask_list, axis=0)
    rob_feasible_mask_all = np.concatenate(rob_feasible_mask_list, axis=0)

    reg_violation_count_all = np.concatenate(reg_violation_count_list, axis=0)
    rob_violation_count_all = np.concatenate(rob_violation_count_list, axis=0)

    reg_per_channel_vprob_all = np.concatenate(reg_per_channel_vprob_list, axis=0)
    rob_per_channel_vprob_all = np.concatenate(rob_per_channel_vprob_list, axis=0)

    # ================================
    # Metrics
    # ================================
    reg_feasible_ratio = float(np.mean(reg_feasible_mask_all))
    rob_feasible_ratio = float(np.mean(rob_feasible_mask_all))

    reg_mean_feasible_min_rate = (
        float(np.mean(reg_feasible_min_rate_all))
        if reg_feasible_min_rate_all.size > 0
        else float("nan")
    )

    rob_mean_feasible_min_rate = (
        float(np.mean(rob_feasible_min_rate_all))
        if rob_feasible_min_rate_all.size > 0
        else float("nan")
    )

    reg_sample_violation_count = int(np.sum(reg_violation_count_all))
    rob_sample_violation_count = int(np.sum(rob_violation_count_all))

    reg_sample_violation_prob = reg_sample_violation_count / n_total_injected_samples
    rob_sample_violation_prob = rob_sample_violation_count / n_total_injected_samples

    reg_mean_per_channel_vprob = float(np.mean(reg_per_channel_vprob_all))
    rob_mean_per_channel_vprob = float(np.mean(rob_per_channel_vprob_all))

    reg_snr_q05_mean_db = float(
        10.0 * np.log10(np.maximum(np.mean(reg_snr_q05_all), 1e-12))
    )

    rob_snr_q05_mean_db = float(
        10.0 * np.log10(np.maximum(np.mean(rob_snr_q05_all), 1e-12))
    )

    print("====================================================")
    print("[EVAL2 Results]")
    print(
        f"Evaluation setting: "
        f"{n_eval_layouts} layouts × "
        f"{SHORTTERM_EST_CHANNELS_PER_LAYOUT} estimated channels/layout × "
        f"{INJECTION_SAMPLES} injections/channel"
    )
    print()

    print("Feasible condition:")
    print(
        f"  A layout-channel pair is feasible if "
        f"Q_{OUTAGE_QUANTILE:.2f}(SNR over injections) "
        f">= {SENSING_SNR_THRESHOLD_dB} dB"
    )
    print()

    print("[REG]")
    print(f"  feasible layout-channel pairs       = {reg_feasible_min_rate_all.size:,} / {n_total_layout_channel_pairs:,}")
    print(f"  layout-channel feasible ratio       = {reg_feasible_ratio * 100: .3f} %")
    print(f"  sample-level violation count        = {reg_sample_violation_count:,} / {n_total_injected_samples:,}")
    print(f"  sample-level violation probability  = {reg_sample_violation_prob * 100: .3f} %")
    print(f"  mean per-channel violation prob     = {reg_mean_per_channel_vprob * 100: .3f} %")
    print(f"  mean feasible min injected SumRate  = {reg_mean_feasible_min_rate: .6f} bits/s/Hz")
    print(f"  mean 5%-SNR                         = {reg_snr_q05_mean_db: .3f} dB")
    print()

    print("[ROB]")
    print(f"  feasible layout-channel pairs       = {rob_feasible_min_rate_all.size:,} / {n_total_layout_channel_pairs:,}")
    print(f"  layout-channel feasible ratio       = {rob_feasible_ratio * 100: .3f} %")
    print(f"  sample-level violation count        = {rob_sample_violation_count:,} / {n_total_injected_samples:,}")
    print(f"  sample-level violation probability  = {rob_sample_violation_prob * 100: .3f} %")
    print(f"  mean per-channel violation prob     = {rob_mean_per_channel_vprob * 100: .3f} %")
    print(f"  mean feasible min injected SumRate  = {rob_mean_feasible_min_rate: .6f} bits/s/Hz")
    print(f"  mean 5%-SNR                         = {rob_snr_q05_mean_db: .3f} dB")

    print("====================================================")

    # ================================
    # Save metrics
    # ================================
    metrics_path = os.path.join(
        OUTPUT_DIR,
        f"feasible_min_injected_sumrate_metrics_{EVAL_TAG}.npz"
    )

    np.savez(
        metrics_path,
        reg_feasible_min_rate_all=reg_feasible_min_rate_all,
        rob_feasible_min_rate_all=rob_feasible_min_rate_all,
        reg_all_min_rate_all=reg_all_min_rate_all,
        rob_all_min_rate_all=rob_all_min_rate_all,
        reg_snr_q05_all=reg_snr_q05_all,
        rob_snr_q05_all=rob_snr_q05_all,
        reg_feasible_mask_all=reg_feasible_mask_all,
        rob_feasible_mask_all=rob_feasible_mask_all,
        reg_violation_count_all=reg_violation_count_all,
        rob_violation_count_all=rob_violation_count_all,
        reg_per_channel_vprob_all=reg_per_channel_vprob_all,
        rob_per_channel_vprob_all=rob_per_channel_vprob_all,
        reg_sample_violation_count=reg_sample_violation_count,
        rob_sample_violation_count=rob_sample_violation_count,
        reg_sample_violation_prob=reg_sample_violation_prob,
        rob_sample_violation_prob=rob_sample_violation_prob,
    )

    print(f"[EVAL2] Saved metrics: {metrics_path}")

    fig_path = os.path.join(
        OUTPUT_DIR,
        f"CDF_feasibleMinInjectedSumRate_{EVAL_TAG}.jpg",
    )

    save_cdf_plot(
        x_reg=reg_feasible_min_rate_all,
        x_rob=rob_feasible_min_rate_all,
        save_path=fig_path,
        eval_tag=EVAL_TAG,
    )

    print("[EVAL2] Finished.")