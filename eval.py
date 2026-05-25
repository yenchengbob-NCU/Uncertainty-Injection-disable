# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ================================
# Evaluation config
# ================================
EVAL_CHUNK = 50


# ================================
# Helpers
# ================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def empirical_cdf(x: np.ndarray):
    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1, dtype=np.float64) / len(x_sorted)
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


def load_test_dataset(npz_path: str):
    """
    載入 test dataset。

    這版 eval.py 假設 rician.py 已正確產生 dataset,
    因此只讀取必要欄位，不做額外檢查。
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
    給定一組 layout,從 long-term net 取得固定 theta_LT。

    回傳:
        theta_lt : torch.Tensor, shape = (1, RIS_UNIT)
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
    )                                                               # shape = (B,M,K)

    h_rk_hat = np_to_torch_complex(
        dataset["st_h_rk_hat"][layout_id][channel_ids]
    )                                                               # shape = (B,N,K)

    G_hat = np_to_torch_complex(
        dataset["st_G_hat"][layout_id][channel_ids]
    )                                                               # shape = (B,N,M)

    g_dt_hat = np_to_torch_complex(
        dataset["st_g_dt_hat"][layout_id][channel_ids]
    )                                                               # shape = (B,M,1)

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

    回傳:
        W_C : torch.Tensor, shape = (B,M,K)
        W_R : torch.Tensor, shape = (B,M,RADAR_STREAMS)
    """
    comm_net.eval()
    radar_net.eval()

    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat    = batch_data["G_hat"]
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
def eval_four_values_one_model(
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
    對單一 model,在 fixed beams 下計算 4 種 value。

    對單一 layout 回傳：

        1. sumrate_raw
           shape = (B * L,)
           每一筆是 injected SumRate。

        2. worst_injection_sumrate
           shape = (B,)
           每一筆 estimated channel 先從 L 個 injections 中取最差 SumRate。

        3. snr_raw
           shape = (B * L,)
           每一筆是 injected sensing SNR，linear scale。

        4. snr_violation_prob
           shape = (B,)
           每一筆 estimated channel 用 L 個 injections 計算 SNR violation probability。

    其中：
        B = estimated channels per layout
        L = injection samples per estimated channel
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
    L = int(injection_samples)

    theta_batch = theta_fixed.expand(B, N)

    sumrate_chunks = []
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
        sum_rate = rates.sum(dim=1)                                    # shape = (B*s,)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real                                                         # shape = (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))
        snr_chunks.append(sense_snr.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)                 # shape = (B,L)
    snr_samples = torch.cat(snr_chunks, dim=1)                         # shape = (B,L)

    sumrate_raw = sumrate_samples.reshape(-1)                          # shape = (B*L,)
    worst_injection_sumrate = sumrate_samples.min(dim=1).values        # shape = (B,)

    snr_raw = snr_samples.reshape(-1)                                  # shape = (B*L,)

    snr_violation_prob = (
        snr_samples < SENSING_SNR_THRESHOLD
    ).float().mean(dim=1)                                              # shape = (B,)

    return {
        "sumrate_raw": sumrate_raw.detach().cpu().numpy().astype(np.float32),
        "worst_injection_sumrate": worst_injection_sumrate.detach().cpu().numpy().astype(np.float32),
        "snr_raw": snr_raw.detach().cpu().numpy().astype(np.float32),
        "snr_violation_prob": snr_violation_prob.detach().cpu().numpy().astype(np.float32),
    }


def save_cdf_plot(
    x_reg: np.ndarray,
    x_rob: np.ndarray,
    label_reg: str,
    label_rob: str,
    xlabel: str,
    title: str,
    save_path: str,
    vline_x=None,
    vline_label=None,
):
    """
    儲存 REG / ROB CDF 圖。
    """
    x_reg_cdf, y_reg_cdf = empirical_cdf(x_reg)
    x_rob_cdf, y_rob_cdf = empirical_cdf(x_rob)

    plt.figure(figsize=(8, 5))
    plt.plot(x_reg_cdf, y_reg_cdf, label=label_reg)
    plt.plot(x_rob_cdf, y_rob_cdf, label=label_rob)

    if vline_x is not None:
        plt.axvline(
            x=vline_x,
            linestyle="--",
            linewidth=1.5,
            label=vline_label,
        )

    plt.xlabel(xlabel)
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, format="jpg", dpi=300)
    print(f"[EVAL] Saved figure: {save_path}")
    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    test_dataset = load_test_dataset(TEST_DATASET_PATH)

    n_layouts = test_dataset["ue_layouts"].shape[0]
    eval_layout_ids = np.arange(n_layouts, dtype=np.int32)
    n_eval_layouts = len(eval_layout_ids)

    n_injected_samples = (
        n_eval_layouts
        * SHORTTERM_EST_CHANNELS_PER_LAYOUT
        * INJECTION_SAMPLES
    )

    n_worst_sumrate_samples = (
        n_eval_layouts
        * SHORTTERM_EST_CHANNELS_PER_LAYOUT
    )

    n_vprob_samples = (
        n_eval_layouts
        * SHORTTERM_EST_CHANNELS_PER_LAYOUT
    )

    print("====================================================")
    print("[EVAL] Four-value TEST evaluation")
    print(f"[EVAL] layouts                         = {n_eval_layouts}")
    print(f"[EVAL] estimated channels/layout       = {SHORTTERM_EST_CHANNELS_PER_LAYOUT}")
    print(f"[EVAL] injections/estimated channel    = {INJECTION_SAMPLES}")
    print(f"[EVAL] injection variance              = {INJECTION_VARIANCE}")
    print(f"[EVAL] SNR threshold                   = {SENSING_SNR_THRESHOLD_dB} dB")
    print(f"[EVAL] chunk                           = {EVAL_CHUNK}")
    print("----------------------------------------------------")
    print(f"[EVAL] SumRate CDF samples             = {n_injected_samples:,}")
    print(f"[EVAL] SNR CDF samples                 = {n_injected_samples:,}")
    print(f"[EVAL] Worst-injection SumRate samples = {n_worst_sumrate_samples:,}")
    print(f"[EVAL] SNR violation prob samples      = {n_vprob_samples:,}")
    print("====================================================")

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

    longterm_net.load_model(verbose=True)
    short_comm_reg.load_model(verbose=True)
    short_radar_reg.load_model(verbose=True)
    short_comm_rob.load_model(verbose=True)
    short_radar_rob.load_model(verbose=True)

    reg_sumrate_raw_list = []
    rob_sumrate_raw_list = []

    reg_worst_sumrate_list = []
    rob_worst_sumrate_list = []

    reg_snr_raw_list = []
    rob_snr_raw_list = []

    reg_snr_vprob_list = []
    rob_snr_vprob_list = []

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

        reg_values = eval_four_values_one_model(
            comm_net=short_comm_reg,
            theta_fixed=theta_fixed,
            W_C=W_C_reg,
            W_R=W_R_reg,
            batch_data=batch_data,
            injection_samples=INJECTION_SAMPLES,
            injection_variance=INJECTION_VARIANCE,
            chunk=EVAL_CHUNK,
        )

        rob_values = eval_four_values_one_model(
            comm_net=short_comm_rob,
            theta_fixed=theta_fixed,
            W_C=W_C_rob,
            W_R=W_R_rob,
            batch_data=batch_data,
            injection_samples=INJECTION_SAMPLES,
            injection_variance=INJECTION_VARIANCE,
            chunk=EVAL_CHUNK,
        )

        reg_sumrate_raw_list.append(reg_values["sumrate_raw"])
        rob_sumrate_raw_list.append(rob_values["sumrate_raw"])

        reg_worst_sumrate_list.append(reg_values["worst_injection_sumrate"])
        rob_worst_sumrate_list.append(rob_values["worst_injection_sumrate"])

        reg_snr_raw_list.append(reg_values["snr_raw"])
        rob_snr_raw_list.append(rob_values["snr_raw"])

        reg_snr_vprob_list.append(reg_values["snr_violation_prob"])
        rob_snr_vprob_list.append(rob_values["snr_violation_prob"])

        if (local_idx + 1) % 10 == 0 or (local_idx + 1) == n_eval_layouts:
            print(f"[EVAL] progress: {local_idx + 1}/{n_eval_layouts} layouts done")

    print("[EVAL] Concatenating values ...")

    reg_sumrate_raw_all = np.concatenate(reg_sumrate_raw_list, axis=0)
    rob_sumrate_raw_all = np.concatenate(rob_sumrate_raw_list, axis=0)

    reg_worst_sumrate_all = np.concatenate(reg_worst_sumrate_list, axis=0)
    rob_worst_sumrate_all = np.concatenate(rob_worst_sumrate_list, axis=0)

    reg_snr_raw_all = np.concatenate(reg_snr_raw_list, axis=0)
    rob_snr_raw_all = np.concatenate(rob_snr_raw_list, axis=0)

    reg_snr_vprob_all = np.concatenate(reg_snr_vprob_list, axis=0)
    rob_snr_vprob_all = np.concatenate(rob_snr_vprob_list, axis=0)

    # Metric A: SumRate averaged over layouts, estimated channels, and injections
    reg_mean_sumrate = float(np.mean(reg_sumrate_raw_all))
    rob_mean_sumrate = float(np.mean(rob_sumrate_raw_all))

    # Metric B: for each estimated channel, pick the worst SumRate among injections, then average
    reg_mean_worst_sumrate = float(np.mean(reg_worst_sumrate_all))
    rob_mean_worst_sumrate = float(np.mean(rob_worst_sumrate_all))

    # Metric C: linear SNR averaged over layouts, estimated channels, and injections, then converted to dB
    reg_mean_snr_db = float(
        10.0 * np.log10(np.maximum(np.mean(reg_snr_raw_all), 1e-12))
    )

    rob_mean_snr_db = float(
        10.0 * np.log10(np.maximum(np.mean(rob_snr_raw_all), 1e-12))
    )

    # Metric D: for each estimated channel, compute violation probability over injections, then average
    reg_mean_vprob = float(np.mean(reg_snr_vprob_all))
    rob_mean_vprob = float(np.mean(rob_snr_vprob_all))

    # Metric E: count violations over all injected SNR samples
    reg_total_violation_count = int(np.sum(reg_snr_raw_all < SENSING_SNR_THRESHOLD))
    rob_total_violation_count = int(np.sum(rob_snr_raw_all < SENSING_SNR_THRESHOLD))

    reg_total_vprob = float(reg_total_violation_count / reg_snr_raw_all.size)
    rob_total_vprob = float(rob_total_violation_count / rob_snr_raw_all.size)

    print("====================================================")
    print("[EVAL Results]")
    print(
        f"Evaluation setting: "
        f"{n_eval_layouts} layouts × "
        f"{SHORTTERM_EST_CHANNELS_PER_LAYOUT} estimated channels/layout × "
        f"{INJECTION_SAMPLES} injections/channel"
    )
    print()

    print("Metric A: SumRate")
    print(f"  REG = {reg_mean_sumrate: .6f} bits/s/Hz")
    print(f"  ROB = {rob_mean_sumrate: .6f} bits/s/Hz")
    print()

    print("Metric B: Mean worst-injection SumRate")
    print(f"  REG = {reg_mean_worst_sumrate: .6f} bits/s/Hz")
    print(f"  ROB = {rob_mean_worst_sumrate: .6f} bits/s/Hz")
    print()

    print("Metric C: SNR")
    print(f"  REG = {reg_mean_snr_db: .3f} dB")
    print(f"  ROB = {rob_mean_snr_db: .3f} dB")
    print()

    print(f"Metric D: SNR violation probability, P(SNR < {SENSING_SNR_THRESHOLD_dB} dB)")
    print(f"  REG = {reg_mean_vprob * 100: .3f} %")
    print(f"  ROB = {rob_mean_vprob * 100: .3f} %")
    print()

    print(f"Metric E: Total SNR violation probability, P(SNR < {SENSING_SNR_THRESHOLD_dB} dB)")
    print(f"  REG = {reg_total_vprob * 100: .3f} %  ({reg_total_violation_count:,} / {reg_snr_raw_all.size:,})")
    print(f"  ROB = {rob_total_vprob * 100: .3f} %  ({rob_total_violation_count:,} / {rob_snr_raw_all.size:,})")
    print("====================================================")

    eval_dir = os.path.join(PROJECT_DIR, "eval_figures")
    os.makedirs(eval_dir, exist_ok=True)

    eval_tag = (
        f"test_"
        f"L{n_layouts}_"
        f"C{SHORTTERM_EST_CHANNELS_PER_LAYOUT}_"
        f"INJ{INJECTION_SAMPLES}"
    )

    metrics_path = os.path.join(
        eval_dir,
        f"four_values_metrics_{eval_tag}.npz"
    )

    np.savez(
        metrics_path,
        reg_sumrate_raw_all=reg_sumrate_raw_all,
        rob_sumrate_raw_all=rob_sumrate_raw_all,
        reg_worst_sumrate_all=reg_worst_sumrate_all,
        rob_worst_sumrate_all=rob_worst_sumrate_all,
        reg_snr_raw_all=reg_snr_raw_all,
        rob_snr_raw_all=rob_snr_raw_all,
        reg_snr_vprob_all=reg_snr_vprob_all,
        rob_snr_vprob_all=rob_snr_vprob_all,
        reg_total_violation_count=reg_total_violation_count,
        rob_total_violation_count=rob_total_violation_count,
        reg_total_vprob=reg_total_vprob,
        rob_total_vprob=rob_total_vprob,
    )

    print(f"[EVAL] Saved metrics: {metrics_path}")

    # ================================
    # CDF 1: SumRate
    # 50 × 500 × 200 injected SumRate samples
    # ================================
    save_cdf_plot(
        x_reg=reg_sumrate_raw_all,
        x_rob=rob_sumrate_raw_all,
        label_reg="REG: SumRate",
        label_rob="ROB: SumRate",
        xlabel="Injected SumRate sample (bits/s/Hz)",
        title=f"CDF of injected SumRate samples — {eval_tag}",
        save_path=os.path.join(
            eval_dir,
            f"CDF_SumRate_{eval_tag}.jpg",
        ),
    )

    # ================================
    # CDF 2: SNR
    # 50 × 500 × 200 injected SNR samples
    # ================================
    reg_snr_db_all = 10.0 * np.log10(np.maximum(reg_snr_raw_all, 1e-12))
    rob_snr_db_all = 10.0 * np.log10(np.maximum(rob_snr_raw_all, 1e-12))

    save_cdf_plot(
        x_reg=reg_snr_db_all,
        x_rob=rob_snr_db_all,
        label_reg="REG: SNR",
        label_rob="ROB: SNR",
        xlabel="Injected SNR sample (dB)",
        title=f"CDF of injected SNR samples — {eval_tag}",
        save_path=os.path.join(
            eval_dir,
            f"CDF_SNR_{eval_tag}.jpg",
        ),
        vline_x=SENSING_SNR_THRESHOLD_dB,
        vline_label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB",
    )

    print("[EVAL] Finished.")