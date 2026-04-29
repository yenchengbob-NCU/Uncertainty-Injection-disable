# -*- coding: utf-8 -*-
import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ============================================================
# Evaluation config
# ============================================================
DEFAULT_CHANNELS_PER_LAYOUT = SHORTTERM_EST_CHANNELS_PER_LAYOUT   # 預設用完整 test pool
DEFAULT_INJECTION_SAMPLES   = INJECTION_SAMPLES
DEFAULT_CHUNK               = 50
DEFAULT_MAX_LAYOUTS         = None


# ============================================================
# Helpers
# ============================================================
def reset_eval_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def empirical_cdf(x: np.ndarray):
    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def load_test_dataset(npz_path: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"找不到 test dataset：{npz_path}\n"
            f"請先執行 rician.py 生成固定 dataset。"
        )

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
        for k in required_keys:
            if k not in data:
                raise KeyError(f"test dataset 缺少欄位：{k}")
        dataset = {k: data[k] for k in required_keys}

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))

    print(f"[test] preloaded into RAM: {npz_path}")
    print(f"[test] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[test] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[test] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[test] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[test] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")
    print(f"[test] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")
    return dataset


def load_required_ckpt(net, tag: str):
    if not net.model_path:
        raise ValueError(f"[{tag}] model_path 為空，無法載入 checkpoint。")

    if not os.path.exists(net.model_path):
        raise FileNotFoundError(f"[{tag}] 找不到 checkpoint：{net.model_path}")

    net.load_model(verbose=True)


def get_fixed_theta_from_longterm(longterm_net: LongTermPositionNet, ue_layout_np: np.ndarray):
    longterm_net.eval()
    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)  # (1,K,2)
        theta_lt, _, _ = longterm_net(layout_t)
    return theta_lt.detach()  # (1,N)


def select_channel_ids(n_pool: int, requested: int | None, seed: int) -> np.ndarray:
    if requested is None or requested >= n_pool:
        return np.arange(n_pool)

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_pool, size=requested, replace=False))


def extract_test_batch(dataset, layout_id: int, channel_ids: np.ndarray):
    h_dk_hat = np_to_torch_complex(dataset["st_h_dk_hat"][layout_id][channel_ids])   # (B,M,K)
    h_rk_hat = np_to_torch_complex(dataset["st_h_rk_hat"][layout_id][channel_ids])   # (B,N,K)
    G_hat    = np_to_torch_complex(dataset["st_G_hat"][layout_id][channel_ids])      # (B,N,M)
    g_dt_hat = np_to_torch_complex(dataset["st_g_dt_hat"][layout_id][channel_ids])   # (B,M,1)

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
    theta_fixed: torch.Tensor,   # (1,N)
    batch_data: dict,
):
    comm_net.eval()
    radar_net.eval()

    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    B = h_dk_hat.shape[0]
    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)   # (B,M,K)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)  # (B,M,1)

    tx_power  = comm_net.compute_tx_power(W_C, W_R)                    # (B,)
    tx_excess = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0)  # (B,)

    return W_C, W_R, tx_excess


@torch.no_grad()
def eval_one_model_with_fixed_beams(
    comm_net: ShortTermCommNet,
    theta_fixed: torch.Tensor,   # (1,N)
    W_C: torch.Tensor,           # (B,M,K)
    W_R: torch.Tensor,           # (B,M,1)
    tx_excess: torch.Tensor,     # (B,)
    batch_data: dict,
    injection_samples: int,
    injection_variance: float,
    outage_q: float,
    chunk: int,
    seed: int,
):
    """
    test / evaluate 使用 empirical violation probability:
        violation probability = P(SNR < threshold)
    """
    # 讓 REG 與 ROB 用同一批 injection noise，比較公平
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    q = float(outage_q)

    theta_batch = theta_fixed.expand(B, N)

    sumrate_chunks = []
    snr_chunks = []

    for s0 in range(0, L, chunk):
        s = min(chunk, L - s0)

        # replicate estimated channels: (B,s,...) -> (B*s,...)
        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        # additive injection
        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape, injection_variance, DEVICE, h_dk_rep.dtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape, injection_variance, DEVICE, h_rk_rep.dtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,    injection_variance, DEVICE, G_rep.dtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape, injection_variance, DEVICE, g_dt_rep.dtype)

        # replicate design vars
        theta_rep = theta_batch.unsqueeze(1).expand(B, s, N).reshape(B * s, N)
        W_C_rep   = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        L_R = W_R.shape[2]
        W_R_rep = W_R.unsqueeze(1).expand(B, s, M, L_R).reshape(B * s, M, L_R)

        # comm
        sinrs = comm_net.compute_comm_sinrs(
            h_dk=h_dk_inj,
            h_rk=h_rk_inj,
            G=G_inj,
            theta=theta_rep,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE,
        )  # (B*s,K)
        rates = comm_net.compute_rates(sinrs)
        sum_rate = rates.sum(dim=1)  # (B*s,)

        # sensing
        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real  # (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))
        snr_chunks.append(sense_snr.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)  # (B,L)
    snr_samples     = torch.cat(snr_chunks, dim=1)      # (B,L)

    # test metrics per estimated-channel sample
    sumrate_mean = sumrate_samples.mean(dim=1)           # (B,)
    sumrate_min  = sumrate_samples.min(dim=1).values     # (B,)

    snr_violation_prob = (snr_samples < SENSING_SNR_THRESHOLD).float().mean(dim=1)  # (B,)
    snr_penalty        = torch.clamp(snr_violation_prob - q, min=0.0)                # (B,)

    obj_out = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT * snr_penalty
        - TX_POWER_LOSS_WEIGHT * tx_excess
    )  # (B,)

    return {
        "sumrate_mean": sumrate_mean.detach().cpu().numpy(),
        "sumrate_min": sumrate_min.detach().cpu().numpy(),
        "snr_violation_prob": snr_violation_prob.detach().cpu().numpy(),
        "snr_penalty": snr_penalty.detach().cpu().numpy(),
        "objective": obj_out.detach().cpu().numpy(),
        "tx_excess": tx_excess.detach().cpu().numpy(),
    }


def summarize_metrics(metrics: dict) -> dict:
    return {
        "objective": float(np.mean(metrics["objective"])),
        "mean_sumrate": float(np.mean(metrics["sumrate_mean"])),
        "min_sumrate": float(np.mean(metrics["sumrate_min"])),
        "violation_prob": float(np.mean(metrics["snr_violation_prob"])),
        "tx_excess": float(np.mean(metrics["tx_excess"])),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-timescale TEST evaluation (fixed dataset)")
    parser.add_argument("--max_layouts", type=int, default=DEFAULT_MAX_LAYOUTS)
    parser.add_argument("--channels_per_layout", type=int, default=DEFAULT_CHANNELS_PER_LAYOUT)
    parser.add_argument("--injection_samples", type=int, default=DEFAULT_INJECTION_SAMPLES)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED + 999)
    args = parser.parse_args()

    reset_eval_seed(args.seed)

    # --------------------------------------------------------
    # load fixed TEST dataset
    # --------------------------------------------------------
    test_dataset = load_test_dataset(TEST_DATASET_PATH)

    n_total_layouts = test_dataset["ue_layouts"].shape[0]
    n_layouts = n_total_layouts if args.max_layouts is None else min(args.max_layouts, n_total_layouts)
    eval_layout_ids = np.arange(n_layouts)

    print("====================================================")
    print("[EVAL] Two-timescale TEST evaluation (fixed dataset)")
    print(f"[EVAL] #test layouts          = {n_layouts}")
    print(f"[EVAL] channels per layout   = {args.channels_per_layout}")
    print(f"[EVAL] injection samples     = {args.injection_samples}")
    print(f"[EVAL] injection variance    = {INJECTION_VARIANCE}")
    print(f"[EVAL] outage quantile q     = {OUTAGE_QUANTILE}")
    print(f"[EVAL] chunk                 = {args.chunk}")
    print("====================================================")

    # --------------------------------------------------------
    # load checkpoints
    # --------------------------------------------------------
    longterm_net = LongTermPositionNet(ckpt_kind="longterm").to(DEVICE)

    short_comm_reg  = ShortTermCommNet(ckpt_kind="short_comm").to(DEVICE)
    short_radar_reg = ShortTermRadarNet(ckpt_kind="short_radar").to(DEVICE)

    short_comm_rob  = ShortTermCommNet(ckpt_kind="short_comm_robust").to(DEVICE)
    short_radar_rob = ShortTermRadarNet(ckpt_kind="short_radar_robust").to(DEVICE)

    load_required_ckpt(longterm_net, "LongTerm")
    load_required_ckpt(short_comm_reg, "Short-Reg-Comm")
    load_required_ckpt(short_radar_reg, "Short-Reg-Radar")
    load_required_ckpt(short_comm_rob, "Short-Rob-Comm")
    load_required_ckpt(short_radar_rob, "Short-Rob-Radar")

    # --------------------------------------------------------
    # collect sample-level metrics across all test layouts
    # --------------------------------------------------------
    reg_sumrate_mean_list, rob_sumrate_mean_list = [], []
    reg_sumrate_min_list,  rob_sumrate_min_list  = [], []
    reg_snr_vprob_list,    rob_snr_vprob_list    = [], []
    reg_snr_pen_list,      rob_snr_pen_list      = [], []
    reg_obj_list,          rob_obj_list          = [], []
    reg_tx_list,           rob_tx_list           = [], []
    layout_id_list = []

    # layout-level summary
    reg_layout_obj_list, rob_layout_obj_list = [], []
    reg_layout_mean_sr_list, rob_layout_mean_sr_list = [], []
    reg_layout_min_sr_list,  rob_layout_min_sr_list  = [], []
    reg_layout_vprob_list,   rob_layout_vprob_list   = [], []

    for local_idx, layout_id in enumerate(eval_layout_ids):
        ue_layout = test_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)  # (1,N)

        n_pool = test_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = select_channel_ids(
            n_pool=n_pool,
            requested=args.channels_per_layout,
            seed=args.seed + layout_id
        )

        batch_data = extract_test_batch(test_dataset, layout_id, channel_ids)
        B = len(channel_ids)

        # short-term outputs
        W_C_reg, W_R_reg, tx_excess_reg = compute_shortterm_outputs(
            short_comm_reg, short_radar_reg, theta_fixed, batch_data
        )
        W_C_rob, W_R_rob, tx_excess_rob = compute_shortterm_outputs(
            short_comm_rob, short_radar_rob, theta_fixed, batch_data
        )

        # same injection noise for REG / ROB on this layout
        layout_seed = args.seed + 100000 + layout_id

        reg_metrics = eval_one_model_with_fixed_beams(
            comm_net=short_comm_reg,
            theta_fixed=theta_fixed,
            W_C=W_C_reg,
            W_R=W_R_reg,
            tx_excess=tx_excess_reg,
            batch_data=batch_data,
            injection_samples=args.injection_samples,
            injection_variance=INJECTION_VARIANCE,
            outage_q=OUTAGE_QUANTILE,
            chunk=args.chunk,
            seed=layout_seed,
        )

        rob_metrics = eval_one_model_with_fixed_beams(
            comm_net=short_comm_rob,
            theta_fixed=theta_fixed,
            W_C=W_C_rob,
            W_R=W_R_rob,
            tx_excess=tx_excess_rob,
            batch_data=batch_data,
            injection_samples=args.injection_samples,
            injection_variance=INJECTION_VARIANCE,
            outage_q=OUTAGE_QUANTILE,
            chunk=args.chunk,
            seed=layout_seed,   # same noise as REG
        )

        # sample-level append
        reg_sumrate_mean_list.append(reg_metrics["sumrate_mean"])
        rob_sumrate_mean_list.append(rob_metrics["sumrate_mean"])

        reg_sumrate_min_list.append(reg_metrics["sumrate_min"])
        rob_sumrate_min_list.append(rob_metrics["sumrate_min"])

        reg_snr_vprob_list.append(reg_metrics["snr_violation_prob"])
        rob_snr_vprob_list.append(rob_metrics["snr_violation_prob"])

        reg_snr_pen_list.append(reg_metrics["snr_penalty"])
        rob_snr_pen_list.append(rob_metrics["snr_penalty"])

        reg_obj_list.append(reg_metrics["objective"])
        rob_obj_list.append(rob_metrics["objective"])

        reg_tx_list.append(reg_metrics["tx_excess"])
        rob_tx_list.append(rob_metrics["tx_excess"])

        layout_id_list.append(np.full(B, local_idx, dtype=np.int32))

        # layout-level summary
        reg_layout_summary = summarize_metrics(reg_metrics)
        rob_layout_summary = summarize_metrics(rob_metrics)

        reg_layout_obj_list.append(reg_layout_summary["objective"])
        rob_layout_obj_list.append(rob_layout_summary["objective"])

        reg_layout_mean_sr_list.append(reg_layout_summary["mean_sumrate"])
        rob_layout_mean_sr_list.append(rob_layout_summary["mean_sumrate"])

        reg_layout_min_sr_list.append(reg_layout_summary["min_sumrate"])
        rob_layout_min_sr_list.append(rob_layout_summary["min_sumrate"])

        reg_layout_vprob_list.append(reg_layout_summary["violation_prob"])
        rob_layout_vprob_list.append(rob_layout_summary["violation_prob"])

        print(f"[EVAL] layout {local_idx + 1}/{n_layouts} done.")

    # --------------------------------------------------------
    # concat sample-level metrics
    # --------------------------------------------------------
    reg_sumrate_mean_all = np.concatenate(reg_sumrate_mean_list, axis=0)
    rob_sumrate_mean_all = np.concatenate(rob_sumrate_mean_list, axis=0)

    reg_sumrate_min_all = np.concatenate(reg_sumrate_min_list, axis=0)
    rob_sumrate_min_all = np.concatenate(rob_sumrate_min_list, axis=0)

    reg_snr_vprob_all = np.concatenate(reg_snr_vprob_list, axis=0)
    rob_snr_vprob_all = np.concatenate(rob_snr_vprob_list, axis=0)

    reg_snr_pen_all = np.concatenate(reg_snr_pen_list, axis=0)
    rob_snr_pen_all = np.concatenate(rob_snr_pen_list, axis=0)

    reg_obj_all = np.concatenate(reg_obj_list, axis=0)
    rob_obj_all = np.concatenate(rob_obj_list, axis=0)

    reg_tx_all = np.concatenate(reg_tx_list, axis=0)
    rob_tx_all = np.concatenate(rob_tx_list, axis=0)

    layout_id_all = np.concatenate(layout_id_list, axis=0)

    # --------------------------------------------------------
    # overall summary
    # --------------------------------------------------------
    reg_mean_sumrate = float(np.mean(reg_sumrate_mean_all))
    rob_mean_sumrate = float(np.mean(rob_sumrate_mean_all))

    reg_min_sumrate = float(np.mean(reg_sumrate_min_all))
    rob_min_sumrate = float(np.mean(rob_sumrate_min_all))

    reg_mean_vprob = float(np.mean(reg_snr_vprob_all))
    rob_mean_vprob = float(np.mean(rob_snr_vprob_all))

    reg_mean_obj = float(np.mean(reg_obj_all))
    rob_mean_obj = float(np.mean(rob_obj_all))

    print("====================================================")
    print(f"[Metric A] Mean E[SumRate] (over {args.injection_samples} injections):")
    print(f"  REG: {reg_mean_sumrate:.6f} bits/s/Hz")
    print(f"  ROB: {rob_mean_sumrate:.6f} bits/s/Hz")

    print(f"[Metric B] Mean min SumRate (over {args.injection_samples} injections):")
    print(f"  REG: {reg_min_sumrate:.6f} bits/s/Hz")
    print(f"  ROB: {rob_min_sumrate:.6f} bits/s/Hz")

    print(f"[Metric C] Mean empirical P(SNR < thr={SENSING_SNR_THRESHOLD_dB} dB):")
    print(f"  REG: {reg_mean_vprob * 100:.3f}%")
    print(f"  ROB: {rob_mean_vprob * 100:.3f}%")

    print(f"[Metric D] Mean objective:")
    print(f"  REG: {reg_mean_obj:.6f}")
    print(f"  ROB: {rob_mean_obj:.6f}")
    print("====================================================")

    # --------------------------------------------------------
    # save metrics
    # --------------------------------------------------------
    eval_dir = os.path.join(PROJECT_DIR, "eval_figures")
    os.makedirs(eval_dir, exist_ok=True)

    eval_tag = (
        f"test_"
        f"L{n_layouts}_"
        f"C{args.channels_per_layout}_"
        f"INJ{args.injection_samples}"
    )

    metrics_path = os.path.join(eval_dir, f"twotimescale_test_metrics_{eval_tag}.npz")
    np.savez(
        metrics_path,
        reg_sumrate_mean_all=reg_sumrate_mean_all,
        rob_sumrate_mean_all=rob_sumrate_mean_all,
        reg_sumrate_min_all=reg_sumrate_min_all,
        rob_sumrate_min_all=rob_sumrate_min_all,
        reg_snr_vprob_all=reg_snr_vprob_all,
        rob_snr_vprob_all=rob_snr_vprob_all,
        reg_snr_pen_all=reg_snr_pen_all,
        rob_snr_pen_all=rob_snr_pen_all,
        reg_obj_all=reg_obj_all,
        rob_obj_all=rob_obj_all,
        reg_tx_all=reg_tx_all,
        rob_tx_all=rob_tx_all,
        layout_id_all=layout_id_all,
        reg_layout_obj=np.array(reg_layout_obj_list, dtype=np.float32),
        rob_layout_obj=np.array(rob_layout_obj_list, dtype=np.float32),
        reg_layout_mean_sr=np.array(reg_layout_mean_sr_list, dtype=np.float32),
        rob_layout_mean_sr=np.array(rob_layout_mean_sr_list, dtype=np.float32),
        reg_layout_min_sr=np.array(reg_layout_min_sr_list, dtype=np.float32),
        rob_layout_min_sr=np.array(rob_layout_min_sr_list, dtype=np.float32),
        reg_layout_vprob=np.array(reg_layout_vprob_list, dtype=np.float32),
        rob_layout_vprob=np.array(rob_layout_vprob_list, dtype=np.float32),
    )
    print(f"[EVAL] Saved metrics: {metrics_path}")

    # --------------------------------------------------------
    # CDF plots
    # --------------------------------------------------------
    # 1) CDF of E[SumRate]
    x_sr_reg, y_sr_reg = empirical_cdf(reg_sumrate_mean_all)
    x_sr_rob, y_sr_rob = empirical_cdf(rob_sumrate_mean_all)

    plt.figure()
    plt.plot(x_sr_reg, y_sr_reg, label="REG: E[SumRate]")
    plt.plot(x_sr_rob, y_sr_rob, label="ROB: E[SumRate]")
    plt.xlabel("E[SumRate] over injections (bits/s/Hz)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"Two-timescale TEST CDF of E[SumRate] — {eval_tag}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"CDF_meanSumRate_{eval_tag}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # 2) CDF of min SumRate
    x_min_reg, y_min_reg = empirical_cdf(reg_sumrate_min_all)
    x_min_rob, y_min_rob = empirical_cdf(rob_sumrate_min_all)

    plt.figure()
    plt.plot(x_min_reg, y_min_reg, label="REG: min SumRate")
    plt.plot(x_min_rob, y_min_rob, label="ROB: min SumRate")
    plt.xlabel("min SumRate over injections (bits/s/Hz)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"Two-timescale TEST CDF of min SumRate — {eval_tag}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"CDF_minSumRate_{eval_tag}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # 3) CDF of empirical violation probability
    x_vp_reg, y_vp_reg = empirical_cdf(reg_snr_vprob_all)
    x_vp_rob, y_vp_rob = empirical_cdf(rob_snr_vprob_all)

    plt.figure()
    plt.plot(x_vp_reg, y_vp_reg, label="REG: P(SNR<thr)")
    plt.plot(x_vp_rob, y_vp_rob, label="ROB: P(SNR<thr)")
    plt.xlabel("Empirical violation probability")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"Two-timescale TEST CDF of P(SNR<thr) — {eval_tag}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, f"CDF_snrViolationProb_{eval_tag}.jpg"), format="jpg")
    plt.show()
    plt.close()

    print("[EVAL] Finished.")