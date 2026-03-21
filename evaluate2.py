# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import *
from rician import large_scale_fading


# ============================================================
# Helpers
# ============================================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    """numpy complex -> torch.complex64 on DEVICE"""
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


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


@torch.no_grad()
def eval_metrics_mean_sumrate_var_snr(
    comm_net, sense_net, ris_net,             # reg 或 rob
    h_dk_hat: torch.Tensor,                   # (B,M,K) estimated
    h_rk_hat: torch.Tensor,                   # (B,N,K) estimated
    G_hat:    torch.Tensor,                   # (B,N,M) estimated
    g_dt_hat: torch.Tensor,                   # (B,M,1) estimated
    pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS,     # large-scale fading (power)
    injection_samples: int,                   # N_VAL (=2000)
    injection_variance: float,                # 0.075
    outage_q: float,                          # 0.05
    chunk: int = 100
):
    """
    對齊圖片公式評估：
      E[SumRate] - λ * max(thr - VaR_q(SNR), 0)  (再扣 phi/tx penalty)

    回傳 (per-sample):
      sumrate_mean:        (B,) = mean over injections
      sumrate_min:         (B,) = min over injections
      snr_var:             (B,) = VaR_q(SNR)  (kth smallest)
      snr_penalty:         (B,) = max(thr - snr_var, 0)
      snr_violation_prob:  (B,) = empirical P(SNR < thr) over injections
      obj_out:             (B,) = sumrate_mean - λsnr*snr_penalty - λphi*phi_penalty - λp*tx_excess
      phi_penalty:         (B,) = mean(max(|phi|-1,0)) over N
      tx_excess:           (B,) = max(Ptx - Pmax, 0)
    """
    comm_net.eval()
    sense_net.eval()
    ris_net.eval()

    B, M, K = h_dk_hat.shape
    N = h_rk_hat.shape[1]
    L = int(injection_samples)
    q = float(outage_q)

    # ----------------------------
    # (A) Design variables from estimated channels (no injection)
    # ----------------------------
    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)   # (B,M,K)
    W_S = sense_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)  # (B,M,1)
    phi = ris_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)    # (B,N)

    # penalty #1: RIS amplitude (per-sample)
    phi_excess = torch.clamp(phi.abs() - 1.0, min=0.0)     # (B,N)
    phi_penalty = phi_excess.mean(dim=1)                   # (B,)

    # penalty #2: TX power excess (per-sample)
    power_comm  = (W_C.abs() ** 2).sum(dim=(1, 2))         # (B,)
    power_sense = (W_S.abs() ** 2).sum(dim=(1, 2))         # (B,)
    tx_power = power_comm + power_sense                    # (B,)
    tx_excess = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0)  # (B,)

    # ----------------------------
    # (B) Injected-channel evaluation: collect (B,L) samples
    # ----------------------------
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
        cdtype = h_dk_rep.dtype
        device = h_dk_rep.device
        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape, injection_variance, device, cdtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape, injection_variance, device, cdtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,    injection_variance, device, cdtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape, injection_variance, device, cdtype)

        # replicate design vars to match (B*s,...)
        W_C_rep = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        W_S_rep = W_S.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)
        phi_rep = phi.unsqueeze(1).expand(B, s, N).reshape(B * s, N)

        # sum-rate on injected channels
        sinrs = comm_net.compute_comm_sinrs(
            h_dk_inj, h_rk_inj, G_inj, phi_rep, W_S_rep, W_C_rep,
            pl_BS_UE, pl_BS_RIS_UE,
        )  # (B*s,K)
        rates = comm_net.compute_rates(sinrs)              # (B*s,K)
        sum_rate = rates.sum(dim=1)                        # (B*s,)

        # sensing SNR on injected channels
        sense_snr = comm_net.compute_sense_snr(
            g_dt_inj, W_S_rep, W_C_rep, pl_BS_TAR_BS
        ).real  # (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))      # (B,s)
        snr_chunks.append(sense_snr.reshape(B, s))         # (B,s)

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)     # (B,L)
    snr_samples     = torch.cat(snr_chunks,     dim=1)     # (B,L)

    # ----------------------------
    # (C) Metrics per sample (B,)
    # ----------------------------
    sumrate_mean = sumrate_samples.mean(dim=1)             # (B,) E[SumRate] approx
    sumrate_min  = sumrate_samples.min(dim=1).values       # (B,) min rate

    k = max(1, int(math.ceil(q * L)))                      # L=2000,q=0.05 -> k=100
    snr_var, _ = torch.kthvalue(snr_samples, k=k, dim=1)   # (B,) VaR_q(SNR)
    snr_penalty = torch.clamp(SENSING_SNR_THRESHOLD - snr_var, min=0.0)  # (B,)

    # 新增：每條 sample 在 L 次 injection 中違反 sensing threshold 的機率
    snr_violation_prob = (snr_samples < SENSING_SNR_THRESHOLD).to(torch.float32).mean(dim=1)  # (B,)

    # ----------------------------
    # (D) Objective per sample (B,)
    # ----------------------------
    obj_out = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT  * snr_penalty
        - RE_POWER_LOSS_WEIGHT * phi_penalty
        - TX_POWER_LOSS_WEIGHT * tx_excess
    )  # (B,)

    return (
        sumrate_mean,
        sumrate_min,
        snr_var,
        snr_penalty,
        snr_violation_prob,
        obj_out,
        phi_penalty,
        tx_excess,
    )


# ============================================================
# Main evaluation
# ============================================================
if __name__ == "__main__":

    # ----------------------------
    # 跑code用的變數
    # ----------------------------
    BATCH = 200
    CHUNK = 100

    # ----------------------------
    # 1) Load estimated test channels (N_TEST=4000)
    # ----------------------------
    npz_path = TEST_NPZ_PATH

    data = np.load(npz_path)
    h_dk_all = np_to_torch_complex(data["h_dk"])   # (N_TEST,M,K)
    h_rk_all = np_to_torch_complex(data["h_rk"])   # (N_TEST,N,K)
    G_all    = np_to_torch_complex(data["G"])      # (N_TEST,N,M)
    g_dt_all = np_to_torch_complex(data["g_dt"])   # (N_TEST,M,1)

    n_test = h_dk_all.shape[0]
    print(f"[EVAL] Loaded estimated channels: {npz_path}")
    print(f"[EVAL] N_TEST={n_test}, M={TX_ANT}, N={RIS_UNIT}, K={UAV_COMM}")
    print(f"[EVAL] N_VAL={N_VAL}, q={OUTAGE_QUANTILE}, inj_var={INJECTION_VARIANCE}, BATCH={BATCH}, CHUNK={CHUNK}")

    # ----------------------------
    # 2) Load large-scale fading (power)
    # ----------------------------
    pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS = large_scale_fading()

    # ----------------------------
    # 3) Load 6 checkpoints (regular + robust)
    # ----------------------------
    ckpt_dir = CKPT_DIR

    reg_comm_ckpt = os.path.join(ckpt_dir, f"comm_{SETTING_STRING}.ckpt")
    reg_sens_ckpt = os.path.join(ckpt_dir, f"sense_{SETTING_STRING}.ckpt")
    reg_ris_ckpt  = os.path.join(ckpt_dir, f"ris_{SETTING_STRING}.ckpt")

    rob_comm_ckpt = os.path.join(ckpt_dir, f"comm_robust_{SETTING_STRING}.ckpt")
    rob_sens_ckpt = os.path.join(ckpt_dir, f"sense_robust_{SETTING_STRING}.ckpt")
    rob_ris_ckpt  = os.path.join(ckpt_dir, f"ris_robust_{SETTING_STRING}.ckpt")

    for p in [reg_comm_ckpt, reg_sens_ckpt, reg_ris_ckpt, rob_comm_ckpt, rob_sens_ckpt, rob_ris_ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到 checkpoint:{p}")

    reg_comm = CommBeamformerNet().to(DEVICE)
    reg_sens = SenseBeamformerNet().to(DEVICE)
    reg_ris  = RISPhaseNet().to(DEVICE)
    reg_comm.load_state_dict(torch.load(reg_comm_ckpt, map_location=DEVICE), strict=True)
    reg_sens.load_state_dict(torch.load(reg_sens_ckpt, map_location=DEVICE), strict=True)
    reg_ris.load_state_dict(torch.load(reg_ris_ckpt,  map_location=DEVICE), strict=True)

    rob_comm = RobustCommBeamformerNet().to(DEVICE)
    rob_sens = RobustSenseBeamformerNet().to(DEVICE)
    rob_ris  = RobustRISPhaseNet().to(DEVICE)
    rob_comm.load_state_dict(torch.load(rob_comm_ckpt, map_location=DEVICE), strict=True)
    rob_sens.load_state_dict(torch.load(rob_sens_ckpt, map_location=DEVICE), strict=True)
    rob_ris.load_state_dict(torch.load(rob_ris_ckpt,  map_location=DEVICE), strict=True)

    print("[EVAL] Loaded 6 checkpoints OK.")

    # ----------------------------
    # 4) Evaluate
    # ----------------------------
    torch.manual_seed(RANDOM_SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    reg_sumrate_mean_list, rob_sumrate_mean_list = [], []
    reg_sumrate_min_list,  rob_sumrate_min_list  = [], []
    reg_snr_var_list,      rob_snr_var_list      = [], []
    reg_snr_pen_list,      rob_snr_pen_list      = [], []
    reg_snr_vprob_list,    rob_snr_vprob_list    = [], []   # <-- 新增
    reg_obj_list,          rob_obj_list          = [], []
    reg_phi_list,          rob_phi_list          = [], []
    reg_tx_list,           rob_tx_list           = [], []

    for i0 in range(0, n_test, BATCH):
        i1 = min(i0 + BATCH, n_test)

        h_dk = h_dk_all[i0:i1]
        h_rk = h_rk_all[i0:i1]
        G    = G_all[i0:i1]
        g_dt = g_dt_all[i0:i1]

        # Regular
        (
            reg_sumrate_mean,
            reg_sumrate_min,
            reg_snr_var,
            reg_snr_pen,
            reg_snr_vprob,
            reg_obj,
            reg_phi,
            reg_tx,
        ) = eval_metrics_mean_sumrate_var_snr(
            reg_comm, reg_sens, reg_ris,
            h_dk, h_rk, G, g_dt,
            pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS,
            injection_samples=N_VAL,
            injection_variance=INJECTION_VARIANCE,
            outage_q=OUTAGE_QUANTILE,
            chunk=CHUNK
        )

        # Robust
        (
            rob_sumrate_mean,
            rob_sumrate_min,
            rob_snr_var,
            rob_snr_pen,
            rob_snr_vprob,
            rob_obj,
            rob_phi,
            rob_tx,
        ) = eval_metrics_mean_sumrate_var_snr(
            rob_comm, rob_sens, rob_ris,
            h_dk, h_rk, G, g_dt,
            pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS,
            injection_samples=N_VAL,
            injection_variance=INJECTION_VARIANCE,
            outage_q=OUTAGE_QUANTILE,
            chunk=CHUNK
        )

        reg_sumrate_mean_list.append(reg_sumrate_mean.detach().cpu().numpy())
        rob_sumrate_mean_list.append(rob_sumrate_mean.detach().cpu().numpy())

        reg_sumrate_min_list.append(reg_sumrate_min.detach().cpu().numpy())
        rob_sumrate_min_list.append(rob_sumrate_min.detach().cpu().numpy())

        reg_snr_var_list.append(reg_snr_var.detach().cpu().numpy())
        rob_snr_var_list.append(rob_snr_var.detach().cpu().numpy())

        reg_snr_pen_list.append(reg_snr_pen.detach().cpu().numpy())
        rob_snr_pen_list.append(rob_snr_pen.detach().cpu().numpy())

        reg_snr_vprob_list.append(reg_snr_vprob.detach().cpu().numpy())   # <-- 新增
        rob_snr_vprob_list.append(rob_snr_vprob.detach().cpu().numpy())   # <-- 新增

        reg_obj_list.append(reg_obj.detach().cpu().numpy())
        rob_obj_list.append(rob_obj.detach().cpu().numpy())

        reg_phi_list.append(reg_phi.detach().cpu().numpy())
        rob_phi_list.append(rob_phi.detach().cpu().numpy())

        reg_tx_list.append(reg_tx.detach().cpu().numpy())
        rob_tx_list.append(rob_tx.detach().cpu().numpy())

        print(f"[EVAL] {i1}/{n_test} done.")

    # concat all (N_TEST,)
    reg_sumrate_mean_all = np.concatenate(reg_sumrate_mean_list, axis=0)
    rob_sumrate_mean_all = np.concatenate(rob_sumrate_mean_list, axis=0)

    reg_sumrate_min_all  = np.concatenate(reg_sumrate_min_list,  axis=0)
    rob_sumrate_min_all  = np.concatenate(rob_sumrate_min_list,  axis=0)

    reg_snr_var_all = np.concatenate(reg_snr_var_list, axis=0)
    rob_snr_var_all = np.concatenate(rob_snr_var_list, axis=0)

    reg_snr_pen_all = np.concatenate(reg_snr_pen_list, axis=0)
    rob_snr_pen_all = np.concatenate(rob_snr_pen_list, axis=0)

    reg_snr_vprob_all = np.concatenate(reg_snr_vprob_list, axis=0)   # <-- 新增
    rob_snr_vprob_all = np.concatenate(rob_snr_vprob_list, axis=0)   # <-- 新增

    reg_obj_all = np.concatenate(reg_obj_list, axis=0)
    rob_obj_all = np.concatenate(rob_obj_list, axis=0)

    reg_phi_all = np.concatenate(reg_phi_list, axis=0)
    rob_phi_all = np.concatenate(rob_phi_list, axis=0)
    reg_tx_all  = np.concatenate(reg_tx_list,  axis=0)
    rob_tx_all  = np.concatenate(rob_tx_list,  axis=0)

    # ----------------------------
    # 5) Quick sanity stats
    # ----------------------------
    # 舊指標：VaR surrogate
    reg_prob_var_viol = float(np.mean(reg_snr_var_all < SENSING_SNR_THRESHOLD))
    rob_prob_var_viol = float(np.mean(rob_snr_var_all < SENSING_SNR_THRESHOLD))

    # 新指標 1：平均 empirical violation probability
    reg_mean_snr_vprob = float(np.mean(reg_snr_vprob_all))
    rob_mean_snr_vprob = float(np.mean(rob_snr_vprob_all))

    # 新指標 2：真正對應 chance constraint 的檢查
    # 每條 test channel 的 empirical P(SNR<thr) 是否超過 epsilon=q
    reg_prob_cc_viol = float(np.mean(reg_snr_vprob_all > OUTAGE_QUANTILE))
    rob_prob_cc_viol = float(np.mean(rob_snr_vprob_all > OUTAGE_QUANTILE))

    print("====================================================")
    print(f"[Metric A] Mean E[SumRate] (over {N_VAL} injections):")
    print(f"  REG: {float(np.mean(reg_sumrate_mean_all)):.6f} bits/s/Hz")
    print(f"  ROB: {float(np.mean(rob_sumrate_mean_all)):.6f} bits/s/Hz")

    print(f"[Metric B] Mean VaR_{OUTAGE_QUANTILE}(SNR) (linear -> dB):")
    print(f"  REG: {10.0*np.log10(max(float(np.mean(reg_snr_var_all)), 1e-12)):.3f} dB")
    print(f"  ROB: {10.0*np.log10(max(float(np.mean(rob_snr_var_all)), 1e-12)):.3f} dB")

    print(f"[Old surrogate] P(VaR_{OUTAGE_QUANTILE}(SNR) < thr={SENSING_SNR_THRESHOLD_dB} dB):")
    print(f"  REG: {reg_prob_var_viol*100:.2f}%")
    print(f"  ROB: {rob_prob_var_viol*100:.2f}%")

    print(f"[New metric] Mean empirical P(SNR < thr={SENSING_SNR_THRESHOLD_dB} dB) over {N_VAL} injections:")
    print(f"  REG: {reg_mean_snr_vprob*100:.2f}%")
    print(f"  ROB: {rob_mean_snr_vprob*100:.2f}%")

    print(f"[Empirical chance constraint] P( empirical P(SNR < thr) > eps={OUTAGE_QUANTILE:.2f} ):")
    print(f"  REG: {reg_prob_cc_viol*100:.2f}%")
    print(f"  ROB: {rob_prob_cc_viol*100:.2f}%")

    print(f"[Objective] Mean objective = E[SumRate] - λ_snr*pen - λ_phi*phi_pen - λ_p*tx_excess:")
    print(f"  REG: {float(np.mean(reg_obj_all)):.6f}")
    print(f"  ROB: {float(np.mean(rob_obj_all)):.6f}")
    print("====================================================")

    # 建資料夾
    fig_dir = os.path.join(MLP_DIR, "eval_figures")
    os.makedirs(fig_dir, exist_ok=True)
    q_pct = int(round(OUTAGE_QUANTILE * 100))

    # ----------------------------
    # 6-1) Plot CDF 1: E[SumRate]
    # ----------------------------
    x_sr_reg, y_sr_reg = empirical_cdf(reg_sumrate_mean_all)
    x_sr_rob, y_sr_rob = empirical_cdf(rob_sumrate_mean_all)

    plt.figure()
    plt.plot(x_sr_reg, y_sr_reg, label="REG: E[SumRate]")
    plt.plot(x_sr_rob, y_sr_rob, label="ROB: E[SumRate]")
    plt.xlabel("Mean Sum Rate over injections (bits/s/Hz)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"Mean Sum Rate over injections — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"CDF_meanSumRate_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ----------------------------
    # 6-2) Plot CDF 2: VaR_q(SNR)  (x 軸用 dB)
    # ----------------------------
    reg_snr_var_db = 10.0 * np.log10(np.clip(reg_snr_var_all, 1e-12, None))
    rob_snr_var_db = 10.0 * np.log10(np.clip(rob_snr_var_all, 1e-12, None))
    x_v_reg, y_v_reg = empirical_cdf(reg_snr_var_db)
    x_v_rob, y_v_rob = empirical_cdf(rob_snr_var_db)

    plt.figure()
    plt.plot(x_v_reg, y_v_reg, label=f"REG: VaR_{OUTAGE_QUANTILE}(SNR)")
    plt.plot(x_v_rob, y_v_rob, label=f"ROB: VaR_{OUTAGE_QUANTILE}(SNR)")
    plt.xlabel(f"VaR_{OUTAGE_QUANTILE}(SNR) (dB)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"VaR_{OUTAGE_QUANTILE}(SNR) CDF — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"CDF_VaR{q_pct:02d}SNR_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ----------------------------
    # 6-3) Plot CDF 3: min SumRate among injections
    # ----------------------------
    x_min_reg, y_min_reg = empirical_cdf(reg_sumrate_min_all)
    x_min_rob, y_min_rob = empirical_cdf(rob_sumrate_min_all)

    plt.figure()
    plt.plot(x_min_reg, y_min_reg, label="REG: min[SumRate]")
    plt.plot(x_min_rob, y_min_rob, label="ROB: min[SumRate]")
    plt.xlabel("Min Sum Rate among injections (bits/s/Hz)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"Min Sum Rate among injections — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"CDF_minSumRate_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ----------------------------
    # 6-4) Plot CDF 4: empirical violation probability
    # ----------------------------
    x_p_reg, y_p_reg = empirical_cdf(reg_snr_vprob_all * 100.0)
    x_p_rob, y_p_rob = empirical_cdf(rob_snr_vprob_all * 100.0)

    plt.figure()
    plt.plot(x_p_reg, y_p_reg, label="REG: empirical violation prob.")
    plt.plot(x_p_rob, y_p_rob, label="ROB: empirical violation prob.")
    plt.xlabel(f"Empirical P(SNR < {SENSING_SNR_THRESHOLD_dB} dB) over injections (%)")
    plt.ylabel("CDF  P(X ≤ x)")
    plt.title(f"Empirical SNR-threshold violation probability CDF — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"CDF_empiricalViolationProb_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()