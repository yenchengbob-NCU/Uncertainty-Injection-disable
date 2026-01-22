# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import *
from rician import large_scale_fading

def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)

def empirical_cdf(x: np.ndarray):
    x_sorted = np.sort(x)
    y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
    return x_sorted, y

def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    real/imag ~ N(0, variance/2)
    """
    sigma = np.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    n = torch.complex(nr, ni).to(dtype=cdtype)
    return n

@torch.no_grad()
def eval_outage_sumrate_and_snr(
    comm_net, sense_net, ris_net, #reg 或 rob
    h_dk_hat: torch.Tensor,  # (B,M,K) estimated
    h_rk_hat: torch.Tensor,  # (B,N,K) estimated
    G_hat:    torch.Tensor,  # (B,N,M) estimated
    g_dt_hat: torch.Tensor,  # (B,M,1) estimated
    beta_dk_row, beta_rk_row, beta_G, beta_dt, #相同的est
    injection_samples: int,
    injection_variance: float,
    outage_q: float,
    chunk: int = 50
):
    """
    核心評估邏輯（完全照你描述）：
    - 先用估測通道 (hat) 得到 W_C, W_S, phi
    - 再把估測通道複製 L 次，做 additive injection 得到 tilde
    - 在 tilde 上算 sum-rate 與 sensing SNR
    - 對每個樣本取 quantile_q 作為 outage 指標
    """
    comm_net.eval(); sense_net.eval(); ris_net.eval()

    B, M, K = h_dk_hat.shape
    N = h_rk_hat.shape[1]
    L = int(injection_samples)
    q = float(outage_q)

    # 1) Design variables from estimated channels
    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)   # (B,M,K)
    W_S = sense_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)  # (B,M,1)
    phi = ris_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)    # (B,N)

    # 2) For each sample, collect L realizations' metrics
    sumrate_chunks = []
    snr_chunks = []

    for s0 in range(0, L, chunk):
        s = min(chunk, L - s0)

        # ---- replicate estimated channels: (B,s,...) -> (B*s,...)
        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B*s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B*s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B*s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B*s, M, 1)

        # ---- additive injection (same variance for 4 channels)
        cdtype = h_dk_rep.dtype
        device = h_dk_rep.device
        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape, injection_variance, device, cdtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape, injection_variance, device, cdtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,    injection_variance, device, cdtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape, injection_variance, device, cdtype)

        # ---- replicate design variables to match (B*s,...)
        W_C_rep = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B*s, M, K)
        W_S_rep = W_S.unsqueeze(1).expand(B, s, M, 1).reshape(B*s, M, 1)
        phi_rep = phi.unsqueeze(1).expand(B, s, N).reshape(B*s, N)

        # ---- compute SINR -> sumrate on injected channels
        sinrs = comm_net.compute_comm_sinrs(
            h_dk_inj, h_rk_inj, G_inj, phi_rep, W_S_rep, W_C_rep,
            beta_dk_row, beta_rk_row, beta_G
        )  # (B*s,K)
        rates = comm_net.compute_rates(sinrs)  # (B*s,K)
        sum_rate = rates.sum(dim=1)            # (B*s,)

        # ---- sensing SNR on injected channels (no RIS per your current model)
        sense_snr = comm_net.compute_sense_snr(g_dt_inj, W_S_rep, W_C_rep, beta_dt)  # (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))
        snr_chunks.append(sense_snr.real.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)  # (B,L)
    snr_samples     = torch.cat(snr_chunks, dim=1)      # (B,L)

    # 3) outage quantile per sample
    sumrate_out = torch.quantile(sumrate_samples, q=q, dim=1)  # (B,)
    snr_out     = torch.quantile(snr_samples,     q=q, dim=1)  # (B,)

    return sumrate_out, snr_out


# ============================================================
# Main evaluation
# ============================================================
if __name__ == "__main__":

    # ----------------------------
    # 1) Load estimated test channels (same path rule as rician.py)
    # ----------------------------
    npz_path = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING, "channelEstimates_test.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"找不到 channelEstimates_test.npz：{npz_path}\n"
        )

    data = np.load(npz_path)
    h_dk_all = np_to_torch_complex(data["h_dk"])   # (N_TEST,M,K)
    h_rk_all = np_to_torch_complex(data["h_rk"])   # (N_TEST,N,K)
    G_all    = np_to_torch_complex(data["G"])      # (N_TEST,N,M)
    g_dt_all = np_to_torch_complex(data["g_dt"])   # (N_TEST,M,1)

    n_test = h_dk_all.shape[0]
    print(f"[EVAL] Loaded estimated channels: {npz_path}")
    print(f"[EVAL] N_TEST={n_test}, M={TX_ANT}, N={RIS_UNIT}, K={UAV_COMM}")

    # ----------------------------
    # 2) Load large-scale fading (power betas)
    # ----------------------------
    beta_G, beta_dt, beta_dk_row, beta_rk_row = large_scale_fading()

    # ----------------------------
    # 3) Load 6 models (regular + robust)
    # ----------------------------
    ckpt_dir = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING, "ckpt")

    reg_comm_ckpt = os.path.join(ckpt_dir, f"comm_{SETTING_STRING}.ckpt")
    reg_sens_ckpt = os.path.join(ckpt_dir, f"sense_{SETTING_STRING}.ckpt")
    reg_ris_ckpt  = os.path.join(ckpt_dir, f"ris_{SETTING_STRING}.ckpt")

    rob_comm_ckpt = os.path.join(ckpt_dir, f"comm_robust_{SETTING_STRING}.ckpt")
    rob_sens_ckpt = os.path.join(ckpt_dir, f"sense_robust_{SETTING_STRING}.ckpt")
    rob_ris_ckpt  = os.path.join(ckpt_dir, f"ris_robust_{SETTING_STRING}.ckpt")

    for p in [reg_comm_ckpt, reg_sens_ckpt, reg_ris_ckpt, rob_comm_ckpt, rob_sens_ckpt, rob_ris_ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"找不到 checkpoint：{p}")

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
    # 4) Evaluate outage metrics on injected channels
    # ----------------------------
    torch.manual_seed(RANDOM_SEED)  # 讓 injection 可重現
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)

    BATCH = 200   # 你可依 GPU 調整：100~500
    CHUNK = 100    # 每次展開的 injection chunk，50~200 通常可

    reg_out_list = []
    rob_out_list = []
    reg_snr_out_list = []
    rob_snr_out_list = []

    for i0 in range(0, n_test, BATCH):
        i1 = min(i0 + BATCH, n_test)

        h_dk = h_dk_all[i0:i1]
        h_rk = h_rk_all[i0:i1]
        G    = G_all[i0:i1]
        g_dt = g_dt_all[i0:i1]

        # Regular outage
        reg_out, reg_snr_out = eval_outage_sumrate_and_snr(
            reg_comm, reg_sens, reg_ris,
            h_dk, h_rk, G, g_dt,
            beta_dk_row, beta_rk_row, beta_G, beta_dt,
            injection_samples=INJECTION_SAMPLES,
            injection_variance=INJECTION_VARIANCE,
            outage_q=OUTAGE_QUANTILE,
            chunk=CHUNK
        )

        # Robust outage
        rob_out, rob_snr_out = eval_outage_sumrate_and_snr(
            rob_comm, rob_sens, rob_ris,
            h_dk, h_rk, G, g_dt,
            beta_dk_row, beta_rk_row, beta_G, beta_dt,
            injection_samples=INJECTION_SAMPLES,
            injection_variance=INJECTION_VARIANCE,
            outage_q=OUTAGE_QUANTILE,
            chunk=CHUNK
        )

        reg_out_list.append(reg_out.detach().cpu().numpy())
        rob_out_list.append(rob_out.detach().cpu().numpy())
        reg_snr_out_list.append(reg_snr_out.detach().cpu().numpy())
        rob_snr_out_list.append(rob_snr_out.detach().cpu().numpy())

        print(f"[EVAL] {i1}/{n_test} done.")

    reg_out_all = np.concatenate(reg_out_list, axis=0)
    rob_out_all = np.concatenate(rob_out_list, axis=0)
    reg_snr_out_all = np.concatenate(reg_snr_out_list, axis=0)
    rob_snr_out_all = np.concatenate(rob_snr_out_list, axis=0)

    # ----------------------------
    # 5) Quick sanity stats (outage SNR constraint)
    # ----------------------------
    reg_violate = np.mean(reg_snr_out_all < SENSING_SNR_THRESHOLD)
    rob_violate = np.mean(rob_snr_out_all < SENSING_SNR_THRESHOLD)

    reg_mean_out = float(np.mean(reg_out_all))
    rob_mean_out = float(np.mean(rob_out_all))

    print("====================================================")
    print(f"[OUTAGE q={OUTAGE_QUANTILE}] Mean outage sum-rate:")
    print(f"  REG: {reg_mean_out:.4f} bits/s/Hz")
    print(f"  ROB: {rob_mean_out:.4f} bits/s/Hz")
    print(f"[OUTAGE q={OUTAGE_QUANTILE}] Sensing SNR violation prob (SNR_out < threshold):")
    print(f"  REG: {reg_violate*100:.2f}%")
    print(f"  ROB: {rob_violate*100:.2f}%")
    print("====================================================")

    # ----------------------------
    # 6) Plot CDF of outage sum-rate
    # ----------------------------
    x1, y1 = empirical_cdf(reg_out_all)
    x2, y2 = empirical_cdf(rob_out_all)

    plt.figure()
    plt.plot(x1, y1, label=f"Regular  (outage q={OUTAGE_QUANTILE})")
    plt.plot(x2, y2, label=f"Robust   (outage q={OUTAGE_QUANTILE})")
    plt.xlabel("Outage Sum Rate (bits/s/Hz)")
    plt.ylabel("CDF  P(OutageSumRate ≤ x)")
    plt.title(f"Outage SumRate CDF (Injection Eval) — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()