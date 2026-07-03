# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from one_timescale_NN import CommNet


# alpha_factor = 0.0 -> ZF
# alpha_factor > 0.0 -> RZF
#
# alpha = alpha_factor * mean(diag(H H^H))
RZF_ALPHA_FACTOR = 1e-3

# ============================================================
# Helpers
# ============================================================

def to_db_np(x, eps=1e-30):
    x = np.asarray(x, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(x, eps))


def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def make_random_ris(B):
    phase = 2.0 * math.pi * torch.rand(
        B,
        RIS_UNIT,
        dtype=torch.float32,
        device=DEVICE,
    )

    theta = torch.exp(1j * phase).to(torch.complex64)

    return theta


def make_rzf_beamformer(H_eff_H, alpha_factor):
    """
    Downlink RZF / ZF precoder.

    H_eff_H : (B,K,M)
        Effective channel matrix H.
        In metrics, received signal is computed as H_eff_H @ W_C.

    alpha_factor:
        Dimensionless regularization factor.
        alpha_factor = 0.0 gives ZF-like precoder.
        alpha_factor > 0.0 gives RZF precoder.

    W_C : (B,M,K)
    """

    B = H_eff_H.shape[0]
    K = H_eff_H.shape[1]

    H = H_eff_H                                      # (B,K,M)
    Hh = torch.conj(H).transpose(1, 2)               # (B,M,K)

    if alpha_factor == 0.0:
        # ZF: W = H^H (H H^H)^(-1)
        # torch.linalg.pinv(H) gives H^+ with shape (B,M,K)
        W_C = torch.linalg.pinv(H)

    else:
        gram = torch.matmul(H, Hh)                   # (B,K,K)

        diag_power = torch.real(
            torch.diagonal(gram, dim1=1, dim2=2)
        )                                            # (B,K)

        avg_power = torch.mean(diag_power, dim=1)    # (B,)

        alpha = alpha_factor * avg_power             # (B,)

        eye = torch.eye(
            K,
            dtype=H_eff_H.dtype,
            device=H_eff_H.device,
        ).unsqueeze(0)                                # (1,K,K)

        reg = alpha[:, None, None] * eye

        # RZF: W = H^H (H H^H + alpha I)^(-1)
        inv_part = torch.linalg.solve(
            gram + reg,
            eye.expand(B, K, K),
        )                                            # (B,K,K)

        W_C = torch.matmul(Hh, inv_part)             # (B,M,K)
        
    return W_C


def mrt_in_H_eff_H_nullspace(H_eff_H, g_dt):
    """
    Base on "Cell-Free ISAC MIMO Systems: Joint Sensing and Communication Beamforming"
    Eq. (17)-style NS sensing beamformer

    H_eff_H : (B, K, M)
    g_dt : (B, M, 1)

    Return
        W_R : (B, M, 1)
    """

    H = torch.as_tensor(H_eff_H, dtype=torch.complex64, device=DEVICE)   # (B,K,M)
    g = torch.as_tensor(g_dt, dtype=torch.complex64, device=DEVICE)      # (B,M,1)

    B = H.shape[0]
    M = H.shape[2]

    eye = torch.eye(
        M,
        dtype=torch.complex64,
        device=DEVICE,
    ).unsqueeze(0).expand(B, M, M)                    # (B,M,M)

    # Projector onto nullspace of H_eff_H:
    # P_NS = I - H^+ H
    H_pinv = torch.linalg.pinv(H)                    # (B,M,K)
    P_NS = eye - torch.matmul(H_pinv, H)             # (B,M,M)

    # Eq. (17)-style NS-MRT:
    # w_R = P_NS g_dt / ||P_NS g_dt||
    W_R = torch.matmul(P_NS, g)                      # (B,M,1)

    W_R_norm = torch.linalg.norm(W_R, dim=1, keepdim=True)
    W_R = W_R / torch.clamp(W_R_norm, min=1e-12)

    return W_R



if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    dataset_path = os.path.join(DATA_DIR, "dataset_val.npz")        # 這裡讀資料
    dataset = physics_net.load_channel_dataset(dataset_path, "val")

    # ============================================================
    # 取出固定 layout 下的所有帶 PL 估測通道
    # ============================================================
    h_dk = torch.as_tensor(dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, K)
    h_rk = torch.as_tensor(dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, N, K)
    G    = torch.as_tensor(dataset["G_hat"],dtype=torch.complex64,device=DEVICE)      # (B, N, M)
    g_dt = torch.as_tensor(dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, 1)

    theta = make_random_ris(1)                     # 針對這 layout 建立 1 組 random RIS

    H_eff_H = physics_net.compute_effective_channel(h_dk,h_rk,G,theta)

    W_C = make_rzf_beamformer(H_eff_H,RZF_ALPHA_FACTOR)

    W_R = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt)

    W_C, W_R = physics_net.normalize_isac_beamformers(W_C, W_R, g_dt)

    W_C_power = torch.mean(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)))
    W_R_power = torch.mean(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2)))
    W_total_power = W_C_power + W_R_power

    print(
        f"[After sensing-first allocation] \n"
        f"W_C power = {float(W_C_power.detach().cpu()):.6e}, \n"
        f"W_R power = {float(W_R_power.detach().cpu()):.6e}, \n"
        f"W_C ratio = {float((W_C_power / W_total_power).detach().cpu()):.2f}, "
        f"W_R ratio = {float((W_R_power / W_total_power).detach().cpu()):.2f}"
    )

    metrics = physics_net.compute_isac_batch_performance(
        H_eff_H=H_eff_H,
        g_dt=g_dt,
        W_C=W_C,
        W_R=W_R,                                               
    )

    sinr_db_all    = metrics["sinr_db"].detach().cpu().numpy().reshape(-1)
    sumrate_all    = metrics["sumrate"].detach().cpu().numpy().reshape(-1)
    target_snr_all = metrics["target_snr"].detach().cpu().numpy().reshape(-1)

    sinr_outage = np.mean(sinr_db_all < COMM_SINR_THRESHOLD_DB)
    q05_sinr_db = np.quantile(sinr_db_all, 0.05)

    logs = {
        "sumrate_mean": float(metrics["sumrate_mean"].detach().cpu()),
        "sinr_outage": float(sinr_outage),
        "q05_sinr_db": float(q05_sinr_db),

        "target_snr_mean": float(metrics["target_snr_mean"].detach().cpu()),
        "target_snr_mean_db": float(metrics["target_snr_mean_db"].detach().cpu()),

        "sinr_user_mean": metrics["sinr_user_mean"].detach().cpu().numpy(),
        "sinr_user_mean_db": metrics["sinr_user_mean_db"].detach().cpu().numpy(),
        "rate_user_mean": metrics["rate_user_mean"].detach().cpu().numpy(),

        "sinr_db_all": sinr_db_all,
        "sumrate_all": sumrate_all,
        "target_snr_all": target_snr_all,
    }

    # ============================================================
    # Aggregate
    # ============================================================
    B = h_dk.shape[0]

    mean_sumrate = logs["sumrate_mean"]
    q05_sinr_db  = logs["q05_sinr_db"]
    sinr_outage  = logs["sinr_outage"]

    mean_target_snr    = logs["target_snr_mean"]
    mean_target_snr_db = logs["target_snr_mean_db"]

    mean_user_sinr    = logs["sinr_user_mean"]
    mean_user_sinr_db = logs["sinr_user_mean_db"]
    mean_user_rate    = logs["rate_user_mean"]

    target_snr_db_all = to_db_np(logs["target_snr_all"])
    q05_target_snr_db = float(np.quantile(target_snr_db_all, OUTAGE_QUANTILE))

    q05_sumrate = float(np.quantile(logs["sumrate_all"], OUTAGE_QUANTILE))

    print("=" * 90)
    print("[Random RIS + RZF Baseline]")
    print("=" * 90)
    print("Scenario        : 固定 UE layout")
    print(f"Val channels    : {B}")
    print(f"RIS phase       : Random RIS, one theta shared by all {B} train estimated channels")
    print(f"W_C             : RZF, alpha_factor = {RZF_ALPHA_FACTOR:g}")
    print(f"W_R             : H_eff_H nullspace MRT")
    print("-" * 90)

    if Debug:
        print("[UE positions]")
        ue_layout = np.asarray(UE_LAYOUT, dtype=np.float32)
        for k in range(UAV_COMM):
            print(f"UE{k:<2d}: [{ue_layout[k, 0]:>8.4f}, {ue_layout[k, 1]:>8.4f}]")
            print("-" * 90)

    print("[Communication metrics over val estimated channels]")
    print(f"Mean sum-rate   : {mean_sumrate:.6f} bps/Hz")
    print(f"Q05 sum-rate    : {q05_sumrate:.6f} bps/Hz")
    print(f"Q05 SINR        : {q05_sinr_db:.3f} dB")
    print(f"SINR outage     : {100.0 * sinr_outage:.2f} %  below {COMM_SINR_THRESHOLD_DB:.2f} dB")
    print(f"UE SINR linear  : {fmt_vec(mean_user_sinr, precision=4)}")
    print(f"UE SINR dB mean : {fmt_vec(mean_user_sinr_db, precision=3)}")
    print(f"UE rate mean    : {fmt_vec(mean_user_rate, precision=4)} bps/Hz")

    print("-" * 90)
    print("[Target sensing metrics over train estimated channels]")
    print(
        f"Target SNR mean : "
        f"{mean_target_snr:.6e} linear / {mean_target_snr_db:.3f} dB"
    )
    print(f"Target SNR Q05  : {q05_target_snr_db:.3f} dB")
    print("=" * 90)



