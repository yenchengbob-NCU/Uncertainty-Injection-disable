# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from 暫時用不到.one_timescale_NN import CommNet


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
    """
    產生B組random RIS phase
    """
    # 產生實數向量,代表每個element相位旋轉角度
    phase = 2.0 * math.pi * torch.rand(B,RIS_UNIT,dtype=torch.float32,device=DEVICE,) # shape = (B, RIS_UNIT)
    # 變成RIS相位旋轉向量
    theta = torch.exp(1j * phase).to(torch.complex64)

    return theta


def make_rzf_beamformer(H_eff_H, lambda_reg):
    """
    Downlink RZF precoder based on Cell-Free ISAC Eq. (18).

    Math:
        W_C = H^H (H H^H + lambda I_K)^(-1)

    Input:
        H_eff_H   : (B, K, M)
            Effective channel matrix H.
            Each row is h_eff,k^H.

        lambda_reg:
            RZF regularization parameter lambda.
            lambda_reg = 0 gives ZF-like precoder if H H^H is invertible.

    Return:
        W_C       : (B, M, K)
    """

    H = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=DEVICE)                    # (B,K,M)
    lambda_reg = torch.as_tensor(lambda_reg,dtype=torch.float32,device=DEVICE)          # scalar

    B = H.shape[0]
    K = H.shape[1]

    Hh = torch.conj(H).transpose(1, 2)                                                  # (B,M,K)

    gram = torch.matmul(H, Hh)                                                          # (B,K,K)

    eye = torch.eye(K,dtype=torch.complex64,device=DEVICE).unsqueeze(0).expand(B, K, K) # (B,K,K)

    reg_gram = gram + lambda_reg * eye                                                  # (B,K,K)

    inv_part = torch.linalg.solve(reg_gram,eye)                                         # (B,K,K)

    W_C = torch.matmul(Hh, inv_part)                                                    # (B,M,K)

    return W_C


def mrt_in_H_eff_H_nullspace(H_eff_H, g_dt):
    """
    Downlink RZF precoder based on Cell-Free ISAC Eq. (17)

    Math:
        P_NS = I_M - H^† H
        W_R_raw = P_NS g_dt
        W_R = W_R_raw / ||W_R_raw||

    Input:
        H_eff_H : (B, K, M)
            Effective communication channel matrix H.
            Each row is h_eff,k^H.

        g_dt : (B, M, 1)
            Target sensing channel direction.

    Return:
        W_R : (B, M, 1)
            Unit-norm sensing beam direction.
            Power is assigned later by normalize_isac_beamformers().
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

    # 這裡不使用net 只是要用one_timescale_NN.py的物理計算副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    dataset_path = os.path.join(DATA_DIR, "dataset_val.npz")
    dataset = physics_net.load_channel_dataset(dataset_path, "val")

    # ============================================================
    # 取出固定 layout 下的所有帶 PL 估測通道
    # ============================================================
    h_dk = torch.as_tensor(dataset["h_dk_hat"], dtype=torch.complex64, device=DEVICE)   # (B, M, K)
    h_rk = torch.as_tensor(dataset["h_rk_hat"], dtype=torch.complex64, device=DEVICE)   # (B, N, K)
    G    = torch.as_tensor(dataset["G_hat"],    dtype=torch.complex64, device=DEVICE)   # (B, N, M)
    g_dt = torch.as_tensor(dataset["g_dt_hat"], dtype=torch.complex64, device=DEVICE)   # (B, M, 1)

    B = h_dk.shape[0]

    theta = make_random_ris(1)                     # 針對這 layout 建立 1 組 random RIS

    H_eff_H = physics_net.compute_effective_channel(h_dk, h_rk, G, theta)

    # ============================================================
    # W_R does not depend on RZF lambda, so compute it once.
    # ============================================================
    W_R_unit = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt)

    # ============================================================
    # Check: H_eff_H @ W_R ≈ 0
    # ============================================================
    with torch.no_grad():
        HWR = torch.matmul(H_eff_H, W_R_unit)           # (B,K,1)
        HWR_abs = torch.abs(HWR)                        # (B,K,1)

        HWR_max = torch.max(HWR_abs)
        HWR_mean = torch.mean(HWR_abs)
        HWR_rms = torch.sqrt(torch.mean(HWR_abs ** 2))

        W_R_unit_norm = torch.sqrt(
            torch.sum(torch.abs(W_R_unit) ** 2, dim=(1, 2))
        )

        print("=" * 90)
        print("[Nullspace check: H_eff_H @ W_R ≈ 0]")
        print(f"HWR shape      : {tuple(HWR.shape)}")
        print(f"max |HWR|      : {float(HWR_max.detach().cpu()):.6e}")
        print(f"mean |HWR|     : {float(HWR_mean.detach().cpu()):.6e}")
        print(f"rms |HWR|      : {float(HWR_rms.detach().cpu()):.6e}")
        print(f"W_R norm mean  : {float(torch.mean(W_R_unit_norm).detach().cpu()):.6e}")
        print(f"W_R norm min   : {float(torch.min(W_R_unit_norm).detach().cpu()):.6e}")
        print(f"W_R norm max   : {float(torch.max(W_R_unit_norm).detach().cpu()):.6e}")
        print("=" * 90)

    # ============================================================
    # Sweep RZF lambda
    # lambda = 0 gives ZF.
    # Positive lambda gives RZF.
    # ============================================================
    RZF_LAMBDA_LIST = [
        0.0,
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
    ]

    sweep_results = []

    print("=" * 150)
    print("[RZF_LAMBDA Sweep]")
    print("=" * 150)
    print(
        f"{'lambda':>12s} | "
        f"{'raw_sig':>10s} {'raw_intf':>10s} {'rawP':>10s} | "
        f"{'dir_sig':>10s} {'dir_intf':>10s} | "
        f"{'Pc%':>7s} {'Pr%':>7s} | "
        f"{'MeanRate':>10s} {'Q05Rate':>10s} {'Q05SINR':>9s} {'Out%':>8s} | "
        f"{'TarMean':>9s} {'TarQ05':>9s} {'TarOut%':>8s}"
    )
    print("-" * 150)

    for RZF_LAMBDA in RZF_LAMBDA_LIST:
        try:
            W_C_raw = make_rzf_beamformer(H_eff_H, RZF_LAMBDA)

            with torch.no_grad():
                # ------------------------------------------------------------
                # Raw RZF/ZF check: H W_C_raw
                # ------------------------------------------------------------
                HWC_raw = torch.matmul(H_eff_H, W_C_raw)      # (B,K,K)
                HWC_raw_power = torch.abs(HWC_raw) ** 2

                raw_signal = torch.diagonal(HWC_raw_power, dim1=1, dim2=2)
                raw_interf = torch.sum(HWC_raw_power, dim=2) - raw_signal

                raw_signal_mean = torch.mean(raw_signal)
                raw_interf_mean = torch.mean(raw_interf)
                raw_power_mean = torch.mean(torch.sum(torch.abs(W_C_raw) ** 2, dim=(1, 2)))

                # ------------------------------------------------------------
                # Direction check: H W_C_dir
                # This is closer to the direction used before power allocation.
                # ------------------------------------------------------------
                W_C_dir = W_C_raw / (
                    torch.sqrt(torch.sum(torch.abs(W_C_raw) ** 2, dim=(1, 2), keepdim=True).real) + 1e-12
                )

                HWC_dir = torch.matmul(H_eff_H, W_C_dir)      # (B,K,K)
                HWC_dir_power = torch.abs(HWC_dir) ** 2

                dir_signal = torch.diagonal(HWC_dir_power, dim1=1, dim2=2)
                dir_interf = torch.sum(HWC_dir_power, dim=2) - dir_signal

                dir_signal_mean = torch.mean(dir_signal)
                dir_interf_mean = torch.mean(dir_interf)

                # ------------------------------------------------------------
                # Actual ISAC normalization and performance
                # ------------------------------------------------------------
                W_C, W_R = physics_net.normalize_isac_beamformers(
                    W_C_raw,
                    W_R_unit,
                    g_dt,
                )

                W_C_power = torch.mean(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)))
                W_R_power = torch.mean(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2)))
                W_total_power = W_C_power + W_R_power

                metrics = physics_net.compute_isac_batch_performance(
                    H_eff_H=H_eff_H,
                    g_dt=g_dt,
                    W_C=W_C,
                    W_R=W_R,
                )

                sinr_db_all = metrics["sinr_db"].detach().cpu().numpy().reshape(-1)
                sumrate_all = metrics["sumrate"].detach().cpu().numpy().reshape(-1)
                target_snr_all = metrics["target_snr"].detach().cpu().numpy().reshape(-1)

                mean_sumrate = float(metrics["sumrate_mean"].detach().cpu())
                q05_sumrate = float(np.quantile(sumrate_all, OUTAGE_QUANTILE))
                q05_sinr_db = float(np.quantile(sinr_db_all, OUTAGE_QUANTILE))
                sinr_outage = float(np.mean(sinr_db_all < COMM_SINR_THRESHOLD_DB))

                target_snr_db_all = to_db_np(target_snr_all)
                mean_target_snr_db = float(metrics["target_snr_mean_db"].detach().cpu())
                q05_target_snr_db = float(np.quantile(target_snr_db_all, OUTAGE_QUANTILE))
                target_outage = float(np.mean(target_snr_all < SENSING_SNR_THRESHOLD))

                Pc_ratio = float((W_C_power / W_total_power).detach().cpu())
                Pr_ratio = float((W_R_power / W_total_power).detach().cpu())

                sweep_results.append({
                    "lambda": float(RZF_LAMBDA),
                    "raw_signal_mean": float(raw_signal_mean.detach().cpu()),
                    "raw_interf_mean": float(raw_interf_mean.detach().cpu()),
                    "raw_power_mean": float(raw_power_mean.detach().cpu()),
                    "dir_signal_mean": float(dir_signal_mean.detach().cpu()),
                    "dir_interf_mean": float(dir_interf_mean.detach().cpu()),
                    "Pc_ratio": Pc_ratio,
                    "Pr_ratio": Pr_ratio,
                    "mean_sumrate": mean_sumrate,
                    "q05_sumrate": q05_sumrate,
                    "q05_sinr_db": q05_sinr_db,
                    "sinr_outage": sinr_outage,
                    "mean_target_snr_db": mean_target_snr_db,
                    "q05_target_snr_db": q05_target_snr_db,
                    "target_outage": target_outage,
                })

                print(
                    f"{RZF_LAMBDA:12.1e} | "
                    f"{float(raw_signal_mean.detach().cpu()):10.3e} "
                    f"{float(raw_interf_mean.detach().cpu()):10.3e} "
                    f"{float(raw_power_mean.detach().cpu()):10.3e} | "
                    f"{float(dir_signal_mean.detach().cpu()):10.3e} "
                    f"{float(dir_interf_mean.detach().cpu()):10.3e} | "
                    f"{100.0 * Pc_ratio:7.2f} "
                    f"{100.0 * Pr_ratio:7.2f} | "
                    f"{mean_sumrate:10.4f} "
                    f"{q05_sumrate:10.4f} "
                    f"{q05_sinr_db:9.3f} "
                    f"{100.0 * sinr_outage:8.2f} | "
                    f"{mean_target_snr_db:9.3f} "
                    f"{q05_target_snr_db:9.3f} "
                    f"{100.0 * target_outage:8.2f}"
                )

        except RuntimeError as err:
            print(
                f"{RZF_LAMBDA:12.1e} | FAILED: {str(err).splitlines()[0]}"
            )

    print("=" * 150)

    # ============================================================
    # Save sweep result
    # ============================================================
    if len(sweep_results) > 0:
        save_path = os.path.join(DATA_DIR, "rzf_lambda_sweep_baseline.npz")

        np.savez(
            save_path,
            rzf_lambda=np.asarray([r["lambda"] for r in sweep_results], dtype=np.float64),
            raw_signal_mean=np.asarray([r["raw_signal_mean"] for r in sweep_results], dtype=np.float64),
            raw_interf_mean=np.asarray([r["raw_interf_mean"] for r in sweep_results], dtype=np.float64),
            raw_power_mean=np.asarray([r["raw_power_mean"] for r in sweep_results], dtype=np.float64),
            dir_signal_mean=np.asarray([r["dir_signal_mean"] for r in sweep_results], dtype=np.float64),
            dir_interf_mean=np.asarray([r["dir_interf_mean"] for r in sweep_results], dtype=np.float64),
            Pc_ratio=np.asarray([r["Pc_ratio"] for r in sweep_results], dtype=np.float64),
            Pr_ratio=np.asarray([r["Pr_ratio"] for r in sweep_results], dtype=np.float64),
            mean_sumrate=np.asarray([r["mean_sumrate"] for r in sweep_results], dtype=np.float64),
            q05_sumrate=np.asarray([r["q05_sumrate"] for r in sweep_results], dtype=np.float64),
            q05_sinr_db=np.asarray([r["q05_sinr_db"] for r in sweep_results], dtype=np.float64),
            sinr_outage=np.asarray([r["sinr_outage"] for r in sweep_results], dtype=np.float64),
            mean_target_snr_db=np.asarray([r["mean_target_snr_db"] for r in sweep_results], dtype=np.float64),
            q05_target_snr_db=np.asarray([r["q05_target_snr_db"] for r in sweep_results], dtype=np.float64),
            target_outage=np.asarray([r["target_outage"] for r in sweep_results], dtype=np.float64),
        )

        print(f"[SAVE] RZF lambda sweep result saved to: {save_path}")
