# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from one_timescale_NN import CommNet


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

    RZF_LAMBDA = 0                                 # 為求簡便，我們設計成ZF 實際上sweep也是ZF較好

    W_C = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)

    W_R = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt)

    # W_C, W_R = physics_net.normalize_isac_beamformers(W_C, W_R, g_dt)

    # ============================================================
    # Sweep P_R / P_C allocation
    # ============================================================
    total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)
    # 1) 先把 W_C, W_R 都變成 direction，消除 raw power 不均問題
    W_C_dir = W_C / (torch.sqrt(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12)
    W_R_dir = W_R / (torch.sqrt(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12)

    power_sweep_logs = []

    print("=" * 130)
    print("[P_R / P_C Sweep]")
    print("=" * 130)
    print(
        f"{'P_R%':>7s} {'P_C%':>7s} | "
        f"{'MeanRate':>10s} {'Q05Rate':>10s} {'Q05SINR':>9s} {'SINROut%':>9s} | "
        f"{'TarMean':>9s} {'TarQ05':>9s} {'TarOut%':>8s} | "
        f"{'Feasible':>8s}"
    )
    print("-" * 130)

    for pr_percent in range(100, -1, -10):

        pr_ratio = pr_percent / 100.0
        pc_ratio = 1.0 - pr_ratio

        p_R = total_power * pr_ratio
        p_C = total_power * pc_ratio

        W_R_eval = torch.sqrt(p_R) * W_R_dir
        W_C_eval = torch.sqrt(p_C) * W_C_dir

        metrics = physics_net.compute_isac_batch_performance(
            H_eff_H=H_eff_H,
            g_dt=g_dt,
            W_C=W_C_eval,
            W_R=W_R_eval,
        )

        sinr_db_all = metrics["sinr_db"].detach().cpu().numpy().reshape(-1)
        sumrate_all = metrics["sumrate"].detach().cpu().numpy().reshape(-1)
        target_snr_all = metrics["target_snr"].detach().cpu().numpy().reshape(-1)

        target_snr_db_all = to_db_np(target_snr_all)

        mean_sumrate_now = float(metrics["sumrate_mean"].detach().cpu())
        q05_sumrate_now = float(np.quantile(sumrate_all, OUTAGE_QUANTILE))
        q05_sinr_db_now = float(np.quantile(sinr_db_all, OUTAGE_QUANTILE))
        sinr_outage_now = float(np.mean(sinr_db_all < COMM_SINR_THRESHOLD_DB))

        mean_target_snr_now = float(metrics["target_snr_mean"].detach().cpu())
        mean_target_snr_db_now = float(metrics["target_snr_mean_db"].detach().cpu())
        q05_target_snr_db_now = float(np.quantile(target_snr_db_all, OUTAGE_QUANTILE))
        target_outage_now = float(np.mean(target_snr_db_all < SENSING_SNR_THRESHOLD_DB))

        feasible_now = q05_target_snr_db_now >= SENSING_SNR_THRESHOLD_DB

        power_sweep_logs.append({
            "pr_ratio": pr_ratio,
            "pc_ratio": pc_ratio,

            "W_C_eval": W_C_eval,
            "W_R_eval": W_R_eval,

            "sumrate_mean": mean_sumrate_now,
            "q05_sumrate": q05_sumrate_now,
            "q05_sinr_db": q05_sinr_db_now,
            "sinr_outage": sinr_outage_now,

            "target_snr_mean": mean_target_snr_now,
            "target_snr_mean_db": mean_target_snr_db_now,
            "q05_target_snr_db": q05_target_snr_db_now,
            "target_outage": target_outage_now,

            "sinr_user_mean": metrics["sinr_user_mean"].detach().cpu().numpy(),
            "sinr_user_mean_db": metrics["sinr_user_mean_db"].detach().cpu().numpy(),
            "rate_user_mean": metrics["rate_user_mean"].detach().cpu().numpy(),

            "sinr_db_all": sinr_db_all,
            "sumrate_all": sumrate_all,
            "target_snr_all": target_snr_all,

            "feasible": feasible_now,
        })

        print(
            f"{100.0 * pr_ratio:7.1f} {100.0 * pc_ratio:7.1f} | "
            f"{mean_sumrate_now:10.4f} "
            f"{q05_sumrate_now:10.4f} "
            f"{q05_sinr_db_now:9.3f} "
            f"{100.0 * sinr_outage_now:9.2f} | "
            f"{mean_target_snr_db_now:9.3f} "
            f"{q05_target_snr_db_now:9.3f} "
            f"{100.0 * target_outage_now:8.2f} | "
            f"{str(feasible_now):>8s}"
        )

    print("=" * 130)

    # ============================================================
    # Selection rule:
    # 在 sensing feasible 的配置中，選 mean sum-rate 最大者
    # feasible condition: Target SNR Q05 >= sensing threshold
    # ============================================================
    feasible_logs = [x for x in power_sweep_logs if x["feasible"]]

    if len(feasible_logs) > 0:
        best_log = max(
            feasible_logs,
            key=lambda x: x["sumrate_mean"],
        )
        selection_note = "Best feasible: max mean sum-rate under target Q05 sensing constraint"
    else:
        best_log = max(
            power_sweep_logs,
            key=lambda x: x["q05_target_snr_db"],
        )
        selection_note = "No feasible split: select highest target Q05 SNR"

    W_C = best_log["W_C_eval"]
    W_R = best_log["W_R_eval"]

    W_C_power = torch.mean(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)))
    W_R_power = torch.mean(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2)))
    W_total_power = W_C_power + W_R_power

    logs = {
        "sumrate_mean": best_log["sumrate_mean"],
        "sinr_outage": best_log["sinr_outage"],
        "q05_sinr_db": best_log["q05_sinr_db"],

        "target_snr_mean": best_log["target_snr_mean"],
        "target_snr_mean_db": best_log["target_snr_mean_db"],

        "sinr_user_mean": best_log["sinr_user_mean"],
        "sinr_user_mean_db": best_log["sinr_user_mean_db"],
        "rate_user_mean": best_log["rate_user_mean"],

        "sinr_db_all": best_log["sinr_db_all"],
        "sumrate_all": best_log["sumrate_all"],
        "target_snr_all": best_log["target_snr_all"],
    }

    # ============================================================
    # Aggregate selected split
    # ============================================================
    B = h_dk.shape[0]

    mean_sumrate = logs["sumrate_mean"]
    q05_sinr_db = logs["q05_sinr_db"]
    sinr_outage = logs["sinr_outage"]

    mean_target_snr = logs["target_snr_mean"]
    mean_target_snr_db = logs["target_snr_mean_db"]

    mean_user_sinr = logs["sinr_user_mean"]
    mean_user_sinr_db = logs["sinr_user_mean_db"]
    mean_user_rate = logs["rate_user_mean"]

    target_snr_db_all = to_db_np(logs["target_snr_all"])
    q05_target_snr_db = float(np.quantile(target_snr_db_all, OUTAGE_QUANTILE))

    q05_sumrate = float(np.quantile(logs["sumrate_all"], OUTAGE_QUANTILE))

    print("=" * 90)
    print("[Selected P_R / P_C Split]")
    print(selection_note)
    print(
        f"P_R ratio = {100.0 * best_log['pr_ratio']:.1f} %, "
        f"P_C ratio = {100.0 * best_log['pc_ratio']:.1f} %"
    )
    print(
        f"W_C power = {float(W_C_power.detach().cpu()):.6e}, "
        f"W_R power = {float(W_R_power.detach().cpu()):.6e}, "
        f"Total = {float(W_total_power.detach().cpu()):.6e}"
    )
    print("=" * 90)










    print("=" * 90)
    print("[Random RIS + ZF + NS-MRT Baseline with P_R/P_C Sweep]")
    print("=" * 90)
    print("Scenario        : 固定 UE layout")
    print(f"Val channels    : {B}")
    print(f"RIS phase       : Random RIS, one theta shared by all {B} train estimated channels")
    print(f"W_C             : ZF, RZF_LAMBDA= {RZF_LAMBDA:g}")
    print(f"Power split     : P_R={100.0 * best_log['pr_ratio']:.1f}%, P_C={100.0 * best_log['pc_ratio']:.1f}%")
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



