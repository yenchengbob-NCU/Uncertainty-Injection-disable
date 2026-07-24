# -*- coding: utf-8 -*-
import os
import math
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, RZF_LAMBDA
from two_timescale_NN import ThetaNet


# Communication power ratio sweep: 0.1, 0.2, ..., 0.9
# Sensing power ratio is always 1 - communication ratio.
COMM_POWER_SWEEP = np.arange(1, 10, dtype=np.float32) / 10.0


# ============================================================
# Helpers
# ============================================================
def beamformers_power_split(W_C, W_R, comm_power_ratio):
    """
    Input W_C and W_R must already be unit-power normalized.

    comm_power_ratio  : communication power / total power
    sense_power_ratio : 1 - comm_power_ratio
    """
    total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)
    comm_power_ratio = float(comm_power_ratio)
    sense_power_ratio = 1.0 - comm_power_ratio

    p_C = total_power * comm_power_ratio
    p_R = total_power * sense_power_ratio

    W_C = torch.sqrt(p_C) * W_C
    W_R = torch.sqrt(p_R) * W_R

    return W_C, W_R


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape,device=device,dtype=torch.float32) * sigma
    ni = torch.randn(shape,device=device,dtype=torch.float32) * sigma
    return torch.complex(nr,ni).to(dtype=cdtype)


if __name__ == "__main__":
    # 簡寫
    S, B, M, N, K = INJECTION_SAMPLES, N_TEST_CHANNELS, TX_ANT, RIS_UNIT, UAV_COMM

    # 這裡不使用 net，只使用 two_timescale_NN.py 的物理計算函式
    physics_net = ThetaNet().to(DEVICE)
    physics_net.eval()

    # 設定路徑
    test_dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")
    ris_only_ckpt = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    result_dir = RESULT_DIR
    os.makedirs(result_dir,exist_ok=True)

    csv_path = os.path.join(result_dir, "RZF_power_split_sweep.csv")
    npz_path = os.path.join(result_dir, "RZF_power_split_sweep.npz")
    plot_path = os.path.join(result_dir, "RZF_power_split_pareto.png")

    # 讀取資料
    test_dataset = physics_net.load_channel_dataset(test_dataset_path, "test")

    # 載入 ThetaNet 權重
    theta_net = ThetaNet().to(DEVICE)
    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    robust_results = []
    target_snr_mean_db_results = []
    sensing_violation_percent_results = []

    with torch.no_grad():
        # 載入估測通道
        h_dk_test = torch.as_tensor(test_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)   # (B,M,K)
        h_rk_test = torch.as_tensor(test_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)   # (B,N,K)
        G_test = torch.as_tensor(test_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)          # (B,N,M)
        g_dt_test = torch.as_tensor(test_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)    # (B,M,1)

        # 每筆估測通道取得一組 learned RIS
        theta = theta_net(h_dk_test,h_rk_test,G_test,g_dt_test)                                     # (B,N)

        # 估測等效通道
        H_eff_H = physics_net.compute_effective_channel(h_dk_test,h_rk_test,G_test,theta)

        # RZF 與 NS-MRT direction，兩者都已各自正規化為 unit power
        W_C_raw = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)                                           # (B,M,K)
        W_R_raw = mrt_in_H_eff_H_nullspace(H_eff_H,g_dt_test)                                       # (B,M,1)

        # ============================================================
        # Generate one shared set of injected channels
        # All power splits are evaluated on exactly the same errors.
        # ============================================================
        h_dk_rep = h_dk_test.unsqueeze(0).expand(S,B,M,K)
        h_rk_rep = h_rk_test.unsqueeze(0).expand(S,B,N,K)
        G_rep = G_test.unsqueeze(0).expand(S,B,N,M)
        g_dt_rep = g_dt_test.unsqueeze(0).expand(S,B,M,1)

        h_dk_power = torch.mean(torch.abs(h_dk_rep) ** 2,dim=(2,3),keepdim=True).real
        h_rk_power = torch.mean(torch.abs(h_rk_rep) ** 2,dim=(2,3),keepdim=True).real
        G_power = torch.mean(torch.abs(G_rep) ** 2,dim=(2,3),keepdim=True).real
        g_dt_power = torch.mean(torch.abs(g_dt_rep) ** 2,dim=(2,3),keepdim=True).real

        h_dk_inj = h_dk_rep + torch.sqrt(INJECTION_VARIANCE * h_dk_power) * complex_awgn(h_dk_rep.shape,1.0,DEVICE,h_dk_rep.dtype)
        h_rk_inj = h_rk_rep + torch.sqrt(INJECTION_VARIANCE * h_rk_power) * complex_awgn(h_rk_rep.shape,1.0,DEVICE,h_rk_rep.dtype)
        G_inj = G_rep + torch.sqrt(INJECTION_VARIANCE * G_power) * complex_awgn(G_rep.shape,1.0,DEVICE,G_rep.dtype)
        g_dt_inj = g_dt_rep + torch.sqrt(INJECTION_VARIANCE * g_dt_power) * complex_awgn(g_dt_rep.shape,1.0,DEVICE,g_dt_rep.dtype)

        # 壓平 S 與 B，維持每個 estimated channel 的 S 筆 realization 順序
        h_dk_flat = h_dk_inj.reshape(S*B,M,K)
        h_rk_flat = h_rk_inj.reshape(S*B,N,K)
        G_flat = G_inj.reshape(S*B,N,M)
        g_dt_flat = g_dt_inj.reshape(S*B,M,1)

        theta_flat = theta.unsqueeze(0).expand(S,B,N).reshape(S*B,N)

        # Injected effective channels do not depend on power split, so compute once.
        H_eff_H_flat = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)

        # ============================================================
        # Power-split sweep
        # ============================================================
        for comm_power_ratio in COMM_POWER_SWEEP:
            sense_power_ratio = 1.0 - float(comm_power_ratio)

            W_C, W_R = beamformers_power_split(W_C_raw,W_R_raw,comm_power_ratio)

            W_C_flat = W_C.unsqueeze(0).expand(S,B,M,K).reshape(S*B,M,K)
            W_R_flat = W_R.unsqueeze(0).expand(S,B,M,1).reshape(S*B,M,1)

            metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)

            robust_allUE_rate = metrics_flat["rate"].reshape(S,B,K)
            robust_target_snr = metrics_flat["target_snr"].reshape(S,B)

            # 每個 injected realization 先取 worst UE，再對每個 estimated channel 取 Q0.05
            worstUE_rate = torch.min(robust_allUE_rate,dim=2).values
            robust_per_channel = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)
            robust = torch.mean(robust_per_channel)

            # 對全部 S*B 筆 target SNR 做 linear mean，再轉 dB
            robust_target_snr_mean = torch.mean(robust_target_snr)
            robust_target_snr_mean_db = 10.0 * torch.log10(robust_target_snr_mean.clamp_min(1e-12))

            # 每一筆 injected target SNR 與 linear threshold 比較
            sensing_violation_mask = robust_target_snr < SENSING_SNR_THRESHOLD
            sensing_violation_count = torch.sum(sensing_violation_mask)
            sensing_total_count = robust_target_snr.numel()
            sensing_violation_percent = 100.0 * sensing_violation_count.float() / sensing_total_count

            robust_value = float(robust.detach().cpu())
            target_snr_mean_db_value = float(robust_target_snr_mean_db.detach().cpu())
            sensing_violation_percent_value = float(sensing_violation_percent.detach().cpu())

            robust_results.append(robust_value)
            target_snr_mean_db_results.append(target_snr_mean_db_value)
            sensing_violation_percent_results.append(sensing_violation_percent_value)

            print("=" * 90)
            print(f"[RZF + NS-MRT | Comm:Sense = {comm_power_ratio:.1f}:{sense_power_ratio:.1f}]")
            print(f"Robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} : {robust_value:.6f} bps/Hz")
            print(f"Target SNR mean                            : {target_snr_mean_db_value:.3f} dB")
            print(f"Sensing violation                          : {sensing_violation_percent_value:.4f}%")
            print(f"Sensing violation count                    : {int(sensing_violation_count.detach().cpu())} / {sensing_total_count}")
            print("=" * 90)

    comm_power_results = np.asarray(COMM_POWER_SWEEP,dtype=np.float64)
    sense_power_results = 1.0 - comm_power_results
    robust_results = np.asarray(robust_results,dtype=np.float64)
    target_snr_mean_db_results = np.asarray(target_snr_mean_db_results,dtype=np.float64)
    sensing_violation_percent_results = np.asarray(sensing_violation_percent_results,dtype=np.float64)

    # ============================================================
    # Save CSV and NPZ
    # ============================================================
    with open(csv_path,"w",newline="",encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "comm_power_ratio",
            "sense_power_ratio",
            "robust_rate_q05_bps_hz",
            "target_snr_mean_db",
            "sensing_violation_percent",
        ])
        for i in range(len(comm_power_results)):
            writer.writerow([
                f"{comm_power_results[i]:.1f}",
                f"{sense_power_results[i]:.1f}",
                f"{robust_results[i]:.9f}",
                f"{target_snr_mean_db_results[i]:.9f}",
                f"{sensing_violation_percent_results[i]:.9f}",
            ])

    np.savez(
        npz_path,
        comm_power_ratio=comm_power_results,
        sense_power_ratio=sense_power_results,
        robust_rate_q05_bps_hz=robust_results,
        target_snr_mean_db=target_snr_mean_db_results,
        sensing_violation_percent=sensing_violation_percent_results,
        outage_quantile=np.asarray(OUTAGE_QUANTILE,dtype=np.float64),
        sensing_snr_threshold_db=np.asarray(SENSING_SNR_THRESHOLD_DB,dtype=np.float64),
        rzf_lambda=np.asarray(RZF_LAMBDA,dtype=np.float64),
    )

    # ============================================================
    # Pareto plot: x = sensing violation, y = robust rate
    # ============================================================
    plt.figure(figsize=(7.2,5.4))
    plt.plot(sensing_violation_percent_results,robust_results,marker="o",linewidth=2)

    for i in range(len(comm_power_results)):
        split_label = f"{int(round(comm_power_results[i] * 10))}:{int(round(sense_power_results[i] * 10))}"
        plt.annotate(
            split_label,
            (sensing_violation_percent_results[i],robust_results[i]),
            textcoords="offset points",
            xytext=(6,6),
            fontsize=9,
        )

    plt.xlabel("Sensing violation probability (%)")
    plt.ylabel(f"Robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
    plt.title("RZF + NS-MRT Power-Split Sweep")
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path,dpi=300,bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 90)
    print("[RZF power-split sweep finished]")
    print(f"CSV  : {csv_path}")
    print(f"NPZ  : {npz_path}")
    print(f"Plot : {plot_path}")
    print("=" * 90)
