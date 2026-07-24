# -*- coding: utf-8 -*-
import os
import math
import csv
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, RZF_LAMBDA
from two_timescale_NN import CommNet, RadarNet, ThetaNet


# ============================================================
# Sweep settings
# ============================================================
ROB_WEIGHT = 0.5
COMM_POWER_SWEEP = np.arange(1,10,dtype=np.float32) / 10.0

# rob_0.5 was trained at Comm:Sense = 4:6.
ROB_TRAIN_COMM_RATIO = 0.4


# ============================================================
# Helpers
# ============================================================
def beamformers_power_split(W_C_dir, W_R_dir, comm_power_ratio):
    """
    W_C_dir and W_R_dir must already be unit-Frobenius-norm directions.
    """
    total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)
    comm_power_ratio = float(comm_power_ratio)
    sense_power_ratio = 1.0 - comm_power_ratio

    W_C = torch.sqrt(total_power * comm_power_ratio) * W_C_dir
    W_R = torch.sqrt(total_power * sense_power_ratio) * W_R_dir

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


def network_beam_direction(net, H_eff_H, g_dt):
    """
    Support both project variants:
        net(H_eff_H)
        net(H_eff_H, g_dt)
    """
    forward_arg_count = len(inspect.signature(net.forward).parameters)

    if forward_arg_count == 1:
        return net(H_eff_H)

    if forward_arg_count == 2:
        return net(H_eff_H,g_dt)

    raise TypeError(f"Unsupported {net.__class__.__name__}.forward signature: {inspect.signature(net.forward)}")


def evaluate_direction_sweep(physics_net, H_eff_H_flat, g_dt_flat, W_C_dir, W_R_dir, S, B, M, K):
    robust_results = []
    target_snr_mean_db_results = []
    sensing_violation_percent_results = []

    for comm_power_ratio in COMM_POWER_SWEEP:
        sense_power_ratio = 1.0 - float(comm_power_ratio)

        W_C, W_R = beamformers_power_split(W_C_dir,W_R_dir,comm_power_ratio)

        W_C_flat = W_C.unsqueeze(0).expand(S,B,M,K).reshape(S*B,M,K)
        W_R_flat = W_R.unsqueeze(0).expand(S,B,M,RADAR_STREAMS).reshape(S*B,M,RADAR_STREAMS)

        metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)

        robust_allUE_rate = metrics_flat["rate"].reshape(S,B,K)
        robust_target_snr = metrics_flat["target_snr"].reshape(S,B)

        worstUE_rate = torch.min(robust_allUE_rate,dim=2).values
        robust_per_channel = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)
        robust = torch.mean(robust_per_channel)

        robust_target_snr_mean = torch.mean(robust_target_snr)
        robust_target_snr_mean_db = 10.0 * torch.log10(robust_target_snr_mean.clamp_min(1e-12))

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

        print(f"Comm:Sense = {comm_power_ratio:.1f}:{sense_power_ratio:.1f} | Robust={robust_value:.6f} bps/Hz | TargetSNR={target_snr_mean_db_value:.3f} dB | Violation={sensing_violation_percent_value:.4f}%")

    return {
        "robust": np.asarray(robust_results,dtype=np.float64),
        "target_snr_mean_db": np.asarray(target_snr_mean_db_results,dtype=np.float64),
        "sensing_violation_percent": np.asarray(sensing_violation_percent_results,dtype=np.float64),
    }


if __name__ == "__main__":
    S, B, M, N, K = INJECTION_SAMPLES, N_TEST_CHANNELS, TX_ANT, RIS_UNIT, UAV_COMM

    physics_net = ThetaNet().to(DEVICE)
    comm_net = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)

    physics_net.eval()
    comm_net.eval()
    radar_net.eval()
    theta_net.eval()

    # ============================================================
    # Paths
    # ============================================================
    test_dataset_path = os.path.join(DATA_DIR,"dataset_test.npz")
    ris_only_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

    # Current folder structure:
    # Two_timescale/<scenario>/robust/rob_0.5/
    rob05_ckpt_dir = os.path.join(BASE_RUN_DIR,"robust",f"rob_{ROB_WEIGHT:g}")
    rob05_comm_ckpt = os.path.join(rob05_ckpt_dir,"two_timescale_comm_rob.ckpt")
    rob05_radar_ckpt = os.path.join(rob05_ckpt_dir,"two_timescale_radar_rob.ckpt")

    result_dir = RESULT_DIR
    os.makedirs(result_dir,exist_ok=True)

    csv_path = os.path.join(result_dir,"RZF_ROB05_power_split_sweep.csv")
    npz_path = os.path.join(result_dir,"RZF_ROB05_power_split_sweep.npz")
    rob_plot_path = os.path.join(result_dir,"ROB05_power_split_pareto.png")
    overlay_plot_path = os.path.join(result_dir,"RZF_ROB05_power_split_pareto.png")

    for required_path in [test_dataset_path,ris_only_ckpt,rob05_comm_ckpt,rob05_radar_ckpt]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required file not found: {required_path}")

    print("=" * 100)
    print("[RZF + ROB0.5 power-split sweep]")
    print(f"ROB checkpoint directory : {rob05_ckpt_dir}")
    print(f"ROB training split       : {ROB_TRAIN_COMM_RATIO:.1f}:{1.0-ROB_TRAIN_COMM_RATIO:.1f}")
    print(f"Evaluation splits        : {COMM_POWER_SWEEP.tolist()}")
    print("=" * 100)

    # ============================================================
    # Load dataset and checkpoints
    # ============================================================
    test_dataset = physics_net.load_channel_dataset(test_dataset_path,"test")

    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    comm_net.load_model(rob05_comm_ckpt,strict=True,verbose=True)
    radar_net.load_model(rob05_radar_ckpt,strict=True,verbose=True)

    with torch.no_grad():
        h_dk_test = torch.as_tensor(test_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
        h_rk_test = torch.as_tensor(test_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
        G_test = torch.as_tensor(test_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
        g_dt_test = torch.as_tensor(test_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

        # Same learned RIS for RZF and ROB.
        theta = theta_net(h_dk_test,h_rk_test,G_test,g_dt_test)
        H_eff_H = physics_net.compute_effective_channel(h_dk_test,h_rk_test,G_test,theta)

        # Unit-power beam directions.
        W_C_rzf_dir = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)
        W_R_rzf_dir = mrt_in_H_eff_H_nullspace(H_eff_H,g_dt_test)

        W_C_rob_dir = network_beam_direction(comm_net,H_eff_H,g_dt_test)
        W_R_rob_dir = network_beam_direction(radar_net,H_eff_H,g_dt_test)

        # ============================================================
        # One shared injection set for both methods and every split.
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

        h_dk_flat = h_dk_inj.reshape(S*B,M,K)
        h_rk_flat = h_rk_inj.reshape(S*B,N,K)
        G_flat = G_inj.reshape(S*B,N,M)
        g_dt_flat = g_dt_inj.reshape(S*B,M,1)

        theta_flat = theta.unsqueeze(0).expand(S,B,N).reshape(S*B,N)
        H_eff_H_flat = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)

        print("\n" + "=" * 100)
        print("[RZF + NS-MRT]")
        print("=" * 100)
        rzf_results = evaluate_direction_sweep(physics_net,H_eff_H_flat,g_dt_flat,W_C_rzf_dir,W_R_rzf_dir,S,B,M,K)

        print("\n" + "=" * 100)
        print("[ROB0.5 network directions]")
        print("=" * 100)
        rob_results = evaluate_direction_sweep(physics_net,H_eff_H_flat,g_dt_flat,W_C_rob_dir,W_R_rob_dir,S,B,M,K)

    comm_power_ratio = np.asarray(COMM_POWER_SWEEP,dtype=np.float64)
    sense_power_ratio = 1.0 - comm_power_ratio

    # ============================================================
    # Save CSV
    # ============================================================
    with open(csv_path,"w",newline="",encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "comm_power_ratio",
            "sense_power_ratio",
            "robust_rate_q05_bps_hz",
            "target_snr_mean_db",
            "sensing_violation_percent",
        ])

        for method_name, method_results in [("RZF+NS-MRT",rzf_results),("ROB0.5",rob_results)]:
            for i in range(len(comm_power_ratio)):
                writer.writerow([
                    method_name,
                    f"{comm_power_ratio[i]:.1f}",
                    f"{sense_power_ratio[i]:.1f}",
                    f"{method_results['robust'][i]:.9f}",
                    f"{method_results['target_snr_mean_db'][i]:.9f}",
                    f"{method_results['sensing_violation_percent'][i]:.9f}",
                ])

    # ============================================================
    # Save NPZ
    # ============================================================
    np.savez(
        npz_path,
        comm_power_ratio=comm_power_ratio,
        sense_power_ratio=sense_power_ratio,
        rzf_robust_rate_q05_bps_hz=rzf_results["robust"],
        rzf_target_snr_mean_db=rzf_results["target_snr_mean_db"],
        rzf_sensing_violation_percent=rzf_results["sensing_violation_percent"],
        rob05_robust_rate_q05_bps_hz=rob_results["robust"],
        rob05_target_snr_mean_db=rob_results["target_snr_mean_db"],
        rob05_sensing_violation_percent=rob_results["sensing_violation_percent"],
        rob_weight=np.asarray(ROB_WEIGHT,dtype=np.float64),
        rob_training_comm_ratio=np.asarray(ROB_TRAIN_COMM_RATIO,dtype=np.float64),
        outage_quantile=np.asarray(OUTAGE_QUANTILE,dtype=np.float64),
        sensing_snr_threshold_db=np.asarray(SENSING_SNR_THRESHOLD_DB,dtype=np.float64),
        rzf_lambda=np.asarray(RZF_LAMBDA,dtype=np.float64),
    )

    split_labels = [
        f"{int(round(comm_power_ratio[i] * 10))}:{int(round(sense_power_ratio[i] * 10))}"
        for i in range(len(comm_power_ratio))
    ]

    # ============================================================
    # ROB-only plot
    # ============================================================
    plt.figure(figsize=(7.5,5.6))
    plt.plot(
        rob_results["sensing_violation_percent"],
        rob_results["robust"],
        marker="o",
        linewidth=2,
        label="ROB0.5",
    )

    for i, split_label in enumerate(split_labels):
        suffix = " train" if np.isclose(comm_power_ratio[i],ROB_TRAIN_COMM_RATIO) else ""
        plt.annotate(
            split_label + suffix,
            (rob_results["sensing_violation_percent"][i],rob_results["robust"][i]),
            textcoords="offset points",
            xytext=(6,6),
            fontsize=9,
        )

    plt.xlabel("Sensing violation probability (%)")
    plt.ylabel(f"Robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
    plt.title("ROB0.5 Power-Split Sensitivity")
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(rob_plot_path,dpi=300,bbox_inches="tight")
    plt.close()

    # ============================================================
    # Overlay plot
    # ============================================================
    plt.figure(figsize=(8.2,6.0))
    plt.plot(
        rzf_results["sensing_violation_percent"],
        rzf_results["robust"],
        marker="o",
        linewidth=2,
        label="RZF + NS-MRT",
    )
    plt.plot(
        rob_results["sensing_violation_percent"],
        rob_results["robust"],
        marker="s",
        linewidth=2,
        linestyle="--",
        label="ROB0.5",
    )

    for i, split_label in enumerate(split_labels):
        plt.annotate(
            split_label,
            (rzf_results["sensing_violation_percent"][i],rzf_results["robust"][i]),
            textcoords="offset points",
            xytext=(5,5),
            fontsize=8,
        )

        rob_label = split_label + (" train" if np.isclose(comm_power_ratio[i],ROB_TRAIN_COMM_RATIO) else "")
        plt.annotate(
            rob_label,
            (rob_results["sensing_violation_percent"][i],rob_results["robust"][i]),
            textcoords="offset points",
            xytext=(5,-11),
            fontsize=8,
        )

    plt.xlabel("Sensing violation probability (%)")
    plt.ylabel(f"Robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
    plt.title("Power-Split Sweep: RZF + NS-MRT vs ROB0.5")
    plt.grid(True,alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(overlay_plot_path,dpi=300,bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 100)
    print("[Power-split comparison finished]")
    print(f"CSV              : {csv_path}")
    print(f"NPZ              : {npz_path}")
    print(f"ROB-only plot    : {rob_plot_path}")
    print(f"Overlay plot     : {overlay_plot_path}")
    print("=" * 100)
