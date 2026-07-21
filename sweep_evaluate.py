# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from baseline import beamformers_power_split
from two_timescale_NN import CommNet, RadarNet, ThetaNet


# ================================
# Penalty sweep
# ================================
REG_PENALTY_SWEEP = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0 ]
ROB_PENALTY_SWEEP = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0 ]


# ================================
# Helpers
# ================================
def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0,variance):
        E|n|^2 = variance
        Re/Im ~ N(0,variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape,device=device,dtype=torch.float32) * sigma
    ni = torch.randn(shape,device=device,dtype=torch.float32) * sigma
    return torch.complex(nr,ni).to(dtype=cdtype)


def evaluate_checkpoint(
    model_type,
    penalty,
    physics_net,
    test_H_eff_H_hat,
    test_g_dt_hat,
    inj_H_eff_H_flat,
    inj_g_dt_flat,
    S,
    B,
    M,
    K,
    R,
):
    """
    評估單一 REG 或 ROB penalty checkpoint。

    Robust objective:
        rate                 : (S,B,K)
        worstUE_rate         : min over K -> (S,B)
        robust_per_channel   : Q0.05 over S -> (B,)
        robust               : mean over B -> scalar

    Sensing violation probability:
        mean(target_snr < SENSING_SNR_THRESHOLD) over S,B
    """
    model_type = model_type.upper()

    if model_type == "REG":
        penalty_tag = f"reg_{penalty:g}"
        ckpt_dir = os.path.join(BASE_RUN_DIR,"regular",penalty_tag)
        comm_ckpt = os.path.join(ckpt_dir,"two_timescale_comm_reg.ckpt")
        radar_ckpt = os.path.join(ckpt_dir,"two_timescale_radar_reg.ckpt")

    elif model_type == "ROB":
        penalty_tag = f"rob_{penalty:g}"
        ckpt_dir = os.path.join(BASE_RUN_DIR,"robust",penalty_tag)
        comm_ckpt = os.path.join(ckpt_dir,"two_timescale_comm_rob.ckpt")
        radar_ckpt = os.path.join(ckpt_dir,"two_timescale_radar_rob.ckpt")

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    for ckpt_path in [comm_ckpt,radar_ckpt]:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    comm_net = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)

    comm_net.load_model(comm_ckpt,strict=True,verbose=False)
    radar_net.load_model(radar_ckpt,strict=True,verbose=False)

    comm_net.eval()
    radar_net.eval()

    with torch.no_grad():
        # 目前 CommNet、RadarNet 輸入：
        # estimated effective channel + estimated g_dt
        W_C_dir = comm_net(test_H_eff_H_hat,test_g_dt_hat)
        W_R_dir = radar_net(test_H_eff_H_hat,test_g_dt_hat)

        if W_C_dir.shape != (B,M,K):
            raise ValueError(
                f"{model_type} penalty={penalty:g}: "
                f"W_C_dir expected {(B,M,K)}, got {tuple(W_C_dir.shape)}"
            )

        if W_R_dir.shape != (B,M,R):
            raise ValueError(
                f"{model_type} penalty={penalty:g}: "
                f"W_R_dir expected {(B,M,R)}, got {tuple(W_R_dir.shape)}"
            )

        W_C,W_R = beamformers_power_split(W_C_dir,W_R_dir)

        # 同一筆 estimated channel 的 S 筆 injected channels
        # 共用同一組 W_C、W_R
        W_C_rep = W_C.unsqueeze(0).expand(S,B,M,K)
        W_R_rep = W_R.unsqueeze(0).expand(S,B,M,R)

        W_C_flat = W_C_rep.reshape(S*B,M,K)
        W_R_flat = W_R_rep.reshape(S*B,M,R)

        metrics_flat = physics_net.compute_isac_batch_performance(
            inj_H_eff_H_flat,
            inj_g_dt_flat,
            W_C_flat,
            W_R_flat,
        )

        # ================================
        # Robust communication objective
        # ================================
        allUE_rate = metrics_flat["rate"].reshape(S,B,K)

        worstUE_rate = torch.min(allUE_rate,dim=2).values
        robust_per_channel = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)
        robust = torch.mean(robust_per_channel)

        # ================================
        # Target sensing metrics
        # ================================
        target_snr = metrics_flat["target_snr"].reshape(S,B)

        # 先在線性尺度對全部 S、B 平均，再轉 dB
        target_snr_mean = torch.mean(target_snr)
        target_snr_mean_db = 10.0 * torch.log10(target_snr_mean.clamp_min(1e-12))

        # 每一筆 injected channel 是否低於 sensing threshold
        sensing_violation = target_snr < SENSING_SNR_THRESHOLD
        sensing_violation_probability = torch.mean(sensing_violation.float())

    result = {
        "model": model_type,
        "penalty": float(penalty),
        "robust": float(robust.detach().cpu()),
        "target_snr_mean_db": float(target_snr_mean_db.detach().cpu()),
        "violation_probability": float(sensing_violation_probability.detach().cpu()),
        "violation_percent": 100.0 * float(sensing_violation_probability.detach().cpu()),
    }

    print(
        f"[{model_type} penalty={penalty:g}] "
        f"Robust={result['robust']:.6f} bps/Hz | "
        f"TargetMeanSNR={result['target_snr_mean_db']:.3f} dB | "
        f"Violation={result['violation_percent']:.3f}%"
    )

    del comm_net
    del radar_net
    del W_C_dir
    del W_R_dir
    del W_C
    del W_R
    del W_C_rep
    del W_R_rep
    del W_C_flat
    del W_R_flat
    del metrics_flat
    del allUE_rate
    del target_snr

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ================================
# Output paths
# ================================
SWEEP_RESULT_DIR = os.path.join(RESULT_DIR,"penalty_sweep_evaluate")

sweep_npz_path = os.path.join(SWEEP_RESULT_DIR,"penalty_sweep_evaluate.npz")
sweep_csv_path = os.path.join(SWEEP_RESULT_DIR,"penalty_sweep_evaluate.csv")
sweep_plot_path = os.path.join(SWEEP_RESULT_DIR,"violation_probability_vs_robust.png")

os.makedirs(SWEEP_RESULT_DIR,exist_ok=True)


# ================================
# Input paths
# ================================
test_dataset_path = os.path.join(DATA_DIR,"dataset_test.npz")
pretrained_theta_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

for required_path in [test_dataset_path,pretrained_theta_ckpt]:
    if not os.path.exists(required_path):
        raise FileNotFoundError(f"找不到必要檔案: {required_path}")


# ================================
# 先檢查全部 12 組 checkpoints
# 避免 injection 建好後才發現資料夾缺檔
# ================================
required_ckpts = []

for penalty in REG_PENALTY_SWEEP:
    penalty_tag = f"reg_{penalty:g}"
    ckpt_dir = os.path.join(BASE_RUN_DIR,"regular",penalty_tag)

    required_ckpts.append(
        os.path.join(ckpt_dir,"two_timescale_comm_reg.ckpt")
    )
    required_ckpts.append(
        os.path.join(ckpt_dir,"two_timescale_radar_reg.ckpt")
    )

for penalty in ROB_PENALTY_SWEEP:
    penalty_tag = f"rob_{penalty:g}"
    ckpt_dir = os.path.join(BASE_RUN_DIR,"robust",penalty_tag)

    required_ckpts.append(
        os.path.join(ckpt_dir,"two_timescale_comm_rob.ckpt")
    )
    required_ckpts.append(
        os.path.join(ckpt_dir,"two_timescale_radar_rob.ckpt")
    )

missing_ckpts = [
    ckpt_path
    for ckpt_path in required_ckpts
    if not os.path.exists(ckpt_path)
]

if missing_ckpts:
    print("\n[ERROR] 以下 checkpoints 不存在：")

    for ckpt_path in missing_ckpts:
        print(f"  {ckpt_path}")

    raise FileNotFoundError(
        f"共有 {len(missing_ckpts)} 個 checkpoint 缺失，停止 evaluation。"
    )


# ================================
# 載入共用 test dataset 與 ThetaNet
# ================================
physics_net = CommNet().to(DEVICE)
physics_net.eval()

theta_net = ThetaNet().to(DEVICE)
theta_net.load_model(pretrained_theta_ckpt,strict=True,verbose=True)
theta_net.eval()

test_dataset = physics_net.load_channel_dataset(test_dataset_path,"test")

print("\n" + "=" * 90)
print("[PENALTY SWEEP ROBUST TEST EVALUATION]")
print("=" * 90)
print(f"Test dataset              : {test_dataset_path}")
print(f"Pretrained ThetaNet       : {pretrained_theta_ckpt}")
print(f"REG penalties             : {REG_PENALTY_SWEEP}")
print(f"ROB penalties             : {ROB_PENALTY_SWEEP}")
print(f"Sensing threshold         : {SENSING_SNR_THRESHOLD_DB:.3f} dB")
print(f"Injection variance        : {INJECTION_VARIANCE}")
print(f"Injection samples         : {INJECTION_SAMPLES}")
print(f"Robust quantile           : {OUTAGE_QUANTILE:.2f}")
print("=" * 90)


with torch.no_grad():
    # ================================
    # Test estimated channels
    # ================================
    test_h_dk_hat = torch.as_tensor(test_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
    test_h_rk_hat = torch.as_tensor(test_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
    test_G_hat = torch.as_tensor(test_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
    test_g_dt_hat = torch.as_tensor(test_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

    S = INJECTION_SAMPLES
    B = test_h_dk_hat.shape[0]
    M = TX_ANT
    N = RIS_UNIT
    K = UAV_COMM
    R = RADAR_STREAMS

    if B != N_TEST_CHANNELS:
        print(
            f"[WARNING] settings N_TEST_CHANNELS={N_TEST_CHANNELS}, "
            f"但 dataset 實際 B={B}。後續以 dataset 的 B={B} 為準。"
        )

    # ================================
    # 每筆 estimated channel 產生自己的 theta
    # ================================
    test_theta = theta_net(
        test_h_dk_hat,
        test_h_rk_hat,
        test_G_hat,
        test_g_dt_hat,
    )

    # 所有 REG / ROB penalty checkpoint
    # 共用相同的 estimated effective channel
    test_H_eff_H_hat = physics_net.compute_effective_channel(
        test_h_dk_hat,
        test_h_rk_hat,
        test_G_hat,
        test_theta,
    )

    # ================================
    # 一次建立共同 injection channels
    # 所有 12 組 checkpoint 完全共用
    # ================================
    inj_h_dk_rep = test_h_dk_hat.unsqueeze(0).expand(S,B,M,K)
    inj_h_rk_rep = test_h_rk_hat.unsqueeze(0).expand(S,B,N,K)
    inj_G_rep = test_G_hat.unsqueeze(0).expand(S,B,N,M)
    inj_g_dt_rep = test_g_dt_hat.unsqueeze(0).expand(S,B,M,1)

    inj_h_dk_power = torch.mean(torch.abs(inj_h_dk_rep) ** 2,dim=(2,3),keepdim=True).real
    inj_h_rk_power = torch.mean(torch.abs(inj_h_rk_rep) ** 2,dim=(2,3),keepdim=True).real
    inj_G_power = torch.mean(torch.abs(inj_G_rep) ** 2,dim=(2,3),keepdim=True).real
    inj_g_dt_power = torch.mean(torch.abs(inj_g_dt_rep) ** 2,dim=(2,3),keepdim=True).real

    inj_h_dk = inj_h_dk_rep + torch.sqrt(INJECTION_VARIANCE * inj_h_dk_power) * complex_awgn(inj_h_dk_rep.shape,1.0,DEVICE,inj_h_dk_rep.dtype)
    inj_h_rk = inj_h_rk_rep + torch.sqrt(INJECTION_VARIANCE * inj_h_rk_power) * complex_awgn(inj_h_rk_rep.shape,1.0,DEVICE,inj_h_rk_rep.dtype)
    inj_G = inj_G_rep + torch.sqrt(INJECTION_VARIANCE * inj_G_power) * complex_awgn(inj_G_rep.shape,1.0,DEVICE,inj_G_rep.dtype)
    inj_g_dt = inj_g_dt_rep + torch.sqrt(INJECTION_VARIANCE * inj_g_dt_power) * complex_awgn(inj_g_dt_rep.shape,1.0,DEVICE,inj_g_dt_rep.dtype)

    # 同一 estimated channel 的 S 筆 injected channels
    # 共用 ThetaNet 根據 estimated channel 產生的 theta
    test_theta_rep = test_theta.unsqueeze(0).expand(S,B,N)

    inj_h_dk_flat = inj_h_dk.reshape(S*B,M,K)
    inj_h_rk_flat = inj_h_rk.reshape(S*B,N,K)
    inj_G_flat = inj_G.reshape(S*B,N,M)
    inj_g_dt_flat = inj_g_dt.reshape(S*B,M,1)
    test_theta_flat = test_theta_rep.reshape(S*B,N)

    # injected effective channel 與 penalty checkpoint 無關
    # 因此只需要計算一次
    inj_H_eff_H_flat = physics_net.compute_effective_channel(
        inj_h_dk_flat,
        inj_h_rk_flat,
        inj_G_flat,
        test_theta_flat,
    )

# effective channel 建立完成後，釋放大型中間 tensors
del inj_h_dk_rep
del inj_h_rk_rep
del inj_G_rep
del inj_g_dt_rep

del inj_h_dk_power
del inj_h_rk_power
del inj_G_power
del inj_g_dt_power

del inj_h_dk
del inj_h_rk
del inj_G
del inj_g_dt

del inj_h_dk_flat
del inj_h_rk_flat
del inj_G_flat
del test_theta_rep
del test_theta_flat

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ================================
# Evaluate REG sweep
# ================================
reg_results = []

print("\n" + "#" * 90)
print("# REG SWEEP EVALUATION")
print("#" * 90)

for penalty in REG_PENALTY_SWEEP:
    result = evaluate_checkpoint(
        model_type="REG",
        penalty=penalty,
        physics_net=physics_net,
        test_H_eff_H_hat=test_H_eff_H_hat,
        test_g_dt_hat=test_g_dt_hat,
        inj_H_eff_H_flat=inj_H_eff_H_flat,
        inj_g_dt_flat=inj_g_dt_flat,
        S=S,
        B=B,
        M=M,
        K=K,
        R=R,
    )

    reg_results.append(result)


# ================================
# Evaluate ROB sweep
# ================================
rob_results = []

print("\n" + "#" * 90)
print("# ROB SWEEP EVALUATION")
print("#" * 90)

for penalty in ROB_PENALTY_SWEEP:
    result = evaluate_checkpoint(
        model_type="ROB",
        penalty=penalty,
        physics_net=physics_net,
        test_H_eff_H_hat=test_H_eff_H_hat,
        test_g_dt_hat=test_g_dt_hat,
        inj_H_eff_H_flat=inj_H_eff_H_flat,
        inj_g_dt_flat=inj_g_dt_flat,
        S=S,
        B=B,
        M=M,
        K=K,
        R=R,
    )

    rob_results.append(result)


# ================================
# Convert results to arrays
# ================================
reg_penalty = np.asarray([x["penalty"] for x in reg_results],dtype=np.float64)
reg_robust = np.asarray([x["robust"] for x in reg_results],dtype=np.float64)
reg_target_snr_mean_db = np.asarray([x["target_snr_mean_db"] for x in reg_results],dtype=np.float64)
reg_violation_probability = np.asarray([x["violation_probability"] for x in reg_results],dtype=np.float64)
reg_violation_percent = np.asarray([x["violation_percent"] for x in reg_results],dtype=np.float64)

rob_penalty = np.asarray([x["penalty"] for x in rob_results],dtype=np.float64)
rob_robust = np.asarray([x["robust"] for x in rob_results],dtype=np.float64)
rob_target_snr_mean_db = np.asarray([x["target_snr_mean_db"] for x in rob_results],dtype=np.float64)
rob_violation_probability = np.asarray([x["violation_probability"] for x in rob_results],dtype=np.float64)
rob_violation_percent = np.asarray([x["violation_percent"] for x in rob_results],dtype=np.float64)


# ================================
# Save NPZ
# ================================
np.savez(
    sweep_npz_path,

    reg_penalty=reg_penalty,
    reg_robust=reg_robust,
    reg_target_snr_mean_db=reg_target_snr_mean_db,
    reg_violation_probability=reg_violation_probability,
    reg_violation_percent=reg_violation_percent,

    rob_penalty=rob_penalty,
    rob_robust=rob_robust,
    rob_target_snr_mean_db=rob_target_snr_mean_db,
    rob_violation_probability=rob_violation_probability,
    rob_violation_percent=rob_violation_percent,

    sensing_snr_threshold_db=np.asarray(SENSING_SNR_THRESHOLD_DB,dtype=np.float64),
    injection_variance=np.asarray(INJECTION_VARIANCE,dtype=np.float64),
    injection_samples=np.asarray(INJECTION_SAMPLES,dtype=np.int32),
    outage_quantile=np.asarray(OUTAGE_QUANTILE,dtype=np.float64),
    test_channels=np.asarray(B,dtype=np.int32),
)

print(f"\n[SAVE] NPZ: {sweep_npz_path}")


# ================================
# Save CSV
# ================================
with open(sweep_csv_path,"w",newline="",encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow([
        "model",
        "penalty",
        "robust_minimum_user_rate_bps_hz",
        "target_snr_mean_db",
        "sensing_violation_probability",
        "sensing_violation_percent",
    ])

    for result in reg_results + rob_results:
        writer.writerow([
            result["model"],
            result["penalty"],
            result["robust"],
            result["target_snr_mean_db"],
            result["violation_probability"],
            result["violation_percent"],
        ])

print(f"[SAVE] CSV: {sweep_csv_path}")


# ================================
# Plot
# X: sensing violation probability
# Y: robust objective
# ================================
plt.figure(figsize=(11,7))

# 依照 X 軸排序後連線
# 原始 penalty 與結果仍完整保存在 NPZ / CSV
reg_plot_order = np.argsort(reg_violation_percent)
rob_plot_order = np.argsort(rob_violation_percent)

plt.plot(
    reg_violation_percent[reg_plot_order],
    reg_robust[reg_plot_order],
    marker="o",
    linewidth=2.0,
    markersize=7,
    label="REG",
)

plt.plot(
    rob_violation_percent[rob_plot_order],
    rob_robust[rob_plot_order],
    marker="s",
    linewidth=2.0,
    markersize=7,
    label="ROB",
)

# REG penalty labels
for idx in reg_plot_order:
    plt.annotate(
        f"λ={reg_penalty[idx]:g}",
        xy=(reg_violation_percent[idx],reg_robust[idx]),
        xytext=(6,8),
        textcoords="offset points",
        fontsize=9,
    )

# ROB penalty labels
for idx in rob_plot_order:
    plt.annotate(
        f"λ={rob_penalty[idx]:g}",
        xy=(rob_violation_percent[idx],rob_robust[idx]),
        xytext=(6,-15),
        textcoords="offset points",
        fontsize=9,
    )

plt.xlabel(
    f"Sensing Violation Probability: "
    f"P(Target SNR < {SENSING_SNR_THRESHOLD_DB:g} dB) (%)"
)

plt.ylabel(
    f"Robust Minimum-User Rate "
    f"Q{OUTAGE_QUANTILE:.2f} (bps/Hz)"
)

plt.title(
    "REG vs ROB: Sensing Violation Probability "
    "vs Robust Communication Objective"
)

plt.grid(True,alpha=0.35)
plt.legend()
plt.tight_layout()

plt.savefig(sweep_plot_path,dpi=300,bbox_inches="tight")
plt.close()

print(f"[SAVE] Plot: {sweep_plot_path}")


# ================================
# Final table
# ================================
print("\n" + "=" * 110)
print("[REG RESULTS]")
print("=" * 110)
print(f"{'Penalty':>10} {'Robust':>14} {'Target SNR':>14} {'Violation':>14}")

for result in reg_results:
    print(
        f"{result['penalty']:>10g} "
        f"{result['robust']:>14.6f} "
        f"{result['target_snr_mean_db']:>11.3f} dB "
        f"{result['violation_percent']:>12.3f}%"
    )

print("\n" + "=" * 110)
print("[ROB RESULTS]")
print("=" * 110)
print(f"{'Penalty':>10} {'Robust':>14} {'Target SNR':>14} {'Violation':>14}")

for result in rob_results:
    print(
        f"{result['penalty']:>10g} "
        f"{result['robust']:>14.6f} "
        f"{result['target_snr_mean_db']:>11.3f} dB "
        f"{result['violation_percent']:>12.3f}%"
    )

print("\n[INFO] Penalty sweep evaluation finished.")