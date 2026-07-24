# -*- coding: utf-8 -*-
import os
import math
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from baseline import beamformers_power_split
from two_timescale_NN import CommNet, RadarNet, ThetaNet


# ================================
# Sweep settings
# ================================
INJECTION_SWEEP = [0.000, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095]
SWEEP_SEED = RANDOM_SEED


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


def evaluate_network(comm_net,W_C,W_R,inj_h_dk_flat,inj_h_rk_flat,inj_G_flat,inj_g_dt_flat,theta_flat,S,B,K):
    H_eff_H_flat = comm_net.compute_effective_channel(inj_h_dk_flat,inj_h_rk_flat,inj_G_flat,theta_flat)
    metrics_flat = comm_net.compute_isac_batch_performance(H_eff_H_flat,inj_g_dt_flat,W_C,W_R)

    allUE_rate = metrics_flat["rate"].reshape(S,B,K)                       # (S,B,K)
    target_snr_db = metrics_flat["target_snr_db"].reshape(S,B)            # (S,B)

    # Robust metric：每筆injection先找worst UE，再對S取Q0.05，最後對B平均
    worstUE_rate = torch.min(allUE_rate,dim=2).values                     # (S,B)
    robust_per_channel = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)
    robust = torch.mean(robust_per_channel)

    # 所有estimated channels的平均sensing violation rate
    sensing_violation_per_channel = torch.mean((target_snr_db < SENSING_SNR_THRESHOLD_DB).float(),dim=0)
    sensing_violation_rate = torch.mean(sensing_violation_per_channel)

    return float(robust.detach().cpu()),float(sensing_violation_rate.detach().cpu())


# ================================
# Paths：由settings中的penalty值決定checkpoint資料夾
# ================================
REG_PENALTY_TAG = f"reg_{REG_SENSING_LOSS_WEIGHT:g}"
ROB_PENALTY_TAG = f"rob_{ROB_SENSING_LOSS_WEIGHT:g}"

reg_ckpt_dir = os.path.join(BASE_RUN_DIR,"regular",REG_PENALTY_TAG)
rob_ckpt_dir = os.path.join(BASE_RUN_DIR,"robust",ROB_PENALTY_TAG)

test_dataset_path = os.path.join(DATA_DIR,"dataset_test.npz")
pretrained_theta_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

reg_comm_ckpt = os.path.join(reg_ckpt_dir,"two_timescale_comm_reg.ckpt")
reg_radar_ckpt = os.path.join(reg_ckpt_dir,"two_timescale_radar_reg.ckpt")
rob_comm_ckpt = os.path.join(rob_ckpt_dir,"two_timescale_comm_rob.ckpt")
rob_radar_ckpt = os.path.join(rob_ckpt_dir,"two_timescale_radar_rob.ckpt")

result_tag = f"reg_{REG_SENSING_LOSS_WEIGHT:g}_rob_{ROB_SENSING_LOSS_WEIGHT:g}"
csv_path = os.path.join(RESULT_DIR,f"inj_sweep_{result_tag}.csv")
npz_path = os.path.join(RESULT_DIR,f"inj_sweep_{result_tag}.npz")
fig_path = os.path.join(RESULT_DIR,f"inj_sweep_{result_tag}.png")

os.makedirs(RESULT_DIR,exist_ok=True)

for file_path in [test_dataset_path,pretrained_theta_ckpt,reg_comm_ckpt,reg_radar_ckpt,rob_comm_ckpt,rob_radar_ckpt]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到檔案: {file_path}")


# ================================
# 建立並載入網路
# ================================
theta_net = ThetaNet().to(DEVICE)

reg_comm_net = CommNet().to(DEVICE)
reg_radar_net = RadarNet().to(DEVICE)

rob_comm_net = CommNet().to(DEVICE)
rob_radar_net = RadarNet().to(DEVICE)

theta_net.load_model(pretrained_theta_ckpt,strict=True,verbose=True)
reg_comm_net.load_model(reg_comm_ckpt,strict=True,verbose=True)
reg_radar_net.load_model(reg_radar_ckpt,strict=True,verbose=True)
rob_comm_net.load_model(rob_comm_ckpt,strict=True,verbose=True)
rob_radar_net.load_model(rob_radar_ckpt,strict=True,verbose=True)

theta_net.eval()
reg_comm_net.eval()
reg_radar_net.eval()
rob_comm_net.eval()
rob_radar_net.eval()


# ================================
# 讀取test dataset
# ================================
test_dataset = reg_comm_net.load_channel_dataset(test_dataset_path,"test")

print("\n" + "=" * 100)
print("[INJECTION VARIANCE SWEEP]")
print("=" * 100)
print(f"REG sensing penalty       : {REG_SENSING_LOSS_WEIGHT}")
print(f"ROB sensing penalty       : {ROB_SENSING_LOSS_WEIGHT}")
print(f"REG checkpoint directory  : {reg_ckpt_dir}")
print(f"ROB checkpoint directory  : {rob_ckpt_dir}")
print(f"Test dataset              : {test_dataset_path}")
print(f"Fixed power split         : W_C/W_R = 0.4/0.6")
print(f"Injection samples         : {INJECTION_SAMPLES}")
print(f"Communication tail Q      : {OUTAGE_QUANTILE}")
print(f"Sensing threshold         : {SENSING_SNR_THRESHOLD_DB} dB")
print(f"Injection sweep           : {INJECTION_SWEEP}")
print("=" * 100)


with torch.no_grad():
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

    # 固定ThetaNet：每筆estimated channel產生一組theta
    test_theta = theta_net(test_h_dk_hat,test_h_rk_hat,test_G_hat,test_g_dt_hat)

    # REG beamformers：只由estimated channels設計一次
    reg_H_eff_H_hat = reg_comm_net.compute_effective_channel(test_h_dk_hat,test_h_rk_hat,test_G_hat,test_theta)
    reg_W_C_dir = reg_comm_net(reg_H_eff_H_hat,test_g_dt_hat)
    reg_W_R_dir = reg_radar_net(reg_H_eff_H_hat,test_g_dt_hat)
    reg_W_C,reg_W_R = beamformers_power_split(reg_W_C_dir,reg_W_R_dir)

    # ROB beamformers：只由estimated channels設計一次
    rob_H_eff_H_hat = rob_comm_net.compute_effective_channel(test_h_dk_hat,test_h_rk_hat,test_G_hat,test_theta)
    rob_W_C_dir = rob_comm_net(rob_H_eff_H_hat,test_g_dt_hat)
    rob_W_R_dir = rob_radar_net(rob_H_eff_H_hat,test_g_dt_hat)
    rob_W_C,rob_W_R = beamformers_power_split(rob_W_C_dir,rob_W_R_dir)

    # 複製固定theta與beamformers
    theta_flat = test_theta.unsqueeze(0).expand(S,B,N).reshape(S*B,N)

    reg_W_C_flat = reg_W_C.unsqueeze(0).expand(S,B,M,K).reshape(S*B,M,K)
    reg_W_R_flat = reg_W_R.unsqueeze(0).expand(S,B,M,R).reshape(S*B,M,R)

    rob_W_C_flat = rob_W_C.unsqueeze(0).expand(S,B,M,K).reshape(S*B,M,K)
    rob_W_R_flat = rob_W_R.unsqueeze(0).expand(S,B,M,R).reshape(S*B,M,R)

    # Estimated channels的每筆block power，只計算一次
    h_dk_power = torch.mean(torch.abs(test_h_dk_hat) ** 2,dim=(1,2),keepdim=True).real.unsqueeze(0)
    h_rk_power = torch.mean(torch.abs(test_h_rk_hat) ** 2,dim=(1,2),keepdim=True).real.unsqueeze(0)
    G_power = torch.mean(torch.abs(test_G_hat) ** 2,dim=(1,2),keepdim=True).real.unsqueeze(0)
    g_dt_power = torch.mean(torch.abs(test_g_dt_hat) ** 2,dim=(1,2),keepdim=True).real.unsqueeze(0)

    h_dk_rep = test_h_dk_hat.unsqueeze(0).expand(S,B,M,K)
    h_rk_rep = test_h_rk_hat.unsqueeze(0).expand(S,B,N,K)
    G_rep = test_G_hat.unsqueeze(0).expand(S,B,N,M)
    g_dt_rep = test_g_dt_hat.unsqueeze(0).expand(S,B,M,1)

    reg_robust_curve = []
    rob_robust_curve = []
    reg_violation_curve = []
    rob_violation_curve = []

    for inj_var in INJECTION_SWEEP:
        # 每個sweep點重設同一seed，使不同inj variance使用同一組standardized errors
        torch.manual_seed(SWEEP_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SWEEP_SEED)

        inj_h_dk = h_dk_rep + torch.sqrt(inj_var * h_dk_power) * complex_awgn(h_dk_rep.shape,1.0,DEVICE,h_dk_rep.dtype)
        inj_h_rk = h_rk_rep + torch.sqrt(inj_var * h_rk_power) * complex_awgn(h_rk_rep.shape,1.0,DEVICE,h_rk_rep.dtype)
        inj_G = G_rep + torch.sqrt(inj_var * G_power) * complex_awgn(G_rep.shape,1.0,DEVICE,G_rep.dtype)
        inj_g_dt = g_dt_rep + torch.sqrt(inj_var * g_dt_power) * complex_awgn(g_dt_rep.shape,1.0,DEVICE,g_dt_rep.dtype)

        inj_h_dk_flat = inj_h_dk.reshape(S*B,M,K)
        inj_h_rk_flat = inj_h_rk.reshape(S*B,N,K)
        inj_G_flat = inj_G.reshape(S*B,N,M)
        inj_g_dt_flat = inj_g_dt.reshape(S*B,M,1)

        # REG與ROB共用完全相同的injected channels
        reg_robust,reg_sensing_violation = evaluate_network(
            reg_comm_net,
            reg_W_C_flat,
            reg_W_R_flat,
            inj_h_dk_flat,
            inj_h_rk_flat,
            inj_G_flat,
            inj_g_dt_flat,
            theta_flat,
            S,
            B,
            K,
        )

        rob_robust,rob_sensing_violation = evaluate_network(
            rob_comm_net,
            rob_W_C_flat,
            rob_W_R_flat,
            inj_h_dk_flat,
            inj_h_rk_flat,
            inj_G_flat,
            inj_g_dt_flat,
            theta_flat,
            S,
            B,
            K,
        )

        reg_robust_curve.append(reg_robust)
        rob_robust_curve.append(rob_robust)
        reg_violation_curve.append(reg_sensing_violation)
        rob_violation_curve.append(rob_sensing_violation)

        print(
            f"INJ={inj_var:.3f} | "
            f"REG Robust={reg_robust:.6f}, SenseViolation={100.0 * reg_sensing_violation:.2f}% | "
            f"ROB Robust={rob_robust:.6f}, SenseViolation={100.0 * rob_sensing_violation:.2f}%"
        )

        del inj_h_dk,inj_h_rk,inj_G,inj_g_dt
        del inj_h_dk_flat,inj_h_rk_flat,inj_G_flat,inj_g_dt_flat

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ================================
# 儲存數值
# ================================
inj_array = np.asarray(INJECTION_SWEEP,dtype=np.float64)
reg_robust_array = np.asarray(reg_robust_curve,dtype=np.float64)
rob_robust_array = np.asarray(rob_robust_curve,dtype=np.float64)
reg_violation_array = np.asarray(reg_violation_curve,dtype=np.float64)
rob_violation_array = np.asarray(rob_violation_curve,dtype=np.float64)

np.savez(
    npz_path,
    injection_variance=inj_array,
    reg_robust=reg_robust_array,
    rob_robust=rob_robust_array,
    reg_sensing_violation=reg_violation_array,
    rob_sensing_violation=rob_violation_array,
    reg_sensing_penalty=np.asarray(REG_SENSING_LOSS_WEIGHT,dtype=np.float64),
    rob_sensing_penalty=np.asarray(ROB_SENSING_LOSS_WEIGHT,dtype=np.float64),
)

with open(csv_path,"w",newline="",encoding="utf-8-sig") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow([
        "injection_variance",
        "reg_robust",
        "rob_robust",
        "reg_sensing_violation",
        "rob_sensing_violation",
    ])

    for idx,inj_var in enumerate(INJECTION_SWEEP):
        writer.writerow([
            f"{inj_var:.3f}",
            f"{reg_robust_array[idx]:.8f}",
            f"{rob_robust_array[idx]:.8f}",
            f"{reg_violation_array[idx]:.8f}",
            f"{rob_violation_array[idx]:.8f}",
        ])


# ================================
# 畫一張sweep figure：上下兩個panel
# ================================
fig,axes = plt.subplots(2,1,figsize=(9,10),sharex=True)

axes[0].plot(inj_array,reg_robust_array,marker="o",linewidth=2.0,label=f"REG penalty={REG_SENSING_LOSS_WEIGHT:g}")
axes[0].plot(inj_array,rob_robust_array,marker="o",linewidth=2.0,label=f"ROB penalty={ROB_SENSING_LOSS_WEIGHT:g}")
axes[0].set_ylabel(f"Robust worst-UE rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
axes[0].set_title("Robust Rate vs Injection Variance")
axes[0].grid(True,alpha=0.35)
axes[0].legend()

axes[1].plot(inj_array,100.0 * reg_violation_array,marker="o",linewidth=2.0,label=f"REG penalty={REG_SENSING_LOSS_WEIGHT:g}")
axes[1].plot(inj_array,100.0 * rob_violation_array,marker="o",linewidth=2.0,label=f"ROB penalty={ROB_SENSING_LOSS_WEIGHT:g}")

if "TAR_OUTAGE_QUANTILE" in globals():
    axes[1].axhline(
        100.0 * TAR_OUTAGE_QUANTILE,
        linestyle="--",
        linewidth=1.5,
        label=f"Sensing violation limit={100.0 * TAR_OUTAGE_QUANTILE:.1f}%",
    )

axes[1].set_xlabel("Injection variance")
axes[1].set_ylabel("Sensing violation rate (%)")
axes[1].set_title(f"Sensing Violation vs Injection Variance, threshold={SENSING_SNR_THRESHOLD_DB:g} dB")
axes[1].grid(True,alpha=0.35)
axes[1].legend()

plt.tight_layout()
plt.savefig(fig_path,dpi=300)
plt.close()

print("\n" + "=" * 100)
print("[INJECTION SWEEP FINISHED]")
print("=" * 100)
print(f"CSV saved : {csv_path}")
print(f"NPZ saved : {npz_path}")
print(f"Figure    : {fig_path}")
print("=" * 100)
