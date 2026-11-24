# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch

from settings import *
from two_timescale_NN import CommNet, RadarNet, ThetaNet


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


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


def project_beamformers_to_tx_power(W_C,W_R):
    W_C_power = torch.sum(torch.abs(W_C)**2,dim=(1,2)).real
    W_R_power = torch.sum(torch.abs(W_R)**2,dim=(1,2)).real
    total_power = W_C_power + W_R_power

    scale = torch.sqrt(TRANSMIT_POWER_TOTAL / (total_power + 1e-12))
    scale = torch.minimum(scale,torch.ones_like(scale))

    W_C = W_C * scale[:,None,None]
    W_R = W_R * scale[:,None,None]

    return W_C,W_R


# ================================
# Main
# ================================
if __name__ == "__main__":

    # ================================
    # 路徑
    # ================================
    test_dataset_path     = os.path.join(DATA_DIR,"dataset_test.npz")
    pretrained_theta_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

    reg_comm_ckpt  = os.path.join(REG_CKPT_DIR,"two_timescale_comm_reg.ckpt")
    reg_radar_ckpt = os.path.join(REG_CKPT_DIR,"two_timescale_radar_reg.ckpt")
    rob_comm_ckpt  = os.path.join(ROB_CKPT_DIR,"two_timescale_comm_rob.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR,"two_timescale_radar_rob.ckpt")

    for ckpt_path in [pretrained_theta_ckpt,reg_comm_ckpt,reg_radar_ckpt,rob_comm_ckpt,rob_radar_ckpt,]:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    # ================================
    # 建立並載入網路
    # ================================
    pretrained_theta_net = ThetaNet().to(DEVICE)

    reg_comm_net = CommNet().to(DEVICE)
    reg_radar_net = RadarNet().to(DEVICE)

    pretrained_theta_net.load_model(pretrained_theta_ckpt,strict=True,verbose=True)
    reg_comm_net.load_model(reg_comm_ckpt,strict=True,verbose=True)
    reg_radar_net.load_model(reg_radar_ckpt,strict=True,verbose=True)

    pretrained_theta_net.eval()
    reg_comm_net.eval()
    reg_radar_net.eval()

    rob_comm_net = CommNet().to(DEVICE)
    rob_radar_net = RadarNet().to(DEVICE)

    rob_comm_net.load_model(rob_comm_ckpt,strict=True,verbose=True)
    rob_radar_net.load_model(rob_radar_ckpt,strict=True,verbose=True)

    rob_comm_net.eval()
    rob_radar_net.eval()
    # ================================
    # 讀取 test dataset
    # ================================
    with np.load(test_dataset_path) as data:
        test_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

    test_trajectories = test_dataset["h_dk"].shape[0]
    test_channels     = test_dataset["h_dk"].shape[1]

    print("\n[INFO] 載入 test dataset 與 checkpoints ...")
    print(f"[INFO] test_dataset_path      = {test_dataset_path}")
    print(f"[INFO] pretrained_theta_ckpt  = {pretrained_theta_ckpt}")
    print(f"[INFO] reg_comm_ckpt          = {reg_comm_ckpt}")
    print(f"[INFO] reg_radar_ckpt         = {reg_radar_ckpt}")
    print(f"[INFO] rob_comm_ckpt          = {rob_comm_ckpt}")
    print(f"[INFO] rob_radar_ckpt         = {rob_radar_ckpt}")
    print(f"[INFO] test trajectories      = {test_trajectories}")
    print(f"[INFO] test channels          = {test_channels}")
    print(f"[INFO] h_dk shape             = {test_dataset['h_dk'].shape}")
    print(f"[INFO] h_rk shape             = {test_dataset['h_rk'].shape}")
    print(f"[INFO] G shape                = {test_dataset['G'].shape}")
    print(f"[INFO] g_dt shape             = {test_dataset['g_dt'].shape}")

    with torch.no_grad():
        # ================================================================
        # 1. 建立 test two-timescale blocks
        # ================================================================
        channel_ids    = np.arange(0,test_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)    # 0,10,...,990
        trajectory_ids = np.repeat(np.arange(test_trajectories),len(channel_ids))
        channel_ids    = np.tile(channel_ids,test_trajectories)

        B_block = len(channel_ids)

        # ================================================================
        # 2. 每個 block 的 start slot 使用 ThetaNet 產生 theta
        # ================================================================
        h_dk_start = torch.as_tensor(test_dataset["h_dk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_start = torch.as_tensor(test_dataset["h_rk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
        G_start    = torch.as_tensor(test_dataset["G"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_start = torch.as_tensor(test_dataset["g_dt"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)

        test_theta = pretrained_theta_net(h_dk_start,h_rk_start,G_start,g_dt_start)   # (B_block,N)

        # ================================================================
        # 3. 每個 theta 固定使用 LT_BLOCK_SIZE 個 slots
        # ================================================================
        block_ids = channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]

        test_h_dk_hat = torch.as_tensor(test_dataset["h_dk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
        test_h_rk_hat = torch.as_tensor(test_dataset["h_rk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
        test_G_hat    = torch.as_tensor(test_dataset["G"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
        test_g_dt_hat = torch.as_tensor(test_dataset["g_dt"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)

        B = B_block * LT_BLOCK_SIZE
        M = TX_ANT
        N = RIS_UNIT
        K = UAV_COMM
        R = RADAR_STREAMS
        S = INJECTION_SAMPLES

        # (B_block,L,...) -> (B,...)
        test_h_dk_hat = test_h_dk_hat.reshape(B,M,K)
        test_h_rk_hat = test_h_rk_hat.reshape(B,N,K)
        test_G_hat    = test_G_hat.reshape(B,N,M)
        test_g_dt_hat = test_g_dt_hat.reshape(B,M,1)

        # 每個 theta 固定使用 LT_BLOCK_SIZE 個 slots
        test_theta = test_theta[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B,N)

        print(f"[INFO] RIS blocks             = {B_block}")
        print(f"[INFO] evaluated test slots   = {B}")

        # ================================================================
        # 4. 使用相同 pretrained theta 設計 REG W_C、W_R
        # ================================================================
        reg_H_eff_H_hat = reg_comm_net.compute_effective_channel(test_h_dk_hat,test_h_rk_hat,test_G_hat,test_theta)

        reg_W_C_dir = reg_comm_net(reg_H_eff_H_hat,test_g_dt_hat)
        reg_W_R_dir = reg_radar_net(reg_H_eff_H_hat,test_g_dt_hat)

        if reg_W_C_dir.shape != (B,M,K):
            raise ValueError(f"reg_W_C_dir shape error: expected {(B,M,K)}, got {tuple(reg_W_C_dir.shape)}")

        if reg_W_R_dir.shape != (B,M,R):
            raise ValueError(f"reg_W_R_dir shape error: expected {(B,M,R)}, got {tuple(reg_W_R_dir.shape)}")

        reg_W_C,reg_W_R = project_beamformers_to_tx_power(reg_W_C_dir,reg_W_R_dir)

        # ================================================================
        # 5. 使用相同 pretrained theta 設計 ROB W_C、W_R
        # ================================================================
        rob_H_eff_H_hat = rob_comm_net.compute_effective_channel(test_h_dk_hat,test_h_rk_hat,test_G_hat,test_theta)

        rob_W_C_dir = rob_comm_net(rob_H_eff_H_hat,test_g_dt_hat)
        rob_W_R_dir = rob_radar_net(rob_H_eff_H_hat,test_g_dt_hat)

        if rob_W_C_dir.shape != (B,M,K):
            raise ValueError(f"rob_W_C_dir shape error: expected {(B,M,K)}, got {tuple(rob_W_C_dir.shape)}")

        if rob_W_R_dir.shape != (B,M,R):
            raise ValueError(f"rob_W_R_dir shape error: expected {(B,M,R)}, got {tuple(rob_W_R_dir.shape)}")

        rob_W_C,rob_W_R = project_beamformers_to_tx_power(rob_W_C_dir,rob_W_R_dir)

        # ================================================================
        # 6. Robust evaluation
        #    REG / ROB 共用完全相同 injection errors
        # ================================================================
        EVAL_CHUNK_SIZE = 200

        reg_robust_all = []
        rob_robust_all = []

        reg_user_rate_q05_sum = torch.zeros(K,device=DEVICE)
        rob_user_rate_q05_sum = torch.zeros(K,device=DEVICE)

        reg_target_snr_sum = 0.0
        rob_target_snr_sum = 0.0

        reg_sensing_violation_count = 0
        rob_sensing_violation_count = 0
        sensing_total_count = 0

        reg_signal_sum = torch.zeros(K,device=DEVICE)
        reg_comm_interf_sum = torch.zeros(K,device=DEVICE)
        reg_radar_interf_sum = torch.zeros(K,device=DEVICE)

        rob_signal_sum = torch.zeros(K,device=DEVICE)
        rob_comm_interf_sum = torch.zeros(K,device=DEVICE)
        rob_radar_interf_sum = torch.zeros(K,device=DEVICE)

        metric_total_count = 0

        for start in range(0,B,EVAL_CHUNK_SIZE):
            end = min(start+EVAL_CHUNK_SIZE,B)
            C = end-start

            h_dk_hat = test_h_dk_hat[start:end]
            h_rk_hat = test_h_rk_hat[start:end]
            G_hat    = test_G_hat[start:end]
            g_dt_hat = test_g_dt_hat[start:end]

            theta_chunk = test_theta[start:end]

            reg_W_C_chunk = reg_W_C[start:end]
            reg_W_R_chunk = reg_W_R[start:end]

            rob_W_C_chunk = rob_W_C[start:end]
            rob_W_R_chunk = rob_W_R[start:end]

            # ============================================================
            # Injection channels
            # ============================================================
            inj_h_dk_rep = h_dk_hat.unsqueeze(0).expand(S,C,M,K)
            inj_h_rk_rep = h_rk_hat.unsqueeze(0).expand(S,C,N,K)
            inj_G_rep    = G_hat.unsqueeze(0).expand(S,C,N,M)
            inj_g_dt_rep = g_dt_hat.unsqueeze(0).expand(S,C,M,1)

            inj_h_dk_power = torch.mean(torch.abs(inj_h_dk_rep)**2,dim=(2,3),keepdim=True).real
            inj_h_rk_power = torch.mean(torch.abs(inj_h_rk_rep)**2,dim=(2,3),keepdim=True).real
            inj_G_power    = torch.mean(torch.abs(inj_G_rep)**2,dim=(2,3),keepdim=True).real
            inj_g_dt_power = torch.mean(torch.abs(inj_g_dt_rep)**2,dim=(2,3),keepdim=True).real

            inj_h_dk = inj_h_dk_rep + torch.sqrt(INJECTION_VARIANCE*inj_h_dk_power) * complex_awgn(inj_h_dk_rep.shape,1.0,DEVICE,inj_h_dk_rep.dtype)
            inj_h_rk = inj_h_rk_rep + torch.sqrt(INJECTION_VARIANCE*inj_h_rk_power) * complex_awgn(inj_h_rk_rep.shape,1.0,DEVICE,inj_h_rk_rep.dtype)
            inj_G    = inj_G_rep + torch.sqrt(INJECTION_VARIANCE*inj_G_power) * complex_awgn(inj_G_rep.shape,1.0,DEVICE,inj_G_rep.dtype)
            inj_g_dt = inj_g_dt_rep + torch.sqrt(INJECTION_VARIANCE*inj_g_dt_power) * complex_awgn(inj_g_dt_rep.shape,1.0,DEVICE,inj_g_dt_rep.dtype)

            # REG / ROB 共用完全相同的 injection channels
            inj_h_dk_flat = inj_h_dk.reshape(S*C,M,K)
            inj_h_rk_flat = inj_h_rk.reshape(S*C,N,K)
            inj_G_flat    = inj_G.reshape(S*C,N,M)
            inj_g_dt_flat = inj_g_dt.reshape(S*C,M,1)

            theta_flat = theta_chunk.unsqueeze(0).expand(S,C,N).reshape(S*C,N)

            reg_W_C_flat = reg_W_C_chunk.unsqueeze(0).expand(S,C,M,K).reshape(S*C,M,K)
            reg_W_R_flat = reg_W_R_chunk.unsqueeze(0).expand(S,C,M,R).reshape(S*C,M,R)

            rob_W_C_flat = rob_W_C_chunk.unsqueeze(0).expand(S,C,M,K).reshape(S*C,M,K)
            rob_W_R_flat = rob_W_R_chunk.unsqueeze(0).expand(S,C,M,R).reshape(S*C,M,R)

            # ============================================================
            # REG
            # ============================================================
            reg_H_eff_H_inj = reg_comm_net.compute_effective_channel(inj_h_dk_flat,inj_h_rk_flat,inj_G_flat,theta_flat)
            reg_metrics = reg_comm_net.compute_isac_batch_performance(reg_H_eff_H_inj,inj_g_dt_flat,reg_W_C_flat,reg_W_R_flat)

            reg_allUE_rate = reg_metrics["rate"].reshape(S,C,K)
            reg_target_snr = reg_metrics["target_snr"].reshape(S,C)
            reg_target_snr_db = reg_metrics["target_snr_db"].reshape(S,C)

            reg_worstUE_rate = torch.min(reg_allUE_rate,dim=2).values
            reg_robust_per_channel = torch.quantile(reg_worstUE_rate,OUTAGE_QUANTILE,dim=0)
            reg_robust_all.append(reg_robust_per_channel.cpu())

            reg_user_rate_q05 = torch.quantile(reg_allUE_rate,OUTAGE_QUANTILE,dim=0)
            reg_user_rate_q05_sum += torch.sum(reg_user_rate_q05,dim=0)

            reg_target_snr_sum += float(torch.sum(reg_target_snr).cpu())
            reg_sensing_violation_count += int(torch.sum(reg_target_snr_db < SENSING_SNR_THRESHOLD_DB).cpu())

            reg_signal_sum += torch.sum(reg_metrics["signal"].reshape(S,C,K),dim=(0,1))
            reg_comm_interf_sum += torch.sum(reg_metrics["comm_interf"].reshape(S,C,K),dim=(0,1))
            reg_radar_interf_sum += torch.sum(reg_metrics["radar_interf"].reshape(S,C,K),dim=(0,1))

            # ============================================================
            # ROB
            # ============================================================
            rob_H_eff_H_inj = rob_comm_net.compute_effective_channel(inj_h_dk_flat,inj_h_rk_flat,inj_G_flat,theta_flat)
            rob_metrics = rob_comm_net.compute_isac_batch_performance(rob_H_eff_H_inj,inj_g_dt_flat,rob_W_C_flat,rob_W_R_flat)

            rob_allUE_rate = rob_metrics["rate"].reshape(S,C,K)
            rob_target_snr = rob_metrics["target_snr"].reshape(S,C)
            rob_target_snr_db = rob_metrics["target_snr_db"].reshape(S,C)

            rob_worstUE_rate = torch.min(rob_allUE_rate,dim=2).values
            rob_robust_per_channel = torch.quantile(rob_worstUE_rate,OUTAGE_QUANTILE,dim=0)
            rob_robust_all.append(rob_robust_per_channel.cpu())

            rob_user_rate_q05 = torch.quantile(rob_allUE_rate,OUTAGE_QUANTILE,dim=0)
            rob_user_rate_q05_sum += torch.sum(rob_user_rate_q05,dim=0)

            rob_target_snr_sum += float(torch.sum(rob_target_snr).cpu())
            rob_sensing_violation_count += int(torch.sum(rob_target_snr_db < SENSING_SNR_THRESHOLD_DB).cpu())

            rob_signal_sum += torch.sum(rob_metrics["signal"].reshape(S,C,K),dim=(0,1))
            rob_comm_interf_sum += torch.sum(rob_metrics["comm_interf"].reshape(S,C,K),dim=(0,1))
            rob_radar_interf_sum += torch.sum(rob_metrics["radar_interf"].reshape(S,C,K),dim=(0,1))

            sensing_total_count += S*C
            metric_total_count += S*C

        # ================================================================
        # 7. Final robust metrics
        # ================================================================
        reg_robust = torch.mean(torch.cat(reg_robust_all))
        rob_robust = torch.mean(torch.cat(rob_robust_all))

        reg_user_rate_q05_mean = reg_user_rate_q05_sum / B
        rob_user_rate_q05_mean = rob_user_rate_q05_sum / B

        reg_target_snr_mean = reg_target_snr_sum / sensing_total_count
        rob_target_snr_mean = rob_target_snr_sum / sensing_total_count

        reg_target_snr_mean_db = 10.0*np.log10(max(reg_target_snr_mean,1e-12))
        rob_target_snr_mean_db = 10.0*np.log10(max(rob_target_snr_mean,1e-12))

        reg_sensing_violation_percent = 100.0*reg_sensing_violation_count/sensing_total_count
        rob_sensing_violation_percent = 100.0*rob_sensing_violation_count/sensing_total_count

        reg_signal_mean = reg_signal_sum / metric_total_count
        reg_comm_interf_mean = reg_comm_interf_sum / metric_total_count
        reg_radar_interf_mean = reg_radar_interf_sum / metric_total_count

        rob_signal_mean = rob_signal_sum / metric_total_count
        rob_comm_interf_mean = rob_comm_interf_sum / metric_total_count
        rob_radar_interf_mean = rob_radar_interf_sum / metric_total_count

        # ================================================================
        # 8. Print REG robust result
        # ================================================================
        print("\n" + "=" * 90)
        print("[REG ROBUST TEST EVALUATION]")
        print("=" * 90)
        print(f"Test trajectories                     : {test_trajectories}")
        print(f"Test slots                            : {B}")
        print(f"RIS blocks                            : {B_block}")
        print(f"Injection samples S                   : {S}")
        print(f"Injection variance                    : {INJECTION_VARIANCE}")
        print(f"Tail quantile                         : {OUTAGE_QUANTILE:.2f}")
        print("-" * 90)
        print(f"REG robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} : {float(reg_robust):.6f} bps/Hz")
        print(f"REG per-UE rate Q{OUTAGE_QUANTILE:.2f} mean          : {fmt_vec_sci(reg_user_rate_q05_mean.cpu().numpy(),precision=4)} bps/Hz")
        print(f"REG robust target SNR mean             : {reg_target_snr_mean_db:.3f} dB")
        print(f"REG sensing SNR threshold              : {SENSING_SNR_THRESHOLD_DB:.3f} dB")
        print(f"REG sensing violation count            : {reg_sensing_violation_count} / {sensing_total_count}")
        print(f"REG sensing violation percentage       : {reg_sensing_violation_percent:.4f}%")
        print(f"REG robust signal power                : {fmt_vec_sci(reg_signal_mean.cpu().numpy(),precision=3)}")
        print(f"REG robust communication interference  : {fmt_vec_sci(reg_comm_interf_mean.cpu().numpy(),precision=3)}")
        print(f"REG robust radar interference          : {fmt_vec_sci(reg_radar_interf_mean.cpu().numpy(),precision=3)}")
        print(f"REG robust noise power                 : {float(NOISE_POWER):.3e}")
        print("=" * 90)

        print("\n" + "=" * 90)
        print("[ROB ROBUST TEST EVALUATION]")
        print("=" * 90)
        print(f"Test trajectories                     : {test_trajectories}")
        print(f"Test slots                            : {B}")
        print(f"RIS blocks                            : {B_block}")
        print(f"Injection samples S                   : {S}")
        print(f"Injection variance                    : {INJECTION_VARIANCE}")
        print(f"Tail quantile                         : {OUTAGE_QUANTILE:.2f}")
        print("-" * 90)
        print(f"ROB robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} : {float(rob_robust):.6f} bps/Hz")
        print(f"ROB per-UE rate Q{OUTAGE_QUANTILE:.2f} mean          : {fmt_vec_sci(rob_user_rate_q05_mean.cpu().numpy(),precision=4)} bps/Hz")
        print(f"ROB robust target SNR mean             : {rob_target_snr_mean_db:.3f} dB")
        print(f"ROB sensing SNR threshold              : {SENSING_SNR_THRESHOLD_DB:.3f} dB")
        print(f"ROB sensing violation count            : {rob_sensing_violation_count} / {sensing_total_count}")
        print(f"ROB sensing violation percentage       : {rob_sensing_violation_percent:.4f}%")
        print(f"ROB robust signal power                : {fmt_vec_sci(rob_signal_mean.cpu().numpy(),precision=3)}")
        print(f"ROB robust communication interference  : {fmt_vec_sci(rob_comm_interf_mean.cpu().numpy(),precision=3)}")
        print(f"ROB robust radar interference          : {fmt_vec_sci(rob_radar_interf_mean.cpu().numpy(),precision=3)}")
        print(f"ROB robust noise power                 : {float(NOISE_POWER):.3e}")
        print("=" * 90)
