# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch

from settings import *
from baseline import beamformers_power_split
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


# ================================
# Main
# ================================
if __name__ == "__main__":

    # ================================
    # 路徑
    # ================================
    test_dataset_path = os.path.join(DATA_DIR,"dataset_test.npz")
    pretrained_theta_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

    reg_comm_ckpt = os.path.join(REG_CKPT_DIR,"two_timescale_comm_reg.ckpt")
    reg_radar_ckpt = os.path.join(REG_CKPT_DIR,"two_timescale_radar_reg.ckpt")
    rob_comm_ckpt = os.path.join(ROB_CKPT_DIR,"two_timescale_comm_rob.ckpt")
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
    test_dataset = reg_comm_net.load_channel_dataset(test_dataset_path,"test")

    print("\n[INFO] 載入 test dataset 與 REG checkpoints ...")
    print(f"[INFO] test_dataset_path      = {test_dataset_path}")
    print(f"[INFO] pretrained_theta_ckpt  = {pretrained_theta_ckpt}")
    print(f"[INFO] reg_comm_ckpt          = {reg_comm_ckpt}")
    print(f"[INFO] reg_radar_ckpt         = {reg_radar_ckpt}")
    print(f"[INFO] rob_comm_ckpt          = {rob_comm_ckpt}")
    print(f"[INFO] rob_radar_ckpt         = {rob_radar_ckpt}")

    with torch.no_grad():
        # ================================================================
        # 1. Test estimated channels
        # ================================================================
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

        # ================================================================
        # 2. 使用 estimated channels 設計 REG theta、W_C、W_R
        # ================================================================
        test_theta = pretrained_theta_net(test_h_dk_hat,test_h_rk_hat,test_G_hat,test_g_dt_hat)

        reg_H_eff_H_hat = reg_comm_net.compute_effective_channel(
            test_h_dk_hat,
            test_h_rk_hat,
            test_G_hat,
            test_theta,
        )

        reg_W_C_dir = reg_comm_net(reg_H_eff_H_hat)
        reg_W_R_dir = reg_radar_net(reg_H_eff_H_hat)

        if reg_W_C_dir.shape != (B,M,K):
            raise ValueError(f"reg_W_C_dir shape error: expected {(B,M,K)}, got {tuple(reg_W_C_dir.shape)}")

        if reg_W_R_dir.shape != (B,M,R):
            raise ValueError(f"reg_W_R_dir shape error: expected {(B,M,R)}, got {tuple(reg_W_R_dir.shape)}")

        reg_W_C,reg_W_R = beamformers_power_split(reg_W_C_dir,reg_W_R_dir)

        # ================================================================
        # 2.使用相同 pretrained theta 設計 ROB W_C、W_R
        # ================================================================
        rob_H_eff_H_hat = rob_comm_net.compute_effective_channel(
            test_h_dk_hat,
            test_h_rk_hat,
            test_G_hat,
            test_theta,
        )

        rob_W_C_dir = rob_comm_net(rob_H_eff_H_hat)
        rob_W_R_dir = rob_radar_net(rob_H_eff_H_hat)

        if rob_W_C_dir.shape != (B,M,K):
            raise ValueError(f"rob_W_C_dir shape error: expected {(B,M,K)}, got {tuple(rob_W_C_dir.shape)}")

        if rob_W_R_dir.shape != (B,M,R):
            raise ValueError(f"rob_W_R_dir shape error: expected {(B,M,R)}, got {tuple(rob_W_R_dir.shape)}")

        rob_W_C,rob_W_R = beamformers_power_split(rob_W_C_dir,rob_W_R_dir)

        # ================================================================
        # 3. 一次性建立完整 injection channels，不使用 chunk
        # ================================================================
        inj_h_dk_rep = test_h_dk_hat.unsqueeze(0).expand(S,B,M,K)
        inj_h_rk_rep = test_h_rk_hat.unsqueeze(0).expand(S,B,N,K)
        inj_G_rep = test_G_hat.unsqueeze(0).expand(S,B,N,M)
        inj_g_dt_rep = test_g_dt_hat.unsqueeze(0).expand(S,B,M,1)

        # 每筆 estimated channel block 的 empirical mean power
        inj_h_dk_power = torch.mean(torch.abs(inj_h_dk_rep) ** 2,dim=(2,3),keepdim=True).real
        inj_h_rk_power = torch.mean(torch.abs(inj_h_rk_rep) ** 2,dim=(2,3),keepdim=True).real
        inj_G_power = torch.mean(torch.abs(inj_G_rep) ** 2,dim=(2,3),keepdim=True).real
        inj_g_dt_power = torch.mean(torch.abs(inj_g_dt_rep) ** 2,dim=(2,3),keepdim=True).real

        # X_inj = X_hat + sqrt(INJECTION_VARIANCE * mean|X_hat|^2) * CN(0,1)
        inj_h_dk = inj_h_dk_rep + torch.sqrt(INJECTION_VARIANCE * inj_h_dk_power) * complex_awgn(inj_h_dk_rep.shape,1.0,DEVICE,inj_h_dk_rep.dtype)
        inj_h_rk = inj_h_rk_rep + torch.sqrt(INJECTION_VARIANCE * inj_h_rk_power) * complex_awgn(inj_h_rk_rep.shape,1.0,DEVICE,inj_h_rk_rep.dtype)
        inj_G = inj_G_rep + torch.sqrt(INJECTION_VARIANCE * inj_G_power) * complex_awgn(inj_G_rep.shape,1.0,DEVICE,inj_G_rep.dtype)
        inj_g_dt = inj_g_dt_rep + torch.sqrt(INJECTION_VARIANCE * inj_g_dt_power) * complex_awgn(inj_g_dt_rep.shape,1.0,DEVICE,inj_g_dt_rep.dtype)

        # ================================================================
        # 4. 對同一 estimated channel 的 S 筆 injection
        #    固定使用相同 theta、REG W_C、REG W_R
        # ================================================================
        test_theta_rep = test_theta.unsqueeze(0).expand(S,B,N)
        reg_W_C_rep = reg_W_C.unsqueeze(0).expand(S,B,M,K)
        reg_W_R_rep = reg_W_R.unsqueeze(0).expand(S,B,M,R)

        # 壓平 S、B 維度，供共用 physics functions 使用
        inj_h_dk_flat = inj_h_dk.reshape(S*B,M,K)
        inj_h_rk_flat = inj_h_rk.reshape(S*B,N,K)
        inj_G_flat = inj_G.reshape(S*B,N,M)
        inj_g_dt_flat = inj_g_dt.reshape(S*B,M,1)

        test_theta_flat = test_theta_rep.reshape(S*B,N)
        reg_W_C_flat = reg_W_C_rep.reshape(S*B,M,K)
        reg_W_R_flat = reg_W_R_rep.reshape(S*B,M,R)

        # 使用 injected channels 重新組成 effective channel
        reg_H_eff_H_inj_flat = reg_comm_net.compute_effective_channel(
            inj_h_dk_flat,
            inj_h_rk_flat,
            inj_G_flat,
            test_theta_flat,
        )

        reg_metrics_flat = reg_comm_net.compute_isac_batch_performance(
            reg_H_eff_H_inj_flat,
            inj_g_dt_flat,
            reg_W_C_flat,
            reg_W_R_flat,
        )

        # ================================================================
        # 5. 還原 shape
        # ================================================================
        reg_allUE_rate = reg_metrics_flat["rate"].reshape(S,B,K)
        reg_target_snr = reg_metrics_flat["target_snr"].reshape(S,B)

        reg_signal = reg_metrics_flat["signal"].reshape(S,B,K)
        reg_comm_interf = reg_metrics_flat["comm_interf"].reshape(S,B,K)
        reg_radar_interf = reg_metrics_flat["radar_interf"].reshape(S,B,K)
        reg_noise = reg_metrics_flat["noise"]

        # ================================================================
        # 6. Canonical robust metric
        #
        # min over K
        # -> Q0.05 over S
        # -> mean over B
        # ================================================================
        reg_worstUE_rate = torch.min(reg_allUE_rate,dim=2).values             # (S,B)

        reg_robust_per_channel = torch.quantile(
            reg_worstUE_rate,
            OUTAGE_QUANTILE,
            dim=0,
        )                                                                      # (B,)

        reg_robust = torch.mean(reg_robust_per_channel)                        # scalar

        # 每個 UE 的 Q0.05 僅作診斷，不作為 objective
        reg_user_rate_q05 = torch.quantile(reg_allUE_rate,OUTAGE_QUANTILE,dim=0)
        reg_user_rate_q05_mean = torch.mean(reg_user_rate_q05,dim=0)

        # Target SNR：先對所有 S、B 做 linear mean，再轉 dB
        reg_target_snr_mean = torch.mean(reg_target_snr)
        reg_target_snr_mean_db = 10.0 * torch.log10(reg_target_snr_mean.clamp_min(1e-12))

        # REG sensing violation：逐筆檢查全部 S*B 筆 injected target SNR
        reg_sensing_violation_mask = reg_target_snr < SENSING_SNR_THRESHOLD       # (S,B), bool
        reg_sensing_violation_count = torch.sum(reg_sensing_violation_mask)       # scalar
        reg_sensing_total_count = reg_target_snr.numel()                          # S*B
        reg_sensing_violation_rate = reg_sensing_violation_count.float() / reg_sensing_total_count
        reg_sensing_violation_percent = 100.0 * reg_sensing_violation_rate

        # SINR components：對所有 injected channels 平均
        reg_signal_mean = torch.mean(reg_signal,dim=(0,1))
        reg_comm_interf_mean = torch.mean(reg_comm_interf,dim=(0,1))
        reg_radar_interf_mean = torch.mean(reg_radar_interf,dim=(0,1))

        # ================================================================
        # ROB robust evaluation
        #
        # 與 REG 共用完全相同的：
        #     inj_h_dk_flat
        #     inj_h_rk_flat
        #     inj_G_flat
        #     inj_g_dt_flat
        #     test_theta_flat
        #
        # 不重新產生 injection errors。
        # ================================================================
        rob_W_C_rep = rob_W_C.unsqueeze(0).expand(S,B,M,K)
        rob_W_R_rep = rob_W_R.unsqueeze(0).expand(S,B,M,R)

        rob_W_C_flat = rob_W_C_rep.reshape(S*B,M,K)
        rob_W_R_flat = rob_W_R_rep.reshape(S*B,M,R)

        # 使用相同 injected channels 與相同 pretrained theta
        rob_H_eff_H_inj_flat = rob_comm_net.compute_effective_channel(
            inj_h_dk_flat,
            inj_h_rk_flat,
            inj_G_flat,
            test_theta_flat,
        )

        rob_metrics_flat = rob_comm_net.compute_isac_batch_performance(
            rob_H_eff_H_inj_flat,
            inj_g_dt_flat,
            rob_W_C_flat,
            rob_W_R_flat,
        )

        # ================================================================
        # 還原 ROB tensors
        # ================================================================
        rob_allUE_rate = rob_metrics_flat["rate"].reshape(S,B,K)
        rob_target_snr = rob_metrics_flat["target_snr"].reshape(S,B)

        rob_signal = rob_metrics_flat["signal"].reshape(S,B,K)
        rob_comm_interf = rob_metrics_flat["comm_interf"].reshape(S,B,K)
        rob_radar_interf = rob_metrics_flat["radar_interf"].reshape(S,B,K)
        rob_noise = rob_metrics_flat["noise"]

        # ================================================================
        # Canonical ROB robust metric
        #
        # min over K
        # -> Q0.05 over S
        # -> mean over B
        # ================================================================
        rob_worstUE_rate = torch.min(rob_allUE_rate,dim=2).values                  # (S,B)

        rob_robust_per_channel = torch.quantile(
            rob_worstUE_rate,
            OUTAGE_QUANTILE,
            dim=0,
        )                                                                          # (B,)

        rob_robust = torch.mean(rob_robust_per_channel)                            # scalar

        # 每個 UE 的 Q0.05，僅作診斷
        rob_user_rate_q05 = torch.quantile(
            rob_allUE_rate,
            OUTAGE_QUANTILE,
            dim=0,
        )                                                                          # (B,K)

        rob_user_rate_q05_mean = torch.mean(
            rob_user_rate_q05,
            dim=0,
        )                                                                          # (K,)

        # Target SNR：先對所有 S、B 做 linear mean，再轉 dB
        rob_target_snr_mean = torch.mean(rob_target_snr)
        rob_target_snr_mean_db = 10.0 * torch.log10(
            rob_target_snr_mean.clamp_min(1e-12)
        )

        # ROB sensing violation：逐筆檢查全部 S*B 筆 injected target SNR
        rob_sensing_violation_mask = rob_target_snr < SENSING_SNR_THRESHOLD       # (S,B), bool
        rob_sensing_violation_count = torch.sum(rob_sensing_violation_mask)       # scalar
        rob_sensing_total_count = rob_target_snr.numel()                          # S*B
        rob_sensing_violation_rate = rob_sensing_violation_count.float() / rob_sensing_total_count
        rob_sensing_violation_percent = 100.0 * rob_sensing_violation_rate

        # SINR components：對所有 injected channels 平均
        rob_signal_mean = torch.mean(rob_signal,dim=(0,1))
        rob_comm_interf_mean = torch.mean(rob_comm_interf,dim=(0,1))
        rob_radar_interf_mean = torch.mean(rob_radar_interf,dim=(0,1))

        # ================================================================
        # 7. Print REG robust result
        # ================================================================
        print("\n" + "=" * 90)
        print("[REG ROBUST TEST EVALUATION]")
        print("=" * 90)
        print(f"Test estimated channels B             : {B}")
        print(f"Injection samples S                    : {S}")
        print(f"Injection variance                     : {INJECTION_VARIANCE}")
        print(f"Tail quantile                          : {OUTAGE_QUANTILE:.2f}")
        print("-" * 90)
        print(f"REG robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} : {float(reg_robust.detach().cpu()):.6f} bps/Hz")
        print(f"REG per-UE rate Q{OUTAGE_QUANTILE:.2f} mean          : {fmt_vec_sci(reg_user_rate_q05_mean.detach().cpu().numpy(),precision=4)} bps/Hz")
        print(f"REG robust target SNR mean             : {float(reg_target_snr_mean_db.detach().cpu()):.3f} dB")
        print(f"REG sensing SNR threshold              : {SENSING_SNR_THRESHOLD_DB:.3f} dB")
        print(f"REG sensing violation count            : {int(reg_sensing_violation_count.detach().cpu().item())} / {reg_sensing_total_count}")
        print(f"REG sensing violation percentage       : {float(reg_sensing_violation_percent.detach().cpu()):.4f}%")
        print(f"REG robust signal power                : {fmt_vec_sci(reg_signal_mean.detach().cpu().numpy(),precision=3)}")
        print(f"REG robust communication interference  : {fmt_vec_sci(reg_comm_interf_mean.detach().cpu().numpy(),precision=3)}")
        print(f"REG robust radar interference          : {fmt_vec_sci(reg_radar_interf_mean.detach().cpu().numpy(),precision=3)}")
        print(f"REG robust noise power                 : {float(reg_noise.detach().cpu()):.3e}")
        print("=" * 90)

        print("\n" + "=" * 90)
        print("[ROB ROBUST TEST EVALUATION]")
        print("=" * 90)
        print(f"Test estimated channels B             : {B}")
        print(f"Injection samples S                    : {S}")
        print(f"Injection variance                     : {INJECTION_VARIANCE}")
        print(f"Tail quantile                          : {OUTAGE_QUANTILE:.2f}")
        print("-" * 90)
        print(f"ROB robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} : {float(rob_robust.detach().cpu()):.6f} bps/Hz")
        print(f"ROB per-UE rate Q{OUTAGE_QUANTILE:.2f} mean          : {fmt_vec_sci(rob_user_rate_q05_mean.detach().cpu().numpy(),precision=4)} bps/Hz")
        print(f"ROB robust target SNR mean             : {float(rob_target_snr_mean_db.detach().cpu()):.3f} dB")
        print(f"ROB sensing SNR threshold              : {SENSING_SNR_THRESHOLD_DB:.3f} dB")
        print(f"ROB sensing violation count            : {int(rob_sensing_violation_count.detach().cpu().item())} / {rob_sensing_total_count}")
        print(f"ROB sensing violation percentage       : {float(rob_sensing_violation_percent.detach().cpu()):.4f}%")
        print(f"ROB robust signal power                : {fmt_vec_sci(rob_signal_mean.detach().cpu().numpy(),precision=3)}")
        print(f"ROB robust communication interference  : {fmt_vec_sci(rob_comm_interf_mean.detach().cpu().numpy(),precision=3)}")
        print(f"ROB robust radar interference          : {fmt_vec_sci(rob_radar_interf_mean.detach().cpu().numpy(),precision=3)}")
        print(f"ROB robust noise power                 : {float(rob_noise.detach().cpu()):.3e}")
        print("=" * 90)