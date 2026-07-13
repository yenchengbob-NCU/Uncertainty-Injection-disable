# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from one_timescale_NN import CommNet, RadarNet, ThetaNet

# ================================
# Helpers
# ================================
def beamformers_power_split(W_C, W_R):
    """
    這裡輸入要是正規化後的W_C, W_R
    """
    total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)
    """
    舊版code
    # 先把 W_C, W_R 都變成 direction,消除 raw power 不均問題
    W_C_dir = W_C / (torch.sqrt(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12)
    W_R_dir = W_R / (torch.sqrt(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12)
    """

    # 根據sweep來的power分配
    p_R = total_power * 0.6 
    p_C = total_power * 0.4

    W_R = torch.sqrt(p_R) * W_R
    W_C = torch.sqrt(p_C) * W_C

    return W_C, W_R


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


# ================================
# Main
# ================================
if __name__ == "__main__":
    
    # 簡寫
    S, B, M, N, K = INJECTION_SAMPLES, N_TEST_CHANNELS, TX_ANT, RIS_UNIT, UAV_COMM

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()
    
    # 讀取資料
    test_dataset_path  = os.path.join(DATA_DIR, "dataset_test.npz")

    reg_comm_ckpt  = os.path.join(REG_CKPT_DIR, "one_timescale_comm_reg.ckpt")
    reg_radar_ckpt = os.path.join(REG_CKPT_DIR, "one_timescale_radar_reg.ckpt")
    reg_theta_ckpt = os.path.join(REG_CKPT_DIR, "one_timescale_theta_reg.ckpt")

    rob_comm_ckpt  = os.path.join(ROB_CKPT_DIR, "one_timescale_comm_rob.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR, "one_timescale_radar_rob.ckpt")
    rob_theta_ckpt = os.path.join(ROB_CKPT_DIR, "one_timescale_theta_rob.ckpt")

    test_dataset       = physics_net.load_channel_dataset(test_dataset_path, "test")
    result_dir         = RESULT_DIR
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] test_dataset_path = {test_dataset_path}")

    # 載入 REG 權重
    reg_comm_net  = CommNet().to(DEVICE)
    reg_radar_net = RadarNet().to(DEVICE)
    reg_theta_net = ThetaNet().to(DEVICE)

    reg_comm_net.load_model(reg_comm_ckpt, verbose=False)
    reg_radar_net.load_model(reg_radar_ckpt, verbose=False)
    reg_theta_net.load_model(reg_theta_ckpt, verbose=False)

    reg_comm_net.eval()
    reg_radar_net.eval()
    reg_theta_net.eval()

    # 載入 ROB 權重
    rob_comm_net  = CommNet().to(DEVICE)
    rob_radar_net = RadarNet().to(DEVICE)
    rob_theta_net = ThetaNet().to(DEVICE)

    rob_comm_net.load_model(rob_comm_ckpt, verbose=False)
    rob_radar_net.load_model(rob_radar_ckpt, verbose=False)
    rob_theta_net.load_model(rob_theta_ckpt, verbose=False)

    rob_comm_net.eval()
    rob_radar_net.eval()
    rob_theta_net.eval()

    with torch.no_grad():
        # 載入估測通道
        h_dk_teat = torch.as_tensor(test_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, K)
        h_rk_teat = torch.as_tensor(test_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, N, K)
        G_test    = torch.as_tensor(test_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)      # (B, N, M)
        g_dt_test = torch.as_tensor(test_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, 1)

        # 輸入 REG 網路
        theta_reg   = reg_theta_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)
        W_C_reg_raw = reg_comm_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)
        W_R_reg_raw = reg_radar_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)

        W_C_reg, W_R_reg = beamformers_power_split(W_C_reg_raw, W_R_reg_raw)            # power split 寫死

        # 輸入 ROB 網路
        theta_rob   = rob_theta_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)
        W_C_rob_raw = rob_comm_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)
        W_R_rob_raw = rob_radar_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)

        W_C_rob, W_R_rob = beamformers_power_split(W_C_rob_raw, W_R_rob_raw)            # power split 寫死

        # 使用 .unsqueeze(0) 新增第0維度 供INJ用
        # 使用 .expand       功能上"複製"通道成S份 (實際較複雜但最終是複製)
        h_dk_rep = h_dk_teat.unsqueeze(0).expand(S, B, M, K)
        h_rk_rep = h_rk_teat.unsqueeze(0).expand(S, B, N, K)
        G_rep    = G_test.unsqueeze(0).expand(S, B, N, M)
        g_dt_rep = g_dt_test.unsqueeze(0).expand(S, B, M, 1)

        # 注入不確定：INJECTION_VARIANCE 是相對通道功率
        # noise power = INJECTION_VARIANCE * mean(|channel|^2)

        h_dk_power = torch.mean(torch.abs(h_dk_rep) ** 2, dim=(2, 3), keepdim=True).real   # (S,B,1,1)
        h_rk_power = torch.mean(torch.abs(h_rk_rep) ** 2, dim=(2, 3), keepdim=True).real   # (S,B,1,1)
        G_power    = torch.mean(torch.abs(G_rep)    ** 2, dim=(2, 3), keepdim=True).real   # (S,B,1,1)
        g_dt_power = torch.mean(torch.abs(g_dt_rep) ** 2, dim=(2, 3), keepdim=True).real   # (S,B,1,1)

        # 注入不確定
        h_dk_inj = h_dk_rep + torch.sqrt(INJECTION_VARIANCE * h_dk_power) * complex_awgn(h_dk_rep.shape, 1.0, DEVICE, h_dk_rep.dtype)   # (S,B,M,K)
        h_rk_inj = h_rk_rep + torch.sqrt(INJECTION_VARIANCE * h_rk_power) * complex_awgn(h_rk_rep.shape, 1.0, DEVICE, h_rk_rep.dtype)   # (S,B,N,K)
        G_inj    = G_rep    + torch.sqrt(INJECTION_VARIANCE * G_power)    * complex_awgn(G_rep.shape,    1.0, DEVICE, G_rep.dtype)      # (S,B,N,M)
        g_dt_inj = g_dt_rep + torch.sqrt(INJECTION_VARIANCE * g_dt_power) * complex_awgn(g_dt_rep.shape, 1.0, DEVICE, g_dt_rep.dtype)   # (S,B,M,1)

        # 為了使用(physics_net.)副函式,先將4維壓到3維"注意! 此舉不會破壞5%的抽取"
        h_dk_flat  = h_dk_inj.reshape(S*B, M, K)
        h_rk_flat  = h_rk_inj.reshape(S*B, N, K)
        G_flat     = G_inj.reshape(S*B, N, M)
        g_dt_flat  = g_dt_inj.reshape(S*B, M, 1)

        # ================================
        # REG Eval
        # ================================
        theta_reg_rep = theta_reg.unsqueeze(0).expand(S, B, N)

        W_C_reg_rep = W_C_reg.unsqueeze(0).expand(S, B, M, K)
        W_R_reg_rep = W_R_reg.unsqueeze(0).expand(S, B, M, 1)

        theta_reg_flat = theta_reg_rep.reshape(S*B, N)
        W_C_reg_flat   = W_C_reg_rep.reshape(S*B, M, K)
        W_R_reg_flat   = W_R_reg_rep.reshape(S*B, M, 1)

        H_eff_H_reg_flat = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_reg_flat)

        reg_metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_reg_flat,g_dt_flat,W_C_reg_flat,W_R_reg_flat)
        
        # 計算結果
        reg_rate_inj = reg_metrics_flat["rate"].reshape(S, B, K)                # (S,B,K)

        reg_rate_user_q05 = torch.quantile(reg_rate_inj,OUTAGE_QUANTILE,dim=0)  # (B,K)

        reg_sumrate_q05 = torch.sum(reg_rate_user_q05, dim=1)                   # (B,)

        reg_tail_sumrate        = torch.mean(reg_sumrate_q05)                   # scalar
        reg_rate_user_q05_mean  = torch.mean(reg_rate_user_q05, dim=0)          # (K,)

        reg_target_snr_mean_db = reg_metrics_flat["target_snr_mean_db"]         # scalar

        # ================================
        # ROB Eval
        # ================================
        theta_rob_rep = theta_rob.unsqueeze(0).expand(S, B, N)

        W_C_rob_rep = W_C_rob.unsqueeze(0).expand(S, B, M, K)
        W_R_rob_rep = W_R_rob.unsqueeze(0).expand(S, B, M, 1)

        theta_rob_flat = theta_rob_rep.reshape(S*B, N)
        W_C_rob_flat   = W_C_rob_rep.reshape(S*B, M, K)
        W_R_rob_flat   = W_R_rob_rep.reshape(S*B, M, 1)

        H_eff_H_rob_flat = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_rob_flat)

        rob_metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_rob_flat,g_dt_flat,W_C_rob_flat,W_R_rob_flat)

        rob_rate_inj = rob_metrics_flat["rate"].reshape(S, B, K)                # (S,B,K)

        rob_rate_user_q05 = torch.quantile(rob_rate_inj,OUTAGE_QUANTILE,dim=0)  # (B,K)

        rob_sumrate_q05 = torch.sum(rob_rate_user_q05, dim=1)                   # (B,)

        rob_tail_sumrate        = torch.mean(rob_sumrate_q05)                   # scalar
        rob_rate_user_q05_mean  = torch.mean(rob_rate_user_q05, dim=0)          # (K,)

        rob_target_snr_mean_db = rob_metrics_flat["target_snr_mean_db"]         # scalar

        # 輸出 REG 結果
        print("=" * 90)
        print("[Injection-tail performance: One-timescale REG Net]")
        print("=" * 90)
        print(f"Injection samples S : {S}")
        print(f"Estimated channels B: {B}")
        print(f"Tail quantile       : {OUTAGE_QUANTILE:.2f}")
        print("-" * 90)

        print("[Communication tail metrics]")
        print(f"Robust sum-rate     : {float(reg_tail_sumrate.detach().cpu()):.6f} bps/Hz")
        print(
            f"UE tail rate        : "
            f"{fmt_vec(reg_rate_user_q05_mean.detach().cpu().numpy(), precision=4)} bps/Hz"
        )

        print("-" * 90)
        print("[Target sensing metrics over injected channels]")
        print(f"Target SNR mean dB  : {float(reg_target_snr_mean_db.detach().cpu()):.3f} dB")
        print("=" * 90)

        # 輸出 ROB 結果
        print("=" * 90)
        print("[Injection-tail performance: One-timescale ROB Net]")
        print("=" * 90)
        print(f"Injection samples S : {S}")
        print(f"Estimated channels B: {B}")
        print(f"Tail quantile       : {OUTAGE_QUANTILE:.2f}")
        print("-" * 90)

        print("[Communication tail metrics]")
        print(f"Robust sum-rate     : {float(rob_tail_sumrate.detach().cpu()):.6f} bps/Hz")
        print(
            f"UE tail rate        : "
            f"{fmt_vec(rob_rate_user_q05_mean.detach().cpu().numpy(), precision=4)} bps/Hz"
        )

        print("-" * 90)
        print("[Target sensing metrics over injected channels]")
        print(f"Target SNR mean dB  : {float(rob_target_snr_mean_db.detach().cpu()):.3f} dB")
        print("=" * 90)