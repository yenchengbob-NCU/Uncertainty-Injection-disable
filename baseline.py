# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from two_timescale_NN import CommNet

Debug = True                        # 終端印出檢查

RZF_LAMBDA = 1e-9                    # 設定為 0 就是ZF
# ============================================================
# Helpers
# ============================================================
def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


def make_random_ris(B):
    """
    產生B組random RIS phase
    """
    # 產生實數向量,代表每個element相位旋轉角度
    phase = 2.0 * math.pi * torch.rand(B,RIS_UNIT,dtype=torch.float32,device=DEVICE,) # shape = (B, RIS_UNIT)
    # 變成RIS相位旋轉向量
    theta = torch.exp(1j * phase).to(torch.complex64)

    return theta # shape = (B, RIS_UNIT)


def make_rzf_beamformer(H_eff_H, lambda_reg):
    """
    RZF communication beamformer.

    Math:
        H_eff = (H_eff_H)^H
        W_C_raw = (H_eff H_eff^H + lambda I_M)^(-1) H_eff
        W_C_dir = W_C_raw / ||W_C_raw||_F

    Input:
        H_eff_H : (B,K,M)
            Effective communication channel matrix H_eff^H.
            Each row is h_eff,k^H.

        lambda_reg:
            RZF regularization parameter.

    Return:
        W_C_dir : (B,M,K)
            The complete communication beamforming matrix is
            Frobenius-normalized to unit power for every batch sample:
            ||W_C_dir[b]||_F^2 = 1.
            Communication power is assigned later.
    """

    H_eff_H    = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=DEVICE)          # (B,K,M)
    lambda_reg = torch.as_tensor(lambda_reg,dtype=torch.float32,device=DEVICE)         # scalar

    B,K,M = H_eff_H.shape

    # H_eff = (H_eff^H)^H
    H_eff = torch.conj(H_eff_H).transpose(1,2)                                        # (B,M,K)

    # I_M
    I_M = torch.eye(M,dtype=torch.complex64,device=DEVICE).unsqueeze(0).expand(B,M,M)  # (B,M,M)

    # H_eff H_eff^H
    H_eff_H_eff_H = torch.matmul(H_eff,H_eff_H)                                       # (B,M,M)

    # H_eff H_eff^H + lambda I_M
    RZF_matrix = H_eff_H_eff_H + lambda_reg * I_M                                     # (B,M,M)

    # (H_eff H_eff^H + lambda I_M)^(-1)
    RZF_matrix_inv = torch.linalg.inv(RZF_matrix)                                     # (B,M,M)

    # W_C_raw = (H_eff H_eff^H + lambda I_M)^(-1) H_eff
    W_C_raw = torch.matmul(RZF_matrix_inv,H_eff)                                      # (B,M,K)

    W_C_dir = W_C_raw / (torch.sqrt(torch.sum(torch.abs(W_C_raw) ** 2,dim=(1,2),keepdim=True).real)+1e-12) # 功率正規化

    return W_C_dir


def mrt_in_H_eff_H_nullspace(H_eff_H, g_dt):
    """
    Cell-Free ISAC Eq. (17): nullspace conjugate sensing beamformer.

    Math:
        H_eff = (H_eff_H)^H
        P_NS = I_M - H_eff (H_eff^H H_eff)^dagger H_eff^H
        W_R_raw = P_NS g_dt
        W_R = W_R_raw / ||W_R_raw||

    Input:
        H_eff_H : (B,K,M)
            Effective communication channel matrix H_eff^H.
            Each row is h_eff,k^H.

        g_dt : (B,M,1)
            Transmit-side target sensing direction.

    Return:
        W_R_dir : (B,M,1)
            Unit-norm sensing beam direction.
            Sensing power is assigned later.
    """

    H_eff_H = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=DEVICE)             # (B,K,M)
    g_dt = torch.as_tensor(g_dt,dtype=torch.complex64,device=DEVICE)                   # (B,M,1)

    B,K,M = H_eff_H.shape

    # H_eff = (H_eff^H)^H
    H_eff = torch.conj(H_eff_H).transpose(1,2)                                        # (B,M,K)

    # I_M
    I_M = torch.eye(M,dtype=torch.complex64,device=DEVICE).unsqueeze(0).expand(B,M,M)  # (B,M,M)

    # H_eff^H H_eff
    H_eff_H_H_eff = torch.matmul(H_eff_H,H_eff)                                       # (B,K,K)

    # (H_eff^H H_eff)^dagger
    H_eff_H_H_eff_pinv = torch.linalg.pinv(H_eff_H_H_eff)                             # (B,K,K)

    # H_eff (H_eff^H H_eff)^dagger
    comm_projection_left = torch.matmul(H_eff,H_eff_H_H_eff_pinv)                     # (B,M,K)

    # H_eff (H_eff^H H_eff)^dagger H_eff^H
    comm_projection = torch.matmul(comm_projection_left,H_eff_H)                      # (B,M,M)

    # P_NS = I_M - H_eff (H_eff^H H_eff)^dagger H_eff^H
    P_NS = I_M - comm_projection                                                      # (B,M,M)

    # W_R_raw = P_NS g_dt
    W_R_raw = torch.matmul(P_NS,g_dt)                                                 # (B,M,1)

    # W_R_dir = W_R_raw / ||W_R_raw||_F
    W_R_dir = W_R_raw / (torch.sqrt(torch.sum(torch.abs(W_R_raw) ** 2,dim=(1,2),keepdim=True).real)+1e-12)

    return W_R_dir


def beamformers_power_split(W_C, W_R):
    """
    這裡輸入要是正規化後的W_C, W_R
    """
    total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)

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


if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    dataset_path = os.path.join(DATA_DIR, "dataset_val.npz")        # 這裡讀資料
    dataset = physics_net.load_channel_dataset(dataset_path, "val")

    # 取出固定 layout 下的所有帶 PL 估測通道
    h_dk = torch.as_tensor(dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)     # (B, M, K)
    h_rk = torch.as_tensor(dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)     # (B, N, K)
    G    = torch.as_tensor(dataset["G_hat"],dtype=torch.complex64,device=DEVICE)        # (B, N, M)
    g_dt = torch.as_tensor(dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)     # (B, M, 1)

    B       =  h_dk.shape[0]
    theta   = make_random_ris(B) # 每個 estimated channel 各產生一組獨立 random RIS，shape=(B,N)

    H_eff_H = physics_net.compute_effective_channel(h_dk,h_rk,G,theta) 

    W_C_dir = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)               # 這裡輸出功率正規化 W_C_raw

    W_R_dir = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt)               # 這裡輸出功率正規化 W_R_raw

    W_C, W_R = beamformers_power_split(W_C_dir, W_R_dir)            # power split 寫死


    W_C_dir_power = torch.mean(torch.sum(torch.abs(W_C_dir) ** 2, dim=(1, 2)))
    W_R_dir_power = torch.mean(torch.sum(torch.abs(W_R_dir) ** 2, dim=(1, 2)))
    W_C_power     = torch.mean(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)))
    W_R_power     = torch.mean(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2)))
    W_total_power = W_C_power + W_R_power

    if Debug:
        print(
            f"[Before power allocation] \n"
            f"W_C_raw_power = {float(W_C_dir_power.detach().cpu()):.6e}, \n"
            f"W_R_raw_power = {float(W_R_dir_power.detach().cpu()):.6e}, \n\n"
            f"[After power allocation] \n"
            f"W_C power = {float(W_C_power.detach().cpu()):.6e}, \n"
            f"W_R power = {float(W_R_power.detach().cpu()):.6e}, \n"
        )

    metrics = physics_net.compute_isac_batch_performance(H_eff_H,g_dt,W_C,W_R)
    """ 輸出結果 :
            SINR power components:
                signal       : (B, K)
                comm_interf  : (B, K)
                radar_interf : (B, K)
                noise        : scalar

            raw per-channel-sample tensors:
                sinr          : (B, K), linear
                sinr_db       : (B, K), dB
                rate          : (B, K)
                sumrate       : (B,)
                target_snr    : (B,), linear
                target_snr_db : (B,), dB

            B-average display values: 對所有channel 平均
                sinr_user_mean      : (K,), linear
                sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                rate_user_mean      : (K,)
                sumrate_mean        : scalar
                target_snr_mean     : scalar, linear
                target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
    """
    
    # 取輸出結果
    nominal_allUE_rate      = metrics["rate"]                               # (B,K)  所有UE在每筆估測通道的rate
    nominal_target_snr      = metrics["target_snr"]                         # (B,)   每一筆估測通道(線性，因為要線性平均後在計算)

    nominal_signal          = metrics["signal"]                             # (B,K)  各UE的signal power
    nominal_comm_interf     = metrics["comm_interf"]                        # (B,K)  
    nominal_radar_interf    = metrics["radar_interf"]                       # (B,K)
    nominal_noise           = metrics["noise"]                              # scalar

    # Nominal 計算
    nominal_with_batch    = torch.min(nominal_allUE_rate, dim=1).values     # (B,)   每一筆估測通道，找出所有UE中 rate中最小的
    nominal = torch.mean(nominal_with_batch)                                # scalar 對B筆估測通道min-rate平均

    # SINR組成 
    signal_power         = torch.mean(nominal_signal,dim=0)                 #(K,)
    comm_interf_power    = torch.mean(nominal_comm_interf, dim=0)           #(K,)
    radar_interf_power   = torch.mean(nominal_radar_interf, dim=0)          #(K,)

    # Nominal target SNR：linear mean 後轉 dB
    nominal_target_snr_mean = torch.mean(nominal_target_snr)                # scalar
    nominal_target_snr_mean_db = 10.0 * torch.log10(nominal_target_snr_mean.clamp_min(1e-12))            

    # ============================================================
    # 接下來看 Random RIS + RZF + NS-MRT Baseline 在 injection channel 下的尾端性能
    # ============================================================
    S, B, M, N, K = INJECTION_SAMPLES, h_dk.shape[0], TX_ANT, RIS_UNIT, UAV_COMM    # 簡寫

    # 使用 .unsqueeze(0) 新增第0維度 供INJ用
    # 使用 .expand       功能上"複製"通道成S份 (實際較複雜但最終是複製)
    h_dk_rep = h_dk.unsqueeze(0).expand(S, B, M, K)
    h_rk_rep = h_rk.unsqueeze(0).expand(S, B, N, K)
    G_rep    = G.unsqueeze(0).expand(S, B, N, M)
    g_dt_rep = g_dt.unsqueeze(0).expand(S, B, M, 1)

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

    # 複製theta
    theta_rep = theta.unsqueeze(0).expand(S, B, N)

    # 複製W_C & W_R
    W_C_rep = W_C.unsqueeze(0).expand(S, B, M, K)
    W_R_rep = W_R.unsqueeze(0).expand(S, B, M, 1)

    # 為了使用(physics_net.)副函式,先將4維壓到3維"注意! 此舉不會破壞5%的抽取"
    h_dk_flat  = h_dk_inj.reshape(S*B, M, K)
    h_rk_flat  = h_rk_inj.reshape(S*B, N, K)
    G_flat     = G_inj.reshape(S*B, N, M)
    g_dt_flat  = g_dt_inj.reshape(S*B, M, 1)

    theta_flat = theta_rep.reshape(S*B, N)
    W_C_flat   = W_C_rep.reshape(S*B, M, K)
    W_R_flat   = W_R_rep.reshape(S*B, M, 1)

    H_eff_H_flat = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)                # (S*B, K, M)

    metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)
    '''Output:
            SINR power components:
                signal       : (S*B, K)
                comm_interf  : (S*B, K)
                radar_interf : (S*B, K)
                noise        : scalar

            raw per-channel-sample tensors:
                sinr          : (S*B, K), linear
                sinr_db       : (S*B, K), dB
                rate          : (S*B, K)
                sumrate       : (S*B,)
                target_snr    : (S*B,), linear
                target_snr_db : (S*B,), dB

            S*B-average  對所有injeton * channel 平均
                注意不可以用這裡! 這混淆了取5%
                sinr_user_mean      : (K,), linear
                sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                rate_user_mean      : (K,)
                sumrate_mean        : scalar
                target_snr_mean     : scalar, linear
                target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
    '''
    
    # 還原shape並取輸出結果
    robust_allUE_rate      = metrics_flat["rate"].reshape(S,B,K)                # (S,B,K) 各UE在每筆injection channel的rate
    robust_target_snr      = metrics_flat["target_snr"].reshape(S,B)            # (S,B)   每一筆injection channel的target SNR，linear

    robust_signal          = metrics_flat["signal"].reshape(S,B,K)              # (S,B,K) 各UE的signal power
    robust_comm_interf     = metrics_flat["comm_interf"].reshape(S,B,K)         # (S,B,K) 各UE的communication interference
    robust_radar_interf    = metrics_flat["radar_interf"].reshape(S,B,K)        # (S,B,K) 各UE的radar interference
    robust_noise           = metrics_flat["noise"]                              # scalar
    
    # Robust計算
    worstUE_rate          = torch.min(robust_allUE_rate,dim=2).values           # (S,B) 每一筆injection realization先找worst UE
    robust_per_channel    = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)  # (B,) 每個estimated channel對S取Q0.05
    robust                = torch.mean(robust_per_channel)                      # scalar 最後對B筆estimated channel平均

    # SINR組成 
    robust_signal_power          = torch.mean(robust_signal,dim=(0,1))          # (K,) 對S,B平均
    robust_comm_interf_power     = torch.mean(robust_comm_interf,dim=(0,1))     # (K,)
    robust_radar_interf_power    = torch.mean(robust_radar_interf,dim=(0,1))    # (K,)

    # Robust target SNR：linear mean 後轉 dB
    robust_target_snr_mean = torch.mean(robust_target_snr,dim=(0,1))            # scalar, linear
    robust_target_snr_mean_db = 10.0 * torch.log10(robust_target_snr_mean.clamp_min(1e-12))

    # ============================================================
    # 輸出結果
    # ============================================================
    print("=" * 90)
    print("[Random RIS + RZF + NS-MRT Baseline]")
    print("=" * 90)
    print(f"RIS phase             : Random RIS")
    print(f"W_C                   : RZF, RZF_LAMBDA = {RZF_LAMBDA:g}")
    print(f"W_R                   : H_eff_H nullspace MRT")
    print(f"Estimated channels B  : {B}")
    print(f"Injection samples S   : {S}")
    print(f"Tail quantile         : {OUTAGE_QUANTILE:.2f}")
    print(f"Total TX power        : {float(W_total_power.detach().cpu()):.6e}")
    print(f"W_C / W_R power ratio : {float((W_C_power / W_total_power).detach().cpu()):.2f} / {float((W_R_power / W_total_power).detach().cpu()):.2f}")

    print("-" * 90)
    print("[Nominal performance: no error injection]")
    print("-" * 90)
    print(f"Nominal minimum-user rate : {float(nominal.detach().cpu()):.6f} bps/Hz")
    print(f"Nominal target SNR mean   : {float(nominal_target_snr_mean_db.detach().cpu()):.3f} dB")

    print("\n[Nominal SINR power components]")
    print(f"Signal power               : {fmt_vec_sci(signal_power.detach().cpu().numpy(), precision=3)}")
    print(f"Communication interference : {fmt_vec_sci(comm_interf_power.detach().cpu().numpy(), precision=3)}")
    print(f"Radar interference         : {fmt_vec_sci(radar_interf_power.detach().cpu().numpy(), precision=3)}")
    print(f"Noise power                : {float(nominal_noise.detach().cpu()):.3e}")

    print("-" * 90)
    print("[Robust performance: error injection]")
    print("-" * 90)
    print(f"Robust minimum-user rate Q{OUTAGE_QUANTILE:.2f} : {float(robust.detach().cpu()):.6f} bps/Hz")
    print(f"Robust target SNR mean     : {float(robust_target_snr_mean_db.detach().cpu()):.3f} dB")

    print("\n[Robust SINR power components]")
    print(f"Signal power               : {fmt_vec_sci(robust_signal_power.detach().cpu().numpy(), precision=3)}")
    print(f"Communication interference : {fmt_vec_sci(robust_comm_interf_power.detach().cpu().numpy(), precision=3)}")
    print(f"Radar interference         : {fmt_vec_sci(robust_radar_interf_power.detach().cpu().numpy(), precision=3)}")
    print(f"Noise power                : {float(robust_noise.detach().cpu()):.3e}")
    print("=" * 90)
    
    
    
    