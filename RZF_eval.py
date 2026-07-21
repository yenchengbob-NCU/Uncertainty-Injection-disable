# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, RZF_LAMBDA
from two_timescale_NN import ThetaNet

Debug = False                        # 終端印出檢查
# ============================================================
# Helpers
# ============================================================
def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


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
    # 簡寫
    S, B, M, N, K = INJECTION_SAMPLES, N_TEST_CHANNELS, TX_ANT, RIS_UNIT, UAV_COMM
    
    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = ThetaNet().to(DEVICE)
    physics_net.eval()
    
    # 設定路徑
    test_dataset_path   = os.path.join(DATA_DIR, "dataset_test.npz")
    ris_only_ckpt       = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    result_dir          = RESULT_DIR

    # 讀取資料
    test_dataset        = physics_net.load_channel_dataset(test_dataset_path, "test")

    # 載入 theta net 權重
    theta_net = ThetaNet().to(DEVICE)
    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    with torch.no_grad():
        # 載入估測通道
        h_dk_teat = torch.as_tensor(test_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, K)
        h_rk_teat = torch.as_tensor(test_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, N, K)
        G_test    = torch.as_tensor(test_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)      # (B, N, M)
        g_dt_test = torch.as_tensor(test_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, 1)

        # 取得theta
        theta = theta_net(h_dk_teat,h_rk_teat,G_test,g_dt_test)                                     # (B, N)

        # 算出等效通道
        H_eff_H = physics_net.compute_effective_channel(h_dk_teat,h_rk_teat,G_test,theta)

        # 取得beamformer
        W_C_raw = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)               # 這裡輸出功率正規化 W_C_raw
        W_R_raw = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt_test)          # 這裡輸出功率正規化 W_R_raw

        W_C, W_R = beamformers_power_split(W_C_raw, W_R_raw)            # power split 寫死

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

        # 先複製beamformer 與 theta S 次
        theta_rep   = theta.unsqueeze(0).expand(S, B, N)
        W_C_rep     = W_C.unsqueeze(0).expand(S, B, M, K)
        W_R_rep     = W_R.unsqueeze(0).expand(S, B, M, 1)

        # 將SBreshape
        theta_flat = theta_rep.reshape(S*B, N)
        W_C_flat   = W_C_rep.reshape(S*B, M, K)
        W_R_flat   = W_R_rep.reshape(S*B, M, 1)

        # 組合出等效通道
        H_eff_H_flat = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)

        # 計算結果
        metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)

        # 取出計算結果
        '''Output:
                raw per-channel-sample tensors:
                    sinr          : (S*B, K), linear
                    sinr_db       : (S*B, K), dB
                    rate          : (S*B, K)
                    sumrate       : (S*B,)
                    target_snr    : (S*B,), linear
                    target_snr_db : (S*B,), dB

                S*B-average display values: 對所有injeton * channel 平均
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
        robust_noise           = metrics_flat["noise"]  

        # Robust計算
        worstUE_rate          = torch.min(robust_allUE_rate,dim=2).values           # (S,B) 每一筆injection realization先找worst UE
        robust_per_channel    = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)  # (B,) 每個estimated channel對S取Q0.05
        robust                = torch.mean(robust_per_channel)  

        # SINR組成 
        robust_signal_power          = torch.mean(robust_signal,dim=(0,1))          # (K,) 對S,B平均
        robust_comm_interf_power     = torch.mean(robust_comm_interf,dim=(0,1))     # (K,)
        robust_radar_interf_power    = torch.mean(robust_radar_interf,dim=(0,1))    # (K,)

        # Robust target SNR：對所有S筆injection與B筆estimated channel做linear mean，再轉dB
        robust_target_snr_mean = torch.mean(robust_target_snr)                    # scalar, linear
        robust_target_snr_mean_db = 10.0 * torch.log10(robust_target_snr_mean.clamp_min(1e-12))
        
        # Sensing outage：逐一檢查所有 S*B 筆 injected target SNR 是否低於門檻
        sensing_violation_mask = robust_target_snr < SENSING_SNR_THRESHOLD          # (S,B), bool
        sensing_violation_count = torch.sum(sensing_violation_mask)                 # scalar, 違反次數
        sensing_total_count = robust_target_snr.numel()                             # S*B
        sensing_violation_rate = sensing_violation_count.float() / sensing_total_count
        sensing_violation_percent = 100.0 * sensing_violation_rate

        # 印出 RZF NSMRT 結果
        print("=" * 90)
        print("[Injection-tail performance: Baseline]")
        print("=" * 90)
        print(f"Injection samples S : {S}")
        print(f"Estimated channels B: {B}")
        print(f"Tail quantile       : {OUTAGE_QUANTILE:.2f}")

        print("-" * 90)
        print("[Communication tail metrics]")
        print(f"Robust      : {float(robust.detach().cpu()):.6f} bps/Hz")
        print("-" * 90)
        
        print("[Target sensing metrics over injected channels]")
        print(f"Target SNR threshold        : {SENSING_SNR_THRESHOLD_DB:.3f} dB")
        print(f"Target SNR mean             : {float(robust_target_snr_mean_db.detach().cpu()):.3f} dB")
        print(f"Sensing violation count     : {int(sensing_violation_count.detach().cpu())} / {sensing_total_count}")
        print(f"Sensing violation percentage: {float(sensing_violation_percent.detach().cpu()):.4f}%")
        print("=" * 90)



