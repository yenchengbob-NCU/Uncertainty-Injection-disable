# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from one_timescale_NN import CommNet

Debug = True                        # 終端印出檢查

RZF_LAMBDA = 0.0                    # 設定為 0 就是ZF
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

    return theta # shape = (B, RIS_UNIT)


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

    W_C_dir = W_C / (torch.sqrt(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12) # 功率正規化

    return W_C_dir  # (B,M,K)


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

    W_R_dir = W_R / (torch.sqrt(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12) # 功率正規化
    
    return W_R_dir  # (B,M,1)


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

    theta = make_random_ris(1)                      # 針對這 layout 建立 1 組 random RIS (B, N)

    H_eff_H = physics_net.compute_effective_channel(h_dk,h_rk,G,theta)

    W_C_raw = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)               # 這裡輸出功率正規化 W_C_raw

    W_R_raw = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt)               # 這裡輸出功率正規化 W_R_raw

    W_C, W_R = beamformers_power_split(W_C_raw, W_R_raw)            # power split 寫死


    W_C_raw_power = torch.mean(torch.sum(torch.abs(W_C_raw) ** 2, dim=(1, 2)))
    W_R_raw_power = torch.mean(torch.sum(torch.abs(W_R_raw) ** 2, dim=(1, 2)))
    W_C_power     = torch.mean(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)))
    W_R_power     = torch.mean(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2)))
    W_total_power = W_C_power + W_R_power

    if Debug:
        print(
            f"[Before power allocation] \n"
            f"W_C_raw_power = {float(W_C_raw_power.detach().cpu()):.6e}, \n"
            f"W_R_raw_power = {float(W_R_raw_power.detach().cpu()):.6e}, \n\n"
            f"[After power allocation] \n"
            f"W_C power = {float(W_C_power.detach().cpu()):.6e}, \n"
            f"W_R power = {float(W_R_power.detach().cpu()):.6e}, \n"
        )

    metrics = physics_net.compute_isac_batch_performance(H_eff_H,g_dt,W_C,W_R)

    if Debug:
        signal_mean         = torch.mean(metrics["signal"], dim=0)
        comm_interf_mean    = torch.mean(metrics["comm_interf"], dim=0)
        radar_interf_mean   = torch.mean(metrics["radar_interf"], dim=0)
        noise_value = metrics["noise"]
        print("SINR公式檢查")
        print(f"signal       : {signal_mean.detach().cpu().numpy()}")
        print(f"comm_interf  : {comm_interf_mean.detach().cpu().numpy()}")
        print(f"radar_interf : {radar_interf_mean.detach().cpu().numpy()}")
        print(f"noise        : {float(noise_value.detach().cpu())}")

    sinr_db_all    = metrics["sinr_db"].detach().cpu().numpy().reshape(-1)
    sumrate_all    = metrics["sumrate"].detach().cpu().numpy().reshape(-1)
    target_snr_all = metrics["target_snr"].detach().cpu().numpy().reshape(-1)


    logs = {
        "sumrate_mean": float(metrics["sumrate_mean"].detach().cpu()),
        "target_snr_mean": float(metrics["target_snr_mean"].detach().cpu()),
        "target_snr_mean_db": float(metrics["target_snr_mean_db"].detach().cpu()),

        "sinr_user_mean": metrics["sinr_user_mean"].detach().cpu().numpy(),
        "sinr_user_mean_db": metrics["sinr_user_mean_db"].detach().cpu().numpy(),
        "rate_user_mean": metrics["rate_user_mean"].detach().cpu().numpy(),

        "sinr_db_all": sinr_db_all,
        "sumrate_all": sumrate_all,
        "target_snr_all": target_snr_all,
    }

    # 對所有channel取平均
    B = h_dk.shape[0]

    mean_sumrate       = logs["sumrate_mean"]
    mean_target_snr    = logs["target_snr_mean"]
    mean_target_snr_db = logs["target_snr_mean_db"]

    mean_user_sinr    = logs["sinr_user_mean"]
    mean_user_sinr_db = logs["sinr_user_mean_db"]
    mean_user_rate    = logs["rate_user_mean"]

    print("=" * 90)
    print("[Random RIS + RZF + NS-MRT Baseline]")
    print("=" * 90)
    print("Scenario        : 固定 UE layout")
    print(f"RIS phase       : Random RIS, one theta shared by all {B} val estimated channels")
    print(f"W_C             : RZF, RZF_LAMBDA = {RZF_LAMBDA:g}")
    print(f"W_R             : H_eff_H nullspace MRT")
    print(
        f"Total TX POWER = {float(W_total_power.detach().cpu()):.6e}, \n"
        f"W_C ratio = {float((W_C_power / W_total_power).detach().cpu()):.2f},"
        f"W_R ratio = {float((W_R_power / W_total_power).detach().cpu()):.2f},"
        )        
    print("-" * 90)

    print("[Communication metrics over val estimated channels]")
    print(f"Mean sum-rate   : {mean_sumrate:.6f} bps/Hz")
    print(f"UE SINR linear  : {fmt_vec(mean_user_sinr, precision=4)}")
    print(f"UE SINR dB      : {fmt_vec(mean_user_sinr_db, precision=3)}")
    print(f"UE rate         : {fmt_vec(mean_user_rate, precision=4)} bps/Hz")

    print("-" * 90)
    print("[Target sensing metrics over val estimated channels]")
    print(f"Target SNR linear : {mean_target_snr:.6e} ")
    print(f"Target SNR dB     : {mean_target_snr_db:.3f} dB")
    print("=" * 90)

    # ============================================================
    # 接下來看 Random RIS + RZF + NS-MRT Baseline 在 injection channel 下的尾端性能
    # ============================================================

    # 簡寫
    S, B, M, N, K = INJECTION_SAMPLES, h_dk.shape[0], TX_ANT, RIS_UNIT, UAV_COMM

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
    
    # 還原shape
    rate_inj      = metrics_flat["rate"].reshape(S, B, K)           # (S,B,K) 各UE的rate

    # 對每個估測通道的S筆注入取5%,各UE分開
    rate_user_q05 = torch.quantile(rate_inj,OUTAGE_QUANTILE,dim=0)  # (B,K)

    # 再算出Sumrate
    sumrate_q05 = torch.sum(rate_user_q05, dim=1)                   # (B,)

    # 這時再對B通道平均
    rob_sumrate         = torch.mean(sumrate_q05)                   # (scalar)
    rate_user_q05_mean  = torch.mean(rate_user_q05, dim=0)          # (K,)

    # 取出對所有S*B平均的感測NSR,這裡可以這樣是因為不需要取5%
    target_snr_mean_db = metrics_flat["target_snr_mean_db"]         # (scalar)

    # 輸出結果
    print("=" * 90)
    print("[Injection-tail performance: Random RIS + ZF + NS-MRT Baseline]")
    print("=" * 90)
    print(f"Injection samples S : {S}")
    print(f"Estimated channels B: {B}")
    print(f"Tail quantile       : {OUTAGE_QUANTILE:.2f}")
    print("-" * 90)

    print("[Communication tail metrics]")
    print(f"Robust sum-rate     : {float(rob_sumrate.detach().cpu()):.6f} bps/Hz")
    print(
        f"UE tail rate        : "
        f"{fmt_vec(rate_user_q05_mean.detach().cpu().numpy(), precision=4)} bps/Hz"
    )

    print("-" * 90)
    print("[Target sensing metrics over injected channels]")
    print(f"Target SNR mean dB  : {float(target_snr_mean_db.detach().cpu()):.3f} dB")
    print("=" * 90)
    
    
    
    