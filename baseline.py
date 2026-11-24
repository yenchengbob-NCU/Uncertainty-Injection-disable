# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
from settings import *
from two_timescale_NN import CommNet

Debug = True                         # 終端印出檢查

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


if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到檔案: {dataset_path}")

    # 載入完整 test channel dataset
    with np.load(dataset_path) as data:
        test_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

    # Dataset size
    test_trajectories = test_dataset["h_dk"].shape[0]
    test_channels     = test_dataset["h_dk"].shape[1]

    print("\n[INFO] test dataset loaded")
    print(f"[INFO] test trajectories = {test_trajectories}")
    print(f"[INFO] test channels     = {test_channels}")
    print(f"[INFO] h_dk shape        = {test_dataset['h_dk'].shape}")
    print(f"[INFO] h_rk shape        = {test_dataset['h_rk'].shape}")
    print(f"[INFO] G shape           = {test_dataset['G'].shape}")
    print(f"[INFO] g_dt shape        = {test_dataset['g_dt'].shape}")

    # ============================================================
    # Random RIS + RZF + NS-MRT Nominal Test
    # ============================================================
    with torch.no_grad():

        # RIS每LT_BLOCK_SIZE個slot更新一次
        test_channel_ids    = np.arange(0,test_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
        test_trajectory_ids = np.repeat(np.arange(test_trajectories),len(test_channel_ids))
        test_channel_ids    = np.tile(test_channel_ids,test_trajectories)

        B_test = len(test_channel_ids)                                                # long-term RIS blocks

        # 每個long-term block產生一組random RIS phase
        theta_test = make_random_ris(B_test)                                          # (B_test,N)

        # 每個start對應後面LT_BLOCK_SIZE個slots
        test_block_ids = test_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]  # (B_test,L)

        h_dk_test_block = torch.as_tensor(test_dataset["h_dk"][test_trajectory_ids[:,None],test_block_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_test_block = torch.as_tensor(test_dataset["h_rk"][test_trajectory_ids[:,None],test_block_ids],dtype=torch.complex64,device=DEVICE)
        G_test_block    = torch.as_tensor(test_dataset["G"][test_trajectory_ids[:,None],test_block_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_test_block = torch.as_tensor(test_dataset["g_dt"][test_trajectory_ids[:,None],test_block_ids],dtype=torch.complex64,device=DEVICE)

        # (B_test,L,...) -> (B_test*L,...)
        h_dk_test_block = h_dk_test_block.reshape(B_test*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
        h_rk_test_block = h_rk_test_block.reshape(B_test*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
        G_test_block    = G_test_block.reshape(B_test*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
        g_dt_test_block = g_dt_test_block.reshape(B_test*LT_BLOCK_SIZE,TX_ANT,1)

        # 每個random theta固定使用LT_BLOCK_SIZE個slots
        theta_test_block = theta_test[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_test*LT_BLOCK_SIZE,RIS_UNIT)

        # 計算每個slot的effective channel
        H_eff_H = physics_net.compute_effective_channel(h_dk_test_block,h_rk_test_block,G_test_block,theta_test_block)

        W_C_dir = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)                            # 這裡輸出功率正規化 W_C_dir

        W_R_dir = mrt_in_H_eff_H_nullspace(H_eff_H,g_dt_test_block)                   # 這裡輸出功率正規化 W_R_dir

        W_C,W_R = beamformers_power_split(W_C_dir,W_R_dir)                           # power 分配

        metrics = physics_net.compute_isac_batch_performance(H_eff_H,g_dt_test_block,W_C,W_R)
        """ 輸出結果 :
                SINR power components:
                    signal       : (B_test*L, K)
                    comm_interf  : (B_test*L, K)
                    radar_interf : (B_test*L, K)
                    noise        : scalar

                raw per-channel-sample tensors:
                    sinr          : (B_test*L, K), linear
                    sinr_db       : (B_test*L, K), dB
                    rate          : (B_test*L, K)
                    sumrate       : (B_test*L,)
                    target_snr    : (B_test*L,), linear
                    target_snr_db : (B_test*L,), dB
        """

        # 取輸出結果
        nominal_allUE_rate = metrics["rate"].reshape(B_test,LT_BLOCK_SIZE,UAV_COMM)   # (B_test,L,K) 所有UE在每個slot的rate
        nominal_target_snr = metrics["target_snr"].reshape(B_test,LT_BLOCK_SIZE)      # (B_test,L) 每個slot的target SNR，linear

        nominal_signal       = metrics["signal"]                                      # (B_test*L,K)
        nominal_comm_interf  = metrics["comm_interf"]                                 # (B_test*L,K)
        nominal_radar_interf = metrics["radar_interf"]                                # (B_test*L,K)
        nominal_noise        = metrics["noise"]                                       # scalar

        # Nominal 計算
        nominal_with_batch = torch.min(nominal_allUE_rate,dim=2).values               # (B_test,L) 每個slot找出K個UE中rate最小的
        nominal_with_block = torch.mean(nominal_with_batch,dim=1)                     # (B_test,) 對每個block的L個slot平均
        nominal = torch.mean(nominal_with_block)                                      # scalar 對全部test block平均

        # SINR組成
        signal_power       = torch.mean(nominal_signal,dim=0)                         # (K,)
        comm_interf_power  = torch.mean(nominal_comm_interf,dim=0)                    # (K,)
        radar_interf_power = torch.mean(nominal_radar_interf,dim=0)                   # (K,)

        # Nominal target SNR：linear mean後轉dB
        nominal_target_snr_mean = torch.mean(nominal_target_snr)
        nominal_target_snr_mean_db = 10.0 * torch.log10(nominal_target_snr_mean.clamp_min(1e-12))

        # Beamformer power檢查
        W_C_dir_power = torch.mean(torch.sum(torch.abs(W_C_dir) ** 2,dim=(1,2)))
        W_R_dir_power = torch.mean(torch.sum(torch.abs(W_R_dir) ** 2,dim=(1,2)))
        W_C_power     = torch.mean(torch.sum(torch.abs(W_C) ** 2,dim=(1,2)))
        W_R_power     = torch.mean(torch.sum(torch.abs(W_R) ** 2,dim=(1,2)))
        W_total_power = W_C_power + W_R_power

    if Debug:
        print(
            f"\n[Before power allocation] \n"
            f"W_C_raw_power = {float(W_C_dir_power.detach().cpu()):.6e}, \n"
            f"W_R_raw_power = {float(W_R_dir_power.detach().cpu()):.6e}, \n\n"
            f"[After power allocation] \n"
            f"W_C power = {float(W_C_power.detach().cpu()):.6e}, \n"
            f"W_R power = {float(W_R_power.detach().cpu()):.6e}, \n"
        )

    # ============================================================
    # 輸出結果
    # ============================================================
    print("=" * 90)
    print("[Random RIS]")
    print("=" * 90)
    print(f"Test dataset              : {dataset_path}")
    print(f"Test trajectories         : {test_trajectories}")
    print(f"Slots per trajectory      : {test_channels}")
    print(f"LT block size             : {LT_BLOCK_SIZE}")
    print(f"RIS decisions             : {B_test}")
    print(f"Evaluated slots           : {B_test * LT_BLOCK_SIZE}")
    print(f"RIS phase                 : Random RIS")
    print(f"W_C                       : RZF, RZF_LAMBDA = {RZF_LAMBDA:g}")
    print(f"W_R                       : H_eff_H nullspace MRT")
    print(f"Total TX power            : {float(W_total_power.detach().cpu()):.6e}")
    print(f"W_C / W_R power ratio     : {float((W_C_power / W_total_power).detach().cpu()):.2f} / {float((W_R_power / W_total_power).detach().cpu()):.2f}")

    print("-" * 90)
    print("[Nominal performance]")
    print("-" * 90)
    print(f"Nominal worst-UE rate     = {float(nominal.detach().cpu()):.6f} bps/Hz")
    print(f"Target SNR                = {float(nominal_target_snr_mean_db.detach().cpu()):.3f} dB")

    if Debug:
        print("\n[Nominal SINR power components]")
        print(f"Signal power               : {fmt_vec_sci(signal_power.detach().cpu().numpy(),precision=3)}")
        print(f"Communication interference : {fmt_vec_sci(comm_interf_power.detach().cpu().numpy(),precision=3)}")
        print(f"Radar interference         : {fmt_vec_sci(radar_interf_power.detach().cpu().numpy(),precision=3)}")
        print(f"Noise power                : {float(nominal_noise.detach().cpu()):.3e}")

    print("=" * 90)