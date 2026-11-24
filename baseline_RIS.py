# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, beamformers_power_split, fmt_vec_sci, RZF_LAMBDA
from two_timescale_NN import CommNet, ThetaNet

Debug = True                         # 終端印出檢查


if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # ============================================================
    # Paths
    # ============================================================
    dataset_path  = os.path.join(DATA_DIR, "dataset_test.npz")
    ris_only_ckpt = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")

    for required_path in [dataset_path,ris_only_ckpt]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"找不到檔案: {required_path}")

    # ============================================================
    # Load trained ThetaNet
    # ============================================================
    theta_net = ThetaNet().to(DEVICE)
    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    # 讀取資料
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
    # NN RIS + RZF + NS-MRT Nominal Test
    # ============================================================
    with torch.no_grad():

        # RIS每LT_BLOCK_SIZE個slot更新一次
        test_channel_ids    = np.arange(0,test_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
        test_trajectory_ids = np.repeat(np.arange(test_trajectories),len(test_channel_ids))
        test_channel_ids    = np.tile(test_channel_ids,test_trajectories)

        B_test = len(test_channel_ids)                                                # long-term RIS blocks

        # 每個long-term block的start slot CSI
        h_dk_test = torch.as_tensor(test_dataset["h_dk"][test_trajectory_ids,test_channel_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_test = torch.as_tensor(test_dataset["h_rk"][test_trajectory_ids,test_channel_ids],dtype=torch.complex64,device=DEVICE)
        G_test    = torch.as_tensor(test_dataset["G"][test_trajectory_ids,test_channel_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_test = torch.as_tensor(test_dataset["g_dt"][test_trajectory_ids,test_channel_ids],dtype=torch.complex64,device=DEVICE)

        # 每個long-term block的start slot由trained ThetaNet產生RIS phase
        theta_test = theta_net(h_dk_test,h_rk_test,G_test,g_dt_test)                  # (B_test,N)

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

        # 每個NN theta固定使用LT_BLOCK_SIZE個slots
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
    print("[NN RIS]")
    print("=" * 90)
    print(f"Test dataset              : {dataset_path}")
    print(f"ThetaNet checkpoint       : {ris_only_ckpt}")
    print(f"Test trajectories         : {test_trajectories}")
    print(f"Slots per trajectory      : {test_channels}")
    print(f"LT block size             : {LT_BLOCK_SIZE}")
    print(f"RIS decisions             : {B_test}")
    print(f"Evaluated slots           : {B_test * LT_BLOCK_SIZE}")
    print(f"RIS phase                 : Trained ThetaNet")
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