# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, RZF_LAMBDA
from one_timescale_NN import TestNet, ThetaNet

# ================================
# Helpers
# ================================
PRETRAIN_COMM_INTERF_WEIGHT = 0.25

def comm_pretrain_loss(H_eff_H,W_C_net_raw,W_C_rzf_raw):
    """
    MSE 比較單位正規化 beamformer。
    Communication interference 使用實際 communication power：
        P_C = 0.4 * TRANSMIT_POWER_TOTAL
    """

    noise = torch.as_tensor(NOISE_POWER,dtype=torch.float32,device=DEVICE)
    p_C   = torch.as_tensor(0.4 * TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)

    # 單位正規化 beamformer 的 supervised MSE
    mse_loss = torch.mean(torch.abs(W_C_net_raw - W_C_rzf_raw.detach()) ** 2)

    # 實際通信功率
    W_C_net = torch.sqrt(p_C) * W_C_net_raw
    W_C_rzf = torch.sqrt(p_C) * W_C_rzf_raw

    Y_net = torch.matmul(H_eff_H,W_C_net)                       # (B,K,K)
    P_net = torch.abs(Y_net) ** 2                              # (B,K,K)

    signal_net = torch.diagonal(P_net,dim1=1,dim2=2)           # (B,K)
    comm_interf_net = torch.sum(P_net,dim=2) - signal_net      # (B,K)

    with torch.no_grad():
        Y_rzf = torch.matmul(H_eff_H,W_C_rzf)                  # (B,K,K)
        P_rzf = torch.abs(Y_rzf) ** 2                          # (B,K,K)
        signal_rzf = torch.diagonal(P_rzf,dim1=1,dim2=2)       # (B,K)

    comm_interf_loss = torch.mean(comm_interf_net / (signal_rzf + noise))

    loss = mse_loss + PRETRAIN_COMM_INTERF_WEIGHT * comm_interf_loss

    return loss,mse_loss,comm_interf_loss


def plot_pretrain_curves(pre_curve_path,pre_curve_dir):
    """
    畫五類圖：

    1. Total loss
    2. MSE loss
    3. Communication interference loss
    4. Train TestNet / Validation TestNet / Validation RZF Nominal
    5. 每個 UE 分開畫 SINR power components，共 8 條線
    """
    os.makedirs(pre_curve_dir,exist_ok=True)

    with np.load(pre_curve_path) as data:
        train_loss = data["train_loss"]
        val_loss = data["val_loss"]

        train_mse_loss = data["train_mse_loss"]
        val_mse_loss = data["val_mse_loss"]

        train_comm_interf_loss = data["train_comm_interf_loss"]
        val_comm_interf_loss = data["val_comm_interf_loss"]

        train_nominal = data["train_nominal"]
        val_nominal = data["val_nominal"]
        val_rzf_nominal = data["val_rzf_nominal"]

        train_signal_power = data["train_signal_power"]                  # (epochs,K)
        val_signal_power = data["val_signal_power"]                      # (epochs,K)

        train_comm_interf_power = data["train_comm_interf_power"]        # (epochs,K)
        val_comm_interf_power = data["val_comm_interf_power"]            # (epochs,K)

        train_radar_interf_power = data["train_radar_interf_power"]      # (epochs,K)
        val_radar_interf_power = data["val_radar_interf_power"]          # (epochs,K)

        train_noise_power = data["train_noise_power"]                    # (epochs,)
        val_noise_power = data["val_noise_power"]                        # (epochs,)

    epochs = np.arange(1,len(train_loss) + 1)

    # ================================
    # Fig 1: Total loss
    # ================================
    plt.figure(figsize=(10,6))

    plt.plot(epochs,train_loss,label="Train Total Loss",linewidth=1.5)
    plt.plot(epochs,val_loss,label="Validation Total Loss",linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TestNet RZF Pretraining Total Loss")
    plt.grid(True,alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(pre_curve_dir,"pretrain_total_loss_curve.png")
    plt.savefig(fig_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {fig_path}")

    # ================================
    # Fig 2: MSE loss
    # ================================
    plt.figure(figsize=(10,6))

    plt.plot(epochs,train_mse_loss,label="Train MSE Loss",linewidth=1.5)
    plt.plot(epochs,val_mse_loss,label="Validation MSE Loss",linewidth=1.5)

    if np.all(train_mse_loss > 0) and np.all(val_mse_loss > 0):
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Complex MSE")
    plt.title("TestNet RZF Pretraining MSE Loss")
    plt.grid(True,which="both",alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(pre_curve_dir,"pretrain_mse_loss_curve.png")
    plt.savefig(fig_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {fig_path}")

    # ================================
    # Fig 3: Communication interference loss
    # ================================
    plt.figure(figsize=(10,6))

    plt.plot(
        epochs,
        train_comm_interf_loss,
        label="Train Communication Interference Loss",
        linewidth=1.5,
    )

    plt.plot(
        epochs,
        val_comm_interf_loss,
        label="Validation Communication Interference Loss",
        linewidth=1.5,
    )

    if np.all(train_comm_interf_loss > 0) and np.all(val_comm_interf_loss > 0):
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Normalized Communication Interference Loss")
    plt.title("TestNet RZF Communication Interference Loss")
    plt.grid(True,which="both",alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        pre_curve_dir,
        "pretrain_comm_interf_loss_curve.png",
    )

    plt.savefig(fig_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {fig_path}")

    # ================================
    # Fig 4: Nominal worst-UE rate
    # ================================
    plt.figure(figsize=(10,6))

    plt.plot(
        epochs,
        train_nominal,
        label="Train TestNet Nominal",
        linewidth=1.5,
    )

    plt.plot(
        epochs,
        val_nominal,
        label="Validation TestNet Nominal",
        linewidth=1.5,
    )

    plt.plot(
        epochs,
        val_rzf_nominal,
        label="Validation RZF Nominal",
        linewidth=1.5,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Nominal Worst-UE Rate (bps/Hz)")
    plt.title("TestNet and RZF Nominal Worst-UE Rate")
    plt.grid(True,alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        pre_curve_dir,
        "pretrain_nominal_curve.png",
    )

    plt.savefig(fig_path,dpi=300,bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {fig_path}")

    # ================================
    # Fig 5: SINR power components
    # 每個 UE 分開畫
    # ================================
    num_users = train_signal_power.shape[1]
    plot_eps = 1e-30

    for ue_idx in range(num_users):
        plt.figure(figsize=(12,7))

        plt.plot(
            epochs,
            np.maximum(train_signal_power[:,ue_idx],plot_eps),
            label="Train Signal",
            linewidth=1.8,
        )

        plt.plot(
            epochs,
            np.maximum(val_signal_power[:,ue_idx],plot_eps),
            label="Validation Signal",
            linewidth=1.8,
            linestyle="--",
        )

        plt.plot(
            epochs,
            np.maximum(train_comm_interf_power[:,ue_idx],plot_eps),
            label="Train Communication Interference",
            linewidth=1.8,
        )

        plt.plot(
            epochs,
            np.maximum(val_comm_interf_power[:,ue_idx],plot_eps),
            label="Validation Communication Interference",
            linewidth=1.8,
            linestyle="--",
        )

        plt.plot(
            epochs,
            np.maximum(train_radar_interf_power[:,ue_idx],plot_eps),
            label="Train Radar Interference",
            linewidth=1.8,
        )

        plt.plot(
            epochs,
            np.maximum(val_radar_interf_power[:,ue_idx],plot_eps),
            label="Validation Radar Interference",
            linewidth=1.8,
            linestyle="--",
        )

        plt.plot(
            epochs,
            np.maximum(train_noise_power,plot_eps),
            label="Train Noise",
            linewidth=1.8,
        )

        plt.plot(
            epochs,
            np.maximum(val_noise_power,plot_eps),
            label="Validation Noise",
            linewidth=1.8,
            linestyle="--",
        )

        plt.yscale("log")

        plt.xlabel("Epoch")
        plt.ylabel("Power (linear scale)")
        plt.title(
            f"TestNet RZF SINR Power Components — UE {ue_idx}"
        )

        plt.grid(True,which="both",alpha=0.35)
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(
            pre_curve_dir,
            f"pretrain_UE{ue_idx}_sinr_power_components.png",
        )

        plt.savefig(fig_path,dpi=300,bbox_inches="tight")
        plt.close()

        print(f"[PLOT] Saved: {fig_path}")


def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def fmt_vec_sci(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


def beamformers_power_split(W_C,W_R):
    """
    Input:
        W_C : Frobenius-normalized communication beamformer
        W_R : Frobenius-normalized sensing beamformer
    """
    total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)

    p_C = total_power * 0.4
    p_R = total_power * 0.6

    W_C = torch.sqrt(p_C) * W_C
    W_R = torch.sqrt(p_R) * W_R

    return W_C,W_R


# ================================
# Main
# ================================
if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = ThetaNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")

    train_dataset = physics_net.load_channel_dataset(train_dataset_path, "train")
    val_dataset   = physics_net.load_channel_dataset(val_dataset_path, "val")

    # ================================
    # Pre train comm net
    # ================================
    test_net  = TestNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)
    optimizer = optim.Adam(list(test_net.parameters()),lr=0.001)

    test_net_ckpt = os.path.join(PRETRAIN_DIR, "test_net_heff_rzf.ckpt")
    ris_only_ckpt  = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    pre_curve_path = os.path.join(PRETRAIN_DIR, "test_net_heff_rzf_curves.npz")
    pre_curve_dir  = os.path.join(PRETRAIN_DIR, "test_net_heff_rzf_curves")

    os.makedirs(pre_curve_dir, exist_ok=True)

    curves = {
        # Total pretraining loss
        # loss = mse_loss + PRETRAIN_COMM_INTERF_WEIGHT * comm_interf_loss
        "train_loss": [],
        "val_loss": [],

        # Beamformer imitation loss
        "train_mse_loss": [],
        "val_mse_loss": [],

        # Communication interference diagnostic / future loss term
        "train_comm_interf_loss": [],
        "val_comm_interf_loss": [],

        # Nominal = mean_B(min_K(rate))
        "train_nominal": [],
        "val_nominal": [],
        "val_rzf_nominal": [],

        # 加入 NS-MRT 後的 sensing performance
        "train_target_snr_db": [],
        "val_target_snr_db": [],

        # SINR power components，每個 epoch 儲存 (K,)
        "train_signal_power": [],
        "val_signal_power": [],

        "train_comm_interf_power": [],
        "val_comm_interf_power": [],

        "train_radar_interf_power": [],
        "val_radar_interf_power": [],

        # Noise，每個 epoch 儲存 scalar
        "train_noise_power": [],
        "val_noise_power": [],
    }

    # 訓練參數
    pre_epoch = 5000

    best_val_loss = float("inf")   # L_best <- infinity
    best_val_net_worstUE_rate = 0.0
    best_val_rzf_worstUE_rate = 0.0
    best_val_epoch = 0             # best loss 出現在哪個 epoch

    patience_counter = 0           
    EARLY_STOP_EPS = 1e-7          
    PATIENCE = 30                  

    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0]
    val_channels   = val_dataset["h_dk_hat"].shape[0]
    

    # 載入已訓練好的 ThetaNet
    """
    這裡我們只訓練comm net
    """
    if not os.path.exists(ris_only_ckpt):
        raise FileNotFoundError(f"找不到 ThetaNet checkpoint: {ris_only_ckpt}")

    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)

    print(f"[INFO] Loaded pretrained ThetaNet: {ris_only_ckpt}")
    print("[INFO] ThetaNet is frozen during CommNet pretraining.")

    # 開始訓練
    for epoch in trange(pre_epoch, desc="CommNet pre Training"):

        test_net.train()

        # 一個batch 產生一個 loss 更新一次網路
        epoch_loss              = []    # 累積N個batch的loss,在輸出log再做平均
        epoch_mse_loss          = []
        epoch_comm_interf_loss  = []

        epoch_nominal = []
        epoch_target_snr = []

        epoch_signal_power = []
        epoch_comm_interf_power = []
        epoch_radar_interf_power = []
        epoch_noise_power = []

        for _ in range(N_BATCHE):
            
            # 從data中抽出 BATCH_CHANNELS 個估計通道
            channel_ids = np.random.choice(train_channels,size=BATCH_CHANNELS,replace=False)

            h_dk_hat = torch.as_tensor(train_dataset["h_dk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)

            optimizer.zero_grad(set_to_none=True)   # 清除梯度，避免上一個 batch 的梯度殘留
            
            # 使用已經訓練的RIS net 拿到theta
            with torch.no_grad():
                theta = theta_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)

                H_eff_H = physics_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)
                W_C_rzf = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)       # 這裡輸出功率正規化 W_C_rzf
                W_R_raw = mrt_in_H_eff_H_nullspace(H_eff_H,g_dt_hat)         # (B,M,1), normalized
            
            W_C_net = test_net(H_eff_H)                             # 正規化的W_C_net

            if epoch == 0 and len(epoch_loss) == 0:
                net_power = torch.mean(torch.sum(torch.abs(W_C_net) ** 2,dim=(1,2)))
                rzf_power = torch.mean(torch.sum(torch.abs(W_C_rzf) ** 2,dim=(1,2)))

                print("\n[TEST NET DEBUG]")
                print(f"H_eff_H shape     = {tuple(H_eff_H.shape)}")
                print(f"W_C_net shape     = {tuple(W_C_net.shape)}")
                print(f"W_C_rzf shape     = {tuple(W_C_rzf.shape)}")
                print(f"H_eff_H raw RMS   = {torch.sqrt(torch.mean(torch.abs(H_eff_H) ** 2)).item():.6e}")
                print(f"H_eff_H input RMS = {torch.sqrt(torch.mean(torch.abs(1.0e4 * H_eff_H) ** 2)).item():.6e}")
                print(f"W_C_net power     = {float(net_power.detach().cpu()):.6f}")
                print(f"W_C_rzf power     = {float(rzf_power.detach().cpu()):.6f}")

            loss,mse_loss,comm_interf_loss = comm_pretrain_loss(H_eff_H,W_C_net,W_C_rzf)

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            # ================================
            # Train ISAC performance
            # 使用更新後的 TestNet + 固定 NS-MRT
            # ================================
            with torch.no_grad():
                W_C_net_metric_raw = test_net(H_eff_H)

                W_C_net_metric,W_R_metric = beamformers_power_split(W_C_net_metric_raw,W_R_raw)

                train_metrics = physics_net.compute_isac_batch_performance(H_eff_H,g_dt_hat,W_C_net_metric,W_R_metric)
                """ 輸出結果：
                        signal       : (B,K)
                        comm_interf  : (B,K)
                        radar_interf : (B,K)
                        noise        : scalar

                        rate          : (B,K)
                        target_snr    : (B,)
                """
                
                # 取輸出結果
                train_nominal_allUE_rate = train_metrics["rate"]              # (B,K)
                train_nominal_target_snr = train_metrics["target_snr"]        # (B,)

                train_nominal_signal = train_metrics["signal"]                # (B,K)
                train_nominal_comm_interf = train_metrics["comm_interf"]      # (B,K)
                train_nominal_radar_interf = train_metrics["radar_interf"]    # (B,K)
                train_nominal_noise = train_metrics["noise"]                  # scalar

                # Nominal = mean_B(min_K(rate))
                train_nominal_with_batch = torch.min(train_nominal_allUE_rate,dim=1,).values    # (B,)
                train_nominal = torch.mean(train_nominal_with_batch)           # scalar

                # Target SNR：先線性平均
                train_target_snr_mean = torch.mean(train_nominal_target_snr)   # scalar

                # SINR power components
                train_signal_mean = torch.mean(train_nominal_signal,dim=0)                 # (K,)
                train_comm_interf_mean = torch.mean(train_nominal_comm_interf,dim=0)       # (K,)
                train_radar_interf_mean = torch.mean(train_nominal_radar_interf,dim=0)     # (K,)


            epoch_loss.append(float(loss.detach().cpu()))
            epoch_mse_loss.append(float(mse_loss.detach().cpu()))
            epoch_comm_interf_loss.append(float(comm_interf_loss.detach().cpu()))

            epoch_nominal.append(float(train_nominal.detach().cpu()))
            epoch_target_snr.append(float(train_target_snr_mean.detach().cpu()))

            epoch_signal_power.append(train_signal_mean.detach().cpu().numpy())
            epoch_comm_interf_power.append(train_comm_interf_mean.detach().cpu().numpy())
            epoch_radar_interf_power.append(train_radar_interf_mean.detach().cpu().numpy())
            epoch_noise_power.append(float(train_nominal_noise.detach().cpu()))

        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "mse_loss": float(np.mean(epoch_mse_loss)),
            "comm_interf_loss": float(np.mean(epoch_comm_interf_loss)),

            "nominal": float(np.mean(epoch_nominal)),
            "target_snr_db": train_target_snr_mean_db,

            "signal_power": np.mean(np.stack(epoch_signal_power,axis=0),axis=0),
            "comm_interf_power": np.mean(np.stack(epoch_comm_interf_power,axis=0),axis=0),
            "radar_interf_power": np.mean(np.stack(epoch_radar_interf_power,axis=0),axis=0),
            "noise_power": float(np.mean(epoch_noise_power)),
        }

        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        test_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

            # 使用已經訓練的RIS net 拿到theta
            theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            H_eff_val   = test_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta_val)
            W_C_rzf_val_raw = make_rzf_beamformer(H_eff_val,RZF_LAMBDA)     # 這裡輸出功率正規化 W_C_rzf
            
            # 輸入網路
            W_C_net_val_raw = test_net(H_eff_val)                           # 正規化的W_C_net

            W_R_val_raw = mrt_in_H_eff_H_nullspace(H_eff_val,g_dt_val)  # 正規化的W_R
 

            # power分配
            W_C_net_val, W_R_val = beamformers_power_split(W_C_net_val_raw,W_R_val_raw)     # power 分配
            W_C_rzf_val, W_R_val = beamformers_power_split(W_C_rzf_val_raw,W_R_val_raw)     # power 分配

            # 計算結果
            val_net_metrics = physics_net.compute_isac_batch_performance(H_eff_val,g_dt_val,W_C_net_val,W_R_val)
            val_rzf_metrics = physics_net.compute_isac_batch_performance(H_eff_val,g_dt_val,W_C_rzf_val,W_R_val)
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

            # 取輸出結果(net)
            net_val_nominal_allUE_rate      = val_net_metrics["rate"]                            # (B,K) 所有UE在每筆估測通道的rate
            net_val_nominal_target_snr      = val_net_metrics["target_snr"]                      # (B,) 每一筆估測通道的target SNR，linear

            net_val_nominal_signal          = val_net_metrics["signal"]                          # (B,K) 各UE的signal power
            net_val_nominal_comm_interf     = val_net_metrics["comm_interf"]                     # (B,K)
            net_val_nominal_radar_interf    = val_net_metrics["radar_interf"]                    # (B,K)
            net_val_nominal_noise           = val_net_metrics["noise"]                           # scalar

            # 取輸出結果(rzf)
            rzf_val_nominal_allUE_rate      = val_rzf_metrics["rate"]                            # (B,K) 所有UE在每筆估測通道的rate
            rzf_val_nominal_target_snr      = val_rzf_metrics["target_snr"]                      # (B,) 每一筆估測通道的target SNR，linear

            rzf_val_nominal_signal          = val_rzf_metrics["signal"]                          # (B,K) 各UE的signal power
            rzf_val_nominal_comm_interf     = val_rzf_metrics["comm_interf"]                     # (B,K)
            rzf_val_nominal_radar_interf    = val_rzf_metrics["radar_interf"]                    # (B,K)
            rzf_val_nominal_noise           = val_rzf_metrics["noise"]                           # scalar


            # Nominal 計算
            val_net_nominal_with_batch = torch.min(net_val_nominal_allUE_rate,dim=1).values      # (B,)
            val_net_nominal = torch.mean(val_net_nominal_with_batch)                             # scalar

            val_rzf_nominal_with_batch = torch.min(rzf_val_nominal_allUE_rate,dim=1).values  # (B,)
            val_rzf_nominal = torch.mean(val_rzf_nominal_with_batch)                         # scalar

            # 顯示用：各 UE 對 B channels 平均
            val_net_rate_user_mean = torch.mean(net_val_nominal_allUE_rate,dim=0)             # (K,)
            val_rzf_rate_user_mean = torch.mean(rzf_val_nominal_allUE_rate,dim=0)         # (K,)

            val_net_signal_mean = torch.mean(net_val_nominal_signal,dim=0)                        # (K,)
            val_rzf_signal_mean = torch.mean(rzf_val_nominal_signal,dim=0)                        # (K,)

            val_net_comm_interf_mean = torch.mean(net_val_nominal_comm_interf,dim=0)              # (K,)
            val_rzf_comm_interf_mean = torch.mean(rzf_val_nominal_comm_interf,dim=0)              # (K,)

            val_net_radar_interf_mean = torch.mean(net_val_nominal_radar_interf,dim=0)            # (K,)
            val_rzf_radar_interf_mean = torch.mean(rzf_val_nominal_radar_interf,dim=0)            # (K,)

            # TestNet target SNR：線性平均後轉 dB
            val_net_target_snr_mean = torch.mean(net_val_nominal_target_snr)
            val_net_target_snr_mean_db = 10.0 * torch.log10(val_net_target_snr_mean.clamp_min(1e-12))

            val_loss,val_mse_loss,val_comm_interf_loss = comm_pretrain_loss(H_eff_val,W_C_net_val_raw,W_C_rzf_val_raw)
        val_logs = {
            "loss": float(val_loss.detach().cpu()),
            "mse_loss": float(val_mse_loss.detach().cpu()),
            "comm_interf_loss": float(val_comm_interf_loss.detach().cpu()),

            "nominal": float(val_net_nominal.detach().cpu()),
            "rzf_nominal": float(val_rzf_nominal.detach().cpu()),
            "target_snr_db": float(val_net_target_snr_mean_db.detach().cpu()),

            # 顯示用：mean_B(rate_k)
            "net_rate_user": val_net_rate_user_mean.detach().cpu().numpy(),
            "rzf_rate_user": val_rzf_rate_user_mean.detach().cpu().numpy(),

            "net_signal_user": val_net_signal_mean.detach().cpu().numpy(),
            "rzf_signal_user": val_rzf_signal_mean.detach().cpu().numpy(),

            "net_comm_interf_user": val_net_comm_interf_mean.detach().cpu().numpy(),
            "rzf_comm_interf_user": val_rzf_comm_interf_mean.detach().cpu().numpy(),

            "net_radar_interf_user": val_net_radar_interf_mean.detach().cpu().numpy(),
            "rzf_radar_interf_user": val_rzf_radar_interf_mean.detach().cpu().numpy(),

            # 儲存 TestNet power-component curves
            "signal_power": val_net_signal_mean.detach().cpu().numpy(),
            "comm_interf_power": val_net_comm_interf_mean.detach().cpu().numpy(),
            "radar_interf_power": val_net_radar_interf_mean.detach().cpu().numpy(),
            "noise_power": float(net_val_nominal_noise.detach().cpu()),
        }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_mse_loss"].append(train_logs["mse_loss"])
        curves["val_mse_loss"].append(val_logs["mse_loss"])

        curves["train_comm_interf_loss"].append(train_logs["comm_interf_loss"])
        curves["val_comm_interf_loss"].append(val_logs["comm_interf_loss"])

        curves["train_nominal"].append(train_logs["nominal"])
        curves["val_nominal"].append(val_logs["nominal"])
        curves["val_rzf_nominal"].append(val_logs["rzf_nominal"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        curves["train_signal_power"].append(train_logs["signal_power"].copy())
        curves["val_signal_power"].append(val_logs["signal_power"].copy())

        curves["train_comm_interf_power"].append(train_logs["comm_interf_power"].copy())
        curves["val_comm_interf_power"].append(val_logs["comm_interf_power"].copy())

        curves["train_radar_interf_power"].append(train_logs["radar_interf_power"].copy())
        curves["val_radar_interf_power"].append(val_logs["radar_interf_power"].copy())

        curves["train_noise_power"].append(train_logs["noise_power"])
        curves["val_noise_power"].append(val_logs["noise_power"])


        np.savez(
            pre_curve_path,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in curves.items()
            },
        )
        
        # early stopping 
        if best_val_loss - val_logs["loss"] > EARLY_STOP_EPS:
            best_val_loss    = val_logs["loss"]
            best_val_epoch   = epoch + 1

            best_val_net_worstUE_rate = val_logs["nominal"]
            best_val_rzf_worstUE_rate = val_logs["rzf_nominal"]
            patience_counter = 0

            test_net.save_model(test_net_ckpt,verbose=False)

        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(
                f"\n[EARLY STOP] "
                f"patience_counter={patience_counter}, "
                f"PATIENCE={PATIENCE}, "
                f"best_val_loss={best_val_loss:.6f}"
            )
            break

        # 印出每個eopch log
        rate_ratio = val_logs["nominal"] / max(val_logs["rzf_nominal"],1e-12)

        print(
            f"\n[Epoch {epoch + 1:03d}/{pre_epoch}] "
            f"TrainLoss={train_logs['loss']:.6e} "
            f"ValLoss={val_logs['loss']:.6e} | "

            f"TrainMSE={train_logs['mse_loss']:.6e} "
            f"ValMSE={val_logs['mse_loss']:.6e} | "

            f"TrainCommLoss={train_logs['comm_interf_loss']:.6e} "
            f"ValCommLoss={val_logs['comm_interf_loss']:.6e} | "

            f"TrainNominal={train_logs['nominal']:.4f} "
            f"ValNetNominal={val_logs['nominal']:.4f} "
            f"ValRZFNominal={val_logs['rzf_nominal']:.4f} "
            f"RateRatio={rate_ratio:.4f}\n"

            f"ValNetRateUser={fmt_vec(val_logs['net_rate_user'],precision=4)} "
            f"ValRZFRateUser={fmt_vec(val_logs['rzf_rate_user'],precision=4)}\n"

            f"ValNetSignal={fmt_vec_sci(val_logs['net_signal_user'],precision=3)} "
            f"ValRZFSignal={fmt_vec_sci(val_logs['rzf_signal_user'],precision=3)}\n"

            f"ValNetCommInterf={fmt_vec_sci(val_logs['net_comm_interf_user'],precision=3)} "
            f"ValRZFCommInterf={fmt_vec_sci(val_logs['rzf_comm_interf_user'],precision=3)}"

            f"\nTrainSignal={fmt_vec_sci(train_logs['signal_power'],precision=3)} "
            f"ValSignal={fmt_vec_sci(val_logs['signal_power'],precision=3)}\n"

            f"TrainCommInterf={fmt_vec_sci(train_logs['comm_interf_power'],precision=3)} "
            f"ValCommInterf={fmt_vec_sci(val_logs['comm_interf_power'],precision=3)}\n"

            f"TrainRadarInterf={fmt_vec_sci(train_logs['radar_interf_power'],precision=3)} "
            f"ValRadarInterf={fmt_vec_sci(val_logs['radar_interf_power'],precision=3)}\n"

            f"TrainNoise={train_logs['noise_power']:.3e} "
            f"ValNoise={val_logs['noise_power']:.3e}"
        )
    plot_pretrain_curves(pre_curve_path,pre_curve_dir)

    best_rate_ratio = (best_val_net_worstUE_rate/ max(best_val_rzf_worstUE_rate, 1e-12))

    print("\n[INFO] CommNet RZF pretraining finished.")
    print("\n[INFO] One-timescale REG training finished.")
    print(f"[INFO] Best validation epoch      = {best_val_epoch}")
    print(f"[INFO] Best validation loss       = {best_val_loss:.6e}")
    print(f"[INFO] Best checkpoint Net rate   = {best_val_net_worstUE_rate:.6f} bps/Hz")
    print(f"[INFO] Corresponding RZF rate     = {best_val_rzf_worstUE_rate:.6f} bps/Hz")
    print(f"[INFO] Net / RZF rate ratio       = {best_rate_ratio:.4f}")  
    print("[INFO] Short-term regular training finished.")

