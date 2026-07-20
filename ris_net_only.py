# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, beamformers_power_split, RZF_LAMBDA
from two_timescale_NN import ThetaNet

# ================================
# Helpers
# ================================
def moving_average(x, window):
    """
    Valid moving average.
    回傳:
        ma_epochs : 對應 moving average 的 epoch index
        y_ma      : moving average 後的曲線
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    window = int(window)

    if len(x) < window:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    kernel = np.ones(window, dtype=np.float64) / window
    y_ma = np.convolve(x, kernel, mode="valid")
    ma_epochs = np.arange(window, len(x) + 1)

    return ma_epochs, y_ma


def plot_theta_pretrain_curves(curve_path, curve_dir, ma_window=20):
    """
    畫兩類圖：
    1. Train / Validation Loss
    2. Train / Validation Nominal worst-UE rate
    """
    data = np.load(curve_path)
    out_dir = curve_dir

    os.makedirs(out_dir, exist_ok=True)

    # ================================
    # 讀取基本 curves
    # ================================
    train_loss = data["train_loss"]
    val_loss   = data["val_loss"]

    train_nominal = data["train_nominal"]
    val_nominal   = data["val_nominal"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig. 1: Train / Validation Loss
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
        alpha=0.35,
        linewidth=1.0,
    )
    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        alpha=0.35,
        linewidth=1.0,
    )

    train_ma_epochs, train_loss_ma = moving_average(train_loss, ma_window)
    val_ma_epochs, val_loss_ma = moving_average(val_loss, ma_window)

    if len(train_loss_ma) > 0:
        plt.plot(
            train_ma_epochs,
            train_loss_ma,
            label=f"Train Loss MA({ma_window})",
            linewidth=2.5,
        )

    if len(val_loss_ma) > 0:
        plt.plot(
            val_ma_epochs,
            val_loss_ma,
            label=f"Validation Loss MA({ma_window})",
            linewidth=2.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fig. 1: ThetaNet Pretraining Loss")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig1_path = os.path.join(out_dir, "Fig1_loss_curve.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig1_path}")

    # ================================
    # Fig. 2: Nominal Worst-UE Rate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_nominal,
        label="Train Nominal Worst-UE Rate",
        alpha=0.35,
        linewidth=1.0,
    )
    plt.plot(
        epochs,
        val_nominal,
        label="Validation Nominal Worst-UE Rate",
        alpha=0.35,
        linewidth=1.0,
    )

    train_ma_epochs, train_nominal_ma = moving_average(train_nominal, ma_window)
    val_ma_epochs, val_nominal_ma = moving_average(val_nominal, ma_window)

    if len(train_nominal_ma) > 0:
        plt.plot(
            train_ma_epochs,
            train_nominal_ma,
            label=f"Train Nominal MA({ma_window})",
            linewidth=2.5,
        )

    if len(val_nominal_ma) > 0:
        plt.plot(
            val_ma_epochs,
            val_nominal_ma,
            label=f"Validation Nominal MA({ma_window})",
            linewidth=2.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Nominal Worst-UE Rate (bps/Hz)")
    plt.title("Fig. 2: ThetaNet Nominal Worst-UE Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_nominal_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    # 這裡不使用net 只是要用neural_net.py的副函式
    theta_net = ThetaNet().to(DEVICE)

    # 讀取資料
    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")

    train_dataset = theta_net.load_channel_dataset(train_dataset_path, "train")
    val_dataset   = theta_net.load_channel_dataset(val_dataset_path, "val")
    
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path   = {val_dataset_path}")

    # ================================
    # Train RIS net only
    # ================================
    optimizer = optim.Adam(list(theta_net.parameters()),lr=REG_LEARNING_RATE)

    pre_theta_ckpt = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    pre_curve_path = os.path.join(PRETRAIN_DIR, "ris_only.npz")
    pre_curve_dir  = os.path.join(PRETRAIN_DIR, "ris_only")

    os.makedirs(pre_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],               # Loss function
        "val_loss": [],

        "train_nominal": [],            # Nominal ≜ 先找出K個UE中 worst UE rate,再對B channel平均
        "val_nominal": [],

        "train_target_snr_db": [],      # 感測SNR 不畫圖
        "val_target_snr_db": [],

        "train_sensing_penalty": [],    # 感測懲罰 不畫圖
        "val_sensing_penalty": [],
    }


    # 訓練參數
    best_val_loss = float("inf")   # L_best <- infinity
    best_val_worstUE_rate = 0.0    # best validation loss 對應的 nominal worst-UE rate
    best_valtarget_snr_db = 0.0
    best_val_epoch = 0             # best loss 出現在哪個 epoch

    patience_counter = 0           
    EARLY_STOP_EPS = 1e-6          
    PATIENCE = 30                  

    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0]
    val_channels   = val_dataset["h_dk_hat"].shape[0]

    # 開始訓練
    for epoch in trange(REG_EPOCHS,desc="ThetaNet Pretraining"):
        
        theta_net.train()
        # 一個batch 產生一個 loss 更新一次網路
        epoch_loss              = []    # 累積N個batch的loss
        epoch_Nominal           = []    # 累積N個batch的Nominal
        epoch_target_snr        = []
        epoch_sensing_penalties = []

        for _ in range(N_BATCHE):
            
            # 從data中抽出 BATCH_CHANNELS 個估計通道
            channel_ids = np.random.choice(train_channels,size=BATCH_CHANNELS,replace=False)

            h_dk_hat = torch.as_tensor(train_dataset["h_dk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)

            optimizer.zero_grad(set_to_none=True)   # 清除梯度，避免上一個 batch 的梯度殘留

            theta = theta_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)

            H_eff_H = theta_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)

            W_C_dir = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)               # 這裡輸出功率正規化 W_C_dir

            W_R_dir = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt_hat)           # 這裡輸出功率正規化 W_R_dir

            W_C,W_R = beamformers_power_split(W_C_dir,W_R_dir)      # power 分配

            metrics = theta_net.compute_isac_batch_performance(H_eff_H,g_dt_hat,W_C,W_R)
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
            
            # Nominal 計算
            nominal_with_batch    = torch.min(nominal_allUE_rate, dim=1).values     # (B,)   每一筆估測通道，找出所有UE中 rate中最小的
            nominal = torch.mean(nominal_with_batch)                                # scalar 對B筆估測通道min-rate平均

            # Nominal target SNR：先計算目前batch的linear mean
            nominal_target_snr_mean = torch.mean(nominal_target_snr)       # scalar, linear

            # 感測懲罰：每個channel使用linear SNR計算violation，再對batch平均
            sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - nominal_target_snr)  # (B,)
            sensing_penalty = torch.mean(sensing_violation)   

            # loss function
            loss = -(nominal) + REG_SENSING_LOSS_WEIGHT  * sensing_penalty

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            epoch_loss.append(float(loss.detach().cpu()))
            epoch_Nominal.append(float(nominal.detach().cpu()))
            epoch_target_snr.append(float(nominal_target_snr_mean.detach().cpu()))     
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))

        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "nominal": float(np.mean(epoch_Nominal)),
            "target_snr_db": train_target_snr_mean_db,
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
        }

        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        theta_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)


            theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            H_eff_val   = theta_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta_val)

            W_C_val_dir = make_rzf_beamformer(H_eff_val,RZF_LAMBDA)
            W_R_val_dir = mrt_in_H_eff_H_nullspace(H_eff_val,g_dt_val)

            W_C_val, W_R_val = beamformers_power_split(W_C_val_dir,W_R_val_dir)

            
            val_metrics = theta_net.compute_isac_batch_performance(H_eff_val,g_dt_val,W_C_val,W_R_val)
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
            val_nominal_allUE_rate      = val_metrics["rate"]                            # (B,K) 所有UE在每筆估測通道的rate
            val_nominal_target_snr      = val_metrics["target_snr"]                      # (B,) 每一筆估測通道的target SNR，linear

            # Nominal計算
            val_nominal_with_batch = torch.min(val_nominal_allUE_rate,dim=1).values     # (B,) 每一筆估測通道找出所有UE中rate最小的
            val_nominal = torch.mean(val_nominal_with_batch)                            # scalar 對B筆估測通道min-rate平均
            
            # Nominal target SNR：linear mean後轉dB
            val_nominal_target_snr_mean = torch.mean(val_nominal_target_snr)            # scalar, linear
            val_target_snr_mean_db = 10.0 * torch.log10(val_nominal_target_snr_mean.clamp_min(1e-12))

            # 感測懲罰：每個channel使用linear SNR計算violation，再對validation channels平均
            val_sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - val_nominal_target_snr)  # (B,)
            val_sensing_penalty = torch.mean(val_sensing_violation)                             # scalar

            # Loss function
            val_loss = -(val_nominal) + REG_SENSING_LOSS_WEIGHT * val_sensing_penalty
            
            val_logs = {
                "loss": float(val_loss.detach().cpu()),
                "nominal": float(val_nominal.detach().cpu()),
                "target_snr_db": float(val_target_snr_mean_db.detach().cpu()),
                "sensing_penalty": float(val_sensing_penalty.detach().cpu()),
            }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_nominal"].append(train_logs["nominal"])
        curves["val_nominal"].append(val_logs["nominal"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        curves["train_sensing_penalty"].append(train_logs["sensing_penalty"])
        curves["val_sensing_penalty"].append(val_logs["sensing_penalty"])

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
            best_val_worstUE_rate = val_logs["nominal"]
            best_valtarget_snr_db = val_logs["target_snr_db"]
            best_val_epoch   = epoch + 1

            patience_counter = 0

            theta_net.save_model(pre_theta_ckpt, verbose=False)
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
        print(
            f"\n[Epoch {epoch + 1:03d}/{REG_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "

            f"TrainNominal={train_logs['nominal']:.4f} "
            f"ValNominal={val_logs['nominal']:.4f} | "

            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "

            f"\nTrainpenalty={train_logs['sensing_penalty']:.3f} "
            f"Valpenalty={val_logs['sensing_penalty']:.3f} | "
        )

    plot_theta_pretrain_curves(pre_curve_path,pre_curve_dir)



    print("\n[INFO] ThetaNet nominal pretraining finished.")
    print(f"[INFO] Sensing SNR threshold     = {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"[INFO] Sensing penalty weight    = {REG_SENSING_LOSS_WEIGHT}")
    print(f"[INFO] Best validation epoch     = {best_val_epoch}")
    print(f"[INFO] Best validation loss      = {best_val_loss:.6f}")
    print(f"[INFO] Nominal worst-UE rate     = {best_val_worstUE_rate:.6f}")
    print(f"[INFO] Nominal target SNR        = {best_valtarget_snr_db:.3f} dB")



