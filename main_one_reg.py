# -*- coding: utf-8 -*-
import os
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
def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


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


def plot_pretrain_curves(reg_curve_path, reg_curve_dir, ma_window=20):
    """
    畫兩張圖
    1.Train,Val 的 Loss
    2.Train,Val 的 sumrate
    """
    data    = np.load(reg_curve_path)   # curve資料
    out_dir = reg_curve_dir             # 放圖片的資料夾

    # 讀資料
    train_loss    = data["train_loss"]
    val_loss      = data["val_loss"]

    train_sumrate = data["train_sumrate"]
    val_sumrate   = data["val_sumrate"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig 1: Train,Val 的 Loss
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
    plt.title("Fig. 1: One-timescale REG Loss")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig1_path = os.path.join(out_dir, "Fig1_loss_curve.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig1_path}")

    # ================================
    # Fig 2: Train,Val 的 Sumrate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_sumrate,
        label="Train Sum-rate",
        alpha=0.35,
        linewidth=1.0,
    )
    plt.plot(
        epochs,
        val_sumrate,
        label="Validation Sum-rate",
        alpha=0.35,
        linewidth=1.0,
    )

    train_ma_epochs, train_sumrate_ma = moving_average(train_sumrate, ma_window)
    val_ma_epochs, val_sumrate_ma = moving_average(val_sumrate, ma_window)

    if len(train_sumrate_ma) > 0:
        plt.plot(
            train_ma_epochs,
            train_sumrate_ma,
            label=f"Train Sum-rate MA({ma_window})",
            linewidth=2.5,
        )

    if len(val_sumrate_ma) > 0:
        plt.plot(
            val_ma_epochs,
            val_sumrate_ma,
            label=f"Validation Sum-rate MA({ma_window})",
            linewidth=2.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Sum-rate (bps/Hz)")
    plt.title("Fig. 2: One-timescale REG Sum-rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_sumrate_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")




# ================================
# Main
# ================================
if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")

    train_dataset = physics_net.load_channel_dataset(train_dataset_path, "train")
    val_dataset   = physics_net.load_channel_dataset(val_dataset_path, "val")
    
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path   = {val_dataset_path}")

    # ================================
    # Train REG
    # ================================
    comm_net  = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)

    optimizer = optim.Adam(list(comm_net.parameters())+ list(radar_net.parameters())+ list(theta_net.parameters()),lr=REG_LEARNING_RATE)

    reg_comm_ckpt  = os.path.join(REG_CKPT_DIR, "one_timescale_comm_reg.ckpt")
    reg_radar_ckpt = os.path.join(REG_CKPT_DIR, "one_timescale_radar_reg.ckpt")
    reg_theta_ckpt = os.path.join(REG_CKPT_DIR, "one_timescale_theta_reg.ckpt")
    reg_curve_path = os.path.join(REG_CKPT_DIR, "one_timescale_reg_curves.npz")
    reg_curve_dir  = os.path.join(REG_CKPT_DIR, "reg_training_curves")

    os.makedirs(reg_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],               # Loss function
        "val_loss": [],

        "train_sumrate": [],            # sumerate
        "val_sumrate": [],

        "train_sensing_penalty": [],    # 感測懲罰    
        "val_sensing_penalty": [],

        "train_target_snr_db": [],      # 感測SNR
        "val_target_snr_db": [],
    }


    # 訓練參數
    best_val_loss = float("inf")   # L_best <- infinity
    best_val_sumrate = 0.0         # best loss 對應的 validation sumrate
    best_val_epoch = 0             # best loss 出現在哪個 epoch

    patience_counter = 0           
    EARLY_STOP_EPS = 1e-6          
    PATIENCE = 30                  

    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0]
    val_channels   = val_dataset["h_dk_hat"].shape[0]

    print("\n" + "=" * 90)
    print("[ONE TIMESCALE REG TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"REG_EPOCHS              : {REG_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"BATCH_CHANNELS          : {BATCH_CHANNELS}")
    print(f"REG learning rate       : {REG_LEARNING_RATE}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {REG_SENSING_LOSS_WEIGHT}")
    print("=" * 90)

    # 開始訓練
    for epoch in trange(REG_EPOCHS, desc="REG Training"):
        
        comm_net.train()
        radar_net.train()
        theta_net.train()

        epoch_loss              = []    # 單一epoch的 N_BATCHE * BATCH_CHANNELS 所平均的LOSS
        epoch_sumrate           = []
        epoch_sensing_penalties = []
        epoch_target_snr_db     = []

        for _ in range(N_BATCHE):
            
            # 從data中抽出 BATCH_CHANNELS 個估計通道
            channel_ids = np.random.choice(train_channels,size=BATCH_CHANNELS,replace=False)

            h_dk_hat = torch.as_tensor(train_dataset["h_dk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)

            optimizer.zero_grad(set_to_none=True)   # 清除梯度，避免上一個 batch 的梯度殘留

            theta = theta_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)
            W_C_raw   = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)  # 正規化
            W_R_raw   = radar_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat) # 正規化

            # W_C, W_R = physics_net.normalize_isac_beamformers(W_C, W_R, g_dt_hat)           # 這裡怎麼分tx power是個問題

            W_C,W_R = beamformers_power_split(W_C_raw,W_R_raw)      # power 分配

            H_eff_H = physics_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)

            metrics = comm_net.compute_isac_batch_performance(H_eff_H,g_dt_hat,W_C,W_R)
            
            # 計算結果
            rate_user         = metrics["rate_user_mean"]       # 各UE rate
            sumrate_mean      = metrics["sumrate_mean"]         # sum  rate (scalar)
            target_snr_db     = metrics["target_snr_db"]        # 感測SNR    (scalar)

            sensing_violation = torch.relu(SENSING_SNR_THRESHOLD_DB - target_snr_db)    # (B,)
            sensing_penalty   = torch.mean(sensing_violation)                           # scalar

            loss = -(sumrate_mean) + REG_SENSING_LOSS_WEIGHT  * sensing_penalty

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            epoch_loss.append(float(loss.detach().cpu()))                       
            epoch_sumrate.append(float(sumrate_mean.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))
            epoch_target_snr_db.append(float(metrics["target_snr_mean_db"].detach().cpu()))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "sumrate": float(np.mean(epoch_sumrate)),
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
            "target_snr_db": float(np.mean(epoch_target_snr_db)),
            # 每個UE的rate
            "rate_user":rate_user.detach().cpu().numpy(),
        }

        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        comm_net.eval()
        radar_net.eval()
        theta_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)


            theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)
            W_C_val_raw = comm_net(h_dk_val,h_rk_val,G_val,g_dt_val)
            W_R_val_raw = radar_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            # W_C_val, W_R_val = comm_net.normalize_isac_beamformers(W_C_val,W_R_val,g_dt_val)
            W_C_val, W_R_val = beamformers_power_split(W_C_val_raw,W_R_val_raw)

            H_eff_val   = comm_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta_val)

            val_metrics = comm_net.compute_isac_batch_performance(H_eff_val,g_dt_val,W_C_val,W_R_val)
            # 計算結果
            val_rate_user         = val_metrics["rate_user_mean"]       # 各UE rate (K,)
            val_sumrate           = val_metrics["sumrate_mean"]         # sum  rate (scalar)
            val_target_snr_db     = val_metrics["target_snr_db"]        # 感測SNR    (B,)
            val_sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - val_metrics["target_snr"])   # (B,)
            val_sensing_penalty   = torch.mean(val_sensing_violation )                              # scalar

            val_loss = -(val_sumrate)+ REG_SENSING_LOSS_WEIGHT * val_sensing_penalty

            val_logs = {
                "loss": float(val_loss.detach().cpu()),

                "sumrate": float(val_sumrate.detach().cpu()),
                "sensing_penalty": float(val_sensing_penalty.detach().cpu()),
                "target_snr_db": float(val_metrics["target_snr_mean_db"].detach().cpu()),
                # 每個UE的rate
                "rate_user": val_rate_user.detach().cpu().numpy(),
            }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_sumrate"].append(train_logs["sumrate"])
        curves["val_sumrate"].append(val_logs["sumrate"])

        curves["train_sensing_penalty"].append(train_logs["sensing_penalty"])
        curves["val_sensing_penalty"].append(val_logs["sensing_penalty"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        np.savez(
            reg_curve_path,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in curves.items()
            },
        )
        # early stopping 
        if best_val_loss - val_logs["loss"] > EARLY_STOP_EPS:
            best_val_loss    = val_logs["loss"]
            best_val_sumrate = val_logs["sumrate"]
            best_val_epoch   = epoch + 1

            patience_counter = 0

            comm_net.save_model(reg_comm_ckpt, verbose=False)
            radar_net.save_model(reg_radar_ckpt, verbose=False)
            theta_net.save_model(reg_theta_ckpt, verbose=False)
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

            f"Trainsumrate={train_logs['sumrate']:.4f} "
            f"Valsumrate={val_logs['sumrate']:.4f} | "

            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "

            f"\nTrainrateUser={fmt_vec(train_logs['rate_user'], precision=3)} "
            f"ValrateUser={fmt_vec(val_logs['rate_user'], precision=3)} "
        )

    plot_pretrain_curves(reg_curve_path,reg_curve_dir)



    print("\n[INFO] One-timescale REG training finished.")
    print(f"[INFO] Best validation epoch   = {best_val_epoch}")
    print(f"[INFO] Best validation loss    = {best_val_loss:.6f}")
    print(f"[INFO] Best validation sumrate = {best_val_sumrate:.6f}")       # 這裡要print Best validation sumrate
    print("[INFO] Short-term regular training finished.")




