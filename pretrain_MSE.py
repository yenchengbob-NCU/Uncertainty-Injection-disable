# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import make_rzf_beamformer, RZF_LAMBDA
from 暫時用不到.one_timescale_NN import CommNet, ThetaNet

# ================================
# Helpers
# ================================
PRETRAIN_COMM_INTERF_WEIGHT = 0.0

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


def plot_pretrain_curves(pre_curve_path, pre_curve_dir):
    """
    畫四張 CommNet pretraining curve：

    1. Total loss
    2. MSE loss
    3. Communication interference loss
    4. Validation Net / RZF worst-UE rate
    """
    os.makedirs(pre_curve_dir, exist_ok=True)

    with np.load(pre_curve_path) as data:
        train_loss = data["train_loss"]
        val_loss = data["val_loss"]

        train_mse_loss = data["train_mse_loss"]
        val_mse_loss = data["val_mse_loss"]

        train_comm_interf_loss = data["train_comm_interf_loss"]
        val_comm_interf_loss = data["val_comm_interf_loss"]

        val_net_worstUE_rate = data["val_net_worstUE_rate"]
        val_rzf_worstUE_rate = data["val_rzf_worstUE_rate"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig 1: Total loss
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, label="Train Total Loss", linewidth=1.5)
    plt.plot(epochs, val_loss, label="Validation Total Loss", linewidth=1.5)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CommNet RZF Pretraining Total Loss")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    total_loss_fig_path = os.path.join(
        pre_curve_dir,
        "pretrain_total_loss_curve.png",
    )

    plt.savefig(total_loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {total_loss_fig_path}")

    # ================================
    # Fig 2: MSE loss
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_mse_loss, label="Train MSE Loss", linewidth=1.5)
    plt.plot(epochs, val_mse_loss, label="Validation MSE Loss", linewidth=1.5)

    if np.all(train_mse_loss > 0) and np.all(val_mse_loss > 0):
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Complex MSE")
    plt.title("CommNet RZF Pretraining MSE Loss")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    mse_fig_path = os.path.join(
        pre_curve_dir,
        "pretrain_mse_loss_curve.png",
    )

    plt.savefig(mse_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {mse_fig_path}")

    # ================================
    # Fig 3: Communication interference loss
    # ================================
    plt.figure(figsize=(10, 6))

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
    plt.title("CommNet RZF Pretraining Communication Interference Loss")
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    comm_interf_fig_path = os.path.join(
        pre_curve_dir,
        "pretrain_comm_interf_loss_curve.png",
    )

    plt.savefig(comm_interf_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {comm_interf_fig_path}")

    # ================================
    # Fig 4: Validation worst-UE rate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        val_net_worstUE_rate,
        label="Validation Net Worst-UE Rate",
        linewidth=1.5,
    )

    plt.plot(
        epochs,
        val_rzf_worstUE_rate,
        label="Validation RZF Worst-UE Rate",
        linewidth=1.5,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Worst-UE Rate (bps/Hz)")
    plt.title("CommNet and RZF Validation Worst-UE Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    worstUE_rate_fig_path = os.path.join(
        pre_curve_dir,
        "pretrain_worstUE_rate_curve.png",
    )

    plt.savefig(worstUE_rate_fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved: {worstUE_rate_fig_path}")


def compute_comm_only_rate(H_eff_H, W_C):
    """
    純通訊 downlink MISO performance不考慮 W_R。

    Input:
        H_eff_H : (B,K,M), complex
        W_C     : (B,M,K), complex

    Return:
        rate              : (B,K)
        worstUE_rate      : (B,)
        worstUE_rate_mean : scalar
        rate_user_mean    : (K,)
        signal            : (B,K)
        comm_interf       : (B,K)
        sinr              : (B,K)

    sinr = signal / (comm_interf + noise)
    """

    H_eff_H = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=DEVICE)  # (B,K,M)
    W_C     = torch.as_tensor(W_C,dtype=torch.complex64,device=DEVICE)          # (B,M,K)
    noise   = torch.as_tensor(NOISE_POWER,dtype=torch.float32,device=DEVICE)  # scalar

    Y_C = torch.matmul(H_eff_H,W_C)                       # (B,K,K)
    P_C = torch.abs(Y_C) ** 2                            # (B,K,K)

    signal = torch.diagonal(P_C,dim1=1,dim2=2)           # (B,K)
    comm_interf = torch.sum(P_C,dim=2) - signal          # (B,K)

    sinr = signal / (comm_interf + noise)                # (B,K)
    rate = torch.log2(1.0 + sinr)                        # (B,K)

    worstUE_rate = torch.min(rate,dim=1).values           # (B,)
    worstUE_rate_mean = torch.mean(worstUE_rate)          # scalar
    rate_user_mean = torch.mean(rate,dim=0)               # (K,)

    return {
        "rate": rate,
        "worstUE_rate": worstUE_rate,
        "worstUE_rate_mean": worstUE_rate_mean,
        "rate_user_mean": rate_user_mean,
        "signal": signal,
        "comm_interf": comm_interf,
        "sinr": sinr,
        "noise": noise,
    }


def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def fmt_vec_sci(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"




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

    # ================================
    # Pre train comm net
    # ================================
    comm_net  = CommNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)
    optimizer = optim.Adam(list(comm_net.parameters()),lr=0.001)

    pre_comm_ckpt  = os.path.join(PRETRAIN_DIR, "pre_comm_reg.ckpt")
    ris_only_ckpt  = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    pre_curve_path = os.path.join(PRETRAIN_DIR, "ris_only.npz")
    pre_curve_dir  = os.path.join(PRETRAIN_DIR, "ris_only_curves")

    os.makedirs(pre_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],
        "val_loss": [],

        "train_mse_loss": [],
        "val_mse_loss": [],

        "train_comm_interf_loss": [],
        "val_comm_interf_loss": [],

        "val_net_worstUE_rate": [],
        "val_rzf_worstUE_rate": [],
    }

    # 訓練參數
    pre_epoch = 1000

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

        comm_net.train()

        # 一個batch 產生一個 loss 更新一次網路
        epoch_loss              = []    # 累積N個batch的loss,在輸出log再做平均
        epoch_mse_loss          = []
        epoch_comm_interf_loss  = []

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

            W_C_net = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)    # 正規化的W_C_net

            loss,mse_loss,comm_interf_loss = comm_pretrain_loss(H_eff_H,W_C_net,W_C_rzf)

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            epoch_loss.append(float(loss.detach().cpu()))    
            epoch_mse_loss.append(float(mse_loss.detach().cpu()))
            epoch_comm_interf_loss.append(float(comm_interf_loss.detach().cpu())) 

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "mse_loss": float(np.mean(epoch_mse_loss)),
            "comm_interf_loss": float(np.mean(epoch_comm_interf_loss)),
        }

        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        comm_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

            # 使用已經訓練的RIS net 拿到theta
            theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            H_eff_val   = comm_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta_val)
            W_C_rzf_val = make_rzf_beamformer(H_eff_val,RZF_LAMBDA)     # 這裡輸出功率正規化 W_C_rzf
            
            # 輸入網路
            W_C_net_val = comm_net(h_dk_val,h_rk_val,G_val,g_dt_val)    # 正規化的W_C_net

            # power分配4成
            p_C = torch.as_tensor(0.4 * TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)

            W_C_net_val_power = torch.sqrt(p_C) * W_C_net_val
            W_C_rzf_val_power = torch.sqrt(p_C) * W_C_rzf_val

            val_net_metrics = compute_comm_only_rate(H_eff_val,W_C_net_val_power)
            val_rzf_metrics = compute_comm_only_rate(H_eff_val,W_C_rzf_val_power)
            """ 輸出結果 :
                    rate              : (B,K)
                    worstUE_rate      : (B,)
                    worstUE_rate_mean : scalar
                    rate_user_mean    : (K,)
                    signal            : (B,K)
                    comm_interf       : (B,K)
                    sinr              : (B,K)
            """

            # 取輸出結果
            val_nominal_allUE_rate = val_net_metrics["rate"]          # (B,K)
            val_rzf_nominal_allUE_rate = val_rzf_metrics["rate"]      # (B,K)

            val_net_signal = val_net_metrics["signal"]                # (B,K)
            val_rzf_signal = val_rzf_metrics["signal"]                # (B,K)

            val_net_comm_interf = val_net_metrics["comm_interf"]      # (B,K)
            val_rzf_comm_interf = val_rzf_metrics["comm_interf"]      # (B,K)


            # Nominal 計算
            val_nominal_with_batch = torch.min(val_nominal_allUE_rate,dim=1).values      # (B,)
            val_nominal = torch.mean(val_nominal_with_batch)                             # scalar

            val_rzf_nominal_with_batch = torch.min(val_rzf_nominal_allUE_rate,dim=1).values  # (B,)
            val_rzf_nominal = torch.mean(val_rzf_nominal_with_batch)                         # scalar

            # 顯示用：各 UE 對 B channels 平均
            val_net_rate_user_mean = torch.mean(val_nominal_allUE_rate,dim=0)             # (K,)
            val_rzf_rate_user_mean = torch.mean(val_rzf_nominal_allUE_rate,dim=0)         # (K,)

            val_net_signal_mean = torch.mean(val_net_signal,dim=0)                        # (K,)
            val_rzf_signal_mean = torch.mean(val_rzf_signal,dim=0)                        # (K,)

            val_net_comm_interf_mean = torch.mean(val_net_comm_interf,dim=0)              # (K,)
            val_rzf_comm_interf_mean = torch.mean(val_rzf_comm_interf,dim=0)              # (K,)

            val_loss,val_mse_loss,val_comm_interf_loss = comm_pretrain_loss(H_eff_val,W_C_net_val,W_C_rzf_val,)

        val_logs = {
            "loss": float(val_loss.detach().cpu()),
            "mse_loss": float(val_mse_loss.detach().cpu()),
            "comm_interf_loss": float(val_comm_interf_loss.detach().cpu()),

            # Nominal = mean_B(min_K(rate))
            "net_worstUE_rate": float(val_nominal.detach().cpu()),
            "rzf_worstUE_rate": float(val_rzf_nominal.detach().cpu()),

            # 顯示用：mean_B(rate_k)
            "net_rate_user": val_net_rate_user_mean.detach().cpu().numpy(),
            "rzf_rate_user": val_rzf_rate_user_mean.detach().cpu().numpy(),

            "net_signal_user": val_net_signal_mean.detach().cpu().numpy(),
            "rzf_signal_user": val_rzf_signal_mean.detach().cpu().numpy(),

            "net_comm_interf_user": val_net_comm_interf_mean.detach().cpu().numpy(),
            "rzf_comm_interf_user": val_rzf_comm_interf_mean.detach().cpu().numpy(),
        }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["val_net_worstUE_rate"].append(val_logs["net_worstUE_rate"])
        curves["val_rzf_worstUE_rate"].append(val_logs["rzf_worstUE_rate"])

        curves["train_mse_loss"].append(train_logs["mse_loss"])
        curves["val_mse_loss"].append(val_logs["mse_loss"])

        curves["train_comm_interf_loss"].append(train_logs["comm_interf_loss"])
        curves["val_comm_interf_loss"].append(val_logs["comm_interf_loss"])


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

            best_val_net_worstUE_rate = val_logs["net_worstUE_rate"]
            best_val_rzf_worstUE_rate = val_logs["rzf_worstUE_rate"]
            patience_counter = 0

            comm_net.save_model(pre_comm_ckpt, verbose=False)

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
        rate_ratio = val_logs["net_worstUE_rate"] / max(val_logs["rzf_worstUE_rate"],1e-12)

        print(
            f"\n[Epoch {epoch + 1:03d}/{pre_epoch}] "
            f"TrainLoss={train_logs['loss']:.6e} "
            f"ValLoss={val_logs['loss']:.6e} | "

            f"TrainMSE={train_logs['mse_loss']:.6e} "
            f"ValMSE={val_logs['mse_loss']:.6e} | "

            f"TrainCommLoss={train_logs['comm_interf_loss']:.6e} "
            f"ValCommLoss={val_logs['comm_interf_loss']:.6e} | "

            f"ValNetMinRate={val_logs['net_worstUE_rate']:.4f} "
            f"ValRZFMinRate={val_logs['rzf_worstUE_rate']:.4f} "
            f"RateRatio={rate_ratio:.4f}\n"

            f"ValNetRateUser={fmt_vec(val_logs['net_rate_user'],precision=4)} "
            f"ValRZFRateUser={fmt_vec(val_logs['rzf_rate_user'],precision=4)}\n"

            f"ValNetSignal={fmt_vec_sci(val_logs['net_signal_user'],precision=3)} "
            f"ValRZFSignal={fmt_vec_sci(val_logs['rzf_signal_user'],precision=3)}\n"

            f"ValNetCommInterf={fmt_vec_sci(val_logs['net_comm_interf_user'],precision=3)} "
            f"ValRZFCommInterf={fmt_vec_sci(val_logs['rzf_comm_interf_user'],precision=3)}"
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

