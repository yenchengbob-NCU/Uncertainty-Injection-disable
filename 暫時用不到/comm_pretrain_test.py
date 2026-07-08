# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import make_random_ris ,make_rzf_beamformer, RZF_LAMBDA
from one_timescale_NN import CommNet ,TestNet

# ================================
# Helpers
# ================================
def plot_one_timescale_reg_curves(curve_path):
    """
    畫兩張圖
    1.NN W_C 與 RZF 的 MSE
    2.NN W_C 與 RZF 的 power 差距
    """
    data = np.load(curve_path)
    out_dir = os.path.dirname(curve_path)

    train_mse = data["train_mse"]
    val_mse = data["val_mse"]

    train_wc_power = data["train_wc_power"]
    val_wc_power = data["val_wc_power"]
    train_rzf_power = data["train_rzf_power"]
    val_rzf_power = data["val_rzf_power"]

    epochs = np.arange(1, len(train_mse) + 1)

    # ================================
    # Fig 1: MSE curve
    # ================================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mse, alpha=0.35, label="Train MSE")
    plt.plot(epochs, val_mse, alpha=0.35, label="Val MSE")

    if np.all(train_mse > 0) and np.all(val_mse > 0):
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("CommNet RZF-like W_C Pretrain MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    mse_fig_path = os.path.join(out_dir, "pretrain_comm_mse.png")
    plt.savefig(mse_fig_path, dpi=200)
    plt.close()

    # ================================
    # Fig 2: W_C power curve
    # ================================
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_wc_power, alpha=0.35, label="Train NN W_C Power")
    plt.plot(epochs, val_wc_power, alpha=0.35, label="Val NN W_C Power")
    plt.plot(epochs, train_rzf_power, alpha=0.35, label="Train RZF W_C Power")
    plt.plot(epochs, val_rzf_power, alpha=0.35, label="Val RZF W_C Power")

    all_power = np.concatenate(
        [
            train_wc_power,
            val_wc_power,
            train_rzf_power,
            val_rzf_power,
        ]
    )

    if np.all(all_power > 0):
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("Power")
    plt.title("CommNet W_C Power During RZF-like Pretrain")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    power_fig_path = os.path.join(out_dir, "pretrain_comm_power.png")
    plt.savefig(power_fig_path, dpi=200)
    plt.close()

    print(f"[plot] saved: {mse_fig_path}")
    print(f"[plot] saved: {power_fig_path}")


def compute_zf_like_metrics(H_eff_H, W_C, W_C_zf, eps=1e-12):
    """
    H_eff_H : (B, K, M)
    W_C     : (B, M, K), NN output, normalized direction
    W_C_zf  : (B, M, K), ZF/RZF label, normalized direction
    """

    # ================================
    # 1. Raw W_C MSE
    # ================================
    raw_mse = torch.mean(torch.abs(W_C - W_C_zf) ** 2)

    # ================================
    # 2. Column-wise phase alignment
    # ================================
    inner = torch.sum(torch.conj(W_C_zf) * W_C, dim=1, keepdim=True)  # (B,1,K)
    phase = inner / (torch.abs(inner) + eps)

    W_C_aligned = W_C * torch.conj(phase)

    phase_mse = torch.mean(torch.abs(W_C_aligned - W_C_zf) ** 2)

    # ================================
    # 3. Complex cosine similarity
    # ================================
    inner2 = torch.sum(torch.conj(W_C_zf) * W_C, dim=1)  # (B,K)

    pred_power = torch.sum(torch.abs(W_C) ** 2, dim=1)       # (B,K)
    zf_power   = torch.sum(torch.abs(W_C_zf) ** 2, dim=1)    # (B,K)

    cos2 = torch.abs(inner2) ** 2 / (pred_power * zf_power + eps)  # (B,K)

    mean_cos2 = torch.mean(cos2)
    min_cos2 = torch.mean(torch.min(cos2, dim=1).values)
    user_cos2 = torch.mean(cos2, dim=0)  # (K,)

    # ================================
    # 4. Effective matrix
    # ================================
    Y_nn = torch.matmul(H_eff_H, W_C)      # (B,K,K)
    Y_zf = torch.matmul(H_eff_H, W_C_zf)   # (B,K,K)

    effective_nmse = torch.mean(
        torch.sum(torch.abs(Y_nn - Y_zf) ** 2, dim=(1, 2))
        / (torch.sum(torch.abs(Y_zf) ** 2, dim=(1, 2)) + eps)
    )

    # ================================
    # 5. ZF leakage ratio
    # ================================
    K = H_eff_H.shape[1]
    eye = torch.eye(K, dtype=torch.bool, device=H_eff_H.device).unsqueeze(0)

    Y_nn_power = torch.abs(Y_nn) ** 2
    Y_zf_power = torch.abs(Y_zf) ** 2

    nn_diag = torch.sum(Y_nn_power.masked_fill(~eye, 0.0), dim=(1, 2))
    nn_off  = torch.sum(Y_nn_power.masked_fill(eye, 0.0), dim=(1, 2))

    zf_diag = torch.sum(Y_zf_power.masked_fill(~eye, 0.0), dim=(1, 2))
    zf_off  = torch.sum(Y_zf_power.masked_fill(eye, 0.0), dim=(1, 2))

    nn_leakage_ratio = torch.mean(nn_off / (nn_diag + eps))
    zf_leakage_ratio = torch.mean(zf_off / (zf_diag + eps))

    nn_leakage_db = 10.0 * torch.log10(nn_leakage_ratio.clamp_min(eps))
    zf_leakage_db = 10.0 * torch.log10(zf_leakage_ratio.clamp_min(eps))

    # ================================
    # 6. Communication-only sum-rate
    # ================================
    p_C = torch.as_tensor(
        TRANSMIT_POWER_TOTAL * 0.2,
        dtype=torch.float32,
        device=H_eff_H.device,
    )

    noise = torch.as_tensor(
        NOISE_POWER,
        dtype=torch.float32,
        device=H_eff_H.device,
    )

    W_nn_powered = torch.sqrt(p_C) * W_C
    W_zf_powered = torch.sqrt(p_C) * W_C_zf

    Y_nn_rate = torch.matmul(H_eff_H, W_nn_powered)
    Y_zf_rate = torch.matmul(H_eff_H, W_zf_powered)

    P_nn = torch.abs(Y_nn_rate) ** 2
    P_zf = torch.abs(Y_zf_rate) ** 2

    nn_signal = torch.diagonal(P_nn, dim1=1, dim2=2)
    zf_signal = torch.diagonal(P_zf, dim1=1, dim2=2)

    nn_interf = torch.sum(P_nn, dim=2) - nn_signal
    zf_interf = torch.sum(P_zf, dim=2) - zf_signal

    nn_sinr = nn_signal / (nn_interf + noise)
    zf_sinr = zf_signal / (zf_interf + noise)

    nn_rate = torch.sum(torch.log1p(nn_sinr) / np.log(2.0), dim=1)
    zf_rate = torch.sum(torch.log1p(zf_sinr) / np.log(2.0), dim=1)

    nn_sumrate = torch.mean(nn_rate)
    zf_sumrate = torch.mean(zf_rate)

    rate_ratio = nn_sumrate / (zf_sumrate + eps)
    rate_gap = zf_sumrate - nn_sumrate

    return {
        "raw_mse": raw_mse,
        "phase_mse": phase_mse,
        "mean_cos2": mean_cos2,
        "min_cos2": min_cos2,
        "user_cos2": user_cos2,

        "effective_nmse": effective_nmse,

        "nn_leakage_db": nn_leakage_db,
        "zf_leakage_db": zf_leakage_db,

        "nn_sumrate": nn_sumrate,
        "zf_sumrate": zf_sumrate,
        "rate_ratio": rate_ratio,
        "rate_gap": rate_gap,
    }


def effective_nmse_loss(H_eff_H, W_C, W_C_zf, eps=1e-12):
    """
    Version A loss:
        NMSE between effective matrices

        Y_NN = H_eff_H @ W_C
        Y_ZF = H_eff_H @ W_C_zf

    H_eff_H : (B, K, M)
    W_C     : (B, M, K), NN output
    W_C_zf  : (B, M, K), ZF/RZF label
    """

    Y_nn = torch.matmul(H_eff_H, W_C)                  # (B,K,K)
    Y_zf = torch.matmul(H_eff_H, W_C_zf.detach())      # (B,K,K)

    numerator = torch.sum(
        torch.abs(Y_nn - Y_zf) ** 2,
        dim=(1, 2),
    )

    denominator = torch.sum(
        torch.abs(Y_zf) ** 2,
        dim=(1, 2),
    ) + eps

    loss = torch.mean(numerator / denominator)

    return loss


# ================================
# Setting
# ================================
pretrain_epoch              = 800
pretrain_batch              = 50
pretrain_channel_per_batch  = 100
pretrain_LR                 = 0.001


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
    # comm_net      = CommNet().to(DEVICE)
    test_net      = TestNet().to(DEVICE)

    # optimizer     = optim.Adam(list(comm_net.parameters()),lr=pretrain_LR)
    optimizer     = optim.Adam(list(test_net.parameters()),lr=pretrain_LR)

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    # pre_comm_ckpt = os.path.join(PRETRAIN_DIR, "pretrain_comm.ckpt")
    pre_test_ckpt = os.path.join(PRETRAIN_DIR, "pretrain_test.ckpt")

    # curve_path    = os.path.join(PRETRAIN_DIR, "pretrain_comm_curves.npz")
    curve_path    = os.path.join(PRETRAIN_DIR, "pretrain_test_curves.npz")

    curves = {
        "train_mse": [],                # 每個 epoch 的平均 MSE
        "val_mse": [],
        "train_wc_power": [],           # NN 輸出的 W_C 平均功率
        "val_wc_power": [],
        "train_rzf_power": [],          # RZF label 的 W_C 平均功率
        "val_rzf_power": [],
    }

    best_val_mse = 1e30
   
    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0] # 一共有多少train channel
    val_channels   = val_dataset["h_dk_hat"].shape[0]   # 一共有多少val channel

    print("\n" + "=" * 90)
    print("[RZF like W_C pre train]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"Pretrain_epoch          : {pretrain_epoch}")
    print(f"BATCHES                 : {pretrain_batch}")
    print(f"BATCH_CHANNELS          : {pretrain_channel_per_batch}")
    print(f"learning rate           : {pretrain_LR}")
    print("=" * 90)

    theta   = make_random_ris(1)                            # 固定theta因為非學習內容
    
    # 開始訓練
    for epoch in trange(pretrain_epoch, desc="Comm Pretraining"):
        
        # comm_net.train()
        test_net.train()

        epoch_loss = []
        epoch_wc_power = []
        epoch_rzf_power = []

        for _ in range(pretrain_batch):
            # 抽出 [pretrain_channel_per_batch] 個 channel
            channel_ids = np.random.choice(train_channels,size=pretrain_channel_per_batch,replace=False)

            h_dk_hat = torch.as_tensor(train_dataset["h_dk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)

            optimizer.zero_grad(set_to_none=True)

            H_eff_H = physics_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)
            
            # W_C = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)        
            W_C = test_net(H_eff_H)                                 # 將channel輸入得到 功率正規化 W_C
            
            W_C_rzf = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)       # 這裡輸出 功率正規化 W_C_RZF

            # loss = torch.mean(torch.abs(W_C - W_C_rzf.detach()) ** 2)
            loss = effective_nmse_loss(H_eff_H,W_C,W_C_rzf)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(list(comm_net.parameters()),max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(list(test_net.parameters()),max_norm=10.0)

            optimizer.step()

            with torch.no_grad():
                W_C_power = torch.mean(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)))
                W_C_rzf_power = torch.mean(torch.sum(torch.abs(W_C_rzf) ** 2, dim=(1, 2)))
            
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_wc_power.append(float(W_C_power.detach().cpu()))
            epoch_rzf_power.append(float(W_C_rzf_power.detach().cpu()))

        train_logs = {
            "mse": float(np.mean(epoch_loss)),
            "wc_power": float(np.mean(epoch_wc_power)),
            "rzf_power": float(np.mean(epoch_rzf_power)),
        }
        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        # comm_net.eval()
        test_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)


            H_eff_H_val =  physics_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta) 

            # W_C_val     = comm_net(h_dk_val,h_rk_val,G_val,g_dt_val)
            W_C_val     = test_net(H_eff_H_val)

            W_C_rzf_val = make_rzf_beamformer(H_eff_H_val,RZF_LAMBDA)

            # val_loss = torch.mean(torch.abs(W_C_val - W_C_rzf_val.detach()) ** 2)
            val_loss = effective_nmse_loss(H_eff_H_val,W_C_val,W_C_rzf_val)

            val_wc_power = torch.mean(torch.sum(torch.abs(W_C_val) ** 2, dim=(1, 2)))
            val_rzf_power = torch.mean(torch.sum(torch.abs(W_C_rzf_val) ** 2, dim=(1, 2)))
            
            # 新增指標
            metrics_val = compute_zf_like_metrics(H_eff_H_val,W_C_val,W_C_rzf_val)

            val_logs = {
                "mse": float(val_loss.detach().cpu()),
                "phase_mse": float(metrics_val["phase_mse"].detach().cpu()),
                "mean_cos2": float(metrics_val["mean_cos2"].detach().cpu()),
                "min_cos2": float(metrics_val["min_cos2"].detach().cpu()),
                "effective_nmse": float(metrics_val["effective_nmse"].detach().cpu()),
                "nn_leakage_db": float(metrics_val["nn_leakage_db"].detach().cpu()),
                "zf_leakage_db": float(metrics_val["zf_leakage_db"].detach().cpu()),
                "nn_sumrate": float(metrics_val["nn_sumrate"].detach().cpu()),
                "zf_sumrate": float(metrics_val["zf_sumrate"].detach().cpu()),
                "rate_ratio": float(metrics_val["rate_ratio"].detach().cpu()),
                "rate_gap": float(metrics_val["rate_gap"].detach().cpu()),
                "wc_power": float(val_wc_power.detach().cpu()),
                "rzf_power": float(val_rzf_power.detach().cpu()),
            }

        curves["train_mse"].append(train_logs["mse"])
        curves["val_mse"].append(val_logs["mse"])

        curves["train_wc_power"].append(train_logs["wc_power"])
        curves["val_wc_power"].append(val_logs["wc_power"])

        curves["train_rzf_power"].append(train_logs["rzf_power"])
        curves["val_rzf_power"].append(val_logs["rzf_power"])

        print(
            f"[Epoch {epoch + 1:04d}/{pretrain_epoch}] "
            f"TrainMSE={train_logs['mse']:.6e} "
            f"ValMSE={val_logs['mse']:.6e} "
            f"ValPhaseMSE={val_logs['phase_mse']:.6e} | "
            f"Cos2={val_logs['mean_cos2']:.4f} "
            f"MinCos2={val_logs['min_cos2']:.4f} | "
            f"LeakNN={val_logs['nn_leakage_db']:.2f}dB "
            f"LeakZF={val_logs['zf_leakage_db']:.2f}dB | "
            f"RateNN={val_logs['nn_sumrate']:.4f} "
            f"RateZF={val_logs['zf_sumrate']:.4f} "
            f"Ratio={val_logs['rate_ratio']:.4f}"
        )

        if val_logs["mse"] < best_val_mse:
            best_val_mse = val_logs["mse"]
            # comm_net.save_model(pre_comm_ckpt, verbose=False)
            test_net.save_model(pre_test_ckpt, verbose=False)


        np.savez(
            curve_path,
            train_mse=np.asarray(curves["train_mse"], dtype=np.float64),
            val_mse=np.asarray(curves["val_mse"], dtype=np.float64),
            train_wc_power=np.asarray(curves["train_wc_power"], dtype=np.float64),
            val_wc_power=np.asarray(curves["val_wc_power"], dtype=np.float64),
            train_rzf_power=np.asarray(curves["train_rzf_power"], dtype=np.float64),
            val_rzf_power=np.asarray(curves["val_rzf_power"], dtype=np.float64),
        )


    plot_one_timescale_reg_curves(curve_path)

    print("[INFO] Short-term regular training finished.")




