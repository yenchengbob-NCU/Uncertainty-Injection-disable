# -*- coding: utf-8 -*-
import os
import math
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


def plot_pretrain_curves(rob_curve_path, rob_curve_dir, ma_window=20):
    """
    畫兩張圖
    1.Train,Val 的 Loss
    2.Train,Val 的 sumrate_q05
    """
    data    = np.load(rob_curve_path)   # curve資料
    out_dir = rob_curve_dir             # 放圖片的資料夾

    # 讀資料
    train_loss    = data["train_loss"]
    val_loss      = data["val_loss"]

    train_sumrate = data["train_sumrate_q05"]
    val_sumrate   = data["val_sumrate_q05"]

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
    plt.title("Fig. 1: One-timescale ROB Loss")
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
        label="Train Sum-rate_q05",
        alpha=0.35,
        linewidth=1.0,
    )
    plt.plot(
        epochs,
        val_sumrate,
        label="Validation Sum-rate_q05",
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
    plt.title("Fig. 2: One-timescale ROB Tail Sum-rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_sumrate_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)



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
    # Train ROB
    # ================================
    comm_net  = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)

    optimizer = optim.Adam(list(comm_net.parameters())+ list(radar_net.parameters())+ list(theta_net.parameters()),lr=ROB_LEARNING_RATE)

    rob_comm_ckpt  = os.path.join(ROB_CKPT_DIR, "one_timescale_comm_rob.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR, "one_timescale_radar_rob.ckpt")
    rob_theta_ckpt = os.path.join(ROB_CKPT_DIR, "one_timescale_theta_rob.ckpt")
    rob_curve_path = os.path.join(ROB_CKPT_DIR, "one_timescale_rob_curves.npz")
    rob_curve_dir  = os.path.join(ROB_CKPT_DIR, "rob_training_curves")

    os.makedirs(rob_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],               # Loss function
        "val_loss": [],

        "train_sumrate_q05": [],        # sumerate_q05 代筆S筆注入中5%的數值
        "val_sumrate_q05": [],

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
    print("[ONE TIMESCALE ROB TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"ROB_EPOCHS              : {ROB_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"BATCH_CHANNELS          : {BATCH_CHANNELS}")
    print(f"ROB learning rate       : {ROB_LEARNING_RATE}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {ROB_SENSING_LOSS_WEIGHT}")
    print("=" * 90)

    # 開始訓練
    for epoch in trange(ROB_EPOCHS, desc="ROB Training"):
        
        comm_net.train()
        radar_net.train()
        theta_net.train()

        epoch_loss              = []    # 單一epoch的 N_BATCHE * BATCH_CHANNELS 所平均的LOSS
        epoch_sumrate_q05       = []    # 現在是robust目標 也就是尾端目標
        epoch_sensing_penalties = []
        epoch_target_snr_db     = []
        epoch_rate_user_q05     = []

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
            

            # 簡寫
            S, B, M, N, K = INJECTION_SAMPLES, h_dk_hat.shape[0], TX_ANT, RIS_UNIT, UAV_COMM

            # 使用 .unsqueeze(0) 新增第0維度 供INJ用
            # 使用 .expand       功能上"複製"通道成S份 (實際較複雜但最終是複製)
            h_dk_rep = h_dk_hat.unsqueeze(0).expand(S, B, M, K)
            h_rk_rep = h_rk_hat.unsqueeze(0).expand(S, B, N, K)
            G_rep    = G_hat.unsqueeze(0).expand(S, B, N, M)
            g_dt_rep = g_dt_hat.unsqueeze(0).expand(S, B, M, 1)

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
            
            # 計算結果
            sensing_violation = torch.relu(SENSING_SNR_THRESHOLD_DB - target_snr_mean_db)    # scalar
            sensing_penalty   = sensing_violation

            loss = -(rob_sumrate) + ROB_SENSING_LOSS_WEIGHT  * sensing_penalty

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            epoch_loss.append(float(loss.detach().cpu()))                       
            epoch_sumrate_q05.append(float(rob_sumrate.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))
            epoch_target_snr_db.append(float(target_snr_mean_db.detach().cpu()))
            epoch_rate_user_q05.append(rate_user_q05_mean.detach().cpu().numpy())

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "sumrate_q05": float(np.mean(epoch_sumrate_q05)),
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
            "target_snr_db": float(np.mean(epoch_target_snr_db)),
            # 每個 UE 的 epoch-average tail rate
            "rate_user": np.mean(np.stack(epoch_rate_user_q05, axis=0), axis=0),
        }

        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        comm_net.eval()
        radar_net.eval()
        theta_net.eval()

        with torch.no_grad():
            # ================================
            # Robust Validation: 全部 val channels，但分 chunk 避免爆 VRAM
            # ================================
            # 會做val個估測通道,但是為了避免爆 VRAM , 所以用一個個 BATCH_CHANNELS 疊加起來
            val_rate_user_q05_all = []      # (val channel個數, )
            val_sumrate_q05_all   = []      # (val channel個數, )     
            val_target_snr_all    = []      # (val channel個數, )

            for start in range(0, val_channels, BATCH_CHANNELS):    # 一個個 BATCH_CHANNELS 做INJ 最後再疊起來
                end = min(start + BATCH_CHANNELS, val_channels)

                h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"][start:end],dtype=torch.complex64,device=DEVICE)
                h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"][start:end],dtype=torch.complex64,device=DEVICE)
                G_val    = torch.as_tensor(val_dataset["G_hat"][start:end],dtype=torch.complex64,device=DEVICE)
                g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"][start:end],dtype=torch.complex64,device=DEVICE)

                theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)
                W_C_val_raw = comm_net(h_dk_val,h_rk_val,G_val,g_dt_val)
                W_R_val_raw = radar_net(h_dk_val,h_rk_val,G_val,g_dt_val)

                W_C_val, W_R_val = beamformers_power_split(W_C_val_raw,W_R_val_raw)

                # 簡寫
                S, B, M, N, K = INJECTION_SAMPLES, h_dk_val.shape[0], TX_ANT, RIS_UNIT, UAV_COMM

                # 複製 val channels 成 S 份 injection samples
                h_dk_rep = h_dk_val.unsqueeze(0).expand(S, B, M, K)
                h_rk_rep = h_rk_val.unsqueeze(0).expand(S, B, N, K)
                G_rep    = G_val.unsqueeze(0).expand(S, B, N, M)
                g_dt_rep = g_dt_val.unsqueeze(0).expand(S, B, M, 1)

                # 相對通道功率的 injection noise
                h_dk_power = torch.mean(torch.abs(h_dk_rep) ** 2, dim=(2, 3), keepdim=True).real
                h_rk_power = torch.mean(torch.abs(h_rk_rep) ** 2, dim=(2, 3), keepdim=True).real
                G_power    = torch.mean(torch.abs(G_rep)    ** 2, dim=(2, 3), keepdim=True).real
                g_dt_power = torch.mean(torch.abs(g_dt_rep) ** 2, dim=(2, 3), keepdim=True).real

                h_dk_inj = h_dk_rep + torch.sqrt(INJECTION_VARIANCE * h_dk_power) * complex_awgn(h_dk_rep.shape, 1.0, DEVICE, h_dk_rep.dtype)
                h_rk_inj = h_rk_rep + torch.sqrt(INJECTION_VARIANCE * h_rk_power) * complex_awgn(h_rk_rep.shape, 1.0, DEVICE, h_rk_rep.dtype)
                G_inj    = G_rep    + torch.sqrt(INJECTION_VARIANCE * G_power)    * complex_awgn(G_rep.shape,    1.0, DEVICE, G_rep.dtype)
                g_dt_inj = g_dt_rep + torch.sqrt(INJECTION_VARIANCE * g_dt_power) * complex_awgn(g_dt_rep.shape, 1.0, DEVICE, g_dt_rep.dtype)

                theta_rep = theta_val.unsqueeze(0).expand(S, B, N)
                W_C_rep   = W_C_val.unsqueeze(0).expand(S, B, M, K)
                W_R_rep   = W_R_val.unsqueeze(0).expand(S, B, M, 1)

                h_dk_flat = h_dk_inj.reshape(S * B, M, K)
                h_rk_flat = h_rk_inj.reshape(S * B, N, K)
                G_flat    = G_inj.reshape(S * B, N, M)
                g_dt_flat = g_dt_inj.reshape(S * B, M, 1)

                theta_flat = theta_rep.reshape(S * B, N)
                W_C_flat   = W_C_rep.reshape(S * B, M, K)
                W_R_flat   = W_R_rep.reshape(S * B, M, 1)

                H_eff_H_flat     = physics_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)

                val_metrics_flat = physics_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)

                # 還原 shape: 每個 val channel 有 S 筆 injected realizations
                val_rate_inj = val_metrics_flat["rate"].reshape(S, B, K)                # (S,B,K)

                # 對每個估測通道，每個 UE 分開取 q05
                val_rate_user_q05 = torch.quantile(val_rate_inj,OUTAGE_QUANTILE,dim=0)  # (B,K)

                # 再算出Sumrate
                val_sumrate_q05 = torch.sum(val_rate_user_q05, dim=1)                   # (B,)

                # 先還原成 (S,B)，對每個 val channel 的 S 筆 injection 做 linear 平均 target SNR 不取 q05
                val_target_snr = val_metrics_flat["target_snr"].reshape(S, B)       # (S,B), linear
                val_target_snr_mean = torch.mean(val_target_snr, dim=0)             # (B,), linear

                # 累加BATCH_CHANNELS個 資料在 大的向量上
                val_sumrate_q05_all.append(val_sumrate_q05.detach())
                val_rate_user_q05_all.append(val_rate_user_q05.detach())
                val_target_snr_all.append(val_target_snr_mean.detach())


            val_sumrate_q05_all     = torch.cat(val_sumrate_q05_all, dim=0)                     # (val_channels,) 累加完成 等效為一次做 (val channel * INJ)
            val_rate_user_q05_all   = torch.cat(val_rate_user_q05_all, dim=0)                   # (val_channels,K) 
            val_target_snr_all    = torch.cat(val_target_snr_all, dim=0)                        # (val_channels,)
            val_target_snr_mean     = torch.mean(val_target_snr_all)                            # scalar, linear
            val_target_snr_mean_db  = 10.0 * torch.log10(val_target_snr_mean.clamp_min(1e-12))

            val_sumrate   = torch.mean(val_sumrate_q05_all)                              # scalar
            val_rate_user = torch.mean(val_rate_user_q05_all, dim=0)                     # (K,)

            val_sensing_violation = torch.relu(SENSING_SNR_THRESHOLD_DB - val_target_snr_mean_db)
            val_sensing_penalty = val_sensing_violation

            val_loss = -(val_sumrate) + ROB_SENSING_LOSS_WEIGHT * val_sensing_penalty

            val_logs = {
                "loss": float(val_loss.detach().cpu()),
                "sumrate_q05": float(val_sumrate.detach().cpu()),
                "sensing_penalty": float(val_sensing_penalty.detach().cpu()),
                "target_snr_db": float(val_target_snr_mean_db.detach().cpu()),
                "rate_user": val_rate_user.detach().cpu().numpy(),
            }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_sumrate_q05"].append(train_logs["sumrate_q05"])
        curves["val_sumrate_q05"].append(val_logs["sumrate_q05"])

        curves["train_sensing_penalty"].append(train_logs["sensing_penalty"])
        curves["val_sensing_penalty"].append(val_logs["sensing_penalty"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        np.savez(
            rob_curve_path,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in curves.items()
            },
        )
        # early stopping 
        if best_val_loss - val_logs["loss"] > EARLY_STOP_EPS:
            best_val_loss    = val_logs["loss"]
            best_val_sumrate = val_logs["sumrate_q05"]
            best_val_epoch   = epoch + 1

            patience_counter = 0

            comm_net.save_model(rob_comm_ckpt, verbose=False)
            radar_net.save_model(rob_radar_ckpt, verbose=False)
            theta_net.save_model(rob_theta_ckpt, verbose=False)
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
            f"\n[Epoch {epoch + 1:03d}/{ROB_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "

            f"Trainsumrate_q05={train_logs['sumrate_q05']:.4f} "
            f"Valsumrate_q05={val_logs['sumrate_q05']:.4f} | "

            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "

            f"\nTrainrateUser_q05={fmt_vec(train_logs['rate_user'], precision=3)} "
            f"ValrateUser_q05={fmt_vec(val_logs['rate_user'], precision=3)} "
        )

    plot_pretrain_curves(rob_curve_path, rob_curve_dir)



    print("\n[INFO] One-timescale ROB training finished.")
    print(f"[INFO] Best validation epoch   = {best_val_epoch}")
    print(f"[INFO] Best validation loss    = {best_val_loss:.6f}")
    print(f"[INFO] Best validation sumrate = {best_val_sumrate:.6f}")       # 這裡要print Best validation sumrate
    print("[INFO] Short-term robust training finished.")




