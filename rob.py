# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import beamformers_power_split
from two_timescale_NN import CommNet, RadarNet, ThetaNet

# ================================
# Helpers
# ================================
def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


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


def plot_rob_curves(rob_curve_path, rob_curve_dir, ma_window=20):
    """
    畫三類圖：
    1. Train / Validation Loss
    2. Train / Validation Robust worst-UE tail rate
    3. 每個 UE 分開畫 injected-channel mean SINR power components
       Train / Val signal
       Train / Val communication interference
       Train / Val radar interference
       Train / Val noise
    """
    data = np.load(rob_curve_path)
    out_dir = rob_curve_dir

    os.makedirs(out_dir, exist_ok=True)

    # ================================
    # 讀取基本 curves
    # ================================
    train_loss = data["train_loss"]
    val_loss   = data["val_loss"]

    train_robust = data["train_robust"]
    val_robust = data["val_robust"]

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
    plt.title("Fig. 1: ROB Training Loss")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig1_path = os.path.join(out_dir, "Fig1_loss_curve.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig1_path}")

    # ================================
    # Fig. 2: Robust Worst-UE Tail Rate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs,train_robust,label="Train Robust Worst-UE Rate",alpha=0.35,linewidth=1.0)
    plt.plot(epochs,val_robust,label="Validation Robust Worst-UE Rate",alpha=0.35,linewidth=1.0)

    train_ma_epochs,train_robust_ma = moving_average(train_robust,ma_window)
    val_ma_epochs,val_robust_ma = moving_average(val_robust,ma_window)

    if len(train_robust_ma) > 0:
        plt.plot(train_ma_epochs,train_robust_ma,label=f"Train Robust MA({ma_window})",linewidth=2.5)

    if len(val_robust_ma) > 0:
        plt.plot(val_ma_epochs,val_robust_ma,label=f"Validation Robust MA({ma_window})",linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel(f"Robust Worst-UE Rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
    plt.title("Fig. 2: ROB Worst-UE Tail Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir,"Fig2_robust_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")

    # ================================
    # Fig. 3: SINR Power Components
    # 每個 UE 分開畫
    # ================================
    train_signal_power = data["train_signal_power"]              # (epochs,K)
    val_signal_power = data["val_signal_power"]                   # (epochs,K)

    train_comm_interf_power = data["train_comm_interf_power"]     # (epochs,K)
    val_comm_interf_power = data["val_comm_interf_power"]         # (epochs,K)

    train_radar_interf_power = data["train_radar_interf_power"]   # (epochs,K)
    val_radar_interf_power = data["val_radar_interf_power"]       # (epochs,K)

    train_noise_power = data["train_noise_power"]                 # (epochs,)
    val_noise_power = data["val_noise_power"]                     # (epochs,)

    num_users = train_signal_power.shape[1]

    for ue_idx in range(num_users):
        plt.figure(figsize=(12, 7))

        # Signal power
        plt.plot(
            epochs,
            train_signal_power[:, ue_idx],
            label="Train Signal",
            linewidth=1.8,
        )
        plt.plot(
            epochs,
            val_signal_power[:, ue_idx],
            label="Validation Signal",
            linewidth=1.8,
            linestyle="--",
        )

        # Communication interference power
        plt.plot(
            epochs,
            train_comm_interf_power[:, ue_idx],
            label="Train Communication Interference",
            linewidth=1.8,
        )
        plt.plot(
            epochs,
            val_comm_interf_power[:, ue_idx],
            label="Validation Communication Interference",
            linewidth=1.8,
            linestyle="--",
        )

        # Radar interference power
        plt.plot(
            epochs,
            train_radar_interf_power[:, ue_idx],
            label="Train Radar Interference",
            linewidth=1.8,
        )
        plt.plot(
            epochs,
            val_radar_interf_power[:, ue_idx],
            label="Validation Radar Interference",
            linewidth=1.8,
            linestyle="--",
        )

        # Noise power
        plt.plot(
            epochs,
            train_noise_power,
            label="Train Noise",
            linewidth=1.8,
        )
        plt.plot(
            epochs,
            val_noise_power,
            label="Validation Noise",
            linewidth=1.8,
            linestyle="--",
        )

        # 功率跨越多個數量級，用 logarithmic y-axis
        plt.yscale("log")

        plt.xlabel("Epoch")
        plt.ylabel("Power (linear scale)")
        plt.title(f"Fig. 3: Mean SINR Power Components over Injected Channels — UE {ue_idx}")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()

        fig3_path = os.path.join(
            out_dir,
            f"Fig3_UE{ue_idx}_sinr_power_components.png",
        )
        plt.savefig(fig3_path, dpi=300)
        plt.close()

        print(f"[PLOT] Saved: {fig3_path}")


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
    
    comm_net = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)

    # 讀取資料
    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")

    train_dataset = comm_net.load_channel_dataset(train_dataset_path,"train")
    val_dataset = comm_net.load_channel_dataset(val_dataset_path,"val")
    
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path   = {val_dataset_path}")

    # ================================
    # Train ROB
    # ================================
    # 僅訓練 CommNet 與 RadarNet，固定 pretrained ThetaNet
    optimizer = optim.Adam(list(comm_net.parameters())+ list(radar_net.parameters()),lr=ROB_LEARNING_RATE)

    rob_comm_ckpt  = os.path.join(ROB_CKPT_DIR, "two_timescale_comm_rob.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR, "two_timescale_radar_rob.ckpt")
    rob_curve_path = os.path.join(ROB_CKPT_DIR, "two_timescale_rob_curves.npz")
    rob_curve_dir  = os.path.join(ROB_CKPT_DIR, "two_timescale_rob_curves")

    # 已經訓練的RIS ckpt 路徑
    ris_only_ckpt  = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")

    os.makedirs(rob_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],               # Loss function 看收斂
        "val_loss": [],

        "train_robust": [],             # Robust ≜ 先找出K個UE中 worst UE rate,再對S取Q0.05,最後對B channel平均
        "val_robust": [],

        "train_target_snr_db": [],      # 感測SNR 不畫圖
        "val_target_snr_db": [],

        "train_sensing_penalty": [],    # 感測懲罰 不畫圖
        "val_sensing_penalty": [],

        "train_signal_power": [],
        "val_signal_power": [],

        "train_comm_interf_power": [],
        "val_comm_interf_power": [],

        "train_radar_interf_power": [],
        "val_radar_interf_power": [],

        "train_noise_power": [],
        "val_noise_power": [],
    }

    # 訓練參數
    best_val_loss = float("inf")        # L_best <- infinity loss 越小越好 先設無限大
    best_val_worstUE_q05 = 0.0          # best loss 對應的 Robust worst-UE tail-rate
    best_valtarget_snr_db = 0.0
    best_val_epoch = 0                  # best loss 出現在哪個 epoch

    patience_counter = 0           
    EARLY_STOP_EPS = 1e-6          
    PATIENCE = 30                  

    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0]
    val_channels   = val_dataset["h_dk_hat"].shape[0]

    print("\n" + "=" * 90)
    print("[TWO TIMESCALE ROB TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"ROB_EPOCHS              : {ROB_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"BATCH_CHANNELS          : {BATCH_CHANNELS}")
    print(f"ROB learning rate       : {ROB_LEARNING_RATE}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {ROB_SENSING_LOSS_WEIGHT}")
    print(f"Outage quantile         : {OUTAGE_QUANTILE}")
    print("=" * 90)

    # 載入已訓練好的 ThetaNet
    """
    載入並固定 pretrained ThetaNet
    僅訓練 CommNet 與 RadarNet。
    """
    if not os.path.exists(ris_only_ckpt):
        raise FileNotFoundError(f"找不到 ThetaNet checkpoint: {ris_only_ckpt}")

    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    # 不更新theta net
    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)

    # 開始訓練
    for epoch in trange(ROB_EPOCHS, desc="ROB Training"):
        
        comm_net.train()
        radar_net.train()

        # 一個batch 產生一個 loss 更新一次網路,這裡累積要在 epoch log 輸出的值,最後在印出時做平均
        epoch_loss                  = []    # 累積N個batch的loss
        epoch_Robust                = []    # 累積N個batch的Robust
        epoch_target_snr            = []
        epoch_sensing_penalties     = []
        epoch_signal_power          = []
        epoch_comm_interf_power     = []
        epoch_radar_interf_power    = []
        epoch_noise_power           = []

        # 一個epoch 有 N_BATCHE 個 batch , 一個 batch 有BATCH_CHANNELS個估測通道
        # 一個估測通道有S個 sample
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
                # 計算出等效通道
                H_eff_H = comm_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)

            # 算出W_C,W_R
            W_C_dir   = comm_net(H_eff_H)  # 正規化
            W_R_dir   = radar_net(H_eff_H) # 正規化

            W_C,W_R = beamformers_power_split(W_C_dir,W_R_dir)      # power 分配


            # ================================
            # 準備注入
            # ================================
            S, B, M, N, K = INJECTION_SAMPLES, h_dk_hat.shape[0], TX_ANT, RIS_UNIT, UAV_COMM    # 簡寫

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
            W_R_rep = W_R.unsqueeze(0).expand(S,B,M,RADAR_STREAMS)

            # 為了使用(comm_net.)副函式,先將4維壓到3維"注意! 此舉不會破壞5%的抽取"
            h_dk_flat  = h_dk_inj.reshape(S*B, M, K)
            h_rk_flat  = h_rk_inj.reshape(S*B, N, K)
            G_flat     = G_inj.reshape(S*B, N, M)
            g_dt_flat  = g_dt_inj.reshape(S*B, M, 1)

            theta_flat = theta_rep.reshape(S*B, N)
            W_C_flat   = W_C_rep.reshape(S*B, M, K)
            W_R_flat   = W_R_rep.reshape(S*B,M,RADAR_STREAMS)

            H_eff_H_flat = comm_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)                # (S*B, K, M)

            metrics_flat = comm_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)
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
            robust_noise           = metrics_flat["noise"]                              # scalar

            # Robust計算
            worstUE_rate          = torch.min(robust_allUE_rate,dim=2).values           # (S,B) 每一筆injection realization先找worst UE
            robust_per_channel    = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)  # (B,) 每個estimated channel對S取Q0.05
            robust                = torch.mean(robust_per_channel)                      # scalar 最後對B筆estimated channel平均

            # SINR組成 
            robust_signal_power          = torch.mean(robust_signal,dim=(0,1))          # (K,) 對S,B平均
            robust_comm_interf_power     = torch.mean(robust_comm_interf,dim=(0,1))     # (K,)
            robust_radar_interf_power    = torch.mean(robust_radar_interf,dim=(0,1))    # (K,)

            # Robust target SNR：對所有S筆injection與B筆estimated channel做linear mean，再轉dB
            robust_target_snr_mean = torch.mean(robust_target_snr)                    # scalar, linear
            robust_target_snr_mean_db = 10.0 * torch.log10(robust_target_snr_mean.clamp_min(1e-12))

            # 感測懲罰：每一筆injection channel使用linear SNR計算violation，再對S、B平均
            sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - robust_target_snr) # (S,B)
            sensing_penalty = torch.mean(sensing_violation)                           # scalar

            # loss function
            loss = -robust + ROB_SENSING_LOSS_WEIGHT * sensing_penalty

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            # 累加在epoch向量，供epoch log顯示
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_Robust.append(float(robust.detach().cpu()))
            epoch_target_snr.append(float(robust_target_snr_mean.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))

            epoch_signal_power.append(robust_signal_power.detach().cpu().numpy())
            epoch_comm_interf_power.append(robust_comm_interf_power.detach().cpu().numpy())
            epoch_radar_interf_power.append(robust_radar_interf_power.detach().cpu().numpy())
            epoch_noise_power.append(float(robust_noise.detach().cpu()))
        
        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "robust": float(np.mean(epoch_Robust)),
            "target_snr_db": train_target_snr_mean_db,
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),

            "signal_power": np.mean(np.stack(epoch_signal_power,axis=0),axis=0),
            "comm_interf_power": np.mean(np.stack(epoch_comm_interf_power,axis=0),axis=0),
            "radar_interf_power": np.mean(np.stack(epoch_radar_interf_power,axis=0),axis=0),
            "noise_power": float(np.mean(epoch_noise_power)),
        }

        # ================================
        # Validation: 全部 validation estimated channels，不使用 chunk
        # 每個 epoch 重新隨機產生 validation injection errors
        # ================================
        comm_net.eval()
        radar_net.eval()

        with torch.no_grad():
            # 載入 validation estimated channels
            val_h_dk = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            val_h_rk = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            val_G = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            val_g_dt = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

            # 固定 pretrained ThetaNet，根據 validation estimated channels 產生 theta
            val_theta = theta_net(val_h_dk,val_h_rk,val_G,val_g_dt)

            # 根據 estimated channels 與 val_theta 組成 estimated effective channel
            val_H_eff_H = comm_net.compute_effective_channel(val_h_dk,val_h_rk,val_G,val_theta)

            # 由本 epoch 的 CommNet 與 RadarNet 產生 unit-power beam directions
            val_W_C_dir = comm_net(val_H_eff_H)
            val_W_R_dir = radar_net(val_H_eff_H)

            # 固定分配 0.4P communication power 與 0.6P sensing power
            val_W_C,val_W_R = beamformers_power_split(val_W_C_dir,val_W_R_dir)

            val_S, val_B, val_M, val_N, val_K = INJECTION_SAMPLES, val_h_dk.shape[0], TX_ANT, RIS_UNIT, UAV_COMM

            # 將每筆 validation estimated channel 複製成 val_S 筆 uncertainty realizations
            val_h_dk_rep = val_h_dk.unsqueeze(0).expand(val_S,val_B,val_M,val_K)
            val_h_rk_rep = val_h_rk.unsqueeze(0).expand(val_S,val_B,val_N,val_K)
            val_G_rep = val_G.unsqueeze(0).expand(val_S,val_B,val_N,val_M)
            val_g_dt_rep = val_g_dt.unsqueeze(0).expand(val_S,val_B,val_M,1)

            # 每筆 estimated channel block 的 empirical mean power
            val_h_dk_power = torch.mean(torch.abs(val_h_dk_rep) ** 2,dim=(2,3),keepdim=True).real
            val_h_rk_power = torch.mean(torch.abs(val_h_rk_rep) ** 2,dim=(2,3),keepdim=True).real
            val_G_power = torch.mean(torch.abs(val_G_rep) ** 2,dim=(2,3),keepdim=True).real
            val_g_dt_power = torch.mean(torch.abs(val_g_dt_rep) ** 2,dim=(2,3),keepdim=True).real

            # 每個 epoch 重新隨機產生 validation injection errors
            val_h_dk_inj = val_h_dk_rep + torch.sqrt(INJECTION_VARIANCE * val_h_dk_power) * complex_awgn(val_h_dk_rep.shape,1.0,DEVICE,val_h_dk_rep.dtype)
            val_h_rk_inj = val_h_rk_rep + torch.sqrt(INJECTION_VARIANCE * val_h_rk_power) * complex_awgn(val_h_rk_rep.shape,1.0,DEVICE,val_h_rk_rep.dtype)
            val_G_inj = val_G_rep + torch.sqrt(INJECTION_VARIANCE * val_G_power) * complex_awgn(val_G_rep.shape,1.0,DEVICE,val_G_rep.dtype)
            val_g_dt_inj = val_g_dt_rep + torch.sqrt(INJECTION_VARIANCE * val_g_dt_power) * complex_awgn(val_g_dt_rep.shape,1.0,DEVICE,val_g_dt_rep.dtype)

            # 同一筆 estimated channel 的 val_S 筆 injected channels
            # 共用同一組 theta、W_C 與 W_R
            val_theta_rep = val_theta.unsqueeze(0).expand(val_S,val_B,val_N)
            val_W_C_rep = val_W_C.unsqueeze(0).expand(val_S,val_B,val_M,val_K)
            val_W_R_rep = val_W_R.unsqueeze(0).expand(val_S,val_B,val_M,RADAR_STREAMS)

            # 將 validation 的 S 與 B 維度壓平，供共用 physics functions 使用
            val_h_dk_flat = val_h_dk_inj.reshape(val_S*val_B,val_M,val_K)
            val_h_rk_flat = val_h_rk_inj.reshape(val_S*val_B,val_N,val_K)
            val_G_flat = val_G_inj.reshape(val_S*val_B,val_N,val_M)
            val_g_dt_flat = val_g_dt_inj.reshape(val_S*val_B,val_M,1)

            # 使用 injected true channels 重新組成 effective channel
            # theta、W_C、W_R 仍由 estimated channels 決定並保持固定
            val_theta_flat = val_theta_rep.reshape(val_S*val_B,val_N)
            val_W_C_flat = val_W_C_rep.reshape(val_S*val_B,val_M,val_K)
            val_W_R_flat = val_W_R_rep.reshape(val_S*val_B,val_M,RADAR_STREAMS)

            val_H_eff_H_flat = comm_net.compute_effective_channel(val_h_dk_flat,val_h_rk_flat,val_G_flat,val_theta_flat)

            val_metrics_flat = comm_net.compute_isac_batch_performance(val_H_eff_H_flat,val_g_dt_flat,val_W_C_flat,val_W_R_flat)
            """ 輸出結果 :
                    SINR power components:
                        signal       : (S*B, K)
                        comm_interf  : (S*B, K)
                        radar_interf : (S*B, K)
                        noise        : scalar

                    raw per-channel-sample tensors:
                        sinr          : (S*B, K), linear
                        sinr_db       : (S*B, K), dB
                        rate          : (S*B, K)
                        sumrate       : (S*B,)
                        target_snr    : (S*B,), linear
                        target_snr_db : (S*B,), dB

                    S*B-average display values: 對所有injection與channel平均
                        sinr_user_mean      : (K,), linear
                        sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                        rate_user_mean      : (K,)
                        sumrate_mean        : scalar
                        target_snr_mean     : scalar, linear
                        target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
            """

            # 還原shape並取輸出結果
            val_robust_allUE_rate = val_metrics_flat["rate"].reshape(val_S,val_B,val_K)             # (S,B,K) 各UE在每筆injection channel的rate
            val_robust_target_snr = val_metrics_flat["target_snr"].reshape(val_S,val_B)             # (S,B) 每一筆injection channel的target SNR，linear

            val_robust_signal = val_metrics_flat["signal"].reshape(val_S,val_B,val_K)               # (S,B,K) 各UE的signal power
            val_robust_comm_interf = val_metrics_flat["comm_interf"].reshape(val_S,val_B,val_K)     # (S,B,K) 各UE的communication interference
            val_robust_radar_interf = val_metrics_flat["radar_interf"].reshape(val_S,val_B,val_K)   # (S,B,K) 各UE的radar interference
            val_robust_noise        = val_metrics_flat["noise"]                                     # scalar

            # Robust計算
            val_worstUE_rate = torch.min(val_robust_allUE_rate,dim=2).values                    # (S,B) 每一筆injection realization先找worst UE
            val_robust_per_channel = torch.quantile(val_worstUE_rate,OUTAGE_QUANTILE,dim=0)     # (B,) 每個estimated channel對S取Q0.05
            val_robust = torch.mean(val_robust_per_channel)                                     # scalar 最後對B筆estimated channel平均

            # SINR組成
            val_robust_signal_power = torch.mean(val_robust_signal,dim=(0,1))               # (K,) 對S、B平均
            val_robust_comm_interf_power = torch.mean(val_robust_comm_interf,dim=(0,1))     # (K,)
            val_robust_radar_interf_power = torch.mean(val_robust_radar_interf,dim=(0,1))   # (K,)

            # Robust target SNR：對所有S筆injection與B筆estimated channel做linear mean，再轉dB
            val_robust_target_snr_mean = torch.mean(val_robust_target_snr)                  # scalar, linear
            val_robust_target_snr_mean_db = 10.0 * torch.log10(val_robust_target_snr_mean.clamp_min(1e-12))

            # 感測懲罰：每一筆injection channel使用linear SNR計算violation，再對S、B平均
            val_sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - val_robust_target_snr)   # (S,B)
            val_sensing_penalty = torch.mean(val_sensing_violation)                             # scalar

            # loss function
            val_loss = -val_robust + ROB_SENSING_LOSS_WEIGHT * val_sensing_penalty

        val_logs = {
            "loss": float(val_loss.detach().cpu()),
            "robust": float(val_robust.detach().cpu()),
            "target_snr_db": float(val_robust_target_snr_mean_db.detach().cpu()),
            "sensing_penalty": float(val_sensing_penalty.detach().cpu()),

            "signal_power": val_robust_signal_power.detach().cpu().numpy(),
            "comm_interf_power": val_robust_comm_interf_power.detach().cpu().numpy(),
            "radar_interf_power": val_robust_radar_interf_power.detach().cpu().numpy(),
            "noise_power": float(val_robust_noise.detach().cpu()),
        }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_robust"].append(train_logs["robust"])
        curves["val_robust"].append(val_logs["robust"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        curves["train_sensing_penalty"].append(train_logs["sensing_penalty"])
        curves["val_sensing_penalty"].append(val_logs["sensing_penalty"])

        curves["train_signal_power"].append(train_logs["signal_power"].copy())
        curves["val_signal_power"].append(val_logs["signal_power"].copy())

        curves["train_comm_interf_power"].append(train_logs["comm_interf_power"].copy())
        curves["val_comm_interf_power"].append(val_logs["comm_interf_power"].copy())

        curves["train_radar_interf_power"].append(train_logs["radar_interf_power"].copy())
        curves["val_radar_interf_power"].append(val_logs["radar_interf_power"].copy())

        curves["train_noise_power"].append(train_logs["noise_power"])
        curves["val_noise_power"].append(val_logs["noise_power"])

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
            best_val_worstUE_q05 = val_logs["robust"]
            best_valtarget_snr_db = val_logs["target_snr_db"]
            best_val_epoch   = epoch + 1

            patience_counter = 0

            comm_net.save_model(rob_comm_ckpt, verbose=False)
            radar_net.save_model(rob_radar_ckpt, verbose=False)

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

        # 印出每個epoch log
        print(
            f"\n[Epoch {epoch + 1:03d}/{ROB_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "
            f"TrainRobust={train_logs['robust']:.4f} "
            f"ValRobust={val_logs['robust']:.4f} | "
            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB"
            f"\nTrainSignal={fmt_vec_sci(train_logs['signal_power'],precision=3)} "
            f"ValSignal={fmt_vec_sci(val_logs['signal_power'],precision=3)}"
            f"\nTrainCommInterf={fmt_vec_sci(train_logs['comm_interf_power'],precision=3)} "
            f"ValCommInterf={fmt_vec_sci(val_logs['comm_interf_power'],precision=3)}"
            f"\nTrainRadarInterf={fmt_vec_sci(train_logs['radar_interf_power'],precision=3)} "
            f"ValRadarInterf={fmt_vec_sci(val_logs['radar_interf_power'],precision=3)}"
            f"\nTrainNoise={train_logs['noise_power']:.3e} "
            f"ValNoise={val_logs['noise_power']:.3e}"
        )

    plot_rob_curves(rob_curve_path,rob_curve_dir)


    print("\n[INFO] ROB training finished.")
    print(f"[INFO] Best validation epoch          = {best_val_epoch}")
    print(f"[INFO] Best validation loss           = {best_val_loss:.6f}")
    print(f"[INFO] Robust rate at best val loss   = {best_val_worstUE_q05:.6f}")       # 這裡要print Best validation robust
    print(f"[INFO] Target SNR at best val loss    = {best_valtarget_snr_db:.3f} dB")
    print("[INFO] Short-term robust training finished.")
