# -*- coding: utf-8 -*-
import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
# from baseline import beamformers_power_split
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

    parser = argparse.ArgumentParser(description="Train two-timescale ROB networks.")
    parser.add_argument("--pretrain",action="store_true",help="Load the pretrained NS-MRT RadarNet checkpoint.")
    args = parser.parse_args()
    
    comm_net = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)

    # 讀取資料
    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")

    with np.load(train_dataset_path) as data:
        train_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

    with np.load(val_dataset_path) as data:
        val_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}
    
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path   = {val_dataset_path}")

    # ================================
    # Train ROB
    # ================================
    rob_comm_ckpt  = os.path.join(ROB_CKPT_DIR, "two_timescale_comm_rob.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR, "two_timescale_radar_rob.ckpt")
    rob_curve_path = os.path.join(ROB_CKPT_DIR, "two_timescale_rob_curves.npz")
    rob_curve_dir  = os.path.join(ROB_CKPT_DIR, "two_timescale_rob_curves")

    # 已經訓練的RIS ckpt 路徑
    ris_only_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")
    radar_pretrain_ckpt = os.path.join(PRETRAIN_DIR,"radar_ns_mrt_pretrain.ckpt")

    os.makedirs(rob_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],               # Loss function 看收斂
        "val_loss": [],

        "train_robust": [],             # Robust ≜ 每slot先找K個UE中 worst UE rate,再對S取Q0.05,最後對L slots與B blocks平均
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

        # Beamformer transmit power
        "train_comm_tx_power": [],
        "val_comm_tx_power": [],

        "train_radar_tx_power": [],
        "val_radar_tx_power": [],

        "train_total_tx_power": [],
        "val_total_tx_power": [],

        "train_tx_power_penalty": [],
        "val_tx_power_penalty": [],

        "train_tx_power_violation_rate": [],
        "val_tx_power_violation_rate": [],
    }

    # 訓練參數
    best_val_loss = float("inf")        # L_best <- infinity loss 越小越好 先設無限大
    best_val_worstUE_q05 = 0.0          # best loss 對應的 Robust worst-UE tail-rate
    best_valtarget_snr_db = 0.0
    best_val_epoch = 0                  # best loss 出現在哪個 epoch

    patience_counter    = 0           
    EARLY_STOP_EPS      = 1e-6          
    PATIENCE            = 30
    VAL_CHUNK_SIZE      = 250           # validation 每次處理的long-term blocks數量，避免一次建立全部injection tensors                 

    # 載入資料
    train_trajectories, train_channels = train_dataset["h_dk"].shape[:2]
    val_trajectories, val_channels     = val_dataset["h_dk"].shape[:2]

    val_channel_ids = np.arange(0,val_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
    val_trajectory_ids = np.repeat(np.arange(val_trajectories),len(val_channel_ids))
    val_channel_ids = np.tile(val_channel_ids,val_trajectories)
    B_val = len(val_channel_ids)

    print("\n" + "=" * 90)
    print("[TWO TIMESCALE ROB TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"ROB_EPOCHS              : {ROB_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"ST_BATCH_SIZE          : {ST_BATCH_SIZE}")
    print(f"ROB learning rate       : {ROB_LEARNING_RATE}")
    print(f"RadarNet initialization : {'NS-MRT pretrained' if args.pretrain else 'Random'}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {ROB_SENSING_LOSS_WEIGHT}")
    print(f"TX power penalty weight : {ROB_TX_POWER_LOSS_WEIGHT}")
    print(f"Outage quantile         : {OUTAGE_QUANTILE}")
    print("=" * 90)

    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    # 不更新ThetaNet
    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)


    # ================================
    # RadarNet initialization
    # ================================
    if args.pretrain:
        if not os.path.exists(radar_pretrain_ckpt):
            raise FileNotFoundError(f"找不到 RadarNet pretrain checkpoint: {radar_pretrain_ckpt}")

        radar_net.load_model(radar_pretrain_ckpt,strict=True,verbose=True)
        radar_init_mode = "NS-MRT pretrained"

        print(f"[INFO] Loaded RadarNet pretrain checkpoint: {radar_pretrain_ckpt}")

    else:
        # 不載入checkpoint，保留RadarNet建立時的隨機初始化權重
        radar_init_mode = "Random initialization"

        print("[INFO] RadarNet uses random initialization.")


    # 不論是否使用預訓練，正式ROB訓練都會更新RadarNet
    for parameter in radar_net.parameters():
        parameter.requires_grad_(True)


    # CommNet固定從隨機初始化開始
    # RadarNet由--pretrain決定使用預訓練或隨機初始化
    optimizer = optim.Adam(list(comm_net.parameters()) + list(radar_net.parameters()),lr=ROB_LEARNING_RATE)

    print("[INFO] ThetaNet is frozen.")
    print(f"[INFO] RadarNet initialization mode: {radar_init_mode}")
    print("[INFO] CommNet and RadarNet remain trainable during ROB joint training.")

    # 開始訓練
    for epoch in trange(ROB_EPOCHS,desc="ROB Training"):
        
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

        epoch_sensing_violation_rates = []
        epoch_sensing_channel_failure_rates = []

        epoch_comm_tx_power = []
        epoch_radar_tx_power = []
        epoch_total_tx_power = []
        epoch_tx_power_penalties = []
        epoch_tx_power_violation_rates = []

        # 一個epoch 有 N_BATCHE 個 batch , 一個 batch 有BATCH_CHANNELS個 long-term blocks
        # 一個 short-term slot 有S個 injection samples
        for _ in range(N_BATCHE):
            
            # 從data中抽出 ST_BATCH_SIZE 個 long-term blocks
            trajectory_ids = np.random.choice(train_trajectories,size=ST_BATCH_SIZE,replace=True)
            channel_ids = np.random.choice(train_channels,size=ST_BATCH_SIZE,replace=False)

            valid = channel_ids + LT_BLOCK_SIZE <= train_channels
            trajectory_ids = trajectory_ids[valid]
            channel_ids = channel_ids[valid]
            B = len(channel_ids)

            # ThetaNet只看每個long-term block的start-slot CSI
            h_dk_hat = torch.as_tensor(train_dataset["h_dk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)  # (B,M,K)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)  # (B,N,K)
            G_hat    = torch.as_tensor(train_dataset["G"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)     # (B,N,M)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)  # (B,M,1)

            block_ids = channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]  # (B,L)

            h_dk_block = torch.as_tensor(train_dataset["h_dk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)  # (B,L,M,K)
            h_rk_block = torch.as_tensor(train_dataset["h_rk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)  # (B,L,N,K)
            G_block    = torch.as_tensor(train_dataset["G"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)     # (B,L,N,M)
            g_dt_block = torch.as_tensor(train_dataset["g_dt"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)  # (B,L,M,1)

            h_dk_block = h_dk_block.reshape(B*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            h_rk_block = h_rk_block.reshape(B*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            G_block    = G_block.reshape(B*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            g_dt_block = g_dt_block.reshape(B*LT_BLOCK_SIZE,TX_ANT,1)

            optimizer.zero_grad(set_to_none=True)   # 清除梯度，避免上一個 batch 的梯度殘留

            # 使用已經訓練的RIS net 拿到theta
            with torch.no_grad():
                theta = theta_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)               # (B,N)
                theta_block = theta[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B*LT_BLOCK_SIZE,RIS_UNIT)
                # 每個slot使用自己的channel計算等效通道，固定的只有theta
                H_eff_H = comm_net.compute_effective_channel(h_dk_block,h_rk_block,G_block,theta_block)

            # NN直接輸出beamformer，同時學習方向與power allocation
            W_C_raw = comm_net(H_eff_H,g_dt_block)
            W_R_raw = radar_net(H_eff_H,g_dt_block)

            # 每個short-term slot對應的communication與sensing power
            W_C_power = torch.sum(torch.abs(W_C_raw) ** 2,dim=(1,2)).real       # (B*L,)
            W_R_power = torch.sum(torch.abs(W_R_raw) ** 2,dim=(1,2)).real       # (B*L,)
            total_tx_power = W_C_power + W_R_power                              # (B*L,)

            # 每個short-term slot都必須滿足 P_C + P_R <= TRANSMIT_POWER_TOTAL
            total_tx_power_ratio = total_tx_power / TRANSMIT_POWER_TOTAL
            tx_power_excess_ratio = torch.relu(total_tx_power_ratio - 1.0)
            tx_power_penalty = torch.mean(tx_power_excess_ratio ** 2                )# scalar

            # 只供log顯示：有多少short-term slots超過總功率
            tx_power_violation_rate = torch.mean((total_tx_power > TRANSMIT_POWER_TOTAL).float())

            # ================================
            # 準備注入
            # ================================
            S, B_short, M, N, K = INJECTION_SAMPLES, h_dk_block.shape[0], TX_ANT, RIS_UNIT, UAV_COMM    # 簡寫，B_short=B*L

            # 使用 .unsqueeze(0) 新增第0維度 供INJ用
            # 使用 .expand       功能上"複製"每個short-term slot成S份 (實際較複雜但最終是複製)
            h_dk_rep = h_dk_block.unsqueeze(0).expand(S,B_short,M,K)
            h_rk_rep = h_rk_block.unsqueeze(0).expand(S,B_short,N,K)
            G_rep    = G_block.unsqueeze(0).expand(S,B_short,N,M)
            g_dt_rep = g_dt_block.unsqueeze(0).expand(S,B_short,M,1)

            # 注入不確定：INJECTION_VARIANCE 是相對通道功率
            # noise power = INJECTION_VARIANCE * mean(|channel|^2)
            h_dk_power = torch.mean(torch.abs(h_dk_rep) ** 2,dim=(2,3),keepdim=True).real   # (S,B*L,1,1)
            h_rk_power = torch.mean(torch.abs(h_rk_rep) ** 2,dim=(2,3),keepdim=True).real   # (S,B*L,1,1)
            G_power    = torch.mean(torch.abs(G_rep)    ** 2,dim=(2,3),keepdim=True).real   # (S,B*L,1,1)
            g_dt_power = torch.mean(torch.abs(g_dt_rep) ** 2,dim=(2,3),keepdim=True).real   # (S,B*L,1,1)

            # 注入不確定
            h_dk_inj = h_dk_rep + torch.sqrt(INJECTION_VARIANCE * h_dk_power) * complex_awgn(h_dk_rep.shape,1.0,DEVICE,h_dk_rep.dtype)
            h_rk_inj = h_rk_rep + torch.sqrt(INJECTION_VARIANCE * h_rk_power) * complex_awgn(h_rk_rep.shape,1.0,DEVICE,h_rk_rep.dtype)
            G_inj    = G_rep    + torch.sqrt(INJECTION_VARIANCE * G_power)    * complex_awgn(G_rep.shape,1.0,DEVICE,G_rep.dtype)
            g_dt_inj = g_dt_rep + torch.sqrt(INJECTION_VARIANCE * g_dt_power) * complex_awgn(g_dt_rep.shape,1.0,DEVICE,g_dt_rep.dtype)

            # 同一個long-term block的L個slots共用theta；每個slot的S組injection也共用該theta
            theta_rep = theta_block.unsqueeze(0).expand(S,B_short,N)

            # 每個short-term slot的S組injection共用該slot由nominal channel產生的beamformer
            W_C_rep = W_C_raw.unsqueeze(0).expand(S,B_short,M,K)
            W_R_rep = W_R_raw.unsqueeze(0).expand(S,B_short,M,RADAR_STREAMS)

            # 為了使用(comm_net.)副函式,先將S與B*L維度壓平，S維稍後會還原取quantile
            h_dk_flat = h_dk_inj.reshape(S*B_short,M,K)
            h_rk_flat = h_rk_inj.reshape(S*B_short,N,K)
            G_flat    = G_inj.reshape(S*B_short,N,M)
            g_dt_flat = g_dt_inj.reshape(S*B_short,M,1)

            theta_flat = theta_rep.reshape(S*B_short,N)
            W_C_flat   = W_C_rep.reshape(S*B_short,M,K)
            W_R_flat   = W_R_rep.reshape(S*B_short,M,RADAR_STREAMS)

            H_eff_H_flat = comm_net.compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,theta_flat)  # (S*B*L,K,M)

            metrics_flat = comm_net.compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_C_flat,W_R_flat)
            '''Output:

                raw per-channel-sample tensors:
                    sinr          : (S*B*L,K), linear
                    sinr_db       : (S*B*L,K), dB
                    rate          : (S*B*L,K)
                    sumrate       : (S*B*L,)
                    target_snr    : (S*B*L,), linear
                    target_snr_db : (S*B*L,), dB

                S*B*L-average display values: 對所有injection * short-term slot平均
                    注意不可以用這裡! 這混淆了取5%
                    sinr_user_mean      : (K,), linear
                    sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                    rate_user_mean      : (K,)
                    sumrate_mean        : scalar
                    target_snr_mean     : scalar, linear
                    target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
            '''
            
            # 還原shape並取輸出結果
            robust_allUE_rate    = metrics_flat["rate"].reshape(S,B,LT_BLOCK_SIZE,K)              # (S,B,L,K)
            robust_target_snr    = metrics_flat["target_snr"].reshape(S,B,LT_BLOCK_SIZE)           # (S,B,L), linear
            robust_target_snr_db = metrics_flat["target_snr_db"].reshape(S,B,LT_BLOCK_SIZE)        # (S,B,L), dB

            robust_signal       = metrics_flat["signal"].reshape(S,B,LT_BLOCK_SIZE,K)              # (S,B,L,K)
            robust_comm_interf  = metrics_flat["comm_interf"].reshape(S,B,LT_BLOCK_SIZE,K)         # (S,B,L,K)
            robust_radar_interf = metrics_flat["radar_interf"].reshape(S,B,LT_BLOCK_SIZE,K)        # (S,B,L,K)
            robust_noise        = metrics_flat["noise"]                                             # scalar

            # Robust計算：每個slot先對K取worst UE，再對S取quantile，最後對L與B平均
            worstUE_rate       = torch.min(robust_allUE_rate,dim=3).values                          # (S,B,L)
            robust_per_slot    = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)                 # (B,L)
            robust_per_channel = torch.mean(robust_per_slot,dim=1)                                  # (B,) 保留原變數名稱，現在代表每個long-term block
            robust             = torch.mean(robust_per_channel)                                     # scalar

            # SINR組成 
            robust_signal_power       = torch.mean(robust_signal,dim=(0,1,2))       # (K,) 對S,B,L平均
            robust_comm_interf_power  = torch.mean(robust_comm_interf,dim=(0,1,2))  # (K,)
            robust_radar_interf_power = torch.mean(robust_radar_interf,dim=(0,1,2)) # (K,)

            # Robust target SNR：對所有S筆injection、B個blocks與L個slots做linear mean，再轉dB
            robust_target_snr_mean = torch.mean(robust_target_snr)
            robust_target_snr_mean_db = 10.0 * torch.log10(robust_target_snr_mean.clamp_min(1e-12))

            # 每個short-term slot各自對S筆injection取target SNR Q10%
            target_snr_q10_db = torch.quantile(robust_target_snr_db,TAR_OUTAGE_QUANTILE,dim=0)  # (B,L)

            # Q10% 與 4 dB 門檻比較，供loss反向傳播
            sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - target_snr_q10_db)       # (B,L)
            sensing_penalty = torch.mean(sensing_deficit_db)                                    # scalar

            # 每個short-term slot在S筆injection下的實際違反率
            sensing_violation_per_channel = torch.mean(
                (robust_target_snr_db < SENSING_SNR_THRESHOLD_DB).float(),dim=0
            )                                                                                   # (B,L)

            # 全部short-term slots的平均違反率
            sensing_violation_rate = torch.mean(sensing_violation_per_channel)                   # scalar

            # 有多少比例的short-term slots未滿足「違反率 <= 10%」
            sensing_channel_failure_rate = torch.mean((sensing_violation_per_channel > TAR_OUTAGE_QUANTILE).float())    # scalar

            # loss function
            loss = -robust + ROB_SENSING_LOSS_WEIGHT * sensing_penalty + ROB_TX_POWER_LOSS_WEIGHT * tx_power_penalty

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            # 累加在epoch向量，供epoch log顯示
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_Robust.append(float(robust.detach().cpu()))
            epoch_target_snr.append(float(robust_target_snr_mean.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))

            epoch_sensing_violation_rates.append(float(sensing_violation_rate.detach().cpu()))
            epoch_sensing_channel_failure_rates.append(float(sensing_channel_failure_rate.detach().cpu()))
            
            epoch_signal_power.append(robust_signal_power.detach().cpu().numpy())
            epoch_comm_interf_power.append(robust_comm_interf_power.detach().cpu().numpy())
            epoch_radar_interf_power.append(robust_radar_interf_power.detach().cpu().numpy())
            epoch_noise_power.append(float(robust_noise.detach().cpu()))
        
            epoch_comm_tx_power.append(float(torch.mean(W_C_power).detach().cpu()))
            epoch_radar_tx_power.append(float(torch.mean(W_R_power).detach().cpu()))
            epoch_total_tx_power.append(float(torch.mean(total_tx_power).detach().cpu()))
            epoch_tx_power_penalties.append(float(tx_power_penalty.detach().cpu()))
            epoch_tx_power_violation_rates.append(float(tx_power_violation_rate.detach().cpu()))

        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "robust": float(np.mean(epoch_Robust)),
            "target_snr_db": train_target_snr_mean_db,
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
            "sensing_violation_rate": float(np.mean(epoch_sensing_violation_rates)),
            "sensing_channel_failure_rate": float(np.mean(epoch_sensing_channel_failure_rates)),

            "signal_power": np.mean(np.stack(epoch_signal_power,axis=0),axis=0),
            "comm_interf_power": np.mean(np.stack(epoch_comm_interf_power,axis=0),axis=0),
            "radar_interf_power": np.mean(np.stack(epoch_radar_interf_power,axis=0),axis=0),
            "noise_power": float(np.mean(epoch_noise_power)),

            "comm_tx_power": float(np.mean(epoch_comm_tx_power)),
            "radar_tx_power": float(np.mean(epoch_radar_tx_power)),
            "total_tx_power": float(np.mean(epoch_total_tx_power)),
            "tx_power_penalty": float(np.mean(epoch_tx_power_penalties)),
            "tx_power_violation_rate": float(np.mean(epoch_tx_power_violation_rates)),
        }

        # ================================
        # Validation: 固定 0,L,2L,... block starts 的全部 val trajectories，injection使用chunk避免GPU OOM
        # 每個 epoch 重新隨機產生 validation injection errors
        # ================================
        comm_net.eval()
        radar_net.eval()

        with torch.no_grad():
            # ThetaNet只看每個validation long-term block的start-slot CSI
            val_h_dk = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)  # (B_val,M,K)
            val_h_rk = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)  # (B_val,N,K)
            val_G    = torch.as_tensor(val_dataset["G"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)     # (B_val,N,M)
            val_g_dt = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)  # (B_val,M,1)

            # 固定 pretrained ThetaNet，只根據block-start CSI產生theta
            val_theta = theta_net(val_h_dk,val_h_rk,val_G,val_g_dt)  # (B_val,N)

            val_block_ids = val_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]  # (B_val,L)

            val_h_dk_block = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            val_h_rk_block = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            val_G_block    = torch.as_tensor(val_dataset["G"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            val_g_dt_block = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)

            val_h_dk_block = val_h_dk_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            val_h_rk_block = val_h_rk_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            val_G_block    = val_G_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            val_g_dt_block = val_g_dt_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,1)
            val_theta_block = val_theta[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT)

            # 每個slot使用自己的nominal channel與固定theta產生effective channel與beamformer
            val_H_eff_H = comm_net.compute_effective_channel(val_h_dk_block,val_h_rk_block,val_G_block,val_theta_block)
            val_W_C_raw = comm_net(val_H_eff_H,val_g_dt_block)
            val_W_R_raw = radar_net(val_H_eff_H,val_g_dt_block)

            # 每個validation short-term slot的beamformer power
            val_W_C_power = torch.sum(torch.abs(val_W_C_raw) ** 2,dim=(1,2)).real
            val_W_R_power = torch.sum(torch.abs(val_W_R_raw) ** 2,dim=(1,2)).real
            val_total_tx_power = val_W_C_power + val_W_R_power

            val_total_tx_power_ratio = val_total_tx_power / TRANSMIT_POWER_TOTAL
            val_tx_power_excess_ratio = torch.relu(val_total_tx_power_ratio - 1.0)
            val_tx_power_penalty = torch.mean(val_tx_power_excess_ratio ** 2)# scalar

            val_tx_power_violation_rate = torch.mean((val_total_tx_power > TRANSMIT_POWER_TOTAL).float())

            val_S, val_M, val_N, val_K = INJECTION_SAMPLES, TX_ANT, RIS_UNIT, UAV_COMM

            # ================================
            # Validation injection chunk
            # ================================
            # 只切B_val這個long-term block維度，每個block仍完整保留L個slots與val_S筆injection
            # 因此robust quantile、sensing quantile與所有validation metric的數學定義完全不變
            val_robust_sum = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_robust_signal_sum = torch.zeros(val_K,dtype=torch.float32,device=DEVICE)
            val_robust_comm_interf_sum = torch.zeros(val_K,dtype=torch.float32,device=DEVICE)
            val_robust_radar_interf_sum = torch.zeros(val_K,dtype=torch.float32,device=DEVICE)
            val_robust_target_snr_sum = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_sensing_penalty_sum = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_sensing_violation_sum = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_sensing_channel_failure_sum = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_robust_noise = None

            for val_chunk_start in range(0,B_val,VAL_CHUNK_SIZE):
                val_chunk_end = min(val_chunk_start + VAL_CHUNK_SIZE,B_val)
                val_B_chunk = val_chunk_end - val_chunk_start

                # val_*_block原本shape第一維是B_val*L，因此依long-term block換算short-term slot範圍
                val_short_start = val_chunk_start * LT_BLOCK_SIZE
                val_short_end = val_chunk_end * LT_BLOCK_SIZE
                val_B_short_chunk = val_B_chunk * LT_BLOCK_SIZE

                val_h_dk_block_chunk = val_h_dk_block[val_short_start:val_short_end]
                val_h_rk_block_chunk = val_h_rk_block[val_short_start:val_short_end]
                val_G_block_chunk = val_G_block[val_short_start:val_short_end]
                val_g_dt_block_chunk = val_g_dt_block[val_short_start:val_short_end]

                val_theta_block_chunk = val_theta_block[val_short_start:val_short_end]
                val_W_C_raw_chunk = val_W_C_raw[val_short_start:val_short_end]
                val_W_R_raw_chunk = val_W_R_raw[val_short_start:val_short_end]

                # 將每個 validation short-term slot 複製成 val_S 筆 uncertainty realizations
                val_h_dk_rep = val_h_dk_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,val_K)
                val_h_rk_rep = val_h_rk_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_N,val_K)
                val_G_rep    = val_G_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_N,val_M)
                val_g_dt_rep = val_g_dt_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,1)

                # 每個short-term slot的 empirical mean channel power
                val_h_dk_power = torch.mean(torch.abs(val_h_dk_rep) ** 2,dim=(2,3),keepdim=True).real
                val_h_rk_power = torch.mean(torch.abs(val_h_rk_rep) ** 2,dim=(2,3),keepdim=True).real
                val_G_power = torch.mean(torch.abs(val_G_rep) ** 2,dim=(2,3),keepdim=True).real
                val_g_dt_power = torch.mean(torch.abs(val_g_dt_rep) ** 2,dim=(2,3),keepdim=True).real

                # 每個 epoch 重新隨機產生 validation injection errors
                val_h_dk_inj = val_h_dk_rep + torch.sqrt(INJECTION_VARIANCE * val_h_dk_power) * complex_awgn(val_h_dk_rep.shape,1.0,DEVICE,val_h_dk_rep.dtype)
                val_h_rk_inj = val_h_rk_rep + torch.sqrt(INJECTION_VARIANCE * val_h_rk_power) * complex_awgn(val_h_rk_rep.shape,1.0,DEVICE,val_h_rk_rep.dtype)
                val_G_inj = val_G_rep + torch.sqrt(INJECTION_VARIANCE * val_G_power) * complex_awgn(val_G_rep.shape,1.0,DEVICE,val_G_rep.dtype)
                val_g_dt_inj = val_g_dt_rep + torch.sqrt(INJECTION_VARIANCE * val_g_dt_power) * complex_awgn(val_g_dt_rep.shape,1.0,DEVICE,val_g_dt_rep.dtype)

                # 同一個block的L slots共用theta；每個slot的val_S筆injection共用該slot beamformer
                val_theta_rep = val_theta_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_N)
                val_W_C_rep = val_W_C_raw_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,val_K)
                val_W_R_rep = val_W_R_raw_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,RADAR_STREAMS)

                # 將 validation 的 S 與 B_chunk*L 維度壓平，供共用 physics functions 使用
                val_h_dk_flat = val_h_dk_inj.reshape(val_S*val_B_short_chunk,val_M,val_K)
                val_h_rk_flat = val_h_rk_inj.reshape(val_S*val_B_short_chunk,val_N,val_K)
                val_G_flat = val_G_inj.reshape(val_S*val_B_short_chunk,val_N,val_M)
                val_g_dt_flat = val_g_dt_inj.reshape(val_S*val_B_short_chunk,val_M,1)

                # 使用 injected true channels 重新組成 effective channel
                # theta、W_C、W_R 仍由nominal channels決定並保持固定
                val_theta_flat = val_theta_rep.reshape(val_S*val_B_short_chunk,val_N)
                val_W_C_flat = val_W_C_rep.reshape(val_S*val_B_short_chunk,val_M,val_K)
                val_W_R_flat = val_W_R_rep.reshape(val_S*val_B_short_chunk,val_M,RADAR_STREAMS)

                val_H_eff_H_flat = comm_net.compute_effective_channel(val_h_dk_flat,val_h_rk_flat,val_G_flat,val_theta_flat)

                val_metrics_flat = comm_net.compute_isac_batch_performance(val_H_eff_H_flat,val_g_dt_flat,val_W_C_flat,val_W_R_flat)
                ''' 輸出結果 :
                        SINR power components:
                            signal       : (S*B_chunk*L,K)
                            comm_interf  : (S*B_chunk*L,K)
                            radar_interf : (S*B_chunk*L,K)
                            noise        : scalar

                        raw per-channel-sample tensors:
                            sinr          : (S*B_chunk*L,K), linear
                            sinr_db       : (S*B_chunk*L,K), dB
                            rate          : (S*B_chunk*L,K)
                            sumrate       : (S*B_chunk*L,)
                            target_snr    : (S*B_chunk*L,), linear
                            target_snr_db : (S*B_chunk*L,), dB
                '''

                # 還原shape並取輸出結果
                val_robust_allUE_rate = val_metrics_flat["rate"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)
                val_robust_target_snr = val_metrics_flat["target_snr"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE)
                val_robust_target_snr_db = val_metrics_flat["target_snr_db"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE)

                val_robust_signal = val_metrics_flat["signal"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)
                val_robust_comm_interf = val_metrics_flat["comm_interf"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)
                val_robust_radar_interf = val_metrics_flat["radar_interf"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)
                val_robust_noise = val_metrics_flat["noise"]

                # Robust計算：每個slot先對K取worst UE，再對S取quantile，最後對L與B_val平均
                val_worstUE_rate = torch.min(val_robust_allUE_rate,dim=3).values                          # (S,B_chunk,L)
                val_robust_per_slot = torch.quantile(val_worstUE_rate,OUTAGE_QUANTILE,dim=0)              # (B_chunk,L)
                val_robust_sum += torch.sum(val_robust_per_slot)

                # SINR組成
                val_robust_signal_sum += torch.sum(val_robust_signal,dim=(0,1,2))
                val_robust_comm_interf_sum += torch.sum(val_robust_comm_interf,dim=(0,1,2))
                val_robust_radar_interf_sum += torch.sum(val_robust_radar_interf,dim=(0,1,2))

                # Robust target SNR：累加所有S筆injection、B_chunk個blocks與L個slots的linear target SNR
                val_robust_target_snr_sum += torch.sum(val_robust_target_snr)

                # 每個short-term slot對S筆injection取target SNR Q10%
                val_target_snr_q10_db = torch.quantile(val_robust_target_snr_db,TAR_OUTAGE_QUANTILE,dim=0)  # (B_chunk,L)

                # Q10%與4 dB門檻比較
                val_sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - val_target_snr_q10_db)
                val_sensing_penalty_sum += torch.sum(val_sensing_deficit_db)

                # 每個short-term slot在S筆injection下的實際違反率
                val_sensing_violation_per_channel = torch.mean((val_robust_target_snr_db < SENSING_SNR_THRESHOLD_DB).float(),dim=0)  # (B_chunk,L)

                # 累加所有short-term slots的違反率
                val_sensing_violation_sum += torch.sum(val_sensing_violation_per_channel)

                # 累加有多少short-term slots違反率超過10%
                val_sensing_channel_failure_sum += torch.sum((val_sensing_violation_per_channel > TAR_OUTAGE_QUANTILE).float())

                # 此chunk計算完成後釋放大型injection tensors與其views，下一個chunk重複使用GPU記憶體
                del val_h_dk_inj,val_h_rk_inj,val_G_inj,val_g_dt_inj,val_h_dk_flat,val_h_rk_flat,val_G_flat,val_g_dt_flat
                del val_theta_flat,val_W_C_flat,val_W_R_flat,val_H_eff_H_flat,val_metrics_flat
                del val_robust_allUE_rate,val_robust_target_snr,val_robust_target_snr_db,val_robust_signal,val_robust_comm_interf,val_robust_radar_interf
                del val_worstUE_rate,val_robust_per_slot,val_target_snr_q10_db,val_sensing_deficit_db,val_sensing_violation_per_channel

            # ================================
            # 合併全部 validation chunks
            # ================================
            # Robust原本是先對每個slot取quantile，再對全部B_val與L平均；chunk只改計算方式，不改數學定義
            val_robust = val_robust_sum / (B_val * LT_BLOCK_SIZE)

            # SINR組成：原本對S,B_val,L平均
            val_metric_count = val_S * B_val * LT_BLOCK_SIZE
            val_robust_signal_power = val_robust_signal_sum / val_metric_count
            val_robust_comm_interf_power = val_robust_comm_interf_sum / val_metric_count
            val_robust_radar_interf_power = val_robust_radar_interf_sum / val_metric_count

            # Robust target SNR：對所有S筆injection、B_val個blocks與L個slots做linear mean，再轉dB
            val_robust_target_snr_mean = val_robust_target_snr_sum / val_metric_count
            val_robust_target_snr_mean_db = 10.0 * torch.log10(val_robust_target_snr_mean.clamp_min(1e-12))

            # Q10%與感測門檻比較後，對全部B_val與L平均
            val_short_slot_count = B_val * LT_BLOCK_SIZE
            val_sensing_penalty = val_sensing_penalty_sum / val_short_slot_count

            # 所有short-term slots的平均違反率
            val_sensing_violation_rate = val_sensing_violation_sum / val_short_slot_count

            # 有多少比例的short-term slots違反率超過10%
            val_sensing_channel_failure_rate = val_sensing_channel_failure_sum / val_short_slot_count

            # loss function
            val_loss = -val_robust + ROB_SENSING_LOSS_WEIGHT * val_sensing_penalty + ROB_TX_POWER_LOSS_WEIGHT * val_tx_power_penalty

        val_logs = {
            "loss": float(val_loss.detach().cpu()),
            "robust": float(val_robust.detach().cpu()),
            "target_snr_db": float(val_robust_target_snr_mean_db.detach().cpu()),
            "sensing_penalty": float(val_sensing_penalty.detach().cpu()),
            "sensing_violation_rate": float(val_sensing_violation_rate.detach().cpu()),
            "sensing_channel_failure_rate": float(val_sensing_channel_failure_rate.detach().cpu()),

            "signal_power": val_robust_signal_power.detach().cpu().numpy(),
            "comm_interf_power": val_robust_comm_interf_power.detach().cpu().numpy(),
            "radar_interf_power": val_robust_radar_interf_power.detach().cpu().numpy(),
            "noise_power": float(val_robust_noise.detach().cpu()),

            "comm_tx_power": float(torch.mean(val_W_C_power).detach().cpu()),
            "radar_tx_power": float(torch.mean(val_W_R_power).detach().cpu()),
            "total_tx_power": float(torch.mean(val_total_tx_power).detach().cpu()),
            "tx_power_penalty": float(val_tx_power_penalty.detach().cpu()),
            "tx_power_violation_rate": float(val_tx_power_violation_rate.detach().cpu()),
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

        curves["train_comm_tx_power"].append(train_logs["comm_tx_power"])
        curves["val_comm_tx_power"].append(val_logs["comm_tx_power"])

        curves["train_radar_tx_power"].append(train_logs["radar_tx_power"])
        curves["val_radar_tx_power"].append(val_logs["radar_tx_power"])

        curves["train_total_tx_power"].append(train_logs["total_tx_power"])
        curves["val_total_tx_power"].append(val_logs["total_tx_power"])

        curves["train_tx_power_penalty"].append(train_logs["tx_power_penalty"])
        curves["val_tx_power_penalty"].append(val_logs["tx_power_penalty"])

        curves["train_tx_power_violation_rate"].append(train_logs["tx_power_violation_rate"])
        curves["val_tx_power_violation_rate"].append(val_logs["tx_power_violation_rate"])

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

        train_comm_power_ratio = train_logs["comm_tx_power"] / TRANSMIT_POWER_TOTAL
        train_radar_power_ratio = train_logs["radar_tx_power"] / TRANSMIT_POWER_TOTAL
        train_total_power_ratio = train_logs["total_tx_power"] / TRANSMIT_POWER_TOTAL

        val_comm_power_ratio = val_logs["comm_tx_power"] / TRANSMIT_POWER_TOTAL
        val_radar_power_ratio = val_logs["radar_tx_power"] / TRANSMIT_POWER_TOTAL
        val_total_power_ratio = val_logs["total_tx_power"] / TRANSMIT_POWER_TOTAL


        # 印出每個epoch log
        print(
            f"\n[Epoch {epoch + 1:03d}/{ROB_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "
            f"TrainRobust={train_logs['robust']:.4f} "
            f"ValRobust={val_logs['robust']:.4f} | "
            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB"
            f"\nTrainSenseViolation={100.0 * train_logs['sensing_violation_rate']:.2f}% "
            f"ValSenseViolation={100.0 * val_logs['sensing_violation_rate']:.2f}% | "

            #f"\nTrainSignal={fmt_vec_sci(train_logs['signal_power'],precision=3)} "
            #f"ValSignal={fmt_vec_sci(val_logs['signal_power'],precision=3)}"
            #f"\nTrainCommInterf={fmt_vec_sci(train_logs['comm_interf_power'],precision=3)} "
            #f"ValCommInterf={fmt_vec_sci(val_logs['comm_interf_power'],precision=3)}"
            #f"\nTrainRadarInterf={fmt_vec_sci(train_logs['radar_interf_power'],precision=3)} "
            #f"ValRadarInterf={fmt_vec_sci(val_logs['radar_interf_power'],precision=3)}"
            #f"\nTrainNoise={train_logs['noise_power']:.3e} "
            #f"ValNoise={val_logs['noise_power']:.3e}"
            f"\nTrainPower W_C_raw={train_logs['comm_tx_power']:.6f} W "
            f"W_R_raw={train_logs['radar_tx_power']:.6f} W | "
            f"W_C:W_R=({train_comm_power_ratio:.4f}:{train_radar_power_ratio:.4f}) "
            f"Total/P_T={train_total_power_ratio:.4f} "
            f"Violation={100.0 * train_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty={train_logs['tx_power_penalty']:.6e}"

            f"\nValPower   W_C_raw={val_logs['comm_tx_power']:.6f} W "
            f"W_R_raw={val_logs['radar_tx_power']:.6f} W | "
            f"W_C:W_R=({val_comm_power_ratio:.4f}:{val_radar_power_ratio:.4f}) "
            f"Total/P_T={val_total_power_ratio:.4f} "
            f"Violation={100.0 * val_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty={val_logs['tx_power_penalty']:.6e}"
        )

    plot_rob_curves(rob_curve_path,rob_curve_dir)


    print("\n[INFO] ROB training finished.")
    print(f"[INFO] Best validation epoch          = {best_val_epoch}")
    print(f"[INFO] Best validation loss           = {best_val_loss:.6f}")
    print(f"[INFO] Robust rate at best val loss   = {best_val_worstUE_q05:.6f}")       # 這裡要print Best validation robust
    print(f"[INFO] Target SNR at best val loss    = {best_valtarget_snr_db:.3f} dB")
    print("[INFO] Short-term robust training finished.")
