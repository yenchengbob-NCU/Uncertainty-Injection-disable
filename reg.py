# -*- coding: utf-8 -*-
import os
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


def plot_reg_curves(reg_curve_path, reg_curve_dir, ma_window=20):
    """
    畫三類圖：
    1. Train / Validation Loss
    2. Train / Validation Nominal worst-UE rate
    3. 每個 UE 分開畫 SINR power components，共 8 條線：
       Train / Val signal
       Train / Val communication interference
       Train / Val radar interference
       Train / Val noise
    """
    data = np.load(reg_curve_path)
    out_dir = reg_curve_dir

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
    plt.title("Fig. 1: Two-stage REG Loss")
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
    plt.title("Fig. 2: Two-stage REG Nominal Worst-UE Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_nominal_curve.png")
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
        plt.title(
            f"Fig. 3: One-timescale REG SINR Power Components — UE {ue_idx}"
        )
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


# ================================
# Main
# ================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train two-timescale REG networks.")
    parser.add_argument("--pretrain",action="store_true",help="Load pretrained RZF CommNet and NS-MRT RadarNet checkpoints.")
    args = parser.parse_args()

    comm_net  = CommNet().to(DEVICE)
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
    # Train REG
    # ================================

    # 存檔路徑
    reg_comm_ckpt  = os.path.join(REG_CKPT_DIR, "two_timescale_comm_reg.ckpt")
    reg_radar_ckpt = os.path.join(REG_CKPT_DIR, "two_timescale_radar_reg.ckpt")
    reg_curve_path = os.path.join(REG_CKPT_DIR, "two_timescale_reg_curves.npz")
    reg_curve_dir  = os.path.join(REG_CKPT_DIR, "two_timescale_reg_curves")
    # 已經訓練的RIS ckpt 路徑
    ris_only_ckpt  = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    comm_pretrain_ckpt = os.path.join(PRETRAIN_DIR,"comm_rzf_pretrain.ckpt")
    radar_pretrain_ckpt = os.path.join(PRETRAIN_DIR,"radar_ns_mrt_pretrain.ckpt")

    os.makedirs(reg_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],               # Loss function 看收斂
        "val_loss": [],

        "train_nominal": [],            # Nominal ≜ 每slot先找K個UE中 worst UE rate,再對L slots與B blocks平均
        "val_nominal": [],

        "train_target_snr_db": [],      # 感測SNR 不畫圖
        "val_target_snr_db": [],

        "train_sensing_penalty": [],    # 感測懲罰 不畫圖
        "val_sensing_penalty": [],

        # SINR power components
        # signal / interference 每個 epoch 儲存 (K,)
        "train_signal_power": [],
        "val_signal_power": [],

        "train_comm_interf_power": [],
        "val_comm_interf_power": [],

        "train_radar_interf_power": [],
        "val_radar_interf_power": [],

        # noise 每個 epoch 儲存 scalar
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
    best_val_loss = float("inf")   # L_best <- infinity loss 越小越好 先設無限大
    best_val_worstUE_rate = 0.0    # best loss 對應的 Nominal worst-UE rate
    best_valtarget_snr_db = 0.0
    best_val_epoch = 0             # best loss 出現在哪個 epoch

    patience_counter = 0           
    EARLY_STOP_EPS = 1e-6          
    PATIENCE = 30                  

    # 載入資料
    train_trajectories, train_channels = train_dataset["h_dk"].shape[:2]
    val_trajectories, val_channels     = val_dataset["h_dk"].shape[:2]

    val_channel_ids = np.arange(0,val_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
    val_trajectory_ids = np.repeat(np.arange(val_trajectories),len(val_channel_ids))
    val_channel_ids = np.tile(val_channel_ids,val_trajectories)
    B_val = len(val_channel_ids)

    print("\n" + "=" * 90)
    print("[TWO TIMESCALE REG TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"REG_EPOCHS              : {REG_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"ST_BATCH_SIZE          : {ST_BATCH_SIZE}")
    print(f"REG learning rate       : {REG_LEARNING_RATE}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {REG_SENSING_LOSS_WEIGHT}")
    print("=" * 90)

    # 載入已訓練好的ThetaNet
    if not os.path.exists(ris_only_ckpt):
        raise FileNotFoundError(f"找不到 ThetaNet checkpoint: {ris_only_ckpt}")

    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)


    # ================================
    # CommNet / RadarNet initialization
    # ================================
    if args.pretrain:
        # CommNet不載入RZF預訓練，保留random initialization
        comm_init_mode = "Random initialization"

        if not os.path.exists(radar_pretrain_ckpt):
            raise FileNotFoundError(f"找不到 RadarNet pretrain checkpoint: {radar_pretrain_ckpt}")

        radar_net.load_model(radar_pretrain_ckpt,strict=True,verbose=True)
        radar_init_mode = "NS-MRT pretrained"

        print("[INFO] CommNet uses random initialization.")
        print(f"[INFO] Loaded RadarNet pretrain checkpoint: {radar_pretrain_ckpt}")

    else:
        comm_init_mode  = "Random initialization"
        radar_init_mode = "Random initialization"

        print("[INFO] CommNet uses random initialization.")
        print("[INFO] RadarNet uses random initialization.")


    for parameter in comm_net.parameters():
        parameter.requires_grad_(True)

    for parameter in radar_net.parameters():
        parameter.requires_grad_(True)


    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=REG_LEARNING_RATE,
    )

    print(f"[INFO] CommNet initialization mode : {comm_init_mode}")
    print(f"[INFO] RadarNet initialization mode: {radar_init_mode}")
    print("[INFO] CommNet and RadarNet remain trainable during REG joint training.")


    COMM_INIT_TARGET_POWER = TRANSMIT_POWER_TOTAL / 2.0   # 0.63 W
    comm_init_power_scaled = False






    # 開始訓練
    for epoch in trange(REG_EPOCHS, desc="REG Training"):
        
        comm_net.train()
        radar_net.train()

        # 一個batch 產生一個 loss 更新一次網路,這裡累積要在 epoch log 輸出的值,最後在印出時做平均
        epoch_loss              = []    # 累積N個batch的loss
        epoch_Nominal           = []    # 累積N個batch的Nominal
        epoch_target_snr        = []    
        epoch_sensing_penalties = []
        epoch_signal_power      = []
        epoch_comm_interf_power = []
        epoch_radar_interf_power= []
        epoch_noise_power       = []
        epoch_sensing_violation_rates = []
        epoch_comm_tx_power = []
        epoch_radar_tx_power = []
        epoch_total_tx_power = []
        epoch_tx_power_penalties = []
        epoch_tx_power_violation_rates = []

        # 一個epoch 有 N_BATCHE 個 batch , 一個 batch 有ST_BATCH_SIZE個 long-term blocks
        for _ in range(N_BATCHE):
            # 從data中抽出 ST_BATCH_SIZE 個 long-term blocks
            trajectory_ids = np.random.choice(train_trajectories,size=ST_BATCH_SIZE,replace=True)
            channel_ids = np.random.choice(train_channels,size=ST_BATCH_SIZE,replace=False)

            valid = channel_ids + LT_BLOCK_SIZE <= train_channels
            trajectory_ids = trajectory_ids[valid]
            channel_ids = channel_ids[valid]
            B = len(channel_ids)

            # ThetaNet只看每個long-term block的start-slot CSI
            h_dk_hat = torch.as_tensor(train_dataset["h_dk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)  # (B, M, K)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)  # (B, N, K)
            G_hat    = torch.as_tensor(train_dataset["G"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)     # (B, N, M)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)  # (B, M, 1)

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

            # NN直接輸出beamformer，包含方向與功率分配
            W_C_raw = comm_net(H_eff_H,g_dt_block)
            # --pretrain模式下，CommNet不載入teacher，只在joint training第一次forward時
            # 將random CommNet的平均初始輸出power校準至P_T/2，之後不再限制其power
            if args.pretrain and not comm_init_power_scaled:

                with torch.no_grad():
                    comm_init_power_before = torch.mean(
                        torch.sum(torch.abs(W_C_raw) ** 2,dim=(1,2)).real
                    )

                    comm_init_scale = torch.sqrt(
                        torch.tensor(COMM_INIT_TARGET_POWER,dtype=torch.float32,device=DEVICE)
                        / comm_init_power_before.clamp_min(1e-12)
                    )

                    comm_net.fc_out_real.weight.mul_(comm_init_scale)
                    comm_net.fc_out_real.bias.mul_(comm_init_scale)
                    comm_net.fc_out_imag.weight.mul_(comm_init_scale)
                    comm_net.fc_out_imag.bias.mul_(comm_init_scale)

                # output layer已改變，因此重新forward一次
                W_C_raw = comm_net(H_eff_H,g_dt_block)

                comm_init_power_after = torch.mean(
                    torch.sum(torch.abs(W_C_raw) ** 2,dim=(1,2)).real
                )

                print(
                    f"\n[INFO] CommNet one-time initial power calibration: "
                    f"{comm_init_power_before.item():.6f} W -> "
                    f"{comm_init_power_after.item():.6f} W"
                )

                comm_init_power_scaled = True
            W_R_raw = radar_net(H_eff_H,g_dt_block)

            # 每一筆estimated channel的beamformer power
            W_C_power = torch.sum(torch.abs(W_C_raw) ** 2,dim=(1,2)).real
            W_R_power = torch.sum(torch.abs(W_R_raw) ** 2,dim=(1,2)).real
            total_tx_power = W_C_power + W_R_power

            # Dimensionless total TX power constraint penalty
            total_tx_power_ratio = total_tx_power / TRANSMIT_POWER_TOTAL
            tx_power_excess_ratio = torch.relu(total_tx_power_ratio - 1.0)
            tx_power_penalty = torch.mean(tx_power_excess_ratio ** 2)

            # 只供log顯示：有多少channel超過總傳輸功率
            tx_power_violation_rate = torch.mean((total_tx_power > TRANSMIT_POWER_TOTAL).float())

            metrics = comm_net.compute_isac_batch_performance(H_eff_H,g_dt_block,W_C_raw,W_R_raw)
            """ 輸出結果 :
                    SINR power components:
                        signal       : (B*L, K)
                        comm_interf  : (B*L, K)
                        radar_interf : (B*L, K)
                        noise        : scalar

                    raw per-channel-sample tensors:
                        sinr          : (B*L, K), linear
                        sinr_db       : (B*L, K), dB
                        rate          : (B*L, K)
                        sumrate       : (B*L,)
                        target_snr    : (B*L,), linear
                        target_snr_db : (B*L,), dB

                    B*L-average display values: 對所有short-term slots 平均
                        sinr_user_mean      : (K,), linear
                        sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                        rate_user_mean      : (K,)
                        sumrate_mean        : scalar
                        target_snr_mean     : scalar, linear
                        target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
            """

            # 取輸出結果
            nominal_allUE_rate      = metrics["rate"].reshape(B,LT_BLOCK_SIZE,UAV_COMM)    # (B,L,K)
            nominal_target_snr      = metrics["target_snr"].reshape(B,LT_BLOCK_SIZE)        # (B,L), linear
            nominal_target_snr_db   = metrics["target_snr_db"].reshape(B,LT_BLOCK_SIZE)     # (B,L), dB

            nominal_signal          = metrics["signal"]                             # (B*L,K)  各UE的signal power
            nominal_comm_interf     = metrics["comm_interf"]                        # (B*L,K)
            nominal_radar_interf    = metrics["radar_interf"]                       # (B*L,K)
            nominal_noise           = metrics["noise"]                              # scalar

            # Nominal 計算
            nominal_with_batch = torch.min(nominal_allUE_rate,dim=2).values          # (B,L) 每個slot找出K個UE中rate最小的
            nominal_with_block = torch.mean(nominal_with_batch,dim=1)                # (B,) 對每個block的L slots平均
            nominal = torch.mean(nominal_with_block)                                 # scalar 對B個long-term blocks平均

            # SINR組成 
            signal_power         = torch.mean(nominal_signal,dim=0)                 #(K,)
            comm_interf_power    = torch.mean(nominal_comm_interf, dim=0)           #(K,)
            radar_interf_power   = torch.mean(nominal_radar_interf, dim=0)          #(K,)

            # Nominal target SNR：先計算目前batch的linear mean
            nominal_target_snr_mean = torch.mean(nominal_target_snr)       # scalar, linear

            # Nominal sensing violation rate：只供log顯示
            sensing_violation_rate  = torch.mean((nominal_target_snr_db < SENSING_SNR_THRESHOLD_DB).float())

            # 每一個 short-term slot 的 nominal target SNR 都與 4 dB 門檻比較
            sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - nominal_target_snr_db)   # (B,L)
            sensing_penalty = torch.mean(sensing_deficit_db)                                    # scalar

            # loss function
            loss = -nominal + REG_SENSING_LOSS_WEIGHT * sensing_penalty + REG_TX_POWER_LOSS_WEIGHT * tx_power_penalty

            loss.backward()     # 反向傳播
            optimizer.step()    # 更新NN權重

            # 累加在 epoch 向量，供 epoch log 顯示
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_Nominal.append(float(nominal.detach().cpu()))
            epoch_target_snr.append(float(nominal_target_snr_mean.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))
            epoch_sensing_violation_rates.append(float(sensing_violation_rate.detach().cpu()))

            epoch_signal_power.append(signal_power.detach().cpu().numpy())              
            epoch_comm_interf_power.append(comm_interf_power.detach().cpu().numpy())    
            epoch_radar_interf_power.append(radar_interf_power.detach().cpu().numpy())  
            epoch_noise_power.append(float(nominal_noise.detach().cpu()))

            epoch_comm_tx_power.append(float(torch.mean(W_C_power).detach().cpu()))
            epoch_radar_tx_power.append(float(torch.mean(W_R_power).detach().cpu()))
            epoch_total_tx_power.append(float(torch.mean(total_tx_power).detach().cpu()))
            epoch_tx_power_penalties.append(float(tx_power_penalty.detach().cpu()))
            epoch_tx_power_violation_rates.append(float(tx_power_violation_rate.detach().cpu()))

        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "nominal": float(np.mean(epoch_Nominal)),
            "target_snr_db": train_target_snr_mean_db,
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
            "sensing_violation_rate": float(np.mean(epoch_sensing_violation_rates)),

            "comm_tx_power": float(np.mean(epoch_comm_tx_power)),
            "radar_tx_power": float(np.mean(epoch_radar_tx_power)),
            "total_tx_power": float(np.mean(epoch_total_tx_power)),
            "tx_power_penalty": float(np.mean(epoch_tx_power_penalties)),
            "tx_power_violation_rate": float(np.mean(epoch_tx_power_violation_rates)),

            "signal_power": np.mean(np.stack(epoch_signal_power,axis=0),axis=0),
            "comm_interf_power": np.mean(np.stack(epoch_comm_interf_power,axis=0),axis=0),
            "radar_interf_power": np.mean(np.stack(epoch_radar_interf_power,axis=0),axis=0),
            "noise_power": float(np.mean(epoch_noise_power)),
        }

        # ================================
        # Validation: 固定 0,L,2L,... block starts 的全部 val trajectories
        # ================================
        comm_net.eval()
        radar_net.eval()

        with torch.no_grad():
            # ThetaNet只看每個validation long-term block的start-slot CSI
            h_dk_val = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)  # (B_val,M,K)
            h_rk_val = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)  # (B_val,N,K)
            G_val    = torch.as_tensor(val_dataset["G"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)     # (B_val,N,M)
            g_dt_val = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)  # (B_val,M,1)

            theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)  # (B_val,N)

            val_block_ids = val_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]  # (B_val,L)

            h_dk_val_block = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_val_block = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            G_val_block    = torch.as_tensor(val_dataset["G"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_val_block = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)

            h_dk_val_block = h_dk_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            h_rk_val_block = h_rk_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            G_val_block    = G_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            g_dt_val_block = g_dt_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,1)
            theta_val_block = theta_val[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT)

            # 每個slot使用自己的channel計算等效通道，固定的只有theta
            H_eff_val = comm_net.compute_effective_channel(h_dk_val_block,h_rk_val_block,G_val_block,theta_val_block)

            W_C_val_raw = comm_net(H_eff_val,g_dt_val_block)
            W_R_val_raw = radar_net(H_eff_val,g_dt_val_block)

            val_W_C_power = torch.sum(torch.abs(W_C_val_raw) ** 2,dim=(1,2)).real
            val_W_R_power = torch.sum(torch.abs(W_R_val_raw) ** 2,dim=(1,2)).real
            val_total_tx_power = val_W_C_power + val_W_R_power

            val_total_tx_power_ratio = val_total_tx_power / TRANSMIT_POWER_TOTAL
            val_tx_power_excess_ratio = torch.relu(val_total_tx_power_ratio - 1.0)
            val_tx_power_penalty = torch.mean(val_tx_power_excess_ratio ** 2)

            val_tx_power_violation_rate = torch.mean((val_total_tx_power > TRANSMIT_POWER_TOTAL).float())

            val_metrics = comm_net.compute_isac_batch_performance(H_eff_val,g_dt_val_block,W_C_val_raw,W_R_val_raw)
            """ 輸出結果 :
                    SINR power components:
                        signal       : (B_val*L, K)
                        comm_interf  : (B_val*L, K)
                        radar_interf : (B_val*L, K)
                        noise        : scalar

                    raw per-channel-sample tensors:
                        sinr          : (B_val*L, K), linear
                        sinr_db       : (B_val*L, K), dB
                        rate          : (B_val*L, K)
                        sumrate       : (B_val*L,)
                        target_snr    : (B_val*L,), linear
                        target_snr_db : (B_val*L,), dB

                    B_val*L-average display values: 對所有validation short-term slots 平均
                        sinr_user_mean      : (K,), linear
                        sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                        rate_user_mean      : (K,)
                        sumrate_mean        : scalar
                        target_snr_mean     : scalar, linear
                        target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
            """

            # 取輸出結果
            val_nominal_allUE_rate      = val_metrics["rate"].reshape(B_val,LT_BLOCK_SIZE,UAV_COMM)  # (B_val,L,K)
            val_nominal_target_snr      = val_metrics["target_snr"].reshape(B_val,LT_BLOCK_SIZE)      # (B_val,L), linear
            val_nominal_target_snr_db   = val_metrics["target_snr_db"].reshape(B_val,LT_BLOCK_SIZE)   # (B_val,L), dB

            val_nominal_signal          = val_metrics["signal"]                          # (B_val*L,K) 各UE的signal power
            val_nominal_comm_interf     = val_metrics["comm_interf"]                     # (B_val*L,K)
            val_nominal_radar_interf    = val_metrics["radar_interf"]                    # (B_val*L,K)
            val_nominal_noise           = val_metrics["noise"]                           # scalar

            # Nominal計算
            val_nominal_with_batch = torch.min(val_nominal_allUE_rate,dim=2).values      # (B_val,L) 每個slot找出K個UE中rate最小的
            val_nominal_with_block = torch.mean(val_nominal_with_batch,dim=1)            # (B_val,) 對每個block的L slots平均
            val_nominal = torch.mean(val_nominal_with_block)                             # scalar 對全部validation long-term blocks平均

            # SINR組成
            val_signal_power = torch.mean(val_nominal_signal,dim=0)                     # (K,)
            val_comm_interf_power = torch.mean(val_nominal_comm_interf,dim=0)           # (K,)
            val_radar_interf_power = torch.mean(val_nominal_radar_interf,dim=0)         # (K,)

            # Nominal target SNR：linear mean後轉dB
            val_nominal_target_snr_mean = torch.mean(val_nominal_target_snr)            # scalar, linear
            val_target_snr_mean_db = 10.0 * torch.log10(val_nominal_target_snr_mean.clamp_min(1e-12))

            # Nominal sensing violation rate：只供log顯示
            val_sensing_violation_rate = torch.mean((val_nominal_target_snr_db < SENSING_SNR_THRESHOLD_DB).float())

            # 每一個 validation short-term slot 的 nominal target SNR 與4 dB門檻比較
            val_sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - val_nominal_target_snr_db)
            val_sensing_penalty = torch.mean(val_sensing_deficit_db)

            # Loss function
            val_loss = -val_nominal + REG_SENSING_LOSS_WEIGHT * val_sensing_penalty + REG_TX_POWER_LOSS_WEIGHT * val_tx_power_penalty

        val_logs = {
            "loss": float(val_loss.detach().cpu()),
            "nominal": float(val_nominal.detach().cpu()),
            "target_snr_db": float(val_target_snr_mean_db.detach().cpu()),
            "sensing_penalty": float(val_sensing_penalty.detach().cpu()),
            "sensing_violation_rate": float(val_sensing_violation_rate.detach().cpu()),

            "comm_tx_power": float(torch.mean(val_W_C_power).detach().cpu()),
            "radar_tx_power": float(torch.mean(val_W_R_power).detach().cpu()),
            "total_tx_power": float(torch.mean(val_total_tx_power).detach().cpu()),
            "tx_power_penalty": float(val_tx_power_penalty.detach().cpu()),
            "tx_power_violation_rate": float(val_tx_power_violation_rate.detach().cpu()),

            "signal_power": val_signal_power.detach().cpu().numpy(),
            "comm_interf_power": val_comm_interf_power.detach().cpu().numpy(),
            "radar_interf_power": val_radar_interf_power.detach().cpu().numpy(),
            "noise_power": float(val_nominal_noise.detach().cpu()),
        }

        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_nominal"].append(train_logs["nominal"])
        curves["val_nominal"].append(val_logs["nominal"])

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
            reg_curve_path,
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

            comm_net.save_model(reg_comm_ckpt, verbose=False)
            radar_net.save_model(reg_radar_ckpt, verbose=False)
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

        # 印出每個eopch log
        print(
            f"\n[Epoch {epoch + 1:03d}/{REG_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "

            f"TrainNominal={train_logs['nominal']:.4f} "
            f"ValNominal={val_logs['nominal']:.4f} | "

            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "

            f"TrainSenseViolation={100.0 * train_logs['sensing_violation_rate']:.2f}% "
            f"ValSenseViolation={100.0 * val_logs['sensing_violation_rate']:.2f}%"

            #f"\nTrainSignal={fmt_vec_sci(train_logs['signal_power'], precision=3)} "
            #f"ValSignal={fmt_vec_sci(val_logs['signal_power'], precision=3)} "

            #f"\nTrainCommInterf={fmt_vec_sci(train_logs['comm_interf_power'], precision=3)} "
            #f"ValCommInterf={fmt_vec_sci(val_logs['comm_interf_power'], precision=3)} "

            #f"\nTrainRadarInterf={fmt_vec_sci(train_logs['radar_interf_power'], precision=3)} "
            #f"ValRadarInterf={fmt_vec_sci(val_logs['radar_interf_power'], precision=3)} "

            #f"\nTrainNoise={train_logs['noise_power']:.3e} "
            #f"ValNoise={val_logs['noise_power']:.3e}"
            f"\nTrainPower W_C_raw={train_logs['comm_tx_power']:.8f} W "
            f"W_R_raw={train_logs['radar_tx_power']:.8f} W | "
            f"P_C/P_T:P_R/P_T=({train_comm_power_ratio:.3f}:{train_radar_power_ratio:.3f}) "
            f"Total/P_T={train_total_power_ratio:.3f} "
            f"Violation={100.0 * train_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty={train_logs['tx_power_penalty']:.6e}"
            f"\nValPower   W_C_raw={val_logs['comm_tx_power']:.4f} W "
            f"W_R_raw={val_logs['radar_tx_power']:.4f} W | "
            f"P_C/P_T:P_R/P_T=({val_comm_power_ratio:.3f}:{val_radar_power_ratio:.3f}) "
            f"Total/P_T={val_total_power_ratio:.3f} "
            f"Violation={100.0 * val_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty={val_logs['tx_power_penalty']:.6e}"
        )

    plot_reg_curves(reg_curve_path,reg_curve_dir)

    print("[INFO] REG training finished.")
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"REG_EPOCHS              : {REG_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"ST_BATCH_SIZE          : {ST_BATCH_SIZE}")
    print(f"REG learning rate       : {REG_LEARNING_RATE}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {REG_SENSING_LOSS_WEIGHT}")
    print("=" * 90)
    print(f"[INFO] Best validation epoch          = {best_val_epoch}")
    print(f"[INFO] Best validation loss           = {best_val_loss:.6f}")
    print(f"[INFO] Nominal rate at best val loss  = {best_val_worstUE_rate:.6f}")
    print(f"[INFO] Target SNR at best val loss    = {best_valtarget_snr_db:.3f} dB")     
    print("[INFO] Short-term regular training finished.")
