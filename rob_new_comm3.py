# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
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

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]
    train_robust = data["train_robust"]
    val_robust = data["val_robust"]
    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig. 1: Train / Validation Loss
    # ================================
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", alpha=0.35, linewidth=1.0)
    plt.plot(epochs, val_loss, label="Validation Loss", alpha=0.35, linewidth=1.0)

    train_ma_epochs, train_loss_ma = moving_average(train_loss, ma_window)
    val_ma_epochs, val_loss_ma = moving_average(val_loss, ma_window)

    if len(train_loss_ma) > 0:
        plt.plot(train_ma_epochs, train_loss_ma, label=f"Train Loss MA({ma_window})", linewidth=2.5)

    if len(val_loss_ma) > 0:
        plt.plot(val_ma_epochs, val_loss_ma, label=f"Validation Loss MA({ma_window})", linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fig. 1: Comm3 Pretrained CommNet + NS-MRT Joint ROB Loss")
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
    plt.plot(epochs, train_robust, label="Train Robust Worst-UE Rate", alpha=0.35, linewidth=1.0)
    plt.plot(epochs, val_robust, label="Validation Robust Worst-UE Rate", alpha=0.35, linewidth=1.0)

    train_ma_epochs, train_robust_ma = moving_average(train_robust, ma_window)
    val_ma_epochs, val_robust_ma = moving_average(val_robust, ma_window)

    if len(train_robust_ma) > 0:
        plt.plot(train_ma_epochs, train_robust_ma, label=f"Train Robust MA({ma_window})", linewidth=2.5)

    if len(val_robust_ma) > 0:
        plt.plot(val_ma_epochs, val_robust_ma, label=f"Validation Robust MA({ma_window})", linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel(f"Robust Worst-UE Rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
    plt.title("Fig. 2: Comm3 Pretrained CommNet + NS-MRT Joint ROB Worst-UE Tail Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_robust_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {fig2_path}")

    # ================================
    # Fig. 3: SINR Power Components
    # ================================
    train_signal_power = data["train_signal_power"]
    val_signal_power = data["val_signal_power"]
    train_comm_interf_power = data["train_comm_interf_power"]
    val_comm_interf_power = data["val_comm_interf_power"]
    train_radar_interf_power = data["train_radar_interf_power"]
    val_radar_interf_power = data["val_radar_interf_power"]
    train_noise_power = data["train_noise_power"]
    val_noise_power = data["val_noise_power"]

    num_users = train_signal_power.shape[1]

    for ue_idx in range(num_users):
        plt.figure(figsize=(12, 7))

        plt.plot(epochs, train_signal_power[:, ue_idx], label="Train Signal", linewidth=1.8)
        plt.plot(epochs, val_signal_power[:, ue_idx], label="Validation Signal", linewidth=1.8, linestyle="--")

        plt.plot(epochs, train_comm_interf_power[:, ue_idx], label="Train Communication Interference", linewidth=1.8)
        plt.plot(epochs, val_comm_interf_power[:, ue_idx], label="Validation Communication Interference", linewidth=1.8, linestyle="--")

        plt.plot(epochs, train_radar_interf_power[:, ue_idx], label="Train Radar Interference", linewidth=1.8)
        plt.plot(epochs, val_radar_interf_power[:, ue_idx], label="Validation Radar Interference", linewidth=1.8, linestyle="--")

        plt.plot(epochs, train_noise_power, label="Train Noise", linewidth=1.8)
        plt.plot(epochs, val_noise_power, label="Validation Noise", linewidth=1.8, linestyle="--")

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Power (linear scale)")
        plt.title(f"Fig. 3: Mean SINR Power Components over Injected Channels — UE {ue_idx}")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()

        fig3_path = os.path.join(out_dir, f"Fig3_UE{ue_idx}_sinr_power_components.png")
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

    # ================================
    # Dataset
    # ================================
    train_dataset_path = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path = os.path.join(DATA_DIR, "dataset_val.npz")

    with np.load(train_dataset_path) as data:
        train_dataset = {key: data[key] for key in ["positions", "h_dk", "h_rk", "G", "g_dt"]}

    with np.load(val_dataset_path) as data:
        val_dataset = {key: data[key] for key in ["positions", "h_dk", "h_rk", "G", "g_dt"]}

    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path   = {val_dataset_path}")

    # ================================
    # ROB outputs
    # 使用 _new 避免覆蓋舊 ROB 實驗
    # ================================
    rob_comm_ckpt = os.path.join(ROB_CKPT_DIR, "two_timescale_comm_rob_new.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR, "two_timescale_radar_rob_new.ckpt")
    rob_curve_path = os.path.join(ROB_CKPT_DIR, "two_timescale_rob_new_curves.npz")
    rob_curve_dir = os.path.join(ROB_CKPT_DIR, "two_timescale_rob_new_curves")

    # ================================
    # Pretrained checkpoints
    # ================================
    ris_only_ckpt       = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")
    comm_pretrain_ckpt  = os.path.join(REG_CKPT_DIR, "two_timescale_comm_reg.ckpt")
    radar_pretrain_ckpt = os.path.join(PRETRAIN_DIR, "radar_ns_mrt_pretrain.ckpt")

    os.makedirs(rob_curve_dir, exist_ok=True)

    curves = {
        "train_loss": [],
        "val_loss": [],

        "train_robust": [],
        "val_robust": [],

        "train_target_snr_db": [],
        "val_target_snr_db": [],

        "train_sensing_penalty": [],
        "val_sensing_penalty": [],

        "train_signal_power": [],
        "val_signal_power": [],

        "train_comm_interf_power": [],
        "val_comm_interf_power": [],

        "train_radar_interf_power": [],
        "val_radar_interf_power": [],

        "train_noise_power": [],
        "val_noise_power": [],

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

    # ================================
    # Training parameters
    # ================================
    best_val_loss = float("inf")
    best_val_worstUE_q05 = 0.0
    best_valtarget_snr_db = 0.0
    best_val_epoch = 0

    patience_counter = 0
    EARLY_STOP_EPS = 1e-6
    PATIENCE = 30
    VAL_CHUNK_SIZE = 250

    train_trajectories, train_channels = train_dataset["h_dk"].shape[:2]
    val_trajectories, val_channels = val_dataset["h_dk"].shape[:2]

    val_channel_ids = np.arange(0, val_channels - LT_BLOCK_SIZE + 1, LT_BLOCK_SIZE)
    val_trajectory_ids = np.repeat(np.arange(val_trajectories), len(val_channel_ids))
    val_channel_ids = np.tile(val_channel_ids, val_trajectories)
    B_val = len(val_channel_ids)

    # ================================
    # Training information
    # ================================
    print("\n" + "=" * 100)
    print("[COMM3 PRETRAINED COMMNET + NS-MRT RADARNET -> JOINT ROB FINE-TUNING]")
    print("=" * 100)
    print(f"Train channels               : {train_channels}")
    print(f"Val channels                 : {val_channels}")
    print(f"ROB_EPOCHS                   : {ROB_EPOCHS}")
    print(f"N_BATCHE                     : {N_BATCHE}")
    print(f"ST_BATCH_SIZE                : {ST_BATCH_SIZE}")
    print(f"ROB learning rate            : {ROB_LEARNING_RATE}")
    print(f"Sensing SNR threshold        : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight       : {ROB_SENSING_LOSS_WEIGHT}")
    print(f"TX power penalty weight      : {ROB_TX_POWER_LOSS_WEIGHT}")
    print(f"Rate outage quantile         : {OUTAGE_QUANTILE}")
    print(f"Sensing outage quantile      : {TAR_OUTAGE_QUANTILE}")
    print(f"Injection samples            : {INJECTION_SAMPLES}")
    print(f"Injection variance           : {INJECTION_VARIANCE}")
    print(f"ThetaNet checkpoint          : {ris_only_ckpt}")
    print(f"CommNet init checkpoint      : {comm_pretrain_ckpt}")
    print(f"RadarNet init checkpoint     : {radar_pretrain_ckpt}")
    print("=" * 100)

    # ================================
    # ThetaNet: frozen ris_only.ckpt
    # ================================
    if not os.path.exists(ris_only_ckpt):
        raise FileNotFoundError(f"找不到 ThetaNet checkpoint: {ris_only_ckpt}")

    theta_net.load_model(ris_only_ckpt, strict=True, verbose=True)
    theta_net.eval()

    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)

    # ================================
    # CommNet: comm_pretrain_3.ckpt
    # ================================
    if not os.path.exists(comm_pretrain_ckpt):
        raise FileNotFoundError(f"找不到 CommNet communication-only pretrain checkpoint: {comm_pretrain_ckpt}")

    comm_net.load_model(comm_pretrain_ckpt, strict=True, verbose=True)

    # ================================
    # RadarNet: NS-MRT pretrain
    # ================================
    if not os.path.exists(radar_pretrain_ckpt):
        raise FileNotFoundError(f"找不到 RadarNet NS-MRT pretrain checkpoint: {radar_pretrain_ckpt}")

    radar_net.load_model(radar_pretrain_ckpt, strict=True, verbose=True)

    # ROB epoch 1 起 CommNet + RadarNet 都可更新
    for parameter in comm_net.parameters():
        parameter.requires_grad_(True)

    for parameter in radar_net.parameters():
        parameter.requires_grad_(True)

    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=ROB_LEARNING_RATE,
    )

    print("[INFO] CommNet initialization mode : comm_pretrain_3 communication-only pretrained")
    print("[INFO] RadarNet initialization mode: NS-MRT pretrained")
    print("[INFO] ThetaNet remains frozen.")
    print("[INFO] CommNet and RadarNet are jointly trainable from ROB epoch 1.")
    print("[INFO] No pretraining loss remains in ROB; training uses the original ROB objective.")

    # ================================
    # ROB Training
    # ================================
    for epoch in trange(ROB_EPOCHS, desc="ROB NEW Comm3 Joint Fine-tuning"):

        comm_net.train()
        radar_net.train()

        epoch_loss = []
        epoch_Robust = []
        epoch_target_snr = []
        epoch_sensing_penalties = []

        epoch_signal_power = []
        epoch_comm_interf_power = []
        epoch_radar_interf_power = []
        epoch_noise_power = []

        epoch_sensing_violation_rates = []
        epoch_sensing_channel_failure_rates = []

        epoch_comm_tx_power = []
        epoch_radar_tx_power = []
        epoch_total_tx_power = []
        epoch_tx_power_penalties = []
        epoch_tx_power_violation_rates = []

        # ================================
        # Training batches
        # ================================
        for _ in range(N_BATCHE):

            trajectory_ids = np.random.choice(train_trajectories, size=ST_BATCH_SIZE, replace=True)
            channel_ids = np.random.choice(train_channels, size=ST_BATCH_SIZE, replace=False)

            valid = channel_ids + LT_BLOCK_SIZE <= train_channels
            trajectory_ids = trajectory_ids[valid]
            channel_ids = channel_ids[valid]
            B = len(channel_ids)

            # ThetaNet只看long-term block start-slot CSI
            h_dk_hat = torch.as_tensor(train_dataset["h_dk"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
            G_hat = torch.as_tensor(train_dataset["G"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)

            block_ids = channel_ids[:, None] + np.arange(LT_BLOCK_SIZE)[None, :]

            h_dk_block = torch.as_tensor(train_dataset["h_dk"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)
            h_rk_block = torch.as_tensor(train_dataset["h_rk"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)
            G_block = torch.as_tensor(train_dataset["G"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)
            g_dt_block = torch.as_tensor(train_dataset["g_dt"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)

            h_dk_block = h_dk_block.reshape(B * LT_BLOCK_SIZE, TX_ANT, UAV_COMM)
            h_rk_block = h_rk_block.reshape(B * LT_BLOCK_SIZE, RIS_UNIT, UAV_COMM)
            G_block = G_block.reshape(B * LT_BLOCK_SIZE, RIS_UNIT, TX_ANT)
            g_dt_block = g_dt_block.reshape(B * LT_BLOCK_SIZE, TX_ANT, 1)

            optimizer.zero_grad(set_to_none=True)

            # Frozen ThetaNet：同一long-term block內固定theta
            with torch.no_grad():
                theta = theta_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat)
                theta_block = theta[:, None, :].expand(-1, LT_BLOCK_SIZE, -1).reshape(B * LT_BLOCK_SIZE, RIS_UNIT)
                H_eff_H = comm_net.compute_effective_channel(h_dk_block, h_rk_block, G_block, theta_block)

            # Nominal CSI決定beamformer
            W_C_raw = comm_net(H_eff_H, g_dt_block)
            W_R_raw = radar_net(H_eff_H, g_dt_block)

            # ================================
            # TX power penalty
            # ================================
            W_C_power = torch.sum(torch.abs(W_C_raw) ** 2, dim=(1, 2)).real
            W_R_power = torch.sum(torch.abs(W_R_raw) ** 2, dim=(1, 2)).real
            total_tx_power = W_C_power + W_R_power

            total_tx_power_ratio = total_tx_power / TRANSMIT_POWER_TOTAL
            tx_power_excess_ratio = torch.relu(total_tx_power_ratio - 1.0)
            tx_power_penalty = torch.mean(tx_power_excess_ratio ** 2)

            tx_power_violation_rate = torch.mean(
                (total_tx_power > TRANSMIT_POWER_TOTAL).float()
            )

            # ================================
            # Uncertainty injection
            # ================================
            S = INJECTION_SAMPLES
            B_short = h_dk_block.shape[0]
            M = TX_ANT
            N = RIS_UNIT
            K = UAV_COMM

            h_dk_rep = h_dk_block.unsqueeze(0).expand(S, B_short, M, K)
            h_rk_rep = h_rk_block.unsqueeze(0).expand(S, B_short, N, K)
            G_rep = G_block.unsqueeze(0).expand(S, B_short, N, M)
            g_dt_rep = g_dt_block.unsqueeze(0).expand(S, B_short, M, 1)

            h_dk_power = torch.mean(torch.abs(h_dk_rep) ** 2, dim=(2, 3), keepdim=True).real
            h_rk_power = torch.mean(torch.abs(h_rk_rep) ** 2, dim=(2, 3), keepdim=True).real
            G_power = torch.mean(torch.abs(G_rep) ** 2, dim=(2, 3), keepdim=True).real
            g_dt_power = torch.mean(torch.abs(g_dt_rep) ** 2, dim=(2, 3), keepdim=True).real

            h_dk_inj = h_dk_rep + torch.sqrt(INJECTION_VARIANCE * h_dk_power) * complex_awgn(h_dk_rep.shape, 1.0, DEVICE, h_dk_rep.dtype)
            h_rk_inj = h_rk_rep + torch.sqrt(INJECTION_VARIANCE * h_rk_power) * complex_awgn(h_rk_rep.shape, 1.0, DEVICE, h_rk_rep.dtype)
            G_inj = G_rep + torch.sqrt(INJECTION_VARIANCE * G_power) * complex_awgn(G_rep.shape, 1.0, DEVICE, G_rep.dtype)
            g_dt_inj = g_dt_rep + torch.sqrt(INJECTION_VARIANCE * g_dt_power) * complex_awgn(g_dt_rep.shape, 1.0, DEVICE, g_dt_rep.dtype)

            # 每個slot的S筆injection共用nominal CSI產生的theta與beamformer
            theta_rep = theta_block.unsqueeze(0).expand(S, B_short, N)
            W_C_rep = W_C_raw.unsqueeze(0).expand(S, B_short, M, K)
            W_R_rep = W_R_raw.unsqueeze(0).expand(S, B_short, M, RADAR_STREAMS)

            h_dk_flat = h_dk_inj.reshape(S * B_short, M, K)
            h_rk_flat = h_rk_inj.reshape(S * B_short, N, K)
            G_flat = G_inj.reshape(S * B_short, N, M)
            g_dt_flat = g_dt_inj.reshape(S * B_short, M, 1)

            theta_flat = theta_rep.reshape(S * B_short, N)
            W_C_flat = W_C_rep.reshape(S * B_short, M, K)
            W_R_flat = W_R_rep.reshape(S * B_short, M, RADAR_STREAMS)

            # injected true channels重新組成effective channel
            H_eff_H_flat = comm_net.compute_effective_channel(
                h_dk_flat,
                h_rk_flat,
                G_flat,
                theta_flat,
            )

            metrics_flat = comm_net.compute_isac_batch_performance(
                H_eff_H_flat,
                g_dt_flat,
                W_C_flat,
                W_R_flat,
            )

            # ================================
            # Restore injection dimensions
            # ================================
            robust_allUE_rate = metrics_flat["rate"].reshape(S, B, LT_BLOCK_SIZE, K)
            robust_target_snr = metrics_flat["target_snr"].reshape(S, B, LT_BLOCK_SIZE)
            robust_target_snr_db = metrics_flat["target_snr_db"].reshape(S, B, LT_BLOCK_SIZE)

            robust_signal = metrics_flat["signal"].reshape(S, B, LT_BLOCK_SIZE, K)
            robust_comm_interf = metrics_flat["comm_interf"].reshape(S, B, LT_BLOCK_SIZE, K)
            robust_radar_interf = metrics_flat["radar_interf"].reshape(S, B, LT_BLOCK_SIZE, K)
            robust_noise = metrics_flat["noise"]

            # ================================
            # Robust rate:
            # 每slot先對K取worst UE -> 對S取Q0.05 -> 對L與B平均
            # ================================
            worstUE_rate = torch.min(robust_allUE_rate, dim=3).values
            robust_per_slot = torch.quantile(worstUE_rate, OUTAGE_QUANTILE, dim=0)
            robust_per_channel = torch.mean(robust_per_slot, dim=1)
            robust = torch.mean(robust_per_channel)

            # injected channel mean SINR components，只供log/plot
            robust_signal_power = torch.mean(robust_signal, dim=(0, 1, 2))
            robust_comm_interf_power = torch.mean(robust_comm_interf, dim=(0, 1, 2))
            robust_radar_interf_power = torch.mean(robust_radar_interf, dim=(0, 1, 2))

            # ================================
            # Robust sensing
            # ================================
            robust_target_snr_mean = torch.mean(robust_target_snr)
            robust_target_snr_mean_db = 10.0 * torch.log10(
                robust_target_snr_mean.clamp_min(1e-12)
            )

            # 每個slot對S筆injection取target SNR Q10%
            target_snr_q10_db = torch.quantile(
                robust_target_snr_db,
                TAR_OUTAGE_QUANTILE,
                dim=0,
            )

            sensing_deficit_db = torch.relu(
                SENSING_SNR_THRESHOLD_DB - target_snr_q10_db
            )
            sensing_penalty = torch.mean(sensing_deficit_db)

            sensing_violation_per_channel = torch.mean(
                (robust_target_snr_db < SENSING_SNR_THRESHOLD_DB).float(),
                dim=0,
            )

            sensing_violation_rate = torch.mean(
                sensing_violation_per_channel
            )

            sensing_channel_failure_rate = torch.mean(
                (
                    sensing_violation_per_channel
                    > TAR_OUTAGE_QUANTILE
                ).float()
            )

            # ================================
            # Original ROB objective
            # ================================
            loss = (
                -robust
                + ROB_SENSING_LOSS_WEIGHT * sensing_penalty
                + ROB_TX_POWER_LOSS_WEIGHT * tx_power_penalty
            )

            loss.backward()
            optimizer.step()

            # ================================
            # Accumulate training logs
            # ================================
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_Robust.append(float(robust.detach().cpu()))
            epoch_target_snr.append(float(robust_target_snr_mean.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))

            epoch_sensing_violation_rates.append(
                float(sensing_violation_rate.detach().cpu())
            )
            epoch_sensing_channel_failure_rates.append(
                float(sensing_channel_failure_rate.detach().cpu())
            )

            epoch_signal_power.append(
                robust_signal_power.detach().cpu().numpy()
            )
            epoch_comm_interf_power.append(
                robust_comm_interf_power.detach().cpu().numpy()
            )
            epoch_radar_interf_power.append(
                robust_radar_interf_power.detach().cpu().numpy()
            )
            epoch_noise_power.append(
                float(robust_noise.detach().cpu())
            )

            epoch_comm_tx_power.append(
                float(torch.mean(W_C_power).detach().cpu())
            )
            epoch_radar_tx_power.append(
                float(torch.mean(W_R_power).detach().cpu())
            )
            epoch_total_tx_power.append(
                float(torch.mean(total_tx_power).detach().cpu())
            )
            epoch_tx_power_penalties.append(
                float(tx_power_penalty.detach().cpu())
            )
            epoch_tx_power_violation_rates.append(
                float(tx_power_violation_rate.detach().cpu())
            )

        # ================================
        # Train epoch logs
        # ================================
        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(
            max(train_target_snr_mean, 1e-12)
        )

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "robust": float(np.mean(epoch_Robust)),
            "target_snr_db": train_target_snr_mean_db,
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
            "sensing_violation_rate": float(np.mean(epoch_sensing_violation_rates)),
            "sensing_channel_failure_rate": float(np.mean(epoch_sensing_channel_failure_rates)),

            "signal_power": np.mean(np.stack(epoch_signal_power, axis=0), axis=0),
            "comm_interf_power": np.mean(np.stack(epoch_comm_interf_power, axis=0), axis=0),
            "radar_interf_power": np.mean(np.stack(epoch_radar_interf_power, axis=0), axis=0),
            "noise_power": float(np.mean(epoch_noise_power)),

            "comm_tx_power": float(np.mean(epoch_comm_tx_power)),
            "radar_tx_power": float(np.mean(epoch_radar_tx_power)),
            "total_tx_power": float(np.mean(epoch_total_tx_power)),
            "tx_power_penalty": float(np.mean(epoch_tx_power_penalties)),
            "tx_power_violation_rate": float(np.mean(epoch_tx_power_violation_rates)),
        }

        # ================================
        # Validation
        # 固定 0,L,2L,... block starts
        # 每epoch重新產生validation injection errors
        # 使用chunk避免GPU OOM
        # ================================
        comm_net.eval()
        radar_net.eval()

        with torch.no_grad():

            val_h_dk = torch.as_tensor(
                val_dataset["h_dk"][val_trajectory_ids, val_channel_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )
            val_h_rk = torch.as_tensor(
                val_dataset["h_rk"][val_trajectory_ids, val_channel_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )
            val_G = torch.as_tensor(
                val_dataset["G"][val_trajectory_ids, val_channel_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )
            val_g_dt = torch.as_tensor(
                val_dataset["g_dt"][val_trajectory_ids, val_channel_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )

            # Frozen ThetaNet只看block-start CSI
            val_theta = theta_net(
                val_h_dk,
                val_h_rk,
                val_G,
                val_g_dt,
            )

            val_block_ids = (
                val_channel_ids[:, None]
                + np.arange(LT_BLOCK_SIZE)[None, :]
            )

            val_h_dk_block = torch.as_tensor(
                val_dataset["h_dk"][val_trajectory_ids[:, None], val_block_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )
            val_h_rk_block = torch.as_tensor(
                val_dataset["h_rk"][val_trajectory_ids[:, None], val_block_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )
            val_G_block = torch.as_tensor(
                val_dataset["G"][val_trajectory_ids[:, None], val_block_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )
            val_g_dt_block = torch.as_tensor(
                val_dataset["g_dt"][val_trajectory_ids[:, None], val_block_ids],
                dtype=torch.complex64,
                device=DEVICE,
            )

            val_h_dk_block = val_h_dk_block.reshape(
                B_val * LT_BLOCK_SIZE,
                TX_ANT,
                UAV_COMM,
            )
            val_h_rk_block = val_h_rk_block.reshape(
                B_val * LT_BLOCK_SIZE,
                RIS_UNIT,
                UAV_COMM,
            )
            val_G_block = val_G_block.reshape(
                B_val * LT_BLOCK_SIZE,
                RIS_UNIT,
                TX_ANT,
            )
            val_g_dt_block = val_g_dt_block.reshape(
                B_val * LT_BLOCK_SIZE,
                TX_ANT,
                1,
            )

            val_theta_block = (
                val_theta[:, None, :]
                .expand(-1, LT_BLOCK_SIZE, -1)
                .reshape(B_val * LT_BLOCK_SIZE, RIS_UNIT)
            )

            # nominal channel產生beamformer
            val_H_eff_H = comm_net.compute_effective_channel(
                val_h_dk_block,
                val_h_rk_block,
                val_G_block,
                val_theta_block,
            )

            val_W_C_raw = comm_net(
                val_H_eff_H,
                val_g_dt_block,
            )
            val_W_R_raw = radar_net(
                val_H_eff_H,
                val_g_dt_block,
            )

            # validation nominal beamformer power
            val_W_C_power = torch.sum(
                torch.abs(val_W_C_raw) ** 2,
                dim=(1, 2),
            ).real
            val_W_R_power = torch.sum(
                torch.abs(val_W_R_raw) ** 2,
                dim=(1, 2),
            ).real
            val_total_tx_power = (
                val_W_C_power
                + val_W_R_power
            )

            val_total_tx_power_ratio = (
                val_total_tx_power
                / TRANSMIT_POWER_TOTAL
            )
            val_tx_power_excess_ratio = torch.relu(
                val_total_tx_power_ratio - 1.0
            )
            val_tx_power_penalty = torch.mean(
                val_tx_power_excess_ratio ** 2
            )

            val_tx_power_violation_rate = torch.mean(
                (
                    val_total_tx_power
                    > TRANSMIT_POWER_TOTAL
                ).float()
            )

            val_S = INJECTION_SAMPLES
            val_M = TX_ANT
            val_N = RIS_UNIT
            val_K = UAV_COMM

            # ================================
            # Validation injection chunk
            # ================================
            val_robust_sum = torch.zeros(
                (),
                dtype=torch.float32,
                device=DEVICE,
            )
            val_robust_signal_sum = torch.zeros(
                val_K,
                dtype=torch.float32,
                device=DEVICE,
            )
            val_robust_comm_interf_sum = torch.zeros(
                val_K,
                dtype=torch.float32,
                device=DEVICE,
            )
            val_robust_radar_interf_sum = torch.zeros(
                val_K,
                dtype=torch.float32,
                device=DEVICE,
            )
            val_robust_target_snr_sum = torch.zeros(
                (),
                dtype=torch.float32,
                device=DEVICE,
            )
            val_sensing_penalty_sum = torch.zeros(
                (),
                dtype=torch.float32,
                device=DEVICE,
            )
            val_sensing_violation_sum = torch.zeros(
                (),
                dtype=torch.float32,
                device=DEVICE,
            )
            val_sensing_channel_failure_sum = torch.zeros(
                (),
                dtype=torch.float32,
                device=DEVICE,
            )
            val_robust_noise = None

            for val_chunk_start in range(
                0,
                B_val,
                VAL_CHUNK_SIZE,
            ):

                val_chunk_end = min(
                    val_chunk_start + VAL_CHUNK_SIZE,
                    B_val,
                )
                val_B_chunk = (
                    val_chunk_end
                    - val_chunk_start
                )

                val_short_start = (
                    val_chunk_start
                    * LT_BLOCK_SIZE
                )
                val_short_end = (
                    val_chunk_end
                    * LT_BLOCK_SIZE
                )
                val_B_short_chunk = (
                    val_B_chunk
                    * LT_BLOCK_SIZE
                )

                val_h_dk_block_chunk = val_h_dk_block[
                    val_short_start:val_short_end
                ]
                val_h_rk_block_chunk = val_h_rk_block[
                    val_short_start:val_short_end
                ]
                val_G_block_chunk = val_G_block[
                    val_short_start:val_short_end
                ]
                val_g_dt_block_chunk = val_g_dt_block[
                    val_short_start:val_short_end
                ]

                val_theta_block_chunk = val_theta_block[
                    val_short_start:val_short_end
                ]
                val_W_C_raw_chunk = val_W_C_raw[
                    val_short_start:val_short_end
                ]
                val_W_R_raw_chunk = val_W_R_raw[
                    val_short_start:val_short_end
                ]

                val_h_dk_rep = (
                    val_h_dk_block_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_M,
                        val_K,
                    )
                )
                val_h_rk_rep = (
                    val_h_rk_block_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_N,
                        val_K,
                    )
                )
                val_G_rep = (
                    val_G_block_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_N,
                        val_M,
                    )
                )
                val_g_dt_rep = (
                    val_g_dt_block_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_M,
                        1,
                    )
                )

                val_h_dk_power = torch.mean(
                    torch.abs(val_h_dk_rep) ** 2,
                    dim=(2, 3),
                    keepdim=True,
                ).real
                val_h_rk_power = torch.mean(
                    torch.abs(val_h_rk_rep) ** 2,
                    dim=(2, 3),
                    keepdim=True,
                ).real
                val_G_power = torch.mean(
                    torch.abs(val_G_rep) ** 2,
                    dim=(2, 3),
                    keepdim=True,
                ).real
                val_g_dt_power = torch.mean(
                    torch.abs(val_g_dt_rep) ** 2,
                    dim=(2, 3),
                    keepdim=True,
                ).real

                val_h_dk_inj = (
                    val_h_dk_rep
                    + torch.sqrt(
                        INJECTION_VARIANCE
                        * val_h_dk_power
                    )
                    * complex_awgn(
                        val_h_dk_rep.shape,
                        1.0,
                        DEVICE,
                        val_h_dk_rep.dtype,
                    )
                )
                val_h_rk_inj = (
                    val_h_rk_rep
                    + torch.sqrt(
                        INJECTION_VARIANCE
                        * val_h_rk_power
                    )
                    * complex_awgn(
                        val_h_rk_rep.shape,
                        1.0,
                        DEVICE,
                        val_h_rk_rep.dtype,
                    )
                )
                val_G_inj = (
                    val_G_rep
                    + torch.sqrt(
                        INJECTION_VARIANCE
                        * val_G_power
                    )
                    * complex_awgn(
                        val_G_rep.shape,
                        1.0,
                        DEVICE,
                        val_G_rep.dtype,
                    )
                )
                val_g_dt_inj = (
                    val_g_dt_rep
                    + torch.sqrt(
                        INJECTION_VARIANCE
                        * val_g_dt_power
                    )
                    * complex_awgn(
                        val_g_dt_rep.shape,
                        1.0,
                        DEVICE,
                        val_g_dt_rep.dtype,
                    )
                )

                # 每個slot的injection共用nominal theta與beamformer
                val_theta_rep = (
                    val_theta_block_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_N,
                    )
                )
                val_W_C_rep = (
                    val_W_C_raw_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_M,
                        val_K,
                    )
                )
                val_W_R_rep = (
                    val_W_R_raw_chunk
                    .unsqueeze(0)
                    .expand(
                        val_S,
                        val_B_short_chunk,
                        val_M,
                        RADAR_STREAMS,
                    )
                )

                val_h_dk_flat = val_h_dk_inj.reshape(
                    val_S * val_B_short_chunk,
                    val_M,
                    val_K,
                )
                val_h_rk_flat = val_h_rk_inj.reshape(
                    val_S * val_B_short_chunk,
                    val_N,
                    val_K,
                )
                val_G_flat = val_G_inj.reshape(
                    val_S * val_B_short_chunk,
                    val_N,
                    val_M,
                )
                val_g_dt_flat = val_g_dt_inj.reshape(
                    val_S * val_B_short_chunk,
                    val_M,
                    1,
                )

                val_theta_flat = val_theta_rep.reshape(
                    val_S * val_B_short_chunk,
                    val_N,
                )
                val_W_C_flat = val_W_C_rep.reshape(
                    val_S * val_B_short_chunk,
                    val_M,
                    val_K,
                )
                val_W_R_flat = val_W_R_rep.reshape(
                    val_S * val_B_short_chunk,
                    val_M,
                    RADAR_STREAMS,
                )

                # injected true channels
                val_H_eff_H_flat = (
                    comm_net.compute_effective_channel(
                        val_h_dk_flat,
                        val_h_rk_flat,
                        val_G_flat,
                        val_theta_flat,
                    )
                )

                val_metrics_flat = (
                    comm_net.compute_isac_batch_performance(
                        val_H_eff_H_flat,
                        val_g_dt_flat,
                        val_W_C_flat,
                        val_W_R_flat,
                    )
                )

                val_robust_allUE_rate = val_metrics_flat[
                    "rate"
                ].reshape(
                    val_S,
                    val_B_chunk,
                    LT_BLOCK_SIZE,
                    val_K,
                )
                val_robust_target_snr = val_metrics_flat[
                    "target_snr"
                ].reshape(
                    val_S,
                    val_B_chunk,
                    LT_BLOCK_SIZE,
                )
                val_robust_target_snr_db = val_metrics_flat[
                    "target_snr_db"
                ].reshape(
                    val_S,
                    val_B_chunk,
                    LT_BLOCK_SIZE,
                )

                val_robust_signal = val_metrics_flat[
                    "signal"
                ].reshape(
                    val_S,
                    val_B_chunk,
                    LT_BLOCK_SIZE,
                    val_K,
                )
                val_robust_comm_interf = val_metrics_flat[
                    "comm_interf"
                ].reshape(
                    val_S,
                    val_B_chunk,
                    LT_BLOCK_SIZE,
                    val_K,
                )
                val_robust_radar_interf = val_metrics_flat[
                    "radar_interf"
                ].reshape(
                    val_S,
                    val_B_chunk,
                    LT_BLOCK_SIZE,
                    val_K,
                )
                val_robust_noise = val_metrics_flat[
                    "noise"
                ]

                # ROB rate
                val_worstUE_rate = torch.min(
                    val_robust_allUE_rate,
                    dim=3,
                ).values
                val_robust_per_slot = torch.quantile(
                    val_worstUE_rate,
                    OUTAGE_QUANTILE,
                    dim=0,
                )
                val_robust_sum += torch.sum(
                    val_robust_per_slot
                )

                # SINR components
                val_robust_signal_sum += torch.sum(
                    val_robust_signal,
                    dim=(0, 1, 2),
                )
                val_robust_comm_interf_sum += torch.sum(
                    val_robust_comm_interf,
                    dim=(0, 1, 2),
                )
                val_robust_radar_interf_sum += torch.sum(
                    val_robust_radar_interf,
                    dim=(0, 1, 2),
                )

                # sensing
                val_robust_target_snr_sum += torch.sum(
                    val_robust_target_snr
                )

                val_target_snr_q10_db = torch.quantile(
                    val_robust_target_snr_db,
                    TAR_OUTAGE_QUANTILE,
                    dim=0,
                )

                val_sensing_deficit_db = torch.relu(
                    SENSING_SNR_THRESHOLD_DB
                    - val_target_snr_q10_db
                )
                val_sensing_penalty_sum += torch.sum(
                    val_sensing_deficit_db
                )

                val_sensing_violation_per_channel = torch.mean(
                    (
                        val_robust_target_snr_db
                        < SENSING_SNR_THRESHOLD_DB
                    ).float(),
                    dim=0,
                )

                val_sensing_violation_sum += torch.sum(
                    val_sensing_violation_per_channel
                )

                val_sensing_channel_failure_sum += torch.sum(
                    (
                        val_sensing_violation_per_channel
                        > TAR_OUTAGE_QUANTILE
                    ).float()
                )

                # 釋放大型injection tensors
                del val_h_dk_inj, val_h_rk_inj, val_G_inj, val_g_dt_inj
                del val_h_dk_flat, val_h_rk_flat, val_G_flat, val_g_dt_flat
                del val_theta_flat, val_W_C_flat, val_W_R_flat
                del val_H_eff_H_flat, val_metrics_flat
                del val_robust_allUE_rate, val_robust_target_snr, val_robust_target_snr_db
                del val_robust_signal, val_robust_comm_interf, val_robust_radar_interf
                del val_worstUE_rate, val_robust_per_slot
                del val_target_snr_q10_db, val_sensing_deficit_db
                del val_sensing_violation_per_channel

            # ================================
            # Combine validation chunks
            # ================================
            val_robust = (
                val_robust_sum
                / (B_val * LT_BLOCK_SIZE)
            )

            val_metric_count = (
                val_S
                * B_val
                * LT_BLOCK_SIZE
            )
            val_robust_signal_power = (
                val_robust_signal_sum
                / val_metric_count
            )
            val_robust_comm_interf_power = (
                val_robust_comm_interf_sum
                / val_metric_count
            )
            val_robust_radar_interf_power = (
                val_robust_radar_interf_sum
                / val_metric_count
            )

            val_robust_target_snr_mean = (
                val_robust_target_snr_sum
                / val_metric_count
            )
            val_robust_target_snr_mean_db = (
                10.0
                * torch.log10(
                    val_robust_target_snr_mean
                    .clamp_min(1e-12)
                )
            )

            val_short_slot_count = (
                B_val
                * LT_BLOCK_SIZE
            )
            val_sensing_penalty = (
                val_sensing_penalty_sum
                / val_short_slot_count
            )
            val_sensing_violation_rate = (
                val_sensing_violation_sum
                / val_short_slot_count
            )
            val_sensing_channel_failure_rate = (
                val_sensing_channel_failure_sum
                / val_short_slot_count
            )

            # Original ROB validation objective
            val_loss = (
                -val_robust
                + ROB_SENSING_LOSS_WEIGHT
                * val_sensing_penalty
                + ROB_TX_POWER_LOSS_WEIGHT
                * val_tx_power_penalty
            )

        # ================================
        # Validation logs
        # ================================
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

        # ================================
        # Save curves
        # ================================
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
                key: np.asarray(
                    value,
                    dtype=np.float32,
                )
                for key, value
                in curves.items()
            },
        )

        # ================================
        # Early stopping
        # ================================
        if (
            best_val_loss
            - val_logs["loss"]
            > EARLY_STOP_EPS
        ):
            best_val_loss = val_logs["loss"]
            best_val_worstUE_q05 = val_logs["robust"]
            best_valtarget_snr_db = val_logs["target_snr_db"]
            best_val_epoch = epoch + 1

            patience_counter = 0

            comm_net.save_model(
                rob_comm_ckpt,
                verbose=False,
            )
            radar_net.save_model(
                rob_radar_ckpt,
                verbose=False,
            )

        else:
            patience_counter += 1

        train_comm_power_ratio = (
            train_logs["comm_tx_power"]
            / TRANSMIT_POWER_TOTAL
        )
        train_radar_power_ratio = (
            train_logs["radar_tx_power"]
            / TRANSMIT_POWER_TOTAL
        )
        train_total_power_ratio = (
            train_logs["total_tx_power"]
            / TRANSMIT_POWER_TOTAL
        )

        val_comm_power_ratio = (
            val_logs["comm_tx_power"]
            / TRANSMIT_POWER_TOTAL
        )
        val_radar_power_ratio = (
            val_logs["radar_tx_power"]
            / TRANSMIT_POWER_TOTAL
        )
        val_total_power_ratio = (
            val_logs["total_tx_power"]
            / TRANSMIT_POWER_TOTAL
        )

        # ================================
        # Epoch log
        # ================================
        print(
            f"\n[Epoch {epoch + 1:03d}/{ROB_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "
            f"TrainRobust={train_logs['robust']:.4f} "
            f"ValRobust={val_logs['robust']:.4f} | "
            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB"
            f"\nTrainSenseViolation={100.0 * train_logs['sensing_violation_rate']:.2f}% "
            f"ValSenseViolation={100.0 * val_logs['sensing_violation_rate']:.2f}%"
            f"\nTrainPower W_C_raw={train_logs['comm_tx_power']:.6f} W "
            f"W_R_raw={train_logs['radar_tx_power']:.6f} W | "
            f"P_C/P_T:P_R/P_T=({train_comm_power_ratio:.4f}:{train_radar_power_ratio:.4f}) "
            f"Total/P_T={train_total_power_ratio:.4f} "
            f"Violation={100.0 * train_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty={train_logs['tx_power_penalty']:.6e}"
            f"\nValPower   W_C_raw={val_logs['comm_tx_power']:.6f} W "
            f"W_R_raw={val_logs['radar_tx_power']:.6f} W | "
            f"P_C/P_T:P_R/P_T=({val_comm_power_ratio:.4f}:{val_radar_power_ratio:.4f}) "
            f"Total/P_T={val_total_power_ratio:.4f} "
            f"Violation={100.0 * val_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty={val_logs['tx_power_penalty']:.6e}"
        )

        if patience_counter >= PATIENCE:
            print(
                f"\n[EARLY STOP] "
                f"patience_counter={patience_counter}, "
                f"PATIENCE={PATIENCE}, "
                f"best_val_loss={best_val_loss:.6f}"
            )
            break

    # ================================
    # Plot
    # ================================
    plot_rob_curves(
        rob_curve_path,
        rob_curve_dir,
    )

    # ================================
    # Final summary
    # ================================
    print("\n" + "=" * 100)
    print("[COMM3 PRETRAINED COMMNET + NS-MRT JOINT ROB FINE-TUNING FINISHED]")
    print("=" * 100)
    print(f"Train channels                  : {train_channels}")
    print(f"Val channels                    : {val_channels}")
    print(f"ROB_EPOCHS                      : {ROB_EPOCHS}")
    print(f"N_BATCHE                        : {N_BATCHE}")
    print(f"ST_BATCH_SIZE                   : {ST_BATCH_SIZE}")
    print(f"ROB learning rate               : {ROB_LEARNING_RATE}")
    print(f"Sensing SNR threshold           : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight          : {ROB_SENSING_LOSS_WEIGHT}")
    print(f"TX power penalty weight         : {ROB_TX_POWER_LOSS_WEIGHT}")
    print(f"Rate outage quantile            : {OUTAGE_QUANTILE}")
    print(f"Sensing outage quantile         : {TAR_OUTAGE_QUANTILE}")
    print(f"CommNet initialization          : {comm_pretrain_ckpt}")
    print(f"RadarNet initialization         : {radar_pretrain_ckpt}")
    print("-")
    print(f"Best validation epoch           : {best_val_epoch}")
    print(f"Best validation loss            : {best_val_loss:.6f}")
    print(f"Robust rate at best val loss    : {best_val_worstUE_q05:.6f}")
    print(f"Target SNR at best val loss     : {best_valtarget_snr_db:.3f} dB")
    print("-")
    print(f"CommNet output checkpoint       : {rob_comm_ckpt}")
    print(f"RadarNet output checkpoint      : {rob_radar_ckpt}")
    print(f"Curves                          : {rob_curve_path}")
    print(f"Plots                           : {rob_curve_dir}")
    print("=" * 100)
