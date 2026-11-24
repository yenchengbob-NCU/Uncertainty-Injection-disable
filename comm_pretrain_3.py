# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from two_timescale_NN import CommNet, ThetaNet


# ================================
# CommNet communication-only task pretrain settings
# ================================
# 直接沿用 REG 設定，避免另外改變 training condition。
COMM_PRETRAIN_EPOCHS = REG_EPOCHS
COMM_PRETRAIN_LEARNING_RATE = REG_LEARNING_RATE
COMM_PRETRAIN_BATCHES = N_BATCHE
COMM_PRETRAIN_BATCH_CHANNELS = ST_BATCH_SIZE

# Early stopping 與 REG 相同
EARLY_STOP_EPS = 1e-6
PATIENCE = 30


# ================================
# Reproducibility
# ================================
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ================================
# Helpers
# ================================
def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "[" + " ".join([f"{float(v):.{precision}e}" for v in x]) + "]"


def moving_average(x, window=20):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    window = int(window)

    if len(x) < window:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    kernel = np.ones(window, dtype=np.float64) / window
    y_ma = np.convolve(x, kernel, mode="valid")
    ma_epochs = np.arange(window, len(x) + 1)

    return ma_epochs, y_ma


# ================================
# Pure communication metrics
# ================================
def compute_comm_metrics(H_eff_H, W_C):
    """
    H_eff_H : (S,K,M)
    W_C     : (S,M,K)

    received = H_eff_H @ W_C : (S,K,K)

    對第 k 個 UE：
        desired signal:
            S_k = |h_k^H w_k|^2

        communication interference:
            I_C,k = sum_{j != k} |h_k^H w_j|^2

        SINR:
            gamma_k = S_k / (I_C,k + noise)

    注意：
        - 沒有 RadarNet
        - 沒有 W_R
        - 沒有 radar interference
        - 沒有 sensing metric
        - 沒有 explicit I_C loss
    """
    received = torch.bmm(H_eff_H, W_C)
    received_power = torch.abs(received) ** 2

    signal_power = torch.diagonal(received_power, dim1=1, dim2=2)
    comm_interf_power = torch.sum(received_power, dim=2) - signal_power

    noise = torch.as_tensor(NOISE_POWER, dtype=torch.float32, device=DEVICE)

    sinr = signal_power / (comm_interf_power + noise)
    rate = torch.log2(1.0 + sinr)

    return {
        "signal": signal_power,
        "comm_interf": comm_interf_power,
        "sinr": sinr,
        "rate": rate,
        "noise": noise,
    }


def compute_pretrain_objective(H_eff_H, W_C, B):
    """
    Pure communication-task CommNet pretraining.

    Loss:
        L = -R_nominal + lambda_P * L_P

    其中：
        R_nominal
            = 對每個 slot 先取 worst UE rate
            -> 對 LT block 內 L 個 slot 平均
            -> 對 B 個 blocks 平均

        P_C
            = ||W_C||_F^2

        L_P
            = mean[ ReLU(P_C / P_T - 1)^2 ]

    明確沒有：
        - RZF teacher
        - RZF complex MSE
        - explicit communication-interference loss
        - 0.63 W target-power loss
        - hard power normalization
        - RadarNet
        - W_R
        - sensing loss
    """
    metrics = compute_comm_metrics(H_eff_H, W_C)

    # (B*L,K) -> (B,L,K)
    all_rate = metrics["rate"].reshape(B, LT_BLOCK_SIZE, UAV_COMM)

    # 每個 slot 先找 worst UE
    worst_ue_per_slot = torch.min(all_rate, dim=2).values          # (B,L)

    # 每個 long-term block 對 L 個 slots 平均
    per_block_rate = torch.mean(worst_ue_per_slot, dim=1)         # (B,)

    # 對 B 個 blocks 平均
    nominal_rate = torch.mean(per_block_rate)                     # scalar

    # 每個 short-term slot 的 CommNet output power
    sample_power = torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)).real
    mean_power = torch.mean(sample_power)

    # 原 REG total-TX-power soft penalty
    power_ratio = sample_power / TRANSMIT_POWER_TOTAL
    power_excess_ratio = torch.relu(power_ratio - 1.0)
    tx_power_penalty = torch.mean(power_excess_ratio ** 2)

    # 只供 log，不進額外 loss
    tx_power_violation_rate = torch.mean(
        (sample_power > TRANSMIT_POWER_TOTAL).float()
    )

    # 唯一 objective：
    # true communication rate + 原 REG TX-power penalty
    total_loss = (
        -nominal_rate
        + REG_TX_POWER_LOSS_WEIGHT * tx_power_penalty
    )

    return (
        total_loss,
        nominal_rate,
        tx_power_penalty,
        tx_power_violation_rate,
        mean_power,
        metrics,
    )


# ================================
# Plot curves
# ================================
def plot_curves(curve_path, out_dir, ma_window=20):
    data = np.load(curve_path)
    os.makedirs(out_dir, exist_ok=True)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    train_nominal = data["train_nominal"]
    val_nominal = data["val_nominal"]

    train_power = data["train_W_C_power"]
    val_power = data["val_W_C_power"]

    train_comm_interf_power = data["train_comm_interf_power"]
    val_comm_interf_power = data["val_comm_interf_power"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig. 1: Loss
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

    e1, y1 = moving_average(train_loss, ma_window)
    e2, y2 = moving_average(val_loss, ma_window)

    if len(y1) > 0:
        plt.plot(
            e1,
            y1,
            label=f"Train Loss MA({ma_window})",
            linewidth=2.5,
        )

    if len(y2) > 0:
        plt.plot(
            e2,
            y2,
            label=f"Validation Loss MA({ma_window})",
            linewidth=2.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fig. 1: CommNet Pure Communication Pretrain Loss")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig1_path = os.path.join(out_dir, "Fig1_loss_curve.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig1_path}")

    # ================================
    # Fig. 2: Worst-UE rate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_nominal,
        label="Train Worst-UE Rate",
        alpha=0.35,
        linewidth=1.0,
    )
    plt.plot(
        epochs,
        val_nominal,
        label="Validation Worst-UE Rate",
        alpha=0.35,
        linewidth=1.0,
    )

    e1, y1 = moving_average(train_nominal, ma_window)
    e2, y2 = moving_average(val_nominal, ma_window)

    if len(y1) > 0:
        plt.plot(
            e1,
            y1,
            label=f"Train Rate MA({ma_window})",
            linewidth=2.5,
        )

    if len(y2) > 0:
        plt.plot(
            e2,
            y2,
            label=f"Validation Rate MA({ma_window})",
            linewidth=2.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel("Worst-UE Rate (bps/Hz)")
    plt.title("Fig. 2: CommNet Pure Communication Worst-UE Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_nominal_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")

    # ================================
    # Fig. 3: W_C power
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        train_power,
        label="Train W_C Power",
        linewidth=1.5,
    )
    plt.plot(
        epochs,
        val_power,
        label="Validation W_C Power",
        linewidth=1.5,
    )

    plt.axhline(
        TRANSMIT_POWER_TOTAL,
        linestyle="--",
        label="Total TX Power Limit",
    )

    plt.xlabel("Epoch")
    plt.ylabel("W")
    plt.title("Fig. 3: CommNet Output Power")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig3_path = os.path.join(out_dir, "Fig3_W_C_power.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig3_path}")

    # ================================
    # Fig. 4: Communication interference
    # 每個 UE 分開畫
    # ================================
    num_users = train_comm_interf_power.shape[1]

    for ue_idx in range(num_users):
        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train_comm_interf_power[:, ue_idx],
            label="Train Communication Interference",
            linewidth=1.5,
        )
        plt.plot(
            epochs,
            val_comm_interf_power[:, ue_idx],
            label="Validation Communication Interference",
            linewidth=1.5,
        )

        plt.axhline(
            NOISE_POWER,
            linestyle="--",
            label="Noise Power",
        )

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Power")
        plt.title(f"Fig. 4: UE {ue_idx} Communication Interference")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()

        fig4_path = os.path.join(
            out_dir,
            f"Fig4_UE{ue_idx}_comm_interference.png",
        )
        plt.savefig(fig4_path, dpi=300)
        plt.close()

        print(f"[PLOT] Saved: {fig4_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":

    # ================================
    # Networks
    # ================================
    theta_net = ThetaNet().to(DEVICE)
    comm_net = CommNet().to(DEVICE)

    # 只有 CommNet 進 optimizer
    optimizer = optim.Adam(
        comm_net.parameters(),
        lr=COMM_PRETRAIN_LEARNING_RATE,
    )

    # ================================
    # Paths
    # ================================
    train_dataset_path = os.path.join(
        DATA_DIR,
        "dataset_train.npz",
    )
    val_dataset_path = os.path.join(
        DATA_DIR,
        "dataset_val.npz",
    )

    ris_only_ckpt = os.path.join(
        PRETRAIN_DIR,
        "ris_only.ckpt",
    )

    # comm_pretrain_3 使用獨立輸出檔名，避免蓋掉舊實驗
    comm_pretrain_ckpt = os.path.join(
        PRETRAIN_DIR,
        "comm_pretrain_3.ckpt",
    )
    comm_pretrain_curve_path = os.path.join(
        PRETRAIN_DIR,
        "comm_pretrain_3_curves.npz",
    )
    comm_pretrain_curve_dir = os.path.join(
        PRETRAIN_DIR,
        "comm_pretrain_3_curves",
    )

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    os.makedirs(comm_pretrain_curve_dir, exist_ok=True)

    for required_path in [
        train_dataset_path,
        val_dataset_path,
        ris_only_ckpt,
    ]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"找不到檔案: {required_path}"
            )

    # ================================
    # Load datasets
    # ================================
    with np.load(train_dataset_path) as data:
        train_dataset = {
            key: data[key]
            for key in [
                "positions",
                "h_dk",
                "h_rk",
                "G",
                "g_dt",
            ]
        }

    with np.load(val_dataset_path) as data:
        val_dataset = {
            key: data[key]
            for key in [
                "positions",
                "h_dk",
                "h_rk",
                "G",
                "g_dt",
            ]
        }

    train_trajectories, train_channels = (
        train_dataset["h_dk"].shape[:2]
    )
    val_trajectories, val_channels = (
        val_dataset["h_dk"].shape[:2]
    )

    # ================================
    # Frozen ThetaNet
    # ================================
    theta_net.load_model(
        ris_only_ckpt,
        strict=True,
        verbose=True,
    )

    theta_net.eval()

    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)

    # ================================
    # Fixed validation blocks
    # ================================
    val_channel_ids = np.arange(
        0,
        val_channels - LT_BLOCK_SIZE + 1,
        LT_BLOCK_SIZE,
    )

    val_trajectory_ids = np.repeat(
        np.arange(val_trajectories),
        len(val_channel_ids),
    )

    val_channel_ids = np.tile(
        val_channel_ids,
        val_trajectories,
    )

    B_val = len(val_channel_ids)

    with torch.no_grad():

        # ThetaNet 只看每個 LT block 的 start-slot CSI
        h_dk_val_start = torch.as_tensor(
            val_dataset["h_dk"][
                val_trajectory_ids,
                val_channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        h_rk_val_start = torch.as_tensor(
            val_dataset["h_rk"][
                val_trajectory_ids,
                val_channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        G_val_start = torch.as_tensor(
            val_dataset["G"][
                val_trajectory_ids,
                val_channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        g_dt_val_start = torch.as_tensor(
            val_dataset["g_dt"][
                val_trajectory_ids,
                val_channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        theta_val = theta_net(
            h_dk_val_start,
            h_rk_val_start,
            G_val_start,
            g_dt_val_start,
        )

        # 同一 LT block 內固定 theta
        # 每個 ST slot 使用自己的 channel
        val_block_ids = (
            val_channel_ids[:, None]
            + np.arange(LT_BLOCK_SIZE)[None, :]
        )

        h_dk_val_block = torch.as_tensor(
            val_dataset["h_dk"][
                val_trajectory_ids[:, None],
                val_block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        h_rk_val_block = torch.as_tensor(
            val_dataset["h_rk"][
                val_trajectory_ids[:, None],
                val_block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        G_val_block = torch.as_tensor(
            val_dataset["G"][
                val_trajectory_ids[:, None],
                val_block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        g_dt_val_block = torch.as_tensor(
            val_dataset["g_dt"][
                val_trajectory_ids[:, None],
                val_block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        # (B_val,L,...) -> (B_val*L,...)
        h_dk_val_block = h_dk_val_block.reshape(
            B_val * LT_BLOCK_SIZE,
            TX_ANT,
            UAV_COMM,
        )

        h_rk_val_block = h_rk_val_block.reshape(
            B_val * LT_BLOCK_SIZE,
            RIS_UNIT,
            UAV_COMM,
        )

        G_val_block = G_val_block.reshape(
            B_val * LT_BLOCK_SIZE,
            RIS_UNIT,
            TX_ANT,
        )

        g_dt_val_block = g_dt_val_block.reshape(
            B_val * LT_BLOCK_SIZE,
            TX_ANT,
            1,
        )

        theta_val_block = (
            theta_val[:, None, :]
            .expand(-1, LT_BLOCK_SIZE, -1)
            .reshape(
                B_val * LT_BLOCK_SIZE,
                RIS_UNIT,
            )
        )

        H_eff_H_val = theta_net.compute_effective_channel(
            h_dk_val_block,
            h_rk_val_block,
            G_val_block,
            theta_val_block,
        )

    # ================================
    # Training information
    # ================================
    print("\n" + "=" * 100)
    print("[COMMNET PURE COMMUNICATION-TASK PRETRAIN 3]")
    print("=" * 100)

    print(
        f"Train trajectories          : "
        f"{train_trajectories}"
    )
    print(
        f"Validation trajectories     : "
        f"{val_trajectories}"
    )
    print(
        f"Train slots / trajectory    : "
        f"{train_channels}"
    )
    print(
        f"Validation slots / traj.    : "
        f"{val_channels}"
    )
    print(
        f"LT block size               : "
        f"{LT_BLOCK_SIZE}"
    )
    print(
        f"Epochs                      : "
        f"{COMM_PRETRAIN_EPOCHS}"
    )
    print(
        f"Batches per epoch           : "
        f"{COMM_PRETRAIN_BATCHES}"
    )
    print(
        f"Blocks per batch            : "
        f"{COMM_PRETRAIN_BATCH_CHANNELS}"
    )
    print(
        f"Learning rate               : "
        f"{COMM_PRETRAIN_LEARNING_RATE}"
    )
    print(
        f"TX power limit              : "
        f"{TRANSMIT_POWER_TOTAL:.6f} W"
    )
    print(
        f"TX power penalty weight     : "
        f"{REG_TX_POWER_LOSS_WEIGHT}"
    )

    print(
        f"Frozen ThetaNet             : "
        f"{ris_only_ckpt}"
    )

    print("CommNet initialization      : RANDOM")
    print("RZF / analytical teacher    : NOT USED")
    print("RZF MSE                     : NOT USED")
    print("Explicit I_C loss           : NOT USED")
    print("0.63 W power target         : NOT USED")
    print("Hard W_C normalization      : NOT USED")
    print("RadarNet / W_R              : NOT USED")
    print("Radar interference          : NOT USED")
    print("Sensing loss                : NOT USED")

    print(
        f"Checkpoint                  : "
        f"{comm_pretrain_ckpt}"
    )

    print("=" * 100)

    # ================================
    # Curves
    # ================================
    curves = {
        "train_loss": [],
        "val_loss": [],

        "train_nominal": [],
        "val_nominal": [],

        "train_tx_power_penalty": [],
        "val_tx_power_penalty": [],

        "train_tx_power_violation_rate": [],
        "val_tx_power_violation_rate": [],

        "train_W_C_power": [],
        "val_W_C_power": [],

        "train_signal_power": [],
        "val_signal_power": [],

        "train_comm_interf_power": [],
        "val_comm_interf_power": [],

        "train_noise_power": [],
        "val_noise_power": [],
    }

    best_val_loss = float("inf")
    best_val_rate = 0.0
    best_val_epoch = 0
    patience_counter = 0

    # ================================
    # CommNet pretraining
    # ================================
    for epoch in trange(
        COMM_PRETRAIN_EPOCHS,
        desc="CommNet Pure Comm Pretrain 3",
    ):

        comm_net.train()

        epoch_loss = []
        epoch_nominal = []

        epoch_tx_power_penalty = []
        epoch_tx_power_violation_rate = []

        epoch_power = []
        epoch_signal = []
        epoch_comm_interf = []
        epoch_noise = []

        for _ in range(COMM_PRETRAIN_BATCHES):

            # 隨機抽 B 個 long-term blocks
            trajectory_ids = np.random.choice(
                train_trajectories,
                size=COMM_PRETRAIN_BATCH_CHANNELS,
                replace=True,
            )

            channel_ids = np.random.choice(
                train_channels,
                size=COMM_PRETRAIN_BATCH_CHANNELS,
                replace=False,
            )

            valid = (
                channel_ids + LT_BLOCK_SIZE
                <= train_channels
            )

            trajectory_ids = trajectory_ids[valid]
            channel_ids = channel_ids[valid]

            if len(channel_ids) == 0:
                continue

            B = len(channel_ids)

            # ================================
            # LT block start-slot CSI
            # ================================
            h_dk_start = torch.as_tensor(
                train_dataset["h_dk"][
                    trajectory_ids,
                    channel_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            h_rk_start = torch.as_tensor(
                train_dataset["h_rk"][
                    trajectory_ids,
                    channel_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            G_start = torch.as_tensor(
                train_dataset["G"][
                    trajectory_ids,
                    channel_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            g_dt_start = torch.as_tensor(
                train_dataset["g_dt"][
                    trajectory_ids,
                    channel_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            # ================================
            # LT block channels
            # ================================
            block_ids = (
                channel_ids[:, None]
                + np.arange(LT_BLOCK_SIZE)[None, :]
            )

            h_dk_block = torch.as_tensor(
                train_dataset["h_dk"][
                    trajectory_ids[:, None],
                    block_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            h_rk_block = torch.as_tensor(
                train_dataset["h_rk"][
                    trajectory_ids[:, None],
                    block_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            G_block = torch.as_tensor(
                train_dataset["G"][
                    trajectory_ids[:, None],
                    block_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            g_dt_block = torch.as_tensor(
                train_dataset["g_dt"][
                    trajectory_ids[:, None],
                    block_ids,
                ],
                dtype=torch.complex64,
                device=DEVICE,
            )

            # (B,L,...) -> (B*L,...)
            h_dk_block = h_dk_block.reshape(
                B * LT_BLOCK_SIZE,
                TX_ANT,
                UAV_COMM,
            )

            h_rk_block = h_rk_block.reshape(
                B * LT_BLOCK_SIZE,
                RIS_UNIT,
                UAV_COMM,
            )

            G_block = G_block.reshape(
                B * LT_BLOCK_SIZE,
                RIS_UNIT,
                TX_ANT,
            )

            g_dt_block = g_dt_block.reshape(
                B * LT_BLOCK_SIZE,
                TX_ANT,
                1,
            )

            # ================================
            # Frozen ThetaNet
            # ================================
            with torch.no_grad():

                theta = theta_net(
                    h_dk_start,
                    h_rk_start,
                    G_start,
                    g_dt_start,
                )

                theta_block = (
                    theta[:, None, :]
                    .expand(
                        -1,
                        LT_BLOCK_SIZE,
                        -1,
                    )
                    .reshape(
                        B * LT_BLOCK_SIZE,
                        RIS_UNIT,
                    )
                )

                H_eff_H = (
                    theta_net.compute_effective_channel(
                        h_dk_block,
                        h_rk_block,
                        G_block,
                        theta_block,
                    )
                )

            # ================================
            # Train CommNet only
            # ================================
            optimizer.zero_grad(
                set_to_none=True
            )

            # CommNet 從 random initialization
            # 直接學 pure communication task
            #
            # g_dt 保留真實輸入，
            # 只為保持與 REG / ROB 的
            # CommNet interface 完全一致。
            W_C = comm_net(
                H_eff_H,
                g_dt_block,
            )

            (
                loss,
                nominal,
                tx_power_penalty,
                tx_power_violation_rate,
                mean_power,
                metrics,
            ) = compute_pretrain_objective(
                H_eff_H,
                W_C,
                B,
            )

            loss.backward()
            optimizer.step()

            epoch_loss.append(
                float(
                    loss.detach().cpu()
                )
            )

            epoch_nominal.append(
                float(
                    nominal.detach().cpu()
                )
            )

            epoch_tx_power_penalty.append(
                float(
                    tx_power_penalty
                    .detach()
                    .cpu()
                )
            )

            epoch_tx_power_violation_rate.append(
                float(
                    tx_power_violation_rate
                    .detach()
                    .cpu()
                )
            )

            epoch_power.append(
                float(
                    mean_power
                    .detach()
                    .cpu()
                )
            )

            epoch_signal.append(
                metrics["signal"]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )

            epoch_comm_interf.append(
                metrics["comm_interf"]
                .mean(dim=0)
                .detach()
                .cpu()
                .numpy()
            )

            epoch_noise.append(
                float(
                    metrics["noise"]
                    .detach()
                    .cpu()
                )
            )

        if len(epoch_loss) == 0:
            raise RuntimeError(
                "本 epoch 沒有有效 training batch，"
                "請檢查 LT_BLOCK_SIZE 與 dataset 長度。"
            )

        # ================================
        # Train logs
        # ================================
        train_logs = {
            "loss": float(
                np.mean(epoch_loss)
            ),

            "nominal": float(
                np.mean(epoch_nominal)
            ),

            "tx_power_penalty": float(
                np.mean(
                    epoch_tx_power_penalty
                )
            ),

            "tx_power_violation_rate": float(
                np.mean(
                    epoch_tx_power_violation_rate
                )
            ),

            "power": float(
                np.mean(epoch_power)
            ),

            "signal": np.mean(
                np.stack(
                    epoch_signal,
                    axis=0,
                ),
                axis=0,
            ),

            "comm_interf": np.mean(
                np.stack(
                    epoch_comm_interf,
                    axis=0,
                ),
                axis=0,
            ),

            "noise": float(
                np.mean(epoch_noise)
            ),
        }

        # ================================
        # Validation
        # ================================
        comm_net.eval()

        with torch.no_grad():

            W_C_val = comm_net(
                H_eff_H_val,
                g_dt_val_block,
            )

            (
                val_loss,
                val_nominal,
                val_tx_power_penalty,
                val_tx_power_violation_rate,
                val_mean_power,
                val_metrics,
            ) = compute_pretrain_objective(
                H_eff_H_val,
                W_C_val,
                B_val,
            )

            val_logs = {
                "loss": float(
                    val_loss.cpu()
                ),

                "nominal": float(
                    val_nominal.cpu()
                ),

                "tx_power_penalty": float(
                    val_tx_power_penalty.cpu()
                ),

                "tx_power_violation_rate": float(
                    val_tx_power_violation_rate.cpu()
                ),

                "power": float(
                    val_mean_power.cpu()
                ),

                "signal": val_metrics["signal"]
                .mean(dim=0)
                .cpu()
                .numpy(),

                "comm_interf": val_metrics["comm_interf"]
                .mean(dim=0)
                .cpu()
                .numpy(),

                "noise": float(
                    val_metrics["noise"].cpu()
                ),
            }

        # ================================
        # Save curves
        # ================================
        curves["train_loss"].append(
            train_logs["loss"]
        )
        curves["val_loss"].append(
            val_logs["loss"]
        )

        curves["train_nominal"].append(
            train_logs["nominal"]
        )
        curves["val_nominal"].append(
            val_logs["nominal"]
        )

        curves["train_tx_power_penalty"].append(
            train_logs["tx_power_penalty"]
        )
        curves["val_tx_power_penalty"].append(
            val_logs["tx_power_penalty"]
        )

        curves[
            "train_tx_power_violation_rate"
        ].append(
            train_logs[
                "tx_power_violation_rate"
            ]
        )

        curves[
            "val_tx_power_violation_rate"
        ].append(
            val_logs[
                "tx_power_violation_rate"
            ]
        )

        curves["train_W_C_power"].append(
            train_logs["power"]
        )
        curves["val_W_C_power"].append(
            val_logs["power"]
        )

        curves["train_signal_power"].append(
            train_logs["signal"].copy()
        )
        curves["val_signal_power"].append(
            val_logs["signal"].copy()
        )

        curves[
            "train_comm_interf_power"
        ].append(
            train_logs[
                "comm_interf"
            ].copy()
        )

        curves[
            "val_comm_interf_power"
        ].append(
            val_logs[
                "comm_interf"
            ].copy()
        )

        curves["train_noise_power"].append(
            train_logs["noise"]
        )
        curves["val_noise_power"].append(
            val_logs["noise"]
        )

        np.savez(
            comm_pretrain_curve_path,
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
        # Best checkpoint / early stop
        # 依 validation full loss
        # ================================
        if (
            best_val_loss
            - val_logs["loss"]
            > EARLY_STOP_EPS
        ):

            best_val_loss = (
                val_logs["loss"]
            )

            best_val_rate = (
                val_logs["nominal"]
            )

            best_val_epoch = (
                epoch + 1
            )

            patience_counter = 0

            comm_net.save_model(
                comm_pretrain_ckpt,
                verbose=False,
            )

        else:
            patience_counter += 1

        # ================================
        # Epoch log
        # ================================
        train_power_ratio = (
            train_logs["power"]
            / TRANSMIT_POWER_TOTAL
        )

        val_power_ratio = (
            val_logs["power"]
            / TRANSMIT_POWER_TOTAL
        )

        print(
            f"\n"
            f"[Epoch "
            f"{epoch + 1:03d}/"
            f"{COMM_PRETRAIN_EPOCHS}] "
            f"TrainLoss="
            f"{train_logs['loss']:.6f} "
            f"ValLoss="
            f"{val_logs['loss']:.6f} | "
            f"TrainNominal="
            f"{train_logs['nominal']:.4f} "
            f"ValNominal="
            f"{val_logs['nominal']:.4f}"
        )

        print(
            f"TrainSignal="
            f"{fmt_vec_sci(train_logs['signal'])} | "
            f"ValSignal="
            f"{fmt_vec_sci(val_logs['signal'])}"
        )

        print(
            f"TrainCommInterf="
            f"{fmt_vec_sci(train_logs['comm_interf'])} | "
            f"ValCommInterf="
            f"{fmt_vec_sci(val_logs['comm_interf'])}"
        )

        print(
            f"TrainNoise="
            f"{train_logs['noise']:.3e} | "
            f"ValNoise="
            f"{val_logs['noise']:.3e}"
        )

        print(
            f"TrainPower="
            f"{train_logs['power']:.8f} W "
            f"Power/P_T="
            f"{train_power_ratio:.3f} "
            f"Violation="
            f"{100.0 * train_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty="
            f"{train_logs['tx_power_penalty']:.6e}"
        )

        print(
            f"ValPower  ="
            f"{val_logs['power']:.8f} W "
            f"Power/P_T="
            f"{val_power_ratio:.3f} "
            f"Violation="
            f"{100.0 * val_logs['tx_power_violation_rate']:.2f}% "
            f"Penalty="
            f"{val_logs['tx_power_penalty']:.6e}"
        )

        if patience_counter >= PATIENCE:

            print(
                f"\n[EARLY STOP] "
                f"patience_counter="
                f"{patience_counter}, "
                f"PATIENCE="
                f"{PATIENCE}, "
                f"best_val_loss="
                f"{best_val_loss:.6f}"
            )

            break

    # ================================
    # Plot
    # ================================
    plot_curves(
        comm_pretrain_curve_path,
        comm_pretrain_curve_dir,
    )

    # ================================
    # Final summary
    # ================================
    print("\n" + "=" * 100)
    print(
        "[COMMNET PURE COMMUNICATION-TASK "
        "PRETRAIN 3 FINISHED]"
    )
    print("=" * 100)

    print(
        f"Best validation epoch      : "
        f"{best_val_epoch}"
    )
    print(
        f"Best validation loss       : "
        f"{best_val_loss:.6f}"
    )
    print(
        f"Best validation nominal    : "
        f"{best_val_rate:.6f} bps/Hz"
    )
    print(
        f"Checkpoint                 : "
        f"{comm_pretrain_ckpt}"
    )
    print(
        f"Curves                     : "
        f"{comm_pretrain_curve_path}"
    )
    print(
        f"Plots                      : "
        f"{comm_pretrain_curve_dir}"
    )

    print("=" * 100)
