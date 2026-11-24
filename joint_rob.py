# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from two_timescale_NN import JointNet, RISNet, compute_effective_channel, compute_isac_batch_performance


# ================================
# Helpers
# ================================
def plot_rob_curves(rob_curve_path, rob_curve_dir):
    """
    畫四類圖：
    1. Train / Validation Loss
    2. Train / Validation Robust worst-UE tail rate
    3. Train / Validation robust target SNR
    4. 每個 UE 分開畫 injected-channel mean SINR power components
    """
    data = np.load(rob_curve_path)
    out_dir = rob_curve_dir

    os.makedirs(out_dir, exist_ok=True)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    train_robust = data["train_robust"]
    val_robust = data["val_robust"]

    train_target_snr_db = data["train_target_snr_db"]
    val_target_snr_db = data["val_target_snr_db"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig. 1: Loss
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, label="Train Loss", linewidth=1.0)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=1.0)

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
    # Fig. 2: Robust Worst-UE Rate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_robust, label="Train Robust Worst-UE Rate", linewidth=1.0)
    plt.plot(epochs, val_robust, label="Validation Robust Worst-UE Rate", linewidth=1.0)

    plt.xlabel("Epoch")
    plt.ylabel(f"Robust Worst-UE Rate Q{OUTAGE_QUANTILE:.2f} (bps/Hz)")
    plt.title("Fig. 2: ROB Worst-UE Tail Rate")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir, "Fig2_robust_curve.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")

    # ================================
    # Fig. 3: Robust Target SNR
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_target_snr_db, label="Train Target SNR", linewidth=1.5)
    plt.plot(epochs, val_target_snr_db, label="Validation Target SNR", linewidth=1.5)
    plt.axhline(y=SENSING_SNR_THRESHOLD_DB, linestyle="--", label="Sensing Threshold")

    plt.xlabel("Epoch")
    plt.ylabel("Target SNR (dB)")
    plt.title("Fig. 3: ROB Target SNR")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig3_path = os.path.join(out_dir, "Fig3_target_snr_curve.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig3_path}")

    # ================================
    # Fig. 4: SINR Power Components
    # ================================
    train_signal_power = data["train_signal_power"]
    val_signal_power = data["val_signal_power"]

    train_comm_interf_power = data["train_comm_interf_power"]
    val_comm_interf_power = data["val_comm_interf_power"]

    train_noise_power = data["train_noise_power"]
    val_noise_power = data["val_noise_power"]

    num_users = train_signal_power.shape[1]

    for ue_idx in range(num_users):
        plt.figure(figsize=(12, 7))

        plt.plot(epochs, train_signal_power[:, ue_idx], label="Train Signal", linewidth=1.8)
        plt.plot(epochs, val_signal_power[:, ue_idx], label="Validation Signal", linewidth=1.8, linestyle="--")

        plt.plot(epochs, train_comm_interf_power[:, ue_idx], label="Train Communication Interference", linewidth=1.8)
        plt.plot(epochs, val_comm_interf_power[:, ue_idx], label="Validation Communication Interference", linewidth=1.8, linestyle="--")

        plt.plot(epochs, train_noise_power, label="Train Noise", linewidth=1.8)
        plt.plot(epochs, val_noise_power, label="Validation Noise", linewidth=1.8, linestyle="--")

        plt.yscale("log")

        plt.xlabel("Epoch")
        plt.ylabel("Power (linear scale)")
        plt.title(f"Fig. 4: Mean SINR Power Components over Injected Channels — UE {ue_idx}")
        plt.grid(True, which="both", alpha=0.35)
        plt.legend()
        plt.tight_layout()

        fig4_path = os.path.join(out_dir, f"Fig4_UE{ue_idx}_sinr_power_components.png")
        plt.savefig(fig4_path, dpi=300)
        plt.close()

        print(f"[PLOT] Saved: {fig4_path}")


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance):
        E|n|^2 = variance
        Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)

    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma

    return torch.complex(nr, ni).to(dtype=cdtype)


def inject_channel(channel_rep, beta):
    """式 (3.27)～(3.28)：每個樣本、時槽、通道元素獨立抽 e，再加上 sqrt(beta) * e。"""
    error = complex_awgn(channel_rep.shape, INJECTION_VARIANCE, channel_rep.device, channel_rep.dtype)
    return channel_rep + torch.sqrt(beta) * error


# ================================
# Main
# ================================
if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # ================================
    # Networks
    # ================================
    ris_net   = RISNet().to(DEVICE)
    joint_net = JointNet().to(DEVICE)


    # ================================
    # Dataset
    # ================================
    dataset_keys = ("h_dk", "h_rk", "G", "g_dt","beta_d", "beta_r", "beta_G", "beta_g")

    with np.load(TRAIN_DATA_PATH) as data:
        train_dataset = {key: data[key] for key in dataset_keys}

    with np.load(VAL_DATA_PATH) as data:
        val_dataset = {key: data[key] for key in dataset_keys}

    train_trajectories, train_channels = train_dataset["h_dk"].shape[:2]
    val_trajectories, val_channels     = val_dataset["h_dk"].shape[:2]

    # 固定鏈路：每份 dataset 各自保存一個 BS → RIS 增益和一個 BS → Target 增益。
    train_beta_G = torch.as_tensor(train_dataset["beta_G"], dtype=torch.float32, device=DEVICE)
    train_beta_g = torch.as_tensor(train_dataset["beta_g"], dtype=torch.float32, device=DEVICE)

    val_beta_G   = torch.as_tensor(val_dataset["beta_G"],   dtype=torch.float32, device=DEVICE)
    val_beta_g   = torch.as_tensor(val_dataset["beta_g"],   dtype=torch.float32, device=DEVICE)

    print("\n[INFO] datasets loaded")
    print(f"[INFO] train trajectories = {train_trajectories}")
    print(f"[INFO] val trajectories   = {val_trajectories}")
    print(f"[INFO] train channels     = {train_channels}")
    print(f"[INFO] val channels       = {val_channels}")
    print(f"[INFO] h_dk shape         = {train_dataset['h_dk'].shape}")
    print(f"[INFO] h_rk shape         = {train_dataset['h_rk'].shape}")
    print(f"[INFO] G shape            = {train_dataset['G'].shape}")
    print(f"[INFO] g_dt shape         = {train_dataset['g_dt'].shape}")

    # ================================
    # Save paths
    # ================================
    # ROB JointNet初始化方式："reg" 或 "random"
    ROB_JOINT_INIT = "random"


    # ROB專屬的兩份checkpoints
    rob_curve_path = os.path.join(ROB_CKPT_DIR,"joint_rob_curves.npz")
    rob_curve_dir  = os.path.join(ROB_CKPT_DIR,"joint_rob_curves")

    os.makedirs(rob_curve_dir,exist_ok=True)

    # ================================
    # ROB networks 初始化
    # ================================
    if ROB_JOINT_INIT not in ("reg","random"):
        raise ValueError(f"不支援的ROB_JOINT_INIT: {ROB_JOINT_INIT}")

    # RISNet固定從REG checkpoint初始化
    if not os.path.isfile(REG_THETA_CKPT):
        raise FileNotFoundError(f"找不到REG RISNet checkpoint: {REG_THETA_CKPT}")

    ris_net.load_model(REG_THETA_CKPT,strict=True,verbose=True)

    # JointNet依照設定選擇REG checkpoint或random initialization
    if ROB_JOINT_INIT == "reg":

        if not os.path.isfile(REG_JOINT_CKPT):
            raise FileNotFoundError(f"找不到REG JointNet checkpoint: {REG_JOINT_CKPT}")

        joint_net.load_model(REG_JOINT_CKPT,strict=True,verbose=True)

    # ================================
    # Initialization information
    # ================================
    print("\n[INFO] RISNet initialization : REG")
    print(f"[INFO] JointNet initialization : {ROB_JOINT_INIT.upper()}")
    print("[INFO] RISNet and JointNet are jointly trainable.")

    # ================================
    # Curves
    # ================================
    curves = {
        "train_loss": [],
        "val_loss": [],

        "train_robust": [],
        "val_robust": [],

        "train_target_snr_db": [],
        "val_target_snr_db": [],

        "train_sensing_violation_rate": [],
        "val_sensing_violation_rate": [],

        "train_signal_power": [],
        "val_signal_power": [],

        "train_comm_interf_power": [],
        "val_comm_interf_power": [],

        "train_noise_power": [],
        "val_noise_power": [],
    }


    # ================================
    # Optimizer
    # ================================
    optimizer = optim.Adam(list(ris_net.parameters()) + list(joint_net.parameters()),lr=LEARNING_RATE)


    # ================================
    # Early stopping
    # ================================
    best_val_loss = float("inf")
    best_val_worstUE_q05    = 0.0
    best_valtarget_snr_db   = 0.0
    best_val_epoch          = 0

    patience_counter        = 0
    PATIENCE                = 200

    # ================================
    # CHUNK
    # ================================
    # validation 每次處理多少個 long-term blocks
    VAL_CHUNK_SIZE = 250


    # ============================================================
    # Validation fixed block starts
    # ============================================================
    val_channel_ids     = np.arange(0,val_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
    val_trajectory_ids  = np.repeat(np.arange(val_trajectories),len(val_channel_ids))
    val_channel_ids     = np.tile(val_channel_ids,val_trajectories)

    B_val = len(val_channel_ids)

    # ================================
    # Training information
    # ================================
    print("\n" + "=" * 90)
    print("[JOINT BEAMFORMER ROB TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"ROB_EPOCHS              : {ROB_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"LT_BLOCKS_PER_BATCH     : {LT_BLOCKS_PER_BATCH}")
    print(f"LT_BLOCK_SIZE           : {LT_BLOCK_SIZE}")
    print(f"ROB learning rate       : {LEARNING_RATE}")

    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {ROB_SENSING_LOSS_WEIGHT}")
    print(f"Sensing quantile        : Q{TAR_OUTAGE_QUANTILE:.2f}")
    print(f"Injection variance      : {INJECTION_VARIANCE}")
    print(f"Injection samples       : {INJECTION_SAMPLES}")
    print(f"TX power                : {TRANSMIT_POWER_TOTAL:.6f} W (hard normalized)")
    print("=" * 90)

    # ============================================================
    # Training
    # ============================================================
    for epoch in trange(ROB_EPOCHS, desc="ROB Training"):

        ris_net.train()
        joint_net.train()
        # curve
        epoch_loss                          = []
        epoch_robust                        = []
        epoch_target_snr                    = []
        epoch_sensing_violation_rates       = []

        epoch_signal_power                  = []
        epoch_comm_interf_power             = []
        epoch_noise_power                   = []

        # ========================================================
        # Train batches
        # ========================================================
        for _ in range(N_BATCHE):

            # 一個batch抽LT_BLOCKS_PER_BATCH個long-term blocks
            trajectory_ids = np.random.choice(train_trajectories,size=LT_BLOCKS_PER_BATCH,replace=True)
            channel_ids    = np.random.choice(train_channels-LT_BLOCK_SIZE+1,size=LT_BLOCKS_PER_BATCH,replace=False)

            B = LT_BLOCKS_PER_BATCH
            B_short = B*LT_BLOCK_SIZE # 50*10=500

            # ----------------------------------------------------
            # Long-term start-slot CSI
            # RISNet每個block只看一次start-slot CSI
            # ----------------------------------------------------
            h_dk_hat = torch.as_tensor(train_dataset["h_dk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)

            # ----------------------------------------------------
            # 對應整個LT block的short-term channels
            # ----------------------------------------------------
            block_ids = channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]

            h_dk_block = torch.as_tensor(train_dataset["h_dk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_block = torch.as_tensor(train_dataset["h_rk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
            G_block    = torch.as_tensor(train_dataset["G"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_block = torch.as_tensor(train_dataset["g_dt"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)

            # (B,Q,...) -> (B*Q,...)
            h_dk_block = h_dk_block.reshape(B_short,TX_ANT,UAV_COMM)
            h_rk_block = h_rk_block.reshape(B_short,RIS_UNIT,UAV_COMM)
            G_block    = G_block.reshape(B_short,RIS_UNIT,TX_ANT)
            g_dt_block = g_dt_block.reshape(B_short,TX_ANT,1)

            optimizer.zero_grad(set_to_none=True)

            # ----------------------------------------------------
            # Long-term RISNet
            # loss必須透過H_eff_H回傳gradient到RISNet
            # ----------------------------------------------------
            phi = ris_net(h_dk_hat,h_rk_hat,G_hat)                   # (B,N)

            # 同一個phi固定作用於整個LT block
            phi_block = phi[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_short,RIS_UNIT)

            # ----------------------------------------------------
            # 每個short-term slot計算自己的effective channel
            # gradient仍然可以經由phi_block回到RISNet
            # ----------------------------------------------------
            H_eff_H = compute_effective_channel(h_dk_block,h_rk_block,G_block,phi_block)

            # ----------------------------------------------------
            # Short-term JointNet
            # 每個slot根據當下nominal effective channel重新產生joint beamformer
            # W shape = (B*Q,M,K)
            # JointNet內部已hard normalization：
            # ||W||_F^2 = TRANSMIT_POWER_TOTAL
            # ----------------------------------------------------
            W = joint_net(H_eff_H,g_dt_block)

            # ----------------------------------------------------
            # Uncertainty Injection
            # 每個nominal short-term slot產生S組uncertainty samples
            # ----------------------------------------------------
            S = INJECTION_SAMPLES
            M = TX_ANT
            N = RIS_UNIT
            K = UAV_COMM

            h_dk_rep = h_dk_block.unsqueeze(0).expand(S,B_short,M,K)            # (S,B*Q,M,K)
            h_rk_rep = h_rk_block.unsqueeze(0).expand(S,B_short,N,K)            # (S,B*Q,N,K)
            G_rep    = G_block.unsqueeze(0).expand(S,B_short,N,M)               # (S,B*Q,N,M)
            g_dt_rep = g_dt_block.unsqueeze(0).expand(S,B_short,M,1)            # (S,B*Q,M,1)

            # ----------------------------------------------------
            # 取出與本批通道相同 trajectory、slot、UE 的路徑增益。
            # 索引結果為 (B, Q, K)，重排成 (1, B*Q, 1, K)，以便沿 S 軸與天線軸廣播。
            # ----------------------------------------------------
            beta_d = torch.as_tensor(train_dataset["beta_d"][trajectory_ids[:, None], block_ids], dtype=torch.float32, device=DEVICE).reshape(1, B_short, 1, K)
            beta_r = torch.as_tensor(train_dataset["beta_r"][trajectory_ids[:, None], block_ids], dtype=torch.float32, device=DEVICE).reshape(1, B_short, 1, K)

            """
                h_dk_power = h_dk_rep.abs().square().mean(dim=2, keepdim=True)        # (S, B*Q, 1, K)
                h_rk_power = h_rk_rep.abs().square().mean(dim=2, keepdim=True)        # (S, B*Q, 1, K)
                G_power    = G_rep.abs().square().mean(dim=(2, 3), keepdim=True)      # (S, B*Q, 1, 1)
                g_dt_power = g_dt_rep.abs().square().mean(dim=(2, 3), keepdim=True)   # (S, B*Q, 1, 1)
            """

            # ----------------------------------------------------
            # 四條鏈路各自獨立抽樣，每個複數通道元素的平均誤差功率為 INJECTION_VARIANCE * beta
            # ----------------------------------------------------
            h_dk_inj = inject_channel(h_dk_rep, beta_d)
            h_rk_inj = inject_channel(h_rk_rep, beta_r)
            G_inj    = inject_channel(G_rep, train_beta_G)
            g_dt_inj = inject_channel(g_dt_rep, train_beta_g)

            # ----------------------------------------------------
            # Repeat nominal decisions
            # Injection只作用於performance，不重新產生phi或W
            # 同一個nominal slot的S組true channels共用相同phi與W
            # ----------------------------------------------------
            phi_rep = phi_block.unsqueeze(0).expand(S,B_short,N)            # (S,B*Q,N)
            W_rep     = W.unsqueeze(0).expand(S,B_short,M,K)                # (S,B*Q,M,K)

            # ----------------------------------------------------
            # Flatten injection and short-term dimensions
            # (S,B*Q,...) -> (S*B*Q,...)
            # ----------------------------------------------------
            h_dk_flat = h_dk_inj.reshape(S*B_short,M,K)
            h_rk_flat = h_rk_inj.reshape(S*B_short,N,K)
            G_flat    = G_inj.reshape(S*B_short,N,M)
            g_dt_flat = g_dt_inj.reshape(S*B_short,M,1)

            phi_flat   = phi_rep.reshape(S*B_short,N)
            W_flat     = W_rep.reshape(S*B_short,M,K)

            # ----------------------------------------------------
            # Injected true effective channels
            # phi_flat與W_flat均保留autograd graph
            # Robust loss可回傳gradient至RISNet與JointNet
            # ----------------------------------------------------
            H_eff_H_flat = compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,phi_flat) # (S*B_short,K,M)
            metrics_flat = compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_flat)

            # ----------------------------------------------------
            # Restore S,B,Q dimensions
            # ----------------------------------------------------
            robust_allUE_rate    = metrics_flat["rate"].reshape(S,B,LT_BLOCK_SIZE,K)          # (S,B,Q,K)
            robust_target_snr    = metrics_flat["target_snr"].reshape(S,B,LT_BLOCK_SIZE)      # (S,B,Q)
            robust_target_snr_db = metrics_flat["target_snr_db"].reshape(S,B,LT_BLOCK_SIZE)   # (S,B,Q)

            robust_signal        = metrics_flat["signal"].reshape(S,B,LT_BLOCK_SIZE,K)        # (S,B,Q,K)
            robust_comm_interf   = metrics_flat["comm_interf"].reshape(S,B,LT_BLOCK_SIZE,K)   # (S,B,Q,K)
            robust_noise         = metrics_flat["noise"]                                      # scalar

            # ----------------------------------------------------
            # Robust communication objective
            # 1. 每個injection sample、每個slot先取worst UE
            # 2. 每個slot沿S維度取Q05
            # 3. 對LT block內的Q個slots平均，再對B個blocks平均
            # ----------------------------------------------------
            worstUE_rate     = torch.min(robust_allUE_rate,dim=3).values            # (S,B,Q)
            robust_per_slot  = torch.quantile(worstUE_rate,OUTAGE_QUANTILE,dim=0)   # (B,Q)
            robust_per_block = torch.mean(robust_per_slot,dim=1)                    # (B,)
            robust = torch.mean(robust_per_block)                                   # scalar

            # ----------------------------------------------------
            # SINR power components
            # 對所有injection samples、blocks及slots平均
            # ----------------------------------------------------
            robust_signal_power      = torch.mean(robust_signal,dim=(0,1,2))      # (K,)
            robust_comm_interf_power = torch.mean(robust_comm_interf,dim=(0,1,2)) # (K,)

            # ----------------------------------------------------
            # Robust sensing
            # 感測SNR：所有injection samples、blocks及slots的linear mean
            # ----------------------------------------------------
            robust_target_snr_mean = torch.mean(robust_target_snr)                # scalar

            # 每個 slot 沿 S 軸對線性 SNR 取分位數並線性插值，再轉成 dB 計算門檻缺額。
            target_snr_q    = torch.quantile(robust_target_snr, TAR_OUTAGE_QUANTILE, dim=0, interpolation="linear") # (B,Q)
            target_snr_q_db = 10.0 * torch.log10(target_snr_q.clamp_min(1e-12))

            # Q10 sensing SNR低於threshold時產生penalty
            sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - target_snr_q_db)         # (B,Q)
            sensing_penalty = sensing_deficit_db.mean()                                         # scalar

            # ----------------------------------------------------
            # Sensing violation statistics
            # 每個slot統計S組uncertainty samples中的違規比例
            # 這裡算sensing_violation_rate 但不是用這當作loss
            # ----------------------------------------------------
            sensing_violation_per_slot = torch.mean((robust_target_snr_db<SENSING_SNR_THRESHOLD_DB).float(),dim=0) # (B,Q)
            sensing_violation_rate = torch.mean(sensing_violation_per_slot)                                        # scalar

            # ----------------------------------------------------
            # Robust loss
            # JointNet已hard enforce ||W||_F^2=P_max，不需要TX power penalty
            # ----------------------------------------------------
            loss = -robust + ROB_SENSING_LOSS_WEIGHT*sensing_penalty

            loss.backward()
            optimizer.step()

            # ----------------------------------------------------
            # Debug / epoch statistics
            # ----------------------------------------------------
            epoch_loss.append(float(loss.detach().cpu()))
            epoch_robust.append(float(robust.detach().cpu()))
            epoch_target_snr.append(float(robust_target_snr_mean.detach().cpu()))
            epoch_sensing_violation_rates.append(float(sensing_violation_rate.detach().cpu()))

            epoch_signal_power.append(robust_signal_power.detach().cpu().numpy())
            epoch_comm_interf_power.append(robust_comm_interf_power.detach().cpu().numpy())
            epoch_noise_power.append(float(robust_noise.detach().cpu()))


        # ========================================================
        # Train logs
        # ========================================================
        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0*np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss":float(np.mean(epoch_loss)),
            "robust":float(np.mean(epoch_robust)),
            "target_snr_db":train_target_snr_mean_db,
            "sensing_violation_rate":float(np.mean(epoch_sensing_violation_rates)),

            "signal_power":np.mean(np.stack(epoch_signal_power,axis=0),axis=0),
            "comm_interf_power":np.mean(np.stack(epoch_comm_interf_power,axis=0),axis=0),
            "noise_power":float(np.mean(epoch_noise_power)),
        }


        # ========================================================
        # Validation
        # ========================================================
        ris_net.eval()
        joint_net.eval()

        with torch.no_grad():
            # ----------------------------------------------------
            # Long-term start-slot CSI
            # RISNet每個validation block只看一次start-slot CSI
            # ----------------------------------------------------
            h_dk_val = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)

            # ----------------------------------------------------
            # Long-term RISNet
            # 每個LT block產生一組phi
            # ----------------------------------------------------
            phi_val = ris_net(h_dk_val,h_rk_val,G_val)                     # (B_val,N)

            # ----------------------------------------------------
            # 對應整個LT block的short-term channels
            # ----------------------------------------------------
            val_block_ids = val_channel_ids[:,None]+np.arange(LT_BLOCK_SIZE)[None,:]

            h_dk_val_block = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_val_block = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            G_val_block    = torch.as_tensor(val_dataset["G"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_val_block = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)

            # (B_val,Q,...) -> (B_val*Q,...)
            h_dk_val_block = h_dk_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            h_rk_val_block = h_rk_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            G_val_block    = G_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            g_dt_val_block = g_dt_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,1)

            # ----------------------------------------------------
            # 同一個phi固定作用於整個LT block
            # ----------------------------------------------------
            phi_val_block = phi_val[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT)

            # ----------------------------------------------------
            # Nominal effective channel
            # 每個short-term slot計算自己的effective channel
            # ----------------------------------------------------
            H_eff_H_val = compute_effective_channel(h_dk_val_block,h_rk_val_block,G_val_block,phi_val_block)

            # ----------------------------------------------------
            # Short-term JointNet
            # 每個slot根據nominal effective channel產生joint beamformer
            # ----------------------------------------------------
            W_val = joint_net(H_eff_H_val,g_dt_val_block)                                 # (B_val*Q,M,K)

            # ----------------------------------------------------
            # Validation injection dimensions
            # ----------------------------------------------------
            val_S = INJECTION_SAMPLES
            val_M = TX_ANT
            val_N = RIS_UNIT
            val_K = UAV_COMM

            # ----------------------------------------------------
            # Validation accumulators
            # 保留原ROB的加總方式，最後再依樣本數正規化
            # ----------------------------------------------------
            val_robust_sum                  = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_robust_signal_sum           = torch.zeros(val_K,dtype=torch.float32,device=DEVICE)
            val_robust_comm_interf_sum      = torch.zeros(val_K,dtype=torch.float32,device=DEVICE)
            val_robust_target_snr_sum       = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_sensing_penalty_sum         = torch.zeros((),dtype=torch.float32,device=DEVICE)
            val_sensing_violation_sum       = torch.zeros((),dtype=torch.float32,device=DEVICE)

            val_robust_noise = None

            # ----------------------------------------------------
            # Validation injection chunks
            # 每次處理VAL_CHUNK_SIZE個LT blocks
            # 每個epoch使用新的uncertainty samples
            # ----------------------------------------------------
            for val_chunk_start in range(0,B_val,VAL_CHUNK_SIZE):

                val_chunk_end = min(val_chunk_start+VAL_CHUNK_SIZE,B_val)
                val_B_chunk = val_chunk_end-val_chunk_start
                val_B_short_chunk = val_B_chunk*LT_BLOCK_SIZE

                val_short_start = val_chunk_start*LT_BLOCK_SIZE
                val_short_end   = val_chunk_end*LT_BLOCK_SIZE

                # ----------------------------------------------------
                # Nominal short-term channels and decisions
                # ----------------------------------------------------
                val_h_dk_block_chunk = h_dk_val_block[val_short_start:val_short_end]
                val_h_rk_block_chunk = h_rk_val_block[val_short_start:val_short_end]
                val_G_block_chunk    = G_val_block[val_short_start:val_short_end]
                val_g_dt_block_chunk = g_dt_val_block[val_short_start:val_short_end]

                val_phi_block_chunk = phi_val_block[val_short_start:val_short_end]
                val_W_chunk           = W_val[val_short_start:val_short_end]

                # ----------------------------------------------------
                # Uncertainty Injection
                # 每個nominal short-term slot產生S組uncertainty samples
                # ----------------------------------------------------
                val_h_dk_rep = val_h_dk_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,val_K)
                val_h_rk_rep = val_h_rk_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_N,val_K)
                val_G_rep    = val_G_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_N,val_M)
                val_g_dt_rep = val_g_dt_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,1)
                """
                # ----------------------------------------------------
                # 每個時槽分別計算各鏈路的平均元素功率。
                # h_dk、h_rk 對每位 UE 各算一個功率；G、g_dt 各算一個鏈路功率。
                # ----------------------------------------------------
                val_h_dk_power = val_h_dk_rep.abs().square().mean(dim=2, keepdim=True)       # (val_S, val_B_short_chunk, 1, val_K)
                val_h_rk_power = val_h_rk_rep.abs().square().mean(dim=2, keepdim=True)       # (val_S, val_B_short_chunk, 1, val_K)
                val_G_power    = val_G_rep.abs().square().mean(dim=(2, 3), keepdim=True)     # (val_S, val_B_short_chunk, 1, 1)
                val_g_dt_power = val_g_dt_rep.abs().square().mean(dim=(2, 3), keepdim=True)  # (val_S, val_B_short_chunk, 1, 1)

                # ----------------------------------------------------
                # Inject channel uncertainty
                # error ~ CN(0,INJECTION_VARIANCE*empirical_power)
                # ----------------------------------------------------
                val_h_dk_inj = val_h_dk_rep+torch.sqrt(INJECTION_VARIANCE*val_h_dk_power)*complex_awgn(val_h_dk_rep.shape,1.0,DEVICE,val_h_dk_rep.dtype)
                val_h_rk_inj = val_h_rk_rep+torch.sqrt(INJECTION_VARIANCE*val_h_rk_power)*complex_awgn(val_h_rk_rep.shape,1.0,DEVICE,val_h_rk_rep.dtype)
                val_G_inj    = val_G_rep+torch.sqrt(INJECTION_VARIANCE*val_G_power)*complex_awgn(val_G_rep.shape,1.0,DEVICE,val_G_rep.dtype)
                val_g_dt_inj = val_g_dt_rep+torch.sqrt(INJECTION_VARIANCE*val_g_dt_power)*complex_awgn(val_g_dt_rep.shape,1.0,DEVICE,val_g_dt_rep.dtype)

                """

                # val_chunk_start/end 是區間索引；取出此 chunk 各區間對應的軌跡與 Q 個 slot。
                chunk_trajectory_ids = val_trajectory_ids[val_chunk_start:val_chunk_end]
                chunk_slot_ids       = val_block_ids[val_chunk_start:val_chunk_end]

                # (B_chunk, Q, K) -> (1, B_chunk*Q, 1, K)，順序與 val_h_dk_block_chunk 相同。
                val_beta_d_chunk = torch.as_tensor(val_dataset["beta_d"][chunk_trajectory_ids[:, None], chunk_slot_ids], dtype=torch.float32, device=DEVICE).reshape(1, val_B_short_chunk, 1, val_K)
                val_beta_r_chunk = torch.as_tensor(val_dataset["beta_r"][chunk_trajectory_ids[:, None], chunk_slot_ids], dtype=torch.float32, device=DEVICE).reshape(1, val_B_short_chunk, 1, val_K)

                # 驗證與訓練共用同一條注入公式。
                val_h_dk_inj = inject_channel(val_h_dk_rep, val_beta_d_chunk)
                val_h_rk_inj = inject_channel(val_h_rk_rep, val_beta_r_chunk)
                val_G_inj    = inject_channel(val_G_rep, val_beta_G)
                val_g_dt_inj = inject_channel(val_g_dt_rep, val_beta_g)

                # ----------------------------------------------------
                # Repeat nominal decisions
                # 同一個nominal slot的S組true channels共用相同phi與W
                # ----------------------------------------------------
                val_phi_rep = val_phi_block_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_N)
                val_W_rep   = val_W_chunk.unsqueeze(0).expand(val_S,val_B_short_chunk,val_M,val_K)

                # ----------------------------------------------------
                # Flatten injection and short-term dimensions
                # (S,B_chunk*Q,...) -> (S*B_chunk*Q,...)
                # ----------------------------------------------------
                val_h_dk_flat = val_h_dk_inj.reshape(val_S*val_B_short_chunk,val_M,val_K)
                val_h_rk_flat = val_h_rk_inj.reshape(val_S*val_B_short_chunk,val_N,val_K)
                val_G_flat    = val_G_inj.reshape(val_S*val_B_short_chunk,val_N,val_M)
                val_g_dt_flat = val_g_dt_inj.reshape(val_S*val_B_short_chunk,val_M,1)

                val_phi_flat = val_phi_rep.reshape(val_S*val_B_short_chunk,val_N)
                val_W_flat     = val_W_rep.reshape(val_S*val_B_short_chunk,val_M,val_K)

                # ----------------------------------------------------
                # Injected true effective channels
                # 不重新生成phi或W，只計算注入通道下的實際效能
                # ----------------------------------------------------
                val_H_eff_H_flat = compute_effective_channel(val_h_dk_flat,val_h_rk_flat,val_G_flat,val_phi_flat)
                val_metrics_flat = compute_isac_batch_performance(val_H_eff_H_flat,val_g_dt_flat,val_W_flat)

                # ----------------------------------------------------
                # Restore S,B,Q dimensions
                # ----------------------------------------------------
                val_robust_allUE_rate    = val_metrics_flat["rate"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)          # (S,B_chunk,Q,K)
                val_robust_target_snr    = val_metrics_flat["target_snr"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE)           # (S,B_chunk,Q)
                val_robust_target_snr_db = val_metrics_flat["target_snr_db"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE)        # (S,B_chunk,Q)

                val_robust_signal        = val_metrics_flat["signal"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)        # (S,B_chunk,Q,K)
                val_robust_comm_interf   = val_metrics_flat["comm_interf"].reshape(val_S,val_B_chunk,LT_BLOCK_SIZE,val_K)   # (S,B_chunk,Q,K)
                val_robust_noise         = val_metrics_flat["noise"]                                                         # scalar

                # ----------------------------------------------------
                # Robust communication objective
                # 1. 每個injection sample、每個slot先取worst UE
                # 2. 每個slot沿S維度取Q05
                # 3. 累加所有validation slots
                # ----------------------------------------------------
                val_worstUE_rate    = torch.min(val_robust_allUE_rate,dim=3).values             # (S,B_chunk,Q)
                val_robust_per_slot = torch.quantile(val_worstUE_rate,OUTAGE_QUANTILE,dim=0)     # (B_chunk,Q)
                val_robust_sum += torch.sum(val_robust_per_slot)

                # ----------------------------------------------------
                # SINR power components
                # 累加所有injection samples、blocks及slots
                # ----------------------------------------------------
                val_robust_signal_sum += torch.sum(val_robust_signal,dim=(0,1,2))
                val_robust_comm_interf_sum += torch.sum(val_robust_comm_interf,dim=(0,1,2))

                # ----------------------------------------------------
                # Robust sensing
                # ----------------------------------------------------
                val_robust_target_snr_sum += torch.sum(val_robust_target_snr)

                # 每個 slot 沿 S 軸對線性 SNR 取分位數並線性插值，再轉成 dB 計算門檻缺額
                val_target_snr_q = torch.quantile(val_robust_target_snr, TAR_OUTAGE_QUANTILE, dim=0, interpolation="linear") # (B_chunk,Q)
                val_target_snr_q_db = 10.0 * torch.log10(val_target_snr_q.clamp_min(1e-12))

                # Q10 sensing SNR低於threshold時產生penalty
                val_sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - val_target_snr_q_db)
                val_sensing_penalty_sum += val_sensing_deficit_db.sum()

                # ----------------------------------------------------
                # Sensing violation statistics
                # 每個slot統計S組uncertainty samples中的違規比例
                # ----------------------------------------------------
                val_sensing_violation_per_slot = torch.mean((val_robust_target_snr_db<SENSING_SNR_THRESHOLD_DB).float(),dim=0)
                val_sensing_violation_sum += torch.sum(val_sensing_violation_per_slot)

                # ----------------------------------------------------
                # Release large injection tensors
                # ----------------------------------------------------
                del val_h_dk_inj,val_h_rk_inj,val_G_inj,val_g_dt_inj
                del val_h_dk_flat,val_h_rk_flat,val_G_flat,val_g_dt_flat
                del val_phi_flat,val_W_flat,val_H_eff_H_flat,val_metrics_flat
                del val_robust_allUE_rate,val_robust_target_snr,val_robust_target_snr_db
                del val_robust_signal,val_robust_comm_interf
                del val_worstUE_rate,val_robust_per_slot
                del val_target_snr_q, val_target_snr_q_db, val_sensing_deficit_db
                del val_sensing_violation_per_slot

            # ----------------------------------------------------
            # Merge validation chunks
            # 先累加全部validation結果，再依正確樣本數平均
            # ----------------------------------------------------
            val_short_slot_count = B_val*LT_BLOCK_SIZE
            val_metric_count = val_S*val_short_slot_count

            val_robust = val_robust_sum/val_short_slot_count

            val_robust_signal_power = val_robust_signal_sum/val_metric_count
            val_robust_comm_interf_power = val_robust_comm_interf_sum/val_metric_count

            val_robust_target_snr_mean = val_robust_target_snr_sum/val_metric_count
            val_robust_target_snr_mean_db = 10.0*torch.log10(val_robust_target_snr_mean.clamp_min(1e-12))

            val_sensing_penalty = val_sensing_penalty_sum/val_short_slot_count
            val_sensing_violation_rate = val_sensing_violation_sum/val_short_slot_count


            # ----------------------------------------------------
            # Validation loss
            # 保留原ROB的Q05 rate + Q10 sensing penalty
            # ----------------------------------------------------
            val_loss = -val_robust+ROB_SENSING_LOSS_WEIGHT*val_sensing_penalty



        # ========================================================
        # Validation logs
        # ========================================================
        val_logs = {
            "loss":float(val_loss.detach().cpu()),
            "robust":float(val_robust.detach().cpu()),
            "target_snr_db":float(val_robust_target_snr_mean_db.detach().cpu()),
            "sensing_violation_rate":float(val_sensing_violation_rate.detach().cpu()),

            "signal_power":val_robust_signal_power.detach().cpu().numpy(),
            "comm_interf_power":val_robust_comm_interf_power.detach().cpu().numpy(),
            "noise_power":float(val_robust_noise.detach().cpu()),
        }


        # ========================================================
        # Save curves
        # ========================================================
        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_robust"].append(train_logs["robust"])
        curves["val_robust"].append(val_logs["robust"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        curves["train_sensing_violation_rate"].append(train_logs["sensing_violation_rate"])
        curves["val_sensing_violation_rate"].append(val_logs["sensing_violation_rate"])

        curves["train_signal_power"].append(train_logs["signal_power"].copy())
        curves["val_signal_power"].append(val_logs["signal_power"].copy())

        curves["train_comm_interf_power"].append(train_logs["comm_interf_power"].copy())
        curves["val_comm_interf_power"].append(val_logs["comm_interf_power"].copy())

        curves["train_noise_power"].append(train_logs["noise_power"])
        curves["val_noise_power"].append(val_logs["noise_power"])

        np.savez(
            rob_curve_path,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in curves.items()
            },
        )

        # ========================================================
        # Early stopping
        # ========================================================
        if best_val_loss - val_logs["loss"] > EARLY_STOP_EPS:

            best_val_loss = val_logs["loss"]
            best_val_worstUE_q05 = val_logs["robust"]
            best_valtarget_snr_db = val_logs["target_snr_db"]
            best_val_epoch = epoch + 1

            patience_counter = 0
            ris_net.save_model(ROB_THETA_CKPT,verbose=False)
            joint_net.save_model(ROB_JOINT_CKPT,verbose=False)

        else:
            patience_counter += 1

        # ========================================================
        # Epoch log
        # ========================================================
        print(
            f"\n[Epoch {epoch + 1:03d}/{ROB_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "

            f"TrainRobust={train_logs['robust']:.4f} "
            f"ValRobust={val_logs['robust']:.4f} | "

            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "

            f"TrainSenseViolation={100.0 * train_logs['sensing_violation_rate']:.2f}% "
            f"ValSenseViolation={100.0 * val_logs['sensing_violation_rate']:.2f}% | "

        )

        if patience_counter >= PATIENCE:
            print(
                f"\n[EARLY STOP] "
                f"patience_counter={patience_counter}, "
                f"PATIENCE={PATIENCE}, "
                f"best_val_loss={best_val_loss:.6f}"
            )
            break

    # ============================================================
    # Plot
    # ============================================================
    plot_rob_curves(rob_curve_path,rob_curve_dir)

    # ============================================================
    # Finish
    # ============================================================
    print("\n" + "=" * 90)
    print("[ROB TRAINING FINISHED]")
    print("=" * 90)
    print(f"RISNet                : Independent ROB RISNet")
    print(f"JointNet                : Independent ROB JointNet")
    print(f"Training mode           : Joint training")
    print(f"ROB_EPOCHS                       : {ROB_EPOCHS}")
    print(f"N_BATCHE                         : {N_BATCHE}")
    print(f"LT_BLOCKS_PER_BATCH              : {LT_BLOCKS_PER_BATCH}")
    print(f"ROB learning rate                : {LEARNING_RATE}")
    print(f"Injection variance               : {INJECTION_VARIANCE}")
    print(f"Injection samples                : {INJECTION_SAMPLES}")
    print(f"Communication outage quantile    : {OUTAGE_QUANTILE}")
    print(f"Sensing outage quantile          : {TAR_OUTAGE_QUANTILE}")
    print(f"Sensing SNR threshold            : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight           : {ROB_SENSING_LOSS_WEIGHT}")
    print(f"TX power                         : {TRANSMIT_POWER_TOTAL:.6f} W")

    print("-" * 90)

    print(f"[INFO] Best validation epoch          = {best_val_epoch}")
    print(f"[INFO] Best validation loss           = {best_val_loss:.6f}")
    print(f"[INFO] Robust rate at best val loss   = {best_val_worstUE_q05:.6f}")
    print(f"[INFO] Target SNR at best val loss    = {best_valtarget_snr_db:.3f} dB")

    print("=" * 90)