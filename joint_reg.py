# -*- coding: utf-8 -*-
import os
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


def plot_joint_reg_curves(reg_curve_path, reg_curve_dir, ma_window=20):
    """
    畫四類圖：
    1. Train / Validation Loss
    2. Train / Validation Nominal worst-UE rate
    3. Train / Validation Target SNR
    4. 每個 UE 的 SINR power components
    """
    data = np.load(reg_curve_path)
    out_dir = reg_curve_dir

    os.makedirs(out_dir, exist_ok=True)

    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    train_nominal = data["train_nominal"]
    val_nominal = data["val_nominal"]

    train_target_snr_db = data["train_target_snr_db"]
    val_target_snr_db = data["val_target_snr_db"]

    epochs = np.arange(1, len(train_loss) + 1)

    # ================================
    # Fig. 1: Loss
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs,train_loss,label="Train Loss",alpha=0.35,linewidth=1.0)
    plt.plot(epochs,val_loss,label="Validation Loss",alpha=0.35,linewidth=1.0)

    train_ma_epochs, train_loss_ma  = moving_average(train_loss,ma_window)
    val_ma_epochs, val_loss_ma      = moving_average(val_loss,ma_window)

    if len(train_loss_ma) > 0:
        plt.plot(train_ma_epochs,train_loss_ma,label=f"Train Loss MA({ma_window})",linewidth=2.5)

    if len(val_loss_ma) > 0:
        plt.plot(val_ma_epochs,val_loss_ma,label=f"Validation Loss MA({ma_window})",linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fig. 1: Joint REG Loss")
    plt.grid(True,alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig1_path = os.path.join(out_dir,"Fig1_loss_curve.png")
    plt.savefig(fig1_path,dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig1_path}")

    # ================================
    # Fig. 2: Nominal Worst-UE Rate
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs,train_nominal,label="Train Nominal Worst-UE Rate",alpha=0.35,linewidth=1.0)
    plt.plot(epochs,val_nominal,label="Validation Nominal Worst-UE Rate",alpha=0.35,linewidth=1.0)

    train_ma_epochs, train_nominal_ma = moving_average(train_nominal,ma_window)
    val_ma_epochs, val_nominal_ma = moving_average(val_nominal,ma_window)

    if len(train_nominal_ma) > 0:
        plt.plot(train_ma_epochs,train_nominal_ma,label=f"Train Nominal MA({ma_window})",linewidth=2.5)

    if len(val_nominal_ma) > 0:
        plt.plot(val_ma_epochs,val_nominal_ma,label=f"Validation Nominal MA({ma_window})",linewidth=2.5)

    plt.xlabel("Epoch")
    plt.ylabel("Nominal Worst-UE Rate (bps/Hz)")
    plt.title("Fig. 2: Joint REG Nominal Worst-UE Rate")
    plt.grid(True,alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig2_path = os.path.join(out_dir,"Fig2_nominal_curve.png")
    plt.savefig(fig2_path,dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig2_path}")

    # ================================
    # Fig. 3: Target SNR
    # ================================
    plt.figure(figsize=(10, 6))

    plt.plot(epochs,train_target_snr_db,label="Train Target SNR",alpha=0.5,linewidth=1.5)
    plt.plot(epochs,val_target_snr_db,label="Validation Target SNR",alpha=0.5,linewidth=1.5)
    plt.axhline(y=SENSING_SNR_THRESHOLD_DB,linestyle="--",label="Sensing Threshold")

    plt.xlabel("Epoch")
    plt.ylabel("Target SNR (dB)")
    plt.title("Fig. 3: Joint REG Target SNR")
    plt.grid(True,alpha=0.35)
    plt.legend()
    plt.tight_layout()

    fig3_path = os.path.join(out_dir,"Fig3_target_snr_curve.png")
    plt.savefig(fig3_path,dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {fig3_path}")

    # ================================
    # Fig. 4: SINR power components
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

        plt.plot(epochs,train_signal_power[:,ue_idx],label="Train Signal",linewidth=1.8)
        plt.plot(epochs,val_signal_power[:,ue_idx],label="Validation Signal",linewidth=1.8,linestyle="--")

        plt.plot(epochs,train_comm_interf_power[:,ue_idx],label="Train Communication Interference",linewidth=1.8)
        plt.plot(epochs,val_comm_interf_power[:,ue_idx],label="Validation Communication Interference",linewidth=1.8,linestyle="--")

        plt.plot(epochs,train_noise_power,label="Train Noise",linewidth=1.8)
        plt.plot(epochs,val_noise_power,label="Validation Noise",linewidth=1.8,linestyle="--")

        plt.yscale("log")

        plt.xlabel("Epoch")
        plt.ylabel("Power")
        plt.title(f"Fig. 4: Joint REG SINR Power Components — UE {ue_idx}")
        plt.grid(True,which="both",alpha=0.35)
        plt.legend()
        plt.tight_layout()

        fig4_path = os.path.join(out_dir,f"Fig4_UE{ue_idx}_sinr_power_components.png")
        plt.savefig(fig4_path,dpi=300)
        plt.close()

        print(f"[PLOT] Saved: {fig4_path}")


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

    print("\n[INFO] RISNet initialization : Random")
    print("[INFO] JointNet initialization   : Random")
    print("[INFO] RISNet and JointNet are jointly trainable.")

    # ================================
    # Dataset
    # ================================
    with np.load(TRAIN_DATA_PATH) as data:
        train_dataset = {key: data[key] for key in ["h_dk","h_rk","G","g_dt"]}

    with np.load(VAL_DATA_PATH) as data:
        val_dataset = {key: data[key] for key in ["h_dk","h_rk","G","g_dt"]}

    train_trajectories, train_channels = train_dataset["h_dk"].shape[:2]
    val_trajectories, val_channels     = val_dataset["h_dk"].shape[:2]

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
    reg_curve_path = os.path.join(REG_CKPT_DIR,"joint_reg_curves.npz")
    reg_curve_dir  = os.path.join(REG_CKPT_DIR,"joint_reg_curves")

    os.makedirs(reg_curve_dir,exist_ok=True)

    # ================================
    # Curves
    # ================================
    curves = {
        "train_loss": [],
        "val_loss": [],

        "train_nominal": [],
        "val_nominal": [],

        "train_target_snr_db": [],
        "val_target_snr_db": [],

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
    optimizer = optim.Adam(list(ris_net.parameters()) + list(joint_net.parameters()), lr=LEARNING_RATE)

    # ================================
    # Early stopping
    # ================================
    best_val_loss = float("inf")
    best_val_worstUE_rate   = 0.0
    best_valtarget_snr_db   = 0.0
    best_val_epoch          = 0

    patience_counter        = 0
    PATIENCE                = 100

    # ================================
    # Validation block indices
    # ================================
    val_channel_ids     = np.arange(0,val_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
    val_trajectory_ids  = np.repeat(np.arange(val_trajectories),len(val_channel_ids))
    val_channel_ids     = np.tile(val_channel_ids,val_trajectories)

    B_val = len(val_channel_ids)

    # ================================
    # Training information
    # ================================
    print("\n" + "=" * 90)
    print("[JOINT TWO-TIMESCALE REG TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"REG_EPOCHS              : {REG_EPOCHS}")
    print(f"N_BATCHE                : {N_BATCHE}")
    print(f"LT_BLOCKS_PER_BATCH     : {LT_BLOCKS_PER_BATCH}")
    print(f"LT block size           : {LT_BLOCK_SIZE}")
    print(f"RISNet learning rate    : {LEARNING_RATE}")
    print(f"JointNet learning rate  : {LEARNING_RATE}")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight  : {REG_SENSING_LOSS_WEIGHT}")
    print(f"TX power                : {TRANSMIT_POWER_TOTAL:.6f} W (hard normalized)")
    print("=" * 90)

    # ============================================================
    # Training
    # ============================================================
    for epoch in trange(REG_EPOCHS,desc="Joint REG Training"):

        ris_net.train()
        joint_net.train()
        # curve
        epoch_loss                      = []
        epoch_nominal                   = []
        epoch_target_snr                = []

        epoch_signal_power              = []
        epoch_comm_interf_power         = []
        epoch_noise_power               = []


        # ========================================================
        # Train batches
        # ========================================================
        for _ in range(N_BATCHE):

            # 一個batch抽LT_BLOCKS_PER_BATCH個long-term blocks
            trajectory_ids = np.random.choice(train_trajectories,size=LT_BLOCKS_PER_BATCH,replace=True)
            channel_ids    = np.random.choice(train_channels-LT_BLOCK_SIZE+1,size=LT_BLOCKS_PER_BATCH,replace=False)
            
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
            h_dk_block = h_dk_block.reshape(LT_BLOCKS_PER_BATCH*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            h_rk_block = h_rk_block.reshape(LT_BLOCKS_PER_BATCH*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            G_block    = G_block.reshape(LT_BLOCKS_PER_BATCH*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            g_dt_block = g_dt_block.reshape(LT_BLOCKS_PER_BATCH*LT_BLOCK_SIZE,TX_ANT,1)

            optimizer.zero_grad(set_to_none=True)

            # ----------------------------------------------------
            # Long-term RISNet
            # loss 必須透過 H_eff_H 回傳 gradient 到RISNet
            # ----------------------------------------------------
            phi = ris_net(h_dk_hat,h_rk_hat,G_hat)  # (B,N)

            # 同一個 phi 固定作用於整個 LT block
            phi_block = phi[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(LT_BLOCKS_PER_BATCH*LT_BLOCK_SIZE,RIS_UNIT)

            # ----------------------------------------------------
            # 每個short-term slot計算自己的effective channel
            # gradient仍然可以經由phi_block回到RISNet
            # ----------------------------------------------------
            H_eff_H = compute_effective_channel(h_dk_block,h_rk_block,G_block,phi_block)

            # ----------------------------------------------------
            # Short-term JointNet
            # 每個slot根據當下effective channel重新產生joint beamformer
            # W shape = (B*Q,M,K)
            # JointNet內部已hard normalization：
            # ||W||_F^2 = TRANSMIT_POWER_TOTAL
            # ----------------------------------------------------
            W = joint_net(H_eff_H,g_dt_block)

            # ----------------------------------------------------
            # Communication + sensing performance
            # ----------------------------------------------------
            metrics = compute_isac_batch_performance(H_eff_H,g_dt_block,W)

            # 轉回(B,Q)
            nominal_allUE_rate      = metrics["rate"].reshape(LT_BLOCKS_PER_BATCH,LT_BLOCK_SIZE,UAV_COMM)   # B block 10slot k UE  
            nominal_target_snr      = metrics["target_snr"].reshape(LT_BLOCKS_PER_BATCH,LT_BLOCK_SIZE)      
            nominal_target_snr_db   = metrics["target_snr_db"].reshape(LT_BLOCKS_PER_BATCH,LT_BLOCK_SIZE)

            nominal_signal          = metrics["signal"]
            nominal_comm_interf     = metrics["comm_interf"]
            nominal_noise           = metrics["noise"]

            # ----------------------------------------------------
            # Nominal worst-UE rate
            # 每slot先找worst UE，再對block slots平均，再對batch平均
            # ----------------------------------------------------
            nominal_with_batch  = torch.min(nominal_allUE_rate,dim=2).values
            nominal_with_block  = torch.mean(nominal_with_batch,dim=1)
            nominal             = torch.mean(nominal_with_block)

            # ----------------------------------------------------
            # Target sensing
            # ----------------------------------------------------
            nominal_target_snr_mean = torch.mean(nominal_target_snr)



            # 每個short-term slot都必須符合sensing threshold
            sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB - nominal_target_snr_db)

            sensing_penalty = torch.mean(sensing_deficit_db)

            # ----------------------------------------------------
            # Loss
            #
            # 因為JointNet已hard enforce：
            # ||W||_F^2 = P_max
            # ----------------------------------------------------
            loss = (-nominal+ REG_SENSING_LOSS_WEIGHT * sensing_penalty)

            loss.backward()
            optimizer.step()

            # ----------------------------------------------------
            # Debug / epoch statistics
            # ----------------------------------------------------
            signal_power      = torch.mean(nominal_signal,dim=0)
            comm_interf_power = torch.mean(nominal_comm_interf,dim=0)

            epoch_loss.append(float(loss.detach().cpu()))
            epoch_nominal.append(float(nominal.detach().cpu()))
            epoch_target_snr.append(float(nominal_target_snr_mean.detach().cpu()))

            epoch_signal_power.append(signal_power.detach().cpu().numpy())
            epoch_comm_interf_power.append(comm_interf_power.detach().cpu().numpy())
            epoch_noise_power.append(float(nominal_noise.detach().cpu()))


        # ========================================================
        # Train logs
        # ========================================================
        train_target_snr_mean = float(np.mean(epoch_target_snr))
        train_target_snr_mean_db = 10.0 * np.log10(max(train_target_snr_mean,1e-12))

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "nominal": float(np.mean(epoch_nominal)),
            "target_snr_db": train_target_snr_mean_db,

            "signal_power": np.mean(np.stack(epoch_signal_power,axis=0),axis=0),
            "comm_interf_power": np.mean(np.stack(epoch_comm_interf_power,axis=0),axis=0),
            "noise_power": float(np.mean(epoch_noise_power)),
        }

        # ========================================================
        # Validation
        # ========================================================
        ris_net.eval()
        joint_net.eval()

        with torch.no_grad():
            # ----------------------------------------------------
            # Long-term start-slot CSI
            # ----------------------------------------------------
            h_dk_val = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)

            # 每個validation block只在start slot決定一次 phi_val
            phi_val = ris_net(h_dk_val,h_rk_val,G_val)  # (B_val,N)

            # ----------------------------------------------------
            # 整個LT block
            # ----------------------------------------------------
            val_block_ids = val_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]

            h_dk_val_block = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_val_block = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            G_val_block    = torch.as_tensor(val_dataset["G"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_val_block = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)

            h_dk_val_block = h_dk_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            h_rk_val_block = h_rk_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            G_val_block    = G_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            g_dt_val_block = g_dt_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,1)

            # 同一個 phi 作用於整個 LT block
            phi_val_block = phi_val[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT)

            # ----------------------------------------------------
            # Effective channel
            # ----------------------------------------------------
            H_eff_H_val = compute_effective_channel(h_dk_val_block,h_rk_val_block,G_val_block,phi_val_block)

            # ----------------------------------------------------
            # Joint beamformer
            # ----------------------------------------------------
            W_val = joint_net(H_eff_H_val,g_dt_val_block)

            # ----------------------------------------------------
            # Performance
            # ----------------------------------------------------
            val_metrics = compute_isac_batch_performance(H_eff_H_val,g_dt_val_block,W_val)

            val_nominal_allUE_rate      = val_metrics["rate"].reshape(B_val,LT_BLOCK_SIZE,UAV_COMM)
            val_nominal_target_snr      = val_metrics["target_snr"].reshape(B_val,LT_BLOCK_SIZE)
            val_nominal_target_snr_db   = val_metrics["target_snr_db"].reshape(B_val,LT_BLOCK_SIZE)

            val_nominal_signal      = val_metrics["signal"]
            val_nominal_comm_interf = val_metrics["comm_interf"]
            val_nominal_noise       = val_metrics["noise"]

            # ----------------------------------------------------
            # Nominal
            # ----------------------------------------------------
            val_nominal_with_batch  = torch.min(val_nominal_allUE_rate,dim=2).values
            val_nominal_with_block  = torch.mean(val_nominal_with_batch,dim=1)
            val_nominal             = torch.mean(val_nominal_with_block)

            # ----------------------------------------------------
            # Target sensing
            # ----------------------------------------------------
            val_nominal_target_snr_mean = torch.mean(val_nominal_target_snr)
            val_target_snr_mean_db = 10.0 * torch.log10(val_nominal_target_snr_mean.clamp_min(1e-12))


            val_sensing_deficit_db = torch.relu(SENSING_SNR_THRESHOLD_DB- val_nominal_target_snr_db)

            val_sensing_penalty = torch.mean(val_sensing_deficit_db)

            # ----------------------------------------------------
            # Validation loss
            # ----------------------------------------------------
            val_loss = (-val_nominal+ REG_SENSING_LOSS_WEIGHT*val_sensing_penalty)

            # ----------------------------------------------------
            # SINR components / power
            # ----------------------------------------------------
            val_signal_power = torch.mean(val_nominal_signal,dim=0)

            val_comm_interf_power = torch.mean(val_nominal_comm_interf,dim=0)



        # ========================================================
        # Validation logs
        # ========================================================
        val_logs = {
            "loss": float(val_loss.detach().cpu()),
            "nominal": float(val_nominal.detach().cpu()),
            "target_snr_db": float(val_target_snr_mean_db.detach().cpu()),

            "signal_power": val_signal_power.detach().cpu().numpy(),
            "comm_interf_power": val_comm_interf_power.detach().cpu().numpy(),
            "noise_power": float(val_nominal_noise.detach().cpu()),
        }

        # ========================================================
        # Save curves
        # ========================================================
        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])

        curves["train_nominal"].append(train_logs["nominal"])
        curves["val_nominal"].append(val_logs["nominal"])

        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])

        curves["train_signal_power"].append(train_logs["signal_power"].copy())
        curves["val_signal_power"].append(val_logs["signal_power"].copy())

        curves["train_comm_interf_power"].append(train_logs["comm_interf_power"].copy())
        curves["val_comm_interf_power"].append(val_logs["comm_interf_power"].copy())

        curves["train_noise_power"].append(train_logs["noise_power"])
        curves["val_noise_power"].append(val_logs["noise_power"])

        np.savez(
            reg_curve_path,
            **{
                key: np.asarray(value,dtype=np.float32)
                for key,value in curves.items()
            },
        )

        # ========================================================
        # Early stopping / checkpoint
        # ========================================================
        if best_val_loss - val_logs["loss"] > EARLY_STOP_EPS:

            best_val_loss = val_logs["loss"]
            best_val_worstUE_rate = val_logs["nominal"]
            best_valtarget_snr_db = val_logs["target_snr_db"]
            best_val_epoch = epoch + 1

            patience_counter = 0
            ris_net.save_model(REG_THETA_CKPT,verbose=False)
            joint_net.save_model(REG_JOINT_CKPT,verbose=False)

        else:
            patience_counter += 1

        # ========================================================
        # Epoch log
        # ========================================================
        print(
            f"\n[Epoch {epoch + 1:03d}/{REG_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} "
            f"ValLoss={val_logs['loss']:.6f} | "

            f"TrainNominal={train_logs['nominal']:.4f} "
            f"ValNominal={val_logs['nominal']:.4f} | "

            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "
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
    plot_joint_reg_curves(reg_curve_path,reg_curve_dir)

    # ============================================================
    # Finish
    # ============================================================
    print("\n" + "=" * 90)
    print("[JOINT REG TRAINING FINISHED]")
    print("=" * 90)

    print(f"Train channels                : {train_channels}")
    print(f"Val channels                  : {val_channels}")
    print(f"REG_EPOCHS                    : {REG_EPOCHS}")
    print(f"N_BATCHE                      : {N_BATCHE}")
    print(f"LT_BLOCKS_PER_BATCH           : {LT_BLOCKS_PER_BATCH}")
    print(f"LT_BLOCK_SIZE                 : {LT_BLOCK_SIZE}")
    print(f"RISNet learning rate        : {LEARNING_RATE}")
    print(f"JointNet learning rate        : {LEARNING_RATE}")
    print(f"Sensing SNR threshold         : {SENSING_SNR_THRESHOLD_DB} dB")
    print(f"Sensing penalty weight        : {REG_SENSING_LOSS_WEIGHT}")
    print(f"TX power                      : {TRANSMIT_POWER_TOTAL:.6f} W")

    print("-" * 90)

    print(f"[INFO] Best validation epoch        = {best_val_epoch}")
    print(f"[INFO] Best validation loss         = {best_val_loss:.6f}")
    print(f"[INFO] Nominal rate at best loss    = {best_val_worstUE_rate:.6f}")
    print(f"[INFO] Target SNR at best loss      = {best_valtarget_snr_db:.3f} dB")

    print(f"[INFO] RISNet checkpoint            = {REG_THETA_CKPT}")
    print(f"[INFO] JointNet checkpoint          = {REG_JOINT_CKPT}")

    print("=" * 90)