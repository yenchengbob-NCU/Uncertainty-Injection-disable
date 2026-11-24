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
# CommNet communication-only pretrain settings
# ================================
COMM_PRETRAIN_EPOCHS = 500
COMM_PRETRAIN_LEARNING_RATE = 1e-3
COMM_PRETRAIN_BATCHES = N_BATCHE
COMM_PRETRAIN_BATCH_CHANNELS = ST_BATCH_SIZE

# CommNet預訓練只希望輸出power落在約P_T/2，這裡不是hard normalization。
# Joint REG/ROB載入此checkpoint後，W_C power仍可自由改變。
COMM_PRETRAIN_TARGET_POWER = TRANSMIT_POWER_TOTAL / 2.0

# 額外明確告訴CommNet壓低multi-user communication interference。
# interference loss使用log(1 + I_C / noise)避免10^-10量級直接進loss後梯度過小。
COMM_INTERF_LOSS_WEIGHT = 0.5

# 只控制預訓練階段的平均W_C power接近0.63 W，避免CommNet靠縮小全部輸出降低interference loss。
COMM_POWER_TARGET_LOSS_WEIGHT = 10.0

EARLY_STOP_EPS = 1e-7
PATIENCE = 50


# ================================
# Reproducibility
# ================================
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ================================
# Communication metrics / loss
# ================================
def compute_comm_metrics(H_eff_H,W_C):
    """
    H_eff_H : (S,K,M)
    W_C     : (S,M,K)

    received = H_eff_H @ W_C : (S,K,K)
    diagonal     -> desired signal
    off-diagonal -> multi-user communication interference
    """
    received = torch.bmm(H_eff_H,W_C)
    received_power = torch.abs(received) ** 2

    signal_power = torch.diagonal(received_power,dim1=1,dim2=2)
    comm_interf_power = torch.sum(received_power,dim=2) - signal_power

    sinr = signal_power / (comm_interf_power + NOISE_POWER)
    rate = torch.log2(1.0 + sinr)

    return {
        "signal": signal_power,
        "comm_interf": comm_interf_power,
        "sinr": sinr,
        "rate": rate,
    }


def compute_pretrain_objective(H_eff_H,W_C,B):
    """
    Communication-only CommNet pretrain objective:

        L = -R_worst
            + lambda_I * mean[ log(1 + I_C / noise) ]
            + lambda_P * ( mean(P_C) / P_target - 1 )^2

    沒有RadarNet、沒有W_R、沒有radar interference、沒有sensing loss。
    """
    metrics = compute_comm_metrics(H_eff_H,W_C)

    # 每個slot先取worst UE，再對一個long-term block內L個slot平均，最後對batch平均。
    all_rate = metrics["rate"].reshape(B,LT_BLOCK_SIZE,UAV_COMM)
    worst_ue_per_slot = torch.min(all_rate,dim=2).values
    per_block_rate = torch.mean(worst_ue_per_slot,dim=1)
    nominal_rate = torch.mean(per_block_rate)

    # 直接壓低multi-user interference；除以noise後再取log，使loss具有合理數值尺度。
    comm_interf_loss = torch.mean(torch.log1p(metrics["comm_interf"] / NOISE_POWER))

    # 只要求batch平均W_C power接近0.63 W，不做per-sample hard normalization。
    sample_power = torch.sum(torch.abs(W_C) ** 2,dim=(1,2)).real
    mean_power = torch.mean(sample_power)
    power_target_loss = (mean_power / COMM_PRETRAIN_TARGET_POWER - 1.0) ** 2

    total_loss = -nominal_rate + COMM_INTERF_LOSS_WEIGHT * comm_interf_loss + COMM_POWER_TARGET_LOSS_WEIGHT * power_target_loss

    return total_loss,nominal_rate,comm_interf_loss,power_target_loss,mean_power,metrics


# ================================
# Main
# ================================
if __name__ == "__main__":
    theta_net = ThetaNet().to(DEVICE)
    comm_net = CommNet().to(DEVICE)

    optimizer = optim.Adam(comm_net.parameters(),lr=COMM_PRETRAIN_LEARNING_RATE)

    train_dataset_path = os.path.join(DATA_DIR,"dataset_train.npz")
    val_dataset_path = os.path.join(DATA_DIR,"dataset_val.npz")

    ris_only_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")
    comm_pretrain_ckpt = os.path.join(PRETRAIN_DIR,"comm_pretrain.ckpt")
    comm_pretrain_curve_path = os.path.join(PRETRAIN_DIR,"comm_pretrain_curves.npz")
    comm_pretrain_plot_path = os.path.join(PRETRAIN_DIR,"comm_pretrain_curves.png")

    os.makedirs(PRETRAIN_DIR,exist_ok=True)

    for required_path in [train_dataset_path,val_dataset_path,ris_only_ckpt]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"找不到檔案: {required_path}")


    # ================================
    # Load train / validation datasets
    # ================================
    with np.load(train_dataset_path) as data:
        train_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

    with np.load(val_dataset_path) as data:
        val_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

    train_trajectories,train_channels = train_dataset["h_dk"].shape[:2]
    val_trajectories,val_channels = val_dataset["h_dk"].shape[:2]


    # ================================
    # Frozen ThetaNet
    # ================================
    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    for parameter in theta_net.parameters():
        parameter.requires_grad_(False)


    # ================================
    # Fixed validation blocks
    # ================================
    val_channel_ids = np.arange(0,val_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
    val_trajectory_ids = np.repeat(np.arange(val_trajectories),len(val_channel_ids))
    val_channel_ids = np.tile(val_channel_ids,val_trajectories)
    B_val = len(val_channel_ids)

    with torch.no_grad():
        # ThetaNet只看每個long-term block的start-slot CSI。
        h_dk_val_start = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_val_start = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
        G_val_start = torch.as_tensor(val_dataset["G"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_val_start = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)

        theta_val = theta_net(h_dk_val_start,h_rk_val_start,G_val_start,g_dt_val_start)

        # 同一個long-term block內固定theta，但每個short-term slot使用自己的channel。
        val_block_ids = val_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]

        h_dk_val_block = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_val_block = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
        G_val_block = torch.as_tensor(val_dataset["G"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_val_block = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids[:,None],val_block_ids],dtype=torch.complex64,device=DEVICE)

        h_dk_val_block = h_dk_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
        h_rk_val_block = h_rk_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
        G_val_block = G_val_block.reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
        g_dt_val_block = g_dt_val_block.reshape(B_val*LT_BLOCK_SIZE,TX_ANT,1)

        theta_val_block = theta_val[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_val*LT_BLOCK_SIZE,RIS_UNIT)
        H_eff_H_val = theta_net.compute_effective_channel(h_dk_val_block,h_rk_val_block,G_val_block,theta_val_block)


    # ================================
    # Training information
    # ================================
    print("\n" + "=" * 90)
    print("[COMMNET COMMUNICATION-ONLY PRETRAIN]")
    print("=" * 90)
    print(f"Train trajectories          : {train_trajectories}")
    print(f"Validation trajectories     : {val_trajectories}")
    print(f"Train slots / trajectory    : {train_channels}")
    print(f"Validation slots / traj.    : {val_channels}")
    print(f"LT block size               : {LT_BLOCK_SIZE}")
    print(f"Epochs                      : {COMM_PRETRAIN_EPOCHS}")
    print(f"Batches per epoch           : {COMM_PRETRAIN_BATCHES}")
    print(f"Blocks per batch            : {COMM_PRETRAIN_BATCH_CHANNELS}")
    print(f"Learning rate               : {COMM_PRETRAIN_LEARNING_RATE}")
    print(f"W_C target mean power       : {COMM_PRETRAIN_TARGET_POWER:.6f} W")
    print(f"Comm interference weight    : {COMM_INTERF_LOSS_WEIGHT:.6f}")
    print(f"Power target loss weight    : {COMM_POWER_TARGET_LOSS_WEIGHT:.6f}")
    print("RZF / analytical teacher    : NOT USED")
    print("RadarNet / W_R              : NOT USED")
    print("Radar interference          : NOT USED")
    print("Sensing loss                : NOT USED")
    print(f"Checkpoint                  : {comm_pretrain_ckpt}")
    print("=" * 90)


    curves = {
        "train_loss": [],
        "val_loss": [],
        "train_nominal": [],
        "val_nominal": [],
        "train_comm_interf_loss": [],
        "val_comm_interf_loss": [],
        "train_power_target_loss": [],
        "val_power_target_loss": [],
        "train_W_C_power": [],
        "val_W_C_power": [],
        "train_signal_power": [],
        "val_signal_power": [],
        "train_comm_interf_power": [],
        "val_comm_interf_power": [],
    }

    best_val_loss = float("inf")
    best_val_rate = 0.0
    best_val_epoch = 0
    patience_counter = 0


    # ================================
    # CommNet pretraining
    # ================================
    for epoch in trange(COMM_PRETRAIN_EPOCHS,desc="CommNet Comm-only Pretrain"):
        comm_net.train()

        epoch_loss = []
        epoch_nominal = []
        epoch_comm_interf_loss = []
        epoch_power_target_loss = []
        epoch_power = []
        epoch_signal = []
        epoch_comm_interf = []

        for _ in range(COMM_PRETRAIN_BATCHES):
            # 一個batch抽COMM_PRETRAIN_BATCH_CHANNELS個long-term blocks。
            trajectory_ids = np.random.choice(train_trajectories,size=COMM_PRETRAIN_BATCH_CHANNELS,replace=True)
            channel_ids = np.random.choice(train_channels,size=COMM_PRETRAIN_BATCH_CHANNELS,replace=False)

            valid = channel_ids + LT_BLOCK_SIZE <= train_channels
            trajectory_ids = trajectory_ids[valid]
            channel_ids = channel_ids[valid]

            if len(channel_ids) == 0:
                continue

            B = len(channel_ids)

            # ThetaNet只看long-term block start-slot CSI。
            h_dk_start = torch.as_tensor(train_dataset["h_dk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_start = torch.as_tensor(train_dataset["h_rk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
            G_start = torch.as_tensor(train_dataset["G"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_start = torch.as_tensor(train_dataset["g_dt"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)

            # 同一long-term block內固定theta，每個short-term slot使用自己的channel。
            block_ids = channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]

            h_dk_block = torch.as_tensor(train_dataset["h_dk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_block = torch.as_tensor(train_dataset["h_rk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
            G_block = torch.as_tensor(train_dataset["G"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_block = torch.as_tensor(train_dataset["g_dt"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)

            h_dk_block = h_dk_block.reshape(B*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
            h_rk_block = h_rk_block.reshape(B*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
            G_block = G_block.reshape(B*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
            g_dt_block = g_dt_block.reshape(B*LT_BLOCK_SIZE,TX_ANT,1)

            with torch.no_grad():
                theta = theta_net(h_dk_start,h_rk_start,G_start,g_dt_start)
                theta_block = theta[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B*LT_BLOCK_SIZE,RIS_UNIT)
                H_eff_H = theta_net.compute_effective_channel(h_dk_block,h_rk_block,G_block,theta_block)

            optimizer.zero_grad(set_to_none=True)

            # CommNet介面保持與REG/ROB joint training完全一致。
            W_C = comm_net(H_eff_H,g_dt_block)

            loss,nominal,comm_interf_loss,power_target_loss,mean_power,metrics = compute_pretrain_objective(H_eff_H,W_C,B)

            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss.detach().cpu()))
            epoch_nominal.append(float(nominal.detach().cpu()))
            epoch_comm_interf_loss.append(float(comm_interf_loss.detach().cpu()))
            epoch_power_target_loss.append(float(power_target_loss.detach().cpu()))
            epoch_power.append(float(mean_power.detach().cpu()))
            epoch_signal.append(metrics["signal"].mean(dim=0).detach().cpu().numpy())
            epoch_comm_interf.append(metrics["comm_interf"].mean(dim=0).detach().cpu().numpy())

        if len(epoch_loss) == 0:
            raise RuntimeError("本epoch沒有有效training batch，請檢查LT_BLOCK_SIZE與dataset長度。")

        train_logs = {
            "loss": float(np.mean(epoch_loss)),
            "nominal": float(np.mean(epoch_nominal)),
            "comm_interf_loss": float(np.mean(epoch_comm_interf_loss)),
            "power_target_loss": float(np.mean(epoch_power_target_loss)),
            "power": float(np.mean(epoch_power)),
            "signal": np.mean(np.stack(epoch_signal,axis=0),axis=0),
            "comm_interf": np.mean(np.stack(epoch_comm_interf,axis=0),axis=0),
        }


        # ================================
        # Validation
        # ================================
        comm_net.eval()

        with torch.no_grad():
            W_C_val = comm_net(H_eff_H_val,g_dt_val_block)

            val_loss,val_nominal,val_comm_interf_loss,val_power_target_loss,val_mean_power,val_metrics = compute_pretrain_objective(H_eff_H_val,W_C_val,B_val)

            val_logs = {
                "loss": float(val_loss.cpu()),
                "nominal": float(val_nominal.cpu()),
                "comm_interf_loss": float(val_comm_interf_loss.cpu()),
                "power_target_loss": float(val_power_target_loss.cpu()),
                "power": float(val_mean_power.cpu()),
                "signal": val_metrics["signal"].mean(dim=0).cpu().numpy(),
                "comm_interf": val_metrics["comm_interf"].mean(dim=0).cpu().numpy(),
            }


        # ================================
        # Save curves
        # ================================
        curves["train_loss"].append(train_logs["loss"])
        curves["val_loss"].append(val_logs["loss"])
        curves["train_nominal"].append(train_logs["nominal"])
        curves["val_nominal"].append(val_logs["nominal"])
        curves["train_comm_interf_loss"].append(train_logs["comm_interf_loss"])
        curves["val_comm_interf_loss"].append(val_logs["comm_interf_loss"])
        curves["train_power_target_loss"].append(train_logs["power_target_loss"])
        curves["val_power_target_loss"].append(val_logs["power_target_loss"])
        curves["train_W_C_power"].append(train_logs["power"])
        curves["val_W_C_power"].append(val_logs["power"])
        curves["train_signal_power"].append(train_logs["signal"].copy())
        curves["val_signal_power"].append(val_logs["signal"].copy())
        curves["train_comm_interf_power"].append(train_logs["comm_interf"].copy())
        curves["val_comm_interf_power"].append(val_logs["comm_interf"].copy())

        np.savez(comm_pretrain_curve_path,**{key: np.asarray(value,dtype=np.float32) for key,value in curves.items()})


        # ================================
        # Best checkpoint / early stop
        # ================================
        if best_val_loss - val_logs["loss"] > EARLY_STOP_EPS:
            best_val_loss = val_logs["loss"]
            best_val_rate = val_logs["nominal"]
            best_val_epoch = epoch + 1
            patience_counter = 0

            comm_net.save_model(comm_pretrain_ckpt,verbose=False)
        else:
            patience_counter += 1


        print(
            f"\n[Epoch {epoch + 1:03d}/{COMM_PRETRAIN_EPOCHS}] "
            f"TrainLoss={train_logs['loss']:.6f} ValLoss={val_logs['loss']:.6f} | "
            f"TrainNominal={train_logs['nominal']:.4f} ValNominal={val_logs['nominal']:.4f} | "
            f"TrainInterfLoss={train_logs['comm_interf_loss']:.4f} ValInterfLoss={val_logs['comm_interf_loss']:.4f} | "
            f"TrainPower={train_logs['power']:.6f} W ValPower={val_logs['power']:.6f} W"
        )
        print(
            f"TrainSignal={np.array2string(train_logs['signal'],precision=3,suppress_small=False)} | "
            f"TrainCommInterf={np.array2string(train_logs['comm_interf'],precision=3,suppress_small=False)}"
        )
        print(
            f"ValSignal  ={np.array2string(val_logs['signal'],precision=3,suppress_small=False)} | "
            f"ValCommInterf  ={np.array2string(val_logs['comm_interf'],precision=3,suppress_small=False)}"
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
    # Plot saved curves
    # ================================
    curve_data = np.load(comm_pretrain_curve_path)
    epochs = np.arange(1,len(curve_data["train_loss"]) + 1)

    fig,axes = plt.subplots(2,2,figsize=(13,9))

    axes[0,0].plot(epochs,curve_data["train_loss"],label="Train Loss")
    axes[0,0].plot(epochs,curve_data["val_loss"],label="Validation Loss")
    axes[0,0].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Loss")
    axes[0,0].set_title("Communication-only Pretrain Loss")
    axes[0,0].grid(True,alpha=0.35)
    axes[0,0].legend()

    axes[0,1].plot(epochs,curve_data["train_nominal"],label="Train Worst-UE Rate")
    axes[0,1].plot(epochs,curve_data["val_nominal"],label="Validation Worst-UE Rate")
    axes[0,1].set_xlabel("Epoch")
    axes[0,1].set_ylabel("bps/Hz")
    axes[0,1].set_title("Worst-UE Rate")
    axes[0,1].grid(True,alpha=0.35)
    axes[0,1].legend()

    axes[1,0].plot(epochs,curve_data["train_comm_interf_loss"],label="Train Interference Loss")
    axes[1,0].plot(epochs,curve_data["val_comm_interf_loss"],label="Validation Interference Loss")
    axes[1,0].set_xlabel("Epoch")
    axes[1,0].set_ylabel("mean log(1 + I_C / noise)")
    axes[1,0].set_title("Communication Interference Loss")
    axes[1,0].grid(True,alpha=0.35)
    axes[1,0].legend()

    axes[1,1].plot(epochs,curve_data["train_W_C_power"],label="Train W_C Power")
    axes[1,1].plot(epochs,curve_data["val_W_C_power"],label="Validation W_C Power")
    axes[1,1].axhline(COMM_PRETRAIN_TARGET_POWER,linestyle="--",label="Target Power")
    axes[1,1].set_xlabel("Epoch")
    axes[1,1].set_ylabel("W")
    axes[1,1].set_title("CommNet Output Power")
    axes[1,1].grid(True,alpha=0.35)
    axes[1,1].legend()

    plt.tight_layout()
    plt.savefig(comm_pretrain_plot_path,dpi=300)
    plt.close()


    print("\n" + "=" * 90)
    print("[COMMNET COMMUNICATION-ONLY PRETRAIN FINISHED]")
    print("=" * 90)
    print(f"Best validation epoch      : {best_val_epoch}")
    print(f"Best validation loss       : {best_val_loss:.6f}")
    print(f"Best validation nominal    : {best_val_rate:.6f} bps/Hz")
    print(f"Checkpoint                 : {comm_pretrain_ckpt}")
    print(f"Curves                     : {comm_pretrain_curve_path}")
    print(f"Plot                       : {comm_pretrain_plot_path}")
    print("=" * 90)
