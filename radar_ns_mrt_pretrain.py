# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import mrt_in_H_eff_H_nullspace
from two_timescale_NN import RadarNet, ThetaNet


RADAR_PRETRAIN_EPOCHS = 1000
RADAR_PRETRAIN_LEARNING_RATE = 1e-3
RADAR_PRETRAIN_BATCHES = N_BATCHE
RADAR_PRETRAIN_BATCH_CHANNELS = ST_BATCH_SIZE

RADAR_PRETRAIN_TARGET_POWER = TRANSMIT_POWER_TOTAL / 2.0
RADAR_PRETRAIN_TARGET_SCALE = np.sqrt(RADAR_PRETRAIN_TARGET_POWER)

EARLY_STOP_EPS = 1e-7
PATIENCE = 50


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


theta_net = ThetaNet().to(DEVICE)
radar_net = RadarNet().to(DEVICE)

optimizer = optim.Adam(radar_net.parameters(),lr=RADAR_PRETRAIN_LEARNING_RATE)


train_dataset_path = os.path.join(DATA_DIR,"dataset_train.npz")
val_dataset_path = os.path.join(DATA_DIR,"dataset_val.npz")

ris_only_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")
radar_pretrain_ckpt = os.path.join(PRETRAIN_DIR,"radar_ns_mrt_pretrain.ckpt")
radar_pretrain_curve_path = os.path.join(PRETRAIN_DIR,"radar_ns_mrt_pretrain_curves.npz")
radar_pretrain_plot_path = os.path.join(PRETRAIN_DIR,"radar_ns_mrt_pretrain_curves.png")

os.makedirs(PRETRAIN_DIR,exist_ok=True)

for required_path in [train_dataset_path,val_dataset_path,ris_only_ckpt]:
    if not os.path.exists(required_path):
        raise FileNotFoundError(f"找不到檔案: {required_path}")


with np.load(train_dataset_path) as data:
    train_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

with np.load(val_dataset_path) as data:
    val_dataset = {key: data[key] for key in ["positions","h_dk","h_rk","G","g_dt"]}

train_trajectories, train_channels = train_dataset["h_dk"].shape[:2]
val_trajectories, val_channels = val_dataset["h_dk"].shape[:2]

val_channel_ids = np.arange(0,val_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)
val_trajectory_ids = np.repeat(np.arange(val_trajectories),len(val_channel_ids))
val_channel_ids = np.tile(val_channel_ids,val_trajectories)
B_val = len(val_channel_ids)


theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
theta_net.eval()

for parameter in theta_net.parameters():
    parameter.requires_grad_(False)


with torch.no_grad():
    # Validation固定使用0,L,2L,...作為long-term block start
    h_dk_val = torch.as_tensor(val_dataset["h_dk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
    h_rk_val = torch.as_tensor(val_dataset["h_rk"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
    G_val = torch.as_tensor(val_dataset["G"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)
    g_dt_val = torch.as_tensor(val_dataset["g_dt"][val_trajectory_ids,val_channel_ids],dtype=torch.complex64,device=DEVICE)

    theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)  # (B_val,N)

    val_block_ids = val_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]  # (B_val,L)

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
    W_R_target_val = RADAR_PRETRAIN_TARGET_SCALE * mrt_in_H_eff_H_nullspace(H_eff_H_val,g_dt_val_block)

    target_val_power = torch.mean(torch.sum(torch.abs(W_R_target_val) ** 2,dim=(1,2))).item()

    # 只供啟動資訊顯示：以固定non-overlapping blocks計算train target mean power
    train_target_power_sum = 0.0
    train_target_power_count = 0
    train_fixed_channel_ids = np.arange(0,train_channels-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)

    for trajectory_id in range(train_trajectories):
        B_train_fixed = len(train_fixed_channel_ids)
        train_fixed_trajectory_ids = np.full(B_train_fixed,trajectory_id,dtype=np.int64)

        h_dk_train = torch.as_tensor(train_dataset["h_dk"][train_fixed_trajectory_ids,train_fixed_channel_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_train = torch.as_tensor(train_dataset["h_rk"][train_fixed_trajectory_ids,train_fixed_channel_ids],dtype=torch.complex64,device=DEVICE)
        G_train = torch.as_tensor(train_dataset["G"][train_fixed_trajectory_ids,train_fixed_channel_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_train = torch.as_tensor(train_dataset["g_dt"][train_fixed_trajectory_ids,train_fixed_channel_ids],dtype=torch.complex64,device=DEVICE)

        theta_train = theta_net(h_dk_train,h_rk_train,G_train,g_dt_train)

        train_block_ids = train_fixed_channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]

        h_dk_train_block = torch.as_tensor(train_dataset["h_dk"][train_fixed_trajectory_ids[:,None],train_block_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_train_block = torch.as_tensor(train_dataset["h_rk"][train_fixed_trajectory_ids[:,None],train_block_ids],dtype=torch.complex64,device=DEVICE)
        G_train_block = torch.as_tensor(train_dataset["G"][train_fixed_trajectory_ids[:,None],train_block_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_train_block = torch.as_tensor(train_dataset["g_dt"][train_fixed_trajectory_ids[:,None],train_block_ids],dtype=torch.complex64,device=DEVICE)

        h_dk_train_block = h_dk_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
        h_rk_train_block = h_rk_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
        G_train_block = G_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
        g_dt_train_block = g_dt_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,TX_ANT,1)

        theta_train_block = theta_train[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_train_fixed*LT_BLOCK_SIZE,RIS_UNIT)

        H_eff_H_train_fixed = theta_net.compute_effective_channel(h_dk_train_block,h_rk_train_block,G_train_block,theta_train_block)
        W_R_target_train_fixed = RADAR_PRETRAIN_TARGET_SCALE * mrt_in_H_eff_H_nullspace(H_eff_H_train_fixed,g_dt_train_block)
        target_power = torch.sum(torch.abs(W_R_target_train_fixed) ** 2,dim=(1,2))

        train_target_power_sum += float(torch.sum(target_power).cpu())
        train_target_power_count += target_power.numel()

    target_train_power = train_target_power_sum / max(train_target_power_count,1)


print("\n" + "=" * 90)
print("[RADARNET NS-MRT MSE PRETRAIN]")
print("=" * 90)
print(f"Train channels              : {train_channels}")
print(f"Val channels                : {val_channels}")
print(f"Epochs                      : {RADAR_PRETRAIN_EPOCHS}")
print(f"Batches per epoch           : {RADAR_PRETRAIN_BATCHES}")
print(f"Batch channels              : {RADAR_PRETRAIN_BATCH_CHANNELS}")
print(f"Learning rate               : {RADAR_PRETRAIN_LEARNING_RATE}")
print(f"Train target mean power     : {target_train_power:.6f}")
print(f"Val target mean power       : {target_val_power:.6f}")
print(f"Checkpoint                  : {radar_pretrain_ckpt}")
print("=" * 90)


curves = {
    "train_mse": [],
    "val_mse": [],
    "train_alignment": [],
    "val_alignment": [],
    "train_pred_power": [],
    "val_pred_power": [],
}

best_val_mse = float("inf")
best_val_alignment = 0.0
best_val_epoch = 0
patience_counter = 0


for epoch in trange(RADAR_PRETRAIN_EPOCHS,desc="RadarNet NS-MRT Pretrain"):
    radar_net.train()

    epoch_mse = []
    epoch_alignment = []
    epoch_pred_power = []

    for _ in range(RADAR_PRETRAIN_BATCHES):
        # 一個batch抽RADAR_PRETRAIN_BATCH_CHANNELS個long-term blocks
        trajectory_ids = np.random.choice(train_trajectories,size=RADAR_PRETRAIN_BATCH_CHANNELS,replace=True)
        channel_ids = np.random.choice(train_channels,size=RADAR_PRETRAIN_BATCH_CHANNELS,replace=False)

        valid = channel_ids + LT_BLOCK_SIZE <= train_channels
        trajectory_ids = trajectory_ids[valid]
        channel_ids = channel_ids[valid]

        if len(channel_ids) == 0:
            continue

        B = len(channel_ids)

        # ThetaNet只看每個long-term block的start-slot CSI
        h_dk_hat = torch.as_tensor(train_dataset["h_dk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_hat = torch.as_tensor(train_dataset["h_rk"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
        G_hat = torch.as_tensor(train_dataset["G"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_hat = torch.as_tensor(train_dataset["g_dt"][trajectory_ids,channel_ids],dtype=torch.complex64,device=DEVICE)

        block_ids = channel_ids[:,None] + np.arange(LT_BLOCK_SIZE)[None,:]  # (B,L)

        h_dk_block = torch.as_tensor(train_dataset["h_dk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
        h_rk_block = torch.as_tensor(train_dataset["h_rk"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
        G_block = torch.as_tensor(train_dataset["G"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)
        g_dt_block = torch.as_tensor(train_dataset["g_dt"][trajectory_ids[:,None],block_ids],dtype=torch.complex64,device=DEVICE)

        h_dk_block = h_dk_block.reshape(B*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
        h_rk_block = h_rk_block.reshape(B*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
        G_block = G_block.reshape(B*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)
        g_dt_block = g_dt_block.reshape(B*LT_BLOCK_SIZE,TX_ANT,1)

        with torch.no_grad():
            theta = theta_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)  # (B,N)
            theta_block = theta[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B*LT_BLOCK_SIZE,RIS_UNIT)

            # 每個slot用自己的channel重新建立H_eff與NS-MRT target，固定的只有theta
            H_eff_H_batch = theta_net.compute_effective_channel(h_dk_block,h_rk_block,G_block,theta_block)
            W_R_target_batch = RADAR_PRETRAIN_TARGET_SCALE * mrt_in_H_eff_H_nullspace(H_eff_H_batch,g_dt_block)

        optimizer.zero_grad(set_to_none=True)

        W_R_pred = radar_net(H_eff_H_batch,g_dt_block)

        mse = torch.mean(torch.abs(W_R_pred - W_R_target_batch) ** 2)

        mse.backward()
        optimizer.step()

        with torch.no_grad():
            pred_power = torch.mean(torch.sum(torch.abs(W_R_pred) ** 2,dim=(1,2)))

            W_R_pred_dir = W_R_pred / torch.sqrt(
                torch.sum(torch.abs(W_R_pred) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12)
            )
            W_R_target_dir = W_R_target_batch / torch.sqrt(
                torch.sum(torch.abs(W_R_target_batch) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12)
            )
            alignment = torch.mean(
                torch.abs(torch.sum(torch.conj(W_R_target_dir) * W_R_pred_dir,dim=(1,2))) ** 2
            )

        epoch_mse.append(float(mse.detach().cpu()))
        epoch_alignment.append(float(alignment.detach().cpu()))
        epoch_pred_power.append(float(pred_power.detach().cpu()))

    train_logs = {
        "mse": float(np.mean(epoch_mse)),
        "alignment": float(np.mean(epoch_alignment)),
        "pred_power": float(np.mean(epoch_pred_power)),
    }


    radar_net.eval()

    with torch.no_grad():
        W_R_val_pred = radar_net(H_eff_H_val,g_dt_val_block)

        val_mse = torch.mean(torch.abs(W_R_val_pred - W_R_target_val) ** 2)
        val_pred_power = torch.mean(torch.sum(torch.abs(W_R_val_pred) ** 2,dim=(1,2)))

        W_R_val_pred_dir = W_R_val_pred / torch.sqrt(
            torch.sum(torch.abs(W_R_val_pred) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12)
        )
        W_R_target_val_dir = W_R_target_val / torch.sqrt(
            torch.sum(torch.abs(W_R_target_val) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12)
        )
        val_alignment = torch.mean(
            torch.abs(torch.sum(torch.conj(W_R_target_val_dir) * W_R_val_pred_dir,dim=(1,2))) ** 2
        )
    val_logs = {
        "mse": float(val_mse.detach().cpu()),
        "alignment": float(val_alignment.detach().cpu()),
        "pred_power": float(val_pred_power.detach().cpu()),
    }


    curves["train_mse"].append(train_logs["mse"])
    curves["val_mse"].append(val_logs["mse"])
    curves["train_alignment"].append(train_logs["alignment"])
    curves["val_alignment"].append(val_logs["alignment"])
    curves["train_pred_power"].append(train_logs["pred_power"])
    curves["val_pred_power"].append(val_logs["pred_power"])

    np.savez(
        radar_pretrain_curve_path,
        **{key: np.asarray(value,dtype=np.float32) for key,value in curves.items()},
    )


    if best_val_mse - val_logs["mse"] > EARLY_STOP_EPS:
        best_val_mse = val_logs["mse"]
        best_val_alignment = val_logs["alignment"]
        best_val_epoch = epoch + 1
        patience_counter = 0

        radar_net.save_model(radar_pretrain_ckpt,verbose=False)
    else:
        patience_counter += 1


    print(
        f"\n[Epoch {epoch + 1:03d}/{RADAR_PRETRAIN_EPOCHS}] "
        f"TrainMSE={train_logs['mse']:.6e} "
        f"ValMSE={val_logs['mse']:.6e} | "
        f"TrainAlign={train_logs['alignment']:.6f} "
        f"ValAlign={val_logs['alignment']:.6f} | "
        f"TrainPower={train_logs['pred_power']:.6f} "
        f"ValPower={val_logs['pred_power']:.6f}"
    )

    if patience_counter >= PATIENCE:
        print(
            f"\n[EARLY STOP] "
            f"patience_counter={patience_counter}, "
            f"PATIENCE={PATIENCE}, "
            f"best_val_mse={best_val_mse:.6e}"
        )
        break


curve_data = np.load(radar_pretrain_curve_path)
epochs = np.arange(1,len(curve_data["train_mse"]) + 1)

plt.figure(figsize=(10,6))
plt.plot(epochs,curve_data["train_mse"],label="Train MSE")
plt.plot(epochs,curve_data["val_mse"],label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("Complex MSE")
plt.yscale("log")
plt.title("RadarNet NS-MRT Pretrain")
plt.grid(True,which="both",alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(radar_pretrain_plot_path,dpi=300)
plt.close()


print("\n" + "=" * 90)
print("[RADARNET NS-MRT PRETRAIN FINISHED]")
print("=" * 90)
print(f"Best validation epoch      : {best_val_epoch}")
print(f"Best validation MSE        : {best_val_mse:.6e}")
print(f"Best validation alignment  : {best_val_alignment:.6f}")
print(f"Checkpoint                 : {radar_pretrain_ckpt}")
print(f"Curves                     : {radar_pretrain_curve_path}")
print(f"Plot                       : {radar_pretrain_plot_path}")
print("=" * 90)
