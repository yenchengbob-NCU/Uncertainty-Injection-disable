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


RADAR_PRETRAIN_EPOCHS = 500
RADAR_PRETRAIN_LEARNING_RATE = 1e-3
RADAR_PRETRAIN_BATCHES = N_BATCHE
RADAR_PRETRAIN_BATCH_CHANNELS = BATCH_CHANNELS

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


train_dataset = theta_net.load_channel_dataset(train_dataset_path,"train")
val_dataset = theta_net.load_channel_dataset(val_dataset_path,"val")

h_dk_train = torch.as_tensor(train_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
h_rk_train = torch.as_tensor(train_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
G_train = torch.as_tensor(train_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
g_dt_train = torch.as_tensor(train_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
G_val = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)

train_channels = h_dk_train.shape[0]
val_channels = h_dk_val.shape[0]


theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
theta_net.eval()

for parameter in theta_net.parameters():
    parameter.requires_grad_(False)


with torch.no_grad():
    theta_train = theta_net(h_dk_train,h_rk_train,G_train,g_dt_train)
    H_eff_H_train = theta_net.compute_effective_channel(h_dk_train,h_rk_train,G_train,theta_train)
    W_R_target_train = mrt_in_H_eff_H_nullspace(H_eff_H_train,g_dt_train)

    theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)
    H_eff_H_val = theta_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta_val)
    W_R_target_val = mrt_in_H_eff_H_nullspace(H_eff_H_val,g_dt_val)

    target_train_power = torch.mean(torch.sum(torch.abs(W_R_target_train) ** 2,dim=(1,2))).item()
    target_val_power = torch.mean(torch.sum(torch.abs(W_R_target_val) ** 2,dim=(1,2))).item()


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
        channel_ids = np.random.choice(train_channels,size=RADAR_PRETRAIN_BATCH_CHANNELS,replace=False)

        H_eff_H_batch = H_eff_H_train[channel_ids]
        g_dt_batch = g_dt_train[channel_ids]
        W_R_target_batch = W_R_target_train[channel_ids]

        optimizer.zero_grad(set_to_none=True)

        W_R_pred = radar_net(H_eff_H_batch,g_dt_batch)

        mse = torch.mean(torch.abs(W_R_pred - W_R_target_batch) ** 2)

        mse.backward()
        optimizer.step()

        with torch.no_grad():
            pred_power = torch.mean(torch.sum(torch.abs(W_R_pred) ** 2,dim=(1,2)))

            W_R_pred_dir = W_R_pred / torch.sqrt(
                torch.sum(torch.abs(W_R_pred) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12)
            )
            alignment = torch.mean(
                torch.abs(torch.sum(torch.conj(W_R_target_batch) * W_R_pred_dir,dim=(1,2))) ** 2
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
        W_R_val_pred = radar_net(H_eff_H_val,g_dt_val)

        val_mse = torch.mean(torch.abs(W_R_val_pred - W_R_target_val) ** 2)
        val_pred_power = torch.mean(torch.sum(torch.abs(W_R_val_pred) ** 2,dim=(1,2)))

        W_R_val_pred_dir = W_R_val_pred / torch.sqrt(
            torch.sum(torch.abs(W_R_val_pred) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12)
        )
        val_alignment = torch.mean(
            torch.abs(torch.sum(torch.conj(W_R_target_val) * W_R_val_pred_dir,dim=(1,2))) ** 2
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