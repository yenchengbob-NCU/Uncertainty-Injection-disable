# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import make_rzf_beamformer, RZF_LAMBDA
from two_timescale_NN import CommNet, ThetaNet


COMM_PRETRAIN_EPOCHS = 1500
COMM_PRETRAIN_LEARNING_RATE = 1e-3
COMM_PRETRAIN_BATCHES = N_BATCHE
COMM_PRETRAIN_BATCH_CHANNELS = ST_BATCH_SIZE

# RZF teacher 的總 communication beamformer power 固定為 P_T / 2。
# make_rzf_beamformer() 本身輸出 ||W_C||_F^2 = 1，
# 因此 beamformer amplitude 乘上 sqrt(P_T / 2) 後，teacher power = P_T / 2 = 0.63 W。
COMM_PRETRAIN_TARGET_POWER = TRANSMIT_POWER_TOTAL / 2.0
COMM_PRETRAIN_TARGET_SCALE = float(np.sqrt(COMM_PRETRAIN_TARGET_POWER))

EARLY_STOP_EPS = 1e-7
PATIENCE = 50


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


theta_net = ThetaNet().to(DEVICE)
comm_net = CommNet().to(DEVICE)
optimizer = optim.Adam(comm_net.parameters(),lr=COMM_PRETRAIN_LEARNING_RATE)


train_dataset_path = os.path.join(DATA_DIR,"dataset_train.npz")
val_dataset_path = os.path.join(DATA_DIR,"dataset_val.npz")

ris_only_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

# 使用 new 檔名，避免覆寫舊版 RZF_pretrain.py 的結果；訓練方法本身不變。
comm_pretrain_ckpt = os.path.join(PRETRAIN_DIR,"comm_rzf_pretrain_new.ckpt")
comm_pretrain_curve_path = os.path.join(PRETRAIN_DIR,"comm_rzf_pretrain_new_curves.npz")
comm_pretrain_plot_path = os.path.join(PRETRAIN_DIR,"comm_rzf_pretrain_new_diagnostics.png")

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


def compute_comm_diagnostics(H_eff_H,W_C,n_blocks):
    """
    純診斷函式，不參與 loss、不參與 backward。

    H_eff_H : (n_blocks*LT_BLOCK_SIZE,K,M)
    W_C     : (n_blocks*LT_BLOCK_SIZE,M,K)

    計算與正式 communication SINR 相同的 physical quantities：
        S_k   = |h_k^H w_k|^2
        I_C,k = sum_{j!=k}|h_k^H w_j|^2
        SINR  = S_k / (I_C,k + noise)
        R_k   = log2(1 + SINR)

    本 RZF pretrain 沒有 W_R，因此 denominator 沒有 radar interference。
    worst_ue_rate 採每個 short-term slot 先 min over users，再對全部 slots 平均。
    """
    received = torch.bmm(H_eff_H,W_C)                                  # (S,K,K)
    received_power = torch.abs(received) ** 2                           # (S,K,K)
    signal_power = torch.diagonal(received_power,dim1=1,dim2=2)         # (S,K)
    comm_interf_power = torch.sum(received_power,dim=2) - signal_power  # (S,K)
    sinr = signal_power / (comm_interf_power + NOISE_POWER)
    rate = torch.log2(1.0 + sinr)

    sample_power = torch.sum(torch.abs(W_C) ** 2,dim=(1,2)).real
    rate_block = rate.reshape(n_blocks,LT_BLOCK_SIZE,UAV_COMM)
    worst_ue_rate = torch.mean(torch.min(rate_block,dim=2).values)

    return {
        "power": torch.mean(sample_power),
        "signal": torch.mean(signal_power),
        "comm_interf": torch.mean(comm_interf_power),
        "interf_to_noise": torch.mean(comm_interf_power / NOISE_POWER),
        "sinr": torch.mean(sinr),
        "rate": torch.mean(rate),
        "worst_ue_rate": worst_ue_rate,
        "signal_user": torch.mean(signal_power,dim=0),
        "comm_interf_user": torch.mean(comm_interf_power,dim=0),
        "interf_to_noise_user": torch.mean(comm_interf_power / NOISE_POWER,dim=0),
        "sinr_user": torch.mean(sinr,dim=0),
        "rate_user": torch.mean(rate,dim=0),
    }


def metrics_to_float(metrics):
    """將 scalar 診斷結果轉成 Python float；per-user tensor 保留為 numpy array。"""
    return {
        "power": float(metrics["power"].detach().cpu()),
        "signal": float(metrics["signal"].detach().cpu()),
        "comm_interf": float(metrics["comm_interf"].detach().cpu()),
        "interf_to_noise": float(metrics["interf_to_noise"].detach().cpu()),
        "sinr": float(metrics["sinr"].detach().cpu()),
        "rate": float(metrics["rate"].detach().cpu()),
        "worst_ue_rate": float(metrics["worst_ue_rate"].detach().cpu()),
        "signal_user": metrics["signal_user"].detach().cpu().numpy().astype(np.float64),
        "comm_interf_user": metrics["comm_interf_user"].detach().cpu().numpy().astype(np.float64),
        "interf_to_noise_user": metrics["interf_to_noise_user"].detach().cpu().numpy().astype(np.float64),
        "sinr_user": metrics["sinr_user"].detach().cpu().numpy().astype(np.float64),
        "rate_user": metrics["rate_user"].detach().cpu().numpy().astype(np.float64),
    }


def mean_metric_logs(metric_logs):
    """將一個 epoch 內各 training batch 的診斷結果做平均。"""
    scalar_keys = ["power","signal","comm_interf","interf_to_noise","sinr","rate","worst_ue_rate"]
    vector_keys = ["signal_user","comm_interf_user","interf_to_noise_user","sinr_user","rate_user"]
    out = {key: float(np.mean([item[key] for item in metric_logs])) for key in scalar_keys}
    out.update({key: np.mean(np.stack([item[key] for item in metric_logs],axis=0),axis=0) for key in vector_keys})
    return out


with torch.no_grad():
    # Validation固定使用0,L,2L,...作為long-term block start。
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
    W_C_target_val = COMM_PRETRAIN_TARGET_SCALE * make_rzf_beamformer(H_eff_H_val,RZF_LAMBDA)

    target_val_metrics = metrics_to_float(compute_comm_diagnostics(H_eff_H_val,W_C_target_val,B_val))

    # 只供啟動資訊顯示：以固定non-overlapping blocks計算train target mean power。
    # 保留舊版行為，這裡只做 teacher power sanity check，不改 training sampling。
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

        h_dk_train_block = h_dk_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,TX_ANT,UAV_COMM)
        h_rk_train_block = h_rk_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,RIS_UNIT,UAV_COMM)
        G_train_block = G_train_block.reshape(B_train_fixed*LT_BLOCK_SIZE,RIS_UNIT,TX_ANT)

        theta_train_block = theta_train[:,None,:].expand(-1,LT_BLOCK_SIZE,-1).reshape(B_train_fixed*LT_BLOCK_SIZE,RIS_UNIT)
        H_eff_H_train_fixed = theta_net.compute_effective_channel(h_dk_train_block,h_rk_train_block,G_train_block,theta_train_block)
        W_C_target_train_fixed = COMM_PRETRAIN_TARGET_SCALE * make_rzf_beamformer(H_eff_H_train_fixed,RZF_LAMBDA)
        target_power = torch.sum(torch.abs(W_C_target_train_fixed) ** 2,dim=(1,2))

        train_target_power_sum += float(torch.sum(target_power).cpu())
        train_target_power_count += target_power.numel()

    target_train_power = train_target_power_sum / max(train_target_power_count,1)


print("\n" + "=" * 110)
print("[COMMNET RZF MSE PRETRAIN - NEW DIAGNOSTIC VERSION]")
print("=" * 110)
print(f"Train channels              : {train_channels}")
print(f"Val channels                : {val_channels}")
print(f"Epochs                      : {COMM_PRETRAIN_EPOCHS}")
print(f"Batches per epoch           : {COMM_PRETRAIN_BATCHES}")
print(f"Batch channels              : {COMM_PRETRAIN_BATCH_CHANNELS}")
print(f"Learning rate               : {COMM_PRETRAIN_LEARNING_RATE}")
print(f"RZF lambda                  : {RZF_LAMBDA:g}")
print(f"Target power setting        : {COMM_PRETRAIN_TARGET_POWER:.6f} W")
print(f"Train target mean power     : {target_train_power:.6f} W")
print(f"Val target mean power       : {target_val_metrics['power']:.6f} W")
print(f"Checkpoint                  : {comm_pretrain_ckpt}")
print("-")
print("[RZF TEACHER - FIXED VALIDATION SANITY CHECK]")
print(f"Worst-UE rate               : {target_val_metrics['worst_ue_rate']:.6f} bps/Hz")
print(f"Mean UE rate                : {target_val_metrics['rate']:.6f} bps/Hz")
print(f"Mean desired signal         : {target_val_metrics['signal']:.6e}")
print(f"Mean comm interference      : {target_val_metrics['comm_interf']:.6e}")
print(f"Mean I_C / noise            : {target_val_metrics['interf_to_noise']:.6f}")
for k in range(UAV_COMM):
    print(f"UE{k}: Signal={target_val_metrics['signal_user'][k]:.6e}  I_C={target_val_metrics['comm_interf_user'][k]:.6e}  I_C/noise={target_val_metrics['interf_to_noise_user'][k]:.6f}  Rate={target_val_metrics['rate_user'][k]:.6f}")
print("=" * 110)


curves = {
    "train_mse": [], "val_mse": [],
    "train_alignment": [], "val_alignment": [],
    "train_pred_power": [], "val_pred_power": [],
    "train_teacher_power": [], "val_teacher_power": [],
    "train_pred_signal": [], "val_pred_signal": [],
    "train_teacher_signal": [], "val_teacher_signal": [],
    "train_pred_comm_interf": [], "val_pred_comm_interf": [],
    "train_teacher_comm_interf": [], "val_teacher_comm_interf": [],
    "train_pred_interf_to_noise": [], "val_pred_interf_to_noise": [],
    "train_teacher_interf_to_noise": [], "val_teacher_interf_to_noise": [],
    "train_pred_mean_rate": [], "val_pred_mean_rate": [],
    "train_teacher_mean_rate": [], "val_teacher_mean_rate": [],
    "train_pred_worst_rate": [], "val_pred_worst_rate": [],
    "train_teacher_worst_rate": [], "val_teacher_worst_rate": [],
}

for k in range(UAV_COMM):
    curves[f"train_pred_signal_ue{k}"] = []
    curves[f"val_pred_signal_ue{k}"] = []
    curves[f"train_teacher_signal_ue{k}"] = []
    curves[f"val_teacher_signal_ue{k}"] = []
    curves[f"train_pred_comm_interf_ue{k}"] = []
    curves[f"val_pred_comm_interf_ue{k}"] = []
    curves[f"train_teacher_comm_interf_ue{k}"] = []
    curves[f"val_teacher_comm_interf_ue{k}"] = []
    curves[f"train_pred_rate_ue{k}"] = []
    curves[f"val_pred_rate_ue{k}"] = []
    curves[f"train_teacher_rate_ue{k}"] = []
    curves[f"val_teacher_rate_ue{k}"] = []


best_val_mse = float("inf")
best_val_alignment = 0.0
best_val_epoch = 0
best_val_pred_metrics = None
best_val_teacher_metrics = None
patience_counter = 0


for epoch in trange(COMM_PRETRAIN_EPOCHS,desc="CommNet RZF Pretrain NEW"):
    comm_net.train()

    epoch_mse = []
    epoch_alignment = []
    epoch_pred_metrics = []
    epoch_teacher_metrics = []

    for _ in range(COMM_PRETRAIN_BATCHES):
        # 保留舊版 sampling：一個batch抽COMM_PRETRAIN_BATCH_CHANNELS個long-term sliding windows。
        # 這裡沒有改成0,L,2L,...，避免改變原 RZF MSE 預訓練行為。
        trajectory_ids = np.random.choice(train_trajectories,size=COMM_PRETRAIN_BATCH_CHANNELS,replace=True)
        channel_ids = np.random.choice(train_channels,size=COMM_PRETRAIN_BATCH_CHANNELS,replace=False)

        valid = channel_ids + LT_BLOCK_SIZE <= train_channels
        trajectory_ids = trajectory_ids[valid]
        channel_ids = channel_ids[valid]

        if len(channel_ids) == 0:
            continue

        B = len(channel_ids)

        # ThetaNet只看每個long-term block的start-slot CSI。
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

            # 每個slot用自己的channel重新建立H_eff與RZF target，固定的只有theta。
            H_eff_H_batch = theta_net.compute_effective_channel(h_dk_block,h_rk_block,G_block,theta_block)
            W_C_target_batch = COMM_PRETRAIN_TARGET_SCALE * make_rzf_beamformer(H_eff_H_batch,RZF_LAMBDA)

        optimizer.zero_grad(set_to_none=True)

        # CommNet介面維持與joint training一致：輸入H_eff_H與當前slot的g_dt。
        W_C_pred = comm_net(H_eff_H_batch,g_dt_block)

        # ======================================================================
        # 唯一的 training loss。
        # 沒有 rate loss、沒有 interference loss、沒有 power penalty、沒有 sensing loss。
        # 所有 communication performance quantities 都只放在 no_grad 診斷，不會影響 gradient。
        # ======================================================================
        mse = torch.mean(torch.abs(W_C_pred - W_C_target_batch) ** 2)
        mse.backward()
        optimizer.step()

        with torch.no_grad():
            W_C_pred_dir = W_C_pred / torch.sqrt(torch.sum(torch.abs(W_C_pred) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12))
            W_C_target_dir = W_C_target_batch / torch.sqrt(torch.sum(torch.abs(W_C_target_batch) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12))
            alignment = torch.mean(torch.abs(torch.sum(torch.conj(W_C_target_dir) * W_C_pred_dir,dim=(1,2))) ** 2)

            pred_metrics = metrics_to_float(compute_comm_diagnostics(H_eff_H_batch,W_C_pred,B))
            teacher_metrics = metrics_to_float(compute_comm_diagnostics(H_eff_H_batch,W_C_target_batch,B))

        epoch_mse.append(float(mse.detach().cpu()))
        epoch_alignment.append(float(alignment.detach().cpu()))
        epoch_pred_metrics.append(pred_metrics)
        epoch_teacher_metrics.append(teacher_metrics)

    train_pred = mean_metric_logs(epoch_pred_metrics)
    train_teacher = mean_metric_logs(epoch_teacher_metrics)

    train_logs = {
        "mse": float(np.mean(epoch_mse)),
        "alignment": float(np.mean(epoch_alignment)),
    }


    comm_net.eval()

    with torch.no_grad():
        W_C_val_pred = comm_net(H_eff_H_val,g_dt_val_block)

        val_mse = torch.mean(torch.abs(W_C_val_pred - W_C_target_val) ** 2)
        W_C_val_pred_dir = W_C_val_pred / torch.sqrt(torch.sum(torch.abs(W_C_val_pred) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12))
        W_C_target_val_dir = W_C_target_val / torch.sqrt(torch.sum(torch.abs(W_C_target_val) ** 2,dim=(1,2),keepdim=True).real.clamp_min(1e-12))
        val_alignment = torch.mean(torch.abs(torch.sum(torch.conj(W_C_target_val_dir) * W_C_val_pred_dir,dim=(1,2))) ** 2)

        val_pred = metrics_to_float(compute_comm_diagnostics(H_eff_H_val,W_C_val_pred,B_val))
        val_teacher = target_val_metrics

    val_logs = {
        "mse": float(val_mse.detach().cpu()),
        "alignment": float(val_alignment.detach().cpu()),
    }


    curves["train_mse"].append(train_logs["mse"])
    curves["val_mse"].append(val_logs["mse"])
    curves["train_alignment"].append(train_logs["alignment"])
    curves["val_alignment"].append(val_logs["alignment"])

    curves["train_pred_power"].append(train_pred["power"])
    curves["val_pred_power"].append(val_pred["power"])
    curves["train_teacher_power"].append(train_teacher["power"])
    curves["val_teacher_power"].append(val_teacher["power"])

    curves["train_pred_signal"].append(train_pred["signal"])
    curves["val_pred_signal"].append(val_pred["signal"])
    curves["train_teacher_signal"].append(train_teacher["signal"])
    curves["val_teacher_signal"].append(val_teacher["signal"])

    curves["train_pred_comm_interf"].append(train_pred["comm_interf"])
    curves["val_pred_comm_interf"].append(val_pred["comm_interf"])
    curves["train_teacher_comm_interf"].append(train_teacher["comm_interf"])
    curves["val_teacher_comm_interf"].append(val_teacher["comm_interf"])

    curves["train_pred_interf_to_noise"].append(train_pred["interf_to_noise"])
    curves["val_pred_interf_to_noise"].append(val_pred["interf_to_noise"])
    curves["train_teacher_interf_to_noise"].append(train_teacher["interf_to_noise"])
    curves["val_teacher_interf_to_noise"].append(val_teacher["interf_to_noise"])

    curves["train_pred_mean_rate"].append(train_pred["rate"])
    curves["val_pred_mean_rate"].append(val_pred["rate"])
    curves["train_teacher_mean_rate"].append(train_teacher["rate"])
    curves["val_teacher_mean_rate"].append(val_teacher["rate"])

    curves["train_pred_worst_rate"].append(train_pred["worst_ue_rate"])
    curves["val_pred_worst_rate"].append(val_pred["worst_ue_rate"])
    curves["train_teacher_worst_rate"].append(train_teacher["worst_ue_rate"])
    curves["val_teacher_worst_rate"].append(val_teacher["worst_ue_rate"])

    for k in range(UAV_COMM):
        curves[f"train_pred_signal_ue{k}"].append(train_pred["signal_user"][k])
        curves[f"val_pred_signal_ue{k}"].append(val_pred["signal_user"][k])
        curves[f"train_teacher_signal_ue{k}"].append(train_teacher["signal_user"][k])
        curves[f"val_teacher_signal_ue{k}"].append(val_teacher["signal_user"][k])
        curves[f"train_pred_comm_interf_ue{k}"].append(train_pred["comm_interf_user"][k])
        curves[f"val_pred_comm_interf_ue{k}"].append(val_pred["comm_interf_user"][k])
        curves[f"train_teacher_comm_interf_ue{k}"].append(train_teacher["comm_interf_user"][k])
        curves[f"val_teacher_comm_interf_ue{k}"].append(val_teacher["comm_interf_user"][k])
        curves[f"train_pred_rate_ue{k}"].append(train_pred["rate_user"][k])
        curves[f"val_pred_rate_ue{k}"].append(val_pred["rate_user"][k])
        curves[f"train_teacher_rate_ue{k}"].append(train_teacher["rate_user"][k])
        curves[f"val_teacher_rate_ue{k}"].append(val_teacher["rate_user"][k])

    np.savez(comm_pretrain_curve_path,**{key: np.asarray(value,dtype=np.float32) for key,value in curves.items()})


    # Checkpoint selection 完全保留舊版：只依 validation complex MSE。
    # 診斷的 rate / interference / signal 不參與 checkpoint selection。
    if best_val_mse - val_logs["mse"] > EARLY_STOP_EPS:
        best_val_mse = val_logs["mse"]
        best_val_alignment = val_logs["alignment"]
        best_val_epoch = epoch + 1
        best_val_pred_metrics = {key: (value.copy() if isinstance(value,np.ndarray) else value) for key,value in val_pred.items()}
        best_val_teacher_metrics = {key: (value.copy() if isinstance(value,np.ndarray) else value) for key,value in val_teacher.items()}
        patience_counter = 0
        comm_net.save_model(comm_pretrain_ckpt,verbose=False)
    else:
        patience_counter += 1


    print(
        f"\n[Epoch {epoch + 1:03d}/{COMM_PRETRAIN_EPOCHS}] "
        f"TrainMSE={train_logs['mse']:.6e} ValMSE={val_logs['mse']:.6e} | "
        f"TrainAlign={train_logs['alignment']:.6f} ValAlign={val_logs['alignment']:.6f}"
    )
    print(
        f"  [TRAIN] Teacher: P={train_teacher['power']:.6f} RateWorst={train_teacher['worst_ue_rate']:.6f} "
        f"Signal={train_teacher['signal']:.6e} I_C={train_teacher['comm_interf']:.6e} I_C/N={train_teacher['interf_to_noise']:.4f} | "
        f"Pred: P={train_pred['power']:.6f} RateWorst={train_pred['worst_ue_rate']:.6f} "
        f"Signal={train_pred['signal']:.6e} I_C={train_pred['comm_interf']:.6e} I_C/N={train_pred['interf_to_noise']:.4f}"
    )
    print(
        f"  [VAL  ] Teacher: P={val_teacher['power']:.6f} RateWorst={val_teacher['worst_ue_rate']:.6f} "
        f"Signal={val_teacher['signal']:.6e} I_C={val_teacher['comm_interf']:.6e} I_C/N={val_teacher['interf_to_noise']:.4f} | "
        f"Pred: P={val_pred['power']:.6f} RateWorst={val_pred['worst_ue_rate']:.6f} "
        f"Signal={val_pred['signal']:.6e} I_C={val_pred['comm_interf']:.6e} I_C/N={val_pred['interf_to_noise']:.4f}"
    )
    for k in range(UAV_COMM):
        print(
            f"           UE{k} Teacher: S={val_teacher['signal_user'][k]:.6e} I_C={val_teacher['comm_interf_user'][k]:.6e} Rate={val_teacher['rate_user'][k]:.6f} | "
            f"Pred: S={val_pred['signal_user'][k]:.6e} I_C={val_pred['comm_interf_user'][k]:.6e} Rate={val_pred['rate_user'][k]:.6f}"
        )

    if patience_counter >= PATIENCE:
        print(f"\n[EARLY STOP] patience_counter={patience_counter}, PATIENCE={PATIENCE}, best_val_mse={best_val_mse:.6e}")
        break


curve_data = np.load(comm_pretrain_curve_path)
epochs = np.arange(1,len(curve_data["train_mse"]) + 1)

fig, axes = plt.subplots(2,3,figsize=(18,10))

axes[0,0].plot(epochs,curve_data["train_mse"],label="Train MSE")
axes[0,0].plot(epochs,curve_data["val_mse"],label="Validation MSE")
axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("Complex MSE")
axes[0,0].set_yscale("log")
axes[0,0].set_title("RZF Complex MSE")
axes[0,0].grid(True,which="both",alpha=0.35)
axes[0,0].legend()

axes[0,1].plot(epochs,curve_data["train_alignment"],label="Train alignment")
axes[0,1].plot(epochs,curve_data["val_alignment"],label="Validation alignment")
axes[0,1].set_xlabel("Epoch")
axes[0,1].set_ylabel("Alignment")
axes[0,1].set_title("Beamformer Alignment")
axes[0,1].grid(True,alpha=0.35)
axes[0,1].legend()

axes[0,2].plot(epochs,curve_data["train_pred_power"],label="Train NN")
axes[0,2].plot(epochs,curve_data["train_teacher_power"],label="Train RZF")
axes[0,2].plot(epochs,curve_data["val_pred_power"],label="Val NN")
axes[0,2].plot(epochs,curve_data["val_teacher_power"],label="Val RZF")
axes[0,2].set_xlabel("Epoch")
axes[0,2].set_ylabel("Power (W)")
axes[0,2].set_title("Communication Beamformer Power")
axes[0,2].grid(True,alpha=0.35)
axes[0,2].legend()

axes[1,0].plot(epochs,curve_data["train_pred_worst_rate"],label="Train NN")
axes[1,0].plot(epochs,curve_data["train_teacher_worst_rate"],label="Train RZF")
axes[1,0].plot(epochs,curve_data["val_pred_worst_rate"],label="Val NN")
axes[1,0].plot(epochs,curve_data["val_teacher_worst_rate"],label="Val RZF")
axes[1,0].set_xlabel("Epoch")
axes[1,0].set_ylabel("Worst-UE rate (bps/Hz)")
axes[1,0].set_title("Worst-UE Communication Rate")
axes[1,0].grid(True,alpha=0.35)
axes[1,0].legend()

axes[1,1].plot(epochs,curve_data["train_pred_comm_interf"],label="Train NN")
axes[1,1].plot(epochs,curve_data["train_teacher_comm_interf"],label="Train RZF")
axes[1,1].plot(epochs,curve_data["val_pred_comm_interf"],label="Val NN")
axes[1,1].plot(epochs,curve_data["val_teacher_comm_interf"],label="Val RZF")
axes[1,1].set_xlabel("Epoch")
axes[1,1].set_ylabel("Mean communication interference")
axes[1,1].set_yscale("log")
axes[1,1].set_title("Communication Interference")
axes[1,1].grid(True,which="both",alpha=0.35)
axes[1,1].legend()

axes[1,2].plot(epochs,curve_data["train_pred_signal"],label="Train NN")
axes[1,2].plot(epochs,curve_data["train_teacher_signal"],label="Train RZF")
axes[1,2].plot(epochs,curve_data["val_pred_signal"],label="Val NN")
axes[1,2].plot(epochs,curve_data["val_teacher_signal"],label="Val RZF")
axes[1,2].set_xlabel("Epoch")
axes[1,2].set_ylabel("Mean desired signal power")
axes[1,2].set_yscale("log")
axes[1,2].set_title("Desired Signal Power")
axes[1,2].grid(True,which="both",alpha=0.35)
axes[1,2].legend()

fig.suptitle("CommNet RZF MSE Pretrain - Teacher vs NN Diagnostics")
fig.tight_layout()
fig.savefig(comm_pretrain_plot_path,dpi=300)
plt.close(fig)


print("\n" + "=" * 110)
print("[COMMNET RZF PRETRAIN NEW FINISHED]")
print("=" * 110)
print(f"Best validation epoch      : {best_val_epoch}")
print(f"Best validation MSE        : {best_val_mse:.6e}")
print(f"Best validation alignment  : {best_val_alignment:.6f}")

if best_val_pred_metrics is not None:
    print("-")
    print("[BEST-MSE CHECKPOINT: VALIDATION RZF TEACHER vs NN]")
    print(f"RZF Teacher power          : {best_val_teacher_metrics['power']:.6f} W")
    print(f"NN Pred power              : {best_val_pred_metrics['power']:.6f} W")
    print(f"RZF Teacher worst-UE rate  : {best_val_teacher_metrics['worst_ue_rate']:.6f} bps/Hz")
    print(f"NN Pred worst-UE rate      : {best_val_pred_metrics['worst_ue_rate']:.6f} bps/Hz")
    print(f"RZF Teacher signal         : {best_val_teacher_metrics['signal']:.6e}")
    print(f"NN Pred signal             : {best_val_pred_metrics['signal']:.6e}")
    print(f"RZF Teacher comm interf    : {best_val_teacher_metrics['comm_interf']:.6e}")
    print(f"NN Pred comm interf        : {best_val_pred_metrics['comm_interf']:.6e}")
    print(f"RZF Teacher I_C/noise      : {best_val_teacher_metrics['interf_to_noise']:.6f}")
    print(f"NN Pred I_C/noise          : {best_val_pred_metrics['interf_to_noise']:.6f}")
    if best_val_teacher_metrics["comm_interf"] > 0.0:
        print(f"NN/RZF interference ratio  : {best_val_pred_metrics['comm_interf']/best_val_teacher_metrics['comm_interf']:.6f} x")
    for k in range(UAV_COMM):
        print(f"UE{k} RZF: S={best_val_teacher_metrics['signal_user'][k]:.6e} I_C={best_val_teacher_metrics['comm_interf_user'][k]:.6e} Rate={best_val_teacher_metrics['rate_user'][k]:.6f}")
        print(f"UE{k} NN : S={best_val_pred_metrics['signal_user'][k]:.6e} I_C={best_val_pred_metrics['comm_interf_user'][k]:.6e} Rate={best_val_pred_metrics['rate_user'][k]:.6f}")

print("-")
print("[IMPORTANT]")
print("Training objective remains PURE complex RZF MSE only.")
print("Signal / interference / SINR / rate / power are diagnostics only and never enter backward().")
print(f"Checkpoint                 : {comm_pretrain_ckpt}")
print(f"Curves                     : {comm_pretrain_curve_path}")
print(f"Diagnostics plot           : {comm_pretrain_plot_path}")
print("=" * 110)
