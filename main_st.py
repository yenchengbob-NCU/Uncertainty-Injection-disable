# -*- coding: utf-8 -*-
import os
import math
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# robust injection 分塊，避免顯存爆
INJECTION_CHUNK = 50

# ================================
# Helpers
# ================================
def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def normalize_channel_power(injected: torch.Tensor,reference: torch.Tensor,norm_dims,):
    """
    將 injected channel 的 power normalize 回 reference channel。
    用法:
        H_inj_norm = H_inj * ||H_ref|| / ||H_inj||
    目的:
        保留 estimated channel 的 large-scale power，
        讓 uncertainty injection 主要代表 channel direction / phase perturbation，
        避免 additive Gaussian error 改變平均 channel power。
    注意:
        norm_dims 不包含 batch 維度。
    """
    ref_norm = torch.linalg.vector_norm(
        reference,
        ord=2,
        dim=norm_dims,
        keepdim=True,
    )

    inj_norm = torch.linalg.vector_norm(
        injected,
        ord=2,
        dim=norm_dims,
        keepdim=True,
    ).clamp_min(1e-12)

    return injected * (ref_norm / inj_norm)


def format_float_for_path(value: float) -> str:
    """
    將 float 轉成適合資料夾名稱的字串。
    例：
        0.125 -> 0p125
        0.5   -> 0p5
        1.0   -> 1p0
        200.0 -> 200
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def build_st_paths(mode: str, penalty: float):
    """
    根據 mode 與 penalty 建立 ST run 的所有輸出路徑。
    REG : ST_sweep/REG_penalty_XXX/
    ROB : ST_sweep/ROB_penalty_XXX/
    """
    mode = mode.lower()

    if mode not in ("reg", "rob"):
        raise ValueError(f"mode 必須是 'reg' 或 'rob'，收到：{mode}")

    penalty_tag = format_float_for_path(penalty)

    if mode == "reg":
        run_name = f"REG_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm.ckpt"
        radar_ckpt_name = "short_radar.ckpt"
        curve_name = "shortterm_regular_curves.npy"
        curve_fig_name = "shortterm_regular_objective_curve.jpg"
        state_name = "shortterm_regular_train_state.pt"
    else:
        run_name = f"ROB_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm_robust.ckpt"
        radar_ckpt_name = "short_radar_robust.ckpt"
        curve_name = "shortterm_robust_curves.npy"
        curve_fig_name = "shortterm_robust_objective_curve.jpg"
        state_name = "shortterm_robust_train_state.pt"

    run_dir = os.path.join(ST_SWEEP_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")
    curve_dir = os.path.join(run_dir, "training_curves")
    state_dir = os.path.join(run_dir, "train_state")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "curve_dir": curve_dir,
        "state_dir": state_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, comm_ckpt_name),
        "radar_ckpt_path": os.path.join(ckpt_dir, radar_ckpt_name),
        "curve_path": os.path.join(curve_dir, curve_name),
        "curve_fig_path": os.path.join(curve_dir, curve_fig_name),
        "state_path": os.path.join(state_dir, state_name),
        "config_path": os.path.join(run_dir, "config.json"),
    }


def save_run_config(
    paths: dict,
    mode: str,
    penalty: float,
    train_injection_variance: float,
    injection_samples: int,
    train_dataset_path: str,
    val_dataset_path: str,
    longterm_ckpt_path: str,
):
    """
    儲存本次 ST run 設定。
    """
    config = {
        "mode": mode,
        "sensing_penalty": float(penalty),
        "train_injection_variance": float(train_injection_variance),
        "injection_samples": int(injection_samples),
        "outage_quantile": float(OUTAGE_QUANTILE),
        "noise_power": float(NOISE_POWER),
        "tx_power": float(TRANSMIT_POWER_TOTAL),
        "sensing_snr_threshold_db": float(SENSING_SNR_THRESHOLD_dB),
        "sensing_snr_threshold_linear": float(SENSING_SNR_THRESHOLD),
        "train_dataset_path": train_dataset_path,
        "val_dataset_path": val_dataset_path,
        "longterm_ckpt_path": longterm_ckpt_path,
        "comm_ckpt_path": paths["comm_ckpt_path"],
        "radar_ckpt_path": paths["radar_ckpt_path"],
        "curve_path": paths["curve_path"],
        "state_path": paths["state_path"],
    }

    with open(paths["config_path"], "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[CONFIG] 已儲存 run config：{paths['config_path']}")


def load_shortterm_dataset(npz_path: str, split_name: str):
    """
    載入 short-term training / validation dataset。

    回傳:
        dataset : dict
    """
    required_keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "pl_BS_TAR_BS",
        "st_h_dk_hat",
        "st_h_rk_hat",
        "st_G_hat",
        "st_g_dt_hat",
    ]

    with np.load(npz_path) as data:
        dataset = {key: data[key] for key in required_keys}

    print(f"[{split_name}] preloaded into RAM: {npz_path}")
    print(f"[{split_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{split_name}] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[{split_name}] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[{split_name}] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[{split_name}] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))
    print(f"[{split_name}] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")

    return dataset


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout,
):
    """
    給一組 UE layout，從 long-term net 取得固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        theta_lt, _, _ = longterm_net(ue_layout)

    return theta_lt.detach()


def extract_shortterm_batch(
    dataset,
    layout_id: int,          # 指定一個layout
    channel_ids: np.ndarray, # 一個array 可以取很多est channels 
):
    """
    取出指定 layout 與 channel ids 的 ets channels
    並轉為torch tensor準備輸入NN
    """
    return {
        "pl_BS_UE": dataset["pl_BS_UE"][layout_id],
        "pl_BS_RIS_UE": dataset["pl_BS_RIS_UE"][layout_id],
        "pl_BS_TAR_BS": dataset["pl_BS_TAR_BS"][layout_id],

        "h_dk_hat": torch.as_tensor(
            dataset["st_h_dk_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "h_rk_hat": torch.as_tensor(
            dataset["st_h_rk_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "G_hat": torch.as_tensor(
            dataset["st_G_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "g_dt_hat": torch.as_tensor(
            dataset["st_g_dt_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
    }


def moving_average(x: np.ndarray, window: int):
    """
    計算 moving average。
    輸入:
        x      : 1D numpy array
        window : moving average window size

    回傳:
        x_ma : moving averaged array
    """
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def save_training_state(
    state_path: str,
    comm_net,
    radar_net,
    optimizer,
    best_val_obj: float,
    finished_epoch: int,
):
    torch.save(
        {
            "comm_model_state_dict": comm_net.state_dict(),
            "radar_model_state_dict": radar_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_obj": best_val_obj,
            "finished_epoch": finished_epoch,
        },
        state_path
    )


def load_training_state(
    state_path: str,
    comm_net,
    radar_net,
    optimizer,
):
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"找不到 training state，無法 resume：{state_path}"
        )

    checkpoint = torch.load(
        state_path,
        map_location=DEVICE
    )

    comm_net.load_state_dict(checkpoint["comm_model_state_dict"])
    radar_net.load_state_dict(checkpoint["radar_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_obj = float(checkpoint["best_val_obj"])
    finished_epoch = int(checkpoint["finished_epoch"])

    return best_val_obj, finished_epoch


# ================================
# Short-term regular objective
# ================================
def forward_shortterm_regular_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,               # 估測通道
    sensing_loss_weight: float,
):
    """
    計算 regular short-term objective
    流程:
        1. fixed theta_LT
        2. estimated channels -> W_C, W_R
        3. W_C, W_R 經過 TX power normalization
        4. 在 estimated channels 上計算 sum-rate 與 sensing SNR penalty
    回傳:
        objective : torch.Tensor
        logs      : dict
    """
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE     = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B = h_dk_hat.shape[0]           # 一個layout 抓多少 估測通道 出來 
    theta_batch = theta_fixed.expand(B, RIS_UNIT) # 將theta 複製B份

    W_C = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat,theta_batch)    # 輸入網路
    W_R = radar_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat,theta_batch)   # 輸入網路
    
    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)                # beamformer 功率map到TX power max

    sumrate_mean = comm_net.compute_sum_rate(
        h_dk=h_dk_hat,      # 使用估測通道
        h_rk=h_rk_hat,      # 使用估測通道
        G=G_hat,            # 使用估測通道
        theta=theta_batch,
        W_R=W_R,
        W_C=W_C,
        pl_BS_UE=pl_BS_UE,
        pl_BS_RIS_UE=pl_BS_RIS_UE,
    ).mean()                # 將單一layout [ST_BATCH_EST_CHANNELS_PER_LAYOUT] 筆估測通道的rate平均

    sense_snr = comm_net.compute_sense_snr(
        g_dt=g_dt_hat,      # 使用估測通道
        W_R=W_R,
        W_C=W_C,
        pl_BS_TAR_BS=pl_BS_TAR_BS,
    ).real                  # 算出[ST_BATCH_EST_CHANNELS_PER_LAYOUT]筆SNR

    snr_penalty = torch.clamp(
        SENSING_SNR_THRESHOLD - sense_snr,
        min=0.0
    ).mean()                # 每一筆SNR算出懲罰後再平均

    objective = (sumrate_mean- sensing_loss_weight * snr_penalty)   # 目標函數

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_snr_mean_db": (10.0 * torch.log10(sense_snr.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "objective": objective.detach(),
    }

    return objective, logs


# ================================
# Short-term robust objective
# ================================
def forward_shortterm_robust_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,               # 估測通道
    sensing_loss_weight: float,
    injection_samples: int,
    train_injection_variance: float,
):
    """
    計算 robust short-term objective
    流程:
        1. NN input 使用原始 estimated channels + fixed theta_LT
        2. 用原始 estimated channels 產生 W_C, W_R
        3. W_C, W_R 經過 TX power normalization
        4. 複製 estimated channels 並加入 uncertainty injection
        5. 在 injected channels 上計算 mean sum-rate 與 VaR-SNR penalty
    回傳:
        objective : torch.Tensor
        logs      : dict
    """
    h_dk_hat = batch_data["h_dk_hat"] # shape = (B, M, K) B 一個layout 抓多少 估測通道 出來
    h_rk_hat = batch_data["h_rk_hat"] # shape = (B, N, K) N RIS 被動元件數
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE     = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B, M, K = h_dk_hat.shape            
    N = h_rk_hat.shape[1]               
    L = int(injection_samples)          # INJ 通道數量

    theta_batch = theta_fixed.expand(B, RIS_UNIT) # 將theta 複製B份

    W_C = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat,theta_batch)    # 輸入網路
    W_R = radar_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat,theta_batch)   # 輸入網路

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)                # beamformer 功率map到TX power max

    # ===利用CHUNK節省記憶體，等效於INJ L 次 只是拆成小塊拼裝===

    sumrate_chunks  = []    # 新增
    snr_chunks      = []    # 

    for s0 in range(0, L, INJECTION_CHUNK):
        # 將原本 L次的注入分成 INJECTION_CHUNK計算 最後在加起來
        s = min(INJECTION_CHUNK, L - s0)

        # 使用 .unsqueeze(1) 插入新維度 供INJ用
        # 使用 .expand       複製通道成s份
        # 使用 .reshape      讓後續計算方便使用
        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        # 注入不確定
        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape,train_injection_variance,DEVICE,h_dk_rep.dtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape,train_injection_variance,DEVICE,h_rk_rep.dtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,train_injection_variance,DEVICE,G_rep.dtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape,train_injection_variance,DEVICE,g_dt_rep.dtype)

        # normalize injection channel power
        h_dk_inj = normalize_channel_power(h_dk_inj, h_dk_rep, norm_dims=1)
        h_rk_inj = normalize_channel_power(h_rk_inj, h_rk_rep, norm_dims=1)
        G_inj    = normalize_channel_power(G_inj,    G_rep,    norm_dims=(1, 2))
        g_dt_inj = normalize_channel_power(g_dt_inj, g_dt_rep, norm_dims=(1, 2))

        # 複製theta
        theta_rep = theta_batch.unsqueeze(1).expand(B, s, N).reshape(B * s, N)
        radar_streams = W_R.shape[2]

        # 複製W_C & W_R
        W_C_rep = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        W_R_rep = W_R.unsqueeze(1).expand(B, s, M, radar_streams).reshape(B * s, M, radar_streams)

        sinrs = comm_net.compute_comm_sinrs(    # shape = (B*s,K)
            h_dk=h_dk_inj,
            h_rk=h_rk_inj,
            G=G_inj,
            theta=theta_rep,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE,
        )

        rates = comm_net.compute_rates(sinrs)
        sum_rate = rates.sum(dim=1)             # 將K個user rate 加起來

        sense_snr = comm_net.compute_sense_snr( # shape = (B*s,)
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real

        # reshape 回 (B, s)
        sumrate_chunks.append(sum_rate.reshape(B, s))   # 每次增加 chunk 筆資料
        snr_chunks.append(sense_snr.reshape(B, s))      # 每次增加 chunk 筆資料

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)  # chunk 合併 shape(B,L)
    snr_samples = torch.cat(snr_chunks, dim=1)          # chunk 合併 shape(B,L)

    # ===利用CHUNK節省記憶體，等效於INJ L 次 只是拆成小塊拼裝====

    sumrate_mean_per_sample = sumrate_samples.mean(dim=1)   # shape(B,) 每一筆 estimated channel 對 L 次 injection 的平均 sum-rate
    sumrate_mean = sumrate_mean_per_sample.mean()           # scalar 再對 B 筆 estimated channels 平均

    q = float(OUTAGE_QUANTILE)
    k = max(1, int(np.ceil(q * L)))

    snr_var     = torch.kthvalue(snr_samples,k=k,dim=1).values # 取5%

    snr_penalty = torch.clamp(SENSING_SNR_THRESHOLD - snr_var,min=0.0).mean()

    objective = (sumrate_mean- sensing_loss_weight * snr_penalty)

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_var_snr_mean_db": (
            10.0 * torch.log10(snr_var.clamp_min(1e-12))
        ).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "objective": objective.detach(),
    }

    return objective, logs


# ================================
# Short-term validation
# ================================
@torch.no_grad()
def validate_shortterm_regular(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,                # 估測通道
    sensing_loss_weight: float,
):
    """
    使用全部 validation layouts
    並且每個 layout 使用全部 ST estimated channels
    """
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0] # val date layout數

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0

    for layout_id in range(n_val_layouts):
        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net,ue_layout) # 拿到RIS向量

        n_channels = val_dataset["st_h_dk_hat"][layout_id].shape[0]         # 該layout下est channel數
        channel_ids = np.arange(n_channels)                                 # 建立channel ID

        batch_data = extract_shortterm_batch(                               # 取得該layout所有est channels
            val_dataset,
            layout_id,
            channel_ids
        )

        obj, logs = forward_shortterm_regular_objective(                    # 輸入網路
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            sensing_loss_weight=sensing_loss_weight,
        )

        total_obj += float(obj.detach().cpu())                              # 將單一layout下的所有est channels所的到的OBJ再根據layout數平均
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_snr_db += float(logs["sense_snr_mean_db"].cpu())

    return {
        "objective": total_obj / n_val_layouts,
        "sum_rate_mean": total_sumrate / n_val_layouts,
        "sense_snr_mean_db": total_snr_db / n_val_layouts,
        "num_val_layouts": n_val_layouts,
        "channels_per_layout": val_dataset["st_h_dk_hat"].shape[1],
    }


@torch.no_grad()
def validate_shortterm_robust(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,                # 估測通道
    sensing_loss_weight: float,
    injection_samples: int,
    train_injection_variance: float,
):
    """
    使用全部 validation layouts
    並且每個 layout 使用全部 ST estimated channels
    ROB validation 會在每筆 estimated channel 上做 uncertainty injection
    """
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0]                      # val data layout數

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0

    for layout_id in range(n_val_layouts):
        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout) # 拿到RIS向量

        n_channels = val_dataset["st_h_dk_hat"][layout_id].shape[0]          # 該layout下est channel數
        channel_ids = np.arange(n_channels)                                  # 建立channel ID

        batch_data = extract_shortterm_batch(                                # 取得該layout所有est channels
            val_dataset,
            layout_id,
            channel_ids
        )

        obj, logs = forward_shortterm_robust_objective(                      # 輸入網路
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            sensing_loss_weight=sensing_loss_weight,
            injection_samples=injection_samples,
            train_injection_variance=train_injection_variance,
        )

        total_obj += float(obj.detach().cpu())                               # 將單一layout下所有est channels的OBJ再根據layout數平均
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_snr_db += float(logs["sense_var_snr_mean_db"].cpu())

    return {
        "objective": total_obj / n_val_layouts,
        "sum_rate_mean": total_sumrate / n_val_layouts,
        "sense_var_snr_mean_db": total_snr_db / n_val_layouts,
        "num_val_layouts": n_val_layouts,
        "channels_per_layout": val_dataset["st_h_dk_hat"].shape[1],
    }


# ================================
# Train regular
# ================================
def train_shortterm_regular(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    train_dataset: dict,            # Train 估測通道
    val_dataset: dict,              # Val   估測通道
    paths: dict,                    # 儲存路徑
    sensing_loss_weight: float,
    resume: bool = False,
):
    """
    訓練 short-term regular networks。
    """
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=REG_LEARNING_RATE
    )

    curve_path = paths["curve_path"]
    state_path = paths["state_path"]

    if resume:
        print("[INFO] Resume short-term regular training state ...")
        best_val_obj, finished_epoch = load_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
        )

        curves = np.load(curve_path).tolist()
        start_epoch = finished_epoch + 1
        end_epoch = finished_epoch + REG_EPOCHS
    else:
        best_val_obj = -np.inf
        curves = []
        start_epoch = 1
        end_epoch = REG_EPOCHS

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    n_train_layouts = train_dataset["ue_layouts"].shape[0]  # layout 數量

    for ep in trange(start_epoch, end_epoch + 1, desc="ShortTerm-Regular"): # 訓練流程
        comm_net.train()
        radar_net.train()

        # 清空 OBJ
        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            # 在train data 中抽出 "ST_BATCH_LAYOUTS" 組 layout
            layout_ids = np.random.choice(n_train_layouts,size=ST_BATCH_LAYOUTS,replace=False)

            optimizer.zero_grad(set_to_none=True)

            mb_obj = 0.0
            mb_sumrate = 0.0
            mb_snr_db = 0.0
            # "ST_BATCH_LAYOUTS" 組 layout 每一組 算一個 OBJ
            for layout_id in layout_ids:
                layout_id = int(layout_id)

                ue_layout = train_dataset["ue_layouts"][layout_id]
                theta_fixed = get_fixed_theta_from_longterm(longterm_net,ue_layout)

                n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]   # 這layout下有多少組est channels
                # 在該 layout 中抽出 "ST_BATCH_EST_CHANNELS_PER_LAYOUT" 組 est channels
                channel_ids = np.random.choice( n_pool,size=ST_BATCH_EST_CHANNELS_PER_LAYOUT,replace=False)

                batch_data = extract_shortterm_batch(                       # 取得該layout所有"被選中"的est channels
                    train_dataset,
                    layout_id,
                    channel_ids
                )

                obj, logs = forward_shortterm_regular_objective(
                    comm_net=comm_net,
                    radar_net=radar_net,
                    theta_fixed=theta_fixed,
                    batch_data=batch_data,
                    sensing_loss_weight=sensing_loss_weight,
                )

                loss = -obj / ST_BATCH_LAYOUTS                              # OBJ除ST_BATCH_LAYOUTS 因為會累加ST_BATCH_LAYOUTS個 OBJ
                loss.backward()

                mb_obj += float(obj.detach().cpu()) / ST_BATCH_LAYOUTS
                mb_sumrate += float(logs["sum_rate_mean"].cpu()) / ST_BATCH_LAYOUTS
                mb_snr_db += float(logs["sense_snr_mean_db"].cpu()) / ST_BATCH_LAYOUTS

            optimizer.step()                                                # 累積ST_BATCH_LAYOUTS次梯度 一次更新

            train_obj_ep += mb_obj / MINIBATCHES                            # 紀錄該epoch資料 經過 ST_BATCH_EST_CHANNELS_PER_LAYOUT&ST_BATCH_LAYOUTS&MINIBATCHES 平均
            train_sumrate_ep += mb_sumrate / MINIBATCHES
            train_snr_db_ep += mb_snr_db / MINIBATCHES

        val_logs = validate_shortterm_regular(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
            sensing_loss_weight=sensing_loss_weight,
        )

        # 增加curve曲線資料
        curves.append([train_obj_ep,val_logs["objective"],train_sumrate_ep,val_logs["sum_rate_mean"],train_snr_db_ep,val_logs["sense_snr_mean_db"],])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[ST-REG Epoch {ep:03d}/{end_epoch}]\n"
            f"  TrainObj      = {train_obj_ep: .4e} | "
            f"TrainSumRate  = {train_sumrate_ep: .4e} | "
            f"TrainSNR(dB)  = {train_snr_db_ep: .3f} |\n"
            f"  ValObj        = {val_logs['objective']: .4e} | "
            f"ValSumRate    = {val_logs['sum_rate_mean']: .4e} | "
            f"ValSNR(dB)    = {val_logs['sense_snr_mean_db']: .3f} |"
        )

        if val_logs["objective"] > best_val_obj:
            best_val_obj = val_logs["objective"]
            comm_net.save_model(
                path=paths["comm_ckpt_path"],
                verbose=False
            )
            radar_net.save_model(
                path=paths["radar_ckpt_path"],
                verbose=False
            )

        save_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
            best_val_obj=best_val_obj,
            finished_epoch=ep,
        )

    print(f"[Short-Reg] best ValObj = {best_val_obj:.4e}")

    comm_net.load_model(path=paths["comm_ckpt_path"],verbose=True)
    radar_net.load_model(path=paths["radar_ckpt_path"],verbose=True)


# ================================
# Train robust
# ================================
def train_shortterm_robust(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    train_dataset: dict,            # Train 估測通道
    val_dataset: dict,              # Val   估測通道
    paths: dict,                    # 儲存路徑
    sensing_loss_weight: float,
    train_injection_variance: float,
    injection_samples: int,
    resume: bool = False,
):
    """
    訓練 short-term robust networks。
    """
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=ROB_LEARNING_RATE
    )

    curve_path = paths["curve_path"]
    state_path = paths["state_path"]

    if resume:
        print("[INFO] Resume short-term robust training state ...")
        best_val_obj, finished_epoch = load_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
        )

        curves = np.load(curve_path).tolist()
        start_epoch = finished_epoch + 1
        end_epoch = finished_epoch + ROB_EPOCHS
    else:
        best_val_obj = -np.inf
        curves = []
        start_epoch = 1
        end_epoch = ROB_EPOCHS

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    n_train_layouts = train_dataset["ue_layouts"].shape[0]  # layout 數量

    for ep in trange(start_epoch, end_epoch + 1, desc="ShortTerm-Robust"): # 訓練流程
        comm_net.train()
        radar_net.train()

        # 清空 OBJ
        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            # 在train data 中抽出 "ST_BATCH_LAYOUTS" 組 layout
            layout_ids = np.random.choice(n_train_layouts,size=ST_BATCH_LAYOUTS,replace=False)

            optimizer.zero_grad(set_to_none=True)

            mb_obj = 0.0
            mb_sumrate = 0.0
            mb_snr_db = 0.0

            # "ST_BATCH_LAYOUTS" 組 layout 每一組 算一個 ROB OBJ
            for layout_id in layout_ids:
                layout_id = int(layout_id)

                ue_layout = train_dataset["ue_layouts"][layout_id]
                theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

                n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]   # 這layout下有多少組est channels

                # 在該 layout 中抽出 "ST_BATCH_EST_CHANNELS_PER_LAYOUT" 組 est channels
                channel_ids = np.random.choice(n_pool,size=ST_BATCH_EST_CHANNELS_PER_LAYOUT,replace=False)

                batch_data = extract_shortterm_batch(                       # 取得該layout所有"被選中"的est channels
                    train_dataset,
                    layout_id,
                    channel_ids
                )

                obj, logs = forward_shortterm_robust_objective(
                    comm_net=comm_net,
                    radar_net=radar_net,
                    theta_fixed=theta_fixed,
                    batch_data=batch_data,
                    sensing_loss_weight=sensing_loss_weight,
                    injection_samples=injection_samples,
                    train_injection_variance=train_injection_variance,
                )

                loss = -obj / ST_BATCH_LAYOUTS                              # OBJ除ST_BATCH_LAYOUTS 因為會累加ST_BATCH_LAYOUTS個 OBJ
                loss.backward()

                mb_obj += float(obj.detach().cpu()) / ST_BATCH_LAYOUTS
                mb_sumrate += float(logs["sum_rate_mean"].cpu()) / ST_BATCH_LAYOUTS
                mb_snr_db += float(logs["sense_var_snr_mean_db"].cpu()) / ST_BATCH_LAYOUTS

            optimizer.step()                                                # 累積ST_BATCH_LAYOUTS次梯度 一次更新

            train_obj_ep += mb_obj / MINIBATCHES                            # 紀錄該epoch資料 經過 ST_BATCH_EST_CHANNELS_PER_LAYOUT&ST_BATCH_LAYOUTS&MINIBATCHES 平均
            train_sumrate_ep += mb_sumrate / MINIBATCHES
            train_snr_db_ep += mb_snr_db / MINIBATCHES

        val_logs = validate_shortterm_robust(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
            sensing_loss_weight=sensing_loss_weight,
            injection_samples=injection_samples,
            train_injection_variance=train_injection_variance,
        )

        # 增加curve曲線資料
        curves.append([
            train_obj_ep,
            val_logs["objective"],
            train_sumrate_ep,
            val_logs["sum_rate_mean"],
            train_snr_db_ep,
            val_logs["sense_var_snr_mean_db"],
        ])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[ST-ROB Epoch {ep:03d}/{end_epoch}]\n"
            f"  TrainObj          = {train_obj_ep: .4e} | "
            f"TrainSumRate      = {train_sumrate_ep: .4e} | "
            f"TrainVaR-SNR(dB)  = {train_snr_db_ep: .3f} |\n"
            f"  ValObj            = {val_logs['objective']: .4e} | "
            f"ValSumRate        = {val_logs['sum_rate_mean']: .4e} | "
            f"ValVaR-SNR(dB)    = {val_logs['sense_var_snr_mean_db']: .3f} |"
        )

        if val_logs["objective"] > best_val_obj:
            best_val_obj = val_logs["objective"]
            comm_net.save_model(path=paths["comm_ckpt_path"],verbose=False)
            radar_net.save_model(path=paths["radar_ckpt_path"],verbose=False)

        save_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
            best_val_obj=best_val_obj,
            finished_epoch=ep,
        )

    print(f"[Short-Rob] best ValObj = {best_val_obj:.4e}")

    comm_net.load_model(path=paths["comm_ckpt_path"], verbose=True)
    radar_net.load_model(path=paths["radar_ckpt_path"], verbose=True)

# ================================
# Plot short-term objective curve
# ================================
def plot_shortterm_objective_curve(
    mode: str,
    penalty: float,
    paths: dict,
):
    """
    依照 mode + penalty 讀取對應的 short-term curve 並畫圖。
    """
    mode = mode.lower()
    curve_path = paths["curve_path"]

    if not os.path.exists(curve_path):
        raise FileNotFoundError(f"找不到 curve 檔案：{curve_path}")

    curves = np.load(curve_path)

    start_idx = 10
    window = PLOT_MOVING_AVG_WINDOW

    if curves.shape[0] <= start_idx + window:
        start_idx = 0
        window = max(1, min(window, curves.shape[0]))

    epochs = np.arange(start_idx + 1, curves.shape[0] + 1)

    train_obj = curves[start_idx:, 0]
    val_obj   = curves[start_idx:, 1]

    train_obj_ma = moving_average(train_obj, window)
    val_obj_ma   = moving_average(val_obj, window)

    ma_epochs = epochs[window - 1:]

    fig_path = paths["curve_fig_path"]

    if mode == "reg":
        title = "Short-term Regular Training / Validation Objective"
        train_label = "REG Train Objective"
        val_label = "REG Validation Objective"
    elif mode == "rob":
        title = "Short-term Robust Training / Validation Objective"
        train_label = "ROB Train Objective"
        val_label = "ROB Validation Objective"
    else:
        raise ValueError(f"mode 必須是 'reg' 或 'rob'，收到：{mode}")

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        epochs,
        train_obj,
        label=train_label,
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        epochs,
        val_obj,
        label=val_label,
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        ma_epochs,
        train_obj_ma,
        label=f"{train_label} MA({window})",
        linewidth=2.2
    )

    plt.plot(
        ma_epochs,
        val_obj_ma,
        label=f"{val_label} MA({window})",
        linewidth=2.2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    print(f"[PLOT] 已儲存 short-term objective curve：{fig_path}")

    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train short-term REG/ROB model")

    # 單獨訓練:reg 或 rob , 預設 both
    parser.add_argument("--mode",   # --mode reg || --mode rob || --mode both
        type=str,
        choices=["both", "reg", "rob"],
        default="both",
        help="單獨訓練:reg 或 rob , 預設 both"
    )

    # 調整penalty
    parser.add_argument("--penalty",# --penalty 100 0.5
    type=float,
    nargs=2,
    default=None,
    metavar=("REG_PENALTY", "ROB_PENALTY"),
    help="REG 與 ROB 的 sensing loss weight, 預設 settings.py值"
)

    # 續訓
    parser.add_argument("--resume", # -mode reg --resume
        action="store_true",
        help="從該 mode + penalty 的 train_state 續訓"
    )
    
    # training curve 作圖
    parser.add_argument(            # -mode reg --plot
        "--plot",
        action="store_true",
        help="只讀取該 mode + penalty 的 curve .npy 並畫圖，不進行訓練。"
    )

    args = parser.parse_args()
    mode = args.mode.lower()

    if args.penalty is None:
        reg_penalty = float(REG_SENSING_LOSS_WEIGHT)
        rob_penalty = float(ROB_SENSING_LOSS_WEIGHT)
    else:
        reg_penalty = float(args.penalty[0])
        rob_penalty = float(args.penalty[1])

    train_injection_variance = float(INJECTION_VARIANCE)
    injection_samples           = int(INJECTION_SAMPLES)

    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")
    longterm_ckpt_path  = os.path.join(LT_CKPT_DIR, "longterm.ckpt")

    if args.resume and mode == "both":
        raise ValueError("--resume 時請指定單一模型：--mode reg 或 --mode rob")

    run_modes = ["reg", "rob"] if mode == "both" else [mode]

    # ================================
    # Plot only
    # ================================
    if args.plot:
        for run_mode in run_modes:
            penalty = reg_penalty if run_mode == "reg" else rob_penalty
            paths = build_st_paths(mode=run_mode, penalty=penalty)

            print("\n[INFO] Plot short-term curve:")
            print(f"[INFO] mode = {run_mode}")
            print(f"[INFO] penalty = {penalty}")
            print(f"[INFO] curve_path = {paths['curve_path']}")

            plot_shortterm_objective_curve(
                mode=run_mode,
                penalty=penalty,
                paths=paths,
            )

        raise SystemExit

    # ================================
    # Load datasets
    # ================================
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path = {val_dataset_path}")

    train_dataset   = load_shortterm_dataset(train_dataset_path, "train")
    val_dataset     = load_shortterm_dataset(val_dataset_path, "val")

    # ================================
    # Load long-term model
    # ================================
    print("\n[INFO] 載入 shared long-term model ...")
    print(f"[INFO] longterm_ckpt_path = {longterm_ckpt_path}")

    longterm_net = LongTermPositionNet(ckpt_kind=None).to(DEVICE)
    longterm_net.load_model(path=longterm_ckpt_path, verbose=True)

    # ================================
    # Train REG
    # ================================
    if "reg" in run_modes:
        reg_paths = build_st_paths(mode="reg", penalty=reg_penalty)

        print("\n[INFO] Short-term REG run setting:")
        print(f"[INFO] mode = reg")
        print(f"[INFO] penalty = {reg_penalty}")
        print(f"[INFO] run_dir = {reg_paths['run_dir']}")
        print(f"[INFO] comm_ckpt_path = {reg_paths['comm_ckpt_path']}")
        print(f"[INFO] radar_ckpt_path = {reg_paths['radar_ckpt_path']}")
        print(f"[INFO] curve_path = {reg_paths['curve_path']}")
        print(f"[INFO] state_path = {reg_paths['state_path']}")

        save_run_config(    # 儲存訓練設定到資料夾
            paths=reg_paths,
            mode="reg",
            penalty=reg_penalty,
            train_injection_variance=0.0,
            injection_samples=0,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            longterm_ckpt_path=longterm_ckpt_path,
        )

        short_comm_reg = ShortTermCommNet(ckpt_kind=None).to(DEVICE)
        short_radar_reg = ShortTermRadarNet(ckpt_kind=None).to(DEVICE)

        print("\n[INFO] 開始訓練 short-term regular ...")
        print(f"[INFO] REG_EPOCHS = {REG_EPOCHS}")
        print(f"[INFO] REG_LEARNING_RATE = {REG_LEARNING_RATE}")
        print(f"[INFO] MINIBATCHES = {MINIBATCHES}")
        print(f"[INFO] ST_BATCH_LAYOUTS = {ST_BATCH_LAYOUTS}")
        print(f"[INFO] ST_BATCH_EST_CHANNELS_PER_LAYOUT = {ST_BATCH_EST_CHANNELS_PER_LAYOUT}")
        print(f"[INFO] REG sensing penalty = {reg_penalty}")
        print(f"[INFO] resume = {args.resume}")

        train_shortterm_regular(    # 訓練REG
            longterm_net=longterm_net,
            comm_net=short_comm_reg,
            radar_net=short_radar_reg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            paths=reg_paths,
            sensing_loss_weight=reg_penalty,
            resume=args.resume,
        )

        print("[INFO] Short-term regular training finished.")

    # ================================
    # Train ROB
    # ================================
    if "rob" in run_modes:
        rob_paths = build_st_paths(mode="rob", penalty=rob_penalty)

        print("\n[INFO] Short-term ROB run setting:")
        print(f"[INFO] mode = rob")
        print(f"[INFO] penalty = {rob_penalty}")
        print(f"[INFO] train_injection_variance = {train_injection_variance}")
        print(f"[INFO] injection_samples = {injection_samples}")
        print(f"[INFO] run_dir = {rob_paths['run_dir']}")
        print(f"[INFO] comm_ckpt_path = {rob_paths['comm_ckpt_path']}")
        print(f"[INFO] radar_ckpt_path = {rob_paths['radar_ckpt_path']}")
        print(f"[INFO] curve_path = {rob_paths['curve_path']}")
        print(f"[INFO] state_path = {rob_paths['state_path']}")

        save_run_config(    # 儲存訓練設定到資料夾
            paths=rob_paths,
            mode="rob",
            penalty=rob_penalty,
            train_injection_variance=train_injection_variance,
            injection_samples=injection_samples,
            train_dataset_path=train_dataset_path,
            val_dataset_path=val_dataset_path,
            longterm_ckpt_path=longterm_ckpt_path,
        )

        short_comm_rob = ShortTermCommNet(ckpt_kind=None).to(DEVICE)
        short_radar_rob = ShortTermRadarNet(ckpt_kind=None).to(DEVICE)

        print("\n[INFO] 開始訓練 short-term robust ...")
        print(f"[INFO] ROB_EPOCHS = {ROB_EPOCHS}")
        print(f"[INFO] ROB_LEARNING_RATE = {ROB_LEARNING_RATE}")
        print(f"[INFO] MINIBATCHES = {MINIBATCHES}")
        print(f"[INFO] ST_BATCH_LAYOUTS = {ST_BATCH_LAYOUTS}")
        print(f"[INFO] ST_BATCH_EST_CHANNELS_PER_LAYOUT = {ST_BATCH_EST_CHANNELS_PER_LAYOUT}")
        print(f"[INFO] ROB sensing penalty = {rob_penalty}")
        print(f"[INFO] train_injection_variance = {train_injection_variance}")
        print(f"[INFO] injection_samples = {injection_samples}")
        print(f"[INFO] OUTAGE_QUANTILE = {OUTAGE_QUANTILE}")
        print(f"[INFO] resume = {args.resume}")

        train_shortterm_robust(     # 訓練ROB
            longterm_net=longterm_net,
            comm_net=short_comm_rob,
            radar_net=short_radar_rob,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            paths=rob_paths,
            sensing_loss_weight=rob_penalty,
            train_injection_variance=train_injection_variance,
            injection_samples=injection_samples,
            resume=args.resume,
        )

        print("[INFO] Short-term robust training finished.")