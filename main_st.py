# -*- coding: utf-8 -*-
import os
import math
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
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


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


def sample_channel_ids(n_pool: int, batch_size: int) -> np.ndarray:
    return np.random.choice(
        n_pool,
        size=batch_size,
        replace=False
    )


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout_np: np.ndarray
):
    """
    給一組 layout，從 long-term net 取固定 theta_LT。

    回傳:
        theta_lt : torch.Tensor, shape = (1, RIS_UNIT)
    """
    longterm_net.eval()

    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)  # shape = (1,K,2)
        theta_lt, _, _ = longterm_net(layout_t)

    return theta_lt.detach()


def extract_shortterm_batch(
    dataset,
    layout_id: int,
    channel_ids: np.ndarray
):
    """
    從 nested dataset 中取出：
        1. 指定 layout 的 ue_layout 與 path loss
        2. 該 layout 底下指定 channel ids 的 estimated channels
    """
    ue_layout = dataset["ue_layouts"][layout_id]                  # shape = (K,2)
    pl_BS_UE = dataset["pl_BS_UE"][layout_id]                    # shape = (K,)
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]            # shape = (K,)
    pl_BS_TAR_BS = dataset["pl_BS_TAR_BS"][layout_id]            # scalar

    h_dk_hat = np_to_torch_complex(
        dataset["st_h_dk_hat"][layout_id][channel_ids]
    )                                                           # shape = (B,M,K)

    h_rk_hat = np_to_torch_complex(
        dataset["st_h_rk_hat"][layout_id][channel_ids]
    )                                                           # shape = (B,N,K)

    G_hat = np_to_torch_complex(
        dataset["st_G_hat"][layout_id][channel_ids]
    )                                                           # shape = (B,N,M)

    g_dt_hat = np_to_torch_complex(
        dataset["st_g_dt_hat"][layout_id][channel_ids]
    )                                                           # shape = (B,M,1)

    return {
        "ue_layout": ue_layout,
        "pl_BS_UE": pl_BS_UE,
        "pl_BS_RIS_UE": pl_BS_RIS_UE,
        "pl_BS_TAR_BS": pl_BS_TAR_BS,
        "h_dk_hat": h_dk_hat,
        "h_rk_hat": h_rk_hat,
        "G_hat": G_hat,
        "g_dt_hat": g_dt_hat,
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


def build_st_train_state_path(kind: str):
    return os.path.join(
        CKPT_DIR,
        f"{kind}_train_state_{SETTING_STRING}.pt"
    )


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
    batch_data: dict,
):
    """
    計算 regular short-term objective。

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

    B = h_dk_hat.shape[0]
    theta_batch = theta_fixed.expand(B, RIS_UNIT)                     # shape = (B,N)

    W_C = comm_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )                                                                 # shape = (B,M,K)

    W_R = radar_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )                                                                 # shape = (B,M,RADAR_STREAMS)

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)                  # TX power scaling

    sumrate_mean = comm_net.compute_sum_rate(
        h_dk=h_dk_hat,
        h_rk=h_rk_hat,
        G=G_hat,
        theta=theta_batch,
        W_R=W_R,
        W_C=W_C,
        pl_BS_UE=pl_BS_UE,
        pl_BS_RIS_UE=pl_BS_RIS_UE,
    ).mean()

    sense_snr = comm_net.compute_sense_snr(
        g_dt=g_dt_hat,
        W_R=W_R,
        W_C=W_C,
        pl_BS_TAR_BS=pl_BS_TAR_BS,
    ).real

    snr_penalty = torch.clamp(
        SENSING_SNR_THRESHOLD - sense_snr,
        min=0.0
    ).mean()

    objective = (
        sumrate_mean
        - REG_SENSING_LOSS_WEIGHT * snr_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_snr_mean_db": (
            10.0 * torch.log10(sense_snr.clamp_min(1e-12))
        ).mean().detach(),
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
    batch_data: dict,
    injection_samples: int,
):
    """
    計算 robust short-term objective。

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
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE     = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B, M, K = h_dk_hat.shape
    N = h_rk_hat.shape[1]
    L = int(injection_samples)

    theta_batch = theta_fixed.expand(B, RIS_UNIT)                     # shape = (B,N)

    W_C = comm_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )                                                                 # shape = (B,M,K)

    W_R = radar_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )                                                                 # shape = (B,M,RADAR_STREAMS)

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)                  # TX power scaling

    sumrate_chunks = []
    snr_chunks = []

    for s0 in range(0, L, INJECTION_CHUNK):
        s = min(INJECTION_CHUNK, L - s0)

        h_dk_rep = h_dk_hat.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        h_rk_rep = h_rk_hat.unsqueeze(1).expand(
            B, s, N, K
        ).reshape(B * s, N, K)

        G_rep = G_hat.unsqueeze(1).expand(
            B, s, N, M
        ).reshape(B * s, N, M)

        g_dt_rep = g_dt_hat.unsqueeze(1).expand(
            B, s, M, 1
        ).reshape(B * s, M, 1)

        h_dk_inj = h_dk_rep + complex_awgn(
            h_dk_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            h_dk_rep.dtype
        )

        h_rk_inj = h_rk_rep + complex_awgn(
            h_rk_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            h_rk_rep.dtype
        )

        G_inj = G_rep + complex_awgn(
            G_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            G_rep.dtype
        )

        g_dt_inj = g_dt_rep + complex_awgn(
            g_dt_rep.shape,
            INJECTION_VARIANCE,
            DEVICE,
            g_dt_rep.dtype
        )

        theta_rep = theta_batch.unsqueeze(1).expand(
            B, s, N
        ).reshape(B * s, N)

        W_C_rep = W_C.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        radar_streams = W_R.shape[2]

        W_R_rep = W_R.unsqueeze(1).expand(
            B, s, M, radar_streams
        ).reshape(B * s, M, radar_streams)

        sinrs = comm_net.compute_comm_sinrs(
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
        sum_rate = rates.sum(dim=1)                                   # shape = (B*s,)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real                                                        # shape = (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))
        snr_chunks.append(sense_snr.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)                # shape = (B,L)
    snr_samples = torch.cat(snr_chunks, dim=1)                        # shape = (B,L)

    sumrate_mean_per_sample = sumrate_samples.mean(dim=1)             # shape = (B,)
    sumrate_mean = sumrate_mean_per_sample.mean()

    q = float(OUTAGE_QUANTILE)
    k = max(1, int(np.ceil(q * L)))
    snr_var = torch.kthvalue(
        snr_samples,
        k=k,
        dim=1
    ).values                                                          # shape = (B,)

    snr_penalty = torch.clamp(
        SENSING_SNR_THRESHOLD - snr_var,
        min=0.0
    ).mean()

    objective = (
        sumrate_mean
        - ROB_SENSING_LOSS_WEIGHT * snr_penalty
    )

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
def validate_shortterm_regular_sampled(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,
    num_val_layouts: int,
    channels_per_layout: int,
):
    """
    從 validation pool 抽樣 layouts，計算 regular short-term validation objective。

    回傳:
        logs : dict
    """
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_pool = val_dataset["ue_layouts"].shape[0]

    layout_ids = np.random.choice(
        n_val_pool,
        size=num_val_layouts,
        replace=False
    )

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0

    for layout_id in layout_ids:
        layout_id = int(layout_id)

        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net,
            ue_layout
        )

        n_pool = val_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = sample_channel_ids(
            n_pool,
            channels_per_layout
        )

        batch_data = extract_shortterm_batch(
            val_dataset,
            layout_id,
            channel_ids
        )

        obj, logs = forward_shortterm_regular_objective(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
        )

        total_obj += float(obj.detach().cpu())
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_snr_db += float(logs["sense_snr_mean_db"].cpu())

    return {
        "objective": total_obj / num_val_layouts,
        "sum_rate_mean": total_sumrate / num_val_layouts,
        "sense_snr_mean_db": total_snr_db / num_val_layouts,
        "num_val_layouts": num_val_layouts,
        "channels_per_layout": channels_per_layout,
    }


@torch.no_grad()
def validate_shortterm_robust_sampled(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,
    num_val_layouts: int,
    channels_per_layout: int,
):
    """
    從 validation pool 抽樣 layouts，計算 robust short-term validation objective。

    回傳:
        logs : dict
    """
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_pool = val_dataset["ue_layouts"].shape[0]

    layout_ids = np.random.choice(
        n_val_pool,
        size=num_val_layouts,
        replace=False
    )

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0

    for layout_id in layout_ids:
        layout_id = int(layout_id)

        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net,
            ue_layout
        )

        n_pool = val_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = sample_channel_ids(
            n_pool,
            channels_per_layout
        )

        batch_data = extract_shortterm_batch(
            val_dataset,
            layout_id,
            channel_ids
        )

        obj, logs = forward_shortterm_robust_objective(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            injection_samples=INJECTION_SAMPLES,
        )

        total_obj += float(obj.detach().cpu())
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_snr_db += float(logs["sense_var_snr_mean_db"].cpu())

    return {
        "objective": total_obj / num_val_layouts,
        "sum_rate_mean": total_sumrate / num_val_layouts,
        "sense_var_snr_mean_db": total_snr_db / num_val_layouts,
        "num_val_layouts": num_val_layouts,
        "channels_per_layout": channels_per_layout,
    }


# ================================
# Train regular
# ================================
def train_shortterm_regular(
    longterm_net,
    comm_net,
    radar_net,
    train_dataset,
    val_dataset,
    resume=False,
):
    """
    訓練 short-term regular networks。
    """
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=REG_LEARNING_RATE
    )

    curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_regular_curves_{SETTING_STRING}.npy"
    )

    state_path = build_st_train_state_path("shortterm_regular")

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

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(start_epoch, end_epoch + 1, desc="ShortTerm-Regular"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_ids = np.random.choice(
                n_train_layouts,
                size=ST_BATCH_LAYOUTS,
                replace=False
            )

            optimizer.zero_grad(set_to_none=True)

            mb_obj = 0.0
            mb_sumrate = 0.0
            mb_snr_db = 0.0

            for layout_id in layout_ids:
                layout_id = int(layout_id)

                ue_layout = train_dataset["ue_layouts"][layout_id]
                theta_fixed = get_fixed_theta_from_longterm(
                    longterm_net,
                    ue_layout
                )

                n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]
                channel_ids = sample_channel_ids(
                    n_pool,
                    ST_BATCH_EST_CHANNELS_PER_LAYOUT
                )

                batch_data = extract_shortterm_batch(
                    train_dataset,
                    layout_id,
                    channel_ids
                )

                obj, logs = forward_shortterm_regular_objective(
                    comm_net=comm_net,
                    radar_net=radar_net,
                    theta_fixed=theta_fixed,
                    batch_data=batch_data,
                )

                loss = -obj / ST_BATCH_LAYOUTS
                loss.backward()

                mb_obj += float(obj.detach().cpu()) / ST_BATCH_LAYOUTS
                mb_sumrate += float(logs["sum_rate_mean"].cpu()) / ST_BATCH_LAYOUTS
                mb_snr_db += float(logs["sense_snr_mean_db"].cpu()) / ST_BATCH_LAYOUTS

            optimizer.step()

            train_obj_ep += mb_obj / MINIBATCHES
            train_sumrate_ep += mb_sumrate / MINIBATCHES
            train_snr_db_ep += mb_snr_db / MINIBATCHES

        val_logs = validate_shortterm_regular_sampled(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
            num_val_layouts=N_VAL_LAYOUTS,
            channels_per_layout=SHORTTERM_EST_CHANNELS_PER_LAYOUT,
        )

        curves.append([
            train_obj_ep,
            val_logs["objective"],
            train_sumrate_ep,
            val_logs["sum_rate_mean"],
            train_snr_db_ep,
            val_logs["sense_snr_mean_db"],
        ])

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
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)
        
        save_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
            best_val_obj=best_val_obj,
            finished_epoch=ep,
        )

    print(f"[Short-Reg] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


# ================================
# Train robust
# ================================
def train_shortterm_robust(
    longterm_net,
    comm_net,
    radar_net,
    train_dataset,
    val_dataset,
    resume=False,
):
    """
    訓練 short-term robust networks。
    """
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=ROB_LEARNING_RATE
    )

    curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_curves_{SETTING_STRING}.npy"
    )

    state_path = build_st_train_state_path("shortterm_robust")

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

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(start_epoch, end_epoch + 1, desc="ShortTerm-Robust"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_ids = np.random.choice(
                n_train_layouts,
                size=ST_BATCH_LAYOUTS,
                replace=False
            )

            optimizer.zero_grad(set_to_none=True)

            mb_obj = 0.0
            mb_sumrate = 0.0
            mb_snr_db = 0.0

            for layout_id in layout_ids:
                layout_id = int(layout_id)

                ue_layout = train_dataset["ue_layouts"][layout_id]
                theta_fixed = get_fixed_theta_from_longterm(
                    longterm_net,
                    ue_layout
                )

                n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]
                channel_ids = sample_channel_ids(
                    n_pool,
                    ST_BATCH_EST_CHANNELS_PER_LAYOUT
                )

                batch_data = extract_shortterm_batch(
                    train_dataset,
                    layout_id,
                    channel_ids
                )

                obj, logs = forward_shortterm_robust_objective(
                    comm_net=comm_net,
                    radar_net=radar_net,
                    theta_fixed=theta_fixed,
                    batch_data=batch_data,
                    injection_samples=INJECTION_SAMPLES,
                )

                loss = -obj / ST_BATCH_LAYOUTS
                loss.backward()

                mb_obj += float(obj.detach().cpu()) / ST_BATCH_LAYOUTS
                mb_sumrate += float(logs["sum_rate_mean"].cpu()) / ST_BATCH_LAYOUTS
                mb_snr_db += float(logs["sense_var_snr_mean_db"].cpu()) / ST_BATCH_LAYOUTS

            optimizer.step()

            train_obj_ep += mb_obj / MINIBATCHES
            train_sumrate_ep += mb_sumrate / MINIBATCHES
            train_snr_db_ep += mb_snr_db / MINIBATCHES

        val_logs = validate_shortterm_robust_sampled(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
            num_val_layouts=N_VAL_LAYOUTS,
            channels_per_layout=SHORTTERM_EST_CHANNELS_PER_LAYOUT,
        )

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
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)
        save_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
            best_val_obj=best_val_obj,
            finished_epoch=ep,
        )

    print(f"[Short-Rob] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


# ================================
# Plot short-term objective curves
# ================================
def plot_shortterm_objective_curve():
    reg_curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_regular_curves_{SETTING_STRING}.npy"
    )

    rob_curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_curves_{SETTING_STRING}.npy"
    )

    reg_curves = np.load(reg_curve_path)
    rob_curves = np.load(rob_curve_path)

    # 跳過 epoch [start_idx]，避免初期 objective 過低壓縮後期曲線
    start_idx = 10

    window = PLOT_MOVING_AVG_WINDOW

    # ---------- REG ----------
    reg_epochs = np.arange(start_idx + 1, reg_curves.shape[0] + 1)

    reg_train_obj = reg_curves[start_idx:, 0]
    reg_val_obj   = reg_curves[start_idx:, 1]

    reg_train_obj_ma = moving_average(reg_train_obj, window)
    reg_val_obj_ma   = moving_average(reg_val_obj, window)

    reg_ma_epochs = reg_epochs[window - 1:]

    reg_fig_path = os.path.join(
        CURVE_DIR,
        f"shortterm_regular_objective_curve_{SETTING_STRING}.jpg"
    )

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        reg_epochs,
        reg_train_obj,
        label="REG Train Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        reg_epochs,
        reg_val_obj,
        label="REG Validation Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        reg_ma_epochs,
        reg_train_obj_ma,
        label=f"REG Train Objective MA({window})",
        linewidth=2.2
    )

    plt.plot(
        reg_ma_epochs,
        reg_val_obj_ma,
        label=f"REG Validation Objective MA({window})",
        linewidth=2.2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("Short-term Regular Training / Validation Objective")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(reg_fig_path, format="jpg", dpi=300)
    print(f"[PLOT] 已儲存 regular objective curve：{reg_fig_path}")

    plt.show()
    plt.close()

    # ---------- ROB ----------
    rob_epochs = np.arange(start_idx + 1, rob_curves.shape[0] + 1)

    rob_train_obj = rob_curves[start_idx:, 0]
    rob_val_obj   = rob_curves[start_idx:, 1]

    rob_train_obj_ma = moving_average(rob_train_obj, window)
    rob_val_obj_ma   = moving_average(rob_val_obj, window)

    rob_ma_epochs = rob_epochs[window - 1:]

    rob_fig_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_objective_curve_{SETTING_STRING}.jpg"
    )

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        rob_epochs,
        rob_train_obj,
        label="ROB Train Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        rob_epochs,
        rob_val_obj,
        label="ROB Validation Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        rob_ma_epochs,
        rob_train_obj_ma,
        label=f"ROB Train Objective MA({window})",
        linewidth=2.2
    )

    plt.plot(
        rob_ma_epochs,
        rob_val_obj_ma,
        label=f"ROB Validation Objective MA({window})",
        linewidth=2.2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("Short-term Robust Training / Validation Objective")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(rob_fig_path, format="jpg", dpi=300)
    print(f"[PLOT] 已儲存 robust objective curve：{rob_fig_path}")

    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train short-term networks")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="只讀取 short-term curve .npy 並畫圖，不進行訓練"
    )
    parser.add_argument(
        "--train_reg",
        action="store_true",
        help="嚴謹續訓 short-term regular,只訓練 REG"
    )

    parser.add_argument(
        "--train_rob",
        action="store_true",
        help="嚴謹續訓 short-term robust,只訓練 ROB"
    )
    args = parser.parse_args()

    if args.plot:
        plot_shortterm_objective_curve()
        raise SystemExit

    print("[INFO] 載入固定 datasets ...")

    train_dataset = load_shortterm_dataset(
        TRAIN_DATASET_PATH,
        "train"
    )

    val_dataset = load_shortterm_dataset(
        VAL_DATASET_PATH,
        "val"
    )

    longterm_net = LongTermPositionNet(
        ckpt_kind="longterm"
    ).to(DEVICE)

    longterm_net.load_model(verbose=True)

    # ================================
    # Resume REG only
    # ================================
    if args.train_reg:
        short_comm_reg = ShortTermCommNet(
            ckpt_kind="short_comm"
        ).to(DEVICE)

        short_radar_reg = ShortTermRadarNet(
            ckpt_kind="short_radar"
        ).to(DEVICE)

        print("[INFO] 開始續訓 short-term regular ...")
        print(f"[INFO] REG_EPOCHS additional = {REG_EPOCHS}")
        print(f"[INFO] REG_LEARNING_RATE = {REG_LEARNING_RATE}")
        print(f"[INFO] MINIBATCHES = {MINIBATCHES}")
        print(f"[INFO] ST_BATCH_LAYOUTS = {ST_BATCH_LAYOUTS}")
        print(f"[INFO] ST_BATCH_EST_CHANNELS_PER_LAYOUT = {ST_BATCH_EST_CHANNELS_PER_LAYOUT}")

        train_shortterm_regular(
            longterm_net=longterm_net,
            comm_net=short_comm_reg,
            radar_net=short_radar_reg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            resume=True,
        )

        print("[INFO] Short-term regular resume finished.")
        raise SystemExit

    # ================================
    # Resume ROB only
    # ================================
    if args.train_rob:
        short_comm_rob = ShortTermCommNet(
            ckpt_kind="short_comm_robust"
        ).to(DEVICE)

        short_radar_rob = ShortTermRadarNet(
            ckpt_kind="short_radar_robust"
        ).to(DEVICE)

        print("[INFO] 開始續訓 short-term robust ...")
        print(f"[INFO] ROB_EPOCHS additional = {ROB_EPOCHS}")
        print(f"[INFO] ROB_LEARNING_RATE = {ROB_LEARNING_RATE}")
        print(f"[INFO] INJECTION_SAMPLES = {INJECTION_SAMPLES}")
        print(f"[INFO] INJECTION_VARIANCE = {INJECTION_VARIANCE}")
        print(f"[INFO] OUTAGE_QUANTILE = {OUTAGE_QUANTILE}")

        train_shortterm_robust(
            longterm_net=longterm_net,
            comm_net=short_comm_rob,
            radar_net=short_radar_rob,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            resume=True,
        )

        print("[INFO] Short-term robust resume finished.")
        raise SystemExit

    # ================================
    # Fresh REG + ROB training
    # ================================
    short_comm_reg = ShortTermCommNet(
        ckpt_kind="short_comm"
    ).to(DEVICE)

    short_radar_reg = ShortTermRadarNet(
        ckpt_kind="short_radar"
    ).to(DEVICE)

    print("[INFO] 開始訓練 short-term regular ...")
    print(f"[INFO] REG_EPOCHS = {REG_EPOCHS}")
    print(f"[INFO] REG_LEARNING_RATE = {REG_LEARNING_RATE}")
    print(f"[INFO] MINIBATCHES = {MINIBATCHES}")
    print(f"[INFO] ST_BATCH_LAYOUTS = {ST_BATCH_LAYOUTS}")
    print(f"[INFO] ST_BATCH_EST_CHANNELS_PER_LAYOUT = {ST_BATCH_EST_CHANNELS_PER_LAYOUT}")

    train_shortterm_regular(
        longterm_net=longterm_net,
        comm_net=short_comm_reg,
        radar_net=short_radar_reg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume=False,
    )

    short_comm_rob = ShortTermCommNet(
        ckpt_kind="short_comm_robust"
    ).to(DEVICE)

    short_radar_rob = ShortTermRadarNet(
        ckpt_kind="short_radar_robust"
    ).to(DEVICE)

    print("[INFO] 開始訓練 short-term robust ...")
    print(f"[INFO] ROB_EPOCHS = {ROB_EPOCHS}")
    print(f"[INFO] ROB_LEARNING_RATE = {ROB_LEARNING_RATE}")
    print(f"[INFO] INJECTION_SAMPLES = {INJECTION_SAMPLES}")
    print(f"[INFO] INJECTION_VARIANCE = {INJECTION_VARIANCE}")
    print(f"[INFO] OUTAGE_QUANTILE = {OUTAGE_QUANTILE}")

    train_shortterm_robust(
        longterm_net=longterm_net,
        comm_net=short_comm_rob,
        radar_net=short_radar_rob,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        resume=False,
    )

    print("[INFO] Short-term training finished.")