# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet

# robust injection 分塊，避免顯存爆
INJECTION_CHUNK = 50


# ============================================================
# Helpers
# ============================================================
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
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"[{split_name}] 找不到 dataset 檔案：{npz_path}\n"
            f"請先執行 rician.py 生成固定 dataset。"
        )

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

    # 一次性把整個 short-term dataset 解壓進 RAM
    with np.load(npz_path) as data:
        for k in required_keys:
            if k not in data:
                raise KeyError(f"[{split_name}] dataset 缺少欄位：{k}")

        dataset = {k: data[k] for k in required_keys}

    print(f"[{split_name}] preloaded into RAM: {npz_path}")
    print(f"[{split_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{split_name}] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[{split_name}] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[{split_name}] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[{split_name}] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))
    print(f"[{split_name}] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")

    return dataset


def sample_layout_id(n_layouts: int) -> int:
    return int(np.random.randint(0, n_layouts))


def sample_channel_ids(n_pool: int, batch_size: int) -> np.ndarray:
    replace = n_pool < batch_size
    return np.random.choice(n_pool, size=batch_size, replace=replace)


def get_fixed_theta_from_longterm(longterm_net: LongTermPositionNet, ue_layout_np: np.ndarray):
    """
    給一組 layout，從 long-term net 取固定 theta_LT
    回傳 shape = (1, RIS_UNIT)
    """
    longterm_net.eval()
    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)  # (1,K,2)
        theta_lt, _, _ = longterm_net(layout_t)
    return theta_lt.detach()


def extract_shortterm_batch(dataset, layout_id: int, channel_ids: np.ndarray):
    """
    從 nested dataset 中取出：
        - 指定 layout 的 ue_layout 與 pathloss
        - 該 layout 底下指定 channel_ids 的 estimated channels
    """
    ue_layout = dataset["ue_layouts"][layout_id]              # (K,2)
    pl_BS_UE = dataset["pl_BS_UE"][layout_id]                # (K,)
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]        # (K,)
    pl_BS_TAR_BS = dataset["pl_BS_TAR_BS"][layout_id]        # scalar

    h_dk_hat = np_to_torch_complex(dataset["st_h_dk_hat"][layout_id][channel_ids])   # (B,M,K)
    h_rk_hat = np_to_torch_complex(dataset["st_h_rk_hat"][layout_id][channel_ids])   # (B,N,K)
    G_hat    = np_to_torch_complex(dataset["st_G_hat"][layout_id][channel_ids])      # (B,N,M)
    g_dt_hat = np_to_torch_complex(dataset["st_g_dt_hat"][layout_id][channel_ids])   # (B,M,1)

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


# ============================================================
# Short-term regular objective
# ============================================================
def forward_shortterm_regular_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,      # (1,N)
    batch_data: dict,
):
    """
    regular short-term:
        fixed theta_LT
        estimated channels -> W_C, W_R
        objective = mean sum-rate - sensing penalty - tx penalty
    """
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat    = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE     = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B = h_dk_hat.shape[0]
    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    # design variables from estimated channels
    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)

    # communication objective on estimated channels
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

    # sensing penalty on estimated channels
    sense_snr = comm_net.compute_sense_snr(
        g_dt=g_dt_hat,
        W_R=W_R,
        W_C=W_C,
        pl_BS_TAR_BS=pl_BS_TAR_BS
    ).real
    snr_penalty = torch.clamp(SENSING_SNR_THRESHOLD - sense_snr, min=0.0).mean()

    # tx penalty
    tx_power = comm_net.compute_tx_power(W_C, W_R)
    tx_penalty = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0).mean()

    objective = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT * snr_penalty
        - TX_POWER_LOSS_WEIGHT * tx_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_snr_mean_db": (10.0 * torch.log10(sense_snr.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "tx_penalty_mean": tx_penalty.detach(),
        "objective": objective.detach(),
    }
    return objective, logs


# ============================================================
# Short-term robust objective
# ============================================================
def forward_shortterm_robust_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,      # (1,N)
    batch_data: dict,
    injection_samples: int,
):
    """
    robust short-term:
        1. NN輸入仍然是 estimated channels + fixed theta_LT
        2. 先用原始 estimated channels 產生 W_C, W_R
        3. 再把 estimated channels 複製 L 次並注入不確定性
        4. 在 injected channels 上計算 robust objective
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

    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    # --------------------------------------------------------
    # (A) design variables from ORIGINAL estimated channels
    # --------------------------------------------------------
    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)

    tx_power = comm_net.compute_tx_power(W_C, W_R)
    tx_penalty = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0).mean()

    # --------------------------------------------------------
    # (B) uncertainty injection only in performance calculation
    # --------------------------------------------------------
    sumrate_chunks = []
    snr_chunks = []

    for s0 in range(0, L, INJECTION_CHUNK):
        s = min(INJECTION_CHUNK, L - s0)

        # replicate estimated channels: (B,s,...) -> (B*s,...)
        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        # injected estimated channels
        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape, INJECTION_VARIANCE, DEVICE, h_dk_rep.dtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape, INJECTION_VARIANCE, DEVICE, h_rk_rep.dtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,    INJECTION_VARIANCE, DEVICE, G_rep.dtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape, INJECTION_VARIANCE, DEVICE, g_dt_rep.dtype)

        # replicate design vars
        theta_rep = theta_batch.unsqueeze(1).expand(B, s, N).reshape(B * s, N)
        W_C_rep   = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        L_R = W_R.shape[2]
        W_R_rep = W_R.unsqueeze(1).expand(B, s, M, L_R).reshape(B * s, M, L_R)

        # sum-rate on injected channels
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
        sum_rate = rates.sum(dim=1)

        # sensing SNR on injected channels
        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real

        sumrate_chunks.append(sum_rate.reshape(B, s))
        snr_chunks.append(sense_snr.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)   # (B,L)
    snr_samples     = torch.cat(snr_chunks, dim=1)       # (B,L)

    # --------------------------------------------------------
    # (C) robust aggregation
    # --------------------------------------------------------
    sumrate_mean_per_sample = sumrate_samples.mean(dim=1)    # (B,)
    sumrate_mean = sumrate_mean_per_sample.mean()

    q = float(OUTAGE_QUANTILE)
    k = max(1, int(np.ceil(q * L)))
    snr_var = torch.kthvalue(snr_samples, k=k, dim=1).values  # (B,)

    snr_penalty = torch.clamp(SENSING_SNR_THRESHOLD - snr_var, min=0.0).mean()

    objective = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT * snr_penalty
        - TX_POWER_LOSS_WEIGHT * tx_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_var_snr_mean_db": (10.0 * torch.log10(snr_var.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "tx_penalty_mean": tx_penalty.detach(),
        "objective": objective.detach(),
    }
    return objective, logs


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate_shortterm_regular_all(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,
):
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0]

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0

    for layout_id in range(n_val_layouts):
        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

        # 使用該 layout 底下全部 val estimated channels
        n_pool = val_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = np.arange(n_pool)

        batch_data = extract_shortterm_batch(val_dataset, layout_id, channel_ids)

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
        "objective": total_obj / n_val_layouts,
        "sum_rate_mean": total_sumrate / n_val_layouts,
        "sense_snr_mean_db": total_snr_db / n_val_layouts,
    }


@torch.no_grad()
def validate_shortterm_robust_all(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,
):
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0]

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0

    for layout_id in range(n_val_layouts):
        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

        # 使用該 layout 底下全部 val estimated channels
        n_pool = val_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = np.arange(n_pool)

        batch_data = extract_shortterm_batch(val_dataset, layout_id, channel_ids)

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
        "objective": total_obj / n_val_layouts,
        "sum_rate_mean": total_sumrate / n_val_layouts,
        "sense_var_snr_mean_db": total_snr_db / n_val_layouts,
    }


# ============================================================
# Train regular
# ============================================================
def train_shortterm_regular(longterm_net, comm_net, radar_net, train_dataset, val_dataset):
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=LEARNING_RATE
    )

    best_val_obj = -np.inf
    curves = []

    curve_path = os.path.join(CURVE_DIR, f"shortterm_regular_curves_{SETTING_STRING}.npy")

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(1, EPOCHS + 1, desc="ShortTerm-Regular"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            # 1 個 minibatch：抽 1 個 layout，再抽 BATCH_SIZE estimated channels
            layout_id = sample_layout_id(n_train_layouts)
            ue_layout = train_dataset["ue_layouts"][layout_id]
            theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

            n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]
            channel_ids = sample_channel_ids(n_pool, BATCH_SIZE)

            batch_data = extract_shortterm_batch(train_dataset, layout_id, channel_ids)

            optimizer.zero_grad(set_to_none=True)

            obj, logs = forward_shortterm_regular_objective(
                comm_net=comm_net,
                radar_net=radar_net,
                theta_fixed=theta_fixed,
                batch_data=batch_data,
            )

            (-obj).backward()
            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_snr_db_ep += float(logs["sense_snr_mean_db"].cpu()) / MINIBATCHES

        val_logs = validate_shortterm_regular_all(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
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
            f"[Short-Reg Epoch {ep:03d}] "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_logs['objective']:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_logs['sum_rate_mean']:.4e} | "
            f"TrainSNR(dB)={train_snr_db_ep:.3f} | ValSNR(dB)={val_logs['sense_snr_mean_db']:.3f}"
        )

        if val_logs["objective"] > best_val_obj:
            best_val_obj = val_logs["objective"]
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)

    print(f"[Short-Reg] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


# ============================================================
# Train robust
# ============================================================
def train_shortterm_robust(longterm_net, comm_net, radar_net, train_dataset, val_dataset):
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=LEARNING_RATE
    )

    best_val_obj = -np.inf
    curves = []

    curve_path = os.path.join(CURVE_DIR, f"shortterm_robust_curves_{SETTING_STRING}.npy")

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(1, 200 + 1, desc="ShortTerm-Robust"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            # 1 個 minibatch：抽 1 個 layout，再抽 BATCH_SIZE estimated channels
            layout_id = sample_layout_id(n_train_layouts)
            ue_layout = train_dataset["ue_layouts"][layout_id]
            theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

            n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]
            channel_ids = sample_channel_ids(n_pool, BATCH_SIZE)

            batch_data = extract_shortterm_batch(train_dataset, layout_id, channel_ids)

            optimizer.zero_grad(set_to_none=True)

            obj, logs = forward_shortterm_robust_objective(
                comm_net=comm_net,
                radar_net=radar_net,
                theta_fixed=theta_fixed,
                batch_data=batch_data,
                injection_samples=INJECTION_SAMPLES,
            )

            (-obj).backward()
            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_snr_db_ep += float(logs["sense_var_snr_mean_db"].cpu()) / MINIBATCHES

        val_logs = validate_shortterm_robust_all(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
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
            f"[Short-Rob Epoch {ep:03d}] "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_logs['objective']:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_logs['sum_rate_mean']:.4e} | "
            f"TrainVaR-SNR(dB)={train_snr_db_ep:.3f} | ValVaR-SNR(dB)={val_logs['sense_var_snr_mean_db']:.3f}"
        )

        if val_logs["objective"] > best_val_obj:
            best_val_obj = val_logs["objective"]
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)

    print(f"[Short-Rob] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("[INFO] 載入固定 nested datasets ...")

    train_dataset = load_shortterm_dataset(TRAIN_DATASET_PATH, "train")
    val_dataset   = load_shortterm_dataset(VAL_DATASET_PATH, "val")

    # 載入 long-term best ckpt，提供 fixed theta_LT
    longterm_net = LongTermPositionNet(ckpt_kind="longterm").to(DEVICE)
    if not longterm_net.model_path or not os.path.exists(longterm_net.model_path):
        raise FileNotFoundError(
            "找不到 long-term checkpoint。\n"
            "請先執行 main_lt.py 訓練 long-term 網路。"
        )
    longterm_net.load_model(verbose=True)

    # ========================================================
    # 1) train short-term regular
    # ========================================================
    short_comm_reg = ShortTermCommNet(ckpt_kind="short_comm").to(DEVICE)
    short_radar_reg = ShortTermRadarNet(ckpt_kind="short_radar").to(DEVICE)

    print("[INFO] 開始訓練 short-term regular ...")
    train_shortterm_regular(
        longterm_net=longterm_net,
        comm_net=short_comm_reg,
        radar_net=short_radar_reg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # ========================================================
    # 2) train short-term robust
    # ========================================================
    short_comm_rob = ShortTermCommNet(ckpt_kind="short_comm_robust").to(DEVICE)
    short_radar_rob = ShortTermRadarNet(ckpt_kind="short_radar_robust").to(DEVICE)

    print("[INFO] 開始訓練 short-term robust ...")
    train_shortterm_robust(
        longterm_net=longterm_net,
        comm_net=short_comm_rob,
        radar_net=short_radar_rob,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("[INFO] Short-term training finished.")