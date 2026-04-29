# -*- coding: utf-8 -*-
import os
import math
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from settings import *
from rician import generate_real_channels, estimate_channels, large_scale_fading
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ============================================================
# 本檔專用設定（先放 main.py，之後若穩定再搬去 settings.py）
# ============================================================
LAYOUT_TRAIN_RATIO = 0.8            # 將1000UE位置組 82分 給train / val

# Long-term
LONGTERM_BATCH_LAYOUTS = 32         # 每個 minibatch 幾個 layout
LONGTERM_MC_TRAIN = 64              # 每個 layout 幾個 channel realizations（train）
LONGTERM_MC_VAL = 128               # 每個 layout 幾個 channel realizations（val）
LONGTERM_VAL_BATCHES = 4            # 每個 epoch 做幾次 val batch

# Short-term
SHORTTERM_VAL_LAYOUTS_PER_EPOCH = 8
SHORTTERM_VAL_CHANNEL_BATCH = min(256, BATCH_SIZE)

# Robust
ROBUST_CHUNK = 50
ROBUST_VAL_INJECTION_SAMPLES = min(200, INJECTION_SAMPLES)


# ============================================================
# Helpers
# ============================================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def split_layout_bank(layout_bank, train_ratio=0.8, seed=RANDOM_SEED):
    idx = np.arange(len(layout_bank))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(train_ratio * len(layout_bank))
    train_bank = [layout_bank[i] for i in idx[:n_train]]
    val_bank   = [layout_bank[i] for i in idx[n_train:]]

    return train_bank, val_bank


def sample_layout_batch(layout_bank, num_layouts):
    replace = len(layout_bank) < num_layouts
    idx = np.random.choice(len(layout_bank), size=num_layouts, replace=replace)
    return [layout_bank[i] for i in idx]


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    ni = torch.randn(shape, device=device, dtype=torch.float32) * sigma
    return torch.complex(nr, ni).to(dtype=cdtype)


def get_theta_from_longterm(longterm_net: LongTermPositionNet, layout):
    """
    給一組 layout，從 long-term net 取出固定 theta
    回傳 shape = (1, RIS_UNIT)
    """
    longterm_net.eval()
    with torch.no_grad():
        layout_t = torch.as_tensor(layout, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,K,2)
        theta_lt, _, _ = longterm_net(layout_t)
    return theta_lt.detach()


# ============================================================
# Long-term objective
# ============================================================
def forward_longterm_objective(
    longterm_net: LongTermPositionNet,
    layout_batch,
    mc_samples: int,
):
    """
    Long-term:
        輸入 layout batch (2個UE位置座標)
        輸出 theta_LT, W_C_LT, W_R_LT
        目標 = mean over layouts [ E_channel(sum-rate) ] - theta penalty - tx penalty
    """
    layout_t = torch.as_tensor(layout_batch, dtype=torch.float32, device=DEVICE)    # (B,K,2) 轉TENSOR
    theta_lt, W_C_lt, W_R_lt = longterm_net(layout_t)                               # 得到輸出

    obj_list = []
    sumrate_list = []
    theta_pen_list = []
    tx_pen_list = []

    B = layout_t.shape[0] # 32

    for b in range(B):
        layout = layout_batch[b]

        # 該 layout 下的統計通道樣本
        h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(mc_samples, layout)
        h_dk = np_to_torch_complex(h_dk_np)
        h_rk = np_to_torch_complex(h_rk_np)
        G    = np_to_torch_complex(G_np)
        g_dt = np_to_torch_complex(g_dt_np)

        pl_BS_UE, pl_BS_RIS_UE, _ = large_scale_fading(layout)

        theta_b = theta_lt[b].unsqueeze(0).expand(mc_samples, RIS_UNIT)           # (mc,N)
        W_C_b   = W_C_lt[b].unsqueeze(0).expand(mc_samples, TX_ANT, UAV_COMM)     # (mc,M,K)
        W_R_b   = W_R_lt[b].unsqueeze(0).expand(mc_samples, TX_ANT, 1)             # (mc,M,1)

        sumrate_mean_b = longterm_net.compute_sum_rate(
            h_dk=h_dk, h_rk=h_rk, G=G, theta=theta_b, W_R=W_R_b, W_C=W_C_b,
            pl_BS_UE=pl_BS_UE, pl_BS_RIS_UE=pl_BS_RIS_UE
        ).mean()

        theta_pen_b = longterm_net.compute_ris_amplitude_penalty(theta_lt[b]).mean()

        tx_power_b = longterm_net.compute_tx_power(
            W_C_lt[b].unsqueeze(0), W_R_lt[b].unsqueeze(0)
        )
        tx_pen_b = torch.clamp(tx_power_b - TRANSMIT_POWER_TOTAL, min=0.0).mean()

        obj_b = (
            sumrate_mean_b
            - RE_POWER_LOSS_WEIGHT * theta_pen_b
            - TX_POWER_LOSS_WEIGHT * tx_pen_b
        )

        obj_list.append(obj_b)
        sumrate_list.append(sumrate_mean_b.detach())
        theta_pen_list.append(theta_pen_b.detach())
        tx_pen_list.append(tx_pen_b.detach())

    objective = torch.stack(obj_list).mean()

    logs = {
        "sum_rate_mean": torch.stack(sumrate_list).mean(),
        "theta_penalty_mean": torch.stack(theta_pen_list).mean(),
        "tx_penalty_mean": torch.stack(tx_pen_list).mean(),
        "objective": objective.detach(),
    }
    return objective, logs


@torch.no_grad()
def validate_longterm(
    longterm_net: LongTermPositionNet,
    val_layout_bank,
):
    longterm_net.eval()

    obj_ep = 0.0
    sumrate_ep = 0.0

    for _ in range(LONGTERM_VAL_BATCHES):
        layout_batch = sample_layout_batch(val_layout_bank, LONGTERM_BATCH_LAYOUTS)
        obj, logs = forward_longterm_objective(
            longterm_net=longterm_net,
            layout_batch=layout_batch,
            mc_samples=LONGTERM_MC_VAL,
        )
        obj_ep += float(obj.detach().cpu()) / LONGTERM_VAL_BATCHES
        sumrate_ep += float(logs["sum_rate_mean"].cpu()) / LONGTERM_VAL_BATCHES

    return obj_ep, sumrate_ep


# ============================================================
# Short-term regular objective
# ============================================================
def forward_shortterm_regular_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,   # (1,N)
    layout,
    batch_size: int,
):
    """
    regular short-term:
        fixed theta
        estimated channels -> W_C, W_R
        objective = sum-rate - sensing penalty - tx penalty
    """
    # 1) 該 layout 生成一個 batch 的真實通道，再估測
    h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(batch_size, layout)
    h_dk_hat_np, h_rk_hat_np, G_hat_np, g_dt_hat_np = estimate_channels(
        h_dk_np, h_rk_np, G_np, g_dt_np
    )

    h_dk_hat = np_to_torch_complex(h_dk_hat_np)
    h_rk_hat = np_to_torch_complex(h_rk_hat_np)
    G_hat    = np_to_torch_complex(G_hat_np)
    g_dt_hat = np_to_torch_complex(g_dt_hat_np)

    pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS = large_scale_fading(layout)

    # 2) 固定 theta
    theta_batch = theta_fixed.expand(batch_size, RIS_UNIT)

    # 3) Short-term 輸出 W_C, W_R
    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)

    # 4) Sum-rate
    sumrate_mean = comm_net.compute_sum_rate(
        h_dk=h_dk_hat,
        h_rk=h_rk_hat,
        G=G_hat,
        theta=theta_batch,
        W_R=W_R,
        W_C=W_C,
        pl_BS_UE=pl_BS_UE,
        pl_BS_RIS_UE=pl_BS_RIS_UE
    ).mean()

    # 5) sensing penalty
    sense_snr = comm_net.compute_sense_snr(g_dt_hat, W_R, W_C, pl_BS_TAR_BS)
    snr_penalty = torch.clamp(SENSING_SNR_THRESHOLD - sense_snr.real, min=0.0).mean()

    # 6) tx penalty
    tx_power = comm_net.compute_tx_power(W_C, W_R)
    tx_penalty = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0).mean()

    objective = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT * snr_penalty
        - TX_POWER_LOSS_WEIGHT * tx_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_snr_mean_db": (10.0 * torch.log10(sense_snr.real.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "tx_penalty_mean": tx_penalty.detach(),
        "objective": objective.detach(),
    }
    return objective, logs


@torch.no_grad()
def validate_shortterm_regular(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_layout_bank,
):
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    obj_ep = 0.0
    sumrate_ep = 0.0
    snr_db_ep = 0.0

    layout_batch = sample_layout_batch(val_layout_bank, SHORTTERM_VAL_LAYOUTS_PER_EPOCH)

    for layout in layout_batch:
        theta_fixed = get_theta_from_longterm(longterm_net, layout)
        obj, logs = forward_shortterm_regular_objective(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            layout=layout,
            batch_size=SHORTTERM_VAL_CHANNEL_BATCH,
        )
        denom = len(layout_batch)
        obj_ep += float(obj.detach().cpu()) / denom
        sumrate_ep += float(logs["sum_rate_mean"].cpu()) / denom
        snr_db_ep += float(logs["sense_snr_mean_db"].cpu()) / denom

    return obj_ep, sumrate_ep, snr_db_ep


# ============================================================
# Short-term robust objective
# ============================================================
def forward_shortterm_robust_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,   # (1,N)
    layout,
    batch_size: int,
    injection_samples: int,
):
    """
    robust short-term:
        fixed theta
        estimated channels -> W_C, W_R
        uncertainty injection on estimated channels
        objective = E[sum-rate] - lambda * max(thr - VaR_q(SNR), 0) - tx penalty
    """
    # 1) estimated channels
    h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(batch_size, layout)
    h_dk_hat_np, h_rk_hat_np, G_hat_np, g_dt_hat_np = estimate_channels(
        h_dk_np, h_rk_np, G_np, g_dt_np
    )

    h_dk_hat = np_to_torch_complex(h_dk_hat_np)
    h_rk_hat = np_to_torch_complex(h_rk_hat_np)
    G_hat    = np_to_torch_complex(G_hat_np)
    g_dt_hat = np_to_torch_complex(g_dt_hat_np)

    pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS = large_scale_fading(layout)

    theta_batch = theta_fixed.expand(batch_size, RIS_UNIT)

    # 2) design variables from estimated channels
    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)

    tx_power = comm_net.compute_tx_power(W_C, W_R)
    tx_penalty = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0).mean()

    # 3) robust injection
    B = batch_size
    M = TX_ANT
    N = RIS_UNIT
    K = UAV_COMM
    S = int(injection_samples)

    sumrate_chunks = []
    snr_chunks = []

    for s0 in range(0, S, ROBUST_CHUNK):
        s = min(ROBUST_CHUNK, S - s0)

        h_dk_rep = h_dk_hat.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk_hat.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G_hat.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt_hat.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        h_dk_inj = h_dk_rep + complex_awgn(h_dk_rep.shape, INJECTION_VARIANCE, DEVICE, h_dk_rep.dtype)
        h_rk_inj = h_rk_rep + complex_awgn(h_rk_rep.shape, INJECTION_VARIANCE, DEVICE, h_rk_rep.dtype)
        G_inj    = G_rep    + complex_awgn(G_rep.shape,    INJECTION_VARIANCE, DEVICE, G_rep.dtype)
        g_dt_inj = g_dt_rep + complex_awgn(g_dt_rep.shape, INJECTION_VARIANCE, DEVICE, g_dt_rep.dtype)

        theta_rep = theta_batch.unsqueeze(1).expand(B, s, N).reshape(B * s, N)
        W_C_rep   = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        W_R_rep   = W_R.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        sinrs = comm_net.compute_comm_sinrs(
            h_dk=h_dk_inj,
            h_rk=h_rk_inj,
            G=G_inj,
            theta=theta_rep,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE
        )
        rates = comm_net.compute_rates(sinrs)
        sumrate = rates.sum(dim=1)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS
        ).real

        sumrate_chunks.append(sumrate.reshape(B, s))
        snr_chunks.append(sense_snr.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)   # (B,S)
    snr_samples     = torch.cat(snr_chunks, dim=1)       # (B,S)

    # 4) robust objective
    sumrate_mean_per_sample = sumrate_samples.mean(dim=1)
    sumrate_mean = sumrate_mean_per_sample.mean()

    q = float(OUTAGE_QUANTILE)
    k = max(1, int(np.ceil(q * S)))
    snr_var = torch.kthvalue(snr_samples, k=k, dim=1).values
    snr_penalty = torch.clamp(SENSING_SNR_THRESHOLD - snr_var, min=0.0).mean()

    objective = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT * snr_penalty
        - TX_POWER_LOSS_WEIGHT * tx_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_snr_mean_db": (10.0 * torch.log10(snr_var.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "tx_penalty_mean": tx_penalty.detach(),
        "objective": objective.detach(),
    }
    return objective, logs


@torch.no_grad()
def validate_shortterm_robust(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_layout_bank,
):
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    obj_ep = 0.0
    sumrate_ep = 0.0
    snr_db_ep = 0.0

    layout_batch = sample_layout_batch(val_layout_bank, SHORTTERM_VAL_LAYOUTS_PER_EPOCH)

    for layout in layout_batch:
        theta_fixed = get_theta_from_longterm(longterm_net, layout)
        obj, logs = forward_shortterm_robust_objective(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            layout=layout,
            batch_size=SHORTTERM_VAL_CHANNEL_BATCH,
            injection_samples=ROBUST_VAL_INJECTION_SAMPLES,
        )
        denom = len(layout_batch)
        obj_ep += float(obj.detach().cpu()) / denom
        sumrate_ep += float(logs["sum_rate_mean"].cpu()) / denom
        snr_db_ep += float(logs["sense_snr_mean_db"].cpu()) / denom

    return obj_ep, sumrate_ep, snr_db_ep


# ============================================================
# Training loops
# ============================================================
def train_longterm(longterm_net, train_layout_bank, val_layout_bank):
    optimizer = optim.Adam(longterm_net.parameters(), lr=LEARNING_RATE)

    best_val_obj = -np.inf
    curves = []

    curve_path = os.path.join(CURVE_DIR, f"longterm_curves_{SETTING_STRING}.npy")

    for ep in trange(1, EPOCHS + 1, desc="LongTerm"):
        longterm_net.train()
        train_obj_ep = 0.0
        train_sumrate_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_batch = sample_layout_batch(train_layout_bank, LONGTERM_BATCH_LAYOUTS) # 在每個mini batch中 有 LONGTERM_BATCH_LAYOUTS (32)個 被train_layout_bank從抽出

            optimizer.zero_grad(set_to_none=True)
            obj, logs = forward_longterm_objective(
                longterm_net=longterm_net,
                layout_batch=layout_batch,
                mc_samples=LONGTERM_MC_TRAIN,
            )
            (-obj).backward()
            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES

        val_obj_ep, val_sumrate_ep = validate_longterm(
            longterm_net=longterm_net,
            val_layout_bank=val_layout_bank,
        )

        curves.append([train_obj_ep, val_obj_ep, train_sumrate_ep, val_sumrate_ep])
        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[LongTerm Epoch {ep:03d}] "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_obj_ep:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_sumrate_ep:.4e}"
        )

        if val_obj_ep > best_val_obj:
            best_val_obj = val_obj_ep
            longterm_net.save_model(verbose=False)

    print(f"[LongTerm] best ValObj = {best_val_obj:.4e}")
    longterm_net.load_model(verbose=True)


def train_shortterm_regular(longterm_net, comm_net, radar_net, train_layout_bank, val_layout_bank):
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

    for ep in trange(1, EPOCHS + 1, desc="ShortTerm-Regular"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            layout = sample_layout_batch(train_layout_bank, 1)[0]
            theta_fixed = get_theta_from_longterm(longterm_net, layout)

            optimizer.zero_grad(set_to_none=True)
            obj, logs = forward_shortterm_regular_objective(
                comm_net=comm_net,
                radar_net=radar_net,
                theta_fixed=theta_fixed,
                layout=layout,
                batch_size=BATCH_SIZE,
            )
            (-obj).backward()
            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_snr_db_ep += float(logs["sense_snr_mean_db"].cpu()) / MINIBATCHES

        val_obj_ep, val_sumrate_ep, val_snr_db_ep = validate_shortterm_regular(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_layout_bank=val_layout_bank,
        )

        curves.append([train_obj_ep, val_obj_ep, train_sumrate_ep, val_sumrate_ep, train_snr_db_ep, val_snr_db_ep])
        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[Short-Reg Epoch {ep:03d}] "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_obj_ep:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_sumrate_ep:.4e} | "
            f"TrainSNR(dB)={train_snr_db_ep:.3f} | ValSNR(dB)={val_snr_db_ep:.3f}"
        )

        if val_obj_ep > best_val_obj:
            best_val_obj = val_obj_ep
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)

    print(f"[Short-Reg] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


def train_shortterm_robust(longterm_net, comm_net, radar_net, train_layout_bank, val_layout_bank):
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

    for ep in trange(1, EPOCHS + 1, desc="ShortTerm-Robust"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0

        for _ in range(MINIBATCHES):
            layout = sample_layout_batch(train_layout_bank, 1)[0]
            theta_fixed = get_theta_from_longterm(longterm_net, layout)

            optimizer.zero_grad(set_to_none=True)
            obj, logs = forward_shortterm_robust_objective(
                comm_net=comm_net,
                radar_net=radar_net,
                theta_fixed=theta_fixed,
                layout=layout,
                batch_size=BATCH_SIZE,
                injection_samples=INJECTION_SAMPLES,
            )
            (-obj).backward()
            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_snr_db_ep += float(logs["sense_snr_mean_db"].cpu()) / MINIBATCHES

        val_obj_ep, val_sumrate_ep, val_snr_db_ep = validate_shortterm_robust(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_layout_bank=val_layout_bank,
        )

        curves.append([train_obj_ep, val_obj_ep, train_sumrate_ep, val_sumrate_ep, train_snr_db_ep, val_snr_db_ep])
        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[Short-Rob Epoch {ep:03d}] "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_obj_ep:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_sumrate_ep:.4e} | "
            f"TrainVaR-SNR(dB)={train_snr_db_ep:.3f} | ValVaR-SNR(dB)={val_snr_db_ep:.3f}"
        )

        if val_obj_ep > best_val_obj:
            best_val_obj = val_obj_ep
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)

    print(f"[Short-Rob] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-timescale ISAC training")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "longterm", "short_reg", "short_rob"],
        help="要執行的訓練階段"
    )
    args = parser.parse_args()

    train_layout_bank, val_layout_bank = split_layout_bank(
        UE_LAYOUT_BANK,
        train_ratio=LAYOUT_TRAIN_RATIO,
        seed=RANDOM_SEED
    )

    print(f"[INFO] total layouts = {len(UE_LAYOUT_BANK)}")
    print(f"[INFO] train layouts = {len(train_layout_bank)}")
    print(f"[INFO] val layouts = {len(val_layout_bank)}")

    # --------------------------
    # 建立網路
    # --------------------------
    longterm_net = LongTermPositionNet(ckpt_kind="longterm").to(DEVICE)

    short_comm_reg = ShortTermCommNet(ckpt_kind="short_comm").to(DEVICE)
    short_radar_reg = ShortTermRadarNet(ckpt_kind="short_radar").to(DEVICE)

    short_comm_rob = ShortTermCommNet(ckpt_kind="short_comm_robust").to(DEVICE)
    short_radar_rob = ShortTermRadarNet(ckpt_kind="short_radar_robust").to(DEVICE)

    # --------------------------
    # 執行模式
    # --------------------------
    if args.mode == "all":
        print("\n========== Stage 1: Train LongTerm ==========")
        train_longterm(longterm_net, train_layout_bank, val_layout_bank)

        print("\n========== Stage 2: Train ShortTerm Regular ==========")
        train_shortterm_regular(
            longterm_net=longterm_net,
            comm_net=short_comm_reg,
            radar_net=short_radar_reg,
            train_layout_bank=train_layout_bank,
            val_layout_bank=val_layout_bank,
        )

        print("\n========== Stage 3: Train ShortTerm Robust ==========")
        train_shortterm_robust(
            longterm_net=longterm_net,
            comm_net=short_comm_rob,
            radar_net=short_radar_rob,
            train_layout_bank=train_layout_bank,
            val_layout_bank=val_layout_bank,
        )

    elif args.mode == "longterm":
        train_longterm(longterm_net, train_layout_bank, val_layout_bank)

    elif args.mode == "short_reg":
        longterm_net.load_model(verbose=True)
        train_shortterm_regular(
            longterm_net=longterm_net,
            comm_net=short_comm_reg,
            radar_net=short_radar_reg,
            train_layout_bank=train_layout_bank,
            val_layout_bank=val_layout_bank,
        )

    elif args.mode == "short_rob":
        longterm_net.load_model(verbose=True)
        train_shortterm_robust(
            longterm_net=longterm_net,
            comm_net=short_comm_rob,
            radar_net=short_radar_rob,
            train_layout_bank=train_layout_bank,
            val_layout_bank=val_layout_bank,
        )

    print("\nTraining finished.")