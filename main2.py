# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from settings import *
from rician import large_scale_fading
from neural_net import *

def plot(curve_dir: str):
    regular_curves_path = os.path.join(curve_dir, f"training_curves_regular_{SETTING_STRING}.npy")
    robust_curves_path  = os.path.join(curve_dir, f"training_curves_robust_{SETTING_STRING}.npy")

    if not os.path.exists(regular_curves_path):
        raise FileNotFoundError(f"找不到 regular 曲線檔案：{regular_curves_path}")
    if not os.path.exists(robust_curves_path):
        raise FileNotFoundError(f"找不到 robust 曲線檔案：{robust_curves_path}")

    reg_curves = np.load(regular_curves_path)  # rows: [train_obj, val_obj, val_sum_rate, val_sense_snr_db]
    rob_curves = np.load(robust_curves_path)

    # ===============================
    # 取出欄位
    # ===============================
    reg_train = reg_curves[:, 0]
    reg_val   = reg_curves[:, 1]
    reg_val_sum_rate = reg_curves[:, 2]
    reg_val_snr_db   = reg_curves[:, 3]

    rob_train = rob_curves[:, 0]
    rob_val   = rob_curves[:, 1]
    rob_val_sum_rate = rob_curves[:, 2]
    rob_val_snr_db   = rob_curves[:, 3]

    x_reg = np.arange(1, len(reg_train) + 1)
    x_rob = np.arange(1, len(rob_train) + 1)

    # ===============================
    # Generalization gap
    # objective 越大越好，所以 gap = train - val
    # gap 越大，通常代表越有過擬合傾向
    # ===============================
    reg_gap = reg_train - reg_val
    rob_gap = rob_train - rob_val

    # 圖片輸出資料夾
    fig_dir = os.path.join(curve_dir, "plot_figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ===============================
    # 1) Regular: Train vs Validation Objective
    # ===============================
    plt.figure()
    plt.plot(x_reg, reg_train, label="REG TrainObj")
    plt.plot(x_reg, reg_val,   label="REG ValObj")
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title(f"REG Train/Validation Objective — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"REG_train_val_obj_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ===============================
    # 2) Robust: Train vs Validation Objective
    # ===============================
    plt.figure()
    plt.plot(x_rob, rob_train, label="ROB TrainObj")
    plt.plot(x_rob, rob_val,   label="ROB ValObj")
    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title(f"ROB Train/Validation Objective — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"ROB_train_val_obj_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ===============================
    # 3) Generalization Gap comparison
    # ===============================
    plt.figure()
    plt.plot(x_reg, reg_gap, label="REG Gap = TrainObj - ValObj")
    plt.plot(x_rob, rob_gap, label="ROB Gap = TrainObj - ValObj")
    plt.xlabel("Epoch")
    plt.ylabel("Generalization Gap")
    plt.title(f"Generalization Gap — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"generalization_gap_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ===============================
    # 4) Validation SumRate comparison
    # ===============================
    plt.figure()
    plt.plot(x_reg, reg_val_sum_rate, label="REG Val SumRate")
    plt.plot(x_rob, rob_val_sum_rate, label="ROB Val SumRate")
    plt.xlabel("Epoch")
    plt.ylabel("Validation SumRate (bits/s/Hz)")
    plt.title(f"Validation SumRate — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"validation_sumrate_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()

    # ===============================
    # 5) Validation SNR comparison
    # ===============================
    plt.figure()
    plt.plot(x_reg, reg_val_snr_db, label="REG Val SNR (dB)")
    plt.plot(x_rob, rob_val_snr_db, label="ROB Val SNR (dB)")
    plt.xlabel("Epoch")
    plt.ylabel("Validation SNR (dB)")
    plt.title(f"Validation SNR — {SETTING_STRING}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"validation_snr_{SETTING_STRING}.jpg"), format="jpg")
    plt.show()
    plt.close()


def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    """numpy 複數陣列 -> torch.complex64 到 DEVICE。"""
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


# ------------------------------
# Regular forward objective
# ------------------------------

def forward_objective(comm_net, sense_net, ris_net,
                      h_dk, h_rk, G, g_dt,
                      pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS):
    """
    4-channel input:
      h_dk: (B,M,K), h_rk: (B,N,K), G: (B,N,M), g_dt: (B,M,1)
    beta are power fading, used only in SINR/SNR computations.
    """

    # 1) 三個網路各自輸出
    W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
    W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)
    phi = ris_net(h_dk, h_rk, G, g_dt)    # (B,N)

    # 2) 通訊 SINR / 速率
    sinrs = comm_net.compute_comm_sinrs(
        h_dk, h_rk, G, phi, W_S, W_C,
        pl_BS_UE, pl_BS_RIS_UE
    )  # (B,K)

    rates = comm_net.compute_rates(sinrs)     # (B,K)
    sum_rate = rates.sum(dim=1)               # (B,)
    sum_rate_mean = sum_rate.mean()           # scalar (對batch 取平均)

    # 3) 感測 SNR
    sense_snr = comm_net.compute_sense_snr(g_dt, W_S, W_C, pl_BS_TAR_BS)            # (B,)

    snr_violation = torch.clamp(SENSING_SNR_THRESHOLD - sense_snr.real, min=0.0)    # (B,) 如果小於門檻就懲罰
    snr_penalty_mean = snr_violation.mean()                                          # scalar (對batch 取平均) L3

    # 4) φ 懲罰
    phi_abs = phi.abs()                                                              # (B,N)
    phi_excess = torch.clamp(phi_abs - 1.0, min=0.0)                                 # (B,N)
    phi_penalty_mean = (phi_excess).mean(dim=1).mean()                               # scalar 對batch 取平均 L1 (可以打開平方懲罰_目前關閉)

    # 5) TX power 懲罰
    power_comm  = (W_C.abs() ** 2).sum(dim=(1, 2))                                   # (B,)
    power_sense = (W_S.abs() ** 2).sum(dim=(1, 2))                                   # (B,)
    tx_power = power_comm + power_sense                                              # (B,)
    tx_power_mean = tx_power.mean()

    tx_excess = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0)                # (B,)
    tx_penalty_mean = (tx_excess).mean()                                             # scalar 對batch 取平均 L2 (可以打開平方懲罰_目前關閉)

    # 6) objective
    objective = (
        sum_rate_mean
        - SENSING_LOSS_WEIGHT  * snr_penalty_mean
        - RE_POWER_LOSS_WEIGHT * phi_penalty_mean
        - TX_POWER_LOSS_WEIGHT * tx_penalty_mean
    )

    logs = {
        "sum_rate_mean":        sum_rate_mean.detach(),
        "sense_snr_mean_db":    (10.0 * torch.log10(sense_snr.real.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean":     snr_penalty_mean.detach(),
        "phi_penalty_mean":     phi_penalty_mean.detach(),
        "tx_power_mean":        tx_power_mean.detach(),
        "tx_penalty_mean":      tx_penalty_mean.detach(),
        "objective":            objective.detach(),
    }
    return objective, logs


# ------------------------------
# Robust forward objective (pl version)
# ------------------------------
def forward_objective_robust(comm_net, sense_net, ris_net,
                             h_dk, h_rk, G, g_dt,
                             pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS):
    """
    Robust objective (pl version):
    - NN 輸入：估測通道 (h_dk, h_rk, G, g_dt)
    - 評估 SINR/SNR 前：對四條通道做 additive injection
    - 複製 INJECTION_SAMPLES 次
    - 用 OUTAGE_QUANTILE 的 quantile 做 sum-rate 目標 + sensing constraint
    """

    def _add_complex_awgn(x: torch.Tensor, variance: float) -> torch.Tensor:
        device = x.device
        sigma = torch.sqrt(torch.as_tensor(variance / 2.0, device=device, dtype=torch.float32))
        nr = sigma * torch.randn(x.shape, device=device, dtype=torch.float32)
        ni = sigma * torch.randn(x.shape, device=device, dtype=torch.float32)
        n = torch.complex(nr, ni).to(dtype=x.dtype)
        return x + n

    # 1) design variables (no injection)
    W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
    W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)
    phi = ris_net(h_dk, h_rk, G, g_dt)    # (B,N)

    # 2) penalties (compute once)
    phi_abs = phi.abs()
    phi_excess = torch.clamp(phi_abs - 1.0, min=0.0)
    phi_penalty_mean = (phi_excess).mean(dim=1).mean()             #變成scalar

    power_comm  = (W_C.abs() ** 2).sum(dim=(1, 2))
    power_sense = (W_S.abs() ** 2).sum(dim=(1, 2))
    tx_power = power_comm + power_sense
    tx_power_mean = tx_power.mean()

    tx_excess = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0)
    tx_penalty_mean = (tx_excess).mean()                           #變成scalar

    # 3) robust performance via injection + quantile
    B, M, K = h_dk.shape

    chunk = 50  # memory guard
    sumrate_chunks = []
    snr_chunks = []

    # CHUNK  把1000個noise拆50 * 20  最後再組回去

    for s0 in range(0, INJECTION_SAMPLES, chunk):
        s = min(chunk, INJECTION_SAMPLES - s0)

        h_dk_rep = h_dk.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk.unsqueeze(1).expand(B, s, RIS_UNIT, K).reshape(B * s, RIS_UNIT, K)
        G_rep    = G.unsqueeze(1).expand(B, s, RIS_UNIT, M).reshape(B * s, RIS_UNIT, M)
        g_dt_rep = g_dt.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        h_dk_inj = _add_complex_awgn(h_dk_rep, INJECTION_VARIANCE)
        h_rk_inj = _add_complex_awgn(h_rk_rep, INJECTION_VARIANCE)
        G_inj    = _add_complex_awgn(G_rep,    INJECTION_VARIANCE)
        g_dt_inj = _add_complex_awgn(g_dt_rep, INJECTION_VARIANCE)
        # 複製通道
        W_C_rep = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        W_S_rep = W_S.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)
        phi_rep = phi.unsqueeze(1).expand(B, s, RIS_UNIT).reshape(B * s, RIS_UNIT)

        sinrs = comm_net.compute_comm_sinrs(
            h_dk_inj, h_rk_inj, G_inj, phi_rep, W_S_rep, W_C_rep,
            pl_BS_UE, pl_BS_RIS_UE
        )  # (B*s,K)

        rates = comm_net.compute_rates(sinrs)
        sum_rate = rates.sum(dim=1)  # (B*s,)

        sense_snr = comm_net.compute_sense_snr(g_dt_inj, W_S_rep, W_C_rep, pl_BS_TAR_BS)  # (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))
        snr_chunks.append(sense_snr.real.reshape(B, s))

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)  # (B,S) Chunk 拼回 RATE
    snr_samples     = torch.cat(snr_chunks,     dim=1)  # (B,S) Chunk 拼回 SNR

    # ============================================================
    # 依照圖片公式：
    # max E[SumRate] - λ max(Γ_th - VaR_0.05(SNR), 0)
    # ============================================================
    eps = 1e-12
    B, S = sumrate_samples.shape
    q = float(OUTAGE_QUANTILE)

    # 1) 通訊目標：E[SumRate] 以 injection 的平均近似
    #    sumrate_samples: (B,S) -> per-sample mean: (B,) -> batch mean: scalar
    sumrate_mean_per_sample = sumrate_samples.mean(dim=1)     # (B,)
    sumrate_mean = sumrate_mean_per_sample.mean()             # scalar = E[SumRate]

    # 2) 機率約束 surrogate：VaR_q(SNR) >= Γ_th
    #    用 kthvalue 取每個 b 的第 k 小 SNR（k = ceil(q*S)）
    k = max(1, int(np.ceil(q * S)))                            # S=1000,q=0.05 -> k=50
    snr_var, _ = torch.kthvalue(snr_samples, k=k, dim=1)      # (B,) = VaR_q(SNR)

    # hinge penalty：max(Γ_th - VaR, 0)
    snr_violation_var = torch.clamp(SENSING_SNR_THRESHOLD - snr_var, min=0.0)  # (B,)
    snr_penalty_mean  = snr_violation_var.mean()                                # scalar

    # 3) objective：照圖 + 你原本的 phi / tx penalty
    objective = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT  * snr_penalty_mean
        - RE_POWER_LOSS_WEIGHT * phi_penalty_mean
        - TX_POWER_LOSS_WEIGHT * tx_penalty_mean
    )

    # 4) logs：維持你 regular 的 keys
    logs = {
        "sum_rate_mean":        sumrate_mean.detach(),  # E[SumRate]
        "sense_snr_mean_db":    (10.0 * torch.log10(snr_var.clamp_min(eps))).mean().detach(),  # mean VaR SNR (dB)
        "snr_penalty_mean":     snr_penalty_mean.detach(),   # hinge on VaR(SNR)
        "phi_penalty_mean":     phi_penalty_mean.detach(),
        "tx_power_mean":        tx_power_mean.detach(),
        "tx_penalty_mean":      tx_penalty_mean.detach(),
        "objective":            objective.detach(),
    }
    return objective, logs


# ------------------------------
# Robust validation (hard probability penalty)
# ------------------------------
@torch.no_grad()
def forward_validation_robust(comm_net, sense_net, ris_net,
                              h_dk, h_rk, G, g_dt,
                              pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS):
    """
    Robust validation:
    - NN輸入:估測通道
    - 對每條估測通道做 N_VAL 次 uncertainty injection
    - ValObj 用硬機率版本 penalty
    - 顯示值先保留：
        1) sum_rate_mean  = batch平均 injection平均後的 SumRate
        2) sense_snr_mean_db = batch平均 injection平均後的 SNR(dB)
    """

    def _add_complex_awgn(x: torch.Tensor, variance: float) -> torch.Tensor:
        device = x.device
        sigma = torch.sqrt(torch.as_tensor(variance / 2.0, device=device, dtype=torch.float32))
        nr = sigma * torch.randn(x.shape, device=device, dtype=torch.float32)
        ni = sigma * torch.randn(x.shape, device=device, dtype=torch.float32)
        n = torch.complex(nr, ni).to(dtype=x.dtype)
        return x + n

    eps = 1e-12

    # 1) design variables (no injection)
    W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
    W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)
    phi = ris_net(h_dk, h_rk, G, g_dt)    # (B,N)

    # 2) penalties (compute once)
    phi_abs = phi.abs()
    phi_excess = torch.clamp(phi_abs - 1.0, min=0.0)
    phi_penalty_mean = phi_excess.mean(dim=1).mean()   # scalar

    power_comm  = (W_C.abs() ** 2).sum(dim=(1, 2))     # (B,)
    power_sense = (W_S.abs() ** 2).sum(dim=(1, 2))     # (B,)
    tx_power = power_comm + power_sense                # (B,)
    tx_power_mean = tx_power.mean()                    # scalar

    tx_excess = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0)
    tx_penalty_mean = tx_excess.mean()                 # scalar

    # 3) validation under injections
    B, M, K = h_dk.shape
    N = h_rk.shape[1]
    S = int(N_VAL)          # validation 用的 injection 次數
    chunk = 50              # memory guard

    sumrate_chunks = []
    snr_chunks = []

    for s0 in range(0, S, chunk):
        s = min(chunk, S - s0)

        h_dk_rep = h_dk.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        h_rk_rep = h_rk.unsqueeze(1).expand(B, s, N, K).reshape(B * s, N, K)
        G_rep    = G.unsqueeze(1).expand(B, s, N, M).reshape(B * s, N, M)
        g_dt_rep = g_dt.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)

        h_dk_inj = _add_complex_awgn(h_dk_rep, INJECTION_VARIANCE)
        h_rk_inj = _add_complex_awgn(h_rk_rep, INJECTION_VARIANCE)
        G_inj    = _add_complex_awgn(G_rep,    INJECTION_VARIANCE)
        g_dt_inj = _add_complex_awgn(g_dt_rep, INJECTION_VARIANCE)

        W_C_rep = W_C.unsqueeze(1).expand(B, s, M, K).reshape(B * s, M, K)
        W_S_rep = W_S.unsqueeze(1).expand(B, s, M, 1).reshape(B * s, M, 1)
        phi_rep = phi.unsqueeze(1).expand(B, s, N).reshape(B * s, N)

        sinrs = comm_net.compute_comm_sinrs(
            h_dk_inj, h_rk_inj, G_inj, phi_rep, W_S_rep, W_C_rep,
            pl_BS_UE, pl_BS_RIS_UE
        )  # (B*s,K)

        rates = comm_net.compute_rates(sinrs)   # (B*s,K)
        sum_rate = rates.sum(dim=1)             # (B*s,)

        sense_snr = comm_net.compute_sense_snr(
            g_dt_inj, W_S_rep, W_C_rep, pl_BS_TAR_BS
        ).real                                  # (B*s,)

        sumrate_chunks.append(sum_rate.reshape(B, s))   # (B,s)
        snr_chunks.append(sense_snr.reshape(B, s))      # (B,s)

    sumrate_samples = torch.cat(sumrate_chunks, dim=1)  # (B,S)
    snr_samples     = torch.cat(snr_chunks,     dim=1)  # (B,S)

    # 4) validation metrics
    q = float(OUTAGE_QUANTILE)

    # ROB mean rate = batch平均 injection平均
    sumrate_mean_per_sample = sumrate_samples.mean(dim=1)     # (B,)
    sumrate_mean = sumrate_mean_per_sample.mean()             # scalar

    # ROB SNR violation probability = 每條通道自己的違反比例
    snr_vprob_per_sample = (snr_samples < SENSING_SNR_THRESHOLD).float().mean(dim=1)   # (B,)

    # 硬機率 penalty：超過 q 才罰
    snr_penalty_per_sample = torch.clamp(snr_vprob_per_sample - q, min=0.0)            # (B,)
    snr_penalty_mean = snr_penalty_per_sample.mean()                                    # scalar

    # 顯示用：ROB SNR = batch平均 injection平均後的 SNR(dB)
    sense_snr_mean_db = (10.0 * torch.log10(snr_samples.clamp_min(eps))).mean()

    # 5) objective
    objective = (
        sumrate_mean
        - SENSING_LOSS_WEIGHT  * snr_penalty_mean
        - RE_POWER_LOSS_WEIGHT * phi_penalty_mean
        - TX_POWER_LOSS_WEIGHT * tx_penalty_mean
    )

    logs = {
        "sum_rate_mean":        sumrate_mean.detach(),        # ROB: batch平均 injection平均後的 SumRate
        "sense_snr_mean_db":    sense_snr_mean_db.detach(),   # ROB: batch平均 injection平均後的 SNR(dB)
        "snr_penalty_mean":     snr_penalty_mean.detach(),
        "phi_penalty_mean":     phi_penalty_mean.detach(),
        "tx_power_mean":        tx_power_mean.detach(),
        "tx_penalty_mean":      tx_penalty_mean.detach(),
        "objective":            objective.detach(),
    }
    return objective, logs


# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAC training script (3-MLP joint) + robust (pl)")
    parser.add_argument("--plot", action="store_true", help="Plot regular vs robust curves and exit")
    args = parser.parse_args()

    # ===============================
    # --plot 畫出收斂圖
    # ===============================
    ckpt_dir  = CKPT_DIR
    curve_dir = CURVE_DIR

    # 曲線檔案
    curves_path = os.path.join(curve_dir, f"training_curves_{SETTING_STRING}.npy")

    # 只繪圖就退出
    if args.plot:
        plot(curve_dir=curve_dir)
        raise SystemExit(0)

    # ===============================
    # 建立 6 個網路（3 regular + 3 robust）與 optimizer
    # ===============================
    # --- Regular nets ---
    regular_comm_net   = CommBeamformerNet().to(DEVICE)
    regular_sense_net  = SenseBeamformerNet().to(DEVICE)
    regular_ris_net    = RISPhaseNet().to(DEVICE)
    # --- Robust nets ---
    robust_comm_net  = RobustCommBeamformerNet().to(DEVICE)
    robust_sense_net = RobustSenseBeamformerNet().to(DEVICE)
    robust_ris_net   = RobustRISPhaseNet().to(DEVICE)

    # --- Optimizers ---
    regular_optimizer = optim.Adam(
        list(regular_comm_net.parameters()) +
        list(regular_sense_net.parameters()) +
        list(regular_ris_net.parameters()),
        lr=LEARNING_RATE
    )

    robust_optimizer = optim.Adam(
        list(robust_comm_net.parameters()) +
        list(robust_sense_net.parameters()) +
        list(robust_ris_net.parameters()),
        lr=LEARNING_RATE
    )

    pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS = large_scale_fading()

    # checkpoint 路徑（最佳 val 依據 objective）
    best_val_regular = -np.inf
    best_val_robust  = -np.inf

    # early stopping state
    rob_wait = 0
    rob_stopped = False

    # 載入固定 Train & validation estimated channels (data_size筆)
    data = np.load(TRAIN_VAL_NPZ_PATH)

    all_h_dk = np_to_torch_complex(data["h_dk"])
    all_h_rk = np_to_torch_complex(data["h_rk"])
    all_G    = np_to_torch_complex(data["G"])
    all_g_dt = np_to_torch_complex(data["g_dt"])

    n_total = all_h_dk.shape[0]
    assert n_total == DATA_SIZE, f"npz 資料量 {n_total} 與 DATA_SIZE={DATA_SIZE} 不符"

    # 先 shuffle 一次，再切 7:3
    perm = torch.randperm(n_total, device=DEVICE)
    n_train = int(0.7 * n_total)
    n_val   = n_total - n_train

    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    print(f"[INFO] total={n_total}, train={n_train}, val={n_val}")

    # train 每個 epoch 需要的資料量
    required_train_samples = MINIBATCHES * BATCH_SIZE
    if required_train_samples > n_train:
        raise ValueError(
            f"Train split 不足以支撐目前設定："
            f"MINIBATCHES*BATCH_SIZE = {required_train_samples} > n_train = {n_train}"
        )

    if BATCH_SIZE > n_val:
        raise ValueError(
            f"Val split 不足以抽一個 validation batch："
            f"BATCH_SIZE = {BATCH_SIZE} > n_val = {n_val}"
        )

    # --- regular ckpt ---
    regular_comm_ckpt_path  = os.path.join(ckpt_dir, f"comm_{SETTING_STRING}.ckpt")
    regular_sense_ckpt_path = os.path.join(ckpt_dir, f"sense_{SETTING_STRING}.ckpt")
    regular_ris_ckpt_path   = os.path.join(ckpt_dir, f"ris_{SETTING_STRING}.ckpt")

    # --- robust ckpt ---
    robust_comm_ckpt_path   = os.path.join(ckpt_dir, f"comm_robust_{SETTING_STRING}.ckpt")
    robust_sense_ckpt_path  = os.path.join(ckpt_dir, f"sense_robust_{SETTING_STRING}.ckpt")
    robust_ris_ckpt_path    = os.path.join(ckpt_dir, f"ris_robust_{SETTING_STRING}.ckpt")

    # ===============================
    # training curves
    # 每 epoch 追加一行 [train_obj, val_obj, val_sum_rate, val_sense_snr_dB]
    # ===============================
    regular_curves_path = os.path.join(curve_dir, f"training_curves_regular_{SETTING_STRING}.npy")
    robust_curves_path  = os.path.join(curve_dir, f"training_curves_robust_{SETTING_STRING}.npy")

    regular_curves = []
    robust_curves  = []

    # ===============================
    # 訓練循環（同時訓練 regular + robust）
    # ===============================
    for ep in trange(1, EPOCHS + 1, desc="Epoch"):

        # ---------- TRAIN ----------
        regular_comm_net.train(); regular_sense_net.train(); regular_ris_net.train()
        robust_comm_net.train();  robust_sense_net.train();  robust_ris_net.train()

        # 每次清空
        reg_obj_ep = 0.0
        rob_obj_ep = 0.0

        # train 每個 epoch 重新 shuffle
        epoch_perm = torch.randperm(n_train, device=DEVICE)

        for num in range(MINIBATCHES):
            # 一個 MINIBATCHES 有 BATCH_SIZE 個通道 一個MINIBATCHES更新一次權重
            batch_ids = epoch_perm[num * BATCH_SIZE : (num + 1) * BATCH_SIZE]
            idx = train_idx[batch_ids]

            h_dk = all_h_dk[idx]
            h_rk = all_h_rk[idx]
            G    = all_G[idx]
            g_dt = all_g_dt[idx]

            # ---- (1) Regular update ----
            regular_optimizer.zero_grad(set_to_none=True)
            reg_objective, reg_logs = forward_objective(
                regular_comm_net, regular_sense_net, regular_ris_net,
                h_dk, h_rk, G, g_dt,
                pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS
            )
            (-reg_objective).backward()
            regular_optimizer.step()
            reg_obj_ep += reg_objective.item() / MINIBATCHES

            # ---- (2) Robust update ----
            robust_optimizer.zero_grad(set_to_none=True)
            rob_objective, rob_logs = forward_objective_robust(
                robust_comm_net, robust_sense_net, robust_ris_net,
                h_dk, h_rk, G, g_dt,
                pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS
            )
            (-rob_objective).backward()
            robust_optimizer.step()
            rob_obj_ep += rob_objective.item() / MINIBATCHES

        # ---------- VALIDATE ----------
        # 一個epoch 一次validation 紀錄curve 視情況存ckpt
        regular_comm_net.eval(); regular_sense_net.eval(); regular_ris_net.eval()
        robust_comm_net.eval();  robust_sense_net.eval();  robust_ris_net.eval()

        # 每個 epoch 從 30% val pool 隨機抽一個 BATCH_SIZE 做驗證
        val_perm = torch.randperm(n_val, device=DEVICE)
        val_batch_ids = val_perm[:BATCH_SIZE]
        idx_val = val_idx[val_batch_ids]

        val_h_dk = all_h_dk[idx_val]
        val_h_rk = all_h_rk[idx_val]
        val_G    = all_G[idx_val]
        val_g_dt = all_g_dt[idx_val]

        with torch.no_grad():
            # 使用 validation pool 隨機抽出的 estimated channels

            # --- regular val ---
            reg_val_obj_t, reg_val_logs = forward_objective(
                regular_comm_net, regular_sense_net, regular_ris_net,
                val_h_dk, val_h_rk, val_G, val_g_dt,
                pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS
            )
            reg_val_obj      = float(reg_val_obj_t.detach().cpu())
            reg_val_sum_rate = float(reg_val_logs["sum_rate_mean"].cpu())
            reg_val_snr_db   = float(reg_val_logs["sense_snr_mean_db"].cpu())

            # --- robust val ---
            rob_val_obj_t, rob_val_logs = forward_validation_robust(
                robust_comm_net, robust_sense_net, robust_ris_net,
                val_h_dk, val_h_rk, val_G, val_g_dt,
                pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS
            )
            rob_val_obj      = float(rob_val_obj_t.detach().cpu())
            rob_val_sum_rate = float(rob_val_logs["sum_rate_mean"].cpu())
            rob_val_snr_db   = float(rob_val_logs["sense_snr_mean_db"].cpu())

        # ===============================
        # 記錄 curves（分開存檔，供 --plot 使用）
        # 每行: [train_obj, val_obj, val_sum_rate, val_sense_snr_db]
        # ===============================
        regular_curves.append([reg_obj_ep, reg_val_obj, reg_val_sum_rate, reg_val_snr_db])
        robust_curves.append([rob_obj_ep, rob_val_obj, rob_val_sum_rate, rob_val_snr_db])

        np.save(regular_curves_path, np.array(regular_curves, dtype=np.float32))
        np.save(robust_curves_path,  np.array(robust_curves,  dtype=np.float32))

        # Log：REG/ROB 分兩行顯示
        print(
            f"[Epoch {ep:03d}]\n"
            f"  REG | TrainObj: {reg_obj_ep: .4e} | ValObj: {reg_val_obj: .4e} | Val SumRate: {reg_val_sum_rate: .4e} | Val SNR(dB): {reg_val_snr_db: .3f}\n"
            f"  ROB | TrainObj: {rob_obj_ep: .4e} | ValObj: {rob_val_obj: .4e} | Val SumRate: {rob_val_sum_rate: .4e} | Val SNR(dB): {rob_val_snr_db: .3f}"
        )

        # ---------- SAVE BEST CKPT ----------
        # REG：只存 best ckpt，不做 early stop
        if reg_val_obj > best_val_regular:
            best_val_regular = reg_val_obj
            torch.save(regular_comm_net.state_dict(),  regular_comm_ckpt_path)
            torch.save(regular_sense_net.state_dict(), regular_sense_ckpt_path)
            torch.save(regular_ris_net.state_dict(),   regular_ris_ckpt_path)

        # ROB：存 best ckpt，並累積 early stopping patience
        if rob_val_obj > best_val_robust + ROB_EARLY_STOPPING_MIN_DELTA:
            best_val_robust = rob_val_obj
            rob_wait = 0
            torch.save(robust_comm_net.state_dict(),  robust_comm_ckpt_path)
            torch.save(robust_sense_net.state_dict(), robust_sense_ckpt_path)
            torch.save(robust_ris_net.state_dict(),   robust_ris_ckpt_path)
        else:
            rob_wait += 1

        # ---------- APPLY EARLY STOPPING ----------
        # 只由 ROB 決定整體停止
        if EARLY_STOPPING_ENABLE and ep >= ROB_EARLY_STOPPING_WARMUP:
            if (not rob_stopped) and (rob_wait >= ROB_EARLY_STOPPING_PATIENCE):
                rob_stopped = True
                print(f"[EARLY STOP] ROB stopped at epoch {ep}, best ValObj = {best_val_robust:.4e}")
                break

print("Training Script finished!")