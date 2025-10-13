# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
from settings import *
from neural_net import Regular_Net, Robust_Net, _normalize_columns
import matplotlib.pyplot as plt

def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)

@torch.no_grad()
def q_min_rate_vector_under_uncertainty(model, H_est_t: torch.Tensor, H_L: torch.Tensor, L: int) -> np.ndarray:
    """
    用 H_est 設計 W，於擾動通道 H_L 上評估，回傳每個 layout 的 q-分位(min-user rate) 向量，shape=(N,)
    """
    N = H_est_t.shape[0]
    # 設計 precoder：B_raw -> B_dir(L2) -> 等功率 W
    B_raw = model.get_beamformer(H_est_t)          # (N,M,K)
    B_dir = _normalize_columns(B_raw)              # (N,M,K)
    W     = apply_power_allocation_torch(B_dir)    # (N,M,K)

    # 重複 W 到 (L*N, M, K) —— 使用 repeat（不使用 expand）
    W_L = W.repeat(L, 1, 1)                        # (L*N, M, K)

    # 在擾動通道上計算 SINR → rate → min-user rate
    sinrs_L = model.compute_comm_sinrs(H_L, W_L)   # (L*N, K)
    rates_L = model.compute_rates(sinrs_L).view(L, N, K)
    min_rates_L = torch.min(rates_L, dim=2).values # (L, N)

    # 每個 layout 的 q-分位（沿 L 取分位）
    q_vec = torch.quantile(min_rates_L, q=OUTAGE_QUANTILE, dim=0)  # (N,)
    return q_vec.cpu().numpy()

@torch.no_grad()
def q_min_rate_under_uncertainty(model, H_est_t: torch.Tensor, H_L: torch.Tensor, L: int) -> float:
    """
    回傳 q-分位(min-user rate) 的樣本平均（標量），供表格/列印用。
    """
    q_vec = q_min_rate_vector_under_uncertainty(model, H_est_t, H_L, L)  # (N,)
    return float(q_vec.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAC evaluate: Regular vs Robust (q-quantile)")
    parser.add_argument("--channels", type=str, default=None,
                        help=f"Path to offline test channel .npy. Default = Data/channelEstimates_test_{SETTING_STRING}.npy")
    parser.add_argument("--L", type=int, default=None,
                        help="Number of uncertainty realizations (default = Robust_Net.L)")
    args = parser.parse_args()

    # 1) 載入離線 H_est
    base_dir = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
    test_dir = os.path.join(base_dir, "channelEstimates_test")
    ch_path = args.channels if args.channels else os.path.join(test_dir, f"{SETTING_STRING}.npy")

    if not os.path.exists(ch_path):
        raise FileNotFoundError(f"找不到測試通道檔案：{ch_path}")
    
    H_est_np = np.load(ch_path)                 # (N,M,K) complex
    H_est_t  = np_to_torch_complex(H_est_np)
    N = H_est_t.shape[0]
    print(f"[EVAL] Loaded test channels: {ch_path} (N={N}, M={M}, K={K})")

    # 2) 準備兩個模型（自動嘗試載入各自 checkpoint）
    reg = Regular_Net().to(DEVICE).eval()
    rob = Robust_Net().to(DEVICE).eval()

    # 3) 產生同一組擾動通道（與估測誤差脫鉤；不考慮角度抖動）
    L = args.L if args.L is not None else rob.L
    with torch.no_grad():
        H_L = rob.inject_uncertainties(H_est_t)       # (L*N, M, K)

    # 4) 計算兩個模型的 q-分位(min-user rate)（標量）
    with torch.no_grad():
        reg_q = q_min_rate_under_uncertainty(reg, H_est_t, H_L, L)
        rob_q = q_min_rate_under_uncertainty(rob, H_est_t, H_L, L)

    # 5) 結果（只報 q=OUTAGE_QUANTILE 的目標）
    q_pct = int(OUTAGE_QUANTILE * 100)
    print(f"\n=== Quantile Objective (q = {q_pct}%) ===")
    print(f"Regular_Net q-min-rate : {reg_q:.4f} bits/s/Hz")
    print(f"Robust_Net  q-min-rate : {rob_q:.4f} bits/s/Hz")
    winner = "Robust_Net" if rob_q > reg_q else "Regular_Net"
    print(f"Winner: {winner}")

    # 6) 畫 CDF：每個 layout 的 q-分位(min-user rate)
    with torch.no_grad():
        reg_q_vec = q_min_rate_vector_under_uncertainty(reg, H_est_t, H_L, L)  # (N,)
        rob_q_vec = q_min_rate_vector_under_uncertainty(rob, H_est_t, H_L, L)  # (N,)

    xs_reg = np.sort(reg_q_vec)
    xs_rob = np.sort(rob_q_vec)
    cdf    = np.arange(1, N + 1) / N

    plt.figure()
    plt.plot(xs_reg, cdf, label="Regular_Net")
    plt.plot(xs_rob, cdf, label="Robust_Net")
    plt.xlabel(f"{q_pct}% quantile of min-user rate (bits/s/Hz)")
    plt.ylabel("Empirical CDF")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(f"CDF of {q_pct}% Quantile Min-Rate — {SETTING_STRING}")
    plt.tight_layout()
    out_fig_path = os.path.join(base_dir, f"CDF_q{q_pct:02d}.png")
    plt.savefig(out_fig_path)
    print(f"[EVAL] CDF saved to: {out_fig_path}")
    plt.show()
