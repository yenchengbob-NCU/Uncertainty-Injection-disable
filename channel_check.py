import os
import math
import torch
import numpy as np
from settings import *
from two_timescale_NN import CommNet

# ============================================================
# Helpers
# ============================================================

def to_db_np(x, eps=1e-30):
    x = np.asarray(x, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(x, eps))


def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def channel_power(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.mean(np.abs(x) ** 2)


if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    dataset_path = os.path.join(DATA_DIR, "dataset_train.npz")        # 這裡讀資料
    dataset = physics_net.load_channel_dataset(dataset_path, "train")

    # 印出UE位置
    print("=" * 90)
    print("[UE positions: 角度以 RIS 為圓心、全域 x 軸正方向為 0 度]")
    ue_layout = np.asarray(UE_LAYOUT, dtype=np.float32)
    for k in range(UAV_COMM):
        dx = ue_layout[k, 0] - Q_RIS[0]
        dy = ue_layout[k, 1] - Q_RIS[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        if angle_deg < 0:
            angle_deg += 360.0
        print(f"UE{k:<2d}: [{ue_layout[k, 0]:>8.4f}, {ue_layout[k, 1]:>8.4f}]  # angle = {angle_deg:7.3f} deg")
    print("-" * 90)

    # 取出固定 layout 下的資料
    h_dk = torch.as_tensor(dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, K)
    h_rk = torch.as_tensor(dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)   # (B, N, K)
    G    = torch.as_tensor(dataset["G_hat"],dtype=torch.complex64,device=DEVICE)      # (B, N, M)
    g_dt = torch.as_tensor(dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)   # (B, M, 1)

    pl_BS_UE      = np.asarray(dataset["pl_BS_UE"]).reshape(-1)                       # Scalar
    pl_RIS_UE     = np.asarray(dataset["pl_RIS_UE"]).reshape(-1)
    pl_BS_RIS     = np.asarray(dataset["pl_BS_RIS"]).reshape(-1)
    pl_BS_RIS_UE  = np.asarray(dataset["pl_BS_RIS_UE"]).reshape(-1)
    pl_BS_TAR     = np.asarray(dataset["pl_BS_TAR"]).reshape(-1)
    pl_BS_TAR_BS  = np.asarray(dataset["pl_BS_TAR_BS"]).reshape(-1)

    # 印出pathloss
    print("[Path loss, unit = dB]")
    print(f"{'pl_BS_UE':<13s} {fmt_vec(to_db_np(pl_BS_UE), precision=3):<45s} BS -> UE direct one-way")
    print(f"{'pl_RIS_UE':<13s} {fmt_vec(to_db_np(pl_RIS_UE), precision=3):<45s} RIS -> UE one-way")
    print(f"{'pl_BS_RIS':<13s} {fmt_vec(to_db_np(pl_BS_RIS), precision=3):<45s} BS -> RIS one-way")
    # print(f"{'pl_BS_RIS_UE':<13s} {fmt_vec(to_db_np(pl_BS_RIS_UE), precision=3):<45s} BS -> RIS -> UE cascaded")
    print(f"{'pl_BS_TAR':<13s} {fmt_vec(to_db_np(pl_BS_TAR), precision=3):<45s} BS -> target one-way")
    print(f"{'pl_BS_TAR_BS':<13s} {fmt_vec(to_db_np(pl_BS_TAR_BS), precision=3):<45s} BS -> target -> BS round-trip")
    print("-" * 90)

    #算出通道功率
    theta_check = torch.ones(G.shape[0], G.shape[1], dtype=torch.complex64, device=DEVICE) # RIS theta設為為1向量 也就是設RIS為單位矩陣

    H_direct_H = torch.conj(h_dk).transpose(1, 2)                 # (B,K,M)
    h_rk_H = torch.conj(h_rk).transpose(1, 2)                     # (B,K,N)
    H_ris_H = torch.einsum("bkn,bn,bnm->bkm", h_rk_H, theta_check, G)
    H_eff_H = H_direct_H + H_ris_H

    gH = torch.conj(g_dt).transpose(1, 2)                         # (B,1,M)
    G_sensing = torch.matmul(g_dt, gH)                            # (B,M,M)

    # 算出各通道的平均線性功率
    p_h_dk = channel_power(h_dk)
    p_h_rk = channel_power(h_rk)
    p_G = channel_power(G)
    p_g_dt = channel_power(g_dt)
    p_H_ris = channel_power(H_ris_H)
    p_G_sensing = channel_power(G_sensing)

    print("[Channel power: dB, linear power, RMS amplitude]")
    print(f"{'h_dk_hat':<15s} {to_db_np(p_h_dk):>9.3f} dB | power={p_h_dk:.6e} | rms={math.sqrt(p_h_dk):.6e}    BS -> UE direct channel")
    print(f"{'h_rk_hat':<15s} {to_db_np(p_h_rk):>9.3f} dB | power={p_h_rk:.6e} | rms={math.sqrt(p_h_rk):.6e}    RIS -> UE channel")
    print(f"{'G_hat':<15s} {to_db_np(p_G):>9.3f} dB | power={p_G:.6e} | rms={math.sqrt(p_G):.6e}    BS -> RIS channel")
    print(f"{'g_dt_hat':<15s} {to_db_np(p_g_dt):>9.3f} dB | power={p_g_dt:.6e} | rms={math.sqrt(p_g_dt):.6e}    BS -> target one-way channel")
    print(f"{'H_ris_H':<15s} {to_db_np(p_H_ris):>9.3f} dB | power={p_H_ris:.6e} | rms={math.sqrt(p_H_ris):.6e}    BS -> RIS -> UE equivalent channel")
    print(f"{'G_sensing':<15s} {to_db_np(p_G_sensing):>9.3f} dB | power={p_G_sensing:.6e} | rms={math.sqrt(p_G_sensing):.6e}    BS -> target -> BS sensing matrix")




