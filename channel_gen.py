import os
import numpy as np
from scipy.special import j0
from settings import *

RICIAN_KAPPA        = 2.0   # 線性尺度

# ================================
# 基本幾何 / 小工具
# ================================
def theta_calculater(p1, p2, normal):
    """
    計算從 p1 指向 p2,相對於陣列法線的 signed angle
    回傳 theta (rad), 範圍 (-pi, pi]
    """
    normal_map = {
        "+X": np.array([ 1.0,  0.0], dtype=float),
        "-X": np.array([-1.0,  0.0], dtype=float),
        "+Y": np.array([ 0.0,  1.0], dtype=float),
        "-Y": np.array([ 0.0, -1.0], dtype=float),
    }

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    n_hat = normal_map[normal]
    v = p2 - p1
    d = np.linalg.norm(v)

    u_hat = v / d

    dot = float(np.dot(n_hat, u_hat))
    dot = float(np.clip(dot, -1.0, 1.0))
    det = float(n_hat[0] * u_hat[1] - n_hat[1] * u_hat[0])

    theta = float(np.arctan2(det, dot))
    return theta


def steering_vector(num_elem, theta):
    """
    ULA steering vector with half-wavelength spacing:

        a(theta) = exp(+j*pi*n*sin(theta)), n=0,...,N-1

    輸入：
        num_elem : 陣列元素數
        theta    : 單一角度 (rad)

    輸出：
        shape = (num_elem, 1)
    """
    n = np.arange(num_elem).reshape(-1, 1)

    return np.exp(1j * np.pi * n * np.sin(theta)).astype(np.complex64)


def cn01(shape):
    """
    每個元素 ~ CN(0,1)
    Re, Im ~ N(0, 1/2)
    """
    std = np.sqrt(0.5)
    real = np.random.normal(loc=0.0, scale=std, size=shape)
    imag = np.random.normal(loc=0.0, scale=std, size=shape)
    return (real + 1j * imag).astype(np.complex64)


def clarke_sos_process(num_slots, shape, K):
    """
    Clarke Sum-of-Sinusoids Rayleigh fading

    輸入：
        num_slots : time slot 數量
        shape     : 天線數量 (TX_ANT 或 RIS_UNIT)
        K         : UE 數量

    輸出：
        process : (num_slots, shape, K)

    每根天線、每個 UE 各自生成一條獨立的
    Clarke SOS fading sequence。
    """
    L           = 64    # propagation-path 數量


    # 每個天線、每個 UE，各自產生 L 個 AoA 與初始相位
    alpha = np.random.uniform(-np.pi, np.pi, size=(L, shape, K)).astype(np.float32) # (每個散射路徑,每個天線,每個UE)
    phi   = np.random.uniform(-np.pi, np.pi, size=(L, shape, K)).astype(np.float32)


    # 計算每個 sinusoid 的 Doppler frequency
    doppler = (MAX_DOPPLER * np.cos(alpha)).astype(np.float32)                      # shape(L,shape,K)

    # 建立完整 trajectory 的時間軸
    time = np.arange(num_slots, dtype=np.float32) * SLOT_DURATION                   # 將slot索引換成真正時間

    # 建立通道矩陣
    process = np.zeros((num_slots, shape, K), dtype=np.complex64)                   # shape(num_slots,shape,K)


    # Clarke SOS：累加 L 個 complex sinusoids
    for l in range(L):
        """補充:
            原本time的shape是(num_slots) 使用time[:, None, None] 轉成(num_slots,1,1)
            doppler 的 shape 為 (L, shape, K)，取出 doppler[l] 後為 (shape, K)。
            time[:, None, None] 與 doppler[l][None, :, :] 相乘後為 (num_slots, shape, K)。

            他等效於for k for m  for l 再第t個slot所產生的通道

        """
        phase = 2.0 * np.pi * time[:, None, None] * doppler[l][None, :, :] + phi[l][None, :, :]

        process += np.exp(1j * phase).astype(np.complex64)

    # 正規化
    process /= np.sqrt(float(L))

    return process


def dist(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


def path_loss_power(d, exponent):
    """
    Distance-dependent path-loss power gain:
        beta(d) = C0 * (d0 / d)^exponent
    其中:
        C0 = -30 dB = 1e-3
        d0 = 1 m
    """
    d = np.asarray(d, dtype=float)
    d = np.maximum(d, 1e-12)

    beta = 1e-3 * ((1.0 / d) ** exponent)

    return beta.astype(np.float32)


# ================================
# Dataset 生成
# ================================
def build_and_save_dataset(layout_path, save_path, dataset_name):
    """
    根據輸入軌跡生成 channels,並儲存成 .npz
    """
    # 載入位置資料
    with np.load(layout_path) as data:
        positions = data["positions"].astype(np.float32)

    N_trajectory, N_slot, K, _ = positions.shape # 有幾條軌跡,1軌跡有多少slot,有幾個用戶,2維座標

    # --------------------------------
    # 建立完整 channel dataset
    # --------------------------------
    h_dk    = np.empty((N_trajectory, N_slot, TX_ANT, K), dtype=np.complex64)           # shape (軌跡數,slot數,M,K)
    h_rk    = np.empty((N_trajectory, N_slot, RIS_UNIT, K), dtype=np.complex64)         # shape (軌跡數,slot數,N,K)
    G       = np.empty((N_trajectory, N_slot, RIS_UNIT, TX_ANT), dtype=np.complex64)    # shape (軌跡數,slot數,N,M)
    g_dt    = np.empty((N_trajectory, N_slot, TX_ANT, 1), dtype=np.complex64)           # shape (軌跡數,slot數,M,1)

    # --------------------------------
    # 建立通道的大尺度路徑功率增益 dataset
    # --------------------------------
    beta_d_data = np.empty((N_trajectory, N_slot, K), dtype=np.float32)                 # BS  → UE，shape (軌跡數,slot數,K)
    beta_r_data = np.empty((N_trajectory, N_slot, K), dtype=np.float32)                 # RIS → UE，shape (軌跡數,slot數,K)
    beta_G_data = np.empty((), dtype=np.float32)                                        # BS  → RIS，純量
    beta_g_data = np.empty((), dtype=np.float32)                                        # BS  → Target，純量

    # --------------------------------
    # 生成h_dk
    # --------------------------------
    for n in range(N_trajectory):

        # 每個 slot 根據 UE 與 BS 的位置計算距離
        """補充:
            positions[n] 指的是"positions"4維(軌跡數,slot數,K,座標維度數) 僅看第一維 shape(slot數,K,座標維度數)
            表示第n條軌跡 每個slot的 每個用戶的 每個維度座標
            np.linalg.norm(np.array, axis=-1)->沿著np.array最後一維算norm 在這裡最後一維是"座標維度數"
        """
        d_dk = np.linalg.norm(positions[n] - np.asarray(Q_BS), axis=-1)     # (N_slot,K) 

        # 大尺度路徑損耗
        beta_dk = path_loss_power(d_dk, PL_EXP_BS_UE)                       # (N_slot,K)
        beta_d_data[n] = beta_dk                                            # 新增：存入第 n 條軌跡的 BS → UE 增益

        # 小尺度 Rayleigh fading：Clarke SOS
        z_dk = clarke_sos_process(N_slot, TX_ANT, K)                        # (N_slot,M,K)

        # 根據 PPT 公式產生完整 BS -> UE channel
        h_dk[n] = np.sqrt(beta_dk)[:, None, :] * z_dk                       # (N_slot,M,K)


    # --------------------------------
    # 生成h_rk
    # --------------------------------
    for n in range(N_trajectory):

        # 每個 slot 根據 UE 與 RIS 的位置計算距離
        d_rk = np.linalg.norm(positions[n] - np.asarray(Q_RIS), axis=-1)    # (N_slot,K)

        # 大尺度路徑損耗
        beta_rk = path_loss_power(d_rk, PL_EXP_RIS_UE)                      # (N_slot,K)
        beta_r_data[n] = beta_rk                                            # 新增：存入第 n 條軌跡的 RIS → UE 增益

        # 小尺度 Rician fading 的 NLoS 部分：Clarke SOS
        z_rk = clarke_sos_process(N_slot, RIS_UNIT, K)                      # (N_slot,N,K)

        # 建立 LoS steering vector
        h_rk_los = np.empty((N_slot, RIS_UNIT, K), dtype=np.complex64)      # (N_slot,N,K)

        for t in range(N_slot):
            for k in range(K):

                # 計算 RIS 指向 UE 的角度 (每個t slot k UE 與 RIS夾角不同)
                theta_rk = theta_calculater(Q_RIS, positions[n, t, k], "-X")

                # 根據角度生成 steering vector
                h_rk_los[t, :, k:k+1] = steering_vector(RIS_UNIT, theta_rk)

        # Rician LoS 與 NLoS 加權
        h_rk_rician = np.sqrt(RICIAN_KAPPA / (RICIAN_KAPPA + 1.0)) * h_rk_los + np.sqrt(1.0 / (RICIAN_KAPPA + 1.0)) * z_rk

        # 根據 PPT 公式產生完整 RIS -> UE channel
        h_rk[n] = np.sqrt(beta_rk)[:, None, :] * h_rk_rician  # (N_slot,N,K)


    # --------------------------------
    # 生成G
    # --------------------------------
    # BS 與 RIS 位置固定，因此距離與 Path Loss 固定
    d_G = dist(Q_BS, Q_RIS)
    beta_G = path_loss_power(d_G, PL_EXP_BS_RIS)
    beta_G_data[()] = beta_G                         # 新增：存入 BS → RIS 增益

    # 計算 BS -> RIS 的 AoD 與 RIS <- BS 的 AoA
    theta_AoD = theta_calculater(Q_BS, Q_RIS, "+X")
    theta_AoA = theta_calculater(Q_RIS, Q_BS, "-X")

    # 建立 LoS steering vectors
    a_M = steering_vector(TX_ANT, theta_AoD)      # (M,1)
    a_N = steering_vector(RIS_UNIT, theta_AoA)    # (N,1)

    # BS → RIS 的 LoS 分量：a_N(theta_AoA) @ a_M(theta_AoD)^H
    G_LoS = a_N @ a_M.conj().T                    # (N, M)

    for n in range(N_trajectory):

        # 每條 trajectory 獨立生成一次 NLoS，5000 slots 內固定
        G_NLoS = cn01((RIS_UNIT, TX_ANT))         # (N,M)

        # Rician LoS 與 NLoS 加權
        G_rician = np.sqrt(beta_G) * (np.sqrt(RICIAN_KAPPA / (RICIAN_KAPPA + 1.0)) * G_LoS + np.sqrt(1.0 / (RICIAN_KAPPA + 1.0)) * G_NLoS)

        # 同一條 trajectory 的所有 slots 共用相同的 G
        G[n] = G_rician                          # (N_slot,N,M)


    # --------------------------------
    # 生成 g_dt：BS -> Target
    # --------------------------------
    # BS 與 Target 位置固定，因此距離與 Path Loss 固定
    d_dt    = dist(Q_BS, Q_TAR)
    beta_dt = path_loss_power(d_dt, PL_EXP_BS_TAR)
    beta_g_data[()] = beta_dt                        # 新增：存入 BS → Target 增益

    # 計算 BS -> Target 的角度
    theta_dt = theta_calculater(Q_BS, Q_TAR, "+X")

    # 建立 LoS steering vector
    a_M = steering_vector(TX_ANT, theta_dt)       # (M,1)

    # 根據 PPT 公式生成 BS -> Target channel
    g_dt_static = np.sqrt(beta_dt) * a_M          # (M,1)

    # 所有 trajectories 與 slots 共用相同的 g_dt
    g_dt[:] = g_dt_static                         # (N_trajectory,N_slot,M,1)

    # --------------------------------
    # 儲存 dataset
    # --------------------------------
    np.savez(
        save_path,
        h_dk=h_dk,
        h_rk=h_rk,
        G=G,
        g_dt=g_dt,
        beta_d=beta_d_data,
        beta_r=beta_r_data,
        beta_G=beta_G_data,
        beta_g=beta_g_data,
    )

    print(f"[{dataset_name}] dataset 已儲存到：{save_path}")


# ================================
# generate train / val / test datasets
# ================================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    build_and_save_dataset(
        layout_path     = TRAIN_LAYOUT_PATH,
        save_path       = TRAIN_DATA_PATH,
        dataset_name    = "train"
    )

    build_and_save_dataset(
        layout_path     = VAL_LAYOUT_PATH,
        save_path       = VAL_DATA_PATH,
        dataset_name    = "val"
    )

    build_and_save_dataset(
        layout_path     = TEST_LAYOUT_PATH,
        save_path       = TEST_DATA_PATH,
        dataset_name    = "test"
    )

    print("\n[INFO] All channels generated successfully.")