import os
import math
import numpy as np
from settings import *

# ============================================================
# Rician / Rayleigh channel model
# 給定 ue_layout，生成：
# 1. 幾何資訊
# 2. large-scale fading
# 3. true channels (給 long-term 用)
# 4. estimated channels (給 short-term 用)
# 5. train / val / test 固定 dataset
# ============================================================

RICIAN_KAPPA = 2.0   # 線性尺度

# ============================================================
# 基本幾何 / 小工具
# ============================================================
def normalize_ue_layout(ue_layout):
    """
    輸入：
        ue_layout: [(x1,y1), (x2,y2)] 或 np.ndarray shape=(K,2)
    回傳：
        np.ndarray, shape=(UAV_COMM, 2)
    """
    ue_layout = np.asarray(ue_layout, dtype=np.float32)
    if ue_layout.shape != (UAV_COMM, 2):
        raise ValueError(
            f"ue_layout shape 應為 ({UAV_COMM}, 2)，目前收到 {ue_layout.shape}"
        )
    return ue_layout


def theta_calculater(p1, p2, normal):
    """
    計算從 p1 指向 p2，相對於陣列法線的 signed angle
    回傳 theta (rad), 範圍 (-pi, pi]
    """
    normal_map = {
        "+X": np.array([ 1.0,  0.0], dtype=float),
        "-X": np.array([-1.0,  0.0], dtype=float),
        "+Y": np.array([ 0.0,  1.0], dtype=float),
        "-Y": np.array([ 0.0, -1.0], dtype=float),
    }

    if normal not in normal_map:
        raise ValueError(f"unknown normal = {normal}")

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    n_hat = normal_map[normal]
    v = p2 - p1
    d = np.linalg.norm(v)
    if d <= 1e-12:
        raise ValueError("p1 and p2 are too close; angle is undefined.")

    u_hat = v / d

    dot = float(np.dot(n_hat, u_hat))
    dot = float(np.clip(dot, -1.0, 1.0))
    det = float(n_hat[0] * u_hat[1] - n_hat[1] * u_hat[0])

    theta = float(np.arctan2(det, dot))
    return theta


def steering_vector(num_elem, theta):
    """
    ULA steering vector with half-wavelength spacing:
        a(theta) = exp(-j*pi*n*sin(theta)), n=0,...,N-1

    輸入：
        num_elem : 陣列元素數
        theta    : scalar 或 shape=(K,)

    輸出：
        shape = (num_elem, 1) 或 (num_elem, K)
    """
    n = np.arange(num_elem, dtype=float).reshape(-1, 1)
    t = np.asarray(theta, dtype=float)

    if t.ndim == 0:
        t = t.reshape(1, 1)
    else:
        t = t.reshape(1, -1)

    s = np.sin(t)
    s = np.clip(s, -1.0, 1.0)
    return np.exp(-1j * np.pi * n * s).astype(np.complex64)


def cn01(shape):
    """
    每個元素 ~ CN(0,1)
    Re, Im ~ N(0, 1/2)
    """
    std = np.sqrt(0.5)
    real = np.random.normal(loc=0.0, scale=std, size=shape)
    imag = np.random.normal(loc=0.0, scale=std, size=shape)
    return (real + 1j * imag).astype(np.complex64)


def dist(p1, p2):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return float(np.linalg.norm(p1 - p2))


def one_way_fading(d):
    d = np.asarray(d, dtype=float)
    d = np.maximum(d, 1e-12)
    return ((LAMBDA ** 2) / (((4.0 * np.pi) ** 2) * (d ** 2))).astype(np.float32)


def two_way_fading(d1, d2):
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    d1 = np.maximum(d1, 1e-12)
    d2 = np.maximum(d2, 1e-12)
    return ((LAMBDA ** 2) / (((4.0 * np.pi) ** 3) * (d1 ** 2) * (d2 ** 2))).astype(np.float32)

# ============================================================
# 幾何相關量：由指定 layout 計算
# ============================================================

def geometry_from_layout(ue_layout):
    """
    由指定的 ue_layout 計算：
    - 幾何角度
    - LoS steering vectors
    - 距離
    """
    ue_layout = normalize_ue_layout(ue_layout)

    M, N, K = TX_ANT, RIS_UNIT, UAV_COMM

    theta_RIS_to_UE = np.array(
        [theta_calculater(Q_RIS, ue, "-X") for ue in ue_layout],
        dtype=np.float32
    )  # (K,)

    theta_BS_to_RIS = theta_calculater(Q_BS, Q_RIS, "+X")
    theta_RIS_fromB = theta_calculater(Q_RIS, Q_BS, "-X")
    theta_BS_to_TAR = theta_calculater(Q_BS, Q_UAV_TAR, "+X")

    aN_RIS_UE  = steering_vector(N, theta_RIS_to_UE)    # (N,K)
    aM_BS_RIS  = steering_vector(M, theta_BS_to_RIS)    # (M,1)
    aN_RIS_frB = steering_vector(N, theta_RIS_fromB)    # (N,1)
    aM_BS_TAR  = steering_vector(M, theta_BS_to_TAR)    # (M,1)

    G_LoS = (aN_RIS_frB @ aM_BS_RIS.conj().T).astype(np.complex64)   # (N,M)

    d_BS_UE  = np.array([dist(Q_BS, ue) for ue in ue_layout], dtype=np.float32)  # (K,)
    d_RIS_UE = np.array([dist(Q_RIS, ue) for ue in ue_layout], dtype=np.float32)  # (K,)
    d_BS_RIS = np.float32(dist(Q_BS, Q_RIS))
    d_BS_TAR = np.float32(dist(Q_BS, Q_UAV_TAR))

    return {
        "ue_layout": ue_layout,
        "aN_RIS_UE": aN_RIS_UE,
        "aM_BS_RIS": aM_BS_RIS,
        "aN_RIS_frB": aN_RIS_frB,
        "aM_BS_TAR": aM_BS_TAR,
        "G_LoS": G_LoS,
        "d_BS_UE": d_BS_UE,
        "d_RIS_UE": d_RIS_UE,
        "d_BS_RIS": d_BS_RIS,
        "d_BS_TAR": d_BS_TAR,
    }

# ============================================================
# Large-scale fading
# ============================================================

def large_scale_fading(ue_layout):
    """
    回傳：
        pl_BS_UE     : (K,)
        pl_BS_RIS_UE : (K,)
        pl_BS_TAR_BS : scalar
    """
    geo = geometry_from_layout(ue_layout)

    pl_BS_UE = one_way_fading(geo["d_BS_UE"])                           # (K,)
    pl_BS_RIS_UE = two_way_fading(geo["d_BS_RIS"], geo["d_RIS_UE"])    # (K,)
    pl_BS_TAR_BS = np.float32(two_way_fading(geo["d_BS_TAR"], geo["d_BS_TAR"]))

    return pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS


# ============================================================
# 真實通道生成（條件於給定 layout）
# raw channels 不含 large-scale fading
# ============================================================
def generate_real_channels(n_networks, ue_layout):
    geo = geometry_from_layout(ue_layout)

    M, N, K = TX_ANT, RIS_UNIT, UAV_COMM

    h_dk_all = np.zeros((n_networks, M, K), dtype=np.complex64)
    h_rk_all = np.zeros((n_networks, N, K), dtype=np.complex64)
    G_all    = np.zeros((n_networks, N, M), dtype=np.complex64)
    g_dt_all = np.zeros((n_networks, M, 1), dtype=np.complex64)

    rho_los  = math.sqrt(RICIAN_KAPPA / (RICIAN_KAPPA + 1.0))
    rho_nlos = math.sqrt(1.0 / (RICIAN_KAPPA + 1.0))

    aN_RIS_UE = geo["aN_RIS_UE"]   # (N,K)
    G_LoS     = geo["G_LoS"]       # (N,M)
    aM_BS_TAR = geo["aM_BS_TAR"]   # (M,1)

    for n in range(n_networks):
        hdk_nlos = cn01((M, K))
        hrk_nlos = cn01((N, K))
        G_nlos   = cn01((N, M))

        # channel model
        hdk = hdk_nlos                                      # BS -> UE : Rayleigh
        hrk = rho_los * aN_RIS_UE + rho_nlos * hrk_nlos     # RIS -> UE : Rician
        Gmm = rho_los * G_LoS     + rho_nlos * G_nlos       # BS -> RIS : Rician
        gdt = aM_BS_TAR                                     # BS -> Target : deterministic LoS

        h_dk_all[n] = hdk.astype(np.complex64, copy=False)
        h_rk_all[n] = hrk.astype(np.complex64, copy=False)
        G_all[n]    = Gmm.astype(np.complex64, copy=False)
        g_dt_all[n] = gdt.astype(np.complex64, copy=False)

    return h_dk_all, h_rk_all, G_all, g_dt_all

# ============================================================
# Pilot-based channel estimation
# ============================================================
def estimate_single_channel(H):
    """
    LMMSE pilot-aided estimate for:
        y = sqrt(Pp) * H + n,  n ~ CN(0, N0 I)
    """
    sigma = np.sqrt(NOISE_POWER / 2.0)

    n = (
        np.random.normal(0.0, sigma, size=H.shape)
        + 1j * np.random.normal(0.0, sigma, size=H.shape)
    ).astype(np.complex64)

    pilots_received = np.sqrt(ESTIMATION_PILOT_POWER) * H + n
    coef = np.sqrt(ESTIMATION_PILOT_POWER) / (ESTIMATION_PILOT_POWER + NOISE_POWER)
    H_hat = coef * pilots_received
    return H_hat.astype(np.complex64)


def estimate_channels(h_dk, h_rk, G, g_dt):
    h_dk_hat = estimate_single_channel(h_dk)
    h_rk_hat = estimate_single_channel(h_rk)
    G_hat    = estimate_single_channel(G)
    g_dt_hat = estimate_single_channel(g_dt)
    return h_dk_hat, h_rk_hat, G_hat, g_dt_hat


def generate_estimated_channels(n_networks, ue_layout):
    h_dk, h_rk, G, g_dt = generate_real_channels(n_networks, ue_layout)
    return estimate_channels(h_dk, h_rk, G, g_dt)


# ============================================================
# 固定 dataset 生成
# ============================================================
def build_split_dataset(split_layouts, n_lt_true, n_st_est, split_name="train"):
    """
    對某個 split 的所有 layouts 生成固定 dataset

    回傳 dict 內含：
    1. ue_layouts / pathloss
    2. long-term 用 true channels
    3. short-term 用 estimated channels
    """
    n_layouts = len(split_layouts)
    M, N, K = TX_ANT, RIS_UNIT, UAV_COMM

    # ---------- layout-level metadata ----------
    ue_layouts_arr = np.zeros((n_layouts, K, 2), dtype=np.float32)
    pl_BS_UE_all = np.zeros((n_layouts, K), dtype=np.float32)
    pl_BS_RIS_UE_all = np.zeros((n_layouts, K), dtype=np.float32)
    pl_BS_TAR_BS_all = np.zeros((n_layouts,), dtype=np.float32)

    # ---------- long-term true channels ----------
    lt_h_dk_true_all = np.zeros((n_layouts, n_lt_true, M, K), dtype=np.complex64)
    lt_h_rk_true_all = np.zeros((n_layouts, n_lt_true, N, K), dtype=np.complex64)
    lt_G_true_all    = np.zeros((n_layouts, n_lt_true, N, M), dtype=np.complex64)
    lt_g_dt_true_all = np.zeros((n_layouts, n_lt_true, M, 1), dtype=np.complex64)

    # ---------- short-term estimated channels ----------
    st_h_dk_hat_all = np.zeros((n_layouts, n_st_est, M, K), dtype=np.complex64)
    st_h_rk_hat_all = np.zeros((n_layouts, n_st_est, N, K), dtype=np.complex64)
    st_G_hat_all    = np.zeros((n_layouts, n_st_est, N, M), dtype=np.complex64)
    st_g_dt_hat_all = np.zeros((n_layouts, n_st_est, M, 1), dtype=np.complex64)

    print(f"[{split_name}] 開始生成 dataset，共 {n_layouts} 個 layouts ...")

    for idx, layout in enumerate(split_layouts):
        ue_layout = normalize_ue_layout(layout)

        # layout-level metadata
        pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS = large_scale_fading(ue_layout)

        ue_layouts_arr[idx]   = ue_layout
        pl_BS_UE_all[idx]     = pl_BS_UE
        pl_BS_RIS_UE_all[idx] = pl_BS_RIS_UE
        pl_BS_TAR_BS_all[idx] = pl_BS_TAR_BS

        # long-term true channels
        lt_h_dk_true, lt_h_rk_true, lt_G_true, lt_g_dt_true = generate_real_channels(n_lt_true, ue_layout)
        lt_h_dk_true_all[idx] = lt_h_dk_true
        lt_h_rk_true_all[idx] = lt_h_rk_true
        lt_G_true_all[idx]    = lt_G_true
        lt_g_dt_true_all[idx] = lt_g_dt_true

        # short-term estimated channels
        st_h_dk_true, st_h_rk_true, st_G_true, st_g_dt_true = generate_real_channels(n_st_est, ue_layout)
        st_h_dk_hat, st_h_rk_hat, st_G_hat, st_g_dt_hat = estimate_channels(
            st_h_dk_true, st_h_rk_true, st_G_true, st_g_dt_true
        )

        st_h_dk_hat_all[idx] = st_h_dk_hat
        st_h_rk_hat_all[idx] = st_h_rk_hat
        st_G_hat_all[idx]    = st_G_hat
        st_g_dt_hat_all[idx] = st_g_dt_hat

        if (idx + 1) % max(1, n_layouts // 10) == 0 or (idx + 1) == n_layouts:
            print(f"[{split_name}] progress: {idx + 1}/{n_layouts}")

    dataset = {
        # metadata
        "ue_layouts": ue_layouts_arr,
        "pl_BS_UE": pl_BS_UE_all,
        "pl_BS_RIS_UE": pl_BS_RIS_UE_all,
        "pl_BS_TAR_BS": pl_BS_TAR_BS_all,

        # long-term true channels
        "lt_h_dk_true": lt_h_dk_true_all,
        "lt_h_rk_true": lt_h_rk_true_all,
        "lt_G_true": lt_G_true_all,
        "lt_g_dt_true": lt_g_dt_true_all,

        # short-term estimated channels
        "st_h_dk_hat": st_h_dk_hat_all,
        "st_h_rk_hat": st_h_rk_hat_all,
        "st_G_hat": st_G_hat_all,
        "st_g_dt_hat": st_g_dt_hat_all,
    }
    return dataset


def save_split_dataset(dataset, save_path, split_name="train"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **dataset)
    print(f"[{split_name}] dataset 已儲存到：{save_path}")


def build_and_save_all_datasets():
    train_dataset = build_split_dataset(
        split_layouts=TRAIN_UE_LAYOUTS,
        n_lt_true=LONGTERM_TRUE_SAMPLES_PER_LAYOUT,
        n_st_est=SHORTTERM_EST_CHANNELS_PER_LAYOUT,
        split_name="train"
    )
    save_split_dataset(train_dataset, TRAIN_DATASET_PATH, "train")

    val_dataset = build_split_dataset(
        split_layouts=VAL_UE_LAYOUTS,
        n_lt_true=LONGTERM_TRUE_SAMPLES_PER_LAYOUT,
        n_st_est=SHORTTERM_EST_CHANNELS_PER_LAYOUT,
        split_name="val"
    )
    save_split_dataset(val_dataset, VAL_DATASET_PATH, "val")

    test_dataset = build_split_dataset(
        split_layouts=TEST_UE_LAYOUTS,
        n_lt_true=LONGTERM_TRUE_SAMPLES_PER_LAYOUT,
        n_st_est=SHORTTERM_EST_CHANNELS_PER_LAYOUT,
        split_name="test"
    )
    save_split_dataset(test_dataset, TEST_DATASET_PATH, "test")


if __name__ == "__main__":
    print("[INFO] 依 settings.py 的固定 split 生成 dataset ...")
    build_and_save_all_datasets()
    print("[INFO] 全部 dataset 生成完成。")