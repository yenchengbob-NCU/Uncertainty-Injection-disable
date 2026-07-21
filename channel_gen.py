import os
import math
import numpy as np
from settings import *

RICIAN_KAPPA = 2.0   # 線性尺度
Debug = False         # 終端印出檢查
# ================================
# 基本幾何 / 小工具
# ================================
def to_db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-30))


def fmt_db(x):
    x = np.asarray(to_db(x)).reshape(-1)
    return "{" + " ".join([f"[{v:.2f}]" for v in x]) + "}"


def channel_power(x):
    return np.mean(np.abs(x) ** 2)


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


def print_dataset_debug(data):
    
    print("\n[DEBUG] dataset_train.npz 內容檢查")

    print("\n[DEBUG] 6 組 path loss, unit = dB")
    print(f"{'pl_BS_UE':<13s}: {fmt_db(data['pl_BS_UE']):<42s} # BS -> UE 單程 PL")
    print(f"{'pl_RIS_UE':<13s}: {fmt_db(data['pl_RIS_UE']):<42s} # RIS -> UE 單程 PL")
    print(f"{'pl_BS_RIS':<13s}: {fmt_db(data['pl_BS_RIS']):<42s} # BS -> RIS 單程 PL")
    print(f"{'pl_BS_RIS_UE':<13s}: {fmt_db(data['pl_BS_RIS_UE']):<42s} # BS -> RIS -> UE 串接 PL")
    print(f"{'pl_BS_TAR':<13s}: {fmt_db(data['pl_BS_TAR']):<42s} # BS -> TAR 單程 PL")
    print(f"{'pl_BS_TAR_BS':<13s}: {fmt_db(data['pl_BS_TAR_BS']):<42s} # BS -> TAR -> BS 雙程 PL")

    print("\n[DEBUG] 4 組帶 PL 的估測通道大小與平均功率")
    print(
        f"{'h_dk_hat':<8s}: shape={str(data['h_dk_hat'].shape):<16s}, "
        f"power={channel_power(data['h_dk_hat']):.4e}, "
        f"power_dB={float(to_db(channel_power(data['h_dk_hat']))):.2f} dB"
    )
    print(
        f"{'h_rk_hat':<8s}: shape={str(data['h_rk_hat'].shape):<16s}, "
        f"power={channel_power(data['h_rk_hat']):.4e}, "
        f"power_dB={float(to_db(channel_power(data['h_rk_hat']))):.2f} dB"
    )
    print(
        f"{'G_hat':<8s}: shape={str(data['G_hat'].shape):<16s}, "
        f"power={channel_power(data['G_hat']):.4e}, "
        f"power_dB={float(to_db(channel_power(data['G_hat']))):.2f} dB"
    )
    print(
        f"{'g_dt_hat':<8s}: shape={str(data['g_dt_hat'].shape):<16s}, "
        f"power={channel_power(data['g_dt_hat']):.4e}, "
        f"power_dB={float(to_db(channel_power(data['g_dt_hat']))):.2f} dB"
    )


# ================================
# 幾何相關量：由指定 layout 計算
# ================================

def geometry_from_layout(ue_layout):
    """
    根據一組 UE layout 計算通道生成需要的幾何量。
    """
    M, N = TX_ANT, RIS_UNIT

    theta_RIS_to_UE = np.array(
        [theta_calculater(Q_RIS, ue, "-X") for ue in ue_layout],
        dtype=np.float32
    )

    theta_BS_to_RIS = theta_calculater(Q_BS, Q_RIS, "+X")
    theta_RIS_fromB = theta_calculater(Q_RIS, Q_BS, "-X")
    theta_BS_to_TAR = theta_calculater(Q_BS, Q_UAV_TAR, "+X")
    
    # steering_vector
    aN_RIS_UE  = steering_vector(N, theta_RIS_to_UE)
    aM_BS_RIS  = steering_vector(M, theta_BS_to_RIS)
    aN_RIS_frB = steering_vector(N, theta_RIS_fromB)
    aM_BS_TAR  = steering_vector(M, theta_BS_to_TAR)

    G_LoS = (aN_RIS_frB @ aM_BS_RIS.conj().T).astype(np.complex64)

    d_BS_UE = np.array(
        [dist(Q_BS, ue) for ue in ue_layout],
        dtype=np.float32
    )

    d_RIS_UE = np.array(
        [dist(Q_RIS, ue) for ue in ue_layout],
        dtype=np.float32
    )

    d_BS_RIS = np.float32(dist(Q_BS, Q_RIS))
    d_BS_TAR = np.float32(dist(Q_BS, Q_UAV_TAR))

    return {
        "aN_RIS_UE": aN_RIS_UE,
        "aM_BS_TAR": aM_BS_TAR,
        "G_LoS": G_LoS,
        "d_BS_UE": d_BS_UE,
        "d_RIS_UE": d_RIS_UE,
        "d_BS_RIS": d_BS_RIS,
        "d_BS_TAR": d_BS_TAR,
    }

# ================================
# Large-scale fading
# ================================

def large_scale_fading(ue_layout):
    """
    輸入: 單組 layout 
    回傳：
        pl_BS_UE      : (K,)   BS -> UE  單程 PL
        pl_BS_RIS     : scalar BS -> RIS 單程 PL
        pl_RIS_UE     : (K,)   RIS -> UE 單程 PL
        pl_BS_RIS_UE  : (K,)   BS -> RIS -> UE cascaded PL
        pl_BS_TAR     : scalar BS -> TAR 單程 PL
        pl_BS_TAR_BS  : scalar BS -> TAR -> BS 雙程 PL
    """
    geo = geometry_from_layout(ue_layout)

    pl_BS_UE  = path_loss_power(geo["d_BS_UE"],PL_EXP_BS_UE)        # (K,)
    pl_RIS_UE = path_loss_power(geo["d_RIS_UE"],PL_EXP_RIS_UE)      # (K,)
    pl_BS_RIS = path_loss_power(geo["d_BS_RIS"],PL_EXP_BS_RIS)      # scalar
    pl_BS_TAR = path_loss_power(geo["d_BS_TAR"],PL_EXP_BS_TAR)      # scalar

    pl_BS_RIS_UE = np.float32(pl_BS_RIS * pl_RIS_UE)                # (K,)
    pl_BS_TAR_BS = np.float32(pl_BS_TAR ** 2)                       # scalar

    return pl_BS_UE, pl_RIS_UE, pl_BS_RIS, pl_BS_RIS_UE, pl_BS_TAR, pl_BS_TAR_BS

# ================================
# 真實通道生成 (現在代表估測通道)
# raw channels 不含 large-scale fading
# ================================
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


# ================================
# Dataset 生成
# ================================
def build_and_save_dataset(channels, save_path, dataset_name):
    """
    根據輸入 channel數量 生成固定 dataset,並儲存成 .npz
    """
    n_channels = int(channels)              # 需要生成的 channels 數量
    M, N, K = TX_ANT, RIS_UNIT, UAV_COMM    # 簡寫

    # 儲存pathloss
    pl_BS_UE, pl_RIS_UE, pl_BS_RIS, pl_BS_RIS_UE, pl_BS_TAR, pl_BS_TAR_BS = large_scale_fading(UE_LAYOUT) # 對一組layout生成 Path loss

    print(f"[{dataset_name}] 開始生成 dataset,共 {n_channels} 個 估測通道 ...")
    # 固定 layout，所以直接生成 n_channels 筆估測通道
    h_dk_hat_all, h_rk_hat_all, G_hat_all, g_dt_hat_all = generate_real_channels(n_channels,UE_LAYOUT)
    
    # 加上 path loss，成為 NN 輸入使用的帶 PL 估測通道
    h_dk_hat_all = h_dk_hat_all * np.sqrt(pl_BS_UE).reshape(1, 1, K)
    h_rk_hat_all = h_rk_hat_all * np.sqrt(pl_RIS_UE).reshape(1, 1, K)
    G_hat_all    = G_hat_all    * np.sqrt(pl_BS_RIS)
    g_dt_hat_all = g_dt_hat_all * np.sqrt(pl_BS_TAR)

    h_dk_hat_all = h_dk_hat_all.astype(np.complex64)
    h_rk_hat_all = h_rk_hat_all.astype(np.complex64)
    G_hat_all    = G_hat_all.astype(np.complex64)
    g_dt_hat_all = g_dt_hat_all.astype(np.complex64)

    dataset = {
        "pl_BS_UE": pl_BS_UE,               # BS  -> UE  單程PL
        "pl_RIS_UE": pl_RIS_UE,             # RIS -> UE  單程PL
        "pl_BS_RIS": pl_BS_RIS,             # BS  -> RIS 單程PL
        "pl_BS_RIS_UE": pl_BS_RIS_UE,       # BS  -> RIS -> UE 串接PL
        "pl_BS_TAR": pl_BS_TAR,             # BS  -> TAR 單程PL
        "pl_BS_TAR_BS": pl_BS_TAR_BS,       # BS  -> TAR -> BS 雙程PL

        "h_dk_hat": h_dk_hat_all,           # 帶有PL的估測通道
        "h_rk_hat": h_rk_hat_all,
        "G_hat": G_hat_all,
        "g_dt_hat": g_dt_hat_all,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **dataset)

    print(f"[{dataset_name}] dataset 已儲存到：{save_path}")

if __name__ == "__main__":
    print("[INFO] 依 settings.py 的 固定layouts 生成 datasets ...")

    build_and_save_dataset(
        channels=N_TRAIN_CHANNELS,
        save_path=os.path.join(DATA_DIR, "dataset_train.npz"),
        dataset_name="train"
    )

    build_and_save_dataset(
        channels=N_VAL_CHANNELS,
        save_path=os.path.join(DATA_DIR, "dataset_val.npz"),
        dataset_name="val"
    )

    build_and_save_dataset(
        channels=N_TEST_CHANNELS,
        save_path=os.path.join(DATA_DIR, "dataset_test.npz"),
        dataset_name="test"
    )


    if Debug:
        with np.load(os.path.join(DATA_DIR, "dataset_train.npz")) as data:
            print_dataset_debug(data)

    print("[INFO] 全部 datasets 生成完成。")