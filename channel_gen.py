import os
import numpy as np
from scipy.special import j0
from settings import *

RICIAN_KAPPA = 2.0    # 線性尺度

# ============================================================
# Temporal correlation / continuous Jakes fading
# ============================================================
# ISAC14 使用 Doppler/Jakes temporal correlation：
#
#     rho[l] = J0(2*pi*l*f_D*Ts)
#
#     E{h[t]h*[s]} = J0(2*pi*f_D*|t-s|*Ts)
#
# 因此 correlation 只與 time lag |t-s| 有關，不與 absolute slot index 有關。
# 例如：
#     Corr(h[0], h[1])   = Corr(h[9], h[10])  = Corr(h[10], h[11])
#     Corr(h[0], h[9])   = Corr(h[10], h[19])
#
# Zhang et al., "Performance analysis of reconfigurable intelligent surface
# assisted systems under channel aging," ICN, 2022：
#     h[t] = rho[t]h[0] + sqrt(1-rho[t]^2)e[t]
#
# Chopra et al., "Performance Analysis of FDD Massive MIMO Systems Under
# Channel Aging," IEEE TWC, 2018：
#     innovation process 不必 temporally white，其 temporal covariance
#     應使完整 channel covariance 符合 Jakes model。
#
# 本程式直接生成整條 continuous WSS Jakes small-scale fading sequence，
# 不在 LT block boundary reset，也不再使用 fixed-rho AR(1) recursion。
#
# 有限個 Doppler components 用來數值近似理想 Jakes ACF；
# 增加 JAKES_COMPONENTS 可使單一 realization 的 empirical ACF 更接近理論 J0。
JAKES_COMPONENTS = 64

Debug = True          # 終端印出檢查


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


def jakes_cn01_process(num_slots, shape):
    """
    產生一整條 continuous normalized complex Jakes fading process。

    目標 temporal covariance：

        E{z[t]z*[s]} = J0(2*pi*f_D*|t-s|*Ts)

    對應 channel-aging representation：

        z[t] = rho[t]z[0] + sqrt(1-rho[t]^2)e[t]

    其中 e[t] 的 marginal distribution 為 CN(0,1)，但不同 time slots
    的 innovation 不應被視為 temporally independent。

    本函式不顯式建立 e[t] 或 5000x5000 covariance matrix，
    而是直接以 finite Doppler spectral components 生成整條 WSS process。

    為使 finite-component empirical ACF 的 imaginary residual 較小，
    Doppler frequencies 使用正負成對配置。

    輸入：
        num_slots : trajectory slots 數
        shape     : 每個 time slot 的 channel shape

    輸出：
        process shape = (num_slots, *shape)
        每個 channel coefficient marginally ~ CN(0,1)
        整條 trajectory 連續，不因 LT_BLOCK_SIZE reset
    """
    if JAKES_COMPONENTS < 2 or JAKES_COMPONENTS % 2 != 0:
        raise ValueError(f"JAKES_COMPONENTS 必須為 >=2 的偶數，目前為 {JAKES_COMPONENTS}")

    num_series = int(np.prod(shape))
    half = JAKES_COMPONENTS // 2

    # 隨機 AoA 對應 Clarke/Jakes isotropic scattering Doppler component
    alpha = np.random.uniform(-np.pi, np.pi, size=half)
    doppler_half = MAX_DOPPLER * np.cos(alpha)
    doppler = np.concatenate([doppler_half, -doppler_half]).astype(np.float32)

    # 每條 small-scale channel coefficient 使用 independent complex-Gaussian weights
    weights = cn01((JAKES_COMPONENTS, num_series))

    # 整條 trajectory 共用同一組 Doppler basis，因此 physical time 軸不會在 LT boundary reset
    time = np.arange(num_slots, dtype=np.float32) * SLOT_DURATION
    oscillator = np.exp(1j * 2.0 * np.pi * time[:, None] * doppler[None, :]).astype(np.complex64)

    process = (oscillator @ weights) / np.sqrt(float(JAKES_COMPONENTS))
    return process.reshape((num_slots,) + tuple(shape)).astype(np.complex64)


def empirical_temporal_corr(process, lag):
    """
    計算 normalized small-scale fading process 的 empirical temporal correlation：

        R_hat[lag] = E{h[t+lag]h*[t]} / sqrt(E|h[t+lag]|^2 E|h[t]|^2)

    process 第一維必須為 time。
    """
    if lag < 0 or lag >= process.shape[0]:
        raise ValueError(f"lag 必須位於 [0,{process.shape[0] - 1}]，目前 lag={lag}")

    if lag == 0:
        return np.complex64(1.0 + 0.0j)

    x0 = process[:-lag].reshape(-1)
    x1 = process[lag:].reshape(-1)

    numerator = np.mean(x1 * np.conj(x0))
    denominator = np.sqrt(np.mean(np.abs(x1) ** 2) * np.mean(np.abs(x0) ** 2))
    return numerator / denominator


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
# Static geometry / channel parts
# ================================
_STATIC_COMPONENTS = None


def generate_static_components():
    """
    只建立真正不隨 UE trajectory 改變的部分：

    1. BS -> RIS 的 path loss 與 LoS array response
       - BS/RIS 位置固定，所以這兩項固定
       - ISAC14 不再對 G 使用 temporal aging
       - G 的 Rician NLoS realization 在每條 trajectory 開始時生成一次，
         並在該整條 trajectory 內保持 static

    2. BS -> Target 的 LoS channel g_dt
       - target 固定，因此目前維持固定
    """
    global _STATIC_COMPONENTS

    if _STATIC_COMPONENTS is not None:
        return _STATIC_COMPONENTS

    # ------------------------------------------------
    # BS -> RIS : fixed geometry component of Rician G
    # ------------------------------------------------
    d_br = dist(Q_BS, Q_RIS)
    beta_br = path_loss_power(d_br, PL_EXP_BS_RIS)

    theta_bs_ris = theta_calculater(Q_BS, Q_RIS, "+X")
    theta_ris_bs = theta_calculater(Q_RIS, Q_BS, "-X")

    a_bs = steering_vector(TX_ANT, theta_bs_ris)      # (M,1)
    a_ris = steering_vector(RIS_UNIT, theta_ris_bs)  # (N,1)

    G_los = (a_ris @ a_bs.conj().T).astype(np.complex64)  # (N,M)

    # ------------------------------------------------
    # BS -> Target : LoS channel
    # g_dt shape = (TX_ANT, 1)
    # ------------------------------------------------
    d_dt = dist(Q_BS, Q_UAV_TAR)
    beta_dt = path_loss_power(d_dt, PL_EXP_BS_TAR)

    theta_bs_tar = theta_calculater(Q_BS, Q_UAV_TAR, "+X")
    a_target = steering_vector(TX_ANT, theta_bs_tar)

    g_dt_static = (np.sqrt(beta_dt) * a_target).astype(np.complex64)

    _STATIC_COMPONENTS = (beta_br, G_los, g_dt_static)
    return _STATIC_COMPONENTS


# ================================
# Dynamic channels
# ================================
def generate_dynamic_channels(ue_positions, h_dk_small_scale, h_rk_nlos_small_scale):
    """
    ue_positions shape = (K,2)

    h_dk_small_scale:
        此 slot 的 normalized BS->UE continuous Jakes Rayleigh fading，
        shape=(TX_ANT,K)

    h_rk_nlos_small_scale:
        此 slot 的 normalized RIS->UE continuous Jakes Rician NLoS fading，
        shape=(RIS_UNIT,K)

    Returns
    -------
    h_dk : BS -> UE
        shape = (TX_ANT, K)

    h_rk : RIS -> UE
        shape = (RIS_UNIT, K)
    """
    K = ue_positions.shape[0]

    # ------------------------------------------------
    # BS -> UE : continuous Jakes Rayleigh
    # ------------------------------------------------
    d_bu = np.linalg.norm(ue_positions - np.asarray(Q_BS), axis=1)
    beta_bu = path_loss_power(d_bu, PL_EXP_BS_UE)

    h_dk = np.sqrt(beta_bu)[None, :] * h_dk_small_scale

    # ------------------------------------------------
    # RIS -> UE : Rician
    # LoS: 每個 slot 由目前 UE geometry 更新
    # NLoS: 沿整條 physical trajectory 使用 continuous Jakes fading
    # ------------------------------------------------
    d_ru = np.linalg.norm(ue_positions - np.asarray(Q_RIS), axis=1)
    beta_ru = path_loss_power(d_ru, PL_EXP_RIS_UE)

    theta_ru = np.array([theta_calculater(Q_RIS, ue_positions[k], "-X") for k in range(K)])
    h_rk_los = steering_vector(RIS_UNIT, theta_ru)   # (N,K)

    h_rk = np.sqrt(beta_ru)[None, :] * (
        np.sqrt(RICIAN_KAPPA / (RICIAN_KAPPA + 1.0)) * h_rk_los
        + np.sqrt(1.0 / (RICIAN_KAPPA + 1.0)) * h_rk_nlos_small_scale
    )

    return h_dk.astype(np.complex64), h_rk.astype(np.complex64)


def generate_bs_ris_channel(beta_br, G_los):
    """
    BS -> RIS Rician channel G。

    BS 與 RIS 都固定，因此 ISAC14 不再對 G 使用 rho=0.87 temporal aging。

    每條 trajectory 開始時：
        1. 生成一次 normalized Rician NLoS realization
        2. 與固定 LoS component 組成完整 G
        3. 該 G 在整條 trajectory 的所有 slots 維持不變

    G shape = (RIS_UNIT, TX_ANT)
    """
    G_nlos = cn01((RIS_UNIT, TX_ANT))

    G = np.sqrt(beta_br) * (
        np.sqrt(RICIAN_KAPPA / (RICIAN_KAPPA + 1.0)) * G_los
        + np.sqrt(1.0 / (RICIAN_KAPPA + 1.0)) * G_nlos
    )

    return G.astype(np.complex64)


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

    N_trajectory, N_slot, K, _ = positions.shape

    # 建立完整 channel dataset
    h_dk    = np.empty((N_trajectory, N_slot, TX_ANT, K), dtype=np.complex64)
    h_rk    = np.empty((N_trajectory, N_slot, RIS_UNIT, K), dtype=np.complex64)
    G       = np.empty((N_trajectory, N_slot, RIS_UNIT, TX_ANT), dtype=np.complex64)
    g_dt    = np.empty((N_trajectory, N_slot, TX_ANT, 1), dtype=np.complex64)

    # 固定 geometry / target component
    beta_br, G_los, g_dt_static = generate_static_components()

    # Debug 用：保留第一條 trajectory 的 normalized Rayleigh process
    debug_h_dk_process = None

    # 每條 trajectory 有自己的 independent fading realization。
    # physical fading 在 trajectory 內從 slot 0 到最後一個 slot 持續演化，
    # 不會因為 RIS 每 LT_BLOCK_SIZE slots 更新而 reset。
    for n in range(N_trajectory):
        h_dk_process = jakes_cn01_process(N_slot, (TX_ANT, K))
        h_rk_nlos_process = jakes_cn01_process(N_slot, (RIS_UNIT, K))
        G_static = generate_bs_ris_channel(beta_br=beta_br, G_los=G_los)

        if n == 0:
            debug_h_dk_process = h_dk_process.copy()

        for t in range(N_slot):
            ue_positions = positions[n, t]

            h_dk[n, t], h_rk[n, t] = generate_dynamic_channels(ue_positions=ue_positions, h_dk_small_scale=h_dk_process[t], h_rk_nlos_small_scale=h_rk_nlos_process[t])

            # BS 與 RIS 固定，因此同一條 trajectory 內 G 完全 static
            G[n, t] = G_static

            # Target 固定，因此目前 g_dt 維持 LoS static channel
            g_dt[n, t] = g_dt_static

    # 儲存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(save_path, positions=positions, h_dk=h_dk, h_rk=h_rk, G=G, g_dt=g_dt)

    print(f"[{dataset_name}] dataset 已儲存到：{save_path}")

    if Debug:
        max_check_lag = min(LT_BLOCK_SIZE - 1, N_slot - 1)
        theoretical_rho = np.array([j0(2.0 * np.pi * MAX_DOPPLER * SLOT_DURATION * lag) for lag in range(max_check_lag + 1)], dtype=float)
        empirical_rho = np.array([np.real(empirical_temporal_corr(debug_h_dk_process, lag)) for lag in range(max_check_lag + 1)], dtype=float)

        print(f"positions = {positions.shape}")
        print(f"h_dk      = {h_dk.shape}")
        print(f"h_rk      = {h_rk.shape}")
        print(f"G         = {G.shape}")
        print(f"g_dt      = {g_dt.shape}")

        print(f"carrier frequency = {CARRIER_FREQUENCY / 1e9:.2f} GHz")
        print(f"maximum Doppler   = {MAX_DOPPLER:.4f} Hz")
        print(f"slot duration     = {SLOT_DURATION * 1e3:.3f} ms")
        print(f"LT block duration = {LT_BLOCK_SIZE * SLOT_DURATION * 1e3:.3f} ms")
        print(f"Jakes components  = {JAKES_COMPONENTS}")

        print(f"theoretical Jakes rho[0:{max_check_lag + 1}] = {np.round(theoretical_rho, 4)}")
        print(f"empirical   Jakes rho[0:{max_check_lag + 1}] = {np.round(empirical_rho, 4)}")

        print(f"Corr(slot 0, slot 1) theory = {theoretical_rho[1]:.4f}, empirical = {empirical_rho[1]:.4f}")
        if max_check_lag >= 9:
            print(f"Corr(slot 0, slot 9) theory = {theoretical_rho[9]:.4f}, empirical = {empirical_rho[9]:.4f}")
            print(f"Corr(slot 9, slot 10) theory = {theoretical_rho[1]:.4f}  <-- LT boundary 不 reset")
            print(f"Corr(slot 10, slot 19) theory = {theoretical_rho[9]:.4f}")

        print(f"G static check max|G[t]-G[0]| = {np.max(np.abs(G[0] - G[0, 0][None, :, :])):.6e}")
        print(f"g_dt static check max|g_dt[t]-g_dt[0]| = {np.max(np.abs(g_dt[0] - g_dt[0, 0][None, :, :])):.6e}")


# ================================
# generate train / val / test datasets
# ================================
if __name__ == "__main__":
    build_and_save_dataset(
        layout_path     = os.path.join(DATA_DIR, "layout_train.npz"),
        save_path       = os.path.join(DATA_DIR, "dataset_train.npz"),
        dataset_name    = "train"
    )

    build_and_save_dataset(
        layout_path     = os.path.join(DATA_DIR, "layout_val.npz"),
        save_path       = os.path.join(DATA_DIR, "dataset_val.npz"),
        dataset_name    = "val"
    )

    build_and_save_dataset(
        layout_path     = os.path.join(DATA_DIR, "layout_test.npz"),
        save_path       = os.path.join(DATA_DIR, "dataset_test.npz"),
        dataset_name    = "test"
    )

    print("\n[INFO] All channels generated successfully.")