import numpy as np
import math
import os
from settings import *  

# ================================
# 1) 產生 4條 通道
#    h_dk h_rk G g_dt
# ================================
def generate_real_channels(n_networks: int) -> np.ndarray:

    M, N, K = TX_ANT, RIS_UNIT, UAV_COMM #簡寫
    ue_list = list(Q_UAV_UE_LIST)        #簡寫
    
    def dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)
    
    def beta_calculater(dist,alpha):
        C0_dB = -30.0
        d0 = 1.0
        L_dB = C0_dB - 10.0 * alpha * math.log10(dist / d0)
        beta_power = 10.0 ** (L_dB / 10.0)
        return math.sqrt(beta_power)

    def theta_calculater(p1, p2, normal):
       # step 1 計算 BS或RIS的向量
        normal_map = {
            "+X": np.array([ 1.0,  0.0], dtype=float),
            "-X": np.array([-1.0,  0.0], dtype=float),
            "+Y": np.array([ 0.0,  1.0], dtype=float),
            "-Y": np.array([ 0.0, -1.0], dtype=float),
        }
        n_hat = normal_map[normal]
        a_hat = np.array([-n_hat[1], n_hat[0]], dtype=float) #DoA AoA 用
        # step 2 計算 目標向量單位向量
        v = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        d = np.linalg.norm(v)
        v_hat = v / (d)
        # step 3 計算 sin theta
        sin_theta = float(np.dot(v_hat, a_hat))
        return sin_theta

    def steering_vector(I, sin_theta):
        n = np.arange(I, dtype=float).reshape(-1, 1)          # (I,1)
        s = np.asarray(sin_theta, dtype=float)

        if s.ndim == 0:                                      # scalar -> (1,1)
            s = s.reshape(1, 1)
        else:                                                # (K,) -> (1,K)
            s = s.reshape(1, -1)

        s = np.clip(s, -1.0, 1.0)                            # numeric safety
        return np.exp(-1j * np.pi * n * s)                   # (I,K) via broadcasting

    def cn01(shape):
        # 每個元素 ~ CN(0,1) → Re,Im ~ N(0, 1/2)
        std = np.sqrt(0.5)
        H_re = np.random.normal(loc=0.0, scale=std, size=shape)
        H_im = np.random.normal(loc=0.0, scale=std, size=shape)
        return H_re + 1j * H_im
    
    #計算beta & alpha
    
    # 距離 (per-UE for dk/rk)
    d_BS_UE  = np.array([dist(Q_BS,  ue) for ue in ue_list], dtype=float)  # (K,) BS  到 k 通訊無人機
    d_RIS_UE = np.array([dist(Q_RIS, ue) for ue in ue_list], dtype=float)  # (K,) RIS 到 k 通訊無人機
    d_BS_RIS = dist(Q_BS,  Q_RIS)                                          # BS 到 RIS
    d_BS_TAR = dist(Q_BS,  Q_UAV_TAR)                                      # BS 到 TARGET

    # large-scale amplitude betas
    beta_dk = np.array([beta_calculater(d, 3.3) for d in d_BS_UE ], dtype=float)  # (K,)
    beta_rk = np.array([beta_calculater(d, 2.2) for d in d_RIS_UE], dtype=float)  # (K,)
    beta_G  = beta_calculater(d_BS_RIS, 2.3)
    beta_dt = beta_calculater(d_BS_TAR, 2.7)

    #print(f"beta_dk  = {beta_dk}")
    #print(f"beta_rk  = {beta_rk}")
    #print(f"beta_G   = {beta_G}")
    #print(f"beta_dt  = {beta_dt}")

    # angles (we return sin(theta))
    # BS normal: +X ; RIS normal: -X 
    sin_RIS_to_UE = np.array([theta_calculater(Q_RIS, ue,      "-X") for ue in ue_list], dtype=float)  # (K,)
    sin_BS_to_RIS = theta_calculater(Q_BS,  Q_RIS,     "+X")  # scalar
    sin_RIS_fromB = theta_calculater(Q_RIS, Q_BS,      "-X")  # scalar (AoA at RIS from BS)
    sin_BS_to_TAR = theta_calculater(Q_BS,  Q_UAV_TAR, "+X")  # scalar
    
    # LoS steering (vectorized over K where applicable)
    aN_RIS_UE  = steering_vector(N, sin_RIS_to_UE)        # (N,K) # RIS 到 UE  Rician 的 Los 部分
    aM_BS_TAR  = steering_vector(M, sin_BS_to_TAR)        # (M,1) # BS  到 target 只有Los
    aM_BS_RIS  = steering_vector(M, sin_BS_to_RIS)        # (M,1) # BS  到 RIS Rician 的 Los 部分
    aN_RIS_frB = steering_vector(N, sin_RIS_fromB)        # (N,1) # RIS 到 BS  Rician 的 Los 部分
    G_LoS = aN_RIS_frB @ aM_BS_RIS.conj().T               # (N,M) # 結合成 G

    # Allocate outputs
    h_dk_all = np.zeros((n_networks, M, K), dtype=np.complex64)
    h_rk_all = np.zeros((n_networks, N, K), dtype=np.complex64)
    G_all    = np.zeros((n_networks, N, M), dtype=np.complex64)
    g_dt_all = np.zeros((n_networks, M, 1), dtype=np.complex64)

    # Rician mixing weights
    kappa = 2.0  # same as your previous baseline
    rho_LoS  = math.sqrt(kappa / (kappa + 1.0))
    rho_NLoS = math.sqrt(1.0 / (kappa + 1.0))

    # reshape betas for broadcasting
    beta_dk_row = beta_dk.reshape(1, K)   # (1,K)
    beta_rk_row = beta_rk.reshape(1, K)   # (1,K)

    # Sample n_networks times (only NLoS changes per sample)
    for n in range(n_networks):
        # NLoS components
        hdk_N = cn01((M, K)) # BS 到 UE 只有NLOS
        hrk_N = cn01((N, K)) # RIS 到 UE  Rician 的 NLOS
        G_N   = cn01((N, M)) # BS  到 RIS Rician 的 NLOS

        # Rician combine + pathloss amplitude beta
        hdk = hdk_N * beta_dk_row                                      # (M,K)
        hrk = (rho_LoS * aN_RIS_UE + rho_NLoS * hrk_N) * beta_rk_row   # (N,K)
        Gmm = (rho_LoS * G_LoS     + rho_NLoS * G_N)   * beta_G        # (N,M)
        gdt = aM_BS_TAR * math.sqrt(beta_dt)                           # (M,1)

        # store
        h_dk_all[n, :, :] = hdk.astype(np.complex64, copy=False)
        h_rk_all[n, :, :] = hrk.astype(np.complex64, copy=False)
        G_all[n, :, :]    = Gmm.astype(np.complex64, copy=False)
        g_dt_all[n, :, :] = gdt.astype(np.complex64, copy=False)

    return h_dk_all, h_rk_all, G_all, g_dt_all


# ================================
# 估測通道計算
# ================================

def _estimate_single_channel(H: np.ndarray) -> np.ndarray:
    """
    LMMSE pilot-aided estimate for y = sqrt(Pp) * H* + n,  n ~ CN(0, N0 I)
    回傳 H_hat(與 H 同形狀)
    """
    sigma = np.sqrt(NOISE_POWER / 2.0)                                              # CN(0, N0 I) 噪聲
    n = (np.random.normal(0.0, sigma, size=H.shape)+ 1j * np.random.normal(0.0, sigma, size=H.shape))
    
    pilots_received = np.sqrt(ESTIMATION_PILOT_POWER) * H + n                       # 接收的導頻
    coef = np.sqrt(ESTIMATION_PILOT_POWER) / (ESTIMATION_PILOT_POWER + NOISE_POWER) # LMMSE 係數
    H_est = coef * pilots_received                                                  # 估測
    return H_est

# ------------------------------
# 直接執行本檔時
# ------------------------------
if __name__ == "__main__":
    print("Generating offline test channels...")

    # 4-channel version: h_dk, h_rk, G, g_dt (no h_rt)
    h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(N_TEST)

    h_dk_est = _estimate_single_channel(h_dk_np)
    h_rk_est = _estimate_single_channel(h_rk_np)
    G_est    = _estimate_single_channel(G_np)
    g_dt_est = _estimate_single_channel(g_dt_np)

    # save files
    out_dir = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "channelEstimates_test.npz")

    np.savez(
        out_path,
        h_dk=h_dk_est,
        h_rk=h_rk_est,
        G=G_est,
        g_dt=g_dt_est
    )
    print(f"[ISAC] Saved: {out_path}")

    # （可選）讓輸出更像 MATLAB：固定小數位、不要科學記號
    # import numpy as np
    # np.set_printoptions(precision=4, suppress=True, linewidth=200)

    """ 
    # ====== h_dk: (n_networks, M, K) = (n, 6, 2) ======
    print("h_dk_np shape =", h_dk_np.shape)          # 例如 (4000, 6, 2)
    print(h_dk_np[0])                                # 第 0 組 sample 的 h_dk，shape = (6, 2)；6 根天線 × 2 個 user

    # ====== h_rk: (n_networks, N, K) = (n, 40, 2) ======
    print("\nh_rk_np shape =", h_rk_np.shape)        # 例如 (4000, 40, 2)
    print(h_rk_np[0, :8, :2])                        # 第 0 組 sample 的 h_rk，取前 8 個 RIS 單元、2 個 user；shape = (8, 2)

    # ====== G: (n_networks, N, M) = (n, 40, 6) ======
    print("\nG_np shape =", G_np.shape)              # 例如 (4000, 40, 6)
    print(G_np[0, :8, :6])                           # 第 0 組 sample 的 G，取前 8 個 RIS 單元、6 根天線；shape = (8, 6)

    # ====== g_dt: (n_networks, M, 1) = (n, 6, 1) ======
    print("\ng_dt_np shape =", g_dt_np.shape)        # 例如 (4000, 6, 1)
    print(g_dt_np[0, :, 0])                          # 第 0 組 sample 的 g_dt，取第 0 欄並壓成一維；shape = (6,)
    # 或保留 column vector 形狀：
    print(g_dt_np[0, :, :])                          # 第 0 組 sample 的 g_dt，保留 (6,1) 
    
    """

