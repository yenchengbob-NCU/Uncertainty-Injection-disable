import numpy as np
import math
import os
from settings import *  

# ================================
# 1) 產生 4條 通道
#    h_dk h_rk G g_dt
# ================================
def generate_real_channels(n_networks: int) -> np.ndarray:
    '''
    generate_real_channels 的說明
    皆不含大尺度衰減!
    :輸入 n_networks(int) 代表要產生多少組通道
        




    :return: h_dk_all, h_rk_all, G_all, g_dt_all (ndarray)
    回傳4組通道
    '''
    M, N, K = TX_ANT, RIS_UNIT, UAV_COMM #簡寫
    ue_list = list(Q_UAV_UE_LIST)        #簡寫

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
        
    # BS normal: +X ; RIS normal: -X 
    sin_RIS_to_UE = np.array([theta_calculater(Q_RIS, ue,   "-X") for ue in ue_list], dtype=float)  # (K,)
    sin_BS_to_RIS = theta_calculater(Q_BS,  Q_RIS,          "+X")  # scalar
    sin_RIS_fromB = theta_calculater(Q_RIS, Q_BS,           "-X")  # scalar (AoA at RIS from BS)
    sin_BS_to_TAR = theta_calculater(Q_BS,  Q_UAV_TAR,      "+X")  # scalar

    # LoS steering (vectorized over K where applicable)
    aN_RIS_UE  = steering_vector(N, sin_RIS_to_UE)        # (N,K)
    aM_BS_RIS  = steering_vector(M, sin_BS_to_RIS)        # (M,1)
    aN_RIS_frB = steering_vector(N, sin_RIS_fromB)        # (N,1)
    G_LoS = aN_RIS_frB @ aM_BS_RIS.conj().T               # (N,M)
    aM_BS_TAR  = steering_vector(M, sin_BS_to_TAR)        # (M,1)

    #print("\n==== sin(theta) debug ====")
    #print("sin_RIS_to_UE =", sin_RIS_to_UE)   # (K,)
    #print("sin_BS_to_RIS =", sin_BS_to_RIS)   # scalar
    #print("sin_RIS_fromB =", sin_RIS_fromB)   # scalar
    #print("sin_BS_to_TAR =", sin_BS_to_TAR)   # scalar

    #print("h_rk LoS aN_RIS_UE[:,0]   =", aN_RIS_UE[:, 0])
    #print("G LoS RIS-side aN_RIS_frB =", aN_RIS_frB[:, 0])
    #print("G LoS BS-side  aM_BS_RIS  =", aM_BS_RIS[:, 0])
    #print("G LoS                     =", G_LoS)
    #print("g_dt LoS aM_BS_TAR        =", aM_BS_TAR[:, 0])

    # Allocate outputs
    h_dk_all = np.zeros((n_networks, M, K), dtype=np.complex64)
    h_rk_all = np.zeros((n_networks, N, K), dtype=np.complex64)
    G_all    = np.zeros((n_networks, N, M), dtype=np.complex64)
    g_dt_all = np.zeros((n_networks, M, 1), dtype=np.complex64)

    # Rician mixing weights
    kappa = 2.0
    rho_LoS  = math.sqrt(kappa / (kappa + 1.0))
    rho_NLoS = math.sqrt(1.0 / (kappa + 1.0))

    # Sample n_networks times (only NLoS changes per sample)
    for n in range(n_networks):
        # NLoS components
        hdk_N = cn01((M, K))
        hrk_N = cn01((N, K))
        G_N   = cn01((N, M))

        # Rician combine 
        hdk = hdk_N                                       # (M,K)
        hrk = (rho_LoS * aN_RIS_UE + rho_NLoS * hrk_N)    # (N,K)
        Gmm = (rho_LoS * G_LoS     + rho_NLoS * G_N)      # (N,M)
        gdt = aM_BS_TAR                                   # (M,1)

        # store
        h_dk_all[n, :, :] = hdk.astype(np.complex64, copy=False)
        h_rk_all[n, :, :] = hrk.astype(np.complex64, copy=False)
        G_all[n, :, :]    = Gmm.astype(np.complex64, copy=False)
        g_dt_all[n, :, :] = gdt.astype(np.complex64, copy=False)

    #print("h_dk_all shape =", h_dk_all.shape)
    #print("h_rk_all shape =", h_rk_all.shape)
    #print("G_all    shape =", G_all.shape)
    #print("g_dt_all shape =", g_dt_all.shape)

    return h_dk_all, h_rk_all, G_all, g_dt_all

def large_scale_fading():
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
        #這是功率的衰減!!!
        return beta_power
    
    # 距離 (per-UE for dk/rk)
    d_BS_UE  = np.array([dist(Q_BS,  ue) for ue in ue_list], dtype=float)  # (K,) BS  到 k 通訊無人機
    d_RIS_UE = np.array([dist(Q_RIS, ue) for ue in ue_list], dtype=float)  # (K,) RIS 到 k 通訊無人機
    d_BS_RIS = dist(Q_BS,  Q_RIS)                                          # BS 到 RIS
    d_BS_TAR = dist(Q_BS,  Q_UAV_TAR)                                      # BS 到 TARGET

    #print(f"d_BS_UE   = {d_BS_UE}")
    #print(f"d_RIS_UE  = {d_RIS_UE}")
    #print(f"d_BS_RIS  = {d_BS_RIS:.4f}")
    #print(f"d_BS_TAR  = {d_BS_TAR:.4f}")

    # large-scale amplitude 
    beta_dk = np.array([beta_calculater(d, 3.3) for d in d_BS_UE ], dtype=float)  # (K,)
    beta_rk = np.array([beta_calculater(d, 2.2) for d in d_RIS_UE], dtype=float)  # (K,)
    beta_G  = beta_calculater(d_BS_RIS, 2.3)
    beta_dt = beta_calculater(d_BS_TAR, 2.7)

    #print(f"beta_dk  = {beta_dk}")
    #print(f"beta_rk  = {beta_rk}")
    #print(f"beta_G   = {beta_G}")
    #print(f"beta_dt  = {beta_dt}")

    # reshape betas for broadcasting
    beta_dk_row = beta_dk.reshape(1, K)   # (1,K)
    beta_rk_row = beta_rk.reshape(1, K)   # (1,K)

    #print("beta_G  =", beta_G)
    #print("beta_dt =", beta_dt)

    #print("beta_dk: K=0 =", beta_dk_row[0, 0], " K=1 =", beta_dk_row[0, 1])
    #print("beta_rk: K=0 =", beta_rk_row[0, 0], " K=1 =", beta_rk_row[0, 1])

    return beta_G ,beta_dt ,beta_dk_row, beta_rk_row

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
    # 4-channel version: h_dk, h_rk, G, g_dt
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
