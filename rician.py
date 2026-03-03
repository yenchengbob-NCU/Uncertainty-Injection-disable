import numpy as np
import math
import os
from settings import *  

# ================================
#    產生 4條 通道
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

    # 計算STEERING VECTOR 所需要的角度
    def theta_calculater(p1, p2, normal):
        normal_map = {
            "+X": np.array([ 1.0,  0.0], dtype=float),
            "-X": np.array([-1.0,  0.0], dtype=float),
            "+Y": np.array([ 0.0,  1.0], dtype=float),
            "-Y": np.array([ 0.0, -1.0], dtype=float),
        }
        n_hat = normal_map[normal]
        v = np.array([p2[0] - p1[0] , p2[1] - p1[1]], dtype=float)
        d = np.linalg.norm(v)
        u_hat = v / d
        # signed angle: atan2(det(n,u), dot(n,u))
        dot = float(np.dot(n_hat, u_hat))
        dot = float(np.clip(dot, -1.0, 1.0))
        det = float(n_hat[0] * u_hat[1] - n_hat[1] * u_hat[0])

        theta = float(np.arctan2(det, dot))      # (-pi, pi]
        theta_deg = float(np.degrees(theta))

        #print(f"[DEBUG] STEERING VECTOR 角度：從 p1={p1} 到 p2={p2} 是 {theta:.4f} rad ({theta_deg:.2f} deg)")
        return theta

    # 計算STEERING VECTOR 
    def steering_vector(I, theta):
        n = np.arange(I, dtype=float).reshape(-1, 1)        # 生成vector (I,1)
        t = np.asarray(theta, dtype=float)                  # 判斷theta 是數值還是陣列
        if t.ndim == 0:          # scalar
            t = t.reshape(1, 1)
        else:                    # (K,) -> (1,K)
            t = t.reshape(1, -1)

        s = np.sin(t)
        s = np.clip(s, -1.0, 1.0)                           # numeric safety
        return np.exp(-1j * np.pi * n * s)                  # 回傳(I,K)
    
    # 計算 NLOS 的高斯雜訊
    def cn01(shape):
        # 每個元素 ~ CN(0,1) → Re,Im ~ N(0, 1/2)
        std = np.sqrt(0.5)
        H_re = np.random.normal(loc=0.0, scale=std, size=shape)
        H_im = np.random.normal(loc=0.0, scale=std, size=shape)
        return H_re + 1j * H_im
    
    # BS normal: +X ; RIS normal: -X
    the_RIS_to_UE = np.array([theta_calculater(Q_RIS, ue,   "-X") for ue in ue_list], dtype=float)  # (K,)
    the_BS_to_RIS = theta_calculater(Q_BS,  Q_RIS,          "+X")                                   # scalar
    the_RIS_fromB = theta_calculater(Q_RIS, Q_BS,           "-X")                                   # scalar 
    the_BS_to_TAR = theta_calculater(Q_BS,  Q_UAV_TAR,      "+X")                                   # scalar

    # LoS steering (vectorized over K where applicable)
    aN_RIS_UE  = steering_vector(N, the_RIS_to_UE)        # (N,K)
    aM_BS_RIS  = steering_vector(M, the_BS_to_RIS)        # (M,1)
    aN_RIS_frB = steering_vector(N, the_RIS_fromB)        # (N,1)
    G_LoS = aN_RIS_frB @ aM_BS_RIS.conj().T               # (N,M)
    aM_BS_TAR  = steering_vector(M, the_BS_to_TAR)        # (M,1)

    ''' 
    #輸出 steering vector
    print("h_rk LoS aN_RIS_UE[:,0]   =", aN_RIS_UE[:, 0])
    print("G LoS RIS-side aN_RIS_frB =", aN_RIS_frB[:, 0])
    print("G LoS BS-side  aM_BS_RIS  =", aM_BS_RIS[:, 0])
    print("G LoS                     =", G_LoS)
    print("g_dt LoS aM_BS_TAR        =", aM_BS_TAR[:, 0])
    '''

    # Allocate outputs
    h_dk_all = np.zeros((n_networks, M, K), dtype=np.complex64)     # (n_networks,M,K)
    h_rk_all = np.zeros((n_networks, N, K), dtype=np.complex64)     # (n_networks,N,K)
    G_all    = np.zeros((n_networks, N, M), dtype=np.complex64)     # (n_networks,N,M)
    g_dt_all = np.zeros((n_networks, M, 1), dtype=np.complex64)     # (n_networks,M,1)

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
        hdk = hdk_N                                       # (M,K) BS⭢USER
        hrk = (rho_LoS * aN_RIS_UE + rho_NLoS * hrk_N)    # (N,K) RIS⭢USER   
        Gmm = (rho_LoS * G_LoS     + rho_NLoS * G_N)      # (N,M) BS⭢RIS
        gdt = aM_BS_TAR                                   # (M,1) BS⭢TAR

        # store
        h_dk_all[n, :, :] = hdk.astype(np.complex64, copy=False)
        h_rk_all[n, :, :] = hrk.astype(np.complex64, copy=False)
        G_all[n, :, :]    = Gmm.astype(np.complex64, copy=False)
        g_dt_all[n, :, :] = gdt.astype(np.complex64, copy=False)
    
    return h_dk_all, h_rk_all, G_all, g_dt_all

def large_scale_fading():
    '''
    Large-scale fading (POWER attenuation factors)
    注意：這裡回傳的是「功率」衰減/增益比例(線性尺度)，不是振幅
    '''
    ue_list = list(Q_UAV_UE_LIST)        #簡寫

    def dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)

    def one_way_fading(d):
        d = np.asarray(d, dtype=float)
        d = np.maximum(d, 1e-12)   # 避免 d=0 造成除零
        return (LAMBDA ** 2) / (((4.0 * np.pi) ** 2) * (d ** 2))

    def two_way_fading(d1, d2):
        d1 = np.asarray(d1, dtype=float)
        d2 = np.asarray(d2, dtype=float)
        d1 = np.maximum(d1, 1e-12) # 避免 d=0 造成除零
        d2 = np.maximum(d2, 1e-12) # 避免 d=0 造成除零
        return (LAMBDA ** 2) / (((4.0 * np.pi) ** 3) * (d1 ** 2) * (d2 ** 2))
    
    # 計算距離
    d_BS_UE  = np.array([dist(Q_BS,  ue) for ue in ue_list], dtype=float)  # (K,) BS  -> UE_k
    d_RIS_UE = np.array([dist(Q_RIS, ue) for ue in ue_list], dtype=float)  # (K,) RIS -> UE_k
    d_BS_RIS = dist(Q_BS,  Q_RIS)                                          # scalar BS -> RIS
    d_BS_TAR = dist(Q_BS,  Q_UAV_TAR)                                      # scalar BS -> Target

    #功率衰減
    # BS->UE      用 one-way
    pl_BS_UE = one_way_fading(d_BS_UE)                                     # (K,)
    # BS->RIS->UE 用 two-way
    pl_BS_RIS_UE = two_way_fading(d_BS_RIS, d_RIS_UE)                      # (K,)
    # BS->Tar->BS 用 two-way
    pl_BS_TAR_BS = two_way_fading(d_BS_TAR, d_BS_TAR)                      # scalar

    '''
    #輸出距離與PathLoss
    print(
        f"[DEBUG] Distances (m): "
        f"BS->UE = {d_BS_UE}, "
        f"BS->RIS = {d_BS_RIS:.3f}, "
        f"RIS->UE = {d_RIS_UE}, "
        f"BS->TAR = {d_BS_TAR:.3f}"
    )
    print(
        f"[DEBUG] Large-Scale Fading (power, linear): "
        f"PL_BS_UE = {pl_BS_UE}, "
        f"PL_BS_RIS_UE = {pl_BS_RIS_UE}, "
        f"PL_BS_TAR_BS = {pl_BS_TAR_BS:.3e}"
    )
    '''
    
    return pl_BS_UE, pl_BS_RIS_UE, pl_BS_TAR_BS

def _estimate_single_channel(H: np.ndarray) -> np.ndarray:
    """
    LMMSE pilot-aided estimate for y = sqrt(Pp) * H + n,  n ~ CN(0, N0 I)
    回傳 H_hat(與 H 同形狀)
    """
    sigma = np.sqrt(NOISE_POWER / 2.0)                                              # CN(0, N0 I) 噪聲
    n = (np.random.normal(0.0, sigma, size=H.shape)+ 1j * np.random.normal(0.0, sigma, size=H.shape))
    
    pilots_received = np.sqrt(ESTIMATION_PILOT_POWER) * H + n                       # 接收的導頻
    coef = np.sqrt(ESTIMATION_PILOT_POWER) / (ESTIMATION_PILOT_POWER + NOISE_POWER) # LMMSE 係數
    H_est = coef * pilots_received                                                  # 估測
    return H_est

if __name__ == "__main__":
    print("Generating offline test channels...")
    # 創建4條通道
    h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(N_TEST)

    # 估測4條通道
    h_dk_est = _estimate_single_channel(h_dk_np)
    h_rk_est = _estimate_single_channel(h_rk_np)
    G_est    = _estimate_single_channel(G_np)
    g_dt_est = _estimate_single_channel(g_dt_np)

    # save files
    out_path = TEST_NPZ_PATH

    np.savez(
        out_path,
        h_dk=h_dk_est,
        h_rk=h_rk_est,
        G=G_est,
        g_dt=g_dt_est
    )
    print(f"[ISAC] Saved: {out_path}")
