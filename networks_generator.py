import numpy as np
from settings import *  # 需要 M, K, ESTIMATION_PILOT_POWER, NOISE_POWER

# ------------------------------
# 1) 產生：Rayleigh i.i.d.
#    回傳 shape = (N, M, K) 複數陣列
# ------------------------------
def generate_real_channels(n_networks: int) -> np.ndarray:
    """
    產生 i.i.d. Rayleigh 通道（複高斯 CN(0,1)）
    每個元素 ~ CN(0,1) => Re, Im ~ N(0, 1/2)
    """
    assert n_networks >= 1, "n_networks (產生的通道數量)必須 >= 1" 
    std = np.sqrt(0.5)
    H_re = np.random.normal(0.0, std, size=(n_networks, M, K))
    H_im = np.random.normal(0.0, std, size=(n_networks, M, K))
    return H_re + 1j * H_im  # CN(0,1)

# ------------------------------
# 2) 由 i.i.d.Rayleigh 得到估測通道（MMSE 估測）
#    y = sqrt(Pp) * conj(H) + n,  n ~ CN(0, NOISE_POWER)
#    H_conj_est = [ sqrt(Pp) / (Pp + NOISE_POWER) ] * y
#    H_est = conj(H_conj_est)
#    輸入/輸出 shape: (N, M, K)
# ------------------------------
def estimate_channels(H: np.ndarray) -> np.ndarray:
    """
    以舊版(MMSE)流程產生估測通道
    """
    assert H.ndim == 3 and H.shape[1] == M and H.shape[2] == K, "H 維度需為 (N,M,K)"
    N = H.shape[0] 
    # n_BS ~ CN(0, NOISE_POWER)
    noise_std = np.sqrt(NOISE_POWER / 2.0)
    n_re = np.random.normal(0.0, noise_std, size=(N, M, K))
    n_im = np.random.normal(0.0, noise_std, size=(N, M, K))
    n_BS = n_re + 1j * n_im

    # 導頻量測與 MMSE 估測
    y = np.sqrt(ESTIMATION_PILOT_POWER) * np.conjugate(H) + n_BS
    coef = np.sqrt(ESTIMATION_PILOT_POWER) / (ESTIMATION_PILOT_POWER + NOISE_POWER)
    H_conj_est = coef * y
    H_est = np.conjugate(H_conj_est)
    return H_est

# ------------------------------
# 內部小工具：安全正規化（避免除 0）
# ------------------------------
def _normalize(v: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    """
    將向量 v 在指定 axis 上做 L2 正規化，使 ||v||_2 = 1
    - 預期 v shape: (N, M, K) 或 (N, M, 1)
    - 預設 axis= 1 表示沿著天線維度 M 正規化
    """
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return v / norm

# ------------------------------
# 3) 由 H_est 產生基準波束器 B（MRT，支援 K>=1）
#    B_nk ∝ ĥ_nk ；每欄向量做 L2 正規化，回傳 shape=(N, M, K)
# ------------------------------
def get_beamformers(H_est: np.ndarray) -> np.ndarray:
    """
    MRT beamforming(多用戶可用;K=1 時即為單用戶 MRT)。
    參數：
      - H_est: shape=(N, M, K) 的估測通道（複數）
    回傳：
      - B: shape=(N, M, K)，每個使用者一欄，且每欄 ||b_k||2 = 1
    """
    assert H_est.ndim == 3 and H_est.shape[1] == M and H_est.shape[2] == K, "H_est 維度需為 (N,M,K)"
    B = _normalize(H_est, axis=1)  # 沿天線維度做 L2 正規化
    return B

# ------------------------------
# 簡易自測（直接執行本檔時）
# ------------------------------
if __name__ == "__main__":
    print(f'[ISAC {SETTING_STRING}] Generating offline test channels...')
    H_est = estimate_channels(generate_real_channels(N_TEST)) # (N, M, K)
    
    # save files
    np.save(f'Data/channelEstimates_test_{SETTING_STRING}.npy', H_est)
    print(f"[ISAC] Saved: Data/channelEstimates_test_{SETTING_STRING}.npy")

