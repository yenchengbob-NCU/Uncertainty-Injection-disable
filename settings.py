import os
import random
import numpy as np
import torch

# ================================
# 基本環境
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 運算裝置（優先使用 GPU）
if DEVICE.type == "cuda":
    print(f"[INFO] 使用 CUDA 裝置：{torch.cuda.get_device_name(DEVICE)}")
else:
    print("[INFO] 使用 CPU 進行計算")

RANDOM_SEED = 123                                                      # 固定隨機性（建議在程式最一開始呼叫）
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ================================
# 場景與天線
# ================================
M = 4          # BS 天線數（Uniform Linear Array, ULA）
K = 1          # 單用戶數（本場景固定單用戶）
N_TARGET = 1   # 感測目標數（單目標）
ARRAY_TYPE = "ULA"     # 陣列型式：ULA（均勻線性陣列）
FC = 3.5e9             # 載波頻率(Hz)；例：3.5 GHz（Sub-6）
C0 = 3e8               # 光速(m/s)
LAMBDA = C0 / FC       # 波長(m)
D_SPACING = 0.5        # 天線間距（以波長為單位）；0.5 = 半波長間距


# ================================
# 頻寬與雜訊（沿用原專案慣例）
# ================================

BANDWIDTH = 10e6       # 系統頻寬(Hz)
N0_dBm_per_Hz = -75    # 雜訊功率密度(dBm/Hz)
NOISE_POWER = (10 ** ((N0_dBm_per_Hz - 30) / 10.0)) * BANDWIDTH*10 # 雜訊總功率(瓦) #3.16 × 10⁻⁴ W


# ================================
# 估測通道 = Rayleigh + 估測誤差
# ================================
ESTIMATION_PILOT_LENGTH   = 1      # 通道估測 pilot 長度（符元數，簡化用 1）
ESTIMATION_ERROR_VARIANCE = 0.075  # 估測誤差方差 σ_e^2 （複高斯）

# 參考用：若需要由 MMSE 關係推估 pilot 功率，可用下式（可視需求調整）
ESTIMATION_PILOT_POWER = (NOISE_POWER * (1 - ESTIMATION_ERROR_VARIANCE)/ (ESTIMATION_PILOT_LENGTH * ESTIMATION_ERROR_VARIANCE))  
# Pilot 發送功率（瓦；僅供參考，實際實作可覆寫）


# ================================
# 功率分配型態--等功率分配
# ================================
POWER_ALLOCATION_MODE = "equal"
TRANSMIT_POWER_TOTAL = 1.0            # 發射總功率（歸一化 1.0；若接實機可改成瓦或 dBm）
"""
為何需要 NumPy / Torch 兩個版本？
 1) 資料流分工不同：
    - 離線資料、npy 存取 → 多半使用 NumPy。
    - 訓練/推論(GPU、autograd) → 必須使用 Torch Tensor。
 2) 效能與裝置一致性：
    - 避免在不同模組間反覆 np↔torch 轉型與 CPU↔GPU 搬移，否則會拖慢速度。
"""
def apply_power_allocation_numpy(B_dir: np.ndarray) -> np.ndarray:

    K_local = B_dir.shape[-1]
    scale = np.sqrt(TRANSMIT_POWER_TOTAL / K_local)
    return B_dir * scale

def apply_power_allocation_torch(B_dir: torch.Tensor) -> torch.Tensor:

    K_local = B_dir.shape[-1]
    scale = (TRANSMIT_POWER_TOTAL / K_local) ** 0.5
    return B_dir * B_dir.new_tensor(scale)

# ================================
# 感測目標（角度與路徑係數）
# ================================
THETA_DEG = 45.0            # 目標方位角(度)（相對陣列法線；正負定義依你的實作）
ALPHA_MEAN_POWER = 1.0      # 目標等效反射係數平均功率 |α|^2（歸一化，含 RCS×路徑損耗）
THETA_JITTER_STD_DEG = 0.0  # 角度不確定性標準差(度)；>0 表示也對角度做不確定性注入


# ================================
# 感測/通訊併行模式與限制
# ================================
#通訊的發射訊號會影響到雷達的感測
SELF_INTERFERENCE_KAPPA_dB = -30.0  # 同時併行殘留自干擾係數 κ(dB)（越負代表抑制越好）--[30]--
SELF_INTERFERENCE_KAPPA = 10 ** (SELF_INTERFERENCE_KAPPA_dB / 10.0)  # 線性比例 κ (0.001)

SENSING_SNR_THRESHOLD_dB = 10  # 感測 SNR 門檻(dB)（不得低於此值）--[10]--
SENSING_SNR_THRESHOLD = 10 ** (SENSING_SNR_THRESHOLD_dB / 10.0)      # 線性比例 10
SENSING_LOSS_WEIGHT = 50.0       # 罰則係數 λ（越大越重視感測門檻）(0~500)

# ================================
# 不確定性注入（核心設定）
# ================================
INJECTION_VARIANCE = 0.075 # 新設定{與通道估計誤差脫勾}
INJECTION_SAMPLES = 1000   # L：每筆樣本注入的通道擾動數（例如 1000）
OUTAGE_QUANTILE = 0.05     # γ：用 5% 分位做健壯化目標（Robust_Net）
INJECT_THETA = (THETA_JITTER_STD_DEG > 0.0)  # 是否對角度做擾動注入（依上方標準差）


# ================================
# 訓練/驗證/測試（數量與優化器）
# ================================
EPOCHS      = 300       # 訓練輪數 一個epoch=MINIBATCHES*BATCH_SIZE 次
MINIBATCHES = 50        # 每個epoch 有多少 mini-batch   (與原專案一致)
BATCH_SIZE  = 1000      # 每個 mini-batch 大小          (與原專案一致)
N_VAL       = 2000      # 驗證用通道樣本數
N_TEST      = 4000      # 測試用通道樣本數
LEARNING_RATE = 1e-3    # 學習率（實際依網路結構微調）


# ================================
# 其他輔助參數
# ================================

# 檔名格式
SETTING_STRING = (
    f"ISAC_N0_{NOISE_POWER:.5f}_INJERR_{INJECTION_VARIANCE:.5f}_"
    f"THR{int(SENSING_SNR_THRESHOLD_dB)}db"
)

SAVE_DIR = "./checkpoints/"
os.makedirs(SAVE_DIR, exist_ok=True)
DEBUG_VISUALIZE = False


"""
專案:單BS easy ISAC comm center Virson 3
檔名:setting.py

說明：
- 單一基地台(BS)M=4、單用戶(K=1)、單一感測目標。
- 單載波3.5 GHz、頻寬10Mhz
- 目標是最大化用戶傳輸速率，同時滿足「感測 SNR 不得低於門檻」。
- 本檔只放「參數設定」，供訓練與模擬引用。
"""