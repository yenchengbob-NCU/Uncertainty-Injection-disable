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
TX_ANT      = 6     #發射天線數(M)
RX_ANT      = 6     #接收天線數(M)   # 未來可能會刪除 用在接收濾波器的應用
RIS_UNIT    = 40    #反射元件數(N)
UAV_COMM    = 2     #通訊無人機(K)
UAV_TAR     = 1     #感測無人機(1)

FC = 3.5e9                          # 載波頻率(Hz)
C0 = 3e8                            # 光速(m/s)
LAMBDA = C0 / FC                    # 波長(m)                           # 會用於計算Path Loss
BANDWIDTH   = 10e6                  # 系統頻寬(Hz)                      # 暫時沒有用到
NOISE_POWER = 10e-11                # 環境雜訊總功率 (單位：W)
ESTIMATION_PILOT_POWER = 10e-10     # 導頻功率(單位：W)
TRANSMIT_POWER_TOTAL = 1            # 傳輸總功率(單位：W)
SENSING_SNR_THRESHOLD_dB = 15       # 感測 SNR 門檻(dB)
SENSING_SNR_THRESHOLD = 10 ** (SENSING_SNR_THRESHOLD_dB / 10.0)         # 門檻換算成線性

SENSING_LOSS_WEIGHT    = 500.0      # 罰則係數 λ1 (0~500)
RE_POWER_LOSS_WEIGHT   = 100.0      # 罰則係數 λ2 (0~500)
TX_POWER_LOSS_WEIGHT   = 100.0      # 罰則係數 λ3 (0~500)

# ================================
# 元件位置 (假設高度在一個平面)
# ================================
# 座標用公尺(m)表示 符合遠場條件 (在RIS正面的3m'圓周'上)
def random_points_on_circle(center, radius, num_points, normal=(-1.0, 0.0)):
    cx, cy = center
    nx, ny = normal

    norm_n = np.hypot(nx, ny)
    if norm_n == 0:
        raise ValueError("normal vector cannot be zero.")
    nx /= norm_n
    ny /= norm_n

    points = []
    while len(points) < num_points:
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)

        if (x - cx) * nx + (y - cy) * ny >= 0:
            points.append((round(float(x), 2), round(float(y), 2)))

    return points

Q_BS            = (0,0)
Q_RIS           = (50,0)
UE_RADIUS       = 3.0
Q_UAV_UE_LIST   = random_points_on_circle(Q_RIS, UE_RADIUS, UAV_COMM)
Q_UAV_TAR       = (10,2)

# ================================
# 不確定性注入
# ================================
INJECTION_VARIANCE  = 0.075     # 注入的雜訊功率大小
INJECTION_SAMPLES   = 1000      # L：每筆樣本注入的通道擾動數
OUTAGE_QUANTILE     = 0.05      # γ：用 5% 分位做健壯化目標（Robust_Net）

# ================================
# 訓練/驗證/測試
# ================================
EPOCHS          = 100        # 訓練輪數 一個epoch=MINIBATCHES*BATCH_SIZE 次
MINIBATCHES     = 50        # 每個epoch 有多少 mini-batch   
BATCH_SIZE      = 1000      # 每個 mini-batch 大小  
DATA_SIZE       = 200000    # 訓練與VAL使用總資料
N_VAL           = 2000      # TEST時通道樣本注入擾動數
N_TEST          = 4000      # TEST時通道樣本數
LEARNING_RATE   = 1e-3      # 學習率（實際依網路結構微調）

# ================================
# Early stopping
# ================================
EARLY_STOPPING_ENABLE    = False

ROB_EARLY_STOPPING_PATIENCE  = 10
ROB_EARLY_STOPPING_MIN_DELTA = 1e-6
ROB_EARLY_STOPPING_WARMUP    = 12


# ================================
# 其他輔助參數
# ================================

# 檔名格式

# 場景標籤：幾個 M / K / N_TARGET
SCENARIO_TAG = f"M{TX_ANT}_Ris{RIS_UNIT}_K{UAV_COMM}"

# 門檻標籤：感測門檻(db)_罰則係數 λ
THR_TAG = (
    f"THR_{SENSING_SNR_THRESHOLD_dB}db_"
    f"punish_{SENSING_LOSS_WEIGHT,RE_POWER_LOSS_WEIGHT,TX_POWER_LOSS_WEIGHT}"
)

SETTING_STRING = (
    f"N0_{NOISE_POWER}_"
    f"INJERR_{INJECTION_VARIANCE}"
    f"TX_power{TRANSMIT_POWER_TOTAL}W"
)

SAVE_DIR = "./checkpoints/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 建立階層資料夾
MLP_DIR                 = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
CKPT_DIR                = os.path.join(MLP_DIR, "ckpt")
CURVE_DIR               = os.path.join(MLP_DIR, "training_curves")
TEST_NPZ_PATH           = os.path.join(MLP_DIR, "channelEstimates_test.npz")
TRAIN_VAL_NPZ_PATH      = os.path.join(MLP_DIR, "channelEstimates_Train_Val.npz")

os.makedirs(MLP_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)


if __name__ == "__main__":
    print("\n[DEBUG] Current UE positions:")
    for k, ue in enumerate(Q_UAV_UE_LIST, start=1):
        print(f"UE {k}: ({ue[0]:.2f}, {ue[1]:.2f})")
    print(f"[INFO] 輸出資料夾已建立：{MLP_DIR}")