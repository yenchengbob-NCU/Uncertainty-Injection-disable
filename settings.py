import os
import random
import numpy as np
import torch
"""
負責工作:
    1.設定環境與參數
    2.建立階層資料夾
    3.生成UE layout 
"""
Debug = False                        # 終端印出檢查

# ================================
# 基本環境
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"[INFO] 使用 CUDA 裝置：{torch.cuda.get_device_name(DEVICE)}")
else:
    print("[INFO] 使用 CPU 進行計算")

RANDOM_SEED = 123
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ================================
# 幾何場景
# ================================
Q_BS      = (0.0 , 0.0)     # 基站座標
Q_RIS     = (50.0, 0.0)     # RIS座標
Q_UAV_TAR = (10.0, 2.0)     # 感測物座標
UE_RADIUS = 3.0             # 用戶距RIS半徑
UE_LAYOUT = [               # 角度以 RIS 為圓心、全域 x 軸正方向為 0 度
    [49.2235,  2.8978],     # 105度
    #[47.7019,  1.9284],     # 140度
    #[47.7019, -1.9284],     # 220度
    [49.2235, -2.8978],     # 255度
]

# ================================
# 場景參數
# ================================
TX_ANT   = 4                        # BS  發射天線數 M
RIS_UNIT = 64                       # RIS 反射單元數 N
UAV_COMM = 2                        # 通訊 UE 數 K
RADAR_STREAMS   = 1                 # sensing waveforms 數量 s_r (1不是chu設定,是為了和Demirhan等價)
PL_EXP_BS_RIS   = 2.3               # 單程PL係數 BS  -> RIS
PL_EXP_RIS_UE   = 2.2               # 單程PL係數 RIS -> UE
PL_EXP_BS_UE    = 3.3               # 單程PL係數 BS  -> UE
PL_EXP_BS_TAR   = 2.7               # 單程PL係數 BS  -> TAR

# ================================
# 物理與功率參數
# ================================
NOISE_POWER             = 1e-11     # 雜訊功率(W) 10^-11 -80dBm
TRANSMIT_POWER_TOTAL    = 1.26      # 傳輸功率(W)

# ================================
# ISAC / 損失權重預設值
# ================================
SENSING_SNR_THRESHOLD_DB = 4.0
SENSING_SNR_THRESHOLD = 10 ** (SENSING_SNR_THRESHOLD_DB / 10.0)


# ================================
# 訓練損失權重
# ================================
REG_SENSING_LOSS_WEIGHT  = 10.0     # reg 感測懲罰權重
ROB_SENSING_LOSS_WEIGHT  = 0.5      # rob 感測懲罰權重

# ================================
# dataset 生成
# ================================
N_TRAIN_CHANNELS = 50000         # "估測通道" (現在的估測通道即是rician rayleigh等等)
N_VAL_CHANNELS   = 2000
N_TEST_CHANNELS  = 2000

# ================================
# Robust / uncertainty injection
# ================================

INJECTION_VARIANCE  = 0.075          # Error power = 7.5% of each estimated channel block's empirical mean power
INJECTION_SAMPLES   = 200            # 一個估測通道要有多少誤差通道
OUTAGE_QUANTILE     = 0.05           # Robust tail quantile：每個 estimated channel 對 S 筆 injection 取 Q0.05
TAR_OUTAGE_QUANTILE = 0.10

# ================================
# 訓練 / 驗證 / 測試
# ================================

REG_EPOCHS      = 2000
ROB_EPOCHS      = 2000
N_BATCHE        = 50        # 一個epoch 內有多少個BATCH
BATCH_CHANNELS  = 1000      # 一個BATCH 內有多少個通道


REG_LEARNING_RATE = 0.0001
ROB_LEARNING_RATE = 0.0001

# ================================
# 資料夾結構
# ================================
SCENARIO_TAG = f"M{TX_ANT}_N{RIS_UNIT}_K{UAV_COMM}"
BASE_RUN_DIR = os.path.join("Two_timescale",SCENARIO_TAG)

REG_PENALTY_TAG = f"reg_{REG_SENSING_LOSS_WEIGHT:g}"
ROB_PENALTY_TAG = f"rob_{ROB_SENSING_LOSS_WEIGHT:g}"

DATA_DIR        = os.path.join(BASE_RUN_DIR, "shared_data")         # shared dataset
PRETRAIN_DIR    = os.path.join(BASE_RUN_DIR, "pretrain")            #

REG_CKPT_DIR    = os.path.join(BASE_RUN_DIR, "regular",REG_PENALTY_TAG)
ROB_CKPT_DIR    = os.path.join(BASE_RUN_DIR, "robust" ,ROB_PENALTY_TAG)

RESULT_DIR      = os.path.join(BASE_RUN_DIR, "results")

# ================================
# 建立共用資料夾
# ================================
os.makedirs(BASE_RUN_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PRETRAIN_DIR, exist_ok=True)
os.makedirs(REG_CKPT_DIR, exist_ok=True)
os.makedirs(ROB_CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ================================
# Debug
# ================================
if __name__ == "__main__":
    print(f"\n[INFO] RANDOM_SEED = {RANDOM_SEED}")
    if Debug:
        print(f"[INFO] 在 layout 固定下預計生成 TRAIN {N_TRAIN_CHANNELS} 筆估測通道")
        print(f"[INFO] 在 layout 固定下預計生成 VAL   {N_VAL_CHANNELS} 筆估測通道")
        print(f"[INFO] 在 layout 固定下預計生成 TEST  {N_TEST_CHANNELS} 筆估測通道")

    print(f"Finish")

