import os
import random
import numpy as np
import torch
""" 程式碼說明 :

    1.設定環境與參數
    2.建立階層資料夾
"""

Debug = True                        # 終端印出檢查

# ================================
# 固定執行環境以及RANDOM_SEED
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"[INFO] 使用 CUDA 裝置：{torch.cuda.get_device_name(DEVICE)}")
else:
    print("[INFO] 一定要使用cuda")

RANDOM_SEED = 123
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# ================================
# 專案元件座標
# ================================
Q_BS      = (0.0 , 0.0)     # 基站座標
Q_RIS     = (50.0, 0.0)     # RIS座標
Q_UAV_TAR = (5.0, 8.6)      # 感測物座標

# ================================
# UE 軌跡生成
# ================================
MOBILITY_ALPHA  = 1.0        # alpha=1: 每條 trajectory 內速度與方向固定
TOTAL_SLOTS     = 5000       # 每條 trajectory 共 5000 time slots
LT_BLOCK_SIZE   = 10         # RIS 每 10 slots 更新一次
N_LT_BLOCKS     = 500
SLOT_DURATION   = 0.001      # 時間尺度：v=1 m/s、slot=1 ms -> 每個 LT block 移動 1 cm；5000 slots 約 5 m

# Trajectory 初始位置：RIS=(50,0)，UE 隨機生成在 RIS 前方、半徑 3 m 的半圓圓周上
# 角度採 global coordinate：0 deg=+x，因此前方 (-x side) 對應 90~270 deg
UE_INIT_RADIUS  = 3.0                           # m
UE_INIT_ARC_RANGE_DEG = (135.0, 225.0)          # 保證 x <= 50；x 範圍為 [47, 50]

# Mobility：alpha=1 時，每條 trajectory 的速度、方向皆固定
UE_SPEED = 1.0                                  # 平均移動速度 m/s
UE_SPEED_STD = 0.0                              # Gauss-Markov 速度 innovation 標準差，alpha=1 時不作用
UE_DIRECTION_RANGE_DEG = (120.0, 240.0)         # 每條 trajectory 的平均移動方向範圍
UE_DIRECTION_STD_DEG = 0.0                      # Gauss-Markov 方向 innovation 標準差，alpha=1 時不作用

# ================================
# channel aging
# ================================
CARRIER_FREQUENCY = 8e9             # 中心頻率
SPEED_OF_LIGHT    = 3e8             # 光速
MAX_DOPPLER       = CARRIER_FREQUENCY * UE_SPEED / SPEED_OF_LIGHT

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
# 感測門檻
# ================================
SENSING_SNR_THRESHOLD_DB = 5
SENSING_SNR_THRESHOLD    = 10 ** (SENSING_SNR_THRESHOLD_DB / 10.0)

# ================================
# 訓練損失權重
# ================================
LT_SENSING_LOSS_WEIGHT   = 5        # 感測懲罰權重
REG_SENSING_LOSS_WEIGHT  = 50        
ROB_SENSING_LOSS_WEIGHT  = 0.5        

REG_TX_POWER_LOSS_WEIGHT = 100.0    # txpower 溢出懲罰權重
ROB_TX_POWER_LOSS_WEIGHT = 100.0  

# ================================
# dataset 生成，軌跡數量
# ================================
N_TRAIN_TRAJECTORIES = 50           # 共有50組UE軌跡
N_VAL_TRAJECTORIES   = 10
N_TEST_TRAJECTORIES  = 10

# ================================
# Robust / uncertainty injection
# ================================

INJECTION_VARIANCE  = 0.075         # Error power = 7.5% of each estimated channel block's empirical mean power
INJECTION_SAMPLES   = 200           # 一個估測通道要有多少誤差通道
OUTAGE_QUANTILE     = 0.05          # Robust tail quantile：每個 estimated channel 對 S 筆 injection 取 Q0.05
TAR_OUTAGE_QUANTILE = 0.10          # 感測可以超出的門檻

# ================================
# 訓練 / 驗證 / 測試
# ================================
LT_EPOCHS       = 500
REG_EPOCHS      = 2000
ROB_EPOCHS      = 2000

N_BATCHE        = 50
LT_BATCH_SIZE   = 20
ST_BATCH_SIZE   = 50

LT_LEARNING_RATE  = 0.0001
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
PRETRAIN_DIR    = os.path.join(BASE_RUN_DIR, "pretrain")            

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
        print(f"[INFO] slots/trajectory={TOTAL_SLOTS}, LT block size={LT_BLOCK_SIZE}, LT blocks/trajectory={N_LT_BLOCKS}")
        print(f"[INFO] UE initial semicircle: center={Q_RIS}, radius={UE_INIT_RADIUS} m, arc={UE_INIT_ARC_RANGE_DEG} deg")
        print(f"[INFO] UE speed={UE_SPEED:.2f} m/s, direction={UE_DIRECTION_RANGE_DEG} deg")
    print("Finish")