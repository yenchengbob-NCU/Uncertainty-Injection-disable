import os
import random
import numpy as np
import torch
"""
負責工作:
    1.設定基本參數
    2.建立階層資料夾
    3.生成UE layout 

程式說明:
    現在的階層資料夾包含
    MLP/M6_Ris40_K2/N0_1e-10_INJERR_0.075_TXpower_1.0_SenTHR_16dB/
"""
# ================================
# 小工具
# ================================
def random_points_on_circle(center, radius, num_points):
    """
    在指定圓周上隨機取點，只保留 normal 指向的正面半圓
    回傳 shape = (num_points, 2) dtype=np.float32
    """
    cx, cy = center
    nx, ny = (-1.0, 0.0)

    norm_n = np.hypot(nx, ny)
    nx /= norm_n
    ny /= norm_n

    points = []
    while len(points) < num_points:
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)

        if (x - cx) * nx + (y - cy) * ny >= 0.0:
            points.append((round(float(x), 2), round(float(y), 2)))

    return np.asarray(points, dtype=np.float32)


def layout_gen(num_layouts):
    layouts = [ 
        random_points_on_circle(
            center=Q_RIS,
            radius=UE_RADIUS,
            num_points=UAV_COMM,
        )
        for _ in range(num_layouts)
    ]
    return np.stack(layouts, axis=0).astype(np.float32)


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
# 場景參數
# ================================
TX_ANT   = 6                # BS 發射天線數 M
RIS_UNIT = 40               # RIS 反射單元數 N
UAV_COMM = 2                # 通訊 UE 數 K
RADAR_STREAMS = TX_ANT      # sensing waveforms 數量

# ================================
# 幾何場景
# ================================
Q_BS      = (0.0 , 0.0)     # 基站座標
Q_RIS     = (50.0, 0.0)     # RIS座標
Q_UAV_TAR = (10.0, 2.0)     # 感測物座標
UE_RADIUS = 3.0             # 用戶距RIS半徑

# ================================
# 物理與功率參數
# ================================
FC = 3.5e9
C0 = 3e8
LAMBDA = C0 / FC

NOISE_POWER             = 10e-11
ESTIMATION_PILOT_POWER  = 10e-10
TRANSMIT_POWER_TOTAL    = 1.0

# ================================
# ISAC / 損失權重預設值
# ================================
SENSING_SNR_THRESHOLD_dB = 16
SENSING_SNR_THRESHOLD = 10 ** (SENSING_SNR_THRESHOLD_dB / 10.0)

# ================================
# 訓練損失權重
# ================================
REG_SENSING_LOSS_WEIGHT = 150.0 # reg 感測懲罰權重
ROB_SENSING_LOSS_WEIGHT = 0.2   # rob 感測懲罰權重
RIS_POWER_LOSS_WEIGHT   = 250.0 # RIS元件 功率懲罰權重

# ================================
# dataset 生成
# ================================
"""
說明：
    1. train / validation / test 各自獨立生成 UE layouts
    2. 每個 layout 底下再由 rician.py 生成固定通道資料：
        - long-term  用的 statistical channels
        - short-term 用的 estimated channels
"""

N_TRAIN_LAYOUTS = 300                           # train layout 數量
N_VAL_LAYOUTS   = 100                           # validation layout 數量
N_TEST_LAYOUTS  = 50                            # test layout 數量

TRAIN_UE_LAYOUTS = layout_gen(N_TRAIN_LAYOUTS)
VAL_UE_LAYOUTS   = layout_gen(N_VAL_LAYOUTS)
TEST_UE_LAYOUTS  = layout_gen(N_TEST_LAYOUTS)

LONGTERM_TRUE_SAMPLES_PER_LAYOUT  = 128         # LT的"統計通道" (沒有估測，純rician&rayleigh) 
SHORTTERM_EST_CHANNELS_PER_LAYOUT = 1000        # ST的"估測通道" (使用LMMSE估測,有誤差)

# ================================
# Robust / uncertainty injection
# ================================

INJECTION_VARIANCE = 0.075                      # 注入誤差
INJECTION_SAMPLES  = 200                        # 一個估測通道要有多少誤差通道
OUTAGE_QUANTILE    = 0.05                       # SNR容許值

# ================================
# 訓練 / 驗證 / 測試
# ================================
LT_EPOCHS   = 300
REG_EPOCHS  = 1000
ROB_EPOCHS  = 400

MINIBATCHES = 50                                # 更新多少次權重 val 一次 
PLOT_MOVING_AVG_WINDOW = 30                     # plot用，畫平滑曲線

# Short-term mixed-layout minibatch
ST_BATCH_LAYOUTS = 30                           # ST 一個batch 抓多少 layout 出來
ST_BATCH_EST_CHANNELS_PER_LAYOUT = 100          # 一個layout 抓多少 估測通道 出來

LT_LEARNING_RATE  = 0.001
REG_LEARNING_RATE = 0.0005
ROB_LEARNING_RATE = 0.0005

# ================================
# 資料夾結構
# ================================

SCENARIO_TAG = f"M{TX_ANT}_Ris{RIS_UNIT}_K{UAV_COMM}"

SETTING_STRING = (
    f"N0_{NOISE_POWER}_"
    f"INJERR_{INJECTION_VARIANCE}_"
    f"SenTHR_{SENSING_SNR_THRESHOLD_dB}dB"
)

# MLP/M6_Ris40_K2/N0_1e-10_INJERR_0.075_SenTHR_16dB/
BASE_RUN_DIR = os.path.join("MLP",SCENARIO_TAG,SETTING_STRING)


DATA_DIR = os.path.join(BASE_RUN_DIR, "shared_data")    # shared dataset

LT_DIR = os.path.join(BASE_RUN_DIR, "LT")               # shared long-term model
LT_CKPT_DIR = os.path.join(LT_DIR, "ckpt")
LT_CURVE_DIR = os.path.join(LT_DIR, "training_curves")

ST_SWEEP_DIR = os.path.join(BASE_RUN_DIR, "ST_sweep")   # short-term penalty sweep

# ================================
# 建立共用資料夾
# ================================
os.makedirs(BASE_RUN_DIR, exist_ok=True)

os.makedirs(DATA_DIR, exist_ok=True)

os.makedirs(LT_DIR, exist_ok=True)
os.makedirs(LT_CKPT_DIR, exist_ok=True)
os.makedirs(LT_CURVE_DIR, exist_ok=True)

os.makedirs(ST_SWEEP_DIR, exist_ok=True)

# ================================
# Debug
# ================================
if __name__ == "__main__":
    print(f"\n[DEBUG] RANDOM_SEED = {RANDOM_SEED}")
    print(f"[DEBUG] train / val / test layouts = {N_TRAIN_LAYOUTS} / {N_VAL_LAYOUTS} / {N_TEST_LAYOUTS}")
    print(f"Finish")