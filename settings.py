import os
import random
import numpy as np
import torch

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
# 系統維度
# ================================
TX_ANT   = 6                # BS 發射天線數 M
RIS_UNIT = 40               # RIS 反射單元數 N
UAV_COMM = 2                # 通訊 UE 數 K
RADAR_STREAMS = TX_ANT      # sensing waveforms 數量

# ================================
# 物理與功率參數
# ================================
FC = 3.5e9
C0 = 3e8
LAMBDA = C0 / FC

NOISE_POWER = 10e-11
ESTIMATION_PILOT_POWER = 10e-10
TRANSMIT_POWER_TOTAL = 1.0

# ================================
# ISAC / 損失權重預設值
# ================================
SENSING_SNR_THRESHOLD_dB = 16
SENSING_SNR_THRESHOLD = 10 ** (SENSING_SNR_THRESHOLD_dB / 10.0)

# 注意：
# 這裡只保留 default value
# 後續 ST penalty sweep 不應該靠修改 settings.py
# 而應由 train_st.py / sweep.py 傳入實際 penalty
REG_SENSING_LOSS_WEIGHT = 200.0
ROB_SENSING_LOSS_WEIGHT = 0.5
RE_POWER_LOSS_WEIGHT    = 250.0

# ================================
# 幾何場景
# ================================
Q_BS      = (0.0, 0.0)
Q_RIS     = (50.0, 0.0)
Q_UAV_TAR = (10.0, 2.0)

UE_RADIUS = 3.0
RIS_FRONT_NORMAL = (-1.0, 0.0)   # RIS 法線朝向原點

# ================================
# dataset 生成
# ================================
"""
說明：
    1. train / validation / test 各自獨立生成 UE layouts。
    2. 每個 layout 底下再由 rician.py 生成固定通道資料：
        - long-term  用的 statistical channels
        - short-term 用的 estimated channels
    3. dataset 與 REG / ROB penalty 無關
    4. 所有 ST penalty sweep 共用同一份 shared_data
"""

N_TRAIN_LAYOUTS = 300
N_VAL_LAYOUTS   = 100
N_TEST_LAYOUTS  = 50

LONGTERM_TRUE_SAMPLES_PER_LAYOUT  = 128
SHORTTERM_EST_CHANNELS_PER_LAYOUT = 500


def random_points_on_circle(center, radius, num_points, normal=(-1.0, 0.0)):
    """
    在指定圓周上隨機取點，並只保留 normal 指向的正面半圓。
    回傳:
        [(x1, y1), (x2, y2), ...]
    """
    cx, cy = center
    nx, ny = normal

    norm_n = np.hypot(nx, ny)
    nx /= norm_n
    ny /= norm_n
    points = []
    while len(points) < num_points:
        theta = np.random.uniform(0.0, 2.0 * np.pi)

        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)

        # 只保留 RIS 正面半圓
        if (x - cx) * nx + (y - cy) * ny >= 0:
            points.append((round(float(x), 2),round(float(y), 2)))

    return points


def layout_gen(num_layouts):
    layouts = []

    for _ in range(num_layouts):
        ue_layout = random_points_on_circle(
            center=Q_RIS,
            radius=UE_RADIUS,
            num_points=UAV_COMM,
            normal=RIS_FRONT_NORMAL,
        )

        layouts.append(ue_layout)

    return layouts

TRAIN_UE_LAYOUTS = layout_gen(N_TRAIN_LAYOUTS)
VAL_UE_LAYOUTS   = layout_gen(N_VAL_LAYOUTS)
TEST_UE_LAYOUTS  = layout_gen(N_TEST_LAYOUTS)

# ================================
# Robust / uncertainty injection
# ================================
"""
注意：
    1. INJECTION_VARIANCE 是本實驗主設定，會進入 BASE_RUN_DIR 名稱。
    2. ROB training 預設使用此 injection variance
    3. eval sweep 時，不應該修改此變數；
       eval script 應額外傳入 test injection variance list
"""

INJECTION_VARIANCE = 0.075
INJECTION_SAMPLES  = 200
OUTAGE_QUANTILE    = 0.05

# ================================
# 訓練 / 驗證 / 測試
# ================================
LT_EPOCHS   = 400
REG_EPOCHS  = 1400
ROB_EPOCHS  = 800

MINIBATCHES = 50
PLOT_MOVING_AVG_WINDOW = 50 # plot用畫平滑曲線

# Short-term mixed-layout minibatch
ST_BATCH_LAYOUTS = 30
ST_BATCH_EST_CHANNELS_PER_LAYOUT = 100

LT_LEARNING_RATE  = 0.001
REG_LEARNING_RATE = 0.0005
ROB_LEARNING_RATE = 0.0005

# ================================
# 實驗資料夾結構
# ================================
SCENARIO_TAG = f"M{TX_ANT}_Ris{RIS_UNIT}_K{UAV_COMM}"

SETTING_STRING = (
    f"N0_{NOISE_POWER}_"
    f"INJERR_{INJECTION_VARIANCE}_"
    f"TXpower_{TRANSMIT_POWER_TOTAL}"
)

# Root:
# MLP/M6_Ris40_K2/N0_1e-10_INJERR_0.075_TXpower_1.0/
BASE_RUN_DIR = os.path.join("MLP",SCENARIO_TAG,SETTING_STRING)

# shared dataset
DATA_DIR = os.path.join(BASE_RUN_DIR, "shared_data")

TRAIN_DATASET_PATH = os.path.join(DATA_DIR, "dataset_train.npz")
VAL_DATASET_PATH   = os.path.join(DATA_DIR, "dataset_val.npz")
TEST_DATASET_PATH  = os.path.join(DATA_DIR, "dataset_test.npz")

# shared long-term model
LT_DIR = os.path.join(BASE_RUN_DIR, "LT")
LT_CKPT_DIR = os.path.join(LT_DIR, "ckpt")
LT_CURVE_DIR = os.path.join(LT_DIR, "training_curves")

LONGTERM_CKPT_PATH = os.path.join(LT_CKPT_DIR, "longterm.ckpt")
LONGTERM_CURVE_PATH = os.path.join(LT_CURVE_DIR, "longterm_curves.npy")
LONGTERM_CURVE_FIG_PATH = os.path.join(LT_CURVE_DIR, "longterm_objective_curve.jpg")

# short-term penalty sweep
ST_SWEEP_DIR = os.path.join(BASE_RUN_DIR, "ST_sweep")

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