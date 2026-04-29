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
UAV_COMM = 2                # 通訊 UE / UAV 數 K
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
# ISAC / 損失權重
# ================================
SENSING_SNR_THRESHOLD_dB = 15
SENSING_SNR_THRESHOLD = 10 ** (SENSING_SNR_THRESHOLD_dB / 10.0)

SENSING_LOSS_WEIGHT  = 500.0
RE_POWER_LOSS_WEIGHT = 100.0
TX_POWER_LOSS_WEIGHT = 100.0

# ================================
# 幾何場景
# ================================
Q_BS      = (0.0, 0.0)
Q_RIS     = (50.0, 0.0)
Q_UAV_TAR = (10.0, 2.0)

UE_RADIUS = 3.0
RIS_FRONT_NORMAL = (-1.0, 0.0)   # RIS 法線朝向原點

# ================================
# dataset生成 (UE位置)
# ================================
# 說明：
# 1. 先生成 LONG_TERM_LAYOUT_SAMPLES 組 UE layouts
# 2. 再固定切成 train / val / test = 8 : 1 : 1
# 3. 每個 layout 底下再由 rician.py 生成固定通道資料：
#    - long-term 用的 true/statistical channels
#    - short-term 用的 estimated channels

# ---------- layout 總數 ----------
LONG_TERM_LAYOUT_SAMPLES = 1000

# ---------- split 比例 ----------
TRAIN_LAYOUT_RATIO = 0.8
VAL_LAYOUT_RATIO   = 0.1
TEST_LAYOUT_RATIO  = 0.1

# ---------- 每個 layout 的固定通道樣本數 ----------
# Long-term 用：true / statistical channels
LONGTERM_TRUE_SAMPLES_PER_LAYOUT = 128

# Short-term 用：estimated channels
SHORTTERM_EST_CHANNELS_PER_LAYOUT = 2000

def random_points_on_circle(center, radius, num_points, normal=(-1.0, 0.0)):
    """
    在指定圓周上隨機取點，並只保留 normal 指向的正面半圓。
    回傳:
        [(x1, y1), (x2, y2), ...]
    """
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

        # 只保留 RIS 正面半圓
        if (x - cx) * nx + (y - cy) * ny >= 0:
            points.append((round(float(x), 2), round(float(y), 2)))

    return points

# ---------- 生成 layout 母體 ----------
# UE_LAYOUT_BANK[n] = [(x1,y1), (x2,y2), ...]

UE_LAYOUT_BANK = [
    random_points_on_circle(
        center=Q_RIS,
        radius=UE_RADIUS,
        num_points=UAV_COMM,
        normal=RIS_FRONT_NORMAL
    )
    for _ in range(LONG_TERM_LAYOUT_SAMPLES)
]


# ---------- 固定 split ----------
layout_indices = np.arange(LONG_TERM_LAYOUT_SAMPLES)
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(layout_indices)

N_TRAIN_LAYOUTS = int(TRAIN_LAYOUT_RATIO * LONG_TERM_LAYOUT_SAMPLES)
N_VAL_LAYOUTS   = int(VAL_LAYOUT_RATIO   * LONG_TERM_LAYOUT_SAMPLES)
N_TEST_LAYOUTS  = LONG_TERM_LAYOUT_SAMPLES - N_TRAIN_LAYOUTS - N_VAL_LAYOUTS

TRAIN_LAYOUT_IDS = layout_indices[:N_TRAIN_LAYOUTS]
VAL_LAYOUT_IDS   = layout_indices[N_TRAIN_LAYOUTS:N_TRAIN_LAYOUTS + N_VAL_LAYOUTS]
TEST_LAYOUT_IDS  = layout_indices[N_TRAIN_LAYOUTS + N_VAL_LAYOUTS:]

TRAIN_UE_LAYOUTS = [UE_LAYOUT_BANK[i] for i in TRAIN_LAYOUT_IDS]
VAL_UE_LAYOUTS   = [UE_LAYOUT_BANK[i] for i in VAL_LAYOUT_IDS]
TEST_UE_LAYOUTS  = [UE_LAYOUT_BANK[i] for i in TEST_LAYOUT_IDS]

# ---------- 統一包裝 ----------
UE_LAYOUT_SPLIT = {
    "train_ids": TRAIN_LAYOUT_IDS,
    "val_ids": VAL_LAYOUT_IDS,
    "test_ids": TEST_LAYOUT_IDS,
    "train_layouts": TRAIN_UE_LAYOUTS,
    "val_layouts": VAL_UE_LAYOUTS,
    "test_layouts": TEST_UE_LAYOUTS,
}

# ================================
# Robust / uncertainty injection
# ================================
INJECTION_VARIANCE = 0.075
INJECTION_SAMPLES  = 1000
OUTAGE_QUANTILE    = 0.05

# ================================
# 訓練 / 驗證 / 測試
# ================================
EPOCHS      = 100
MINIBATCHES = 50
BATCH_SIZE  = 1000

LEARNING_RATE = 1e-3

# ================================
# 檔名與資料夾
# ================================
SCENARIO_TAG = f"M{TX_ANT}_Ris{RIS_UNIT}_K{UAV_COMM}"

THR_TAG = (
    f"THR_{SENSING_SNR_THRESHOLD_dB}db_"
    f"punish_{(SENSING_LOSS_WEIGHT, RE_POWER_LOSS_WEIGHT, TX_POWER_LOSS_WEIGHT)}"
)

SETTING_STRING = (
    f"N0_{NOISE_POWER}_"
    f"INJERR_{INJECTION_VARIANCE}_"
    f"TXpower_{TRANSMIT_POWER_TOTAL}W_"
    f"LTlayouts_{LONG_TERM_LAYOUT_SAMPLES}"
)

PROJECT_DIR = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
CKPT_DIR    = os.path.join(PROJECT_DIR, "ckpt")
CURVE_DIR   = os.path.join(PROJECT_DIR, "training_curves")
DATA_DIR    = os.path.join(PROJECT_DIR, "data")

# 之後固定資料集可存這裡
TRAIN_DATASET_PATH = os.path.join(DATA_DIR, "dataset_train.npz")
VAL_DATASET_PATH   = os.path.join(DATA_DIR, "dataset_val.npz")
TEST_DATASET_PATH  = os.path.join(DATA_DIR, "dataset_test.npz")

os.makedirs(PROJECT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == "__main__":
    print(f"\n[DEBUG] RANDOM_SEED = {RANDOM_SEED}")
    print(f"[DEBUG] LONG_TERM_LAYOUT_SAMPLES = {LONG_TERM_LAYOUT_SAMPLES}")
    print(f"[DEBUG] train / val / test = {N_TRAIN_LAYOUTS} / {N_VAL_LAYOUTS} / {N_TEST_LAYOUTS}")

    print("[DEBUG] First 3 train layouts:")
    for i in range(min(3, len(TRAIN_UE_LAYOUTS))):
        print(f"  Train Layout {i}: {TRAIN_UE_LAYOUTS[i]}")

    print(f"[INFO] 專案資料夾：{PROJECT_DIR}")