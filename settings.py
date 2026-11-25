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
# 設定RANDOM_SEED
# ================================
RANDOM_SEED = 123

# ================================
# 固定執行環境以及RANDOM_SEED
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    print(f"[INFO] 使用 CUDA 裝置：{torch.cuda.get_device_name(DEVICE)}")
else:
    print("[INFO] 一定要使用cuda")

# ================================
# 場景參數 物理與功率參數
# ================================
TX_ANT   = 4                        # BS  發射天線數 M
RIS_UNIT = 64                       # RIS 反射單元數 N
UAV_COMM = 2                        # 通訊 UE 數 K

NOISE_POWER             = 1e-11     # 雜訊功率(W) 10^-11 -80dBm
TRANSMIT_POWER_TOTAL    = 1.26      # 傳輸功率(W)
RCS_POWER               = 1.0       # 目標複數散射係數的平均功率 E[|alpha_RCS|^2]

# ================================
# 專案元件座標與大尺度衰變係數
# ================================
Q_BS      = (0.0 , 0.0)     # 基站座標
Q_RIS     = (50.0, 0.0)     # RIS座標
Q_TAR     = (5.0, 8.66)     # 感測物座標 60度

PL_EXP_BS_RIS   = 2.3               # 單程PL係數 BS  -> RIS
PL_EXP_RIS_UE   = 2.2               # 單程PL係數 RIS -> UE
PL_EXP_BS_UE    = 3.3               # 單程PL係數 BS  -> UE
PL_EXP_BS_TAR   = 2.7               # 單程PL係數 BS  -> TAR

# ================================
# UE 移動與時槽
# ================================
MOBILITY_ALPHA  = 1.0        # alpha=1: 每條 trajectory 內速度與方向固定 (Gauss_Markov)
V_MEAN          = 1.0        # 長期平均速度1(m/s)
V_INIT          = 1.0        # 初始速度
V_VAR           = 0.01       # 速度高斯擾動變異數 (m/s)^2
THETA_VAR       = 0.01       # 方向高斯擾動變異數 rad^2

LT_BLOCK_SIZE   = 10         # channel coherence intervals , RIS 每 10 slots 更新一次
TOTAL_SLOTS     = 5000       # 每條 trajectory 共 5000 time slots
SLOT_DURATION   = 0.001      # 時間尺度：v=1 m/s、slot=1 ms -> 每個 LT block 移動 1 cm；5000 slots 約 5 m

# Trajectory 初始位置：RIS=(50,0)，UE 隨機生成在 RIS 前方、半徑 3 m 的左側90度圓周上
# 角度採 global coordinate：0 deg=+x，因此前方 (-x side) 對應 90~270 deg
UE_INIT_RADIUS          = 3.0                     # m
UE_INIT_ARC_RANGE_DEG   = (135.0, 225.0)          # 保證 x <= 50；x 範圍為 [47, 47.879]

# Mobility：alpha=1 時，每條 trajectory 的速度、方向皆固定
UE_DIRECTION_RANGE_DEG = (120.0, 240.0)         # 每條 trajectory 的平均移動方向範圍

# ================================
# channel aging
# ================================
CARRIER_FREQUENCY = 8e9             # 中心頻率
SPEED_OF_LIGHT    = 3e8             # 光速
MAX_DOPPLER       = CARRIER_FREQUENCY * V_MEAN / SPEED_OF_LIGHT

# ================================
# 感測門檻
# ================================
SENSING_SNR_THRESHOLD_DB = 5
SENSING_SNR_THRESHOLD    = 10 ** (SENSING_SNR_THRESHOLD_DB / 10.0)

# ================================
# 訓練損失權重
# ================================
REG_SENSING_LOSS_WEIGHT  = 5        
ROB_SENSING_LOSS_WEIGHT  = 1        

# ================================
# dataset 生成，軌跡數量
# ================================
N_TRAIN_TRAJECTORIES = 50           # 共有50組UE軌跡
N_VAL_TRAJECTORIES   = 10
N_TEST_TRAJECTORIES  = 10

# ================================
# Robust / uncertainty injection
# ================================
INJECTION_VARIANCE  = 0.075         # 正規化誤差 e ~ CN(0, 0.075)；乘上 sqrt(beta) 後，每個通道元素的誤差功率為 0.075 * beta
INJECTION_SAMPLES   = 400           # 一個估測通道要有多少誤差通道
OUTAGE_QUANTILE     = 0.05          # Robust tail quantile：每個 estimated channel 對 S 筆 injection 取 Q0.05
TAR_OUTAGE_QUANTILE = 0.10          # 感測可以超出的門檻

# ================================
# 訓練 / 驗證 / 測試
# ================================
REG_EPOCHS      = 2000
ROB_EPOCHS      = 2000

N_BATCHE                = 25
LT_BLOCKS_PER_BATCH     = 50

LEARNING_RATE  = 0.0001
EARLY_STOP_EPS = 1e-6               # REG / ROB 判定 validation loss 改善的最小差值

# ================================
# 神經網路參數
# ================================
LAYER          = 3                  # 隱藏層數，不含輸入層與輸出層
NET            = 1024               # RISNet 和 JointNet 共用的基準隱藏層寬度

# ================================
# 資料夾結構
# ================================
SCENARIO_TAG = f"M{TX_ANT}_N{RIS_UNIT}_K{UAV_COMM}"         # 天線 反射元件 用戶 數量
BASE_RUN_DIR = os.path.join("Two_timescale", SCENARIO_TAG)

MOBILITY_TAG = f"V{V_MEAN:g}_alpha{MOBILITY_ALPHA:g}"       # UE移動速度與alpha更新
MOBILITY_RUN_DIR = os.path.join(BASE_RUN_DIR, MOBILITY_TAG)

# 相同場景和移動條件下，各種 NN 架構共用同一份資料
DATA_DIR = os.path.join(MOBILITY_RUN_DIR, "shared_data")

NN_TAG = f"L{LAYER:02d}_NET{NET}"
Q_SENSING_TAG = f"Q{LT_BLOCK_SIZE}_sense_{SENSING_SNR_THRESHOLD_DB:g}dB" # Q 與 感測門檻
NN_RUN_DIR = os.path.join(MOBILITY_RUN_DIR, "nn", NN_TAG, Q_SENSING_TAG)

REG_PENALTY_TAG = f"reg_{REG_SENSING_LOSS_WEIGHT:g}"
ROB_PENALTY_TAG = f"rob_{ROB_SENSING_LOSS_WEIGHT:g}"

INJECTION_VARIANCE_TAG = str(INJECTION_VARIANCE).replace(".", "p")
ROB_INJECTION_TAG = f"injvar_{INJECTION_VARIANCE_TAG}_injS{INJECTION_SAMPLES}" # 注入大小與數量

REG_CKPT_DIR = os.path.join(NN_RUN_DIR, "regular", REG_PENALTY_TAG)
ROB_CKPT_DIR = os.path.join(NN_RUN_DIR, "robust", ROB_PENALTY_TAG, ROB_INJECTION_TAG)

# 沿用既有的 pretrain 與 AO 結果位置
RESULT_DIR = os.path.join(BASE_RUN_DIR, "results")

# ================================
# 跨程式共用檔案路徑
# ================================
# trajectory_gen.py 產生，channel_gen.py 讀取
TRAIN_LAYOUT_PATH = os.path.join(DATA_DIR, "layout_train.npz")
VAL_LAYOUT_PATH   = os.path.join(DATA_DIR, "layout_val.npz")
TEST_LAYOUT_PATH  = os.path.join(DATA_DIR, "layout_test.npz")

# channel_gen.py 產生，訓練與評估程式讀取
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "dataset_train.npz")
VAL_DATA_PATH   = os.path.join(DATA_DIR, "dataset_val.npz")
TEST_DATA_PATH  = os.path.join(DATA_DIR, "dataset_test.npz")

# 訓練儲存與評估載入使用同一份路徑
REG_THETA_CKPT = os.path.join(REG_CKPT_DIR, "joint_theta_reg.ckpt")
REG_JOINT_CKPT = os.path.join(REG_CKPT_DIR, "joint_beamformer_reg.ckpt")
ROB_THETA_CKPT = os.path.join(ROB_CKPT_DIR, "joint_theta_rob.ckpt")
ROB_JOINT_CKPT = os.path.join(ROB_CKPT_DIR, "joint_beamformer_rob.ckpt")

# ================================
# 建立共用資料夾
# ================================
os.makedirs(BASE_RUN_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REG_CKPT_DIR, exist_ok=True)
os.makedirs(ROB_CKPT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ================================
# Debug
# ================================
if __name__ == "__main__":
    print(f"\n[INFO] RANDOM_SEED = {RANDOM_SEED}")
    if Debug:
        print(f"[INFO] slots/trajectory={TOTAL_SLOTS}, LT block size={LT_BLOCK_SIZE}")
        print(f"[INFO] UE initial semicircle: center={Q_RIS}, radius={UE_INIT_RADIUS} m, arc={UE_INIT_ARC_RANGE_DEG} deg")
        print(f"[INFO] UE speed={V_MEAN:.2f} m/s, direction={UE_DIRECTION_RANGE_DEG} deg")
    print("Finish")