# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from settings import *
# ================================
# UE軌跡生成
# ================================
def Gauss_Markov(initial_positions, alpha, N_slot):    
    N_trajectory, K, _ = initial_positions.shape # 生成維度(軌跡數,UE數,2維座標)

    # 每個 UE 隨機抽取初始方向，同時作為長期平均方向
    directions = np.deg2rad(np.random.uniform(*UE_DIRECTION_RANGE_DEG, size=(N_trajectory, K))).astype(np.float32)
    theta_mean = directions.copy()
    theta = directions.copy()

    # 每個 UE 的初始速度
    speeds = np.full((N_trajectory, K), V_INIT, dtype=np.float32)

    # 儲存所有 trajectory 在所有 time slot 的二維座標
    positions = np.empty((N_trajectory, N_slot, K, 2), dtype=np.float32) # 創空間存放(有幾條軌跡,1軌跡有多少slot,有幾個用戶,2維座標)
    positions[:, 0] = initial_positions                                  # 將initial_positions存放到一開始

    for t in range(1, N_slot):

        # 每個 UE 產生獨立高斯擾動
        v_hat     = np.random.normal(0, np.sqrt(V_VAR), size=(N_trajectory, K)).astype(np.float32)
        theta_hat = np.random.normal(0, np.sqrt(THETA_VAR), size=(N_trajectory, K)).astype(np.float32)

        # Gauss-Markov 速度更新
        speeds = alpha * speeds + (1 - alpha) * V_MEAN + np.sqrt(1 - alpha**2) * v_hat

        # Gauss-Markov 方向更新
        theta = alpha * theta + (1 - alpha) * theta_mean + np.sqrt(1 - alpha**2) * theta_hat

        # 公式 (2)：UE 二維座標更新
        direction_vector = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
        positions[:, t] = positions[:, t-1] + speeds[..., None] * SLOT_DURATION * direction_vector

    return positions, directions


# ================================
# Dataset 生成
# ================================
def build_and_save_dataset(trajectories, save_path, dataset_name):
    """
    輸入
    trajectories -> 要生成的軌跡數量
    save_path    -> 儲存npz路徑(已經自帶名子)
    dataset_name -> 判斷要不要畫圖用的檔名

    根據輸入軌跡數量 生成固定 dataset,並儲存成 .npz
    """
    n_trajectories = int(trajectories)               # 需要生成的軌跡數量
    # 生成初始位置
    phi_min_deg, phi_max_deg = UE_INIT_ARC_RANGE_DEG # 初始位置用角度由算出
    
    initial_angles_deg = np.random.uniform(
        low  = phi_min_deg,
        high = phi_max_deg,
        size=(n_trajectories, UAV_COMM),
    ).astype(np.float32)

    initial_angles_rad = np.deg2rad(initial_angles_deg).astype(np.float32)
    initial_positions  = (np.asarray(Q_RIS)+ UE_INIT_RADIUS* np.stack([np.cos(initial_angles_rad), np.sin(initial_angles_rad)], axis=-1)).astype(np.float32) # (trajectories, UAV_COMM, 2)
    # 產生完整 trajectories
    positions, directions = Gauss_Markov(initial_positions,MOBILITY_ALPHA,TOTAL_SLOTS)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        positions=positions,
        initial_positions=initial_positions,
        directions=directions,
    )
    print(f"[{dataset_name}] dataset 已儲存到：{save_path}")

    if Debug:
        print(f"\n[{dataset_name}]")
        print(f"positions          = {positions.shape}")
        print(f"initial_positions  = {initial_positions.shape}")
        print(f"directions         = {directions.shape}")
        print(f"saved to           = {save_path}")
    if dataset_name == "train":
        plot_train_trajectories(positions)


# ================================
# 畫 training trajectories
# ================================
def plot_train_trajectories(positions):
    """
    positions:
        shape = (N_trajectory, N_slot, K, 2)

    畫出 training dataset 中所有 UE 軌跡。
    """

    plt.figure(figsize=(10, 6))

    N_trajectory, _, K, _ = positions.shape

    # 畫每一條 trajectory 的每一個 UE
    for n in range(N_trajectory):
        for k in range(K):
            plt.plot(
                positions[n, :, k, 0],   # x
                positions[n, :, k, 1],   # y
                alpha=0.4,
            )

    # 固定節點
    plt.scatter(
        Q_BS[0], Q_BS[1],
        marker="s",
        s=80,
        label="BS",
    )

    plt.scatter(
        Q_RIS[0], Q_RIS[1],
        marker="D",
        s=80,
        label="RIS",
    )

    plt.scatter(
        Q_TAR[0], Q_TAR[1],
        marker="*",
        s=120,
        label="Target",
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Training UE Trajectories")

    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


# ================================
# Main: generate train / val / test datasets
# ================================
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print(f"[INFO] TOTAL_SLOTS={TOTAL_SLOTS}, SLOT_DURATION={SLOT_DURATION}s, LT_BLOCK_SIZE={LT_BLOCK_SIZE}")
    print(f"[INFO] init: RIS={Q_RIS}, radius={UE_INIT_RADIUS}m, arc={UE_INIT_ARC_RANGE_DEG}deg")

    build_and_save_dataset(
        trajectories    = N_TRAIN_TRAJECTORIES,
        save_path       = TRAIN_LAYOUT_PATH,
        dataset_name    = "train"
    )

    build_and_save_dataset(
        trajectories    = N_VAL_TRAJECTORIES,
        save_path       = VAL_LAYOUT_PATH,
        dataset_name    = "val"
    )

    build_and_save_dataset(
        trajectories    = N_TEST_TRAJECTORIES,
        save_path       = TEST_LAYOUT_PATH,
        dataset_name    = "test"
    )

    print("\n[INFO] All trajectories generated successfully.")

