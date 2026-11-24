import os
import numpy as np
import matplotlib.pyplot as plt
from settings import *

Debug = True          # 終端印出檢查
# ================================
# 基本幾何 / 小工具
# ================================
def Gauss_Markov(initial_positions, alpha, N_slot):
    N_trajectory, K, _ = initial_positions.shape

    # 每個 UE 隨機抽取一次長期平均移動方向，並以固定 UE_SPEED 作為平均速度
    mean_directions = np.deg2rad(np.random.uniform(*UE_DIRECTION_RANGE_DEG, size=(N_trajectory, K))).astype(np.float32)
    mean_speeds = np.full((N_trajectory, K), UE_SPEED, dtype=np.float32)

    # t=0 的瞬時速度與方向設為各 trajectory 的長期平均值
    speeds = mean_speeds.copy()
    directions = mean_directions.copy()

    # 儲存所有 trajectory 在所有 time slot 的二維座標
    positions = np.empty((N_trajectory, N_slot, K, 2), dtype=np.float32)
    positions[:, 0] = initial_positions

    for t in range(1, N_slot):
        speed_noise     = np.random.normal(0.0, UE_SPEED_STD, size=(N_trajectory, K)).astype(np.float32)
        direction_noise = np.deg2rad(np.random.normal(0.0, UE_DIRECTION_STD_DEG, size=(N_trajectory, K))).astype(np.float32)

        speeds      = alpha * speeds + (1.0 - alpha) * mean_speeds + np.sqrt(1.0 - alpha ** 2) * speed_noise
        directions  = alpha * directions + (1.0 - alpha) * mean_directions + np.sqrt(1.0 - alpha ** 2) * direction_noise

        dx = speeds * SLOT_DURATION * np.cos(directions)
        dy = speeds * SLOT_DURATION * np.sin(directions)

        positions[:, t, :, 0] = positions[:, t - 1, :, 0] + dx
        positions[:, t, :, 1] = positions[:, t - 1, :, 1] + dy

    return positions, mean_directions

# ================================
# Dataset 生成
# ================================
def build_and_save_dataset(trajectories, save_path, dataset_name):
    """
    根據輸入軌跡數量 生成固定 dataset,並儲存成 .npz
    """
    n_trajectories = int(trajectories)      # 需要生成的軌跡數量
    # 生成初始位置
    phi_min_deg, phi_max_deg = UE_INIT_ARC_RANGE_DEG # 初始位置用角度由算出
    initial_angles_deg = np.random.uniform(
        low=phi_min_deg,
        high=phi_max_deg,
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
        Q_UAV_TAR[0], Q_UAV_TAR[1],
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
    print(f"[INFO] TOTAL_SLOTS={TOTAL_SLOTS}, DT={SLOT_DURATION}s, LT_BLOCK_SIZE={LT_BLOCK_SIZE}")
    print(f"[INFO] init: RIS={Q_RIS}, radius={UE_INIT_RADIUS}m, arc={UE_INIT_ARC_RANGE_DEG}deg")

    build_and_save_dataset(
        trajectories    = N_TRAIN_TRAJECTORIES,
        save_path       = os.path.join(DATA_DIR, "layout_train.npz"),
        dataset_name    = "train"
    )

    build_and_save_dataset(
        trajectories    = N_VAL_TRAJECTORIES,
        save_path       = os.path.join(DATA_DIR, "layout_val.npz"),
        dataset_name    = "val"
    )

    build_and_save_dataset(
        trajectories    = N_TEST_TRAJECTORIES,
        save_path       = os.path.join(DATA_DIR, "layout_test.npz"),
        dataset_name    = "test"
    )

    print("\n[INFO] All trajectories generated successfully.")

