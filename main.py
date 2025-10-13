# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange
from settings import *
from networks_generator import generate_real_channels, estimate_channels
from neural_net import Regular_Net, Robust_Net

# 繪圖
def plot_training_curves(curves: np.ndarray, start: int = 0):
    """
    curves.shape = (epochs, 4)
    [:,0] regular_train, [:,1] robust_train, [:,2] regular_val, [:,3] robust_val
    """
    import matplotlib.pyplot as plt
    x = np.arange(start, curves.shape[0])
    plt.figure()
    plt.plot(x, curves[start:, 0], label="Regular Train")
    plt.plot(x, curves[start:, 1], label="Robust Train")
    plt.plot(x, curves[start:, 2], label="Regular Val")
    plt.plot(x, curves[start:, 3], label="Robust Val")
    plt.xlabel("Epoch")
    plt.ylabel("Objective (mean)")
    plt.title(f"Training Curves - {SETTING_STRING}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()


def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    """將 numpy 複數陣列轉成 torch.complex64 並移到 DEVICE。"""
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAC training script")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the saved training curves and exit")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index for plotting training curves")
    args = parser.parse_args()

    # ===============================
    # 建立階層式輸出資料夾
    # ===============================
    base_dir = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
    ckpt_dir = os.path.join(base_dir, "ckpt")
    curve_dir = os.path.join(base_dir, "training_curves")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)

    # 訓練曲線檔案路徑
    curves_path = os.path.join(curve_dir, f"training_curves_Min-Rate_{SETTING_STRING}.npy")

    # 只繪圖就退出
    if args.plot:
        if not os.path.exists(curves_path):
            raise FileNotFoundError(f"找不到曲線檔案：{curves_path}")
        curves = np.load(curves_path)
        plot_training_curves(curves, args.start)
        raise SystemExit(0)

    # 建立兩個網路與 optimizer
    regular_net = Regular_Net().to(DEVICE)
    robust_net  = Robust_Net().to(DEVICE)
    regular_opt = optim.Adam(regular_net.parameters(), lr=LEARNING_RATE)
    robust_opt  = optim.Adam(robust_net.parameters(),  lr=LEARNING_RATE)
    best_regular, best_robust = -np.inf, -np.inf
    curves = []

    for ep in trange(1, EPOCHS + 1, desc="Epoch"):
        # -----------------------------
        # [TRAIN] 線上產生每個 mini-batch 通道
        # -----------------------------
        regular_net.train()
        robust_net.train()
        reg_obj_ep, rob_obj_ep = 0.0, 0.0

        for _ in range(MINIBATCHES):
            # 產生訓練資料
            H_est_np = estimate_channels(generate_real_channels(BATCH_SIZE))  # (B,M,K) complex np
            H_est = np_to_torch_complex(H_est_np)

            # Regular
            regular_opt.zero_grad(set_to_none=True)
            reg_obj, _ = regular_net(H_est)        # forward 回傳 (objective.mean(), B_dir)
            (-reg_obj).backward()                  # 最大化目標 ⇒ 最小化 -objective
            regular_opt.step()
            reg_obj_ep += reg_obj.item() / MINIBATCHES

            # Robust
            robust_opt.zero_grad(set_to_none=True)
            rob_obj, _ = robust_net(H_est)
            (-rob_obj).backward()
            robust_opt.step()
            rob_obj_ep += rob_obj.item() / MINIBATCHES

        # -----------------------------
        # [VALIDATE] 另一批線上通道
        # -----------------------------
        regular_net.eval()
        robust_net.eval()
        with torch.no_grad():
            H_val_np = estimate_channels(generate_real_channels(N_VAL))
            H_val = np_to_torch_complex(H_val_np)

            reg_val_obj, _ = regular_net(H_val)
            rob_val_obj, _  = robust_net(H_val)

            reg_val = reg_val_obj.item()
            rob_val = rob_val_obj.item()

        curves.append([reg_obj_ep, rob_obj_ep, reg_val, rob_val])

        # 顯示本 epoch 結果
        print(f"[Epoch {ep:03d}] "
              f"[Regular] Tr:{reg_obj_ep: .4e}; Va:{reg_val: .4e} | "
              f"[Robust]  Tr:{rob_obj_ep: .4e}; Va:{rob_val: .4e}")

        # 儲存最佳 checkpoint（以驗證目標為準）
        if reg_val > best_regular:
            # 指定 Regular 模型的輸出路徑
            regular_net.model_path = os.path.join(ckpt_dir, f"regular_net_{SETTING_STRING}.ckpt")
            regular_net.save_model()
            best_regular = reg_val

        if rob_val > best_robust:
            # 指定 Robust 模型的輸出路徑
            robust_net.model_path = os.path.join(ckpt_dir, f"robust_net_{SETTING_STRING}.ckpt")
            robust_net.save_model()
            best_robust = rob_val

        # 追加儲存曲線
        np.save(curves_path, np.array(curves, dtype=np.float32))

    print("Training Script finished!")
