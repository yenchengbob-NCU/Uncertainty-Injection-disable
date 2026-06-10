# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet


# ================================
# Helpers
# ================================
def load_longterm_dataset(npz_path: str, dataset_name: str):
    """
    載入 long-term 訓練需要的 dataset 欄位
    回傳:dataset : dict
    """
    required_keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "lt_h_dk_true",
        "lt_h_rk_true",
        "lt_G_true",
    ]

    with np.load(npz_path) as data:
        dataset = {key: data[key] for key in required_keys}

    print(f"[{dataset_name}] loaded: {npz_path}")
    print(f"[{dataset_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{dataset_name}] lt_h_dk_true shape = {dataset['lt_h_dk_true'].shape}")
    print(f"[{dataset_name}] lt_h_rk_true shape = {dataset['lt_h_rk_true'].shape}")
    print(f"[{dataset_name}] lt_G_true shape    = {dataset['lt_G_true'].shape}")

    return dataset


def moving_average(x: np.ndarray, window: int):
    """
    計算 moving average。
    輸入:
        x      : 1D numpy array
        window : moving average window size
    回傳:
        x_ma : moving averaged array
    """
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

# ================================
# Long-term objective
# ================================
def forward_longterm_one_layout_objective(
    longterm_net: LongTermPositionNet,
    dataset,
    layout_id: int,
):
    """
    這裡僅計算1個layout下128統計通道
    """
    ue_layout = dataset["ue_layouts"][layout_id]                        # 載入單一layout
    theta_lt, W_C_lt, W_R_lt = longterm_net(ue_layout)                  # 輸入網路

    W_C_lt, W_R_lt = longterm_net.normalize_tx_power(W_C_lt, W_R_lt)    # TXpower的正規化
    
    # 載入該layout 統計通道
    h_dk_true = dataset["lt_h_dk_true"][layout_id]
    h_rk_true = dataset["lt_h_rk_true"][layout_id]
    G_true    = dataset["lt_G_true"][layout_id]
    # 載入該layout Path loss
    pl_BS_UE     = dataset["pl_BS_UE"][layout_id]
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]

    S = h_dk_true.shape[0] # 該layout下統計通道數量

    theta_rep = theta_lt.expand(S, RIS_UNIT)
    W_C_rep = W_C_lt.expand(S, TX_ANT, UAV_COMM)
    W_R_rep = W_R_lt.expand(S, TX_ANT, RADAR_STREAMS)

    sumrate_mean = longterm_net.compute_sum_rate(
        h_dk=h_dk_true,             # 統計通道
        h_rk=h_rk_true,
        G=G_true,
        theta=theta_rep,            # NN輸出 : RIS向量
        W_R=W_R_rep,                # NN輸出 : sensing beamformer
        W_C=W_C_rep,                # NN輸出 : cumm beamformer
        pl_BS_UE=pl_BS_UE,
        pl_BS_RIS_UE=pl_BS_RIS_UE,
    ).mean()                        # 計算同一個 UE layout 底下，S = 128 組 statistical channels 的 sum-rate 平均

    # 計算同一個 UE layout 底下，S = 128 組 statistical channels 的 theta_penalty 平均
    theta_penalty = longterm_net.compute_ris_amplitude_penalty(theta_lt).mean()

    objective = sumrate_mean - RIS_POWER_LOSS_WEIGHT * theta_penalty # 目標函數

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "theta_penalty_mean": theta_penalty.detach(),
        "objective": objective.detach(),
    }

    return objective, logs


# ================================
# Long-term validation
# ================================
@torch.no_grad()
def validate_longterm(
    longterm_net: LongTermPositionNet,
    val_dataset,
):
    """
    使用固定 validation dataset 的全部 layouts 進行 validation。
    """
    longterm_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0]

    total_obj = 0.0
    total_sumrate = 0.0
    total_theta_pen = 0.0

    for layout_id in range(n_val_layouts):
        obj, logs = forward_longterm_one_layout_objective(  # 算出"1個"統計通道的 obj & log
            longterm_net=longterm_net,
            dataset=val_dataset,
            layout_id=layout_id,
        )

        #該輪 epoch 的 val log 統計
        total_obj += float(obj.detach().cpu())
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_theta_pen += float(logs["theta_penalty_mean"].cpu())

    return {
        "objective": total_obj / n_val_layouts,
        "sum_rate_mean": total_sumrate / n_val_layouts,
        "theta_penalty_mean": total_theta_pen / n_val_layouts,
        "num_val_layouts": n_val_layouts,
    }


# ================================
# Train long-term
# ================================
def train_longterm(longterm_net, train_dataset, val_dataset):
    """
    訓練 long-term network
    每次 update:
        1. 隨機抽 1 個 UE layout
        2. 使用該 layout 的所有 LT statistical channels
        3. 更新 long-term network,使平均 sum-rate 最大並限制 RIS amplitude
    """
    optimizer = optim.Adam(longterm_net.parameters(), lr=LT_LEARNING_RATE)
    ckpt_path = os.path.join(LT_CKPT_DIR, "longterm.ckpt")
    curve_path = os.path.join(LT_CURVE_DIR, "longterm_curves.npy")

    best_val_obj = -np.inf
    curves = []

    for ep in trange(1, LT_EPOCHS + 1, desc="LongTerm"):
        longterm_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_theta_pen_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_id = int(np.random.randint(0, train_dataset["ue_layouts"].shape[0]))  # 隨機抽出1組layout

            optimizer.zero_grad(set_to_none=True)                   # 重製模型參數的梯度

            obj, logs = forward_longterm_one_layout_objective(      # 算出"1個"統計通道的 obj & log
                longterm_net=longterm_net,
                dataset=train_dataset,
                layout_id=layout_id,
            )

            loss = -obj
            loss.backward()                                         # 計算梯度
            optimizer.step()                                        # 更新網路

            #該輪 epoch 的 training log 統計
            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_theta_pen_ep += float(logs["theta_penalty_mean"].cpu()) / MINIBATCHES

        val_logs = validate_longterm(
            longterm_net=longterm_net,
            val_dataset=val_dataset,
        )

        curves.append([
            train_obj_ep,
            val_logs["objective"],
            train_sumrate_ep,
            val_logs["sum_rate_mean"],
            train_theta_pen_ep,
            val_logs["theta_penalty_mean"],
        ])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[LT Epoch {ep:03d}/{LT_EPOCHS}]\n"
            f"  TrainObj      = {train_obj_ep:.4e} | "
            f"TrainSumRate  = {train_sumrate_ep:.4e} | "
            f"TrainThetaPen = {train_theta_pen_ep:.4e} |\n"
            f"  ValObj        = {val_logs['objective']:.4e} | "
            f"ValSumRate    = {val_logs['sum_rate_mean']:.4e} | "
            f"ValThetaPen   = {val_logs['theta_penalty_mean']:.4e} |"
        )

        if val_logs["objective"] > best_val_obj:
            best_val_obj = val_logs["objective"]
            longterm_net.save_model(path=ckpt_path, verbose=False)

    print(f"[LongTerm] best ValObj = {best_val_obj:.4e}")
    longterm_net.load_model(path=ckpt_path, verbose=True)


# ================================
# Plot long-term objective curve
# ================================
def plot_longterm_objective_curve():
    curve_path = os.path.join(LT_CURVE_DIR, "longterm_curves.npy")
    curves = np.load(curve_path)

    # 從 epoch N 開始畫圖，避免初期 objective 過低壓縮後期曲線
    start_idx = 0

    epochs = np.arange(start_idx + 1, curves.shape[0] + 1)

    train_obj = curves[start_idx:, 0]
    val_obj   = curves[start_idx:, 1]

    window = PLOT_MOVING_AVG_WINDOW

    train_obj_ma = moving_average(train_obj, window)
    val_obj_ma   = moving_average(val_obj, window)

    ma_epochs = epochs[window - 1:]

    fig_path = os.path.join(LT_CURVE_DIR, "longterm_objective_curve.jpg")

    plt.figure(figsize=(9, 5.5))

    # raw curves
    plt.plot(
        epochs,
        train_obj,
        label="Train Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        epochs,
        val_obj,
        label="Validation Objective",
        alpha=0.35,
        linewidth=1.0
    )

    # moving average curves
    plt.plot(
        ma_epochs,
        train_obj_ma,
        label=f"Train Objective MA({window})",
        linewidth=2.2
    )

    plt.plot(
        ma_epochs,
        val_obj_ma,
        label=f"Validation Objective MA({window})",
        linewidth=2.2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("Long-term Training / Validation Objective")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    print(f"[PLOT] 已儲存 long-term objective curve : {fig_path}")

    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train long-term network only")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="只讀取 long-term curve .npy 並畫圖，不進行訓練"
    )
    args = parser.parse_args()

    #  --plot 功能
    if args.plot:
        plot_longterm_objective_curve()
        raise SystemExit

    print("[INFO] 載入固定 datasets ...")

    train_dataset = load_longterm_dataset(os.path.join(DATA_DIR, "dataset_train.npz"),"train")

    val_dataset = load_longterm_dataset(os.path.join(DATA_DIR, "dataset_val.npz"),"val")

    longterm_net = LongTermPositionNet(ckpt_kind=None).to(DEVICE)

    print("[INFO] 開始 long-term 訓練 ...")
    print(f"[INFO] LT_EPOCHS = {LT_EPOCHS}")
    print(f"[INFO] MINIBATCHES = {MINIBATCHES}")
    print(f"[INFO] LT_LEARNING_RATE = {LT_LEARNING_RATE}")
    print("[INFO] Training rule: one minibatch = one layout objective.")

    train_longterm(
        longterm_net=longterm_net,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("[INFO] Long-term training finished.")