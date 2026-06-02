# -*- coding: utf-8 -*-
import argparse
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
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def load_longterm_dataset(npz_path: str, dataset_name: str):
    """
    載入 long-term 訓練需要的 dataset 欄位。

    回傳:
        dataset : dict
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
    計算單一 layout 的 long-term objective。

    流程:
        1. layout -> theta_LT, W_C_LT, W_R_LT
        2. 使用該 layout 的 statistical channels
        3. 計算 layout-level mean sum-rate
        4. 扣除 RIS amplitude penalty

    回傳:
        objective : torch.Tensor
        logs      : dict
    """
    ue_layout_np = dataset["ue_layouts"][layout_id]                     # shape = (K,2)
    ue_layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)          # shape = (1,K,2)

    theta_lt, W_C_lt, W_R_lt = longterm_net(ue_layout_t)                # (1,N), (1,M,K), (1,M,RADAR_STREAMS)
    W_C_lt, W_R_lt = longterm_net.normalize_tx_power(W_C_lt, W_R_lt)    # TX power scaling

    h_dk_true = np_to_torch_complex(dataset["lt_h_dk_true"][layout_id]) # shape = (S,M,K)
    h_rk_true = np_to_torch_complex(dataset["lt_h_rk_true"][layout_id]) # shape = (S,N,K)
    G_true    = np_to_torch_complex(dataset["lt_G_true"][layout_id])    # shape = (S,N,M)

    pl_BS_UE     = dataset["pl_BS_UE"][layout_id]                       # shape = (K,)
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]                   # shape = (K,)

    S = h_dk_true.shape[0]                                              # statistical channels 數量

    theta_rep = theta_lt.expand(S, RIS_UNIT)                            # shape = (S,N)
    W_C_rep   = W_C_lt.expand(S, TX_ANT, UAV_COMM)                      # shape = (S,M,K)
    W_R_rep   = W_R_lt.expand(S, TX_ANT, RADAR_STREAMS)                 # shape = (S,M,RADAR_STREAMS)

    sumrate_mean = longterm_net.compute_sum_rate(
        h_dk=h_dk_true,
        h_rk=h_rk_true,
        G=G_true,
        theta=theta_rep,
        W_R=W_R_rep,
        W_C=W_C_rep,
        pl_BS_UE=pl_BS_UE,
        pl_BS_RIS_UE=pl_BS_RIS_UE,
    ).mean()

    theta_penalty = longterm_net.compute_ris_amplitude_penalty(
        theta_lt
    ).mean()

    objective = (
        sumrate_mean
        - RE_POWER_LOSS_WEIGHT * theta_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "theta_penalty_mean": theta_penalty.detach(),
        "objective": objective.detach(),
        "layout_id": layout_id,
        "num_statistical_channels": S,
    }

    return objective, logs


# ================================
# Long-term validation
# ================================
@torch.no_grad()
def validate_longterm_sampled(
    longterm_net: LongTermPositionNet,
    val_dataset,
    num_val_layouts: int,
):
    """
    從 validation pool 抽出 num_val_layouts 個 layouts 進行 validation。

    回傳:
        logs : dict
    """
    longterm_net.eval()

    n_val_pool = val_dataset["ue_layouts"].shape[0]
    layout_ids = np.random.choice(
        n_val_pool,
        size=num_val_layouts,
        replace=False
    )

    total_obj = 0.0
    total_sumrate = 0.0
    total_theta_pen = 0.0

    for layout_id in layout_ids:
        obj_t, logs = forward_longterm_one_layout_objective(
            longterm_net=longterm_net,
            dataset=val_dataset,
            layout_id=int(layout_id),
        )

        total_obj += float(obj_t.detach().cpu())
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_theta_pen += float(logs["theta_penalty_mean"].cpu())

    return {
        "objective": total_obj / num_val_layouts,
        "sum_rate_mean": total_sumrate / num_val_layouts,
        "theta_penalty_mean": total_theta_pen / num_val_layouts,
        "num_val_layouts": num_val_layouts,
    }


# ================================
# Train long-term
# ================================
def train_longterm(longterm_net, train_dataset, val_dataset):
    """
    訓練 long-term network。

    訓練規則:
        1. 每個 epoch 有 MINIBATCHES 次 update
        2. 每次 update 只抽 1 個 train layout
        3. 該 layout 使用所有 LT statistical channels 計算 objective
    """
    optimizer = optim.Adam(
        longterm_net.parameters(),
        lr=LT_LEARNING_RATE
    )

    best_val_obj = -np.inf
    curves = []

    curve_path = LONGTERM_CURVE_PATH

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(1, LT_EPOCHS + 1, desc="LongTerm"):
        longterm_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_theta_pen_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_id = int(np.random.randint(0, n_train_layouts))       # 每次 update 只抽 1 個 layout

            optimizer.zero_grad(set_to_none=True)

            obj, logs = forward_longterm_one_layout_objective(
                longterm_net=longterm_net,
                dataset=train_dataset,
                layout_id=layout_id,
            )

            loss = -obj
            loss.backward()
            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_theta_pen_ep += float(logs["theta_penalty_mean"].cpu()) / MINIBATCHES

        val_logs = validate_longterm_sampled(
            longterm_net=longterm_net,
            val_dataset=val_dataset,
            num_val_layouts=N_VAL_LAYOUTS,
        )

        val_obj_ep = val_logs["objective"]
        val_sumrate_ep = val_logs["sum_rate_mean"]
        val_theta_pen_ep = val_logs["theta_penalty_mean"]

        curves.append([
            train_obj_ep,
            val_obj_ep,
            train_sumrate_ep,
            val_sumrate_ep,
            train_theta_pen_ep,
            val_theta_pen_ep,
        ])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[LT Epoch {ep:03d}/{LT_EPOCHS}]\n"
            f"  TrainObj      = {train_obj_ep:.4e} | "
            f"TrainSumRate  = {train_sumrate_ep:.4e} | "
            f"TrainThetaPen = {train_theta_pen_ep:.4e} |\n"
            f"  ValObj        = {val_obj_ep:.4e} | "
            f"ValSumRate    = {val_sumrate_ep:.4e} | "
            f"ValThetaPen   = {val_theta_pen_ep:.4e} |"
        )

        if val_obj_ep > best_val_obj:
            best_val_obj = val_obj_ep
            longterm_net.save_model(path=LONGTERM_CKPT_PATH, verbose=False)

    print(f"[LongTerm] best ValObj = {best_val_obj:.4e}")
    longterm_net.load_model(path=LONGTERM_CKPT_PATH, verbose=True)


# ================================
# Plot long-term objective curve
# ================================
def plot_longterm_objective_curve():
    curve_path = LONGTERM_CURVE_PATH

    curves = np.load(curve_path)

    # 跳過 epoch 1，避免初期 objective 過低壓縮後期曲線
    start_idx = 1

    epochs = np.arange(start_idx + 1, curves.shape[0] + 1)

    train_obj = curves[start_idx:, 0]
    val_obj   = curves[start_idx:, 1]

    window = PLOT_MOVING_AVG_WINDOW

    train_obj_ma = moving_average(train_obj, window)
    val_obj_ma   = moving_average(val_obj, window)

    ma_epochs = epochs[window - 1:]

    fig_path = LONGTERM_CURVE_FIG_PATH

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

    if args.plot:
        plot_longterm_objective_curve()
        raise SystemExit

    print("[INFO] 載入固定 datasets ...")

    train_dataset = load_longterm_dataset(TRAIN_DATASET_PATH,"train")

    val_dataset = load_longterm_dataset(VAL_DATASET_PATH,"val")

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