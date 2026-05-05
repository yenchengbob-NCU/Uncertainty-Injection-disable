# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet

# ================================
# Long-term validation config
# ================================
LT_VAL_LAYOUTS_PER_EPOCH = 40                  # 每個 epoch 從 validation pool 抽幾個 layouts 驗證


# ================================
# Helpers
# ================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def load_longterm_dataset(npz_path: str, dataset_name: str):
    """
    載入  long-term 訓練需要的 dataset 欄位。
    回傳: dataset : dict
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"[{dataset_name}] 找不到 dataset 檔案：{npz_path}\n"
            f"請先執行 rician.py 生成固定 dataset。"
        )

    required_keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "lt_h_dk_true",
        "lt_h_rk_true",
        "lt_G_true",
    ]

    with np.load(npz_path) as data:
        for key in required_keys:
            if key not in data:
                raise KeyError(f"[{dataset_name}] dataset 缺少欄位：{key}")

        dataset = {key: data[key] for key in required_keys}

    print(f"[{dataset_name}] loaded: {npz_path}")
    print(f"[{dataset_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{dataset_name}] lt_h_dk_true shape = {dataset['lt_h_dk_true'].shape}")
    print(f"[{dataset_name}] lt_h_rk_true shape = {dataset['lt_h_rk_true'].shape}")
    print(f"[{dataset_name}] lt_G_true shape    = {dataset['lt_G_true'].shape}")

    return dataset


# ================================
# Long-term objective
# ================================
def forward_longterm_one_layout_objective(
    longterm_net: LongTermPositionNet,
    dataset,
    layout_id: int,
):
    """
    計算單一 layout 的 long-term objective
    流程:
        1. layout -> theta_LT, W_C_LT, W_R_LT
        2. 使用該 layout 的 statistical channels
        3. 計算 layout-level mean sum-rate
        4. 扣除 RIS amplitude penalty 與 TX power penalty
    回傳:
        objective : torch.Tensor
        logs      : dict
    """

    ue_layout_np = dataset["ue_layouts"][layout_id]                    # shape = (K,2)
    ue_layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)         # shape = (1,K,2)

    theta_lt, W_C_lt, W_R_lt = longterm_net(ue_layout_t)               # (1,N), (1,M,K), (1,M,RADAR_STREAMS)

    h_dk_true = np_to_torch_complex(dataset["lt_h_dk_true"][layout_id])  # shape = (S,M,K)
    h_rk_true = np_to_torch_complex(dataset["lt_h_rk_true"][layout_id])  # shape = (S,N,K)
    G_true    = np_to_torch_complex(dataset["lt_G_true"][layout_id])     # shape = (S,N,M)

    pl_BS_UE     = dataset["pl_BS_UE"][layout_id]                      # shape = (K,)
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]                  # shape = (K,)

    S = h_dk_true.shape[0]                                             # statistical channels 數量

    theta_rep = theta_lt.expand(S, RIS_UNIT)                           # shape = (S,N)
    W_C_rep   = W_C_lt.expand(S, TX_ANT, UAV_COMM)                     # shape = (S,M,K)
    W_R_rep   = W_R_lt.expand(S, TX_ANT, RADAR_STREAMS)                # shape = (S,M,RADAR_STREAMS)

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

    tx_power = longterm_net.compute_tx_power(
        W_C_lt,
        W_R_lt
    )

    tx_penalty = torch.clamp(
        tx_power - TRANSMIT_POWER_TOTAL,
        min=0.0
    ).mean()

    objective = (
        sumrate_mean
        - RE_POWER_LOSS_WEIGHT * theta_penalty
        - TX_POWER_LOSS_WEIGHT * tx_penalty
    )

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "theta_penalty_mean": theta_penalty.detach(),
        "tx_penalty_mean": tx_penalty.detach(),
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
    num_val_layouts = min(num_val_layouts, n_val_pool)

    replace = n_val_pool < num_val_layouts                              # pool 不足時允許重複抽樣
    layout_ids = np.random.choice(n_val_pool, size=num_val_layouts, replace=replace)

    total_obj = 0.0
    total_sumrate = 0.0
    total_theta_pen = 0.0
    total_tx_pen = 0.0

    for layout_id in layout_ids:
        obj_t, logs = forward_longterm_one_layout_objective(
            longterm_net=longterm_net,
            dataset=val_dataset,
            layout_id=int(layout_id),
        )

        total_obj += float(obj_t.detach().cpu())
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_theta_pen += float(logs["theta_penalty_mean"].cpu())
        total_tx_pen += float(logs["tx_penalty_mean"].cpu())

    return {
        "objective": total_obj / num_val_layouts,
        "sum_rate_mean": total_sumrate / num_val_layouts,
        "theta_penalty_mean": total_theta_pen / num_val_layouts,
        "tx_penalty_mean": total_tx_pen / num_val_layouts,
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
    optimizer = optim.Adam(longterm_net.parameters(), lr=LEARNING_RATE)

    best_val_obj = -np.inf
    curves = []

    curve_path = os.path.join(CURVE_DIR, f"longterm_curves_{SETTING_STRING}.npy")

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(1, EPOCHS + 1, desc="LongTerm"):
        longterm_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_theta_pen_ep = 0.0
        train_tx_pen_ep = 0.0

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
            train_tx_pen_ep += float(logs["tx_penalty_mean"].cpu()) / MINIBATCHES

        val_logs = validate_longterm_sampled(
            longterm_net=longterm_net,
            val_dataset=val_dataset,
            num_val_layouts=LT_VAL_LAYOUTS_PER_EPOCH,
        )

        val_obj_ep = val_logs["objective"]
        val_sumrate_ep = val_logs["sum_rate_mean"]
        val_theta_pen_ep = val_logs["theta_penalty_mean"]
        val_tx_pen_ep = val_logs["tx_penalty_mean"]

        curves.append([
            train_obj_ep,
            val_obj_ep,
            train_sumrate_ep,
            val_sumrate_ep,
            train_theta_pen_ep,
            val_theta_pen_ep,
            train_tx_pen_ep,
            val_tx_pen_ep,
        ])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[LongTerm Epoch {ep:03d}] "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_obj_ep:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_sumrate_ep:.4e} | "
            f"TrainThetaPen={train_theta_pen_ep:.4e} | ValThetaPen={val_theta_pen_ep:.4e} | "
            f"TrainTxPen={train_tx_pen_ep:.4e} | ValTxPen={val_tx_pen_ep:.4e} | "
            f"ValLayouts={val_logs['num_val_layouts']}"
        )

        if val_obj_ep > best_val_obj:
            best_val_obj = val_obj_ep
            longterm_net.save_model(verbose=False)

    print(f"[LongTerm] best ValObj = {best_val_obj:.4e}")
    longterm_net.load_model(verbose=True)

# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train long-term network only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若已存在 long-term ckpt，則先載入再繼續訓練"
    )
    args = parser.parse_args()

    print("[INFO] 載入固定 datasets ...")

    train_dataset = load_longterm_dataset(TRAIN_DATASET_PATH, "train")
    val_dataset   = load_longterm_dataset(VAL_DATASET_PATH, "val")

    longterm_net = LongTermPositionNet(ckpt_kind="longterm").to(DEVICE)

    if args.resume and longterm_net.model_path and os.path.exists(longterm_net.model_path):
        print(f"[INFO] resume from ckpt: {longterm_net.model_path}")
        longterm_net.load_model(verbose=True)

    print("[INFO] 開始 long-term 訓練 ...")
    print("[INFO] Training rule: one minibatch = one layout objective.")

    train_longterm(
        longterm_net=longterm_net,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("[INFO] Long-term training finished.")