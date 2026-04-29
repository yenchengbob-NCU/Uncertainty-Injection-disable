# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet


# ============================================================
# Long-term training config
# ============================================================
LONGTERM_BATCH_LAYOUTS = 32


# ============================================================
# Helpers
# ============================================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def load_longterm_dataset(npz_path: str, split_name: str):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"[{split_name}] 找不到 dataset 檔案：{npz_path}\n"
            f"請先執行 rician.py 生成固定 dataset。"
        )

    required_keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "pl_BS_TAR_BS",
        "lt_h_dk_true",
        "lt_h_rk_true",
        "lt_G_true",
        "lt_g_dt_true",
    ]

    with np.load(npz_path) as data:
        for k in required_keys:
            if k not in data:
                raise KeyError(f"[{split_name}] dataset 缺少欄位：{k}")

        dataset = {k: data[k] for k in required_keys}

    print(f"[{split_name}] loaded: {npz_path}")
    print(f"[{split_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{split_name}] lt_h_dk_true shape = {dataset['lt_h_dk_true'].shape}")
    return dataset


def sample_train_layout_ids(n_layouts: int, batch_size: int) -> np.ndarray:
    replace = n_layouts < batch_size
    return np.random.choice(n_layouts, size=batch_size, replace=replace)


# ============================================================
# Long-term objective for ONE minibatch of layouts
# ============================================================
def forward_longterm_batch_objective(
    longterm_net: LongTermPositionNet,
    dataset,
    layout_ids: np.ndarray,
):
    """
    一個 minibatch = 32 個 layouts
    對每個 layout:
        1. layout -> theta_LT, W_C_LT, W_R_LT
        2. 使用對應的 long-term true/statistical channels
        3. 算出 layout-level mean sum-rate
        4. 減去 theta penalty, tx penalty
    最後把 32 個 layout-level objectives 平均成 batch objective
    """
    # --------------------------------------------------------
    # 1) 這批 layouts 一次送進 long-term net
    # --------------------------------------------------------
    ue_layouts_np = dataset["ue_layouts"][layout_ids]     # (B,K,2)
    ue_layouts_t = np_to_torch_float(ue_layouts_np)       # (B,K,2)

    theta_lt, W_C_lt, W_R_lt = longterm_net(ue_layouts_t)  # (B,N), (B,M,K), (B,M,RADAR_STREAMS)

    B = len(layout_ids)

    obj_list = []
    sumrate_list = []
    theta_pen_list = []
    tx_pen_list = []

    # --------------------------------------------------------
    # 2) 對每個 layout 使用它自己的 true/statistical channels
    # --------------------------------------------------------
    for b, layout_id in enumerate(layout_ids):
        # 該 layout 的 true/statistical channels
        h_dk_true = np_to_torch_complex(dataset["lt_h_dk_true"][layout_id])   # (S,M,K)
        h_rk_true = np_to_torch_complex(dataset["lt_h_rk_true"][layout_id])   # (S,N,K)
        G_true    = np_to_torch_complex(dataset["lt_G_true"][layout_id])      # (S,N,M)
        g_dt_true = np_to_torch_complex(dataset["lt_g_dt_true"][layout_id])   # (S,M,RADAR_STREAMS)

        # 該 layout 的 pathloss
        pl_BS_UE     = dataset["pl_BS_UE"][layout_id]         # (K,)
        pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]     # (K,)

        S = h_dk_true.shape[0]

        # 複製這個 layout 的輸出到 S 組通道上
        theta_b = theta_lt[b].unsqueeze(0).expand(S, RIS_UNIT)               # (S,N)
        W_C_b   = W_C_lt[b].unsqueeze(0).expand(S, TX_ANT, UAV_COMM)         # (S,M,K)
        W_R_b   = W_R_lt[b].unsqueeze(0).expand(S, TX_ANT, RADAR_STREAMS)    # (S,M,RADAR_STREAMS)

        # 64 / 128 組 sum-rate -> layout-level mean sum-rate
        sumrate_mean_b = longterm_net.compute_sum_rate(
            h_dk=h_dk_true,
            h_rk=h_rk_true,
            G=G_true,
            theta=theta_b,
            W_R=W_R_b,
            W_C=W_C_b,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE,
        ).mean()

        # theta penalty
        theta_pen_b = longterm_net.compute_ris_amplitude_penalty(
            theta_lt[b].unsqueeze(0)
        ).mean()

        # tx penalty
        tx_power_b = longterm_net.compute_tx_power(
            W_C_lt[b].unsqueeze(0),
            W_R_lt[b].unsqueeze(0)
        )  # (1,)
        tx_pen_b = torch.clamp(tx_power_b - TRANSMIT_POWER_TOTAL, min=0.0).mean()

        # layout-level objective
        obj_b = (
            sumrate_mean_b
            - RE_POWER_LOSS_WEIGHT * theta_pen_b
            - TX_POWER_LOSS_WEIGHT * tx_pen_b
        )

        obj_list.append(obj_b)
        sumrate_list.append(sumrate_mean_b.detach())
        theta_pen_list.append(theta_pen_b.detach())
        tx_pen_list.append(tx_pen_b.detach())

    # --------------------------------------------------------
    # 3) 平均 32 個 layout-level objectives
    # --------------------------------------------------------
    batch_objective = torch.stack(obj_list).mean()

    logs = {
        "sum_rate_mean": torch.stack(sumrate_list).mean(),
        "theta_penalty_mean": torch.stack(theta_pen_list).mean(),
        "tx_penalty_mean": torch.stack(tx_pen_list).mean(),
        "objective": batch_objective.detach(),
        "num_layouts": B,
    }
    return batch_objective, logs


# ============================================================
# Validation over ALL val layouts
# ============================================================
@torch.no_grad()
def validate_longterm_all(
    longterm_net: LongTermPositionNet,
    val_dataset,
):
    """
    直接掃完整個 val split
    每個 val layout 各自算一次 layout-level objective
    最後平均成 val objective
    """
    longterm_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0]

    total_obj = 0.0
    total_sumrate = 0.0
    total_theta_pen = 0.0
    total_tx_pen = 0.0

    for start in range(0, n_val_layouts, LONGTERM_BATCH_LAYOUTS):
        end = min(start + LONGTERM_BATCH_LAYOUTS, n_val_layouts)
        layout_ids = np.arange(start, end)

        obj_t, logs = forward_longterm_batch_objective(
            longterm_net=longterm_net,
            dataset=val_dataset,
            layout_ids=layout_ids,
        )

        n_chunk = logs["num_layouts"]
        total_obj += float(obj_t.detach().cpu()) * n_chunk
        total_sumrate += float(logs["sum_rate_mean"].cpu()) * n_chunk
        total_theta_pen += float(logs["theta_penalty_mean"].cpu()) * n_chunk
        total_tx_pen += float(logs["tx_penalty_mean"].cpu()) * n_chunk

    val_obj = total_obj / n_val_layouts
    val_sumrate = total_sumrate / n_val_layouts
    val_theta_pen = total_theta_pen / n_val_layouts
    val_tx_pen = total_tx_pen / n_val_layouts

    return {
        "objective": val_obj,
        "sum_rate_mean": val_sumrate,
        "theta_penalty_mean": val_theta_pen,
        "tx_penalty_mean": val_tx_pen,
    }


# ============================================================
# Train long-term
# ============================================================
def train_longterm(longterm_net, train_dataset, val_dataset):
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

        # ----------------------------------------------------
        # 每個 epoch：50 個 minibatches
        # 每個 minibatch：抽 32 個 train layouts
        # ----------------------------------------------------
        for _ in range(MINIBATCHES):
            layout_ids = sample_train_layout_ids(
                n_layouts=n_train_layouts,
                batch_size=LONGTERM_BATCH_LAYOUTS
            )

            optimizer.zero_grad(set_to_none=True)

            obj, logs = forward_longterm_batch_objective(
                longterm_net=longterm_net,
                dataset=train_dataset,
                layout_ids=layout_ids,
            )

            (-obj).backward()
            optimizer.step()

            train_obj_ep       += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep   += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_theta_pen_ep += float(logs["theta_penalty_mean"].cpu()) / MINIBATCHES
            train_tx_pen_ep    += float(logs["tx_penalty_mean"].cpu()) / MINIBATCHES

        # ----------------------------------------------------
        # validation：掃過全部 val layouts 一次
        # ----------------------------------------------------
        val_logs = validate_longterm_all(
            longterm_net=longterm_net,
            val_dataset=val_dataset,
        )

        val_obj_ep       = val_logs["objective"]
        val_sumrate_ep   = val_logs["sum_rate_mean"]
        val_theta_pen_ep = val_logs["theta_penalty_mean"]
        val_tx_pen_ep    = val_logs["tx_penalty_mean"]

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
            f"TrainTxPen={train_tx_pen_ep:.4e} | ValTxPen={val_tx_pen_ep:.4e}"
        )

        if val_obj_ep > best_val_obj:
            best_val_obj = val_obj_ep
            longterm_net.save_model(verbose=False)

    print(f"[LongTerm] best ValObj = {best_val_obj:.4e}")
    longterm_net.load_model(verbose=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train long-term network only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="若已存在 long-term ckpt，則先載入再繼續訓練"
    )
    args = parser.parse_args()

    print("[INFO] 載入固定 nested datasets ...")

    train_dataset = load_longterm_dataset(TRAIN_DATASET_PATH, "train")
    val_dataset   = load_longterm_dataset(VAL_DATASET_PATH, "val")

    longterm_net = LongTermPositionNet(ckpt_kind="longterm").to(DEVICE)

    if args.resume and longterm_net.model_path and os.path.exists(longterm_net.model_path):
        print(f"[INFO] resume from ckpt: {longterm_net.model_path}")
        longterm_net.load_model(verbose=True)

    print("[INFO] 開始 long-term 訓練 ...")
    train_longterm(
        longterm_net=longterm_net,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("[INFO] Long-term training finished.")