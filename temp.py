# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ================================
# Helpers
# ================================
def format_float_for_path(value: float) -> str:
    """
    將 float 轉成適合資料夾名稱的字串。

    例:
        0.125 -> 0p125
        0.5   -> 0p5
        1.0   -> 1p0
        200.0 -> 200
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")
    return text


def build_reg_paths(penalty: float):
    """
    建立 ST REG run 的輸出路徑。
    """
    penalty_tag = format_float_for_path(penalty)

    run_name = f"REG_penalty_{penalty_tag}"
    run_dir = os.path.join(ST_SWEEP_DIR, run_name)

    ckpt_dir = os.path.join(run_dir, "ckpt")
    curve_dir = os.path.join(run_dir, "training_curves")
    state_dir = os.path.join(run_dir, "train_state")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "curve_dir": curve_dir,
        "state_dir": state_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, "short_comm.ckpt"),
        "radar_ckpt_path": os.path.join(ckpt_dir, "short_radar.ckpt"),
        "curve_path": os.path.join(curve_dir, "shortterm_regular_curves.npy"),
        "curve_fig_path": os.path.join(curve_dir, "shortterm_regular_objective_curve.jpg"),
        "state_path": os.path.join(state_dir, "shortterm_regular_train_state.pt"),
        "config_path": os.path.join(run_dir, "config.json"),
    }


def save_run_config(
    paths: dict,
    penalty: float,
    train_dataset_path: str,
    val_dataset_path: str,
    longterm_ckpt_path: str,
):
    """
    儲存本次 REG run 設定，方便後續確認 checkpoint / curve 來源。
    """
    config = {
        "mode": "reg",
        "sensing_penalty": float(penalty),
        "validation_type": "full_val_layouts_full_est_channels",
        "noise_power": float(NOISE_POWER),
        "tx_power": float(TRANSMIT_POWER_TOTAL),
        "sensing_snr_threshold_db": float(SENSING_SNR_THRESHOLD_dB),
        "sensing_snr_threshold_linear": float(SENSING_SNR_THRESHOLD),
        "train_dataset_path": train_dataset_path,
        "val_dataset_path": val_dataset_path,
        "longterm_ckpt_path": longterm_ckpt_path,
        "comm_ckpt_path": paths["comm_ckpt_path"],
        "radar_ckpt_path": paths["radar_ckpt_path"],
        "curve_path": paths["curve_path"],
        "state_path": paths["state_path"],
    }

    with open(paths["config_path"], "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"[CONFIG] 已儲存 run config：{paths['config_path']}")


def load_shortterm_dataset(npz_path: str, split_name: str):
    """
    載入 short-term training / validation dataset。
    """
    required_keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "pl_BS_TAR_BS",
        "st_h_dk_hat",
        "st_h_rk_hat",
        "st_G_hat",
        "st_g_dt_hat",
    ]

    with np.load(npz_path) as data:
        dataset = {key: data[key] for key in required_keys}

    print(f"[{split_name}] loaded: {npz_path}")
    print(f"[{split_name}] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[{split_name}] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[{split_name}] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[{split_name}] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[{split_name}] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")

    total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))
    print(f"[{split_name}] RAM usage ≈ {total_bytes / (1024**3):.2f} GiB")

    return dataset


def sample_channel_ids(n_pool: int, batch_size: int) -> np.ndarray:
    """
    training 用：從單一 layout 的 estimated channel pool 中抽出一批 channels。
    """
    return np.random.choice(
        n_pool,
        size=batch_size,
        replace=False
    )


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout,
):
    """
    給一組 UE layout，從 long-term net 取得固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        theta_lt, _, _ = longterm_net(ue_layout)

    return theta_lt.detach()


def extract_shortterm_batch(
    dataset,
    layout_id: int,
    channel_ids: np.ndarray,
):
    """
    取出指定 layout 與 channel ids 的 estimated channels，
    並轉成 torch.complex64 準備輸入 ST networks。

    注意：
        這裡一次取 1 個 layout + 多個 estimated channels。
    """
    return {
        "pl_BS_UE": dataset["pl_BS_UE"][layout_id],
        "pl_BS_RIS_UE": dataset["pl_BS_RIS_UE"][layout_id],
        "pl_BS_TAR_BS": dataset["pl_BS_TAR_BS"][layout_id],

        "h_dk_hat": torch.as_tensor(
            dataset["st_h_dk_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "h_rk_hat": torch.as_tensor(
            dataset["st_h_rk_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "G_hat": torch.as_tensor(
            dataset["st_G_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
        "g_dt_hat": torch.as_tensor(
            dataset["st_g_dt_hat"][layout_id][channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        ),
    }


def moving_average(x: np.ndarray, window: int):
    """
    計算 moving average。
    """
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def save_training_state(
    state_path: str,
    comm_net,
    radar_net,
    optimizer,
    best_val_obj: float,
    finished_epoch: int,
):
    """
    儲存續訓用 training state。
    """
    torch.save(
        {
            "comm_model_state_dict": comm_net.state_dict(),
            "radar_model_state_dict": radar_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_obj": best_val_obj,
            "finished_epoch": finished_epoch,
        },
        state_path
    )


def load_training_state(
    state_path: str,
    comm_net,
    radar_net,
    optimizer,
):
    """
    載入續訓用 training state。
    """
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"找不到 training state，無法 resume：{state_path}"
        )

    checkpoint = torch.load(
        state_path,
        map_location=DEVICE
    )

    comm_net.load_state_dict(checkpoint["comm_model_state_dict"])
    radar_net.load_state_dict(checkpoint["radar_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_obj = float(checkpoint["best_val_obj"])
    finished_epoch = int(checkpoint["finished_epoch"])

    return best_val_obj, finished_epoch


# ================================
# Short-term regular objective
# ================================
def forward_shortterm_regular_objective(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
    sensing_loss_weight: float,
):
    """
    計算 REG short-term objective。

    流程:
        1. fixed theta_LT
        2. estimated channels -> W_C, W_R
        3. W_C, W_R 經過 TX power normalization
        4. 在 estimated channels 上計算 sum-rate 與 sensing SNR penalty
    """
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B = h_dk_hat.shape[0]
    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    W_C = comm_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)
    W_R = radar_net(h_dk_hat, h_rk_hat, G_hat, g_dt_hat, theta_batch)

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)

    sumrate_mean = comm_net.compute_sum_rate(
        h_dk=h_dk_hat,
        h_rk=h_rk_hat,
        G=G_hat,
        theta=theta_batch,
        W_R=W_R,
        W_C=W_C,
        pl_BS_UE=pl_BS_UE,
        pl_BS_RIS_UE=pl_BS_RIS_UE,
    ).mean()

    sense_snr = comm_net.compute_sense_snr(
        g_dt=g_dt_hat,
        W_R=W_R,
        W_C=W_C,
        pl_BS_TAR_BS=pl_BS_TAR_BS,
    ).real

    snr_penalty = torch.clamp(
        SENSING_SNR_THRESHOLD - sense_snr,
        min=0.0
    ).mean()

    objective = sumrate_mean - sensing_loss_weight * snr_penalty

    logs = {
        "sum_rate_mean": sumrate_mean.detach(),
        "sense_snr_mean_db": (
            10.0 * torch.log10(sense_snr.clamp_min(1e-12))
        ).mean().detach(),
        "snr_penalty_mean": snr_penalty.detach(),
        "objective": objective.detach(),
    }

    return objective, logs


# ================================
# Short-term regular validation
# ================================
@torch.no_grad()
def validate_shortterm_regular(
    longterm_net: LongTermPositionNet,
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    val_dataset,
    sensing_loss_weight: float,
):
    """
    使用全部 validation layouts，
    並且每個 layout 使用全部 ST estimated channels。
    """
    longterm_net.eval()
    comm_net.eval()
    radar_net.eval()

    n_val_layouts = val_dataset["ue_layouts"].shape[0]

    total_obj = 0.0
    total_sumrate = 0.0
    total_snr_db = 0.0
    total_snr_penalty = 0.0

    for layout_id in range(n_val_layouts):
        ue_layout = val_dataset["ue_layouts"][layout_id]
        theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

        n_channels = val_dataset["st_h_dk_hat"][layout_id].shape[0]
        channel_ids = np.arange(n_channels)

        batch_data = extract_shortterm_batch(
            val_dataset,
            layout_id,
            channel_ids
        )

        obj, logs = forward_shortterm_regular_objective(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            sensing_loss_weight=sensing_loss_weight,
        )

        total_obj += float(obj.detach().cpu())
        total_sumrate += float(logs["sum_rate_mean"].cpu())
        total_snr_db += float(logs["sense_snr_mean_db"].cpu())
        total_snr_penalty += float(logs["snr_penalty_mean"].cpu())

    return {
        "objective": total_obj / n_val_layouts,
        "sum_rate_mean": total_sumrate / n_val_layouts,
        "sense_snr_mean_db": total_snr_db / n_val_layouts,
        "snr_penalty_mean": total_snr_penalty / n_val_layouts,
        "num_val_layouts": n_val_layouts,
        "channels_per_layout": val_dataset["st_h_dk_hat"].shape[1],
    }


# ================================
# Train regular
# ================================
def train_shortterm_regular(
    longterm_net,
    comm_net,
    radar_net,
    train_dataset,
    val_dataset,
    paths: dict,
    sensing_loss_weight: float,
    resume: bool = False,
):
    """
    訓練 short-term REG networks。

    training:
        每個 minibatch 隨機抽 ST_BATCH_LAYOUTS 個 layouts。
        每個 layout 隨機抽 ST_BATCH_EST_CHANNELS_PER_LAYOUT 筆 estimated channels。

    validation:
        使用全部 validation layouts。
        每個 layout 使用全部 estimated channels。
    """
    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=REG_LEARNING_RATE
    )

    curve_path = paths["curve_path"]
    state_path = paths["state_path"]

    if resume:
        print("[INFO] Resume short-term regular training state ...")
        best_val_obj, finished_epoch = load_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
        )

        curves = np.load(curve_path).tolist()
        start_epoch = finished_epoch + 1
        end_epoch = finished_epoch + REG_EPOCHS
    else:
        best_val_obj = -np.inf
        curves = []
        start_epoch = 1
        end_epoch = REG_EPOCHS

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    for ep in trange(start_epoch, end_epoch + 1, desc="ShortTerm-Regular"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0
        train_snr_penalty_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_ids = np.random.choice(
                n_train_layouts,
                size=ST_BATCH_LAYOUTS,
                replace=False
            )

            optimizer.zero_grad(set_to_none=True)

            mb_obj = 0.0
            mb_sumrate = 0.0
            mb_snr_db = 0.0
            mb_snr_penalty = 0.0

            for layout_id in layout_ids:
                layout_id = int(layout_id)

                ue_layout = train_dataset["ue_layouts"][layout_id]
                theta_fixed = get_fixed_theta_from_longterm(
                    longterm_net,
                    ue_layout
                )

                n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]
                channel_ids = sample_channel_ids(
                    n_pool,
                    ST_BATCH_EST_CHANNELS_PER_LAYOUT
                )

                batch_data = extract_shortterm_batch(
                    train_dataset,
                    layout_id,
                    channel_ids
                )

                obj, logs = forward_shortterm_regular_objective(
                    comm_net=comm_net,
                    radar_net=radar_net,
                    theta_fixed=theta_fixed,
                    batch_data=batch_data,
                    sensing_loss_weight=sensing_loss_weight,
                )

                loss = -obj / ST_BATCH_LAYOUTS
                loss.backward()

                mb_obj += float(obj.detach().cpu()) / ST_BATCH_LAYOUTS
                mb_sumrate += float(logs["sum_rate_mean"].cpu()) / ST_BATCH_LAYOUTS
                mb_snr_db += float(logs["sense_snr_mean_db"].cpu()) / ST_BATCH_LAYOUTS
                mb_snr_penalty += float(logs["snr_penalty_mean"].cpu()) / ST_BATCH_LAYOUTS

            optimizer.step()

            train_obj_ep += mb_obj / MINIBATCHES
            train_sumrate_ep += mb_sumrate / MINIBATCHES
            train_snr_db_ep += mb_snr_db / MINIBATCHES
            train_snr_penalty_ep += mb_snr_penalty / MINIBATCHES

        val_logs = validate_shortterm_regular(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
            sensing_loss_weight=sensing_loss_weight,
        )

        curves.append([
            train_obj_ep,
            val_logs["objective"],
            train_sumrate_ep,
            val_logs["sum_rate_mean"],
            train_snr_db_ep,
            val_logs["sense_snr_mean_db"],
            train_snr_penalty_ep,
            val_logs["snr_penalty_mean"],
        ])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[ST-REG Epoch {ep:03d}/{end_epoch}]\n"
            f"  TrainObj      = {train_obj_ep: .4e} | "
            f"TrainSumRate  = {train_sumrate_ep: .4e} | "
            f"TrainSNR(dB)  = {train_snr_db_ep: .3f} | "
            f"TrainSNRPen   = {train_snr_penalty_ep: .4e} |\n"
            f"  ValObj        = {val_logs['objective']: .4e} | "
            f"ValSumRate    = {val_logs['sum_rate_mean']: .4e} | "
            f"ValSNR(dB)    = {val_logs['sense_snr_mean_db']: .3f} | "
            f"ValSNRPen     = {val_logs['snr_penalty_mean']: .4e} |"
        )

        if val_logs["objective"] > best_val_obj:
            best_val_obj = val_logs["objective"]
            comm_net.save_model(
                path=paths["comm_ckpt_path"],
                verbose=False
            )
            radar_net.save_model(
                path=paths["radar_ckpt_path"],
                verbose=False
            )

        save_training_state(
            state_path=state_path,
            comm_net=comm_net,
            radar_net=radar_net,
            optimizer=optimizer,
            best_val_obj=best_val_obj,
            finished_epoch=ep,
        )

    print(f"[Short-Reg] best ValObj = {best_val_obj:.4e}")

    comm_net.load_model(
        path=paths["comm_ckpt_path"],
        verbose=True
    )
    radar_net.load_model(
        path=paths["radar_ckpt_path"],
        verbose=True
    )


# ================================
# Plot short-term objective curve
# ================================
def plot_shortterm_objective_curve(paths: dict):
    """
    讀取 REG curve 並畫 objective curve。
    """
    curve_path = paths["curve_path"]

    if not os.path.exists(curve_path):
        raise FileNotFoundError(f"找不到 curve 檔案：{curve_path}")

    curves = np.load(curve_path)

    start_idx = 3
    window = PLOT_MOVING_AVG_WINDOW

    if curves.shape[0] < window:
        window = max(1, curves.shape[0])

    epochs = np.arange(start_idx + 1, curves.shape[0] + 1)

    train_obj = curves[start_idx:, 0]
    val_obj = curves[start_idx:, 1]

    train_obj_ma = moving_average(train_obj, window)
    val_obj_ma = moving_average(val_obj, window)

    ma_epochs = epochs[window - 1:]

    fig_path = paths["curve_fig_path"]

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        epochs,
        train_obj,
        label="REG Train Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        epochs,
        val_obj,
        label="REG Validation Objective",
        alpha=0.35,
        linewidth=1.0
    )

    plt.plot(
        ma_epochs,
        train_obj_ma,
        label=f"REG Train Objective MA({window})",
        linewidth=2.2
    )

    plt.plot(
        ma_epochs,
        val_obj_ma,
        label=f"REG Validation Objective MA({window})",
        linewidth=2.2
    )

    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title("Short-term Regular Training / Validation Objective")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    print(f"[PLOT] 已儲存 short-term objective curve：{fig_path}")

    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train short-term REG model with full validation")

    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="REG sensing loss weight；不輸入則使用 settings.py 的 REG_SENSING_LOSS_WEIGHT"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="從目前 REG penalty 的 train_state 續訓"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="只讀取目前 REG penalty 的 curve .npy 並畫圖，不進行訓練"
    )

    args = parser.parse_args()

    penalty = (
        float(REG_SENSING_LOSS_WEIGHT)
        if args.penalty is None
        else float(args.penalty)
    )

    paths = build_reg_paths(penalty=penalty)

    if args.plot:
        print("\n[INFO] Plot short-term REG curve:")
        print(f"[INFO] penalty = {penalty}")
        print(f"[INFO] curve_path = {paths['curve_path']}")

        plot_shortterm_objective_curve(paths=paths)
        raise SystemExit

    train_dataset_path = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path = os.path.join(DATA_DIR, "dataset_val.npz")
    longterm_ckpt_path = os.path.join(LT_CKPT_DIR, "longterm.ckpt")

    save_run_config(
        paths=paths,
        penalty=penalty,
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        longterm_ckpt_path=longterm_ckpt_path,
    )

    print("\n[INFO] Short-term REG run setting:")
    print(f"[INFO] penalty = {penalty}")
    print(f"[INFO] REG_EPOCHS = {REG_EPOCHS}")
    print(f"[INFO] REG_LEARNING_RATE = {REG_LEARNING_RATE}")
    print(f"[INFO] MINIBATCHES = {MINIBATCHES}")
    print(f"[INFO] ST_BATCH_LAYOUTS = {ST_BATCH_LAYOUTS}")
    print(f"[INFO] ST_BATCH_EST_CHANNELS_PER_LAYOUT = {ST_BATCH_EST_CHANNELS_PER_LAYOUT}")
    print(f"[INFO] run_dir = {paths['run_dir']}")
    print(f"[INFO] comm_ckpt_path = {paths['comm_ckpt_path']}")
    print(f"[INFO] radar_ckpt_path = {paths['radar_ckpt_path']}")
    print(f"[INFO] curve_path = {paths['curve_path']}")
    print(f"[INFO] state_path = {paths['state_path']}")
    print(f"[INFO] resume = {args.resume}")

    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path = {val_dataset_path}")

    train_dataset = load_shortterm_dataset(train_dataset_path, "train")
    val_dataset = load_shortterm_dataset(val_dataset_path, "val")

    print("\n[INFO] 載入 shared long-term model ...")
    print(f"[INFO] longterm_ckpt_path = {longterm_ckpt_path}")

    longterm_net = LongTermPositionNet(ckpt_kind=None).to(DEVICE)
    longterm_net.load_model(path=longterm_ckpt_path, verbose=True)

    short_comm_reg = ShortTermCommNet(ckpt_kind=None).to(DEVICE)
    short_radar_reg = ShortTermRadarNet(ckpt_kind=None).to(DEVICE)

    print("\n[INFO] 開始訓練 short-term regular ...")

    train_shortterm_regular(
        longterm_net=longterm_net,
        comm_net=short_comm_reg,
        radar_net=short_radar_reg,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        paths=paths,
        sensing_loss_weight=penalty,
        resume=args.resume,
    )

    print("[INFO] Short-term regular training finished.")