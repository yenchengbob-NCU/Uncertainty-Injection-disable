# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet

from main_st import (
    load_shortterm_dataset,
    get_fixed_theta_from_longterm,
    extract_shortterm_batch,
    sample_channel_ids,
    forward_shortterm_robust_objective,
    validate_shortterm_robust_sampled,
)


# ============================================================
# Robust retraining config
# ============================================================
RETRAIN_EPOCHS = 200
RETRAIN_LR = 2e-4

# validation 用比較多 layout，讓 retrain 選 ckpt 更穩
RETRAIN_VAL_LAYOUTS_PER_EPOCH = 100
RETRAIN_VAL_CHANNELS_PER_LAYOUT = 1500

# gradient clipping，避免 VaR objective 大幅震盪造成更新過猛
GRAD_CLIP_NORM = 1.0

# LR scheduler：若 validation objective 長時間沒有改善，降低 LR
LR_PATIENCE = 25
LR_FACTOR = 0.5
LR_MIN = 5e-5


# ============================================================
# Robust retraining
# ============================================================
def retrain_shortterm_robust(
    longterm_net,
    comm_net,
    radar_net,
    train_dataset,
    val_dataset,
):
    """
    只針對 short-term robust networks 進行 fine-tune。

    特點：
        1. 從既有 robust checkpoint 開始
        2. 使用較小 learning rate
        3. 使用 gradient clipping
        4. 使用較穩定的 validation 設定
        5. 若 validation objective 改善，覆蓋儲存 robust ckpt
    """

    optimizer = optim.Adam(
        list(comm_net.parameters()) + list(radar_net.parameters()),
        lr=RETRAIN_LR
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=LR_MIN
    )

    curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_retrain_curves_{SETTING_STRING}.npy"
    )

    curves = []

    longterm_net.eval()
    for p in longterm_net.parameters():
        p.requires_grad_(False)

    n_train_layouts = train_dataset["ue_layouts"].shape[0]

    print("[INFO] Initial validation before retraining ...")
    init_val_logs = validate_shortterm_robust_sampled(
        longterm_net=longterm_net,
        comm_net=comm_net,
        radar_net=radar_net,
        val_dataset=val_dataset,
        num_val_layouts=RETRAIN_VAL_LAYOUTS_PER_EPOCH,
        channels_per_layout=RETRAIN_VAL_CHANNELS_PER_LAYOUT,
    )

    best_val_obj = init_val_logs["objective"]

    print(
        f"[Initial ROB] "
        f"ValObj={init_val_logs['objective']:.4e} | "
        f"ValSumRate={init_val_logs['sum_rate_mean']:.4e} | "
        f"ValVaR-SNR(dB)={init_val_logs['sense_var_snr_mean_db']:.3f}"
    )

    print(
        f"[INFO] Start ROB retraining: "
        f"epochs={RETRAIN_EPOCHS}, lr={RETRAIN_LR:.2e}, "
        f"grad_clip={GRAD_CLIP_NORM}, "
        f"val_layouts={RETRAIN_VAL_LAYOUTS_PER_EPOCH}, "
        f"val_channels={RETRAIN_VAL_CHANNELS_PER_LAYOUT}"
    )

    for ep in trange(1, RETRAIN_EPOCHS + 1, desc="Retrain-ROB"):
        comm_net.train()
        radar_net.train()

        train_obj_ep = 0.0
        train_sumrate_ep = 0.0
        train_snr_db_ep = 0.0
        train_tx_pen_ep = 0.0
        train_snr_pen_ep = 0.0

        for _ in range(MINIBATCHES):
            layout_id = int(np.random.randint(0, n_train_layouts))

            ue_layout = train_dataset["ue_layouts"][layout_id]
            theta_fixed = get_fixed_theta_from_longterm(longterm_net, ue_layout)

            n_pool = train_dataset["st_h_dk_hat"][layout_id].shape[0]
            channel_ids = sample_channel_ids(n_pool, BATCH_SIZE)

            batch_data = extract_shortterm_batch(
                dataset=train_dataset,
                layout_id=layout_id,
                channel_ids=channel_ids,
            )

            optimizer.zero_grad(set_to_none=True)

            obj, logs = forward_shortterm_robust_objective(
                comm_net=comm_net,
                radar_net=radar_net,
                theta_fixed=theta_fixed,
                batch_data=batch_data,
                injection_samples=INJECTION_SAMPLES,
            )

            loss = -obj
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(comm_net.parameters()) + list(radar_net.parameters()),
                max_norm=GRAD_CLIP_NORM
            )

            optimizer.step()

            train_obj_ep += float(obj.detach().cpu()) / MINIBATCHES
            train_sumrate_ep += float(logs["sum_rate_mean"].cpu()) / MINIBATCHES
            train_snr_db_ep += float(logs["sense_var_snr_mean_db"].cpu()) / MINIBATCHES
            train_tx_pen_ep += float(logs["tx_penalty_mean"].cpu()) / MINIBATCHES
            train_snr_pen_ep += float(logs["snr_penalty_mean"].cpu()) / MINIBATCHES

        val_logs = validate_shortterm_robust_sampled(
            longterm_net=longterm_net,
            comm_net=comm_net,
            radar_net=radar_net,
            val_dataset=val_dataset,
            num_val_layouts=RETRAIN_VAL_LAYOUTS_PER_EPOCH,
            channels_per_layout=RETRAIN_VAL_CHANNELS_PER_LAYOUT,
        )

        val_obj = val_logs["objective"]
        scheduler.step(val_obj)

        current_lr = optimizer.param_groups[0]["lr"]

        curves.append([
            train_obj_ep,
            val_logs["objective"],
            train_sumrate_ep,
            val_logs["sum_rate_mean"],
            train_snr_db_ep,
            val_logs["sense_var_snr_mean_db"],
            train_snr_pen_ep,
            train_tx_pen_ep,
            current_lr,
        ])

        np.save(curve_path, np.array(curves, dtype=np.float32))

        print(
            f"[Retrain-ROB Epoch {ep:03d}] "
            f"LR={current_lr:.2e} | "
            f"TrainObj={train_obj_ep:.4e} | ValObj={val_logs['objective']:.4e} | "
            f"TrainSumRate={train_sumrate_ep:.4e} | ValSumRate={val_logs['sum_rate_mean']:.4e} | "
            f"TrainVaR-SNR(dB)={train_snr_db_ep:.3f} | "
            f"ValVaR-SNR(dB)={val_logs['sense_var_snr_mean_db']:.3f} | "
            f"TrainSNRPen={train_snr_pen_ep:.4e} | "
            f"TrainTxPen={train_tx_pen_ep:.4e}"
        )

        if val_obj > best_val_obj:
            best_val_obj = val_obj
            comm_net.save_model(verbose=False)
            radar_net.save_model(verbose=False)
            print(f"[SAVE] New best ROB ckpt, ValObj={best_val_obj:.4e}")

    print(f"[Retrain-ROB] best ValObj = {best_val_obj:.4e}")
    comm_net.load_model(verbose=True)
    radar_net.load_model(verbose=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("[INFO] 載入 fixed short-term datasets ...")

    train_dataset = load_shortterm_dataset(TRAIN_DATASET_PATH, "train")
    val_dataset = load_shortterm_dataset(VAL_DATASET_PATH, "val")

    print("[INFO] 載入 long-term checkpoint ...")
    longterm_net = LongTermPositionNet(ckpt_kind="longterm").to(DEVICE)

    if not longterm_net.model_path or not os.path.exists(longterm_net.model_path):
        raise FileNotFoundError(
            "找不到 long-term checkpoint。\n"
            "請先執行 main_lt.py 訓練 long-term 網路。"
        )

    longterm_net.load_model(verbose=True)

    print("[INFO] 載入 short-term robust checkpoints ...")
    short_comm_rob = ShortTermCommNet(ckpt_kind="short_comm_robust").to(DEVICE)
    short_radar_rob = ShortTermRadarNet(ckpt_kind="short_radar_robust").to(DEVICE)

    if not short_comm_rob.model_path or not os.path.exists(short_comm_rob.model_path):
        raise FileNotFoundError(
            f"找不到 short-term robust comm checkpoint：{short_comm_rob.model_path}"
        )

    if not short_radar_rob.model_path or not os.path.exists(short_radar_rob.model_path):
        raise FileNotFoundError(
            f"找不到 short-term robust radar checkpoint：{short_radar_rob.model_path}"
        )

    short_comm_rob.load_model(verbose=True)
    short_radar_rob.load_model(verbose=True)

    retrain_shortterm_robust(
        longterm_net=longterm_net,
        comm_net=short_comm_rob,
        radar_net=short_radar_rob,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    print("[INFO] Robust retraining finished.")