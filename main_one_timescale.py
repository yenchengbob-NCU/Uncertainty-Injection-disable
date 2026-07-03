# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from one_timescale_NN import CommNet, RadarNet, ThetaNet, TestNet
from baseline import make_rzf_beamformer , mrt_in_H_eff_H_nullspace,RZF_ALPHA_FACTOR

# ================================
# Helpers
# ================================
def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def moving_average(x, window):
    """
    Valid moving average.
    回傳:
        ma_epochs : 對應 moving average 的 epoch index
        y_ma      : moving average 後的曲線
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    window = int(window)

    if len(x) < window:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    kernel = np.ones(window, dtype=np.float64) / window
    y_ma = np.convolve(x, kernel, mode="valid")
    ma_epochs = np.arange(window, len(x) + 1)

    return ma_epochs, y_ma


def plot_train_val_curve(curves, train_key, val_key, ylabel, title, save_path, ma_window=20, scale=1.0):
    """
    畫單一 metric 的 train / validation curve。
    每個 metric 獨立存一張圖。
    """
    train_curve = np.asarray(curves[train_key], dtype=np.float64) * scale
    val_curve   = np.asarray(curves[val_key], dtype=np.float64) * scale

    raw_epochs = np.arange(1, len(train_curve) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        raw_epochs,
        train_curve,
        label="Train",
        alpha=0.35,
        linewidth=1.0,
    )

    plt.plot(
        raw_epochs,
        val_curve,
        label="Validation",
        alpha=0.35,
        linewidth=1.0,
    )

    train_ma_epochs, train_smooth = moving_average(train_curve, ma_window)
    val_ma_epochs, val_smooth = moving_average(val_curve, ma_window)

    if len(train_smooth) > 0:
        plt.plot(
            train_ma_epochs,
            train_smooth,
            label=f"Train MA({ma_window})",
            linewidth=2.5,
        )

    if len(val_smooth) > 0:
        plt.plot(
            val_ma_epochs,
            val_smooth,
            label=f"Validation MA({ma_window})",
            linewidth=2.5,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[PLOT] Saved: {save_path}")


def plot_one_timescale_reg_curves(curve_path, save_dir, ma_window=20):
    """
    讀取 one_timescale_reg_curves.npz，並輸出多張 training curves。
    """
    os.makedirs(save_dir, exist_ok=True)

    with np.load(curve_path) as data:
        curves = {key: data[key] for key in data.files}

    plot_train_val_curve(
        curves,
        train_key="train_objective",
        val_key="val_objective",
        ylabel="Objective",
        title="One-timescale REG Objective",
        save_path=os.path.join(save_dir, "reg_objective.png"),
        ma_window=ma_window,
    )

    plot_train_val_curve(
        curves,
        train_key="train_sumrate",
        val_key="val_sumrate",
        ylabel="Sum-rate (bps/Hz)",
        title="One-timescale REG Sum-rate",
        save_path=os.path.join(save_dir, "reg_sumrate.png"),
        ma_window=ma_window,
    )

    plot_train_val_curve(
        curves,
        train_key="train_comm_sinr_penalty",
        val_key="val_comm_sinr_penalty",
        ylabel="Penalty",
        title="One-timescale REG Communication SINR Penalty",
        save_path=os.path.join(save_dir, "reg_comm_sinr_penalty.png"),
        ma_window=ma_window,
    )

    plot_train_val_curve(
        curves,
        train_key="train_sensing_penalty",
        val_key="val_sensing_penalty",
        ylabel="Penalty",
        title="One-timescale REG Sensing Penalty",
        save_path=os.path.join(save_dir, "reg_sensing_penalty.png"),
        ma_window=ma_window,
    )

    plot_train_val_curve(
        curves,
        train_key="train_min_sinr_db",
        val_key="val_min_sinr_db",
        ylabel="Minimum UE SINR (dB)",
        title="One-timescale REG Minimum UE SINR",
        save_path=os.path.join(save_dir, "reg_min_sinr_db.png"),
        ma_window=ma_window,
    )

    plot_train_val_curve(
        curves,
        train_key="train_target_snr_db",
        val_key="val_target_snr_db",
        ylabel="Target SNR (dB)",
        title="One-timescale REG Target SNR",
        save_path=os.path.join(save_dir, "reg_target_snr_db.png"),
        ma_window=ma_window,
    )

    plot_train_val_curve(
        curves,
        train_key="train_sinr_outage",
        val_key="val_sinr_outage",
        ylabel="SINR outage (%)",
        title="One-timescale REG SINR Outage",
        save_path=os.path.join(save_dir, "reg_sinr_outage.png"),
        ma_window=ma_window,
        scale=100.0,
    )

    plot_train_val_curve(
        curves,
        train_key="train_target_snr_outage",
        val_key="val_target_snr_outage",
        ylabel="Target SNR outage (%)",
        title="One-timescale REG Target SNR Outage",
        save_path=os.path.join(save_dir, "reg_target_snr_outage.png"),
        ma_window=ma_window,
        scale=100.0,
    )

    print(f"[PLOT] All REG curves saved to: {save_dir}")


# ================================
# Main
# ================================
if __name__ == "__main__":

    # 這裡不使用net 只是要用neural_net.py的副函式
    physics_net = CommNet().to(DEVICE)
    physics_net.eval()

    # 讀取資料
    train_dataset_path  = os.path.join(DATA_DIR, "dataset_train.npz")
    val_dataset_path    = os.path.join(DATA_DIR, "dataset_val.npz")

    train_dataset = physics_net.load_channel_dataset(train_dataset_path, "train")
    val_dataset   = physics_net.load_channel_dataset(val_dataset_path, "val")
    
    print("\n[INFO] 載入固定 datasets ...")
    print(f"[INFO] train_dataset_path = {train_dataset_path}")
    print(f"[INFO] val_dataset_path   = {val_dataset_path}")

    # ================================
    # Train REG
    # ================================
    comm_net  = CommNet().to(DEVICE)
    radar_net = RadarNet().to(DEVICE)
    theta_net = ThetaNet().to(DEVICE)
    test_net  = TestNet().to(DEVICE)

    # optimizer = optim.Adam(list(comm_net.parameters())+ list(radar_net.parameters())+ list(theta_net.parameters()),lr=REG_LEARNING_RATE)
    # optimizer = optim.Adam(list(test_net.parameters())+ list(radar_net.parameters())+ list(theta_net.parameters()),lr=REG_LEARNING_RATE)

    optimizer   = optim.Adam(list(theta_net.parameters()),lr=REG_LEARNING_RATE)    

    # reg_comm_ckpt  = os.path.join(REG_CKPT_DIR, "one_timescale_comm_reg.ckpt")
    # reg_radar_ckpt = os.path.join(REG_CKPT_DIR, "one_timescale_radar_reg.ckpt")
    reg_theta_ckpt = os.path.join(REG_CKPT_DIR, "one_timescale_theta_reg.ckpt")
    # reg_test_ckpt  = os.path.join(REG_CKPT_DIR, "one_timescale_test_reg.ckpt")
    reg_curve_path = os.path.join(REG_CKPT_DIR, "one_timescale_reg_curves.npz")

    reg_curve_dir = os.path.join(RESULT_DIR, "reg_training_curves")
    reg_curve_fig_dir = os.path.join(reg_curve_dir, "figures")

    os.makedirs(reg_curve_dir, exist_ok=True)
    os.makedirs(reg_curve_fig_dir, exist_ok=True)

    curves = {
        "train_objective": [],
        "val_objective": [],
        "train_sumrate": [],
        "val_sumrate": [],
        "train_comm_sinr_penalty": [],
        "val_comm_sinr_penalty": [],
        "train_sensing_penalty": [],
        "val_sensing_penalty": [],
        "train_min_sinr_db": [],
        "val_min_sinr_db": [],
        "train_sinr_outage": [],
        "val_sinr_outage": [],
        "train_target_snr_db": [],
        "val_target_snr_db": [],
        "train_target_snr_outage": [],
        "val_target_snr_outage": [],
    }

    best_val_objective = -1e30

    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0]
    val_channels   = val_dataset["h_dk_hat"].shape[0]

    print("\n" + "=" * 90)
    print("[ONE TIMESCALE REG TRAIN]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"REG_EPOCHS              : {REG_EPOCHS}")
    print(f"MINIBATCHES             : {MINIBATCHES}")
    print(f"BATCH_CHANNELS          : {BATCH_CHANNELS}")
    print(f"REG learning rate       : {REG_LEARNING_RATE}")
    print(f"COMM SINR threshold     : {COMM_SINR_THRESHOLD_DB} dB")
    print(f"Sensing SNR threshold   : {SENSING_SNR_THRESHOLD_dB} dB")
    print(f"COMM penalty weight     : {COMM_SINR_PENALTY_WEIGHT}")
    print(f"Sensing penalty weight  : {REG_SENSING_LOSS_WEIGHT}")
    # print(f"Comm ckpt               : {reg_comm_ckpt}")
    # print(f"Test ckpt               : {reg_comm_ckpt}")
    # print(f"Radar ckpt              : {reg_radar_ckpt}")
    print(f"Theta ckpt              : {reg_theta_ckpt}")
    print("=" * 90)

    # 開始訓練
    for epoch in trange(REG_EPOCHS, desc="REG Training"):
        
        # comm_net.train()
        # radar_net.train()
        theta_net.train()
        # test_net.train()

        epoch_objectives = []
        epoch_sumrates = []
        epoch_comm_penalties = []
        epoch_sensing_penalties = []
        epoch_min_sinr_db = []
        epoch_sinr_outage = []
        epoch_target_snr_db = []
        epoch_target_snr_outage = []

        for _ in range(MINIBATCHES):

            channel_ids = np.random.choice(train_channels,size=BATCH_CHANNELS,replace=False)

            h_dk_hat = torch.as_tensor(train_dataset["h_dk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)

            optimizer.zero_grad(set_to_none=True)

            theta = theta_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)

            H_eff_H = physics_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)


            # W_C = test_net(H_eff_H)

            # W_C = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)

            W_C = make_rzf_beamformer(H_eff_H,RZF_ALPHA_FACTOR)

            # W_R = radar_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)

            W_R = mrt_in_H_eff_H_nullspace(H_eff_H, g_dt_hat)

            W_C, W_R = physics_net.normalize_isac_beamformers(W_C, W_R, g_dt_hat)


            metrics = comm_net.compute_isac_batch_performance(H_eff_H,g_dt_hat,W_C,W_R)

            sumrate_mean  = metrics["sumrate_mean"]
            sinr_db       = metrics["sinr_db"]
            target_snr_db = metrics["target_snr_db"]

            min_sinr_per_channel = torch.min(metrics["sinr"],dim=1,).values                 # (B,)
            comm_violation = torch.relu(COMM_SINR_THRESHOLD - min_sinr_per_channel)         # (B,)
            comm_sinr_penalty = torch.mean((comm_violation / COMM_SINR_THRESHOLD) ** 2)     # scalar

            sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - metrics["target_snr"])   # (B,)
            sensing_penalty = torch.mean((sensing_violation / SENSING_SNR_THRESHOLD) ** 2)  # scalar

            objective = (
                sumrate_mean
                - COMM_SINR_PENALTY_WEIGHT * comm_sinr_penalty
                - REG_SENSING_LOSS_WEIGHT  * sensing_penalty
            )

            loss = -objective
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(theta_net.parameters()),
                max_norm=10.0,
            )

            optimizer.step()

            sinr_out = (sinr_db < COMM_SINR_THRESHOLD_DB).to(torch.float32).mean()
            target_out = (target_snr_db < SENSING_SNR_THRESHOLD_dB).to(torch.float32).mean()

            epoch_objectives.append(float(objective.detach().cpu()))
            epoch_sumrates.append(float(sumrate_mean.detach().cpu()))
            epoch_comm_penalties.append(float(comm_sinr_penalty.detach().cpu()))
            epoch_sensing_penalties.append(float(sensing_penalty.detach().cpu()))
            epoch_min_sinr_db.append(float(torch.min(metrics["sinr_user_mean_db"]).detach().cpu()))
            epoch_sinr_outage.append(float(sinr_out.detach().cpu()))
            epoch_target_snr_db.append(float(metrics["target_snr_mean_db"].detach().cpu()))
            epoch_target_snr_outage.append(float(target_out.detach().cpu()))

        train_logs = {
            "objective": float(np.mean(epoch_objectives)),
            "sumrate": float(np.mean(epoch_sumrates)),
            "comm_sinr_penalty": float(np.mean(epoch_comm_penalties)),
            "sensing_penalty": float(np.mean(epoch_sensing_penalties)),
            "min_sinr_db": float(np.mean(epoch_min_sinr_db)),
            "sinr_outage": float(np.mean(epoch_sinr_outage)),
            "target_snr_db": float(np.mean(epoch_target_snr_db)),
            "target_snr_outage": float(np.mean(epoch_target_snr_outage)),
        }
        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        # comm_net.eval()
        # radar_net.eval()
        theta_net.eval()
        # test_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)


            theta_val = theta_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            H_eff_val = comm_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta_val)

            # W_C_val = test_net(H_eff_val)

            # W_C_val = comm_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            W_C_val = make_rzf_beamformer(H_eff_val, RZF_ALPHA_FACTOR)

            # W_R_val = radar_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            W_R_val = mrt_in_H_eff_H_nullspace(H_eff_val, g_dt_val)

            W_C_val, W_R_val = comm_net.normalize_isac_beamformers(W_C_val,W_R_val,g_dt_val)

            val_W_C_power = torch.mean(torch.sum(torch.abs(W_C_val) ** 2, dim=(1, 2)))
            val_W_R_power = torch.mean(torch.sum(torch.abs(W_R_val) ** 2, dim=(1, 2)))
            val_total_power = val_W_C_power + val_W_R_power

            val_power_logs = {
                "W_C_ratio": float((val_W_C_power / val_total_power).detach().cpu()),
                "W_R_ratio": float((val_W_R_power / val_total_power).detach().cpu()),
            }

            val_metrics = comm_net.compute_isac_batch_performance(H_eff_val,g_dt_val,W_C_val,W_R_val)

            val_sumrate_mean  = val_metrics["sumrate_mean"]
            val_sinr_db       = val_metrics["sinr_db"]
            val_target_snr_db = val_metrics["target_snr_db"]

            min_sinr_per_channel = torch.min(val_metrics["sinr"],dim=1,).values             # (B,)
            val_comm_violation = torch.relu(COMM_SINR_THRESHOLD - min_sinr_per_channel)         # (B,)
            val_comm_sinr_penalty = torch.mean((val_comm_violation / COMM_SINR_THRESHOLD) ** 2)     # scalar

            val_sensing_violation = torch.relu(SENSING_SNR_THRESHOLD - val_metrics["target_snr"])   # (B,)
            val_sensing_penalty = torch.mean((val_sensing_violation / SENSING_SNR_THRESHOLD) ** 2)  # scalar

            val_objective = (
                val_sumrate_mean
                - COMM_SINR_PENALTY_WEIGHT * val_comm_sinr_penalty
                - REG_SENSING_LOSS_WEIGHT  * val_sensing_penalty
            )

            val_sinr_out = (
                val_sinr_db < COMM_SINR_THRESHOLD_DB
            ).to(torch.float32).mean()

            val_target_out = (
                val_target_snr_db < SENSING_SNR_THRESHOLD_dB
            ).to(torch.float32).mean()

            val_logs = {
                "objective": float(val_objective.detach().cpu()),
                "sumrate": float(val_sumrate_mean.detach().cpu()),
                "comm_sinr_penalty": float(val_comm_sinr_penalty.detach().cpu()),
                "sensing_penalty": float(val_sensing_penalty.detach().cpu()),
                "min_sinr_db": float(torch.min(val_metrics["sinr_user_mean_db"]).detach().cpu()),
                "sinr_outage": float(val_sinr_out.detach().cpu()),
                "target_snr_db": float(val_metrics["target_snr_mean_db"].detach().cpu()),
                "target_snr_outage": float(val_target_out.detach().cpu()),
                "sinr_user_db": val_metrics["sinr_user_mean_db"].detach().cpu().numpy(),
                "rate_user": val_metrics["rate_user_mean"].detach().cpu().numpy(),
                "W_C_ratio": val_power_logs["W_C_ratio"],
                "W_R_ratio": val_power_logs["W_R_ratio"],
            }

        curves["train_objective"].append(train_logs["objective"])
        curves["val_objective"].append(val_logs["objective"])
        curves["train_sumrate"].append(train_logs["sumrate"])
        curves["val_sumrate"].append(val_logs["sumrate"])
        curves["train_comm_sinr_penalty"].append(train_logs["comm_sinr_penalty"])
        curves["val_comm_sinr_penalty"].append(val_logs["comm_sinr_penalty"])
        curves["train_sensing_penalty"].append(train_logs["sensing_penalty"])
        curves["val_sensing_penalty"].append(val_logs["sensing_penalty"])
        curves["train_min_sinr_db"].append(train_logs["min_sinr_db"])
        curves["val_min_sinr_db"].append(val_logs["min_sinr_db"])
        curves["train_sinr_outage"].append(train_logs["sinr_outage"])
        curves["val_sinr_outage"].append(val_logs["sinr_outage"])
        curves["train_target_snr_db"].append(train_logs["target_snr_db"])
        curves["val_target_snr_db"].append(val_logs["target_snr_db"])
        curves["train_target_snr_outage"].append(train_logs["target_snr_outage"])
        curves["val_target_snr_outage"].append(val_logs["target_snr_outage"])

        np.savez(
            reg_curve_path,
            **{
                key: np.asarray(value, dtype=np.float32)
                for key, value in curves.items()
            },
        )

        if val_logs["objective"] > best_val_objective:
            best_val_objective = val_logs["objective"]
            # comm_net.save_model(reg_comm_ckpt, verbose=False)
            # test_net.save_model(reg_comm_ckpt, verbose=False)
            # radar_net.save_model(reg_radar_ckpt, verbose=False)
            theta_net.save_model(reg_theta_ckpt, verbose=False)

        print(
            f"\n[Epoch {epoch + 1:03d}/{REG_EPOCHS}] "
            f"TrainObj={train_logs['objective']:.4f} "
            f"ValObj={val_logs['objective']:.4f} | "
            f"TrainRate={train_logs['sumrate']:.4f} "
            f"ValRate={val_logs['sumrate']:.4f} | "
            f"TrainMinSINR={train_logs['min_sinr_db']:.3f} dB "
            f"ValMinSINR={val_logs['min_sinr_db']:.3f} dB | "
            f"TrainSINROut={100.0 * train_logs['sinr_outage']:.2f}% "
            f"ValSINROut={100.0 * val_logs['sinr_outage']:.2f}% | "
            f"TrainTarSNR={train_logs['target_snr_db']:.3f} dB "
            f"ValTarSNR={val_logs['target_snr_db']:.3f} dB | "
            f"TrainTarOut={100.0 * train_logs['target_snr_outage']:.2f}% "
            f"ValTarOut={100.0 * val_logs['target_snr_outage']:.2f}% | "
            f"ValPc={100.0 * val_logs['W_C_ratio']:.2f}% "
            f"ValPr={100.0 * val_logs['W_R_ratio']:.2f}% | "
            f"ValSINRUserDB={fmt_vec(val_logs['sinr_user_db'], precision=3)} "
            f"ValRateUser={fmt_vec(val_logs['rate_user'], precision=4)}"
        )


    plot_one_timescale_reg_curves(
        curve_path=reg_curve_path,
        save_dir=reg_curve_fig_dir,
        ma_window=20,
    )
    print("\n[INFO] One-timescale REG training finished.")
    print(f"[INFO] Best validation objective = {best_val_objective:.6f}")
    # print(f"[INFO] Best comm model saved to  : {reg_comm_ckpt}")
    # print(f"[INFO] Best test model saved to  : {reg_test_ckpt}")
    # print(f"[INFO] Best radar model saved to : {reg_radar_ckpt}")
    print(f"[INFO] Best theta model saved to : {reg_theta_ckpt}")
    print(f"[INFO] Curves saved to           : {reg_curve_path}")

    print("[INFO] Short-term regular training finished.")




