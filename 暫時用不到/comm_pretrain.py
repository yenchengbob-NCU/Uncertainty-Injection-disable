# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange

from settings import *
from baseline import make_random_ris ,make_rzf_beamformer, RZF_LAMBDA
from 暫時用不到.one_timescale_NN import CommNet

# ================================
# Helpers
# ================================
def fmt_vec(x, precision=4):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}f}]" for v in x]) + "}"


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


def moving_average(x, window):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    window = int(window)

    if window <= 1 or len(x) < window:
        return np.arange(1, len(x) + 1), x

    kernel = np.ones(window, dtype=np.float64) / window
    y_ma = np.convolve(x, kernel, mode="valid")
    ma_epochs = np.arange(window, len(x) + 1)

    return ma_epochs, y_ma


def compute_comm_sumrate(H_eff_H, W_C):
    """
    只用 H_eff_H 與 W_C 計算 communication sum-rate。
    Input:
        H_eff_H : (B, K, M), complex
        W_C     : (B, M, K), complex
    Return:
        mean_sumrate : scalar tensor
    """
    noise = torch.as_tensor(NOISE_POWER, dtype=torch.float32, device=W_C.device)

    Y = torch.matmul(H_eff_H, W_C)                  # (B,K,K)
    P = torch.abs(Y) ** 2                           # (B,K,K)

    signal = torch.diagonal(P, dim1=1, dim2=2)      # (B,K)
    comm_interf = torch.sum(P, dim=2) - signal      # (B,K)

    sinr = signal / (comm_interf + noise)           # (B,K) 
    rate = torch.log2(1.0 + sinr)                   # (B,K)

    signal_user        = torch.mean(signal, dim=0)      # (K,)
    comm_interf_user   = torch.mean(comm_interf, dim=0) # (K,)
    rate_user_mean      = torch.mean(rate, dim=0)        # (K,)

    sumrate = torch.sum(rate, dim=1)                # (B,)
    sumrate_mean = torch.mean(sumrate)              # scalar
    return {
        # debug check
        "signal_user": signal_user,                 # (K,)
        "comm_interf_user": comm_interf_user,       # (K,)
        "noise": noise,                             # scalar

        # B-average display values
        "rate_user_mean": rate_user_mean,           # 各UE的rate (K,)
        "sumrate_mean": sumrate_mean,               # sumrate    scalar
    }



def plot_pretrain_curves(curve_path, ma_window=20):
    """
    畫三張圖
    1.NN W_C 與 RZF 的 MSE
    2.NN W_C 與 RZF 的 sumrate
    3.NN W_C 與 RZF 的 sumrate 差距
    """
    data    = np.load(curve_path)
    out_dir = os.path.dirname(curve_path)

    # 讀資料
    train_mse           = data["train_mse"]
    val_mse             = data["val_mse"]

    train_NN_sumrate    = data["train_NN_sumrate"]
    val_NN_sumrate      = data["val_NN_sumrate"]

    train_rzf_sumrate   = data["train_rzf_sumrate"]
    val_rzf_sumrate     = data["val_rzf_sumrate"]

    epochs = np.arange(1, len(train_mse) + 1)

    # ================================
    # Fig 1: MSE curve
    # ================================
    plt.figure(figsize=(9, 5))

    plt.plot(epochs, train_mse, alpha=0.25, linewidth=1.0, label="Train MSE")
    plt.plot(epochs, val_mse,   alpha=0.25, linewidth=1.0, label="Val MSE")

    ma_ep, train_mse_ma = moving_average(train_mse, ma_window)
    ma_ep, val_mse_ma   = moving_average(val_mse, ma_window)

    plt.plot(ma_ep, train_mse_ma, linewidth=2.0, label=f"Train MSE MA({ma_window})")
    plt.plot(ma_ep, val_mse_ma,   linewidth=2.0, label=f"Val MSE MA({ma_window})")

    mse_all = np.concatenate([train_mse, val_mse])
    if np.all(mse_all[np.isfinite(mse_all)] > 0):
        plt.yscale("log")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("CommNet ZF/RZF Pretrain MSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    mse_fig_path = os.path.join(out_dir, "pretrain_comm_mse.png")
    plt.savefig(mse_fig_path, dpi=300)
    plt.close()

    # ================================
    # Fig 2: Sumrate
    # ================================
    plt.figure(figsize=(9, 5))

    plt.plot(epochs, train_NN_sumrate,  alpha=0.25, linewidth=1.0, label="Train NN Sumrate")
    plt.plot(epochs, val_NN_sumrate,    alpha=0.25, linewidth=1.0, label="Val NN Sumrate")
    plt.plot(epochs, train_rzf_sumrate, alpha=0.25, linewidth=1.0, label="Train RZF Sumrate")
    plt.plot(epochs, val_rzf_sumrate,   alpha=0.25, linewidth=1.0, label="Val RZF Sumrate")

    ma_ep, train_NN_ma  = moving_average(train_NN_sumrate, ma_window)
    ma_ep, val_NN_ma    = moving_average(val_NN_sumrate, ma_window)
    ma_ep, train_rzf_ma = moving_average(train_rzf_sumrate, ma_window)
    ma_ep, val_rzf_ma   = moving_average(val_rzf_sumrate, ma_window)

    plt.plot(ma_ep, train_NN_ma,  linewidth=2.0, label=f"Train NN MA({ma_window})")
    plt.plot(ma_ep, val_NN_ma,    linewidth=2.0, label=f"Val NN MA({ma_window})")
    plt.plot(ma_ep, train_rzf_ma, linewidth=2.0, label=f"Train RZF MA({ma_window})")
    plt.plot(ma_ep, val_rzf_ma,   linewidth=2.0, label=f"Val RZF MA({ma_window})")

    plt.xlabel("Epoch")
    plt.ylabel("Sumrate")
    plt.title("CommNet Pretrain Sumrate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    sumrate_fig_path = os.path.join(out_dir, "pretrain_comm_sumrate.png")
    plt.savefig(sumrate_fig_path, dpi=300)
    plt.close()

    # ================================
    # Fig 3: Rate gap
    # ================================
    train_rate_gap = train_rzf_sumrate - train_NN_sumrate
    val_rate_gap   = val_rzf_sumrate   - val_NN_sumrate

    plt.figure(figsize=(9, 5))

    plt.plot(epochs, train_rate_gap, alpha=0.25, linewidth=1.0, label="Train Gap")
    plt.plot(epochs, val_rate_gap,   alpha=0.25, linewidth=1.0, label="Val Gap")

    ma_ep, train_gap_ma = moving_average(train_rate_gap, ma_window)
    ma_ep, val_gap_ma   = moving_average(val_rate_gap, ma_window)

    plt.plot(ma_ep, train_gap_ma, linewidth=2.0, label=f"Train Gap MA({ma_window})")
    plt.plot(ma_ep, val_gap_ma,   linewidth=2.0, label=f"Val Gap MA({ma_window})")

    plt.axhline(0.0, linestyle="--", linewidth=1.0)

    plt.xlabel("Epoch")
    plt.ylabel("RZF sumrate - NN sumrate")
    plt.title("CommNet Pretrain Rate Gap")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    gap_fig_path = os.path.join(out_dir, "pretrain_comm_rate_gap.png")
    plt.savefig(gap_fig_path, dpi=300)
    plt.close()

    print(f"[PLOT] saved: {mse_fig_path}")
    print(f"[PLOT] saved: {sumrate_fig_path}")
    print(f"[PLOT] saved: {gap_fig_path}")

# ================================
# Setting
# ================================
pretrain_epoch              = 1000
pretrain_batch              = 500
pretrain_channel_per_batch  = 10
pretrain_LR                 = 0.001


# ================================
# Main
# ================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plot"  , action="store_true")
    args = parser.parse_args()


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

    comm_net      = CommNet().to(DEVICE)
    optimizer     = optim.Adam(list(comm_net.parameters()),lr=pretrain_LR)

    os.makedirs(PRETRAIN_DIR, exist_ok=True)
    pre_comm_ckpt = os.path.join(PRETRAIN_DIR, "pretrain_comm.ckpt")            # best model
    pre_comm_state = os.path.join(PRETRAIN_DIR, "pretrain_comm_state.pt")       # latest training state
    curve_path    = os.path.join(PRETRAIN_DIR, "pretrain_comm_curves.npz")

    curves = {
        "train_mse": [],                # 每個 epoch 的平均 MSE
        "val_mse": [],

        "train_NN_sumrate": [],         # 每個 epoch 的平均 sumrate
        "val_NN_sumrate": [],

        "train_rzf_sumrate": [],        # 每個 epoch 的平均 sumrate
        "val_rzf_sumrate": [],
    }

    best_val_mse = 1e30
    start_epoch = 0

    # --resum
    if args.resume:                                     # 續訓區塊
        print("\n[INFO] Resume mode enabled.")
        # 載入資料
        with np.load(curve_path) as data:
            for key in curves:
                if key in data.files:
                    curves[key] = list(data[key].astype(np.float64))

        start_epoch  = len(curves["train_mse"])
        best_val_mse = float(np.min(curves["val_mse"]))

        print(f"[INFO] Loaded curves from: {curve_path}")
        print(f"[INFO] Existing epochs   : {start_epoch}")

        state = torch.load(pre_comm_state, map_location=DEVICE)
        comm_net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])

        best_val_mse = float(state.get("best_val_mse", best_val_mse))
        start_epoch = max(start_epoch, int(state.get("epoch", start_epoch)))

        print(f"[INFO] Loaded latest training state from: {pre_comm_state}")
        print(f"[INFO] Resume from epoch index          : {start_epoch}")


    # --plot
    if args.plot:                                       # 繪圖區塊
        plot_pretrain_curves(curve_path)
        raise SystemExit


    # 載入資料
    train_channels = train_dataset["h_dk_hat"].shape[0] # 一共有多少train channel
    val_channels   = val_dataset["h_dk_hat"].shape[0]   # 一共有多少val channel

    end_epoch = start_epoch + pretrain_epoch
    print("\n" + "=" * 90)
    print("[RZF like W_C pre train]")
    print("=" * 90)
    print(f"Train channels          : {train_channels}")
    print(f"Val channels            : {val_channels}")
    print(f"Pretrain_epoch          : {pretrain_epoch}")
    print(f"Resume                  : {args.resume}")
    print(f"Start epoch index       : {start_epoch}")
    print(f"End epoch number        : {end_epoch}")
    print(f"BATCHES                 : {pretrain_batch}")
    print(f"BATCH_CHANNELS          : {pretrain_channel_per_batch}")
    print(f"learning rate           : {pretrain_LR}")
    print("=" * 90)

    theta   = make_random_ris(1)                            # 固定theta因為非學習內容
    
    # 開始訓練
    for epoch in trange(start_epoch, end_epoch, desc="Comm Pretraining"):
        
        comm_net.train()

        epoch_loss = []                     # epoch 平均 MSE

        epoch_NN_rate_user  = []            # epoch 平均 各個 UErate
        epoch_rzf_rate_user = []

        epoch_NN_sumrate  = []              # epoch 平均 sumrate
        epoch_rzf_sumrate = []

        epoch_NN_comm_interf_user  = []     # epoch 平均 多用戶干擾
        epoch_rzf_comm_interf_user = []

        for _ in range(pretrain_batch):
            # 抽出 [pretrain_channel_per_batch] 個 channel
            channel_ids = np.random.choice(train_channels,size=pretrain_channel_per_batch,replace=False)

            h_dk_hat = torch.as_tensor(train_dataset["h_dk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            h_rk_hat = torch.as_tensor(train_dataset["h_rk_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            G_hat    = torch.as_tensor(train_dataset["G_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)
            g_dt_hat = torch.as_tensor(train_dataset["g_dt_hat"][channel_ids],dtype=torch.complex64,device=DEVICE)

            optimizer.zero_grad(set_to_none=True)

            H_eff_H = physics_net.compute_effective_channel(h_dk_hat,h_rk_hat,G_hat,theta)
            
            W_C = comm_net(h_dk_hat,h_rk_hat,G_hat,g_dt_hat)        # 將channel輸入得到 功率正規化 W_C                            
            
            W_C_rzf = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)       # 這裡輸出 功率正規化 W_C_RZF

            loss = torch.mean(torch.abs(W_C - W_C_rzf.detach()) ** 2)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(list(comm_net.parameters()),max_norm=10.0)

            optimizer.step()

            with torch.no_grad():               # 解釋為何這裡要 torch.no_grad(): ANS : 節約GPU運算
                train_NN_metrics    = compute_comm_sumrate(H_eff_H, W_C)
                train_rzf_metrics   = compute_comm_sumrate(H_eff_H, W_C_rzf)

                NN_rate_user        = train_NN_metrics["rate_user_mean"]       # 各UE rate
                RZF_rate_user       = train_rzf_metrics["rate_user_mean"]

                train_NN_sumrate    = train_NN_metrics["sumrate_mean"]          # Sumrate
                train_rzf_sumrate   = train_rzf_metrics["sumrate_mean"]

                train_NN_comm_interf_user = train_NN_metrics["comm_interf_user"] # 各UE受到的多用戶干擾
                train_rzf_comm_interf_user = train_rzf_metrics["comm_interf_user"]

            epoch_loss.append(float(loss.detach().cpu()))
            epoch_NN_sumrate.append(float(train_NN_sumrate.detach().cpu()))
            epoch_rzf_sumrate.append(float(train_rzf_sumrate.detach().cpu()))

            epoch_NN_rate_user.append(NN_rate_user.detach().cpu().numpy())
            epoch_rzf_rate_user.append(RZF_rate_user.detach().cpu().numpy())

            epoch_NN_comm_interf_user.append(train_NN_comm_interf_user.detach().cpu().numpy())
            epoch_rzf_comm_interf_user.append(train_rzf_comm_interf_user.detach().cpu().numpy())

        train_logs = {
            "mse": float(np.mean(epoch_loss)),
            "NN_sumrate": float(np.mean( epoch_NN_sumrate)),
            "rzf_sumrate": float(np.mean(epoch_rzf_sumrate)),
            "NN_rate_user": np.mean(np.asarray(epoch_NN_rate_user), axis=0),
            "RZF_rate_user": np.mean(np.asarray(epoch_rzf_rate_user), axis=0),

            "NN_comm_interf_user": np.mean(np.asarray(epoch_NN_comm_interf_user), axis=0),
            "RZF_comm_interf_user": np.mean(np.asarray(epoch_rzf_comm_interf_user), axis=0),
        }
        # ================================
        # Validation: 固定 layout 的全部 val channels
        # ================================
        comm_net.eval()

        with torch.no_grad():
            h_dk_val = torch.as_tensor(val_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)
            h_rk_val = torch.as_tensor(val_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)
            G_val    = torch.as_tensor(val_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)
            g_dt_val = torch.as_tensor(val_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)


            H_eff_H_val =  physics_net.compute_effective_channel(h_dk_val,h_rk_val,G_val,theta) 

            W_C_val     = comm_net(h_dk_val,h_rk_val,G_val,g_dt_val)

            W_C_rzf_val = make_rzf_beamformer(H_eff_H_val,RZF_LAMBDA)

            val_loss = torch.mean(torch.abs(W_C_val - W_C_rzf_val.detach()) ** 2)

            val_NN_metrics    = compute_comm_sumrate(H_eff_H_val, W_C_val)
            val_rzf_metrics   = compute_comm_sumrate(H_eff_H_val, W_C_rzf_val)

            val_NN_rate_user  = val_NN_metrics["rate_user_mean"]       # (K,)
            val_RZF_rate_user = val_rzf_metrics["rate_user_mean"]      # (K,)

            val_NN_sumrate  = val_NN_metrics["sumrate_mean"]           # scalar
            val_rzf_sumrate = val_rzf_metrics["sumrate_mean"]          # scalar

            val_NN_comm_interf_user  = val_NN_metrics["comm_interf_user"]   # (K,)
            val_RZF_comm_interf_user = val_rzf_metrics["comm_interf_user"]  # (K,)

            val_logs = {
                "mse": float(val_loss.detach().cpu()),

                "NN_sumrate": float(val_NN_sumrate.detach().cpu()),
                "rzf_sumrate": float(val_rzf_sumrate.detach().cpu()),

                "NN_rate_user": val_NN_rate_user.detach().cpu().numpy(),
                "RZF_rate_user": val_RZF_rate_user.detach().cpu().numpy(),

                "NN_comm_interf_user": val_NN_comm_interf_user.detach().cpu().numpy(),
                "RZF_comm_interf_user": val_RZF_comm_interf_user.detach().cpu().numpy(),
            }

        curves["train_mse"].append(train_logs["mse"])
        curves["val_mse"].append(val_logs["mse"])

        curves["train_NN_sumrate"].append(train_logs["NN_sumrate"])
        curves["val_NN_sumrate"].append(val_logs["NN_sumrate"])

        curves["train_rzf_sumrate"].append(train_logs["rzf_sumrate"])
        curves["val_rzf_sumrate"].append(val_logs["rzf_sumrate"])


        print(
            f"\n[Epoch {epoch + 1:04d}/{end_epoch}] "
            f"TrainMSE={train_logs['mse']:.6e} "
            f"ValMSE={val_logs['mse']:.6e} | "

            f"TrainNNRate={train_logs['NN_sumrate']:.4f} "
            f"ValNNRate={val_logs['NN_sumrate']:.4f} | "

            f"TrainRZFRate={train_logs['rzf_sumrate']:.4f} "
            f"ValRZFRate={val_logs['rzf_sumrate']:.4f}\n"

            f"TrainNNRateUser={fmt_vec(train_logs['NN_rate_user'], precision=4)} "
            f"ValNNRateUser={fmt_vec(val_logs['NN_rate_user'], precision=4)}\n"

            f"TrainRZFRateUser={fmt_vec(train_logs['RZF_rate_user'], precision=4)} "
            f"ValRZFRateUser={fmt_vec(val_logs['RZF_rate_user'], precision=4)}\n"

            f"TrainNNInterfUser={fmt_vec_sci(train_logs['NN_comm_interf_user'], precision=3)} "
            f"ValNNInterfUser={fmt_vec_sci(val_logs['NN_comm_interf_user'], precision=3)}\n"

            f"TrainRZFInterfUser={fmt_vec_sci(train_logs['RZF_comm_interf_user'], precision=3)} "
            f"ValRZFInterfUser={fmt_vec_sci(val_logs['RZF_comm_interf_user'], precision=3)}"
        )

        if val_logs["mse"] < best_val_mse:
            best_val_mse = val_logs["mse"]
            comm_net.save_model(pre_comm_ckpt, verbose=False)


        np.savez(
            curve_path,
            **{
                key: np.asarray(value, dtype=np.float64)
                for key, value in curves.items()
            },
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model": comm_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_mse": best_val_mse,
            },
            pre_comm_state,
        )

    # plot_one_timescale_reg_curves(curve_path)

    print("[INFO] Short-term regular training finished.")




