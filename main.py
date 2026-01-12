# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
from settings import *
from rician import generate_real_channels, _estimate_single_channel
from neural_net import *

def plot_training_curves(curves: np.ndarray, start: int = 0):
    x = np.arange(start, curves.shape[0])
    train_obj = curves[start:, 0]
    val_obj = curves[start:, 1]

    plt.figure()
    plt.plot(x, train_obj, label="Train Objective", color="blue")
    plt.plot(x, val_obj, label="Val Objective", color="orange")

    # 標出每 20 epoch + 最後一點
    step = 20
    label_indices = list(range(0, len(x), step))
    if (len(x)-1) not in label_indices:
        label_indices.append(len(x)-1)

    for i in label_indices:
        xi = x[i]
        plt.text(xi, train_obj[i], f"{train_obj[i]:.2f}", color="blue", fontsize=8, ha='right', va='bottom')
        plt.text(xi, val_obj[i], f"{val_obj[i]:.2f}", color="orange", fontsize=8, ha='left', va='top')

    plt.xlabel("Epoch")
    plt.ylabel("Objective Value")
    plt.title(f"Training Objective Curves — {SETTING_STRING}")
    plt.xticks(np.arange(start, curves.shape[0]+1, step=10))
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    """numpy 複數陣列 -> torch.complex64 到 DEVICE。"""
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


# ------------------------------
# 單一步驗證（或訓練）前向：回傳 objective 與統計
# ------------------------------
def forward_objective(comm_net, sense_net, ris_net,
                      h_dk, h_rk, G, g_dt):

    # 1) 三個網路各自輸出
    W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
    W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)
    phi = ris_net(h_dk, h_rk, G, g_dt)    # (B,N)

    # 2) 通訊 SINR / 速率
    sinrs = comm_net.compute_comm_sinrs(
        h_dk, h_rk, G, phi, W_S, W_C
    )  # (B,K)

    rates = comm_net.compute_rates(sinrs)     # (B,K)
    sum_rate = rates.sum(dim=1)               # (B,)
    sum_rate_mean = sum_rate.mean()           # scalar

    # 3) 感測 SNR
    sense_snr = comm_net.compute_sense_snr(g_dt, W_S, W_C)  # (B,)
    snr_violation = torch.clamp(SENSING_SNR_THRESHOLD - sense_snr.real, min=0.0)
    snr_penalty_mean = snr_violation.mean()

    # 4) φ 懲罰
    phi_abs = phi.abs()
    phi_excess = torch.clamp(phi_abs - 1.0, min=0.0)
    phi_penalty_mean = (phi_excess ** 2).mean(dim=1).mean()

    # 5) TX power 懲罰
    power_comm  = (W_C.abs() ** 2).sum(dim=(1, 2))
    power_sense = (W_S.abs() ** 2).sum(dim=(1, 2))
    tx_power = power_comm + power_sense
    tx_power_mean = tx_power.mean()

    tx_excess = torch.clamp(tx_power - TRANSMIT_POWER_TOTAL, min=0.0)
    tx_penalty_mean = (tx_excess ** 2).mean()

    # 6) objective
    objective = (
        sum_rate_mean
        - SENSING_LOSS_WEIGHT  * snr_penalty_mean
        - RE_POWER_LOSS_WEIGHT * phi_penalty_mean
        - TX_POWER_LOSS_WEIGHT * tx_penalty_mean
    )

    logs = {
        "sum_rate_mean":        sum_rate_mean.detach(),
        "sense_snr_mean_db":    (10.0 * torch.log10(sense_snr.real.clamp_min(1e-12))).mean().detach(),
        "snr_penalty_mean":     snr_penalty_mean.detach(),
        "phi_penalty_mean":     phi_penalty_mean.detach(),
        "tx_power_mean":        tx_power_mean.detach(),
        "tx_penalty_mean":      tx_penalty_mean.detach(),
        "objective":            objective.detach(),
    }
    return objective, logs


# ------------------------------
# main
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISAC training script (3-MLP joint)")
    parser.add_argument("--plot", action="store_true",  help="Plot the saved training curves and exit")
    parser.add_argument("--start", type=int, default=0, help="Start index for plotting training curves")
    args = parser.parse_args()

    # ===============================
    # 輸出資料夾
    # ===============================
    base_dir  = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
    ckpt_dir  = os.path.join(base_dir, "ckpt")
    curve_dir = os.path.join(base_dir, "training_curves")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)

    # 曲線檔案
    curves_path = os.path.join(curve_dir, f"training_curves_{SETTING_STRING}.npy")

    # 只繪圖就退出
    if args.plot:
        if not os.path.exists(curves_path):
            raise FileNotFoundError(f"找不到曲線檔案：{curves_path}")
        curves = np.load(curves_path)
        plot_training_curves(curves, args.start)
        raise SystemExit(0)

    # ===============================
    # 建立 3 個網路與 optimizer
    # ===============================
    comm_net  = CommBeamformerNet().to(DEVICE)
    sense_net = SenseBeamformerNet().to(DEVICE)
    ris_net   = RISPhaseNet().to(DEVICE)

    optimizer = optim.Adam(
        list(comm_net.parameters()) +
        list(sense_net.parameters()) +
        list(ris_net.parameters()),
        lr=LEARNING_RATE
    )

    

    # checkpoint 路徑（最佳 val 依據 objective）
    best_val = -np.inf
    comm_ckpt_path  = os.path.join(ckpt_dir,  f"comm_{SETTING_STRING}.ckpt")
    sense_ckpt_path = os.path.join(ckpt_dir,  f"sense_{SETTING_STRING}.ckpt")
    ris_ckpt_path   = os.path.join(ckpt_dir,  f"ris_{SETTING_STRING}.ckpt")

    curves = []  # 每 epoch 追加一行 [train_obj, val_obj, val_sum_rate, val_sense_snr_dB]

    # ===============================
    # 訓練循環
    # ===============================
    for ep in trange(1, EPOCHS + 1, desc="Epoch"):
        # ---------- TRAIN ----------
        comm_net.train(); sense_net.train(); ris_net.train()
        obj_ep = 0.0

        for num in range(MINIBATCHES):
            # 產生一個 mini-batch 通道、加入估測
            h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(BATCH_SIZE)

            h_dk_est = _estimate_single_channel(h_dk_np)
            h_rk_est = _estimate_single_channel(h_rk_np)
            G_est    = _estimate_single_channel(G_np)
            g_dt_est = _estimate_single_channel(g_dt_np)

            h_dk = np_to_torch_complex(h_dk_est)
            h_rk = np_to_torch_complex(h_rk_est)
            G    = np_to_torch_complex(G_est)
            g_dt = np_to_torch_complex(g_dt_est)

            optimizer.zero_grad(set_to_none=True)
            objective, logs = forward_objective(
                comm_net, sense_net, ris_net,
                h_dk, h_rk, G, g_dt)
            
            loss = -objective
            loss.backward()
            optimizer.step()
            obj_ep += objective.item() / MINIBATCHES

        # ---------- VALIDATE ----------
        comm_net.eval(); sense_net.eval(); ris_net.eval()
        with torch.no_grad():

            h_dk_np, h_rk_np, G_np, g_dt_np = generate_real_channels(BATCH_SIZE)

            h_dk_est = _estimate_single_channel(h_dk_np)
            h_rk_est = _estimate_single_channel(h_rk_np)
            G_est    = _estimate_single_channel(G_np)
            g_dt_est = _estimate_single_channel(g_dt_np)

            h_dk = np_to_torch_complex(h_dk_est)
            h_rk = np_to_torch_complex(h_rk_est)
            G    = np_to_torch_complex(G_est)
            g_dt = np_to_torch_complex(g_dt_est)

            val_obj, logs = forward_objective(
                comm_net, sense_net, ris_net,
                h_dk, h_rk, G, g_dt
                )
            val_obj = float(val_obj.detach().cpu())
            val_sum_rate = float(logs["sum_rate_mean"].cpu())
            val_sense_snr_db = float(logs["sense_snr_mean_db"].cpu())

        # 記錄曲線並顯示
        curves.append([obj_ep, val_obj, val_sum_rate, val_sense_snr_db])
        np.save(curves_path, np.array(curves, dtype=np.float32))

        print(f"[Epoch {ep:03d}] "
              f"TrainObj: {obj_ep: .4e} | "
              f"ValObj: {val_obj: .4e} | "
              f"Val SumRate: {val_sum_rate: .4e} | "
              f"Val SenseSNR(dB): {val_sense_snr_db: .3f}")

        # 儲存最佳 checkpoint（以 val_obj 為準）
        if val_obj > best_val:
            best_val = val_obj
            torch.save(comm_net.state_dict(),  comm_ckpt_path)
            torch.save(sense_net.state_dict(), sense_ckpt_path)
            torch.save(ris_net.state_dict(),   ris_ckpt_path)
            print(f"[CKPT] Saved best @ {ep}: "
                  f"{os.path.basename(comm_ckpt_path)}, "
                  f"{os.path.basename(sense_ckpt_path)}, "
                  f"{os.path.basename(ris_ckpt_path)}")

    print("Training Script finished!")