# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt

from settings import *


# ================================
# Plot config
# ================================
MOVING_AVG_WINDOW = 20      # moving average 視窗長度
SKIP_EPOCHS = 1             # 跳過前幾個 epoch；1 代表跳過 epoch 1


# ================================
# Helpers
# ================================
def moving_average(x, window):
    """
    trailing moving average.
    輸出長度與原始 x 相同。
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)

    for i in range(len(x)):
        start = max(0, i - window + 1)
        y[i] = np.mean(x[start:i + 1])

    return y


def plot_objective_curve(curve_path, fig_path, title, train_label, val_label):
    curves = np.load(curve_path)

    # col 0: train objective
    # col 1: validation objective
    train_obj = curves[:, 0]
    val_obj = curves[:, 1]

    epochs = np.arange(1, curves.shape[0] + 1)

    # 跳過前幾個 epoch
    if SKIP_EPOCHS > 0:
        epochs = epochs[SKIP_EPOCHS:]
        train_obj = train_obj[SKIP_EPOCHS:]
        val_obj = val_obj[SKIP_EPOCHS:]

    train_ma = moving_average(train_obj, MOVING_AVG_WINDOW)
    val_ma = moving_average(val_obj, MOVING_AVG_WINDOW)

    plt.figure(figsize=(8, 5))

    # raw curve
    plt.plot(epochs, train_obj, alpha=0.25, label=f"{train_label} Raw")
    plt.plot(epochs, val_obj, alpha=0.25, label=f"{val_label} Raw")

    # moving average curve
    plt.plot(epochs, train_ma, linewidth=2.0, label=f"{train_label} MA-{MOVING_AVG_WINDOW}")
    plt.plot(epochs, val_ma, linewidth=2.0, label=f"{val_label} MA-{MOVING_AVG_WINDOW}")

    plt.xlabel("Epoch")
    plt.ylabel("Objective")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, format="jpg", dpi=300)
    print(f"[PLOT] saved: {fig_path}")

    plt.show()
    plt.close()


# ================================
# Main
# ================================
if __name__ == "__main__":

    # ---------- Long-term ----------
    lt_curve_path = os.path.join(
        CURVE_DIR,
        f"longterm_curves_{SETTING_STRING}.npy"
    )

    lt_fig_path = os.path.join(
        CURVE_DIR,
        f"longterm_objective_moving_average_W{MOVING_AVG_WINDOW}_{SETTING_STRING}.jpg"
    )

    plot_objective_curve(
        curve_path=lt_curve_path,
        fig_path=lt_fig_path,
        title="Long-term Objective Moving Average",
        train_label="LT Train Objective",
        val_label="LT Validation Objective",
    )

    # ---------- Short-term Regular ----------
    reg_curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_regular_curves_{SETTING_STRING}.npy"
    )

    reg_fig_path = os.path.join(
        CURVE_DIR,
        f"shortterm_regular_objective_moving_average_W{MOVING_AVG_WINDOW}_{SETTING_STRING}.jpg"
    )

    plot_objective_curve(
        curve_path=reg_curve_path,
        fig_path=reg_fig_path,
        title="Short-term Regular Objective Moving Average",
        train_label="REG Train Objective",
        val_label="REG Validation Objective",
    )

    # ---------- Short-term Robust ----------
    rob_curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_curves_{SETTING_STRING}.npy"
    )

    rob_fig_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_objective_moving_average_W{MOVING_AVG_WINDOW}_{SETTING_STRING}.jpg"
    )

    plot_objective_curve(
        curve_path=rob_curve_path,
        fig_path=rob_fig_path,
        title="Short-term Robust Objective Moving Average",
        train_label="ROB Train Objective",
        val_label="ROB Validation Objective",
    )

    # ---------- Short-term Robust Retrain ----------
    rob_retrain_curve_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_retrain_curves_{SETTING_STRING}.npy"
    )

    rob_retrain_fig_path = os.path.join(
        CURVE_DIR,
        f"shortterm_robust_retrain_objective_moving_average_W{MOVING_AVG_WINDOW}_{SETTING_STRING}.jpg"
    )

    plot_objective_curve(
        curve_path=rob_retrain_curve_path,
        fig_path=rob_retrain_fig_path,
        title="Short-term Robust Retrain Objective Moving Average",
        train_label="ROB-Retrain Train Objective",
        val_label="ROB-Retrain Validation Objective",
    )



    print("[INFO] All moving-average plots finished.")