# -*- coding: utf-8 -*-
"""
temp.py

用途：
    將 main_st.py 儲存的 short-term curve .npy
    轉成 block-averaged curve。

目前支援你的 .npy 格式：
    shortterm_regular_curves_*.npy shape = (epochs, 6)
    shortterm_robust_curves_*.npy  shape = (epochs, 6)

欄位定義：
    column 0: Train Objective
    column 1: Validation Objective
    column 2: Train SumRate
    column 3: Validation SumRate
    column 4: Train SNR / VaR-SNR dB
    column 5: Validation SNR / VaR-SNR dB

目前只畫：
    Train Objective
    Validation Objective

分段平均規則：
    x = 0:
        取 epoch 1 的值

    x = 100:
        平均 epoch 50~150

    x = 200:
        平均 epoch 151~250

    x = 300:
        平均 epoch 251~350

    依此類推。
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from settings import CURVE_DIR, SETTING_STRING


# ================================
# Block-average config
# ================================
CENTER_STEP = 50
FIRST_AVG_START = 25
FIRST_AVG_END = 75

SAVE_NAME = "temp_block_average_st_objective_curves.jpg"


# ================================
# Helpers
# ================================
def block_average_curve(y):
    """
    將原始 curve 轉成：
        x = 0, 100, 200, ...
        y = first point, average blocks...

    epoch 使用 1-based 解讀：
        epoch 50~150 -> Python index 49:150
    """
    y = np.asarray(y, dtype=np.float64)

    xs = []
    ys = []

    # x = 0，取第一個 epoch 的值
    xs.append(0)
    ys.append(float(y[0]))

    center = CENTER_STEP
    start_epoch = FIRST_AVG_START
    end_epoch = FIRST_AVG_END

    while start_epoch <= len(y):
        start_idx = start_epoch - 1
        end_idx = min(end_epoch, len(y))

        if end_idx <= start_idx:
            break

        block = y[start_idx:end_idx]

        if block.size == 0:
            break

        xs.append(center)
        ys.append(float(np.mean(block)))

        start_epoch = end_epoch + 1
        end_epoch = end_epoch + CENTER_STEP
        center += CENTER_STEP

    return np.asarray(xs), np.asarray(ys)


def find_shortterm_curve_files():
    """
    只尋找 shortterm regular / robust curve。
    """
    patterns = [
        os.path.join(CURVE_DIR, "shortterm_regular_curves_*.npy"),
        os.path.join(CURVE_DIR, "shortterm_robust_curves_*.npy"),
    ]

    files = []

    for pattern in patterns:
        files.extend(sorted(glob.glob(pattern)))

    if len(files) == 0:
        raise FileNotFoundError(
            f"CURVE_DIR 裡找不到 shortterm curve .npy：{CURVE_DIR}"
        )

    return files


def load_curve_array(path):
    """
    讀取 curve .npy，並檢查是否為二維 array。
    """
    arr = np.load(path, allow_pickle=True)

    if arr.ndim != 2:
        raise ValueError(
            f"{path} 應為 2D array，例如 shape=(epochs, 6)，但收到 shape={arr.shape}"
        )

    if arr.shape[1] < 2:
        raise ValueError(
            f"{path} 至少需要 2 欄：TrainObj / ValObj，但收到 shape={arr.shape}"
        )

    return arr.astype(np.float64)


def model_name_from_filename(filename):
    lower = filename.lower()

    if "regular" in lower:
        return "REG"

    if "robust" in lower:
        return "ROB"

    return "UNKNOWN"


# ================================
# Main
# ================================
if __name__ == "__main__":
    print("====================================================")
    print("[TEMP] Block-average ST objective curve plot")
    print(f"[TEMP] CURVE_DIR = {CURVE_DIR}")
    print(f"[TEMP] SETTING_STRING = {SETTING_STRING}")
    print("====================================================")

    curve_files = find_shortterm_curve_files()

    plt.figure(figsize=(10, 6))

    plotted = 0

    for curve_path in curve_files:
        filename = os.path.basename(curve_path)
        model_name = model_name_from_filename(filename)

        print(f"[TEMP] Loading: {curve_path}")

        curves = load_curve_array(curve_path)

        print(f"  shape = {curves.shape}")

        train_obj = curves[:, 0]
        val_obj = curves[:, 1]

        x_train, y_train = block_average_curve(train_obj)
        x_val, y_val = block_average_curve(val_obj)

        plt.plot(
            x_train,
            y_train,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=f"{model_name} Train Objective",
        )

        plt.plot(
            x_val,
            y_val,
            marker="o",
            linewidth=2.0,
            markersize=4,
            label=f"{model_name} Validation Objective",
        )

        print(f"  [plot] {model_name} TrainObj: raw N={len(train_obj)}, block N={len(x_train)}")
        print(f"  [plot] {model_name} ValObj  : raw N={len(val_obj)}, block N={len(x_val)}")

        plotted += 2

    if plotted == 0:
        raise RuntimeError("沒有畫出任何 curve。")

    plt.xlabel("Epoch block center")
    plt.ylabel("Block-averaged Objective")
    plt.title("Block-averaged Short-term Training / Validation Objective")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(CURVE_DIR, SAVE_NAME)
    plt.savefig(save_path, format="jpg", dpi=300)

    print("====================================================")
    print(f"[TEMP] Saved figure: {save_path}")
    print("====================================================")

    plt.show()
    plt.close()