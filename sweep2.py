# -*- coding: utf-8 -*-
import sys
import argparse
import subprocess
from datetime import datetime

from settings import (
    INJECTION_VARIANCE,
    INJECTION_SAMPLES,
    ST_SWEEP_DIR,
)


# ================================
# Paired penalty list
# ================================
# 這裡要和你 sweep.py 實際訓練的 7 組一致
PAIRED_PENALTY_LIST = [
    (50.0,  0.10),
    (70.0,  0.15),
    (100.0, 0.20),
    (150.0, 0.25),
    (175.0, 0.30),
    (200.0, 0.50),
    (300.0, 1.00),
]


# ================================
# Helpers
# ================================
def run_command(cmd: list[str], dry_run: bool = False):
    """
    執行 terminal command。

    dry_run=True:
        只印出 command，不實際畫圖。
    """
    print("\n" + "=" * 80)
    print("[SWEEP2-PLOT] Command:")
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        print("[SWEEP2-PLOT] dry-run mode，未實際執行。")
        return

    subprocess.run(
        cmd,
        check=True,
    )


def plot_one_run(
    mode: str,
    reg_penalty: float,
    rob_penalty: float,
    dry_run: bool,
):
    """
    呼叫 main_st.py 畫單一 REG 或 ROB run 的 training curve。

    注意：
        main_st.py 的 --penalty 需要兩個值：
            --penalty REG_PENALTY ROB_PENALTY

        即使只畫 REG，也要傳入 ROB_PENALTY。
        即使只畫 ROB，也要傳入 REG_PENALTY。
    """
    mode = mode.lower()

    if mode not in ("reg", "rob"):
        raise ValueError(f"mode 必須是 reg 或 rob，收到：{mode}")

    cmd = [
        sys.executable,
        "main_st.py",
        "--mode",
        mode,
        "--penalty",
        str(float(reg_penalty)),
        str(float(rob_penalty)),
        "--plot",
    ]

    run_command(
        cmd=cmd,
        dry_run=dry_run,
    )


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot all short-term REG / ROB penalty sweep curves."
    )

    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "reg", "rob"],
        help="選擇要 plot 的模型類型：all / reg / rob。預設 all。"
    )

    parser.add_argument(
        "--dry_run",
        "--dry-run",
        action="store_true",
        help="只印出將執行的 command，不實際畫圖。"
    )

    args = parser.parse_args()

    print("\n[SWEEP2-PLOT] Short-term penalty sweep plotting started.")
    print(f"[SWEEP2-PLOT] Time = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SWEEP2-PLOT] ST_SWEEP_DIR = {ST_SWEEP_DIR}")
    print(f"[SWEEP2-PLOT] only = {args.only}")
    print(f"[SWEEP2-PLOT] dry_run = {args.dry_run}")
    print(f"[SWEEP2-PLOT] ROB train_inj = {INJECTION_VARIANCE}")
    print(f"[SWEEP2-PLOT] ROB injection_samples = {INJECTION_SAMPLES}")

    if args.only in ("all", "reg"):
        print("\n" + "#" * 80)
        print("[SWEEP2-PLOT] REG penalty plots")
        print("#" * 80)

        for reg_penalty, rob_penalty in PAIRED_PENALTY_LIST:
            plot_one_run(
                mode="reg",
                reg_penalty=reg_penalty,
                rob_penalty=rob_penalty,
                dry_run=args.dry_run,
            )

    if args.only in ("all", "rob"):
        print("\n" + "#" * 80)
        print("[SWEEP2-PLOT] ROB penalty plots")
        print("#" * 80)

        for reg_penalty, rob_penalty in PAIRED_PENALTY_LIST:
            plot_one_run(
                mode="rob",
                reg_penalty=reg_penalty,
                rob_penalty=rob_penalty,
                dry_run=args.dry_run,
            )

    print("\n[SWEEP2-PLOT] Short-term penalty sweep plotting finished.")