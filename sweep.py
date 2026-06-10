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
# Paired penalty sweep settings
# ================================
# 第 i 個 REG penalty 會搭配第 i 個 ROB penalty
REG_PENALTY_LIST = [
    50,
    70,
    100,
    150,
    175,
    200,
    300,
]

ROB_PENALTY_LIST = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.50,
    1.00,
]


# ================================
# Helpers
# ================================
def run_command(cmd: list[str], dry_run: bool = False):
    """
    執行 terminal command。

    dry_run=True:
        只印出 command，不實際訓練。
    """
    print("\n" + "=" * 80)
    print("[SWEEP] Command:")
    print(" ".join(cmd))
    print("=" * 80)

    if dry_run:
        print("[SWEEP] dry-run mode，未實際執行。")
        return

    subprocess.run(
        cmd,
        check=True,
    )


def train_one_pair(
    reg_penalty: float,
    rob_penalty: float,
    dry_run: bool,
):
    """
    使用 main_st.py --mode both 訓練一組 REG + ROB。

    指令格式:
        python main_st.py --mode both --penalty REG_PENALTY ROB_PENALTY

    例:
        python main_st.py --mode both --penalty 100.0 0.05
    """
    cmd = [
        sys.executable,
        "main_st.py",
        "--mode",
        "both",
        "--penalty",
        str(float(reg_penalty)),
        str(float(rob_penalty)),
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
        description="Paired sweep for short-term REG / ROB sensing penalties."
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只印出將執行的 command，不實際訓練。"
    )

    args = parser.parse_args()

    if len(REG_PENALTY_LIST) != len(ROB_PENALTY_LIST):
        raise ValueError(
            "REG_PENALTY_LIST 與 ROB_PENALTY_LIST 長度必須相同，"
            "因為 paired sweep 會一對一配對。"
        )

    print("\n[SWEEP] Paired short-term penalty sweep started.")
    print(f"[SWEEP] Time = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SWEEP] ST_SWEEP_DIR = {ST_SWEEP_DIR}")
    print(f"[SWEEP] dry_run = {args.dry_run}")
    print(f"[SWEEP] ROB train_inj = {INJECTION_VARIANCE}")
    print(f"[SWEEP] ROB injection_samples = {INJECTION_SAMPLES}")

    for reg_penalty, rob_penalty in zip(REG_PENALTY_LIST, ROB_PENALTY_LIST):
        train_one_pair(
            reg_penalty=reg_penalty,
            rob_penalty=rob_penalty,
            dry_run=args.dry_run,
        )

    print("\n[SWEEP] Paired short-term penalty sweep finished.")