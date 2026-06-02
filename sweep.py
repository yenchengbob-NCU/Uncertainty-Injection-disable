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
# Penalty sweep settings
# ================================
REG_PENALTY_LIST = [70,100,150,200,300]

ROB_PENALTY_LIST = [0.025,0.05,0.1,0.5,1]


# ================================
# Helpers
# ================================
def run_command(cmd: list[str], dry_run: bool = False):
    """
    執行 subprocess 指令。
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


def train_one_run(
    mode: str,
    penalty: float,
    resume: bool,
    dry_run: bool,
):
    """
    呼叫 main_st.py 訓練單一 REG 或 ROB run。
    """
    mode = mode.lower()

    cmd = [
        sys.executable,
        "main_st.py",
        "--mode",
        mode,
        "--penalty",
        str(penalty),
    ]

    if resume:
        cmd += ["--resume"]

    run_command(
        cmd=cmd,
        dry_run=dry_run,
    )


# ================================
# Main
# ================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep short-term REG / ROB sensing penalties."
    )

    parser.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "reg", "rob"],
        help="選擇要 sweep 的模型類型：all / reg / rob。預設 all。"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="對每個 run 使用 --resume 續訓。"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只印出將執行的 command，不實際訓練。"
    )

    args = parser.parse_args()

    print("\n[SWEEP] Short-term penalty sweep started.")
    print(f"[SWEEP] Time = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SWEEP] ST_SWEEP_DIR = {ST_SWEEP_DIR}")
    print(f"[SWEEP] only = {args.only}")
    print(f"[SWEEP] resume = {args.resume}")
    print(f"[SWEEP] dry_run = {args.dry_run}")
    print(f"[SWEEP] ROB train_inj = {INJECTION_VARIANCE}")
    print(f"[SWEEP] ROB injection_samples = {INJECTION_SAMPLES}")

    if args.only in ("all", "reg"):
        print("\n" + "#" * 80)
        print("[SWEEP] REG penalty sweep")
        print("#" * 80)

        for penalty in REG_PENALTY_LIST:
            train_one_run(
                mode="reg",
                penalty=penalty,
                resume=args.resume,
                dry_run=args.dry_run,
            )

    if args.only in ("all", "rob"):
        print("\n" + "#" * 80)
        print("[SWEEP] ROB penalty sweep")
        print("#" * 80)

        for penalty in ROB_PENALTY_LIST:
            train_one_run(
                mode="rob",
                penalty=penalty,
                resume=args.resume,
                dry_run=args.dry_run,
            )

    print("\n[SWEEP] Short-term penalty sweep finished.")