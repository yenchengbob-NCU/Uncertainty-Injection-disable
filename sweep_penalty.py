# -*- coding: utf-8 -*-
import re
import sys
import shutil
import subprocess
from pathlib import Path


# ================================
# Sweep penalties
# ================================
REG_PENALTY_SWEEP = [0.0, 1.0, 3.0, 5.0, 7.0, 10.0 ]

ROB_PENALTY_SWEEP = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0 ]


# ================================
# 執行選項
# ================================
RUN_REG = True
RUN_ROB = False

# False：任何一次訓練失敗就停止
# True ：記錄失敗後繼續下一個 penalty
CONTINUE_ON_ERROR = False


# ================================
# Project paths
# ================================
PROJECT_DIR = Path(__file__).resolve().parent

SETTINGS_PATH = PROJECT_DIR / "settings.py"
REG_PATH = PROJECT_DIR / "reg.py"
ROB_PATH = PROJECT_DIR / "rob.py"

BACKUP_PATH = PROJECT_DIR / "settings.py.sweep_backup"


# ================================
# 修改 settings.py 中指定參數
# ================================
def update_setting(setting_name, setting_value):
    text = SETTINGS_PATH.read_text(encoding="utf-8")

    pattern = rf"^(\s*{re.escape(setting_name)}\s*=\s*)([^#\r\n]*)(.*)$"
    replacement = rf"\g<1>{setting_value:.12g}\g<3>"

    updated_text, replace_count = re.subn(
        pattern,
        replacement,
        text,
        flags=re.MULTILINE,
    )

    if replace_count != 1:
        raise RuntimeError(
            f"settings.py 中預期只有一個 `{setting_name}`，"
            f"但實際找到 {replace_count} 個。"
        )

    SETTINGS_PATH.write_text(updated_text, encoding="utf-8")


# ================================
# 執行單次訓練
# ================================
def run_training(script_path, model_name, penalty_weight):
    print("\n" + "=" * 90)
    print(f"[RUN] {model_name}")
    print(f"[RUN] penalty weight = {penalty_weight:g}")
    print(f"[RUN] script         = {script_path.name}")
    print("=" * 90)

    result = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        cwd=str(PROJECT_DIR),
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"{script_path.name} 執行失敗，"
            f"penalty={penalty_weight:g}，"
            f"return code={result.returncode}"
        )


# ================================
# 檢查必要檔案
# ================================
required_paths = [
    SETTINGS_PATH,
    REG_PATH,
    ROB_PATH,
]

for required_path in required_paths:
    if not required_path.exists():
        raise FileNotFoundError(f"找不到必要檔案：{required_path}")


# ================================
# Backup settings.py
# ================================
original_settings = SETTINGS_PATH.read_text(encoding="utf-8")
shutil.copy2(SETTINGS_PATH, BACKUP_PATH)

failed_runs = []

try:
    # ================================
    # REG sweep
    # ================================
    if RUN_REG:
        print("\n" + "#" * 90)
        print("# REG sensing penalty sweep")
        print("#" * 90)

        for sweep_idx, penalty_weight in enumerate(REG_PENALTY_SWEEP, start=1):
            print(
                f"\n[REG {sweep_idx:02d}/{len(REG_PENALTY_SWEEP):02d}] "
                f"REG_SENSING_LOSS_WEIGHT = {penalty_weight:g}"
            )

            try:
                update_setting("REG_SENSING_LOSS_WEIGHT", penalty_weight)
                run_training(REG_PATH, "REG", penalty_weight)

            except Exception as error:
                failed_runs.append(("REG", penalty_weight, str(error)))
                print(f"[ERROR] REG penalty={penalty_weight:g} failed")
                print(f"[ERROR] {error}")

                if not CONTINUE_ON_ERROR:
                    raise

    # ================================
    # ROB sweep
    # ================================
    if RUN_ROB:
        print("\n" + "#" * 90)
        print("# ROB sensing penalty sweep")
        print("#" * 90)

        for sweep_idx, penalty_weight in enumerate(ROB_PENALTY_SWEEP, start=1):
            print(
                f"\n[ROB {sweep_idx:02d}/{len(ROB_PENALTY_SWEEP):02d}] "
                f"ROB_SENSING_LOSS_WEIGHT = {penalty_weight:g}"
            )

            try:
                update_setting("ROB_SENSING_LOSS_WEIGHT", penalty_weight)
                run_training(ROB_PATH, "ROB", penalty_weight)

            except Exception as error:
                failed_runs.append(("ROB", penalty_weight, str(error)))
                print(f"[ERROR] ROB penalty={penalty_weight:g} failed")
                print(f"[ERROR] {error}")

                if not CONTINUE_ON_ERROR:
                    raise

finally:
    SETTINGS_PATH.write_text(original_settings, encoding="utf-8")

    if BACKUP_PATH.exists():
        BACKUP_PATH.unlink()

    print("\n[INFO] Original settings.py restored.")


# ================================
# Final summary
# ================================
print("\n" + "=" * 90)
print("[INFO] Penalty sweep finished.")
print("=" * 90)

if failed_runs:
    print("[WARNING] Failed runs:")

    for model_name, penalty_weight, error_message in failed_runs:
        print(f"  {model_name}: penalty={penalty_weight:g}")
        print(f"    {error_message}")
else:
    print("[INFO] All requested training runs completed successfully.")