# -*- coding: utf-8 -*-
"""
run_sweep.py

用途：
    自動執行 ISAC9 paired penalty sweep。

自動 sweep 三組：
    (REG, ROB) = (70, 10)
    (REG, ROB) = (50, 5)
    (REG, ROB) = (30, 2)

每組自動執行：
    python settings.py
    python rician.py
    python main_lt.py
    python main_st.py

注意：
    這版不跑 plot，不跑 eval。
    避免圖片視窗跳出造成 sweep 卡住。
"""

import os
import re
import sys
import subprocess
from datetime import datetime

# ================================
# Force UTF-8 output on Windows
# ================================
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ================================
# Sweep settings
# ================================
EXPERIMENTS = [
    {
        "name": "REG70_ROB10",
        "reg_sensing_weight": 100.0,
        "rob_sensing_weight": 0.05,
    },
    {
        "name": "REG50_ROB5",
        "reg_sensing_weight": 100.0,
        "rob_sensing_weight": 0.01,
    },
    {
        "name": "REG30_ROB2",
        "reg_sensing_weight": 100.0,
        "rob_sensing_weight": 0.005,
    },
]


# ================================
# Files
# ================================
SETTINGS_FILE = "settings.py"

SCRIPT_SETTINGS = "settings.py"
SCRIPT_RICIAN = "rician.py"
SCRIPT_MAIN_LT = "main_lt.py"
SCRIPT_MAIN_ST = "main_st.py"


# ================================
# Helpers
# ================================
def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到必要檔案：{path}")


def backup_settings_once():
    backup_path = SETTINGS_FILE + ".bak"

    if os.path.exists(backup_path):
        print(f"[INFO] settings backup already exists: {backup_path}")
        return

    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[INFO] Created backup: {backup_path}")


def replace_assignment(content, variable_name, value_repr):
    pattern = rf"^({re.escape(variable_name)}\s*=\s*)(.+?)$"
    replacement = rf"\g<1>{value_repr}"

    new_content, count = re.subn(
        pattern,
        replacement,
        content,
        count=1,
        flags=re.MULTILINE,
    )

    if count != 1:
        raise ValueError(f"settings.py 中找不到變數：{variable_name}")

    return new_content


def patch_settings(reg_sensing_weight, rob_sensing_weight):
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    content = replace_assignment(
        content,
        "REG_SENSING_LOSS_WEIGHT",
        f"{float(reg_sensing_weight):.5g}",
    )

    content = replace_assignment(
        content,
        "ROB_SENSING_LOSS_WEIGHT",
        f"{float(rob_sensing_weight):.5g}",
    )

    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def run_command(command, log_file):
    cmd_str = " ".join(command)

    header = (
        "\n"
        "====================================================\n"
        f"[{timestamp()}] RUN: {cmd_str}\n"
        "====================================================\n"
    )

    print(header, end="")
    log_file.write(header)
    log_file.flush()

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="strict",
        bufsize=1,
        env=env,
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")
        log_file.write(line)
        log_file.flush()

    process.wait()

    footer = (
        "\n"
        f"[{timestamp()}] EXIT CODE: {process.returncode}\n"
        "====================================================\n"
    )

    print(footer, end="")
    log_file.write(footer)
    log_file.flush()

    if process.returncode != 0:
        raise RuntimeError(
            f"指令失敗：{cmd_str}\n"
            f"exit code = {process.returncode}"
        )

def get_project_dir_from_settings_output():
    """
    直接執行 settings.py 後，settings.py 自己會建立資料夾。
    這裡不解析 PROJECT_DIR，run_log 統一放在目前資料夾底下的 sweep_logs。
    """
    log_dir = "sweep_logs"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def run_one_experiment(exp):
    exp_name = exp["name"]
    reg_w = exp["reg_sensing_weight"]
    rob_w = exp["rob_sensing_weight"]

    print("\n")
    print("####################################################")
    print(f"[SWEEP] Experiment: {exp_name}")
    print(f"[SWEEP] REG_SENSING_LOSS_WEIGHT = {reg_w}")
    print(f"[SWEEP] ROB_SENSING_LOSS_WEIGHT = {rob_w}")
    print("####################################################")

    patch_settings(
        reg_sensing_weight=reg_w,
        rob_sensing_weight=rob_w,
    )

    log_dir = get_project_dir_from_settings_output()
    log_path = os.path.join(log_dir, f"{exp_name}_run_log.txt")

    pipeline = [
        [sys.executable, SCRIPT_SETTINGS],
        [sys.executable, SCRIPT_RICIAN],
        [sys.executable, SCRIPT_MAIN_LT],
        [sys.executable, SCRIPT_MAIN_ST],
    ]

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n\n")
        log_file.write("####################################################\n")
        log_file.write(f"[{timestamp()}] START EXPERIMENT: {exp_name}\n")
        log_file.write(f"REG_SENSING_LOSS_WEIGHT = {reg_w}\n")
        log_file.write(f"ROB_SENSING_LOSS_WEIGHT = {rob_w}\n")
        log_file.write("####################################################\n")

        for command in pipeline:
            run_command(command, log_file)

        log_file.write("\n")
        log_file.write("####################################################\n")
        log_file.write(f"[{timestamp()}] FINISH EXPERIMENT: {exp_name}\n")
        log_file.write("####################################################\n")

    print(f"[SWEEP] Finished experiment: {exp_name}")
    print(f"[SWEEP] Log saved to: {log_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    ensure_file_exists(SETTINGS_FILE)
    ensure_file_exists(SCRIPT_RICIAN)
    ensure_file_exists(SCRIPT_MAIN_LT)
    ensure_file_exists(SCRIPT_MAIN_ST)

    backup_settings_once()

    print("====================================================")
    print("[SWEEP] ISAC9 simple paired penalty sweep")
    print("----------------------------------------------------")

    for i, exp in enumerate(EXPERIMENTS, start=1):
        print(
            f"[{i}] {exp['name']} | "
            f"REG={exp['reg_sensing_weight']} | "
            f"ROB={exp['rob_sensing_weight']}"
        )

    print("====================================================")

    for exp in EXPERIMENTS:
        run_one_experiment(exp)

    print("====================================================")
    print("[SWEEP] All experiments finished.")
    print("====================================================")