# -*- coding: utf-8 -*-
"""
eval5.py

用途：
    事後整理 ISAC penalty sweep 結果。

功能：
    1. 檢查所有指定 penalty 的 npz 是否存在。
    2. 對 INJERR=0.075：
        - 畫 REG 不同 penalty 的 SNR CDF
        - 畫 ROB 不同 penalty 的 SNR CDF
        - 選出 Q0.05(SNR) 最接近 15 dB 的 REG / ROB
        - 畫這兩者的 injected SumRate CDF
    3. 對 INJERR=0.095 重複上述流程。

重要：
    - 不重新載入 NN
    - 不重新跑 injection
    - 不重新 evaluate
    - 只讀取 eval.py 產生的 four_values_metrics_test_L50_C500_INJ200.npz
"""

import os
import re
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt


# ================================
# User config
# ================================
ROOT_DIR = os.path.join("MLP", "M6_Ris40_K2")

THRESHOLD_DB = 15
OUTAGE_TARGET = 0.05

N_TEST_LAYOUTS = 50
SHORTTERM_EST_CHANNELS_PER_LAYOUT = 500
INJECTION_SAMPLES = 200
EXPECTED_SNR_SAMPLES = (
    N_TEST_LAYOUTS
    * SHORTTERM_EST_CHANNELS_PER_LAYOUT
    * INJECTION_SAMPLES
)

METRICS_FILENAME = "four_values_metrics_test_L50_C500_INJ200.npz"

OUTPUT_DIR = os.path.join(ROOT_DIR, "eval5_sweep_summary")

# 如果缺檔就中止
STRICT_CHECK = True

# 是否顯示圖片視窗
SHOW_FIGURES = True

SWEEP_CONFIGS = {
    0.075: {
        "reg_weights": [100.0, 150.0, 200.0],
        "rob_weights": [0.5, 1.0, 5.0],
    },
    0.095: {
        "reg_weights": [30.0, 50.0, 70.0, 150.0],
        "rob_weights": [1.0, 2.0, 5.0, 10.0],
    },
}


# ================================
# Helpers
# ================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def fmt_float(x: float) -> str:
    """
    用於 label 顯示。
    例如 1.0 -> 1, 0.5 -> 0.5
    """
    x = float(x)
    if abs(x - int(x)) < 1e-12:
        return str(int(x))
    return str(x)


def fmt_injerr(x: float) -> str:
    return f"{float(x):.3f}"


def snr_linear_to_db(snr_linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(snr_linear, 1e-12))


def empirical_cdf(x: np.ndarray):
    x_sorted = np.sort(x)
    y = np.arange(1, x_sorted.size + 1, dtype=np.float64) / x_sorted.size
    return x_sorted, y


def parse_thr_tag(thr_tag: str):
    """
    解析：
        THR_15db_punish_(30.0, 2.0, 250.0)

    回傳：
        threshold_db, reg_w, rob_w, re_w

    若格式不是三個權重，回傳 None。
    """
    pattern = (
        r"^THR_(?P<thr>\d+)db_punish_"
        r"\(\s*(?P<reg>[-+]?\d*\.?\d+)\s*,\s*"
        r"(?P<rob>[-+]?\d*\.?\d+)\s*,\s*"
        r"(?P<re>[-+]?\d*\.?\d+)\s*\)$"
    )

    m = re.match(pattern, thr_tag)

    if m is None:
        return None

    return {
        "threshold_db": int(m.group("thr")),
        "reg_w": float(m.group("reg")),
        "rob_w": float(m.group("rob")),
        "re_w": float(m.group("re")),
    }


def scan_experiments_for_injerr(injerr: float):
    """
    掃描 ROOT_DIR 底下所有符合：
        THR_15db_punish_(REG, ROB, RE)/
            N0_1e-10_INJERR_xxx_TXpower_1.0/
                eval_figures/
                    four_values_metrics_test_L50_C500_INJ200.npz

    回傳：
        list of experiment dict
    """
    setting_tag = f"N0_1e-10_INJERR_{fmt_injerr(injerr)}_TXpower_1.0"
    pattern = os.path.join(
        ROOT_DIR,
        "THR_*db_punish_*",
        setting_tag,
        "eval_figures",
        METRICS_FILENAME,
    )

    paths = sorted(glob.glob(pattern))

    experiments = []

    for metrics_path in paths:
        setting_dir = os.path.dirname(os.path.dirname(metrics_path))
        thr_dir = os.path.basename(os.path.dirname(setting_dir))

        parsed = parse_thr_tag(thr_dir)

        if parsed is None:
            print(f"[WARN] 跳過無法解析的資料夾：{thr_dir}")
            continue

        if parsed["threshold_db"] != THRESHOLD_DB:
            continue

        experiments.append({
            "injerr": float(injerr),
            "threshold_db": parsed["threshold_db"],
            "reg_w": parsed["reg_w"],
            "rob_w": parsed["rob_w"],
            "re_w": parsed["re_w"],
            "thr_tag": thr_dir,
            "metrics_path": metrics_path,
        })

    return experiments


def select_unique_by_method_weight(
    experiments: list[dict],
    method: str,
    target_weights: list[float],
):
    """
    從掃描到的 experiments 中挑出指定 method 的 target weights。

    method:
        "REG" -> 用 exp["reg_w"] 比對
        "ROB" -> 用 exp["rob_w"] 比對

    如果同一個 weight 有多個 npz，使用第一個並警告。
    """
    assert method in ["REG", "ROB"]

    key = "reg_w" if method == "REG" else "rob_w"

    selected = []
    missing = []

    for target_w in target_weights:
        candidates = [
            exp for exp in experiments
            if abs(float(exp[key]) - float(target_w)) < 1e-9
        ]

        if len(candidates) == 0:
            missing.append(target_w)
            continue

        if len(candidates) > 1:
            print("====================================================")
            print(f"[WARN] {method} λ={target_w} 找到多個 npz，使用第一個：")
            for c in candidates:
                print(f"  {c['metrics_path']}")
            print("====================================================")

        chosen = candidates[0].copy()
        chosen["method"] = method
        chosen["penalty_weight"] = float(target_w)

        selected.append(chosen)

    return selected, missing


def load_arrays(metrics_path: str):
    """
    讀取 eval.py 產生的 npz。
    """
    with np.load(metrics_path) as data:
        arrays = {
            "reg_snr_db": snr_linear_to_db(data["reg_snr_raw_all"]),
            "rob_snr_db": snr_linear_to_db(data["rob_snr_raw_all"]),
            "reg_sumrate": data["reg_sumrate_raw_all"].astype(np.float32),
            "rob_sumrate": data["rob_sumrate_raw_all"].astype(np.float32),
        }

    return arrays


def get_method_arrays(exp: dict):
    """
    根據 method 取出對應 REG 或 ROB 的 SNR / SumRate。
    """
    arrays = load_arrays(exp["metrics_path"])

    if exp["method"] == "REG":
        snr_db = arrays["reg_snr_db"]
        sumrate = arrays["reg_sumrate"]
    elif exp["method"] == "ROB":
        snr_db = arrays["rob_snr_db"]
        sumrate = arrays["rob_sumrate"]
    else:
        raise ValueError(f"未知 method：{exp['method']}")

    return snr_db, sumrate


def summarize_entry(exp: dict):
    """
    計算單一 curve 的 summary。

    違反機率定義：
        violation_count / (50 * 500 * 200)
    """
    snr_db, sumrate = get_method_arrays(exp)

    violation_count = int(np.sum(snr_db < THRESHOLD_DB))
    violation_prob = violation_count / EXPECTED_SNR_SAMPLES

    if snr_db.size != EXPECTED_SNR_SAMPLES:
        print("====================================================")
        print("[WARN] SNR sample 數量與 EXPECTED_SNR_SAMPLES 不一致")
        print(f"  path       = {exp['metrics_path']}")
        print(f"  actual     = {snr_db.size:,}")
        print(f"  expected   = {EXPECTED_SNR_SAMPLES:,}")
        print("  violation_prob 仍使用 expected denominator 計算。")
        print("====================================================")

    q05_snr_db = float(np.quantile(snr_db, OUTAGE_TARGET))

    mean_snr_linear = float(np.mean(10.0 ** (snr_db / 10.0)))
    mean_snr_db = float(10.0 * np.log10(max(mean_snr_linear, 1e-12)))

    return {
        "method": exp["method"],
        "penalty_weight": exp["penalty_weight"],
        "injerr": exp["injerr"],
        "threshold_db": THRESHOLD_DB,
        "metrics_path": exp["metrics_path"],

        "snr_samples": int(snr_db.size),
        "violation_count": violation_count,
        "violation_prob": violation_prob,
        "violation_percent": violation_prob * 100.0,

        "q05_snr_db": q05_snr_db,
        "mean_snr_db": mean_snr_db,

        "sumrate_samples": int(sumrate.size),
        "mean_sumrate": float(np.mean(sumrate)),
        "median_sumrate": float(np.quantile(sumrate, 0.50)),
    }


def save_summary_csv(rows: list[dict], csv_path: str):
    fieldnames = [
        "injerr",
        "method",
        "penalty_weight",
        "threshold_db",

        "snr_samples",
        "violation_count",
        "violation_prob",
        "violation_percent",

        "q05_snr_db",
        "q05_distance",
        "mean_snr_db",

        "sumrate_samples",
        "mean_sumrate",
        "median_sumrate",

        "metrics_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[EVAL5] Saved CSV: {csv_path}")


def print_check_result(injerr: float, reg_selected, reg_missing, rob_selected, rob_missing):
    print("====================================================")
    print(f"[CHECK] INJERR = {fmt_injerr(injerr)}")

    print("[REG npz]")
    for exp in reg_selected:
        print(
            f"  REG λ={fmt_float(exp['penalty_weight'])} | "
            f"folder REG={exp['reg_w']}, ROB={exp['rob_w']} | "
            f"{exp['metrics_path']}"
        )

    if len(reg_missing) > 0:
        print("[REG missing]")
        for w in reg_missing:
            print(f"  REG λ={fmt_float(w)}")

    print("[ROB npz]")
    for exp in rob_selected:
        print(
            f"  ROB λ={fmt_float(exp['penalty_weight'])} | "
            f"folder REG={exp['reg_w']}, ROB={exp['rob_w']} | "
            f"{exp['metrics_path']}"
        )

    if len(rob_missing) > 0:
        print("[ROB missing]")
        for w in rob_missing:
            print(f"  ROB λ={fmt_float(w)}")

    print("====================================================")


# ================================
# Plot functions
# ================================
def plot_snr_cdf(entries: list[dict], injerr: float, method: str):
    plt.figure(figsize=(9.2, 5.8))

    for exp in entries:
        snr_db, _ = get_method_arrays(exp)
        x, y = empirical_cdf(snr_db)

        plt.plot(
            x,
            y,
            linewidth=2.1,
            label=f"{method} λ={fmt_float(exp['penalty_weight'])}",
        )

    plt.axhline(
        y=OUTAGE_TARGET,
        linestyle=":",
        linewidth=2.2,
        color="black",
        label=f"Outage target = {OUTAGE_TARGET:.2f}",
    )

    plt.axvline(
        x=THRESHOLD_DB,
        linestyle="-.",
        linewidth=2.2,
        color="black",
        label=f"SNR threshold = {THRESHOLD_DB} dB",
    )

    plt.xlabel("Injected SNR sample (dB)")
    plt.ylabel("CDF  P(SNR ≤ x)")
    plt.title(
        f"{method} SNR CDF sweep — INJERR={fmt_injerr(injerr)}, THR={THRESHOLD_DB} dB"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"CDF_SNR_{method}_sweep_INJERR_{fmt_injerr(injerr)}_THR{THRESHOLD_DB}.jpg"
    )

    plt.savefig(save_path, format="jpg", dpi=300)

    if SHOW_FIGURES:
        plt.show()

    plt.close()

    print(f"[EVAL5] Saved SNR CDF: {save_path}")


def select_best_by_q05(entries: list[dict]):
    """
    選 Q0.05(SNR) 最接近 THRESHOLD_DB 的 entry。
    """
    summaries = []

    for exp in entries:
        summary = summarize_entry(exp)
        summary["exp"] = exp
        summary["q05_distance"] = abs(summary["q05_snr_db"] - THRESHOLD_DB)
        summaries.append(summary)

    best = min(summaries, key=lambda r: r["q05_distance"])
    return best, summaries


def plot_best_rate_cdf(injerr: float, best_reg: dict, best_rob: dict):
    reg_exp = best_reg["exp"]
    rob_exp = best_rob["exp"]

    _, reg_rate = get_method_arrays(reg_exp)
    _, rob_rate = get_method_arrays(rob_exp)

    reg_x, reg_y = empirical_cdf(reg_rate)
    rob_x, rob_y = empirical_cdf(rob_rate)

    plt.figure(figsize=(9.2, 5.8))

    plt.plot(
        reg_x,
        reg_y,
        linestyle="--",
        linewidth=2.2,
        label=f"REG λ={fmt_float(reg_exp['penalty_weight'])}",
    )

    plt.plot(
        rob_x,
        rob_y,
        linestyle="-",
        linewidth=2.6,
        label=f"ROB λ={fmt_float(rob_exp['penalty_weight'])}",
    )

    plt.xlabel("Injected SumRate sample (bits/s/Hz)")
    plt.ylabel("CDF  P(SumRate ≤ x)")
    plt.title(
        f"Rate CDF of selected penalties — INJERR={fmt_injerr(injerr)}, THR={THRESHOLD_DB} dB"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"CDF_Rate_selected_REG{fmt_float(reg_exp['penalty_weight'])}_"
        f"ROB{fmt_float(rob_exp['penalty_weight'])}_"
        f"INJERR_{fmt_injerr(injerr)}_THR{THRESHOLD_DB}.jpg"
    )

    plt.savefig(save_path, format="jpg", dpi=300)

    if SHOW_FIGURES:
        plt.show()

    plt.close()

    print(f"[EVAL5] Saved selected Rate CDF: {save_path}")


def print_best_result(injerr: float, best_reg: dict, best_rob: dict):
    print("====================================================")
    print(f"[BEST] INJERR = {fmt_injerr(injerr)}")
    print("Selection rule: minimum |Q0.05(SNR) - 15 dB|")
    print("----------------------------------------------------")

    print("[REG]")
    print(f"  penalty λ          = {fmt_float(best_reg['penalty_weight'])}")
    print(f"  Q0.05(SNR)         = {best_reg['q05_snr_db']:.3f} dB")
    print(f"  violation count    = {best_reg['violation_count']:,} / {EXPECTED_SNR_SAMPLES:,}")
    print(f"  violation prob     = {best_reg['violation_percent']:.3f} %")
    print(f"  mean SumRate       = {best_reg['mean_sumrate']:.6f} bits/s/Hz")

    print("[ROB]")
    print(f"  penalty λ          = {fmt_float(best_rob['penalty_weight'])}")
    print(f"  Q0.05(SNR)         = {best_rob['q05_snr_db']:.3f} dB")
    print(f"  violation count    = {best_rob['violation_count']:,} / {EXPECTED_SNR_SAMPLES:,}")
    print(f"  violation prob     = {best_rob['violation_percent']:.3f} %")
    print(f"  mean SumRate       = {best_rob['mean_sumrate']:.6f} bits/s/Hz")

    print("====================================================")


# ================================
# Main process
# ================================
def process_one_injerr(injerr: float, config: dict):
    experiments = scan_experiments_for_injerr(injerr)

    reg_selected, reg_missing = select_unique_by_method_weight(
        experiments=experiments,
        method="REG",
        target_weights=config["reg_weights"],
    )

    rob_selected, rob_missing = select_unique_by_method_weight(
        experiments=experiments,
        method="ROB",
        target_weights=config["rob_weights"],
    )

    print_check_result(
        injerr=injerr,
        reg_selected=reg_selected,
        reg_missing=reg_missing,
        rob_selected=rob_selected,
        rob_missing=rob_missing,
    )

    if STRICT_CHECK and (len(reg_missing) > 0 or len(rob_missing) > 0):
        raise FileNotFoundError(
            f"INJERR={fmt_injerr(injerr)} 缺少指定 npz，請先確認資料夾。"
        )

    if len(reg_selected) > 0:
        plot_snr_cdf(reg_selected, injerr, method="REG")

    if len(rob_selected) > 0:
        plot_snr_cdf(rob_selected, injerr, method="ROB")

    best_reg, reg_summaries = select_best_by_q05(reg_selected)
    best_rob, rob_summaries = select_best_by_q05(rob_selected)

    all_summaries = reg_summaries + rob_summaries

    csv_rows = []

    for s in all_summaries:
        row = {k: v for k, v in s.items() if k not in ["exp", "q05_distance"]}
        row["q05_distance"] = s["q05_distance"]
        csv_rows.append(row)

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"summary_INJERR_{fmt_injerr(injerr)}_THR{THRESHOLD_DB}.csv"
    )

    save_summary_csv(csv_rows, csv_path)

    print_best_result(injerr, best_reg, best_rob)

    plot_best_rate_cdf(injerr, best_reg, best_rob)


# ================================
# Entry point
# ================================
if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    print("====================================================")
    print("[EVAL5] ISAC sweep post-processing")
    print("[EVAL5] This script does NOT reload NN or rerun injections.")
    print(f"[EVAL5] ROOT_DIR = {ROOT_DIR}")
    print(f"[EVAL5] OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"[EVAL5] THRESHOLD_DB = {THRESHOLD_DB}")
    print(f"[EVAL5] EXPECTED_SNR_SAMPLES = {EXPECTED_SNR_SAMPLES:,}")
    print("====================================================")

    for injerr, config in SWEEP_CONFIGS.items():
        process_one_injerr(injerr, config)

    print("[EVAL5] Finished.")