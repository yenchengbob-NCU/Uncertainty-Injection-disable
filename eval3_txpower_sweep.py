# -*- coding: utf-8 -*-
import os
import csv
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import LongTermPositionNet, ShortTermCommNet, ShortTermRadarNet


# ================================
# TX power sweep setting
# ================================
BEST_REG_PENALTY = 100.0
BEST_ROB_PENALTY = 0.5

TX_POWER_DBM_LIST = [0, 5, 10, 15, 20, 25, 30]

INJECTION_CASES = [
    ("ROB inj=0", "rob", BEST_ROB_PENALTY, 0.0),
    ("REG inj=0", "reg", BEST_REG_PENALTY, 0.0),
    ("ROB inj=0.075", "rob", BEST_ROB_PENALTY, 0.075),
    ("REG inj=0.075", "reg", BEST_REG_PENALTY, 0.075),
]

EVAL_INJECTION_CHUNK = 50

# SNR 圖預設畫 lower-tail sensing SNR。
# 若你想改成平均 sensing SNR，可把 "q05_snr_db" 改成 "mean_snr_db"。
SNR_PLOT_KEY = "q05_snr_db"


# ================================
# Path helpers
# ================================
def format_float_for_path(value: float) -> str:
    """
    將 float 轉成資料夾名稱格式。

    例：
        0.5   -> 0p5
        100.0 -> 100
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    return str(value).replace(".", "p").replace("-", "m")


def dbm_to_watt(dbm: float) -> float:
    """
    dBm 轉 W。

    P[W] = 10^((P[dBm] - 30) / 10)
    """
    return 10.0 ** ((float(dbm) - 30.0) / 10.0)


def build_st_paths(mode: str, penalty: float) -> dict:
    """
    根據 mode 與 penalty 建立 ST checkpoint 路徑。
    """
    mode = mode.lower()
    penalty_tag = format_float_for_path(penalty)

    if mode == "reg":
        run_name = f"REG_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm.ckpt"
        radar_ckpt_name = "short_radar.ckpt"
    elif mode == "rob":
        run_name = f"ROB_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm_robust.ckpt"
        radar_ckpt_name = "short_radar_robust.ckpt"
    else:
        raise ValueError(f"mode must be reg or rob, got {mode}")

    run_dir = os.path.join(ST_SWEEP_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")

    return {
        "run_dir": run_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, comm_ckpt_name),
        "radar_ckpt_path": os.path.join(ckpt_dir, radar_ckpt_name),
    }


def build_eval_dir() -> str:
    """
    建立 TX power sweep 輸出資料夾。
    """
    eval_dir = os.path.join(
        BASE_RUN_DIR,
        "eval_results",
        "eval3_txpower_sweep"
    )

    os.makedirs(eval_dir, exist_ok=True)
    return eval_dir


# ================================
# Tensor helpers
# ================================
def np_to_torch_complex(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.complex64).to(DEVICE)


def np_to_torch_float(x_np: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x_np).to(torch.float32).to(DEVICE)


def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0, variance): E|n|^2 = variance
    Re/Im ~ N(0, variance/2)
    """
    if variance == 0.0:
        return torch.zeros(shape, device=device, dtype=cdtype)

    sigma = math.sqrt(variance / 2.0)

    nr = torch.randn(
        shape,
        device=device,
        dtype=torch.float32
    ) * sigma

    ni = torch.randn(
        shape,
        device=device,
        dtype=torch.float32
    ) * sigma

    return torch.complex(nr, ni).to(dtype=cdtype)


def reset_eval_seed():
    """
    每個 case / TX power eval 前重設 seed。
    讓不同模型在同一個 injection variance 下使用相同擾動序列。
    """
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


# ================================
# Dataset
# ================================
def load_test_dataset(npz_path: str):
    """
    載入 test dataset。
    """
    keys = [
        "ue_layouts",
        "pl_BS_UE",
        "pl_BS_RIS_UE",
        "pl_BS_TAR_BS",
        "st_h_dk_hat",
        "st_h_rk_hat",
        "st_G_hat",
        "st_g_dt_hat",
    ]

    with np.load(npz_path) as data:
        dataset = {key: data[key] for key in keys}

    print(f"[TEST] loaded: {npz_path}")
    print(f"[TEST] #layouts = {dataset['ue_layouts'].shape[0]}")
    print(f"[TEST] st_h_dk_hat shape = {dataset['st_h_dk_hat'].shape}")
    print(f"[TEST] st_h_rk_hat shape = {dataset['st_h_rk_hat'].shape}")
    print(f"[TEST] st_G_hat shape    = {dataset['st_G_hat'].shape}")
    print(f"[TEST] st_g_dt_hat shape = {dataset['st_g_dt_hat'].shape}")

    return dataset


def get_fixed_theta_from_longterm(
    longterm_net: LongTermPositionNet,
    ue_layout_np: np.ndarray
):
    """
    給一組 layout，從 shared LT net 取固定 theta_LT。
    """
    longterm_net.eval()

    with torch.no_grad():
        layout_t = np_to_torch_float(ue_layout_np).unsqueeze(0)
        theta_lt, _, _ = longterm_net(layout_t)

    return theta_lt.detach()


def extract_shortterm_batch(
    dataset,
    layout_id: int,
    channel_ids: np.ndarray
):
    """
    取出指定 layout / channels。
    """
    h_dk_hat = np_to_torch_complex(
        dataset["st_h_dk_hat"][layout_id][channel_ids]
    )

    h_rk_hat = np_to_torch_complex(
        dataset["st_h_rk_hat"][layout_id][channel_ids]
    )

    G_hat = np_to_torch_complex(
        dataset["st_G_hat"][layout_id][channel_ids]
    )

    g_dt_hat = np_to_torch_complex(
        dataset["st_g_dt_hat"][layout_id][channel_ids]
    )

    return {
        "pl_BS_UE": dataset["pl_BS_UE"][layout_id],
        "pl_BS_RIS_UE": dataset["pl_BS_RIS_UE"][layout_id],
        "pl_BS_TAR_BS": dataset["pl_BS_TAR_BS"][layout_id],
        "h_dk_hat": h_dk_hat,
        "h_rk_hat": h_rk_hat,
        "G_hat": G_hat,
        "g_dt_hat": g_dt_hat,
    }


# ================================
# Model loading
# ================================
def load_longterm_model():
    """
    載入 shared LT model。
    """
    print(f"[LOAD] LONGTERM_CKPT_PATH = {LONGTERM_CKPT_PATH}")

    longterm_net = LongTermPositionNet(
        ckpt_kind=None
    ).to(DEVICE)

    longterm_net.load_model(
        path=LONGTERM_CKPT_PATH,
        verbose=True
    )

    longterm_net.eval()

    for p in longterm_net.parameters():
        p.requires_grad_(False)

    return longterm_net


def load_shortterm_model(mode: str, penalty: float):
    """
    載入指定 mode / penalty 的 ST comm/radar nets。
    """
    paths = build_st_paths(mode, penalty)

    comm_net = ShortTermCommNet(
        ckpt_kind=None
    ).to(DEVICE)

    radar_net = ShortTermRadarNet(
        ckpt_kind=None
    ).to(DEVICE)

    comm_net.load_model(
        path=paths["comm_ckpt_path"],
        verbose=True
    )

    radar_net.load_model(
        path=paths["radar_ckpt_path"],
        verbose=True
    )

    comm_net.eval()
    radar_net.eval()

    for p in comm_net.parameters():
        p.requires_grad_(False)

    for p in radar_net.parameters():
        p.requires_grad_(False)

    return comm_net, radar_net


# ================================
# Evaluation core
# ================================
@torch.no_grad()
def eval_one_batch_txpower(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
    tx_power_watt: float,
    injection_variance: float,
):
    """
    對一個 estimated-channel batch 做 TX power sweep evaluation。

    流程：
        1. ST net 根據 estimated channel 輸出 W_C, W_R。
        2. 先 normalize 到 settings.TRANSMIT_POWER_TOTAL。
        3. 再縮放到目前 sweep 的 tx_power_watt。
        4. 在指定 injection_variance 下計算 sum-rate 與 sensing SNR。

    回傳：
        sumrate_np : shape = (B * L_eff,)
        snr_db_np  : shape = (B * L_eff,)
    """
    h_dk_hat = batch_data["h_dk_hat"]
    h_rk_hat = batch_data["h_rk_hat"]
    G_hat = batch_data["G_hat"]
    g_dt_hat = batch_data["g_dt_hat"]

    pl_BS_UE = batch_data["pl_BS_UE"]
    pl_BS_RIS_UE = batch_data["pl_BS_RIS_UE"]
    pl_BS_TAR_BS = batch_data["pl_BS_TAR_BS"]

    B, M, K = h_dk_hat.shape
    N = h_rk_hat.shape[1]

    if injection_variance == 0.0:
        L = 1
    else:
        L = int(INJECTION_SAMPLES)

    theta_batch = theta_fixed.expand(B, RIS_UNIT)

    W_C = comm_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )

    W_R = radar_net(
        h_dk_hat,
        h_rk_hat,
        G_hat,
        g_dt_hat,
        theta_batch
    )

    # 先正規化到 settings.TRANSMIT_POWER_TOTAL
    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)

    # 再縮放到目標 TX power
    tx_scale = math.sqrt(float(tx_power_watt) / float(TRANSMIT_POWER_TOTAL))
    W_C = W_C * tx_scale
    W_R = W_R * tx_scale

    sumrate_chunks = []
    snr_db_chunks = []

    for s0 in range(0, L, EVAL_INJECTION_CHUNK):
        s = min(EVAL_INJECTION_CHUNK, L - s0)

        h_dk_rep = h_dk_hat.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        h_rk_rep = h_rk_hat.unsqueeze(1).expand(
            B, s, N, K
        ).reshape(B * s, N, K)

        G_rep = G_hat.unsqueeze(1).expand(
            B, s, N, M
        ).reshape(B * s, N, M)

        g_dt_rep = g_dt_hat.unsqueeze(1).expand(
            B, s, M, 1
        ).reshape(B * s, M, 1)

        h_dk_inj = h_dk_rep + complex_awgn(
            h_dk_rep.shape,
            injection_variance,
            DEVICE,
            h_dk_rep.dtype
        )

        h_rk_inj = h_rk_rep + complex_awgn(
            h_rk_rep.shape,
            injection_variance,
            DEVICE,
            h_rk_rep.dtype
        )

        G_inj = G_rep + complex_awgn(
            G_rep.shape,
            injection_variance,
            DEVICE,
            G_rep.dtype
        )

        g_dt_inj = g_dt_rep + complex_awgn(
            g_dt_rep.shape,
            injection_variance,
            DEVICE,
            g_dt_rep.dtype
        )

        theta_rep = theta_batch.unsqueeze(1).expand(
            B, s, N
        ).reshape(B * s, N)

        W_C_rep = W_C.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        W_R_rep = W_R.unsqueeze(1).expand(
            B, s, M, W_R.shape[2]
        ).reshape(B * s, M, W_R.shape[2])

        sinrs = comm_net.compute_comm_sinrs(
            h_dk=h_dk_inj,
            h_rk=h_rk_inj,
            G=G_inj,
            theta=theta_rep,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_UE=pl_BS_UE,
            pl_BS_RIS_UE=pl_BS_RIS_UE,
        )

        rates = comm_net.compute_rates(sinrs)
        sumrate = rates.sum(dim=1)

        sense_snr = comm_net.compute_sense_snr(
            g_dt=g_dt_inj,
            W_R=W_R_rep,
            W_C=W_C_rep,
            pl_BS_TAR_BS=pl_BS_TAR_BS,
        ).real

        sense_snr_db = 10.0 * torch.log10(
            sense_snr.clamp_min(1e-12)
        )

        sumrate_chunks.append(
            sumrate.detach().cpu().numpy().astype(np.float32)
        )

        snr_db_chunks.append(
            sense_snr_db.detach().cpu().numpy().astype(np.float32)
        )

    sumrate_np = np.concatenate(sumrate_chunks, axis=0)
    snr_db_np = np.concatenate(snr_db_chunks, axis=0)

    return sumrate_np, snr_db_np


@torch.no_grad()
def evaluate_case_txpower(
    label: str,
    mode: str,
    penalty: float,
    injection_variance: float,
    tx_power_dbm: float,
    tx_power_watt: float,
    longterm_net: LongTermPositionNet,
    test_dataset,
):
    """
    評估單一 case 在單一 TX power 下的表現。
    """
    print(
        f"[EVAL3] {label} | mode={mode.upper()} | "
        f"penalty={penalty} | inj={injection_variance} | "
        f"TX={tx_power_dbm} dBm"
    )

    reset_eval_seed()

    comm_net, radar_net = load_shortterm_model(
        mode=mode,
        penalty=penalty
    )

    n_layouts = test_dataset["ue_layouts"].shape[0]
    n_eval_channels = test_dataset["st_h_dk_hat"].shape[1]
    channel_ids = np.arange(n_eval_channels)

    sumrate_all = []
    snr_db_all = []

    for layout_id in range(n_layouts):
        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net=longterm_net,
            ue_layout_np=test_dataset["ue_layouts"][layout_id]
        )

        batch_data = extract_shortterm_batch(
            dataset=test_dataset,
            layout_id=layout_id,
            channel_ids=channel_ids
        )

        sumrate_np, snr_db_np = eval_one_batch_txpower(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            tx_power_watt=tx_power_watt,
            injection_variance=injection_variance,
        )

        sumrate_all.append(sumrate_np)
        snr_db_all.append(snr_db_np)

    sumrate_all = np.concatenate(sumrate_all, axis=0)
    snr_db_all = np.concatenate(snr_db_all, axis=0)

    mean_sumrate = float(np.mean(sumrate_all))
    q05_sumrate = float(np.quantile(sumrate_all, OUTAGE_QUANTILE))

    mean_snr_db = float(np.mean(snr_db_all))
    q05_snr_db = float(np.quantile(snr_db_all, OUTAGE_QUANTILE))
    p_out = float(np.mean(snr_db_all < SENSING_SNR_THRESHOLD_dB))

    result = {
        "label": label,
        "mode": mode.upper(),
        "penalty": float(penalty),
        "injection_variance": float(injection_variance),
        "tx_power_dbm": float(tx_power_dbm),
        "tx_power_watt": float(tx_power_watt),
        "mean_sumrate": mean_sumrate,
        "q05_sumrate": q05_sumrate,
        "mean_snr_db": mean_snr_db,
        "q05_snr_db": q05_snr_db,
        "p_out": p_out,
        "num_layouts": int(n_layouts),
        "channels_per_layout": int(n_eval_channels),
        "injection_samples": int(INJECTION_SAMPLES if injection_variance > 0.0 else 1),
    }

    print(
        f"[RESULT] {label} | TX={tx_power_dbm:>4.1f} dBm | "
        f"MeanRate={mean_sumrate:.6f} | "
        f"Q0.05Rate={q05_sumrate:.6f} | "
        f"MeanSNR={mean_snr_db:.3f} dB | "
        f"Q0.05SNR={q05_snr_db:.3f} dB | "
        f"Pout={100.0 * p_out:.3f}%"
    )

    return result


# ================================
# Save and plot
# ================================
def save_csv(rows: list[dict], csv_path: str):
    """
    儲存 CSV。
    """
    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[SAVE] CSV saved: {csv_path}")


def plot_txpower_metric(
    rows: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    fig_path: str,
):
    """
    畫 TX power sweep metric。
    """
    plt.figure(figsize=(9, 5.5))

    for label, _, _, _ in INJECTION_CASES:
        case_rows = [
            r for r in rows
            if r["label"] == label
        ]

        case_rows = sorted(
            case_rows,
            key=lambda r: r["tx_power_dbm"]
        )

        x = np.array(
            [r["tx_power_dbm"] for r in case_rows],
            dtype=np.float64
        )

        y = np.array(
            [r[metric_key] for r in case_rows],
            dtype=np.float64
        )

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=label
        )

    plt.xlabel("Transmit Power (dBm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        fig_path,
        dpi=300,
        format="jpg"
    )

    plt.close()

    print(f"[SAVE] Figure saved: {fig_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    print("[INFO] Eval3 TX power sweep started.")
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] BASE_RUN_DIR = {BASE_RUN_DIR}")
    print(f"[INFO] TEST_DATASET_PATH = {TEST_DATASET_PATH}")
    print(f"[INFO] LONGTERM_CKPT_PATH = {LONGTERM_CKPT_PATH}")
    print(f"[INFO] ST_SWEEP_DIR = {ST_SWEEP_DIR}")

    print("\n[INFO] Sweep setting:")
    print(f"[INFO] Best REG penalty = {BEST_REG_PENALTY}")
    print(f"[INFO] Best ROB penalty = {BEST_ROB_PENALTY}")
    print(f"[INFO] TX_POWER_DBM_LIST = {TX_POWER_DBM_LIST}")
    print(f"[INFO] INJECTION_CASES = {INJECTION_CASES}")
    print(f"[INFO] SNR_PLOT_KEY = {SNR_PLOT_KEY}")

    eval_dir = build_eval_dir()
    print(f"\n[INFO] EVAL_DIR = {eval_dir}")

    test_dataset = load_test_dataset(TEST_DATASET_PATH)
    longterm_net = load_longterm_model()

    rows = []

    for tx_dbm in TX_POWER_DBM_LIST:
        tx_watt = dbm_to_watt(tx_dbm)

        for label, mode, penalty, inj_var in INJECTION_CASES:
            result = evaluate_case_txpower(
                label=label,
                mode=mode,
                penalty=penalty,
                injection_variance=inj_var,
                tx_power_dbm=tx_dbm,
                tx_power_watt=tx_watt,
                longterm_net=longterm_net,
                test_dataset=test_dataset,
            )

            rows.append(result)

    csv_path = os.path.join(
        eval_dir,
        "txpower_sweep_metrics.csv"
    )

    save_csv(
        rows=rows,
        csv_path=csv_path
    )

    npz_path = os.path.join(
        eval_dir,
        "txpower_sweep_metrics.npz"
    )

    np.savez_compressed(
        npz_path,
        rows=np.array(
            [
                (
                    r["label"],
                    r["mode"],
                    r["penalty"],
                    r["injection_variance"],
                    r["tx_power_dbm"],
                    r["tx_power_watt"],
                    r["mean_sumrate"],
                    r["q05_sumrate"],
                    r["mean_snr_db"],
                    r["q05_snr_db"],
                    r["p_out"],
                )
                for r in rows
            ],
            dtype=[
                ("label", "U32"),
                ("mode", "U8"),
                ("penalty", "f8"),
                ("injection_variance", "f8"),
                ("tx_power_dbm", "f8"),
                ("tx_power_watt", "f8"),
                ("mean_sumrate", "f8"),
                ("q05_sumrate", "f8"),
                ("mean_snr_db", "f8"),
                ("q05_snr_db", "f8"),
                ("p_out", "f8"),
            ]
        )
    )

    print(f"[SAVE] NPZ saved: {npz_path}")

    snr_fig_path = os.path.join(
        eval_dir,
        f"txpower_sweep_{SNR_PLOT_KEY}.jpg"
    )

    rate_fig_path = os.path.join(
        eval_dir,
        "txpower_sweep_mean_sumrate.jpg"
    )

    if SNR_PLOT_KEY == "q05_snr_db":
        snr_ylabel = "Q0.05 Sensing SNR (dB)"
        snr_title = "Transmit Power vs Q0.05 Sensing SNR"
    elif SNR_PLOT_KEY == "mean_snr_db":
        snr_ylabel = "Mean Sensing SNR (dB)"
        snr_title = "Transmit Power vs Mean Sensing SNR"
    else:
        raise ValueError("SNR_PLOT_KEY must be q05_snr_db or mean_snr_db")

    plot_txpower_metric(
        rows=rows,
        metric_key=SNR_PLOT_KEY,
        ylabel=snr_ylabel,
        title=snr_title,
        fig_path=snr_fig_path
    )

    plot_txpower_metric(
        rows=rows,
        metric_key="mean_sumrate",
        ylabel="Mean Sum-Rate (bits/s/Hz)",
        title="Transmit Power vs Mean Sum-Rate",
        fig_path=rate_fig_path
    )

    print("\n" + "=" * 80)
    print("[INFO] Eval3 TX power sweep finished.")
    print("=" * 80)

    print(f"[INFO] CSV        : {csv_path}")
    print(f"[INFO] NPZ        : {npz_path}")
    print(f"[INFO] SNR figure : {snr_fig_path}")
    print(f"[INFO] Rate figure: {rate_fig_path}")