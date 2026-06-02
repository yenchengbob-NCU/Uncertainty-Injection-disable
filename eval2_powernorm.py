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
# Eval2 settings
# ================================
FIXED_TEST_INJECTION_VARIANCE = 0.075
TEST_INJECTION_SAMPLES = INJECTION_SAMPLES

EVAL_INJECTION_CHUNK = 50

# 若要用全部 test estimated channels，設為 None。
EVAL_CHANNELS_PER_LAYOUT = None

REG_PENALTY_LIST = [70, 100, 150, 200, 300]
ROB_PENALTY_LIST = [0.025, 0.05, 0.1, 0.5, 1]

INJECTION_SWEEP_LIST = [ 0.0, 0.025, 0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250]

OUTAGE_PROB_THRESHOLD = 0.05
EVAL_SEED = RANDOM_SEED + 777

# ================================
# Injected-channel power normalization
# ================================
# True:
#   對 sweep evaluation 中的 injected channels 做 power normalization，
#   讓 h_hat + error 的 norm / Frobenius norm 回到 h_hat 的 power。
#   目的：避免 test injection variance 增大時，因 additive Gaussian error
#   額外增加 channel power，導致 mean sum-rate 反直覺上升。
#
# False:
#   保持原本 additive Gaussian injection，不做 normalization。
POWER_NORMALIZE_INJECTED_CHANNELS = True
POWER_NORM_EPS = 1e-12


# ================================
# Path helpers
# ================================
def format_float_for_path(value: float) -> str:
    """
    將 float 轉成與 main_st.py / eval.py 相同的資料夾與檔名格式。

    例：
        0.025 -> 0p025
        0.05  -> 0p05
        0.1   -> 0p1
        0.5   -> 0p5
        1.0   -> 1
        200.0 -> 200
    """
    value = float(value)

    if value.is_integer():
        return str(int(value))

    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "m")

    return text


def build_st_paths(mode: str, penalty: float) -> dict:
    """
    根據 mode 與 penalty 建立 ST checkpoint 路徑。
    需與 main_st.py 的命名一致。
    """
    mode = mode.lower()

    if mode not in ("reg", "rob"):
        raise ValueError(f"mode must be 'reg' or 'rob', got {mode}")

    penalty_tag = format_float_for_path(penalty)

    if mode == "reg":
        run_name = f"REG_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm.ckpt"
        radar_ckpt_name = "short_radar.ckpt"
    else:
        run_name = f"ROB_penalty_{penalty_tag}"
        comm_ckpt_name = "short_comm_robust.ckpt"
        radar_ckpt_name = "short_radar_robust.ckpt"

    run_dir = os.path.join(ST_SWEEP_DIR, run_name)
    ckpt_dir = os.path.join(run_dir, "ckpt")

    return {
        "run_dir": run_dir,
        "ckpt_dir": ckpt_dir,
        "comm_ckpt_path": os.path.join(ckpt_dir, comm_ckpt_name),
        "radar_ckpt_path": os.path.join(ckpt_dir, radar_ckpt_name),
    }


def build_eval1_dir() -> str:
    """
    eval.py 固定 test injection = 0.075 的 selection result 資料夾。
    """
    inj_tag = format_float_for_path(FIXED_TEST_INJECTION_VARIANCE)

    return os.path.join(
        BASE_RUN_DIR,
        "eval_results",
        f"selection_testinj_{inj_tag}",
    )


def build_eval2_dir() -> str:
    """
    eval2.py 輸出資料夾。
    """
    inj_tag = format_float_for_path(FIXED_TEST_INJECTION_VARIANCE)

    eval2_dir = os.path.join(
        BASE_RUN_DIR,
        "eval_results",
        f"eval2_best_sweep_tradeoff_testinj_{inj_tag}",
    )

    os.makedirs(eval2_dir, exist_ok=True)

    return eval2_dir


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



def normalize_injected_channel_power(
    injected: torch.Tensor,
    reference: torch.Tensor,
    norm_dims,
    eps: float = POWER_NORM_EPS,
) -> torch.Tensor:
    """
    將 injected channel 的 power 正規化回 reference channel。

    對 vector channel：
        h_inj_norm = h_inj * ||h_ref|| / ||h_inj||

    對 matrix channel：
        G_inj_norm = G_inj * ||G_ref||_F / ||G_inj||_F

    這裡的 norm_dims 決定要在哪些維度上計算 norm。
    第一個 batch 維度不應該被包含在 norm_dims 裡。

    目的：
        保留 estimated channel 的 large-scale power，
        讓 injection 主要代表 channel direction / phase perturbation，
        避免 additive Gaussian error 額外增加平均 channel power。
    """
    ref_norm = torch.linalg.vector_norm(
        reference,
        ord=2,
        dim=norm_dims,
        keepdim=True,
    )

    inj_norm = torch.linalg.vector_norm(
        injected,
        ord=2,
        dim=norm_dims,
        keepdim=True,
    ).clamp_min(eps)

    return injected * (ref_norm / inj_norm)


def apply_power_normalization_to_injected_channels(
    h_dk_inj: torch.Tensor,
    h_rk_inj: torch.Tensor,
    G_inj: torch.Tensor,
    g_dt_inj: torch.Tensor,
    h_dk_ref: torch.Tensor,
    h_rk_ref: torch.Tensor,
    G_ref: torch.Tensor,
    g_dt_ref: torch.Tensor,
):
    """
    對四種 injected channels 做 large-scale power preserving normalization。

    Shape convention:
        h_dk : (B*L, TX_ANT, K)
               每個 UE 的 direct channel vector 各自保留 norm。
        h_rk : (B*L, RIS_UNIT, K)
               每個 UE 的 RIS-user channel vector 各自保留 norm。
        G    : (B*L, RIS_UNIT, TX_ANT) 或目前程式中的 (B*L, N, M)
               每個 sample 的 BS-RIS matrix 保留 Frobenius norm。
        g_dt : (B*L, TX_ANT, 1) 或目前 dataset 中的 target channel vector
               每個 sample 的 sensing target vector 保留 norm。
    """
    if not POWER_NORMALIZE_INJECTED_CHANNELS:
        return h_dk_inj, h_rk_inj, G_inj, g_dt_inj

    # h_dk / h_rk：對每個 user k 的 channel vector 分別正規化。
    # shape = (batch, antenna_or_ris, K)，所以 norm_dims=1。
    h_dk_inj = normalize_injected_channel_power(
        injected=h_dk_inj,
        reference=h_dk_ref,
        norm_dims=1,
    )

    h_rk_inj = normalize_injected_channel_power(
        injected=h_rk_inj,
        reference=h_rk_ref,
        norm_dims=1,
    )

    # G：matrix channel，對每個 sample 的整個 matrix 做 Frobenius norm normalization。
    G_inj = normalize_injected_channel_power(
        injected=G_inj,
        reference=G_ref,
        norm_dims=(1, 2),
    )

    # g_dt：target channel vector，最後一維通常是 1，所以對 channel 維度一起取 norm。
    g_dt_inj = normalize_injected_channel_power(
        injected=g_dt_inj,
        reference=g_dt_ref,
        norm_dims=(1, 2),
    )

    return h_dk_inj, h_rk_inj, G_inj, g_dt_inj


def reset_eval_seed():
    """
    每次 model / injection variance eval 前重設 seed。

    目的：
        1. REG / ROB 在同一個 injection variance 下使用相同隨機擾動序列。
        2. 減少 Monte Carlo noise 對模型比較的影響。
    """
    np.random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(EVAL_SEED)


# ================================
# Load eval.py results
# ================================
def read_selection_table(csv_path: str) -> list[dict]:
    """
    讀取 eval.py 輸出的 selection table。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"找不到 selection table。請先執行 eval.py：{csv_path}"
        )

    rows = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            parsed = {
                "mode": row["mode"],
                "penalty": float(row["penalty"]),
                "mean_sumrate": float(row["mean_sumrate"]),
                "q05_snr_db": float(row["q05_snr_db"]),
                "p_out": float(row["p_out"]),
                "feasible": str(row["feasible"]).lower() in ("true", "1", "yes"),
                "run_dir": row.get("run_dir", ""),
            }

            rows.append(parsed)

    return rows


def select_best_feasible(rows: list[dict], mode: str):
    """
    在 P_out <= 5% 的模型中，選 mean sum-rate 最大者。
    """
    mode = mode.upper()

    candidates = [
        r for r in rows
        if r["mode"].upper() == mode and r["p_out"] <= OUTAGE_PROB_THRESHOLD
    ]

    if len(candidates) == 0:
        return None

    return max(
        candidates,
        key=lambda r: r["mean_sumrate"]
    )


def select_best_available(rows: list[dict], mode: str):
    """
    若沒有 feasible model，選 p_out 最小者。
    若 p_out 相同，選 mean_sumrate 最大者。
    """
    mode = mode.upper()

    candidates = [
        r for r in rows
        if r["mode"].upper() == mode
    ]

    if len(candidates) == 0:
        return None

    return min(
        candidates,
        key=lambda r: (r["p_out"], -r["mean_sumrate"])
    )


def load_eval1_npz(npz_path: str):
    """
    讀取 eval.py 儲存的 NPZ samples。
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"找不到 eval.py 的 npz 結果。請先執行 eval.py：{npz_path}"
        )

    return np.load(npz_path)


def get_fixed_samples_from_eval1(
    eval1_npz,
    mode: str,
    penalty: float,
):
    """
    從 eval.py 的 npz 取出指定 model 的 SNR / SumRate samples。
    """
    mode = mode.upper()
    penalty_tag = format_float_for_path(penalty)

    snr_key = f"snr_db_{mode}_penalty_{penalty_tag}"
    rate_key = f"sumrate_{mode}_penalty_{penalty_tag}"

    if snr_key not in eval1_npz:
        raise KeyError(f"eval1 npz 中找不到 key: {snr_key}")

    if rate_key not in eval1_npz:
        raise KeyError(f"eval1 npz 中找不到 key: {rate_key}")

    snr_db = eval1_npz[snr_key]
    sumrate = eval1_npz[rate_key]

    return snr_db, sumrate


# ================================
# Dataset
# ================================
def load_test_dataset(npz_path: str):
    """
    載入 test dataset。
    """
    required_keys = [
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
        dataset = {key: data[key] for key in required_keys}

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
    channel_ids: np.ndarray,
):
    """
    從 test dataset 取出指定 layout / channels 的 estimated channels。
    """
    ue_layout = dataset["ue_layouts"][layout_id]
    pl_BS_UE = dataset["pl_BS_UE"][layout_id]
    pl_BS_RIS_UE = dataset["pl_BS_RIS_UE"][layout_id]
    pl_BS_TAR_BS = dataset["pl_BS_TAR_BS"][layout_id]

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
        "ue_layout": ue_layout,
        "pl_BS_UE": pl_BS_UE,
        "pl_BS_RIS_UE": pl_BS_RIS_UE,
        "pl_BS_TAR_BS": pl_BS_TAR_BS,
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

    if not os.path.exists(paths["comm_ckpt_path"]):
        raise FileNotFoundError(
            f"找不到 comm checkpoint: {paths['comm_ckpt_path']}"
        )

    if not os.path.exists(paths["radar_ckpt_path"]):
        raise FileNotFoundError(
            f"找不到 radar checkpoint: {paths['radar_ckpt_path']}"
        )

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

    return comm_net, radar_net, paths


# ================================
# Evaluation core
# ================================
@torch.no_grad()
def eval_one_batch_under_injection(
    comm_net: ShortTermCommNet,
    radar_net: ShortTermRadarNet,
    theta_fixed: torch.Tensor,
    batch_data: dict,
    test_injection_variance: float,
    test_injection_samples: int,
):
    """
    對一個 estimated-channel batch 做 test injection evaluation。

    回傳：
        sumrate_np : shape = (B * L,)
        snr_db_np  : shape = (B * L,)
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
    L = int(test_injection_samples)

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

    W_C, W_R = comm_net.normalize_tx_power(W_C, W_R)

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
            test_injection_variance,
            DEVICE,
            h_dk_rep.dtype
        )

        h_rk_inj = h_rk_rep + complex_awgn(
            h_rk_rep.shape,
            test_injection_variance,
            DEVICE,
            h_rk_rep.dtype
        )

        G_inj = G_rep + complex_awgn(
            G_rep.shape,
            test_injection_variance,
            DEVICE,
            G_rep.dtype
        )

        g_dt_inj = g_dt_rep + complex_awgn(
            g_dt_rep.shape,
            test_injection_variance,
            DEVICE,
            g_dt_rep.dtype
        )

        h_dk_inj, h_rk_inj, G_inj, g_dt_inj = apply_power_normalization_to_injected_channels(
            h_dk_inj=h_dk_inj,
            h_rk_inj=h_rk_inj,
            G_inj=G_inj,
            g_dt_inj=g_dt_inj,
            h_dk_ref=h_dk_rep,
            h_rk_ref=h_rk_rep,
            G_ref=G_rep,
            g_dt_ref=g_dt_rep,
        )

        theta_rep = theta_batch.unsqueeze(1).expand(
            B, s, N
        ).reshape(B * s, N)

        W_C_rep = W_C.unsqueeze(1).expand(
            B, s, M, K
        ).reshape(B * s, M, K)

        radar_streams = W_R.shape[2]

        W_R_rep = W_R.unsqueeze(1).expand(
            B, s, M, radar_streams
        ).reshape(B * s, M, radar_streams)

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
def evaluate_one_model_at_injection(
    mode: str,
    penalty: float,
    longterm_net: LongTermPositionNet,
    test_dataset,
    test_injection_variance: float,
):
    """
    評估單一 mode / penalty model 在指定 test injection variance 下的表現。
    """
    print("\n" + "=" * 80)
    print(
        f"[EVAL2] mode={mode.upper()} penalty={penalty} "
        f"test_inj={test_injection_variance}"
    )
    print("=" * 80)

    reset_eval_seed()

    comm_net, radar_net, paths = load_shortterm_model(
        mode=mode,
        penalty=penalty
    )

    n_layouts = test_dataset["ue_layouts"].shape[0]
    n_channels_pool = test_dataset["st_h_dk_hat"].shape[1]

    if EVAL_CHANNELS_PER_LAYOUT is None:
        n_eval_channels = n_channels_pool
    else:
        n_eval_channels = min(
            int(EVAL_CHANNELS_PER_LAYOUT),
            n_channels_pool
        )

    sumrate_all = []
    snr_db_all = []

    for layout_id in range(n_layouts):
        ue_layout = test_dataset["ue_layouts"][layout_id]

        theta_fixed = get_fixed_theta_from_longterm(
            longterm_net=longterm_net,
            ue_layout_np=ue_layout
        )

        channel_ids = np.arange(n_eval_channels)

        batch_data = extract_shortterm_batch(
            dataset=test_dataset,
            layout_id=layout_id,
            channel_ids=channel_ids
        )

        sumrate_np, snr_db_np = eval_one_batch_under_injection(
            comm_net=comm_net,
            radar_net=radar_net,
            theta_fixed=theta_fixed,
            batch_data=batch_data,
            test_injection_variance=test_injection_variance,
            test_injection_samples=TEST_INJECTION_SAMPLES,
        )

        sumrate_all.append(sumrate_np)
        snr_db_all.append(snr_db_np)

    sumrate_all = np.concatenate(sumrate_all, axis=0)
    snr_db_all = np.concatenate(snr_db_all, axis=0)

    mean_sumrate = float(np.mean(sumrate_all))
    q05_snr_db = float(np.quantile(snr_db_all, 0.05))
    p_out = float(np.mean(snr_db_all < SENSING_SNR_THRESHOLD_dB))
    reliability = float(1.0 - p_out)

    result = {
        "mode": mode.upper(),
        "penalty": float(penalty),
        "test_injection_variance": float(test_injection_variance),
        "mean_sumrate": mean_sumrate,
        "q05_snr_db": q05_snr_db,
        "p_out": p_out,
        "reliability": reliability,
        "num_layouts": int(n_layouts),
        "channels_per_layout": int(n_eval_channels),
        "test_injection_samples": int(TEST_INJECTION_SAMPLES),
        "run_dir": paths["run_dir"],
    }

    print(
        f"[RESULT] {mode.upper()} penalty={penalty} "
        f"test_inj={test_injection_variance} | "
        f"Mean SumRate={mean_sumrate:.6f} | "
        f"Q0.05 SNR={q05_snr_db:.3f} dB | "
        f"P_out={100.0 * p_out:.3f}%"
    )

    return result, snr_db_all, sumrate_all


# ================================
# CDF helpers
# ================================
def make_cdf(values: np.ndarray):
    """
    由 samples 建立 CDF。
    """
    x = np.sort(values)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)

    return x, y


# ================================
# Save helpers
# ================================
def save_csv(rows: list[dict], csv_path: str):
    """
    儲存 list[dict] 成 CSV。
    """
    if len(rows) == 0:
        raise ValueError("rows is empty, cannot save csv.")

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


def save_best_pair_summary(best_reg, best_rob, summary_path: str):
    """
    儲存並印出 best feasible REG / ROB 的文字摘要。
    """
    lines = []

    lines.append("Eval2 selected REG / ROB pair summary\n")
    lines.append("=" * 80 + "\n\n")

    lines.append("Selection rule:\n")
    lines.append("Among models with P_out <= 5%, choose the highest mean sum-rate.\n\n")

    def add_best(name, best):
        if best is None:
            lines.append(f"[{name}] No feasible model found.\n\n")
            return

        lines.append(f"[{name}]\n")
        lines.append(f"  penalty      = {best['penalty']}\n")
        lines.append(f"  mean_sumrate = {best['mean_sumrate']:.6f}\n")
        lines.append(f"  q05_snr_db   = {best['q05_snr_db']:.3f} dB\n")
        lines.append(f"  p_out        = {100.0 * best['p_out']:.3f}%\n")
        lines.append(f"  feasible     = {best['feasible']}\n\n")

    add_best("Best feasible REG", best_reg)
    add_best("Best feasible ROB", best_rob)

    summary_text = "".join(lines)

    print("\n" + summary_text)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"[SAVE] Summary saved: {summary_path}")


# ================================
# Plot functions
# ================================
def plot_best_snr_cdf(
    best_reg,
    best_rob,
    reg_snr_db: np.ndarray,
    rob_snr_db: np.ndarray,
    fig_path: str,
):
    """
    Best feasible REG / ROB SNR CDF.
    """
    plt.figure(figsize=(9, 5.5))

    reg_x, reg_y = make_cdf(reg_snr_db)
    rob_x, rob_y = make_cdf(rob_snr_db)

    plt.plot(
        reg_x,
        reg_y,
        linewidth=2.2,
        label=f"Best REG penalty={best_reg['penalty']}"
    )

    plt.plot(
        rob_x,
        rob_y,
        linewidth=2.2,
        label=f"Best ROB penalty={best_rob['penalty']}"
    )

    plt.axvline(
        x=SENSING_SNR_THRESHOLD_dB,
        linestyle="--",
        linewidth=2.0,
        label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB"
    )

    plt.axhline(
        y=OUTAGE_PROB_THRESHOLD,
        linestyle=":",
        linewidth=2.0,
        label="5% outage level"
    )

    plt.xlabel("Sensing SNR (dB)")
    plt.ylabel("CDF")
    plt.title(
        f"Best Feasible REG vs ROB Sensing SNR CDF "
        f"(test injection variance = {FIXED_TEST_INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Best SNR CDF saved: {fig_path}")


def plot_best_rate_cdf(
    best_reg,
    best_rob,
    reg_sumrate: np.ndarray,
    rob_sumrate: np.ndarray,
    fig_path: str,
):
    """
    Best feasible REG / ROB Rate CDF.
    """
    plt.figure(figsize=(9, 5.5))

    reg_x, reg_y = make_cdf(reg_sumrate)
    rob_x, rob_y = make_cdf(rob_sumrate)

    plt.plot(
        reg_x,
        reg_y,
        linewidth=2.2,
        label=f"Best REG penalty={best_reg['penalty']}"
    )

    plt.plot(
        rob_x,
        rob_y,
        linewidth=2.2,
        label=f"Best ROB penalty={best_rob['penalty']}"
    )

    plt.xlabel("Sum-Rate (bits/s/Hz)")
    plt.ylabel("CDF")
    plt.title(
        f"Best Feasible REG vs ROB Sum-Rate CDF "
        f"(test injection variance = {FIXED_TEST_INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Best Rate CDF saved: {fig_path}")


def plot_injection_sweep_metric(
    sweep_rows: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    fig_path: str,
    horizontal_line=None,
    horizontal_label=None,
    scale_percent: bool = False,
):
    """
    畫 injection variance sweep metric。
    """
    plt.figure(figsize=(9, 5.5))

    for mode in ["REG", "ROB"]:
        rows = [
            r for r in sweep_rows
            if r["mode"] == mode
        ]

        rows = sorted(
            rows,
            key=lambda r: r["test_injection_variance"]
        )

        x = np.array(
            [r["test_injection_variance"] for r in rows],
            dtype=np.float64
        )

        y = np.array(
            [r[metric_key] for r in rows],
            dtype=np.float64
        )

        if scale_percent:
            y = 100.0 * y

        penalty = rows[0]["penalty"] if len(rows) > 0 else None

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=f"{mode} penalty={penalty}"
        )

    if horizontal_line is not None:
        h_value = horizontal_line
        if scale_percent:
            h_value = 100.0 * h_value

        plt.axhline(
            y=h_value,
            linestyle="--",
            linewidth=2.0,
            label=horizontal_label
        )

    plt.xlabel("Test Injection Variance")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Injection sweep figure saved: {fig_path}")


def plot_tradeoff_rate_reliability(
    selection_rows: list[dict],
    fig_path: str,
):
    """
    Rate-Reliability trade-off frontier.
    """
    plt.figure(figsize=(9, 5.5))

    for mode in ["REG", "ROB"]:
        rows = [
            r for r in selection_rows
            if r["mode"].upper() == mode
        ]

        rows = sorted(
            rows,
            key=lambda r: r["mean_sumrate"]
        )

        x = np.array(
            [r["mean_sumrate"] for r in rows],
            dtype=np.float64
        )

        y = np.array(
            [100.0 * (1.0 - r["p_out"]) for r in rows],
            dtype=np.float64
        )

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=mode
        )

        for r in rows:
            plt.annotate(
                str(r["penalty"]),
                (
                    r["mean_sumrate"],
                    100.0 * (1.0 - r["p_out"])
                ),
                fontsize=8
            )

    plt.axhline(
        y=95.0,
        linestyle="--",
        linewidth=2.0,
        label="95% reliability"
    )

    plt.xlabel("Mean Sum-Rate (bits/s/Hz)")
    plt.ylabel("Sensing Reliability (%)")
    plt.title(
        f"Rate-Reliability Trade-off "
        f"(test injection variance = {FIXED_TEST_INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Trade-off figure saved: {fig_path}")


def plot_tradeoff_rate_q05snr(
    selection_rows: list[dict],
    fig_path: str,
):
    """
    Rate-Q0.05 SNR trade-off frontier.
    """
    plt.figure(figsize=(9, 5.5))

    for mode in ["REG", "ROB"]:
        rows = [
            r for r in selection_rows
            if r["mode"].upper() == mode
        ]

        rows = sorted(
            rows,
            key=lambda r: r["mean_sumrate"]
        )

        x = np.array(
            [r["mean_sumrate"] for r in rows],
            dtype=np.float64
        )

        y = np.array(
            [r["q05_snr_db"] for r in rows],
            dtype=np.float64
        )

        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.2,
            label=mode
        )

        for r in rows:
            plt.annotate(
                str(r["penalty"]),
                (
                    r["mean_sumrate"],
                    r["q05_snr_db"]
                ),
                fontsize=8
            )

    plt.axhline(
        y=SENSING_SNR_THRESHOLD_dB,
        linestyle="--",
        linewidth=2.0,
        label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB"
    )

    plt.xlabel("Mean Sum-Rate (bits/s/Hz)")
    plt.ylabel("Q0.05 Sensing SNR (dB)")
    plt.title(
        f"Rate-Q0.05 Sensing SNR Trade-off "
        f"(test injection variance = {FIXED_TEST_INJECTION_VARIANCE})"
    )

    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path, dpi=300, format="jpg")
    plt.close()

    print(f"[SAVE] Trade-off figure saved: {fig_path}")


# ================================
# Main
# ================================
if __name__ == "__main__":
    print("[INFO] Eval2 started.")
    print(f"[INFO] DEVICE = {DEVICE}")
    print(f"[INFO] POWER_NORMALIZE_INJECTED_CHANNELS = {POWER_NORMALIZE_INJECTED_CHANNELS}")

    eval1_dir = build_eval1_dir()
    eval2_dir = build_eval2_dir()

    inj_tag = format_float_for_path(FIXED_TEST_INJECTION_VARIANCE)

    selection_csv_path = os.path.join(
        eval1_dir,
        f"selection_table_testinj_{inj_tag}.csv"
    )

    eval1_npz_path = os.path.join(
        eval1_dir,
        f"selection_metrics_testinj_{inj_tag}.npz"
    )

    print(f"[INFO] eval1_dir = {eval1_dir}")
    print(f"[INFO] eval2_dir = {eval2_dir}")
    print(f"[INFO] selection_csv_path = {selection_csv_path}")
    print(f"[INFO] eval1_npz_path = {eval1_npz_path}")

    selection_rows = read_selection_table(
        selection_csv_path
    )

    best_reg = select_best_feasible(
        rows=selection_rows,
        mode="REG"
    )

    best_rob = select_best_feasible(
        rows=selection_rows,
        mode="ROB"
    )

    if best_reg is None:
        print(
            "\n[WARN] No feasible REG model found under "
            f"P_out <= {100.0 * OUTAGE_PROB_THRESHOLD:.2f}%."
        )
        print("[WARN] Fallback: select REG model with minimum P_out.")
        best_reg = select_best_available(
            rows=selection_rows,
            mode="REG"
        )
        best_reg["feasible"] = False
        best_reg["selection_type"] = "best_available_min_pout"
    else:
        best_reg["selection_type"] = "best_feasible"

    if best_rob is None:
        raise RuntimeError("找不到 feasible ROB model。請檢查 eval.py selection table。")

    summary_path = os.path.join(
        eval2_dir,
        f"eval2_best_pair_summary_testinj_{inj_tag}.txt"
    )

    save_best_pair_summary(
        best_reg=best_reg,
        best_rob=best_rob,
        summary_path=summary_path
    )

    # ================================
    # 5. Best feasible CDF comparison
    # ================================
    eval1_npz = load_eval1_npz(
        eval1_npz_path
    )

    reg_snr_db_fixed, reg_sumrate_fixed = get_fixed_samples_from_eval1(
        eval1_npz=eval1_npz,
        mode="REG",
        penalty=best_reg["penalty"]
    )

    rob_snr_db_fixed, rob_sumrate_fixed = get_fixed_samples_from_eval1(
        eval1_npz=eval1_npz,
        mode="ROB",
        penalty=best_rob["penalty"]
    )

    best_snr_cdf_path = os.path.join(
        eval2_dir,
        f"best_feasible_snr_cdf_testinj_{inj_tag}.jpg"
    )

    best_rate_cdf_path = os.path.join(
        eval2_dir,
        f"best_feasible_rate_cdf_testinj_{inj_tag}.jpg"
    )

    plot_best_snr_cdf(
        best_reg=best_reg,
        best_rob=best_rob,
        reg_snr_db=reg_snr_db_fixed,
        rob_snr_db=rob_snr_db_fixed,
        fig_path=best_snr_cdf_path
    )

    plot_best_rate_cdf(
        best_reg=best_reg,
        best_rob=best_rob,
        reg_sumrate=reg_sumrate_fixed,
        rob_sumrate=rob_sumrate_fixed,
        fig_path=best_rate_cdf_path
    )

    # ================================
    # 6. Injection sweep robustness margin
    # ================================
    test_dataset = load_test_dataset(
        TEST_DATASET_PATH
    )

    longterm_net = load_longterm_model()

    injection_sweep_rows = []

    for test_inj in INJECTION_SWEEP_LIST:
        reg_result, _, _ = evaluate_one_model_at_injection(
            mode="reg",
            penalty=best_reg["penalty"],
            longterm_net=longterm_net,
            test_dataset=test_dataset,
            test_injection_variance=test_inj,
        )

        rob_result, _, _ = evaluate_one_model_at_injection(
            mode="rob",
            penalty=best_rob["penalty"],
            longterm_net=longterm_net,
            test_dataset=test_dataset,
            test_injection_variance=test_inj,
        )

        injection_sweep_rows.append(reg_result)
        injection_sweep_rows.append(rob_result)

    injection_sweep_csv_path = os.path.join(
        eval2_dir,
        "injection_sweep_best_feasible.csv"
    )

    save_csv(
        rows=injection_sweep_rows,
        csv_path=injection_sweep_csv_path
    )

    # 儲存 sweep metrics 成 npz
    sweep_npz_path = os.path.join(
        eval2_dir,
        "injection_sweep_best_feasible_metrics.npz"
    )

    np.savez_compressed(
        sweep_npz_path,
        rows=np.array(
            [
                (
                    r["mode"],
                    r["penalty"],
                    r["test_injection_variance"],
                    r["mean_sumrate"],
                    r["q05_snr_db"],
                    r["p_out"],
                    r["reliability"],
                )
                for r in injection_sweep_rows
            ],
            dtype=[
                ("mode", "U8"),
                ("penalty", "f8"),
                ("test_injection_variance", "f8"),
                ("mean_sumrate", "f8"),
                ("q05_snr_db", "f8"),
                ("p_out", "f8"),
                ("reliability", "f8"),
            ]
        )
    )

    print(f"[SAVE] Sweep NPZ saved: {sweep_npz_path}")

    pout_fig_path = os.path.join(
        eval2_dir,
        "injection_sweep_pout.jpg"
    )

    q05_fig_path = os.path.join(
        eval2_dir,
        "injection_sweep_q05_snr.jpg"
    )

    rate_fig_path = os.path.join(
        eval2_dir,
        "injection_sweep_mean_sumrate.jpg"
    )

    plot_injection_sweep_metric(
        sweep_rows=injection_sweep_rows,
        metric_key="p_out",
        ylabel="Sensing Outage Probability (%)",
        title="Injection Variance vs Sensing Outage Probability",
        fig_path=pout_fig_path,
        horizontal_line=OUTAGE_PROB_THRESHOLD,
        horizontal_label="5% outage constraint",
        scale_percent=True,
    )

    plot_injection_sweep_metric(
        sweep_rows=injection_sweep_rows,
        metric_key="q05_snr_db",
        ylabel="Q0.05 Sensing SNR (dB)",
        title="Injection Variance vs Q0.05 Sensing SNR",
        fig_path=q05_fig_path,
        horizontal_line=SENSING_SNR_THRESHOLD_dB,
        horizontal_label=f"SNR threshold = {SENSING_SNR_THRESHOLD_dB} dB",
        scale_percent=False,
    )

    plot_injection_sweep_metric(
        sweep_rows=injection_sweep_rows,
        metric_key="mean_sumrate",
        ylabel="Mean Sum-Rate (bits/s/Hz)",
        title="Injection Variance vs Mean Sum-Rate",
        fig_path=rate_fig_path,
        horizontal_line=None,
        horizontal_label=None,
        scale_percent=False,
    )

    # ================================
    # 7. Trade-off frontier
    # ================================
    rate_reliability_fig_path = os.path.join(
        eval2_dir,
        f"tradeoff_rate_reliability_testinj_{inj_tag}.jpg"
    )

    rate_q05snr_fig_path = os.path.join(
        eval2_dir,
        f"tradeoff_rate_q05snr_testinj_{inj_tag}.jpg"
    )

    plot_tradeoff_rate_reliability(
        selection_rows=selection_rows,
        fig_path=rate_reliability_fig_path
    )

    plot_tradeoff_rate_q05snr(
        selection_rows=selection_rows,
        fig_path=rate_q05snr_fig_path
    )

    print("\n" + "=" * 80)
    print("[INFO] Eval2 finished.")
    print("=" * 80)

    print(f"[INFO] Summary                    : {summary_path}")
    print(f"[INFO] Best SNR CDF               : {best_snr_cdf_path}")
    print(f"[INFO] Best Rate CDF              : {best_rate_cdf_path}")
    print(f"[INFO] Injection sweep CSV        : {injection_sweep_csv_path}")
    print(f"[INFO] Injection sweep NPZ        : {sweep_npz_path}")
    print(f"[INFO] Injection sweep Pout fig   : {pout_fig_path}")
    print(f"[INFO] Injection sweep Q0.05 fig  : {q05_fig_path}")
    print(f"[INFO] Injection sweep Rate fig   : {rate_fig_path}")
    print(f"[INFO] Tradeoff Reliability fig   : {rate_reliability_fig_path}")
    print(f"[INFO] Tradeoff Q0.05 SNR fig     : {rate_q05snr_fig_path}")