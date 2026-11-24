# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from channel_gen import steering_vector, theta_calculater
from two_timescale_NN import CommNet, RadarNet, ThetaNet


# ============================================================
# Settings for beampattern analysis
# ============================================================
ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = 90.0
ANGLE_STEP_DEG = 0.25
PATTERN_CHUNK_SIZE = 1000
PERCENTILE_SAMPLE_SLOTS = 5000
WC_PERCENTILE = 0.90
PLOT_FLOOR_DB = -60.0
SAFE_SEARCH_RANGE_DEG = (10.0, 89.0)
SAFE_CANDIDATES = 5
SAFE_MIN_SEPARATION_DEG = 5.0


# ============================================================
# Helpers
# ============================================================
def project_beamformers_to_tx_power(W_C, W_R):
    """
    與 evaluate_new.py 完全相同：
    只有當 P_C + P_R > P_T 時才共同縮放；
    若總功率未超標，保留 NN 原始 power allocation。
    """
    W_C_power = torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)).real
    W_R_power = torch.sum(torch.abs(W_R) ** 2, dim=(1, 2)).real
    total_power = W_C_power + W_R_power

    scale = torch.sqrt(TRANSMIT_POWER_TOTAL / (total_power + 1e-12))
    scale = torch.minimum(scale, torch.ones_like(scale))

    W_C = W_C * scale[:, None, None]
    W_R = W_R * scale[:, None, None]

    return W_C, W_R


def to_db_relative_to_pt(power):
    """
    將 angular power 除以總 TX power P_T 後轉 dB。
    這裡不是每條曲線各自 normalize 到 0 dB，
    因此可以保留 W_C / W_R 真實 power allocation 差異。
    """
    power = np.asarray(power, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(power / TRANSMIT_POWER_TOTAL, 1e-12))


def build_angle_steering(device):
    angles_deg = np.arange(
        ANGLE_MIN_DEG,
        ANGLE_MAX_DEG + 0.5 * ANGLE_STEP_DEG,
        ANGLE_STEP_DEG,
        dtype=np.float32,
    )
    angles_rad = np.deg2rad(angles_deg)

    # 完全沿用 channel_gen.py 的 steering_vector:
    # a(theta)[n] = exp(-j*pi*n*sin(theta))
    A = steering_vector(TX_ANT, angles_rad)  # (M,A)
    A_H = torch.as_tensor(A.conj().T, dtype=torch.complex64, device=device)  # (A,M)

    return angles_deg, A_H


def angular_power(W, A_H):
    """
    W:
        communication: (B,M,K)
        radar        : (B,M,R)

    A_H: (A,M), each row is a(theta)^H

    Return:
        P: (B,A)

    Definition:
        P(theta) = (1/M) * sum_s |a(theta)^H w_s|^2

    除以 M 是因為 ||a(theta)||^2=M。
    若單一 beam 為 sqrt(P)*a(theta0)/sqrt(M)，
    則在 theta0 的峰值剛好等於該 beam 的 TX power P。
    """
    response = torch.einsum("am,bms->bas", A_H, W)          # (B,A,S)
    power = torch.sum(torch.abs(response) ** 2, dim=2)       # (B,A)
    power = power / float(TX_ANT)
    return power.real


def compute_beampattern_stats(W_C, W_R, A_H):
    """
    對全部 test short-term slots：
        1. linear power mean -> dB
        2. 抽最多 5000 slots 算 Q90 envelope

    Q90 的用途：找一個不只是平均低、而是在多數 slots 都不容易被 W_C 照到的角度。
    """
    B = W_C.shape[0]
    A = A_H.shape[0]

    sum_C = torch.zeros(A, dtype=torch.float64, device="cpu")
    sum_R = torch.zeros(A, dtype=torch.float64, device="cpu")

    for start in range(0, B, PATTERN_CHUNK_SIZE):
        end = min(start + PATTERN_CHUNK_SIZE, B)

        P_C = angular_power(W_C[start:end], A_H)
        P_R = angular_power(W_R[start:end], A_H)

        sum_C += torch.sum(P_C, dim=0).double().cpu()
        sum_R += torch.sum(P_R, dim=0).double().cpu()

    mean_C = (sum_C / B).numpy()
    mean_R = (sum_R / B).numpy()

    sample_count = min(B, PERCENTILE_SAMPLE_SLOTS)
    sample_ids = np.linspace(0, B - 1, sample_count, dtype=np.int64)

    q_C_parts = []
    q_R_parts = []

    for start in range(0, sample_count, PATTERN_CHUNK_SIZE):
        ids = sample_ids[start:start + PATTERN_CHUNK_SIZE]

        ids_t = torch.as_tensor(ids, dtype=torch.long, device=W_C.device)
        P_C = angular_power(W_C[ids_t], A_H).cpu().numpy()
        P_R = angular_power(W_R[ids_t], A_H).cpu().numpy()

        q_C_parts.append(P_C)
        q_R_parts.append(P_R)

    q_C_all = np.concatenate(q_C_parts, axis=0)
    q_R_all = np.concatenate(q_R_parts, axis=0)

    q_C = np.quantile(q_C_all, WC_PERCENTILE, axis=0)
    q_R = np.quantile(q_R_all, WC_PERCENTILE, axis=0)

    return {
        "mean_C": mean_C,
        "mean_R": mean_R,
        "q_C": q_C,
        "q_R": q_R,
        "sample_count": sample_count,
    }


def find_separated_minima(angles_deg, values, search_range, num_candidates, min_separation_deg):
    mask = (angles_deg >= search_range[0]) & (angles_deg <= search_range[1])
    candidate_ids = np.where(mask)[0]
    candidate_ids = candidate_ids[np.argsort(values[candidate_ids])]

    selected = []
    for idx in candidate_ids:
        angle = float(angles_deg[idx])
        if all(abs(angle - prev_angle) >= min_separation_deg for prev_angle, _ in selected):
            selected.append((angle, float(values[idx])))
        if len(selected) >= num_candidates:
            break

    return selected


def plot_model_beampattern(model_name, angles_deg, stats, target_angle_deg, output_path):
    mean_C_db = to_db_relative_to_pt(stats["mean_C"])
    mean_R_db = to_db_relative_to_pt(stats["mean_R"])
    q_C_db = to_db_relative_to_pt(stats["q_C"])
    q_R_db = to_db_relative_to_pt(stats["q_R"])

    safe_candidates = find_separated_minima(
        angles_deg=angles_deg,
        values=stats["q_C"],
        search_range=SAFE_SEARCH_RANGE_DEG,
        num_candidates=SAFE_CANDIDATES,
        min_separation_deg=SAFE_MIN_SEPARATION_DEG,
    )

    best_safe_angle = safe_candidates[0][0]

    plt.figure(figsize=(12, 7))
    plt.plot(angles_deg, mean_C_db, label=r"$W_C$ mean")
    plt.plot(angles_deg, q_C_db, linestyle="--", label=fr"$W_C$ Q{int(WC_PERCENTILE*100)}")
    plt.plot(angles_deg, mean_R_db, label=r"$W_R$ mean")
    plt.plot(angles_deg, q_R_db, linestyle="--", label=fr"$W_R$ Q{int(WC_PERCENTILE*100)}")

    plt.axvline(target_angle_deg, linestyle=":", linewidth=2.0, label=f"Current target {target_angle_deg:.2f} deg")
    plt.axvline(best_safe_angle, linestyle="-.", linewidth=2.0, label=f"Lowest W_C Q90 {best_safe_angle:.2f} deg")

    plt.xlim(ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    plt.ylim(PLOT_FLOOR_DB, 1.0)
    plt.xlabel("BS steering angle (deg)")
    plt.ylabel(r"Angular transmit power / $P_T$ (dB)")
    plt.title(f"{model_name} W_C / W_R Beampattern")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return safe_candidates


def print_candidates(model_name, angles_deg, stats, target_angle_deg, safe_candidates):
    target_idx = int(np.argmin(np.abs(angles_deg - target_angle_deg)))

    print("\n" + "=" * 90)
    print(f"[{model_name} BEAMPATTERN]")
    print("=" * 90)
    print(f"Current target angle                  : {target_angle_deg:.3f} deg")
    print(f"W_C mean at current target            : {to_db_relative_to_pt(stats['mean_C'][target_idx]):.3f} dB / P_T")
    print(f"W_C Q{int(WC_PERCENTILE*100)} at current target             : {to_db_relative_to_pt(stats['q_C'][target_idx]):.3f} dB / P_T")
    print(f"W_R mean at current target            : {to_db_relative_to_pt(stats['mean_R'][target_idx]):.3f} dB / P_T")
    print(f"W_R Q{int(WC_PERCENTILE*100)} at current target             : {to_db_relative_to_pt(stats['q_R'][target_idx]):.3f} dB / P_T")
    print(f"Percentile sampled slots              : {stats['sample_count']}")
    print(f"Safe-angle search range               : {SAFE_SEARCH_RANGE_DEG[0]:.1f} ~ {SAFE_SEARCH_RANGE_DEG[1]:.1f} deg")
    print("-" * 90)
    print(f"Top {len(safe_candidates)} low-W_C candidates based on W_C Q{int(WC_PERCENTILE*100)}:")

    for rank, (angle, linear_value) in enumerate(safe_candidates, start=1):
        print(f"  #{rank}: angle={angle:7.2f} deg, W_C Q{int(WC_PERCENTILE*100)}={to_db_relative_to_pt(linear_value):8.3f} dB / P_T")

    print("=" * 90)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    test_dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")
    pretrained_theta_ckpt = os.path.join(PRETRAIN_DIR, "ris_only.ckpt")

    reg_comm_ckpt = os.path.join(REG_CKPT_DIR, "two_timescale_comm_reg.ckpt")
    reg_radar_ckpt = os.path.join(REG_CKPT_DIR, "two_timescale_radar_reg.ckpt")
    rob_comm_ckpt = os.path.join(ROB_CKPT_DIR, "two_timescale_comm_rob.ckpt")
    rob_radar_ckpt = os.path.join(ROB_CKPT_DIR, "two_timescale_radar_rob.ckpt")

    required_paths = [
        test_dataset_path,
        pretrained_theta_ckpt,
        reg_comm_ckpt,
        reg_radar_ckpt,
        rob_comm_ckpt,
        rob_radar_ckpt,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到檔案: {path}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    theta_net = ThetaNet().to(DEVICE)
    reg_comm_net = CommNet().to(DEVICE)
    reg_radar_net = RadarNet().to(DEVICE)
    rob_comm_net = CommNet().to(DEVICE)
    rob_radar_net = RadarNet().to(DEVICE)

    theta_net.load_model(pretrained_theta_ckpt, strict=True, verbose=True)
    reg_comm_net.load_model(reg_comm_ckpt, strict=True, verbose=True)
    reg_radar_net.load_model(reg_radar_ckpt, strict=True, verbose=True)
    rob_comm_net.load_model(rob_comm_ckpt, strict=True, verbose=True)
    rob_radar_net.load_model(rob_radar_ckpt, strict=True, verbose=True)

    theta_net.eval()
    reg_comm_net.eval()
    reg_radar_net.eval()
    rob_comm_net.eval()
    rob_radar_net.eval()

    with np.load(test_dataset_path) as data:
        test_dataset = {key: data[key] for key in ["positions", "h_dk", "h_rk", "G", "g_dt"]}

    test_trajectories = test_dataset["h_dk"].shape[0]
    test_channels = test_dataset["h_dk"].shape[1]

    target_angle_rad = theta_calculater(Q_BS, Q_UAV_TAR, "+X")
    target_angle_deg = float(np.rad2deg(target_angle_rad))
    target_distance = float(np.linalg.norm(np.asarray(Q_UAV_TAR, dtype=float) - np.asarray(Q_BS, dtype=float)))

    print("\n" + "=" * 90)
    print("[W_C / W_R BEAMPATTERN EVALUATION]")
    print("=" * 90)
    print(f"Test dataset                : {test_dataset_path}")
    print(f"Test trajectories           : {test_trajectories}")
    print(f"Slots / trajectory          : {test_channels}")
    print(f"Target position             : {Q_UAV_TAR}")
    print(f"Target distance from BS     : {target_distance:.3f} m")
    print(f"Target steering angle       : {target_angle_deg:.3f} deg")
    print(f"Angle sweep                 : {ANGLE_MIN_DEG:.1f} ~ {ANGLE_MAX_DEG:.1f} deg, step={ANGLE_STEP_DEG:.2f} deg")
    print(f"TX antennas                 : M={TX_ANT}")
    print(f"Total TX power              : {TRANSMIT_POWER_TOTAL:.6f} W")
    print("=" * 90)

    with torch.no_grad():
        # ------------------------------------------------------------
        # Build the exact same two-timescale test slots as evaluate_new.py
        # ------------------------------------------------------------
        channel_ids = np.arange(0, test_channels - LT_BLOCK_SIZE + 1, LT_BLOCK_SIZE)
        trajectory_ids = np.repeat(np.arange(test_trajectories), len(channel_ids))
        channel_ids = np.tile(channel_ids, test_trajectories)

        B_block = len(channel_ids)

        h_dk_start = torch.as_tensor(test_dataset["h_dk"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
        h_rk_start = torch.as_tensor(test_dataset["h_rk"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
        G_start = torch.as_tensor(test_dataset["G"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
        g_dt_start = torch.as_tensor(test_dataset["g_dt"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)

        test_theta = theta_net(h_dk_start, h_rk_start, G_start, g_dt_start)  # (B_block,N)

        block_ids = channel_ids[:, None] + np.arange(LT_BLOCK_SIZE)[None, :]

        test_h_dk = torch.as_tensor(test_dataset["h_dk"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)
        test_h_rk = torch.as_tensor(test_dataset["h_rk"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)
        test_G = torch.as_tensor(test_dataset["G"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)
        test_g_dt = torch.as_tensor(test_dataset["g_dt"][trajectory_ids[:, None], block_ids], dtype=torch.complex64, device=DEVICE)

        B = B_block * LT_BLOCK_SIZE
        M = TX_ANT
        N = RIS_UNIT
        K = UAV_COMM

        test_h_dk = test_h_dk.reshape(B, M, K)
        test_h_rk = test_h_rk.reshape(B, N, K)
        test_G = test_G.reshape(B, N, M)
        test_g_dt = test_g_dt.reshape(B, M, 1)
        test_theta = test_theta[:, None, :].expand(-1, LT_BLOCK_SIZE, -1).reshape(B, N)

        print(f"[INFO] RIS blocks            : {B_block}")
        print(f"[INFO] Evaluated test slots  : {B}")

        # ------------------------------------------------------------
        # REG beamformers
        # ------------------------------------------------------------
        reg_H_eff_H = reg_comm_net.compute_effective_channel(test_h_dk, test_h_rk, test_G, test_theta)
        reg_W_C_raw = reg_comm_net(reg_H_eff_H, test_g_dt)
        reg_W_R_raw = reg_radar_net(reg_H_eff_H, test_g_dt)
        reg_W_C, reg_W_R = project_beamformers_to_tx_power(reg_W_C_raw, reg_W_R_raw)

        # ------------------------------------------------------------
        # ROB beamformers
        # ------------------------------------------------------------
        rob_H_eff_H = rob_comm_net.compute_effective_channel(test_h_dk, test_h_rk, test_G, test_theta)
        rob_W_C_raw = rob_comm_net(rob_H_eff_H, test_g_dt)
        rob_W_R_raw = rob_radar_net(rob_H_eff_H, test_g_dt)
        rob_W_C, rob_W_R = project_beamformers_to_tx_power(rob_W_C_raw, rob_W_R_raw)

        # ------------------------------------------------------------
        # Power summary
        # ------------------------------------------------------------
        reg_pc = torch.mean(torch.sum(torch.abs(reg_W_C) ** 2, dim=(1, 2))).item()
        reg_pr = torch.mean(torch.sum(torch.abs(reg_W_R) ** 2, dim=(1, 2))).item()
        rob_pc = torch.mean(torch.sum(torch.abs(rob_W_C) ** 2, dim=(1, 2))).item()
        rob_pr = torch.mean(torch.sum(torch.abs(rob_W_R) ** 2, dim=(1, 2))).item()

        print(f"[REG] mean P_C={reg_pc:.6e} W, mean P_R={reg_pr:.6e} W, P_R/P_T={reg_pr/TRANSMIT_POWER_TOTAL:.6f}")
        print(f"[ROB] mean P_C={rob_pc:.6e} W, mean P_R={rob_pr:.6e} W, P_R/P_T={rob_pr/TRANSMIT_POWER_TOTAL:.6f}")

        # ------------------------------------------------------------
        # Beampattern
        # ------------------------------------------------------------
        angles_deg, A_H = build_angle_steering(DEVICE)

        reg_stats = compute_beampattern_stats(reg_W_C, reg_W_R, A_H)
        rob_stats = compute_beampattern_stats(rob_W_C, rob_W_R, A_H)

    reg_plot_path = os.path.join(RESULT_DIR, "REG_WC_WR_beampattern.png")
    rob_plot_path = os.path.join(RESULT_DIR, "ROB_WC_WR_beampattern.png")
    compare_plot_path = os.path.join(RESULT_DIR, "WC_beampattern_compare.png")
    stats_path = os.path.join(RESULT_DIR, "WC_WR_beampattern_stats.npz")

    reg_candidates = plot_model_beampattern("REG", angles_deg, reg_stats, target_angle_deg, reg_plot_path)
    rob_candidates = plot_model_beampattern("ROB", angles_deg, rob_stats, target_angle_deg, rob_plot_path)

    print_candidates("REG", angles_deg, reg_stats, target_angle_deg, reg_candidates)
    print_candidates("ROB", angles_deg, rob_stats, target_angle_deg, rob_candidates)

    # ------------------------------------------------------------
    # W_C comparison: this is the most useful plot for target relocation
    # ------------------------------------------------------------
    plt.figure(figsize=(12, 7))
    plt.plot(angles_deg, to_db_relative_to_pt(reg_stats["mean_C"]), label="REG W_C mean")
    plt.plot(angles_deg, to_db_relative_to_pt(reg_stats["q_C"]), linestyle="--", label=f"REG W_C Q{int(WC_PERCENTILE*100)}")
    plt.plot(angles_deg, to_db_relative_to_pt(rob_stats["mean_C"]), label="ROB W_C mean")
    plt.plot(angles_deg, to_db_relative_to_pt(rob_stats["q_C"]), linestyle="--", label=f"ROB W_C Q{int(WC_PERCENTILE*100)}")
    plt.axvline(target_angle_deg, linestyle=":", linewidth=2.0, label=f"Current target {target_angle_deg:.2f} deg")
    plt.xlim(ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    plt.ylim(PLOT_FLOOR_DB, 1.0)
    plt.xlabel("BS steering angle (deg)")
    plt.ylabel(r"W_C angular transmit power / $P_T$ (dB)")
    plt.title("REG / ROB W_C Beampattern Comparison")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_plot_path, dpi=300)
    plt.close()

    np.savez(
        stats_path,
        angles_deg=angles_deg,
        target_angle_deg=np.float32(target_angle_deg),
        reg_mean_C=reg_stats["mean_C"],
        reg_q_C=reg_stats["q_C"],
        reg_mean_R=reg_stats["mean_R"],
        reg_q_R=reg_stats["q_R"],
        rob_mean_C=rob_stats["mean_C"],
        rob_q_C=rob_stats["q_C"],
        rob_mean_R=rob_stats["mean_R"],
        rob_q_R=rob_stats["q_R"],
    )

    print("\n[INFO] Saved:")
    print(f"  {reg_plot_path}")
    print(f"  {rob_plot_path}")
    print(f"  {compare_plot_path}")
    print(f"  {stats_path}")
