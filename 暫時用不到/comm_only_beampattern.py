# -*- coding: utf-8 -*-
"""
Communication-only BS beampattern study.

This script intentionally contains:
    - no g_dt input
    - no RadarNet
    - no W_R
    - no sensing metric
    - no sensing loss

It loads the communication-only ThetaNet checkpoint created by ris_comm_only.py,
uses RZF for W_C in every short-term slot, allocates the full BS power budget to W_C,
and scans the M=TX_ANT ULA beampattern over -90...90 degrees.

The principal spatial-hole metric is W_C Q95 across all test short-term slots:
    lower Q95 => W_C is rarely strong at that angle.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from baseline import make_rzf_beamformer, RZF_LAMBDA
from two_timescale_NN import ThetaNet
from channel_gen import steering_vector


# ================================
# Scan settings
# ================================
ANGLE_MIN_DEG = -90.0
ANGLE_MAX_DEG = 90.0
ANGLE_STEP_DEG = 0.5

# Target is currently on +y side, so also report the best positive-angle holes.
POSITIVE_SEARCH_MIN_DEG = 5.0
POSITIVE_SEARCH_MAX_DEG = 85.0

# Prevent top-N from reporting five adjacent samples from the same null.
MIN_CANDIDATE_SEPARATION_DEG = 5.0
TOP_N = 8

EVAL_CHUNK_SIZE = 2000


# ================================
# Helpers
# ================================
def zero_target(batch_size):
    return torch.zeros((batch_size, TX_ANT, 1), dtype=torch.complex64, device=DEVICE)


def allocate_all_power_to_wc(W_C_dir):
    power = torch.sum(torch.abs(W_C_dir) ** 2, dim=(1, 2), keepdim=True).real
    scale = torch.sqrt(TRANSMIT_POWER_TOTAL / power.clamp_min(1e-12))
    return W_C_dir * scale


def select_separated_minima(angles_deg, values, mask, top_n=TOP_N, min_sep_deg=MIN_CANDIDATE_SEPARATION_DEG):
    candidate_indices = np.where(mask)[0]
    order = candidate_indices[np.argsort(values[candidate_indices])]

    selected = []
    for idx in order:
        angle = float(angles_deg[idx])
        if all(abs(angle - float(angles_deg[j])) >= min_sep_deg for j in selected):
            selected.append(idx)
        if len(selected) >= top_n:
            break
    return selected


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


# ================================
# Main
# ================================
if __name__ == "__main__":
    test_dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")
    comm_only_theta_ckpt = os.path.join(PRETRAIN_DIR, "ris_comm_only.ckpt")

    if not os.path.exists(comm_only_theta_ckpt):
        raise FileNotFoundError(
            f"找不到 communication-only ThetaNet checkpoint: {comm_only_theta_ckpt}\n"
            f"請先執行: python ris_comm_only.py"
        )

    with np.load(test_dataset_path) as data:
        test_dataset = {key: data[key] for key in ["positions", "h_dk", "h_rk", "G"]}

    test_trajectories, test_channels = test_dataset["h_dk"].shape[:2]

    theta_net = ThetaNet().to(DEVICE)
    theta_net.load_model(comm_only_theta_ckpt, strict=True, verbose=True)
    theta_net.eval()

    # ------------------------------------------------
    # Two-timescale indexing: same structure as evaluate_new.py
    # ------------------------------------------------
    block_start_ids = np.arange(0, test_channels - LT_BLOCK_SIZE + 1, LT_BLOCK_SIZE)
    trajectory_ids = np.repeat(np.arange(test_trajectories), len(block_start_ids))
    channel_ids = np.tile(block_start_ids, test_trajectories)

    B_block = len(channel_ids)
    block_ids = channel_ids[:, None] + np.arange(LT_BLOCK_SIZE)[None, :]
    B = B_block * LT_BLOCK_SIZE

    print("\n" + "=" * 90)
    print("[COMMUNICATION-ONLY BS BEAMPATTERN]")
    print("=" * 90)
    print(f"Test trajectories        : {test_trajectories}")
    print(f"Test slots/trajectory    : {test_channels}")
    print(f"Evaluated short-term slots: {B}")
    print(f"RIS block size           : {LT_BLOCK_SIZE}")
    print(f"BS antennas M            : {TX_ANT}")
    print(f"UEs K                    : {UAV_COMM}")
    print(f"RZF lambda               : {RZF_LAMBDA}")
    print(f"W_C total power          : {TRANSMIT_POWER_TOTAL:.6f} W")
    print("Target channel            : NOT USED")
    print("RadarNet / W_R            : NOT USED")
    print("Sensing objective         : NOT USED")
    print("=" * 90)

    # ------------------------------------------------
    # Long-term RIS configuration from communication-only ThetaNet
    # ------------------------------------------------
    with torch.no_grad():
        h_dk_start = torch.as_tensor(test_dataset["h_dk"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
        h_rk_start = torch.as_tensor(test_dataset["h_rk"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)
        G_start = torch.as_tensor(test_dataset["G"][trajectory_ids, channel_ids], dtype=torch.complex64, device=DEVICE)

        theta_block_start = theta_net(h_dk_start, h_rk_start, G_start, zero_target(B_block))  # (B_block,N)
        theta_all = theta_block_start[:, None, :].expand(-1, LT_BLOCK_SIZE, -1).reshape(B, RIS_UNIT)

    # ------------------------------------------------
    # Angle grid and steering vectors
    # steering_vector uses exactly the same convention as channel_gen.py:
    # a(theta)=exp(-j*pi*n*sin(theta))
    # ------------------------------------------------
    angles_deg = np.arange(ANGLE_MIN_DEG, ANGLE_MAX_DEG + 0.5 * ANGLE_STEP_DEG, ANGLE_STEP_DEG, dtype=np.float32)
    angles_rad = np.deg2rad(angles_deg)

    A_np = steering_vector(TX_ANT, angles_rad)                 # (M,A)
    A = torch.as_tensor(A_np, dtype=torch.complex64, device=DEVICE)
    num_angles = len(angles_deg)

    # Store normalized angular power for every slot; float32 memory is manageable.
    pattern_ratio_all = np.empty((B, num_angles), dtype=np.float32)
    wc_power_all = np.empty(B, dtype=np.float32)

    # ------------------------------------------------
    # Evaluate all short-term slots in chunks
    # ------------------------------------------------
    with torch.no_grad():
        for start in range(0, B, EVAL_CHUNK_SIZE):
            end = min(start + EVAL_CHUNK_SIZE, B)
            C = end - start

            # Convert flattened short-term index back to block + slot index
            flat_idx = np.arange(start, end)
            block_idx = flat_idx // LT_BLOCK_SIZE
            slot_offset = flat_idx % LT_BLOCK_SIZE

            traj_chunk = trajectory_ids[block_idx]
            slot_chunk = channel_ids[block_idx] + slot_offset

            h_dk = torch.as_tensor(test_dataset["h_dk"][traj_chunk, slot_chunk], dtype=torch.complex64, device=DEVICE)
            h_rk = torch.as_tensor(test_dataset["h_rk"][traj_chunk, slot_chunk], dtype=torch.complex64, device=DEVICE)
            G = torch.as_tensor(test_dataset["G"][traj_chunk, slot_chunk], dtype=torch.complex64, device=DEVICE)
            theta = theta_all[start:end]

            H_eff_H = theta_net.compute_effective_channel(h_dk, h_rk, G, theta)     # (C,K,M)
            W_C_dir = make_rzf_beamformer(H_eff_H, RZF_LAMBDA)                      # (C,M,K)
            W_C = allocate_all_power_to_wc(W_C_dir)                                 # full power to communication

            wc_power = torch.sum(torch.abs(W_C) ** 2, dim=(1, 2)).real               # (C,)
            wc_power_all[start:end] = wc_power.cpu().numpy().astype(np.float32)

            # a(theta)^H W_C for every slot, angle, and user
            response = torch.einsum("ma,cmk->cak", torch.conj(A), W_C)              # (C,A,K)

            # Sum the two independent communication-stream powers.
            # Divide by M so the angular quantity is on the same transmit-power scale.
            angular_power = torch.sum(torch.abs(response) ** 2, dim=2).real / float(TX_ANT)  # (C,A)
            pattern_ratio = angular_power / float(TRANSMIT_POWER_TOTAL)

            pattern_ratio_all[start:end] = pattern_ratio.cpu().numpy().astype(np.float32)

            if start % (10 * EVAL_CHUNK_SIZE) == 0:
                print(f"[INFO] evaluated {end:6d} / {B} short-term slots")

    # ------------------------------------------------
    # Long-term statistics across all moving-UE slots
    # ------------------------------------------------
    pattern_mean = np.mean(pattern_ratio_all, axis=0)
    pattern_q50 = np.quantile(pattern_ratio_all, 0.50, axis=0)
    pattern_q90 = np.quantile(pattern_ratio_all, 0.90, axis=0)
    pattern_q95 = np.quantile(pattern_ratio_all, 0.95, axis=0)
    pattern_q99 = np.quantile(pattern_ratio_all, 0.99, axis=0)

    # Q95 is the main spatial-hole criterion:
    # 95% of test slots have W_C angular power <= this value.
    positive_mask = (angles_deg >= POSITIVE_SEARCH_MIN_DEG) & (angles_deg <= POSITIVE_SEARCH_MAX_DEG)
    positive_best = select_separated_minima(angles_deg, pattern_q95, positive_mask)

    full_mask = np.ones_like(angles_deg, dtype=bool)
    global_best = select_separated_minima(angles_deg, pattern_q95, full_mask)

    # UE geometry shown only as context; the selection itself uses W_C pattern, not UE angle.
    positions = test_dataset["positions"].reshape(-1, 2)
    ue_geo_angles = np.rad2deg(np.arctan2(positions[:, 1] - Q_BS[1], positions[:, 0] - Q_BS[0]))
    ue_geo_min = float(np.min(ue_geo_angles))
    ue_geo_max = float(np.max(ue_geo_angles))

    print("\n" + "-" * 90)
    print("[POWER CHECK]")
    print(f"mean ||W_C||_F^2 = {np.mean(wc_power_all):.8f} W")
    print(f"min  ||W_C||_F^2 = {np.min(wc_power_all):.8f} W")
    print(f"max  ||W_C||_F^2 = {np.max(wc_power_all):.8f} W")
    print(f"P_T                = {TRANSMIT_POWER_TOTAL:.8f} W")
    print(f"UE geometric angles from BS over test set: [{ue_geo_min:.2f}, {ue_geo_max:.2f}] deg")

    print("\n" + "-" * 90)
    print("[TOP POSITIVE-ANGLE SPATIAL HOLES: ranked by W_C Q95]")
    for rank, idx in enumerate(positive_best, start=1):
        print(
            f"#{rank}: angle={angles_deg[idx]:7.2f} deg | "
            f"mean={db(pattern_mean[idx]):7.2f} dB | "
            f"Q90={db(pattern_q90[idx]):7.2f} dB | "
            f"Q95={db(pattern_q95[idx]):7.2f} dB | "
            f"Q99={db(pattern_q99[idx]):7.2f} dB"
        )

    print("\n" + "-" * 90)
    print("[TOP GLOBAL SPATIAL HOLES: ranked by W_C Q95]")
    for rank, idx in enumerate(global_best, start=1):
        print(
            f"#{rank}: angle={angles_deg[idx]:7.2f} deg | "
            f"mean={db(pattern_mean[idx]):7.2f} dB | "
            f"Q90={db(pattern_q90[idx]):7.2f} dB | "
            f"Q95={db(pattern_q95[idx]):7.2f} dB | "
            f"Q99={db(pattern_q99[idx]):7.2f} dB"
        )

    # ------------------------------------------------
    # Save numerical results
    # ------------------------------------------------
    stats_path = os.path.join(RESULT_DIR, "comm_only_WC_beampattern_stats.npz")
    np.savez(
        stats_path,
        angles_deg=angles_deg,
        pattern_mean=pattern_mean.astype(np.float32),
        pattern_q50=pattern_q50.astype(np.float32),
        pattern_q90=pattern_q90.astype(np.float32),
        pattern_q95=pattern_q95.astype(np.float32),
        pattern_q99=pattern_q99.astype(np.float32),
        positive_best_angles_deg=np.asarray([angles_deg[i] for i in positive_best], dtype=np.float32),
        global_best_angles_deg=np.asarray([angles_deg[i] for i in global_best], dtype=np.float32),
        ue_geometric_angle_min_deg=np.float32(ue_geo_min),
        ue_geometric_angle_max_deg=np.float32(ue_geo_max),
    )

    # ------------------------------------------------
    # Plot 1: main communication-only beampattern
    # ------------------------------------------------
    fig_path = os.path.join(RESULT_DIR, "comm_only_WC_beampattern.png")

    plt.figure(figsize=(12, 7))
    plt.plot(angles_deg, db(pattern_mean), label=r"$W_C$ mean", linewidth=2.0)
    plt.plot(angles_deg, db(pattern_q90), label=r"$W_C$ Q90", linestyle="--", linewidth=2.0)
    plt.plot(angles_deg, db(pattern_q95), label=r"$W_C$ Q95", linestyle="-.", linewidth=2.0)
    plt.plot(angles_deg, db(pattern_q99), label=r"$W_C$ Q99", linestyle=":", linewidth=2.0)

    # Geometry is context only, not the spatial-hole decision criterion.
    plt.axvspan(ue_geo_min, ue_geo_max, alpha=0.08, label="UE geometric angle range")

    if positive_best:
        best_idx = positive_best[0]
        plt.axvline(angles_deg[best_idx], linestyle="--", linewidth=2.0, label=f"Best +angle Q95 hole {angles_deg[best_idx]:.1f} deg")

    plt.xlabel("BS steering angle (deg)")
    plt.ylabel(r"Communication angular transmit power / $P_T$ (dB)")
    plt.title("Communication-only $W_C$ Beampattern — Moving UEs")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # ------------------------------------------------
    # Plot 2: positive-angle zoom for target placement
    # ------------------------------------------------
    zoom_path = os.path.join(RESULT_DIR, "comm_only_WC_positive_spatial_holes.png")
    zoom = positive_mask

    plt.figure(figsize=(12, 7))
    plt.plot(angles_deg[zoom], db(pattern_mean[zoom]), label=r"$W_C$ mean", linewidth=2.0)
    plt.plot(angles_deg[zoom], db(pattern_q90[zoom]), label=r"$W_C$ Q90", linestyle="--", linewidth=2.0)
    plt.plot(angles_deg[zoom], db(pattern_q95[zoom]), label=r"$W_C$ Q95", linestyle="-.", linewidth=2.0)
    plt.plot(angles_deg[zoom], db(pattern_q99[zoom]), label=r"$W_C$ Q99", linestyle=":", linewidth=2.0)

    for rank, idx in enumerate(positive_best[:4], start=1):
        plt.axvline(angles_deg[idx], alpha=0.55, linewidth=1.2)
        plt.text(angles_deg[idx], db(pattern_q95[idx]), f"  #{rank} {angles_deg[idx]:.1f}°", rotation=90, va="bottom")

    plt.xlabel("Positive BS steering angle (deg)")
    plt.ylabel(r"Communication angular transmit power / $P_T$ (dB)")
    plt.title("Communication-only Spatial Holes for Target Placement")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(zoom_path, dpi=300)
    plt.close()

    print("\n" + "=" * 90)
    print("[SAVED]")
    print(f"Main beampattern : {fig_path}")
    print(f"Positive holes   : {zoom_path}")
    print(f"Statistics       : {stats_path}")
    print("=" * 90)
