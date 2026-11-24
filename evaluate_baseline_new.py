# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch

from settings import *
from two_timescale_NN import CommNet, ThetaNet
from baseline import (
    make_rzf_beamformer,
    mrt_in_H_eff_H_nullspace,
    beamformers_power_split,
    RZF_LAMBDA,
)


# ================================
# Helpers
# ================================
def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0,variance):
        E|n|^2 = variance
        Re/Im ~ N(0,variance/2)
    """
    sigma = math.sqrt(variance / 2.0)
    nr = torch.randn(shape,device=device,dtype=torch.float32) * sigma
    ni = torch.randn(shape,device=device,dtype=torch.float32) * sigma
    return torch.complex(nr,ni).to(dtype=cdtype)


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)
    return "{" + " ".join([f"[{float(v):.{precision}e}]" for v in x]) + "}"


# ================================
# Main
# ================================
if __name__ == "__main__":

    # ================================================================
    # 1. Paths
    # ================================================================
    test_dataset_path     = os.path.join(DATA_DIR,"dataset_test.npz")
    pretrained_theta_ckpt = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

    for required_path in [
        test_dataset_path,
        pretrained_theta_ckpt,
    ]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"找不到檔案: {required_path}")

    # ================================================================
    # 2. Networks
    #    ThetaNet：與 REG / ROB evaluation 完全相同
    #    physics_net：只使用共用 physics functions，不使用 NN beamformer
    # ================================================================
    theta_net = ThetaNet().to(DEVICE)
    physics_net = CommNet().to(DEVICE)

    theta_net.load_model(
        pretrained_theta_ckpt,
        strict=True,
        verbose=True,
    )

    theta_net.eval()
    physics_net.eval()

    # ================================================================
    # 3. Test dataset
    # ================================================================
    with np.load(test_dataset_path) as data:
        test_dataset = {
            key: data[key]
            for key in [
                "positions",
                "h_dk",
                "h_rk",
                "G",
                "g_dt",
            ]
        }

    test_trajectories = test_dataset["h_dk"].shape[0]
    test_channels = test_dataset["h_dk"].shape[1]

    print("\n[INFO] 載入 baseline test dataset 與 ThetaNet ...")
    print(f"[INFO] test_dataset_path      = {test_dataset_path}")
    print(f"[INFO] pretrained_theta_ckpt  = {pretrained_theta_ckpt}")
    print(f"[INFO] test trajectories      = {test_trajectories}")
    print(f"[INFO] test channels          = {test_channels}")
    print(f"[INFO] h_dk shape             = {test_dataset['h_dk'].shape}")
    print(f"[INFO] h_rk shape             = {test_dataset['h_rk'].shape}")
    print(f"[INFO] G shape                = {test_dataset['G'].shape}")
    print(f"[INFO] g_dt shape             = {test_dataset['g_dt'].shape}")

    with torch.no_grad():

        # ================================================================
        # 4. 建立與 evaluate_new.py 完全相同的 two-timescale test blocks
        # ================================================================
        channel_ids = np.arange(
            0,
            test_channels - LT_BLOCK_SIZE + 1,
            LT_BLOCK_SIZE,
        )

        trajectory_ids = np.repeat(
            np.arange(test_trajectories),
            len(channel_ids),
        )

        channel_ids = np.tile(
            channel_ids,
            test_trajectories,
        )

        B_block = len(channel_ids)

        # ================================================================
        # 5. 每個 LT block start slot 使用同一個 trained ThetaNet 產生 theta
        # ================================================================
        h_dk_start = torch.as_tensor(
            test_dataset["h_dk"][trajectory_ids,channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        h_rk_start = torch.as_tensor(
            test_dataset["h_rk"][trajectory_ids,channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        G_start = torch.as_tensor(
            test_dataset["G"][trajectory_ids,channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        g_dt_start = torch.as_tensor(
            test_dataset["g_dt"][trajectory_ids,channel_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_theta = theta_net(
            h_dk_start,
            h_rk_start,
            G_start,
            g_dt_start,
        )

        # ================================================================
        # 6. 同一 theta 固定使用 LT_BLOCK_SIZE 個 slots
        # ================================================================
        block_ids = (
            channel_ids[:,None]
            + np.arange(LT_BLOCK_SIZE)[None,:]
        )

        test_h_dk_hat = torch.as_tensor(
            test_dataset["h_dk"][trajectory_ids[:,None],block_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_h_rk_hat = torch.as_tensor(
            test_dataset["h_rk"][trajectory_ids[:,None],block_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_G_hat = torch.as_tensor(
            test_dataset["G"][trajectory_ids[:,None],block_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_g_dt_hat = torch.as_tensor(
            test_dataset["g_dt"][trajectory_ids[:,None],block_ids],
            dtype=torch.complex64,
            device=DEVICE,
        )

        B = B_block * LT_BLOCK_SIZE
        M = TX_ANT
        N = RIS_UNIT
        K = UAV_COMM
        R = RADAR_STREAMS
        S = INJECTION_SAMPLES

        test_h_dk_hat = test_h_dk_hat.reshape(B,M,K)
        test_h_rk_hat = test_h_rk_hat.reshape(B,N,K)
        test_G_hat = test_G_hat.reshape(B,N,M)
        test_g_dt_hat = test_g_dt_hat.reshape(B,M,1)

        test_theta = (
            test_theta[:,None,:]
            .expand(-1,LT_BLOCK_SIZE,-1)
            .reshape(B,N)
        )

        print(f"[INFO] RIS blocks             = {B_block}")
        print(f"[INFO] evaluated test slots   = {B}")

        # ================================================================
        # 7. Analytical baseline beamformer
        #
        #    RIS:
        #        與 NN evaluation 共用 trained ThetaNet
        #
        #    Communication:
        #        RZF
        #
        #    Radar:
        #        H_eff^H nullspace MRT
        #
        #    Power:
        #        使用 baseline.py 原始固定 power split
        #        P_C / P_R = 0.4 / 0.6
        # ================================================================
        H_eff_H_hat = physics_net.compute_effective_channel(
            test_h_dk_hat,
            test_h_rk_hat,
            test_G_hat,
            test_theta,
        )

        W_C_dir = make_rzf_beamformer(
            H_eff_H_hat,
            RZF_LAMBDA,
        )

        W_R_dir = mrt_in_H_eff_H_nullspace(
            H_eff_H_hat,
            test_g_dt_hat,
        )

        W_C, W_R = beamformers_power_split(
            W_C_dir,
            W_R_dir,
        )

        if W_C.shape != (B,M,K):
            raise ValueError(
                f"W_C shape error: expected {(B,M,K)}, got {tuple(W_C.shape)}"
            )

        if W_R.shape != (B,M,R):
            raise ValueError(
                f"W_R shape error: expected {(B,M,R)}, got {tuple(W_R.shape)}"
            )

        # ================================================================
        # 8. Power check
        #    baseline 原本就固定使用 P_T 的 40/60 split，
        #    因此不再額外改變 power allocation。
        # ================================================================
        W_C_power_per_slot = torch.sum(
            torch.abs(W_C) ** 2,
            dim=(1,2),
        ).real

        W_R_power_per_slot = torch.sum(
            torch.abs(W_R) ** 2,
            dim=(1,2),
        ).real

        total_power_per_slot = (
            W_C_power_per_slot
            + W_R_power_per_slot
        )

        W_C_power = torch.mean(W_C_power_per_slot)
        W_R_power = torch.mean(W_R_power_per_slot)
        total_power = torch.mean(total_power_per_slot)

        max_total_power = torch.max(total_power_per_slot)

        # ================================================================
        # 9. Robust evaluation
        #    完全比照 evaluate_new.py：
        #
        #    - 同樣 S = INJECTION_SAMPLES
        #    - 同樣 INJECTION_VARIANCE
        #    - 同樣 h_dk / h_rk / G / g_dt uncertainty model
        #    - 每個 test slot 對 S 組 uncertainty：
        #          先取 worst UE
        #          再取 Q0.05
        #    - sensing violation：
        #          直接統計所有 test uncertainty realizations 中
        #          target SNR < threshold 的 empirical probability
        # ================================================================
        EVAL_CHUNK_SIZE = 200

        robust_all = []

        user_rate_q05_sum = torch.zeros(
            K,
            device=DEVICE,
        )

        target_snr_sum = 0.0

        sensing_violation_count = 0
        sensing_total_count = 0

        signal_sum = torch.zeros(
            K,
            device=DEVICE,
        )

        comm_interf_sum = torch.zeros(
            K,
            device=DEVICE,
        )

        radar_interf_sum = torch.zeros(
            K,
            device=DEVICE,
        )

        metric_total_count = 0

        for start in range(
            0,
            B,
            EVAL_CHUNK_SIZE,
        ):

            end = min(
                start + EVAL_CHUNK_SIZE,
                B,
            )

            C = end - start

            h_dk_hat = test_h_dk_hat[start:end]
            h_rk_hat = test_h_rk_hat[start:end]
            G_hat = test_G_hat[start:end]
            g_dt_hat = test_g_dt_hat[start:end]

            theta_chunk = test_theta[start:end]

            W_C_chunk = W_C[start:end]
            W_R_chunk = W_R[start:end]

            # ============================================================
            # 9-1. Injection channels
            # ============================================================
            inj_h_dk_rep = (
                h_dk_hat
                .unsqueeze(0)
                .expand(S,C,M,K)
            )

            inj_h_rk_rep = (
                h_rk_hat
                .unsqueeze(0)
                .expand(S,C,N,K)
            )

            inj_G_rep = (
                G_hat
                .unsqueeze(0)
                .expand(S,C,N,M)
            )

            inj_g_dt_rep = (
                g_dt_hat
                .unsqueeze(0)
                .expand(S,C,M,1)
            )

            inj_h_dk_power = torch.mean(
                torch.abs(inj_h_dk_rep) ** 2,
                dim=(2,3),
                keepdim=True,
            ).real

            inj_h_rk_power = torch.mean(
                torch.abs(inj_h_rk_rep) ** 2,
                dim=(2,3),
                keepdim=True,
            ).real

            inj_G_power = torch.mean(
                torch.abs(inj_G_rep) ** 2,
                dim=(2,3),
                keepdim=True,
            ).real

            inj_g_dt_power = torch.mean(
                torch.abs(inj_g_dt_rep) ** 2,
                dim=(2,3),
                keepdim=True,
            ).real

            inj_h_dk = (
                inj_h_dk_rep
                + torch.sqrt(
                    INJECTION_VARIANCE
                    * inj_h_dk_power
                )
                * complex_awgn(
                    inj_h_dk_rep.shape,
                    1.0,
                    DEVICE,
                    inj_h_dk_rep.dtype,
                )
            )

            inj_h_rk = (
                inj_h_rk_rep
                + torch.sqrt(
                    INJECTION_VARIANCE
                    * inj_h_rk_power
                )
                * complex_awgn(
                    inj_h_rk_rep.shape,
                    1.0,
                    DEVICE,
                    inj_h_rk_rep.dtype,
                )
            )

            inj_G = (
                inj_G_rep
                + torch.sqrt(
                    INJECTION_VARIANCE
                    * inj_G_power
                )
                * complex_awgn(
                    inj_G_rep.shape,
                    1.0,
                    DEVICE,
                    inj_G_rep.dtype,
                )
            )

            inj_g_dt = (
                inj_g_dt_rep
                + torch.sqrt(
                    INJECTION_VARIANCE
                    * inj_g_dt_power
                )
                * complex_awgn(
                    inj_g_dt_rep.shape,
                    1.0,
                    DEVICE,
                    inj_g_dt_rep.dtype,
                )
            )

            # ============================================================
            # 9-2. Flatten
            # ============================================================
            inj_h_dk_flat = inj_h_dk.reshape(
                S*C,
                M,
                K,
            )

            inj_h_rk_flat = inj_h_rk.reshape(
                S*C,
                N,
                K,
            )

            inj_G_flat = inj_G.reshape(
                S*C,
                N,
                M,
            )

            inj_g_dt_flat = inj_g_dt.reshape(
                S*C,
                M,
                1,
            )

            theta_flat = (
                theta_chunk
                .unsqueeze(0)
                .expand(S,C,N)
                .reshape(S*C,N)
            )

            W_C_flat = (
                W_C_chunk
                .unsqueeze(0)
                .expand(S,C,M,K)
                .reshape(S*C,M,K)
            )

            W_R_flat = (
                W_R_chunk
                .unsqueeze(0)
                .expand(S,C,M,R)
                .reshape(S*C,M,R)
            )

            # ============================================================
            # 9-3. 使用 injected true channels 評估固定 baseline beamformer
            # ============================================================
            H_eff_H_inj = physics_net.compute_effective_channel(
                inj_h_dk_flat,
                inj_h_rk_flat,
                inj_G_flat,
                theta_flat,
            )

            metrics = physics_net.compute_isac_batch_performance(
                H_eff_H_inj,
                inj_g_dt_flat,
                W_C_flat,
                W_R_flat,
            )

            allUE_rate = metrics["rate"].reshape(
                S,
                C,
                K,
            )

            target_snr = metrics["target_snr"].reshape(
                S,
                C,
            )

            target_snr_db = metrics["target_snr_db"].reshape(
                S,
                C,
            )

            # ============================================================
            # 9-4. Robust communication objective
            #      與 evaluate_new.py 完全一致
            # ============================================================
            worstUE_rate = torch.min(
                allUE_rate,
                dim=2,
            ).values

            robust_per_channel = torch.quantile(
                worstUE_rate,
                OUTAGE_QUANTILE,
                dim=0,
            )

            robust_all.append(
                robust_per_channel.cpu()
            )

            user_rate_q05 = torch.quantile(
                allUE_rate,
                OUTAGE_QUANTILE,
                dim=0,
            )

            user_rate_q05_sum += torch.sum(
                user_rate_q05,
                dim=0,
            )

            # ============================================================
            # 9-5. Sensing empirical outage probability
            #      Test 不需要 gradient，直接統計真實 violation fraction
            # ============================================================
            target_snr_sum += float(
                torch.sum(target_snr).cpu()
            )

            sensing_violation_count += int(
                torch.sum(
                    target_snr_db
                    < SENSING_SNR_THRESHOLD_DB
                ).cpu()
            )

            # ============================================================
            # 9-6. Mean SINR components
            # ============================================================
            signal_sum += torch.sum(
                metrics["signal"].reshape(S,C,K),
                dim=(0,1),
            )

            comm_interf_sum += torch.sum(
                metrics["comm_interf"].reshape(S,C,K),
                dim=(0,1),
            )

            radar_interf_sum += torch.sum(
                metrics["radar_interf"].reshape(S,C,K),
                dim=(0,1),
            )

            sensing_total_count += S*C
            metric_total_count += S*C

        # ================================================================
        # 10. Final robust metrics
        # ================================================================
        robust = torch.mean(
            torch.cat(robust_all)
        )

        user_rate_q05_mean = (
            user_rate_q05_sum
            / B
        )

        target_snr_mean = (
            target_snr_sum
            / sensing_total_count
        )

        target_snr_mean_db = (
            10.0
            * np.log10(
                max(
                    target_snr_mean,
                    1e-12,
                )
            )
        )

        sensing_violation_percent = (
            100.0
            * sensing_violation_count
            / sensing_total_count
        )

        signal_mean = (
            signal_sum
            / metric_total_count
        )

        comm_interf_mean = (
            comm_interf_sum
            / metric_total_count
        )

        radar_interf_mean = (
            radar_interf_sum
            / metric_total_count
        )

        # ================================================================
        # 11. Print
        # ================================================================
        print("\n" + "=" * 94)
        print("[RZF + NS-MRT BASELINE ROBUST TEST EVALUATION]")
        print("=" * 94)

        print(f"Test trajectories                     : {test_trajectories}")
        print(f"Test slots                            : {B}")
        print(f"RIS blocks                            : {B_block}")
        print(f"RIS phase                             : Trained ThetaNet, block-fixed")
        print(f"RZF lambda                            : {RZF_LAMBDA:g}")
        print(f"Communication beamformer              : RZF")
        print(f"Radar beamformer                      : H_eff_H nullspace MRT")
        print(f"Power split P_C / P_R                 : 0.40 / 0.60")
        print(f"Mean P_C                              : {float(W_C_power.cpu()):.6f} W")
        print(f"Mean P_R                              : {float(W_R_power.cpu()):.6f} W")
        print(f"Mean total TX power                   : {float(total_power.cpu()):.6f} W")
        print(f"Maximum total TX power                : {float(max_total_power.cpu()):.6f} W")
        print(f"TX power limit                        : {float(TRANSMIT_POWER_TOTAL):.6f} W")
        print(f"Injection samples S                   : {S}")
        print(f"Injection variance                    : {INJECTION_VARIANCE}")
        print(f"Tail quantile                         : {OUTAGE_QUANTILE:.2f}")

        print("-" * 94)

        print(
            f"Baseline robust minimum-user rate "
            f"Q{OUTAGE_QUANTILE:.2f} : "
            f"{float(robust):.6f} bps/Hz"
        )

        print(
            f"Baseline per-UE rate "
            f"Q{OUTAGE_QUANTILE:.2f} mean          : "
            f"{fmt_vec_sci(user_rate_q05_mean.cpu().numpy(),precision=4)} "
            f"bps/Hz"
        )

        print(
            f"Baseline robust target SNR mean       : "
            f"{target_snr_mean_db:.3f} dB"
        )

        print(
            f"Baseline sensing SNR threshold        : "
            f"{SENSING_SNR_THRESHOLD_DB:.3f} dB"
        )

        print(
            f"Baseline sensing violation count      : "
            f"{sensing_violation_count} / {sensing_total_count}"
        )

        print(
            f"Baseline sensing violation percentage : "
            f"{sensing_violation_percent:.4f}%"
        )

        print(
            f"Baseline robust signal power          : "
            f"{fmt_vec_sci(signal_mean.cpu().numpy(),precision=3)}"
        )

        print(
            f"Baseline robust communication interf. : "
            f"{fmt_vec_sci(comm_interf_mean.cpu().numpy(),precision=3)}"
        )

        print(
            f"Baseline robust radar interference    : "
            f"{fmt_vec_sci(radar_interf_mean.cpu().numpy(),precision=3)}"
        )

        print(
            f"Baseline robust noise power           : "
            f"{float(NOISE_POWER):.3e}"
        )

        print("=" * 94)
