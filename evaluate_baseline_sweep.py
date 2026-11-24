# -*- coding: utf-8 -*-
import os
import csv
import math
import argparse
import numpy as np
import torch

from settings import *
from two_timescale_NN import CommNet, ThetaNet
from baseline import (
    make_rzf_beamformer,
    mrt_in_H_eff_H_nullspace,
    RZF_LAMBDA,
)


# ================================================================
# Helpers
# ================================================================
def complex_awgn(shape, variance: float, device, cdtype: torch.dtype):
    """
    CN(0,variance):
        E|n|^2 = variance
        Re/Im ~ N(0,variance/2)
    """
    sigma = math.sqrt(variance / 2.0)

    nr = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    ) * sigma

    ni = torch.randn(
        shape,
        device=device,
        dtype=torch.float32,
    ) * sigma

    return torch.complex(
        nr,
        ni,
    ).to(dtype=cdtype)


def fmt_vec_sci(x, precision=3):
    x = np.asarray(x).reshape(-1)

    return "{" + " ".join(
        [
            f"[{float(v):.{precision}e}]"
            for v in x
        ]
    ) + "}"


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Sweep RZF + NS-MRT baseline communication/radar power allocation."
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="P_C/P_T sweep step. Default = 0.05.",
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=200,
        help="Evaluation chunk size. Default = 200 test slots.",
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=20260902,
        help="Random seed used only for test uncertainty realizations.",
    )

    args = parser.parse_args()

    POWER_SPLIT_STEP = float(args.step)
    EVAL_CHUNK_SIZE = int(args.chunk)
    EVAL_SEED = int(args.eval_seed)

    if POWER_SPLIT_STEP <= 0.0 or POWER_SPLIT_STEP > 1.0:
        raise ValueError(
            f"--step 必須介於 (0,1]，目前={POWER_SPLIT_STEP}"
        )

    # ================================================================
    # 1. Power sweep
    #
    #    power_ratio_C:
    #        0.00 -> P_C=0%,   P_R=100%
    #        0.40 -> P_C=40%,  P_R=60%
    #        1.00 -> P_C=100%, P_R=0%
    # ================================================================
    power_ratio_C_list = np.arange(
        0.0,
        1.0 + POWER_SPLIT_STEP / 2.0,
        POWER_SPLIT_STEP,
        dtype=np.float64,
    )

    power_ratio_C_list = np.clip(
        power_ratio_C_list,
        0.0,
        1.0,
    )

    power_ratio_C_list = np.unique(
        np.round(
            power_ratio_C_list,
            10,
        )
    )

    # 保證 0、0.4、1 一定存在，方便與原本 baseline 40/60 比較
    power_ratio_C_list = np.unique(
        np.concatenate(
            [
                power_ratio_C_list,
                np.asarray(
                    [0.0, 0.4, 1.0],
                    dtype=np.float64,
                ),
            ]
        )
    )

    power_ratio_C_list.sort()

    num_splits = len(
        power_ratio_C_list
    )

    power_ratio_R_list = (
        1.0
        - power_ratio_C_list
    )

    # ================================================================
    # 2. Paths
    # ================================================================
    test_dataset_path = os.path.join(
        DATA_DIR,
        "dataset_test.npz",
    )

    pretrained_theta_ckpt = os.path.join(
        PRETRAIN_DIR,
        "ris_only.ckpt",
    )

    result_csv_path = os.path.join(
        RESULT_DIR,
        "baseline_power_sweep_test.csv",
    )

    result_npz_path = os.path.join(
        RESULT_DIR,
        "baseline_power_sweep_test.npz",
    )

    for required_path in [
        test_dataset_path,
        pretrained_theta_ckpt,
    ]:
        if not os.path.exists(
            required_path
        ):
            raise FileNotFoundError(
                f"找不到檔案: {required_path}"
            )

    # ================================================================
    # 3. Networks
    #
    #    ThetaNet:
    #        與 REG / ROB 完全共用同一個 trained ris_only.ckpt
    #
    #    physics_net:
    #        只使用共用 physics functions
    # ================================================================
    theta_net = ThetaNet().to(
        DEVICE
    )

    physics_net = CommNet().to(
        DEVICE
    )

    theta_net.load_model(
        pretrained_theta_ckpt,
        strict=True,
        verbose=True,
    )

    theta_net.eval()
    physics_net.eval()

    # ================================================================
    # 4. Test dataset
    # ================================================================
    with np.load(
        test_dataset_path
    ) as data:

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

    test_trajectories = (
        test_dataset["h_dk"].shape[0]
    )

    test_channels = (
        test_dataset["h_dk"].shape[1]
    )

    print(
        "\n[INFO] 載入 baseline power-sweep test dataset 與 ThetaNet ..."
    )

    print(
        f"[INFO] test_dataset_path      = {test_dataset_path}"
    )

    print(
        f"[INFO] pretrained_theta_ckpt  = {pretrained_theta_ckpt}"
    )

    print(
        f"[INFO] result_csv_path        = {result_csv_path}"
    )

    print(
        f"[INFO] result_npz_path        = {result_npz_path}"
    )

    print(
        f"[INFO] test trajectories      = {test_trajectories}"
    )

    print(
        f"[INFO] test channels          = {test_channels}"
    )

    print(
        f"[INFO] h_dk shape             = {test_dataset['h_dk'].shape}"
    )

    print(
        f"[INFO] h_rk shape             = {test_dataset['h_rk'].shape}"
    )

    print(
        f"[INFO] G shape                = {test_dataset['G'].shape}"
    )

    print(
        f"[INFO] g_dt shape             = {test_dataset['g_dt'].shape}"
    )

    print(
        f"[INFO] power split step       = {POWER_SPLIT_STEP:g}"
    )

    print(
        f"[INFO] number of power splits = {num_splits}"
    )

    print(
        f"[INFO] P_C/P_T sweep          = {power_ratio_C_list}"
    )

    # ================================================================
    # 5. 固定 evaluation seed
    #
    #    所有 power split 在同一個 chunk 裡共用完全相同的
    #    injected h_dk / h_rk / G / g_dt。
    #
    #    因此不同 power split 之間是 paired comparison。
    # ================================================================
    np.random.seed(
        EVAL_SEED
    )

    torch.manual_seed(
        EVAL_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            EVAL_SEED
        )

    print(
        f"[INFO] evaluation seed        = {EVAL_SEED}"
    )

    with torch.no_grad():

        # ================================================================
        # 6. 建立與 evaluate_new.py 完全相同的 two-timescale test blocks
        # ================================================================
        channel_ids = np.arange(
            0,
            test_channels - LT_BLOCK_SIZE + 1,
            LT_BLOCK_SIZE,
        )

        trajectory_ids = np.repeat(
            np.arange(
                test_trajectories
            ),
            len(
                channel_ids
            ),
        )

        channel_ids = np.tile(
            channel_ids,
            test_trajectories,
        )

        B_block = len(
            channel_ids
        )

        # ================================================================
        # 7. 每個 LT block start slot 使用同一 trained ThetaNet
        # ================================================================
        h_dk_start = torch.as_tensor(
            test_dataset["h_dk"][
                trajectory_ids,
                channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        h_rk_start = torch.as_tensor(
            test_dataset["h_rk"][
                trajectory_ids,
                channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        G_start = torch.as_tensor(
            test_dataset["G"][
                trajectory_ids,
                channel_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        g_dt_start = torch.as_tensor(
            test_dataset["g_dt"][
                trajectory_ids,
                channel_ids,
            ],
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
        # 8. 同一 theta 固定使用 LT_BLOCK_SIZE 個 slots
        # ================================================================
        block_ids = (
            channel_ids[:, None]
            + np.arange(
                LT_BLOCK_SIZE
            )[None, :]
        )

        test_h_dk_hat = torch.as_tensor(
            test_dataset["h_dk"][
                trajectory_ids[:, None],
                block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_h_rk_hat = torch.as_tensor(
            test_dataset["h_rk"][
                trajectory_ids[:, None],
                block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_G_hat = torch.as_tensor(
            test_dataset["G"][
                trajectory_ids[:, None],
                block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        test_g_dt_hat = torch.as_tensor(
            test_dataset["g_dt"][
                trajectory_ids[:, None],
                block_ids,
            ],
            dtype=torch.complex64,
            device=DEVICE,
        )

        B = (
            B_block
            * LT_BLOCK_SIZE
        )

        M = TX_ANT
        N = RIS_UNIT
        K = UAV_COMM
        R = RADAR_STREAMS
        S = INJECTION_SAMPLES

        test_h_dk_hat = (
            test_h_dk_hat
            .reshape(
                B,
                M,
                K,
            )
        )

        test_h_rk_hat = (
            test_h_rk_hat
            .reshape(
                B,
                N,
                K,
            )
        )

        test_G_hat = (
            test_G_hat
            .reshape(
                B,
                N,
                M,
            )
        )

        test_g_dt_hat = (
            test_g_dt_hat
            .reshape(
                B,
                M,
                1,
            )
        )

        test_theta = (
            test_theta[:, None, :]
            .expand(
                -1,
                LT_BLOCK_SIZE,
                -1,
            )
            .reshape(
                B,
                N,
            )
        )

        print(
            f"[INFO] RIS blocks             = {B_block}"
        )

        print(
            f"[INFO] evaluated test slots   = {B}"
        )

        # ================================================================
        # 9. Analytical baseline directions
        #
        #    Communication direction:
        #        RZF
        #
        #    Radar direction:
        #        H_eff^H nullspace MRT
        #
        #    注意：
        #        這裡完全不使用 beamformers_power_split()
        #        因為這次要 sweep P_C / P_R。
        #
        #        W_C_dir 與 W_R_dir 本身仍使用 baseline.py
        #        原本的方向設計。
        # ================================================================
        H_eff_H_hat = (
            physics_net
            .compute_effective_channel(
                test_h_dk_hat,
                test_h_rk_hat,
                test_G_hat,
                test_theta,
            )
        )

        W_C_dir = make_rzf_beamformer(
            H_eff_H_hat,
            RZF_LAMBDA,
        )

        W_R_dir = mrt_in_H_eff_H_nullspace(
            H_eff_H_hat,
            test_g_dt_hat,
        )

        if W_C_dir.shape != (
            B,
            M,
            K,
        ):
            raise ValueError(
                f"W_C_dir shape error: "
                f"expected {(B,M,K)}, "
                f"got {tuple(W_C_dir.shape)}"
            )

        if W_R_dir.shape != (
            B,
            M,
            R,
        ):
            raise ValueError(
                f"W_R_dir shape error: "
                f"expected {(B,M,R)}, "
                f"got {tuple(W_R_dir.shape)}"
            )

        # ================================================================
        # 10. Direction power check
        #
        #     baseline.py 的 RZF / NS-MRT direction 原本就會
        #     normalize 成每個 slot Frobenius power = 1。
        #
        #     這裡只檢查，不重新改方向。
        # ================================================================
        W_C_dir_power_per_slot = torch.sum(
            torch.abs(
                W_C_dir
            ) ** 2,
            dim=(1, 2),
        ).real

        W_R_dir_power_per_slot = torch.sum(
            torch.abs(
                W_R_dir
            ) ** 2,
            dim=(1, 2),
        ).real

        print(
            f"[INFO] Mean ||W_C_dir||_F^2   = "
            f"{float(torch.mean(W_C_dir_power_per_slot).cpu()):.8f}"
        )

        print(
            f"[INFO] Mean ||W_R_dir||_F^2   = "
            f"{float(torch.mean(W_R_dir_power_per_slot).cpu()):.8f}"
        )

        print(
            f"[INFO] Min  ||W_C_dir||_F^2   = "
            f"{float(torch.min(W_C_dir_power_per_slot).cpu()):.8f}"
        )

        print(
            f"[INFO] Min  ||W_R_dir||_F^2   = "
            f"{float(torch.min(W_R_dir_power_per_slot).cpu()):.8f}"
        )

        print(
            f"[INFO] Max  ||W_C_dir||_F^2   = "
            f"{float(torch.max(W_C_dir_power_per_slot).cpu()):.8f}"
        )

        print(
            f"[INFO] Max  ||W_R_dir||_F^2   = "
            f"{float(torch.max(W_R_dir_power_per_slot).cpu()):.8f}"
        )

        # ================================================================
        # 11. 每個 power split 的累加器
        # ================================================================
        robust_sum = np.zeros(
            num_splits,
            dtype=np.float64,
        )

        user_rate_q05_sum = np.zeros(
            (
                num_splits,
                K,
            ),
            dtype=np.float64,
        )

        target_snr_sum = np.zeros(
            num_splits,
            dtype=np.float64,
        )

        sensing_violation_count = np.zeros(
            num_splits,
            dtype=np.int64,
        )

        # ------------------------------------------------
        # 每個 short-term slot：
        # 在 S 組 uncertainty 下的 violation probability
        # 若 > TAR_OUTAGE_QUANTILE，算該 slot chance constraint failure
        # ------------------------------------------------
        sensing_slot_failure_count = np.zeros(
            num_splits,
            dtype=np.int64,
        )

        # ------------------------------------------------
        # 與 ROB training surrogate 完全同形式：
        #
        # Q10(target SNR dB) < threshold
        #
        # 就算該 short-term slot Q10 constraint failure
        # ------------------------------------------------
        sensing_q10_failure_count = np.zeros(
            num_splits,
            dtype=np.int64,
        )

        signal_sum = np.zeros(
            (
                num_splits,
                K,
            ),
            dtype=np.float64,
        )

        comm_interf_sum = np.zeros(
            (
                num_splits,
                K,
            ),
            dtype=np.float64,
        )

        radar_interf_sum = np.zeros(
            (
                num_splits,
                K,
            ),
            dtype=np.float64,
        )

        # ================================================================
        # 12. Robust test evaluation
        #
        #     重要：
        #     每個 chunk 只生成一次 injection channels，
        #     所有 power split 共用完全相同 injected true channels。
        #
        #     這樣不同 power split 之間沒有 Monte Carlo sample 差異。
        # ================================================================
        print(
            "\n[INFO] Start baseline power sweep robust evaluation ..."
        )

        for start in range(
            0,
            B,
            EVAL_CHUNK_SIZE,
        ):

            end = min(
                start
                + EVAL_CHUNK_SIZE,
                B,
            )

            C = (
                end
                - start
            )

            print(
                f"\r[SWEEP] test slots "
                f"{start:6d} - {end:6d} / {B}",
                end="",
                flush=True,
            )

            # ============================================================
            # 12-1. Nominal channels
            # ============================================================
            h_dk_hat = (
                test_h_dk_hat[
                    start:end
                ]
            )

            h_rk_hat = (
                test_h_rk_hat[
                    start:end
                ]
            )

            G_hat = (
                test_G_hat[
                    start:end
                ]
            )

            g_dt_hat = (
                test_g_dt_hat[
                    start:end
                ]
            )

            theta_chunk = (
                test_theta[
                    start:end
                ]
            )

            W_C_dir_chunk = (
                W_C_dir[
                    start:end
                ]
            )

            W_R_dir_chunk = (
                W_R_dir[
                    start:end
                ]
            )

            # ============================================================
            # 12-2. Injection channels
            #       每個 chunk 只生成一次
            # ============================================================
            inj_h_dk_rep = (
                h_dk_hat
                .unsqueeze(0)
                .expand(
                    S,
                    C,
                    M,
                    K,
                )
            )

            inj_h_rk_rep = (
                h_rk_hat
                .unsqueeze(0)
                .expand(
                    S,
                    C,
                    N,
                    K,
                )
            )

            inj_G_rep = (
                G_hat
                .unsqueeze(0)
                .expand(
                    S,
                    C,
                    N,
                    M,
                )
            )

            inj_g_dt_rep = (
                g_dt_hat
                .unsqueeze(0)
                .expand(
                    S,
                    C,
                    M,
                    1,
                )
            )

            inj_h_dk_power = torch.mean(
                torch.abs(
                    inj_h_dk_rep
                ) ** 2,
                dim=(2, 3),
                keepdim=True,
            ).real

            inj_h_rk_power = torch.mean(
                torch.abs(
                    inj_h_rk_rep
                ) ** 2,
                dim=(2, 3),
                keepdim=True,
            ).real

            inj_G_power = torch.mean(
                torch.abs(
                    inj_G_rep
                ) ** 2,
                dim=(2, 3),
                keepdim=True,
            ).real

            inj_g_dt_power = torch.mean(
                torch.abs(
                    inj_g_dt_rep
                ) ** 2,
                dim=(2, 3),
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
            # 12-3. Flatten injected true channels
            # ============================================================
            inj_h_dk_flat = (
                inj_h_dk
                .reshape(
                    S * C,
                    M,
                    K,
                )
            )

            inj_h_rk_flat = (
                inj_h_rk
                .reshape(
                    S * C,
                    N,
                    K,
                )
            )

            inj_G_flat = (
                inj_G
                .reshape(
                    S * C,
                    N,
                    M,
                )
            )

            inj_g_dt_flat = (
                inj_g_dt
                .reshape(
                    S * C,
                    M,
                    1,
                )
            )

            theta_flat = (
                theta_chunk
                .unsqueeze(0)
                .expand(
                    S,
                    C,
                    N,
                )
                .reshape(
                    S * C,
                    N,
                )
            )

            # ============================================================
            # 12-4. Injected true effective channel
            #
            #       所有 power splits 共用同一 H_eff_H_inj，
            #       不需要每個 split 重新計算。
            # ============================================================
            H_eff_H_inj = (
                physics_net
                .compute_effective_channel(
                    inj_h_dk_flat,
                    inj_h_rk_flat,
                    inj_G_flat,
                    theta_flat,
                )
            )

            # ============================================================
            # 12-5. Sweep each P_C / P_R
            # ============================================================
            for split_idx, power_ratio_C in enumerate(
                power_ratio_C_list
            ):

                power_ratio_R = (
                    1.0
                    - power_ratio_C
                )

                p_C = (
                    TRANSMIT_POWER_TOTAL
                    * power_ratio_C
                )

                p_R = (
                    TRANSMIT_POWER_TOTAL
                    * power_ratio_R
                )

                # ========================================================
                # 固定 RZF / NS-MRT directions，只改 power
                # ========================================================
                W_C_chunk = (
                    math.sqrt(
                        p_C
                    )
                    * W_C_dir_chunk
                )

                W_R_chunk = (
                    math.sqrt(
                        p_R
                    )
                    * W_R_dir_chunk
                )

                # ========================================================
                # 每個 nominal slot 的 beamformer
                # 在 S 組 uncertainty realizations 下保持固定
                # ========================================================
                W_C_flat = (
                    W_C_chunk
                    .unsqueeze(0)
                    .expand(
                        S,
                        C,
                        M,
                        K,
                    )
                    .reshape(
                        S * C,
                        M,
                        K,
                    )
                )

                W_R_flat = (
                    W_R_chunk
                    .unsqueeze(0)
                    .expand(
                        S,
                        C,
                        M,
                        R,
                    )
                    .reshape(
                        S * C,
                        M,
                        R,
                    )
                )

                # ========================================================
                # 使用同一 injected true channel 評估該 power split
                # ========================================================
                metrics = (
                    physics_net
                    .compute_isac_batch_performance(
                        H_eff_H_inj,
                        inj_g_dt_flat,
                        W_C_flat,
                        W_R_flat,
                    )
                )

                allUE_rate = (
                    metrics["rate"]
                    .reshape(
                        S,
                        C,
                        K,
                    )
                )

                target_snr = (
                    metrics["target_snr"]
                    .reshape(
                        S,
                        C,
                    )
                )

                target_snr_db = (
                    metrics["target_snr_db"]
                    .reshape(
                        S,
                        C,
                    )
                )

                # ========================================================
                # Communication robust objective
                #
                # 每 realization:
                #     先對 K 個 UE 取 minimum rate
                #
                # 每 nominal slot:
                #     再對 S 組 uncertainty 取 Q0.05
                # ========================================================
                worstUE_rate = torch.min(
                    allUE_rate,
                    dim=2,
                ).values

                robust_per_slot = torch.quantile(
                    worstUE_rate,
                    OUTAGE_QUANTILE,
                    dim=0,
                )

                robust_sum[
                    split_idx
                ] += float(
                    torch.sum(
                        robust_per_slot
                    ).cpu()
                )

                # ========================================================
                # Per-UE Q0.05
                # ========================================================
                user_rate_q05 = torch.quantile(
                    allUE_rate,
                    OUTAGE_QUANTILE,
                    dim=0,
                )

                user_rate_q05_sum[
                    split_idx
                ] += (
                    torch.sum(
                        user_rate_q05,
                        dim=0,
                    )
                    .cpu()
                    .numpy()
                )

                # ========================================================
                # Mean target SNR
                # ========================================================
                target_snr_sum[
                    split_idx
                ] += float(
                    torch.sum(
                        target_snr
                    ).cpu()
                )

                # ========================================================
                # Overall empirical sensing violation
                #
                # 所有 S*C realizations:
                #     target SNR < threshold
                # ========================================================
                violation_mask = (
                    target_snr_db
                    < SENSING_SNR_THRESHOLD_DB
                )

                sensing_violation_count[
                    split_idx
                ] += int(
                    torch.sum(
                        violation_mask
                    ).cpu()
                )

                # ========================================================
                # Slot-level empirical outage constraint
                #
                # 每一個 short-term slot:
                #     violation probability over S uncertainty samples
                #
                # 若 > 10%，該 slot 判定 chance constraint failure
                # ========================================================
                sensing_violation_per_slot = torch.mean(
                    violation_mask.float(),
                    dim=0,
                )

                sensing_slot_failure_count[
                    split_idx
                ] += int(
                    torch.sum(
                        sensing_violation_per_slot
                        > TAR_OUTAGE_QUANTILE
                    ).cpu()
                )

                # ========================================================
                # ROB training 同形式的 Q10 sensing condition
                #
                # 每個 slot:
                #     Q10(target SNR dB) >= threshold
                # ========================================================
                target_snr_q10_db = torch.quantile(
                    target_snr_db,
                    TAR_OUTAGE_QUANTILE,
                    dim=0,
                )

                sensing_q10_failure_count[
                    split_idx
                ] += int(
                    torch.sum(
                        target_snr_q10_db
                        < SENSING_SNR_THRESHOLD_DB
                    ).cpu()
                )

                # ========================================================
                # Mean SINR components
                # ========================================================
                signal_sum[
                    split_idx
                ] += (
                    torch.sum(
                        metrics["signal"]
                        .reshape(
                            S,
                            C,
                            K,
                        ),
                        dim=(0, 1),
                    )
                    .cpu()
                    .numpy()
                )

                comm_interf_sum[
                    split_idx
                ] += (
                    torch.sum(
                        metrics["comm_interf"]
                        .reshape(
                            S,
                            C,
                            K,
                        ),
                        dim=(0, 1),
                    )
                    .cpu()
                    .numpy()
                )

                radar_interf_sum[
                    split_idx
                ] += (
                    torch.sum(
                        metrics["radar_interf"]
                        .reshape(
                            S,
                            C,
                            K,
                        ),
                        dim=(0, 1),
                    )
                    .cpu()
                    .numpy()
                )

            # ============================================================
            # 釋放大型 chunk tensors
            # ============================================================
            del inj_h_dk
            del inj_h_rk
            del inj_G
            del inj_g_dt

            del inj_h_dk_flat
            del inj_h_rk_flat
            del inj_G_flat
            del inj_g_dt_flat

            del theta_flat
            del H_eff_H_inj

        print()

        # ================================================================
        # 13. Final metrics
        # ================================================================
        robust_rate = (
            robust_sum
            / B
        )

        user_rate_q05_mean = (
            user_rate_q05_sum
            / B
        )

        total_uncertainty_count = (
            S
            * B
        )

        target_snr_mean = (
            target_snr_sum
            / total_uncertainty_count
        )

        target_snr_mean_db = (
            10.0
            * np.log10(
                np.maximum(
                    target_snr_mean,
                    1e-12,
                )
            )
        )

        sensing_violation_percent = (
            100.0
            * sensing_violation_count
            / total_uncertainty_count
        )

        sensing_slot_failure_percent = (
            100.0
            * sensing_slot_failure_count
            / B
        )

        sensing_q10_failure_percent = (
            100.0
            * sensing_q10_failure_count
            / B
        )

        signal_mean = (
            signal_sum
            / total_uncertainty_count
        )

        comm_interf_mean = (
            comm_interf_sum
            / total_uncertainty_count
        )

        radar_interf_mean = (
            radar_interf_sum
            / total_uncertainty_count
        )

        # ================================================================
        # 14. Actual TX powers
        #
        #     利用 baseline direction 的實際 Frobenius norm，
        #     不直接假設 norm 一定精確等於 1。
        # ================================================================
        mean_W_C_dir_power = float(
            torch.mean(
                W_C_dir_power_per_slot
            ).cpu()
        )

        mean_W_R_dir_power = float(
            torch.mean(
                W_R_dir_power_per_slot
            ).cpu()
        )

        mean_P_C = (
            TRANSMIT_POWER_TOTAL
            * power_ratio_C_list
            * mean_W_C_dir_power
        )

        mean_P_R = (
            TRANSMIT_POWER_TOTAL
            * power_ratio_R_list
            * mean_W_R_dir_power
        )

        mean_total_power = (
            mean_P_C
            + mean_P_R
        )

        # ================================================================
        # 15. Overall sensing-feasible mask
        #
        #     目前先沿用你 evaluate_new.py 最直觀的 test 定義：
        #
        #         overall empirical violation <= 10%
        #
        #     同時另外報 slot-level 與 Q10 failure，
        #     不把不同 sensing definition 混在一起。
        # ================================================================
        sensing_limit_percent = (
            100.0
            * TAR_OUTAGE_QUANTILE
        )

        overall_feasible = (
            sensing_violation_percent
            <= sensing_limit_percent
        )

        # ================================================================
        # 16. Print sweep table
        # ================================================================
        print(
            "\n"
            + "=" * 132
        )

        print(
            "[RZF + NS-MRT BASELINE POWER-SPLIT ROBUST TEST SWEEP]"
        )

        print(
            "=" * 132
        )

        print(
            f"Test trajectories          : {test_trajectories}"
        )

        print(
            f"Test slots                 : {B}"
        )

        print(
            f"RIS blocks                 : {B_block}"
        )

        print(
            f"RIS phase                  : Trained ThetaNet, block-fixed"
        )

        print(
            f"RZF lambda                 : {RZF_LAMBDA:g}"
        )

        print(
            f"Communication beamformer   : RZF direction"
        )

        print(
            f"Radar beamformer           : H_eff_H nullspace MRT direction"
        )

        print(
            f"TX power                   : {TRANSMIT_POWER_TOTAL:.6f} W"
        )

        print(
            f"Injection samples          : {S}"
        )

        print(
            f"Injection variance         : {INJECTION_VARIANCE}"
        )

        print(
            f"Communication tail Q       : {OUTAGE_QUANTILE:.2f}"
        )

        print(
            f"Sensing outage limit       : {TAR_OUTAGE_QUANTILE:.2f}"
        )

        print(
            f"Sensing SNR threshold      : {SENSING_SNR_THRESHOLD_DB:.3f} dB"
        )

        print(
            "-" * 132
        )

        print(
            f"{'P_C/P_T':>8} "
            f"{'P_R/P_T':>8} "
            f"{'P_C(W)':>9} "
            f"{'P_R(W)':>9} "
            f"{'RobustQ05':>11} "
            f"{'TarSNR(dB)':>11} "
            f"{'OverallViol%':>13} "
            f"{'SlotFail%':>10} "
            f"{'Q10Fail%':>10} "
            f"{'Feasible':>9}"
        )

        print(
            "-" * 132
        )

        for i in range(
            num_splits
        ):

            feasible_text = (
                "YES"
                if overall_feasible[i]
                else "NO"
            )

            print(
                f"{power_ratio_C_list[i]:8.2f} "
                f"{power_ratio_R_list[i]:8.2f} "
                f"{mean_P_C[i]:9.4f} "
                f"{mean_P_R[i]:9.4f} "
                f"{robust_rate[i]:11.6f} "
                f"{target_snr_mean_db[i]:11.3f} "
                f"{sensing_violation_percent[i]:13.4f} "
                f"{sensing_slot_failure_percent[i]:10.4f} "
                f"{sensing_q10_failure_percent[i]:10.4f} "
                f"{feasible_text:>9}"
            )

        print(
            "=" * 132
        )

        # ================================================================
        # 17. Best unconstrained robust split
        # ================================================================
        best_unconstrained_idx = int(
            np.argmax(
                robust_rate
            )
        )

        print(
            "\n[BEST ROBUST RATE WITHOUT SENSING FEASIBILITY FILTER]"
        )

        print(
            f"P_C / P_R                  : "
            f"{power_ratio_C_list[best_unconstrained_idx]:.2f} / "
            f"{power_ratio_R_list[best_unconstrained_idx]:.2f}"
        )

        print(
            f"Robust minimum-user Q{OUTAGE_QUANTILE:.2f} : "
            f"{robust_rate[best_unconstrained_idx]:.6f} bps/Hz"
        )

        print(
            f"Target SNR mean            : "
            f"{target_snr_mean_db[best_unconstrained_idx]:.3f} dB"
        )

        print(
            f"Overall sensing violation  : "
            f"{sensing_violation_percent[best_unconstrained_idx]:.4f}%"
        )

        print(
            f"Slot failure percentage    : "
            f"{sensing_slot_failure_percent[best_unconstrained_idx]:.4f}%"
        )

        print(
            f"Q10 failure percentage     : "
            f"{sensing_q10_failure_percent[best_unconstrained_idx]:.4f}%"
        )

        # ================================================================
        # 18. Best overall-sensing-feasible split
        # ================================================================
        feasible_indices = np.where(
            overall_feasible
        )[0]

        print(
            "\n[BEST ROBUST RATE WITH OVERALL SENSING VIOLATION <= "
            f"{sensing_limit_percent:.1f}%]"
        )

        if len(
            feasible_indices
        ) == 0:

            print(
                "No feasible baseline power split."
            )

            min_violation_idx = int(
                np.argmin(
                    sensing_violation_percent
                )
            )

            print(
                f"Minimum overall sensing violation occurs at "
                f"P_C/P_R = "
                f"{power_ratio_C_list[min_violation_idx]:.2f}/"
                f"{power_ratio_R_list[min_violation_idx]:.2f}"
            )

            print(
                f"Minimum overall sensing violation = "
                f"{sensing_violation_percent[min_violation_idx]:.4f}%"
            )

            print(
                f"Robust rate at minimum violation  = "
                f"{robust_rate[min_violation_idx]:.6f} bps/Hz"
            )

        else:

            best_feasible_local_idx = int(
                np.argmax(
                    robust_rate[
                        feasible_indices
                    ]
                )
            )

            best_feasible_idx = int(
                feasible_indices[
                    best_feasible_local_idx
                ]
            )

            print(
                f"P_C / P_R                  : "
                f"{power_ratio_C_list[best_feasible_idx]:.2f} / "
                f"{power_ratio_R_list[best_feasible_idx]:.2f}"
            )

            print(
                f"Robust minimum-user Q{OUTAGE_QUANTILE:.2f} : "
                f"{robust_rate[best_feasible_idx]:.6f} bps/Hz"
            )

            print(
                f"Target SNR mean            : "
                f"{target_snr_mean_db[best_feasible_idx]:.3f} dB"
            )

            print(
                f"Overall sensing violation  : "
                f"{sensing_violation_percent[best_feasible_idx]:.4f}%"
            )

            print(
                f"Slot failure percentage    : "
                f"{sensing_slot_failure_percent[best_feasible_idx]:.4f}%"
            )

            print(
                f"Q10 failure percentage     : "
                f"{sensing_q10_failure_percent[best_feasible_idx]:.4f}%"
            )

        # ================================================================
        # 19. Save NPZ
        # ================================================================
        np.savez(
            result_npz_path,

            power_ratio_C=power_ratio_C_list.astype(
                np.float32
            ),

            power_ratio_R=power_ratio_R_list.astype(
                np.float32
            ),

            mean_P_C=mean_P_C.astype(
                np.float32
            ),

            mean_P_R=mean_P_R.astype(
                np.float32
            ),

            mean_total_power=mean_total_power.astype(
                np.float32
            ),

            robust_rate=robust_rate.astype(
                np.float32
            ),

            user_rate_q05_mean=user_rate_q05_mean.astype(
                np.float32
            ),

            target_snr_mean_db=target_snr_mean_db.astype(
                np.float32
            ),

            sensing_violation_percent=sensing_violation_percent.astype(
                np.float32
            ),

            sensing_slot_failure_percent=sensing_slot_failure_percent.astype(
                np.float32
            ),

            sensing_q10_failure_percent=sensing_q10_failure_percent.astype(
                np.float32
            ),

            signal_mean=signal_mean.astype(
                np.float32
            ),

            comm_interf_mean=comm_interf_mean.astype(
                np.float32
            ),

            radar_interf_mean=radar_interf_mean.astype(
                np.float32
            ),

            overall_feasible=overall_feasible.astype(
                np.int32
            ),

            sensing_snr_threshold_db=np.asarray(
                SENSING_SNR_THRESHOLD_DB,
                dtype=np.float32,
            ),

            sensing_outage_limit=np.asarray(
                TAR_OUTAGE_QUANTILE,
                dtype=np.float32,
            ),

            communication_outage_quantile=np.asarray(
                OUTAGE_QUANTILE,
                dtype=np.float32,
            ),

            injection_variance=np.asarray(
                INJECTION_VARIANCE,
                dtype=np.float32,
            ),

            injection_samples=np.asarray(
                INJECTION_SAMPLES,
                dtype=np.int32,
            ),

            evaluation_seed=np.asarray(
                EVAL_SEED,
                dtype=np.int64,
            ),
        )

        # ================================================================
        # 20. Save CSV
        # ================================================================
        with open(
            result_csv_path,
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:

            writer = csv.writer(
                f
            )

            header = [
                "P_C_ratio",
                "P_R_ratio",
                "mean_P_C_W",
                "mean_P_R_W",
                "mean_total_power_W",
                "robust_min_user_rate_Q05",
                "target_snr_mean_db",
                "sensing_violation_percent",
                "sensing_slot_failure_percent",
                "sensing_q10_failure_percent",
                "overall_feasible",
            ]

            for k in range(
                K
            ):
                header.append(
                    f"UE{k}_rate_Q05_mean"
                )

            for k in range(
                K
            ):
                header.append(
                    f"UE{k}_signal_mean"
                )

            for k in range(
                K
            ):
                header.append(
                    f"UE{k}_comm_interf_mean"
                )

            for k in range(
                K
            ):
                header.append(
                    f"UE{k}_radar_interf_mean"
                )

            writer.writerow(
                header
            )

            for i in range(
                num_splits
            ):

                row = [
                    float(
                        power_ratio_C_list[i]
                    ),
                    float(
                        power_ratio_R_list[i]
                    ),
                    float(
                        mean_P_C[i]
                    ),
                    float(
                        mean_P_R[i]
                    ),
                    float(
                        mean_total_power[i]
                    ),
                    float(
                        robust_rate[i]
                    ),
                    float(
                        target_snr_mean_db[i]
                    ),
                    float(
                        sensing_violation_percent[i]
                    ),
                    float(
                        sensing_slot_failure_percent[i]
                    ),
                    float(
                        sensing_q10_failure_percent[i]
                    ),
                    int(
                        overall_feasible[i]
                    ),
                ]

                row.extend(
                    user_rate_q05_mean[
                        i
                    ].tolist()
                )

                row.extend(
                    signal_mean[
                        i
                    ].tolist()
                )

                row.extend(
                    comm_interf_mean[
                        i
                    ].tolist()
                )

                row.extend(
                    radar_interf_mean[
                        i
                    ].tolist()
                )

                writer.writerow(
                    row
                )

        print(
            "\n[INFO] Baseline power sweep finished."
        )

        print(
            f"[INFO] Saved NPZ : {result_npz_path}"
        )

        print(
            f"[INFO] Saved CSV : {result_csv_path}"
        )