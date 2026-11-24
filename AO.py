# -*- coding: utf-8 -*-
import os

from settings import *
import numpy as np
import cvxpy as cp
from tqdm import tqdm


"""
AO6:Two-Timescale Robust AO Beamforming and RIS Optimization

本程式使用 Bounded error 求解 RIS 輔助 ISAC 系統的
最差使用者保守 SINR 最大化問題，並滿足發射功率與感測 SNR 限制。

採用 Two-Timescale 架構，每個 LT block 包含 10 個 time slots。

Phase I：首槽 AO（solve_ao）
    1. 初始化 RIS phi，並使用 MRT 初始化 W。
    2. 子問題一：固定 phi，使用 SCA + Bisection 更新 W。
       首輪使用 MRT，後續 AO 輪次繼承上一輪 W。
    3. 子問題二：固定 W，使用 QT + SCA + PCCP 更新 phi。
       最大化 Omega - mu * sum(delta)。
    4. PCCP 內層以 J_PCCP 的變化量判斷收斂。
    5. 每輪 AO 完成後，使用相同 mu 比較 J_AO。
       收斂則停止，否則繼續交替更新 W 與 phi。
    6. AO_MAX_ITERS 與 PCCP_MAX_ITERS 作為安全迭代上限。

Phase II：後續 9 個 slots（solve_SCA）
    1. 固定首槽 AO 求得的 RIS phi。
    2. 根據每個 slot 的通道重新計算等效通道。
    3. 每個 slot 獨立使用 MRT 初始化 W。
    4. 使用 SCA + Bisection 求解 W，不執行 AO。

輸出結果：
    AO_phi   : 每個 slot 的 RIS 反射向量。
    AO_W     : 每個 slot 的波束成形矩陣。
    success  : 各 slot 求解是否成功。
    status   : 各 slot 的求解狀態。

備註：
    AO6 的主要改進為 AO 專用的 W 更新方式，
    以及 PCCP 內層與 AO 外層的收斂判斷。
    保留原本的 Two-Timescale 資料處理與 NPZ 輸出架構。
"""


# ================================
# AO6 離線求解設定
# ================================
AO_SAVE_EVERY       = 25                    # 每處理完 25 個 LT block 就存一批，供中斷後續跑

EPSILON_K           = 0.2                   # 各 UE 的 bounded-error 半徑比例
EPSILON_G           = 0.2                   # 感測通道的 bounded-error 半徑比例
MAX_AO_BLOCKS       = 100                   # None: 所有軌跡的全部完整區間；整數 N: 第 0 條軌跡的前 N 個區間

AO_RESULT_FILENAME  = "ao6_test_results.npz"
AO_SAVE_PATH        = os.path.join(RESULT_DIR,AO_RESULT_FILENAME)

# AO 停止條件
AO_MAX_ITERS        = 12                    # 首槽 AO 最大迭代次數
AO_TOL              = 1e-4                  # AO 目標值改善量的停止容差

# 子問題2停止條件
PCCP_MAX_ITERS      = 30                    # RIS 子問題 QT + SCA + PCCP 最大迭代次數
PCCP_TOL            = 1e-4                  # PCCP 目標值改善量的停止容差
PCCP_MU_INITIAL     = 10.0                  # RIS 鬆弛懲罰係數初始值
PCCP_MU_MULTIPLIER  = 2.5                   # 完成一輪 AO 後更新 mu，PCCP 內層保持固定
PCCP_MU_MAX         = 1e4                   # RIS 鬆弛懲罰係數最大值

# 子問題1的 SCA 與 二分搜尋收斂條件 

SCA_MAX_ITERS       = 20      # 外層感測 SCA 的最大迭代次數
SCA_TOL             = 1e-4    # 判斷最低保守 SINR 改善量的容差

BISECTION_MAX_ITERS = 30      # 每輪 SCA 中，二分搜尋的最大迭代次數
BISECTION_TOL       = 1e-4    # 判斷 tau 搜尋區間寬度的容差

# ================================
# Select AO blocks
# ================================
def select_ao_blocks(trajectories,slots):
    # None：逐條軌跡選取全部完整區間，每個區間以 (trajectory_id, start_slot) 表示。
    if MAX_AO_BLOCKS is None:
        return [(trajectory_id,start_slot) for trajectory_id in range(trajectories)
                for start_slot in range(0,slots-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)]

    # 整數 N：選取第 0 條軌跡從第 0 槽開始的前 N 個區間。
    return [(0,block_id*LT_BLOCK_SIZE) for block_id in range(MAX_AO_BLOCKS)]


# ================================
# 固定 RIS，建立求 W 的凸可行性問題 (SCA)
# ================================
def solve_SCA(H_eff_H,g_dt,epsilon_k,epsilon_g):
    """
    固定當槽的 RIS 配置，求解波束成形矩陣 W。
    H_eff_H   : (K,M)，第 k 列為第 k 位 UE 的等效通道 h_eff,k^H。
    g_dt      : (M,1)，當槽的目標通道。
    epsilon_k : (K,)，各 UE 的通道誤差半徑。
    epsilon_g : scalar,目標通道的誤差半徑。
    """
    # 以滿功率 target MRT 初始化當槽的 W_sca_ref
    g_norm = np.linalg.norm(g_dt)
    target_direction = g_dt / g_norm
    W_sca_ref = np.sqrt(TRANSMIT_POWER_TOTAL / UAV_COMM) * np.repeat(target_direction, UAV_COMM, axis=1)

    # 用本輪等效通道對齊各欄相位，使期望訊號為非負實數(滿足phase rotation限制)
    for k in range(UAV_COMM):
        desired = H_eff_H[k] @ W_sca_ref[:, k]
        W_sca_ref[:, k] *= np.exp(-1j * np.angle(desired))

    # 以雜訊振幅為單位，正規化 等效通道與誤差 方便後面CVX解
    noise_std           = np.sqrt(NOISE_POWER)
    H_eff_H_scaled      = H_eff_H / noise_std
    epsilon_k_scaled    = epsilon_k / noise_std

    # 建立子問題一的待求波束矩陣，形狀為 (M, K)
    W = cp.Variable((TX_ANT, UAV_COMM), complex=True, name="W")

    # ================================
    # SCA 外層迴圈
    # ================================
    for sca_iter in range(SCA_MAX_ITERS):

        # 當輪W_sca_ref固定後，感測振幅的一階下界固定
        x   = (W_sca_ref.conj().T @ g_dt).ravel()
        y   = W.H @ g_dt[:, 0]
        L_g = cp.real(x.conj() @ y) / np.linalg.norm(x)

        # 本輪二分搜尋共用的功率與感測限制
        sensing_floor       = np.sqrt(SENSING_SNR_THRESHOLD * NOISE_POWER / RCS_POWER) / (g_norm - epsilon_g)
        power_constraint    = cp.norm(W, "fro") <= np.sqrt(TRANSMIT_POWER_TOTAL)
        sensing_constraint  = L_g / g_norm >= (epsilon_g * cp.norm(W, "fro") + sensing_floor) / g_norm

        # ================================
        # 進入二分搜尋前的初始化
        # ================================

        # 回算 W_sca_ref 的最低保守 SINR 這就是 tau_SCA_ref
        amplitude = np.abs(H_eff_H @ W_sca_ref)
        beam_norm = np.linalg.norm(W_sca_ref, axis=0)
        error_bound = epsilon_k[:, None] * beam_norm[None, :]
        desired_power = np.maximum(np.diag(amplitude - error_bound), 0.0)**2
        interference_power = (amplitude + error_bound)**2
        np.fill_diagonal(interference_power, 0.0)
        tau_SCA_ref = np.min(desired_power / (interference_power.sum(axis=1) + NOISE_POWER))

        # 二分搜初始值
        W_bisection_ref   = W_sca_ref
        tau_bisection_ref = tau_SCA_ref

        # 二分搜上下界
        tau_high = np.min(TRANSMIT_POWER_TOTAL * np.linalg.norm(H_eff_H,axis=1)**2 / NOISE_POWER)   # 假設無干擾、無通道誤差，且全部功率給單一 UE
        tau_low  = tau_bisection_ref                                                                # tau_bisection_ref 充當 本輪二分搜下界

        # ================================
        # Bisection 內層迴圈
        # ================================
        for bisection_iter in range(BISECTION_MAX_ITERS):

            # 任一側的界更新後，都在這裡判斷二分搜尋是否停止
            if tau_high - tau_low <= BISECTION_TOL:
                break

            tau = (tau_low + tau_high) / 2
            constraints = [power_constraint, sensing_constraint]

            # 相位旋轉限制
            for k in range(UAV_COMM):
                desired = H_eff_H_scaled[k] @ W[:, k]
                constraints += [cp.imag(desired) == 0, cp.real(desired) >= 0]

                # 其他波束的干擾振幅上界
                interference_bounds = []
                for ell in range(UAV_COMM):
                    if ell == k:
                        continue

                    interference_bound = cp.Variable(nonneg=True)
                    interference = cp.abs(H_eff_H_scaled[k] @ W[:, ell])
                    interference += epsilon_k_scaled[k] * cp.norm(W[:, ell], 2)
                    constraints.append(interference <= interference_bound)
                    interference_bounds.append(interference_bound)

                # 使用這次的tau算 SINR 限制
                interference_noise = cp.norm(cp.hstack(interference_bounds + [1.0]), 2)
                desired_lower      = cp.real(desired) - epsilon_k_scaled[k] * cp.norm(W[:, k], 2)
                constraints.append(np.sqrt(tau) * interference_noise <= desired_lower)

            # 解最佳化問題，只問是否存在可行的 W
            problem = cp.Problem(cp.Minimize(0), constraints)
            problem.solve(solver=cp.SCS, eps=1e-6, max_iters=15000)

            # 這次沒有確認達標，降低上界，下一輪再試
            if problem.status in (cp.INFEASIBLE, cp.OPTIMAL_INACCURATE, cp.INFEASIBLE_INACCURATE):
                # if problem.status != cp.INFEASIBLE:
                    # print(f"[子問題一] , SCA={sca_iter + 1}, 二分={bisection_iter + 1}, tau={tau:.9g}, status={problem.status}")
                tau_high = tau
                continue

            # 其他非預期狀態直接報錯
            if problem.status != cp.OPTIMAL:
                raise RuntimeError(f"子問題一狀態異常:tau={tau:.9g}, status={problem.status}")

            # 取出 SCS 求得的 W，回算最低保守 SINR
            W_bisection_ref = W.value.copy()
            amplitude   = np.abs(H_eff_H @ W_bisection_ref)
            beam_norm   = np.linalg.norm(W_bisection_ref, axis=0)
            error_bound = epsilon_k[:, None] * beam_norm[None, :]
            desired_power = np.maximum(np.diag(amplitude - error_bound), 0.0)**2
            interference_power = (amplitude + error_bound)**2
            np.fill_diagonal(interference_power, 0.0)
            tau_bisection_ref = np.min(desired_power / (interference_power.sum(axis=1) + NOISE_POWER))

            tau_low      = tau_bisection_ref

        # 二分搜尋結束，接收本輪保留的解
        sca_gain  = tau_bisection_ref - tau_SCA_ref
        W_sca_ref = W_bisection_ref

        # 本輪最低保守 SINR 的改善量夠小，停止 SCA
        if sca_gain <= SCA_TOL:
            return W_sca_ref, True, "step1_converged"

    return W_sca_ref, True, "step1_sca_max_iters"


# ================================
# 子問題1 & 2：(AO)
# ================================
def solve_ao(h_dk,h_rk,G,g_dt):
    """
    使用 LT block 首槽通道，最多執行 AO_MAX_ITERS 輪交替最佳化
    子問題一：固定 RIS 使用 Bisection + SCA 求 W
    首輪使用 MRT,後續 AO 輪次繼承上一輪 W, 這裡不能跟其他slot一樣

    子問題二：固定 W 使用 QT + SCA + PCCP 求 phi 最大化 Omega - mu * sum(delta)。
    h_dk : (M,K)
    h_rk : (N,K)
    G    : (N,M)
    g_dt : (M,1)
    """
    # ================================
    # 進入 AO 前的初始化
    # ================================

    # 初始化 RIS phi^(0)，所有反射係數設為 1（零相位）                (這裡符合RIS power限制)
    phi_AO_ref = np.ones(h_rk.shape[0],dtype=np.complex128)                # (N,) 

    # 使用初始 phi 計算等效通道
    H_eff_H = h_dk.conj().T + h_rk.conj().T @ np.diag(phi_AO_ref) @ G      # (K,M)

    # 決定本次 AO 使用的誤差半徑
    epsilon_k = EPSILON_K * np.linalg.norm(H_eff_H,axis=1)          # (K,)
    epsilon_g = EPSILON_G * np.linalg.norm(g_dt)                    # scalar

    # 初始化 W^(0)，使用滿功率 target MRT                            (這裡符合發射power)
    target_direction = g_dt / np.linalg.norm(g_dt)
    W_AO_ref = np.sqrt(TRANSMIT_POWER_TOTAL / UAV_COMM) * np.repeat(target_direction,UAV_COMM,axis=1)    # (M,K)

    # 懲罰係數從 10 開始 每輪AO乘上2.5倍
    mu = PCCP_MU_INITIAL

    # 檢查初始W^(0) phi^(0) 有沒有符合感測門檻
    g_norm = np.linalg.norm(g_dt)
    sensing_snr_MRT = RCS_POWER * TRANSMIT_POWER_TOTAL * (g_norm - epsilon_g)**4 / NOISE_POWER

    if sensing_snr_MRT < SENSING_SNR_THRESHOLD:
        raise RuntimeError("初始 MRT 的感測 SNR 未達門檻，停止 AO6!")

    for ao_iter in range(AO_MAX_ITERS):

        # ================================
        # 子問題一：進入SCA前的初始化
        # ================================

        # 使用當輪 phi 更新等效通道，epsilon_k 與 epsilon_g 保持固定
        H_eff_H = h_dk.conj().T + h_rk.conj().T @ np.diag(phi_AO_ref) @ G

        # 以目前已接受的 W_ref 作為本輪第一個 SCA 參考點
        W_sca_ref = W_AO_ref.copy()

        # 用本輪等效通道對齊各欄相位，使期望訊號為非負實數(滿足phase rotation限制)
        for k in range(UAV_COMM):
            desired = H_eff_H[k] @ W_sca_ref[:, k]
            W_sca_ref[:, k] *= np.exp(-1j * np.angle(desired))

        # 以雜訊振幅為單位，正規化 等效通道與誤差 方便後面CVX解
        noise_std           = np.sqrt(NOISE_POWER)
        H_eff_H_scaled      = H_eff_H / noise_std
        epsilon_k_scaled    = epsilon_k / noise_std

        # 建立子問題一的待求波束矩陣，形狀為 (M, K)
        W = cp.Variable((TX_ANT, UAV_COMM), complex=True, name="W")

        # ================================
        # SCA 外層迴圈
        # ================================
        for sca_iter in range(SCA_MAX_ITERS):

            # 當輪W_sca_ref固定後，感測振幅的一階下界固定
            x   = (W_sca_ref.conj().T @ g_dt).ravel()
            y   = W.H @ g_dt[:, 0]
            L_g = cp.real(x.conj() @ y) / np.linalg.norm(x)

            # 本輪二分搜尋共用的功率與感測限制
            sensing_floor       = np.sqrt(SENSING_SNR_THRESHOLD * NOISE_POWER / RCS_POWER) / (g_norm - epsilon_g)
            power_constraint    = cp.norm(W, "fro") <= np.sqrt(TRANSMIT_POWER_TOTAL)
            sensing_constraint  = L_g / g_norm >= (epsilon_g * cp.norm(W, "fro") + sensing_floor) / g_norm

            # ================================
            # 進入二分搜尋前的初始化
            # ================================

            # 回算 W_sca_ref 的最低保守 SINR 這就是 tau_SCA_ref
            amplitude = np.abs(H_eff_H @ W_sca_ref)
            beam_norm = np.linalg.norm(W_sca_ref, axis=0)
            error_bound = epsilon_k[:, None] * beam_norm[None, :]
            desired_power = np.maximum(np.diag(amplitude - error_bound), 0.0)**2
            interference_power = (amplitude + error_bound)**2
            np.fill_diagonal(interference_power, 0.0)
            tau_SCA_ref = np.min(desired_power / (interference_power.sum(axis=1) + NOISE_POWER))

            # 二分搜初始值
            W_bisection_ref   = W_sca_ref
            tau_bisection_ref = tau_SCA_ref

            # 二分搜上下界
            tau_high = np.min(TRANSMIT_POWER_TOTAL * np.linalg.norm(H_eff_H,axis=1)**2 / NOISE_POWER)   # 假設無干擾、無通道誤差，且全部功率給單一 UE
            tau_low  = tau_bisection_ref                                                                # tau_bisection_ref 充當 本輪二分搜下界

            # ================================
            # Bisection 內層迴圈
            # ================================
            for bisection_iter in range(BISECTION_MAX_ITERS):

                # 任一側的界更新後，都在這裡判斷二分搜尋是否停止
                if tau_high - tau_low <= BISECTION_TOL:
                    break

                tau = (tau_low + tau_high) / 2
                constraints = [power_constraint, sensing_constraint]

                # 相位旋轉限制
                for k in range(UAV_COMM):
                    desired = H_eff_H_scaled[k] @ W[:, k]
                    constraints += [cp.imag(desired) == 0, cp.real(desired) >= 0]

                    # 其他波束的干擾振幅上界
                    interference_bounds = []
                    for ell in range(UAV_COMM):
                        if ell == k:
                            continue

                        interference_bound = cp.Variable(nonneg=True)
                        interference = cp.abs(H_eff_H_scaled[k] @ W[:, ell])
                        interference += epsilon_k_scaled[k] * cp.norm(W[:, ell], 2)
                        constraints.append(interference <= interference_bound)
                        interference_bounds.append(interference_bound)

                    # 使用這次的tau算 SINR 限制
                    interference_noise = cp.norm(cp.hstack(interference_bounds + [1.0]), 2)
                    desired_lower      = cp.real(desired) - epsilon_k_scaled[k] * cp.norm(W[:, k], 2)
                    constraints.append(np.sqrt(tau) * interference_noise <= desired_lower)

                # 解最佳化問題，只問是否存在可行的 W
                problem = cp.Problem(cp.Minimize(0), constraints)
                problem.solve(solver=cp.SCS, eps=1e-6, max_iters=15000)

                # 這次沒有確認達標，降低上界，下一輪再試
                if problem.status in (cp.INFEASIBLE, cp.OPTIMAL_INACCURATE, cp.INFEASIBLE_INACCURATE):
                    # if problem.status != cp.INFEASIBLE:
                        # print(f"[子問題一] AO={ao_iter + 1}, SCA={sca_iter + 1}, 二分={bisection_iter + 1}, tau={tau:.9g}, status={problem.status}")
                    tau_high = tau
                    continue

                # 其他非預期狀態直接報錯
                if problem.status != cp.OPTIMAL:
                    raise RuntimeError(f"子問題一狀態異常:tau={tau:.9g}, status={problem.status}")

                # 取出 SCS 求得的 W，回算最低保守 SINR
                W_bisection_ref = W.value.copy()
                amplitude   = np.abs(H_eff_H @ W_bisection_ref)
                beam_norm   = np.linalg.norm(W_bisection_ref, axis=0)
                error_bound = epsilon_k[:, None] * beam_norm[None, :]
                desired_power = np.maximum(np.diag(amplitude - error_bound), 0.0)**2
                interference_power = (amplitude + error_bound)**2
                np.fill_diagonal(interference_power, 0.0)
                tau_bisection_ref = np.min(desired_power / (interference_power.sum(axis=1) + NOISE_POWER))

                tau_low      = tau_bisection_ref

            # 二分搜尋結束，接收本輪保留的解
            sca_gain  = tau_bisection_ref - tau_SCA_ref
            W_sca_ref = W_bisection_ref

            # 本輪最低保守 SINR 的改善量夠小，停止 SCA
            if sca_gain <= SCA_TOL:
                break

        # 把子問題一的結果交給子問題二
        W = W_sca_ref

        # ================================
        # 子問題二：固定 W，更新 RIS 反射係數 phi
        # ================================

        # 以目前的 phi_AO_ref 初始化 PCCP 參考點
        phi_pccp_ref = phi_AO_ref.copy()

        # 固定 W 後，先算好所有波束的直達項與 RIS 項
        G_W = G @ W
        c_dir = h_dk.conj().T @ W
        c_ris = np.empty((UAV_COMM, UAV_COMM, h_rk.shape[0]), dtype=np.complex128)

        for k in range(UAV_COMM):
            for ell in range(UAV_COMM):
                c_ris[k, ell] = h_rk[:, k].conj() * G_W[:, ell]

        # W 與 epsilon_k 固定，整個 PCCP 共用同一組誤差上界
        beam_norm   = np.linalg.norm(W, axis=0)
        error       = epsilon_k[:, None] * beam_norm[None, :]

        # ================================
        # PCCP 迴圈
        # ================================
        for pccp_iter in range(PCCP_MAX_ITERS):

            # 用目前的 PCCP 參考點更新 QT 輔助變數
            received_ref = c_ris @ phi_pccp_ref + c_dir
            amplitude_ref = np.abs(received_ref)
            x_k_ref = np.diag(received_ref)
            a_ref_scaled = np.maximum(np.abs(x_k_ref) - np.diag(error), 0.0) / noise_std

            interference_power_ref = ((amplitude_ref + error) / noise_std)**2
            np.fill_diagonal(interference_power_ref, 0.0)
            D_ref_scaled = interference_power_ref.sum(axis=1) + 1.0
            eta_k = a_ref_scaled / D_ref_scaled

            # 用本次 PCCP 迭代的 phi_pccp_ref 計算參考分數
            Omega_ref  = np.min(a_ref_scaled**2 / D_ref_scaled)
            delta_ref  = np.abs(np.abs(phi_pccp_ref)**2 - 1.0)
            J_PCCP_ref = Omega_ref - mu * np.sum(delta_ref)            

            # 建立本輪待求的 phi、Omega 和 delta
            phi       = cp.Variable(h_rk.shape[0], complex=True, name="phi")
            Omega     = cp.Variable(name="Omega")
            delta     = cp.Variable(h_rk.shape[0], nonneg=True, name="delta")
            constraints = []

            # 每位 UE 的 QT + SCA 下界都要至少達到 Omega
            for k in range(UAV_COMM):
                y_k = phi @ c_ris[k, k] + c_dir[k, k]
                L_k = cp.real(np.exp(-1j * np.angle(x_k_ref[k])) * y_k)
                a_lower_scaled = (L_k - error[k, k]) / noise_std

                D_k_scaled = 1.0
                for ell in range(UAV_COMM):
                    if ell == k:
                        continue

                    interference = cp.abs(phi @ c_ris[k, ell] + c_dir[k, ell])
                    D_k_scaled += cp.square((interference + error[k, ell]) / noise_std)

                sinr_lower = 2 * eta_k[k] * a_lower_scaled - eta_k[k]**2 * D_k_scaled
                constraints.append(sinr_lower >= Omega)

            # PCCP 的 RIS 模量限制
            modulus_lower = 2 * cp.real(cp.multiply(np.conj(phi_pccp_ref), phi)) - np.abs(phi_pccp_ref)**2
            constraints.append(1.0 - delta <= modulus_lower)
            constraints.append(cp.square(cp.abs(phi)) <= 1.0 + delta)

            # 固定 mu，求解本輪 PCCP 問題
            problem = cp.Problem(cp.Maximize(Omega - mu * cp.sum(delta)), constraints)
            problem.solve(solver=cp.SCS, eps=1e-6, max_iters=15000)

            # 計算本輪改善量，先留下本輪求出的 phi
            J_PCCP = float(Omega.value) - mu * np.sum(delta.value)
            PCCP_gain = J_PCCP - J_PCCP_ref
            phi_pccp_ref = phi.value.copy()

            # 本輪改善量夠小，停止 PCCP
            if PCCP_gain <= PCCP_TOL:
                break

        # 子問題二完成
        phi = phi_pccp_ref

        # ================================
        # 比較本輪 AO 更新前後的 J_AO
        # ================================

        # H_eff_H 是子問題一用 phi_ref 算好的舊通道
        H_new_H = h_dk.conj().T + h_rk.conj().T @ np.diag(phi) @ G
        J_AO_values = []

        for H_eval_H, W_eval, phi_eval in ((H_eff_H, W_AO_ref, phi_AO_ref), (H_new_H, W, phi)):
            amplitude = np.abs(H_eval_H @ W_eval)
            beam_norm = np.linalg.norm(W_eval, axis=0)
            error_bound = epsilon_k[:, None] * beam_norm[None, :]

            desired_power = np.maximum(np.diag(amplitude - error_bound), 0.0)**2
            interference_power = (amplitude + error_bound)**2
            np.fill_diagonal(interference_power, 0.0)
            min_sinr = np.min(desired_power / (interference_power.sum(axis=1) + NOISE_POWER))

            ris_penalty = np.sum(np.abs(np.abs(phi_eval)**2 - 1.0))
            J_AO_values.append(min_sinr - mu * ris_penalty)

        J_AO_before, J_AO_after = J_AO_values
        J_AO_gain = J_AO_after - J_AO_before

        # 先留下本輪求出的 W 和 phi
        W_AO_ref   = W
        phi_AO_ref = phi

        # 本輪已用 mu 上限，且改善量夠小，就回傳最新解
        if mu == PCCP_MU_MAX and J_AO_gain <= AO_TOL:
            return phi_AO_ref, W_AO_ref, True, "ao_small_gain"

        # 繼續下一輪時才更新 mu
        mu = min(mu * PCCP_MU_MULTIPLIER, PCCP_MU_MAX)

    return phi_AO_ref, W_AO_ref, True, "ao_max_iters"

# ================================
# Solve one LT block 
# ================================
def solve_two_timescale_block(h_dk,h_rk,G,g_dt):
    """
    求解一個完整 LT block 的 AO 決策。輸入是整個LT block Q slot的通道。

    h_dk : (LT_BLOCK_SIZE,M,K)
    h_rk : (LT_BLOCK_SIZE,N,K)
    G    : (LT_BLOCK_SIZE,N,M)
    g_dt : (LT_BLOCK_SIZE,M,1)

    對第一個槽 使用 solve_ao() 求得本區間共用的 phi 與第一個槽的 W。
    後續各槽沿用首槽的 phi,依當前槽的通道呼叫 solve_SCA() 更新 W。

    回傳各槽的 AO_phi、AO_W、success、status。(會儲存在NPZ)
    """

    # 取出本 LT block first slot 通道，供 solve_ao() 使用 對應到 phase I
    h_dk_first = h_dk[0]    # (M,K)
    h_rk_first = h_rk[0]    # (N,K)
    G_first    = G[0]       # (N,M)
    g_dt_first = g_dt[0]    # (M,1)

    # 預留整個 LT block 的逐槽結果。尚未取得解的槽以 NaN 表示。(長度為Q的陣列)
    AO_phi  = np.full((LT_BLOCK_SIZE,h_rk.shape[1]),np.nan,dtype=np.complex128)                 # (Q,N) 這Q個phi會一樣
    AO_W    = np.full((LT_BLOCK_SIZE,h_dk.shape[1],h_dk.shape[2]),np.nan,dtype=np.complex128)   # (Q,M,K)
    success = np.zeros(LT_BLOCK_SIZE,dtype=bool)                                                # (Q,)
    status  = ["not_run"] * LT_BLOCK_SIZE                                                       # 每槽一個狀態字串

    # Phase I：使用首槽的四條 nominal 通道做 AO，求得本區間共用的 phi 與首槽的 W。
    phi_first,W_first,success_first,status_first = solve_ao(h_dk_first,h_rk_first,G_first,g_dt_first)

    # 記錄首槽求解結果
    success[0]  = success_first
    status[0]   = status_first

    # 整個 LT block 共用同一組 phi
    AO_phi[:]   = phi_first
    AO_W[0]     = W_first


    # Phase II：固定首槽求得的 phi，逐槽求解 W
    for t in range(1,LT_BLOCK_SIZE):

        # 使用第 t 槽的通訊通道與固定的 phi_first，計算等效通道 "注意這是是當下slot的等效通道"
        H_eff_H = h_dk[t].conj().T + h_rk[t].conj().T @ np.diag(phi_first) @ G[t]   # (K,M)

        # H_eff_H 每一列對應一位 UE，沿天線軸取二範數，得到 K 個誤差半徑。
        epsilon_k = EPSILON_K * np.linalg.norm(H_eff_H,axis=1)                      # (K,)

        # g_dt[t] 是 (M,1) 的目標通道，取範數後得到一個感測誤差半徑。
        epsilon_g = EPSILON_G * np.linalg.norm(g_dt[t])                             # scalar

        # 求第 t 槽的 W。固定的 phi_first 已經包含在 H_eff_H 中。
        W_t,success_t,status_t = solve_SCA(H_eff_H,g_dt[t],epsilon_k,epsilon_g)

        # 儲存當槽的波束矩陣與求解狀態
        AO_W[t]    = W_t
        success[t] = success_t
        status[t]  = status_t

    return AO_phi,AO_W,success,status



# ================================
# Main
# ================================
if __name__ == "__main__":

    np.random.seed(RANDOM_SEED)

    # Load test dataset
    # AO 求解使用四條 nominal 通道，beta 留給 Evaluation 做 uncertainty injection。
    with np.load(TEST_DATA_PATH) as data:
        h_dk = data["h_dk"]    # (trajectories, slots, M, K)
        h_rk = data["h_rk"]    # (trajectories, slots, N, K)
        G    = data["G"]       # (trajectories, slots, N, M)
        g_dt = data["g_dt"]    # (trajectories, slots, M, 1)

    # Select AO blocks
    trajectories,slots = h_dk.shape[:2]
    blocks = select_ao_blocks(trajectories,slots)

    print(f"[INFO] Selected AO blocks = {len(blocks)}")

    # ================================
    # 分批求解與存檔
    # ================================

    # 全量執行與小規模測試使用各自的暫存資料夾
    run_name = "all" if MAX_AO_BLOCKS is None else f"first_{MAX_AO_BLOCKS}"
    checkpoint_dir = os.path.join(RESULT_DIR,f"ao6_checkpoints_{run_name}_batch{AO_SAVE_EVERY}")
    os.makedirs(checkpoint_dir,exist_ok=True)

    # 依照 blocks 的順序記錄各批檔案，最後用相同順序合併。
    batch_paths = []

    with tqdm(total=len(blocks),desc="Solve AO blocks") as progress:
        for batch_start in range(0,len(blocks),AO_SAVE_EVERY):
            batch_end = min(batch_start + AO_SAVE_EVERY,len(blocks))
            batch_path = os.path.join(checkpoint_dir,f"blocks_{batch_start}_{batch_end-1}.npz")
            batch_paths.append(batch_path)

            # 只有完整寫好的批次才會取得正式檔名。重新執行時，直接略過這一批。
            if os.path.isfile(batch_path):
                progress.update(batch_end - batch_start)
                continue

            # 收集本批各 LT block 的原始索引與逐槽求解結果。
            saved_trajectory_id = []
            saved_start_slot    = []
            saved_phi           = []
            saved_W             = []
            saved_success       = []
            saved_status        = []

            for trajectory_id,start_slot in blocks[batch_start:batch_end]:
                end_slot = start_slot + LT_BLOCK_SIZE

                AO_phi,AO_W,success,status = solve_two_timescale_block(
                    h_dk[trajectory_id,start_slot:end_slot],
                    h_rk[trajectory_id,start_slot:end_slot],
                    G[trajectory_id,start_slot:end_slot],
                    g_dt[trajectory_id,start_slot:end_slot]
                )

                saved_trajectory_id.append(trajectory_id)
                saved_start_slot.append(start_slot)
                saved_phi.append(AO_phi)
                saved_W.append(AO_W)
                saved_success.append(success)
                saved_status.append(status)
                progress.update(1)

            # 完成整批後才存檔。即使整批首槽都失敗，也保存具有正確形狀的空結果，
            # 表示這一批已經處理完，重新執行時不必再算一次。
            saved_blocks = len(saved_trajectory_id)
            temp_batch_path = batch_path + ".tmp.npz"

            np.savez_compressed(
                temp_batch_path,
                trajectory_id=np.asarray(saved_trajectory_id,dtype=np.int64),
                start_slot=np.asarray(saved_start_slot,dtype=np.int64),
                AO_phi=np.asarray(saved_phi,dtype=np.complex128).reshape(saved_blocks,LT_BLOCK_SIZE,RIS_UNIT),
                AO_W=np.asarray(saved_W,dtype=np.complex128).reshape(saved_blocks,LT_BLOCK_SIZE,TX_ANT,UAV_COMM),
                success=np.asarray(saved_success,dtype=bool).reshape(saved_blocks,LT_BLOCK_SIZE),
                status=np.asarray(saved_status,dtype=str).reshape(saved_blocks,LT_BLOCK_SIZE)
            )

            # 先寫暫存檔，成功後才改成正式檔名。若寫到一半中斷，下次會重算這一批。
            os.replace(temp_batch_path,batch_path)

    # ================================
    # 合併各批結果，輸出 Evaluation 使用的 NPZ
    # ================================
    result_keys = ("trajectory_id","start_slot","AO_phi","AO_W","success","status")
    result_parts = {key: [] for key in result_keys}

    for batch_path in batch_paths:
        with np.load(batch_path,allow_pickle=False) as data:
            for key in result_keys:
                result_parts[key].append(data[key])

    # 目前的 test dataset 有完整 block，因此 batch_paths 至少有一筆。
    results = {key: np.concatenate(result_parts[key],axis=0) for key in result_keys}

    # 合併結果先寫入暫存檔，寫入完成後再取代 AO_SAVE_PATH。
    temp_save_path = AO_SAVE_PATH + ".tmp.npz"
    np.savez_compressed(temp_save_path,**results)
    os.replace(temp_save_path,AO_SAVE_PATH)

    print(f"[INFO] Saved AO blocks = {len(results['trajectory_id'])}/{len(blocks)}")
    print(f"[INFO] AO results saved to {AO_SAVE_PATH}")
