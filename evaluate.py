# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

from settings import *
from two_timescale_NN import RISNet, JointNet, compute_effective_channel, compute_isac_batch_performance
from joint_rob import inject_channel

# ================================
# 評估參數
# ================================
RUN_AO          = True    # False：評估 REG、ROB；True：加入 AO4 的離線結果一起比較
MAX_EVAL_BLOCKS = 100      # None：所有軌跡的全部完整區間；整數 N：第 0 條軌跡的前 N 個區間

METHODS         = ("REG","ROB","AO") if RUN_AO else ("REG","ROB")


def select_test_blocks(trajectories,slots):
    if MAX_EVAL_BLOCKS is None:
        return [(trajectory_id,start_slot) for trajectory_id in range(trajectories) for start_slot in range(0,slots-LT_BLOCK_SIZE+1,LT_BLOCK_SIZE)]

    return [(0,block_id*LT_BLOCK_SIZE) for block_id in range(MAX_EVAL_BLOCKS)]


def robust_block_result(h_dk_inj,h_rk_inj,G_inj,g_dt_inj,phi,W):
    # 四條通道的前兩軸都是 (S,Q)。
    # S 是注入樣本數；Q 是這個 LT block 的時槽數。
    # 共用效能函數接受單一 batch 軸，因此先將 (S,Q) 展平成 (S*Q,)。
    h_dk_flat = h_dk_inj.flatten(0,1)                    # (S*Q,M,K)
    h_rk_flat = h_rk_inj.flatten(0,1)                    # (S*Q,N,K)
    G_flat    = G_inj.flatten(0,1)                       # (S*Q,N,M)
    g_dt_flat = g_dt_inj.flatten(0,1)                    # (S*Q,M,1)

    # 此方法在這個 block 只有一組 phi，所有樣本及時槽共用。
    # W 保留逐槽差異：同一個 slot 的 S 組注入樣本都使用該槽的 nominal W。
    phi_flat = phi.reshape(1,RIS_UNIT).expand(h_dk_flat.shape[0],RIS_UNIT)
    W_flat = W.unsqueeze(0).expand(INJECTION_SAMPLES,-1,-1,-1).flatten(0,1)

    H_eff_H_flat = compute_effective_channel(h_dk_flat,h_rk_flat,G_flat,phi_flat)
    metrics = compute_isac_batch_performance(H_eff_H_flat,g_dt_flat,W_flat)

    # 還原注入樣本軸與時槽軸，才能對每個 slot 分別計算分位數。
    rate        = metrics["rate"].reshape(INJECTION_SAMPLES,LT_BLOCK_SIZE,UAV_COMM)
    target_snr = metrics["target_snr"].reshape(INJECTION_SAMPLES,LT_BLOCK_SIZE)
    target_snr_db = metrics["target_snr_db"].reshape(INJECTION_SAMPLES,LT_BLOCK_SIZE)

    # 通訊：每個 (s,t) 先取最差 UE，再對同一個 slot 的 S 筆結果取分位數。
    worst_ue_rate = rate.min(dim=2).values                                    # (S,Q)
    rate_q = torch.quantile(worst_ue_rate,OUTAGE_QUANTILE,dim=0)              # (Q,)

    # 感測：每個 slot 先在線性 SNR 上做分位數插值，再轉成 dB。
    snr_q = torch.quantile(target_snr,TAR_OUTAGE_QUANTILE,dim=0,interpolation="linear")
    snr_q_db = 10.0*torch.log10(snr_q.clamp_min(1e-12))                       # (Q,)

    # 每個 slot 另外保留兩個指標：S 組樣本的平均線性 SNR，以及低於門檻的比例。
    snr_mean = target_snr.mean(dim=0)                                        # (Q,)
    violation = (target_snr_db < SENSING_SNR_THRESHOLD_DB).float().mean(dim=0)

    # 回傳本 block 的逐槽加總。最後統一除以接受的總槽數。
    return {
        "rate": rate_q.sum(dtype=torch.float64).item(),
        "snr": snr_mean.sum(dtype=torch.float64).item(),
        "snr_q_db": snr_q_db.sum(dtype=torch.float64).item(),
        "violation": violation.sum(dtype=torch.float64).item(),
    }


# ================================
# Main
# ================================
if __name__ == "__main__":

    # Load test dataset
    dataset_keys = ("h_dk","h_rk","G","g_dt","beta_d","beta_r","beta_G","beta_g")
    with np.load(TEST_DATA_PATH) as data:
        dataset = {key: data[key] for key in dataset_keys}

     # h_dk 的四個軸依序為：軌跡、時槽、BS 發射天線、通訊 UE。
    trajectories,slots,M,K = dataset["h_dk"].shape 

    # 載入reg模型
    reg_ris_net = RISNet().to(DEVICE)
    reg_joint_net = JointNet().to(DEVICE)

    reg_ris_net.load_model(REG_THETA_CKPT,strict=True,verbose=True)
    reg_joint_net.load_model(REG_JOINT_CKPT,strict=True,verbose=True)

    reg_ris_net.eval()
    reg_joint_net.eval()

    # 載入rob模型
    rob_ris_net = RISNet().to(DEVICE)
    rob_joint_net = JointNet().to(DEVICE)

    rob_ris_net.load_model(ROB_THETA_CKPT,strict=True,verbose=True)
    rob_joint_net.load_model(ROB_JOINT_CKPT,strict=True,verbose=True)

    rob_ris_net.eval()
    rob_joint_net.eval()

    # Select test blocks
    blocks = select_test_blocks(trajectories,slots)

    print(f"[INFO] Selected LT blocks = {len(blocks)}")

    # ================================
    # Load AO5 results
    # ================================
    if RUN_AO:
        from AO import AO_SAVE_PATH

        # 沿用 AO5.py 的輸出路徑。整份結果只讀一次，各 block 依原始 dataset 座標取用。
        with np.load(AO_SAVE_PATH,allow_pickle=False) as data:
            ao_trajectory_id = data["trajectory_id"]    # (saved_blocks,)
            ao_start_slot    = data["start_slot"]       # (saved_blocks,)
            ao_phi_saved     = data["AO_phi"]           # (saved_blocks,Q,N)
            ao_W_saved       = data["AO_W"]             # (saved_blocks,Q,M,K)
            ao_success_saved = data["success"]          # (saved_blocks,Q)，每槽是否有可用解

        # AO5 只保存首槽成功的 block，所以用 (trajectory_id,start_slot) 找到對應的結果列。
        ao_block_index = {}
        for ao_row in range(len(ao_trajectory_id)):
            ao_block_index[(int(ao_trajectory_id[ao_row]),int(ao_start_slot[ao_row]))] = ao_row

        print(f"[INFO] Loaded AO5 blocks = {len(ao_block_index)}, path = {AO_SAVE_PATH}")

    # ================================
    # Evaluation setup
    # ================================
    # inject_channel() 使用 PyTorch 的預設亂數產生器。
    # seed 在區間迴圈外設定一次，各 block 會依序抽到不同的誤差。
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # 每個方法各自累加 nominal / robust 指標。
    # nominal 的 rate 是逐槽最差 UE rate；robust 的 rate 是逐槽最差 UE rate 的分位數。
    nominal_total = {name: {"rate": 0.0, "snr": 0.0, "violation": 0.0} for name in METHODS}
    robust_total = {name: {"rate": 0.0, "snr": 0.0, "snr_q_db": 0.0, "violation": 0.0} for name in METHODS}
    accepted_blocks = 0

    # 固定鏈路在整份 test dataset 中各有一個純量 beta。
    beta_G = torch.as_tensor(dataset["beta_G"],dtype=torch.float32,device=DEVICE)
    beta_g = torch.as_tensor(dataset["beta_g"],dtype=torch.float32,device=DEVICE)

    for trajectory_id,start_slot in blocks:

        # ================================
        # 取出目前 block 的 nominal 通道
        # ================================
        # slice 的右端不包含在內。例如 start_slot=10、LT_BLOCK_SIZE=10 時，取第 10～19 槽。
        slot_ids = slice(start_slot,start_slot+LT_BLOCK_SIZE)

        # REG、ROB、AO4 都使用同一個 block 的 nominal 通道計算效能。
        h_dk = torch.as_tensor(dataset["h_dk"][trajectory_id,slot_ids],dtype=torch.complex64,device=DEVICE)    # (Q,M,K)
        h_rk = torch.as_tensor(dataset["h_rk"][trajectory_id,slot_ids],dtype=torch.complex64,device=DEVICE)    # (Q,N,K)
        G    = torch.as_tensor(dataset["G"][trajectory_id,slot_ids],dtype=torch.complex64,device=DEVICE)       # (Q,N,M)
        g_dt = torch.as_tensor(dataset["g_dt"][trajectory_id,slot_ids],dtype=torch.complex64,device=DEVICE)    # (Q,M,1)

        # ================================
        # Common uncertainty samples
        # ================================
        # UE 的 beta 依軌跡、slot 及 UE 取值。
        # (Q,K) -> (1,Q,1,K)，沿注入樣本軸與天線軸廣播。
        beta_d = torch.as_tensor(dataset["beta_d"][trajectory_id,slot_ids],dtype=torch.float32,device=DEVICE).reshape(1,LT_BLOCK_SIZE,1,UAV_COMM)
        beta_r = torch.as_tensor(dataset["beta_r"][trajectory_id,slot_ids],dtype=torch.float32,device=DEVICE).reshape(1,LT_BLOCK_SIZE,1,UAV_COMM)

        h_dk_rep = h_dk.unsqueeze(0).expand(INJECTION_SAMPLES,-1,-1,-1)       # (S,Q,M,K)
        h_rk_rep = h_rk.unsqueeze(0).expand(INJECTION_SAMPLES,-1,-1,-1)       # (S,Q,N,K)
        G_rep    = G.unsqueeze(0).expand(INJECTION_SAMPLES,-1,-1,-1)          # (S,Q,N,M)
        g_dt_rep = g_dt.unsqueeze(0).expand(INJECTION_SAMPLES,-1,-1,-1)       # (S,Q,M,1)

        # 依 joint_rob.py 的順序注入。每個 block 只抽這一次，三種方法共用以下四個結果。
        h_dk_inj = inject_channel(h_dk_rep,beta_d)
        h_rk_inj = inject_channel(h_rk_rep,beta_r)
        G_inj    = inject_channel(G_rep,beta_G)
        g_dt_inj = inject_channel(g_dt_rep,beta_g)

        # 開啟 AO 比較時，三種方法只在 AO4 有完整 Q 槽決策的 block 上計分。
        # 此處放在共同注入之後，即使略過本 block，後續 block 的注入亂數順序仍一致。
        if RUN_AO:
            ao_row = ao_block_index.get((trajectory_id,start_slot))

            # AO4 首槽失敗時會丟棄整個 block。若讀到較小規模的結果檔，也可能查不到。
            if ao_row is None:
                print(f"[AO4 缺少 block] trajectory_id={trajectory_id}, start_slot={start_slot}")
                continue

            # AO4 首槽成功時仍可能有後續槽失敗。沿用目前以完整 block 為單位的計分方式。
            if not ao_success_saved[ao_row].all():
                failed_slots = start_slot + np.flatnonzero(~ao_success_saved[ao_row])
                print(f"[AO4 有失敗槽] trajectory_id={trajectory_id}, start_slot={start_slot}, failed_slots={failed_slots.tolist()}")
                continue

        # 暫存本 block 各方法的 nominal 與 robust 指標。
        nominal_block = {}
        robust_block = {}

        # ================================
        # REG test
        # ================================
        with torch.inference_mode():
            # REG nominal：用首槽 CSI 產生一組 REG_phi，固定套用至整個 block。
            # 每一槽再用自己的 nominal 有效通道及目標通道產生 REG_W。
            REG_phi = reg_ris_net(h_dk[0:1],h_rk[0:1],G[0:1])                  # (1,N)
            REG_phi_block = REG_phi.expand(LT_BLOCK_SIZE,RIS_UNIT)                # (Q,N)
            REG_H_eff_H = compute_effective_channel(h_dk,h_rk,G,REG_phi_block)   # (Q,K,M)
            REG_W = reg_joint_net(REG_H_eff_H,g_dt)                              # (Q,M,K)

            nominal_block["REG"] = compute_isac_batch_performance(REG_H_eff_H,g_dt,REG_W)

            # REG robust：沿用剛才的 REG_phi、REG_W，在共同注入通道上評估。
            robust_block["REG"] = robust_block_result(h_dk_inj,h_rk_inj,G_inj,g_dt_inj,REG_phi,REG_W)

        # ================================
        # ROB test
        # ================================
        with torch.inference_mode():
            # ROB nominal：使用 ROB checkpoint 的兩個網路，在相同 nominal CSI 下產生決策。
            ROB_phi = rob_ris_net(h_dk[0:1],h_rk[0:1],G[0:1])                  # (1,N)
            ROB_phi_block = ROB_phi.expand(LT_BLOCK_SIZE,RIS_UNIT)                # (Q,N)
            ROB_H_eff_H = compute_effective_channel(h_dk,h_rk,G,ROB_phi_block)   # (Q,K,M)
            ROB_W = rob_joint_net(ROB_H_eff_H,g_dt)                              # (Q,M,K)

            nominal_block["ROB"] = compute_isac_batch_performance(ROB_H_eff_H,g_dt,ROB_W)

            # ROB robust：使用 ROB 自己的決策，配上和 REG 完全相同的注入通道。
            robust_block["ROB"] = robust_block_result(h_dk_inj,h_rk_inj,G_inj,g_dt_inj,ROB_phi,ROB_W)

        # ================================
        # AO test
        # ================================
        if RUN_AO:
            with torch.inference_mode():
                # AO4 儲存的 AO_phi 每個 block 為 (Q,N)，Q 個槽共用首槽求得的 phi。
                # 取第 0 槽得到 (1,N)，傳給 robust_block_result()；展開成 (Q,N) 供 nominal 使用。
                AO_phi = torch.as_tensor(ao_phi_saved[ao_row,0:1],dtype=torch.complex64,device=DEVICE)
                AO_phi_block = AO_phi.expand(LT_BLOCK_SIZE,RIS_UNIT)

                # AO4 的 W 是逐槽求解，因此取出本 block 全部 Q 個槽的 W，形狀為 (Q,M,K)。
                AO_W = torch.as_tensor(ao_W_saved[ao_row],dtype=torch.complex64,device=DEVICE)

                # AO nominal：使用當槽的 nominal 通道與離線求得的 AO 決策。
                AO_H_eff_H = compute_effective_channel(h_dk,h_rk,G,AO_phi_block)
                nominal_block["AO"] = compute_isac_batch_performance(AO_H_eff_H,g_dt,AO_W)

                # AO robust：固定相同的 AO_phi、AO_W，使用和 REG、ROB 完全相同的注入通道。
                robust_block["AO"] = robust_block_result(h_dk_inj,h_rk_inj,G_inj,g_dt_inj,AO_phi,AO_W)

        # ================================
        # 接受本 block，累加各方法結果
        # ================================
        # RUN_AO=True 時，能走到這裡代表 AO 已成功。
        # RUN_AO=False 時，METHODS 只包含 REG、ROB。
        for name in METHODS:
            metrics = nominal_block[name]

            # rate 的 shape 為 (Q,K)：沿 UE 軸取最小值，再加總 Q 個時槽。
            nominal_total[name]["rate"] += metrics["rate"].min(dim=1).values.sum(dtype=torch.float64).item()

            # SNR 以線性值累加；違規數量按每個 nominal slot 是否低於門檻來計算。
            nominal_total[name]["snr"] += metrics["target_snr"].sum(dtype=torch.float64).item()
            nominal_total[name]["violation"] += (metrics["target_snr_db"] < SENSING_SNR_THRESHOLD_DB).sum().item()

            # robust_block_result() 已將本 block 的各項逐槽結果加總。
            for key,value in robust_block[name].items():
                robust_total[name][key] += value

        accepted_blocks += 1

    # ================================
    # Final results
    # ================================
    accepted_slots = accepted_blocks*LT_BLOCK_SIZE

    print("\n" + "="*70)
    print("[EVALUATION RESULTS]")
    print(f"Accepted LT blocks = {accepted_blocks}/{len(blocks)}")
    print(f"Accepted slots     = {accepted_slots}")
    print(f"Sensing threshold  = {SENSING_SNR_THRESHOLD_DB:g} dB")

    if accepted_slots == 0:
        raise RuntimeError("沒有可供比較的成功區間，無法計算平均結果")

    # METHODS 的順序為 REG、ROB、AO。每個方法先印 nominal，再印 robust。
    for name in METHODS:
        nominal = nominal_total[name]
        robust = robust_total[name]

        # 兩個 Mean target SNR 都是先平均線性 SNR，再將平均值轉為 dB。
        nominal_snr_db = 10.0*np.log10(max(nominal["snr"]/accepted_slots,1e-12))
        robust_snr_db = 10.0*np.log10(max(robust["snr"]/accepted_slots,1e-12))

        print("\n" + "-"*70)
        print(f"[{name} nominal]")
        print(f"Mean worst-UE rate        = {nominal['rate']/accepted_slots:.6f} bps/Hz")
        print(f"Mean target SNR           = {nominal_snr_db:.3f} dB")
        print(f"Sensing violation         = {100.0*nominal['violation']/accepted_slots:.3f}%")

        print(f"\n[{name} robust]")
        print(f"Mean per-slot worst-UE rate ({OUTAGE_QUANTILE:.0%} quantile) = {robust['rate']/accepted_slots:.6f} bps/Hz")
        print(f"Mean target SNR           = {robust_snr_db:.3f} dB")
        print(f"Mean per-slot sensing SNR ({TAR_OUTAGE_QUANTILE:.0%} quantile, dB) = {robust['snr_q_db']/accepted_slots:.3f} dB")
        print(f"Sensing violation         = {100.0*robust['violation']/accepted_slots:.3f}%")