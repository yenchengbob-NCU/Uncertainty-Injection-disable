# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

from settings import *
from neural_net import *


def sum_rate(comm_net, sense_net, ris_net,
             h_dk, h_rk, G, g_dt):
    """
    Sum-rate (B,) using channels that already include large-scale fading.
    No beta is applied here.
    """
    with torch.no_grad():
        W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
        W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)
        phi = ris_net(h_dk, h_rk, G, g_dt)    # (B,N)

        sinrs = comm_net.compute_comm_sinrs(
            h_dk, h_rk, G, phi, W_S, W_C
        )  # (B,K)
        
        rates = comm_net.compute_rates(sinrs)            # (B,K)
        return rates.sum(dim=1).detach().cpu().numpy()   # (B,)


def eval_sense_snr(comm_net, sense_net, ris_net,
                   h_dk, h_rk, G, g_dt):
    """
    Return sensing SNR (B,) in linear scale.
    Assumption: g_dt already includes large-scale fading (your latest convention).
    """
    with torch.no_grad():
        W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
        W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)

        sense_snr = comm_net.compute_sense_snr(g_dt, W_S, W_C)  # (B,)

        if sense_snr is None:
            raise RuntimeError(
                "compute_sense_snr() 回傳 None。請確認 neural_net.py 內該函式已完成並 return SNR。"
            )

        return sense_snr.detach().cpu().numpy()


if __name__ == "__main__":
    # ===============================
    # 1) 載入離線 H_est (npz)
    # ===============================
    base_dir = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
    npz_path = os.path.join(base_dir, "channelEstimates_test.npz")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"[EVAL] 找不到測試通道檔：{npz_path}")

    data = np.load(npz_path)

    # 必要 key 檢查
    required = ["h_dk", "h_rk", "G"]
    for k in required:
        if k not in data.files:
            raise KeyError(f"[EVAL] npz 缺少 key: {k}，目前 keys={data.files}")

    h_dk = torch.from_numpy(data["h_dk"]).to(torch.complex64).to(DEVICE)  # (B,M,K)
    h_rk = torch.from_numpy(data["h_rk"]).to(torch.complex64).to(DEVICE)  # (B,N,K)
    G    = torch.from_numpy(data["G"]).to(torch.complex64).to(DEVICE)     # (B,N,M)
    g_dt = torch.from_numpy(data["g_dt"]).to(torch.complex64).to(DEVICE)  # (B,M,1)

    print("[EVAL] shapes:",
          "h_dk", tuple(h_dk.shape),
          "h_rk", tuple(h_rk.shape),
          "G", tuple(G.shape),
          "g_dt", tuple(g_dt.shape))

    # ===============================
    # 2) 載入模型 checkpoint
    # ===============================
    ckpt_dir = os.path.join(base_dir, "ckpt")
    comm_ckpt  = os.path.join(ckpt_dir, f"comm_{SETTING_STRING}.ckpt")
    sense_ckpt = os.path.join(ckpt_dir, f"sense_{SETTING_STRING}.ckpt")
    ris_ckpt   = os.path.join(ckpt_dir, f"ris_{SETTING_STRING}.ckpt")

    for p in [comm_ckpt, sense_ckpt, ris_ckpt]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[EVAL] 找不到 checkpoint：{p}")

    reg_Wc  = CommBeamformerNet().to(DEVICE)
    reg_Ws  = SenseBeamformerNet().to(DEVICE)
    reg_phi = RISPhaseNet().to(DEVICE)

    reg_Wc.load_state_dict(torch.load(comm_ckpt,  map_location=DEVICE))
    reg_Ws.load_state_dict(torch.load(sense_ckpt, map_location=DEVICE))
    reg_phi.load_state_dict(torch.load(ris_ckpt,  map_location=DEVICE))

    reg_Wc.eval(); reg_Ws.eval(); reg_phi.eval()

    print(f"[EVAL] SETTING_STRING = {SETTING_STRING}")
    print(f"[EVAL] ckpt_dir       = {ckpt_dir}")

    # ===============================
    # 3) 計算 sum-rate
    # ===============================
    print("[EVAL] Computing sum rates ...")
    reg_sum_rates = sum_rate(
        reg_Wc, reg_Ws, reg_phi,
        h_dk, h_rk, G, g_dt
    )
    reg_mean = float(np.mean(reg_sum_rates))
    print("[EVAL] mean sum-rate (bps/Hz):", reg_mean)

    # ===============================
    # 4) 計算 sensing SNR
    # ===============================
    print("[EVAL] Computing sensing SNR ...")
    reg_sense_snr = eval_sense_snr(
        reg_Wc, reg_Ws, reg_phi,
        h_dk, h_rk, G, g_dt
    )

    sense_mean_lin = float(np.mean(reg_sense_snr))
    sense_mean_db = 10.0 * np.log10(max(sense_mean_lin, 1e-12))
    print("[EVAL] mean sensing SNR (dB):", sense_mean_db)

    viol_mask = reg_sense_snr < SENSING_SNR_THRESHOLD
    num_viol = int(viol_mask.sum())
    total = int(reg_sense_snr.size)
    viol_ratio = 100.0 * num_viol / max(total, 1)

    print(f"[EVAL] sensing SNR < threshold ({SENSING_SNR_THRESHOLD_dB} dB): "
          f"{viol_ratio:.2f}%  ({num_viol}/{total})")
