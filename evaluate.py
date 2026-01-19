# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from settings import *
from neural_net import *
from rician import large_scale_fading


def sum_rate(comm_net, sense_net, ris_net,
             h_dk, h_rk, G, g_dt,
             beta_dk_row, beta_rk_row, beta_G):
    with torch.no_grad():
        W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
        W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)
        phi = ris_net(h_dk, h_rk, G, g_dt)    # (B,N)

        sinrs = comm_net.compute_comm_sinrs(
            h_dk, h_rk, G, phi, W_S, W_C,
            beta_dk_row, beta_rk_row, beta_G
        )  # (B,K)

        rates = comm_net.compute_rates(sinrs)            # (B,K)
        return rates.sum(dim=1).detach().cpu().numpy()   # (B,)


def eval_sense_snr(comm_net, sense_net, ris_net,
                   h_dk, h_rk, G, g_dt,
                   beta_dt):
    """
    回傳 sensing SNR (B,) 線性尺度
    新版：不含 RIS sensing，compute_sense_snr(g_dt, W_S, W_C, beta_dt)
    """
    with torch.no_grad():
        W_C = comm_net(h_dk, h_rk, G, g_dt)   # (B,M,K)
        W_S = sense_net(h_dk, h_rk, G, g_dt)  # (B,M,1)

        sense_snr = comm_net.compute_sense_snr(g_dt, W_S, W_C, beta_dt)  # (B,)
        return sense_snr.detach().cpu().numpy()


if __name__ == "__main__":
    # 1) 載入離線 H_est (npz)
    base_dir = os.path.join("MLP", SCENARIO_TAG, THR_TAG, SETTING_STRING)
    npz_path = os.path.join(base_dir, "channelEstimates_test.npz")
    data = np.load(npz_path)

    h_dk = torch.from_numpy(data["h_dk"]).to(torch.complex64).to(DEVICE)  # (B,M,K)
    h_rk = torch.from_numpy(data["h_rk"]).to(torch.complex64).to(DEVICE)  # (B,N,K)
    G    = torch.from_numpy(data["G"]).to(torch.complex64).to(DEVICE)     # (B,N,M)
    g_dt = torch.from_numpy(data["g_dt"]).to(torch.complex64).to(DEVICE)  # (B,M,1)

    print("[EVAL] shapes:",
          "h_dk", tuple(h_dk.shape),
          "h_rk", tuple(h_rk.shape),
          "G", tuple(G.shape),
          "g_dt", tuple(g_dt.shape))

    # 2) 載入 large-scale (power) fading（只在 SINR/SNR 時計入）
    beta_G, beta_dt, beta_dk_row, beta_rk_row = large_scale_fading()

    # 3) 載入模型 checkpoint
    ckpt_dir = os.path.join(base_dir, "ckpt")
    comm_ckpt  = os.path.join(ckpt_dir, f"comm_{SETTING_STRING}.ckpt")
    sense_ckpt = os.path.join(ckpt_dir, f"sense_{SETTING_STRING}.ckpt")
    ris_ckpt   = os.path.join(ckpt_dir, f"ris_{SETTING_STRING}.ckpt")

    reg_Wc  = CommBeamformerNet().to(DEVICE)
    reg_Ws  = SenseBeamformerNet().to(DEVICE)
    reg_phi = RISPhaseNet().to(DEVICE)

    reg_Wc.load_state_dict(torch.load(comm_ckpt,  map_location=DEVICE))
    reg_Ws.load_state_dict(torch.load(sense_ckpt, map_location=DEVICE))
    reg_phi.load_state_dict(torch.load(ris_ckpt,  map_location=DEVICE))

    reg_Wc.eval(); reg_Ws.eval(); reg_phi.eval()

    print(f"[EVAL] SETTING_STRING = {SETTING_STRING}")
    print(f"[EVAL] ckpt_dir       = {ckpt_dir}")

    # 4) 計算 sum-rate
    print("[EVAL] Computing sum rates ...")
    reg_sum_rates = sum_rate(
        reg_Wc, reg_Ws, reg_phi,
        h_dk, h_rk, G, g_dt,
        beta_dk_row, beta_rk_row, beta_G
    )
    reg_mean = float(np.mean(reg_sum_rates))
    print("[EVAL] mean sum-rate (bps/Hz):", reg_mean)

    # 5) 計算 sensing SNR
    print("[EVAL] Computing sensing SNR ...")
    reg_sense_snr = eval_sense_snr(
        reg_Wc, reg_Ws, reg_phi,
        h_dk, h_rk, G, g_dt,
        beta_dt
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
