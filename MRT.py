# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

from settings import *


# ============================================================
# Load target sensing channel
# ============================================================

dataset_path = os.path.join(DATA_DIR, "dataset_test.npz")

with np.load(dataset_path) as data:
    g_dt = torch.as_tensor(data["g_dt_hat"],dtype=torch.complex64,device=DEVICE)  # (B,M,1)

B, M, _ = g_dt.shape

total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)

noise = torch.as_tensor(NOISE_POWER,dtype=torch.float32,device=DEVICE)


# ============================================================
# Full-power MRT sensing beamformer
# W_R = sqrt(P_total) * g_dt / ||g_dt||
# ============================================================

g_dt_norm = torch.sqrt(torch.sum(torch.abs(g_dt) ** 2,dim=(1,2),keepdim=True).real).clamp_min(1e-12)

W_R_dir = g_dt / g_dt_norm                    # (B,M,1), unit Frobenius norm

radar_power = 0.6 * total_power
W_R = torch.sqrt(radar_power) * W_R_dir       # (B,M,1), full TX power


# ============================================================
# Check TX power
# ============================================================

W_R_power = torch.sum(torch.abs(W_R) ** 2,dim=(1,2)).real           # (B,)
W_R_power_mean = torch.mean(W_R_power)
W_R_power_max_error = torch.max(torch.abs(W_R_power-total_power))


# ============================================================
# Target sensing SNR
# A = g_dt g_dt^H
# target_power = ||A W_R||_F^2
# ============================================================

g_dt_H = torch.conj(g_dt).transpose(1,2)                            # (B,1,M)
G_sensing = torch.matmul(g_dt,g_dt_H)                               # (B,M,M)

target_echo_R = torch.matmul(G_sensing,W_R)                         # (B,M,1)
target_power = torch.sum(torch.abs(target_echo_R) ** 2,dim=(1,2))  # (B,)

target_snr = target_power / noise                                   # (B,)
target_snr_db = 10.0 * torch.log10(target_snr.clamp_min(1e-12))    # (B,)


# ============================================================
# Closed-form check
# target_power = P_total * ||g_dt||^4
# ============================================================

g_dt_power = torch.sum(torch.abs(g_dt) ** 2,dim=(1,2)).real         # (B,)
target_power_closed = total_power * g_dt_power ** 2                 # (B,)

closed_form_max_error = torch.max(torch.abs(target_power-target_power_closed))


# ============================================================
# Statistics
# ============================================================

target_snr_db_mean = torch.mean(target_snr_db)
target_snr_db_q05 = torch.quantile(target_snr_db,0.05)
target_snr_db_min = torch.min(target_snr_db)
target_snr_db_max = torch.max(target_snr_db)

target_snr_mean_linear = torch.mean(target_snr)
target_snr_mean_linear_db = 10.0 * torch.log10(target_snr_mean_linear.clamp_min(1e-12))


# ============================================================
# Print
# ============================================================

print("=" * 90)
print("[Full-Power MRT Sensing Baseline]")
print("=" * 90)
print(f"Dataset                : {os.path.basename(dataset_path)}")
print(f"Estimated channels B   : {B}")
print(f"TX antennas M          : {M}")
print(f"W_C                    : Not used")
print(f"W_R                    : MRT toward g_dt")
print(f"Total TX power         : {float(total_power.detach().cpu()):.6e}")
print(f"Mean W_R power         : {float(W_R_power_mean.detach().cpu()):.6e}")
print(f"Maximum power error    : {float(W_R_power_max_error.detach().cpu()):.3e}")
print("-" * 90)
print("[Target sensing performance]")
print(f"Target SNR mean of dB  : {float(target_snr_db_mean.detach().cpu()):.3f} dB")
print(f"Target SNR Q0.05       : {float(target_snr_db_q05.detach().cpu()):.3f} dB")
print(f"Target SNR minimum     : {float(target_snr_db_min.detach().cpu()):.3f} dB")
print(f"Target SNR maximum     : {float(target_snr_db_max.detach().cpu()):.3f} dB")
print(f"Target SNR mean linear : {float(target_snr_mean_linear.detach().cpu()):.6e}")
print(f"dB of mean linear SNR  : {float(target_snr_mean_linear_db.detach().cpu()):.3f} dB")
print("-" * 90)
print("[Formula checks]")
print(f"Target power matrix/closed-form max error : {float(closed_form_max_error.detach().cpu()):.3e}")
print("=" * 90)