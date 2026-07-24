# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

# ================================
# helpers
# ================================

def normalize_complex_block(x, eps=1e-12):
    """
    將一整個 complex feature block 做 batch-level RMS normalization。
    目的：避免不同 feature block 數值尺度差太多。

    注意：
        這不是 per-sample normalize。
        它用目前 batch 的整體 RMS,保留 block 內 user/link 相對差異。
    """

    rms = torch.sqrt(torch.mean(torch.abs(x) ** 2).real + eps)
    x = x / rms

    return x


def pathloss_db_norm(pl, batch_size, device, out_dim):
    """
    將 pathloss linear scale 轉成 dB 後縮放到約 O(1)。

    pl:
        scalar or (K,)

    return:
        shape = (B, out_dim)
    """

    pl = torch.as_tensor(
        pl,
        dtype=torch.float32,
        device=device,
    ).reshape(1, out_dim)

    pl_db = 10.0 * torch.log10(pl.clamp_min(1e-30))

    # 讓 -80 dB -> -0.8, -110 dB -> -1.1
    pl_db_norm = pl_db / 100.0

    pl_db_norm = pl_db_norm.expand(batch_size, out_dim)

    return pl_db_norm

NET = 64
Debug = False                      # 終端印出檢查
# ================================
# neural net
# ================================

class ISACNetBase(nn.Module):
    """
        放一些基礎函式
    """
    def __init__(self, ckpt_kind: str ):
        super().__init__()
        self.ckpt_kind  = ckpt_kind
    
    @property
    def model_device(self):
        return next(self.parameters()).device

    def save_model(self, path, verbose=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

        if verbose:
            print(f"[save_model] 已儲存：{path}")

    def load_model(self, path, strict=True, verbose=True):
        if not os.path.exists(path):
            if verbose:
                print(f"[load_model] 找不到 checkpoint:{path}")
            return

        state = torch.load(path, map_location=self.model_device)
        self.load_state_dict(state, strict=strict)

        if verbose:
            print(f"[load_model] 已載入：{path}")

    def load_channel_dataset(self, npz_path: str, split_name: str):
        """
        載入 one-timescale 固定 layout dataset
        注意：
            h_dk_hat, h_rk_hat, G_hat, g_dt_hat
            已經是 channel_gen.py 產生的「帶 PL 估測通道」
            後續 compute_effective_channel 不應再重複乘 path loss
        """
        with np.load(npz_path) as data:
            dataset = {
                # path loss, debug / 紀錄用
                "pl_BS_UE": data["pl_BS_UE"],
                "pl_RIS_UE": data["pl_RIS_UE"],
                "pl_BS_RIS": data["pl_BS_RIS"],
                "pl_BS_RIS_UE": data["pl_BS_RIS_UE"],
                "pl_BS_TAR": data["pl_BS_TAR"],
                "pl_BS_TAR_BS": data["pl_BS_TAR_BS"],

                # 帶 PL 的估測通道，NN input / baseline input
                "h_dk_hat": data["h_dk_hat"],      # (C, M, K)
                "h_rk_hat": data["h_rk_hat"],      # (C, N, K)
                "G_hat": data["G_hat"],            # (C, N, M)
                "g_dt_hat": data["g_dt_hat"],      # (C, M, 1)
            }
        if Debug:
            total_bytes = sum(v.nbytes for v in dataset.values() if hasattr(v, "nbytes"))
            print(f"[{split_name}] loaded from {npz_path}")
            print(f"[{split_name}] RAM usage ≈ {total_bytes / (1024 ** 2):.2f} MiB")
            print(f"[{split_name}] h_dk_hat shape = {dataset['h_dk_hat'].shape}")
            print(f"[{split_name}] h_rk_hat shape = {dataset['h_rk_hat'].shape}")
            print(f"[{split_name}] G_hat    shape = {dataset['G_hat'].shape}")
            print(f"[{split_name}] g_dt_hat shape = {dataset['g_dt_hat'].shape}")

        return dataset

    def expand_theta_batch(self, theta, batch_size):
        """
        將 theta 統一轉成 (B, RIS_UNIT),因為LT輸出只有(RIS_UNIT,)
        但後續計算要用(B, RIS_UNIT)
        Accept:
            theta: (RIS_UNIT,)
            theta: (1, RIS_UNIT)
            theta: (B, RIS_UNIT)
        Return:
            theta: (B, RIS_UNIT)
        """
        theta = torch.as_tensor(theta,dtype=torch.complex64,device=self.model_device)

        if theta.ndim == 1:
            theta = theta.reshape(1, RIS_UNIT)

        if theta.shape[0] == 1 and batch_size > 1:
            theta = theta.expand(batch_size, RIS_UNIT)

        return theta.reshape(batch_size, RIS_UNIT)

    # 注意以下函式需要吃batch 

    def normalize_isac_beamformers(self, W_C, W_R, g_dt_hat):
        """
        Sensing-first power allocation
        先讓 W_R 滿足 sensing SNR threshold
        剩餘功率才給 W_C
        """
        total_power = torch.as_tensor(TRANSMIT_POWER_TOTAL,dtype=torch.float32,device=DEVICE)
        noise       = torch.as_tensor(NOISE_POWER,dtype=torch.float32,device=DEVICE)

        # 1) 先把 W_C, W_R 都變成 direction，消除 raw power 不均問題
        W_C_dir = W_C / (torch.sqrt(torch.sum(torch.abs(W_C) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12)
        W_R_dir = W_R / (torch.sqrt(torch.sum(torch.abs(W_R) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12)

        # 2) 算 W_R_dir 的 sensing gain
        gH = torch.conj(g_dt_hat).transpose(1, 2)       # (B,1,M)
        G_sensing = torch.matmul(g_dt_hat, gH)          # (B,M,M)

        target_echo_R_dir = torch.matmul(G_sensing, W_R_dir)     # (B,M,R)
        target_gain_R = torch.sum(
            torch.abs(target_echo_R_dir) ** 2,
            dim=(1, 2),
            keepdim=True,
        ).real                                                   # (B,1,1)

        # 3) 為了達到 sensing threshold，需要多少 W_R power
        p_R_required = (SENSING_SNR_THRESHOLD * noise/ (target_gain_R + 1e-12))  # (B,1,1)

        # 4) sensing-first:
        #    若需要功率超過總功率，全部給 W_R
        #    若不用那麼多，剩下給 W_C
        p_R = torch.minimum(
            p_R_required,
            total_power.reshape(1, 1, 1),
        )

        p_R = torch.clamp(p_R, min=0.0)
        p_C = total_power.reshape(1, 1, 1) - p_R

        # 5) 套用功率
        W_R = torch.sqrt(p_R) * W_R_dir
        W_C = torch.sqrt(p_C) * W_C_dir

        return W_C, W_R

    def compute_effective_channel(self, h_dk, h_rk, G, theta):
        """
        根據已帶 PL 的估測通道組裝 RIS-assisted effective channel
        Input:
            h_dk  : (B, M, K)    , complex
            h_rk  : (B, N, K)    , complex
            G     : (B, N, M)    , complex
            theta : (B, RIS_UNIT), complex

            RIS reflection coefficients, |theta_n| = 1
        Return:
            H_eff_H : (B, K, M), complex

        Effective channel convention:
            H_eff_H[b, k, m]
            = h_dk[b, m, k]^*
              + sum_n h_rk[b, n, k]^* theta[b, n] G[b, n, m]
        """
        h_dk = torch.as_tensor(h_dk,dtype=torch.complex64,device=self.model_device)
        h_rk = torch.as_tensor(h_rk,dtype=torch.complex64,device=self.model_device)
        G    = torch.as_tensor(G,dtype=torch.complex64,device=self.model_device)

        B = h_dk.shape[0]
        theta = self.expand_theta_batch(theta, B)       # (B, N)

        # Direct link: h_d,k^H
        H_direct_H = torch.conj(h_dk).transpose(1, 2)   # (B, K, M)

        # RIS link: h_r,k^H diag(theta) G
        h_rk_H = torch.conj(h_rk).transpose(1, 2)       # (B, K, N)

        H_ris_H = torch.einsum(
            "bkn,bn,bnm->bkm",
            h_rk_H,
            theta,
            G,
        )                                               # (B, K, M)

        H_eff_H = H_direct_H + H_ris_H

        return H_eff_H
    
    def compute_isac_batch_performance(self, H_eff_H, g_dt, W_C, W_R):
        """
        Compute communication and sensing performance.
        Input:
            H_eff_H : (B, K, M), complex
            g_dt    : (B, M, 1), complex
            W_C     : (B, M, K), complex
            W_R     : (B, M, RADAR_STREAMS), complex

        Output:
            SINR power components:
                signal       : (B, K)
                comm_interf  : (B, K)
                radar_interf : (B, K)
                noise        : scalar

            raw per-channel-sample tensors:
                sinr          : (B, K), linear
                sinr_db       : (B, K), dB
                rate          : (B, K)
                sumrate       : (B,)
                target_snr    : (B,), linear
                target_snr_db : (B,), dB

            B-average display values: 對所有channel 平均
                sinr_user_mean      : (K,), linear
                sinr_user_mean_db   : (K,), dB = 10log10(mean linear SINR)
                rate_user_mean      : (K,)
                sumrate_mean        : scalar
                target_snr_mean     : scalar, linear
                target_snr_mean_db  : scalar, dB = 10log10(mean linear target SNR)
        """

        # 確定資料型態
        H_eff_H = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=self.model_device)
        g_dt    = torch.as_tensor(g_dt,dtype=torch.complex64,device=self.model_device)
        W_C     = torch.as_tensor(W_C,dtype=torch.complex64,device=self.model_device)
        W_R     = torch.as_tensor(W_R,dtype=torch.complex64,device=self.model_device)
        noise   = torch.as_tensor(NOISE_POWER,dtype=torch.float32,device=self.model_device)

        # Communication SINR / Rate
        Y_C = torch.matmul(H_eff_H, W_C)                # (B,K,K)
        P_C = torch.abs(Y_C) ** 2                       # (B,K,K)

        signal = torch.diagonal(P_C, dim1=1, dim2=2)    # (B,K)
        comm_interf = torch.sum(P_C, dim=2) - signal    # (B,K)

        Y_R = torch.matmul(H_eff_H, W_R)                            # (B,K,R)
        radar_interf = torch.sum(torch.abs(Y_R) ** 2, dim=2)

        sinr = signal / (comm_interf + radar_interf + noise)        # (B,K)
        sinr_db = 10.0 * torch.log10(sinr.clamp_min(1e-12))         # (B,K)

        sumsinr = torch.sum(sinr, dim=1)                            # (B,)
        sumsinr_db = torch.sum(sinr_db, dim=1)                      # (B,)

        rate    = torch.log2(1.0 + sinr)                            # (B,K)
        sumrate = torch.sum(rate, dim=1)                            # (B,)

        sinr_user_mean = torch.mean(sinr, dim=0)                    # (K,) (average linear SINR first, then convert to dB)
        sinr_user_mean_db = 10.0 * torch.log10(                     # (K,)
            sinr_user_mean.clamp_min(1e-12)
        )                                                        
        rate_user_mean = torch.mean(rate, dim=0)                    # (K,)
        sumrate_mean = torch.mean(sumrate)                          # scalar
        sumsinr_mean = torch.mean(sumsinr)                          # scalar
        sumsinr_mean_db = torch.mean(sumsinr_db)                    # scalar

        # Target sensing SNR

        gH = torch.conj(g_dt).transpose(1, 2)                       # (B,1,M)
        G_sensing = torch.matmul(g_dt, gH)                          # (B,M,M)

        target_echo_C = torch.matmul(G_sensing, W_C)                # (B,M,K)
        target_power  = torch.sum(torch.abs(target_echo_C) ** 2,dim=(1, 2))                                               # (B,)

        target_echo_R = torch.matmul(G_sensing, W_R)            # (B,M,R)
        target_power = target_power + torch.sum(torch.abs(target_echo_R) ** 2,dim=(1, 2))

        target_snr = target_power / noise                           # (B,)
        target_snr_db = 10.0 * torch.log10(
            target_snr.clamp_min(1e-12)
        )                                                           # (B,)

        target_snr_mean = torch.mean(target_snr)                    # scalar
        target_snr_mean_db = 10.0 * torch.log10(                    # scalar
            target_snr_mean.clamp_min(1e-12)
        )              

        return {
            # debug check
            "signal": signal,                         # (B,K)
            "comm_interf": comm_interf,               # (B,K)
            "radar_interf": radar_interf,             # (B,K)
            "noise": noise,                           # scalar

            # raw per-sample tensors
            "sinr": sinr,
            "sinr_db": sinr_db,
            "rate": rate,
            "sumrate": sumrate,
            "target_snr": target_snr,
            "target_snr_db": target_snr_db,

            # B-average display values
            "sinr_user_mean": sinr_user_mean,       # 各個UE SINR
            "sinr_user_mean_db": sinr_user_mean_db,

            "rate_user_mean": rate_user_mean,       # 各個UE rate

            "sumsinr_mean": sumsinr_mean,           # sumsinr
            "sumsinr_mean_db": sumsinr_mean_db,

            "sumrate_mean": sumrate_mean,           # sumrate 

            "target_snr_mean": target_snr_mean,     # 感測SNR
            "target_snr_mean_db": target_snr_mean_db,
        }


class CommNet(ISACNetBase):
    """
    communication beamformer network.
    Input:
        h_dk_hat_pl,    shape = (B, M, K), complex
        h_rk_hat_pl,    shape = (B, N, K), complex
        G_hat_pl   ,    shape = (B, N, M), complex
        g_dt_hat_pl,    shape = (B, M, 1), complex
        -> input,       shape = (B, input_dim), real
    Output:
        output,         shape = (B, 2*M*K) real
        -> W_C,         shape = (B, M, K), complex , Frobenius-normalized
    """

    def __init__(self, hidden_dim: int = 128, ckpt_kind: str = "shortterm_comm"):
            super().__init__(ckpt_kind)

            self.in_dim = 2 * (
                TX_ANT * UAV_COMM +          # h_dk_hat_pl
                RIS_UNIT * UAV_COMM +        # h_rk_hat_pl
                RIS_UNIT * TX_ANT +          # G_hat_pl
                TX_ANT                       # g_dt_hat_pl
            )                                # 3152 real

            self.hidden_dim = hidden_dim
            self.out_complex_dim = TX_ANT * UAV_COMM   # 8*4 = 32 complex entries

            self.fc1 = nn.Linear(self.in_dim, hidden_dim)                   # indim -> 64
            self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)                # 64  -> 128
            self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)                # 128 -> 64

            self.fc_out_real = nn.Linear(hidden_dim, self.out_complex_dim)  # 64  -> 32 (W real part)
            self.fc_out_imag = nn.Linear(hidden_dim, self.out_complex_dim)  # 64  -> 32 (W img  part)

            # 新增 設置初始bias=0
            for layer in [self.fc1,self.fc2,self.fc3,self.fc_out_real,self.fc_out_imag]:
                nn.init.zeros_(layer.bias)

    def forward_mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y_real = self.fc_out_real(x)
        y_imag = self.fc_out_imag(x)

        y = torch.cat([y_real, y_imag], dim=1) # 32+32=64 real 

        return y

    def encode_channels(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl, g_dt_hat_pl):
        """
        將含有 PL 的估測通道 轉成 NN real input

        h_dk_hat_pl : (B, M, K), complex
        h_rk_hat_pl : (B, N, K), complex
        G_hat_pl    : (B, N, M), complex
        g_dt_hat_pl : (B, M, 1), complex

        return:
            x, shape = (B, input_dim), real
        """

        h_dk_hat_pl = torch.as_tensor(h_dk_hat_pl,dtype=torch.complex64,device=self.model_device)
        h_rk_hat_pl = torch.as_tensor(h_rk_hat_pl,dtype=torch.complex64,device=self.model_device)
        G_hat_pl    = torch.as_tensor(G_hat_pl,dtype=torch.complex64,device=self.model_device)
        g_dt_hat_pl = torch.as_tensor(g_dt_hat_pl,dtype=torch.complex64,device=self.model_device)

        """
        # ================================
        # Input scale：每個 complex block 各自 RMS normalize
        # ================================
        h_dk_hat_pl = normalize_complex_block(h_dk_hat_pl)
        h_rk_hat_pl = normalize_complex_block(h_rk_hat_pl)
        G_hat_pl    = normalize_complex_block(G_hat_pl)
        g_dt_hat_pl = normalize_complex_block(g_dt_hat_pl)
        """
        h_dk_hat_pl = 2.8e4 * (h_dk_hat_pl)
        h_rk_hat_pl = 1.5e2   * (h_rk_hat_pl)
        G_hat_pl    = 4.0e3 * (G_hat_pl)
        g_dt_hat_pl = 1.0e3  * (g_dt_hat_pl)


        B = h_dk_hat_pl.shape[0]
        
        h_dk_hat_pl = h_dk_hat_pl.reshape(B, -1)
        h_rk_hat_pl = h_rk_hat_pl.reshape(B, -1)
        G_hat_pl    = G_hat_pl.reshape(B, -1)
        g_dt_hat_pl = g_dt_hat_pl.reshape(B, -1)

        x = torch.cat(
            [
                h_dk_hat_pl.real, h_dk_hat_pl.imag,
                h_rk_hat_pl.real, h_rk_hat_pl.imag,
                G_hat_pl.real,    G_hat_pl.imag,
                g_dt_hat_pl.real, g_dt_hat_pl.imag,
            ],
            dim=1,
        )

        return x
     
    def decode_comm_beamformer(self, y):
        """
        將 NN real output 轉成 communication beamformer W_C.
        y:shape = (B, 2*M*K), real
        return:
            W_C, shape = (B, M, K), complex
        """
        B = y.shape[0]

        y = y.reshape(B, 2, TX_ANT, UAV_COMM)

        W_real = y[:, 0]
        W_imag = y[:, 1]

        W_C = torch.complex(W_real, W_imag).to(torch.complex64)

        return W_C
    
    def forward(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl, g_dt_hat_pl):
        """
        return:
            W_C, shape = (B, M, K), complex
        """

        x = self.encode_channels(h_dk_hat_pl,h_rk_hat_pl,G_hat_pl,g_dt_hat_pl)

        y = self.forward_mlp(x)

        W_C_raw = self.decode_comm_beamformer(y)

        W_C = W_C_raw / (torch.sqrt(torch.sum(torch.abs(W_C_raw) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12) # 功率正規化

        return W_C


class RadarNet(ISACNetBase):
    """
    radar beamformer network.
    Input:
        h_dk_hat_pl,    shape = (B, M, K), complex
        h_rk_hat_pl,    shape = (B, N, K), complex
        G_hat_pl   ,    shape = (B, N, M), complex
        g_dt_hat_pl,    shape = (B, M, 1), complex
        -> input,       shape = (B, input_dim), real
    Output:
        output,         shape = (B, 2*M*RADAR_STREAMS) real
        -> W_R,         shape = (B, M, RADAR_STREAMS), complex , Frobenius-normalized
    """

    def __init__(self, hidden_dim: int = NET, ckpt_kind: str = "shortterm_radar"):
            super().__init__(ckpt_kind)

            self.in_dim = 2 * (
                TX_ANT * UAV_COMM +          # h_dk_hat_pl
                RIS_UNIT * UAV_COMM +        # h_rk_hat_pl
                RIS_UNIT * TX_ANT +          # G_hat_pl
                TX_ANT                       # g_dt_hat_pl
            )                                # 3152 real

            self.hidden_dim = hidden_dim
            self.out_complex_dim = TX_ANT * RADAR_STREAMS   # 8*1 = 8 complex entries

            self.fc1 = nn.Linear(self.in_dim, hidden_dim)                   # indim -> 64
            self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)                # 64  -> 128
            self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)                # 128 -> 64

            self.fc_out_real = nn.Linear(hidden_dim, self.out_complex_dim)  # 64  -> 8 (W real part)
            self.fc_out_imag = nn.Linear(hidden_dim, self.out_complex_dim)  # 64  -> 8 (W img  part)
            
            # 新增 設置初始bias=0
            for layer in [self.fc1,self.fc2,self.fc3,self.fc_out_real,self.fc_out_imag]:
                nn.init.zeros_(layer.bias)

    def forward_mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y_real = self.fc_out_real(x)
        y_imag = self.fc_out_imag(x)

        y = torch.cat([y_real, y_imag], dim=1) # 8+8=16 real 

        return y

    def encode_channels(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl, g_dt_hat_pl):
        """
        將含有 PL 的估測通道與 LT RIS theta 轉成 NN real input

        h_dk_hat_pl : (B, M, K), complex
        h_rk_hat_pl : (B, N, K), complex
        G_hat_pl    : (B, N, M), complex
        g_dt_hat_pl : (B, M, 1), complex

        return:
            x, shape = (B, input_dim), real
        """

        h_dk_hat_pl = torch.as_tensor(h_dk_hat_pl, dtype=torch.complex64, device=self.model_device)
        h_rk_hat_pl = torch.as_tensor(h_rk_hat_pl, dtype=torch.complex64, device=self.model_device)
        G_hat_pl    = torch.as_tensor(G_hat_pl,    dtype=torch.complex64, device=self.model_device)
        g_dt_hat_pl = torch.as_tensor(g_dt_hat_pl, dtype=torch.complex64, device=self.model_device)

        """
        # ================================
        # Input scale：每個 complex block 各自 RMS normalize
        # ================================
        h_dk_hat_pl = normalize_complex_block(h_dk_hat_pl)
        h_rk_hat_pl = normalize_complex_block(h_rk_hat_pl)
        G_hat_pl    = normalize_complex_block(G_hat_pl)
        g_dt_hat_pl = normalize_complex_block(g_dt_hat_pl)
        """

        h_dk_hat_pl = 2.8e4 * (h_dk_hat_pl)
        h_rk_hat_pl = 1.5e2   * (h_rk_hat_pl)
        G_hat_pl    = 4.0e3 * (G_hat_pl)
        g_dt_hat_pl = 1.0e3  * (g_dt_hat_pl)

        B = h_dk_hat_pl.shape[0]

        h_dk_hat_pl = h_dk_hat_pl.reshape(B, -1)
        h_rk_hat_pl = h_rk_hat_pl.reshape(B, -1)
        G_hat_pl    = G_hat_pl.reshape(B, -1)
        g_dt_hat_pl = g_dt_hat_pl.reshape(B, -1)

        x = torch.cat(
            [
                h_dk_hat_pl.real, h_dk_hat_pl.imag,
                h_rk_hat_pl.real, h_rk_hat_pl.imag,
                G_hat_pl.real,    G_hat_pl.imag,
                g_dt_hat_pl.real, g_dt_hat_pl.imag,
            ],
            dim=1,
        )

        return x

    def decode_sensing_beamformer(self, y):
        """
        將 NN real output 轉成 sensing beamformer W_R.
        y:shape = (B, 2*M*RADAR_STREAMS), real
        return:
            W_R, shape = (B, M, RADAR_STREAMS), complex
        """

        B = y.shape[0]

        y = y.reshape(B, 2, TX_ANT, RADAR_STREAMS)

        W_real = y[:, 0]
        W_imag = y[:, 1]

        W_R = torch.complex(W_real, W_imag).to(torch.complex64)

        return W_R

    def forward(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl, g_dt_hat_pl):
        """
        return:
            W_R, shape = (B, M, RADAR_STREAMS), complex
        """

        x = self.encode_channels(h_dk_hat_pl,h_rk_hat_pl,G_hat_pl,g_dt_hat_pl,)

        y = self.forward_mlp(x)

        W_R_raw = self.decode_sensing_beamformer(y)

        W_R = W_R_raw / (torch.sqrt(torch.sum(torch.abs(W_R_raw) ** 2, dim=(1, 2), keepdim=True).real)+ 1e-12) # 功率正規化

        return W_R
    

class ThetaNet(ISACNetBase):
    """
    RIS theta network
    Input:
        h_dk_hat_pl,    shape = (B, M, K), complex
        h_rk_hat_pl,    shape = (B, N, K), complex
        G_hat_pl   ,    shape = (B, N, M), complex
        g_dt_hat_pl,    shape = (B, M, 1), complex
        -> input,       shape = (B, input_dim), real
    Output:
        output,         shape = (B, RIS_UNIT), real
        -> theta,       shape = (B, RIS_UNIT), complex
    """

    def __init__(self, hidden_dim: int = NET, ckpt_kind: str = "shortterm_theta"):
        super().__init__(ckpt_kind)

        self.in_dim = 2 * (
            TX_ANT * UAV_COMM +          # h_dk_hat_pl
            RIS_UNIT * UAV_COMM +        # h_rk_hat_pl
            RIS_UNIT * TX_ANT +          # G_hat_pl
            TX_ANT                       # g_dt_hat_pl
        )

        self.hidden_dim = hidden_dim
        self.out_dim = RIS_UNIT          # 128 real

        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, self.out_dim)

        # 新增 設置初始bias=0
        for layer in [self.fc1,self.fc2,self.fc3,self.fc_out]:
            nn.init.zeros_(layer.bias)
        
    def forward_mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)

        return x

    def encode_channels(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl, g_dt_hat_pl):
        """
        將含有 PL 的估測通道與 LT RIS theta 轉成 NN real input

        h_dk_hat_pl : (B, M, K), complex
        h_rk_hat_pl : (B, N, K), complex
        G_hat_pl    : (B, N, M), complex
        g_dt_hat_pl : (B, M, 1), complex

        return:
            x, shape = (B, input_dim), real
        """

        h_dk_hat_pl = torch.as_tensor(h_dk_hat_pl, dtype=torch.complex64, device=self.model_device)
        h_rk_hat_pl = torch.as_tensor(h_rk_hat_pl, dtype=torch.complex64, device=self.model_device)
        G_hat_pl    = torch.as_tensor(G_hat_pl,    dtype=torch.complex64, device=self.model_device)
        g_dt_hat_pl = torch.as_tensor(g_dt_hat_pl, dtype=torch.complex64, device=self.model_device)

        """
        # ================================
        # Input scale：每個 complex block 各自 RMS normalize
        # ================================
        h_dk_hat_pl = normalize_complex_block(h_dk_hat_pl)
        h_rk_hat_pl = normalize_complex_block(h_rk_hat_pl)
        G_hat_pl    = normalize_complex_block(G_hat_pl)
        g_dt_hat_pl = normalize_complex_block(g_dt_hat_pl)
        """

        h_dk_hat_pl = 2.8e4 * (h_dk_hat_pl)
        h_rk_hat_pl = 1.5e2   * (h_rk_hat_pl)
        G_hat_pl    = 4.0e3 * (G_hat_pl)
        g_dt_hat_pl = 1.0e3  * (g_dt_hat_pl)

        B = h_dk_hat_pl.shape[0]

        h_dk_hat_pl = h_dk_hat_pl.reshape(B, -1)
        h_rk_hat_pl = h_rk_hat_pl.reshape(B, -1)
        G_hat_pl    = G_hat_pl.reshape(B, -1)
        g_dt_hat_pl = g_dt_hat_pl.reshape(B, -1)

        x = torch.cat(
            [
                h_dk_hat_pl.real, h_dk_hat_pl.imag,
                h_rk_hat_pl.real, h_rk_hat_pl.imag,
                G_hat_pl.real,    G_hat_pl.imag,
                g_dt_hat_pl.real, g_dt_hat_pl.imag,
            ],
            dim=1,
        )

        return x

    def decode_theta(self, y):
        """
        y: shape = (B, RIS_UNIT), real-valued phase output
        return:
            theta: shape = (B, RIS_UNIT), complex
            |theta_n| = 1
        """
        # 產生實數向量,代表每個element相位旋轉角度
        phase = y
        # 變成RIS相位旋轉向量
        theta = torch.exp(1j * phase).to(torch.complex64)
        return theta

    def forward(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl, g_dt_hat_pl):
        """
        return:
            theta, shape = (B, RIS_UNIT), complex RIS vector
        """

        x = self.encode_channels(
            h_dk_hat_pl,
            h_rk_hat_pl,
            G_hat_pl,
            g_dt_hat_pl,
        )

        y = self.forward_mlp(x)

        theta = self.decode_theta(y)

        return theta



class TestNet(ISACNetBase):
    """
    Direct effective-channel-to-RZF network.

    Input:
        H_eff_H : (B,K,M), complex

    Output:
        W_C     : (B,M,K), complex
                  Frobenius-normalized
    """

    def __init__(self, hidden_dim: int = 64, ckpt_kind: str = "test_net"):
        super().__init__(ckpt_kind)

        self.in_dim = 2 * UAV_COMM * TX_ANT
        self.hidden_dim = hidden_dim
        self.out_complex_dim = TX_ANT * UAV_COMM

        self.fc1 = nn.Linear(self.in_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,2 * hidden_dim)
        self.fc3 = nn.Linear(2 * hidden_dim,hidden_dim)

        self.fc_out_real = nn.Linear(hidden_dim,self.out_complex_dim)
        self.fc_out_imag = nn.Linear(hidden_dim,self.out_complex_dim)

        for layer in [self.fc1,self.fc2,self.fc3,self.fc_out_real,self.fc_out_imag]:
            nn.init.zeros_(layer.bias)

    def forward_mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        y_real = self.fc_out_real(x)
        y_imag = self.fc_out_imag(x)

        y = torch.cat([y_real,y_imag],dim=1)

        return y

    def encode_channels(self, H_eff_H):
        """
        H_eff_H : (B,K,M), complex

        Return:
            x : (B,2*K*M), real
        """

        H_eff_H = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=self.model_device)

        B = H_eff_H.shape[0]

        # 固定尺度縮放，不改變不同 channel 間的相對大小
        H_eff_H_scaled = 1.0e4 * H_eff_H
        H_eff_H_flat = H_eff_H_scaled.reshape(B,-1)

        x = torch.cat(
            [
                H_eff_H_flat.real,
                H_eff_H_flat.imag,
            ],
            dim=1,
        )

        if x.shape[1] != self.in_dim:
            raise RuntimeError(
                f"TestNet input dimension mismatch: "
                f"x.shape={tuple(x.shape)}, expected in_dim={self.in_dim}"
            )

        return x

    def decode_comm_beamformer(self, y):
        """
        y   : (B,2*M*K), real
        W_C : (B,M,K), complex
        """

        B = y.shape[0]

        if y.shape[1] != 2 * self.out_complex_dim:
            raise RuntimeError(
                f"TestNet output dimension mismatch: "
                f"y.shape={tuple(y.shape)}, "
                f"expected second dimension={2 * self.out_complex_dim}"
            )

        y = y.reshape(B,2,TX_ANT,UAV_COMM)

        W_real = y[:,0]
        W_imag = y[:,1]

        W_C = torch.complex(W_real,W_imag).to(torch.complex64)

        return W_C

    def forward(self, H_eff_H):
        """
        Return:
            W_C_dir : (B,M,K), complex
                      ||W_C_dir||_F^2 = 1 for each sample
        """

        x = self.encode_channels(H_eff_H)
        y = self.forward_mlp(x)

        W_C_raw = self.decode_comm_beamformer(y)

        W_C_norm = torch.sqrt(
            torch.sum(torch.abs(W_C_raw) ** 2,dim=(1,2),keepdim=True).real
        )

        W_C_dir = W_C_raw / (W_C_norm + 1e-12)

        return W_C_dir
    






