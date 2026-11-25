# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

# ================================
# 共用計算函式
# 注意以下函式需要吃batch
# ================================
def compute_effective_channel(h_dk, h_rk, G, phi):
    """
    根據已帶 PL 的通道組裝 RIS-assisted effective channel

    Input:
        h_dk  : (B, M, K), complex
        h_rk  : (B, N, K), complex
        G     : (B, N, M), complex
        phi   : (B, N), complex

    Return:
        H_eff_H : (B, K, M), complex

    H_eff_H[b,k,m]
        = conj(h_dk[b, m, k])
        + sum_n conj(h_rk[b, n, k]) * phi[b,n] * G[b,n,m]
    """
    h_dk  = torch.as_tensor(h_dk, dtype=torch.complex64)
    h_rk  = torch.as_tensor(h_rk, dtype=torch.complex64)
    G     = torch.as_tensor(G, dtype=torch.complex64)
    phi   = torch.as_tensor(phi, dtype=torch.complex64)

    B = h_dk.shape[0]

    if phi.shape != (B, RIS_UNIT):
        raise ValueError(f"phi shape must be {(B,RIS_UNIT)}, but got {tuple(phi.shape)}")

    # 各使用者的直接通道 h_{d,k}^H
    H_direct_H = h_dk.conj().transpose(1, 2)    # (B, K, M)

    # 各使用者的反射通道 h_{r,k}^H diag(phi) G
    h_rk_H = h_rk.conj().transpose(1, 2)        # (B, K, N)
    H_ris_H = torch.einsum("bkn,bn,bnm->bkm", h_rk_H, phi, G)

    H_eff_H = H_direct_H + H_ris_H              # (B, K, M)

    return H_eff_H


def compute_isac_batch_performance(H_eff_H, g_dt, W):
    """
    計算 communication 與 sensing performance

    Input:
        H_eff_H : (B,K,M), complex
        g_dt    : (B,M,1), complex
        W       : (B,M,K), complex joint beamformer

    Return:
        signal        : (B,K)
        comm_interf   : (B,K)
        noise         : scalar
        sinr          : (B,K)
        rate          : (B,K)
        target_snr    : (B,)
        target_snr_db : (B,)
    """
    H_eff_H = torch.as_tensor(H_eff_H, dtype=torch.complex64)
    g_dt    = torch.as_tensor(g_dt, dtype=torch.complex64)
    W       = torch.as_tensor(W, dtype=torch.complex64)

    noise       = H_eff_H.real.new_tensor(NOISE_POWER)
    radar_noise = H_eff_H.real.new_tensor(NOISE_POWER)

    B = H_eff_H.shape[0]
    
    if H_eff_H.shape != (B,UAV_COMM,TX_ANT):
        raise ValueError(f"H_eff_H shape error: got {tuple(H_eff_H.shape)}")

    if g_dt.shape != (B,TX_ANT,1):
        raise ValueError(f"g_dt shape error: got {tuple(g_dt.shape)}")

    if W.shape != (B,TX_ANT,UAV_COMM):
        raise ValueError(f"W shape error: got {tuple(W.shape)}")

    # --------------------------------
    # Communication SINR / Rate
    # --------------------------------
    # 所有 UE 接收各個 beamformer 的訊號
    Y = torch.matmul(H_eff_H, W)                                   # (B,K,K)
    P = torch.abs(Y)**2                                            # (B,K,K)

    # 期望訊號功率
    signal = torch.diagonal(P, dim1=1, dim2=2)                     # (B,K)

    # 多使用者干擾功率
    comm_interf = (P - torch.diag_embed(signal)).sum(dim=2)        # (B,K)

    # Communication SINR 與 Rate
    sinr = signal / (comm_interf + noise)                          # (B,K)
    rate = torch.log2(1.0 + sinr)                                  # (B,K)

    # --------------------------------
    # Target sensing SNR：Rayleigh Quotient
    # --------------------------------
    # W^H g: (B, K, M) @ (B, M, 1) -> (B, K, 1)
    W_H_g = torch.matmul(W.conj().transpose(1, 2), g_dt)

    # ||W^H g||_2^2
    sensing_power = W_H_g.abs().square().sum(dim=(1, 2))    # (B,)

    # ||g||_2^2
    channel_power = g_dt.abs().square().sum(dim=(1, 2))     # (B,)

    # 最佳接收波束下的感測 SNR，式 (3.22)
    target_snr    = RCS_POWER * sensing_power * channel_power / radar_noise  # (B,)
    target_snr_db = 10.0 * torch.log10(target_snr.clamp_min(1e-12))          # (B,)

    return {
        "signal": signal,
        "comm_interf": comm_interf,
        "noise": noise,
        "sinr": sinr,
        "rate": rate,
        "target_snr": target_snr,
        "target_snr_db": target_snr_db,
    }


# ================================
# neural net
# ================================
class ISACNetBase(nn.Module):
    """
    Neural network 共用基礎函式
    """
    def __init__(self):     #初始化
        super().__init__()
    
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
            raise FileNotFoundError(f"[load_model] 找不到 checkpoint: {os.path.abspath(path)}")

        state = torch.load(path, map_location=self.model_device)
        self.load_state_dict(state, strict=strict)

        if verbose:
            print(f"[load_model] 已載入：{path}")


class JointNet(ISACNetBase):
    """
    Joint sensing and communication beamformer network.
    Input:
        H_eff_H : (B, K, M), complex
        g_dt    : (B, M, 1), complex

        -> NN real input:
           shape = (B, 2*(K*M + M))

    Output:
        output : (B, 2*M*K), real

        -> W : (B, M, K), complex

    W simultaneously performs:
        1. multi-user communication
        2. target sensing

    Total transmit power is hard-normalized to:
        ||W||_F^2 = TRANSMIT_POWER_TOTAL
    """

    def __init__(self, hidden_dim: int = NET, num_hidden_layers: int = LAYER):
        super().__init__()
        # H_eff_H: K*M complex
        # g_dt   : M complex
        # complex -> real/imag，因此乘 2
        self.in_dim = 2 * (UAV_COMM * TX_ANT + TX_ANT)

        self.hidden_dim = hidden_dim
        # Joint beamformer W shape = (M,K)共 M*K 個 complex coefficients
        self.out_complex_dim = TX_ANT * UAV_COMM 
        
        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)

        # 額外增加的隱藏層 2d → 2d，各有自己的權重
        self.extra_hidden_layers = nn.ModuleList(nn.Linear(2 * hidden_dim, 2 * hidden_dim)for _ in range(num_hidden_layers - 3))

        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)

        self.fc_out_real = nn.Linear(hidden_dim, self.out_complex_dim)
        self.fc_out_imag = nn.Linear(hidden_dim, self.out_complex_dim)
        """# 所有 layer bias 初始化為 0
        for layer in [
            self.fc1,
            self.fc2,
            self.fc3,
            self.fc_out_real,
            self.fc_out_imag,
        ]:
            nn.init.zeros_(layer.bias)
        
        """


    def forward_mlp(self, x):
        """
        Input:
            x : (B, in_dim), real

        Output:
            y : (B, 2*M*K), real
        """

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 額外增加的隱藏層
        for layer in self.extra_hidden_layers:
            x = F.relu(layer(x))
            
        x = F.relu(self.fc3(x))

        y_real = self.fc_out_real(x)
        y_imag = self.fc_out_imag(x)

        # 前半部為 real，後半部為 imaginary
        y = torch.cat([y_real, y_imag], dim=1)

        return y


    def encode_channels(self, H_eff_H, g_dt):
        """
        將含 path loss 的 effective communication channel
        與 target channel 轉成 NN real-valued input。

        Input:
            H_eff_H : (B, K, M), complex
            g_dt    : (B, M, 1), complex

        Output:
            x : (B, 2*(K*M + M)), real
        """

        H_eff_H = torch.as_tensor(H_eff_H,dtype=torch.complex64,device=self.model_device)
        g_dt    = torch.as_tensor(g_dt,dtype=torch.complex64,device=self.model_device)

        B       = H_eff_H.shape[0]

        # 固定尺度縮放
        # 通道已包含 path loss，數值通常很小；
        # 先放大到較適合神經網路學習的數值範圍，避免輸入過小造成梯度與特徵差異不明顯
        H_eff_H_scaled = 1.0e4*H_eff_H
        g_dt_scaled    = 1.0e3*g_dt

        # 展平成 NN 輸入向量
        H_eff_H_flat = H_eff_H_scaled.reshape(B,-1)
        g_dt_flat    = g_dt_scaled.reshape(B,-1)

        # complex channel 拆成 real / imaginary
        x = torch.cat([H_eff_H_flat.real,H_eff_H_flat.imag,g_dt_flat.real,g_dt_flat.imag],dim=1)

        return x


    def decode_joint_beamformer(self, y):
        """
        將 NN real output 轉成 joint beamformer W,並投影至完整 BS transmit power。
        Input:
            y : (B, 2*M*K), real
        Output:
            W : (B, M, K), complex
        每個 sample 滿足：
            ||W||_F^2 = TRANSMIT_POWER_TOTAL
        """
        B = y.shape[0]

        # (B,2*M*K) -> (B,2,M,K)
        y = y.reshape(B,2,TX_ANT,UAV_COMM)

        W_real = y[:,0]
        W_imag = y[:,1]

        # real / imaginary -> complex beamformer
        W = torch.complex(W_real,W_imag).to(torch.complex64)

        # 計算每個 sample 的 beamformer total power
        W_power = torch.sum(torch.abs(W)**2,dim=(1,2),keepdim=True).real

        # Frobenius normalization 並投影至完整 BS transmit power
        if torch.any(W_power == 0):
            raise ValueError("原始波束矩陣為零，無法依式 (3.24) 正規化。")

        W = (TRANSMIT_POWER_TOTAL ** 0.5) * W / torch.sqrt(W_power)
        
        return W


    def forward(self, H_eff_H, g_dt):
        """
        Input:
            H_eff_H : (B, K, M), complex
            g_dt    : (B, M, 1), complex

        Return:
            W : (B, M, K), complex

        且每個 sample 都滿足：
            ||W||_F^2 = TRANSMIT_POWER_TOTAL
        """

        # channel -> NN real input
        x = self.encode_channels(H_eff_H,g_dt)

        # NN prediction
        y = self.forward_mlp(x)

        # real output -> complex joint beamformer，並投影至完整 transmit power
        W = self.decode_joint_beamformer(y)

        return W

    
class RISNet(ISACNetBase):
    """
    RIS theta network
    Input:
        h_dk_hat_pl,    shape = (B, M, K), complex
        h_rk_hat_pl,    shape = (B, N, K), complex
        G_hat_pl   ,    shape = (B, N, M), complex
        -> input,       shape = (B, input_dim), real
    Output:
        output,         shape = (B, RIS_UNIT), real
        -> phi,         shape = (B, RIS_UNIT), complex
    """

    def __init__(self, hidden_dim: int = NET, num_hidden_layers: int = LAYER):
        super().__init__()

        self.in_dim = 2 * (
            TX_ANT * UAV_COMM +          # h_dk_hat_pl
            RIS_UNIT * UAV_COMM +        # h_rk_hat_pl
            RIS_UNIT * TX_ANT            # G_hat_pl
        )                                # 2*(8+128+256) = 784

        self.hidden_dim = hidden_dim
        self.out_dim = RIS_UNIT          # N = 1024 real

        self.fc1 = nn.Linear(self.in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * hidden_dim)

        # 額外增加的隱藏層 2d → 2d，各有自己的權重
        self.extra_hidden_layers = nn.ModuleList(nn.Linear(2 * hidden_dim, 2 * hidden_dim)for _ in range(num_hidden_layers - 3))

        self.fc3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, self.out_dim)

        """# 新增 設置初始bias=0
        for layer in [self.fc1,self.fc2,self.fc3,self.fc_out]:
            nn.init.zeros_(layer.bias)
        """

    def forward_mlp(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 額外增加的隱藏層
        for layer in self.extra_hidden_layers:
            x = F.relu(layer(x))

        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)

        return x

    def encode_channels(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl):
        """
        將三組帶 PL 的估測通道轉成 RISNet real-valued input

        h_dk_hat_pl : (B, M, K), complex
        h_rk_hat_pl : (B, N, K), complex
        G_hat_pl    : (B, N, M), complex

        return:
            x, shape = (B, input_dim), real
        """

        h_dk_hat_pl = torch.as_tensor(h_dk_hat_pl, dtype=torch.complex64, device=self.model_device)
        h_rk_hat_pl = torch.as_tensor(h_rk_hat_pl, dtype=torch.complex64, device=self.model_device)
        G_hat_pl    = torch.as_tensor(G_hat_pl,    dtype=torch.complex64, device=self.model_device)

        h_dk_hat_pl = 2.8e4  * (h_dk_hat_pl)
        h_rk_hat_pl = 1.5e2  * (h_rk_hat_pl)
        G_hat_pl    = 4.0e3  * (G_hat_pl)

        B = h_dk_hat_pl.shape[0]

        h_dk_hat_pl = h_dk_hat_pl.reshape(B, -1)
        h_rk_hat_pl = h_rk_hat_pl.reshape(B, -1)
        G_hat_pl    = G_hat_pl.reshape(B, -1)

        x = torch.cat(
            [
                h_dk_hat_pl.real, h_dk_hat_pl.imag,
                h_rk_hat_pl.real, h_rk_hat_pl.imag,
                G_hat_pl.real,    G_hat_pl.imag,
            ],
            dim=1,
        )

        return x

    def decode_phi(self, y):
        """
        y: shape = (B, RIS_UNIT), real-valued phase output
        return:
            phi: shape = (B, RIS_UNIT), complex
            |phi_n| = 1
        """
        # 產生實數向量,代表每個element相位旋轉角度
        phase = y
        # 變成RIS相位旋轉向量 
        phi = torch.exp(1j * phase).to(torch.complex64)
        return phi

    def forward(self, h_dk_hat_pl, h_rk_hat_pl, G_hat_pl):
        """
        return:
            phi, shape = (B, RIS_UNIT), complex RIS vector
        """

        x = self.encode_channels(h_dk_hat_pl,h_rk_hat_pl,G_hat_pl)

        y = self.forward_mlp(x)    # 輸出時數的旋轉相位

        phi = self.decode_phi(y) # 轉換為RIS 向量phi

        return phi


