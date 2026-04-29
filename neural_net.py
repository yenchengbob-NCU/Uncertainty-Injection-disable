# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

def build_ckpt_path(kind: str) -> str:
    return os.path.join(CKPT_DIR, f"{kind}_{SETTING_STRING}.ckpt")


class ISACNetBase(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 200, ckpt_kind: str | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

        self.model_path = build_ckpt_path(ckpt_kind) if ckpt_kind is not None else ""

    @property
    def model_device(self):
        return next(self.parameters()).device
    
    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc_out(x)
        return x

    def decode_complex(self, y: torch.Tensor, out_shape) -> torch.Tensor:
        B = y.shape[0]
        y = y.view(B, -1, 2).contiguous()
        z = torch.view_as_complex(y)
        return z.view(B, *out_shape)

    def load_model(self, path: str | None = None, strict: bool = True, verbose: bool = True):
        load_path = path if path is not None else self.model_path
        if not load_path:
            if verbose:
                print("[load_model] 未提供 checkpoint 路徑。")
            return

        if not os.path.exists(load_path):
            if verbose:
                print(f"[load_model] 找不到 checkpoint：{load_path}")
            return

        state = torch.load(load_path, map_location=self.model_device)
        self.load_state_dict(state, strict=strict)
        if verbose:
            print(f"[load_model] 已載入：{load_path}")

    def save_model(self, path: str | None = None, verbose: bool = True):
        save_path = path if path is not None else self.model_path
        if not save_path:
            raise ValueError("save_model() 沒有可用的儲存路徑。")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        if verbose:
            print(f"[save_model] 已儲存：{save_path}")

    # ============================================================
    # Long-term 輸入編碼 (1*4)
    # ============================================================

    def encode_positions(self, ue_layout_batch) -> torch.Tensor:
        x = torch.as_tensor(ue_layout_batch, dtype=torch.float32, device=self.model_device)

        if x.ndim == 2:
            if x.shape != (UAV_COMM, 2):
                raise ValueError(f"ue_layout shape 應為 ({UAV_COMM}, 2)，收到 {tuple(x.shape)}")
            x = x.unsqueeze(0)

        if x.ndim != 3 or x.shape[1:] != (UAV_COMM, 2):
            raise ValueError(f"ue_layout_batch shape 應為 (B, {UAV_COMM}, 2)，收到 {tuple(x.shape)}")

        B = x.shape[0]
        return x.reshape(B, -1)

    # ============================================================
    # Short-term 輸入編碼
    # ============================================================

    def encode_shortterm_inputs(self, h_dk, h_rk, G, g_dt, theta) -> torch.Tensor:
        h_dk = torch.as_tensor(h_dk, dtype=torch.complex64, device=self.model_device)
        h_rk = torch.as_tensor(h_rk, dtype=torch.complex64, device=self.model_device)
        G    = torch.as_tensor(G,    dtype=torch.complex64, device=self.model_device)
        g_dt = torch.as_tensor(g_dt, dtype=torch.complex64, device=self.model_device)

        if h_dk.ndim != 3 or h_dk.shape[1:] != (TX_ANT, UAV_COMM):
            raise ValueError(f"h_dk shape 應為 (B,{TX_ANT},{UAV_COMM})，收到 {tuple(h_dk.shape)}")
        if h_rk.ndim != 3 or h_rk.shape[1:] != (RIS_UNIT, UAV_COMM):
            raise ValueError(f"h_rk shape 應為 (B,{RIS_UNIT},{UAV_COMM})，收到 {tuple(h_rk.shape)}")
        if G.ndim != 3 or G.shape[1:] != (RIS_UNIT, TX_ANT):
            raise ValueError(f"G shape 應為 (B,{RIS_UNIT},{TX_ANT})，收到 {tuple(G.shape)}")
        if g_dt.ndim != 3 or g_dt.shape[1:] != (TX_ANT, 1):
            raise ValueError(f"g_dt shape 應為 (B,{TX_ANT},1)，收到 {tuple(g_dt.shape)}")

        B = h_dk.shape[0]
        if h_rk.shape[0] != B or G.shape[0] != B or g_dt.shape[0] != B:
            raise ValueError("short-term channel batch 維度不一致。")

        theta = self._expand_theta_batch(theta, B)

        x = torch.cat([
            torch.view_as_real(h_dk).reshape(B, -1),
            torch.view_as_real(h_rk).reshape(B, -1),
            torch.view_as_real(G).reshape(B, -1),
            torch.view_as_real(g_dt).reshape(B, -1),
            torch.view_as_real(theta).reshape(B, -1),
        ], dim=1)

        return x.to(torch.float32)

    def _expand_theta_batch(self, theta, batch_size: int) -> torch.Tensor:
        theta = torch.as_tensor(theta, dtype=torch.complex64, device=self.model_device)

        if theta.ndim == 1:
            if theta.shape[0] != RIS_UNIT:
                raise ValueError(f"theta shape 應為 ({RIS_UNIT},)，收到 {tuple(theta.shape)}")
            theta = theta.unsqueeze(0)

        if theta.ndim != 2 or theta.shape[1] != RIS_UNIT:
            raise ValueError(f"theta shape 應為 (B,{RIS_UNIT}) 或 ({RIS_UNIT},)，收到 {tuple(theta.shape)}")

        if theta.shape[0] == 1 and batch_size > 1:
            theta = theta.expand(batch_size, RIS_UNIT)

        if theta.shape[0] != batch_size:
            raise ValueError(
                f"theta batch={theta.shape[0]} 與 channel batch={batch_size} 不一致，且無法 broadcast。"
            )
        return theta

    # ============================================================
    # Utility
    # ============================================================

    def compute_comm_sinrs(self, h_dk, h_rk, G, theta, W_R, W_C, pl_BS_UE, pl_BS_RIS_UE):
        h_dk = torch.as_tensor(h_dk, dtype=torch.complex64, device=self.model_device)
        h_rk = torch.as_tensor(h_rk, dtype=torch.complex64, device=self.model_device)
        G    = torch.as_tensor(G,    dtype=torch.complex64, device=self.model_device)
        W_R  = torch.as_tensor(W_R,  dtype=torch.complex64, device=self.model_device)
        W_C  = torch.as_tensor(W_C,  dtype=torch.complex64, device=self.model_device)

        B = h_dk.shape[0]
        theta = self._expand_theta_batch(theta, B)
        theta_mat = torch.diag_embed(theta)

        pl_dk_t  = torch.as_tensor(pl_BS_UE, dtype=torch.float32, device=self.model_device).view(1, UAV_COMM, 1)
        pl_ris_t = torch.as_tensor(pl_BS_RIS_UE, dtype=torch.float32, device=self.model_device).view(1, UAV_COMM, 1)

        amp_dk  = torch.sqrt(pl_dk_t).to(torch.complex64)
        amp_ris = torch.sqrt(pl_ris_t).to(torch.complex64)

        Hdk_H  = torch.conj(h_dk).transpose(1, 2)
        hrk_H  = torch.conj(h_rk).transpose(1, 2)
        thetaG = torch.matmul(theta_mat, G)
        Hris_H = torch.matmul(hrk_H, thetaG)

        H_eff_H = amp_dk * Hdk_H + amp_ris * Hris_H

        S = torch.matmul(H_eff_H, W_C)
        P = (S.abs()) ** 2

        signal = torch.diagonal(P, dim1=1, dim2=2)
        interf_comm = P.sum(dim=2) - signal
        radar_proj = torch.matmul(H_eff_H, W_R)             # (B,K,L_R)
        interf_radar = (radar_proj.abs() ** 2).sum(dim=2)   # (B,K)

        noise = torch.as_tensor(NOISE_POWER, dtype=signal.dtype, device=self.model_device)
        sinrs = signal / (interf_comm + interf_radar + noise)
        return sinrs

    def compute_rates(self, sinrs):
        return torch.log1p(sinrs) / np.log(2.0)

    def compute_sum_rate(self, h_dk, h_rk, G, theta, W_R, W_C, pl_BS_UE, pl_BS_RIS_UE):
        sinrs = self.compute_comm_sinrs(h_dk, h_rk, G, theta, W_R, W_C, pl_BS_UE, pl_BS_RIS_UE)
        rates = self.compute_rates(sinrs)
        return rates.sum(dim=1)

    def compute_sense_snr(self, g_dt, W_R, W_C, pl_BS_TAR_BS):
        g_dt = torch.as_tensor(g_dt, dtype=torch.complex64, device=self.model_device)
        W_R  = torch.as_tensor(W_R,  dtype=torch.complex64, device=self.model_device)
        W_C  = torch.as_tensor(W_C,  dtype=torch.complex64, device=self.model_device)

        v_t = g_dt @ g_dt.conj().transpose(1, 2)
        Q_comm  = W_C @ W_C.conj().transpose(1, 2)
        Q_radar = W_R @ W_R.conj().transpose(1, 2)
        Q = Q_comm + Q_radar

        A = v_t @ Q @ v_t.conj().transpose(1, 2)
        eigvals = torch.linalg.eigvalsh(A)
        lambda_max = eigvals[:, -1].clamp_min(0.0)

        pl = torch.as_tensor(pl_BS_TAR_BS, dtype=lambda_max.dtype, device=self.model_device)
        noise = torch.as_tensor(NOISE_POWER, dtype=lambda_max.dtype, device=self.model_device)

        return (pl / noise) * lambda_max

    def compute_tx_power(self, W_C, W_R):
        W_C = torch.as_tensor(W_C, dtype=torch.complex64, device=self.model_device)
        W_R = torch.as_tensor(W_R, dtype=torch.complex64, device=self.model_device)
        return (W_C.abs() ** 2).sum(dim=(1, 2)) + (W_R.abs() ** 2).sum(dim=(1, 2))

    def compute_ris_amplitude_penalty(self, theta):
        theta = torch.as_tensor(theta, dtype=torch.complex64, device=self.model_device)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        excess = torch.clamp(theta.abs() - 1.0, min=0.0)
        return excess.mean(dim=1)


class LongTermPositionNet(ISACNetBase):
    def __init__(self, hidden_dim: int = 200, ckpt_kind: str = "longterm"):
        in_dim = 2 * UAV_COMM                       # 輸入維度 UE座標組
        out_theta = 2 * RIS_UNIT                    # 輸出維度 RIS向量
        out_wc = 2 * (TX_ANT * UAV_COMM)            # 輸出維度 LT_W_c
        out_wr = 2 * (TX_ANT * RADAR_STREAMS)       # 輸出維度 LT_W_r
        out_dim = out_theta + out_wc + out_wr

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            ckpt_kind=ckpt_kind
        )

        self.out_theta = out_theta
        self.out_wc = out_wc
        self.out_wr = out_wr

    def forward(self, ue_layout_batch):
        x = self.encode_positions(ue_layout_batch)
        y = self.forward_mlp(x)

        y_theta = y[:, :self.out_theta]
        y_wc = y[:, self.out_theta:self.out_theta + self.out_wc]
        y_wr = y[:, self.out_theta + self.out_wc:]

        theta_lt = self.decode_complex(y_theta, (RIS_UNIT,))            # 還原
        W_C_lt   = self.decode_complex(y_wc, (TX_ANT, UAV_COMM))        # 還原
        W_R_lt   = self.decode_complex(y_wr, (TX_ANT, RADAR_STREAMS))   # 還原
        return theta_lt, W_C_lt, W_R_lt

class ShortTermCommNet(ISACNetBase):
    def __init__(self, hidden_dim: int = 200, ckpt_kind: str = "short_comm"):
        in_dim = 2 * (
            TX_ANT * UAV_COMM +
            RIS_UNIT * UAV_COMM +
            RIS_UNIT * TX_ANT +
            TX_ANT +
            RIS_UNIT
        )
        out_dim = 2 * (TX_ANT * UAV_COMM)

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            ckpt_kind=ckpt_kind
        )

    def forward(self, h_dk, h_rk, G, g_dt, theta):
        x = self.encode_shortterm_inputs(h_dk, h_rk, G, g_dt, theta)
        y = self.forward_mlp(x)
        W_C = self.decode_complex(y, (TX_ANT, UAV_COMM))
        return W_C

class ShortTermRadarNet(ISACNetBase):
    def __init__(self, hidden_dim: int = 200, ckpt_kind: str = "short_radar"):
        in_dim = 2 * (
            TX_ANT * UAV_COMM +
            RIS_UNIT * UAV_COMM +
            RIS_UNIT * TX_ANT +
            TX_ANT +
            RIS_UNIT
        )
        out_dim = 2 * (TX_ANT * RADAR_STREAMS)

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            ckpt_kind=ckpt_kind
        )

    def forward(self, h_dk, h_rk, G, g_dt, theta):
        x = self.encode_shortterm_inputs(h_dk, h_rk, G, g_dt, theta)
        y = self.forward_mlp(x)
        W_R = self.decode_complex(y, (TX_ANT, 1))
        return W_R
