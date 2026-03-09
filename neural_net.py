# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *  

def build_ckpt_path(kind: str, robust: bool = False) -> str:
    ckpt_dir = CKPT_DIR
    if robust:
        fname = f"{kind}_robust_{SETTING_STRING}.ckpt"
    else:
        fname = f"{kind}_{SETTING_STRING}.ckpt"
    return os.path.join(ckpt_dir, fname)

VERBOSE_LOAD = True # DEBUG log 開關
# ------------------------------
# 基底網路
# ------------------------------
class BaseMLP(nn.Module):
    def __init__(self,out_dim):
        super().__init__()
        self.model_path = ""
        self.n_hidden_layer_neurons = 200
        # in_dim {h_dk (M*1) h_rk (N*1) G (N*M) g_dt(M*1)}*2 實虛部
        in_dim = 2 * (TX_ANT*UAV_COMM + RIS_UNIT*UAV_COMM + RIS_UNIT*TX_ANT + TX_ANT*1)
        self.fc_1 = nn.Linear(in_dim, self.n_hidden_layer_neurons)
        self.fc_2 = nn.Linear(self.n_hidden_layer_neurons, self.n_hidden_layer_neurons)
        self.fc_3 = nn.Linear(self.n_hidden_layer_neurons, self.n_hidden_layer_neurons)
        self.fc_4 = nn.Linear(self.n_hidden_layer_neurons, self.n_hidden_layer_neurons)
        self.fc_5 = nn.Linear(self.n_hidden_layer_neurons, out_dim)

    def encode_inputs(self,
                      h_dk: torch.Tensor,
                      h_rk: torch.Tensor,
                      G: torch.Tensor,
                      g_dt: torch.Tensor) -> torch.Tensor:
        
        # 1) 檢查維度
        assert h_dk.ndim == 3 and h_dk.shape[1:] == (TX_ANT, UAV_COMM)
        assert h_rk.ndim == 3 and h_rk.shape[1:] == (RIS_UNIT, UAV_COMM)
        assert G.ndim    == 3 and G.shape[1:]    == (RIS_UNIT, TX_ANT)
        assert g_dt.ndim == 3 and g_dt.shape[1:] == (TX_ANT, 1)

        B = h_dk.shape[0]
        assert all(t.shape[0] == B for t in (h_rk, G, g_dt)), "batch 維不一致"

        # 2) 各自轉成 real，展平後在最後一維 concat
        x = torch.cat([
            torch.view_as_real(h_dk).reshape(B, -1),
            torch.view_as_real(h_rk).reshape(B, -1),
            torch.view_as_real(G).reshape(B, -1),
            torch.view_as_real(g_dt).reshape(B, -1),
        ], dim=1)

        # 3) 確保是 float32
        return x.to(torch.float32)

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        共用的 MLP 前向運算:
        x: (B, in_dim) real -> (B, out_dim) real
        不做 softmax / normalize,交給上層決定怎麼 decode 成 beamformer / RIS。
        """

        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        x = self.fc_5(x)
        return x

    def decode(self, y: torch.Tensor, out_shape) -> torch.Tensor:
        """
        將輸出反解回複數: (B, 2*D) -> (B, *out_shape) complex
        out_shape 例如 (M, K) 或 (N,) 或 (RX_ANT,1)
        """
        B = y.shape[0]
        y = y.view(B, -1, 2).contiguous()
        z = torch.view_as_complex(y)               # (B, D) complex
        return z.view(B, *out_shape)

    def compute_comm_sinrs(self,
                        h_dk: torch.Tensor,   # (B,M,K)
                        h_rk: torch.Tensor,   # (B,N,K)
                        G: torch.Tensor,      # (B,N,M)
                        phi: torch.Tensor,    # (B,N)
                        W_S: torch.Tensor,    # (B,M,1)
                        W_C: torch.Tensor,    # (B,M,K)
                        pl_BS_UE,             # (K,)
                        pl_BS_RIS_UE          # (K,)
                        ) -> torch.Tensor:    # (B,K)
        """
        回傳:SINR (B,K)
        H_k^H = h_dk^H + h_rk^H Φ G
        SINR_k = |H_k^H w_k|^2 / (Σ_{i≠k}|H_k^H w_i|^2 + |H_k^H w_ϑ|^2 + σ_k^2)
        """
        B, M, K = h_dk.shape

        # 先建立 Φ，避免未賦值使用
        phi_mat = torch.diag_embed(phi)          # (B,N,N)

        # 型別/裝置對齊
        device, dtype = W_C.device, W_C.dtype
        h_dk, h_rk, G, Phi, W_S, W_C = [t.to(device=device, dtype=dtype)
                                        for t in (h_dk, h_rk, G, phi_mat, W_S, W_C)]

        # --- Pathloss 只在 SINR 用: 輸入功率衰減 先轉 tensor，再取 sqrt 當振幅縮放 ---
        pl_dk_t  = torch.as_tensor(pl_BS_UE,     device=device, dtype=torch.float32).view(1, K, 1)
        pl_ris_t = torch.as_tensor(pl_BS_RIS_UE, device=device, dtype=torch.float32).view(1, K, 1)

        amp_dk  = torch.sqrt(pl_dk_t).to(dtype=dtype)    # (1,K,1) complex dtype
        amp_ris = torch.sqrt(pl_ris_t).to(dtype=dtype)   # (1,K,1) complex dtype

        # --- 分開算 direct 與 RIS 路徑，再套用 amp ---
        Hdk_H = torch.conj(h_dk).transpose(1, 2)                          # (B,K,M)
        hrk_H = torch.conj(h_rk).transpose(1, 2)                          # (B,K,N)
        PhG   = torch.matmul(Phi, G)                                      # (B,N,M)
        Hris_H = torch.matmul(hrk_H, PhG)                                 # (B,K,M)

        # 有效通道（相干相加）：sqrt(beta_dk)*direct + sqrt(beta_rk*beta_G)*RIS
        H_eff_H = amp_dk * Hdk_H + amp_ris * Hris_H                       # (B,K,M)

        # --- 你原本的 SINR 計算完全照舊 ---
        S = torch.matmul(H_eff_H, W_C)                                    # (B,K,K)
        P = (S.abs())**2                                                  # (B,K,K) real

        signal = torch.diagonal(P, dim1=1, dim2=2)                        # (B,K)
        interf_comm  = P.sum(dim=2) - signal                              # (B,K)
        interf_sense = (torch.matmul(H_eff_H, W_S).abs()**2).squeeze(-1)   # (B,K)

        noise = torch.as_tensor(NOISE_POWER, dtype=signal.dtype, device=device)
        sinr = signal / (interf_comm + interf_sense + noise)
        return sinr

    # -- 速率（底 2；bits/s/Hz）
    def compute_rates(self, sinrs: torch.Tensor) -> torch.Tensor:
        # sinrs: (B,K) -> rates: (B,K)
        return torch.log1p(sinrs) / np.log(2.0)

    # -- 感測 SNR
    def compute_sense_snr(self,
                        g_dt: torch.Tensor,   # (B,M,1)
                        W_S: torch.Tensor,    # (B,M,1)
                        W_C: torch.Tensor,    # (B,M,K)
                        pl_BS_TAR_BS          # scalar power
                        ) -> torch.Tensor:    # (B,)

        # 0) type/device align
        device, dtype = W_C.device, W_C.dtype
        g_dt, W_S, W_C = [t.to(device=device, dtype=dtype) for t in (g_dt, W_S, W_C)]
        eps = 1e-12

        # 1) 等效通道 v_t = g_dt g_dt^H
        #    (B,M,1) -> (B,M,M)
        v_t = g_dt @ g_dt.conj().transpose(1, 2)  # (B,M,M)

        # 2) Q = W_C W_C^H + W_S W_S^H  (發射端能量/協方差)
        Q_comm  = W_C @ W_C.conj().transpose(1, 2)  # (B,M,M)
        Q_sense = W_S @ W_S.conj().transpose(1, 2)  # (B,M,M)
        Q = Q_comm + Q_sense                        # (B,M,M)

        # 3) A = v_t Q v_t^H   (理論上 Hermitian PSD)
        A = v_t @ Q @ v_t.conj().transpose(1, 2)    # (B,M,M)

        # 4) SNR_max = (pl / noise) * lambda_max(A)
        #    用 eigvalsh 只取 eigenvalues，可反傳，避開 complex eigenvector backward 問題
        eigvals = torch.linalg.eigvalsh(A)          # (B,M) real, ascending
        lambda_max = eigvals[:, -1].clamp_min(0.0)  # (B,) 避免數值誤差出現極小負值

        pl = torch.as_tensor(pl_BS_TAR_BS, device=device, dtype=lambda_max.dtype)
        noise = torch.as_tensor(NOISE_POWER, device=device, dtype=lambda_max.dtype)

        snr_max = (pl / noise) * lambda_max         # (B,)
        return snr_max

    # -- 模型存取
    def load_model(self, tag: str = "", verbose: bool = False):
        """
        讀取 checkpoint。
        - verbose=False: 預設不印出「尚無模型 / 已載入模型」等訊息，避免訓練 log 雜亂
        - verbose=True : 需要除錯時才開啟
        """
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.load_state_dict(torch.load(self.model_path, map_location=DEVICE), strict=True)
                if verbose and tag:
                    print(f"[{tag}] 已載入模型：{self.model_path}")
            except Exception as e:
                if verbose and tag:
                    print(f"[{tag}] 既有 checkpoint 與網路結構不相容，將從隨機初始化開始。原因：{e}")
        else:
            if verbose and tag:
                print(f"[{tag}] 尚無已訓練模型，從隨機初始化開始。")

    def save_model(self):
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.state_dict(), self.model_path)
            print(f"[Model saved] {self.model_path}")

# ------------------------------
# 1) 通訊 Tx beamformer W_comm ∈ ℂ^{M×K}
# ------------------------------
class CommBeamformerNet(BaseMLP):
    def __init__(self):
        out_dim = 2 * (TX_ANT*UAV_COMM)
        super().__init__(out_dim)
        self.model_path = build_ckpt_path("comm", robust=False)
        self.load_model(tag="Comm", verbose=VERBOSE_LOAD)
        
    def forward(self, h_dk, h_rk, G, g_dt):
        x = self.encode_inputs(h_dk, h_rk, G, g_dt)             # (B, in_dim)
        y = self.forward_mlp(x)                                 # (B, out_dim)
        W_C = self.decode(y, (TX_ANT, UAV_COMM))                # (B,M,K) complex
        return W_C

# ------------------------------
# 2) 感測 Tx beamformer W_sense ∈ ℂ^{M×1}
# ------------------------------

class SenseBeamformerNet(BaseMLP):
    def __init__(self):
        out_dim = 2 * (TX_ANT)
        super().__init__(out_dim)
        self.model_path = build_ckpt_path("sense", robust=False)
        self.load_model(tag="Sens", verbose=VERBOSE_LOAD)

    def forward(self, h_dk, h_rk, G, g_dt):
        x = self.encode_inputs(h_dk, h_rk, G, g_dt)  # (B, in_dim)
        y = self.forward_mlp(x)                      # (B, out_dim)
        W_S = self.decode(y, (TX_ANT, 1))            # (B,M,1)
        return W_S

# ------------------------------
# 3) RIS 反射矩陣 Φ = diag(φ),  φ ∈ ℂ^{N}, |φ_n|=1
# ------------------------------
class RISPhaseNet(BaseMLP):
    def __init__(self):
        out_dim = 2 * (RIS_UNIT)
        super().__init__(out_dim)
        self.model_path = build_ckpt_path("ris", robust=False)
        self.load_model(tag="Ris", verbose=VERBOSE_LOAD)

    def forward(self, h_dk, h_rk, G, g_dt):
        x = self.encode_inputs(h_dk, h_rk, G, g_dt)  # (B, in_dim)
        y = self.forward_mlp(x)                            # (B, out_dim)
        phi = self.decode(y, (RIS_UNIT,))                  # (B,N) complex
        return phi                                         # (B,N)(符合大小)

# 魯棒模型

# ------------------------------
# 4) 通訊 Tx beamformer W_comm ∈ ℂ^{M×K}
# ------------------------------
class RobustCommBeamformerNet(CommBeamformerNet):
    def __init__(self):
        out_dim = 2 * (TX_ANT * UAV_COMM)
        BaseMLP.__init__(self, out_dim)  # avoid loading regular ckpt in parent __init__
        self.model_path = build_ckpt_path("comm", robust=True)
        self.load_model(tag="Comm-Robust", verbose=VERBOSE_LOAD)

# ------------------------------
# 5) 感測 Tx beamformer W_sense ∈ ℂ^{M×1}
# ------------------------------
class RobustSenseBeamformerNet(SenseBeamformerNet):
    def __init__(self):
        out_dim = 2 * (TX_ANT)
        BaseMLP.__init__(self, out_dim)  # avoid loading regular ckpt in parent __init__
        self.model_path = build_ckpt_path("sense", robust=True)
        self.load_model(tag="Sens-Robust", verbose=VERBOSE_LOAD)

# ------------------------------
# 6) RIS 反射矩陣 Φ = diag(φ),  φ ∈ ℂ^{N}, |φ_n|=1
# ------------------------------
class RobustRISPhaseNet(RISPhaseNet):
    def __init__(self):
        out_dim = 2 * (RIS_UNIT)
        BaseMLP.__init__(self, out_dim)  # avoid loading regular ckpt in parent __init__
        self.model_path = build_ckpt_path("ris", robust=True)
        self.load_model(tag="Ris-Robust", verbose=VERBOSE_LOAD)