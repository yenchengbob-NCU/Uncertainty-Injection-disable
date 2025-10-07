# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *  

# ------------------------------
# L2 正規化工具
# ------------------------------
def _normalize_columns(B: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    對複數張量 B 逐欄(沿天線維度 M)做 L2 正規化：
      B: (N, M, K) complex  ->  回傳 (N, M, K)，每欄 ||b_k||_2 = 1
    """
    # dim=1 對應你的 M 維；vector_norm 對 complex 安全
    norms = torch.linalg.vector_norm(B, dim=1, keepdim=True)
    norms = torch.clamp(norms, min=eps)
    return B / norms

# ------------------------------
# 角度→Torch ULA 方向向量 a(θ)（不除 sqrt(M)）
# ------------------------------
def torch_ula_steering_vector(theta_deg: torch.Tensor, M_: int = M, d_lambda: float = D_SPACING) -> torch.Tensor:
    """
    theta_deg: (batch,) 角度（度）
    回傳 shape: (batch, M_, 1) 複數
    """
    theta = torch.deg2rad(theta_deg.to(torch.float32))
    m = torch.arange(M_, device=theta.device, dtype=torch.float32).view(1, M_, 1)
    phase = 2.0 * np.pi * d_lambda * m * torch.sin(theta).view(-1, 1, 1)
    a = torch.exp(1j * phase)                                           # ★ 不除以 sqrt(M)
    return a.to(torch.complex64)

# ------------------------------
# 基底網路：輸入 H_est -> 輸出波束 B（複數）
# ------------------------------
class Neural_Net(nn.Module):
    def __init__(self, hidden: int = 200, depth: int = 4):
        """
        對標:4 層隱藏、每層 200、ReLU
        in_dim = 2*M*K, out_dim = 2*M*K
        """
        super().__init__()
        self.model_path = ""    # 子類會賦值
        self.n_hidden = hidden  # 每層有200個神經元
        self.depth    = depth   # 一共建立4層隱藏層

        in_dim  = 2 * M * K     # 通道展平 實部&虛部
        out_dim = 2 * M * K
        # sizes 是個list，定義每一層的神經元數量
        # 在這裡[2*M*K, 200, 200, 200, 200, 2*M*K]
        sizes = [in_dim] + [hidden] * depth + [out_dim]
        #一共建立6層
        self.layers = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])

        # 權重初始化（Kaiming for ReLU）
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

    # -- 編碼輸入：H_est (N,M,K, complex) -> (N, 2*M*K) 把複數張量轉實數 作為NN input
    def _encode_input(self, H_est: torch.Tensor) -> torch.Tensor:
        assert H_est.ndim == 3 and H_est.shape[1] == M and H_est.shape[2] == K, "H_est shape 應為 (N,M,K)"
        x = torch.view_as_real(H_est).reshape(H_est.shape[0], -1)  # (N, 2*M*K)
        return x.to(torch.float32)
    
    # -- 將輸出重塑為複數波束 B  把實數張量(NN 輸出)轉複數 B
    def _decode_beam(self, y: torch.Tensor) -> torch.Tensor:
        N = y.shape[0]
        y = y.view(N, M, K, 2)                                    # (N,M,K,2)
        B = torch.view_as_complex(y.contiguous())                 # (N,M,K) complex
        return B

    # -- NN計算  input -> NN -> output
    def get_beamformer(self, H_est: torch.Tensor) -> torch.Tensor:
        x = self._encode_input(H_est)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        y = self.layers[-1](x)                     # (N, 2*M*K)
        B = self._decode_beam(y)                   # (N, M, K)
        return B

    # -- 通訊 SINR 計算（不含功率分配；呼叫前請先把 B_dir -> 等功率後的 W）
    def compute_comm_sinrs(self, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        參數：
            H, W: shape=(N, M, K) 的複數張量
                - W 應該已經是「方向單位化後，再做等功率分配」的最終 precoder
        回傳：
            SINR: shape=(N, K)，對每個使用者的通訊 SINR
        說明：
            S = H^H W  =>  |S[n,k,j]|^2 是第 n 筆樣本、使用者 k 收到第 j 條流的能量
            signal = |S[n,k,k]|^2
            interf = Σ_{j≠k} |S[n,k,j]|^2
            SINR = signal / (interf + NOISE_POWER)
        """
        assert H.shape == W.shape and H.ndim == 3, "H, W 必須同形狀 (N,M,K)"
        # H^H W -> (N,K,K)
        S = torch.matmul(torch.conj(H).transpose(1, 2), W)
        P = (S.abs()) ** 2                                # (N,K,K), 實數
        signal = torch.diagonal(P, dim1=1, dim2=2)        # (N,K)
        interf = P.sum(dim=2) - signal                    # (N,K)
        noise = torch.as_tensor(NOISE_POWER, dtype=signal.dtype, device=signal.device)
        sinr = signal / (interf + noise)
        return sinr

    # -- 速率（底 2；bits/s/Hz） 
    def compute_rates(self, sinrs: torch.Tensor) -> torch.Tensor:
        return torch.log1p(sinrs) / np.log(2.0)   # 使用 log1p 提升小 SINR 穩定性

    # -- 感測 SNR（單一 θ；或批次 θ_b）
    def compute_sense_snr(self, W: torch.Tensor, theta_deg: torch.Tensor = None) -> torch.Tensor:
        """
        參數：
            W : (N, M, K) complex   # 已完成方向正規化與等功率分配的最終 precoder
            theta_deg : (N,) 角度（度）；若 None 則使用常數 THETA_DEG
        回傳：
            SNR_sense (N,)（線性值）
        公式：
            tx_gain(n) = sum_j |a(θ_n)^H w_{n,j}|^2
            P_tx(n)    = ||W_n||_F^2
            SNR_sense  = |a|^2 * tx_gain / ( N0 + κ * P_tx )
        """
        Nbatch = W.shape[0]
        if theta_deg is None:
            theta_deg = torch.full((Nbatch,), float(THETA_DEG), device=W.device, dtype=torch.float32)

        # a(θ): (N, M, 1)
        a = torch_ula_steering_vector(theta_deg, M_=M, d_lambda=D_SPACING).to(W.device)

        # a^H W : (N, 1, K)
        AW = torch.matmul(torch.conj(torch.transpose(a, 1, 2)), W)

        # 逐流能量並加總：tx_gain(n) = Σ_j |a^H w_j|^2  -> (N,)
        tx_gain = (AW.abs() ** 2).sum(dim=2).squeeze(1)

        # 實際發射功率 P_tx(n) = ||W_n||_F^2  -> (N,)
        P_tx = torch.linalg.vector_norm(W, dim=(1, 2)) ** 2

        denom = torch.as_tensor(NOISE_POWER, dtype=tx_gain.dtype, device=tx_gain.device) + \
                torch.as_tensor(SELF_INTERFERENCE_KAPPA, dtype=tx_gain.dtype, device=tx_gain.device) * P_tx

        snr = ALPHA_MEAN_POWER * tx_gain / denom
        return snr

    # -- 模型存取
    def load_model(self, tag: str = ""):
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.load_state_dict(torch.load(self.model_path, map_location=DEVICE), strict=True)
                if tag:
                    print(f"[{tag}] 已載入模型：{self.model_path}")
            except Exception as e:
                if tag:
                    print(f"[{tag}] 既有 checkpoint 與網路結構不相容，將從隨機初始化開始。原因：{e}")
        else:
            if tag:
                print(f"[{tag}] 尚無已訓練模型，從隨機初始化開始。")

    def save_model(self):
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            torch.save(self.state_dict(), self.model_path)
            print(f"[Model saved] {self.model_path}")

# ------------------------------
# Regular_Net：最小用戶率 + 感測門檻懲罰
# ------------------------------
class Regular_Net(Neural_Net):
    def __init__(self, hidden: int = 200, depth: int = 4):
        # 結構沿用父類（[2*M*K, 200, 200, 200, 200, 2*M*K]）
        super().__init__(hidden=hidden, depth=depth)
        # 預設 checkpoint 路徑（可在外部覆寫 self.model_path）
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Trained_Models_ISAC",
            f"regular_net_{SETTING_STRING}.ckpt"
        )
        # 與舊寫法一致：建構時嘗試載入既有模型（若不存在就略過）
        self.load_model("Regular Net")

    def forward(self, H_est: torch.Tensor):
        # 1) NN → raw B
        B_raw = self.get_beamformer(H_est)          # (N,M,K)

        # 2) 欄向量 L2 正規化（方向）
        B_dir = _normalize_columns(B_raw)           # (N,M,K)

        # 3) 等功率分配（拿到最終 precoder）
        W = apply_power_allocation_torch(B_dir)     # (N,M,K)

        # 4) 通訊：SINR → rate → min over user
        sinrs = self.compute_comm_sinrs(H_est, W)   # (N,K)
        rates = self.compute_rates(sinrs)           # (N,K)
        min_rates = torch.min(rates, dim=1).values  # (N,)

        # 5) 感測：用 precoder W
        sense_snr = self.compute_sense_snr(W)       # (N,)

        # 6) 目標：f_reg = min_rate - λ·max(0, Γ - SNR_sense)
        penalty = F.relu(torch.as_tensor(SENSING_SNR_THRESHOLD, device=sense_snr.device, dtype=sense_snr.dtype) - sense_snr)
        objectives = min_rates - SENSING_LOSS_WEIGHT * penalty

        # 與 MIMO 一樣：回傳 batch 平均目標 & B（方向）
        return objectives.mean(), B_dir

# ------------------------------
# Robust_Net：通道注入 + 分位最小率 + 感測門檻懲罰
# ------------------------------
class Robust_Net(Neural_Net):
    
    def __init__(self, hidden: int = 200, depth: int = 4):
        # 結構沿用父類（[2*M*K, 200, 200, 200, 200, 2*M*K]）
        super().__init__(hidden=hidden, depth=depth)
        self.L = INJECTION_SAMPLES      #1000
        self.q = OUTAGE_QUANTILE        #5%
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Trained_Models_ISAC",
            f"robust_net_{SETTING_STRING}.ckpt"
        )
        # 與舊寫法一致：建構時嘗試載入既有模型（若不存在就略過）
        self.load_model("Robust Net")

    # 通道不確定性注入：H_realize = H_est + e, e~CN(0, σ_e^2 I)
    def inject_uncertainties(self, H_est: torch.Tensor) -> torch.Tensor:
        N = H_est.shape[0]
        sigma2 = INJECTION_VARIANCE
        re = torch.randn((self.L, N, M, K), device=H_est.device) * np.sqrt(sigma2/2)
        im = torch.randn((self.L, N, M, K), device=H_est.device) * np.sqrt(sigma2/2)
        e = torch.complex(re, im)
        H_realize = (H_est.unsqueeze(0) + e).reshape(self.L * N, M, K)  # (L*N,M,K)
        return H_realize

    def forward(self, H_est: torch.Tensor):
        """
        輸入:H_est (N,M,K) complex
        輸出:robust_objective.mean(), B_dir
        目標（分位健壯）：  f = Q_q[min_k R_k] - λ * max(0, Γ - Q_q[SNR_sense])
        其中 Q_q 表示在 L 個擾動上的 q-分位數。
        """
        N = H_est.shape[0]

        # 1) NN → raw B
        B_raw = self.get_beamformer(H_est)          # (N,M,K)

        # 2) 欄向量 L2 正規化（方向）
        B_dir = _normalize_columns(B_raw)           # (N,M,K)

        # 3) 等功率分配（拿到最終 precoder）
        W     = apply_power_allocation_torch(B_dir) # (N,M,K)

        # 4) 產生 L 組擾動通道（與估測誤差脫鉤）
        H_L = self.inject_uncertainties(H_est)                      # (L*N,M,K)
        W_L = W.repeat(self.L, 1, 1)                                # (L*N,M,K)

        # 5) 通訊：SINR → rate → min over user
        sinrs_L = self.compute_comm_sinrs(H_L, W_L)                 # (L*N,K)
        rates_L = self.compute_rates(sinrs_L).view(self.L, N, K)    # (L,N,K)
        min_rates_L = torch.min(rates_L, dim=2).values              # (L,N)

        # 6) 感測：用 precoder W
        sense_L = self.compute_sense_snr(W_L).view(self.L, N)       # (L,N)

        # 7) 取 q 分位（沿 L 維度）
        q_min_rate = torch.quantile(min_rates_L, q=self.q, dim=0)  # (N,)
        q_sense    = torch.quantile(sense_L,     q=self.q, dim=0)  # (N,)

        # 8) 目標：f = q_min_rate - λ * max(0, Γ - q_sense)
        thr = torch.as_tensor(SENSING_SNR_THRESHOLD, dtype=q_sense.dtype, device=q_sense.device)
        penalty = F.relu(thr - q_sense)                             # (N,)
        objectives = q_min_rate - SENSING_LOSS_WEIGHT * penalty     # (N,)

        return objectives.mean(), B_dir

"""
專案:單BS easy ISAC comm center Version 3
檔名:neural_net.py

說明：
- 目標:在單基地台M=4、單用戶K=1、單目標感測的場景下,學習通訊/感測共用的發射波束，
      最大化「最小用戶速率」(K=1 即單用戶速率)，並以懲罰方式滿足感測 SNR 門檻。
- 網路結構: MLP層數 depth=4、每層 hidden=200;輸入/輸出維度均為 2*M*K(複數通道以 Re/Im 展平)。
- 波束設計：
  (1) get_beamformer() 產生 raw 複數波束 B_raw
  (2) _normalize_columns() 將欄向量 L2 正規化 → B_dir(僅方向)
  (3) settings.apply_power_allocation_torch(B_dir) → 等功率後的最終 precoder W
- 指標計算：
  - 通訊 SINR:S = H^H W,signal=|S[k,k]|^2,interf=Σ_{j≠k}|S[k,j]|^2,SINR=signal/(interf+NOISE_POWER)
  - 速率:log2(1+SINR)
  - 感測 SNR:以 ULA 方向向量 a(θ) 計算 tx_gain=Σ_j|a^H w_j|^2,並考慮殘留自干擾 κ 與雜訊 N0
- Regular_Net:
  - 目標 f_reg = min_user_rate - λ·max(0, Γ - SNR_sense)
  - forward() 回傳 (f_reg 的 batch mean, B_dir)
- Robust_Net:
  - 不確定性注入與通道估測誤差脫鉤:H_realize = H_est + e,e ~ CN(0, INJECTION_VARIANCE·I)
  - 在 L 組擾動上計算 min_user_rate 與 SNR_sense,取 q 分位數做健壯目標：
    f_robust = Q_q[min_user_rate] - λ·max(0, Γ - Q_q[SNR_sense])
- 訓練/評估：
  - 訓練/驗證：每個 epoch 線上生成通道；測試：僅離線存通道(不綁定特定 beamformer)
  - 等功率分配在 settings.py 提供 NumPy / Torch 兩版，確保 baseline 與 NN 一致
- 檢查點(checkpoint):
  - 預設儲存於 Trained_Models_ISAC/regular_net_{SETTING_STRING}.ckpt 或 robust_net_{SETTING_STRING}.ckpt
  - 可用 load_model()/save_model() 載入/儲存 state_dict

備註：
- 務必遵守順序:B_raw →(欄向量 L2)B_dir →(等功率)W → 計算通訊/感測指標。
- 若後續啟用角度抖動，請於 Robust_Net 中加入 θ 的擾動產生與批次計算。
"""