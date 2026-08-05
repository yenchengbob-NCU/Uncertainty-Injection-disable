# -*- coding: utf-8 -*-
import os
import torch

from settings import *
from baseline import make_rzf_beamformer, mrt_in_H_eff_H_nullspace, beamformers_power_split, RZF_LAMBDA
from two_timescale_NN import ThetaNet





if __name__ == "__main__":
    # ============================================================
    # Paths
    # ============================================================
    test_dataset_path = os.path.join(DATA_DIR,"dataset_test.npz")
    ris_only_ckpt     = os.path.join(PRETRAIN_DIR,"ris_only.ckpt")

    for required_path in [test_dataset_path,ris_only_ckpt]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"找不到檔案: {required_path}")

    # ============================================================
    # Load trained ThetaNet and test dataset
    # ============================================================
    theta_net = ThetaNet().to(DEVICE)
    theta_net.load_model(ris_only_ckpt,strict=True,verbose=True)
    theta_net.eval()

    test_dataset = theta_net.load_channel_dataset(test_dataset_path,"test")

    with torch.no_grad():
        h_dk_test   = torch.as_tensor(test_dataset["h_dk_hat"],dtype=torch.complex64,device=DEVICE)   # (B,M,K)
        h_rk_test   = torch.as_tensor(test_dataset["h_rk_hat"],dtype=torch.complex64,device=DEVICE)   # (B,N,K)
        G_test      = torch.as_tensor(test_dataset["G_hat"],dtype=torch.complex64,device=DEVICE)      # (B,N,M)
        g_dt_test   = torch.as_tensor(test_dataset["g_dt_hat"],dtype=torch.complex64,device=DEVICE)   # (B,M,1)

        B = h_dk_test.shape[0]

        # 每筆 test estimated channel 由 trained ThetaNet 輸出 RIS phase
        theta = theta_net(h_dk_test,h_rk_test,G_test,g_dt_test)                                      # (B,N)

        # Learned RIS 對應的 nominal effective channel
        H_eff_H = theta_net.compute_effective_channel(h_dk_test,h_rk_test,G_test,theta)              # (B,K,M)

        # RZF communication beam direction and NS-MRT sensing beam direction
        W_C_dir = make_rzf_beamformer(H_eff_H,RZF_LAMBDA)                                            # (B,M,K)
        W_R_dir = mrt_in_H_eff_H_nullspace(H_eff_H,g_dt_test)                                        # (B,M,1)

        # Fixed power split: W_C / W_R = 0.4 / 0.6
        W_C,W_R = beamformers_power_split(W_C_dir,W_R_dir)

        # Nominal test performance
        metrics = theta_net.compute_isac_batch_performance(H_eff_H,g_dt_test,W_C,W_R)

        # Nominal objective = mean_B(min_K(rate))
        nominal_allUE_rate = metrics["rate"]                                                         # (B,K)
        nominal_per_channel = torch.min(nominal_allUE_rate,dim=1).values                             # (B,)
        nominal_objective = torch.mean(nominal_per_channel)                                          # scalar

        # Target SNR = 10log10(mean_B(linear target SNR))
        nominal_target_snr = metrics["target_snr"]                                                   # (B,), linear
        target_snr_mean = torch.mean(nominal_target_snr)                                             # scalar, linear
        target_snr_mean_db = 10.0 * torch.log10(target_snr_mean.clamp_min(1e-12))

    print("=" * 90)
    print("[Trained ThetaNet + RZF + NS-MRT Nominal Test]")
    print("=" * 90)
    print(f"Test dataset              : {test_dataset_path}")
    print(f"ThetaNet checkpoint       : {ris_only_ckpt}")
    print(f"Test estimated channels B : {B}")
    print(f"RZF lambda                : {RZF_LAMBDA:g}")
    print("-" * 90)
    print(f"Nominal objective         : {float(nominal_objective.detach().cpu()):.6f} bps/Hz")
    print(f"Target SNR                : {float(target_snr_mean_db.detach().cpu()):.3f} dB")
    print("=" * 90)
