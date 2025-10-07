Uncertainty Injection — ISAC/MIMO

以「不確定性注入 (Uncertainty Injection)」訓練法，實作單 BS、單用戶、單目標的 ISAC 場景（含 MIMO 版本）。目標在滿足感測 SNR 門檻的前提下，最大化最小用戶速率；並比較 Regular_Net 與 Robust_Net。

📦 專案結構（簡述）
.
├─ ISAC/
│  ├─ main.py                 # 訓練（Regular vs Robust）
│  ├─ evaluate.py             # 離線評估（q-quantile min-rate）
│  ├─ neural_net.py           # NN + 目標函數 + SINR/感測SNR
│  ├─ networks_generator.py   # 產生測試通道/估測通道
│  └─ settings.py             # 場景與訓練參數
├─ MIMO/
│  ├─ main_MIMO.py, evaluate_MIMO.py, neural_net_MIMO.py, ...
└─ papers/
   ├─ Uncertainty Injection A.pdf
   └─ 2504.19091v1.pdf


🔒 Git 已忽略：ml311/（虛擬環境）、Data*/、checkpoints/、Trained_Models_*、eval_plots/、各種 .npy/.ckpt/.pt 等大型檔案（見 .gitignore）。

🧪 環境（venv）說明【務必在專案資料夾內建立 venv】

團隊約定：每位成員都要在專案根目錄建立並使用本地虛擬環境 ml311；不要用系統 Python。

Windows (PowerShell)
# 1) 建立與啟用虛擬環境（專案根目錄下）
python -m venv ml311
.\ml311\Scripts\activate

# 2) 升級 pip
python -m pip install --upgrade pip

# 3) 安裝相依套件（先安裝 PyTorch，再安裝其餘）
#    請到 PyTorch 官網照你的 CUDA/平台選指令安裝 torch/torchvision/torchaudio
#    例如（CUDA 12.4 範例，請依官網為準）：
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4) 其餘通用套件
pip install numpy matplotlib tqdm

常見問答

為什麼不把 venv 傳到 Git？
venv 依平台不同且檔案龐大，應由每人本地建立；相依版本請寫在 README 或 requirements。

行尾 LF/CRLF 警告？
建議各自設定一次：git config core.autocrlf true。

🚀 快速開始（ISAC）

產生測試通道（會輸出到 Data/，已被 .gitignore 排除）：

(ml311) python ISAC/networks_generator.py


訓練（Regular vs Robust）：

(ml311) python ISAC/main.py


繪製訓練曲線（僅畫圖不訓練）：

(ml311) python ISAC/main.py --plot


評估（同一組擾動通道上比較 q-quantile min-rate）：

(ml311) python ISAC/evaluate.py
# 可選參數：
#   --channels <path/to/channelEstimates_test_*.npy>
#   --L <int>   # 擾動樣本數（預設等於 Robust_Net.L）


檢查點（checkpoints）與訓練曲線會儲存在 Trained_Models_ISAC/，已被 .gitignore 排除。

🧷 重要參數（部分）

ISAC/settings.py

M=4, K=1, N_TARGET=1

BANDWIDTH, NOISE_POWER

估測與注入：ESTIMATION_ERROR_VARIANCE、INJECTION_VARIANCE、INJECTION_SAMPLES (L)、OUTAGE_QUANTILE (q)

感測約束：SENSING_SNR_THRESHOLD_dB、SENSING_LOSS_WEIGHT

功率：TRANSMIT_POWER_TOTAL（等功率分配）

🧰 Git 使用注意

首次初始化後請確認 .gitignore 已存在且包含：

ml311/、Data*/、checkpoints/、Trained_Models_*、eval_plots/、*.npy/*.ckpt/*.pt

若誤把 venv 或資料加入追蹤：
git rm -r --cached ml311 Data Data_MIMO checkpoints Trained_Models_ISAC Trained_Models_MIMO eval_plots

📌 版本與平台（範例，供參考）

Python 3.11.x（團隊內以 3.11.5 測試）

Windows 10/11（RTX 4080S 測試）

PyTorch：請依各自 CUDA/驅動版本安裝對應輪檔（torch 版本需與 CUDA 對應）

📜 引用

Cui & Wei Yu, Uncertainty Injection: A Deep Learning Method for Robust Optimization.

相關 ISAC/MIMO 參考文件見 papers/。

有任何環境安裝或訓練問題，先確認是否已啟用 ml311（提示符應該有 (ml311)），以及你未將 venv/資料推上 Git。
