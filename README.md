# Uncertainty Injection for Two-Timescale RIS-Assisted ISAC

本專案研究**被動式 RIS 輔助的雙時間尺度整合感測與通訊（ISAC）**。
一座多天線基地台服務多位單天線通訊用戶，同時用同一組雙功能波束感測一個固定目標，預設配置為 4 根發射天線、64 個 RIS 元件及 2 位通訊用戶。

## 專案結構
.
├── settings.py               # 系統、資料、訓練與評估設定
├── trajectory_gen.py         # 產生 UE 軌跡與 layout_*.npz
├── channel_gen.py            # 產生四條鏈路的通道、路徑增益及 dataset_*.npz
├── two_timescale_NN.py       # RISNet、JointNet、等效通道與 ISAC 效能計算
├── joint_reg.py              # 訓練 REG 與輸出權重、曲線
├── joint_rob.py              # 訓練 ROB、共用誤差注入函式與輸出權重、曲線
├── AO.py                     # 有界誤差下的離線 AO 求解與分批存檔
└── evaluate.py               # 載入 REG、ROB 與 AO6 決策，統一評估


## 環境設定

使用 Python 3.11。在放置上述 8 支程式的目錄建立本地虛擬環境，並從該目錄執行下方指令，讓 `settings.py` 的相對資料路徑解析到正確位置。

Windows (PowerShell)

## 參考文獻

- Wei Cui and Wei Yu. *Uncertainty Injection: A Deep Learning Method for Robust Optimization*. [論文](https://arxiv.org/abs/2302.12304)