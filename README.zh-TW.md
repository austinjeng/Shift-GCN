🌐 語言: [English](README.md) | **繁體中文**

# MediaPipe Shift-GCN：基於骨架的跌倒偵測

以 [Shift-GCN](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.pdf)（CVPR 2020）為基礎的骨架式跌倒偵測系統。使用 [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)（33 個關節點）取代深度感測器骨架，實現從一般 RGB 影片進行跌倒偵測。

在 [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) 資料集上訓練四模型集成（joint、bone、joint motion、bone motion），並可透過 CLI 或 Tkinter GUI 對任意影片進行推論。

```
                         ┌──────────────────┐
                         │   RGB Video      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  MediaPipe Pose   │
                         │  (33 landmarks)   │
                         └────────┬─────────┘
                                  │
              ┌───────────┬───────┴───────┬────────────┐
              ▼           ▼               ▼            ▼
         ┌─────────┐ ┌─────────┐ ┌─────────────┐ ┌──────────────┐
         │  Joint  │ │  Bone   │ │Joint Motion │ │ Bone Motion  │
         └────┬────┘ └────┬────┘ └──────┬──────┘ └──────┬───────┘
              │           │             │               │
         ┌────▼────┐ ┌────▼────┐ ┌──────▼──────┐ ┌──────▼───────┐
         │Shift-GCN│ │Shift-GCN│ │  Shift-GCN  │ │  Shift-GCN   │
         │ (×0.6)  │ │ (×0.6)  │ │   (×0.4)    │ │   (×0.4)     │
         └────┬────┘ └────┬────┘ └──────┬──────┘ └──────┬───────┘
              │           │             │               │
              └───────────┴───────┬─────┴───────────────┘
                                  │
                         ┌────────▼─────────┐
                         │ Weighted Ensemble │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Fall / Non-Fall  │
                         └──────────────────┘
```

## 實驗結果

使用 MediaPipe Pose 關節點在 NTU RGB+D（cross-subject 分割）上訓練。

| 模型 | Top-1 準確率 |
|------|-------------|
| Joint | 99.49% |
| Bone | 99.51% |
| Joint Motion | 99.46% |
| Bone Motion | 99.64% |
| **集成（Ensemble）** | **99.77%** |

**混淆矩陣（Confusion Matrix）—— 集成模型**

|  | 預測：非跌倒 | 預測：跌倒 |
|--|-------------|-----------|
| **實際：非跌倒** | 16,273 | 11 |
| **實際：跌倒** | 27 | 249 |

跌倒 F1 分數：**92.91%**（精確率 95.77%、召回率 90.22%）。

資料集：2,688 筆訓練樣本（跌倒:非跌倒 = 1:3）/ 16,560 筆驗證樣本（自然分佈）。詳細結果請參閱 [TRAINING_REPORT.md](TRAINING_REPORT.md)。

## 環境需求

- Python 3.10+
- 支援 CUDA 的 PyTorch
- CUDA Toolkit 12.x（或相容版本）
- C++ 編譯器（Linux 使用 g++，Windows 使用 Visual Studio）
- MediaPipe、OpenCV、NumPy、scikit-learn

### 環境設定

```bash
conda create -n shiftgcn python=3.12
conda activate shiftgcn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mediapipe opencv-python numpy scikit-learn
```

請根據你的 CUDA 版本調整 PyTorch 的 index URL。選項請參閱 [pytorch.org](https://pytorch.org/get-started/locally/)。

## 編譯 CUDA 擴充模組

位移操作（Shift operation）使用自訂 CUDA 核心，必須在訓練或推論前編譯。

### Linux

```bash
cd model/Temporal_shift/cuda
python setup.py install
```

### Windows

需要安裝 Visual Studio 2022（或 2019），並啟用 **「使用 C++ 的桌面開發」** 工作負載。

```bat
:: 清除舊的環境變數，然後啟動 VS 建置工具
set "VSINSTALLDIR="
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: 編譯擴充模組
set DISTUTILS_USE_SDK=1
cd model\Temporal_shift\cuda
python setup.py install
```

在 conda 的啟動腳本中加入以下設定，以避免 OpenMP 衝突：

```bat
set KMP_DUPLICATE_LIB_OK=TRUE
```

### 驗證編譯結果

```bash
python -c "from cuda.shift import Shift; print('CUDA extension OK')"
```

## 資料準備

準備訓練資料分為三個步驟。所有腳本位於 `data_gen/`。

### 步驟一：擷取 MediaPipe 關節點

下載 [NTU RGB+D](https://rose1.ntu.edu.sg/dataset/actionRecognition/) RGB 影片，並擷取 33 關節點骨架：

```bash
python data_gen/mediapipe_gendata.py \
  --video_dir /path/to/nturgb+d_rgb \
  --out_dir ./data/mediapipe/ \
  --ntu_mode \
  --subsample_ratio 3 \
  --benchmark xsub
```

| 參數 | 用途 |
|------|------|
| `--ntu_mode` | 解析 NTU 檔名以判斷 subject/action 分割 |
| `--subsample_ratio 3` | 平衡訓練集，跌倒:非跌倒 = 1:3 |
| `--benchmark xsub` | 使用 cross-subject 評估協定 |

此步驟產生 `train_data_joint.npy`（305 MB）、`val_data_joint.npy`（1.9 GB）及標籤檔案。

### 步驟二：生成骨骼資料（Bone Data）

```bash
python data_gen/gen_bone_data_mediapipe.py
```

沿骨架樹計算關節到父節點的差分向量（32 根骨骼）。

### 步驟三：生成動態資料（Motion Data）

```bash
python data_gen/gen_motion_data_mediapipe.py
```

計算 joint 與 bone 兩種模態的逐幀差分。

完成三個步驟後，`./data/mediapipe/` 將包含 8 個 `.npy` 資料檔 + 2 個 `.pkl` 標籤檔（共約 8.6 GB）。

## 訓練

訓練四個模型 —— 每種模態各一個。每個模型在單顆 GPU 上約需 2 小時。

```bash
python main.py --config ./config/mediapipe/train_joint.yaml
python main.py --config ./config/mediapipe/train_bone.yaml
python main.py --config ./config/mediapipe/train_joint_motion.yaml
python main.py --config ./config/mediapipe/train_bone_motion.yaml
```

四個設定檔使用相同的超參數（Hyperparameters）：

| 參數 | 值 |
|------|-----|
| 訓練輪數（Epochs） | 140 |
| 批次大小（Batch size） | 64 |
| 優化器（Optimizer） | SGD + Nesterov momentum |
| 學習率（Learning rate） | 0.1（在第 60、80、100 輪以 ×0.1 階梯衰減） |
| 權重衰減（Weight decay） | 0.0001 |

模型檢查點（Checkpoint）每 2 個訓練輪儲存至 `./save_models/`。

### 從檢查點恢復訓練

如果訓練中斷，可從最新的檢查點恢復：

```bash
python main.py --config ./config/mediapipe/train_joint.yaml \
  --resume ./save_models/mediapipe_ShiftGCN_joint-60-2520.pt
```

### 覆寫先前的訓練紀錄

如需清除舊的檢查點並重新開始：

```bash
python main.py --config ./config/mediapipe/train_joint.yaml --overwrite True
```

## 集成評估（Ensemble Evaluation）

訓練完四個模型後，評估加權集成的表現：

```bash
python ensemble_mediapipe.py
```

此腳本將各模態最佳檢查點的預測結果以權重 `[0.6, 0.6, 0.4, 0.4]`（joint、bone、joint motion、bone motion）進行組合。

## 推論（Inference）

對任意 RGB 影片進行跌倒偵測。推論管線（Pipeline）會擷取 MediaPipe 關節點、透過四模型集成在滑動視窗上執行推論，並產生帶標註的輸出影片與 JSON 報告。

### CLI 模式

```bash
python inference_pipeline.py --cli --video /path/to/video.mp4
```

### GUI 模式

```bash
python inference_pipeline.py
```

開啟 Tkinter 視窗，可選擇影片檔案並檢視結果。

### 選項

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--video` | — | 輸入影片路徑（CLI 模式必填） |
| `--cli` | `False` | 以 CLI 模式執行（不開啟 GUI） |
| `--window_size` | `300` | 每個滑動視窗的幀數 |
| `--stride` | `150` | 視窗步進大小（幀） |
| `--threshold` | `0.5` | 跌倒信心門檻值 |
| `--output_dir` | `./inference_output` | 輸出檔案目錄 |
| `--ensemble_weights` | `0.6,0.6,0.4,0.4` | 以逗號分隔的模型權重 |
| `--save_dir` | `./save_models` | 已訓練檢查點的目錄 |

### 輸出

- **`results.json`** —— 各視窗分數、逐幀聚合信心值，以及偵測到的跌倒區間。
- **標註影片** —— 帶有骨架疊加與即時信心值條的影片。

檢查點會從 `--save_dir` 自動偵測（選擇各模態中最高訓練輪數的檢查點）。

## 架構

Shift-GCN 以通道位移（Channel-wise circular shift）取代傳統圖卷積（Graph Convolution），消除了可學習的鄰接矩陣（Adjacency Matrix）。這使得模型非常輕量（約 720K 參數），同時保持高準確率。

**輸入張量形狀：** `(N, C, T, V, M)` = （批次, 3 座標, 300 幀, 33 關節點, 1 人）

```
data_bn → 10 TCN_GCN blocks → Global Average Pooling → FC(256, 2)

Block layout:
  Blocks 1–4:   64 channels, stride 1
  Block 5:     128 channels, stride 2  (temporal downsampling)
  Blocks 6–7:  128 channels, stride 1
  Block 8:     256 channels, stride 2  (temporal downsampling)
  Blocks 9–10: 256 channels, stride 1
```

每個 `TCN_GCN_unit` 先進行空間位移卷積（`Shift_gcn`），再進行時間位移卷積（`Shift_tcn`），並加上殘差連接（Residual connection）。

完整細節請參閱[原始論文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.pdf)。

## 引用

如使用本專案，請引用原始 Shift-GCN 論文：

```bibtex
@inproceedings{cheng2020shiftgcn,
  title     = {Skeleton-Based Action Recognition with Shift Graph Convolutional Network},
  author    = {Ke Cheng and Yifan Zhang and Xiangyu He and Weihan Chen and Jian Cheng and Hanqing Lu},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
}
```

## 致謝

本專案以 Ke Cheng 等人的 [Shift-GCN](https://github.com/kchengiva/Shift-GCN) 程式碼為基礎。MediaPipe Pose 的適配、二元跌倒偵測訓練管線與影片推論系統為獨立開發。
