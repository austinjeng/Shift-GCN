# Shift-GCN MediaPipe 適配報告

**專案**：用於骨架式動作辨識的 Shift-GCN  
**適配目標**：在原始 NTU RGB+D（25 關節）之外，新增對 MediaPipe Pose（33 標記點）的支援  
**作者**：[Your Name]  
**日期**：2026 年 2 月

---

## 執行摘要

本報告記錄了為支援 **MediaPipe Pose** 標記點而對 Shift-GCN 程式碼庫所做的修改。此適配使系統可將 Google MediaPipe Pose（33 標記點）作為原始 NTU RGB+D 骨架格式（25 關節）的替代方案，進而開啟以標準 RGB 影片進行 **即時視訊推論** 與 **自訂資料集建立** 的可能。

我們亦以 NTU RGB+D 影片搭配 MediaPipe 實作了概念驗證（POC）級的 **二元跌倒偵測** 流程，示範完整端到端工作流程。

---

## 變更內容

### MediaPipe Pose 圖結構定義

**檔案**：`graph/mediapipe_pose.py`（新檔，55 行）

#### 什麼是「圖結構」？為什麼需要它？

想像你的身體是一個「點與線」的網絡：
- **點（節點）**：身體上的關鍵位置，例如鼻子、肩膀、手肘、膝蓋等
- **線（邊）**：連接這些點的骨頭或身體部位

這就是「圖結構」——它告訴電腦「哪些身體部位是相連的」。例如：手肘連接手腕、肩膀連接手肘。AI 模型需要知道這些連接關係，才能理解人體動作的整體性，而不是把每個點當作獨立的資料。

#### 為什麼要加入「橋接邊」？

MediaPipe 原本定義的身體連接有一個問題：**它把身體分成了三個「孤島」**：

```
孤島 1：頭部（眼睛、耳朵、鼻子互相連接）
孤島 2：嘴部（左嘴角、右嘴角）
孤島 3：身體（肩膀以下的四肢和軀幹）
```

這三個部分各自內部有連接，但彼此之間沒有連接。這就像三個獨立的拼圖片段，電腦無法把它們組成完整的人體。

**橋接邊的作用**就是把這些孤島連起來：
- `鼻子 → 左肩膀`：把頭部連到身體（就像脖子的作用）
- `鼻子 → 左嘴角`：把嘴部連到頭部

加入這兩條「橋」之後，所有 33 個點就形成一個完整連通的網絡，AI 才能正確理解整個人體的動作。

#### 為什麼需要「生成樹」結構？

「生成樹」是一種特殊的圖結構，它確保：
1. 所有點都被連接（沒有孤立的點）
2. 沒有「繞圈」的路徑（從 A 到 B 只有一條路）
3. 有一個「根」節點（我們選擇鼻子）

這種結構讓 AI 可以有系統地「從頭到腳」或「從腳到頭」處理身體資訊，就像家族族譜一樣有清楚的層級關係。

**技術細節**：
- **33 個節點**，對應 MediaPipe Pose 標記點（0-32）
- **32 條有向邊**，形成生成樹結構
- **新增橋接邊**：
  - `NOSE(0) → LEFT_SHOULDER(11)`：連接頭部與身體
  - `NOSE(0) → MOUTH_LEFT(9)`：連接嘴部群與頭部
- 圖結構輸出 `inward`、`outward`、`neighbor` 邊列表，以及空間鄰接矩陣 `A`

**MediaPipe 標記點對應**：
```text
Face:    0-10 (nose, eyes, ears, mouth)
Arms:    11-22 (shoulders, elbows, wrists, fingers)
Torso:   11-12, 23-24 (shoulders, hips)
Legs:    23-32 (hips, knees, ankles, feet)
```

---

### 支援可變關節數的模型參數化

**檔案**：`model/shift_gcn.py`（已修改，+20/-5 行）

#### 什麼是「參數化」？為什麼需要它？

原始的 Shift-GCN 是專門為 NTU 資料集設計的，它「寫死」了一個數字：**25 個關節點**。這就像一件只有 M 號的衣服——如果你需要其他尺寸，就穿不下了。

MediaPipe 使用 **33 個標記點**（比 NTU 多了臉部細節和手指位置）。如果不修改程式，就像試圖把 33 顆珠子塞進只有 25 個洞的盒子——會出錯。

**「參數化」的意思是**：把「25」這個寫死的數字，改成一個可以調整的設定。現在只要在設定檔裡寫 `num_point: 33`，程式就會自動適應 33 個點的資料。

#### 為什麼這很重要？

Shift-GCN 的核心運算是「位移（shift）」操作——它會把不同關節的資訊互相交換、混合。這個操作需要知道「總共有幾個關節」才能正確執行。如果程式以為有 25 個關節，但實際資料有 33 個，就會發生：
- 資料超出範圍錯誤
- 部分身體部位的資訊被忽略
- 模型輸出完全錯誤

**變更內容**：

1. **`Model` 類別**：新增 `num_point` 參數，並傳遞給所有 `TCN_GCN_unit` 層
   ```python
   def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, ...):
       self.l1 = TCN_GCN_unit(3, 64, A, residual=False, num_point=num_point)
       # ... 所有 10 層現在都會接收 num_point
   ```

2. **`TCN_GCN_unit` 類別**：新增 `num_point` 參數，並傳遞給 `Shift_gcn`
   ```python
   def __init__(self, in_channels, out_channels, A, stride=1, residual=True, num_point=25):
       self.gcn1 = Shift_gcn(in_channels, out_channels, A, num_point=num_point)
   ```

3. **`Shift_gcn` 類別**：原本已具備 `num_point` 參數，現在改由設定檔傳入
   - `Feature_Mask` 形狀：`(1, num_point, in_channels)`
   - `shift_in`/`shift_out` 索引陣列：`(num_point * channels)`
   - `BatchNorm1d` 特徵數：`num_point * out_channels`

---

### 骨架前處理強化

**檔案**：`data_gen/preprocess.py`（已修改，+15/-6 行）

#### 什麼是「前處理」？為什麼需要「正規化」？

想像你要比較兩個人的動作：一個人站在鏡頭前方，另一個人站在鏡頭右側。即使他們做相同的動作，因為位置和角度不同，他們的骨架座標會完全不一樣。這會讓 AI 誤以為是不同的動作。

**正規化**就是把所有骨架資料「校正」到統一的標準：
1. **位置校正**：把每個人都「移動」到同一個中心點
2. **角度校正**：把每個人都「旋轉」到面向同一個方向

這樣 AI 就能專注於「動作本身」，而不會被「站在哪裡」或「面向哪邊」干擾。

#### 為什麼 MediaPipe 需要不同的中心點？

**NTU 資料集**有一個叫「脊椎（spine）」的關節點，正好在身體正中央——這是一個完美的中心參考點。

**MediaPipe 沒有這個點**。它有左髖關節和右髖關節，但沒有「正中間」的點。

**解決方案**：取左髖和右髖的「平均位置」當作中心點。這就像沒有中間那顆糖，就把左邊和右邊的糖各切一半拼起來。

程式原本只能接受「一個」中心點，現在修改為也能接受「一組」中心點（然後自動計算平均）。

**原始版本**（NTU）：
```python
center_joint = 1  # Spine joint
main_body_center = skeleton[0][:, center_joint:center_joint+1, :]
```

**修改後**（相容 MediaPipe）：
```python
center_joint = [23, 24]  # Hip midpoint (left hip, right hip)
if isinstance(center_joint, (list, tuple)):
    main_body_center = np.mean(
        [skeleton[0][:, j:j+1, :] for j in center_joint], axis=0
    )
```

**正規化參數**：
| 參數 | NTU RGB+D | MediaPipe Pose |
|------|-----------|----------------|
| `center_joint` | `1`（spine） | `[23, 24]`（髖部中點） |
| `zaxis` | `[0, 1]`（hip→spine） | `[23, 11]`（hip→shoulder） |
| `xaxis` | `[8, 4]`（shoulders） | `[12, 11]`（shoulders） |

---

### MediaPipe 資料生成流程

**檔案**：`data_gen/mediapipe_gendata.py`（新檔，389 行）

#### 這個程式做什麼？

這是一個「影片轉骨架」的轉換器。它的工作流程：

```
輸入：一支普通的影片檔（.avi）
  ↓
步驟 1：把影片拆成一張張圖片（幀）
  ↓
步驟 2：用 MediaPipe 分析每張圖片，找出人體的 33 個關鍵點
  ↓
步驟 3：把所有幀的關鍵點座標整理成統一格式
  ↓
輸出：一個數字陣列檔案（.npy），可以直接餵給 AI 模型訓練
```

#### 為什麼需要「NTU 模式」？

NTU RGB+D 是一個大型動作資料集，它的影片檔名遵循特殊格式，例如：
```
S001C001P001R001A043_rgb.avi
```

這個檔名包含豐富資訊：
- `S001`：拍攝場景編號 1
- `C001`：攝影機編號 1
- `P001`：受試者編號 1
- `R001`：第 1 次重複拍攝
- `A043`：動作編號 43（跌倒）

程式可以自動解析這些資訊，用來：
1. **自動標記**：A043 = 跌倒，其他 = 非跌倒
2. **正確分組**：哪些影片用於訓練、哪些用於驗證

#### 什麼是「類別平衡」？為什麼需要它？

假設你有 100 支影片：95 支是「正常走路」，5 支是「跌倒」。如果直接用這些資料訓練 AI，它可能會學到一個「偷懶」的策略：**不管看到什麼，都猜「正常走路」**——這樣就能答對 95%！但這顯然不是我們要的。

**類別平衡**就是減少「正常走路」的數量（例如只留 10 支），讓兩類資料數量接近，強迫 AI 真正學習區分它們。

**核心函式**：

1. **`extract_landmarks(video_path, max_frame=300)`**
   - 以 OpenCV 開啟影片
   - 對每一幀執行 MediaPipe Pose（`model_complexity=1`）
   - 使用 `pose_world_landmarks`（以公尺為單位的 3D 座標）
   - 回傳 `(3, T, 33, 1)` 陣列；若未偵測到姿態則回傳 `None`

2. **`gendata(video_dir, out_path, label_map, split_file=None)`**
   - 通用模式：依目錄結構中的類別標籤處理影片
   - 輸出 `data_joint.npy` 與 `label.pkl`

3. **`gendata_ntu(video_dir, out_path, falling_action=43, benchmark='xsub', ...)`**
   - NTU 專用模式：解析 NTU 檔名慣例
   - 二元跌倒偵測：動作 43 = 跌倒（標籤 1），其餘皆為非跌倒（標籤 0）
   - 支援 cross-subject（`xsub`）與 cross-view（`xview`）切分
   - 可透過 `subsample_ratio` 參數做類別平衡

**NTU 檔名解析器**：
```python
def parse_ntu_filename(filename):
    # SsssCcccPpppRrrrAaaa.ext
    # S=setup, C=camera, P=subject, R=replication, A=action
    action = int(name[name.find('A')+1:name.find('A')+4])  # e.g., A043 → 43
```

**類別平衡**：
```python
def _subsample_negatives(videos, ratio, seed):
    # 以決定性方式下採樣 negatives 至 ratio * positives
    # 確保不同執行輪次間具可重現性
```

---

### POC 影片子集測試

**檔案**：`data_gen/poc_videos.txt`（新檔，10 行）

#### 什麼是 POC？為什麼需要它？

**POC（Proof of Concept，概念驗證）** 是一種「先小規模試跑」的策略。

想像你要煮一大鍋湯給 100 人喝。你不會一開始就買齊所有材料、煮滿整鍋——你會先煮一小碗試試味道對不對。如果味道不對，只浪費了一點點材料和時間。

同樣地，處理完整的 NTU 資料集（數萬支影片）需要好幾天。如果程式有 bug，就浪費了好幾天。

**POC 策略**：先挑 10 支影片測試。只需要幾分鐘就能跑完，可以快速驗證：
- 影片能正確讀取嗎？
- MediaPipe 能正確偵測姿態嗎？
- 輸出格式正確嗎？
- 訓練程式能接受這個資料嗎？

確認一切正常後，再跑完整資料集。

#### 為什麼挑選這 10 支影片？

```text
S001C001P001R001A043_rgb.avi  # Fall, training subject
S001C001P002R001A043_rgb.avi  # Fall, validation subject
S001C001P001R002A043_rgb.avi  # Fall, training subject
S001C001P001R001A001_rgb.avi  # Non-fall (drinking), training
S001C001P002R001A001_rgb.avi  # Non-fall, validation
S001C001P001R001A010_rgb.avi  # Non-fall (clapping), training
S001C001P003R001A043_rgb.avi  # Fall, validation subject
S001C001P003R002A043_rgb.avi  # Fall, validation subject
S001C001P003R001A002_rgb.avi  # Non-fall (eating), validation
S001C001P003R001A010_rgb.avi  # Non-fall, validation
```

**挑選準則**：
- **平衡**：~50% 跌倒（A043）、~50% 非跌倒——確保兩種情況都有測試到
- **涵蓋訓練與驗證**：確保資料分組邏輯正確
- **多種動作**：喝水、拍手、吃東西等——確保不只是跌倒能被處理
- **快速**：10 支影片幾分鐘就能處理完

---

### Bone 資料生成（MediaPipe）

**檔案**：`data_gen/gen_bone_data_mediapipe.py`（新檔，67 行）

#### 什麼是「Bone 資料」？跟「Joint 資料」有什麼不同？

- **Joint（關節）資料**：記錄每個關節點的「絕對位置」
  - 例如：「手腕在座標 (0.3, 0.5, 0.2)」

- **Bone（骨頭）資料**：記錄關節之間的「相對方向和長度」
  - 例如：「從手肘到手腕的方向是 (0.1, -0.2, 0.0)」

#### 為什麼需要 Bone 資料？

想像兩個人做相同的「舉手」動作，但一個手臂長、一個手臂短。他們的 Joint 資料會很不一樣（手腕位置不同），但 Bone 資料會很相似（手臂都是往上伸）。

**Bone 資料幫助 AI 專注於「動作方向」，而非「身體尺寸」**。

結合 Joint 和 Bone 資料訓練的模型，通常比只用 Joint 的模型更準確。

#### 計算方式

對每個關節，找到它的「父關節」（在身體結構上比它更靠近中心的關節），然後計算兩者的座標差：

```
Bone[手腕] = Joint[手腕] - Joint[手肘]
Bone[手肘] = Joint[手肘] - Joint[肩膀]
```

**Bone 配對**（1-indexed，與圖拓樸一致）：
```python
paris = {
    'mediapipe': (
        (1, 1),    # NOSE (root, self-reference)
        (2, 1),    # LEFT_EYE_INNER → NOSE
        ...
        (24, 12),  # LEFT_HIP → LEFT_SHOULDER
        (25, 13),  # RIGHT_HIP → RIGHT_SHOULDER
        ...
        (33, 29),  # RIGHT_FOOT_INDEX → RIGHT_ANKLE
    )
}
```

**輸出**：`{train,val}_data_bone.npy`，形狀為 `(N, 3, T, 33, 1)`

---

### Motion 資料生成（MediaPipe）

**檔案**：`data_gen/gen_motion_data_mediapipe.py`（新檔，28 行）

#### 什麼是「Motion 資料」？

Motion 資料記錄的是「變化」——每個關節在相鄰兩幀之間移動了多少。

想像一部動畫是由 100 張靜態圖片組成的：
- **Joint 資料**：第 1 張圖手在哪裡、第 2 張圖手在哪裡...
- **Motion 資料**：第 1→2 張圖手移動了多少、第 2→3 張圖手移動了多少...

#### 為什麼需要 Motion 資料？

**「跌倒」和「緩慢蹲下」的最終姿勢可能很像**（都是人在地上），但移動的「速度」完全不同：
- 跌倒：短時間內大幅度位移
- 蹲下：每幀只有微小位移

Motion 資料直接捕捉這種「速度」資訊，幫助 AI 區分這類動作。

#### 計算方式

非常直接：下一幀減掉這一幀

```python
for t in range(T - 1):
    motion[:, :, t, :, :] = data[:, :, t+1, :, :] - data[:, :, t, :, :]
motion[:, :, T-1, :, :] = 0  # Last frame has no successor
```

最後一幀沒有「下一幀」可以比較，所以設為 0。

**輸出**：
- `{train,val}_data_joint_motion.npy`
- `{train,val}_data_bone_motion.npy`

---

### 設定檔

**目錄**：`config/mediapipe/`（新建，4 個檔案）

#### 什麼是設定檔？為什麼需要它？

設定檔就像食譜的「材料清單」和「烹飪參數」：
- 要讀取哪個資料檔？
- 使用哪種圖結構？
- 要辨識幾種動作？
- 用幾張 GPU 訓練？

把這些資訊寫在設定檔裡（而不是寫死在程式碼裡），好處是：
1. **不用改程式碼**：換資料集只要換設定檔
2. **易於追蹤**：可以保存每次實驗的設定，方便比較和重現
3. **減少錯誤**：所有參數集中在一個地方，不會漏改

#### 四種設定檔對應四種資料

| Config | 資料類型 | 說明 |
|--------|----------|------|
| `train_joint.yaml` | 關節位置 | 最基本的骨架資料 |
| `train_bone.yaml` | 骨頭方向 | 關節間的相對關係 |
| `train_joint_motion.yaml` | 關節速度 | 關節位置的時間變化 |
| `train_bone_motion.yaml` | 骨頭速度 | 骨頭方向的時間變化 |

**與 NTU 設定的主要差異**：
```yaml
# MediaPipe                      # NTU RGB+D
num_class: 2                     num_class: 60
num_point: 33                    num_point: 25
num_person: 1                    num_person: 2
graph: graph.mediapipe_pose      graph: graph.ntu_rgb_d
device: [0]                      device: [0,1,2,3]
```

差異說明：
- **num_class: 2 vs 60**：我們只做「跌倒/非跌倒」二元分類，NTU 原本有 60 種動作
- **num_point: 33 vs 25**：MediaPipe 有 33 個標記點，NTU 只有 25 個關節
- **num_person: 1 vs 2**：我們假設畫面中只有一個人（簡化問題）
- **device: [0] vs [0,1,2,3]**：我們只用一張 GPU（POC 規模較小）

---

### 集成評估腳本

**檔案**：`ensemble_mediapipe.py`（新檔，36 行）

#### 什麼是「集成（Ensemble）」？

集成是一種「集思廣益」的策略：不依賴單一模型的判斷，而是綜合多個模型的意見。

想像你去看病，與其只聽一位醫生的診斷，不如諮詢四位專科醫生，然後綜合他們的意見做決定。這通常比單一意見更可靠。

#### 為什麼用四個模型？

我們訓練了四個模型，每個模型「看」的資料角度不同：

| 模型 | 關注重點 |
|------|----------|
| Joint 模型 | 關節的「位置」在哪裡 |
| Bone 模型 | 骨頭的「方向」如何 |
| Joint Motion 模型 | 關節「移動」了多少 |
| Bone Motion 模型 | 骨頭方向「改變」了多少 |

就像四位不同專長的醫生，各自從不同角度分析同一個病人。

#### 為什麼權重是 [0.6, 0.6, 0.4, 0.4]？

```python
alpha = [0.6, 0.6, 0.4, 0.4]  # [joint, bone, joint_motion, bone_motion]
r = r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]
```

這些權重代表「信任程度」：
- Joint 和 Bone 模型權重較高（0.6）：它們的意見佔比較重
- Motion 模型權重較低（0.4）：它們的意見佔比較輕

這是根據過去經驗調整的——通常位置和方向資訊比速度資訊更穩定可靠。但這些權重可以根據實際表現調整。

---

### 錯誤修正

**檔案**：`main.py`（已修改，1 行）

#### 這是什麼問題？

PyYAML 是一個讀取設定檔的工具。在新版本中，它要求使用者明確指定「如何解讀設定檔」（出於安全考量——惡意的設定檔可能包含危險指令）。

舊的寫法沒有指定，所以新版會跳出警告。這個修正只是加上明確的指定，讓程式不再警告。

```python
# Before (deprecated)
arg = yaml.load(f)

# After (explicit loader)
arg = yaml.load(f, Loader=yaml.FullLoader)
```

---

## 資料流程圖

```text
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA GENERATION                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
┌─────────┐                  ┌─────────────┐                  ┌─────────┐
│  Video  │                  │  MediaPipe  │                  │   NTU   │
│  Files  │ ───────────────▶ │    Pose     │ ◀─────────────── │Skeleton │
│ (.avi)  │   RGB frames     │ Extraction  │  (or use native) │  Data   │
└─────────┘                  └─────────────┘                  └─────────┘
                                    │
                                    ▼ (3, T, 33, 1)
                          ┌─────────────────┐
                          │ pre_normalization│
                          │ • center at hips │
                          │ • align z-axis   │
                          │ • align x-axis   │
                          └─────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
    ┌───────────┐            ┌───────────┐            ┌───────────────┐
    │   Joint   │            │   Bone    │            │    Motion     │
    │   Data    │            │   Data    │            │  (temporal)   │
    │ (N,3,T,V,M)│           │(diff pairs)│           │ (frame diff)  │
    └───────────┘            └───────────┘            └───────────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                             TRAINING                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │    Shift-GCN    │
                          │   (10 layers)   │
                          │  num_point=33   │
                          └─────────────────┘
                                    │
                      ┌─────────────┼─────────────┐
                      │             │             │
                      ▼             ▼             ▼
                ┌─────────┐   ┌─────────┐   ┌─────────┐
                │  Joint  │   │  Bone   │   │ Motion  │
                │ Model   │   │ Model   │   │ Models  │
                └─────────┘   └─────────┘   └─────────┘
                      │             │             │
                      └─────────────┼─────────────┘
                                    │
                                    ▼
                          ┌─────────────────┐
                          │    Ensemble     │
                          │  α=[0.6,0.6,    │
                          │     0.4,0.4]    │
                          └─────────────────┘
```

---

## POC 驗證結果

**指令**：
```bash
python mediapipe_gendata.py \
    --video_dir "E:\nturgb+d_rgb" \
    --out_dir ../data/mediapipe/ \
    --ntu_mode \
    --video_list poc_videos.txt \
    --subsample_ratio 0 \
    --benchmark xsub
```

**輸出**：
| Split | Shape | Fall | Non-Fall |
|-------|-------|------|----------|
| Train | (6, 3, 300, 33, 1) | 3 | 3 |
| Val | (4, 3, 300, 33, 1) | 2 | 2 |

- 標籤：`{0, 1}`（二元跌倒偵測）
- 每個樣本 300 幀（較短影片會補零）
- 33 個 MediaPipe 標記點 × 3 座標（X、Y、Z）

---

## 檔案變更摘要

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `graph/mediapipe_pose.py` | **New** | 55 | MediaPipe 33 標記點圖拓樸 |
| `data_gen/mediapipe_gendata.py` | **New** | 389 | 影片 → 骨架資料流程 |
| `data_gen/gen_bone_data_mediapipe.py` | **New** | 67 | Bone 特徵生成 |
| `data_gen/gen_motion_data_mediapipe.py` | **New** | 28 | 時序 Motion 特徵 |
| `data_gen/poc_videos.txt` | **New** | 10 | POC 測試影片清單 |
| `config/mediapipe/*.yaml` | **New** | 160 | 訓練設定（4 個檔案） |
| `ensemble_mediapipe.py` | **New** | 36 | 集成評估 |
| `model/shift_gcn.py` | Modified | +20 | 參數化 `num_point` |
| `data_gen/preprocess.py` | Modified | +15 | 支援多關節中心 |
| `main.py` | Modified | +1 | YAML loader 修正 |
| `CLAUDE.md` | **New** | 64 | 專案文件 |

**總計**：新增 850+ 行，修改 25 行

---

## 後續步驟

1. **在 POC 資料上訓練**：驗證端到端訓練可正常運作
   ```bash
   python main.py --config ./config/mediapipe/train_joint.yaml
   ```

2. **生成完整資料集**：處理所有 NTU RGB+D 影片
   ```bash
   python mediapipe_gendata.py --video_dir "E:\nturgb+d_rgb" --ntu_mode --subsample_ratio 1.0
   ```

3. **生成 bone/motion 資料**：
   ```bash
   cd data_gen && python gen_bone_data_mediapipe.py
   cd data_gen && python gen_motion_data_mediapipe.py
   ```

4. **訓練四個模型**，並執行集成評估

5. **即時推論**：適配為 webcam / 視訊串流輸入

---

## 附錄：MediaPipe Pose 標記點參考

```text
 0: NOSE                    17: RIGHT_PINKY
 1: LEFT_EYE_INNER          18: LEFT_INDEX
 2: LEFT_EYE                19: RIGHT_INDEX
 3: LEFT_EYE_OUTER          20: LEFT_THUMB
 4: RIGHT_EYE_INNER         21: RIGHT_THUMB
 5: RIGHT_EYE               22: LEFT_HIP
 6: RIGHT_EYE_OUTER         23: RIGHT_HIP
 7: LEFT_EAR                24: LEFT_KNEE
 8: RIGHT_EAR               25: RIGHT_KNEE
 9: MOUTH_LEFT              26: LEFT_ANKLE
10: MOUTH_RIGHT             27: RIGHT_ANKLE
11: LEFT_SHOULDER           28: LEFT_HEEL
12: RIGHT_SHOULDER          29: RIGHT_HEEL
13: LEFT_ELBOW              30: LEFT_FOOT_INDEX
14: RIGHT_ELBOW             31: RIGHT_FOOT_INDEX
15: LEFT_WRIST              32: (Reserved)
16: RIGHT_WRIST
```

---

*此報告為 Shift-GCN MediaPipe 適配專案產出*
