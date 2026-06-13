# Learning Transformer

從零開始理解 Transformer，最終目標是實作 [nanoGPT](notebooks/NB4-nanoGPT.ipynb)。

每個概念都有三個層次：**直覺說明 → 數學推導 → 程式實作**，可以依自己的程度選擇切入點。

---

## 環境安裝

### 方法一：Conda（建議）

使用 Conda 可以隔離 Python 版本與套件，避免與系統環境衝突。

```bash
# 建立獨立環境（Python 3.11）
conda create -n transformer python=3.11 -y

# 啟動環境
conda activate transformer

# 安裝 PyTorch（含 CUDA 支援；純 CPU 可移除 pytorch-cuda=12.1）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安裝其他套件
pip install numpy matplotlib jupyterlab pandas

# 啟動 JupyterLab
jupyter lab
```

> **純 CPU 環境**（無 GPU）將最後兩行替換為：
> ```bash
> conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
> pip install numpy matplotlib jupyterlab pandas
> ```

### 方法二：pip

```bash
pip install torch torchvision torchaudio numpy matplotlib jupyterlab pandas
jupyter lab
```

### 核心套件

| 套件 | 版本 | 用途 |
|---|---|---|
| `python` | 3.11 | 執行環境 |
| `torch` | 2.11 | 神經網路、autograd（NB2、NB4） |
| `numpy` | 2.x | 手刻矩陣運算（NB1、NB3） |
| `matplotlib` | 3.x | 損失曲線、注意力熱圖（NB2、NB3、NB4） |
| `jupyterlab` | 4.x | Notebook 執行環境 |
| `pandas` | 2.x | 少量資料整理（NB3） |

---

## 學習路線

### 兩個起點，選一個

| | 直覺版 | 數學版 |
|---|---|---|
| **適合對象** | 高中數學，初次接觸 | 大學線性代數，想看完整推導 |
| **第一篇** | [`01a-prerequisites-intuition.md`](theory/01a-prerequisites-intuition.md) | [`01b-prerequisites-math.md`](theory/01b-prerequisites-math.md) |

兩個版本都通向相同的後續內容。

---

### 完整路線圖

```
理論                                   對應實作
─────────────────────────────────────────────────────────
01a 直覺版前置知識            ──┐
  或                              ├──▶  NB1 §1–§4
01b 數學版前置知識            ──┘     Embedding / Softmax

        ↓

02 注意力的直覺               ──────▶  NB1 §5
  (QKV 翻譯範例)                       SelfAttention 類別

        ↓

03 Transformer 架構           ──────▶  NB1 §6–§8
  (Multi-Head、Block、PE)              NB2 完整模型

        ↓

04 GPT Decoder-Only           ──────▶  NB4 §4
  (Causal Mask、語言模型目標)           Head / Block / GPT 類別

        ↓

05 反向傳播推導               ──────▶  NB3 每個 .backward()
  (QKV / LayerNorm / Embedding 梯度)

        ↓

06 當代 Transformer 架構      ──────▶  LLaMA 等開源模型原始碼
  (RMSNorm、SwiGLU、RoPE、GQA)         （主線出口，選讀）
```

---

### 文件清單

#### 理論主線 (`theory/`)

| 文件 | 說明 |
|---|---|
| [`01a-prerequisites-intuition.md`](theory/01a-prerequisites-intuition.md) | Embedding、Softmax、加權平均（白話版） |
| [`01b-prerequisites-math.md`](theory/01b-prerequisites-math.md) | 同上，附完整統計推導（數學版） |
| [`02-attention-intuition.md`](theory/02-attention-intuition.md) | QKV 直覺、翻譯範例逐步計算 |
| [`03-transformer-architecture.md`](theory/03-transformer-architecture.md) | Multi-Head Attention、Transformer Block、Positional Encoding |
| [`04-gpt-decoder-only.md`](theory/04-gpt-decoder-only.md) | Causal Masking、語言模型訓練目標、nanoGPT 架構解析 |
| [`05-backpropagation.md`](theory/05-backpropagation.md) | Self-Attention、LayerNorm 與 Embedding 的完整梯度推導 |
| [`06-modern-transformer-variants.md`](theory/06-modern-transformer-variants.md) | RMSNorm、SwiGLU、RoPE、GQA、Flash Attention——nanoGPT 到 LLaMA 的橋接（選讀） |

#### 實作主線 (`notebooks/`)

| Notebook | 說明 | 前置理論 |
|---|---|---|
| [`NB1-simple-llm-vanilla.ipynb`](notebooks/NB1-simple-llm-vanilla.ipynb) | NumPy 從零實作，無框架依賴 | 01 + 02 + 03 |
| [`NB2-simple-llm-pytorch.ipynb`](notebooks/NB2-simple-llm-pytorch.ipynb) | PyTorch 版本 | 01 + 02 + 03 |
| [`NB3-llm-backpropagation.ipynb`](notebooks/NB3-llm-backpropagation.ipynb) | NumPy 手刻完整反向傳播 | 01–05 |
| [`NB4-nanoGPT.ipynb`](notebooks/NB4-nanoGPT.ipynb) | 完整 nanoGPT，訓練莎士比亞文本 | 01–04 |

#### 進階補充 (`advanced/`，選讀)

- `Attention-Mechanism-Part1.md` — Nadaraya-Watson 核回歸視角
- `Attention-Mechanism-Part2.md` — Bahdanau 注意力、seq2seq 歷史
- `Attention-NW-Kernel-Regression.md` — 注意力的非參數統計解釋
- `Transformer-in-Nushell.md` — 精簡速查版
- `Suggested-Papers.md` — 延伸閱讀論文清單

---

## 常用指令

```bash
# 執行單一 notebook（非互動模式）
jupyter nbconvert --to notebook --execute "notebooks/NB1-simple-llm-vanilla.ipynb"

# 啟動互動式環境
jupyter lab
```

---

## 文件品質改善紀錄

本倉庫的理論文件經過多輪系統性檢視與補強，規劃與執行紀錄保存在 `draft/` 資料夾（僅供維護參考，不在主線閱讀路徑上）：

| 計劃 | 重點 |
|---|---|
| `draft/improvement-00-fixes.md` | 錯誤修正、數值範例、ASCII 圖表、章節銜接語 |
| `draft/improvement-01-mainline-gaps.md` | 主線概念缺口（$W_O$、FFN、PE、Dropout、KV Cache、Embedding 梯度）、新增 `06` 當代架構文件 |
| `draft/improvement-02-writing.md` | 數學推導逐步化（Softmax Jacobian、LayerNorm 合併代數等）、程式範例說明、失效引用修正 |
| `draft/improvement-03-notebooks.md` | Notebook 執行驗證：NB3 梯度驗證 bug 修復、NB4 首次執行、路徑隔離與 .gitignore 補強 |
| `draft/improvement-04-llama.md` | （規劃中）把 `06` 文末「下一步」做成可執行出口：新增 NB5 改造實作、`theory/07` 官方碼對照 |
