# Learning Transformer

從零開始理解 Transformer，最終目標是實作 [nanoGPT](notebooks/NB4-nanoGPT.ipynb)。

每個概念都有三個層次：**直覺說明 → 數學推導 → 程式實作**，可以依自己的程度選擇切入點。

---

## 前言：開始之前

在常見的神經網路模型裡，Transformer 屬於**進階難度（★★★★☆）**——但只要拆成「直覺 → 數學 → 實作」三層，從零讀懂完全可行。這份教材的唯一目標，就是**讓你能獨立看懂、並親手刻出 nanoGPT**。

> 📍 **第一次來、或想確認方向？** 先讀 [`theory/00-learning-path.md`](theory/00-learning-path.md)——它說明 Transformer 在機器學習史上的位置、本教材會用到與值得延伸的數學、完整學習路線，以及學完後的自我檢查與下一步。

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
| `torch` | 2.x | 神經網路、autograd（NB2、NB4、NB5） |
| `numpy` | 2.x | 手刻矩陣運算（NB1、NB3） |
| `matplotlib` | 3.x | 損失曲線、注意力熱圖（NB2、NB3、NB4、NB5） |
| `jupyterlab` | 4.x | Notebook 執行環境 |
| `pandas` | 2.x | 少量資料整理（NB3） |
| `transformers` | 選用 | 僅 NB5 附錄「載入 HuggingFace 預訓練 BERT」選讀段需要（`pip install transformers`；預設不執行）|

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

03a Transformer 架構          ──────▶  NB1 §6–§8
  (Multi-Head、Block、PE)              NB2 完整模型
  └ 03b1→03b2→03b3 計算案例（選讀）     NB1 §13 可重現
    (簡單→中等→完整，三階段)
  └ 03b4 含位置編碼版（選讀對照支線）    NB1 §13b 可重現
    (P≠0，把位置真的加進去算一次)

        ↓

04a GPT 基本概念與 Pipeline   ──────▶  NB4 §4
  (架構差異、Causal Mask、資料流)        Head / Block / GPT 類別
        ↓
05a1 前向數學 ／ 05a2 前向數值範例 ─▶  NB4 前向
  (Scaled/Causal/MHA/FFN/LN/CE)

        ↓

05b1 後向數學 ／ 05b2 後向數值範例 ─▶  NB3 每個 .backward()
  (CE/LN/FFN/Attention/Embedding 梯度)

        ↓

06 當代 Transformer 架構      ──────▶  LLaMA 等開源模型原始碼
  (RMSNorm、SwiGLU、RoPE、GQA)         （decoder 家族出口，選讀）

        ↓

07 BERT Encoder-Only          ──────▶  NB5 最小 BERT（MLM）
  (雙向、MLM、[CLS]/[SEP])              （encoder 家族分支，選讀）

        ↓

09 文字轉向量與 RAG            ──────▶  語意檢索應用
  (Word2Vec、動態 embedding、餘弦)      （encoder 分支的應用出口，選讀）
```

> 主線 01→04→NB4 走 decoder-only 的 GPT；學完後有選讀分支：
> **06 / 未來的 NB6** 往 decoder 家族的 LLaMA，**07 / NB5** 往 encoder 家族的 BERT，
> **09** 再把 encoder 的句向量接到 RAG 語意檢索。

---

### 文件清單

#### 理論主線 (`theory/`)

| 文件 | 說明 |
|---|---|
| [`00-learning-path.md`](theory/00-learning-path.md) | 學習路線與背景（前言／導讀）：ML 歷史、必學與延伸數學、最短主線、自我檢查與後續路線 |
| [`01a-prerequisites-intuition.md`](theory/01a-prerequisites-intuition.md) | Embedding、Softmax、加權平均（白話版） |
| [`01b-prerequisites-math.md`](theory/01b-prerequisites-math.md) | 同上，附完整統計推導（數學版） |
| [`02-attention-intuition.md`](theory/02-attention-intuition.md) | QKV 直覺、翻譯範例逐步計算 |
| [`03a-transformer-architecture.md`](theory/03a-transformer-architecture.md) | Multi-Head Attention、Transformer Block、Positional Encoding（含 QKV／縮放／多頭逐步數值範例） |
| [`03a-transformer-block-plain.md`](theory/03a-transformer-block-plain.md) | 03a §6 的白話輔助版（選讀）：用生活比喻串起 Attention／FFN／Residual／LayerNorm 四模組，零公式，服務零基礎讀者 |
| [`03b1-transformer-example-basic.md`](theory/03b1-transformer-example-basic.md) | 03a 計算案例・簡單版（選讀）：$2\times4$ 輸入手算單頭 attention（$X\to\tilde X\to C^{(1)}$） |
| [`03b2-transformer-example-block.md`](theory/03b2-transformer-example-block.md) | 03a 計算案例・中等版（選讀）：承接 03b1，補上多頭、$W_O$、殘差、FFN，算到 Block 輸出 $Y$ |
| [`03b3-transformer-architecture-example.md`](theory/03b3-transformer-architecture-example.md) | 03a 計算案例・完整版（選讀）：§0 依前向順序推導每個矩陣的設計歷程，再算整個 Pre-LN Block，含縮放對照與 PE 旋轉驗證，對應 NB1 §13 |
| [`03b4-transformer-example-with-position.md`](theory/03b4-transformer-example-with-position.md) | 03b 選讀對照支線（純計算展演）：把位置編碼 $P$ 真的加進輸入（$X_{\text{in}}=X+P$，P≠0），沿用同一組權重從頭算一次完整 Block，對應 NB1 §13b（數字自成一組，不與 03b1–03b3 共用）|
| [`04a-gpt-decoder-only.md`](theory/04a-gpt-decoder-only.md) | GPT Decoder-Only 的**基本概念、架構差異與 Pipeline 總覽**（前向＋反向一覽；數學細節見 05a1/05a2、05b1/05b2）|
| [`05a1-forward-propagation.md`](theory/05a1-forward-propagation.md) | GPT **向前傳播數學（符號）**：Scaled Dot-Product、Causal Masking、Multi-Head／FFN／Pre-LN、Embedding／PE、Next-token 與 Cross-Entropy；高中數學程度，附記號約定、逐節學習目標與文末「一次 forward 的 11 步」回顧 |
| [`05a2-forward-example.md`](theory/05a2-forward-example.md) | GPT **前向數值範例**：用一組範例資料（$T=2$、$d=3$）把前向每個階段實際算一次，對照 05a1 各節 |
| [`04b-nanogpt-walkthrough.md`](theory/04b-nanogpt-walkthrough.md) | GPT Decoder-Only 的**程式對照**：nanoGPT 逐行解析、Pre-LN vs Post-LN、Tokenizer、自迴歸生成與 KV Cache（每節回指 05a1／05b1 數學）|
| [`05b1-backward-propagation.md`](theory/05b1-backward-propagation.md) | GPT **向後傳播數學（符號）**：以 05a1 的**倒序**走一遍（CE→lm_head→LayerNorm→FFN→Multi-Head→Attention→Embedding），高中數學程度，附最小工具箱與逐節學習目標 |
| [`05b2-backward-example.md`](theory/05b2-backward-example.md) | GPT **後向數值範例**：沿用 05a2 的數字，把反向每個階段的梯度實際算一次，對照 05b1 各節 |
| [`06-modern-transformer-variants.md`](theory/06-modern-transformer-variants.md) | RMSNorm、SwiGLU、RoPE、GQA、Flash Attention——nanoGPT 到 LLaMA 的橋接（選讀，decoder 家族出口） |
| [`07-bert-encoder-only.md`](theory/07-bert-encoder-only.md) | BERT／Encoder-Only：雙向 Self-Attention、MLM 預訓練、`[CLS]`/`[SEP]`、預訓練+微調、encoder vs decoder 選型（選讀，encoder 家族分支） |
| [`09-text-to-vector-rag.md`](theory/09-text-to-vector-rag.md) | 文字轉向量與語意檢索：分佈假說、Word2Vec、動態 embedding、餘弦相似度、RAG 檢索流程（選讀，encoder 分支的應用出口） |
| [`10a1-seq2seq-forward.md`](theory/10a1-seq2seq-forward.md) | Seq2Seq **前向數學（符號）**：RNN Seq2Seq＋Bahdanau 加性 attention、Transformer Encoder-Decoder、Cross-Attention、三種 attention 對照（選讀，encoder-decoder 家族分支）|
| [`10a2-seq2seq-forward-example.md`](theory/10a2-seq2seq-forward-example.md) | Seq2Seq **前向數值範例**：同一個翻譯任務（`我吃`→`I eat`）兩代架構各完整算一次到 loss |
| [`10b1-seq2seq-backward.md`](theory/10b1-seq2seq-backward.md) | Seq2Seq **後向數學（符號）**：BPTT 完整推導、梯度消失的定量說明、Cross-Attention 的梯度分岔 $G^H=G^H|_{(K)}+G^H|_{(V)}$ |
| [`10b2-seq2seq-backward-example.md`](theory/10b2-seq2seq-backward-example.md) | Seq2Seq **後向數值範例**：沿用 10a2 的兩組數字反推全部參數梯度，量化「attention vs 時間鏈」與「encoder vs decoder」的梯度差距 |

#### 實作主線 (`notebooks/`)

| Notebook | 說明 | 前置理論 |
|---|---|---|
| [`NB1-simple-llm-vanilla.ipynb`](notebooks/NB1-simple-llm-vanilla.ipynb) | NumPy 從零實作，無框架依賴 | 01 + 02 + 03 |
| [`NB2-simple-llm-pytorch.ipynb`](notebooks/NB2-simple-llm-pytorch.ipynb) | PyTorch 版本 | 01 + 02 + 03 |
| [`NB3-llm-backpropagation.ipynb`](notebooks/NB3-llm-backpropagation.ipynb) | NumPy 手刻完整反向傳播 | 01–03 + 05b1 |
| [`NB4-nanoGPT.ipynb`](notebooks/NB4-nanoGPT.ipynb) | 完整 nanoGPT，訓練莎士比亞文本 | 01–03 + 04a/05a1/04b |
| [`NB5-bert-mlm.ipynb`](notebooks/NB5-bert-mlm.ipynb) | 從零手刻最小 BERT（去 causal mask + MLM）＋ HuggingFace 選讀延伸（選讀分支） | 01–03 + 07 |

#### 進階補充 (`advanced/`，選讀)

- [`Attention-Mechanism-Part1.md`](advanced/Attention-Mechanism-Part1.md) — Nadaraya-Watson 核回歸視角
- [`Attention-Mechanism-Part2.md`](advanced/Attention-Mechanism-Part2.md) — Bahdanau 注意力、seq2seq 歷史
- [`Attention-NW-Kernel-Regression.md`](advanced/Attention-NW-Kernel-Regression.md) — 注意力的非參數統計解釋
- [`Seq2Seq-and-Decoding-Techniques.md`](advanced/Seq2Seq-and-Decoding-Techniques.md) — Seq2Seq 應用脈絡、Encoder/Decoder、解碼與訓練工藝（teacher forcing、exposure bias、beam search、NAT、copy、guided attention、BLEU vs CE、RL；整理自李宏毅課程）
- [`Transformer-in-Nushell.md`](advanced/Transformer-in-Nushell.md) — 精簡速查版
- [`Suggested-Papers.md`](advanced/Suggested-Papers.md) — 延伸閱讀論文清單

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
| `draft/improvement-04-llama.md` | （規劃中）把 `06` 文末「下一步」做成可執行出口：新增 NB6 改造實作、`theory/08` 官方碼對照（原規劃 07／NB5 已改給 BERT 選讀分支） |
