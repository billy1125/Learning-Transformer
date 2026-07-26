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

### 教材怎麼組織：五個區 × 四種呈現層次

整份教材依**兩條軸**排列。第一條軸是**五個區**，依序前進：

| 區 | 名稱 | 終點：讀完你能做到 | 成員 |
|---|---|---|---|
| **0** | 導覽 | 選定起點與路線 | `00` |
| **1** | 地基：一個 Transformer Block | **手算一個完整的 Pre-LN Block** | `01a`/`01b`、`02`、`03a`（＋輔助與計算案例）、NB1、NB2 |
| **2** | 三大家族 | 說出三家的遮罩與訓練目標差異，**手刻其中任一** | 2A GPT、2B BERT、2C Seq2Seq（三條平行支線）|
| **3** | 家族演進與模型家譜 | 看懂 LLaMA／RoBERTa 的原始碼在改什麼 | `08`、`06`、`07` §6 |
| **4** | 系統層應用 | 說明 RAG 為何要 encoder ＋ decoder 併用 | `09` |

第二條軸是**呈現層次**——同一份知識，本教材會用四種深度各講一次。看懂這條軸，就知道每份文件在做什麼、可以跳過哪些：

| 層次 | 標記 | 回答什麼 |
|---|---|---|
| 概念 | **概** | 這是什麼、為什麼需要 |
| 符號數學 | **數** | 公式怎麼推出來 |
| 數值範例 | **例** | 代真實數字算一次 |
| 程式對照 | **碼** | 公式對應到程式的哪一行 |

下面的文件清單每一列都標了層次。**最短主線**只走「概 ＋ 數 ＋ 碼」，跳過所有「例」與選讀分支：

```
01 → 02 → 03a → 04a → 05a1 → 04b → NB1 → NB2 → NB4
```

---

### 完整路線圖

```
【區 1】地基：一個 Transformer Block（三家共用，不分流）
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
  ├ 03a-plain  零公式輔助版（選讀）
  ├ 03b1→03b2→03b3 計算案例（選讀）     NB1 §13 可重現
  │   (簡單→中等→完整，三階段)
  └ 03b4 含位置編碼版（選讀對照支線）    NB1 §13b 可重現
      (P≠0，把位置真的加進去算一次)

═════════════════════════════════════════════════════════
【區 2】三大家族——三條【平行】支線，都建立在區 1 的 Block 上
        （不是串接！可以只走 2A，也可以三條都走）

  2A Decoder-only（GPT）       2B Encoder-only    2C Encoder-Decoder
     ★ 主線                       （選讀分支）        （選讀分支）
  ─────────────────────────    ───────────────    ──────────────────
  04a  概念與 Pipeline         07 BERT            10a1 前向數學
  05a1 前向數學                                   10a2 前向數值範例
  05a2 前向數值範例                                10b1 後向數學
  05b1 後向數學                                   10b2 後向數值範例
  05b2 後向數值範例
  04b  nanoGPT 程式對照
       │                          │                   │
       ▼                          ▼                   ▼
  NB3 手刻反向傳播             NB5 最小 BERT      （實作待補）
  NB4 完整 nanoGPT

  ※ 深度刻意不對稱：反向傳播只在 2A 完整推導一次，2B 直接共用，
    2C 只推「與 2A 不同的那部分」（BPTT、cross-attention 梯度分岔）。
  ※ 建議順序 2A → 2B → 2C；做完 2A 已抵達本教材主要目標。

═════════════════════════════════════════════════════════
【區 3】家族演進（架構層）        【區 4】系統層應用
─────────────────────────      ─────────────────────────
08 模型家譜速查                  09 文字轉向量與 RAG
   (三家時間軸、代表模型、           (Word2Vec、動態 embedding、
    當代主戰場)                      餘弦相似度、檢索流程)
06 當代 Transformer 架構              │
   (RMSNorm、SwiGLU、RoPE、GQA)       ▼
   → 讀 LLaMA 原始碼               Agent / Multimodal（本教材範圍外）
07 §6 encoder 家族速覽
   → RoBERTa / ELECTRA / SBERT
```

> **三家今天都還活著，只是主戰場不同**——2A 是幾乎所有通用大型語言模型；2B 的重心已從分類微調轉到 **embedding model**（BGE、E5、gte），是 RAG 的檢索端；2C 在純文字生成上被 decoder-only 取代，但在**跨模態／跨語言**仍是主流（Whisper、NLLB）。完整家譜見 [`08-model-family-tree.md`](theory/08-model-family-tree.md)。

---

### 文件清單

#### 區 0–1：導覽與地基 (`theory/`)

| 文件 | 層次 | 說明 |
|---|---|---|
| [`00-learning-path.md`](theory/00-learning-path.md) | 概 | 學習路線與背景（前言／導讀）：ML 歷史、必學與延伸數學、五個區 × 四種層次、自我檢查與後續路線 |
| [`01a-prerequisites-intuition.md`](theory/01a-prerequisites-intuition.md) | 概 | Embedding、Softmax、加權平均（白話版，高中數學程度） |
| [`01b-prerequisites-math.md`](theory/01b-prerequisites-math.md) | 數 | 同上，附完整統計推導（數學版，大學線性代數程度） |
| [`02-attention-intuition.md`](theory/02-attention-intuition.md) | 概 | QKV 直覺、翻譯範例逐步計算（`I eat fish` → `我吃魚`） |
| [`03a-transformer-architecture.md`](theory/03a-transformer-architecture.md) | 數 | **本區核心**：Multi-Head Attention、Transformer Block、Positional Encoding（含 QKV／縮放／多頭逐步數值範例） |
| [`03a-transformer-block-plain.md`](theory/03a-transformer-block-plain.md) | 概 | 03a §6 的白話輔助版（選讀）：用生活比喻串起 Attention／FFN／Residual／LayerNorm 四模組，零公式 |
| [`03b1-transformer-example-basic.md`](theory/03b1-transformer-example-basic.md) | 例 | 計算案例・簡單版（選讀）：$2\times4$ 輸入手算單頭 attention（$X\to\tilde X\to C^{(1)}$） |
| [`03b2-transformer-example-block.md`](theory/03b2-transformer-example-block.md) | 例 | 計算案例・中等版（選讀）：承接 03b1，補上多頭、$W_O$、殘差、FFN，算到 Block 輸出 $Y$ |
| [`03b3-transformer-architecture-example.md`](theory/03b3-transformer-architecture-example.md) | 例 | 計算案例・完整版（選讀）：§0 依前向順序推導每個矩陣的設計歷程，再算整個 Pre-LN Block，含縮放對照與 PE 旋轉驗證，對應 NB1 §13 |
| [`03b4-transformer-example-with-position.md`](theory/03b4-transformer-example-with-position.md) | 例 | 對照支線（選讀，純計算展演）：把位置編碼 $P$ 真的加進輸入（$X_{\text{in}}=X+P$，P≠0）從頭算一次，對應 NB1 §13b（數字自成一組）|
| [`NB1-simple-llm-vanilla.ipynb`](notebooks/NB1-simple-llm-vanilla.ipynb) | 碼 | NumPy 從零實作前向傳播，無框架依賴 |
| [`NB2-simple-llm-pytorch.ipynb`](notebooks/NB2-simple-llm-pytorch.ipynb) | 碼 | 同一架構改用 PyTorch `nn.Module` ＋ autograd，對照框架如何簡化實作 |

#### 區 2A：Decoder-only（GPT）★ 主線

| 文件 | 層次 | 說明 |
|---|---|---|
| [`04a-gpt-decoder-only.md`](theory/04a-gpt-decoder-only.md) | 概 | GPT Decoder-Only 的**基本概念、架構差異與 Pipeline 總覽**（前向＋反向一覽；數學細節見 05a1/05a2、05b1/05b2）|
| [`05a1-forward-propagation.md`](theory/05a1-forward-propagation.md) | 數 | **向前傳播數學**：Scaled Dot-Product、Causal Masking、Multi-Head／FFN／Pre-LN、Embedding／PE、Next-token 與 Cross-Entropy；高中數學程度，附記號約定、逐節學習目標與文末「一次 forward 的 11 步」回顧 |
| [`05a2-forward-example.md`](theory/05a2-forward-example.md) | 例 | **前向數值範例**：用一組範例資料（$T=2$、$d=3$）把前向每個階段實際算一次，對照 05a1 各節 |
| [`05b1-backward-propagation.md`](theory/05b1-backward-propagation.md) | 數 | **向後傳播數學**：以 05a1 的**倒序**走一遍（CE→lm_head→LayerNorm→FFN→Multi-Head→Attention→Embedding），高中數學程度，附最小工具箱與逐節學習目標 |
| [`05b2-backward-example.md`](theory/05b2-backward-example.md) | 例 | **後向數值範例**：沿用 05a2 的數字，把反向每個階段的梯度實際算一次，對照 05b1 各節 |
| [`04b-nanogpt-walkthrough.md`](theory/04b-nanogpt-walkthrough.md) | 碼 | **程式對照**：nanoGPT 逐行解析、Pre-LN vs Post-LN、Tokenizer、自迴歸生成與 KV Cache（每節回指 05a1／05b1 數學）|
| [`NB3-llm-backpropagation.ipynb`](notebooks/NB3-llm-backpropagation.ipynb) | 碼 | NumPy 手刻完整反向傳播（含 finite-difference 梯度驗證）|
| [`NB4-nanoGPT.ipynb`](notebooks/NB4-nanoGPT.ipynb) | 碼 | **完整 nanoGPT，訓練莎士比亞文本——本教材的主要目標** |

#### 區 2B：Encoder-only（BERT，選讀分支）

| 文件 | 層次 | 說明 |
|---|---|---|
| [`07-bert-encoder-only.md`](theory/07-bert-encoder-only.md) | 概＋數 | BERT／Encoder-Only：雙向 Self-Attention、MLM 預訓練、`[CLS]`/`[SEP]`、預訓練+微調與遷移學習（含 HuggingFace 微調實作與適用場景）、encoder vs decoder 選型。**與 2A 的差別只有遮罩與訓練目標兩處** |
| [`NB5-bert-mlm.ipynb`](notebooks/NB5-bert-mlm.ipynb) | 碼 | 從零手刻最小 BERT（重用 NB4 元件、去 causal mask ＋ MLM 目標）＋ HuggingFace 選讀延伸 |

#### 區 2C：Encoder-Decoder（Seq2Seq，選讀分支）

| 文件 | 層次 | 說明 |
|---|---|---|
| [`10a1-seq2seq-forward.md`](theory/10a1-seq2seq-forward.md) | 數 | **前向數學**：RNN Seq2Seq＋Bahdanau 加性 attention、Transformer Encoder-Decoder、Cross-Attention、三種 attention 對照（前置只需 `03a`）|
| [`10a2-seq2seq-forward-example.md`](theory/10a2-seq2seq-forward-example.md) | 例 | **前向數值範例**：同一個翻譯任務（`我吃`→`I eat`）兩代架構各完整算一次到 loss |
| [`10b1-seq2seq-backward.md`](theory/10b1-seq2seq-backward.md) | 數 | **後向數學**：BPTT 完整推導、梯度消失的定量說明、Cross-Attention 的梯度分岔 $G^H=G^H\vert_{(K)}+G^H\vert_{(V)}$ |
| [`10b2-seq2seq-backward-example.md`](theory/10b2-seq2seq-backward-example.md) | 例 | **後向數值範例**：沿用 10a2 的兩組數字反推全部參數梯度，量化「attention vs 時間鏈」與「encoder vs decoder」的梯度差距 |

#### 區 3–4：家族演進與系統層應用（選讀）

| 文件 | 層次 | 說明 |
|---|---|---|
| [`08-model-family-tree.md`](theory/08-model-family-tree.md) | 概 | **模型家譜速查**：三大家族的時間軸、代表模型（GPT/LLaMA、BERT/RoBERTa/BGE、T5/BART/Whisper）與各家的**當代主戰場**，附三選一選型判準。純導覽，不推導公式 |
| [`06-modern-transformer-variants.md`](theory/06-modern-transformer-variants.md) | 數 | RMSNorm、SwiGLU、RoPE、GQA、Flash Attention——nanoGPT 到 LLaMA 的橋接（decoder 家族出口）|
| [`09-text-to-vector-rag.md`](theory/09-text-to-vector-rag.md) | 概 | 文字轉向量與語意檢索：分佈假說、Word2Vec、動態 embedding、餘弦相似度、RAG 檢索流程（encoder 分支的應用出口）|

---

### 旁支資源（不在閱讀路線上）

隨時可查，不必依序讀。

#### 進階補充 (`advanced/`)

| 文件 | 對應主線 | 什麼時候讀 |
|---|---|---|
| [`Attention-Mechanism-Part1.md`](advanced/Attention-Mechanism-Part1.md) | `02`、`03a` | 想看第二種講法：從生物學動機與 Nadaraya-Watson 核回歸出發，推到掩蔽／加性／縮放點積注意力 |
| [`Attention-Mechanism-Part2.md`](advanced/Attention-Mechanism-Part2.md) | `03a`、`10a1` §A | 承接 Part1：Bahdanau 注意力、Multi-Head、位置編碼、FFN、殘差與 LayerNorm |
| [`Attention-NW-Kernel-Regression.md`](advanced/Attention-NW-Kernel-Regression.md) | Part1 §10.2 的**深入版**（內容重疊）| 想把注意力理解成「可學習的核平滑器」，看它的非參數統計解釋 |
| [`Seq2Seq-and-Decoding-Techniques.md`](advanced/Seq2Seq-and-Decoding-Techniques.md) | `10a1`（數學）、`04b`（生成）| 想看故事與工藝而非公式：teacher forcing、exposure bias、beam search、NAT、copy、guided attention、BLEU vs CE、RL（整理自李宏毅課程，零公式）|
| [`Transformer-highschool-plain.md`](advanced/Transformer-highschool-plain.md) | `03a` | 高中生版零公式導覽，用圖書館／螢光筆比喻。與 [`03a-transformer-block-plain.md`](theory/03a-transformer-block-plain.md) 定位重疊，是**另一份**白話版 |
| [`Suggested-Papers.md`](advanced/Suggested-Papers.md) | 全部 | 讀完任一區後想追原始論文時 |

#### 附錄與環境 (`appendices/`、`environment/`)

| 檔案 | 用途 |
|---|---|
| [`appendices/trig-reference.html`](appendices/trig-reference.html) | 三角函數參考。讀 `03a` §7 的正弦式位置編碼時可查 |
| [`environment/test.ipynb`](environment/test.ipynb) | 環境檢測：印出 `torch` 版本與 MPS／CUDA 可用性 |

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

本倉庫的理論文件經過多輪系統性檢視與補強，重點如下：

| 輪次 | 重點 |
|---|---|
| 第一輪 | 錯誤修正、數值範例、圖表、章節銜接語 |
| 第二輪 | 主線概念缺口（$W_O$、FFN、PE、Dropout、KV Cache、Embedding 梯度）、新增 `06` 當代架構文件 |
| 第三輪 | 數學推導逐步化（Softmax Jacobian、LayerNorm 合併代數等）、程式範例說明、失效引用修正 |
| 第四輪 | Notebook 執行驗證：NB3 梯度驗證 bug 修復、NB4 首次執行、路徑隔離與 .gitignore 補強 |
| 第五輪 | `05a1`／`05b1` 兩份數學文件改寫為高中數學程度（每節加「讀完這一節，你會」、全文回顧表）|

另有一輪**分類體系重整**：把教材整理成「五個區 × 四種呈現層次」的雙軸結構，重畫路線圖（區 2 的三家族改為平行呈現）、新增 `theory/08` 模型家譜、把 `advanced/` 與 `appendices/` 納入導覽。

各輪的規劃與執行紀錄原存於 `draft/improvement-*.md`，均已完成並移除，內容可從 git 歷史查閱。`draft/` 目前保留 `learning-route-notes.md`、`roadmap-transformer.md`（皆為導覽文件的素材來源）與 `classification-plan.md`（分類體系重整的規劃），都不在主線閱讀路徑上。

尚未完成的延伸方向：把 `06` 文末「下一步」做成可執行出口（新增 NB6 LLaMA 改造實作、`theory/06b` 官方碼對照）、RAG demo notebook、解碼策略 demo；內容缺口待補 BPE／SentencePiece 分詞、MoE、ALiBi、DeBERTa。
