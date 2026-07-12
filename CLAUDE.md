# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 語言與風格
- 對話一律用繁體中文回覆，台灣用語習慣
- 專有技術名詞可保留英文(例如 API、CI/CD)
- 程式碼與註解也用同樣風格
- 回答簡潔直接，不要過度解釋

## 專案概述

從零開始學習 Transformer / Attention 機制的中文學習資源庫，目標是讓讀者能夠獨立實作 nanoGPT。每個概念提供三個層次：直覺說明 → 數學推導 → 程式實作。

## 環境設置

```bash
# 建立獨立環境（Python 3.11）
conda create -n transformer python=3.11 -y
conda activate transformer

# 安裝 PyTorch（純 CPU 將 pytorch-cuda=12.1 換成 cpuonly）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install numpy matplotlib jupyterlab pandas

jupyter lab
```

主要依賴：Python 3.11、PyTorch 2.x、NumPy、JupyterLab（完整安裝說明見 `README.md`）。NB5 主體零額外相依；其附錄「載入 HuggingFace 預訓練 BERT」選讀段才需 `pip install transformers`，且預設 `RUN_HF=False` 不執行。

## 執行 Notebook

```bash
# 啟動互動環境
jupyter lab

# 非互動模式執行單一 notebook
jupyter nbconvert --to notebook --execute "notebooks/NB1-simple-llm-vanilla.ipynb"
```

## 資料夾結構

```
theory/          ← 理論主線（依序閱讀；00 為前言導讀、06 為 decoder 選讀出口、07 為 encoder 選讀分支、09 為 embedding→RAG 應用出口；04a 為 GPT 原理與數學、04b 為 04a 的 nanoGPT 程式對照續篇；03a-transformer-block-plain 為 03a §6 的白話輔助版、03b1→03b2→03b3 為 03a 的選讀計算案例三階段、03b4 為含位置編碼（P≠0）的選讀對照支線）
theory/images/   ← 理論文件內嵌圖檔（03a §5 的 attention_projection_vs_interaction、§5.5 的 multi_head_attention_diagram、§6.1 的 transformer_block_pre_ln_diagram）
notebooks/       ← 實作主線（NB1–NB4）＋選讀分支（NB5 對應 07）
notebooks/data/  ← Notebook 訓練資料（如 NB4／NB5 莎士比亞文本）
advanced/        ← 進階補充（選讀，非主線）
archive/         ← 所有舊版文件備份（不會動到）
environment/     ← 環境檢測 notebook（test.ipynb：驗證 torch / MPS / CUDA；安裝說明見 README.md）
```

## 理論文件（`theory/`）

| 文件 | 說明 |
|---|---|
| `00-learning-path.md` | 學習路線與背景（前言／導讀，無公式）：ML 歷史與歷程、本教材必學與延伸數學分級、最短主線、學完後的自我檢查與後續路線 |
| `01a-prerequisites-intuition.md` | Embedding、Softmax、加權平均（白話版，高中數學程度）|
| `01b-prerequisites-math.md` | 同上，附完整統計推導（大學線性代數程度）|
| `02-attention-intuition.md` | QKV 直覺、翻譯範例逐步計算（I eat fish → 我吃魚）|
| `03a-transformer-architecture.md` | Multi-Head Attention、Transformer Block、Positional Encoding（§2.3／§3.4 QKV 與縮放逐步數值、§5.6 多頭數值範例皆內嵌於本文）|
| `03a-transformer-block-plain.md` | 03a §6 的**白話輔助版**（選讀）：用「開會／整理筆記／改作文」比喻串起 Attention／FFN／Residual／LayerNorm 四模組，零公式；服務零基礎讀者，讀完回 03a §6 讀正式版，與 01a 互連 |
| `03b1-transformer-example-basic.md` | 03a 計算案例・簡單版（選讀）：$2\times4$ 輸入手算單頭 attention（$X\to\tilde X\to C^{(1)}$），不含多頭／FFN |
| `03b2-transformer-example-block.md` | 03a 計算案例・中等版（選讀）：承接 03b1，補上第二頭、$W_O$、殘差、FFN，算到完整 Block 輸出 $Y$ |
| `03b3-transformer-architecture-example.md` | 03a 計算案例・完整版（選讀）：§0 先依前向順序推導每個矩陣的設計歷程（維度咬合、投影＝選欄矩陣、$W_O$ 為可逆基底變換、FFN 形狀鏈），再從頭算整個 Pre-LN Block，含縮放對照與 PE 旋轉驗證，對應 NB1 §13；三份共用同一組數字 |
| `03b4-transformer-example-with-position.md` | 03b 選讀對照支線（純計算展演）：把位置編碼 $P$ 真的加進輸入（$X_{\text{in}}=X+P$，P≠0），沿用 03b3 同一組權重從頭算一次完整 Pre-LN Block，對應 NB1 §13b；不重述概念，數字自成一組（不與 03b1–03b3 共用）|
| `04a-gpt-decoder-only.md` | GPT Decoder-Only **原理與數學**（§1-2 概念、§3 Scaled Dot-Product、§4 Causal Masking、§5 Multi-Head、§6 FFN、§7 LayerNorm/Pre-LN、§8 Embedding/PE、§9 Next-token/CE、§10 梯度鏈）|
| `04b-nanogpt-walkthrough.md` | GPT Decoder-Only **程式對照**（04a 的選讀續篇）：nanoGPT `Head`/`MultiHeadAttention`/`FeedForward`/`Block`/`GPT` 逐行、Pre-LN vs Post-LN、字元級 Tokenizer、自迴歸生成＋KV Cache、速查清單；每節回指 04a 對應數學節 |
| `05-backpropagation.md` | Self-Attention、LayerNorm 與 Embedding 的完整梯度推導 |
| `06-modern-transformer-variants.md` | RMSNorm、SwiGLU、RoPE、GQA、Flash Attention（nanoGPT → LLaMA 橋接，選讀；decoder 家族出口）|
| `07-bert-encoder-only.md` | BERT／Encoder-Only：雙向 Self-Attention（拿掉 Causal Mask）、MLM 預訓練、`[CLS]`/`[SEP]`/三種 embedding、預訓練+微調、BERT 家族、encoder vs decoder 選型（選讀；encoder 家族分支，對應 NB5）|
| `09-text-to-vector-rag.md` | 文字轉向量與語意檢索：分佈假說、Word2Vec（靜態）、Transformer/BERT 動態 embedding、餘弦相似度、RAG 檢索流程（選讀；encoder 分支的應用出口，重疊內容交叉引用 01a/02/03a/07 不重推）|

## Notebook（`notebooks/`）

| Notebook | 說明 | 前置理論 |
|---|---|---|
| `NB1-simple-llm-vanilla.ipynb` | NumPy 從零實作（前向傳播）| 01 + 02 + 03 |
| `NB2-simple-llm-pytorch.ipynb` | PyTorch 版本 | 01 + 02 + 03 |
| `NB3-llm-backpropagation.ipynb` | NumPy 手刻完整反向傳播（含梯度驗證）| 01–03 + 05 |
| `NB4-nanoGPT.ipynb` | 完整 nanoGPT，訓練莎士比亞文本 | 01–03 + 04a/04b |
| `NB5-bert-mlm.ipynb` | 從零手刻最小 BERT（重用 NB4 元件、去 causal mask、MLM 目標）＋ HuggingFace 選讀延伸段（選讀分支）| 01–03 + 07 |

## 核心設計原則

- 所有文件以**繁體中文**撰寫，數學公式用 LaTeX，程式碼用 Python
- 理論文件與 Notebook 相互對應，每份理論文件的開頭都標示對應的 Notebook
- `archive/` 保存所有舊版原始文件，不應修改；新版本在 `theory/` 和 `notebooks/`
- `04a-gpt-decoder-only.md`（原理與數學）＋ `04b-nanogpt-walkthrough.md`（nanoGPT 程式對照）是關鍵橋接文件，連接理論與 nanoGPT 實作；04b 每節回指 04a 的數學節，兩者章節編號需保持對得上

## 行文品質原則（編修理論文件時遵守）

- **推導不跳步**：每個等號的成立理由要能在上下文找到；「整理後得到」「略」「同理可得」不可隱藏非顯然的代數
- **關鍵公式不憑空出現**：當場推導，或明確標注「推導見某文件某節」；斷言（如「梯度趨近於零」）必須附最短可行的數學理由
- **程式碼行行有著落**：教學程式片段每行對應理論公式或有註解，但不加無關的工程細節
- **引用必須有效**：指向的章節、文件必須真實存在；文件內容必須與 Notebook 實際程式碼一致（如本倉庫 NB4 無 Weight Tying、dropout 在三處）

## 改善計劃文件

歷次規劃與草稿統一存放於 `draft/`（不在主線閱讀路徑上，僅供維護參考）：

- `draft/restructure-plan.md` — 倉庫重整計劃草稿（理論／實作／進階／archive 四區結構的原始規劃，已完成）
- `draft/learning-route-notes.md` — 學習路線與未來 AI 趨勢的原始筆記（`theory/00-learning-path.md` 的素材來源）

`draft/improvement-*.md` 為歷次品質改善的規劃與執行紀錄（均已完成，狀態標記在各檔的優先順序表中；檔名編號對應各檔標題的「改善計劃 0X」）：

- `draft/improvement-00-fixes.md` — 第一輪：修正錯誤、補數值範例與圖表、銜接語
- `draft/improvement-01-mainline-gaps.md` — 第二輪：主線缺口（W_O、FFN、PE、Dropout、KV Cache、Embedding 梯度）與新增 `06` 當代架構文件
- `draft/improvement-02-writing.md` — 第三輪：行文清晰度（數學推導補跳步、程式範例說明、失效引用修正）
- `draft/improvement-03-notebooks.md` — 第四輪：Notebook 逐 cell 執行驗證（NB3 梯度 bug 修復、NB4 首次執行、路徑隔離）

（`improvement-00`～`03` 為已完成的歷史存檔，保留供決策脈絡查閱，不再更動。）

**臨時新增內容（非改善輪次，已完成，見 `add-bert-encoder-only` 分支）：**

- `theory/07-bert-encoder-only.md` ＋ `notebooks/NB5-bert-mlm.ipynb` — BERT／Encoder-Only 選讀分支（雙向、MLM、`[CLS]`/`[SEP]`、預訓練+微調；NB5 從零手刻最小 BERT ＋ HuggingFace 選讀延伸）
- `theory/09-text-to-vector-rag.md` — 文字轉向量與語意檢索（分佈假說、Word2Vec、動態 embedding、餘弦、RAG；encoder 分支的應用出口）
- `advanced/Seq2Seq-and-Decoding-Techniques.md` — Seq2Seq 應用脈絡與解碼/訓練工藝（整理自李宏毅課程，延伸閱讀）

**開放中的 backlog：**

- `draft/improvement-04-llama.md` — 第五輪（規劃中）：把 `06` 文末「下一步」做成可執行出口（新增 NB6 改造實作、`theory/08` 官方碼對照；原規劃的 07／NB5 已改給 BERT 選讀分支）
- `draft/improvement-05-current-backlog.md` — 整倉盤點後的當前待辦（C1 RAG demo notebook、~~C2 孤兒檔~~【✅ 已改名 `03a-transformer-block-plain.md` 並納入導覽】、C3 06 補連結 07/09、C4 Notebook 前向連結、C5 解碼策略 demo），並彙整承接自 04 的 LLaMA 出口項目
