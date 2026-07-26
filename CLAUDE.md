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
theory/          ← 理論主線（依五個區閱讀，見下方「分類體系」）
theory/images/   ← 理論文件內嵌圖檔（03a §5 的 attention_projection_vs_interaction、§5.5 的 multi_head_attention_diagram、§6.1 的 transformer_block_pre_ln_diagram）
notebooks/       ← 實作主線（NB1–NB4）＋選讀分支（NB5 對應 07）
notebooks/data/  ← Notebook 訓練資料（如 NB4／NB5 莎士比亞文本）
advanced/        ← 進階補充（旁支資源，非閱讀路線；6 份，README 有定位對照表）
appendices/      ← 附錄工具（trig-reference.html：三角函數參考，供 03a §7 正弦式 PE 查閱）
environment/     ← 環境檢測 notebook（test.ipynb：驗證 torch / MPS / CUDA；安裝說明見 README.md）
draft/           ← 素材與規劃草稿（不在閱讀路線上，僅供維護參考；見「改善計劃與維護紀錄」）
archive/         ← 舊版文件備份，已壓成 old-version.zip 單一檔（不會動到）
```

## 分類體系（雙軸）

教材依兩條軸組織，**新增文件前先確定它落在哪一格**。導覽層在 `README.md`〈學習路線〉與 `theory/00-learning-path.md` §4。

**軸一：五個區（學習階段）**

| 區 | 名稱 | 成員 |
|---|---|---|
| 0 | 導覽 | `00` |
| 1 | 地基：一個 Transformer Block（三家共用，不分流）| `01a`/`01b`、`02`、`03a`、`03a-transformer-block-plain`（§6 白話輔助版）、`03b1`→`03b2`→`03b3`（計算案例三階段）、`03b4`（含位置編碼 P≠0 的對照支線）、NB1、NB2 |
| 2 | 三大家族（三條**平行**支線，非串接）| **2A** decoder-only／GPT＝主線：`04a`（概念與 Pipeline）→`05a1`（前向數學）→`05a2`（前向數值）→`05b1`（後向數學）→`05b2`（後向數值）→`04b`（nanoGPT 程式對照）＋NB3／NB4；**2B** encoder-only／BERT：`07`＋NB5；**2C** encoder-decoder／Seq2Seq：`10a1`→`10a2`→`10b1`→`10b2`（實作待補）|
| 3 | 家族演進與模型家譜 | `08`（三家時間軸與代表模型）、`06`（decoder 當代變體 → LLaMA）、`07` §6（encoder 家族速覽）|
| 4 | 系統層應用 | `09`（文字轉向量 → RAG）|

**軸二：呈現層次** — 同一主題用四種深度各講一次，README 分區表每列都標了層次：

| 層次 | 代表文件 |
|---|---|
| 概念 | `04a`、`03a-transformer-block-plain`、`08`、`09`、`01a`、`02` |
| 符號數學 | `03a`、`05a1`、`05b1`、`10a1`、`10b1`、`06`、`01b` |
| 數值範例 | `03b1`–`03b4`、`05a2`、`05b2`、`10a2`、`10b2` |
| 程式對照 | `04b`、NB1–NB5 |

這條軸解釋了為何 `04b` 的閱讀順序在 `05a2` 之後（程式對照要等數學講完）。**區 2 三支線的深度刻意不對稱**：反向傳播只在 2A 完整推導一次，2B 共用，2C 只推與 2A 不同的部分（BPTT、cross-attention 梯度分岔）——這一點要在導覽層明講，否則 2B 只有一份文件會被誤認為漏寫。

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
| `04a-gpt-decoder-only.md` | GPT Decoder-Only **基本概念、架構差異與 Pipeline 總覽**（§1 Encoder-Decoder、§2 為何只要 Decoder、完整 Pipeline 前向＋反向一覽；數學細節分流到 05a1/05a2、05b1/05b2）|
| `05a1-forward-propagation.md` | GPT **向前傳播數學（符號）**（§1 Scaled Dot-Product、§2 Causal Masking、§3 Multi-Head、§4 FFN、§5 LayerNorm/Pre-LN、§6 Embedding/PE、§7 Next-token/CE；由原 04a §3–§9 搬移而來）。**寫作層級為高中數學程度**：§1 前有不編號的「開始之前：三個記號約定」、每節末有「讀完這一節，你會」、文末有不編號的「全文回顧：七節怎麼串成一次 forward」11 步表；**§1–§7 與所有子節號未動**（被 04b／05b1／03a／05a2 大量硬引用）|
| `05a2-forward-example.md` | GPT **前向數值範例**：一組範例資料（T=2、d=3、單頭、含因果遮罩）逐階段 embedding→…→CE loss 實算，對照 05a1 各節；與 05b2 共用同一組數字（勿改動數值）|
| `04b-nanogpt-walkthrough.md` | GPT Decoder-Only **程式對照**（04a 的選讀續篇）：nanoGPT `Head`/`MultiHeadAttention`/`FeedForward`/`Block`/`GPT` 逐行、Pre-LN vs Post-LN、字元級 Tokenizer、自迴歸生成＋KV Cache、速查清單；每節回指 05a1／05b1 對應數學節 |
| `05b1-backward-propagation.md` | GPT **向後傳播數學（符號）**：**章節順序＝05a1 的倒序**（§3 CE+Softmax ↔ 05a1 §7、§4 lm_head／線性層通則、§5 LayerNorm+Residual ↔ 05a1 §5、§6 FFN ↔ 05a1 §4、§7 Multi-Head ↔ 05a1 §3、§8 Attention+因果遮罩 ↔ 05a1 §1/§2、§9 QKV 投影、§10 Embedding+PE ↔ 05a1 §6、§11 optimizer；§1 最小數學工具箱、§2 五步總覽與鏡射表、附錄 A 查閱表、附錄 B RNN 對比）。**寫作層級為高中數學程度**，每章末有「讀完這一節，你會」；數值範例在 05b2 |
| `05b2-backward-example.md` | GPT **後向數值範例**：沿用 05a2 的前向數字，逐階段 logits→…→embedding 梯度實算，對照 05b1 各節（LayerNorm 在 d=2 反向會全歸零，故範例用 d=3）|
| `06-modern-transformer-variants.md` | RMSNorm、SwiGLU、RoPE、GQA、Flash Attention（nanoGPT → LLaMA 橋接，選讀；decoder 家族出口）|
| `08-model-family-tree.md` | **模型家譜速查**（區 3，選讀，純導覽，不推導公式）：§1 時間軸（2014 Seq2Seq→2017 Transformer→2018 分家→2019-20 各家優化→2023- 開源配方）、§2 三張家譜表（2.1 decoder：GPT-1/2/3→LLaMA/Mistral/Qwen；2.2 encoder：BERT→RoBERTa/ALBERT/DistilBERT/ELECTRA/DeBERTa→SBERT→BGE/E5/gte；2.3 encoder-decoder：RNN+Bahdanau→原始 Transformer→T5/BART→M2M-100/NLLB/Whisper）、§3 每一家的當代主戰場、§4 三選一選型速查、§5 下一步。與 `06`／`07` §6／`10a1` 的分工：那三處講技術差異，本文講時間順序與應用落點 |
| `07-bert-encoder-only.md` | BERT／Encoder-Only：雙向 Self-Attention（拿掉 Causal Mask）、MLM 預訓練（§2.1 附三種 mask 對照）、`[CLS]`/`[SEP]`/三種 embedding、§5 預訓練+微調＝遷移學習（§5.1 遷移學習 vs 微調、§5.4 HuggingFace 微調程式、§5.5 適用場景三判準）、BERT 家族、encoder vs decoder 選型（選讀；encoder 家族分支，對應 NB5）|
| `09-text-to-vector-rag.md` | 文字轉向量與語意檢索：分佈假說、Word2Vec（靜態）、Transformer/BERT 動態 embedding、餘弦相似度、RAG 檢索流程（選讀；encoder 分支的應用出口，重疊內容交叉引用 01a/02/03a/07 不重推）|
| `10a1-seq2seq-forward.md` | Seq2Seq **前向數學（符號）**（選讀；encoder-decoder 家族分支，前置只需 01–03a）：§A RNN Seq2Seq＋Bahdanau 加性 attention（含固定 context vector 的瓶頸）、§B Transformer Encoder-Decoder（§B3 Cross-Attention、§B4 三種 attention 一表對照）、§C 兩代對照、§D 四份文件的節號鏡射表 |
| `10a2-seq2seq-forward-example.md` | Seq2Seq **前向數值範例**：同一個翻譯任務（`我吃`→`I eat`，teacher forcing、target 右移一位）兩代架構各完整算一次到 loss。模型 A：$d_e=d_h=d_a=2$、14 張量 55 參數、$L=1.945$；模型 B：$d=3$、$T_s=T_t=2$、單頭、各一層 Pre-LN Block、32 張量 234 參數、$L=2.250$。模型 B 的 encoder 輸入刻意與 05a2 的 $x_0$ 相同（唯一差別是無因果遮罩）；$W_V^d$ 刻意非 $I$ 以免 LN 退化 |
| `10b1-seq2seq-backward.md` | Seq2Seq **後向數學（符號）**：§A BPTT（§A3 $G^{h_i}$ 的時間鏈／attention 兩股分解、§A5 參數共享⇒累加、§A6 梯度消失的定量推導）、§B Transformer 反向（§B2 Cross-Attention 梯度分岔 $G^H=G^H\vert_{(K)}+G^H\vert_{(V)}$、§B5 反向五步總覽）、§C 兩代對照 |
| `10b2-seq2seq-backward-example.md` | Seq2Seq **後向數值範例**：沿用 10a2 的兩組數字，14＋32 個參數梯度全部實算＋一步 SGD 示範；§D 三個量化結論（attention 那股是時間鏈的 1.48／2.90 倍、零梯度成因表、encoder 梯度比 decoder 小約 9 倍）|

## Notebook（`notebooks/`）

| Notebook | 說明 | 前置理論 |
|---|---|---|
| `NB1-simple-llm-vanilla.ipynb` | NumPy 從零實作（前向傳播）| 01 + 02 + 03 |
| `NB2-simple-llm-pytorch.ipynb` | PyTorch 版本 | 01 + 02 + 03 |
| `NB3-llm-backpropagation.ipynb` | NumPy 手刻完整反向傳播（含梯度驗證）| 01–03 + 05b1 |
| `NB4-nanoGPT.ipynb` | 完整 nanoGPT，訓練莎士比亞文本 | 01–03 + 04a/05a1/04b |
| `NB5-bert-mlm.ipynb` | 從零手刻最小 BERT（重用 NB4 元件、去 causal mask、MLM 目標）＋ HuggingFace 選讀延伸段（選讀分支）| 01–03 + 07 |

## 核心設計原則

- 所有文件以**繁體中文**撰寫，數學公式用 LaTeX，程式碼用 Python
- 理論文件與 Notebook 相互對應，每份理論文件的開頭都標示對應的 Notebook
- **分類體系是雙軸的**（五個區 × 呈現層次，見上方「分類體系」）。新增文件時先確定它落在哪一格，並同步 `README.md` 的分區表（含層次欄）與 `theory/00-learning-path.md` §4。分區只是導覽層的呈現，**不改變既有節號互鎖規則**（見下兩條）
- 導覽層要維持三件事的一致性：(1) 區 2 的三家族是**平行**關係，路線圖不可用 `↓` 串接；(2) 三支線深度不對稱要明講；(3) 每份文件在 README 分區表中只出現一次（Notebook 也編在所屬區內，沒有獨立的 notebook 清單）
- `archive/old-version.zip` 保存所有舊版原始文件（已封存為壓縮檔），不應修改；新版本在 `theory/` 和 `notebooks/`
- `04a-gpt-decoder-only.md`（基本概念與 Pipeline）＋ `05a1-forward-propagation.md`（前向數學符號）＋ `05a2-forward-example.md`（前向數值範例）＋ `05b1-backward-propagation.md`（後向數學符號）＋ `05b2-backward-example.md`（後向數值範例）＋ `04b-nanogpt-walkthrough.md`（nanoGPT 程式對照）是關鍵橋接文件，連接理論與 nanoGPT 實作；05b1 的章節順序是 05a1 的倒序（對照表在 05b1 §2.2），改任一邊的節號要同步另一邊與那張表；05b1 的節號另被 05b2、10b1、04b、01a／01b、03a 硬引用，改號時要一併更新；04b 每節回指 05a1／05b1 的數學節，改章節號時兩邊要同步；05a2 與 05b2 共用同一組 T=2/d=3 數字（改一邊要同步另一邊）；04a 的 Pipeline 總覽節號亦指向 05a1（前向）／05b1（反向）
- `10a1`／`10a2`／`10b1`／`10b2` 是 Seq2Seq 四件套，**四份的節號互為鏡射**（10a1 §D、10a2 §0.1 各有一張對照表）：§A1–§A5 是 RNN 版、§B1–§B4 是 Transformer 版，改任一份的節號要同步四份與那兩張表。10a2 與 10b2 共用同兩組數字（模型 A 55 參數、模型 B 234 參數），**改一邊要同步另一邊**；所有數值都以 PyTorch autograd 驗證過，勿手動調整

## 行文品質原則（編修理論文件時遵守）

- **推導不跳步**：每個等號的成立理由要能在上下文找到；「整理後得到」「略」「同理可得」不可隱藏非顯然的代數
- **關鍵公式不憑空出現**：當場推導，或明確標注「推導見某文件某節」；斷言（如「梯度趨近於零」）必須附最短可行的數學理由
- **程式碼行行有著落**：教學程式片段每行對應理論公式或有註解，但不加無關的工程細節
- **引用必須有效**：指向的章節、文件必須真實存在；文件內容必須與 Notebook 實際程式碼一致（如本倉庫 NB4 無 Weight Tying、dropout 在三處）

## 改善計劃與維護紀錄

`draft/` 有三份文件（都不在主線閱讀路徑上，僅供維護參考；**素材檔一律保留，不因內容已整合而刪除**）：

- `draft/learning-route-notes.md` — 學習路線與未來 AI 趨勢的原始筆記（`theory/00-learning-path.md` §1／§3 難度分級與數學 Level 1–5 的素材來源）
- `draft/roadmap-transformer.md` — 三分流架構與各家當代銜接的原始筆記（`theory/08` 與 `00` §4.3「當代主戰場」的素材來源）
- `draft/classification-plan.md` — 分類體系重整（第六輪）的規劃文件，已執行完畢

歷次規劃文件（`draft/restructure-plan.md`、`draft/improvement-00`～`05`）都已執行完畢並自倉庫移除，需要決策脈絡時從 git 歷史查閱。各輪重點：

- 第一輪：修正錯誤、補數值範例與圖表、銜接語
- 第二輪：主線缺口（W_O、FFN、PE、Dropout、KV Cache、Embedding 梯度）與新增 `06` 當代架構文件
- 第三輪：行文清晰度（數學推導補跳步、程式範例說明、失效引用修正）
- 第四輪：Notebook 逐 cell 執行驗證（NB3 梯度 bug 修復、NB4 首次執行、路徑隔離）
- 第五輪：`05a1`／`05b1` 改寫為高中數學程度（加「讀完這一節，你會」與全文回顧表，節號未動）
- 第六輪：分類體系重整（規劃見 `draft/classification-plan.md`）——建立「五個區 × 呈現層次」雙軸、README 路線圖改為區 2 三家族並列（原本被 `↓` 串成直線，且 `10a1`–`10b2` 完全缺席）、文件清單改為分區表＋層次欄、新增 `theory/08` 模型家譜、`advanced/`＋`appendices/` 納入導覽、`Transformer-in-Nushell.md` 改名為 `Transformer-highschool-plain.md`（檔名與內容不符）。**既有 22 份理論文件與 5 份 Notebook 的檔名、節號、內文全部未動**
- 另有倉庫重整計劃（理論／實作／進階／archive 四區結構）已完成

**臨時新增內容（非改善輪次，已完成，見 `add-bert-encoder-only` 分支）：**

- `theory/07-bert-encoder-only.md` ＋ `notebooks/NB5-bert-mlm.ipynb` — BERT／Encoder-Only 選讀分支（雙向、MLM、`[CLS]`/`[SEP]`、預訓練+微調；NB5 從零手刻最小 BERT ＋ HuggingFace 選讀延伸）
- `theory/09-text-to-vector-rag.md` — 文字轉向量與語意檢索（分佈假說、Word2Vec、動態 embedding、餘弦、RAG；encoder 分支的應用出口）
- `advanced/Seq2Seq-and-Decoding-Techniques.md` — Seq2Seq 應用脈絡與解碼/訓練工藝（整理自李宏毅課程，延伸閱讀）

**開放中的 backlog**（原記於已刪除的 `draft/improvement-04-llama.md`／`improvement-05-current-backlog.md`，內容移錄於此）：

- LLaMA 出口：把 `06` 文末「下一步」做成可執行出口 — 新增 NB6 改造實作、`theory/06b-llama-walkthrough.md` 官方碼對照（`a` = 概念／數學、`b` = 程式對照，比照 `04a`／`04b`；**原本規劃佔用 `08`，該編號已改給模型家譜**）
- C1 — RAG demo notebook（承接 `09`）
- C4 — Notebook 補前向連結（指回對應理論文件）
- C5 — 解碼策略 demo（承接 `advanced/Seq2Seq-and-Decoding-Techniques.md`）
- （C2 孤兒檔已完成：改名為 `03a-transformer-block-plain.md` 並納入導覽）
- （C3 已由 `08` 的家譜表覆蓋：三家族互連的導覽需求已滿足；若仍想在 `06` 文末直接加 `07`／`09` 連結，屬 nice-to-have）

**內容缺口**（第六輪比對 `draft/roadmap-transformer.md` 後確認，尚未實作）：

| 缺口 | 現況 | 建議落點 | 優先 |
|---|---|---|---|
| Tokenization（BPE／SentencePiece）| 只有 `04b` 的字元級 tokenizer，沒講子詞切分；但 `07` §5.4 已在講「tokenizer 必須與預訓練配對」，讀者會卡在這裡 | `04b` 新增一節，或獨立成 `theory/04c` | 高 |
| MoE（Mixture of Experts）| `06` 完全沒提，是當代 decoder 架構最明顯的缺口（Mixtral、DeepSeek、Qwen-MoE 都用）| `06` 新增一節，與 SwiGLU 並列在 FFN 段 | 高 |
| ALiBi | `06` §3 只講 RoPE，另一條長上下文路線缺對照 | `06` §3 補一段 | 低 |
| DeBERTa | `07` §6 家族表缺這一列（解耦注意力＋相對位置編碼）；`08` §2.2 已列入 | `07` §6 表格補一列 | 低 |
