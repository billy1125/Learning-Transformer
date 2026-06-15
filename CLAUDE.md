# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

從零開始學習 Transformer / Attention 機制的中文學習資源庫，目標是讓讀者能夠獨立實作 nanoGPT。每個概念提供三個層次：直覺說明 → 數學推導 → 程式實作。

## 環境設置

```bash
# macOS
conda env create -f environment/cona-env-install-mac.yml
# Windows
conda env create -f environment/cona-env-install-pc.yml

conda activate transformer
jupyter lab
```

主要依賴：Python 3.11、PyTorch 2.x、NumPy、JupyterLab。

## 執行 Notebook

```bash
# 啟動互動環境
jupyter lab

# 非互動模式執行單一 notebook
jupyter nbconvert --to notebook --execute "notebooks/NB1-simple-llm-vanilla.ipynb"
```

## 資料夾結構

```
theory/          ← 理論主線（7 份文件，依序閱讀；06 為選讀出口；03b 為 03a 的選讀計算案例）
notebooks/       ← 實作主線（4 個 Notebook，依序執行）
advanced/        ← 進階補充（選讀，非主線）
archive/         ← 所有舊版文件備份（不會動到）
environment/     ← Conda 環境設定檔
```

## 理論文件（`theory/`）

| 文件 | 說明 |
|---|---|
| `01a-prerequisites-intuition.md` | Embedding、Softmax、加權平均（白話版，高中數學程度）|
| `01b-prerequisites-math.md` | 同上，附完整統計推導（大學線性代數程度）|
| `02-attention-intuition.md` | QKV 直覺、翻譯範例逐步計算（I eat fish → 我吃魚）|
| `03a-transformer-architecture.md` | Multi-Head Attention、Transformer Block、Positional Encoding（§2.3／§3.4 QKV 與縮放逐步數值、§5.6 多頭數值範例皆內嵌於本文）|
| `03b-transformer-architecture-example.md` | 03a 的計算案例（選讀）：$2\times4$ 輸入手算整個 Pre-LN Block，對應 NB1 §13 |
| `04-gpt-decoder-only.md` | Causal Masking、GPT Decoder-Only 架構、nanoGPT 逐行解析 |
| `05-backpropagation.md` | Self-Attention、LayerNorm 與 Embedding 的完整梯度推導 |
| `06-modern-transformer-variants.md` | RMSNorm、SwiGLU、RoPE、GQA、Flash Attention（nanoGPT → LLaMA 橋接，選讀）|

## Notebook（`notebooks/`）

| Notebook | 說明 | 前置理論 |
|---|---|---|
| `NB1-simple-llm-vanilla.ipynb` | NumPy 從零實作（前向傳播）| 01 + 02 + 03 |
| `NB2-simple-llm-pytorch.ipynb` | PyTorch 版本 | 01 + 02 + 03 |
| `NB3-llm-backpropagation.ipynb` | NumPy 手刻完整反向傳播（含梯度驗證）| 01–03 + 05 |
| `NB4-nanoGPT.ipynb` | 完整 nanoGPT，訓練莎士比亞文本 | 01–04 |

## 核心設計原則

- 所有文件以**繁體中文**撰寫，數學公式用 LaTeX，程式碼用 Python
- 理論文件與 Notebook 相互對應，每份理論文件的開頭都標示對應的 Notebook
- `archive/` 保存所有舊版原始文件，不應修改；新版本在 `theory/` 和 `notebooks/`
- `04-gpt-decoder-only.md` 是關鍵橋接文件，連接理論與 nanoGPT 實作

## 行文品質原則（編修理論文件時遵守）

- **推導不跳步**：每個等號的成立理由要能在上下文找到；「整理後得到」「略」「同理可得」不可隱藏非顯然的代數
- **關鍵公式不憑空出現**：當場推導，或明確標注「推導見某文件某節」；斷言（如「梯度趨近於零」）必須附最短可行的數學理由
- **程式碼行行有著落**：教學程式片段每行對應理論公式或有註解，但不加無關的工程細節
- **引用必須有效**：指向的章節、文件必須真實存在；文件內容必須與 Notebook 實際程式碼一致（如本倉庫 NB4 無 Weight Tying、dropout 在三處）

## 改善計劃文件

歷次規劃與草稿統一存放於 `draft/`（不在主線閱讀路徑上，僅供維護參考）：

- `draft/restructure-plan.md` — 倉庫重整計劃草稿（理論／實作／進階／archive 四區結構的原始規劃，已完成）

`draft/improvement-*.md` 為歷次品質改善的規劃與執行紀錄（均已完成，狀態標記在各檔的優先順序表中；檔名編號對應各檔標題的「改善計劃 0X」）：

- `draft/improvement-00-fixes.md` — 第一輪：修正錯誤、補數值範例與圖表、銜接語
- `draft/improvement-01-mainline-gaps.md` — 第二輪：主線缺口（W_O、FFN、PE、Dropout、KV Cache、Embedding 梯度）與新增 `06` 當代架構文件
- `draft/improvement-02-writing.md` — 第三輪：行文清晰度（數學推導補跳步、程式範例說明、失效引用修正）
- `draft/improvement-03-notebooks.md` — 第四輪：Notebook 逐 cell 執行驗證（NB3 梯度 bug 修復、NB4 首次執行、路徑隔離）
- `draft/improvement-04-llama.md` — 第五輪（規劃中）：把 `06` 文末「下一步」做成可執行出口（新增 NB5 改造實作、`theory/07` 官方碼對照）
