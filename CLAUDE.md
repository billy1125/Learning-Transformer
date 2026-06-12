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
theory/          ← 理論主線（7 份文件，依序閱讀；06 為選讀出口）
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
| `03-transformer-architecture.md` | Multi-Head Attention、Transformer Block、Positional Encoding |
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
