# 倉庫重整計劃草稿

> 目標：將現有學習內容由淺到深重新整理，讓讀者能從零出發，最終獨立實作 nanoGPT。
> 保留兩份前置知識文件（分別服務不同程度的讀者）。
> 舊版文件統一備份至 `archive/`，不刪除。

---

## 最終目標結構

```
Learning-Transformer/
│
├── README.md                          ← 入口，含完整學習路線圖
│
├── 理論/
│   ├── 01a-前置知識-直覺版.md          ← 改自 Pre Knowledge Explain.md
│   ├── 01b-前置知識-數學版.md          ← 改自 Pre Knowledge Basic Math.md
│   ├── 02-注意力的直覺.md              ← 改自 Transformer Attention.md
│   ├── 03-Transformer架構.md          ← 改自 Transformer.md（移除反向傳播節）
│   ├── 04-GPT-Decoder-Only.md         ← 新建（橋接理論到 nanoGPT）
│   └── 05-反向傳播推導.md              ← 從 Transformer.md §9–§13 拆出
│
├── 實作/
│   ├── NB1-NumPy-前向傳播.ipynb        ← 改自 Simple LLM Vanilla.ipynb
│   ├── NB2-PyTorch-版本.ipynb          ← 改自 Simple LLM Pytorch.ipynb
│   ├── NB3-NumPy-完整反向傳播.ipynb    ← 改自 LLM Backpropagation.ipynb
│   └── NB4-nanoGPT.ipynb              ← 改自 nanoGPT.ipynb
│
├── 進階補充/（選讀，不在主線上）
│   ├── Attention-Mechanism-Part1.md
│   ├── Attention-Mechanism-Part2.md
│   ├── Attention-NW-Kernel-Regression.md
│   ├── Transformer-in-Nushell.md
│   └── Suggested-Papers.md
│
├── archive/                           ← 備份所有舊版原始文件
│
├── environment/                       ← 不動
└── CLAUDE.md                          ← 最後更新
```

---

## 理論與實作的對應關係

| 理論文件 | 對應 Notebook | 學習目標 |
|---|---|---|
| 01a / 01b 前置知識 | NB1 §1–§4（Embedding、Softmax） | 理解文字如何變成向量，能做加權平均 |
| 02 注意力的直覺 | NB1 §5（SelfAttention 類別） | 理解 attention 在「做什麼」 |
| 03 Transformer 架構 | NB1 §6–§8、NB2 完整模型 | 理解每個元件（QKV、Multi-Head、Block） |
| 04 GPT Decoder-Only | NB4 §4（Head / Block / GPT 類別） | 理解 Causal Mask、next-token prediction |
| 05 反向傳播推導 | NB3 每個 `.backward()` 方法 | 能手推梯度，理解訓練為何可行 |

---

## 六個工作階段

### 階段一：備份 + 建立新資料夾結構
**工作內容：**
- 建立 `archive/` 資料夾，複製所有現有 `.md` 和 `.ipynb` 至此（保留原檔）
- 建立 `理論/`、`實作/`、`進階補充/` 三個資料夾
- 將進階補充文件移入 `進階補充/`（Attention Mechanism Part 1 & 2、NW Kernel Regression、Transformer in Nushell、Suggested Papers）

**驗收條件：**
- `archive/` 有完整備份
- 資料夾結構存在

---

### 階段二：README.md 重建
**工作內容：**
- 完整重寫 `README.md`（現在只有兩行）
- 內容包含：
  - 這個倉庫是什麼、適合誰
  - 讀者前置需求（微積分、線性代數基本概念）
  - 完整學習路線圖（含理論↔實作對應表）
  - 環境安裝說明（指向 `environment/`）
  - 兩條路線說明：「只看直覺版」 vs「完整數學版」

**驗收條件：**
- README.md 可作為新讀者的唯一入口

---

### 階段三：前置知識文件重整（兩份保留）
**工作內容：**
- `Pre Knowledge Explain.md` → `理論/01a-前置知識-直覺版.md`
  - 在文件頂部加上「適合對象」與「讀完後你能做什麼」說明
  - 在文件底部加上「下一步」（指向 02）
- `Pre Knowledge Basic Math.md` → `理論/01b-前置知識-數學版.md`
  - 同上，加首尾導航
  - 保持原有數學嚴謹度，不修改內文

**驗收條件：**
- 兩份文件都有明確的適合對象說明
- 都有指向下一份文件的導航

---

### 階段四：注意力直覺 + Transformer 架構文件重整
**工作內容（兩份一起處理）：**

**4a — `Transformer Attention.md` → `理論/02-注意力的直覺.md`**
- 加首尾導航（來自 01a/01b，下到 03）
- 確認翻譯範例（I eat fish → 我吃魚）仍清晰完整

**4b — `Transformer.md` → 拆成兩份**
- 架構部分（§1–§8：QKV 動機、Scaled Dot-Product、Multi-Head、Block、Positional Encoding、RNN 比較）→ `理論/03-Transformer架構.md`
- 反向傳播部分（§9–§13）→ `理論/05-反向傳播推導.md`
- 兩份都加首尾導航

**驗收條件：**
- `03-Transformer架構.md` 不含反向傳播內容
- `05-反向傳播推導.md` 開頭有「需先讀 03」的說明

---

### 階段五：新建 GPT Decoder-Only 橋接文件
**工作內容：**
- 新建 `理論/04-GPT-Decoder-Only.md`
- 內容結構：
  1. 原始 Transformer 是 encoder-decoder，GPT 為何只要 decoder？
  2. Causal Masking：下三角矩陣遮罩的動機與實作
  3. Next-token Prediction 作為訓練目標（語言模型的本質）
  4. nanoGPT 架構解析：`Head` / `MultiHeadAttention` / `Block` / `GPT` 類別與理論的逐一對應
  5. 字元級 Tokenizer vs. 詞級 Tokenizer
  6. 「打開 NB4-nanoGPT.ipynb 之前，你需要知道的事」

**驗收條件：**
- 讀完此文件，直接打開 `nanoGPT.ipynb` 不會有斷層感

---

### 階段六：Notebook 重命名 + CLAUDE.md 更新
**工作內容：**
- 在每個 notebook 的第一個 Markdown cell 加上「前置需求」與「本 notebook 目標」說明
- 重命名四個 notebook（加上 NB1–NB4 前綴）
- 更新 `CLAUDE.md`，反映新的資料夾結構、文件名稱、學習路徑

**驗收條件：**
- CLAUDE.md 的架構說明與實際目錄一致

---

## 新建內容工作量估計

| 文件 | 性質 | 預估字數 | 備註 |
|---|---|---|---|
| README.md | 全新撰寫 | ~800 字 | 含路線圖表格 |
| `04-GPT-Decoder-Only.md` | 全新撰寫 | ~3000 字 | 最多新內容，含數學與程式碼片段 |
| 各文件首尾導航 | 小幅修改 | ~100 字/份 | 重複性工作 |

---

## 不在這次計劃內的工作

- 修改任何 notebook 的程式碼內容（只加說明 cell）
- 修改 `Attention Mechanism Part 1 & 2` 或 `NW Kernel Regression` 的內文
- 加入新的數學推導（除了 `04-GPT-Decoder-Only.md`）
