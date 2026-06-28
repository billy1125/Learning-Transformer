# 00｜學習路線與背景：在開始之前

> **適合對象：** 準備開始、或學到一半想確認方向的讀者。
>
> **讀完後你會知道：**
> - Transformer 在機器學習發展史上的位置，以及它為什麼算「進階」
> - 這份教材**會用到**哪些數學（必學），又有哪些**值得延伸**（選學）
> - 整個倉庫的最短主線與選讀分支
> - 學完後如何自我檢查，以及下一步往哪走
>
> **這不是理論章節**，沒有公式推導；正式內容請從 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)（直覺版）或 [`01b-prerequisites-math.md`](01b-prerequisites-math.md)（數學版）開始。

---

## 1. Transformer 在哪裡？——機器學習的歷史與歷程

要理解 Transformer 為什麼長這樣，先看它解決了前人沒解決的什麼問題。下面是一條精簡的脈絡：

| 年代 | 階段 | 代表 | 解決／留下的問題 |
|---|---|---|---|
| 1958–1980s | 早期類神經與符號 AI | 感知機（Perceptron）、規則式系統 | 單層無法處理非線性；規則難以擴展 |
| 1986 | 反向傳播普及 | 多層感知機（MLP）| 能訓練多層網路，但不擅長處理「結構化輸入」|
| 1998 / 1997 | 專用結構登場 | CNN（影像）、RNN／LSTM（序列）| CNN 抓空間局部性；RNN 處理序列，但**逐步計算、長距離依賴會衰減** |
| 2013–2014 | 表示與對齊 | word2vec（Embedding）、seq2seq + Attention | 詞被映射成向量；Attention 讓 decoder 能「對齊」到輸入任意位置 |
| **2017** | **Transformer** | *Attention Is All You Need* | **完全用 Attention 取代遞迴**：可並行、長距離依賴變成 $O(1)$ 路徑 |
| 2018–2020 | 預訓練範式 | BERT（Encoder）、GPT（Decoder）| 「大規模預訓練 + 下游微調」成為主流 |
| 2022– | 對齊與指令 | ChatGPT、GPT-4（RLHF、指令遵循）| 讓模型「聽得懂指令、答得有用」|
| 2024– | 推理與行動 | o 系列、DeepSeek-R1、Agent、Multimodal | 多步推理、自我檢查、工具使用、跨模態 |

**難度階梯**（相對而言，幫你定位這趟學習的位置）：

```
入門  MLP ★ ─ CNN ★★ ─ Autoencoder ★★
中階  RNN / LSTM / GRU / ResNet ★★★
進階  U-Net / Transformer ★★★★ / ViT ★★★★      ← 你在這裡
高階  VAE / GAN ★★★★★
```

Transformer 之所以是 ★★★★，不是因為單一公式有多難，而是它**同時疊了好幾個概念**（QKV、多頭、位置編碼、殘差、正規化）。本教材的策略就是把這些**逐一拆開**，每個都先給直覺、再給數學、最後動手實作。

---

## 2. 這個專案會用到的數學（必學）

卡關的往往不是數學太深，而是不知道「哪些才必要」。**本教材只預設你具備以下 Level 1**，其餘在文中即用即補。

| 概念 | 在 Transformer 裡的角色 | 本教材對應 |
|---|---|---|
| **向量 / Embedding** | token 變成可運算的向量 | [`01a`](01a-prerequisites-intuition.md) / [`01b`](01b-prerequisites-math.md) §1 |
| **矩陣乘法** | QKV 投影、注意力分數、FFN 全靠它 | `01a` / `01b`、[`03a`](03a-transformer-architecture.md) §4 Shape 分析 |
| **Dot Product（內積）** | 衡量兩個向量的相似度＝注意力分數 | [`02`](02-attention-intuition.md)、`03a` §3 |
| **Softmax** | 把分數變成一組和為 1 的權重 | `01a` / `01b`、`03a` §3 |
| **Gradient（梯度）** | 訓練＝沿梯度更新參數 | [`05`](05-backpropagation.md) |

> 這五項就是入場券。只要看得懂「矩陣相乘」和「對一排數字取 softmax」，就能開始讀 `01` → `02` → `03a`。

---

## 3. 值得延伸的數學（選學）

學完主線後，下面這些能讓你讀懂更進階的模型與論文。**不必在開始前先學**，遇到再回來補即可。依優先順序分層：

### Level 2：進階（建議學）

| 主題 | 內容 | 之後用在哪 |
|---|---|---|
| 線性代數 | Projection、Basis、Linear Transformation | 理解 QKV「投影到子空間」的本質 |
| 機率 | 機率分布、期望值 | `03a` §3.4 縮放的統計推導、[`01b`](01b-prerequisites-math.md) |
| 資訊理論 | Entropy、Cross Entropy | 語言模型的損失函數（[`04`](04-gpt-decoder-only.md)）|
| 特徵值 / 特徵向量 | PCA 基礎 | 理解降維與表示空間 |

### Level 3：旁支應用

- **Cosine 相似度** → Embedding 檢索、RAG
- **最佳化**：SGD、Momentum、Adam → 實際訓練（NB2、NB4）
- **SVD** → PCA、LoRA（參數高效微調）
- **數值穩定性** → Softmax overflow、浮點誤差（`03a` §3.4 的實作注意、[`01b`](01b-prerequisites-math.md) §4.3）

### Level 4：高階（進入生成模型 / RL 才需要）

- **KL Divergence** → VAE、RLHF、PPO、DPO
- **貝氏理論** → 條件機率、不確定性估計
- **馬可夫過程** → 強化學習、Agent
- **凸優化** → Loss surface 分析

### Level 5：通常可跳過

測度論、泛函分析、實分析、微分幾何、證明導向統計（大數法則／中央極限定理的證明）——**實務上知道結論即可**，不影響理解與實作。

### 數學學習優先順序（濃縮）

```
第一優先  向量 → 矩陣 → Dot Product → Softmax → Gradient   （= 本教材必學）
第二優先  線性代數 → 機率 → Cross Entropy
第三優先  Cosine 相似度 → Adam → SVD
第四優先  KL Divergence → 貝氏 → 馬可夫過程
```

---

## 4. 這個專案的學習路徑

**最短主線**（理論與實作交錯進行）：

```
01 前置數學        →  02 Attention 直覺  →  03a 架構       →  04 GPT        →  實作
(向量/softmax/梯度)   (QKV 翻譯範例)        (多頭/Block/PE)    (Causal Mask)    NB1→NB2→NB4
```

**選讀深入**（想算得更細或推得更深時再走）：

- [`03b3`](03b3-transformer-architecture-example.md)：用 $2\times4$ 輸入手算整個 Pre-LN Block（對應 NB1 §13）
- [`05`](05-backpropagation.md)：Self-Attention／LayerNorm／Embedding 的完整梯度推導（對應 NB3）
- [`06`](06-modern-transformer-variants.md)：RMSNorm、SwiGLU、RoPE、GQA——nanoGPT 到 LLaMA 的橋接

> 完整的「理論 ↔ Notebook」對應表，以及兩個起點（直覺版／數學版）的選擇，見 [`../README.md`](../README.md) 的〈學習路線〉。

---

## 5. 學完之後：自我檢查與下一步

### 先確認自己能回答

學完主線後，試著**用自己的話**回答下列問題。答得出來，代表你真的懂了，而不只是「聽過 Attention」：

| 問題 | 對應章節 |
|---|---|
| Q、K、V 如何產生？為什麼要分成三組？ | [`02`](02-attention-intuition.md)、[`03a`](03a-transformer-architecture.md) §1–§2 |
| 為什麼 Self-Attention 要算 $QK^\top$、除以 $\sqrt{d_k}$、再 Softmax？ | `03a` §3 |
| 為什麼需要 Multi-Head？比單頭好在哪？ | `03a` §5 |
| Transformer 本身沒有順序概念，位置資訊怎麼加入？ | `03a` §7 |
| 為什麼需要 Residual Connection 與 LayerNorm？ | `03a` §6 |
| GPT（Decoder-only）與 BERT（Encoder）差在哪？ | [`04`](04-gpt-decoder-only.md) |

若以上都能清楚解釋，就**不必再深鑽 Transformer 的理論細節**，可以往應用與前沿走。

### 接下來往哪走

```
Transformer（本教材）
   ↓
GPT / LLaMA        ← 06 已是 nanoGPT → LLaMA 的橋接
   ↓
Embedding → RAG    ← 向量資料庫、相似度搜尋、檢索增強
   ↓
Agent              ← Tool Calling、Planning、Workflow、Multi-Agent
   ↓
Multimodal         ← ViT、CLIP、視覺-語言模型
   ↓
Reasoning / RLHF   ← 多步推理、自我檢查、強化學習
```

> **趨勢備註：** 未來主流不太可能是「更大的單純 Transformer」，而是
> **Transformer + Reasoning + Tool Use + Verification + Planning** 的組合。
> 但無論上層怎麼變，Transformer 都是地基——這也是為什麼值得從零把它學懂。

---

**準備好了嗎？** → 從 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)（直覺版）或 [`01b-prerequisites-math.md`](01b-prerequisites-math.md)（數學版）開始。
