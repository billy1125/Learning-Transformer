# 00｜學習路線與背景：在開始之前

> **適合對象：** 準備開始、或學到一半想確認方向的讀者。
>
> **讀完後你會知道：**
> - Transformer 在機器學習發展史上的位置，以及它為什麼算「進階」
> - 這份教材**會用到**哪些數學（必學），又有哪些**值得延伸**（選學）
> - 整個倉庫的組織方式：**五個區 × 四種呈現層次**，以及最短主線走哪幾份
> - 三大家族（decoder／encoder／encoder-decoder）分別在哪一區、為什麼是平行而非串接
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
| 2013–2014 | 表示與對齊 | word2vec（Embedding）、[seq2seq + Attention](10a1-seq2seq-forward.md) | 詞被映射成向量；Attention 讓 decoder 能「對齊」到輸入任意位置 |
| **2017** | **Transformer** | *Attention Is All You Need* | **完全用 Attention 取代遞迴**：可並行、長距離依賴變成 $O(1)$ 路徑 |
| 2018–2020 | 預訓練範式（[三大家族分家](08-model-family-tree.md)）| [BERT（Encoder）](07-bert-encoder-only.md)、GPT（Decoder）| 「大規模預訓練 + 下游微調」成為主流 |
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
| **Gradient（梯度）** | 訓練＝沿梯度更新參數 | [`05`](05b1-backward-propagation.md) |

> 這五項就是入場券。只要看得懂「矩陣相乘」和「對一排數字取 softmax」，就能開始讀 `01` → `02` → `03a`。

---

## 3. 值得延伸的數學（選學）

學完主線後，下面這些能讓你讀懂更進階的模型與論文。**不必在開始前先學**，遇到再回來補即可。依優先順序分層：

### Level 2：進階（建議學）

| 主題 | 內容 | 之後用在哪 |
|---|---|---|
| 線性代數 | Projection、Basis、Linear Transformation | 理解 QKV「投影到子空間」的本質 |
| 機率 | 機率分布、期望值 | `03a` §3.4 縮放的統計推導、[`01b`](01b-prerequisites-math.md) |
| 資訊理論 | Entropy、Cross Entropy | 語言模型的損失函數（[`04a`](04a-gpt-decoder-only.md)）|
| 特徵值 / 特徵向量 | PCA 基礎 | 理解降維與表示空間 |

### Level 3：旁支應用

- **Cosine 相似度** → Embedding 檢索、RAG（展開見 [`09`](09-text-to-vector-rag.md)）
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

整份教材依**兩條軸**組織。先看第一條軸：內容分成五個區，依序前進。

### 4.1 軸一：五個區（學習階段）

| 區 | 名稱 | 這一區在回答 | 終點：讀完你能做到 |
|---|---|---|---|
| **0** | 導覽 | 我要學什麼、需要多少數學 | 選定起點與路線（就是本文）|
| **1** | 地基：一個 Transformer Block | 注意力怎麼算、Block 怎麼組起來 | **手算一個完整的 Pre-LN Block** |
| **2** | 三大家族 | 同一個 Block 怎麼變成三種模型 | 說出三家的遮罩與訓練目標差異，並**手刻其中任一** |
| **3** | 家族演進與模型家譜 | 2018 年分家之後各家怎麼進化 | 看懂 LLaMA／RoBERTa 的原始碼在改什麼 |
| **4** | 系統層應用 | 架構之上還有什麼 | 說明 RAG 為何要 encoder ＋ decoder 併用 |

各區成員：

**區 1｜地基**（三家共用，不分流）

```
01a 直覺版 或 01b 數學版  →  02 Attention 直覺  →  03a 架構        →  NB1 → NB2
(向量/softmax/加權平均)      (QKV 翻譯範例)        (多頭/Block/PE)
                                                    ├ 03a-plain     零公式輔助版（§6 的白話版）
                                                    ├ 03b1→03b2→03b3 計算案例三階段（簡單→中等→完整）
                                                    └ 03b4          含位置編碼 P≠0 的對照支線
```

**區 2｜三大家族**（三條**平行**支線，都建立在區 1 的 Block 之上）

```
                     區 1 終點：一個 Transformer Block
                                   │
        ┌──────────────────────────┼──────────────────────────┐
   2A Decoder-only            2B Encoder-only          2C Encoder-Decoder
      GPT（★ 主線）                BERT                    Seq2Seq / T5
   ──────────────────         ───────────────          ──────────────────
   04a  概念與 Pipeline        07 BERT                  10a1  前向數學
   05a1 前向數學                                        10a2  前向數值範例
   05a2 前向數值範例                                     10b1  後向數學
   05b1 後向數學                                        10b2  後向數值範例
   05b2 後向數值範例
   04b  nanoGPT 程式對照
        │                          │                        │
        ▼                          ▼                        ▼
   NB3 / NB4                  NB5                      （實作待補）
```

**區 3｜家族演進** — [`08`](08-model-family-tree.md) 模型家譜速查（三家的時間軸、代表模型、當代主戰場）、[`06`](06-modern-transformer-variants.md) decoder 當代變體（RMSNorm、SwiGLU、RoPE、GQA、Flash Attention）、[`07`](07-bert-encoder-only.md) §6 encoder 家族速覽

**區 4｜系統層應用** — [`09`](09-text-to-vector-rag.md) 文字轉向量與 RAG（未來延伸：Agent、Multimodal，見 §5）

### 4.2 軸二：呈現層次（同一主題的四種切法）

同一份知識，本教材會用四種深度各講一次。**看懂這條軸，就知道每份文件在做什麼、可以跳過哪些**：

| 層次 | 回答什麼 | 代表文件 |
|---|---|---|
| **概念** | 這是什麼、為什麼需要 | [`01a`](01a-prerequisites-intuition.md)、[`02`](02-attention-intuition.md)、[`03a-plain`](03a-transformer-block-plain.md)、[`04a`](04a-gpt-decoder-only.md)、[`08`](08-model-family-tree.md)、[`09`](09-text-to-vector-rag.md) |
| **符號數學** | 公式怎麼推出來 | [`01b`](01b-prerequisites-math.md)、[`03a`](03a-transformer-architecture.md)、[`05a1`](05a1-forward-propagation.md)、[`05b1`](05b1-backward-propagation.md)、[`06`](06-modern-transformer-variants.md)、[`10a1`](10a1-seq2seq-forward.md)、[`10b1`](10b1-seq2seq-backward.md) |
| **數值範例** | 代真實數字算一次 | [`03b1`](03b1-transformer-example-basic.md)–[`03b4`](03b4-transformer-example-with-position.md)、[`05a2`](05a2-forward-example.md)、[`05b2`](05b2-backward-example.md)、[`10a2`](10a2-seq2seq-forward-example.md)、[`10b2`](10b2-seq2seq-backward-example.md) |
| **程式對照** | 公式對應到程式的哪一行 | [`04b`](04b-nanogpt-walkthrough.md)、NB1–NB5 |

這條軸順帶解釋兩件事：為什麼 [`04b`](04b-nanogpt-walkthrough.md) 的閱讀順序排在 [`05a2`](05a2-forward-example.md) 之後（程式對照要等數學講完），以及為什麼 [`03a`](03a-transformer-architecture.md) 之外還有 `03b1`–`03b4`（同一主題的「數值範例」層，三階段遞進）。

**最短主線**（只走「概念 ＋ 符號數學 ＋ 程式對照」，跳過所有數值範例與選讀分支）：

```
01 → 02 → 03a → 04a → 05a1 → 04b → NB1 → NB2 → NB4
```

### 4.3 三件關於區 2 的事

**一、三條支線的深度刻意不對稱。** 反向傳播只在 2A 完整推導一次（[`05b1`](05b1-backward-propagation.md)／[`05b2`](05b2-backward-example.md)），2B 直接共用，2C 只推「與 2A 不同的那部分」（BPTT 的時間鏈、cross-attention 的梯度分岔）。所以 2B 只有一份文件不是漏寫——BERT 與 GPT 的差別**只有遮罩與訓練目標兩處**。

**二、建議的閱讀順序是 2A → 2B → 2C。** 理由：2A 的反向推導最完整，另兩支都建立在它之上；2B 只需理解兩處 diff，一天可讀完；2C 是三者中前置最寬鬆的（讀完 [`03a`](03a-transformer-architecture.md) 就能讀），但內容量最大。若時間有限，做完 2A 就已經抵達本教材的主要目標。

**三、三家今天都還活著，只是主戰場不同。** 這是每一支的「為什麼要學」：

| 支線 | 當代主戰場 | 代表 |
|---|---|---|
| 2A Decoder-only | 幾乎所有通用大型語言模型 | GPT-4、Claude、Gemini、LLaMA、Qwen |
| 2B Encoder-only | 重心已從分類微調轉到 **embedding model**，是 RAG 的檢索端 | BGE、E5、gte、Sentence-BERT |
| 2C Encoder-Decoder | 純文字生成上被 decoder-only 取代，但**跨模態／跨語言**仍是主流 | Whisper、NLLB、M2M-100、T5 |

完整的家譜與時間軸見 [`08`](08-model-family-tree.md)。

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
| GPT（Decoder-only）與 BERT（Encoder）差在哪？ | [`04a`](04a-gpt-decoder-only.md)、[`07`](07-bert-encoder-only.md) |
| 三大家族分別是什麼？各自的代表模型與當代主戰場？ | [`08`](08-model-family-tree.md) |

若以上都能清楚解釋，就**不必再深鑽 Transformer 的理論細節**，可以往應用與前沿走。

### 接下來往哪走

分成兩層看：**架構層**（區 3，還在改 Transformer 本身）與**系統層**（區 4，架構已定，開始組系統）。

```
架構層（區 3）——三家各自演進，家譜見 08
                 Transformer（本教材區 1）
                            │
    ┌───────────────────────┼───────────────────────┐
Decoder 家族         Encoder-Decoder 家族      Encoder 家族
GPT → LLaMA          原始 Transformer → T5     BERT → RoBERTa
Qwen / Mistral       BART / Whisper / NLLB     → Sentence-BERT
← 06 是出口          ← 10a1–10b2 是入口        → BGE / E5 / gte
                     （翻譯、語音、跨模態）      ← 07 是入口
    └───────────────────────┬───────────────────────┘
                            ↓
─────────────────────────────────────────────────────────
系統層（區 4）——架構不再是主角，開始組系統
                  Embedding → RAG   ← encoder 檢索、decoder 生成（見 09）
                            ↓
                        Agent       ← Tool Calling、Planning、Multi-Agent
                            ↓
                     Multimodal     ← ViT、CLIP、視覺-語言模型
                            ↓
                  Reasoning / RLHF  ← 多步推理、自我檢查、強化學習
```

本教材涵蓋到 `09`（RAG）為止；Agent 之後屬於系統層的未來延伸，不在本倉庫範圍內。

> **趨勢備註：** 未來主流不太可能是「更大的單純 Transformer」，而是
> **Transformer + Reasoning + Tool Use + Verification + Planning** 的組合。
> 但無論上層怎麼變，Transformer 都是地基——這也是為什麼值得從零把它學懂。

---

**準備好了嗎？** → 從 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)（直覺版）或 [`01b-prerequisites-math.md`](01b-prerequisites-math.md)（數學版）開始。
