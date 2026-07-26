# 08｜模型家譜速查：三大家族與代表模型

> **適合對象：** 讀完 [`04a`](04a-gpt-decoder-only.md)（GPT）之後，想知道「我學的這個架構在真實世界對應到哪些模型、它們彼此差在哪」的讀者。也適合學到一半想確認方向時翻閱。
>
> **讀完後你能做什麼：**
> - 用一張時間軸說出 2014→今天，Transformer 三大家族各自的演化路線
> - 對常見模型名稱（RoBERTa、LLaMA、T5、Whisper、BGE…）立刻判斷它屬於哪一家、解決了前一代什麼問題
> - 說明每一家目前的**實際主戰場**，也就是「學完這一支之後實務上會遇到什麼」
> - 面對一個新任務，用三選一的判準決定該用哪一家
>
> **前置文件：** [`03a-transformer-architecture.md`](03a-transformer-architecture.md)（Transformer Block）、[`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)（§1 Encoder-Decoder 與三家的分岔）
>
> **定位：** **區 3（家族演進）的導覽文件，選讀，純速查**。本文**不推導任何公式**，每個模型最多兩句話，深入一律連出去：decoder 家族的技術細節在 [`06`](06-modern-transformer-variants.md)、encoder 家族在 [`07`](07-bert-encoder-only.md) §6、encoder-decoder 在 [`10a1`](10a1-seq2seq-forward.md)。本文與那三處的分工是：**它們講技術差異，本文講時間順序與應用落點**。
>
> **學完後的下一步：** → [`06`](06-modern-transformer-variants.md)（decoder 深入）、[`07`](07-bert-encoder-only.md)（encoder 深入）、[`09`](09-text-to-vector-rag.md)（系統層應用）

---

## 目錄

1. 時間軸：從 Seq2Seq 到今天
2. 三張家譜表
3. 每一家的當代主戰場
4. 選型速查：三選一
5. 下一步

---

## 1. 時間軸：從 Seq2Seq 到今天

```
2014  Seq2Seq（RNN encoder-decoder）＋ Bahdanau attention
        └ 問題：context vector 是瓶頸、逐步計算無法平行     → 10a1 §A
          ↓
2017  Transformer（Attention Is All You Need）
        └ 完全用 attention 取代遞迴，encoder-decoder 兩半俱全  → 03a
          ↓
2018  分家：兩大預訓練範式各取一半
        ├ BERT      取 encoder，雙向 ＋ MLM                → 07
        └ GPT-1     取 decoder，因果 ＋ next-token          → 04a
          ↓
2019  各家優化：RoBERTa / ALBERT / DistilBERT / Sentence-BERT
      GPT-2（規模放大）、T5 / BART（回頭把兩半接起來）
          ↓
2020  GPT-3 證明「規模＋in-context learning」路線可行 → decoder-only 成為主流
      ELECTRA、DeBERTa 把 encoder 訓練效率再推一階
          ↓
2023– LLaMA / Mistral / Qwen（開源 decoder 標準配方：RoPE＋RMSNorm＋SwiGLU＋GQA）→ 06
      encoder 那支轉進 embedding model（BGE / E5 / gte）→ 09
      encoder-decoder 那支轉進跨模態與跨語言（Whisper、NLLB）
```

要抓住的一句話：**2018 年的分家是這張圖的樞紐**。在那之前只有一種 Transformer，之後才有「三家」；而三家用的 Block 到今天仍是 [`03a`](03a-transformer-architecture.md) 那一個，差別只在遮罩、訓練目標，以及後來累積的工程優化。

---

## 2. 三張家譜表

三張表的欄位相同：模型、年份、一句話差異（**它解決了前一代什麼問題**）、本教材對應。

### 2.1 Decoder-only（生成型）

| 模型 | 年份 | 解決了前一代什麼問題 | 本教材對應 |
|---|---|---|---|
| **GPT-1** | 2018 | 證明「無監督預訓練 ＋ 下游微調」在 decoder 上也可行 | [`04a`](04a-gpt-decoder-only.md) 就是這個架構 |
| **GPT-2** | 2019 | 規模放大 ＋ 拿掉任務專屬微調，開始展現 zero-shot | 同上；nanoGPT（NB4）對應的正是 GPT-2 的縮小版 |
| **GPT-3** | 2020 | 規模再放大後出現 **in-context learning**——不改權重、靠 prompt 就能做任務。**與 GPT-2 的差別主要在規模，不在架構** | [`07`](07-bert-encoder-only.md) §5.3 的「微調 vs 提示」對比 |
| **LLaMA** | 2023 | 用較小的參數量配較多的訓練資料達到同級效果，並定下開源配方：RoPE ＋ RMSNorm ＋ SwiGLU | [`06`](06-modern-transformer-variants.md) 逐項展開 |
| **Mistral / Qwen** | 2023– | 在 LLaMA 配方上加 GQA、滑動窗注意力、MoE 等，推理成本再降 | [`06`](06-modern-transformer-variants.md) §4（GQA）|
| **GPT-4 / Claude / Gemini** | 2023– | 閉源前沿，架構細節未公開，但公認仍是 decoder-only ＋ 大量對齊工程 | — |

**這一家的關鍵認知**：從 GPT-1 到 GPT-3，**架構幾乎沒變**，變的是規模、資料量與訓練配方。所以你在 NB4 手刻的那個 nanoGPT，結構上就是現代 LLM 的核心；差距在規模與 [`06`](06-modern-transformer-variants.md) 那些工程優化（RoPE、GQA、Flash Attention、KV Cache）。

### 2.2 Encoder-only（雙向理解型）

| 模型 | 年份 | 解決了前一代什麼問題 | 本教材對應 |
|---|---|---|---|
| **BERT** | 2018 | 拿掉因果遮罩換成雙向 ＋ MLM，理解類任務大幅超越單向模型 | [`07`](07-bert-encoder-only.md) 全文、NB5 |
| **RoBERTa** | 2019 | 發現 BERT「根本沒訓練夠」：拿掉 NSP、動態遮罩、更多資料 | [`07`](07-bert-encoder-only.md) §4、§6 |
| **ALBERT** | 2019 | 跨層共享參數 ＋ embedding 分解，大幅減少參數量 | [`07`](07-bert-encoder-only.md) §6 |
| **DistilBERT** | 2019 | 知識蒸餾：約 40% 參數、60% 速度，保留約 97% 效果 | [`07`](07-bert-encoder-only.md) §6 |
| **ELECTRA** | 2020 | 改用 replaced token detection，讓**所有**位置都有訓練訊號，解掉 MLM 只有 15% 的稀疏問題 | [`07`](07-bert-encoder-only.md) §2.3、§6 |
| **DeBERTa** | 2020 | 解耦注意力（內容與位置分開算）＋ 相對位置編碼，同規模下效果再升一階 | [`07`](07-bert-encoder-only.md) §6（相對位置的概念見 [`06`](06-modern-transformer-variants.md) §3）|
| **Sentence-BERT** | 2019 | 用 siamese 結構微調，讓句向量可直接用**餘弦相似度**比對 | [`07`](07-bert-encoder-only.md) §6、[`09`](09-text-to-vector-rag.md) |
| **BGE / E5 / gte** | 2022– | 專門為檢索訓練的 embedding model，是 Sentence-BERT 路線的當代版本 | [`09`](09-text-to-vector-rag.md) |

**這一家的關鍵認知**：前六個是「怎麼把 encoder 訓得更好、更省」，後兩列是**路線轉向**——encoder 的重心已從「接 head 做分類微調」移到「產生好的句向量」。這也是為什麼 [`07`](07-bert-encoder-only.md) 的下一步是 [`09`](09-text-to-vector-rag.md)。

### 2.3 Encoder-Decoder（序列轉換型）

| 模型 | 年份 | 解決了前一代什麼問題 | 本教材對應 |
|---|---|---|---|
| **RNN Seq2Seq ＋ Bahdanau** | 2014 | attention 讓 decoder 能對齊輸入任意位置，打破固定長度 context vector 的瓶頸 | [`10a1`](10a1-seq2seq-forward.md) §A |
| **原始 Transformer** | 2017 | 用 self-attention 取代遞迴，可平行訓練；三家的共同祖先 | [`03a`](03a-transformer-architecture.md)、[`10a1`](10a1-seq2seq-forward.md) §B |
| **T5** | 2019 | 把所有 NLP 任務改寫成 text-to-text，一個模型一種介面 | [`10a1`](10a1-seq2seq-forward.md) §B |
| **BART** | 2019 | BERT 式雙向 encoder ＋ GPT 式自迴歸 decoder，摘要與翻譯表現強 | [`10a1`](10a1-seq2seq-forward.md) §B |
| **M2M-100 / NLLB** | 2020 / 2022 | 多語直譯，不必都經過英文中轉 | [`10a1`](10a1-seq2seq-forward.md) |
| **Whisper** | 2022 | encoder 吃**音訊特徵**、decoder 出文字——證明 cross-attention 的兩端不必是同一種模態 | [`10a1`](10a1-seq2seq-forward.md) §B3（Cross-Attention）|

**這一家的關鍵認知**：Whisper 那一列最值得看。cross-attention 的價值不在「翻譯」這個任務，而在「**輸入與輸出可以是兩個完全不同的序列，甚至不同模態**」。這就是為什麼這一支在語音與多模態仍是主流。

---

## 3. 每一家的當代主戰場

上面三張表回答「怎麼演化來的」，這一節回答「**所以我學完這一支之後，實務上會遇到什麼**」。

| 支線 | 當代主戰場 | 代表 | 學完之後接哪裡 |
|---|---|---|---|
| **Decoder-only** | 幾乎所有通用大型語言模型（LLM）。這是目前最主流的一支 | GPT-4、Claude、Gemini、LLaMA、Qwen、Mistral | [`06`](06-modern-transformer-variants.md) → 讀 LLaMA 原始碼 |
| **Encoder-only** | 重心已從「分類微調」轉到 **embedding model**，是 RAG 的**檢索端** | BGE、E5、gte、Sentence-BERT | [`09`](09-text-to-vector-rag.md) → 語意檢索與 RAG |
| **Encoder-Decoder** | 純文字生成上已被 decoder-only 取代（用 prompt 取代明確的任務分工），但在**跨模態與跨語言**仍是主流 | Whisper（語音→文字）、NLLB / M2M-100（翻譯）、影像描述生成 | [`10a1`](10a1-seq2seq-forward.md)–[`10b2`](10b2-seq2seq-backward-example.md) |

三家**今天都還活著，只是戰場不同**。特別是 encoder-decoder：它在聊天機器人這條路上輸給了 decoder-only，但只要任務的輸入與輸出是兩個不同的序列（尤其跨模態），它仍是首選。所以 [`10a1`](10a1-seq2seq-forward.md)–[`10b2`](10b2-seq2seq-backward-example.md) 那條支線不是歷史考古。

> **關於工程優化**：三家共用的效能議題——KV Cache、Flash Attention、RoPE 的長上下文擴展、MoE（Mixture of Experts）——大多在 decoder 家族先落地，本教材整理在 [`06`](06-modern-transformer-variants.md) 與 [`04b`](04b-nanogpt-walkthrough.md)（KV Cache）。

---

## 4. 選型速查：三選一

```
你的任務要「產生新文字」嗎？
   ├─ 是 ──┬─ 輸入也是一段序列，且與輸出屬於不同語言／模態？
   │       │     ├─ 是 → Encoder-Decoder（翻譯、語音辨識、影像描述）
   │       │     └─ 否 → Decoder-only（對話、續寫、摘要、通用任務）
   │       └─（純文字任務優先考慮 Decoder-only，用 prompt 取代任務專屬架構）
   │
   └─ 否，是「理解 / 分類 / 比對既有文字」
           └─ Encoder-only（分類、NER、抽取式 QA、句向量與檢索）
```

兩個容易踩的判斷：

- **摘要、翻譯這類任務兩家都能做**。差別在：encoder-decoder 是為它們原生設計的、小模型就有好效果；decoder-only 則靠規模與 prompt 取勝。資源有限、任務固定時選前者，要一個模型通吃時選後者。
- **跨模態任務幾乎一定走 encoder-decoder**。Whisper 是最好的例子：音訊與文字的長度、單位、分佈全都不同，硬塞進單一序列並不自然。

決定走 encoder-only 之後，還要再確認標註資料量與領域落差是否適合微調——三條判準見 [`07`](07-bert-encoder-only.md) §5.5。

---

## 5. 下一步

- **decoder 深入** → [`06-modern-transformer-variants.md`](06-modern-transformer-variants.md)：RMSNorm、SwiGLU、RoPE、GQA、Flash Attention，把 nanoGPT 補成 LLaMA。
- **encoder 深入** → [`07-bert-encoder-only.md`](07-bert-encoder-only.md)：雙向注意力、MLM、預訓練＋微調與遷移學習。
- **encoder-decoder 深入** → [`10a1-seq2seq-forward.md`](10a1-seq2seq-forward.md)：兩代架構的前後向數學與手算範例（四件套）。
- **系統層應用** → [`09-text-to-vector-rag.md`](09-text-to-vector-rag.md)：句向量、向量資料庫、RAG 檢索流程。
