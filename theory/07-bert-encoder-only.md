# 07｜BERT Encoder-Only：另一條路——雙向理解與 MLM 預訓練

> **適合對象：** 讀完 [`04-gpt-decoder-only.md`](04-gpt-decoder-only.md) 後，想理解「不是為了生成，而是為了理解」的那一支 Transformer 的讀者。
>
> **讀完後你能做什麼：**
> - 說明 BERT 為什麼拿掉 Causal Mask，用「雙向」注意力，以及這對任務的意義
> - 解釋 MLM（Masked Language Modeling）與 next-token prediction 的差異，並寫出它的損失
> - 描述 `[CLS]` / `[SEP]` 與 token / segment / position 三種 embedding 的角色
> - 說明「預訓練 + 微調」範式，以及分類 / 序列標註 / QA 各自怎麼接 head
> - 分辨 encoder 家族（RoBERTa、ELECTRA、Sentence-BERT…）並知道何時該選 encoder、何時選 decoder
>
> **前置文件：** [`03a-transformer-architecture.md`](03a-transformer-architecture.md)（Transformer Block）、[`04-gpt-decoder-only.md`](04-gpt-decoder-only.md)（§1 Encoder-Decoder、§3 Causal Masking）
>
> **定位：** 主線的 **encoder-family 選讀分支**。主線（01→04→NB4）走的是 decoder-only 的 GPT；[`06`](06-modern-transformer-variants.md) 是往 LLaMA 的 decoder 出口；本文是往 BERT 的 encoder 出口。兩者共用同一個 Transformer Block，差別只在**遮罩**與**訓練目標**。
>
> **學完後的下一步：** → [`../notebooks/NB5-bert-mlm.ipynb`](../notebooks/NB5-bert-mlm.ipynb)（從零手刻最小 BERT）

---

## 目錄

0. 閱讀地圖：GPT 與 BERT 的分岔
1. 雙向 Self-Attention：拿掉 Causal Mask
2. MLM 預訓練目標（Masked Language Modeling）
3. 輸入表示：`[CLS]` / `[SEP]` 與三種 Embedding
4. NSP 與後續的質疑
5. 預訓練 + 微調範式
6. BERT 家族速覽
7. encoder vs decoder 選型指南
8. 對照表：BERT vs GPT

---

## 0. 閱讀地圖：GPT 與 BERT 的分岔

[`04`](04-gpt-decoder-only.md) §1 講過：2017 年原始 Transformer 是 **Encoder-Decoder**。之後兩大預訓練範式各自只取一半：

```
                原始 Transformer（Encoder + Decoder）
                        │
        ┌───────────────┴───────────────┐
   取 Encoder                        取 Decoder
        │                                │
      BERT（2018）                       GPT（2018–）
   雙向、理解                         因果、生成
   Masked LM 預訓練                   Next-token 預訓練
   → 本文 07 / NB5                    → 主線 04 / NB4
```

一句話定調：

| | GPT（decoder-only）| BERT（encoder-only）|
|---|---|---|
| 注意力方向 | **因果**（只看過去）| **雙向**（左右都看）|
| 預訓練目標 | 預測下一個詞 | 填被遮住的詞 |
| 天生擅長 | **生成** | **理解 / 表示** |

關鍵是：**這兩者的 Transformer Block 幾乎一模一樣**——多頭注意力、殘差、LayerNorm、FFN 全部照 [`03a`](03a-transformer-architecture.md) 學過的。真正的差別只有兩處，就是本文 §1（遮罩）與 §2（目標）。理解了這兩處，你就同時懂了 encoder 與 decoder 兩大家族。

### 0.1 全景：從 nanoGPT 到這裡，實作只改兩處

把整份教材攤平來看，主線走到 nanoGPT（NB4）就分出兩條選讀分支，本文是其中的 **encoder** 這條：

```
01→02→03  打好 Transformer Block 地基（多頭 / 殘差 / LayerNorm / FFN）
                    │
                    ▼
        04 + NB4   nanoGPT（causal mask + next-token）── 主線終點
                    │
        ┌───────────┴────────────┐
   decoder 分支               encoder 分支（本文）
   06 現代變體                07 BERT ── NB5（重用 NB4，去 mask + 換 MLM）
   → LLaMA（decoder 出口）          │
                             09 文字轉向量 / RAG（encoder 應用出口）
```

主線終點是 nanoGPT，但 BERT **不是更進階的續集，而是平行的另一半**。你在 NB4 已經有一個能跑的 Transformer；[`NB5`](../notebooks/NB5-bert-mlm.ipynb) 幾乎原封不動**重用 NB4 的元件**（多頭注意力、殘差、LayerNorm、FFN、Pre-LN Block 全部照舊），只動兩個地方：

1. **拿掉 causal mask**（§1）：注意力矩陣從下三角變回全連接，位置 $i$ 能看全序列。
2. **換掉訓練目標**（§2）：把「預測下一字」的輸出頭改成「填被遮住的字」（MLM）。

所以學過 nanoGPT 的人不必重學架構，只要理解這兩個 diff。讀完本文再往下走，encoder 產生的向量會在 [`09`](09-text-to-vector-rag.md) 接到語意檢索與 RAG——那是這條分支的**應用出口**（與 decoder 分支的 [`06`](06-modern-transformer-variants.md) → LLaMA 平行；完整分支圖見 [`00`](00-learning-path.md) §5）。

---

## 1. 雙向 Self-Attention：拿掉 Causal Mask

### 1.1 從下三角回到全連接

回顧 [`04`](04-gpt-decoder-only.md) §3：GPT 為了「生成時不偷看未來」，在 softmax 前把未來位置的分數設成 $-\infty$，注意力矩陣因此是**下三角**：

```
GPT（Causal）— 位置 i 只能看 0..i：
位置 0：[1, 0, 0, 0]
位置 1：[1, 1, 0, 0]
位置 2：[1, 1, 1, 0]
位置 3：[1, 1, 1, 1]
```

BERT 不做生成，它的任務是**理解一整句已經給定的句子**——句子的每個字左右都在，沒有「未來」要保護。所以 BERT 直接**不加遮罩**，每個位置都能看到全序列：

```
BERT（雙向）— 位置 i 能看全部：
位置 0：[1, 1, 1, 1]
位置 1：[1, 1, 1, 1]
位置 2：[1, 1, 1, 1]
位置 3：[1, 1, 1, 1]
```

程式上的差異小到只有一行。GPT 的單頭注意力（[`04`](04-gpt-decoder-only.md) §5.1）長這樣：

```python
wei = q @ k.transpose(-2, -1) * C**-0.5          # (B, T, T) 原始分數
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))  # ← GPT 專屬：遮住未來
wei = F.softmax(wei, dim=-1)
```

BERT 版把中間那行**刪掉**即可：

```python
wei = q @ k.transpose(-2, -1) * C**-0.5          # (B, T, T)
# 沒有 masked_fill —— 這就是「雙向」
wei = F.softmax(wei, dim=-1)
```

> 注意：這裡刪的是**因果遮罩**（下三角）。實務上仍會保留一個 **padding mask**，把批次中補齊長度用的 `[PAD]` 位置遮掉，避免真實 token 去關注無意義的填充。padding mask 與因果 mask 是兩回事，前者是工程細節、後者才是 encoder/decoder 的本質差異。

### 1.2 雙向為什麼「更好理解」——以及它換來了什麼

考慮這個填空：

```
The animal didn't cross the street because it was too ___.
```

要決定 `it` 指的是 `animal` 還是 `street`、空格該填 `tired` 還是 `wide`，模型必須同時利用**左邊**（animal、cross）與**右邊**（如果句子後面還有線索）的資訊。因果模型在預測某個位置時右側被遮住，只能靠單邊；雙向模型左右通吃，對「理解」類任務（分類、抽取、標註）先天有利。

但天下沒有白吃的午餐——**雙向模型不能拿來自迴歸生成**。因為它訓練時每個位置都看過完整句子，一旦讓它「一個字一個字往下寫」，它從沒學過「只根據左文預測右邊」這件事。這正是為什麼生成走 decoder、理解走 encoder。§2 會看到，這個限制其實是被**預訓練目標**逼出來的。

> 有趣的巧合：naive self-attention（[`01b`](01b-prerequisites-math.md) §「注意力權重矩陣」）在還沒引入 $W_Q, W_K$ 時，相似度矩陣 $E = XX^\top$ 本來就是**對稱**的（$E_{ij}=E_{ji}$）——也就是說「雙向」才是注意力的原始面貌，**因果遮罩反而是為了生成而後加的限制**。BERT 只是把這個後加的限制拿掉而已。

---

## 2. MLM 預訓練目標（Masked Language Modeling）

拿掉遮罩之後，還有一個問題：**訓練目標要換掉**。

GPT 的目標是 next-token prediction（[`04`](04-gpt-decoder-only.md) §4）：位置 $i$ 看 $x_1..x_i$、預測 $x_{i+1}$。這個目標**只在因果遮罩下才成立**——如果雙向模型也做「預測下一個詞」，那位置 $i$ 早就直接看到答案 $x_{i+1}$ 了，等於抄答案，什麼都學不到。

BERT 的解法是 **Masked Language Modeling（MLM，克漏字）**：把輸入句子隨機挖掉一些字，讓模型用**左右文**把它們填回來。

### 2.1 遮罩規則：15% 與 80/10/10

BERT 隨機挑 **15%** 的 token 當作預測目標。對每個被挑中的 token，再依下列比例決定它在**輸入端**長什麼樣：

| 比例 | 輸入端替換成 | 目的 |
|---|---|---|
| 80% | `[MASK]` 特殊 token | 主要的克漏字訊號 |
| 10% | 隨機一個其他 token | 逼模型別盲信輸入、要靠上下文糾錯 |
| 10% | 維持原字不變 | 讓模型對「非 `[MASK]`」的位置也保持表示能力 |

**為什麼不是全部換成 `[MASK]`？** 因為微調與推論時輸入裡**沒有** `[MASK]`。若預訓練時模型只在看到 `[MASK]` 才需要輸出好的表示，就會與下游任務產生落差（train/inference mismatch）。混入 10% 隨機字與 10% 原字，讓模型對「每一個位置」都維持好的上下文表示，而不只針對 `[MASK]`。

### 2.2 損失：只在被遮位置算 Cross-Entropy

設被挑中的位置集合為 $\mathcal{M}$（約佔 15%）。模型對每個位置輸出一個 vocab 上的機率分佈 $p_i = \text{softmax}(\text{logits}_i)$，損失**只累加 $\mathcal{M}$ 裡的位置**：

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \log p_i\big[\,y_i\,\big]
$$

其中 $y_i$ 是位置 $i$ 的原始（正確）token。這與 [`04`](04-gpt-decoder-only.md) §4 的 cross-entropy 形式完全相同，唯一差別是**求和範圍**：next-token 對**每個**位置都算損失（一個序列同時訓練 $T$ 個預測），MLM 只對**被遮的 15%** 算損失。

程式上對應 `F.cross_entropy` 的 `ignore_index`——把沒被遮的位置的 target 設成 `-100`，就不計入損失：

```python
# logits: (B, T, vocab)   labels: (B, T)，未被遮的位置填 -100
loss = F.cross_entropy(logits.view(B*T, vocab), labels.view(B*T), ignore_index=-100)
```

### 2.3 代價：訊號稀疏

MLM 每個序列只從 15% 的位置得到學習訊號，next-token 則是 100%（每個位置都預測下一字）。所以 MLM 的**樣本效率較低**、通常需要更多預訓練步數——這是換取「雙向」的代價。後來的 ELECTRA（§6）就是為了解決這個稀疏問題而設計的。

---

## 3. 輸入表示：`[CLS]` / `[SEP]` 與三種 Embedding

BERT 在把句子送進 Transformer 之前，做了兩件 GPT 沒有的事。

### 3.1 特殊 token：`[CLS]` 與 `[SEP]`

```
輸入： [CLS]  the  cat  sat  [SEP]  it  slept  [SEP]
        └句首          └句A結束        └句B結束
```

- **`[CLS]`**（classification）：放在最前面。因為是雙向注意力，這個位置能看到整句，經過數層後它的輸出向量就成了**整句的表示**——下游分類任務直接接在 `[CLS]` 的輸出上（§5）。
- **`[SEP]`**（separator）：分隔兩個句子，讓 BERT 能吃「句子對」輸入（問答、句子關係判斷都需要一次餵兩句）。

### 3.2 三種 Embedding 相加

主線的 GPT 把 **token embedding + position embedding** 相加（[`03a`](03a-transformer-architecture.md) §7.5 的可學習 PE）。BERT 多加一種 **segment embedding**，用來標記「這個 token 屬於句 A 還是句 B」：

$$
\text{input}_i = \underbrace{E_{\text{tok}}[x_i]}_{\text{哪個字}} + \underbrace{E_{\text{seg}}[s_i]}_{\text{句A/句B}} + \underbrace{E_{\text{pos}}[i]}_{\text{第幾位}}
$$

三者都是可學習的向量、逐元素相加，維度都是 $d$。segment embedding 只有兩個向量（$s_i \in \{A, B\}$）；若輸入只有單句，全部用 $A$ 即可。

> BERT 的 position embedding 是**可學習**的（同 [`03a`](03a-transformer-architecture.md) §7.5），不是正弦式，也還沒有 RoPE（RoPE 見 [`06`](06-modern-transformer-variants.md) §3）——這一點與 nanoGPT 相同。

---

## 4. NSP 與後續的質疑

原始 BERT 除了 MLM，還有第二個預訓練目標 **NSP（Next Sentence Prediction，下一句預測）**：給定句 A 與句 B，用 `[CLS]` 的輸出做二分類，判斷「B 是不是 A 在原文中的下一句」。動機是讓模型學會**句子之間**的關係，對問答、自然語言推理（NLI）這類需要理解句對的任務有幫助。

但後續研究（尤其 **RoBERTa**，§6）發現：**NSP 幾乎沒有貢獻，甚至可能有害**。原因是 NSP 把「主題預測」和「連貫性預測」混在一起，任務太容易，模型學不到真正有用的句間關係。RoBERTa 直接**拿掉 NSP**、只留 MLM，並用更多資料、更長訓練、更大 batch，效果反而更好。

**結論：** MLM 是 BERT 真正的核心；NSP 屬於「當年這樣設計、後來被證明可省」的歷史細節。手刻最小 BERT（NB5）時只實作 MLM 即可。

---

## 5. 預訓練 + 微調範式

BERT 讓「**預訓練一次、到處微調**」成為主流。流程分兩階段：

```
階段一：預訓練（一次，昂貴）
  海量無標註文本 ──MLM──> 一個「懂語言」的 encoder（權重 θ）

階段二：微調（每個任務一次，便宜）
  θ 當起點 + 接一個小 head + 少量標註資料 ──> 專用模型
```

**下游任務怎麼接 head：**

| 任務類型 | 接在哪 | head 形狀 | 例子 |
|---|---|---|---|
| 句子分類 | `[CLS]` 的輸出向量 | $d \to$ 類別數 的 Linear | 情感分析、垃圾郵件 |
| 句子對分類 | `[CLS]`（吃 A `[SEP]` B）| $d \to$ 類別數 | NLI、語意相似 |
| 序列標註 | **每個 token** 的輸出 | $d \to$ 標籤數（逐位置）| 命名實體辨識（NER）、詞性標註 |
| 抽取式 QA | 每個 token 的輸出 | 兩個 $d \to 1$（預測答案 span 的起、迄）| SQuAD |

關鍵在於：**主體 encoder 不變，只換最上面那層薄薄的 head**，再用該任務的少量標註資料微調整個網路。這就是為什麼一個 BERT 預訓練權重能撐起幾十種下游應用。

> 這與 GPT 家族後來的路線（zero-shot / few-shot prompting，不改權重）形成對比：BERT 時代靠「微調」，GPT-3 之後靠「提示」。兩條路各有適用場景。

---

## 6. BERT 家族速覽

BERT（2018）之後，encoder 家族沿著「更好的預訓練」與「更小更快」兩個方向演化：

| 模型 | 一句話差異 |
|---|---|
| **RoBERTa** (2019) | 拿掉 NSP、動態遮罩、更多資料更久訓練——證明「BERT 沒訓練夠」 |
| **ALBERT** (2019) | 跨層共享參數 + embedding 分解，大幅減參數量 |
| **ELECTRA** (2020) | 改用 **replaced token detection**：判斷每個 token 是否被替換過，讓**所有**位置都有訓練訊號，解決 §2.3 的 15% 稀疏問題，樣本效率大增 |
| **DistilBERT** (2019) | 知識蒸餾，約 40% 參數、60% 速度，保留約 97% 效果 |
| **Sentence-BERT** (2019) | 用 siamese 結構微調，讓 `[CLS]` / 池化向量可直接用**餘弦相似度**比對——句子嵌入與語意檢索的基石 |

前四個是「怎麼把 encoder 訓得更好 / 更省」，Sentence-BERT 則指向下一節的應用出口。

---

## 7. encoder vs decoder 選型指南

學到這裡，你手上有兩種架構。實務上怎麼選？

```
你的任務是「產生新文字」嗎？
   ├─ 是 → decoder-only（GPT / LLaMA）         → 主線 04 / 06
   │        生成、對話、續寫、翻譯、摘要
   └─ 否，是「理解 / 分類 / 比對既有文字」→ encoder-only（BERT 家族）→ 本文
            分類、NER、抽取式 QA、句子相似、檢索嵌入
```

其中「**把句子變成一個向量**」是 encoder 最有價值的出口，直接銜接 RAG 檢索（完整展開見 [`09`](09-text-to-vector-rag.md)）：

```
Sentence-BERT ──> 句子 / 文件 embedding ──> 向量資料庫 ──> 相似度檢索（RAG）
```

RAG（檢索增強生成）常見的組合，正是**用 encoder 做檢索、用 decoder 做生成**——兩大家族各司其職。所以這條 encoder 分支不是主線的替代品，而是**補上另一半**：讀完主線你會生成，讀完本文你會理解與檢索。從句向量到 RAG 檢索流程的完整說明，見 [`09-text-to-vector-rag.md`](09-text-to-vector-rag.md)。

> 補充：也有 **encoder-decoder** 模型（T5、BART），把「理解輸入」與「生成輸出」都要的 seq2seq 任務（翻譯、摘要）用兩半一起做。它就是原始 Transformer 的直系後代（[`04`](04-gpt-decoder-only.md) §1）。

---

## 8. 對照表：BERT vs GPT

| | BERT（encoder-only）| GPT（decoder-only）|
|---|---|---|
| 取自原始 Transformer 的 | Encoder | Decoder |
| 注意力遮罩 | 無因果遮罩（雙向）| 下三角（因果）|
| 注意力矩陣 | 全連接 | 下三角 |
| 預訓練目標 | MLM（填 15% 被遮的字）| Next-token（預測下一字）|
| 訓練訊號密度 | 稀疏（15% 位置）| 稠密（每個位置）|
| 特殊 token | `[CLS]` / `[SEP]` | 通常只有句界／BOS |
| Embedding | token + **segment** + position | token + position |
| 天生擅長 | 理解、分類、抽取、嵌入 | 生成、續寫、對話 |
| 下游用法 | 接 head 微調 | 微調或 prompting |
| 代表模型 | BERT、RoBERTa、ELECTRA、Sentence-BERT | GPT、LLaMA、Mistral、Qwen |
| 本教材對應 | 07 / NB5 | 04 / NB4、06 |

**不變的部分**：多頭注意力、$\text{softmax}(QK^\top/\sqrt{d_k})V$、殘差、LayerNorm、FFN、Pre-LN Block——[`03a`](03a-transformer-architecture.md) 學到的骨架**兩家共用**。變的只有這張表列出的「遮罩」與「目標」兩條主軸，其餘全是它們的衍生。

---

## 下一步

- 動手：[`../notebooks/NB5-bert-mlm.ipynb`](../notebooks/NB5-bert-mlm.ipynb) 從零手刻最小 BERT——把 NB4 的 `Head` 刪掉一行 `masked_fill`、把 next-token 換成 MLM，親眼看到「同一個 Block、換遮罩換目標」就從 GPT 變成 BERT。Notebook 末尾附**選讀延伸**：載入 HuggingFace 預訓練 BERT 做克漏字與分類，對照手刻版。
- 應用：[`09-text-to-vector-rag.md`](09-text-to-vector-rag.md)——從 Word2Vec 到 Sentence-BERT，再到 RAG 的完整檢索流程（encoder 分支的應用出口）。
- 對照：想看 decoder 家族怎麼演化到 LLaMA，見 [`06`](06-modern-transformer-variants.md)。
