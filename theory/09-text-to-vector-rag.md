# 09｜文字轉向量與語意檢索：從 Word2Vec 到 RAG

> **適合對象：** 讀完 [`07-bert-encoder-only.md`](07-bert-encoder-only.md) 後，想知道「把句子變成一個向量」到底能拿來做什麼、以及 RAG 怎麼運作的讀者。
>
> **讀完後你能做什麼：**
> - 用「分佈假說」說明為什麼上下文能定義一個詞的語意
> - 寫出 Word2Vec（Skip-gram）的目標函數，並解釋它「一詞一向量」的限制
> - 分辨**靜態** embedding（Word2Vec）與**動態／上下文** embedding（Transformer / BERT）
> - 說明餘弦相似度在語意檢索裡的值域與解讀
> - 描述 RAG（檢索增強生成）的完整流程，以及 encoder 與 decoder 各自的角色
>
> **前置文件：** [`01a`](01a-prerequisites-intuition.md)（Embedding 與餘弦相似度）、[`02`](02-attention-intuition.md)／[`03a`](03a-transformer-architecture.md)（自注意力）、[`07`](07-bert-encoder-only.md)（encoder 產生的上下文向量）
>
> **定位：** encoder 分支的**應用出口**，與 `06`（decoder → LLaMA）、`07`（encoder → BERT）平行。[`00`](00-learning-path.md) §5 流程圖裡「Embedding → RAG」那個節點，就是本文。

---

## 目錄

0. 閱讀地圖：從「一個字一個向量」到「語意檢索」
1. 分佈假說：語意來自上下文
2. Word2Vec：靜態向量的里程碑
3. 從靜態到動態：上下文相關的 Embedding
4. 餘弦相似度：檢索用的度量
5. RAG：檢索增強生成的完整流程
6. 總結

---

## 0. 閱讀地圖：從「一個字一個向量」到「語意檢索」

把文字轉成向量（**Embedding，文字嵌入**）的核心目標，是把人類語言映射到一個高維度的幾何空間，使**語意相近**的詞或句子在空間中**距離靠得更近**。一旦文字變成座標，「找出語意最相關的段落」就化約成「找出向量夾角最小的鄰居」——這正是 RAG 檢索的底層。

本文走一條時間線：靜態向量（Word2Vec）→ 動態向量（Transformer / BERT）→ 用餘弦相似度做檢索 → RAG。前半是「怎麼得到好向量」，後半是「拿向量來做什麼」。凡是 [`01a`](01a-prerequisites-intuition.md)／[`02`](02-attention-intuition.md)／[`03a`](03a-transformer-architecture.md)／[`07`](07-bert-encoder-only.md) 已經講過的（Embedding 基礎、自注意力、餘弦公式），本文只連結、不重推。

**你在教材地圖的哪裡：** 主線 nanoGPT（NB4）之後分出兩條選讀分支——decoder 那條往 [`06`](06-modern-transformer-variants.md) → LLaMA，encoder 那條是 [`07`](07-bert-encoder-only.md) BERT，而本文是 **encoder 分支的應用出口**：BERT 產生的上下文向量，正是這裡拿來做語意檢索的原料。

```
01→02→03  Transformer Block 地基
                    │
                    ▼
        04 + NB4   nanoGPT（主線終點）
                    │
        ┌───────────┴────────────┐
   decoder 分支               encoder 分支
   06 → LLaMA                 07 BERT ── NB5（去 mask + 換 MLM）
                                   │
                             09 文字轉向量 / RAG（本文，encoder 應用出口）
```

> encoder 與 decoder 在 RAG 裡各司其職——**用 encoder（BERT）做檢索、用 decoder（GPT）做生成**，這也是本文 §5 的主軸。分支全景另見 [`07`](07-bert-encoder-only.md) §0.1 與 [`00`](00-learning-path.md) §5。

---

## 1. 分佈假說：語意來自上下文

現代文字向量化的基石，是語言學家 J.R. Firth 的一句名言：

> *"You shall know a word by the company it keeps."*
> （要認識一個詞，看它跟哪些詞作伴。）

這叫**分佈假說（Distributional Hypothesis）**：一個詞的語意，由它出現的**上下文分佈**決定。

對比最原始的做法 **One-Hot Encoding**——每個詞是一個只有單一位置為 1、其餘全 0 的向量。它的致命問題是**任意兩個詞的內積都是 0**：「貓」與「狗」和「貓」與「桌子」一樣不相關，向量之間完全沒有語意結構（[`01a`](01a-prerequisites-intuition.md) §2 說明過為什麼不能只給編號）。

分佈假說給出的解法：不要人工定義語意，而是讓模型**從海量文本的上下文統計**去學。如果「貓」和「狗」經常出現在「寵物」「飼料」「獸醫」等相似上下文裡，模型學完後，這兩個詞的向量自然會靠得很近。下一節的 Word2Vec 就是把這個假說變成一個可訓練目標的第一個里程碑。

---

## 2. Word2Vec：靜態向量的里程碑

### 2.1 Skip-gram 的目標函數

Word2Vec（2013）把分佈假說寫成一個明確的預測任務。以 **Skip-gram** 為例：給定中央目標詞 $w_t$，去預測它周圍窗口大小為 $c$ 的上下文詞。目標是**最大化上下文的對數似然**：

$$
\mathcal{L} = \sum_{t=1}^{T} \sum_{-c \le j \le c,\; j \neq 0} \log P(w_{t+j} \mid w_t)
$$

其中條件機率用 **Softmax** 把兩個詞向量的**內積**轉成機率（Softmax 見 [`01a`](01a-prerequisites-intuition.md) §4）：

$$
P(w_O \mid w_I) = \frac{\exp\!\left(v'_{w_O}{}^{\top} v_{w_I}\right)}{\sum_{w=1}^{W} \exp\!\left(v'_{w}{}^{\top} v_{w_I}\right)}
$$

- $v_w$ 與 $v'_w$ 分別是詞 $w$ 當「目標詞」與「上下文詞」時的向量。
- 分子的內積 $v'_{w_O}{}^{\top} v_{w_I}$ 衡量兩個向量的方向與長度乘積：內積越大 → 共現機率越高 → 模型判定語意越相關（內積為何能代表相似度，見 [`01a`](01a-prerequisites-intuition.md) §3）。

訓練收斂後，副產品就是我們要的東西：一張 $W \times d$ 的向量表，每個詞一列。著名的 King − Man + Woman ≈ Queen 就是在這種向量上觀察到的（例子見 [`01a`](01a-prerequisites-intuition.md) §2）。

### 2.2 限制：一詞一向量

Word2Vec 的向量是**靜態**的：每個詞查表得到**固定**的一個向量，與它出現在哪個句子無關。於是多義詞會被壓成一個向量——

> 「蘋果」不論指水果還是科技公司，拿到的是**同一個**向量。

這在語意檢索上是硬傷：無法區分「吃蘋果」與「蘋果發表會」。要突破它，得讓向量**隨上下文改變**——這就是 Transformer / BERT 帶來的動態 embedding。

---

## 3. 從靜態到動態：上下文相關的 Embedding

Transformer 的**自注意力機制**讓句子裡每個字，依周圍其他字**動態調整**自己的向量。核心公式

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

本文不重推——它的逐步推導與直覺在 [`02`](02-attention-intuition.md) 與 [`03a`](03a-transformer-architecture.md) §3。這裡只點出對 embedding 的意義：$QK^{\top}$ 算出「每個字對其他所有字」的關聯度，softmax 歸一化後與 $V$ 相乘，**輸出的每個向量都融合了整句的上下文**。於是「蘋果」在「吃蘋果」與「蘋果發表會」裡會得到**不同**的向量——這就是**動態（上下文相關）embedding**。

| | 靜態（Word2Vec）| 動態（Transformer / BERT）|
|---|---|---|
| 向量怎麼來 | 查表，固定 | 依整句上下文即時算出 |
| 多義詞 | 壓成同一向量 | 依語境給不同向量 |
| 代表 | Word2Vec、GloVe | BERT、`text-embedding-3` 等 |

**要得到「一整句」的向量**（檢索的基本單位是句子或段落，不是單字），常見做法是拿 encoder 的輸出再池化：BERT 的 `[CLS]` 向量或平均池化，經 **Sentence-BERT**（[`07`](07-bert-encoder-only.md) §6）微調後，就能直接用餘弦相似度比較兩句的語意。這正是 [`07`](07-bert-encoder-only.md) §7 所說「encoder 最有價值的出口」。

---

## 4. 餘弦相似度：檢索用的度量

有了句向量，怎麼衡量兩段文字「像不像」？RAG 檢索階段最常用**餘弦相似度**——把兩個向量的內積除以各自的長度，**只看方向、不管長度**：

$$
\cos(A, B) = \frac{A \cdot B}{\|A\|\,\|B\|}
$$

（公式與「為何除以長度」的推導見 [`01a`](01a-prerequisites-intuition.md) §3.2；這裡補的是它在**檢索場景**的讀法。）

- **值域**：$[-1, 1]$。
- **解讀**：
  - $\cos = 1$（夾角 $0°$）：兩段文字語意方向完全一致。
  - $\cos = 0$（夾角 $90°$）：幾何垂直，語意互不相關。
  - $\cos < 0$（鈍角）：方向相反；在多數 embedding 檢索裡當作「很不相關」處理。

**為什麼檢索用餘弦、而不是內積或歐氏距離？** 因為我們要比的是**語意方向**，不希望「向量比較長」（例如較長或較常見的句子）就佔便宜。除以長度把「長度」這個干擾項消掉，只留方向。實務上，若事先把所有向量正規化成單位長度，餘弦相似度就等於內積，向量資料庫因此能用最快的內積運算完成檢索。

---

## 5. RAG：檢索增強生成的完整流程

RAG（Retrieval-Augmented Generation，檢索增強生成）解決一個現實問題：語言模型的知識停在訓練時點、也記不住你的私有文件。RAG 的做法是——**先檢索相關資料，再讓模型根據這些資料回答**。前半靠 encoder embedding，後半靠 decoder 生成。

```
【離線：建索引，一次】
  私有文件 ──切塊──> chunks ──encoder embedding──> 向量 ──存入──> 向量資料庫
                                                          (每個 chunk 一個向量)

【線上：每次查詢】
  使用者問題 ──encoder embedding──> query 向量
                                       │
                                       ├─ 對向量庫做餘弦相似度，取 top-k 最相關的 chunk
                                       ↓
  [問題 + 檢索到的 chunks] ──餵給──> decoder（GPT / LLaMA）──> 根據證據生成答案
```

逐步拆解：

1. **切塊（chunking）**：把文件切成段落大小的 chunk（太大稀釋語意、太小失去上下文）。
2. **建索引**：每個 chunk 用 encoder（如 Sentence-BERT）轉成一個向量，存進**向量資料庫**（FAISS、Milvus、pgvector 等）。這步離線做一次。
3. **查詢**：使用者的問題也用**同一個 encoder** 轉成 query 向量。
4. **檢索**：在向量庫裡用餘弦相似度找出與 query 最近的 top-k 個 chunk（§4）。
5. **生成**：把「問題＋檢索到的 chunks」一起餵給 **decoder** 語言模型，讓它**根據提供的證據**作答，而不是憑記憶編造。

**兩大家族在這裡分工合作**（呼應 [`07`](07-bert-encoder-only.md) §7）：

| 角色 | 用哪種模型 | 為什麼 |
|---|---|---|
| 把文字變向量、做檢索 | **encoder**（BERT / Sentence-BERT）| 雙向、擅長理解與表示，池化成好句向量 |
| 根據證據生成答案 | **decoder**（GPT / LLaMA）| 因果、擅長生成流暢文字 |

這也是為什麼「讀完主線（decoder）你會生成、讀完 07（encoder）你會理解與檢索」——RAG 把兩半拼起來，就是一套能查私有知識、又能好好回答的系統。

---

## 6. 總結

文字向量化的本質，是**把複雜的自然語言邏輯，縮減為高維空間中的幾何座標與夾角計算**。電腦不需要真正「理解」文字的抽象定義，只要透過幾何距離與方向的遠近，就能完成大規模的語意檢索。

一條線收束全文：

```
分佈假說（上下文定義語意）
   ↓
Word2Vec（靜態向量，一詞一向量）
   ↓
Transformer / BERT（動態向量，隨語境改變）→ Sentence-BERT 句向量
   ↓
餘弦相似度（比方向）
   ↓
RAG（encoder 檢索 + decoder 生成）
```

**下一步：** 沿 [`00`](00-learning-path.md) §5 的路線往後，RAG 之後是 Agent（工具使用、規劃）與 Multimodal（ViT、CLIP）。想回顧兩大家族的架構，見 [`06`](06-modern-transformer-variants.md)（decoder → LLaMA）與 [`07`](07-bert-encoder-only.md)（encoder → BERT）。
