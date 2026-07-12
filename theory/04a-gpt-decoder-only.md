# 04a｜GPT Decoder-Only：原理與數學

> **適合對象：** 讀完 03a 後，想從數學層面完整理解 GPT（Decoder-Only）架構，並準備對照 nanoGPT 程式的讀者。
>
> **讀完後你能做什麼：**
> - 解釋為什麼 GPT 只有 Decoder，不需要 Encoder
> - 推導 Scaled Dot-Product Attention，並說明為何除以 $\sqrt{d_k}$
> - 描述 Causal Masking 的原理與數值
> - 寫出 Multi-Head、FFN、LayerNorm／Pre-LN Block 的數學形式
> - 說明 Next-token Prediction 的 Cross-Entropy 損失，以及從 loss 到 Embedding 的完整梯度鏈
>
> **前置文件：** [`03a-transformer-architecture.md`](03a-transformer-architecture.md)
>
> **對照實作：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md)（每段 nanoGPT 程式如何落實本文數學）
>
> **學完後的下一步：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md) → [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)

---

## 目錄

1. 原始 Transformer 是 Encoder-Decoder
2. GPT 為什麼只要 Decoder？
3. Scaled Dot-Product Attention 的數學
4. Causal Masking：只能看過去，不能偷看未來
5. Multi-Head Attention 的數學
6. Position-wise FFN 的數學
7. LayerNorm 與 Pre-LN Block 的數學
8. Token Embedding 與位置編碼的數學
9. Next-token Prediction 與 Cross-Entropy
10. 反向傳播：從 loss 到 Embedding

> **本文與 [`04b`](04b-nanogpt-walkthrough.md) 的分工：** 本文（04a）負責**數學原理**，每個式子當場推導、自成一體；[`04b`](04b-nanogpt-walkthrough.md) 負責**程式對照**，逐行把 nanoGPT 對回本文的節號。建議 04a → 04b → NB4 依序讀。

---

## 1. 原始 Transformer 是 Encoder-Decoder

2017 年「Attention Is All You Need」提出的原始 Transformer 是為**翻譯任務**設計的 Encoder-Decoder 架構。

先用一個生活比喻建立直覺：把翻譯想成一位**譯者**在工作。他要先「**讀懂**」整句英文（這是 Encoder 的事），再一個字一個字「**寫出**」中文（這是 Decoder 的事）。讀的時候可以來回看整句、前後對照；寫的時候卻只能順著往下寫，還沒寫出來的字當然不能先看。這兩件事分別對應架構裡的兩大模組：

- **Encoder（讀懂輸入）**：雙向（bidirectional）— 處理每個英文詞時，可以同時參考它前面和後面的所有詞，藉此把整句話的語意「消化」成一份內部筆記。
- **Decoder（產生輸出）**：因果（causal，意思是「只受過去影響」）— 逐字生成中文，處理第 $t$ 個位置時只能看自己和它**之前**已經產生的字，不能偷看還沒生成的未來（否則就等於抄答案，訓練與實際生成的情境會不一致）。

那 Decoder 怎麼知道要翻的是哪句英文？答案是 **Cross-Attention（交叉注意力）**。這裡先給最小背景：Attention 的運作可以想成一次「查資料」，由三個角色組成——

- **Query（Q，查詢）**：我現在想找什麼？
- **Key（K，索引）**：每筆資料的標籤是什麼？
- **Value（V，內容）**：每筆資料實際的內容是什麼？

拿 Q 去和每個 K 比對相似度，愈相似的那筆，它的 V 就被取用愈多（Q/K/V 的完整推導見本文 §3 與 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §2）。**Self-Attention** 是 Q、K、V 都來自同一串序列（自己查自己）；而 **Cross-Attention** 讓 Decoder 拿自己的 **Q**（「我下一個中文字該對到哪裡？」）去查 Encoder 那份筆記的 **K/V**（英文各詞的語意），這樣寫中文時就能對準正在翻的英文詞。資料流如下：

```
英文句子 → [Encoder] ──讀懂後產出 K, V（一份語意筆記）
                           │
                           ▼
中文前綴 → [Decoder] ──用自己的 Q 查 Encoder 的 K/V──▶ 下一個中文字
        （Cross-Attention：Q 來自 Decoder，K/V 來自 Encoder）
```

整理成一張表，三個模組各司其職：

| 模組 | 看的範圍 | 用途 |
|---|---|---|
| Encoder Self-Attention | 雙向（全序列）| 讀懂輸入句的語意 |
| Decoder Self-Attention | 因果（只看過去）| 生成時不偷看未來 |
| Decoder Cross-Attention | Q 來自 Decoder，K/V 來自 Encoder | 讓輸出對齊輸入 |

**一句話總結：** Encoder 負責「讀懂」、Decoder 負責「寫出」，兩者靠 Cross-Attention 對接。記住這個分工，下一節就能看懂 GPT 為什麼可以把 Encoder 整個拿掉。

---

## 2. GPT 為什麼只要 Decoder？

回到第 1 節的比喻：翻譯需要「先讀懂英文、再寫出中文」兩件事，所以要 Encoder＋Decoder。但 GPT 做的不是翻譯，而是**語言建模**——它只做「接話」這一件事：

> 給定一段文字 $x_1, x_2, \ldots, x_t$，預測下一個詞 $x_{t+1}$。

例如看到「今天天氣真」，模型要預測下一個字很可能是「好」。這裡**沒有另一種語言要先讀懂**，從頭到尾只有同一個文字串流，一路往下接。既然不存在「來源句」這個東西：

- **不需要 Encoder** — 沒有一段獨立的來源序列要先消化成筆記。
- **不需要 Cross-Attention** — Encoder 都不存在了，自然沒有別人的 K/V 可查（回顧 §1：Cross-Attention 就是拿 Q 去查 Encoder 的 K/V）。
- **只需要 Decoder 的 Causal Self-Attention** — 「接話」和「逐字寫中文」一樣是順著往下、不能偷看未來，所以保留 Decoder 那套「只看過去」的自注意力就夠了。

**結果：** GPT 把原始 Transformer 裡的 Decoder 單獨拿出來（連帶拿掉它內部的 Cross-Attention），堆疊 $N$ 層，就是完整模型——**架構其實比原始 Transformer 更簡單**。

```
原始 Transformer：Encoder + Decoder + Cross-Attention
GPT：            只有 N 層 Causal Decoder Block
```

| 架構 | 模型代表 | 典型用途 |
|---|---|---|
| Encoder only | BERT | 分類、問答（雙向理解）|
| Decoder only | GPT, LLaMA | 文字生成、語言建模 |
| Encoder-Decoder | T5, 原始 Transformer | 翻譯、摘要（seq2seq）|

> 本文往下只談 Decoder-Only。Encoder-Only（BERT）那一支——雙向注意力、MLM 預訓練、預訓練+微調——是主線的選讀分支，見 [`07-bert-encoder-only.md`](07-bert-encoder-only.md)。

Decoder-Only 架構確定了，但還有一個問題：訓練時如果讓模型看到未來的詞，等於作弊——第 4 節說明如何用遮罩阻止這件事（在此之前，第 3 節先把「Attention 到底在算什麼」寫成數學）。

事實上，目前的頂級主流大型語言模型（例如 ChatGPT、Claude、Gemini、LLaMA 等），其核心本質全都屬於 **Decoder-Only（僅解碼器）** 的架構，而不是把這三種架構混在一起。

原因在於：Decoder-Only 用更簡單、更統一的結構（對照上面 Encoder-Decoder 的複雜度就有感），換來更高的計算效率；而且只要把任務改寫成「接話」的形式，它就能涵蓋包含「翻譯」在內的各種通用任務——例如把輸入寫成「請翻譯成英文：今天天氣真好 →」，模型接著往下生成譯文即可。

---

## 3. Scaled Dot-Product Attention 的數學

§1 用「查資料」比喻帶過 Q/K/V，這一節把它寫成可以計算的式子。本節自成一體；幾何直覺與更多數值範例見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §1–§4。

### 3.1 從輸入到 Q、K、V

設輸入序列有 $T$ 個 token，每個是 $d$ 維向量，堆成矩陣 $X \in \mathbb{R}^{T \times d}$（第 $i$ 列 $x_i$ 是第 $i$ 個 token 的向量）。Attention 先用三個**可訓練的投影矩陣**把 $X$ 映射成 Query、Key、Value：

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$、$W_V \in \mathbb{R}^{d \times d_v}$，於是 $Q, K \in \mathbb{R}^{T \times d_k}$、$V \in \mathbb{R}^{T \times d_v}$。第 $i$ 列 $q_i = x_i W_Q$ 就是「第 $i$ 個 token 的查詢向量」，$k_j$、$v_j$ 同理。

> **Self vs Cross：** 這裡 $Q,K,V$ 都由同一個 $X$ 投影而來，是 **Self-Attention**。若 $Q$ 來自 Decoder、$K,V$ 來自 Encoder，就是 §1 的 **Cross-Attention**——公式一模一樣，只差 $K,V$ 的來源。

### 3.2 相似度分數

用**點積**衡量查詢 $q_i$ 和索引 $k_j$ 的相似度，湊成 $T \times T$ 的分數矩陣：

$$
S = Q K^\top \in \mathbb{R}^{T \times T}, \qquad
S_{ij} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il}\, k_{jl}
$$

$S_{ij}$ 愈大，代表 token $i$ 愈想參考 token $j$。

### 3.3 為什麼要除以 $\sqrt{d_k}$

直接對 $S$ 做 softmax 有個隱患：$d_k$ 一大，點積的數值就容易變得很大，把 softmax 推進「幾乎 one-hot」的飽和區，梯度趨近於 0，訓練變慢。用一個標準假設把這件事量化：設 $q_{il}, k_{jl}$ 互相獨立、平均為 0、變異數為 1。那麼單項乘積的期望與整條點積的變異數為

$$
\mathbb{E}[q_i \cdot k_j] = \sum_{l=1}^{d_k} \mathbb{E}[q_{il}]\,\mathbb{E}[k_{jl}] = 0,
$$

$$
\mathrm{Var}(q_i \cdot k_j) = \sum_{l=1}^{d_k} \mathrm{Var}(q_{il} k_{jl})
= \sum_{l=1}^{d_k} \mathbb{E}[q_{il}^2]\,\mathbb{E}[k_{jl}^2]
= \sum_{l=1}^{d_k} 1 \cdot 1 = d_k
$$

（第二步用獨立性把變異數拆成逐項相加，且因平均為 0 使 $\mathrm{Var}=\mathbb{E}[(\cdot)^2]$。）所以點積的標準差是 $\sqrt{d_k}$。把分數除以 $\sqrt{d_k}$，就把變異數壓回 1、與 $d_k$ 無關，softmax 不會因維度變大而飽和：

$$
\tilde{S} = \frac{Q K^\top}{\sqrt{d_k}}
$$

### 3.4 Softmax 與加權平均

對 $\tilde{S}$ **逐列**做 softmax，得到注意力權重矩陣 $A$：

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad
A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{j'=1}^{T} \exp(\tilde{S}_{ij'})}
$$

每一列 $\sum_j A_{ij} = 1$，是一組機率權重。最後用這組權重對 Value 做加權平均，得到輸出：

$$
C = A V \in \mathbb{R}^{T \times d_v}, \qquad
c_i = \sum_{j=1}^{T} A_{ij}\, v_j
$$

$c_i$ 就是「第 $i$ 個 token 融合了它所關注的其他 token 之後」的新向量。整條式子合起來就是著名的

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

下一節在 $\tilde{S}$ 進 softmax **之前**插入一步遮罩，就得到 GPT 用的 Causal Self-Attention。

---

## 4. Causal Masking：只能看過去，不能偷看未來

§2 說 GPT「只看過去」，不過那還只是一句口號。這一節就要把「只看過去」變成電腦真的做得出來的一步運算。

### 原理

**為什麼需要遮罩？先看人類是怎麼寫字的。** 寫字有先後順序：當你已經寫下「今天天氣真」，正要接下一個字時，後面還沒寫的內容根本還不存在，你只能靠眼前這幾個字加上自己的想法往下推。這種「只能參考前面、看不到後面」是很自然的限制。

**大語言模型的處境卻不太一樣。** 它也要先「學寫字」（訓練），但為了加快速度，訓練時是把一整句完整範文（例如「今天天氣真好」六個字）一次餵進去的——連答案「好」都攤在眼前。這時如果不設限，模型要猜第六個字時就會直接瞄到後面的「好」照抄。這就像考試時把答案放在桌上：分數很漂亮，卻沒真的學會「一個字接下一個字」的規律；等到實際上場、後面的字還沒出現時，它立刻就不會了。

**因果遮罩（Causal Masking）就是用來補這個差距的。** 它在訓練時把「當前位置後面的字」在計算上遮掉，逼模型只能靠左邊（前面）已經出現的字來猜下一個字。這樣模型在「訓練」和「實際生成」時看到的資訊範圍就一致了，練出來的預測能力才真的管用。兩者對照如下：

| 比較項目 | 人類寫字 | 因果遮罩（Causal Masking）|
| :--- | :--- | :--- |
| **能看到的範圍** | 受時間限制，天生看不到還沒寫出來的未來文字。 | 用矩陣運算，人為擋掉序列後面的未來 token。 |
| **猜字的依據** | 眼前已寫出的前文 ＋ 自己的想法。 | 當前位置左側、已經產生的歷史文字。 |
| **學與用的一致性** | 學（閱讀）和用（書寫）本來就都是單向、由前往後。 | 訓練時整句一次輸入，得靠遮罩模擬出實際生成的單向環境。 |

因此大語言模型的訓練邏輯是：只要給它足夠的資料學習，之後你丟一句話給它，它就能「補上」你可能想看的下文——這正是目前對話機器人的基礎邏輯。

### 遮罩實際上遮的是什麼？

這裡要接回 §3 的 Attention 運作：每個字拿自己的 Query 去和每個字的 Key 比對，算出一組「關注分數」$\tilde{S}_{ij}$——分數愈高，代表這個字愈想參考對方。

$T$ 個字兩兩比對，就排成一張 $T \times T$ 的分數表，第 $i$ 列就是「第 $i$ 個字對每個字的關注程度」。

沒有遮罩時，這張表是完整填滿的，每個字都能關注到所有字（包含後面的）。所謂 Causal Masking（因果遮罩），就是動手把這張表「右上半邊」——也就是「往後看」的那些格子——全部封住，只留下對角線和左下半邊：

| 每個字能關注的範圍 | 今 | 天 | 天 | 氣 | 真 | 好 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **今**（第1字） | 👁️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **天**（第2字） | 👁️ | 👁️ | ❌ | ❌ | ❌ | ❌ |
| **天**（第3字） | 👁️ | 👁️ | 👁️ | ❌ | ❌ | ❌ |
| **氣**（第4字） | 👁️ | 👁️ | 👁️ | 👁️ | ❌ | ❌ |
| **真**（第5字） | 👁️ | 👁️ | 👁️ | 👁️ | 👁️ | ❌ |
| **好**（第6字） | 👁️ | 👁️ | 👁️ | 👁️ | 👁️ | 👁️ |

* 👁️ 代表**可以關注**（保留分數）
* ❌ 代表**被遮罩封鎖**（在數學上會被設為 $-\infty$，經過 Softmax 計算後權重會變成 $0$，等同於完全看不到）

👁️ 是保留、❌ 是封住。每一列往右都在某一格之後全變 ❌，因為那些格子代表「看向自己後面的字」，正是要禁止的偷看。留下來的 👁️ 剛好排成一個**下三角形**，這就是它又叫「下三角遮罩」的原因。「因果（causal）」則是指：一個字只受它**之前**的字影響，符合「原因在前、結果在後」的先後順序。

這種「只能看自己和左半邊（過去），不能看右上半邊（未來）」的限制，在時間軸上具備了因果關係（原因在前，結果在後），因此被稱為**因果遮罩（Causal Masking）**。這正是 Decoder-only 模型（如 ChatGPT、Claude）能夠學會正確寫字的關鍵技術。

> **怎麼封住一個格子？** 不是把分數刪掉，而是把它改成負無限大（$-\infty$）。因為分數表接下來要經過 softmax 換算成「關注比例」，而 softmax 裡的 $e^{-\infty}=0$，會讓被封住的格子比例自動變成 0——等於那個字被完全忽略。下一小節就用真實數字把這個過程算一遍。

### 實作：三行程式

把上面的原理翻成 PyTorch，其實只有三行——分別對應「算分數 → 封住右上半 → 換算比例」：

```python
# 建立 T×T 的下三角矩陣（1 = 保留，0 = 封住），就是上面那張 ✓/✗ 表
tril = torch.tril(torch.ones(T, T))

wei = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T)  ① 算出 T×T 關注分數表
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))  # ② 把 0 的格子（未來）改成 -∞
wei = F.softmax(wei, dim=-1)               # ③ softmax 換算成比例，-∞ 自動變 0
```

三行逐一對照原理：

- `tril` 用 `torch.tril`（下三角）產生那張通行證，`1` 就是 ✓、`0` 就是 ✗。
- `masked_fill(... == 0, -inf)` 把所有 ✗ 的格子（望向未來的位置）改寫成 $-\infty$，正是「封住格子」那一步。
- `F.softmax` 把每一列的分數換算成加總為 1 的比例，$e^{-\infty}=0$ 讓被封住的格子比例歸零（softmax 的細節見 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)）。

原理和程式都到位了，接著代進真實數字算一遍。

### 數值演示（T=3）

上面是「形狀」，這裡代進真實數字跑一遍，看權重怎麼被算出來。為了讓手算清楚，改用 3 個 token（$T=3$）。假設 Q、K 相乘並除以 $\sqrt{d}$ 之後（這個縮放步驟見 §3.3）得到的原始注意力分數矩陣如下，第 $i$ 列代表「位置 $i$ 對每個位置的關注分數」：

```
E_raw（Q·Kᵀ 縮放後的原始分數）：
位置 0：[2.0,  1.0,  0.0]
位置 1：[1.0,  2.0,  1.0]
位置 2：[0.0,  1.0,  2.0]
```

**套用 Causal Mask（上三角設為 -∞）：**

```
E_masked：
位置 0：[2.0,  -inf, -inf]   → 只能看自己
位置 1：[1.0,   2.0, -inf]   → 能看 0 和 1
位置 2：[0.0,   1.0,  2.0]   → 能看全部
```

**逐行做 softmax（-∞ → 0）：**

| 位置 | softmax 輸入 | softmax 輸出 |
|---|---|---|
| 0 | `[2.0]`（只有自己）| `[1.000, 0.000, 0.000]` |
| 1 | `[1.0, 2.0]` | `[0.269, 0.731, 0.000]` |
| 2 | `[0.0, 1.0, 2.0]` | `[0.090, 0.245, 0.665]` |

以位置 1 為例驗算：$e^{1.0} \approx 2.718$、$e^{2.0} \approx 7.389$，總和 $= 10.107$，故 $2.718 / 10.107 \approx 0.269$、$7.389 / 10.107 \approx 0.731$；被遮罩的位置因為 $e^{-\infty} = 0$，權重恰為 0。

**結果解讀：**
- 位置 0：100% 關注自己（沒有其他可看）
- 位置 1：73.1% 關注位置 1，26.9% 關注位置 0
- 位置 2：66.5% 關注位置 2，其餘分給位置 0、1

每一列加總恰好等於 1（權重是機率分佈），且未來位置的權重都是 0——位置 0 完全看不到 1、2，位置 1 看不到 2。這就是因果遮罩在訓練中防止「作弊」的方式：每個位置只把注意力分給自己和過去。

### 關鍵差異

把有沒有這張遮罩帶來的差別攤開來看，也正好對比出 GPT 和 BERT 兩大家族的根本分歧：

| | 原始 Encoder（BERT）| Causal Decoder（GPT）|
|---|---|---|
| 遮罩 | 無（全部可看）| 下三角（只看過去）|
| 注意力矩陣 | 對稱 | 下三角 |
| 適合任務 | 理解（分類/問答）| 生成（逐 token 輸出）|

差別的根源只有一件事：**要不要看未來**。BERT 做的是「理解」，把整句攤在眼前前後對照最有利，所以不加遮罩、注意力矩陣左右對稱；GPT 做的是「生成」，必須逐字往下寫、不能偷看，所以套上下三角遮罩。

> Encoder（BERT）那一欄的完整展開——為何雙向、MLM 怎麼訓練、`[CLS]`/`[SEP]` 的角色——見 [`07-bert-encoder-only.md`](07-bert-encoder-only.md)。

---

## 5. Multi-Head Attention 的數學

§3、§4 講的是**單一組** $Q/K/V$ 的注意力（單頭）。但一個句子裡的關係有很多種——語法上的主謂、語意上的指代、位置上的鄰近……單頭只能學一種「關注模式」。**多頭注意力**讓多組投影並行，各自關注不同面向，再把結果合起來。

### 5.1 把維度切給多個頭

設模型維度 $d$ 可被頭數 $H$ 整除，每頭分到 $d_k = d / H$ 維。第 $h$ 個頭有自己的一組投影矩陣 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d \times d_k}$，各自算一份（含 §4 的因果遮罩）注意力輸出：

$$
C^{(h)} = \text{Attention}\!\left(X W_Q^{(h)},\; X W_K^{(h)},\; X W_V^{(h)}\right) \in \mathbb{R}^{T \times d_k},
\qquad h = 1, \ldots, H
$$

### 5.2 拼接與輸出投影

把 $H$ 個頭的輸出沿特徵維**拼接**回 $d$ 維，再乘一個輸出投影矩陣 $W_O \in \mathbb{R}^{d \times d}$：

$$
\text{MultiHead}(X) = \underbrace{\text{Concat}\!\left(C^{(1)}, \ldots, C^{(H)}\right)}_{\in\; \mathbb{R}^{T \times d}} \, W_O
$$

（因為 $H \cdot d_k = d$，拼接後剛好回到 $d$ 維。）

### 5.3 $W_O$ 在做什麼？

拼接只是把各頭的輸出**塞進彼此不重疊的座標區塊**——第 1 頭佔前 $d_k$ 維、第 2 頭佔接下來 $d_k$ 維……此時各頭的資訊是「分隔」的。$W_O$ 是一個（一般可逆的）**基底變換**，把這些分區的資訊重新混合、重組成一個共用的表示，讓後面的層可以跨頭整合。少了 $W_O$，各頭的輸出就永遠卡在自己的座標區塊裡無法互通（這個「投影＝可逆基底變換」的觀點見 [`03b3-transformer-architecture-example.md`](03b3-transformer-architecture-example.md) §0）。

> 多頭的逐步數值範例見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §5–§5.6。

---

## 6. Position-wise FFN 的數學

Attention 負責「跨位置」混合資訊；混完之後，還需要一層對**每個位置各自**做非線性變換的網路，這就是 Position-wise Feed-Forward Network（FFN）。「Position-wise」指同一個 MLP 套用到序列的每一個位置（每一列），彼此參數共用、互不干擾。

對單一位置的向量 $z \in \mathbb{R}^{d}$：

$$
\text{FFN}(z) = \text{ReLU}(z W_1 + b_1)\, W_2 + b_2,
\qquad \text{ReLU}(u) = \max(0, u)
$$

其中 $W_1 \in \mathbb{R}^{d \times d_{ff}}$、$W_2 \in \mathbb{R}^{d_{ff} \times d}$，慣例 $d_{ff} = 4d$。維度先擴大再壓縮，形成一條**形狀鏈**：

$$
\underbrace{z}_{d}
\;\xrightarrow{\;W_1\;}\;
\underbrace{\cdot}_{4d}
\;\xrightarrow{\;\text{ReLU}\;}\;
\underbrace{\cdot}_{4d}
\;\xrightarrow{\;W_2\;}\;
\underbrace{\text{FFN}(z)}_{d}
$$

先擴到 $4d$ 給模型較大的中間空間做非線性特徵組合，再投影回 $d$ 維以便接續殘差相加（見 §7）。ReLU 提供非線性，否則兩層線性可合併成一層、失去表達力。

> FFN 的角色與數值範例見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §6.4。

---

## 7. LayerNorm 與 Pre-LN Block 的數學

把 §3–§6 的模組疊成一個 Transformer Block，還需要兩個黏合劑：**LayerNorm**（穩定數值）與 **Residual（殘差）**（穩定梯度）。

### 7.1 LayerNorm

對單一 token 的向量 $z \in \mathbb{R}^{d}$，LayerNorm 沿它自己的 $d$ 個特徵做標準化（注意：是對**特徵維**，不是對 batch 或位置維）：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} z_i, \qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (z_i - \mu)^2,
$$

$$
\text{LN}(z) = \gamma \odot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

先把向量拉成平均 0、變異數 1，再用可訓練的 $\gamma, \beta \in \mathbb{R}^{d}$（逐元素縮放與平移）還給模型調整的自由；$\epsilon$ 防止除以 0。

### 7.2 Pre-LN Block

原始論文把 LayerNorm 放在殘差相加**之後**（Post-LN）：

$$
Z' = \text{LN}\bigl(X + \text{MultiHead}(X)\bigr)
$$

nanoGPT 等現代模型改放在子層**之前**（Pre-LN），一個 Block 的兩個子層寫成：

$$
X_1 = X + \text{MultiHead}\bigl(\text{LN}(X)\bigr), \qquad
X_2 = X_1 + \text{FFN}\bigl(\text{LN}(X_1)\bigr)
$$

即通式 $x \leftarrow x + f(\text{LN}(x))$，其中殘差項 $x+\,\cdot\,$ 讓輸入可以「直通」到輸出。

### 7.3 為什麼 Pre-LN 的梯度比較穩定

看反向傳播時梯度怎麼流。對 Pre-LN 的 $y = x + f(\text{LN}(x))$，對 $x$ 微分：

$$
\frac{\partial y}{\partial x} = I + \frac{\partial f(\text{LN}(x))}{\partial x}
$$

那個**單位矩陣 $I$** 就是關鍵：它讓上游梯度 $\dfrac{\partial L}{\partial y}$ 有一條**完全不經過 LayerNorm、也不衰減**的路徑直接傳回 $x$——

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial f(\text{LN}(x))}{\partial x}\right)
= \underbrace{\frac{\partial L}{\partial y}}_{\text{恆等直通}} + \frac{\partial L}{\partial y}\frac{\partial f(\text{LN}(x))}{\partial x}
$$

疊 $L$ 層時，這條恆等路徑保證最深處仍能收到一份未被反覆縮放的梯度。反觀 Post-LN 的 $\text{LN}(x + f(x))$，梯度每穿一層都要乘上一次 LayerNorm 的 Jacobian，$L$ 一大尺度就容易爆炸或消失，所以才需要 learning-rate warm-up「小心起步」。

| | Post-LN | Pre-LN |
|---|---|---|
| 訓練穩定性 | 需要 warm-up | 更穩定，學習率更寬容 |
| 深層表現 | 容易梯度爆炸/消失 | 梯度流更均勻 |
| 代表模型 | 原始 Transformer | GPT-2、LLaMA、nanoGPT |

> LayerNorm 的完整 Jacobian 與 Residual 的梯度恆等推導見 [`05-backpropagation.md`](05-backpropagation.md) §5.9／§5.10。Pre-LN 在 nanoGPT 程式裡長什麼樣，見 [`04b`](04b-nanogpt-walkthrough.md) §4、§7。

---

## 8. Token Embedding 與位置編碼的數學

前面幾節的輸入 $X$ 是「一堆 $d$ 維向量」，但原始輸入其實是**離散的 token id**。這一節補上「id → 向量」這一步。

### 8.1 Token Embedding：查表就是 one-hot 乘法

設詞彙表大小為 $V$，embedding 矩陣 $E \in \mathbb{R}^{V \times d}$（每一列是一個 token 的向量）。token id $t_i$ 對應的向量就是取 $E$ 的第 $t_i$ 列：

$$
x_i = E[t_i] = e_{t_i}^\top E
$$

其中 $e_{t_i} \in \mathbb{R}^{V}$ 是第 $t_i$ 個位置為 1 的 one-hot 向量。所以「查表（Lookup）」在數學上等於一次 one-hot 乘矩陣，實作上則是 $O(1)$ 的索引，不必真的做乘法。（形式化亦見 [`01b-prerequisites-math.md`](01b-prerequisites-math.md) §2；它如何被訓練見 §10。）

### 8.2 位置編碼：把順序加回去

Attention 本身對「順序」無感——打亂 token 的位置，$QK^\top$ 的集合不變。所以要**額外注入位置資訊**。nanoGPT 用 **Learned PE**：另有一個位置矩陣 $P \in \mathbb{R}^{T_{\max} \times d}$，位置 $i$ 取第 $i$ 列 $P[i]$，直接與 token 向量**相加**：

$$
h_{0,i} = \underbrace{E[t_i]}_{\text{token 語意}} + \underbrace{P[i]}_{\text{位置}}
$$

相加（而非拼接）能保持維度仍是 $d$，模型有能力在後續層把兩者解開。另一種做法是固定公式的 **Sinusoidal PE**（不需訓練、理論上可外推到更長序列），推導見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §7.2；Learned PE 的細節與兩者比較見 03a §7.5 與 [`04b`](04b-nanogpt-walkthrough.md) §5。

---

## 9. Next-token Prediction 與 Cross-Entropy

架構齊備了，最後定義**訓練目標**：讓模型對每個位置預測「下一個 token」。

### 9.1 資料準備

給定文字序列，把它切成 `(input, target)` 對：

```
文字： h  e  l  l  o
input:  [h, e, l, l]   (前 T 個 token)
target: [e, l, l, o]   (後 T 個 token，即 input 右移一位)
```

位置 $i$ 的 **input** 是 $x_i$，**target** 是 $x_{i+1}$。

### 9.2 損失函數

模型在位置 $i$ 輸出一個 logit 向量 $z_i \in \mathbb{R}^{V}$，經 softmax 成機率分佈 $p^{(i)} = \mathrm{softmax}(z_i)$，再與正確答案 $y_i = x_{i+1}$ 用 **Cross-Entropy** 比較。整個序列的損失是各位置的平均：

$$
p^{(i)}_k = \frac{\exp(z_{i,k})}{\sum_{k'} \exp(z_{i,k'})}, \qquad
L = -\frac{1}{T}\sum_{i=1}^{T} \log p^{(i)}_{y_i}
$$

程式上就是一行：

```python
# logits: (B, T, vocab_size)
# targets: (B, T)
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
```

有了 §4 的 Causal Mask，位置 $i$ 的 logit 只用到了 $x_1, \ldots, x_i$ 的資訊，預測 $x_{i+1}$，**不會洩漏未來**。因此一個序列可以同時訓練 $T$ 個預測任務，訓練效率極高。

---

## 10. 反向傳播：從 loss 到 Embedding

訓練程式碼只有三行：

```python
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
loss.backward()
optimizer.step()
```

但每次 `loss.backward()` 都完整走一遍以下路徑，梯度從 loss 一路流回 `token_embedding.weight`（即 Embedding 矩陣 $E$）。括號內為關鍵公式，完整推導見 [`05-backpropagation.md`](05-backpropagation.md) §6：

**前向傳播（forward pass）**

$$
\underbrace{t_i}_{\text{token\_id }(B,T)}
\;\xrightarrow{\;x_i = E[t_i]\;(\text{Lookup，見 §8.1})\;}\;
\underbrace{x_{\text{embed}}}_{(B,T,d)}
$$

$$
h_0 = x_{\text{embed}} + p_{\text{embed}}
\;\xrightarrow{\;\text{Block}_1 \to \cdots \to \text{Block}_L\;(\text{§5–§7})\;}\;
\underbrace{h_L}_{(B,T,d)}
$$

$$
\underbrace{z_i = \mathrm{LN}(h_{L,i}) \, W_{lm}^\top}_{\text{LayerNorm} \to \text{lm\_head}}
\;\longrightarrow\;
\underbrace{\text{logits}}_{(B,T,V)}
\;\xrightarrow{\;p^{(i)}_k = \mathrm{softmax}(z_i)_k\;}\;
L = -\frac{1}{T}\sum_i \log p^{(i)}_{y_i}
\;\;(\text{loss，純量})
$$

**反向傳播（backward pass）**

**Step 1｜Cross-Entropy + Softmax：**

$$
\delta_i = \frac{\partial L}{\partial z_i}, \qquad
\delta_i^{(k)} = \frac{1}{T}\bigl(p^{(i)}_k - \mathbb{1}[k = y_i]\bigr)
$$

在正確 token 的位置減 $1/T$，其餘位置加 softmax 機率 $/T$。

**Step 2｜lm_head 反向（$z_i = \mathrm{LN}(h)_i \, W_{lm}^\top$）：**

$$
\frac{\partial L}{\partial \mathrm{LN}(h_i)} = \delta_i \, W_{lm}
\qquad (\text{梯度流向上一層，}d\text{ 維})
$$

$$
\frac{\partial L}{\partial W_{lm}} = \sum_i \delta_i^\top \, \mathrm{LN}(h_i)^\top
\qquad (\text{lm\_head 的參數梯度，稠密 }V \times d\text{；}1/T\text{ 已含在 }\delta_i\text{ 中})
$$

**Step 3｜穿越 LayerNorm、Residual、FFN、Attention：**

梯度沿 §5–§7 的每個模組反向傳回（Pre-LN 的殘差直通見 §7.3），最終到達 $x_{\text{embed}}$ 的梯度記為 $g_i \in \mathbb{R}^d$。

**Step 4｜Lookup 反向（$x_i = E[t_i]$）：**

$$
\frac{\partial L}{\partial E[k]} = \sum_{i:\, t_i = k} g_i
\qquad (E\text{ 的梯度，稀疏：只有出現過的列非零})
$$

**Step 5｜Optimizer 更新：**

$$
E[k] \leftarrow E[k] - \eta \cdot \frac{\partial L}{\partial E[k]}
$$

> **註（Weight Tying）：** Karpathy 的原版 nanoGPT 讓 `lm_head.weight` 與 `token_embedding.weight` 共用同一份矩陣（$W_{lm} = E$），此時 Step 2 的稠密梯度與 Step 4 的稀疏梯度會**累加**到同一個 $E$ 上。完整的雙通道梯度推導見 [`05-backpropagation.md`](05-backpropagation.md) §6。（本倉庫 NB4 未做 Weight Tying，見 [`04b`](04b-nanogpt-walkthrough.md) §5。）

**三個關鍵特性：**

1. **稀疏更新**：Lookup 反向（Step 4）只更新本 batch 出現過的 token 列，沒出現的 token 其 embedding 本步完全不動。
2. **同 token 累加**：token $k$ 在同一序列出現 $m$ 次，Step 4 的梯度是 $m$ 個 $g_i$ 的加總。
3. **直覺含義**：出現頻繁的 token 每步都被更新，embedding 收斂快；稀有 token 需要大量訓練步驟才被充分觸及。

### 前向／反向 ↔ nanoGPT 元件對照

本文每個數學階段，都對應 [`04b`](04b-nanogpt-walkthrough.md) 裡的一段程式：

| 本文數學 | nanoGPT 程式（[`04b`](04b-nanogpt-walkthrough.md)）|
|---|---|
| §3 Scaled Dot-Product ＋ §4 Causal Mask | §1 `Head` |
| §5 Multi-Head ＋ $W_O$ | §2 `MultiHeadAttention` |
| §6 FFN | §3 `FeedForward` |
| §7 LayerNorm／Pre-LN Block | §4 `Block`、§7 Pre-LN vs Post-LN |
| §8 Embedding／Learned PE | §5 `GPT`（`token_embedding`／`position_embedding`）|
| §9 Cross-Entropy、§10 梯度鏈 | §5 `lm_head`、§6 對照總表 |

---

## 下一步

**對照程式實作：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md)

把本文每個數學節對回 nanoGPT 的 `Head` / `MultiHeadAttention` / `FeedForward` / `Block` / `GPT`，並補上 Tokenizer、自迴歸生成、KV Cache 與「打開 nanoGPT 之前的速查清單」，然後打開 [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)。

**若想深入訓練背後的完整梯度推導：**
→ [`05-backpropagation.md`](05-backpropagation.md) — Self-Attention 與 LayerNorm 的完整梯度推導
→ [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb) — NumPy 手刻反向傳播

**若想銜接 LLaMA 等當代模型：**
→ [`06-modern-transformer-variants.md`](06-modern-transformer-variants.md) — RMSNorm、RoPE 等 nanoGPT → LLaMA 之間的架構演化
