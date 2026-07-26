# 05a1｜向前傳播（Forward Pass）：GPT Decoder-Only 的數學（符號推導）

> **適合對象：** 讀完 [`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md) 的基本概念與 Pipeline 後，想逐節推導 GPT 前向傳播每個模組數學式的讀者。**數學只需要高中程度**：會矩陣乘法、看得懂 $\exp$／$\log$、知道什麼是平均與變異數就夠了。記號約定與形狀檢查的習慣在「開始之前」一節從頭講起。
>
> **讀完後你能做什麼（依本文順序）：**
> - §1　推導 Scaled Dot-Product Attention 的四個步驟（QKV 投影 → 點積分數 → 縮放 → softmax 加權平均），並說明為何要除以 $\sqrt{d_k}$
> - §2　描述 Causal Masking 的原理與數值，說出「$-\infty$ → softmax → 權重 0」這條鏈
> - §3　寫出 Multi-Head 的切維、拼接與輸出投影 $W_O$，並說明 $W_O$ 少不得的理由
> - §4　寫出 Position-wise FFN 的兩層形狀鏈，並解釋為何要先擴大再壓縮、為何需要 ReLU
> - §5　寫出 LayerNorm 與 Pre-LN Block 的數學形式，並解釋 Pre-LN 的梯度為何比較穩定
> - §6　說明 Token Embedding 是 one-hot 乘法、位置編碼為何要相加
> - §7　寫出 Next-token Prediction 的資料切法與 Cross-Entropy 損失，並說明為何一次 forward 能同時訓練 $T$ 個預測任務
>
> **前置文件：** [`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)（基本概念與 Pipeline）、[`03a-transformer-architecture.md`](03a-transformer-architecture.md)
>
> **對照實作：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md)（每段 nanoGPT 程式如何落實本文數學）→ [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)
>
> **數值演算（選讀續篇）：** → [`05a2-forward-example.md`](05a2-forward-example.md)（用一組範例資料把本文每個階段實際算一次，T=2、d=3）
>
> **學完後的下一步：** → [`05b1-backward-propagation.md`](05b1-backward-propagation.md)（反向傳播數學）→ [`05b2-backward-example.md`](05b2-backward-example.md)（反向數值範例）

---

## 目錄

1. Scaled Dot-Product Attention 的數學
2. Causal Masking：只能看過去，不能偷看未來
3. Multi-Head Attention 的數學
4. Position-wise FFN 的數學
5. LayerNorm 與 Pre-LN Block 的數學
6. Token Embedding 與位置編碼的數學
7. Next-token Prediction 與 Cross-Entropy

> **本文的定位：** [`04a`](04a-gpt-decoder-only.md) 講「基本概念、架構差異與 Pipeline 總覽」；本文（05a1）補上前向每個模組的**符號數學推導**；配套的 [`05a2`](05a2-forward-example.md) 用一組範例資料把每個階段實算一次。反向則見 [`05b1`](05b1-backward-propagation.md)（數學）＋ [`05b2`](05b2-backward-example.md)（數值）。建議 04a → 05a1 → 05a2 → 05b1 → 05b2 → 04b → NB4 依序讀。

> **本文的行進順序（一句話版）：** 一句話進來，先變成一堆向量（§6），再讓這些向量互相參考、彼此融合（§1–§3），各自加工一次（§4），中間靠兩個黏合劑穩住（§5），最後輸出「下一個字是什麼」的預測並算出分數（§7）。**§1–§5 先把零件逐個講清楚，§6、§7 才把頭尾接上**——如果你想照資料真正流動的順序讀，也可以走 §6 → §1 → §2 → §3 → §4 → §5 → §7。

---

## 開始之前：三個記號約定

七節的公式都建立在同一套記號上，先花兩分鐘講清楚，後面就不會卡。

**（一）一句話 ＝ 一個矩陣。**

一個長度 $T$ 的句子，每個 token 用一個 $d$ 維向量表示，全部**橫著疊起來**變成矩陣：

$$
X \in \mathbb{R}^{T \times d}
$$

- **列（row）** ＝ 一個 token。第 $i$ 列寫成 $x_i$，是一個 $1\times d$ 的**列向量**。
- **欄（column）** ＝ 一個特徵維度。

全文所有向量都是列向量，矩陣乘法一律寫成「列向量乘矩陣」（$x_i W$）。這樣 $XW$ 的第 $i$ 列剛好就是 $x_i W$，整批算和逐個算完全一致。

**（二）內積 ＝ 相似度。**

兩個同維度的向量 $a, b \in \mathbb{R}^{n}$ 的內積（點積）是「逐項相乘再全部加起來」：

$$
a \cdot b = \sum_{l=1}^{n} a_l\, b_l
$$

**為什麼它可以當相似度？** 因為 $a\cdot b = \|a\|\,\|b\|\cos\theta$。兩個向量方向愈接近，$\cos\theta$ 愈大、內積愈大。整個 Attention 的核心就建在這一件事上。

**（三）養成形狀檢查的習慣。**

矩陣乘法只有在「前者的欄數 ＝ 後者的列數」時才成立：

$$
(T \times d) \cdot (d \times d_k) = (T \times d_k)
$$

讀本文時，每看到一條式子就先確認形狀對不對——**形狀對不上，公式一定有問題**。這個習慣到了反向傳播（[`05b1`](05b1-backward-propagation.md) §1.6）會變成主要的除錯工具。

本文用到的符號一覽：

| 符號 | 意思 | 典型大小 |
|---|---|---|
| $T$ | 序列長度（一次看幾個 token）| 8～2048 |
| $d$ | 模型維度（每個 token 幾維向量）| 64～4096 |
| $d_k, d_v$ | 單一個頭的 Query／Key、Value 維度 | $d/H$ |
| $H$ | 注意力頭數 | 4～32 |
| $V$ | 詞彙表大小（總共有幾種 token）| 65～50257 |
| $d_{ff}$ | FFN 中間層維度 | 通常 $4d$ |

---

## 1. Scaled Dot-Product Attention 的數學

[`04a`](04a-gpt-decoder-only.md) §1 用「查資料」比喻帶過 Q/K/V，這一節把它寫成可以計算的式子。本節自成一體；幾何直覺與更多數值範例見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §1–§4。

**先講這一節要解決的問題。** 讀到「小狗很可愛，**牠**在睡覺」時，「牠」這個字本身沒有任何意義——它的意思完全來自前面的「小狗」。所以模型需要一個機制，讓每個 token 能**去看句子裡的其他 token，並自己決定要參考誰、參考多少**。這個機制就是 Attention，做完之後每個 token 的向量都被「上下文」重新染色過一次。

整節分四步，順序不能顛倒：

$$
\text{①投影出 Q/K/V} \;\to\; \text{②算兩兩相似度} \;\to\; \text{③縮放} \;\to\; \text{④softmax 加權平均}
$$

### 1.1 從輸入到 Q、K、V

設輸入序列有 $T$ 個 token，每個是 $d$ 維向量，堆成矩陣 $X \in \mathbb{R}^{T \times d}$（第 $i$ 列 $x_i$ 是第 $i$ 個 token 的向量）。Attention 先用三個**可訓練的投影矩陣**把 $X$ 映射成 Query、Key、Value：

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

其中 $W_Q, W_K \in \mathbb{R}^{d \times d_k}$、$W_V \in \mathbb{R}^{d \times d_v}$，於是 $Q, K \in \mathbb{R}^{T \times d_k}$、$V \in \mathbb{R}^{T \times d_v}$。第 $i$ 列 $q_i = x_i W_Q$ 就是「第 $i$ 個 token 的查詢向量」，$k_j$、$v_j$ 同理。

**為什麼同一個 $x_i$ 要拆成三個？** 因為一個 token 在對話裡同時扮演三種角色，把它們分開才不會互相打架：

| | 角色 | 白話 | 對應圖書館的比喻 |
|---|---|---|---|
| $q_i$ | Query（查詢）| 我現在想找什麼 | 你手上寫的搜尋關鍵字 |
| $k_j$ | Key（索引）| 我這裡有什麼、方便被誰找到 | 書背上的標題 |
| $v_j$ | Value（內容）| 被找到之後，我實際要交出去的資訊 | 書裡真正的內容 |

關鍵在於 **Key 和 Value 是分開的**：一本書怎麼「被搜尋到」（標題）和它「內容是什麼」可以不一樣。如果不拆開，模型就只能用「內容本身」當索引，表達力會少一大截（這個對稱性限制的詳細討論見 [`02-attention-intuition.md`](02-attention-intuition.md)）。

> **Self vs Cross：** 這裡 $Q,K,V$ 都由同一個 $X$ 投影而來，是 **Self-Attention**。若 $Q$ 來自 Decoder、$K,V$ 來自 Encoder，就是 [`04a`](04a-gpt-decoder-only.md) §1 的 **Cross-Attention**——公式一模一樣，只差 $K,V$ 的來源。

### 1.2 相似度分數

用**點積**衡量查詢 $q_i$ 和索引 $k_j$ 的相似度，湊成 $T \times T$ 的分數矩陣：

$$
S = Q K^\top \in \mathbb{R}^{T \times T}, \qquad
S_{ij} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il}\, k_{jl}
$$

$S_{ij}$ 愈大，代表 token $i$ 愈想參考 token $j$。

**這張表怎麼讀：** $S$ 是一張 $T\times T$ 的「關注分數表」，第 $i$ 列 $S_{i,:}$ 就是「第 $i$ 個 token 對句子裡每一個 token 的關注程度」。$T$ 個 token 兩兩比對，總共 $T^2$ 個分數——這也是 Attention 的計算量隨序列長度平方成長的原因（後續怎麼緩解見 [`06`](06-modern-transformer-variants.md) §5 Flash Attention）。

> **為什麼是 $QK^\top$ 而不是 $QK$？** 因為要讓「$Q$ 的第 $i$ 列」去點「$K$ 的第 $j$ 列」。矩陣乘法是「列點欄」，所以要先把 $K$ 轉置、讓原本的列變成欄。形狀檢查：$(T\times d_k)\cdot(d_k\times T) = T\times T$ ✓。

### 1.3 為什麼要除以 $\sqrt{d_k}$

直接對 $S$ 做 softmax 有個隱患：$d_k$ 一大，點積的數值就容易變得很大，把 softmax 推進「幾乎 one-hot」的飽和區，梯度趨近於 0，訓練變慢。

**先看直覺。** 點積是 $d_k$ 個乘積相加。$d_k$ 愈大，加的項數愈多，總和的擺盪幅度就愈大——像丟 100 顆骰子加總，總和的波動一定比丟 4 顆大。而 softmax 對「分數之間的差距」極度敏感：兩個分數差 2 分時，softmax 給的比例約是 88% 對 12%；差 10 分時就變成約 99.995% 對 0.005%，幾乎只剩一個贏家。**一旦變成「贏者全拿」，模型就等於沒在權衡，而且梯度會趨近於零（見 [`05b1`](05b1-backward-propagation.md) §3.2）。**

用一個標準假設把這件事量化：設 $q_{il}, k_{jl}$ 互相獨立、平均為 0、變異數為 1。那麼單項乘積的期望與整條點積的變異數為

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

**一句話總結：** 除以標準差 $\sqrt{d_k}$ 是把分數「標準化」，讓 softmax 的輸入不論模型多大都待在同一個合理範圍。

> **注意除的是 $d_k$（單頭的維度），不是 $d$。** 多頭時每個頭只看 $d_k = d/H$ 維，縮放要跟著頭的維度走。這是實作上很常見的錯誤，見 [`04b`](04b-nanogpt-walkthrough.md) §1 的註。

### 1.4 Softmax 與加權平均

對 $\tilde{S}$ **逐列**做 softmax，得到注意力權重矩陣 $A$：

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad
A_{ij} = \frac{\exp(\tilde{S}_{ij})}{\sum_{j'=1}^{T} \exp(\tilde{S}_{ij'})}
$$

**softmax 在做什麼（三步）：** ①每個分數取 $\exp$——負數變正數、大的變更大；②全部加起來當分母；③各自除以分母。結果是一組**全為正、加起來剛好等於 1** 的比例。也就是把「一排任意大小的分數」翻譯成「一組百分比」（完整說明見 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)）。

**注意是「逐列」做。** 每一列自己歸一化成 100%，代表「第 $i$ 個 token 把它的注意力預算分配給誰」。列與列之間互不相干。

每一列 $\sum_j A_{ij} = 1$，是一組機率權重。最後用這組權重對 Value 做加權平均，得到輸出：

$$
C = A V \in \mathbb{R}^{T \times d_v}, \qquad
c_i = \sum_{j=1}^{T} A_{ij}\, v_j
$$

$c_i$ 就是「第 $i$ 個 token 融合了它所關注的其他 token 之後」的新向量。

**用果汁比喻收尾：** $v_j$ 是各種果汁，$A_{ij}$ 是「第 $i$ 杯要倒多少比例的第 $j$ 種果汁」，$c_i$ 就是調出來的那一杯。因為比例加起來是 100%，這確實是一次「按比例調配」。

整條式子合起來就是著名的

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V .
$$

> **讀完這一節，你會：**
> - 說出 Q、K、V 三個角色各自代表什麼，以及為什麼 Key 和 Value 要分開
> - 寫出 $S = QK^\top$，並解釋為什麼要轉置
> - 用「變異數等於 $d_k$」的論證說明為何要除以 $\sqrt{d_k}$，以及除的為什麼是 $d_k$ 不是 $d$
> - 依 ①投影 →②點積 →③縮放 →④softmax 加權平均 的順序寫出完整的 Attention 公式

下一節在 $\tilde{S}$ 進 softmax **之前**插入一步遮罩，就得到 GPT 用的 Causal Self-Attention。

---

## 2. Causal Masking：只能看過去，不能偷看未來

[`04a`](04a-gpt-decoder-only.md) §2 說 GPT「只看過去」，不過那還只是一句口號。這一節就要把「只看過去」變成電腦真的做得出來的一步運算。

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

這裡要接回 §1 的 Attention 運作：每個字拿自己的 Query 去和每個字的 Key 比對，算出一組「關注分數」$\tilde{S}_{ij}$——分數愈高，代表這個字愈想參考對方。

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

**為什麼不乾脆把分數設成 0？** 因為 softmax 之後 $e^{0}=1$ 並不是 0，那個位置反而會拿到一份不小的權重。真正要的是「經過 softmax 後權重為 0」，倒推回去，softmax 之前就必須是 $-\infty$。

### 實作：三行程式

把上面的原理翻成 PyTorch，其實只有三行——分別對應「算分數 → 封住右上半 → 換算比例」：

```python
# 建立 T×T 的下三角矩陣（1 = 保留，0 = 封住），就是上面那張 ✓/✗ 表
tril = torch.tril(torch.ones(T, T))

wei = q @ k.transpose(-2, -1) * d_k**-0.5  # (B, T, T)  ① 算出 T×T 關注分數表（d_k = k.shape[-1]）
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))  # ② 把 0 的格子（未來）改成 -∞
wei = F.softmax(wei, dim=-1)               # ③ softmax 換算成比例，-∞ 自動變 0
```

三行逐一對照原理：

- `tril` 用 `torch.tril`（下三角）產生那張通行證，`1` 就是 ✓、`0` 就是 ✗。
- `masked_fill(... == 0, -inf)` 把所有 ✗ 的格子（望向未來的位置）改寫成 $-\infty$，正是「封住格子」那一步。
- `F.softmax` 把每一列的分數換算成加總為 1 的比例，$e^{-\infty}=0$ 讓被封住的格子比例歸零（softmax 的細節見 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)）。

原理和程式都到位了，接著代進真實數字算一遍。

### 數值演示（T=3）

上面是「形狀」，這裡代進真實數字跑一遍，看權重怎麼被算出來。為了讓手算清楚，改用 3 個 token（$T=3$）。假設 Q、K 相乘並除以 $\sqrt{d}$ 之後（這個縮放步驟見 §1.3）得到的原始注意力分數矩陣如下，第 $i$ 列代表「位置 $i$ 對每個位置的關注分數」：

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

> **順帶留意位置 0 那一列。** 它的輸出是完美的 one-hot `[1, 0, 0]`——因為它只有自己可看，沒得選。這件事在反向傳播時會有後果：one-hot 會讓 softmax 的梯度整個歸零，所以**第 0 列學不到東西**（見 [`05b1`](05b1-backward-propagation.md) §8.5）。這不是 bug，是因果遮罩的必然結果。

### 關鍵差異

把有沒有這張遮罩帶來的差別攤開來看，也正好對比出 GPT 和 BERT 兩大家族的根本分歧：

| | 原始 Encoder（BERT）| Causal Decoder（GPT）|
|---|---|---|
| 遮罩 | 無（全部可看）| 下三角（只看過去）|
| 注意力矩陣 | 對稱 | 下三角 |
| 適合任務 | 理解（分類/問答）| 生成（逐 token 輸出）|

差別的根源只有一件事：**要不要看未來**。BERT 做的是「理解」，把整句攤在眼前前後對照最有利，所以不加遮罩、注意力矩陣左右對稱；GPT 做的是「生成」，必須逐字往下寫、不能偷看，所以套上下三角遮罩。

> Encoder（BERT）那一欄的完整展開——為何雙向、MLM 怎麼訓練、`[CLS]`/`[SEP]` 的角色——見 [`07-bert-encoder-only.md`](07-bert-encoder-only.md)。

> **讀完這一節，你會：**
> - 用「考試把答案放桌上」解釋為什麼訓練時需要因果遮罩
> - 畫出 $T\times T$ 的下三角遮罩，並說出每一列的意義
> - 說明為什麼要填 $-\infty$ 而不是 0，並寫出對應的三行 PyTorch
> - 手算一列被遮罩後的 softmax，驗證未來位置的權重確實是 0

---

## 3. Multi-Head Attention 的數學

§1、§2 講的是**單一組** $Q/K/V$ 的注意力（單頭）。但一個句子裡的關係有很多種——語法上的主謂、語意上的指代、位置上的鄰近……單頭只能學一種「關注模式」。**多頭注意力**讓多組投影並行，各自關注不同面向，再把結果合起來。

**比喻：** 單頭像是只派一個人去讀這篇文章，他只能挑一個角度做筆記。多頭像是派 8 個人各自負責一個角度——一個盯文法、一個盯指代、一個盯情緒——讀完再把 8 份筆記合成一份。

### 3.1 把維度切給多個頭

設模型維度 $d$ 可被頭數 $H$ 整除，每頭分到 $d_k = d / H$ 維。第 $h$ 個頭有自己的一組投影矩陣 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)} \in \mathbb{R}^{d \times d_k}$，各自算一份（含 §2 的因果遮罩）注意力輸出：

$$
C^{(h)} = \text{Attention}\!\left(X W_Q^{(h)},\; X W_K^{(h)},\; X W_V^{(h)}\right) \in \mathbb{R}^{T \times d_k},
\qquad h = 1, \ldots, H
$$

**注意這裡是「切」不是「加」。** 總維度仍然是 $d$，只是被 $H$ 個頭分著用（例如 $d=64$、$H=8$ 時每頭 8 維）。所以多頭**幾乎不增加參數量與計算量**，卻換到了多個獨立的關注視角——這是 Transformer 設計裡 CP 值最高的一手。

### 3.2 拼接與輸出投影

把 $H$ 個頭的輸出沿特徵維**拼接**回 $d$ 維，再乘一個輸出投影矩陣 $W_O \in \mathbb{R}^{d \times d}$：

$$
\text{MultiHead}(X) = \underbrace{\text{Concat}\!\left(C^{(1)}, \ldots, C^{(H)}\right)}_{\in\; \mathbb{R}^{T \times d}} \, W_O
$$

（因為 $H \cdot d_k = d$，拼接後剛好回到 $d$ 維。）

**Concat 是什麼運算？** 就是「並排放在一起」——把第 1 頭的 $d_k$ 維放在最前面、第 2 頭接在後面……沒有任何乘法或加法，純粹是搬家。所以維度自然回到 $H \cdot d_k = d$。

### 3.3 $W_O$ 在做什麼？

拼接只是把各頭的輸出**塞進彼此不重疊的座標區塊**——第 1 頭佔前 $d_k$ 維、第 2 頭佔接下來 $d_k$ 維……此時各頭的資訊是「分隔」的。$W_O$ 是一個（一般可逆的）**基底變換**，把這些分區的資訊重新混合、重組成一個共用的表示，讓後面的層可以跨頭整合。少了 $W_O$，各頭的輸出就永遠卡在自己的座標區塊裡無法互通（這個「投影＝可逆基底變換」的觀點見 [`03b3-transformer-architecture-example.md`](03b3-transformer-architecture-example.md) §0）。

**回到比喻：** Concat 只是把 8 個人的筆記本疊在一起交上來，8 份筆記各寫各的、彼此還沒對過話。$W_O$ 才是那個「把 8 份筆記整合成一份總結」的動作——讓第 3 頭發現的線索能和第 7 頭發現的線索產生關聯。少了它，多頭就只是 8 個各做各的單頭，白白浪費了並行的意義。

> 多頭的逐步數值範例見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §5–§5.6。

> **讀完這一節，你會：**
> - 說出多頭為什麼幾乎不增加參數量（切維度，不是加維度）
> - 寫出 $C^{(h)}$、Concat 與 $W_O$ 三步，並確認維度如何回到 $d$
> - 解釋少了 $W_O$ 會發生什麼事

---

## 4. Position-wise FFN 的數學

Attention 負責「跨位置」混合資訊；混完之後，還需要一層對**每個位置各自**做非線性變換的網路，這就是 Position-wise Feed-Forward Network（FFN）。「Position-wise」指同一個 MLP 套用到序列的每一個位置（每一列），彼此參數共用、互不干擾。

**比喻（接續 §3 的開會）：** Attention 是「大家開會互相交換意見」，FFN 就是「散會後每個人回自己座位，把剛剛聽到的東西消化成自己的筆記」。開會要跨人，整理筆記只跟自己有關——這就是為什麼 FFN 不看其他位置。

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

先擴到 $4d$ 給模型較大的中間空間做非線性特徵組合，再投影回 $d$ 維以便接續殘差相加（見 §5）。ReLU 提供非線性，否則兩層線性可合併成一層、失去表達力。

**這句「否則會合併成一層」值得展開。** 若拿掉 ReLU，整段就是 $z W_1 W_2 + (b_1 W_2 + b_2)$，而 $W_1 W_2$ 只是**另一個 $d\times d$ 矩陣**——兩層線性層疊起來仍然只是一層線性層，多花的參數完全白費。ReLU 把負數砍成 0，這個「折一下」的動作打破了線性，兩層才真的比一層強。

**為什麼是先胖再瘦？** 直觀上，$d$ 維空間裡分不開的東西，升到 $4d$ 維常常就分得開了（像把糾纏的毛線攤到桌面上比較好解）。在寬的地方做完非線性組合，再壓回 $d$ 維，是為了讓輸出能和輸入相加（殘差要求形狀一致）並接進下一層。**順帶一提：FFN 通常是整個模型參數量最大的部分**——兩個矩陣合計 $8d^2$ 個參數，比 Attention 的四個 $d\times d$ 矩陣（$4d^2$）還多一倍。

> FFN 的角色與數值範例見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §6.4。

> **讀完這一節，你會：**
> - 說出 FFN 與 Attention 的分工（跨位置混合 vs 逐位置加工）
> - 寫出 $d \to 4d \to 4d \to d$ 的形狀鏈
> - 用「兩層線性會塌成一層」解釋 ReLU 為什麼不能省

---

## 5. LayerNorm 與 Pre-LN Block 的數學

把 §1–§4 的模組疊成一個 Transformer Block，還需要兩個黏合劑：**LayerNorm**（穩定數值）與 **Residual（殘差）**（穩定梯度）。

**為什麼需要黏合劑？** 因為要疊很多層。一層 Attention ＋ 一層 FFN 就是一個 Block，GPT-2 疊 12～48 個。層一多，兩個問題就冒出來：數值一路乘下去可能爆掉或縮到接近 0（LayerNorm 管這個），梯度一路傳回去可能消失（Residual 管這個）。

### 5.1 LayerNorm

對單一 token 的向量 $z \in \mathbb{R}^{d}$，LayerNorm 沿它自己的 $d$ 個特徵做標準化（注意：是對**特徵維**，不是對 batch 或位置維）：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} z_i, \qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (z_i - \mu)^2,
$$

$$
\text{LN}(z) = \gamma \odot \frac{z - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

先把向量拉成平均 0、變異數 1，再用可訓練的 $\gamma, \beta \in \mathbb{R}^{d}$（逐元素縮放與平移）還給模型調整的自由；$\epsilon$ 防止除以 0。

**這就是高中統計的標準化 $z$ 分數。** 「減平均、除標準差」是同一件事：不管原本的數字是幾百還是零點零幾，換算完都落在同一個尺度上。

**比喻：** 每一層開始前，先把所有人的音量調到差不多大聲，後面的討論才不會被某個特別大聲的人蓋過去。

**那 $\gamma, \beta$ 是幹嘛的？** 硬把每一層都壓成「平均 0、變異數 1」其實限制太死了——有時模型就是需要某個維度特別突出。所以標準化完之後再乘 $\gamma$、加 $\beta$，讓模型自己決定要不要把尺度調回來。極端情況下模型可以學到 $\gamma = \sigma$、$\beta = \mu$ 把標準化整個抵銷掉，等於「有選擇權但不強迫」。

> **注意 LayerNorm 是「橫著算」。** 它對**每個 token 自己的 $d$ 個特徵**取平均和變異數，和其他 token、其他句子完全無關。這一點和 CNN 常用的 BatchNorm（沿 batch 方向算）不同，也是它在變長序列上好用的原因——不管一批有幾個句子、每句多長，算法都一樣。

### 5.2 Pre-LN Block

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

**逐字讀一次這個通式：** 先把 $x$ 標準化，丟進模組 $f$（Attention 或 FFN）算出「這一層想做的修改」，最後把這個修改**加回**原本的 $x$。關鍵是那個加號——模組的輸出不是拿來取代 $x$，而是拿來修正 $x$。

**比喻（改作文）：** 不是把原稿丟掉重寫，而是保留原稿、在旁邊註記修改。這樣原本的重要資訊不容易在層層傳遞中消失，模型也可以一層一層慢慢改進。

所以一個完整的 Block 是「兩次 $x \leftarrow x + f(\text{LN}(x))$」——第一次的 $f$ 是 Multi-Head Attention（§3），第二次的 $f$ 是 FFN（§4）。把這個 Block 疊 $N$ 次，就是 GPT 的主體。

### 5.3 為什麼 Pre-LN 的梯度比較穩定

看反向傳播時梯度怎麼流。對 Pre-LN 的 $y = x + f(\text{LN}(x))$，對 $x$ 微分：

$$
\frac{\partial y}{\partial x} = I + \frac{\partial f(\text{LN}(x))}{\partial x}
$$

（加法的微分就是各項分別微分：$x$ 對自己微分得到單位矩陣 $I$，後面那項照鏈式法則展開。）

那個**單位矩陣 $I$** 就是關鍵：它讓上游梯度 $\dfrac{\partial L}{\partial y}$ 有一條**完全不經過 LayerNorm、也不衰減**的路徑直接傳回 $x$——

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\left(I + \frac{\partial f(\text{LN}(x))}{\partial x}\right)
= \underbrace{\frac{\partial L}{\partial y}}_{\text{恆等直通}} + \frac{\partial L}{\partial y}\frac{\partial f(\text{LN}(x))}{\partial x}
$$

疊 $L$ 層時，這條恆等路徑保證最深處仍能收到一份未被反覆縮放的梯度。

**用數字感受一下差別。** 假設每穿過一層模組梯度就被乘上 0.9。走 24 層：沒有殘差時剩下 $0.9^{24} \approx 0.08$，只剩 8%；有殘差時每一層都額外有一份「乘 1」的原封不動的梯度，最深層照樣收得到有效訊號。**殘差就是給梯度開的一條高速公路。**

反觀 Post-LN 的 $\text{LN}(x + f(x))$，梯度每穿一層都要乘上一次 LayerNorm 的 Jacobian，$L$ 一大尺度就容易爆炸或消失，所以才需要 learning-rate warm-up「小心起步」。

| | Post-LN | Pre-LN |
|---|---|---|
| 訓練穩定性 | 需要 warm-up | 更穩定，學習率更寬容 |
| 深層表現 | 容易梯度爆炸/消失 | 梯度流更均勻 |
| 代表模型 | 原始 Transformer | GPT-2、LLaMA、nanoGPT |

> LayerNorm 的完整 Jacobian 與 Residual 的梯度恆等推導見 [`05b1-backward-propagation.md`](05b1-backward-propagation.md) §5.9／§5.10。Pre-LN 在 nanoGPT 程式裡長什麼樣，見 [`04b`](04b-nanogpt-walkthrough.md) §4、§7。

> **讀完這一節，你會：**
> - 說出 LayerNorm 是沿哪個方向做標準化，以及 $\gamma, \beta$ 存在的理由
> - 寫出 Pre-LN Block 的通式 $x \leftarrow x + f(\text{LN}(x))$，並說出一個 Block 裡的兩個 $f$ 分別是什麼
> - 用「$I + \partial f/\partial x$」解釋殘差為什麼是梯度的高速公路
> - 說出 Pre-LN 與 Post-LN 的差別，以及後者為何需要 warm-up

---

## 6. Token Embedding 與位置編碼的數學

前面幾節的輸入 $X$ 是「一堆 $d$ 維向量」，但原始輸入其實是**離散的 token id**。這一節補上「id → 向量」這一步。

**位置在整條 Pipeline 的哪裡？** 這其實是**最前面**那一步：文字先被切成 token 並轉成整數 id，接著才變成 §1 用的那個矩陣 $X$。本文把它排在 §6，是因為要先看懂中間的模組在算什麼，才知道輸入該長什麼樣。

### 6.1 Token Embedding：查表就是 one-hot 乘法

設詞彙表大小為 $V$，embedding 矩陣 $E \in \mathbb{R}^{V \times d}$（每一列是一個 token 的向量）。token id $t_i$ 對應的向量就是取 $E$ 的第 $t_i$ 列：

$$
x_i = E[t_i] = e_{t_i}^\top E
$$

其中 $e_{t_i} \in \mathbb{R}^{V}$ 是第 $t_i$ 個位置為 1 的 one-hot 向量。所以「查表（Lookup）」在數學上等於一次 one-hot 乘矩陣，實作上則是 $O(1)$ 的索引，不必真的做乘法。（形式化亦見 [`01b-prerequisites-math.md`](01b-prerequisites-math.md) §2；它如何被訓練見 [`05b1-backward-propagation.md`](05b1-backward-propagation.md) §10。）

**把 $E$ 想成一本字典。** 它有 $V$ 列，每一列是某個 token 的「意義向量」。查字典（給 id、拿向量）就是取出第 $t_i$ 列，如此而已。

**那為什麼還要寫成 one-hot 乘法？** 因為這個寫法把「查表」變成了一個標準的矩陣乘法，於是它就能套用一般線性層的梯度規則，反向傳播時不必為它另立規矩（見 [`05b1`](05b1-backward-propagation.md) §10.2）。**實作照樣用索引（快），數學上用乘法（好推導）**，兩者等價。

重點是：$E$ 是**可訓練參數**。一開始每一列都是隨機數字，完全沒有意義；靠著幾十萬步訓練，語意相近的 token 才慢慢被推到相近的位置（這個「向量會學到語意」的性質，是 RAG 與語意檢索的基礎，見 [`09-text-to-vector-rag.md`](09-text-to-vector-rag.md)）。

### 6.2 位置編碼：把順序加回去

Attention 本身對「順序」無感——打亂 token 的位置，$QK^\top$ 的集合不變。

**為什麼會這樣？** 回頭看 §1.4 的 $c_i = \sum_j A_{ij} v_j$：它是一個**加總**，而加總不在乎順序。「我吃魚」和「魚吃我」餵給純 Attention，每個 token 算出來的東西一模一樣。這顯然不行，所以要**額外注入位置資訊**。

nanoGPT 用 **Learned PE**：另有一個位置矩陣 $P \in \mathbb{R}^{T_{\max} \times d}$，位置 $i$ 取第 $i$ 列 $P[i]$，直接與 token 向量**相加**：

$$
h_{0,i} = \underbrace{E[t_i]}_{\text{token 語意}} + \underbrace{P[i]}_{\text{位置}}
$$

**$P$ 也是可訓練的**，和 $E$ 一樣從隨機開始學。差別在於 $E$ 按「是哪個字」查表，$P$ 按「排在第幾個」查表。

相加（而非拼接）能保持維度仍是 $d$，模型有能力在後續層把兩者解開。

**「相加會不會把兩種資訊混在一起分不開？」** 直覺上會擔心，但在高維空間裡（$d$ 通常幾百到幾千），語意與位置可以被學到接近正交的方向上，後面的線性層有能力把它們拆開。而拼接的代價是維度變成 $2d$，後面每一層的計算量都跟著上升——相加是划算得多的選擇。

另一種做法是固定公式的 **Sinusoidal PE**（不需訓練、理論上可外推到更長序列），推導見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §7.2；Learned PE 的細節與兩者比較見 03a §7.5 與 [`04b`](04b-nanogpt-walkthrough.md) §5。

> **讀完這一節，你會：**
> - 說出 $E$ 的形狀與意義，並解釋為何查表等價於 one-hot 乘法
> - 用「加總不在乎順序」說明 Attention 為什麼需要位置編碼
> - 說出位置編碼為什麼用相加而不是拼接

---

## 7. Next-token Prediction 與 Cross-Entropy

架構齊備了，最後定義**訓練目標**：讓模型對每個位置預測「下一個 token」。

**這是整個 GPT 最樸素、也最關鍵的一件事：** 沒有人工標註、沒有題庫，訓練目標只有一個——「猜下一個字」。任何一段現成的文字都自帶答案（下一個字就寫在那裡），所以網路上的文本可以直接拿來訓練。

### 7.1 資料準備

給定文字序列，把它切成 `(input, target)` 對：

```
文字： h  e  l  l  o
input:  [h, e, l, l]   (前 T 個 token)
target: [e, l, l, o]   (後 T 個 token，即 input 右移一位)
```

位置 $i$ 的 **input** 是 $x_i$，**target** 是 $x_{i+1}$。

**就是把同一段文字錯開一格。** target 不需要另外準備，把 input 整個往左挪一位就是了。所以一段 $T+1$ 個字的文字，可以同時生出 $T$ 組「題目 → 答案」：看到 `h` 該接 `e`、看到 `he` 該接 `l`、看到 `hel` 該接 `l`……

### 7.2 損失函數

模型在位置 $i$ 輸出一個 logit 向量 $z_i \in \mathbb{R}^{V}$，經 softmax 成機率分佈 $p^{(i)} = \mathrm{softmax}(z_i)$，再與正確答案 $y_i = x_{i+1}$ 用 **Cross-Entropy** 比較。整個序列的損失是各位置的平均：

$$
p^{(i)}_k = \frac{\exp(z_{i,k})}{\sum_{k'} \exp(z_{i,k'})}, \qquad
L = -\frac{1}{T}\sum_{i=1}^{T} \log p^{(i)}_{y_i}
$$

**這條 loss 在做什麼（逐段拆開）：**

| 片段 | 意思 |
|---|---|
| $p^{(i)}_{y_i}$ | 模型分給**正確答案**的機率（0 到 1 之間）|
| $\log p^{(i)}_{y_i}$ | 取 log。猜得準（機率接近 1）時接近 0，猜得爛（機率接近 0）時是很大的負數 |
| 前面的負號 | 把它翻正，變成「猜得愈爛、值愈大」——這才適合當「損失」 |
| $\frac1T\sum_i$ | 把序列裡 $T$ 個位置的損失平均起來 |

**注意它只看正確答案那一格。** 其他 $V-1$ 個 token 分到多少機率，loss 完全不管——但因為 softmax 的總和固定是 1，把正確答案的機率推高，其他自然就被壓低了。

**一個好用的直覺：** 如果模型完全瞎猜（$V$ 個選項機率均等），$p = 1/V$，loss $= \log V$。所以看到 loss 大約等於 $\log V$，就知道模型還沒學到東西；訓練有效的話 loss 會明顯降到這個值以下。

程式上就是一行：

```python
# logits: (B, T, vocab_size)
# targets: (B, T)
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
```

（`view(B*T, ...)` 是把 batch 和位置兩個維度攤平，讓 $B\times T$ 個預測一次算完；PyTorch 的 `cross_entropy` 內部已經包含 softmax，所以傳進去的是 logits 而不是機率。）

有了 §2 的 Causal Mask，位置 $i$ 的 logit 只用到了 $x_1, \ldots, x_i$ 的資訊，預測 $x_{i+1}$，**不會洩漏未來**。因此一個序列可以同時訓練 $T$ 個預測任務，訓練效率極高。

**這句話值得停下來想一下。** 一次 forward 餵進 $T$ 個 token，模型同時產出 $T$ 個預測、算出 $T$ 份 loss、更新一次全部參數。如果沒有因果遮罩，就只能一次訓練一個位置（後面的都得藏起來），效率差 $T$ 倍。**遮罩不只是為了防作弊，也是 Transformer 訓練得起來的關鍵。**

> **讀完這一節，你會：**
> - 說出 input／target 的切法，並解釋為什麼不需要人工標註
> - 逐段解釋 $L = -\frac1T\sum_i \log p^{(i)}_{y_i}$ 每個部分的作用
> - 用 $\log V$ 判斷模型有沒有真的開始學
> - 說明因果遮罩如何讓一次 forward 同時訓練 $T$ 個預測任務

---

## 全文回顧：七節怎麼串成一次 forward

把七節按**資料真正流動的順序**重排一次，就是一次完整的前向傳播：

| 順序 | 這一步 | 輸入 → 輸出 | 本文節次 |
|---|---|---|---|
| 1 | token id 查表 ＋ 加位置 | $(T,) \to (T, d)$ | §6 |
| 2 | LayerNorm① | $(T, d) \to (T, d)$ | §5.1 |
| 3 | 投影出 Q/K/V、算分數、縮放 | $(T, d) \to (T, T)$ | §1.1–§1.3 |
| 4 | 套因果遮罩 | $(T, T) \to (T, T)$ | §2 |
| 5 | softmax、加權平均 Value | $(T, T) \to (T, d_v)$ | §1.4 |
| 6 | 多頭拼接 ＋ $W_O$ | $(T, d_v)\times H \to (T, d)$ | §3 |
| 7 | 殘差① 相加 | $(T, d) \to (T, d)$ | §5.2 |
| 8 | LayerNorm② → FFN → 殘差② | $(T, d) \to (T, d)$ | §5.1、§4、§5.2 |
| 9 | 重複 2–8 共 $N$ 個 Block | $(T, d) \to (T, d)$ | §5.2 |
| 10 | 最終 LayerNorm ＋ lm_head | $(T, d) \to (T, V)$ | §5.1、§7.2 |
| 11 | softmax ＋ Cross-Entropy | $(T, V) \to$ 一個數字 | §7.2 |

反向傳播就是把這張表**由下往上倒著走一遍**——[`05b1`](05b1-backward-propagation.md) §2.2 有一張對照的鏡射表。

---

## 下一步

**先看數值：把本文每個階段實算一次** → [`05a2-forward-example.md`](05a2-forward-example.md)

用一組範例資料（T=2、d=3、單頭、含因果遮罩）從 embedding 一路算到 CE loss，每個階段都給出實際數字，與本文各節的符號公式逐一對照。

**反向傳播數學：** → [`05b1-backward-propagation.md`](05b1-backward-propagation.md)

把 loss 沿同一條路徑反向傳回 Embedding：softmax+CE 的合併梯度、穿越 LayerNorm／Residual／FFN／Attention 的梯度。其逐階段數值計算（沿用 05a2 的數字）見 [`05b2-backward-example.md`](05b2-backward-example.md)。

**對照程式實作：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md) → [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)
