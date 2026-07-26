# 10a1｜Seq2Seq 向前傳播：兩代架構的數學（符號推導）

> **這是 encoder-decoder 家族的選讀分支。** 主線 [`04a`](04a-gpt-decoder-only.md)–[`05b2`](05b2-backward-example.md) 走的是 decoder-only 的 GPT，[`07`](07-bert-encoder-only.md) 是 encoder-only 的 BERT，本文補上第三條路：**輸入一個序列、輸出另一個序列**的 Encoder-Decoder。
>
> **本文做什麼：** 把同一個翻譯任務用**兩代架構**各推導一次前向傳播——2014 年的 **RNN Seq2Seq + Bahdanau Attention**，以及 2017 年的 **Transformer Encoder-Decoder**。兩代放在一起看，才會知道 attention 當初是為了解決什麼問題被發明出來、Transformer 又額外解決了什麼。
>
> **前置文件：** [`01a`](01a-prerequisites-intuition.md)／[`01b`](01b-prerequisites-math.md)（Softmax、加權平均）、[`02`](02-attention-intuition.md)（QKV 直覺）、[`03a`](03a-transformer-architecture.md)（Transformer Block）。**讀完 `03a` 就能讀本文**，不必先走完 `04a`／`05`；若已讀過 `05a1`，§B 會輕鬆很多，本文也會標出哪些節可直接沿用。
>
> **與 [`advanced/Seq2Seq-and-Decoding-Techniques.md`](../advanced/Seq2Seq-and-Decoding-Techniques.md) 的分工：** 那篇講**故事**——Seq2Seq 能解什麼任務、teacher forcing／beam search／copy mechanism 等訓練與解碼工藝，全篇零公式。本文講**數學**——每個模組的前向公式與形狀鏈。兩篇互補，先讀哪篇都可以。
>
> **本組四份文件：**
>
> | 文件 | 內容 |
> |---|---|
> | **10a1（本文）** | 前向・符號推導 |
> | [`10a2`](10a2-seq2seq-forward-example.md) | 前向・數值範例（每個公式代進真實數字）|
> | [`10b1`](10b1-seq2seq-backward.md) | 後向・符號推導（BPTT ＋ Cross-Attention 反向）|
> | [`10b2`](10b2-seq2seq-backward-example.md) | 後向・數值範例（沿用 10a2 的數字）|

---

## 目錄

- [0. Seq2Seq 要解的問題](#0-seq2seq-要解的問題)
- [1. 共用任務設定與兩代鳥瞰](#1-共用任務設定與兩代鳥瞰)
- [A. RNN Seq2Seq + Bahdanau Attention（2014）](#a-rnn-seq2seq--bahdanau-attention2014)
  - [A1. Encoder RNN](#a1-encoder-rnn)
  - [A2. 固定 context vector 的瓶頸](#a2-固定-context-vector-的瓶頸)
  - [A3. Bahdanau 加性 Attention](#a3-bahdanau-加性-attention)
  - [A4. Decoder RNN](#a4-decoder-rnn)
  - [A5. 輸出層與 Cross-Entropy](#a5-輸出層與-cross-entropy)
  - [A6. 參數清單](#a6-參數清單)
- [B. Transformer Encoder-Decoder（2017）](#b-transformer-encoder-decoder2017)
  - [B1. Encoder：雙向、無遮罩](#b1-encoder雙向無遮罩)
  - [B2. Decoder 的 Masked Self-Attention](#b2-decoder-的-masked-self-attention)
  - [B3. Cross-Attention](#b3-cross-attention)
  - [B4. 三種 Attention 一表對照](#b4-三種-attention-一表對照)
  - [B5. 完整 Pipeline](#b5-完整-pipeline)
  - [B6. 參數清單](#b6-參數清單)
- [C. 兩代對照](#c-兩代對照)
- [D. 四份文件的節號鏡像](#d-四份文件的節號鏡像)

---

## 0. Seq2Seq 要解的問題

一般的分類問題輸入一筆資料、輸出一個類別。**Sequence-to-Sequence** 處理的是另一種形狀：

$$
\text{輸入序列 } x=(x_1,\dots,x_{T_s}) \;\longrightarrow\; \text{輸出序列 } y=(y_1,\dots,y_{T_t})
$$

關鍵在於 $T_t$ **不由 $T_s$ 決定**，而且事先未知。「機器學習」四個字翻成 `Machine Learning` 兩個詞，但不是所有中文句子翻成英文都會變一半長。模型必須自己決定何時停止——實作上是在目標詞彙表裡放一個 `<eos>` token，模型生出它就結束。

這件事逼出了一個**兩段式**的架構：

```
        ┌──────────┐            ┌──────────┐
 來源 → │ Encoder  │ → 表示 →   │ Decoder  │ → 目標
        └──────────┘            └──────────┘
        「讀懂輸入」            「逐步寫出輸出」
```

Encoder 負責把長度 $T_s$ 的輸入讀成某種內部表示，Decoder 負責從那個表示逐步生出長度 $T_t$ 的輸出。**兩代架構的差別，全在於這兩個框框裡面裝什麼、以及中間那條「表示」是什麼形狀。** 這正是本文要推導的東西。

> **為什麼 GPT 不需要 Encoder？** 因為 GPT 把「翻譯」改寫成「接話」：輸入寫成「請翻譯成英文：今天天氣真好 →」，模型往下生成即可，來源與目標串成同一個序列，一個 decoder 就夠了。這條路線的取捨見 [`04a`](04a-gpt-decoder-only.md) §2。本文要看的是**沒有這樣改寫**時，標準的兩段式怎麼運作。

---

## 1. 共用任務設定與兩代鳥瞰

四份文件從頭到尾用同一個任務，兩代架構才能逐項對照：

```
來源：我 吃        →        目標：I eat
```

| 項目 | 取值 |
|---|---|
| 來源詞彙表 $V_s$ | `我`=0、`吃`=1、`魚`=2（$\lvert V_s\rvert=3$）|
| 目標詞彙表 $V_t$ | `<bos>`=0、`I`=1、`eat`=2（$\lvert V_t\rvert=3$）|
| 來源序列 | $x=(0,1)$，長度 $T_s=2$ |
| Decoder 輸入 | $(0,1)$＝`<bos> I`，長度 $T_t=2$ |
| 目標 | $y=(1,2)$＝`I eat` |

**Decoder 輸入與目標的關係，就是「右移一位」。** 位置 0 餵 `<bos>` 要它答 `I`；位置 1 餵 `I` 要它答 `eat`。這與 GPT 的 next-token 切法（[`05a1`](05a1-forward-propagation.md) §7.1）是同一件事，差別只在 decoder 這一側多了 encoder 提供的來源資訊。

**這個排法有個正式名稱叫 Teacher Forcing。** 位置 1 餵進去的是**正確答案** `I`，而不是模型在位置 0 實際預測出來的 token。訓練時這樣做，好處是所有位置可以**同時**算（不必等前一步的輸出），代價是訓練與推論不一致——推論時模型只能吃自己的輸出，一旦第一步錯了，後面看到的上下文就與訓練時不同。這個落差叫 **Exposure Bias**，直覺討論見 [`advanced/Seq2Seq`](../advanced/Seq2Seq-and-Decoding-Techniques.md) 幕三。本文四份文件全部在 teacher forcing 的設定下推導。

**兩代架構鳥瞰：**

| | RNN Seq2Seq（2014）| Transformer Encoder-Decoder（2017）|
|---|---|---|
| Encoder | RNN 逐步捲動，狀態 $h_1,\dots,h_{T_s}$ | Self-Attention Block，輸出 $H\in\mathbb{R}^{T_s\times d}$ |
| 傳給 Decoder 的東西 | 每步重算的 context vector $c_t$ | 整個 $H$，由 Cross-Attention 讀取 |
| Decoder | RNN 逐步捲動，狀態 $s_1,\dots,s_{T_t}$ | Masked Self-Attn → Cross-Attn → FFN |
| Attention 打分方式 | **加性**（先相加、過 $\tanh$、再投影成純量）| **點積**（直接內積、除 $\sqrt{d_k}$）|
| 位置資訊 | 由遞迴順序天然帶入 | 靠位置編碼補上 |
| 能否平行 | 否（$h_t$ 要等 $h_{t-1}$）| 是（矩陣一次算完）|

---

## A. RNN Seq2Seq + Bahdanau Attention（2014）

本節推導的模型用 **vanilla RNN（$\tanh$）** 當遞迴單元。

> **為什麼不用 LSTM？** Sutskever 等人 2014 年的原始論文用的確實是 LSTM。但 LSTM 一個時間步有四個閘（輸入、遺忘、輸出、候選記憶），手算與梯度推導會被閘的代數淹沒，而本文真正要講的是**遞迴結構本身**帶來的性質——狀態逐步傳遞、參數跨時間共享、梯度連乘。這些性質在 vanilla RNN 裡最乾淨。[`05b1`](05b1-backward-propagation.md) 附錄 B 寫的那條連乘積 $\prod_t \partial h_t/\partial h_{t-1}$ 正是 vanilla 形式。LSTM 的閘控之所以能緩解連乘衰減，是因為它讓記憶單元有一條**加法**的傳遞路徑（$c_t = f_t\odot c_{t-1}+i_t\odot\tilde c_t$），$\partial c_t/\partial c_{t-1}$ 裡有一項是遺忘閘 $f_t$ 而不是權重矩陣的乘積——這在 [`10b1`](10b1-seq2seq-backward.md) §A6 會再回來談。

### A1. Encoder RNN

Encoder 把來源序列一個一個讀進去，每讀一個就更新一次內部狀態。設來源 token $x_i$ 的詞向量為 $e^s_i = E_s[x_i]\in\mathbb{R}^{d_e}$（$E_s\in\mathbb{R}^{|V_s|\times d_e}$ 是查表矩陣，機制同 [`05a1`](05a1-forward-propagation.md) §6.1），則

$$
\boxed{\;h_i = \tanh\!\big(e^s_i W_{xh} + h_{i-1} W_{hh} + b_h\big)\;},\qquad h_0=\mathbf0,\quad i=1,\dots,T_s
$$

形狀：$e^s_i\in\mathbb{R}^{1\times d_e}$、$W_{xh}\in\mathbb{R}^{d_e\times d_h}$、$h_{i-1}\in\mathbb{R}^{1\times d_h}$、$W_{hh}\in\mathbb{R}^{d_h\times d_h}$、$b_h\in\mathbb{R}^{1\times d_h}$，相加後每一項都是 $1\times d_h$，$\tanh$ 逐元素作用，故 $h_i\in\mathbb{R}^{1\times d_h}$。

把括號裡的線性部分記成 $a_i$（後面反向會反覆用到這個名字）：

$$
a_i = e^s_i W_{xh} + h_{i-1} W_{hh} + b_h,\qquad h_i=\tanh(a_i)
$$

有三件事現在就要看清楚，它們決定了後面所有的推導：

1. **順序是強制的。** 算 $h_2$ 必須先有 $h_1$。$T_s$ 個時間步只能一步一步跑，**無法平行**。這是 RNN 與 Transformer 最根本的差異。
2. **參數跨時間共享。** 每一步用的都是**同一組** $W_{xh},W_{hh},b_h$，不是每步一份。所以反向時這些參數的梯度會是所有時間步的**累加**（[`10b1`](10b1-seq2seq-backward.md) §A5）。
3. **$\tanh$ 把狀態壓在 $(-1,1)$。** 這讓數值穩定，但也讓導數 $\tanh'(a)=1-\tanh^2(a)\le 1$ 恆成立——梯度每穿過一個時間步就至少乘一個不大於 1 的數。這是梯度消失的來源（§A2 與 [`10b1`](10b1-seq2seq-backward.md) §A6）。

跑完 $T_s$ 步得到一排狀態，堆成矩陣：

$$
H = \begin{bmatrix} h_1 \\ \vdots \\ h_{T_s} \end{bmatrix}\in\mathbb{R}^{T_s\times d_h}
$$

### A2. 固定 context vector 的瓶頸

2014 年最初的 Seq2Seq 做法是：**只把最後一個狀態 $h_{T_s}$ 交給 decoder**，當成整個來源句子的摘要，decoder 全程只看這一個向量。

這個設計有一個結構性的問題。$h_{T_s}\in\mathbb{R}^{d_h}$ 的維度是**固定**的，但來源句子可以任意長。要把 50 個詞的句子壓進同一個 $d_h$ 維向量，資訊必然被壓縮到失真；而且 decoder 在生**每一個**目標詞時看到的來源資訊完全相同——翻譯第 1 個詞和第 20 個詞，該注意的來源位置顯然不一樣，但模型沒有辦法區分。

還有一個梯度層面的問題。來源第 1 個詞要影響目標第 1 個詞，梯度必須從 $s_1$ 一路穿回 $h_{T_s}, h_{T_s-1},\dots,h_1$，路徑長度是 $O(T_s)$，而每穿一步就乘一次 $\tanh'\le1$ 與一次 $W_{hh}$。序列一長，前段的梯度就衰減到近乎為零——**來源句子的開頭學不到東西**。

**Bahdanau 等人 2014 年的解法：不要只給一個向量，把整排 $h_1,\dots,h_{T_s}$ 都留著，讓 decoder 在每一步自己決定要看哪幾個。** 這就是 attention 的誕生。下一節推導它。

> **請注意這個歷史順序：** attention 是為了修 RNN 的瓶頸而發明的，比 Transformer 早三年。Transformer 的貢獻不是發明 attention，而是**把遞迴整個拿掉、只留 attention**（時間軸見 [`00`](00-learning-path.md) 的 ML 歷史表）。

### A3. Bahdanau 加性 Attention

Decoder 在生第 $t$ 個目標詞之前，先用自己**前一步的狀態** $s_{t-1}$ 去問：來源的每個位置對我現在有多重要？

**第一步：對每個來源位置打一個分數。** Bahdanau 用的是**加性（additive）** 打分：

$$
\boxed{\;u_{t,i} = \tanh\!\big(s_{t-1} W_a + h_i U_a\big),\qquad
e_{t,i} = u_{t,i}\,v_a^\top\;}
$$

形狀：$W_a\in\mathbb{R}^{d_h\times d_a}$、$U_a\in\mathbb{R}^{d_h\times d_a}$，兩項相加得 $u_{t,i}\in\mathbb{R}^{1\times d_a}$；再與 $v_a\in\mathbb{R}^{1\times d_a}$ 做內積壓成**純量** $e_{t,i}\in\mathbb{R}$。

拆開看它在做什麼：$s_{t-1}W_a$ 把 decoder 狀態投影到一個 $d_a$ 維的「比對空間」，$h_iU_a$ 把 encoder 狀態投影到**同一個**空間，兩者**相加**後過 $\tanh$ 得到一個混合表示，最後用可訓練向量 $v_a$ 把它壓成一個分數。「相加再過非線性再投影」——這就是「加性」這個名字的由來。

**第二步：softmax 正規化成權重。**

$$
\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j=1}^{T_s}\exp(e_{t,j})},
\qquad \sum_{i=1}^{T_s}\alpha_{t,i}=1
$$

**第三步：加權平均出 context vector。**

$$
\boxed{\;c_t = \sum_{i=1}^{T_s}\alpha_{t,i}\,h_i \;=\; \alpha_{t,\cdot}\,H\;}\in\mathbb{R}^{1\times d_h}
$$

寫成矩陣形式 $c_t=\alpha_{t,\cdot}H$（$1\times T_s$ 乘 $T_s\times d_h$）——這就是加權平均，機制與 [`01a`](01a-prerequisites-intuition.md) §5 講的完全一樣。

**這一步解決了 §A2 的兩個問題。** 其一，$c_t$ **每一步都重算**，$t=1$ 與 $t=2$ 可以看向不同的來源位置。其二，$c_t$ 是 $h_i$ 的直接加權和，所以梯度可以從 $c_t$ **一步**回到任何一個 $h_i$，不必穿過時間鏈——路徑長度從 $O(T_s)$ 變成 $O(1)$。[`10b2`](10b2-seq2seq-backward-example.md) §D 會用實際數字顯示：這條捷徑帶回來的梯度**比時間鏈那條還大**。

**與點積 attention 的對照（[`05a1`](05a1-forward-propagation.md) §1.2）：**

| | Bahdanau 加性 | Transformer 點積 |
|---|---|---|
| 打分 | $v_a^\top\tanh(s W_a + h U_a)$ | $\dfrac{q\cdot k}{\sqrt{d_k}}$ |
| 參數 | $W_a,U_a,v_a$ 三個 | 打分本身**零參數**（參數在 $W_Q,W_K$ 裡）|
| 非線性 | 有（$\tanh$）| 無 |
| 能否寫成一次矩陣乘法 | **否**（每個 $(t,i)$ 配對都要算一次 $\tanh$）| **是**（$QK^\top$ 一次算完）|

最後一列是重點：加性 attention 對每一組 $(t,i)$ 都要獨立算一次 $\tanh$，$T_t\times T_s$ 組就是 $T_t T_s$ 次非線性運算，沒辦法折疊成單一矩陣乘法；點積 attention 則整張分數表 $QK^\top$ 一次 GEMM 就算完。**這是 Transformer 選點積而非加性的主要理由**，也是 §C 平行度對照的伏筆。

### A4. Decoder RNN

Decoder 也是一個 RNN，但每一步的輸入多了 context vector。設 decoder 第 $t$ 步餵進去的 token 是 $\tilde y_t$（teacher forcing，見 §1），其詞向量 $e^t_t = E_t[\tilde y_t]$，則

$$
\boxed{\;s_t = \tanh\!\big(e^t_t W_{ys} + s_{t-1} W_{ss} + c_t W_{cs} + b_s\big)\;},\qquad s_0 = h_{T_s}
$$

括號裡三項的角色各不相同：

| 項 | 帶進來的資訊 |
|---|---|
| $e^t_t W_{ys}$ | 上一個**已知的目標詞**（teacher forcing 餵的正確答案）|
| $s_{t-1} W_{ss}$ | decoder 自己的**歷史**（已經生成到哪裡了）|
| $c_t W_{cs}$ | 這一步該注意的**來源資訊**（§A3 算出來的）|

同樣把線性部分命名下來，反向會用到：

$$
g_t = e^t_t W_{ys} + s_{t-1} W_{ss} + c_t W_{cs} + b_s,\qquad s_t = \tanh(g_t)
$$

**初始狀態 $s_0=h_{T_s}$。** Decoder 的起點直接接上 encoder 的最後一個狀態。注意這條路徑**仍然存在**——加了 attention 不代表把它拿掉。所以 $h_{T_s}$ 有**兩條**出路：一條經 $s_0$ 進入時間鏈，一條經 attention 進入每一步的 $c_t$。[`10b1`](10b1-seq2seq-backward.md) §A3 反向時要把這兩股加起來。

**注意 §A3 用的是 $s_{t-1}$，不是 $s_t$。** 這不是筆誤：算 $s_t$ 需要 $c_t$，算 $c_t$ 需要 attention 分數，而分數若用 $s_t$ 就會循環定義。所以 Bahdanau 的順序固定是

$$
s_{t-1}\;\longrightarrow\;\alpha_{t,\cdot}\;\longrightarrow\;c_t\;\longrightarrow\;s_t
$$

### A5. 輸出層與 Cross-Entropy

每一步的狀態投影到目標詞彙表大小，過 softmax 得機率：

$$
z_t = s_t W_{out} + b_{out}\in\mathbb{R}^{1\times|V_t|},\qquad
p_t = \text{softmax}(z_t)
$$

損失取各步 cross-entropy 的平均（$y_t$ 是第 $t$ 步的正確 token id）：

$$
\boxed{\;L = \frac{1}{T_t}\sum_{t=1}^{T_t}\text{CE}\big(p_t,\,y_t\big)
= -\frac{1}{T_t}\sum_{t=1}^{T_t}\log p_t[y_t]\;}
$$

**每一步都是一個 $|V_t|$ 類的分類問題**，整句的 loss 是這些分類問題的平均。這與 GPT 的 next-token 訓練（[`05a1`](05a1-forward-propagation.md) §7.2）在形式上完全相同——差別只在 GPT 的每一步條件是「同一個序列的左側」，這裡的條件是「decoder 左側 ＋ 整個來源句」。

> **實務上 $L$ 常寫成總和而非平均。** 除以 $T_t$ 只是把不同長度的句子放到可比的尺度上；它會讓每一項梯度都乘上 $1/T_t$，不改變任何梯度的方向。本文取平均，與 [`05a1`](05a1-forward-propagation.md) §7.2 及 PyTorch `F.cross_entropy` 的預設一致。

### A6. 參數清單

這個模型要訓練的就是下面這 **14 個張量**：

| # | 參數 | 形狀 | 出現在 |
|---|---|---|---|
| 1 | $E_s$ 來源詞嵌入 | $\lvert V_s\rvert\times d_e$ | §A1 |
| 2 | $W_{xh}$ | $d_e\times d_h$ | §A1 |
| 3 | $W_{hh}$ | $d_h\times d_h$ | §A1 |
| 4 | $b_h$ | $d_h$ | §A1 |
| 5 | $E_t$ 目標詞嵌入 | $\lvert V_t\rvert\times d_e$ | §A4 |
| 6 | $W_a$（decoder 側投影）| $d_h\times d_a$ | §A3 |
| 7 | $U_a$（encoder 側投影）| $d_h\times d_a$ | §A3 |
| 8 | $v_a$（打分向量）| $d_a$ | §A3 |
| 9 | $W_{ys}$ | $d_e\times d_h$ | §A4 |
| 10 | $W_{ss}$ | $d_h\times d_h$ | §A4 |
| 11 | $W_{cs}$ | $d_h\times d_h$ | §A4 |
| 12 | $b_s$ | $d_h$ | §A4 |
| 13 | $W_{out}$ | $d_h\times\lvert V_t\rvert$ | §A5 |
| 14 | $b_{out}$ | $\lvert V_t\rvert$ | §A5 |

**參數量與序列長度無關。** $T_s$ 或 $T_t$ 變成 100，張量個數與大小完全不變——因為每個時間步用的是同一組權重。這是 RNN 的優點（模型小、能吃任意長度），也是它的束縛（同一組權重要應付所有位置）。

[`10a2`](10a2-seq2seq-forward-example.md) §0.2 會給這 14 個張量的具體取值，[`10b2`](10b2-seq2seq-backward-example.md) §C 會把它們的梯度全部算出來。

---

## B. Transformer Encoder-Decoder（2017）

同一個任務，換成 2017 年「Attention Is All You Need」的架構。本節的 Encoder Block 與 Decoder Block 大量沿用主線已推導過的模組，會明確標出「這一段等同於某文件某節」，只在**新東西**上展開。

三個新東西：Encoder 的**無遮罩** self-attention、Decoder 的**三個**子層（比 GPT 多一層）、以及把兩邊接起來的 **Cross-Attention**。

本文的模型規模：Encoder 一層 Block、Decoder 一層 Block、單頭、**Pre-LN**（[`05a1`](05a1-forward-propagation.md) §5.2）。

> **原始論文用的是 Post-LN、6 層、8 個頭。** 本文用 Pre-LN 是為了與主線 [`05a1`](05a1-forward-propagation.md)／[`04b`](04b-nanogpt-walkthrough.md) 一致（也因為 Pre-LN 才是現代實作的預設）；用單頭一層是為了讓 [`10a2`](10a2-seq2seq-forward-example.md) 能手算。多頭不改變任何一條公式，只是把同一套機制平行跑 $H$ 次再拼接，見 [`05a1`](05a1-forward-propagation.md) §3 與 [`05a2`](05a2-forward-example.md) §3.1。

### B1. Encoder：雙向、無遮罩

Encoder 輸入是來源序列。先查表加位置編碼：

$$
X^{\text{enc}} = E_s[x] + P_s \in\mathbb{R}^{T_s\times d}
$$

然後過一個標準的 Pre-LN Block：

$$
\begin{aligned}
X_1 &= X^{\text{enc}} + \text{SelfAttn}\big(\text{LN}_{e1}(X^{\text{enc}})\big)\\
X_2 &= X_1 + \text{FFN}\big(\text{LN}_{e2}(X_1)\big)\\
H &= \text{LN}_{ef}(X_2)\in\mathbb{R}^{T_s\times d}
\end{aligned}
$$

其中 SelfAttn、FFN、LayerNorm 的公式與 [`05a1`](05a1-forward-propagation.md) §1、§4、§5 **逐字相同**，本文不重推。

**唯一的差別是遮罩：encoder 的 self-attention 沒有因果遮罩。**

$$
S^{\text{enc}} = \frac{QK^\top}{\sqrt{d_k}}\in\mathbb{R}^{T_s\times T_s},\qquad
A^{\text{enc}} = \text{softmax}_{\text{row}}\big(S^{\text{enc}}\big)
$$

沒有 `masked_fill`、沒有 $-\infty$。每個來源位置都能看到**全部**來源位置，包括自己右邊的。這是合理的：來源句子在一開始就完整給定了，沒有「偷看未來」的問題——要偷看的「未來」是**目標**句，不是來源句。

把它與 GPT 對照：

| | GPT（[`05a1`](05a1-forward-propagation.md) §2）| Encoder |
|---|---|---|
| $A$ 的形狀 | $T\times T$ 下三角 | $T_s\times T_s$ 全滿 |
| 位置 0 看得到 | 只有自己 | 全部 $T_s$ 個 |
| 第一列的 softmax | 只有一項存活 ⇒ 恆為 $[1,0,\dots,0]$ | 正常分佈 |

最後一列在反向時很關鍵：GPT 第 0 列的 attention 權重是 one-hot，而 one-hot 會讓 softmax 的 Jacobian 退化成零矩陣（[`05b1`](05b1-backward-propagation.md) §3.2、§8.5），於是那一列**完全拿不到梯度**（[`05b2`](05b2-backward-example.md) §5.4 的 $G^S$ 第 0 列就是 $[0,0]$）。Encoder 沒有遮罩，就沒有這個飽和問題。

**這正是 BERT。** 把 GPT 的因果遮罩拿掉，剩下的就是 encoder；BERT 就是把這種 Block 疊 $N$ 層（[`07`](07-bert-encoder-only.md) §1）。所以本節不是新架構，而是主線已有的兩個東西的交集。

**Encoder 的輸出 $H$ 就是「讀懂輸入」的成果**：$T_s$ 個向量，每一個都帶著它與整個來源句互動後的語意。這排向量會原封不動地交給 decoder 的每一層 cross-attention 使用——**算一次，用很多次**（本文 decoder 只有一層，多層時 $H$ 會被每一層各讀一次）。

### B2. Decoder 的 Masked Self-Attention

Decoder 輸入是右移後的目標序列：

$$
Y_0 = E_t[\tilde y] + P_t\in\mathbb{R}^{T_t\times d}
$$

第一個子層是**帶因果遮罩**的 self-attention，與 GPT 完全相同：

$$
Y_1 = Y_0 + \text{MaskedSelfAttn}\big(\text{LN}_{d1}(Y_0)\big)
$$

$$
S^{\text{self}}_{ij} = \begin{cases}
\dfrac{q_i\cdot k_j}{\sqrt{d_k}} & j\le i\\[4pt]
-\infty & j> i
\end{cases}
$$

遮罩的必要性與 GPT 一模一樣：teacher forcing 把整個目標序列一次餵進去，若不遮，位置 0 就會直接看到位置 1 的答案 `I`，訓練變成抄答案。完整說明與三行實作見 [`05a1`](05a1-forward-propagation.md) §2。

**所以 decoder 的第一個子層 ＝ 一個 GPT block 的 attention 部分。** 真正屬於 encoder-decoder 架構的新東西是下一個子層。

### B3. Cross-Attention

這是本文的核心。Decoder 到目前為止只看了自己，它是怎麼讀到來源資訊的？

**答案：再做一次 attention，但這次 Query 與 Key/Value 來自不同的序列。**

$$
\boxed{\;
Q^{\text{cross}} = \text{LN}_{d2}(Y_1)\,W_Q^{c},\qquad
K^{\text{cross}} = H\,W_K^{c},\qquad
V^{\text{cross}} = H\,W_V^{c}
\;}
$$

$Q$ 來自 **decoder**（它想問什麼），$K,V$ 來自 **encoder 輸出 $H$**（可以查到什麼）。之後的步驟與一般 attention 一字不差：

$$
S^{\text{cross}} = \frac{Q^{\text{cross}}\,(K^{\text{cross}})^\top}{\sqrt{d_k}},\qquad
A^{\text{cross}} = \text{softmax}_{\text{row}}\big(S^{\text{cross}}\big),\qquad
C^{\text{cross}} = A^{\text{cross}}V^{\text{cross}}
$$

$$
Y_2 = Y_1 + C^{\text{cross}}
$$

**形狀鏈要看清楚，這裡是唯一會出現非方陣的地方：**

$$
\underbrace{Q^{\text{cross}}}_{T_t\times d_k}
\;\cdot\;
\underbrace{(K^{\text{cross}})^\top}_{d_k\times T_s}
\;=\;
\underbrace{S^{\text{cross}}}_{\textstyle T_t\times T_s}
\;\xrightarrow{\ \text{softmax}\ }\;
\underbrace{A^{\text{cross}}}_{T_t\times T_s}
\;\cdot\;
\underbrace{V^{\text{cross}}}_{T_s\times d_k}
\;=\;
\underbrace{C^{\text{cross}}}_{T_t\times d_k}
$$

self-attention 的 $A$ 一定是方陣（$T\times T$），因為 query 與 key 是同一批位置；cross-attention 的 $A^{\text{cross}}$ 則是 $T_t\times T_s$——**列數是目標長度、欄數是來源長度**。第 $t$ 列告訴你「生第 $t$ 個目標詞時，注意力怎麼分配到 $T_s$ 個來源詞上」。這張表就是機器翻譯論文裡常見的**對齊圖**（alignment map）。

**兩個關鍵性質：**

1. **不需要因果遮罩。** 遮罩的目的是防止看到「還沒生成的目標詞」，但 $K,V$ 全部來自**來源句**，來源句從一開始就完整可見。所以 $A^{\text{cross}}$ 每一列都是完整的 $T_s$ 項分佈。（真實系統若有 batch padding，仍需 padding mask 把補位遮掉——那與因果遮罩是兩回事。）

2. **每一列仍然各自 softmax、和為 1。** $\sum_{i=1}^{T_s}A^{\text{cross}}_{t,i}=1$。注意這裡是沿**來源**維度正規化，不是沿目標維度。

**Cross-Attention 就是 §A3 的 Bahdanau attention 換了打分方式。** 對照一下就清楚：

| | Bahdanau（§A3）| Cross-Attention（本節）|
|---|---|---|
| 誰發問 | $s_{t-1}$（decoder 前一狀態）| $\text{LN}_{d2}(Y_1)$ 的第 $t$ 列 |
| 查什麼 | $h_i$（encoder 狀態）| $H$ 的第 $i$ 列 |
| 打分 | $v_a^\top\tanh(sW_a+h_iU_a)$ | $\dfrac{q_t\cdot k_i}{\sqrt{d_k}}$ |
| 權重 | $\alpha_{t,i}$（沿 $i$ softmax）| $A^{\text{cross}}_{t,i}$（沿 $i$ softmax）|
| 讀出 | $c_t=\sum_i\alpha_{t,i}h_i$ | $C_t=\sum_i A^{\text{cross}}_{t,i}V_i$ |

**結構完全同構**：發問 → 打分 → softmax → 加權讀取。差別只有兩點——打分函數從加性換成點積（所以可以整批矩陣運算），以及讀出的是 $V=HW_V^c$ 而非 $h_i$ 本身（多了一次可訓練投影）。這就是為什麼說「cross attention 早於 Transformer」：機制 2014 年就有了，Transformer 換掉了它的打分方式並把它矩陣化。

第三個子層是 FFN，與 GPT 相同：

$$
Y_3 = Y_2 + \text{FFN}\big(\text{LN}_{d3}(Y_2)\big),\qquad
h^{\text{out}} = \text{LN}_{df}(Y_3)
$$

$$
\text{logits} = h^{\text{out}} W_{lm}^\top,\qquad
P = \text{softmax}_{\text{row}}(\text{logits}),\qquad
L = \frac{1}{T_t}\sum_t \text{CE}(P_t, y_t)
$$

### B4. 三種 Attention 一表對照

到這裡，三種 attention 都出現過了。它們用的是**同一條公式** $\text{softmax}(QK^\top/\sqrt{d_k})V$，差別只在 $Q,K,V$ 從哪來、要不要遮：

| | Encoder Self-Attn | Decoder Masked Self-Attn | Cross-Attn |
|---|---|---|---|
| $Q$ 來源 | 來源序列 | 目標序列 | **目標序列** |
| $K,V$ 來源 | 來源序列 | 目標序列 | **來源序列（$H$）** |
| 因果遮罩 | 否 | **是** | 否 |
| $A$ 形狀 | $T_s\times T_s$ | $T_t\times T_t$（下三角）| $T_t\times T_s$ |
| 在哪個文件推導 | 本文 §B1、[`07`](07-bert-encoder-only.md) | [`05a1`](05a1-forward-propagation.md) §2 | 本文 §B3 |
| GPT 有沒有 | 沒有 | 有 | 沒有 |

**GPT 只保留中間那一欄**——這就是 [`04a`](04a-gpt-decoder-only.md) §2 說的「把 Encoder 整個拿掉，連帶拿掉 Cross-Attention」。

### B5. 完整 Pipeline

```
【Encoder】來源 (T_s,)
  → E_s[x] + P_s                            (T_s, d)
  → LN_e1 → Self-Attn（無遮罩）→ 殘差 ①      (T_s, d)
  → LN_e2 → FFN → 殘差 ②                    (T_s, d)
  → LN_ef                                   (T_s, d) = H ────┐
                                                              │
【Decoder】目標（右移）(T_t,)                                  │
  → E_t[ỹ] + P_t                            (T_t, d)         │
  → LN_d1 → Masked Self-Attn → 殘差 ①        (T_t, d)         │
  → LN_d2 → Cross-Attn（K,V 來自 H）→ 殘差 ② (T_t, d) ◄───────┘
  → LN_d3 → FFN → 殘差 ③                    (T_t, d)
  → LN_df → lm_head                         (T_t, |V_t|)
  → softmax → Cross-Entropy                 純量 L
```

注意 **decoder 有三個子層、三處殘差、四個 LayerNorm**，比 GPT 的 block 多一整層。多出來的就是 cross-attention。

**這張圖裡只有一條線把兩邊連起來：$H$ 進入 decoder 的 cross-attention。** 這件事在反向時會有戲劇性的後果——encoder 的**全部**參數都只能透過這條線拿到梯度（[`10b1`](10b1-seq2seq-backward.md) §B2）。

### B6. 參數清單

| 區塊 | 參數 | 出現在 |
|---|---|---|
| Encoder | $E_s,\ P_s$ | §B1 |
| | $\gamma_{e1},\beta_{e1}$；$W_Q^{e},W_K^{e},W_V^{e}$ | §B1 |
| | $\gamma_{e2},\beta_{e2}$；$W_1^{e},W_2^{e}$ | §B1 |
| | $\gamma_{ef},\beta_{ef}$ | §B1 |
| Decoder | $E_t,\ P_t$ | §B2 |
| | $\gamma_{d1},\beta_{d1}$；$W_Q^{d},W_K^{d},W_V^{d}$ | §B2 |
| | $\gamma_{d2},\beta_{d2}$；$W_Q^{c},W_K^{c},W_V^{c}$ | §B3 |
| | $\gamma_{d3},\beta_{d3}$；$W_1^{d},W_2^{d}$ | §B3 |
| | $\gamma_{df},\beta_{df}$；$W_{lm}$ | §B3 |

具體形狀與取值見 [`10a2`](10a2-seq2seq-forward-example.md) §0.2（本文的設定共 **32 個張量、234 個參數**）。

**與 §A6 的 RNN 版比一比：** 張量個數從 14 變成 32，而且這還只是**各一層**。真實的原始 Transformer 是 6 層 encoder ＋ 6 層 decoder，每層各自一份參數——參數量隨層數線性成長。RNN 則不論跑幾個時間步都是同一份權重。**參數共享 vs 每層獨立**，是兩代最根本的取捨之一。

---

## C. 兩代對照

| 面向 | RNN Seq2Seq（§A）| Transformer Enc-Dec（§B）|
|---|---|---|
| Encoder 能否平行 | **否**，$h_i$ 要等 $h_{i-1}$，$O(T_s)$ 個序列步 | **是**，一次矩陣運算 |
| Decoder 訓練能否平行 | 否（$s_t$ 要等 $s_{t-1}$）| **是**（teacher forcing ＋ 因果遮罩，所有位置同時算）|
| Decoder 推論能否平行 | 否 | 否（自迴歸，本質限制）|
| 來源位置 $i$ → 目標位置 $t$ 的路徑長度 | attention 提供 $O(1)$ 捷徑，但 encoder 內部仍是 $O(T_s)$ | 全程 $O(1)$ |
| 位置資訊怎麼來 | 遞迴順序天然帶入 | 必須另加位置編碼 |
| 參數與層數 | 跨時間共享，一份 | 每層獨立，$N$ 層 $N$ 份 |
| Attention 打分 | 加性，含 $\tanh$，逐配對計算 | 點積，無非線性，整批矩陣 |

**第二列是 Transformer 真正的殺手鐧。** RNN 的 decoder 訓練時也不能平行——雖然 teacher forcing 讓所有 $\tilde y_t$ 都已知，但 $s_t$ 依賴 $s_{t-1}$，還是得逐步跑。Transformer 用因果遮罩取代了「逐步」：所有位置的計算互不依賴，遮罩負責保證位置 $t$ 看不到 $t$ 之後的東西。於是**整個訓練樣本一次算完**。

**第四列是梯度的關鍵。** RNN 加了 attention 之後，$c_t\to h_i$ 確實是 $O(1)$ 的捷徑；但 encoder 內部 $h_i\to h_1$ 仍然是 $O(T_s)$ 的連乘鏈，來源句開頭的詞向量 $E_s$ 還是要靠這條長鏈才拿得到完整梯度。Transformer 連這一段都拆掉了。

> **RNN 沒有被淘汰的部分。** 推論時 RNN 的狀態是固定大小的 $s_t$，每生一個 token 的成本是 $O(1)$；Transformer 得看整個已生成序列，即使用 KV Cache（[`04b`](04b-nanogpt-walkthrough.md) §9）成本仍隨長度成長。這是近年 Mamba 等狀態空間模型重新被關注的動機之一。

---

## D. 四份文件的節號鏡像

本組四份文件的節號互相對應，可以直接對開核對：

| 主題 | 前向・符號 | 前向・數值 | 後向・符號 | 後向・數值 |
|---|---|---|---|---|
| 任務設定／參數清單 | **§1、§A6、§B6** | [`10a2`](10a2-seq2seq-forward-example.md) §0、§0.2 | — | [`10b2`](10b2-seq2seq-backward-example.md) §C |
| Encoder RNN | **§A1** | [`10a2`](10a2-seq2seq-forward-example.md) §A1 | [`10b1`](10b1-seq2seq-backward.md) §A5–§A6 | [`10b2`](10b2-seq2seq-backward-example.md) §A5 |
| Bahdanau Attention | **§A3** | [`10a2`](10a2-seq2seq-forward-example.md) §A2 | [`10b1`](10b1-seq2seq-backward.md) §A3–§A4 | [`10b2`](10b2-seq2seq-backward-example.md) §A3 |
| Decoder RNN | **§A4** | [`10a2`](10a2-seq2seq-forward-example.md) §A3 | [`10b1`](10b1-seq2seq-backward.md) §A2 | [`10b2`](10b2-seq2seq-backward-example.md) §A2 |
| 輸出層＋CE（RNN）| **§A5** | [`10a2`](10a2-seq2seq-forward-example.md) §A4 | [`10b1`](10b1-seq2seq-backward.md) §A1 | [`10b2`](10b2-seq2seq-backward-example.md) §A1 |
| Transformer Encoder | **§B1** | [`10a2`](10a2-seq2seq-forward-example.md) §B1 | [`10b1`](10b1-seq2seq-backward.md) §B4 | [`10b2`](10b2-seq2seq-backward-example.md) §B4 |
| Decoder Masked Self-Attn | **§B2** | [`10a2`](10a2-seq2seq-forward-example.md) §B2 | [`10b1`](10b1-seq2seq-backward.md) §B3 | [`10b2`](10b2-seq2seq-backward-example.md) §B3 |
| **Cross-Attention** | **§B3** | [`10a2`](10a2-seq2seq-forward-example.md) §B3 | [`10b1`](10b1-seq2seq-backward.md) §B2 | [`10b2`](10b2-seq2seq-backward-example.md) §B2 |
| 輸出層＋CE（Transformer）| **§B5** | [`10a2`](10a2-seq2seq-forward-example.md) §B4 | [`10b1`](10b1-seq2seq-backward.md) §B1 | [`10b2`](10b2-seq2seq-backward-example.md) §B1 |

---

## 下一步

- **看數字** → [`10a2-seq2seq-forward-example.md`](10a2-seq2seq-forward-example.md)：把本文每條公式代進實際數值，兩個模型各算一次到 loss。
- **看反向** → [`10b1-seq2seq-backward.md`](10b1-seq2seq-backward.md)：BPTT 的完整推導、Cross-Attention 的梯度分岔。
- **看故事與工藝** → [`advanced/Seq2Seq-and-Decoding-Techniques.md`](../advanced/Seq2Seq-and-Decoding-Techniques.md)：beam search、copy mechanism、scheduled sampling、BLEU 與 RL。
- **看 encoder 單獨用** → [`07-bert-encoder-only.md`](07-bert-encoder-only.md)：把 §B1 的 encoder 疊 $N$ 層、換成 MLM 目標，就是 BERT。
- **看 decoder 單獨用** → [`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)：把 §B2 留下、§B1 與 §B3 拿掉，就是 GPT。
