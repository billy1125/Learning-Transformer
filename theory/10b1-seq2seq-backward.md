# 10b1｜Seq2Seq 向後傳播：BPTT 與 Cross-Attention 的梯度（符號推導）

> **本文做什麼：** 把 [`10a1`](10a1-seq2seq-forward.md) 推導的兩個模型，各自的**反向傳播完整推導一次**。
>
> - **§A：BPTT（Backpropagation Through Time）**——RNN Seq2Seq 的梯度怎麼穿越時間、為什麼會消失、Bahdanau attention 又怎麼提供一條捷徑。
> - **§B：Transformer Encoder-Decoder 的反向**——重點是 **Cross-Attention 的梯度分岔**：梯度在這裡裂成兩股，一股回 decoder、一股回 encoder，而**這是 encoder 唯一的梯度入口**。
>
> **前置文件：** [`10a1-seq2seq-forward.md`](10a1-seq2seq-forward.md)（前向符號）。§B 大量沿用 [`05b1-backward-propagation.md`](05b1-backward-propagation.md) 已推導過的模組（Softmax、LayerNorm、Linear Projection 的梯度），會明確標出哪一節可直接引用。
>
> **下一步：** → [`10b2-seq2seq-backward-example.md`](10b2-seq2seq-backward-example.md)（沿用 [`10a2`](10a2-seq2seq-forward-example.md) 的兩組數字，把本文每條公式實際算一次）

> **記號約定（全文一致）：** 對任一量 $X$，寫 $G^{X}\equiv\dfrac{\partial L}{\partial X}$，形狀與 $X$ 相同。所有向量都是**列向量**（$1\times d$），矩陣乘法一律「列向量乘矩陣」。$\odot$ 是逐元素乘法。$\mathbb{1}[\cdot]$ 是指示函數。

---

## 目錄

- [A. BPTT：RNN Seq2Seq 的反向傳播](#a-bptt-rnn-seq2seq-的反向傳播)
  - [A1. 從 Loss 到輸出層](#a1-從-loss-到輸出層)
  - [A2. Decoder 的時間鏈](#a2-decoder-的時間鏈)
  - [A3. Encoder 狀態的兩條梯度來源](#a3-encoder-狀態的兩條梯度來源)
  - [A4. 加性 Attention 的梯度](#a4-加性-attention-的梯度)
  - [A5. 參數共享 ⇒ 梯度跨時間累加](#a5-參數共享--梯度跨時間累加)
  - [A6. 梯度消失：為什麼連乘一定衰減](#a6-梯度消失為什麼連乘一定衰減)
- [B. Transformer Encoder-Decoder 的反向傳播](#b-transformer-encoder-decoder-的反向傳播)
  - [B1. 從 Loss 到 Decoder 的 FFN](#b1-從-loss-到-decoder-的-ffn)
  - [B2. Cross-Attention：梯度在這裡分岔](#b2-cross-attention梯度在這裡分岔)
  - [B3. Decoder 的 Masked Self-Attention](#b3-decoder-的-masked-self-attention)
  - [B4. Encoder 端的反向](#b4-encoder-端的反向)
  - [B5. 反向五步總覽](#b5-反向五步總覽)
- [C. 兩代反向對照](#c-兩代反向對照)

---

# A. BPTT：RNN Seq2Seq 的反向傳播

前向見 [`10a1`](10a1-seq2seq-forward.md) §A。回顧一下計算圖的形狀——這決定了反向要怎麼走：

```
E_s[x_1] ─┐              E_s[x_2] ─┐
          ▼                        ▼
   h_0 ─→ a_1 ─→ h_1 ────→ a_2 ─→ h_2 ──────┐（s_0 = h_2）
                  │  W_hh        │           │
                  └───┐      ┌───┘           ▼
                      ▼      ▼         ┌──→ g_1 ─→ s_1 ─→ z_1 ─→ CE_1
                   Bahdanau attention  │      ▲
                      │      │         │      │ W_ss
                      ├─ c_1 ┘         │      ▼
                      └─ c_2 ──────────┼──→ g_2 ─→ s_2 ─→ z_2 ─→ CE_2
                                       │
                              E_t[ỹ_t] ┘
```

有兩件事現在就要注意，它們是 §A3 與 §A5 的伏筆：

1. **$h_i$ 有很多條出路**——時間鏈（$a_{i+1}$）、decoder 起點（只有 $h_{T_s}$）、以及每一個 $c_t$ 與每一個 attention 分數。反向時這些梯度要**全部加起來**。
2. **每一個時間步用的是同一份權重**。所以 $W_{hh}$ 這種參數的梯度是各時間步貢獻的**總和**。

## A1. 從 Loss 到輸出層（前向見 [`10a1`](10a1-seq2seq-forward.md) §A5；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §A1）

前向：$z_t = s_tW_{out}+b_{out}$，$p_t=\text{softmax}(z_t)$，$L=\frac{1}{T_t}\sum_t\big(-\log p_t[y_t]\big)$。

**Softmax 與 Cross-Entropy 合併之後，梯度出奇地簡單。** 直接推導：先把單步的 CE 寫開，

$$
\text{CE}_t = -\log p_t[y_t]
= -\log\frac{e^{z_{t,y_t}}}{\sum_k e^{z_{t,k}}}
= -z_{t,y_t} + \log\sum_k e^{z_{t,k}}
$$

對第 $j$ 個分量微分。第一項只在 $j=y_t$ 時有貢獻，貢獻 $-1$；第二項是 log-sum-exp，其導數正是 softmax：

$$
\frac{\partial}{\partial z_{t,j}}\log\sum_k e^{z_{t,k}}
= \frac{e^{z_{t,j}}}{\sum_k e^{z_{t,k}}} = p_{t,j}
$$

兩項合起來：

$$
\frac{\partial\,\text{CE}_t}{\partial z_{t,j}} = p_{t,j} - \mathbb{1}[j=y_t]
$$

再乘上 $L$ 對 $\text{CE}_t$ 的係數 $1/T_t$：

$$
\boxed{\;G^{z_t} = \frac{1}{T_t}\Big(p_t - \text{onehot}(y_t)\Big)\;}
$$

**解讀：正確類別的分量為負（要調高），其餘為正（要調低），整列和為零**（因為 $\sum_j p_{t,j}=1$）。梯度下降就是把機率質量從錯的類別搬到對的類別。這與 GPT 的情形完全相同（[`05b1`](05b1-backward-propagation.md) §3）。

**輸出層參數：** $z_t=s_tW_{out}+b_{out}$ 是標準的線性層（[`05b1`](05b1-backward-propagation.md) §4.1）：

$$
G^{W_{out}} = \sum_{t=1}^{T_t} s_t^\top\,G^{z_t},\qquad
G^{b_{out}} = \sum_{t=1}^{T_t} G^{z_t}
$$

**注意這裡的 $\sum_t$。** $W_{out}$ 被 $T_t$ 個時間步共用，每一步都貢獻一個外積，全部加起來——這就是 §A5 要展開講的「參數共享 ⇒ 累加」。

往回傳給狀態：

$$
G^{s_t}\big|_{\text{輸出層}} = G^{z_t}\,W_{out}^\top
$$

寫成「$\big|_{\text{輸出層}}$」是因為 $s_t$ 還有別的下游，下一節補齊。

## A2. Decoder 的時間鏈（前向見 [`10a1`](10a1-seq2seq-forward.md) §A4；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §A2）

前向：$g_t = e^t_tW_{ys} + s_{t-1}W_{ss} + c_tW_{cs} + b_s$，$s_t=\tanh(g_t)$。

**先看 $s_t$ 到底有幾個下游。** 翻回前向公式找 $s_t$ 出現的地方：

| 下游 | 出現在 | 存在條件 |
|---|---|---|
| $z_t$（輸出層）| [`10a1`](10a1-seq2seq-forward.md) §A5 | 恆有 |
| $g_{t+1}$（經 $W_{ss}$，時間鏈）| [`10a1`](10a1-seq2seq-forward.md) §A4 | $t<T_t$ |
| $w_{t+1,i}$（經 $W_a$，attention 打分）| [`10a1`](10a1-seq2seq-forward.md) §A3 | $t<T_t$，對所有 $i$ |

第三條容易漏掉：Bahdanau 的分數用的是 $s_{t-1}$，所以 $s_t$ 會被**第 $t+1$ 步的 attention** 用到。這裡 $w_{t,i}\equiv s_{t-1}W_a+h_iU_a$ 是 $\tanh$ 之前的量（$u_{t,i}=\tanh(w_{t,i})$），它的梯度在 §A4 算。

三條相加：

$$
\boxed{\;
G^{s_t} = \underbrace{G^{z_t}W_{out}^\top}_{\text{輸出層}}
+ \underbrace{\mathbb{1}[t<T_t]\;G^{g_{t+1}}W_{ss}^\top}_{\text{時間鏈}}
+ \underbrace{\mathbb{1}[t<T_t]\sum_{i=1}^{T_s} G^{w_{t+1,i}}W_a^\top}_{\text{attention 打分}}
\;}
$$

**穿過 $\tanh$。** $s_t=\tanh(g_t)$ 逐元素作用，導數 $\tanh'(g)=1-\tanh^2(g)=1-s^2$，所以

$$
\boxed{\;G^{g_t} = G^{s_t}\odot\big(\mathbf1 - s_t\odot s_t\big)\;}
$$

**這個式子是梯度消失的元兇之一。** 因為 $s_t\in(-1,1)$，恆有 $0<1-s_t^2\le1$：**每穿過一個時間步，梯度至少被乘上一個不大於 1 的數**，而且 $|s_t|$ 越接近 1（越飽和）衰減越劇烈。§A6 會定量。

**$g_t$ 往三個方向分配梯度**（三個加項各自往回走）：

$$
G^{c_t} = G^{g_t}W_{cs}^\top,\qquad
G^{e^t_t} = G^{g_t}W_{ys}^\top,\qquad
G^{s_{t-1}}\big|_{\text{時間鏈}} = G^{g_t}W_{ss}^\top
$$

$G^{e^t_t}$ 要累加回目標詞嵌入矩陣被查到的那一列：

$$
G^{E_t[\tilde y_t]}\ \mathrel{+}=\ G^{e^t_t}
$$

**沒被查到的列梯度為零**——這就是 embedding 的**稀疏更新**（機制同 [`05b1`](05b1-backward-propagation.md) §10.4）。本例的 `eat` 只當過目標、沒當過 decoder 輸入，所以 $G^{E_t[2]}=\mathbf0$，即使它是正確答案。

**Decoder 的四個參數：**

$$
G^{W_{ys}} = \sum_t (e^t_t)^\top G^{g_t},\quad
G^{W_{ss}} = \sum_t (s_{t-1})^\top G^{g_t},\quad
G^{W_{cs}} = \sum_t c_t^\top G^{g_t},\quad
G^{b_s} = \sum_t G^{g_t}
$$

**$t=1$ 時 $s_0=h_{T_s}$。** 所以 $G^{s_0}\big|_{\text{時間鏈}}=G^{g_1}W_{ss}^\top$ 不會消失在虛空裡，它會直接變成 $G^{h_{T_s}}$ 的一部分——見下一節。

## A3. Encoder 狀態的兩條梯度來源（前向見 [`10a1`](10a1-seq2seq-forward.md) §A1、§A3；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §A3）

**這是整份文件最重要的一節。**

Encoder 狀態 $h_i$ 的下游有四個，但可以歸成兩類：

$$
\boxed{\;
G^{h_i} = \underbrace{
\underbrace{\mathbb{1}[i<T_s]\;G^{a_{i+1}}W_{hh}^\top}_{\text{下一個 encoder 狀態}}
+ \underbrace{\mathbb{1}[i=T_s]\;G^{s_0}}_{\text{decoder 起點}}
}_{\textstyle \text{（一）時間鏈那股}}
\;+\;
\underbrace{
\underbrace{\sum_{t=1}^{T_t} G^{w_{t,i}}U_a^\top}_{\text{attention 打分}}
+ \underbrace{\sum_{t=1}^{T_t}\alpha_{t,i}\,G^{c_t}}_{\text{attention 讀取}}
}_{\textstyle \text{（二）attention 那股}}
\;}
$$

四項的來歷逐一對照前向公式：

| 項 | 來自前向哪一條 |
|---|---|
| $G^{a_{i+1}}W_{hh}^\top$ | $a_{i+1}=e^s_{i+1}W_{xh}+h_iW_{hh}+b_h$ |
| $G^{s_0}$ | $s_0 = h_{T_s}$（恆等，梯度原樣傳）|
| $\sum_t G^{w_{t,i}}U_a^\top$ | $w_{t,i}=s_{t-1}W_a+h_iU_a$，每個 $t$ 都用到 $h_i$ |
| $\sum_t\alpha_{t,i}G^{c_t}$ | $c_t=\sum_j\alpha_{t,j}h_j$，對 $h_i$ 的偏導是純量 $\alpha_{t,i}$ |

> **第二項的歸類要說清楚。** $s_0=h_{T_s}$ 是恆等映射，所以 $h_{T_s}$ 收到的是**整個** $G^{s_0}$——而 $G^{s_0}$ 依 §A2 又可再拆成「時間鏈」與「attention 打分」兩份。這裡把它整包算進「時間鏈那股」，是**保守**的歸類：它讓時間鏈那股看起來比實際大，所以下面「attention 那股更大」的結論若成立，只會被低估、不會被高估。[`10b2`](10b2-seq2seq-backward-example.md) §A3 會把 $G^{s_0}$ 的兩份分別列出來。

**兩股的差別在路徑長度，這正是重點。**

**（一）時間鏈那股**要一步一步往回走。梯度想從 $s_1$ 回到 $h_1$（$T_s=50$ 時），必須依序穿過 $h_{50},h_{49},\dots,h_2$，總共 $O(T_s)$ 次「乘 $W_{hh}$、乘 $\tanh'$」。§A6 會證明這條路必然衰減。

**（二）attention 那股是直達的。** 看第四項：

$$
\frac{\partial c_t}{\partial h_i} = \alpha_{t,i}\cdot I
\qquad\Longrightarrow\qquad
G^{h_i}\big|_{\text{讀取}} = \sum_t \alpha_{t,i}\,G^{c_t}
$$

$c_t$ 是 $h_i$ 的**線性**加權和，梯度回傳只是乘上純量權重 $\alpha_{t,i}$——**沒有矩陣連乘、沒有 $\tanh'$、路徑長度 $O(1)$**，而且與 $i$ 距離序列尾端多遠完全無關。來源句第 1 個詞與第 50 個詞，透過這條路拿到的梯度是同一個量級。

**這就是 Bahdanau attention 真正的貢獻。** 它表面上是「讓 decoder 每步能看不同的來源位置」（前向的好處），實質上同時開了一條**梯度高速公路**（反向的好處）。[`10b2`](10b2-seq2seq-backward-example.md) §D 會用實際數字顯示：即使在 $T_s=2$ 這麼短的序列上，attention 那股帶回來的梯度就已經**大於**時間鏈那股。

最後穿過 encoder 的 $\tanh$：

$$
G^{a_i} = G^{h_i}\odot\big(\mathbf1 - h_i\odot h_i\big)
$$

## A4. 加性 Attention 的梯度（前向見 [`10a1`](10a1-seq2seq-forward.md) §A3；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §A3）

前向鏈（把 $\tanh$ 之前的量命名為 $w_{t,i}$）：

$$
w_{t,i}=s_{t-1}W_a+h_iU_a
\;\to\;
u_{t,i}=\tanh(w_{t,i})
\;\to\;
e_{t,i}=u_{t,i}v_a^\top
\;\to\;
\alpha_{t,\cdot}=\text{softmax}(e_{t,\cdot})
\;\to\;
c_t=\sum_i\alpha_{t,i}h_i
$$

反向就是把這條鏈倒著走。

**(a) 從 $c_t$ 到 $\alpha_{t,i}$。** $c_t=\sum_i\alpha_{t,i}h_i$ 對純量 $\alpha_{t,i}$ 的偏導是向量 $h_i$，所以

$$
G^{\alpha_{t,i}} = G^{c_t}\cdot h_i^\top = \langle G^{c_t},\,h_i\rangle
$$

（一個純量：把 $G^{c_t}$ 與 $h_i$ 做內積。）

**(b) 穿過 softmax。** 這是 softmax 的標準反向，完整的 Jacobian 推導見 [`05b1`](05b1-backward-propagation.md) §3.2，結論是

$$
\boxed{\;G^{e_{t,i}} = \alpha_{t,i}\Big(G^{\alpha_{t,i}} - \sum_{j=1}^{T_s}\alpha_{t,j}G^{\alpha_{t,j}}\Big)\;}
$$

括號裡是「自己的梯度減掉整列的加權平均」。**這保證 $\sum_i G^{e_{t,i}}=0$**：

$$
\sum_i \alpha_{t,i}G^{\alpha_{t,i}} - \Big(\sum_i\alpha_{t,i}\Big)\sum_j\alpha_{t,j}G^{\alpha_{t,j}}
= \sum_i \alpha_{t,i}G^{\alpha_{t,i}} - 1\cdot\sum_j\alpha_{t,j}G^{\alpha_{t,j}} = 0
$$

（用到 $\sum_i\alpha_{t,i}=1$。）合理：softmax 的輸出被約束在單純形上，梯度只能沿「一邊加、一邊減」的方向推。

**(c) 從純量分數回到向量。** $e_{t,i}=u_{t,i}v_a^\top$ 對 $u_{t,i}$ 的偏導是 $v_a$：

$$
G^{u_{t,i}} = G^{e_{t,i}}\,v_a
$$

**(d) 穿過 $\tanh$：**

$$
G^{w_{t,i}} = G^{u_{t,i}}\odot\big(\mathbf1-u_{t,i}\odot u_{t,i}\big)
$$

**(e) 三個 attention 參數。** 注意 $W_a,U_a,v_a$ 被**所有** $(t,i)$ 配對共用，所以是雙重求和：

$$
G^{v_a} = \sum_{t=1}^{T_t}\sum_{i=1}^{T_s} G^{e_{t,i}}\,u_{t,i},\qquad
G^{W_a} = \sum_{t,i} s_{t-1}^\top\,G^{w_{t,i}},\qquad
G^{U_a} = \sum_{t,i} h_i^\top\,G^{w_{t,i}}
$$

$T_t\times T_s$ 個配對，每個都貢獻一份。

**(f) 往回傳的兩個方向**（已在 §A2、§A3 用到）：

$$
G^{s_{t-1}}\big|_{\text{attention}} = \sum_i G^{w_{t,i}}W_a^\top,
\qquad
G^{h_i}\big|_{\text{打分}} = \sum_t G^{w_{t,i}}U_a^\top
$$

> **加性 attention 的梯度比點積 attention 長。** 對照 [`05b1`](05b1-backward-propagation.md) §8.6：點積 attention 從分數回到 $Q,K$ 只要一次矩陣乘法（$G^Q=G^SK/\sqrt{d_k}$）。這裡要先過 $v_a$、再過 $\tanh$、再過 $W_a$ 或 $U_a$——多兩個環節，其中 $\tanh'$ 又是一個 $\le1$ 的因子。**加性 attention 不只前向較慢，反向的梯度也被多壓一次。**

## A5. 參數共享 ⇒ 梯度跨時間累加（前向見 [`10a1`](10a1-seq2seq-forward.md) §A1；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §A5）

Encoder 的三個參數：

$$
G^{W_{xh}} = \sum_{i=1}^{T_s} (e^s_i)^\top G^{a_i},\qquad
G^{W_{hh}} = \sum_{i=1}^{T_s} h_{i-1}^\top\,G^{a_i},\qquad
G^{b_h} = \sum_{i=1}^{T_s} G^{a_i}
$$

以及來源詞嵌入（同樣是稀疏更新）：

$$
G^{E_s[x_i]}\ \mathrel{+}=\ G^{a_i}W_{xh}^\top
$$

**為什麼一定是求和？** 因為前向時 $W_{hh}$ 在 $T_s$ 個不同的位置各用了一次。多元微分的鏈鎖法則對「同一個變數出現多次」的處理就是把各處的偏導加起來：

$$
\frac{\partial L}{\partial W_{hh}}
= \sum_{i=1}^{T_s}\frac{\partial L}{\partial a_i}\cdot\frac{\partial a_i}{\partial W_{hh}}
$$

這與 Transformer 形成鮮明對比：Transformer 每一層有自己的 $W_Q^{(\ell)}$，第 $\ell$ 層的梯度不會與第 $\ell'$ 層混在一起。**RNN 的一份 $W_{hh}$ 要同時滿足所有時間步的需求**——這是它參數少的原因，也是它難訓練的原因之一。

> **$h_0=\mathbf0$ 會讓第一項消失。** $G^{W_{hh}}$ 的 $i=1$ 那一項是 $h_0^\top G^{a_1}=\mathbf0$。所以序列長度 $T_s$ 時，實際有貢獻的只有 $T_s-1$ 項。[`10b2`](10b2-seq2seq-backward-example.md) §A5 會看到這個現象（$T_s=2$ 時 $G^{W_{hh}}$ 只有一項有效）。

## A6. 梯度消失：為什麼連乘一定衰減

把 §A3 的「時間鏈那股」與 §A2 的 $\tanh$ 反向串起來，梯度從 $h_i$ 傳到 $h_{i-1}$ 的規則是：

$$
G^{h_{i-1}}\big|_{\text{時間鏈}}
= G^{a_i}W_{hh}^\top
= \Big(G^{h_i}\odot(\mathbf1-h_i\odot h_i)\Big)W_{hh}^\top
$$

把「逐元素乘」寫成對角矩陣，就看得出這是一次線性變換：

$$
G^{h_{i-1}} = G^{h_i}\,D_i\,W_{hh}^\top,
\qquad D_i \equiv \text{diag}\big(\mathbf1-h_i\odot h_i\big)
$$

**往回走 $k$ 步就是連乘 $k$ 次：**

$$
\boxed{\;G^{h_{i-k}}\big|_{\text{時間鏈}} = G^{h_i}\prod_{m=0}^{k-1}\Big(D_{i-m}W_{hh}^\top\Big)\;}
$$

這正是 [`05b1`](05b1-backward-propagation.md) 附錄 B 寫的那條連乘積 $\prod_t\partial h_t/\partial h_{t-1}$，只是這裡把它的兩個因子明確拆開了。

**定量估計。** 取矩陣的 2-範數（最大奇異值 $\sigma_{\max}$），由次可乘性：

$$
\big\|G^{h_{i-k}}\big\| \;\le\; \big\|G^{h_i}\big\|\cdot\prod_{m=0}^{k-1}\|D_{i-m}\|_2\cdot\|W_{hh}\|_2^{\,k}
$$

兩個因子分別看：

1. **$\|D_m\|_2 = \max_j\big(1-h_{m,j}^2\big)\le 1$**，因為 $h_{m,j}=\tanh(\cdot)\in(-1,1)$。等號只在 $h_{m,j}=0$ 時成立。**$\tanh$ 保證這個因子永遠不放大，而且狀態越飽和（$|h|$ 越接近 1）壓縮越狠**——$h=0.9$ 時因子只剩 $0.19$。
2. **$\|W_{hh}\|_2^k$ 隨 $k$ 指數變化。** 若 $\sigma_{\max}(W_{hh})<1$，整體指數衰減；若 $>1$，有機會抵銷因子 1，但也可能指數爆炸（**梯度爆炸**，實務上靠 gradient clipping 處理）。

**兩個因子相乘的結果是：穩定訓練的區間非常窄。** 要讓梯度既不消失也不爆炸，需要 $\sigma_{\max}(W_{hh})\cdot\max(1-h^2)\approx1$ 對所有時間步都近似成立——而 $\max(1-h^2)$ 隨資料變動，沒辦法靠設計權重固定。實務上結果幾乎總是衰減：**序列一長，前段就學不到東西。**

以本文 [`10a2`](10a2-seq2seq-forward-example.md) 的 $W_{hh}=0.5I_2$ 為例，$\sigma_{\max}=0.5$，即使 $D$ 全取上界 1：

$$
\text{10 步後：}\ 0.5^{10}\approx 9.8\times10^{-4},
\qquad
\text{20 步後：}\ 0.5^{20}\approx 9.5\times10^{-7}
$$

**三種緩解手段，順著這條公式看就很清楚：**

| 手段 | 動到公式的哪裡 |
|---|---|
| **LSTM／GRU 的閘控** | 記憶單元有一條**加法**路徑 $c_m=f_m\odot c_{m-1}+i_m\odot\tilde c_m$，於是 $\partial c_m/\partial c_{m-1}=\text{diag}(f_m)$——連乘的是遺忘閘 $f_m$ 而**不是權重矩陣**。模型可以學會讓 $f_m\approx1$，把某些維度的梯度**原封不動**傳很多步。 |
| **Bahdanau Attention**（§A3）| 不動這條公式，而是**另開一條路**：$\sum_t\alpha_{t,i}G^{c_t}$ 完全不經過連乘鏈。 |
| **Transformer**（§B）| 直接把遞迴拿掉，連乘鏈根本不存在；任兩個位置之間只隔一層 attention（[`05b1`](05b1-backward-propagation.md) 附錄 B）。|

**注意 Bahdanau attention 沒有修好 encoder 內部。** $h_1$ 從 attention 拿到的梯度是直達的，但 $E_s[x_1]$（來源第一個詞的**詞向量**）仍然只能透過 $G^{a_1}=G^{h_1}\odot(1-h_1^2)$ 這一條路——而 $G^{h_1}$ 裡的時間鏈那股照樣衰減。attention 縮短的是「$h_i$ 到 loss」的距離，不是「$h_1$ 到 $h_{T_s}$」的距離。要連這一段都拆掉，得等 Transformer。

---

# B. Transformer Encoder-Decoder 的反向傳播

前向見 [`10a1`](10a1-seq2seq-forward.md) §B。整個 decoder 的反向與 GPT **形式相同**，只多一個子層；真正的新東西是 §B2 的 cross-attention。

本節反覆用到的三組結果，都在 [`05b1`](05b1-backward-propagation.md) 完整推導過，這裡只列結論並標出處：

| 模組 | 反向公式 | 推導 |
|---|---|---|
| 線性層 $Y=XW$ | $G^{W}=X^\top G^{Y}$，$G^{X}=G^{Y}W^\top$ | [`05b1`](05b1-backward-propagation.md) §4.1 |
| Softmax（逐列）| $G^{S}_{ij}=A_{ij}\big(G^{A}_{ij}-\sum_k A_{ik}G^{A}_{ik}\big)$ | [`05b1`](05b1-backward-propagation.md) §8.4 |
| LayerNorm | $G^{x}=\frac{1}{\sigma}\big(g-\overline{g}-\hat x\odot\overline{g\odot\hat x}\big)$，其中 $g=G^{y}\odot\gamma$、$\overline{\cdot}$ 是沿特徵維的平均；$G^{\gamma}_j=\sum_i G^y_{ij}\hat x_{ij}$、$G^{\beta}_j=\sum_i G^y_{ij}$ | [`05b1`](05b1-backward-propagation.md) §5.7 |

殘差連接的反向也一樣簡單：$Y=X+F(X)$ 時 $G^{X}=G^{Y}+G^{F}$，梯度**兩條路都走**（[`05b1`](05b1-backward-propagation.md) §5.10）。

## B1. 從 Loss 到 Decoder 的 FFN（前向見 [`10a1`](10a1-seq2seq-forward.md) §B5；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §B1）

與 §A1 同一條推導（softmax + CE 合併），只是現在一次算 $T_t$ 列：

$$
G^{\text{logit}}_t = \frac{1}{T_t}\Big(P_t - \text{onehot}(y_t)\Big),\qquad t=1,\dots,T_t
$$

lm_head（前向 $\text{logits}=h^{\text{out}}W_{lm}^\top$）：

$$
G^{W_{lm}} = \big(G^{\text{logit}}\big)^\top h^{\text{out}},\qquad
G^{h^{\text{out}}} = G^{\text{logit}}\,W_{lm}
$$

接著依序穿過：最終 LayerNorm → 殘差 ③ → FFN → LayerNorm ③。這四步與 GPT 的對應段落**逐字相同**，可直接引用 [`05b1`](05b1-backward-propagation.md) §5.7（LayerNorm）與 §6（FFN 的兩個線性層＋ReLU），其中 ReLU 的反向是遮罩：

$$
G^{z} = G^{\text{ReLU}(z)}\odot\mathbb{1}[z>0]
$$

走完得到 $G^{Y_2}$——cross-attention 那個殘差的輸出梯度。**從這裡開始才是新東西。**

> **ReLU 遮罩要用嚴格大於零。** 浮點運算常讓本該是 $0$ 的位置變成 $\pm10^{-16}$，用 `z > 0` 判斷會讓梯度從死區漏出去。實作與手算都應該用一個小容差（例如 `z > 1e-9`）。這個坑在 [`05b2`](05b2-backward-example.md) §4.2 有具體案例。

## B2. Cross-Attention：梯度在這裡分岔（前向見 [`10a1`](10a1-seq2seq-forward.md) §B3；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §B2）

**這是整份文件最重要的一節。**

回顧前向（[`10a1`](10a1-seq2seq-forward.md) §B3）：

$$
Q = \text{LN}_{d2}(Y_1)W_Q^{c},\quad
K = H\,W_K^{c},\quad
V = H\,W_V^{c}
$$
$$
S = \frac{QK^\top}{\sqrt{d_k}},\quad
A = \text{softmax}_{\text{row}}(S),\quad
C = AV,\quad
Y_2 = Y_1 + C
$$

形狀：$Q\in\mathbb{R}^{T_t\times d_k}$、$K,V\in\mathbb{R}^{T_s\times d_k}$、$S,A\in\mathbb{R}^{T_t\times T_s}$、$C\in\mathbb{R}^{T_t\times d_k}$。

由殘差，$G^{C}=G^{Y_2}$。接著倒著走。

**(a) $C=AV$ 分給兩邊。** 這是兩個矩陣的乘積，標準結果：

$$
G^{V} = A^\top G^{C}\ \in\mathbb{R}^{T_s\times d_k},
\qquad
G^{A} = G^{C}V^\top\ \in\mathbb{R}^{T_t\times T_s}
$$

檢查形狀：$A^\top$ 是 $T_s\times T_t$、$G^C$ 是 $T_t\times d_k$，乘起來 $T_s\times d_k$ ✓ 與 $V$ 同形。

**(b) 穿過 softmax**（逐列，沿來源維度）：

$$
G^{S}_{t,i} = A_{t,i}\Big(G^{A}_{t,i} - \sum_{j=1}^{T_s}A_{t,j}G^{A}_{t,j}\Big)
$$

**這裡沒有遮罩要處理。** 對照 §B3 的 masked self-attention：那裡被遮住的位置前向是 $-\infty$、$A=0$，反向 $G^S$ 也必須是 $0$。cross-attention 沒有遮罩，所以 $G^S$ 每一格都可能非零。

**(c) 分數回到 $Q$ 與 $K$——注意這裡出現轉置。** 由 $S=QK^\top/\sqrt{d_k}$：

$$
\boxed{\;
G^{Q} = \frac{1}{\sqrt{d_k}}\,G^{S}K\ \in\mathbb{R}^{T_t\times d_k},
\qquad
G^{K} = \frac{1}{\sqrt{d_k}}\,\big(G^{S}\big)^\top Q\ \in\mathbb{R}^{T_s\times d_k}
\;}
$$

形狀檢查：$G^S$ 是 $T_t\times T_s$、$K$ 是 $T_s\times d_k$ → $G^Q$ 是 $T_t\times d_k$ ✓；$(G^S)^\top$ 是 $T_s\times T_t$、$Q$ 是 $T_t\times d_k$ → $G^K$ 是 $T_s\times d_k$ ✓。

> **這個轉置在 self-attention 裡看不出來。** self-attention 的 $S$ 是方陣、$Q$ 與 $K$ 同形，$(G^S)^\top Q$ 與 $G^SK$ 形狀相同，轉置的存在容易被忽略。cross-attention 因為 $T_t\ne T_s$（一般情形），**形狀直接強迫你把轉置寫對**——寫錯連維度都對不上。

**(d) 三個投影矩陣，以及梯度往哪去。**

$$
G^{W_Q^{c}} = \big(\text{LN}_{d2}\big)^\top G^{Q},\qquad
G^{W_K^{c}} = H^\top G^{K},\qquad
G^{W_V^{c}} = H^\top G^{V}
$$

$$
\underbrace{G^{\text{LN}_{d2}} = G^{Q}\big(W_Q^{c}\big)^\top}_{\textstyle\text{往 \textbf{decoder} 走}}
\qquad
\underbrace{
G^{H}\big|_{(K)} = G^{K}\big(W_K^{c}\big)^\top,\qquad
G^{H}\big|_{(V)} = G^{V}\big(W_V^{c}\big)^\top
}_{\textstyle\text{往 \textbf{encoder} 走}}
$$

**(e) 兩股合流。** $H$ 在前向被用了**兩次**（一次當 $K$、一次當 $V$），所以反向要把兩份加起來：

$$
\boxed{\;G^{H} = G^{H}\big|_{(K)} + G^{H}\big|_{(V)}\;}
$$

這是 §A3 同一個原理的又一次應用：**一個量在前向被用幾次，反向就收幾份梯度。**

### 三個必須看懂的結論

**結論一：這是 encoder 唯一的梯度入口。**

翻回 [`10a1`](10a1-seq2seq-forward.md) §B5 的 pipeline 圖——encoder 與 decoder 之間**只有一條線**，就是 $H$ 進入 cross-attention。因此：

$$
\text{encoder 的每一個參數（}E_s,P_s,\gamma_{e\cdot},\beta_{e\cdot},W_Q^e,W_K^e,W_V^e,W_1^e,W_2^e\text{）}
$$
$$
\text{都只能透過 }G^{H}\text{ 拿到梯度。}
$$

把 $G^H$ 設成零，整個 encoder 就完全不會被更新。這在工程上有直接後果：**凍結 encoder** 只要切斷這一條邊即可；而如果 cross-attention 學到把注意力集中在少數幾個來源位置，$G^H$ 其他列會很小，那些位置的 encoder 表示就幾乎不更新。

**結論二：encoder 的梯度必然比 decoder 小。**

$G^H$ 是走完「輸出層 → 最終 LN → FFN → LN₃ → cross-attention」才產生的，中間穿過了兩個 LayerNorm 與一次 softmax。LayerNorm 的反向會扣掉均值與沿 $\hat x$ 的分量兩個方向（[`05b1`](05b1-backward-propagation.md) §5.9），softmax 又把梯度壓縮到「和為零」的子空間。**每穿一層就削一次**，所以 encoder 參數的梯度量級通常比 decoder 小一個數量級。[`10b2`](10b2-seq2seq-backward-example.md) §D 會量出這個差距。

**結論三：$V$ 那條路的梯度比 $K$ 那條大。**

$G^{H}|_{(V)}$ 只穿過一次矩陣乘法（$G^V=A^\top G^C$）；$G^{H}|_{(K)}$ 要先穿過 softmax 的 Jacobian（步驟 b）才到得了 $K$。softmax 反向會把梯度投影到「和為零」的方向、並乘上 $A_{t,i}\le1$，所以**同一個 $G^C$ 經 $V$ 傳回的梯度通常明顯大於經 $K$ 傳回的**。這與 [`05b2`](05b2-backward-example.md) §5.7 觀察到的「$W_V$ 學得比 $W_Q,W_K$ 快」是同一個現象。

## B3. Decoder 的 Masked Self-Attention（前向見 [`10a1`](10a1-seq2seq-forward.md) §B2；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §B3）

從 §B2 拿到 $G^{\text{LN}_{d2}}$ 之後，穿過 LayerNorm ② 得到 $G^{Y_1}$ 的一部分；再加上殘差直傳的那部分：

$$
G^{Y_1} = G^{Y_2} + G^{\text{LN}_{d2}}\text{ 穿過 LN}_{d2}\text{ 之後的結果}
$$

接著是帶因果遮罩的 self-attention。**公式與 GPT 逐字相同**（[`05b1`](05b1-backward-propagation.md) §8），這裡只強調遮罩帶來的兩個差異：

**(a) 被遮住的位置梯度必為零。** 前向 $S_{ij}=-\infty\Rightarrow A_{ij}=0$。反向的 softmax 公式 $G^{S}_{ij}=A_{ij}(\cdots)$ 前面有一個因子 $A_{ij}$，所以

$$
A_{ij}=0\ \Longrightarrow\ G^{S}_{ij}=0
$$

遮罩不需要在反向另外處理——它自動維持。

**(b) 第 0 列的梯度整列歸零。** 位置 0 只看得到自己，softmax 只有一項存活，故 $A_{0,\cdot}=[1,0,\dots,0]$。代進 softmax 反向：

$$
G^{S}_{0,0} = A_{0,0}\Big(G^{A}_{0,0} - \sum_j A_{0,j}G^{A}_{0,j}\Big)
= 1\cdot\big(G^{A}_{0,0} - 1\cdot G^{A}_{0,0}\big) = 0
$$

其餘 $j>0$ 因 $A_{0,j}=0$ 也是零。**整個第 0 列拿不到任何梯度**——這是 one-hot 分佈下 softmax Jacobian 退化的必然結果（[`05b1`](05b1-backward-propagation.md) §3.2、§8.5），[`05b2`](05b2-backward-example.md) §5.4 有數值印證。

**對照 §B2 的 cross-attention：沒有遮罩，就沒有這個問題**，每一列都拿得到梯度。這是「encoder 端與 cross-attention 端的 softmax 比 decoder self-attention 端更『通暢』」的結構性理由。

穿完 self-attention 與 LayerNorm ① 得到 $G^{Y_0}$，再拆給兩個 embedding：

$$
G^{P_t} = G^{Y_0},\qquad
G^{E_t[\tilde y_t]}\ \mathrel{+}=\ G^{Y_0}_t
$$

$P_t$ 拿到全部（位置編碼每個位置各一列，一一對應）；$E_t$ 則是稀疏更新，沒被查到的列為零。

## B4. Encoder 端的反向（前向見 [`10a1`](10a1-seq2seq-forward.md) §B1；數值見 [`10b2`](10b2-seq2seq-backward-example.md) §B4）

**起點是 §B2 算出的 $G^{H}$**，不是 loss。這是 encoder-decoder 與 GPT 最大的結構差異：GPT 的每一層都直接連到 loss，encoder 則隔了整個 decoder。

從 $G^H$ 開始的每一步都與 GPT 同形（依序：最終 LayerNorm → 殘差 ② → FFN → LayerNorm ② → self-attention → 殘差 ① → LayerNorm ① → embedding），可直接套 [`05b1`](05b1-backward-propagation.md) §5.7、§4.1、§8、§10 的結果。

**唯一的差異仍然是遮罩：encoder 的 self-attention 沒有遮罩，所以**

$$
G^{S^{\text{enc}}}_{ij} = A^{\text{enc}}_{ij}\Big(G^{A}_{ij}-\sum_k A^{\text{enc}}_{ik}G^{A}_{ik}\Big)
$$

**每一格都可能非零**，不像 GPT 的上三角恆為零、第 0 列恆為零。

最後：

$$
G^{P_s} = G^{X^{\text{enc}}},\qquad
G^{E_s[x_i]}\ \mathrel{+}=\ G^{X^{\text{enc}}}_i
$$

## B5. 反向五步總覽

```
① loss → decoder 輸出端
   G^logit = (P − onehot)/T_t → G^{W_lm} → G^{h_out}
   → 最終 LN → 殘差③ → FFN → LN③        ⇒ G^{Y_2}

② cross-attention（分岔點）
   G^C = G^{Y_2}
   ├─ G^V = Aᵀ G^C ────────────┐
   ├─ G^A = G^C Vᵀ → G^S       │
   │   ├─ G^Q = G^S K/√d_k     │  ⇒ G^{LN_d2} → 回 decoder
   │   └─ G^K = (G^S)ᵀ Q/√d_k ─┤
   └───────────────────────────┴─ G^H = G^H|(K) + G^H|(V) → 送進 encoder

③ decoder 的 masked self-attention
   G^{Y_1} = G^{Y_2} + （G^{LN_d2} 穿過 LN②）
   → masked self-attn（遮住的位置 G^S = 0；第 0 列整列為 0）
   → LN① → G^{E_t}, G^{P_t}

④ encoder（起點是 ② 給的 G^H，不是 loss）
   G^H → 最終 LN → 殘差② → FFN → LN②
       → self-attn（無遮罩，G^S 每格都可能非零）
       → 殘差① → LN① → G^{E_s}, G^{P_s}

⑤ optimizer 用 32 個張量的梯度各做一次更新
```

**② 是整張圖的關鍵節點。** 它同時是「decoder 內部梯度流」的一環，也是「encoder 的唯一入口」。

---

# C. 兩代反向對照

| 面向 | RNN Seq2Seq（§A）| Transformer Enc-Dec（§B）|
|---|---|---|
| 目標位置 $t$ → 來源位置 $i$ | attention 直達，$O(1)$ | cross-attention 直達，$O(1)$ |
| 來源位置 $i$ → 來源位置 $1$ | **時間鏈連乘，$O(T_s)$**，必然衰減（§A6）| self-attention 一層，$O(1)$ |
| 目標位置 $t$ → 目標位置 $1$ | **時間鏈連乘，$O(T_t)$** | masked self-attention 一層，$O(1)$ |
| 參數梯度的形式 | 跨時間步**累加**（§A5）| 每層獨立，不累加（跨 batch 才累加）|
| Attention 反向的長度 | 過 $v_a$ → $\tanh'$ → $W_a/U_a$，4 個環節 | 過 softmax → 一次矩陣乘，2 個環節 |
| 有沒有梯度爆炸風險 | 有（$\sigma_{\max}(W_{hh})>1$），需 clipping | 大幅降低（無連乘鏈）|
| 反向能否平行 | 否，必須逐時間步 | 是，整批矩陣運算 |

**第二、三列是 Transformer 的核心優勢。** RNN 就算加了 attention，也只修好了「跨序列」那條路（來源 ↔ 目標），**序列內部**（來源第 1 個詞 ↔ 第 50 個詞）仍然是連乘鏈。Transformer 把序列內部也換成 attention，於是**所有**位置對之間都是 $O(1)$。

**第四列是常被誤解的一點。** Transformer 的參數梯度不是「不累加」，而是**累加的維度不同**：RNN 沿**時間**累加（同一份 $W_{hh}$ 服務 $T_s$ 個時間步），Transformer 沿 **batch 與序列位置**累加（同一份 $W_Q^{(\ell)}$ 服務 $B\times T$ 個 token，但只在第 $\ell$ 層）。層與層之間不共享。

---

## 下一步

- **看數字** → [`10b2-seq2seq-backward-example.md`](10b2-seq2seq-backward-example.md)：把本文每條公式代進 [`10a2`](10a2-seq2seq-forward-example.md) 的數字，兩個模型的全部參數梯度都算出來，並量出「attention 那股 vs 時間鏈那股」與「encoder vs decoder」的梯度差距。
- **回顧 GPT 的反向** → [`05b1-backward-propagation.md`](05b1-backward-propagation.md)：本文 §B 引用的 Softmax、LayerNorm、Linear、Embedding 四組梯度都在那裡完整推導。
- **看實作** → [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb)：用 NumPy 手刻反向傳播並用數值梯度驗證（多頭版本）。
