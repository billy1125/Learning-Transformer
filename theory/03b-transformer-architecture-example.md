# 03b｜Transformer Block 計算案例（完整版）：一條前向流程從頭算到尾

> **這是三階段計算案例的最後一階段（完整版）。** 若你是第一次接觸，建議從輕量的前兩階段循序而上（三份共用同一組 $X$ 與權重，數字完全銜接）：
> 1. [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（簡單版）— 單頭 attention：$X \to \tilde X \to C^{(1)}$
> 2. [`03b2-transformer-example-block.md`](03b2-transformer-example-block.md)（中等版）— 補上第二頭、$W_O$、殘差、FFN，算到 Block 輸出 $Y$
> 3. **03b（本文，完整版）** — 在前兩階段之上，再補縮放數值對照、Positional Encoding 與相對位置旋轉驗證、NB1 重現
>
> 本文自成一體（重述完整流程），可單獨閱讀；若已讀過 03b1／03b2，可略過重複的步驟、直接看 §2.3 縮放對照、§6 Positional Encoding 與 §7 NB1 重現。

> **適合對象：** 已讀完 [`03a-transformer-architecture.md`](03a-transformer-architecture.md)，想用具體數字把整個 Transformer Block 親手算一遍的讀者。
>
> **讀完後你能做什麼：**
> - 用一組 $2\times4$ 的輸入，手算 Pre-LN → Multi-Head Attention → 殘差 → FFN → 殘差的**每一步**
> - 親眼看到除以 $\sqrt{d_k}$ 如何讓 softmax 分佈變平滑（含縮放／不縮放對照）
> - 驗證 $W_O$ 確實「混合重組」各 head 的資訊（而非只是拼接）
> - 手算 LayerNorm 的 mean／var／normalize 三步
> - 用具體數字驗證 Positional Encoding 的「相對位置 = 旋轉」性質
>
> **前置文件：** [`03a-transformer-architecture.md`](03a-transformer-architecture.md)（符號、公式、Pre-LN 定義皆沿用該文）
>
> **對應實作：** [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13「與 03a 對照：逐步數值驗證」可一鍵重現本文所有數字。

---

> 本文是 [`03a`](03a-transformer-architecture.md) §5.6「最小數值例」的**完整版**：§5.6 只把 Multi-Head Attention 單獨作用在輸入上；這裡把它放進一個**完整的 Pre-LN Block**，從頭算到尾。
>
> **與 §5.6 的關鍵差異：** 本文採 Pre-LN，Attention 的輸入是 $\text{LayerNorm}(X)$ 而非原始 $X$，因此 $Q,K,V$ 的數字與 §5.6 不同——這正是 Pre-LN 的特徵，後面會清楚標出。
>
> 所有符號（$X, S, E, A, O, W_O, \tilde X, \ldots$）定義見 [`03a` 的符號表](03a-transformer-architecture.md#符號表)。數字一律四捨五入到小數第 4 位，最末位可能有 $\pm 0.0001$ 的進位誤差。

---

## 目錄

0. 設定：一次給齊所有矩陣
1. Pre-LN 第一步：$\tilde X = \text{LayerNorm}(X)$
2. 單頭 Scaled Dot-Product（Head 1 逐步）＋ 縮放對照
3. Head 2、拼接與 $W_O$ 混合
4. 第一個殘差連接
5. 第二子層：LayerNorm → FFN → 第二個殘差
6. Positional Encoding 數值與相對位置驗證
7. 用 NB1 重現本文數字

---

## 0. 設定：一次給齊所有矩陣

超參數：序列長度 $T=2$、模型維度 $d=4$、head 數 $H=2$、每頭維度 $d_k=d_v=2$、FFN 中間維度 $d_{ff}=8$。

**輸入序列**（每列一個 token，沿用 §5.6）：

$$
X = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

**各 head 的投影矩陣**（Head 1 取前 2 維、Head 2 取後 2 維；本例令 $W_Q=W_K=W_V$ 以聚焦流程）：

$$
W_Q^{(1)}=W_K^{(1)}=W_V^{(1)}=\begin{bmatrix}1&0\\0&1\\0&0\\0&0\end{bmatrix},\qquad
W_Q^{(2)}=W_K^{(2)}=W_V^{(2)}=\begin{bmatrix}0&0\\0&0\\1&0\\0&1\end{bmatrix}\;\in\mathbb{R}^{4\times2}
$$

**輸出投影**（刻意設計成「前 2 維取兩頭的和、後 2 維取兩頭的差」，方便驗證混合效果）：

$$
W_O = \frac{1}{2}\begin{bmatrix}1&0&1&0\\0&1&0&1\\1&0&-1&0\\0&1&0&-1\end{bmatrix}\in\mathbb{R}^{4\times4}
$$

**兩個 LayerNorm** 的可學習參數取初始值 $\gamma=[1,1,1,1]$、$\beta=[0,0,0,0]$（即先只做標準化，方便看清效果）。

**FFN 權重**（$b_1=b_2=0$）：

$$
W_1=\frac12\begin{bmatrix}
1&-1&0&1&1&0&-1&0\\
0&1&1&-1&0&1&0&-1\\
1&0&-1&0&1&-1&0&1\\
-1&1&0&1&0&0&1&-1
\end{bmatrix}\in\mathbb{R}^{4\times8},\qquad
W_2=\frac12\begin{bmatrix}
1&0&1&-1\\0&1&-1&0\\1&-1&0&1\\-1&0&1&0\\
0&1&0&1\\1&0&-1&0\\0&-1&0&1\\-1&0&1&0
\end{bmatrix}\in\mathbb{R}^{8\times4}
$$

> 這些矩陣的數值是教學用的人造值，唯一要求是「內部自洽、可被 NB1 重現」。重點在流程與 shape，不在權重本身的語意。

---

## 1. Pre-LN 第一步：$\tilde X = \text{LayerNorm}(X)$

Pre-LN 規定：**先正規化、再進子層**（[`03a` §6.1](03a-transformer-architecture.md)）。LayerNorm 對**每一列**（每個 token 的 $d$ 維）獨立做標準化。

以第 1 列 $x_1=[1,0,0,1]$ 為例，三步：

$$
\mu_1=\frac{1+0+0+1}{4}=0.5,\qquad
\sigma_1^2=\frac{(0.5)^2+(0.5)^2+(0.5)^2+(0.5)^2}{4}=0.25,\qquad
\sigma_1=0.5
$$

$$
\hat x_1=\frac{x_1-\mu_1}{\sigma_1}=\frac{[0.5,\,-0.5,\,-0.5,\,0.5]}{0.5}=[1,-1,-1,1]
$$

因為 $\gamma=1,\beta=0$，$\tilde x_1=\gamma\odot\hat x_1+\beta=\hat x_1$。第 2 列同理（$\mu_2=0.5,\sigma_2=0.5$）：

$$
\tilde X = \text{LayerNorm}(X)=\begin{bmatrix} 1 & -1 & -1 & 1 \\ -1 & 1 & 1 & -1 \end{bmatrix}
$$

> **這就是與 §5.6 的分岔點。** §5.6 直接拿 $X$ 算 attention；Pre-LN 拿的是 $\tilde X$。下面所有 $Q,K,V$ 都由 $\tilde X$ 算出。

---

## 2. 單頭 Scaled Dot-Product（Head 1 逐步）

### 2.1 投影

Head 1 取 $\tilde X$ 的前 2 維：

$$
Q^{(1)}=K^{(1)}=V^{(1)}=\tilde X\,W_Q^{(1)}=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

（驗算第 1 列：$\tilde x_1=[1,-1,-1,1]$ 乘 $W_Q^{(1)}$ 只留前 2 維 → $[1,-1]$ ✓）

### 2.2 原始分數 $S$

$$
S^{(1)}=Q^{(1)}\big(K^{(1)}\big)^\top
$$

逐格手算（$\big(K^{(1)}\big)^\top$ 的兩列就是 $K^{(1)}$ 的兩列）：

$$
S^{(1)}_{11}=[1,-1]\cdot[1,-1]=1+1=2,\qquad
S^{(1)}_{12}=[1,-1]\cdot[-1,1]=-1-1=-2
$$

$$
S^{(1)}=\begin{bmatrix}2&-2\\-2&2\end{bmatrix}
$$

### 2.3 縮放：為什麼要除以 $\sqrt{d_k}$（數值對照）

這裡把 [`03a` §3.2](03a-transformer-architecture.md) 的論點**用數字看一次**。對第 1 列的分數 $[2,-2]$：

| | 分數 | softmax 結果 |
|---|---|---|
| **不縮放** $S$ | $[2,\,-2]$ | $\dfrac{e^2}{e^2+e^{-2}}=\dfrac{7.389}{7.524}=0.9820,\ \ 0.0180$ |
| **縮放** $E=S/\sqrt2$ | $[1.4142,\,-1.4142]$ | $\dfrac{e^{1.4142}}{e^{1.4142}+e^{-1.4142}}=\dfrac{4.113}{4.356}=0.9442,\ \ 0.0558$ |

不縮放時權重 $0.982$ 更接近 one-hot（更尖銳），梯度更容易飽和；除以 $\sqrt{d_k}=\sqrt2$ 後分佈平緩成 $0.944$。$d_k$ 越大差距越明顯——這裡 $d_k$ 只有 2 就已可見。

$$
E^{(1)}=\frac{S^{(1)}}{\sqrt2}=\begin{bmatrix}1.4142&-1.4142\\-1.4142&1.4142\end{bmatrix}
$$

### 2.4 注意力權重 $A$ 與 context $C$

$$
A^{(1)}=\text{softmax}_\text{row}\!\big(E^{(1)}\big)=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix},\qquad \textstyle\sum_j A^{(1)}_{ij}=1\ \checkmark
$$

$$
C^{(1)}=A^{(1)}V^{(1)},\quad V^{(1)}=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

第 1 列：$C^{(1)}_1=0.9442\,[1,-1]+0.0558\,[-1,1]=[0.9442-0.0558,\ -0.9442+0.0558]=[0.8884,\,-0.8884]$。

$$
C^{(1)}=\begin{bmatrix}0.8884&-0.8884\\-0.8884&0.8884\end{bmatrix}\in\mathbb{R}^{2\times2}
$$

---

## 3. Head 2、拼接與 $W_O$ 混合

### 3.1 Head 2

取 $\tilde X$ 的後 2 維（注意兩列順序與 Head 1 相反）：

$$
Q^{(2)}=K^{(2)}=V^{(2)}=\tilde X\,W_Q^{(2)}=\begin{bmatrix}-1&1\\1&-1\end{bmatrix}
$$

分數 $S^{(2)}=Q^{(2)}(K^{(2)})^\top=\begin{bmatrix}2&-2\\-2&2\end{bmatrix}$，與 Head 1 **相同**，所以 $A^{(2)}=A^{(1)}$。但 Value 不同：

$$
C^{(2)}=A^{(2)}V^{(2)}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}\begin{bmatrix}-1&1\\1&-1\end{bmatrix}=\begin{bmatrix}-0.8884&0.8884\\0.8884&-0.8884\end{bmatrix}
$$

> 「該看誰」相同（$A$ 一樣），「看到什麼」不同（$V$ 不同）→ 輸出不同。這正是 [`03a` §5.6](03a-transformer-architecture.md) 強調的多頭價值。

### 3.2 拼接

$$
\text{Concat}\big(C^{(1)},C^{(2)}\big)=\begin{bmatrix}0.8884&-0.8884&-0.8884&0.8884\\-0.8884&0.8884&0.8884&-0.8884\end{bmatrix}\in\mathbb{R}^{2\times4}
$$

### 3.3 $W_O$ 混合：拼接 ≠ 終點

$O=\text{Concat}\cdot W_O$。依 $W_O$ 的設計，輸出四維分別是：

$$
O_{:,0}=\tfrac12\big(C^{(1)}_{:,0}+C^{(2)}_{:,0}\big),\quad
O_{:,1}=\tfrac12\big(C^{(1)}_{:,1}+C^{(2)}_{:,1}\big),\quad
O_{:,2}=\tfrac12\big(C^{(1)}_{:,0}-C^{(2)}_{:,0}\big),\quad
O_{:,3}=\tfrac12\big(C^{(1)}_{:,1}-C^{(2)}_{:,1}\big)
$$

第 1 列代入（$C^{(1)}_1=[0.8884,-0.8884]$、$C^{(2)}_1=[-0.8884,0.8884]$）：

- 前兩維（兩頭之**和**）：$\tfrac12(0.8884+(-0.8884))=0$、$\tfrac12(-0.8884+0.8884)=0$
- 後兩維（兩頭之**差**）：$\tfrac12(0.8884-(-0.8884))=0.8884$、$\tfrac12(-0.8884-0.8884)=-0.8884$

$$
O=\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}\in\mathbb{R}^{2\times4}
$$

> **看到 $W_O$ 在做事了。** 本例兩頭恰好方向相反，「和」通道被抵消為 0、「差」通道被放大——若沒有 $W_O$（即拼接後直接輸出），這個跨頭的相消／相長根本不會發生。$W_O$ 讓模型能學習如何融合各頭，呼應 [`03a` §5.7](03a-transformer-architecture.md)。

---

## 4. 第一個殘差連接

Pre-LN 的殘差加在**原始** $X$ 上（不是 $\tilde X$）：

$$
Z'=X+O=\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix}+\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}
=\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}
$$

維度仍是 $T\times d=2\times4$。

---

## 5. 第二子層：LayerNorm → FFN → 第二個殘差

### 5.1 $\tilde Z=\text{LayerNorm}(Z')$

對第 1 列 $z'_1=[1,0,0.8884,0.1116]$：

$$
\mu=\frac{1+0+0.8884+0.1116}{4}=0.5,\qquad
\sigma^2=\frac{0.5^2+0.5^2+0.3884^2+0.3884^2}{4}=0.2004,\qquad \sigma=0.4477
$$

$$
\tilde z_1=\frac{[0.5,\,-0.5,\,0.3884,\,-0.3884]}{0.4477}=[1.1169,\,-1.1169,\,0.8675,\,-0.8675]
$$

$$
\tilde Z=\begin{bmatrix}1.1169&-1.1169&0.8675&-0.8675\\-1.1169&1.1169&-0.8675&0.8675\end{bmatrix}
$$

### 5.2 FFN：$F=\text{ReLU}(\tilde Z W_1)\,W_2$

**Step 1：升維 $\tilde Z W_1$（$2\times4 \cdot 4\times8 = 2\times8$）。** 示範第 1 列、第 1 個輸出（$W_1$ 第 0 行為 $\tfrac12[1,0,1,-1]=[0.5,0,0.5,-0.5]$）：

$$
(\tilde Z W_1)_{1,0}=1.1169(0.5)+(-1.1169)(0)+0.8675(0.5)+(-0.8675)(-0.5)=0.5585+0.4338+0.4338=1.4260
$$

整列結果：

$$
\tilde Z W_1=\begin{bmatrix}
1.4260&-1.5506&-0.9922&0.6831&0.9922&-0.9922&-0.9922&1.4260\\
-1.4260&1.5506&0.9922&-0.6831&-0.9922&0.9922&0.9922&-1.4260
\end{bmatrix}
$$

**Step 2：ReLU（負值歸零）。** 這是 FFN 唯一的非線性來源：

$$
\text{ReLU}(\tilde Z W_1)=\begin{bmatrix}
1.4260&0&0&0.6831&0.9922&0&0&1.4260\\
0&1.5506&0.9922&0&0&0.9922&0.9922&0
\end{bmatrix}
$$

**Step 3：降維 $\cdot\,W_2$（$2\times8 \cdot 8\times4 = 2\times4$）。**

$$
F=\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}
$$

> 若沒有 ReLU，$W_1W_2$ 會塌縮成單一 $4\times4$ 線性映射，中間的 8 維「展開空間」毫無意義（[`03a` §6.3](03a-transformer-architecture.md)）。本例 $d_{ff}=8=2d$ 是為了手算方便；實務通常 $d_{ff}=4d$。

### 5.3 第二個殘差 → Block 輸出

$$
Y=Z'+F=\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}+\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}
$$

$$
\boxed{\,Y=\begin{bmatrix}0.6585&0.4961&2.6559&-0.1053\\0.9922&0.7831&-1.1598&1.8806\end{bmatrix}\in\mathbb{R}^{2\times4}\,}
$$

輸出 shape 與輸入 $X$ 相同（$2\times4$），因此可以直接把 $Y$ 餵進下一個 Block——這就是 $N$ 層 Transformer 能堆疊的原因（[`03a` §6.5](03a-transformer-architecture.md)）。

---

## 6. Positional Encoding 數值與相對位置驗證

### 6.1 編碼表（$d=4$，位置 $0\sim3$）

維度索引 $m\in\{0,1\}$，角頻率 $\omega_m=1/10000^{2m/d}$：$\omega_0=1$、$\omega_1=1/10000^{0.5}=0.01$。

依 $p_{i,2m}=\sin(\omega_m i)$、$p_{i,2m+1}=\cos(\omega_m i)$（[`03a` §7.2](03a-transformer-architecture.md)）：

| 位置 $i$ | $p_{i,0}=\sin(i)$ | $p_{i,1}=\cos(i)$ | $p_{i,2}=\sin(0.01i)$ | $p_{i,3}=\cos(0.01i)$ |
|---|---|---|---|---|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

低維（$m=0$）隨位置快速振盪（捕捉局部），高維（$m=1$）變化極慢（捕捉全局）。

### 6.2 「相對位置 = 旋轉」數值驗證

[`03a` §7.3](03a-transformer-architecture.md) 證明：位置平移 $\delta$ 等價於對每個 2D 子平面旋轉 $\omega_m\delta$。取 $m=0$（$\omega_0=1$）、$i=1$、$\delta=2$，旋轉矩陣

$$
R=\begin{bmatrix}\cos(\omega_0\delta)&\sin(\omega_0\delta)\\-\sin(\omega_0\delta)&\cos(\omega_0\delta)\end{bmatrix}
=\begin{bmatrix}\cos2&\sin2\\-\sin2&\cos2\end{bmatrix}
=\begin{bmatrix}-0.4161&0.9093\\-0.9093&-0.4161\end{bmatrix}
$$

驗證 $\begin{bmatrix}p_{3,0}\\p_{3,1}\end{bmatrix}\overset{?}{=}R\begin{bmatrix}p_{1,0}\\p_{1,1}\end{bmatrix}$：

$$
R\begin{bmatrix}0.8415\\0.5403\end{bmatrix}
=\begin{bmatrix}-0.4161(0.8415)+0.9093(0.5403)\\-0.9093(0.8415)-0.4161(0.5403)\end{bmatrix}
=\begin{bmatrix}0.1411\\-0.9900\end{bmatrix}
$$

與表中 $i=3$ 的 $[p_{3,0},p_{3,1}]=[0.1411,-0.9900]$ **完全吻合** ✓。模型因此能用固定的線性變換表達「相隔 $\delta$ 格」，無論 $i$ 在哪裡——這正是 RoPE 的前身（[`06` §3](06-modern-transformer-variants.md)）。

---

## 7. 用 NB1 重現本文數字

本文每個數字都可在 [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13「與 03a 對照：逐步數值驗證」一鍵跑出。該 cell 是**獨立自足**的 NumPy 程式（不依賴 NB1 其他類別），直接硬編碼本文的 $X$ 與所有權重，逐步印出 $\tilde X, S, E, A, C, O, Z', \tilde Z, F, Y$ 與 PE 表，方便你對著本文逐格核對。

---

## 下一步

- **回主線理論：** [`04-gpt-decoder-only.md`](04-gpt-decoder-only.md) — 在本文的 attention 之上加 Causal Masking，走向 GPT。
- **往實作走：** [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) — 用 NumPy 從零實作；§13 即本文的可執行版。
- **往反向傳播走：** [`05-backpropagation.md`](05-backpropagation.md) — 有了前向數字，接著手推每個元件的梯度。
