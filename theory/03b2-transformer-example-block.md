# 03b2｜計算案例（中等版）：補齊多頭、$W_O$、殘差與 FFN，算到 Block 輸出

> **適合對象：** 已讀完 [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（單頭 attention），想把剩下的元件接上、把**整個 Pre-LN Block** 從頭算到尾的讀者。
>
> **讀完後你能做什麼：**
> - 算出第二個 head，並驗證 $W_O$ 確實「混合重組」各 head 的資訊（而非只是拼接）
> - 手算第一個殘差連接、第二子層的 LayerNorm 與 FFN（升維 → ReLU → 降維三步）
> - 一路算到完整的 Block 輸出 $Y$，並理解為什麼它能直接餵進下一個 Block
>
> **前置文件：** [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（沿用其 $\tilde X$ 與 $C^{(1)}$）；符號與公式見 [`03a-transformer-architecture.md`](03a-transformer-architecture.md)。
>
> **這是三階段計算案例的第二階段：**
> 1. [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（簡單版）— 單頭 attention：$X \to \tilde X \to C^{(1)}$
> 2. **03b2（本文，中等版）** — 補上第二頭、$W_O$、殘差、FFN，算到 Block 輸出 $Y$
> 3. [`03b-transformer-architecture-example.md`](03b-transformer-architecture-example.md)（完整版）— 再補縮放數值對照、Positional Encoding 與相對位置旋轉驗證、NB1 重現
>
> 三份文件**共用同一組 $X$ 與權重**，數字完全銜接。數字一律四捨五入到小數第 4 位，最末位可能有 $\pm 0.0001$ 的進位誤差。

---

## 目錄

0. 設定：本階段新增的矩陣
1. 接續 03b1：已有的 $\tilde X$ 與 $C^{(1)}$
2. Head 2、拼接與 $W_O$ 混合
3. 第一個殘差連接
4. 第二子層：LayerNorm → FFN → 第二個殘差
5. 小結與下一階段

---

## 0. 設定：本階段新增的矩陣

超參數與 [`03b1`](03b1-transformer-example-basic.md) 相同：$T=2$、$d=4$、$H=2$、$d_k=d_v=2$、FFN 中間維度 $d_{ff}=8$。本階段補上 Head 2、$W_O$ 與 FFN 權重。

**Head 2 的投影矩陣**（取後 2 維；同樣令 $W_Q=W_K=W_V$）：

$$
W_Q^{(2)}=W_K^{(2)}=W_V^{(2)}=\begin{bmatrix}0&0\\0&0\\1&0\\0&1\end{bmatrix}\;\in\mathbb{R}^{4\times2}
$$

**輸出投影**（刻意設計成「前 2 維取兩頭的和、後 2 維取兩頭的差」，方便驗證混合效果）：

$$
W_O = \frac{1}{2}\begin{bmatrix}1&0&1&0\\0&1&0&1\\1&0&-1&0\\0&1&0&-1\end{bmatrix}\in\mathbb{R}^{4\times4}
$$

**第二個 LayerNorm** 同樣取 $\gamma=[1,1,1,1]$、$\beta=[0,0,0,0]$。

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

---

## 1. 接續 03b1：已有的 $\tilde X$ 與 $C^{(1)}$

[`03b1`](03b1-transformer-example-basic.md) 已算出 Pre-LN 後的輸入與 Head 1 的 context：

$$
\tilde X=\begin{bmatrix} 1 & -1 & -1 & 1 \\ -1 & 1 & 1 & -1 \end{bmatrix},\qquad
C^{(1)}=\begin{bmatrix}0.8884&-0.8884\\-0.8884&0.8884\end{bmatrix}
$$

其中 Head 1 的注意力權重 $A^{(1)}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}$。下面直接沿用。

---

## 2. Head 2、拼接與 $W_O$ 混合

### 2.1 Head 2

取 $\tilde X$ 的後 2 維（注意兩列順序與 Head 1 相反）：

$$
Q^{(2)}=K^{(2)}=V^{(2)}=\tilde X\,W_Q^{(2)}=\begin{bmatrix}-1&1\\1&-1\end{bmatrix}
$$

分數 $S^{(2)}=Q^{(2)}(K^{(2)})^\top=\begin{bmatrix}2&-2\\-2&2\end{bmatrix}$，與 Head 1 **相同**，所以 $A^{(2)}=A^{(1)}$。但 Value 不同：

$$
C^{(2)}=A^{(2)}V^{(2)}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}\begin{bmatrix}-1&1\\1&-1\end{bmatrix}=\begin{bmatrix}-0.8884&0.8884\\0.8884&-0.8884\end{bmatrix}
$$

> 「該看誰」相同（$A$ 一樣），「看到什麼」不同（$V$ 不同）→ 輸出不同。這正是 [`03a` §5.6](03a-transformer-architecture.md) 強調的多頭價值。

### 2.2 拼接

$$
\text{Concat}\big(C^{(1)},C^{(2)}\big)=\begin{bmatrix}0.8884&-0.8884&-0.8884&0.8884\\-0.8884&0.8884&0.8884&-0.8884\end{bmatrix}\in\mathbb{R}^{2\times4}
$$

### 2.3 $W_O$ 混合：拼接 ≠ 終點

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

## 3. 第一個殘差連接

Pre-LN 的殘差加在**原始** $X$ 上（不是 $\tilde X$）：

$$
Z'=X+O=\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix}+\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}
=\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}
$$

維度仍是 $T\times d=2\times4$。

---

## 4. 第二子層：LayerNorm → FFN → 第二個殘差

### 4.1 $\tilde Z=\text{LayerNorm}(Z')$

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

### 4.2 FFN：$F=\text{ReLU}(\tilde Z W_1)\,W_2$

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

### 4.3 第二個殘差 → Block 輸出

$$
Y=Z'+F=\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}+\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}
$$

$$
\boxed{\,Y=\begin{bmatrix}0.6585&0.4961&2.6559&-0.1053\\0.9922&0.7831&-1.1598&1.8806\end{bmatrix}\in\mathbb{R}^{2\times4}\,}
$$

輸出 shape 與輸入 $X$ 相同（$2\times4$），因此可以直接把 $Y$ 餵進下一個 Block——這就是 $N$ 層 Transformer 能堆疊的原因（[`03a` §6.5](03a-transformer-architecture.md)）。

---

## 5. 小結與下一階段

你已經把**整個 Pre-LN Block** 從 $X$ 算到 $Y$：

$$
X \;\to\; \underbrace{\tilde X \to \text{MHA}(C^{(1)},C^{(2)}) \to O}_{\text{第一子層}} \;\to\; Z'=X+O \;\to\; \underbrace{\tilde Z \to \text{FFN} \to F}_{\text{第二子層}} \;\to\; Y=Z'+F
$$

**下一階段** [`03b-transformer-architecture-example.md`](03b-transformer-architecture-example.md)（完整版）會在同一組數字上再補：

- **縮放的數值對照**：除以 $\sqrt{d_k}$ 與不除，softmax 分佈差多少（一張對照表）
- **Positional Encoding** 的編碼表，以及「相對位置 = 旋轉」的數值驗證
- 如何用 [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13 一鍵重現本文所有數字
