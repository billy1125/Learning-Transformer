# 03b1｜計算案例（簡單版）：單頭 Attention 從輸入算到 context

> **適合對象：** 已讀完 [`03a-transformer-architecture.md`](03a-transformer-architecture.md)，第一次想用具體數字把 attention 「算一次」的讀者。本文只走**最短路徑**，刻意不碰多頭、$W_O$、FFN，先把單頭的資料流摸熟。
>
> **讀完後你能做什麼：**
> - 用一組 $2\times4$ 的輸入，手算 Pre-LN 的 $\tilde X=\text{LayerNorm}(X)$（mean／var／normalize 三步）
> - 對單一個 head 走完 投影 → 分數 $S$ → softmax → 加權 $V$，得到 context
> - 看懂整條前向流程的**起點到 attention 輸出**長什麼樣
>
> **前置文件：** [`03a-transformer-architecture.md`](03a-transformer-architecture.md)（符號、公式、Pre-LN 定義皆沿用該文）
>
> **這是三階段計算案例的第一階段：**
> 1. **03b1（本文，簡單版）** — 單頭 attention：$X \to \tilde X \to C^{(1)}$
> 2. [`03b2-transformer-example-block.md`](03b2-transformer-example-block.md)（中等版）— 補上第二頭、$W_O$、殘差、FFN，算到完整 Block 輸出 $Y$
> 3. [`03b3-transformer-architecture-example.md`](03b3-transformer-architecture-example.md)（完整版）— 再補縮放數值對照、Positional Encoding 與相對位置旋轉驗證、NB1 重現
>
> 三份文件**共用同一組 $X$ 與權重**，數字完全銜接：本文算出的 $\tilde X$、$C^{(1)}$ 會被下一階段直接沿用，不需重算。數字一律四捨五入到小數第 4 位，最末位可能有 $\pm 0.0001$ 的進位誤差。

---

## 目錄

0. 設定：本階段用到的矩陣
1. Pre-LN：$\tilde X = \text{LayerNorm}(X)$
2. 單頭 Scaled Dot-Product：$\tilde X \to C^{(1)}$
3. 動手算：換 Head 2 試試看（練習題＋解答）
4. 小結與下一階段

---

## 0. 設定：本階段用到的矩陣

超參數：序列長度 $T=2$、模型維度 $d=4$、head 數 $H=2$、每頭維度 $d_k=d_v=2$。本階段正文只用 **Head 1** 走完整條流程；**Head 2** 會在 §3 以**練習題**形式登場（其投影矩陣在該節就地給出，讓你自己算一遍）。把各頭拼接、混合的 $W_O$ 與 FFN 仍留到 [`03b2`](03b2-transformer-example-block.md) 再登場。

**輸入序列**（每列一個 token，沿用 [`03a` §5.6](03a-transformer-architecture.md)）：

$$
X = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

**Head 1 的投影矩陣**（取前 2 維；本例令 $W_Q=W_K=W_V$ 以聚焦流程）：

$$
W_Q^{(1)}=W_K^{(1)}=W_V^{(1)}=\begin{bmatrix}1&0\\0&1\\0&0\\0&0\end{bmatrix}\;\in\mathbb{R}^{4\times2}
$$

**LayerNorm** 的可學習參數取初始值 $\gamma=[1,1,1,1]$、$\beta=[0,0,0,0]$（即先只做標準化，方便看清效果）。

> 這些矩陣的數值是教學用的人造值，唯一要求是「內部自洽、可被 NB1 重現」。重點在流程與 shape，不在權重本身的語意。

---

## 1. Pre-LN：$\tilde X = \text{LayerNorm}(X)$

Pre-LN 規定：**先正規化、再進子層**（[`03a` §6.2](03a-transformer-architecture.md)）。LayerNorm 對**每一列**（每個 token 的 $d$ 維）獨立做標準化。

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

> **記住這個 $\tilde X$。** Pre-LN 的 attention 吃的是 $\tilde X$ 而不是原始 $X$，下面的 $Q,K,V$ 全由 $\tilde X$ 算出。（與直接拿 $X$ 算 attention 的差異，見 [`03a` §5.6](03a-transformer-architecture.md)。）

---

## 2. 單頭 Scaled Dot-Product：$\tilde X \to C^{(1)}$

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

### 2.3 縮放與 softmax

接著除以 $\sqrt{d_k}=\sqrt2$ 再取 softmax。

> **為什麼要除以 $\sqrt{d_k}$？** 一句話：避免分數過大讓 softmax 太「尖」、梯度飽和。本階段先照做，完整的數值對照（縮放／不縮放會差多少）見 [`03b3` §2.3](03b3-transformer-architecture-example.md)。

$$
E^{(1)}=\frac{S^{(1)}}{\sqrt2}=\begin{bmatrix}1.4142&-1.4142\\-1.4142&1.4142\end{bmatrix}
$$

對每一列取 softmax：

$$
A^{(1)}=\text{softmax}_\text{row}\!\big(E^{(1)}\big)=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix},\qquad \textstyle\sum_j A^{(1)}_{ij}=1\ \checkmark
$$

（第 1 列驗算：$\dfrac{e^{1.4142}}{e^{1.4142}+e^{-1.4142}}=\dfrac{4.113}{4.113+0.243}=0.9442$ ✓）

### 2.4 加權 $V$ 得 context

$$
C^{(1)}=A^{(1)}V^{(1)},\quad V^{(1)}=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

第 1 列：$C^{(1)}_1=0.9442\,[1,-1]+0.0558\,[-1,1]=[0.9442-0.0558,\ -0.9442+0.0558]=[0.8884,\,-0.8884]$。

$$
\boxed{\,C^{(1)}=\begin{bmatrix}0.8884&-0.8884\\-0.8884&0.8884\end{bmatrix}\in\mathbb{R}^{2\times2}\,}
$$

這就是 Head 1 對每個 token 算出的 context：它是「依注意力權重 $A^{(1)}$ 對所有 token 的 $V^{(1)}$ 做加權平均」的結果。

---

## 3. 動手算：換 Head 2 試試看

「多頭」的意思，就是**同一條流程平行跑好幾遍**，每個 head 用不同的投影權重，因此看到資料的不同面向。你已經把 Head 1 算完了——把 §2 的四步（投影 → $S$ → 縮放 softmax → $AV$）原封不動再走一次，就能算出 Head 2。先自己動手，再對答案。

### 3.1 題目

Head 2 改取 $\tilde X$ 的**後 2 維**（Head 1 取的是前 2 維），投影矩陣為：

$$
W_Q^{(2)}=W_K^{(2)}=W_V^{(2)}=\begin{bmatrix}0&0\\0&0\\1&0\\0&1\end{bmatrix}\;\in\mathbb{R}^{4\times2}
$$

**請沿用 §1 的 $\tilde X=\begin{bmatrix}1&-1&-1&1\\-1&1&1&-1\end{bmatrix}$**，照 §2.1～§2.4 算出 Head 2 的 context $C^{(2)}$。提示：投影後別忘了照樣除以 $\sqrt{d_k}=\sqrt2$ 再取 softmax。

> 動手前先想一秒：Head 2 的投影只是把 $\tilde X$ 的**後 2 維**挑出來，你不必重算 LayerNorm。

### 3.2 解答

**第一步：投影。** 取 $\tilde X$ 的後 2 維（注意兩列的順序與 Head 1 相反）：

$$
Q^{(2)}=K^{(2)}=V^{(2)}=\tilde X\,W_Q^{(2)}=\begin{bmatrix}-1&1\\1&-1\end{bmatrix}
$$

（驗算第 1 列：$\tilde x_1=[1,-1,-1,1]$ 乘 $W_Q^{(2)}$ 只留**後 2 維** → $[-1,1]$ ✓）

**第二步：原始分數 $S^{(2)}=Q^{(2)}(K^{(2)})^\top$。** 逐格手算：

$$
S^{(2)}_{11}=[-1,1]\cdot[-1,1]=1+1=2,\qquad
S^{(2)}_{12}=[-1,1]\cdot[1,-1]=-1-1=-2
$$

$$
S^{(2)}=\begin{bmatrix}2&-2\\-2&2\end{bmatrix}\quad(\textbf{與 Head 1 的 }S^{(1)}\textbf{ 完全相同})
$$

> **為什麼兩頭的分數會一模一樣？** 關鍵在這組 $\tilde X$ 的**後 2 維恰好是前 2 維的相反數**：
> $$
> \underbrace{\begin{bmatrix}1&-1\\-1&1\end{bmatrix}}_{\text{前 2 維（Head 1）}}\quad\xrightarrow{\ \times(-1)\ }\quad\underbrace{\begin{bmatrix}-1&1\\1&-1\end{bmatrix}}_{\text{後 2 維（Head 2）}}
> $$
> 因此 $Q^{(2)}=-Q^{(1)}$、$K^{(2)}=-K^{(1)}$。代進分數時，兩個負號在乘積裡**相乘抵消**：
> $$
> S^{(2)}=Q^{(2)}\big(K^{(2)}\big)^\top=(-Q^{(1)})(-K^{(1)})^\top=Q^{(1)}\big(K^{(1)}\big)^\top=S^{(1)}
> $$
> **這是本例刻意挑的數字造成的巧合，不是通則。** 一般情況下不同 head 的 $W_Q,W_K$ 不會讓 $Q,K$ 差一個整體符號，分數 $S$（與注意力 $A$）通常各不相同——各頭因此能關注不同的 token。本例特意讓 $A$ 相同、只讓 $V$ 不同，是為了把「該看誰相同、看到什麼不同」這件事孤立出來看清楚。

**第三步：縮放與 softmax。** 因為 $S^{(2)}=S^{(1)}$，除以 $\sqrt2$ 再取 softmax 的結果也一樣：

$$
A^{(2)}=\text{softmax}_\text{row}\!\Big(\tfrac{S^{(2)}}{\sqrt2}\Big)=A^{(1)}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}
$$

**第四步：加權 $V$ 得 context $C^{(2)}=A^{(2)}V^{(2)}$。** 第 1 列：

$$
C^{(2)}_1=0.9442\,[-1,1]+0.0558\,[1,-1]=[-0.9442+0.0558,\ 0.9442-0.0558]=[-0.8884,\,0.8884]
$$

$$
\boxed{\,C^{(2)}=\begin{bmatrix}-0.8884&0.8884\\0.8884&-0.8884\end{bmatrix}\in\mathbb{R}^{2\times2}\,}
$$

### 3.3 一個關鍵觀察

兩個 head 的 $S$（因而 $A$）**完全相同**——也就是「**該看誰**」一樣（兩個 token 都主要關注自己）；但因為 $V$ 不同（一個取前 2 維、一個取後 2 維），「**看到什麼**」不同，於是 $C^{(2)}\neq C^{(1)}$。這正是多頭的價值：用不同的投影，從同一份輸入抽出不同面向（見 [`03a` §5.6](03a-transformer-architecture.md)）。

至於這兩個 head 接下來**如何拼接、再用 $W_O$ 混合重組**成一條輸出，就是下一階段 [`03b2`](03b2-transformer-example-block.md) 的內容——你在這裡算出的 $C^{(2)}$ 會被它直接沿用。

---

## 4. 小結與下一階段

你剛剛走完了一條最短的 attention 流程：

$$
X \;\xrightarrow{\text{LayerNorm}}\; \tilde X \;\xrightarrow{\text{投影}}\; Q,K,V \;\xrightarrow{QK^\top/\sqrt{d_k}}\; \text{分數} \;\xrightarrow{\text{softmax}}\; A \;\xrightarrow{AV}\; C^{(1)}
$$

但這還不是一個完整的 Transformer Block——你雖然在 §3 練習中算出了 $C^{(2)}$，但兩個 head 還沒被**拼接、混合**成一條輸出，也還沒接殘差與 FFN。

**下一階段** [`03b2-transformer-example-block.md`](03b2-transformer-example-block.md)（中等版）會沿用本文的 $\tilde X$、$C^{(1)}$ 與你練習算出的 $C^{(2)}$，補上：

- **拼接兩個 head**，再用 $W_O$ 把各頭「混合重組」
- 第一個**殘差連接**
- 第二子層：LayerNorm → **FFN** → 第二個殘差

一路算到完整的 Block 輸出 $Y$。
