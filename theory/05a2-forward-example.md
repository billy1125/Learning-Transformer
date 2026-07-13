# 05a2｜向前傳播數值範例：一次前向逐階段實算

> **適合對象：** 讀完 [`05a1-forward-propagation.md`](05a1-forward-propagation.md)（前向符號數學）後，想看「每個公式代進真實數字長什麼樣」的讀者。
>
> **本文做什麼：** 選一組**小到能手算、又能跑完整條 pipeline** 的範例資料（$T=2$、$d=3$、單頭、含因果遮罩），從 token 一路算到 Cross-Entropy loss，**每個階段都給出實際數字**，並標明對應 [`05a1`](05a1-forward-propagation.md) 的哪一節。
>
> **前置文件：** [`05a1-forward-propagation.md`](05a1-forward-propagation.md)、[`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)
>
> **下一步：** → [`05b2-backward-example.md`](05b2-backward-example.md)（**沿用本文的同一組數字**，把反向傳播每個階段也實際算一次）

> **數字精度：** 全部四捨五入到小數 3 位，可能有末位進位誤差；$\epsilon$ 取極小值忽略。所有數字都以 PyTorch autograd 交叉驗證過。

---

## 0. 範例設定

一個序列只有 **2 個 token**（$T=2$），模型維度 **$d=3$**，**單一** attention head（$d_k=3$），FFN 中間層 $d_{ff}=6$（慣例是 $4d$，這裡縮小以便手算），詞彙表 $|V|=3$。

| 元件 | 取值 |
|---|---|
| token ids | $t = (0,\ 1)$ |
| 詞嵌入 $E\in\mathbb{R}^{3\times3}$ | 第 0 列 $[1,1,0]$、第 1 列 $[2,0,0]$、第 2 列 $[0,0,0]$（未用到）|
| 位置嵌入 $P\in\mathbb{R}^{2\times3}$ | 位置 0 $[1,0,0]$、位置 1 $[0,0,1]$ |
| LayerNorm | 三處都用 $\gamma=\mathbf{1}$、$\beta=\mathbf{0}$（標準化後不再縮放平移）|
| Attention 投影 | $W_Q=W_K=W_V=I_3$（單位矩陣，故 $Q=K=V$）|
| FFN | $W_1\in\mathbb{R}^{3\times6}$、$W_2\in\mathbb{R}^{6\times3}$（見 §5），$b_1=b_2=0$ |
| lm_head | $W_{lm}=I_3$（$|V|=d$，故 logits 就是最終 LN 輸出）|
| 目標 | $y=(0,\ 1)$（位置 0 應預測 token 0、位置 1 應預測 token 1）|

> 取 $W_Q=W_K=W_V=I$、$W_{lm}=I$ 只是為了讓每一步的數字乾淨；真實模型這些都是可訓練矩陣。整條計算的形狀鏈是 $ (2,3)\to\text{Block}\to(2,3)\to\text{lm\_head}\to(2,3)\to\text{loss}$。

---

## 1. Token Embedding + 位置編碼（對應 [`05a1`](05a1-forward-propagation.md) §6）

查表取出每個 token 的詞向量，再加上對應位置的向量：

$$
x_0 = E[t] + P =
\begin{bmatrix} 1&1&0 \\ 2&0&0 \end{bmatrix}
+
\begin{bmatrix} 1&0&0 \\ 0&0&1 \end{bmatrix}
=
\begin{bmatrix} 2&1&0 \\ 2&0&1 \end{bmatrix}
$$

第 0 列是位置 0 的輸入向量 $[2,1,0]$，第 1 列是位置 1 的 $[2,0,1]$。這個 $x_0$ 就是進入第一個 Transformer Block 的輸入。

---

## 2. LayerNorm ①（對應 [`05a1`](05a1-forward-propagation.md) §5）

Pre-LN：先對 $x_0$ 每一列（沿 $d=3$ 個特徵）標準化。以第 0 列 $[2,1,0]$ 為例：

$$
\mu = \tfrac{2+1+0}{3}=1,\qquad
\sigma^2 = \tfrac{(2-1)^2+(1-1)^2+(0-1)^2}{3}=\tfrac{2}{3}=0.667,\qquad
\sigma=0.816
$$

$$
\text{LN}_1(x_0)_0 = \frac{[2,1,0]-1}{0.816} = \frac{[1,0,-1]}{0.816} = [1.225,\ 0,\ -1.225]
$$

第 1 列 $[2,0,1]$ 同樣 $\mu=1,\sigma=0.816$，得 $[1.225,-1.225,0]$。合起來：

$$
\text{LN}_1(x_0) = \begin{bmatrix} 1.225 & 0 & -1.225 \\ 1.225 & -1.225 & 0 \end{bmatrix}
$$

---

## 3. Causal Self-Attention（對應 [`05a1`](05a1-forward-propagation.md) §1–§2）

因為 $W_Q=W_K=W_V=I$，所以 $Q=K=V=\text{LN}_1(x_0)$。

**分數（縮放後）** $S=\dfrac{QK^\top}{\sqrt{d_k}}=\dfrac{QK^\top}{\sqrt3}$。例如 $S_{00}=\dfrac{q_0\cdot k_0}{\sqrt3}=\dfrac{1.225^2+0+1.225^2}{1.732}=\dfrac{3.0}{1.732}=1.732$；$S_{01}=\dfrac{q_0\cdot k_1}{\sqrt3}=\dfrac{1.225^2+0+0}{1.732}=\dfrac{1.5}{1.732}=0.866$：

$$
S = \begin{bmatrix} 1.732 & 0.866 \\ 0.866 & 1.732 \end{bmatrix}
$$

**因果遮罩：** 位置 0 只能看自己，把 $S_{01}$ 設成 $-\infty$；位置 1 可看 0、1：

$$
S_{\text{masked}} = \begin{bmatrix} 1.732 & -\infty \\ 0.866 & 1.732 \end{bmatrix}
$$

**逐列 softmax：**
- 位置 0：只剩一項 → $A_0 = [1,\ 0]$。
- 位置 1：$e^{0.866}=2.377$、$e^{1.732}=5.652$，和 $=8.029$，故 $A_1=[2.377/8.029,\ 5.652/8.029]=[0.296,\ 0.704]$。

$$
A = \begin{bmatrix} 1 & 0 \\ 0.296 & 0.704 \end{bmatrix}
$$

> 注意位置 0 的權重是 $[1,0]$，**不是**對稱的 $[0.67,0.33]$——因為因果遮罩讓它只能看自己。這正是「含遮罩」與「不含遮罩」attention 的差別。

**加權讀取** $C=AV$：
- $C_0 = 1\cdot v_0 + 0\cdot v_1 = [1.225,0,-1.225]$。
- $C_1 = 0.296\,v_0 + 0.704\,v_1 = 0.296[1.225,0,-1.225]+0.704[1.225,-1.225,0]=[1.225,-0.862,-0.363]$。

$$
C = \begin{bmatrix} 1.225 & 0 & -1.225 \\ 1.225 & -0.862 & -0.363 \end{bmatrix}
$$

---

## 4. 殘差①（對應 [`05a1`](05a1-forward-propagation.md) §5.2）

$$
x_1 = x_0 + C =
\begin{bmatrix} 2&1&0 \\ 2&0&1 \end{bmatrix}+
\begin{bmatrix} 1.225&0&-1.225 \\ 1.225&-0.862&-0.363 \end{bmatrix}=
\begin{bmatrix} 3.225 & 1 & -1.225 \\ 3.225 & -0.862 & 0.637 \end{bmatrix}
$$

---

## 5. LayerNorm ② + FFN（對應 [`05a1`](05a1-forward-propagation.md) §5、§4）

**LN②：** 第 0 列 $[3.225,1,-1.225]$：$\mu=1$、$\sigma^2=3.30$、$\sigma=1.817$ → $[1.225,0,-1.225]$。第 1 列 $[3.225,-0.862,0.637]$：$\mu=1$、$\sigma^2=2.849$、$\sigma=1.688$ → $[1.318,-1.103,-0.215]$。

$$
\text{LN}_2(x_1)=\begin{bmatrix} 1.225 & 0 & -1.225 \\ 1.318 & -1.103 & -0.215 \end{bmatrix}
$$

**FFN：** $\text{FFN}(u)=\text{ReLU}(uW_1)W_2$，其中

$$
W_1=\begin{bmatrix} 1&0&-1&0&1&0 \\ 0&1&0&-1&0&1 \\ -1&0&1&0&1&0 \end{bmatrix},\qquad
W_2=\begin{bmatrix} 1&0&0 \\ 0&1&0 \\ 0&0&1 \\ 1&0&0 \\ 0&1&0 \\ 0&0&1 \end{bmatrix}
$$

先升維 $z=\text{LN}_2\,W_1$（$2\times6$）：

$$
z=\begin{bmatrix} 2.449 & 0 & -2.449 & 0 & 0 & 0 \\ 1.533 & -1.103 & -1.533 & 1.103 & 1.103 & -1.103 \end{bmatrix}
$$

ReLU 把負值歸零：

$$
\text{ReLU}(z)=\begin{bmatrix} 2.449 & 0 & 0 & 0 & 0 & 0 \\ 1.533 & 0 & 0 & 1.103 & 1.103 & 0 \end{bmatrix}
$$

再降維 $\text{FFN}=\text{ReLU}(z)\,W_2$（$2\times3$）：

$$
\text{FFN}=\begin{bmatrix} 2.449 & 0 & 0 \\ 2.636 & 1.103 & 0 \end{bmatrix}
$$

**殘差②：**

$$
x_2 = x_1 + \text{FFN} = \begin{bmatrix} 5.674 & 1 & -1.225 \\ 5.861 & 0.241 & 0.637 \end{bmatrix}
$$

---

## 6. 最終 LayerNorm（對應 [`05a1`](05a1-forward-propagation.md) §5）

第 0 列 $[5.674,1,-1.225]$：$\mu=1.816$、$\sigma^2=8.266$、$\sigma=2.875$ → $[1.342,-0.284,-1.058]$。第 1 列 $[5.861,0.241,0.637]$：$\mu=2.246$、$\sigma^2=6.558$、$\sigma=2.561$ → $[1.411,-0.783,-0.628]$。

$$
h = \text{LN}_f(x_2)=\begin{bmatrix} 1.342 & -0.284 & -1.058 \\ 1.411 & -0.783 & -0.628 \end{bmatrix}
$$

---

## 7. lm_head + Softmax + Cross-Entropy（對應 [`05a1`](05a1-forward-propagation.md) §7）

$W_{lm}=I$，所以 **logits 就是 $h$**：

$$
z^{\text{logit}} = h W_{lm}^\top = \begin{bmatrix} 1.342 & -0.284 & -1.058 \\ 1.411 & -0.783 & -0.628 \end{bmatrix}
$$

**逐列 softmax → 機率 $p$。** 位置 0：$e^{1.342}=3.827$、$e^{-0.284}=0.753$、$e^{-1.058}=0.347$，和 $=4.927$ → $[0.777,0.153,0.070]$。位置 1 同法：

$$
p = \begin{bmatrix} 0.777 & 0.153 & 0.070 \\ 0.805 & 0.090 & 0.105 \end{bmatrix}
$$

**Cross-Entropy**（目標 $y=(0,1)$，取各位置正確類別的 $-\log p$ 再平均）：

$$
L = -\tfrac{1}{2}\big(\log p^{(0)}_{0} + \log p^{(1)}_{1}\big)
= -\tfrac{1}{2}\big(\log 0.777 + \log 0.090\big)
= \tfrac{1}{2}(0.252 + 2.408) = \boxed{1.332}
$$

位置 0 猜對（$p=0.777$，loss 小）；位置 1 幾乎猜錯（正確類別只有 $0.090$，loss 大）。這個 $L$ 與各階段的中間量，全部會在 [`05b2-backward-example.md`](05b2-backward-example.md) 反向用到。

---

## 各階段數值一覽

| 階段 | 輸出 | 對應 05a1 |
|---|---|---|
| Embedding+PE | $x_0=[[2,1,0],[2,0,1]]$ | §6 |
| LN① | $[[1.225,0,-1.225],[1.225,-1.225,0]]$ | §5 |
| Attention 權重 $A$ | $[[1,0],[0.296,0.704]]$ | §1–§2 |
| Attention 輸出 $C$ | $[[1.225,0,-1.225],[1.225,-0.862,-0.363]]$ | §1 |
| 殘差① $x_1$ | $[[3.225,1,-1.225],[3.225,-0.862,0.637]]$ | §5.2 |
| FFN 輸出 | $[[2.449,0,0],[2.636,1.103,0]]$ | §4 |
| 殘差② $x_2$ | $[[5.674,1,-1.225],[5.861,0.241,0.637]]$ | §5.2 |
| 最終 LN $h$ | $[[1.342,-0.284,-1.058],[1.411,-0.783,-0.628]]$ | §5 |
| 機率 $p$ | $[[0.777,0.153,0.070],[0.805,0.090,0.105]]$ | §7 |
| loss $L$ | $1.332$ | §7 |

---

## 下一步

**反向數值範例：** → [`05b2-backward-example.md`](05b2-backward-example.md)

沿用本文的 $x_0,\text{LN}_1,A,C,x_1,\text{LN}_2,\text{FFN},x_2,h,p$ 等每一個數字，從 $\partial L/\partial\text{logits}$ 一路反向算到 $\partial L/\partial E$，每個梯度都對照 [`05b1-backward-propagation.md`](05b1-backward-propagation.md) 的符號公式。
