# 03b4｜計算案例（含位置編碼）：把 $P$ 真的加進去，從頭算一次

> **這是 03b 系列的選讀對照支線，純計算展演。** 本文不重述概念——位置編碼是什麼、為什麼相加不弄混、相對位置為何等價旋轉，都已在 [`03a` §7](03a-transformer-architecture.md) 與計算案例 [`03b3` §6](03b3-transformer-architecture-example.md) 說明。這裡只做一件事：**把真實位置向量 $P$ 加進輸入（$X_{\text{in}}=X+P$），沿用與 03b3 完全相同的權重，把整個 Pre-LN Block 從頭到尾詳算一次，把每一步的矩陣中間值攤開。**
>
> **與 03b3 的差別：** 03b1／03b2／03b3 為維持手算數字乾淨，一律取 $P=0$（直接用 $X$）；本文取 $P\neq0$。因此**本文的數字自成一組**，不與 03b1–03b3／NB1 §13 共用——這也正是它單獨成篇的原因：真實 PE（含 $\sin 1$ 這類無理數）一加進去，後面每一步都變成小數，兩列不再鏡像對稱。
>
> **對應實作：** [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13b 可一鍵重現本文所有數字。數字四捨五入到小數第 4 位，最末位可能有 $\pm0.0001$ 進位誤差。

---

## 0. 設定：輸入改成 $X_{\text{in}}=X+P$，權重沿用 03b3

超參數與所有權重矩陣（$W^{(1)},W^{(2)},W_O,W_1,W_2$、LayerNorm $\gamma=1,\beta=0$）**與 [`03b3` §0](03b3-transformer-architecture-example.md) 完全相同**，此處不重列。唯一的差別是輸入：把 token embedding $X$ 加上位置 0、1 的位置向量 $P$（取自 [`03b3` §6.1](03b3-transformer-architecture-example.md) 的編碼表）：

$$
X=\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix},\qquad
P=\begin{bmatrix}0&1&0&1\\0.8415&0.5403&0.0100&1.0000\end{bmatrix}
$$

$$
\boxed{\,X_{\text{in}}=X+P=\begin{bmatrix}1&1&0&2\\0.8415&1.5403&1.0100&1.0000\end{bmatrix}\in\mathbb{R}^{2\times4}\,}
$$

以下所有步驟的骨架與 03b3 §1–§5 相同，只是輸入換成 $X_{\text{in}}$、數字全部重算。

---

## 1. Pre-LN：$\tilde X=\text{LayerNorm}(X_{\text{in}})$

對每一列（每個 token 的 4 維）各自減均值、除以標準差（population variance）。

- 第 0 列 $[1,1,0,2]$：均值 $1$、變異數 $0.5$、標準差 $0.7071$ → $[0,\,0,\,-1.4142,\,1.4142]$
- 第 1 列 $[0.8415,1.5403,1.0100,1.0000]$：均值 $1.0980$、變異數 $0.0637$、標準差 $0.2524$ → $[-0.9714,\,1.6756,\,-0.3331,\,-0.3711]$

$$
\tilde X=\begin{bmatrix}0&0&-1.4142&1.4142\\-0.9714&1.6756&-0.3331&-0.3711\end{bmatrix}
$$

> 對照 03b3：那裡 $\tilde X$ 是乾淨的 $\begin{bmatrix}1&-1&-1&1\\-1&1&1&-1\end{bmatrix}$。加了 $P$ 之後，這份對稱性消失了。

---

## 2. Head 1：取前 2 維走 Scaled Dot-Product

$W^{(1)}$ 取 $\tilde X$ 的**前 2 維**，本例 $W_Q=W_K=W_V$，故 $Q=K=V$：

$$
Q^{(1)}=K^{(1)}=V^{(1)}=\begin{bmatrix}0&0\\-0.9714&1.6756\end{bmatrix}
$$

原始分數 $S=QK^\top$、縮放 $E=S/\sqrt{d_k}$（$\sqrt{2}=1.4142$）、softmax 得 $A$：

$$
S=\begin{bmatrix}0&0\\0&3.7513\end{bmatrix}\ \to\
E=\begin{bmatrix}0&0\\0&2.6526\end{bmatrix}\ \to\
A^{(1)}=\begin{bmatrix}0.5000&0.5000\\0.0658&0.9342\end{bmatrix}
$$

加權 $V$ 得 context：

$$
C^{(1)}=A^{(1)}V^{(1)}=\begin{bmatrix}-0.4857&0.8378\\-0.9075&1.5653\end{bmatrix}
$$

---

## 3. Head 2：取後 2 維，同一套流程

$W^{(2)}$ 取 $\tilde X$ 的**後 2 維**：

$$
Q^{(2)}=K^{(2)}=V^{(2)}=\begin{bmatrix}-1.4142&1.4142\\-0.3331&-0.3711\end{bmatrix}
$$

$$
S=\begin{bmatrix}4.0000&-0.0538\\-0.0538&0.2487\end{bmatrix}\ \to\
E=\begin{bmatrix}2.8284&-0.0381\\-0.0381&0.1758\end{bmatrix}\ \to\
A^{(2)}=\begin{bmatrix}0.9462&0.0538\\0.4467&0.5533\end{bmatrix}
$$

$$
C^{(2)}=A^{(2)}V^{(2)}=\begin{bmatrix}-1.3560&1.3181\\-0.8160&0.4264\end{bmatrix}
$$

---

## 4. 拼接與輸出投影：$O=\text{Concat}(C^{(1)},C^{(2)})\,W_O$

沿 feature 維把兩個 $2\times2$ 接成 $2\times4$，再乘 $W_O$（$4\times4$）：

$$
\text{Concat}=\begin{bmatrix}-0.4857&0.8378&-1.3560&1.3181\\-0.9075&1.5653&-0.8160&0.4264\end{bmatrix}
$$

$$
O=\text{Concat}\cdot W_O=\begin{bmatrix}-0.9209&1.0780&0.4352&-0.2401\\-0.8618&0.9959&-0.0457&0.5694\end{bmatrix}
$$

---

## 5. 第一個殘差連接：$Z'=X_{\text{in}}+O$

Pre-LN 的殘差加在**原始輸入**（這裡是 $X_{\text{in}}$，不是 $\tilde X$）上：

$$
Z'=X_{\text{in}}+O=\begin{bmatrix}0.0791&2.0780&0.4352&1.7599\\-0.0203&2.5362&0.9643&1.5694\end{bmatrix}
$$

---

## 6. 第二子層：$\text{LayerNorm}\to\text{FFN}\to$ 第二個殘差

**Step 1：$\tilde Z=\text{LayerNorm}(Z')$。**

$$
\tilde Z=\begin{bmatrix}-1.1899&1.1676&-0.7700&0.7924\\-1.3810&1.3714&-0.3209&0.3305\end{bmatrix}
$$

**Step 2：升維 $\tilde Z W_1$（$2\times4\cdot4\times8=2\times8$）→ ReLU（負值歸零）。**

$$
\tilde Z W_1=\begin{bmatrix}-1.3762&1.5749&0.9688&-0.7826&-0.9800&0.9688&0.9912&-1.3650\\-1.0162&1.5414&0.8462&-1.2109&-0.8510&0.8462&0.8557&-1.0114\end{bmatrix}
$$

$$
\text{ReLU}(\tilde Z W_1)=\begin{bmatrix}0&1.5749&0.9688&0&0&0.9688&0.9912&0\\0&1.5414&0.8462&0&0&0.8462&0.8557&0\end{bmatrix}
$$

**Step 3：降維 $\cdot\,W_2$（$2\times8\cdot8\times4=2\times4$）得 FFN 輸出 $F$。**

$$
F=\begin{bmatrix}0.9688&-0.1925&-1.2719&0.9800\\0.8462&-0.0802&-1.1938&0.8510\end{bmatrix}
$$

**Step 4：第二個殘差 $Y=Z'+F$。**

$$
\boxed{\,Y=Z'+F=\begin{bmatrix}1.0479&1.8854&-0.8367&2.7398\\0.8259&2.4559&-0.2295&2.4203\end{bmatrix}\in\mathbb{R}^{2\times4}\,}
$$

---

## 7. 檢核

全程 shape 維持 $2\times4$（$X_{\text{in}}\to\tilde X\to O\to Z'\to F\to Y$，heads 為 $2\times2$），Block 的形狀契約與可堆疊性不因加入位置編碼而改變——變的只是數字。若要一鍵重現，見 [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13b。
