# 03b2｜計算案例（中等版）：補齊多頭、$W_O$、殘差與 FFN，算到 Block 輸出

> **適合對象：** 已讀完 [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（單頭 attention），想把剩下的元件接上，並把 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) 中的 **Transformer Block 概念**落到同一組數字上的讀者。
>
> **本文定位：** 這不是另一組新例子，而是承接 03b1 的同一筆資料。03b1 先把「單頭 attention」算清楚；本文在同一個 $X$ 上繼續補上 Head 2、Concat、$W_O$、Residual、LayerNorm、FFN，最後得到完整 Pre-LN Block 的輸出 $Y$。
>
> **讀完後你能做什麼：**
> - 把 03b1 的 $\tilde X$、$C^{(1)}$ 接到本文，理解「簡單版 → 中等版」不是換題目，而是同一條資料流逐步加元件
> - 算出第二個 head，並驗證 $W_O$ 確實「混合重組」各 head 的資訊，而不是只是把 head 拼起來
> - 手算第一個殘差連接、第二子層的 LayerNorm 與 FFN（升維 → ReLU → 降維）
> - 一路算到完整 Block 輸出 $Y$，並對照 03a 的 Transformer Block 公式
>
> **前置文件：**
> - [`03a-transformer-architecture.md`](03a-transformer-architecture.md)：概念與公式，尤其 §5 Multi-Head Attention、§6 Transformer Block
> - [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)：本文直接沿用其中的 $X$、$\tilde X$、$C^{(1)}$
>
> **這是三階段計算案例的第二階段：**
> 1. [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（簡單版）— 單頭 attention：$X \to \tilde X \to C^{(1)}$
> 2. **03b2（本文，中等版）** — 補上第二頭、$W_O$、殘差、FFN，算到 Block 輸出 $Y$
> 3. [`03b3-transformer-architecture-example.md`](03b3-transformer-architecture-example.md)（完整版）— 再補縮放數值對照、Positional Encoding、相對位置旋轉驗證、NB1 重現
>
> 三份文件共用同一組 $X$ 與權重，數字完全銜接。本文數字一律四捨五入到小數第 4 位，最末位可能有 $\pm 0.0001$ 的進位誤差。

---

## 目錄

0. 先看銜接地圖：03b1 算到哪裡，03b2 接著算什麼
1. 設定：本文新增的矩陣與權重
2. 接續 03b1：已有的 $\tilde X$ 與 $C^{(1)}$
3. Head 2、拼接與 $W_O$ 混合
4. 第一個殘差連接：$Z'=X+O$
5. 第二子層：LayerNorm → FFN → 第二個殘差
6. 對照 03a：本文每一步對應 Transformer Block 的哪個元件
7. 小結與下一階段

---

## 0. 先看銜接地圖：03b1 算到哪裡，03b2 接著算什麼

03b1 的目標很小：只讓讀者看懂一個 head 的 attention 怎麼算。因此 03b1 只算到：

$$
X
\;\xrightarrow{\text{LayerNorm}}\;
\tilde X
\;\xrightarrow{\text{Head 1 Attention}}\;
C^{(1)}
$$

本文則從這裡接下去，把完整 Pre-LN Transformer Block 補完：

$$
X
\;\xrightarrow{\text{LN}}\;
\tilde X
\;\xrightarrow{\text{MHA}}\;
O
\;\xrightarrow{+X}\;
Z'
\;\xrightarrow{\text{LN}}\;
\tilde Z
\;\xrightarrow{\text{FFN}}\;
F
\;\xrightarrow{+Z'}\;
Y
$$

這正是 03a §6 的 Pre-LN Block：

$$
\tilde X=\text{LN}(X),\qquad
O=\text{MHA}(\tilde X),\qquad
Z'=X+O
$$

$$
\tilde Z=\text{LN}(Z'),\qquad
F=\text{FFN}(\tilde Z),\qquad
Y=Z'+F
$$

為了讓「簡單到複雜」的銜接更清楚，先把三份文件的任務分開看：

| 文件 | 同一筆資料流中的位置 | 本質上在學什麼 |
|---|---|---|
| 03b1 | $X \to \tilde X \to C^{(1)}$ | 單頭 attention 的核心計算 |
| 03b2 | $C^{(1)},C^{(2)} （MHA） \to O \to Z' \to F \to Y$ | 把 attention 接成完整 Transformer Block |
| 03b3 | 在同一組數字上再補 PE、縮放對照、旋轉驗證 | 把 Block 放回更完整的 Transformer 架構脈絡 |

本文會盡量避免突然出現新資料。新的只有三類：Head 2 的投影、$W_O$、FFN 權重。原始輸入 $X$ 與 03b1 算出的 $\tilde X$、$C^{(1)}$ 都直接沿用。

---

## 1. 設定：本文新增的矩陣與權重

超參數與 03b1 完全相同：

$$
T=2,\qquad d=4,\qquad H=2,\qquad d_k=d_v=2,\qquad d_{ff}=8
$$

其中 $T=2$ 表示有 2 個 token，$d=4$ 表示每個 token 是 4 維向量，$H=2$ 表示 Multi-Head Attention 有 2 個 head，每個 head 的維度為 $d_k=d_v=2$。

### 1.1 原始輸入 $X$（沿用 03b1）

$$
X=
\begin{bmatrix}
1&0&0&1\\
0&1&1&0
\end{bmatrix}
\in\mathbb{R}^{2\times4}
$$

03b1 已經先對 $X$ 做過第一個 LayerNorm，本文不重算這一步，只在 §2 直接接續使用。

### 1.2 Head 2 的投影矩陣

Head 1 在 03b1 取 $\tilde X$ 的前 2 維；Head 2 則取後 2 維：

$$
W_Q^{(2)}=W_K^{(2)}=W_V^{(2)}
=
\begin{bmatrix}
0&0\\
0&0\\
1&0\\
0&1
\end{bmatrix}
\in\mathbb{R}^{4\times2}
$$

本例刻意令 $W_Q=W_K=W_V$，目的是讓讀者專注在流程與 shape，而不是權重語意。真實模型中三者通常是不同的可學習矩陣。

### 1.3 輸出投影 $W_O$

兩個 head 算出來後，不能只停在 Concat。Multi-Head Attention 還需要 $W_O$ 把各 head 的資訊混合回 $d=4$ 維：

$$
W_O
=
\frac12
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
1&0&-1&0\\
0&1&0&-1
\end{bmatrix}
\in\mathbb{R}^{4\times4}
$$

這個 $W_O$ 是教學用設計：前兩維取兩個 head 的「和」，後兩維取兩個 head 的「差」。因此待會可以直接看到 $W_O$ 如何讓 head 之間發生相消與相長。

### 1.4 第二個 LayerNorm

第二個 LayerNorm 的可學習參數同樣取初始值：

$$
\gamma=[1,1,1,1],\qquad \beta=[0,0,0,0]
$$

也就是只做標準化，不再額外縮放或平移。

### 1.5 FFN 權重

本文使用一個很小的 Position-wise FFN：

$$
\text{FFN}(\tilde Z)=\text{ReLU}(\tilde ZW_1+b_1)W_2+b_2
$$

其中 $b_1=b_2=0$，而

$$
W_1=
\frac12
\begin{bmatrix}
1&-1&0&1&1&0&-1&0\\
0&1&1&-1&0&1&0&-1\\
1&0&-1&0&1&-1&0&1\\
-1&1&0&1&0&0&1&-1
\end{bmatrix}
\in\mathbb{R}^{4\times8}
$$

$$
W_2=
\frac12
\begin{bmatrix}
1&0&1&-1\\
0&1&-1&0\\
1&-1&0&1\\
-1&0&1&0\\
0&1&0&1\\
1&0&-1&0\\
0&-1&0&1\\
-1&0&1&0
\end{bmatrix}
\in\mathbb{R}^{8\times4}
$$

Shape 上是：

$$
(2\times4)(4\times8)(8\times4)=(2\times4)
$$

所以 FFN 雖然中間升到 8 維，最後仍回到 $2\times4$，才能與殘差相加。

---

## 2. 接續 03b1：已有的 $\tilde X$ 與 $C^{(1)}$

03b1 已經算出第一個 LayerNorm 後的輸入：

$$
\tilde X=
\begin{bmatrix}
1&-1&-1&1\\
-1&1&1&-1
\end{bmatrix}
$$

也已經算出 Head 1 的 context：

$$
C^{(1)}
=
\begin{bmatrix}
0.8884&-0.8884\\
-0.8884&0.8884
\end{bmatrix}
\in\mathbb{R}^{2\times2}
$$

其中 Head 1 的注意力權重為：

$$
A^{(1)}
=
\begin{bmatrix}
0.9442&0.0558\\
0.0558&0.9442
\end{bmatrix}
$$

這裡要注意 Pre-LN 的資料流：attention 的輸入是 $\tilde X$，但第一個殘差稍後會加回原始 $X$，不是加回 $\tilde X$。這一點對應 03a §6 的 Pre-LN 結構：

$$
Z'=X+\text{MHA}(\text{LN}(X))
$$

---

## 3. Head 2、拼接與 $W_O$ 混合

### 3.1 Head 2：沿用同一個 attention 流程

Head 2 取 $\tilde X$ 的後 2 維：

$$
Q^{(2)}=K^{(2)}=V^{(2)}
=
\tilde XW_Q^{(2)}
=
\begin{bmatrix}
-1&1\\
1&-1
\end{bmatrix}
$$

原始分數：

$$
S^{(2)}
=
Q^{(2)}(K^{(2)})^\top
=
\begin{bmatrix}
2&-2\\
-2&2
\end{bmatrix}
$$

這裡剛好與 Head 1 相同，因此縮放與 softmax 後的注意力權重也相同：

$$
A^{(2)}
=
\text{softmax}_\text{row}\left(\frac{S^{(2)}}{\sqrt2}\right)
=
\begin{bmatrix}
0.9442&0.0558\\
0.0558&0.9442
\end{bmatrix}
=
A^{(1)}
$$

但是 $V^{(2)}$ 與 $V^{(1)}$ 不同，所以 context 不同：

$$
C^{(2)}
=
A^{(2)}V^{(2)}
=
\begin{bmatrix}
0.9442&0.0558\\
0.0558&0.9442
\end{bmatrix}
\begin{bmatrix}
-1&1\\
1&-1
\end{bmatrix}
=
\begin{bmatrix}
-0.8884&0.8884\\
0.8884&-0.8884
\end{bmatrix}
$$

這裡可以連回 03a §5 的概念：Multi-Head 的重點不是一定要讓每個 head 的注意力權重都不同；即使本例兩個 head 的 $A$ 相同，只要 $V$ 的子空間不同，讀出的內容仍會不同。

簡化地說：

$$
A^{(1)}=A^{(2)}
\quad\text{但}\quad
V^{(1)}\ne V^{(2)}
\quad\Rightarrow\quad
C^{(1)}\ne C^{(2)}
$$

### 3.2 拼接：把兩個 $2\times2$ 接回 $2\times4$

把兩個 head 的輸出沿 feature 維度拼接：

$$
\text{Concat}(C^{(1)},C^{(2)})
=
\begin{bmatrix}
0.8884&-0.8884&-0.8884&0.8884\\
-0.8884&0.8884&0.8884&-0.8884
\end{bmatrix}
\in\mathbb{R}^{2\times4}
$$

Shape 對照：

$$
C^{(1)}\in\mathbb{R}^{2\times2},\quad
C^{(2)}\in\mathbb{R}^{2\times2}
\quad\Rightarrow\quad
\text{Concat}\in\mathbb{R}^{2\times4}
$$

此時只是把兩個 head 的結果並排放在一起。若到這裡就停下，Head 1 的資訊永遠留在前兩維，Head 2 的資訊永遠留在後兩維，兩者還沒有真正混合。

### 3.3 $W_O$ 混合：拼接不是終點

Multi-Head Attention 的輸出不是停在 Concat，而是還要再乘上一個輸出投影矩陣 $W_O$：

$$
O=\text{Concat}(C^{(1)},C^{(2)})W_O
$$

先把 Concat 的結果命名為 $H$，也就是：

$$
H=\text{Concat}(C^{(1)},C^{(2)})
=
\begin{bmatrix}
0.8884&-0.8884&-0.8884&0.8884\\
-0.8884&0.8884&0.8884&-0.8884
\end{bmatrix}
\in\mathbb{R}^{2\times4}
$$

其中每一列代表一個 token 的 multi-head 輸出。以第 1 個 token 為例：

$$
H_1=[0.8884,-0.8884,-0.8884,0.8884]
$$

這一列其實是由 Head 1 與 Head 2 拼起來的：

$$
C^{(1)}_1=[0.8884,-0.8884],\qquad
C^{(2)}_1=[-0.8884,0.8884]
$$

所以可以把 $H_1$ 看成：

$$
H_1=[a,b,c,d]
$$

其中：

$$
a=0.8884,\qquad b=-0.8884,\qquad c=-0.8884,\qquad d=0.8884
$$

也就是：

$$
[a,b]\text{ 來自 Head 1},\qquad [c,d]\text{ 來自 Head 2}
$$

本文設定的 $W_O$ 為：

$$
W_O
=
\frac12
\begin{bmatrix}
1&0&1&0\\
0&1&0&1\\
1&0&-1&0\\
0&1&0&-1
\end{bmatrix}
$$

矩陣乘法可以用「一個輸出維度看一欄」來理解。把 $W_O$ 拆成四個欄向量：

$$
W_O
=
\begin{bmatrix}
|&|&|&|\\
w_{O,1}&w_{O,2}&w_{O,3}&w_{O,4}\\
|&|&|&|
\end{bmatrix}
$$

其中：

$$
w_{O,1}=\frac12
\begin{bmatrix}
1\\0\\1\\0
\end{bmatrix},\quad
w_{O,2}=\frac12
\begin{bmatrix}
0\\1\\0\\1
\end{bmatrix},\quad
w_{O,3}=\frac12
\begin{bmatrix}
1\\0\\-1\\0
\end{bmatrix},\quad
w_{O,4}=\frac12
\begin{bmatrix}
0\\1\\0\\-1
\end{bmatrix}
$$

因此第 1 個 token 的輸出 $O_1$ 不是一次憑空得到，而是由四個內積組成：

$$
O_1=H_1W_O
=
[H_1w_{O,1},\ H_1w_{O,2},\ H_1w_{O,3},\ H_1w_{O,4}]
$$

逐一計算如下。

第 1 個輸出維度：

$$
H_1w_{O,1}
=
[0.8884,-0.8884,-0.8884,0.8884]
\cdot
\frac12
\begin{bmatrix}
1\\0\\1\\0
\end{bmatrix}
$$

$$
=\frac12(0.8884+(-0.8884))=0
$$

這一維是在取 Head 1 第 1 維與 Head 2 第 1 維的平均，也就是：

$$
\frac12(a+c)
$$

第 2 個輸出維度：

$$
H_1w_{O,2}
=
[0.8884,-0.8884,-0.8884,0.8884]
\cdot
\frac12
\begin{bmatrix}
0\\1\\0\\1
\end{bmatrix}
$$

$$
=\frac12((-0.8884)+0.8884)=0
$$

這一維是在取 Head 1 第 2 維與 Head 2 第 2 維的平均，也就是：

$$
\frac12(b+d)
$$

第 3 個輸出維度：

$$
H_1w_{O,3}
=
[0.8884,-0.8884,-0.8884,0.8884]
\cdot
\frac12
\begin{bmatrix}
1\\0\\-1\\0
\end{bmatrix}
$$

$$
=\frac12(0.8884-(-0.8884))=0.8884
$$

這一維是在取 Head 1 第 1 維與 Head 2 第 1 維的差，也就是：

$$
\frac12(a-c)
$$

第 4 個輸出維度：

$$
H_1w_{O,4}
=
[0.8884,-0.8884,-0.8884,0.8884]
\cdot
\frac12
\begin{bmatrix}
0\\1\\0\\-1
\end{bmatrix}
$$

$$
=\frac12((-0.8884)-0.8884)=-0.8884
$$

這一維是在取 Head 1 第 2 維與 Head 2 第 2 維的差，也就是：

$$
\frac12(b-d)
$$

因此第 1 個 token 的輸出是：

$$
O_1=[0,0,0.8884,-0.8884]
$$

用同樣方式計算第 2 個 token。第 2 列為：

$$
H_2=[-0.8884,0.8884,0.8884,-0.8884]
$$

也就是：

$$
a=-0.8884,\qquad b=0.8884,\qquad c=0.8884,\qquad d=-0.8884
$$

套入同一個規則：

$$
H_2W_O
=
\left[
\frac12(a+c),\
\frac12(b+d),\
\frac12(a-c),\
\frac12(b-d)
\right]
$$

得到：

$$
O_2
=
\left[
\frac12(-0.8884+0.8884),\
\frac12(0.8884+(-0.8884)),\
\frac12(-0.8884-0.8884),\
\frac12(0.8884-(-0.8884))
\right]
$$

$$
O_2=[0,0,-0.8884,0.8884]
$$

所以完整輸出為：

$$
O=HW_O
=
\begin{bmatrix}
0&0&0.8884&-0.8884\\
0&0&-0.8884&0.8884
\end{bmatrix}
\in\mathbb{R}^{2\times4}
$$

從公式上看，本文設計的 $W_O$ 對任一列 $[a,b,c,d]$ 都會做下列轉換：

$$
[a,b,c,d]W_O
=
\left[
\frac12(a+c),\
\frac12(b+d),\
\frac12(a-c),\
\frac12(b-d)
\right]
$$

也就是：

$$
\text{前兩維}=\text{Head 1 與 Head 2 的和},\qquad
\text{後兩維}=\text{Head 1 與 Head 2 的差}
$$

在本例中，兩個 head 的方向剛好相反：

$$
C^{(2)}=-C^{(1)}
$$

因此前兩個「和」通道相互抵消，變成 0；後兩個「差」通道則把兩個 head 的差異保留下來。

這就是 $W_O$ 的作用：它讓不同 head 的輸出進行線性組合。Concat 只是把各 head 並排放在一起；$W_O$ 才是真正把不同 head 的資訊重新混合回模型維度 $d=4$ 的步驟。

---

## 4. 第一個殘差連接：$Z'=X+O$

Pre-LN 的第一個殘差連接加回原始輸入 $X$：

$$
Z'=X+O
$$

代入數值：

$$
Z'
=
\begin{bmatrix}
1&0&0&1\\
0&1&1&0
\end{bmatrix}
+
\begin{bmatrix}
0&0&0.8884&-0.8884\\
0&0&-0.8884&0.8884
\end{bmatrix}
$$

$$
Z'
=
\begin{bmatrix}
1&0&0.8884&0.1116\\
0&1&0.1116&0.8884
\end{bmatrix}
\in\mathbb{R}^{2\times4}
$$

這一步對應 03a §6.3 的殘差概念：

$$
\text{輸出}=\text{原始輸入}+\text{子層學到的修正量}
$$

在本例中，$X$ 是原始資訊，$O$ 是 Multi-Head Attention 提供的上下文修正。殘差不是把 $X$ 丟掉重算，而是在保留 $X$ 的基礎上加入 $O$。

這也說明為什麼 MHA 輸出必須是 $T\times d=2\times4$。只有形狀與 $X$ 相同，才可以逐元素相加：

$$
X\in\mathbb{R}^{2\times4},\qquad
O\in\mathbb{R}^{2\times4},\qquad
Z'=X+O\in\mathbb{R}^{2\times4}
$$

---

## 5. 第二子層：LayerNorm → FFN → 第二個殘差

第一子層已完成：

$$
X\to \tilde X\to O\to Z'
$$

接下來進入第二子層：

$$
Z'
\;\xrightarrow{\text{LayerNorm}}\;
\tilde Z
\;\xrightarrow{\text{FFN}}\;
F
\;\xrightarrow{+Z'}\;
Y
$$

### 5.1 第二個 LayerNorm：$\tilde Z=\text{LayerNorm}(Z')$

先對 $Z'$ 的每一列獨立做 LayerNorm。

以第 1 列為例：

$$
z'_1=[1,0,0.8884,0.1116]
$$

平均值：

$$
\mu=\frac{1+0+0.8884+0.1116}{4}=0.5
$$

變異數：

$$
\sigma^2
=
\frac{(1-0.5)^2+(0-0.5)^2+(0.8884-0.5)^2+(0.1116-0.5)^2}{4}
=
0.2004
$$

標準差：

$$
\sigma=\sqrt{0.2004}=0.4477
$$

標準化：

$$
\tilde z_1
=
\frac{[0.5,-0.5,0.3884,-0.3884]}{0.4477}
=
[1.1169,-1.1169,0.8675,-0.8675]
$$

第 2 列同理，得到：

$$
\tilde Z
=
\begin{bmatrix}
1.1169&-1.1169&0.8675&-0.8675\\
-1.1169&1.1169&-0.8675&0.8675
\end{bmatrix}
$$

這一步的角色與 03a §6.2 的第一個 LayerNorm 相同：在進入子層前先穩定每個 token 的 hidden vector 數值分佈。

### 5.2 FFN Step 1：升維 $\tilde ZW_1$

FFN 先把 $d=4$ 維展開到 $d_{ff}=8$ 維：

$$
\tilde ZW_1
\in\mathbb{R}^{2\times8}
$$

以第 1 列、第 1 個輸出為例，取 $W_1$ 的第 1 欄：

$$
\tfrac12[1,0,1,-1]^\top=[0.5,0,0.5,-0.5]^\top
$$

因此：

$$
(\tilde ZW_1)_{1,1}
=
1.1169(0.5)+(-1.1169)(0)+0.8675(0.5)+(-0.8675)(-0.5)
$$

$$
=0.5585+0.4338+0.4338=1.4260
$$

完整升維結果為：

$$
\tilde ZW_1
=
\begin{bmatrix}
1.4260&-1.5506&-0.9922&0.6831&0.9922&-0.9922&-0.9922&1.4260\\
-1.4260&1.5506&0.9922&-0.6831&-0.9922&0.9922&0.9922&-1.4260
\end{bmatrix}
$$

### 5.3 FFN Step 2：ReLU 非線性

ReLU 把負值歸零，保留正值：

$$
\text{ReLU}(\tilde ZW_1)
=
\begin{bmatrix}
1.4260&0&0&0.6831&0.9922&0&0&1.4260\\
0&1.5506&0.9922&0&0&0.9922&0.9922&0
\end{bmatrix}
$$

這一步是 FFN 的關鍵。若沒有 ReLU，$W_1W_2$ 會合併成單一線性矩陣，整個 FFN 只等於一次線性投影。加入 ReLU 後，FFN 才能做非線性加工，這呼應 03a §6.4 對 FFN 的說明。

### 5.4 FFN Step 3：降維回 $d=4$

再乘上 $W_2$，把 8 維降回 4 維：

$$
F=\text{ReLU}(\tilde ZW_1)W_2
$$

$$
F
=
\begin{bmatrix}
-0.3415&0.4961&1.7675&-0.2169\\
0.9922&-0.2169&-1.2714&0.9922
\end{bmatrix}
\in\mathbb{R}^{2\times4}
$$

Shape 對照：

$$
(2\times8)(8\times4)=(2\times4)
$$

FFN 是 position-wise：它對每個 token 的 4 維向量各自加工，不在這一步跨 token 互動。跨 token 互動已經在 MHA 子層發生；FFN 的任務是對每個位置的表示做非線性轉換。

### 5.5 第二個殘差連接：$Y=Z'+F$

最後把 FFN 輸出加回第二子層的輸入 $Z'$：

$$
Y=Z'+F
$$

代入數值：

$$
Y
=
\begin{bmatrix}
1&0&0.8884&0.1116\\
0&1&0.1116&0.8884
\end{bmatrix}
+
\begin{bmatrix}
-0.3415&0.4961&1.7675&-0.2169\\
0.9922&-0.2169&-1.2714&0.9922
\end{bmatrix}
$$

$$
\boxed{
Y=
\begin{bmatrix}
0.6585&0.4961&2.6559&-0.1053\\
0.9922&0.7831&-1.1598&1.8806
\end{bmatrix}
\in\mathbb{R}^{2\times4}
}
$$

至此，完整 Pre-LN Transformer Block 已經從 $X$ 算到 $Y$。

關鍵是 shape 沒有變：

$$
X\in\mathbb{R}^{2\times4}
\quad\Rightarrow\quad
Y\in\mathbb{R}^{2\times4}
$$

因此 $Y$ 可以直接作為下一個 Transformer Block 的輸入。這就是 03a §6 所說「Block 可以堆疊」的實際數值版本。

---

## 6. 對照 03a：本文每一步對應 Transformer Block 的哪個元件

把本文計算結果整理成一張總表：

| 03a 概念 | 本文公式 | 本文數值結果 | Shape |
|---|---|---|---|
| 輸入 | $X$ | $\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix}$ | $2\times4$ |
| 第一個 LayerNorm | $\tilde X=\text{LN}(X)$ | $\begin{bmatrix}1&-1&-1&1\\-1&1&1&-1\end{bmatrix}$ | $2\times4$ |
| Head 1 | $C^{(1)}$ | $\begin{bmatrix}0.8884&-0.8884\\-0.8884&0.8884\end{bmatrix}$ | $2\times2$ |
| Head 2 | $C^{(2)}$ | $\begin{bmatrix}-0.8884&0.8884\\0.8884&-0.8884\end{bmatrix}$ | $2\times2$ |
| Concat | $\text{Concat}(C^{(1)},C^{(2)})$ | $\begin{bmatrix}0.8884&-0.8884&-0.8884&0.8884\\-0.8884&0.8884&0.8884&-0.8884\end{bmatrix}$ | $2\times4$ |
| 輸出投影 | $O=\text{Concat}W_O$ | $\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}$ | $2\times4$ |
| 第一個殘差 | $Z'=X+O$ | $\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}$ | $2\times4$ |
| 第二個 LayerNorm | $\tilde Z=\text{LN}(Z')$ | $\begin{bmatrix}1.1169&-1.1169&0.8675&-0.8675\\-1.1169&1.1169&-0.8675&0.8675\end{bmatrix}$ | $2\times4$ |
| FFN | $F=\text{ReLU}(\tilde ZW_1)W_2$ | $\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}$ | $2\times4$ |
| 第二個殘差／Block 輸出 | $Y=Z'+F$ | $\begin{bmatrix}0.6585&0.4961&2.6559&-0.1053\\0.9922&0.7831&-1.1598&1.8806\end{bmatrix}$ | $2\times4$ |

整條資料流可以壓成一句話：

$$
X
\to
\text{LN}
\to
\text{MHA}
\to
\text{Residual}
\to
\text{LN}
\to
\text{FFN}
\to
\text{Residual}
\to
Y
$$

但本文的重點是讓這句話不再只是架構圖，而是可以逐格算出來的矩陣流程。

---

## 7. 小結與下一階段

本文完成的是 03b1 的延伸版。03b1 已經讓你知道單頭 attention 如何從 $\tilde X$ 算到 $C^{(1)}$；本文把同一筆資料繼續往後推，補上完整 Transformer Block 的剩餘元件：

1. Head 2：同一個 attention 流程再跑一次，得到 $C^{(2)}$
2. Concat：把 $C^{(1)}$ 與 $C^{(2)}$ 從兩個 $2\times2$ 接成一個 $2\times4$
3. $W_O$：把各 head 的資訊混合回模型維度 $d=4$
4. 第一個殘差：$Z'=X+O$
5. 第二個 LayerNorm：$\tilde Z=\text{LN}(Z')$
6. FFN：$\text{ReLU}(\tilde ZW_1)W_2$
7. 第二個殘差：$Y=Z'+F$

最重要的結論是：Transformer Block 內部做了很多事，但輸入與輸出的 shape 維持一致。

$$
X\in\mathbb{R}^{T\times d}
\quad\Longrightarrow\quad
Y\in\mathbb{R}^{T\times d}
$$

這個 shape 契約讓多個 Block 可以直接堆疊，也讓 03a 的抽象架構與本文的具體數字接在一起。

**下一階段** [`03b3-transformer-architecture-example.md`](03b3-transformer-architecture-example.md) 會在同一組數字上再補三件事：

- 縮放對 softmax 分佈的影響：除以 $\sqrt{d_k}$ 與不除的差異
- Positional Encoding 的實際編碼表
- 「相對位置 = 旋轉」的數值驗證，以及如何用 notebook 重現所有矩陣
