# 注意力機制與 Nadaraya-Watson 核回歸的數學整合說明

## 一、核心觀點

注意力機制（attention mechanism）可以被視為一種資料驅動的核平滑方法。從數學形式來看，它與 **Nadaraya-Watson kernel regression** 高度一致：兩者都根據「查詢點與既有資料點之間的相似度」建立權重，然後對對應的輸出值進行加權平均。

簡要地說：

$$
\text{attention}
=
\text{learned kernel smoother}
$$
也就是說，注意力機制可以理解為一種「可學習的核平滑器」。

---

## 二、Nadaraya-Watson 核回歸

給定資料集：

$$
\{(x_i, y_i)\}_{i=1}^n
$$
其中：

- \(x_i\) 是第 \(i\) 個輸入資料點；
- \(y_i\) 是對應的輸出值；
- \(x\) 是欲估計的查詢點。

Nadaraya-Watson 核回歸用下式估計 \(f(x)\)：

$$
\hat{f}(x)
=
\frac{\sum_{i=1}^n K(x,x_i)y_i}
{\sum_{i=1}^n K(x,x_i)}
$$
其中 \(K(x,x_i)\) 是核函數，用來衡量查詢點 \(x\) 與資料點 \(x_i\) 的相似度。

將核權重正規化後，可寫成：

$$
\hat{f}(x)
=
\sum_{i=1}^n \alpha_i(x)y_i
$$
其中：

$$
\alpha_i(x)
=
\frac{K(x,x_i)}
{\sum_{j=1}^n K(x,x_j)}
$$
並且：

$$
\sum_{i=1}^n \alpha_i(x)=1
$$
因此，Nadaraya-Watson 核回歸的本質是：

$$
\text{預測值}
=
\text{根據相似度加權後的輸出平均}
$$
---

## 三、注意力機制的基本形式

在注意力機制中，通常有三組向量：

$$
q: \text{query}
$$
$$
k_i: \text{key}
$$
$$
v_i: \text{value}
$$
對於一個查詢向量 \(q\)，注意力輸出定義為：

$$
\text{Attention}(q,K,V)
=
\sum_{i=1}^n a_i(q)v_i
$$
其中注意力權重為：

$$
a_i(q)
=
\frac{\exp(s(q,k_i))}
{\sum_{j=1}^n \exp(s(q,k_j))}
$$
這裡 \(s(q,k_i)\) 是 query 與 key 的相似度分數。

在 scaled dot-product attention 中，相似度分數通常定義為：

$$
s(q,k_i)
=
\frac{q^\top k_i}{\sqrt{d_k}}
$$
因此注意力權重為：

$$
a_i(q)
=
\frac{
\exp\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)
}{
\sum_{j=1}^n
\exp\left(\frac{q^\top k_j}{\sqrt{d_k}}\right)
}
$$
最後，注意力輸出為：

$$
\text{Attention}(q,K,V)
=
\sum_{i=1}^n
\frac{
\exp\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)
}{
\sum_{j=1}^n
\exp\left(\frac{q^\top k_j}{\sqrt{d_k}}\right)
}
v_i
$$
---

## 四、注意力機制與 Nadaraya-Watson 核回歸的對應關係

兩者之間可以建立如下對應：

| Nadaraya-Watson 核回歸 | 注意力機制 |
|---|---|
| 查詢點 \(x\) | query \(q\) |
| 資料點 \(x_i\) | key \(k_i\) |
| 輸出值 \(y_i\) | value \(v_i\) |
| 核函數 \(K(x,x_i)\) | 指數相似度 \(\exp(s(q,k_i))\) |
| 正規化核權重 \(\alpha_i(x)\) | attention weight \(a_i(q)\) |

因此，注意力機制可以寫成 Nadaraya-Watson 型態的估計器：

$$
\text{Attention}(q,K,V)
=
\frac{
\sum_{i=1}^n \kappa(q,k_i)v_i
}{
\sum_{i=1}^n \kappa(q,k_i)
}
$$
其中：

$$
\kappa(q,k_i)=\exp(s(q,k_i))
$$
若使用 scaled dot-product attention，則：

$$
\kappa(q,k_i)
=
\exp\left(
\frac{q^\top k_i}{\sqrt{d_k}}
\right)
$$
所以：

$$
\text{Attention}(q,K,V)
=
\frac{
\sum_{i=1}^n
\exp\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)v_i
}{
\sum_{i=1}^n
\exp\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)
}
$$
這與 Nadaraya-Watson 核回歸：

$$
\hat{f}(x)
=
\frac{
\sum_{i=1}^n K(x,x_i)y_i
}{
\sum_{i=1}^n K(x,x_i)
}
$$
在形式上完全一致。

---

## 五、固定核函數與可學習核函數的差異

Nadaraya-Watson 核回歸通常使用事先指定的核函數，例如 Gaussian kernel：

$$
K(x,x_i)
=
\exp\left(
-\frac{\|x-x_i\|^2}{2\sigma^2}
\right)
$$
這表示距離越近，權重越大。

相較之下，注意力機制中的核函數通常不是固定的距離核，而是透過神經網路學習出來的相似度結構。

在 Transformer 中，query、key、value 通常由輸入經過線性轉換得到：

$$
q = W_Q x
$$
$$
k_i = W_K x_i
$$
$$
v_i = W_V x_i
$$
因此，attention 的核可以寫成：

$$
\kappa(q,k_i)
=
\exp\left(
\frac{
(W_Qx)^\top(W_Kx_i)
}{
\sqrt{d_k}
}
\right)
$$
進一步展開：

$$
\kappa(x,x_i)
=
\exp\left(
\frac{
x^\top W_Q^\top W_K x_i
}{
\sqrt{d_k}
}
\right)
$$
若令：

$$
M = W_Q^\top W_K
$$
則：

$$
\kappa(x,x_i)
=
\exp\left(
\frac{x^\top M x_i}{\sqrt{d_k}}
\right)
$$
這表示 Transformer 的注意力機制其實是在學習一個資料依賴的相似度度量。相較於傳統固定核函數，這種 learned kernel 具有更高的表達能力。

---

## 六、Self-Attention 作為多點核回歸

在 self-attention 中，輸入序列為：

$$
X = [x_1,x_2,\dots,x_n]
$$
每個 token 都產生自己的 query、key、value：

$$
q_i = W_Qx_i
$$
$$
k_j = W_Kx_j
$$
$$
v_j = W_Vx_j
$$
第 \(i\) 個 token 的輸出為：

$$
o_i
=
\sum_{j=1}^n
\frac{
\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)
}{
\sum_{\ell=1}^n
\exp\left(\frac{q_i^\top k_\ell}{\sqrt{d_k}}\right)
}
v_j
$$
這等價於對每一個查詢點 \(x_i\)，都做一次 Nadaraya-Watson 型核回歸：

$$
o_i
=
\frac{
\sum_{j=1}^n
\kappa(x_i,x_j)v_j
}{
\sum_{j=1}^n
\kappa(x_i,x_j)
}
$$
其中：

$$
\kappa(x_i,x_j)
=
\exp\left(
\frac{
(W_Qx_i)^\top(W_Kx_j)
}{
\sqrt{d_k}
}
\right)
$$
因此，self-attention 可以理解為：

> 每個 token 都以自己為查詢點，對整個序列做一次核回歸。

---

## 七、矩陣形式下的整合

令輸入矩陣為 \(X\)，則：

$$
Q=XW_Q
$$
$$
K=XW_K
$$
$$
V=XW_V
$$
注意力矩陣為：

$$
A
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)
$$
輸出為：

$$
O = AV
$$
其中第 \(i\) 行、第 \(j\) 列的注意力權重為：

$$
A_{ij}
=
\frac{
\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)
}{
\sum_{\ell=1}^n
\exp\left(\frac{q_i^\top k_\ell}{\sqrt{d_k}}\right)
}
$$
這正是 Nadaraya-Watson 核回歸中的正規化核權重。

因此，矩陣形式的 self-attention 可以理解為同時對所有查詢點進行核回歸：

$$
O_i
=
\sum_j A_{ij}V_j
$$
---

## 八、Multi-Head Attention 作為多組核回歸

Multi-head attention 可以理解為多組不同核函數的核回歸。

第 \(h\) 個 head 有自己的投影矩陣：

$$
W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}
$$
因此，第 \(h\) 個 head 對應的核函數為：

$$
\kappa^{(h)}(x_i,x_j)
=
\exp\left(
\frac{
(W_Q^{(h)}x_i)^\top(W_K^{(h)}x_j)
}{
\sqrt{d_h}
}
\right)
$$
每個 head 產生一個輸出：

$$
O^{(h)}
=
\text{softmax}
\left(
\frac{Q^{(h)}K^{(h)\top}}{\sqrt{d_h}}
\right)
V^{(h)}
$$
最後再串接並線性轉換：

$$
O
=
\text{Concat}(O^{(1)},\dots,O^{(H)})W_O
$$
這表示模型不是只學一個相似度結構，而是同時學習多個不同的核平滑方式。每個 head 可以捕捉不同的關係、尺度或語意結構。

---

## 九、整合後的總結公式

注意力機制與 Nadaraya-Watson 核回歸的整合形式可以寫成：

$$
\boxed{
\text{Attention}(q,K,V)
=
\frac{
\sum_i \kappa(q,k_i)v_i
}{
\sum_i \kappa(q,k_i)
}
}
$$
其中：

$$
\kappa(q,k_i)
=
\exp(s(q,k_i))
$$
若使用 scaled dot-product attention，則：

$$
\boxed{
\kappa(q,k_i)
=
\exp\left(
\frac{q^\top k_i}{\sqrt{d_k}}
\right)
}
$$
因此：

$$
\boxed{
\text{Scaled Dot-Product Attention}
=
\text{Nadaraya-Watson Kernel Regression with a learned exponential dot-product kernel}
}
$$
---

## 十、概念性結論

Nadaraya-Watson 核回歸是根據輸入點之間的相似度，對輸出值做加權平均。注意力機制則是根據 query 與 key 的相似度，對 value 做加權平均。

兩者在數學形式上是一致的。主要差異在於：

1. Nadaraya-Watson 核回歸通常使用固定核函數，例如 Gaussian kernel。
2. 注意力機制使用可學習的 query、key、value 投影。
3. Transformer 中的 attention 可被視為 learned kernel regression。
4. Multi-head attention 則可視為多組 learned kernels 的並行核回歸。

因此，從數學角度而言，注意力機制可以被嚴格理解為一種 Nadaraya-Watson 型的核回歸架構，只是其核函數與特徵空間由神經網路端到端學習而來。

---

## 十一、數字範例：用身高預測體重

以下用一個具體的數字範例，說明 Nadaraya-Watson 核回歸（以及注意力機制）的運算過程。

### 情境設定

已知三筆資料，目標是預測身高 $x = 1.75$ m 的體重：

| 資料點 | 身高 $x_i$ (m) | 體重 $y_i$ (kg) |
|--------|----------------|-----------------|
| $i=1$  | 1.60           | 55              |
| $i=2$  | 1.75           | 70              |
| $i=3$  | 1.90           | 85              |

查詢點：$x = 1.75$

---

### 步驟一：計算核相似度

使用 Gaussian kernel，設 $\sigma = 0.10$：

$$
K(x, x_i) = \exp\!\left(-\frac{(x - x_i)^2}{2\sigma^2}\right)
$$

各點的距離平方 $(x - x_i)^2$：

$$
(1.75 - 1.60)^2 = 0.0225, \quad
(1.75 - 1.75)^2 = 0.0000, \quad
(1.75 - 1.90)^2 = 0.0225
$$

代入核函數（$2\sigma^2 = 0.02$）：

$$
K_1 = e^{-0.0225/0.02} = e^{-1.125} \approx 0.3247
$$

$$
K_2 = e^{0} = 1.0000
$$

$$
K_3 = e^{-1.125} \approx 0.3247
$$

---

### 步驟二：正規化權重

$$
\sum K = 0.3247 + 1.0000 + 0.3247 = 1.6494
$$

$$
\alpha_1 = \frac{0.3247}{1.6494} \approx 0.197, \quad
\alpha_2 = \frac{1.0000}{1.6494} \approx 0.606, \quad
\alpha_3 = \frac{0.3247}{1.6494} \approx 0.197
$$

驗證：$\alpha_1 + \alpha_2 + \alpha_3 = 1$ ✓

---

### 步驟三：加權平均輸出

$$
\hat{f}(1.75)
= \alpha_1 \cdot 55 + \alpha_2 \cdot 70 + \alpha_3 \cdot 85
$$

$$
= 0.197 \times 55 + 0.606 \times 70 + 0.197 \times 85
$$

$$
= 10.84 + 42.42 + 16.74 \approx 70.0 \text{ kg}
$$

因為查詢點 $x = 1.75$ 與 $x_2 = 1.75$ 完全相同，$\alpha_2$ 最大，預測值被拉向 $y_2 = 70$。

---

### 對應到注意力機制

若將此例改寫為注意力機制的語言：

| 角色 | NW 核回歸 | 注意力機制 |
|------|-----------|------------|
| 查詢 | $x = 1.75$ | query $q$ |
| 鍵   | $x_i \in \{1.60, 1.75, 1.90\}$ | key $k_i$ |
| 值   | $y_i \in \{55, 70, 85\}$ | value $v_i$ |
| 相似度 | $\exp\!\left(-\frac{(x-x_i)^2}{2\sigma^2}\right)$ | $\exp\!\left(\frac{q^\top k_i}{\sqrt{d_k}}\right)$ |
| 正規化權重 | $\alpha_i$ | $a_i$ |
| 輸出 | $\hat{f}(x) = \sum_i \alpha_i y_i$ | $\text{Attention}(q,K,V) = \sum_i a_i v_i$ |

兩者運算流程完全一致。主要差異在於：NW 核回歸的相似度函數由人工指定（如 Gaussian kernel），而注意力機制的 query、key、value 由神經網路透過 $W_Q, W_K, W_V$ 學習而來。

---

## 十二、文字範例：根據上下文預測詞義

### 情境設定

在自然語言處理中，同一個詞在不同語境下有不同含義。以「蘋果」為例，模型需要根據上下文判斷它指的是「水果」還是「科技公司」。

假設訓練資料中有三個句子片段（已編碼為向量，這裡用簡化數字表示）：

| 資料點 | 句子片段（key） | 對應語義向量（value） |
|--------|----------------|----------------------|
| $i=1$  | 「吃了一顆蘋果」 | $v_1 = $ 水果語義（編碼為 1.0） |
| $i=2$  | 「蘋果發布新品」 | $v_2 = $ 科技語義（編碼為 5.0） |
| $i=3$  | 「蘋果樹開花了」 | $v_3 = $ 植物語義（編碼為 2.0） |

現在查詢句子（query）為：「他買了最新款的蘋果手機」

---

### 步驟一：計算 query 與每個 key 的相似度分數

使用 scaled dot-product，設相似度分數（已簡化）如下：

$$
s(q, k_1) = -2.0 \quad \text{（與「吃了一顆蘋果」差異大）}
$$

$$
s(q, k_2) = +3.0 \quad \text{（與「蘋果發布新品」最相近）}
$$

$$
s(q, k_3) = -1.0 \quad \text{（與「蘋果樹開花了」稍有差異）}
$$

---

### 步驟二：Softmax 正規化為注意力權重

$$
a_i = \frac{\exp(s(q, k_i))}{\sum_j \exp(s(q, k_j))}
$$

各項指數值：

$$
e^{-2.0} \approx 0.135, \quad e^{3.0} \approx 20.086, \quad e^{-1.0} \approx 0.368
$$

總和：

$$
\sum = 0.135 + 20.086 + 0.368 = 20.589
$$

注意力權重：

$$
a_1 \approx \frac{0.135}{20.589} \approx 0.007
$$

$$
a_2 \approx \frac{20.086}{20.589} \approx 0.976
$$

$$
a_3 \approx \frac{0.368}{20.589} \approx 0.018
$$

驗證：$a_1 + a_2 + a_3 = 1$ ✓

---

### 步驟三：加權平均輸出語義向量

$$
o = a_1 \cdot v_1 + a_2 \cdot v_2 + a_3 \cdot v_3
$$

$$
= 0.007 \times 1.0 + 0.976 \times 5.0 + 0.018 \times 2.0
$$

$$
= 0.007 + 4.880 + 0.036 \approx 4.92
$$

輸出向量 $o \approx 4.92$，非常接近科技語義（5.0），模型成功判斷此處的「蘋果」指科技公司。

---

### 直覺解釋

Softmax 具有「贏家通吃」的放大效果：分數差距不大，但指數運算後，$k_2$ 的權重從原本的相對優勢被放大為 **97.6%**，幾乎壓制其他選項。這正是注意力機制能夠聚焦在最相關上下文的原因。

對應到 Nadaraya-Watson 核回歸的語言：

- **查詢點 $x$**：待理解的句子「他買了最新款的蘋果手機」
- **資料點 $x_i$**：訓練語料中的句子片段
- **核函數 $K(x, x_i)$**：句子之間的語意相似度（由 $W_Q, W_K$ 學習）
- **輸出值 $y_i$**：對應的語義表示
- **預測 $\hat{f}(x)$**：根據上下文融合後的詞義向量

這說明注意力機制的本質就是：**用查詢句的語意，去訓練資料中找最相似的上下文，再加權融合其對應的語義表示。**

---

## 十三、完整數值範例：模擬 Scaled Dot-Product Attention

### 情境設定

輸入句子為三個 token：

$$
\text{「貓」, 「追」, 「鼠」}
$$

每個 token 已經過 embedding，得到 **2 維向量**（實務上為 512 或 768 維，此處簡化為 2 維以便手算）：

$$
x_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
x_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
x_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

寫成矩陣形式：

$$
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
$$

---

### 步驟一：定義投影矩陣

設定三個可學習矩陣（$2 \times 2$，實務中由訓練得到，此處手動給定）：

$$
W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad
W_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad
W_V = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}
$$

---

### 步驟二：計算 Q、K、V 矩陣

$$
Q = X W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
$$

$$
K = X W_K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \\ 1 & 2 \end{bmatrix}
$$

$$
V = X W_V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{bmatrix}
$$

各 token 的 query、key、value：

| Token | Query $q_i$ | Key $k_i$ | Value $v_i$ |
|-------|-------------|-----------|-------------|
| 貓 ($i=1$) | $[1, 0]$ | $[1, 1]$ | $[0, 1]$ |
| 追 ($i=2$) | $[0, 1]$ | $[0, 1]$ | $[1, 0]$ |
| 鼠 ($i=3$) | $[1, 1]$ | $[1, 2]$ | $[1, 1]$ |

---

### 步驟三：計算原始注意力分數 $QK^\top$

$$
QK^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \end{bmatrix}
= \begin{bmatrix}
1{\cdot}1+0{\cdot}1 & 1{\cdot}0+0{\cdot}1 & 1{\cdot}1+0{\cdot}2 \\
0{\cdot}1+1{\cdot}1 & 0{\cdot}0+1{\cdot}1 & 0{\cdot}1+1{\cdot}2 \\
1{\cdot}1+1{\cdot}1 & 1{\cdot}0+1{\cdot}1 & 1{\cdot}1+1{\cdot}2
\end{bmatrix}
= \begin{bmatrix}
1 & 0 & 1 \\
1 & 1 & 2 \\
2 & 1 & 3
\end{bmatrix}
$$

矩陣第 $i$ 行第 $j$ 列的元素 $= q_i^\top k_j$，表示 token $i$ 對 token $j$ 的原始注意力分數。

---

### 步驟四：Scaling（除以 $\sqrt{d_k}$）

$d_k = 2$（key 向量維度），$\sqrt{d_k} = \sqrt{2} \approx 1.414$

$$
\frac{QK^\top}{\sqrt{d_k}} = \frac{1}{1.414}\begin{bmatrix} 1 & 0 & 1 \\ 1 & 1 & 2 \\ 2 & 1 & 3 \end{bmatrix} \approx \begin{bmatrix} 0.707 & 0.000 & 0.707 \\ 0.707 & 0.707 & 1.414 \\ 1.414 & 0.707 & 2.121 \end{bmatrix}
$$

> **為何要 scaling？** 當 $d_k$ 很大時，點積值會隨維度增大而膨脹，導致 softmax 進入梯度極小的飽和區。除以 $\sqrt{d_k}$ 可穩定梯度。

---

### 步驟五：Softmax 正規化，得注意力矩陣 $A$

對每一**行**做 softmax（每個 token 的查詢對所有 key 的權重加總為 1）。

**第 1 行（「貓」查詢所有 token）：**

$$
\text{scores} = [0.707,\ 0.000,\ 0.707]
$$

$$
e^{0.707} \approx 2.028, \quad e^{0.000} = 1.000, \quad e^{0.707} \approx 2.028
$$

$$
\text{sum} = 5.056
$$

$$
a_{1,\cdot} = \left[\frac{2.028}{5.056},\ \frac{1.000}{5.056},\ \frac{2.028}{5.056}\right] \approx [0.401,\ 0.198,\ 0.401]
$$

**第 2 行（「追」查詢所有 token）：**

$$
\text{scores} = [0.707,\ 0.707,\ 1.414]
$$

$$
e^{0.707} \approx 2.028, \quad e^{0.707} \approx 2.028, \quad e^{1.414} \approx 4.113
$$

$$
\text{sum} = 8.169
$$

$$
a_{2,\cdot} = [0.248,\ 0.248,\ 0.504]
$$

**第 3 行（「鼠」查詢所有 token）：**

$$
\text{scores} = [1.414,\ 0.707,\ 2.121]
$$

$$
e^{1.414} \approx 4.113, \quad e^{0.707} \approx 2.028, \quad e^{2.121} \approx 8.337
$$

$$
\text{sum} = 14.478
$$

$$
a_{3,\cdot} = [0.284,\ 0.140,\ 0.576]
$$

完整注意力矩陣：

$$
A = \begin{bmatrix}
0.401 & 0.198 & 0.401 \\
0.248 & 0.248 & 0.504 \\
0.284 & 0.140 & 0.576
\end{bmatrix}
$$

**解讀：** $A_{ij}$ 表示 token $i$ 在生成自身輸出時，分配給 token $j$ 的注意力比例。例如「鼠」（第 3 行）對自身的注意力高達 57.6%，對「貓」為 28.4%，對「追」最低僅 14.0%。

---

### 步驟六：計算輸出 $O = AV$

$$
O = AV = \begin{bmatrix}
0.401 & 0.198 & 0.401 \\
0.248 & 0.248 & 0.504 \\
0.284 & 0.140 & 0.576
\end{bmatrix}
\begin{bmatrix}
0 & 1 \\
1 & 0 \\
1 & 1
\end{bmatrix}
$$

**第 1 行（「貓」的輸出）：**

$$
o_1 = 0.401 \cdot \begin{bmatrix}0\\1\end{bmatrix} + 0.198 \cdot \begin{bmatrix}1\\0\end{bmatrix} + 0.401 \cdot \begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix} 0.198+0.401 \\ 0.401+0.401 \end{bmatrix} = \begin{bmatrix} 0.599 \\ 0.802 \end{bmatrix}
$$

**第 2 行（「追」的輸出）：**

$$
o_2 = 0.248 \cdot \begin{bmatrix}0\\1\end{bmatrix} + 0.248 \cdot \begin{bmatrix}1\\0\end{bmatrix} + 0.504 \cdot \begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix} 0.248+0.504 \\ 0.248+0.504 \end{bmatrix} = \begin{bmatrix} 0.752 \\ 0.752 \end{bmatrix}
$$

**第 3 行（「鼠」的輸出）：**

$$
o_3 = 0.284 \cdot \begin{bmatrix}0\\1\end{bmatrix} + 0.140 \cdot \begin{bmatrix}1\\0\end{bmatrix} + 0.576 \cdot \begin{bmatrix}1\\1\end{bmatrix} = \begin{bmatrix} 0.140+0.576 \\ 0.284+0.576 \end{bmatrix} = \begin{bmatrix} 0.716 \\ 0.860 \end{bmatrix}
$$

最終輸出矩陣：

$$
O = \begin{bmatrix} 0.599 & 0.802 \\ 0.752 & 0.752 \\ 0.716 & 0.860 \end{bmatrix}
$$

---

### 與 Nadaraya-Watson 核回歸的對照

以「貓」（$i=1$）為例，其輸出等價於一次核回歸估計：

$$
o_1 = \frac{\kappa(q_1, k_1) \cdot v_1 + \kappa(q_1, k_2) \cdot v_2 + \kappa(q_1, k_3) \cdot v_3}{\kappa(q_1, k_1) + \kappa(q_1, k_2) + \kappa(q_1, k_3)}
$$

其中：

$$
\kappa(q_1, k_j) = \exp\!\left(\frac{q_1^\top k_j}{\sqrt{d_k}}\right)
$$

分子為 value 的加權和，分母為核值總和（softmax 已自動完成正規化）。這正是 Nadaraya-Watson 估計式的完整形式。

---

### 整體計算流程總結

$$
X \xrightarrow{W_Q, W_K, W_V} Q, K, V \xrightarrow{QK^\top / \sqrt{d_k}} \text{分數矩陣} \xrightarrow{\text{softmax}} A \xrightarrow{\times V} O
$$

| 步驟 | 操作 | 輸出形狀（本例） |
|------|------|-----------------|
| Embedding | 輸入 token 向量化 | $3 \times 2$ |
| 線性投影 | $Q=XW_Q,\ K=XW_K,\ V=XW_V$ | 各 $3 \times 2$ |
| 點積 | $QK^\top$ | $3 \times 3$ |
| Scaling | $\div \sqrt{d_k}$ | $3 \times 3$ |
| Softmax | 每行正規化 | $3 \times 3$（注意力矩陣） |
| 輸出 | $O = AV$ | $3 \times 2$ |

每個 token 的輸出向量，都是整個序列所有 value 向量的加權平均，權重由該 token 的 query 與所有 key 的相似度決定——這就是 self-attention 作為多點 Nadaraya-Watson 核回歸的完整體現。
