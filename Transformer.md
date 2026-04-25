# Transformer：Self-Attention 的完整形式

> 本文件將 Pre-Transformer 的基本形式：
>
> $$
> c_i = \sum_j \text{softmax}(x_i^T x_j)x_j
> $$
>
> 推廣為完整 Transformer 架構。

---

## 目錄

1. Self-Attention 的限制
2. Query / Key / Value 的動機
3. Scaled Dot-Product Attention
4. 矩陣形式與 shape
5. Multi-Head Attention
6. Transformer Block
7. Positional Encoding
8. 與 RNN 的結構差異

---

## 1. Self-Attention 的限制

原始形式：

$$
x_i^T x_j
$$

問題：

- 同一空間同時負責「查詢」與「內容」
- 表達能力不足

---

## 2. Query / Key / Value 的動機

引入三個投影：

$$
q_i = W_Q x_i,\quad
k_j = W_K x_j,\quad
v_j = W_V x_j
$$

shape：

- $X \in \mathbb{R}^{T \times d}$
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$

### 語意分離

- Query：我要找什麼
- Key：我可以被怎麼匹配
- Value：我要提供什麼資訊

---

## 3. Scaled Dot-Product Attention

### 3.1 Score

$$
e_{i,j} = q_i^T k_j
$$

### 3.2 Scaling 動機

若：

- $q_i, k_j$ 每個元素 variance = 1

則：

$$
\text{Var}(q_i^T k_j) = d_k
$$

維度越大：

- 內積越大
- softmax 越接近 one-hot（梯度消失）

因此使用：

$$
\frac{1}{\sqrt{d_k}}
$$

---

### 3.3 完整公式

$$
\alpha_{i,j} = \text{softmax}\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right)
$$

$$
c_i = \sum_j \alpha_{i,j} v_j
$$

---

## 4. 矩陣形式與 shape

$$
Q = X W_Q \in \mathbb{R}^{T \times d_k}
$$

$$
K = X W_K \in \mathbb{R}^{T \times d_k}
$$

$$
V = X W_V \in \mathbb{R}^{T \times d_v}
$$

### 計算流程

$$
E = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}
$$

$$
A = \text{softmax}(E)
$$

$$
C = A V \in \mathbb{R}^{T \times d_v}
$$

---

## 5. Multi-Head Attention

定義多組：

$$
Q^{(h)}, K^{(h)}, V^{(h)}
$$

每一 head：

$$
C^{(h)} = \text{Attention}(Q^{(h)}, K^{(h)}, V^{(h)})
$$

拼接：

$$
C = \text{Concat}(C^{(1)}, \ldots, C^{(H)}) W_O
$$

### 為什麼需要多頭？

單一 attention：

- 只能學一種相似度

多頭：

- 同時學多個子空間
- 提升表示能力

---

## 6. Transformer Block

### 6.1 Self-Attention（全域關聯）

$$
Z = \text{Attention}(Q,K,V)
$$

### 6.2 Residual + Norm

$$
Z' = \text{LayerNorm}(X + Z)
$$

### 6.3 Feedforward（局部非線性）

$$
F = \max(0, Z'W_1 + b_1)W_2 + b_2
$$

### 6.4 Residual + Norm

$$
Y = \text{LayerNorm}(Z' + F)
$$

---

## 7. Positional Encoding

因為沒有時間結構：

$$
x_i \leftarrow x_i + p_i
$$

$$
p_{i,2k} = \sin\left(\frac{i}{10000^{2k/d}}\right)
$$

$$
p_{i,2k+1} = \cos\left(\frac{i}{10000^{2k/d}}\right)
$$

---

## 8. 與 RNN 的結構差異

| 面向 | RNN | Transformer |
|------|-----|-------------|
| 計算 | sequential | parallel |
| interaction | 鄰近傳遞 | 全連接 |
| path length | $O(T)$ | $O(1)$ |

---

## 核心總結

Transformer 的核心運算為：

$$
\text{Attention}(Q,K,V)
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

本質為：

1. 相似度計算
2. 機率化
3. 加權讀取
4. 多頭擴展
5. 層級堆疊

## 9. Self-Attention 的反向傳播推導

我們考慮單一 attention（不含 multi-head），其前向傳播為：

$$
E = \frac{QK^T}{\sqrt{d_k}}
$$

$$
A = \text{softmax}(E)
$$

$$
C = A V
$$

目標是推導：

$$
\frac{\partial \mathcal{L}}{\partial Q},\quad
\frac{\partial \mathcal{L}}{\partial K},\quad
\frac{\partial \mathcal{L}}{\partial V}
$$

---

## 9.1 記號定義

令：

- $Q \in \mathbb{R}^{T \times d_k}$
- $K \in \mathbb{R}^{T \times d_k}$
- $V \in \mathbb{R}^{T \times d_v}$
- $E \in \mathbb{R}^{T \times T}$
- $A \in \mathbb{R}^{T \times T}$
- $C \in \mathbb{R}^{T \times d_v}$

令：

$$
G^C = \frac{\partial \mathcal{L}}{\partial C}
$$

---

## 9.2 對 V 的梯度

由：

$$
C = A V
$$

可得：

$$
\frac{\partial \mathcal{L}}{\partial V}
= A^T G^C
$$

推導：

$$
C_i = \sum_j A_{i,j} V_j
$$

$$
\Rightarrow \frac{\partial \mathcal{L}}{\partial V_j}
= \sum_i A_{i,j} G^C_i
$$

---

## 9.3 對 A 的梯度

仍由：

$$
C = A V
$$

可得：

$$
\frac{\partial \mathcal{L}}{\partial A}
= G^C V^T
$$

元素形式：

$$
\frac{\partial \mathcal{L}}{\partial A_{i,j}}
= (G^C_i)^T V_j
$$

---

## 9.4 Softmax 的梯度

對於每一列 $i$：

$$
A_{i,:} = \text{softmax}(E_{i,:})
$$

softmax Jacobian：

$$
\frac{\partial A_{i,j}}{\partial E_{i,k}}
= A_{i,j}(\mathbf{1}_{j=k} - A_{i,k})
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial E_{i,k}}
= \sum_j \frac{\partial \mathcal{L}}{\partial A_{i,j}}
\frac{\partial A_{i,j}}{\partial E_{i,k}}
$$

整理得：

$$
\frac{\partial \mathcal{L}}{\partial E_{i,k}}
= A_{i,k}\left(
G^A_{i,k} - \sum_j A_{i,j} G^A_{i,j}
\right)
$$

其中：

$$
G^A = \frac{\partial \mathcal{L}}{\partial A}
$$

---

## 9.5 對 E 的矩陣形式

定義：

$$
G^E = \frac{\partial \mathcal{L}}{\partial E}
$$

則 row-wise 計算：

$$
G^E_i = \text{softmax-gradient}(G^A_i, A_i)
$$

---

## 9.6 對 Q 與 K 的梯度

回到：

$$
E = \frac{QK^T}{\sqrt{d_k}}
$$

### 對 Q：

$$
\frac{\partial \mathcal{L}}{\partial Q}
= \frac{1}{\sqrt{d_k}} G^E K
$$

### 對 K：

$$
\frac{\partial \mathcal{L}}{\partial K}
= \frac{1}{\sqrt{d_k}} (G^E)^T Q
$$

---

### 推導（對 Q）

$$
E_{i,j} = \frac{q_i^T k_j}{\sqrt{d_k}}
$$

$$
\frac{\partial E_{i,j}}{\partial q_i}
= \frac{k_j}{\sqrt{d_k}}
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial q_i}
= \sum_j G^E_{i,j} \frac{k_j}{\sqrt{d_k}}
$$

即：

$$
\frac{\partial \mathcal{L}}{\partial Q}
= \frac{1}{\sqrt{d_k}} G^E K
$$

---

## 9.7 梯度流總結

整體梯度路徑為：

$$
\mathcal{L}
\rightarrow C
\rightarrow A
\rightarrow E
\rightarrow Q, K
$$

以及：

$$
\mathcal{L}
\rightarrow C
\rightarrow V
$$

---

## 10. Linear Projection 的反向傳播

回到：

$$
Q = X W_Q,\quad
K = X W_K,\quad
V = X W_V
$$

### 對權重：

$$
\frac{\partial \mathcal{L}}{\partial W_Q}
= X^T \frac{\partial \mathcal{L}}{\partial Q}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_K}
= X^T \frac{\partial \mathcal{L}}{\partial K}
$$

$$
\frac{\partial \mathcal{L}}{\partial W_V}
= X^T \frac{\partial \mathcal{L}}{\partial V}
$$

### 對輸入：

$$
\frac{\partial \mathcal{L}}{\partial X}
= \frac{\partial \mathcal{L}}{\partial Q} W_Q^T
+ \frac{\partial \mathcal{L}}{\partial K} W_K^T
+ \frac{\partial \mathcal{L}}{\partial V} W_V^T
$$

---

## 11. Multi-Head Attention 的梯度

每個 head 獨立：

$$
\frac{\partial \mathcal{L}}{\partial C^{(h)}}
$$

經 concat：

$$
C = \text{Concat}(C^{(1)}, \ldots, C^{(H)})
$$

因此：

- 梯度直接分配回各 head
- 再各自套用單頭推導

---

## 12. 與 RNN Attention 的關鍵差異（反向傳播）

RNN Attention：

- 梯度需經過時間鏈：

$$
\prod \frac{\partial h_t}{\partial h_{t-1}}
$$

Transformer：

- 任意兩 token：

$$
x_i \rightarrow x_j
$$

只有一層 attention：

- path length = 1

因此：

- 梯度不會因時間長度衰減
- 長距離依賴更容易學習

---

## 核心結論

Self-Attention 的反向傳播本質為：

1. matrix multiplication 的梯度（$AV$）
2. softmax Jacobian
3. dot-product 的線性結構

最終關鍵結果：

$$
\frac{\partial \mathcal{L}}{\partial Q}
= \frac{1}{\sqrt{d_k}} G^E K
$$

$$
\frac{\partial \mathcal{L}}{\partial K}
= \frac{1}{\sqrt{d_k}} (G^E)^T Q
$$

$$
\frac{\partial \mathcal{L}}{\partial V}
= A^T G^C
$$

這三式構成 Transformer 訓練的核心梯度。

## 13. LayerNorm 的反向傳播推導

LayerNorm 常出現在 Transformer block 中：

$$
Y = \text{LayerNorm}(X)
$$

對每一個 token 的 hidden vector 獨立做 normalization。

假設某一個 token 的輸入為：

$$
x \in \mathbb{R}^{d}
$$

LayerNorm 定義為：

$$
\mu = \frac{1}{d}\sum_{j=1}^{d}x_j
$$

$$
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^2
$$

$$
\hat{x}_j = \frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$$
y_j = \gamma_j \hat{x}_j + \beta_j
$$

其中：

- $d$ 是 hidden dimension
- $\gamma$ 是 scale 參數
- $\beta$ 是 shift 參數
- $\epsilon$ 是避免除以零的小常數

---

## 13.1 記號定義

令 loss 對輸出的梯度為：

$$
g^y_j = \frac{\partial \mathcal{L}}{\partial y_j}
$$

我們要推導：

$$
\frac{\partial \mathcal{L}}{\partial x_j},\quad
\frac{\partial \mathcal{L}}{\partial \gamma_j},\quad
\frac{\partial \mathcal{L}}{\partial \beta_j}
$$

---

## 13.2 對 $\gamma$ 與 $\beta$ 的梯度

由：

$$
y_j = \gamma_j \hat{x}_j + \beta_j
$$

可得：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_j}
= g^y_j \hat{x}_j
$$

$$
\frac{\partial \mathcal{L}}{\partial \beta_j}
= g^y_j
$$

若對 batch 與 sequence 所有 token 累加：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_j}
= \sum_{b,t} g^y_{b,t,j}\hat{x}_{b,t,j}
$$

$$
\frac{\partial \mathcal{L}}{\partial \beta_j}
= \sum_{b,t} g^y_{b,t,j}
$$

---

## 13.3 先傳回 normalized vector

因為：

$$
y_j = \gamma_j \hat{x}_j + \beta_j
$$

所以：

$$
g^{\hat{x}}_j
=
\frac{\partial \mathcal{L}}{\partial \hat{x}_j}
=
g^y_j\gamma_j
$$

接下來只需要推導：

$$
\hat{x}_j = \frac{x_j-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

對 $x_j$ 的梯度。

---

## 13.4 簡化記號

令：

$$
r = \sqrt{\sigma^2+\epsilon}
$$

則：

$$
\hat{x}_j = \frac{x_j-\mu}{r}
$$

也就是：

$$
\hat{x}_j = (x_j-\mu)r^{-1}
$$

---

## 13.5 對輸入 $x$ 的梯度核心公式

LayerNorm 對輸入的反向傳播結果為：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
=
\frac{1}{r}
\left(
g^{\hat{x}}_j
-
\frac{1}{d}\sum_{k=1}^{d}g^{\hat{x}}_k
-
\hat{x}_j
\frac{1}{d}\sum_{k=1}^{d}g^{\hat{x}}_k\hat{x}_k
\right)
$$

這是 LayerNorm 最重要的反向傳播公式。

---

## 13.6 推導：為什麼會有三項？

LayerNorm 的輸入 $x_j$ 會透過三條路徑影響 loss：

1. 直接影響自己的 $x_j-\mu$
2. 影響平均值 $\mu$
3. 影響 variance $\sigma^2$

因此：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
=
\text{direct term}
+
\text{mean term}
+
\text{variance term}
$$

---

## 13.7 從中心化變數開始

定義：

$$
\tilde{x}_j = x_j-\mu
$$

則：

$$
\hat{x}_j = \tilde{x}_j r^{-1}
$$

其中：

$$
\sigma^2 = \frac{1}{d}\sum_{k=1}^{d}\tilde{x}_k^2
$$

---

## 13.8 對 $\tilde{x}$ 的梯度

由：

$$
\hat{x}_j = \tilde{x}_j r^{-1}
$$

先忽略 $\mu$ 對所有維度的耦合，對 $\tilde{x}$ 求導。

已知：

$$
g^{\hat{x}}_j = \frac{\partial \mathcal{L}}{\partial \hat{x}_j}
$$

第一項直接路徑：

$$
g^{\tilde{x}}_j\Big|_{direct}
=
g^{\hat{x}}_j r^{-1}
$$

---

## 13.9 variance 路徑

因為：

$$
r = \sqrt{\sigma^2+\epsilon}
$$

且：

$$
\hat{x}_j = \tilde{x}_j r^{-1}
$$

所以：

$$
\frac{\partial \hat{x}_j}{\partial \sigma^2}
=
\tilde{x}_j \cdot \frac{\partial r^{-1}}{\partial \sigma^2}
$$

又：

$$
r^{-1} = (\sigma^2+\epsilon)^{-1/2}
$$

因此：

$$
\frac{\partial r^{-1}}{\partial \sigma^2}
=
-\frac{1}{2}(\sigma^2+\epsilon)^{-3/2}
=
-\frac{1}{2}r^{-3}
$$

所以：

$$
\frac{\partial \hat{x}_j}{\partial \sigma^2}
=
-\frac{1}{2}\tilde{x}_j r^{-3}
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial \sigma^2}
=
\sum_{j=1}^{d}
g^{\hat{x}}_j
\left(
-\frac{1}{2}\tilde{x}_j r^{-3}
\right)
$$

記：

$$
g^{\sigma^2}
=
\frac{\partial \mathcal{L}}{\partial \sigma^2}
=
-\frac{1}{2}r^{-3}
\sum_{j=1}^{d}g^{\hat{x}}_j\tilde{x}_j
$$

又因為：

$$
\sigma^2 = \frac{1}{d}\sum_{j=1}^{d}\tilde{x}_j^2
$$

所以：

$$
\frac{\partial \sigma^2}{\partial \tilde{x}_j}
=
\frac{2}{d}\tilde{x}_j
$$

因此 variance 路徑傳回：

$$
g^{\tilde{x}}_j\Big|_{var}
=
g^{\sigma^2}\frac{2}{d}\tilde{x}_j
$$

---

## 13.10 合併對 $\tilde{x}$ 的梯度

因此：

$$
g^{\tilde{x}}_j
=
g^{\hat{x}}_j r^{-1}
+
g^{\sigma^2}\frac{2}{d}\tilde{x}_j
$$

---

## 13.11 mean 路徑

因為：

$$
\tilde{x}_j = x_j-\mu
$$

且：

$$
\mu = \frac{1}{d}\sum_{k=1}^{d}x_k
$$

所以：

$$
\frac{\partial \tilde{x}_k}{\partial x_j}
=
\mathbf{1}_{j=k} - \frac{1}{d}
$$

因此：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
=
\sum_{k=1}^{d}
g^{\tilde{x}}_k
\left(
\mathbf{1}_{j=k} - \frac{1}{d}
\right)
$$

整理得：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
=
g^{\tilde{x}}_j
-
\frac{1}{d}\sum_{k=1}^{d}g^{\tilde{x}}_k
$$

---

## 13.12 化成常用閉式公式

由於：

$$
\tilde{x}_j = r\hat{x}_j
$$

且：

$$
g^{\sigma^2}
=
-\frac{1}{2}r^{-3}
\sum_{k=1}^{d}g^{\hat{x}}_k\tilde{x}_k
$$

代入：

$$
g^{\tilde{x}}_j
=
g^{\hat{x}}_j r^{-1}
+
g^{\sigma^2}\frac{2}{d}\tilde{x}_j
$$

得到：

$$
g^{\tilde{x}}_j
=
\frac{1}{r}
\left(
g^{\hat{x}}_j
-
\frac{1}{d}
\hat{x}_j
\sum_{k=1}^{d}g^{\hat{x}}_k\hat{x}_k
\right)
$$

再代入 mean 路徑：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
=
g^{\tilde{x}}_j
-
\frac{1}{d}\sum_{k=1}^{d}g^{\tilde{x}}_k
$$

最後得到：

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial x_j}
=
\frac{1}{r}
\left(
g^{\hat{x}}_j
-
\frac{1}{d}\sum_{k=1}^{d}g^{\hat{x}}_k
-
\hat{x}_j
\frac{1}{d}\sum_{k=1}^{d}g^{\hat{x}}_k\hat{x}_k
\right)
}
$$

其中：

$$
r = \sqrt{\sigma^2+\epsilon}
$$

---

## 13.13 向量形式

令：

$$
g^{\hat{x}} =
\frac{\partial \mathcal{L}}{\partial \hat{x}}
$$

則：

$$
\frac{\partial \mathcal{L}}{\partial x}
=
\frac{1}{r}
\left(
g^{\hat{x}}
-
\text{mean}(g^{\hat{x}})
-
\hat{x}\odot \text{mean}(g^{\hat{x}}\odot \hat{x})
\right)
$$

其中 mean 是沿 hidden dimension 計算。

---

## 13.14 放回 Transformer Block

在 Transformer 中常見結構為：

$$
Z' = \text{LayerNorm}(X+Z)
$$

令：

$$
U = X + Z
$$

則：

$$
Z' = \text{LayerNorm}(U)
$$

若已知：

$$
G^{Z'} = \frac{\partial \mathcal{L}}{\partial Z'}
$$

先經 LayerNorm 得到：

$$
G^U = \frac{\partial \mathcal{L}}{\partial U}
$$

再由 residual connection：

$$
U = X + Z
$$

所以：

$$
\frac{\partial \mathcal{L}}{\partial X}\Big|_{res}
=
G^U
$$

$$
\frac{\partial \mathcal{L}}{\partial Z}
=
G^U
$$

也就是說，residual connection 會讓梯度直接分流回原輸入與 attention 輸出。

---

## 13.15 LayerNorm 反向傳播總結

LayerNorm 的反向傳播包含三個核心部分：

1. scale / shift 參數梯度：

$$
\frac{\partial \mathcal{L}}{\partial \gamma}
=
g^y \odot \hat{x}
$$

$$
\frac{\partial \mathcal{L}}{\partial \beta}
=
g^y
$$

2. normalized vector 梯度：

$$
g^{\hat{x}} = g^y \odot \gamma
$$

3. input 梯度：

$$
\frac{\partial \mathcal{L}}{\partial x}
=
\frac{1}{r}
\left(
g^{\hat{x}}
-
\text{mean}(g^{\hat{x}})
-
\hat{x}\odot \text{mean}(g^{\hat{x}}\odot \hat{x})
\right)
$$

其中：

$$
r = \sqrt{\sigma^2+\epsilon}
$$

---

## 核心結論

LayerNorm 的反向傳播難點在於：

$$
\mu
\quad\text{與}\quad
\sigma^2
$$

都由同一個 hidden vector 的所有維度共同決定。

因此每個輸入維度 $x_j$ 的梯度，不只來自自己的輸出 $y_j$，也會受到所有其他維度影響。

這使得 LayerNorm 的梯度不是單純的 element-wise operation，而是 hidden dimension 內部的耦合運算。