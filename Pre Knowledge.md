# Pre-Transformer：Self-Attention 的數學前置

> 本文件的目標不是介紹工具，而是回答一個核心問題：
>
> **如何讓一個 token 從整個序列中選擇性讀取資訊？**
>
> 最終我們希望得到：
>
> $$
> c_i = \sum_{j=1}^{T} \alpha_{i,j} x_j
> $$
>
> 其中 $\alpha_{i,j}$ 是由資料自動學出的權重分佈。

---

## 目錄

1. 問題形式化
2. 向量表示（Embedding）
3. 相似度作為匹配函數
4. Softmax：從分數到分佈
5. 加權平均：資訊讀取
6. 矩陣化表示
7. 總結公式

---

## 1. 問題形式化

對於序列：

$$
X = (x_1, x_2, \ldots, x_T), \quad x_i \in \mathbb{R}^d
$$

我們希望每個位置 $i$ 可以：

- 看整個序列
- 選擇重要部分
- 產生新的表示 $c_i$

---

## 2. 向量表示（Embedding）

每個 token 被表示為：

$$
x_i \in \mathbb{R}^d
$$

整個序列：

$$
X \in \mathbb{R}^{T \times d}
$$

幾何意義：

- 向量距離 → 語意差異
- 向量方向 → 語意特徵

---

## 3. 相似度作為匹配函數

我們需要一個函數：

$$
f(x_i, x_j)
$$

來衡量「$i$ 是否應該關注 $j$」。

最簡單選擇是內積：

$$
e_{i,j} = x_i^T x_j
$$

性質：

$$
x_i^T x_j = \|x_i\|\|x_j\|\cos\theta
$$

因此：

- 大 → 相似
- 小 → 不相關

---

## 4. Softmax：從分數到分佈

原始分數：

$$
e_{i,1}, e_{i,2}, \ldots, e_{i,T}
$$

轉為機率分佈：

$$
\alpha_{i,j} = \frac{\exp(e_{i,j})}{\sum_{k=1}^T \exp(e_{i,k})}
$$

### 為什麼需要 softmax？

若直接使用 $e_{i,j}$：

- 無法保證總和為 1
- scale 不穩定

softmax 提供：

- normalization
- 平滑選擇（soft selection）

---

## 5. 加權平均：資訊讀取

定義：

$$
c_i = \sum_{j=1}^T \alpha_{i,j} x_j
$$

語意：

- $x_j$ 是記憶
- $\alpha_{i,j}$ 是讀取權重
- $c_i$ 是讀取結果

---

## 6. 矩陣化表示

令：

$$
X \in \mathbb{R}^{T \times d}
$$

則：

### 相似度矩陣

$$
E = X X^T \in \mathbb{R}^{T \times T}
$$

### 權重矩陣

$$
A = \text{softmax}(E)
$$

（row-wise）

### 輸出

$$
C = A X
$$

---

## 7. 最小例子（3 tokens）

假設：

$$
X =
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix}
$$

則：

$$
E =
\begin{bmatrix}
x_1^T x_1 & x_1^T x_2 & x_1^T x_3 \\
x_2^T x_1 & x_2^T x_2 & x_2^T x_3 \\
x_3^T x_1 & x_3^T x_2 & x_3^T x_3
\end{bmatrix}
$$

每一列代表：

- 該 token 對所有 token 的關注程度

---

## 8. 核心總結公式

Self-Attention 的原型：

$$
c_i = \sum_j \text{softmax}(x_i^T x_j) \cdot x_j
$$

之後 Transformer 只是將：

- $x_i^T x_j$ → 改成更一般的 QK
- $x_j$ → 改成 V