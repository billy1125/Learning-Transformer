# 01b｜數學前置：Self-Attention 是什麼（數學版）

> **適合對象：** 大學數學程度，熟悉線性代數（矩陣乘法、轉置）與基礎機率（期望值、變異數）。
>
> **讀完後你能做什麼：**
> - 從第一性原理推導 $C = \text{softmax}(XX^\top)X$
> - 用統計學解釋為什麼要除以 $\sqrt{d}$（與 z-score 的關係）
> - 寫出 Scaled Dot-Product Attention 完整公式並說明每個參數的形狀
>
> **預備知識：** 線性代數（矩陣乘法、轉置）、基礎微積分（導數概念）。如果沒有這些背景，請先讀 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md)。
>
> **學完後的下一步：** → [`02-attention-intuition.md`](02-attention-intuition.md)（注意力的直覺）

---

> 本文件的目標不是介紹工具，而是回答一個核心問題：
>
> **如何讓一個 token 從整個序列中選擇性讀取資訊？**
>
> 最終我們希望得到：
>
> $$
> c_i = \sum_{j=1}^{T} \alpha_{i,j} \, x_j
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
7. 最小例子（3 tokens）
8. 數值穩定性：Temperature Scaling 與維度縮放
9. 核心總結公式
10. 從 Naive Self-Attention 到 QKV 的動機

---

## 1. 問題形式化

對於序列：

$$
X = (x_1, x_2, \ldots, x_T), \quad x_i \in \mathbb{R}^d
$$

我們希望每個位置 $i$ 可以：

- 看整個序列（全局視野）
- 依照相關性選擇重要部分（選擇性讀取）
- 產生新的表示 $c_i$，融合了序列中的上下文資訊

**符號約定：**

| 符號 | 意義 |
|---|---|
| $T$ | 序列長度（token 數量）|
| $d$ | 每個 token 的向量維度 |
| $x_i \in \mathbb{R}^d$ | 第 $i$ 個 token 的向量 |
| $X \in \mathbb{R}^{T \times d}$ | 整個序列矩陣 |
| $c_i \in \mathbb{R}^d$ | 第 $i$ 個 token 的輸出（上下文表示）|

---

## 2. 向量表示（Embedding）

每個 token 被表示為一個 $d$ 維向量：

$$
x_i \in \mathbb{R}^d
$$

整個序列堆疊成矩陣：

$$
X =
\begin{bmatrix}
— x_1 — \\
— x_2 — \\
\vdots \\
— x_T —
\end{bmatrix}
\in \mathbb{R}^{T \times d}
$$

### 幾何意義

向量空間賦予語意幾何結構：

- **向量距離** → 語意差異：語意相近的詞，歐式距離較小
- **向量方向** → 語意特徵：不同方向編碼不同語言特性

**經典例子：** Word2Vec 中觀察到

$$
\text{vec}(\text{King}) - \text{vec}(\text{Man}) + \text{vec}(\text{Woman}) \approx \text{vec}(\text{Queen})
$$

這說明向量空間中的**線性關係**可以捕捉語意類比。

### 為什麼需要 Embedding？

原始文字（字元或詞語）是離散符號，無法直接進行數學運算。Embedding 將離散符號映射到連續空間，使得：

1. 可以計算符號之間的相似度
2. 神經網路可以對其進行梯度更新
3. 語意相近的符號在向量空間中彼此靠近

Embedding 在實作上是一張可訓練的查找表，數學細節見附錄 D。

---

## 3. 相似度作為匹配函數

我們需要一個函數：

$$
f(x_i, x_j) \in \mathbb{R}
$$

來衡量「位置 $i$ 應該關注位置 $j$ 的程度」。

### 3.1 內積（Dot Product）

最簡單的選擇是**內積**：

$$
e_{i,j} = x_i^\top x_j = \sum_{k=1}^d (x_i)_k \cdot (x_j)_k
$$

利用向量夾角公式展開：

$$
x_i^\top x_j = \|x_i\| \, \|x_j\| \cos\theta_{ij}
$$

因此：

| 情況 | $\cos\theta$ | $e_{i,j}$ | 語意 |
|---|---|---|---|
| 方向相同 | $\approx 1$ | 大正值 | 高度相似 |
| 方向垂直 | $= 0$ | $0$ | 無關 |
| 方向相反 | $\approx -1$ | 大負值 | 語意相反 |

### 3.2 其他相似度函數

內積只是眾多選擇之一。常見的相似度函數包括：

**餘弦相似度（Cosine Similarity）：**

$$
\text{cos}(x_i, x_j) = \frac{x_i^\top x_j}{\|x_i\| \, \|x_j\|}
$$

消除了向量長度的影響，只看方向。

**縮放內積（Scaled Dot Product）：**

$$
e_{i,j} = \frac{x_i^\top x_j}{\sqrt{d}}
$$

除以 $\sqrt{d}$ 防止高維時內積過大（詳見第 8 節）。

**加性注意力（Additive Attention，Bahdanau 2015）：**

$$
e_{i,j} = v^\top \tanh(W_1 x_i + W_2 x_j)
$$

其中 $v, W_1, W_2$ 是可學習參數，表達能力更強，但計算成本較高。

相似度分數是任意實數，直接當作注意力權重沒有語意——需要 Softmax 把它轉成非負且加總為 1 的機率分佈。

---

## 4. Softmax：從分數到分佈

對位置 $i$ 而言，我們已計算出它對序列中每個位置 $j$ 的原始分數：

$$
e_{i,1}, \quad e_{i,2}, \quad \ldots, \quad e_{i,T}
$$

我們希望將這組分數轉為**機率分佈**：

$$
\alpha_{i,j} = \frac{\exp(e_{i,j})}{\displaystyle\sum_{k=1}^T \exp(e_{i,k})}
$$

### 4.1 為什麼需要 Softmax？

若直接使用原始分數 $e_{i,j}$ 作為權重：

- **無歸一化**：$\sum_j e_{i,j} \neq 1$，加權平均失去語意
- **尺度不穩定**：不同層、不同輸入的分數尺度差異極大
- **無法表達「忽略」**：負值分數含義模糊

Softmax 提供：

| 性質 | 數學保證 |
|---|---|
| 非負性 | $\alpha_{i,j} > 0$（因 $\exp > 0$）|
| 歸一化 | $\sum_j \alpha_{i,j} = 1$ |
| 單調性 | $e_{i,j} > e_{i,k} \Rightarrow \alpha_{i,j} > \alpha_{i,k}$ |
| 平滑選擇（soft selection）| 非 hard argmax，梯度可流動 |

關於 softmax 相較於直接線性歸一化的直覺優勢，見 [`01a-prerequisites-intuition.md`](01a-prerequisites-intuition.md) §4。

### 4.2 Softmax 的溫度效應

引入溫度參數 $\tau > 0$：

$$
\alpha_{i,j} = \frac{\exp(e_{i,j}/\tau)}{\displaystyle\sum_{k=1}^T \exp(e_{i,k}/\tau)}
$$

| 溫度 $\tau$ | 效果 |
|---|---|
| $\tau \to 0^+$ | 趨近 hard max，集中在最高分 |
| $\tau = 1$ | 標準 softmax |
| $\tau \to \infty$ | 趨近均勻分佈，$\alpha_{i,j} \to 1/T$ |

在 Transformer 中，縮放因子 $\frac{1}{\sqrt{d_k}}$ 起到類似調溫的作用。

### 4.3 數值穩定的 Softmax

直接計算 $\exp(e_{i,j})$ 在 $e_{i,j}$ 很大時會數值溢位（overflow）。

**穩定版本：** 減去最大值再取 exp

$$
\alpha_{i,j} = \frac{\exp(e_{i,j} - m_i)}{\displaystyle\sum_{k=1}^T \exp(e_{i,k} - m_i)}, \quad m_i = \max_k e_{i,k}
$$

**正確性驗證：**

$$
\frac{\exp(e_{i,j} - m_i)}{\sum_k \exp(e_{i,k} - m_i)}
= \frac{\exp(e_{i,j}) \cdot \exp(-m_i)}{\sum_k \exp(e_{i,k}) \cdot \exp(-m_i)}
= \frac{\exp(e_{i,j})}{\sum_k \exp(e_{i,k})}
$$

結果與原始公式等價，但數值安全。

---

## 5. 加權平均：資訊讀取

有了權重分佈 $\{\alpha_{i,j}\}$ 之後，位置 $i$ 的輸出為：

$$
c_i = \sum_{j=1}^T \alpha_{i,j} \, x_j
$$

### 語意解讀

可以把這個操作理解為一個**記憶讀取機制**：

| 角色 | 符號 | 意義 |
|---|---|---|
| 記憶庫 | $\{x_j\}$ | 序列中所有 token 的資訊 |
| 讀取鍵 | $x_i$ | 當前 token 用來查詢的向量 |
| 注意力權重 | $\alpha_{i,j}$ | 從 $x_j$ 讀取多少資訊 |
| 讀取結果 | $c_i$ | 融合上下文後的新表示 |

### 特殊情形分析

**情形 1：均勻分佈** $\alpha_{i,j} = \frac{1}{T}$

$$
c_i = \frac{1}{T} \sum_{j=1}^T x_j
$$

這就是所有 token 的平均值，無選擇性，等同於 Bag-of-Words。

**情形 2：單點集中** $\alpha_{i,k} = 1$，其餘 $= 0$

$$
c_i = x_k
$$

這等同於直接複製位置 $k$ 的表示，完全選擇性讀取。

**情形 3：一般情況** — 在上述兩個極端之間，根據相關性做加權混合。

逐個位置計算可以理解原理，但真實序列有幾百到幾千個 token——矩陣形式讓 GPU 可以一次並行完成所有計算。

---

## 6. 矩陣化表示

將上述運算整合為高效的矩陣形式，便於 GPU 並行計算。

設：

$$
X \in \mathbb{R}^{T \times d}
$$

### Step 1：相似度矩陣

$$
E = X X^\top \in \mathbb{R}^{T \times T}
$$

其中 $E_{ij} = x_i^\top x_j$，計算所有 token 對之間的內積。

矩陣展開：

$$
E =
\begin{bmatrix}
x_1^\top x_1 & x_1^\top x_2 & \cdots & x_1^\top x_T \\
x_2^\top x_1 & x_2^\top x_2 & \cdots & x_2^\top x_T \\
\vdots & \vdots & \ddots & \vdots \\
x_T^\top x_1 & x_T^\top x_2 & \cdots & x_T^\top x_T
\end{bmatrix}
$$

注意 $E$ 是**對稱矩陣**：$E_{ij} = E_{ji}$（因此 naive self-attention 的注意力是雙向對稱的）。

### Step 2：注意力權重矩陣

對 $E$ 的每一行做 softmax：

$$
A = \text{softmax}_{\text{row}}(E) \in \mathbb{R}^{T \times T}
$$

其中 $A_{ij} = \alpha_{i,j}$，每一行總和為 1：

$$
\sum_{j=1}^T A_{ij} = 1, \quad \forall i
$$

### Step 3：輸出矩陣

$$
C = A X \in \mathbb{R}^{T \times d}
$$

每一行 $C_i = \sum_j A_{ij} x_j = c_i$，即位置 $i$ 的上下文輸出。

### 完整流程總結

$$
\boxed{C = \text{softmax}(X X^\top) \, X}
$$

---

## 7. 最小例子（3 tokens）

假設 $T = 3$，$d = 2$，具體數值如下：

$$
x_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
x_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
x_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

$$
X =
\begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

### Step 1：計算 $E = XX^\top$

$$
E = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
=
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}
$$

### Step 2：對每行做 softmax

以第 3 行為例，$e_{3,\cdot} = [1, 1, 2]$：

$$
\alpha_{3,1} = \frac{e^1}{e^1 + e^1 + e^2} = \frac{e}{2e + e^2} \approx 0.212
$$

$$
\alpha_{3,2} = \frac{e^1}{e^1 + e^1 + e^2} \approx 0.212
$$

$$
\alpha_{3,3} = \frac{e^2}{e^1 + e^1 + e^2} \approx 0.576
$$

可以看出：$x_3$ 對自身的關注最高（因為 $x_3$ 與自身的內積最大），對 $x_1, x_2$ 的關注對稱且較低。

### Step 3：計算輸出

$$
c_3 = 0.212 \cdot \begin{bmatrix} 1 \\ 0 \end{bmatrix}
+ 0.212 \cdot \begin{bmatrix} 0 \\ 1 \end{bmatrix}
+ 0.576 \cdot \begin{bmatrix} 1 \\ 1 \end{bmatrix}
= \begin{bmatrix} 0.788 \\ 0.788 \end{bmatrix}
$$

$c_3$ 是三個向量的加權混合，偏向 $x_3$ 本身，並整合了 $x_1, x_2$ 的資訊。

### 完整結果（所有三行）

對第 1、2 行做同樣的計算（$e_{1,\cdot} = [1, 0, 1]$、$e_{2,\cdot} = [0, 1, 1]$），得到完整的注意力矩陣與輸出矩陣：

$$
A = \begin{bmatrix} 0.422 & 0.155 & 0.422 \\ 0.155 & 0.422 & 0.422 \\ 0.212 & 0.212 & 0.576 \end{bmatrix}, \qquad
C = AX = \begin{bmatrix} 0.844 & 0.577 \\ 0.577 & 0.844 \\ 0.788 & 0.788 \end{bmatrix}
$$

兩個觀察：

1. **每一行加總為 1**（softmax 的歸一化保證），對應 $\sum_j A_{ij} = 1$。
2. **$c_1$ 與 $c_2$ 互為鏡像**（$[0.844, 0.577]$ vs $[0.577, 0.844]$）——這來自 $E$ 的對稱性：$x_1$ 與 $x_2$ 在這個例子中地位完全對稱，naive self-attention 無法打破這種對稱（第 9 節的限制一）。

---

## 8. 數值穩定性：Temperature Scaling 與維度縮放

第 7 節示範了用內積計算相似度、再套 softmax 得到注意力權重的完整流程。softmax 的歸一化確保每一行的權重加總等於 1，讓 $c_i$ 成為名副其實的加權平均。但歸一化本身帶來一個隱憂：**softmax 的歸一化是「你多我必少」的零和競爭**——只要有一個分數特別大，$e^{\text{大數}}$ 在分母中佔壓倒性比例，其他 token 的權重就被擠壓到幾乎為零。換言之，權重分配從「多個 token 各貢獻一點」退化成「某一個 token 獨佔幾乎全部注意力」。這在第 7 節的低維小範例裡不成問題，因為分數的數值差距小；問題出在高維：**維度越高，內積的典型幅度越大，這種傾斜就越嚴重**。

### 問題：高維時內積爆炸

**先用數字感受現象。** 假設向量各分量的典型大小約為 $O(1)$（例如初始化後的 embedding 向量）。在一維時（$d=1$），兩個分量相乘大約是 $O(1)$。但在 $d=64$ 維時，內積是 64 個這樣的乘積之和——哪怕每一項都很小，加總後的典型幅度已是 $O(\sqrt{64})=8$；到了 $d=512$，典型幅度變成 $O(\sqrt{512})\approx 22$。

內積變大本身不是問題，問題在於 **softmax 對輸入差距極其敏感**。當分數幅度從個位數放大到幾十，softmax 的輸出從分散分佈急速走向近似 argmax：

| 注意力分數向量 | softmax 輸出 |
|---|---|
| $[1,\ 2,\ 3]$ | $[0.09,\ 0.24,\ 0.67]$ |
| $[10,\ 20,\ 30]$ | $[\approx 0,\ \approx 0,\ \approx 1]$ |

前者三個 token 都有機會貢獻；後者幾乎只有第三個 token 有效，其餘兩個的權重已接近零。兩組分數的**相對差距比例完全相同**（1:2:3），造成結果天差地遠的唯一原因是絕對幅度。**低維時分數差距小，接近第一行；高維時分數差距大，接近第二行。** 這就是「內積爆炸」的本質：**維度越高，內積典型幅度越大，softmax 越容易飽和**。

**正式推導其幅度。** 設 $x_i, x_j \in \mathbb{R}^d$，各分量 i.i.d. 來自 $\mathcal{N}(0, 1)$，則：

$$
\mathbb{E}[x_i^\top x_j] = \sum_{k=1}^d \mathbb{E}[(x_i)_k (x_j)_k] = \sum_{k=1}^d \underbrace{\mathbb{E}[(x_i)_k]}_0 \cdot \underbrace{\mathbb{E}[(x_j)_k]}_0 = 0
$$

每一項的變異數（利用獨立性與 $\mathbb{E}[z^2]=1$ for $z \sim \mathcal{N}(0,1)$）：

$$
\text{Var}[(x_i)_k (x_j)_k]
= \mathbb{E}\!\left[((x_i)_k)^2 ((x_j)_k)^2\right] - \left(\mathbb{E}[(x_i)_k (x_j)_k]\right)^2
= \underbrace{\mathbb{E}\!\left[((x_i)_k)^2\right]}_{=1} \cdot \underbrace{\mathbb{E}\!\left[((x_j)_k)^2\right]}_{=1} - 0 = 1
$$

因此 $d$ 個獨立項的變異數相加：

$$
\text{Var}[x_i^\top x_j] = \sum_{k=1}^d \text{Var}[(x_i)_k (x_j)_k] = d \cdot 1 = d
$$

因此內積的**標準差為 $\sqrt{d}$**。當 $d$ 很大（如 $d = 512$ 或 $d = 1024$）時，內積的尺度也非常大，導致 softmax 的梯度趨近於零（梯度消失）：

$$
\text{若 } e_{i,j} \gg e_{i,k} \text{ 對所有 } k \neq j, \quad \text{則 } \alpha_{i,j} \to 1, \; \alpha_{i,k} \to 0
$$

此時 softmax 飽和，梯度幾乎為 $0$，訓練停滯。

**為什麼飽和會讓梯度為零？** 先用 sigmoid 類比建立直覺：sigmoid 的輸出接近 0 或 1 時，函數曲線幾乎水平，輸入改變一點點對輸出幾乎沒有影響，梯度因此接近零。softmax 是同樣的情況——輸出分布越尖銳，各個輸出對輸入的敏感度越低。正式地，softmax 的導數為 $\frac{\partial \alpha_j}{\partial e_l} = \alpha_j(\delta_{jl} - \alpha_l)$（完整推導見 [`05-backpropagation.md`](05-backpropagation.md) §1.4）。觀察對角項 $\alpha_j(1 - \alpha_j)$：當 $\alpha_j \to 1$ 或 $\alpha_j \to 0$ 時，這個乘積都趨近 $0$；非對角項 $-\alpha_j \alpha_l$ 同樣趨近 $0$。也就是說，**飽和時 softmax 的整個導數矩陣趨近零**，loss 傳回 attention score 的調整訊號變得極弱——模型知道自己錯了，卻很難有效修正 query、key 的參數。

### 解法：除以 $\sqrt{d}$ 把標準差壓回 $O(1)$

上面推導出內積的標準差是 $\sqrt{d}$，因此最自然的修正就是除以這個量，讓縮放後的分數標準差恰好等於 1：

$$
e_{i,j} = \frac{x_i^\top x_j}{\sqrt{d}}
$$

驗證：對常數 $c$，$\text{Var}[cX] = c^2\,\text{Var}[X]$，取 $c = 1/\sqrt{d}$：

$$
\text{Var}\left[\frac{x_i^\top x_j}{\sqrt{d}}\right] = \frac{1}{d}\cdot\text{Var}[x_i^\top x_j] = \frac{d}{d} = 1
$$

縮放後分數的典型幅度回到 $O(1)$，softmax 不再飽和，梯度正常流動。

> **一句話總結：** 高維向量的內積隨維度增大而放大（標準差 $\sqrt{d}$），直接送進 softmax 會讓分布過度尖銳、梯度趨近於零；除以 $\sqrt{d}$ 把標準差壓回 1，讓 attention 權重保持分散、訓練訊號能有效傳遞。

數值穩定問題解決了，接下來把整個推導整合成一個乾淨的公式，並點出它的根本限制。

---

## 9. 核心總結公式

### Naive Self-Attention（本文所推導）

$$
\boxed{
C = \text{softmax}\!\left(X X^\top\right) X
}
$$

或逐元素表示：

$$
c_i = \sum_{j=1}^T \underbrace{\text{softmax}_j(x_i^\top x_j)}_{\alpha_{i,j}} \cdot x_j
$$

**限制：**

- $E = XX^\top$ 是對稱矩陣 → 「我關注你」與「你關注我」程度相同，無法建模非對稱關係
- 「查詢向量」與「被查詢的記憶向量」都是同一個 $x_i$，角色未分離
- 「讀取的值」也是 $x_j$，無法對輸入做不同的線性變換

三個限制都源自「同一個向量扮演太多角色」——第 10 節引入三個獨立的線性投影矩陣，讓查詢、索引、內容各司其職。

---

## 10. 從 Naive Self-Attention 到 QKV

Naive self-attention 的核心限制可以用一個直覺類比說明：

> 一個人在圖書館查資料，用**同一本書**作為「查詢條件」、「索引目錄」、和「閱讀內容」——這顯然不合理。

### QKV 的解法

引入三個可學習的線性投影矩陣：

$$
W_Q \in \mathbb{R}^{d \times d_k}, \quad
W_K \in \mathbb{R}^{d \times d_k}, \quad
W_V \in \mathbb{R}^{d \times d_v}
$$

對輸入做線性變換，分離三種角色：

| 角色 | 公式 | 語意 |
|---|---|---|
| Query（查詢）| $Q = XW_Q$ | 「我想找什麼資訊？」 |
| Key（鍵）| $K = XW_K$ | 「我能提供什麼資訊的索引？」 |
| Value（值）| $V = XW_V$ | 「我實際提供的內容是什麼？」 |

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

對照 Naive Self-Attention：

| | Naive | Transformer |
|---|---|---|
| 相似度計算 | $XX^\top$ | $QK^\top / \sqrt{d_k}$ |
| 對稱性 | 是（$E_{ij} = E_{ji}$）| 否（$QK^\top \neq KQ^\top$）|
| 角色分離 | 否 | 是（Q / K / V 獨立）|
| 可學習參數 | 無 | $W_Q, W_K, W_V$ |
| 維度縮放 | 無 | $1/\sqrt{d_k}$ |

### 推導過程

**Step 1：** 計算注意力分數矩陣

$$
E = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}, \quad E_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}
$$

**Step 2：** 歸一化

$$
A = \text{softmax}_{\text{row}}(E) \in \mathbb{R}^{T \times T}
$$

**Step 3：** 加權讀取

$$
C = AV \in \mathbb{R}^{T \times d_v}
$$

其中 $C_i = \sum_j \alpha_{i,j} v_j$，每個輸出是 Value 向量的加權平均。

### 參數量分析

每一層 Self-Attention 的可學習參數：

$$
W_Q, W_K \in \mathbb{R}^{d \times d_k}, \quad W_V \in \mathbb{R}^{d \times d_v}
$$

參數總量：$d \cdot d_k + d \cdot d_k + d \cdot d_v = 2d \cdot d_k + d \cdot d_v$

若 $d_k = d_v = d$，共 $3d^2$ 個參數。

---

## 附錄：關鍵數學工具回顧

### A. 矩陣乘法維度檢查

$$
(T \times d) \cdot (d \times T) = (T \times T) \quad [E = XX^\top]
$$

$$
(T \times T) \cdot (T \times d) = (T \times d) \quad [C = AX]
$$

### B. Softmax 的梯度

設 $a = \text{softmax}(z)$，則：

$$
\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j)
$$

其中 $\delta_{ij}$ 是 Kronecker delta。這個雅可比矩陣說明：softmax 輸出之間相互耦合，某一項增大必然導致其他項減小。

### C. 內積的線性代數性質

$$
(AB)^\top = B^\top A^\top
$$

$$
\text{tr}(AB) = \text{tr}(BA)
$$

$$
x^\top y = \text{tr}(xy^\top) = \sum_k x_k y_k
$$

### D. Embedding 作為查找表

上面說 $x_i \in \mathbb{R}^d$，但這個向量是怎麼產生的？

設詞彙表大小為 $V$，令 $E \in \mathbb{R}^{V \times d}$ 為 Embedding 矩陣（可訓練參數）。對 Token ID 為 $t_i \in \{0, 1, \ldots, V-1\}$ 的 token，其 embedding 為：

$$
x_i = E[t_i] \in \mathbb{R}^d \quad \text{（取出 } E \text{ 的第 } t_i \text{ 列）}
$$

等價地，令 $\delta_{t_i} \in \mathbb{R}^V$ 為第 $t_i$ 個 one-hot 向量，則：

$$
x_i = \delta_{t_i}^\top E
$$

也就是說，「查表」在數學上等價於「one-hot 向量乘以矩陣」——但實作上直接索引取列（$O(1)$），不做矩陣乘法。

**梯度特性：** 反向傳播時，梯度 $\frac{\partial \mathcal{L}}{\partial E}$ 只有第 $t_i$ 列非零——未被本 batch 選中的 token，其 embedding 本步不更新。這意味著稀有詞需要更多訓練樣本才能讓 embedding 收斂。（完整推導見 [`05-backpropagation.md`](05-backpropagation.md) §6）

---

---

## 下一步

你已經從第一性原理推導了 Scaled Dot-Product Attention，並理解 $\sqrt{d}$ 縮放的統計學意義。

**接下來：** [`02-attention-intuition.md`](02-attention-intuition.md) — 用完整的翻譯範例（I eat fish → 我吃魚）走一遍 QKV attention 的每個計算步驟，建立從公式到實際計算的連結。
