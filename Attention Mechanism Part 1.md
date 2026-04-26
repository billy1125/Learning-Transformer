# Attention Mechanism 完整數學推導（上）
## 10.1 注意力提示 → 10.3 注意力評分函數

> **承接前篇**：前篇我們推導了 Naive Self-Attention 的核心公式 $C = \text{softmax}(XX^\top)X$，並指出其限制。本篇從「注意力」的認知科學動機出發，系統性地推導各種評分函數。

---

## 目錄

1. [10.1 注意力提示：生物學動機與抽象框架](#101-注意力提示)
2. [10.2 注意力匯聚：Nadaraya-Watson 核回歸](#102-nadaraya-watson-核回歸)
3. [10.3 注意力評分函數：掩蔽、加性、縮放點積](#103-注意力評分函數)

---

## 10.1 注意力提示

### 10.1.1 生物學中的注意力提示

人類視覺系統在面對複雜場景時，**不會均勻地處理所有資訊**，而是根據兩種機制選擇焦點：

| 機制 | 觸發方式 | 特性 |
|---|---|---|
| **非自主性提示**（Non-volitional）| 環境刺激（突然的聲音、高對比度物體）| 自動、被動、快速 |
| **自主性提示**（Volitional）| 目標導向（「我想找一把紅色的雨傘」）| 刻意、主動、可控 |

**類比到機器學習：**

- 非自主性提示 → **Key**：資料本身的顯著特徵（「我能提供什麼」）
- 自主性提示 → **Query**：當前任務的需求（「我在找什麼」）
- 感知對象的內容 → **Value**：實際讀取的資訊（「我提供的具體內容」）

### 10.1.2 查詢、鍵與值（Query, Key, Value）

#### 抽象框架

設有一組**鍵值對資料庫**（Key-Value Store）：

$$
\mathcal{D} = \{(k_1, v_1), (k_2, v_2), \ldots, (k_n, v_n)\}
$$

給定一個查詢 $q$，注意力機制計算：

$$
\text{Attention}(q, \mathcal{D}) = \sum_{i=1}^n \alpha(q, k_i) \cdot v_i
$$

其中 $\alpha(q, k_i)$ 是注意力權重，滿足：

$$
\alpha(q, k_i) \geq 0, \qquad \sum_{i=1}^n \alpha(q, k_i) = 1
$$

#### 與資料庫查詢的對比

| 傳統資料庫（硬查詢）| 注意力機制（軟查詢）|
|---|---|
| $q$ 必須**完全匹配**某個 $k_i$ | $q$ 與所有 $k_i$ **部分匹配** |
| 返回單一 $v_i$ | 返回所有 $v_i$ 的**加權組合** |
| 不可微 | **可微**，支援梯度下降 |

這個「軟查詢」特性正是讓注意力機制可以端對端訓練的關鍵。

### 10.1.3 注意力的視覺化理解

注意力矩陣 $A \in \mathbb{R}^{T_q \times T_k}$ 中：

- **行**（row）：每個 Query 對所有 Key 的注意力分佈
- **列**（column）：每個 Key 被所有 Query 關注的程度

熱力圖（Heatmap）是常見的視覺化方式：顏色越深代表注意力越集中。

```
Query\Key  k1    k2    k3
q1        [0.7  0.2   0.1]  ← q1 主要關注 k1
q2        [0.1  0.8   0.1]  ← q2 主要關注 k2
q3        [0.3  0.3   0.4]  ← q3 的注意力較分散
```

### 10.1.4 小結

**核心抽象**：注意力機制 = **可微分的軟查詢系統**

$$
\boxed{\text{Attention}(q, \mathcal{D}) = \sum_i \underbrace{\text{softmax}(a(q, k_i))}_{\text{權重}} \cdot v_i}
$$

其中 $a(q, k_i)$ 是評分函數（scoring function），本章後續重點推導其各種形式。

---

## 10.2 Nadaraya-Watson 核回歸

> 這一節提供一個具體的數學模型，說明注意力機制的非參數版本。

### 10.2.1 問題設定

給定訓練資料集：

$$
\{(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)\}
$$

目標：對新輸入 $x$ 估計 $\mathbb{E}[y \mid x]$。

### 10.2.2 平均匯聚（基準方法）

最簡單的估計是**樣本均值**：

$$
f(x) = \frac{1}{n} \sum_{i=1}^n y_i
$$

這完全忽略了輸入 $x$ 的資訊，相當於注意力均勻分佈（$\alpha_i = 1/n$）。

**限制**：無法捕捉 $x$ 與 $y$ 之間的局部關係。

### 10.2.3 非參數注意力匯聚（Nadaraya-Watson）

**核心想法**：距離查詢 $x$ 越近的訓練點 $x_i$，對預測的影響應越大。

**Nadaraya-Watson 核回歸**（1964）：

$$
f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\displaystyle\sum_{j=1}^n K(x - x_j)} \cdot y_i
$$

其中 $K(\cdot)$ 是核函數（kernel function），常用的高斯核：

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{u^2}{2}\right)
$$

#### 改寫為注意力形式

代入高斯核，令 $\alpha(x, x_i)$ 表示注意力權重：

$$
\alpha(x, x_i) = \frac{\exp\!\left(-\frac{(x - x_i)^2}{2}\right)}{\displaystyle\sum_{j=1}^n \exp\!\left(-\frac{(x - x_j)^2}{2}\right)} = \text{softmax}\!\left(-\frac{(x - x_i)^2}{2}\right)
$$

對比注意力機制的通用形式：

$$
f(x) = \sum_{i=1}^n \underbrace{\text{softmax}(a(x, x_i))}_{\alpha(x, x_i)} \cdot \underbrace{y_i}_{v_i}
$$

**評分函數**為：$a(x, x_i) = -\frac{(x - x_i)^2}{2}$

這等價於：**$x$ 與 $x_i$ 越接近（差值越小），分數越高，注意力越集中**。

#### 核回歸的性質分析

| 性質 | 說明 |
|---|---|
| **局部性** | 只有附近的訓練點有顯著影響 |
| **平滑性** | 預測值是訓練點的連續加權平均 |
| **一致性** | 當 $n \to \infty$ 時，收斂到真實條件期望 |
| **無參數** | 不需要訓練，但需要儲存所有訓練資料 |

### 10.2.4 帶參數的注意力匯聚

為了讓模型具備更強的表達能力，引入**可學習的參數** $w$：

$$
f(x) = \sum_{i=1}^n \text{softmax}\!\left(-\frac{1}{2}\bigl[(x - x_i) w\bigr]^2\right) \cdot y_i
$$

其中 $w$ 是標量參數，控制注意力的「寬窄」：

| $w$ 值 | 效果 |
|---|---|
| 大的 $\lvert w \rvert$ | 注意力集中（窄核，過擬合風險）|
| 小的 $\lvert w \rvert$ | 注意力分散（寬核，欠擬合風險）|

通過梯度下降學習最優的 $w$，使得模型在驗證集上表現最佳。

**矩陣化形式**：對於批次查詢 $\mathbf{x} = (x_{\text{query},1}, \ldots, x_{\text{query},m})$：

$$
\text{Attention Weights} = \text{softmax}\!\left(-\frac{1}{2}\left(X_{\text{query}} - X_{\text{key}}\right)^2 w^2\right) \in \mathbb{R}^{m \times n}
$$

### 10.2.5 小結

Nadaraya-Watson 核回歸揭示了注意力機制的**統計學根源**：

$$
\boxed{f(x) = \sum_{i=1}^n \text{softmax}\bigl(a(x, x_i)\bigr) \cdot y_i}
$$

| 注意力組件 | 核回歸對應 |
|---|---|
| Query $q$ | 測試輸入 $x$ |
| Key $k_i$ | 訓練輸入 $x_i$ |
| Value $v_i$ | 訓練標籤 $y_i$ |
| 評分函數 $a$ | $-\frac{1}{2}(x - x_i)^2$ |

從此出發，我們可以將評分函數推廣到向量空間，得到現代注意力機制。

---

## 10.3 注意力評分函數

評分函數 $a(q, k)$ 是注意力機制的核心設計選擇。本節推導三種主要形式。

### 10.3.1 掩蔽 Softmax 操作（Masked Softmax）

在實際應用中，序列往往有**填充**（padding）或需要**因果遮蔽**（causal masking），需要讓某些位置的注意力權重強制為零。

#### 10.3.1.1 填充遮蔽（Padding Mask）

批次訓練時，不同長度的序列需要對齊（zero-padding）。在計算注意力時，填充位置不應被關注。

**做法**：將填充位置的分數設為 $-\infty$：

$$
\tilde{e}_{i,j} = \begin{cases} e_{i,j} & \text{若位置 } j \text{ 是真實 token} \\ -\infty & \text{若位置 } j \text{ 是填充} \end{cases}
$$

由於 $\exp(-\infty) = 0$，softmax 後填充位置的權重精確為零：

$$
\alpha_{i,j} = \frac{\exp(\tilde{e}_{i,j})}{\sum_k \exp(\tilde{e}_{i,k})} \xrightarrow{\text{填充位置}} 0
$$

#### 10.3.1.2 因果遮蔽（Causal Mask）

在語言模型中，位置 $i$ 只能關注位置 $j \leq i$（不能「看到未來」）。

用**上三角遮蔽矩陣**實現：

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & \cdots \\
0 & 0 & -\infty & \cdots \\
0 & 0 & 0 & \cdots \\
\vdots & & & \ddots
\end{bmatrix}
$$

$$
A = \text{softmax}(E + M)
$$

**效果**：$A$ 是嚴格下三角矩陣（每行的 softmax 只在允許位置上有非零值）。

#### 10.3.1.3 掩蔽 Softmax 的數值實作

```
def masked_softmax(X, valid_lens):
    # 對每行的有效位置做 softmax，無效位置填 -1e6（代替 -inf 避免 NaN）
    mask = (positions < valid_lens)  # 廣播比較
    X[~mask] = -1e6
    return softmax(X, dim=-1)
```

注意：使用 $-10^6$ 而非 $-\infty$ 是為了避免某些框架的 NaN 問題。

---

### 10.3.2 加性注意力（Additive Attention）

**出處**：Bahdanau et al. (2015)，最早的可學習注意力機制之一。

#### 動機

當 Query 和 Key 的維度不同時（$q \in \mathbb{R}^{d_q}$，$k \in \mathbb{R}^{d_k}$），點積無法直接計算。需要一個能處理**不同維度**的評分函數。

#### 公式

$$
a(q, k) = w_v^\top \tanh(W_q q + W_k k)
$$

其中：

$$
W_q \in \mathbb{R}^{h \times d_q}, \quad W_k \in \mathbb{R}^{h \times d_k}, \quad w_v \in \mathbb{R}^h
$$

$h$ 是隱藏層維度（超參數）。

#### 計算過程拆解

**Step 1：線性投影到公共空間**

$$
\tilde{q} = W_q q \in \mathbb{R}^h, \qquad \tilde{k} = W_k k \in \mathbb{R}^h
$$

無論 $d_q, d_k$ 為何，投影後都在 $\mathbb{R}^h$ 中。

**Step 2：加性融合 + 非線性**

$$
z = \tanh(\tilde{q} + \tilde{k}) \in \mathbb{R}^h
$$

$\tanh$ 壓縮到 $(-1, 1)$，提供非線性。

**Step 3：投影到標量**

$$
a(q, k) = w_v^\top z = w_v^\top \tanh(W_q q + W_k k) \in \mathbb{R}
$$

#### 批次化實作

對 $n_q$ 個 Query 和 $n_k$ 個 Key：

$$
E \in \mathbb{R}^{n_q \times n_k}, \quad E_{ij} = w_v^\top \tanh(W_q q_i + W_k k_j)
$$

技巧：用廣播（broadcasting）避免顯式雙重迴圈：

$$
\text{queries\_proj} \in \mathbb{R}^{n_q \times 1 \times h}, \quad \text{keys\_proj} \in \mathbb{R}^{1 \times n_k \times h}
$$

相加後廣播得到 $(n_q, n_k, h)$，再與 $w_v$ 做內積得到 $(n_q, n_k)$。

#### 性質比較

| 性質 | 加性注意力 |
|---|---|
| Query/Key 維度 | **可以不同** |
| 計算複雜度 | $O(n_q \cdot n_k \cdot h)$ |
| 可學習參數 | $W_q, W_k, w_v$（共 $h(d_q + d_k + 1)$ 個）|
| 表達能力 | 較強（有非線性 $\tanh$）|

---

### 10.3.3 縮放點積注意力（Scaled Dot-Product Attention）

**出處**：Vaswani et al., "Attention is All You Need" (2017)。

#### 前提條件

Query 和 Key 必須有**相同維度** $d_k$：

$$
q, k \in \mathbb{R}^{d_k}
$$

#### 公式

$$
a(q, k) = \frac{q^\top k}{\sqrt{d_k}}
$$

#### 為什麼除以 $\sqrt{d_k}$？

**數學推導**：

假設 $q$ 和 $k$ 的各分量 i.i.d. 來自 $\mathcal{N}(0, 1)$，則：

$$
q^\top k = \sum_{i=1}^{d_k} q_i k_i
$$

$$
\mathbb{E}[q^\top k] = \sum_{i=1}^{d_k} \mathbb{E}[q_i]\mathbb{E}[k_i] = 0
$$

$$
\text{Var}[q^\top k] = \sum_{i=1}^{d_k} \text{Var}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = d_k
$$

因此 $\text{Std}[q^\top k] = \sqrt{d_k}$。縮放後：

$$
\text{Var}\!\left[\frac{q^\top k}{\sqrt{d_k}}\right] = \frac{d_k}{d_k} = 1
$$

**無論 $d_k$ 多大，縮放後的分數都保持單位方差**，防止 softmax 飽和。

#### 批次化矩陣形式

對 $n_q$ 個 Query 和 $n_k$ 個 Key（以及 Value）：

$$
Q \in \mathbb{R}^{n_q \times d_k}, \quad K \in \mathbb{R}^{n_k \times d_k}, \quad V \in \mathbb{R}^{n_k \times d_v}
$$

$$
\boxed{\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \in \mathbb{R}^{n_q \times d_v}}
$$

**維度追蹤：**

$$
QK^\top: (n_q \times d_k) \cdot (d_k \times n_k) = (n_q \times n_k)
$$

$$
\text{softmax}(\cdot): (n_q \times n_k) \xrightarrow{\text{row-wise}} (n_q \times n_k)
$$

$$
(\cdot) V: (n_q \times n_k) \cdot (n_k \times d_v) = (n_q \times d_v)
$$

#### 計算效率優勢

| | 加性注意力 | 縮放點積注意力 |
|---|---|---|
| 時間複雜度 | $O(n_q n_k h)$ | $O(n_q n_k d_k)$（但常數更小）|
| GPU 矩陣乘法 | 難以並行 | **高度並行** |
| 參數量 | 多（$W_q, W_k, w_v$）| 零（只有一個常數 $\sqrt{d_k}$）|

**關鍵優勢**：$QK^\top$ 是一個純矩陣乘法，可以用高度優化的 BLAS 運算實現，在 GPU 上效率極高。

#### Dropout 在注意力中的應用

在 softmax 之後、矩陣乘 $V$ 之前，加入 Dropout：

$$
\text{Attention}(Q, K, V) = \text{Dropout}\!\left(\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)\right) V
$$

這隨機「遮蔽」部分注意力連接，具有正規化效果，防止模型過度依賴特定的注意力模式。

### 10.3.4 小結：三種評分函數對比

| 評分函數 | 公式 | 維度要求 | 特點 |
|---|---|---|---|
| 平均匯聚 | $a = 1/n$（常數）| 無 | 無選擇性 |
| 加性注意力 | $w_v^\top \tanh(W_q q + W_k k)$ | $d_q \neq d_k$ 可 | 可學習，有非線性 |
| 縮放點積 | $q^\top k / \sqrt{d_k}$ | $d_q = d_k$ | 高效，GPU 友好 |

**Transformer 的選擇**：縮放點積注意力，原因是計算效率最高，且搭配 QKV 投影後表達能力足夠強。

---

## 附錄 A：掩蔽 Softmax 的數值行為

以下說明為什麼 $-\infty$ 在理論上正確，但實際中常用 $-10^6$：

$$
\lim_{e \to -\infty} \frac{\exp(e)}{\exp(e) + \sum_{k \neq j} \exp(e_k)} = \frac{0}{0 + \sum_{k \neq j} \exp(e_k)} = 0
$$

但如果**所有位置都被遮蔽**（valid_len = 0），分母為 $0$，產生 NaN。實作時需特別處理此邊界情形。

## 附錄 B：加性 vs. 點積：哪個更好？

Britz et al. (2017) 的實驗結論：

- **小維度** $d_k$：兩者表現相近
- **大維度** $d_k$：點積注意力在縮放後**優於**加性注意力（更穩定，訓練更快）
- **計算資源**：點積注意力因矩陣乘法優化，速度顯著更快

這解釋了為何 Transformer 最終選擇了縮放點積注意力。

---

*下一篇（Part 2）將推導：Bahdanau 注意力（Seq2Seq 的突破）、Multi-Head Attention 的完整數學、自注意力與位置編碼。*
