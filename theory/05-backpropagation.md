# 05｜反向傳播推導：Self-Attention 與 LayerNorm

> **適合對象：** 讀完 03 並實際跑過 NB1 或 NB2 後，想從頭推導梯度的讀者。需要熟悉矩陣微分與鏈式法則。
>
> **讀完後你能做什麼：**
> - 推導 $\partial \mathcal{L}/\partial Q$、$\partial \mathcal{L}/\partial K$、$\partial \mathcal{L}/\partial V$ 的完整公式
> - 推導 Softmax 的 Jacobian 及其在 attention 中的應用
> - 推導 LayerNorm 的三條梯度路徑（直接路徑 / mean 路徑 / variance 路徑）
> - 解釋為什麼 Residual Connection 能讓梯度直接流過而不衰減
> - 推導 Embedding 矩陣的完整梯度（含 Weight Tying 的稀疏／稠密雙通道）
>
> **前置文件：** [`03-transformer-architecture.md`](03-transformer-architecture.md)，以及 NB1 或 NB2 中的前向傳播實作
>
> **對應 Notebook：** [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb) — 本文每個公式都有對應的 Python 實作

---

## 數值驗證範例（T=2, d_k=2）

> 先用具體數字跑一次完整的前向 + 反向傳播，再讀符號推導。

### 設定

$$
Q = K = V = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad d_k = 2,\quad
G^C = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}（假設上游梯度為單位矩陣）
$$

### 前向傳播

**Step 1：注意力分數**

$$
E = \frac{QK^\top}{\sqrt{2}} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.707 & 0 \\ 0 & 0.707 \end{bmatrix}
$$

**Step 2：Softmax（逐行）**

$$
A = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
\quad \left(\text{例如第一行：}\frac{e^{0.707}}{e^{0.707}+e^0} = \frac{2.028}{3.028} \approx 0.67\right)
$$

**Step 3：加權讀取**

$$
C = AV = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
$$

### 反向傳播

**對 V 的梯度**

$$
G^V = A^\top G^C = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
$$

**對 A 的梯度**

$$
G^A = G^C V^\top = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Softmax 反向（逐行）**

以第一行為例，$A_{0,:} = [0.67, 0.33]$，$G^A_{0,:} = [1, 0]$：

$$
s_0 = \langle A_{0,:},\, G^A_{0,:} \rangle = 0.67 \times 1 + 0.33 \times 0 = 0.67
$$

$$
G^E_{0,:} = A_{0,:} \odot (G^A_{0,:} - s_0) = [0.67, 0.33] \odot [0.33, -0.67] = [0.221, -0.221]
$$

完整矩陣：

$$
G^E = \begin{bmatrix} 0.221 & -0.221 \\ -0.221 & 0.221 \end{bmatrix}
$$

**對 Q 與 K 的梯度**

$$
G^Q = \frac{1}{\sqrt{2}}\, G^E K = \begin{bmatrix} 0.156 & -0.156 \\ -0.156 & 0.156 \end{bmatrix}
$$

$$
G^K = \frac{1}{\sqrt{2}}\, (G^E)^\top Q = \begin{bmatrix} 0.156 & -0.156 \\ -0.156 & 0.156 \end{bmatrix}
$$

因為 $Q=K$ 且 $G^C$ 是對稱矩陣，所以 $G^Q = G^K$——這是對稱輸入的特性，一般情況下兩者不同。

**完整梯度流一覽：**

$$
G^C \xrightarrow{A^\top \cdot} G^V = \begin{bmatrix}0.67&0.33\\0.33&0.67\end{bmatrix},\quad
G^C \xrightarrow{\cdot V^\top} G^A = I,\quad
G^A \xrightarrow{\text{softmax}^{-1}} G^E = \begin{bmatrix}0.221&{-0.221}\\{-0.221}&0.221\end{bmatrix}
$$

$$
G^E \xrightarrow{\frac{1}{\sqrt{d_k}}\cdot K} G^Q = \begin{bmatrix}0.156&{-0.156}\\{-0.156}&0.156\end{bmatrix},\quad
G^E \xrightarrow{\frac{1}{\sqrt{d_k}}(\cdot)^\top Q} G^K = \begin{bmatrix}0.156&{-0.156}\\{-0.156}&0.156\end{bmatrix}
$$

以上每個數字都可以對照後面的符號推導逐步驗證。

具體數字跑過一遍了，接下來用符號推導同樣的計算，得到對任意輸入都成立的一般公式。

---

## 1. Self-Attention 的反向傳播推導

我們考慮單一 attention（不含 multi-head），前向傳播為：

$$
E = \frac{QK^\top}{\sqrt{d_k}}, \qquad A = \text{softmax}(E), \qquad C = AV
$$

目標：推導 $\dfrac{\partial \mathcal{L}}{\partial Q}$，$\dfrac{\partial \mathcal{L}}{\partial K}$，$\dfrac{\partial \mathcal{L}}{\partial V}$。

### 1.1 記號定義

| 符號 | Shape | 說明 |
|---|---|---|
| $Q, K$ | $T \times d_k$ | Query / Key 矩陣 |
| $V$ | $T \times d_v$ | Value 矩陣 |
| $E$ | $T \times T$ | 注意力分數矩陣（縮放後）|
| $A$ | $T \times T$ | 注意力權重矩陣 |
| $C$ | $T \times d_v$ | 輸出矩陣 |
| $G^C = \frac{\partial \mathcal{L}}{\partial C}$ | $T \times d_v$ | 從上游傳入的梯度 |

### 1.2 對 V 的梯度

由 $C = AV$，對 $V$ 求偏導：

$$
C_i = \sum_j A_{ij} V_j \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial V_j} = \sum_i A_{ij} \cdot G^C_i
$$

矩陣形式：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial V} = A^\top G^C}
$$

Shape 驗證：$(T \times T)^\top \cdot (T \times d_v) = (T \times d_v) \checkmark$

### 1.3 對 A 的梯度

仍由 $C = AV$，對 $A$ 求偏導：

$$
\frac{\partial \mathcal{L}}{\partial A_{ij}} = (G^C_i)^\top V_j
$$

矩陣形式：

$$
G^A = \frac{\partial \mathcal{L}}{\partial A} = G^C V^\top
$$

Shape 驗證：$(T \times d_v) \cdot (d_v \times T) = (T \times T) \checkmark$

### 1.4 Softmax 的反向傳播

對每一行 $i$，$A_{i,:} = \text{softmax}(E_{i,:})$，其 Jacobian 為：

$$
\frac{\partial A_{i,j}}{\partial E_{i,l}} = A_{i,j}(\delta_{jl} - A_{i,l})
$$

對第 $i$ 行施加鏈式法則：

$$
\frac{\partial \mathcal{L}}{\partial E_{i,l}}
= \sum_j \frac{\partial \mathcal{L}}{\partial A_{i,j}} \cdot \frac{\partial A_{i,j}}{\partial E_{i,l}}
= \sum_j G^A_{i,j} \cdot A_{i,j}(\delta_{jl} - A_{i,l})
$$

整理：

$$
\frac{\partial \mathcal{L}}{\partial E_{i,l}}
= A_{i,l} \underbrace{\left(G^A_{i,l} - \sum_j A_{i,j} G^A_{i,j}\right)}_{\text{去中心化}}
$$

記 $s_i = \sum_j A_{i,j} G^A_{i,j} = \langle A_{i,:},\, G^A_{i,:} \rangle$，則：

$$
G^E_{i,l} = A_{i,l} \left(G^A_{i,l} - s_i\right)
$$

向量形式（對第 $i$ 行）：

$$
G^E_{i,:} = A_{i,:} \odot \left(G^A_{i,:} - s_i \mathbf{1}\right)
$$

### 1.5 對 Q 與 K 的梯度

由 $E = QK^\top / \sqrt{d_k}$：

$$
E_{ij} = \frac{q_i^\top k_j}{\sqrt{d_k}}
$$

**對 $Q$ 的推導：**

$$
\frac{\partial E_{ij}}{\partial q_i} = \frac{k_j}{\sqrt{d_k}}
\quad \Rightarrow \quad
\frac{\partial \mathcal{L}}{\partial q_i} = \sum_j G^E_{ij} \cdot \frac{k_j}{\sqrt{d_k}}
$$

矩陣形式：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} G^E K}
$$

**對 $K$ 的推導：**

$$
\frac{\partial E_{ij}}{\partial k_j} = \frac{q_i}{\sqrt{d_k}}
\quad \Rightarrow \quad
\frac{\partial \mathcal{L}}{\partial k_j} = \sum_i G^E_{ij} \cdot \frac{q_i}{\sqrt{d_k}}
$$

矩陣形式：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} (G^E)^\top Q}
$$

### 1.6 梯度流總結

整體梯度路徑如下：

$$
\mathcal{L}
\;\xrightarrow{G^C}\;
C = AV
\;\xrightarrow{G^A,\, G^V}\;
\begin{cases}
V \xleftarrow{A^\top G^C} \\
A \xleftarrow{G^C V^\top}
\end{cases}
\;\xrightarrow{G^E}\;
E = \frac{QK^\top}{\sqrt{d_k}}
\;\xrightarrow{}\;
\begin{cases}
Q \xleftarrow{\frac{1}{\sqrt{d_k}} G^E K} \\
K \xleftarrow{\frac{1}{\sqrt{d_k}} (G^E)^\top Q}
\end{cases}
$$

**核心結果：**

$$
\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} G^E K, \qquad
\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} (G^E)^\top Q, \qquad
\frac{\partial \mathcal{L}}{\partial V} = A^\top G^C
$$

Attention 本身的梯度推導完了，但 $Q, K, V$ 都是由輸入 $X$ 經投影矩陣得到的——梯度還需要繼續往 $W_Q, W_K, W_V$ 和 $X$ 傳。

---

## 2. Linear Projection 的反向傳播

回到 $Q = XW_Q$，$K = XW_K$，$V = XW_V$，應用矩陣乘法的梯度法則：

### 對投影矩陣的梯度

$$
\frac{\partial \mathcal{L}}{\partial W_Q} = X^\top \frac{\partial \mathcal{L}}{\partial Q}, \qquad
\frac{\partial \mathcal{L}}{\partial W_K} = X^\top \frac{\partial \mathcal{L}}{\partial K}, \qquad
\frac{\partial \mathcal{L}}{\partial W_V} = X^\top \frac{\partial \mathcal{L}}{\partial V}
$$

Shape 驗證（以 $W_Q$ 為例）：

$$
(d \times T) \cdot (T \times d_k) = (d \times d_k) \checkmark
$$

### 對輸入 $X$ 的梯度

因為 $X$ 同時流入 $Q, K, V$ 三條路徑，梯度需**累加**：

$$
\frac{\partial \mathcal{L}}{\partial X}
= \frac{\partial \mathcal{L}}{\partial Q} W_Q^\top
+ \frac{\partial \mathcal{L}}{\partial K} W_K^\top
+ \frac{\partial \mathcal{L}}{\partial V} W_V^\top
$$

Shape 驗證（以第一項為例）：

$$
(T \times d_k) \cdot (d_k \times d) = (T \times d) \checkmark
$$

投影矩陣的梯度推完了，多頭 attention 的各 head 梯度本質上是同樣的計算分別執行——第 3 節說明拼接與輸出投影 $W_O$ 的梯度如何處理。

---

## 3. Multi-Head Attention 的梯度

各 head 的前向傳播獨立，反向傳播也相互獨立：

$$
C^{(h)} = \text{Attention}(Q^{(h)}, K^{(h)}, V^{(h)})
$$

拼接後經過輸出投影：

$$
\text{Cat} = \text{Concat}(C^{(1)}, \ldots, C^{(H)}) \in \mathbb{R}^{T \times d}
$$

$$
C = \text{Cat} \cdot W_O
$$

**梯度流：**

$$
G^{\text{Cat}} = \frac{\partial \mathcal{L}}{\partial \text{Cat}} = G^C W_O^\top, \qquad
\frac{\partial \mathcal{L}}{\partial W_O} = \text{Cat}^\top G^C
$$

再將 $G^{\text{Cat}}$ 按列切割，分配回各 head：

$$
G^{C^{(h)}} = G^{\text{Cat}}[:, (h-1)d_v : h \cdot d_v]
$$

每個 head 再各自套用第 1 節的單頭反向傳播，互不干擾。

---

## 4. 與 RNN Attention 的關鍵差異（反向傳播）

**RNN 的梯度問題：**

梯度需要穿越所有時間步，形成連乘積：

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

若 $\left\|\dfrac{\partial h_t}{\partial h_{t-1}}\right\| < 1$，連乘後梯度趨於 $0$（梯度消失）；若 $> 1$，梯度爆炸。

**Transformer 的梯度優勢：**

任意兩個 token $i \to j$ 的梯度路徑只穿越**一個** attention 層：

$$
\frac{\partial C_i}{\partial x_j} \propto A_{ij} W_V^\top
$$

路徑長度 $= O(1)$，不隨序列長度 $T$ 增長。因此：

- 梯度不因序列長度衰減
- 長距離依賴（如 100 個 token 之外的指代關係）與短距離依賴一樣容易學習
- 深層 Transformer 靠 **Residual Connection** 而非時間鏈傳遞梯度

Attention 的梯度優勢說清楚了，Transformer Block 裡還有一個關鍵模組的梯度推導：LayerNorm 有三條互相耦合的路徑——第 5 節完整推導。

---

## 5. LayerNorm 的反向傳播推導

LayerNorm 對每個 token 的 hidden vector **獨立歸一化**。

設某一 token 的輸入 $x \in \mathbb{R}^d$，LayerNorm 定義為：

$$
\mu = \frac{1}{d}\sum_{j=1}^d x_j, \qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2
$$

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_j = \gamma_j \hat{x}_j + \beta_j
$$

其中 $\gamma, \beta \in \mathbb{R}^d$ 是可學習的 scale / shift 參數，$\epsilon > 0$ 防止除以零。

### 5.1 記號定義

$$
r = \sqrt{\sigma^2 + \epsilon}, \qquad
g^y_j = \frac{\partial \mathcal{L}}{\partial y_j} \quad \text{（上游梯度）}
$$

目標：推導 $\dfrac{\partial \mathcal{L}}{\partial x_j}$，$\dfrac{\partial \mathcal{L}}{\partial \gamma_j}$，$\dfrac{\partial \mathcal{L}}{\partial \beta_j}$。

### 5.2 對 $\gamma$ 與 $\beta$ 的梯度

由 $y_j = \gamma_j \hat{x}_j + \beta_j$，直接求偏導：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_j} = g^y_j \hat{x}_j, \qquad
\frac{\partial \mathcal{L}}{\partial \beta_j} = g^y_j
$$

若對 batch 中所有 token 累加（$\gamma, \beta$ 在所有 token 間共享）：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_j} = \sum_{b,t} g^y_{b,t,j} \hat{x}_{b,t,j}, \qquad
\frac{\partial \mathcal{L}}{\partial \beta_j} = \sum_{b,t} g^y_{b,t,j}
$$

### 5.3 傳回 normalized vector 的梯度

由 $y_j = \gamma_j \hat{x}_j + \beta_j$：

$$
g^{\hat{x}}_j = \frac{\partial \mathcal{L}}{\partial \hat{x}_j} = g^y_j \gamma_j
$$

接下來推導 $\dfrac{\partial \mathcal{L}}{\partial x_j}$，即梯度如何透過歸一化操作反傳。

### 5.4 三條梯度路徑分析

$x_j$ 影響 loss 的路徑有三條：

**路徑 1（直接路徑）：** $x_j \to \hat{x}_j \to y_j \to \mathcal{L}$

$$
\hat{x}_j = \frac{x_j - \mu}{r} \quad \Rightarrow \quad
\frac{\partial \hat{x}_j}{\partial x_j}\Big|_{\mu, r \text{ 固定}} = \frac{1}{r}
$$

**路徑 2（透過 $\mu$）：** $x_j \to \mu \to \hat{x}_k \to \mathcal{L}$（影響所有 $k$）

$$
\mu = \frac{1}{d}\sum_{l=1}^d x_l \quad \Rightarrow \quad
\frac{\partial \mu}{\partial x_j} = \frac{1}{d}
$$

**路徑 3（透過 $\sigma^2$）：** $x_j \to \sigma^2 \to r \to \hat{x}_k \to \mathcal{L}$（影響所有 $k$）

$$
\sigma^2 = \frac{1}{d}\sum_{l=1}^d \tilde{x}_l^2, \quad \tilde{x}_l = x_l - \mu
\quad \Rightarrow \quad
\frac{\partial \sigma^2}{\partial x_j} = \frac{2\tilde{x}_j}{d}
$$

**三條梯度路徑示意圖：**

```
                         ┌─ 路徑1（直接）─────────────────── ŷ_j → L
                         │                                        ↑
x_j ─→ x̃_j=x_j-μ ──→ ÷r ──→ ŷ_j          g^ŷ_j（直接梯度 g^x̂/r）
  │                      ↑
  │                 r=√(σ²+ε)              路徑3（variance）
  │                      ↑               g^σ² × (2x̃_j/d) 傳回
  │               σ²=Σx̃²/d
  │                      ↑
  │         所有 j 的 (x_j-μ)² 累積
  │
  └──→ (1/d)→ μ ── 影響所有 x̃_k=x_k-μ    路徑2（mean）
                                           g^μ × (-1/d) 傳給每個 j
```

三條路徑需**同時計算**，因為 $\mu$ 與 $\sigma^2$ 由同一個 hidden vector 的所有維度共同決定，無法逐元素獨立處理。

### 5.5 Variance 路徑的詳細推導

$$
r^{-1} = (\sigma^2 + \epsilon)^{-1/2} \quad \Rightarrow \quad
\frac{\partial r^{-1}}{\partial \sigma^2} = -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2} = -\frac{1}{2}r^{-3}
$$

因此 $\hat{x}_j = \tilde{x}_j r^{-1}$ 透過 $r$ 對 $\sigma^2$ 的梯度：

$$
\frac{\partial \hat{x}_j}{\partial \sigma^2} = \tilde{x}_j \cdot \left(-\frac{1}{2}r^{-3}\right) = -\frac{\tilde{x}_j}{2r^3}
$$

Loss 透過 variance 路徑反傳：

$$
g^{\sigma^2} = \frac{\partial \mathcal{L}}{\partial \sigma^2}
= \sum_{j=1}^d g^{\hat{x}}_j \cdot \left(-\frac{\tilde{x}_j}{2r^3}\right)
= -\frac{1}{2r^3} \sum_{j=1}^d g^{\hat{x}}_j \tilde{x}_j
$$

利用 $\tilde{x}_j = r\hat{x}_j$：

$$
g^{\sigma^2} = -\frac{1}{2r^2} \sum_{j=1}^d g^{\hat{x}}_j \hat{x}_j
$$

### 5.6 合併三條路徑

令 $\tilde{x}_j = x_j - \mu$，對 $\tilde{x}_j$ 的梯度（先忽略 $\mu$ 對 $x_j$ 的依賴）：

$$
g^{\tilde{x}}_j = \underbrace{g^{\hat{x}}_j r^{-1}}_{\text{直接路徑}}
+ \underbrace{g^{\sigma^2} \cdot \frac{2\tilde{x}_j}{d}}_{\text{variance 路徑}}
$$

代入 $g^{\sigma^2}$ 和 $\tilde{x}_j = r\hat{x}_j$：

$$
g^{\tilde{x}}_j
= \frac{g^{\hat{x}}_j}{r}
- \frac{1}{d} \cdot \frac{\hat{x}_j}{r} \sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k
= \frac{1}{r}\left(g^{\hat{x}}_j - \frac{\hat{x}_j}{d}\sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k\right)
$$

加上 mean 路徑（$x_j$ 透過 $\mu$ 影響所有 $\tilde{x}_k = x_k - \mu$）：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
= g^{\tilde{x}}_j - \frac{1}{d}\sum_{k=1}^d g^{\tilde{x}}_k
$$

### 5.7 最終閉式公式

代入 $g^{\tilde{x}}$ 整理，最終得到：

$$
\boxed{
\frac{\partial \mathcal{L}}{\partial x_j}
= \frac{1}{r}\left(
g^{\hat{x}}_j
- \frac{1}{d}\sum_{k=1}^d g^{\hat{x}}_k
- \hat{x}_j \cdot \frac{1}{d}\sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k
\right)
}
$$

其中三項的語意：

| 項 | 來源 | 語意 |
|---|---|---|
| $g^{\hat{x}}_j$ | 直接路徑 | 自身的梯度貢獻 |
| $-\dfrac{1}{d}\sum_k g^{\hat{x}}_k$ | mean 路徑 | 平均值耦合的修正 |
| $-\hat{x}_j \cdot \dfrac{1}{d}\sum_k g^{\hat{x}}_k \hat{x}_k$ | variance 路徑 | 方差耦合的修正 |

### 5.8 向量形式

令 $g^{\hat{x}} = \dfrac{\partial \mathcal{L}}{\partial \hat{x}} \in \mathbb{R}^d$，向量化寫法：

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{1}{r}\left(
g^{\hat{x}}
- \text{mean}(g^{\hat{x}})
- \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})
\right)
$$

其中 $\text{mean}(\cdot) = \dfrac{1}{d}\sum_j (\cdot)_j$，$\odot$ 為逐元素乘法。

### 5.9 關鍵結論：為什麼梯度是耦合的？

LayerNorm 的 $\mu$ 與 $\sigma^2$ 都由同一個 hidden vector 的**所有維度共同決定**。因此：

- 每個輸入維度 $x_j$ 的梯度不只來自自身的輸出 $y_j$
- 也受到所有其他維度 $\{x_k\}_{k \neq j}$ 的影響（透過 $\mu$ 和 $\sigma^2$）

這使得 LayerNorm 的梯度不是 element-wise 操作，而是 hidden dimension 內部的**耦合運算**，必須整體計算（如公式所示），不能逐元素獨立處理。

### 5.10 在 Transformer Block 中的梯度流

典型結構：

$$
Z' = \text{LayerNorm}(X + Z), \quad U = X + Z
$$

**Step 1：** 從 $G^{Z'}$ 經 LayerNorm 反傳得 $G^U$（套用 13.7 的公式）

**Step 2：** 由 Residual Connection $U = X + Z$：

$$
\frac{\partial \mathcal{L}}{\partial X}\Big|_{\text{res}} = G^U, \qquad
\frac{\partial \mathcal{L}}{\partial Z} = G^U
$$

Residual Connection 讓梯度**不經任何非線性操作**直接流回輸入，這是 Transformer 能訓練很多層的關鍵。

### 5.11 LayerNorm 反向傳播總結

$$
\frac{\partial \mathcal{L}}{\partial \gamma} = g^y \odot \hat{x}, \qquad
\frac{\partial \mathcal{L}}{\partial \beta} = g^y
$$

$$
g^{\hat{x}} = g^y \odot \gamma
$$

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{1}{r}\!\left(
g^{\hat{x}} - \text{mean}(g^{\hat{x}}) - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})
\right), \quad r = \sqrt{\sigma^2 + \epsilon}
$$

Self-Attention 與 LayerNorm 都推完了，但梯度還有最後一站：Embedding 矩陣是整個計算圖的**起點**，梯度如何流到它、哪些列會被更新——第 6 節補上完整反向傳播的最後一塊拼圖。

---

## 6. Embedding 矩陣的完整梯度推導

Embedding 矩陣 $E$ 在 GPT 中扮演兩個角色：輸入側把 token ID 查表成向量（Lookup）；若使用 Weight Tying（如 Karpathy 原版 nanoGPT），輸出側的 lm_head 也共用同一份 $E$。本節推導兩條路徑的梯度，並說明它們的稀疏／稠密差異。

### 6.1 符號定義

| 符號 | 意義 | 維度 |
|---|---|---|
| $V$ | 詞彙表大小 | — |
| $E \in \mathbb{R}^{V \times d}$ | Embedding 矩陣（含 Weight Tying） | $V \times d$ |
| $t_i \in \{0,\ldots,V-1\}$ | 位置 $i$ 的 Token ID | — |
| $x_i = E[t_i]$ | 位置 $i$ 的 embedding | $d$ |
| $\delta_{t_i} \in \mathbb{R}^V$ | 第 $t_i$ 個 one-hot 向量 | $V$ |
| $\hat{h}_i$ | lm_head 輸入（LN 後的 hidden state） | $d$ |
| $z_i = \hat{h}_i E^\top$ | lm_head 輸出（logits） | $V$ |
| $p^{(i)}_k$ | 位置 $i$ 預測 token $k$ 的機率 | — |
| $y_i$ | 位置 $i$ 的目標 Token ID | — |

### 6.2 前向傳播（關鍵步驟）

$$
x_i = \delta_{t_i}^\top E \quad \text{（Lookup = one-hot 乘以 } E\text{）}
$$

$$
z_i = \hat{h}_i E^\top \quad \text{（lm\_head，Weight Tying 共用 } E\text{）}
$$

$$
p^{(i)}_k = \frac{\exp(z_i^{(k)})}{\sum_{j=1}^{V} \exp(z_i^{(j)})}
$$

$$
\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p^{(i)}_{y_i}
$$

### 6.3 反向傳播逐步推導

**Step 1：Cross-Entropy + Softmax 合併梯度**

對 Cross-Entropy $\mathcal{L} = -\frac{1}{T}\log p^{(i)}_{y_i}$ 與 Softmax 合併求導（標準結果）：

$$
\frac{\partial \mathcal{L}}{\partial z_i^{(k)}} = \frac{1}{T}\left(p^{(i)}_k - \mathbf{1}[k = y_i]\right)
$$

記 $\delta_i = \frac{\partial \mathcal{L}}{\partial z_i} \in \mathbb{R}^V$。白話：在正確 token 的位置值為 $\frac{p_{y_i}-1}{T}$（負值，梯度要把這個 logit 推高），其餘位置值為 $\frac{p_k}{T}$（正值，把其他 logit 往下壓）。

**Step 2：lm_head 反向**（$z_i = \hat{h}_i E^\top$）

對輸入 $\hat{h}_i$（梯度往上傳）：

$$
\frac{\partial \mathcal{L}}{\partial \hat{h}_i} = \delta_i E \in \mathbb{R}^d
$$

對參數 $E$（Weight Tying，輸出側梯度）：

$$
\frac{\partial \mathcal{L}}{\partial E}\bigg|_{\text{output}} = \sum_{i=1}^{T} \delta_i^\top \hat{h}_i^\top \in \mathbb{R}^{V \times d}
$$

取第 $k$ 列：$\displaystyle\frac{\partial \mathcal{L}}{\partial E[k]}\bigg|_{\text{output}} = \sum_{i=1}^{T} \delta_i^{(k)} \cdot \hat{h}_i$

**Step 3：穿越 LayerNorm 和各 Block**

此段梯度路徑在 §1–§5 已詳細推導。設最終到達 $x_i = E[t_i]$ 的梯度為：

$$
g_i = \frac{\partial \mathcal{L}}{\partial x_i} \in \mathbb{R}^d
$$

**Step 4：Lookup 反向**（$x_i = \delta_{t_i}^\top E$）

由鏈式法則：

$$
\frac{\partial \mathcal{L}}{\partial E} = \sum_{i=1}^{T} \frac{\partial \mathcal{L}}{\partial x_i} \cdot \frac{\partial x_i}{\partial E} = \sum_{i=1}^{T} \delta_{t_i} g_i^\top
$$

取第 $k$ 列（僅 $t_i = k$ 的項非零，因為 $\delta_{t_i}^{(k)} = \mathbf{1}[t_i = k]$）：

$$
\frac{\partial \mathcal{L}}{\partial E[k]}\bigg|_{\text{input}} = \sum_{\{i\,:\,t_i = k\}} g_i
$$

**Step 5：Weight Tying 下的總梯度**

$E$ 同時作為輸入 Embedding 和輸出 lm_head，梯度來自兩條路徑：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial E[k]} = \underbrace{\sum_{\{i\,:\,t_i = k\}} g_i}_{\text{輸入側（稀疏）}} + \underbrace{\sum_{i=1}^{T} \delta_i^{(k)} \hat{h}_i}_{\text{輸出側（稠密）}}}
$$

> **註：** 若 lm_head 與 token embedding 是**獨立參數**（如本倉庫 NB4 的寫法），則輸入側梯度屬於 `token_embedding.weight`、輸出側梯度屬於 `lm_head.weight`，兩者各自更新、不累加。Weight Tying 時才合併到同一份 $E$。

**Step 6：Optimizer 更新**

$$
E[k] \leftarrow E[k] - \eta \cdot \frac{\partial \mathcal{L}}{\partial E[k]}
$$

### 6.4 三個重要特性

| 特性 | 輸入側 | 輸出側 |
|---|---|---|
| **稀疏性** | 只有出現過的 token 列有梯度 | 所有 $V$ 列都有梯度（稠密）|
| **梯度來源** | 語言模型「讀入」時的表示學習 | 語言模型「預測」時的對比信號 |
| **Weight Tying 的收益** | — | 稀有 token 即使沒被選為輸入，也能從輸出側持續收到梯度 |

**實作注意：** PyTorch `nn.Embedding(sparse=True)` 只傳輸有梯度的列，在詞彙量 $V > 10^5$ 時顯著節省反向傳播的記憶體與頻寬。

---

## 核心總結

Transformer 的核心運算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

整體設計的五個核心概念：

1. **QKV 分離**：將查詢、索引、內容解耦，突破對稱性限制
2. **縮放點積**：$1/\sqrt{d_k}$ 保持 softmax 分佈平滑，梯度正常流動
3. **多頭機制**：在 $H$ 個子空間並行捕捉不同的注意力模式
4. **Residual + LayerNorm**：穩定深層訓練，提供梯度高速通道
5. **位置編碼**：以正弦或可學習向量注入序列的位置資訊

**Self-Attention 反向傳播核心公式：**

$$
\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} G^E K, \qquad
\frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} (G^E)^\top Q, \qquad
\frac{\partial \mathcal{L}}{\partial V} = A^\top G^C
$$

**LayerNorm 反向傳播核心公式：**

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{1}{r}\!\left(g^{\hat{x}} - \text{mean}(g^{\hat{x}}) - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})\right)
$$

---

## 反向傳播完整梯度查閱表

從損失 $\mathcal{L}$ 到所有葉節點（可訓練參數）的完整梯度公式：

### Self-Attention（$C = \text{softmax}(QK^\top/\sqrt{d_k}) V$）

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $G^V = \partial\mathcal{L}/\partial V$ | $A^\top G^C$ | §1.2 |
| $G^A = \partial\mathcal{L}/\partial A$ | $G^C V^\top$ | §1.3 |
| $G^E_{i,:}$（Softmax 反向）| $A_{i,:} \odot (G^A_{i,:} - \langle A_{i,:}, G^A_{i,:}\rangle)$ | §1.4 |
| $\partial\mathcal{L}/\partial Q$ | $\dfrac{1}{\sqrt{d_k}} G^E K$ | §1.5 |
| $\partial\mathcal{L}/\partial K$ | $\dfrac{1}{\sqrt{d_k}} (G^E)^\top Q$ | §1.5 |

### Linear Projection（$Q=XW_Q$, $K=XW_K$, $V=XW_V$）

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $\partial\mathcal{L}/\partial W_Q$ | $X^\top \cdot \partial\mathcal{L}/\partial Q$ | §2 |
| $\partial\mathcal{L}/\partial W_K$ | $X^\top \cdot \partial\mathcal{L}/\partial K$ | §2 |
| $\partial\mathcal{L}/\partial W_V$ | $X^\top \cdot \partial\mathcal{L}/\partial V$ | §2 |
| $\partial\mathcal{L}/\partial X$ | $(\partial\mathcal{L}/\partial Q)W_Q^\top + (\partial\mathcal{L}/\partial K)W_K^\top + (\partial\mathcal{L}/\partial V)W_V^\top$ | §2 |

### LayerNorm（$y_j = \gamma_j \hat{x}_j + \beta_j$）

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $\partial\mathcal{L}/\partial \gamma$ | $g^y \odot \hat{x}$ | §5.2 |
| $\partial\mathcal{L}/\partial \beta$ | $g^y$ | §5.2 |
| $g^{\hat{x}}$（中間量）| $g^y \odot \gamma$ | §5.3 |
| $\partial\mathcal{L}/\partial x$ | $\dfrac{1}{r}\!\left(g^{\hat{x}} - \text{mean}(g^{\hat{x}}) - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})\right)$ | §5.7 |

### Embedding（$x_i = E[t_i]$，lm_head $z_i = \hat{h}_i E^\top$）

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $\delta_i = \partial\mathcal{L}/\partial z_i$（CE + Softmax）| $\delta_i^{(k)} = \tfrac{1}{T}(p^{(i)}_k - \mathbf{1}[k=y_i])$ | §6.3 Step 1 |
| $\partial\mathcal{L}/\partial \hat{h}_i$ | $\delta_i E$ | §6.3 Step 2 |
| $\partial\mathcal{L}/\partial E[k]$（輸出側，稠密）| $\sum_i \delta_i^{(k)} \hat{h}_i$ | §6.3 Step 2 |
| $\partial\mathcal{L}/\partial E[k]$（輸入側，稀疏）| $\sum_{\{i:\,t_i=k\}} g_i$ | §6.3 Step 4 |
| $\partial\mathcal{L}/\partial E[k]$（Weight Tying 總梯度）| 輸入側 + 輸出側 | §6.3 Step 5 |

> **說明：** $r = \sqrt{\sigma^2 + \epsilon}$，$\hat{x} = (x-\mu)/r$，$\text{mean}(\cdot) = \tfrac{1}{d}\sum_j (\cdot)_j$。查閱表中所有公式皆有對應的符號推導章節與數值驗證。

---

## 下一步

你已經能從頭推導 Self-Attention 和 LayerNorm 的完整梯度。

**接下來：** [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb) — 用 NumPy 實作本文推導的每一條梯度公式，並以數值梯度驗證正確性。

*本文件為 Transformer 數學基礎系列的第二篇，承接 Pre-Knowledge.md 的 Self-Attention 數學前置，後續將介紹完整 Encoder-Decoder 架構、Masked Attention 與 Cross-Attention 的推導。*