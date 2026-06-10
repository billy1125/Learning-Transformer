# Transformer：Self-Attention 的完整形式

> 本文件將 Pre-Transformer 的基本形式：
>
> $$
> c_i = \sum_j \text{softmax}(x_i^\top x_j) \, x_j
> $$
>
> 推廣為完整 Transformer 架構，並包含 Self-Attention 與 LayerNorm 的完整反向傳播推導。



---

## 目錄

1. Self-Attention 的限制
2. Query / Key / Value 的動機
3. Scaled Dot-Product Attention
4. 矩陣形式與 Shape 分析
5. Multi-Head Attention
6. Transformer Block
7. Positional Encoding
8. 與 RNN 的結構差異
9. Self-Attention 的反向傳播推導
10. Linear Projection 的反向傳播
11. Multi-Head Attention 的梯度
12. 與 RNN Attention 的關鍵差異（反向傳播）
13. LayerNorm 的反向傳播推導

---

## 1. Self-Attention 的限制

回顧 Pre-Transformer 的原始形式：

$$
e_{i,j} = x_i^\top x_j, \qquad c_i = \sum_j \text{softmax}_j(x_i^\top x_j) \cdot x_j
$$

這個設計存在兩個根本限制：

### 限制一：角色混淆

同一個向量 $x_i$ 同時扮演三種角色：

| 角色 | 語意 | 在 naive 版本中 |
|---|---|---|
| 查詢（Query） | 「我要找什麼」 | $x_i$（用於計算 $x_i^\top x_j$）|
| 鍵（Key） | 「我能被如何匹配」 | $x_j$（用於計算 $x_i^\top x_j$）|
| 值（Value） | 「我實際提供的資訊」 | $x_j$（用於加權平均）|

三種語意不同的操作共用同一個向量，表達能力受限。

### 限制二：對稱性

因為 $e_{i,j} = x_i^\top x_j = x_j^\top x_i = e_{j,i}$，注意力矩陣 $E$ 是**對稱矩陣**。

這意味著：「$i$ 關注 $j$ 的程度」= 「$j$ 關注 $i$ 的程度」。

但自然語言中的依賴關係往往是非對稱的。例如代名詞「它」指向名詞的關係，與名詞被代名詞引用的關係，語意強度不同。

### 限制三：無可學習的投影

整個計算過程沒有任何可訓練參數（除了 embedding 本身），模型無法透過梯度學習「什麼是好的相似度」。

---

## 2. Query / Key / Value 的動機

解決方案：引入三個**獨立的線性投影矩陣**，將輸入向量投影到不同的子空間，分別扮演不同角色。

### 2.1 定義投影

$$
q_i = W_Q x_i \in \mathbb{R}^{d_k}, \qquad
k_j = W_K x_j \in \mathbb{R}^{d_k}, \qquad
v_j = W_V x_j \in \mathbb{R}^{d_v}
$$

其中可學習參數：

$$
W_Q \in \mathbb{R}^{d_k \times d}, \quad
W_K \in \mathbb{R}^{d_k \times d}, \quad
W_V \in \mathbb{R}^{d_v \times d}
$$

**Shape 說明：**

| 變數 | Shape | 說明 |
|---|---|---|
| $X$ | $T \times d$ | 輸入序列 |
| $W_Q, W_K$ | $d \times d_k$ | Query/Key 投影（通常 $d_k = d/H$）|
| $W_V$ | $d \times d_v$ | Value 投影（通常 $d_v = d/H$）|
| $Q, K$ | $T \times d_k$ | Query/Key 矩陣 |
| $V$ | $T \times d_v$ | Value 矩陣 |

### 2.2 語意分離

| 投影 | 問題 | 語意 |
|---|---|---|
| Query $q_i$ | 「我要找什麼資訊？」 | 當前 token 的查詢意圖 |
| Key $k_j$ | 「我能提供什麼資訊的索引？」 | 其他 token 的可被匹配特徵 |
| Value $v_j$ | 「我實際提供的內容是什麼？」 | 其他 token 的資訊內容 |

分離之後：

- $q_i^\top k_j$：匹配分數，不再對稱（因為 $W_Q \neq W_K$）
- 讀取的值 $v_j$ 可與匹配用的 $k_j$ 完全不同
- $W_Q, W_K, W_V$ 均為可學習參數

---

## 3. Scaled Dot-Product Attention

### 3.1 注意力分數

$$
e_{i,j} = q_i^\top k_j = (W_Q x_i)^\top (W_K x_j)
$$

### 3.2 縮放的必要性（統計推導）

設 $q_i, k_j$ 的各分量 i.i.d. 來自 $\mathcal{N}(0, 1)$，則：

$$
\mathbb{E}[q_i^\top k_j] = \sum_{l=1}^{d_k} \mathbb{E}[(q_i)_l (k_j)_l] = 0
$$

$$
\text{Var}(q_i^\top k_j) = \sum_{l=1}^{d_k} \text{Var}[(q_i)_l (k_j)_l] = d_k \cdot 1 = d_k
$$

因此內積的**標準差為 $\sqrt{d_k}$**。當 $d_k$ 很大時（如 $d_k = 64$），分數尺度也大，softmax 趨近 one-hot：

$$
\text{若 } e_{i,j} \gg e_{i,k} \; \forall k \neq j, \quad \text{則 } \alpha_{i,j} \to 1, \quad \nabla_{E} \mathcal{L} \to 0
$$

梯度幾乎消失，訓練停滯。除以 $\sqrt{d_k}$ 後：

$$
\text{Var}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

方差回到 $O(1)$，softmax 分佈保持平滑。

### 3.3 完整公式

**逐元素形式：**

$$
\alpha_{i,j} = \frac{\exp\!\left(\dfrac{q_i^\top k_j}{\sqrt{d_k}}\right)}{\displaystyle\sum_{l=1}^T \exp\!\left(\dfrac{q_i^\top k_l}{\sqrt{d_k}}\right)}, \qquad c_i = \sum_{j=1}^T \alpha_{i,j} \, v_j
$$

**矩陣形式：**

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

---

## 4. 矩陣形式與 Shape 分析

### 4.1 投影

$$
Q = X W_Q \in \mathbb{R}^{T \times d_k}, \qquad
K = X W_K \in \mathbb{R}^{T \times d_k}, \qquad
V = X W_V \in \mathbb{R}^{T \times d_v}
$$

### 4.2 計算流程與 Shape 追蹤

**Step 1：注意力分數矩陣**

$$
E = \frac{QK^\top}{\sqrt{d_k}} \in \mathbb{R}^{T \times T}
$$

$$
(T \times d_k) \cdot (d_k \times T) = (T \times T) \checkmark
$$

每個元素 $E_{ij} = \dfrac{q_i^\top k_j}{\sqrt{d_k}}$，衡量 token $i$ 對 token $j$ 的注意力分數。

**Step 2：歸一化**

$$
A = \text{softmax}_{\text{row}}(E) \in \mathbb{R}^{T \times T}, \qquad \sum_{j=1}^T A_{ij} = 1 \; \forall i
$$

**Step 3：加權讀取**

$$
C = AV \in \mathbb{R}^{T \times d_v}
$$

$$
(T \times T) \cdot (T \times d_v) = (T \times d_v) \checkmark
$$

每一行 $C_i = \sum_j A_{ij} v_j$。

### 4.3 完整流程圖（Shape 標注）

$$
X \xrightarrow{W_Q, W_K, W_V} Q, K, V
\xrightarrow{QK^\top / \sqrt{d_k}} E
\xrightarrow{\text{softmax}} A
\xrightarrow{\times V} C
$$

$$
(T \times d) \;\to\; (T \times d_k),\, (T \times d_k),\, (T \times d_v) \;\to\; (T \times T) \;\to\; (T \times T) \;\to\; (T \times d_v)
$$

---

## 5. Multi-Head Attention

### 5.1 定義

將 $d$ 維空間分成 $H$ 個 head，每個 head 獨立執行一次 attention：

$$
Q^{(h)} = X W_Q^{(h)}, \quad
K^{(h)} = X W_K^{(h)}, \quad
V^{(h)} = X W_V^{(h)}
$$

其中各 head 的投影矩陣：

$$
W_Q^{(h)}, W_K^{(h)} \in \mathbb{R}^{d \times d_k}, \quad W_V^{(h)} \in \mathbb{R}^{d \times d_v}
$$

通常令 $d_k = d_v = d / H$，使總計算量與單頭相當。

每個 head 的輸出：

$$
C^{(h)} = \text{Attention}(Q^{(h)}, K^{(h)}, V^{(h)}) \in \mathbb{R}^{T \times d_v}
$$

### 5.2 拼接與輸出投影

$$
C = \text{Concat}(C^{(1)}, C^{(2)}, \ldots, C^{(H)}) \, W_O
$$

其中：

- $\text{Concat}(\cdots) \in \mathbb{R}^{T \times (H \cdot d_v)} = \mathbb{R}^{T \times d}$（當 $d_v = d/H$）
- $W_O \in \mathbb{R}^{d \times d}$：輸出投影矩陣
- 最終輸出 $C \in \mathbb{R}^{T \times d}$

### 5.3 為什麼需要多頭？

**單一 attention 的限制：** $A = \text{softmax}(QK^\top / \sqrt{d_k})$ 每行只能生成一種注意力分佈。對於同一個 token，可能同時需要：

- 關注句法依賴（如主語和動詞）
- 關注語意相似性（如同義詞）
- 關注局部位置（如相鄰詞）

這些是截然不同的「注意力模式」，單頭無法同時建模。

**多頭的優勢：**

| 面向 | 單頭 | 多頭（$H$ 個）|
|---|---|---|
| 注意力模式 | 1 種 | $H$ 種，各自獨立 |
| 子空間 | 整個 $d$ 維 | $H$ 個 $d_k$ 維子空間 |
| 可學習參數 | $3d^2$ | $3Hd \cdot d_k + d^2 = 3d^2 + d^2$（含 $W_O$）|
| 表示能力 | 低 | 高 |

直觀理解：多個 head 就像多個「關注角度」，讓模型從不同維度理解 token 之間的關係。

### 5.4 參數量分析

單個 Multi-Head Attention 層的參數：

$$
\underbrace{H \cdot (d \cdot d_k + d \cdot d_k + d \cdot d_v)}_{\text{各 head 的 } W_Q, W_K, W_V} + \underbrace{d \cdot d}_{\text{輸出投影 } W_O}
$$

若 $d_k = d_v = d/H$，則：

$$
= H \cdot 3d \cdot \frac{d}{H} + d^2 = 3d^2 + d^2 = 4d^2
$$

---

## 6. Transformer Block

一個完整的 Transformer Block 由四個子模組依序組成：

### 6.1 Multi-Head Self-Attention（全域關聯）

$$
Z = \text{MultiHead}(X)
$$

捕捉序列中任意兩個位置之間的依賴關係（路徑長度 $O(1)$）。

### 6.2 第一個 Residual Connection + Layer Normalization

$$
Z' = \text{LayerNorm}(X + Z)
$$

**Residual Connection 的作用：**

- 提供梯度直接流動的「高速公路」，緩解深層網路的梯度消失
- 允許模型學習「殘差」（與恆等映射的差距），而非從頭學習整個映射

**Layer Normalization 的作用：**

- 穩定每一層的數值分佈，加速訓練
- 對 hidden dimension 做歸一化（與 Batch Norm 不同，不依賴 batch size）

### 6.3 Position-wise Feed-Forward Network（局部非線性）

$$
F = \text{ReLU}(Z' W_1 + b_1) W_2 + b_2
$$

其中：

- $W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$
- 通常 $d_{ff} = 4d$（例如 $d = 512, d_{ff} = 2048$）
- 對每個位置**獨立且相同地**施加（position-wise），不跨位置互動

FFN 的作用：在 attention 捕捉全局關聯後，對每個 token 的表示做非線性變換，增加模型的非線性表達能力。

### 6.4 第二個 Residual Connection + Layer Normalization

$$
Y = \text{LayerNorm}(Z' + F)
$$

### 6.5 完整 Block 的計算圖

$$
X \xrightarrow{+\text{MHA}} X + Z \xrightarrow{\text{LN}} Z' \xrightarrow{+\text{FFN}} Z' + F \xrightarrow{\text{LN}} Y
$$

**參數量總覽（單一 Block）：**

| 子模組 | 參數量 |
|---|---|
| Multi-Head Attention（含 $W_O$）| $4d^2$ |
| FFN | $d \cdot d_{ff} + d_{ff} \cdot d = 2d \cdot d_{ff}$ |
| LayerNorm（$\times 2$）| $2 \cdot 2d = 4d$ |
| **合計（$d_{ff}=4d$）** | $4d^2 + 8d^2 + 4d \approx 12d^2$ |

---

## 7. Positional Encoding

### 7.1 為什麼需要位置資訊？

Self-Attention 計算的是集合操作（set operation）：

$$
C = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

對輸入序列做任意**排列置換**不改變結果（permutation equivariant），因為 $E_{ij} = q_i^\top k_j$ 只依賴向量本身，不依賴位置。

因此，序列「貓 吃 魚」與「魚 吃 貓」在純 attention 中是不可分辨的，必須顯式引入位置資訊。

### 7.2 正弦位置編碼（Sinusoidal Positional Encoding）

在輸入 embedding 中加入位置編碼向量：

$$
x_i \leftarrow x_i + p_i
$$

位置編碼 $p_i \in \mathbb{R}^d$ 定義為：

$$
p_{i, 2k} = \sin\!\left(\frac{i}{10000^{2k/d}}\right), \qquad
p_{i, 2k+1} = \cos\!\left(\frac{i}{10000^{2k/d}}\right)
$$

其中 $i$ 是位置索引（$1 \leq i \leq T$），$k$ 是維度索引（$0 \leq k < d/2$）。

### 7.3 設計動機

**多頻率正弦波**：不同 $k$ 對應不同「波長」：

$$
\lambda_k = 2\pi \cdot 10000^{2k/d}
$$

- $k = 0$：波長 $= 2\pi$（最短，捕捉局部位置）
- $k = d/2$：波長 $= 2\pi \cdot 10000$（最長，捕捉全局位置）

**相對位置可計算性：** 對任意固定偏移 $\delta$，位置 $p_{i+\delta}$ 可以表達為 $p_i$ 的線性函數：

$$
\begin{bmatrix} p_{i+\delta, 2k} \\ p_{i+\delta, 2k+1} \end{bmatrix}
=
\begin{bmatrix} \cos(\omega_k \delta) & \sin(\omega_k \delta) \\ -\sin(\omega_k \delta) & \cos(\omega_k \delta) \end{bmatrix}
\begin{bmatrix} p_{i, 2k} \\ p_{i, 2k+1} \end{bmatrix}
$$

其中 $\omega_k = 1 / 10000^{2k/d}$。這使得模型可以透過內積學習相對位置關係。

**外推能力：** 正弦函數定義在整個實數域，理論上可以處理訓練時未見過的序列長度。

### 7.4 可學習位置編碼（Learned Positional Encoding）

另一種做法是讓 $p_i$ 成為可訓練參數（如 BERT、GPT 所採用）：

$$
p_i \in \mathbb{R}^d \quad \text{（可學習，隨梯度更新）}
$$

| | 正弦編碼 | 可學習編碼 |
|---|---|---|
| 參數量 | 0 | $T_{\max} \cdot d$ |
| 外推能力 | 強 | 弱（超出訓練長度退化）|
| 訓練穩定性 | 好 | 需正則化 |

---

## 8. 與 RNN 的結構差異

| 面向 | RNN | Transformer |
|---|---|---|
| 計算模式 | Sequential（必須逐步）| Parallel（可完全並行）|
| token 互動方式 | 鄰近傳遞（$h_t$ 依賴 $h_{t-1}$）| 全連接（任意兩 token 直接互動）|
| 最長梯度路徑 | $O(T)$（需穿越所有時間步）| $O(1)$（任意兩 token 只隔一層 attention）|
| 長距離依賴 | 困難（梯度連乘衰減）| 容易（直接連接）|
| 訓練速度 | 慢（無法並行）| 快（矩陣運算可 GPU 並行）|
| 記憶體複雜度 | $O(T)$ | $O(T^2)$（注意力矩陣）|
| 位置資訊 | 天然具備（時序結構）| 需要顯式注入（PE）|

**關鍵洞察：** Transformer 以 $O(T^2)$ 的記憶體複雜度（注意力矩陣）換取了 $O(1)$ 的梯度路徑，使長距離依賴的學習從根本上變得可行。

---

## 9. Self-Attention 的反向傳播推導

我們考慮單一 attention（不含 multi-head），前向傳播為：

$$
E = \frac{QK^\top}{\sqrt{d_k}}, \qquad A = \text{softmax}(E), \qquad C = AV
$$

目標：推導 $\dfrac{\partial \mathcal{L}}{\partial Q}$，$\dfrac{\partial \mathcal{L}}{\partial K}$，$\dfrac{\partial \mathcal{L}}{\partial V}$。

### 9.1 記號定義

| 符號 | Shape | 說明 |
|---|---|---|
| $Q, K$ | $T \times d_k$ | Query / Key 矩陣 |
| $V$ | $T \times d_v$ | Value 矩陣 |
| $E$ | $T \times T$ | 注意力分數矩陣（縮放後）|
| $A$ | $T \times T$ | 注意力權重矩陣 |
| $C$ | $T \times d_v$ | 輸出矩陣 |
| $G^C = \frac{\partial \mathcal{L}}{\partial C}$ | $T \times d_v$ | 從上游傳入的梯度 |

### 9.2 對 V 的梯度

由 $C = AV$，對 $V$ 求偏導：

$$
C_i = \sum_j A_{ij} V_j \quad \Rightarrow \quad \frac{\partial \mathcal{L}}{\partial V_j} = \sum_i A_{ij} \cdot G^C_i
$$

矩陣形式：

$$
\boxed{\frac{\partial \mathcal{L}}{\partial V} = A^\top G^C}
$$

Shape 驗證：$(T \times T)^\top \cdot (T \times d_v) = (T \times d_v) \checkmark$

### 9.3 對 A 的梯度

仍由 $C = AV$，對 $A$ 求偏導：

$$
\frac{\partial \mathcal{L}}{\partial A_{ij}} = (G^C_i)^\top V_j
$$

矩陣形式：

$$
G^A = \frac{\partial \mathcal{L}}{\partial A} = G^C V^\top
$$

Shape 驗證：$(T \times d_v) \cdot (d_v \times T) = (T \times T) \checkmark$

### 9.4 Softmax 的反向傳播

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

### 9.5 對 Q 與 K 的梯度

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

### 9.6 梯度流總結

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

---

## 10. Linear Projection 的反向傳播

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

---

## 11. Multi-Head Attention 的梯度

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

每個 head 再各自套用第 9 節的單頭反向傳播，互不干擾。

---

## 12. 與 RNN Attention 的關鍵差異（反向傳播）

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

---

## 13. LayerNorm 的反向傳播推導

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

### 13.1 記號定義

$$
r = \sqrt{\sigma^2 + \epsilon}, \qquad
g^y_j = \frac{\partial \mathcal{L}}{\partial y_j} \quad \text{（上游梯度）}
$$

目標：推導 $\dfrac{\partial \mathcal{L}}{\partial x_j}$，$\dfrac{\partial \mathcal{L}}{\partial \gamma_j}$，$\dfrac{\partial \mathcal{L}}{\partial \beta_j}$。

### 13.2 對 $\gamma$ 與 $\beta$ 的梯度

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

### 13.3 傳回 normalized vector 的梯度

由 $y_j = \gamma_j \hat{x}_j + \beta_j$：

$$
g^{\hat{x}}_j = \frac{\partial \mathcal{L}}{\partial \hat{x}_j} = g^y_j \gamma_j
$$

接下來推導 $\dfrac{\partial \mathcal{L}}{\partial x_j}$，即梯度如何透過歸一化操作反傳。

### 13.4 三條梯度路徑分析

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

### 13.5 Variance 路徑的詳細推導

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

### 13.6 合併三條路徑

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

### 13.7 最終閉式公式

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

### 13.8 向量形式

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

### 13.9 關鍵結論：為什麼梯度是耦合的？

LayerNorm 的 $\mu$ 與 $\sigma^2$ 都由同一個 hidden vector 的**所有維度共同決定**。因此：

- 每個輸入維度 $x_j$ 的梯度不只來自自身的輸出 $y_j$
- 也受到所有其他維度 $\{x_k\}_{k \neq j}$ 的影響（透過 $\mu$ 和 $\sigma^2$）

這使得 LayerNorm 的梯度不是 element-wise 操作，而是 hidden dimension 內部的**耦合運算**，必須整體計算（如公式所示），不能逐元素獨立處理。

### 13.10 在 Transformer Block 中的梯度流

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

### 13.11 LayerNorm 反向傳播總結

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

*本文件為 Transformer 數學基礎系列的第二篇，承接 Pre-Knowledge.md 的 Self-Attention 數學前置，後續將介紹完整 Encoder-Decoder 架構、Masked Attention 與 Cross-Attention 的推導。*