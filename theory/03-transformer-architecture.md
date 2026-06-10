# 03｜Transformer 架構：QKV → Multi-Head → Block → 位置編碼

> **適合對象：** 讀完 01b（或熟悉 01a）與 02 後，想理解完整 Transformer 架構的讀者。
>
> **讀完後你能做什麼：**
> - 推導 Naive Self-Attention 的三個限制，以及 QKV 如何解決它們
> - 說明 Scaled Dot-Product Attention 中除以 $\sqrt{d_k}$ 的統計學原因
> - 解釋 Multi-Head Attention 的架構與 Shape 計算
> - 描述一個完整 Transformer Block 的四個子模組（MHA → Residual → FFN → Residual）
> - 解釋 Sinusoidal Positional Encoding 的設計動機
>
> **前置文件：** [`01b-prerequisites-math.md`](01b-prerequisites-math.md)、[`02-attention-intuition.md`](02-attention-intuition.md)
>
> **注意：** 本文件不含反向傳播推導，梯度推導請見 [`05-backpropagation.md`](05-backpropagation.md)
>
> **學完後的下一步：** → [`04-gpt-decoder-only.md`](04-gpt-decoder-only.md)（Causal Masking 與 nanoGPT 橋接）

---

> 本文件將 Pre-Transformer 的基本形式：
>
> $$
> c_i = \sum_j \text{softmax}(x_i^\top x_j) \, x_j
> $$
>
> 推廣為完整 Transformer 架構，涵蓋 Multi-Head Attention、Transformer Block、Positional Encoding。
> 反向傳播推導請見 [`05-backpropagation.md`](05-backpropagation.md)。



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

三個限制有一個統一的解法：引入三組獨立的線性投影——下一節說明 QKV 如何分別解決這三個問題。

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

公式確定了，接下來追蹤每個矩陣在每一步的形狀，確認維度計算前後一致。

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

Shape 追蹤完了，但單頭 attention 每次只能學一種「注意力模式」——下一節說明多頭如何讓模型同時關注不同面向。

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

### 5.5 Multi-Head Attention 架構圖

```
          X  (T × d)
         /        \
        /           \
  ┌─────────┐   ┌─────────┐   (H 個 head 並行，以 H=2 為例)
  │  Head 1 │   │  Head 2 │
  │  W_Q¹   │   │  W_Q²   │
  │  W_K¹   │   │  W_K²   │
  │  W_V¹   │   │  W_V²   │
  │         │   │         │
  │softmax  │   │softmax  │
  │(QK⊤/√dk)│   │(QK⊤/√dk)│
  └────┬────┘   └────┬────┘
       │              │
       C¹(T×dk)      C²(T×dk)
        \            /
         Concat → (T × d)
              │
             W_O
              │
          Output (T × d)
```

### 5.6 最小數值例子（H=2, d=4, d_k=2）

設 T=2（2 個 token），d=4，H=2，d_k=d_v=2。

**輸入：**

$$
X = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

**Head 1 的投影矩陣（取前 2 維）：**

$$
W_Q^{(1)} = W_K^{(1)} = W_V^{(1)} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}^{\!\top} \in \mathbb{R}^{4 \times 2}
\quad \Rightarrow \quad
Q^{(1)} = K^{(1)} = V^{(1)} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

**Head 1 的 attention（與 §3 的 2-token 例子相同）：**

$$
E^{(1)} = \frac{Q^{(1)} (K^{(1)})^\top}{\sqrt{2}} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix},\quad
A^{(1)} = \text{softmax}(E^{(1)}) = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
$$

$$
C^{(1)} = A^{(1)} V^{(1)} = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

**Head 2 投影矩陣取後 2 維，得到 $C^{(2)}$（略，做法相同）。**

**拼接與輸出：**

$$
\text{Concat}(C^{(1)}, C^{(2)}) \in \mathbb{R}^{2 \times 4} \xrightarrow{W_O} \text{Output} \in \mathbb{R}^{2 \times 4}
$$

**關鍵觀察：** 兩個 head 分別處理輸入的不同子空間（前 2 維 vs 後 2 維），最後透過 $W_O$ 融合，輸出維度 $d=4$ 不變。

Multi-Head Attention 是 Transformer Block 的第一個子模組——第 6 節把四個子模組（MHA → Residual → FFN → Residual）組裝成完整的 Block。

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

想像每一層都有兩條路可以走：一條是做完 Attention 後的「新路」，一條是完全跳過、直接把輸入傳下去的「舊路」。訓練時，模型只需要學習「新路和舊路的差距」（殘差），而不是從零學習整個映射。更重要的是：梯度往回走時，舊路永遠暢通無阻，不會因為層數太多而衰減消失。

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

```
    輸入 X  (T × d)
       │
       ├──────────────────────┐  ← Residual shortcut
       │                      │
       ↓                      │
  LayerNorm (Pre-LN)          │
       │                      │
  Multi-Head Attention        │
       │                      │
       └──────────── + ───────┘
                     │
                    Z'  (T × d)
                     │
       ┌─────────────┴──────────┐  ← Residual shortcut
       │                        │
       ↓                        │
  LayerNorm (Pre-LN)            │
       │                        │
  Feed-Forward (d → 4d → d)     │
       │                        │
       └──────────── + ─────────┘
                     │
                 輸出 Y  (T × d)
```

Shape 全程不變（始終為 $T \times d$），使得 $N$ 個 Block 可以直接串疊。

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

## 下一步

你已經完整理解 Transformer 架構的所有元件。

**兩條岔路，可以依興趣選擇順序：**

- **往實作走：** [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) — 用 NumPy 從零實作本文所有元件
- **往 GPT 走：** [`04-gpt-decoder-only.md`](04-gpt-decoder-only.md) — 了解 GPT 的 Decoder-Only 架構與 Causal Masking，然後打開 nanoGPT
- **往反向傳播走：** [`05-backpropagation.md`](05-backpropagation.md) — 手推 Self-Attention 與 LayerNorm 的完整梯度

