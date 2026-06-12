# Attention Mechanism 完整數學推導（下）
## 10.4 Bahdanau 注意力 → 10.6 自注意力與位置編碼 → Feed-Forward 層

> **承接上篇**：我們已推導了三種評分函數（平均、加性、縮放點積）。本篇以 Bahdanau 注意力的歷史脈絡為起點，完整推導 Multi-Head Attention、位置編碼，以及 Transformer 的 Feed-Forward 子層。

---

## 目錄

1. [10.4 Bahdanau 注意力：Seq2Seq 的突破](#104-bahdanau-注意力)
2. [10.5 多頭注意力（Multi-Head Attention）](#105-多頭注意力)
3. [10.6 自注意力與位置編碼](#106-自注意力與位置編碼)
4. [Feed-Forward 層的完整數學](#feed-forward-層)
5. [Transformer 子層的組合：殘差連接與層歸一化](#殘差連接與層歸一化)

---

## 10.4 Bahdanau 注意力

### 歷史背景：Seq2Seq 的瓶頸

在 Bahdanau（2015）之前，神經機器翻譯使用**編碼器-解碼器（Encoder-Decoder）**架構：

```
源序列: [x1, x2, ..., xT]
         ↓ Encoder (RNN)
    固定維度向量 c (context vector)
         ↓ Decoder (RNN)
目標序列: [y1, y2, ..., yS]
```

**致命問題**：整個源序列必須壓縮到**一個固定大小的向量** $c$。序列越長，資訊損失越嚴重——這被稱為**資訊瓶頸**（Information Bottleneck）。

### 10.4.1 Bahdanau 模型

**核心創新**：解碼器在生成每個目標詞時，**動態計算**與源序列所有位置的注意力，而非依賴固定的 $c$。

#### 編碼器（雙向 RNN）

設源序列長度為 $T$，編碼器產生隱狀態序列：

$$
\mathbf{h} = (h_1, h_2, \ldots, h_T), \quad h_t \in \mathbb{R}^{2d_{\text{enc}}}
$$

（雙向 RNN：正向 $\overrightarrow{h}_t$ 和反向 $\overleftarrow{h}_t$ 拼接）

這些隱狀態同時扮演 **Key** 和 **Value** 的角色：

$$
k_t = v_t = h_t \in \mathbb{R}^{2d_{\text{enc}}}
$$

#### 解碼器（帶注意力的 RNN）

解碼器在第 $s$ 步的狀態：$s_{s-1} \in \mathbb{R}^{d_{\text{dec}}}$（作為 **Query**）。

**Step 1：計算注意力分數**（加性注意力）

$$
e_{s,t} = w_v^\top \tanh\!\left(W_q s_{s-1} + W_k h_t\right), \quad t = 1, \ldots, T
$$

**Step 2：softmax 得到注意力權重**

$$
\alpha_{s,t} = \frac{\exp(e_{s,t})}{\displaystyle\sum_{j=1}^T \exp(e_{s,j})}
$$

**Step 3：計算上下文向量**（動態的，每步不同）

$$
c_s = \sum_{t=1}^T \alpha_{s,t} \cdot h_t
$$

**Step 4：更新解碼器狀態**

$$
s_s = \text{RNN}(s_{s-1},\; [y_{s-1};\; c_s])
$$

**Step 5：預測下一個詞**

$$
P(y_s \mid y_{<s}, \mathbf{x}) = \text{softmax}(W_o [s_s;\; c_s])
$$

### Bahdanau 注意力的完整計算圖

```
源序列       h1    h2    h3    ...   hT     ← Encoder 輸出（Key & Value）
              ↑     ↑     ↑           ↑
              │     │     │           │      a(s_{s-1}, h_t) 加性評分
              └─────┴─────┴─────...──┘
                           ↓
               [α_{s,1}, α_{s,2}, ..., α_{s,T}]  softmax
                           ↓
                    c_s = Σ α_{s,t} h_t           加權求和
                           ↓
                    s_s = RNN(s_{s-1}, [y_{s-1}; c_s])
```

### 10.4.2 為何這是重大突破？

| | 舊 Seq2Seq | Bahdanau 注意力 |
|---|---|---|
| 上下文 | 固定向量 $c$（瓶頸）| 動態向量 $c_s$（每步不同）|
| 源序列利用 | 只用最後隱狀態 | 利用**所有**隱狀態 |
| 長序列表現 | 急劇下降 | 顯著改善 |
| 可解釋性 | 無 | 注意力矩陣可視覺化 |

### 10.4.3 訓練

使用**教師強制**（Teacher Forcing）：解碼器輸入使用真實標籤 $y_{s-1}$（而非模型預測），加速訓練收斂。

損失函數：

$$
\mathcal{L} = -\sum_{s=1}^S \log P(y_s^* \mid y_{<s}^*, \mathbf{x})
$$

其中 $y^*$ 是目標序列的真實標籤。通過 BPTT（Backpropagation Through Time）計算梯度，參數 $W_q, W_k, w_v, W_o$ 及 RNN 參數聯合更新。

### 10.4.4 小結

$$
\boxed{c_s = \sum_{t=1}^T \text{softmax}_t\!\left(w_v^\top \tanh(W_q s_{s-1} + W_k h_t)\right) \cdot h_t}
$$

Bahdanau 注意力的三個角色：
- **Query** $= s_{s-1}$（解碼器前一步狀態）
- **Key** $= h_t$（編碼器隱狀態）
- **Value** $= h_t$（同 Key，在此模型中未分離）

---

## 10.5 多頭注意力

### 動機：單一注意力的限制

縮放點積注意力每次只從一個「角度」讀取序列資訊。但自然語言中的依存關係是多維的：

- 句法依存（主語-謂語）
- 語義關聯（同義詞、反義詞）
- 指代關係（代詞-先行詞）
- 位置鄰近關係

**多頭注意力**允許模型同時從 $H$ 個不同的「表示子空間」學習注意力模式。

### 10.5.1 模型架構

#### 參數設定

給定輸入：$Q \in \mathbb{R}^{n_q \times d}$，$K \in \mathbb{R}^{n_k \times d}$，$V \in \mathbb{R}^{n_v \times d}$

每個頭 $h$ 有獨立的線性投影矩陣：

$$
W_Q^{(h)} \in \mathbb{R}^{d \times d_k}, \quad
W_K^{(h)} \in \mathbb{R}^{d \times d_k}, \quad
W_V^{(h)} \in \mathbb{R}^{d \times d_v}
$$

以及最終的輸出投影：

$$
W_O \in \mathbb{R}^{H d_v \times d_{\text{out}}}
$$

Transformer 原文設定：$d_k = d_v = d / H$（每頭維度 = 總維度 / 頭數）。

#### 單頭計算

第 $h$ 個頭的注意力輸出：

$$
\text{head}_h = \text{Attention}(Q W_Q^{(h)},\; K W_K^{(h)},\; V W_V^{(h)})
$$

$$
= \text{softmax}\!\left(\frac{(Q W_Q^{(h)})(K W_K^{(h)})^\top}{\sqrt{d_k}}\right) (V W_V^{(h)})
$$

其中 $\text{head}_h \in \mathbb{R}^{n_q \times d_v}$。

#### 多頭拼接與線性投影

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W_O
$$

$$
\text{Concat}(\cdot) \in \mathbb{R}^{n_q \times (H d_v)}, \quad W_O \in \mathbb{R}^{(H d_v) \times d_{\text{out}}}
$$

$$
\boxed{\text{MultiHead}(Q, K, V) \in \mathbb{R}^{n_q \times d_{\text{out}}}}
$$

### 10.5.2 完整的維度追蹤

設 $H = 8$，$d = 512$，$d_k = d_v = 64$，$n_q = n_k = n_v = T$：

| 步驟 | 運算 | 輸出維度 |
|---|---|---|
| 線性投影 $Q$ | $QW_Q^{(h)}$ | $(T, 64)$ |
| 線性投影 $K$ | $KW_K^{(h)}$ | $(T, 64)$ |
| 線性投影 $V$ | $VW_V^{(h)}$ | $(T, 64)$ |
| 注意力分數 | $(QW_Q^{(h)})(KW_K^{(h)})^\top / \sqrt{64}$ | $(T, T)$ |
| Softmax | softmax（逐行）| $(T, T)$ |
| 加權求和 | $A \cdot VW_V^{(h)}$ | $(T, 64)$ |
| 8頭拼接 | Concat(head₁,...,head₈) | $(T, 512)$ |
| 輸出投影 | $\cdot W_O$ | $(T, 512)$ |

**每個頭的參數量**：$d \cdot d_k + d \cdot d_k + d \cdot d_v = 512 \times 64 \times 3 = 98{,}304$

**總參數量**：$H \times 3 \times d \times d_k + Hd_v \times d = 8 \times 3 \times 512 \times 64 + 512 \times 512 = 786{,}432$（每層）

### 10.5.3 為什麼多頭有效？直觀理解

**直覺類比**：CNN 用多個卷積核提取不同的局部特徵（邊緣、紋理、顏色）。多頭注意力用多組投影提取不同的**全局關係模式**。

**各頭的分工**（實驗觀察，Vaswani 2017）：

| 頭 | 學習到的模式 |
|---|---|
| 某些頭 | 句法依存（如動詞與其賓語）|
| 某些頭 | 位置鄰近（關注附近詞）|
| 某些頭 | 稀有詞的語義（關注遠距離的相關詞）|

**數學上的表達力增益**：

單頭注意力的輸出空間受限：

$$
\text{head} = AV W_V = \text{softmax}(QK^\top/\sqrt{d_k}) \cdot V W_V
$$

多頭後通過 $W_O$ 的投影，可以在拼接的高維空間中學習更複雜的混合：

$$
\text{MultiHead} = \text{Concat}(A_1 V_1, \ldots, A_H V_H) W_O
$$

不同頭的 $A_h$ 可以學習到截然不同的注意力模式，$W_O$ 再把它們整合。

### 10.5.4 高效實作：批次矩陣乘法

實際上不需要對每個頭分別運算，可以通過**重塑（reshape）**批次化：

```python
# Q: (batch, T, d)
# 投影並重塑為多頭格式
Q_proj = Q @ W_Q  # (batch, T, H*d_k)
Q_heads = Q_proj.reshape(batch, T, H, d_k).transpose(1, 2)  # (batch, H, T, d_k)

# 批次注意力（同時計算所有頭）
scores = Q_heads @ K_heads.transpose(-2, -1) / sqrt(d_k)  # (batch, H, T, T)
A = softmax(scores, dim=-1)
output = A @ V_heads  # (batch, H, T, d_v)

# 拼接並投影
output = output.transpose(1, 2).reshape(batch, T, H*d_v) @ W_O
```

這樣所有頭的運算通過一次大矩陣乘法完成，充分利用 GPU 並行性。

---

## 10.6 自注意力與位置編碼

### 10.6.1 自注意力（Self-Attention）

自注意力是注意力機制的特殊情形：**Query、Key、Value 均來自同一序列**。

$$
Q = K = V = X \in \mathbb{R}^{T \times d}
$$

（實際上通過不同的線性投影得到 $Q = XW_Q$，$K = XW_K$，$V = XW_V$）

$$
\text{Self-Attention}(X) = \text{softmax}\!\left(\frac{XW_Q (XW_K)^\top}{\sqrt{d_k}}\right) XW_V
$$

**語意**：序列中每個 token 通過注意力讀取整個序列的資訊，更新自己的表示。

### 10.6.2 自注意力 vs. CNN vs. RNN

從三個維度比較：

#### 維度 1：計算路徑長度（任意兩點之間的最大路徑）

| 架構 | 最大路徑長度 | 含義 |
|---|---|---|
| RNN | $O(T)$ | 遠距離資訊需經過 $T$ 步傳遞 |
| CNN（核大小 $k$）| $O(T/k)$ | 需要 $\lceil T/k \rceil$ 層堆疊 |
| **Self-Attention** | $O(1)$ | 任意兩個 token 直接互動 |

**結論**：自注意力的**長距離依賴建模能力**最強。

#### 維度 2：每層計算複雜度

| 架構 | 複雜度（每層）|
|---|---|
| RNN | $O(T \cdot d^2)$ |
| CNN（核大小 $k$）| $O(k \cdot T \cdot d^2)$ |
| **Self-Attention** | $O(T^2 \cdot d)$ |

**Self-Attention 的瓶頸**：當序列長度 $T$ 很大時（如長文件），$O(T^2)$ 的計算量成為問題（這也是後來 Sparse Attention、Longformer 等研究的動機）。

#### 維度 3：並行化能力

| 架構 | 並行化 |
|---|---|
| RNN | **串行**（$h_t$ 依賴 $h_{t-1}$）|
| CNN | **並行**（各位置獨立卷積）|
| **Self-Attention** | **完全並行**（矩陣乘法一步完成）|

**綜合評估**：

| 應用場景 | 推薦架構 |
|---|---|
| 短序列、長距離依賴 | Self-Attention |
| 超長序列（$T > 10^4$）| 稀疏注意力或 CNN |
| 需要嚴格順序建模 | RNN（或帶因果遮蔽的 Transformer）|

### 10.6.3 位置編碼（Positional Encoding）

#### 問題：自注意力無位置感知

自注意力是**置換不變的**（permutation invariant）：

$$
\text{Self-Attention}(\sigma(X)) = \sigma(\text{Self-Attention}(X))
$$

其中 $\sigma$ 是任意行置換。也就是說，打亂 token 的順序，輸出只是相應地被打亂，模型無法區分不同位置的 token。

但語言中位置資訊至關重要：「狗咬人」 $\neq$ 「人咬狗」。

#### 解決方案：注入位置編碼

在輸入 Embedding 中加入位置資訊：

$$
\tilde{x}_i = x_i + p_i
$$

其中 $p_i \in \mathbb{R}^d$ 是位置 $i$ 的位置編碼向量。

#### Transformer 的正弦位置編碼

Vaswani et al. (2017) 提出的固定（非學習）位置編碼：

$$
p_{i, 2k} = \sin\!\left(\frac{i}{10000^{2k/d}}\right)
$$

$$
p_{i, 2k+1} = \cos\!\left(\frac{i}{10000^{2k/d}}\right)
$$

其中 $i$ 是位置索引（$0, 1, \ldots, T-1$），$k$ 是維度索引（$0, 1, \ldots, d/2 - 1$）。

#### 正弦位置編碼的數學性質

**性質 1：有界性**

$$
p_{i, k} \in [-1, 1] \quad \forall i, k
$$

與 Embedding 的尺度相容，不會引入過大的偏移。

**性質 2：相對位置的線性可表示性**

對固定的偏移 $\delta$，存在線性變換 $M_\delta$ 使得：

$$
p_{i+\delta} = M_\delta \cdot p_i
$$

**推導**（以二維情形為例，設 $d = 2$）：

$$
p_i = \begin{bmatrix} \sin(\omega i) \\ \cos(\omega i) \end{bmatrix}, \quad \omega = \frac{1}{10000^{2k/d}}
$$

$$
p_{i+\delta} = \begin{bmatrix} \sin(\omega i + \omega \delta) \\ \cos(\omega i + \omega \delta) \end{bmatrix}
= \begin{bmatrix} \cos(\omega\delta) & \sin(\omega\delta) \\ -\sin(\omega\delta) & \cos(\omega\delta) \end{bmatrix}
\begin{bmatrix} \sin(\omega i) \\ \cos(\omega i) \end{bmatrix}
= M_\delta p_i
$$

$M_\delta$ 是**旋轉矩陣**！這意味著相對位置關係可以通過線性運算表示，理論上模型可以學習到「位置 $i$ 比位置 $j$ 提前 $\delta$ 步」這樣的關係。

**性質 3：多頻率覆蓋不同尺度**

不同維度對應不同頻率：

$$
\omega_k = \frac{1}{10000^{2k/d}}
$$

| 維度 $k$ | 頻率 $\omega_k$ | 週期 $2\pi/\omega_k$ | 捕捉的距離尺度 |
|---|---|---|---|
| $k = 0$ | $1$ | $\approx 6$ | 短距離（詞序）|
| $k = d/4$ | $10000^{-1/2} \approx 0.01$ | $\approx 628$ | 中距離（句子結構）|
| $k = d/2 - 1$ | $10000^{-1} = 0.0001$ | $\approx 62831$ | 長距離（段落結構）|

**高維情形**（$d = 512$）的視覺化理解：

位置編碼矩陣 $P \in \mathbb{R}^{T \times d}$ 中，每一行是一個位置的編碼，每一列對應一個頻率的正弦/餘弦波。不同列的波長從 $2\pi$（最短）到 $10000 \cdot 2\pi$（最長），形成類似二進位表示的多尺度結構。

#### 位置編碼與 Embedding 的疊加

$$
\tilde{X} = X_{\text{embed}} + P \in \mathbb{R}^{T \times d}
$$

注意這裡是**加法**，不是拼接（concatenation）。原因：

- 拼接會增加維度（$2d$），增加計算量
- 加法保持維度，且位置資訊被「融合」進入 Embedding 空間

常見問題：加法會不會讓位置資訊和語義資訊相互干擾？

答：理論上存在，但實踐中模型可以學習到如何分離這兩類資訊（通過 QKV 投影），效果良好。

#### 學習式位置編碼（Learned Positional Encoding）

BERT、GPT 等模型使用**可學習的位置 Embedding**：

$$
p_i = \text{embedding\_table}[i] \in \mathbb{R}^d
$$

其中 `embedding_table` 是形狀為 $(T_{\max}, d)$ 的可學習矩陣。

| | 正弦位置編碼 | 學習式位置編碼 |
|---|---|---|
| 參數量 | 零（固定）| $T_{\max} \times d$ |
| 外推能力 | 可泛化到未見過的長度（理論上）| 無法超過 $T_{\max}$ |
| 靈活性 | 低（固定模式）| 高（任務自適應）|
| 常見使用 | 原始 Transformer | BERT、GPT 系列 |

### 10.6.4 小結

**自注意力的關鍵特性：**

$$
\text{Self-Attention}(X) = \text{softmax}\!\left(\frac{XW_Q W_K^\top X^\top}{\sqrt{d_k}}\right) XW_V
$$

**位置編碼的作用：**

$$
\tilde{x}_i = x_i + p_i \quad \text{（讓模型感知 token 的位置順序）}
$$

兩者合力：自注意力提供**全局資訊整合**，位置編碼提供**序列順序感知**。

---

## Feed-Forward 層

### 動機：注意力之後還需要什麼？

自注意力（含多頭）完成的是**跨 token 的資訊聚合**（Token Mixing），即讓每個位置看到其他位置的資訊。

但聚合之後，還需要對每個位置的表示進行**非線性特徵變換**，以增強模型的表達能力。這由 **Position-wise Feed-Forward Network（FFN）** 完成。

### 數學定義

對序列中每個位置 $i$ 的向量 $x_i \in \mathbb{R}^d$，**獨立**應用相同的兩層全連接網路：

$$
\text{FFN}(x_i) = W_2 \cdot \text{ReLU}(W_1 x_i + b_1) + b_2
$$

其中：

$$
W_1 \in \mathbb{R}^{d_{ff} \times d}, \quad b_1 \in \mathbb{R}^{d_{ff}}
$$
$$
W_2 \in \mathbb{R}^{d \times d_{ff}}, \quad b_2 \in \mathbb{R}^d
$$

**Transformer 原文設定**：$d = 512$，$d_{ff} = 2048$（$d_{ff} = 4d$，中間層展開 4 倍）。

### 矩陣化（批次）形式

對整個序列矩陣 $X \in \mathbb{R}^{T \times d}$：

$$
\text{FFN}(X) = \text{ReLU}(X W_1^\top + \mathbf{1}b_1^\top) W_2^\top + \mathbf{1}b_2^\top
$$

或簡寫（略去偏置廣播）：

$$
\text{FFN}(X) = \text{ReLU}(X W_1^\top) W_2^\top \in \mathbb{R}^{T \times d}
$$

**維度追蹤**：

$$
X: (T, d) \xrightarrow{W_1^\top} (T, d_{ff}) \xrightarrow{\text{ReLU}} (T, d_{ff}) \xrightarrow{W_2^\top} (T, d)
$$

### Position-wise 的含義

**關鍵性質**：FFN 對序列中每個位置**獨立且相同地**應用。

這意味著：

1. $W_1, W_2$ 在所有位置**共享**（不是每個位置有獨立的網路）
2. 不同位置之間**沒有直接交互**（交互已由自注意力完成）
3. 計算可以完全**並行化**（$T$ 個位置同時計算）

可以把 FFN 理解為對每個 token 的表示做一次「反思與精煉」：

$$
\text{token表示} \xrightarrow{\text{升維}} \text{豐富特徵空間} \xrightarrow{\text{非線性}} \xrightarrow{\text{降維}} \text{精煉後的表示}
$$

### 激活函數的演進

| 版本 | 激活函數 | 公式 | 使用模型 |
|---|---|---|---|
| 原始 Transformer | ReLU | $\max(0, x)$ | Transformer (2017) |
| BERT | GELU | $x \cdot \Phi(x)$ | BERT, GPT-2 |
| 改進版 | SwiGLU | $\text{Swish}(xW_1) \odot (xW_3)$ | LLaMA, PaLM |

**GELU（Gaussian Error Linear Unit）**：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right]
$$

其中 $\Phi$ 是標準正態分佈的 CDF。GELU 比 ReLU 更平滑，在負值區域有非零梯度，實踐中效果更好。

**SwiGLU**（Shazeer 2020）：

$$
\text{FFN}_{\text{SwiGLU}}(x) = (\text{Swish}_\beta(xW_1) \odot (xW_2)) W_3
$$

其中 $\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)$，$\odot$ 是逐元素乘法。SwiGLU 引入了門控機制，現代大型語言模型（LLaMA, PaLM）普遍採用。

### FFN 的參數量

$$
|W_1| + |W_2| = d \cdot d_{ff} + d_{ff} \cdot d = 2d \cdot d_{ff}
$$

設 $d = 512$，$d_{ff} = 2048$：$2 \times 512 \times 2048 = 2{,}097{,}152$（每層 FFN）

對比同層 Multi-Head Attention 的參數量：$\approx 786{,}432$

**FFN 的參數量約為 Multi-Head Attention 的 2.7 倍**，佔 Transformer 層總參數量的多數。

### 為什麼需要 FFN？

**理論視角**：Transformer 的 FFN 層在功能上類似**鍵值記憶（Key-Value Memory）**（Geva et al. 2021）：

- $W_1$ 的每一行 $w_{1,i} \in \mathbb{R}^d$ 是一個「記憶鍵」
- $W_2$ 的每一列 $w_{2,i} \in \mathbb{R}^d$ 是對應的「記憶值」
- 前向傳播 = 查詢最相關的記憶並讀取對應值

這解釋了為什麼大型語言模型能夠「記住」大量事實知識——它們被儲存在 FFN 的權重中。

---

## 殘差連接與層歸一化

### 完整的 Transformer 子層公式

每個 Transformer 編碼器層由兩個子層組成，每個子層都有**殘差連接**和**層歸一化**：

**子層 1（Multi-Head Attention）：**

$$
Z = \text{LayerNorm}\!\left(X + \text{MultiHead}(X, X, X)\right)
$$

**子層 2（Feed-Forward）：**

$$
\text{Output} = \text{LayerNorm}\!\left(Z + \text{FFN}(Z)\right)
$$

### 殘差連接（Residual Connection）

$$
\text{子層輸出} = X + f(X)
$$

**動機**（He et al. 2016，ResNet）：

梯度在反向傳播時可以「跳過」子層，直接流回更早的層：

$$
\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial (X + f(X))} \cdot \left(1 + \frac{\partial f(X)}{\partial X}\right)
$$

即使 $\frac{\partial f}{\partial X} \approx 0$，梯度仍然有 $\frac{\partial \mathcal{L}}{\partial (X + f(X))}$ 這個「殘差梯度」，有效緩解梯度消失。

### 層歸一化（Layer Normalization）

對每個 token 的特徵向量**獨立**進行歸一化（與 Batch Normalization 歸一化方向不同）：

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta
$$

其中：

$$
\mu = \frac{1}{d}\sum_{i=1}^d x_i \quad \text{（向量均值）}
$$

$$
\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2 + \epsilon} \quad \text{（向量標準差）}
$$

$\gamma, \beta \in \mathbb{R}^d$ 是可學習的縮放和偏移參數，$\epsilon$ 是防止除零的小常數（如 $10^{-5}$）。

#### Layer Norm vs. Batch Norm

| | Batch Norm | Layer Norm |
|---|---|---|
| 歸一化方向 | 跨樣本（batch 維度）| 跨特徵（feature 維度）|
| NLP 適用性 | 差（序列長度不固定，batch 維度語意不清）| **好**（每個 token 獨立歸一化）|
| 推理依賴 | 需要 running mean/var | **無需**（只用當前輸入統計）|

### Pre-Norm vs. Post-Norm

原始 Transformer 使用 **Post-Norm**（如上所示）：

$$
Z = \text{LayerNorm}(X + f(X))
$$

現代大型模型（如 GPT-3、LLaMA）使用 **Pre-Norm**：

$$
Z = X + f(\text{LayerNorm}(X))
$$

**Pre-Norm 的優勢**：訓練更穩定，不需要精心調整學習率，更容易擴展到更深的網路。

### 完整編碼器層的計算流程

```
輸入 X ∈ ℝ^{T×d}
    │
    ├── [殘差路徑 1] ────────────────────────────────┐
    │                                                 │
    └─→ LayerNorm → MultiHead Attention(X,X,X) ─→ + ─┘ → Z
                                                   ↑
                                               殘差加法
    │
    ├── [殘差路徑 2] ────────────────────────────────┐
    │                                                 │
    └─→ LayerNorm → FFN ─────────────────────────→ + ─┘ → 輸出
```

（上圖為 Pre-Norm 版本）

---

## 全局總結：Transformer 的數學全景

### 從輸入到輸出的完整公式鏈

**1. Embedding + 位置編碼**

$$
\tilde{X} = \text{Embed}(X) + P \in \mathbb{R}^{T \times d}
$$

**2. 每個 Transformer 編碼器層**（$l = 1, \ldots, L$）

$$
Z^{(l)} = \text{LayerNorm}\!\left(X^{(l)} + \text{MultiHead}(X^{(l)}, X^{(l)}, X^{(l)})\right)
$$

$$
X^{(l+1)} = \text{LayerNorm}\!\left(Z^{(l)} + \text{FFN}(Z^{(l)})\right)
$$

**3. 輸出**

$$
\text{logits} = X^{(L)} W_{\text{out}} \in \mathbb{R}^{T \times |\mathcal{V}|}
$$

### 每個組件的角色

| 組件 | 作用 | 數學本質 |
|---|---|---|
| Embedding | 離散 → 連續 | 查表 |
| 位置編碼 | 注入順序資訊 | 加性偏移 |
| Multi-Head Attention | 跨 token 資訊聚合 | 軟查詢 + 加權平均 |
| FFN | 每個 token 的特徵變換 | 兩層 MLP |
| 殘差連接 | 緩解梯度消失 | 跳躍連接 |
| 層歸一化 | 穩定訓練 | 特徵標準化 |

### 參數量全景（以 Transformer-Base 為例）

$d = 512$，$H = 8$，$d_{ff} = 2048$，$L = 6$，$|\mathcal{V}| = 37000$：

| 組件 | 參數量 |
|---|---|
| Embedding | $|\mathcal{V}| \times d = 18.9M$ |
| 每層 Multi-Head Attention | $4 \times d^2 = 1.05M$ |
| 每層 FFN | $2 \times d \times d_{ff} = 2.1M$ |
| 每層 LayerNorm | $4 \times d = 2048$（可忽略）|
| $L = 6$ 層編碼器 | $6 \times (1.05 + 2.1)M = 18.9M$ |
| **總計** | $\approx 65M$ |

---

## 附錄：關鍵數學工具補充

### A. 旋轉位置編碼（RoPE）

現代大型語言模型（LLaMA、GPT-NeoX）使用 **RoPE（Rotary Position Embedding）**（Su et al. 2021）：

不再將位置編碼**加**到 Embedding，而是在計算注意力分數時，對 $Q$ 和 $K$ 的每對維度**旋轉**：

$$
q_m = R_m q, \quad k_n = R_n k
$$

$$
q_m^\top k_n = q^\top R_m^\top R_n k = q^\top R_{n-m} k
$$

其中 $R_\theta$ 是旋轉矩陣，注意 $R_m^\top R_n = R_{n-m}$ 只依賴**相對位置** $n - m$，使模型天然具備相對位置感知。

### B. 注意力複雜度的改進方向

| 方法 | 複雜度 | 思想 |
|---|---|---|
| 標準注意力 | $O(T^2 d)$ | 全連接 |
| Sparse Attention | $O(T\sqrt{T} d)$ | 只關注部分位置 |
| Longformer | $O(T d)$ | 局部 + 全局注意力 |
| FlashAttention | $O(T^2 d)$（計算同，IO 優化）| 分塊計算，減少記憶體 |
| Linear Attention | $O(T d^2)$ | 核函數近似 |

### C. 從 Pre-Knowledge 到 Transformer 的完整推導路徑

```
離散符號
    ↓ Embedding
連續向量 x_i ∈ ℝ^d
    ↓ 相似度 + Softmax
注意力權重 α_{i,j}
    ↓ 加權平均
Naive Self-Attention: C = softmax(XX^T)X
    ↓ 問題：對稱、無角色分離
引入 QKV 投影: W_Q, W_K, W_V
    ↓
Scaled Dot-Product Attention: softmax(QK^T/√d_k)V
    ↓ 多角度表示
Multi-Head Attention: Concat(head_1,...,head_H)W_O
    ↓ 位置資訊缺失
+ Positional Encoding: x̃_i = x_i + p_i
    ↓ 非線性特徵變換
+ FFN: ReLU(XW_1)W_2
    ↓ 訓練穩定性
+ Residual + LayerNorm
    ↓
完整 Transformer 編碼器層 ✓
```

---

*本文件為 Transformer 數學基礎系列的第二篇（完結）。系列涵蓋：Pre-Knowledge（Naive Self-Attention）→ Part 1（注意力提示、核回歸、評分函數）→ Part 2（Bahdanau、Multi-Head、位置編碼、FFN）。*
