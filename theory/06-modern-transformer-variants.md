# 06｜當代 Transformer 架構：從 nanoGPT 到 LLaMA

> **適合對象：** 完成主線（01–05 ＋ NB4）後，想閱讀 LLaMA、Mistral、Qwen 等當代開源模型原始碼的讀者。
>
> **讀完後你能做什麼：**
> - 說明 RMSNorm 與 LayerNorm 的公式差異與梯度路徑差異
> - 解釋 SwiGLU 的門控機制，追蹤 LLaMA FFN 的 shape 計算
> - 推導 RoPE 為什麼對相對位置天然不變
> - 描述 MHA → GQA → MQA 的演化，計算各自的 KV Cache 大小
> - 解釋 Flash Attention 為什麼能把記憶體複雜度從 $O(T^2)$ 降到 $O(T)$
>
> **前置文件：** [`04-gpt-decoder-only.md`](04-gpt-decoder-only.md)、[`05-backpropagation.md`](05-backpropagation.md)（§1 的梯度對比會用到）
>
> **定位：** 本文是主線的「出口」——讀完 nanoGPT 之後，打開 LLaMA 原始碼之前。

---

## 目錄

0. 閱讀地圖：nanoGPT → LLaMA 的七個差異
1. RMSNorm（取代 LayerNorm）
2. SwiGLU（取代 ReLU FFN）
3. RoPE（取代 Sinusoidal / Learned PE）
4. GQA（取代 MHA）
5. Flash Attention（Attention 計算加速）
6. 對比表：nanoGPT vs LLaMA 2 7B

---

## 0. 閱讀地圖：nanoGPT → LLaMA 的七個差異

nanoGPT 是 2022 年的教學模型，忠實還原 GPT-2（2019）的架構。2023 年後的主流開源模型（LLaMA 1/2/3、Mistral、Qwen、Gemma）在**不改變核心公式**的前提下，替換了幾乎每一個元件：

| # | 元件 | nanoGPT（GPT-2 風格）| LLaMA 風格 | 本文章節 |
|---|---|---|---|---|
| 1 | Normalization | LayerNorm | RMSNorm | §1 |
| 2 | FFN 啟動函數 | ReLU（原版 GPT-2 用 GELU）| SwiGLU | §2 |
| 3 | 位置編碼 | Learned PE | RoPE | §3 |
| 4 | Attention 頭設計 | MHA | GQA | §4 |
| 5 | Attention 實作 | 樸素矩陣乘法 | Flash Attention | §5 |
| 6 | Tokenizer | 字元級 | BPE / SentencePiece | `04` §7 已介紹 |
| 7 | Bias 項 | Linear 含 bias | 幾乎全部移除 bias | （簡化，無需專節）|

核心公式 $\text{softmax}(QK^\top/\sqrt{d_k})V$、Pre-LN Residual 結構、Decoder-Only 堆疊——這些主線學過的東西**全部不變**。本文每一節都是「你已經懂的概念」的變體。

---

## 1. RMSNorm（取代 LayerNorm）

### 1.1 公式對比

回顧 LayerNorm（`05` §5）：先減均值、再除以標準差、最後 scale + shift：

$$
\text{LayerNorm}(x)_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma_j + \beta_j
$$

RMSNorm（Zhang & Sennrich, 2019）把「減均值」和「shift」都拿掉，只保留「除以尺度、再 scale」：

$$
\text{RMSNorm}(x)_j = \frac{x_j}{\text{RMS}(x)} \cdot \gamma_j, \qquad
\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{k=1}^d x_k^2 + \epsilon}
$$

| | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | $(x - \mu) / \sigma \cdot \gamma + \beta$ | $x / \text{RMS}(x) \cdot \gamma$ |
| 參數 | $\gamma, \beta$（scale + shift）| 只有 $\gamma$（scale）|
| 計算 | 需算 mean + variance | 只需算 RMS |
| 梯度路徑 | 三條（直接、mean、variance）| 兩條（直接、RMS）|
| 效果 | 相近 | 相近，計算稍快，目前主流選擇 |

### 1.2 為什麼 mean centering 不是必要的？

RMSNorm 論文的核心假設：LayerNorm 的效果主要來自**re-scaling 不變性**（把向量縮放到固定尺度），而不是 re-centering（把均值拉到 0）。實驗顯示拿掉減均值後效果幾乎不變，但省下了一次對 $d$ 維的求和與減法，在大模型中累積成可觀的加速。

### 1.3 梯度對比（與 `05` §5 對照）

用與 `05` 相同的記號：$g^{\hat{x}}$ 為傳到歸一化輸出 $\hat{x} = x/r$ 的梯度，$r = \text{RMS}(x)$。

**簡短推導**（仿 `05` §5.4–5.6 的路徑分析，但只剩兩條路徑）：

$x_k$ 影響 loss 的路徑只有兩條——直接路徑（$x_k \to \hat{x}_k$）與 RMS 路徑（$x_k \to r \to$ 所有 $\hat{x}_j$）。先算 $r$ 對 $x_k$ 的導數（由 $r^2 = \frac{1}{d}\sum_m x_m^2 + \epsilon$，兩邊對 $x_k$ 微分得 $2r \cdot \frac{\partial r}{\partial x_k} = \frac{2x_k}{d}$）：

$$
\frac{\partial r}{\partial x_k} = \frac{x_k}{d \cdot r}
$$

因此 $\hat{x}_j = x_j \, r^{-1}$ 對 $x_k$ 的完整導數（直接路徑只在 $j=k$ 時出現）：

$$
\frac{\partial \hat{x}_j}{\partial x_k}
= \frac{\delta_{jk}}{r} + x_j \cdot \left(-\frac{1}{r^2}\right) \frac{\partial r}{\partial x_k}
= \frac{\delta_{jk}}{r} - \frac{x_j x_k}{d \, r^3}
$$

套用鏈式法則並用 $x_j = r\hat{x}_j$ 化簡：

$$
\frac{\partial \mathcal{L}}{\partial x_k}
= \sum_j g^{\hat{x}}_j \frac{\partial \hat{x}_j}{\partial x_k}
= \frac{g^{\hat{x}}_k}{r} - \frac{x_k}{d \, r^3} \sum_j g^{\hat{x}}_j x_j
= \frac{g^{\hat{x}}_k}{r} - \frac{\hat{x}_k}{r} \cdot \frac{1}{d}\sum_j g^{\hat{x}}_j \hat{x}_j
$$

整理成向量形式，RMSNorm 的輸入梯度為：

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{1}{r}\left(
g^{\hat{x}} - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})
\right)
$$

對照 LayerNorm 的最終公式（`05` §5.7）：

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{1}{r}\left(
g^{\hat{x}} - \underbrace{\text{mean}(g^{\hat{x}})}_{\text{mean 路徑}} - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})
\right)
$$

**差異恰好就是少了 mean 路徑那一項**——因為 RMSNorm 沒有 $\mu$，$x_j$ 影響 loss 只剩兩條路：直接路徑與 RMS 路徑。如果你推過 `05` §5 的三條路徑，把路徑 2 整條刪掉，就得到 RMSNorm 的反向傳播。

---

## 2. SwiGLU（取代 ReLU FFN）

### 2.1 從 ReLU 到 GELU

nanoGPT 的 FFN（`03a` §6.4）：

$$
\text{FFN}(x) = \text{ReLU}(xW_1)W_2
$$

ReLU 在 0 處不可導、負半軸梯度恆為 0（神經元可能「死掉」）。GELU（GPT-2 實際使用）把硬截斷換成平滑曲線：

$$
\text{GELU}(x) = x \cdot \Phi(x) \quad (\Phi \text{ 為標準常態 CDF})
$$

負值不會被完全歸零而是平滑衰減，梯度處處存在。

### 2.2 門控：GLU 與 SwiGLU

Gated Linear Unit（GLU）引入第二個投影做「門」：

$$
\text{GLU}(x) = \sigma(xW) \odot xV
$$

一路算內容（$xV$），一路算門控值（$\sigma(xW) \in (0,1)$），逐元素相乘——門控值決定內容「通過多少」。

SwiGLU 把 sigmoid 門換成 Swish（$\text{Swish}(x) = x \cdot \sigma(x)$）：

$$
\text{FFN}_{\text{SwiGLU}}(x) = \left(\text{Swish}(xW_1) \odot xW_3\right) W_2
$$

**為什麼門控有效？** 直覺上，普通 FFN 對所有輸入「一視同仁」地做同一個非線性變換；門控讓模型可以依輸入內容**動態決定哪些中間維度要啟用**，形成稀疏激活——不相關的特徵通道被門關掉，表達效率更高。

### 2.3 Shape 追蹤：nanoGPT FFN vs LLaMA FFN

```
nanoGPT FFN（2 個矩陣 + 1 個啟動）：
  x (T×d) → W_1 (d×4d) → ReLU → (T×4d) → W_2 (4d×d) → (T×d)
  參數量：2 × 4d² = 8d²

LLaMA FFN（3 個矩陣 + 1 個門控）：
  x (T×d) ─→ W_1 (d×d_ff) → Swish ─┐
                                    ⊙ (逐元素相乘, T×d_ff)
  x (T×d) ─→ W_3 (d×d_ff) ─────────┘
                                    → W_2 (d_ff×d) → (T×d)
  參數量：3 × d·d_ff
```

因為多了一個矩陣，LLaMA 把 $d_{ff}$ 從 $4d$ 縮到約 $\frac{8}{3}d$，讓總參數量與傳統 FFN 持平（$3 \cdot \frac{8}{3}d^2 = 8d^2$）。LLaMA 2 7B 的實際數字：$d = 4096$，$d_{ff} = 11008 \approx 2.69d$。讀原始碼時看到 `gate_proj`、`up_proj`、`down_proj` 三個 Linear，分別就是 $W_1$、$W_3$、$W_2$。

---

## 3. RoPE（取代 Sinusoidal / Learned PE）

### 3.1 加法式 PE 的限制

主線學過的兩種 PE 都是**加在 embedding 上**（$x_i \leftarrow x_i + p_i$）：

- **Learned PE**（nanoGPT）：沒看過的位置沒有 embedding，完全無法外插（`04` §5.5 的對比表）。
- **Sinusoidal PE**（`03a` §7.2）：理論上可外插，但實務上超出訓練長度後品質快速劣化；且「位置資訊用加法混進語意向量」這個假設本身不自然——位置和語意被攪在同一個向量裡，attention 需要自己學會把它們分開。

### 3.2 RoPE 的核心思想：旋轉，而不是相加

RoPE（Su et al., 2021）改變了注入位置的「位置」：不動 embedding，而是在計算 attention 之前，**把 Q 和 K 向量依其位置旋轉一個角度**：

$$
\text{RoPE}(x, m) = R_m \, x
$$

其中 $R_m$ 是塊對角旋轉矩陣——把 $d$ 維向量視為 $d/2$ 個 2D 平面，第 $k$ 個平面旋轉角度 $m\theta_k$（$\theta_k = 10000^{-2k/d}$，與 Sinusoidal PE 同款的多頻率設計）：

$$
R_m = \begin{bmatrix}
\cos m\theta_0 & -\sin m\theta_0 & & \\
\sin m\theta_0 & \cos m\theta_0 & & \\
& & \cos m\theta_1 & -\sin m\theta_1 \\
& & \sin m\theta_1 & \cos m\theta_1 \\
& & & & \ddots
\end{bmatrix}
$$

### 3.3 為什麼 RoPE 對相對位置天然不變？

關鍵性質：attention 分數是 Q 與 K 的內積。對位置 $m$ 的 query 和位置 $n$ 的 key：

$$
(R_m q)^\top (R_n k) = q^\top R_m^\top R_n \, k = q^\top R_{n-m} \, k
$$

（旋轉矩陣的性質：$R_m^\top R_n = R_{n-m}$，即「先轉 $-m$ 再轉 $n$」= 「轉 $n-m$」。）

**注意力分數只依賴相對距離 $n - m$，與絕對位置無關。** 這正是語言需要的性質：「形容詞修飾下一個名詞」這種模式，不管出現在句首還是第 1000 個 token，行為應該一樣。加法式 PE 只能讓模型「間接學出」這個性質，RoPE 直接把它內建在幾何結構裡。

這也帶來更好的長度外插潛力——後續的 NTK-aware scaling、YaRN 等長 context 技術，都是在 RoPE 的 $\theta_k$ 上做文章。

### 3.4 與 ALiBi 的對比

ALiBi（Press et al., 2021）是另一種相對位置方案：完全不動 Q、K，直接在注意力分數上**減去與距離成正比的懲罰**：

$$
E_{mn} = \frac{q_m^\top k_n}{\sqrt{d_k}} - \lambda \cdot (m - n)
$$

| | RoPE | ALiBi |
|---|---|---|
| 注入位置 | 旋轉 Q、K 向量 | 直接修改注意力分數 |
| 相對位置 | 內積自動只含 $n-m$ | 顯式以 $m-n$ 懲罰 |
| 長度外插 | 需搭配 scaling 技巧 | 天然較好（懲罰隨距離平滑增長）|
| 採用模型 | LLaMA、Mistral、Qwen（主流）| BLOOM、MPT |

---

## 4. GQA（取代 MHA）

### 4.1 動機：KV Cache 是推理的瓶頸

`04` §8.1 算過：KV Cache 需要存 $2 \times n_{\text{layer}} \times H \times T \times d_k$ 個值。以 LLaMA 2 70B（80 層、64 頭、$d_k=128$）生成 4096 tokens 為例，每條序列的 KV Cache 約 $2 \times 80 \times 64 \times 4096 \times 128 \times 2\,\text{bytes} \approx 10.7\,\text{GB}$——比很多 GPU 的整張 VRAM 還大，而且這是**每條並行請求**都要的量。

關鍵觀察：cache 裡存的是 K 和 V，**Query 不需要 cache**（每步現算）。所以縮小 KV Cache 的方法就是：減少 K、V 的頭數。

### 4.2 MHA → GQA → MQA 的演化

```
MHA（Multi-Head Attention，nanoGPT 的做法）：
  H 個 head，每個 head 有自己的 W_Q, W_K, W_V
  Q: ●●●●●●●●  (H=8)
  K: ●●●●●●●●  每個 Q head 配一組專屬 KV
  V: ●●●●●●●●
  → KV Cache 需要 2 × H 組

GQA（Grouped Query Attention，LLaMA 2 70B / LLaMA 3 / Mistral）：
  H 個 Query head 分成 G 組，每組共用 1 組 KV
  Q: ●●●●●●●●  (H=8)
  K: ●─┘●─┘●─┘●─┘  (G=4，每 2 個 Q head 共用)
  V: ●   ●   ●   ●
  → KV Cache 只需要 2 × G 組（G << H）

MQA（Multi-Query Attention，極限情況）：
  H 個 Query head 全部共用 1 組 KV
  Q: ●●●●●●●●  (H=8)
  K: ●←──全部共用
  V: ●
  → KV Cache 只需要 2 × 1 組，品質略降
```

GQA 是兩個極端的折衷：實驗顯示 $G = H/8$ 左右時，品質幾乎與 MHA 持平，KV Cache 縮小 8 倍。

### 4.3 Shape 計算

Query head 數 $H$ 與 KV head 數 $G$ 不同時，矩陣操作如下（LLaMA 2 70B：$H=64, G=8, d_k=128$）：

```
X (T×d) → W_Q → Q (T, H·d_k)   = (T, 64×128)   reshape → (64, T, 128)
X (T×d) → W_K → K (T, G·d_k)   = (T,  8×128)   reshape → ( 8, T, 128)
X (T×d) → W_V → V (T, G·d_k)   = (T,  8×128)   reshape → ( 8, T, 128)

計算 attention 前，把每組 KV「廣播」給組內的 H/G = 8 個 Q head：
K, V: (8, T, 128) → repeat_interleave → (64, T, 128)
之後與 MHA 完全相同：softmax(QK^T/√d_k)V，逐 head 並行
```

注意 $W_K, W_V$ 的參數量也縮小為 $H/G$ 分之一——GQA 同時省了參數和 cache。讀 LLaMA 原始碼時看到 `n_kv_heads` 與 `repeat_kv()`，對應的就是上面的 $G$ 和廣播操作。

---

## 5. Flash Attention（Attention 計算加速）

### 5.1 先理解瓶頸在哪裡：HBM vs SRAM

GPU 的記憶體有層次：

```
SRAM（片上快取）  ：~20 MB，  頻寬 ~19 TB/s   ← 快但極小
HBM （顯示記憶體）：~40–80 GB，頻寬 ~2-3 TB/s  ← 大但相對慢
```

現代 GPU 的算力（FLOPs）成長遠快於 HBM 頻寬。對 attention 這種計算，**瓶頸往往不是「算得不夠快」，而是「資料在 HBM 和 SRAM 之間搬運太多次」**。

### 5.2 樸素 Attention 的問題

樸素實作（nanoGPT 的寫法）需要把完整的 $T \times T$ 中間矩陣寫進 HBM 再讀出來：

```
1. 算 E = QK^T     → 把 T×T 矩陣寫入 HBM   （T=8192 時，fp16 佔 128 MB）
2. 算 A = softmax(E) → 從 HBM 讀回、再寫入
3. 算 C = AV        → 再從 HBM 讀回
```

記憶體佔用 $O(T^2)$，HBM 讀寫量 $O(T^2)$——$T$ 翻倍，記憶體與搬運量翻四倍。這就是長 context 困難的根源之一。

### 5.3 Flash Attention 的思路：分塊 + online softmax

Flash Attention（Dao et al., 2022）的核心：**永遠不要把完整的 $T \times T$ 矩陣寫進 HBM**。

1. **分塊（tiling）**：把 Q、K、V 切成能放進 SRAM 的小塊，一次只算一小塊 $Q_i K_j^\top$，算完立即用掉，不存。
2. **Online softmax**：softmax 需要整行的 max 和 sum 才能歸一化，但分塊時一次只看到一部分。解法是邊掃邊維護「目前為止的 max 和 sum」，每處理一個新塊就**回頭修正**之前累積的部分結果（數學上精確，不是近似）。

結果：HBM 佔用從 $O(T^2)$ 降到 $O(T)$，搬運量大幅減少，實際速度提升 2–4 倍——**計算的 FLOPs 沒有變少，省的全是記憶體搬運**。

### 5.4 對讀者的實際意義

- PyTorch 2.x 的 `F.scaled_dot_product_attention` 內部就會自動調用 Flash Attention——把 nanoGPT 的 `Head.forward` 換成這一行呼叫，行為相同、速度更快。
- 它是**精確算法**，不是近似：輸出與樸素實作在數值誤差範圍內相同。
- 完整演算法（含反向傳播的 recomputation 技巧）超出本文範圍，建立「瓶頸在 HBM 讀寫、解法是分塊」的直覺即可。

---

## 6. 對比表：nanoGPT vs LLaMA 2 7B

| | nanoGPT（本倉庫 NB4）| LLaMA 2 7B |
|---|---|---|
| 參數量 | ~10M | 7B |
| 層數 $n_{\text{layer}}$ | 6 | 32 |
| 維度 $d$ | 384 | 4096 |
| Attention | MHA（6 頭）| MHA（32 頭；70B 版用 GQA）|
| Normalization | LayerNorm（Pre-LN）| RMSNorm（Pre-LN）|
| FFN | ReLU，$d_{ff} = 4d$ | SwiGLU，$d_{ff} = 11008$ |
| 位置編碼 | Learned PE | RoPE |
| Context 長度 | 256 | 4096 |
| Tokenizer | 字元級（vocab 65）| SentencePiece BPE（vocab 32000）|
| Bias | Linear 含 bias | 全部無 bias |
| Attention 實作 | 樸素矩陣乘法 | Flash Attention |

**不變的部分**：Decoder-Only、Causal Mask、Pre-LN Residual 結構、$\text{softmax}(QK^\top/\sqrt{d_k})V$、next-token prediction + cross-entropy。主線學到的骨架完全通用，變的只是每個元件的「實作選型」。

> nanoGPT 欄的數字（`n_embd=384`、6 層、context 256）取自 NB4 註解區塊中的「完整版超參數」，用來對照 LLaMA 規模；NB4 目前預設啟用的是可在 CPU 上跑完的輕量驗證版（`n_embd=64, n_head=2, n_layer=2, block_size=64`），骨架與對比結論不受影響。

---

## 下一步

讀完本文，你已經具備閱讀當代開源模型原始碼的所有概念。建議的驗證方式：

- 打開 [LLaMA 官方實作](https://github.com/meta-llama/llama/blob/main/llama/model.py)（約 500 行），對照本文逐節辨認 `RMSNorm`、`apply_rotary_emb`、`repeat_kv`、`gate_proj/up_proj/down_proj`
- 回到 [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)，動手把 LayerNorm 換成 RMSNorm、ReLU FFN 換成 SwiGLU，觀察訓練曲線的差異
