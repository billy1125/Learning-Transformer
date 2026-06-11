# 學習路徑改善計劃 01

> 撰寫日期：2026-06-11
> 前提：`improvement-plan.md` 的 P1–P4 全部完成，本文件規劃**下一階段**的方向。
> 兩條主軸：（A）主線路徑的剩餘缺口、（B）與當代 Transformer 架構的對齊。

---

## 一、現況評估

P1–P4 完成後，六份文件在以下面向已達到高品質：

- 數值範例覆蓋了所有關鍵步驟（QKV 計算、Causal Masking、LayerNorm 梯度）
- 圖表（ASCII）補齊了 Multi-Head 並行架構、Block 方塊、LayerNorm 路徑
- 各節銜接語引導讀者知道「為什麼要讀下一節」

**仍存在兩類缺口：**

1. **主線缺口（A 類）**：主線路徑（01→02→03→04→NB4）中有幾個概念「閃過去了」，讀者實際打開 nanoGPT 時會遇到但文件沒有解釋。
2. **架構缺口（B 類）**：nanoGPT 是 2022 年的教學模型。讀完後想看 LLaMA、GPT-4、Qwen 的讀者會遇到陌生的元件（RoPE、RMSNorm、SwiGLU、GQA），目前沒有橋接。

---

## 二、A 類：主線路徑剩餘缺口

這類改善**不新增文件**，直接修補現有六份理論文件。

---

### A1　03 文件：W_O（Output Projection）說明不足

**問題：**
§5 Multi-Head Attention 的 ASCII 圖顯示 Concat 之後有 $W_O$，但只有一行說明。讀者不清楚：
- 為什麼 Concat 後需要再做一次線性投影？
- $W_O$ 的維度是多少？它「消化」了什麼？

**當前文字（§5 末尾）：**
```
Concat (T×d)  →  W_O  →  Output (T×d)
```
沒有解釋。

**建議修改：**
在 §5 的數值例（H=2, d=4, d_k=2）之後，補一段：

> $W_O \in \mathbb{R}^{d \times d}$，作用是**把 H 個 head 獨立學到的特徵「混合重組」**。
> 如果沒有 $W_O$，每個 head 的資訊只是被拼在一起，彼此之間沒有互動。
> 加了 $W_O$ 之後，位置 $i$ 的輸出是 H 個 head 的線性組合，模型可以學習「哪個 head 的資訊在這個情境下更重要」。
>
> Shape：$[T \times d] \cdot [d \times d] = [T \times d]$（維度不變，可接 Residual）

**優先級：P1 | 難度：低 | 建議文件：`03-transformer-architecture.md` §5**

---

### A2　03 文件：FFN 設計動機未說明

**問題：**
§6.3 Feed-Forward Network 展示了公式 $\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$，但沒有解釋：
- 為什麼是「4x expansion then back」（$d \to 4d \to d$）？
- Attention 已經整合了不同 token 的資訊，FFN 做的是什麼不同的事情？
- 為什麼要用 ReLU/GELU 而不是線性？

這是讀者最常問的「這層為什麼存在」問題。

**建議修改：**
在 §6.3 FFN 公式後補 3 句直覺說明：

> **FFN 的角色**：Attention 負責「整合序列中不同位置的資訊」，FFN 負責「對每個位置獨立地做非線性轉換」。
> 可以把 FFN 想成一個「查閱表」——把 Attention 輸出的向量映射到一個更豐富的表示，再壓回原始維度。
> 4x expansion（$d \to 4d$）是 Vaswani 2017 的經驗設計，給中間層足夠的「展開空間」來學習複雜的映射；
> 非線性（ReLU/GELU）讓 FFN 能表達線性投影無法學到的函數。

**優先級：P1 | 難度：低 | 建議文件：`03-transformer-architecture.md` §6**

---

### A3　04 文件：Learned PE 與 Sinusoidal PE 的銜接

**問題：**
`03` 完整推導了 Sinusoidal Positional Encoding（$\sin/\cos$ 公式）。
`04` 的 nanoGPT 使用 `nn.Embedding` 做 **Learned PE**，只有一行程式碼，沒有橋接說明。

讀者看到 nanoGPT 的 `position_embedding_table = nn.Embedding(block_size, n_embd)` 時，會疑惑：
「這和 03 講的 PE 是同一件事嗎？為什麼長得不一樣？哪個比較好？」

**建議修改：**
在 `04` §5.2（Embedding 部分）後補一個對比框：

| | Sinusoidal PE（03 所介紹）| Learned PE（nanoGPT 所用）|
|---|---|---|
| 參數量 | 無（固定公式）| $T_{\max} \times d$（可訓練）|
| 泛化超出訓練長度 | 理論上可以 | 不能（沒看過的位置沒有 embedding）|
| 表達能力 | 固定模式 | 更靈活，由資料決定 |
| 現代模型 | 幾乎不再使用 | 早期 GPT-2；現代多用 RoPE（見 B3）|

> **結論**：nanoGPT 用 Learned PE 是因為簡單，也因為訓練語料長度固定（block_size=256）。
> 生產模型需要處理任意長度時，才需要 RoPE 等設計。

**優先級：P1 | 難度：低 | 建議文件：`04-gpt-decoder-only.md` §5**

---

### A4　04 文件：Dropout 完全未解釋

**問題：**
nanoGPT 在三個地方使用 dropout（Attention weights、FFN output、Residual 之前），但所有六份文件都沒有解釋 dropout 的作用。

讀者第一次看到 `self.dropout(wei)` 時，不明白：為什麼把一些值隨機歸零能讓模型更好？

**建議修改：**
在 §5 nanoGPT 解析中，第一次出現 `dropout` 的地方加一段說明：

> **Dropout**：訓練時隨機把 p% 的神經元輸出設為 0，迫使模型不能依賴任何單一路徑。
> 直覺上類似「考試時遮住幾個數字，強迫你記住整張表而不是死背某幾格」。
> nanoGPT 在 Attention weights 上做 dropout（`attn_dropout`），讓每個 token 不過度依賴某個固定的注意力模式；
> 在殘差路徑上做 dropout（`resid_dropout`），防止殘差直接跳過整個 Block。
> **推理時**（`model.eval()`）dropout 關閉，所有連接都恢復。

**優先級：P1 | 難度：低 | 建議文件：`04-gpt-decoder-only.md` §5**

---

### A5　04 文件：Weight Tying 未解釋

**問題：**
nanoGPT 在 `__init__` 末尾有一行：

```python
self.lm_head.weight = self.transformer.wte.weight
```

這讓 token embedding 矩陣和輸出的線性層**共用同一份參數**。
這是一個重要的設計決策（減少約 25% 參數量），但目前文件沒有解釋。

**建議修改：**
在 §5.4 或 §5.5（`GPT` 類別解析）中加一段：

> **Weight Tying（權重共享）**：輸入側把 token ID 查表得到向量（`wte: vocab_size × n_embd`），輸出側把向量映射回詞彙表（`lm_head: n_embd × vocab_size`）——這兩個矩陣的形狀互為轉置。
> 共用參數的邏輯：「意義接近的詞，embedding 向量接近；接近的向量，預測時也應該分配相近的機率。」
> 實作上共用同一份矩陣，模型的 embedding 訓練得更好，同時參數量減少 `vocab_size × n_embd`（nanoGPT 約 2.5M 參數）。

**優先級：P2 | 難度：低 | 建議文件：`04-gpt-decoder-only.md` §5**

---

### A6　04 文件：KV Cache 推理加速

**問題：**
§8 自迴歸生成的程式碼示範了每一步 forward 整個 `idx`，但這在長序列時非常低效：
每多生成一個 token，就要重新計算前面所有位置的 K 和 V。

KV Cache 是現代推理引擎（vLLM、TensorRT-LLM）的核心技術，也是理解「為什麼 context window 大了之後推理貴這麼多」的基礎。

**建議修改：**
在 §8 之後新增一節「§8.1 推理效率：KV Cache」：

```
沒有 KV Cache（nanoGPT 的做法）：
  step t：計算位置 0..t 的全部 K, V → 輸出位置 t+1
  step t+1：重新計算位置 0..t+1 的全部 K, V → 輸出位置 t+2
  → 每步的計算量 O(t)，總計算量 O(T²)

有 KV Cache：
  step t：計算位置 t 的 K_t, V_t，存入 cache
  step t+1：只計算位置 t+1 的 Q_{t+1}，與 cache 中的 K₀..K_t 做 attention
  → 每步的計算量 O(1)，總計算量 O(T)
  → 代價：VRAM 需要多存 2 × n_layer × n_head × T × d_k 個值
```

**優先級：P2 | 難度：中 | 建議文件：`04-gpt-decoder-only.md` §8 後**

---

### A7　03 文件：Softmax 數值穩定性

**問題：**
§3 推導了為什麼要除以 $\sqrt{d_k}$（避免分數過大），但沒有提及 softmax 實作中另一個必要技巧：

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

直接用 $e^{x_i}$ 在 $x_i$ 很大時會 overflow。減去 max 值後數學等價，但數值穩定。

讀者看 PyTorch 原始碼或自己實作時會遇到這個問題，應該在理論層面先解釋。

**建議修改：**
在 §3.2（縮放的必要性）之後加一個「數值實作注意」框：

> **實作：Softmax 數值穩定性**
> 縮放可以控制分數的整體尺度，但不保證不 overflow。標準做法是在做 exp 前減去每行的最大值：
> $$\text{softmax}(e_i) = \frac{\exp(e_i - \max_k e_k)}{\sum_j \exp(e_j - \max_k e_k)}$$
> 數學上等價（分子分母同除 $\exp(\max)$），但 exp 的輸入變成 $\leq 0$，不會 overflow。
> PyTorch 的 `F.softmax` 內部已做這個處理，手寫時需要注意。

**優先級：P2 | 難度：低 | 建議文件：`03-transformer-architecture.md` §3**

---

### A9　01b 文件：Embedding 缺少 Lookup Table 機制與梯度說明

**問題：**
`01a` 剛補充了「Token ID → Lookup Table → 訓練」的完整說明，但 `01b`（數學版）的 §2 只有矩陣形式 $X \in \mathbb{R}^{T \times d}$ 和幾何意義，沒有解釋這個矩陣是怎麼產生的。

`01b` 的讀者數學背景較強，反而更需要形式化的 Lookup Table 說明：

**建議修改：**
在 `01b` §2 的「為什麼需要 Embedding」之後，補一段：

> **形式化：Embedding 作為查表（Lookup Table）**
>
> 設詞彙表大小為 $V$，令 $E \in \mathbb{R}^{V \times d}$ 為 Embedding 矩陣（可訓練參數）。
> 對 Token ID 為 $t_i \in \{0, 1, \ldots, V-1\}$ 的 token，其 embedding 為：
>
> $$x_i = E[t_i] = e_{t_i}^\top \in \mathbb{R}^d$$
>
> 等價地，令 $\delta_{t_i} \in \mathbb{R}^V$ 為第 $t_i$ 個 one-hot 向量，則 $x_i = \delta_{t_i}^\top E$。
>
> **梯度特性：** 反向傳播時，梯度 $\frac{\partial \mathcal{L}}{\partial E}$ 只有第 $t_i$ 列非零——未被本 batch 選中的 token，其 embedding 本步不更新。這意味著稀有詞需要更多訓練樣本才能讓 embedding 收斂。

**優先級：P2 | 難度：低 | 建議文件：`01b-prerequisites-math.md` §2**

---

### A10　04 文件：`nn.Embedding` 未解釋 Lookup Table 機制

**問題：**
§5.5 的架構表只寫「離散 token → 連續向量」，但讀者第一次看到 `nn.Embedding(vocab_size, n_embd)` 時，不清楚：
- 這行程式碼在記憶體裡分配了一個 `vocab_size × n_embd` 的矩陣
- forward pass 是直接索引取列（O(1)），不是矩陣乘法
- 為什麼 nanoGPT 的 `token_embedding` 和 `lm_head` 形狀互為轉置（Weight Tying 的前提）

**建議修改：**
在 §5.5 的 `token_embedding = nn.Embedding(vocab_size, n_embd)` 那行旁，補一段說明（可與 A5 Weight Tying 合併處理）：

> `nn.Embedding(V, d)` 內部是一個 $V \times d$ 的矩陣。forward 時輸入 token ID（整數），直接返回對應列——這就是「查表（Lookup）」。
> 注意 `lm_head = nn.Linear(d, V)` 的 weight shape 是 $V \times d$，與 `token_embedding.weight` 完全相同，這是 Weight Tying（A5）的幾何依據。

**優先級：P2 | 難度：低 | 建議文件：`04-gpt-decoder-only.md` §5.5**

---

### A11　05 文件：Embedding 梯度更新未涵蓋

**問題：**
`05` 詳細推導了 Self-Attention 三條梯度路徑與 LayerNorm 三條路徑，但完全未涵蓋 Embedding 矩陣 $E$ 的反向傳播。

這是一個重要的缺口：
1. Embedding 是整個計算圖的起點，梯度如何流到 $E$ 是完整反向傳播的最後一步
2. 稀疏更新的特性（只有被選中的列有梯度）是理解「訓練資料量 vs. 詞彙覆蓋率」的關鍵
3. NB3 手刻反向傳播時，讀者需要知道 embedding backward 的正確寫法

**建議修改：**
在 `05` 末尾（§5 LayerNorm 之後）新增 §6，包含完整的逐步數學推導：

> **§6　Embedding 矩陣的完整梯度推導**
>
> #### 6.1　符號定義
>
> | 符號 | 意義 | 維度 |
> |---|---|---|
> | $V$ | 詞彙表大小 | — |
> | $E \in \mathbb{R}^{V \times d}$ | Embedding 矩陣（含 Weight Tying） | $V \times d$ |
> | $t_i \in \{0,\ldots,V-1\}$ | 位置 $i$ 的 Token ID | — |
> | $x_i = E[t_i]$ | 位置 $i$ 的 embedding | $d$ |
> | $\delta_{t_i} \in \mathbb{R}^V$ | 第 $t_i$ 個 one-hot 向量 | $V$ |
> | $\hat{h}_i$ | lm_head 輸入（LN 後的 hidden state） | $d$ |
> | $z_i = \hat{h}_i E^\top$ | lm_head 輸出（logits） | $V$ |
> | $p^{(i)}_k$ | 位置 $i$ 預測 token $k$ 的機率 | — |
> | $y_i$ | 位置 $i$ 的目標 Token ID | — |
>
> #### 6.2　前向傳播（關鍵步驟）
>
> $$x_i = \delta_{t_i}^\top E \quad \text{（Lookup = one-hot 乘以 } E\text{）}$$
>
> $$z_i = \hat{h}_i E^\top \quad \text{（lm\_head，Weight Tying 共用 } E\text{）}$$
>
> $$p^{(i)}_k = \frac{\exp(z_i^{(k)})}{\sum_{j=1}^{V} \exp(z_i^{(j)})}$$
>
> $$\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p^{(i)}_{y_i}$$
>
> #### 6.3　反向傳播逐步推導
>
> **Step 1：Cross-Entropy + Softmax 合併梯度**
>
> 對 Cross-Entropy $\mathcal{L} = -\frac{1}{T}\log p^{(i)}_{y_i}$ 與 Softmax 合併求導（標準結果）：
>
> $$\frac{\partial \mathcal{L}}{\partial z_i^{(k)}} = \frac{1}{T}\left(p^{(i)}_k - \mathbf{1}[k = y_i]\right)$$
>
> 記 $\delta_i = \frac{\partial \mathcal{L}}{\partial z_i} \in \mathbb{R}^V$。白話：在正確 token 的位置值為 $\frac{p_{y_i}-1}{T}$（負值，梯度要把這個 logit 推高），其餘位置值為 $\frac{p_k}{T}$（正值，把其他 logit 往下壓）。
>
> **Step 2：lm_head 反向**（$z_i = \hat{h}_i E^\top$）
>
> 對輸入 $\hat{h}_i$（梯度往上傳）：
>
> $$\frac{\partial \mathcal{L}}{\partial \hat{h}_i} = \delta_i E \in \mathbb{R}^d$$
>
> 對參數 $E$（Weight Tying，輸出側梯度）：
>
> $$\frac{\partial \mathcal{L}}{\partial E}\bigg|_{\text{output}} = \frac{1}{T}\sum_{i=1}^{T} \delta_i^\top \hat{h}_i^\top \in \mathbb{R}^{V \times d}$$
>
> 取第 $k$ 列：$\displaystyle\frac{\partial \mathcal{L}}{\partial E[k]}\bigg|_{\text{output}} = \frac{1}{T}\sum_{i=1}^{T} \delta_i^{(k)} \cdot \hat{h}_i$
>
> **Step 3：穿越 LayerNorm 和各 Block**
>
> 此段梯度路徑在 §3–§5 已詳細推導。設最終到達 $x_i = E[t_i]$ 的梯度為：
>
> $$g_i = \frac{\partial \mathcal{L}}{\partial x_i} \in \mathbb{R}^d$$
>
> **Step 4：Lookup 反向**（$x_i = \delta_{t_i}^\top E$）
>
> 由鏈式法則：
>
> $$\frac{\partial \mathcal{L}}{\partial E} = \sum_{i=1}^{T} \frac{\partial \mathcal{L}}{\partial x_i} \cdot \frac{\partial x_i}{\partial E} = \sum_{i=1}^{T} g_i^\top \delta_{t_i}^\top$$
>
> 取第 $k$ 列（僅 $t_i = k$ 的項非零，因為 $\delta_{t_i}^{(k)} = \mathbf{1}[t_i = k]$）：
>
> $$\frac{\partial \mathcal{L}}{\partial E[k]}\bigg|_{\text{input}} = \sum_{\{i\,:\,t_i = k\}} g_i$$
>
> **Step 5：Weight Tying 下的總梯度**
>
> $E$ 同時作為輸入 Embedding 和輸出 lm_head，梯度來自兩條路徑：
>
> $$\boxed{\frac{\partial \mathcal{L}}{\partial E[k]} = \underbrace{\sum_{\{i\,:\,t_i = k\}} g_i}_{\text{輸入側（稀疏）}} + \underbrace{\frac{1}{T}\sum_{i=1}^{T} \delta_i^{(k)} \hat{h}_i}_{\text{輸出側（稠密）}}}$$
>
> **Step 6：Optimizer 更新**
>
> $$E[k] \leftarrow E[k] - \eta \cdot \frac{\partial \mathcal{L}}{\partial E[k]}$$
>
> #### 6.4　三個重要特性
>
> | 特性 | 輸入側 | 輸出側 |
> |---|---|---|
> | **稀疏性** | 只有出現過的 token 列有梯度 | 所有 $V$ 列都有梯度（稠密）|
> | **梯度來源** | 語言模型「讀入」時的表示學習 | 語言模型「預測」時的對比信號 |
> | **Weight Tying 的收益** | — | 稀有 token 即使沒被選為輸入，也能從輸出側持續收到梯度 |
>
> **實作注意：** PyTorch `nn.Embedding(sparse=True)` 只傳輸有梯度的列，在詞彙量 $V > 10^5$ 時顯著節省反向傳播的記憶體與頻寬。

**優先級：P1 | 難度：中 | 建議文件：`05-backpropagation.md` §6（新增）**

---

### A12　04 文件：Embedding 訓練方式的完整路徑未說明

**問題：**
`04` 的 §4 解釋了訓練目標（next-token prediction + cross-entropy），但從未把這個 loss 和「Embedding 矩陣 $E$ 是怎麼被更新的」連起來。

讀者看到這段程式碼之後：
```python
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
loss.backward()
optimizer.step()
```
只知道「loss 算出來、backward、更新」——不知道梯度如何從 loss 一路流回到 `token_embedding.weight`，以及哪些列（rows）會被更新、哪些不會。

**`01a` 雖然補充了高層說明，但缺少計算細節；`05` 只推導 Self-Attention 和 LayerNorm，未涵蓋從輸出層反傳到 Embedding 的完整路徑。**

**建議修改：**
在 `04` §4「Next-token Prediction 訓練目標」之後，新增 §4.1：

> **§4.1　Embedding 如何在一次訓練步驟中被更新**
>
> 每次 `loss.backward()` 都完整走一遍以下路徑（括號內為關鍵公式，完整推導見 `05` §6）：
>
> ```
> 前向傳播（forward pass）
> ─────────────────────────────────────────────────────────
> token_id (B, T)
>   ↓  x_i = E[t_i]          ← Lookup：取出 E 的第 t_i 列
> x_embed (B, T, d)
>   ↓  h_0 = x_embed + p_embed
>   ↓  Block_1 → ... → Block_L
> h_L (B, T, d)
>   ↓  LayerNorm → lm_head: z_i = LN(h_L_i) · E^T   （Weight Tying）
> logits (B, T, V)
>   ↓  p^(i)_k = softmax(z_i)_k
>   ↓  L = −(1/T) Σ log p^(i)_{y_i}
> loss（純量）
>
> 反向傳播（backward pass）
> ─────────────────────────────────────────────────────────
> Step 1｜Cross-Entropy + Softmax：
>   δ_i = ∂L/∂z_i，其中 δ_i^(k) = (1/T)(p^(i)_k − 1[k = y_i])
>   → 在正確 token 的位置減 1/T，其餘位置加 softmax 機率/T
>
> Step 2｜lm_head 反向（z_i = LN(h)_i · E^T）：
>   ∂L/∂LN(h_i) = δ_i · E         ← 梯度流向上一層（d 維）
>   ∂L/∂E|output = (1/T) Σ_i δ_i^T · LN(h_i)^T  ← E 的輸出側梯度（稠密，V×d）
>
> Step 3｜穿越 LayerNorm、Residual、FFN、Attention：
>   → 最終到達 x_embed 的梯度記為 g_i ∈ R^d
>
> Step 4｜Lookup 反向（x_i = E[t_i]）：
>   ∂L/∂E[k]|input = Σ_{i: t_i = k} g_i   ← E 的輸入側梯度（稀疏）
>
> Step 5｜Weight Tying 下 E 的總梯度：
>   ∂L/∂E[k] = Σ_{i: t_i=k} g_i           （輸入側，稀疏）
>             + (1/T) Σ_i δ_i^(k) · LN(h_i) （輸出側，稠密）
>
> Step 6｜Optimizer 更新：
>   E[k] ← E[k] − η · ∂L/∂E[k]
> ```
>
> **三個關鍵特性：**
> 1. **稀疏 vs 稠密**：輸入側梯度（Step 4）只更新出現過的 token 列；輸出側梯度（Step 2）每步都對 E 的所有列有貢獻。
> 2. **同 token 累加**：token $k$ 在同一序列出現 $m$ 次，Step 4 的梯度是 $m$ 個 $g_i$ 的加總。
> 3. **直覺含義**：出現頻繁的 token 每步都被更新，embedding 收斂快；稀有 token 需要大量訓練步驟才被充分觸及。

**優先級：P1 | 難度：低 | 建議文件：`04-gpt-decoder-only.md` §4 後新增 §4.1**

---

### A8　02 文件：「8 項限制」的交叉引用

**問題：**
`02` 第六章列出了簡化模型的 8 項限制（無位置編碼、單頭、無 FFN……），是非常好的橋接設計。
但目前只列了限制，沒有指出「讀哪份文件來了解解法」。

讀者看到「無位置編碼」時，不知道 §7 of `03` 解決了這個問題。

**建議修改：**
在 8 項限制的表格中新增「解法所在」欄：

| 限制 | 解法 | 文件位置 |
|---|---|---|
| 無位置編碼 | Sinusoidal / Learned PE | `03` §7 |
| 單頭 | Multi-Head Attention | `03` §5 |
| 無 FFN | Transformer Block | `03` §6 |
| 無 Residual | Residual Connection | `03` §6.2 |
| 無 LayerNorm | Pre-LN 架構 | `03` §6.3、`05` §4–§5 |
| 無 Causal Mask | Decoder-Only | `04` §3 |
| 無 Output Projection | $W_O$ | `03` §5（見 A1 補強後）|
| 無 Dropout | Regularization | `04` §5（見 A4 補強後）|

**優先級：P2 | 難度：低 | 建議文件：`02-attention-intuition.md` 第六章**

---

## 三、B 類：當代 Transformer 架構對齊

這類改善**建議新增 `theory/06-modern-transformer-variants.md`**，讓讀者知道「nanoGPT → LLaMA 之間發生了什麼」。
不修改現有文件（避免讓主線過重）；改在現有 `04` 末尾或 `advanced/` 加一句指路即可。

---

### B1　RoPE：旋轉位置編碼（Rotary Position Embedding）

**為什麼重要：**
2023 年後，幾乎所有主流模型（LLaMA 1/2/3、Mistral、Qwen、Gemma）都放棄 Sinusoidal PE，改用 RoPE。
nanoGPT 用 Learned PE，但讀者若要讀 LLaMA 原始碼，第一個就會遇到 RoPE。

**建議涵蓋：**
1. Sinusoidal PE 的限制（長度外插能力差、可加性假設不自然）
2. RoPE 的核心思想：把「位置資訊」編碼在 Q、K 的旋轉角度裡，而不是加在 embedding 上
3. 公式：$\text{RoPE}(x, m) = R_m x$，其中 $R_m$ 是塊對角旋轉矩陣
4. 為什麼 RoPE 對相對位置天然不變：$(R_m q)^\top (R_n k) = q^\top R_{m-n}^\top R_0 k$ 只與差 $m-n$ 有關
5. 與 ALiBi 的對比（另一種相對位置方案）

---

### B2　RMSNorm：更簡潔的 Normalization

**為什麼重要：**
LLaMA、Mistral、Qwen 全部使用 RMSNorm 取代 LayerNorm。
`05` 詳細推導了 LayerNorm 的三條梯度路徑，讀者學完後轉去看這些模型時會遇到陌生的公式。

**LayerNorm vs RMSNorm 的差異：**

| | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | $(x - \mu) / \sigma \cdot \gamma + \beta$ | $x / \text{RMS}(x) \cdot \gamma$ |
| 參數 | $\gamma, \beta$（scale + shift）| 只有 $\gamma$（scale）|
| 計算 | 需算 mean + variance | 只需算 RMS（$\sqrt{\frac{1}{d}\sum x_i^2}$）|
| 梯度路徑 | 三條（直接、mean、variance）| 兩條（直接、RMS）|
| 效果 | 相近，RMSNorm 稍快 | 目前主流選擇 |

**建議涵蓋：**
1. 為什麼 mean centering 不是必要的（RMSNorm 的假設）
2. 簡化的梯度推導（可與 `05` 形成對比）

---

### B3　SwiGLU / GELU：現代 FFN 的啟動函數

**為什麼重要：**
nanoGPT 的 FFN 用 ReLU：$\text{FFN}(x) = \max(0, xW_1)W_2$
LLaMA 的 FFN 用 SwiGLU，**多了一個門控矩陣**：$\text{FFN}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2$

這讓 FFN 從「2 個線性層 + 1 個啟動」變成「3 個線性層 + 1 個門控」，shape 計算完全不同。

**建議涵蓋：**
1. GELU vs ReLU 的曲線差異與梯度特性
2. Gated Linear Unit（GLU）：$\text{GLU}(x) = \sigma(xW) \odot xV$
3. SwiGLU = $\text{Swish}(xW) \odot xV$（Swish = $x \cdot \sigma(x)$）
4. 為什麼加門控能改善效果（稀疏激活的直覺）
5. shape 追蹤（nanoGPT FFN vs LLaMA FFN 的對比）

---

### B4　GQA：Grouped Query Attention

**為什麼重要：**
LLaMA 2 70B、LLaMA 3、Mistral 7B 都用 GQA，這是讓大型模型能在有限 VRAM 下推理的關鍵。

**MHA → GQA → MQA 的演化：**

```
MHA（Multi-Head Attention，nanoGPT 的做法）：
  H 個 head，每個 head 有自己的 W_Q, W_K, W_V
  → KV Cache 需要 2 × H 組 KV

GQA（Grouped Query Attention）：
  H 個 Query head，分成 G 組，每組共用 1 組 KV
  → KV Cache 只需要 2 × G 組 KV（G << H）

MQA（Multi-Query Attention）：
  H 個 Query head，全部共用 1 組 KV
  → KV Cache 只需要 2 × 1 組 KV
  → 極限情況，品質略降
```

**建議涵蓋：**
1. 為什麼 KV Cache 是推理的瓶頸（A6 的延伸）
2. GQA 如何在不犧牲太多品質的前提下縮小 KV Cache
3. shape 計算：Query head 和 KV head 數量不同時的矩陣操作

---

### B5　Flash Attention：高效 Attention 的演算法

**為什麼重要：**
Flash Attention（Dao et al., 2022）把 Attention 的 GPU 記憶體複雜度從 $O(T^2)$ 降到 $O(T)$，讓超長 context 成為可能。
現代 PyTorch 的 `F.scaled_dot_product_attention` 內部就是 Flash Attention。

理解它需要理解「為什麼矩陣乘法的瓶頸不在 FLOP 而在 HBM 讀寫」，這是一個計算機架構的概念。

**建議涵蓋：**
1. GPU 記憶體層次：HBM（高頻寬記憶體）vs SRAM（片上快取）
2. Naive Attention 的瓶頸：$QK^\top$ 矩陣在 $T$ 大時無法放入 SRAM，需要多次讀寫 HBM
3. Flash Attention 的思路：分塊（tiling）計算 + online softmax，避免把完整 $T \times T$ 矩陣寫入 HBM
4. 只建立直覺，不要求讀者推導完整演算法

---

## 四、改善優先順序

### A 類（現有文件補強）

| 優先 | 編號 | 文件 | 說明 | 難度 | 狀態 |
|---|---|---|---|---|---|
| **P1** | A1 | 03 | 補 $W_O$ 設計動機與 shape | 低 | 待執行 |
| **P1** | A2 | 03 | 補 FFN 4x expansion 直覺 | 低 | 待執行 |
| **P1** | A3 | 04 | 補 Learned PE vs Sinusoidal PE 對比表 | 低 | 待執行 |
| **P1** | A4 | 04 | 補 Dropout 說明（首次出現處）| 低 | 待執行 |
| **P1** | A12 | 04 | 新增 §4.1：Embedding 訓練完整路徑（forward → loss → backward → sparse update）| 低 | 待執行 |
| **P2** | A5 | 04 | 補 Weight Tying 說明 | 低 | 待執行 |
| **P2** | A6 | 04 | 新增 §8.1 KV Cache ASCII 示意 | 中 | 待執行 |
| **P2** | A7 | 03 | 補 Softmax 數值穩定性說明 | 低 | 待執行 |
| **P2** | A8 | 02 | 補「8 項限制」→「解法所在」欄 | 低 | 待執行 |
| **P2** | A9 | 01b | 補 Embedding Lookup Table 形式化 + 稀疏梯度說明 | 低 | 待執行 |
| **P2** | A10 | 04 | 補 `nn.Embedding` Lookup Table 機制說明 | 低 | 待執行 |
| **P1** | A11 | 05 | 新增 §6 Embedding 完整梯度推導（6 步驟，含稀疏 vs 稠密分析）| 中 | 待執行 |

---

### B 類（新文件 `06-modern-transformer-variants.md`）

| 優先 | 編號 | 主題 | 建議篇幅 |
|---|---|---|---|
| **P1** | B1 | RoPE | ≈ 500 字 + 公式推導 |
| **P1** | B2 | RMSNorm | ≈ 300 字 + 梯度對比 |
| **P2** | B3 | SwiGLU / GELU | ≈ 400 字 + shape 追蹤 |
| **P3** | B4 | GQA | ≈ 400 字 + ASCII 對比圖 |
| **P3** | B5 | Flash Attention | ≈ 500 字（直覺為主）|

建議文件結構：
```
06-modern-transformer-variants.md
├── 0. 閱讀地圖：nanoGPT → LLaMA 的七個差異
├── 1. RMSNorm（取代 LayerNorm）
├── 2. SwiGLU（取代 ReLU FFN）
├── 3. RoPE（取代 Sinusoidal/Learned PE）
├── 4. GQA（取代 MHA）
├── 5. Flash Attention（Attention 計算加速）
└── 6. 對比表：nanoGPT vs LLaMA 2 7B
```

---

## 五、不建議更動的部分

以下內容已達高品質，改動風險大於收益：

- **`05` 的完整梯度推導**：Self-Attention 三個梯度推導 + LayerNorm 三條路徑已完整，不增加內容。
- **`04` §9 速查清單**：7 個問題對 NB4 有精準映射。
- **`02` 附錄「常見易混淆點」**：5 個對比經過考驗，維持。
- **`03` §8 RNN vs Transformer 表格**：資訊密度高，維持。

---

## 六、建議執行順序

1. **先做 A1–A4**：四個 P1 修改都是低難度補充，1–2 小時內可完成，大幅降低讀者打開 nanoGPT 時的困惑。
2. **再做 A5–A8**：補齊 Weight Tying、KV Cache 等推理相關說明。
3. **最後做 B 類**：新增 `06` 文件，連結到 `04` 末尾，讓完成主線的讀者有明確的「進階路徑」。

---

*本計劃書作為下一輪改善的參考依據，每完成一項請在優先順序表中標記完成。*
