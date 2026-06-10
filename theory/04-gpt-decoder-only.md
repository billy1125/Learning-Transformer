# 04｜GPT Decoder-Only：從 Transformer 到 nanoGPT

> **適合對象：** 讀完 03 後，想理解 GPT 架構與原始 Transformer 的差異，並準備打開 nanoGPT Notebook 的讀者。
>
> **讀完後你能做什麼：**
> - 解釋為什麼 GPT 只有 Decoder，不需要 Encoder
> - 描述 Causal Masking 的動機與實作（下三角矩陣遮罩）
> - 說明 Next-token Prediction 如何定義訓練目標
> - 對照 `Head` / `MultiHeadAttention` / `Block` / `GPT` 類別與理論文件的對應關係
> - 解釋 Pre-LN 與 Post-LN 的差異
>
> **前置文件：** [`03-transformer-architecture.md`](03-transformer-architecture.md)
>
> **學完後的下一步：** → [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)

---

## 目錄

1. 原始 Transformer 是 Encoder-Decoder
2. GPT 為什麼只要 Decoder？
3. Causal Masking：只能看過去，不能偷看未來
4. Next-token Prediction：訓練目標的定義
5. nanoGPT 架構逐行解析
6. Pre-LN vs Post-LN：一個重要的實作差異
7. 字元級 Tokenizer
8. 自迴歸生成（Autoregressive Generation）
9. 打開 nanoGPT 之前的速查清單

---

## 1. 原始 Transformer 是 Encoder-Decoder

2017 年「Attention Is All You Need」提出的原始 Transformer 是為**翻譯任務**設計的 Encoder-Decoder 架構：

```
英文句子 → [Encoder] → K, V
                           ↓
中文前綴 → [Decoder] → [Cross-Attention: Q 來自 Decoder, K/V 來自 Encoder] → 下一個中文詞
```

- **Encoder**：雙向（bidirectional）— 每個位置可以看整個輸入序列（前後都看）
- **Decoder**：因果（causal）— 每個位置只能看自己和之前的輸出（不能偷看未來）
- 兩者通過 **Cross-Attention** 溝通：Decoder 的 Query 去查 Encoder 的 Key 和 Value

| 模組 | 看的範圍 | 用途 |
|---|---|---|
| Encoder Self-Attention | 雙向（全序列）| 理解輸入句的語意 |
| Decoder Self-Attention | 因果（只看過去）| 生成時不偷看未來 |
| Decoder Cross-Attention | Q 來自 Decoder，K/V 來自 Encoder | 對齊輸入與輸出 |

---

## 2. GPT 為什麼只要 Decoder？

GPT 的任務不是翻譯，而是**語言建模**：

> 給定一段文字 $x_1, x_2, \ldots, x_t$，預測下一個詞 $x_{t+1}$。

這個任務沒有「來源語言」需要 encode，只有一個文字串流。所以：

- 不需要 Encoder（沒有來源序列）
- 不需要 Cross-Attention（Encoder 不存在，無從查詢）
- 只需要 Decoder 的 **Causal Self-Attention**，讓每個位置只看自己和之前的 token

**結果：** GPT 把 Transformer 的 Decoder 拿出來，堆疊 $N$ 層，就是完整模型。

```
原始 Transformer：Encoder + Decoder + Cross-Attention
GPT：            只有 N 層 Causal Decoder Block
```

| 架構 | 模型代表 | 典型用途 |
|---|---|---|
| Encoder only | BERT | 分類、問答（雙向理解）|
| Decoder only | GPT, LLaMA | 文字生成、語言建模 |
| Encoder-Decoder | T5, 原始 Transformer | 翻譯、摘要（seq2seq）|

Decoder-Only 架構確定了，但還有一個問題：訓練時如果讓模型看到未來的詞，等於作弊——第 3 節說明如何用遮罩阻止這件事。

---

## 3. Causal Masking：只能看過去，不能偷看未來

### 動機

生成文字時，在產生位置 $t$ 的預測時，位置 $t+1, t+2, \ldots$ 的詞**還不存在**。如果 attention 讓模型看到未來，訓練時等於作弊，學不到真正的生成能力。

### 實作：下三角矩陣遮罩

在計算注意力分數後、softmax 之前，把「未來位置」的分數設為 $-\infty$：

```python
# 建立 T×T 的下三角矩陣（1 = 可看，0 = 遮罩）
tril = torch.tril(torch.ones(T, T))

wei = q @ k.transpose(-2, -1) * C**-0.5   # (B, T, T)  原始分數
wei = wei.masked_fill(tril[:T, :T] == 0, float('-inf'))  # 遮住未來
wei = F.softmax(wei, dim=-1)               # softmax 後 -inf → 0
```

遮罩矩陣的樣子（4 個 token 為例）：

```
位置 0 可看：[1, 0, 0, 0]   → 只看自己
位置 1 可看：[1, 1, 0, 0]   → 看 0 和 1
位置 2 可看：[1, 1, 1, 0]   → 看 0, 1, 2
位置 3 可看：[1, 1, 1, 1]   → 看全部
```

softmax 前：`[s00, -inf, -inf, -inf]` → softmax 後：`[1.0, 0.0, 0.0, 0.0]`

### 數值演示（T=3）

以 3 個 token 為例，假設縮放後的原始注意力分數為：

```
E_raw（縮放後）：
位置 0：[2.0,  1.0,  0.0]
位置 1：[1.0,  2.0,  1.0]
位置 2：[0.0,  1.0,  2.0]
```

**套用 Causal Mask（上三角設為 -∞）：**

```
E_masked：
位置 0：[2.0,  -inf, -inf]   → 只能看自己
位置 1：[1.0,   2.0, -inf]   → 能看 0 和 1
位置 2：[0.0,   1.0,  2.0]   → 能看全部
```

**逐行做 softmax（-∞ → 0）：**

| 位置 | softmax 輸入 | softmax 輸出 |
|---|---|---|
| 0 | `[2.0]`（只有自己）| `[1.000, 0.000, 0.000]` |
| 1 | `[1.0, 2.0]` | `[0.269, 0.731, 0.000]` |
| 2 | `[0.0, 1.0, 2.0]` | `[0.090, 0.245, 0.665]` |

**結果解讀：**
- 位置 0：100% 關注自己（沒有其他可看）
- 位置 1：73.1% 關注位置 1，26.9% 關注位置 0
- 位置 2：66.5% 關注位置 2，其餘分給位置 0、1

每一列加總恰好等於 1，且未來位置的權重為 0。這就是因果遮罩在訓練中防止「作弊」的方式。

### 關鍵差異

| | 原始 Encoder（BERT）| Causal Decoder（GPT）|
|---|---|---|
| 遮罩 | 無（全部可看）| 下三角（只看過去）|
| 注意力矩陣 | 對稱 | 下三角 |
| 適合任務 | 理解（分類/問答）| 生成（逐 token 輸出）|

---

## 4. Next-token Prediction：訓練目標的定義

### 資料準備

給定文字序列，把它切成 `(input, target)` 對：

```
文字： h  e  l  l  o
input:  [h, e, l, l]   (前 T 個 token)
target: [e, l, l, o]   (後 T 個 token，即 input 右移一位)
```

位置 $i$ 的 **input** 是 $x_i$，**target** 是 $x_{i+1}$。

### 損失函數

用 Cross-Entropy 比較模型輸出的機率分佈和正確答案：

```python
# logits: (B, T, vocab_size)
# targets: (B, T)
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
```

有了 Causal Mask，位置 $i$ 的 logit 只用到了 $x_1, \ldots, x_i$ 的資訊，預測 $x_{i+1}$，**不會洩漏未來**。因此一個序列可以同時訓練 $T$ 個預測任務，訓練效率極高。

訓練目標清楚了，第 5 節逐行解析 nanoGPT 如何把以上概念翻譯成不到 300 行的 Python 程式。

---

## 5. nanoGPT 架構逐行解析

### 5.1 `Head`：單頭 Causal Self-Attention

對應理論：[`03-transformer-architecture.md`](03-transformer-architecture.md) §1–§4

```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # W_K
        self.query = nn.Linear(n_embd, head_size, bias=False)  # W_Q
        self.value = nn.Linear(n_embd, head_size, bias=False)  # W_V
        # register_buffer：不是參數，但會跟模型一起存檔
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # K = X W_K
        q = self.query(x)  # Q = X W_Q
        wei = q @ k.transpose(-2, -1) * C**-0.5           # QK^T / sqrt(d_k)
        wei = wei.masked_fill(self.tril[:T,:T]==0, -inf)   # Causal mask
        wei = F.softmax(wei, dim=-1)                        # A = softmax(...)
        v = self.value(x)                                   # V = X W_V
        return wei @ v                                      # C = AV
```

**注意：** `C**-0.5` 中 `C = head_size`，即除以 $\sqrt{d_k}$，與理論一致。

### 5.2 `MultiHeadAttention`：多頭 Attention

對應理論：§5

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)  # W_O，輸出投影
```

- `num_heads` 個 `Head` 並行執行
- 輸出 concat 後乘以 $W_O$ 投影回 `n_embd`
- 等價於理論中的 $\text{Concat}(C^{(1)}, \ldots, C^{(H)}) W_O$

在 nanoGPT 中：`n_embd=384, n_head=6` → 每個 head 的 `head_size = 384/6 = 64`

### 5.3 `FeedForward`：Position-wise FFN

對應理論：§6.3

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # W_1，擴大 4 倍
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),   # W_2，壓縮回來
    nn.Dropout(dropout),
)
```

與理論的 $\text{ReLU}(Z'W_1 + b_1)W_2 + b_2$ 完全對應，`d_ff = 4 * n_embd`。

### 5.4 `Block`：完整 Transformer Block

對應理論：§6，但注意是 **Pre-LN**（見第 6 節）

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Residual + Pre-LN + Multi-Head Attention
        x = x + self.ffwd(self.ln2(x))  # Residual + Pre-LN + FFN
        return x
```

### 5.5 `GPT`：完整模型

```python
class GPT(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)   # Token embedding
        self.position_embedding = nn.Embedding(block_size, n_embd)   # 可學習位置編碼
        self.blocks  = nn.Sequential(*[Block(...) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)          # 最後一層 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size) # 輸出層：映射到詞彙表
```

**資料流：**

```
idx (B, T)
  → token_embedding → (B, T, n_embd)
  + position_embedding → (B, T, n_embd)     # 位置資訊
  → n_layer 個 Block → (B, T, n_embd)
  → LayerNorm
  → lm_head → logits (B, T, vocab_size)     # 每個位置預測下一個 token
```

### 架構對照總表

| nanoGPT 類別/方法 | 對應理論 | 關鍵操作 |
|---|---|---|
| `Head` | §3–§4 Scaled Dot-Product Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ + Causal Mask |
| `MultiHeadAttention` | §5 Multi-Head Attention | $H$ 個 Head concat + $W_O$ 投影 |
| `FeedForward` | §6.3 Position-wise FFN | Linear → ReLU → Linear |
| `Block` | §6 Transformer Block | Pre-LN + Residual × 2 |
| `GPT.token_embedding` | §2 Embedding | 離散 token → 連續向量 |
| `GPT.position_embedding` | §7.4 Learned PE | 可學習位置向量 |
| `GPT.lm_head` | 語言模型輸出層 | $\mathbb{R}^d \to \mathbb{R}^{|\mathcal{V}|}$ |
| `F.cross_entropy(...)` | 訓練目標 | Next-token prediction |

---

## 6. Pre-LN vs Post-LN：一個重要的實作差異

理論文件 (03) 描述的是原始論文的 **Post-LN**：

$$
Z' = \text{LayerNorm}(X + \text{Attention}(X))
$$

nanoGPT 使用 **Pre-LN**（現代模型的主流做法）：

$$
Z' = X + \text{Attention}(\text{LayerNorm}(X))
$$

```python
# Post-LN（原始論文）       # Pre-LN（nanoGPT）
x = LayerNorm(x + Attn(x)) # x = x + Attn(LayerNorm(x))
```

| | Post-LN | Pre-LN |
|---|---|---|
| 訓練穩定性 | 需要 warm-up | 更穩定，學習率更寬容 |
| 深層表現 | 容易梯度爆炸 | 梯度流更均勻 |
| 代表模型 | 原始 Transformer | GPT-2、LLaMA、nanoGPT |

架構設計清楚了，但模型怎麼讀取文字？第 7 節說明 nanoGPT 使用的字元級 tokenizer，以及與真實 BPE 的差異。

---

## 7. 字元級 Tokenizer

nanoGPT 使用最簡單的 tokenizer：**每個字元是一個 token**。

```python
# 建立詞彙表
chars    = sorted(list(set(text)))   # 全部不重複字元
vocab_size = len(chars)              # nanoGPT 莎士比亞資料集約 65

# 編碼 / 解碼
stoi = {ch: i for i, ch in enumerate(chars)}   # char → int
itos = {i: ch for i, ch in enumerate(chars)}   # int → char

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
```

**優點：** 實作簡單，詞彙表小（約 65 個字元）
**缺點：** 序列很長（一個詞需要 3–6 個 token），效率低

真實的 GPT 模型使用 BPE（Byte-Pair Encoding），詞彙表大小約 5 萬～10 萬，同樣的文字只需要約 1/4 長度的 token 序列。

訓練完成後，模型如何一個 token 一個 token 地生成新文字？第 8 節說明自迴歸生成的實作細節。

---

## 8. 自迴歸生成（Autoregressive Generation）

訓練完成後，用 `generate` 方法逐 token 生成文字：

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]        # 只取最後 block_size 個 token（context window）
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :]              # 只要最後一個位置的預測
        probs  = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1) # 依機率採樣（不是取 argmax）
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**關鍵細節：**

- `idx[:, -block_size:]`：模型的 context window 有上限（`block_size=256`），超出就截掉最舊的
- `logits[:, -1, :]`：只看最後一個時間步的輸出（它包含了前面所有 token 的資訊）
- `torch.multinomial`：依機率採樣，而非直接取最大值 → 輸出有多樣性

---

## 9. 打開 nanoGPT 之前的速查清單

確認以下問題都能回答，就可以打開 `NB4-nanoGPT.ipynb`：

| 問題 | 對應概念 |
|---|---|
| `Head` 裡的 `self.tril` 遮罩在做什麼？ | Causal Masking（§3）|
| `C**-0.5` 是什麼？ | $1/\sqrt{d_k}$ 縮放（03 §3.2）|
| `Block` 裡兩個 `x = x + ...` 是什麼結構？ | Residual Connection + Pre-LN（§6）|
| `lm_head` 輸出的 `(B, T, vocab_size)` 裡，哪個位置是訓練用的目標？ | 每個位置 $i$ 預測 $i+1$（§4）|
| 為什麼 `generate` 要截取 `idx[:, -block_size:]`？ | Context window 上限（§8）|
| `n_embd=384, n_head=6` → 每個 head 的維度是多少？ | $384/6=64$（03 §5）|
| 訓練和生成時 `targets` 的差異？ | 訓練時傳入 targets 算 loss；生成時不傳（§4, §8）|

---

## 下一步

**打開 Notebook：** [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)

按照 Notebook 的順序執行：超參數 → 資料載入 → 模型定義 → 訓練 → 視覺化 → 文字生成。

**完成 nanoGPT 後，若想深入理解訓練背後的數學：**
→ [`05-backpropagation.md`](05-backpropagation.md) — Self-Attention 與 LayerNorm 的完整梯度推導
→ [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb) — NumPy 手刻反向傳播
