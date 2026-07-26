# 04b｜nanoGPT 程式對照：逐行解析與數學回指

> **適合對象：** 讀完 [`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)（基本概念與 Pipeline）與 [`05a1-forward-propagation.md`](05a1-forward-propagation.md)（前向數學）後，想把每一段 nanoGPT 程式碼對回數學式、並準備打開 nanoGPT Notebook 的讀者。
>
> **讀完後你能做什麼：**
> - 對照 `Head` / `MultiHeadAttention` / `FeedForward` / `Block` / `GPT` 類別與 05a1 的數學節
> - 解釋 Pre-LN 與 Post-LN 在程式上的差異
> - 看懂 nanoGPT 的字元級 tokenizer 與自迴歸生成
> - 說明 KV Cache 為什麼能加速推理
>
> **前置文件：** [`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)（基本概念與 Pipeline）、[`05a1-forward-propagation.md`](05a1-forward-propagation.md)（前向數學）、[`03a-transformer-architecture.md`](03a-transformer-architecture.md)
>
> **學完後的下一步：** → [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)

---

## 目錄

1. `Head`：單頭 Causal Self-Attention
2. `MultiHeadAttention`：多頭 Attention
3. `FeedForward`：Position-wise FFN（含 Dropout）
4. `Block`：完整 Transformer Block
5. `GPT`：完整模型
6. 架構對照總表
7. Pre-LN vs Post-LN：一個重要的實作差異
8. 字元級 Tokenizer
9. 自迴歸生成（Autoregressive Generation）
10. 打開 nanoGPT 之前的速查清單

> **怎麼讀：** 每節先看程式，再回指 [`05a1`](05a1-forward-propagation.md) 對應的數學節（式子在那裡完整推導）；整體資料流與概念見 [`04a`](04a-gpt-decoder-only.md) 的 Pipeline 總覽。本文只負責「程式如何落實數學」，不重推公式。

---

## 1. `Head`：單頭 Causal Self-Attention

對應數學：[`05a1`](05a1-forward-propagation.md) §1（Scaled Dot-Product）＋ §2（Causal Masking）；幾何直覺見 [`03a`](03a-transformer-architecture.md) §1–§4

```python
class Head(nn.Module):
    def __init__(self, head_size):
        self.key     = nn.Linear(n_embd, head_size, bias=False)  # W_K
        self.query   = nn.Linear(n_embd, head_size, bias=False)  # W_Q
        self.value   = nn.Linear(n_embd, head_size, bias=False)  # W_V
        self.dropout = nn.Dropout(dropout)
        # register_buffer：不是參數，但會跟模型一起存檔
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # 注意 C = n_embd，不是 head_size
        k = self.key(x)    # K = X W_K
        q = self.query(x)  # Q = X W_Q
        d_k = k.shape[-1]                                   # d_k = head_size
        wei = q @ k.transpose(-2, -1) * d_k**-0.5           # QK^T / sqrt(d_k)
        wei = wei.masked_fill(self.tril[:T,:T]==0, -inf)   # Causal mask
        wei = F.softmax(wei, dim=-1)                        # A = softmax(...)
        wei = self.dropout(wei)                             # Dropout（見 §3）
        v = self.value(x)                                   # V = X W_V
        return wei @ v                                      # C = AV
```

逐行對回 [`05a1`](05a1-forward-propagation.md) §1 的式子：`self.key/query/value` 就是投影矩陣 $W_K,W_Q,W_V$；`q @ k.transpose(-2,-1)` 是 $QK^\top$；`* d_k**-0.5` 是除以 $\sqrt{d_k}$（見 05a1 §1 的方差論證）；`masked_fill(...,-inf)` 是 05a1 §2 的因果遮罩；`F.softmax` → `@ v` 就是 $A=\text{softmax}(\cdot)$、輸出 $AV$。

> **注意縮放要用 `head_size` 而不是 `n_embd`。** `B, T, C = x.shape` 解出的 `C` 是進入這個 head 的輸入維度，也就是 `n_embd`；但 $d_k$ 是 `head_size = n_embd // n_head`。多頭時（本倉庫 NB4 的 `n_head` 是 2 或 6）兩者並不相等，所以程式用 `k.shape[-1]` 取 $k$ 的最後一維，這才是 $d_k$。官方 nanoGPT 的 `model.py` 也是這樣寫（`1.0 / math.sqrt(k.size(-1))`）。
>
> Karpathy 影片版的早期程式碼寫成 `C**-0.5`，等於除以 $\sqrt{n\_embd}$——縮放常數偏大、attention 分佈偏平，不算致命但與 05a1 §1 的推導不符。本倉庫 NB4 已改為上面的寫法。

## 2. `MultiHeadAttention`：多頭 Attention

對應數學：[`05a1`](05a1-forward-propagation.md) §3

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)  # W_O，輸出投影
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concat(C¹, ..., C^H)
        return self.dropout(self.proj(out))                  # 乘 W_O 再 dropout
```

forward 的兩行與 05a1 §3 的公式 $\text{Concat}(C^{(1)}, \ldots, C^{(H)}) \, W_O$ 逐一對應：

- `[h(x) for h in self.heads]`：`num_heads` 個 `Head` 並行執行，各得 `(B, T, head_size)`
- `torch.cat(..., dim=-1)`：沿最後一維拼接 → `(B, T, n_embd)`（因為 `num_heads × head_size = n_embd`）
- `self.proj(out)`：乘以 $W_O$，把各 head 的資訊混合重組（05a1 §3 說明 $W_O$ 為何是可逆基底變換）
- `self.dropout(...)`：殘差路徑前的 dropout（見 §3 的 Dropout 說明）

在 nanoGPT 中：`n_embd=384, n_head=6` → 每個 head 的 `head_size = 384/6 = 64`（這是 NB4 註解區塊中「完整版超參數」的參考值；NB4 目前預設啟用的是跑得動 CPU 的輕量驗證版 `n_embd=64, n_head=2`，把 384/6 換成 64/2 一樣成立，$64/2=32$）

## 3. `FeedForward`：Position-wise FFN

對應數學：[`05a1`](05a1-forward-propagation.md) §4

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),   # W_1，擴大 4 倍
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),   # W_2，壓縮回來
    nn.Dropout(dropout),
)
```

與 05a1 §4 的 $\text{ReLU}(Z'W_1 + b_1)W_2 + b_2$ 完全對應，`d_ff = 4 * n_embd`。

**Dropout 是什麼？**（這是 dropout 在本文第一次出現）

訓練時隨機把 $p\%$ 的神經元輸出設為 0，迫使模型不能依賴任何單一路徑。直覺上類似「考試時遮住幾個數字，強迫你記住整張表而不是死背某幾格」。

nanoGPT 在三個地方使用 dropout：

| 位置 | 作用 |
|---|---|
| Attention 權重上（`Head` 中 softmax 之後）| 讓每個 token 不過度依賴某個固定的注意力模式 |
| MultiHead 輸出投影之後 | 防止殘差路徑直接記住 attention 的固定輸出 |
| FFN 輸出之後（上方程式碼）| 防止 FFN 過擬合訓練資料 |

**推理時**（`model.eval()`）dropout 自動關閉，所有連接都恢復。這就是為什麼生成文字前要呼叫 `model.eval()`。

## 4. `Block`：完整 Transformer Block

對應數學：[`05a1`](05a1-forward-propagation.md) §5，注意是 **Pre-LN**（見本文 §7）

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Residual + Pre-LN + Multi-Head Attention
        x = x + self.ffwd(self.ln2(x))  # Residual + Pre-LN + FFN
        return x
```

兩行都是「$x \leftarrow x + f(\text{LN}(x))$」的殘差結構，對應 05a1 §5 的 Pre-LN Block 公式；`ln1`/`ln2` 是兩個獨立的 LayerNorm。

## 5. `GPT`：完整模型

```python
class GPT(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)   # Token embedding
        self.position_embedding = nn.Embedding(block_size, n_embd)   # 可學習位置編碼
        self.blocks  = nn.Sequential(*[Block(...) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)          # 最後一層 LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size) # 輸出層：映射到詞彙表
```

**`nn.Embedding` 在做什麼？**

`nn.Embedding(V, d)` 內部就是一個 $V \times d$ 的矩陣。forward 時輸入 token ID（整數），直接返回對應的列——這就是「查表（Lookup）」，是 $O(1)$ 的索引操作，不是矩陣乘法。（數學形式化見 [`05a1`](05a1-forward-propagation.md) §6 與 [`01b`](01b-prerequisites-math.md) §2；它如何被訓練見 [`05b1`](05b1-backward-propagation.md) §10）

**Weight Tying（權重共享）——一個值得知道的設計**

注意 `lm_head = nn.Linear(n_embd, vocab_size)` 的 weight shape 是 $V \times d$，與 `token_embedding.weight` **完全相同**。輸入側把 token ID 查表得到向量，輸出側把向量映射回詞彙表——兩個矩陣的形狀互為轉置關係。

Karpathy 的原版 nanoGPT 因此讓兩者共用同一份參數：

```python
self.lm_head.weight = self.transformer.wte.weight   # Weight Tying
```

共用的邏輯：「意義接近的詞，embedding 向量接近；接近的向量，預測時也應該分配相近的機率。」實作上共用同一份矩陣，embedding 訓練得更好，同時參數量減少 `vocab_size × n_embd`（GPT-2 規模約 38M 參數；本倉庫的字元級模型約 2.5 萬）。本倉庫的 NB4 為求簡單，未做 Weight Tying，兩個矩陣獨立訓練。（Weight Tying 對梯度的影響見 [`05b1`](05b1-backward-propagation.md) §10.3）

**等一下——這個 `position_embedding` 和 03 講的 PE 是同一件事嗎？**

是同一個目的（注入位置資訊），但做法不同。03a §7.2 推導的是 Sinusoidal PE（固定公式），nanoGPT 用的是 `nn.Embedding` 實作的 **Learned PE**（[`05a1`](05a1-forward-propagation.md) §6、03a §7.5）——每個位置一個可訓練向量：

| | Sinusoidal PE（03a §7.2 所介紹）| Learned PE（nanoGPT 所用）|
|---|---|---|
| 參數量 | 無（固定公式）| $T_{\max} \times d$（可訓練）|
| 泛化超出訓練長度 | 理論上可以 | 不能（沒看過的位置沒有 embedding）|
| 表達能力 | 固定模式 | 更靈活，由資料決定 |
| 現代模型 | 幾乎不再使用 | 早期 GPT-2；現代多用 RoPE（見 [`06`](06-modern-transformer-variants.md)）|

> **結論**：nanoGPT 用 Learned PE 是因為簡單，也因為訓練語料長度固定（`block_size=256`）。
> 生產模型需要處理任意長度時，才需要 RoPE 等設計（見 [`06-modern-transformer-variants.md`](06-modern-transformer-variants.md)）。

**資料流：**

```
idx (B, T)
  → token_embedding → (B, T, n_embd)
  + position_embedding → (B, T, n_embd)     # 位置資訊
  → n_layer 個 Block → (B, T, n_embd)
  → LayerNorm
  → lm_head → logits (B, T, vocab_size)     # 每個位置預測下一個 token
```

## 6. 架構對照總表

| nanoGPT 類別/方法 | 對應數學（[`05a1`](05a1-forward-propagation.md)）| 關鍵操作 |
|---|---|---|
| `Head` | §1 Scaled Dot-Product ＋ §2 Causal Mask | $\text{softmax}(QK^\top/\sqrt{d_k})V$ + 下三角遮罩 |
| `MultiHeadAttention` | §3 Multi-Head Attention | $H$ 個 Head concat + $W_O$ 投影 |
| `FeedForward` | §4 Position-wise FFN | Linear → ReLU → Linear |
| `Block` | §5 LayerNorm 與 Pre-LN Block | Pre-LN + Residual × 2 |
| `GPT.token_embedding` | §6 Embedding（＋[`01b`](01b-prerequisites-math.md) §2）| 離散 token → 連續向量 |
| `GPT.position_embedding` | §6 Learned PE（＋03a §7.5）| 可學習位置向量 |
| `GPT.lm_head` | §7 語言模型輸出層 | $\mathbb{R}^d \to \mathbb{R}^{|\mathcal{V}|}$ |
| `F.cross_entropy(...)` | §7 訓練目標 | Next-token prediction |

---

## 7. Pre-LN vs Post-LN：一個重要的實作差異

[`05a1`](05a1-forward-propagation.md) §5 描述的原始論文做法是 **Post-LN**：

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

**為什麼 Pre-LN 比較穩定？** 關鍵在殘差主幹 $x + f(\text{LN}(x))$ 保留了一條**完全不經過 LayerNorm 的直通路徑**，梯度可以恆等流過；Post-LN 的 $\text{LN}(x + f(x))$ 則每穿一層都要經過 LayerNorm 的耦合，層數一深梯度尺度就容易失控，因此需要 learning rate warm-up。完整的梯度流推導見 [`05a1`](05a1-forward-propagation.md) §5（與 [`05b1`](05b1-backward-propagation.md) §5.9／§5.10）。

架構設計清楚了，但模型怎麼讀取文字？第 8 節說明 nanoGPT 使用的字元級 tokenizer，以及與真實 BPE 的差異。

---

## 8. 字元級 Tokenizer

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

訓練完成後，模型如何一個 token 一個 token 地生成新文字？第 9 節說明自迴歸生成的實作細節。

---

## 9. 自迴歸生成（Autoregressive Generation）

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

### 9.1 推理效率：KV Cache

仔細看上面的 `generate`：每生成一個新 token，都把**整個** `idx` 重新 forward 一遍。這意味著前面所有位置的 K 和 V 每一步都被重新計算——但它們根本沒變。

```
沒有 KV Cache（nanoGPT 的做法）：
  step t：  計算位置 0..t 的全部 K, V → 輸出位置 t+1
  step t+1：重新計算位置 0..t+1 的全部 K, V → 輸出位置 t+2
  → 每步的計算量 O(t)，生成 T 個 token 總計算量 O(T²)

有 KV Cache（vLLM、TensorRT-LLM 等推理引擎的做法）：
  step t：  只計算位置 t 的 K_t, V_t，存入 cache
  step t+1：只計算位置 t+1 的 Q_{t+1}，與 cache 中的 K₀..K_t 做 attention
  → 每步的計算量 O(1)，總計算量 O(T)
  → 代價：VRAM 需要多存 2 × n_layer × n_head × T × d_k 個值
```

兩個值得記住的推論：

1. **為什麼可以 cache？** Causal Mask 保證位置 $t$ 的 K、V 不受未來 token 影響——一旦算出來就永遠不變，可以安心重用（因果性見 [`05a1`](05a1-forward-propagation.md) §2）。
2. **為什麼 context 越長推理越貴？** KV Cache 的大小隨 $T$ 線性成長，長 context 模型（128K tokens）的推理瓶頸往往不是計算而是 VRAM。這也是 GQA 等技術出現的動機（見 [`06-modern-transformer-variants.md`](06-modern-transformer-variants.md) §4）。

nanoGPT 為了教學簡潔沒有實作 KV Cache，但讀懂它之後，看任何推理引擎的原始碼都會先遇到這個概念。

> **延伸閱讀：** 本節只講「怎麼一步步生成」。至於**解碼策略**（greedy vs beam search、取樣的隨機性）與 Seq2Seq 的**訓練工藝**（teacher forcing、exposure bias、scheduled sampling、copy mechanism、guided attention、CE vs BLEU、用 RL 直攻不可微分指標），整理成一篇流暢的故事在 [`../advanced/Seq2Seq-and-Decoding-Techniques.md`](../advanced/Seq2Seq-and-Decoding-Techniques.md)（含 Encoder-Decoder 與 NAT decoder 的脈絡）。

---

## 10. 打開 nanoGPT 之前的速查清單

確認以下問題都能回答，就可以打開 `NB4-nanoGPT.ipynb`：

| 問題 | 對應概念 |
|---|---|
| `Head` 裡的 `self.tril` 遮罩在做什麼？ | Causal Masking（[`05a1`](05a1-forward-propagation.md) §2）|
| `d_k**-0.5` 是什麼？ | $1/\sqrt{d_k}$ 縮放，$d_k=$ `head_size`（[`05a1`](05a1-forward-propagation.md) §1；幾何見 03a §3.4；別誤用 `n_embd`，見 §1 的註）|
| `Block` 裡兩個 `x = x + ...` 是什麼結構？ | Residual Connection + Pre-LN（本文 §7、[`05a1`](05a1-forward-propagation.md) §5）|
| `lm_head` 輸出的 `(B, T, vocab_size)` 裡，哪個位置是訓練用的目標？ | 每個位置 $i$ 預測 $i+1$（[`05a1`](05a1-forward-propagation.md) §7）|
| 為什麼 `generate` 要截取 `idx[:, -block_size:]`？ | Context window 上限（本文 §9）|
| `n_embd=384, n_head=6` → 每個 head 的維度是多少？ | $384/6=64$（[`05a1`](05a1-forward-propagation.md) §3）|
| 訓練和生成時 `targets` 的差異？ | 訓練時傳入 targets 算 loss；生成時不傳（[`05a1`](05a1-forward-propagation.md) §7、本文 §9）|

---

## 下一步

**打開 Notebook：** [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)

按照 Notebook 的順序執行：超參數 → 資料載入 → 模型定義 → 訓練 → 視覺化 → 文字生成。

**完成 nanoGPT 後，若想深入理解訓練背後的數學：**
→ [`05b1-backward-propagation.md`](05b1-backward-propagation.md) — 從 loss 到 Embedding 的完整梯度推導＋數值計算
→ [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb) — NumPy 手刻反向傳播

**完成 nanoGPT 後，若想銜接 LLaMA 等當代模型：**
→ [`06-modern-transformer-variants.md`](06-modern-transformer-variants.md) — RMSNorm、RoPE 等 nanoGPT → LLaMA 之間的架構演化
