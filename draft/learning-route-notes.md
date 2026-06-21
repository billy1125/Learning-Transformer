# Transformer 學習路線與未來 AI 趨勢筆記

## 一、神經網路模型難度比較

### 入門級

| 模型          | 難度    |
| ----------- | ----- |
| MLP         | ★☆☆☆☆ |
| CNN         | ★★☆☆☆ |
| Autoencoder | ★★☆☆☆ |

### 中階

| 模型     | 難度    |
| ------ | ----- |
| RNN    | ★★★☆☆ |
| LSTM   | ★★★☆☆ |
| GRU    | ★★★☆☆ |
| ResNet | ★★★☆☆ |

### 進階

| 模型          | 難度    |
| ----------- | ----- |
| U-Net       | ★★★★☆ |
| Transformer | ★★★★☆ |
| ViT         | ★★★★☆ |

### 高階

| 模型  | 難度    |
| --- | ----- |
| VAE | ★★★★★ |
| GAN | ★★★★★ |

---

## 二、學到 Transformer 後的建議

### 不建議立刻跳下一個模型

應先確認是否真正理解 Transformer 的核心機制。

若只知道：

* Attention
* GPT 使用 Transformer

其實還不足以支撐後續學習。

---

### 建議掌握的核心內容

#### Attention

* Query (Q)
* Key (K)
* Value (V)

#### Self-Attention

理解：

* 為什麼要計算 QKᵀ
* 為什麼要除以 √d
* 為什麼要 Softmax

#### Multi-Head Attention

理解：

* 多個 Head 的意義
* 為何效果優於單一 Attention

#### Positional Encoding

理解：

* Transformer 本身沒有順序概念
* 位置資訊如何加入

#### Residual Connection

理解：

* 深層網路如何避免梯度消失

#### Layer Normalization

理解：

* 為什麼需要正規化

#### Encoder 與 Decoder

理解：

* BERT 使用 Encoder
* GPT 使用 Decoder

---

## 三、Transformer 後續學習路線

### 第一階段

Transformer

↓

BERT

↓

GPT

---

### 第二階段

LLaMA

↓

Qwen

↓

Gemma

↓

Mistral

---

### 第三階段

RAG（Retrieval-Augmented Generation）

學習：

* Embedding
* Vector Database
* Retrieval
* Similarity Search

---

### 第四階段

Agent

學習：

* Tool Calling
* Planning
* Workflow
* Multi-Agent

---

### 第五階段

Multimodal

學習：

* ViT
* CLIP
* Qwen-VL
* Gemini

---

### 第六階段

Reasoning Models

學習：

* RLHF
* Reinforcement Learning
* Self-Reflection
* Verification
* Test-Time Compute

---

## 四、Transformer 所需數學分級

---

# Level 1：基本（必學）

### 向量

* Vector
* Embedding

### 矩陣

* Matrix
* Matrix Multiplication

### Dot Product

* Similarity

### Softmax

* 機率分布

### Gradient

* 基本梯度概念

---

# Level 2：進階（建議學）

### Linear Algebra

* Projection
* Basis
* Linear Transformation

### Probability

* 機率分布
* 期望值

### Information Theory

* Entropy
* Cross Entropy

### Eigenvalue / Eigenvector

* PCA 基礎概念

---

# Level 3：旁門進階

### Cosine Similarity

應用：

* Embedding
* RAG

### Optimization

* SGD
* Momentum
* Adam

### SVD

應用：

* PCA
* LoRA

### Numerical Stability

理解：

* Softmax Overflow
* Floating Point Error

---

# Level 4：高階

### KL Divergence

應用：

* VAE
* RLHF
* PPO
* DPO

### Bayesian Theory

* 條件機率
* 不確定性估計

### Markov Process

* RL
* Agent

### Convex Optimization

* Loss Surface

---

# Level 5：非必要（可跳過）

### Measure Theory

測度論

### Functional Analysis

泛函分析

### Real Analysis

實分析

### Differential Geometry

微分幾何

### 證明導向統計學

例如：

* 大數法則證明
* 中央極限定理證明

實務上知道結論即可。

---

## 五、數學學習優先順序

### 第一優先

1. 向量
2. 矩陣
3. Dot Product
4. Softmax
5. Gradient

---

### 第二優先

6. Linear Algebra
7. Probability
8. Cross Entropy

---

### 第三優先

9. Cosine Similarity
10. Adam Optimizer
11. SVD

---

### 第四優先

12. KL Divergence
13. Bayesian Theory
14. Markov Process

---

## 六、未來是否會以推理模型（Reasoning Model）為主流？

### 結論

是，但不完全是。

未來主流更可能是：

> Transformer + Reasoning + Tool Use + Verification + Planning

而非單純的 Transformer。

---

### 第一代

傳統 LLM

特徵：

* 預測下一個 Token
* 依賴大量資料

代表：

* GPT-3
* LLaMA 1

---

### 第二代

Instruction Model

特徵：

* RLHF
* 指令遵循

代表：

* ChatGPT
* GPT-4

---

### 第三代

Reasoning Model

特徵：

* 多步推理
* 自我檢查
* 錯誤修正

代表：

* OpenAI o 系列
* DeepSeek-R1

---

### 第四代

Agent Model

特徵：

* 工具使用
* 長期任務規劃
* 工作流程管理

---

## 七、未來最有投資報酬率的學習順序

```text
Transformer
    ↓
BERT
    ↓
GPT
    ↓
LLaMA
    ↓
Embedding
    ↓
RAG
    ↓
Agent
    ↓
Multimodal
    ↓
Reasoning Model
    ↓
RLHF / RL
```

---

## 八、目前階段建議

如果已經學到 Transformer：

### 先確認自己是否能回答

* Q、K、V 如何產生？
* 為什麼要 Self-Attention？
* 為什麼要 Multi-Head？
* 為什麼要 Positional Encoding？
* 為什麼需要 Residual Connection？
* 為什麼需要 LayerNorm？
* GPT 與 BERT 的差異？
* Decoder-only Transformer 如何運作？

若以上能清楚解釋，

則可開始學習：

* GPT
* LLaMA
* Embedding
* RAG

而不必繼續深鑽 Transformer 的理論細節。
