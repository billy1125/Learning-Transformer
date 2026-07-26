## 結論

Transformer 學習路線可以分成三個階段：先打好「先備知識」，接著理解 Transformer 本體架構（不分流的共同核心），最後才依照 encoder-only、decoder-only、encoder-decoder 三個分支分別深入，並對應到目前主流模型。由於你已經理解 Q/K/V 推導、Multi-Head Attention、Attention Is All You Need (Vaswani et al., 2017）的原始論文架構，也具備線性代數的幾何直覺，先備知識部分可以快速複習，把重心放在架構分流與現代模型的銜接上。

---

## 一、先備知識（可快速複習）

在深入分流之前，建議確認以下概念已經掌握，這些是理解後續所有變形架構的基礎：

- **序列建模的基本問題**：為什麼 RNN（Recurrent Neural Network，循環神經網路）與 LSTM（Long Short-Term Memory，長短期記憶網路）會遇到長距離依賴（long-range dependency）與無法平行化訓練的問題，這是 Transformer 誕生的動機。
- **Self-Attention 與 Multi-Head Attention**：你已經熟悉 Q/K/V 的推導，這裡建議補強的是「為什麼要做 scaled dot-product」（避免 softmax 梯度消失）以及 Multi-Head 為什麼能捕捉不同子空間的關係。
- **Positional Encoding（位置編碼）**：原始論文用 sinusoidal（正弦）編碼，但後續模型多半改用 RoPE（Rotary Position Embedding，旋轉位置編碼）或 ALiBi（Attention with Linear Biases），這點會直接影響你理解現代模型為什麼能處理更長的上下文。
- **Layer Normalization 與 Residual Connection**：理解 Pre-LN 與 Post-LN 的差異，這會影響訓練穩定性，是後續大型模型普遍採用 Pre-LN 的原因。
- **Tokenization（分詞）**：BPE（Byte Pair Encoding）與 SentencePiece 的基本原理，因為這決定了模型輸入輸出的最小單位。

這些如果都已熟悉，可以直接跳到下一段。

---

## 二、Transformer 本體架構（共同核心）

不論最終走哪個分支，都需要理解 Vaswani et al.（2017）原始論文中完整的 encoder-decoder 架構，因為它是所有變形的共同祖先：

- Encoder 堆疊（stack）：Self-Attention + Feed-Forward Network，每層都有 residual connection 與 layer normalization。
- Decoder 堆疊：多了 Masked Self-Attention（避免看到未來 token）與 Cross-Attention（連接 encoder 輸出）。
- 訓練目標：原始論文是機器翻譯（machine translation），這也是為什麼它天生就是 encoder-decoder 架構。

理解這個共同核心之後，你會發現後續三個分支其實都是「拿掉某一部分」或「調整訓練目標」而來。

---

## 三、三個分流

### 1. Encoder-only（雙向理解型）

**核心概念**：只保留 encoder，訓練目標從「生成下一個字」改成「理解整段文字的雙向語境」。代表性訓練方式是 MLM（Masked Language Model，遮罩語言模型），也就是隨機遮住句子中的部分字詞，讓模型從上下文雙向推測。

**代表模型**：

- BERT（Bidirectional Encoder Representations from Transformers）：奠定 encoder-only 架構的經典模型。
- RoBERTa、DeBERTa：對 BERT 訓練方式與位置編碼的改良版本。

**銜接目前主流技術**：encoder-only 架構目前的主戰場已經轉移到 embedding model（嵌入模型），用於語意搜尋（semantic search）、RAG（Retrieval-Augmented Generation，檢索增強生成）的檢索端。例如 BGE、E5、gte 這類 embedding 模型，本質上都是 encoder-only 架構的延伸，是目前 RAG 系統中不可或缺的一環。

### 2. Decoder-only（生成型）

**核心概念**：只保留 decoder，拿掉 cross-attention（因為沒有 encoder 輸出可以對接），訓練目標是 next-token prediction（預測下一個 token），這是一種自迴歸（autoregressive）生成方式。

**代表模型**：

- GPT 系列（GPT-2、GPT-3）：奠定 decoder-only 架構的經典模型，也是你正在研究的 nanoGPT 所對應的架構原型。
- LLaMA、Mistral、Qwen：目前開源社群主流的 decoder-only 架構，加入了 RoPE、RMSNorm（RMS Normalization）、SwiGLU（一種 activation function 的變形）等優化。

**銜接目前主流技術**：這是目前最主流的分支，幾乎所有你聽過的大型語言模型（LLM，Large Language Model）都是 decoder-only 架構，包含 GPT-4、Claude、Gemini、LLaMA 系列等。你手上正在做的 nanoGPT notebook 就是這個分支最簡化的教學版本，理解它之後可以直接對應到現代 LLM 的核心邏輯，差異主要在規模、訓練資料量與若干工程優化（例如 Flash Attention、KV Cache）。

### 3. Encoder-Decoder（序列轉換型）

**核心概念**：保留完整架構，encoder 負責理解輸入，decoder 負責根據 encoder 輸出與已生成內容進行生成，透過 cross-attention 連接兩者。適合輸入與輸出是兩個不同序列的任務。

**代表模型**：

- T5（Text-to-Text Transfer Transformer）：把所有 NLP 任務都改寫成文字轉文字的形式，是這個分支的代表作。
- BART：結合了 BERT 式的雙向 encoder 與 GPT 式的自迴歸 decoder，常用於摘要、翻譯。

**銜接目前主流技術**：encoder-decoder 架構在純文字生成任務上目前已被 decoder-only 模型（透過 prompt 設計取代明確的任務分工）大幅取代，但在特定領域仍是主流，例如：

- 機器翻譯（NLLB、M2M-100）。
- 語音辨識與語音合成（Whisper 就是 encoder-decoder 架構，encoder 處理音訊特徵，decoder 生成文字）。
- 多模態模型中的部分架構設計，例如影像描述生成（image captioning）。

---

## 四、學習路線建議順序

1. 複習先備知識（如果已熟悉可跳過）。
2. 精讀 Vaswani et al.（2017）原始論文，理解完整 encoder-decoder 架構的每個元件。
3. 動手實作 decoder-only（你正在做的 nanoGPT notebook 就是最佳起點），因為這是目前最主流、也是你已經有進度的分支。
4. 完成 decoder-only 後，再回頭理解 encoder-only（可透過 Hugging Face 上的 BERT 或 embedding model 做小範圍實驗）。
5. 最後補齊 encoder-decoder，建議透過 Whisper 或 T5 的實際應用場景切入，會比單純讀論文更有感。
6. 全部理解後，可以往上銜接目前的工程優化議題：KV Cache、Flash Attention、RoPE 的長上下文擴展、MoE（Mixture of Experts）架構，這些是現代大型模型在 Transformer 基礎上的延伸。

由於你目前正在推進 nanoGPT notebook，建議把它當作 decoder-only 分支的實作主軸，其餘兩個分支可以先讀論文與跑現成模型體驗，之後有需要再深入實作。
