# Seq2Seq、Encoder、Decoder：從問題脈絡到解碼工藝

> **這是一篇延伸閱讀。** 內容整理自**李宏毅老師的機器學習課程**中關於 Sequence-to-Sequence 的上、下兩部分，改寫成一條連貫的故事線：先講「為什麼需要一種能把序列變序列的模型」，再看 Encoder 怎麼理解輸入、Decoder 怎麼生成輸出，最後談訓練與解碼時各種實務工藝。
>
> **它與主線的關係：** 本倉庫的主線（[`theory/`](../theory/) → nanoGPT）是**從零實作、數學優先**的路徑，聚焦 decoder-only 的 GPT。這篇則是**廣度與脈絡**的補充——用直覺把 Seq2Seq 的全景說清楚。凡是主線已經嚴謹推導過的（self-attention 的數學、殘差與 LayerNorm 的梯度、causal mask 的實作），本文只講直覺並附上「詳見主線某文件」的橋接，讓你能在「故事」與「數學」之間來回。
>
> 建議讀法：想先建立全局圖像的人，可以先讀本文再進主線；已經走完主線的人，可以把本文當成把知識接回真實應用（語音、翻譯、生成）與訓練工藝的收尾。

---

## 幕一・為什麼需要 Seq2Seq

### 一種「序列進、序列出」的模型框架

一般的分類問題，是輸入一筆資料、輸出一個類別：丟一篇評論進去，得到「正面」或「負面」。但很多真實任務不是這個形狀——它的輸出是**一整串東西**，而且事先不知道會有多長。

**Sequence-to-Sequence（Seq2Seq）** 就是為這種任務設計的框架，它處理的問題長這樣：

> 輸入是一個 sequence，輸出也是一個 sequence，而且輸出的長度通常無法事先固定。

關鍵差異在於四件事：輸入長度可能不固定、輸出長度也可能不固定、輸入與輸出長度之間沒有簡單固定的關係，而且**模型必須自己決定何時停止輸出**。這最後一點，之後會用一個特殊的「結束符號」來解決。

### 三個最典型的例子：語音辨識、機器翻譯、語音翻譯

**語音辨識**的輸入是一段聲音訊號，輸出是一串文字。使用者說出「機器學習」，模型收到的是一段連續的聲音訊號；聲音的長度可以用時間步表示，但輸出有幾個字，沒辦法由聲音長度直接推算。這就是一個標準的「聲音訊號序列 → 文字序列」問題。

**機器翻譯**的輸入是一種語言的句子，輸出是另一種語言的句子：

```text
機器學習 → Machine Learning
```

中文四個字、英文兩個詞，但這不代表所有中文句子翻成英文都會變一半長。不同語言的語序、詞彙密度、文法結構都不一樣，所以輸出長度只能讓模型自己學。

**語音翻譯**又更進一步——它不是先把聲音轉成同語言文字、再翻譯，而是**直接**把某種語言的聲音轉成另一種語言的文字：

```text
臺語聲音訊號 → 中文文字
```

這在現實中很有意義，因為世界上有許多語言沒有普及的文字系統，或者文字系統不是一般使用者熟悉的形式。臺語就是課程用的例子：臺語雖有文字與拼音，但一般使用者未必讀得順臺語書寫形式，所以「臺語語音 → 中文文字」對使用者反而更實用。

### 臺語案例的教學意義：端到端的威力與限制

課程提到，可以收集臺語語音與中文字幕的對應資料——例如從鄉土劇裡取得臺語聲音配上中文字幕。這種資料當然不乾淨：有背景音樂、字幕不完全對齊、各種雜訊。但即使如此，仍可能訓練出堪用的模型。

這個案例真正想讓人理解的，不是「可以無視資料品質」，而是幾件事：Seq2Seq 模型能**直接**學輸入到輸出的對應；不一定要人工設計中介步驟（例如先轉臺羅拼音再轉中文）；在資料足夠時，端到端模型有機會學到很複雜的轉換。當然，資料品質、語序轉換、詞彙對齊仍然會影響效果。課程裡用「硬 train 一發」這種口語來形容「直接把資料丟進去訓練」，正式一點的說法就是**以端到端（end-to-end）方式訓練**。

### 一個更大的觀點：很多 NLP 任務都能改寫成「序列到序列」

課程反覆強調一件事：許多 NLP 任務，都可以被改寫成 **QA（問答）或文字生成**的形式，於是都能用 Seq2Seq 來做。

- **翻譯當成 QA**：輸入某句英文，問題是「這句話的德文翻譯是什麼？」，輸出就是德文。
- **摘要當成 QA**：輸入一篇長文章，問題是「這篇文章的摘要是什麼？」，輸出是一段摘要。
- **情感分析當成 QA**：輸入一段評論，問題是「這段文字是正面還是負面？」，輸出「正面」或「負面」。

重點是：只要能把任務改寫成「輸入一段序列、輸出一段序列」，就可以嘗試用 Seq2Seq。

甚至看起來完全不像序列輸出的任務也能塞進來。**文法剖析（parsing）** 的輸出本來是一棵樹，但只要把樹狀結構用括號、標籤、詞彙**線性化**，就變成一串序列：

```text
(S (NP deep learning) (VP is (ADJP very powerful)))
```

於是輸入是句子、輸出是線性化後的樹，照樣能用 Seq2Seq。核心觀念是：**有些任務表面上不是序列輸出，但只要能把輸出格式轉成序列，就能被 Seq2Seq 處理。**

同樣的彈性也出現在 **multi-label classification**（一篇文章可能對應一個、兩個或多個標籤，數量不固定，就讓模型自己輸出一串標籤、自己決定幾個）與 **object detection**（輸入一張圖，輸出一串物件類別與位置）。後者不是最直觀的做法，但正好說明 Seq2Seq 框架有多能屈能伸。

### 泛用不等於最佳：瑞士刀的比喻

Seq2Seq 很泛用，但不代表它永遠是最好的解。課程用**瑞士刀**打比方：瑞士刀什麼都能做一點，但真要砍柴、切菜、修機器，專用工具往往更好。所以要分清楚兩件事——Seq2Seq 提供的是一種「把任務統一成同一種格式」的思路；而特定任務若用客製化模型，常常能得到更好的結果。例如語音辨識可以用 Seq2Seq，但實務上也有專為語音特性設計的 RNN Transducer 等模型。

### Transformer 在這張地圖上的位置

Transformer 可以視為一種 Seq2Seq 模型，它通常有兩個部分：

```text
Encoder：處理輸入序列
Decoder：產生輸出序列
```

Encoder 把輸入序列變成一排向量表示；Decoder 根據 Encoder 的輸出、加上自己已經產生過的 token，一步步生出目標序列。Transformer 之所以重要，是因為它用 **self-attention** 取代了傳統 RNN 的逐步處理，讓模型更能捕捉序列中任意兩個位置之間的關係。

接下來兩幕，就分別把 Encoder 與 Decoder 拆開來看。

---

## 幕二・Encoder 怎麼理解輸入

### 一排向量進、一排向量出

Transformer Encoder 的工作可以濃縮成一句話：

```text
輸入一排向量 → 輸出另一排向量
```

輸入序列裡每個 token 先被轉成向量，Encoder 收下這些向量後，輸出**同樣長度**的一排新向量。差別在於：每個輸出向量不再只是單一 token 自己的表示，而是**整合了上下文之後**的新表示。句子裡每個字原本都有自己的 embedding，經過 Encoder，每個位置的向量都會帶著它與其他 token 互動後的語意。

### Encoder block 的整體結構

Encoder 不是只有一層，而是由多個 block 疊起來，每個 block 大致是：

```text
輸入向量序列
→ Multi-head Self-Attention
→ Add & Norm
→ Feed-Forward Network
→ Add & Norm
→ 輸出向量序列
```

其中 Multi-head Self-Attention 讓每個 token 去參考其他 token；Add 是 residual connection，把輸入直接加到輸出；Norm 是 layer normalization，穩定數值分布；Feed-Forward Network 對每個位置的向量做非線性轉換。整個 Encoder 就是把這樣的 block 重複堆疊 N 次。

> 這套 block 的**嚴謹數學與逐步數值範例**（QKV 投影、$\text{softmax}(QK^\top/\sqrt{d_k})V$、多頭拼接、殘差與 LayerNorm 的梯度）在主線 [`theory/03a-transformer-architecture.md`](../theory/03a-transformer-architecture.md) 有完整推導；attention 的另一條數學史（含 Bahdanau）見 [`Attention-Mechanism-Part2.md`](Attention-Mechanism-Part2.md)。本文只講每個零件的直覺。

### 位置編碼：補上「順序」這個資訊

Self-attention 本身不天然知道 token 的順序——只看 attention，模型可能分不出某個 token 是在句首、句中還是句尾。所以除了 token embedding，還要加上 **positional encoding**：告訴模型每個 token 在序列中的位置。在 Transformer 裡，進 Encoder 前通常把 token embedding 與 positional encoding **相加**。（位置編碼的多種做法——正弦式、可學習式、乃至 RoPE——見 [`theory/03a`](../theory/03a-transformer-architecture.md) §7 與 [`theory/06`](../theory/06-modern-transformer-variants.md) §3。）

### Self-Attention 在做什麼

Self-attention 的核心功能是：讓每個位置的 token，根據**整個序列**的資訊來更新自己的表示。對每個 token，它會產生 Query、Key、Value，用自己的 Query 去和其他位置的 Key 算關聯度，再依這些 attention weight 對 Value 做加權總和。白話說，每個 token 都在問一句：

```text
我在理解自己時，應該參考句子中哪些其他 token？
```

例如句子裡的某個代名詞可能要參考前面的名詞、某個動詞要參考主詞或受詞——self-attention 提供的正是這種跨位置整合資訊的能力。

### 為什麼要 Multi-head

單一 attention head 只能用**一種**方式看待 token 之間的關係。Multi-head attention 用好幾組 head，讓模型同時從不同角度理解序列。不同 head 可能學到不同的關係：有的關注語法、有的關注長距離依賴、有的關注鄰近詞、有的關注特定語意角色。它的直覺是：

> 同一句話可以從多種關係角度被閱讀，而不是只用單一種注意力模式。

（多頭的維度切分與數值範例見 [`theory/03a`](../theory/03a-transformer-architecture.md) §5。）

### Residual Connection：留一條捷徑

Residual connection 就是把某一層的輸入，直接加到它的輸出上。在 Encoder block 裡，self-attention 的輸出會與它的輸入相加，feed-forward network 的輸出也與它的輸入相加：

```text
新輸出 = 子層輸出 + 子層輸入
```

它的好處是：避免深層網路在訓練時把早期資訊弄丟；讓梯度比較容易往前傳；也讓模型在需要時可以「保留原本表示」，而不是每一層都被迫完全改寫。白話說，residual 像在每層旁邊搭一條捷徑，資訊不必完全靠中間那段複雜轉換才能傳到下一層。（殘差為何能緩解梯度消失，見 [`theory/03a`](../theory/03a-transformer-architecture.md) §6 與梯度推導 [`theory/05`](../theory/05-backpropagation.md)。）

### Layer Normalization，以及它和 Batch Normalization 的差別

Layer normalization 的作用是穩定每一層的數值分布。它針對**同一個 example 裡、同一個向量的不同 dimension** 算 mean 與標準差，再做正規化。這和 batch normalization 不一樣——後者通常是在**一個 batch 裡、不同 examples 的同一個 dimension** 上算統計量。

| 方法 | 統計範圍 | 常見用途 |
|---|---|---|
| Batch Normalization | 同一 batch 中，不同 examples 的同一 dimension | CNN 等模型常見 |
| Layer Normalization | 同一 example 中，同一向量的不同 dimensions | Transformer 常見 |

在 Transformer 裡，layer normalization 常和 residual connection 搭在一起，就是架構圖上的 **Add & Norm**。

### Feed-Forward Network：逐位置的非線性加工

Encoder block 裡的 feed-forward network 是 **position-wise** 的——它對每個位置的向量**各自**做相同形式的轉換，但不同位置之間不在這個模組裡交換資訊。它負責對每個 token 的上下文表示再加工、提供非線性、增加表達能力。可以把 self-attention 與 FFN 的分工這樣理解：

| 模組 | 主要功能 |
|---|---|
| Self-Attention | 讓 token 之間交換資訊 |
| Feed-Forward Network | 對每個 token 自己的表示做非線性加工 |

### Add & Norm 的完整位置，以及「原版不是唯一解」

在原始 Transformer Encoder 中，每個 block 內有兩次 Add & Norm：

```text
Self-Attention → Residual Add → LayerNorm
→ Feed-Forward Network → Residual Add → LayerNorm
```

也就是常見架構圖裡的：

```text
Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm
```

要特別提醒：這個設計**不是唯一、也不永遠最佳**。後續研究探討過很多問題——layer normalization 該放在 residual 之後、還是放在 block 的輸入端（也就是 Pre-LN 與 Post-LN 之爭，主線 [`theory/04a`](../theory/04a-gpt-decoder-only.md) §7 與 [`04b`](../theory/04b-nanogpt-walkthrough.md) §7 有專門對比）？為什麼 Transformer 裡 layer norm 通常比 batch norm 更常用？能不能提出 power normalization 之類的替代方案？這些都說明 Transformer 是一個**可以持續改良**的架構，而不是固定不變的標準答案。

### Encoder 與 RNN、CNN 的關係

Encoder 要做的事——一排向量進、一排向量出——其實 RNN 與 CNN 也能做到類似形式，差別在於：RNN 依序處理 token，天然有順序性，但難以平行化；CNN 用局部視窗處理序列，可以平行化，但要抓長距離關係得疊很多層；self-attention 則能直接計算任意兩個位置之間的關係，比較容易捕捉長距離依賴。Encoder 以 self-attention 為核心，正是它和早期序列模型的重要分野。

### BERT 就是 Transformer Encoder

課程裡也點到：**BERT 基本上就是 Transformer Encoder 的延伸應用**。它不是完整的 encoder-decoder，而主要用 Transformer Encoder 來產生上下文表示：

```text
Transformer Encoder block × 多層堆疊 → BERT 的核心架構
```

所以把 Encoder 的 block 結構搞懂，對理解 BERT 非常關鍵。（BERT 的雙向注意力、MLM 預訓練、`[CLS]`/`[SEP]`、預訓練+微調，主線有一整份 [`theory/07-bert-encoder-only.md`](../theory/07-bert-encoder-only.md)。）

---

## 幕三・Decoder 與生成的工藝

Encoder 把輸入讀成一排向量之後，換 Decoder 上場，負責**一步一步生出輸出序列**。以語音辨識為例：

```text
聲音訊號 → Encoder → 一排向量 → Decoder → 文字序列
```

### Auto-Regressive Decoder：把自己的輸出當下一步的輸入

最常見的 Decoder 是 **Auto-Regressive（AR）decoder**。它的生成方式是：

```text
Begin → 產生第 1 個 token
Begin + 第 1 個 token → 產生第 2 個 token
Begin + 第 1、2 個 token → 產生第 3 個 token
...
直到產生 End token
```

也就是說，Decoder 每一步都會把自己前一步的輸出，當成下一步的輸入。以「機器學習」為例：

```text
Begin → 機
Begin, 機 → 器
Begin, 機, 器 → 學
Begin, 機, 器, 學 → 習
Begin, 機, 器, 學, 習 → End
```

好處是能建模輸出 token 之間的依賴關係；缺點是沒辦法一次生出整句，速度較慢。

### Error Propagation：一步錯，步步錯

正因為 AR decoder 會把自己的輸出當下一步輸入，一旦前面出錯，後面就可能跟著歪掉——這叫 **error propagation**：

> 一步錯，步步錯。

模型本來該輸出「機」，卻吐了個錯字，下一步它看到的上下文就已經不對了，後面很可能連鎖惡化。

### Masked Self-Attention：不准偷看未來

Decoder 的 self-attention 和 Encoder 不完全一樣。Encoder 可以一次看到整個輸入序列，但 Decoder 在生成時，只能看到**已經生成的左側 token**，不能看右側還沒出現的未來 token。所以 Decoder 用的是 **Masked Self-Attention**：

```text
產生第 1 個 token：只能看位置 1
產生第 2 個 token：只能看位置 1、2
產生第 3 個 token：只能看位置 1、2、3
```

正在生第 3 個位置時，只能看位置 1、2、3，不能看第 4 個。這就是 **causal mask**（也叫 look-ahead mask）的概念，目的是避免訓練時模型偷看未來答案。（它在 nanoGPT 裡怎麼用下三角矩陣把未來位置設成 $-\infty$、以及數值演示，見主線 [`theory/04a`](../theory/04a-gpt-decoder-only.md) §4。）

### End Token：讓生成自己停下來

Seq2Seq 的輸出長度通常不固定，所以 Decoder 必須**自己學會何時停**。做法是在 vocabulary 裡加一個特殊的 **End token**，模型一旦生出 End token，整個輸出就結束。相對地也有 **Begin token** 代表生成開始。實作上，Begin 與 End 可以是不同 token，也可以在某些助教程式裡共用同一個特殊符號，只要輸入與輸出的語境能區分就行。

### Non-Auto-Regressive Decoder：一次生出整句

**Non-Auto-Regressive（NAT）decoder** 想做的是：一次產生**整個**輸出序列，而不是一次一個 token。概念上像這樣：

```text
Begin, Begin, Begin, Begin → 機, 器, 學, 習
```

它最大的優點是平行化。AR decoder 要生 100 個 token 就得跑 100 步；NAT decoder 理論上一步就能生出整段，因此快很多。

但 NAT 有個麻煩：**要給幾個 Begin token？** 也就是輸出長度怎麼決定。課程提了兩種做法。第一，另外訓練一個 classifier 去預測輸出長度：先根據 Encoder 的輸入或輸出預測該生幾個 token，再讓 NAT decoder 生對應長度。第二，乾脆給一個很長的 Begin 序列——假設輸出最多 300 個 token，就給 300 個 Begin，模型生出 End token 後，End 右邊的輸出全部忽略。

NAT 的取捨大致是：優點是可平行、生成快、較容易控制輸出長度，在語音合成等任務有實用價值；缺點是效能通常不如 AR decoder、需要額外技巧才能逼近 AR 品質、還容易遇到 multimodality 問題。課程提到，語音合成裡 Tacotron 屬於較偏 AR 的模型，而 FastSpeech 則是 NAT 思路的重要例子。

### Cross Attention：Encoder 與 Decoder 之間的橋

到目前為止 Decoder 都只看自己。它是怎麼讀到 Encoder 的輸入資訊的？靠 **Cross Attention**。在 self-attention 裡 Query、Key、Value 都來自同一個序列；但在 cross attention 裡：

```text
Query 來自 Decoder
Key 來自 Encoder
Value 來自 Encoder
```

也就是說，Decoder 在生某個 token 時，用自己當下的狀態產生 Query，去 Encoder 的輸出裡尋找相關資訊。流程可以拆成這樣：

1. Encoder 輸出一排向量，例如 A1、A2、A3。
2. Decoder 目前狀態產生一個 Query。
3. Encoder 的向量產生 Key 與 Value。
4. Query 與每個 Key 算 attention score。
5. 經 softmax 得到 attention weight。
6. 用 attention weight 對 Value 做加權總和。
7. 得到的向量交給後面的 feed-forward network 產生輸出。

直覺是：

> Decoder 每產生一個 token，都會回頭看 Encoder 輸入裡哪些部分最相關。

值得一提的是，**cross attention 其實早於 Transformer**。早期的 Seq2Seq 語音辨識模型 Listen, Attend and Spell（LAS）就已經用了類似機制。這說明：cross attention 早於 Transformer、encoder-decoder 架構與 attention 概念在 Transformer 之前就存在，而 Transformer 真正的創新重點之一，是**大量使用 self-attention**。（原始 Transformer 的 encoder-decoder 與 cross-attention 分工，主線 [`theory/04a`](../theory/04a-gpt-decoder-only.md) §1 也有對照。）

### 訓練 Decoder：每一步都是一個分類問題

訓練 Decoder 時，每個輸出位置都可以看成一個**分類問題**。假設 vocabulary 有 4000 個中文字，每一步模型都要從這 4000 個類別裡選出正確的 token；模型輸出的是 softmax 後的機率分布，正確答案是一個 one-hot vector。訓練目標是讓輸出分布靠近正確答案，因此用 **Cross Entropy**。整句的 loss，可以理解為每個位置的 cross entropy 加總，而且最後還要包含 **End token** 的預測。（next-token 訓練與 cross-entropy 在 nanoGPT 裡的實作，見主線 [`theory/04a`](../theory/04a-gpt-decoder-only.md) §9。）

### Teacher Forcing 與它帶來的 Exposure Bias

訓練 AR decoder 有個技巧叫 **Teacher Forcing**：訓練時，把**正確答案**當作 Decoder 的輸入。例如正解是「機器學習」，訓練時模型在生「器」的那一步，輸入的是正確的「機」，而不是它自己剛剛預測出來的 token。這讓訓練更穩定、更有效率。

但它埋了一個問題：訓練時 Decoder 看到的是正確答案，測試時卻只能看到**自己產生的**答案。這種訓練與測試不一致，叫 **Exposure Bias**：

```text
訓練時：Decoder 看到正確答案
測試時：Decoder 看到自己的輸出
```

如果模型訓練時從沒看過「錯誤的上下文」，那測試時只要前一步出錯，它可能根本不知道怎麼收拾，於是又回到那個「一步錯、步步錯」的連鎖錯誤。

### Scheduled Sampling：讓模型見識一點錯誤

處理 exposure bias 的一種方法是 **Scheduled Sampling**：訓練時不要永遠餵正確答案，而是**偶爾**餵模型自己的輸出、或帶點錯誤的輸入，讓它學會在不完美的上下文裡繼續生成。不過對 Transformer 來說，傳統的 scheduled sampling 可能會影響平行化能力，所以在 Transformer 上會有一些變形做法。

### Copy Mechanism：有些字用抄的比較好

**Copy Mechanism** 讓 Decoder 不必**全部**從 vocabulary 生成，而是可以直接從**輸入序列**複製一段內容。這對很多任務很有用：聊天機器人複製使用者的名字或專有名詞、摘要從原文複製關鍵詞、問答從文章裡抽答案、翻譯保留人名地名。

舉個例子：使用者說「你好，我是酷洛洛。」模型回「酷洛洛你好。」——這裡的「酷洛洛」最好直接從輸入**複製**，而不是硬要模型在 vocabulary 裡生出這個罕見詞。

### Guided Attention：把先驗知識塞進 attention

**Guided Attention** 是在訓練時**引導 attention 依特定模式運作**。在語音辨識與語音合成裡，attention 通常應該由左到右移動；如果模型先看句尾、再看句首，可能導致漏字、重複、亂讀，或合成出來的語音不自然。Guided attention 的目的，就是把人類對任務的先驗知識加進訓練——例如語音合成中，輸入文字與輸出聲音大致該按順序對齊，就可以引導 attention 呈現**單調遞增**的趨勢。相關概念包括 monotonic attention 與 location-aware attention。

### Greedy Decoding 與 Beam Search

生成時每一步該怎麼挑 token？最簡單的是 **Greedy Decoding**：每步都選當下分數最高的。問題是，**局部最佳不一定通向整體最佳**——第一步選了分數最高的 token，可能害後面整條路徑變差；第一步稍微退讓、選個分數略低的，後面反而可能得到整體分數更高的句子。

**Beam Search** 就是為此而生：它保留**多條**候選路徑，而不是只留一條。它不是暴力窮舉所有路徑，而是用一個有限的「beam 寬度」保留幾條最有希望的候選序列。Beam search 適合答案比較明確的任務，例如語音辨識或機器翻譯；但對開放式的文字生成、故事續寫、聊天，beam search 反而可能導致輸出重複、僵硬，甚至「鬼打牆」。

### Decoder 需要一點隨機性

對某些任務來說，找出「機率最高」的輸出，不一定是人類覺得最自然的結果。故事續寫有很多都合理的答案、不是只有一個標準解；模型如果總是挑最高分 token，往往生出平庸又重複的句子。（這也是為什麼 nanoGPT 生成時用 temperature 與 top-k 取樣，而不是純 greedy——見主線 [`04b`](../theory/04b-nanogpt-walkthrough.md) §9 與 [`NB4`](../notebooks/NB4-nanoGPT.ipynb)。）

語音合成也可能需要隨機性。課程提到一個反直覺的現象：TTS 在**測試時**加入 noise，有時反而能產生比較自然的聲音——這跟一般機器學習的直覺不同，因為我們通常只在訓練時加 noise、不會在測試時加。這說明：Decoder 最好的解碼策略，取決於任務本身的特性。

### Cross Entropy 與 BLEU：訓練目標和評估目標的落差

訓練 Seq2Seq 常用 cross entropy 當 loss，但在機器翻譯或作業評估裡，往往用 **BLEU score** 來評估整句輸出。這就產生了落差：

```text
訓練目標：最小化 token-level 的 Cross Entropy
評估目標：最大化 sentence-level 的 BLEU score
```

兩者相關，但不完全一致——cross entropy 最低的模型，不一定 BLEU 最高。所以做 validation 挑模型時，可能要看 BLEU，而不是只看 cross entropy。

### 用 Reinforcement Learning 對付不可微分的目標

那能不能**直接**去最佳化 BLEU？麻煩在於 BLEU 是句子層級指標、而且**不可微分**，沒辦法當一般 gradient descent 的 loss。一種思路是把它丟給 **Reinforcement Learning**：把 BLEU score 當成 reward，把 Decoder 當成 agent，把「生成一整句」的過程看成一連串 action，於是就能嘗試直接最大化 BLEU。這種做法比較難，通常不是入門作業的首選，但它示範了一條路——當評估指標不可微分時，可以用 RL 繞過去。

---

## 尾聲・回到主線

這條故事線走完了：Seq2Seq 把「序列進、序列出」變成一個統一的問題框架，涵蓋語音辨識、翻譯、乃至一大票能被改寫成 QA 的 NLP 任務；Transformer 用 Encoder 把輸入讀成帶上下文的向量、用 Decoder 逐步生成輸出，中間靠 cross attention 溝通；而真正讓模型好用的，是一整套訓練與解碼的工藝——teacher forcing、exposure bias、scheduled sampling、copy、guided attention、beam search、取樣、乃至用 RL 直攻 BLEU。

如果你想把這些直覺換成能親手實作的數學與程式，回到主線：

- Encoder 的 block、self-attention、多頭、殘差與 LayerNorm 的嚴謹版 → [`theory/03a-transformer-architecture.md`](../theory/03a-transformer-architecture.md)
- Decoder-only、causal mask、next-token 訓練、自迴歸生成 → [`theory/04a-gpt-decoder-only.md`](../theory/04a-gpt-decoder-only.md)（原理）、[`04b-nanogpt-walkthrough.md`](../theory/04b-nanogpt-walkthrough.md)（程式）、[`NB4-nanoGPT`](../notebooks/NB4-nanoGPT.ipynb)
- BERT＝Encoder 的完整展開 → [`theory/07-bert-encoder-only.md`](../theory/07-bert-encoder-only.md)
- 更多延伸論文 → [`Suggested-Papers.md`](Suggested-Papers.md)
