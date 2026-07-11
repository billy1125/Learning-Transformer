# 03a 白話前導｜Transformer Block 到底在做什麼（超白話版）

> **適合對象：** 完全零基礎、想在讀 [`03a`](03a-transformer-architecture.md) 的數學之前，先用生活比喻抓住「一個 Transformer Block 有哪些零件、各自幹嘛」的讀者。
>
> **讀完後你能做什麼：**
> - 說出 Transformer Block 四個模組（Multi-Head Attention / FFN / Residual Connection / LayerNorm）各自負責什麼
> - 用一個比喻串起「先交換資訊 → 再個別整理 → 保留原稿 → 保持穩定」的完整流程
> - 說明為什麼「只有 Attention」還不等於一個完整的 Block
>
> **定位：** [`03a`](03a-transformer-architecture.md) §6 的**超白話輔助版**（選讀）。這裡只給比喻與直覺、**不含任何公式**；看懂之後回到 `03a` §6 讀正式的「形狀契約」與數學，會輕鬆很多。前置直覺見 [`01a`](01a-prerequisites-intuition.md)。
>
> **對應 Notebook：** 無（純觀念）；Block 的實際前向見 [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13。
>
> **學完後的下一步：** → 回到 [`03a-transformer-architecture.md`](03a-transformer-architecture.md) §6 讀正式版

---

## 一、先用一句話理解

一個 **Transformer Block** 可以想成一輪完整的學習流程：

1. 先看看別人的資訊
2. 再整理成自己的想法
3. 同時保留原本的重要內容
4. 並讓整個過程保持穩定

所以，Transformer Block 不只是 Attention，還包含了其他幾個重要模組一起合作。

---

## 二、Transformer Block 的四個主要模組

| 模組 | 做什麼 | 為什麼需要 | 白話比喻 |
|---|---|---|---|
| Multi-Head Attention | 讓 token 和其他 token 交換資訊 | 建立上下文關係 | 開會時每個人聽別人說話，決定要參考誰的意見 |
| FFN | 對每個 token 自己再做加工 | 增加表達能力 | 開完會回到座位，自己整理筆記 |
| Residual Connection | 保留原本資訊，再加上新學到的內容 | 讓深層模型更容易訓練 | 改文章時保留原稿，只在旁邊註記修改 |
| LayerNorm | 穩定每一層數值的大小 | 避免訓練不穩 | 每回合開始前先把大家的音量調得差不多 |

---

## 三、四個模組的簡單理解

### 1. Multi-Head Attention：先互相看別人的資訊

在一句話裡，每個 token 不只看自己，還會看其他 token，判斷哪些資訊和自己有關。

例如在句子中，一個代名詞要知道自己指的是誰，就要參考前後文。

所以可以把 Multi-Head Attention 想成：

- 大家一起開會
- 每個人都聽別人說話
- 再判斷誰的意見對自己最重要

**重點：它負責資訊交換。**

---

### 2. FFN：再把資訊整理成自己的想法

只是聽別人講還不夠，還要把聽到的內容整理一下。

FFN（Feed Forward Network）就是做這件事。它會對每個 token 自己做進一步加工。

可以把它想成：

- 剛剛大家開完會
- 現在每個人回到座位
- 把聽到的內容整理成自己的筆記

**重點：它負責個別加工。**

---

### 3. Residual Connection：保留原本內容，再慢慢修改

Residual Connection 的意思是：

> 不要把原本資訊丟掉，而是保留它，再加上這一層學到的新內容。

簡單數學可以寫成：

```text
y = x + F(x)
```

- `x`：原本的資訊
- `F(x)`：這一層學到的新資訊
- `y`：最後輸出

可以把它想成改作文：

- 不是把原稿丟掉重寫
- 而是保留原稿
- 再在旁邊加上修改和補充

這樣做的好處是：

- 原本的重要資訊比較不容易消失
- 模型可以一層一層慢慢改進
- 深層模型比較容易訓練

**重點：它負責保留原資訊並加入修正。**

---

### 4. LayerNorm：先把狀態調穩

神經網路裡傳遞的其實是一堆數字。

如果某些數字太大、某些太小，模型學習時就可能不穩定。LayerNorm 的工作就是把這些數值調整到比較平衡的範圍。

可以把它想成：

- 一群人在討論
- 有人講話太大聲，有人太小聲
- 先把每個人的音量調整得差不多
- 討論就會比較順利

**重點：它負責穩定數值。**

---

## 四、為什麼 Multi-Head Attention 還不等於完整的 Transformer Block？

因為 Attention 只做了一件事：

> 讓不同 token 互相交換資訊。

但一個完整的 Transformer Block 還需要：

- **FFN**：把交換後的資訊再加工
- **Residual Connection**：保留原本資訊，不要被新資訊完全覆蓋
- **LayerNorm**：讓每一層的數值保持穩定

所以，Attention 很重要，但它只是整個流程的一部分。

---

## 五、一個完整 Transformer Block 的流程

可以簡化成下面這樣：

```text
輸入
→ LayerNorm
→ Multi-Head Attention
→ Residual Connection
→ LayerNorm
→ FFN
→ Residual Connection
→ 輸出
```

也可以用更白話的方式理解成：

```text
先調整狀態
→ 大家開會交換資訊
→ 保留原本想法並加入新資訊
→ 再調整狀態
→ 每個人自己整理內容
→ 再保留原本想法並加入整理後的結果
```

---

## 六、整體比喻：像一群同學合作學習

你可以把一個 Transformer Block 想成一群同學學習一篇文章：

1. **LayerNorm**：先把大家的狀態調整好，讓每個人聲音差不多
2. **Multi-Head Attention**：大家開始互相討論，聽別人的意見
3. **Residual Connection**：保留自己原本的想法，再加上剛剛學到的新內容
4. **LayerNorm**：再把狀態調整穩定
5. **FFN**：每個人回去自己整理筆記
6. **Residual Connection**：保留原本內容，再加上整理後的新理解

這樣一輪做完，就會得到更好的表示，接著再交給下一層。

---

## 六之補充、位置編碼在哪裡？（開會前先發座位號碼牌）

前面四個模組講的是「進到 Block 之後」發生的事。但還有一個動作發生在**進 Block 之前**：**位置編碼（Positional Encoding）**。

Attention 開會時，大家是「同時聽彼此」的，本身**分不出誰先講、誰後講**。所以在開會前，要先給每個人發一張**座位號碼牌**（位置向量），貼到他原本的名牌（token 向量）上：

- **名牌**：這個字是什麼意思（語意）
- **號碼牌**：這個字排在第幾個（位置）

兩張牌**黏在一起**（相加）交給大家，開會時就同時知道「你是誰」和「你坐第幾位」。

> **重點：位置編碼不是 Block 的第五個模組**，而是**進 Block 前的準備動作**——把順序資訊先貼到每個 token 上。想看它實際長什麼樣、數字怎麼加，見 [`03a`](03a-transformer-architecture.md) §7 與計算案例 [`03b3`](03b3-transformer-architecture-example.md) §6。

---

## 七、總結

一句話總結四個模組：

- **Multi-Head Attention**：互相聽
- **FFN**：自己整理
- **Residual Connection**：保留原稿再修改
- **LayerNorm**：先把狀態調穩

因此，Transformer Block 不是只有 Attention，而是一套完整流程：

> 先交換資訊，再個別整理，同時保留原本資訊，並讓整體訓練保持穩定。
