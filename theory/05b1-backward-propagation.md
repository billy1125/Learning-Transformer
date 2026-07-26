# 05b1｜向後傳播（Backward Pass）：梯度推導（符號）

> **適合對象：** 讀完 [`05a1-forward-propagation.md`](05a1-forward-propagation.md)（前向數學）後，想知道「模型到底怎麼學」的讀者。**數學只需要高中程度**：會微分（含 $\log$、$\exp$、商法則）、看得懂矩陣乘法就夠了。偏微分、鏈式法則、$G^X$ 記號都在 §1 從頭講起，不預設你學過大學的矩陣微分。
>
> **讀完後你能做什麼（依本文順序）：**
> - §1–§2　說出「梯度」是什麼、鏈式法則怎麼用，並畫出梯度從 loss 回到 Embedding 的完整路線圖
> - §3　推導 Cross-Entropy ＋ Softmax 的合併梯度，並說明為什麼 softmax 飽和會讓梯度消失
> - §4　寫出線性層的三條梯度通則（$G^W$、$G^X$、$G^b$），並套用到 lm_head
> - §5　推導 LayerNorm 的三條梯度路徑（直接／mean／variance），並解釋 Residual 為什麼能讓梯度不衰減地流過
> - §6　推導 FFN（兩層線性 ＋ ReLU）的梯度，說明 ReLU 反向為什麼只是一張遮罩
> - §7–§8　推導 Multi-Head 的 $W_O$ 與拆頭、以及 $\partial\mathcal{L}/\partial Q$、$\partial\mathcal{L}/\partial K$、$\partial\mathcal{L}/\partial V$ 的完整公式，並說明因果遮罩在反向為何「自動」歸零
> - §9–§10　推導 $W_Q,W_K,W_V$ 的梯度、以及 Embedding／位置編碼的梯度（含 Weight Tying 的稀疏／稠密雙通道）
> - §11　說出 optimizer 拿到梯度後做的最後一件事
>
> **前置文件：** [`05a1-forward-propagation.md`](05a1-forward-propagation.md)（前向數學）、[`04a-gpt-decoder-only.md`](04a-gpt-decoder-only.md)（基本概念與 Pipeline）、[`03a-transformer-architecture.md`](03a-transformer-architecture.md)
>
> **對應 Notebook：** [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb) — 本文每個公式都有對應的 Python 實作
>
> **數值演算（選讀續篇）：** → [`05b2-backward-example.md`](05b2-backward-example.md)（沿用 [`05a2`](05a2-forward-example.md) 的一組數字，把本文每條梯度實際算一次）
>
> **與前向的對應：** 本文的章節順序 **就是 [`05a1`](05a1-forward-propagation.md) 倒著走一遍**（05a1 §7 → §5 → §4 → §3 → §1／§2 → §6），完整對照表見 §2.2；整體資料流見 [`04a`](04a-gpt-decoder-only.md) 的「完整 Pipeline 總覽」。

---

## 目錄

1. 開始之前：反向傳播的最小工具箱
2. 全景圖：梯度回家的路線
3. 第一站：Cross-Entropy ＋ Softmax（前向見 05a1 §7）
4. 第二站：lm_head，順便把「線性層通則」講完（前向見 05a1 §7、§6.1）
5. 第三站：LayerNorm 與 Residual（前向見 05a1 §5）
6. 第四站：FFN（前向見 05a1 §4）
7. 第五站：Multi-Head 的 $W_O$ 與拆頭（前向見 05a1 §3）
8. 第六站：Self-Attention 核心與因果遮罩（前向見 05a1 §1、§2）
9. 第七站：$W_Q, W_K, W_V$ 三個投影矩陣（前向見 05a1 §1.1）
10. 終點站：Embedding 與位置編碼（前向見 05a1 §6）
11. 最後一步：optimizer 拿梯度做什麼
- 核心總結
- 附錄 A：完整梯度查閱表
- 附錄 B：為什麼 Transformer 比 RNN 好訓練

> **本文的定位：** [`05a1`](05a1-forward-propagation.md) 講「資料怎麼一路變成 loss」（前向）；本文講「loss 怎麼一路變回每個參數的修正量」（反向）。兩者是同一條路的來回。配套的 [`05b2`](05b2-backward-example.md) 用一組真實數字把本文每條公式算一次。建議 04a → 05a1 → 05a2 → 05b1 → 05b2 → 04b → NB4 依序讀。
>
> **數值範例在哪裡：** 本文只做符號推導，全文不出現具體數字。**前向**逐階段數字見 [`05a2-forward-example.md`](05a2-forward-example.md)、**反向**逐階段梯度見 [`05b2-backward-example.md`](05b2-backward-example.md)（沿用 05a2 的同一組數字）。

---

## 1. 開始之前：反向傳播的最小工具箱

訓練程式碼其實只有三行：

```python
loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
loss.backward()      # ← 本文從頭到尾就在解釋這一行做了什麼
optimizer.step()
```

`loss.backward()` 這一行背後，是幾十條梯度公式依序執行。這一節先把看懂那些公式所需的**全部**數學工具準備好——只有四樣，都是高中程度。

### 1.1 梯度是什麼：一句話定義

模型裡有幾百萬個可以調整的數字（參數）。訓練的問題是：**每個數字該往哪個方向調、調多少？**

梯度就是這個問題的答案。對某個參數 $w$：

$$
\frac{\partial \mathcal{L}}{\partial w}
\;=\;
\text{「把 } w \text{ 加一點點，loss 會跟著變多少」}
$$

更精確地說，把 $w$ 加上一個很小的量 $h$，loss 大約會變化 $\frac{\partial \mathcal{L}}{\partial w} \cdot h$。

**比喻：** 想像調音響的旋鈕。$\frac{\partial \mathcal{L}}{\partial w}$ 就是「這顆旋鈕往右轉一格，噪音會增加多少」。

- 這個值是**正的** → 往右轉會更吵 → 應該**往左轉**（減少 $w$）
- 這個值是**負的** → 往右轉反而變安靜 → 應該**往右轉**（增加 $w$）
- 這個值是**零** → 轉了沒差 → 這一步不動它

所以更新規則永遠是「往梯度的反方向走」：

$$
w \leftarrow w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

$\eta$ 是學習率（learning rate），控制「一次轉多少格」。這就是 §11 會再回來講的 optimizer。

### 1.2 偏微分：一次只轉一顆旋鈕

音響上有很多顆旋鈕，$\mathcal{L}$ 同時受它們全部影響。**偏微分**（符號 $\partial$，唸作「partial」）的意思很單純：

> 算 $\frac{\partial \mathcal{L}}{\partial w}$ 的時候，**把其他旋鈕全部當成不會動的常數**，只對 $w$ 這一顆做一般的微分。

例如 $f(x,y) = 3xy^2 + y$，那麼 $\frac{\partial f}{\partial x} = 3y^2$（把 $y$ 當常數，$y$ 那一項的微分是 0）。除了「其他變數當常數」這條規則以外，微分法則（乘法法則、商法則、鏈式法則）跟高中學的完全一樣。

### 1.3 鏈式法則：唯一真正要會的那條規則

反向傳播 99% 的內容就是重複使用鏈式法則。它處理的情況是：$x$ 影響 $y$，$y$ 影響 $\mathcal{L}$。

$$
x \;\longrightarrow\; y \;\longrightarrow\; \mathcal{L}
$$

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

**比喻（齒輪）：** 小齒輪 $x$ 帶動大齒輪 $y$，大齒輪帶動指針 $\mathcal{L}$。「$x$ 轉一圈指針動多少」＝「$x$ 轉一圈 $y$ 動多少」×「$y$ 轉一圈指針動多少」。兩段變速率相乘。

反向傳播就是把這個乘法**從右往左**一路串下去。實務上每一站都在做同一件事：

$$
\boxed{\text{這一層的梯度} \;=\; \text{上游傳下來的梯度} \;\times\; \text{這一層自己的變化率}}
$$

「上游」指的是靠近 loss 那一側。所以只要拿到上游梯度，剩下的工作就是算出「這一層自己的變化率」——本文 §3–§10 就是在一站一站算這件事。

### 1.4 分岔要相加：前向用幾次，反向就收幾份

第二條規則同樣重要，卻很容易漏掉。如果 $x$ **同時**影響 $y_1$ 和 $y_2$，兩條路最後都通到 $\mathcal{L}$：

$$
\frac{\partial \mathcal{L}}{\partial x}
= \frac{\partial \mathcal{L}}{\partial y_1}\cdot\frac{\partial y_1}{\partial x}
+ \frac{\partial \mathcal{L}}{\partial y_2}\cdot\frac{\partial y_2}{\partial x}
$$

**記成一句話：一個量在前向被用了幾次，反向就要收幾份梯度、加起來。**

這條規則在本文會反覆出現，每次都是同一個道理：

| 出現的地方 | 前向被用了幾次 | 反向收幾份 |
|---|---|---|
| 殘差 $x_1 = x_0 + C$（§5.10）| $x_0$ 走主幹、也走 attention 分支 | 2 份相加 |
| LayerNorm 的 $x_j$（§5.4）| 走自己、也透過 $\mu$ 和 $\sigma^2$ 影響所有維度 | 3 條路徑相加 |
| Attention 的輸入 $X$（§9.2）| 分別投影成 $Q$、$K$、$V$ | 3 份相加 |
| 同一個 token 出現多次（§10.2）| 出現幾次就查表幾次 | 出現幾次就加幾份 |

### 1.5 記號約定：$G^X$

完整寫 $\frac{\partial \mathcal{L}}{\partial X}$ 太長，全文一律簡寫：

$$
G^{X} \;\equiv\; \frac{\partial \mathcal{L}}{\partial X}
$$

三個要記住的性質：

1. **$G^X$ 的形狀和 $X$ 一模一樣。** $X$ 是 $T\times d$ 的矩陣，$G^X$ 也是 $T\times d$。這很合理：每個數字都要有自己的「該往哪調」。
2. **所有向量都當作列向量**（$1\times d$，橫的一排），矩陣乘法一律寫成「列向量乘矩陣」。這樣 $x_i W$ 的形狀才對得上 $X W$ 的第 $i$ 列。
3. **$\odot$ 是逐元素相乘**（Hadamard 乘法）：$[a_1,a_2]\odot[b_1,b_2] = [a_1b_1,\,a_2b_2]$。

### 1.6 保命技巧：形狀檢查

矩陣的梯度公式最常見的錯誤是「該轉置的地方沒轉置」。有一個幾乎萬無一失的檢查法：

> **算完一條梯度公式，先檢查它的形狀是不是和被微分的對象一樣。形狀對不上，公式一定錯。**

例如 $W_Q$ 是 $d\times d_k$，那麼 $G^{W_Q}$ 也必須是 $d\times d_k$。而 $X^\top$ 是 $d\times T$、$G^Q$ 是 $T\times d_k$，相乘得 $d\times d_k$ ✓ ——所以 $G^{W_Q}=X^\top G^Q$ 的轉置擺法是對的。

本文每條矩陣公式後面都會附一次這種檢查，標記為 $\checkmark$。**這不是裝飾，是你自己推導時的除錯工具。**

> **讀完這一節，你會：**
> - 用一句話說出 $\partial \mathcal{L}/\partial w$ 的意義，並解釋為什麼更新要「減去」梯度
> - 用鏈式法則把「上游梯度 × 本層變化率」串起來
> - 看到一個量被用了兩次時，知道反向要把兩份梯度相加
> - 用形狀檢查抓出自己推導時的轉置錯誤

---

## 2. 全景圖：梯度回家的路線

工具備齊了，先看地圖再上路。**前向是從 token 走到 loss，反向就是原路走回來**——經過的站完全相同，只是方向相反。

### 2.1 反向五步地圖

**Step 1｜Cross-Entropy ＋ Softmax（詳見 §3）**

$$
\delta_i \equiv G^{z_i} = \frac{\partial \mathcal{L}}{\partial z_i}, \qquad
\delta_i^{(k)} = \frac{1}{T}\bigl(p^{(i)}_k - \mathbb{1}[k = y_i]\bigr)
$$

白話：在正確答案那個位置減 $1/T$，其餘位置就是 softmax 機率除以 $T$。

**Step 2｜lm_head 反向（$z_i = \hat h_i W_{lm}^\top$，詳見 §4）**

$$
G^{\hat h_i} = \delta_i \, W_{lm}
\qquad (\text{梯度往上一層走，}d\text{ 維})
$$

$$
G^{W_{lm}} = \sum_i \delta_i^\top \, \hat h_i
\qquad (\text{lm\_head 的參數梯度，稠密 }V \times d\text{；}1/T\text{ 已含在 }\delta_i\text{ 中})
$$

**Step 3｜穿越 LayerNorm、Residual、FFN、Attention（詳見 §5–§9）**

梯度沿著 Block 裡的每個模組反向傳回。Pre-LN 的殘差直通見 §5.10 與 [`05a1`](05a1-forward-propagation.md) §5.3。最終到達 $x_{\text{embed}}$ 的梯度記為 $g_i \in \mathbb{R}^d$。

**Step 4｜Lookup 反向（$x_i = E[t_i]$，詳見 §10）**

$$
G^{E[k]} = \sum_{i:\, t_i = k} g_i
\qquad (E\text{ 的梯度，稀疏：只有出現過的列非零})
$$

**Step 5｜Optimizer 更新（詳見 §11）**

$$
E[k] \leftarrow E[k] - \eta \cdot G^{E[k]}
$$

> **註（Weight Tying）：** Karpathy 的原版 nanoGPT 讓 `lm_head.weight` 與 `token_embedding.weight` 共用同一份矩陣（$W_{lm} = E$）。此時 Step 2 的稠密梯度與 Step 4 的稀疏梯度會**累加**到同一個 $E$ 上——這正是 §1.4「用幾次收幾份」的又一次應用。完整的雙通道梯度推導見 §10.3。（本倉庫 NB4 未做 Weight Tying，見 [`04b`](04b-nanogpt-walkthrough.md) §5。）

**三個關鍵特性：**

1. **稀疏更新**：Lookup 反向（Step 4）只更新本 batch 出現過的 token 列，沒出現的 token 其 embedding 這一步完全不動。
2. **同 token 累加**：token $k$ 在同一序列出現 $m$ 次，Step 4 的梯度就是 $m$ 個 $g_i$ 的加總。
3. **直覺含義**：出現頻繁的 token 每步都被更新，embedding 收斂快；稀有 token 需要大量訓練步驟才被充分觸及。

### 2.2 反向 ↔ 前向的節號鏡射表

本文的章節順序**就是把 [`05a1`](05a1-forward-propagation.md) 倒著讀**。對照如下——這張表也是本文「不亂跳」的依據：

| 本文（反向，從 loss 往回走）| [`05a1`](05a1-forward-propagation.md)（前向，從 token 往前走）| 這一站在算什麼 |
|---|---|---|
| §3 Cross-Entropy ＋ Softmax | §7 Next-token Prediction 與 CE | $\delta_i = G^{z_i}$ |
| §4 lm_head（＋線性層通則）| §7.2 損失函數的輸入端 | $G^{W_{lm}}$、$G^{\hat h}$ |
| §5 LayerNorm ＋ Residual | §5 LayerNorm 與 Pre-LN Block | $G^{\gamma}, G^{\beta}, G^{x}$ |
| §6 FFN | §4 Position-wise FFN | $G^{W_1}, G^{W_2}$ |
| §7 Multi-Head 的 $W_O$ 與拆頭 | §3 Multi-Head Attention | $G^{W_O}, G^{C^{(h)}}$ |
| §8 Self-Attention 核心 ＋ 因果遮罩 | §1 Scaled Dot-Product ＋ §2 Causal Mask | $G^V, G^A, G^{\tilde S}, G^Q, G^K$ |
| §9 QKV 三個投影矩陣 | §1.1 從輸入到 Q、K、V | $G^{W_Q}, G^{W_K}, G^{W_V}, G^X$ |
| §10 Embedding ＋ 位置編碼 | §6 Token Embedding 與位置編碼 | $G^{E}, G^{P}$ |

> **一個記號上的調整：** 舊版 05b1 用 $E$ 同時表示「注意力分數矩陣」和「Embedding 矩陣」，容易混淆。本文改成與 [`05a1`](05a1-forward-propagation.md) 一致：$S = QK^\top$ 是原始分數、$\tilde S = S/\sqrt{d_k}$ 是縮放後進 softmax 的分數，$E$ 專指 Embedding 矩陣。[`05b2`](05b2-backward-example.md) 把 $G^{\tilde S}$ 簡寫成 $G^{S}$，是同一個東西。

### 2.3 前向／反向 ↔ nanoGPT 元件對照

前向數學（[`05a1`](05a1-forward-propagation.md)）與反向（本文），都對應 [`04b`](04b-nanogpt-walkthrough.md) 裡的一段程式：

| 前向（[`05a1`](05a1-forward-propagation.md)）| 反向（本文）| nanoGPT 程式（[`04b`](04b-nanogpt-walkthrough.md)）|
|---|---|---|
| §1 Scaled Dot-Product ＋ §2 Causal Mask | §8、§9 | §1 `Head` |
| §3 Multi-Head ＋ $W_O$ | §7 | §2 `MultiHeadAttention` |
| §4 FFN | §6 | §3 `FeedForward` |
| §5 LayerNorm／Pre-LN Block | §5 | §4 `Block`、§7 Pre-LN vs Post-LN |
| §6 Embedding／Learned PE | §10 | §5 `GPT`（`token_embedding`／`position_embedding`）|
| §7 Cross-Entropy | §3、§4 | §5 `lm_head`、§6 對照總表 |

> **讀完這一節，你會：**
> - 背出反向五步：CE+Softmax → lm_head → Block 內部 → Lookup → optimizer
> - 說出本文每一章對應 05a1 的哪一節（§2.2 的鏡射表）
> - 解釋為什麼 Embedding 的梯度是稀疏的

---

## 3. 第一站：Cross-Entropy ＋ Softmax

**前向見 [`05a1`](05a1-forward-propagation.md) §7。** 梯度的起點在 loss，所以反向的第一站，就是前向的最後一站。

前向的最後兩步是：模型在位置 $i$ 吐出一個 $V$ 維的 logit 向量 $z_i$（$V$ 是詞彙表大小），先用 softmax 換成機率，再和正確答案比對算出 loss：

$$
p^{(i)}_k = \frac{\exp(z_i^{(k)})}{\sum_{j=1}^{V} \exp(z_i^{(j)})}, \qquad
\mathcal{L} = -\frac{1}{T}\sum_{i=1}^{T} \log p^{(i)}_{y_i}
$$

要求的是 $G^{z_i} = \frac{\partial \mathcal{L}}{\partial z_i}$。路徑是 $z_i \to p^{(i)} \to \mathcal{L}$，所以要串兩段鏈式法則。先各算一段。

### 3.1 第一段：loss 對機率的微分

先只看單一位置 $i$ 的損失 $\ell_i = -\log p^{(i)}_{y_i}$（暫時把 $\frac1T$ 拿掉，最後再乘回來）。這是高中的 $\log$ 微分：

$$
\frac{\partial \ell_i}{\partial p^{(i)}_{y_i}} = -\frac{1}{p^{(i)}_{y_i}}
$$

**白話：** 正確答案的機率愈小，這個值的絕對值愈大——模型錯得愈離譜，訊號就愈強。

### 3.2 第二段：Softmax 的 Jacobian

現在要算 $\frac{\partial p^{(i)}_{j}}{\partial z_i^{(l)}}$：「第 $l$ 個 logit 動一點點，第 $j$ 個機率會變多少」。$V$ 個輸出對 $V$ 個輸入，總共 $V\times V$ 個偏導數，這張表就叫 **Jacobian（雅可比矩陣）**。

為了讓式子乾淨，本小節固定位置 $i$，記 $a_j = p^{(i)}_j$、$e_j = z_i^{(j)}$：

$$
a_j = \frac{\exp(e_j)}{S}, \qquad S = \sum_{k=1}^{V} \exp(e_k)
$$

關鍵觀察：**$e_l$ 一定出現在分母 $S$ 裡，但只有 $l=j$ 時才出現在分子。** 所以要分兩種情況。

**情況一：$l = j$（分子分母都含 $e_j$）**

用高中的商法則 $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$，其中 $u = \exp(e_j)$、$v = S$，而 $\frac{\partial}{\partial e_j}\exp(e_j) = \exp(e_j)$、$\frac{\partial S}{\partial e_j} = \exp(e_j)$：

$$
\frac{\partial a_j}{\partial e_j}
= \frac{\exp(e_j) \cdot S - \exp(e_j) \cdot \exp(e_j)}{S^2}
= \frac{\exp(e_j)}{S} \cdot \frac{S - \exp(e_j)}{S}
= a_j (1 - a_j)
$$

（最後一步把 $S^2$ 拆成兩個 $S$ 分給前後兩項，再各自認出 $\exp(e_j)/S = a_j$。）

**情況二：$l \neq j$（只有分母含 $e_l$）**

分子 $\exp(e_j)$ 和 $e_l$ 無關，可以當常數提出來，只需對 $\frac{1}{S}$ 微分（$\frac{\partial}{\partial S}\frac1S = -\frac{1}{S^2}$，再乘上 $\frac{\partial S}{\partial e_l} = \exp(e_l)$）：

$$
\frac{\partial a_j}{\partial e_l}
= \exp(e_j) \cdot \left(-\frac{1}{S^2}\right) \cdot \exp(e_l)
= -\frac{\exp(e_j)}{S} \cdot \frac{\exp(e_l)}{S}
= -a_j \, a_l
$$

**合併兩種情況。** 引入 Kronecker delta $\delta_{jl}$（$j = l$ 時為 1、否則為 0），兩式可以寫成同一條：

$$
\boxed{\frac{\partial a_j}{\partial e_l} = a_j(\delta_{jl} - a_l)}
$$

驗證：$l = j$ 時得 $a_j(1 - a_j)$ ✓；$l \neq j$ 時 $\delta_{jl}=0$，得 $-a_j a_l$ ✓。

**白話解讀：** 把 $e_l$ 調高一點，$a_l$ 自己會變大（$a_l(1-a_l) > 0$），而**所有其他** $a_j$ 都會被壓小（$-a_j a_l < 0$）。因為 softmax 的輸出總和恆為 1，是一場零和遊戲——有人多拿，就有人少拿。

> **附帶收穫——飽和為什麼導致梯度消失：** 對角項 $a_j(1 - a_j)$ 在 $a_j \to 0$ 或 $a_j \to 1$ 時都趨近 0，非對角項 $-a_j a_l$ 也是。也就是說，softmax 一旦輸出接近 one-hot（飽和），**整個 Jacobian 趨近零矩陣**，任何上游梯度乘上它都會消失。這正是 [`03a`](03a-transformer-architecture.md) §3.4 說「不除以 $\sqrt{d_k}$ 會讓訓練停滯」的數學原因，也是 §8.5 會看到「因果遮罩那一列拿不到梯度」的原因。

### 3.3 串起來：CE ＋ Softmax 的合併梯度

現在把兩段接上。$z_i^{(k)}$ 影響 $\mathcal{L}$ 的路徑只有一條——透過 $p^{(i)}_{y_i}$（因為 $\ell_i$ 只用到正確答案那一格的機率）。所以取 §3.2 的結果並固定 $j = y_i$、$l = k$：

$$
\frac{\partial p^{(i)}_{y_i}}{\partial z_i^{(k)}} = p^{(i)}_{y_i}\left(\mathbb{1}[k=y_i] - p^{(i)}_k\right)
$$

兩式相乘（鏈式法則），$p^{(i)}_{y_i}$ 恰好上下消掉：

$$
\frac{\partial \ell_i}{\partial z_i^{(k)}}
= -\frac{1}{p^{(i)}_{y_i}} \cdot p^{(i)}_{y_i}\left(\mathbb{1}[k=y_i] - p^{(i)}_k\right)
= p^{(i)}_k - \mathbb{1}[k=y_i]
$$

因為 $\mathcal{L} = \frac{1}{T}\sum_i \ell_i$，最後把 $\frac1T$ 乘回來：

$$
\boxed{\;\delta_i^{(k)} \;\equiv\; \frac{\partial \mathcal{L}}{\partial z_i^{(k)}} = \frac{1}{T}\left(p^{(i)}_k - \mathbb{1}[k = y_i]\right)\;}
$$

**這是整份文件最漂亮的結果之一：** softmax 那一堆指數和商法則全部消失了，只剩「預測機率減掉正確答案的 one-hot」。深度學習框架把 softmax 和 cross-entropy 合成一個 op（PyTorch 的 `F.cross_entropy`）就是為了直接用這條式子，既快又不會有數值問題。

### 3.4 怎麼解讀 $\delta_i$

記 $\delta_i = G^{z_i} \in \mathbb{R}^V$，逐格看：

| 位置 | 值 | 正負 | 更新的效果 |
|---|---|---|---|
| 正確答案 $k = y_i$ | $\dfrac{p^{(i)}_{y_i}-1}{T}$ | 負（因為 $p \le 1$）| 把這個 logit **推高** |
| 其他 $k \neq y_i$ | $\dfrac{p^{(i)}_k}{T}$ | 正 | 把這些 logit **壓低** |

還有一個好用的檢查：**每一列的和恰好為零**，因為 $\sum_k p^{(i)}_k = 1$ 且 one-hot 也只加起來 1。也就是說梯度下降在做的事，是把機率質量從錯的類別**搬**到對的類別，總量不變。

> **讀完這一節，你會：**
> - 用商法則自己推出 softmax 的 Jacobian $a_j(\delta_{jl}-a_l)$
> - 解釋為什麼 softmax 飽和（輸出接近 one-hot）會讓梯度整個消失
> - 寫出 CE＋Softmax 的合併梯度 $\delta_i^{(k)} = \frac1T(p_k - \mathbb{1}[k=y_i])$，並說出它每一格的正負意義
> - 用「每列和為零」檢查自己算的 $\delta_i$

---

## 4. 第二站：lm_head，順便把「線性層通則」講完

**前向見 [`05a1`](05a1-forward-propagation.md) §7.2 與 §6.1。** 拿到 $\delta_i$ 之後，梯度要繼續往回走一層：logit 是怎麼算出來的？答案是一個線性層（矩陣乘法）。

線性層是整個 Transformer 裡出現最多次的元件——lm_head、$W_Q/W_K/W_V$、$W_O$、FFN 的兩層，全都是。所以這一節把它的梯度**一次講完**，後面 §6、§7、§9 就可以直接引用。

### 4.1 線性層的三條通則

設前向是（記得 §1.5 的列向量約定）：

$$
Y = X W + b, \qquad X \in \mathbb{R}^{T\times n},\; W \in \mathbb{R}^{n\times m},\; Y \in \mathbb{R}^{T\times m}
$$

給定上游梯度 $G^Y$（形狀 $T\times m$），三條結果是：

$$
\boxed{G^{W} = X^\top G^{Y}}, \qquad
\boxed{G^{X} = G^{Y} W^\top}, \qquad
\boxed{G^{b} = \sum_{i=1}^{T} G^{Y}_{i,:}}
$$

**背法：** 對權重的梯度是「輸入的轉置 乘 輸出的梯度」；對輸入的梯度是「輸出的梯度 乘 權重的轉置」。轉置擺哪邊不用死背——用 §1.6 的形狀檢查現場推：

$$
G^W:\; (n \times T)\cdot(T \times m) = (n\times m) \;\checkmark \qquad
G^X:\; (T \times m)\cdot(m \times n) = (T\times n) \;\checkmark
$$

### 4.2 為什麼是這樣（用索引推一次）

不想只是背公式的話，展開一格看就懂了。前向逐格寫是：

$$
Y_{ij} = \sum_{k=1}^{n} X_{ik} W_{kj} + b_j
$$

**對 $W_{kj}$：** 哪些 $Y_{ij}$ 用到了 $W_{kj}$？固定 $j$ 這一欄，所有 $i$ 都用到了（每一列都做同一個線性變換）。所以要對 $i$ 全部加起來（§1.4 的「用幾次收幾份」）：

$$
G^{W}_{kj} = \sum_{i=1}^{T} G^{Y}_{ij} \cdot \frac{\partial Y_{ij}}{\partial W_{kj}} = \sum_{i=1}^{T} G^{Y}_{ij} X_{ik}
$$

右邊正是「$X^\top$ 的第 $k$ 列」點「$G^Y$ 的第 $j$ 欄」，也就是 $(X^\top G^Y)_{kj}$。✓

**對 $X_{ik}$：** 第 $i$ 列的輸入只影響第 $i$ 列的輸出，但影響那一整列的 $m$ 個格子：

$$
G^{X}_{ik} = \sum_{j=1}^{m} G^{Y}_{ij} \cdot \frac{\partial Y_{ij}}{\partial X_{ik}} = \sum_{j=1}^{m} G^{Y}_{ij} W_{kj} = (G^Y W^\top)_{ik}
$$

✓ **對 $b_j$：** $\frac{\partial Y_{ij}}{\partial b_j} = 1$，所以就是把所有列的梯度加起來。

### 4.3 套用到 lm_head

nanoGPT 的 lm_head 把最後一層 LayerNorm 的輸出 $\hat h_i \in \mathbb{R}^d$ 變成 $V$ 維 logit（PyTorch 的 `nn.Linear(d, V, bias=False)` 內部存的是 $V\times d$ 的矩陣，所以式子裡帶一個轉置）：

$$
z_i = \hat h_i W_{lm}^\top, \qquad W_{lm} \in \mathbb{R}^{V\times d}
$$

上游梯度就是 §3.3 算出的 $\delta_i$。直接套 §4.1：

$$
\boxed{G^{\hat h_i} = \delta_i \, W_{lm} \;\in \mathbb{R}^{d}}
\qquad
\boxed{G^{W_{lm}} = \sum_{i=1}^{T} \delta_i^\top \, \hat h_i \;\in \mathbb{R}^{V\times d}}
$$

形狀檢查：$\delta_i$ 是 $1\times V$、$W_{lm}$ 是 $V\times d$，相乘得 $1\times d$ ✓。第二條是**外積**：$\delta_i^\top$ 是 $V\times 1$、$\hat h_i$ 是 $1\times d$，相乘得 $V\times d$ ✓。

取第 $k$ 列來看更直觀：

$$
G^{W_{lm}[k]} = \sum_{i=1}^{T} \delta_i^{(k)} \cdot \hat h_i
$$

**白話：** 詞彙表裡**每一個** token 的那一列都會拿到梯度（因為 $\delta_i^{(k)}$ 對所有 $k$ 都非零），所以 lm_head 的梯度是**稠密**的。這一點和 §10 的 Embedding 剛好相反，是後面比較兩者的關鍵。

> **讀完這一節，你會：**
> - 默寫線性層的三條梯度通則，並用形狀檢查驗證轉置擺對了
> - 用索引展開解釋 $G^W = X^\top G^Y$ 為什麼要對所有位置求和
> - 算出 lm_head 的 $G^{W_{lm}}$ 與往上傳的 $G^{\hat h_i}$
> - 說出 lm_head 的梯度為什麼是稠密的

---

## 5. 第三站：LayerNorm 與 Residual

**前向見 [`05a1`](05a1-forward-propagation.md) §5。** 梯度離開 lm_head，下一個撞上的就是最後那層 LayerNorm。之後每穿過一個 Block，還會再遇到兩次。

LayerNorm 是本文最難的一節，原因只有一個：**它不是逐元素運算。** 前面的線性層裡，每個輸出只跟自己那條路有關；但 LayerNorm 要先算整個向量的平均 $\mu$ 和變異數 $\sigma^2$，**每個輸入都參與了這兩個統計量**，所以每個 $x_j$ 都會透過三條不同的路影響 loss。§1.4 的「分岔要相加」在這裡要用滿。

LayerNorm 對每個 token 的 hidden vector **獨立**歸一化。設某一 token 的輸入 $x \in \mathbb{R}^d$：

$$
\mu = \frac{1}{d}\sum_{j=1}^d x_j, \qquad
\sigma^2 = \frac{1}{d}\sum_{j=1}^d (x_j - \mu)^2
$$

$$
\hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad
y_j = \gamma_j \hat{x}_j + \beta_j
$$

其中 $\gamma, \beta \in \mathbb{R}^d$ 是可學習的 scale／shift 參數，$\epsilon > 0$ 防止除以零。

### 5.1 記號定義

$$
r = \sqrt{\sigma^2 + \epsilon}, \qquad
g^y_j = \frac{\partial \mathcal{L}}{\partial y_j} \quad \text{（上游梯度）}
$$

目標：推導 $G^{\gamma_j}$、$G^{\beta_j}$、$G^{x_j}$。**先做簡單的兩個（§5.2、§5.3），再花力氣處理 $x$（§5.4–§5.7）。**

### 5.2 對 $\gamma$ 與 $\beta$ 的梯度

這兩個最容易——$y_j = \gamma_j \hat{x}_j + \beta_j$ 就是一條一次函數，直接微分：

$$
\frac{\partial \mathcal{L}}{\partial \gamma_j} = g^y_j \hat{x}_j, \qquad
\frac{\partial \mathcal{L}}{\partial \beta_j} = g^y_j
$$

但 $\gamma, \beta$ 是**所有 token 共用**的參數（一個 $d$ 維向量，套用到每個位置），所以要把 batch 裡所有位置的貢獻加起來——又一次「用幾次收幾份」：

$$
\boxed{G^{\gamma_j} = \sum_{b,t} g^y_{b,t,j}\, \hat{x}_{b,t,j}, \qquad
G^{\beta_j} = \sum_{b,t} g^y_{b,t,j}}
$$

### 5.3 先退回歸一化後的向量

繼續往回走一步，先算梯度傳到 $\hat x$（還沒穿過歸一化）：

$$
g^{\hat{x}}_j \equiv \frac{\partial \mathcal{L}}{\partial \hat{x}_j} = g^y_j \gamma_j
$$

接下來的 §5.4–§5.7 就是全文最硬的一段：把 $g^{\hat x}$ 變成 $G^{x}$，也就是穿過 $\mu$ 和 $\sigma^2$。

### 5.4 三條梯度路徑

$x_j$ 影響 loss 的路有三條——第一條顯而易見，另外兩條是 LayerNorm 特有的：

**路徑 1（直接路徑）：** $x_j \to \hat{x}_j \to y_j \to \mathcal{L}$

暫時把 $\mu$ 和 $r$ 當成固定的常數：

$$
\hat{x}_j = \frac{x_j - \mu}{r} \quad \Rightarrow \quad
\frac{\partial \hat{x}_j}{\partial x_j}\Big|_{\mu, r \text{ 固定}} = \frac{1}{r}
$$

**路徑 2（透過 $\mu$）：** $x_j \to \mu \to \hat{x}_k \to \mathcal{L}$，**影響所有的 $k$**

$$
\mu = \frac{1}{d}\sum_{l=1}^d x_l \quad \Rightarrow \quad
\frac{\partial \mu}{\partial x_j} = \frac{1}{d}
$$

**路徑 3（透過 $\sigma^2$）：** $x_j \to \sigma^2 \to r \to \hat{x}_k \to \mathcal{L}$，**也影響所有的 $k$**

$$
\sigma^2 = \frac{1}{d}\sum_{l=1}^d \tilde{x}_l^2, \quad \tilde{x}_l = x_l - \mu
\quad \Rightarrow \quad
\frac{\partial \sigma^2}{\partial x_j} = \frac{2\tilde{x}_j}{d}
$$

**三條路徑示意圖：**

```
                         ┌─ 路徑1（直接）─────────────────── ŷ_j → L
                         │                                        ↑
x_j ─→ x̃_j=x_j-μ ──→ ÷r ──→ ŷ_j          g^ŷ_j（直接梯度 g^x̂/r）
  │                      ↑
  │                 r=√(σ²+ε)              路徑3（variance）
  │                      ↑               g^σ² × (2x̃_j/d) 傳回
  │               σ²=Σx̃²/d
  │                      ↑
  │         所有 j 的 (x_j-μ)² 累積
  │
  └──→ (1/d)→ μ ── 影響所有 x̃_k=x_k-μ    路徑2（mean）
                                           g^μ × (-1/d) 傳給每個 j
```

三條路徑必須**同時計算再相加**，因為 $\mu$ 與 $\sigma^2$ 由同一個 hidden vector 的所有維度共同決定，無法逐元素獨立處理。

**比喻：** 想像班上考完試後按「離平均幾個標準差」重新換算分數。你自己多考 1 分，不只你的換算分數會變（路徑 1），全班平均也被你拉高一點（路徑 2），全班的分散程度也跟著變（路徑 3）——結果是**每一位同學**的換算分數都被你影響了。反向傳播必須把這三種影響全部算進去。

### 5.5 Variance 路徑的詳細推導

三條路徑裡只有路徑 3 需要多一點計算。先對 $r^{-1}$ 微分（把 $\sigma^2+\epsilon$ 當成一個變數，用冪次微分 $\frac{d}{du}u^{-1/2} = -\frac12 u^{-3/2}$）：

$$
r^{-1} = (\sigma^2 + \epsilon)^{-1/2} \quad \Rightarrow \quad
\frac{\partial r^{-1}}{\partial \sigma^2} = -\frac{1}{2}(\sigma^2 + \epsilon)^{-3/2} = -\frac{1}{2}r^{-3}
$$

因為 $\hat{x}_j = \tilde{x}_j \, r^{-1}$（$\tilde x_j$ 在這裡當常數），所以：

$$
\frac{\partial \hat{x}_j}{\partial \sigma^2} = \tilde{x}_j \cdot \left(-\frac{1}{2}r^{-3}\right) = -\frac{\tilde{x}_j}{2r^3}
$$

loss 透過 variance 這條路傳回 $\sigma^2$ 時，要把**所有維度**的貢獻加起來（因為 $\sigma^2$ 影響了每一個 $\hat x_j$）：

$$
g^{\sigma^2} = \frac{\partial \mathcal{L}}{\partial \sigma^2}
= \sum_{j=1}^d g^{\hat{x}}_j \cdot \left(-\frac{\tilde{x}_j}{2r^3}\right)
= -\frac{1}{2r^3} \sum_{j=1}^d g^{\hat{x}}_j \tilde{x}_j
$$

利用 $\tilde{x}_j = r\hat{x}_j$ 把式子換成 $\hat x$ 的形式（消掉一個 $r$），比較好看：

$$
g^{\sigma^2} = -\frac{1}{2r^2} \sum_{j=1}^d g^{\hat{x}}_j \hat{x}_j
$$

### 5.6 合併三條路徑

分兩步走比較不容易亂：**先合併路徑 1 和路徑 3**（都是對 $\tilde x_j = x_j-\mu$ 的影響），**最後再補上路徑 2**。

第一步，令 $\tilde{x}_j = x_j - \mu$，先算對 $\tilde x_j$ 的梯度（此時暫時忽略 $\mu$ 對 $x_j$ 的依賴）：

$$
g^{\tilde{x}}_j = \underbrace{g^{\hat{x}}_j \, r^{-1}}_{\text{路徑 1：直接}}
+ \underbrace{g^{\sigma^2} \cdot \frac{2\tilde{x}_j}{d}}_{\text{路徑 3：variance}}
$$

代入 §5.5 的 $g^{\sigma^2}$ 與 $\tilde{x}_j = r\hat{x}_j$：

$$
g^{\tilde{x}}_j
= \frac{g^{\hat{x}}_j}{r}
- \frac{1}{d} \cdot \frac{\hat{x}_j}{r} \sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k
= \frac{1}{r}\left(g^{\hat{x}}_j - \frac{\hat{x}_j}{d}\sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k\right)
$$

第二步，補上 mean 路徑。$x_j$ 透過 $\mu$ 影響**所有**的 $\tilde{x}_k = x_k - \mu$，而 $\frac{\partial \tilde x_k}{\partial \mu} = -1$、$\frac{\partial \mu}{\partial x_j} = \frac1d$，所以：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
= g^{\tilde{x}}_j - \frac{1}{d}\sum_{k=1}^d g^{\tilde{x}}_k
$$

### 5.7 最終閉式公式

把 §5.6 的兩條式子合成一條。先準備一個關鍵的小引理：

**引理：歸一化後的向量均值為零，即 $\sum_{k=1}^d \hat{x}_k = 0$。**

證明：$\hat{x}_k = (x_k - \mu)/r$，而 $\sum_k (x_k - \mu) = \sum_k x_k - d\mu = d\mu - d\mu = 0$，除以常數 $r$ 後仍為 0。∎

接著計算 mean 路徑需要的 $\frac{1}{d}\sum_k g^{\tilde{x}}_k$，把 §5.6 的 $g^{\tilde{x}}_k$ 代進去：

$$
\frac{1}{d}\sum_{k=1}^d g^{\tilde{x}}_k
= \frac{1}{d}\sum_{k=1}^d \frac{1}{r}\left(g^{\hat{x}}_k - \frac{\hat{x}_k}{d}\sum_{m=1}^d g^{\hat{x}}_m \hat{x}_m\right)
= \frac{1}{r}\left(\frac{1}{d}\sum_k g^{\hat{x}}_k - \underbrace{\frac{1}{d}\sum_k \hat{x}_k}_{=\,0\text{（引理）}} \cdot \frac{1}{d}\sum_m g^{\hat{x}}_m \hat{x}_m\right)
= \frac{1}{r} \cdot \frac{1}{d}\sum_k g^{\hat{x}}_k
$$

第二項因為引理整個消失——這就是最終公式比想像中乾淨的原因。代回 $\frac{\partial \mathcal{L}}{\partial x_j} = g^{\tilde{x}}_j - \frac{1}{d}\sum_k g^{\tilde{x}}_k$：

$$
\frac{\partial \mathcal{L}}{\partial x_j}
= \frac{1}{r}\left(g^{\hat{x}}_j - \frac{\hat{x}_j}{d}\sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k\right) - \frac{1}{r}\cdot\frac{1}{d}\sum_{k=1}^d g^{\hat{x}}_k
$$

把 $\frac1r$ 提出來、三項排好，最終得到：

$$
\boxed{
G^{x_j}
= \frac{1}{r}\left(
g^{\hat{x}}_j
- \frac{1}{d}\sum_{k=1}^d g^{\hat{x}}_k
- \hat{x}_j \cdot \frac{1}{d}\sum_{k=1}^d g^{\hat{x}}_k \hat{x}_k
\right)
}
$$

三項各自的來歷與意義：

| 項 | 來源 | 意義 | 白話 |
|---|---|---|---|
| $g^{\hat{x}}_j$ | 路徑 1（直接）| 自身的梯度貢獻 | 你自己該怎麼調 |
| $-\dfrac{1}{d}\sum_k g^{\hat{x}}_k$ | 路徑 2（mean）| 平均值耦合的修正 | 扣掉「全班一起調」的部分 |
| $-\hat{x}_j \cdot \dfrac{1}{d}\sum_k g^{\hat{x}}_k \hat{x}_k$ | 路徑 3（variance）| 方差耦合的修正 | 扣掉「把分佈整個拉寬／縮窄」的部分 |

**為什麼要扣掉那兩項？** 因為 LayerNorm 的輸出對「全體平移」和「全體縮放」完全無感——把 $x$ 每個維度都加 5，$\hat x$ 一模一樣。既然這兩種調整不會改變輸出，梯度就不該指往那兩個方向，反向公式自動把它們扣掉了。

### 5.8 向量形式

令 $g^{\hat{x}} = G^{\hat x} \in \mathbb{R}^d$，寫成向量比較好對照程式碼：

$$
G^{x}
= \frac{1}{r}\left(
g^{\hat{x}}
- \text{mean}(g^{\hat{x}})
- \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})
\right)
$$

其中 $\text{mean}(\cdot) = \frac{1}{d}\sum_j (\cdot)_j$（沿特徵維取平均，得到一個純量再廣播），$\odot$ 為逐元素乘法。

### 5.9 關鍵結論：為什麼梯度是耦合的

LayerNorm 的 $\mu$ 與 $\sigma^2$ 都由同一個 hidden vector 的**所有維度共同決定**。因此：

- 每個輸入維度 $x_j$ 的梯度不只來自自身的輸出 $y_j$
- 也受到所有其他維度 $\{x_k\}_{k \neq j}$ 的影響（透過 $\mu$ 和 $\sigma^2$）

這使得 LayerNorm 的梯度不是 element-wise 操作，而是 hidden dimension 內部的**耦合運算**，必須整體計算（如 §5.7 的公式），不能逐元素獨立處理。**實務後果：** 梯度每穿過一次 LayerNorm，就會被扣掉「沿 $\mathbf{1}$」與「沿 $\hat x$」兩個方向的分量，量級會被削一次。這是深層網路裡梯度逐漸變小的原因之一。

### 5.10 在 Transformer Block 中的梯度流：Residual

**前向見 [`05a1`](05a1-forward-propagation.md) §5.2、§5.3。** Pre-LN Block 的通式是 $x \leftarrow x + f(\text{LN}(x))$。反向時，這個加號有奇效。

典型結構：

$$
U = X + Z, \qquad Z' = \text{LayerNorm}(U)
$$

**Step 1：** 從 $G^{Z'}$ 經 LayerNorm 反傳得 $G^U$（套用 §5.7 的公式）。

**Step 2：** 由 Residual Connection $U = X + Z$，因為加法對兩個輸入的偏導都是 1：

$$
G^{X}\Big|_{\text{res}} = G^U, \qquad
G^{Z} = G^U
$$

**這正是 §1.4「分岔要相加」的鏡像：前向是兩份相加成一份，反向就是一份原封不動複製成兩份。** 其中往 $X$ 那一份**完全沒有經過任何非線性運算、也沒有被任何 Jacobian 縮放**——這條路叫「恆等直通」。

**為什麼這件事這麼重要？** 疊 24 層時，梯度從最上層回到最下層要穿過 24 次 LayerNorm、24 次 attention。如果每一次都被乘上一個小於 1 的因子，走完 24 層就幾乎歸零了。有了殘差，梯度**永遠有一條不縮放的捷徑**可以直達底層：

$$
G^{X} = \underbrace{G^{U}}_{\text{恆等直通，不衰減}} + \underbrace{G^{U}\frac{\partial f(\text{LN}(X))}{\partial X}}_{\text{經過模組，可能被削}}
$$

這就是 Transformer 能訓練幾十層甚至上百層的關鍵（對照 [`05a1`](05a1-forward-propagation.md) §5.3 的 $I + \partial f/\partial x$）。

### 5.11 LayerNorm 反向傳播總結

$$
G^{\gamma} = \textstyle\sum_{b,t} g^y \odot \hat{x}, \qquad
G^{\beta} = \textstyle\sum_{b,t} g^y
$$

$$
g^{\hat{x}} = g^y \odot \gamma
$$

$$
G^{x}
= \frac{1}{r}\!\left(
g^{\hat{x}} - \text{mean}(g^{\hat{x}}) - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})
\right), \quad r = \sqrt{\sigma^2 + \epsilon}
$$

> **讀完這一節，你會：**
> - 說出 LayerNorm 為什麼不能逐元素反向，並列出三條梯度路徑各自的來源
> - 自己推一次 variance 路徑（$-\frac12 r^{-3}$ 那一步）
> - 用「$\sum_k \hat x_k = 0$」的引理把 mean 路徑化簡，得到 §5.7 的閉式公式
> - 解釋 Residual 的恆等直通為什麼讓深層網路訓練得起來

---

## 6. 第四站：FFN

**前向見 [`05a1`](05a1-forward-propagation.md) §4。** 穿過殘差的分岔之後，其中一支進入 FFN。這一節沒有新工具——**它就是 §4 的線性層通則用兩次，中間夾一個 ReLU**，很適合拿來練手。

單一位置的前向（把中間結果都命名出來，反向才有東西可用）：

$$
u = z W_1 + b_1, \qquad
a = \text{ReLU}(u) = \max(0, u), \qquad
\text{FFN}(z) = a W_2 + b_2
$$

其中 $W_1 \in \mathbb{R}^{d \times d_{ff}}$、$W_2 \in \mathbb{R}^{d_{ff} \times d}$，慣例 $d_{ff} = 4d$。整段對整個序列做，就把 $z$ 換成 $Z \in \mathbb{R}^{T\times d}$、$a$ 換成 $A_{\text{act}} \in \mathbb{R}^{T\times d_{ff}}$。

給定上游梯度 $G^{\text{FFN}} \in \mathbb{R}^{T\times d}$，**倒著走三步**。

### 6.1 第二層線性（$W_2, b_2$）

直接套 §4.1：

$$
G^{W_2} = A_{\text{act}}^\top \, G^{\text{FFN}}, \qquad
G^{b_2} = \sum_{i=1}^{T} G^{\text{FFN}}_{i,:}, \qquad
G^{A_{\text{act}}} = G^{\text{FFN}} W_2^\top
$$

形狀檢查：$(d_{ff}\times T)\cdot(T\times d) = (d_{ff}\times d)\;\checkmark$，與 $W_2$ 同形。

### 6.2 ReLU 反向：一張開關遮罩

ReLU 是逐元素函數 $a = \max(0,u)$，微分只有兩種可能：

$$
\frac{\partial a}{\partial u} =
\begin{cases}
1, & u > 0 \\
0, & u \le 0
\end{cases}
$$

所以反向就是把上游梯度**逐元素**乘上這張 0／1 遮罩：

$$
\boxed{G^{U} = G^{A_{\text{act}}} \odot \mathbb{1}[U > 0]}
$$

**白話：** 前向時被 ReLU 砍成 0 的那些格子，代表「這條神經元這次沒有出力」，所以它也不該為這次的錯誤負責——梯度在那裡直接被切斷。反過來說，前向有通過的格子，梯度就原封不動放行。

> **注意：** 遮罩要用**前向時的 $U$**（ReLU 之前的值），不是 $A_{\text{act}}$。實作上兩者在 $u>0$ 時等價，但養成用 $U$ 的習慣比較安全。這也是為什麼前向要把 $U$ 存下來。

### 6.3 第一層線性（$W_1, b_1$）與傳回輸入

再套一次 §4.1，這次上游梯度是 $G^U$：

$$
G^{W_1} = Z^\top G^{U}, \qquad
G^{b_1} = \sum_{i=1}^{T} G^{U}_{i,:}, \qquad
G^{Z} = G^{U} W_1^\top
$$

形狀檢查：$(d\times T)\cdot(T\times d_{ff}) = (d\times d_{ff})\;\checkmark$；$G^Z$ 是 $(T\times d_{ff})\cdot(d_{ff}\times d) = (T\times d)\;\checkmark$，回到 $d$ 維，可以繼續往下一站（LayerNorm②）走。

### 6.4 三步小結

| 順序 | 這一步在做什麼 | 公式 |
|---|---|---|
| 1 | 穿過第二層線性 | $G^{W_2}=A_{\text{act}}^\top G^{\text{FFN}}$、$G^{A_{\text{act}}}=G^{\text{FFN}}W_2^\top$ |
| 2 | 穿過 ReLU | $G^{U}=G^{A_{\text{act}}}\odot\mathbb{1}[U>0]$ |
| 3 | 穿過第一層線性 | $G^{W_1}=Z^\top G^{U}$、$G^{Z}=G^{U}W_1^\top$ |

> **讀完這一節，你會：**
> - 把 §4 的線性層通則連續套兩次，推完整個 FFN 的梯度
> - 說出 ReLU 反向為什麼只是一張 0／1 遮罩，以及遮罩要用哪個量算
> - 用形狀檢查確認 $G^Z$ 確實回到了 $T\times d$

---

## 7. 第五站：Multi-Head 的 $W_O$ 與拆頭

**前向見 [`05a1`](05a1-forward-propagation.md) §3。** 再穿過一次殘差分岔與 LayerNorm①，梯度就抵達 Multi-Head Attention 的出口。

前向時，$H$ 個頭各自算出自己的輸出，拼接起來再過一個輸出投影：

$$
\text{Cat} = \text{Concat}\!\left(C^{(1)}, \ldots, C^{(H)}\right) \in \mathbb{R}^{T \times d}, \qquad
C = \text{Cat} \cdot W_O
$$

反向就是把這兩步倒過來。

### 7.1 穿過輸出投影 $W_O$

又是線性層，直接套 §4.1：

$$
G^{\text{Cat}} = G^{C} W_O^\top, \qquad
G^{W_O} = \text{Cat}^\top G^{C}
$$

形狀檢查：$G^{\text{Cat}}$ 是 $(T\times d)\cdot(d\times d) = T\times d\;\checkmark$；$G^{W_O}$ 是 $(d\times T)\cdot(T\times d) = d\times d\;\checkmark$。

### 7.2 拆回各頭：切片就是反向的拼接

前向的 Concat 只是「把 $H$ 塊並排放在一起」，沒有做任何運算。所以反向也只是**把梯度按同樣的邊界切開**，各自送回原來的頭：

$$
G^{C^{(h)}} = G^{\text{Cat}}\bigl[:,\; (h-1)d_v \,:\, h \cdot d_v\bigr] \;\in \mathbb{R}^{T\times d_v}
$$

**白話：** 拼接的反向是切片，切片的反向是拼接。這是一對互為反向的「搬家」操作，沒有乘法、不會改變梯度大小。

### 7.3 各頭獨立

前向時每個頭有自己的 $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$，彼此不交換資訊；反向也一樣**互不干擾**——拿到自己那塊 $G^{C^{(h)}}$ 之後，每個頭各自套 §8 和 §9 的單頭公式，跑完自己的一條路。

所以接下來的 §8、§9 都只討論**單一個頭**，多頭只是把同一套計算做 $H$ 次。

> **讀完這一節，你會：**
> - 算出 $G^{W_O}$ 與 $G^{\text{Cat}}$
> - 說出「Concat 的反向就是切片」，並寫出切片的索引範圍
> - 解釋為什麼推導單頭就夠了

---

## 8. 第六站：Self-Attention 核心與因果遮罩

**前向見 [`05a1`](05a1-forward-propagation.md) §1、§2。** 這是全文的重頭戲。單一個頭的前向有三步：

$$
\tilde S = \frac{QK^\top}{\sqrt{d_k}}, \qquad
A = \text{softmax}\bigl(\text{mask}(\tilde S)\bigr), \qquad
C = AV
$$

目標：從上游的 $G^C$ 出發，推出 $G^Q$、$G^K$、$G^V$。**照著前向的三步倒著走**：先過 $C=AV$（§8.2、§8.3），再過 softmax（§8.4、§8.5），最後過 $\tilde S = QK^\top/\sqrt{d_k}$（§8.6）。

### 8.1 記號與形狀

| 符號 | Shape | 說明 |
|---|---|---|
| $Q, K$ | $T \times d_k$ | Query／Key 矩陣 |
| $V$ | $T \times d_v$ | Value 矩陣 |
| $\tilde S$ | $T \times T$ | 注意力分數矩陣（縮放後）|
| $A$ | $T \times T$ | 注意力權重矩陣（softmax 後，每列和為 1）|
| $C$ | $T \times d_v$ | 輸出矩陣 |
| $G^C$ | $T \times d_v$ | 從上游傳入的梯度（§7.2 切片得到）|

### 8.2 對 $V$ 的梯度

前向 $C = AV$ 逐列寫是「用權重把各個 $V_j$ 混起來」：

$$
C_i = \sum_j A_{ij} V_j
$$

問：$V_j$ 被用了幾次？答：**每一列 $C_i$ 都用到了它**（權重是 $A_{ij}$）。所以套 §1.4，要對所有 $i$ 求和：

$$
G^{V_j} = \sum_i A_{ij} \cdot G^C_i
$$

寫成矩陣：

$$
\boxed{G^{V} = A^\top G^C}
$$

形狀檢查：$(T \times T)^\top \cdot (T \times d_v) = (T \times d_v)\;\checkmark$

**白話：** 一個 token 的 value 被別人關注得愈多（$A_{ij}$ 愈大），它收到的梯度就愈多——「被引用得多，責任就大」。

### 8.3 對 $A$ 的梯度

同樣從 $C = AV$ 出發，這次問「權重 $A_{ij}$ 動一點點，$C$ 會怎麼變」。$A_{ij}$ 只出現在 $C_i$ 這一列，且係數是 $V_j$：

$$
G^{A}_{ij} = G^C_i \cdot V_j \quad (\text{兩個 } d_v \text{ 維向量的內積})
$$

寫成矩陣：

$$
G^A = G^C V^\top
$$

形狀檢查：$(T \times d_v) \cdot (d_v \times T) = (T \times T)\;\checkmark$

**白話：** 「該不該多關注 $j$？」的答案，取決於 $j$ 的內容 $V_j$ 和「我現在想要的東西」$G^C_i$ 有多合拍。方向一致（內積為正）就是「該多看一點」。

### 8.4 穿過 Softmax

$A$ 是對 $\tilde S$ 逐列做 softmax 得到的。**Jacobian 在 §3.2 已經推導完畢**，這裡直接套用——把 §3.2 的 $a_j$ 換成 $A_{i,j}$、$e_l$ 換成 $\tilde S_{i,l}$（固定第 $i$ 列）：

$$
\frac{\partial A_{i,j}}{\partial \tilde S_{i,l}} = A_{i,j}(\delta_{jl} - A_{i,l})
$$

套鏈式法則。注意這一列的**每一個** $A_{i,j}$ 都受 $\tilde S_{i,l}$ 影響，所以要對 $j$ 求和：

$$
G^{\tilde S}_{i,l}
= \sum_j G^A_{i,j} \cdot \frac{\partial A_{i,j}}{\partial \tilde S_{i,l}}
= \sum_j G^A_{i,j} \cdot A_{i,j}(\delta_{jl} - A_{i,l})
$$

把括號拆開、分成兩個求和：

$$
= \underbrace{\sum_j G^A_{i,j} A_{i,j} \delta_{jl}}_{\delta_{jl}\text{ 只有 } j=l \text{ 那一項是 1，其餘為 0}}
\;-\; \underbrace{\sum_j G^A_{i,j} A_{i,j} A_{i,l}}_{A_{i,l} \text{ 與 } j \text{ 無關，可提到 } \sum \text{ 外}}
= G^A_{i,l} A_{i,l} - A_{i,l} \sum_j A_{i,j} G^A_{i,j}
$$

提出公因子 $A_{i,l}$：

$$
G^{\tilde S}_{i,l}
= A_{i,l} \underbrace{\left(G^A_{i,l} - \sum_j A_{i,j} G^A_{i,j}\right)}_{\text{去中心化}}
$$

記 $s_i = \sum_j A_{i,j} G^A_{i,j} = \langle A_{i,:},\, G^A_{i,:} \rangle$（這一列的加權平均梯度），得到最終結果：

$$
\boxed{G^{\tilde S}_{i,:} = A_{i,:} \odot \left(G^A_{i,:} - s_i \mathbf{1}\right)}
$$

**白話：** 兩個動作。先「去中心化」——每一格減掉這一列的加權平均，於是 $\sum_l G^{\tilde S}_{i,l} = 0$，也就是**梯度只告訴模型「該把注意力從哪裡搬到哪裡」，不會叫它整列一起變大**（因為 softmax 後每列必須和為 1，整列一起變大是沒有意義的方向）。再乘上 $A_{i,:}$——原本就沒被關注的位置（$A$ 接近 0），梯度也接近 0。

### 8.5 因果遮罩的反向：自動歸零

**前向見 [`05a1`](05a1-forward-propagation.md) §2。** 前向時，因果遮罩把「看向未來」的格子設成 $-\infty$，softmax 之後那些格子的權重恰好是 0。反向要怎麼處理？

**好消息：什麼都不用做。** 把 $A_{i,l} = 0$ 代進 §8.4 的公式：

$$
G^{\tilde S}_{i,l} = \underbrace{A_{i,l}}_{=\,0} \cdot \left(G^A_{i,l} - s_i\right) = 0
$$

被遮住的位置**自動**拿到零梯度。這件事在數學上很合理：那些格子在前向完全沒有參與計算（權重是 0），對 loss 沒有貢獻，自然也不該收到任何修正訊號。實作上，PyTorch 的 `masked_fill(..., -inf)` 反向會做同樣的事。

> **一個容易忽略的副作用：** 序列的**第 0 個位置**只能看自己，所以它那一列的 softmax 輸出是 $[1, 0, 0, \ldots]$——完美的 one-hot，也就是 §3.2 說的「飽和」。代進公式：$s_0 = 1\cdot G^A_{0,0}$，於是 $G^{\tilde S}_{0,0} = 1\cdot(G^A_{0,0} - G^A_{0,0}) = 0$，其餘 $l>0$ 因 $A_{0,l}=0$ 也是 0。**整個第 0 列拿不到任何梯度。** 這不是 bug，是 one-hot 分佈下 softmax Jacobian 退化的必然結果（[`05b2`](05b2-backward-example.md) §5.4 有數值印證）。

### 8.6 對 $Q$ 與 $K$ 的梯度

最後一步，穿過 $\tilde S = QK^\top/\sqrt{d_k}$。逐格寫是：

$$
\tilde S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}}
$$

**對 $Q$：** $q_i$（第 $i$ 個 query）出現在 $\tilde S$ 的**整個第 $i$ 列**（它跟每個 $k_j$ 都比對過），所以要對 $j$ 求和：

$$
\frac{\partial \tilde S_{ij}}{\partial q_i} = \frac{k_j}{\sqrt{d_k}}
\quad \Rightarrow \quad
G^{q_i} = \sum_j G^{\tilde S}_{ij} \cdot \frac{k_j}{\sqrt{d_k}}
$$

矩陣形式：

$$
\boxed{G^{Q} = \frac{1}{\sqrt{d_k}} \, G^{\tilde S} K}
$$

**對 $K$：** $k_j$ 出現在 $\tilde S$ 的**整個第 $j$ 欄**（每個 query 都跟它比對過），所以改成對 $i$ 求和：

$$
\frac{\partial \tilde S_{ij}}{\partial k_j} = \frac{q_i}{\sqrt{d_k}}
\quad \Rightarrow \quad
G^{k_j} = \sum_i G^{\tilde S}_{ij} \cdot \frac{q_i}{\sqrt{d_k}}
$$

矩陣形式（因為變成沿欄求和，$G^{\tilde S}$ 要轉置）：

$$
\boxed{G^{K} = \frac{1}{\sqrt{d_k}} \, (G^{\tilde S})^\top Q}
$$

形狀檢查：$G^Q$ 是 $(T\times T)\cdot(T\times d_k) = T\times d_k\;\checkmark$；$G^K$ 是 $(T\times T)^\top\cdot(T\times d_k) = T\times d_k\;\checkmark$。

**兩條式子唯一的差別就是那個轉置**，來源是「$q_i$ 管一列、$k_j$ 管一欄」。

### 8.7 $1/\sqrt{d_k}$ 在反向做了什麼

前向除以 $\sqrt{d_k}$ 是為了不讓 softmax 飽和（[`05a1`](05a1-forward-propagation.md) §1.3）。反向時，這個常數原封不動出現在 $G^Q$、$G^K$ 裡——常數的微分還是常數。

它的真正作用是**間接的**：沒有縮放的話，$A$ 會非常接近 one-hot，§3.2 的 Jacobian 趨近零矩陣，於是 §8.4 算出的 $G^{\tilde S}$ 幾乎全是 0，$Q$ 和 $K$ 就再也學不動了。**縮放保護的不是梯度的大小，而是 softmax 沒有飽和這件事。**

### 8.8 梯度流總結

$$
\mathcal{L}
\;\xrightarrow{G^C}\;
C = AV
\;\longrightarrow\;
\begin{cases}
V \xleftarrow{\;A^\top G^C\;} \\[4pt]
A \xleftarrow{\;G^C V^\top\;}
\end{cases}
\;\xrightarrow{\text{softmax}}\;
G^{\tilde S}
\;\longrightarrow\;
\begin{cases}
Q \xleftarrow{\;\frac{1}{\sqrt{d_k}} G^{\tilde S} K\;} \\[4pt]
K \xleftarrow{\;\frac{1}{\sqrt{d_k}} (G^{\tilde S})^\top Q\;}
\end{cases}
$$

**三條核心結果：**

$$
G^{Q} = \frac{1}{\sqrt{d_k}} G^{\tilde S} K, \qquad
G^{K} = \frac{1}{\sqrt{d_k}} (G^{\tilde S})^\top Q, \qquad
G^{V} = A^\top G^C
$$

注意 $G^V$ 的路徑短得多——它不必穿過 softmax。這有實際後果：**同一個 Block 裡，$W_V$ 通常學得比 $W_Q, W_K$ 快**（[`05b2`](05b2-backward-example.md) §5.7 有數值印證）。

> **讀完這一節，你會：**
> - 依「$C=AV$ → softmax → $QK^\top$」的倒序推完 $G^V$、$G^A$、$G^{\tilde S}$、$G^Q$、$G^K$
> - 說出 $G^{\tilde S}_{i,:} = A_{i,:}\odot(G^A_{i,:}-s_i)$ 的「去中心化」在做什麼
> - 解釋因果遮罩為什麼在反向自動歸零，以及第 0 列為什麼拿不到梯度
> - 說出 $G^Q$ 與 $G^K$ 只差一個轉置的原因

---

## 9. 第七站：$W_Q, W_K, W_V$ 三個投影矩陣

**前向見 [`05a1`](05a1-forward-propagation.md) §1.1。** $Q, K, V$ 不是憑空來的，是輸入 $X$（在 Pre-LN Block 裡就是 $\text{LN}_1$ 的輸出）經三個投影矩陣算出來的：

$$
Q = X W_Q, \qquad K = X W_K, \qquad V = X W_V
$$

三條都是線性層，直接套 §4.1。

### 9.1 三個權重的梯度

$$
G^{W_Q} = X^\top G^{Q}, \qquad
G^{W_K} = X^\top G^{K}, \qquad
G^{W_V} = X^\top G^{V}
$$

形狀檢查（以 $W_Q$ 為例）：$(d \times T) \cdot (T \times d_k) = (d \times d_k)\;\checkmark$

> **注意上游要用對。** 這裡的上游梯度是 §8.6／§8.2 算出的 $G^Q, G^K, G^V$ **各自**的值，**不是**下面 §9.2 合流後的 $G^X$。順序不能顛倒。

### 9.2 三條路合流回 $X$

$X$ 在前向被用了三次（分別投影成 $Q$、$K$、$V$），所以套 §1.4，反向要收三份梯度相加：

$$
\boxed{G^{X}
= G^{Q} W_Q^\top
+ G^{K} W_K^\top
+ G^{V} W_V^\top}
$$

形狀檢查（第一項）：$(T \times d_k) \cdot (d_k \times d) = (T \times d)\;\checkmark$

拿到 $G^X$ 之後，梯度就繼續往 LayerNorm①（§5.7 的公式）走，再經過殘差①的合流，最後抵達 embedding。

> **讀完這一節，你會：**
> - 寫出三個投影矩陣的梯度，並說出上游該用哪個量
> - 解釋 $G^X$ 為什麼是三項相加
> - 用形狀檢查確認 $G^X$ 回到了 $T\times d$

---

## 10. 終點站：Embedding 與位置編碼

**前向見 [`05a1`](05a1-forward-propagation.md) §6。** 梯度走完所有 Block，終於回到起點：token id 變成向量的那一步。這是計算圖的**葉節點**，梯度到此為止。

### 10.1 符號與前向回顧

| 符號 | 意義 | 維度 |
|---|---|---|
| $V$ | 詞彙表大小 | — |
| $E \in \mathbb{R}^{V \times d}$ | Token Embedding 矩陣 | $V \times d$ |
| $P \in \mathbb{R}^{T_{\max} \times d}$ | 位置編碼矩陣（Learned PE）| $T_{\max} \times d$ |
| $t_i \in \{0,\ldots,V-1\}$ | 位置 $i$ 的 token ID | — |
| $\delta_{t_i} \in \mathbb{R}^V$ | 第 $t_i$ 個 one-hot 向量 | $V$ |
| $g_i = G^{x_{\text{embed},i}}$ | 走完所有 Block 傳回來的梯度 | $d$ |

前向（[`05a1`](05a1-forward-propagation.md) §6.1、§6.2）：

$$
x_{\text{embed},i} = \underbrace{E[t_i]}_{\text{查表}} + \underbrace{P[i]}_{\text{位置}}, \qquad
E[t_i] = \delta_{t_i}^\top E
$$

「查表」在數學上等於 one-hot 乘矩陣，這個寫法讓它變成一個標準線性層，梯度就好推了。

### 10.2 Lookup 反向：稀疏更新

套 §4.1 的 $G^W = X^\top G^Y$，這裡的「輸入」是 one-hot 向量：

$$
G^{E} = \sum_{i=1}^{T} \delta_{t_i}^\top \, g_i
$$

這是一堆外積相加（$\delta_{t_i}^\top$ 是 $V\times1$、$g_i$ 是 $1\times d$，每一項都是 $V\times d$ ✓）。但 one-hot 讓它變得極其簡單——取出第 $k$ 列：

$$
\boxed{G^{E[k]}\Big|_{\text{input}} = \sum_{\{i\,:\,t_i = k\}} g_i}
$$

因為 $\delta_{t_i}$ 只有第 $t_i$ 格是 1、其餘全是 0，所以第 $k$ 列只會從「$t_i = k$ 的那些位置」收到梯度。

**兩個直接推論：**

1. **稀疏**：這一步只有本 batch 出現過的 token 才有非零梯度，其餘 $V$ 減掉那幾個 token 的列**完全不動**。
2. **累加**：同一個 token 在序列裡出現 $m$ 次，就把 $m$ 份 $g_i$ 加起來（又是「用幾次收幾份」）。

**白話：** 「這個字這次沒出現，那模型對它的理解這一步就不會改變。」這也解釋了為什麼稀有詞的 embedding 需要更多訓練資料才學得好。

### 10.3 輸出側與 Weight Tying

Embedding 矩陣還有機會從**另一條路**拿到梯度。回想 §4.3：lm_head 的參數梯度 $G^{W_{lm}[k]} = \sum_i \delta_i^{(k)}\hat h_i$ 是**稠密**的（每一列都非零）。

- **本倉庫 NB4 的做法（不共用）：** $W_{lm}$ 和 $E$ 是兩個獨立參數。輸入側的稀疏梯度歸 $E$、輸出側的稠密梯度歸 $W_{lm}$，各自更新、不相加。
- **Weight Tying（Karpathy 原版 nanoGPT）：** 讓 $W_{lm} = E$ 共用同一份矩陣。此時 $E$ 在前向被用了兩次（查表一次、算 logit 一次），依 §1.4 反向就要收兩份：

$$
\boxed{G^{E[k]} = \underbrace{\sum_{\{i\,:\,t_i = k\}} g_i}_{\text{輸入側（稀疏）}} + \underbrace{\sum_{i=1}^{T} \delta_i^{(k)} \, \hat{h}_i}_{\text{輸出側（稠密）}}}
$$

**Weight Tying 的好處就在這條式子裡：** 一個稀有 token 就算這個 batch 完全沒出現過（輸入側梯度為 0），它仍然會從輸出側收到「這次不該選你」的梯度，所以它的 embedding 一樣在被訓練。

### 10.4 三個重要特性

| 特性 | 輸入側（Lookup）| 輸出側（lm_head）|
|---|---|---|
| **稀疏性** | 只有出現過的 token 列有梯度 | 所有 $V$ 列都有梯度（稠密）|
| **梯度來源** | 語言模型「讀入」時的表示學習 | 語言模型「預測」時的對比信號 |
| **Weight Tying 的收益** | — | 稀有 token 即使沒被選為輸入，也能從輸出側持續收到梯度 |

**實作注意：** PyTorch `nn.Embedding(sparse=True)` 只傳輸有梯度的列，在詞彙量 $V > 10^5$ 時顯著節省反向傳播的記憶體與頻寬。

### 10.5 位置編碼的梯度

Learned PE 的前向是單純的相加 $x_{\text{embed},i} = E[t_i] + P[i]$，而加法的反向就是**原封不動複製**（§5.10 已經看過一次）：

$$
\boxed{G^{P[i]} = \sum_{b} g_{b,i}}
$$

（對 batch 維求和，因為同一個位置 $i$ 的 $P[i]$ 被 batch 裡每一個序列共用。）

和 Embedding 的差別很有意思：**$E$ 是「哪些 token 出現就更新哪幾列」，$P$ 則是「前 $T$ 列每步必定更新」**——只要序列長度是 $T$，位置 0 到 $T-1$ 就一定被用到。所以訓練時很少出現的長位置（接近 $T_{\max}$）學得比較差，這是 Learned PE 難以外推到更長序列的原因之一。

如果用的是 Sinusoidal PE（固定公式、沒有參數），那就沒有 $G^P$ 這回事——梯度到 $E$ 就結束了。

> **讀完這一節，你會：**
> - 推出 $G^{E[k]} = \sum_{t_i=k} g_i$，並說明 one-hot 為什麼讓它變成稀疏
> - 說出同一個 token 出現多次時梯度怎麼處理
> - 寫出 Weight Tying 的雙通道總梯度，並解釋它為什麼對稀有 token 有幫助
> - 說出 $G^P$ 與 $G^E$ 在「哪幾列會被更新」上的差別

---

## 11. 最後一步：optimizer 拿梯度做什麼

全部參數的梯度都算完了，`loss.backward()` 到此結束。最後一行 `optimizer.step()` 做的就是 §1.1 講過的那件事——對**每一個**參數 $\theta$：

$$
\theta \leftarrow \theta - \eta \cdot G^{\theta}
$$

這是最單純的 SGD。實務上用的 Adam 會再加上動量與逐參數的自適應步長，但核心不變：**往梯度的反方向走一小步。**

一次訓練迭代的完整循環：

| 步驟 | 程式 | 本文對應 |
|---|---|---|
| 1. 前向 | `logits, loss = model(x, y)` | [`05a1`](05a1-forward-propagation.md) §1–§7 |
| 2. 清空舊梯度 | `optimizer.zero_grad()` | —（否則會和上一步的梯度累加）|
| 3. 反向 | `loss.backward()` | 本文 §3–§10 |
| 4. 更新 | `optimizer.step()` | 本文 §11 |

然後換下一個 batch，重複幾千次到幾十萬次。**這就是「訓練一個語言模型」的全部。**

> **讀完這一節，你會：**
> - 說出 optimizer 用梯度做的唯一一件事
> - 依序說出一次訓練迭代的四個步驟，並指出每一步對應本文哪一章

---

## 核心總結

**反向傳播只有兩條規則**，其餘都是把它們套在不同模組上：

1. **鏈式法則**：這一層的梯度 ＝ 上游梯度 × 這一層自己的變化率（§1.3）
2. **分岔相加**：一個量在前向被用幾次，反向就收幾份梯度（§1.4）

**四個最常用的模組公式：**

| 模組 | 梯度 | 出處 |
|---|---|---|
| Softmax ＋ Cross-Entropy | $\delta_i^{(k)} = \frac1T(p^{(i)}_k - \mathbb{1}[k=y_i])$ | §3.3 |
| 線性層 $Y = XW$ | $G^W = X^\top G^Y$、$G^X = G^Y W^\top$ | §4.1 |
| LayerNorm | $G^{x} = \frac1r\bigl(g^{\hat x} - \text{mean}(g^{\hat x}) - \hat x \odot \text{mean}(g^{\hat x}\odot\hat x)\bigr)$ | §5.7 |
| Self-Attention | $G^Q = \frac{1}{\sqrt{d_k}}G^{\tilde S}K$、$G^K = \frac{1}{\sqrt{d_k}}(G^{\tilde S})^\top Q$、$G^V = A^\top G^C$ | §8.8 |

**五個關於「梯度好不好流」的結論：**

1. **Residual 是梯度高速公路**（§5.10）：加法的反向是原樣複製，提供一條完全不衰減的直通路徑，深層網路才訓練得起來。
2. **LayerNorm 每穿一次就削一次**（§5.9）：反向會扣掉沿 $\mathbf{1}$ 與沿 $\hat x$ 的兩個分量。
3. **Softmax 飽和 ⇒ 梯度消失**（§3.2）：Jacobian 含 $a_j(1-a_j)$，輸出接近 one-hot 時整個趨近零矩陣。$1/\sqrt{d_k}$ 縮放就是為了避開這件事（§8.7）。
4. **遮罩位置零梯度是自動的**（§8.5）：前向沒參與，反向就收不到，不需要額外處理。
5. **Embedding 稀疏、lm_head 稠密**（§10.4）：這是「查表」與「比對全詞彙表」兩種運算的必然差異，也是 Weight Tying 有效的原因。

---

## 附錄 A：完整梯度查閱表

從損失 $\mathcal{L}$ 到所有葉節點（可訓練參數）的完整梯度公式，依**反向的行進順序**排列。

### A.1 輸出端：Cross-Entropy、Softmax、lm_head

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $\delta_i = G^{z_i}$（CE ＋ Softmax 合併）| $\delta_i^{(k)} = \tfrac{1}{T}(p^{(i)}_k - \mathbb{1}[k=y_i])$ | §3.3 |
| Softmax 的 Jacobian | $\partial a_j/\partial e_l = a_j(\delta_{jl} - a_l)$ | §3.2 |
| $G^{\hat h_i}$ | $\delta_i W_{lm}$ | §4.3 |
| $G^{W_{lm}}$（稠密）| $\sum_i \delta_i^\top \hat h_i$ | §4.3 |

### A.2 線性層通則（$Y = XW + b$）

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $G^{W}$ | $X^\top G^{Y}$ | §4.1 |
| $G^{X}$ | $G^{Y} W^\top$ | §4.1 |
| $G^{b}$ | $\sum_i G^{Y}_{i,:}$ | §4.1 |

### A.3 LayerNorm（$y_j = \gamma_j \hat{x}_j + \beta_j$）與 Residual

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $G^{\gamma}$ | $\sum_{b,t} g^y \odot \hat{x}$ | §5.2 |
| $G^{\beta}$ | $\sum_{b,t} g^y$ | §5.2 |
| $g^{\hat{x}}$（中間量）| $g^y \odot \gamma$ | §5.3 |
| $G^{x}$ | $\dfrac{1}{r}\!\left(g^{\hat{x}} - \text{mean}(g^{\hat{x}}) - \hat{x} \odot \text{mean}(g^{\hat{x}} \odot \hat{x})\right)$ | §5.7 |
| Residual $U = X + Z$ | $G^X\big|_{\text{res}} = G^U$、$G^Z = G^U$ | §5.10 |

### A.4 FFN（$\text{FFN}(z) = \text{ReLU}(zW_1+b_1)W_2 + b_2$）

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $G^{W_2}$、$G^{A_{\text{act}}}$ | $A_{\text{act}}^\top G^{\text{FFN}}$、$G^{\text{FFN}} W_2^\top$ | §6.1 |
| $G^{U}$（ReLU 反向）| $G^{A_{\text{act}}} \odot \mathbb{1}[U > 0]$ | §6.2 |
| $G^{W_1}$、$G^{Z}$ | $Z^\top G^{U}$、$G^{U} W_1^\top$ | §6.3 |

### A.5 Multi-Head 與 Self-Attention

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $G^{W_O}$、$G^{\text{Cat}}$ | $\text{Cat}^\top G^{C}$、$G^{C} W_O^\top$ | §7.1 |
| $G^{C^{(h)}}$（拆頭）| $G^{\text{Cat}}[:, (h-1)d_v : h\,d_v]$ | §7.2 |
| $G^V$ | $A^\top G^C$ | §8.2 |
| $G^A$ | $G^C V^\top$ | §8.3 |
| $G^{\tilde S}$（Softmax 反向）| $A_{i,:} \odot (G^A_{i,:} - \langle A_{i,:}, G^A_{i,:}\rangle)$ | §8.4 |
| 遮罩位置 | 自動為 0（因 $A_{i,l}=0$）| §8.5 |
| $G^Q$ | $\dfrac{1}{\sqrt{d_k}} G^{\tilde S} K$ | §8.6 |
| $G^K$ | $\dfrac{1}{\sqrt{d_k}} (G^{\tilde S})^\top Q$ | §8.6 |
| $G^{W_Q}, G^{W_K}, G^{W_V}$ | $X^\top G^{Q}$、$X^\top G^{K}$、$X^\top G^{V}$ | §9.1 |
| $G^{X}$（三路合流）| $G^{Q}W_Q^\top + G^{K}W_K^\top + G^{V}W_V^\top$ | §9.2 |

### A.6 輸入端：Embedding 與位置編碼

| 梯度目標 | 公式 | 章節 |
|---|---|---|
| $G^{E[k]}$（輸入側，稀疏）| $\sum_{\{i:\,t_i=k\}} g_i$ | §10.2 |
| $G^{E[k]}$（Weight Tying 總梯度）| 輸入側 ＋ 輸出側 $\sum_i \delta_i^{(k)}\hat h_i$ | §10.3 |
| $G^{P[i]}$ | $\sum_b g_{b,i}$ | §10.5 |
| 參數更新 | $\theta \leftarrow \theta - \eta G^{\theta}$ | §11 |

> **說明：** $r = \sqrt{\sigma^2 + \epsilon}$，$\hat{x} = (x-\mu)/r$，$\text{mean}(\cdot) = \tfrac{1}{d}\sum_j (\cdot)_j$，$\tilde S = QK^\top/\sqrt{d_k}$（[`05b2`](05b2-backward-example.md) 簡寫成 $S$）。查閱表中所有公式皆有對應的符號推導章節，並在 [`05b2`](05b2-backward-example.md) 有數值驗證。

---

## 附錄 B：為什麼 Transformer 比 RNN 好訓練

本文推完之後，可以回答一個架構層面的問題：為什麼 Transformer 取代了 RNN。答案就藏在梯度的路徑長度裡。

**RNN 的問題：梯度要穿越所有時間步，形成連乘積**

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$

若每一項的範數 $\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| < 1$，$T$ 個小於 1 的數連乘會指數衰減到 0（**梯度消失**）；若 $> 1$ 則指數放大（**梯度爆炸**）。序列愈長，問題愈嚴重。

> **這裡只給結論，完整推導在 [`10b1`](10b1-seq2seq-backward.md) §A6。** 那裡把 $\partial h_t/\partial h_{t-1}$ 拆成 $\text{diag}(1-h_t^2)$ 與 $W_{hh}^\top$ 兩個因子，說明為何 $\tanh$ 保證前者恆 $\le 1$、後者的 $\sigma_{\max}$ 又決定指數衰減或爆炸，並對照 LSTM 閘控與 Bahdanau attention 兩種緩解手段。數值版（含「attention 那條路帶回的梯度比時間鏈還大」的實測）見 [`10b2`](10b2-seq2seq-backward-example.md) §A3。

**Transformer 的優勢：任意兩個 token 之間只隔一層 attention**

從 §8.2 的 $G^V = A^\top G^C$ 可以直接看出來，位置 $i$ 到位置 $j$ 的梯度路徑是：

$$
\frac{\partial C_i}{\partial x_j} \propto A_{ij} W_V^\top
$$

**路徑長度 $= O(1)$，完全不隨序列長度 $T$ 增長。** 因此：

- 梯度不因序列長度衰減
- 長距離依賴（例如 100 個 token 之外的指代關係）與短距離依賴一樣容易學習
- 深層 Transformer 靠 **Residual Connection**（§5.10）而非時間鏈傳遞梯度

| | RNN | Transformer |
|---|---|---|
| 跨 $T$ 步的梯度路徑 | 連乘 $T$ 次 | $O(1)$，只穿一層 attention |
| 跨 $L$ 層的梯度路徑 | — | 有 Residual 恆等直通（§5.10）|
| 主要風險 | 梯度消失／爆炸 | Softmax 飽和（靠 $1/\sqrt{d_k}$ 緩解，§8.7）|

---

## 下一步

你已經能從頭推導 GPT 每個模組的梯度，並說出梯度從 loss 回到 Embedding 的完整路線。

**先看數值：把本文每條公式實算一次** → [`05b2-backward-example.md`](05b2-backward-example.md)

沿用 [`05a2`](05a2-forward-example.md) 的一組數字（$T=2$、$d=3$、單頭、含因果遮罩），從 $\partial L/\partial \text{logits}$ 一路算到 $\partial L/\partial E$，每一步乘加都展開。

**動手實作：** → [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb)

用 NumPy 實作本文推導的每一條梯度公式，並以數值梯度驗證正確性。

**對照程式：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md) → [`../notebooks/NB4-nanoGPT.ipynb`](../notebooks/NB4-nanoGPT.ipynb)
