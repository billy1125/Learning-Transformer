# 03b3｜Transformer Block 計算案例（完整版）：一條前向流程從頭算到尾

> **這是三階段計算案例的最後一階段（完整版）。** 若你是第一次接觸，建議從輕量的前兩階段循序而上（三份共用同一組 $X$ 與權重，數字完全銜接）：
> 1. [`03b1-transformer-example-basic.md`](03b1-transformer-example-basic.md)（簡單版）— 單頭 attention：$X \to \tilde X \to C^{(1)}$
> 2. [`03b2-transformer-example-block.md`](03b2-transformer-example-block.md)（中等版）— 補上第二頭、$W_O$、殘差、FFN，算到 Block 輸出 $Y$
> 3. **03b3（本文，完整版）** — 在前兩階段之上，再補縮放數值對照、Positional Encoding 與相對位置旋轉驗證、NB1 重現
>
> 本文自成一體（重述完整流程），可單獨閱讀；若已讀過 03b1／03b2，可略過重複的步驟、直接看 §0 設定的設計推導（解釋每個矩陣為什麼長這樣）、§2.3 縮放對照、§6 Positional Encoding 與 §7 NB1 重現。

> **閱讀定位：** 03b1 是「只把一個 head 算清楚」，03b2 是「把同一筆資料補成完整 Block」，03b3 則是「把為什麼這樣設計、縮放效果、位置編碼與程式重現一次收齊」。因此本文不是換一組新範例，而是把同一條數值流從簡單版延伸到完整解釋版。

> **適合對象：** 已讀完 [`03a-transformer-architecture.md`](03a-transformer-architecture.md)，想用具體數字把整個 Transformer Block 親手算一遍的讀者。
>
> **讀完後你能做什麼：**
> - 用一組 $2\times4$ 的輸入，手算 Pre-LN → Multi-Head Attention → 殘差 → FFN → 殘差的**每一步**
> - 親眼看到除以 $\sqrt{d_k}$ 如何讓 softmax 分佈變平滑（含縮放／不縮放對照）
> - 驗證 $W_O$ 確實「混合重組」各 head 的資訊（而非只是拼接）
> - 手算 LayerNorm 的 mean／var／normalize 三步
> - 用具體數字驗證 Positional Encoding 的「相對位置 = 旋轉」性質
>
> **前置文件：** [`03a-transformer-architecture.md`](03a-transformer-architecture.md)（符號、公式、Pre-LN 定義皆沿用該文）
>
> **對應實作：** [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13「與 03a 對照：逐步數值驗證」可一鍵重現本文所有數字。

---

> 本文是 [`03a`](03a-transformer-architecture.md) §5.6「最小數值例」的**完整版**：§5.6 只把 Multi-Head Attention 單獨作用在輸入上；這裡把它放進一個**完整的 Pre-LN Block**，從頭算到尾。
>
> **與 §5.6 的關鍵差異：** 本文採 Pre-LN，Attention 的輸入是 $\text{LayerNorm}(X)$ 而非原始 $X$，因此 $Q,K,V$ 的數字與 §5.6 不同——這正是 Pre-LN 的特徵，後面會清楚標出。
>
> 所有符號（$X, S, E, A, O, W_O, \tilde X, \ldots$）定義見 [`03a` 的符號表](03a-transformer-architecture.md#符號表)。數字一律四捨五入到小數第 4 位，最末位可能有 $\pm 0.0001$ 的進位誤差。

---

## 目錄

0. 設定：一次給齊所有矩陣
1. Pre-LN 第一步：$\tilde X = \text{LayerNorm}(X)$
2. 單頭 Scaled Dot-Product（Head 1 逐步）＋ 縮放對照
3. Head 2、拼接與 $W_O$ 混合
4. 第一個殘差連接
5. 第二子層：LayerNorm → FFN → 第二個殘差
6. Positional Encoding 數值與相對位置驗證
7. 用 NB1 重現本文數字
8. 全流程總表與檢核

---

## 0. 設定：一次給齊所有矩陣

本文為了能一條龍算到尾，**在開頭就把所有權重給齊**。但「給齊」不等於「天上掉下來」——每個矩陣的形狀都是被前後維度咬合逼出來的，數值也都是為了某個示範目的而設計的。本節就把這個**設計歷程**講清楚：先定維度約束，再逐一說明 $X$、投影矩陣、$W_O$、LayerNorm、FFN 各自「為什麼長這樣」。各矩陣的具體運算留到 §1 起逐步登場，這裡只交代來歷。

### 0.0 先把三份文件的銜接看清楚

三份 03b 文件的核心策略是：**同一筆資料，逐步加上更多 Transformer 元件**。這比每一份都換新例子更適合初學者，因為讀者可以把前一份算出的中間結果直接帶到下一份，不必重新適應新的數字。

| 文件 | 主要問題 | 算到哪裡 | 與本文的關係 |
|---|---|---|---|
| 03b1 | 單頭 attention 到底怎麼算？ | $X \to \tilde X \to C^{(1)}$ | 本文 §1–§2 會重述，但可快速略讀 |
| 03b2 | 如何從 attention 接成完整 Block？ | $C^{(1)},C^{(2)} \to O \to Z' \to F \to Y$ | 本文 §3–§5 會重述，並補更多設計原因 |
| 03b3 | 完整案例還缺什麼理解橋？ | 加上設計推導、縮放對照、PE、NB1 重現 | 本文定位：完整參考版與查核版 |

因此，本文的閱讀方式有兩種。若你還沒讀 03b1／03b2，可以從 §0 開始順讀，本文會完整重算一次。若你已讀過前兩份，建議把 §1–§5 當成「核對版」，重點放在 §0 的矩陣設計理由、§2.3 的縮放對照、§6 的位置編碼，以及 §8 的總表。

### 0.1 超參數與「維度必須咬合」

超參數：序列長度 $T=2$、模型維度 $d=4$、head 數 $H=2$、每頭維度 $d_k=d_v=2$、FFN 中間維度 $d_{ff}=8$。

這幾個數字不能亂填，唯一的硬約束是 **多頭要把 $d$ 維均分給 $H$ 個頭**：

$$
d = H\cdot d_k \quad\Longrightarrow\quad 4 = 2\times 2\ \checkmark
$$

一旦定下 $d,H,d_k$，**每個矩陣的形狀就被整條前向流程「逼」出來了**——下面這條形狀流把全文要用的矩陣按出場順序串起來，每個箭頭上方就是該步的矩陣與它必須有的 shape：

$$
\underset{2\times4}{X}
\xrightarrow{\ \text{LN}\ }
\underset{2\times4}{\tilde X}
\xrightarrow{\ W^{(h)}\,(4\times2)\ }
\underset{2\times2}{Q,K,V}
\to
\underset{2\times4}{\text{Concat}}
\xrightarrow{\ W_O\,(4\times4)\ }
\underset{2\times4}{O}
\to
\underset{2\times4}{Z'}
\xrightarrow{\ W_1\,(4\times8)\ }
\underset{2\times8}{\cdot}
\xrightarrow{\ W_2\,(8\times4)\ }
\underset{2\times4}{F}
\to
\underset{2\times4}{Y}
$$

讀法：投影矩陣必須是 $d\times d_k=4\times2$（才能把 $\tilde X$ 的 $4$ 維壓到每頭的 $2$ 維）；兩頭拼接回 $H\cdot d_v=4$ 維後，$W_O$ 必須是 $4\times4$（把 $d$ 維重新映回 $d$ 維）；FFN 先 $4\to8$ 再 $8\to4$，所以 $W_1$ 是 $4\times8$、$W_2$ 是 $8\times4$。**全程輸入輸出都維持 $2\times4$**，這正是 Block 能反覆堆疊的前提（[`03a` §6.1](03a-transformer-architecture.md)）。形狀確定後，剩下的只是「填什麼數值」——這就是 §0.2 之後要交代的設計。

把 [`03a`](03a-transformer-architecture.md) §6 的 Pre-LN Block 寫成本文的符號，就是下面這條主線：

$$
\tilde X=\text{LN}_1(X),\qquad O=\text{MHA}(\tilde X),\qquad Z'=X+O
$$

$$
\tilde Z=\text{LN}_2(Z'),\qquad F=\text{FFN}(\tilde Z),\qquad Y=Z'+F
$$

也就是：第一個子層負責跨 token 的資訊交換，第二個子層負責每個 token 自己的非線性加工；兩個子層外面各有一條 residual。本文後面的每一步都只是把這六個等式用具體數字展開。

### 0.2 輸入 $X$：為什麼挑這組

**輸入序列**（每列一個 token，沿用 [`03a` §5.6](03a-transformer-architecture.md)）：

$$
X = \begin{bmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 1 & 0 \end{bmatrix} \in \mathbb{R}^{2 \times 4}
$$

挑這組數字有兩個用意：(1) 兩個 token 的非零位置**互不重疊**（token 1 在第 0、3 維，token 2 在第 1、2 維），彼此正交，方便看清資訊如何流動；(2) 每列「兩個 $1$、兩個 $0$」的結構，經 LayerNorm 後會得到乾淨的 $\pm1$（見 §1），讓後面所有手算都落在整數或簡單小數上。這組 $X$ 與 03b1／03b2 完全相同，數字一路銜接。

### 0.3 LayerNorm 參數：先只做標準化

**兩個 LayerNorm** 的可學習參數取初始值 $\gamma=[1,1,1,1]$、$\beta=[0,0,0,0]$。這是 LayerNorm 的標準初始化，代入縮放／平移那一步：

$$
\tilde x=\gamma\odot\hat x+\beta=1\odot\hat x+0=\hat x
$$

也就是「先只做標準化（減均值、除標準差），暫不縮放也不平移」。這讓 §1 與 §5.1 的 LayerNorm 輸出乾淨可驗（直接等於標準化結果），不被額外參數干擾。LayerNorm 是前向流程第一個被套用的元件（§1），所以先交代它的參數。

### 0.4 投影矩陣（$W_Q,W_K,W_V$）：其實是「選欄矩陣」

這裡的「投影矩陣」**就是注意力機制裡 Query／Key／Value 的投影權重 $W_Q,W_K,W_V$**——也是整個 Block 裡與「QKV」直接相關的那組權重。它們是**固定的參數（權重）**，不是 $Q,K,V$ 本身；要等 $\tilde X$ 乘上它們、得到 $Q=\tilde X W_Q$、$K=\tilde X W_K$、$V=\tilde X W_V$ 之後，才是每個 token 的 Query／Key／Value 向量（實際相乘見 §2.1）。

**各 head 的 $W_Q,W_K,W_V$**（每個 head 各一組；Head 1 取前 2 維、Head 2 取後 2 維；本例令 $W_Q=W_K=W_V$ 以聚焦流程）：

$$
W_Q^{(1)}=W_K^{(1)}=W_V^{(1)}=\begin{bmatrix}1&0\\0&1\\0&0\\0&0\end{bmatrix},\qquad
W_Q^{(2)}=W_K^{(2)}=W_V^{(2)}=\begin{bmatrix}0&0\\0&0\\1&0\\0&1\end{bmatrix}\;\in\mathbb{R}^{4\times2}
$$

這兩個 0/1 排列不是隨手寫的——它們是把「多頭切分」寫得最透明的形式。攤開矩陣乘法的定義：

$$
(\tilde X\,W)_{ij}=\sum_{k=0}^{3}\tilde X_{ik}\,W_{kj}
$$

對 $W_Q^{(1)}$：第 $0$ 欄只有 $W_{00}=1$ 非零，故 $(\tilde X W^{(1)})_{i0}=\tilde X_{i0}$；第 $1$ 欄只有 $W_{11}=1$，故 $(\tilde X W^{(1)})_{i1}=\tilde X_{i1}$。也就是說，乘上 $W^{(1)}$ 的效果就是**挑出 $\tilde X$ 的第 0、1 欄**。同理 $W^{(2)}$ 的兩個 $1$ 落在第 2、3 列，效果是**挑出第 2、3 欄**。兩個頭合起來，恰好把 $d=4$ 維**不重不漏**地切成兩塊 $d_k=2$ 維：

$$
\underbrace{\{\text{第 }0,1\text{ 維}\}}_{\text{Head 1}}\;\sqcup\;\underbrace{\{\text{第 }2,3\text{ 維}\}}_{\text{Head 2}}=\{0,1,2,3\}
$$

> **與真實模型的關係：** 「將 $d$ 維分成 $H$ 個 head」的定義見 [`03a` §5.1](03a-transformer-architecture.md)；實務上各頭的 $W_Q,W_K,W_V$ 是**學出來的稠密矩陣**，切分靠的是把一個大投影的輸出 reshape 成 $H$ 段。本例改用顯式的選欄矩陣，是為了讓「哪一維進了哪個頭」一眼可見，數值意義不變。
>
> 另外令 $W_Q=W_K=W_V$（同一頭的三個投影相同），會使 $Q^{(h)}=K^{(h)}=V^{(h)}$，凸顯對稱、把注意力公式裡真正在變動的部分孤立出來——這也是教學簡化，真實模型三者各自獨立。

### 0.5 輸出投影權重 $W_O$：把「混合」寫進矩陣裡

$W_O$ **也是注意力子層的權重**，但和 §0.4 的 $W_Q,W_K,W_V$ 角色不同：QKV 權重在 attention **之前**把 $\tilde X$ 投影成各 head 的 $Q,K,V$；$W_O$ 則在各 head 算完、拼接**之後**才作用，把它們混合投影回 $d$ 維。換句話說，一個注意力子層的權重共有兩組——**QKV 投影權重（§0.4）**與**輸出投影權重 $W_O$（本節）**。

**輸出投影權重 $W_O$**（刻意設計成「前 2 維取兩頭的和、後 2 維取兩頭的差」，方便驗證混合效果）：

$$
W_O = \frac{1}{2}\begin{bmatrix}1&0&1&0\\0&1&0&1\\1&0&-1&0\\0&1&0&-1\end{bmatrix}\in\mathbb{R}^{4\times4}
$$

這個矩陣是本文最「有戲」的設計，值得當場把它在做什麼推出來。拼接後的輸入是 $\text{Concat}=[\,C^{(1)}_{:,0}\;\;C^{(1)}_{:,1}\;\;C^{(2)}_{:,0}\;\;C^{(2)}_{:,1}\,]$（四欄，前兩欄來自 Head 1、後兩欄來自 Head 2）。輸出 $O=\text{Concat}\cdot W_O$，其第 $j$ 欄就是 $\text{Concat}$ 以 $W_O$ 的第 $j$ **欄**為權重的線性組合。逐欄讀 $W_O$（記得整體有 $\tfrac12$）：

- 第 0 欄 $\tfrac12[1,0,1,0]^\top$：取 $\text{Concat}$ 的第 0、2 欄 → $O_{:,0}=\tfrac12\big(C^{(1)}_{:,0}+C^{(2)}_{:,0}\big)$
- 第 1 欄 $\tfrac12[0,1,0,1]^\top$：取第 1、3 欄 → $O_{:,1}=\tfrac12\big(C^{(1)}_{:,1}+C^{(2)}_{:,1}\big)$
- 第 2 欄 $\tfrac12[1,0,-1,0]^\top$：取第 0 欄減第 2 欄 → $O_{:,2}=\tfrac12\big(C^{(1)}_{:,0}-C^{(2)}_{:,0}\big)$
- 第 3 欄 $\tfrac12[0,1,0,-1]^\top$：取第 1 欄減第 3 欄 → $O_{:,3}=\tfrac12\big(C^{(1)}_{:,1}-C^{(2)}_{:,1}\big)$

$$
\boxed{\;O_{:,0}=\tfrac12\big(C^{(1)}_{:,0}+C^{(2)}_{:,0}\big),\quad
O_{:,1}=\tfrac12\big(C^{(1)}_{:,1}+C^{(2)}_{:,1}\big),\quad
O_{:,2}=\tfrac12\big(C^{(1)}_{:,0}-C^{(2)}_{:,0}\big),\quad
O_{:,3}=\tfrac12\big(C^{(1)}_{:,1}-C^{(2)}_{:,1}\big)\;}
$$

這四條就是 §3.3 會直接套用的公式——前兩維取兩頭的**和**、後兩維取兩頭的**差**。其中 $\tfrac12$ 是和／差的正規化：若兩頭數值同號，相加會放大兩倍，除以 $2$ 把尺度拉回原本的量級。

最關鍵的一點：$W_O$ **可逆**。它的行列式

$$
\det(W_O)=0.25\neq 0
$$

所以 $O\mapsto\text{Concat}$ 是一一對應，$W_O$ 沒有丟掉任何資訊，只是把「兩頭的座標」換成「兩頭的和與差」這組新座標——一次無損的**基底變換**。這正說明「拼接不是終點」：$W_O$ 真的在跨頭重組資訊，而不只是把兩頭擺在一起（[`03a` §5.7](03a-transformer-architecture.md)）。

### 0.6 FFN 權重（$W_1,W_2$）：形狀鏈與唯一的實質約束

$W_1,W_2$ 屬於 Block 的**第二個子層（前饋網路 FFN）**，和注意力**完全分開**——它們既不是 QKV 投影權重（§0.4），也不是輸出投影 $W_O$（§0.5）。整理一下，一個 Pre-LN Block 的可學習權重就這幾組：QKV 投影 $W_Q,W_K,W_V$、輸出投影 $W_O$、FFN 的 $W_1,W_2$，外加兩個 LayerNorm 的 $\gamma,\beta$（§0.3）。

**FFN 權重 $W_1,W_2$**（$b_1=b_2=0$）：

$$
W_1=\frac12\begin{bmatrix}
1&-1&0&1&1&0&-1&0\\
0&1&1&-1&0&1&0&-1\\
1&0&-1&0&1&-1&0&1\\
-1&1&0&1&0&0&1&-1
\end{bmatrix}\in\mathbb{R}^{4\times8},\qquad
W_2=\frac12\begin{bmatrix}
1&0&1&-1\\0&1&-1&0\\1&-1&0&1\\-1&0&1&0\\
0&1&0&1\\1&0&-1&0\\0&-1&0&1\\-1&0&1&0
\end{bmatrix}\in\mathbb{R}^{8\times4}
$$

形狀由 §0.1 的鏈決定：FFN 先升維 $4\xrightarrow{W_1}8$、再降維 $8\xrightarrow{W_2}4$，故 $W_1$ 是 $4\times8$、$W_2$ 是 $8\times4$。數值取 $\pm\tfrac12$ 的人造值，唯一的實質要求是 **$W_1$ 不能退化**：升維後的 $8$ 維要有正有負，中間的 ReLU 才有東西可裁（§5.2 會看到 ReLU 把約一半的元素歸零）。若 $W_1$ 退化到輸出恆正或恆負，ReLU 形同虛設，FFN 就塌回線性映射。

> 本例 $d_{ff}=8=2d$ 只是為了手算方便；實務通常 $d_{ff}=4d$。

> 以上矩陣的數值都是教學用的人造值，唯一要求是「內部自洽、可被 NB1 重現」。重點在流程、shape 與每個設計各自要示範的性質，不在權重本身的語意。

---

## 1. Pre-LN 第一步：$\tilde X = \text{LayerNorm}(X)$

Pre-LN 規定：**先正規化、再進子層**（[`03a` §6.2](03a-transformer-architecture.md)）。LayerNorm 對**每一列**（每個 token 的 $d$ 維）獨立做標準化。

以第 1 列 $x_1=[1,0,0,1]$ 為例，三步：

$$
\mu_1=\frac{1+0+0+1}{4}=0.5,\qquad
\sigma_1^2=\frac{(0.5)^2+(0.5)^2+(0.5)^2+(0.5)^2}{4}=0.25,\qquad
\sigma_1=0.5
$$

$$
\hat x_1=\frac{x_1-\mu_1}{\sigma_1}=\frac{[0.5,\,-0.5,\,-0.5,\,0.5]}{0.5}=[1,-1,-1,1]
$$

因為 $\gamma=1,\beta=0$，$\tilde x_1=\gamma\odot\hat x_1+\beta=\hat x_1$。第 2 列同理（$\mu_2=0.5,\sigma_2=0.5$）：

$$
\tilde X = \text{LayerNorm}(X)=\begin{bmatrix} 1 & -1 & -1 & 1 \\ -1 & 1 & 1 & -1 \end{bmatrix}
$$

> **這就是與 §5.6 的分岔點。** §5.6 直接拿 $X$ 算 attention；Pre-LN 拿的是 $\tilde X$。下面所有 $Q,K,V$ 都由 $\tilde X$ 算出。

---

## 2. 單頭 Scaled Dot-Product（Head 1 逐步）

### 2.1 投影

Head 1 取 $\tilde X$ 的前 2 維：

$$
Q^{(1)}=K^{(1)}=V^{(1)}=\tilde X\,W_Q^{(1)}=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

（驗算第 1 列：$\tilde x_1=[1,-1,-1,1]$ 乘 $W_Q^{(1)}$ 只留前 2 維 → $[1,-1]$ ✓）

### 2.2 原始分數 $S$

$$
S^{(1)}=Q^{(1)}\big(K^{(1)}\big)^\top
$$

逐格手算（$\big(K^{(1)}\big)^\top$ 的兩列就是 $K^{(1)}$ 的兩列）：

$$
S^{(1)}_{11}=[1,-1]\cdot[1,-1]=1+1=2,\qquad
S^{(1)}_{12}=[1,-1]\cdot[-1,1]=-1-1=-2
$$

$$
S^{(1)}=\begin{bmatrix}2&-2\\-2&2\end{bmatrix}
$$

### 2.3 縮放：為什麼要除以 $\sqrt{d_k}$（數值對照）

這裡把 [`03a` §3.4](03a-transformer-architecture.md) 的論點**用數字看一次**。對第 1 列的分數 $[2,-2]$：

| | 分數 | softmax 結果 |
|---|---|---|
| **不縮放** $S$ | $[2,\,-2]$ | $\dfrac{e^2}{e^2+e^{-2}}=\dfrac{7.389}{7.524}=0.9820,\ \ 0.0180$ |
| **縮放** $E=S/\sqrt2$ | $[1.4142,\,-1.4142]$ | $\dfrac{e^{1.4142}}{e^{1.4142}+e^{-1.4142}}=\dfrac{4.113}{4.356}=0.9442,\ \ 0.0558$ |

不縮放時權重 $0.982$ 更接近 one-hot（更尖銳），梯度更容易飽和；除以 $\sqrt{d_k}=\sqrt2$ 後分佈平緩成 $0.944$。$d_k$ 越大差距越明顯——這裡 $d_k$ 只有 2 就已可見。

$$
E^{(1)}=\frac{S^{(1)}}{\sqrt2}=\begin{bmatrix}1.4142&-1.4142\\-1.4142&1.4142\end{bmatrix}
$$

### 2.4 注意力權重 $A$ 與 context $C$

$$
A^{(1)}=\text{softmax}_\text{row}\!\big(E^{(1)}\big)=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix},\qquad \textstyle\sum_j A^{(1)}_{ij}=1\ \checkmark
$$

$$
C^{(1)}=A^{(1)}V^{(1)},\quad V^{(1)}=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}
$$

第 1 列：$C^{(1)}_1=0.9442\,[1,-1]+0.0558\,[-1,1]=[0.9442-0.0558,\ -0.9442+0.0558]=[0.8884,\,-0.8884]$。

$$
C^{(1)}=\begin{bmatrix}0.8884&-0.8884\\-0.8884&0.8884\end{bmatrix}\in\mathbb{R}^{2\times2}
$$

---

## 3. Head 2、拼接與 $W_O$ 混合

### 3.1 Head 2

取 $\tilde X$ 的後 2 維（注意兩列順序與 Head 1 相反）：

$$
Q^{(2)}=K^{(2)}=V^{(2)}=\tilde X\,W_Q^{(2)}=\begin{bmatrix}-1&1\\1&-1\end{bmatrix}
$$

分數 $S^{(2)}=Q^{(2)}(K^{(2)})^\top=\begin{bmatrix}2&-2\\-2&2\end{bmatrix}$，與 Head 1 **相同**，所以 $A^{(2)}=A^{(1)}$。但 Value 不同：

$$
C^{(2)}=A^{(2)}V^{(2)}=\begin{bmatrix}0.9442&0.0558\\0.0558&0.9442\end{bmatrix}\begin{bmatrix}-1&1\\1&-1\end{bmatrix}=\begin{bmatrix}-0.8884&0.8884\\0.8884&-0.8884\end{bmatrix}
$$

> 「該看誰」相同（$A$ 一樣），「看到什麼」不同（$V$ 不同）→ 輸出不同。這正是 [`03a` §5.6](03a-transformer-architecture.md) 強調的多頭價值。

### 3.2 拼接

$$
\text{Concat}\big(C^{(1)},C^{(2)}\big)=\begin{bmatrix}0.8884&-0.8884&-0.8884&0.8884\\-0.8884&0.8884&0.8884&-0.8884\end{bmatrix}\in\mathbb{R}^{2\times4}
$$

### 3.3 $W_O$ 混合：拼接 ≠ 終點

$O=\text{Concat}\cdot W_O$。依 $W_O$ 的設計（逐欄展開見 §0.5），輸出四維分別是：

$$
O_{:,0}=\tfrac12\big(C^{(1)}_{:,0}+C^{(2)}_{:,0}\big),\quad
O_{:,1}=\tfrac12\big(C^{(1)}_{:,1}+C^{(2)}_{:,1}\big),\quad
O_{:,2}=\tfrac12\big(C^{(1)}_{:,0}-C^{(2)}_{:,0}\big),\quad
O_{:,3}=\tfrac12\big(C^{(1)}_{:,1}-C^{(2)}_{:,1}\big)
$$

第 1 列代入（$C^{(1)}_1=[0.8884,-0.8884]$、$C^{(2)}_1=[-0.8884,0.8884]$）：

- 前兩維（兩頭之**和**）：$\tfrac12(0.8884+(-0.8884))=0$、$\tfrac12(-0.8884+0.8884)=0$
- 後兩維（兩頭之**差**）：$\tfrac12(0.8884-(-0.8884))=0.8884$、$\tfrac12(-0.8884-0.8884)=-0.8884$

$$
O=\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}\in\mathbb{R}^{2\times4}
$$

> **看到 $W_O$ 在做事了。** 本例兩頭恰好方向相反，「和」通道被抵消為 0、「差」通道被放大——若沒有 $W_O$（即拼接後直接輸出），這個跨頭的相消／相長根本不會發生。$W_O$ 讓模型能學習如何融合各頭，呼應 [`03a` §5.7](03a-transformer-architecture.md)。

---

## 4. 第一個殘差連接

Pre-LN 的殘差加在**原始** $X$ 上（不是 $\tilde X$）：

$$
Z'=X+O=\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix}+\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}
=\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}
$$

維度仍是 $T\times d=2\times4$。

---

## 5. 第二子層：LayerNorm → FFN → 第二個殘差

### 5.1 $\tilde Z=\text{LayerNorm}(Z')$

對第 1 列 $z'_1=[1,0,0.8884,0.1116]$：

$$
\mu=\frac{1+0+0.8884+0.1116}{4}=0.5,\qquad
\sigma^2=\frac{0.5^2+0.5^2+0.3884^2+0.3884^2}{4}=0.2004,\qquad \sigma=0.4477
$$

$$
\tilde z_1=\frac{[0.5,\,-0.5,\,0.3884,\,-0.3884]}{0.4477}=[1.1169,\,-1.1169,\,0.8675,\,-0.8675]
$$

$$
\tilde Z=\begin{bmatrix}1.1169&-1.1169&0.8675&-0.8675\\-1.1169&1.1169&-0.8675&0.8675\end{bmatrix}
$$

### 5.2 FFN：$F=\text{ReLU}(\tilde Z W_1)\,W_2$

**Step 1：升維 $\tilde Z W_1$（$2\times4 \cdot 4\times8 = 2\times8$）。** 示範第 1 列、第 1 個輸出（$W_1$ 第 0 行為 $\tfrac12[1,0,1,-1]=[0.5,0,0.5,-0.5]$）：

$$
(\tilde Z W_1)_{1,0}=1.1169(0.5)+(-1.1169)(0)+0.8675(0.5)+(-0.8675)(-0.5)=0.5585+0.4338+0.4338=1.4260
$$

整列結果：

$$
\tilde Z W_1=\begin{bmatrix}
1.4260&-1.5506&-0.9922&0.6831&0.9922&-0.9922&-0.9922&1.4260\\
-1.4260&1.5506&0.9922&-0.6831&-0.9922&0.9922&0.9922&-1.4260
\end{bmatrix}
$$

**Step 2：ReLU（負值歸零）。** 這是 FFN 唯一的非線性來源：

$$
\text{ReLU}(\tilde Z W_1)=\begin{bmatrix}
1.4260&0&0&0.6831&0.9922&0&0&1.4260\\
0&1.5506&0.9922&0&0&0.9922&0.9922&0
\end{bmatrix}
$$

**Step 3：降維 $\cdot\,W_2$（$2\times8 \cdot 8\times4 = 2\times4$）。**

$$
F=\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}
$$

> 若沒有 ReLU，$W_1W_2$ 會塌縮成單一 $4\times4$ 線性映射，中間的 8 維「展開空間」毫無意義（[`03a` §6.4](03a-transformer-architecture.md)）。本例 $d_{ff}=8=2d$ 是為了手算方便；實務通常 $d_{ff}=4d$。

### 5.3 第二個殘差 → Block 輸出

$$
Y=Z'+F=\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}+\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}
$$

$$
\boxed{\,Y=\begin{bmatrix}0.6585&0.4961&2.6559&-0.1053\\0.9922&0.7831&-1.1598&1.8806\end{bmatrix}\in\mathbb{R}^{2\times4}\,}
$$

輸出 shape 與輸入 $X$ 相同（$2\times4$），因此可以直接把 $Y$ 餵進下一個 Block——這就是 $N$ 層 Transformer 能堆疊的原因（[`03a` §6.1](03a-transformer-architecture.md)）。

---

## 6. Positional Encoding 數值與相對位置驗證

### 6.0 為什麼位置編碼放在 Block 計算之後說明？

在真正的 Transformer 中，位置資訊會在進入第一個 Block 之前加到 token embedding 上：

$$
X_{\text{in}} = X_{\text{token}} + P
$$

接著才把 $X_{\text{in}}$ 丟進 Pre-LN Block。本文前面 §1–§5 為了維持與 03b1／03b2 **同一筆資料完全銜接**，故意先不把 $P$ 加進 $X$，否則所有 attention、residual、FFN 的數字都會改掉，前兩份文件就無法直接接上。

因此，§6 的角色不是重新改寫前面的 Block 輸入，而是補上 [`03a`](03a-transformer-architecture.md) §7 的位置概念：說明若要讓模型知道 token 的順序，$P$ 長什麼樣（§6.1）、$P$ 如何相加擴充 token 向量而不弄混語意（§6.2 用真實數字走一遍），以及「相對位置可以用旋轉表示」這件事如何用數字驗證（§6.3）。換句話說，前面 §1–§5 是 **Block 內部怎麼算**；本節是 **進 Block 前，序列順序如何被寫進向量**。

### 6.1 編碼表（$d=4$，位置 $0\sim3$）

維度索引 $m\in\{0,1\}$，角頻率 $\omega_m=1/10000^{2m/d}$：$\omega_0=1$、$\omega_1=1/10000^{0.5}=0.01$。

依 $p_{i,2m}=\sin(\omega_m i)$、$p_{i,2m+1}=\cos(\omega_m i)$（[`03a` §7.2](03a-transformer-architecture.md)）：

| 位置 $i$ | $p_{i,0}=\sin(i)$ | $p_{i,1}=\cos(i)$ | $p_{i,2}=\sin(0.01i)$ | $p_{i,3}=\cos(0.01i)$ |
|---|---|---|---|---|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.0100 | 1.0000 |
| 2 | 0.9093 | −0.4161 | 0.0200 | 0.9998 |
| 3 | 0.1411 | −0.9900 | 0.0300 | 0.9996 |

低維（$m=0$）隨位置快速振盪（捕捉局部），高維（$m=1$）變化極慢（捕捉全局）。

### 6.2 相加融合：位置向量如何擴充 token 向量

位置編碼不是 Block 內部的新子層，而是**進第一個 Block 之前**，對 token 向量做一次**逐元素相加**（[`03a` §7.1](03a-transformer-architecture.md)）：token embedding 帶的是**語意**，位置編碼帶的是**順序**，兩者維度都是 $d$，直接相加成 $X_{\text{in}}=X+P$（§6.0 的式子）。本節就把這一步用本文的真實數字走一遍。

本例序列長度 $T=2$，故只用到位置 $0$、$1$ 兩列（取自 §6.1 表）：

$$
P_0=[\,0,\ 1,\ 0,\ 1\,],\qquad
P_1=[\,0.8415,\ 0.5403,\ 0.0100,\ 1.0000\,]
$$

把 §0.2 的 $X=\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix}$ 逐列加上對應位置向量：

$$
\begin{aligned}
\text{token 1 @ pos 0}:\quad &[1,0,0,1]+[0,1,0,1]=[\,1,\ 1,\ 0,\ 2\,]\\
\text{token 2 @ pos 1}:\quad &[0,1,1,0]+[0.8415,0.5403,0.0100,1.0000]=[\,0.8415,\ 1.5403,\ 1.0100,\ 1.0000\,]
\end{aligned}
$$

$$
\boxed{\,X_{\text{in}}=X+P=\begin{bmatrix}1&1&0&2\\0.8415&1.5403&1.0100&1.0000\end{bmatrix}\in\mathbb{R}^{2\times4}\,}
$$

這就是「位置資訊如何擴充原來 token 向量」的具體樣貌：原本只帶語意的 $X$，每一列都被加進一段隨位置變化的量，**同一個 token 放在不同位置就會得到不同的輸入向量**。

**為什麼相加不會把語意和位置弄混？** 直覺上 $3+5=8$ 會遺失成分，但在高維空間中，不同資訊可以佔據**不同方向（子空間）**而互不干擾。先看最乾淨的 2D 例子：若語意只寫在第 0 維、位置只寫在第 1 維，則

$$
\underbrace{[9,0]}_{\text{語意「貓」}}+\underbrace{[0,1]}_{\text{位置 1}}=[9,1],
$$

下游只要「看第 0 維」就取回語意 $9$、「看第 1 維」就取回位置 $1$，毫無混淆。本例的 $X$ 正好是這種結構的乾淨版：§0.2 特意讓兩個 token 佔**互不重疊**的維度（token 1 在第 $0,3$ 維、token 2 在第 $1,2$ 維），位置向量疊加上去後，下游的線性投影（Attention 的 $Q/K/V$ 投影矩陣 $W$、FFN 的 $W_1$）就能把語意分量與位置分量各自線性抽取出來。

**但要提醒三件事，別把教學簡化當成真實架構：**

1. **維度切分是人為簡化。** 上面「某維專管語意、某維專管位置」只為講解方便；真實模型的語意與位置是**混合、分散**在整個 $d$ 維空間、由訓練學出來的分布，沒有人指定哪一維做什麼。
2. **關鍵不是「正交」，而是「可學習的萃取」。** Sinusoidal PE 是固定公式算出來的，不是隨機向量，不能直接套用「高維隨機向量彼此近似正交」的統計性質，也沒有嚴謹證明保證它與 token embedding 正交。真正讓相加後仍可分離的原因是：$Q/K/V$ 投影矩陣會在訓練中**學會**如何從混合向量萃取各自需要的成分——只要兩者不完全共線（不需嚴格正交），線性代數上就有分離空間，模型再靠大量參數與資料把這個分離能力學出來。
3. **現代模型未必用「輸入端相加」。** 輸入端相加是**絕對位置編碼**（原始 Transformer、BERT、GPT-2）的做法；LLaMA 系列等多數現代模型改用 **RoPE**、部分用 **ALiBi**，把位置資訊放在 Attention 計算 $Q/K$ 時融入，而非在輸入端與語意向量相加（見 [`06` §3](06-modern-transformer-variants.md)）。

> **注意：本節的 $X_{\text{in}}$ 只是示範「相加這一步」，不會回饋進前面 §1–§5 的計算。** 如 §6.0 所說，本文 §1–§5 刻意讓 $P=0$（即用原始 $X$），才能與 03b1／03b2 的數字完全銜接；一旦把 $X_{\text{in}}$ 當作 Block 輸入，後面所有 attention、殘差、FFN 的數字都會改掉。此處相加後的 $X_{\text{in}}$ 可在 NB1 §13 一鍵重現（見 §7）。

### 6.3 「相對位置 = 旋轉」數值驗證

[`03a` §7.3](03a-transformer-architecture.md) 證明：位置平移 $\delta$ 等價於對每個 2D 子平面旋轉 $\omega_m\delta$。取 $m=0$（$\omega_0=1$）、$i=1$、$\delta=2$，旋轉矩陣

$$
R=\begin{bmatrix}\cos(\omega_0\delta)&\sin(\omega_0\delta)\\-\sin(\omega_0\delta)&\cos(\omega_0\delta)\end{bmatrix}
=\begin{bmatrix}\cos2&\sin2\\-\sin2&\cos2\end{bmatrix}
=\begin{bmatrix}-0.4161&0.9093\\-0.9093&-0.4161\end{bmatrix}
$$

驗證 $\begin{bmatrix}p_{3,0}\\p_{3,1}\end{bmatrix}\overset{?}{=}R\begin{bmatrix}p_{1,0}\\p_{1,1}\end{bmatrix}$：

$$
R\begin{bmatrix}0.8415\\0.5403\end{bmatrix}
=\begin{bmatrix}-0.4161(0.8415)+0.9093(0.5403)\\-0.9093(0.8415)-0.4161(0.5403)\end{bmatrix}
=\begin{bmatrix}0.1411\\-0.9900\end{bmatrix}
$$

與表中 $i=3$ 的 $[p_{3,0},p_{3,1}]=[0.1411,-0.9900]$ **完全吻合** ✓。模型因此能用固定的線性變換表達「相隔 $\delta$ 格」，無論 $i$ 在哪裡——這正是 RoPE 的前身（[`06` §3](06-modern-transformer-variants.md)）。

---

## 7. 用 NB1 重現本文數字

本文每個數字都可在 [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) §13「與 03a 對照：逐步數值驗證」一鍵跑出。該 cell 是**獨立自足**的 NumPy 程式（不依賴 NB1 其他類別），直接硬編碼本文的 $X$ 與所有權重，逐步印出 $\tilde X, S, E, A, C, O, Z', \tilde Z, F, Y$、PE 表與位置擴充後的 $X_{\text{in}}=X+P$，方便你對著本文逐格核對。

---

## 8. 全流程總表與檢核

### 8.1 從 03b1 到 03b3：同一筆資料的擴充路徑

把三份文件放在同一張圖上，主線分成 **① 進 Block 前的位置注入** 與 **② Block 內的前向** 兩段。

**① 進 Block 前（§6）：** 先把位置向量 $P$ 逐列加到 token embedding 上，才送進 Block：

$$
X_{\text{token}} \xrightarrow{\;+\,P\;} X_{\text{in}}
$$

本文為與 03b1／03b2 的數字完全銜接，手算時取 $P=0$（故 $X_{\text{in}}=X$，下面直接用 $X$）；把 $P$ 真正加進去、看位置如何擴充 token 向量的數值，見 §6.2。

**② Block 內的前向：**

$$
X
\xrightarrow{\text{LN}_1}
\tilde X
\xrightarrow{\text{Head 1}}
C^{(1)}
\xrightarrow{+\text{Head 2}}
C^{(1)},C^{(2)}
\xrightarrow{\text{Concat}+W_O}
O
\xrightarrow{+X}
Z'
\xrightarrow{\text{LN}_2+\text{FFN}}
F
\xrightarrow{+Z'}
Y
$$

| 階段 | 輸入 | 主要運算 | 輸出 | 對應文件 |
|---|---:|---|---:|---|
| 位置注入（進 Block 前）| $X_{\text{token}}, P$ | sin/cos 多頻率編碼 $P$，逐列相加 $X_{\text{in}}=X_{\text{token}}+P$（本例取 $P=0$）；另附旋轉性質驗證 | $X_{\text{in}}$ | 本文 §6.1–§6.3 |
| Pre-LN | $X$ | 每列減均值、除標準差 | $\tilde X$ | 03b1／本文 §1 |
| Head 1 | $\tilde X$ | $QK^\top/\sqrt{d_k} \to A \to AV$ | $C^{(1)}$ | 03b1／本文 §2 |
| Head 2 | $\tilde X$ | 同一套 attention 流程，但取後 2 維 | $C^{(2)}$ | 03b2／本文 §3.1 |
| MHA 輸出 | $C^{(1)},C^{(2)}$ | Concat 後乘 $W_O$ | $O$ | 03b2／本文 §3.2–§3.3 |
| 第一殘差 | $X,O$ | $Z'=X+O$ | $Z'$ | 03b2／本文 §4 |
| FFN 子層 | $Z'$ | $\text{LN}_2 \to W_1 \to \text{ReLU} \to W_2$ | $F$ | 03b2／本文 §5.1–§5.2 |
| 第二殘差 | $Z',F$ | $Y=Z'+F$ | $Y$ | 03b2／本文 §5.3 |

這張表也說明了為什麼 03b3 比 03b2 多：03b2 的目標是算到 $Y$；03b3 則多回答「這些矩陣為什麼這樣設計」「縮放到底差多少」「位置資訊如何進入模型」「如何用 NB1 重現」。

### 8.2 最終數值總表

| 符號 | Shape | 數值 | 角色 |
|---|---:|---|---|
| $X$ | $2\times4$ | $\begin{bmatrix}1&0&0&1\\0&1&1&0\end{bmatrix}$ | 原始輸入 token 表示 |
| $\tilde X$ | $2\times4$ | $\begin{bmatrix}1&-1&-1&1\\-1&1&1&-1\end{bmatrix}$ | 第一個 LayerNorm 後輸入 |
| $C^{(1)}$ | $2\times2$ | $\begin{bmatrix}0.8884&-0.8884\\-0.8884&0.8884\end{bmatrix}$ | Head 1 context |
| $C^{(2)}$ | $2\times2$ | $\begin{bmatrix}-0.8884&0.8884\\0.8884&-0.8884\end{bmatrix}$ | Head 2 context |
| $O$ | $2\times4$ | $\begin{bmatrix}0&0&0.8884&-0.8884\\0&0&-0.8884&0.8884\end{bmatrix}$ | Multi-Head Attention 輸出 |
| $Z'$ | $2\times4$ | $\begin{bmatrix}1&0&0.8884&0.1116\\0&1&0.1116&0.8884\end{bmatrix}$ | 第一個殘差後結果 |
| $\tilde Z$ | $2\times4$ | $\begin{bmatrix}1.1169&-1.1169&0.8675&-0.8675\\-1.1169&1.1169&-0.8675&0.8675\end{bmatrix}$ | 第二個 LayerNorm 後輸入 |
| $F$ | $2\times4$ | $\begin{bmatrix}-0.3415&0.4961&1.7675&-0.2169\\0.9922&-0.2169&-1.2714&0.9922\end{bmatrix}$ | FFN 輸出修正量 |
| $Y$ | $2\times4$ | $\begin{bmatrix}0.6585&0.4961&2.6559&-0.1053\\0.9922&0.7831&-1.1598&1.8806\end{bmatrix}$ | 完整 Block 輸出 |

最重要的檢核點是：$X$、$O$、$Z'$、$F$、$Y$ 都是 $2\times4$。這代表 attention 子層與 FFN 子層雖然內部做了切分、拼接、升維與降維，但最後都回到同一個 model dimension $d=4$。這就是 Transformer Block 可以一層接一層堆疊的形狀契約。

### 8.3 讀完本文應該建立的四個觀念

第一，**LayerNorm 改變的是數值分佈，不改變 shape**。本文的 Pre-LN 先把 $X$ 正規化為 $\tilde X$，再送進 attention；這也是本文和 03a §5.6 直接用 $X$ 算 MHA 的主要差異。

第二，**Multi-Head 不是把句子切段，而是把 feature 維度切成多個子空間**。本例 $d=4,H=2$，所以 Head 1 取前 2 維，Head 2 取後 2 維。真實模型的投影通常是稠密矩陣，但概念仍是把不同 head 放在不同子空間中學不同關係。

第三，**Residual 的核心是「保留原表示，再加上子層修正量」**。第一個殘差是 $Z'=X+O$，第二個殘差是 $Y=Z'+F$。因此 Block 不需要每一層都重寫表示，而是逐層累積修正。

第四，**Positional Encoding 是進入 Block 前的順序資訊，不是 Block 內部的新子層**。本文將它放在 §6 獨立說明，是為了維持 §1–§5 與 03b1／03b2 的數值完全銜接，同時補上 Transformer 為何能辨識順序的關鍵。

---

## 下一步

- **回主線理論：** [`04-gpt-decoder-only.md`](04-gpt-decoder-only.md) — 在本文的 attention 之上加 Causal Masking，走向 GPT。
- **往實作走：** [`../notebooks/NB1-simple-llm-vanilla.ipynb`](../notebooks/NB1-simple-llm-vanilla.ipynb) — 用 NumPy 從零實作；§13 即本文的可執行版。
- **往反向傳播走：** [`05-backpropagation.md`](05-backpropagation.md) — 有了前向數字，接著手推每個元件的梯度。
