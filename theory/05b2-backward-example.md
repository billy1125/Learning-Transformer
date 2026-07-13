# 05b2｜向後傳播數值範例：一次反向逐階段實算

> **適合對象：** 讀完 [`05b1-backward-propagation.md`](05b1-backward-propagation.md)（反向符號數學）並看過 [`05a2-forward-example.md`](05a2-forward-example.md)（前向數值）的讀者。
>
> **本文做什麼：** **沿用 [`05a2`](05a2-forward-example.md) 的同一組數字**（$T=2$、$d=3$、單頭、含因果遮罩），從 $\partial L/\partial\text{logits}$ 一路反向算到 $\partial L/\partial E$，**每個階段都給出實際梯度**，並對照 [`05b1`](05b1-backward-propagation.md) 的符號公式。反向的順序正好是前向的鏡像。
>
> **前置文件：** [`05b1-backward-propagation.md`](05b1-backward-propagation.md)、[`05a2-forward-example.md`](05a2-forward-example.md)

> **數字精度：** 四捨五入到小數 3 位（末位可能有進位誤差）；全部以 PyTorch autograd 交叉驗證。前向的所有中間量（$A,C,x_1,\text{LN}_2,x_2,h,p$…）直接取自 [`05a2`](05a2-forward-example.md)。

---

## 0. 前向數字回顧（取自 [`05a2`](05a2-forward-example.md)）

反向會用到這些前向量：

$$
A=\begin{bmatrix}1&0\\0.296&0.704\end{bmatrix},\quad
\text{LN}_1=\begin{bmatrix}1.225&0&-1.225\\1.225&-1.225&0\end{bmatrix},\quad
p=\begin{bmatrix}0.777&0.153&0.070\\0.805&0.090&0.105\end{bmatrix}
$$

目標 $y=(0,1)$，序列長度 $T=2$。以下每個 $\partial L/\partial(\cdot)$ 簡記為 $G^{(\cdot)}$。

---

## 1. Cross-Entropy + Softmax（對應 [`05b1`](05b1-backward-propagation.md) 總覽 Step 1）

合併梯度 $G^{\text{logit}}_i=\tfrac1T\big(p^{(i)}-\text{onehot}(y_i)\big)$：

- 位置 0（$y_0=0$）：$\tfrac12([0.777,0.153,0.070]-[1,0,0])=\tfrac12[-0.223,0.153,0.070]=[-0.112,0.076,0.035]$
- 位置 1（$y_1=1$）：$\tfrac12([0.805,0.090,0.105]-[0,1,0])=\tfrac12[0.805,-0.910,0.105]=[0.403,-0.455,0.052]$

$$
G^{\text{logit}} = \begin{bmatrix} -0.112 & 0.076 & 0.035 \\ 0.403 & -0.455 & 0.052 \end{bmatrix}
$$

「正確類別減、其餘加」——位置 1 正確類別 $-0.455$ 特別大，因為它前向幾乎猜錯。

---

## 2. lm_head 反向（對應 [`05b1`](05b1-backward-propagation.md) 總覽 Step 2）

$\text{logits}=hW_{lm}^\top$，$W_{lm}=I$，所以梯度直接穿過：

$$
G^{h} = G^{\text{logit}} W_{lm} = \begin{bmatrix} -0.112 & 0.076 & 0.035 \\ 0.403 & -0.455 & 0.052 \end{bmatrix}
$$

參數梯度 $G^{W_{lm}}=\sum_i (G^{\text{logit}}_i)^\top h_i$：

$$
G^{W_{lm}} = \begin{bmatrix} 0.419 & -0.284 & -0.135 \\ -0.540 & 0.335 & 0.205 \\ 0.121 & -0.051 & -0.070 \end{bmatrix}
$$

---

## 3. 最終 LayerNorm 反向（對應 [`05b1`](05b1-backward-propagation.md) §5）

把 $G^h$ 代進 LayerNorm 的三路徑閉式公式（[`05b1`](05b1-backward-propagation.md) §5.7），用前向的 $\mu_f=[1.816,2.246]$、$\sigma_f=[2.875,2.561]$。結果：

$$
G^{x_2} = \begin{bmatrix} -0.006 & 0.020 & -0.013 \\ -0.007 & -0.087 & 0.093 \end{bmatrix}
$$

> 數值變小是 LayerNorm 的正常現象：它會把「整體平移」與「整體縮放」兩個方向的梯度分量濾掉（$d=3$ 時仍留下 1 個自由度，不像 $d=2$ 會全歸零）。

---

## 4. 殘差② + FFN + LayerNorm② 反向（對應 [`05b1`](05b1-backward-propagation.md) §5、§2）

**殘差② $x_2=x_1+\text{FFN}$：** 梯度同時流向兩條路，$G^{\text{FFN}}=G^{x_2}$，同時 $x_1$ 先拿到一份 $G^{x_2}$。

**FFN 反向**（$\text{FFN}=\text{ReLU}(z)W_2$，$z=\text{LN}_2 W_1$）：

$$
G^{\text{ReLU}} = G^{\text{FFN}} W_2^\top,\qquad
G^{z} = G^{\text{ReLU}} \odot \mathbb{1}[z>0]
$$

用前向的 $z$ 遮罩（$z>0$ 的位置才通過），得

$$
G^{z} = \begin{bmatrix} -0.006 & 0 & 0 & 0 & 0 & 0 \\ -0.007 & 0 & 0 & -0.007 & -0.087 & 0 \end{bmatrix}
$$

再往回 $G^{\text{LN}_2}=G^{z}W_1^\top$：

$$
G^{\text{LN}_2} = \begin{bmatrix} -0.006 & 0 & 0.006 \\ -0.093 & 0.007 & -0.080 \end{bmatrix}
$$

**LN② 反向**（[`05b1`](05b1-backward-propagation.md) §5，用 $\mu_2=[1,1]$、$\sigma_2=[1.817,1.688]$）把 $G^{\text{LN}_2}$ 換成對 $x_1$ 的梯度，再**加上**殘差直通的那份 $G^{x_2}$，合計：

$$
G^{x_1} = \begin{bmatrix} -0.006 & 0.020 & -0.013 \\ 0.001 & -0.075 & 0.074 \end{bmatrix}
$$

---

## 5. 殘差① + Attention 反向（對應 [`05b1`](05b1-backward-propagation.md) §1）

**殘差① $x_1=x_0+C$：** 一份給 $C$、一份直通 $x_0$。所以進 attention 的上游梯度是

$$
G^{C} = G^{x_1} = \begin{bmatrix} -0.006 & 0.020 & -0.013 \\ 0.001 & -0.075 & 0.074 \end{bmatrix}
$$

> **與 [`05b1`](05b1-backward-propagation.md) 舊版孤立範例的差別：** 那裡為了單獨練 attention，假設上游梯度 $G^C=I$；這裡的 $G^C$ 是**真正從 CE 一路傳下來的值**，所以數字不同，但用的公式一模一樣。

**對 V：** $G^{V}=A^\top G^{C}$：

$$
G^{V} = \begin{bmatrix} -0.006 & -0.002 & 0.009 \\ 0.001 & -0.053 & 0.052 \end{bmatrix}
$$

**對 A：** $G^{A}=G^{C}V^\top$（$V=\text{LN}_1$）：

$$
G^{A} = \begin{bmatrix} 0.009 & -0.032 \\ -0.090 & 0.092 \end{bmatrix}
$$

**Softmax 反向**（[`05b1`](05b1-backward-propagation.md) §1.4，逐列 $G^{S}_i=A_i\odot(G^A_i-s_i)$，$s_i=\langle A_i,G^A_i\rangle$）：

- 位置 0：$s_0 = 1\cdot0.009 + 0\cdot(-0.032)=0.009$，$G^S_0=[1,0]\odot([0.009,-0.032]-0.009)=[0,\ 0]$（只看自己，梯度不外流）。
- 位置 1：$s_1 = 0.296\cdot(-0.090)+0.704\cdot0.092=0.038$，$G^S_1=[0.296,0.704]\odot([-0.090,0.092]-0.038)=[-0.038,\ 0.038]$。

$$
G^{S} = \begin{bmatrix} 0 & 0 \\ -0.038 & 0.038 \end{bmatrix}
$$

**對 Q、K**（[`05b1`](05b1-backward-propagation.md) §1.5，$G^{Q}=\tfrac{1}{\sqrt3}G^{S}K$、$G^{K}=\tfrac{1}{\sqrt3}(G^{S})^\top Q$，$Q=K=\text{LN}_1$）：

$$
G^{Q} = \begin{bmatrix} 0 & 0 & 0 \\ 0 & -0.027 & 0.027 \end{bmatrix},\qquad
G^{K} = \begin{bmatrix} -0.027 & 0.027 & 0 \\ 0.027 & -0.027 & 0 \end{bmatrix}
$$

**合流到 LN₁ 輸出：** 因為 $Q=K=V=\text{LN}_1$（投影都是 $I$），三份梯度相加：

$$
G^{\text{LN}_1} = G^{V}+G^{Q}+G^{K}
= \begin{bmatrix} -0.033 & 0.024 & 0.009 \\ 0.027 & -0.106 & 0.079 \end{bmatrix}
$$

---

## 6. LayerNorm① 反向 → 輸入梯度（對應 [`05b1`](05b1-backward-propagation.md) §5）

把 $G^{\text{LN}_1}$ 代進 LN 反向公式（$\mu_1=[1,1]$、$\sigma_1=0.816$），得到 attention 分支對 $x_0$ 的梯度；再**加上**殘差① 直通的那份 $G^{x_1}$，合計：

$$
G^{x_0} = \begin{bmatrix} -0.021 & 0.050 & -0.028 \\ -0.048 & -0.123 & 0.171 \end{bmatrix}
$$

---

## 7. Embedding + 位置編碼 反向（對應 [`05b1`](05b1-backward-propagation.md) §6 / 總覽 Step 4）

$x_0=E[t]+P$，所以梯度直接**分配**回被查到的那幾列：

$$
G^{E[0]} = G^{x_0}_0 = [-0.021,\ 0.050,\ -0.028],\qquad
G^{E[1]} = G^{x_0}_1 = [-0.048,\ -0.123,\ 0.171]
$$

$$
G^{E[2]} = 0 \quad(\text{token 2 沒被用到，這一列本步不更新——稀疏更新})
$$

$$
G^{P[0]} = G^{x_0}_0,\qquad G^{P[1]} = G^{x_0}_1
$$

到此，梯度已從 loss 一路流回 Embedding 矩陣的對應列。optimizer 再依這些梯度更新參數（[`05b1`](05b1-backward-propagation.md) 總覽 Step 5），一個 training step 就完成。

---

## 反向各階段梯度一覽

| 階段 | 梯度 | 對應 05b1 |
|---|---|---|
| CE+Softmax | $G^{\text{logit}}=[[-0.112,0.076,0.035],[0.403,-0.455,0.052]]$ | Step 1 |
| lm_head | $G^{h}=G^{\text{logit}}$ | Step 2 |
| 最終 LN | $G^{x_2}=[[-0.006,0.020,-0.013],[-0.007,-0.087,0.093]]$ | §5 |
| FFN/LN② | $G^{x_1}=[[-0.006,0.020,-0.013],[0.001,-0.075,0.074]]$ | §5、§2 |
| Attention | $G^{\text{LN}_1}=[[-0.033,0.024,0.009],[0.027,-0.106,0.079]]$ | §1 |
| LN① | $G^{x_0}=[[-0.021,0.050,-0.028],[-0.048,-0.123,0.171]]$ | §5 |
| Embedding | $G^{E[0]},G^{E[1]}=G^{x_0}_0,G^{x_0}_1$；$G^{E[2]}=0$ | §6 |

每個數字都可用 [`05b1`](05b1-backward-propagation.md) 的符號公式獨立驗算，也與 PyTorch `loss.backward()` 的輸出一致。

---

## 下一步

**回顧符號推導：** → [`05b1-backward-propagation.md`](05b1-backward-propagation.md)（每條公式的完整證明）
**對照程式：** → [`04b-nanogpt-walkthrough.md`](04b-nanogpt-walkthrough.md) → [`../notebooks/NB3-llm-backpropagation.ipynb`](../notebooks/NB3-llm-backpropagation.ipynb)（NumPy 手刻反向並以數值梯度驗證）
