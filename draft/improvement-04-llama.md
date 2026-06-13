# 學習路徑改善計劃 04：把 06 的「下一步」做成可執行的實作出口

> 撰寫日期：2026-06-13
> 前提：`improvement-00-fixes.md`（P1–P4）、`improvement-01-mainline-gaps.md`（A1–A12、B1–B5）、`improvement-02-writing.md`（W1–W18）、`improvement-03-notebooks.md`（N1–N9）全部完成。
> 本計劃聚焦：把 `theory/06-modern-transformer-variants.md` 文末「下一步」的兩條建議，從**口頭建議**變成**可執行、可驗證的學習資產**。

---

## 一、問題定位：06 是「只讀不寫」的出口

`theory/06` 把 nanoGPT → LLaMA 的七個差異講得很完整（RMSNorm / SwiGLU / RoPE / GQA / Flash Attention），但它的結尾「下一步」給的是兩條**沒有對應產出物**的建議：

> - 打開 LLaMA 官方實作（約 500 行），對照本文逐節辨認 `RMSNorm`、`apply_rotary_emb`、`repeat_kv`、`gate_proj/up_proj/down_proj`
> - 回到 `NB4-nanoGPT.ipynb`，動手把 LayerNorm 換成 RMSNorm、ReLU FFN 換成 SwiGLU，觀察訓練曲線的差異

兩條建議都很好，但讀者實際執行時會卡住：

1. **「對照辨認」缺地圖**：官方 `model.py` 用了 `fairscale` 的張量並行層（`ColumnParallelLinear`、`RowParallelLinear`）、`complex64` 的 RoPE 實作、KV cache 等等，和 06 的教學寫法差距不小。沒有逐行對照表，讀者很容易在工程細節裡迷路。
2. **「動手改 NB4」缺骨架**：把 LayerNorm 換 RMSNorm 還算單純，但 06 同時建議換 SwiGLU——而 SwiGLU 改變了 FFN 的參數量與 shape（需要把 $d_{ff}=4d$ 調成 $\frac{2}{3}\cdot 4d$ 才能維持參數量持平），若沒有可參照的正確實作與 A/B 對照協定，讀者改完看到 loss 變化也不知道是「真的有差」還是「自己改錯了」。

本計劃的核心主張：**主線的終點不該停在「讀懂理論」，而該停在「親手把 nanoGPT 改造成 mini-LLaMA 並驗證」**。為此新增一個 Notebook（NB5）與一份程式碼對照文件（07），讓 06 的「下一步」真正落地。

---

## 二、改善項目（依優先順序）

---

### L1　新增 `NB5-nanoGPT-to-llama.ipynb`：把 NB4 改造成 mini-LLaMA（P1，核心）

**目標：** 以 NB4 訓練好的 nanoGPT 為基準，逐元件替換成 LLaMA 風格，每換一個元件就跑一次短訓練，畫出 A/B 損失曲線。讓讀者「邊改邊看效果」，把 06 的五節理論變成五段可執行的 diff。

**前置理論：** 01–04 ＋ 06（NB5 是 06 的實作對應，正如 NB4 是 04 的實作對應）。

**建議 cell 結構：**

| 段落 | 內容 | 對應 06 章節 |
|---|---|---|
| §0 | 導覽：本 notebook 從 NB4 的 baseline 出發，逐步替換五個元件 | 06 §0 |
| §1 | 載入 NB4 的 baseline 模型與 config（`n_embd=384, n_head=6, n_layer=6`，字元級莎士比亞），確立對照基準 loss | — |
| §2 | **RMSNorm**：實作 `RMSNorm` class，替換 `nn.LayerNorm`，跑短訓練比較 | 06 §1 |
| §3 | **SwiGLU**：實作 `SwiGLU` FFN（`gate_proj/up_proj/down_proj`），將 $d_{ff}$ 調為 $\frac{2}{3}\cdot 4d$ 維持參數量，比較 | 06 §2 |
| §4 | **RoPE**：實作 `precompute_rope` 與 `apply_rotary_emb`，移除 learned PE，比較 | 06 §3 |
| §5 | **GQA**：把 MHA 改為 `n_kv_heads < n_head`（如 6 query head、2 KV head），含 `repeat_kv`，比較 KV cache 大小與 loss | 06 §4 |
| §6 | **Flash Attention**：把樸素 attention 換成 `F.scaled_dot_product_attention`，驗證輸出在數值誤差內相同、計時加速 | 06 §5 |
| §7 | 整合：把 §2–§6 全部接上，得到一個 mini-LLaMA，與 baseline 做最終對照表（參數量、loss、生成樣本） | 06 §6 |

**驗證點：**

- [ ] 每個元件替換後模型仍能訓練（loss 下降，不發散）
- [ ] §3 SwiGLU 調整後總參數量與 baseline 持平（誤差 < 5%），證明「換 SwiGLU ≠ 偷加參數」
- [ ] §4 RoPE 版本移除了 `position_embedding_table`，但 context 內位置資訊仍有效（loss 不劣於 learned PE）
- [ ] §6 Flash Attention 與樸素實作在相同輸入下輸出 `allclose`（atol≈1e-4）
- [ ] §7 最終 mini-LLaMA 的生成樣本仍是「莎士比亞風格」可讀文字

**和現有 notebook 的一致性約束（沿用 CLAUDE.md 行文品質原則）：**

- 路徑隔離沿用 N3 的修法：所有輸出寫到 `notebooks/data/`，不污染根目錄
- 字元級 tokenizer、莎士比亞資料集與 NB4 完全相同，確保 A/B 對照只有「架構」這一個變因
- 短訓練步數（如 200–500 steps）以求 CPU 可在數分鐘內跑完；說明清楚「這是趨勢觀察，不是最終收斂」

**優先級：P1 | 難度：高 | 檔案：新增 `notebooks/NB5-nanoGPT-to-llama.ipynb`**

---

### L2　新增 `theory/07-reading-llama-source.md`：官方 model.py 逐節對照地圖（P1）

**目標：** 把 06「下一步」第一條（對照官方原始碼）做成一份可跟著走的對照文件。讀者一邊開 `meta-llama/llama/llama/model.py`，一邊用本文把每段官方程式碼對回 06 學過的概念。

**建議章節：**

1. **閱讀前的心理準備**：官方碼為何看起來比 06 複雜——`fairscale` 張量並行層（`ColumnParallelLinear` / `RowParallelLinear`）本質就是 `nn.Linear`，只是切到多 GPU；先在腦中把它們還原成普通 Linear。
2. **逐節對照表**：

   | 官方符號 | 06 章節 | 教學等價物 |
   |---|---|---|
   | `class RMSNorm` | §1 | NB5 §2 的 `RMSNorm` |
   | `precompute_freqs_cis` / `apply_rotary_emb` | §3 | NB5 §4（官方用 `complex64`，NB5 用實數對旋轉，數學等價——需附證明片段）|
   | `repeat_kv` / `n_kv_heads` | §4 | NB5 §5 的 `repeat_kv` |
   | `class FeedForward`（`w1/w2/w3`）| §2 | NB5 §3 的 `gate/up/down_proj`（標明 `w1=gate, w3=up, w2=down`）|
   | `class Attention`（含 KV cache）| 04 §6 + 06 §4 | NB4/NB5 attention |
   | `class Transformer` 主迴圈 | 04 | NB4 `GPTModel` |

3. **三個容易誤讀的點**（每點附最短說明，遵守「斷言必須附理由」）：
   - 官方 RoPE 用複數乘法，為何等價於 06 §3 的 2×2 旋轉矩陣（補一段 $e^{i\theta}$ 與旋轉矩陣的對應）
   - `w1/w2/w3` 的命名與 SwiGLU 公式 $\text{down}(\text{SiLU}(\text{gate}(x)) \odot \text{up}(x))$ 的對應，不要被編號順序誤導
   - KV cache 的 `cache_k/cache_v` 為何只在推論用、訓練時不需要（對接 04 已介紹的 KV Cache）

4. **延伸**：和 Mistral（sliding window attention）、Qwen（QK-Norm）等的差異一句話帶過，標明屬本倉庫範圍外。

**引用有效性檢查（CLAUDE.md 要求）：** 官方 `model.py` 連結需釘選到具體 commit/tag（避免 main 漂移後行號失效），或只引用 class/函式名而不引行號。

**優先級：P1 | 難度：中 | 檔案：新增 `theory/07-reading-llama-source.md`**

---

### L3　更新 06 文末「下一步」與全倉導覽，接上 NB5 / 07（P2）

**問題：** L1、L2 完成後，06 的「下一步」、`CLAUDE.md` 的資料夾表、`README.md` 的學習路徑都還指向舊狀態（06 是終點）。需要把新出口接上。

**修改清單：**

- `theory/06` §下一步：把兩條口頭建議改寫成「跟著 `NB5` 動手改造」「搭配 `theory/07` 對照官方碼」，並標示 NB5/07 為 06 的實作與精讀延伸。
- `CLAUDE.md`：
  - 理論文件表新增 `07-reading-llama-source.md`
  - Notebook 表新增 `NB5-nanoGPT-to-llama.ipynb`（前置：01–04 ＋ 06）
  - 「核心設計原則」補一句：06/07/NB5 構成「當代架構」的理論—精讀—實作三件組
- `README.md`：學習路徑圖補上 06 → NB5 → 07 的選讀分支。

**優先級：P2 | 難度：低 | 檔案：`theory/06`、`CLAUDE.md`、`README.md`**

---

### L4　`.gitignore` 與環境檢查：NB5 不引入新重依賴（P2）

**問題：** NB5 應該只靠 NB4 既有的依賴（PyTorch 2.x，`F.scaled_dot_product_attention` 內建 Flash Attention 後端），**不要**引入 `fairscale`、`flash-attn`（需編譯 CUDA）等重依賴，否則違背「字元級、CPU 可跑」的倉庫定位。

**修改清單：**

- 確認 NB5 的 Flash Attention 段落用 `torch.nn.functional.scaled_dot_product_attention`，不裝 `flash-attn` 套件；並加註說明 CPU 上 SDPA 會走 memory-efficient / math 後端，加速效果在 GPU 才明顯。
- `.gitignore` 沿用 N9，確認 NB5 的 checkpoint 輸出（如 `notebooks/data/mini_llama_*.pt`）被忽略。
- 不修改 `environment/*.yml`（無新依賴）。

**優先級：P2 | 難度：低 | 檔案：`NB5`、`.gitignore`**

---

### L5　RoPE 實作的數學橋接片段補強（P3）

**問題：** 06 §3 從直覺與 2×2 旋轉講 RoPE，但 NB5/07 要對上官方的 `complex64` 寫法，中間有一段「複數旋轉 ⇔ 實數對旋轉」的等價需要明確寫出，否則讀者會覺得 NB5 的實數實作和官方碼「長得完全不一樣」。

**修復：** 在 `theory/07` 或 06 §3 補一段最短推導：把成對維度 $(x_{2i}, x_{2i+1})$ 視為複數 $x_{2i} + i\,x_{2i+1}$，乘以 $e^{i m\theta_i}$ 等價於左乘旋轉矩陣 $\begin{pmatrix}\cos & -\sin\\ \sin & \cos\end{pmatrix}$。三行即可，符合「不跳步」原則。

**優先級：P3 | 難度：低 | 檔案：`theory/07` §3 或 `theory/06` §3**

---

## 三、改善優先順序總表

| 優先 | 編號 | 對象 | 說明 | 難度 | 狀態 |
|---|---|---|---|---|---|
| **P1** | L1 | 新 `NB5` | 把 NB4 逐元件改造成 mini-LLaMA，A/B 訓練對照（RMSNorm→SwiGLU→RoPE→GQA→Flash）| 高 | ⬜ 待辦 |
| **P1** | L2 | 新 `theory/07` | 官方 `model.py` 逐節對照地圖，把 06「對照辨認」做成可跟讀文件 | 中 | ⬜ 待辦 |
| **P2** | L3 | 06 / CLAUDE.md / README | 把新出口（NB5、07）接進「下一步」與全倉導覽 | 低 | ⬜ 待辦 |
| **P2** | L4 | NB5 / .gitignore | 確保 NB5 不引入重依賴、輸出路徑隔離 | 低 | ⬜ 待辦 |
| **P3** | L5 | 07 / 06 §3 | 補「複數旋轉 ⇔ 實數對旋轉」等價推導，橋接官方 RoPE 寫法 | 低 | ⬜ 待辦 |

---

## 四、建議執行順序

1. **先做 L2（theory/07）**：純文字、難度中，先把對照地圖立起來，L1 寫 NB5 時可直接引用它的命名與公式，避免重工。
2. **做 L1（NB5）**：核心且最費時。建議按 §2→§6 的順序一個元件一個元件實作並驗證，每步確認 loss 不發散再往下。Flash Attention（§6）放最後，因為它不改變數學、只改實作。
3. **做 L4**：在 L1 過程中順手確認依賴與路徑，避免 NB5 完成後才發現引入了重依賴。
4. **做 L3**：所有產出物就位後，統一更新導覽與「下一步」，確保引用全部有效（CLAUDE.md 要求）。
5. **做 L5**：清理性補強，可併入 L2 一起寫。

---

## 五、不納入本計劃的部分（守住倉庫定位）

- **真實規模訓練**：NB5 仍是字元級、短訓練的教學模型，不追求 LLaMA 的實際性能；只觀察「架構替換的趨勢差異」。
- **張量並行 / 多 GPU**：`fairscale` 的 `ColumnParallelLinear` 等只在 07 用一句話還原成普通 Linear，不實作。
- **量化、LoRA、推論最佳化**：屬部署主題，超出「從零學 Transformer」的範圍。
- **`flash-attn` pip 套件**：不安裝；只用 PyTorch 內建 SDPA（見 L4）。
- **Mistral / Qwen / Gemma 的專屬機制**（sliding window、QK-Norm 等）：07 一句話帶過，不展開。

---

*本計劃書作為「當代架構實作出口」階段的參考依據，每完成一項請在優先順序表中標記完成，並沿用 CLAUDE.md 的行文品質原則與引用有效性檢查。*
