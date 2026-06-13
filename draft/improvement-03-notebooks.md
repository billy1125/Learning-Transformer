# 學習路徑改善計劃 03：Notebook 逐 Cell 檢視與執行驗證

> 撰寫日期：2026-06-12
> 前提：`improvement-00-fixes.md`（P1–P4）、`improvement-01-mainline-gaps.md`（A1–A12、B1–B5）、`improvement-02-writing.md`（W1–W16）全部完成。
> 本計劃聚焦：**四個 Notebook 的逐 cell 正確性與可執行性**，目標是讀者能從頭到尾執行每個 notebook 不遇到錯誤，且輸出數字與理論文件一致。

---

## 一、現況總覽

執行前的掃描結果：

| Notebook | Code Cells | 有輸出 | 狀態 |
|---|---|---|---|
| NB1-simple-llm-vanilla.ipynb | 12 | 12 | 已執行，輸出完整 |
| NB2-simple-llm-pytorch.ipynb | 12 | 12 | 已執行，輸出完整 |
| NB3-llm-backpropagation.ipynb | 15 | 15 | 已執行，但**梯度驗證失敗** |
| NB4-nanoGPT.ipynb | 14 | 0 | **從未執行** |

---

## 二、發現的問題（依優先順序）

---

### N1　NB3 §9：端對端梯度驗證失敗（P1，嚴重）

**問題：**

NB3 的核心價值是「手刻梯度，數值驗證通過」。§9 的梯度驗證 cell 輸出為：

```
端對端梯度驗證（Wq[0:4,0:4]）最大誤差: 1.07e-01  ✗ 有問題
```

緊接著的注意事項寫「< 1e-4 均可接受」，但 1.07e-01 比門檻高出 **1000 倍**，顯然是真實的 bug，不是數值精度問題。

**需要調查的方向：**

1. 驗證時只取 `Wq[0:4, 0:4]`——是否恰好落在梯度流不健全的區域？試改驗證 `Wq` 全部元素或取不同子集。
2. 多頭注意力（H=4）的梯度合並方式是否正確？各頭的梯度是否有被錯誤地累加或截斷？
3. 數值微分的 ε 是否過大（有效位數不夠）？當 ε=1e-3 而模型輸出在 1e-1 量級，誤差可能放大。
4. 損失函數中是否有非微分處（如 causal mask 邊界）影響了特定索引。

**修復目標：**

最大梯度誤差 < 1e-4，或清楚解釋哪些部分預期誤差較高（如 causal mask 邊界）並改用更合理的驗證範圍。

**優先級：P1 | 難度：中 | 文件：`NB3-llm-backpropagation.ipynb` §9**

---

### N2　NB4：從未執行（P1，必做）

**問題：**

NB4 是整個學習路徑的終點（nanoGPT 完整訓練），但**所有 14 個 code cell 均無輸出紀錄**。讀者打開 notebook 時無法參照預期輸出，也無法判斷自己的環境是否正確。

**執行驗證清單：**

- [ ] Cell `imports`：`torch.__version__` 和裝置確認無誤
- [ ] Cell `data`：莎士比亞資料集下載成功（`urllib.request` 或本地快取）
- [ ] Cell `tokenizer`：詞彙表大小應為 65 個唯一字元
- [ ] Cell `batch_fn`：`get_batch` 回傳 shape `(64, 256)` 無誤
- [ ] Cell `head/mha/block/gpt_model`：模型初始化，總參數量應在 10M 左右（`n_embd=384, n_head=6, n_layer=6, vocab_size=65`）
- [ ] Cell `eval_fn`：`estimate_loss()` 定義無誤
- [ ] Cell `train_loop`：5000 steps 完整訓練，最終 val loss 約 1.48
- [ ] Cell `plot`：matplotlib 損失曲線圖顯示正常
- [ ] Cell `generate`：生成 500 字元的「莎士比亞風格」文字
- [ ] Cell `save_load`：checkpoint 成功儲存

**優先級：P1 | 難度：低（只需執行）| 文件：`NB4-nanoGPT.ipynb`**

---

### N3　NB4：輸出檔案路徑未隔離（P2）

**問題：**

NB4 有兩個 side effect：

1. `urllib.request.urlretrieve(url, 'input.txt')` — 下載至執行時的 CWD
2. `torch.save(..., 'nanogpt_checkpoint.pt')` — 儲存至 CWD

從 JupyterLab 開啟時 CWD 通常是倉庫根目錄，但從 VSCode 或命令列執行可能不同，導致路徑行為不一致。更嚴重的是，這兩個檔案會出現在根目錄，未被 `.gitignore` 忽略可能被誤 commit。

**修復：**

```python
# 統一使用相對路徑，並建立 notebooks/data/ 目錄
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

data_path = os.path.join(data_dir, 'input.txt')
if not os.path.exists(data_path):
    urllib.request.urlretrieve(url, data_path)

# checkpoint 同理
torch.save(..., os.path.join(data_dir, 'nanogpt_checkpoint.pt'))
```

同時確認 `notebooks/data/` 在 `.gitignore` 中（大型訓練輸出不應 commit）。

**優先級：P2 | 難度：低 | 文件：`NB4-nanoGPT.ipynb` cell `data`、`save_load`**

---

### N4　NB2：模型與 checkpoint 存至 CWD（P2）

**問題：**

NB2 的最後一個 code cell：

```python
torch.save(model.state_dict(), 'mini_gpt.pt')
# ...
torch.load('mini_gpt.pt')
```

同 N3，CWD 相依問題，且 `mini_gpt.pt` 未被 `.gitignore` 忽略。

**修復：**

改為 `notebooks/data/mini_gpt.pt`，或使用 `os.path.join(os.path.dirname(__file__), 'mini_gpt.pt')`（確保存在 notebooks/ 目錄而非根目錄）。

**優先級：P2 | 難度：低 | 文件：`NB2-simple-llm-pytorch.ipynb` §12**

---

### N5　NB1：訓練示範誤導讀者（P2）

**問題：**

NB1 §10 的訓練迴圈輸出：

```
Step  1/10  Loss: 3.4389
Step  2/10  Loss: 3.4389
...
Step 10/10  Loss: 3.4388
```

損失幾乎完全沒有下降（差距 0.0001），而 §9 的「loss 接近隨機猜測的理論損失」說明也似乎暗示訓練有效。

**根本原因：**

訓練迴圈只更新 `model_small.emb.W`（Embedding 矩陣），而在前向傳播中 Embedding 輸出後立即被加上隨機初始化的注意力層所掩蓋；數值微分的 ε=1e-3 對 Embedding 梯度訊號也可能不夠精確。

**修復選項（擇一）：**

**Option A（說明清楚）**：在 §10 開頭加一段說明：
> 本示範只更新 Embedding 矩陣（以節省時間），因此損失下降幅度極小。完整訓練需更新所有參數（`all_params()`），但數值微分的計算量是 O(參數數量)，對幾千個參數仍需數分鐘。NB2（PyTorch 版）展示了使用 autograd 訓練的正確效果。

**Option B（換一個更好的示範）**：只用 4 個字元的超小訓練集（如 `"ab"`），讓 10 步內損失明顯下降。

**建議採用 Option A**，配合 Option B（對 `hello world` 改用 `"ab"` 短序列）。

**優先級：P2 | 難度：低 | 文件：`NB1-simple-llm-vanilla.ipynb` §10**

---

### N6　NB3 §9：梯度驗證範圍說明不清（P2）

**問題：**

即使 N1 的 bug 修復後，現有的梯度驗證只取 `Wq[0:4, 0:4]`（4×4 子矩陣，共 16 個元素），但模型的 `Wq` 維度是 `(64, 64/4) = (64, 16)`（多頭分割後）或 `(D, D) = (64, 64)`，讀者不清楚：

- 為什麼只驗證子集？
- 子集是否代表性足夠？
- 如何驗證其他參數（Wk、Wv、W1、W2）？

**修復：**

在 §9 的 markdown cell 補充說明：
> 全部參數的端對端數值微分約需 70,000 × 2 次前向傳播，在教學環境中太慢。我們取 `Wq` 的前 4×4 子矩陣（16 個元素）作為代表性樣本。若這 16 個元素梯度正確，表示計算圖中的矩陣乘法、softmax、layer_norm 等路徑都沒有問題。

同時增加一個「快速全參數梯度統計」cell，計算所有參數的相對誤差分佈（抽樣 32 個元素），而非深度驗證。

**優先級：P2 | 難度：低 | 文件：`NB3-llm-backpropagation.ipynb` §9**

---

### N7　全 Notebook：Cross-Reference 一致性檢查（P3）

**問題：**

四個 notebook 的 navigation header 中引用了理論文件路徑，需要確認這些相對連結在重構後仍正確：

- NB1: `../theory/01a`, `../theory/02`, `../theory/03` ← 需確認
- NB2: 同上
- NB3: `../theory/01a`, `../theory/02`, `../theory/03`, `../theory/05`
- NB4: `../theory/01a`, `../theory/02`, `../theory/03`, `../theory/04`

同時確認各 notebook 內 markdown cell 中的「前置理論」說明與當前 theory/ 目錄實際文件名一致。

**優先級：P3 | 難度：低 | 全部四個 notebook**

---

### N8　NB3 §14：梯度健康度分析 cell 輸出空白（P3）

**問題：**

NB3 的 §14「各層梯度流分析」cell 的輸出是空的（`last_out = ''`），這個 cell 定義了 `analyze_grad_flow` 函式並呼叫，但 JupyterLab 的 cell 輸出似乎沒有被儲存。

**需要確認：**

此 cell 在重新執行時是否產生任何輸出（圖表或文字）。若函式只定義不呼叫，補上一個呼叫示範。

**優先級：P3 | 難度：低 | 文件：`NB3-llm-backpropagation.ipynb` §14**

---

### N9　.gitignore：補充 Notebook 產生的大型檔案（P2）

**問題：**

目前倉庫的 `.gitignore` 可能未涵蓋 notebook 執行產生的大型輸出：
- `input.txt`（莎士比亞資料，~1MB）
- `nanogpt_checkpoint.pt`（模型 checkpoint，~10MB）
- `mini_gpt.pt`
- `training_loss.png`（NB3 的訓練曲線圖）
- `notebooks/data/`（修復 N3/N4 後的輸出目錄）

**修復：**

確認 `.gitignore` 包含這些條目，避免大型二進位檔案被意外 commit。

**優先級：P2 | 難度：低 | `.gitignore`**

---

## 三、改善優先順序總表

| 優先 | 編號 | Notebook | 說明 | 難度 | 狀態 |
|---|---|---|---|---|---|
| **P1** | N1 | NB3 §9 | 梯度驗證誤差 1.07e-01（應 < 1e-4）：調查 bug 並修復 | 中 | ✅ 完成（根因：`clip=1.0` 縮放解析梯度；修復：加 `clip=1e9`）|
| **P1** | N2 | NB4 全部 | 首次執行：14 個 code cell 全部執行並儲存輸出 | 低 | ✅ 完成（快速驗證版：200 steps，CPU ~30 秒；完整 5000 steps 需 ~30 分鐘 CPU）|
| **P2** | N3 | NB4 | 修復 `input.txt` 和 `checkpoint.pt` 路徑至 `notebooks/data/` | 低 | ✅ 完成（改用 `data_dir = 'data'` 相對路徑）|
| **P2** | N4 | NB2 | 修復 `mini_gpt.pt` 路徑 | 低 | ✅ 完成（改為 `data/mini_gpt.pt`）|
| **P2** | N5 | NB1 §10 | 補說明：訓練示範只更新 Embedding，指向 NB2 看完整訓練效果 | 低 | ✅ 完成 |
| **P2** | N6 | NB3 §9 | 補充梯度驗證的取樣說明（為何只驗 4×4 子集、為何 clip=1e9）| 低 | ✅ 完成 |
| **P2** | N9 | `.gitignore` | 補充大型輸出檔案排除規則 | 低 | ✅ 完成（新增 `notebooks/data/`、`input.txt`、`training_loss.png`）|
| **P3** | N7 | 全部 | Cross-reference 路徑一致性檢查 | 低 | ✅ 完成（12 個引用全部有效）|
| **P3** | N8 | NB3 §14 | 確認梯度健康度分析 cell 有正確輸出 | 低 | ✅ 完成（3 個輸出，含圖表）|

---

## 四、建議執行順序

1. **先修 N3/N4/N9（路徑與 gitignore）**：5 分鐘，避免 N2 執行後誤 commit 大型檔案。
2. **執行 N2（NB4 首次執行）**：預估 CPU ~10 分鐘、GPU ~2 分鐘；執行後儲存輸出到 notebook。
3. **調查並修 N1（NB3 梯度驗證 bug）**：這是技術難度最高的一項，需要逐步除錯手刻反向傳播的每一條梯度路徑。
4. **修 N5/N6**：低難度說明補充，改善 NB1 和 NB3 的可讀性。
5. **最後做 N7/N8**：清理性工作，確保引用和輸出的完整性。

---

## 五、不需處理的部分

- **NB1/NB2/NB3 的訓練輸出數字**：現有 cached 輸出是上次執行的真實結果，重新執行因隨機種子不同數字可能略有差異，不算 bug（只要數量級一致即可）。
- **NB4 的生成文字品質**：莎士比亞風格生成結果帶有隨機性，不需要固定輸出。
- **NB3 §11 的訓練曲線圖**：matplotlib 的圖片輸出已在 cell output 中，不需另存。

---

*本計劃書作為 Notebook 改善階段的參考依據，每完成一項請在優先順序表中標記完成。*
