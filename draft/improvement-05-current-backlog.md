# 改善計劃 05：當前 backlog（整倉盤點）

> 前提：`improvement-00`～`03` 全部完成（歷史存檔）；`improvement-04-llama`（L1–L5）仍為**規劃中**。
> 本檔是一次**整倉盤點**後的當前待辦清單：已解決的不再列入，開放項目保留，新發現的建議新增於此。
> 沿用各改善檔的行文品質原則與「引用必須有效」檢查。

## 盤點結論（2026-07 當前狀態）

倉庫在 00–03 四輪改善後，又臨時新增了三條 encoder 家族 / 應用出口內容（**非改善輪次**，
均已完成並提交於 `add-bert-encoder-only` 分支）：

- `theory/07-bert-encoder-only.md` ＋ `notebooks/NB5-bert-mlm.ipynb`（BERT／MLM 選讀分支）
- `theory/09-text-to-vector-rag.md`（文字轉向量與 RAG，encoder 應用出口）
- `advanced/Seq2Seq-and-Decoding-Techniques.md`（李宏毅課程整理的延伸閱讀）

**健康度檢查：** 全倉 22 份 md 的 markdown 連結掃描 **零斷連**；理論↔Notebook 配對除下述 C1 外皆完整。
因此本輪 backlog **沒有殘留 bug**，全是前瞻性補強與一致性收尾。

---

## 一、開放項目（依優先順序）

### C1　`theory/09`（RAG）缺 Notebook 對應（P2）

**問題：** 倉庫的核心原則是「理論↔Notebook 相互對應」（見 `CLAUDE.md`）。目前 `07↔NB5` 已配對，
但 `09-text-to-vector-rag.md` 沒有實作對應，RAG 檢索流程只停在文字說明。

**建議：** 新增一個最小 RAG 檢索 demo notebook（可命名 `NB-rag-retrieval` 或視編號政策定）——
**重用 `NB5` 的 mini-BERT** 當 encoder，把幾句話編碼成向量、對 query 做**餘弦 top-k**檢索，
印出最相關的句子；可選再把檢索結果拼進 prompt 餵給 `NB4` 的 GPT 做「檢索增強生成」的玩具版。
零新相依（純 PyTorch），落實 09 的「encoder 檢索 + decoder 生成」。

**優先級：P2 | 難度：中 | 對象：新增 notebook + `09` §5 加實作連結**

### C2　`theory/transformer_block_simple_explanation.md` 是孤兒檔（P3）

**問題：** 這份「Transformer Block 高中生版」未被任何導覽（`CLAUDE.md` 理論表、`README.md`、
其他 theory 文件）引用，處於孤立狀態；與倉庫「每份理論文件開頭標示對應 Notebook、彼此交叉引用」
的規範不一致。

**建議（二擇一）：** (a) **納入導覽**——列入 `CLAUDE.md`／`README` 理論表，定位為 `03a` 的
「超白話輔助版」並與 `01a`／`03a` 互連；或 (b) **移入 `archive/`**（若判定與 `01a`／
`transformer_block` 相關內容重複、不再維護）。需先比對它與 `03a §6`／`01a` 的重疊度再定。

**✅ 已完成（2026-07）採 (a)：** 改名為 `theory/03a-transformer-block-plain.md`、補上倉庫規範檔頭
（適合對象／定位／對應 Notebook／下一步），定位為 `03a §6` 白話輔助版；`03a §6` 開頭與 `01a`
下一步各補一則指向連結，並列入 `CLAUDE.md`／`README` 理論表與資料夾說明。

**優先級：P3 | 難度：低 | 對象：`transformer_block_simple_explanation.md`、`CLAUDE.md`、`README.md`**

### C3　`06` 未連結 encoder 姊妹分支（07／09）（P3）

**問題：** `06`（decoder→LLaMA）與 `07`（encoder→BERT）是主線學完後的兩條平行分支，
`09` 是 encoder 分支的應用出口。但 `06` 全文未提及 07／09，導覽只往 decoder 方向單向延伸。

**建議：** 在 `06` 開頭定位或文末「下一步」補一句——除了 decoder 家族（本文→LLaMA），
另有 encoder 家族分支 [`07`](../theory/07-bert-encoder-only.md) 與其應用出口
[`09`](../theory/09-text-to-vector-rag.md)，與 `00 §5` 的分支圖一致。

**優先級：P3 | 難度：低 | 對象：`theory/06`**

### C4　Notebook 對新分支的可發現性（P3）

**問題：** `NB4` 附錄「進一步改進方向」列了 RoPE／SwiGLU／Flash（皆已寫成 `06`），卻沒有前向連結；
`NB5` 也可指回 `09`／Seq2Seq essay。Notebook 讀者不易發現這些延伸文件。

**建議：** `NB4` 附錄的每條改進方向加上「詳見 `06` §X」；末尾補一句指向 `07`（encoder 分支）與
`09`（RAG）。`NB5` 末尾的延伸段補指 `09`（RAG 應用）與 `advanced/Seq2Seq-and-Decoding-Techniques`。

**優先級：P3 | 難度：低 | 對象：`NB4` 附錄、`NB5` 末段**

### C5　解碼策略的 Notebook 展示（P3，選作）

**問題：** `advanced/Seq2Seq-and-Decoding-Techniques.md` 與 `04 §8` 都談 greedy / beam search /
取樣（temperature、top-k），但沒有 notebook 實際對照這些策略的輸出差異。`NB4.generate()` 已有
temperature／top_k，缺 beam search 與並排對照。

**建議：** 在 `NB4` 末尾加一小節，用同一個訓練好的模型，並排展示 greedy vs temperature vs top-k
（beam search 可選，因對開放式生成不一定更好，可只用文字說明），呼應 essay 的解碼策略段。

**優先級：P3 | 難度：低 | 對象：`NB4` 生成小節**

---

## 二、承接自 improvement-04-llama（仍規劃中）

以下項目維持在 [`improvement-04-llama.md`](improvement-04-llama.md)，此處僅列引以保持單一 backlog 視圖，
內容不重複（詳見該檔 L1–L5 與總表）：

| 編號 | 對象 | 說明 | 狀態 |
|---|---|---|---|
| L1 | 新 `NB6-nanoGPT-to-llama.ipynb` | 把 NB4 逐元件改造成 mini-LLaMA（RMSNorm→SwiGLU→RoPE→GQA→Flash）| ⬜ 待辦 |
| L2 | 新 `theory/08-reading-llama-source.md` | 官方 `model.py` 逐節對照地圖 | ⬜ 待辦 |
| L3 | `06`／`CLAUDE.md`／`README` | 把 NB6／08 出口接進導覽 | ⬜ 待辦 |
| L4 | `NB6`／`.gitignore` | 確保不引入重依賴、輸出路徑隔離 | ⬜ 待辦 |
| L5 | `08`／`06 §3` | 「複數旋轉 ⇔ 實數對旋轉」等價推導 | ⬜ 待辦 |

> 編號協調備忘：`07`／`NB5` 已歸給 BERT 分支，LLaMA 出口用 `08`／`NB6`（見 `improvement-04-llama` 開頭）。

---

## 三、本輪 backlog 總表

| 優先 | 編號 | 對象 | 說明 | 難度 | 狀態 |
|---|---|---|---|---|---|
| **P2** | C1 | 新 notebook + `09` | RAG 檢索 demo（重用 NB5 mini-BERT → 餘弦 top-k）| 中 | ⬜ 待辦 |
| **P3** | C2 | `transformer_block_simple_explanation.md` | 孤兒檔：改名 `03a-transformer-block-plain.md`、補檔頭、納入導覽並與 01a／03a §6 互連 | 低 | ✅ 完成 |
| **P3** | C3 | `theory/06` | 補連結到 encoder 姊妹分支 07／09 | 低 | ⬜ 待辦 |
| **P3** | C4 | `NB4` 附錄、`NB5` 末段 | 補前向連結，提升分支可發現性 | 低 | ⬜ 待辦 |
| **P3** | C5 | `NB4` 生成小節 | 並排展示解碼策略（greedy/temperature/top-k）| 低 | ⬜ 待辦（選作）|
| **P1–P3** | L1–L5 | 見第二節 | LLaMA 實作出口（`improvement-04-llama`）| 高–低 | ⬜ 待辦 |

**建議執行順序：** 先做零散低難度的 C3／C4（一致性收尾，順手）→ C1（RAG demo，中難度、補完 09 的實作缺口）
→ C2（孤兒檔決策）→ 進入 L1–L5 的 LLaMA 出口大工程。C5 視需要選作。

---

*本檔為當前 backlog 的單一視圖。每完成一項請在總表標記 ✅；解決後可從本檔移除，並在 `CLAUDE.md`
的改善計劃索引同步狀態。*
