# Microblock Pattern Slice Workspace

這個目錄用來準備 microblock pattern implementation 的候選 slice。

目前已產生：

- `microblock_pattern_candidates.csv`

這份 CSV 是從 `model/ETHZ_v6/*.tflite` 掃出的 pattern candidate manifest，
還不是實際切好的 `.tflite` 子模型。下一步要補真正的 FlatBuffer subgraph
slicer，依照 `suggested_slice` 欄位產生對應小模型。

## 欄位

| 欄位 | 說明 |
|---|---|
| `pattern` | 掃到的 microblock pattern 類型 |
| `model` | 原始 ETHZ V6 模型 |
| `start_op` | 原始模型中的起始 operator index |
| `end_op` | 原始模型中的結束 operator index |
| `op_sequence` | operator sequence |
| `input_shape` | 起始 op 的 input shape |
| `output_shape` | 結束 op 的 output shape |
| `suggested_slice` | 建議未來產出的 slice 檔名 |
| `notes` | candidate 補充說明 |
| `source_path` | 原始 `.tflite` 路徑 |

## 目前 ETHZ V6 掃描結果

已依 `pattern + op_sequence + input_shape` 去重；同構候選只保留第一筆代表項。

| Pattern | Count |
|---|---:|
| `producer_compute` | 1214 |
| `fanout_live_range` | 762 |
| `udma_as_engine` | 455 |
| `layout_bridge` | 198 |
| `consumer_tail` | 161 |
| `streaming_preload` | 125 |

這六類對應 handoff 裡的 reusable microblock pattern：

- Producer compute
- Consumer tail
- Layout bridge
- Fanout/live-range
- UDMA as engine
- Streaming preload

## 使用方式

```bash
python3 systemc/scripts/scan_microblock_patterns.py
```

輸出：

```text
model/MB_Path_Slice/microblock_pattern_candidates.csv
```

## 下一步

1. 依 `pattern` 挑小而代表性的候選。
2. 補 FlatBuffer subgraph slicer，產出實際 `model/MB_Path_Slice/*.tflite`。
3. 對每個 slice 跑 `compile_model.py + mdla7_model_runner`。
4. 用 slice 驗證通用 pattern，而不是每遇到一個模型就新增 special-case path。
