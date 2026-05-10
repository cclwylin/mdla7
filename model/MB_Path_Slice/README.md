# Microblock Pattern Slice Workspace

這個目錄用來準備 microblock pattern implementation 的候選 slice。

目前已產生：

- `microblock_pattern_candidates.csv`
- `microblock_pattern_slices.csv`
- 每個 pattern 子目錄下的第一批代表 `.tflite`

這份 CSV 是從 `model/ETHZ_v6/*.tflite` 掃出的 pattern candidate manifest，
`microblock_pattern_slices.csv` 則記錄已切出的代表子模型。

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
目前候選集也會濾掉太長的 operator range，預設只保留 `max_ops <= 32`，
避免像 `L9-L71` 這類跨太遠 live-range 被拿來當 microblock slice。

| Pattern | Count |
|---|---:|
| `producer_compute` | 1214 |
| `fanout_live_range` | 691 |
| `udma_as_engine` | 455 |
| `layout_bridge` | 198 |
| `consumer_tail` | 161 |
| `streaming_preload` | 125 |

總候選數：2844。

這六類對應 handoff 裡的 reusable microblock pattern：

- Producer compute
- Consumer tail
- Layout bridge
- Fanout/live-range
- UDMA as engine
- Streaming preload

## 使用方式

掃描候選：

```bash
python3 systemc/scripts/scan_microblock_patterns.py
```

輸出：

```text
model/MB_Path_Slice/microblock_pattern_candidates.csv
```

切出第一批代表 `.tflite`：

```bash
python3 systemc/scripts/slice_microblock_patterns.py --max-per-pattern 3
```

全切目前乾淨候選集：

```bash
python3 systemc/scripts/slice_microblock_patterns.py --max-per-pattern 0
```

輸出：

```text
model/MB_Path_Slice/<pattern>/*.tflite
model/MB_Path_Slice/microblock_pattern_slices.csv
```

目前已切出 23 個代表 slice：

| Pattern | Slice count |
|---|---:|
| `consumer_tail` | 8 |
| `fanout_live_range` | 3 |
| `layout_bridge` | 3 |
| `producer_compute` | 3 |
| `streaming_preload` | 3 |
| `udma_as_engine` | 3 |

`consumer_tail` 額外保留多個自然模型代表，用來覆蓋 EWE / POOL / TNPS
tail 變形。

切片方式：

- 透過 `flatc` 與 `third_party/tflite/schema.fbs` 做 JSON round-trip。
- 只保留 selected operator range 用到的 tensors / buffers。
- 重寫 subgraph inputs / outputs 成 operator range boundary。
- 保留原 `operator_codes`，避免 opcode remap 風險。

## 目前 regression 狀態

最近一次 fast-only 全跑：

```bash
python3 batch/run_mb_path.py --fast-only --rerun-all
```

結果：`23/23 ok`。

代表改善：

| Pattern / slice | Before | After |
|---|---:|---:|
| `layout_bridge/deeplab_v3_plus_float_L68_L69` | `0.093 ms / mb=0` | `0.070 ms / mb=5` |
| `layout_bridge/deeplab_v3_plus_float_L73_L74` | `1.412 ms / mb=0` | `0.902 ms / mb=64` |
| `producer_compute/deeplab_v3_plus_float_L2_L2` | `0.377 ms / mb=0` | `0.285 ms / mb=16` |
| `streaming_preload/deeplab_v3_plus_float_L16_L16` | `0.197 ms / mb=0` | `0.131 ms / mb=10` |

這批改善來自 FP H-tiled `CONV/DWCONV` ping-pong microblock streaming：
FP tile 使用和 INT8 相同的 L1 slot hazard tags，因此可進 `DF_STREAM`
Command Engine lookahead；INT16 仍保守關閉。

## 下一步

1. 補 `sslice/layout -> CONV` true L1 slice-feed，避免 materialized input 從 DRAM reload。
2. 補更通用的 multi-source concat / live-range ownership model。
3. 依 regression 結果調整 candidate ranking，挑更小更準的代表。
4. 用 slice 驗證通用 pattern，而不是每遇到一個模型就新增 special-case path。
