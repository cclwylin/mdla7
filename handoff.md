# MDLA7 Handoff

日期時間：2026-05-10 05:49:40 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 microblock commit：`d51a330 Strengthen microblock paths 7-10`
- 前一版 microblock tail commit：`a8dd50f Expand microblock fused pipeline tails`
- Gantt task-meta commit：`a490aab Add task-meta microblock Gantt timeline`
- Fast-only runner commit：`09d97cb Implement fused microblock fast-only runners`
- 工作樹仍有使用者/文件/profile 類未提交修改；不要混進 microblock code commit。

## 已完成

- Profile HTML 有兩張 Gantt：
  - original engine timeline
  - microblock stage timeline
- `task_meta` 已從 Command Engine trace 到 profile JSON，Gantt 用 descriptor metadata 判斷 microblock。
- `run_*.py` 已統一支援 `--fast-only`。
- Path 1-10 已可用：
  - CONV/DW/FC -> Requant -> store/forward
  - CONV -> Requant -> binary EWE ADD/MUL/SUB -> store/forward
  - CONV -> Requant -> D2SPACE
  - Binary EWE chain -> store/forward
  - Binary EWE chain -> D2SPACE
  - CONV/Requant -> binary EWE -> unary EWE(HARD_SWISH/GELU) -> store/forward
  - CONV/Requant -> EWE -> POOL/TNPS consumer
  - POOL -> unary/binary EWE/TNPS consumer
  - TNPS/layout -> compute consumer handoff for safe GraphMeta boundaries
  - producer tile -> multiple consumers / concat fanout safe subset
- 已補強：
  - Binary EWE chain -> unary EWE tail
  - CONV/Requant -> EWE -> real-window/global POOL tail
  - POOL producer microblock tail to binary EWE / unary EWE / D2S(TNPS)
  - TRANSPOSE/PACK/UNPACK/SPLIT GraphMeta layout handoff no-store
  - conv fanout generalized to safe CONV/DW/FC row-tiled subset
  - tiled POOL / unary EWE descriptor metadata
  - POOL layer cycle accounting
  - RESHAPE/SQUEEZE/EXPAND_DIMS L1 view passthrough safe subset

## 最近驗證

```bash
make -C systemc -s
git diff --check
./batch/run_model.py --model model/MLPerf_Tiny/vww_96_int8.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
```

結果：

- `vww_96_int8`: `31/31 PASS`
- `llama2_quant_L35_L74`: `43/43 PASS`
- `mobilenet_v3_quant`: `123/123 PASS`
- `gpt2_quant_L24_L63`: `42/42 PASS`

## GPT2 L2 結論

`MDLA7 profile - gpt2_quant_L24_L63.tflite` 的 L2 現在有 FC OC-slice microblock。

- L2 是 `FC in=1x1x768 out=1x1x768`
- `tiles=1x3`，以 256 OC 為預設 slice，3 個 microblocks
- profile JSON 已有 `task_meta` layer 2：
  - `udma_r`: input preload + 3 個 weight-slice loads
  - `conv`: 3 個 FC compute microblocks
  - `requant`: 3 個 requant microblocks
- `gpt2_quant_L24_L63`: `42/42 PASS`

實作邊界：

- 這版是 `OC slice x full-K`，也就是每個 microblock 載一段 weight rows，
  input vector / params 留在 L1。
- 還不是 partial-K accumulation；真的 K-slice 需要 CONV engine 增加 psum
  buffer / accumulate descriptor 協定。

## Next To-do

Path 7-10 已完成 conservative implementation。下一步若要再加強，可做：

1. True TNPS tiled kernels
   目前 `TRANSPOSE/PACK/UNPACK/SPLIT` 對 GraphMeta-confirmed intermediate boundary 會做 no-store handoff；任意 layout permute 的真正 tile kernel仍是後續工作。

2. Multi-source concat descriptor
   現在 concat fanout 可以 suppress intermediate stores，但 CONCAT 本身仍主要靠 compiler packed tensor / materialized layout path。

3. FC partial-K accumulation
   GPT2 L2 已有 FC OC-slice microblock；若要做到真正 K-slice，CONV engine
   需要支援多段 partial sum accumulate，再由最後一段觸發 Requant。

4. Wider fanout ownership model
   目前 fanout 放寬到 safe CONV/DW/FC subset；任意 producer multiple-consumer 仍需更完整的 live-range / slot ownership。

## 快速命令

```bash
make -C systemc -s
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
```

Profile entry：

```text
batch/profile_hotspot.html
batch/profile_ethz_v6.html
batch/output/<stem>.html
```
