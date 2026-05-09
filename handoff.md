# MDLA7 Handoff

日期時間：2026-05-10 03:19:30 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 microblock commit：`a8dd50f Expand microblock fused pipeline tails`
- Gantt task-meta commit：`a490aab Add task-meta microblock Gantt timeline`
- Fast-only runner commit：`09d97cb Implement fused microblock fast-only runners`
- 工作樹仍有使用者/文件/profile 類未提交修改；不要混進 microblock code commit。

## 已完成

- Profile HTML 有兩張 Gantt：
  - original engine timeline
  - microblock stage timeline
- `task_meta` 已從 Command Engine trace 到 profile JSON，Gantt 用 descriptor metadata 判斷 microblock。
- `run_*.py` 已統一支援 `--fast-only`。
- Path 1-6 已可用：
  - CONV/DW/FC -> Requant -> store/forward
  - CONV -> Requant -> binary EWE ADD/MUL/SUB -> store/forward
  - CONV -> Requant -> D2SPACE
  - Binary EWE chain -> store/forward
  - Binary EWE chain -> D2SPACE
  - CONV/Requant -> binary EWE -> unary EWE(HARD_SWISH/GELU) -> store/forward
- 已補強：
  - Binary EWE chain -> unary EWE tail
  - CONV/Requant -> EWE -> POOL safe subset
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

`MDLA7 profile - gpt2_quant_L24_L63.tflite` 的 L2 目前沒有 microblock。

- L2 是 `FC in=1x1x768 out=1x1x768`
- `tiles=1x1`
- profile JSON 沒有 `task_meta` layer 2
- 有 microblock metadata 的 layer 從 L3 / EWE 類路徑開始

原因：

- `try_stream_conv_chain()` 目前只吃 `OK_CONV/OK_DWCONV`，不吃 `OK_FC`
- `try_stream_conv_ewe()` 雖接受 `OK_FC`，但 L2 -> L3 shape 不合：
  - L2 output `1x1x768`
  - L3 ADD input `1x77x768`
- L2 working set 約 583 KB，低於 2 MB L1，不是容量驅動的必切 microblock

若要讓 L2 顯示 microblock，可選：

- 加 single-tile pseudo-microblock metadata，只改善 Gantt 可視性
- 真做 FC microblock，需決定按 output-channel 切或改 compiler descriptor 表示成 token-batched FC

## Next To-do

優先強化 Path 7-10，方向如下：

1. Path 7 完整化
   `CONV/Requant -> EWE -> POOL/TNPS consumer`
   目前 POOL 只支援 safe subset：`1x1 / stride1 / no-pad`。下一步要處理 real window pool 的 tile halo / partial window。

2. Path 8 完整化
   `POOL -> unary/EWE/TNPS consumer`
   目前 POOL 能被計時與標 metadata，但還不是完整 producer tail model。

3. Path 9 完整化
   `TNPS/layout -> CONV or compute consumer`
   目前只有 RESHAPE/SQUEEZE/EXPAND_DIMS 這種 L1 view safe subset。TRANSPOSE/PACK/UNPACK 還會 materialize，是 GPT/ViT 類模型的主要斷點。

4. Path 10 泛化
   `producer tile -> multiple consumers / concat fanout`
   目前仍主要是 `conv_fanout` special-case，不是通用 fanout framework。

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
