# MDLA7 Handoff

日期時間：2026-05-10 19:33:56 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 commit：`66af747 Remove old MDLA7 report markdown`
- 最新 code/doc commit：`7bce6ac Update 4096-bit requant chain and pool fanout`
- 前一版 microblock commit：`3df0b4b Stream layout slice tails in microblocks`
- FP tiled microblock commit：`8763613 Enable FP tiled conv microblock streaming`
- 前一版 microblock path commit：`239afe7 Optimize TNPS strided slice tail`
- Path 7-10 強化 commit：`d51a330 Strengthen microblock paths 7-10`
- 前一版 microblock tail commit：`a8dd50f Expand microblock fused pipeline tails`
- Gantt task-meta commit：`a490aab Add task-meta microblock Gantt timeline`
- Fast-only runner commit：`09d97cb Implement fused microblock fast-only runners`
- 工作樹目前只剩非 md/code 的未提交項目：舊
  `reports/MDLA7_System_Report.pptx` deletion、`image/`、新的
  `reports/uArcSim Implementation by AI.pptx` 與 Office lock file。
  這些不要混進 microblock code/doc commit。

## 已完成

- Profile HTML 有兩張 Gantt：
  - original engine timeline
  - microblock stage timeline
- `task_meta` 已從 Command Engine trace 到 profile JSON，Gantt 用 descriptor metadata 判斷 microblock。
- `run_*.py` 已統一支援 `--fast-only`。
- Microblock engine path 狀態：

| Path | Microblock engine pipeline | 狀態 |
|---|---|---|
| 1 | `CONV/DW/FC -> Requant -> store/forward` | Done；INT8/FP H-tiled ping-pong stream 已開，CONV->Requant chain 已更新為 4096 bit/cyc，INT16 仍保守 |
| 2 | `CONV -> Requant -> EWE ADD/MUL/SUB -> store/forward` | Done |
| 3 | `CONV -> Requant -> D2SPACE` | Done |
| 4 | `Binary EWE chain -> store/forward` | Done |
| 5 | `Binary EWE chain -> D2SPACE` | Done |
| 6 | `CONV -> Requant -> EWE ADD -> unary EWE/ReLU -> store/forward` | Done |
| 7 | `CONV -> Requant -> EWE ADD -> pool/tnps consumer` | Done；`pool_tail` corpus reach 還需要自然模型覆蓋 |
| 8 | `POOL -> unary/EWE/TNPS consumer` | Done basic path |
| 9 | `TNPS/layout -> CONV` / `layout producer -> compute consumer` | Partial+；standalone `CONCAT -> CONV/DWCONV` 的 FP compute consumer 已有 MB，`SPACE_TO_DEPTH -> CONV` 已有 true L1 tile-feed；channel-only `SLICE/STRIDED_SLICE -> CONV` cut/final path 已有，full intermediate 先 guard |
| 10 | `producer -> multiple consumers / concat fanout` | Done；包含 direct fanout、concat pointwise、Requant direct packed concat、immediate pool-head + later skip consumer |
| 11 | `UDMA_R activation preload -> CONV/Requant` head/tail streaming | Done；3x3 和 1x1 都已套用 |
| 12 | `Requant -> packed concat L1 -> pointwise CONV` | Done；不再需要 TNPS pack descriptor |
| 13 | `UDMA_W store as microblock tail` | Done |
| 14 | `UDMA_R/UDMA_W as engine lanes in scheduling/timeline` | Done |
- 已補強：
  - Binary EWE chain -> unary EWE tail
  - CONV/Requant -> EWE -> real-window/global POOL tail
  - POOL producer microblock tail to binary EWE / unary EWE / D2S(TNPS)
  - TRANSPOSE/PACK/UNPACK/SPLIT GraphMeta layout handoff no-store
  - conv fanout generalized to safe CONV/DW/FC row-tiled subset
  - tiled POOL / unary EWE descriptor metadata
  - POOL layer cycle accounting
  - RESHAPE/SQUEEZE/EXPAND_DIMS L1 view passthrough safe subset
  - Requant strided output mode for direct packed concat L1 writes
  - UDMA activation streaming preload：用 head-ready / tail-busy descriptor
    降低 tiled Conv bubble。延伸到 1x1 pointwise Conv 後，
    `xlsr_quant` fast 從 0.698 ms 改善到 0.497 ms。
  - FP H-tiled `CONV/DWCONV` 已解除 microblock streaming guard，使用同一套
    ping-pong L1 slot hazard tags；`layout_bridge` / `producer_compute` /
    `udma_as_engine` 的 Deeplab FP slices 現在會產生真 `DF_STREAM` microblocks。
  - 架構資料寬度：`CONV -> Requant` chain 從 16 lanes 改成
    `CONV_REQUANT_CHAIN_LANES=128`，也就是 `4096 bits/cyc`。`mdla7.drawio`、
    `spec/spec.md`、教材 md/html/pdf 都已同步。
  - `unet_quant` L2 regression 已修：L1 CONV 是 multi-consumer residual
    fanout（immediate L2 maxpool + later skip），現在允許 direct
    `CONV -> POOL` fanout-head microblock。L2 從一般 tiled pool fallback
    改成 `tiles=74x1` pool tail，功能通過。
  - `SPACE_TO_DEPTH -> CONV/DWCONV` layout bridge 已新增 true L1 tile-feed：
    TNPS 先把 S2D tile 寫到 L1，後續 CONV 直接吃該 L1 tile，不再把 S2D
    當成必須 materialize 的 DRAM boundary。已加 descriptor row 16-bit
    encodable guard，避免大 row tile 溢位。
  - channel-only `SLICE/STRIDED_SLICE -> CONV/DWCONV` layout bridge 已新增
    L1 tile-feed safe subset：只接受空間維度不變、channel 連續切片、stride=1。
    cut/final consumer 預設啟用；full graph 的 intermediate consumer 目前用
    `MDLA7_EXPERIMENTAL_SLICE_COMPUTE=1` guard，因為 `xlsr_quant` full model
    直接啟用會從 `0.721 ms` 退到 `0.807 ms`。

## Microblock Path 策略

未來模型一定還會出現更多 layer 組合，但實作不應該一直用
Path 15、Path 16 這種方式加 special-case。比較正確的方向是把
engine path 收斂成可重用的 microblock pattern：

| Pattern | 覆蓋範圍 | 狀態 |
|---|---|---|
| Producer compute | `CONV/DW/FC/EWE/POOL/TNPS -> L1 tile` | 大多已實作 |
| Consumer tail | `producer -> EWE/POOL/TNPS/store` | 已實作，部分已泛化 |
| Layout bridge | `sslice/concat/d2space/transpose -> compute` | Partial；`CONCAT -> FP CONV/DWCONV` consumer 已有 MB，`SPACE_TO_DEPTH -> CONV` true L1 tile-feed 已補，channel-only `SLICE/STRIDED_SLICE -> CONV` cut/final path 已補，true arbitrary `sslice/transpose -> compute` 待補 |
| Fanout/live-range | `one producer -> multiple consumers` | Partial；safe subset 已實作，已包含 immediate pool-head + later skip consumer |
| UDMA as engine | `udma_r/udma_w preload/store` 參與 pipeline | Done |
| Streaming preload | `udma_r head/tail -> compute overlap` | Done；INT8/FP H-tiled CONV/DWCONV 已覆蓋，INT16 仍保守 |

已知未來可能遇到的 layer 組合，應該映射到這些 pattern，而不是再增加新的
hardcoded path：

| 未來可能的 path 形狀 | 應落到的實作 pattern |
|---|---|
| `CONV -> activation -> POOL -> store` | Consumer tail |
| `CONV -> Requant -> CONCAT -> CONV` | Packed concat layout bridge |
| `sslice -> CONV` | Layout bridge with L1 slice-feed |
| `transpose/layout -> FC/CONV` | Layout bridge |
| `producer -> two CONV + one EWE` | Fanout/live-range |
| `POOL -> CONV` | Producer/consumer L1 handoff |
| `TNPS pack -> CONV/EWE` | Layout bridge |
| `Requant output -> multi-consumer live range` | Fanout/live-range |

## 最近驗證

```bash
make -C systemc -s
git diff --check
./batch/run_model.py --model model/MLPerf_Tiny/vww_96_int8.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
make -C systemc -j8
python3 batch/run_mb_path.py --fast-only --rerun-all
python3 batch/run_mb_path.py --filter layout_bridge --fast-only --rerun-all --keep-bin
python3 batch/run_ethz_v6.py --fast-only --rerun-all --filter deeplab_v3_plus_float --limit 1
python3 batch/run_ethz_v6.py --filter dped_float --fast-only --rerun-all --keep-bin
./batch/run_model.py unet_quant --fast-only --keep-intermediate
./batch/run_model.py xlsr_quant --fast-only
./batch/run_model.py midas_v3_quant --fast-only
```

結果：

- `vww_96_int8`: `31/31 PASS`
- `llama2_quant_L35_L74`: `43/43 PASS`
- `mobilenet_v3_quant`: `123/123 PASS`
- `gpt2_quant_L24_L63`: `42/42 PASS`
- `path7_pool_tail_synth`: `3/3 PASS`，`CONV -> ADD -> MAX_POOL`
  觸發 Path 7 pool_tail，三層皆 `tiles=12x1`
- `run_mb_path.py --fast-only --rerun-all`: `27/27 ok`
- `run_mb_path.py --filter layout_bridge --fast-only --rerun-all --keep-bin`:
  `6/6 ok`
- `layout_bridge/dped_float_L1_L2`: `SPACE_TO_DEPTH -> CONV` 已觸發
  true L1 tile-feed，`1.096 ms / fuse=no` -> `1.007 ms / fuse=yes`，
  `mb=104:load+conv+requant+consumer+store`
- `layout_bridge/imdn_float_L5_L6`: channel `STRIDED_SLICE -> CONV` cut
  已觸發 L1 tile-feed，`0.399 ms ok`，
  `mb=42:conv+requant+consumer+store`
- `layout_bridge/xlsr_quant_L8_L9`: channel `STRIDED_SLICE -> CONV` cut
  已觸發 L1 tile-feed，`0.109 ms ok`，
  `mb=8:conv+requant+consumer+store`
- `dped_float` full ETHZ fast profile：`28.660 ms ok`
- `xlsr_quant` full ETHZ fast profile：`0.721 ms ok`；channel slice compute
  intermediate path 已 guard，避免 `0.807 ms` regression
- `deeplab_v3_plus_float` full ETHZ fast profile：`5.632 ms ok`
- FP MB slice 改善：
  - `layout_bridge/deeplab_v3_plus_float_L68_L69`: `0.093 ms / mb=0` -> `0.070 ms / mb=5`
  - `layout_bridge/deeplab_v3_plus_float_L73_L74`: `1.412 ms / mb=0` -> `0.902 ms / mb=64`
  - `producer_compute/deeplab_v3_plus_float_L2_L2`: `0.377 ms` -> `0.285 ms / mb=16`
  - `streaming_preload/deeplab_v3_plus_float_L16_L16`: `0.197 ms` -> `0.131 ms / mb=10`
- `unet_quant`: `27/27 PASS`，`2.311 ms`；L2 maxpool 現在
  `tiles=74x1 PASS`，pool tail 直接接 L1 CONV tile。
- `xlsr_quant`: `41/41 PASS`，`0.721 ms`
- `midas_v3_quant`: `348/348 PASS`，`0.907 ms`
- `make -C systemc -j 8 build/mdla7_model_runner`: OK
- `git diff --check` / `git diff --cached --check`: OK

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

目前下一步若要再加強，可做：

1. Path 7 POOL tail corpus coverage
   需要自然模型或切片覆蓋 `CONV/DW/FC -> binary EWE -> POOL` topology。

2. `sslice/layout -> CONV` full graph profitability
   channel-only `SLICE/STRIDED_SLICE -> CONV` 目前 cut/final path 可用；
   full graph intermediate slice 仍需要 ownership/profitability model，
   避免 `xlsr_quant` 這類模型因 TNPS overhead 多於省下的 DRAM reload 而退步。

3. True TNPS tiled kernels
   目前 `TRANSPOSE/PACK/UNPACK/SPLIT` 對 GraphMeta-confirmed intermediate
   boundary 會做 no-store handoff；任意 layout permute 的真正 tile kernel
   仍是後續工作。

4. Multi-source concat / packed layout bridge cleanup
   目前 concat pointwise path 已可用 Requant direct packed L1 output；
   任意 concat source mix 仍需要更通用的 layout bridge / ownership model。

5. FC partial-K accumulation
   GPT2 L2 已有 FC OC-slice microblock；若要做到真正 K-slice，CONV engine
   需要支援多段 partial sum accumulate，再由最後一段觸發 Requant。

6. Wider fanout ownership model
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
