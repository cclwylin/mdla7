# MDLA7 Handoff

日期時間：2026-05-11 03:16:12 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新本機 commit：`642a641 Expand microblock path slices and ratio chart`
- 最新已 push commit：`31068a7 Remove old MDLA7 system report slides`
- 前一版 code/doc commit：`9e3f70d Strengthen layout microblock handoffs`
- 前一版 microblock commit：`3df0b4b Stream layout slice tails in microblocks`
- FP tiled microblock commit：`8763613 Enable FP tiled conv microblock streaming`
- 前一版 microblock path commit：`239afe7 Optimize TNPS strided slice tail`
- Path 7-10 強化 commit：`d51a330 Strengthen microblock paths 7-10`
- 前一版 microblock tail commit：`a8dd50f Expand microblock fused pipeline tails`
- Gantt task-meta commit：`a490aab Add task-meta microblock Gantt timeline`
- Fast-only runner commit：`09d97cb Implement fused microblock fast-only runners`
- 工作樹目前有本輪 D2S layout-bridge code/doc/profile 修改，另有非本輪的
  `image/`、`reports/uArcSim Implementation by AI.pptx` 本機變更與 Office lock file。
  commit 時不要把非本輪項目混進 microblock code/doc commit。
- 本輪另新增 MDLA6 Pattern Profiles 對照圖：
  `batch/mdla6_pattern_ratio_chart.svg`，由
  `batch/plot_mdla6_pattern_ratio.py` 從
  `batch/output/mdla6_pattern_regression.csv` 產生。Y 軸是 Pattern，
  X 軸標為 `MDLA7 / MDLA6 CX （Cycle Ratio）`，並有 `X=2` 紅色虛線。

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
| 9 | `TNPS/layout -> CONV/FC` / `layout producer -> compute consumer` | Partial+；standalone `CONCAT -> CONV/DWCONV` 的 FP compute consumer 已有 MB，`SPACE_TO_DEPTH -> CONV` 已有 true L1 tile-feed；channel-only `SLICE/STRIDED_SLICE -> CONV` cut/final path 已有；`RESHAPE/TRANSPOSE/CONCAT -> batched FC` final-store safe subset 已有 row x OC microblock；full intermediate 先 guard |
| 10 | `producer -> multiple consumers / concat fanout` | Done safe subset；包含 direct fanout、concat pointwise、Requant direct packed concat、immediate pool-head + later skip consumer；新增更多 Path10 corpus slice 覆蓋 |
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
  - `DEPTH_TO_SPACE -> CONV/DWCONV` layout bridge 已新增 true L1 tile-feed：
    D2S source tile 先載到 L1，TNPS D2S 在 L1 產出 aligned output rows，
    後續 CONV/DW 直接從 L1 row offset 讀取需要的 conv input window。
    已用 `systemc/scripts/gen_d2s_compute_synth.py` 產生 connected synthetic
    graph 驗證，`d2s_compute_synth` 觸發 `tiles=5x1`，D2S intermediate
    DRAM write 為 0，CONV PASS。
  - channel-only `SLICE/STRIDED_SLICE -> CONV/DWCONV` layout bridge 已新增
    L1 tile-feed safe subset：只接受空間維度不變、channel 連續切片、stride=1。
    cut/final consumer 預設啟用；full graph 的 intermediate consumer 目前用
    `MDLA7_EXPERIMENTAL_SLICE_COMPUTE=1` guard，因為 `xlsr_quant` full model
    直接啟用會從 `0.721 ms` 退到 `0.807 ms`。
  - Path9/Path10 已從 `microblock_pattern_candidates.csv` 補切更多代表 slice：
    `layout_bridge` 目前 17 個 slice、`fanout_live_range` 目前 16 個 slice，
    全部 fast-only regression 通過。
  - `RESHAPE/TRANSPOSE/CONCAT -> FC` final-store safe subset 已新增 batched
    FC row x OC microblock。做法是 row tile 留完整 FC output tile 在 L1，
    OC slice 用 Requant strided 寫入 L1，最後以 UDMA_W linear store 出去。
    已讓 `gpt2_quant_L14_L15_layout_bridge` 從 `mb=0` 變 `mb=9` 並 PASS，
    `llama2_quant_L19_L20_layout_bridge` 從 `mb=0` 變 `mb=2` 並 PASS，
    `mobilebert_quant_L7_L8_layout_bridge` 從 `mb=0` 變 `mb=2` 並 PASS。

## Microblock Path Next-Step

後續新增 path 時，先映射到 reusable microblock pattern，不要再往
Path 15、Path 16 這種 hardcoded special-case 擴張。

| Next-Step | Pattern | 目標 |
|---|---|---|
| 1 | Layout bridge | `DEPTH_TO_SPACE -> CONV/DW` true L1 tile-feed 已補；下一步補 `PAD/TRANSPOSE/RESHAPE -> CONV/DW/FC` 更完整 tile-feed，尤其 arbitrary layout permute |
| 2 | Layout bridge profitability | full graph intermediate `SLICE/STRIDED_SLICE -> CONV` 需要 ownership/profitability model；避免 `xlsr_quant` 這類 TNPS overhead 大於省下 DRAM reload 的 regression |
| 3 | Fanout/live-range | 把 safe subset 擴成一般 producer multi-consumer live-range model，涵蓋 `producer -> two CONV + EWE/POOL/TNPS` 與 Requant output fanout |
| 4 | Multi-source concat | cleanup packed concat / multi-source layout bridge，讓 `CONV/Requant -> CONCAT -> CONV` 不只靠個別 case |
| 5 | True TNPS tiled kernels | 將 `TRANSPOSE/PACK/UNPACK/SPLIT/TILE` 從 metadata no-store handoff 推進到可 tile 化的 TNPS kernel |
| 6 | Consumer tail corpus reach | 補自然模型或 synthetic slice 覆蓋 `CONV/DW/FC -> binary EWE -> POOL/TNPS`，尤其 Path 7 pool tail |
| 7 | FC partial-K accumulation | GPT2 L2 已有 FC OC-slice microblock；真正 K-slice 需要 CONV engine psum accumulate descriptor 協定 |
| 8 | INT16 microblock streaming | INT8/FP H-tiled 已可 stream；INT16 目前仍保守，後續要補 INT16 L1/hazard/width 驗證 |

## 最近驗證

```bash
make -C systemc -s
git diff --check
./batch/run_model.py --model model/MLPerf_Tiny/vww_96_int8.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
make -C systemc -j8
python3 batch/run_mb_path.py --fast-only --rerun-all --keep-bin
python3 batch/run_mb_path.py --filter layout_bridge --fast-only --rerun-all --keep-bin
python3 batch/run_mb_path.py --filter fanout_live_range --fast-only --rerun-all --keep-bin
python3 systemc/scripts/gen_d2s_compute_synth.py --output batch/output/d2s_compute_synth.bin
./systemc/build/mdla7_model_runner batch/output/d2s_compute_synth.bin --quiet --l1-timing=fast
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
- `run_mb_path.py --fast-only --rerun-all --keep-bin`: `50/50 ok`
- `run_mb_path.py --filter fanout_live_range --fast-only --rerun-all --keep-bin`:
  `16/16 ok`
- `run_mb_path.py --filter layout_bridge --fast-only --rerun-all --keep-bin`:
  `17/17 ok`
- `d2s_compute_synth`: `2/2 PASS`，`DEPTH_TO_SPACE -> CONV` 觸發
  `tiles=5x1` microblock handoff；D2S layer `DRAM W=0`，summary 為
  `1 fused/streamed`
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
- `python3 batch/plot_mdla6_pattern_ratio.py`: OK；產出
  `batch/mdla6_pattern_ratio_chart.svg`，共 51 筆 pattern，其中 4 筆
  `MDLA7 / MDLA6 CX （Cycle Ratio） >= 2`
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

## 快速命令

```bash
make -C systemc -s
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
python3 batch/plot_mdla6_pattern_ratio.py
```

Profile entry：

```text
batch/profile_hotspot.html
batch/profile_ethz_v6.html
batch/output/<stem>.html
```
