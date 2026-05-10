# MDLA7 Handoff

日期時間：2026-05-11 04:36:24 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新本機 commit：`72088dc Fix fast-mode microblock pipeline correctness`
- 最新已 push commit：`31068a7 Remove old MDLA7 system report slides`
- 前一版 code/doc commit：`9e3f70d Strengthen layout microblock handoffs`
- 前一版 microblock commit：`3df0b4b Stream layout slice tails in microblocks`
- FP tiled microblock commit：`8763613 Enable FP tiled conv microblock streaming`
- 前一版 microblock path commit：`239afe7 Optimize TNPS strided slice tail`
- Path 7-10 強化 commit：`d51a330 Strengthen microblock paths 7-10`
- 前一版 microblock tail commit：`a8dd50f Expand microblock fused pipeline tails`
- Gantt task-meta commit：`a490aab Add task-meta microblock Gantt timeline`
- Fast-only runner commit：`09d97cb Implement fused microblock fast-only runners`
- 工作樹仍有非本輪的 `batch/profile_mb_path.html`、
  `batch/profile_mdla6_pattern.html`、`image/`、
  `reports/uArcSim Implementation by AI.pptx` 本機變更。commit 時不要混入。
- 本輪另新增 MDLA6 Pattern Profiles 對照圖：
  `batch/chart/mdla6_pattern_ratio_chart.{svg,png}`，由
  `batch/plot_mdla6_pattern_ratio.py` 從
  `batch/output/mdla6_pattern_regression.csv` 產生。X 軸是 Pattern，
  Y 軸是 `MDLA7 / MDLA6 CX （Cycle Ratio）`，Pattern 依 ratio
  由小到大排列，`Y=2` 紅色虛線，只有 `> 2.0` 的點標數字。
- L1 mesh evaluation 圖在
  `batch/chart/mdla6_pattern_mode_ratio_chart.{svg,png}`；標題為
  `MDLA7 L1 Mesh Evaluation (v0.1)`，三條線是 `conflict/fast`、
  `mesh/fast`、`mesh/conflict`，X 軸依 `mesh/conflict` 由小到大排列。

## Microblock Path Next-Step

後續新增 path 時，先映射到 reusable microblock pattern，不要再往
Path 15、Path 16 這種 hardcoded special-case 擴張。

## MB_Path_Slice Pattern Corpus

`model/MB_Path_Slice/microblock_pattern_candidates.csv` 已整理成
implementation representative set。流程是先用舊規則
`pattern + op_sequence + input_shape` normalize，再用 `(model, op_sequence)`
做全域代表去重；跨分類重疊時優先保留較 specific 的 pattern：

```text
consumer_tail -> layout_bridge -> fanout_live_range ->
streaming_preload -> udma_as_engine -> producer_compute
```

目前候選從原本 2922 筆壓到 559 筆，`microblock_pattern_slices.csv`
也同步濾成同一組 559 筆代表 slice。

| Pattern | Count |
|---|---:|
| `fanout_live_range` | 286 |
| `producer_compute` | 134 |
| `layout_bridge` | 52 |
| `consumer_tail` | 44 |
| `streaming_preload` | 34 |
| `udma_as_engine` | 9 |

相關腳本：

```bash
python3 systemc/scripts/scan_microblock_patterns.py
python3 systemc/scripts/scan_microblock_patterns.py --dedupe-key pattern_op_sequence_input_shape
python3 systemc/scripts/slice_microblock_patterns.py --max-per-pattern 0 --clean-output
```

詳細說明在 `model/MB_Path_Slice/README.md`。

| Next-Step | Pattern                     | 目標                                                                                                                                                                  |
| --------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1         | Layout bridge               | `DEPTH_TO_SPACE -> CONV/DW` true L1 tile-feed 已補；下一步補 `PAD/TRANSPOSE/RESHAPE -> CONV/DW/FC` 更完整 tile-feed，尤其 arbitrary layout permute                |
| 2         | Layout bridge profitability | full graph intermediate `SLICE/STRIDED_SLICE -> CONV` 需要 ownership/profitability model；避免 `xlsr_quant` 這類 TNPS overhead 大於省下 DRAM reload 的 regression |
| 3         | Fanout/live-range           | 把 safe subset 擴成一般 producer multi-consumer live-range model，涵蓋 `producer -> two CONV + EWE/POOL/TNPS` 與 Requant output fanout                              |
| 4         | Multi-source concat         | cleanup packed concat / multi-source layout bridge，讓 `CONV/Requant -> CONCAT -> CONV` 不只靠個別 case                                                             |
| 5         | True TNPS tiled kernels     | 將 `TRANSPOSE/PACK/UNPACK/SPLIT/TILE` 從 metadata no-store handoff 推進到可 tile 化的 TNPS kernel                                                                   |
| 6         | Consumer tail corpus reach  | 補自然模型或 synthetic slice 覆蓋 `CONV/DW/FC -> binary EWE -> POOL/TNPS`，尤其 Path 7 pool tail                                                                    |
| 7         | FC partial-K accumulation   | 真正 K-slice 需要 CONV engine psum accumulate descriptor 協定                                                                                                         |
| 8         | INT16 microblock streaming  | INT8/FP H-tiled 已可 stream；INT16 目前仍保守，後續要補 INT16 L1/hazard/width 驗證                                                                                    |

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
