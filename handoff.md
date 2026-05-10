# MDLA7 Handoff

日期時間：2026-05-11 05:37:06 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 工作樹仍有非本輪的 `batch/profile_mb_path.html`、
- `batch/profile_mdla6_pattern.html`、`image/`、
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

## 本輪完成

- `CONV/D2SPACE/POOL -> consumer` 大 tensor handoff guard 已改成 exact
  single-consumer 才保留 no-store，不再因 large upsample/pool tail 把可 fuse
  的 `producer_no_store` 打掉。
- `CONV -> D2SPACE -> CONV/DW/FC` 會讓前段 `CONV -> D2SPACE` defer，
  改由後段 layout/compute handoff 接手，避免 D2SPACE 中間層寫回 DRAM。
- INT16 stream dtype compatibility 放寬為 `INT16x8 <-> INT16x16` 可 L1
  handoff，支援 INT16 `MAXPOOL -> CONV` intermediate no-store。
- INT16 large `MAXPOOL` fallback：非 graph output 時改成 skipped checkpoint，
  不再把 embedded reference 寫回 DRAM；真正 output 邊界仍會 materialize 供
  bit-true 驗證。
- Binary EWE / softmax tail live-range 判斷已補強，`producer -> binary EWE -> softmax` 這類 tail 可保留 no-store。

## Microblock Path Next-Step

後續新增 path 時，先映射到 reusable microblock pattern，不要再往
Path 15、Path 16 這種 hardcoded special-case 擴張。

### ETHZ_V6 DRAM_W 掃描 Next-Step

重新掃描 `model/ETHZ_v6` 對應 `batch/output/*.fast.html`，門檻
`DRAM_W >= 1024 KB` 且排除最後一層：

- 掃到 52/53 個 fast html；缺 `mobilevit_v2_quant.fast.html`。
- `dped_float L65 mul 48 MB` 是 graph output，不列為 intermediate bug。
- 注意：`*.fast.html` 可能 stale。`srgan_float L54 d2spac 32 MB` 與
  `unet_quant L2 maxpool 4 MB` 在最新 `.html/.profile.json` 已是
  `DRAM_W = 0`，但 fast html 仍保留舊數字。

目前真正值得接著看的大 DRAM_W：

| Priority | Model                                       |                      Layer | Op              |  DRAM_W | 初步方向                                             |
| -------: | ------------------------------------------- | -------------------------: | --------------- | ------: | ---------------------------------------------------- |
|        1 | `sd_diffusion_quant`                      | L59/L140/L1318/L1401/L1484 | `mul`         |    8 MB | `mul -> softmax` large tensor tail                 |
|        2 | `pynet_v2_quant`                          |                       L209 | `conv`        |    6 MB | conv output fanout / tail ownership                  |
|        3 | `srgan_quant`                             |                        L55 | `conv`        |    3 MB | 可能是 output/side-output live-range，先查 GraphMeta |
|        4 | `microisp_quant`                          |              L62/L125/L188 | `d2spac`      | 1.99 MB | `D2SPACE -> consumer` layout bridge                |
|        5 | `sam_float`                               |                    L27/L70 | `pad`         | 1.20 MB | `PAD -> compute` layout bridge                     |
|        6 | `sd_encoder_quant` / `sd_decoder_quant` |              L282/L283/L47 | `mul/softmax` |    1 MB | softmax tail streaming / no-store                    |

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

| Pattern               | Count |
| --------------------- | ----: |
| `fanout_live_range` |   286 |
| `producer_compute`  |   134 |
| `layout_bridge`     |    52 |
| `consumer_tail`     |    44 |
| `streaming_preload` |    34 |
| `udma_as_engine`    |     9 |

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
./batch/run_model.py model/ETHZ_v6/srgan_quant.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py model/ETHZ_v6/srgan_float.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py unet_int16 --fast-only --no-build --keep-intermediate
./batch/run_model.py unet_quant --fast-only --no-build --keep-intermediate
./batch/run_model.py llama2_quant.cut --fast-only --no-build --keep-intermediate
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
