# MDLA7 Handoff

日期時間：2026-05-12 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 local commit：`6052c6e Add MDLA7 synth Verilog simulation`。
- 此 commit 尚未確認已 push；之前 GitHub `main` 記錄仍是
  `8e6a1a9 Fuse attention EWE softmax microblocks`。
- 本次 commit 只收 `rtl/` synth Verilog source / runner / README；
  沒有收 `rtl/bin`、`rtl/obj_dir`、`rtl/verilator`。
- 工作樹仍有既有未提交變更在 `batch/`、`systemc/`、`reports/`
  與幾個 generated profile HTML；這些不是本次 RTL commit 的內容。
- 本輪重點：`sd_diffusion_quant` 的 5 組大型 `mul -> softmax`
  attention score tensor 已走 microblock fused path，非最後層 `DRAM_W >= 1MB`
  掃描已清到 0 筆。
- MDLA6 Pattern Profiles 對照圖：
  `batch/chart/mdla6_pattern_ratio_chart.{svg,png}`，由
  `batch/plot_mdla6_pattern_ratio.py` 從
  `batch/output/mdla6_pattern_regression.csv` 產生。X 軸是 Pattern，
  Y 軸是 `MDLA7 / MDLA6 CX （Cycle Ratio）`，Pattern 依 ratio
  由小到大排列，`Y=2` 紅色虛線，只有 `> 2.0` 的點標數字。
- L1 mesh evaluation 圖在
  `batch/chart/mdla6_pattern_mode_ratio_chart.{svg,png}`；標題為
  `MDLA7 L1 Mesh Evaluation (v0.1)`，三條線是 `conflict/fast`、
  `mesh/fast`、`mesh/conflict`，X 軸依 `mesh/conflict` 由小到大排列。

## 2026-05-12 RTL / Synth Verilog Update

已新增並 commit 一套 `rtl/` synth Verilog simulation path：

```text
6052c6e Add MDLA7 synth Verilog simulation
```

### Commit 內容

- `rtl/synth/`：
  `command.v`、`conv.v`、`requant.v`、`ewe.v`、`pool.v`、`tnps.v`、
  `udma.v`、`l1manager.v`、`l1mesh.v`、`mdla7_top.v`、
  `host.v`、`dram.v`、`Testbench.v`、`common.v`、
  `mdla7_dpi.cpp`、`filelist_system_tb.f`、`README.md`。
- `rtl/batch/run_mdla7_verilog.py`：
  quiet Verilator runner，支援 `--compare-synth-verilog`、
  `--show-verilog-output`、`--verbose-build`、macOS Intel / Apple Silicon
  host/architecture check，`rtl/obj_dir` build output，profile timing sidecar。
- `rtl/.gitignore`：
  忽略 `obj_dir/`、`verilator/`、`bin/`、macOS `._*` sidecar、
  `__pycache__`。

### 已實作的 RTL 架構

- Top-level：
  `Testbench.v` instance `host.v`、`dram.v`、`mdla7_top.v`。
  `host.v` 讀 `+PROGRAM=<.bin>`，展開 layer microblock descriptor，
  驅動 `mdla7_top.v`，並用 fast-model reference CRC 檢查 datapath 結果。
- Engine modules：
  `conv/requant/ewe/pool/tnps/udma` 都已 instance 並會動起來。
  datapath correctness 目前透過 Verilator DPI-C `mdla7_dpi.cpp`
  做 true datapath CRC compare；Verilog module 本身是 timing shell +
  phase/handshake/backpressure 模型，不是完整 cycle-accurate arithmetic RTL。
- Engine-side payload token handshake：
  `udma/requant/ewe/pool/tnps` 會在 payload phase 發 `l1_req_valid` 到
  L1Manager；若 L1Manager input FIFO 滿，engine phase 會透過
  `phase_stall` 停住，反映 microblock 執行中的 backpressure。
- CONV path：
  依設計決策，`conv` 直接接 `L1Mesh`，不經 `L1Manager`。
- L1Manager：
  per-source input FIFO 2-deep，source 包含 UDMA / REQUANT / EWE /
  POOL / TNPS / legacy；內建 arbiter，FIFO full 會回壓對應 engine。
  `busy` 包含 FIFO queued token / phase busy / response pending，避免 top
  太早判斷 L1 path drained。
- L1Mesh：
  已實作 `4 x Mesh4x4` 架構。`l1mesh.v` 內有 4 個
  `mdla7_l1mesh4x4_tile` instance；address decode 用 64-bank global bank id：
  `bank_global[5:4]` 選 4 個 tile，`bank_global[3:0]` 選 tile 內 4x4
  node/bank。Timing phase 拆成：
  `ADDR_DECODE -> GLOBAL_MESH -> TILE_MESH -> BANK_ARB -> SRAM_MACRO -> RESP`。

### RTL 驗證結果

最近一輪驗證全部 PASS：

```bash
rtl/verilator/bin/verilator --lint-only --sv --timing -Wall -Wno-fatal \
  -Irtl/synth -f rtl/synth/filelist_system_tb.f --top-module Testbench

rtl/verilator/bin/verilator --lint-only --sv --timing -Wall -Wno-fatal \
  -Irtl/synth rtl/synth/common.v rtl/synth/command.v rtl/synth/conv.v \
  rtl/synth/requant.v rtl/synth/ewe.v rtl/synth/pool.v rtl/synth/tnps.v \
  rtl/synth/udma.v rtl/synth/l1manager.v rtl/synth/l1mesh.v \
  rtl/synth/mdla7_top.v --top-module mdla7_top

./rtl/batch/run_mdla7_verilog.py --compare-synth-verilog --filter '*.bin' \
  --require-host-load --timeout 300 --stop-on-fail --build
```

Full Hotspot bin compare：

```text
summary: pass=11 fail=0 total=11
compare: comparable=11/11 synth_total_ms=3.424 verilog_total_ms=3.326 verilog/synth=0.972x
```

重要：runner 預期 `.bin` 在 `rtl/bin/Hotspot/*.bin`，Verilator checkout 在
`rtl/verilator`，但兩者都被 `.gitignore` 忽略，不在 commit 內。

### RTL Next-Step

1. 將 DPI-C datapath 逐步替換為真正 Verilog datapath：
   先從較小的 `requant/pool/ewe/tnps` arithmetic pipeline 開始，再推進
   `conv` MAC array / psum / tiling。
2. 補 L1Mesh contention test：
   synthetic pattern 讓多 source token 打到同 tile/bank，確認
   `BANK_ARB` / FIFO backpressure / engine `phase_stall` 都能觀察到。
3. 讓 `l1mesh_route_cycles` 更貼近 compiler/profile 的 physical placement，
   目前 4x Mesh4x4 會根據 address tile/bank hops 加 latency，但 placement
   還是簡化模型。
4. 若要 push，先確認是否也要另外 commit/處理現有 `batch/`、`systemc/`、
   `reports/` 未提交變更，避免混進 RTL commit。

## 本輪完成

- 新增 `binary EWE ADD/MUL/SUB -> attention SOFTMAX` microblock fused path：
  `try_stream_binary_ewe_softmax()` 以 L1 ping-pong slot 做
  `load A/B -> EWE -> SOFTMAX -> optional store`，避免 attention score
  matrix 先寫回 DRAM 再由 softmax 讀回。
- `is_attention_softmax_meta()` 共用化 attention softmax 判斷；binary EWE
  producer no-store 規則限定 INT8 且 exact single-consumer，避免 FP16 被標成
  no-store 但還沒有對應 fused path。
- `sd_diffusion_quant` L59/L140/L1318/L1401/L1484 的 8 MB `mul`
  intermediate 已消掉：`mul DRAM_W = 0`，對應 softmax `DRAM_R/W = 0`。
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

## RTL / Synth RTL Next-Step

- `mobilenet_v3_float` 的 `rtl/fast = 1.46` 曾主要卡在 EWE `h_swsh`：
  RTL-style timing 把 HARD_SWISH 算成 4-pass unary compute，導致
  `h_swsh` 累計比 fast 多約 `945k cycles`。這是 timing model 太保守，
  不是 functional datapath 錯；profile 仍 PASS。
- 設計決策：HARD_SWISH 要視為 EWE 硬體 fused pipeline，
  throughput 以 lane pipeline 為主，不應拆成 add/clamp/mul/scale 四個
  full tensor pass。RTL-style model 已把 `ES_HARD_SWISH` 改成 fused
  1-pass timing；`GELU/LOGISTIC/SOFTMAX` 保留各自多 pass / LUT /
  reduction 行為。`mobilenet_v3_float` smoke: `rtl/fast` 由 `1.46`
  降到 `1.17`。
- Synthesizable RTL TODO：實作 EWE HARD_SWISH pipeline datapath：
  `fp16 input -> fp32 convert -> x+3 -> clamp[0,6] -> multiply x -> scale 1/6
  -> clip -> fp16 output`。目標是 initiation interval 1 element/lane，
  latency 用 pipeline depth 表示，不用 full tensor multi-pass 表示。

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
|        1 | `pynet_v2_quant`                          |                       L209 | `conv`        |    6 MB | conv output fanout / tail ownership                  |
|        2 | `srgan_quant`                             |                        L55 | `conv`        |    3 MB | 可能是 output/side-output live-range，先查 GraphMeta |
|        3 | `microisp_quant`                          |              L62/L125/L188 | `d2spac`      | 1.99 MB | `D2SPACE -> consumer` layout bridge                |
|        4 | `sam_float`                               |                    L27/L70 | `pad`         | 1.20 MB | `PAD -> compute` layout bridge                     |
|        5 | `sd_encoder_quant` / `sd_decoder_quant` |              L282/L283/L47 | `mul/softmax` |    1 MB | softmax tail streaming / no-store                    |

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
./batch/run_model.py sd_diffusion_quant --fast-only --no-build --keep-intermediate
./batch/run_model.py midas_v3_quant --fast-only --no-build --keep-intermediate
./batch/run_model.py llama2_quant.cut --fast-only --no-build --keep-intermediate
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

最近一輪結果：

- `sd_diffusion_quant`：`1524/1524 PASS`，`4551105 cycles = 2.395 ms`。
- `sd_diffusion_quant` 非最後層 `DRAM_W >= 1MB`：0 筆。
- `midas_v3_quant`：`348/348 PASS`。
- `llama2_quant`：`400/400 PASS`。

## 快速命令

```bash
make -C systemc -s
./batch/run_model.py --model model/Hotspot/gpt2_quant_L24_L63.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/Hotspot/llama2_quant_L35_L74.tflite --fast-only --no-build --keep-intermediate
./batch/run_model.py --model model/ETHZ_v6/mobilenet_v3_quant.tflite --fast-only --no-build --keep-intermediate
python3 batch/plot_mdla6_pattern_ratio.py
```

ETHZ_v6 runner usage：

```bash
# Pure analytical fast model
./batch/run_ethz_v6.py --fast-only
./batch/run_ethz_v6.py --filter mobilebert_quant --fast-only --rerun-all

# RTL-style engine fast model
./batch/run_ethz_v6.py --rtl-fast
./batch/run_ethz_v6.py --filter mobilebert_quant --rtl-fast --rerun-all

# Compare pure fast vs RTL fast and emit combined CSV/HTML
./batch/run_ethz_v6.py --compare-rtl-fast
./batch/run_ethz_v6.py --filter resnet_quant --limit 1 --compare-rtl-fast --rerun-all
```

ETHZ_v6 output paths：

```text
batch/output/ethz_v6_regression.csv
batch/output/ethz_v6_regression.rtl.csv
batch/output/ethz_v6_regression.rtl_compare.csv
batch/profile_ethz_v6.html
batch/profile_ethz_v6.rtl.html
batch/profile_ethz_v6.rtl_compare.html
batch/output/<model>.html
batch/output/<model>.rtl.html
batch/output/<model>.rtl_compare.html
batch/output/<model>.rtl.profile.json
```

Profile entry：

```text
batch/profile_hotspot.html
batch/profile_ethz_v6.html
batch/output/<stem>.html
```
