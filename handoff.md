# MDLA7 Handoff

日期時間：2026-05-12 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 local commit：`e7e9c1e Tag L1 responses with source and stream tid`。
- 此 commit 尚未確認已 push；之前 GitHub `main` 記錄仍是
  `8e6a1a9 Fuse attention EWE softmax microblocks`。
- 近期 RTL commits：
  `6052c6e` 建立 synth/verilog_ctrl path；
  `8d9a1f0` 修 slice runner / TNPS datapath；
  `18ed74d` 整理 generated profile 目錄；
  `246fda9` 新增 `run_verilog_ctrl.py` / `run_verilog_final.py`；
  `55be55e` 新增 EWE sample path；
  `c130339` 擴充 verilog_final sample datapath；
  `3a4d11e` 新增 INT16 CONV/EWE/POOL sample datapath。
- RTL commits 沒有收 `rtl/bin`、`rtl/obj*`、`rtl/verilator`。
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

## 2026-05-12 Verilog Final Streaming / L1 Response Handoff

最新完成並已 commit：

```text
e7e9c1e Tag L1 responses with source and stream tid
3305dde Probe requant L1 producer path
1f7aa93 Feed UDMA store CRC from L1 response
a26c5ad Check microblock output SRAM CRC
d544211 Add microblock final tensor CRC probes
1627ffd Drive verilog final with microblock descriptors
```

### 目前 verilog_final 狀態

- `run_verilog_final.py` default 已走 microblock descriptors；`--sample-descriptors`
  才回到舊 sample descriptor path。
- generator 會在 microblock sequence 後加 tensor coverage probes：
  - UDMA seed L1 bytes -> UDMA STORE from L1 response -> output SRAM CRC。
  - REQUANT producer 寫 L1 -> L1CRC 驗 `[requant_out] + zeros`。
  - UDMA ref-fill output SRAM image -> full output SRAM CRC / final CRC。
- UDMA STORE path 已可從 L1Mesh read response 寫入 UDMA output SRAM。
  `vf_udma_engine` 新增 L1 response ports，`final_write_mode && direction_write`
  會等 read response，再把 16B line 寫到 `output_sram[out_byte_offset + lane]`。
- UDMA SRAMCRC descriptor 不再打一筆 L1 request；它只掃 UDMA output SRAM。
- L1Manager/L1Mesh response metadata 已打通：
  - L1Manager 將 arbitration 選到的 `source/tid` 帶到 mesh request。
  - L1Mesh response 回傳 `resp_read/resp_source/resp_tid`。
  - `mdla7_top_final` 只把符合當前 engine source 且 `tid == stream_slot_q`
    的 read response 餵給 REQUANT/POOL/EWE/UDMA。
- 已觀察到的 hazard：L1Mesh response/data 在 back-to-back request 時可能有
  stale beat / metadata-data 同拍風險。`resp_source/resp_tid` 是必要地基，
  但 REQUANT -> UDMA consumer probe 還需要 per-command response queue 或
  更嚴格的 response-valid/data register 才能穩定接回。

### 最近驗證

已驗證 PASS：

```bash
rm -rf rtl/obj/verilog_final/host
./rtl/batch/run_verilog_final.py --filter slice --limit 1 \
  --rerun-all --require-crc-coverage --require-final-output-crc

./rtl/batch/run_verilog_final.py --filter slice --limit 3 \
  --rerun-all --no-build --require-crc-coverage --require-final-output-crc
```

最近結果：

```text
limit 1: pass=1 fail=0 skip=0 sample_only=0 total=1
coverage: refcrc=0 sramcrc=3 finalcrc=2 refB=0 sramB=16777248 finalB=16777232

limit 3: pass=3 fail=0 skip=0 sample_only=0 total=3
coverage: refcrc=0 sramcrc=12 finalcrc=7 refB=0 sramB=23069440 finalB=23069360
```

### 下一步

1. 在 L1 response side 加 per-command response queue / skid register：
   response valid、read bit、source、tid、rdata 必須同拍鎖住，避免 back-to-back
   request 時 stale `resp_rdata` 被新 command source/tid 吃掉。
2. 把暫時沒有 commit 的 consumer probe 接回：
   `REQUANT producer writes L1 -> UDMA STORE reads same L1 address -> UDMA output SRAM CRC`。
   先用 16B `[requant_out] + zeros`，通過後再擴成多 byte/tile。
3. 同樣模式推到 POOL/EWE/TNPS：
   producer engine result byte/line 寫 L1，consumer/UDMA 從同一 L1 address 讀，
   再做 output SRAM CRC。
4. 接 full tensor path：
   將目前 ref-fill full output CRC 逐步替換成真正 producer output SRAM/L1Mesh
   image，再做 full output tensor compare/CRC。
5. 注意 Verilator include dependency：
   `rtl/synth/*.v` 被 include 時，`run_verilog_final_smoke.py` 可能
   `make: Nothing to be done`。改 L1Manager/L1Mesh 後建議先
   `rm -rf rtl/obj/verilog_final/host` 再跑 host smoke/regression。

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
- `rtl/batch/run_verilog_ctrl.py`：
  verilog_ctrl quiet Verilator runner，支援 `--compare-synth-verilog_ctrl`、
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

./rtl/batch/run_verilog_ctrl.py --filter '*.bin' \
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

## 2026-05-12 Verilog Ctrl / Final Update

### Runner naming

- `rtl/batch/run_verilog_ctrl.py`：control/timing shell regression path，
  default compare mode 是 `--compare-synth-verilog_ctrl`。
- `rtl/batch/run_verilog_final.py`：full Verilog datapath bring-up path，
  目前是 sample descriptor + real Verilog sample datapath，不是完整 tensor
  tile streaming。
- 舊的 `run_mdla7_verilog.py` 已移除，避免和 ctrl/final 命名混淆。
- `rtl/obj_dir` 是 verilog_ctrl 既有 build output；verilog_final 產物走
  `rtl/obj/verilog_final/`。後續若整理 obj 目錄，維持 ctrl/final 分開。

### verilog_ctrl 狀態

- `rtl/synth` 仍是 `verilog_ctrl` target，datapath correctness 透過
  DPI-C / fast-model CRC 比對，RTL module 主要提供 control、microblock、
  L1Manager/L1Mesh timing/backpressure。
- `rtl/synth/host.v` 的 capacity 已在 `3a4d11e` 放大：
  `MAX_PROGRAM_BYTES=262144`、`MAX_LAYERS=4096`。
  這修掉 `mobilebert_quant` huge pattern 的錯誤：

```text
$readmem file address beyond bounds of array
mobilebert_quant.timing.hex line 1024
```

- `mobilebert_quant.bin` header layer count 是 `0x734 = 1844`，舊
  `MAX_LAYERS=1024` 不夠。重建 `rtl/obj_dir/VTestbench` 後 bounds error
  已消失；用手動 `--timeout 120` 會變成正常長跑 timeout，實際 regression
  應用 huge default `3600s`。

### verilog_final 狀態

`rtl/verilog_final` 已有以下 sample datapath：

- CONV：INT8 MAC + bias + MBQM + clamp；FP16 real-valued sample MAC；
  INT16/hybrid signed sample MAC；`vf_conv2d_addrgen` 2D NHWC address-walk
  primitive。
- REQUANT：MBQM + output zero-point + activation clamp sample。
- POOL：INT8 max/avg；FP16 real-valued max/avg；INT16/hybrid signed max/avg。
- EWE：INT8 ADD/MUL/SUB；FP16 ADD/MUL/SUB/LOGISTIC；INT16 ADD/MUL/SUB。
- TNPS：SPACE_TO_DEPTH / DEPTH_TO_SPACE address mapping。
- UDMA：byte-moving timing token path。
- L1Manager/L1Mesh：multi-source token path、2-deep FIFO backpressure、
  placement-aware route estimator、contention smoke test。

Descriptor generator：

```bash
./rtl/batch/gen_verilog_final_program.py <program.bin> -o <program.final.hex>
```

Regression examples：

```bash
./rtl/batch/run_verilog_final_smoke.py
./rtl/batch/run_verilog_final.py --filter dped_int16_L3_6.bin --rerun-all
./rtl/batch/run_verilog_final.py --filter microisp_int16_L5_20.bin --rerun-all
./rtl/batch/run_verilog_final.py --filter slice --limit 10 --rerun-all
```

Recent targeted results:

```text
dped_int16_L3_6        PASS cmds=4 conv=2 pool=0 requant=0 ewe=2 tnps=0 udma=0 done=4
microisp_int16_L5_20   PASS cmds=9 conv=6 pool=1 requant=0 ewe=1 tnps=0 udma=1 done=9
ETHZ_v6_slice census   PASS 175 fail=0 skip=0 total=175
```

Smoke:

```text
./rtl/batch/run_verilog_final_smoke.py
[verilog_final_smoke] PASS
./rtl/batch/run_verilog_final_smoke.py --test conv
PASS: verilog_final conv int8 MAC datapath and 2D address walk
```

### verilog_final descriptor flags

- CONV word 12：bit8 `fp_mode`、bit11 `int16_mode`。
- POOL word 12：bit8 `avg_mode`、bit9 `fp_mode`、bit11 `int16_mode`。
- EWE word 12：bits9:8 `op_mode`、bit10 `fp_mode`、bit11 `int16_mode`。
- FP descriptors use words 16/17 for expected double result bits.
- INT16 descriptors use word 18 as signed 32-bit expected sample result.

### verilog_final Next-Step

1. `ETHZ_v6_slice` 已跑到 `pass=175 fail=0 skip=0 total=175`，所以目前
   不是補 SKIP descriptor，而是往更真 datapath 前進。
2. 下一步把 `vf_conv2d_addrgen` 接進 `vf_conv_sample_engine` 與 generated
   CONV descriptors，讓 final path 從 flat payload sample MAC 變成
   descriptor-driven 2D window sample + MAC。
   需要在 final CONV descriptor 帶最小 shape/window info：
   `in_h/in_w/in_c`、`out_h/out_w/out_c`、`k_h/k_w`、stride/dilation/pad
   或先用 default；host 比對 expected input/weight/output byte offset 與
   valid bit。
3. 再推 CONV multi-sample / multi-output tile walk：activation/weight tile walk、
   psum accumulate、writeback buffering、CRC/full tensor compare。
4. 補更完整 byte-moving / layout descriptor：`RESHAPE/CONCAT/SLICE` 目前多
   還是 UDMA-style placeholder，後續要接到 TNPS/meta shape ports。
5. 將 FP sample path 從 Verilator real-valued bring-up primitive 逐步替換成
   synthesizable FP pipeline 或明確分層成 simulation-only checker。

### Docs / textbook update

- 新增 `md/22_rtl_bringup.md`，正式成為 textbook 第 22 章。
- 已重生：
  `md/mdla7_textbook.md`、`md/mdla7_textbook.html`、
  `pdf/mdla7_textbook.pdf`。
- PDF 產生命令：

```bash
bash scripts/build_pdf.sh
```

- 若 `/tmp/mdpdf_venv/bin/python` 不存在，可用：

```bash
python3 -m venv --system-site-packages /tmp/mdpdf_venv
```

  目前機器上已用這個方式建立 venv，build script 可正常跑。

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
