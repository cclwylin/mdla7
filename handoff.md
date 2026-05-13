# MDLA7 Handoff

## Project Target

- 最後要做到 FPGA。
- 用 Fast/Synth mode 驗證 Compile、Function、Performance。
- Verilog 的 arithmetic 加速用 DPI。
- Synth 跟 Verilog correlation 誤差不能超過 10%。

Date: 2026-05-13 CST
Repo: `/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch: `main`

## Update 2026-05-14 Unsupported / CX / Verilog

- BMM and ETHZ_v6 unsupported-op audit is now clean:
  `systemc/scripts/audit_unsupported_ops.py model/BMM model/ETHZ_v6` reports
  BMM `0/3` and ETHZ_v6 `0/53` models with unsupported ops.
- Materialized fallback remains explicitly reported as supported-but-not-native;
  it is not native RTL datapath coverage. Use `--strict-native` when fallback
  should fail the audit.
- Added forced materialized boundaries for correctness holes found by ETHZ CX:
  `unet_float` L1-L13, `imdn_float` L1, `dped_float` L1, and
  `efficientnet_b4_float` L473/L474, in addition to the earlier
  `inception_v3_float` L2/L5 and `unet_quant` L8/L14 exceptions.
- Fixed `OK_MATERIALIZE` execution in the SystemC runner so fallback reference
  bytes are staged at `dram_out` with a self-copy descriptor. Staging at
  `dram_in` could alias the previous layer output and caused
  `sd_encoder_quant` materialized layers to fail.
- Fixed producer-no-store handling for explicit materialized boundaries and for
  binary EWE consumers (`ADD`/`MUL`/`SUB`). BMM SAM failed when the scale `MUL`
  store was suppressed before the mask `ADD`.
- Fixed Verilog host cycle reporting for closed-loop probe descriptors. BMM
  full final coverage uses final/check probes; excluding those from
  `measured_cycle_count` made real PASS runs report `verilog_cyc=0`.

Validation after these fixes:

- `python3 -m py_compile systemc/scripts/compile_model.py batch/gen_verilog_program.py batch/run_verilog.py batch/run_systemc.py`
  passes.
- `make -C systemc -j$(sysctl -n hw.ncpu) build/mdla7_model_runner` passes.
- `systemc/scripts/audit_unsupported_ops.py model/BMM model/ETHZ_v6`
  -> BMM `0/3` unsupported; ETHZ_v6 `0/53` unsupported.
- `./batch/run_systemc.py --filter bmm --fast-only --rerun-all --no-html`
  -> BMM `3/3` clean.
- `./batch/run_systemc.py --filter bmm --cx --fast-only --rerun-all --no-html`
  -> BMM `3/3` clean.
- `./batch/run_verilog.py --filter bmm --rerun-all --timeout 180`
  -> BMM `3/3` PASS with full final coverage, `verilog/cx=0.44x` aggregate.
- `./batch/run_verilog.py --filter rtl/bin/ETHZ_v6/midas_v3_quant.bin --rerun-all --timeout 240 --no-build`
  -> `midas_v3_quant` PASS with full final coverage.
- ETHZ_v6 CX was validated in chunks while fixing the failures. After the final
  conservative no-store and Verilog cycle fixes, targeted clean reruns include
  `efficientnet_b4_float`, `imdn_float`, and `dped_float`; earlier targeted
  clean reruns covered `sd_encoder_quant` and `unet_float`. A single monolithic
  53-model rerun was not repeated after the final conservative no-store guard.

## Latest Handoff 2026-05-13

Current active goal:

- Fix the old false-pass problem: ETHZ/BMM regressions must not report clean
  PASS/ok when `compile_model.py` skipped unsupported original TFLite ops.
- Fill unsupported ETHZ/BMM ops through all required layers:
  compiler lowering, SystemC fast/cx execution, and Verilog closed-loop
  descriptor/datapath coverage.
- A model is only a clean SystemC regression pass when every original TFLite op
  is compiled/supported and the compiled graph verifies. `compile-skipped:N`
  is now a regression failure, not a warning.
- Verilog BMM must not count sampled/partial closed-loop coverage as normal
  pass; BMM closed-loop requires full final output coverage.

Latest implementation status:

- `compile_model.py` now reads compatible constant tensor inputs for binary
  ops instead of always synthesizing RNG input-B. If a constant cannot be
  reshaped/broadcast to the synthetic fallback shape, it safely falls back to
  the old deterministic synthetic path.
- Large FP binary ops with scalar input-B now use a compact scalar-broadcast
  weight payload. SystemC EWE detects the marker and expands the scalar in
  compute, avoiding duplicated multi-MB B tensors in DPED-style programs.
- Program images still write v3 when offsets fit. v4 64-bit offset support was
  added to the compiler, SystemC runner, and Verilog parser for future >4GB
  cases; current `dped_float` fits v3 again after scalar compaction.
- DPED `dped_float` now compiles all 103/103 layers, no `compile-skipped`.
  The old failure was not unsupported op coverage; it was a combination of
  huge program storage, scalar-B duplication, and a store-skip barrier
  corrupting aliased DRAM input by one byte.
- SystemC runner now prevents skipped producer stores in front of large FP
  binary/D2SPACE consumers that reload aliased input from DRAM. This fixes the
  one-byte DPED tail corruption and keeps final output comparison meaningful.

Validation from this handoff:

- `./batch/run_systemc.py --filter bmm --cx --fast-only --rerun-all --no-html`
  -> BMM `3/3` clean.
- `./batch/run_verilog.py --filter bmm --rerun-all --timeout 180`
  -> BMM `3/3` PASS, full final coverage.
- `./batch/run_systemc.py --filter ethz_v6 --model-filter dped_float --cx --fast-only --rerun-all --no-html`
  -> `dped_float` clean, `cx=42.19 ms`, `103/103` SystemC layers PASS in the
  underlying runner.
- `./batch/run_verilog.py --filter rtl/bin/ETHZ_v6/midas_v3_quant.bin --rerun-all --timeout 240 --no-build`
  -> `midas_v3_quant` PASS, full coverage.
- `systemc/scripts/audit_unsupported_ops.py model/BMM model/ETHZ_v6`
  -> BMM `0/3` unsupported; ETHZ_v6 `0/53` unsupported. Materialized fallback
  counts remain reported separately and are not native RTL coverage.

Unsupported-op inventory snapshot:

- `systemc/scripts/audit_unsupported_ops.py model/BMM model/ETHZ_v6`
  now reports:
  - BMM: `0/3 models have unsupported ops`
  - ETHZ_v6: `0/53 models have unsupported ops`
- The same audit now also reports supported-but-not-native materialized
  fallback ops so `unsupported=0` cannot be mistaken for native RTL coverage:
  - BMM: `3/3` models use `matrlz`
    (`BATCH_MATMUL=6`, `GELU:non-fp-dtype=1`, `RSQRT=3`,
    `SQUARED_DIFFERENCE=2`)
  - ETHZ_v6: `46/53` models use `matrlz`
    (`BATCH_MATMUL=258`, `CAST=1`, `GELU:non-fp-dtype=77`, `GREATER=2`,
    `HARD_SWISH:non-fp-dtype=61`, `LEAKY_RELU=246`,
    `LOGISTIC:non-fp-dtype=346`, `MINIMUM=2`, `PRELU=296`,
    `QUANTIZE=236`, `RELU=30`, `RESIZE_BILINEAR=36`,
    `RESIZE_NEAREST_NEIGHBOR=20`, `RSQRT=392`,
    `SQUARED_DIFFERENCE=238`, `SUM=18`, `TANH=24`,
    `TRANSPOSE_CONV=23`)
- Use `systemc/scripts/audit_unsupported_ops.py --strict-native ...` when
  materialized fallbacks should fail the run. Example: BMM currently exits
  non-zero in strict-native mode because `BATCH_MATMUL` is still `matrlz`.

Implementation note:

- Newly cleared ops are lowered by `compile_model.py` as materialized reference
  byte boundaries unless they already had a native MDLA7 op path.
- This is intentionally not hidden as a hardware arithmetic datapath. It means
  compiler coverage, SystemC fast/cx execution, and Verilog final-byte coverage
  are present; later performance/RTL work can replace selected `matrlz` layers
  with native engines.
- Covered formerly-unsupported ops include:
  `BATCH_MATMUL`, `CAST`, `GREATER`, `LEAKY_RELU`, `MINIMUM`, `PRELU`,
  `QUANTIZE`, `RELU`, `RESIZE_BILINEAR`, `RESIZE_NEAREST_NEIGHBOR`,
  `RSQRT`, `SQUARED_DIFFERENCE`, `SUM`, `TANH`, `TRANSPOSE_CONV`.
- `HARD_SWISH`, `GELU`, and `LOGISTIC` keep their FP native path; quantized/int
  variants are explicitly audited as dtype materialized fallbacks.

New BMM assets:

- `model/BMM/bmm_softmax_bmm_fp32.tflite`
- `model/BMM/bmm_softmax_bmm_int8.tflite`
- `model/BMM/bmm_softmax_bmm_sam_quant_L22_L61.tflite`
- `model/BMM/README.md`
- `systemc/scripts/gen_bmm_tflite_models.py`

Literal BMM TFLite op sequence:

```text
BATCH_MATMUL, MUL, SOFTMAX, BATCH_MATMUL
```

Important caveat:

- Unsupported ops are still surfaced in compile logs / HTML as
  `skipped (unsupported op)` and SystemC corpus status becomes
  `compile-skipped:N`.
- `run_systemc.py` exits non-zero if any row status is not clean `ok`.
- BMM no longer relies on sampled final coverage in Verilog; BMM runs pass only
  when the final layer's full reference byte range is checked.

Runner changes:

- `batch/run_systemc.py` now has `--filter bmm`, mapped to `model/BMM`.
- `batch/run_systemc.py --cx` now uses `.cx.*` naming, e.g.
  `batch/profile/profile_bmm.cx.html` and
  `batch/output/bmm_softmax_bmm_sam_quant_L22_L61.cx.html`.
- `batch/run_verilog.py` now has `--filter bmm`, mapped to `rtl/bin/BMM`.
- Because `rtl/bin` is gitignored, `run_verilog.py --filter bmm` auto-compiles
  `model/BMM/*.tflite` into `rtl/bin/BMM/*.bin` when needed. It also rebuilds
  those bins when `compile_model.py` is newer than the cached `.bin`.
- `run_verilog.py` also refreshes direct `rtl/bin/ETHZ_v6/*.bin` paths with the
  current compiler. If current compile fails, the run aborts instead of using a
  stale old `.bin` and reporting PASS.
- `batch/gen_verilog_program.py --full-final-ref` emits a full final-reference
  UDMA fill + CRC check; `run_verilog.py` enables it for BMM.
- `batch/gen_verilog_program.py --check-materialized-layers` emits compact
  ref-fill + SRAM CRC checks for every `OK_MATERIALIZE` layer; `run_verilog.py`
  enables this automatically for BMM and ETHZ_v6 bins.
- `batch/run_verilog_smoke.py` detects host-architecture-stale Verilator
  build products and rebuilds them instead of failing with `Bad CPU type`.
- `.gitignore` now allows `model/BMM/*.md` and `model/BMM/*.tflite` to be tracked.
- `TRANSPOSE_CONV` materialized lowering now handles TFLite OHWI filter layout.
  This fixed the old `midas_v3_quant` reshape crash.

Current expected BMM status:

```bash
./batch/run_systemc.py --filter bmm --rerun-all --cx --fast-only
```

Result:

```text
clean 3/3
bmm_softmax_bmm_fp32               ok
bmm_softmax_bmm_int8               ok
bmm_softmax_bmm_sam_quant_L22_L61  ok
```

```bash
./batch/run_verilog.py --filter bmm --rerun-all
```

Result:

```text
pass=3 fail=0 skip=0 total=3
coverage: full, sramcrc=22, finalcrc=3, sramB=3013192, finalB=19200
comparable=0/3
```

`./batch/run_verilog.py --filter bmm --rerun-all --dpi --timeout 180` has the
same correctness result: `pass=3 fail=0 skip=0`, full coverage,
`sramcrc=22`, `finalcrc=3`.

Interpretation: this is full final-byte closed-loop correctness coverage plus
materialized-boundary CRC coverage, not a native BMM performance datapath.
`verilog/cx` is blank because the BMM row is not performance-comparable until
native traversal/cycles exist.

Additional spot checks after the materialized/native audit split:

```text
./batch/run_systemc.py --filter ethz_v6 --model-filter mobilenet_v3_quant --cx --fast-only --rerun-all --no-html
  clean 1/1, ok
./batch/run_systemc.py --filter ethz_v6 --model-filter mobilebert_quant --cx --fast-only --rerun-all --no-html
  clean 1/1, ok
./batch/run_systemc.py --filter ethz_v6 --model-filter midas_v3_quant --cx --fast-only --rerun-all --no-html
  clean 1/1, ok
~/.venvs/mdla7/bin/python systemc/scripts/compile_model.py model/ETHZ_v6/midas_v3_quant.tflite /tmp/midas_v3_quant.current.bin
  383 layers, no skipped rows, current TRANSPOSE_CONV lowering passes
python3 batch/gen_verilog_program.py /tmp/midas_v3_quant.current.bin -o /tmp/midas_v3_quant.current.verilog.hex --max-commands 2048 --check-materialized-layers
  commands=70 sramcrc=35 srambytes=4578132 finalcrc=1 finalbytes=50176
./batch/run_verilog_smoke.py --test host --program /tmp/midas_v3_quant.current.verilog.hex --ref-program /tmp/midas_v3_quant.current.bin --no-build
  PASS issued=70 done=70
./batch/run_verilog.py --filter rtl/bin/ETHZ_v6/midas_v3_quant.bin --rerun-all --timeout 240 --no-build
  PASS full, sramcrc=35, finalcrc=1, sramB=4578132, finalB=50176
./batch/run_verilog.py --filter g1op_ethz --rerun-all --timeout 90 --check-materialized-layers
  pass=25 fail=0 skip=0 total=25, sramcrc=55, finalcrc=55
```

Do not treat old `rtl/bin/ETHZ_v6/*.bin` files from May 12 as proof. They were
built before the current compiler fixes. `run_verilog.py` now refuses stale-bin
PASS when a required refresh compile fails.

Known failing/next item:

```bash
./batch/run_verilog.py --filter bmm --dpi --cycle 500k --rerun-all --timeout 120
```

Current result:

```text
bmm_softmax_bmm_fp32               FAIL stream drain timeout issued=7 done=6
bmm_softmax_bmm_int8               FAIL stream drain timeout issued=7 done=6
bmm_softmax_bmm_sam_quant_L22_L61  FAIL stream drain timeout issued=1118 done=69
```

Interpretation:

- BMM corpus connection is working.
- Closed-loop correctness path passes.
- Cycle-only stream path has a done/drain accounting or completion issue for these streams.

Latest committed chip-level MB overlap work:

- `4845071 Add chip-level microblock overlap scheduler`
- Verified before commit:
  - `./batch/run_verilog.py --filter g1op --dpi --cycle 10k --rerun-all --timeout 60`: 40/40 PASS
  - `./batch/run_verilog.py --filter sd_diffusion_quant_L71_72 --dpi --cycle --rerun-all --timeout 120`: PASS
  - `./batch/run_verilog.py --filter llama2_quant_L41_42 --dpi --cycle --rerun-all --timeout 120`: PASS
  - `./batch/run_verilog.py --filter g1op_conv2d_int8 --dpi --rerun-all --timeout 90`: PASS

Uncommitted local changes to remember:

- `.gitignore`
- `batch/run_systemc.py`
- `batch/run_verilog.py`
- `model/BMM/`
- `systemc/scripts/gen_bmm_tflite_models.py`
- `batch/profile/profile_bmm.html` is generated output; commit only if explicitly wanted.

Pre-existing dirty files from before BMM work may still exist. Do not revert user changes.

## Current Direction

`verilog` 正在往真正 dataflow 走：

```text
Testbench loads .bin into DRAM
  -> Host / descriptors
  -> Command Engine
  -> UDMA load DRAM to L1/SRAM
  -> CONV / TNPS / POOL / EWE work from L1/SRAM
  -> CONV / TNPS / POOL / EWE write L1/SRAM
  -> UDMA store L1/SRAM to DRAM
  -> checker / CRC
```

重要原則：

- `.bin` 由 Testbench DRAM model 透過 `+VERILOG_REF_PROGRAM` 讀入。
- Engine 不應該偷開檔，也不應該靠 Python 展開成大量 per-byte / per-output descriptor 來假裝 full tensor。
- `--full-tensor` 目前只能當 legacy/debug coverage path；後續 full datapath 要用 compact descriptor + DRAM/L1/engine traversal。

## This Round

完成 first true byte-moving dataflow slice：

- `rtl/verilog/Testbench_host_program.v`
  - 新增 writable `vf_dram_model`。
  - DRAM model 從 `+VERILOG_REF_PROGRAM` 讀 `.bin`。
  - Reads 先查 writable override memory，沒有 override 才 fallback 到 `.bin` file bytes。
  - Writes 用 `req_wdata` / `req_wstrb` 寫入 DRAM model backing store。
  - 接上 top UDMA DRAM request/response wires。

- `rtl/verilog/mdla7_top.v`
  - 新增 `udma_dram_resp_rdata` input。
  - 接到 `vf_udma_engine`。

- `rtl/verilog/conv.v`
- `rtl/verilog/requant.v`
- `rtl/verilog/pool.v`
- `rtl/verilog/ewe.v`
- `rtl/verilog/tnps.v`
- `rtl/verilog/udma.v`
- `rtl/verilog/route.v`
  - `vf_udma_engine` 新增 `dram_resp_rdata` input。
  - UDMA load 現在會把 DRAM response 的 16B beat 寫進 L1Mesh。
  - UDMA store 現在會 capture L1 response，再寫回 DRAM model。

- `rtl/verilog/Testbench_top_byte_movers.v`
  - 補齊 new top DRAM response/request ports 的 dummy connection。

Also present from previous steps:

- `verilog` top has microblock control path.
- L1 response has skid/tag path: source, tid, rdata, read valid.
- Host reports `verilog_cycles`.
- `run_verilog.py` report columns include coverage, synth cycles, verilog cycles, ratio, wall time.

## Verified

Compile / static checks:

```bash
python3 -m py_compile batch/gen_verilog_program.py batch/run_verilog.py
git diff --check -- rtl/verilog batch/run_verilog.py batch/gen_verilog_program.py
```

Both passed.

UDMA DRAM-to-L1 smoke:

```bash
./batch/run_verilog_smoke.py --test host \
  --program rtl/obj/verilog/programs/udma_dram_to_l1_smoke.verilog.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/resnet_quant_L1.bin --no-build
```

Passed:

```text
PASS: verilog host-driven ... issued=2 done=2 verilog_cycles=87
```

UDMA DRAM -> L1 -> DRAM -> L1 roundtrip smoke:

```bash
./batch/run_verilog_smoke.py --test host \
  --program rtl/obj/verilog/programs/udma_dram_l1_store_roundtrip.verilog.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/resnet_quant_L1.bin
```

Passed:

```text
PASS: verilog host-driven ... issued=4 done=4 verilog_cycles=260
```

This proves the current UDMA/L1/DRAM byte-moving path:

```text
DRAM(.bin) -> UDMA -> L1 -> UDMA -> DRAM -> UDMA -> L1
```

Target verilog dataflow:

```text
DRAM -> UDMA -> L1 -> CONV/TNPS/POOL/EWE -> L1 -> UDMA -> DRAM
```

## Verilog Status

Completed after this handoff was first written:

1. CONV L1 traffic now goes through `L1Manager` arbitration.
2. The unused DPI-C CRC/datapath helper was removed from active `rtl/verilog`.
3. `run_verilog.py` now has only one real path: closed-loop dataflow.
   - Removed legacy/sample/microblock-control/full-tensor shortcut modes from
     the runner and generator CLI.
   - `run_verilog.py` always emits closed-loop descriptors:
     `DRAM -> UDMA -> L1 -> engine -> L1 -> UDMA -> DRAM -> L1CRC`.
   - `--option dpi` only changes arithmetic backend inside the engine datapath;
     it does not bypass Host / Command / UDMA / L1Manager / L1Mesh control.
4. L1 read alignment was fixed at the CONV vector boundary while preserving the
   aligned-line L1Mesh bus contract used by TNPS/POOL/EWE/REQUANT.
5. TNPS closed-loop descriptors now drive real addrgen indices (`word14/15`) and
   use tile-local scratch mapping so large tensor offsets do not exceed L1.
6. POOL/EWE store paths now drive multi-byte write data/strobes for FP/INT16
   style results instead of advertising multi-byte transfers while writing only
   one byte.
7. Optional DPI datapath helpers were added without splitting the Verilog tree.
   - `conv.v` can switch INT8 MAC to DPI C++ at runtime.
   - `pool.v` and `ewe.v` can switch FP16 pure arithmetic to DPI C++ at runtime.
   - Use `run_verilog.py --option dpi`.
   - The control path stays in the same modules; no `verilog_ctrl/final` split.
8. FP CONV sample closed-loop is connected.
   - Descriptor loads FP16 activation and weight samples from DRAM into L1.
   - CONV reads samples from L1, computes FP sample MAC, writes L1 result.
   - UDMA stores/reloads result through DRAM and L1CRC checks it.
   - Optional DPI FP16 MAC exists in `rtl/verilog/mdla7_dpi_datapath.cpp`.
9. `slice --option dpi --rerun-all` was verified after shortcut removal:
   `pass=108 fail=0 skip=67 total=175`.

Still unfinished before performance tuning:

1. Current closed-loop coverage is still sample/tilelet sized.
   - `verilog_cycles` is therefore much smaller than `synth_cycles`.
   - Example: `deeplab_v3_plus_float_L3` reports about 405 verilog cycles vs
     about 669.4K synth cycles because only a small closed-loop sample runs.
   - This is not a shortcut path anymore; it is real control/data path with too
     small a payload.

2. Full tile/full tensor traversal is not implemented yet.
   - Need compact hardware-side tile loops instead of Python expanding many
     sampled commands.
   - Need enough payload per microblock for UDMA/L1/engine cycles to resemble
     cx/silicon timing.

3. Fast/cx bit-exact golden for full FP output tensors is not done.
   - Current FP closed-loop checks Verilog sample result bytes.
   - Full FP16 output packing/rounding and traversal must be added before this
     can be treated as full golden coverage.

4. Cycle performance calibration should wait until full/tiled traversal exists.
   - `run_verilog.py` already reports `synth_cycles` and `verilog_cycles`.
   - Do not tune ratios using the current sample-sized payloads.

## Next Step

目標：讓 `verilog_cycles` 可以開始跟 `synth_cycles` 比較。現在 cycle 太少的原因是
closed-loop payload 仍是 sample/tilelet，不是完整 layer traversal。後續 task：

1. 定義 Verilog microblock tile 規格。
   - 每個 engine 一個 microblock 要處理多少 bytes / elements / output pixels。
   - CONV 要定義 output tile、KH/KW/IC traversal、partial sum accumulation、
     final writeback。
   - 這會決定 cycle model 的基本粒度。

2. Generator 從 sample descriptor 改成 tile descriptor。
   - 現在每層通常只挑一個 sample。
   - 要改成對 layer 產生 compact tiled microblocks：
     `load input/weight tile -> compute tile -> store output tile -> check tile`。
   - 先讓 small patterns 完整覆蓋，再擴大到 medium/large。

3. 先做 CONV tile descriptor + Verilog/DPI full tile compute。
   - FP CONV 目前只有 sample MAC。
   - 要做完整 `OH x OW x OC x KH x KW x IC` traversal 的 tile loop。
   - INT8 / INT16 / FP 可以用各自 Verilog datapath 或 `--option dpi`
     compute backend，但 Host / Command / UDMA / L1 control path 必須不變。

4. 補 POOL / EWE / TNPS tile sweep。
   - 目前這些 engine 的 closed-loop regression 收斂成 first sample。
   - 下一步要擴成 tile sweep，至少覆蓋完整 output tensor 或 representative tiles。
   - 每個 tile 都要走：
     `DRAM -> UDMA -> L1 -> engine -> L1 -> UDMA -> DRAM -> check`。

5. 讓 L1 / UDMA payload cycles 來自真 bytes。
   - 現在很多 command bytes 很小，所以 cycle 很少。
   - Full tile 後，UDMA load/store bytes、L1 route、FIFO backpressure 才會自然變大。

6. Report 拆分性能統計。
   - 建議新增/保留：
     `load_cycles`, `compute_cycles`, `store_cycles`, `check_cycles`, `total_cycles`。
   - `check/reload` cycles 要另外標出，避免和未來 silicon datapath 混在一起。

7. Golden check 策略。
   - 初期每 tile check CRC。
   - 穩定後改成最後 output CRC，減少 checker overhead。
   - 不能取消 load / compute / store 真路徑。

建議實作順序：

1. CONV tile descriptor + full tile compute。
2. 跑 1~3 個 small FP/INT CONV patterns，讓 `verilog_cycles` 從幾百 cycle
   變成跟 payload 大小相關。
3. 補 POOL / EWE / TNPS tile sweep。
4. 最後調 L1Mesh contention / UDMA bandwidth / placement route timing，讓 cycle
   比例接近 cx。

## Important Files

- `rtl/verilog/Testbench_host_program.v`
- `rtl/verilog/mdla7_top.v`
- `rtl/verilog/conv.v`
- `rtl/verilog/requant.v`
- `rtl/verilog/pool.v`
- `rtl/verilog/ewe.v`
- `rtl/verilog/tnps.v`
- `rtl/verilog/udma.v`
- `rtl/verilog/route.v`
- `rtl/verilog/host.v`
- `rtl/verilog/Testbench_top_byte_movers.v`
- `batch/gen_verilog_program.py`
- `batch/run_verilog.py`
- `batch/run_verilog_smoke.py`

## Commands

Run verilog smoke:

```bash
./batch/run_verilog_smoke.py --test host \
  --program rtl/obj/verilog/programs/udma_dram_l1_store_roundtrip.verilog.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/resnet_quant_L1.bin
```

Run regression:

```bash
./batch/run_verilog.py --filter slice
./batch/run_verilog.py --filter ethz
```

Run with optional DPI datapath helpers:

```bash
./batch/run_verilog.py --filter slice --option dpi
```

If Verilator output is stale:

```bash
rm -rf rtl/obj/verilog/host
```

## Warnings

- Workspace is dirty; several unrelated files were already modified before this handoff. Do not revert user changes.
- Smoke `.verilog.hex` files under `rtl/obj/verilog/programs/` are generated local artifacts.
- `--full-tensor` exists, but should not become the final architecture. Use compact descriptors plus Verilog-side traversal.
- `rtl/verilog` is now the single hardware Verilog tree.
- Legacy `verilog_ctrl` is retired. `rtl/synth` and `run_verilog_ctrl.py` were removed; do not recreate a `verilog_ctrl` / `verilog_final` split.

## Recent Commits

```text
9f50a5a Trim handoff to current verilog final state
ab51879 Update handoff for verilog final streaming
e7e9c1e Tag L1 responses with source and stream tid
3305dde Probe requant L1 producer path
1f7aa93 Feed UDMA store CRC from L1 response
```
