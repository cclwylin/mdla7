# MDLA7 交接紀錄

## 專案目標

- 最後要做到 FPGA。
- 用 Fast/Synth mode 驗證 Compile、Function、Performance。
- Verilog 的 arithmetic 加速用 DPI。
- Synth 跟 Verilog correlation 誤差不能超過 10%。

Date: 2026-05-13 CST
Repo: `/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch: `main`

## 2026-05-14 更新：文件與 commit 狀態

- code / coverage 的 commit 是 `e6d1435 Cover BMM and ETHZ unsupported ops`。
- 文件同步的 commit 是 `434f141 Document BMM and ETHZ coverage policy`。
- 相關 markdown 已經同步：
  `model/BMM/README.md`, `profile_html.md`, `rtl/verilog/README.md`,
  `md/06_tflite_flatbuffer.md`, `md/13_ewe_pool_softmax_d2space.md`,
  `md/17_verification_coverage.md`, `md/18_regression_profile_html.md`，並且已重新產生
  `md/mdla7_textbook.md`。
- `memory.md` 在文件 commit 後還有本地更新，內容記錄最新 commit、BMM profile
  命名規則、相關文件位置、textbook 重新產生規則，以及 generated HTML 要和 docs-only
  commit 分開的規則。
- 目前已知 dirty 的 generated output 可能包含
  `batch/profile/profile_bmm.cx.html`，這通常是背景 regression 刷新的結果。
  除非使用者明確要求提交 profile snapshot，否則不要把它混進 docs/code commit。
- BMM profile 檔名固定為：
  `batch/profile/profile_bmm.html` 和 `batch/profile/profile_bmm.cx.html`。
  不要再引入舊式雙重 mode 名稱。

## 2Current Direction

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

目標：讓 `verilog_cycles` 可以開始跟 `cx_cycles` 比較。現在 cycle 太少的原因是
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
