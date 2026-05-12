# MDLA7 Handoff

Date: 2026-05-12 CST  
Repo: `/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`  
Branch: `main`

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

## Verilog Unfinished Items

1. CONV still bypasses L1Manager arbitration.
   - `mdla7_top.v` currently selects `run_conv ? conv_l1_req_* : l1mgr_*`.
   - Next step is to route CONV L1 traffic through L1Manager like UDMA,
     REQUANT, POOL, EWE, and TNPS.

2. Several datapaths are still sample / bounded traversal paths.
   - POOL / EWE / TNPS / CONV have real Verilog primitives, but not every
     layer runs full tensor traversal.
   - Large tensors may still use windowed, sample, or bounded descriptors.

3. FP datapath is not yet full silicon-like traversal.
   - FP CONV / POOL / EWE samples exist.
   - Full FP tensor datapath still needs to be expanded.

4. `common.v` still contains the DPI CRC helper.
   - `mdla7_true_datapath` uses Verilator DPI-C under non-synthesis builds.
   - This should be isolated or removed from the final pure synthable Verilog
     path so it cannot be mistaken for hardware datapath.

5. Cycle performance is not yet naturally aligned with `cx`.
   - `run_verilog.py` reports `synth_cycles` and `verilog_cycles`.
   - Closed-loop load / compute / store / reload / check timing still needs
     calibration against the `cx` model without relying on debug padding.

6. CRC coverage gates are not default strict enough.
   - Default `verilog` regression can still pass with zero `refB/sramB`.
   - Use `--crc-coverage --require-crc-coverage` when a regression must prove
     datapath bytes were actually checked.

## Next Step

1. Connect TNPS into the same closed loop:
   `DRAM -> UDMA load input tensor to L1 -> TNPS reads L1 -> TNPS writes L1 output -> UDMA store -> DRAM -> CRC/checker`.

2. Then connect POOL / EWE into the same closed loop:
   - UDMA loads input tensor(s) into L1.
   - Engine reads L1, not descriptor sample bytes.
   - Engine writes output L1.
   - UDMA stores output back to DRAM.

3. Then connect CONV into the same closed loop:
   - Load activation / weights / bias / params.
   - Implement tile/full traversal MAC loop.
   - Write output to L1 and store back to DRAM.

4. Replace ref-fill/probe-only coverage with real producer-output coverage.

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
