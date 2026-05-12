# MDLA7 Handoff

Date: 2026-05-12 CST  
Repo: `/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`  
Branch: `main`

## Current Direction

`verilog_final` 正在往真正 dataflow 走：

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

- `.bin` 由 Testbench DRAM model 透過 `+FINAL_REF_PROGRAM` 讀入。
- Engine 不應該偷開檔，也不應該靠 Python 展開成大量 per-byte / per-output descriptor 來假裝 full tensor。
- `--full-tensor` 目前只能當 legacy/debug coverage path；後續 full datapath 要用 compact descriptor + DRAM/L1/engine traversal。

## This Round

完成 first true byte-moving dataflow slice：

- `rtl/verilog_final/Testbench_host_program.v`
  - 新增 writable `vf_dram_model`。
  - DRAM model 從 `+FINAL_REF_PROGRAM` 讀 `.bin`。
  - Reads 先查 writable override memory，沒有 override 才 fallback 到 `.bin` file bytes。
  - Writes 用 `req_wdata` / `req_wstrb` 寫入 DRAM model backing store。
  - 接上 top UDMA DRAM request/response wires。

- `rtl/verilog_final/mdla7_top_final.v`
  - 新增 `udma_dram_resp_rdata` input。
  - 接到 `vf_udma_engine`。

- `rtl/verilog_final/final_datapath.v`
  - `vf_udma_engine` 新增 `dram_resp_rdata` input。
  - UDMA load 現在會把 DRAM response 的 16B beat 寫進 L1Mesh。
  - UDMA store 現在會 capture L1 response，再寫回 DRAM model。

- `rtl/verilog_final/Testbench_top_byte_movers.v`
  - 補齊 new top DRAM response/request ports 的 dummy connection。

Also present from previous steps:

- `verilog_final` top has microblock control path.
- L1 response has skid/tag path: source, tid, rdata, read valid.
- Host final reports `vf_cycles`.
- `run_verilog_final.py` report columns include coverage, synth cycles, verilog final cycles, ratio, wall time.

## Verified

Compile / static checks:

```bash
python3 -m py_compile rtl/batch/gen_verilog_final_program.py rtl/batch/run_verilog_final.py
git diff --check -- rtl/verilog_final/Testbench_host_program.v rtl/verilog_final/final_datapath.v rtl/verilog_final/mdla7_top_final.v rtl/verilog_final/Testbench_top_byte_movers.v rtl/batch/run_verilog_final.py rtl/batch/gen_verilog_final_program.py
```

Both passed.

UDMA DRAM-to-L1 smoke:

```bash
./rtl/batch/run_verilog_final_smoke.py --test host \
  --program rtl/obj/verilog_final/programs/udma_dram_to_l1_smoke.final.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/resnet_quant_L1.bin --no-build
```

Passed:

```text
PASS: verilog_final host-driven ... issued=2 done=2 vf_cycles=87
```

UDMA DRAM -> L1 -> DRAM -> L1 roundtrip smoke:

```bash
./rtl/batch/run_verilog_final_smoke.py --test host \
  --program rtl/obj/verilog_final/programs/udma_dram_l1_store_roundtrip.final.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/resnet_quant_L1.bin
```

Passed:

```text
PASS: verilog_final host-driven ... issued=4 done=4 vf_cycles=260
```

This proves the current UDMA/L1/DRAM byte-moving path:

```text
DRAM(.bin) -> UDMA -> L1 -> UDMA -> DRAM -> UDMA -> L1
```

Target final dataflow:

```text
DRAM -> UDMA -> L1 -> CONV/TNPS/POOL/EWE -> L1 -> UDMA -> DRAM
```

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

- `rtl/verilog_final/Testbench_host_program.v`
- `rtl/verilog_final/mdla7_top_final.v`
- `rtl/verilog_final/final_datapath.v`
- `rtl/verilog_final/host_final.v`
- `rtl/verilog_final/Testbench_top_byte_movers.v`
- `rtl/batch/gen_verilog_final_program.py`
- `rtl/batch/run_verilog_final.py`
- `rtl/batch/run_verilog_final_smoke.py`

## Commands

Run final smoke:

```bash
./rtl/batch/run_verilog_final_smoke.py --test host \
  --program rtl/obj/verilog_final/programs/udma_dram_l1_store_roundtrip.final.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/resnet_quant_L1.bin
```

Run regression:

```bash
./rtl/batch/run_verilog_final.py --filter slice
./rtl/batch/run_verilog_final.py --filter ethz
```

If Verilator output is stale:

```bash
rm -rf rtl/obj/verilog_final/host
```

## Warnings

- Workspace is dirty; several unrelated files were already modified before this handoff. Do not revert user changes.
- Smoke `.final.hex` files under `rtl/obj/verilog_final/programs/` are generated local artifacts.
- `--full-tensor` exists, but should not become the final architecture. Use compact descriptors plus Verilog-side traversal.
- `run_verilog_ctrl.py` and `run_verilog_final.py` are separate flows. `verilog_ctrl` is control/timing compare; `verilog_final` is the path for true datapath.

## Recent Commits

```text
9f50a5a Trim handoff to current verilog final state
ab51879 Update handoff for verilog final streaming
e7e9c1e Tag L1 responses with source and stream tid
3305dde Probe requant L1 producer path
1f7aa93 Feed UDMA store CRC from L1 response
```
