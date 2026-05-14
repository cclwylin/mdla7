# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MDLA7 is a neural-network accelerator simulator. The project has three simulation layers that must stay in sync:

1. **SystemC functional model** (`systemc/`) — fast analytical simulation; primary development target
2. **Verilog RTL** (`rtl/verilog/`) — real hardware Verilog; DPI testbenches validate bit-exact behaviour
3. **Batch regression harness** (`batch/`) — Python runners that compile `.tflite` → `program.bin` → run sim → produce HTML profiles

## Build

```bash
# SystemC (primary target)
cd systemc && make                    # auto-detects SystemC via `brew --prefix systemc`
cd systemc && make clean && make      # required after pulling pre-built binaries from Intel mac

# Verilog smoke tests (Verilator)
./batch/run_verilog_smoke.py
```

macOS: SystemC is at `/opt/homebrew/opt/systemc` (Apple Silicon) or `/usr/local/opt/systemc` (Intel). The Makefile auto-selects via `brew --prefix systemc`. If a pre-compiled binary was built on a different arch, `make clean && make` is required.

## Running Regressions

```bash
# SystemC — BMM corpus, fast mode only
~/.venvs/mdla7/bin/python batch/run_systemc.py --filter bmm --fast-only

# SystemC — single model
~/.venvs/mdla7/bin/python batch/run_systemc.py --filter bmm --model-filter tiled_2.5ms_int8 --rerun-all

# SystemC — cx (L1Mesh contention) mode
~/.venvs/mdla7/bin/python batch/run_systemc.py --filter bmm --cx

# Verilog DPI — BMM corpus
./batch/run_verilog.py --filter bmm --rerun-all --dpi

# Verilog DPI — force clean obj rebuild
rm -rf rtl/obj/verilog/host && ./batch/run_verilog.py --filter bmm --rerun-all --dpi
```

Regression CSVs land in `batch/output/`. Per-model HTML profiles land in `batch/output/fast/` or `batch/output/cx/`.

## Corpus → Model Directory Mapping

| `--filter` | Model directory |
|---|---|
| `bmm` | `model/BMM/` |
| `ethz` / `ethz_v6` | `model/ETHZ_v6/` |
| `hotspot` | `model/Hotspot/` |
| `unit` | `model/UnitTest/` |

## Regenerating Models

```bash
# INT8 KV-cache attention (TFLite)
~/.venvs/mdla7/bin/python systemc/scripts/gen_qwen35_kvcache_tflite.py --dtype int8 --kv-len 128 512 1024

# Tiled BMM attention
~/.venvs/mdla7/bin/python systemc/scripts/gen_bmm_attention_tiled_tflite.py

# Restore git-tracked tflite models after exFAT remount loss
git checkout model/BMM/
```

**Warning**: `model/` lives on an exFAT external drive. Newly generated `.tflite` files are lost on unmount unless committed.

## Architecture

### Descriptor Pipeline

The fundamental unit is a 64-byte `Descriptor` (`systemc/include/mdla7/descriptor.h`). Every operation — CONV, REQUANT, EWE, POOL, TNPS, UDMA — is a descriptor. The Host pushes descriptors into `desc_stream`; the `CommandEngine` fans them out to per-engine FIFOs.

Dependency ordering uses **wait tags**: each descriptor carries a `wait_tag` (must complete before this starts) and emits a `done_tag` on completion. The softmax decomposition chain is: `UDMA_load → POOL_MAX → EWE_SUB → EWE_EXP → POOL_SUM → EWE_DIV → UDMA_store` — seven descriptors with explicit tag edges.

### Engine Op Classes

| `OpClass` | Engine | Notes |
|---|---|---|
| `OC_CONV` | ConvEngine | INT8/INT16 MAC; CONV→Requant psum chain is dedicated 128-lane INT32 bus |
| `OC_REQUANT` | RequantEngine | MBQM + clamp; owns output writeback to L1 |
| `OC_EWE` | EweEngine | Element-wise: ADD/MUL/SUB/EXP/DIV; lane saturation |
| `OC_POOL` | PoolEngine | MAX/AVG/SUM row-reduce; shape convention `[in_h=rows, in_w=K, in_c=1, k_w=K]` |
| `OC_TNPS` | TnpsEngine | Transpose/space-to-depth/gather |
| `OC_UDMA` | UDMA | DRAM↔L1 DMA; modes: linear, strided-2D, indexed-gather, scatter-concat, strided-slice, D2S, ACT-compress/decompress |

### Memory Map

- **L1Mesh**: `0x0000_0000–0x002F_FFFF` (3 MB, `L1MESH_BYTES`). All engine scratch, activations, and weights live here.
- **DRAM**: above L1Mesh; `.bin` files set `dram_in/wgt/out` offsets.
- L1 budget constant: `L1_BUDGET = L1MESH_BYTES = 3145728`.

### Compiler (`systemc/scripts/compile_model.py`)

Reads a `.tflite` flatbuffer → emits `program.bin`. Blob layout: 16-byte header (`magic=MDL7`, `version=3/4`, `num_layers`, `data_offset`) → `LayerMeta[N]` (64-byte v3 / 76-byte v4) → `GraphMeta[N]` (32-byte) → data section (inputs, weights, refs).

Op lowering highlights:
- `BATCH_MATMUL` → `OK_FC_BMM` (kind=30) — lowered to CONV
- `REVERSE_V2` (compile-time constant) → `OK_REVERSE` (kind=32) — UDMA constant-load
- `SOFTMAX` → 5- or 7-descriptor chain (FP16 or INT8 dequant→FP→quant)

### Softmax Decomposition (`systemc/src/mdla7_model_runner.cpp` ~L9462–L9694)

`MDLA7_SOFTMAX_DECOMPOSE=1` (default). For a softmax over shape `[rows, K]`:
- **Batched whole-shape** (current): one set of 5/7 descriptors operating on the full `[rows, K]` tensor — not a per-row loop. This is the fast path; per-row serialization was the original bottleneck.
- **INT8 path**: dequant FP16 → 5-op FP chain → quant back to INT8.
- **Scratch layout**: `addr_ctr/exp/fp_in/fp_out` (each `rows×K×2` bytes), `addr_max/sum` (each `rows×2` bytes), all allocated from L1.
- L1 fit check governs whether decomposition fires; fallback is monolithic LUT/FP path.

### CONV Spatial Tile Loops (Phase 6c/6d, descriptor v12)

CONV descriptors can loop over OW and OH tiles without re-issuing from host:

| Field | word | bits | Activation |
|---|---|---|---|
| `conv_spatial_tile_enable` | [3] | [18] | **Must be set** or OW/OH loops are disabled |
| `conv_tile_ow_count` | [8] | [31:16] | iterations |
| `conv_tile_oh_count` | [9] | [23:8] | iterations |
| `act_tile_col_stride` | [10] | [21:0] | L1 bytes per OW step |
| `act_tile_row_stride` | [11] | [21:0] | L1 bytes per OH step |

Backward-compat: word[8..11] carry WGT data in legacy descriptors; the enable flag prevents collision.

### RTL Verilog (`rtl/verilog/`)

Active hardware tree. Do **not** create `verilog_ctrl` / `verilog_final` subdirectories — all hardware work belongs here. DPI testbench entry point: `Testbench_host_program.v`. Datapath primitives tested by `run_verilog_smoke.py`: conv MAC, requant MBQM, pool max/avg, EWE vector ops, TNPS space-to-depth, L1Mesh route timing, L1Manager contention.

Target architecture (not fully implemented in RTL yet):
- CONV psum chains directly into Requant; Requant owns L1 output writeback.
- UDMA↔L1 target bus: 16×16B R + 16×16B W. DRAM model still uses simpler 128-bit interface.

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `MDLA7_SOFTMAX_DECOMPOSE` | `1` | `0` disables softmax decomposition (monolithic LUT/FP path) |
| `MDLA7_CONV_WS` | unset | Override CONV workspace sizing |
| `MDLA7_DUMP_FAIL_DIR` | unset | Dump failing model data to this directory |
| `MDLA7_FORCE_EWE_CONV_STREAM` | unset | Force EWE→CONV stream path |
| `MDLA7_EXPERIMENTAL_SLICE_FANOUT` | unset | Enable experimental slice fanout |
| `MDLA7_EXPERIMENTAL_SLICE_COMPUTE` | unset | Enable experimental slice compute |
