# MDLA7 Synth RTL Shells

This directory contains synthesizable Verilog control/latency shells for the
SystemC `--engine=synth --L1=synth` blocks:

- `command.v`
- `conv.v`
- `requant.v`
- `ewe.v`
- `pool.v`
- `tnps.v`
- `udma.v`
- `l1manager.v`
- `l1mesh.v`
- `mdla7_top.v`
- `host.v`
- `dram.v`
- `Testbench.v`
- `mdla7_dpi.cpp` (Verilator DPI-C functional datapath core)

The compute modules model the same synth-mode phase boundaries used by the
SystemC profiler. Under Verilator simulation, the engines also call a shared
DPI-C datapath core that parses the MDL7 `.bin` image and computes output CRCs
from input/weight/parameter bytes for CONV/FC/DWCONV, EWE, POOL, and TNPS
layers.

`mdla7_top.v` instances the blocks above behind a simple descriptor-dispatch
wrapper. It first runs `command.v` and then starts one selected block by
`desc_op_class`:

| op_class | block |
|---:|---|
| 0 | command only |
| 1 | conv |
| 2 | requant |
| 3 | ewe |
| 4 | pool |
| 5 | tnps |
| 6 | udma |
| 7 | l1manager |
| 8 | l1mesh |

`host.v` loads a compiler generated MDLA7 program image and expands each layer
into a microblock stream. CONV/FC/DWCONV layers issue UDMA weight/parameter
load, UDMA activation load, CONV compute, REQUANT compute, and UDMA store
descriptors per microblock. EWE layers issue UDMA load A, UDMA load B/params,
EWE compute, and UDMA store. POOL/TNPS layers issue UDMA load, compute, and
UDMA store. The descriptor metadata carries `layer_id`, `microblock_id`,
`stream_slot`, and stream flags for load/compute/store/final-tile.

The default program is:

```sh
rtl/bin/Hotspot/gpt2_quant_L24_L63.bin
```

Override it with:

```sh
vvp /tmp/mdla7_system_tb.vvp +PROGRAM=rtl/bin/Hotspot/swin_quant_L298_L319.bin
```

The checked-in testbench validates both microblock control dispatch and
functional datapath results. `host.v` reads each layer's fast-model reference
bytes from the program image, computes an FNV-1a checksum, and compares it with
the checksum produced on the final microblock store descriptor. Earlier
load/compute descriptors carry `ref_size=0`, so they exercise control without
re-running the full-layer DPI compare. The engine checksum is computed from the
layer's real input/weight/parameter bytes. `OP_MATERIALIZE` remains the
compiler's pre-materialized-output fallback. Synthesis builds see a stable
file-I/O-free stub around the DPI-only simulation core.

Regenerate Hotspot bins:

```sh
PY=${MDLA7_VENV:-$HOME/.venvs/mdla7}/bin/python
find model/Hotspot -type f -name '*.tflite' -print0 | while IFS= read -r -d '' f; do
  rel=${f#model/}
  out="rtl/bin/${rel%.tflite}.bin"
  mkdir -p "$(dirname "$out")"
  "$PY" systemc/scripts/compile_model.py "$f" "$out"
done
```

Single filelist:

```sh
rtl/synth/filelist_system_tb.f
```

Syntax smoke with Host + DRAM:

```sh
iverilog -g2012 -Wall -I rtl/synth -o /tmp/mdla7_system_tb.vvp -f rtl/synth/filelist_system_tb.f
```

Verilator batch smoke over the compiled bins:

```sh
rtl/batch/run_mdla7_verilog.py --filter '*.bin' --require-host-load --timeout 300
```

The Verilator object directory is `rtl/obj_dir`; `rtl/verilator` is only used as
the Verilator tool checkout. If the simulator is missing, the runner builds it
there. On macOS it also auto-adds the libc++ include directory needed by some
Command Line Tools installs; override with `MDLA7_CXX_STDLIB_INCLUDE` if needed.
The runner prints the detected host (`macOS Intel` or `macOS Apple Silicon`) and
checks the `VTestbench` binary architecture before running, rebuilding the
default simulator if the arch does not match the current machine.

Synthesis-oriented hierarchy check for `mdla7_top` uses explicit RTL sources, so
there is only one filelist to keep straight:

```sh
yosys -q -p "read_verilog -Irtl/synth rtl/synth/command.v rtl/synth/conv.v rtl/synth/requant.v rtl/synth/ewe.v rtl/synth/pool.v rtl/synth/tnps.v rtl/synth/udma.v rtl/synth/l1manager.v rtl/synth/l1mesh.v rtl/synth/mdla7_top.v; hierarchy -top mdla7_top -check"
```
