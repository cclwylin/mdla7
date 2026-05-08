# MDLA7 Handoff

**Current date:** 2026-05-08
**Repo:** `/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
**Branch:** `main`
**Last pushed commit:** `d743c5c Add L1 timing modes and conflict profiles`

This file is intentionally short. Finished task logs were removed; use git
history for old checkpoints.

## Next Priority

The L1Mesh mesh / NoC overhead pass is implemented. Internal Engine/L1 traffic
now uses `Payload {engineid, tid, opcode, addr, data[16B], last}` rather than
AXI. `MeshConflict` uses a transparent Payload NoC model with per-bank SRAM
service, 8 perimeter edge queues per read/write direction, bank swizzle, and
fixed simulator scheduling chunks. Payload carries no burst metadata; `tid` +
`last` mark the logical transaction. Per-lane profile tables now report
avg/max latency, wait, service, accesses, and KB for Payload R/W lanes 0..15.
CONV ACT_R and WGT_R are two dedicated CONV-facing direct links into L1Mesh;
they bypass L1Manager and do not share the same ingress with each other.

Latest Hotspot rerun after this change brought `mesh/fast` close to 1.0. High
`max_latency` values can still appear when many large FC transfer chunks queue
behind earlier chunks.

Hotspot compile coverage was also cleaned up. The previous 73 Hotspot
`skipped` compile-log rows now lower to a `matrlz` fallback layer, so the
latest 11-slice Hotspot sweep has 364 compile rows and 0 skipped rows. `matrlz`
means the compiler pre-materializes the reference tensor and the simulator
models a chunked `DRAM -> L1 -> DRAM` UDMA copy; it is a coverage fallback, not
a claim that the arithmetic path exists yet. It currently covers non-spatial
MEAN axes, runtime-matmul FC, INT GELU/HARD_SWISH fallback, reshape shape-prop
mismatches, and tensors that would exceed the 16-bit descriptor dim fields.

Next useful priority: extend MicroBlock fusion beyond current safe paths.
`try_stream_conv_chain()` now keeps generic Conv→Conv streaming to pointwise
`1x1` linear chains, and preserves the validated deep `3x3 Conv... -> D2S ->
EWE` path. `try_stream_binary_ewe_chain()` also fuses linear INT8
`ADD/MUL/SUB` chains tile-by-tile, so sd decoder flows such as
`L9 -> L10 -> L11` avoid re-reading intermediate activations from DRAM. Plain
spatial `3x3 Conv -> 3x3 Conv` still needs explicit line-buffer / halo
ownership before it should bypass DRAM on U-Net-like tiled pairs. After that,
replace high-volume `matrlz` fallbacks with real engine lowering where it
matters most, especially non-spatial MEAN/reduce and attention
reshape/transpose style movement.

## Current Layout

`systemc/` now holds simulator/backend code:

```text
systemc/
  include/mdla7/
  src/
  scripts/              # compiler, validator, plotter backend
  build/                # ignored build products
  Makefile
  setup.sh
  requirements.txt
```

`batch/` now holds user-facing runners, regression inputs, profile indexes, and
generated run output:

```text
batch/
  run_model.py
  run_mdla6_pattern.py
  run_hotspot.py
  run_ethz_v5.py
  run_ethz_v6.py
  run_mlperf.py
  gen_model_profile.py
  mdla6_ethz_v6.csv
  mdla6_ethz_v6_sorted.csv
  profile_mdla6_pattern.html
  profile_hotspot.html
  output/               # ignored generated reports/profiles/binaries
```

Important path rule:

- Run Python entry points from repo root as `./batch/<runner>.py`.
- Runners write artifacts under `batch/output/`.
- Runners call compiler/plotter backend scripts from `systemc/scripts/`.
- Runners call simulator binaries from `systemc/build/`.

## Quick Commands

```bash
make -C systemc -s
./batch/run_model.py --list
./batch/run_model.py model/MLPerf_Tiny/vww_96_int8.tflite --no-build
./batch/run_mlperf.py --filter vww_96_int8 --limit 1
./batch/run_hotspot.py --filter swin_quant --limit 1
./batch/run_mdla6_pattern.py --filter mobilenet_v3_b4_quant --limit 1 --rerun-all
```

If already inside `batch/`, keep the current-directory prefix:

```bash
./run_hotspot.py --rerun-all
```

Do not use bare `run_hotspot.py`; zsh does not search the current directory by
default.

Profile entry pages:

```text
batch/profile_mdla6_pattern.html
batch/profile_hotspot.html
batch/profile_ethz_v5.html
batch/profile_ethz_v6.html
batch/profile_mlperf.html
```

`profile_mdla6_pattern.html` keeps MDLA6 `cx` and `myms/cx` columns.
All other profile indexes hide `cx` fields and default to sorting by `mesh/fast`.

## L1 Timing Modes

`systemc/include/mdla7/memory.h` has:

- `L1TimingMode::FastEstimate`
- `L1TimingMode::PortConflict`
- `L1TimingMode::MeshConflict`
- `L1TimingMode::MeshOptimistic`

`systemc/src/test_model.cpp` accepts:

```text
--l1-timing=fast
--l1-timing=conflict
--l1-timing=mesh
--l1-timing=mesh-opt
```

Mode relationship:

```text
mesh = port-conflict SRAM bank timing + transparent NoC resource contention
mesh-opt = mesh-style Payload chunking + SRAM bank timing, skipping NoC resource reservations
```

So `mesh/conflict` isolates the extra NoC overhead, while `conflict/fast`
isolates the SRAM bank/port conflict overhead. In the current model, `mesh`
also chops long blocking calls into simulator scheduling chunks:

```text
Payload lane width = 16B
Payload protocol   = no burst metadata
transaction group  = tid + last
```

Aliases still exist:

```text
--l1-fast
--l1-conflict
--l1-mesh
--l1-mesh-opt
```

`run_mdla6_pattern.py`, `run_hotspot.py`, `run_ethz_v5.py`, `run_ethz_v6.py`,
and `run_mlperf.py` run fast/conflict/mesh together and produce combined HTML
debug pages:

```text
batch/output/<stem>.html
batch/output/<stem>.fast.html
batch/output/<stem>.conflict.html
batch/output/<stem>.mesh.html
```

## Hotspot Slices

Hotspot `.tflite` slices are under `model/Hotspot/` and are intentionally
tracked in git as small regression/debug micro-patterns. The rest of `model/*`
stays ignored. Current slices include transformer-like repeated blocks such as:

```text
vit_b16_quant_L94_L119.tflite
swin_quant_L298_L319.tflite
llama2_quant_L35_L74.tflite
mobilebert_quant_L41_L80.tflite
sam_quant_L22_L61.tflite
sd_encoder_quant_L12_L51.tflite
sd_decoder_quant_L152_L191.tflite
gpt2_quant_L24_L63.tflite
swin_float_L53_L92.tflite
sam_float_L33_L72.tflite
mobilevit_v2_float_L19_L56.tflite
```

Use:

```bash
./batch/run_hotspot.py --list
./batch/run_hotspot.py --filter vit --rerun-all
```

## L1Mesh Drawings

L1Mesh-specific drawio pages were split out:

```text
spec/l1mesh.drawio
```

`spec/l1mesh.drawio` now includes:

```text
L1Mesh-4x4-NoC
L1Mesh-Arbitration
L1Mesh-Bank-Tile
L1Mesh-Payload-Edge-Map
```

`L1Mesh-Payload-Edge-Map` shows the 16 Payload R and 16 Payload W lanes mapped
onto the 8 logical W/E perimeter edges. `spec/mdla7.drawio` keeps the top-level
MDLA7 diagrams. The user has manually modified the `L1Mesh-4x4-NoC` diagram;
future diagram edits should read and modify the current file in place, not
redraw from an old base.

## Current Validation

Recently verified after moving runners to `batch/`:

```bash
python3 -m py_compile batch/run_model.py batch/run_ethz_v5.py batch/run_ethz_v6.py \
  batch/run_mlperf.py batch/run_mdla6_pattern.py batch/run_hotspot.py \
  batch/gen_model_profile.py batch/corpus_runner.py
./batch/run_hotspot.py --filter swin_quant --limit 1
./batch/run_mlperf.py --filter vww_96_int8 --limit 1 --keep-bin
./batch/run_model.py model/MLPerf_Tiny/vww_96_int8.tflite --no-build
git diff --check
```

Recently verified after the Hotspot `matrlz` fallback update:

```bash
python3 -m py_compile systemc/scripts/compile_model.py batch/run_model.py batch/run_hotspot.py
make -C systemc -s
./batch/run_model.py sd_encoder_quant_L12_L51 --keep-intermediate
./batch/run_hotspot.py --rerun-all
```

Latest Hotspot result:

```text
fast 11/11 ran, conflict 11/11 ran, mesh 11/11 ran
Compile log: 364 rows, 0 skipped
matrlz fallback rows: 73
```

All passed.

Recently verified after L1Mesh Payload interface/lane-stat updates:

```bash
python3 -m py_compile batch/run_model.py batch/run_mdla6_pattern.py batch/run_hotspot.py
make -C systemc -s
./batch/run_hotspot.py --filter gpt2 --limit 1 --rerun-all
```

The mesh profile reports per-lane Payload R/W latency. `max_latency` can remain
much larger than `max_service` because it includes queue wait across many chunks
in large FC transfers.

## Git Notes

Expected pending changes include the batch-directory migration plus prior
L1Mesh/spec/textbook updates. `batch/output/` is ignored. Do not restore the old
runner/profile/output locations under `systemc/`.
