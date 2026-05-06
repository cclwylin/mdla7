# MDLA7 Handoff

**Current date:** 2026-05-07
**Repo:** `/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
**Branch:** `main`
**Last pushed commit:** `d743c5c Add L1 timing modes and conflict profiles`

This file is intentionally short. Finished task logs were removed; use git
history for old checkpoints.

## Next Priority

First thing next: investigate L1Mesh mesh / NoC overhead.

From the current Hotspot profile, `conflict/fast` is almost flat
(`~1.00-1.04x`), while `mesh/conflict` is much larger (`~1.6-1.9x`). That means
the dominant Hotspot penalty is not SRAM bank-port conflict; it is the mesh
approximation's edge ingress / router output / directed link / local output
queueing. Treat `profile_hotspot.html` as the starting debug page, sorted by
`mesh/fast`, and focus first on why transformer-like repeated slices create NoC
hotspots.

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

`systemc/src/test_model.cpp` accepts:

```text
--l1-timing=fast
--l1-timing=conflict
--l1-timing=mesh
```

Mode relationship:

```text
mesh = port-conflict SRAM bank timing + NoC edge/router/link timing
```

So `mesh/conflict` isolates the extra NoC overhead, while `conflict/fast`
isolates the SRAM bank/port conflict overhead.

Aliases still exist:

```text
--l1-fast
--l1-conflict
--l1-mesh
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

`spec/mdla7.drawio` keeps the top-level MDLA7 diagrams. The user has manually
modified the `L1Mesh-4x4-NoC` diagram; future diagram edits should read and
modify the current file in place, not redraw from an old base.

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

All passed.

## Git Notes

Expected pending changes include the batch-directory migration plus prior
L1Mesh/spec/textbook updates. `batch/output/` is ignored. Do not restore the old
runner/profile/output locations under `systemc/`.
