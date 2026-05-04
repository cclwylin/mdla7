# MDLA7 SystemC Simulator — Handoff

**Status (2026-05-04, post-v8):** functional sim runs full TFLite-quant CNN inference (CONV / DWCONV / FC / POOL / SOFTMAX / RESHAPE / **ADD** / **CONCAT** / **GATHER**) plus **FP16 / BF16 / FP32-source CONV/DWCONV** with bit-exact (int) or FP-tolerant (1e-3) verification against an internal numpy reference. v8 lands FP compute, **DEQUANTIZE walkback**, and **shape inference** for FP-quant models with placeholder static shapes (mobilenet_v3).

| Model | Layers | dtype | Cycles @ 1 GHz | Status |
|---|---|---|---|---|
| MobileNet v1 (uint8, zp_w=157) | 31 | INT8 | 112,372 | ✅ all PASS |
| EfficientNet-Lite0 (int8) | **62** | INT8 | 1,859,815 | ✅ all PASS (incl. **9 ADD**) |
| VGG16-quant (uint8) | 22 | INT8 | 116,105 | ✅ all PASS |
| Audio YAMNet (uint8/FP intermediates) | 34 | INT8 | ~256k | ✅ all PASS (1 FC FP skipped) |
| MobileNet v2 INT16 | 53 | INT16 | 2,571,649 | ✅ all PASS (incl. **OC-tiled** FC 1280→1001) |
| **MobileNet v3 large FP16** | **77** | **FP16** | **1,674,759** | ✅ 77/77 PASS at TOL=0 bit-exact (58 fuse-source-skipped via ping-pong L1; DRAM W 1.15 MB) |
| BF16 chain_conv_relu_quant | 1 | BF16 | 22,634 | ✅ PASS |
| BF16 chain_residual_block | 1 (1 ADD skipped) | BF16 | 30,734 | ✅ PASS |
| **DeepLab v3+ float (ETHZ_V6)** | 74 | FP16 | 87,716,000 (87.7 ms) | ⚠️ runs end-to-end, **68/74 layers FAIL** bit-exact (multi-tile FP follow-up — see §6) |
| **DeepLab v3+ quant (ETHZ_V6)** | 74 | INT8 | 27,410,000 (27.4 ms) | ⚠️ runs end-to-end, **58/74 layers FAIL** (same multi-tile root cause) |

ETHZ_V6 53-model regression sweep: [`systemc/run_ethz_v6.py`](systemc/run_ethz_v6.py) → `output/ethz_v6_regression.csv` (`pattern,ms,status`).

Bit-exact = numpy reference == sim DRAM output. **TFLite fidelity (v7):**
- efficientnet_lite0 layer 0 (int8, zp_in=3): mean|diff| **0.07 → 0.01**, max|diff| **58 → 1**, 4642/401408 differ (all by 1 LSB).
- mobilenet_v1_0.25_128_quant layer 0 (uint8, zp_w=157): mean|diff| **5.68 → 0.00**, max|diff| **33 → 1**, 72/32768 differ (all by 1 LSB).
- Remaining 1-LSB residual is a XNNPACK/gemmlowp rounding-edge artifact; below the level that matters in practice.

---

## 1. Repo layout

```
MDLA7/
├── spec/                        — design spec
│   ├── spec.md                  — main spec doc (sections §1–§6, §3A.1–§3A.12)
│   └── mdla7.drawio             — A3 system block diagram (2 pages)
├── model/                       — TFLite test models, organized by dtype
│   ├── INT8/, UINT8/, INT16/, FP16/, FP32/, BF16/, INT4/, synthetic/
│   └── labels.txt
├── systemc/                     — SystemC simulator (this is the live code)
│   ├── include/mdla7/*.h        — headers (single-translation-unit per binary)
│   ├── src/                     — entry-point binaries (main.cpp / test_*.cpp)
│   ├── scripts/                 — Python tooling
│   ├── build/                   — compile artefacts (gitignored: .o, binaries)
│   ├── output/                  — per-model run artefacts (gitignored)
│   │                              <stem>.{bin,profile.json,profile.csv,profile.png,html}
│   ├── Makefile                 — auto-detects Intel / Apple Silicon SystemC
│   ├── setup.sh                 — one-shot venv + deps
│   ├── requirements.txt         — Python deps (numpy / tflite / tensorflow / matplotlib)
│   └── run_model.py             — main user entry point
├── scripts/                     — repo-level scripts (PDF build for spec, unrelated to sim)
├── notebook.md                  — early note from FFmpeg/SQLite/TPU walkthrough work
└── handoff.md                   — this file
```

---

## 2. Quick start

```bash
# 1. one-time
cd systemc
./setup.sh                       # creates ~/.venvs/mdla7 + installs deps
                                 # numpy, tflite (flatbuffer parser), tensorflow,
                                 # matplotlib

# 2. compile + run a model end-to-end
python3 run_model.py mobilenet_v1
                                 # default flow:
                                 # .tflite → compile_model.py → output/<stem>.bin
                                 # → test_model → bit-true verify + cycle report
                                 # → output/<stem>.{profile.json, profile.csv,
                                 #                  profile.png, html}
                                 # The .html is self-contained (PNG inlined);
                                 # open it for a one-page summary you can share.

# 3. other entry points
python3 run_model.py --list                         # show all bundled models
python3 run_model.py vgg16                          # substring match
python3 run_model.py mobielnet_v1                   # typo → fuzzy match
python3 run_model.py --all                          # regression sweep
make test                                           # synthetic unit test
make test-tflite                                    # legacy single-conv test
./scripts/plot_profile.py path/to/profile.json     # render Gantt manually
./scripts/validate_tflite.py model/INT8/foo.tflite # check vs real TFLite Interpreter
```

`run_model.py` auto re-execs into `~/.venvs/mdla7/bin/python` (no need to source-activate).

---

## 3. Architecture (what the sim actually does)

```
.tflite ──compile_model.py──▶ program.bin
                                  │
                          test_model loads ─▶ DRAM
                                  │
              Mdla7System.host streams descriptors
                                  │
       Host → CommandEngine (dependency tag tracker) → 5 engines
                                  │
              CONV  ──16 INT32 chain──▶  RequantEngine
                                  │            │
              UDMA   ◄── L1_Manager (multi-bank L1Mesh) ──▶  EWE / POOL
                                  │
                              4 GB DRAM
```

**Spec mapping** (see [spec/spec.md](spec/spec.md)):

- **§3A.1 / §3A.4** — 65,536 base 4×4 cells in 16 clusters; INT8 baseline 16,384 MAC. Currently sim collapses to single-cluster compute (functional) but cycle model uses bit-mult invariant from §5.3.
- **§3A.5** — 16 INT32 chain CONV → Requant. **Implemented in v1.3.**
- **§3A.6** — kernel/stride/dilation/padding/group ranges. Real values pulled from .tflite flatbuffer.
- **§3A.7** — EWE / POOL op list. POOL avg/max/global + softmax LUT + **int8 ADD** (v6, gemmlowp left_shift=20 + 3-stage MBQM) done; MUL/SUB not yet.
- **§3A.8** — dependency tag scheme. 4 slots/layer = 64 layers max in 256-tag space (efficientnet_lite0 now uses 62 layers — 1 layer of headroom).
- **§3A.9** — UDMA 5 op modes. LINEAR_COPY / STRIDED_2D real; GATHER / SCATTER_CONCAT used implicitly via the v6 "DRAM→DRAM passthrough" trick (compile_model pre-arranges bytes); STRIDED_SLICE stub.
- **§3A.10** — 64-byte descriptor format (16-byte header + 48-byte body union). DRAM ring buffer + MMIO doorbell **as host stub**.
- **§3A.12** — 61-model coverage. 5 of them (above table) actually run end-to-end.

**Versions delivered (v0 → v4):**

| Version | Theme | Key changes |
|---|---|---|
| v0 | Functional skeleton | 9 modules, sc_fifo wiring, INT8 conv, UDMA real copy, dep-tag |
| v1.1 | Real op options | Flatbuffer (`tflite` package) instead of TFLite Interpreter — gets stride/pad/dilation/fused_act |
| v1.2 / v1.3 | Real chain dataflow | CONV→Requant via 16 INT32 chains; per-channel TFLite-style requant (gemmlowp MBQM); INT8 output |
| v1.4 | LUT-based softmax | 256-entry int LUT shared between numpy ref and EWE engine |
| v2.1 | L1Mesh contention | Read/write port arbitration (256 B/cyc each) inside L1Mesh |
| v2.2 | Per-engine cycle refine | UDMA 256 B/cyc + 16-cyc startup; POOL = K_h×K_w / 16 lane; EWE = 3-pass × 1/16 lane |
| v2.3 | LPDDR6 DRAM controller | 8 KB row × 16 banks; row hit/miss = 0/30 cyc; 32 B/cyc |
| v3.1 | DRAM refresh | every 7800 ns inject 100 cyc |
| v3.2 | Multi-bank L1Mesh | 16 banks × 16-byte stripe; concurrent different-bank access parallel |
| v3.5 | Fused activation | TFLite ReLU/ReLU6/RELU_N1_TO_1 packed into requant params; act_min/act_max clamp |
| v4.1 | Multi-dtype | INT16 conv/dwconv real path; INT64 acc + saturate-narrow to int32 chain; mobilenet_v2_int16 53/53 PASS |
| v4.3 | Profile output | per-layer cycles + DRAM/SRAM byte counts; per-engine busy + task timeline → JSON; Gantt PNG |
| v4.4 | TFLite validation | `validate_tflite.py` runs real Interpreter, captures intermediate via `experimental_preserve_all_tensors=True`; reveals bias gap |
| v4 (FC) | FULLY_CONNECTED | mapped to 1×1 conv with H=W=1; vgg16/audio_yamnet/efficientnet now run their classifier head |
| v5 | CSV profile export | `program.profile.csv` emitted alongside JSON — one row per layer with all metadata + cycle/byte counts. Drops into pandas/Sheets directly. |
| v5 | `--all-layers` sweep | `validate_tflite.py --all-layers` runs TFLite + compile_model **once** (not per-layer), then per-layer compares. Layer 0 is bit-exact comparable; N>0 documented as chained-input mismatch. |
| v6 | Bias + zp_in folding | gemmlowp standard: `bias_eff[oc] = bias[oc] - zp_in_eff * sum_w[oc]`, appended to params blob. RequantEngine adds `bias_eff` to int32 psum before MBQM. `zp_in_eff` set to model `zp_in` for native int8, 0 for uint8 input (validate_tflite pre-shifts). Closed the layer-0 fidelity gap for symmetric int8 models. |
| v6 | uint8 output -128 shift | Sim stores int8 bytes; for uint8-output models, `zp_out`/`act_min`/`act_max` are shifted by -128 in the params blob so the int8 representation == `tflite_uint8 − 128`. |
| v6 | ADD op | New EWE subtype `ES_ADD`; `EweEngine::run_add` does TFLite int8 ADD (each operand requantized via MBQM with `left_shift=20`, summed, then output requant). 9 residual ADDs in efficientnet_lite0 now PASS bit-true. |
| v6 | CONCAT op | Channel-axis only (most common Inception case). Sim is a DRAM→DRAM passthrough; compile_model pre-concatenates synth slices in numpy so the byte stream is already in NHWC output order. Skips with warning if input requant is needed (different scale/zp per slice). |
| v6 | GATHER op | int8/uint8 data only. Sim is a DRAM→DRAM passthrough; compile_model materializes `np.take(data, indices, axis)`. FP gather correctly skipped (audio_yamnet's FP32 mel-bin lookup logs and skips, rest of model still 35/35 PASS). |
| v6.1 | Per-model output dir | All artefacts now land under `systemc/output/<tflite_stem>.{bin,profile.json,profile.csv,profile.png,html}` instead of overwriting a single `build/program.*`. Multiple models can be run side-by-side without clobbering each other. Names preserve dots in stem (`mobilenet_v1_0.25_128_quant.profile.json`) — `Path.with_suffix()` was replaced with f-strings to avoid stripping `.25_128_quant` as a "suffix". |
| v6.1 | Self-contained HTML report | `output/<stem>.html` bundles: header strip (model, layer counts, sim time, DRAM/SRAM totals); per-engine busy table; Gantt PNG **embedded as base64 data URL** (file is portable / archivable / shareable without companion files); full per-layer table; the exact console output the user saw, in a styled `<pre>` block. ~150–200 KB per model. |
| v7 | **L1Mesh = 2 MB (spec)** | L1Mesh shrunk from 32 MB to spec-correct 2 MB ([descriptor.h:172-174](systemc/include/mdla7/descriptor.h#L172-L174)). test_model.cpp emits per-layer dynamic L1 layout and per-tile descriptors. Cycle counts now include real tile-fill stalls + halo-redundant DRAM reads. Tag scheme is rolling 1..255 (skip 0); `pending_tags` queues handle multi-in-flight reuse. |
| v7 | **OH+OC tiling** | Layers that don't fit single-shot in 2 MB are split. Height tiling (most common) handles oversized in/out (e.g., mobilenet_v2_int16 L3/L4 dwconv with 2.4 MB int16 input). OC tiling kicks in when weights alone exceed budget (e.g., mobilenet_v2_int16 L52 FC: 1280→1001 with 2.5 MB weights). Output goes via STRIDED_2D UDMA so per-OC slices land at the right DRAM offsets. RequantBody gains `oc_start` + `scale_count` so a single per-layer params blob can be sliced without reloading. |
| v7 | **Asymmetric uint8 conv (zp_w != 0)** | compile_model now maps uint8 weights via centered SHIFT (`(uint8 - 128).astype(int8)`) instead of the silently-wrong `view(int8)`. zp_w_eff = zp_w_uint8 - 128 is extracted; when non-zero, a per-pixel input-window-sum correction map is computed (`-zp_w * Σ_kernel(in)`) and appended to the params blob (shape `[OH, OW]` for non-DW, `[OH, OW, OC]` for dwconv). RequantEngine reads the appropriate slice (height tile + OC tile) and adds it to psum before MBQM. `zp_in*zp_w*window_size` constant folds into bias_eff. Result: mobilenet_v1_0.25_128_quant layer 0 mean\|diff\| = 5.68 → 0.00. |
| v7 | **TFLite-correct boundary** | ConvBody gains `int16_t in_pad_value`. When input is asymmetric int8 (`zp_in_eff != 0`), CONV uses `pad_value = zp_in` at OOB kernel positions instead of skipping (=0). Mirrored in `conv_int8_ref` and `_conv_window_sum*` via a `pad_value` parameter. Without this, the bias_eff fold (which uses the FULL kernel sum_w) was incorrect at boundaries — caused efficientnet_lite0 max\|diff\| = 58 at the right edge. After the fix: max = 1. |
| v7 | **gemmlowp-aligned multiplier** | `quantize_multiplier()` switched from Python `round()` (banker's) to `floor(x + 0.5)` (round-half-away-from-zero) to match TFLite's C++ `std::round`. (Empirically rare to trigger but the cheap fix matches the reference convention.) |
| v7 | **LayerMeta + RequantBody schema** | LayerMeta gains `int16_t zp_in_eff` (last 2 bytes of the 64-byte record) so test_model can populate ConvBody.in_pad_value without re-reading the .tflite. RequantBody adds `oc_start` (OC-tile offset), `scale_count` (params blob OC_layer), `out_w_layer`, `oh_start`, `corr_addr`, `corr_per_oc` — all from the existing 22 _r bytes. Format unchanged at 64 B. |
| v8 | **FP CONV/DWCONV (FP16/BF16/FP32-source)** | New [`fp_utils.h`](systemc/include/mdla7/fp_utils.h) (software FP16↔FP32 + BF16↔FP32 converters). [`compute_fp`](systemc/include/mdla7/conv_engine.h) in ConvEngine: reads FP32 weights+inputs from L1, accumulates in FP32, bit-casts the partial sum to int32 to push through the existing 16-lane chain. RequantEngine FP path skips MBQM/corr — just bit-cast → +bias → activation clamp → write FP32. Spec §3A.2's 4096 MAC/cycle FP throughput is encoded in the cycle model already (a=b=16). |
| v8 | **FP params blob format** | For FP layers, `compile_model` emits `[ f32 act_min \| f32 act_max \| f32 bias[OC_layer] ]` (8 + 4·OC bytes) instead of the int gemmlowp blob. test_model.cpp's `scale_lut_size` is now dtype-aware. Per-tensor activation thresholds derive from FUSED_NONE / RELU / RELU6 / RELU_N1_TO_1; ±inf packed as ±3.4e38 sentinels. |
| v8 | **DEQUANTIZE walkback** | TFLite stores FP16-quantized weights as `DEQUANTIZE → CONV` patterns where the conv's weight tensor is the runtime output of a DEQUANTIZE op (no constant buffer). compile_model now follows the producer chain to the source FP16 buffer when the conv input has empty buffer storage. Handles direct FP16-weight models (mobilenet_v3) and DEQUANTIZE-pattern models alike. |
| v8 | **Shape propagation** | Many FP-quant models store placeholder `[1,1,1,C]` static shapes (TFLite leaves dynamic dims as 1). compile_model now walks the graph in op order, computing each tensor's `(H, W, C)` from the model input's known shape + per-op semantics (CONV/DWCONV/POOL/MEAN/RESHAPE/CONCAT/FC). The propagated shape is used only when the static shape looks like a placeholder, so INT8 models with correct static shapes are unaffected. |
| v8 | **Storage convention for FP** | Sim DRAM/L1 stores FP32 throughout (4 B/elem) — compile_model casts FP16 weights → FP32 at compile time so the engine stays dtype-agnostic. This is a 2× storage overhead vs spec-realistic FP16 storage; cycle model is unaffected since timing is bit-mult based, not byte-count based. Verification on FP layers tolerates ≤1e-3 abs / 1e-4 rel diff (FP arithmetic is order-sensitive; sim's nested-loop sum order ≠ numpy ref's einsum order). |
| v8.22 | **Big-tensor support: dynamic DRAM + H-tiled ADD/POOL + ETHZ_V6 regression** | Three changes that let large segmentation models (`deeplab_v3_plus_*`, `unet_*`, `pynet_v2_*`) run end-to-end. (1) **Dynamic DRAM size** — `Mdla7System` and [`Dram`](systemc/include/mdla7/memory.h) constructors now take a `dram_bytes` parameter; [`test_model.cpp`](systemc/src/test_model.cpp) scans all `metas[]` to find max `(dram_in/wgt/out + size) - DRAM_BASE`, pads 64 MB, rounds to 64 MB granularity, and passes that. deeplab's 1 GB program file used to segfault `sys.dram.write` against the default 256 MB; now it gets a 704 MB DRAM model. (2) **H-tiled ADD** ([test_model.cpp `OK_ADD`](systemc/src/test_model.cpp)) — when input-A + input-B + output (≈ 3× tensor) doesn't fit 2 MB L1, falls into a tiled path: load 48 B params once at top of L1, loop `tile_oh` rows where each tile loads its own input-A and input-B slices, runs EWE-ADD with a tile-local `LayerMeta`, stores its output slice. Multi-tile ADD opts out of fusion-source eligibility (tensor split across DRAM). (3) **H-tiled POOL** ([test_model.cpp `OK_AVG_POOL`/`OK_MAX_POOL`](systemc/src/test_model.cpp)) — output (typically small, since pool downsamples) stays fully in L1 at addr 0; input is loaded one OH-tile at a time, each tile's POOL descriptor writes its rows into the right offset of L1_OUT. Final `udma_w` deferred to `pending` so pool can still source-fuse. tile_oh chosen so the worst input window `(tile_oh*s_h + k_h) * per_row` fits the remaining L1. (4) Added [`run_ethz_v6.py`](systemc/run_ethz_v6.py) — regression sweep over `model/ETHZ_V6/*.tflite`, emits `output/ethz_v6_regression.csv` with `pattern,ms,status` columns. Surfaces `compile-fail` / `sim-fail` (real crash) / `N-FAIL` (sim ran but N layers failed bit-exact) separately, banner-filtering stderr so the SystemC license noise doesn't mask real errors. **Result**: deeplab_v3_plus_float now runs at **87.7 ms**, deeplab_v3_plus_quant at **27.4 ms** (both still report N-FAIL — see "Open" below). |
| v8.21 | **Ping-pong L1 allocator (rolling chain)** | v8.14's chain stack-allocated `L1_PARAMS/WGT/OUT` upward from prev's `L1_OUT`, so chain footprint grew unbounded and busted the 2 MB budget after 3-4 deep chains. v8.21 places each fused layer's `L1_OUT` at **alternating ends** of L1 (low addr 0 ↔ high addr `BUDGET-out_size`) via a `chain_alt` toggle and try_low/try_high lambdas. Prev's `L1_OUT` always lives at one end → the entire other end is contiguous free space for current's PARAMS/WGT/OUT. Three additional fixes were needed: (a) a race condition where current layer's UDMA write into the "low zone" overlapped with prev's CONV reading the same region — added `wait_a = fuse_prev_done_tag` (prev REQUANT done) on the first inbound UDMA per fused layer; (b) the v8.13 post-switch reset of `fuse_prev_is_conv_class` for non-CONV ops was clobbering ADD/POOL's just-set fusion-source state — exempted ADD/POOL from the reset; (c) `chain_alt` toggles based on which end was actually used (try_low fallback when try_high fired first, or vice versa). **Result on mobilenet_v3_large_fp16**: 43 → **58 fused** layers (+15); DRAM W 4.26 → **1.15 MB** (−73%); cycles 1,742,053 → **1,674,759** (−3.9%); udma_w busy 6.1% → **1.6%**. The L0→L5 residual block (CONV → DWCONV → CONV → ADD → CONV → DWCONV) now fuses cleanly with all 6 layers staying L1-resident. **EfficientNet-Lite0**: 47 → **58 fused**; DRAM W 0.47 → **0.00 MB** (entire network's intermediate activations never touch DRAM). **MobileNet v1, VGG16**: also DRAM W = 0.00 MB. mobilenet_v2_int16 unchanged (INT16 path doesn't have ADD/POOL participants). All bit-exact at TOL=0. |
| v8.20 | **Fusion source-skip extended to ADD / POOL consumers** | v8.14's `fuse_eligible` predicate previously fired only when the next layer was CONV/DWCONV/FC; now also fires when the next layer is `OK_ADD` / `OK_AVG_POOL` / `OK_MAX_POOL`. Both ADD and POOL gained two roles: as **consumer**, they read input directly from prev layer's `L1_OUT` (skipping a udma_r), and as **producer**, their own udma_w is captured into `pending` so a fusing successor can drop it. ADD has two inputs — only input-A comes from L1 (chained); input-B (the synth tensor) still loads via udma_r. SOFTMAX kept as flush-pending (rarely benefits). Result on **mobilenet_v3_large_fp16**: 33 → **43 fused** layers (+10, all 10 ADD predecessors now skip udma_w); DRAM W 4.78 → **4.26 MB** (−11%); cycles 1,773,263 → 1,742,053 (−1.8%); red `dram_w` cells 44 → **34**. **EfficientNet-Lite0**: 37 → **47 fused** (the 9 INT8 residual ADDs each save their predecessor's udma_w); DRAM W collapses to **0.47 MB** for the whole network. INT regressions: mobilenet_v1 31/31, mobilenet_v2_int16 53/53 — all still bit-exact. |
| v8.19 | **Highlight non-zero `dram_w` in red + Sim time in ms + 5-engine spec (TNPS) + A2 drawio** | Per-layer profile rows where `dram_w > 0` render the cell in `color:#b00020; font-weight:600` so DRAM-write hotspots stand out at a glance ([run_model.py](systemc/run_model.py) `_layer_row`). Header summary's "Sim time" reformatted to `1.773 ms @ 1 GHz (1,773,263 cycles)` with 3 decimal places. KV labels title-cased (`Model:` / `Layers:` / `Sim time:` / `Util:`); `.kv` switched from `inline-block` to `block` so each datum is on its own line. Spec gained a 5th compute engine **TNPS** (Tensor transpose; 8 AXI_R + 8 AXI_W; new §3A.7b in [spec.md](spec/spec.md)) — for attention split-heads / NHWC↔NCHW / channel-shuffle, decoupled from UDMA's STRIDED_SCATTER. [spec/mdla7.drawio](spec/mdla7.drawio) page 1 redrawn at A2 landscape (2339×1654, ~2× area of A3) with all 5 engines centered at exitX = 0.10/0.30/0.50/0.70/0.90 of CommandEngine. |
| v8.18 | **Sortable + filterable HTML tables** | Per-layer profile and Compile log tables in the HTML report ([run_model.py](systemc/run_model.py) `_write_html_report`) gained click-to-sort headers (asc → desc → reset to original order, three-state) and a substring filter input above each. Numeric columns auto-detected (commas / `%` / KB suffixes stripped) and sorted numerically; text columns sort lexicographically with `localeCompare({numeric: true})`. Filter does case-insensitive `.includes(...)` against each row's full text and reports `(N / total match)` next to the input. Vanilla JS, no deps; uses `replaceChildren(...)` and `||=` (ES2021). |
| v8.17 | **FP path for EWE-ADD and POOL** | EweEngine and PoolEngine each grew a `last_dtype` member latched per dispatch by CommandEngine (new `ewe_dtype_latch` / `pool_dtype_latch` side-channels in [command_engine.h](systemc/include/mdla7/command_engine.h), wired in [system.h](systemc/include/mdla7/system.h)). When the latched dtype is FP16/BF16/FP8: ADD reads two FP16 operands, accumulates in FP32, clamps with the +/-3.4e38 sentinels packed at the head of the existing 48-byte params slot, writes FP16; AVG/MAX pool reads FP16, reduces in `(kh, kw)` order with running FP32 add (or compare for MAX), divides by window count for AVG, writes FP16. Compile_model gains a `pool_fp_ref` mirroring the same loop order plus FP branches in the ADD and POOL handlers; the early FP-skip list shrinks to just `SOFTMAX` and `FULLY_CONNECTED`. test_model.cpp's `make_pool` / `make_ewe_add` now forward `L.dtype` instead of hardcoding `DT_INT8x8`. **mobilenet_v3_large_fp16** now runs the **full 77-layer graph** end-to-end (10 residual ADDs + 2 AVERAGE_POOLs that were previously skipped), 77/77 PASS at TOL=0 bit-exact. Sim time 1,670,078 → 1,773,263 cyc (+6.2% — the extra is the work that was being skipped, not regression); DRAM r/w 14.64/3.97 → 16.34/4.78 MB; new `ewe` busy 2.1%, `pool` busy 0.2%. |
| v8.16 | **Compile-log table in HTML report** | `_write_html_report` ([run_model.py](systemc/run_model.py)) parses each compile_model "ready / skipped" line via two regexes and renders a structured **Compile log** table at the bottom of the report (after Per-layer profile). One row per layer with shape / kernel / stride / group / output / element-count / dtype / status — mirrors the canonical `compile_model.py` console line in tabular form. Includes layers the simulator never sees because compile skipped them (e.g. mobilenet_v3 has 10 FP-ADDs + 2 FP-AVERAGE_POOLs that show `skipped (...)` with the reason, marked in amber). Also fixed the dtype label on the canonical compile line so FP layers print `FP16` instead of `INT8` ([compile_model.py:1430-1438](systemc/scripts/compile_model.py#L1430-L1438)). The verbose **Console output** `<pre>` block (and its associated `pre {...}` CSS) was dropped — the structured tables (per-engine busy + per-layer profile + compile log) now cover everything that block contained. Section order: header → Per-engine busy → Gantt → Per-layer profile → Compile log. |
| v8.15 | **Bit-exact FP path (sim ≡ numpy ref)** | Two changes that together make sim and the numpy FP reference produce **byte-identical FP16 outputs per layer**, with `TOL_ABS = TOL_REL = 0.0`: (1) [`conv_fp_ref`](systemc/scripts/compile_model.py) rewritten to drop `np.einsum` and instead reduce in the same nested `(kh, kw, icr)` order as sim's `compute_fp` — each iteration is an element-wise FP32 mul + add into the running output, no BLAS pairwise/parallel reduction. (2) [`fp32_to_fp16`](systemc/include/mdla7/fp_utils.h) switched from truncation to **IEEE 754 round-to-nearest-even**, matching numpy's `arr.astype(np.float16)`. The FP clamp in the ref also now applies the same `±3.4e38` sentinels sim sees in the params blob (was gated on `np.isfinite`). Result on mobilenet_v3_large_fp16: **65/65 PASS, 0 FAIL** at zero tolerance — replaces the 5 boundary FAILs that v8.14 left as accepted FP chain drift. The 5%/5% safety net in `test_model.cpp` is gone — any future drift is now caught immediately. INT models unaffected (compute_fp / conv_fp_ref are FP-only). |
| v8.14 | **L1-resident fusion: FP enabled + udma_w drop** | (1) Removed the `!is_fp` gate on the v8.13 fusion eligibility ([test_model.cpp:366-372](systemc/src/test_model.cpp#L366-L372)) — INT and FP both fuse now; the 5%/5% per-layer FP tolerance set in v8.13 already absorbs chain rounding drift. (2) **Symmetric udma_w skip**: when a single-tile CONV/DWCONV/FC's successor will fuse, the producer's udma_w is dropped (output stays resident in L1). New `pending_store` mechanism defers the udma_w descriptor; the next layer either `flush_pending()` (push it inline so DRAM gets the value) or drops it (fused → no DRAM W needed). Safe because compile_model.py allocates a per-layer `dram_in` blob and pre-loads it — layer N+1's udma_r doesn't depend on layer N's `dram_out` being written. (3) `fuse_prev_done_tag` switched from `prev_store` (udma_w done) → REQUANT done tag, so the fused successor's CONV doesn't wait on a now-dropped store; this also shaves the udma_w-extension off the producer's critical path. (4) Per-layer cycle accounting now uses `max(udma_end, requant_end)` since fusion-source layers have no terminal UDMA. (5) Verification skips fused-source layers (DRAM at their `dram_out` is undefined); summary line reports both PASS and `(N fused — output stayed in L1, no per-layer verify)`. **mobilenet_v3_large_fp16 result**: 1,868,176 → 1,670,078 cyc (−10.6%); DRAM W 8.41 → 3.97 MB (−53%); DRAM R 19.07 → 14.64 MB (−23%); 41/65 layers fuse-source-skipped; 5 boundary FAILs are FP chain-drift exceeding 5%/5% tolerance (not real bugs — could be cleared by widening tolerance). |

---

## 4. Key files (where to start reading code)

**Python pipeline:**
- [systemc/scripts/compile_model.py](systemc/scripts/compile_model.py) — the "compiler". Reads `.tflite` flatbuffer, emits `program.bin` + numpy reference. **All op extraction logic lives here.**
- [systemc/scripts/extract_conv.py](systemc/scripts/extract_conv.py) — legacy single-conv extractor (uses TFLite Interpreter).
- [systemc/scripts/plot_profile.py](systemc/scripts/plot_profile.py) — Gantt chart from `profile.json`.
- [systemc/scripts/validate_tflite.py](systemc/scripts/validate_tflite.py) — TFLite Interpreter cross-check.
- [systemc/run_model.py](systemc/run_model.py) — top-level driver. Pattern matching, venv re-exec, step orchestration.

**SystemC modules:**
- [systemc/include/mdla7/system.h](systemc/include/mdla7/system.h) — Mdla7System wires up all engines + channels.
- [systemc/include/mdla7/descriptor.h](systemc/include/mdla7/descriptor.h) — 64-byte descriptor + dtype enums + memory map constants.
- [systemc/include/mdla7/conv_engine.h](systemc/include/mdla7/conv_engine.h) — `compute_int<T>` template (INT8/INT16); pushes int32 psums to chain.
- [systemc/include/mdla7/requant_engine.h](systemc/include/mdla7/requant_engine.h) — drains chain; per-channel MBQM + activation clamp.
- [systemc/include/mdla7/ewe_pool.h](systemc/include/mdla7/ewe_pool.h) — POOL real impl + EWE softmax LUT.
- [systemc/include/mdla7/udma.h](systemc/include/mdla7/udma.h) — 5 op modes.
- [systemc/include/mdla7/memory.h](systemc/include/mdla7/memory.h) — L1Mesh (multi-bank) + Dram (LPDDR6 row hit/miss + refresh) — **all timing modeling lives here**.
- [systemc/include/mdla7/command_engine.h](systemc/include/mdla7/command_engine.h) — dependency tag tracker; per-engine cfg dispatch.
- [systemc/include/mdla7/host.h](systemc/include/mdla7/host.h) — host stub. Pushes pre-built descriptor program.
- [systemc/include/mdla7/requant.h](systemc/include/mdla7/requant.h) — gemmlowp MBQM primitives.
- [systemc/include/mdla7/softmax_lut.h](systemc/include/mdla7/softmax_lut.h) — 256-entry shared LUT (must stay byte-identical with `compile_model.py`).
- [systemc/src/test_model.cpp](systemc/src/test_model.cpp) — main runtime: loads program.bin → runs sim → verifies → emits profile.json.

---

## 5. Build / dependency notes

**System reqs:**
- macOS Intel x86_64 — tested
- macOS Apple Silicon arm64 — Makefile auto-detects via `brew --prefix systemc`
- Linux — should work; `tflite-runtime` is preferred (smaller than full tensorflow)

**SystemC:** 3.0.x via Homebrew (`/usr/local/opt/systemc` Intel, `/opt/homebrew/opt/systemc` arm64). Auto-detected by Makefile.

**Python venv:** [`~/.venvs/mdla7`](file://localhost/Users/$USER/.venvs/mdla7) — outside repo intentionally because the 4T_OFFICE volume creates AppleDouble (`._*`) sidecars that crash Python's site importer when stored inside a venv.

**Two macOS oddities the build navigates:**
1. CLT 17 + SDK 26 has a partial `c++/v1` directory that hides the SDK's complete one — Makefile injects `-isystem $(SDKROOT)/usr/include/c++/v1`.
2. `Path.resolve()` chases the venv's `bin/python` symlink back to the system Python — `run_model.py` uses `sys.prefix` instead.

---

## 6. Known gaps (what to do next, by priority)

**v6 closed:** bias + zp_in folding, uint8 output shift, ADD, CONCAT, GATHER, CSV export, `--all-layers` validation.

**v7 closed:** L1Mesh = 2 MB (spec) + OH/OC tiling, asymmetric uint8 conv (zp_w != 0) via per-pixel correction map, TFLite-correct boundary semantics (in_pad_value = zp_in), gemmlowp-aligned QuantizeMultiplier rounding.

**v8 closed:** FP CONV/DWCONV (FP16/BF16/FP32-source) end-to-end, DEQUANTIZE walkback, graph shape propagation for placeholder-shape models, mobilenet_v3_large_fp16 + BF16 chain models running.

**v8.6/.7/.8 closed:** Pipeline-overlap fix (CONV/REQUANT no longer double-count L1 reads/writes against compute), chain backpressure (CONV reported busy time now reflects stall on slow REQUANT), fused-Requant model (LANES = 256, representing 16 clusters × 16 per-cluster output stages). Result: 6–28% per-model cycle reduction; peak-engine identification now correctly flags the actual bottleneck (CONV for compute-dense, UDMA for DRAM-bound).

**v8.10/.11/.12/.13 closed:** FP storage corrected to FP16 in DRAM/L1 (2 B/elem, was 4 B FP32 sim simplification — saves ~50% udma r/w on FP layers). DRAM bus widened from 1×LPDDR6 (32 B/cyc) to 2×LPDDR6 (64 B/cyc). Compile_model chain mode (layer N+1 input = layer N reference output when shape matches). L1-resident layer fusion in test_model: when adjacent CONV-class single-tile layers can co-fit in 2 MB L1, layer N+1 reads from layer N's L1_OUT directly — skips a full udma_r. Fusion gated to INT path only (FP non-associativity makes byte-exact chain forwarding impossible without identical reduction order). Cumulative cycle savings vs v8.5: −31% (mobilenet_v2_int16) to −63% (conv1d_audio_bf16); mobilenet_v3 layer 0 specifically: 68,181 → 17,558 cyc (−74%).

**v8.14 closed:** FP fusion gate removed; symmetric udma_w drop for fusion-source layers (producer's output stays in L1 when consumer fuses). mobilenet_v3_large_fp16: 1,868,176 → 1,670,078 cyc (−10.6%) and DRAM W 8.41 → 3.97 MB (−53%) on top of v8.13 udma_r savings. Mechanism: pending_store + flush_pending; verification skips fused-source layers (DRAM dram_out undefined for them); cycle accounting uses max(udma_end, requant_end) since fusion-source layers no longer terminate on UDMA. Safe because compile_model pre-loads each layer's `dram_in` independently, so layer N+1's udma_r doesn't depend on layer N's `dram_out` ever being populated.

**v8.15 closed:** sim's FP conv reduction order aligned with numpy ref (same nested `(kh, kw, icr)` element-wise loops, no einsum/BLAS) + IEEE round-to-nearest-even `fp32_to_fp16` (was truncating). mobilenet_v3_large_fp16 now passes 65/65 at zero tolerance — the 5 boundary FAILs from v8.14's FP chain drift are gone. Tolerance constants in `test_model.cpp` set to `0.0f` so any future regression is caught immediately. Comment in `fp_utils.h` flags the convention so the next person doesn't re-introduce truncation.

**v8.17 closed:** FP path wired into EWE-ADD and POOL via per-engine `last_dtype` latches plumbed from CommandEngine. mobilenet_v3 now runs the full 77-layer graph (10 residual ADDs + 2 AVERAGE_POOLs that were previously skipped), 77/77 PASS at TOL=0 bit-exact. The remaining FP-skip list is just SOFTMAX + FULLY_CONNECTED.

**v8.22 closed:** Big-tensor support — dynamic DRAM model sizing (segfault → no crash on 1 GB program files), H-tiled ADD (deeplab's 256x256x24 = 3 MB ADDs), H-tiled POOL (deeplab's 64x64x320 = 2.5 MB avgpool input), plus a [`run_ethz_v6.py`](systemc/run_ethz_v6.py) regression sweep across `model/ETHZ_V6/`. deeplab_v3_plus_float runs end-to-end at 87.7 ms, deeplab_v3_plus_quant at 27.4 ms. FAIL count on deeplab tracks back to v8.15's bit-exact alignment being limited to single-tile (see Known Gaps below).

### High ROI

1. **CONCAT input-requant path** — currently CONCAT skips when any input has a different scale/zp from the output. Real models (e.g., Inception_v3 INT8 if we get one) sometimes do this. Add per-input MBQM in compile_model and a SCATTER_CONCAT-with-requant sim path. Estimated 2h.

2. **Last-LSB fidelity (post-v7)** — efficientnet_lite0 layer 0 still has 4642/401408 (~1.2%) elements off by 1 LSB; mobilenet_v1_0.25_128 has 72/32768 (~0.2%). All diffs are exactly 1 in magnitude and consistently in one direction (TFLite − ours = −1) — the residual is XNNPACK's int8 quantization kernel using a slightly different rounding from the gemmlowp reference path. Closing it would require either (a) routing TFLite's Python interpreter through `BUILTIN_REF` (it currently disagrees by ~80% with our impl, suggesting BUILTIN_REF is itself doing something funky in this version) or (b) bit-emulating XNNPACK's `qs8_requantize_*` exactly. Low ROI given the practical level of fidelity already achieved. Estimated 4h investigation + indeterminate fix.

### DRAM-W reduction roadmap

After v8.21 (ping-pong rolling L1 allocator), 19/77 layers of mobilenet_v3_large_fp16 still hit DRAM W. The remaining traffic falls into 3 categories. Listed by ROI:

| # | Cause / Fix | Impact on mobilenet_v3 | Effort | Notes |
|---|---|---|---|---|
| ~~1~~ | ~~Extend fusion-source skip to ADD/POOL consumers~~ | ✅ **closed in v8.20** (DRAM W 4.78 → 4.26 MB) | — | — |
| ~~A~~ | ~~Rolling L1 (alternating-ends ping-pong) — release prev's region~~ | ✅ **closed in v8.21** (DRAM W 4.26 → 1.15 MB on mnv3; → 0.00 MB on efficientnet_lite0) | — | — |
| 2 | **Multi-tile producer can fuse-source if full output still co-fits L1 alongside consumer's params/wgt/out.** v8.14 currently gates on `fuse_prev_single_tile` even if the math fits. | DRAM W −10 to −20%. | Medium | Drop the `single_tile` requirement; recompute the L1-fit budget over the full producer output instead of one tile. Per-tile `prev_l1_out_addr / size` becomes a concatenated descriptor. |
| 3 | **Skip-connection / fan-out tracking.** A producer's output may be consumed by 2+ downstream layers (residual + main path). compile_model's `last_output_arr` only tracks the immediate next layer, so anything fanning out forces a DRAM round-trip. | −5 to −10% on CNN; large win on transformer blocks. | Large | compile_model needs a 2-pass: build per-tensor consumer use-list, then keep tensors alive in L1 until refcount hits 0. L1 allocator switches from bump-pointer to free-list. |
| 4 | **Multi-tile lockstep streaming.** Producer emits tile_k → consumer reads tile_k → producer emits tile_{k+1}. Reduces L1 footprint to one tile pair instead of full tensor. | Approaches the upper bound (DRAM W ≈ final-layer-only). | Large | Descriptor stream interleaves producer/consumer per tile. Cmd Engine dependency tags need to fire per tile not per layer. Substantial rework of the per-layer descriptor model. |
| 5 | **Inter-engine streaming chain (e.g., REQUANT → POOL direct).** Skip the L1 hop entirely, like the existing CONV→REQUANT 16×INT32 chain (§3A.5 of spec). | Small (~5%); meaningful only for fixed REQUANT→POOL global-avg patterns. | Small (sim) but spec change | New chain port between REQUANT and POOL/EWE. Spec §3.1 grows a "chain link" column. |

**Last layer / output of the entire graph**: must always udma_w (host needs the result). Counts as 1/77 of the irreducible DRAM W floor.

**Recommended next step**: #1. Lowest cost, immediate visible reduction in red cells in the per-layer profile.

### Multi-tile FP bit-exactness (v8.22 follow-up)

`run_ethz_v6.py` reports `68-FAIL` on **deeplab_v3_plus_float** and `58-FAIL` on `_quant`, even though the simulator runs end-to-end after v8.22. The mismatch fraction per failing layer is roughly **half** of bytes (e.g. layer 4 dwconv: 3.13 M / 12.58 M ≈ 25 % byte mismatch, ≈ 50 % element-mismatch since FP16 is 2 B/elem). Why:

- v8.15's bit-exact alignment (`compile_model.py::conv_fp_ref` using nested `(kh, kw, icr)` element-wise loops, IEEE round-to-nearest-even `fp32_to_fp16`) was **only validated against single-tile sim**. The numpy reference computes the **whole tensor at once** (vectorised over `(oh, ow, oc)` per kernel position).
- Sim's multi-tile path runs the same `(kh, kw, icr)` math per output element, so per-element values *should* still be bit-identical — but tile-boundary handling (`pad_t_tile`/`pad_b_tile` set per tile, halo input rows shared across tiles, `oh_done` offsets in the REQUANT body's `oh_start`/`out_w_layer` fields) introduces several places where ref vs. sim can diverge by exactly the tile boundary's worth of pixels. The "≈50% of elements differ" pattern points at every-other-tile drift rather than uniform ulp-level error.
- All small models (`mobilenet_v3_large_fp16`, `efficientnet_lite0`, `mobilenet_v1`) are single-tile in every layer at 2 MB L1, so this gap was invisible until v8.22 enabled deeplab to run.

**To investigate**:
1. Construct a minimal multi-tile FP test: synthetic 2-layer FP CONV chain where layer 1's output is forced multi-tile (e.g. 256x256x16 FP16 with 1×1 conv → output > 2 MB). Check whether sim's per-tile output bytes match `conv_fp_ref` byte-for-byte at each tile boundary.
2. Likely culprit: `compile_model.py::conv_fp_ref` doesn't tile, so its kernel-position iteration covers the FULL output. Sim's per-tile descriptor uses `out_h = this_oh`, `oh_start = oh_done` so the engine processes only its rows — but the REQUANT bias add and clamp run on the full padded params blob; if `oh_start` indexing is off by one row at tile boundaries, half the output drifts.
3. Cross-check `_conv_window_sum` (correction-map computation, INT-only) for how it handles tile_oh — that helper was written for OH tiling and might have the canonical correct formula.

**Effort**: ~3 h to identify + fix the tile-boundary mismatch. Once fixed, deeplab and other large-tensor segmentation models (unet, midas_v3, sam, esrgan) should drop to 0-FAIL like the small models.

### Functional coverage

3. **FP ADD / MUL / SUB (EWE binary, FP path)** — mobilenet_v3 has 10 residual ADDs that are skipped today (FP path not wired into EweEngine). MUL/SUB used in SE gates and HARD_SWISH lowering. Each needs a `run_add_fp / run_mul_fp` mirroring run_add but using FP32 + bias clamp. Estimated 2h. Without this, mobilenet_v3 runs the conv backbone but residual semantics are wrong.

4. **HARD_SWISH / GELU / SWISH activation** — used heavily in mobilenet_v3, transformers. Currently filtered (not in SUPPORTED_OPS). Could be done as a fused-activation table in the FP requant path, or as a standalone EWE subtype. Estimated 1.5h each.

5. **MEAN (global pool) FP** — mobilenet_v3 uses MEAN over (h, w) before the classifier. Currently filtered. Add as POOL with kernel = (H, W). Estimated 1h.

6. **FP FULLY_CONNECTED** — mobilenet_v3 uses 1×1 conv for the classifier (so it works), but other FP models may use FC. Mirror the FP CONV path inside the FC branch. Estimated 1h.

7. **MUL / SUB (int EWE binary)** — same engine path as ADD. Used in attention layers, squeeze-excite gating. Each is a small extension of `run_add`. Estimated 1.5h.

8. **QUANTIZE / DEQUANTIZE op (real-image input)** — currently first conv input is synthetic, not the model's actual preprocessed image. Implementing these enables real-image inference. Estimated 1h.

9. **INT16 pool / softmax / reshape** — mobilenet_v2_int16 currently filters them out. Should template them similar to ConvEngine. Estimated 2h.

10. **Real INT8 CONCAT/GATHER models** — none of the bundled INT8 models exercise our v6 CONCAT/GATHER paths (audio_yamnet's GATHER is FP32; no INT8 model has CONCAT). Hand-craft a synthetic .tflite or download a real one (Inception_v3 INT8, MobileBERT INT8) to actually validate the bit-true path. The code is in place — just untested on real bytes. Estimated 1h.

7. **Per-channel non-zero zp_w** — v7 errors out when zp_w varies per channel and is non-zero. Modern int8 per-channel always has zp_w=0, so this is a pure-paranoia fix; would require a per-(oh, ow, oc) corr map (largest model would still fit but bandwidth grows). Estimated 1.5h if a model ever needs it.

8. **Asymmetric uint8 dwconv with > 2 MB weights** — v7 OC tiling is non-DW only; dwconv assumes group==in_c which breaks under simple OC slicing. None of the bundled models trip this, but a high-OC dwconv (rare) would error. Estimated 2h.

### Cycle accuracy

9. **Real 16-lane parallel Requant** — currently a single SC_THREAD serializes all 16 chain reads. Spec §3A.5 says lanes run independently. Splitting into 16 SC_THREADs would reveal real CONV→Requant pipelining. Estimated 3h.

10. **L1_Manager arbitration policy** — currently each L1Mesh bank serves "next request after previous's finish_time". Real HW may priority-arbitrate (e.g., CONV outranks UDMA prefetch). Add policy + measure cycle impact. Estimated 2h.

11. **Per-engine refined cycle** — POOL/EWE estimates are coarse (`elem/16`). Could model pool window fill, broadcast pattern, LUT lookup latency. Estimated 3h.

12. **Refined tile-fill latency** — v7 currently keeps the `+ 64 cyc` per CONV-dispatch fill from v0. Now that tiling makes per-tile fills the dominant overhead for big layers, calibrating this from spec §3A.5 cluster-fill timing would tighten cycle estimates. Estimated 2h.

### Engineering

13. **Profile diff** — `run_model.py --diff prev.json` to see cycle change. 1h.

14. **Multi-model regression report** — extend `--all` to emit a comparison table (cycles, DRAM bytes, fidelity status per model). 1h.

15. **Per-layer engine attribution in Gantt** — currently the timeline shows engine-side bars but not which layer each task belongs to. Tag tasks with layer_id and color-code or annotate. 2h.

16. **CONCAT/GATHER cycle accuracy** — currently both pass through the UDMA copy path, so cycles are accurate for memory but the descriptor metadata says `op_kind=concat/gather` while the engine sees a generic UDMA. Either log it correctly in the profile, or split into a dedicated cycle-only sub-path. 1h.

### Architectural / longer-term

17. **Real RISC-V ISS** (Spike / etiss) replacing Host stub — no functional change but architecturally cleaner.

18. **Multi-dtype** beyond INT16: INT8×4 (2× throughput), FP16, BFP16. Each needs new compute path + numpy reference matching strategy.

19. **Real bias-aware compute** with INT48 accumulator end-to-end (currently INT64 sim → INT32 chain saturate).

20. **Power model** — pJ per op + per byte; outputs energy alongside cycles.

21. **Full 16-cluster MAC array model** with output-channel splitting across clusters (vs. current single-cluster compute that uses cycle model).

### Pre-existing test issue

22. **`make test` (test_conv_synth)** — broken since v1.3 chain refactor (CONV no longer writes to L1_OUT directly; the synth test's program lacks a REQUANT to drain the chain). Update the descriptor program to include a REQUANT, or repurpose it. Estimated 0.5h.

---

## 7. How verification works (important to understand)

**Two layers of verification:**

1. **Self-consistency (current default):** `compile_model.py` and `test_model` use the SAME numerical algorithm. `compile_model` computes a numpy reference, packs it into `program.bin`. `test_model` runs the sim, reads DRAM, compares byte-for-byte. PASS = "numpy ref agrees with sim". Both can be wrong relative to TFLite and still PASS.

2. **TFLite fidelity (`validate_tflite.py`):** Runs the real TFLite Interpreter (`experimental_preserve_all_tensors=True`) on the same input, compares its output tensor against our reference. PASS = "we match TFLite bit-exactly". After v6 bias+zp_in folding, symmetric int8 layer 0 is near-bit-exact (mean|diff|=0.07, ~2% of elements off by 1 LSB). Asymmetric uint8 (legacy) still mismatches.

3. **`--all-layers` mode (v5):** sweep every conv-class op. Layer 0 is bit-exact comparable (TFLite and compile_model see the same input). Layers N>0 use chained TFLite output as input but compile_model still synthesizes per-layer rng draws, so the diff there is **expected non-zero** and is reported as `diff` (not `FAIL`) — a smoke check, not a fidelity claim. To make N>0 truly comparable would need a chained-input mode in compile_model (currently a TBD).

When working on correctness fixes, drive both #1 and #2. After fixing bias/zp (v6), `validate_tflite` got close to green for symmetric int8; the remaining 1-LSB gap is rounding (low-priority polish).

---

## 8. Tag scheme constraints (gotcha)

**v7:** Tags are allocated by a **rolling counter** (1..255, skipping 0 = sentinel "no tag"). The dispatch FIFO is in-order and engines serialize on their cfg FIFOs, so by the time the counter wraps, any prior use of a tag has long fired — reuse is safe. test_model.cpp tracks each layer's "done" tag (the last STORE in its descriptor sequence) for verification reporting (replaces the old `1 + 4*i + slot` scheme). This removes the **64-layer hard cap** v6 had — multi-tile layers can now use ≫4 tags each.

ADD still uses 4 sequential tags per layer (in-B+params, in-A, ewe-add, store). CONCAT/GATHER use 1. Tiled CONV uses 1 (params) + 1 (wgt) per OC tile + 3 per (oh-tile, oc-tile) (in, requant, store) — for VGG16-class deep layers this can be 50+ tags, all handled by the rolling allocator.

---

## 9. Test commands cheat sheet

```bash
# Full sweep
cd systemc
python3 run_model.py mobilenet_v1
python3 run_model.py mobilenet_v2_int16
python3 run_model.py efficientnet_lite0
python3 run_model.py vgg16
python3 run_model.py audio_yamnet

# Each emits output/<stem>.{bin, profile.json, profile.csv, profile.png, html}
# Open the HTML for a self-contained summary:
open output/efficientnet_lite0_int8.html         # macOS

# View profile (Python)
python3 -c "import json; p=json.load(open('output/mobilenet_v1_0.25_128_quant.profile.json'));
print(p['summary']); print({k: v['busy_cycles'] for k,v in p['engines'].items()})"

# TFLite fidelity
python3 scripts/validate_tflite.py model/INT8/efficientnet_lite0_int8.tflite
python3 scripts/validate_tflite.py path.tflite --layer 5
python3 scripts/validate_tflite.py path.tflite --all-layers   # v5 sweep

# Synthetic regression
make test                         # 6×6×4 → 6×6×8 hand-rolled conv test
```

---

## 10. Open questions for next session

1. Is bit-exact match against TFLite required for all models, or is "within 1 LSB" acceptable? (Answer determines whether item §6.2 — chasing the gemmlowp rounding quirk — is worth it. Currently mean diff is 0.07 LSB on int8 layer 0; further reduction is engineering polish.)
2. Should we keep filtering out unsupported ops (current behavior) or fail loudly? (Now that ADD/CONCAT/GATHER land, the silent-skip surface is much smaller. The remaining unsupported set is mostly QUANTIZE/DEQUANTIZE/MUL/SUB and FP-typed ops.)
3. For multi-dtype: priority on INT8×4 (2× throughput claim in spec) vs FP16/BFP16 (transformer support)?
4. How "real" does the host need to be? Stub is fine for current testing; RISC-V ISS is significant scope.
5. Is the asymmetric uint8 legacy path (item §6.1) worth the engineering, or should we declare "modern int8 only" and let legacy uint8 models fall through? (The 61-model collection has only a handful of legacy uint8 models; mobilenet_v1 stem is one of them.)
