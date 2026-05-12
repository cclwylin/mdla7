# MDLA7 Verilog

This directory is the real hardware Verilog path. It contains the control path,
L1 fabric, byte movers, and datapath bring-up blocks.

Simulator naming:

- `fast`: analytical SystemC model.
- `synth`: SystemC synth timing model.
- `verilog`: Verilator hardware path with control and datapath.

Scope for this path:

- Replace DPI-C golden datapath CRC generation with real Verilog datapath blocks.
- Add L1Mesh contention tests that intentionally collide multiple requesters on
  the same tile/bank and check backpressure.
- Add placement-aware route timing inputs so L1Mesh latency reflects physical
  tile/bank placement instead of only simplified address-derived hops.

Do not split new work into separate `verilog_ctrl` / `verilog_final` trees.
New hardware work belongs here. The legacy `rtl/synth` control shell and
`run_verilog_ctrl.py` runner have been retired.

Smoke tests:

```sh
./rtl/batch/run_verilog_smoke.py
```

Current smoke coverage:

- `conv`: true Verilog INT8 MAC + bias + MBQM + clamp primitive, INT16
  sample MAC, and a 2D NHWC convolution address-walk primitive.
- `requant`: true Verilog MBQM + output zero-point + activation clamp sample.
- `pool`: true Verilog INT8 max/avg sample reduction.
- `ewe`: true Verilog INT8 ADD/MUL/SUB vector sample with lane saturation, plus
  INT16 ADD/MUL/SUB sample vectors.
- `tnps`: true Verilog TNPS SPACE_TO_DEPTH / DEPTH_TO_SPACE address mapping.
- `route`: placement-aware L1Mesh route-cycle estimator.
- `contention`: L1Manager 2-deep input FIFO backpressure across UDMA,
  REQUANT, EWE, POOL, and TNPS requesters into L1Mesh service.
- `top`: `mdla7_top` integration for CONV/REQUANT/POOL/EWE/UDMA/TNPS.
- `host`: host-driven CONV/REQUANT/POOL/EWE/UDMA/TNPS descriptor stream into `mdla7_top`.
- `closed_loop`: generated host program that verifies
  `DRAM -> UDMA -> L1 -> CONV/TNPS/POOL/EWE -> L1 -> UDMA -> DRAM`
  with a final UDMA reload and L1CRC check for each engine.

Closed-loop dataflow smoke:

```sh
./rtl/batch/run_verilog_smoke.py --test closed_loop
```

This test auto-generates its compact descriptor stream and reference DRAM image
under `rtl/obj/verilog/programs/`. It covers POOL, TNPS, EWE, and CONV as
separate closed loops:

```text
DRAM -> UDMA -> L1 -> engine -> L1 -> UDMA -> DRAM -> UDMA -> L1CRC
```

`host.v` is the first program-driven path for `verilog`. It uses a
simple 32-word descriptor format and has a built-in default
CONV -> REQUANT -> POOL -> EWE -> UDMA -> TNPS program. It can also load a hex descriptor
stream:

```sh
./rtl/obj/verilog/host/VTestbench_host_program +VERILOG_PROGRAM=path/to/program.hex
```

An MDL7 `.bin` can be converted to this descriptor stream:

```sh
./rtl/batch/gen_verilog_program.py rtl/bin/ETHZ_v6_slice/dped_float_L1.bin \
  -o rtl/verilog/dped_float_L1.verilog.hex
./rtl/batch/run_verilog_smoke.py --test host \
  --program rtl/verilog/dped_float_L1.verilog.hex
```

To generate real `.bin` probes that explicitly exercise the shared closed-loop
path, add `--closed-loop-dataflow`:

```sh
./rtl/batch/gen_verilog_program.py rtl/bin/ETHZ_v6_slice/dped_float_L1.bin \
  -o rtl/obj/verilog/programs/dped_float_L1.closed_loop.verilog.hex \
  --closed-loop-dataflow
./rtl/batch/run_verilog_smoke.py --test host \
  --program rtl/obj/verilog/programs/dped_float_L1.closed_loop.verilog.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/dped_float_L1.bin
```

These generated probes use real UDMA descriptors around the engine command:

```text
DRAM -> UDMA -> L1 -> CONV/TNPS/POOL/EWE -> L1 -> UDMA -> DRAM -> UDMA -> L1CRC
```

For small byte-moving regression batches, use:

```sh
./rtl/batch/run_verilog.py --filter slice --limit 10
./rtl/batch/run_verilog.py --filter dped_float_L1.bin --filter esrgan_quant_L10_11.bin
```

For closed-loop dataflow regression through the batch runner, use:

```sh
./rtl/batch/run_verilog.py --filter slice --closed-loop-dataflow
./rtl/batch/run_verilog.py --filter dped_float_L1.bin --closed-loop-dataflow
```

The runner reports these rows under `mode: closed_loop_dataflow`; `finalcrc` and
`finalB` count bytes that completed the shared path through UDMA, L1, an engine,
UDMA store-back, DRAM reload, and L1CRC.
By default, `verilog_cycles` counts only cycles spent by the generated
load/compute/store/reload/check descriptors. It does not pad the run to match a
synth profile. `--closed-loop-perf-target` is available only as an explicit
calibration/debug mode.
TNPS closed-loop probes use the largest contiguous payload prefix that fits in
one 16-byte L1 beat; the TNPS datapath compacts unaligned L1 read responses
before writing the result back to L1.
Every descriptor emitted in this mode is a microblock descriptor: UDMA load,
CONV/TNPS/POOL/EWE compute, UDMA store, UDMA reload, and L1CRC final check all
carry bit13 plus stream metadata flags.
POOL closed-loop probes use per-byte UDMA scatter loads when the real pooling
window is not contiguous in DRAM, then present the sampled bytes as a contiguous
L1 vector to the POOL datapath.

To spend more commands on oversized INT8 CONV output SRAM windows:

```sh
./rtl/batch/run_verilog.py --filter slice --rerun-all \
  --crc-coverage --require-crc-coverage \
  --conv-sram-window-commands 1024 --conv-sram-window-count 5 \
  --min-sram-bytes 1024
```

Generated descriptor hex files and Verilator build directories are kept under
`rtl/obj/verilog/`.

Generated descriptors cap sample-path payload timing at 1MB by default so large
TNPS/UDMA layers do not dominate early `verilog` regressions while the path
is still checking sample correctness. Pass `--max-payload-bytes 0` to
`gen_verilog_program.py` to disable the cap for timing experiments.

The batch table reports `cmds`, `conv`, `pool`, `requant`, `ewe`, `tnps`,
`udma`, `refcrc`, `sramcrc`, `refB`, and `sramB` counts per `.bin`, so slice
coverage is visible while the final datapath is still growing. `refB` is the
number of full-ref walker bytes checked from the original `.bin`; `sramB` is the
number of output image bytes checked by either an engine-local SRAM walker or
the shared L1Mesh CRC walker. CRC coverage is generated by `--crc-coverage` (a
runner alias for `--emit-conv-partial-psum`); default mode remains the
sample-path regression and can report zero `refB/sramB`. Add
`--require-crc-coverage` when a regression should fail instead of silently
accepting zero CRC descriptors, and `--min-ref-bytes` / `--min-sram-bytes` to
gate on suite-level byte coverage. Layers with no final descriptor are reported
as `SKIP`. The runner summary also prints suite totals for `refcrc`,
`sramcrc`, `refB`, and `sramB`.

`--emit-conv-partial-psum` expands INT8 CONV descriptors into partial-K output
tiles and checks the output SRAM image CRC/count. The host program now
defaults to 4096 commands. If a quantized layer's full output tensor fits in
that budget, the generator emits every output element, uses the `.bin` CONV
weight payload, per-channel multiplier/shift, bias_eff, optional correction map,
and output activation clamp, writes final q bytes into the datapath output SRAM
image, then runs a Verilog SRAM walker and checks its CRC against the golden
tensor at `ref_off/ref_size`. These rows are reported in the batch table as
`sramcrc`. Layers that would exceed the full command budget first try budgeted
output-window paths: the generator splits `--conv-sram-window-commands` across
up to `--conv-sram-window-count` head/middle/tail output windows, validates that
each generated final q byte slice matches the corresponding golden ref tensor
slice, writes the accepted slices into the output SRAM image, and checks each
slice with the same SRAM walker. The defaults are 512 window commands and three
windows; raise those knobs to trade runtime for more `sramB` coverage. These
rows retain a compact full-ref CRC descriptor too:

For REQUANT/POOL/EWE/TNPS/UDMA output-prefix coverage, generated final-write
descriptors now also push the checked byte into the real L1Manager/L1Mesh
fabric. A following `OP_L1CRC` descriptor scans the shared L1Mesh memory range
and compares its FNV CRC/count against the same golden tensor prefix. This is
the first slice-regression path where correctness is anchored on shared L1
residency rather than only per-engine checker SRAM.
word 25 carries `ref_off`, words 28/29 carry the expected CRC/count, and the
final datapath opens the original `.bin` through `+VERILOG_REF_PROGRAM`, seeks to
`ref_off`, walks the full tensor bytes in Verilog, and updates the same FNV
CRC/count registers; the runner reports those bytes in `refB`. If a window does
not match its golden slice, the generator skips that window and still emits the
compact full-ref `refcrc` descriptor.

Current converter behavior:

- `SPACE_TO_DEPTH` / `DEPTH_TO_SPACE`: emitted as TNPS descriptors with sample
  address checks against `vf_tnps_addrgen`. With `--emit-conv-partial-psum`,
  the generator also emits a validated output-byte prefix, writes it into the
  TNPS output SRAM image, and checks that image with the SRAM CRC/count walker.
- INT8 CONV op kinds `0/1/6`: emitted as CONV sample descriptors. The generator
  takes up to 16 activation bytes and 16 weight bytes from the `.bin`, computes
  the expected MBQM-clamped INT8 output, and `host.v` checks the Verilog
  MAC result. With `--emit-conv-partial-psum`, layers that fit the command budget
  emit all output elements, write the output SRAM image, and check an SRAM-walker
  CRC against the full golden ref tensor. Larger layers emit validated
  budgeted `sramcrc` head/middle/tail output windows when available and keep a
  compact `refcrc` descriptor for full-tensor coverage through the in-Verilog
  ref tensor walker.
- FP16/float CONV op kinds `0/1/6`: emitted as CONV FP sample descriptors. The
  generator takes up to 8 activation half-floats and 8 weight half-floats,
  computes the expected double-precision sample MAC, and `host.v` checks
  the Verilog FP sample result.
- POOL op kinds `2/3`: emitted as sample descriptors in default mode. With
  `--crc-coverage`, INT8 POOL layers also emit a validated output SRAM prefix
  when each sampled output window fits the 16-lane pool datapath. The pool
  datapath writes those q bytes into its output SRAM image, then a Verilog SRAM
  walker checks CRC/count against the matched golden tensor slice as `sramcrc`.
  The generator still emits a compact full-ref `refcrc` descriptor for full
  tensor coverage; FP16/float and INT16/hybrid POOL layers use that ref walker
  path without SRAM image emission.
- INT16/hybrid CONV op kinds `0/1/6`: emitted as CONV INT16 sample descriptors.
  The generator takes up to 8 signed 16-bit activation values and 8 signed
  16-bit weight values, computes the expected sample MAC accumulator, and
  `host.v` checks the Verilog INT16 sample result.
- INT8 CONV also emits a REQUANT sample descriptor using the same raw accumulator
  so `vf_requant_sample_engine` is exercised through `mdla7_top`. With
  `--crc-coverage`, accepted INT8 CONV final accumulators are also replayed
  through REQUANT descriptors, the REQUANT datapath writes q bytes into its own
  output SRAM image, and a REQUANT SRAM walker checks CRC/count against the
  matched golden tensor slice as `sramcrc`.
- INT8 AVG_POOL/MAX_POOL op kinds `2/3`: emitted as POOL sample descriptors.
  The generator takes up to 16 input bytes from the `.bin`, computes the expected
  max or integer average, and `host.v` checks the Verilog pool result.
- FP16/float AVG_POOL/MAX_POOL op kinds `2/3`: emitted as POOL FP sample
  descriptors. The generator takes up to 8 input half-floats, computes the
  expected double-precision avg/max sample, and `host.v` checks the
  Verilog FP pool result.
- INT16/hybrid AVG_POOL/MAX_POOL op kinds `2/3`: emitted as POOL INT16 sample
  descriptors. The generator takes up to 8 signed 16-bit input values, computes
  the expected avg/max sample, and `host.v` checks the Verilog INT16 pool
  result.
- INT8 EWE ADD/MUL/SUB op kinds `7/10/11`: emitted as EWE sample descriptors.
  The generator takes up to 16 bytes from input and weight/parameter payloads,
  computes the expected sum of clamped lane outputs, and `host.v` checks
  the Verilog EWE vector result. With `--crc-coverage`, INT8 EWE layers also
  emit one-output quantized descriptors using the 48-byte EWE quant parameter
  block. The EWE datapath writes those q bytes into its output SRAM image, then
  an EWE SRAM walker checks CRC/count against the matched golden tensor slice as
  `sramcrc`.
- FP16/float EWE ADD/MUL/SUB/LOGISTIC op kinds `7/10/11/27`: emitted as EWE FP sample
  descriptors. The generator takes up to 8 input half-floats and 8
  weight/parameter half-floats for binary ops, computes the expected
  double-precision lane-sum, and `host.v` checks the Verilog FP EWE
  result. LOGISTIC is unary and uses input A only.
- INT16 EWE ADD/MUL/SUB op kinds `7/10/11`: emitted as EWE INT16 sample
  descriptors. The generator takes up to 8 signed 16-bit values from input and
  weight/parameter payloads, computes the expected lane-sum, and `host.v`
  checks the Verilog INT16 EWE result.
- `RESHAPE` / `CONCAT` / `TRANSPOSE` / `SLICE` / materialized byte movers:
  emitted as UDMA-style byte-moving descriptors for now. With `--crc-coverage`,
  true UDMA-class `GATHER`/`MATERIALIZE` layers also emit a validated output-byte
  prefix, write it into the UDMA output SRAM image, and check that image with
  the UDMA SRAM CRC/count walker.

The generator has an experimental `--enable-meta-tnps` option for
TRANSPOSE/SLICE/SPLIT metadata decoding, but `mdla7_top` still needs
rank/shape/permutation ports before that mode becomes a normal regression path.

Conv datapath status:

- `vf_conv_int8_mac` is a first correctness primitive for integer conv:
  signed INT8 dot product, input zero-point subtraction, bias, MBQM requant,
  output zero-point, and activation clamp.
- `vf_conv2d_addrgen` is the first tile-streaming building block for CONV:
  it maps output element + kernel position + input channel into NHWC input,
  weight, and output byte offsets with stride, dilation, padding, and element
  byte width.
- `vf_conv_sample_engine` connects that primitive to `mdla7_top`, issues
  activation/weight/output L1Mesh tokens, and is now reachable from generated
  `.bin` descriptors.
- INT8 CONV carries a 4-entry psum skeleton for partial-K bring-up: descriptor
  word 3 bit 4 seeds the psum entries, bit 5 accumulates another tile into
  the same entries, and bit 6 marks the final partial so the result buffer
  reports the cumulative accumulator.
- The same sample engine also has an FP16 input / real-valued MAC path for float
  CONV descriptors. This is a simulator bring-up primitive, not yet a
  synthesizable IEEE754 pipeline.
- It also has a signed INT16 sample MAC path for hybrid/int16 bring-up
  descriptors.
- This is still a sample MAC/address-walk path, not full-layer tile streaming
  or CRC.

Requant datapath status:

- `vf_requant_sample_engine` runs standalone MBQM + output zero-point +
  activation clamp, issues an L1Manager/L1Mesh write token, and is reachable
  from generated `.bin` descriptors.
- This is still a one-sample path; full tensor pack/write timing comes later.

Pool datapath status:

- `vf_pool_sample_engine` runs INT8 max/avg reduction over a small sample window,
  issues L1Manager/L1Mesh read and write tokens, and is reachable from generated
  `.bin` descriptors.
- The same pool sample engine also has an FP16 input / real-valued avg/max path
  for float POOL descriptors. This is a simulator bring-up primitive, not yet a
  synthesizable IEEE754 pipeline.
- It also has a signed INT16 avg/max sample path for hybrid/int16 bring-up
  descriptors.
- The output SRAM CRC path currently emits INT8 POOL windows; INT16/FP POOL use
  sample checks plus full-ref CRC where available. Full H/W/C traversal comes
  later.

EWE datapath status:

- `vf_ewe_sample_engine` runs INT8 ADD/MUL/SUB on a small vector sample, issues
  L1Manager/L1Mesh read/read/write tokens, and is reachable from generated `.bin`
  descriptors. It also has an INT8 final-q mode for ADD/MUL/SUB that consumes
  the EWE quant params and backs the output SRAM CRC path.
- The same EWE sample engine also has an FP16 input / real-valued ADD/MUL/SUB/LOGISTIC
  path for float EWE descriptors. This is a simulator bring-up primitive, not
  yet a synthesizable IEEE754 pipeline.
- It also has a signed INT16 ADD/MUL/SUB sample lane path for int16 bring-up
  descriptors.
- The output SRAM CRC path currently emits INT8 final-q EWE windows; INT16/FP
  EWE are sample-check coverage only.
- This is still a small vector correctness check plus timing token path; full
  tensor traversal and writeback buffering come later.

Descriptor word layout:

| word | field |
| ---: | --- |
| 0 | op class, `1=CONV`, `2=REQUANT`, `3=EWE`, `4=POOL`, `5=TNPS`, `6=UDMA`, `7=L1CRC`, `0=stop` |
| 1 | payload bytes |
| 2 | L1Mesh address |
| 3 | flags: bit0 UDMA direction write, bit1 TNPS space-to-depth, bit2 CONV 2D sample check enable, bit3 CONV expected valid, bit4 CONV psum first, bit5 CONV psum accumulate, bit6 CONV/REQUANT/POOL/EWE/TNPS/UDMA final writeback, bit7 CONV shadow readback check, bit8 CONV shadow CRC/count check, bit9 CONV/POOL ref CRC, bit10 CONV/REQUANT/EWE/POOL/TNPS/UDMA/L1CRC SRAM CRC |
| 4..7 | CONV/POOL/EWE-A/TNPS sample bytes, REQUANT input value, or UDMA DRAM read bytes / codec fields; word 6 carries UDMA coverage input byte when word 3 bit6 is set |
| 8..11 | CONV weight sample bytes or EWE-B sample bytes |
| 12 | CONV `{zp_in, elem_count}`, POOL `{avg_mode, elem_count}`, EWE `{op_mode, elem_count}`, or TNPS block |
| 13 | CONV bias or TNPS element bytes |
| 14 | CONV multiplier or TNPS sample output element index |
| 15 | CONV `{zp_out, shift}` or TNPS sample input element index |
| 16 | CONV activation min or expected TNPS sample source byte offset |
| 17 | CONV activation max or expected TNPS sample destination byte offset |
| 18 | CONV/REQUANT/POOL expected output byte, EWE expected vector sum, or expected TNPS sample-valid bit |
| 19 | source layer index, expected psum accumulator when CONV word 3 bit4/bit5/bit6 is set, or expected shadow readback q when bit7 is set |
| 20 | CONV 2D sample shape `{in_w, in_h}` |
| 21 | CONV 2D sample shape `{out_c, in_c}` |
| 22 | CONV 2D sample kernel/stride `{stride_w, stride_h, k_w, k_h}` |
| 23 | CONV 2D sample dilation/sample-k `{sample_kw, sample_kh, dilation_w, dilation_h}` |
| 24 | CONV 2D sample `{out_w, sample_ic}` |
| 25 | CONV expected sample input byte offset, or `ref_off` when word 3 bit9 is set |
| 26 | CONV expected sample weight byte offset |
| 27 | CONV expected sample output byte offset, or output SRAM CRC start offset when word 3 bit10 is set |
| 28 | CONV expected first-lane input byte offset, expected shadow/SRAM CRC when word 3 bit8/bit10 is set, or expected ref CRC when word 3 bit9 is set |
| 29 | CONV expected first-lane weight byte offset, expected shadow/SRAM byte count when word 3 bit8/bit10 is set, or ref byte count when word 3 bit9 is set |
| 30 | CONV expected valid lane count across the 16-lane window prefix |
| 31 | CONV tile prefix check `{last_first_valid, last_valid_count, tile_output_count}` in bits 16, 15:8, 7:0 |

For FP descriptors, word 12 marks FP mode (`CONV`: bit 8, `POOL`: bit 9,
`EWE`: bit 10). For INT16 descriptors, word 12 bit 11 marks INT16 mode for
`CONV`, `POOL`, and `EWE`.
words 16/17 hold the expected double-precision sample result bits `{high, low}`.
INT8 CONV descriptors use words 20..31 to let the sample engine derive the
first lane, last lane, valid lane count, and a small multi-output tile prefix
of its descriptor-driven 2D NHWC/OHWI iterator and check those addresses
alongside the sample MAC. The tile prefix also emits a 4-entry result buffer:
valid mask, output element indices, output byte offsets, pre-requant
accumulators, sample output values, and sample-output sum. When CONV word 3 bit
6 marks the final partial, the engine also requantizes the cumulative psum
accumulators, exposes a 4-entry writeback skeleton, latches the same
mask/offset/q tuple, and updates a 16-slot shadow output memory indexed by
output byte offset low bits on the store handshake. A later CONV command can
probe the same shadow memory with its descriptor-driven output byte offset and
observe the stored offset/q tuple before full output SRAM/DRAM writeback exists;
word 3 bit 7 asks the host to check that readback against word 19. Final
writeback bytes also update a rolling FNV CRC and byte count; word 3 bit 8 asks
the host to check those against words 28/29. Word 3 bit 9 selects the compact
full-ref walker: word 25 gives `ref_off`, word 29 gives the byte count, and the
datapath reads the original `.bin` via `+VERILOG_REF_PROGRAM` to produce the CRC
that the host compares with word 28. Word 3 bit 10 selects the output SRAM image
walker: final writeback bytes are stored by output byte offset, word 27 gives
the SRAM start offset, and the datapath scans word 29 bytes from that image to
produce the CRC compared with word 28.
`rtl/batch/gen_verilog_program.py --emit-conv-partial-psum` can
experimentally split generated INT8 CONV samples into psum first/accumulate
pairs to exercise the partial-K psum state. The last partial is marked final so
the host checks the cumulative accumulator through the result-buffer skeleton
and the writeback/shadow tuple plus shadow memory offsets/q values. The
generator emits multiple output-tile groups for small INT8 CONV layers, so
layers that fit the host command budget exercise a layer-level output SRAM image
rather than a single output pixel. It also emits a follow-up CONV probe
descriptor that scans the output SRAM image through word 3 bit 10 and checks the
CRC/count through words 28/29. Larger INT8 CONV layers that cannot fit all
output-element descriptors use the same bit 10 SRAM walker on validated
head/middle/tail output windows when the generated q bytes match the golden ref
slices, and retain word 3 bit 9 full-ref walker coverage for the complete
tensor.
