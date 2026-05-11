# MDLA7 Verilog Final Datapath

This directory is reserved for the full Verilog datapath simulator path.

Simulator naming:

- `fast`: analytical SystemC model.
- `synth`: SystemC synth timing model.
- `verilog_ctrl`: current Verilator control/timing shell under `rtl/synth`.
- `verilog_final`: Verilator path being built for true Verilog datapath.

Scope for this path:

- Replace DPI-C golden datapath CRC generation with real Verilog datapath blocks.
- Add L1Mesh contention tests that intentionally collide multiple requesters on
  the same tile/bank and check backpressure.
- Add placement-aware route timing inputs so L1Mesh latency reflects physical
  tile/bank placement instead of only simplified address-derived hops.

The existing `rtl/synth` tree remains the `verilog_ctrl` regression target.
Do not add new full datapath logic there except for compatibility fixes needed
to keep the control-path regression running.

Smoke tests:

```sh
./rtl/batch/run_verilog_final_smoke.py
```

Current smoke coverage:

- `conv`: true Verilog INT8 MAC + bias + MBQM + clamp primitive.
- `tnps`: true Verilog TNPS SPACE_TO_DEPTH / DEPTH_TO_SPACE address mapping.
- `route`: placement-aware L1Mesh route-cycle estimator.
- `contention`: L1Manager 2-deep input FIFO backpressure into L1Mesh service.
- `top`: `mdla7_top_final` integration for UDMA/TNPS byte movers.
- `host`: host-driven CONV/REQUANT/POOL/EWE/UDMA/TNPS descriptor stream into `mdla7_top_final`.

`host_final.v` is the first program-driven path for `verilog_final`. It uses a
simple 20-word descriptor format and has a built-in default
CONV -> REQUANT -> POOL -> EWE -> UDMA -> TNPS program. It can also load a hex descriptor
stream:

```sh
./rtl/obj/verilog_final/host/VTestbench_host_program +FINAL_PROGRAM=path/to/program.hex
```

An MDL7 `.bin` can be converted to this descriptor stream:

```sh
./rtl/batch/gen_verilog_final_program.py rtl/bin/ETHZ_v6_slice/dped_float_L1.bin \
  -o rtl/verilog_final/dped_float_L1.final.hex
./rtl/batch/run_verilog_final_smoke.py --test host \
  --program rtl/verilog_final/dped_float_L1.final.hex
```

For small byte-moving regression batches, use:

```sh
./rtl/batch/run_verilog_final.py --filter slice --limit 10
./rtl/batch/run_verilog_final.py --filter dped_float_L1.bin --filter esrgan_quant_L10_11.bin
```

Generated descriptor hex files and Verilator build directories are kept under
`rtl/obj/verilog_final/`.

Generated descriptors cap sample-path payload timing at 1MB by default so large
TNPS/UDMA layers do not dominate early `verilog_final` regressions while the path
is still checking sample correctness. Pass `--max-payload-bytes 0` to
`gen_verilog_final_program.py` to disable the cap for timing experiments.

The batch table reports `cmds`, `conv`, `pool`, `requant`, `ewe`, `tnps`, and
`udma` counts per `.bin`, so slice coverage is visible while the final datapath
is still growing. Layers with no final descriptor are reported as `SKIP`.

Current converter behavior:

- `SPACE_TO_DEPTH` / `DEPTH_TO_SPACE`: emitted as TNPS descriptors with sample
  address checks against `vf_tnps_addrgen`.
- INT8 CONV op kinds `0/1/6`: emitted as CONV sample descriptors. The generator
  takes up to 16 activation bytes and 16 weight bytes from the `.bin`, computes
  the expected MBQM-clamped INT8 output, and `host_final.v` checks the Verilog
  MAC result.
- INT8 CONV also emits a REQUANT sample descriptor using the same raw accumulator
  so `vf_requant_sample_engine` is exercised through `mdla7_top_final`.
- INT8 AVG_POOL/MAX_POOL op kinds `2/3`: emitted as POOL sample descriptors.
  The generator takes up to 16 input bytes from the `.bin`, computes the expected
  max or integer average, and `host_final.v` checks the Verilog pool result.
- INT8 EWE ADD/MUL/SUB op kinds `7/10/11`: emitted as EWE sample descriptors.
  The generator takes up to 16 bytes from input and weight/parameter payloads,
  computes the expected clamped first-lane output, and `host_final.v` checks the
  Verilog EWE result.
- `RESHAPE` / `CONCAT` / `TRANSPOSE` / `SLICE` / materialized byte movers:
  emitted as UDMA-style byte-moving descriptors for now.

The generator has an experimental `--enable-meta-tnps` option for
TRANSPOSE/SLICE/SPLIT metadata decoding, but `mdla7_top_final` still needs
rank/shape/permutation ports before that mode becomes a normal regression path.

Conv datapath status:

- `vf_conv_int8_mac` is a first correctness primitive for integer conv:
  signed INT8 dot product, input zero-point subtraction, bias, MBQM requant,
  output zero-point, and activation clamp.
- `vf_conv_sample_engine` connects that primitive to `mdla7_top_final`, issues
  activation/weight/output L1Mesh tokens, and is now reachable from generated
  `.bin` descriptors.
- This is still a sample MAC path, not full-layer tile streaming or CRC.

Requant datapath status:

- `vf_requant_sample_engine` runs standalone MBQM + output zero-point +
  activation clamp, issues an L1Manager/L1Mesh write token, and is reachable
  from generated `.bin` descriptors.
- This is still a one-sample path; full tensor pack/write timing comes later.

Pool datapath status:

- `vf_pool_sample_engine` runs INT8 max/avg reduction over a small sample window,
  issues L1Manager/L1Mesh read and write tokens, and is reachable from generated
  `.bin` descriptors.
- This is still a sample-window path; full H/W/C window traversal comes later.

EWE datapath status:

- `vf_ewe_sample_engine` runs INT8 ADD/MUL/SUB on a small vector sample, issues
  L1Manager/L1Mesh read/read/write tokens, and is reachable from generated `.bin`
  descriptors.
- This is still a first-lane correctness check plus timing token path; full
  tensor traversal and vector writeback come later.

Descriptor word layout:

| word | field |
| ---: | --- |
| 0 | op class, `1=CONV`, `2=REQUANT`, `3=EWE`, `4=POOL`, `5=TNPS`, `6=UDMA`, `0=stop` |
| 1 | payload bytes |
| 2 | L1Mesh address |
| 3 | flags: bit0 UDMA direction write, bit1 TNPS space-to-depth |
| 4..7 | CONV/POOL/EWE-A sample bytes, REQUANT input value, or UDMA DRAM read bytes / codec fields |
| 8..11 | CONV weight sample bytes or EWE-B sample bytes |
| 12 | CONV `{zp_in, elem_count}`, POOL `{avg_mode, elem_count}`, EWE `{op_mode, elem_count}`, or TNPS block |
| 13 | CONV bias or TNPS element bytes |
| 14 | CONV multiplier or TNPS sample output element index |
| 15 | CONV `{zp_out, shift}` or TNPS sample input element index |
| 16 | CONV activation min or expected TNPS sample source byte offset |
| 17 | CONV activation max or expected TNPS sample destination byte offset |
| 18 | CONV/REQUANT/POOL/EWE expected output byte or expected TNPS sample-valid bit |
| 19 | source layer index |
