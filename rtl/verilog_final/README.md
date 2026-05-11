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
- `host`: host-driven UDMA/TNPS descriptor stream into `mdla7_top_final`.

`host_final.v` is the first program-driven path for `verilog_final`. It uses a
simple 20-word descriptor format and has a built-in default UDMA -> TNPS program.
It can also load a hex descriptor stream:

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

The batch table reports `cmds`, `tnps`, and `udma` counts per `.bin`, so slice
coverage is visible while the final datapath is still growing. Layers with no
byte-moving descriptor are reported as `SKIP`.

Current converter behavior:

- `SPACE_TO_DEPTH` / `DEPTH_TO_SPACE`: emitted as TNPS descriptors with sample
  address checks against `vf_tnps_addrgen`.
- `RESHAPE` / `CONCAT` / `TRANSPOSE` / `SLICE` / materialized byte movers:
  emitted as UDMA-style byte-moving descriptors for now.

The generator has an experimental `--enable-meta-tnps` option for
TRANSPOSE/SLICE/SPLIT metadata decoding, but `mdla7_top_final` still needs
rank/shape/permutation ports before that mode becomes a normal regression path.

Conv datapath status:

- `vf_conv_int8_mac` is a first correctness primitive for integer conv:
  signed INT8 dot product, input zero-point subtraction, bias, MBQM requant,
  output zero-point, and activation clamp.
- The current conv smoke is a direct primitive test. It does not yet read
  activation/weight/params from `.bin` or compute full-layer CRC.

Descriptor word layout:

| word | field |
| ---: | --- |
| 0 | op class, `5=TNPS`, `6=UDMA`, `0=stop` |
| 1 | payload bytes |
| 2 | L1Mesh address |
| 3 | flags: bit0 UDMA direction write, bit1 TNPS space-to-depth |
| 4 | UDMA DRAM read bytes |
| 5 | UDMA codec cycles |
| 6..13 | TNPS in/out shape, block, element bytes |
| 14 | TNPS sample output element index |
| 15 | TNPS sample input element index |
| 16 | expected TNPS sample source byte offset |
| 17 | expected TNPS sample destination byte offset |
| 18 | expected TNPS sample-valid bit |
| 19 | source layer index |
