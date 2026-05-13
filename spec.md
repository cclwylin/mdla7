# MDLA7 Module Spec: Current RTL + Target Notes

Date: 2026-05-13

This document records the current Verilog implementation plus explicit target
architecture notes where the project has already chosen the final direction.
The active hardware tree is `rtl/verilog`; do not recreate a `verilog_ctrl` /
`verilog_final` split.

## System-Level Dataflow

The intended closed-loop path is:

```text
Testbench DRAM(.bin)
  -> Host descriptor stream
  -> mdla7_top command dispatch
  -> UDMA DRAM-to-L1 load
  -> L1Manager arbitration
  -> L1Mesh SRAM
  -> CONV / REQUANT / POOL / EWE / TNPS engine/postprocess
  -> L1Manager / L1Mesh writeback
  -> UDMA L1-to-DRAM store
  -> UDMA DRAM-to-L1 reload
  -> L1CRC / checker
```

Current coverage is real byte movement through Host / UDMA / L1Manager /
L1Mesh / engine control, but most engine payloads are still sample/tilelet
sized. Cycle correlation should wait until compact full-tile traversal exists.

Target CONV/Requant writeback policy:

- CONV reads ACT and WGT directly from L1Mesh.
- CONV does not have a target Payload W path to L1 for INT32/INT48
  partial sums.
- CONV psum/acc values chain directly into Requant through the dedicated
  CONV -> Requant path.
- Requant owns final quantized output writeback to L1.
- Any current CONV L1 output write is a bring-up/debug/sample path, not the
  target architecture.

Target UDMA/DRAM bus policy:

- UDMA <-> L1Manager target payload bus is 16 x 16B R plus 16 x 16B W.
- UDMA <-> DRAM target bus is 4 x 16B AXI R/W.
- Current Verilog DRAM model still uses the simpler 128-bit request/response
  interface and timing defaults; this is a current RTL model, not the final bus
  target.

## Common Descriptor Format

Descriptors are 32 words, each 32 bits. `host.v` loads them from
`+VERILOG_PROGRAM=<hex>` or from its built-in default program.

Common words:

| Word | Field |
| --- | --- |
| `0[3:0]` | opcode |
| `0[19:4]` | `layer_id` |
| `0[31:20]` | low 12 bits of `microblock_id`; top 4 bits are zeroed |
| `1` | generic `bytes` / workload bytes |
| `2[21:0]` | L1 base address |
| `3` | shared flags |
| `3[23:16]` | `stream_slot` |
| `3[31:24]` | stream metadata flags |
| `25` | reference DRAM offset for ref walkers / store source |
| `27` | output byte offset / L1 output address |
| `28` | expected FNV CRC |
| `29` | expected byte count |

Opcodes:

| Opcode | Name |
| --- | --- |
| `0` | DONE |
| `1` | CONV |
| `2` | REQUANT |
| `3` | EWE |
| `4` | POOL |
| `5` | TNPS |
| `6` | UDMA |
| `7` | L1CRC |

Shared flag bits in word 3:

| Bit | Meaning |
| --- | --- |
| `0` | UDMA direction write, L1 to DRAM |
| `1` | TNPS space-to-depth mode |
| `4` | CONV partial first |
| `5` | CONV partial accumulate |
| `6` | final write mode |
| `9` | engine ref CRC mode |
| `10` | engine/output SRAM CRC mode |
| `11` | read sample/input from L1 |
| `13` | microblock descriptor mode |
| `14` | UDMA ref fill mode |
| `15` | probe descriptor mode |

Stream metadata flags in word 3 bits `[31:24]`:

| Flag | Meaning |
| --- | --- |
| `0x01` | LOAD_A |
| `0x02` | LOAD_B |
| `0x04` | COMPUTE |
| `0x08` | STORE |
| `0x10` | FINAL_TILE |

## `host.v` / `host`

Role:

- Program-driven descriptor sequencer.
- Decodes each 32-word descriptor into `mdla7_top` control inputs.
- Issues one descriptor at a time using `desc_valid` / `desc_ready`.
- Waits for `top_done_valid`, validates selected debug/checker outputs, then
  advances to the next descriptor.
- Tracks `issued_count`, `done_count`, and `measured_cycle_count`.

State machine:

| State | Purpose |
| --- | --- |
| `ST_LOAD` | Load descriptor words into output registers |
| `ST_ISSUE` | Assert `desc_valid` until accepted |
| `ST_WAIT` | Wait for top-level completion and run checks |
| `ST_NEXT` | Advance command index |
| `ST_DONE` | Finish test |

Important behavior:

- `top_done_ready` is tied high.
- Built-in default program exists, but generated programs are the main path.
- For microblock descriptors, host checks that active layer/microblock/stream
  metadata matches the issued descriptor.
- Host counts LOAD / COMPUTE / STORE / FINAL_TILE metadata and checks these
  against `vf_microblock_control` counters.

## `mdla7_top.v` / `mdla7_top`

Role:

- Top-level Verilog integration shell.
- Latches host descriptor fields.
- Starts exactly one block per accepted descriptor.
- Instantiates and connects CONV, REQUANT, POOL, EWE, UDMA, TNPS, L1Manager,
  L1Mesh, route estimator, and microblock control.

Top-level descriptor handshake:

- Input: `desc_valid`.
- Output: `desc_ready`.
- Descriptor is accepted when both are high.
- Completion is signaled through `done_valid`; host drives `done_ready`.

Dispatch:

| Opcode | Block |
| --- | --- |
| CONV | `vf_conv_sample_engine` |
| REQUANT | `vf_requant_sample_engine` |
| EWE | `vf_ewe_sample_engine` |
| POOL | `vf_pool_sample_engine` |
| TNPS | `vf_tnps_engine` |
| UDMA | `vf_udma_engine` |
| L1CRC | L1Mesh debug CRC path |

L1 response routing:

- L1Mesh response carries `resp_source`, `resp_tid`, `resp_read`, and
  `resp_rdata`.
- `mdla7_top` registers the response and routes it back to the corresponding
  engine request source.
- The current design assumes one active descriptor, but L1Manager still models
  multi-source arbitration and source-tagged return data.

## `mdla7_top.v` / `vf_microblock_control`

Role:

- Records active descriptor identity and stream metadata.
- Counts accepted microblock phases.

Counters:

| Counter | Increment condition |
| --- | --- |
| `load_count` | `LOAD_A` or `LOAD_B` flag present |
| `compute_count` | `COMPUTE` flag present |
| `store_count` | `STORE` flag present |
| `final_count` | `FINAL_TILE` flag present |

## `common.v` / `mdla7_synth_phase_engine`

Role:

- Shared latency phase sequencer.
- Consumes packed phase cycle counts and phase IDs.
- Skips zero-cycle phases.
- Supports a `phase_stall` input for backpressure-sensitive phases.

Interface contract:

- `start_ready` is high when idle and no pending `done_valid`.
- `done_valid` stays high until `done_ready`.
- `phase_id` and `remaining_cycles` expose the currently active phase.

## `l1manager.v` / `l1manager`

Role:

- Arbitrates L1 requests from legacy, UDMA, REQUANT, EWE, POOL, and TNPS.
- Presents one selected request to L1Mesh.
- Models request fetch, arbitration, payload, and response timing.

Sources:

| Internal source | Priority | Debug source |
| --- | --- | --- |
| UDMA | 1 | `6` |
| REQUANT | 2 | `2` |
| EWE | 3 | `3` |
| POOL | 4 | `4` |
| TNPS | 5 | `5` |
| LEGACY | 6 | `0` |

Queueing:

- Each source has a two-entry queue.
- Ready deasserts when the second entry for that source is occupied.

Timing phases:

| Phase | Meaning |
| --- | --- |
| `PH_REQ_FETCH` | Capture request |
| `PH_ARB` | Arbitration |
| `PH_L1_PAYLOAD` | L1 payload service |
| `PH_DRAM_READ_DATA` | DRAM read timing model path |
| `PH_DRAM_WRITE_DATA` | DRAM write timing model path |
| `PH_RESP` | Completion response |

## `l1mesh.v` / `l1mesh`

Role:

- L1 SRAM mesh model with 8 edge Mesh4x4 fabrics and 4 storage Mesh4x4 blocks.
- Stores and returns real data for engine and UDMA traffic.
- Exposes a debug FNV CRC scanner for final L1 checks.
- Target hierarchy:
  `8 edge Mesh4x4 fabrics -> 4 storage Mesh4x4 blocks -> 16 Quad SRAM per storage Mesh4x4 -> 4 SRAM Macro ports per Quad`.
- The 8 edge Mesh4x4 fabrics are an injection/route resource. Every 2 edge
  Mesh4x4 map to the same storage Mesh4x4, so storage remains 3 MB:
  `4 * 16 * 4 * 768 * 16B`.
- Total SRAM Macro count is 256:
  `4 Storage Mesh4x4 * 16 Quad SRAM * 4 SRAM Macro`.
- L1Mesh ingress has a two-entry request FIFO. `req_ready` deasserts when the
  second entry is occupied, and queued requests feed the phase sequencer from
  the oldest entry.
- Phase dispatch is gated by selected edge-fabric availability and the selected
  SRAM macro port. RTL tracks 256 SRAM macro busy counters matching
  `4 Storage Mesh4x4 * 16 Quad SRAM * 4 SRAM Macro`.
- Edge injection is lane-level in RTL for the target groups:
  CONV ACT `32R`, CONV WGT `32R`, L1Manager `16R + 16W`, and Requant `8W`.
  A request selects an available lane in its source group; if all lanes are
  busy, injection backpressures the queued request.
- Edge Mesh4x4 fabrics use flit-level `valid/ready`. Each router node has
  4-deep input FIFOs for N/S/W/E/local directions, and downstream FIFO fullness
  stalls upstream flit movement.
- Storage data is held in the same 256 SRAM macro hierarchy, not in a compact
  four-tile shadow store. Response return has a two-entry FIFO so external
  `resp_ready` backpressure does not directly collapse the internal phase pipe.
- Verilog instantiates explicit edge Mesh4x4 router/link fabrics with
  N/S/W/E/local route outputs for the 8 edge fabrics.

Parameters:

| Parameter | Current default |
| --- | --- |
| `ADDR_WIDTH` | `22` |
| `DATA_WIDTH` | `128` |
| `MEM_WORDS` | `196608` |
| `BYTES_PER_CYCLE` | `16` |
| `SYNTH_L1_PIPE_CYCLES` | `3` |
| `EDGE_MESH4X4_COUNT` | `8` |
| `STORAGE_MESH4X4_COUNT` | `4` |
| `QUAD_SRAM_PORTS` | `4` |
| `SRAM_MACRO_WORDS` | `768` |

Address mapping:

- Byte address is converted to 16-byte word address.
- `word_addr[1:0]` selects one of 4 storage Mesh4x4 blocks.
- `word_addr[5:2]` selects one of 16 Quad SRAM nodes in that storage Mesh4x4.
- `word_addr[7:6]` selects one of 4 SRAM macro ports in the Quad.
- `word_addr[8]` selects the edge-injection half, giving 8 edge Mesh4x4
  fabrics as `{word_addr[8], word_addr[1:0]}`.
- `word_addr[31:8]` selects the 768-word SRAM macro depth.
- Writes use byte strobes.
- Reads return a full 128-bit line.

Timing phases:

| Phase | Meaning |
| --- | --- |
| `PH_ADDR_DECODE` | Address decode |
| `PH_L1MESH_SELECT` | Edge Mesh4x4 selection and placement route |
| `PH_MESH4X4_ROUTE` | Route inside the selected Mesh4x4 |
| `PH_QUAD_SRAM_SELECT` | Quad SRAM / macro port selection |
| `PH_SRAM_MACRO` | SRAM macro access |
| `PH_RESP` | Response |

Debug CRC:

- `debug_crc_start` scans `debug_crc_count` bytes from `debug_crc_addr`.
- The scanner consumes up to 16 bytes per cycle and computes FNV-1a style CRC.

## `l1mesh.v` / `mdla7_mesh4x4_edge_fabric`

Role:

- Verilog edge Mesh4x4 structure for L1 route tokens.
- Instantiates 16 `mdla7_mesh4x4_router_node` nodes.
- Each node exposes N/S/W/E/local route outputs and has 4-deep input FIFO
  accounting for all five directions.
- Connects adjacent nodes through explicit horizontal E/W link buses and
  vertical S/N link buses.
- Uses flit `valid/ready`; full downstream input FIFO backpressures upstream
  route movement.
- Uses XY routing toward the selected Quad SRAM node.

## `route.v` / `vf_l1mesh_route_estimator`

Role:

- Placement-aware route cycle estimator.
- Maps source ID and L1 address to edge Mesh4x4 and Quad SRAM coordinates.
- Computes `BASE + edge Mesh4x4 Manhattan hops + local Quad/port hops`.

Source placement:

| Source ID | Module | Coord |
| --- | --- | --- |
| `1` | CONV | `(0,0)` |
| `2` | REQUANT | `(1,0)` |
| `3` | EWE | `(0,1)` |
| `4` | POOL | `(1,1)` |
| `5` | TNPS | `(0,1)` |
| `6` | UDMA | `(1,0)` |

## `udma.v` / `vf_udma_engine`

Role:

- Byte mover between DRAM model and L1Mesh.
- Supports DRAM-to-L1 load, L1-to-DRAM store, literal final writes, ref-fill,
  and UDMA output SRAM CRC modes.

Key inputs:

| Input | Meaning |
| --- | --- |
| `direction_write` | `0`: DRAM to L1, `1`: L1 to DRAM |
| `bytes` | transfer byte count |
| `dram_read_bytes` | optional DRAM timing/read size override |
| `codec_cycles` | optional codec latency |
| `final_write_mode` | final/literal write behavior |
| `sramcrc_mode` | scan UDMA output SRAM image |
| `ref_fill_mode` | fill output SRAM from reference `.bin` |
| `input_byte` | literal byte for simple final-write descriptors |
| `out_byte_offset` | L1/DRAM output offset |
| `ref_off` | DRAM reference/base offset |

Implemented data movement:

- DRAM-to-L1 load issues DRAM read requests and writes returned 16-byte beats
  into L1 with aligned write strobes.
- L1-to-DRAM store reads one L1 response beat and writes captured data/strobes
  into the writable DRAM model.
- Literal write mode can write a descriptor-provided byte into L1.
- SRAM CRC mode checks the internal UDMA output SRAM image.

Timing phases:

| Phase | Meaning |
| --- | --- |
| `PH_CFG_DECODE` | Decode descriptor |
| `PH_L1_PAYLOAD_READ` | Read payload from L1 |
| `PH_CODEC_PIPE` | Optional codec latency |
| `PH_DRAM_CMD` | DRAM command latency |
| `PH_DRAM_WRITE_DATA` | DRAM write payload |
| `PH_DRAM_READ_DATA` | DRAM read payload |
| `PH_L1_PAYLOAD_WRITE` | Write payload to L1 |
| `PH_RETIRE` | Completion |

## `conv.v` / `vf_conv_int8_mac`

Role:

- Combinational INT8 MAC primitive.
- Applies input zero point, bias, multiply-by-quantized-multiplier,
  output zero point, and activation clamp.

Behavior:

- Processes up to 16 INT8 activation/weight lanes.
- `elem_count` limits active lanes.
- Optional DPI backend is enabled with `MDLA7_DPI_DATAPATH` plus runtime
  `+MDLA7_DATAPATH_DPI`.

## `conv.v` / `vf_conv2d_addrgen`

Role:

- NHWC activation / OHWI weight / output byte offset generator.
- Computes one sampled output element and one sampled `kh/kw/ic`.

Inputs:

- Tensor shape: input/output H/W/C.
- Kernel, stride, dilation, padding.
- Element byte width.
- Output element index plus sampled kernel/channel indices.

Outputs:

- `input_byte_offset`.
- `weight_byte_offset`.
- `output_byte_offset`.
- `input_valid` after bounds and padding checks.

## `conv.v` / `vf_conv_sample_engine`

Role:

- Current CONV datapath engine.
- Supports INT8, INT16, and FP16/float sample MAC paths.
- Can read sample vectors from L1.
- Current RTL can write sample/debug output bytes to L1, but target CONV
  writeback is CONV psum/acc -> Requant -> L1.
- Maintains shadow/scoreboard outputs for host-side checks.

Current scope:

- Not yet a full `OH x OW x OC x KH x KW x IC` tile traversal engine.
- Supports sampled windows, partial-psum descriptors, small output tilelets
  up to 4 outputs in the scoreboard path, ref CRC mode, and output SRAM CRC
  mode.

Descriptor words used by host:

| Word | CONV field |
| --- | --- |
| `4..7` | activation sample vector |
| `8..11` | weight sample vector |
| `12[7:0]` | element count |
| `12[8]` | FP mode |
| `12[11]` | INT16 mode |
| `12[31:16]` | input zero point |
| `13` | bias |
| `14` | multiplier |
| `15[7:0]` | shift |
| `15[15:8]` | output zero point |
| `16` | activation min |
| `17` | activation max |
| `18` | expected q byte / final byte |
| `19` | expected accumulator / partial accumulator |
| `20[15:0]` | input H |
| `20[31:16]` | input W |
| `21[15:0]` | input C |
| `21[31:16]` | output C |
| `22[7:0]` | kernel H |
| `22[15:8]` | kernel W |
| `22[23:16]` | stride H |
| `22[31:24]` | stride W |
| `23[7:0]` | dilation H |
| `23[15:8]` | dilation W |
| `23[23:16]` | sample KH |
| `23[31:24]` | sample KW |
| `24[15:0]` | sample IC |
| `24[31:16]` | output W |
| `30[31:16]` | output H, default 1 if zero |
| `31[7:0]` | tile output count, clamped to 1..4 for checks |

## `requant.v` / `vf_requant_sample_engine`

Role:

- Requantizes one signed 32-bit accumulator to INT8.
- Can read the input accumulator from L1 and write one q byte back to L1.
- Can scan its internal output SRAM image for CRC coverage.

Descriptor words used by host:

| Word | REQUANT field |
| --- | --- |
| `4` | input accumulator when not reading from L1 |
| `14` | multiplier |
| `15[7:0]` | shift |
| `16` | activation min |
| `17` | activation max |
| `18[7:0]` | expected output q |
| `27` | output byte offset |
| `28` | expected SRAM CRC |
| `29` | expected SRAM byte count |

Timing states:

| State | Meaning |
| --- | --- |
| `ST_PARAM` | Optional L1 input fetch |
| `ST_PIPE` | MBQM and clamp |
| `ST_STORE` | L1 output write |
| `ST_SRAMCRC` | Internal SRAM CRC scan |
| `ST_DONE` | Completion |

## `pool.v` / `vf_pool_sample_engine`

Role:

- Pooling sample datapath.
- Supports INT8, INT16, and FP16/float max/avg sample reductions.
- Can fetch a sample window from L1, write result bytes to L1, and run ref or
  output SRAM CRC modes.

Current scope:

- Processes up to 16 INT8 lanes or 8 FP16/INT16 lanes per sample.
- Not yet a full output tile/window sweep engine.
- FP16 arithmetic can use optional DPI helper.

Descriptor words used by host:

| Word | POOL field |
| --- | --- |
| `4..7` | sample vector |
| `12[7:0]` | element count |
| `12[8]` | avg mode; otherwise max |
| `12[9]` | FP mode |
| `12[11]` | INT16 mode |
| `25` | ref CRC source offset |
| `27` | output byte offset |
| `28` | expected CRC |
| `29` | expected byte count |

Timing states:

| State | Meaning |
| --- | --- |
| `ST_FETCH` | Optional L1 window fetch |
| `ST_PIPE` | Reduce lanes |
| `ST_STORE` | L1 result write |
| `ST_REFCRC` | Reference `.bin` CRC scan |
| `ST_SRAMCRC` | Internal output SRAM CRC scan |
| `ST_DONE` | Completion |

## `ewe.v` / `vf_ewe_sample_engine`

Role:

- Element-wise sample datapath.
- Supports ADD, MUL, SUB, plus FP LOGISTIC-style unary support in the generator
  path.
- Supports INT8 quantized output mode, INT16 sample mode, FP16/float sample
  mode, optional L1 read for input A, L1 writeback, and output SRAM CRC.

Key modes:

| Field | Meaning |
| --- | --- |
| `op_mode` | operation selector |
| `fp_mode` | FP16/float sample arithmetic |
| `int16_mode` | signed 16-bit sample arithmetic |
| `final_q_mode` | write quantized q result |
| `read_a_from_l1` | fetch A vector from L1 |
| `sramcrc_mode` | scan internal output SRAM image |

Descriptor words used by host:

| Word | EWE field |
| --- | --- |
| `4..7` | A vector |
| `8..11` | B vector |
| `12[7:0]` | element count |
| `12[9:8]` | op mode |
| `12[10]` | FP mode |
| `12[11]` | INT16 mode |
| `13` | zero point A |
| `14` | zero point B |
| `15` | zero point output |
| `16` | multiplier A |
| `17[7:0]` | shift A |
| `20` | multiplier B |
| `21[7:0]` | shift B |
| `22` | multiplier output |
| `23[7:0]` | shift output |
| `24` | left shift |
| `25` | activation min |
| `26` | activation max |
| `27` | output byte offset |
| `28` | expected SRAM CRC |
| `29` | expected SRAM byte count |

## `tnps.v` / `vf_tnps_addrgen`

Role:

- Address generator for SPACE_TO_DEPTH and DEPTH_TO_SPACE.
- Computes source and destination byte offsets for one sampled element.

Validity:

- Checks tensor dimensions, block compatibility, element index range, and mode
  shape constraints.

## `tnps.v` / `vf_tnps_engine`

Role:

- TNPS sample/byte permutation engine.
- Uses `vf_tnps_addrgen`.
- Can read compacted bytes from L1, write permuted bytes to L1, and scan
  internal output SRAM CRC.

Current scope:

- Closed-loop descriptors currently exercise compact byte chunks that fit in
  one 16-byte L1 beat.
- Full tensor/tile sweep is not implemented yet.

Descriptor words used by host:

| Word | TNPS field |
| --- | --- |
| `4[7:0]` | literal input byte / first input vector word |
| `4..7` | input vector |
| `6[15:0]` | input H |
| `7[15:0]` | input W |
| `8[15:0]` | input C |
| `9[15:0]` | output H |
| `10[15:0]` | output W |
| `11[15:0]` | output C |
| `12[15:0]` | block |
| `13[1:0]` | element bytes; zero means 1 byte |
| `14` | sample output element index |
| `15` | sample input element index |
| `27` | output byte offset |
| `28` | expected SRAM CRC |
| `29` | expected SRAM byte count |

Timing phases:

| Phase | Meaning |
| --- | --- |
| `PH_CFG_DECODE` | Decode descriptor |
| `PH_PAYLOAD_READ` | Read source bytes |
| `PH_PERMUTE_PIPE` | Permute/compact bytes |
| `PH_PAYLOAD_WRITE` | Write output bytes |
| `PH_RETIRE` | Completion |

## `Testbench_host_program.v` / `vf_dram_model`

Role:

- Writable DRAM model for host-driven closed-loop tests.
- Loads original `.bin` from `+VERILOG_REF_PROGRAM`.
- Reads first check writable override memory, then fall back to file bytes.
- Writes use `req_wdata` and `req_wstrb`.

Interface:

| Signal | Meaning |
| --- | --- |
| `req_valid` | request active |
| `req_write` | write when high, read when low |
| `req_addr` | byte address |
| `req_bytes` | byte count |
| `req_wdata` | 128-bit write data |
| `req_wstrb` | per-byte strobes |
| `resp_rdata` | 128-bit read response |

## Testbenches

| File | Scope |
| --- | --- |
| `Testbench_conv_datapath.v` | CONV MAC, INT16, FP/sample/address-walk checks |
| `Testbench_requant_datapath.v` | MBQM, clamp, output-zero-point checks |
| `Testbench_pool_datapath.v` | INT8/INT16/FP pool sample checks |
| `Testbench_ewe_datapath.v` | INT8/INT16/FP EWE sample checks |
| `Testbench_tnps_datapath.v` | TNPS address mapping checks |
| `Testbench_route_timing.v` | Route estimator checks |
| `Testbench_l1mesh_contention.v` | L1Manager queue/backpressure contention |
| `Testbench_l1mesh_storage.v` | L1Mesh SRAM macro hierarchy storage/readback and response backpressure |
| `Testbench_top_byte_movers.v` | Top-level dummy DRAM integration smoke |
| `Testbench_host_program.v` | Host program, DRAM model, closed-loop system test |

## Generator / Runner

`batch/gen_verilog_program.py`:

- Converts MDL7 `.bin` images into 32-word Verilog host descriptors.
- Emits closed-loop descriptor groups:

```text
UDMA DRAM->L1 load
engine compute
UDMA L1->DRAM store
UDMA DRAM->L1 reload
L1CRC final check
```

- Current generated closed-loop payloads are sample/tilelet sized.
- Descriptor generation must evolve toward compact tile descriptors and
  hardware-side traversal.

`batch/run_verilog.py`:

- Runs generated programs through the Verilator host test.
- Reports `cmds`, per-engine descriptor counts, CRC coverage, synth cycles,
  verilog cycles, ratio, and wall time.
- `--option dpi` enables optional arithmetic DPI helpers; it does not bypass
  Host / Command / UDMA / L1Manager / L1Mesh control.

## Current Gaps

- CONV full tile traversal is not implemented.
- POOL / EWE / TNPS full output tile sweeps are not implemented.
- Current payload sizes are too small for reliable synth-vs-Verilog cycle
  correlation.
- FP full-output golden packing/rounding is not complete.
- Performance calibration should wait until full/tiled traversal exists.
