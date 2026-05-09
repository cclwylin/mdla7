# MDLA7 System Report PPT Outline

Generated PPT: `reports/MDLA7_System_Report.pptx`

## Slide 1: Title
- MDLA7 SystemC Model and Performance Optimization
- Architecture, throughput, tuning history, regression status, and synthesis gaps.

## Slide 2: Executive Summary
- SystemC model is functional and profileable.
- ETHZ_V6 fast 51/51 OK; Hotspot 11/11 OK across fast/conflict/mesh.
- Key tuning shifted dataflow from DRAM round trips to on-chip microblock handoff.

## Slide 3: Given Specification Snapshot
- 1.9 GHz, 62.3 TOPS INT8 baseline, 3 MB L1Mesh, LPDDR5X-10667.
- Descriptor ring + MMIO doorbell + wait/signal dependency model.

## Slide 4: Module Dataflow
- UDMA moves DRAM/L1 traffic; TNPS handles layout movement; CONV directly reads L1Mesh; non-CONV engines share L1_Manager.
- Flow IDs expose fused producer-consumer groups.

## Slide 5: Module Functions and Throughput
- CONV: Conv / FC / DWConv MAC; INT8x8: 16,384 MAC/cyc; 62.3 TOPS @1.9GHz
- Requant: INT32/FP post-processing, clamp, scale/shift; 16 CONV-chain lanes; Payload W 128 B/cyc
- EWE: Add/Mul/Sub/activation/math/softmax pieces; 64 elem/cyc; R 256 B/cyc, W 128 B/cyc
- POOL: Max/Avg/Global pooling; R 256 B/cyc, W 128 B/cyc
- TNPS: Transpose/Slice/Space-depth/Layout materialize; 128 B/cyc read + 128 B/cyc write
- UDMA: DRAM↔L1 copy, prefetch, store, ACT codec; LPDDR model peak ≈85.3 GB/s external
- L1Mesh: 3 MB banked scratchpad / routing fabric; 16 banks x 16B per SRAM cycle backend

## Slide 6: SystemC and Profiling Deliverables
- Compiler/runtime, SystemC model, profile reports, Gantt waveform, flow IDs, TNPS lane.

## Slide 7: Performance Tuning Journey
- 1. Baseline profiling: Per-layer execution; large intermediate DRAM writes visible in profile.
- 2. Flow reporting: Added flow ID so fused chains are visible and unfused layers remain layer=flow.
- 3. Tiling and microblocks: Conv/FC/EWE/POOL handoff through L1; reduce intermediate DRAM R/W.
- 4. Overlap scheduling: Allow different engines to run concurrently when dependencies permit.
- 5. Layout offload: Moved transpose/slice/concat/space-depth style ops to TNPS lane.
- 6. PAD→CONV fold: Eliminated PAD materialization before 3x3 CONV; sd_decoder slice improved 18.1%.

## Slide 8: Case Study: sd_decoder_quant_L152_L191
- Before 0.456 ms; after 0.373 ms; L16 PAD now 0 cycles / 0 DRAM.
- Optimization: fold PAD into following CONV instead of attempting overlap with its producer.

## Slide 9: Regression Status
- ETHZ_V6 fast-only 51/51 OK, 0 function FAIL.
- Hotspot 11/11 fast/conflict/mesh OK.
- Conflict/mesh are not yet the full ETHZ_V6 gate.

## Slide 10: Gap to Synthesizable RTL
- Translate TLM model into RTL, finalize interfaces, calibrate timing, close verification.
- Next tuning gaps: multi-source TNPS concat, true kernels for layout ops, further intermediate write removal.
