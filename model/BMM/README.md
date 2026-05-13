# BMM Unit Patterns

Tiny attention-style patterns for L1Mesh validation.

Shape:

- Q: `[1, 2, 8, 8]`
- K: `[1, 2, 8, 8]`
- V: `[1, 2, 8, 8]`
- Output: `[1, 2, 8, 8]`

Graph intent:

1. `BMM(Q, K^T)`
2. scale
3. `Softmax(axis=-1)`
4. `BMM(Softmax, V)`

Generated files:

- `bmm_softmax_bmm_fp32.tflite`: BATCH_MATMUL, MUL, SOFTMAX, BATCH_MATMUL
- `bmm_softmax_bmm_int8.tflite`: BATCH_MATMUL, MUL, SOFTMAX, BATCH_MATMUL
- `bmm_softmax_bmm_sam_quant_L22_L61.tflite`: real SAM attention slice lowered into supported MDLA7 ops

Current coverage:

- `compile_model.py` recognizes `BATCH_MATMUL` and emits it as explicit
  `matrlz` fallback, not as a skipped op.
- `./batch/run_systemc.py --filter bmm --fast-only --rerun-all --no-html`
  runs the corpus through fast SystemC.
- `./batch/run_systemc.py --filter bmm --cx --fast-only --rerun-all --no-html`
  runs the corpus through CX SystemC.
- `./batch/run_verilog.py --filter bmm --rerun-all --timeout 180` runs
  closed-loop Verilog with full final-output coverage required.

`matrlz` is supported-but-not-native coverage: it preserves graph correctness
and final/layer checks while making clear that native BMM RTL datapath work is
still separate. Use `systemc/scripts/audit_unsupported_ops.py --strict-native`
when materialized fallback should fail the audit.

The SAM candidate is kept as the current runnable L1Mesh stress pattern:

- QK score path: `trnps -> trnps -> mul`, shape `4,49,49 -> 4,49,49`
- Attention normalization: `softmax`, shape `4,49,49 -> 4,49,49`
- Value/output path: `trnps -> trnps -> matrlz -> fc`, shape `4900,1,128 -> 4900,1,128`
