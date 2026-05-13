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

Note: current `compile_model.py` does not list `BATCH_MATMUL` in supported ops, so it ignores the two BMM ops and only emits the middle `MUL + SOFTMAX` today. These files are the target BMM patterns first; the next step is to add BatchMatMul lowering/execution support.

The SAM candidate is kept as the current runnable L1Mesh stress pattern:

- QK score path: `trnps -> trnps -> mul`, shape `4,49,49 -> 4,49,49`
- Attention normalization: `softmax`, shape `4,49,49 -> 4,49,49`
- Value/output path: `trnps -> trnps -> matrlz -> fc`, shape `4900,1,128 -> 4900,1,128`
