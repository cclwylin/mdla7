# L1Mesh Report — Softmax MB Chain Decomposition

Model: `bmm_softmax_bmm_tile_32h_64q_64k_int8`  (BMM₁ → SOFTMAX → BMM₂, score tile 128 KB)

## Total simulation

| Mode | Total cycles | EWE busy | L1Mesh busy | L1Mesh accesses | L1Mesh bytes | L1Mesh wait |
|------|-------------:|---------:|------------:|----------------:|-------------:|------------:|
| fast-monolithic | 59,780 | 12,123 | 28,896 | 1,127 | 3,552,865 | 28,897 |
| fast-decomposed | 72,915 | 24,858 | 39,583 | 1,136 | 5,424,737 | 39,584 |
| cx-monolithic | 42,577 | 9,058 | 6,005 | 1,127 | 3,552,865 | 2,624 |
| cx-decomposed | 45,322 | 11,803 | 6,701 | 1,136 | 5,424,737 | 3,294 |

## Softmax layer alone (layer 34)

| Mode | Cycles | SRAM read | SRAM write |
|------|-------:|----------:|-----------:|
| fast-monolithic | 7,641 | 262,144 | 131,072 |
| fast-decomposed | 20,376 | 262,144 | 131,072 |
| cx-monolithic | 6,244 | 262,144 | 131,072 |
| cx-decomposed | 8,988 | 262,144 | 131,072 |

## L1Mesh lane balance (fast-decomposed)

- 16 read lanes — avg 177,715 B/lane
- 16 write lanes — avg 161,331 B/lane
- Router wait: 11,507 cyc
- SRAM wait:   62,436 cyc

## Verilog mode

All 13 BMM models PASS in Verilog mode (closed_loop_dataflow + final-layer).
SOFTMAX layers use `closed_loop_ref_passthrough_probe` (pre-loaded reference)
so the SystemC-side EWE+POOL decomposition does not change Verilog behavior;
the surrounding BMM₁/BMM₂ chain still validates the L1 handoff at the RTL
boundary.

| Program                              | Class  | Result | verilog_cyc |
|--------------------------------------|--------|--------|------------:|
| bmm_softmax_bmm_2.5ms_1g_int8        | huge   | PASS   | 21,600      |
| bmm_softmax_bmm_fp32                 | small  | PASS   | 142         |
| bmm_softmax_bmm_int8                 | small  | PASS   | 140         |
| bmm_softmax_bmm_sam_quant_L22_L61    | medium | PASS   | 179,500     |
| bmm_softmax_bmm_tile_32h_64q_64k_int8| medium | PASS   | 772         |
| qwen35_attention_s1024               | medium | PASS   | 1,400       |
| qwen35_attention_s128                | medium | PASS   | 1,400       |
| qwen35_kvcache_{128,512,1024}_fp16   | small  | PASS   | ≈156        |
| qwen35_kvcache_{128,512,1024}_int8   | small  | PASS   | ≈146        |

## 5-phase decomposition (POOL_MAX → EWE_SUB → EWE_EXP → POOL_SUM → EWE_DIV)

Each phase issues real `l1mgr.read()` / `l1mgr.write()` against a 1 MB
scratchpad at the top of L1MESH. For the tile model softmax (32 heads ×
64 query × 64 key, INT8):

| Phase    | L1 read         | L1 write       |
|----------|-----------------|----------------|
| POOL_MAX | 128 KB in_a     | 2 KB max       |
| EWE_SUB  | 128 KB + 2 KB   | 128 KB diff    |
| EWE_EXP  | 128 KB diff     | 512 KB exp_q (INT32) |
| POOL_SUM | 512 KB exp_q    | 16 KB sum (INT64) |
| EWE_DIV  | 512 KB + 16 KB  | 128 KB out     |
| **Total**| **1.43 MB**     | **786 KB**     |

vs monolithic: 128 KB read + 128 KB write. Extra L1 traffic accounts for
the ~1.87 MB increase in L1Mesh bytes in the totals table above.

## Activate

```bash
# SystemC fast/cx modes pick up the decomposition with this env flag:
export MDLA7_SOFTMAX_DECOMPOSE=1

./batch/run_systemc.py --filter bmm --fast-only --rerun-all --no-html
./batch/run_systemc.py --filter bmm --cx --fast-only --rerun-all --no-html
./batch/run_verilog.py  --filter bmm --rerun-all --timeout 180 --no-html
```

Decomposition is opt-in via env flag because the 1 MB scratchpad at the
top of L1MESH collides with models that already saturate L1; toggle off
to revert to the monolithic LUT softmax.
