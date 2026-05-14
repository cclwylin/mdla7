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
# Engine-internal decomposition runs by default — one ES_SOFTMAX
# descriptor that internally runs the 5-phase chain (visible per-phase
# in the EWE engine trace, but still a single descriptor in the
# program stream). Set the env to 0 to opt out for a baseline diff:
# export MDLA7_SOFTMAX_DECOMPOSE=0

# Descriptor-level (Stage C, commits 679783f + 13f54aa): compiler
# emits the chain as 5 distinct descriptors per row — POOL_MAX,
# EWE_SUB, EWE_EXP, POOL_SUM, EWE_DIV — so the EWE and POOL engines
# pick the work up independently and the L1Mesh sees real per-engine
# R/W traffic. INT8 inputs get DEQUANT_INT8 + chain + QUANT_FP_INT8
# (7 sub-descriptors per row). Still opt-in:
export MDLA7_DECOMPOSE_SOFTMAX=1

./batch/run_systemc.py --filter bmm --fast-only --rerun-all --no-html
./batch/run_systemc.py --filter bmm --cx --fast-only --rerun-all --no-html
./batch/run_verilog.py  --filter bmm --rerun-all --timeout 180 --no-html
```

The engine-internal decomposition is bit-exact vs the monolithic
LUT/FP softmax and so it is on by default. The descriptor-level
decomposition is still opt-in via `MDLA7_DECOMPOSE_SOFTMAX=1` because
it changes the program stream layout (5–7× more descriptors per
softmax row) which affects scheduling and is not always what a perf
sweep wants.

## Descriptor-level decomposition (MDLA7_DECOMPOSE_SOFTMAX, FP only)

Verified by running each model with the flag on and off in cx mode and
diffing the per-engine task counts:

`bmm_softmax_bmm_fp32` (rows = 16, K = 8, FP16, fused with prev MUL):

| Mode | Total cyc | EWE busy | EWE tasks | POOL busy | POOL tasks | softmax row | tiles_h |
|------|----------:|---------:|----------:|----------:|-----------:|------------:|--------:|
| cx-monolithic | 764   | 67    |  2 | 0   |  0 | streamed (0) |  1 |
| cx-decomposed | 2,620 | 1,138 | 49 | 831 | 32 | 1,856        | 16 |

`qwen35_attention_s128` (softmax layer 52: rows = 8, K = 1, FP16):

| Mode | Total cyc | EWE busy | EWE tasks | POOL busy | POOL tasks |
|------|----------:|---------:|----------:|----------:|-----------:|
| cx-monolithic | 186,545 | 364 |  8 | 0   |  0 |
| cx-decomposed | 187,574 | 890 | 31 | 304 | 16 |

`bmm_softmax_bmm_int8` (rows = 16, K = 8, **INT8** with dequant/requant wrap):

| Mode | Total cyc | EWE busy | EWE tasks | POOL busy | POOL tasks |
|------|----------:|---------:|----------:|----------:|-----------:|
| cx-monolithic |   676 |   48 |  2 |   0 |  0 |
| cx-decomposed | 3,298 | 1,764 | 81 | 829 | 32 |

For the INT8 case every row emits 7 sub-descriptors — DEQUANT_INT8,
POOL_MAX, EWE_SUB, EWE_EXP, POOL_SUM, EWE_DIV, QUANT_FP_INT8 — so EWE
jumps from 2 → `1 + 16 × 5 = 81` tasks (baseline MUL plus DEQUANT +
SUB + EXP + DIV + QUANT per row) and POOL goes from 0 → `16 × 2 = 32`.

Task-count delta matches the predicted chain. POOL goes from 0 →
`rows × 2` tasks (it had no work in the monolithic path); EWE picks up
the remaining `rows × 3` (or `rows × 5` on INT8) on top of its baseline
traffic.

Functional regression: BMM `fast` and `cx` both clean 9/13 with the
flag on, matching the pre-decomp baseline. The 4 pre-existing fails
(bmm_softmax_bmm_2.5ms_1g_int8 and three qwen35 fp16 compile-skips)
are unchanged. INT8 softmax outputs no longer come from the LUT path
when the flag is on, but the downstream BMM₂ still matches its
reference for every model with an INT8 softmax layer that previously
passed.
