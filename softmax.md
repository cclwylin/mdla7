# Softmax MB Chain → L1Mesh Stress Test

## 目標

把 `bmm_softmax_bmm_2.5ms_1g_int8.tflite` 的三個 op 串成
**全程留在 L1Mesh 的 microblock chain**，不讓 score/probs tensor spill DRAM，
藉此**壓測 L1Mesh 的 bank conflict、NoC 路由、L1_Manager 仲裁能力**，
並以 fast / cx / verilog 三種 timing mode 跑結果來調整 MDLA7 設計參數。

---

## Tile Loop 演算法（Online Softmax）

```
for each Q_tile [1, H, T_q, D]:          # outer loop：query rows
    m = -∞,  s = 0,  O = 0              # running stats（L1 常駐）

    for each K_tile / V_tile [1, H, T_k, D]:   # inner loop：key/value blocks

        # ── Phase A：BMM1（CONV）──────────────────────────────
        score_tile = CONV(Q_tile, K_tile^T)     # [1,H,T_q,T_k]，留在 L1

        # ── Phase B：EWE + POOL（softmax stats）───────────────
        m_new    = max(m, POOL_max(score_tile)) # POOL，scalar per row
        rescale  = EWE_exp(m − m_new)           # EWE，scalar
        O = O × rescale                         # EWE
        e_tile   = EWE_exp(score_tile − m_new)  # EWE，[1,H,T_q,T_k]
        s = s × rescale + POOL_sum(e_tile)      # EWE + POOL
        m = m_new

        # ── Phase C：BMM2（CONV）──────────────────────────────
        O = O + CONV(e_tile, V_tile)            # [1,H,T_q,D]，留在 L1

    output_tile = EWE(O / s)                    # 只在此 flush DRAM
```

### L1 佔用（T_q = T_k = 64，B=1，H=32，D=128，INT8）

| Buffer | Shape | Size |
|--------|-------|------|
| Q_tile | [1,32,64,128] | 262 KB |
| K_tile | [1,32,64,128] | 262 KB |
| V_tile | [1,32,64,128] | 262 KB |
| score_tile / e_tile | [1,32,64,64] | 131 KB |
| O（output accumulator）| [1,32,64,128] | 262 KB |
| m, s（running stats）| [1,32,64] × 2 | 4 KB |
| **合計** | | **~1.2 MB < 3 MB ✓** |

Score matrix [134 MB] 完全不 materialize，DRAM 只寫最終 output。

---

## L1Mesh 壓力分析

### Phase A / C：BMM（CONV 主導）

| Engine | 方向 | Payload 路數 | 走哪條路 |
|--------|------|------------|---------|
| CONV | ACT_R（Q/e_tile） | **32** | CONV 專線，bypass L1_Manager |
| CONV | WGT_R（K/V_tile） | **32** | CONV 專線，bypass L1_Manager |
| Requant | W（score/O_delta 寫回）| **8** | L1_Manager |
| UDMA | W（預取下一組 K/V tile）| **16** | L1_Manager |

**3 engines；L1_Manager W 需求 24W → 上限 16W（Requant + UDMA 競爭）**

### Phase B：EWE + POOL（softmax stats）

| Engine | 方向 | Payload 路數 | 走哪條路 |
|--------|------|------------|---------|
| POOL | R（score_tile/e_tile）| **16** | L1_Manager |
| POOL | W（m, s scalar）| **8** | L1_Manager |
| EWE | R（O, score_tile）| **16** | L1_Manager |
| EWE | W（e_tile, O）| **8** | L1_Manager |

**2 engines；需求 32R + 16W → 上限 16R + 16W（EWE + POOL round-robin 分時）**

### 三個被壓測的面向

| 壓測面向 | 發生時機 | 用哪個 timing mode 看 |
|---------|---------|----------------------|
| **Bank conflict** | BMM phase：CONV 64R + Requant/UDMA W 爭同一 bank | `--l1-timing=conflict` |
| **NoC 路由競爭** | Phase A↔B 切換：CONV 專線與 L1_Manager lane 同時 in-flight | `--l1-timing=mesh` |
| **L1_Manager 仲裁** | Phase B：EWE+POOL 搶 16R；Phase A：Requant+UDMA 搶 16W | `--l1-timing=mesh` |

---

## 跑測試計畫

```bash
# Fast（aggregate BW baseline）
./batch/run_systemc.py --filter bmm --fast-only --rerun-all --no-html

# CX（per-bank conflict）
./batch/run_systemc.py --filter bmm --cx --fast-only --rerun-all --no-html

# Verilog（closed-loop 最終驗證）
./batch/run_verilog.py --filter bmm --rerun-all --timeout 180
```

### 結果 → 設計參數對應表

| 觀察到的問題 | 可調參數 |
|------------|---------|
| BMM phase bank conflict 嚴重 | L1_Manager W lane 數 / Requant+UDMA 分時策略 |
| Phase B EWE stall 多 | EWE R lane 數（目前 16，可考慮升至 32） |
| UDMA 預取與 Requant 衝突 | L1_Manager arbitration policy（round-robin → priority） |
| NoC router 擁塞 | Router input FIFO depth（目前 provisional 2 flits） |
| Tile overhead 大 | T_q / T_k 往上調（64 → 128），重新算 L1 fit |

---

## 實作待辦

### Step 1：生成 TFLite
- [ ] 確認 EWE RTL 有無 exp activation（或需 TNPS LUT）
- [ ] `gen_bmm25_int8_tflite.py` 加 `--tiled` flag，展開 tile loop graph
- [ ] 輸出 `model/BMM/bmm_softmax_bmm_2.5ms_1g_mb_int8.tflite`

### Step 2：compile_model.py lowering
- [ ] 辨識 `BATCH_MATMUL → SOFTMAX → BATCH_MATMUL` pattern
- [ ] Lower 成 tile loop（outer=Q tile，inner=K/V tile，T_q=T_k=64）
- [ ] loop body 插入 rescale step（running max 更新時）
- [ ] running stats `m`, `s`, `O` 配置在 L1，不走 DRAM

### Step 3：跑三種 timing mode 並記錄
- [ ] fast vs cx vs mesh 的 cycle 差距
- [ ] 識別主要 bottleneck
- [ ] 提出 RTL 參數調整建議

---

## 相關檔案

- `model/BMM/bmm_softmax_bmm_2.5ms_1g_int8.tflite` — 基礎 model
- `model/BMM/bmm_softmax_bmm_tile_32h_64q_64k_int8.tflite` — tile-sized 驗證 model (commit 6782435 新增)
- `systemc/scripts/gen_bmm25_int8_tflite.py` — tile-model generator (commit 6782435 新增)
- `systemc/scripts/compile_model.py` — lowering pass 修改點 (Stage C 改這裡)
- `systemc/include/mdla7/descriptor.h` — Stage A 改這裡 (EweSubtype / PoolMode)
- `systemc/include/mdla7/ewe_pool.h` — Stage B 改這裡 (EweEngine / PoolEngine compute)
- `systemc/src/mdla7_model_runner.cpp` — Stage D 改這裡 (post-override L1 chain pass)
- `rtl/verilog/ewe.v`, `rtl/verilog/pool.v` — Stage E 改這裡 (subtype dispatch)
- `spec/spec.md §3.2b` — L1Mesh 16-bank 組織與 edge lane 分配

---

## Handoff — 2026-05-14 session end

### 上一輪做到哪 (commit 6782435)

1. ✅ `gen_bmm25_int8_tflite.py` 完成 (tile model: 32h × 64q × 64k INT8, score tile = 128 KB)
2. ✅ runner 加 post-override pass: fc(bmm) run → softmax 全鏈 `producer_no_store=true`
   - score tile DRAM 寫入 128 KB → 0 B
   - fast (57,259 cy) / cx (42,923 cy) / Verilog (13/13 DPI) 全過
3. ❌ **softmax 仍是 monolithic `ES_SOFTMAX`** — 沒做到 §Phase B 的真 EWE+POOL 分解

User re-set goal → 本輪 (下一個 session) 要做真分解.

### 本輪 scope 確認 (Phase 1+2)

把 SOFTMAX 拆成 5-op chain，全程留在 L1:

```
POOL_MAX (PM_MAX)
  → EWE_SUB  (ES_SUB, broadcast last axis)
  → EWE_EXP  (ES_EXP new)
  → POOL_SUM (PM_SUM new)
  → EWE_DIV  (ES_DIV new, broadcast last axis)
```

不做 Stage 3 (full flash-attention tile loop with running m/s/O) — 那是 softmax.md 長期計畫，
本次 goal 文字裡沒要求。

### 設計決策

1. **中間 tensor dtype**: INT8 model 在 POOL_MAX 之後改 FP16，最後 EWE_DIV requant 回 INT8.
   - 理由: INT8 每個 sub-op 之間做 quant/dequant 太麻煩，會 break bit-exactness.
   - L1 size check: score [32,64,64] FP16 = 256 KB；centered + exp = 2 × 256 KB = 512 KB.
   - max/sum scalars [32,64] FP16 = 4 KB.
   - 總計 ~770 KB intermediates + 256 KB × 3 (Q/K/V) = ~1.5 MB ✓ < 3 MB L1.
2. **每個 sub-op 的 reference output 各算各的**: compile_model.py 在 lower SOFTMAX
   時 emit 5 個 layer，每個 layer 自己的 ref 是 FP32 算完後 cast 到 layer dtype.
   每個 layer 對自己的 ref bit-exact；downstream BMM2 讀第 5 layer 的 INT8 輸出，
   不用改 BMM2 ref.
3. **不保留 monolithic path**: 直接 lower 成 5 layer. 舊的 `softmax_int8_ref` /
   `softmax_fp_ref` 保留給 non-decomposed model (env flag default off).
4. **POOL row-axis reduction trick**: 把 score [rows, K] 視為 PoolBody
   `in_h=rows, in_w=K, in_c=1`, `k_h=1, k_w=K, stride=1, pad=0` → 每 row 一個 reduction.
   PM_MAX 和 PM_SUM 都用這 trick (不用新 axis 機制).

### 實作清單

#### Stage A — Descriptor (systemc/include/mdla7/descriptor.h)
- [ ] `enum EweSubtype`: 加 `ES_EXP = 9`, `ES_DIV = 10`
- [ ] `enum PoolMode`:   加 `PM_SUM = 3`

#### Stage B — SystemC compute (systemc/include/mdla7/ewe_pool.h)
- [ ] `run_pool_sum<T>` (mirror `run_pool_int` AVG path, skip divide)
- [ ] `run_pool_sum_fp` (mirror `run_pool_fp` AVG path, skip divide; FP16 storage)
- [ ] `run_exp_int8_lut` / `run_exp_fp` (unary; 可重用 existing LUT 路徑)
- [ ] `run_div_int8` (binary; 需要 quant params blob)
- [ ] `run_div_fp` (FP16 binary; 加到 `run_binary_fp` 的 op 列表)
- [ ] `EweEngine::run()` dispatch: 加 `ES_EXP`, `ES_DIV` branches
- [ ] `PoolEngine::run()` dispatch: 加 `PM_SUM` branch (or extend AVG → no-div mode)

#### Stage C — compile_model.py
- [ ] 新 helper `decompose_softmax(layer, prev_dtype)`:
  - L1: POOL_MAX (PM_MAX) → max per row (INT8/FP16)
  - L2: EWE_SUB (ES_SUB, broadcast) → centered (FP16 if input INT8)
  - L3: EWE_EXP (ES_EXP) → e_tile FP16
  - L4: POOL_SUM (PM_SUM) → s per row FP16
  - L5: EWE_DIV (ES_DIV, broadcast) → prob INT8 (用原 softmax 的 output zp/scale)
- [ ] 在 SOFTMAX lowering 點 (compile_model.py:3259-3277) 改：emit 5 layer 而不是 1 個 OK_SOFTMAX
- [ ] env flag `MDLA7_DECOMPOSE_SOFTMAX=1` 控制是否分解 (default off → existing regression 不影響)
- [ ] 每個 sub-op layer metadata 正確 (out_h/out_w/out_c, dtype, op_kind)

#### Stage D — runner post-override pass (systemc/src/mdla7_model_runner.cpp)
- [ ] 現有 pass (line ~1531-1558) cover "fc(bmm)→softmax"
- [ ] 擴充 detect "fc(bmm)→POOL_MAX→EWE_SUB→EWE_EXP→POOL_SUM→EWE_DIV→fc(bmm)" 6+ layer 鏈
- [ ] 全鏈 `producer_no_store = true` (EWE_DIV → BMM2 也只寫 L1, 不寫 DRAM)
- [ ] L1 budget check: 確認 ~1.5 MB intermediate fits

#### Stage E — Verilog RTL
- [ ] `rtl/verilog/ewe.v`: 看現有 subtype dispatch (現在 case 用 2'h0~2'h3, line 472)
  - 加 ES_EXP (subtype 9) — INT8 LUT path (可共用 RSQRT/TANH/LOGISTIC datapath)
  - 加 ES_DIV (subtype 10) — FP16 binary path (新 ALU)
- [ ] `rtl/verilog/pool.v`: 加 PM_SUM mode (跟 AVG datapath 共用, 最後一步 skip divide)
- [ ] `batch/gen_verilog_program.py`: 確認 subtype 寫入 descriptor word 正確

#### Stage F — verify + L1Mesh report
- [ ] fast: `./batch/run_systemc.py --filter bmm_softmax_bmm_tile --fast-only --rerun-all --no-html`
- [ ] cx:   `./batch/run_systemc.py --filter bmm_softmax_bmm_tile --cx --fast-only --rerun-all --no-html`
- [ ] verilog: `./batch/run_verilog.py --filter bmm_softmax_bmm_tile --rerun-all --timeout 180`
- [ ] L1Mesh report: 確認 POOL_R / POOL_W / EWE_R / EWE_W 各 column 出現 5 個 sub-op 的 traffic 分布

### 已知風險 / 未知

1. **INT8 ↔ FP16 dequant overhead**: 5-op chain 之間多 quant 轉換, total cycles 可能比 monolithic 多 2-3×
2. **Verilog EWE subtype dispatch**: `ewe.v:472` 現用 2'h0~2'h3 (只 2 bit), 不確定還能不能直接擴 — 可能要加 wider subtype decode
3. **POOL_SUM bit-exactness**: FP32 sum 順序敏感, `run_pool_fp_ref` 跟 RTL pool reduction tree order 要對齊
4. **Backward compat**: 舊 INT8 model (EfficientNet 含 softmax) 要走 monolithic path 才不會 regress → env flag default off; 確認 regression suite 沒 set 該 flag

### Pickup points (給下一個 session)

讀完以下 6 處然後直接從 Stage A 開始：

1. [systemc/include/mdla7/descriptor.h:117-127](systemc/include/mdla7/descriptor.h#L117-L127) — EweSubtype enum
2. [systemc/include/mdla7/descriptor.h:59-63](systemc/include/mdla7/descriptor.h#L59-L63) — PoolMode enum
3. [systemc/include/mdla7/ewe_pool.h:511-666](systemc/include/mdla7/ewe_pool.h#L511-L666) — EweEngine::run dispatch
4. [systemc/include/mdla7/ewe_pool.h:875-948](systemc/include/mdla7/ewe_pool.h#L875-L948) — PoolEngine::run dispatch
5. [systemc/src/mdla7_model_runner.cpp:1531-1558](systemc/src/mdla7_model_runner.cpp#L1531-L1558) — 現有 fc(bmm)→softmax chain pass (擴充這裡)
6. [systemc/scripts/compile_model.py:3259-3277](systemc/scripts/compile_model.py#L3259-L3277) — SOFTMAX lowering

每完成一個 Stage 就 commit 一次。Stage A+B 是 mechanical; Stage C 是 reference math (要小心 quant);
Stage D 沿用 commit 6782435 的 pattern; Stage E 是最大未知 (RTL subtype 解碼可能要加 bit).
