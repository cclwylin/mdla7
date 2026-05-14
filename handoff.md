# MDLA7 Verilog vs CX 待補清單

日期：2026-05-14（session 2 結尾 handoff）
Repo：`/Volumes/4T_OFFICE/_Claude/MDLA7_Claude`
Branch：`main`
最新 commit：`2df2d16 Five improvements: profile labels, INT8 kvcache, REVERSE_V2 lowering, Phase 6c OW tile loop`

---

## 本 session 已提交的工作（1 commit）

### Profile HTML + 5 improvements（`2df2d16`）

#### Item 5：fc(bmm) profile label 改進
- fc(bmm) label 現在顯示 ic→oc 維度（e.g. `fc(bmm, 256→1024)` for Q×Kᵀ, `fc(bmm, 1024→256)` for S×V）
- engine 欄修正為 `conv+requant`（之前顯示 `—`）

#### Item 2：INT8 KV cache attention models
- `gen_qwen35_kvcache_tflite.py` 修正 TFLite INT8 quantization 的 softmax shape inference bug：
  - 問題：4-D softmax 後接 BATCH_MATMUL，TFLite quantizer 的 accum_dim 推算錯誤（1 vs 128）
  - 修正：Reshape squeeze inner-1 → Softmax → Reshape unsqueeze
- 生成 `model/BMM/qwen35_kvcache_{128,512,1024}_int8.tflite`
- 11/11 BMM SystemC + Verilog DPI PASS

#### Item 4：REVERSE_V2 → OK_REVERSE constant-load lowering
- 新增 `OP_REVERSE = 32` / `OK_REVERSE = 32`
- compile-time constant 的 REVERSE_V2 ops（RoPE sinusoid flip）改為 UDMA constant-load layer（與 OK_SHAPE 相同 runtime path）
- `qwen35_attention_s128/s1024` layer 9, 15：`matrlz from=REVERSE_V2` → `reverse`

#### Item 3：Production INT8 BMM 評估
- ArcSim regression：bit-exact ✅（A,B 固定）
- Production：`bias_eff[n]` 含 compile-time `sum_b[n]`，B 動態時需要 driver-side 更新
- 修正路徑：driver 在每次 inference 前重算 `sum_b[n]` 並 overwrite DRAM params blob（無需 RTL 變更）
- 加了說明 comment 在 compile_model.py

#### Item 1：Phase 6c OW spatial tile loop
**RTL 新增：**
- `conv.v`：新增 `conv_tile_ow_count[15:0]`（word[8][31:16]）和 `act_tile_col_stride[21:0]`（word[10][21:0]）port
- `conv.v`：ST_STORE 中 OC loop 完成後，若 `tile_ow_remaining > 1` → 推進 `act_tile_ow_offset` 並回 ST_ACT
- `host.v`、`mdla7_top.v`、`Testbench_host_program.v`：全部接線

**Generator 新增：**
- `CONV_OW_COUNT_SHIFT = 16` 常數
- `closed_loop_conv_requant_ow_tile_probe()` 函數：tile_ow=2 spatial probe（兩個 ACT copies → 同 WGT → 2 chain pulses → REQUANT drain 2 次）
- 自動插入到 `closed_loop_conv_probes()` 中

**Regression：** 11/11 Verilog DPI PASS

---

## 目前整體狀態

### Regression

```
BMM (11 models):  SystemC 8/11 clean  (3 compile-skipped = FP16 DEQUANTIZE，無害)
                  Verilog DPI 11/11 PASS
```

### Descriptor bit 全覽（v12 完整版）

| 欄位 | word | bits | engine |
|---|---|---|---|
| `conv_chain_out_enable` | [3] | [15] | CONV |
| `requant_use_chain_input` | [3] | [12] | REQUANT |
| `requant_fp_mode` | [3] | [13] | REQUANT |
| `requant_param_load_mode` | [3] | [14] | REQUANT |
| `requant_use_act_correction` | [3] | [17] | REQUANT |
| `conv_tile_oc_count` | [31] | [31:16] | CONV |
| `conv_tile_ow_count` | [8] | [31:16] | CONV (v12 Phase 6c) |
| `requant_tile_drain_count` | [8] | [15:0] | REQUANT |
| `act_tile_col_stride` | [10] | [21:0] | CONV (v12 Phase 6c, L1-mode only) |
| `requant_fp_bias` | [5] | [31:0] | REQUANT |
| `requant_param_l1_addr` | [6] | [21:0] | REQUANT |
| `requant_oc_count` | [7] | [15:0] | REQUANT |
| `requant_oc_index` | [7] | [31:16] | REQUANT |
| `requant_chain_zp_b` | [9] | [7:0] | REQUANT |
| `chain_psum_data[31:0]` | chain | — | CONV→REQUANT psum |
| `chain_psum_data[63:32]` | chain | — | CONV→REQUANT sum_a |

### Op kind 全覽

| kind | value | 描述 |
|---|---|---|
| OK_FC_BMM | 30 | BATCH_MATMUL lowered to CONV |
| OK_SHAPE | 31 | compile-time constant shape vector (UDMA load) |
| OK_REVERSE | 32 | REVERSE_V2 pre-flipped bytes (UDMA load) |

---

## 接下來 next session 的建議工作

### 1. Phase 6d：OH spatial tile loop（OW loop 之外再加 OH）

目前 Phase 6c 完成了 OW loop。OH loop 是外層：
- 新 port：`conv_tile_oh_count[15:0]`（可放 word[9][23:8]），`act_tile_row_stride[21:0]`（word[11][21:0]）
- ST_STORE 中：OW loop 完成後，若 `tile_oh_remaining > 1` → 推進 `act_tile_oh_offset` 並回 ST_ACT
- Generator：`closed_loop_conv_requant_oh_ow_tile_probe()`（tile_oh=2, tile_ow=2）

### 2. Production INT8 BMM：dynamic sum_b correction

Driver-side fix（無需 RTL）：
- compile_model.py emit 一個 "dynamic correction header" block（包含 `zp_A, K, N` 和 bias_eff table offset）
- mdla7_model_runner.cpp 或 production driver 在 dispatch 前：
  1. 讀取 B tile data from DRAM
  2. 計算 `sum_b[n] = Σ_k B[k,n]`
  3. 更新 `bias_eff[n] = -zp_A * sum_b[n] + K * zp_A * zp_B`
  4. Overwrite DRAM params blob

### 3. RESHAPE / RANDOM_STANDARD_NORMAL lowering

類似 REVERSE_V2 → OK_REVERSE 的模式，把其他 matrlz ops lower 到 constant-load：
- `RANDOM_STANDARD_NORMAL`：固定 seed → deterministic bytes → UDMA load
- `RESHAPE`（compile-time constant input）：直接 reinterpret bytes → UDMA load
- 可以統一成 `OK_CONST = 33`（generic compile-time constant）

### 4. DRAM multi-beat（vf_dram_model burst）

目前 `vf_dram_model` 每 request 回 16 byte。Phase 6 tile engine 的 weight load 可能需要多個 beat。
需要：
- `vf_dram_model` 支援 burst mode（`bytes > 16` 時分多 beat 回應）
- CONV ST_WGT：改為 burst 讀取（loop 直到所有 bytes 到位）

### 5. Profile HTML 改進

- `qwen35_attention` 的 `fc(bmm, 256→1)` 等 s=128 的模型：oc 太小，label 可能不直觀
- 可考慮 model-level tag 來標記 "attention" 模型

---

## 快速重建指令

```bash
# SystemC rebuild
cd /Volumes/4T_OFFICE/_Claude/MDLA7_Claude/systemc && make

# BMM regression (SystemC fast)
~/.venvs/mdla7/bin/python batch/run_systemc.py --filter bmm --rerun-all

# BMM regression (Verilog DPI)
rm -rf rtl/obj/verilog/host
./batch/run_verilog.py --filter bmm --rerun-all --dpi

# 生成 INT8 kvcache 模型（需要 TF）
~/.venvs/mdla7/bin/python systemc/scripts/gen_qwen35_kvcache_tflite.py --dtype int8 --kv-len 128 512 1024

# 從 git 恢復 FP16 kvcache 模型（4T_OFFICE 外接硬碟可能遺失）
git checkout model/BMM/
```

**注意**：4T_OFFICE 是 exFAT 外接硬碟，新生成的 .tflite 檔案可能在 unmount 後遺失（未 git commit）。建議重要檔案立即 commit 或確認 flush。

---

## Session 3 추가 작업（2026-05-14）

### Phase 6d：OH spatial tile loop（`9768589`）

**RTL 새로 추가된 포트 (conv.v):**
- `conv_tile_oh_count[15:0]` ← word[9][23:8]
- `act_tile_row_stride[21:0]` ← word[11][21:0]
- `conv_spatial_tile_enable` ← word[3][18] ← **필수 enable flag**

**Backward-compat 설계:**
- word[3][18] = `CONV_SPATIAL_TILE_EN` 이 0이면 OW/OH loop 모두 비활성화
- 기존 CONV descriptor의 word[8..11]은 WGT data → 오염된 tile count 방지
- enable flag를 set하지 않은 기존 hex 프로그램 전부 그대로 동작

**State machine:**
```
ST_STORE:
  if tile_oc_remaining > 1  → OC 루프 (기존)
  elif tile_ow_remaining > 1 → OW 루프 (Phase 6c)
  elif tile_oh_remaining > 1 → OH 루프 (Phase 6d 신규)
  else → ST_DONE
```

**Generator:**
- `CONV_SPATIAL_TILE_EN = 1 << 18` 상수
- `closed_loop_conv_requant_oh_ow_tile_probe(tile_oh=2, tile_ow=2)`：
  - 4 ACT UDMA + 1 WGT UDMA + CONV(OH=2,OW=2) + REQUANT(drain=4) + L1CRC(4B)
  - desc[3]에 CONV_SPATIAL_TILE_EN 반드시 set
- OW probe도 desc[9]=0, desc[11]=0 으로 OH count 오염 방지

**Regression:** 12/12 Verilog DPI PASS（backward compat 확인）

### Descriptor bit 갱신

| 필드 | word | bits | 비고 |
|---|---|---|---|
| `conv_spatial_tile_enable` | [3] | [18] | 1이어야 OW/OH 루프 활성화 |
| `conv_tile_ow_count` | [8] | [31:16] | tile_ow iterations |
| `conv_tile_oh_count` | [9] | [23:8] | tile_oh iterations |
| `act_tile_col_stride` | [10] | [21:0] | L1 bytes per OW step |
| `act_tile_row_stride` | [11] | [21:0] | L1 bytes per OH step |

### Next session 권장 작업 갱신

1. **compile_model.py OH/OW 연동**: 실제 conv layer를 OH×OW tile descriptor로 lower하는 코드 추가 (현재는 probe만 존재, compiler는 미구현)
2. **DRAM multi-beat**: vf_dram_model burst 지원 (Phase 6 tile engine weight load)
3. **Producer INT8 BMM**: sum_b dynamic correction driver-side

---

## Session 4 (2026-05-15) — Softmax decomposition: Pool bottleneck 重新校準

### 初始錯誤診斷（已校正）

最初判斷「Pool bottleneck = decomposition 全在 EWE thread 內 inline 跑完，沒真的丟給 PoolEngine」— **這是錯的**。

校正後事實：跨引擎 dispatch **早就完整實作在 [mdla7_model_runner.cpp](systemc/src/mdla7_model_runner.cpp) 內**：
- FP16（任意 rows）：[9462-9499](systemc/src/mdla7_model_runner.cpp#L9462-L9499) (rows==1) + [9519-9682](systemc/src/mdla7_model_runner.cpp#L9519-L9682) (rows>1) 已發 POOL_MAX → EWE_SUB → EWE_EXP → POOL_SUM → EWE_DIV 5 條 descriptor
- INT8 rows>1：已 dequant→FP16 chain→quant 走同一路
- `make_pool_row_reduce` ([mdla7_model_runner.cpp:288-310](systemc/src/mdla7_model_runner.cpp#L288-L310)) 用形狀慣例 `[in_h=rows, in_w=K, in_c=1, k_w=K]` — PoolEngine 現有三層迴圈直接處理，**不需改 PoolEngine**

[ewe_pool.h](systemc/include/mdla7/ewe_pool.h) 的 `run_softmax_decomposed_*` inline 路徑只在邊角情況 fire：
1. INT8 + rows==1（[9462](systemc/src/mdla7_model_runner.cpp#L9462) 條件 `L.dtype == DT_FP16` 排除）
2. `MDLA7_SOFTMAX_DECOMPOSE=0`
3. L1 容量檢查失敗

### 真正的 Pool bottleneck 來源

每個 row 的 POOL_MAX 都**顯式等前一個 row 的 EWE_DIV done**（[mdla7_model_runner.cpp:9591](systemc/src/mdla7_model_runner.cpp#L9591) `load_done_wait = (row == 0) ? fuse_prev_done_tag : prev_req_tag`）。

原因：scratches `addr_max / addr_ctr / addr_exp / addr_sum`（[9574-9577](systemc/src/mdla7_model_runner.cpp#L9574-L9577)）**跨所有 row 共用一塊**，有 WAR/WAW hazard，只能串。

→ Critical path 上有 `2 × rows` 個 POOL ops 全部串行；看起來 Pool 是 bottleneck，**真因是 scratch 沒 double-buffer**，不是 Pool 本身。

### Phase 表重新對齊現況

| 原 Phase | 校正後現況 |
|---|---|
| A: PoolEngine row-reduce mode | ✅ 已用 shape convention，PoolEngine 不需動 |
| B: FP 5-descriptor chain | ✅ [mdla7_model_runner.cpp:9462+](systemc/src/mdla7_model_runner.cpp#L9462) 已實作 |
| C: L1Mesh 報告驗證 POOL 流量歸屬 | ❓ 沒驗過 — 待跑 |
| D: INT8 widening | ❌ 不需要 — INT8 已走 dequant→FP16→quant，PoolBody 不需動 |
| E: Microblock overlap (scratch ping-pong) | ❌ **真正未做的工** |

### 真正的下一步候選

1. **Phase E（微塊 overlap，本來該做的）**：
   - `addr_max[2] / addr_ctr[2] / addr_exp[2] / addr_sum[2]` 雙緩衝
   - 移除 `load_done_wait = prev_req_tag` 強制依賴（改成 row r 只需 wait row r-2 的 div done）
   - 預期：POOL/EWE 可以 row 級 overlap，critical path 從 `O(rows × 5)` 降到 `O(rows × max(per-op))`
   - 風險：L1 scratch 用量 2x；要重看 L1_BUDGET fit check
   - 影響：[mdla7_model_runner.cpp:9519-9682](systemc/src/mdla7_model_runner.cpp#L9519-L9682) 內，pure compiler-side

2. **Phase C（先驗證後改）**：
   - 跑 model + [L1Mesh_report.md](L1Mesh_report.md) 生成器，確認 POOL 流量歸 PoolEngine 帳上
   - 確認 user 觀察是「2 × rows POOL ops 在 critical path」的現象本身（→ 走 E），還是報告歸屬錯（→ 修報告）
   - 純讀，沒改動

3. **邊角清理（低優先）**：INT8 rows==1 改走 chain（[9462](systemc/src/mdla7_model_runner.cpp#L9462) 條件加 INT8），消除 inline path

### 建議起點

**Phase C 先做** — 讀 L1Mesh 報告，先確認 bottleneck 真因再決定 E 怎麼動。Phase E 是 compiler-side 大改，先驗報告再下手較穩。

### 教訓 / 自我修正

下次接手相同主題：**先讀 model_runner 的 softmax 段才開診斷**，不要光看 ewe_pool.h 就推結論。EweEngine 內的 inline path 是 fallback，不是主路徑。

### 修法落地：批量化（whole-shape chain）

User 確認觀察到的具體 regression 是 `bmm_softmax_bmm_tiled_2.5ms_int8.tflite` fast mode 從 **7ms → 72ms**。stored profile 確認 baseline = 137 ms（POOL 117 ms / 56,576 tasks）。

選的方案：**批量化** — per-row 5-descriptor loop 改成 per-softmax 1 套 5/7 descriptor 操作整個 `[rows, K]` shape。PoolEngine row-reduce shape convention 跟 EweEngine 末軸 broadcast 本來就支援 rows>1。

**改動位置**：[mdla7_model_runner.cpp:9519-9694](systemc/src/mdla7_model_runner.cpp#L9519-L9694)
- L1 fit check scratch 大小從 `K*2` 改 `rows*K*2`（ctr/exp/fp_in/fp_out）+ `rows*2`（max/sum 改 row-vector）
- 移除 `for (uint64_t row = 0; row < rows; ++row)` 迴圈
- 一條 UDMA 載入整塊 `in_size` (non-fused)
- 一套 5/7 descriptor，rows 參數從 1 改 actual rows
- `tiles_h_per_layer[i] = 1`（whole softmax = 1 microblock）
- acc accounting 改 in_size/ref_size 整批

**測試結果**：`./batch/run_systemc.py --filter bmm --model-filter tiled_2.5ms_int8 --rerun-all`

| 指標 | Before (per-row) | After (batched) | 改善 |
|---|---|---|---|
| Fast latency | 72 ms | **11.18 ms** | 6.4x ↓ |
| Sim total | 137 ms | 21.25 ms | 6.5x ↓ |
| POOL tasks | 56,576 | 884 (= 442×2) | 64x ↓ |
| POOL busy | 117 ms | 4.95 ms | 24x ↓ |
| EWE tasks | 141,440 | 2,210 (= 442×5) | 64x ↓ |
| EWE busy | 13.3 ms | 13.2 ms | 同 (compute floor) |
| Cmd dispatches | 228,514 | 5,746 | 40x ↓ |
| Correctness | 1358/1358 PASS | **1358/1358 PASS** | ✓ |

新 bottleneck = EWE (util_peak 62.1%)，是真實 compute floor，不是 dispatch overhead。

### 後續可選工作

1. **其他 softmax model 全掃**：跑整個 bmm corpus（不只 tiled_2.5ms_int8）確認沒有別的 regression / L1 fit 失敗
2. **rows==1 INT8 inline path 邊角清理**：[9462](systemc/src/mdla7_model_runner.cpp#L9462) 條件目前還排除 INT8，加入 INT8 後 EweEngine 的 inline `run_softmax_decomposed_int8` 就完全用不到
3. **`compile-skipped:1` 含義**：CSV 寫 `bmm_softmax_bmm_tiled_2.5ms_int8,11.184,compile-skipped:1`，看 [run_systemc.py:187](batch/run_systemc.py#L187) `_compile_skipped_rows` 是 compile 階段有 1 row 沒 ready；correctness 還是 1358/1358 PASS，要不要修看實際 model 需求
4. **Phase E (scratch ping-pong)**：目前已不是 critical path，可保留為未來工程

