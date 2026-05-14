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
