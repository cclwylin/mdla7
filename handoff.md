# MDLA7 Verilog vs CX 待補清單

日期：2026-05-14（session 結尾 handoff）
Repo：`/Volumes/4T_OFFICE/_Claude/MDLA7_Claude`
Branch：`main`
最新 commit：`7755b00 Add qwen35 KV-cache attention models + fix FP16 BMM reference`

---

## 本 session 全部已提交的工作（10 commits）

### v12 Phase 1-5 RTL chain wiring（`ca31f7e`）

完成 CONV→REQUANT chain descriptor 接線：

| 新 descriptor 欄位 | bit 位置 | 用途 |
|---|---|---|
| `conv_chain_out_enable` | word[3][15] | CONV 是否 push chain（sample mode=0）|
| `requant_use_chain_input` | word[3][12] | REQUANT 是否從 chain 讀 psum |
| `requant_fp_mode` | word[3][13] | REQUANT FP16 輸出模式 |
| `requant_param_load_mode` | word[3][14] | REQUANT per-OC table 模式 |
| `requant_fp_bias` | word[5] | FP bias（chain 模式用）|
| `requant_param_l1_addr` | word[6][21:0] | per-OC param table L1 位址 |
| `requant_oc_count` | word[7][15:0] | per-OC param 數量 |
| `requant_oc_index` | word[7][31:16] | 當前 OC 索引 |

- `conv.v`：chain_out_enable gating；`chain_skip_store`（chain mode 跳過 L1 write）；FP chain payload 改用 `$shortrealtobits(fp_sum)` 正確 fp32 cast
- `requant.v`：fp_bias 套用 `$bitstoshortreal` FP32 add
- 全部通過 `host.v` → `mdla7_top.v (_q flop)` → instance 接線 → `Testbench`

### v12 Phase 4-6 tile engine + generator（`4785e3d`）

- **Phase 5 step 1**：`chain_skip_store = chain_out_enable && !conv_partial_final`，chain mode 下 ST_STORE 不發 L1 request 直接 advance
- **Phase 4**：`closed_loop_conv_requant_chain_probe()`：UDMA + CONV(chain) + REQUANT(use_chain) + L1CRC；`CONV_CHAIN_OUT_FLAG=1<<15`、`REQUANT_USE_CHAIN=1<<12` 常數
- **Phase 6a** REQUANT tile drain：`tile_drain_count` port（word[8][15:0]），`tile_drain_remaining` reg，ST_DONE 後 advance `active_out_byte_offset`，loop 回 ST_CHAIN_WAIT
- **Phase 6b** CONV OC tile loop：`conv_tile_oc_count` port（word[31][31:16]），`wgt_tile_l1_offset` reg，ST_STORE 後 advance `wgt_tile_l1_offset += workload_sample_bytes`，loop 回 ST_WGT
- **Phase 6c**：`closed_loop_conv_requant_oc_tile_probe()` generator

### INT8 BATCH_MATMUL ZP correction（`5eaefcd`）

chain interface 從 32→64 bit 解決 INT8 BATCH_MATMUL 非零 ZP 問題：

```
chain_psum_data[31:0]  = psum (acc_out)
chain_psum_data[63:32] = sum_a = Σ a_q[k]  ← per-sample correction term
```

- `conv.v`：加 `sum_a` 累加
- `requant.v`：加 `chain_zp_b`（word[9][7:0]）、`use_act_correction`（word[3][17]）
- 計算：`corrected = psum - zp_B × sum_a`
- `bias_eff[n] = -zp_A×sum_b[n] + K×zp_A×zp_B`（compile-time，per-OC）

### BATCH_MATMUL → CONV lowering（`81acbe9`、`5eaefcd`）

`compile_model.py`：BATCH_MATMUL 全面 lower 到 CONV engine：

- FP16：永遠 lower；INT8：ZP=0 guard 已移除，靠 per-sample correction 處理任意 ZP
- 每個 batch/head slice → 一個 OP_FC_BMM 層（`fc(bmm)` label）
- qwen35 BMM：362K cy → 185K cy（−49%）；bmm_int8：366 → 140 cy（−62%）

### fc(bmm) label + profile 顯示（`19c3e55`）

- `OK_FC_BMM = 30`：新 op kind，hardware 同 OK_FC，profile 可區分
- `program_image.h` / `compile_model.py` / `mdla7_model_runner.cpp`（is_fc_kind() helper）/ `run_systemc.py`
- View-class TNPS（cy=0, bus=0）：engine 欄顯示 `—`，op 名保留（灰色）

### SHAPE → constant-load layer（`de48fb4`）

TFLite SHAPE op（輸入 tensor 的維度向量，compile-time 常數）從 `matrlz` 改為：
- `OK_SHAPE = 31`；wgt area 存 INT32 shape bytes
- runner：UDMA load wgt→L1，不寫回 DRAM（dram_w=0）
- Profile：`shape` label，engine=`—`

### qwen35 KV cache attention models + FP16 BMM reference fix（`7755b00`）

新增 generator `gen_qwen35_kvcache_tflite.py`：

```
Q [1,8,1,256] × K [1,8,N,256]^T → scores [1,8,1,N] → softmax
→ scores × V [1,8,N,256] → [1,8,1,256]
```

生成 `model/BMM/qwen35_kvcache_{128,512,1024}_fp16.tflite`，cycle 結果：

| KV 長度 | total cy | Q×Kᵀ/head | softmax | S×V/head |
|---|---|---|---|---|
| 128 | 25,299 | 1,559 | 59 | 1,546 |
| 512 | 88,534 | 5,440 | 285 | 5,478 |
| 1024 | 172,833 | 10,660 | 416 | 10,672 |

FP16 BMM reference bug fix：
- 舊：`np.matmul(a.float32, b.float32)` — BLAS 化，對大 N 可能有 1-ULP 差異
- 新：`conv_fp_ref(a.fp16→f32, b.T.fp16→f32, ...)` — 與 engine `compute_fp()` 相同的 sequential FP32 累加 + FP16 operands；也修了遺漏的 `.T` 轉置

---

## 目前整體狀態

### Regression

```
BMM (8 models):  SystemC 5/8 clean  (3 compile-skipped:1 = DEQUANTIZE in FP16 TFLite，無害)
                 Verilog DPI 8/8 PASS
```

`compile-skipped:1` 只是 FP16 conversion 產生的 DEQUANTIZE op 被 skip 的資訊，不影響 pass/fail。

### Descriptor bit 全覽（v12 完整版）

| 欄位 | word | bits | engine |
|---|---|---|---|
| `conv_chain_out_enable` | [3] | [15] | CONV |
| `requant_use_chain_input` | [3] | [12] | REQUANT |
| `requant_fp_mode` | [3] | [13] | REQUANT |
| `requant_param_load_mode` | [3] | [14] | REQUANT |
| `requant_use_act_correction` | [3] | [17] | REQUANT |
| `conv_tile_oc_count` | [31] | [31:16] | CONV |
| `requant_fp_bias` | [5] | [31:0] | REQUANT |
| `requant_param_l1_addr` | [6] | [21:0] | REQUANT |
| `requant_oc_count` | [7] | [15:0] | REQUANT |
| `requant_oc_index` | [7] | [31:16] | REQUANT |
| `requant_tile_drain_count` | [8] | [15:0] | REQUANT |
| `requant_chain_zp_b` | [9] | [7:0] | REQUANT |
| `chain_psum_data[31:0]` | chain | — | CONV→REQUANT psum |
| `chain_psum_data[63:32]` | chain | — | CONV→REQUANT sum_a |

---

## 接下來 next session 的建議工作

### 1. Phase 6：full tile traversal（最大 scope，multi-session）

目前 CONV/REQUANT 的 Phase 6b/6a 是 OC-only tile loop（每個 descriptor 處理多個 OC，相同 spatial position）。真正的 tile engine 需要：

- CONV：OH/OW/OC 三層 nested loop（每個 descriptor 掃 tile_h × tile_w × tile_oc 個輸出）
- REQUANT：對應的 drain loop
- Generator：為每個 layer 的完整 tile traversal 發 descriptor pair
- DRAM model multi-beat：`vf_dram_model` 目前每次只回 16 byte，weight load 要 burst

預估規模：~1000 行 RTL + ~500 行 Python。**分多個 sub-session 做。**

### 2. INT8 KV cache attention model

`qwen35_kvcache_{N}_fp16.tflite` 目前是 FP16。要加 INT8 版本：
- TFLite quantization 在 softmax 邊界有 shape inference 問題（見 generator 的 error）
- Fix：在 `gen_qwen35_kvcache_tflite.py` 改 Keras model 寫法避免 softmax→BMM shape mismatch

### 3. BATCH_MATMUL INT8 注意事項

目前 INT8 BMM lowering 正確的前提：`bias_eff[n]` 用 compile-time 的 B reference data 計算 `-zp_A × sum_b[n]`。B 是 dynamic activation，inference 時換 token 就會錯。

- ArcSim regression（固定 binary data）：✅ bit-exact
- Production chip driver：需要 dynamic `sum_b` correction（CONV 也送 sum_b 給 REQUANT）

如果要做 production path，需要把 `chain_psum_data` 再擴成 96-bit（加 `sum_b[n]` per OC）或用別的機制。

### 4. REVERSE_V2 lowering

目前 `REVERSE_V2` 仍然是 `matrlz`（flip along axes）。類似 SHAPE 的做法：
- Reverse 是 byte-movement，走 TNPS
- Compile-time 計算 flipped bytes → 存 wgt area → UDMA load 或 TNPS copy

### 5. Profile HTML 改進

- `fc(bmm)` 在 profile HTML 中顯示藍色粗體（已做）
- 可考慮加 KV length 標籤到 fc(bmm) 行（`fc(bmm, kv=1024)`）方便識別
- matrlz 裡還有 RESHAPE、RANDOM_STANDARD_NORMAL 可考慮類似 SHAPE 的處理

---

## 已知 follow-up（不在 immediate scope）

1. **DRAM multi-beat**：`vf_dram_model` 16 byte/request，weight load 慢；Phase 6 tile engine 需要 burst
2. **INT16 LUT engine**：SystemC 有 `run_unary_int16_lut`，Verilog 還沒（需要 64K-entry SRAM）
3. **SOFTMAX native Verilog**：目前走 generic ref-CRC，沒有真正 engine
4. **FP32 BATCH_MATMUL**：FP32 模型的 BMM 也用 FP16 路（compile 轉成 FP16 存），精度 OK 但概念上不純

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

# 生成新的 KV cache 模型
~/.venvs/mdla7/bin/python systemc/scripts/gen_qwen35_kvcache_tflite.py --kv-len 128 512 1024 --dtype fp16
```
