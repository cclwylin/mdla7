# MDLA7 Verilog vs CX 待補清單

日期：2026-05-14（更新）
Repo：`/Volumes/4T_OFFICE/_Claude/MDLA7_Claude`
Branch：`main`
最新已提交 commit：`fd13aab Update Verilog CX handoff todo`

## 自上次 commit 以來的成果（未 commit）

### 1. EWE LUT engine 完整實作（SystemC + RTL，已通過 BMM 5/5 PASS）

| 元件 | 變更 |
|---|---|
| `systemc/include/mdla7/program_image.h` | 新增 `OK_RSQRT=28`, `OK_TANH=29` |
| `systemc/include/mdla7/descriptor.h` | 新增 `ES_RSQRT=7`, `ES_TANH=8` |
| `systemc/scripts/compile_model.py` | INT8 + INT16 RSQRT/TANH/LOGISTIC/HARD_SWISH/GELU 全部 lower 成 LUT in `wgt_b`（INT8 256 byte、INT16 128 KB）|
| `systemc/include/mdla7/ewe_pool.h` | `run_unary_int8_lut()` + `run_unary_int16_lut()` byte-indexed lookup |
| `systemc/src/mdla7_model_runner.cpp` | `make_ewe_unary` subtype 支援 RSQRT/TANH，所有 unary 分類 lambda 都加進來 |
| `rtl/verilog/ewe.v` | 真實 `lut_mem[0:255]` register array、`ST_LUT_LOAD` state、16-beat L1 read FSM、`compute_lane` 在 `lut_mode` 下做 `lut_mem[av[7:0]]` 索引查表 |
| `rtl/verilog/host.v` + `mdla7_top.v` + `Testbench_host_program.v` | `ewe_lut_mode` (word[12][12])、`ewe_lut_addr` (word[30]) 接線 |
| `batch/gen_verilog_program.py` | `int8_lut_unary_descriptor` + `closed_loop_lut_unary_probes`：先發 16 個 16-byte UDMA-load-LUT，再發 EWE descriptor with `lut_addr` |

驗證：BMM verilog DPI 3/3 PASS，`perf_ewe = 7180 cyc`（real LUT load + lookup work）。

**重要踩雷**：`vf_dram_model` 一次只回 16 byte（不管 `req_bytes` 多大），所以 256-byte LUT 必須拆 16 個 16-byte UDMA。Phase 6 改 tile-engine 時要解決這個 DRAM-model bottleneck。

### 2. qwen35_attention bug fix

| 修正 | 細節 |
|---|---|
| FP BATCH_MATMUL `np.matmul` shape mismatch | qwen35 attention 的 `(1,8,1,256) × (1,8,1,1)` 需 broadcast K=1→256；compile_model.py 加偵測 |
| 新增 `SHAPE` / `REVERSE_V2` / `RANDOM_STANDARD_NORMAL` materialize fallback | qwen35 graph 有這 3 個 op，全進 `MATERIALIZED_FALLBACK_OPS` + `shape_changing_materialized` set，加 reference compute（SHAPE pad 到 out_hwc / REVERSE_V2 用 np.flip / RANDOM 固定 seed `0xC0DEBEEF`）|

驗證：BMM regression 5/5 PASS（SystemC + Verilog），qwen35 兩個都 PASS。

### 3. memory.md 規劃文字修正

明示「**CONV/DWCONV/FC 永遠不直接寫 L1**，所有 dtype 都過 REQUANT」這個 invariant。REQUANT row 描述拆 INT mode / FP mode 兩段 datapath，匯流排表把 W lane 共用講清楚。

### 4. Per-layer profile 報表格式調整

| 變更 | 細節 |
|---|---|
| 新欄位 `engine` | per-layer 表第 4 欄、跟 op→engine 對照（`conv+requant` / `ewe` / `pool` / `tnps` / matrlz=`—`）|
| 欄序重排 | `iH..tiles` 12 欄移到 `verify` 之前 |
| matrlz 顯示 | `matrlz(原TFLite_op)` 紅字粗體，例：`matrlz(BATCH_MATMUL)` |
| TNPS-class fold tag | `reshape(view)` 灰、`reshape(folded)` 橘、`reshape(copy)` 黑 |
| TNPS ideal model | reshape/trnps/d2spac/concat 算 `bytes/128 + setup` (TNPS architectural target) |
| matrlz ideal model | 也按 TNPS self-copy 估 |

## 進行中且 BROKEN 的工作

### CONV→REQUANT chain 重構（Phase 1，BUILD FAIL）

要把 Verilog CONV/REQUANT 從「sample bring-up + CONV 自己寫 L1」拉到「CONV → chain → REQUANT → L1」這個 production 規劃。Phase 1 寫到一半時 `vf_requant_sample_engine` port 列出現重複 `fp_mode`、`mdla7_top.v` 1353 行 binding 也有重複，目前 verilator build 失敗。

#### 已寫進去的東西

| 檔 | 變更 | 狀態 |
|---|---|---|
| `rtl/verilog/conv.v` | 加 `chain_psum_valid` / `chain_psum_data[31:0]` / `chain_psum_ready` ports；ST_STORE→ST_DONE 時 latch acc 並 pulse valid（INT path 用 `acc_out`、INT16 用 `int16_acc_out`、FP path 用 `fp_sum_bits[31:0]` bit-cast）| ✓ |
| `rtl/verilog/requant.v` | 加 `use_chain_input` / `chain_psum_valid` / `chain_psum_data` / `chain_psum_ready` ports；新 `ST_CHAIN_WAIT` state；IDLE 在 `use_chain_input=1` 時轉 ST_CHAIN_WAIT；state 寬度從 3 擴到 4 bit | ✓ |
| `rtl/verilog/requant.v` | linter 多加了 `fp_mode` / `fp_bias` / `fp32_lt()` / FP32→FP16 cast helper（Phase 2 work），但 **`fp_mode` 被宣告兩次**（line 30 + line 37）| ✗ DUPLICATE |
| `rtl/verilog/host.v` | `requant_use_chain_input` 解碼自 word[3] bit 12 + reset | ✓ |
| `rtl/verilog/mdla7_top.v` | `requant_use_chain_input_q` 暫存 + 接到 REQUANT instance；新 `conv_chain_psum_valid/data/ready` wire 連 CONV→REQUANT；CONV instance 已加 chain pin binding | ✓ 但 1353 行 REQUANT instance 有 **重複 fp_mode pin** |
| `rtl/verilog/Testbench_host_program.v` | `requant_use_chain_input` wire + 兩個 `host`/`mdla7_top` instance 接線 | ✓ |

#### 修復步驟

1. `rtl/verilog/requant.v`：刪除 line 30（或 line 37）的重複 `input fp_mode,` 宣告
2. `rtl/verilog/mdla7_top.v`：line 1353 REQUANT instance 找重複的 `.fp_mode(...)`，刪一個
3. `rtl/verilog/host.v` + `mdla7_top.v` + `Testbench_host_program.v`：補 `requant_fp_mode` 和 `requant_fp_bias` 接線（descriptor 解碼建議用 word[3] bit 13、word[14] full word）
4. `rm -rf rtl/obj/verilog/host && ./batch/run_verilog.py --filter bmm --rerun-all --dpi`，預期 5/5 PASS
5. （可選）跑 `MDLA7_DEBUG_EWE_LUT=1 ./batch/run_verilog.py --filter bmm_softmax_bmm_sam` 確認 EWE LUT path 仍正常

#### Phase 1 還沒做完的剩餘工作

- 確認 chain wire 的 sample-mode 行為：`use_chain_input=0` 時 `chain_psum_ready=0`，CONV 的 chain pulse 應該是無害的（latch 之後在 IDLE 清掉）。需要在 BMM regression PASS 之後加一個 trace 驗證 chain 真的 idle
- `host.v` 對 `ewe_lut_mode` 等其他 word[3] bit 用法要做衝突檢查；目前我用 word[3][12] 給 `requant_use_chain_input`，沒跟其他衝到（已驗）

## 接下來 6 phase 的整體計劃

### Phase 1（IN PROGRESS）：CONV↔REQUANT chain infrastructure

見上面「進行中且 BROKEN」段。修好就算 Phase 1 完成。

**收尾條件**：BMM 5/5 PASS、`use_chain_input=1` 路徑可走（雖然 generator 還沒用，但 wire path 通）。

### Phase 2：REQUANT FP mode

linter 已經把 port + helper function 寫進 `requant.v`：`fp_mode` / `fp_bias` / `fp32_lt()` / FP32→FP16 cast。剩下要做的：

1. 在 `always @*` 加 fp_mode 分支（mirror SystemC `requant_engine.h:174-213` FP path）：
   - `quantized_fp = bit_cast(active_input_value) + fp_bias`
   - `clamped_fp = clamp(quantized_fp, act_min_fp, act_max_fp)`（act_min/act_max 重新解釋為 fp32 bit pattern）
   - `out_fp16 = fp32_to_fp16(clamped_fp)`
2. ST_STORE 在 fp_mode 下寫 2 byte（FP16 lane）而不是 1 byte（INT8 lane）
3. host.v / mdla7_top.v 把 `requant_fp_mode` / `requant_fp_bias` 接線（見上面「修復步驟 3」）
4. 驗證：synthetic FP CONV layer 跑 chain mode，比對 SystemC FP requant 結果 bit-exact

### Phase 3：REQUANT per-OC params table

目前 sample REQUANT 一次接 1 組 mult/shift/zp。tile mode 需要：

1. 加 `param_l1_addr` port + `oc_count` port
2. 新 state `ST_PARAM_LOAD_OC`：從 L1 連續讀 `oc_count` 組 (mult, shift, bias_eff)，存進 `param_mem[]` regs 或 SRAM
3. PIPE 階段按 oc index 從 `param_mem[oc % CHAIN_LANES]` 取 mult/shift/bias_eff
4. SystemC `requant_engine.h:245-256` 已有對應 layout（`scale_lut_addr` 起點，後接 `[zp_out|act_min|act_max | mult[OC] | shift[OC] | bias_eff[OC]]`），照搬

### Phase 4：Generator emit chain-mode descriptor pair

選一個最小 BMM CONV 層（例如 BMM SAM L24 STRIDED_SLICE 後面的小 CONV），發：

1. UDMA load act + wgt 進 L1
2. UDMA load REQUANT params 進 L1
3. CONV descriptor with `chain_out_enable`（不寫 L1）
4. REQUANT descriptor with `use_chain_input=1` + `param_l1_addr`
5. UDMA store result + L1CRC verify

驗證：BMM 5/5 仍 PASS，新加的 chain CONV 那層的 `perf_conv` + `perf_requant` 都有 cycles。

### Phase 5：CONV 拔 L1 store production path

目前 `vf_conv_sample_engine` 有 `ST_STORE`（寫 L1）。Production 規劃下這條 path 不該存在（per memory.md）。但 sample/debug 還想留。

做法：

1. 加 `chain_out_enable` port — generator 設 1 時 CONV 跳過 ST_STORE，直接 ST_DONE，只 push chain
2. FP path 同步改：`chain_psum_data` 用 fp32 bit-cast（目前 `fp_sum_bits[31:0]` 取低 32 位，可能不對；要確認 `fp_sum_bits` 的 layout 是 fp32 在 [31:0] 還是 fp64。看 conv.v line 930 `$realtobits(fp_sum)` 是 fp64 → 取 [31:0] 不對，需要先 `$shortrealtobits` 或手動 cast）
3. 移除 `vf_conv_int8_mac` 內建的 `mbqm + zp + clamp`（這是不該在 CONV 做的事）— 但這會破壞 sample mode 的 out_q 輸出，需要慎重；可能要保留兩個 primitive：`vf_conv_int8_mac_legacy`（含 requant，sample 用）vs `vf_conv_int8_mac_pure`（純 MAC，chain 用）

### Phase 6：Sample-engine → tile-engine

最大改動。整體 CONV/REQUANT 從「一次處理 1 sample」改成「一次掃整個 tile / layer」。涉及：

- 描述子格式重新設計（tile shape、stride、padding 等已在 sample 描述子裡，但 traversal loop 要從 testbench/host 移到 engine 內部）
- CONV 內部加 OH/OW/OC 三層 nested loop，每個 sample 推一筆 chain
- REQUANT 內部加同樣的 NHWC drain loop
- Generator 改成「一個 tile 一個 CONV+REQUANT descriptor pair」而非「一個 sample 一個」
- BMM regression 的 perf_conv / perf_requant cycles 才會跟 SystemC `conv_cycles()` 對齊
- DRAM-model 16-byte/req 限制要修（multi-beat read 或 burst 模式），否則 weight load 會超慢

預估規模：~1500 行 RTL changes、~500 行 Python generator changes、testbench 補 driver。**這是多 session 工作，不要嘗試一次完成**。

## 已知 follow-up（不在這次 chain 重構直接 scope 內）

1. **DRAM model multi-beat**：`vf_dram_model` 一次只回 16 byte。Phase 6 weight load 要解決（要嘛 model 升級成 burst，要嘛 generator 用 16-beat UDMA chain）
2. **INT16 LUT engine RTL path**：SystemC 有 `run_unary_int16_lut`，verilog 還沒實作（需要 64K-entry SRAM + 16-bit indexed lookup）。等 INT16 unary model 出現再做
3. **qwen35 7 個 materialize fallback** 算 deterministic reference，但 reference bytes 跟 TFLite 比可能不 bit-exact（特別 RANDOM_STANDARD_NORMAL）；如果未來要用 qwen35 做 perf comparison 要重新校準
4. **`bring-up sample CONV L1 write`**：memory.md 已標 production datapath 不該有，但目前還在用。Phase 5 才會處理

---

# 原 P0–P6 計劃（仍部分有效）

下方原本的 P0–P6 是更早期（CONV/REQUANT 還沒進行重構時）的待補清單。Phase 1–6 chain 重構完成後，原 P0–P3 的「regression coverage / generator full tile / RTL full traversal」會自然收斂。原 P4（final answer verification）和 P5（performance correlation）仍是獨立工作。

## P0：報表與 regression 語意修正（仍有效）

1. `run_verilog.py` 必須偵測 `.bin` 內有 compute layers，但 generated descriptor 沒有 compute op 的情況。

   - 條件：model layers 包含 CONV/EWE/POOL/TNPS/REQUANT/SOFTMAX 等 compute op。
   - 但 generated program 的 `conv/ewe/pool/tnps/requant == 0`。
   - 結果不可顯示成一般 PASS performance row。
   - 應標成 `NO-COMPUTE`、`MATERIALIZED-CHECK` 或直接 FAIL，擇一固定 policy。

2. HTML per-model report 要清楚標出 descriptor coverage。

   - 顯示 compute descriptor counts。
   - 顯示 materialized/fallback descriptor counts。
   - 顯示 unsupported / not-lowered layer list。
   - 如果 engine counters 全 0，要在頁面上明確寫出原因。

3. profile index 的 ratio 欄位要排除 materialized-only rows。

   - materialized-only 不可計入 `verilog/cx`。
   - `verilog_total_cyc` 只統計真正有 compute descriptor 的 row。

## P1：Compiler / `.bin` metadata 要補齊（仍有效）

1. `.bin` 需要讓 Verilog 能知道每一層是：

   - lowered to hardware engine
   - fused into previous/next layer
   - materialized fallback
   - unsupported

2. 對 fused layer，要能追蹤 final answer 比對來源。

   - 如果 CX fuse 多層，Verilog 要知道 fused group 的 output tensor。
   - report 要列出 fused group：起始 layer、結束 layer、output tensor offset/size。

3. unsupported op 不可靜默 materialize。

   - `SOFTMAX`、複雜 `TRANSPOSE`、`RESHAPE`、`GATHER/MATERIALIZE` 等要有明確 status。
   - regression 要能區分：
     - supported and executed in Verilog
     - unsupported but checked through materialized output
     - unsupported and unchecked

## P2：Generator 要從 sample/check 改成 full layer/tile descriptor（被 Phase 4-6 取代）

詳見 Phase 4-6。原文保留：

1. `batch/gen_verilog_program.py` 要為 supported layers 產生完整 tile traversal。

   每個 layer 應是：

   ```text
   DRAM input/weight -> UDMA -> L1
   L1 -> engine compute tile
   engine output -> L1
   L1 -> UDMA -> DRAM output
   final/check CRC
   ```

2. BMM SAM model 先補這些 op 的 descriptor：

   - INT8 CONV full tile descriptor
   - INT8 EWE/MUL/SUB full tile descriptor
   - TNPS / TRANSPOSE descriptor
   - RESHAPE / S2D / D2S descriptor
   - SOFTMAX descriptor or explicit unsupported status

3. Descriptor 不可只挑 sample。

   - 目前 sample/probe 只能用於 bring-up。
   - regression PASS 必須跑足夠 tile 覆蓋，最後答案要和 golden 對上。

4. 對每個 layer 產生 descriptor summary。

   - layer id
   - op type
   - tile count
   - read bytes
   - write bytes
   - expected output bytes
   - check bytes

## P3：RTL engine 要補完整 compute path（被 Phase 1-6 取代）

1. CONV — 補完整 OH/OW/OC/KH/KW/IC traversal，輸出走 chain 不寫 L1（見 Phase 5-6）
2. REQUANT — 接 CONV chain，full tile（見 Phase 1-3）
3. EWE / MUL / SUB — 完整 tensor tile（部分已隨 LUT engine 完成；INT8 ADD/MUL/SUB 還是 sample）
4. POOL — 完整 pooling window traversal（未動）
5. TNPS / TRANSPOSE / RESHAPE — 已支援 BMM SAM 大部分，仍 sample-level
6. SOFTMAX — 仍只在 SystemC 跑，Verilog 沒原生支援，generator 走 generic ref-CRC

## P4：Final answer verification 要補齊（仍有效）

1. 每個 model 必須有最後 output tensor check。

   - 最終 output CRC 或 byte compare。
   - 不是只 check 中間 materialized layer。

2. fused group 要比對 fused group output。

   - 如果多層 fuse，Verilog 不一定逐層 check。
   - 但 fused output 必須和 CX/fast golden 對上。

3. layer check 與 final check 要分開顯示。

   - layer partial check 不能讓 final answer 變 PASS。
   - final output 沒 check 到，答案不可顯示 PASS。

## P5：Performance correlation（仍有效）

1. `verilog_cyc` 只使用 RTL `perf_total`。

   - 不恢復 `--cycle`。
   - 不使用 host stopwatch。
   - 不新增 host synthetic cycle command。

2. Engine utilization 只使用 RTL counters。

   - `perf_conv` / `perf_requant` / `perf_ewe` / `perf_pool` / `perf_tnps` / `perf_udma_r` / `perf_udma_w`

3. 只有 full compute descriptor row 可以和 CX 算 ratio。

   - materialized-only row 不計入。
   - sample/probe row 不計入。
   - unsupported unchecked row 直接 FAIL。

4. 目標：Synth/CX vs Verilog correlation 誤差小於 10%。

## P6：建議實作順序（被 Phase 1-6 取代，原文保留參考）

1. 在 `run_verilog.py` 加 descriptor coverage classification
2. 在 `gen_verilog_program.py` 對 BMM SAM supported ops 發 compute descriptors（先 INT8 EWE/TNPS/RESHAPE，再 CONV full tile）
3. 補 RTL EWE/TNPS/RESHAPE full tile traversal
4. 補 RTL CONV full tile traversal
5. 補 final output verification
6. 跑 BMM 全集：`./batch/run_verilog.py --filter bmm --rerun-all --dpi`

---

# 立即下一步（next session 接手用）

1. 把 `requant.v` line 30/37 重複 `fp_mode` 砍一個
2. 把 `mdla7_top.v` line 1353 附近重複 `.fp_mode(...)` binding 砍一個
3. 把 `requant_fp_mode` / `requant_fp_bias` 補到 host.v / mdla7_top.v / Testbench_host_program.v 三邊
4. `rm -rf rtl/obj/verilog/host && ./batch/run_verilog.py --filter bmm --rerun-all --dpi`
5. 預期：5/5 PASS（chain wire 在 use_chain_input=0 時應為 no-op）
6. 完成 Phase 1，進 Phase 2（FP mode functional path），逐步推 3→6
