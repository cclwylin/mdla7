# MDLA7 Verilog vs CX 待補清單

日期：2026-05-14
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`
最新已提交 commit：`3739a9c Add Verilog HW performance counters`

## 目標

Verilog version 要追上 CX version：同一個 compiled `.bin`，Verilog 必須真的跑硬體資料路徑並驗證最後答案，cycle 必須來自 RTL hardware performance counter，不接受 host synthetic command 或 materialized-only check 當 performance PASS。

## 現況判斷

`bmm_softmax_bmm_sam_quant_L22_L61.html` 顯示 `perf_udma_*` 有值，但 `perf_conv/requant/ewe/pool/tnps` 全 0。

原因不是 RTL counter 壞掉，而是 Verilog generator 目前對這個 model 只發出 materialized fallback / UDMA / CRC 類 descriptor，沒有發出真正的 CONV/EWE/TNPS/SOFTMAX compute descriptor。這種 PASS 只能算 coverage/materialized check，不能算 Verilog full compute PASS，也不能拿來和 CX 做 performance correlation。

## P0：先把報表與 regression 語意修正

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

## P1：Compiler / `.bin` metadata 要補齊

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

## P2：Generator 要從 sample/check 改成 full layer/tile descriptor

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

## P3：RTL engine 要補完整 compute path

1. CONV

   - 補完整 OH/OW/OC/KH/KW/IC traversal。
   - 支援 INT8 full output tile。
   - 支援 accumulation / psum / requant handoff。
   - output 必須寫回 L1，而不是只跑 sample。

2. REQUANT

   - 支援完整 INT32/psum 到 INT8 output tile。
   - 要能接 CONV chain 或從 L1 讀 accumulator。
   - output writeback 到 L1。

3. EWE / MUL / SUB

   - 支援完整 tensor tile。
   - INT8/INT16/FP16 policy 要固定。
   - output writeback 到 L1。

4. POOL

   - 支援完整 pooling window traversal。
   - max/avg mode 要和 CX 對齊。

5. TNPS / TRANSPOSE / RESHAPE

   - 支援 BMM SAM 需要的 layout movement。
   - 不能用 materialized output 代替。

6. SOFTMAX

   - 決定硬體支援方案。
   - 若短期不做硬體 SOFTMAX，要在 compiler/report 明確標 unsupported，且不能算 full Verilog compute PASS。

## P4：Final answer verification 要補齊

1. 每個 model 必須有最後 output tensor check。

   - 最終 output CRC 或 byte compare。
   - 不是只 check 中間 materialized layer。

2. fused group 要比對 fused group output。

   - 如果多層 fuse，Verilog 不一定逐層 check。
   - 但 fused output 必須和 CX/fast golden 對上。

3. layer check 與 final check 要分開顯示。

   - layer partial check 不能讓 final answer 變 PASS。
   - final output 沒 check 到，答案不可顯示 PASS。

## P5：Performance correlation

1. `verilog_cyc` 只使用 RTL `perf_total`。

   - 不恢復 `--cycle`。
   - 不使用 host stopwatch。
   - 不新增 host synthetic cycle command。

2. Engine utilization 只使用 RTL counters。

   - `perf_conv`
   - `perf_requant`
   - `perf_ewe`
   - `perf_pool`
   - `perf_tnps`
   - `perf_udma_r`
   - `perf_udma_w`

3. 只有 full compute descriptor row 可以和 CX 算 ratio。

   - materialized-only row 不計入。
   - sample/probe row 不計入。
   - unsupported unchecked row 直接 FAIL。

4. 目標：Synth/CX vs Verilog correlation 誤差小於 10%。

## P6：建議實作順序

1. 在 `run_verilog.py` 加 descriptor coverage classification。

   - 先讓 `bmm_softmax_bmm_sam_quant_L22_L61` 顯示 `MATERIALIZED-CHECK` 或 FAIL。
   - 不再讓 engine counter 全 0 的 row 看起來像正常 PASS。

2. 在 `gen_verilog_program.py` 對 BMM SAM supported ops 發 compute descriptors。

   - 先從 INT8 EWE/MUL/SUB 和 TNPS/RESHAPE 做起。
   - 再補 CONV full tile。

3. 補 RTL EWE/TNPS/RESHAPE full tile traversal。

   - 讓 BMM SAM 的非 CONV layout/data movement 先跑起來。

4. 補 RTL CONV full tile traversal。

   - 讓 `perf_conv` 在 BMM SAM 真正動起來。

5. 補 final output verification。

   - final tensor 沒 check 到不可 PASS。

6. 跑 BMM 全集：

   ```bash
   ./batch/run_verilog.py --filter bmm --rerun-all --dpi
   ```

   目標：

   - supported rows：PASS full compute
   - unsupported rows：明確列出 unsupported op
   - materialized-only：不可算 performance PASS
