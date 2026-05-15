# MDLA7 Verilog vs CX 待補清單

日期：2026-05-14（session 2 結尾 handoff）
Repo：`/Volumes/4T_OFFICE/_Claude/MDLA7_Claude`
Branch：`main`
最新 commit：`2df2d16 Five improvements: profile labels, INT8 kvcache, REVERSE_V2 lowering, Phase 6c OW tile loop`

---

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

CONV/EWE/POOL microblock pipeline plan

### 動機

批量化後 11.18 ms / 21 ms (fast / sim)，但 F34 周邊測得**三引擎完全沒 overlap**：

| Layer              | cyc    | 累計 cum |
| ------------------ | ------ | -------- |
| L33 BMM (Q×Kᵀ)   | 7,425  | 7,425    |
| L34 sm_pmax (POOL) | 11,262 | 18,687   |
| L34 sm_sub (EWE)   | 5,631  | 24,318   |
| L34 sm_exp (EWE)   | 5,631  | 29,949   |
| L34 sm_psum (POOL) | 11,262 | 41,211   |
| L34 sm_div (EWE)   | 5,631  | 46,842   |
| L36 BMM (S×V)     | 8,629  | 55,473   |

每 attention block 55.5 K cyc。引擎用量分布：POOL 22.5K（兩個 reduce）/ EWE 16.9K (sub+exp+div) / CONV+Requant 16.1K。完美 overlap → critical path = **max(22.5K) ≈ 40% of 55.5K**。

對 442 個 attention block 推估：sim 21 ms × 0.4 ≈ **~8 ms**，fast latency 11 ms → **~4–5 ms**，接近原 baseline 7 ms。

### 為什麼現在沒 overlap

1. **每層 1 個 descriptor 整批**（剛批量化的副作用）— 後一引擎要等前一引擎整批做完
2. **Softmax 內 5 ops 強制序列**（chain wait_tag）— 結構性無法 overlap 同 softmax 內
3. **Scratches `addr_max/ctr/exp/sum` 跨 softmax 共用一塊** — 不同 softmax 也不能 overlap（早 Session 4 的 Phase E 分析）

### 真正可以 overlap 的方向

**不是**同一個 softmax 內的 5 ops（dep chain 寫死），**是**：

- **Softmax N 的 POOL_SUM** ↔ **Softmax N+1 的 POOL_MAX**（同 POOL engine，scratch 要分）
- **Softmax 的 EWE 階段** ↔ **下一個 BMM 的 CONV/Requant**（不同 engine，要 microblock 切細）
- **BMM 的 CONV 出 row 0..k** ↔ **Softmax 的 POOL_MAX 處理 row 0..k**（cross-layer row streaming，要 BMM 也 tile）

→ 結構是 **跨 attention block 的 row-level pipeline**，不是 softmax 內部 overlap。

### 階段化 Plan

| Phase                                                    | 內容                                                                                                                                                                                                       | 預期 latency 改善                      | 風險                                                            | session 數 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------- | ---------- |
| **P1: Softmax 間 scratch ping-pong**               | `addr_max[2]/ctr[2]/exp[2]/sum[2]` 雙緩衝；softmax N+1 的 POOL_MAX 只 wait softmax N-1 而非 N                                                                                                            | 中（POOL 兩 op 不再卡同 softmax 連續） | 低 — 純 compiler-side，~80 行                                  | 1          |
| **P2: 同 softmax 內 dequant/quant 平行 BMM**       | INT8 chain 的 dequant (entry) 跟前一個 BMM (Requant tail) overlap；quant (exit) 跟下一個 BMM (CONV head) overlap；靠調整 wait_tag dep 來鬆綁                                                               | 小–中（dequant/quant ~2K cyc × 442） | 低 — 改 tag 不改結構                                           | 1          |
| **P3: BMM tile-by-row + softmax microblock chain** | BMM L33/L36 改成 M-row microblock（M=16 試）；softmax chain 也切成 M-row microblock，每 microblock 7 ops；scratch 開 2×M-row ping-pong；BMM 第 k 個 microblock done → softmax chain 第 k microblock 啟動 | 大（target 4-5 ms）                    | 中–高 — 動 BMM tile loop + scratch allocator + L1 budget 再驗 | 2          |
| **P4: 三引擎 steady-state 驗證**                   | 跑 profile + L1Mesh，看 timeline 是否真的三層疊；找剩餘 stall 點（DRAM bandwidth? sslice? udma_w?）                                                                                                        | 確認/微調                              | 低                                                              | 0.5        |

### 起點建議：P1（softmax 間 ping-pong）

最小可驗、改動隔離。改 [mdla7_model_runner.cpp:9573-9582](systemc/src/mdla7_model_runner.cpp#L9573-L9582) 的 scratch 分配 + 跨 softmax 的 wait_tag 路徑（layer-loop 那層的 `prev_*_done` 追蹤）。

具體：

- 把 `addr_max/addr_sum` 改成 per-layer pair `{addr_max[0], addr_max[1]}`
- 在 model_runner 的 layer 迴圈外維持 `softmax_scratch_slot` (mod 2) 跟 `prev2_softmax_done_tag`
- 每個 softmax chain 用 `softmax_scratch_slot` 對應的 buffer，wait 改 `prev2_softmax_done_tag`（兩個 softmax 前的）
- L1 用量加倍：`fp_buf_b × 4 + scalar_b × 4`（INT8 chain），對 typical (rows=64, K=2048) = 2 MB scratch，跟 L1_BUDGET 一樣 — **不夠**！

→ **P1 需先解 L1 fit**：要嘛縮 microblock（< 64 rows）配 ping-pong，要嘛先做 P3 把 microblock 切細再 ping-pong。

→ 結論：**P1 跟 P3 必須一起做**，因為 ping-pong 在 whole-shape (64 rows × 2048 K) 上 L1 裝不下。建議直接走 P3 + 內含 ping-pong。

### 修訂後的真正起點：P3（BMM tile + softmax microblock + 2-slot ping-pong）

選 M（microblock rows 大小）讓 L1 裝得下 2 個 slot 的所有 scratch：

對 (rows=64, K=2048, INT8)：

- 每 slot 需要 `ctr(M×K×2) + exp(M×K×2) + fp_in(M×K×2) + fp_out(M×K×2) + max(M×2) + sum(M×2)` ≈ `M × K × 8 + M × 4`
- 2 slots: `M × K × 16 + M × 8`
- L1_BUDGET = 2 MB = 2,097,152 bytes
- 留 ~512 KB 給 input/output → ~1.5 MB 給 scratch
- M ≤ 1,572,864 / (2048 × 16) ≈ 48 rows

M=16 安全（每 slot ~512 KB scratch），同時切成 4 個 microblock per softmax，足夠跨層 overlap。

要不要走這條？要的話下一步：先 P3 草圖 + L1 fit 自動 sizing helper。

### P3 執行步驟（待 user 確認啟動）

1. **L1 fit auto-sizing helper**

   - 輸入 `(rows, K, dtype, fuse_eligible, in_size, ref_size)` → 輸出 `(M, num_microblocks)`
   - 預設選 `M = max{2^k : 2 slots 全 scratch + in + out ≤ L1_BUDGET}`，clamp 到 `M ≤ rows`
   - 對 (rows=64, K=2048, INT8) 應該選出 M=16
   - 對小 model（rows=8, K=128）會選出 M=rows（單 microblock，退化為當前批量化）
2. **改 [mdla7_model_runner.cpp:9519-9694](systemc/src/mdla7_model_runner.cpp#L9519-L9694) — softmax chain microblock loop**

   - 外層 `for (mb = 0; mb < num_microblocks; ++mb)`
   - Scratch 分配雙倍：`addr_ctr[2] / addr_exp[2] / addr_max[2] / addr_sum[2] / addr_fp_in[2] / addr_fp_out[2]`
   - 每 microblock 用 `slot = mb % 2` 對應 scratch
   - Microblock k 的 7 ops 內部仍鏈式（dep on prev op），第 1 op (dequant 或 POOL_MAX) 改 wait 於 microblock k-2 的最後 op（同 slot 之前的釋放）
   - mb=0,1 的 first-op wait 於 fuse_prev_done_tag / load_done_wait（同今）
   - Stream 標記：每 microblock = 1 個 `Microblock` 結構，`mb.id = k`, `mb.rows = M`, `mb.elem_off = k*M*K`
3. **BMM L33/L36 row-tile 能力盤點**（**open question，先盤再動**）

   - 看 [mdla7_model_runner.cpp](systemc/src/mdla7_model_runner.cpp) `OK_CONV` / `OK_BMM` 的 descriptor 發送，搜尋既有 `tile_h` / `oh_tile` / `tiles_h_per_layer` 對 BMM 是否生效
   - Phase 6c/6d (CONV OW/OH spatial tile) 已存在（commit `9768589`），但 BMM 走 conv path 是否會用到？需確認
   - 如有：直接 reuse；如無：要新增 BMM row-tile 路徑（風險與 session 數會多 1）
4. **驗證 timeline overlap**

   - 跑 `./batch/run_systemc.py --filter bmm --model-filter tiled_2.5ms_int8 --rerun-all`
   - 看 `cycles_cum` delta：若 microblock k+1 的 cum 比 microblock k 少（i.e., 沒等 microblock k 完全結束），表示 overlap 成功
   - 看 [L1Mesh_report.md](L1Mesh_report.md)：POOL/EWE 同時間區間的 busy 是否真的重疊
   - 預期 fast latency: 11 ms → ~5 ms

### 待 user 回答的開放問題

- **要不要走 P3？** 估 2 sessions（含步驟 3 的 BMM 盤點 + 可能新增）
- **是否保留批量化 (current code) 作 fallback？** M=rows 自動退化已涵蓋，可不留環境變數
- **要不要先做步驟 3 的 BMM 盤點再回頭決策？** 純讀，0.5 session 內可結束
