# MDLA7 Handoff

日期：2026-05-08
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## Top To-do

- [ ] GOAL：先保留目前 ETHZ_V6 fast baseline；利用 tiling / microblock fuse 更深層 Layer，讓 ETHZ_V6 pattern 盡量做到 DRAM R 只讀一次、DRAM W 只寫最後一層，且不同 Engine 可以 overlap。
- [ ] 驗證條件：先只跑 fast，不用跑 conflict / mesh；每次修改都要確認 functional PASS，且 fast cycles 比修改前更好。
- [ ] 從 ETHZ_V6 先挑原本 fast cycle 較小、容易快速驗證的模型開始，優先找高 DRAM R/W、線性 producer -> consumer、final boundary 明確的 chain。
- [ ] 檢查 profile / Gantt flow 顯示是否能看出 fused flow；fuse 不起來時 layer = flow。
- [ ] 決定是否 commit 目前 `systemc/src/test_model.cpp` 未提交修改。

## 目前狀態

- 工作樹目前有未提交修改：`systemc/src/test_model.cpp`。
- ETHZ_V6 baseline copy：`batch/output/baseline_ethz_v6_fast_20260508_220613`
- ETHZ_V6 after copy：`batch/output/ethz_v6_fast_after_conv_d2s_20260508_221412`
- `xlsr_quant` fast 已可作為目前比較點：functional PASS，cycles `1,729,167 -> 1,548,380`。
- 暫時不要把 conflict / mesh 當成驗證門檻；使用者目前指定只看 fast。

## 快速命令

```bash
make -C systemc -s
./batch/run_ethz_v6.py --list
./batch/run_ethz_v6.py --filter xlsr_quant --limit 1 --mode fast --rerun-all
./batch/run_ethz_v6.py --filter resnet_quant --limit 1 --mode fast --rerun-all
./batch/run_ethz_v6.py --filter mobilenet_v3_quant --limit 1 --mode fast --rerun-all
./batch/run_ethz_v6.py --filter mv3_depth_quant --limit 1 --mode fast --rerun-all
```

Profile entry：

```text
batch/profile_ethz_v6.html
batch/output/<stem>.fast.html
```

注意：Python entry point 從 repo root 用 `./batch/<runner>.py` 執行；產物都在 `batch/output/`。
