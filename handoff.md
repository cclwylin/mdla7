# MDLA7 Handoff

日期時間：2026-05-12 CST
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- 最新 handoff commit：`ab51879 Update handoff for verilog final streaming`。
- 最新 verilog_final RTL commit：`e7e9c1e Tag L1 responses with source and stream tid`。
- 此 repo 工作樹仍有其他既有未提交變更；本 handoff 只描述目前
  verilog_final streaming / L1 response bring-up 狀態。
- `rtl/bin`、`rtl/obj*`、`rtl/verilator` 仍視為本機產物，不應混入 commit。

## Verilog Final Streaming / L1 Response

最新相關 commits：

```text
e7e9c1e Tag L1 responses with source and stream tid
3305dde Probe requant L1 producer path
1f7aa93 Feed UDMA store CRC from L1 response
a26c5ad Check microblock output SRAM CRC
d544211 Add microblock final tensor CRC probes
1627ffd Drive verilog final with microblock descriptors
```

### 現有能力

- `run_verilog_final.py` default 已走 microblock descriptors；
  `--sample-descriptors` 才回到舊 sample descriptor path。
- generator 會在 microblock sequence 後加 tensor coverage probes：
  - UDMA seed L1 bytes -> UDMA STORE from L1 response -> output SRAM CRC。
  - REQUANT producer 寫 L1 -> L1CRC 驗 `[requant_out] + zeros`。
  - UDMA ref-fill output SRAM image -> full output SRAM CRC / final CRC。
- UDMA STORE path 已可從 L1Mesh read response 寫入 UDMA output SRAM。
  `vf_udma_engine` 會在 `final_write_mode && direction_write` 等 L1 read
  response，並把 16B line 寫到 `output_sram[out_byte_offset + lane]`。
- UDMA SRAMCRC descriptor 不再打一筆 L1 request；它只掃 UDMA output SRAM。
- L1Manager/L1Mesh response metadata 已打通：
  - L1Manager 將 arbitration 選到的 `source/tid` 帶到 mesh request。
  - L1Mesh response 回傳 `resp_read/resp_source/resp_tid`。
  - `mdla7_top_final` 只把符合當前 engine source 且 `tid == stream_slot_q`
    的 read response 餵給 REQUANT/POOL/EWE/UDMA。

### 已知風險

- L1Mesh response/data 在 back-to-back request 時仍可能出現 stale beat 或
  metadata-data 同拍風險。
- `resp_source/resp_tid` 是必要地基，但 REQUANT -> UDMA consumer probe
  還需要 per-command response queue / skid register，讓
  `resp_valid/read/source/tid/rdata` 同拍鎖住後再接回。
- 改 `rtl/synth/*.v` 後，Verilator include dependency 可能不會自動重建；
  建議先清掉 host obj：

```bash
rm -rf rtl/obj/verilog_final/host
```

## 最近驗證

已驗證 PASS：

```bash
rm -rf rtl/obj/verilog_final/host
./rtl/batch/run_verilog_final.py --filter slice --limit 1 \
  --rerun-all --require-crc-coverage --require-final-output-crc

./rtl/batch/run_verilog_final.py --filter slice --limit 3 \
  --rerun-all --no-build --require-crc-coverage --require-final-output-crc
```

最近結果：

```text
limit 1: pass=1 fail=0 skip=0 sample_only=0 total=1
coverage: refcrc=0 sramcrc=3 finalcrc=2 refB=0 sramB=16777248 finalB=16777232

limit 3: pass=3 fail=0 skip=0 sample_only=0 total=3
coverage: refcrc=0 sramcrc=12 finalcrc=7 refB=0 sramB=23069440 finalB=23069360
```

## 下一步

1. 在 L1 response side 加 per-command response queue / skid register。
   必須同拍鎖住 `resp_valid`、`resp_read`、`resp_source`、`resp_tid`、
   `resp_rdata`，避免 back-to-back request 時 stale rdata 被新 command 吃掉。
2. 接回 consumer probe：
   `REQUANT producer writes L1 -> UDMA STORE reads same L1 address -> UDMA output SRAM CRC`。
   先用 16B `[requant_out] + zeros`，通過後再擴成多 byte/tile。
3. 同樣模式推到 POOL/EWE/TNPS：
   producer engine result byte/line 寫 L1，consumer/UDMA 從同一 L1 address 讀，
   再做 output SRAM CRC。
4. 將目前 ref-fill full output CRC 逐步替換成真正 producer output
   SRAM/L1Mesh image，再做 full output tensor compare/CRC。

## 快速命令

```bash
./rtl/batch/run_verilog_final.py --filter slice --limit 1 --rerun-all \
  --require-crc-coverage --require-final-output-crc

./rtl/batch/run_verilog_final.py --filter slice --limit 3 --rerun-all --no-build \
  --require-crc-coverage --require-final-output-crc

./rtl/batch/run_verilog_final_smoke.py --test host \
  --program rtl/obj/verilog_final/programs/deeplab_v3_plus_float_L3.final.hex \
  --ref-program rtl/bin/ETHZ_v6_slice/deeplab_v3_plus_float_L3.bin
```
