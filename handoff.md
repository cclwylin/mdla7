# MDLA7 Handoff

日期：2026-05-10
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## 目前狀態

- Commit 基準：`8a101b2 Add Host Command interface diagram`
- 工作樹目前有未提交修改。
- ETHZ_V6 fast-only：`51/51 ok`，Function FAIL `0`，Skipped OP `0`。
- 暫時不要把 conflict / mesh 當成驗證門檻；使用者目前指定只看 fast。

## 本輪變更重點

- `systemc/src/test_model.cpp`
  - 已接 microblock-level fused pipeline 的主要路徑。
  - 已驗證：
    - `microisp_quant`：`178/178 PASS`
    - `dped_quant`：`84/84 PASS`
    - `pynet_v2_quant`：`212/212 PASS`
- `batch/run_*.py`
  - 所有 runner 都支援 `--fast-only`。
  - corpus runner fast-only 只跑 fast，CSV conflict/mesh 欄位留空，summary 只顯示 fast。
  - `run_model.py --fast-only` 是 `--l1-timing fast` 的 alias。

## Next To-do

- 優先處理 ETHZ_V6 中間層 DRAM W > 1KB 的 pattern；大 write 主要集中在 `d2spac` / `add` / `maxpool` / `softmax`：

```text
srgan_quant        83.000 MB
  L54 F53 d2spac   64.000 MB  out=(1024,1024,64)
  L52 F51 d2spac   16.000 MB  out=(512,512,64)
  L55 F55 conv      3.000 MB  out=(1024,1024,3)

srgan_float        32.000 MB
  L54 F53 d2spac   32.000 MB  out=(512,512,64)

microisp_quant     11.953 MB
  L41/L84/L127 add       each 1.992 MB
  L42/L85/L128 d2spac    each 1.992 MB

pynet_v2_quant      6.000 MB
  L162 F161 d2spac  6.000 MB  out=(1024,1536,4)

dped_quant          4.500 MB
  L81 F81 add       4.500 MB  out=(512,768,12)

unet_quant          4.000 MB
  L2 F2 maxpool     4.000 MB  out=(512,512,16)

sd_encoder_quant    1.000 MB
  L246 F246 softmax 1.000 MB  out=(1,1024,1024)

sd_decoder_quant    1.000 MB
  L42 F42 softmax   1.000 MB  out=(1,1024,1024)

mobilevit_v2_float  0.013 MB
  several softmax writes, about 1-4 KB each
```

- 優先看 `d2spac -> consumer`、`add -> d2spac`、`maxpool -> consumer` handoff，目標是把上面大中間寫回收掉。

## 快速命令

```bash
make -C systemc -s
./batch/run_ethz_v6.py --list
./batch/run_ethz_v6.py --filter xlsr_quant --limit 1 --fast-only --rerun-all
./batch/run_ethz_v6.py --filter resnet_quant --limit 1 --fast-only --rerun-all
./batch/run_ethz_v6.py --filter mobilenet_v3_quant --limit 1 --fast-only --rerun-all
./batch/run_ethz_v6.py --filter mv3_depth_quant --limit 1 --fast-only --rerun-all
```

Profile entry：

```text
batch/profile_ethz_v6.html
batch/output/<stem>.html
```

注意：Python entry point 從 repo root 用 `./batch/<runner>.py` 執行；產物都在 `batch/output/`。
