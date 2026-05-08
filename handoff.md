# MDLA7 Handoff

日期：2026-05-09
Repo：`/Volumes/4T_OFFICE/_Codex/MDLA7_Codex`
Branch：`main`

## ETHZ_V6 新 OP 完成紀錄

Baseline：
`batch/output/baseline_ethz_v6_before_new_ops_20260509_051832`

已補完：
- CONV stride=3 / stride=16
- int16 `ADD/MUL/SUB`
- int16 `MAX_POOL/AVG_POOL/MEAN`
- int16 `RESHAPE/CONCAT`

```text
ETHZ_V6 fast-only: 51/51 ok
Function FAIL:     0
Skipped OP:        0
conflict/mesh:     未跑
```

新增 OP 後 fast simulation 增加：

```text
total: +16,522,995 cycles / +8.696 ms
```

```text
dped_int16        +5,698,100 cyc  +2.999 ms
microisp_int16    +4,970,807 cyc  +2.616 ms
unet_int16        +2,806,989 cyc  +1.477 ms
microisp_float    +1,051,391 cyc  +0.553 ms
esrgan__int16       +943,075 cyc  +0.496 ms
microisp_quant      +555,751 cyc  +0.293 ms
pynet_v2_float      +320,612 cyc  +0.169 ms
pynet_v2_quant      +155,174 cyc  +0.082 ms
vit_b16_quant        +21,096 cyc  +0.011 ms
```

## Top To-do

- [ ] 優先處理 ETHZ_V6 中間層 DRAM W > 1KB 的 pattern；大 write 主要集中在 `d2spac` / `add` / `maxpool` / `softmax`：

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

- [ ] 優先看 `d2spac -> consumer`、`add -> d2spac`、`maxpool -> consumer` handoff，目標是把上面大中間寫回收掉。

## 目前狀態

- 最新 commit：`96d6df7 Optimize EWE CONV microblock overlap`
- 工作樹目前有未提交修改。
- ETHZ_V6 fast baseline copy：`batch/output/baseline_ethz_v6_before_new_ops_20260509_051832`
- 最新 ETHZ_V6 fast-only：`51/51 ok`，Function FAIL `0`，Skipped OP `0`。
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
