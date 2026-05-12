# 第 22 章 — RTL Bring-up：verilog

本章記錄目前 MDLA7 的 Verilog bring-up 狀態。`verilog_ctrl` 與
`verilog_final` 兩條路已經退役並合併；現在只有一個硬體 Verilog path：

| mode | 角色 | 主要目錄 |
|---|---|---|
| `fast` | analytical SystemC model | `systemc/` |
| `synth` | SystemC synth timing / microblock model | `systemc/` |
| `verilog` | real hardware Verilog control + datapath | `rtl/verilog/` |

舊的 `rtl/synth` control shell 和 `batch/run_verilog_ctrl.py` 已移除。
後續不要再建立 `verilog_ctrl` / `verilog_final` 分支目錄。

## 22.1 Verilog 目標

`rtl/verilog` 是接近硬體的路徑：

```text
Testbench loads .bin into DRAM
  -> Host / descriptors
  -> UDMA load DRAM to L1
  -> CONV / TNPS / POOL / EWE / REQUANT work
  -> engine writes L1
  -> UDMA store L1 to DRAM
  -> reload / CRC / checker
```

目前 `verilog` 已有：

- microblock control path
- L1Manager 2-deep input FIFO and backpressure
- L1Mesh route timing estimator
- CONV / REQUANT / POOL / EWE / TNPS / UDMA Verilog modules
- host-driven descriptor testbench
- closed-loop DRAM -> UDMA -> L1 -> engine -> L1 -> UDMA -> DRAM smoke

## 22.2 主要檔案

| 檔案 | 角色 |
|---|---|
| `rtl/verilog/mdla7_top.v` | hardware top integration |
| `rtl/verilog/host.v` | host descriptor issue / checker |
| `rtl/verilog/common.v` | shared types / helper modules |
| `rtl/verilog/l1manager.v` | per-source FIFO and arbitration |
| `rtl/verilog/l1mesh.v` | L1 fabric |
| `rtl/verilog/conv.v` | CONV sample datapath and address walk |
| `rtl/verilog/requant.v` | REQUANT sample datapath |
| `rtl/verilog/pool.v` | POOL sample datapath |
| `rtl/verilog/ewe.v` | EWE sample datapath |
| `rtl/verilog/tnps.v` | TNPS address mapping and engine |
| `rtl/verilog/udma.v` | UDMA load/store path |
| `rtl/verilog/route.v` | placement-aware route estimator |
| `rtl/verilog/Testbench_*.v` | block/top/host smoke tests |

Batch tools:

```bash
./batch/gen_verilog_program.py
./batch/run_verilog_smoke.py
./batch/run_verilog.py
```

## 22.3 Smoke

Run all major Verilog smoke tests:

```bash
./batch/run_verilog_smoke.py \
  --test conv --test requant --test pool --test ewe --test tnps \
  --test route --test top --test host --test closed_loop
```

Closed-loop-only smoke:

```bash
./batch/run_verilog_smoke.py --test closed_loop
```

A passing closed-loop smoke means the descriptor path exercised:

```text
DRAM -> UDMA -> L1 -> CONV/TNPS/POOL/EWE -> L1 -> UDMA -> DRAM -> UDMA -> L1CRC
```

## 22.4 Regression

Small regression:

```bash
./batch/run_verilog.py --filter slice
```

Single pattern:

```bash
./batch/run_verilog.py --filter deeplab_v3_plus_float_L64 --rerun-all
```

Closed-loop dataflow:

```bash
./batch/run_verilog.py --filter slice --closed-loop-dataflow
```

Useful report columns:

| column | meaning |
|---|---|
| `ans` | PASS / SAMPLE / SKIP / FAIL |
| `cov` | tensor CRC coverage level |
| `synth_cycles` | SystemC synth profile cycles |
| `verilog_cycles` | Verilog measured cycles |
| `v/synth` | Verilog cycle ratio against synth |

## 22.5 Current Caveats

- Some paths are still sample datapath, not full tensor traversal.
- `--full-tensor` exists for coverage/debug, but the target architecture is
  compact descriptors plus Verilog-side traversal.
- `run_verilog.py --filter deeplab_v3_plus_float_L64 --rerun-all` currently
  exposes a UDMA CRC mismatch on the default microblock-control path; the
  legacy sample descriptor path passes. Treat this as datapath/dataflow work,
  not a file-structure issue.

## 22.6 Rule Going Forward

There is one Verilog implementation:

```text
rtl/verilog
```

Do not add new RTL under `rtl/synth`, and do not recreate `verilog_ctrl` or
`verilog_final` runners. All future hardware control/datapath work goes through
`run_verilog.py` and `run_verilog_smoke.py`.
