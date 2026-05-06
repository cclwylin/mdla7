# 第 9 章 — Mdla7System Top Wiring

> 上一章：[第 8 章 — program.bin 格式與 Reference Generation](08_program_bin_reference.md)

本章你會學到什麼：

- `Mdla7System` 如何把 Host、Command Engine、engines、memory 接起來。
- SystemC `sc_fifo` 在這個 simulator 裡扮演什麼角色。
- CONV → Requant chain 如何 wiring。
- done FIFO 和 dtype latch 如何回到 Command Engine。
- top module 和硬體 block diagram 如何對應。

---

## 9.1 Top module 入口

Top module 在 [`system.h`](../systemc/include/mdla7/system.h)：

```cpp
class Mdla7System : public sc_core::sc_module {
    ...
};
```

它的責任不是做 compute，而是 instantiate 和 bind：

```text
Host
CommandEngine
Udma
ConvEngine
RequantEngine
EweEngine
PoolEngine
L1Mesh / Dram / L1Manager
FIFOs
```

這對應 HW spec 的 top-level wiring。

---

## 9.2 Descriptor stream

Host 到 Command Engine：

```cpp
sc_core::sc_fifo<Descriptor> desc_stream{"desc_stream", 256};

host.desc_out(desc_stream);
cmd.desc_in(desc_stream);
```

`desc_stream` 可以看成 command ring buffer 的 simulator 版本。

| FIFO | depth | 說明 |
|---|---:|---|
| `desc_stream` | 256 | Host 上傳 descriptor 給 Command Engine |

Host 每寫一筆 descriptor，Command Engine 就有機會 decode / schedule。

---

## 9.3 Per-engine config FIFO

Command Engine 到 engines：

```cpp
sc_fifo<DescriptorBody> conv_cfg;
sc_fifo<DescriptorBody> requant_cfg;
sc_fifo<DescriptorBody> ewe_cfg;
sc_fifo<DescriptorBody> pool_cfg;
sc_fifo<DescriptorBody> udma_cfg;
```

binding：

```cpp
cmd.conv_cfg_out(conv_cfg);       conv.cfg_in(conv_cfg);
cmd.requant_cfg_out(requant_cfg); requant.cfg_in(requant_cfg);
cmd.ewe_cfg_out(ewe_cfg);         ewe.cfg_in(ewe_cfg);
cmd.pool_cfg_out(pool_cfg);       pool.cfg_in(pool_cfg);
cmd.udma_cfg_out(udma_cfg);       udma.cfg_in(udma_cfg);
```

每個 engine 只看到自己的 `DescriptorBody`。header 的 `dtype`、`layer_id` 等不直接進 FIFO，dtype 透過 latch 處理。

---

## 9.4 Done FIFO

engines 到 Command Engine：

```cpp
sc_fifo<uint8_t> conv_done;
sc_fifo<uint8_t> requant_done;
sc_fifo<uint8_t> ewe_done;
sc_fifo<uint8_t> pool_done;
sc_fifo<uint8_t> udma_done;
```

binding：

```cpp
conv.done_tag_out(conv_done);       cmd.conv_done(conv_done);
requant.done_tag_out(requant_done); cmd.requant_done(requant_done);
...
```

payload 目前固定 0。Command Engine 自己根據 per-engine pending queue 找出對應 signal tag。

這是簡化設計。若未來 engine 內部 out-of-order completion，done payload 必須帶 task ID。

---

## 9.5 CONV → Requant chain

CONV 和 Requant 之間不是透過 L1 output，而是 16 條 int32 FIFO：

```cpp
std::array<std::unique_ptr<sc_fifo<int32_t>>, 16> chain;
```

constructor：

```cpp
for (int i = 0; i < 16; ++i) {
    chain[i] = make_unique<sc_fifo<int32_t>>(..., 2);
    conv.chain_out[i] = chain[i].get();
    requant.chain_in[i] = chain[i].get();
}
```

這代表：

```text
CONV pushes psum to chain[oc % 16]
Requant drains chain[oc % 16]
```

chain depth 是 2，足夠讓 functional dataflow block / unblock，並保留 backpressure 味道。

---

## 9.6 Memory subsystem wiring

Top module instantiate：

```cpp
L1Mesh    l1mesh;
Dram      dram;
L1Manager l1mgr;
```

constructor：

```cpp
l1mesh("l1mesh"),
dram("dram", dram_bytes),
l1mgr("l1mgr", l1mesh, dram),
udma("udma", l1mgr),
conv("conv", l1mgr),
requant("requant", l1mgr),
ewe("ewe", l1mgr),
pool("pool", l1mgr)
```

所有 engine 共用同一個 `L1Manager&`：

```text
engine -> L1Manager -> L1Mesh or Dram
```

這讓 latency model 集中在 memory module 裡。

---

## 9.7 dtype latch wiring

Command Engine 有 pointer：

```cpp
uint8_t* conv_dtype_latch;
uint8_t* req_dtype_latch;
uint8_t* ewe_dtype_latch;
uint8_t* pool_dtype_latch;
```

Top module bind：

```cpp
cmd.conv_dtype_latch = &conv.last_dtype;
cmd.req_dtype_latch  = &requant.last_dtype;
cmd.ewe_dtype_latch  = &ewe.last_dtype;
cmd.pool_dtype_latch = &pool.last_dtype;
```

因此 Command Engine issue descriptor 時會先寫 engine 的 `last_dtype`，再寫 body FIFO。

這是一個重要的 simulator shortcut。若未來要更接近真實 command packet，可以把 header 也傳進 engine FIFO。

---

## 9.8 dram_bytes parameter

`Mdla7System` constructor：

```cpp
Mdla7System(sc_module_name nm,
            std::size_t dram_bytes = 256 * 1024 * 1024)
```

`test_model.cpp` 會根據 program 需要的最大 address 動態 sizing DRAM，避免大模型 out-of-bounds。

這代表 top module 可用在兩種情境：

| 情境 | dram size |
|---|---|
| small unit test | default 256 MB |
| full model test | test_model 傳入 computed size |

---

## 9.9 SystemC thread map

每個 module 通常有自己的 `SC_THREAD`：

| Module | thread |
|---|---|
| Host | `run()` |
| CommandEngine | `dispatch()`、`collect()` |
| UDMA | `run()` |
| CONV | `run()` |
| Requant | `run()` |
| EWE | `run()` |
| POOL | `run()` |

這使得只要 dependency tag 允許，engines 可以在 simulation time 上 overlap。

---

## 9.10 常見誤解

| 誤解 | 正確理解 |
|---|---|
| Top module 有 scheduling logic | scheduling 在 Command Engine / test_model descriptor generation |
| CONV output 先寫 L1 再給 Requant | CONV 到 Requant 是 chain FIFO |
| done FIFO payload 就是 tag | payload 目前固定 0，tag 在 Command Engine pending queue |
| cfg FIFO 傳完整 descriptor | 目前傳 body，dtype 另用 latch |
| L1Manager 是 engine | 它是 memory router / latency entry |
| SystemC FIFO depth 不重要 | depth 會影響 blocking / backpressure 行為 |

---

## 9.11 本章小結

`Mdla7System` 是架構圖的具體 wiring：

```text
Host -> desc_stream -> CommandEngine
CommandEngine -> cfg FIFOs -> engines
engines -> done FIFOs -> CommandEngine
CONV -> chain -> Requant
engines -> L1Manager -> L1Mesh / DRAM
```

你讀任何 engine 或 scheduler 問題時，都可以回到本章的 wiring 圖確認資料和 event 走哪條路。

> 下一章 → [第 10 章 — Host 與 Command Engine Module Design](10_host_command_engine.md)
