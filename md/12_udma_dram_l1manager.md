# 第 12 章 — UDMA、DRAM Model、L1Manager Module Design

> 上一章：[第 11 章 — CONV / Requant Data Path](11_conv_requant_datapath.md)

本章你會學到什麼：

- 第 4 章 memory spec 如何落到 SystemC module design。
- CONV ACT/WGT read 為什麼 direct 接 L1Mesh。
- `L1Manager` 為什麼是 non-CONV engine / UDMA memory access 的入口。
- UDMA descriptor 如何變成 memory copy / transform。
- DRAM / L1 timing 在 module 裡如何 impose。
- profile 裡的 UDMA read/write lane 從哪裡來。

---

## 12.1 Module map

memory-related modules：

| Module | Source | 責任 |
|---|---|---|
| `L1Mesh` | [`memory.h`](../systemc/include/mdla7/memory.h) | SRAM storage + bank latency |
| `Dram` | [`memory.h`](../systemc/include/mdla7/memory.h) | DRAM storage + row/refresh latency |
| `L1Manager` | [`memory.h`](../systemc/include/mdla7/memory.h) | address decode + route |
| `Udma` | [`udma.h`](../systemc/include/mdla7/udma.h) | descriptor-driven data movement |

Engine 不直接知道 address 在哪裡：

```cpp
l1mgr.read(addr, dst, bytes);
l1mgr.write(addr, src, bytes);
```

---

## 12.2 L1Manager design

硬體 spec 上，CONV ACT/WGT AXI_R 直接接 L1Mesh，不經過 L1Manager。
這條 direct path 讓 CONV read 取得最高服務優先權，目標是讓 CONV cluster
不被 EWE / POOL / UDMA background traffic 餓住。

`L1Manager` 是 non-CONV engine / UDMA 到 L1Mesh 的入口 arbiter。

L1Manager 入口 priority 目前定義為：

| Priority | Traffic |
|---:|---|
| 1 | Requant writeback / parameter read |
| 2 | EWE / POOL / TNPS |
| 3 | UDMA background load/store |

目前 SystemC implementation 還是簡化 router：

```cpp
if addr_in_l1mesh(addr):
    mesh.read/write
elif addr_in_dram(addr):
    dram.read/write
else:
    SC_REPORT_ERROR
```

也就是說，HW spec 已定義 CONV direct read path 與 L1Manager
non-CONV 入口 priority；SystemC 尚未做完整 port-accurate arbitration。
這是刻意簡化，因為現階段重點是：

- functional correctness
- first-order bandwidth / latency
- descriptor scheduling
- tiling effects

未來可在 L1Manager / L1Mesh service model 擴充 CONV direct path 與
non-CONV traffic 的 contention。

---

## 12.3 L1Mesh storage

L1Mesh 是 3 MB dual-4x4 banked SRAM NoC。兩個 mesh plane 共用同一組
16-bank SRAM backend：

| Item | Value |
|---|---:|
| banks | 16 |
| mesh planes | 2 x 4x4 |
| macro | 768 x 16B = 12 KB |
| macros per bank | 16 |
| bank capacity | 192 KB |
| total macros | 256 |
| total capacity | 3 MB |

NoC edge ports 採四邊分流，每個 mesh plane 都有：

| Edge | Ports | Primary traffic |
|---|---:|---|
| left | 4R + 4W | CONV ACT_R direct ingress |
| top | 4R + 4W | CONV WGT_R direct ingress |
| right | 4R + 4W | L1Manager_R |
| bottom | 4R + 4W | L1Manager_W |

單一 plane 是 16R + 16W；兩個 plane 合計是 32R + 32W NoC
injection/ejection capacity。每個 bank 的 SRAM backend service 仍是每
SRAM cycle 一個 16B read beat / 一個 16B write beat。

目前 SystemC functional storage 仍用：

```cpp
std::vector<uint8_t> mem;
```

read / write：

```cpp
memcpy(...)
impose_bank_latency(...)
```

對 simulator 來說，data copy 和 time wait 分開：

```text
functional state 更新
simulation time 推進
```

這讓 engine code 寫起來像普通 memory access，但仍有 timing。

---

## 12.4 L1 bank finish arrays

L1 有：

```cpp
read_bank_finish_[16]
write_bank_finish_[16]
```

這代表 read/read 和 write/write 同 bank 會 serialize，不同 bank 可 overlap。

注意目前 read/write 分開，對同 bank simultaneous read/write 較樂觀。這是 model simplification。

硬體 spec 的 per-bank read priority 是：

```text
1. CONV ACT_R
2. CONV WGT_R
3. L1Manager_R
```

write slot 由 L1Manager_W 使用。

---

## 12.5 Dram storage

Dram 也用 vector：

```cpp
std::vector<uint8_t> mem;
```

address 是 DRAM absolute address，所以存取時 subtract base：

```cpp
mem[addr - DRAM_BASE]
```

若 test_model 沒 sizing 夠大，這裡可能 out-of-bounds。v8.22 之後由 program scan 動態 sizing。

---

## 12.6 Dram timing

DRAM timing 包含：

| 項目 | 值 |
|---|---:|
| bandwidth | 48 B/cycle |
| row miss | 50 cycles |
| refresh period | 7800 cycles |
| refresh stall | 200 cycles |

每次 access：

```text
start = max(last_finish, now)
penalty = row miss ? 50 : 0
access = ceil(bytes / 48)
refresh maybe added
finish = start + penalty + access
```

目前 `last_finish_` 是 single queue，所以 DRAM access 大致 serialize。

---

## 12.7 UDMA run loop

UDMA：

```cpp
DescriptorBody body = cfg_in.read();
switch (body.udma.mode):
    do_linear
    do_strided
    do_gather
    do_concat
    do_slice
    do_depth_to_space
done_tag_out.write(0)
```

每個 UDMA descriptor 是 atomic task：做完才 signal done。

---

## 12.8 UDMA read / write profiling

UDMA 根據 `direction` 分 lane：

```cpp
if direction == 1:
    busy_time_write += ...
    tasks_write.push(...)
else:
    busy_time_read += ...
    tasks_read.push(...)
```

所以 profile 裡可以分辨：

```text
UDMA_R: DRAM -> L1 loads
UDMA_W: L1 -> DRAM stores
```

這對看 overlap 很重要。好的 scheduler 會讓 UDMA_R 提早餵 compute，UDMA_W 背景 drain。

---

## 12.9 UDMA startup cost

UDMA 每筆 descriptor 有：

```cpp
wait(16, SC_NS)
```

代表 decode / startup cost。

大量小 UDMA descriptor 會累積：

```text
1000 descriptors * 16 cycles = 16000 cycles
```

所以 tiling 太碎不只增加 CONV fill，也增加 UDMA startup。

---

## 12.10 Data transform correctness

UDMA 不只 copy，也做 layout transform：

| Mode | 風險 |
|---|---|
| STRIDED_2D | stride 是 bytes，不是 elements |
| INDEXED_GATHER | index table dtype / location |
| SCATTER_CONCAT | concat axis layout assumption |
| STRIDED_SLICE | column offset 是 byte offset |
| DEPTH_TO_SPACE | NHWC mapping 和 element size |

layout op fail 時，先看 UDMA mode 欄位，再看 compiler reference 是否同一 mapping。

---

## 12.11 Memory and dependency

Memory module 不知道「這塊 L1 是否還有人要讀」。資料生命週期靠 descriptor tags：

```text
producer write done tag
consumer wait tag
reuse wait last consumer / store tag
```

所以 memory bug 常常要回到 scheduler 找，而不是改 `L1Mesh`。

---

## 12.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| L1Manager 會防止資料 hazard | 它只 route address，不管理 lifetime |
| UDMA direction 決定合法 address | 合法性由 address range 決定 |
| DRAM 有完整 multi-bank parallel model | 目前有 row/bank identity，但 request queue 是簡化 single finish |
| L1 memcpy 代表零時間 | memcpy 後仍 impose bank latency |
| UDMA transform 不需 reference | layout transform 最需要 reference 對齊 |

---

## 12.13 本章小結

memory module design 的主線：

```text
L1Manager routes
L1Mesh / Dram impose timing
UDMA consumes descriptors and performs data movement
Dependency tags protect memory lifetime
```

Debug memory 問題時，把它拆成：

1. address range
2. byte count
3. stride / layout
4. dependency
5. timing

> 下一章 → [第 13 章 — EWE / POOL / SOFTMAX / D2SPACE](13_ewe_pool_softmax_d2space.md)
