# 第 10 章 — Host 與 Command Engine Module Design

> 上一章：[第 9 章 — Mdla7System Top Wiring](09_system_top_wiring.md)

本章你會學到什麼：

- Host module 在 simulator 裡如何上傳 descriptor。
- Command Engine 的 dispatch / collect 兩條 thread。
- pending lookahead queue 如何工作。
- normal descriptor 與 stream descriptor 的 scheduling 差異。
- dependency tag table、pending_tags、tag_changed event 如何互動。
- junior 如何 debug Command Engine 卡住或越序問題。

---

## 10.1 Host module

Host 在 [`host.h`](../systemc/include/mdla7/host.h)：

```cpp
SC_MODULE(Host) {
    sc_fifo_out<Descriptor> desc_out;
    std::vector<Descriptor> program;
    void run();
};
```

`test_model.cpp` 在 `sc_start()` 前填：

```cpp
sys.host.program = std::move(program);
```

Host thread：

```cpp
for (auto& d : program) {
    desc_out.write(d);
    wait(1, SC_NS);
}
```

因此 Host 是簡化的 RISC-V runtime stub：

| 真實系統 | simulator |
|---|---|
| firmware writes command ring | Host writes `desc_stream` FIFO |
| MMIO doorbell | FIFO data_written_event |
| command buffer in DRAM | `std::vector<Descriptor> program` |

---

## 10.2 Command Engine 兩條 thread

Command Engine 有兩條 `SC_THREAD`：

| thread | 責任 |
|---|---|
| `dispatch()` | 讀 descriptor、檢查 wait_tags、issue 到 engine cfg FIFO |
| `collect()` | 等 engine done、set signal tag、notify scheduler |

這個 split 很自然：

```text
dispatch side = forward path
collect side  = completion path
```

Completion path 會喚醒 dispatch path，讓等待 dependency 的 descriptor 變 ready。

---

## 10.3 tag_done table

Command Engine 內部：

```cpp
bool tag_done[256];
```

初始全部 true：

```cpp
for (int i = 0; i < 256; ++i) tag_done[i] = true;
```

descriptor issue 或進入 stream pending 時：

```text
signal_tag reserved -> tag_done[tag] = false
```

engine done 時：

```text
tag_done[tag] = true
tag_changed.notify()
```

`waits_ready()` 是 AND-check：

```cpp
for wait tag:
    if not done: return false
return true
```

---

## 10.4 pending_tags per engine

Engine done payload 沒帶 tag，所以 Command Engine 需要 queue：

```cpp
std::queue<uint8_t> pending_tags[OC_NUM];
```

issue 時：

```cpp
pending_tags[op_class].push(signal_tag);
```

collect 時：

```cpp
t = pending_tags[cls].front();
pending_tags[cls].pop();
tag_done[t] = true;
```

這假設同一 engine 完成順序等於 issue 順序。現在每個 engine 都是 single-thread in-order，所以成立。

---

## 10.5 dispatch pending queue

`dispatch()` 裡有：

```cpp
std::deque<Descriptor> pending;
constexpr size_t LOOKAHEAD_LIMIT = 64;
```

流程：

```text
fill pending from desc_in until limit
find best ready descriptor
issue it
if none ready, wait new descriptor or tag_changed
```

LOOKAHEAD_LIMIT 的目的：

- 允許 stream descriptor bypass。
- 不讓 8-bit tag wrap hazard 變大。
- 控制 scheduler search cost。

---

## 10.6 normal descriptor scheduling

Normal descriptor 沒有 `DF_STREAM`。

規則：

```text
只看 pending front。
front ready -> issue。
front not ready -> stop，不看後面 normal work。
```

原因是 normal schedule 可能重用固定 L1 region。保守 in-order 可以避免 data overwrite。

這是 correctness-first design。

---

## 10.7 stream descriptor scheduling

Stream descriptor 有 `DF_STREAM`，可以 bypass。

stream metadata：

| 欄位 | 用途 |
|---|---|
| `layer_id` | profile / debug |
| `stream_slot` | ping-pong slot |
| `microblock_id` | wavefront order |
| `stream_meta_flags` | load / compute / store / final tile |

priority：

| 類型 | priority |
|---|---:|
| tail | 0 |
| EWE | 10 |
| UDMA read | 20 |
| CONV | 30 |
| Requant / POOL | 40 |
| UDMA write | 60 |

同類再用 `microblock_id` tie-break。

---

## 10.8 tail waiting

stream tail 還沒 ready 時，Command Engine 允許部分 younger work 先走：

| allowed | 原因 |
|---|---|
| UDMA read | 預取下一個 tile |
| CONV / Requant compute | 保持淺層 compute front |

但不允許任意 store 越過 tail，避免 L1 lifetime 被破壞。

---

## 10.9 last_activity 與 tag_fire_time

Command Engine 記錄：

```cpp
sc_time last_activity;
sc_time tag_fire_time[256];
```

用途：

| 欄位 | 用途 |
|---|---|
| `last_activity` | test harness 報告真正完成時間 |
| `tag_fire_time[tag]` | per-layer cycle reporting |

`sc_start()` 可能跑到預算上限，但真正最後一個 tag 早就完成。`last_activity` 可避免 sim time report 被 budget 污染。

---

## 10.10 Debug：Command Engine 卡住

卡住通常是某個 wait tag 永遠不 done。

檢查順序：

1. 找最後一個 dispatch log。
2. 找 pending descriptor 的 wait tags。
3. 查該 tag 是否有 upstream signal descriptor。
4. 查 upstream descriptor 是否真的 issue 到 engine。
5. 查 engine 是否完成並寫 done FIFO。
6. 查 `pending_tags[op_class]` 是否 pairing 正確。

常見原因：

| 原因 | 現象 |
|---|---|
| wait tag 填錯 | descriptor 永遠不 ready |
| signal_tag = 0 | 完成但不 notify dependency |
| engine path 沒寫 done | collect 收不到 |
| tag wrap live range 太近 | old/new tag state 混淆 |
| stream reservation 錯 | younger descriptor 提早或永遠等 |

---

## 10.11 Debug：越序造成 wrong output

越序 bug 常見於 stream scheduling：

```text
load new tile overwrites buffer
old store / compute 還沒讀完
```

要看：

| 檢查 | 說明 |
|---|---|
| descriptor 是否 `DF_STREAM` | 只有 stream 可 bypass |
| `stream_slot` | ping-pong buffer 是否對 |
| `SMF_STORE` | store 是否被排太後面 |
| tail barrier | 是否缺少 final ordering |
| `allowed_during_tail_wait` | younger work 是否被允許 |
| L1 address overlap | 是否覆蓋同一區 |

這類 bug 不一定會卡住，通常表現為多 tile model mismatch。

---

## 10.12 常見誤解

| 誤解 | 正確理解 |
|---|---|
| Host 會執行真正 firmware | 目前 Host 是 descriptor uploader |
| Command Engine 做 compute | 它只 dispatch / dependency tracking |
| tag 由 engine 決定 | tag 由 Command Engine pending queue 對應 |
| normal descriptor 也可 bypass | normal descriptor 保持 in-order |
| lookahead 越大越好 | 太大會增加 tag wrap / lifetime risk |
| tail barrier 只是 performance hint | tail 也保護 ordering |

---

## 10.13 本章小結

Host 和 Command Engine 是 MDLA7 的 control plane：

```text
Host uploads descriptors
CommandEngine checks dependency tags
CommandEngine dispatches body to engines
Engines finish
CommandEngine sets signal tags
```

記住一句話：

```text
Command Engine 不知道 tensor math，但它決定 tensor math 何時能安全開始。
```

> 下一章 → [第 11 章 — CONV / Requant Data Path](11_conv_requant_datapath.md)
