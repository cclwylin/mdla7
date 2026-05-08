# 第 4 章 — Memory Hierarchy：DRAM、UDMA、L1Mesh

> 上一章：[第 3 章 — Descriptor ISA 與 Dependency Tag](03_descriptor_tag.md)

本章你會學到什麼：

- MDLA7 的 address space 如何區分 L1Mesh 與 DRAM。
- `L1Manager` 在 simulator 裡扮演什麼角色。
- L1Mesh 16 banks、16-byte stripe、256 B/cycle peak 是什麼意思。
- DRAM row hit / row miss / refresh model 如何影響 cycle。
- UDMA 六種 mode 如何搬資料。
- memory hierarchy 如何和 descriptor tag、tiling、cycle accuracy 連在一起。

---

## 4.1 先用一張文字圖理解

MDLA7 的 memory path 可以先看成：

```text
Compute Engines
  CONV / Requant / EWE / POOL
        |
        v
L1Manager
        |
        +--> L1Mesh SRAM  0x00000000 - 0x002FFFFF
        |
        +--> DRAM         0x10000000 - 0xFFFFFFFF

UDMA 也透過 L1Manager 搬資料
```

CONV ACT_R / WGT_R read 各有專線到 CONV Engine，是 direct L1Mesh path；其他 engine / UDMA 透過
[`L1Manager`](../systemc/include/mdla7/memory.h)。這讓大部分 memory latency
入口集中，同時保留 CONV read 的最高服務優先權：

```cpp
l1mgr.read(addr, dst, n);
l1mgr.write(addr, src, n);
```

只要 address 在 L1 range，就走 L1Mesh；address 在 DRAM range，就走 Dram。

---

## 4.2 必讀檔案

本章主要看：

| 檔案 | 重點 |
|---|---|
| [`memory.h`](../systemc/include/mdla7/memory.h) | L1Mesh、Dram、L1Manager |
| [`udma.h`](../systemc/include/mdla7/udma.h) | UDMA descriptor execution |
| [`descriptor.h`](../systemc/include/mdla7/descriptor.h) | address range helper、UdmaBody |
| [`test_model.cpp`](../systemc/src/test_model.cpp) | L1 address allocation、DRAM tensor layout、UDMA descriptor emission |
| [`spec/spec.md`](../spec/spec.md) | memory bandwidth 與 architecture target |

讀法建議：

1. 從 `descriptor.h` 的 `L1MESH_BASE` / `DRAM_BASE` 開始。
2. 讀 `L1Manager::read()` / `write()`。
3. 讀 `L1Mesh::impose_bank_latency()`。
4. 讀 `Dram::impose_latency()`。
5. 讀 `Udma::run()` 和各個 `do_*()`。

這個順序會先建立「位址去哪裡」，再理解「時間怎麼算」。

---

## 4.3 Address space

MDLA7 simulator 的 address space 在 [`descriptor.h`](../systemc/include/mdla7/descriptor.h) 定義：

```cpp
constexpr uint32_t L1MESH_BASE  = 0x0000'0000;
constexpr uint32_t L1MESH_END   = 0x002F'FFFF;
constexpr uint32_t L1MESH_BYTES = L1MESH_END + 1;
constexpr uint32_t DRAM_BASE    = 0x1000'0000;
constexpr uint32_t DRAM_END     = 0xFFFF'FFFF;
```

換成表格：

| Range | 大小 / 意義 |
|---|---|
| `0x00000000` 到 `0x002FFFFF` | L1Mesh，3 MB on-chip SRAM |
| `0x10000000` 到 `0xFFFFFFFF` | DRAM address space |
| 其他 range | illegal，`L1Manager` 會報錯 |

helper：

```cpp
inline bool addr_in_l1mesh(uint32_t a) { return a <= L1MESH_END; }
inline bool addr_in_dram(uint32_t a) { return a >= DRAM_BASE; }
```

初學 debug memory 問題時，第一步就是把 address 分類：

```text
0x000xxxxx -> L1
0x100xxxxx -> DRAM
其他       -> 可疑
```

---

## 4.4 L1Manager：Non-CONV 入口 arbiter + memory router

硬體 spec 上，CONV ACT/WGT Payload R **直接接 L1Mesh，不經過 L1Manager**。
這條 direct path 讓 CONV read 取得最高服務優先權，避免 CONV compute
cluster starvation。

`L1Manager` 是 non-CONV engine / UDMA 進入 L1Mesh 的仲裁點。

L1Manager 入口 priority 目前定義為：

| Priority | Traffic |
|---:|---|
| 1 | Requant writeback / parameter read |
| 2 | EWE / POOL / TNPS |
| 3 | UDMA background load/store |

目前 SystemC 實作仍是簡化的一階模型，`L1Manager` code path 接近 pass-through router：

```cpp
void read(uint32_t addr, void* dst, uint32_t n) {
    if (addr_in_l1mesh(addr)) mesh_.read(addr, dst, n);
    else if (addr_in_dram(addr)) dram_.read(addr, dst, n);
    else SC_REPORT_ERROR("L1Manager", "addr out of range");
}
```

它負責：

| 功能 | 目前狀態 |
|---|---|
| address decode | implemented |
| route to L1Mesh / DRAM | implemented |
| out-of-range error | implemented |
| Non-CONV entrance arbitration | spec defined; simulator simplified |
| QoS / priority | CONV ACT_R / WGT_R bypass L1Manager via two dedicated direct L1Mesh paths |
| cache coherency | not relevant，這裡是 scratchpad |

換句話說，HW spec 有 CONV ACT_R / WGT_R dedicated direct paths + L1Manager non-CONV arbitration；
目前 simulator 還不是完整 port-accurate interconnect model。真正的 latency
主要在 L1Mesh / DRAM 裡 imposing wait。

這個設計對 simulator 有好處：

- non-CONV engine code 不需要知道某個 address 在 L1 還是 DRAM。
- UDMA mode 可以用同一套 `read()` / `write()`。
- future 如果要加完整 priority contention，可以集中在 `L1Manager`。

---

## 4.5 L1Mesh 的基本 spec

L1Mesh 是 3 MB SRAM scratchpad，採 2 個 parallel 4x4 banked NoC plane。
兩個 mesh plane 共用同一組 16-bank SRAM backend。CONV ACT/WGT read 直接接
L1Mesh，不經過 L1Manager；L1Manager 負責 non-CONV engine / UDMA traffic。

目前 model：

| 參數 | 值 |
|---|---:|
| 容量 | 3 MB |
| bank 數 | 16 |
| NoC topology | 2 x 4x4 mesh planes |
| SRAM macro | 768 x 16B = 12 KB |
| macro 總數 | 256 |
| 每 bank macro | 16 |
| 每 bank 容量 | 192 KB |
| stripe | 16 bytes |
| Payload beat | 16 bytes |
| Payload transaction grouping | tid + last |
| router input FIFO depth | 2 flits provisional |
| per-bank bandwidth | 16 B/cycle |
| sequential peak | 16 banks × 16 B/cycle = 256 B/cycle |
| SRAM clock | 1.3 GHz |
| core clock axis | 1.9 GHz |

NoC edge ports per mesh plane：

目前 simulator 的 logical edge map 是 8 個 perimeter edges：

| Edge | Banks | Payload R lanes | Payload W lanes |
|---|---|---|---|
| W0 | B0, B1 | R0, R1 | W0, W1 |
| E0 | B2, B3 | R2, R3 | W2, W3 |
| W1 | B4, B5 | R4, R5 | W4, W5 |
| E1 | B6, B7 | R6, R7 | W6, W7 |
| W2 | B8, B9 | R8, R9 | W8, W9 |
| E2 | B10, B11 | R10, R11 | W10, W11 |
| W3 | B12, B13 | R12, R13 | W12, W13 |
| E3 | B14, B15 | R14, R15 | W14, W15 |

單一 mesh plane 合計是 16R + 16W edge injection/ejection；2 個 plane 合計是
32R + 32W。這是在降低 NoC 邊界/router/link hot spot，不是把 SRAM backend
bandwidth 乘二；真正的 per-bank service 仍是每 bank 每 SRAM cycle 一個
16B read 或 write beat。

Router input FIFO depth 暫訂為 2 flits。目前 SystemC mesh timing 只記錄
edge/router-output/link finish time，還沒有 input FIFO occupancy / upstream
backpressure，所以這個 depth 先是 architecture knob，不會改變現有 timing。

L1Mesh Payload input probe 可用環境變數打開：

```bash
MDLA7_L1_PAYLOAD_PROBE=batch/output/l1mesh_payload_probe.csv ./systemc/build/test_model ...
```

log 欄位是：

```text
cycle,1st payload (engineid) addr,2nd payload (engineid) addr,...,32nd payload (engineid) addr
```

前 16 欄對應 read Payload input lane 0..15，後 16 欄對應 write Payload input
lane 0..15。每格內容像 `(ewe) 0x00000100`；同一 cycle 同 lane 若有多個
16B beat，會用 `|` 接在同一格。

### 4.5.1 怎麼讀 `L1Mesh-4x4-NoC` 圖

如果你對 NoC 不熟，先不要把圖看成「一大塊 SRAM」。比較好的看法是：

```text
外面的 engine / L1Manager
        |
        v
  mesh 邊界入口 ingress
        |
        v
  2 x 4x4 router grid plane
        |
        v
  shared 目標 bank 的 SRAM macro port
```

圖上的每個名詞可以這樣拆：

| 名詞 | 白話意思 | 在 L1Mesh 圖上代表什麼 |
|---|---|---|
| NoC | Network-on-Chip，晶片內的小網路 | 4x4 router grid，用來把 request 送到正確 SRAM bank |
| mesh | 網格狀連線 | router 只跟上下左右鄰居相連，不是所有點互接 |
| router | 交叉路口 | 決定下一步往左、右、上、下，或送進本地 bank |
| link | 兩個 router 中間的路 | 同一方向同一 cycle 只能過有限資料，會塞車 |
| bank | 可以獨立服務的 SRAM 區塊 | B0..B15，總共 16 個 |
| macro | bank 裡更小的 SRAM 顆粒 | 每顆 `768 x 16B = 12 KB` |
| ingress | 進入 NoC 的入口 | request 從外部 engine 進到 mesh 的邊界 port |
| egress | 離開 NoC 的出口 | read data 從 SRAM bank 回 engine，或 write ack 離開 |

`ingress` 是最容易誤會的詞。它不是一顆 SRAM，也不是一個 bank。
它只是「traffic 從外面進入 mesh 的門口」。

用日常比喻：

```text
SRAM bank = 目的地建築物
router    = 路口
link      = 道路
ingress   = 高速公路交流道入口
```

所以「ACT_R left-edge ingress」的意思是：

```text
CONV 要讀 activation。
這些 read request 不走 L1Manager。
ACT_R 是 CONV Engine 專用線，不和 WGT_R 或 L1Manager_R 共用同一組入口。
它們從 L1Mesh 左邊的入口進入 NoC。
進去後再沿著其中一個 4x4 mesh plane 路由到目標 bank。
```

同理：

```text
WGT_R top-edge ingress
```

意思是：

```text
CONV 要讀 weight。
這些 read request 也不走 L1Manager。
WGT_R 是另一組 CONV Engine 專用線，不和 ACT_R 共用。
它們從 L1Mesh 上方的入口進入 NoC。
```

為什麼要分 left / top / right / bottom？因為不同 traffic 如果全部從同一邊進來，
很容易在同一批 edge port 和前幾個 router link 塞住。四邊分流的目的，是讓
traffic 一開始就分散：

```text
left   edge -> ACT_R dedicated CONV link
top    edge -> WGT_R dedicated CONV link
right  edge -> L1Manager_R
bottom edge -> L1Manager_W
```

這裡的重點是「分散入口」，不是「容量變成四倍」。最後資料還是要進某個 SRAM
bank，而 bank 本身每 cycle 能服務的 16B beat 數仍然有限。

### 4.5.2 一個 16B read beat 怎麼走

假設 CONV 要讀一個 activation byte range，其中某個 16B beat 的 address 算出來是
bank 6：

```text
bank_id = (addr >> 4) & 0xF = 6
```

4x4 bank 位置可以這樣看：

```text
row 0: B0  B1  B2  B3
row 1: B4  B5  B6  B7
row 2: B8  B9  B10 B11
row 3: B12 B13 B14 B15
```

所以 bank 6 在：

```text
x = bank_id % 4 = 2
y = bank_id / 4 = 1
```

如果 request 目標是 bank 6，logical edge map 會讓它從比較近的 E1 edge 進入：

```text
E1 edge -> B7 router -> B6 router -> local SRAM bank
```

用座標表示：

```text
source = (3, 1)
dest   = (2, 1)
route  = (3,1) -> (2,1)
```

目前 mesh mode 不再把每個 16B beat 的 hop latency 串起來加到 critical path；
它會為 edge/router/link/local resource 做 contention reservation，真正完成時間
主要由 bank SRAM service 和 queue wait 決定。

如果目標是 bank 14：

```text
B14 -> x=2, y=3
```

bank 14 走 E3 edge：

```text
source = (3, 3)
dest   = (2, 3)
route  = (3,3) -> (2,3)
```

目前 `L1Mesh-Payload-Edge-Map` 頁面列出 16R / 16W lanes 對 W/E edge 和 bank 的完整 mapping。

### 4.5.3 什麼叫 router/link conflict

假設兩個 request 同一個 cycle 都想走同一條路：

```text
request A: B4 -> B5 link
request B: B4 -> B5 link
```

如果這條 directed link 一個 cycle 只能送一個 flit，那第二個 request 就要等。
這就是 link conflict。

router output conflict 類似。假設同一個 router 同一時間有兩個 request 都想從 east
output 出去：

```text
router B4 east output -> B5
```

那也要仲裁，一個先走，一個等。

在 `--l1-timing=mesh` 裡，一個 16B beat 會依序消耗：

```text
1. long blocking call 先切成 simulator scheduling chunks
2. edge/router/link/local resource 做 contention reservation
3. 目標 bank 的 SRAM lane 服務 Payload beat/chunk
```

所以 mesh mode 比 port-conflict mode 多看了「路上資源是否有 queue buildup」，
但不再把每個 hop 固定 latency 疊到每個 16B stripe 上。

### 4.5.4 edge ports 為什麼不是 SRAM bandwidth

圖上寫：

```text
top edge    : 4R + 4W
right edge  : 4R + 4W
bottom edge : 4R + 4W
left edge   : 4R + 4W
per plane   : 16R + 16W
aggregate   : 32R + 32W across 2 planes
```

其中 left/top edge 分別對應 ACT_R / WGT_R 到 CONV Engine 的專用線。這代表 NoC 邊界有很多入口/出口，讓 traffic 比較容易進出 mesh。dual-plane
model 會把每個 16B flit 放到較不忙的 mesh plane，但最後仍然進同一個
SRAM bank backend。

但 SRAM backend 是另一回事。最後每個 bank 還是：

```text
bank service = 16B / SRAM cycle
```

因此不要把 `16R edge ports` 解讀成：

```text
每個 bank 都變 16 倍快
```

正確解讀是：

```text
很多 request 可以從不同邊進 mesh，降低入口塞車。
但是如果它們最後都打到同一個 bank，同一個 bank 還是會 serialize。
```

這也是為什麼我們分三種 timing mode：

| Mode | 看什麼瓶頸 | 沒看的東西 |
|---|---|---|
| `fast` | 總頻寬大概夠不夠 | bank hotspot、router/link hotspot |
| `conflict` | bank port 會不會撞 | NoC 入口、router、link 會不會撞 |
| `mesh` | edge/router/link/bank 一起估 | 還沒分 ACT/WGT/UDMA 的完整 QoS priority |

### 4.5.5 現在 simulator 的 mesh mode 做了什麼、沒做什麼

目前 code 在 [`memory.h`](../systemc/include/mdla7/memory.h) 裡，核心是
`L1Mesh::impose_mesh_latency()`。

它做了：

```text
Payload lane width = 16B
long blocking API call 先切成 simulator scheduling chunks
Payload 本身只帶 tid + last，不帶 burst metadata
每個 chunk 再分散到 16 banks
bank swizzle 減少 repeated-slice row/column 熱點
edge/router/link/local resource 只當 contention reservation
SRAM bank port service 仍是真正完成時間的主要限制
```

它還沒做完整硬體 QoS：

```text
還沒有把 requester 精確分成 CONV ACT_R / CONV WGT_R / L1Manager_R / L1Manager_W。
目前 L1Manager read/write API 沒帶 requester class。
所以 mesh mode 是 NoC 壅塞近似模型，不是 final RTL-grade NoC simulator。
```

這個限制很重要。它表示 `mesh/fast` 很高時，你應該把它當成：

```text
這個 layer 的 access pattern 可能有 NoC hotspot，值得看。
```

而不是馬上解讀成：

```text
硬體一定會慢這麼多。
```

source 裡的 constants：

```cpp
static constexpr unsigned N_BANKS = 16;
static constexpr unsigned BANK_STRIDE = 16;
static constexpr unsigned BYTES_PER_CYCLE = 16;
static constexpr unsigned PAYLOAD_SCHED_CHUNK_BEATS = 16;
static constexpr double CORE_CLOCK_GHZ = 1.9;
static constexpr double SRAM_CLOCK_GHZ = 1.3;
```

最重要的一句：

```text
Sequential access fans out across all 16 banks.
```

因為 stripe 是 16 bytes：

```text
address 0x0000 - 0x000F -> bank 0
address 0x0010 - 0x001F -> bank 1
address 0x0020 - 0x002F -> bank 2
...
address 0x00F0 - 0x00FF -> bank 15
address 0x0100 - 0x010F -> bank 0
```

連續大塊資料可以分散到 16 banks，理想 peak 就是 256 B/cycle。

Bank 內 macro mapping：

```text
byte_addr[3:0] = byte offset inside 16B beat
bank_id        = (byte_addr >> 4) & 0xF
bank_line      = byte_addr >> 8
macro_id       = bank_line / 768
row_addr       = bank_line % 768
```

每個 bank 的 read priority：

```text
1. CONV ACT_R
2. CONV WGT_R
3. L1Manager_R
```

write slot 由 `L1Manager_W` 使用。

Simulator 提供三種 L1 timing mode：

| Mode | CLI | 用途 |
|---|---|---|
| fast estimate | `--l1-timing=fast` | 預設。用 aggregate bandwidth 估算，不逐 bank 計算 port conflict，適合 regression sweep。 |
| port conflict | `--l1-timing=conflict` | 逐 bank finish-array 模型，read/read、write/write、read/write 同 bank 都會 serialize，適合架構分析。 |
| mesh conflict | `--l1-timing=mesh` | 逐 bank shared 1R/W SRAM service + dual-4x4 transparent NoC contention reservation；長 blocking call 依 Payload scheduling chunk 拆開估 timing；適合找 lane imbalance / queue buildup。 |
| mesh optimistic | `--l1-timing=mesh-opt` | 使用 mesh-style Payload chunk + SRAM bank service，但跳過 NoC resource reservation。 |

明確地說，`mesh` 不是取代 `conflict`，而是：

```text
mesh = SRAM bank/port conflict + NoC edge/router/link conflict
```

因此 `conflict/fast` 看 SRAM bank/port overhead，`mesh/conflict` 看額外 NoC
overhead，`mesh/fast` 則是兩者合併後的總 overhead。

mesh profile 的 HTML 會額外顯示 `L1Mesh Payload lane latency` 表：

| 欄位 | 意義 |
|---|---|
| accesses | 此 lane 被分配到的 Payload scheduling chunks 數 |
| KB | 此 lane 總服務資料量 |
| avg cyc / max cyc | request 到 scheduling chunk 完成的 latency |
| avg wait / max wait | queue wait，包含前序 chunk / resource contention |
| avg service / max service | 單一 scheduling chunk 在 SRAM lane 上的服務時間 |

如果 `max service` 很高，通常代表 scheduling chunk 或 beat 寬度不合理。
若 `max latency` 仍很高，多半是 large FC / weight transfer 的後段 chunks 排隊。

`batch/run_model.py` 可用 `--l1-timing` 單跑其中一種模式。`batch/run_mdla6_pattern.py`、
`batch/run_hotspot.py`、`batch/run_ethz_v5.py`、`batch/run_ethz_v6.py`、
`batch/run_mlperf.py` 則預設一次跑 fast / conflict / mesh 三種模式，方便同一份 HTML
裡直接比較 L1Mesh overhead。

---

## 4.6 L1 bank latency 怎麼算

`L1Mesh::read()` 和 `write()` 先 `memcpy`，再排入 beat-level timing queue：

```cpp
std::memcpy(dst, &mem[offset], n);
if (in_process()) schedule_latency(offset, n, is_read);
```

這表示 functional data 先被複製，然後 simulation time 被推進。對 single-thread deterministic simulator 來說，這是常見寫法。

latency model 的精神：

```text
每個 bank 有自己的 shared R/W finish time。
同一個 bank 的 access 會 serialize。
不同 bank 的 access 可以 parallel。
一個 request 的完成時間是所有 touched banks 的 max finish。
```

pseudo code：

```cpp
for each 16-byte stripe touched:
    bank = (addr / 16) % 16
    start = max(sram_bank_finish[bank], now)
    finish = start + beat_time
    sram_bank_finish[bank] = finish
    max_finish = max(max_finish, finish)

wait(max_finish - now)
```

如果一筆 access 是連續 256 bytes，它剛好 touch 16 banks：

```text
bank 0..15 各拿 16 bytes
理想上約 1 SRAM beat
換成 core cycle 要乘 1.9 / 1.3
```

如果一筆 access 每次都打同一個 bank，例如 stride 剛好讓地址落在同 bank：

```text
bank conflict 增加
parallelism 下降
latency 上升
```

這就是 banked SRAM model 想捕捉的現象。

---

## 4.7 SRAM clock 與 core clock 的縮放

Simulator 的時間軸用 core clock cycle 表示，註解裡提到 core clock 是 1.9 GHz。但 L1Mesh SRAM 是 1.3 GHz。

因此一個 SRAM beat 對 core cycle 的成本是：

```text
CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ = 1.9 / 1.3 ~= 1.46 core cycles
```

source 裡：

```cpp
const sc_core::sc_time access(
    beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
    sc_core::SC_NS);
```

這裡 `SC_NS` 在此 simulator 裡被當成抽象 cycle unit。你會看到很多地方用：

```cpp
wait(cycles, sc_core::SC_NS);
```

所以讀 code 時不要把它理解成真實 nanosecond，而要理解成：

```text
1 SC_NS ~= 1 simulator cycle
```

再由外層用 1.9 GHz 換算成 ms。

---

## 4.8 L1 read / write 共用 SRAM macro port

`L1Mesh` 裡每個 bank 只有一組 SRAM macro finish time：

```cpp
sc_core::sc_time sram_bank_finish_[N_BANKS];
```

這代表 SRAM macro 是 **1R/W**，同一個 bank 同時間只能服務 read 或 write 其中之一，不是 1R+1W。

對 model 的意義：

| 行為 | 模型 |
|---|---|
| read-read 同 bank | 會 serialize |
| write-write 同 bank | 會 serialize |
| read-write 同 bank | 會 serialize |
| different banks | 可 overlap |

這反映 spec 裡 L1Mesh 有讀寫 ingress lane，但最後共享同一組 16-bank SRAM macro backend。

如果未來要更接近 RTL，可以補：

- engine master arbitration。
- outstanding transaction depth。
- requester class（CONV ACT_R / CONV WGT_R / L1Manager_R / L1Manager_W）傳到 `L1Mesh`。

目前的 model 已足夠讓 tiling、bank conflict、DRAM pressure 對 performance 有一階效果。

---

## 4.9 DRAM model 的基本 spec

`Dram` model 是 LPDDR-class abstract model。

目前 constants：

| 參數 | 值 |
|---|---:|
| default capacity | 256 MB |
| row size | 8 KB |
| banks | 16 |
| bandwidth | 48 B/cycle |
| AXI burst length | 16 beats |
| AXI burst bytes | 256 B |
| row miss penalty | 50 cycles |
| refresh period | 7800 cycles |
| refresh stall | 200 cycles |

註解中的 bandwidth 推導：

```text
LPDDR5X-10667 dual x32:
10.667 Gbps/pin * 32 pins * 2 channels / 8 = 85.3 GB/s
85.3 GB/s / 1.9 G cycles/s ~= 44.9 B/cycle
round to 48 B/cycle
```

所以 DRAM sequential access 的理想 bandwidth 是 48 B/cycle，比 L1Mesh sequential peak 256 B/cycle 小很多。

DRAM timing 以 AXI burst window 收費：一個 beat 是 128b = 16B，burst
length 是 16 beats，所以一個 burst 是 256B。未對齊或很小的 transfer
會被 round 到它碰到的 256B burst window；跨兩個 window 就收兩個 burst。

這也是為什麼 NPU performance 很重視：

- L1 reuse
- tiling
- weight / activation locality
- L1-resident handoff
- avoiding redundant store/read

---

## 4.10 DRAM row hit / row miss

DRAM model 追蹤每個 bank 目前 open row：

```cpp
int32_t open_row_[N_BANKS];
```

address 會被拆成：

```cpp
off  = addr - DRAM_BASE;
bank = (off / ROW_BYTES) % N_BANKS;
row  = (off / ROW_BYTES) / N_BANKS;
```

如果同一個 bank 再次讀寫相同 row：

```text
row hit -> no row miss penalty
```

如果 row 不同：

```text
row miss -> +50 cycles
```

access bandwidth 本身：

```cpp
access = ceil(bytes / 48) cycles
```

總 latency：

```text
finish = start + row_miss_penalty_if_any + ceil(bytes / 48)
```

這是一個簡化模型，但可以捕捉兩件重要事情：

| 現象 | 模型反映 |
|---|---|
| 大塊連續讀寫比較有效 | row hit 多，bandwidth 主導 |
| 小碎片、跳躍讀寫比較慢 | row miss penalty 比例變大 |

---

## 4.11 DRAM refresh

DRAM 需要週期性 refresh。model 裡：

```cpp
REFRESH_PERIOD = 7800 cycles
REFRESH_STALL  = 200 cycles
```

當 access start time 跨過新的 refresh period：

```cpp
start += missed * REFRESH_STALL;
```

refresh overhead 比例約：

```text
200 / 7800 ~= 2.6%
```

這個數字不會主導每個 layer，但長模型、DRAM-heavy workload 會累積。

debug performance 時，如果你看到某些 UDMA access 比單純 bytes / 48 更慢，可能包含：

- row miss
- refresh stall
- previous DRAM access serialization

---

## 4.12 DRAM 目前是 single finish time

`Dram` 裡有一個 `last_finish_`：

```cpp
sc_core::sc_time last_finish_{sc_core::SC_ZERO_TIME};
```

每次 access start：

```cpp
start = max(last_finish_, now)
```

這代表 DRAM requests 在 model 裡大致 serialize。雖然它有 16 banks 與 open rows，但沒有完整 bank-level parallel outstanding model。

這是比較保守還是比較樂觀？

| 面向 | 影響 |
|---|---|
| 沒有多 request overlap | 對高併發 DRAM workload 較保守 |
| row hit / row miss 有建模 | 對 access locality 有區分 |
| fixed 48 B/cycle + 256B burst window | 保留 burst length / alignment penalty 的一階效果 |
| refresh 有建模 | 對長時間 bandwidth 有 overhead |

對目前 SystemC simulator，這是合理的一階模型。未來若要更接近 DRAM controller，可加入 per-bank finish time、read/write turnaround、outstanding queue。

---

## 4.13 UDMA 的角色

UDMA 是 DRAM 與 L1Mesh 之間的 data mover，也支援一些 layout transform。

在 descriptor graph 裡，UDMA 常見位置：

```text
DRAM input / weights
        |
        v
UDMA read
        |
        v
L1 tensors
        |
        v
CONV / EWE / POOL
        |
        v
L1 output
        |
        v
UDMA write
        |
        v
DRAM output
```

UDMA descriptor 的 `direction`：

| direction | 語意 | profile lane |
|---:|---|---|
| 0 | DRAM 到 L1，load | `tasks_read` |
| 1 | L1 到 DRAM，store | `tasks_write` |

UDMA implementation 裡會把 read / write busy time 分開記：

```cpp
busy_time_read
busy_time_write
tasks_read
tasks_write
```

這對 profile 很有用，因為 load 和 store 對 pipeline 的意義不同：

- load 餵 compute。
- store drain output，通常可以背景化。

---

## 4.14 UDMA descriptor 的共同欄位

`UdmaBody`：

| 欄位 | 說明 |
|---|---|
| `mode` | UDMA mode |
| `direction` | 0 read / 1 write |
| `src_addr` | source address |
| `dst_addr` | destination address |
| `length` | bytes；依 mode 有不同語意 |
| `src_stride` | source stride bytes |
| `dst_stride` | destination stride bytes |
| `num_chunks` | row / chunk count |
| `idx_table_addr` | gather / concat table |
| `slice_begin[4]` | slice / d2s metadata |
| `slice_end[4]` | slice metadata |

UDMA 的 `run()`：

```cpp
DescriptorBody body = cfg_in.read();
const UdmaBody& u = body.udma;

switch (u.mode) {
case UM_LINEAR_COPY:    do_linear(u); break;
case UM_STRIDED_2D:     do_strided(u); break;
case UM_INDEXED_GATHER: do_gather(u); break;
case UM_SCATTER_CONCAT: do_concat(u); break;
case UM_STRIDED_SLICE:  do_slice(u); break;
case UM_DEPTH_TO_SPACE: do_depth_to_space(u); break;
}
done_tag_out.write(0);
```

UDMA 本身只加 16-cycle decode startup：

```cpp
void wait_bytes(uint64_t) {
    wait(16, sc_core::SC_NS);
}
```

真正 memory bandwidth cost 由 `l1mgr.read()` / `write()` 裡的 L1Mesh / Dram 來 impose。

---

## 4.15 UDMA mode 0：LINEAR_COPY

最單純的 mode：

```text
copy length bytes from src_addr to dst_addr
```

source shape：

```cpp
std::vector<uint8_t> buf(u.length);
l1mgr.read(u.src_addr, buf.data(), u.length);
l1mgr.write(u.dst_addr, buf.data(), u.length);
wait_bytes(u.length);
```

常見用途：

| 用途 | 方向 |
|---|---|
| input tensor load | DRAM -> L1 |
| weight tile load | DRAM -> L1 |
| output tensor store | L1 -> DRAM |
| params blob load | DRAM -> L1 或直接放 L1 |

debug LINEAR_COPY：

| 檢查 | 說明 |
|---|---|
| `length` 是否等於 tensor bytes | dtype 會影響 byte count |
| source / destination 是否重疊 | L1 scratchpad reuse 時要小心 |
| direction 是否符合語意 | 雖非合法性必要，但 profile 會依它分 lane |
| wait tag 是否保護 consumer | compute 不可早於 load |

---

## 4.16 UDMA mode 1：STRIDED_2D

STRIDED_2D 用來複製多列資料：

```text
for r in rows:
    copy length bytes
    src += src_stride
    dst += dst_stride
```

source：

```cpp
for (uint16_t r = 0; r < u.num_chunks; ++r) {
    l1mgr.read(u.src_addr + r * u.src_stride, buf.data(), u.length);
    l1mgr.write(u.dst_addr + r * u.dst_stride, buf.data(), u.length);
}
```

常見用途：

- 只搬 tensor 的一個 rectangular tile。
- 從 full tensor row 裡取部分 columns / channels。
- 把 compact tile 寫回有 stride 的 destination layout。

你可以把欄位理解成：

| 欄位 | 語意 |
|---|---|
| `length` | 每 row 要複製多少 bytes |
| `num_chunks` | row 數 |
| `src_stride` | source 下一 row 距離 |
| `dst_stride` | destination 下一 row 距離 |

常見 bug：

| 現象 | 可能原因 |
|---|---|
| 每 row 開頭正確但下一 row 錯 | stride 設錯 |
| tile 只對第一 row | `num_chunks` 錯 |
| channel tail 錯 | `length` 沒乘 dtype bytes 或 channel count |

---

## 4.17 UDMA mode 2：INDEXED_GATHER

INDEXED_GATHER 先讀 index table：

```cpp
std::vector<uint32_t> idx(u.num_chunks);
l1mgr.read(u.idx_table_addr, idx.data(), idx.size() * sizeof(uint32_t));
```

再依 index 複製：

```text
dst[i] = src[idx[i]]
```

精確一點：

```cpp
s = src_addr + idx[i] * src_stride;
d = dst_addr + i * dst_stride;
copy length bytes
```

常見用途：

- gather 非連續 tensor block。
- 重新排列資料。
- 某些 op lowering 後需要 indirect copy。

debug 時注意：

| 檢查 | 說明 |
|---|---|
| `idx_table_addr` 在哪個 memory space | table 可能在 L1 或 DRAM |
| index 是 element index 還是 row index | source 用 `idx[i] * src_stride` |
| `dst_stride` 是否等於 output element pitch | gather 結果可能有 padding |

---

## 4.18 UDMA mode 3：SCATTER_CONCAT

SCATTER_CONCAT 用 metadata table 描述多個 source：

```cpp
struct ConcatEntry {
    uint32_t src_addr;
    uint32_t length;
};
```

流程：

```text
cursor = dst_addr
for each entry:
    copy entry.length bytes from entry.src_addr to cursor
    cursor += entry.length
```

常見用途：

- TFLite CONCATENATION lowering。
- 多個 branch output 合併成一個 tensor。
- 某些 logical concat 如果不能 L1 view 化，就需要實際 copy。

容易出錯的地方：

| 問題 | 說明 |
|---|---|
| concat axis 不是最後一維 | memory layout 可能不是單純 append |
| source 還沒完成 | concat descriptor wait tags 不完整 |
| source 留在 L1 但被覆蓋 | 需要 store barrier 或 dependency 保護 |
| non-conservative path | 若 compiler 做 view / handoff，要確認 data lifetime |

CONCAT 類 bug 通常不是 UDMA copy loop 本身錯，而是 upstream descriptor DAG 或 layout assumption 錯。

---

## 4.19 UDMA mode 4：STRIDED_SLICE

STRIDED_SLICE 支援 2D slice：

```text
rows = [slice_begin[0], slice_end[0])
col_off = slice_begin[1]
each output row length = length bytes
```

source address：

```cpp
s = src_addr + r * src_stride + col_off;
d = dst_addr + (r - r0) * dst_stride;
```

常見用途：

- TFLite STRIDED_SLICE。
- 取 tensor 的 row / column 子區域。
- 作為 compiler lowering 的簡化 copy primitive。

注意這裡的 `col_off` 是 byte offset，不一定是 element index。如果 dtype 是 int16 / fp16，要記得乘 2。

---

## 4.20 UDMA mode 5：DEPTH_TO_SPACE

DEPTH_TO_SPACE 是 NHWC layout transform。

descriptor encoding：

| 欄位 | 語意 |
|---|---|
| `num_chunks` | input H |
| `slice_begin[0]` | input W |
| `slice_begin[1]` | input Cin |
| `slice_begin[2]` | block size |
| `slice_begin[3]` | output Cout |
| `length` | element bytes |
| `src_stride` | input row bytes |
| `dst_stride` | output row bytes |

合法性檢查：

```text
Cin == Cout * block * block
```

核心 mapping：

```text
input  [ih, iw, ic]
q  = ic / Cout
oc = ic % Cout
bh = q / block
bw = q % block
output [ih * block + bh, iw * block + bw, oc]
```

這類 op 最容易被誤認為「只是 reshape」。但在 NHWC memory layout 下，它通常需要真的搬動 bytes。

---

## 4.21 Memory latency 與 compute latency 的 overlap

這份 simulator 裡，engine 常見 pattern 是：

```cpp
t_begin = sc_time_stamp();

l1mgr.read(...);   // memory latency pushes time
compute functional data

cyc = compute_cycle_formula(...);
elapsed = sc_time_stamp() - t_begin;
if (cyc > elapsed) wait(cyc - elapsed);
```

這代表：

```text
engine wall time = max(memory latency already paid, compute cycle estimate)
```

而不是：

```text
memory latency + compute cycles
```

這是刻意的。真實硬體中，operand streaming 與 compute pipeline 通常 overlap。若直接相加，會太悲觀。

例子：

```text
CONV input/weight read 花 100 cycles
CONV compute formula 250 cycles

engine total ~= 250 cycles
額外 wait = 250 - 100 = 150
```

如果 memory 花 300 cycles：

```text
engine total ~= 300 cycles
compute wait = 0
```

這個設計會在後面 cycle accuracy 章再深入。

---

## 4.22 UDMA 與 engine 的 overlap

UDMA 是獨立 `SC_THREAD`，CONV / EWE / POOL / Requant 也各自是獨立 thread。因此只要 dependency tag 允許，它們可以在 SystemC time 上 overlap。

例如：

```text
tile 0 CONV 正在跑
tile 1 UDMA read input 可以同時跑
tile -1 UDMA write output 也可能背景跑
```

這取決於 descriptor DAG：

| DAG 寫法 | 結果 |
|---|---|
| tile 1 load wait tile 0 store | 太保守，overlap 少 |
| tile 1 load 只 wait L1 buffer safe tag | overlap 較多 |
| compute 不 wait load | functional hazard |
| store 不 wait requant | output wrong |

memory hierarchy 的效能不是單看 L1 / DRAM bandwidth，還要看 Command Engine 是否把工作排得出來。

---

## 4.23 L1 capacity 與 tiling

L1Mesh 只有 3 MB。大模型 layer 不可能把所有 input、weight、output 都同時放進 L1。

所以 compiler / scheduler 需要 tiling：

```text
把 tensor 拆成 tile
每個 tile 搬入 L1
compute
寫出或 handoff
重用 L1 buffer 給下一個 tile
```

一個 tile 至少要考慮：

| 區塊 | 佔用 |
|---|---|
| input tile | activation bytes |
| weight tile | weight bytes |
| output tile | output bytes |
| params blob | scale / bias / LUT |
| scratch / correction map | op-specific temporary |
| double buffer | 若要 overlap load/compute，可能需要兩份 |

junior 常犯錯是只算 input + output，忘記 weight 和 params。對 convolution，weight 有時比 activation tile 更大。

---

## 4.24 L1-resident handoff

如果 producer layer 的 output 可以留在 L1，consumer layer 直接讀 L1，就可以避免：

```text
producer output L1 -> DRAM store
consumer input DRAM -> L1 load
```

這是 performance 上很有價值的 optimization。

但它要求 compiler 正確管理 lifetime：

```text
producer output buffer 不能在 consumer 讀完前被覆蓋
```

handoff 問題常見於：

- multi-tile layer。
- concat / branch。
- depth-to-space / reshape 類 tail op。
- suppressed producer store。
- buffer ping-pong slot reuse。

如果 functional regression 出現：

```text
single tile PASS
multi tile FAIL
```

請優先懷疑 L1 buffer lifetime 或 store barrier。

---

## 4.25 Descriptor wait tag 如何保護 memory

Memory correctness 通常不是由 `L1Manager` 保護，而是由 descriptor dependency 保護。

例如：

```text
UDMA load input
  signal tag 10

CONV
  wait tag 10
```

這保證 CONV 不會在 input load 完成前讀 L1。

另一個例子：

```text
Requant writes L1 output
  signal tag 20

UDMA store output
  wait tag 20
```

這保證 store 不會在 output tensor 完成前讀 L1。

L1 buffer reuse 也需要 tag：

```text
UDMA store old tile from buffer A
  signal tag 30

UDMA load new tile into buffer A
  wait tag 30
```

如果缺了這種 wait，functional data 可能被覆蓋，`L1Mesh` 不會阻止你。scratchpad 的精神就是 compiler / scheduler 自己管理。

---

## 4.26 看 profile 時怎麼判讀 memory bottleneck

Profile 裡通常會有 per-engine busy time 或 Gantt-like timeline。你可以問：

| 問題 | 解讀 |
|---|---|
| UDMA read lane 很長嗎？ | input / weight load 可能是瓶頸 |
| UDMA write lane 很長嗎？ | output store 或 concat / d2s tail 可能重 |
| CONV lane 有空洞嗎？ | load 太晚、dependency 太保守、tiling 不佳 |
| DRAM-heavy layer 是否接近 bytes / 48？ | bandwidth 主導 |
| 很多小 UDMA descriptor 嗎？ | 16-cycle decode startup 和 row miss 可能累積 |
| L1 access 是否跨很多 small stride？ | bank conflict 或 fragmented access |

簡單估算：

```text
DRAM cycles ~= bytes / 48 + row_miss_penalty + refresh overhead
L1 cycles   ~= bytes / 256 * 1.46，若 sequential 且 bank conflict 少
UDMA extra  ~= 16 cycles per descriptor
```

這只是 first-order estimate，但足夠幫你判斷 cycle 是否離譜。

---

## 4.27 Activation Compression 在 memory hierarchy 的位置

當 profile 顯示 `UDMA_R` 長期 dominate，代表 DRAM→L1 的 activation load 可能比 compute 更重要。這時可以考慮在 memory hierarchy 加一個 ACT compression/decompression path。

最保守的設計是：

```text
DRAM compressed activation
  -> UDMA_R + ACT_DECOMP
  -> L1 raw NHWC tile
  -> CONV / EWE / POOL existing engines
```

這個設計有一個關鍵原則：**L1 裡仍然放 raw tensor**。所以 CONV 的 3x3 window、padding、stride、halo address 都不用知道 compression。Compression 只存在 DRAM storage format 與 UDMA_R/UDMA_W path 中。

如果一開始就做：

```text
L1 compressed activation -> CONV on-the-fly decompress
```

會馬上遇到幾個困難：

| 困難 | 原因 |
|---|---|
| random window access | CONV 會重複讀 halo row / overlapping window |
| block boundary | 3x3 window 可能跨 compressed block |
| stride / padding | address mapping 不再是簡單 NHWC offset |
| cycle model | 每個 MAC 讀 operand 前可能有 decompress latency |
| L1 lifetime | compressed block 和 raw window cache 都要管理 |

所以教材建議先做 DRAM compressed、L1 decompressed。

### 4.27.1 新增 UDMA mode 的想法

可以把 ACTC 建成 UDMA 的新 mode：

```text
UM_ACT_DECOMP_COPY:
  src_addr = DRAM compressed stream
  dst_addr = L1 raw tile
  idx_table_addr = block metadata table
  length = raw output bytes

UM_ACT_COMP_COPY:
  src_addr = L1 raw tile
  dst_addr = DRAM compressed stream
  idx_table_addr = block metadata table
  length = raw input bytes
```

Cycle model 至少要計：

| 項目 | 是否會變 |
|---|---|
| DRAM read bytes | 降低，讀 compressed bytes |
| metadata read bytes | 增加，讀 block table/header |
| L1 write bytes | 不變，寫 raw tile |
| ACT_DECOMP cycles | 增加，取決於 lanes / bytes per cycle |
| UDMA descriptor startup | 仍存在 |

因此 ACT compression 不是把 `dram_r` 直接除以 2。它是用較少 DRAM bytes 換一個新硬體 block 的 decompress latency。

### 4.27.2 Profile 上怎麼看值不值得做

先拆 `dram_r`：

```text
dram_r = activation read + weight read + params read + metadata/layout read
```

ACT compression 只影響 activation read。若 layer 的 DRAM read 主要是 weights，例如大 1x1 convolution 的 weight table，ACTC 幫助就小。若 layer 是高解析度、多 H tile、重複讀 input window，ACTC 才會有明顯收益。

判斷順序：

```text
1. UDMA_R 是否是 peak utilization？
2. top dram_r layer 的 bytes 是否主要來自 activation？
3. weight 是否已經 persistent？
4. fanout input 是否已經共用？
5. halo reload 是否還很多？
6. activation entropy 是否可能壓縮？
```

只有前幾個 scheduling / tiling 問題先排掉後，ACTC 的收益才比較乾淨。

---

## 4.28 Memory debug checklist

遇到 output mismatch，請照這個順序看：

| Step | 檢查 |
|---:|---|
| 1 | fail layer 的 input / output dtype byte width |
| 2 | UDMA `length` 是否等於預期 bytes |
| 3 | `src_addr` / `dst_addr` 是否在正確 address range |
| 4 | `src_stride` / `dst_stride` 是否用 bytes，不是 elements |
| 5 | compute descriptor 是否 wait load tags |
| 6 | store descriptor 是否 wait producer tags |
| 7 | L1 buffer reuse 是否 wait old consumer / store |
| 8 | concat / slice / d2s layout 是否符合 NHWC |
| 9 | INT16 / FP16 output 是否用 2 bytes |
| 10 | stream mode 是否讓 store / load 越序造成 overwrite |

遇到 performance regression，請照這個順序看：

| Step | 檢查 |
|---:|---|
| 1 | DRAM bytes 是否突然變多 |
| 2 | UDMA descriptor count 是否暴增 |
| 3 | L1-resident handoff 是否失效 |
| 4 | tiling 是否變碎，造成 CONV fill 重複付 |
| 5 | `wait_tags` 是否過度 serialize |
| 6 | UDMA write 是否擋住 UDMA read |
| 7 | L1 bank conflict 是否增加 |

---

## 4.29 一個手算例子：linear load

假設有一筆 UDMA read：

```text
src = DRAM
dst = L1
length = 98,304 bytes
```

粗估 DRAM：

```text
bytes / 48 = 2048 cycles
row miss = 50 cycles，至少第一個 row
refresh = 視時間點，可能 0 或多個 200 cycles
```

粗估 L1 write：

```text
bytes / 256 = 384 cycles
乘 SRAM/core ratio 1.46 -> 約 561 core cycles
```

UDMA decode：

```text
16 cycles
```

因為 UDMA `do_linear()` 是：

```text
read src -> write dst -> wait 16
```

所以這筆 descriptor 可能約：

```text
DRAM read 2098 + L1 write 561 + 16 = 2675 cycles
```

這是粗估。實際還會受 row locality、previous DRAM last_finish、L1 bank finish 影響。

---

## 4.30 一個手算例子：compute overlap

假設 CONV descriptor 需要：

```text
input L1 read = 200 cycles
weight L1 read = 500 cycles
compute formula = 1200 cycles
```

因為 model 用 max overlap：

```text
CONV total ~= max(memory elapsed, compute formula)
           ~= max(700, 1200)
           ~= 1200 cycles
```

如果 tiling 改變後：

```text
input L1 read = 200
weight L1 read = 1800
compute formula = 1200
```

那 CONV 變成 memory-dominated：

```text
CONV total ~= 2000 cycles
```

這能幫你理解為什麼有些 layer 增加算力不會變快，因為瓶頸在 memory。

---

## 4.31 常見誤解

| 誤解 | 正確理解 |
|---|---|
| L1 address 和 DRAM address 可以任意混用 | `L1Manager` 用 address range route，錯 range 會錯或報錯 |
| UDMA direction 決定讀寫哪種 memory | 真正讀寫由 `src_addr` / `dst_addr` range 決定；direction 是語意和 profile |
| L1 是無限快 | L1 有 bank latency、clock ratio、bank conflict |
| DRAM 只看 bytes / bandwidth | row miss、refresh、serialization 也會影響 |
| scratchpad 會自動避免 overwrite | 不會，必須靠 descriptor dependency 和 compiler lifetime 管理 |
| `length` 是 element count | UDMA 裡大多是 bytes，dtype 要自己換算 |
| CONV time = memory + compute | 目前 model 對 engine 內部採用 overlap，近似 max(memory, compute) |
| 很多小 UDMA 沒關係 | 每筆有 decode startup，也更容易 row miss |
| ACT compression 會讓 L1 也自動變小 | 若採 DRAM compressed / L1 decompressed，L1 footprint 不變，只省 DRAM bandwidth |

---

## 4.32 本章小結

MDLA7 memory hierarchy 的主線是：

```text
DRAM 大但慢
L1Mesh 小但快
UDMA 負責搬資料
descriptor tags 保護資料生命週期
tiling 決定 L1 是否裝得下與能否 overlap
```

你要特別記住：

1. L1Mesh 是 3 MB、4x4 NoC、16 banks、16-byte stripe，sequential peak 約 256 B/cycle。
2. DRAM 是 48 B/cycle，含 row miss 與 refresh，通常是大模型瓶頸之一。
3. UDMA 的功能正確仰賴 address、length、stride、wait tag 全部正確。
4. Scratchpad 不會自動保護資料，compiler / scheduler 必須管理 lifetime。
5. ACT compression 若採 DRAM compressed / L1 decompressed，可以先省 DRAM activation read，而不打擾 CONV/EWE/POOL 的 raw NHWC path。

下一章會進入 compute engines，看看 CONV、Requant、EWE、POOL 如何消費 descriptor body，並把 memory data 轉成 tensor output。

> 下一章 → [第 5 章 — Compute Engines Overview](05_compute_engines.md)
