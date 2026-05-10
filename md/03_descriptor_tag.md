# 第 3 章 — Descriptor ISA 與 Dependency Tag

> 上一章：[第 2 章 — HW Spec Top Architecture](02_hw_spec.md)

本章你會學到什麼：

- MDLA7 為什麼使用 descriptor-driven control。
- 一個 64-byte descriptor 如何拆成 common header 與 op-specific body。
- `signal_tag` 與 `wait_tags` 怎麼形成 dependency graph。
- Command Engine 如何 dispatch 到 CONV、Requant、EWE、POOL、UDMA。
- stream descriptor、lookahead、tail barrier 是為了處理什麼 performance 問題。
- junior 在 debug descriptor / tag 問題時該從哪裡下手。

---

## 3.1 先建立一個直覺

NPU simulator 不會像 CPU 一樣一行一行執行 C code。它比較像硬體 command processor：

```text
Host / compiler 產生 descriptor stream
        |
        v
Command Engine 檢查 wait_tags
        |
        +--> CONV Engine
        +--> Requant Engine
        +--> EWE Engine
        +--> POOL Engine
        +--> UDMA
```

Descriptor 是「工作單」。每張工作單都說：

| 問題 | Descriptor 欄位 |
|---|---|
| 要哪個 engine 做事？ | `op_class_subtype` |
| 資料型態是 INT8、INT16 還是 FP？ | `dtype` |
| 做完要發哪個完成訊號？ | `signal_tag` |
| 開始前要等哪些完成訊號？ | `wait_count`、`wait_tags` |
| 這個工作屬於哪一層、哪個 microblock？ | `layer_id`、`microblock_id` |
| 實際參數在哪裡？ | `body` union |

對 junior 來說，最重要的不是記欄位順序，而是理解這句話：

```text
Descriptor = engine command + dependency contract + profiling hint
```

其中 dependency contract 就是 `wait_tags` 與 `signal_tag`。

---

## 3.2 必讀檔案

本章對照這幾個檔案讀：

| 檔案 | 讀法 |
|---|---|
| [`descriptor.h`](../systemc/include/mdla7/descriptor.h) | 看 descriptor binary layout 與 enum 定義 |
| [`command_engine.h`](../systemc/include/mdla7/command_engine.h) | 看 dependency tag 如何被檢查、dispatch、完成 |
| [`test_model.cpp`](../systemc/src/test_model.cpp) | 看 compiler / test harness 如何產生 descriptor |
| [`conv_engine.h`](../systemc/include/mdla7/conv_engine.h) | 看 CONV descriptor body 如何被消費 |
| [`requant_engine.h`](../systemc/include/mdla7/requant_engine.h) | 看 Requant descriptor body 如何被消費 |
| [`ewe_pool.h`](../systemc/include/mdla7/ewe_pool.h) | 看 EWE / POOL descriptor body 如何被消費 |
| [`udma.h`](../systemc/include/mdla7/udma.h) | 看 UDMA descriptor body 如何被消費 |

讀 code 的順序建議：

1. 先讀 `DescriptorHeader`。
2. 再讀 `CommandEngine::dispatch()`。
3. 再讀 `CommandEngine::issue()`。
4. 最後讀各 engine 的 `run()`。

這樣你會先知道「工作如何被發出去」，再看「工作被收到後如何執行」。

---

## 3.3 Descriptor 的 64-byte 格式

MDLA7 descriptor 固定是 64 bytes：

```text
Descriptor
  Header: 16 bytes
  Body  : 48 bytes
```

source 裡用 `static_assert` 保證這些大小：

```cpp
static_assert(sizeof(DescriptorHeader) == 16, "header must be 16 bytes");
static_assert(sizeof(DescriptorBody) == 48, "body union must be 48 bytes");
static_assert(sizeof(Descriptor) == 64, "Descriptor must be 64 bytes");
```

為什麼要固定 64 bytes？

| 原因 | 說明 |
|---|---|
| 硬體 decode 簡單 | command FIFO 每筆固定長度，decoder 不必處理 variable length |
| host 產生容易 | compiler 只要填 struct，不需要複雜 packetizer |
| alignment 好處 | 64 bytes 對 cache line / burst / ring buffer 都友善 |
| body 可擴充 | header 固定，body 用 union 給不同 engine |

你可以把它想成硬體版的 function call：

```text
op_class = 呼叫哪個 function
body     = function arguments
wait     = 呼叫前要等的 event
signal   = function return 時要 set 的 event
```

---

## 3.4 Common Header 欄位

`DescriptorHeader` 是所有 op 共用的 16 bytes。

| 欄位 | 型態 | 意義 |
|---|---:|---|
| `op_class_subtype` | `uint8_t` | low 4 bits 是 op class，high 4 bits 是 subtype |
| `flags` | `uint8_t` | trace、chain、stream、tail 等 bit flags |
| `dtype` | `uint8_t` | INT8 / INT16 / FP dtype |
| `signal_tag` | `uint8_t` | 此 descriptor 完成時要 set 的 tag；0 表示不 signal |
| `wait_count` | `uint8_t` | 有幾個 dependency tag 要等待，最多 4 個 |
| `wait_tags[4]` | `uint8_t[4]` | dependency tag 清單 |
| `layer_id` | `uint16_t` | profiler / debug 用的 layer index |
| `microblock_id` | `uint16_t` | stream scheduler 的 microblock 順序 |
| `stream_slot` | `uint8_t` | ping-pong / slot hint |
| `stream_meta_flags` | `uint8_t` | load、compute、store、final tile 等 metadata |

`op_class_subtype` 的解碼方式在 header method：

```cpp
OpClass op_class() const {
    return static_cast<OpClass>(op_class_subtype & 0xF);
}

uint8_t op_subtype() const {
    return (op_class_subtype >> 4) & 0xF;
}
```

目前主要的 op class：

| enum | 數值 | Engine |
|---|---:|---|
| `OC_CONV` | 0 | CONV Engine |
| `OC_REQUANT` | 1 | Requant Engine |
| `OC_EWE` | 2 | Element-wise / activation / softmax |
| `OC_POOL` | 3 | Pool Engine |
| `OC_UDMA` | 4 | DMA |

初學時可以先忽略 `op_subtype()`，因為 EWE subtype 目前主要放在 `EweBody.subtype`。真正決定 dispatch 到哪個 engine 的是 `op_class()`。

---

## 3.5 DType：功能正確與 timing 都會用到

`dtype` 在 header 裡，看起來只是 metadata，但實際上會影響兩件事：

| 影響 | 說明 |
|---|---|
| functional path | INT8、INT16、FP 使用不同讀取型別與運算流程 |
| cycle model | CONV 的 bit-mult cycle 會根據 activation bits 和 weight bits 改變 |

目前 enum 包含：

| DType | 數值 | 大意 |
|---|---:|---|
| `DT_INT8x4` | 0 | 8-bit activation × 4-bit weight |
| `DT_INT8x8` | 1 | 8-bit activation × 8-bit weight |
| `DT_INT16x4` | 2 | 16-bit activation × 4-bit weight |
| `DT_INT16x8` | 3 | 16-bit activation × 8-bit weight |
| `DT_INT16x16` | 4 | 16-bit activation × 16-bit weight |
| `DT_FP8` | 8 | E4M3-style FP8 |
| `DT_FP16` | 9 | FP16 |
| `DT_BFP16` | 10 | BFP16 |

Command Engine dispatch 時只把 `DescriptorBody` 寫進 engine FIFO。body 本身沒有 `dtype` 欄位，所以目前 simulator 用 side-channel latch：

```text
CommandEngine sees descriptor header dtype
        |
        +--> writes last_dtype into target engine
        |
        +--> writes body to target engine FIFO
```

這段在 [`command_engine.h`](../systemc/include/mdla7/command_engine.h) 的 `issue()`：

```cpp
if (conv_dtype_latch) *conv_dtype_latch = d.hdr.dtype;
conv_cfg_out.write(d.body);
```

這是一個 simulator-friendly 的寫法。真實硬體通常會把 dtype 和 body 一起放在 command payload，或在 decoder 裡保留完整 header。

常見 bug：

| 現象 | 可能原因 |
|---|---|
| INT16 model output 全錯 | dtype 沒有正確 latch 到 CONV / Requant |
| FP layer 走到 INT path | dtype enum 或 `is_fp_dtype()` 沒接上 |
| Requant output byte width 錯 | `DT_INT16x8` / `DT_INT16x16` 沒被當成 int16 output |

---

## 3.6 Dependency tag 的核心語意

Dependency tag 是 8-bit ID，共 0 到 255。

在目前 simulator 裡：

- `tag_done[t] = true` 表示 tag `t` 已經完成。
- `tag_done[t] = false` 表示有工作還沒完成。
- `signal_tag = 0` 表示此 descriptor 完成時不發 tag。
- `wait_count = 0` 表示此 descriptor 可以立刻 issue。
- `wait_tags[]` 是 AND dependency，也就是所有 wait tag 都 done 才能 issue。

`waits_ready()` 的邏輯很直接：

```cpp
for (int w = 0; w < d.hdr.wait_count; ++w) {
    uint8_t tg = d.hdr.wait_tags[w];
    if (!tag_done[tg]) return false;
}
return true;
```

用圖表示：

```text
Descriptor A
  signal_tag = 7

Descriptor B
  wait_tags = [7]
  signal_tag = 8

執行順序：
  A issue -> A done -> tag 7 done -> B eligible -> B issue
```

如果有多個 wait tags：

```text
Descriptor C
  wait_tags = [3, 9, 12]

C 必須等 tag 3、9、12 都完成
```

這是 hardware scheduler 最常見的 primitive：用小小的 tag table 表示一個 DAG。

---

## 3.7 為什麼 tag_done 初始是 true

Command Engine constructor 裡：

```cpp
for (int i = 0; i < 256; ++i) tag_done[i] = true;
```

這代表「沒有被 active descriptor 佔用的 tag，都視為已完成」。

為什麼不是 false？

因為 descriptor 可能 wait 一個前面沒有出現過的 tag。若全部初始 false，任何誤填 wait tag 都會讓 simulation 卡死。初始 true 的語意是：

```text
只有被某個尚未完成的 descriptor reserve 的 tag，才是 not done。
```

這也帶來一個責任：當 descriptor 宣告 `signal_tag` 時，Command Engine 必須在正確時間把該 tag 標成 false，避免後面的 descriptor 太早通過。

---

## 3.8 Normal descriptor 的 tag reservation

Normal descriptor 是沒有 `DF_STREAM` flag 的 descriptor。它們保持 in-order issue。

對 normal descriptor，Command Engine 在 `issue()` 時才 reserve signal tag：

```cpp
if (!(d.hdr.flags & DF_STREAM) && d.hdr.signal_tag)
    tag_done[d.hdr.signal_tag] = false;
```

原因是 normal descriptor 不會越過前面的 normal descriptor，所以它只需要在真的發給 engine 前標記 pending。

這可以避免 8-bit tag wrap 的 hazard：

```text
tag 5 used by old descriptor
tag 5 later reused by new descriptor

如果新 descriptor 還只是在 pending queue 裡就把 tag 5 設 false，
可能會擾亂舊 descriptor 完成時的狀態。
```

因為 tag 只有 8 bits，compiler 必須小心不要讓同一個 tag 在太短距離內重用。Command Engine 也用 `LOOKAHEAD_LIMIT = 64` 控制 stream lookahead，不讓 pending window 大到超過合理 wrap distance。

---

## 3.9 Stream descriptor 的 tag reservation

Stream descriptor 有 `DF_STREAM` flag。它們可以在 lookahead window 裡被越序 issue。

對 stream descriptor，Command Engine 在 descriptor 進入 pending window 時就 reserve signal tag：

```cpp
if ((d.hdr.flags & DF_STREAM) && d.hdr.signal_tag)
    tag_done[d.hdr.signal_tag] = false;
```

為什麼 stream 要提早 reserve？

因為 stream descriptor 可以被越序。假設有一個較晚讀進 pending 的 stream descriptor `B`，後面某個 descriptor `C` 可能 wait `B.signal_tag`。如果 `B` 的 tag 還沒有被標成 false，`C` 可能誤以為 B 已完成，造成資料 hazard。

這裡的關鍵是：

| descriptor 類型 | issue 順序 | signal tag 何時 reserve |
|---|---|---|
| normal | in-order | issue 時 |
| stream | 可越序 | 進入 lookahead pending 時 |

這個差異很重要。讀 scheduler bug 時，先確認 descriptor 有沒有 `DF_STREAM`。

---

## 3.10 Dispatch loop 的 mental model

`CommandEngine::dispatch()` 大致分成三段：

```text
1. 從 desc_in 拉 descriptor 到 pending deque
2. 在 pending 裡挑一個 waits_ready 的最佳 descriptor
3. issue 後從 pending 移除
```

簡化 pseudo code：

```cpp
while (true) {
    fill_pending_until_lookahead_limit();

    best = find_ready_descriptor();

    if (best) {
        issue(best);
        remove(best);
        continue;
    }

    wait(new_descriptor_or_tag_changed);
}
```

這裡有兩個 SystemC event 很重要：

| event | 何時發生 |
|---|---|
| `desc_in.data_written_event()` | Host / upstream 寫入新 descriptor |
| `tag_changed` | 某個 engine done，Command Engine set tag_done true |

如果 pending 是空的，Command Engine 只等新 descriptor。

如果 pending 不空但都還不能 issue，Command Engine 同時等：

```text
新 descriptor 進來
或
某個 wait tag 完成
```

這就是 event-driven simulator 的味道：沒有工作時不空轉，時間由 engine 的 wait 與 memory latency 推進。

---

## 3.11 Normal in-order 與 stream bypass

Dispatch loop 有一個很重要的規則：

```text
Only descriptors explicitly marked as stream-pipeline work may be bypassed.
Normal descriptors keep in-order issue.
```

原因是 normal schedule 很可能重用固定 L1 address。如果讓 normal descriptor 隨便越序，就可能出現：

```text
store old tile 還沒完成
load new tile 覆蓋同一塊 L1
compute 讀到錯資料
```

所以 normal descriptor 的規則保守：

```text
pending front 如果 ready，就 issue
pending front 如果不 ready，後面的 normal descriptor 不看
```

stream descriptor 則不同。stream schedule 會透過 `layer_id`、`microblock_id`、`stream_slot`、`stream_meta_flags` 提供更多結構化資訊，讓 Command Engine 可以更積極 overlap：

```text
load tile N+1
compute tile N
store tile N-1
```

這就是 tile pipeline 的基本形狀。

---

## 3.12 Stream issue priority

stream descriptor ready 之後，Command Engine 用 priority 選誰先 issue。

目前 priority 概念如下：

| 工作 | priority 越小越優先 | 原因 |
|---|---:|---|
| `DF_STREAM_TAIL` | 0 | tail / barrier 要盡快清掉，避免整個 pipeline 收尾卡住 |
| EWE | 10 | element-wise compute 越早開始，越能和後續 UDMA overlap |
| UDMA read | 20 | DRAM 到 L1 的 load 會餵 compute |
| CONV | 30 | 主 compute |
| Requant / POOL | 40 | 通常跟 compute chain 或 tensor output 有關 |
| UDMA write | 60 | store 可以背景化，不能阻擋 younger load |

實作中還把 `microblock_id` 加進 tie-breaker：

```cpp
return base * 4096 + int(d.hdr.microblock_id);
```

這代表同一類 priority 裡，較小 microblock 大致先跑。

為什麼 UDMA write priority 比 UDMA read 低？

因為 output store 通常不是下一個 compute 的 immediate input。若 store 佔住 scheduling front，可能讓下一個 tile 的 input / weight load 太晚開始，降低 overlap。

---

## 3.13 Tail barrier 與 allowed_during_tail_wait

`DF_STREAM_TAIL` 用來標記 tail / barrier 類工作。

常見例子是：

```text
某些 L1-resident handoff 需要在真正覆蓋 buffer 前補一個 store barrier
```

當有 tail descriptor 在 pending 裡但還沒 ready，Command Engine 不是完全停住。它允許某些 younger work 先做：

| 允許類型 | 原因 |
|---|---|
| UDMA read | 可以先把後續 tile 的 input / weight 搬進 L1 |
| CONV / Requant 且有 `SMF_COMPUTE` | 允許淺層 compute front overlap |

這段邏輯在 `allowed_during_tail_wait()`：

```cpp
if (d.hdr.op_class() == OC_UDMA && d.body.udma.direction == 0)
    return true;

if ((d.hdr.op_class() == OC_CONV || d.hdr.op_class() == OC_REQUANT)
    && (d.hdr.stream_meta_flags & SMF_COMPUTE))
    return true;
```

這是 performance 和 correctness 的折衷：

```text
不要讓 tail 等待把整個 pipe 停死，
但也不要放任所有 younger store / arbitrary work 越過安全邊界。
```

---

## 3.14 完成訊號如何回到 Command Engine

每個 engine 做完工作後會寫 `done_tag_out`：

```cpp
done_tag_out.write(0);
```

payload 目前固定是 0，真正的 tag 不是 engine 回傳的，而是 Command Engine 自己用 per-engine queue 對應。

`issue()` 時：

```cpp
pending_tags[d.hdr.op_class()].push(d.hdr.signal_tag);
```

`collect()` 收到某個 engine done 後：

```cpp
uint8_t t = pending_tags[cls].front();
pending_tags[cls].pop();
tag_done[t] = true;
tag_changed.notify();
```

這代表 simulator 假設：

```text
同一個 engine 內部完成順序與 Command Engine issue 到該 engine 的順序一致。
```

目前每個 engine 都是一個 `SC_THREAD(run)`，每次 `cfg_in.read()` 後執行到 done，所以這個假設成立。

如果未來某個 engine 內部支援 out-of-order completion，就不能只用 FIFO pairing。那時 done payload 必須帶回 task ID 或 signal tag。

---

## 3.15 Descriptor body union

Header 決定 dispatch，body 決定 engine 具體要做什麼。

`DescriptorBody` 是 union：

```cpp
union DescriptorBody {
    ConvBody    conv;
    RequantBody requant;
    EweBody     ewe;
    PoolBody    pool;
    UdmaBody    udma;
    uint8_t     raw[48];
};
```

各 body 固定 48 bytes：

| Body | 用途 |
|---|---|
| `ConvBody` | input / weight address、shape、kernel、stride、pad、group、params address |
| `RequantBody` | chain output shape、output address、requant params、correction map |
| `EweBody` | element-wise / softmax / activation input-output address、shape、LUT / params |
| `PoolBody` | pooling input-output shape、kernel、stride、padding、mode |
| `UdmaBody` | copy mode、direction、address、length、stride、slice metadata |

body 是 union，所以你一定要根據 `op_class()` 解讀它。用錯 body 讀欄位，數值會看起來很怪，debug 時不要被騙。

---

## 3.16 ConvBody 導讀

`ConvBody` 的重點欄位：

| 欄位 | 意義 |
|---|---|
| `in_addr` | input tensor 在 L1 / DRAM address space 的位址 |
| `wgt_addr` | weight tensor 位址 |
| `out_addr` | 保留 / trace 用；目前 CONV functional output 走 chain 到 Requant |
| `in_h`、`in_w`、`in_c` | input shape |
| `out_c` | output channel count |
| `k_h`、`k_w` | kernel size |
| `stride_dilation` | stride / dilation packed field |
| `pad_tb`、`pad_lr` | top/bottom/left/right padding |
| `group` | group convolution / depthwise support |
| `cluster_mask` | active cluster hint |
| `in_pad_value` | padding value，量化模型通常是 `zp_in` |
| `bias_addr` | bias 位址；目前實際 bias 多在 Requant params fold |
| `scale_lut_addr` | requant / scale parameter blob 位址 |
| `scale_count` | per-channel scale 數量 |

`stride_dilation` 是 packed encoding：

```text
[1:0] = stride_h
[3:2] = stride_w
[5:4] = dilation_h
[7:6] = dilation_w
```

目前 `decode_stride()` 使用 log2 encoding：

| encoding | stride |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |

讀 CONV descriptor 時，請先自己算一次 output shape：

```text
out_h = (in_h + pad_t + pad_b - k_h) / stride_h + 1
out_w = (in_w + pad_l + pad_r - k_w) / stride_w + 1
```

如果 log 裡 output shape 和你期待不同，第一個檢查 `pad_tb`、`pad_lr`、`stride_dilation`。

---

## 3.17 RequantBody 導讀

Requant 是 CONV 後面很重要的 sink，也是 CONV / EWE 共用的 quantize-pack / clamp resource。CONV 把 int32 / FP32 partial sum 推進 128-lane functional chain（4096 bit/cyc），Requant timing model 以 512 elem/cycle 估算後段 MBQM / clamp / pack，把它轉成 output tensor。

`RequantBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `out_addr` | requant 後 tensor 寫到哪裡 |
| `n`、`h`、`w`、`c` | 這次 dispatch 要處理的 output shape |
| `scale_lut_addr` | requant parameter blob |
| `scale_count` | full layer 的 channel count |
| `oc_start` | 這個 OC tile 從第幾個 output channel 開始 |
| `out_w_layer` | full layer output width，correction map index 用 |
| `oh_start` | 此 tile 的 global output row offset |
| `corr_addr` | asymmetric weight zero-point correction map |
| `corr_per_oc` | correction map 是否 per output channel |

對 junior 來說，Requant 最容易混淆的是：

```text
c         = this dispatch 的 channel count
scale_count = full layer 的 channel count
oc_start     = this dispatch 在 full layer channel 裡的 offset
```

這三個值在 OC tiling 時不一樣。若 `oc_start` 錯，通常會看到某些 channel 整片 wrong，而不是隨機 noise。

---

## 3.18 EweBody 導讀

EWE 是 element-wise engine。它支援：

| subtype | 功能 |
|---|---|
| `ES_SOFTMAX` | softmax |
| `ES_ADD` | element-wise add |
| `ES_MUL` | element-wise multiply |
| `ES_SUB` | element-wise subtract |
| `ES_HARD_SWISH` | unary hard swish |
| `ES_GELU` | unary gelu |

`EweBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `in_a_addr` | 第一個 input |
| `in_b_addr` | 第二個 input；unary op 時可為 0 |
| `out_addr` | output 位址 |
| `n`、`h`、`w`、`c` | tensor shape |
| `broadcast_axes` | broadcast metadata，目前支援程度要看 compiler path |
| `reduce_axes` | reduce metadata |
| `scalar_imm` | scalar immediate |
| `lut_addr` | params / LUT 位址 |
| `subtype` | EWE subtype |

EWE 的 cycle model 依 dtype 選 lanes：

| dtype | EWE lanes |
|---|---:|
| INT8 | 64 |
| INT16 | 32 |
| FP | 32 |

binary op 通常：

```text
cycles ~= ceil(elements / lanes)
```

softmax 是三 pass：

```text
cycles ~= 3 * ceil(elements / lanes)
```

功能上，FP path 使用 FP16 storage、FP32 compute；INT path 使用量化參數 blob。

---

## 3.19 PoolBody 導讀

POOL 支援 max、average、global：

| enum | 功能 |
|---|---|
| `PM_MAX` | max pool |
| `PM_AVG` | average pool |
| `PM_GLOBAL` | global average pool 類型 |

`PoolBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `in_addr`、`out_addr` | input / output 位址 |
| `in_n`、`in_h`、`in_w`、`in_c` | input shape |
| `out_n`、`out_h`、`out_w`、`out_c` | output shape |
| `mode` | max / avg / global |
| `k_h`、`k_w` | kernel size |
| `stride` | stride_h / stride_w packed field |
| `pad_tb`、`pad_lr` | padding |
| `count_include_pad` | avg pool 是否把 padding 算進 divisor |

POOL 的 stride decode 也支援 1、2、4、8。這對某些模型的 global / large stride pooling 很重要。

POOL 的 cycle model 跟 EWE 使用同一套 dtype lanes：

| dtype | POOL lanes |
|---|---:|
| INT8 | 64 |
| INT16 | 32 |
| FP | 32 |

```text
cycles ~= ceil(out_elements / lanes) * max(k_h * k_w, 1)
```

---

## 3.20 UdmaBody 導讀

UDMA descriptor 用來搬資料或做部分 data layout transform。

目前 modes：

| mode | 功能 |
|---|---|
| `UM_LINEAR_COPY` | 連續 memory copy |
| `UM_STRIDED_2D` | 逐 row copy，source / destination stride 可不同 |
| `UM_INDEXED_GATHER` | 根據 index table gather |
| `UM_SCATTER_CONCAT` | concat 多個 source 到連續 destination |
| `UM_STRIDED_SLICE` | 2D slice |
| `UM_DEPTH_TO_SPACE` | NHWC depth-to-space transform |

`UdmaBody` 重點欄位：

| 欄位 | 意義 |
|---|---|
| `mode` | UDMA mode |
| `direction` | 0 = DRAM 到 L1，1 = L1 到 DRAM |
| `src_addr`、`dst_addr` | source / destination |
| `length` | copy bytes；某些 mode 表示每 row bytes 或 element bytes |
| `src_stride`、`dst_stride` | stride bytes |
| `num_chunks` | rows / chunks / input rows |
| `idx_table_addr` | gather / concat metadata table |
| `slice_begin`、`slice_end` | slice 或 depth-to-space metadata |

UDMA descriptor 最常見的 debug 問題是 address space：

| address range | 意義 |
|---|---|
| `0x00000000` 到 `0x001FFFFF` | L1Mesh 2 MB |
| `0x10000000` 以上 | DRAM |

如果 direction 是 DRAM 到 L1，通常期待：

```text
src_addr in DRAM
dst_addr in L1
```

如果 direction 是 L1 到 DRAM，通常期待：

```text
src_addr in L1
dst_addr in DRAM
```

但 UDMA implementation 其實透過 `L1Manager` 讀寫，source / destination 只要 address space 合法就能工作。direction 主要用於 trace / profiling 分 lane。

---

## 3.21 Descriptor 產生：test_model.cpp 的 helper

Descriptor 不是手寫的。現在主要由 [`test_model.cpp`](../systemc/src/test_model.cpp) 的 helper 產生。

常見 helper：

| helper | 用途 |
|---|---|
| `alloc_tag()` | 分配新的 dependency tag |
| `make_desc()` | 建立 common header |
| `make_udma()` | 建立 UDMA descriptor |
| `make_pool()` | 建立 POOL descriptor |
| `make_softmax()` | 建立 EWE softmax descriptor |
| `make_ewe_add()` | 建立 EWE ADD descriptor |
| `make_ewe_unary()` | 建立 unary activation descriptor |
| `mark_stream()` | 把 descriptor 標成 stream work 並填 stream metadata |
| `make_store_barrier()` | 建立小型 UDMA store barrier |

`make_desc()` 的概念：

```text
填 op_class
填 dtype
填 signal_tag
填 wait_count / wait_tags
其他 header metadata 之後再補
```

`alloc_tag()` 通常會遞增 tag ID。因為 tag 是 8-bit，compiler 需要避免 live range 太長導致 wrap hazard。

讀 program generation 時請畫出：

```text
descriptor index
op class
wait tags
signal tag
L1 / DRAM address
```

很多 functional fail 都可以用這張表找到第一個可疑點。

---

## 3.22 一個典型 CONV layer 的 descriptor chain

一個普通 convolution layer 可能長這樣：

```text
UDMA read input
  signal tag = 11

UDMA read weight
  signal tag = 12

CONV
  wait tags  = [11, 12]
  signal tag = 13

REQUANT
  wait tags  = [13]
  signal tag = 14

UDMA write output
  wait tags  = [14]
  signal tag = 15
```

用 dependency DAG 表示：

```text
input load  ----\
                 +--> CONV --> Requant --> store
weight load ----/
```

如果 layer 有 tiling，就會有多組類似 chain：

```text
tile 0: load -> conv -> requant -> store
tile 1: load -> conv -> requant -> store
tile 2: load -> conv -> requant -> store
```

stream scheduler 的目標是讓它變成 overlap：

```text
time --->

UDMA read:   tile0 load   tile1 load   tile2 load
CONV:                    tile0 conv   tile1 conv   tile2 conv
Requant:                              tile0 req    tile1 req
UDMA write:                                      tile0 store  tile1 store
```

這種 overlap 成功與否，會反映在 profile Gantt 和 per-engine busy time。

---

## 3.23 L1-resident handoff 與 store barrier

有些 layer 不一定每次都把 output 立刻 store 回 DRAM。如果 producer output 可以直接留在 L1 給 consumer 用，稱為 L1-resident handoff。

優點：

| 優點 | 說明 |
|---|---|
| 少一次 DRAM write | output 不必先寫回 off-chip |
| 少一次 DRAM read | 下一層 input 不必再從 off-chip 讀回 |
| performance 好 | DRAM bandwidth pressure 降低 |

風險：

```text
如果 buffer 被下一個 tile 或下一層覆蓋，
而某個 consumer / store 還沒有讀完，
就會發生資料被提早覆蓋。
```

所以 code 裡有 `make_store_barrier()` 這類 helper，用一個很小的 UDMA store descriptor 當作 ordering barrier。它不一定代表真正大量資料搬運，有時更像「讓 dependency graph 有一個可等待的完成點」。

debug 這類問題時，要找：

| 要看什麼 | 為什麼 |
|---|---|
| `prev_store` tag | 上一個 store 是否真的被等待 |
| `suppress_producer_store` | 是否開啟 L1-resident handoff |
| `single_tile_layer` | 單 tile 與多 tile hazard 不同 |
| `make_store_barrier()` | 是否補了保護性 ordering |

---

## 3.24 Descriptor log 怎麼看

Command Engine dispatch 時會印：

```text
[CmdEng] dispatch op_class=... layer_id=... signal_tag=... wait_count=...
```

stream descriptor 還會印：

```text
slot=... mb=... smeta=0x...
```

engine 完成時會印：

```text
[CmdEng] engine 4 done; tag 12 set
```

讀 log 的方法：

1. 先找 fail layer 的 `layer_id`。
2. 找該 layer 的第一個 dispatch。
3. 把每個 descriptor 的 `signal_tag` 記下來。
4. 檢查等待的 tag 是否真的在前面完成。
5. 如果 pending 卡住，找最後一個沒有 set 的 tag。
6. 如果 output wrong，檢查 store 是否太早或太晚。

一個簡單 trace 表：

| 順序 | op | wait | signal | 解讀 |
|---:|---|---|---|---|
| 0 | UDMA read input | none | 10 | input loaded |
| 1 | UDMA read weight | none | 11 | weight loaded |
| 2 | CONV | 10, 11 | 12 | compute after both loads |
| 3 | Requant | 12 | 13 | output after conv |
| 4 | UDMA store | 13 | 14 | store after requant |

如果看到：

```text
Requant wait tag 不包含 CONV signal tag
```

那通常就是嚴重 correctness bug。

---

## 3.25 Debug：descriptor / tag 問題常見症狀

| 症狀 | 可能原因 |
|---|---|
| simulation 卡住 | 某個 wait tag 永遠沒有被 signal |
| output 大量 wrong | consumer 太早 issue，讀到未完成資料 |
| 只有多 tile model wrong | store / L1 overwrite ordering 問題 |
| stream mode 才 wrong | `DF_STREAM` reservation 或 priority 造成 hazard |
| 長模型後段 wrong | 8-bit tag wrap / reuse 太近 |
| INT16 model wrong | dtype latch 或 output byte width |
| FP model wrong | FP path 沒進對 subtype 或 params blob layout 錯 |

排查順序：

1. 確認 descriptor count 是否符合預期。
2. 確認每個 wait tag 都有 upstream signal。
3. 確認 signal tag 沒有在 live range 內重用。
4. 確認 UDMA direction 與 address range。
5. 確認 `layer_id` 和 fail layer 對得上。
6. 若只有 stream fail，檢查 `stream_meta_flags`。
7. 若只有 particular dtype fail，檢查 `dtype` latch。

---

## 3.26 Cycle accuracy 與 descriptor 的關係

Command Engine 目前主要是 functional scheduler，不太加入額外 dispatch cycle cost。simulation time 主要來自：

| 來源 | 在哪裡 wait |
|---|---|
| CONV compute cycles | `ConvEngine::run()` |
| Requant lane cycles | `RequantEngine::run()` |
| EWE lane cycles | `EweEngine::run()` |
| POOL lane cycles | `PoolEngine::run()` |
| L1 bank latency | `L1Mesh::read()` / `write()` |
| DRAM bandwidth / row miss / refresh | `Dram::read()` / `write()` |
| UDMA decode startup | `Udma::wait_bytes()` |

Descriptor 影響 cycle accuracy 的地方：

| Descriptor 設計 | Cycle 影響 |
|---|---|
| tiling 拆得越細 | CONV tile fill latency 會重複付 |
| stream scheduling | 可增加 overlap，降低 wall time |
| UDMA direction | read / write profile lane 分開 |
| wait tags 太保守 | engine idle 增加 |
| wait tags 太鬆 | functional wrong，cycle 再漂亮也沒用 |

所以 performance tuning 不是只改 engine cycles。你也要看 descriptor DAG 是否讓 engine 有機會 overlap。

---

## 3.27 Junior 練習：手畫一個 descriptor DAG

找一個小模型或小 layer，從 log 裡抽出 5 到 10 個 descriptor，做一張表：

| idx | op_class | layer_id | wait_tags | signal_tag | address 摘要 |
|---:|---|---:|---|---:|---|
| 0 | UDMA | 3 | none | 21 | DRAM input -> L1 |
| 1 | UDMA | 3 | none | 22 | DRAM weight -> L1 |
| 2 | CONV | 3 | 21, 22 | 23 | L1 input / weight |
| 3 | REQUANT | 3 | 23 | 24 | chain -> L1 output |
| 4 | UDMA | 3 | 24 | 25 | L1 output -> DRAM |

然後問自己：

- 哪些工作理論上可以平行？
- 哪個 wait tag 讓 CONV 開始？
- 哪個 wait tag 讓 store 開始？
- 如果 store 被延後，下一層會不會受影響？
- 如果 input load 被延後，CONV 會 idle 嗎？

這個練習比直接讀 1000 行 scheduler code 更有效。

---

## 3.28 常見誤解

| 誤解 | 正確理解 |
|---|---|
| tag 是 engine 回傳的 | 目前 engine 只回 done pulse，tag 由 Command Engine 的 pending queue 對應 |
| `signal_tag = 0` 是 tag 0 完成 | 在此格式中 0 表示 no signal |
| 所有 descriptor 都可以越序 | 只有 `DF_STREAM` descriptor 可 bypass |
| `wait_count = 0` 表示等 tag 0 | 它表示沒有 dependency |
| body 裡會帶 dtype | dtype 在 header，engine 用 side-channel latch |
| UDMA direction 決定 address 合法性 | address 合法性由 L1 / DRAM range 決定；direction 主要是語意與 profiling |
| cycle fail 一定是 engine cycle formula | descriptor DAG 太保守或太鬆也會造成 timing / correctness 問題 |

---

## 3.29 本章小結

Descriptor 是 MDLA7 最核心的控制介面。它把 compiler、scheduler、SystemC module、profile 全部接在一起。

本章要記住三件事：

1. `op_class` 決定送去哪個 engine，`body` 是該 engine 的參數。
2. `wait_tags` / `signal_tag` 是 dependency DAG，錯了就會卡住或讀錯資料。
3. `DF_STREAM` descriptor 允許 lookahead bypass，所以 tag reservation 和 priority 都比 normal descriptor 複雜。

你讀後面章節時，請一直把 descriptor 當成主線。看到任何 engine code，都問：

```text
它消費哪一種 body？
它何時 done？
它產生的資料被誰用？
它的 signal tag 被誰 wait？
```

這樣就不會迷路。

> 下一章 → [第 4 章 — Memory Hierarchy：DRAM、UDMA、L1Mesh](04_memory_hierarchy.md)
