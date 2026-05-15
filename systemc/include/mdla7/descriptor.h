#pragma once

// MDLA7 descriptor format — see spec §3A.10
// 64 byte total: 16 byte common header + 48 byte op-specific body union.

#include <cstdint>
#include <array>
#include <ostream>
#include <type_traits>

namespace mdla7 {

enum OpClass : uint8_t {
    OC_CONV    = 0,
    OC_REQUANT = 1,
    OC_EWE     = 2,
    OC_POOL    = 3,
    OC_TNPS    = 4,
    OC_UDMA    = 5,
    OC_NUM     = 6,
};

enum DType : uint8_t {
    DT_INT8x4  = 0,
    DT_INT8x8  = 1,
    DT_INT16x4 = 2,
    DT_INT16x8 = 3,
    DT_INT16x16= 4,
    DT_FP8     = 8,   // E4M3
    DT_FP16    = 9,   // E5M10
    DT_BFP16   = 10,  // E8M7
};

enum UdmaMode : uint8_t {
    UM_LINEAR_COPY    = 0,
    UM_STRIDED_2D     = 1,
    UM_INDEXED_GATHER = 2,
    UM_SCATTER_CONCAT = 3,
    UM_STRIDED_SLICE  = 4,
    UM_DEPTH_TO_SPACE = 5,
    UM_ACT_DECOMP_COPY = 6,
    UM_ACT_COMP_COPY   = 7,
    UM_ACT_DECOMP_STREAM_HEAD = 8,
    UM_ACT_DECOMP_STREAM_TAIL = 9,
};

enum TnpsMode : uint8_t {
    TM_LINEAR_COPY    = 0,
    TM_STRIDED_2D     = 1,
    TM_INDEXED_GATHER = 2,
    TM_SCATTER_CONCAT = 3,
    TM_STRIDED_SLICE  = 4,
    TM_DEPTH_TO_SPACE = 5,
    TM_SPACE_TO_DEPTH = 6,
    TM_TRANSPOSE      = 7,
    TM_CHANNEL_PACK   = 8,
};

enum PoolMode : uint8_t {
    PM_MAX    = 0,
    PM_AVG    = 1,
    PM_GLOBAL = 2,
    PM_SUM    = 3,   // v13: sum-reduce without division (softmax decomposition)
};

enum EngineId : uint8_t {
    EID_HOST      = 0,
    EID_CONV      = 1,
    EID_REQUANT   = 2,
    EID_EWE       = 3,
    EID_POOL      = 4,
    EID_TNPS      = 5,
    EID_UDMA      = 6,
    EID_L1MANAGER = 7,
    EID_L1MESH    = 8,
};

enum PayloadOpcode : uint8_t {
    PL_READ_REQ  = 0,
    PL_READ_RESP = 1,
    PL_WRITE_REQ = 2,
    PL_WRITE_ACK = 3,
};

constexpr unsigned PAYLOAD_BYTES = 16;
constexpr unsigned CONV_REQUANT_CHAIN_LANES = 128;
constexpr unsigned CONV_REQUANT_CHAIN_BITS_PER_CYCLE =
    CONV_REQUANT_CHAIN_LANES * 32;

struct Payload {
    uint8_t engineid;
    uint8_t tid;
    uint8_t opcode;
    uint8_t last;
    uint32_t addr;
    std::array<uint8_t, PAYLOAD_BYTES> data;
};
static_assert(sizeof(Payload) == 24, "Payload must be 24 bytes");

struct PayloadPortCount {
    static constexpr unsigned REQUANT_R = 8;
    static constexpr unsigned REQUANT_W = 8;
    static constexpr unsigned EWE_R = 32;   // 512 B/cyc (×2 to unblock read after ×4 compute)
    static constexpr unsigned EWE_W = 32;   // 512 B/cyc (×4 to match ×4 compute)
    static constexpr unsigned POOL_R = 32;  // 512 B/cyc (×2 to match ×2 compute)
    static constexpr unsigned POOL_W = 16;  // 256 B/cyc (×2)
    static constexpr unsigned TNPS_R = 8;
    static constexpr unsigned TNPS_W = 8;
    static constexpr unsigned L1MESH_R = 16;
    static constexpr unsigned L1MESH_W = 16;
    static constexpr unsigned CONV_ACT_R = 32;
    static constexpr unsigned CONV_WGT_R = 32;
};

// EWE engine subtypes (v6) — distinguishes softmax vs binary ADD vs other elwise ops.
// Stored in EweBody.subtype; avoids needing a new dispatch tag.
// v8.30: extended with MUL / SUB binary ops + HARD_SWISH / GELU unary activations.
enum EweSubtype : uint8_t {
    ES_SOFTMAX    = 0,
    ES_ADD        = 1,
    ES_MUL        = 2,    // element-wise binary multiply
    ES_SUB        = 3,    // element-wise binary subtract
    ES_HARD_SWISH = 4,    // unary: x * relu6(x+3) / 6
    ES_GELU       = 5,    // unary: x * Φ(x), tanh-approximation
    ES_LOGISTIC   = 6,    // unary: sigmoid(x); INT8 path uses 256-byte LUT
    ES_RSQRT      = 7,    // unary: 1/sqrt(x), INT8 LUT-based
    ES_TANH       = 8,    // unary: tanh(x), INT8 LUT-based
    ES_EXP        = 9,    // v13: unary exp(x), INT8 LUT-based / FP analytical
    ES_DIV        = 10,   // v13: binary a/b, broadcast over last axis; FP-only
    ES_DEQUANT_INT8 = 11, // v13: cast INT8 -> FP16, (in - scalar_imm) (softmax decomp)
    ES_QUANT_FP_INT8 = 12,// v13: cast FP16 -> INT8, round(in * 256) + scalar_imm clamped
};

// Descriptor flags.
constexpr uint8_t DF_PREEMPT      = 0x01;
constexpr uint8_t DF_CHAIN_SRC    = 0x02;
constexpr uint8_t DF_CHAIN_SINK   = 0x04;
constexpr uint8_t DF_TRACE        = 0x08;
constexpr uint8_t DF_STREAM       = 0x10;  // participates in CommandEngine lookahead bypass.
constexpr uint8_t DF_STREAM_TAIL  = 0x20;  // tail/barrier work; later safe prefetch may bypass.

// Tile-command metadata for stream descriptors.  These bytes are advisory for
// scheduling/trace only; engines still consume the 48-byte op body unchanged.
enum StreamMetaFlags : uint8_t {
    SMF_NONE       = 0,
    SMF_LOAD_A     = 1 << 0,
    SMF_LOAD_B     = 1 << 1,
    SMF_COMPUTE    = 1 << 2,
    SMF_STORE      = 1 << 3,
    SMF_FINAL_TILE = 1 << 4,
};

enum RequantStoreMode : uint8_t {
    RQ_STORE_LINEAR = 0,
    RQ_STORE_D2SPACE = 1,
    RQ_STORE_SKIP = 2,
    RQ_STORE_STRIDED_2D = 3,
};

enum ConvDataflowMode : uint8_t {
    CONV_DF_OS = 0,
    CONV_DF_WS = 1,
};

// 16 bytes — common header.
struct DescriptorHeader {
    uint8_t  op_class_subtype;   // [3:0]=op_class, [7:4]=op_subtype
    uint8_t  flags;              // DF_* bits above.
    uint8_t  dtype;              // DType enum
    uint8_t  signal_tag;         // 0 = no signal
    uint8_t  wait_count;         // 0..4
    uint8_t  wait_tags[4];
    uint16_t layer_id;
    uint16_t microblock_id;      // TileCommand microblock id within layer/tile.
    uint8_t  stream_slot;        // TileCommand slot/ping-pong id (trace/scheduler hint).
    uint8_t  stream_meta_flags;  // SMF_* bits above.

    OpClass op_class()   const { return static_cast<OpClass>(op_class_subtype & 0xF); }
    uint8_t op_subtype() const { return (op_class_subtype >> 4) & 0xF; }
};
static_assert(sizeof(DescriptorHeader) == 16, "header must be 16 bytes");

// CONV body — 48 bytes.
struct ConvBody {
    uint32_t in_addr;
    uint32_t wgt_addr;
    uint32_t out_addr;
    uint16_t in_h, in_w, in_c, out_c;
    uint8_t  k_h;            // 1..11
    uint8_t  k_w;            // 1..11
    uint8_t  stride_dilation;// CONV stride: [3:0]=s_h, [7:4]=s_w direct, 0=>16
    uint8_t  pad_tb;         // [2:0]=pad_t, [5:3]=pad_b
    uint8_t  pad_lr;         // [2:0]=pad_l, [5:3]=pad_r
    uint8_t  _r0;
    uint16_t group;
    uint16_t cluster_mask;   // 16 cluster active bits
    int16_t  in_pad_value;   // v7: padding value (= zp_in for asymmetric int8 input; 0 = TFLite-default)
    uint32_t bias_addr;          // 0 = no bias
    uint32_t scale_lut_addr;
    uint16_t scale_count;
    uint8_t  _r2[6];
};
static_assert(sizeof(ConvBody) == 48, "ConvBody must be 48 bytes");

struct RequantBody {
    uint32_t in_addr, out_addr;
    uint16_t n, h, w, c;            // c = this dispatch's OC slice (== scale_count if no OC tile)
    uint32_t scale_lut_addr;
    uint16_t scale_count;           // total OC channels packed in the params blob (i.e. layer's full OC)
    uint8_t  per_channel_flag;
    uint8_t  shift_global;
    int16_t  zp_global;
    uint16_t oc_start;              // v7: OC-tile offset into params blob (0 = no tiling)
    uint16_t out_w_layer;           // v7: layer's full OW (for indexing into per-pixel corr map)
    uint16_t oh_start;              // v7: OH-tile offset (global oh of this tile's first row)
    uint32_t corr_addr;             // v7: per-pixel correction map (0 = none)
    uint8_t  corr_per_oc;           // v7: 1 = corr is shape [OH, OW, OC_layer] (dwconv); 0 = [OH, OW]
    uint8_t  _r[11];
};
static_assert(sizeof(RequantBody) == 48, "RequantBody must be 48 bytes");

struct EweBody {
    uint32_t in_a_addr, in_b_addr, out_addr;     // in_b_addr=0 => unary op
    uint16_t n, h, w, c;
    uint8_t  broadcast_axes;
    uint8_t  reduce_axes;
    int16_t  scalar_imm;
    uint32_t lut_addr;
    uint8_t  subtype;                            // v6: EweSubtype (0=softmax, 1=add)
    uint8_t  _r[19];
};
static_assert(sizeof(EweBody) == 48, "EweBody must be 48 bytes");

struct PoolBody {
    uint32_t in_addr, out_addr;
    uint16_t in_n, in_h, in_w, in_c;
    uint16_t out_n, out_h, out_w, out_c;
    uint8_t  mode;            // PoolMode
    uint8_t  k_h, k_w;
    uint8_t  stride;          // [1:0]=s_h, [3:2]=s_w (enum 0=>1, 1=>2)
    uint8_t  pad_tb;          // [2:0]=pad_t, [5:3]=pad_b
    uint8_t  pad_lr;          // [2:0]=pad_l, [5:3]=pad_r
    uint8_t  count_include_pad;
    uint8_t  _r[17];
};
static_assert(sizeof(PoolBody) == 48, "PoolBody must be 48 bytes");

struct UdmaBody {
    uint8_t  mode;            // UdmaMode
    uint8_t  direction;       // 0: DRAM->L1, 1: L1->DRAM
    uint16_t _r0;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t length;          // bytes
    uint32_t src_stride;
    uint32_t dst_stride;
    uint16_t num_chunks;
    uint16_t _r1;
    uint32_t idx_table_addr;
    uint16_t slice_begin[4];
    uint16_t slice_end[4];
};
static_assert(sizeof(UdmaBody) == 48, "UdmaBody must be 48 bytes");

struct TnpsBody {
    uint8_t  mode;            // TnpsMode
    uint8_t  direction;       // reserved; mirrors UdmaBody for helper reuse
    uint16_t _r0;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t length;          // bytes or element bytes, mode-dependent
    uint32_t src_stride;
    uint32_t dst_stride;
    uint16_t num_chunks;
    uint16_t _r1;
    uint32_t idx_table_addr;
    uint16_t slice_begin[4];
    uint16_t slice_end[4];
};
static_assert(sizeof(TnpsBody) == 48, "TnpsBody must be 48 bytes");

union DescriptorBody {
    ConvBody    conv;
    RequantBody requant;
    EweBody     ewe;
    PoolBody    pool;
    TnpsBody    tnps;
    UdmaBody    udma;
    uint8_t     raw[48];
};
static_assert(sizeof(DescriptorBody) == 48, "body union must be 48 bytes");

struct Descriptor {
    DescriptorHeader hdr;
    DescriptorBody   body;
};
static_assert(sizeof(Descriptor) == 64, "Descriptor must be 64 bytes");

// SystemC's sc_fifo<T> requires operator<< for sc_trace. We don't actually
// trace these aggregate types, so just provide stream stubs.
inline std::ostream& operator<<(std::ostream& os, const DescriptorBody&) {
    return os << "<DescriptorBody>";
}
inline std::ostream& operator<<(std::ostream& os, const Descriptor& d) {
    return os << "<Descriptor op_class=" << int(d.hdr.op_class()) << ">";
}
inline std::ostream& operator<<(std::ostream& os, const Payload& p) {
    return os << "<Payload engineid=" << int(p.engineid)
              << " tid=" << int(p.tid)
              << " opcode=" << int(p.opcode)
              << " addr=0x" << std::hex << p.addr << std::dec
              << " last=" << int(p.last) << ">";
}

// Address-space helpers — see spec §3A.10
constexpr uint32_t L1MESH_BASE  = 0x0000'0000;
constexpr uint32_t L1MESH_END   = 0x002F'FFFF;          // 3 MB (spec §3A.10)
constexpr uint32_t L1MESH_BYTES = L1MESH_END + 1;
constexpr uint32_t DRAM_BASE    = 0x0030'0000;
constexpr uint32_t DRAM_END     = 0xFFFF'FFFF;          // 4 GB high half

inline bool addr_in_l1mesh(uint32_t a) { return a <= L1MESH_END; }
inline bool addr_in_dram  (uint32_t a) { return a >= DRAM_BASE; }

} // namespace mdla7
