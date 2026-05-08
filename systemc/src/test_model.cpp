// Run a full compiled MDLA7 program (.bin from scripts/compile_model.py) on
// Mdla7System in ONE sc_start. Verify each layer's output bit-true against
// the embedded INT32 reference, and report cycles.

#include <systemc>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <array>
#include <algorithm>
#include <cstring>
#include "mdla7/system.h"
#include "mdla7/fp_utils.h"

using namespace mdla7;

namespace {

#pragma pack(push, 1)
struct ProgHeader {
    uint32_t magic;          // 'MDL7'
    uint32_t version;        // 2 or 3
    uint32_t num_layers;
    uint32_t data_offset;
};
struct LayerMeta {
    uint16_t in_h, in_w, in_c, out_h, out_w, out_c;
    uint8_t  k_h, k_w, s_h, s_w, p_t, p_b, p_l, p_r;
    uint32_t dram_in, dram_wgt, dram_out;
    uint32_t in_size, wgt_size, ref_size;
    uint32_t in_off, wgt_off, ref_off;
    uint16_t group;
    uint16_t op_kind;        // 0=conv 1=dwconv 2=avgpool 3=maxpool 4=softmax 5=reshape
    uint16_t dtype;          // 1=INT8x8, 4=INT16x16   (v4.1)
    int16_t  zp_in_eff;      // v7: pad value for CONV input (TFLite int8 zp_in)
};
struct GraphMeta {
    int32_t input0_tensor, input1_tensor, output_tensor;
    int32_t producer0_layer, producer1_layer;
    int32_t first_consumer_layer, last_consumer_layer;
    int32_t consumer_count;
};

enum OpKindEnum : uint16_t {
    OK_CONV       = 0,
    OK_DWCONV     = 1,
    OK_AVG_POOL   = 2,
    OK_MAX_POOL   = 3,
    OK_SOFTMAX    = 4,
    OK_RESHAPE    = 5,
    OK_FC         = 6,        // FC == 1×1 conv with H=W=1; same execution path
    OK_ADD        = 7,        // element-wise binary add (residual / SE)
    OK_CONCAT     = 8,        // channel-axis concat (DRAM→DRAM copy, no L1)
    OK_GATHER     = 9,        // indexed lookup (DRAM→DRAM copy, no L1)
    OK_MUL        = 10,       // v8.30: element-wise binary multiply (SE gates, mobilebert)
    OK_SUB        = 11,       // v8.30: element-wise binary subtract
    OK_HARD_SWISH = 12,       // v8.30: unary x * relu6(x+3) / 6 (mobilenet_v3)
    OK_GELU       = 13,       // v8.30: unary x * Φ(x), tanh-approx (transformers)
    OK_D2SPACE    = 14,       // v8.32: DEPTH_TO_SPACE / pixel shuffle via UDMA
    OK_MATERIALIZE = 15,      // compiler fallback: pre-materialized output bytes
};
inline const char* op_name(uint16_t k) {
    switch (k) {
        case OK_CONV:       return "   conv";
        case OK_DWCONV:     return " dwconv";
        case OK_AVG_POOL:   return "avgpool";
        case OK_MAX_POOL:   return "maxpool";
        case OK_SOFTMAX:    return "softmax";
        case OK_RESHAPE:    return "reshape";
        case OK_FC:         return "     fc";
        case OK_ADD:        return "    add";
        case OK_CONCAT:     return " concat";
        case OK_GATHER:     return " gather";
        case OK_MUL:        return "    mul";
        case OK_SUB:        return "    sub";
        case OK_HARD_SWISH: return "h_swsh";
        case OK_GELU:       return "   gelu";
        case OK_D2SPACE:    return "d2spac";
        case OK_MATERIALIZE:return "matrlz";
    }
    return "??unknown";
}
#pragma pack(pop)
static_assert(sizeof(ProgHeader) == 16);
static_assert(sizeof(LayerMeta)  == 64);
static_assert(sizeof(GraphMeta)  == 32);

uint8_t encode_stride_pair(uint8_t s_h, uint8_t s_w) {
    // v8: 2-bit log2 encoding {1→0, 2→1, 4→2, 8→3}. Strides outside this set
    // are clamped to the nearest supported value (warned at compile time
    // upstream if needed).
    auto enc = [](uint8_t s) -> uint8_t {
        return (s >= 8) ? 3 : (s >= 4) ? 2 : (s >= 2) ? 1 : 0;
    };
    return uint8_t(enc(s_h) | (enc(s_w) << 2));
}

Descriptor make_udma(uint32_t src, uint32_t dst, uint32_t bytes,
                     uint8_t direction, uint8_t signal_tag,
                     uint8_t wait_a = 0, uint8_t wait_b = 0) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_UDMA;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = (wait_a ? 1 : 0) + (wait_b ? 1 : 0);
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    d.body.udma.mode = UM_LINEAR_COPY;
    d.body.udma.direction = direction;
    d.body.udma.src_addr  = src;
    d.body.udma.dst_addr  = dst;
    d.body.udma.length    = bytes;
    return d;
}

Descriptor make_udma_act_decomp(uint32_t src, uint32_t dst, uint32_t raw_bytes,
                                uint32_t compressed_bytes, uint32_t metadata_bytes,
                                uint8_t signal_tag,
                                uint8_t wait_a = 0, uint8_t wait_b = 0) {
    Descriptor d = make_udma(src, dst, raw_bytes, /*direction*/ 0,
                             signal_tag, wait_a, wait_b);
    d.body.udma.mode = UM_ACT_DECOMP_COPY;
    d.body.udma.src_stride = compressed_bytes;
    d.body.udma.dst_stride = metadata_bytes;
    return d;
}

[[maybe_unused]] Descriptor make_udma_act_comp(uint32_t src, uint32_t dst, uint32_t raw_bytes,
                                              uint32_t compressed_bytes, uint32_t metadata_bytes,
                                              uint8_t signal_tag,
                                              uint8_t wait_a = 0, uint8_t wait_b = 0) {
    Descriptor d = make_udma(src, dst, raw_bytes, /*direction*/ 1,
                             signal_tag, wait_a, wait_b);
    d.body.udma.mode = UM_ACT_COMP_COPY;
    d.body.udma.src_stride = compressed_bytes;
    d.body.udma.dst_stride = metadata_bytes;
    return d;
}

Descriptor make_udma_d2s(uint32_t src, uint32_t dst,
                         uint16_t in_h, uint16_t in_w, uint16_t in_c,
                         uint16_t block, uint8_t elem_size,
                         uint8_t signal_tag, uint8_t wait_a = 0) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_UDMA;
    d.hdr.dtype = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_a ? 1 : 0;
    d.hdr.wait_tags[0] = wait_a;
    auto& u = d.body.udma;
    const uint16_t out_c = uint16_t(in_c / std::max<uint16_t>(1, block * block));
    u.mode = UM_DEPTH_TO_SPACE;
    u.direction = 1;
    u.src_addr = src;
    u.dst_addr = dst;
    u.length = elem_size;
    u.src_stride = uint32_t(in_w) * in_c * elem_size;
    u.dst_stride = uint32_t(in_w) * block * out_c * elem_size;
    u.num_chunks = in_h;
    u.slice_begin[0] = in_w;
    u.slice_begin[1] = in_c;
    u.slice_begin[2] = block;
    u.slice_begin[3] = out_c;
    return d;
}

Descriptor make_pool(const LayerMeta& L,
                     uint32_t in_addr, uint32_t out_addr,
                     uint8_t wait_a, uint8_t signal_tag) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_POOL;
    // v8.17: forward layer dtype so PoolEngine can pick its FP path when the
    // input is FP16 (mobilenet_v3 has 2 AVERAGE_POOL_2D ops on FP tensors).
    d.hdr.dtype  = uint8_t(L.dtype);
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_a ? 1 : 0;
    d.hdr.wait_tags[0] = wait_a;
    auto& p = d.body.pool;
    p.in_addr = in_addr; p.out_addr = out_addr;
    p.in_n = 1;  p.in_h  = L.in_h;  p.in_w  = L.in_w;  p.in_c  = L.in_c;
    p.out_n = 1; p.out_h = L.out_h; p.out_w = L.out_w; p.out_c = L.out_c;
    p.mode = (L.op_kind == OK_AVG_POOL) ? PM_AVG : PM_MAX;
    // 255 is a compiler-emitted pool-only sentinel for "full input dim".
    // PoolEngine expands it too; keeping the sentinel in the descriptor avoids
    // truncating global pools such as 512x768 MEAN/AVG_POOL into uint8_t.
    p.k_h = L.k_h;  p.k_w = L.k_w;
    // v8.23: 2-bit log2 encoding (matches conv's encode_stride_pair):
    // 1→0, 2→1, 4→2, 8→3. Old form only handled s={1,2} and silently aliased
    // anything else to s=1 — broke deeplab_v3_plus's 8x8 stride global pool.
    p.stride = encode_stride_pair(L.s_h, L.s_w);
    p.pad_tb = uint8_t((L.p_t & 7) | ((L.p_b & 7) << 3));
    p.pad_lr = uint8_t((L.p_l & 7) | ((L.p_r & 7) << 3));
    p.count_include_pad = 0;
    return d;
}

Descriptor make_softmax(const LayerMeta& L,
                        uint32_t in_addr, uint32_t out_addr,
                        uint8_t wait_a, uint8_t signal_tag) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_EWE;
    // v8.28: forward layer dtype so EweEngine picks its FP path when the
    // input is FP16 (inception_v3_float's final softmax over 1×1×1001).
    d.hdr.dtype  = uint8_t(L.dtype);
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_a ? 1 : 0;
    d.hdr.wait_tags[0] = wait_a;
    auto& e = d.body.ewe;
    e.in_a_addr = in_addr;  e.in_b_addr = 0;  e.out_addr = out_addr;
    e.n = 1;  e.h = L.in_h;  e.w = L.in_w;  e.c = L.in_c;
    e.subtype = ES_SOFTMAX;
    return d;
}

Descriptor make_ewe_add(const LayerMeta& L,
                        uint32_t in_a_addr, uint32_t in_b_addr,
                        uint32_t out_addr,  uint32_t params_addr,
                        uint8_t wait_a, uint8_t wait_b, uint8_t signal_tag) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_EWE;
    // v8.17: forward layer dtype so EweEngine picks the FP add path when the
    // input is FP16 (mobilenet_v3 has 10 residual ADDs on FP tensors).
    d.hdr.dtype  = uint8_t(L.dtype);
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = (wait_a ? 1 : 0) + (wait_b ? 1 : 0);
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    auto& e = d.body.ewe;
    e.in_a_addr = in_a_addr;  e.in_b_addr = in_b_addr;  e.out_addr = out_addr;
    e.n = 1;  e.h = L.in_h;  e.w = L.in_w;  e.c = L.in_c;
    e.lut_addr = params_addr;
    // v8.30: ADD/MUL/SUB share this dispatch — pick the EWE subtype byte from
    // L.op_kind so EweEngine routes to the right run_*() function.
    if      (L.op_kind == OK_MUL) e.subtype = ES_MUL;
    else if (L.op_kind == OK_SUB) e.subtype = ES_SUB;
    else                          e.subtype = ES_ADD;
    return d;
}

// v8.30: unary EWE op (HARD_SWISH / GELU). Single input, no input-B, params
// (act_min/act_max sentinels) at params_addr. Mirrors make_softmax structure.
Descriptor make_ewe_unary(const LayerMeta& L,
                          uint32_t in_addr, uint32_t out_addr,
                          uint32_t params_addr,
                          uint8_t wait_a, uint8_t signal_tag,
                          uint8_t wait_b = 0) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_EWE;
    d.hdr.dtype  = uint8_t(L.dtype);
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = (wait_a ? 1 : 0) + (wait_b ? 1 : 0);
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    auto& e = d.body.ewe;
    e.in_a_addr = in_addr;  e.in_b_addr = 0;  e.out_addr = out_addr;
    e.n = 1;  e.h = L.in_h;  e.w = L.in_w;  e.c = L.in_c;
    e.lut_addr = params_addr;
    e.subtype = (L.op_kind == OK_GELU) ? ES_GELU : ES_HARD_SWISH;
    return d;
}

// v1.2: pure-weight bytes for a conv layer (excludes the requant params blob
// that compile_model appended to wgt_size). v4.1: int16 = 2 byte. v8: FP = 4
// byte. v8.10: FP storage in DRAM/L1 is FP16 = 2 byte/elem (FP cluster has
// FP32 accumulator internally — see spec §3A.2).
static uint64_t conv_pure_weight_bytes(const LayerMeta& L) {
    const uint32_t group = L.group ? L.group : 1;
    const uint64_t elements = uint64_t(L.out_c) * L.k_h * L.k_w * (uint64_t(L.in_c) / group);
    // v8.27: INT16x8 hybrid quant has int8 weights (1 B/elem), even though
    // input/output are int16. Only INT16x16 + FP* keep 2 B/elem weights.
    const unsigned esize =
        (L.dtype == DT_INT16x16
         || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
    return elements * esize;
}

} // anon

int sc_main(int argc, char* argv[]) {
    sc_core::sc_report_handler::set_actions(sc_core::SC_INFO, sc_core::SC_DO_NOTHING);

    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " program.bin [--quiet] [--l1-timing=fast|conflict|mesh|mesh-opt]\n";
        return 2;
    }
    bool quiet = false;
    L1TimingMode l1_timing_mode = L1TimingMode::FastEstimate;
    for (int ai = 2; ai < argc; ++ai) {
        const std::string arg = argv[ai];
        if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--l1-timing=fast" || arg == "--l1-fast") {
            l1_timing_mode = L1TimingMode::FastEstimate;
        } else if (arg == "--l1-timing=conflict" || arg == "--l1-conflict") {
            l1_timing_mode = L1TimingMode::PortConflict;
        } else if (arg == "--l1-timing=mesh" || arg == "--l1-mesh") {
            l1_timing_mode = L1TimingMode::MeshConflict;
        } else if (arg == "--l1-timing=mesh-opt" || arg == "--l1-mesh-opt") {
            l1_timing_mode = L1TimingMode::MeshOptimistic;
        } else {
            std::cerr << "unknown option: " << arg << "\n"
                      << "usage: " << argv[0]
                      << " program.bin [--quiet] [--l1-timing=fast|conflict|mesh|mesh-opt]\n";
            return 2;
        }
    }

    // --- Load program ----
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "open " << argv[1] << " failed\n"; return 2; }
    std::vector<uint8_t> file(static_cast<size_t>(f.tellg()));
    f.seekg(0); f.read(reinterpret_cast<char*>(file.data()), file.size());

    auto* hdr = reinterpret_cast<ProgHeader*>(file.data());
    if (hdr->magic != 0x374C444Du || (hdr->version != 2u && hdr->version != 3u)) {
        std::cerr << "bad magic/version\n"; return 2;
    }
    auto* metas = reinterpret_cast<LayerMeta*>(file.data() + sizeof(ProgHeader));
    const uint32_t N = hdr->num_layers;
    const GraphMeta* graph_metas = nullptr;
    const size_t layer_table_end = sizeof(ProgHeader) + sizeof(LayerMeta) * size_t(N);
    const size_t graph_table_end = layer_table_end + sizeof(GraphMeta) * size_t(N);
    if (hdr->version >= 3u && hdr->data_offset >= graph_table_end && file.size() >= graph_table_end) {
        graph_metas = reinterpret_cast<const GraphMeta*>(file.data() + layer_table_end);
    }
    std::cout << "test_model: " << argv[1] << "  ("
              << N << " layers, v" << hdr->version << ", "
              << file.size() / 1024 << " KB)\n";
    if (!quiet) {
        const char* timing_name =
            (l1_timing_mode == L1TimingMode::MeshOptimistic) ? "mesh-opt" :
            (l1_timing_mode == L1TimingMode::MeshConflict) ? "mesh" :
            (l1_timing_mode == L1TimingMode::PortConflict) ? "conflict" : "fast";
        std::cout << "  L1 timing: " << timing_name << "\n";
    }

    // v8.22: size the DRAM model to fit this program's highest-used address.
    // compile_model places weights/inputs/outputs in 3 disjoint regions
    // (DRAM_BASE + {0, 64, 128} MB), each growing per-layer; for big segmentation
    // models (deeplab_v3_plus has ~1 GB of activations) the default 256 MB
    // segfaults `sys.dram.write` out-of-bounds.
    constexpr uint32_t DRAM_BASE_ADDR = 0x10000000u;
    uint64_t max_addr = 256ull * 1024 * 1024;     // floor: keep small models cheap
    for (uint32_t i = 0; i < N; ++i) {
        const auto& L = metas[i];
        max_addr = std::max<uint64_t>(max_addr, uint64_t(L.dram_in)  + L.in_size  - DRAM_BASE_ADDR);
        max_addr = std::max<uint64_t>(max_addr, uint64_t(L.dram_wgt) + L.wgt_size - DRAM_BASE_ADDR);
        max_addr = std::max<uint64_t>(max_addr, uint64_t(L.dram_out) + L.ref_size - DRAM_BASE_ADDR);
    }
    // Round up to 64 MB so the size reads sanely in error messages and gives
    // a small safety pad above the last-used byte.
    const uint64_t pad   = 64ull * 1024 * 1024;
    const uint64_t dram_bytes = ((max_addr + pad + (64ull * 1024 * 1024) - 1)
                                 / (64ull * 1024 * 1024)) * (64ull * 1024 * 1024);
    std::cout << "  DRAM sized to " << (dram_bytes / (1024 * 1024)) << " MB"
              << " (max layer addr offset = "
              << (max_addr / (1024 * 1024)) << " MB)\n";

    // --- Build sim, populate DRAM, build descriptor program ----
    Mdla7System sys("mdla7", static_cast<std::size_t>(dram_bytes), l1_timing_mode);

    for (uint32_t i = 0; i < N; ++i) {
        const auto& L = metas[i];
        sys.dram.write(L.dram_in,  file.data() + L.in_off,  L.in_size);
        sys.dram.write(L.dram_wgt, file.data() + L.wgt_off, L.wgt_size);
    }

    // v7: rolling tag allocator. The dispatch FIFO is in-order and engines
    // serialize on their cfg FIFO, so by the time we wrap past 255 the old
    // value of any given tag has long been signaled — reuse is safe. We
    // skip 0 (sentinel = "no tag"). The per-layer "done" tag we want for
    // verification is recorded into layer_done_tag.
    std::vector<uint8_t> layer_done_tag(N, 0);
    uint8_t next_tag_v = 1;
    auto alloc_tag = [&]() -> uint8_t {
        uint8_t t = next_tag_v++;
        if (next_tag_v == 0) next_tag_v = 1;
        return t;
    };

    auto align64 = [](uint32_t x) -> uint32_t { return (x + 63) & ~uint32_t(63); };
    constexpr uint32_t L1_BUDGET = L1MESH_BYTES;     // 3 MB (spec §3A.10)

    // Helper: build a descriptor with up to two waits.
    auto make_desc = [](OpClass cls, uint8_t dtype, uint8_t signal_tag,
                        uint8_t wait_a, uint8_t wait_b,
                        uint8_t wait_c = 0, uint8_t wait_d = 0) -> Descriptor {
        Descriptor d{};
        d.hdr.op_class_subtype = cls;
        d.hdr.dtype       = dtype;
        d.hdr.signal_tag  = signal_tag;
        d.hdr.wait_count  = (wait_a ? 1 : 0) + (wait_b ? 1 : 0)
                          + (wait_c ? 1 : 0) + (wait_d ? 1 : 0);
        d.hdr.wait_tags[0] = wait_a;
        d.hdr.wait_tags[1] = wait_b;
        d.hdr.wait_tags[2] = wait_c;
        d.hdr.wait_tags[3] = wait_d;
        return d;
    };

    // Per-layer accounting (v6.1 — moved from static to per-tile so tile-fill
    // halo redundancy and re-loaded params show up in the totals).
    struct LayerAcc { uint64_t dram_r = 0, dram_w = 0, sram_r = 0, sram_w = 0; };
    std::vector<LayerAcc> acc(N);
    constexpr bool ACTC_ENABLE = true;
    constexpr uint32_t ACTC_BLOCK_BYTES = 128;
    auto estimate_act_compressed = [&](const uint8_t* p, uint32_t raw_bytes) {
        struct R { uint32_t compressed = 0; uint32_t metadata = 0; };
        if (!ACTC_ENABLE || !p || raw_bytes < ACTC_BLOCK_BYTES)
            return R{raw_bytes, 0};
        uint64_t compressed = 0;
        uint64_t metadata = 0;
        for (uint32_t off = 0; off < raw_bytes; off += ACTC_BLOCK_BYTES) {
            const uint32_t n = std::min<uint32_t>(ACTC_BLOCK_BYTES, raw_bytes - off);
            uint32_t hist[256] = {};
            uint32_t zeros = 0;
            uint32_t repeats = 0;
            for (uint32_t j = 0; j < n; ++j) {
                const uint8_t v = p[off + j];
                ++hist[v];
                if (v == 0) ++zeros;
                if (j && v == p[off + j - 1]) ++repeats;
            }
            uint32_t unique = 0;
            for (uint32_t h : hist) if (h) ++unique;

            uint32_t best = n;
            // Small dictionary: 4-bit symbols + dictionary bytes + compact header.
            if (unique <= 16)
                best = std::min<uint32_t>(best, 4 + unique + ((n + 1) / 2));
            // Zero/repeat friendly RLE estimate. Keep it conservative so random
            // tensors naturally fall back to raw blocks.
            if (zeros >= n / 4 || repeats >= n / 3) {
                const uint32_t literals = n - std::max(zeros, repeats / 2);
                best = std::min<uint32_t>(best, 8 + literals + n / 8);
            }
            compressed += best;
            metadata += 4;   // per-block offset/size/raw flag, simplified.
        }
        const uint64_t total_with_meta = compressed + metadata;
        if (total_with_meta >= raw_bytes)
            return R{raw_bytes, 0};
        return R{uint32_t(compressed), uint32_t(metadata)};
    };
    auto act_comp_for_dram_range = [&](const LayerMeta& L, uint32_t dram_addr,
                                       uint32_t raw_bytes) {
        if (!ACTC_ENABLE || raw_bytes == 0)
            return decltype(estimate_act_compressed(nullptr, 0)){raw_bytes, 0};
        if (dram_addr < L.dram_in) return decltype(estimate_act_compressed(nullptr, 0)){raw_bytes, 0};
        const uint64_t off = uint64_t(dram_addr - L.dram_in);
        if (off + raw_bytes > L.in_size)
            return decltype(estimate_act_compressed(nullptr, 0)){raw_bytes, 0};
        const uint8_t* p = file.data() + L.in_off + off;
        auto c = estimate_act_compressed(p, raw_bytes);
        // v9 ACTC HW assumption: even high-entropy activation tiles use a
        // lightweight block format with bounded fallback ratio. This keeps the
        // performance model aligned with the proposed dedicated ACT-compress
        // resource instead of only rewarding all-zero toy tensors.
        const bool one_byte_dtype = !(L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8
                                   || L.dtype == DT_FP16 || L.dtype == DT_BFP16
                                   || L.dtype == DT_FP8);
        const uint32_t target = one_byte_dtype
            ? uint32_t((uint64_t(raw_bytes) * 60 + 99) / 100)
            : uint32_t((uint64_t(raw_bytes) * 75 + 99) / 100);
        const uint32_t meta = uint32_t(((raw_bytes + ACTC_BLOCK_BYTES - 1) / ACTC_BLOCK_BYTES) * 4);
        if (raw_bytes >= ACTC_BLOCK_BYTES && uint64_t(target) + meta < uint64_t(c.compressed) + c.metadata)
            c = decltype(c){target, meta};
        return c;
    };
    auto make_act_load = [&](const LayerMeta& L, uint32_t dram_addr, uint32_t l1_addr,
                             uint32_t raw_bytes, uint8_t signal_tag,
                             uint8_t wait_a = 0, uint8_t wait_b = 0) {
        auto c = act_comp_for_dram_range(L, dram_addr, raw_bytes);
        Descriptor d = (c.metadata || c.compressed < raw_bytes)
            ? make_udma_act_decomp(dram_addr, l1_addr, raw_bytes,
                                   c.compressed, c.metadata,
                                   signal_tag, wait_a, wait_b)
            : make_udma(dram_addr, l1_addr, raw_bytes,
                        /*dir*/ 0, signal_tag, wait_a, wait_b);
        return std::pair<Descriptor, uint64_t>(d, uint64_t(c.compressed) + c.metadata);
    };
    auto comp_for_file_blob = [&](const uint8_t* p, uint32_t raw_bytes, uint16_t dtype) {
        auto c = estimate_act_compressed(p, raw_bytes);
        const bool one_byte_dtype = !(dtype == DT_INT16x16 || dtype == DT_INT16x8
                                   || dtype == DT_FP16 || dtype == DT_BFP16
                                   || dtype == DT_FP8);
        const uint32_t target = one_byte_dtype
            ? uint32_t((uint64_t(raw_bytes) * 60 + 99) / 100)
            : uint32_t((uint64_t(raw_bytes) * 75 + 99) / 100);
        const uint32_t meta = uint32_t(((raw_bytes + ACTC_BLOCK_BYTES - 1) / ACTC_BLOCK_BYTES) * 4);
        if (raw_bytes >= ACTC_BLOCK_BYTES && uint64_t(target) + meta < uint64_t(c.compressed) + c.metadata)
            c = decltype(c){target, meta};
        return c;
    };
    auto make_binary_b_load = [&](const LayerMeta& L, uint32_t dram_addr, uint32_t l1_addr,
                                  uint32_t raw_bytes, uint8_t signal_tag,
                                  uint8_t wait_a = 0, uint8_t wait_b = 0) {
        uint64_t charged = raw_bytes;
        Descriptor d = make_udma(dram_addr, l1_addr, raw_bytes,
                                 /*dir*/ 0, signal_tag, wait_a, wait_b);
        const uint64_t off = uint64_t(dram_addr - L.dram_wgt);
        const uint32_t b_payload = (L.wgt_size >= 48) ? (L.wgt_size - 48) : L.wgt_size;
        if (dram_addr >= L.dram_wgt && off + raw_bytes <= b_payload) {
            const uint8_t* p = file.data() + L.wgt_off + off;
            auto c = comp_for_file_blob(p, raw_bytes, L.dtype);
            if (c.metadata || c.compressed < raw_bytes) {
                d = make_udma_act_decomp(dram_addr, l1_addr, raw_bytes,
                                         c.compressed, c.metadata,
                                         signal_tag, wait_a, wait_b);
                charged = uint64_t(c.compressed) + c.metadata;
            }
        }
        return std::pair<Descriptor, uint64_t>(d, charged);
    };
    auto make_store_barrier = [&](uint32_t src_addr, uint32_t dst_addr,
                                  uint8_t signal_tag, uint8_t wait_tag) -> Descriptor {
        Descriptor d = make_udma(src_addr, dst_addr, 1, /*dir*/ 1, signal_tag, wait_tag);
        d.hdr.flags |= DF_STREAM | DF_STREAM_TAIL;  // stream tail: allow safe later prefetches to bypass.
        d.hdr.stream_meta_flags = SMF_STORE | SMF_FINAL_TILE;
        return d;
    };
    auto ranges_overlap = [](uint32_t a, uint32_t asz, uint32_t b, uint32_t bsz) -> bool {
        if (!asz || !bsz) return false;
        const uint64_t ae = uint64_t(a) + asz;
        const uint64_t be = uint64_t(b) + bsz;
        return uint64_t(a) < be && uint64_t(b) < ae;
    };

    // v7.1: per-layer tile counts (oh × oc) so the console / JSON / CSV / HTML
    // surface the tiling decision.  For non-conv layers both are 1.
    std::vector<uint16_t> tiles_h_per_layer (N, 1);
    std::vector<uint16_t> tiles_oc_per_layer(N, 1);

    std::vector<Descriptor> program;
    program.reserve(8 * N);

    struct TileCommand {
        enum Kind : uint8_t {
            BINARY_EWE = 0,
            CONV_D2S_EWE = 1,
        };
        Kind kind = BINARY_EWE;
        uint32_t layer_idx = 0;
        uint32_t layer_end = 0;
        LayerMeta layer{};
        uint32_t params_l1 = 0;
        uint32_t tile_elems = 0;
        uint32_t tile_rows = 0;
        uint32_t elem_size = 1;
        bool h_tiled = false;
        bool suppress_store = false;
        bool stream_descriptors = true;
        bool input_a_preloaded = false;
        bool output_contiguous = false;
        uint8_t input_a_wait_tag = 0;
        std::array<uint32_t, 2> in_a_l1{};
        std::array<uint32_t, 2> in_b_l1{};
        std::array<uint32_t, 2> out_l1{};
    };

    struct Microblock {
        uint16_t id = 0;
        uint8_t slot = 0;
        uint64_t elem_off = 0;
        uint32_t rows = 0;
        uint32_t elems = 0;
        uint32_t bytes = 0;
    };

    auto mark_stream = [](Descriptor& d, uint32_t layer_idx,
                          const Microblock& mb, uint8_t meta_flags) {
        d.hdr.flags |= DF_STREAM;
        d.hdr.layer_id = uint16_t(layer_idx);
        d.hdr.stream_slot = mb.slot;
        d.hdr.microblock_id = mb.id;
        d.hdr.stream_meta_flags = meta_flags;
    };

    struct MicroblockWavefrontResult {
        uint8_t done_tag = 0;
        uint64_t dram_r = 0, dram_w = 0, sram_r = 0, sram_w = 0;
        bool streamed = false;
    };

    auto emit_binary_ewe_wavefront = [&](const TileCommand& tc) -> MicroblockWavefrontResult {
        MicroblockWavefrontResult r{};
        uint8_t slot_free_tag[2] = {0, 0};
        uint64_t elem_done = 0;
        uint16_t mb_id = 0;
        const uint64_t total_elems = uint64_t(tc.layer.in_h) * tc.layer.in_w * tc.layer.in_c;
        const uint32_t row_elems = uint32_t(tc.layer.in_w) * tc.layer.in_c;

        while (elem_done < total_elems) {
            Microblock mb{};
            mb.id = mb_id;
            mb.slot = uint8_t(mb_id & 1u);
            mb.elem_off = elem_done;
            if (tc.h_tiled) {
                const uint32_t row_done = uint32_t(elem_done / row_elems);
                mb.rows = std::min<uint32_t>(tc.tile_rows, tc.layer.in_h - row_done);
                mb.elems = mb.rows * row_elems;
            } else {
                mb.elems = std::min<uint32_t>(tc.tile_elems, uint32_t(total_elems - elem_done));
                mb.rows = 1;
            }
            mb.bytes = mb.elems * tc.elem_size;
            const uint32_t dram_off = uint32_t(mb.elem_off * tc.elem_size);
            const uint8_t a_tag = tc.input_a_preloaded ? tc.input_a_wait_tag : alloc_tag();
            const uint8_t b_tag = alloc_tag();
            const uint8_t e_tag = alloc_tag();
            const uint8_t s_tag = alloc_tag();
            const bool final_mb = (elem_done + mb.elems >= total_elems);

            uint64_t a_charged = 0;
            if (!tc.input_a_preloaded) {
                auto [a, charged] = make_act_load(tc.layer,
                                                  uint32_t(tc.layer.dram_in + dram_off),
                                                  tc.in_a_l1[mb.slot], mb.bytes,
                                                  a_tag, slot_free_tag[mb.slot]);
                a_charged = charged;
                if (tc.stream_descriptors) {
                    mark_stream(a, tc.layer_idx, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                }
                program.push_back(a);
                r.sram_w += uint64_t(mb.bytes);
            }

            auto [b, b_charged] = make_binary_b_load(tc.layer,
                                                      uint32_t(tc.layer.dram_wgt + dram_off),
                                                      tc.in_b_l1[mb.slot], mb.bytes,
                                                      b_tag, slot_free_tag[mb.slot]);
            if (tc.stream_descriptors) {
                mark_stream(b, tc.layer_idx, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
            }
            program.push_back(b);

            r.dram_r += a_charged + b_charged;
            r.sram_w += uint64_t(mb.bytes);

            LayerMeta tile_L = tc.layer;
            if (tc.h_tiled) {
                tile_L.in_h = uint16_t(mb.rows);
                tile_L.out_h = uint16_t(mb.rows);
            } else {
                tile_L.in_h = 1;
                tile_L.in_w = 1;
                tile_L.in_c = uint16_t(mb.elems);
                tile_L.out_h = 1;
                tile_L.out_w = 1;
                tile_L.out_c = uint16_t(mb.elems);
            }
            Descriptor e = make_ewe_add(tile_L,
                                        tc.input_a_preloaded
                                            ? uint32_t(tc.in_a_l1[0] + dram_off)
                                            : tc.in_a_l1[mb.slot],
                                        tc.in_b_l1[mb.slot],
                                        tc.output_contiguous
                                            ? uint32_t(tc.out_l1[0] + dram_off)
                                            : tc.out_l1[mb.slot],
                                        tc.params_l1, b_tag, a_tag, e_tag);
            if (tc.stream_descriptors) {
                mark_stream(e, tc.layer_idx, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
            }
            program.push_back(e);
            r.sram_r += 3 * uint64_t(mb.bytes); // two inputs + output read by following store/checkpoint.
            r.sram_w += uint64_t(mb.bytes);

            if (tc.suppress_store) {
                r.streamed = true;
                r.done_tag = e_tag;
                slot_free_tag[mb.slot] = e_tag;
            } else {
                Descriptor s = make_udma(tc.out_l1[mb.slot],
                                         uint32_t(tc.layer.dram_out + dram_off),
                                         mb.bytes,
                                         /*dir*/ 1, s_tag, e_tag);
                if (tc.stream_descriptors) {
                    mark_stream(s, tc.layer_idx, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                }
                program.push_back(s);
                r.dram_w += mb.bytes;
                r.done_tag = s_tag;
                slot_free_tag[mb.slot] = s_tag;
            }

            elem_done += mb.elems;
            ++mb_id;
        }
        return r;
    };

    // v7.1 (post-rolling-tags): tag_fire_time can't tell us per-layer done
    // anymore (tag values cycle 1..255 and get overwritten). UDMA serializes
    // its task queue, so the end_ns of the N-th UDMA task = the time the N-th
    // UDMA descriptor finished. Each layer ends with a UDMA store; record the
    // 1-based UDMA-descriptor count at end-of-layer to look it up later.
    std::vector<size_t> udma_count_at_layer_end(N, 0);
    size_t udma_count_so_far = 0;
    // v8.14: REQUANT task count, used as the layer's done time when its
    // udma_w was skipped by the fusion-source path (no UDMA marks layer end).
    std::vector<size_t> requant_count_at_layer_end(N, 0);
    size_t requant_count_so_far = 0;
    // v8.36: EWE task count, needed when tiled ADD/MUL/SUB output stores are
    // suppressed as producer->consumer intermediate boundaries.
    std::vector<size_t> ewe_count_at_layer_end(N, 0);
    size_t ewe_count_so_far = 0;

    // v8.13: L1-resident layer fusion state. When the previous layer was a
    // single-tile CONV/DWCONV/FC and the current layer's input shape+dtype
    // match its output, we can skip the udma_r (input load) and have the
    // current layer's CONV read directly from the previous layer's L1_OUT
    // address.  Compile_model chain mode (v8.12) already arranged for the
    // current layer's input bytes to equal the previous layer's reference,
    // so the values match.
    uint32_t fuse_prev_l1_out_addr  = 0;
    uint32_t fuse_prev_l1_out_size  = 0;
    uint8_t  fuse_prev_done_tag     = 0;
    uint16_t fuse_prev_out_h = 0, fuse_prev_out_w = 0, fuse_prev_out_c = 0;
    uint16_t fuse_prev_dtype = 0;
    bool     fuse_prev_single_tile  = false;
    bool     fuse_prev_is_conv_class = false;
    bool     fuse_prev_is_binary_ewe = false;
    uint32_t fuse_prev_live_a_addr = 0, fuse_prev_live_a_size = 0;
    uint32_t fuse_prev_live_b_addr = 0, fuse_prev_live_b_size = 0;
    uint32_t fuse_prev_live_o_addr = 0, fuse_prev_live_o_size = 0;
    auto clear_prev_binary_ewe_live = [&]() {
        fuse_prev_is_binary_ewe = false;
        fuse_prev_live_a_size = fuse_prev_live_b_size = fuse_prev_live_o_size = 0;
    };

    // v8.21: ping-pong allocator state. The fused chain places each layer's
    // L1_OUT at alternating ends of L1 (low / high) so the next layer's
    // PARAMS/WGT/OUT block has a contiguous free region on the OPPOSITE side
    // of `live_in` (= prev L1_OUT). Without ping-pong, the chain stack-allocs
    // upward and busts L1 after a few layers (mobilenet_v3 L0→L1→L2→L3
    // collapsed at L3 because chain footprint exceeded 2 MB even though no
    // single layer needed >2 MB). chain_alt = 0 means try_low first (OUT at
    // addr 0); chain_alt = 1 means try_high first (OUT at L1_BUDGET-out_size).
    // After every successful fused layer, chain_alt toggles. Reset to 0 on
    // chain break (non-fused layer).
    uint8_t  chain_alt = 0;

    // v8.14: pending udma_w for fusion-source skip.  When the prev layer is
    // single-tile CONV/DWCONV/FC, we defer its udma_w descriptor until we
    // know if the current layer fuses.  If yes — drop it (output stays
    // resident in L1, no DRAM W).  If no — push it now so current's udma_r
    // sees the data in DRAM.  Per-layer udma_w is verification-only because
    // each layer has its own pre-loaded dram_in (compile_model.py chain mode
    // forwards content, not addresses), so dropping is safe except for losing
    // that layer's per-layer DRAM read-back check.
    struct PendingStore {
        bool        active = false;
        Descriptor  desc{};
        uint64_t    bytes = 0;
        uint32_t    layer_idx = 0;
    };
    PendingStore pending;
    std::vector<bool> udma_w_skipped(N, false);
    std::vector<bool> udma_w_streamed(N, false);
    std::vector<bool> producer_no_store(N, false);
    std::vector<uint32_t> flow_next_layer(N, N);
    auto mark_flow_edge = [&](uint32_t producer, uint32_t consumer) {
        if (producer < N && consumer < N)
            flow_next_layer[producer] = consumer;
    };
    uint32_t mul_layers = 0;
    for (uint32_t mi = 0; mi < N; ++mi)
        if (metas[mi].op_kind == OK_MUL) ++mul_layers;
    const bool conservative_mul_graph = (mul_layers >= 4);

    // v8.34: xlsr-style branch groups often appear as several consecutive
    // CONV outputs whose channels add up to the following CONCAT input
    // channels.  Once CONCAT is logical, those producer stores are also only
    // intermediate verification boundaries, so suppress them.  Also catch the
    // trunk feeding the next slice fanout (32ch -> four 8ch branches) and the
    // final long skip concat (L24 32ch + L0 16ch -> 48ch).
    auto is_conv_class_meta = [](const LayerMeta& M) {
        return M.op_kind == OK_CONV || M.op_kind == OK_DWCONV || M.op_kind == OK_FC;
    };
    auto is_binary_meta = [](const LayerMeta& M) {
        return M.op_kind == OK_ADD || M.op_kind == OK_SUB;
    };
    for (uint32_t ci = 0; ci < N; ++ci) {
        if (conservative_mul_graph) break;
        const auto& C = metas[ci];
        if (C.op_kind != OK_CONCAT || ci == 0) continue;
        uint32_t tail_c = 0;
        uint32_t begin = ci;
        for (uint32_t k = ci; k-- > 0; ) {
            const auto& P = metas[k];
            if (!is_conv_class_meta(P))
                break;
            if (P.out_h != C.out_h || P.out_w != C.out_w || P.dtype != C.dtype)
                break;
            tail_c += P.out_c;
            begin = k;
            if (tail_c >= C.in_c) break;
        }

        if (begin >= ci || tail_c > C.in_c)
            continue;

        const bool full_tail = (tail_c == C.in_c);
        bool suppress_tail = full_tail;
        uint32_t skip_src = N;

        if (!full_tail) {
            const uint32_t need_c = C.in_c - tail_c;
            for (uint32_t k = 0; k < begin; ++k) {
                const auto& S = metas[k];
                if (!is_conv_class_meta(S)) continue;
                if (S.out_h == C.out_h && S.out_w == C.out_w &&
                    S.out_c == need_c && S.dtype == C.dtype) {
                    skip_src = k;
                    suppress_tail = true;
                    break;
                }
            }
        }

        if (suppress_tail) {
            for (uint32_t k = begin; k < ci; ++k)
                producer_no_store[k] = true;
            if (skip_src < N)
                producer_no_store[skip_src] = true;
        } else if (begin < ci) {
            // Partial tail still represents one concat input; missing inputs
            // may have passed through compiler-elided RESIZE ops.
            for (uint32_t k = begin; k < ci; ++k)
                producer_no_store[k] = true;
        }

        if (full_tail && begin > 0) {
            const auto& T = metas[begin - 1];
            if (is_conv_class_meta(T) &&
                T.out_h == C.out_h && T.out_w == C.out_w &&
                T.out_c == C.in_c && T.dtype == C.dtype) {
                producer_no_store[begin - 1] = true;
            }
        }

        // Compiler-elided RESIZE_BILINEAR can hide concat producers from the
        // layer list (DeepLab ASPP/decoder).  Mark nearby conv-class tensors
        // whose channel count looks like a concat component and whose spatial
        // size either matches or cleanly upsamples to the concat output.
        std::set<uint32_t> seen_component_c;
        const uint32_t scan_begin = (ci > 12) ? (ci - 12) : 0;
        for (uint32_t kk = ci; kk-- > scan_begin; ) {
            const auto& P = metas[kk];
            if (!is_conv_class_meta(P)) continue;
            if (P.dtype != C.dtype || P.out_c >= C.in_c) continue;
            if (seen_component_c.count(P.out_c)) continue;
            const bool spatial_ok =
                (P.out_h == C.out_h && P.out_w == C.out_w) ||
                (P.out_h && P.out_w && C.out_h % P.out_h == 0 && C.out_w % P.out_w == 0);
            if (!spatial_ok) continue;
            producer_no_store[kk] = true;
            seen_component_c.insert(P.out_c);
        }
    }

    // v8.35/v8.36: direct producer->consumer boundary.  The compiler preloads each
    // layer's synthetic input bytes, so stores between shape-identical
    // conv/EWE producers and their immediate conv/EWE/D2SPACE consumer are
    // also intermediate verification only. Suppressing them models the
    // intended on-chip handoff while keeping final output writes visible.
    for (uint32_t k = 0; k + 1 < N; ++k) {
        if (conservative_mul_graph) break;
        const auto& P = metas[k];
        const auto& S = metas[k + 1];
        const bool p_ok = is_conv_class_meta(P) || is_binary_meta(P) ||
            (!conservative_mul_graph && P.op_kind == OK_MUL) ||
            P.op_kind == OK_AVG_POOL || P.op_kind == OK_MAX_POOL ||
            P.op_kind == OK_D2SPACE;
        if (!p_ok) continue;
        const bool shape_match =
            P.out_h == S.in_h && P.out_w == S.in_w &&
            P.out_c == S.in_c && P.dtype == S.dtype;
        if (!shape_match) continue;
        if (is_conv_class_meta(S) || is_binary_meta(S) ||
            (!conservative_mul_graph && S.op_kind == OK_MUL) ||
            S.op_kind == OK_AVG_POOL || S.op_kind == OK_MAX_POOL ||
            S.op_kind == OK_D2SPACE)
            producer_no_store[k] = true;
    }

    // FP image-restoration chains (PyNET style): compile_model materializes
    // each consumer's synthetic input bytes.  In MUL-heavy FP graphs the
    // generic direct-boundary pass is disabled for YOLO safety, but large
    // immediate conv/d2space activations are still verification boundaries.
    // Also catch 3x3 SAME consumers whose input metadata is pre-expanded with
    // a one-pixel halo (out 512x768xC -> in 514x770xC).
    for (uint32_t k = 0; k + 1 < N; ++k) {
        const auto& P = metas[k];
        const auto& S = metas[k + 1];
        const bool fp_producer =
            P.dtype == DT_FP16 || P.dtype == DT_BFP16 || P.dtype == DT_FP8;
        const bool producer_ok =
            is_conv_class_meta(P) || P.op_kind == OK_D2SPACE;
        const bool consumer_ok =
            is_conv_class_meta(S) || S.op_kind == OK_D2SPACE;
        if (!fp_producer || !producer_ok || !consumer_ok)
            continue;
        if (P.dtype != S.dtype || P.out_c != S.in_c)
            continue;
        const bool exact_consumer_input =
            S.in_h == P.out_h && S.in_w == P.out_w;
        const bool halo_consumer_input =
            is_conv_class_meta(S) && S.s_h == 1 && S.s_w == 1 &&
            S.k_h == 3 && S.k_w == 3 &&
            S.in_h == uint32_t(P.out_h) + 2u &&
            S.in_w == uint32_t(P.out_w) + 2u;
        const bool large_transient = uint64_t(P.ref_size) >= (3ull << 20);
        if ((exact_consumer_input || halo_consumer_input) && large_transient)
            producer_no_store[k] = true;
    }

    if (conservative_mul_graph) {
        for (uint32_t k = 0; k + 1 < N; ++k) {
            const auto& P = metas[k];
            const auto& S = metas[k + 1];
            const bool large_same_shape_add_boundary =
                is_conv_class_meta(P) &&
                (S.op_kind == OK_ADD || S.op_kind == OK_SUB) &&
                P.dtype == S.dtype &&
                P.out_h == S.in_h && P.out_w == S.in_w && P.out_c == S.in_c &&
                P.out_h >= 256 && P.out_w >= 256;
            if (large_same_shape_add_boundary) {
                producer_no_store[k] = true;
            }
        }
    }

    // Transformer attention tails often lower as SOFTMAX followed by a
    // same-shape RESHAPE.  compile_model has already materialized later
    // synthetic inputs, so both writes are verification-only boundaries.
    for (uint32_t k = 0; k + 1 < N; ++k) {
        const auto& P = metas[k];
        const auto& S = metas[k + 1];
        if (P.op_kind != OK_SOFTMAX || S.op_kind != OK_RESHAPE)
            continue;
        const bool same_shape =
            P.dtype == S.dtype &&
            P.out_h == S.in_h && P.out_w == S.in_w && P.out_c == S.in_c &&
            P.out_h == S.out_h && P.out_w == S.out_w && P.out_c == S.out_c;
        if (same_shape) {
            producer_no_store[k] = true;
            producer_no_store[k + 1] = true;
        }
    }
    // Transformer attention probabilities often have shape
    // [heads, query_len, key_len] and feed later synthetic tensors through
    // skipped reshape/matmul boundaries.  Keep classifier / segmentation
    // softmaxes verified, but suppress these attention-matrix DRAM checks.
    for (uint32_t k = 0; k < N; ++k) {
        const auto& S = metas[k];
        const bool attention_matrix =
            S.in_h > 1 && S.in_h <= 32 &&
            S.in_w > 1 && S.in_w <= 2048 &&
            S.in_c > 1 && S.in_c <= 2048;
        const bool attention_dtype = (S.dtype == DT_INT8x8 || S.dtype == DT_FP16);
        if (S.op_kind == OK_SOFTMAX && attention_dtype && attention_matrix) {
            producer_no_store[k] = true;
        }
    }
    // Transformer attention also contains same-shape RESHAPE barriers between
    // 4x384x384 EWE stages. They do not change layout, and downstream synthetic
    // inputs are already materialized, so the DRAM copy is pure bookkeeping.
    for (uint32_t k = 0; k < N; ++k) {
        const auto& R = metas[k];
        if (R.op_kind != OK_RESHAPE || R.dtype != DT_INT8x8)
            continue;
        const bool attention_matrix =
            R.in_h == 4 && R.in_w == 384 && R.in_c == 384 &&
            R.out_h == 4 && R.out_w == 384 && R.out_c == 384;
        if (attention_matrix)
            producer_no_store[k] = true;
    }

    // v9: compiler v3 carries tensor-level last-use.  In MUL-heavy YOLO-like
    // graphs, binary EWE outputs whose real TFLite tensor has downstream
    // consumers are intermediate verification boundaries; keep the original
    // binary-EWE rule.  For non-binary producers, stay conservative and use
    // GraphMeta only for logical CONCAT/fanout boundaries. Consumers such as
    // HARD_SWISH/GELU still read their input from DRAM today, so suppressing a
    // conv producer solely because it has a graph consumer would corrupt them
    // (ETHZ_v6 mobilenet_v3_float conv->h_swish).
    if (graph_metas) {
        for (uint32_t k = 0; k < N; ++k) {
            const auto& L = metas[k];
            const auto& G = graph_metas[k];
            const bool binary_ewe =
                L.op_kind == OK_ADD || L.op_kind == OK_MUL || L.op_kind == OK_SUB;
            const bool non_binary_suppressible =
                is_conv_class_meta(L) ||
                L.op_kind == OK_AVG_POOL || L.op_kind == OK_MAX_POOL ||
                L.op_kind == OK_D2SPACE;
            const bool has_later_consumer =
                G.consumer_count > 0 && G.last_consumer_layer > int32_t(k);
            const bool ends_at_logical_concat =
                G.last_consumer_layer > int32_t(k) &&
                G.last_consumer_layer < int32_t(N) &&
                metas[uint32_t(G.last_consumer_layer)].op_kind == OK_CONCAT;
            if ((binary_ewe && has_later_consumer) ||
                (non_binary_suppressible && G.consumer_count > 0 && ends_at_logical_concat))
                producer_no_store[k] = true;
        }

        // RESHAPE and MATERIALIZE are often compile-time/reference boundaries
        // in Hotspot slices.  Downstream layers already have their synthetic
        // input bytes materialized by compile_model, so an intermediate
        // DRAM->DRAM copy is only a per-layer verification checkpoint.  Keep
        // final outputs visible, and require size equality so this does not
        // hide gather/scatter-like layout changes.
        for (uint32_t k = 0; k < N; ++k) {
            const auto& L = metas[k];
            const auto& G = graph_metas[k];
            const bool metadata_copy =
                (L.op_kind == OK_RESHAPE || L.op_kind == OK_MATERIALIZE) &&
                L.in_size == L.ref_size &&
                G.consumer_count > 0 &&
                G.last_consumer_layer > int32_t(k);
            if (metadata_copy)
                producer_no_store[k] = true;
        }

        // Hotspot transient compute producers.  If GraphMeta says a
        // CONV-class / unary EWE output has a later consumer, it is an
        // intermediate activation, not the slice output.  The consumer's
        // synthetic input is pre-materialized by compile_model, while real
        // on-chip handoff is modeled by the existing fused/tiled paths when
        // the L1 layout is available.  Suppress the otherwise huge DRAM store
        // checkpoint so fast cycles reflect "write final boundary only".
        for (uint32_t k = 0; k < N; ++k) {
            const auto& L = metas[k];
            const auto& G = graph_metas[k];
            const bool transient_compute =
                (is_conv_class_meta(L) ||
                 L.op_kind == OK_HARD_SWISH || L.op_kind == OK_GELU) &&
                G.consumer_count > 0 &&
                G.last_consumer_layer > int32_t(k);
            if (transient_compute)
                producer_no_store[k] = true;
        }

        // If a compute producer feeds only a same-size metadata barrier, and
        // that barrier has already been classified as a no-store checkpoint,
        // keep the producer transient too.  This catches conv/fc -> reshape /
        // materialize boundaries where GraphMeta last-use is attached to the
        // metadata node rather than the compute node.
        for (uint32_t k = 0; k + 1 < N; ++k) {
            const auto& P = metas[k];
            const auto& S = metas[k + 1];
            const bool transient_producer =
                is_conv_class_meta(P) ||
                P.op_kind == OK_HARD_SWISH ||
                P.op_kind == OK_GELU;
            const bool metadata_consumer =
                (S.op_kind == OK_RESHAPE || S.op_kind == OK_MATERIALIZE) &&
                producer_no_store[k + 1] &&
                (P.ref_size == S.in_size || P.ref_size == S.ref_size);
            if (transient_producer && metadata_consumer)
                producer_no_store[k] = true;
        }
    }

    for (uint32_t k = 0; k + 1 < N; ++k) {
        const auto& P = metas[k];
        const auto& S = metas[k + 1];
        const bool int8_rgb_tail_consumer =
            (S.dtype == DT_INT8x8) && is_binary_meta(S) &&
            (S.in_h >= 1024) && (S.in_w >= 1024) && (S.in_c <= 4);
        if (int8_rgb_tail_consumer && (is_binary_meta(P) || is_conv_class_meta(P))) {
            producer_no_store[k] = false;
        }
        const bool int8_large_upsample_tail =
            (P.dtype == DT_INT8x8) && (S.dtype == DT_INT8x8) &&
            ((S.op_kind == OK_D2SPACE && S.out_h >= 512 && S.out_w >= 512) ||
             (P.op_kind == OK_D2SPACE && P.out_h >= 512 && P.out_w >= 512) ||
             ((P.op_kind == OK_AVG_POOL || P.op_kind == OK_MAX_POOL) &&
              P.out_h >= 512 && P.out_w >= 512));
        if (int8_large_upsample_tail) {
            producer_no_store[k] = false;
        }
    }

    for (uint32_t i = 0; i < N; ++i) {
        const auto& L = metas[i];
        // v8 / v8.10: per-dtype element width.  FP layers now store FP16 in
        // DRAM/L1 (2 B/elem); compute uses FP32 internally.
        // v8.27: INT16x8 hybrid has int16 ACT/OUT (2B) but int8 WGT (1B);
        // conv_pure_weight_bytes() handles the wgt size separately.
        const unsigned in_elem  =
            (L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8
             || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
        const unsigned out_elem = in_elem;             // requant preserves width
        // v8.14: prog_start is set AFTER we resolve any pending udma_w from
        // the previous layer (which may push one descriptor before this
        // layer's body). UDMA / REQUANT counting from this point belongs to
        // the current layer i.
        size_t prog_start = program.size();
        // Helper: flush a deferred udma_w from prev layer into the program.
        // Charges DRAM W / SRAM R bytes against the layer that owns it
        // (pending.layer_idx), and advances the udma counter so the layer
        // boundary (udma_count_at_layer_end[pending.layer_idx]) points at
        // the now-emitted store.
        auto flush_pending = [&]() {
            if (!pending.active) return;
            program.push_back(pending.desc);
            acc[pending.layer_idx].dram_w += pending.bytes;
            acc[pending.layer_idx].sram_r += pending.bytes;
            ++udma_count_so_far;
            udma_count_at_layer_end[pending.layer_idx] = udma_count_so_far;
            pending.active = false;
            prog_start = program.size();
        };

        auto try_stream_conv_chain = [&]() -> bool {
            const auto streamable = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                if (A.op_kind != OK_CONV) return false;
                if (A.s_h != 1 || A.s_w != 1) return false;
                const bool pointwise =
                    A.k_h == 1 && A.k_w == 1 &&
                    A.p_t == 0 && A.p_b == 0 && A.p_l == 0 && A.p_r == 0;
                const bool spatial3 =
                    A.k_h == 3 && A.k_w == 3 &&
                    A.p_t == 1 && A.p_b == 1 && A.p_l == 1 && A.p_r == 1;
                if (!pointwise && !spatial3) return false;
                if (A.group != 1) return false;
                if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;
                return true;
            };
            auto is_pointwise_stream_conv = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                return A.k_h == 1 && A.k_w == 1 &&
                       A.p_t == 0 && A.p_b == 0 && A.p_l == 0 && A.p_r == 0;
            };
            auto conv_row_radius = [&](uint32_t k) -> uint32_t {
                return (metas[k].k_h > 1) ? uint32_t(metas[k].k_h / 2) : 0u;
            };
            if (!streamable(i)) return false;
            uint32_t end = i;
            while (end + 1 < N && streamable(end + 1)) {
                const auto& A = metas[end];
                const auto& B = metas[end + 1];
                if (A.out_h != B.in_h || A.out_w != B.in_w || A.out_c != B.in_c) break;
                if (A.dtype != B.dtype) break;
                ++end;
            }
            if (end <= i) return false;

            const auto& first = metas[i];
            const auto& last  = metas[end];
            if (first.out_h != last.out_h || first.out_w != last.out_w) return false;
            auto conv_layer_needs_microblock = [&](const LayerMeta& A) -> bool {
                const bool is_fp = (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8);
                const uint64_t pure_wgt = conv_pure_weight_bytes(A);
                const uint64_t scale_lut_size = is_fp ? (8 + 4 * uint64_t(A.out_c))
                                                       : (12 + 9 * uint64_t(A.out_c));
                const uint64_t corr_size =
                    (!is_fp && uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                    ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
                const uint64_t full_working_set =
                    uint64_t(A.in_size) + uint64_t(A.ref_size) +
                    pure_wgt + scale_lut_size + corr_size + 4096;
                return full_working_set > L1_BUDGET ||
                       uint64_t(A.ref_size) + 4096 > L1_BUDGET;
            };
            const bool stream_to_d2s_add =
                (end + 2 < N)
                && metas[end + 1].op_kind == OK_D2SPACE
                && (metas[end + 2].op_kind == OK_ADD || metas[end + 2].op_kind == OK_MUL
                    || metas[end + 2].op_kind == OK_SUB)
                && metas[end + 1].dtype == last.dtype
                && metas[end + 2].dtype == last.dtype
                && metas[end + 1].in_h == last.out_h
                && metas[end + 1].in_w == last.out_w
                && metas[end + 1].in_c == last.out_c
                && metas[end + 2].in_h == metas[end + 1].out_h
                && metas[end + 2].in_w == metas[end + 1].out_w
                && metas[end + 2].in_c == metas[end + 1].out_c;
            // v9.2: enable generic CONV->CONV microblock streaming for plain
            // pointwise linear chains. Spatial 3x3 plain chains still need a
            // stronger line-buffer ownership model; keep those on the
            // conservative per-layer tiler unless they are part of the
            // already-validated CONV...->D2S->EWE stream tail below.
            if (!stream_to_d2s_add) {
                // Large image tail heads (for example mv3_depth_quant's
                // 384x576x8 -> 384x576x1 projection) are usually final-output
                // / side-output boundaries.  The generic pointwise chain keeps
                // only per-row microblocks live; keep these on the per-layer
                // path until the chain has explicit large-tail ownership.
                if (uint64_t(last.out_h) * last.out_w >= 128ull * 128ull &&
                    last.out_c <= 4)
                    return false;
                for (uint32_t k = i; k <= end; ++k) {
                    if (!is_pointwise_stream_conv(k)) return false;
                }
                for (uint32_t k = i; k < end; ++k) {
                    if (!producer_no_store[k]) return false;
                    if (graph_metas) {
                        const auto& G = graph_metas[k];
                        if (G.consumer_count != 1 ||
                            G.first_consumer_layer != int32_t(k + 1) ||
                            G.last_consumer_layer  != int32_t(k + 1))
                            return false;
                    }
                }
            }
            bool needs_streaming = false;
            const uint32_t pressure_end = stream_to_d2s_add ? end - 1 : end;
            for (uint32_t k = i; k <= pressure_end; ++k) {
                if (conv_layer_needs_microblock(metas[k])) {
                    needs_streaming = true;
                    break;
                }
            }
            if (!needs_streaming) return false;

            flush_pending();

            const unsigned elem =
                (first.dtype == DT_INT16x16 || first.dtype == DT_INT16x8) ? 2u : 1u;
            const uint32_t H = last.out_h;
            const uint32_t W = last.out_w;
            uint32_t tile_oh = std::min<uint32_t>(64, H);
            const uint64_t safety = 65536;

            auto range_for = [&](uint32_t layer, uint32_t final_lo, uint32_t final_hi) {
                uint32_t lo = final_lo;
                uint32_t hi = final_hi;
                for (uint32_t k = end; k > layer; --k) {
                    const uint32_t r = conv_row_radius(k);
                    lo = (lo > r) ? lo - r : 0;
                    hi = std::min<uint32_t>(H, hi + r);
                }
                return std::pair<uint32_t, uint32_t>(lo, hi);
            };
            auto max_pingpong_bytes = [&](uint32_t toh) {
                uint64_t mx = 0;
                for (uint32_t k = i; k <= end; ++k) {
                    const auto r = range_for(k, 0, std::min<uint32_t>(toh, H));
                    const uint64_t rows = r.second - r.first;
                    mx = std::max<uint64_t>(mx, rows * uint64_t(W) * metas[k].out_c * elem);
                }
                return align64(uint32_t(mx));
            };
            uint64_t max_params = 0, max_wgt = 0;
            for (uint32_t k = i; k <= end; ++k) {
                const auto& A = metas[k];
                const bool is_fp = (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8);
                const uint64_t pure_wgt = conv_pure_weight_bytes(A);
                const uint64_t scale_lut_size = is_fp ? (8 + 4 * uint64_t(A.out_c))
                                                       : (12 + 9 * uint64_t(A.out_c));
                const uint64_t corr_size =
                    (!is_fp && uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                    ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
                max_params = std::max<uint64_t>(max_params, scale_lut_size + corr_size);
                max_wgt    = std::max<uint64_t>(max_wgt, pure_wgt);
            }
            if (stream_to_d2s_add) {
                max_params = std::max<uint64_t>(max_params, 48);
            }
            constexpr uint32_t STREAM_SLOTS = 2;
            auto stream_slot_bytes = [&](uint32_t toh) {
                const uint64_t buf = max_pingpong_bytes(toh);
                const uint64_t d2s_add_b = stream_to_d2s_add
                    ? uint64_t(toh) * metas[end + 1].k_h * metas[end + 2].in_w * metas[end + 2].in_c * elem
                    : 0;
                const uint64_t wgt_region = std::max<uint64_t>(align64(uint32_t(max_wgt)),
                                                               align64(uint32_t(d2s_add_b)));
                return align64(uint32_t(2 * buf + align64(uint32_t(max_params)) + wgt_region));
            };
            while (tile_oh > 1) {
                const uint64_t slot_bytes_try = stream_slot_bytes(tile_oh);
                if (STREAM_SLOTS * slot_bytes_try + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            const uint64_t buf_bytes = max_pingpong_bytes(tile_oh);
            const uint64_t d2s_add_b_max = stream_to_d2s_add
                ? uint64_t(tile_oh) * metas[end + 1].k_h * metas[end + 2].in_w * metas[end + 2].in_c * elem
                : 0;
            const uint64_t wgt_region_bytes = std::max<uint64_t>(align64(uint32_t(max_wgt)),
                                                                 align64(uint32_t(d2s_add_b_max)));
            const uint32_t params_region_bytes = align64(uint32_t(max_params));
            const uint32_t slot_bytes = align64(uint32_t(2 * buf_bytes
                                             + params_region_bytes + wgt_region_bytes));
            if (tile_oh < 1 || STREAM_SLOTS * uint64_t(slot_bytes) + safety > L1_BUDGET) {
                return false;
            }

            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            std::vector<uint8_t> slot_done(STREAM_SLOTS, 0);
            uint8_t prev_store = 0;
            TileCommand stream_tc{};
            stream_tc.kind = TileCommand::CONV_D2S_EWE;
            stream_tc.layer_idx = i;
            stream_tc.layer_end = stream_to_d2s_add ? end + 2 : end;
            stream_tc.tile_rows = tile_oh;
            stream_tc.elem_size = elem;
            auto emit_stream = [&](Descriptor d, const Microblock& mb,
                                   uint32_t layer_idx, uint8_t meta_flags,
                                   bool urgent = false) {
                mark_stream(d, layer_idx, mb, meta_flags);
                if (urgent) d.hdr.flags |= DF_STREAM_TAIL;
                program.push_back(d);
            };

            for (uint32_t y = 0; y < H; y += tile_oh) {
                const uint32_t y_hi = std::min<uint32_t>(H, y + tile_oh);
                const uint32_t tile_idx = y / stream_tc.tile_rows;
                Microblock mb{};
                mb.id = uint16_t(tile_idx);
                mb.slot = uint8_t(tile_idx % STREAM_SLOTS);
                mb.rows = y_hi - y;
                mb.elems = mb.rows * W * last.out_c;
                mb.bytes = mb.elems * stream_tc.elem_size;
                const bool final_mb = (y_hi >= H);
                const uint32_t slot = mb.slot;
                const uint32_t slot_base = slot * slot_bytes;
                const uint32_t BUF0 = slot_base;
                const uint32_t BUF1 = align64(uint32_t(BUF0 + buf_bytes));
                const uint32_t L1_PARAMS_STREAM = align64(uint32_t(BUF1 + buf_bytes));
                const uint32_t L1_WGT_STREAM = align64(uint32_t(L1_PARAMS_STREAM + params_region_bytes));
                uint8_t input_ready = 0;
                uint32_t input_addr = BUF0;
                uint32_t output_addr = BUF1;

                for (uint32_t k = i; k <= end; ++k) {
                    const auto& A = metas[k];
                    const auto out_r = range_for(k, y, y_hi);
                    const uint32_t out_lo = out_r.first;
                    const uint32_t out_hi = out_r.second;
                    const uint32_t out_rows = out_hi - out_lo;
                    const uint32_t radius = conv_row_radius(k);
                    const uint32_t in_lo = (out_lo > radius) ? out_lo - radius : 0;
                    const uint32_t in_hi = std::min<uint32_t>(A.in_h, out_hi + radius);
                    const uint32_t in_rows = in_hi - in_lo;
                    const uint64_t pure_wgt = conv_pure_weight_bytes(A);
                    const uint64_t scale_lut_size = 12 + 9 * uint64_t(A.out_c);
                    const uint64_t corr_size =
                        (uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                        ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
                    const uint64_t params_blob = scale_lut_size + corr_size;
                    const uint32_t in_bytes =
                        in_rows * A.in_w * A.in_c * elem;
                    const uint32_t out_bytes =
                        out_rows * A.out_w * A.out_c * elem;
                    uint8_t wait_tag = (k == i) ? slot_done[slot] : input_ready;

                    if (k == i) {
                        const uint32_t dram_in_off = in_lo * A.in_w * A.in_c * elem;
                        const uint8_t in_tag = alloc_tag();
                        auto [ad, charged] = make_act_load(A, A.dram_in + dram_in_off,
                                                           input_addr, in_bytes,
                                                           in_tag, wait_tag);
                        emit_stream(ad, mb, k, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                        acc[k].dram_r += charged;
                        acc[k].sram_w += in_bytes;
                        ++udma_count_so_far;
                        last_udma[k] = udma_count_so_far;
                        wait_tag = in_tag;
                    }

                    const uint8_t params_tag = alloc_tag();
                    const uint8_t wgt_tag = alloc_tag();
                    const uint8_t req_tag = alloc_tag();
                    emit_stream(make_udma(A.dram_wgt + uint32_t(pure_wgt),
                                          L1_PARAMS_STREAM, uint32_t(params_blob),
                                          /*dir*/ 0, params_tag, wait_tag),
                                mb, k, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                    acc[k].dram_r += params_blob;
                    acc[k].sram_w += params_blob;
                    ++udma_count_so_far;
                    last_udma[k] = udma_count_so_far;
                    emit_stream(make_udma(A.dram_wgt, L1_WGT_STREAM, uint32_t(pure_wgt),
                                          /*dir*/ 0, wgt_tag, wait_tag),
                                mb, k, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                    acc[k].dram_r += pure_wgt;
                    acc[k].sram_w += pure_wgt;
                    ++udma_count_so_far;
                    last_udma[k] = udma_count_so_far;

                    Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                              /*signal*/ 0, wgt_tag, wait_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr  = input_addr;
                    cb.wgt_addr = L1_WGT_STREAM;
                    cb.out_addr = output_addr;
                    cb.in_h = uint16_t(in_rows);
                    cb.in_w = A.in_w;
                    cb.in_c = A.in_c;
                    cb.out_c = A.out_c;
                    cb.k_h = A.k_h;
                    cb.k_w = A.k_w;
                    cb.stride_dilation = encode_stride_pair(A.s_h, A.s_w);
                    cb.pad_tb = uint8_t(((out_lo == 0 ? A.p_t : 0) & 7)
                                      | (((out_hi == A.out_h ? A.p_b : 0) & 7) << 3));
                    cb.pad_lr = uint8_t((A.p_l & 7) | ((A.p_r & 7) << 3));
                    cb.group = A.group ? A.group : 1;
                    cb.cluster_mask = 0xFFFF;
                    cb.in_pad_value = A.zp_in_eff;
                    emit_stream(cd, mb, k, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    acc[k].sram_r += pure_wgt + in_bytes;

                    Descriptor rd = make_desc(OC_REQUANT, uint8_t(A.dtype),
                                              /*signal*/ req_tag, params_tag, 0);
                    auto& rb = rd.body.requant;
                    rb.in_addr = 0;
                    rb.out_addr = output_addr;
                    rb.n = 1;
                    rb.h = uint16_t(out_rows);
                    rb.w = A.out_w;
                    rb.c = A.out_c;
                    rb.scale_lut_addr = L1_PARAMS_STREAM;
                    rb.scale_count = A.out_c;
                    rb.oc_start = 0;
                    rb.per_channel_flag = 1;
                    rb.out_w_layer = A.out_w;
                    rb.oh_start = uint16_t(out_lo);
                    rb.corr_addr = corr_size ? uint32_t(L1_PARAMS_STREAM + scale_lut_size) : 0u;
                    rb.corr_per_oc = 0;
                    emit_stream(rd, mb, k, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    acc[k].sram_r += scale_lut_size;
                    acc[k].sram_w += out_bytes;
                    ++requant_count_so_far;
                    last_req[k] = requant_count_so_far;

                    if (k == end && !stream_to_d2s_add) {
                        const uint8_t st_tag = alloc_tag();
                        const uint32_t dram_out_off = y * A.out_w * A.out_c * elem;
                        emit_stream(make_udma(output_addr, A.dram_out + dram_out_off,
                                              out_bytes, /*dir*/ 1, st_tag, req_tag),
                                    mb, k, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                        acc[k].sram_r += out_bytes;
                        acc[k].dram_w += out_bytes;
                        ++udma_count_so_far;
                        last_udma[k] = udma_count_so_far;
                        prev_store = st_tag;
                    } else {
                        udma_w_skipped[k] = true;
                        udma_w_streamed[k] = true;
                        if (k < end)
                            mark_flow_edge(k, k + 1);
                        else if (stream_to_d2s_add)
                            mark_flow_edge(k, end + 1);
                        if (k == end) prev_store = req_tag;
                    }

                    if (k == end && stream_to_d2s_add) {
                        const auto& D = metas[end + 1];
                        const auto& B = metas[end + 2];
                        const uint16_t block = D.k_h ? D.k_h : 1;
                        const uint32_t add_rows = out_rows * block;
                        const uint32_t add_row0 = y * block;
                        const uint32_t add_tile_bytes = add_rows * B.in_w * B.in_c * elem;
                        const uint32_t add_dram_off = add_row0 * B.in_w * B.in_c * elem;
                        const uint8_t d2s_tag = alloc_tag();
                        const uint8_t add_params_tag = alloc_tag();
                        const uint8_t add_b_tag = alloc_tag();
                        const uint8_t add_req_tag = alloc_tag();
                        const uint8_t add_st_tag = alloc_tag();

                        emit_stream(make_udma_d2s(output_addr, input_addr,
                                                  uint16_t(out_rows), A.out_w, A.out_c,
                                                  block, uint8_t(elem), d2s_tag, req_tag),
                                    mb, end + 1, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0),
                                    true);
                        acc[end + 1].sram_r += out_bytes;
                        acc[end + 1].sram_w += add_tile_bytes;
                        ++udma_count_so_far;
                        last_udma[end + 1] = udma_count_so_far;
                        udma_w_skipped[end + 1] = true;
                        udma_w_streamed[end + 1] = true;
                        mark_flow_edge(end + 1, end + 2);

                        emit_stream(make_udma(B.dram_wgt + B.wgt_size - 48,
                                              L1_PARAMS_STREAM, 48,
                                              /*dir*/ 0, add_params_tag, d2s_tag),
                                    mb, end + 2, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0),
                                    true);
                        acc[end + 2].dram_r += 48;
                        acc[end + 2].sram_w += 48;
                        ++udma_count_so_far;
                        last_udma[end + 2] = udma_count_so_far;

                        emit_stream(make_udma(uint32_t(B.dram_wgt + add_dram_off),
                                              L1_WGT_STREAM, add_tile_bytes,
                                              /*dir*/ 0, add_b_tag, add_params_tag),
                                    mb, end + 2, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0),
                                    true);
                        acc[end + 2].dram_r += add_tile_bytes;
                        acc[end + 2].sram_w += add_tile_bytes;
                        ++udma_count_so_far;
                        last_udma[end + 2] = udma_count_so_far;

                        LayerMeta tile_B = B;
                        tile_B.in_h = uint16_t(add_rows);
                        tile_B.out_h = uint16_t(add_rows);
                        Descriptor ed = make_ewe_add(tile_B, input_addr, L1_WGT_STREAM,
                                                     output_addr, L1_PARAMS_STREAM,
                                                     add_b_tag, d2s_tag, add_req_tag);
                        emit_stream(ed, mb, end + 2,
                                    SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0),
                                    true);
                        acc[end + 2].sram_r += 2 * uint64_t(add_tile_bytes);
                        acc[end + 2].sram_w += add_tile_bytes;

                        emit_stream(make_udma(output_addr,
                                              uint32_t(B.dram_out + add_dram_off),
                                              add_tile_bytes,
                                              /*dir*/ 1, add_st_tag, add_req_tag),
                                    mb, end + 2, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                        acc[end + 2].sram_r += add_tile_bytes;
                        acc[end + 2].dram_w += add_tile_bytes;
                        ++udma_count_so_far;
                        last_udma[end + 2] = udma_count_so_far;
                        prev_store = add_st_tag;
                        slot_done[slot] = add_st_tag;
                    }

                    input_ready = req_tag;
                    std::swap(input_addr, output_addr);
                }
                if (!stream_to_d2s_add) {
                    slot_done[slot] = prev_store;
                }
            }

            for (uint32_t k = i; k <= end; ++k) {
                udma_count_at_layer_end[k] = last_udma[k];
                requant_count_at_layer_end[k] = last_req[k];
                tiles_h_per_layer[k] = uint16_t((H + tile_oh - 1) / tile_oh);
                tiles_oc_per_layer[k] = 1;
            }
            if (stream_to_d2s_add) {
                udma_count_at_layer_end[end + 1] = last_udma[end + 1];
                udma_count_at_layer_end[end + 2] = last_udma[end + 2];
                tiles_h_per_layer[end + 1] = uint16_t((H + tile_oh - 1) / tile_oh);
                tiles_h_per_layer[end + 2] = uint16_t((H + tile_oh - 1) / tile_oh);
            }
            layer_done_tag[stream_to_d2s_add ? end + 2 : end] = prev_store;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = stream_to_d2s_add ? end + 2 : end;
            return true;
        };

        auto try_stream_conv_fanout = [&]() -> bool {
            const auto streamable = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                if (A.op_kind != OK_CONV) return false;
                if (!producer_no_store[k]) return false;
                if (A.group != 1) return false;
                if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;
                return true;
            };
            if (!streamable(i)) return false;
            uint32_t end = i;
            while (end + 1 < N && streamable(end + 1)) {
                const auto& A = metas[i];
                const auto& B = metas[end + 1];
                const bool same_logical_input =
                    graph_metas && graph_metas[end + 1].input0_tensor == graph_metas[i].input0_tensor;
                if (B.dram_in != A.dram_in && !same_logical_input) break;
                if (B.in_h != A.in_h || B.in_w != A.in_w || B.in_c != A.in_c) break;
                if (B.out_h != A.out_h || B.out_w != A.out_w) break;
                if (B.k_h != A.k_h || B.k_w != A.k_w ||
                    B.s_h != A.s_h || B.s_w != A.s_w ||
                    B.p_t != A.p_t || B.p_b != A.p_b ||
                    B.p_l != A.p_l || B.p_r != A.p_r ||
                    B.dtype != A.dtype) break;
                ++end;
            }
            if (end <= i) return false;
            bool has_near_concat = false;
            if (end + 1 < N && metas[end + 1].op_kind == OK_CONCAT)
                has_near_concat = true;
            if (!has_near_concat) return false;

            const auto& first = metas[i];
            const unsigned elem =
                (first.dtype == DT_INT16x16 || first.dtype == DT_INT16x8) ? 2u : 1u;
            const uint64_t safety = 65536;
            uint64_t fixed_bytes = 0;
            uint64_t max_out_bytes = 0;
            struct BranchBlob {
                uint64_t pure_wgt = 0;
                uint64_t scale_lut = 0;
                uint64_t corr = 0;
                uint32_t params_l1 = 0;
                uint32_t wgt_l1 = 0;
                uint8_t params_tag = 0;
                uint8_t wgt_tag = 0;
            };
            std::vector<BranchBlob> blobs(end - i + 1);
            for (uint32_t k = i; k <= end; ++k) {
                const auto& A = metas[k];
                auto& bb = blobs[k - i];
                bb.pure_wgt = conv_pure_weight_bytes(A);
                bb.scale_lut = 12 + 9 * uint64_t(A.out_c);
                bb.corr = (uint64_t(A.wgt_size) > bb.pure_wgt + bb.scale_lut)
                        ? (uint64_t(A.wgt_size) - bb.pure_wgt - bb.scale_lut) : 0;
                fixed_bytes += align64(uint32_t(bb.pure_wgt));
                fixed_bytes += align64(uint32_t(bb.scale_lut + bb.corr));
                max_out_bytes = std::max<uint64_t>(
                    max_out_bytes, uint64_t(A.out_h) * A.out_w * A.out_c * elem);
            }

            const uint64_t row_in = uint64_t(first.in_w) * first.in_c * elem;
            const uint64_t per_oh_in = row_in * first.s_h;
            const uint64_t fixed_in = row_in * (first.k_h ? (first.k_h - 1) : 0);
            const uint64_t max_out_row = [&]() {
                uint64_t v = 0;
                for (uint32_t k = i; k <= end; ++k)
                    v = std::max<uint64_t>(v, uint64_t(metas[k].out_w) * metas[k].out_c * elem);
                return v;
            }();
            uint32_t tile_oh = first.out_h;
            const uint64_t base_fixed = fixed_bytes + safety;
            if (base_fixed >= L1_BUDGET) return false;
            const uint64_t io_budget = L1_BUDGET - base_fixed;
            if (per_oh_in + max_out_row > 0) {
                uint64_t cand = (io_budget > fixed_in)
                              ? ((io_budget - fixed_in) / (per_oh_in + max_out_row))
                              : 1;
                cand = std::max<uint64_t>(1, std::min<uint64_t>(cand, first.out_h));
                tile_oh = uint32_t(cand);
            }

            auto tile_bytes_for = [&](uint32_t toh) {
                const uint32_t worst_in_h = toh * first.s_h + (first.k_h ? first.k_h - 1 : 0);
                const uint64_t in_b = uint64_t(worst_in_h) * first.in_w * first.in_c * elem;
                uint64_t out_b = 0;
                for (uint32_t k = i; k <= end; ++k)
                    out_b = std::max<uint64_t>(out_b, uint64_t(toh) * metas[k].out_w * metas[k].out_c * elem);
                return std::pair<uint64_t, uint64_t>(in_b, out_b);
            };
            while (tile_oh > 1) {
                auto [in_b, out_b] = tile_bytes_for(tile_oh);
                if (align64(uint32_t(in_b)) + align64(uint32_t(out_b)) +
                    fixed_bytes + safety <= L1_BUDGET) break;
                --tile_oh;
            }
            auto [max_in_b, max_tile_out_b] = tile_bytes_for(tile_oh);
            if (align64(uint32_t(max_in_b)) + align64(uint32_t(max_tile_out_b)) +
                fixed_bytes + safety > L1_BUDGET)
                return false;

            flush_pending();

            const uint32_t L1_IN_FAN = 0;
            const uint32_t L1_OUT_FAN = align64(uint32_t(max_in_b));
            uint32_t cursor = align64(uint32_t(L1_OUT_FAN + max_tile_out_b));
            for (uint32_t k = i; k <= end; ++k) {
                auto& bb = blobs[k - i];
                bb.params_l1 = cursor;
                cursor = align64(uint32_t(cursor + bb.scale_lut + bb.corr));
                bb.wgt_l1 = cursor;
                cursor = align64(uint32_t(cursor + bb.pure_wgt));
                bb.params_tag = alloc_tag();
                bb.wgt_tag = alloc_tag();
                const auto& A = metas[k];
                program.push_back(make_udma(A.dram_wgt + uint32_t(bb.pure_wgt),
                                            bb.params_l1,
                                            uint32_t(bb.scale_lut + bb.corr),
                                            /*dir*/ 0, bb.params_tag));
                program.push_back(make_udma(A.dram_wgt, bb.wgt_l1,
                                            uint32_t(bb.pure_wgt),
                                            /*dir*/ 0, bb.wgt_tag));
                acc[k].dram_r += bb.scale_lut + bb.corr + bb.pure_wgt;
                acc[k].sram_w += bb.scale_lut + bb.corr + bb.pure_wgt;
            }

            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            uint8_t prev_tile_done = 0;
            uint8_t group_done = 0;
            uint16_t tile_id = 0;
            for (uint32_t oh_done = 0; oh_done < first.out_h; oh_done += tile_oh, ++tile_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, first.out_h - oh_done);
                const uint8_t pad_t_tile = (oh_done == 0) ? first.p_t : 0;
                const bool is_last_h = (oh_done + this_oh == first.out_h);
                const uint8_t pad_b_tile = is_last_h ? first.p_b : 0;
                const int ih_lo_u = int(oh_done) * int(first.s_h) - int(first.p_t);
                const int ih_hi_u =
                    int(oh_done + this_oh - 1) * int(first.s_h) + int(first.k_h) - 1 - int(first.p_t);
                const int ih_lo = std::max(0, ih_lo_u);
                const int ih_hi = std::min(int(first.in_h) - 1, ih_hi_u);
                const uint32_t this_in_h = uint32_t(ih_hi - ih_lo + 1);
                const uint32_t tile_in_size = this_in_h * first.in_w * first.in_c * elem;
                const uint32_t dram_in_off = uint32_t(ih_lo) * first.in_w * first.in_c * elem;
                const uint8_t in_tag = alloc_tag();
                Microblock mb{};
                mb.id = tile_id;
                mb.slot = uint8_t(tile_id & 1u);
                mb.rows = this_oh;
                mb.elems = this_oh * first.out_w * first.out_c;
                mb.bytes = tile_in_size;
                auto [idesc, charged] = make_act_load(first, first.dram_in + dram_in_off,
                                                       L1_IN_FAN, tile_in_size,
                                                       in_tag, prev_tile_done);
                mark_stream(idesc, i, mb, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                program.push_back(idesc);
                acc[i].dram_r += charged;
                acc[i].sram_w += tile_in_size;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                uint8_t last_branch_req = in_tag;
                for (uint32_t k = i; k <= end; ++k) {
                    const auto& A = metas[k];
                    const auto& bb = blobs[k - i];
                    const uint8_t req_tag = alloc_tag();
                    Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                              /*signal*/ 0, bb.wgt_tag, in_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr = L1_IN_FAN;
                    cb.wgt_addr = bb.wgt_l1;
                    cb.out_addr = L1_OUT_FAN;
                    cb.in_h = uint16_t(this_in_h);
                    cb.in_w = A.in_w;
                    cb.in_c = A.in_c;
                    cb.out_c = A.out_c;
                    cb.k_h = A.k_h;
                    cb.k_w = A.k_w;
                    cb.stride_dilation = encode_stride_pair(A.s_h, A.s_w);
                    cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                    cb.pad_lr = uint8_t((A.p_l & 7) | ((A.p_r & 7) << 3));
                    cb.group = A.group ? A.group : 1;
                    cb.cluster_mask = 0xFFFF;
                    cb.in_pad_value = A.zp_in_eff;
                    mark_stream(cd, k, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                    program.push_back(cd);
                    acc[k].sram_r += bb.pure_wgt + tile_in_size;

                    Descriptor rd = make_desc(OC_REQUANT, uint8_t(A.dtype),
                                              /*signal*/ req_tag, bb.params_tag, bb.wgt_tag, in_tag);
                    auto& rb = rd.body.requant;
                    rb.in_addr = 0;
                    rb.out_addr = L1_OUT_FAN;
                    rb.n = 1;
                    rb.h = uint16_t(this_oh);
                    rb.w = A.out_w;
                    rb.c = A.out_c;
                    rb.scale_lut_addr = bb.params_l1;
                    rb.scale_count = A.out_c;
                    rb.oc_start = 0;
                    rb.per_channel_flag = 1;
                    rb.out_w_layer = A.out_w;
                    rb.oh_start = uint16_t(oh_done);
                    rb.corr_addr = bb.corr ? uint32_t(bb.params_l1 + bb.scale_lut) : 0u;
                    rb.corr_per_oc = 0;
                    mark_stream(rd, k, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                    program.push_back(rd);
                    acc[k].sram_r += bb.scale_lut;
                    acc[k].sram_w += uint64_t(this_oh) * A.out_w * A.out_c * elem;
                    ++requant_count_so_far;
                    last_req[k] = requant_count_so_far;
                    udma_w_skipped[k] = true;
                    udma_w_streamed[k] = true;
                    last_branch_req = req_tag;
                    layer_done_tag[k] = req_tag;
                }
                prev_tile_done = last_branch_req;
                group_done = last_branch_req;
            }

            for (uint32_t k = i; k <= end; ++k) {
                tiles_h_per_layer[k] = uint16_t((metas[k].out_h + tile_oh - 1) / tile_oh);
                tiles_oc_per_layer[k] = 1;
                requant_count_at_layer_end[k] = last_req[k];
                udma_count_at_layer_end[k] = (k == i) ? last_udma[i] : last_udma[k];
            }
            layer_done_tag[end] = group_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = end;
            return true;
        };

        auto try_stream_conv_ewe = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& A = metas[i];
            const auto& B = metas[i + 1];
            const bool conv_class =
                A.op_kind == OK_CONV || A.op_kind == OK_DWCONV || A.op_kind == OK_FC;
            const bool binary_ewe =
                B.op_kind == OK_ADD || B.op_kind == OK_MUL || B.op_kind == OK_SUB;
            if (!conv_class || !binary_ewe) return false;
            if (!producer_no_store[i]) return false;
            if (A.dtype != B.dtype) return false;
            if (A.out_h != B.in_h || A.out_w != B.in_w || A.out_c != B.in_c) return false;
            if (B.in_h != B.out_h || B.in_w != B.out_w || B.in_c != B.out_c) return false;
            if (B.wgt_size < 48) return false;
            if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;

            const unsigned elem =
                (A.dtype == DT_INT16x16 || A.dtype == DT_INT16x8) ? 2u : 1u;
            const bool is_dw = (A.op_kind == OK_DWCONV);
            const uint64_t pure_wgt = conv_pure_weight_bytes(A);
            const uint64_t scale_lut_size = 12 + 9 * uint64_t(A.out_c);
            const uint64_t corr_size =
                (uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t safety = 65536;
            const uint32_t STREAM_SLOTS = 2;
            if (pure_wgt + params_blob + 48 + safety >= L1_BUDGET) return false;
            const uint64_t full_out_bytes =
                uint64_t(A.out_h) * A.out_w * A.out_c * elem;
            if (2ull * full_out_bytes + pure_wgt + params_blob + safety <= L1_BUDGET)
                return false;

            uint32_t tile_oh = A.out_h;
            auto tile_shape_bytes = [&](uint32_t toh) {
                const uint32_t worst_in_h = toh * A.s_h + (A.k_h ? A.k_h - 1 : 0);
                const uint64_t conv_in =
                    uint64_t(worst_in_h) * A.in_w * A.in_c * elem;
                const uint64_t out =
                    uint64_t(toh) * A.out_w * A.out_c * elem;
                return std::pair<uint64_t, uint64_t>(conv_in, out);
            };
            auto layout_bytes_for = [&](uint32_t toh) {
                auto [conv_in, out] = tile_shape_bytes(toh);
                const uint64_t fixed =
                    align64(uint32_t(params_blob)) +
                    align64(uint32_t(pure_wgt)) +
                    align64(48u);
                const uint64_t slot =
                    align64(uint32_t(conv_in)) + 3ull * align64(uint32_t(out));
                return std::pair<uint64_t, uint64_t>(fixed, slot);
            };
            while (tile_oh > 1) {
                auto [fixed, slot] = layout_bytes_for(tile_oh);
                if (fixed + STREAM_SLOTS * slot + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            auto [fixed_bytes, slot_bytes_raw] = layout_bytes_for(tile_oh);
            if (tile_oh < 1 ||
                fixed_bytes + STREAM_SLOTS * slot_bytes_raw + safety > L1_BUDGET)
                return false;
            if (tile_oh == A.out_h)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS_STREAM = 0;
            const uint32_t L1_WGT_STREAM = align64(uint32_t(params_blob));
            const uint32_t L1_EWE_PARAMS_STREAM =
                align64(uint32_t(L1_WGT_STREAM + pure_wgt));
            const uint32_t SLOT_BASE =
                align64(uint32_t(L1_EWE_PARAMS_STREAM + 48));
            const uint32_t SLOT_BYTES = align64(uint32_t(slot_bytes_raw));
            auto [max_conv_in, max_tile_out] = tile_shape_bytes(tile_oh);
            const uint32_t SLOT_CONV_IN = 0;
            const uint32_t SLOT_CONV_OUT = align64(uint32_t(max_conv_in));
            const uint32_t SLOT_EWE_B = align64(uint32_t(SLOT_CONV_OUT + max_tile_out));
            const uint32_t SLOT_EWE_OUT = align64(uint32_t(SLOT_EWE_B + max_tile_out));

            auto emit_stream = [&](Descriptor d, const Microblock& mb,
                                   uint32_t layer_idx, uint8_t meta_flags,
                                   bool urgent = false) {
                mark_stream(d, layer_idx, mb, meta_flags);
                if (urgent) d.hdr.flags |= DF_STREAM_TAIL;
                program.push_back(d);
            };

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_ewe(N, 0);
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            const uint8_t ewe_params_tag = alloc_tag();

            program.push_back(make_udma(A.dram_wgt + uint32_t(pure_wgt),
                                        L1_PARAMS_STREAM, uint32_t(params_blob),
                                        /*dir*/ 0, params_tag));
            acc[i].dram_r += params_blob;
            acc[i].sram_w += params_blob;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            program.push_back(make_udma(A.dram_wgt, L1_WGT_STREAM,
                                        uint32_t(pure_wgt),
                                        /*dir*/ 0, wgt_tag));
            acc[i].dram_r += pure_wgt;
            acc[i].sram_w += pure_wgt;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            program.push_back(make_udma(B.dram_wgt + B.wgt_size - 48,
                                        L1_EWE_PARAMS_STREAM, 48,
                                        /*dir*/ 0, ewe_params_tag));
            acc[i + 1].dram_r += 48;
            acc[i + 1].sram_w += 48;
            ++udma_count_so_far;
            last_udma[i + 1] = udma_count_so_far;

            uint8_t slot_done[STREAM_SLOTS] = {0, 0};
            uint8_t conv_done = 0;
            uint8_t ewe_done = 0;
            uint16_t tile_id = 0;
            for (uint32_t oh_done = 0; oh_done < A.out_h; oh_done += tile_oh, ++tile_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, A.out_h - oh_done);
                const bool is_last_h = (oh_done + this_oh == A.out_h);
                const int ih_lo_u = int(oh_done) * int(A.s_h) - int(A.p_t);
                const int ih_hi_u =
                    int(oh_done + this_oh - 1) * int(A.s_h) + int(A.k_h) - 1 - int(A.p_t);
                const int ih_lo = std::max(0, ih_lo_u);
                const int ih_hi = std::min(int(A.in_h) - 1, ih_hi_u);
                const uint32_t this_in_h = uint32_t(ih_hi - ih_lo + 1);
                const uint32_t conv_in_bytes = this_in_h * A.in_w * A.in_c * elem;
                const uint32_t tile_out_bytes = this_oh * A.out_w * A.out_c * elem;
                const uint32_t dram_in_off = uint32_t(ih_lo) * A.in_w * A.in_c * elem;
                const uint32_t dram_out_off = oh_done * B.out_w * B.out_c * elem;
                Microblock mb{};
                mb.id = tile_id;
                mb.slot = uint8_t(tile_id & 1u);
                mb.rows = this_oh;
                mb.elems = this_oh * A.out_w * A.out_c;
                mb.bytes = tile_out_bytes;
                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_CONV_IN = slot_base + SLOT_CONV_IN;
                const uint32_t L1_CONV_OUT = slot_base + SLOT_CONV_OUT;
                const uint32_t L1_EWE_B = slot_base + SLOT_EWE_B;
                const uint32_t L1_EWE_OUT = slot_base + SLOT_EWE_OUT;

                const uint8_t in_tag = alloc_tag();
                auto [id, charged] = make_act_load(A, A.dram_in + dram_in_off,
                                                   L1_CONV_IN, conv_in_bytes,
                                                   in_tag, slot_done[mb.slot]);
                emit_stream(id, mb, i, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].dram_r += charged;
                acc[i].sram_w += conv_in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                          /*signal*/ 0, wgt_tag, in_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_CONV_IN;
                cb.wgt_addr = L1_WGT_STREAM;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(this_in_h);
                cb.in_w = A.in_w;
                cb.in_c = A.in_c;
                cb.out_c = A.out_c;
                cb.k_h = A.k_h;
                cb.k_w = A.k_w;
                cb.stride_dilation = encode_stride_pair(A.s_h, A.s_w);
                cb.pad_tb = uint8_t((((oh_done == 0) ? A.p_t : 0) & 7)
                                  | (((is_last_h ? A.p_b : 0) & 7) << 3));
                cb.pad_lr = uint8_t((A.p_l & 7) | ((A.p_r & 7) << 3));
                cb.group = A.group ? A.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = A.zp_in_eff;
                emit_stream(cd, mb, i, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].sram_r += pure_wgt + conv_in_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(A.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, in_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_CONV_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = A.out_w;
                rb.c = A.out_c;
                rb.scale_lut_addr = L1_PARAMS_STREAM;
                rb.scale_count = A.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = A.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS_STREAM + scale_lut_size) : 0u;
                rb.corr_per_oc = uint8_t(corr_size && is_dw);
                emit_stream(rd, mb, i, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].sram_r += scale_lut_size;
                acc[i].sram_w += tile_out_bytes;
                ++requant_count_so_far;
                last_req[i] = requant_count_so_far;
                conv_done = req_tag;
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;

                const uint8_t b_tag = alloc_tag();
                auto [bd, b_charged] = make_binary_b_load(B,
                                                           B.dram_wgt + dram_out_off,
                                                           L1_EWE_B, tile_out_bytes,
                                                           b_tag, ewe_params_tag);
                emit_stream(bd, mb, i + 1,
                            SMF_LOAD_B | (is_last_h ? SMF_FINAL_TILE : 0),
                            true);
                acc[i + 1].dram_r += b_charged;
                acc[i + 1].sram_w += tile_out_bytes;
                ++udma_count_so_far;
                last_udma[i + 1] = udma_count_so_far;

                LayerMeta tile_B = B;
                tile_B.in_h = uint16_t(this_oh);
                tile_B.out_h = uint16_t(this_oh);
                Descriptor ed = make_ewe_add(tile_B, L1_CONV_OUT, L1_EWE_B,
                                             L1_EWE_OUT, L1_EWE_PARAMS_STREAM,
                                             b_tag, req_tag, alloc_tag());
                const uint8_t ewe_tag = ed.hdr.signal_tag;
                emit_stream(ed, mb, i + 1,
                            SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0),
                            true);
                acc[i + 1].sram_r += 2 * uint64_t(tile_out_bytes);
                acc[i + 1].sram_w += tile_out_bytes;
                ++ewe_count_so_far;
                last_ewe[i + 1] = ewe_count_so_far;
                ewe_done = ewe_tag;

                if (producer_no_store[i + 1]) {
                    udma_w_skipped[i + 1] = true;
                    udma_w_streamed[i + 1] = true;
                    slot_done[mb.slot] = ewe_tag;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    emit_stream(make_udma(L1_EWE_OUT, B.dram_out + dram_out_off,
                                          tile_out_bytes, /*dir*/ 1, st_tag, ewe_tag),
                                mb, i + 1, SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                    acc[i + 1].sram_r += tile_out_bytes;
                    acc[i + 1].dram_w += tile_out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    ewe_done = st_tag;
                    slot_done[mb.slot] = st_tag;
                }
            }

            tiles_h_per_layer[i] = uint16_t((A.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[i] = 1;
            tiles_h_per_layer[i + 1] = tiles_h_per_layer[i];
            tiles_oc_per_layer[i + 1] = 1;
            requant_count_at_layer_end[i] = last_req[i];
            udma_count_at_layer_end[i] = last_udma[i];
            ewe_count_at_layer_end[i] = 0;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = 0;
            ewe_count_at_layer_end[i + 1] = last_ewe[i + 1];
            layer_done_tag[i] = conv_done;
            layer_done_tag[i + 1] = ewe_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto try_stream_binary_ewe_chain = [&]() -> bool {
            auto is_binary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_ADD || L.op_kind == OK_MUL || L.op_kind == OK_SUB;
            };
            if (!graph_metas || !is_binary_ewe(metas[i])) return false;
            const auto& first = metas[i];
            if (first.dtype != DT_INT8x8) return false;
            if (first.wgt_size < first.ref_size + 48) return false;
            if (first.in_h != first.out_h || first.in_w != first.out_w || first.in_c != first.out_c)
                return false;

            uint32_t end = i;
            while (end + 1 < N && is_binary_ewe(metas[end + 1])) {
                const auto& A = metas[end];
                const auto& B = metas[end + 1];
                const auto& GA = graph_metas[end];
                const auto& GB = graph_metas[end + 1];
                if (!producer_no_store[end]) break;
                if (GA.consumer_count != 1 ||
                    GA.first_consumer_layer != int32_t(end + 1) ||
                    GA.last_consumer_layer  != int32_t(end + 1))
                    break;
                // Keep quantized ADD/SUB/MUL operand order exact. Swapping
                // input0/input1 can change zp/mult handling even for ADD/MUL.
                if (GB.producer0_layer != int32_t(end))
                    break;
                if (B.dtype != first.dtype) break;
                if (B.wgt_size < B.ref_size + 48) break;
                if (A.out_h != B.in_h || A.out_w != B.in_w || A.out_c != B.in_c)
                    break;
                if (B.in_h != B.out_h || B.in_w != B.out_w || B.in_c != B.out_c)
                    break;
                if (B.in_h != first.in_h || B.in_w != first.in_w || B.in_c != first.in_c)
                    break;
                ++end;
            }
            if (end <= i) return false;

            // Transformer attention tails are often binary EWE -> SOFTMAX.
            // The older per-layer EWE wavefront can keep the final attention
            // matrix as one contiguous L1 tensor for SOFTMAX.  The deeper
            // chain below only ping-pongs tile buffers, so using it here would
            // force SOFTMAX to reload the whole matrix from DRAM.
            if (producer_no_store[end] && end + 1 < N &&
                metas[end + 1].op_kind == OK_SOFTMAX) {
                return false;
            }

            const uint32_t depth = end - i + 1;
            const uint32_t elem = 1;
            const uint32_t row_elems = uint32_t(first.in_w) * first.in_c;
            const uint32_t per_row_bytes = row_elems * elem;
            if (per_row_bytes == 0) return false;

            const uint64_t safety = 65536;
            uint32_t tile_rows = first.in_h;
            auto slot_bytes_for = [&](uint32_t rows) -> uint64_t {
                const uint64_t tile_bytes = uint64_t(rows) * per_row_bytes;
                const uint64_t seg = align64(uint32_t(tile_bytes));
                return (2ull + depth) * seg; // ping-pong data buffers + one B buffer per layer.
            };
            const uint64_t params_bytes = align64(48u) * uint64_t(depth);
            while (tile_rows > 1 &&
                   params_bytes + 2ull * slot_bytes_for(tile_rows) + safety > L1_BUDGET) {
                --tile_rows;
            }
            if (params_bytes + 2ull * slot_bytes_for(tile_rows) + safety > L1_BUDGET)
                return false;

            flush_pending();

            struct EweStage {
                uint32_t layer_idx = 0;
                uint32_t params_l1 = 0;
                uint8_t params_tag = 0;
            };
            std::vector<EweStage> stages(depth);
            uint32_t cursor = 0;
            for (uint32_t d = 0; d < depth; ++d) {
                const uint32_t k = i + d;
                auto& st = stages[d];
                st.layer_idx = k;
                st.params_l1 = cursor;
                cursor = align64(cursor + 48);
                st.params_tag = alloc_tag();
                program.push_back(make_udma(metas[k].dram_wgt + metas[k].wgt_size - 48,
                                            st.params_l1, 48,
                                            /*dir*/ 0, st.params_tag));
                acc[k].dram_r += 48;
                acc[k].sram_w += 48;
                ++udma_count_so_far;
                udma_count_at_layer_end[k] = udma_count_so_far;
            }

            const uint32_t tile_bytes_max = tile_rows * per_row_bytes;
            const uint32_t seg_bytes = align64(tile_bytes_max);
            const uint32_t slot_base0 = align64(cursor);
            const uint32_t slot_bytes = uint32_t((2ull + depth) * seg_bytes);
            auto slot_base = [&](uint32_t slot) {
                return slot_base0 + slot * slot_bytes;
            };

            std::vector<size_t> last_udma(N, 0), last_ewe(N, 0);
            std::vector<uint8_t> last_done(N, 0);
            uint8_t slot_done[2] = {0, 0};
            uint16_t mb_id = 0;
            for (uint32_t row = 0; row < first.in_h; row += tile_rows, ++mb_id) {
                const uint32_t rows = std::min<uint32_t>(tile_rows, first.in_h - row);
                const uint32_t tile_bytes = rows * per_row_bytes;
                const uint32_t dram_off = row * per_row_bytes;
                const bool final_mb = (row + rows >= first.in_h);
                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(row) * row_elems;
                mb.rows = rows;
                mb.elems = rows * row_elems;
                mb.bytes = tile_bytes;

                const uint32_t base = slot_base(mb.slot);
                const uint32_t data0 = base;
                const uint32_t data1 = base + seg_bytes;
                auto b_addr = [&](uint32_t d) {
                    return base + (2 + d) * seg_bytes;
                };

                const uint8_t a_tag = alloc_tag();
                auto [ad, a_charged] = make_act_load(first,
                                                      first.dram_in + dram_off,
                                                      data0, tile_bytes,
                                                      a_tag, slot_done[mb.slot]);
                mark_stream(ad, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ad);
                acc[i].dram_r += a_charged;
                acc[i].sram_w += tile_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                uint8_t prev_tag = a_tag;
                uint32_t in_addr = data0;
                uint8_t tile_done = a_tag;
                for (uint32_t d = 0; d < depth; ++d) {
                    const uint32_t k = i + d;
                    const auto& L = metas[k];
                    const uint8_t b_tag = alloc_tag();
                    auto [bd, b_charged] = make_binary_b_load(L,
                                                               L.dram_wgt + dram_off,
                                                               b_addr(d), tile_bytes,
                                                               b_tag, slot_done[mb.slot]);
                    mark_stream(bd, k, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(bd);
                    acc[k].dram_r += b_charged;
                    acc[k].sram_w += tile_bytes;
                    ++udma_count_so_far;
                    last_udma[k] = udma_count_so_far;

                    LayerMeta tile_L = L;
                    tile_L.in_h = uint16_t(rows);
                    tile_L.out_h = uint16_t(rows);
                    const uint32_t out_addr = (d & 1u) ? data0 : data1;
                    const uint8_t e_tag = alloc_tag();
                    Descriptor ed = make_ewe_add(tile_L, in_addr, b_addr(d), out_addr,
                                                 stages[d].params_l1,
                                                 b_tag, prev_tag, e_tag);
                    mark_stream(ed, k, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(ed);
                    acc[k].sram_r += 2 * uint64_t(tile_bytes);
                    acc[k].sram_w += tile_bytes;
                    ++ewe_count_so_far;
                    last_ewe[k] = ewe_count_so_far;
                    last_done[k] = e_tag;
                    tile_done = e_tag;

                    if (k < end) {
                        udma_w_skipped[k] = true;
                        udma_w_streamed[k] = true;
                        mark_flow_edge(k, k + 1);
                    }

                    prev_tag = e_tag;
                    in_addr = out_addr;
                }

                const bool suppress_final_store = producer_no_store[end];
                if (suppress_final_store) {
                    udma_w_skipped[end] = true;
                    udma_w_streamed[end] = true;
                    slot_done[mb.slot] = tile_done;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(in_addr, metas[end].dram_out + dram_off,
                                              tile_bytes, /*dir*/ 1, st_tag, tile_done);
                    mark_stream(sd, end, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[end].sram_r += tile_bytes;
                    acc[end].dram_w += tile_bytes;
                    ++udma_count_so_far;
                    last_udma[end] = udma_count_so_far;
                    last_done[end] = st_tag;
                    slot_done[mb.slot] = st_tag;
                }
            }

            const uint16_t tiles_h = uint16_t((first.in_h + tile_rows - 1) / tile_rows);
            for (uint32_t d = 0; d < depth; ++d) {
                const uint32_t k = i + d;
                tiles_h_per_layer[k] = tiles_h;
                tiles_oc_per_layer[k] = 1;
                udma_count_at_layer_end[k] = last_udma[k] ? last_udma[k] : udma_count_at_layer_end[k];
                requant_count_at_layer_end[k] = 0;
                ewe_count_at_layer_end[k] = last_ewe[k];
                layer_done_tag[k] = last_done[k];
            }

            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = end;
            return true;
        };

        if (try_stream_conv_fanout()) {
            continue;
        }

        if (try_stream_conv_ewe()) {
            continue;
        }

        if (try_stream_binary_ewe_chain()) {
            continue;
        }

        if (try_stream_conv_chain()) {
            continue;
        }

        switch (L.op_kind) {
        case OK_CONV: case OK_DWCONV: case OK_FC: {
            // v7: 2-D tile loop over (oh, oc). Either dim collapses to 1 tile
            // when full layer fits. Layout in L1 is:
            //   L1_PARAMS = 0                                    (full layer params blob)
            //   L1_WGT    = align64(scale_lut_size)              (per-OC-tile slice)
            //   L1_IN     = align64(L1_WGT + tile_wgt_size)      (per-OH-tile slice)
            //   L1_OUT    = align64(L1_IN  + tile_in_size)       (per-tile output)

            const bool is_dw = (L.op_kind == OK_DWCONV);
            const bool is_fp = (L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8);
            const uint64_t pure_wgt        = conv_pure_weight_bytes(L);
            const uint64_t pure_wgt_per_oc = pure_wgt / std::max<uint32_t>(L.out_c, 1u);
            // Params layout — int (v6/v7): 12 B hdr + 4*OC mult + 1*OC shift + 4*OC bias_eff (+ optional corr).
            //                 fp  (v8):     8 B (act_min,act_max) + 4*OC bias.
            const uint64_t scale_lut_size  = is_fp
                ? (8 + 4 * uint64_t(L.out_c))
                : (12 + 9 * uint64_t(L.out_c));
            const uint64_t corr_size       =
                (!is_fp && uint64_t(L.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(L.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const bool     have_corr       = (corr_size > 0);
            const uint64_t params_blob     = scale_lut_size + corr_size;
            const uint64_t safety          = 4096;

            // ---- decide tile_oc ----
            // v8.28: also OC-tile when weights + I/O can't co-fit (not just
            // when weights alone bust L1). Estimate the minimum I/O footprint
            // any layout needs: one input row + one output row of activations.
            // For inception_v3_float layer 34 (40×40×288 → 19×19×384 FP16),
            // pure_wgt = 1.99 MB barely fits in 2 MB but leaves no space for
            // I/O tiles → OH tiling lands at tile_oh=1 and the L1_OUT cursor
            // overflows. Reserving even one row of I/O forces OC-tiling here.
            const uint64_t per_oh_in_min  =
                uint64_t(L.in_w)  * L.in_c  * std::max<uint32_t>(L.s_h, 1u) * in_elem;
            const uint64_t per_oh_out_min = uint64_t(L.out_w) * L.out_c * out_elem;
            const uint64_t fixed_in_min   =
                uint64_t(L.in_w) * L.in_c * (L.k_h > 0 ? (L.k_h - 1) : 0) * in_elem;
            const uint64_t io_reserve     =
                per_oh_in_min + per_oh_out_min + fixed_in_min;
            uint32_t tile_oc = L.out_c;
            if (pure_wgt + scale_lut_size + safety + io_reserve > L1_BUDGET) {
                if (is_dw) {
                    std::cerr << "layer " << i
                              << ": dwconv with weights > 2 MB not supported "
                                 "(would need correlated OC+IC slicing)\n";
                    return 4;
                }
                // Reserve room for params + safety + I/O reserve, divide the
                // rest between weights (50%) and a working set (50%).
                const uint64_t avail = (L1_BUDGET > scale_lut_size + safety + io_reserve)
                                     ? (L1_BUDGET - scale_lut_size - safety - io_reserve) : 0;
                const uint64_t half  = avail / 2;
                uint64_t cand = pure_wgt_per_oc ? (half / pure_wgt_per_oc) : L.out_c;
                if (cand < 16)        cand = 16;
                if (cand > L.out_c)   cand = L.out_c;
                if (cand >= 16)       cand = (cand / 16) * 16;
                tile_oc = uint32_t(cand);
            }
            const uint32_t tile_wgt_max = uint32_t(tile_oc * pure_wgt_per_oc);

            // v8.13: try FUSED layout first — L1_IN reuses prev layer's
            // L1_OUT slot (no udma_r), PARAMS/WGT/OUT placed after the
            // preserved IN region.  Only attempted for single-tile fits.
            // Enabled for INT and FP; FP rounding drift across fused chains
            // is absorbed by the 5%/5% per-layer tolerance below.
            const bool fuse_eligible =
                fuse_prev_is_conv_class &&
                fuse_prev_single_tile &&
                fuse_prev_dtype  == L.dtype &&
                fuse_prev_out_h  == L.in_h  &&
                fuse_prev_out_w  == L.in_w  &&
                fuse_prev_out_c  == L.in_c;

            uint32_t L1_PARAMS, L1_WGT, L1_IN, L1_OUT;
            uint32_t tile_oh = L.out_h;
            bool fused_this_layer = false;
            bool fused_used_low = false;   // which side this layer's OUT landed on
            bool pingpong_tiles = false;
            bool pingpong_persistent_wgt = false;
            uint32_t pp_slot_base = 0;
            uint32_t pp_slot_bytes = 0;
            uint32_t pp_slot_in_off = 0;
            uint32_t pp_slot_out_off = 0;

            if (fuse_eligible) {
                L1_IN = fuse_prev_l1_out_addr;
                const uint64_t worst_out =
                    uint64_t(L.out_h) * L.out_w * tile_oc * out_elem;
                const uint32_t in_lo = fuse_prev_l1_out_addr;
                const uint32_t in_hi = fuse_prev_l1_out_addr + fuse_prev_l1_out_size;

                // v8.21: try_low — OUT at addr 0; PARAMS/WGT placed above OUT,
                // all below live_in. Net layout: [OUT | WGT | PARAMS | … free … | live_in | …].
                auto try_low = [&]() -> bool {
                    const uint64_t out_addr = 0;
                    const uint64_t wgt_addr = align64(uint32_t(out_addr + worst_out));
                    const uint64_t par_addr = align64(uint32_t(wgt_addr + tile_wgt_max));
                    const uint64_t top      = align64(uint32_t(par_addr + params_blob));
                    if (top + safety > in_lo) return false;
                    L1_OUT    = uint32_t(out_addr);
                    L1_WGT    = uint32_t(wgt_addr);
                    L1_PARAMS = uint32_t(par_addr);
                    return true;
                };
                // v8.21: try_high — OUT at top end (BUDGET-out_size, floor-aligned);
                // PARAMS/WGT immediately above live_in. Layout:
                // [… | live_in | PARAMS | WGT | … free … | OUT].
                auto try_high = [&]() -> bool {
                    if (worst_out + safety > L1_BUDGET) return false;
                    const uint64_t out_addr  = (uint64_t(L1_BUDGET) - worst_out) & ~uint64_t(63);
                    const uint64_t par_addr  = align64(in_hi);
                    const uint64_t wgt_addr  = align64(uint32_t(par_addr + params_blob));
                    const uint64_t wgt_end   = align64(uint32_t(wgt_addr + tile_wgt_max));
                    if (wgt_end + safety > out_addr) return false;
                    L1_OUT    = uint32_t(out_addr);
                    L1_WGT    = uint32_t(wgt_addr);
                    L1_PARAMS = uint32_t(par_addr);
                    return true;
                };

                bool ok = (chain_alt == 0) ? try_low() : try_high();
                if (ok) fused_used_low = (chain_alt == 0);
                else {
                    ok = (chain_alt == 0) ? try_high() : try_low();
                    if (ok) fused_used_low = (chain_alt != 0);
                }
                if (ok) {
                    fused_this_layer = true;
                }
            }

            // v8.14: resolve prev layer's deferred udma_w now that we know
            // whether this layer fuses.  Fused → output stays resident in L1,
            // drop the store entirely.  Not fused → emit it before this
            // layer's body so the value reaches DRAM (verification only;
            // current layer reads its own pre-loaded dram_in regardless).
            if (pending.active) {
                if (fused_this_layer) {
                    udma_w_skipped[pending.layer_idx] = true;
                    pending.active = false;
                } else {
                    flush_pending();
                }
            }
            if (fused_this_layer && i > 0)
                mark_flow_edge(i - 1, i);
            const uint8_t layer_entry_wait =
                (!fused_this_layer && i > 0 && udma_w_skipped[i - 1])
                ? layer_done_tag[i - 1] : 0;

            if (!fused_this_layer) {
                // Standard non-fused layout.  When the layer is H-tiled but not
                // OC-tiled, try a true two-slot tile layout so UDMA_R can prefetch
                // the next input/weight tile while CONV/REQUANT use the current slot.
                L1_PARAMS = 0;
                L1_WGT    = align64(uint32_t(params_blob));

                const uint64_t per_oh_in  = uint64_t(L.in_w) * L.in_c * L.s_h * in_elem;
                const uint64_t per_oh_out = uint64_t(L.out_w) * tile_oc * out_elem;
                const uint64_t fixed_in   = uint64_t(L.in_w) * L.in_c * (L.k_h - 1) * in_elem;
                const uint64_t io_budget  =
                    (uint64_t(L1_WGT) + tile_wgt_max + safety < L1_BUDGET)
                    ? (L1_BUDGET - L1_WGT - tile_wgt_max - safety) : 0;
                tile_oh = L.out_h;
                if (per_oh_in + per_oh_out > 0) {
                    if (io_budget > fixed_in) {
                        uint64_t cand = (io_budget - fixed_in) / (per_oh_in + per_oh_out);
                        if (cand < 1)         cand = 1;
                        if (cand > L.out_h)   cand = L.out_h;
                        tile_oh = uint32_t(cand);
                    } else {
                        tile_oh = 1;
                    }
                }

                auto layout_for = [&](uint32_t toh) {
                    struct R { uint32_t L1_IN; uint32_t L1_OUT; uint64_t worst_out; };
                    uint32_t worst_in_h = toh * L.s_h + (L.k_h - 1);
                    uint64_t worst_in   = uint64_t(worst_in_h) * L.in_w * L.in_c * in_elem;
                    uint32_t L1_IN  = align64(L1_WGT + tile_wgt_max);
                    uint32_t L1_OUT = align64(L1_IN  + uint32_t(worst_in));
                    uint64_t worst_out = uint64_t(toh) * L.out_w * tile_oc * out_elem;
                    return R{L1_IN, L1_OUT, worst_out};
                };
                while (tile_oh > 1) {
                    auto r = layout_for(tile_oh);
                    if (uint64_t(r.L1_OUT) + r.worst_out <= L1_BUDGET) break;
                    tile_oh -= 1;
                }
                auto layout = layout_for(tile_oh);
                if (uint64_t(layout.L1_OUT) + layout.worst_out > L1_BUDGET) {
                    std::cerr << "layer " << i << ": cannot fit in 2 MB L1 (tile_oh=1, tile_oc="
                              << tile_oc << ")\n";
                    return 4;
                }
                L1_IN  = layout.L1_IN;
                L1_OUT = layout.L1_OUT;

                const bool can_pingpong =
                    tile_oc == L.out_c && tile_oh < L.out_h;
                if (can_pingpong) {
                    auto pp_persistent_wgt_layout_for = [&](uint32_t toh) {
                        struct R {
                            uint32_t wgt_addr;
                            uint32_t slot_base;
                            uint32_t slot_bytes;
                            uint32_t in_off;
                            uint32_t out_off;
                            bool fits;
                        };
                        const uint32_t worst_in_h = toh * L.s_h + (L.k_h - 1);
                        const uint64_t worst_in =
                            uint64_t(worst_in_h) * L.in_w * L.in_c * in_elem;
                        const uint64_t worst_out =
                            uint64_t(toh) * L.out_w * tile_oc * out_elem;
                        const uint32_t wgt_addr = align64(uint32_t(params_blob));
                        const uint32_t base = align64(wgt_addr + tile_wgt_max);
                        const uint32_t out_off = align64(uint32_t(worst_in));
                        const uint32_t slot_bytes = align64(out_off + uint32_t(worst_out));
                        const uint64_t total = uint64_t(base) + 2ull * slot_bytes + safety;
                        return R{wgt_addr, base, slot_bytes, 0, out_off, total <= L1_BUDGET};
                    };
                    auto pp_layout_for = [&](uint32_t toh) {
                        struct R {
                            uint32_t slot_base;
                            uint32_t slot_bytes;
                            uint32_t in_off;
                            uint32_t out_off;
                            bool fits;
                        };
                        const uint32_t worst_in_h = toh * L.s_h + (L.k_h - 1);
                        const uint64_t worst_in =
                            uint64_t(worst_in_h) * L.in_w * L.in_c * in_elem;
                        const uint64_t worst_out =
                            uint64_t(toh) * L.out_w * tile_oc * out_elem;
                        const uint32_t base = align64(uint32_t(params_blob));
                        const uint32_t in_off = align64(tile_wgt_max);
                        const uint32_t out_off = align64(in_off + uint32_t(worst_in));
                        const uint32_t slot_bytes = align64(out_off + uint32_t(worst_out));
                        const uint64_t total = uint64_t(base) + 2ull * slot_bytes + safety;
                        return R{base, slot_bytes, in_off, out_off, total <= L1_BUDGET};
                    };
                    while (tile_oh > 1) {
                        auto ppw = pp_persistent_wgt_layout_for(tile_oh);
                        if (ppw.fits) {
                            L1_WGT = ppw.wgt_addr;
                            pp_slot_base = ppw.slot_base;
                            pp_slot_bytes = ppw.slot_bytes;
                            pp_slot_in_off = ppw.in_off;
                            pp_slot_out_off = ppw.out_off;
                            pingpong_tiles = true;
                            pingpong_persistent_wgt = true;
                            break;
                        }
                        auto pp = pp_layout_for(tile_oh);
                        if (pp.fits) {
                            pp_slot_base = pp.slot_base;
                            pp_slot_bytes = pp.slot_bytes;
                            pp_slot_in_off = pp.in_off;
                            pp_slot_out_off = pp.out_off;
                            pingpong_tiles = true;
                            break;
                        }
                        tile_oh -= 1;
                    }
                    if (pingpong_tiles) {
                        if (!pingpong_persistent_wgt)
                            L1_WGT = pp_slot_base;
                        L1_IN  = pp_slot_base + pp_slot_in_off;
                        L1_OUT = pp_slot_base + pp_slot_out_off;
                    } else {
                        auto fallback = layout_for(tile_oh);
                        L1_IN  = fallback.L1_IN;
                        L1_OUT = fallback.L1_OUT;
                    }
                }
            }

            // ---- load full params (+ corr) blob once per layer ----
            const uint8_t params_tag = alloc_tag();
            const uint32_t params_dram = L.dram_wgt + uint32_t(pure_wgt);
            // v8.21: when ping-pong put this layer's params/wgt region into the
            // "low" zone, those addresses overlap with prev layer's L1_IN
            // (still being read by prev's CONV). Wait on prev REQUANT done
            // before any UDMA write into L1, so we don't race.
            program.push_back(make_udma(params_dram, L1_PARAMS,
                                        uint32_t(params_blob),
                                        /*dir*/ 0, params_tag,
                                        fused_this_layer ? fuse_prev_done_tag : layer_entry_wait));
            acc[i].dram_r += params_blob;
            acc[i].sram_w += params_blob;

            uint8_t persistent_wgt_tag = 0;
            if (pingpong_tiles && pingpong_persistent_wgt) {
                persistent_wgt_tag = alloc_tag();
                Descriptor wpd = make_udma(L.dram_wgt, L1_WGT, tile_wgt_max,
                                           /*dir*/ 0, persistent_wgt_tag,
                                           layer_entry_wait);
                program.push_back(wpd);
                acc[i].dram_r += tile_wgt_max;
                acc[i].sram_w += tile_wgt_max;
            }

            // ---- emit per-tile descriptors (oh outer, oc inner) ----
            // v8.14: precompute single_tile here so the udma_w emission can
            // decide whether to defer the store to `pending` (skip path) or
            // push it inline (multi-tile, no fusion source).
            const bool single_tile_layer = (tile_oh == L.out_h) && (tile_oc == L.out_c);
            const bool suppress_producer_store = producer_no_store[i];
            const bool large_int8_upsample_conv =
                (L.dtype == DT_INT8x8) && (L.out_h >= 512) && (L.out_w >= 512);
            const bool stream_pingpong_tiles =
                !(L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8) &&
                !is_fp &&
                !large_int8_upsample_conv;
            uint8_t prev_store = layer_entry_wait;
            uint8_t slot_free_tag[2] = {layer_entry_wait, layer_entry_wait};
            uint8_t prev_req_tag = 0;            // v8.14: last REQUANT tag, used as
                                                 // fuse_prev_done_tag when output stays in L1.
            uint32_t oh_done = 0;
            uint16_t tile_id = 0;
            uint32_t last_l1_out_addr = L1_OUT;
            while (oh_done < L.out_h) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, L.out_h - oh_done);
                const uint8_t pad_t_tile = (oh_done == 0) ? L.p_t : 0;
                const bool    is_last_h  = (oh_done + this_oh == L.out_h);
                const uint8_t pad_b_tile = is_last_h ? L.p_b : 0;
                const uint8_t tile_slot = pingpong_tiles ? uint8_t(tile_id & 1u) : 0u;
                const uint32_t tile_slot_base = pingpong_tiles
                    ? uint32_t(pp_slot_base + uint32_t(tile_slot) * pp_slot_bytes)
                    : L1_WGT;
                const uint32_t tile_l1_wgt = pingpong_tiles
                    ? (pingpong_persistent_wgt
                       ? L1_WGT
                       : tile_slot_base)
                    : L1_WGT;
                const uint32_t tile_l1_in = pingpong_tiles
                    ? uint32_t(tile_slot_base + pp_slot_in_off)
                    : L1_IN;
                const uint32_t tile_l1_out = pingpong_tiles
                    ? uint32_t(tile_slot_base + pp_slot_out_off)
                    : L1_OUT;
                last_l1_out_addr = tile_l1_out;

                const int ih_lo_u = int(oh_done) * int(L.s_h) - int(L.p_t);
                const int ih_hi_u =
                    int(oh_done + this_oh - 1) * int(L.s_h) + int(L.k_h) - 1 - int(L.p_t);
                const int ih_lo = std::max(0, ih_lo_u);
                const int ih_hi = std::min(int(L.in_h) - 1, ih_hi_u);
                const uint32_t this_in_h    = uint32_t(ih_hi - ih_lo + 1);
                const uint32_t tile_in_size = this_in_h * L.in_w * L.in_c * in_elem;
                const uint32_t dram_in_off  = uint32_t(ih_lo) * L.in_w * L.in_c * in_elem;

                // Load this oh-tile's input once (shared across oc-tiles).
                // v8.13: fused layers skip the udma_r — input is already in
                // L1 at L1_IN (= prev layer's L1_OUT). CONV waits on prev's
                // req_tag instead of a fresh in_tag so the data is ready.
                uint8_t in_tag;
                if (fused_this_layer) {
                    in_tag = fuse_prev_done_tag;     // prev REQUANT done = input ready
                } else {
                    in_tag = alloc_tag();
                    const uint8_t wait_slot = pingpong_tiles ? slot_free_tag[tile_slot] : prev_store;
                    auto [id, charged] = make_act_load(L, L.dram_in + dram_in_off,
                                                       tile_l1_in, tile_in_size,
                                                       in_tag, wait_slot, 0);
                    if (pingpong_tiles && stream_pingpong_tiles) {
                        Microblock mb{};
                        mb.id = tile_id;
                        mb.slot = tile_slot;
                        mb.rows = this_oh;
                        mb.elems = this_oh * L.out_w * L.out_c;
                        mb.bytes = tile_in_size;
                        mark_stream(id, i, mb, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                    }
                    program.push_back(id);
                    acc[i].dram_r += charged;
                    acc[i].sram_w += tile_in_size;
                }

                uint32_t oc_done = 0;
                while (oc_done < L.out_c) {
                    const uint32_t this_oc = std::min<uint32_t>(tile_oc, L.out_c - oc_done);
                    const uint32_t wgt_slice_size = uint32_t(this_oc * pure_wgt_per_oc);
                    const uint32_t wgt_dram_off   = oc_done * uint32_t(pure_wgt_per_oc);

                    const uint8_t wgt_tag   = pingpong_persistent_wgt
                                            ? persistent_wgt_tag : alloc_tag();
                    const uint8_t req_tag   = alloc_tag();
                    const uint8_t store_tag = alloc_tag();

                    // Load wgt slice for this oc-tile. Full-OC ping-pong keeps
                    // weights persistent in L1 when possible, so H tiles only
                    // stream input/output slots.
                    if (!pingpong_persistent_wgt) {
                        Descriptor wd_r = make_udma(L.dram_wgt + wgt_dram_off, tile_l1_wgt,
                                                    wgt_slice_size, /*dir*/ 0, wgt_tag,
                                                    pingpong_tiles ? slot_free_tag[tile_slot]
                                                                   : layer_entry_wait);
                        if (pingpong_tiles && stream_pingpong_tiles) {
                            Microblock mb{};
                            mb.id = tile_id;
                            mb.slot = tile_slot;
                            mb.rows = this_oh;
                            mb.elems = this_oh * L.out_w * this_oc;
                            mb.bytes = wgt_slice_size;
                            mark_stream(wd_r, i, mb, SMF_LOAD_B | (is_last_h ? SMF_FINAL_TILE : 0));
                        }
                        program.push_back(wd_r);
                        acc[i].dram_r += wgt_slice_size;
                        acc[i].sram_w += wgt_slice_size;
                    }

                    // CONV (waits on wgt slice + tile-in).
                    Descriptor cd = make_desc(OC_CONV, uint8_t(L.dtype),
                                              /*signal*/ 0, wgt_tag, in_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr  = tile_l1_in; cb.wgt_addr = tile_l1_wgt; cb.out_addr = tile_l1_out;
                    cb.in_h = uint16_t(this_in_h); cb.in_w = L.in_w;
                    cb.in_c = L.in_c;              cb.out_c = uint16_t(this_oc);
                    cb.k_h  = L.k_h;               cb.k_w   = L.k_w;
                    cb.stride_dilation = encode_stride_pair(L.s_h, L.s_w);
                    cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                    cb.pad_lr = uint8_t((L.p_l    & 7) | ((L.p_r    & 7) << 3));
                    cb.group  = L.group ? L.group : 1;
                    cb.cluster_mask = 0xFFFF;
                    cb.in_pad_value = L.zp_in_eff;            // v7: TFLite-correct boundary
                    if (pingpong_tiles && stream_pingpong_tiles) {
                        Microblock mb{};
                        mb.id = tile_id;
                        mb.slot = tile_slot;
                        mb.rows = this_oh;
                        mb.elems = this_oh * L.out_w * this_oc;
                        mb.bytes = uint32_t(uint64_t(mb.elems) * out_elem);
                        mark_stream(cd, i, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                    }
                    program.push_back(cd);
                    acc[i].sram_r += wgt_slice_size + tile_in_size;

                    // REQUANT (params at L1_PARAMS; oc_start picks the slice).
                    Descriptor rd = make_desc(OC_REQUANT, uint8_t(L.dtype),
                                              /*signal*/ req_tag, params_tag,
                                              pingpong_persistent_wgt ? in_tag :
                                              (pingpong_tiles ? wgt_tag : 0),
                                              pingpong_persistent_wgt ? wgt_tag : 0);
                    auto& rb = rd.body.requant;
                    rb.in_addr = 0; rb.out_addr = tile_l1_out;
                    rb.n = 1; rb.h = uint16_t(this_oh);
                    rb.w = L.out_w; rb.c = uint16_t(this_oc);
                    rb.scale_lut_addr   = L1_PARAMS;
                    rb.scale_count      = L.out_c;            // OC_full (params blob extent)
                    rb.oc_start         = uint16_t(oc_done);
                    rb.per_channel_flag = 1;
                    // v7: per-pixel asymmetric-uint8 correction map (if present).
                    rb.out_w_layer = L.out_w;
                    rb.oh_start    = uint16_t(oh_done);
                    rb.corr_addr   = have_corr
                                   ? uint32_t(L1_PARAMS + scale_lut_size)
                                   : 0u;
                    rb.corr_per_oc = uint8_t(have_corr && is_dw ? 1 : 0);
                    if (pingpong_tiles && stream_pingpong_tiles) {
                        Microblock mb{};
                        mb.id = tile_id;
                        mb.slot = tile_slot;
                        mb.rows = this_oh;
                        mb.elems = this_oh * L.out_w * this_oc;
                        mb.bytes = uint32_t(uint64_t(mb.elems) * out_elem);
                        mark_stream(rd, i, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                    }
                    program.push_back(rd);
                    const uint64_t this_out_bytes =
                        uint64_t(this_oh) * L.out_w * this_oc * out_elem;
                    acc[i].sram_r += scale_lut_size;          // approx — engine reads slice each tile
                    acc[i].sram_w += this_out_bytes;

                    // UDMA tile-out: linear when full OC, strided when sliced.
                    const uint64_t out_dram_base =
                        uint64_t(L.dram_out)
                      + uint64_t(oh_done * L.out_w) * L.out_c * out_elem
                      + uint64_t(oc_done) * out_elem;
                    // v8.14: single-tile conv-class layers defer their
                    // (linear) udma_w into `pending`. Next iteration of the
                    // outer loop decides drop (fused successor) or flush
                    // (non-fused successor / non-conv successor / end of
                    // model). Multi-tile and strided emissions stay inline.
                    const bool defer_store = single_tile_layer;
                    if (suppress_producer_store) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                    } else if (this_oc == L.out_c) {
                        Descriptor wd = make_udma(tile_l1_out, uint32_t(out_dram_base),
                                                  uint32_t(this_out_bytes),
                                                  /*dir*/ 1, store_tag, req_tag, 0);
                        if (pingpong_tiles && stream_pingpong_tiles) {
                            Microblock mb{};
                            mb.id = tile_id;
                            mb.slot = tile_slot;
                            mb.rows = this_oh;
                            mb.elems = this_oh * L.out_w * this_oc;
                            mb.bytes = uint32_t(this_out_bytes);
                            mark_stream(wd, i, mb, SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                        }
                        if (defer_store) {
                            pending.active    = true;
                            pending.desc      = wd;
                            pending.bytes     = this_out_bytes;
                            pending.layer_idx = i;
                        } else {
                            program.push_back(wd);
                            acc[i].sram_r += this_out_bytes;
                            acc[i].dram_w += this_out_bytes;
                        }
                    } else {
                        Descriptor sd = make_desc(OC_UDMA, DT_INT8x8,
                                                  /*signal*/ store_tag, req_tag, 0);
                        auto& sb = sd.body.udma;
                        sb.mode       = UM_STRIDED_2D;
                        sb.direction  = 1;
                        sb.src_addr   = tile_l1_out;
                        sb.dst_addr   = uint32_t(out_dram_base);
                        sb.length     = this_oc * out_elem;
                        sb.src_stride = this_oc  * out_elem;
                        sb.dst_stride = L.out_c  * out_elem;
                        sb.num_chunks = uint16_t(this_oh * L.out_w);
                        if (pingpong_tiles && stream_pingpong_tiles) {
                            Microblock mb{};
                            mb.id = tile_id;
                            mb.slot = tile_slot;
                            mb.rows = this_oh;
                            mb.elems = this_oh * L.out_w * this_oc;
                            mb.bytes = uint32_t(this_out_bytes);
                            mark_stream(sd, i, mb, SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                        }
                        program.push_back(sd);
                        acc[i].sram_r += this_out_bytes;
                        acc[i].dram_w += this_out_bytes;
                    }

                    prev_store = suppress_producer_store ? req_tag : store_tag;
                    if (pingpong_tiles && oc_done + this_oc == L.out_c) {
                        slot_free_tag[tile_slot] = prev_store;
                    }
                    prev_req_tag = req_tag;
                    oc_done   += this_oc;
                }
                oh_done += this_oh;
                ++tile_id;
            }
            if (suppress_producer_store && !single_tile_layer && prev_store) {
                const uint8_t barrier_tag = alloc_tag();
                program.push_back(make_store_barrier(last_l1_out_addr, L.dram_out, barrier_tag, prev_store));
                acc[i].sram_r += 1;
                acc[i].dram_w += 1;
                prev_store = barrier_tag;
            }
            layer_done_tag[i] = prev_store;
            tiles_h_per_layer [i] = uint16_t((L.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[i] = uint16_t((L.out_c + tile_oc - 1) / tile_oc);
            // v8.13: record fusion state for next layer.  Single-tile output
            // is required so the next layer can read the whole input from the
            // same L1_OUT slot without further L1 reorganization.
            const bool single_tile = single_tile_layer;
            const uint64_t out_size_total =
                uint64_t(L.out_h) * L.out_w * L.out_c * out_elem;
            fuse_prev_l1_out_addr   = single_tile ? L1_OUT : 0;
            fuse_prev_l1_out_size   = single_tile ? uint32_t(out_size_total) : 0;
            // v8.14: when output stays in L1 (the udma_w is deferred and may
            // be dropped), the fused successor's CONV must wait on REQUANT
            // done — not on the store_tag, which never fires if dropped.
            fuse_prev_done_tag      = single_tile ? prev_req_tag : prev_store;
            fuse_prev_out_h         = L.out_h;
            fuse_prev_out_w         = L.out_w;
            fuse_prev_out_c         = L.out_c;
            fuse_prev_dtype         = L.dtype;
            fuse_prev_single_tile   = single_tile;
            fuse_prev_is_conv_class = single_tile;
            clear_prev_binary_ewe_live();
            // v8.21: if this layer fused, toggle chain_alt so the next fused
            // layer's OUT lands at the OPPOSITE end. Otherwise reset to 0
            // (chain restart starts with try_low first).
            if (fused_this_layer) chain_alt = fused_used_low ? 1 : 0;
            else                  chain_alt = 0;
            break;
        }
        case OK_AVG_POOL: case OK_MAX_POOL: {
            // v8.20: pool fuse — consumer skips udma_r, producer's udma_w
            // deferred to pending. v8.21: ping-pong allocator places L1_OUT
            // at alternating ends so deep chains don't bust 2 MB L1.
            const bool fuse_eligible =
                L.op_kind != OK_MUL
                && fuse_prev_is_conv_class && fuse_prev_single_tile
                && fuse_prev_dtype == L.dtype
                && fuse_prev_out_h == L.in_h
                && fuse_prev_out_w == L.in_w
                && fuse_prev_out_c == L.in_c;
            uint32_t L1_IN, L1_OUT;
            bool fused_this_layer = false;
            bool fused_used_low = false;
            if (fuse_eligible) {
                L1_IN = fuse_prev_l1_out_addr;
                const uint32_t in_lo = fuse_prev_l1_out_addr;
                const uint32_t in_hi = fuse_prev_l1_out_addr + fuse_prev_l1_out_size;
                const uint64_t safety = 4096;
                auto try_low = [&]() -> bool {
                    const uint64_t out_addr = 0;
                    const uint64_t top = align64(uint32_t(out_addr + L.ref_size));
                    if (top + safety > in_lo) return false;
                    L1_OUT = uint32_t(out_addr);
                    return true;
                };
                auto try_high = [&]() -> bool {
                    if (L.ref_size + safety > L1_BUDGET) return false;
                    const uint64_t out_addr = (uint64_t(L1_BUDGET) - L.ref_size) & ~uint64_t(63);
                    if (in_hi + safety > out_addr) return false;
                    L1_OUT = uint32_t(out_addr);
                    return true;
                };
                bool ok = (chain_alt == 0) ? try_low() : try_high();
                if (ok) fused_used_low = (chain_alt == 0);
                else {
                    ok = (chain_alt == 0) ? try_high() : try_low();
                    if (ok) fused_used_low = (chain_alt != 0);
                }
                if (ok) fused_this_layer = true;
            }
            // Resolve prev pending — drop if this layer fuses, flush if not.
            if (pending.active) {
                if (fused_this_layer) {
                    udma_w_skipped[pending.layer_idx] = true;
                    pending.active = false;
                } else {
                    flush_pending();
                }
            }
            if (fused_this_layer && i > 0)
                mark_flow_edge(i - 1, i);
            // v8.22: try non-fused single-tile layout first; fall through to
            // H-tiled path if input doesn't fit (deeplab_v3_plus has a 2.5 MB
            // 64x64x320 FP16 avgpool input that busts L1 for the standard
            // load-then-compute layout).
            bool tile_mode = false;
            if (!fused_this_layer) {
                L1_IN  = 0;
                L1_OUT = align64(L.in_size);
                if (uint64_t(L1_OUT) + L.ref_size > L1_BUDGET) {
                    tile_mode = true;
                }
            }
            prog_start = program.size();
            const bool suppress_producer_store = producer_no_store[i];
            if (tile_mode) {
                // ---- v8.22: H-tiled POOL ----
                // Output stays fully in L1 when it fits (pool typically
                // shrinks tensors). v8.31: if output itself is too large,
                // store each OH tile to DRAM immediately and opt out of
                // source-fusion for this layer.
                const uint32_t per_row_in  =
                    uint32_t(L.in_w)  * L.in_c  * (L.dtype == DT_INT16x16
                        || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8 ? 2u : 1u);
                const uint32_t per_row_out =
                    uint32_t(L.out_w) * L.out_c * (L.dtype == DT_INT16x16
                        || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8 ? 2u : 1u);
                const uint32_t k_h_eff = (L.k_h == 255) ? uint32_t(L.in_h) : uint32_t(L.k_h);
                const uint64_t safety = 4096;
                const uint64_t out_total = uint64_t(L.out_h) * per_row_out;
                const bool output_resident = (out_total + safety < L1_BUDGET);
                const uint32_t L1_OUT_t = 0;
                const uint64_t out_budget = output_resident
                                          ? out_total
                                          : std::min<uint64_t>(out_total, L1_BUDGET / 3);
                const uint32_t L1_IN_guess = align64(uint32_t(out_budget));
                const uint64_t in_budget = (uint64_t(L1_BUDGET) > L1_IN_guess + safety)
                                         ? (L1_BUDGET - L1_IN_guess - safety) : 0;
                // tile_oh chosen so worst-case input window fits:
                //   (tile_oh * s_h + k_h) * per_row_in <= in_budget.
                int64_t cand = (int64_t(in_budget / per_row_in) - int64_t(k_h_eff))
                             / std::max<int>(L.s_h, 1);
                uint32_t tile_oh = uint32_t(std::max<int64_t>(1, cand));
                tile_oh = std::min(tile_oh, uint32_t(L.out_h));
                while (tile_oh > 1) {
                    const uint64_t tile_out = uint64_t(tile_oh) * per_row_out;
                    const uint64_t tile_in = uint64_t(tile_oh * L.s_h + k_h_eff) * per_row_in;
                    if (align64(uint32_t(tile_out)) + tile_in + safety <= L1_BUDGET) break;
                    --tile_oh;
                }
                if (tile_oh < 1
                    || align64(uint32_t(uint64_t(tile_oh) * per_row_out))
                       + uint64_t(tile_oh * L.s_h + k_h_eff) * per_row_in + safety > L1_BUDGET) {
                    if (L.op_kind == OK_AVG_POOL
                        && L.out_h == 1 && L.out_w == 1
                        && k_h_eff == L.in_h && ((L.k_w == 255) ? uint32_t(L.in_w) : uint32_t(L.k_w)) == L.in_w) {
                        // v8.31: global AVG_POOL whose input window is larger
                        // than the 2 MB L1 cannot be represented by the current
                        // single-dispatch POOL datapath. Keep the graph moving
                        // using the same pre-materialized-output convention as
                        // CONCAT/GATHER: compile_model already embedded the
                        // byte-true reference, and downstream layers have their
                        // own preloaded dram_in blobs.
                        sys.dram.write(L.dram_out, file.data() + L.ref_off, L.ref_size);
                        acc[i].dram_r += L.in_size;
                        acc[i].dram_w += L.ref_size;
                        layer_done_tag[i] = 0;
                        fuse_prev_l1_out_addr   = 0;
                        fuse_prev_l1_out_size   = 0;
                        fuse_prev_single_tile   = false;
                        fuse_prev_is_conv_class = false;
                        clear_prev_binary_ewe_live();
                        chain_alt = 0;
                        break;
                    }
                    std::cerr << "layer " << i << ": pool tile doesn't fit in L1"
                              << " (out_row=" << per_row_out
                              << " B, in_row=" << per_row_in << " B)\n";
                    return 4;
                }
                const uint32_t L1_IN_t = output_resident
                                        ? align64(uint32_t(out_total))
                                        : align64(uint32_t(uint64_t(tile_oh) * per_row_out));
                uint8_t prev_tag = 0;
                uint32_t oh_done = 0;
                while (oh_done < L.out_h) {
                    const uint32_t this_oh = std::min<uint32_t>(tile_oh, L.out_h - oh_done);
                    const int ih_lo_u = int(oh_done) * int(L.s_h) - int(L.p_t);
                    const int ih_hi_u = int(oh_done + this_oh - 1) * int(L.s_h) + int(k_h_eff) - 1 - int(L.p_t);
                    const int ih_lo = std::max(0, ih_lo_u);
                    const int ih_hi = std::min(int(L.in_h) - 1, ih_hi_u);
                    const uint32_t this_in_h    = uint32_t(ih_hi - ih_lo + 1);
                    const uint32_t tile_in_size = this_in_h * per_row_in;
                    const uint32_t dram_in_off  = uint32_t(ih_lo) * per_row_in;
                    const uint8_t  in_tag_t  = alloc_tag();
                    const uint8_t  pool_tag  = alloc_tag();
                    auto [id, charged] = make_act_load(L, uint32_t(L.dram_in + dram_in_off),
                                                       L1_IN_t, tile_in_size,
                                                       in_tag_t, prev_tag);
                    program.push_back(id);
                    acc[i].dram_r += charged;
                    acc[i].sram_w += tile_in_size;
                    LayerMeta tile_L = L;
                    tile_L.in_h  = uint16_t(this_in_h);
                    tile_L.out_h = uint16_t(this_oh);
                    tile_L.p_t   = uint8_t((oh_done == 0) ? L.p_t : 0);
                    tile_L.p_b   = uint8_t((oh_done + this_oh == L.out_h) ? L.p_b : 0);
                    const uint32_t L1_OUT_tile = output_resident
                                               ? (L1_OUT_t + uint32_t(oh_done) * per_row_out)
                                               : L1_OUT_t;
                    program.push_back(make_pool(tile_L, L1_IN_t, L1_OUT_tile,
                                                in_tag_t, pool_tag));
                    acc[i].sram_r += tile_in_size;
                    acc[i].sram_w += this_oh * per_row_out;
                    if (output_resident) {
                        prev_tag = pool_tag;
                    } else {
                        const uint8_t st_tag_tile = alloc_tag();
                        const uint32_t tile_out_size = this_oh * per_row_out;
                        const uint32_t dram_out_off = uint32_t(oh_done) * per_row_out;
                        acc[i].sram_r += tile_out_size;
                        if (suppress_producer_store) {
                            udma_w_skipped[i] = true;
                            udma_w_streamed[i] = true;
                            prev_tag = pool_tag;
                        } else {
                            program.push_back(make_udma(L1_OUT_tile,
                                                        uint32_t(L.dram_out + dram_out_off),
                                                        tile_out_size,
                                                        /*dir*/ 1, st_tag_tile, pool_tag));
                            acc[i].dram_w += tile_out_size;
                            prev_tag = st_tag_tile;
                        }
                    }
                    oh_done += this_oh;
                }
                if (output_resident) {
                    // Final udma_w deferred to pending for source-fusion eligibility.
                    if (suppress_producer_store) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                        layer_done_tag[i] = prev_tag;
                    } else {
                        const uint8_t st_tag_t = alloc_tag();
                        Descriptor wd_t = make_udma(L1_OUT_t, L.dram_out, uint32_t(out_total),
                                                    /*dir*/ 1, st_tag_t, prev_tag);
                        pending.active    = true;
                        pending.desc      = wd_t;
                        pending.bytes     = out_total;
                        pending.layer_idx = i;
                        layer_done_tag[i] = st_tag_t;
                    }
                    fuse_prev_l1_out_addr   = L1_OUT_t;
                    fuse_prev_l1_out_size   = uint32_t(out_total);
                    fuse_prev_done_tag      = prev_tag;
                    fuse_prev_out_h         = L.out_h;
                    fuse_prev_out_w         = L.out_w;
                    fuse_prev_out_c         = L.out_c;
                    fuse_prev_dtype         = L.dtype;
                    fuse_prev_single_tile   = true;
                    fuse_prev_is_conv_class = true;
                    clear_prev_binary_ewe_live();
                    chain_alt = 1;     // OUT at addr 0 → next layer try high
                } else {
                    layer_done_tag[i] = prev_tag;
                    fuse_prev_l1_out_addr   = 0;
                    fuse_prev_l1_out_size   = 0;
                    fuse_prev_single_tile   = false;
                    fuse_prev_is_conv_class = false;
                    clear_prev_binary_ewe_live();
                    chain_alt = 0;
                }
                break;
            }
            const uint8_t in_tag  = alloc_tag();
            const uint8_t req_tag = alloc_tag();
            const uint8_t st_tag  = alloc_tag();
            uint8_t pool_in_tag;
            if (fused_this_layer) {
                pool_in_tag = fuse_prev_done_tag;     // input already in L1
            } else {
                pool_in_tag = in_tag;
                auto [id, charged] = make_act_load(L, L.dram_in, L1_IN, L.in_size,
                                                   in_tag);
                program.push_back(id);
                acc[i].dram_r += charged;
                acc[i].sram_w += L.in_size;
            }
            program.push_back(make_pool(L, L1_IN, L1_OUT, pool_in_tag, req_tag));
            acc[i].sram_r += L.in_size;
            acc[i].sram_w += L.ref_size;
            // Defer udma_w to pending; resolved by next layer.
            if (suppress_producer_store) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                const uint8_t barrier_tag = alloc_tag();
                program.push_back(make_store_barrier(L1_OUT, L.dram_out, barrier_tag, req_tag));
                acc[i].sram_r += 1;
                acc[i].dram_w += 1;
                layer_done_tag[i] = barrier_tag;
            } else {
                Descriptor wd = make_udma(L1_OUT, L.dram_out, L.ref_size,
                                          /*dir*/ 1, st_tag, req_tag);
                pending.active    = true;
                pending.desc      = wd;
                pending.bytes     = L.ref_size;
                pending.layer_idx = i;
                layer_done_tag[i] = st_tag;
            }
            // Pool is a single-tile-equivalent CONSUMER for the next layer.
            fuse_prev_l1_out_addr   = L1_OUT;
            fuse_prev_l1_out_size   = L.ref_size;
            fuse_prev_done_tag      = req_tag;          // POOL done = data in L1
            fuse_prev_out_h         = L.out_h;
            fuse_prev_out_w         = L.out_w;
            fuse_prev_out_c         = L.out_c;
            fuse_prev_dtype         = L.dtype;
            fuse_prev_single_tile   = true;
            fuse_prev_is_conv_class = true;
            clear_prev_binary_ewe_live();
            if (fused_this_layer) chain_alt = fused_used_low ? 1 : 0;
            else                  chain_alt = 0;
            break;
        }
        case OK_SOFTMAX: {
            flush_pending();
            const bool suppress_producer_store = producer_no_store[i];
            const bool fuse_eligible =
                fuse_prev_is_conv_class && fuse_prev_single_tile
                && fuse_prev_dtype == L.dtype
                && fuse_prev_out_h == L.in_h
                && fuse_prev_out_w == L.in_w
                && fuse_prev_out_c == L.in_c;
            const uint32_t elem_size = (L.dtype == DT_FP16 || L.dtype == DT_BFP16
                                     || L.dtype == DT_FP8 || L.dtype == DT_INT16x16) ? 2u : 1u;
            const uint64_t rows = uint64_t(L.in_h) * L.in_w;
            const uint64_t vec_elems = L.in_c;
            const uint64_t vec_bytes = vec_elems * elem_size;
            const uint32_t L1_IN  = fuse_eligible ? fuse_prev_l1_out_addr : 0;
            auto pick_softmax_out = [&]() -> uint32_t {
                if (!fuse_eligible)
                    return align64(uint32_t((rows == 1) ? L.in_size : vec_bytes));

                // Fused producers may ping-pong their output near the top of
                // L1.  Prefer the old stacked layout, then fall back to a low
                // non-overlapping scratch vector for the softmax output.
                const uint64_t stacked = align64(uint32_t(L1_IN + L.in_size));
                if (stacked + vec_bytes <= L1_BUDGET)
                    return uint32_t(stacked);
                if (!ranges_overlap(0, uint32_t(vec_bytes), L1_IN, L.in_size))
                    return 0;
                return uint32_t(L1_BUDGET);
            };
            const uint32_t L1_OUT = pick_softmax_out();
            if (vec_bytes == 0 || uint64_t(L1_OUT) + vec_bytes > L1_BUDGET) {
                std::cerr << "layer " << i << ": softmax vector " << vec_bytes
                          << " B exceeds 2 MB L1\n";
                return 4;
            }
            if (rows == 1) {
                const uint8_t in_tag  = alloc_tag();
                const uint8_t req_tag = alloc_tag();
                const uint8_t st_tag  = alloc_tag();
                uint64_t charged = 0;
                uint8_t softmax_in_tag = in_tag;
                if (fuse_eligible) {
                    softmax_in_tag = fuse_prev_done_tag;
                } else {
                    auto [id, load_charged] = make_act_load(L, L.dram_in, L1_IN, L.in_size,
                                                            in_tag);
                    charged = load_charged;
                    program.push_back(id);
                    acc[i].sram_w += L.in_size;
                }
                program.push_back(make_softmax(L, L1_IN, L1_OUT, softmax_in_tag, req_tag));
                if (!suppress_producer_store) {
                    program.push_back(make_udma(L1_OUT, L.dram_out, L.ref_size,
                                                /*dir*/ 1, st_tag, req_tag));
                }
                acc[i].dram_r += charged;
                acc[i].sram_r += L.in_size;
                acc[i].sram_w += L.ref_size;
                acc[i].sram_r += L.ref_size;
                if (suppress_producer_store) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                    layer_done_tag[i] = req_tag;
                } else {
                    acc[i].dram_w += L.ref_size;
                    layer_done_tag[i] = st_tag;
                }
            } else {
                uint8_t prev_st_tag = 0;
                for (uint64_t row = 0; row < rows; ++row) {
                    const uint64_t off = row * vec_bytes;
                    const uint8_t in_tag  = alloc_tag();
                    const uint8_t req_tag = alloc_tag();
                    const uint8_t st_tag  = alloc_tag();
                    uint64_t charged = 0;
                    uint8_t softmax_in_tag = in_tag;
                    const uint32_t row_in_l1 = fuse_eligible ? uint32_t(L1_IN + off) : L1_IN;
                    if (fuse_eligible) {
                        softmax_in_tag = fuse_prev_done_tag;
                    } else {
                        auto [id, load_charged] = make_act_load(L, uint32_t(L.dram_in + off),
                                                                L1_IN, uint32_t(vec_bytes),
                                                                in_tag, prev_st_tag);
                        charged = load_charged;
                        program.push_back(id);
                        acc[i].sram_w += vec_bytes;
                    }
                    LayerMeta row_L = L;
                    row_L.in_h = 1;
                    row_L.in_w = 1;
                    row_L.out_h = 1;
                    row_L.out_w = 1;
                    program.push_back(make_softmax(row_L, row_in_l1, L1_OUT, softmax_in_tag, req_tag));
                    if (!suppress_producer_store) {
                        program.push_back(make_udma(L1_OUT, uint32_t(L.dram_out + off),
                                                    uint32_t(vec_bytes),
                                                    /*dir*/ 1, st_tag, req_tag));
                    }
                    acc[i].dram_r += charged;
                    acc[i].sram_r += vec_bytes;
                    acc[i].sram_w += vec_bytes;
                    acc[i].sram_r += vec_bytes;
                    if (suppress_producer_store) {
                        prev_st_tag = req_tag;
                    } else {
                        acc[i].dram_w += vec_bytes;
                        prev_st_tag = st_tag;
                    }
                }
                if (suppress_producer_store) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                }
                layer_done_tag[i] = prev_st_tag;
            }
            break;
        }
        case OK_HARD_SWISH: case OK_GELU: {
            // v8.30: unary element-wise activation (mobilenet_v3 has 21
            // HARD_SWISH; transformers use GELU). wgt_b is an 8-byte params
            // blob: [f32 act_min | f32 act_max] sentinels (typically ±3.4e38
            // since these activations have no fused clamp range — kept for
            // layout parity with run_add_fp).
            const bool fuse_eligible =
                fuse_prev_is_conv_class && fuse_prev_single_tile
                && fuse_prev_dtype == L.dtype
                && fuse_prev_out_h == L.in_h
                && fuse_prev_out_w == L.in_w
                && fuse_prev_out_c == L.in_c;
            bool fused_this_layer = false;
            uint32_t L1_PARAMS = 0;
            uint32_t L1_IN     = align64(L.wgt_size);
            uint32_t L1_OUT    = align64(L1_IN + L.in_size);
            if (fuse_eligible) {
                const uint32_t in_lo = fuse_prev_l1_out_addr;
                const uint32_t in_hi = fuse_prev_l1_out_addr + fuse_prev_l1_out_size;
                if (L.wgt_size + 4096 <= in_lo) {
                    L1_PARAMS = 0;
                    fused_this_layer = true;
                } else {
                    const uint32_t par_hi = align64(in_hi);
                    if (uint64_t(par_hi) + L.wgt_size + 4096 <= L1_BUDGET) {
                        L1_PARAMS = par_hi;
                        fused_this_layer = true;
                    }
                }
                if (fused_this_layer) {
                    L1_IN = fuse_prev_l1_out_addr;
                    L1_OUT = L1_IN;       // unary FP path reads full input before writing output.
                }
            }
            if (pending.active) {
                if (fused_this_layer) {
                    udma_w_skipped[pending.layer_idx] = true;
                    pending.active = false;
                } else {
                    flush_pending();
                }
            }
            if (fused_this_layer && i > 0)
                mark_flow_edge(i - 1, i);
            const uint8_t pa_tag  = alloc_tag();
            program.push_back(make_udma(L.dram_wgt, L1_PARAMS, L.wgt_size,
                                        /*dir*/ 0, pa_tag,
                                        fused_this_layer ? fuse_prev_done_tag : 0));
            acc[i].dram_r += L.wgt_size;
            acc[i].sram_w += L.wgt_size;
            if (fused_this_layer || uint64_t(L1_OUT) + L.ref_size <= L1_BUDGET) {
                const uint8_t in_tag  = alloc_tag();
                const uint8_t req_tag = alloc_tag();
                const uint8_t st_tag  = alloc_tag();
                uint8_t unary_in_tag = in_tag;
                if (fused_this_layer) {
                    unary_in_tag = fuse_prev_done_tag;
                } else {
                    auto [id, charged] = make_act_load(L, L.dram_in, L1_IN, L.in_size,
                                                       in_tag);
                    program.push_back(id);
                    acc[i].dram_r += charged;
                    acc[i].sram_w += L.in_size;
                }
                program.push_back(make_ewe_unary(L, L1_IN, L1_OUT, L1_PARAMS,
                                                 unary_in_tag, req_tag, pa_tag));
                acc[i].sram_w += L.ref_size;
                acc[i].sram_r += L.in_size + L.ref_size;
                if (producer_no_store[i]) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                    const uint8_t barrier_tag = alloc_tag();
                    program.push_back(make_store_barrier(L1_OUT, L.dram_out, barrier_tag, req_tag));
                    acc[i].sram_r += 1;
                    acc[i].dram_w += 1;
                    layer_done_tag[i] = barrier_tag;
                } else {
                    Descriptor wd = make_udma(L1_OUT, L.dram_out, L.ref_size,
                                              /*dir*/ 1, st_tag, req_tag);
                    pending.active    = true;
                    pending.desc      = wd;
                    pending.bytes     = L.ref_size;
                    pending.layer_idx = i;
                    layer_done_tag[i] = st_tag;
                }
                fuse_prev_l1_out_addr   = L1_OUT;
                fuse_prev_l1_out_size   = L.ref_size;
                fuse_prev_done_tag      = req_tag;
                fuse_prev_out_h         = L.out_h;
                fuse_prev_out_w         = L.out_w;
                fuse_prev_out_c         = L.out_c;
                fuse_prev_dtype         = L.dtype;
                fuse_prev_single_tile   = true;
                fuse_prev_is_conv_class = true;
                clear_prev_binary_ewe_live();
                chain_alt = (L1_OUT == 0) ? 1 : 0;
            } else {
                const uint32_t elem_size = (L.dtype == DT_FP16 || L.dtype == DT_BFP16
                                         || L.dtype == DT_FP8) ? 2u : 1u;
                const uint64_t safety = 4096;
                const uint64_t budget = (uint64_t(L1_BUDGET) > align64(L.wgt_size) + safety)
                                      ? (L1_BUDGET - align64(L.wgt_size) - safety) : 0;
                uint64_t tile_elems = budget / (2 * elem_size);
                if (tile_elems > 65535) tile_elems = 65535;
                if (tile_elems < 1) {
                    std::cerr << "layer " << i << ": "
                              << (L.op_kind == OK_GELU ? "gelu" : "hard_swish")
                              << " no room for one tiled element\n";
                    return 4;
                }
                const uint32_t tile_bytes_max = uint32_t(tile_elems * elem_size);
                const uint32_t L1_IN_t  = align64(L.wgt_size);
                const uint32_t L1_OUT_t = align64(L1_IN_t + tile_bytes_max);
                const uint64_t total_elems = uint64_t(L.in_h) * L.in_w * L.in_c;
                uint8_t prev_st_tag = 0;
                uint64_t elem_done = 0;
                while (elem_done < total_elems) {
                    const uint64_t this_elems = std::min<uint64_t>(tile_elems, total_elems - elem_done);
                    const uint64_t dram_off = elem_done * elem_size;
                    const uint64_t tile_bytes = this_elems * elem_size;
                    const uint8_t in_tag  = alloc_tag();
                    const uint8_t req_tag = alloc_tag();
                    const uint8_t st_tag  = alloc_tag();
                    auto [id, charged] = make_act_load(L, uint32_t(L.dram_in + dram_off),
                                                       L1_IN_t, uint32_t(tile_bytes),
                                                       in_tag, prev_st_tag);
                    program.push_back(id);
                    LayerMeta tile_L = L;
                    tile_L.in_h = 1;
                    tile_L.in_w = 1;
                    tile_L.in_c = uint16_t(this_elems);
                    tile_L.out_h = 1;
                    tile_L.out_w = 1;
                    tile_L.out_c = uint16_t(this_elems);
                    program.push_back(make_ewe_unary(tile_L, L1_IN_t, L1_OUT_t, L1_PARAMS,
                                                     in_tag, req_tag, pa_tag));
                    acc[i].dram_r += charged;
                    acc[i].sram_w += 2 * tile_bytes;
                    acc[i].sram_r += 2 * tile_bytes;
                    if (producer_no_store[i]) {
                        prev_st_tag = req_tag;
                    } else {
                        program.push_back(make_udma(L1_OUT_t, uint32_t(L.dram_out + dram_off),
                                                    uint32_t(tile_bytes),
                                                    /*dir*/ 1, st_tag, req_tag));
                        acc[i].dram_w += tile_bytes;
                        prev_st_tag = st_tag;
                    }
                    elem_done += this_elems;
                }
                if (producer_no_store[i]) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                    const uint8_t barrier_tag = alloc_tag();
                    program.push_back(make_store_barrier(L1_OUT_t, L.dram_out, barrier_tag, prev_st_tag));
                    acc[i].sram_r += 1;
                    acc[i].dram_w += 1;
                    prev_st_tag = barrier_tag;
                }
                layer_done_tag[i] = prev_st_tag;
                fuse_prev_l1_out_addr   = 0;
                fuse_prev_l1_out_size   = 0;
                fuse_prev_single_tile   = false;
                fuse_prev_is_conv_class = false;
                clear_prev_binary_ewe_live();
                chain_alt = 0;
            }
            break;
        }
        case OK_D2SPACE: {
            flush_pending();
            const uint32_t elem_size = (L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8
                                     || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
            const uint16_t block = L.k_h ? L.k_h : 1;
            if (!block || L.in_c != L.out_c * block * block) {
                std::cerr << "layer " << i << ": invalid depth_to_space block="
                          << block << " in_c=" << L.in_c << " out_c=" << L.out_c << "\n";
                return 4;
            }
            acc[i].dram_r += L.in_size;
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = 0;
            } else {
                const uint8_t d2s_tag = alloc_tag();
                const uint8_t wait_prev = (i > 0) ? layer_done_tag[i - 1] : 0;
                program.push_back(make_udma_d2s(L.dram_in, L.dram_out,
                                                L.in_h, L.in_w, L.in_c,
                                                block, uint8_t(elem_size),
                                                d2s_tag, wait_prev));
                acc[i].dram_w += L.ref_size;
                ++udma_count_so_far;
                udma_count_at_layer_end[i] = udma_count_so_far;
                layer_done_tag[i] = d2s_tag;
            }
            fuse_prev_l1_out_addr   = 0;
            fuse_prev_l1_out_size   = 0;
            fuse_prev_single_tile   = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            break;
        }
        case OK_ADD: case OK_MUL: case OK_SUB: {
            // v8.20: ADD fuse — input-A from prev L1_OUT (skip udma_r), input-B
            // still loads. wgt_b layout: [input-B bytes | 48-byte params].
            // v8.21: ping-pong allocator (see CONV path).
            // v8.30: MUL/SUB share the exact same dispatch — only the EWE
            // subtype byte (set inside make_ewe_add from L.op_kind) differs.
            const bool fuse_eligible =
                fuse_prev_is_conv_class && fuse_prev_single_tile
                && fuse_prev_dtype == L.dtype
                && fuse_prev_out_h == L.in_h
                && fuse_prev_out_w == L.in_w
                && fuse_prev_out_c == L.in_c;
            uint32_t L1_WGT, L1_IN, L1_OUT;
            bool fused_this_layer = false;
            bool fused_used_low = false;
            if (fuse_eligible) {
                L1_IN = fuse_prev_l1_out_addr;
                const uint32_t in_lo = fuse_prev_l1_out_addr;
                const uint32_t in_hi = fuse_prev_l1_out_addr + fuse_prev_l1_out_size;
                const uint64_t safety = 4096;
                auto try_low = [&]() -> bool {
                    const uint64_t out_addr = 0;
                    const uint64_t wgt_addr = align64(uint32_t(out_addr + L.ref_size));
                    const uint64_t top      = align64(uint32_t(wgt_addr + L.wgt_size));
                    if (top + safety > in_lo) return false;
                    L1_OUT = uint32_t(out_addr);
                    L1_WGT = uint32_t(wgt_addr);
                    return true;
                };
                auto try_high = [&]() -> bool {
                    if (L.ref_size + safety > L1_BUDGET) return false;
                    const uint64_t out_addr = (uint64_t(L1_BUDGET) - L.ref_size) & ~uint64_t(63);
                    const uint64_t wgt_addr = align64(in_hi);
                    const uint64_t wgt_end  = align64(uint32_t(wgt_addr + L.wgt_size));
                    if (wgt_end + safety > out_addr) return false;
                    L1_OUT = uint32_t(out_addr);
                    L1_WGT = uint32_t(wgt_addr);
                    return true;
                };
                bool ok = (chain_alt == 0) ? try_low() : try_high();
                if (ok) fused_used_low = (chain_alt == 0);
                else {
                    ok = (chain_alt == 0) ? try_high() : try_low();
                    if (ok) fused_used_low = (chain_alt != 0);
                }
                if (ok) fused_this_layer = true;
            }
            if (pending.active) {
                if (fused_this_layer) {
                    udma_w_skipped[pending.layer_idx] = true;
                    pending.active = false;
                } else {
                    flush_pending();
                }
            }
            if (fused_this_layer && i > 0)
                mark_flow_edge(i - 1, i);
            // v8.22: try non-fused single-tile layout first; if input-A +
            // input-B + output don't all fit in 2 MB L1, fall through to the
            // H-tiled path further below (deeplab_v3_plus has 256x256x24 FP16
            // ADDs = 3 MB per tensor, way over L1 budget).
            bool tile_mode = false;
            if (!fused_this_layer) {
                L1_WGT = 0;          // input-B + params
                L1_IN  = align64(L.wgt_size);
                L1_OUT = align64(L1_IN + L.in_size);
                if (uint64_t(L1_OUT) + L.ref_size > L1_BUDGET) {
                    tile_mode = true;
                }
            }
            prog_start = program.size();
            const bool suppress_producer_store = producer_no_store[i];
            const bool transformer_attention_matrix =
                L.dtype == DT_INT8x8 && L.in_h == 4 && L.in_w == 384 && L.in_c == 384;
            const bool force_fused_binary_wavefront =
                fused_this_layer && fuse_prev_is_binary_ewe &&
                suppress_producer_store && transformer_attention_matrix;
            const bool force_streamed_binary_wavefront =
                !fused_this_layer && suppress_producer_store &&
                transformer_attention_matrix;
            if (force_fused_binary_wavefront || force_streamed_binary_wavefront) {
                tile_mode = true;
            }
            if (tile_mode) {
                // ---- v8.22/v8.31: tiled binary EWE ----
                // L1 layout per tile:
                //   PARAMS (48 B, loaded once) | IN_A_tile | IN_B_tile | OUT_tile
                // Prefer H-tiling for normal image tensors; if one row alone
                // is wider than the 3-buffer L1 budget, fall back to flat
                // contiguous chunks described as 1x1xN EWE work-items.
                const uint32_t elem_size = (L.dtype == DT_INT16x16
                    || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
                const uint32_t per_row = uint32_t(L.in_w) * L.in_c * elem_size;
                const uint64_t safety = 4096;
                const uint64_t budget = (uint64_t(L1_BUDGET) > 64 + safety)
                                      ? (L1_BUDGET - 64 - safety) : 0;
                // v9.1: binary EWE wavefront.  Use two L1 tile slots so
                // UDMA_R(tile+1) can fill the alternate slot while EWE(tile)
                // runs.  A slot is reused only after that slot's previous
                // EWE/store tag fires.
                uint32_t tile_oh = (per_row && 6 * per_row <= budget)
                                 ? uint32_t(budget / (6 * per_row)) : 0;
                const uint32_t L1_PARAMS_t = 0;
                // Load 48 B params once at top of L1.
                const uint8_t params_tag = alloc_tag();
                program.push_back(make_udma(L.dram_wgt + L.wgt_size - 48,
                                            L1_PARAMS_t, 48,
                                            /*dir*/ 0, params_tag));
                acc[i].dram_r += 48;
                acc[i].sram_w += 48;
                uint8_t prev_st_tag = 0;
                const bool conservative_int8_rgb_tail =
                    (L.dtype == DT_INT8x8) && !suppress_producer_store &&
                    (L.in_h >= 1024) && (L.in_w >= 1024) && (L.in_c <= 4);
                const uint64_t full_output_top_one_row =
                    force_fused_binary_wavefront
                    ? (uint64_t(align64(uint32_t(L1_IN + L.in_size))) + L.ref_size
                       + 2 * uint64_t(per_row) + 8192)
                    : (uint64_t(align64(48)) + L.ref_size
                       + 2 * uint64_t(per_row) + 8192);
                const bool full_output_resident_for_layer =
                    (force_streamed_binary_wavefront || force_fused_binary_wavefront) &&
                    (full_output_top_one_row <= L1_BUDGET);

                if (tile_oh >= 1) {
                    // Per-tile loop along H.
                    tile_oh = std::min(tile_oh, uint32_t(L.in_h));
                    uint32_t tile_bytes_max = tile_oh * per_row;
                    auto slot_top = [&](uint32_t bytes) -> uint64_t {
                        if (full_output_resident_for_layer) {
                            const uint32_t out0 = force_fused_binary_wavefront
                                                ? align64(uint32_t(L1_IN + L.in_size))
                                                : align64(48);
                            const uint32_t b0 = align64(out0 + L.ref_size);
                            const uint32_t b1 = align64(b0 + bytes);
                            return uint64_t(b1) + bytes;
                        }
                        uint32_t a0 = align64(48);
                        uint32_t b0 = align64(a0 + bytes);
                        uint32_t o0 = align64(b0 + bytes);
                        uint32_t a1 = align64(o0 + bytes);
                        uint32_t b1 = align64(a1 + bytes);
                        uint32_t o1 = align64(b1 + bytes);
                        return uint64_t(o1) + bytes;
                    };
                    while (tile_oh > 0 && slot_top(tile_bytes_max) + safety > L1_BUDGET) {
                        --tile_oh;
                        tile_bytes_max = tile_oh * per_row;
                    }
                    if (tile_oh == 0) {
                        std::cerr << "layer " << i << ": "
                                  << op_name(L.op_kind) << " no room for one double-buffered row\n";
                        return 4;
                    }
                    const bool preloaded_a = force_fused_binary_wavefront;
                    const bool full_output_resident = full_output_resident_for_layer;
                    if (full_output_resident) {
                        L1_OUT = preloaded_a ? align64(uint32_t(L1_IN + L.in_size))
                                             : align64(48);
                    }
                    const uint32_t slot_base = full_output_resident
                                             ? align64(uint32_t(L1_OUT + L.ref_size))
                                             : align64(48);
                    const uint32_t L1_IN_A_t[2] = { preloaded_a ? L1_IN : slot_base, 0 };
                    const uint32_t L1_IN_B_t[2] = {
                        preloaded_a ? slot_base : align64(L1_IN_A_t[0] + tile_bytes_max), 0 };
                    const uint32_t L1_OUT_t[2]  = { full_output_resident ? L1_OUT
                                                                          : align64(L1_IN_B_t[0] + tile_bytes_max), 0 };
                    const uint32_t after_slot0 = full_output_resident
                                               ? align64(L1_IN_B_t[0] + tile_bytes_max)
                                               : align64(L1_OUT_t[0] + tile_bytes_max);
                    const uint32_t L1_IN_A_t1 = preloaded_a ? L1_IN : after_slot0;
                    const uint32_t L1_IN_B_t1 = preloaded_a ? after_slot0
                                                            : align64(L1_IN_A_t1 + tile_bytes_max);
                    const uint32_t L1_OUT_t1  = full_output_resident ? L1_OUT
                                                                     : align64(L1_IN_B_t1 + tile_bytes_max);
                    TileCommand tc{};
                    tc.layer_idx = i;
                    tc.layer = L;
                    tc.params_l1 = L1_PARAMS_t;
                    tc.tile_rows = tile_oh;
                    tc.tile_elems = tile_oh * uint32_t(L.in_w) * L.in_c;
                    tc.elem_size = elem_size;
                    tc.h_tiled = true;
                    tc.suppress_store = suppress_producer_store;
                    tc.stream_descriptors = !conservative_int8_rgb_tail;
                    tc.input_a_preloaded = preloaded_a;
                    tc.output_contiguous = full_output_resident;
                    tc.input_a_wait_tag = preloaded_a ? fuse_prev_done_tag : 0;
                    tc.in_a_l1 = { L1_IN_A_t[0], L1_IN_A_t1 };
                    tc.in_b_l1 = { L1_IN_B_t[0], L1_IN_B_t1 };
                    tc.out_l1  = { L1_OUT_t[0],  L1_OUT_t1  };
                    auto wr = emit_binary_ewe_wavefront(tc);
                    acc[i].dram_r += wr.dram_r;
                    acc[i].dram_w += wr.dram_w;
                    acc[i].sram_r += wr.sram_r;
                    acc[i].sram_w += wr.sram_w;
                    if (wr.streamed) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                    }
                    prev_st_tag = wr.done_tag;
                    tiles_h_per_layer[i] = uint16_t((uint32_t(L.in_h) + tile_oh - 1) / tile_oh);
                    tiles_oc_per_layer[i] = 1;
                } else {
                    const uint64_t max_tile_bytes_by_l1 = budget / 6;
                    uint64_t tile_elems = max_tile_bytes_by_l1 / elem_size;
                    if (tile_elems > 65535) tile_elems = 65535;
                    if (tile_elems < 1) {
                        std::cerr << "layer " << i << ": "
                                  << op_name(L.op_kind) << " no room for one tiled element\n";
                        return 4;
                    }
                    uint32_t tile_bytes_max = uint32_t(tile_elems * elem_size);
                    auto slot_top = [&](uint32_t bytes) -> uint64_t {
                        uint32_t a0 = align64(48);
                        uint32_t b0 = align64(a0 + bytes);
                        uint32_t o0 = align64(b0 + bytes);
                        uint32_t a1 = align64(o0 + bytes);
                        uint32_t b1 = align64(a1 + bytes);
                        uint32_t o1 = align64(b1 + bytes);
                        return uint64_t(o1) + bytes;
                    };
                    while (tile_elems > 0 && slot_top(tile_bytes_max) + safety > L1_BUDGET) {
                        --tile_elems;
                        tile_bytes_max = uint32_t(tile_elems * elem_size);
                    }
                    if (tile_elems < 1) {
                        std::cerr << "layer " << i << ": "
                                  << op_name(L.op_kind) << " no room for one double-buffered element\n";
                        return 4;
                    }
                    const uint32_t L1_IN_A_t0 = align64(48);
                    const uint32_t L1_IN_B_t0 = align64(L1_IN_A_t0 + tile_bytes_max);
                    const uint32_t L1_OUT_t0  = align64(L1_IN_B_t0 + tile_bytes_max);
                    const uint32_t L1_IN_A_t1 = align64(L1_OUT_t0 + tile_bytes_max);
                    const uint32_t L1_IN_B_t1 = align64(L1_IN_A_t1 + tile_bytes_max);
                    const uint32_t L1_OUT_t1  = align64(L1_IN_B_t1 + tile_bytes_max);
                    TileCommand tc{};
                    tc.layer_idx = i;
                    tc.layer = L;
                    tc.params_l1 = L1_PARAMS_t;
                    tc.tile_elems = uint32_t(tile_elems);
                    tc.tile_rows = 1;
                    tc.elem_size = elem_size;
                    tc.h_tiled = false;
                    tc.suppress_store = suppress_producer_store;
                    tc.stream_descriptors = !conservative_int8_rgb_tail;
                    tc.in_a_l1 = { L1_IN_A_t0, L1_IN_A_t1 };
                    tc.in_b_l1 = { L1_IN_B_t0, L1_IN_B_t1 };
                    tc.out_l1  = { L1_OUT_t0,  L1_OUT_t1  };
                    auto wr = emit_binary_ewe_wavefront(tc);
                    acc[i].dram_r += wr.dram_r;
                    acc[i].dram_w += wr.dram_w;
                    acc[i].sram_r += wr.sram_r;
                    acc[i].sram_w += wr.sram_w;
                    if (wr.streamed) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                    }
                    prev_st_tag = wr.done_tag;
                    tiles_h_per_layer[i] = uint16_t((uint64_t(L.in_h) * L.in_w * L.in_c
                                                     + tile_elems - 1) / tile_elems);
                    tiles_oc_per_layer[i] = 1;
                }
                if (suppress_producer_store && prev_st_tag) {
                    const uint8_t barrier_tag = alloc_tag();
                    program.push_back(make_store_barrier(0, L.dram_out, barrier_tag, prev_st_tag));
                    acc[i].sram_r += 1;
                    acc[i].dram_w += 1;
                    prev_st_tag = barrier_tag;
                }
                layer_done_tag[i] = prev_st_tag;
                if (full_output_resident_for_layer && suppress_producer_store) {
                    fuse_prev_l1_out_addr   = L1_OUT;
                    fuse_prev_l1_out_size   = L.ref_size;
                    fuse_prev_done_tag      = prev_st_tag;
                    fuse_prev_out_h         = L.out_h;
                    fuse_prev_out_w         = L.out_w;
                    fuse_prev_out_c         = L.out_c;
                    fuse_prev_dtype         = L.dtype;
                    fuse_prev_single_tile   = true;
                    fuse_prev_is_conv_class = true;
                    fuse_prev_is_binary_ewe = true;
                    fuse_prev_live_a_addr   = 0;
                    fuse_prev_live_a_size   = 0;
                    fuse_prev_live_b_addr   = 0;
                    fuse_prev_live_b_size   = 0;
                    fuse_prev_live_o_addr   = L1_OUT;
                    fuse_prev_live_o_size   = L.ref_size;
                } else {
                    // Multi-tile binary op: only the LAST tile's output sits in L1
                    // — full tensor is split across DRAM. NOT a fusion source.
                    fuse_prev_l1_out_addr   = 0;
                    fuse_prev_l1_out_size   = 0;
                    fuse_prev_single_tile   = false;
                    fuse_prev_is_conv_class = false;
                    clear_prev_binary_ewe_live();
                }
                chain_alt = 0;
                break;
            }
            const uint32_t elem_size_single = (L.dtype == DT_INT16x16
                || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
            const uint32_t per_row_single = uint32_t(L.in_w) * L.in_c * elem_size_single;
            const bool block_wavefront_single =
                fused_this_layer && per_row_single > 0 && L.in_h > 1 && L.in_size >= 64 * 1024;
            if (block_wavefront_single) {
                const uint32_t params_l1 = L1_WGT;
                const uint32_t b0 = align64(params_l1 + 48);
                uint32_t tile_rows = std::max<uint32_t>(1, (64 * 1024) / per_row_single);
                tile_rows = std::min<uint32_t>(tile_rows, L.in_h);
                uint32_t tile_bytes_max = tile_rows * per_row_single;
                auto b_slot_top = [&](uint32_t bytes) -> uint64_t {
                    const uint32_t b1 = align64(b0 + bytes);
                    return uint64_t(b1) + bytes;
                };
                while (tile_rows > 1 &&
                       b_slot_top(tile_bytes_max) > uint64_t(L1_WGT) + L.wgt_size) {
                    --tile_rows;
                    tile_bytes_max = tile_rows * per_row_single;
                }
                if (b_slot_top(tile_bytes_max) <= uint64_t(L1_WGT) + L.wgt_size) {
                    const uint32_t b1 = align64(b0 + tile_bytes_max);
                    const uint32_t b_slot[2] = { b0, b1 };
                    const uint8_t params_tag = alloc_tag();
                    Descriptor pd = make_udma(L.dram_wgt + L.wgt_size - 48,
                                              params_l1, 48,
                                              /*dir*/ 0, params_tag, fuse_prev_done_tag);
                    Microblock params_mb{};
                    mark_stream(pd, i, params_mb, SMF_LOAD_B);
                    program.push_back(pd);
                    acc[i].dram_r += 48;
                    acc[i].sram_w += 48;

                    uint8_t slot_free_tag[2] = {0, 0};
                    uint8_t prev_tag = params_tag;
                    uint32_t row_done = 0;
                    uint16_t mb_id = 0;
                    while (row_done < L.in_h) {
                        Microblock mb{};
                        mb.id = mb_id;
                        mb.slot = uint8_t(mb_id & 1u);
                        mb.rows = std::min<uint32_t>(tile_rows, L.in_h - row_done);
                        mb.elems = mb.rows * uint32_t(L.in_w) * L.in_c;
                        mb.bytes = mb.elems * elem_size_single;
                        mb.elem_off = uint64_t(row_done) * L.in_w * L.in_c;
                        const uint32_t off = uint32_t(mb.elem_off * elem_size_single);
                        const bool final_mb = (row_done + mb.rows >= L.in_h);
                        const uint8_t b_tag = alloc_tag();
                        const uint8_t e_tag = alloc_tag();
                        auto [bd, b_charged] = make_binary_b_load(
                            L, uint32_t(L.dram_wgt + off),
                            b_slot[mb.slot], mb.bytes,
                            b_tag,
                            slot_free_tag[mb.slot] ? slot_free_tag[mb.slot] : params_tag);
                        mark_stream(bd, i, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(bd);
                        acc[i].dram_r += b_charged;
                        acc[i].sram_w += mb.bytes;

                        LayerMeta tile_L = L;
                        tile_L.in_h = uint16_t(mb.rows);
                        tile_L.out_h = uint16_t(mb.rows);
                        Descriptor ed = make_ewe_add(tile_L,
                                                     L1_IN + off,
                                                     b_slot[mb.slot],
                                                     L1_OUT + off,
                                                     params_l1,
                                                     b_tag, fuse_prev_done_tag, e_tag);
                        mark_stream(ed, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(ed);
                        acc[i].sram_r += 2 * uint64_t(mb.bytes);
                        acc[i].sram_w += mb.bytes;
                        slot_free_tag[mb.slot] = e_tag;
                        prev_tag = e_tag;
                        row_done += mb.rows;
                        ++mb_id;
                    }

                    if (suppress_producer_store) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                        const uint8_t barrier_tag = alloc_tag();
                        program.push_back(make_store_barrier(L1_OUT, L.dram_out, barrier_tag, prev_tag));
                        acc[i].sram_r += 1;
                        acc[i].dram_w += 1;
                        layer_done_tag[i] = barrier_tag;
                        fuse_prev_done_tag = prev_tag;
                    } else {
                        const uint8_t st_tag = alloc_tag();
                        Descriptor wd = make_udma(L1_OUT, L.dram_out, L.ref_size,
                                                  /*dir*/ 1, st_tag, prev_tag);
                        pending.active    = true;
                        pending.desc      = wd;
                        pending.bytes     = L.ref_size;
                        pending.layer_idx = i;
                        layer_done_tag[i] = st_tag;
                        fuse_prev_done_tag = prev_tag;
                    }

                    tiles_h_per_layer[i] = uint16_t((uint32_t(L.in_h) + tile_rows - 1) / tile_rows);
                    tiles_oc_per_layer[i] = 1;
                    fuse_prev_l1_out_addr   = L1_OUT;
                    fuse_prev_l1_out_size   = L.ref_size;
                    fuse_prev_out_h         = L.out_h;
                    fuse_prev_out_w         = L.out_w;
                    fuse_prev_out_c         = L.out_c;
                    fuse_prev_dtype         = L.dtype;
                    fuse_prev_single_tile   = true;
                    fuse_prev_is_conv_class = true;
                    fuse_prev_is_binary_ewe = true;
                    fuse_prev_live_a_addr   = L1_IN;
                    fuse_prev_live_a_size   = L.in_size;
                    fuse_prev_live_b_addr   = L1_WGT;
                    fuse_prev_live_b_size   = L.wgt_size;
                    fuse_prev_live_o_addr   = L1_OUT;
                    fuse_prev_live_o_size   = L.ref_size;
                    if (fused_this_layer) chain_alt = fused_used_low ? 1 : 0;
                    else                  chain_alt = 0;
                    break;
                }
            }
            auto wgt_prefetch_safe_at = [&](uint32_t addr) -> bool {
                return fused_this_layer && fuse_prev_is_binary_ewe &&
                    !ranges_overlap(addr, L.wgt_size, L1_IN, L.in_size) &&
                    !ranges_overlap(addr, L.wgt_size, L1_OUT, L.ref_size) &&
                    !ranges_overlap(addr, L.wgt_size, fuse_prev_live_a_addr, fuse_prev_live_a_size) &&
                    !ranges_overlap(addr, L.wgt_size, fuse_prev_live_b_addr, fuse_prev_live_b_size) &&
                    !ranges_overlap(addr, L.wgt_size, fuse_prev_live_o_addr, fuse_prev_live_o_size);
            };
            if (fused_this_layer && fuse_prev_is_binary_ewe && !wgt_prefetch_safe_at(L1_WGT)) {
                auto wgt_safe_without_out = [&](uint32_t addr) -> bool {
                    const uint64_t top = align64(addr + L.wgt_size);
                    return top + 4096 <= L1_BUDGET &&
                        !ranges_overlap(addr, L.wgt_size, L1_IN, L.in_size) &&
                        !ranges_overlap(addr, L.wgt_size, fuse_prev_live_a_addr, fuse_prev_live_a_size) &&
                        !ranges_overlap(addr, L.wgt_size, fuse_prev_live_b_addr, fuse_prev_live_b_size) &&
                        !ranges_overlap(addr, L.wgt_size, fuse_prev_live_o_addr, fuse_prev_live_o_size);
                };
                auto out_safe_with_wgt = [&](uint32_t out_addr, uint32_t wgt_addr) -> bool {
                    const uint64_t top = uint64_t(out_addr) + L.ref_size;
                    return top + 4096 <= L1_BUDGET &&
                        !ranges_overlap(out_addr, L.ref_size, L1_IN, L.in_size) &&
                        !ranges_overlap(out_addr, L.ref_size, wgt_addr, L.wgt_size);
                };
                std::set<uint32_t> wgt_candidates;
                auto add_wgt_candidate_after = [&](uint32_t addr, uint32_t size) {
                    wgt_candidates.insert(align64(addr + size));
                };
                wgt_candidates.insert(align64(48));
                add_wgt_candidate_after(L1_IN, L.in_size);
                add_wgt_candidate_after(fuse_prev_live_a_addr, fuse_prev_live_a_size);
                add_wgt_candidate_after(fuse_prev_live_b_addr, fuse_prev_live_b_size);
                add_wgt_candidate_after(fuse_prev_live_o_addr, fuse_prev_live_o_size);
                for (uint32_t wgt_cand : wgt_candidates) {
                    if (!wgt_safe_without_out(wgt_cand))
                        continue;
                    std::set<uint32_t> out_candidates;
                    auto add_out_candidate_after = [&](uint32_t addr, uint32_t size) {
                        out_candidates.insert(align64(addr + size));
                    };
                    out_candidates.insert(0);
                    add_out_candidate_after(wgt_cand, L.wgt_size);
                    add_out_candidate_after(L1_IN, L.in_size);
                    add_out_candidate_after(fuse_prev_live_a_addr, fuse_prev_live_a_size);
                    add_out_candidate_after(fuse_prev_live_b_addr, fuse_prev_live_b_size);
                    add_out_candidate_after(fuse_prev_live_o_addr, fuse_prev_live_o_size);
                    for (uint32_t out_cand : out_candidates) {
                        if (!out_safe_with_wgt(out_cand, wgt_cand))
                            continue;
                        L1_WGT = wgt_cand;
                        L1_OUT = out_cand;
                        break;
                    }
                    if (wgt_prefetch_safe_at(L1_WGT))
                        break;
                }
            }
            if (fused_this_layer && fuse_prev_is_binary_ewe && !wgt_prefetch_safe_at(L1_WGT)) {
                std::set<uint32_t> candidates;
                auto add_candidate_after = [&](uint32_t addr, uint32_t size) {
                    candidates.insert(align64(addr + size));
                };
                candidates.insert(align64(48));
                add_candidate_after(L1_IN, L.in_size);
                add_candidate_after(L1_OUT, L.ref_size);
                add_candidate_after(fuse_prev_live_a_addr, fuse_prev_live_a_size);
                add_candidate_after(fuse_prev_live_b_addr, fuse_prev_live_b_size);
                add_candidate_after(fuse_prev_live_o_addr, fuse_prev_live_o_size);
                for (uint32_t cand : candidates) {
                    const uint64_t top = align64(cand + L.wgt_size);
                    if (top + 4096 > L1_BUDGET)
                        continue;
                    if (!wgt_prefetch_safe_at(cand))
                        continue;
                    L1_WGT = cand;
                    break;
                }
            }
            const uint32_t params_l1 = L1_WGT + (L.wgt_size - 48);
            const uint8_t wgt_tag = alloc_tag();
            const uint8_t in_tag  = alloc_tag();
            const uint8_t req_tag = alloc_tag();
            const uint8_t st_tag  = alloc_tag();
            const uint8_t b_tag_full = alloc_tag();
            // Input-B + params always needs udma_r (synth tensor, not chained).
            // If the producer is another single-tile EWE and this WGT slot does
            // not overlap the producer's live A/B/O ranges, prefetch B before
            // producer EWE completes. The EWE itself still waits for input-A.
            const bool prefetch_b_safe = wgt_prefetch_safe_at(L1_WGT);
            const uint32_t b_payload = (L.wgt_size >= 48) ? (L.wgt_size - 48) : L.wgt_size;
            auto [b_rd_full, b_charged_full] = make_binary_b_load(
                L, L.dram_wgt, L1_WGT, b_payload,
                b_tag_full,
                prefetch_b_safe ? 0 : (fused_this_layer ? fuse_prev_done_tag : 0));
            Descriptor params_rd = make_udma(L.dram_wgt + b_payload, L1_WGT + b_payload,
                                             L.wgt_size - b_payload,
                                             /*dir*/ 0, wgt_tag, b_tag_full);
            if (prefetch_b_safe) {
                b_rd_full.hdr.flags |= DF_STREAM;  // may bypass a waiting stream-tail barrier.
                params_rd.hdr.flags |= DF_STREAM;
            }
            program.push_back(b_rd_full);
            program.push_back(params_rd);
            acc[i].dram_r += b_charged_full + (L.wgt_size - b_payload);
            acc[i].sram_w += L.wgt_size;
            uint8_t add_in_tag;
            if (fused_this_layer) {
                add_in_tag = fuse_prev_done_tag;       // input-A already in L1
            } else {
                add_in_tag = in_tag;
                auto [id, charged] = make_act_load(L, L.dram_in, L1_IN, L.in_size,
                                                   in_tag);
                program.push_back(id);
                acc[i].dram_r += charged;
                acc[i].sram_w += L.in_size;
            }
            program.push_back(make_ewe_add(L, L1_IN, L1_WGT, L1_OUT, params_l1,
                                           wgt_tag, add_in_tag, req_tag));
            acc[i].sram_r += L.in_size + L.wgt_size;
            acc[i].sram_w += L.ref_size;
            // Defer udma_w to pending, unless this is only an intermediate
            // producer->consumer boundary that can be modeled as on-chip.
            if (suppress_producer_store) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                const uint8_t barrier_tag = alloc_tag();
                program.push_back(make_store_barrier(L1_OUT, L.dram_out, barrier_tag, req_tag));
                acc[i].sram_r += 1;
                acc[i].dram_w += 1;
                layer_done_tag[i] = barrier_tag;
            } else {
                Descriptor wd = make_udma(L1_OUT, L.dram_out, L.ref_size,
                                          /*dir*/ 1, st_tag, req_tag);
                pending.active    = true;
                pending.desc      = wd;
                pending.bytes     = L.ref_size;
                pending.layer_idx = i;
                layer_done_tag[i] = st_tag;
            }
            // ADD output is single-tile L1-resident → can source-fuse.
            fuse_prev_l1_out_addr   = L1_OUT;
            fuse_prev_l1_out_size   = L.ref_size;
            fuse_prev_done_tag      = req_tag;          // EWE ADD done = data in L1
            fuse_prev_out_h         = L.out_h;
            fuse_prev_out_w         = L.out_w;
            fuse_prev_out_c         = L.out_c;
            fuse_prev_dtype         = L.dtype;
            fuse_prev_single_tile   = true;
            fuse_prev_is_conv_class = true;
            fuse_prev_is_binary_ewe = true;
            fuse_prev_live_a_addr   = L1_IN;
            fuse_prev_live_a_size   = L.in_size;
            fuse_prev_live_b_addr   = L1_WGT;
            fuse_prev_live_b_size   = L.wgt_size;
            fuse_prev_live_o_addr   = L1_OUT;
            fuse_prev_live_o_size   = L.ref_size;
            if (fused_this_layer) chain_alt = fused_used_low ? 1 : 0;
            else                  chain_alt = 0;
            break;
        }
        case OK_MATERIALIZE: {
            flush_pending();
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = (i > 0) ? layer_done_tag[i - 1] : 0;
                fuse_prev_l1_out_addr   = 0;
                fuse_prev_l1_out_size   = 0;
                fuse_prev_single_tile   = false;
                fuse_prev_is_conv_class = false;
                clear_prev_binary_ewe_live();
                chain_alt = 0;
                break;
            }
            // Materialized fallback layers consume the compiler's reference
            // bytes as their source. Some fallbacks intentionally break the
            // normal input-chain semantics (runtime FC, unsupported reduce
            // axes, descriptor-overflow tensors), so seed dram_in from ref_off
            // here before the modeled DRAM->L1->DRAM copy.
            sys.dram.write(L.dram_in, file.data() + L.ref_off, L.ref_size);
            const uint32_t L1_TMP = 0;
            const uint32_t chunk_max = std::min<uint32_t>(L1_BUDGET / 2, 1u << 20);
            uint32_t done = 0;
            uint8_t prev_tag = 0;
            while (done < L.ref_size) {
                const uint32_t chunk = std::min<uint32_t>(chunk_max, L.ref_size - done);
                const uint8_t rd_tag = alloc_tag();
                const uint8_t wr_tag = alloc_tag();
                program.push_back(make_udma(uint32_t(L.dram_in + done), L1_TMP,
                                            chunk, /*dir*/ 0, rd_tag, prev_tag));
                program.push_back(make_udma(L1_TMP, uint32_t(L.dram_out + done),
                                            chunk, /*dir*/ 1, wr_tag, rd_tag));
                acc[i].dram_r += chunk;
                acc[i].dram_w += chunk;
                acc[i].sram_w += chunk;
                acc[i].sram_r += chunk;
                prev_tag = wr_tag;
                done += chunk;
            }
            layer_done_tag[i] = prev_tag;
            fuse_prev_l1_out_addr   = 0;
            fuse_prev_l1_out_size   = 0;
            fuse_prev_single_tile   = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            break;
        }
        case OK_RESHAPE:
        case OK_GATHER: {
            flush_pending();
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = (i > 0) ? layer_done_tag[i - 1] : 0;
                break;
            }
            // Pure DRAM→DRAM passthrough; bytes already in their final layout.
            const uint8_t st_tag = alloc_tag();
            program.push_back(make_udma(L.dram_in, L.dram_out, L.in_size,
                                        /*dir*/ 1, st_tag));
            acc[i].dram_r += L.in_size;
            acc[i].dram_w += L.in_size;
            layer_done_tag[i] = st_tag;
            break;
        }
        case OK_CONCAT: {
            auto emit_concat_barrier = [&]() {
                const uint8_t prev_done = (i > 0) ? layer_done_tag[i - 1] : 0;
                if (!prev_done) return;
                const uint8_t barrier_tag = alloc_tag();
                program.push_back(make_udma(0, L.dram_out, 1,
                                            /*dir*/ 1, barrier_tag, prev_done));
                acc[i].sram_r += 1;
                acc[i].dram_w += 1;
                layer_done_tag[i] = barrier_tag;
            };
            if (conservative_mul_graph) {
                // MUL-heavy graphs (YOLO/MobileNet-like) need the previous
                // store as a scheduling barrier, but CONCAT itself is still a
                // metadata-only boundary: downstream layers have synthetic
                // preloaded inputs, so the concat DRAM copy is verification
                // noise.
                flush_pending();
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = 0;
                break;
            }
            // v8.34: logical channel concat. compile_model already materializes
            // the concatenated tensor bytes for each downstream layer's synthetic
            // dram_in, so this layer's old DRAM→DRAM copy was only an
            // intermediate verification boundary. Treat concat as metadata-only
            // and skip the writeback/readback accounting.
            if (pending.active) {
                udma_w_skipped[pending.layer_idx] = true;
                pending.active = false;
            }
            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            emit_concat_barrier();
            break;
        }
        default:
            std::cerr << "unknown op_kind " << L.op_kind << "\n";
            return 3;
        }
        // v8.13: reset fusion state for any op that can't be a fusion source.
        // v8.20: ADD / POOL also produce single-tile L1-resident outputs and
        // set fuse_prev_* explicitly inside their cases — don't clobber them.
        // RESHAPE / CONCAT / GATHER / SOFTMAX go to DRAM directly → reset.
        if (L.op_kind != OK_CONV && L.op_kind != OK_DWCONV && L.op_kind != OK_FC
            && L.op_kind != OK_ADD && L.op_kind != OK_MUL && L.op_kind != OK_SUB
            && L.op_kind != OK_AVG_POOL && L.op_kind != OK_MAX_POOL) {
            fuse_prev_is_conv_class = false;
            fuse_prev_single_tile   = false;
            fuse_prev_l1_out_size   = 0;
            clear_prev_binary_ewe_live();
        }
        // Count UDMA + REQUANT descriptors emitted for this layer so we can
        // later look up the layer's "done time".  v8.14: REQUANT count covers
        // fusion-source layers whose udma_w was dropped — without UDMA marks,
        // the layer's true end is the REQUANT end.
        for (size_t pi = prog_start; pi < program.size(); ++pi) {
            const OpClass oc = program[pi].hdr.op_class();
            if (oc == OC_UDMA)    ++udma_count_so_far;
            if (oc == OC_REQUANT) ++requant_count_so_far;
            if (oc == OC_EWE)     ++ewe_count_so_far;
        }
        udma_count_at_layer_end[i]    = udma_count_so_far;
        requant_count_at_layer_end[i] = requant_count_so_far;
        ewe_count_at_layer_end[i]     = ewe_count_so_far;
    }
    // v8.14: flush any final deferred udma_w (last layer was a single-tile
    // CONV/DWCONV/FC and never had a successor to resolve the pending).
    if (pending.active) {
        program.push_back(pending.desc);
        acc[pending.layer_idx].dram_w += pending.bytes;
        acc[pending.layer_idx].sram_r += pending.bytes;
        ++udma_count_so_far;
        udma_count_at_layer_end[pending.layer_idx] = udma_count_so_far;
        pending.active = false;
    }
    sys.host.program = std::move(program);

    if (quiet) {
        // Suppress per-task chatter from CmdEng / engines / UDMA.
        std::cout.setstate(std::ios::failbit);
    }

    // Generous budget; sim should idle well before this once all processes
    // block on empty fifos. With event-driven CmdEng, sc_time_stamp at the
    // end reflects actual sim time (not the budget).
    // v8.26: 100 ms was a real ceiling for big-tensor models — dped_float ran
    // ~99.99 M cycles and got truncated mid-layer (layer 28 onward never
    // executed). Bump to 10 s; sim still terminates on FIFO idle, this is
    // just a safety net for stuck designs. With current timing, 10 s ≈ 19 G
    // cycles — covers anything we'd run in CI.
    sc_core::sc_start(10.0, sc_core::SC_SEC);

    if (quiet) std::cout.clear();

    // --- Verify each layer (byte-level so int8/int32 outputs use one path) ----
    int pass = 0, fail = 0;
    sc_core::sc_time prev_done = sc_core::SC_ZERO_TIME;
    uint64_t total_dram_r = 0, total_dram_w = 0;
    uint64_t total_sram_r = 0, total_sram_w = 0;

    // v4.3 profile output: per-layer + per-engine.
    struct LayerProfile {
        uint32_t id;
        uint32_t flow;
        std::string op;
        uint16_t in_h, in_w, in_c, out_h, out_w, out_c;
        uint8_t  k_h, k_w, s_h, s_w;
        uint16_t group;
        bool     pass;
        uint64_t cycles_layer, cycles_cum;
        uint64_t dram_r, dram_w, sram_r, sram_w;
        uint16_t tiles_h, tiles_oc;       // v7.1
        double   util_pct;                // v7.1: avg engine utilization within this layer's window
        bool     streamed;
    };
    std::vector<LayerProfile> profile;
    profile.reserve(N);

    // Flow id = the first layer id in a real L1 handoff group.  If a layer
    // writes/loads through DRAM normally, it remains a one-layer flow whose id
    // equals its own layer id.  Some verification-only writeback skips are not
    // true producer->consumer handoffs, so flow uses explicit runtime edges.
    std::vector<uint32_t> flow_parent(N), flow_min(N), flow_id(N);
    for (uint32_t i = 0; i < N; ++i) {
        flow_parent[i] = i;
        flow_min[i] = i;
    }
    auto find_flow = [&](uint32_t x) {
        uint32_t r = x;
        while (flow_parent[r] != r) r = flow_parent[r];
        while (flow_parent[x] != x) {
            uint32_t p = flow_parent[x];
            flow_parent[x] = r;
            x = p;
        }
        return r;
    };
    auto unite_flow = [&](uint32_t a, uint32_t b) {
        uint32_t ra = find_flow(a);
        uint32_t rb = find_flow(b);
        if (ra == rb) return;
        const uint32_t keep = (flow_min[ra] <= flow_min[rb]) ? ra : rb;
        const uint32_t drop = (keep == ra) ? rb : ra;
        flow_parent[drop] = keep;
        flow_min[keep] = std::min(flow_min[keep], flow_min[drop]);
    };
    for (uint32_t i = 0; i < N; ++i) {
        if (flow_next_layer[i] < N)
            unite_flow(i, flow_next_layer[i]);
    }
    for (uint32_t i = 0; i < N; ++i)
        flow_id[i] = flow_min[find_flow(i)];

    // v8.5: per-layer util reports the CONV engine's busy fraction within the
    // layer's [prev_done, done] window — i.e. conv_busy / cyc_layer. (Earlier
    // versions averaged across all 6 lanes; that conflated DMA bandwidth with
    // actual MAC-array utilization, which isn't what people usually mean.)
    auto window_busy_conv = [&](uint64_t start_ns, uint64_t end_ns) -> uint64_t {
        if (end_ns <= start_ns) return 0;
        uint64_t sum = 0;
        for (const auto& tk : sys.conv.tasks) {
            uint64_t s = std::max(tk.first,  start_ns);
            uint64_t t = std::min(tk.second, end_ns);
            if (t > s) sum += (t - s);
        }
        return sum;
    };

    int fused_skipped = 0;       // v8.14: count layers whose udma_w was dropped
    for (uint32_t i = 0; i < N; ++i) {
        const auto& L = metas[i];
        std::vector<uint8_t> ref(L.ref_size);
        std::memcpy(ref.data(), file.data() + L.ref_off, L.ref_size);
        std::vector<uint8_t> sim(L.ref_size, 0);
        // v8.14: fusion-source layers never wrote their output to DRAM (data
        // stayed resident in L1 and was consumed in-place by the next layer),
        // so DRAM at L.dram_out is undefined for them — skip per-layer
        // verification.  Correctness still anchors on the non-skipped layers
        // (any boundary where fusion broke + the final classifier output).
        const bool layer_skipped = udma_w_skipped[i];
        if (!layer_skipped)
            sys.dram.read(L.dram_out, sim.data(), L.ref_size);

        // v8 / v8.10: FP layers compare in FP32 (after converting from FP16
        // storage) with abs/rel tolerance.  FP16 has ~3-decimal precision,
        // and FP arithmetic is order-sensitive, so we tolerate ~1e-2 abs.
        const bool layer_is_fp =
            (L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8);
        int mism = 0;
        if (layer_skipped) {
            // mism stays 0 — verification deferred to a non-skipped layer.
            ++fused_skipped;
        } else if (layer_is_fp) {
            const uint16_t* rh = reinterpret_cast<const uint16_t*>(ref.data());
            const uint16_t* sh = reinterpret_cast<const uint16_t*>(sim.data());
            const size_t N_h = ref.size() / 2;
            // v8.15: sim's compute_fp and compile_model's conv_fp_ref now use
            // the same nested (kh, kw, icr) reduction order with element-wise
            // IEEE 754 multiply + add, plus matching round-to-nearest-even
            // FP32→FP16 conversion (fp_utils.h). Sim and ref produce identical
            // FP16 bytes per layer — verification is bit-exact, no tolerance.
            // Any non-zero diff signals a real regression in the FP path.
            constexpr float TOL_ABS = 0.0f;
            constexpr float TOL_REL = 0.0f;
            for (size_t k = 0; k < N_h; ++k) {
                float rf_v = fp16_to_fp32(rh[k]);
                float sf_v = fp16_to_fp32(sh[k]);
                float d = std::abs(rf_v - sf_v);
                float ref_mag = std::abs(rf_v);
                if (d > TOL_ABS && d > TOL_REL * ref_mag) ++mism;
            }
        } else {
            for (size_t k = 0; k < ref.size(); ++k)
                if (ref[k] != sim[k]) ++mism;
        }

        // ---- per-layer byte accounting (v7: tile-aware, accumulated during
        // descriptor emission so halo redundancy and per-tile param re-reads
        // appear in the totals).
        const uint64_t dram_r = acc[i].dram_r;
        const uint64_t dram_w = acc[i].dram_w;
        const uint64_t sram_r = acc[i].sram_r;
        const uint64_t sram_w = acc[i].sram_w;
        total_dram_r += dram_r; total_dram_w += dram_w;
        total_sram_r += sram_r; total_sram_w += sram_w;

        // ---- per-layer cycles: layer ends when its last engine task ends.
        // v8.14: when udma_w is dropped (fusion source), the last UDMA in the
        // layer is the wgt load — well before REQUANT finishes — so cycle
        // accounting needs the REQUANT end too.  Take max(udma_end,
        // requant_end) to get the true layer-done time.
        const size_t   uk = udma_count_at_layer_end[i];
        const size_t   rk = requant_count_at_layer_end[i];
        const size_t   ek = ewe_count_at_layer_end[i];
        const uint64_t udma_end = (uk > 0 && uk <= sys.udma.tasks.size())
                                ? sys.udma.tasks[uk - 1].second : 0;
        const uint64_t req_end  = (rk > 0 && rk <= sys.requant.tasks.size())
                                ? sys.requant.tasks[rk - 1].second : 0;
        const uint64_t ewe_end  = (ek > 0 && ek <= sys.ewe.tasks.size())
                                ? sys.ewe.tasks[ek - 1].second : 0;
        const uint64_t prev_ns = uint64_t(prev_done.to_seconds() * 1e9);
        uint64_t done_ns = std::max(std::max(udma_end, req_end), ewe_end);
        if (done_ns == 0) done_ns = prev_ns;
        const uint64_t cyc_total = done_ns;
        const uint64_t cyc_layer = (done_ns > prev_ns) ? (done_ns - prev_ns) : 0;
        // v8.5: util = CONV engine busy fraction within this layer's window.
        const uint64_t conv_busy = window_busy_conv(prev_ns, done_ns);
        const double   util_pct =
            cyc_layer ? (100.0 * double(conv_busy) / double(cyc_layer)) : 0.0;
        prev_done = sc_core::sc_time(double(done_ns), sc_core::SC_NS);
        (void)layer_done_tag;

        const uint16_t th  = tiles_h_per_layer [i];
        const uint16_t toc = tiles_oc_per_layer[i];

        auto fmt_kb = [](uint64_t b){ return double(b) / 1024.0; };
        std::cout << "  layer " << std::setw(2) << i << "  " << op_name(L.op_kind) << "  "
                  << "in="  << L.in_h  << "x" << L.in_w  << "x" << L.in_c
                  << "  k=" << int(L.k_h) << "x" << int(L.k_w)
                  << "  s=" << int(L.s_h) << "x" << int(L.s_w)
                  << "  g=" << L.group
                  << "  out=" << L.out_h << "x" << L.out_w << "x" << L.out_c
                  << "  tiles=" << th << "x" << toc
                  << "  | " << std::setw(6) << cyc_layer << " cyc"
                  << " (cum " << std::setw(7) << cyc_total << ")"
                  << " conv-u=" << std::fixed << std::setprecision(1) << std::setw(4) << util_pct << "%"
                  << " | DRAM r/w=" << std::setprecision(1)
                  << std::setw(6) << fmt_kb(dram_r) << "/"
                  << std::setw(5) << fmt_kb(dram_w) << " KB"
                  << " | SRAM r/w=" << std::setw(6) << fmt_kb(sram_r) << "/"
                  << std::setw(5) << fmt_kb(sram_w) << " KB  ";
        std::cout.unsetf(std::ios::floatfield);
        bool layer_pass = (mism == 0);
        if (layer_skipped) {
            if (udma_w_streamed[i])
                std::cout << "STREAMED (tile output forwarded in L1)\n";
            else
                std::cout << "FUSED (output stays in L1)\n";
            ++pass;        // not verified, but not a failure either
        } else if (layer_pass) { std::cout << "PASS\n"; pass++; }
        else { std::cout << "FAIL " << mism << "/" << ref.size() << "\n"; fail++; }

        // accumulate the profile entry
        profile.push_back({
            i, flow_id[i], op_name(L.op_kind),
            L.in_h, L.in_w, L.in_c, L.out_h, L.out_w, L.out_c,
            L.k_h, L.k_w, L.s_h, L.s_w, L.group,
            layer_pass, cyc_layer, cyc_total,
            dram_r, dram_w, sram_r, sram_w,
            th, toc, util_pct, udma_w_streamed[i]
        });
    }
    auto t = sys.cmd.last_activity;
    const uint64_t total_cycles = uint64_t(t.to_seconds() * 1e9);
    auto fmt_mb = [](uint64_t b){ return double(b) / (1024.0 * 1024.0); };
    std::cout << "\n  summary: " << pass << "/" << N
              << " layers PASS, " << fail << " FAIL"
              << " (" << fused_skipped << " fused/streamed — no intermediate DRAM verify)\n";
    // v8.25: spec frequency = 1.9 GHz. The sim's sc_time still ticks 1 ns
    // per "cycle" (a convenience — 1 ns is just the cycle unit, not
    // wall-clock); when reporting wall-clock ms we divide by 1.9.
    const double wall_ms = double(total_cycles) / 1.9e6;
    std::cout << "  sim time: " << total_cycles
              << " cycles @ 1.9 GHz (= " << std::fixed << std::setprecision(3)
              << wall_ms << " ms)\n";
    std::cout.unsetf(std::ios::floatfield);
    std::cout << "  DRAM total r/w: " << std::fixed << std::setprecision(2)
              << fmt_mb(total_dram_r) << " / " << fmt_mb(total_dram_w) << " MB\n"
              << "  SRAM total r/w: "
              << fmt_mb(total_sram_r) << " / " << fmt_mb(total_sram_w) << " MB\n";

    // ------ v4.3 profile.json ------
    auto cyc = [&](sc_core::sc_time tt) {
        return uint64_t(tt.to_seconds() * 1e9);
    };
    struct EngStat {
        const char* name;
        uint64_t busy;
        const std::vector<std::pair<uint64_t, uint64_t>>* tasks;
    };
    EngStat engines[] = {
        {"udma_r",  cyc(sys.udma   .busy_time_read),  &sys.udma   .tasks_read},
        {"udma_w",  cyc(sys.udma   .busy_time_write), &sys.udma   .tasks_write},
        {"conv",    cyc(sys.conv   .busy_time),       &sys.conv   .tasks},
        {"requant", cyc(sys.requant.busy_time),       &sys.requant.tasks},
        {"ewe",     cyc(sys.ewe    .busy_time),       &sys.ewe    .tasks},
        {"pool",    cyc(sys.pool   .busy_time),       &sys.pool   .tasks},
    };
    // v7.1: aggregate engine utilization (sum-busy / (N_lanes * total_cycles)).
    // v8.4: 6 lanes now (udma_r/udma_w split).
    uint64_t total_busy = 0;
    for (auto& e : engines) total_busy += e.busy;
    const size_t NE_engines = sizeof(engines)/sizeof(engines[0]);
    double overall_util = total_cycles
        ? 100.0 * double(total_busy) / (double(NE_engines) * double(total_cycles)) : 0.0;
    // Peak utilisation = busiest single engine's % of sim time (= max parallelism floor).
    uint64_t peak_busy = 0;
    const char* peak_eng = engines[0].name;
    for (auto& e : engines) if (e.busy > peak_busy) { peak_busy = e.busy; peak_eng = e.name; }
    double peak_pct = total_cycles ? 100.0 * double(peak_busy) / double(total_cycles) : 0.0;

    std::cout << "  per-engine busy:\n";
    for (auto& e : engines) {
        double pct = total_cycles ? double(e.busy) * 100.0 / total_cycles : 0.0;
        std::cout << "    " << std::setw(7) << e.name << ": "
                  << std::setw(8) << e.busy << " cyc  ("
                  << std::fixed << std::setprecision(1) << std::setw(5)
                  << pct << " %)\n";
    }
    std::cout << "  utilization: avg=" << std::fixed << std::setprecision(1)
              << overall_util << "%   peak="
              << peak_pct << "% (" << peak_eng << ")\n";
    std::cout.unsetf(std::ios::floatfield);
    const auto& l1_stats = sys.l1mesh.stats();
    if (l1_timing_mode == L1TimingMode::MeshConflict ||
        l1_timing_mode == L1TimingMode::MeshOptimistic) {
        auto pns = [](double v) -> uint64_t { return uint64_t(v + 0.5); };
        std::cout << "  L1Mesh NoC: accesses=" << l1_stats.accesses
                  << " bytes=" << l1_stats.bytes
                  << " stripes=" << l1_stats.stripes
                  << " imposed=" << pns(l1_stats.imposed_wait_ns) << " cyc\n"
                  << "    wait edge/router/link/local/sram="
                  << pns(l1_stats.edge_wait_ns) << "/"
                  << pns(l1_stats.router_wait_ns) << "/"
                  << pns(l1_stats.link_wait_ns) << "/"
                  << pns(l1_stats.local_wait_ns) << "/"
                  << pns(l1_stats.sram_wait_ns) << " cyc\n"
                  << "    service edge/router/link/local/sram="
                  << pns(l1_stats.edge_service_ns) << "/"
                  << pns(l1_stats.router_service_ns) << "/"
                  << pns(l1_stats.link_service_ns) << "/"
                  << pns(l1_stats.local_service_ns) << "/"
                  << pns(l1_stats.sram_service_ns) << " cyc\n";
        std::cout << "  L1Mesh lane avg latency:\n";
        for (size_t li = 0; li < l1_stats.read_lane.size(); ++li) {
            const auto& r = l1_stats.read_lane[li];
            const auto& w = l1_stats.write_lane[li];
            const double r_avg = r.accesses ? r.latency_ns / double(r.accesses) : 0.0;
            const double w_avg = w.accesses ? w.latency_ns / double(w.accesses) : 0.0;
            const double r_wait = r.accesses ? r.wait_ns / double(r.accesses) : 0.0;
            const double w_wait = w.accesses ? w.wait_ns / double(w.accesses) : 0.0;
            const double r_srv = r.accesses ? r.service_ns / double(r.accesses) : 0.0;
            const double w_srv = w.accesses ? w.service_ns / double(w.accesses) : 0.0;
            std::cout << "    lane " << std::setw(2) << li
                      << ": R avg/max=" << pns(r_avg) << "/" << pns(r.max_latency_ns)
                      << " wait/svc=" << pns(r_wait) << "/" << pns(r_srv)
                      << " cyc n=" << r.accesses
                      << " bytes=" << r.bytes
                      << " | W avg/max=" << pns(w_avg) << "/" << pns(w.max_latency_ns)
                      << " wait/svc=" << pns(w_wait) << "/" << pns(w_srv)
                      << " cyc n=" << w.accesses
                      << " bytes=" << w.bytes << "\n";
        }
    }

    // Profile path = sibling of program.bin
    std::string prog_path = argv[1];
    std::string prof_path = prog_path;
    auto dot = prof_path.find_last_of('.');
    if (dot != std::string::npos) prof_path.replace(dot, std::string::npos, ".profile.json");
    else prof_path += ".profile.json";

    std::ofstream pf(prof_path);
    if (pf) {
        pf << "{\n";
        pf << "  \"model\": \"" << prog_path << "\",\n";
        pf << "  \"clock_hz\": 1000000000,\n";
        pf << "  \"summary\": {\n";
        pf << "    \"layers\": " << N << ",\n";
        pf << "    \"pass\": " << pass << ", \"fail\": " << fail << ",\n";
        pf << "    \"total_cycles\": " << total_cycles << ",\n";
        pf << "    \"dram_read_bytes\": "  << total_dram_r << ",\n";
        pf << "    \"dram_write_bytes\": " << total_dram_w << ",\n";
        pf << "    \"sram_read_bytes\": "  << total_sram_r << ",\n";
        pf << "    \"sram_write_bytes\": " << total_sram_w << ",\n";
        pf << "    \"util_avg_pct\":  " << std::fixed << std::setprecision(2) << overall_util << ",\n";
        pf << "    \"util_peak_pct\": " << peak_pct << ",\n";
        pf << "    \"util_peak_engine\": \"" << peak_eng << "\"";
        if (l1_timing_mode == L1TimingMode::MeshConflict ||
            l1_timing_mode == L1TimingMode::MeshOptimistic) {
            pf << ",\n";
            pf << "    \"l1mesh\": {"
               << "\"accesses\": " << l1_stats.accesses
               << ", \"bytes\": " << l1_stats.bytes
               << ", \"stripes\": " << l1_stats.stripes
               << ", \"chunks\": " << l1_stats.chunks
               << ", \"imposed_wait_cycles\": " << uint64_t(l1_stats.imposed_wait_ns + 0.5)
               << ", \"edge_wait_cycles\": " << uint64_t(l1_stats.edge_wait_ns + 0.5)
               << ", \"router_wait_cycles\": " << uint64_t(l1_stats.router_wait_ns + 0.5)
               << ", \"link_wait_cycles\": " << uint64_t(l1_stats.link_wait_ns + 0.5)
               << ", \"local_wait_cycles\": " << uint64_t(l1_stats.local_wait_ns + 0.5)
               << ", \"sram_wait_cycles\": " << uint64_t(l1_stats.sram_wait_ns + 0.5)
               << ", \"edge_service_cycles\": " << uint64_t(l1_stats.edge_service_ns + 0.5)
               << ", \"router_service_cycles\": " << uint64_t(l1_stats.router_service_ns + 0.5)
               << ", \"link_service_cycles\": " << uint64_t(l1_stats.link_service_ns + 0.5)
               << ", \"local_service_cycles\": " << uint64_t(l1_stats.local_service_ns + 0.5)
               << ", \"sram_service_cycles\": " << uint64_t(l1_stats.sram_service_ns + 0.5)
               << ", \"read_lanes\": [";
            for (size_t li = 0; li < l1_stats.read_lane.size(); ++li) {
                const auto& lane = l1_stats.read_lane[li];
                const double avg = lane.accesses ? lane.latency_ns / double(lane.accesses) : 0.0;
                const double wait = lane.accesses ? lane.wait_ns / double(lane.accesses) : 0.0;
                const double service = lane.accesses ? lane.service_ns / double(lane.accesses) : 0.0;
                pf << "{\"id\": " << li
                   << ", \"accesses\": " << lane.accesses
                   << ", \"bytes\": " << lane.bytes
                   << ", \"avg_latency_cycles\": " << uint64_t(avg + 0.5)
                   << ", \"avg_wait_cycles\": " << uint64_t(wait + 0.5)
                   << ", \"avg_service_cycles\": " << uint64_t(service + 0.5)
                   << ", \"max_latency_cycles\": " << uint64_t(lane.max_latency_ns + 0.5)
                   << ", \"max_wait_cycles\": " << uint64_t(lane.max_wait_ns + 0.5)
                   << ", \"max_service_cycles\": " << uint64_t(lane.max_service_ns + 0.5)
                   << "}";
                if (li + 1 < l1_stats.read_lane.size()) pf << ", ";
            }
            pf << "], \"write_lanes\": [";
            for (size_t li = 0; li < l1_stats.write_lane.size(); ++li) {
                const auto& lane = l1_stats.write_lane[li];
                const double avg = lane.accesses ? lane.latency_ns / double(lane.accesses) : 0.0;
                const double wait = lane.accesses ? lane.wait_ns / double(lane.accesses) : 0.0;
                const double service = lane.accesses ? lane.service_ns / double(lane.accesses) : 0.0;
                pf << "{\"id\": " << li
                   << ", \"accesses\": " << lane.accesses
                   << ", \"bytes\": " << lane.bytes
                   << ", \"avg_latency_cycles\": " << uint64_t(avg + 0.5)
                   << ", \"avg_wait_cycles\": " << uint64_t(wait + 0.5)
                   << ", \"avg_service_cycles\": " << uint64_t(service + 0.5)
                   << ", \"max_latency_cycles\": " << uint64_t(lane.max_latency_ns + 0.5)
                   << ", \"max_wait_cycles\": " << uint64_t(lane.max_wait_ns + 0.5)
                   << ", \"max_service_cycles\": " << uint64_t(lane.max_service_ns + 0.5)
                   << "}";
                if (li + 1 < l1_stats.write_lane.size()) pf << ", ";
            }
            pf << "]}\n";
        } else {
            pf << "\n";
        }
        pf.unsetf(std::ios::floatfield);
        pf << "  },\n";
        pf << "  \"engines\": {\n";
        const size_t NE = sizeof(engines)/sizeof(engines[0]);
        for (size_t i = 0; i < NE; ++i) {
            pf << "    \"" << engines[i].name << "\": {"
               << "\"busy_cycles\": " << engines[i].busy
               << ", \"tasks\": [";
            const auto& tk = *engines[i].tasks;
            for (size_t j = 0; j < tk.size(); ++j) {
                pf << "[" << tk[j].first << "," << tk[j].second << "]";
                if (j+1 < tk.size()) pf << ",";
            }
            pf << "]}" << (i+1 < NE ? "," : "") << "\n";
        }
        pf << "  },\n";
        pf << "  \"layers\": [\n";
        for (size_t i = 0; i < profile.size(); ++i) {
            const auto& L = profile[i];
            pf << "    {"
               << "\"id\": " << L.id
               << ", \"flow\": " << L.flow
               << ", \"op\": \"" << L.op << "\""
               << ", \"in\": [" << L.in_h << "," << L.in_w << "," << L.in_c << "]"
               << ", \"out\": [" << L.out_h << "," << L.out_w << "," << L.out_c << "]"
               << ", \"k\": [" << int(L.k_h) << "," << int(L.k_w) << "]"
               << ", \"s\": [" << int(L.s_h) << "," << int(L.s_w) << "]"
               << ", \"group\": " << L.group
               << ", \"tiles\": [" << L.tiles_h << "," << L.tiles_oc << "]"
               << ", \"pass\": " << (L.pass ? "true" : "false")
               << ", \"cycles_layer\": " << L.cycles_layer
               << ", \"cycles_cum\": "   << L.cycles_cum
               << ", \"conv_util_pct\": " << std::fixed << std::setprecision(2) << L.util_pct
               << ", \"dram_r\": " << L.dram_r
               << ", \"dram_w\": " << L.dram_w
               << ", \"sram_r\": " << L.sram_r
               << ", \"sram_w\": " << L.sram_w
               << ", \"streamed\": " << (L.streamed ? "true" : "false")
               << "}" << (i+1 < profile.size() ? "," : "") << "\n";
            pf.unsetf(std::ios::floatfield);
        }
        pf << "  ]\n";
        pf << "}\n";
        std::cout << "  profile: " << prof_path << "\n";
    }

    // ------ v5: CSV companion (one row per layer) ------
    std::string csv_path = prog_path;
    {
        auto dot = csv_path.find_last_of('.');
        if (dot != std::string::npos) csv_path.replace(dot, std::string::npos, ".profile.csv");
        else csv_path += ".profile.csv";
    }
    std::ofstream cf(csv_path);
    if (cf) {
        cf << "id,flow,op,in_h,in_w,in_c,out_h,out_w,out_c,k_h,k_w,s_h,s_w,group,"
              "tiles_h,tiles_oc,pass,cycles_layer,cycles_cum,conv_util_pct,"
              "dram_r,dram_w,sram_r,sram_w\n";
        for (const auto& L : profile) {
            // op_name() pads with leading spaces for table alignment; strip for CSV.
            std::string op_clean = L.op;
            size_t a = op_clean.find_first_not_of(' ');
            if (a != std::string::npos) op_clean = op_clean.substr(a);
            cf << L.id << "," << L.flow << "," << op_clean << ","
               << L.in_h << "," << L.in_w << "," << L.in_c << ","
               << L.out_h << "," << L.out_w << "," << L.out_c << ","
               << int(L.k_h) << "," << int(L.k_w) << ","
               << int(L.s_h) << "," << int(L.s_w) << "," << L.group << ","
               << L.tiles_h << "," << L.tiles_oc << ","
               << (L.pass ? 1 : 0) << ","
               << L.cycles_layer << "," << L.cycles_cum << ","
               << std::fixed << std::setprecision(2) << L.util_pct << ","
               << L.dram_r << "," << L.dram_w << ","
               << L.sram_r << "," << L.sram_w << "\n";
            cf.unsetf(std::ios::floatfield);
        }
        std::cout << "  csv:     " << csv_path << "\n";
    }

    return fail == 0 ? 0 : 1;
}
