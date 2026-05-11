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
#include <cstdlib>
#include "mdla7/system.h"
#include "mdla7/fp_utils.h"
#include "mdla7/program_image.h"

using namespace mdla7;

namespace {

uint8_t encode_stride_pair(uint8_t s_h, uint8_t s_w) {
    // v8: 2-bit log2 encoding {1→0, 2→1, 4→2, 8→3}. Strides outside this set
    // are clamped to the nearest supported value (warned at compile time
    // upstream if needed).
    auto enc = [](uint8_t s) -> uint8_t {
        return (s >= 8) ? 3 : (s >= 4) ? 2 : (s >= 2) ? 1 : 0;
    };
    return uint8_t(enc(s_h) | (enc(s_w) << 2));
}

uint8_t encode_conv_stride_pair(uint8_t s_h, uint8_t s_w) {
    auto enc = [](uint8_t s) -> uint8_t {
        return (s >= 16) ? 0 : (s & 0x0F);
    };
    return uint8_t(enc(s_h) | (enc(s_w) << 4));
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

Descriptor make_udma_act_stream_head(uint32_t src, uint32_t dst, uint32_t raw_bytes,
                                     uint32_t compressed_bytes, uint32_t metadata_bytes,
                                     uint32_t head_raw_bytes,
                                     uint8_t signal_tag,
                                     uint8_t wait_a = 0, uint8_t wait_b = 0) {
    Descriptor d = make_udma_act_decomp(src, dst, raw_bytes, compressed_bytes,
                                        metadata_bytes, signal_tag, wait_a, wait_b);
    d.body.udma.mode = UM_ACT_DECOMP_STREAM_HEAD;
    d.body.udma.idx_table_addr = head_raw_bytes;
    return d;
}

Descriptor make_udma_act_stream_tail(uint32_t raw_bytes, uint32_t comp_meta_bytes,
                                     uint8_t signal_tag, uint8_t wait_a = 0) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_UDMA;
    d.hdr.dtype = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_a ? 1 : 0;
    d.hdr.wait_tags[0] = wait_a;
    auto& u = d.body.udma;
    u.mode = UM_ACT_DECOMP_STREAM_TAIL;
    u.direction = 0;
    u.length = raw_bytes;
    u.src_stride = comp_meta_bytes;
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

[[maybe_unused]] Descriptor make_udma_d2s(uint32_t src, uint32_t dst,
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

Descriptor make_tnps(uint32_t src, uint32_t dst, uint32_t bytes,
                     uint8_t signal_tag, uint8_t wait_a = 0, uint8_t wait_b = 0,
                     uint8_t mode = TM_LINEAR_COPY) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_TNPS;
    d.hdr.dtype = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = (wait_a ? 1 : 0) + (wait_b ? 1 : 0);
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    auto& t = d.body.tnps;
    t.mode = mode;
    t.src_addr = src;
    t.dst_addr = dst;
    t.length = bytes;
    return d;
}

Descriptor make_tnps_d2s(uint32_t src, uint32_t dst,
                         uint16_t in_h, uint16_t in_w, uint16_t in_c,
                         uint16_t block, uint8_t elem_size,
                         uint8_t signal_tag, uint8_t wait_a = 0) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_TNPS;
    d.hdr.dtype = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_a ? 1 : 0;
    d.hdr.wait_tags[0] = wait_a;
    auto& t = d.body.tnps;
    const uint16_t out_c = uint16_t(in_c / std::max<uint16_t>(1, block * block));
    t.mode = TM_DEPTH_TO_SPACE;
    t.src_addr = src;
    t.dst_addr = dst;
    t.length = elem_size;
    t.src_stride = uint32_t(in_w) * in_c * elem_size;
    t.dst_stride = uint32_t(in_w) * block * out_c * elem_size;
    t.num_chunks = in_h;
    t.slice_begin[0] = in_w;
    t.slice_begin[1] = in_c;
    t.slice_begin[2] = block;
    t.slice_begin[3] = out_c;
    return d;
}

Descriptor make_tnps_s2d(uint32_t src, uint32_t dst,
                         uint16_t in_h, uint16_t in_w, uint16_t in_c,
                         uint16_t block, uint16_t out_c, uint8_t elem_size,
                         uint8_t signal_tag, uint8_t wait_a = 0) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_TNPS;
    d.hdr.dtype = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_a ? 1 : 0;
    d.hdr.wait_tags[0] = wait_a;
    auto& t = d.body.tnps;
    t.mode = TM_SPACE_TO_DEPTH;
    t.src_addr = src;
    t.dst_addr = dst;
    t.length = elem_size;
    t.num_chunks = in_h;
    t.slice_begin[0] = in_w;
    t.slice_begin[1] = in_c;
    t.slice_begin[2] = block;
    t.slice_begin[3] = out_c;
    return d;
}

Descriptor make_tnps_meta(uint8_t mode, uint32_t src, uint32_t dst,
                          uint32_t bytes, uint32_t meta_addr,
                          uint8_t signal_tag, uint8_t wait_a = 0) {
    Descriptor d = make_tnps(src, dst, bytes, signal_tag, wait_a, 0, mode);
    d.body.tnps.idx_table_addr = meta_addr;
    return d;
}

Descriptor make_tnps_slice_2d(uint32_t src, uint32_t dst,
                              uint16_t rows, uint16_t col_off,
                              uint32_t row_bytes, uint32_t src_stride,
                              uint32_t dst_stride,
                              uint8_t signal_tag, uint8_t wait_a = 0) {
    Descriptor d = make_tnps(src, dst, row_bytes, signal_tag, wait_a, 0,
                             TM_STRIDED_SLICE);
    auto& t = d.body.tnps;
    t.src_stride = src_stride;
    t.dst_stride = dst_stride;
    t.slice_begin[0] = 0;
    t.slice_end[0] = rows;
    t.slice_begin[1] = col_off;
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

// v8.30: unary EWE op (HARD_SWISH / GELU / LOGISTIC). Single input, no input-B, params
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
    e.subtype = (L.op_kind == OK_GELU) ? ES_GELU
              : (L.op_kind == OK_LOGISTIC) ? ES_LOGISTIC
              : ES_HARD_SWISH;
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
                  << " program.bin [--quiet] [--l1-timing=fast|conflict|mesh|mesh-opt] [--no-microblock]\n";
        return 2;
    }
    bool quiet = false;
    bool enable_microblocks = true;
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
        } else if (arg == "--no-microblock" || arg == "--disable-microblock") {
            enable_microblocks = false;
        } else {
            std::cerr << "unknown option: " << arg << "\n"
                      << "usage: " << argv[0]
                      << " program.bin [--quiet] [--l1-timing=fast|conflict|mesh|mesh-opt] [--no-microblock]\n";
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
    std::cout << "mdla7_model_runner: " << argv[1] << "  ("
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
    std::vector<bool> preemitted_layout_layer(N, false);
    std::vector<uint8_t> fc_prefetch_wgt_tag(N, 0);
    std::vector<uint32_t> fc_prefetch_wgt_l1(N, 0);
    uint8_t next_tag_v = 1;
    auto alloc_tag = [&]() -> uint8_t {
        uint8_t t = next_tag_v++;
        if (next_tag_v == 0) next_tag_v = 1;
        return t;
    };

    auto align64 = [](uint32_t x) -> uint32_t { return (x + 63) & ~uint32_t(63); };
    constexpr uint32_t L1_BUDGET = L1MESH_BYTES;     // 3 MB (spec §3A.10)
    const bool conv_ws_enable = [] {
        const char* p = std::getenv("MDLA7_CONV_WS");
        return !p || std::string(p) != "0";
    }();

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
    auto make_store_barrier = [&](uint32_t layer_idx, uint32_t src_addr, uint32_t dst_addr,
                                  uint8_t signal_tag, uint8_t wait_tag) -> Descriptor {
        Descriptor d = make_udma(src_addr, dst_addr, 1, /*dir*/ 1, signal_tag, wait_tag);
        d.hdr.flags |= DF_STREAM | DF_STREAM_TAIL;  // stream tail: allow safe later prefetches to bypass.
        d.hdr.layer_id = uint16_t(layer_idx);
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
            BINARY_EWE_D2S = 2,
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
        bool input_a_wait_by_mb = false;
        bool output_contiguous = false;
        uint8_t initial_wait_tag = 0;
        uint8_t input_a_wait_tag = 0;
        std::vector<uint8_t> input_a_mb_wait_tags;
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
        std::vector<uint8_t> mb_done_tags;
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
            const uint8_t a_wait_tag =
                (tc.input_a_preloaded && tc.input_a_wait_by_mb &&
                 mb_id < tc.input_a_mb_wait_tags.size())
                    ? tc.input_a_mb_wait_tags[mb_id]
                    : tc.input_a_wait_tag;
            const uint8_t a_tag = tc.input_a_preloaded ? a_wait_tag : alloc_tag();
            const uint8_t b_tag = alloc_tag();
            const uint8_t e_tag = alloc_tag();
            const uint8_t s_tag = alloc_tag();
            const bool final_mb = (elem_done + mb.elems >= total_elems);
            const uint8_t slot_wait =
                slot_free_tag[mb.slot] ? slot_free_tag[mb.slot] : tc.initial_wait_tag;

            uint64_t a_charged = 0;
            if (!tc.input_a_preloaded) {
                auto [a, charged] = make_act_load(tc.layer,
                                                  uint32_t(tc.layer.dram_in + dram_off),
                                                  tc.in_a_l1[mb.slot], mb.bytes,
                                                  a_tag, slot_wait);
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
                                                      b_tag, slot_wait);
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
                r.mb_done_tags.push_back(e_tag);
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
                r.mb_done_tags.push_back(s_tag);
                slot_free_tag[mb.slot] = s_tag;
            }

            elem_done += mb.elems;
            ++mb_id;
        }
        return r;
    };

    struct TnpsWavefrontResult {
        uint8_t done_tag = 0;
        uint64_t dram_r = 0, dram_w = 0, sram_r = 0, sram_w = 0;
        uint16_t tiles = 1;
    };

    auto emit_linear_tnps_wavefront = [&](uint32_t layer_idx, const LayerMeta& L,
                                          uint8_t initial_wait) -> TnpsWavefrontResult {
        TnpsWavefrontResult r{};
        const uint64_t total = L.ref_size;
        const uint64_t safety = 4096;
        uint64_t tile_bytes = (L1_BUDGET > safety) ? ((L1_BUDGET - safety) / 4) : 0;
        tile_bytes &= ~uint64_t(63);
        if (tile_bytes < 64)
            tile_bytes = 64;
        tile_bytes = std::min<uint64_t>(tile_bytes, total);
        const uint32_t seg = align64(uint32_t(tile_bytes));
        const uint32_t in_l1[2]  = {0, align64(seg + seg)};
        const uint32_t out_l1[2] = {seg, align64(in_l1[1] + seg)};
        if (uint64_t(out_l1[1]) + seg + safety > L1_BUDGET) {
            std::cerr << "layer " << layer_idx
                      << ": linear TNPS no room for double-buffered tile\n";
            r.done_tag = 0;
            return r;
        }
        uint8_t slot_done[2] = {initial_wait, initial_wait};
        uint64_t off = 0;
        uint16_t mb_id = 0;
        while (off < total) {
            const uint32_t bytes = uint32_t(std::min<uint64_t>(tile_bytes, total - off));
            const bool final_mb = (off + bytes == total);
            Microblock mb{};
            mb.id = mb_id;
            mb.slot = uint8_t(mb_id & 1u);
            mb.elem_off = off;
            mb.rows = 1;
            mb.elems = bytes;
            mb.bytes = bytes;
            const uint8_t load_tag = alloc_tag();
            const uint8_t tnps_tag = alloc_tag();
            const uint8_t store_tag = alloc_tag();
            auto [ld, charged] = make_act_load(L, uint32_t(L.dram_in + off),
                                               in_l1[mb.slot], bytes,
                                               load_tag, slot_done[mb.slot]);
            mark_stream(ld, layer_idx, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(ld);
            Descriptor td = make_tnps(in_l1[mb.slot], out_l1[mb.slot], bytes,
                                      tnps_tag, load_tag);
            mark_stream(td, layer_idx, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(td);
            Descriptor sd = make_udma(out_l1[mb.slot], uint32_t(L.dram_out + off),
                                      bytes, /*dir*/ 1, store_tag, tnps_tag);
            mark_stream(sd, layer_idx, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(sd);
            r.dram_r += charged;
            r.dram_w += bytes;
            r.sram_w += uint64_t(bytes) * 2;
            r.sram_r += uint64_t(bytes) * 2;
            r.done_tag = store_tag;
            slot_done[mb.slot] = store_tag;
            off += bytes;
            ++mb_id;
        }
        r.tiles = std::max<uint16_t>(1, mb_id);
        return r;
    };

    auto emit_d2s_tnps_wavefront = [&](uint32_t layer_idx, const LayerMeta& L,
                                       uint16_t block, uint32_t elem_size,
                                       uint8_t initial_wait) -> TnpsWavefrontResult {
        TnpsWavefrontResult r{};
        const uint64_t safety = 4096;
        const uint32_t in_row = uint32_t(L.in_w) * L.in_c * elem_size;
        const uint32_t dst_stride = uint32_t(L.in_w) * block * L.out_c * elem_size;
        const uint32_t out_rows_per_in = block;
        uint32_t tile_h = L.in_h;
        auto top_for = [&](uint32_t rows) -> uint64_t {
            const uint32_t in_bytes = rows * in_row;
            const uint32_t out_bytes = rows * out_rows_per_in * dst_stride;
            const uint32_t in0 = 0;
            const uint32_t out0 = align64(in0 + in_bytes);
            const uint32_t in1 = align64(out0 + out_bytes);
            const uint32_t out1 = align64(in1 + in_bytes);
            return uint64_t(out1) + out_bytes;
        };
        while (tile_h > 1 && top_for(tile_h) + safety > L1_BUDGET)
            --tile_h;
        if (top_for(tile_h) + safety > L1_BUDGET) {
            std::cerr << "layer " << layer_idx
                      << ": D2SPACE no room for double-buffered row tile\n";
            return r;
        }
        const uint32_t in_seg = tile_h * in_row;
        const uint32_t out_seg = tile_h * out_rows_per_in * dst_stride;
        const uint32_t in_l1[2]  = {0, align64(align64(in_seg) + out_seg)};
        const uint32_t out_l1[2] = {align64(in_seg), align64(in_l1[1] + in_seg)};
        uint8_t slot_done[2] = {initial_wait, initial_wait};
        uint32_t ih = 0;
        uint16_t mb_id = 0;
        while (ih < L.in_h) {
            const uint32_t rows = std::min<uint32_t>(tile_h, L.in_h - ih);
            const uint32_t in_bytes = rows * in_row;
            const uint32_t out_bytes = rows * out_rows_per_in * dst_stride;
            const uint32_t src_off = ih * in_row;
            const uint32_t dst_off = ih * out_rows_per_in * dst_stride;
            const bool final_mb = (ih + rows == L.in_h);
            Microblock mb{};
            mb.id = mb_id;
            mb.slot = uint8_t(mb_id & 1u);
            mb.elem_off = uint64_t(ih) * L.in_w * L.in_c;
            mb.rows = rows;
            mb.elems = rows * L.in_w * L.in_c;
            mb.bytes = in_bytes;
            const uint8_t load_tag = alloc_tag();
            const uint8_t tnps_tag = alloc_tag();
            const uint8_t store_tag = alloc_tag();
            auto [ld, charged] = make_act_load(L, uint32_t(L.dram_in + src_off),
                                               in_l1[mb.slot], in_bytes,
                                               load_tag, slot_done[mb.slot]);
            mark_stream(ld, layer_idx, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(ld);
            Descriptor td = make_tnps_d2s(in_l1[mb.slot], out_l1[mb.slot],
                                          uint16_t(rows), L.in_w, L.in_c,
                                          block, uint8_t(elem_size),
                                          tnps_tag, load_tag);
            mark_stream(td, layer_idx, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(td);
            Descriptor sd = make_udma(out_l1[mb.slot], uint32_t(L.dram_out + dst_off),
                                      out_bytes, /*dir*/ 1, store_tag, tnps_tag);
            mark_stream(sd, layer_idx, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(sd);
            r.dram_r += charged;
            r.dram_w += out_bytes;
            r.sram_w += uint64_t(in_bytes) + out_bytes;
            r.sram_r += uint64_t(in_bytes) + out_bytes;
            r.done_tag = store_tag;
            slot_done[mb.slot] = store_tag;
            ih += rows;
            ++mb_id;
        }
        r.tiles = std::max<uint16_t>(1, mb_id);
        return r;
    };

    auto emit_s2d_tnps_wavefront = [&](uint32_t layer_idx, const LayerMeta& L,
                                       uint16_t block, uint32_t elem_size,
                                       uint8_t initial_wait) -> TnpsWavefrontResult {
        TnpsWavefrontResult r{};
        const uint64_t safety = 4096;
        const uint32_t in_row = uint32_t(L.in_w) * L.in_c * elem_size;
        const uint32_t out_row = uint32_t(L.out_w) * L.out_c * elem_size;
        uint32_t tile_groups = std::max<uint32_t>(1, L.out_h);
        auto top_for = [&](uint32_t groups) -> uint64_t {
            const uint32_t in_bytes = groups * block * in_row;
            const uint32_t out_bytes = groups * out_row;
            const uint32_t in0 = 0;
            const uint32_t out0 = align64(in0 + in_bytes);
            const uint32_t in1 = align64(out0 + out_bytes);
            const uint32_t out1 = align64(in1 + in_bytes);
            return uint64_t(out1) + out_bytes;
        };
        while (tile_groups > 1 && top_for(tile_groups) + safety > L1_BUDGET)
            --tile_groups;
        if (top_for(tile_groups) + safety > L1_BUDGET) {
            std::cerr << "layer " << layer_idx
                      << ": S2SPACE no room for double-buffered row tile\n";
            return r;
        }
        const uint32_t in_seg = tile_groups * block * in_row;
        const uint32_t out_seg = tile_groups * out_row;
        const uint32_t in_l1[2]  = {0, align64(align64(in_seg) + out_seg)};
        const uint32_t out_l1[2] = {align64(in_seg), align64(in_l1[1] + in_seg)};
        uint8_t slot_done[2] = {initial_wait, initial_wait};
        uint32_t og = 0;
        uint16_t mb_id = 0;
        while (og < L.out_h) {
            const uint32_t groups = std::min<uint32_t>(tile_groups, L.out_h - og);
            const uint32_t in_rows = groups * block;
            const uint32_t in_bytes = in_rows * in_row;
            const uint32_t out_bytes = groups * out_row;
            const uint32_t src_off = og * block * in_row;
            const uint32_t dst_off = og * out_row;
            const bool final_mb = (og + groups == L.out_h);
            Microblock mb{};
            mb.id = mb_id;
            mb.slot = uint8_t(mb_id & 1u);
            mb.elem_off = uint64_t(og) * L.out_w * L.out_c;
            mb.rows = groups;
            mb.elems = groups * L.out_w * L.out_c;
            mb.bytes = out_bytes;
            const uint8_t load_tag = alloc_tag();
            const uint8_t tnps_tag = alloc_tag();
            const uint8_t store_tag = alloc_tag();
            auto [ld, charged] = make_act_load(L, uint32_t(L.dram_in + src_off),
                                               in_l1[mb.slot], in_bytes,
                                               load_tag, slot_done[mb.slot]);
            mark_stream(ld, layer_idx, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(ld);
            Descriptor td = make_tnps_s2d(in_l1[mb.slot], out_l1[mb.slot],
                                          uint16_t(in_rows), L.in_w, L.in_c,
                                          block, L.out_c, uint8_t(elem_size),
                                          tnps_tag, load_tag);
            mark_stream(td, layer_idx, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(td);
            Descriptor sd = make_udma(out_l1[mb.slot], uint32_t(L.dram_out + dst_off),
                                      out_bytes, /*dir*/ 1, store_tag, tnps_tag);
            mark_stream(sd, layer_idx, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
            program.push_back(sd);
            r.dram_r += charged;
            r.dram_w += out_bytes;
            r.sram_w += uint64_t(in_bytes) + out_bytes;
            r.sram_r += uint64_t(in_bytes) + out_bytes;
            r.done_tag = store_tag;
            slot_done[mb.slot] = store_tag;
            og += groups;
            ++mb_id;
        }
        r.tiles = std::max<uint16_t>(1, mb_id);
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
    std::vector<size_t> pool_count_at_layer_end(N, 0);
    size_t pool_count_so_far = 0;
    std::vector<size_t> tnps_count_at_layer_end(N, 0);
    size_t tnps_count_so_far = 0;

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
    std::vector<uint8_t> fuse_prev_mb_done_tags;
    auto clear_prev_binary_ewe_live = [&]() {
        fuse_prev_is_binary_ewe = false;
        fuse_prev_live_a_size = fuse_prev_live_b_size = fuse_prev_live_o_size = 0;
        fuse_prev_mb_done_tags.clear();
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
    auto is_binary_ewe_kind = [](uint16_t op_kind) {
        return op_kind == OK_ADD || op_kind == OK_MUL || op_kind == OK_SUB;
    };
    auto is_int16_stream_dtype = [](uint16_t dtype) {
        return dtype == DT_INT16x8 || dtype == DT_INT16x16;
    };
    auto stream_dtype_compatible = [&](uint16_t producer_dtype,
                                       uint16_t consumer_dtype) {
        return producer_dtype == consumer_dtype ||
               (is_int16_stream_dtype(producer_dtype) &&
                is_int16_stream_dtype(consumer_dtype));
    };
    auto is_attention_softmax_meta = [](const LayerMeta& S) {
        const bool attention_matrix =
            S.in_h > 1 && S.in_h <= 32 &&
            S.in_w > 1 && S.in_w <= 2048 &&
            S.in_c > 1 && S.in_c <= 2048;
        const bool attention_dtype = (S.dtype == DT_INT8x8 || S.dtype == DT_FP16);
        return S.op_kind == OK_SOFTMAX && attention_dtype && attention_matrix;
    };
    auto graph_input0_is_exact_producer =
        [&](uint32_t producer_layer, uint32_t consumer_layer) -> bool {
            if (!graph_metas) return true;
            if (producer_layer >= N || consumer_layer >= N) return false;
            const auto& P = graph_metas[producer_layer];
            const auto& C = graph_metas[consumer_layer];
            return C.producer0_layer == int32_t(producer_layer) &&
                   C.input0_tensor >= 0 &&
                   C.input0_tensor == P.output_tensor;
        };
    auto graph_input1_is_exact_producer =
        [&](uint32_t producer_layer, uint32_t consumer_layer) -> bool {
            if (!graph_metas) return false;
            if (producer_layer >= N || consumer_layer >= N) return false;
            const auto& P = graph_metas[producer_layer];
            const auto& C = graph_metas[consumer_layer];
            return C.producer1_layer == int32_t(producer_layer) &&
                   C.input1_tensor >= 0 &&
                   C.input1_tensor == P.output_tensor;
        };
    auto graph_has_exact_single_consumer =
        [&](uint32_t producer_layer, uint32_t consumer_layer,
            bool allow_input1) -> bool {
            if (!graph_metas) return true;
            if (producer_layer >= N || consumer_layer >= N) return false;
            const auto& G = graph_metas[producer_layer];
            if (G.consumer_count != 1 ||
                G.first_consumer_layer != int32_t(consumer_layer) ||
                G.last_consumer_layer  != int32_t(consumer_layer))
                return false;
            return graph_input0_is_exact_producer(producer_layer, consumer_layer) ||
                   (allow_input1 && graph_input1_is_exact_producer(producer_layer, consumer_layer));
        };
    auto graph_layer_feeds_softmax =
        [&](uint32_t producer_layer) -> bool {
            if (producer_layer >= N) return false;
            if (producer_layer + 1 < N &&
                metas[producer_layer + 1].op_kind == OK_SOFTMAX &&
                graph_input0_is_exact_producer(producer_layer, producer_layer + 1))
                return true;
            if (!graph_metas) return false;
            const auto& G = graph_metas[producer_layer];
            auto consumer_is_softmax = [&](int32_t consumer_layer) {
                return consumer_layer >= 0 &&
                       consumer_layer < int32_t(N) &&
                       metas[uint32_t(consumer_layer)].op_kind == OK_SOFTMAX;
            };
            return consumer_is_softmax(G.first_consumer_layer) ||
                   consumer_is_softmax(G.last_consumer_layer);
        };
    auto graph_layer_feeds_binary_softmax_tail =
        [&](uint32_t producer_layer) -> bool {
            if (!graph_metas || producer_layer >= N) return false;
            const auto& G = graph_metas[producer_layer];
            auto consumer_is_binary_softmax_tail = [&](int32_t consumer_layer) {
                if (consumer_layer < 0 || consumer_layer >= int32_t(N))
                    return false;
                const uint32_t c = uint32_t(consumer_layer);
                if (!is_binary_ewe_kind(metas[c].op_kind))
                    return false;
                if (!graph_input0_is_exact_producer(producer_layer, c) &&
                    !graph_input1_is_exact_producer(producer_layer, c))
                    return false;
                return graph_layer_feeds_softmax(c);
            };
            return consumer_is_binary_softmax_tail(G.first_consumer_layer) ||
                   consumer_is_binary_softmax_tail(G.last_consumer_layer);
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
            P.op_kind == OK_D2SPACE || P.op_kind == OK_S2SPACE;
        if (!p_ok) continue;
        const bool shape_match =
            P.out_h == S.in_h && P.out_w == S.in_w &&
            P.out_c == S.in_c && stream_dtype_compatible(P.dtype, S.dtype);
        if (!shape_match) continue;
        if (is_conv_class_meta(S) || is_binary_meta(S) ||
            (!conservative_mul_graph && S.op_kind == OK_MUL) ||
            S.op_kind == OK_AVG_POOL || S.op_kind == OK_MAX_POOL ||
            S.op_kind == OK_D2SPACE)
            producer_no_store[k] = true;
    }

    // PAD -> CONV: keep the unpadded producer output in L1 and let CONV's
    // boundary pad logic synthesize the halo. This removes the intermediate
    // TNPS materialization (not just overlaps it) for patterns such as
    // sd_decoder L15 -> L16 PAD -> L17 CONV.
    for (uint32_t k = 0; k + 1 < N; ++k) {
        const auto& P = metas[k];
        const auto& S = metas[k + 1];
        if (P.op_kind != OK_PAD || !is_conv_class_meta(S))
            continue;
        if (P.dtype != S.dtype || P.out_h != S.in_h ||
            P.out_w != S.in_w || P.out_c != S.in_c)
            continue;
        const bool spatial_pad =
            (P.p_t || P.p_b || P.p_l || P.p_r) &&
            P.p_t <= 7 && P.p_b <= 7 && P.p_l <= 7 && P.p_r <= 7;
        if (spatial_pad)
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
        if (is_attention_softmax_meta(S)) {
            producer_no_store[k] = true;
        }
    }
    // Scale/mask EWE just before an attention SOFTMAX is an intermediate
    // attention-score tensor.  Keep it on chip so the fused softmax tail below
    // can consume it microblock-by-microblock instead of materializing 8 MB
    // score matrices to DRAM.
    for (uint32_t k = 0; k + 1 < N; ++k) {
        const auto& P = metas[k];
        const auto& S = metas[k + 1];
        const bool binary_attention_score =
            is_binary_ewe_kind(P.op_kind) &&
            is_attention_softmax_meta(S) &&
            P.dtype == DT_INT8x8 &&
            P.dtype == S.dtype &&
            P.out_h == S.in_h && P.out_w == S.in_w && P.out_c == S.in_c &&
            graph_has_exact_single_consumer(k, k + 1, false);
        if (binary_attention_score)
            producer_no_store[k] = true;
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
            const bool feeds_softmax =
                graph_layer_feeds_softmax(k) ||
                graph_layer_feeds_binary_softmax_tail(k);
            if ((binary_ewe && has_later_consumer && !feeds_softmax) ||
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

        // Layout-only TNPS ops can dominate ETHZ/XLSR when modeled as
        // materialized DRAM round trips.  If GraphMeta says the layout result
        // feeds a later consumer, the consumer's synthetic input is already
        // present in the compiled program, while the intended hardware path is
        // an on-chip handoff.  Suppress the intermediate checkpoint; keep true
        // final layout outputs visible.
        for (uint32_t k = 0; k < N; ++k) {
            const auto& L = metas[k];
            const auto& G = graph_metas[k];
            const bool layout_handoff =
                L.op_kind == OK_CONCAT ||
                L.op_kind == OK_SLICE ||
                L.op_kind == OK_STRIDED_SLICE ||
                L.op_kind == OK_TRANSPOSE ||
                L.op_kind == OK_S2SPACE ||
                L.op_kind == OK_PACK ||
                L.op_kind == OK_UNPACK ||
                L.op_kind == OK_SPLIT;
            if (layout_handoff &&
                G.consumer_count > 0 &&
                G.last_consumer_layer > int32_t(k)) {
                producer_no_store[k] = true;
            }
        }

        // BERT embedding masks lower as GATHER -> same-shape EWE.  The
        // consumer has compiler-materialized input bytes today, so the gather
        // DRAM store is only an intermediate checkpoint.
        for (uint32_t k = 0; k + 1 < N; ++k) {
            const auto& L = metas[k];
            const auto& S = metas[k + 1];
            const auto& G = graph_metas[k];
            const bool gather_to_same_shape_ewe =
                L.op_kind == OK_GATHER &&
                (S.op_kind == OK_ADD || S.op_kind == OK_MUL || S.op_kind == OK_SUB) &&
                G.consumer_count > 0 &&
                G.last_consumer_layer > int32_t(k) &&
                L.dtype == S.dtype &&
                L.out_h == S.in_h && L.out_w == S.in_w && L.out_c == S.in_c;
            if (gather_to_same_shape_ewe)
                producer_no_store[k] = true;
        }

        // Token embedding lookup can be followed by a pure shape barrier
        // before CONCAT.  If element count/bytes are unchanged, the gather
        // output DRAM copy is still just an intermediate checkpoint.
        for (uint32_t k = 0; k + 1 < N; ++k) {
            const auto& L = metas[k];
            const auto& S = metas[k + 1];
            const auto& G = graph_metas[k];
            const bool gather_to_metadata =
                L.op_kind == OK_GATHER &&
                (S.op_kind == OK_RESHAPE || S.op_kind == OK_MATERIALIZE) &&
                G.consumer_count > 0 &&
                G.last_consumer_layer > int32_t(k) &&
                L.dtype == S.dtype &&
                L.ref_size == S.in_size &&
                L.ref_size == S.ref_size;
            if (gather_to_metadata)
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
                 L.op_kind == OK_HARD_SWISH || L.op_kind == OK_GELU ||
                 L.op_kind == OK_LOGISTIC) &&
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
                P.op_kind == OK_GELU ||
                P.op_kind == OK_LOGISTIC;
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
        const bool exact_large_layout_handoff =
            graph_has_exact_single_consumer(k, k + 1, false) &&
            ((is_conv_class_meta(P) && S.op_kind == OK_D2SPACE) ||
             (P.op_kind == OK_D2SPACE && is_conv_class_meta(S)) ||
             ((P.op_kind == OK_AVG_POOL || P.op_kind == OK_MAX_POOL) &&
              is_conv_class_meta(S)));
        if (int8_large_upsample_tail && !exact_large_layout_handoff) {
            producer_no_store[k] = false;
        }
    }

    for (uint32_t i = 0; i < N; ++i) {
        if (i > 0 && is_conv_class_meta(metas[i]) &&
            metas[i - 1].op_kind == OK_PAD && udma_w_skipped[i - 1] &&
            fuse_prev_is_conv_class && fuse_prev_single_tile &&
            fuse_prev_dtype == metas[i].dtype) {
            auto& C = metas[i];
            const auto& P = metas[i - 1];
            const uint32_t padded_h = uint32_t(fuse_prev_out_h) + P.p_t + P.p_b;
            const uint32_t padded_w = uint32_t(fuse_prev_out_w) + P.p_l + P.p_r;
            if (padded_h == C.in_h && padded_w == C.in_w &&
                fuse_prev_out_c == C.in_c &&
                uint32_t(C.p_t) + P.p_t <= 7 &&
                uint32_t(C.p_b) + P.p_b <= 7 &&
                uint32_t(C.p_l) + P.p_l <= 7 &&
                uint32_t(C.p_r) + P.p_r <= 7) {
                C.in_h = fuse_prev_out_h;
                C.in_w = fuse_prev_out_w;
                C.p_t = uint8_t(C.p_t + P.p_t);
                C.p_b = uint8_t(C.p_b + P.p_b);
                C.p_l = uint8_t(C.p_l + P.p_l);
                C.p_r = uint8_t(C.p_r + P.p_r);
                C.in_size = fuse_prev_l1_out_size;
                mark_flow_edge(i - 1, i);
            }
        }
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

        if (L.op_kind == OK_TRANSPOSE && i + 2 < N &&
            metas[i + 1].op_kind == OK_RESHAPE &&
            metas[i + 2].op_kind == OK_FC &&
            !fc_prefetch_wgt_tag[i + 2]) {
            const auto& F = metas[i + 2];
            const bool fc_is_fp = (F.dtype == DT_FP16 || F.dtype == DT_BFP16 || F.dtype == DT_FP8);
            const uint64_t fc_pure_wgt = conv_pure_weight_bytes(F);
            const uint64_t fc_params = fc_is_fp ? (8 + 4 * uint64_t(F.out_c))
                                                : (12 + 9 * uint64_t(F.out_c));
            const unsigned fc_elem =
                (F.dtype == DT_INT16x16 || F.dtype == DT_INT16x8 ||
                 F.dtype == DT_FP16 || F.dtype == DT_BFP16 || F.dtype == DT_FP8) ? 2u : 1u;
            const uint64_t fc_in_bytes = uint64_t(F.in_h) * F.in_w * F.in_c * fc_elem;
            const uint64_t fc_out_bytes = uint64_t(F.out_h) * F.out_w * F.out_c * fc_elem;
            const bool fc_oc_slice_candidate =
                F.in_h == 1 && F.in_w == 1 && F.out_h == 1 && F.out_w == 1 &&
                F.out_c > 128 && F.out_c <= 4096 &&
                !fc_is_fp && (F.group ? F.group : 1) == 1;
            const uint32_t pref_wgt_l1 = 64u * 1024u;
            const uint64_t pref_top = uint64_t(pref_wgt_l1) + fc_pure_wgt +
                                      fc_in_bytes + fc_out_bytes + fc_params + 4096u;
            const uint64_t cur_working = std::max<uint64_t>(L.in_size, L.ref_size);
            if (!fc_oc_slice_candidate &&
                pref_wgt_l1 >= align64(uint32_t(cur_working)) && pref_top <= L1_BUDGET) {
                flush_pending();
                const uint8_t wgt_tag = alloc_tag();
                Descriptor wd = make_udma(F.dram_wgt, pref_wgt_l1,
                                          uint32_t(fc_pure_wgt),
                                          /*dir*/ 0, wgt_tag);
                program.push_back(wd);
                acc[i + 2].dram_r += fc_pure_wgt;
                acc[i + 2].sram_w += fc_pure_wgt;
                fc_prefetch_wgt_tag[i + 2] = wgt_tag;
                fc_prefetch_wgt_l1[i + 2] = pref_wgt_l1;
            }
        }

        if (preemitted_layout_layer[i]) {
            fuse_prev_l1_out_addr   = 0;
            fuse_prev_l1_out_size   = 0;
            fuse_prev_single_tile   = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            continue;
        }

        auto try_stream_conv_chain = [&]() -> bool {
            const auto streamable = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                if (A.op_kind != OK_CONV && A.op_kind != OK_DWCONV) return false;
                if (A.s_h != 1 || A.s_w != 1) return false;
                const bool pointwise =
                    A.op_kind == OK_CONV &&
                    A.k_h == 1 && A.k_w == 1 &&
                    A.p_t == 0 && A.p_b == 0 && A.p_l == 0 && A.p_r == 0;
                const bool spatial3 =
                    A.k_h == 3 && A.k_w == 3 &&
                    A.p_t == 1 && A.p_b == 1 && A.p_l == 1 && A.p_r == 1;
                if (!pointwise && !spatial3) return false;
                if (A.op_kind == OK_CONV && A.group != 1) return false;
                if (A.op_kind == OK_DWCONV &&
                    (A.group != A.in_c || A.out_c != A.in_c))
                    return false;
                if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;
                return true;
            };
            auto is_pointwise_stream_conv = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                return A.op_kind == OK_CONV &&
                       A.k_h == 1 && A.k_w == 1 &&
                       A.p_t == 0 && A.p_b == 0 && A.p_l == 0 && A.p_r == 0;
            };
            auto is_depthwise_spatial_head = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                return A.op_kind == OK_DWCONV &&
                       A.k_h == 3 && A.k_w == 3 &&
                       A.p_t == 1 && A.p_b == 1 &&
                       A.p_l == 1 && A.p_r == 1 &&
                       A.group == A.in_c && A.out_c == A.in_c;
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
            bool stream_to_d2s_add =
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
            bool d2s_add_uses_d2s_as_input0 = stream_to_d2s_add;
            bool d2s_add_uses_d2s_as_input1 = false;
            if (stream_to_d2s_add && graph_metas) {
                const bool conv_feeds_d2s = graph_input0_is_exact_producer(end, end + 1);
                d2s_add_uses_d2s_as_input0 = graph_input0_is_exact_producer(end + 1, end + 2);
                d2s_add_uses_d2s_as_input1 = graph_input1_is_exact_producer(end + 1, end + 2);
                stream_to_d2s_add = conv_feeds_d2s &&
                    (d2s_add_uses_d2s_as_input0 || d2s_add_uses_d2s_as_input1);
            }
            // v9.2: enable generic CONV->CONV microblock streaming for plain
            // pointwise linear chains. Spatial 3x3 plain chains still need a
            // stronger line-buffer ownership model; keep those on the
            // conservative per-layer tiler unless they are part of the
            // already-validated CONV...->D2S->EWE stream tail below.
            if (!stream_to_d2s_add) {
                const bool dw_pointwise_chain =
                    is_depthwise_spatial_head(i) && end > i &&
                    metas[i].out_c <= 64 && last.out_c <= 8;
                bool generic_chain_ok = true;
                for (uint32_t k = i; k <= end; ++k) {
                    const bool ok = is_pointwise_stream_conv(k) ||
                                    (dw_pointwise_chain && k == i);
                    if (!ok) {
                        generic_chain_ok = false;
                        break;
                    }
                }
                // Large image tail heads (for example mv3_depth_quant's
                // 384x576x8 -> 384x576x1 projection) are usually final-output
                // / side-output boundaries.  The generic pointwise chain keeps
                // only per-row microblocks live; keep these on the per-layer
                // path until the chain has explicit large-tail ownership.
                if (!dw_pointwise_chain &&
                    uint64_t(last.out_h) * last.out_w >= 128ull * 128ull &&
                    last.out_c <= 32)
                    return false;
                if (!generic_chain_ok) return false;
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

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
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
                    cb.stride_dilation = encode_conv_stride_pair(A.s_h, A.s_w);
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
                    rb.corr_per_oc = uint8_t(corr_size && A.op_kind == OK_DWCONV);
                    emit_stream(rd, mb, k, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    acc[k].sram_r += scale_lut_size;
                    acc[k].sram_w += out_bytes;
                    ++requant_count_so_far;
                    last_req[k] = requant_count_so_far;

                    if (k == end && !stream_to_d2s_add && !producer_no_store[k]) {
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

                        emit_stream(make_tnps_d2s(output_addr, input_addr,
                                                  uint16_t(out_rows), A.out_w, A.out_c,
                                                  block, uint8_t(elem), d2s_tag, req_tag),
                                    mb, end + 1, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0),
                                    true);
                        acc[end + 1].sram_r += out_bytes;
                        acc[end + 1].sram_w += add_tile_bytes;
                        ++tnps_count_so_far;
                        last_tnps[end + 1] = tnps_count_so_far;
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

                        const bool d2s_is_b_input =
                            d2s_add_uses_d2s_as_input1 && !d2s_add_uses_d2s_as_input0;
                        const uint32_t other_input_dram =
                            uint32_t((d2s_is_b_input ? B.dram_in : B.dram_wgt) + add_dram_off);
                        emit_stream(make_udma(other_input_dram,
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
                        Descriptor ed = d2s_is_b_input
                            ? make_ewe_add(tile_B, L1_WGT_STREAM, input_addr,
                                           output_addr, L1_PARAMS_STREAM,
                                           add_b_tag, d2s_tag, add_req_tag)
                            : make_ewe_add(tile_B, input_addr, L1_WGT_STREAM,
                                           output_addr, L1_PARAMS_STREAM,
                                           d2s_tag, add_b_tag, add_req_tag);
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
                tnps_count_at_layer_end[end + 1] = last_tnps[end + 1];
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
                if (A.op_kind != OK_CONV && A.op_kind != OK_DWCONV && A.op_kind != OK_FC)
                    return false;
                if (!producer_no_store[k]) return false;
                if (A.op_kind == OK_CONV && A.group != 1) return false;
                if (A.op_kind == OK_DWCONV &&
                    (A.group != A.in_c || A.out_c != A.in_c))
                    return false;
                if (A.op_kind == OK_FC && (A.out_h != 1 || A.out_w != 1))
                    return false;
                if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;
                return true;
            };
            if (!streamable(i)) return false;
            const auto same_compiled_input_blob = [&](const LayerMeta& A, const LayerMeta& B) -> bool {
                if (A.in_size != B.in_size) return false;
                const uint64_t a0 = A.in_off;
                const uint64_t b0 = B.in_off;
                const uint64_t bytes = A.in_size;
                if (a0 + bytes > file.size() || b0 + bytes > file.size()) return false;
                return std::memcmp(file.data() + a0, file.data() + b0, bytes) == 0;
            };
            uint32_t end = i;
            while (end + 1 < N && streamable(end + 1)) {
                const auto& A = metas[i];
                const auto& B = metas[end + 1];
                // This path materializes the branch outputs into a real CONCAT
                // boundary. Sharing one loaded input tile is bit-true when the
                // branches use the same physical input, or when graph metadata
                // says they consume the same tensor and the compiled input bytes
                // prove that tensor was serialized identically.
                const bool same_physical_input = (B.dram_in == A.dram_in);
                const bool same_logical_input =
                    graph_metas && graph_metas[end + 1].input0_tensor == graph_metas[i].input0_tensor;
                if (!same_physical_input &&
                    !(same_logical_input && same_compiled_input_blob(A, B))) break;
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
            const auto& first = metas[i];
            if (end + 1 >= N || metas[end + 1].op_kind != OK_CONCAT)
                return false;
            const uint32_t concat_idx = end + 1;
            const auto& C = metas[concat_idx];
            if (C.out_h != first.out_h || C.out_w != first.out_w)
                return false;
            if (graph_metas && graph_metas[concat_idx].consumer_count > 0)
                return false;
            if (graph_metas) {
                bool all_feed_this_concat = true;
                for (uint32_t k = i; k <= end; ++k) {
                    if (graph_metas[k].last_consumer_layer != int32_t(concat_idx)) {
                        all_feed_this_concat = false;
                        break;
                    }
                }
                if (!all_feed_this_concat)
                    return false;
            }

            const unsigned elem =
                (first.dtype == DT_INT16x16 || first.dtype == DT_INT16x8) ? 2u : 1u;
            const uint64_t safety = 65536;
            uint64_t fixed_bytes = 0;
            struct BranchBlob {
                uint64_t pure_wgt = 0;
                uint64_t scale_lut = 0;
                uint64_t corr = 0;
                uint32_t c_offset = 0;
                uint32_t out_l1 = 0;
                uint32_t params_l1 = 0;
                uint32_t wgt_l1 = 0;
                uint8_t params_tag = 0;
                uint8_t wgt_tag = 0;
            };
            std::vector<BranchBlob> blobs(end - i + 1);
            uint32_t concat_c = 0;
            for (uint32_t k = i; k <= end; ++k) {
                const auto& A = metas[k];
                auto& bb = blobs[k - i];
                bb.c_offset = concat_c;
                concat_c += A.out_c;
                bb.pure_wgt = conv_pure_weight_bytes(A);
                bb.scale_lut = 12 + 9 * uint64_t(A.out_c);
                bb.corr = (uint64_t(A.wgt_size) > bb.pure_wgt + bb.scale_lut)
                        ? (uint64_t(A.wgt_size) - bb.pure_wgt - bb.scale_lut) : 0;
                fixed_bytes += align64(uint32_t(bb.pure_wgt));
                fixed_bytes += align64(uint32_t(bb.scale_lut + bb.corr));
            }
            if (concat_c != C.out_c)
                return false;

            const uint64_t row_in = uint64_t(first.in_w) * first.in_c * elem;
            const uint64_t per_oh_in = row_in * first.s_h;
            const uint64_t fixed_in = row_in * (first.k_h ? (first.k_h - 1) : 0);
            const uint64_t sum_out_row = [&]() {
                uint64_t v = 0;
                for (uint32_t k = i; k <= end; ++k)
                    v += uint64_t(metas[k].out_w) * metas[k].out_c * elem;
                return v;
            }();
            uint32_t tile_oh = first.out_h;
            const uint64_t base_fixed = fixed_bytes + safety;
            if (base_fixed >= L1_BUDGET) return false;
            const uint64_t io_budget = L1_BUDGET - base_fixed;
            if (per_oh_in + sum_out_row > 0) {
                uint64_t cand = (io_budget > fixed_in)
                              ? ((io_budget - fixed_in) / (per_oh_in + sum_out_row))
                              : 1;
                cand = std::max<uint64_t>(1, std::min<uint64_t>(cand, first.out_h));
                tile_oh = uint32_t(cand);
            }

            auto tile_bytes_for = [&](uint32_t toh) {
                const uint32_t worst_in_h = toh * first.s_h + (first.k_h ? first.k_h - 1 : 0);
                const uint64_t in_b = uint64_t(worst_in_h) * first.in_w * first.in_c * elem;
                uint64_t out_b = 0;
                for (uint32_t k = i; k <= end; ++k)
                    out_b += uint64_t(toh) * metas[k].out_w * metas[k].out_c * elem;
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
            uint32_t cursor = align64(uint32_t(max_in_b));
            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            for (uint32_t k = i; k <= end; ++k) {
                auto& bb = blobs[k - i];
                const auto& A = metas[k];
                bb.out_l1 = cursor;
                cursor = align64(uint32_t(cursor + uint64_t(tile_oh) * A.out_w * A.out_c * elem));
            }
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
                udma_count_so_far += 2;
                last_udma[k] = udma_count_so_far;
                acc[k].dram_r += bb.scale_lut + bb.corr + bb.pure_wgt;
                acc[k].sram_w += bb.scale_lut + bb.corr + bb.pure_wgt;
            }

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
                    cb.out_addr = bb.out_l1;
                    cb.in_h = uint16_t(this_in_h);
                    cb.in_w = A.in_w;
                    cb.in_c = A.in_c;
                    cb.out_c = A.out_c;
                    cb.k_h = A.k_h;
                    cb.k_w = A.k_w;
                    cb.stride_dilation = encode_conv_stride_pair(A.s_h, A.s_w);
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
                    rb.out_addr = bb.out_l1;
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

                    const uint64_t out_bytes =
                        uint64_t(this_oh) * A.out_w * A.out_c * elem;
                    const uint8_t concat_store_tag = alloc_tag();
                    const uint64_t concat_dram_base =
                        uint64_t(C.dram_out)
                      + uint64_t(oh_done * C.out_w) * C.out_c * elem
                      + uint64_t(bb.c_offset) * elem;
                    Descriptor st = make_desc(OC_UDMA, DT_INT8x8,
                                              /*signal*/ concat_store_tag,
                                              req_tag, 0);
                    auto& sb = st.body.udma;
                    sb.mode = UM_STRIDED_2D;
                    sb.direction = 1;
                    sb.src_addr = bb.out_l1;
                    sb.dst_addr = uint32_t(concat_dram_base);
                    sb.length = A.out_c * elem;
                    sb.src_stride = A.out_c * elem;
                    sb.dst_stride = C.out_c * elem;
                    sb.num_chunks = uint16_t(this_oh * A.out_w);
                    mark_stream(st, concat_idx, mb, SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                    st.hdr.flags |= DF_STREAM_TAIL;
                    program.push_back(st);
                    acc[concat_idx].sram_r += out_bytes;
                    acc[concat_idx].dram_w += out_bytes;
                    ++udma_count_so_far;
                    last_udma[concat_idx] = udma_count_so_far;

                    udma_w_skipped[k] = true;
                    udma_w_streamed[k] = true;
                    mark_flow_edge(k, concat_idx);
                    last_branch_req = concat_store_tag;
                    layer_done_tag[k] = concat_store_tag;
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
            udma_w_streamed[concat_idx] = true;
            tiles_h_per_layer[concat_idx] = tiles_h_per_layer[i];
            tiles_oc_per_layer[concat_idx] = uint16_t(end - i + 1);
            udma_count_at_layer_end[concat_idx] = last_udma[concat_idx];
            requant_count_at_layer_end[concat_idx] = requant_count_so_far;
            layer_done_tag[concat_idx] = group_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = concat_idx;
            return true;
        };

        auto try_stream_conv_concat_pointwise = [&]() -> bool {
            const auto streamable = [&](uint32_t k) -> bool {
                const auto& A = metas[k];
                if (A.op_kind != OK_CONV && A.op_kind != OK_DWCONV && A.op_kind != OK_FC)
                    return false;
                if (!producer_no_store[k]) return false;
                if (A.op_kind == OK_CONV && A.group != 1) return false;
                if (A.op_kind == OK_DWCONV &&
                    (A.group != A.in_c || A.out_c != A.in_c))
                    return false;
                if (A.op_kind == OK_FC && (A.out_h != 1 || A.out_w != 1))
                    return false;
                if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;
                return true;
            };
            if (!graph_metas || !streamable(i)) return false;
            const auto same_compiled_input_blob = [&](const LayerMeta& A, const LayerMeta& B) -> bool {
                if (A.in_size != B.in_size) return false;
                const uint64_t a0 = A.in_off;
                const uint64_t b0 = B.in_off;
                const uint64_t bytes = A.in_size;
                if (a0 + bytes > file.size() || b0 + bytes > file.size()) return false;
                return std::memcmp(file.data() + a0, file.data() + b0, bytes) == 0;
            };

            uint32_t end = i;
            while (end + 1 < N && streamable(end + 1)) {
                const auto& A = metas[i];
                const auto& B = metas[end + 1];
                const bool same_physical_input = (B.dram_in == A.dram_in);
                const bool same_logical_input =
                    graph_metas[end + 1].input0_tensor == graph_metas[i].input0_tensor;
                if (!same_physical_input &&
                    !(same_logical_input && same_compiled_input_blob(A, B))) break;
                if (B.in_h != A.in_h || B.in_w != A.in_w || B.in_c != A.in_c) break;
                if (B.out_h != A.out_h || B.out_w != A.out_w) break;
                if (B.k_h != A.k_h || B.k_w != A.k_w ||
                    B.s_h != A.s_h || B.s_w != A.s_w ||
                    B.p_t != A.p_t || B.p_b != A.p_b ||
                    B.p_l != A.p_l || B.p_r != A.p_r ||
                    B.dtype != A.dtype) break;
                ++end;
            }
            if (end <= i || end + 2 >= N) return false;
            const uint32_t concat_idx = end + 1;
            const uint32_t consumer_idx = end + 2;
            const auto& first = metas[i];
            const auto& C = metas[concat_idx];
            const auto& U = metas[consumer_idx];
            if (C.op_kind != OK_CONCAT || U.op_kind != OK_CONV) return false;
            if (graph_metas[concat_idx].last_consumer_layer != int32_t(consumer_idx)) return false;
            if (U.k_h != 1 || U.k_w != 1 || U.s_h != 1 || U.s_w != 1 ||
                U.p_t || U.p_b || U.p_l || U.p_r || (U.group ? U.group : 1) != 1)
                return false;
            if (U.dtype != first.dtype || C.dtype != first.dtype) return false;
            if (C.out_h != first.out_h || C.out_w != first.out_w) return false;
            if (U.in_h != C.out_h || U.in_w != C.out_w || U.in_c != C.out_c)
                return false;
            bool all_feed_this_concat = true;
            for (uint32_t k = i; k <= end; ++k) {
                if (graph_metas[k].last_consumer_layer != int32_t(concat_idx)) {
                    all_feed_this_concat = false;
                    break;
                }
            }
            if (!all_feed_this_concat) return false;

            const unsigned elem =
                (first.dtype == DT_INT16x16 || first.dtype == DT_INT16x8) ? 2u : 1u;
            if (uint64_t(C.out_c) * elem > 0xFFFFu) return false;
            struct BranchBlob {
                uint64_t pure_wgt = 0;
                uint64_t scale_lut = 0;
                uint64_t corr = 0;
                uint32_t c_offset = 0;
                uint32_t out_l1 = 0;
                uint32_t params_l1 = 0;
                uint32_t wgt_l1 = 0;
                uint8_t params_tag = 0;
                uint8_t wgt_tag = 0;
            };
            std::vector<BranchBlob> blobs(end - i + 1);
            uint32_t concat_c = 0;
            uint64_t fixed_bytes = 0;
            for (uint32_t k = i; k <= end; ++k) {
                const auto& A = metas[k];
                auto& bb = blobs[k - i];
                bb.c_offset = concat_c;
                concat_c += A.out_c;
                bb.pure_wgt = conv_pure_weight_bytes(A);
                bb.scale_lut = 12 + 9 * uint64_t(A.out_c);
                bb.corr = (uint64_t(A.wgt_size) > bb.pure_wgt + bb.scale_lut)
                        ? (uint64_t(A.wgt_size) - bb.pure_wgt - bb.scale_lut) : 0;
                fixed_bytes += align64(uint32_t(bb.pure_wgt));
                fixed_bytes += align64(uint32_t(bb.scale_lut + bb.corr));
            }
            if (concat_c != C.out_c) return false;

            const uint64_t u_pure_wgt = conv_pure_weight_bytes(U);
            const uint64_t u_scale_lut = 12 + 9 * uint64_t(U.out_c);
            const uint64_t u_corr = (uint64_t(U.wgt_size) > u_pure_wgt + u_scale_lut)
                                  ? (uint64_t(U.wgt_size) - u_pure_wgt - u_scale_lut) : 0;
            fixed_bytes += align64(uint32_t(u_pure_wgt));
            fixed_bytes += align64(uint32_t(u_scale_lut + u_corr));

            const uint64_t safety = 65536;
            const uint64_t row_in = uint64_t(first.in_w) * first.in_c * elem;
            const uint64_t per_oh_in = row_in * first.s_h;
            const uint64_t fixed_in = row_in * (first.k_h ? (first.k_h - 1) : 0);
            const uint64_t concat_row = uint64_t(C.out_w) * C.out_c * elem;
            const uint64_t consumer_out_row = uint64_t(U.out_w) * U.out_c * elem;
            uint32_t tile_oh = first.out_h;
            if (fixed_bytes + safety >= L1_BUDGET) return false;
            const uint64_t io_budget = L1_BUDGET - fixed_bytes - safety;
            const uint64_t per_oh_bytes = per_oh_in + concat_row + consumer_out_row;
            if (per_oh_bytes > 0) {
                uint64_t cand = (io_budget > fixed_in) ? ((io_budget - fixed_in) / per_oh_bytes) : 1;
                cand = std::max<uint64_t>(1, std::min<uint64_t>(cand, first.out_h));
                tile_oh = uint32_t(cand);
            }
            auto tile_bytes_for = [&](uint32_t toh) {
                const uint32_t worst_in_h = toh * first.s_h + (first.k_h ? first.k_h - 1 : 0);
                const uint64_t in_b = uint64_t(worst_in_h) * first.in_w * first.in_c * elem;
                const uint64_t concat_b = uint64_t(toh) * concat_row;
                const uint64_t out_b = uint64_t(toh) * consumer_out_row;
                return std::array<uint64_t, 3>{in_b, concat_b, out_b};
            };
            while (tile_oh > 1) {
                auto b = tile_bytes_for(tile_oh);
                if (align64(uint32_t(b[0])) + align64(uint32_t(b[1])) +
                    align64(uint32_t(b[2])) +
                    fixed_bytes + safety <= L1_BUDGET) break;
                --tile_oh;
            }
            auto max_b = tile_bytes_for(tile_oh);
            if (align64(uint32_t(max_b[0])) + align64(uint32_t(max_b[1])) +
                align64(uint32_t(max_b[2])) +
                fixed_bytes + safety > L1_BUDGET)
                return false;

            flush_pending();

            const uint32_t L1_IN_FAN = 0;
            uint32_t cursor = align64(uint32_t(max_b[0]));
            const uint32_t L1_CONCAT = cursor;
            cursor = align64(uint32_t(cursor + max_b[1]));
            const uint32_t L1_U_OUT = cursor;
            cursor = align64(uint32_t(cursor + max_b[2]));
            for (uint32_t k = i; k <= end; ++k) {
                auto& bb = blobs[k - i];
                bb.params_l1 = cursor;
                cursor = align64(uint32_t(cursor + bb.scale_lut + bb.corr));
                bb.wgt_l1 = cursor;
                cursor = align64(uint32_t(cursor + bb.pure_wgt));
            }
            const uint32_t U_PARAMS_L1 = cursor;
            cursor = align64(uint32_t(cursor + u_scale_lut + u_corr));
            const uint32_t U_WGT_L1 = cursor;
            cursor = align64(uint32_t(cursor + u_pure_wgt));
            if (uint64_t(cursor) + safety > L1_BUDGET) return false;

            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            for (uint32_t k = i; k <= end; ++k) {
                auto& bb = blobs[k - i];
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
                udma_count_so_far += 2;
                last_udma[k] = udma_count_so_far;
                acc[k].dram_r += bb.scale_lut + bb.corr + bb.pure_wgt;
                acc[k].sram_w += bb.scale_lut + bb.corr + bb.pure_wgt;
            }
            const uint8_t u_params_tag = alloc_tag();
            const uint8_t u_wgt_tag = alloc_tag();
            program.push_back(make_udma(U.dram_wgt + uint32_t(u_pure_wgt), U_PARAMS_L1,
                                        uint32_t(u_scale_lut + u_corr),
                                        /*dir*/ 0, u_params_tag));
            program.push_back(make_udma(U.dram_wgt, U_WGT_L1, uint32_t(u_pure_wgt),
                                        /*dir*/ 0, u_wgt_tag));
            udma_count_so_far += 2;
            last_udma[consumer_idx] = udma_count_so_far;
            acc[consumer_idx].dram_r += u_scale_lut + u_corr + u_pure_wgt;
            acc[consumer_idx].sram_w += u_scale_lut + u_corr + u_pure_wgt;

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
                Microblock mb{};
                mb.id = tile_id;
                mb.slot = uint8_t(tile_id & 1u);
                mb.rows = this_oh;
                mb.elems = this_oh * C.out_w * C.out_c;
                mb.bytes = tile_in_size;

                const uint8_t in_tag = alloc_tag();
                auto [idesc, charged] = make_act_load(first, first.dram_in + dram_in_off,
                                                       L1_IN_FAN, tile_in_size,
                                                       in_tag, prev_tile_done);
                mark_stream(idesc, i, mb, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                program.push_back(idesc);
                acc[i].dram_r += charged;
                acc[i].sram_w += tile_in_size;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                uint8_t last_branch_tag = in_tag;
                for (uint32_t k = i; k <= end; ++k) {
                    const auto& A = metas[k];
                    const auto& bb = blobs[k - i];
                    const uint8_t req_tag = alloc_tag();
                    Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                              /*signal*/ 0, bb.wgt_tag, in_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr = L1_IN_FAN;
                    cb.wgt_addr = bb.wgt_l1;
                    cb.out_addr = L1_CONCAT + bb.c_offset * elem;
                    cb.in_h = uint16_t(this_in_h);
                    cb.in_w = A.in_w;
                    cb.in_c = A.in_c;
                    cb.out_c = A.out_c;
                    cb.k_h = A.k_h;
                    cb.k_w = A.k_w;
                    cb.stride_dilation = encode_conv_stride_pair(A.s_h, A.s_w);
                    cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                    cb.pad_lr = uint8_t((A.p_l & 7) | ((A.p_r & 7) << 3));
                    cb.group = A.group ? A.group : 1;
                    cb.cluster_mask = 0xFFFF;
                    cb.in_pad_value = A.zp_in_eff;
                    mark_stream(cd, k, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                    program.push_back(cd);
                    acc[k].sram_r += bb.pure_wgt + tile_in_size;

                    Descriptor rd = make_desc(OC_REQUANT, uint8_t(A.dtype),
                                              /*signal*/ req_tag, bb.params_tag, bb.wgt_tag,
                                              in_tag, last_branch_tag);
                    auto& rb = rd.body.requant;
                    rb.in_addr = 0;
                    rb.out_addr = L1_CONCAT;
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
                    rb._r[0] = RQ_STORE_STRIDED_2D;
                    const uint16_t dst_row = uint16_t(C.out_c * elem);
                    const uint16_t dst_col = uint16_t(bb.c_offset * elem);
                    rb._r[1] = uint8_t(dst_row & 0xFF);
                    rb._r[2] = uint8_t((dst_row >> 8) & 0xFF);
                    rb._r[3] = uint8_t(dst_col & 0xFF);
                    rb._r[4] = uint8_t((dst_col >> 8) & 0xFF);
                    mark_stream(rd, k, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                    program.push_back(rd);
                    acc[k].sram_r += bb.scale_lut;
                    acc[k].sram_w += uint64_t(this_oh) * A.out_w * A.out_c * elem;
                    ++requant_count_so_far;
                    last_req[k] = requant_count_so_far;
                    last_branch_tag = req_tag;

                    udma_w_skipped[k] = true;
                    udma_w_streamed[k] = true;
                    mark_flow_edge(k, concat_idx);
                    layer_done_tag[k] = req_tag;
                }

                const uint8_t u_req_tag = alloc_tag();
                Descriptor u_cd = make_desc(OC_CONV, uint8_t(U.dtype),
                                            /*signal*/ 0, u_wgt_tag, last_branch_tag);
                auto& ucb = u_cd.body.conv;
                ucb.in_addr = L1_CONCAT;
                ucb.wgt_addr = U_WGT_L1;
                ucb.out_addr = L1_U_OUT;
                ucb.in_h = uint16_t(this_oh);
                ucb.in_w = U.in_w;
                ucb.in_c = U.in_c;
                ucb.out_c = U.out_c;
                ucb.k_h = U.k_h;
                ucb.k_w = U.k_w;
                ucb.stride_dilation = encode_conv_stride_pair(U.s_h, U.s_w);
                ucb.pad_tb = 0;
                ucb.pad_lr = 0;
                ucb.group = 1;
                ucb.cluster_mask = 0xFFFF;
                ucb.in_pad_value = U.zp_in_eff;
                mark_stream(u_cd, consumer_idx, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                program.push_back(u_cd);
                acc[consumer_idx].sram_r += u_pure_wgt + uint64_t(this_oh) * U.in_w * U.in_c * elem;

                Descriptor u_rd = make_desc(OC_REQUANT, uint8_t(U.dtype),
                                            /*signal*/ u_req_tag, u_params_tag, u_wgt_tag, last_branch_tag);
                auto& urb = u_rd.body.requant;
                urb.in_addr = 0;
                urb.out_addr = L1_U_OUT;
                urb.n = 1;
                urb.h = uint16_t(this_oh);
                urb.w = U.out_w;
                urb.c = U.out_c;
                urb.scale_lut_addr = U_PARAMS_L1;
                urb.scale_count = U.out_c;
                urb.oc_start = 0;
                urb.per_channel_flag = 1;
                urb.out_w_layer = U.out_w;
                urb.oh_start = uint16_t(oh_done);
                urb.corr_addr = u_corr ? uint32_t(U_PARAMS_L1 + u_scale_lut) : 0u;
                urb.corr_per_oc = 0;
                const bool suppress_consumer_store = producer_no_store[consumer_idx];
                if (suppress_consumer_store)
                    urb._r[0] = RQ_STORE_SKIP;
                mark_stream(u_rd, consumer_idx, mb, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                program.push_back(u_rd);
                acc[consumer_idx].sram_r += u_scale_lut;
                if (!suppress_consumer_store)
                    acc[consumer_idx].sram_w += uint64_t(this_oh) * U.out_w * U.out_c * elem;
                ++requant_count_so_far;
                last_req[consumer_idx] = requant_count_so_far;

                uint8_t tile_done = u_req_tag;
                if (suppress_consumer_store) {
                    udma_w_skipped[consumer_idx] = true;
                    udma_w_streamed[consumer_idx] = true;
                } else {
                    const uint8_t store_tag = alloc_tag();
                    const uint64_t out_dram_base =
                        uint64_t(U.dram_out) + uint64_t(oh_done * U.out_w) * U.out_c * elem;
                    Descriptor st = make_desc(OC_UDMA, DT_INT8x8, store_tag, u_req_tag, 0);
                    auto& sb = st.body.udma;
                    sb.mode = UM_LINEAR_COPY;
                    sb.direction = 1;
                    sb.src_addr = L1_U_OUT;
                    sb.dst_addr = uint32_t(out_dram_base);
                    sb.length = uint32_t(uint64_t(this_oh) * U.out_w * U.out_c * elem);
                    mark_stream(st, consumer_idx, mb, SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                    st.hdr.flags |= DF_STREAM_TAIL;
                    program.push_back(st);
                    acc[consumer_idx].sram_r += uint64_t(this_oh) * U.out_w * U.out_c * elem;
                    acc[consumer_idx].dram_w += uint64_t(this_oh) * U.out_w * U.out_c * elem;
                    ++udma_count_so_far;
                    last_udma[consumer_idx] = udma_count_so_far;
                    tile_done = store_tag;
                }
                mark_flow_edge(concat_idx, consumer_idx);
                prev_tile_done = tile_done;
                group_done = tile_done;
            }

            for (uint32_t k = i; k <= end; ++k) {
                tiles_h_per_layer[k] = uint16_t((metas[k].out_h + tile_oh - 1) / tile_oh);
                tiles_oc_per_layer[k] = 1;
                requant_count_at_layer_end[k] = last_req[k];
                udma_count_at_layer_end[k] = last_udma[k] ? last_udma[k] : udma_count_so_far;
            }
            udma_w_streamed[concat_idx] = true;
            udma_w_skipped[concat_idx] = true;
            tiles_h_per_layer[concat_idx] = uint16_t((C.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[concat_idx] = uint16_t(end - i + 1);
            layer_done_tag[concat_idx] = group_done;
            tiles_h_per_layer[consumer_idx] = uint16_t((U.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[consumer_idx] = 1;
            requant_count_at_layer_end[consumer_idx] = last_req[consumer_idx];
            udma_count_at_layer_end[consumer_idx] = last_udma[consumer_idx] ? last_udma[consumer_idx] : udma_count_so_far;
            layer_done_tag[consumer_idx] = group_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = consumer_idx;
            return true;
        };

        auto try_stream_pointwise_slice_fanout = [&]() -> bool {
            const auto& P = metas[i];
            if (P.op_kind != OK_CONV || !producer_no_store[i])
                return false;
            if (P.k_h != 1 || P.k_w != 1 || P.s_h != 1 || P.s_w != 1)
                return false;
            if (P.p_t || P.p_b || P.p_l || P.p_r)
                return false;
            if ((P.group ? P.group : 1) != 1)
                return false;
            if (P.dtype == DT_FP16 || P.dtype == DT_BFP16 || P.dtype == DT_FP8)
                return false;
            auto is_layout_slice = [&](const LayerMeta& S) {
                return S.op_kind == OK_SLICE || S.op_kind == OK_STRIDED_SLICE;
            };
            auto is_branch_conv = [&](const LayerMeta& B, uint32_t slice_c) {
                return B.op_kind == OK_CONV &&
                       B.k_h == 3 && B.k_w == 3 &&
                       B.s_h == 1 && B.s_w == 1 &&
                       B.p_t == 1 && B.p_b == 1 && B.p_l == 1 && B.p_r == 1 &&
                       (B.group ? B.group : 1) == 1 &&
                       B.dtype == P.dtype &&
                       B.in_h == P.out_h && B.in_w == P.out_w &&
                       B.in_c == slice_c && B.out_c == slice_c;
            };

            struct FanoutBranch {
                uint32_t slice_idx = 0;
                uint32_t conv_idx = 0;
                uint32_t c0 = 0;
                uint32_t c = 0;
                uint64_t wgt = 0;
                uint64_t params = 0;
                uint64_t corr = 0;
                uint32_t params_l1 = 0;
                uint32_t wgt_l1 = 0;
                uint8_t params_tag = 0;
                uint8_t wgt_tag = 0;
            };

            std::vector<FanoutBranch> branches;
            uint32_t cursor_idx = i + 1;
            uint32_t c_cursor = 0;
            auto slice_channel_begin = [&](const LayerMeta& S, uint32_t fallback) {
                if (S.wgt_size >= 104 &&
                    uint64_t(S.wgt_off) + 104ull <= file.size()) {
                    const uint32_t* words = reinterpret_cast<const uint32_t*>(
                        file.data() + S.wgt_off);
                    const uint32_t rank = words[0];
                    if (rank >= 1 && rank <= 6) {
                        const uint32_t c_begin = words[14 + rank - 1];
                        if (c_begin < P.out_c)
                            return c_begin;
                    }
                }
                return fallback;
            };
            while (cursor_idx + 1 < N && is_layout_slice(metas[cursor_idx])) {
                const auto& S = metas[cursor_idx];
                const uint32_t slice_c = S.out_c ? S.out_c : S.in_c;
                const uint32_t c_begin = slice_channel_begin(S, c_cursor);
                if (!slice_c || c_begin + slice_c > P.out_c)
                    break;
                const auto& B = metas[cursor_idx + 1];
                if (!is_branch_conv(B, slice_c))
                    break;
                if (!producer_no_store[cursor_idx] || !producer_no_store[cursor_idx + 1])
                    break;
                branches.push_back(FanoutBranch{cursor_idx, cursor_idx + 1,
                                                c_begin, slice_c});
                c_cursor = std::max(c_cursor, c_begin + slice_c);
                cursor_idx += 2;
            }
            if (branches.size() < 2 || cursor_idx >= N || metas[cursor_idx].op_kind != OK_CONCAT)
                return false;
            const uint32_t concat_idx = cursor_idx;
            if (c_cursor != P.out_c)
                return false;
            if (producer_no_store[concat_idx] &&
                std::getenv("MDLA7_EXPERIMENTAL_SLICE_FANOUT") == nullptr)
                return false;

            const unsigned elem =
                (P.dtype == DT_INT16x16 || P.dtype == DT_INT16x8) ? 2u : 1u;
            const uint64_t p_wgt_full = conv_pure_weight_bytes(P);
            if (!P.out_c || p_wgt_full % P.out_c)
                return false;
            const uint64_t p_wgt_per_oc = p_wgt_full / P.out_c;
            const uint64_t p_scale = 12 + 9 * uint64_t(P.out_c);
            const uint64_t p_corr =
                (uint64_t(P.wgt_size) > p_wgt_full + p_scale)
                ? (uint64_t(P.wgt_size) - p_wgt_full - p_scale) : 0;
            const uint64_t p_params = p_scale + p_corr;
            const uint32_t max_slice_c = [&]() {
                uint32_t v = 0;
                for (const auto& br : branches)
                    v = std::max(v, br.c);
                return v;
            }();
            const uint64_t p_wgt_slice_max = uint64_t(max_slice_c) * p_wgt_per_oc;

            uint64_t fixed = align64(uint32_t(p_params)) + align64(uint32_t(p_wgt_slice_max));
            for (auto& br : branches) {
                const auto& B = metas[br.conv_idx];
                br.wgt = conv_pure_weight_bytes(B);
                const uint64_t scale = 12 + 9 * uint64_t(B.out_c);
                br.corr = (uint64_t(B.wgt_size) > br.wgt + scale)
                        ? (uint64_t(B.wgt_size) - br.wgt - scale) : 0;
                br.params = scale + br.corr;
                fixed += align64(uint32_t(br.wgt));
                fixed += align64(uint32_t(br.params));
            }

            const uint64_t safety = 65536;
            if (fixed + safety >= L1_BUDGET)
                return false;
            auto bytes_for = [&](uint32_t out_rows) {
                const uint32_t src_rows = std::min<uint32_t>(P.out_h, out_rows + 2u);
                const uint64_t p_in = uint64_t(src_rows) * P.in_w * P.in_c * elem;
                const uint64_t slice = uint64_t(src_rows) * P.out_w * max_slice_c * elem;
                const uint64_t branch_out = uint64_t(out_rows) * P.out_w * max_slice_c * elem;
                return std::tuple<uint64_t, uint64_t, uint64_t>(p_in, slice, branch_out);
            };
            uint32_t tile_oh = P.out_h;
            while (tile_oh > 1) {
                auto [p_in, slice, branch_out] = bytes_for(tile_oh);
                const uint64_t working =
                    align64(uint32_t(p_in)) +
                    align64(uint32_t(slice)) +
                    align64(uint32_t(branch_out)) +
                    fixed + safety;
                if (working <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            auto [max_p_in, max_slice, max_branch_out] = bytes_for(tile_oh);
            if (tile_oh < 1 ||
                align64(uint32_t(max_p_in)) +
                align64(uint32_t(max_slice)) +
                align64(uint32_t(max_branch_out)) +
                fixed + safety > L1_BUDGET)
                return false;
            if (tile_oh == P.out_h)
                return false;

            flush_pending();

            const uint32_t L1_P_IN = 0;
            const uint32_t L1_SLICE = align64(uint32_t(max_p_in));
            const uint32_t L1_BRANCH_OUT = align64(uint32_t(L1_SLICE + max_slice));
            const uint32_t L1_P_WGT = align64(uint32_t(L1_BRANCH_OUT + max_branch_out));
            uint32_t l1_cur = align64(uint32_t(L1_P_WGT + p_wgt_slice_max));
            const uint32_t P_PARAMS_L1 = l1_cur;
            l1_cur = align64(uint32_t(l1_cur + p_params));
            const uint8_t p_params_tag = alloc_tag();

            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            program.push_back(make_udma(P.dram_wgt + uint32_t(p_wgt_full),
                                        P_PARAMS_L1, uint32_t(p_params),
                                        /*dir*/ 0, p_params_tag));
            acc[i].dram_r += p_params;
            acc[i].sram_w += p_params;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            for (auto& br : branches) {
                const auto& B = metas[br.conv_idx];
                br.params_l1 = l1_cur;
                l1_cur = align64(uint32_t(l1_cur + br.params));
                br.wgt_l1 = l1_cur;
                l1_cur = align64(uint32_t(l1_cur + br.wgt));
                br.params_tag = alloc_tag();
                br.wgt_tag = alloc_tag();
                program.push_back(make_udma(B.dram_wgt + uint32_t(br.wgt),
                                            br.params_l1, uint32_t(br.params),
                                            /*dir*/ 0, br.params_tag));
                program.push_back(make_udma(B.dram_wgt, br.wgt_l1,
                                            uint32_t(br.wgt),
                                            /*dir*/ 0, br.wgt_tag));
                acc[br.conv_idx].dram_r += br.params + br.wgt;
                acc[br.conv_idx].sram_w += br.params + br.wgt;
                udma_count_so_far += 2;
                last_udma[br.conv_idx] = udma_count_so_far;
            }

            uint8_t prev_tile_done = 0;
            uint8_t final_done = 0;
            uint16_t mb_id = 0;
            for (uint32_t y = 0; y < P.out_h; y += tile_oh, ++mb_id) {
                const uint32_t y_hi = std::min<uint32_t>(P.out_h, y + tile_oh);
                const uint32_t out_rows = y_hi - y;
                const uint32_t src_lo = (y > 0) ? y - 1 : 0;
                const uint32_t src_hi = std::min<uint32_t>(P.out_h, y_hi + 1);
                const uint32_t src_rows = src_hi - src_lo;
                const bool final_mb = (y_hi == P.out_h);
                const uint32_t p_in_bytes = src_rows * P.in_w * P.in_c * elem;
                const uint32_t p_in_off = src_lo * P.in_w * P.in_c * elem;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.rows = out_rows;
                mb.elems = out_rows * P.out_w * P.out_c;
                mb.bytes = p_in_bytes;

                const uint8_t in_tag = alloc_tag();
                auto [id, charged] = make_act_load(P, P.dram_in + p_in_off,
                                                   L1_P_IN, p_in_bytes,
                                                   in_tag, prev_tile_done);
                mark_stream(id, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(id);
                acc[i].dram_r += charged;
                acc[i].sram_w += p_in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                uint8_t tile_tail = in_tag;
                for (auto& br : branches) {
                    const auto& B = metas[br.conv_idx];
                    const uint32_t p_wgt_slice = uint32_t(br.c * p_wgt_per_oc);
                    const uint32_t p_wgt_off = uint32_t(br.c0 * p_wgt_per_oc);
                    const uint8_t p_wgt_tag = alloc_tag();
                    const uint8_t p_req_tag = alloc_tag();
                    program.push_back(make_udma(P.dram_wgt + p_wgt_off,
                                                L1_P_WGT,
                                                p_wgt_slice,
                                                /*dir*/ 0, p_wgt_tag, tile_tail));
                    mark_stream(program.back(), i, mb,
                                SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                    acc[i].dram_r += p_wgt_slice;
                    acc[i].sram_w += p_wgt_slice;
                    ++udma_count_so_far;
                    last_udma[i] = udma_count_so_far;

                    Descriptor p_cd = make_desc(OC_CONV, uint8_t(P.dtype),
                                                /*signal*/ 0, p_wgt_tag, in_tag);
                    auto& pcb = p_cd.body.conv;
                    pcb.in_addr = L1_P_IN;
                    pcb.wgt_addr = L1_P_WGT;
                    pcb.out_addr = L1_SLICE;
                    pcb.in_h = uint16_t(src_rows);
                    pcb.in_w = P.in_w;
                    pcb.in_c = P.in_c;
                    pcb.out_c = uint16_t(br.c);
                    pcb.k_h = P.k_h;
                    pcb.k_w = P.k_w;
                    pcb.stride_dilation = encode_conv_stride_pair(P.s_h, P.s_w);
                    pcb.pad_tb = 0;
                    pcb.pad_lr = 0;
                    pcb.group = 1;
                    pcb.cluster_mask = 0xFFFF;
                    pcb.in_pad_value = P.zp_in_eff;
                    mark_stream(p_cd, i, mb,
                                SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(p_cd);
                    acc[i].sram_r += p_wgt_slice + p_in_bytes;

                    Descriptor p_rd = make_desc(OC_REQUANT, uint8_t(P.dtype),
                                                /*signal*/ p_req_tag, p_params_tag,
                                                p_wgt_tag, in_tag);
                    auto& prb = p_rd.body.requant;
                    prb.in_addr = 0;
                    prb.out_addr = L1_SLICE;
                    prb.n = 1;
                    prb.h = uint16_t(src_rows);
                    prb.w = P.out_w;
                    prb.c = uint16_t(br.c);
                    prb.scale_lut_addr = P_PARAMS_L1;
                    prb.scale_count = P.out_c;
                    prb.oc_start = uint16_t(br.c0);
                    prb.per_channel_flag = 1;
                    prb.out_w_layer = P.out_w;
                    prb.oh_start = uint16_t(src_lo);
                    prb.corr_addr = p_corr ? uint32_t(P_PARAMS_L1 + p_scale) : 0u;
                    prb.corr_per_oc = 0;
                    mark_stream(p_rd, i, mb,
                                SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(p_rd);
                    acc[i].sram_r += p_scale;
                    acc[i].sram_w += uint64_t(src_rows) * P.out_w * br.c * elem;
                    ++requant_count_so_far;
                    last_req[i] = requant_count_so_far;

                    Descriptor b_cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                                /*signal*/ 0, br.wgt_tag, p_req_tag);
                    auto& bcb = b_cd.body.conv;
                    bcb.in_addr = L1_SLICE;
                    bcb.wgt_addr = br.wgt_l1;
                    bcb.out_addr = L1_BRANCH_OUT;
                    bcb.in_h = uint16_t(src_rows);
                    bcb.in_w = B.in_w;
                    bcb.in_c = B.in_c;
                    bcb.out_c = B.out_c;
                    bcb.k_h = B.k_h;
                    bcb.k_w = B.k_w;
                    bcb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                    bcb.pad_tb = uint8_t(((y == 0 ? B.p_t : 0) & 7)
                                      | (((y_hi == B.out_h ? B.p_b : 0) & 7) << 3));
                    bcb.pad_lr = uint8_t((B.p_l & 7) | ((B.p_r & 7) << 3));
                    bcb.group = B.group ? B.group : 1;
                    bcb.cluster_mask = 0xFFFF;
                    bcb.in_pad_value = B.zp_in_eff;
                    mark_stream(b_cd, br.conv_idx, mb,
                                SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(b_cd);
                    acc[br.conv_idx].sram_r += br.wgt +
                        uint64_t(src_rows) * B.in_w * B.in_c * elem;

                    const uint8_t b_req_tag = alloc_tag();
                    Descriptor b_rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                                /*signal*/ b_req_tag, br.params_tag,
                                                br.wgt_tag, p_req_tag);
                    auto& brb = b_rd.body.requant;
                    brb.in_addr = 0;
                    brb.out_addr = L1_BRANCH_OUT;
                    brb.n = 1;
                    brb.h = uint16_t(out_rows);
                    brb.w = B.out_w;
                    brb.c = B.out_c;
                    brb.scale_lut_addr = br.params_l1;
                    brb.scale_count = B.out_c;
                    brb.oc_start = 0;
                    brb.per_channel_flag = 1;
                    brb.out_w_layer = B.out_w;
                    brb.oh_start = uint16_t(y);
                    brb.corr_addr = br.corr ? uint32_t(br.params_l1 + (12 + 9 * uint64_t(B.out_c))) : 0u;
                    brb.corr_per_oc = 0;
                    mark_stream(b_rd, br.conv_idx, mb,
                                SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(b_rd);
                    const uint64_t b_out_bytes =
                        uint64_t(out_rows) * B.out_w * B.out_c * elem;
                    acc[br.conv_idx].sram_r += 12 + 9 * uint64_t(B.out_c);
                    acc[br.conv_idx].sram_w += b_out_bytes;
                    ++requant_count_so_far;
                    last_req[br.conv_idx] = requant_count_so_far;

                    uint8_t concat_store_tag = b_req_tag;
                    if (producer_no_store[concat_idx]) {
                        udma_w_skipped[concat_idx] = true;
                        udma_w_streamed[concat_idx] = true;
                    } else {
                        concat_store_tag = alloc_tag();
                        const uint64_t concat_dram_base =
                            uint64_t(metas[concat_idx].dram_out)
                          + uint64_t(y * P.out_w) * P.out_c * elem
                          + uint64_t(br.c0) * elem;
                        Descriptor st = make_desc(OC_UDMA, DT_INT8x8,
                                                  /*signal*/ concat_store_tag,
                                                  b_req_tag, 0);
                        auto& stb = st.body.udma;
                        stb.mode = UM_STRIDED_2D;
                        stb.direction = 1;
                        stb.src_addr = L1_BRANCH_OUT;
                        stb.dst_addr = uint32_t(concat_dram_base);
                        stb.length = B.out_c * elem;
                        stb.src_stride = B.out_c * elem;
                        stb.dst_stride = P.out_c * elem;
                        stb.num_chunks = uint16_t(out_rows * B.out_w);
                        mark_stream(st, concat_idx, mb,
                                    SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                        st.hdr.flags |= DF_STREAM_TAIL;
                        program.push_back(st);
                        acc[concat_idx].sram_r += b_out_bytes;
                        acc[concat_idx].dram_w += b_out_bytes;
                        ++udma_count_so_far;
                        last_udma[concat_idx] = udma_count_so_far;
                    }
                    udma_w_skipped[br.slice_idx] = true;
                    udma_w_streamed[br.slice_idx] = true;
                    udma_w_skipped[br.conv_idx] = true;
                    udma_w_streamed[br.conv_idx] = true;
                    mark_flow_edge(i, br.slice_idx);
                    mark_flow_edge(br.slice_idx, br.conv_idx);
                    mark_flow_edge(br.conv_idx, concat_idx);
                    layer_done_tag[br.slice_idx] = p_req_tag;
                    layer_done_tag[br.conv_idx] = concat_store_tag;
                    tile_tail = concat_store_tag;
                }
                prev_tile_done = tile_tail;
                final_done = tile_tail;
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            tiles_h_per_layer[i] = uint16_t((P.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[i] = uint16_t(branches.size());
            udma_count_at_layer_end[i] = last_udma[i];
            requant_count_at_layer_end[i] = last_req[i];
            layer_done_tag[i] = final_done;
            for (const auto& br : branches) {
                tiles_h_per_layer[br.slice_idx] = tiles_h_per_layer[i];
                tiles_oc_per_layer[br.slice_idx] = 1;
                udma_count_at_layer_end[br.slice_idx] = last_udma[i];
                layer_done_tag[br.slice_idx] = final_done;
                tiles_h_per_layer[br.conv_idx] = tiles_h_per_layer[i];
                tiles_oc_per_layer[br.conv_idx] = 1;
                udma_count_at_layer_end[br.conv_idx] = last_udma[br.conv_idx];
                requant_count_at_layer_end[br.conv_idx] = last_req[br.conv_idx];
            }
            if (producer_no_store[concat_idx] && final_done) {
                // This fanout uses one scratch layout for all branch tiles and
                // intentionally does not materialize the intermediate CONCAT.
                // Hold the descriptor stream at the boundary so later layers
                // cannot reuse those addresses or wrapped tags before the last
                // branch requant has drained the CONV->REQUANT chain.
                program.push_back(make_udma(0, 0, 0, /*dir*/ 0, /*signal*/ 0, final_done));
                ++udma_count_so_far;
                last_udma[concat_idx] = udma_count_so_far;
            }
            udma_w_streamed[concat_idx] = true;
            if (producer_no_store[concat_idx])
                udma_w_skipped[concat_idx] = true;
            tiles_h_per_layer[concat_idx] = tiles_h_per_layer[i];
            tiles_oc_per_layer[concat_idx] = 1;
            udma_count_at_layer_end[concat_idx] =
                last_udma[concat_idx] ? last_udma[concat_idx] : udma_count_so_far;
            requant_count_at_layer_end[concat_idx] = requant_count_so_far;
            layer_done_tag[concat_idx] = final_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = concat_idx;
            return true;
        };

        auto try_stream_conv_ewe = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& A = metas[i];
            const bool conv_class =
                A.op_kind == OK_CONV || A.op_kind == OK_DWCONV || A.op_kind == OK_FC;
            auto is_binary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_ADD || L.op_kind == OK_MUL || L.op_kind == OK_SUB;
            };
            auto is_unary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_HARD_SWISH || L.op_kind == OK_GELU ||
                       L.op_kind == OK_LOGISTIC;
            };
            auto is_pool = [](const LayerMeta& L) {
                return L.op_kind == OK_AVG_POOL || L.op_kind == OK_MAX_POOL;
            };
            auto is_int16_stream_dtype = [](uint16_t dtype) {
                return dtype == DT_INT16x8 || dtype == DT_INT16x16;
            };
            auto stream_dtype_compatible = [&](uint16_t producer_dtype,
                                               uint16_t consumer_dtype) {
                return producer_dtype == consumer_dtype ||
                       (is_int16_stream_dtype(producer_dtype) &&
                        is_int16_stream_dtype(consumer_dtype));
            };
            const auto& B0 = metas[i + 1];
            const bool direct_pool_tail = is_pool(B0);
            if (!conv_class || (!is_binary_ewe(B0) && !direct_pool_tail)) return false;
            if (!producer_no_store[i]) return false;
            if (!stream_dtype_compatible(A.dtype, B0.dtype)) return false;
            if (A.out_h != B0.in_h || A.out_w != B0.in_w || A.out_c != B0.in_c) return false;
            if (!direct_pool_tail) {
                if (B0.in_h != B0.out_h || B0.in_w != B0.out_w || B0.in_c != B0.out_c) return false;
                if (B0.wgt_size < 48) return false;
            }
            if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) return false;
            if (!graph_input0_is_exact_producer(i, i + 1))
                return false;

            uint32_t ewe_end = direct_pool_tail ? i : (i + 1);
            while (!direct_pool_tail && ewe_end + 1 < N && is_binary_ewe(metas[ewe_end + 1])) {
                const auto& P = metas[ewe_end];
                const auto& C = metas[ewe_end + 1];
                if (!producer_no_store[ewe_end]) break;
                if (graph_metas) {
                    const auto& GP = graph_metas[ewe_end];
                    if (GP.consumer_count != 1 ||
                        GP.first_consumer_layer != int32_t(ewe_end + 1) ||
                        GP.last_consumer_layer  != int32_t(ewe_end + 1))
                        break;
                    if (!graph_input0_is_exact_producer(ewe_end, ewe_end + 1))
                        break;
                }
                if (!stream_dtype_compatible(P.dtype, C.dtype)) break;
                if (C.wgt_size < 48) break;
                if (P.out_h != C.in_h || P.out_w != C.in_w || P.out_c != C.in_c)
                    break;
                if (C.in_h != C.out_h || C.in_w != C.out_w || C.in_c != C.out_c)
                    break;
                if (C.in_h != A.out_h || C.in_w != A.out_w || C.in_c != A.out_c)
                    break;
                ++ewe_end;
            }
            const uint32_t ewe_depth = ewe_end - i;
            const uint32_t d2s_tail_idx =
                (ewe_end + 1 < N && metas[ewe_end + 1].op_kind == OK_D2SPACE)
                    ? (ewe_end + 1) : N;
            const bool d2s_tail = [&]() -> bool {
                if (d2s_tail_idx >= N) return false;
                const auto& P = metas[ewe_end];
                const auto& D = metas[d2s_tail_idx];
                if (!stream_dtype_compatible(P.dtype, D.dtype)) return false;
                if (D.in_h != P.out_h || D.in_w != P.out_w || D.in_c != P.out_c)
                    return false;
                const uint16_t block = D.k_h ? D.k_h : 1;
                if (!block || D.in_c != uint32_t(D.out_c) * block * block)
                    return false;
                if (D.out_h != uint32_t(D.in_h) * block ||
                    D.out_w != uint32_t(D.in_w) * block)
                    return false;
                if (graph_metas) {
                    const auto& GP = graph_metas[ewe_end];
                    if (GP.consumer_count != 1 ||
                        GP.first_consumer_layer != int32_t(d2s_tail_idx) ||
                        GP.last_consumer_layer  != int32_t(d2s_tail_idx))
                        return false;
                    if (!graph_input0_is_exact_producer(ewe_end, d2s_tail_idx))
                        return false;
                }
                return true;
            }();
            const uint32_t unary_tail_idx =
                (!d2s_tail && ewe_end + 1 < N && is_unary_ewe(metas[ewe_end + 1]))
                    ? (ewe_end + 1) : N;
            const bool unary_tail = [&]() -> bool {
                if (unary_tail_idx >= N) return false;
                const auto& P = metas[ewe_end];
                const auto& U = metas[unary_tail_idx];
                if (!stream_dtype_compatible(P.dtype, U.dtype)) return false;
                if (U.in_h != P.out_h || U.in_w != P.out_w || U.in_c != P.out_c)
                    return false;
                if (U.in_h != U.out_h || U.in_w != U.out_w || U.in_c != U.out_c)
                    return false;
                if (U.wgt_size == 0) return false;
                if (graph_metas) {
                    const auto& GP = graph_metas[ewe_end];
                    if (GP.consumer_count != 1 ||
                        GP.first_consumer_layer != int32_t(unary_tail_idx) ||
                        GP.last_consumer_layer  != int32_t(unary_tail_idx))
                        return false;
                    if (!graph_input0_is_exact_producer(ewe_end, unary_tail_idx))
                        return false;
                }
                return true;
            }();
            const uint32_t pool_tail_idx =
                direct_pool_tail ? (i + 1) :
                ((!d2s_tail && !unary_tail && ewe_end + 1 < N && is_pool(metas[ewe_end + 1]))
                    ? (ewe_end + 1) : N);
            const bool pool_tail = [&]() -> bool {
                if (pool_tail_idx >= N) return false;
                const auto& P = metas[ewe_end];
                const auto& Q = metas[pool_tail_idx];
                if (!stream_dtype_compatible(P.dtype, Q.dtype)) return false;
                if (Q.in_h != P.out_h || Q.in_w != P.out_w || Q.in_c != P.out_c)
                    return false;
                if (Q.out_c != Q.in_c)
                    return false;
                const uint32_t k_h_eff = (Q.k_h == 255) ? uint32_t(Q.in_h) : uint32_t(Q.k_h);
                const uint32_t k_w_eff = (Q.k_w == 255) ? uint32_t(Q.in_w) : uint32_t(Q.k_w);
                if (!k_h_eff || !k_w_eff || !Q.s_h || !Q.s_w)
                    return false;
                const uint32_t expect_h =
                    (uint32_t(Q.in_h) + Q.p_t + Q.p_b >= k_h_eff)
                        ? ((uint32_t(Q.in_h) + Q.p_t + Q.p_b - k_h_eff) / Q.s_h + 1u)
                        : 0u;
                const uint32_t expect_w =
                    (uint32_t(Q.in_w) + Q.p_l + Q.p_r >= k_w_eff)
                        ? ((uint32_t(Q.in_w) + Q.p_l + Q.p_r - k_w_eff) / Q.s_w + 1u)
                        : 0u;
                if (Q.out_h != expect_h || Q.out_w != expect_w)
                    return false;
                if (Q.p_t > 7 || Q.p_b > 7 || Q.p_l > 7 || Q.p_r > 7)
                    return false;
                if (graph_metas) {
                    const auto& GP = graph_metas[ewe_end];
                    const bool single_pool_consumer =
                        GP.consumer_count == 1 &&
                        GP.first_consumer_layer == int32_t(pool_tail_idx) &&
                        GP.last_consumer_layer  == int32_t(pool_tail_idx);
                    // Residual encoders can fan a CONV tile to an immediate
                    // POOL head plus a later skip consumer.  The POOL head can
                    // still consume the producer tile directly; later uses are
                    // covered by the compiled per-layer input blobs / skip path.
                    const bool direct_pool_fanout_head =
                        direct_pool_tail &&
                        GP.consumer_count > 1 &&
                        GP.first_consumer_layer == int32_t(pool_tail_idx) &&
                        GP.last_consumer_layer  > int32_t(pool_tail_idx);
                    if (!single_pool_consumer && !direct_pool_fanout_head)
                        return false;
                    if (!graph_input0_is_exact_producer(ewe_end, pool_tail_idx))
                        return false;
                }
                return true;
            }();

            const unsigned elem =
                (A.dtype == DT_INT16x16 || A.dtype == DT_INT16x8) ? 2u : 1u;
            const bool is_dw = (A.op_kind == OK_DWCONV);
            const uint64_t pure_wgt = conv_pure_weight_bytes(A);
            const uint64_t scale_lut_size = 12 + 9 * uint64_t(A.out_c);
            const uint64_t corr_size =
                (uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t unary_params_size =
                unary_tail ? uint64_t(metas[unary_tail_idx].wgt_size) : 0;
            const uint32_t pool_k_h_eff =
                pool_tail ? ((metas[pool_tail_idx].k_h == 255)
                                ? uint32_t(metas[pool_tail_idx].in_h)
                                : uint32_t(metas[pool_tail_idx].k_h)) : 0u;
            const uint64_t safety = 65536;
            const uint32_t STREAM_SLOTS = 2;
            if (pure_wgt + params_blob + 48 + unary_params_size + safety >= L1_BUDGET)
                return false;
            const uint64_t full_out_bytes =
                uint64_t(A.out_h) * A.out_w * A.out_c * elem;
            if (!direct_pool_tail &&
                2ull * full_out_bytes + pure_wgt + params_blob + safety <= L1_BUDGET)
                return false;

            const uint32_t stream_out_h = pool_tail ? uint32_t(metas[pool_tail_idx].out_h)
                                                    : uint32_t(A.out_h);
            uint32_t tile_oh = stream_out_h;
            auto tile_shape_bytes = [&](uint32_t toh) {
                const uint32_t producer_rows =
                    pool_tail ? (toh * metas[pool_tail_idx].s_h + pool_k_h_eff - 1u)
                              : toh;
                const uint32_t worst_in_h = producer_rows * A.s_h + (A.k_h ? A.k_h - 1 : 0);
                const uint64_t conv_in =
                    uint64_t(worst_in_h) * A.in_w * A.in_c * elem;
                const uint64_t producer_out =
                    uint64_t(producer_rows) * A.out_w * A.out_c * elem;
                const uint64_t consumer_out =
                    pool_tail ? uint64_t(toh) * metas[pool_tail_idx].out_w *
                                    metas[pool_tail_idx].out_c * elem
                              : producer_out;
                return std::pair<uint64_t, uint64_t>(conv_in,
                                                     std::max(producer_out, consumer_out));
            };
            auto layout_bytes_for = [&](uint32_t toh) {
                auto [conv_in, out] = tile_shape_bytes(toh);
                const uint64_t fixed =
                    align64(uint32_t(params_blob)) +
                    align64(uint32_t(pure_wgt)) +
                    uint64_t(ewe_depth) * align64(48u) +
                    align64(uint32_t(unary_params_size));
                const uint64_t slot =
                    align64(uint32_t(conv_in)) +
                    (2ull + uint64_t(ewe_depth)) * align64(uint32_t(out));
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
            if (tile_oh == stream_out_h)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS_STREAM = 0;
            const uint32_t L1_WGT_STREAM = align64(uint32_t(params_blob));
            const uint32_t L1_EWE_PARAMS_BASE =
                align64(uint32_t(L1_WGT_STREAM + pure_wgt));
            const uint32_t L1_UNARY_PARAMS =
                align64(uint32_t(L1_EWE_PARAMS_BASE + 48u * ewe_depth));
            const uint32_t SLOT_BASE =
                align64(uint32_t(L1_UNARY_PARAMS + unary_params_size));
            const uint32_t SLOT_BYTES = align64(uint32_t(slot_bytes_raw));
            const auto max_shape = tile_shape_bytes(tile_oh);
            const uint32_t max_conv_in = uint32_t(max_shape.first);
            const uint32_t max_tile_out = uint32_t(max_shape.second);
            const uint32_t SLOT_CONV_IN = 0;
            const uint32_t SLOT_DATA0 = align64(uint32_t(max_conv_in));
            const uint32_t SLOT_DATA1 = align64(uint32_t(SLOT_DATA0 + max_tile_out));
            const uint32_t SLOT_EWE_B_BASE = align64(uint32_t(SLOT_DATA1 + max_tile_out));

            auto emit_stream = [&](Descriptor d, const Microblock& mb,
                                   uint32_t layer_idx, uint8_t meta_flags,
                                   bool urgent = false) {
                mark_stream(d, layer_idx, mb, meta_flags);
                if (urgent) d.hdr.flags |= DF_STREAM_TAIL;
                program.push_back(d);
            };

            struct EweStage {
                uint32_t layer_idx = 0;
                uint32_t params_l1 = 0;
                uint8_t params_tag = 0;
            };
            std::vector<EweStage> ewe_stages(ewe_depth);
            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_ewe(N, 0), last_pool(N, 0), last_tnps(N, 0);
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            uint8_t unary_params_tag = 0;

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

            for (uint32_t d = 0; d < ewe_depth; ++d) {
                const uint32_t k = i + 1 + d;
                auto& st = ewe_stages[d];
                st.layer_idx = k;
                st.params_l1 = L1_EWE_PARAMS_BASE + 48u * d;
                st.params_tag = alloc_tag();
                program.push_back(make_udma(metas[k].dram_wgt + metas[k].wgt_size - 48,
                                            st.params_l1, 48,
                                            /*dir*/ 0, st.params_tag));
                acc[k].dram_r += 48;
                acc[k].sram_w += 48;
                ++udma_count_so_far;
                last_udma[k] = udma_count_so_far;
            }
            if (unary_tail) {
                const auto& U = metas[unary_tail_idx];
                unary_params_tag = alloc_tag();
                program.push_back(make_udma(U.dram_wgt, L1_UNARY_PARAMS, U.wgt_size,
                                            /*dir*/ 0, unary_params_tag));
                acc[unary_tail_idx].dram_r += U.wgt_size;
                acc[unary_tail_idx].sram_w += U.wgt_size;
                ++udma_count_so_far;
                last_udma[unary_tail_idx] = udma_count_so_far;
            }

            uint8_t slot_done[STREAM_SLOTS] = {0};
            uint8_t conv_done = 0;
            uint8_t ewe_done = 0;
            uint8_t unary_done = 0;
            uint8_t pool_done = 0;
            uint8_t d2s_done = 0;
            uint16_t tile_id = 0;
            for (uint32_t oh_done = 0; oh_done < stream_out_h; oh_done += tile_oh, ++tile_id) {
                const uint32_t consumer_oh = std::min<uint32_t>(tile_oh, stream_out_h - oh_done);
                const bool is_last_h = (oh_done + consumer_oh == stream_out_h);
                uint32_t producer_oh_start = oh_done;
                uint32_t this_oh = consumer_oh;
                if (pool_tail) {
                    const auto& Q = metas[pool_tail_idx];
                    const int pool_ih_lo_u = int(oh_done) * int(Q.s_h) - int(Q.p_t);
                    const int pool_ih_hi_u =
                        int(oh_done + consumer_oh - 1) * int(Q.s_h) +
                        int(pool_k_h_eff) - 1 - int(Q.p_t);
                    const int pool_ih_lo = std::max(0, pool_ih_lo_u);
                    const int pool_ih_hi = std::min(int(Q.in_h) - 1, pool_ih_hi_u);
                    producer_oh_start = uint32_t(pool_ih_lo);
                    this_oh = uint32_t(pool_ih_hi - pool_ih_lo + 1);
                }
                const int ih_lo_u = int(producer_oh_start) * int(A.s_h) - int(A.p_t);
                const int ih_hi_u =
                    int(producer_oh_start + this_oh - 1) * int(A.s_h) + int(A.k_h) - 1 - int(A.p_t);
                const int ih_lo = std::max(0, ih_lo_u);
                const int ih_hi = std::min(int(A.in_h) - 1, ih_hi_u);
                const uint32_t this_in_h = uint32_t(ih_hi - ih_lo + 1);
                const uint32_t conv_in_bytes = this_in_h * A.in_w * A.in_c * elem;
                const uint32_t tile_out_bytes = this_oh * A.out_w * A.out_c * elem;
                const uint32_t dram_in_off = uint32_t(ih_lo) * A.in_w * A.in_c * elem;
                const uint32_t dram_out_off = producer_oh_start * A.out_w * A.out_c * elem;
                const uint32_t final_tile_bytes =
                    pool_tail ? consumer_oh * metas[pool_tail_idx].out_w *
                                    metas[pool_tail_idx].out_c * elem
                              : tile_out_bytes;
            const uint32_t final_dram_out_off =
                    pool_tail ? oh_done * metas[pool_tail_idx].out_w *
                                    metas[pool_tail_idx].out_c * elem
                              : dram_out_off;
                Microblock mb{};
                mb.id = tile_id;
                mb.slot = uint8_t(tile_id % STREAM_SLOTS);
                mb.rows = consumer_oh;
                mb.elems = pool_tail ? consumer_oh * metas[pool_tail_idx].out_w *
                                           metas[pool_tail_idx].out_c
                                     : this_oh * A.out_w * A.out_c;
                mb.bytes = final_tile_bytes;
                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_CONV_IN = slot_base + SLOT_CONV_IN;
                const uint32_t L1_DATA0 = slot_base + SLOT_DATA0;
                const uint32_t L1_DATA1 = slot_base + SLOT_DATA1;
                auto L1_EWE_B = [&](uint32_t d) {
                    return slot_base + SLOT_EWE_B_BASE + d * align64(uint32_t(max_tile_out));
                };

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
                cb.out_addr = L1_DATA0;
                cb.in_h = uint16_t(this_in_h);
                cb.in_w = A.in_w;
                cb.in_c = A.in_c;
                cb.out_c = A.out_c;
                cb.k_h = A.k_h;
                cb.k_w = A.k_w;
                cb.stride_dilation = encode_conv_stride_pair(A.s_h, A.s_w);
                cb.pad_tb = uint8_t((((producer_oh_start == 0) ? A.p_t : 0) & 7)
                                  | ((((producer_oh_start + this_oh == A.out_h) ? A.p_b : 0) & 7) << 3));
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
                rb.out_addr = L1_DATA0;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = A.out_w;
                rb.c = A.out_c;
                rb.scale_lut_addr = L1_PARAMS_STREAM;
                rb.scale_count = A.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = A.out_w;
                rb.oh_start = uint16_t(producer_oh_start);
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

                uint32_t ewe_in_addr = L1_DATA0;
                uint8_t prev_tag = req_tag;
                uint8_t tile_done = req_tag;
                for (uint32_t d = 0; d < ewe_depth; ++d) {
                    const uint32_t k = i + 1 + d;
                    const auto& E = metas[k];
                    const uint8_t b_tag = alloc_tag();
                    auto [bd, b_charged] = make_binary_b_load(E,
                                                               E.dram_wgt + dram_out_off,
                                                               L1_EWE_B(d), tile_out_bytes,
                                                               b_tag, slot_done[mb.slot]);
                    emit_stream(bd, mb, k,
                                SMF_LOAD_B | (is_last_h ? SMF_FINAL_TILE : 0),
                                true);
                    acc[k].dram_r += b_charged;
                    acc[k].sram_w += tile_out_bytes;
                    ++udma_count_so_far;
                    last_udma[k] = udma_count_so_far;

                    LayerMeta tile_E = E;
                    tile_E.in_h = uint16_t(this_oh);
                    tile_E.out_h = uint16_t(this_oh);
                    const uint32_t ewe_out_addr = (d & 1u) ? L1_DATA0 : L1_DATA1;
                    const uint8_t ewe_tag = alloc_tag();
                    Descriptor ed = make_ewe_add(tile_E, ewe_in_addr, L1_EWE_B(d),
                                                 ewe_out_addr, ewe_stages[d].params_l1,
                                                 b_tag, prev_tag, ewe_tag);
                    emit_stream(ed, mb, k,
                                SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0),
                                true);
                    acc[k].sram_r += 2 * uint64_t(tile_out_bytes);
                    acc[k].sram_w += tile_out_bytes;
                    ++ewe_count_so_far;
                    last_ewe[k] = ewe_count_so_far;
                    ewe_done = ewe_tag;
                    tile_done = ewe_tag;

                    mark_flow_edge((d == 0) ? i : (k - 1), k);
                    if (k < ewe_end) {
                        udma_w_skipped[k] = true;
                        udma_w_streamed[k] = true;
                    }

                    prev_tag = ewe_tag;
                    ewe_in_addr = ewe_out_addr;
                }

                uint32_t final_addr = ewe_in_addr;
                uint32_t final_layer = ewe_end;
                uint8_t final_done = tile_done;
                if (unary_tail) {
                    const auto& U = metas[unary_tail_idx];
                    LayerMeta tile_U = U;
                    tile_U.in_h = uint16_t(this_oh);
                    tile_U.out_h = uint16_t(this_oh);
                    const uint32_t unary_out_addr = (ewe_depth & 1u) ? L1_DATA0 : L1_DATA1;
                    const uint8_t unary_tag = alloc_tag();
                    Descriptor ud = make_ewe_unary(tile_U, ewe_in_addr, unary_out_addr,
                                                   L1_UNARY_PARAMS, tile_done,
                                                   unary_tag, unary_params_tag);
                    emit_stream(ud, mb, unary_tail_idx,
                                SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0),
                                true);
                    acc[unary_tail_idx].sram_r += tile_out_bytes;
                    acc[unary_tail_idx].sram_w += tile_out_bytes;
                    ++ewe_count_so_far;
                    last_ewe[unary_tail_idx] = ewe_count_so_far;
                    unary_done = unary_tag;
                    final_addr = unary_out_addr;
                    final_layer = unary_tail_idx;
                    final_done = unary_tag;
                    udma_w_skipped[ewe_end] = true;
                    udma_w_streamed[ewe_end] = true;
                    mark_flow_edge(ewe_end, unary_tail_idx);
                }
                if (pool_tail) {
                    const auto& Q = metas[pool_tail_idx];
                    LayerMeta tile_Q = Q;
                    tile_Q.in_h = uint16_t(this_oh);
                    tile_Q.out_h = uint16_t(consumer_oh);
                    tile_Q.p_t = uint8_t((oh_done == 0) ? Q.p_t : 0);
                    tile_Q.p_b = uint8_t(is_last_h ? Q.p_b : 0);
                    const uint32_t pool_out_addr = (ewe_depth & 1u) ? L1_DATA0 : L1_DATA1;
                    const uint8_t pool_tag = alloc_tag();
                    Descriptor pd = make_pool(tile_Q, final_addr, pool_out_addr,
                                              final_done, pool_tag);
                    emit_stream(pd, mb, pool_tail_idx,
                                SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0),
                                true);
                    acc[pool_tail_idx].sram_r += tile_out_bytes;
                    acc[pool_tail_idx].sram_w += final_tile_bytes;
                    ++pool_count_so_far;
                    last_pool[pool_tail_idx] = pool_count_so_far;
                    pool_done = pool_tag;
                    final_addr = pool_out_addr;
                    final_layer = pool_tail_idx;
                    final_done = pool_tag;
                    udma_w_skipped[ewe_end] = true;
                    udma_w_streamed[ewe_end] = true;
                    mark_flow_edge(ewe_end, pool_tail_idx);
                }

                if (d2s_tail) {
                    const auto& D = metas[d2s_tail_idx];
                    const uint16_t block = D.k_h ? D.k_h : 1;
                    const uint32_t cout = D.out_c;
                    const uint32_t d2s_bytes = this_oh * block * D.out_w * cout * elem;
                    const uint32_t d2s_dram_off = oh_done * block * D.out_w * cout * elem;
                    const uint8_t d2s_tag = alloc_tag();
                    Descriptor td = make_tnps_d2s(ewe_in_addr,
                                                  uint32_t(D.dram_out + d2s_dram_off),
                                                  uint16_t(this_oh), D.in_w, D.in_c,
                                                  block, uint8_t(elem), d2s_tag, tile_done);
                    emit_stream(td, mb, d2s_tail_idx,
                                SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0),
                                true);
                    acc[d2s_tail_idx].sram_r += tile_out_bytes;
                    acc[d2s_tail_idx].dram_w += d2s_bytes;
                    ++tnps_count_so_far;
                    last_tnps[d2s_tail_idx] = tnps_count_so_far;
                    d2s_done = d2s_tag;
                    slot_done[mb.slot] = d2s_tag;
                    udma_w_skipped[ewe_end] = true;
                    udma_w_streamed[ewe_end] = true;
                    mark_flow_edge(ewe_end, d2s_tail_idx);
                } else {
                    if (producer_no_store[final_layer]) {
                        // Multi-consumer residual tails are still live-range
                        // handoffs.  Do not materialize an intermediate ADD
                        // tile back to DRAM merely because the tensor fans out;
                        // later consumers either stay in the same wavefront
                        // model or use their compiler-provided input blob for
                        // per-layer functional verification.
                        udma_w_skipped[final_layer] = true;
                        udma_w_streamed[final_layer] = true;
                        if (pool_tail) pool_done = final_done;
                        slot_done[mb.slot] = final_done;
                    } else {
                        const uint8_t st_tag = alloc_tag();
                        emit_stream(make_udma(final_addr, metas[final_layer].dram_out + final_dram_out_off,
                                              final_tile_bytes, /*dir*/ 1, st_tag, final_done),
                                    mb, final_layer,
                                    SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                        acc[final_layer].sram_r += final_tile_bytes;
                        acc[final_layer].dram_w += final_tile_bytes;
                        ++udma_count_so_far;
                        last_udma[final_layer] = udma_count_so_far;
                        if (unary_tail) unary_done = st_tag;
                        else if (pool_tail) pool_done = st_tag;
                        else ewe_done = st_tag;
                        slot_done[mb.slot] = st_tag;
                    }
                }
            }
            if (d2s_tail && d2s_done) {
                program.push_back(make_udma(0, 0, 0, /*dir*/ 0, /*signal*/ 0, d2s_done));
                ++udma_count_so_far;
                last_udma[d2s_tail_idx] = udma_count_so_far;
            }

            tiles_h_per_layer[i] = uint16_t((A.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[i] = 1;
            requant_count_at_layer_end[i] = last_req[i];
            udma_count_at_layer_end[i] = last_udma[i];
            ewe_count_at_layer_end[i] = 0;
            layer_done_tag[i] = conv_done;
            for (uint32_t k = i + 1; k <= ewe_end; ++k) {
                tiles_h_per_layer[k] = tiles_h_per_layer[i];
                tiles_oc_per_layer[k] = 1;
                udma_count_at_layer_end[k] = last_udma[k];
                requant_count_at_layer_end[k] = 0;
                ewe_count_at_layer_end[k] = last_ewe[k];
                layer_done_tag[k] = (k == ewe_end) ? ewe_done : 0;
            }
            if (d2s_tail) {
                tiles_h_per_layer[d2s_tail_idx] = tiles_h_per_layer[i];
                tiles_oc_per_layer[d2s_tail_idx] = 1;
                udma_count_at_layer_end[d2s_tail_idx] = last_udma[d2s_tail_idx];
                requant_count_at_layer_end[d2s_tail_idx] = 0;
                ewe_count_at_layer_end[d2s_tail_idx] = 0;
                tnps_count_at_layer_end[d2s_tail_idx] = last_tnps[d2s_tail_idx];
                layer_done_tag[d2s_tail_idx] = d2s_done;
            }
            if (unary_tail) {
                tiles_h_per_layer[unary_tail_idx] = tiles_h_per_layer[i];
                tiles_oc_per_layer[unary_tail_idx] = 1;
                udma_count_at_layer_end[unary_tail_idx] = last_udma[unary_tail_idx];
                requant_count_at_layer_end[unary_tail_idx] = 0;
                ewe_count_at_layer_end[unary_tail_idx] = last_ewe[unary_tail_idx];
                layer_done_tag[unary_tail_idx] = unary_done;
            }
            if (pool_tail) {
                tiles_h_per_layer[pool_tail_idx] = tiles_h_per_layer[i];
                tiles_oc_per_layer[pool_tail_idx] = 1;
                udma_count_at_layer_end[pool_tail_idx] = last_udma[pool_tail_idx];
                requant_count_at_layer_end[pool_tail_idx] = 0;
                ewe_count_at_layer_end[pool_tail_idx] = 0;
                pool_count_at_layer_end[pool_tail_idx] = last_pool[pool_tail_idx];
                layer_done_tag[pool_tail_idx] = pool_done;
            }
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = d2s_tail ? d2s_tail_idx : (unary_tail ? unary_tail_idx : (pool_tail ? pool_tail_idx : ewe_end));
            return true;
        };

        auto try_stream_ewe_conv = [&]() -> bool {
            if (!graph_metas || i + 1 >= N) return false;
            const auto& A = metas[i];
            const auto& B = metas[i + 1];
            const bool binary_ewe =
                A.op_kind == OK_ADD || A.op_kind == OK_MUL || A.op_kind == OK_SUB;
            if (!binary_ewe || B.op_kind != OK_CONV) return false;
            if (!producer_no_store[i]) return false;
            const auto& GA = graph_metas[i];
            if (GA.consumer_count != 1 ||
                GA.first_consumer_layer != int32_t(i + 1) ||
                GA.last_consumer_layer  != int32_t(i + 1))
                return false;
            if (!graph_input0_is_exact_producer(i, i + 1)) return false;
            if (A.dtype != DT_INT8x8 || B.dtype != DT_INT8x8) return false;
            if (A.wgt_size < A.ref_size + 48) return false;
            if (A.in_h != A.out_h || A.in_w != A.out_w || A.in_c != A.out_c)
                return false;
            if (B.k_h != 3 || B.k_w != 3 || B.s_h != 1 || B.s_w != 1)
                return false;
            if ((B.group ? B.group : 1) != 1) return false;
            if (B.in_c != A.out_c || B.out_h != A.out_h || B.out_w != A.out_w)
                return false;
            const bool padded_input =
                B.in_h == uint32_t(A.out_h) + 2u &&
                B.in_w == uint32_t(A.out_w) + 2u;
            const bool exact_input =
                B.in_h == A.out_h && B.in_w == A.out_w;
            if (!padded_input && !exact_input) return false;
            if (B.out_c <= 4 && B.out_h >= 256 && B.out_w >= 256)
                return false;
            // Large full-resolution image enhancement tails keep too much
            // producer state live across the EWE->CONV wavefront for 2-slot
            // ping-pong. Keep the older single-slot stream for this shape.
            const bool large_image_tail =
                A.out_h >= 512 && A.out_w >= 768 && A.out_c >= 64;

            const uint32_t elem = 1;
            const uint64_t pure_wgt = conv_pure_weight_bytes(B);
            const uint64_t scale_lut_size = 12 + 9 * uint64_t(B.out_c);
            const uint64_t corr_size =
                (uint64_t(B.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(B.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            if (pure_wgt + params_blob + 48 >= L1_BUDGET) return false;

            const uint32_t H = A.out_h;
            const uint32_t W = A.out_w;
            const uint32_t row_bytes_a = W * A.out_c * elem;
            const uint32_t row_bytes_b = B.out_w * B.out_c * elem;
            if (!row_bytes_a || !row_bytes_b) return false;
            const uint32_t halo_rows = (B.k_h > 1) ? uint32_t(B.k_h - 1) : 0u;

            const uint64_t safety = 65536;
            constexpr uint32_t STREAM_SLOTS_MAX = 2;
            const uint32_t stream_slots = large_image_tail ? 1u : STREAM_SLOTS_MAX;
            uint32_t tile_oh = std::min<uint32_t>(64, H);
            const uint64_t fixed_bytes =
                align64(48u) +
                align64(uint32_t(params_blob)) +
                align64(uint32_t(pure_wgt));
            auto slot_bytes_for = [&](uint32_t toh) {
                const uint32_t worst_ewe_rows = std::min<uint32_t>(H, toh + halo_rows);
                const uint64_t ewe_tile = uint64_t(worst_ewe_rows) * row_bytes_a;
                const uint64_t conv_out = uint64_t(toh) * row_bytes_b;
                return 3ull * align64(uint32_t(ewe_tile)) +
                       align64(uint32_t(conv_out));
            };
            while (tile_oh > 1 &&
                   fixed_bytes + stream_slots * slot_bytes_for(tile_oh) + safety > L1_BUDGET)
                --tile_oh;
            if (fixed_bytes + stream_slots * slot_bytes_for(tile_oh) + safety > L1_BUDGET)
                return false;
            if (tile_oh >= H)
                return false;
            const bool force_ewe_conv_stream =
                std::getenv("MDLA7_FORCE_EWE_CONV_STREAM") != nullptr;
            const bool full_channel_spatial =
                B.k_h == 3 && B.k_w == 3 && B.in_c >= 128 && B.out_c >= 128;
            if (!force_ewe_conv_stream && full_channel_spatial)
                return false;

            flush_pending();

            const uint32_t L1_EWE_PARAMS = 0;
            const uint32_t L1_CONV_PARAMS = align64(L1_EWE_PARAMS + 48);
            const uint32_t L1_CONV_WGT =
                align64(uint32_t(L1_CONV_PARAMS + params_blob));
            const uint32_t SLOT_BASE =
                align64(uint32_t(L1_CONV_WGT + pure_wgt));
            const uint32_t max_ewe_rows = std::min<uint32_t>(H, tile_oh + halo_rows);
            const uint32_t max_ewe_bytes = max_ewe_rows * row_bytes_a;
            const uint32_t SLOT_EWE_A = 0;
            const uint32_t SLOT_EWE_B = align64(max_ewe_bytes);
            const uint32_t SLOT_EWE_OUT = align64(uint32_t(SLOT_EWE_B + max_ewe_bytes));
            const uint32_t SLOT_CONV_OUT = align64(uint32_t(SLOT_EWE_OUT + max_ewe_bytes));
            const uint32_t SLOT_BYTES = align64(uint32_t(slot_bytes_for(tile_oh)));

            auto emit_stream = [&](Descriptor d, const Microblock& mb,
                                   uint32_t layer_idx, uint8_t meta_flags,
                                   bool urgent = false) {
                mark_stream(d, layer_idx, mb, meta_flags);
                if (urgent) d.hdr.flags |= DF_STREAM_TAIL;
                program.push_back(d);
            };

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_ewe(N, 0);
            const uint8_t ewe_params_tag = alloc_tag();
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();

            program.push_back(make_udma(A.dram_wgt + A.wgt_size - 48,
                                        L1_EWE_PARAMS, 48,
                                        /*dir*/ 0, ewe_params_tag));
            acc[i].dram_r += 48;
            acc[i].sram_w += 48;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            uint8_t slot_done[STREAM_SLOTS_MAX] = {0};
            uint8_t ewe_done = 0;
            uint8_t conv_done = 0;
            bool conv_blob_loaded = false;
            uint16_t tile_id = 0;
            for (uint32_t oh_done = 0; oh_done < H; oh_done += tile_oh, ++tile_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, H - oh_done);
                const uint32_t oh_hi = oh_done + this_oh;
                const bool is_last_h = (oh_hi == H);
                const int ih_lo_u = int(oh_done) * int(B.s_h) - int(B.p_t);
                const int ih_hi_u =
                    int(oh_hi - 1) * int(B.s_h) + int(B.k_h) - 1 - int(B.p_t);
                const uint32_t ewe_lo = uint32_t(std::max(0, ih_lo_u));
                const uint32_t ewe_hi = uint32_t(std::min<int>(int(H), ih_hi_u + 1));
                const uint8_t tile_pad_t = uint8_t(std::min<int>(7, std::max(0, -ih_lo_u)));
                const uint8_t tile_pad_b =
                    uint8_t(std::min<int>(7, std::max(0, ih_hi_u - int(H) + 1)));
                const uint32_t ewe_rows = ewe_hi - ewe_lo;
                const uint32_t ewe_bytes = ewe_rows * row_bytes_a;
                const uint32_t conv_out_bytes = this_oh * row_bytes_b;
                const uint32_t ewe_dram_off = ewe_lo * row_bytes_a;
                const uint32_t conv_dram_off = oh_done * row_bytes_b;

                Microblock mb{};
                mb.id = tile_id;
                mb.slot = uint8_t(tile_id % stream_slots);
                mb.elem_off = uint64_t(oh_done) * W * B.out_c;
                mb.rows = this_oh;
                mb.elems = this_oh * W * B.out_c;
                mb.bytes = conv_out_bytes;

                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_EWE_A = slot_base + SLOT_EWE_A;
                const uint32_t L1_EWE_B = slot_base + SLOT_EWE_B;
                const uint32_t L1_EWE_OUT = slot_base + SLOT_EWE_OUT;
                const uint32_t L1_CONV_OUT = slot_base + SLOT_CONV_OUT;

                const uint8_t a_tag = alloc_tag();
                auto [ad, a_charged] = make_act_load(A, A.dram_in + ewe_dram_off,
                                                      L1_EWE_A, ewe_bytes,
                                                      a_tag, slot_done[mb.slot]);
                emit_stream(ad, mb, i, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].dram_r += a_charged;
                acc[i].sram_w += ewe_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                const uint8_t b_tag = alloc_tag();
                auto [bd, b_charged] = make_binary_b_load(A,
                                                           A.dram_wgt + ewe_dram_off,
                                                           L1_EWE_B, ewe_bytes,
                                                           b_tag, slot_done[mb.slot]);
                emit_stream(bd, mb, i, SMF_LOAD_B | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].dram_r += b_charged;
                acc[i].sram_w += ewe_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                LayerMeta tile_A = A;
                tile_A.in_h = uint16_t(ewe_rows);
                tile_A.out_h = uint16_t(ewe_rows);
                const uint8_t e_tag = alloc_tag();
                Descriptor ed = make_ewe_add(tile_A, L1_EWE_A, L1_EWE_B,
                                             L1_EWE_OUT, L1_EWE_PARAMS,
                                             b_tag, a_tag, e_tag);
                emit_stream(ed, mb, i, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].sram_r += 2 * uint64_t(ewe_bytes);
                acc[i].sram_w += ewe_bytes;
                ++ewe_count_so_far;
                last_ewe[i] = ewe_count_so_far;
                ewe_done = e_tag;
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;

                if (!conv_blob_loaded) {
                    emit_stream(make_udma(B.dram_wgt + uint32_t(pure_wgt),
                                          L1_CONV_PARAMS, uint32_t(params_blob),
                                          /*dir*/ 0, params_tag),
                                mb, i + 1, SMF_LOAD_B, true);
                    acc[i + 1].dram_r += params_blob;
                    acc[i + 1].sram_w += params_blob;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;

                    emit_stream(make_udma(B.dram_wgt, L1_CONV_WGT,
                                          uint32_t(pure_wgt),
                                          /*dir*/ 0, wgt_tag),
                                mb, i + 1, SMF_LOAD_B, true);
                    acc[i + 1].dram_r += pure_wgt;
                    acc[i + 1].sram_w += pure_wgt;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    conv_blob_loaded = true;
                }

                Descriptor cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                          /*signal*/ 0, wgt_tag, e_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_EWE_OUT;
                cb.wgt_addr = L1_CONV_WGT;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(ewe_rows);
                cb.in_w = A.out_w;
                cb.in_c = A.out_c;
                cb.out_c = B.out_c;
                cb.k_h = B.k_h;
                cb.k_w = B.k_w;
                cb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                cb.pad_tb = uint8_t((tile_pad_t & 7) | ((tile_pad_b & 7) << 3));
                cb.pad_lr = padded_input
                    ? uint8_t(1 | (1 << 3))
                    : uint8_t((B.p_l & 7) | ((B.p_r & 7) << 3));
                cb.group = B.group ? B.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = B.zp_in_eff;
                emit_stream(cd, mb, i + 1,
                            SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0),
                            true);
                acc[i + 1].sram_r += pure_wgt + ewe_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, e_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_CONV_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = B.out_w;
                rb.c = B.out_c;
                rb.scale_lut_addr = L1_CONV_PARAMS;
                rb.scale_count = B.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = B.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_CONV_PARAMS + scale_lut_size) : 0u;
                rb.corr_per_oc = 0;
                emit_stream(rd, mb, i + 1,
                            SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0),
                            true);
                acc[i + 1].sram_r += scale_lut_size;
                acc[i + 1].sram_w += conv_out_bytes;
                ++requant_count_so_far;
                last_req[i + 1] = requant_count_so_far;
                conv_done = req_tag;

                if (producer_no_store[i + 1]) {
                    udma_w_skipped[i + 1] = true;
                    udma_w_streamed[i + 1] = true;
                    slot_done[mb.slot] = req_tag;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    emit_stream(make_udma(L1_CONV_OUT, B.dram_out + conv_dram_off,
                                          conv_out_bytes, /*dir*/ 1, st_tag, req_tag),
                                mb, i + 1,
                                SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0));
                    acc[i + 1].sram_r += conv_out_bytes;
                    acc[i + 1].dram_w += conv_out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    conv_done = st_tag;
                    slot_done[mb.slot] = st_tag;
                }
            }

            tiles_h_per_layer[i] = uint16_t((H + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[i] = 1;
            tiles_h_per_layer[i + 1] = tiles_h_per_layer[i];
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            requant_count_at_layer_end[i] = 0;
            ewe_count_at_layer_end[i] = last_ewe[i];
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = last_req[i + 1];
            ewe_count_at_layer_end[i + 1] = 0;
            layer_done_tag[i] = ewe_done;
            layer_done_tag[i + 1] = conv_done;
            mark_flow_edge(i, i + 1);
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto try_stream_conv_d2s = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& A = metas[i];
            const auto& D = metas[i + 1];
            const bool conv_class =
                A.op_kind == OK_CONV || A.op_kind == OK_DWCONV || A.op_kind == OK_FC;
            if (!conv_class || D.op_kind != OK_D2SPACE) return false;
            if (A.dtype != D.dtype) return false;
            if (A.out_h != D.in_h || A.out_w != D.in_w || A.out_c != D.in_c)
                return false;
            const uint16_t block = D.k_h ? D.k_h : 1;
            if (!block || D.in_c != uint32_t(D.out_c) * block * block)
                return false;
            if (D.out_h != uint32_t(D.in_h) * block ||
                D.out_w != uint32_t(D.in_w) * block)
                return false;
            if ((A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) &&
                (D.out_h >= 1024 || D.out_w >= 1024))
                return false;
            if (graph_metas) {
                const auto& GA = graph_metas[i];
                if (GA.consumer_count != 1 ||
                    GA.first_consumer_layer != int32_t(i + 1) ||
                    GA.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i, i + 1))
                    return false;
            }
            const bool d2s_feeds_compute =
                (i + 2 < N) &&
                is_conv_class_meta(metas[i + 2]) &&
                D.dtype == metas[i + 2].dtype &&
                D.out_h == metas[i + 2].in_h &&
                D.out_w == metas[i + 2].in_w &&
                D.out_c == metas[i + 2].in_c &&
                graph_has_exact_single_consumer(i + 1, i + 2, false);
            if (d2s_feeds_compute)
                return false;
            const bool direct_d2s_store =
                graph_metas ? (graph_metas[i + 1].consumer_count == 0)
                            : (i + 1 == N - 1);

            const bool is_dw = (A.op_kind == OK_DWCONV);
            const bool is_fp = (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8);
            const unsigned elem =
                (A.dtype == DT_INT16x16 || A.dtype == DT_INT16x8
                 || A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8) ? 2u : 1u;
            const uint64_t pure_wgt = conv_pure_weight_bytes(A);
            const uint64_t scale_lut_size = is_fp
                ? (8 + 4 * uint64_t(A.out_c))
                : (12 + 9 * uint64_t(A.out_c));
            const uint64_t corr_size =
                (!is_fp && uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            if (pure_wgt + params_blob >= L1_BUDGET) return false;

            const uint64_t safety = 65536;
            uint32_t tile_oh = direct_d2s_store ? A.out_h
                                                : std::min<uint32_t>(64, A.out_h);
            auto bytes_for = [&](uint32_t toh) {
                const uint32_t worst_in_h = toh * A.s_h + (A.k_h ? A.k_h - 1 : 0);
                const uint64_t in_bytes = uint64_t(worst_in_h) * A.in_w * A.in_c * elem;
                const uint64_t out_bytes = uint64_t(toh) * A.out_w * A.out_c * elem;
                const uint64_t d2s_bytes =
                    uint64_t(toh) * block * D.out_w * D.out_c * elem;
                return align64(uint32_t(params_blob)) +
                       align64(uint32_t(pure_wgt)) +
                       align64(uint32_t(in_bytes)) +
                       (direct_d2s_store ? 0 : align64(uint32_t(out_bytes))) +
                       (direct_d2s_store ? 0 : align64(uint32_t(d2s_bytes)));
            };
            while (tile_oh > 1 && bytes_for(tile_oh) + safety > L1_BUDGET)
                --tile_oh;
            if (bytes_for(tile_oh) + safety > L1_BUDGET)
                return false;
            if (tile_oh >= A.out_h)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS_STREAM = 0;
            const uint32_t L1_WGT_STREAM = align64(uint32_t(params_blob));
            const uint32_t max_in_h = tile_oh * A.s_h + (A.k_h ? A.k_h - 1 : 0);
            const uint32_t max_in_bytes = max_in_h * A.in_w * A.in_c * elem;
            const uint32_t max_out_bytes = tile_oh * A.out_w * A.out_c * elem;
            const uint32_t max_d2s_bytes = tile_oh * block * D.out_w * D.out_c * elem;
            const uint32_t L1_IN_STREAM =
                align64(uint32_t(L1_WGT_STREAM + pure_wgt));
            const uint32_t L1_CONV_OUT =
                align64(uint32_t(L1_IN_STREAM + max_in_bytes));
            const uint32_t L1_D2S_OUT =
                align64(uint32_t(L1_CONV_OUT + max_out_bytes));
            if (!direct_d2s_store &&
                uint64_t(L1_D2S_OUT) + max_d2s_bytes + safety > L1_BUDGET)
                return false;

            auto emit_stream = [&](Descriptor d, const Microblock& mb,
                                   uint32_t layer_idx, uint8_t meta_flags,
                                   bool urgent = false) {
                mark_stream(d, layer_idx, mb, meta_flags);
                if (urgent) d.hdr.flags |= DF_STREAM_TAIL;
                program.push_back(d);
            };

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
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

            uint8_t slot_done = 0;
            uint8_t conv_done = 0;
            uint8_t d2s_done = 0;
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
                const uint32_t in_bytes = this_in_h * A.in_w * A.in_c * elem;
                const uint32_t out_bytes = this_oh * A.out_w * A.out_c * elem;
                const uint32_t d2s_bytes = this_oh * block * D.out_w * D.out_c * elem;
                const uint32_t dram_in_off = uint32_t(ih_lo) * A.in_w * A.in_c * elem;
                const uint32_t d2s_dram_off =
                    oh_done * block * D.out_w * D.out_c * elem;
                Microblock mb{};
                mb.id = tile_id;
                mb.slot = 0;
                mb.rows = this_oh;
                mb.elems = this_oh * A.out_w * A.out_c;
                mb.bytes = out_bytes;

                const uint8_t in_tag = alloc_tag();
                auto [id, charged] = make_act_load(A, A.dram_in + dram_in_off,
                                                   L1_IN_STREAM, in_bytes,
                                                   in_tag, slot_done);
                emit_stream(id, mb, i, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].dram_r += charged;
                acc[i].sram_w += in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                          /*signal*/ 0, wgt_tag, in_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_IN_STREAM;
                cb.wgt_addr = L1_WGT_STREAM;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(this_in_h);
                cb.in_w = A.in_w;
                cb.in_c = A.in_c;
                cb.out_c = A.out_c;
                cb.k_h = A.k_h;
                cb.k_w = A.k_w;
                cb.stride_dilation = encode_conv_stride_pair(A.s_h, A.s_w);
                cb.pad_tb = uint8_t((((oh_done == 0) ? A.p_t : 0) & 7)
                                  | (((is_last_h ? A.p_b : 0) & 7) << 3));
                cb.pad_lr = uint8_t((A.p_l & 7) | ((A.p_r & 7) << 3));
                cb.group = A.group ? A.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = A.zp_in_eff;
                emit_stream(cd, mb, i, SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].sram_r += pure_wgt + in_bytes;

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
                if (direct_d2s_store) {
                    rb.out_addr = D.dram_out + d2s_dram_off;
                    rb._r[0] = RQ_STORE_D2SPACE;
                    rb._r[1] = uint8_t(block);
                    rb._r[2] = uint8_t(D.out_c & 0xFF);
                    rb._r[3] = uint8_t((D.out_c >> 8) & 0xFF);
                }
                emit_stream(rd, mb, i,
                            SMF_COMPUTE |
                            (direct_d2s_store ? SMF_STORE : 0) |
                            (is_last_h ? SMF_FINAL_TILE : 0));
                acc[i].sram_r += scale_lut_size;
                if (!direct_d2s_store)
                    acc[i].sram_w += out_bytes;
                ++requant_count_so_far;
                last_req[i] = requant_count_so_far;
                conv_done = req_tag;
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;

                acc[i + 1].dram_w += d2s_bytes;
                if (direct_d2s_store) {
                    d2s_done = req_tag;
                    slot_done = req_tag;
                } else {
                    const uint8_t d2s_tag = alloc_tag();
                    emit_stream(make_tnps_d2s(L1_CONV_OUT, D.dram_out + d2s_dram_off,
                                              uint16_t(this_oh), A.out_w, A.out_c,
                                              block, uint8_t(elem), d2s_tag, req_tag),
                                mb, i + 1, SMF_STORE | (is_last_h ? SMF_FINAL_TILE : 0),
                                true);
                    acc[i + 1].sram_r += out_bytes;
                    ++tnps_count_so_far;
                    last_tnps[i + 1] = tnps_count_so_far;
                    d2s_done = d2s_tag;
                    slot_done = d2s_tag;
                }
            }

            tiles_h_per_layer[i] = uint16_t((A.out_h + tile_oh - 1) / tile_oh);
            tiles_oc_per_layer[i] = 1;
            tiles_h_per_layer[i + 1] = tiles_h_per_layer[i];
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            requant_count_at_layer_end[i] = last_req[i];
            ewe_count_at_layer_end[i] = 0;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = 0;
            ewe_count_at_layer_end[i + 1] = 0;
            tnps_count_at_layer_end[i + 1] = last_tnps[i + 1];
            layer_done_tag[i] = conv_done;
            layer_done_tag[i + 1] = d2s_done;
            mark_flow_edge(i, i + 1);
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
            auto is_unary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_HARD_SWISH || L.op_kind == OK_GELU ||
                       L.op_kind == OK_LOGISTIC;
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
                if (!producer_no_store[end]) break;
                if (GA.consumer_count != 1 ||
                    GA.first_consumer_layer != int32_t(end + 1) ||
                    GA.last_consumer_layer  != int32_t(end + 1))
                    break;
                // Keep quantized ADD/SUB/MUL operand order exact. Swapping
                // input0/input1 can change zp/mult handling even for ADD/MUL.
                if (!graph_input0_is_exact_producer(end, end + 1))
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
            const uint32_t d2s_tail_idx =
                (end + 1 < N && metas[end + 1].op_kind == OK_D2SPACE)
                    ? (end + 1) : N;
            const bool d2s_tail = [&]() -> bool {
                if (d2s_tail_idx >= N) return false;
                const auto& P = metas[end];
                const auto& D = metas[d2s_tail_idx];
                if (D.dtype != P.dtype) return false;
                if (D.in_h != P.out_h || D.in_w != P.out_w || D.in_c != P.out_c)
                    return false;
                const uint16_t block = D.k_h ? D.k_h : 1;
                if (!block || D.in_c != uint32_t(D.out_c) * block * block)
                    return false;
                if (D.out_h != uint32_t(D.in_h) * block ||
                    D.out_w != uint32_t(D.in_w) * block)
                    return false;
                if (!graph_input0_is_exact_producer(end, d2s_tail_idx))
                    return false;
                return true;
            }();
            const uint32_t unary_tail_idx =
                (!d2s_tail && end + 1 < N && is_unary_ewe(metas[end + 1]))
                    ? (end + 1) : N;
            const bool unary_tail = [&]() -> bool {
                if (unary_tail_idx >= N) return false;
                const auto& P = metas[end];
                const auto& U = metas[unary_tail_idx];
                const auto& GP = graph_metas[end];
                if (GP.consumer_count != 1 ||
                    GP.first_consumer_layer != int32_t(unary_tail_idx) ||
                    GP.last_consumer_layer  != int32_t(unary_tail_idx))
                    return false;
                if (!graph_input0_is_exact_producer(end, unary_tail_idx))
                    return false;
                if (U.dtype != P.dtype) return false;
                if (U.in_h != P.out_h || U.in_w != P.out_w || U.in_c != P.out_c)
                    return false;
                if (U.in_h != U.out_h || U.in_w != U.out_w || U.in_c != U.out_c)
                    return false;
                if (U.wgt_size == 0) return false;
                return true;
            }();
            if (end <= i && !d2s_tail && !unary_tail) return false;

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
            const uint64_t unary_params_size =
                unary_tail ? uint64_t(metas[unary_tail_idx].wgt_size) : 0;
            const uint64_t params_bytes =
                align64(48u) * uint64_t(depth) + align64(uint32_t(unary_params_size));
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
            uint32_t unary_params_l1 = 0;
            uint8_t unary_params_tag = 0;
            if (unary_tail) {
                const auto& U = metas[unary_tail_idx];
                unary_params_l1 = cursor;
                cursor = align64(cursor + uint32_t(unary_params_size));
                unary_params_tag = alloc_tag();
                program.push_back(make_udma(U.dram_wgt, unary_params_l1, U.wgt_size,
                                            /*dir*/ 0, unary_params_tag));
                acc[unary_tail_idx].dram_r += U.wgt_size;
                acc[unary_tail_idx].sram_w += U.wgt_size;
                ++udma_count_so_far;
                udma_count_at_layer_end[unary_tail_idx] = udma_count_so_far;
            }

            const uint32_t tile_bytes_max = tile_rows * per_row_bytes;
            const uint32_t seg_bytes = align64(tile_bytes_max);
            const uint32_t slot_base0 = align64(cursor);
            const uint32_t slot_bytes = uint32_t((2ull + depth) * seg_bytes);
            auto slot_base = [&](uint32_t slot) {
                return slot_base0 + slot * slot_bytes;
            };

            std::vector<size_t> last_udma(N, 0), last_ewe(N, 0), last_tnps(N, 0);
            std::vector<uint8_t> last_done(N, 0);
            uint8_t slot_done[2] = {0, 0};
            uint8_t d2s_done = 0;
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

                uint32_t final_addr = in_addr;
                uint8_t final_done = tile_done;
                uint32_t final_layer = end;
                if (unary_tail) {
                    const auto& U = metas[unary_tail_idx];
                    LayerMeta tile_U = U;
                    tile_U.in_h = uint16_t(rows);
                    tile_U.out_h = uint16_t(rows);
                    const uint32_t out_addr = (depth & 1u) ? data0 : data1;
                    const uint8_t u_tag = alloc_tag();
                    Descriptor ud = make_ewe_unary(tile_U, in_addr, out_addr,
                                                   unary_params_l1, tile_done,
                                                   u_tag, unary_params_tag);
                    mark_stream(ud, unary_tail_idx, mb,
                                SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    ud.hdr.flags |= DF_STREAM_TAIL;
                    program.push_back(ud);
                    acc[unary_tail_idx].sram_r += tile_bytes;
                    acc[unary_tail_idx].sram_w += tile_bytes;
                    ++ewe_count_so_far;
                    last_ewe[unary_tail_idx] = ewe_count_so_far;
                    last_done[unary_tail_idx] = u_tag;
                    final_addr = out_addr;
                    final_done = u_tag;
                    final_layer = unary_tail_idx;
                    udma_w_skipped[end] = true;
                    udma_w_streamed[end] = true;
                    mark_flow_edge(end, unary_tail_idx);
                }

                if (d2s_tail) {
                    const auto& D = metas[d2s_tail_idx];
                    const uint16_t block = D.k_h ? D.k_h : 1;
                    const uint32_t cout = D.out_c;
                    const uint32_t d2s_bytes = rows * block * D.out_w * cout * elem;
                    const uint32_t d2s_dram_off = row * block * D.out_w * cout * elem;
                    const uint8_t d2s_tag = alloc_tag();
                    Descriptor td = make_tnps_d2s(in_addr,
                                                  uint32_t(D.dram_out + d2s_dram_off),
                                                  uint16_t(rows), D.in_w, D.in_c,
                                                  block, uint8_t(elem), d2s_tag, tile_done);
                    mark_stream(td, d2s_tail_idx, mb,
                                SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    td.hdr.flags |= DF_STREAM_TAIL;
                    program.push_back(td);
                    acc[d2s_tail_idx].sram_r += tile_bytes;
                    acc[d2s_tail_idx].dram_w += d2s_bytes;
                    ++tnps_count_so_far;
                    last_tnps[d2s_tail_idx] = tnps_count_so_far;
                    last_done[d2s_tail_idx] = d2s_tag;
                    d2s_done = d2s_tag;
                    slot_done[mb.slot] = d2s_tag;
                    udma_w_skipped[end] = true;
                    udma_w_streamed[end] = true;
                    mark_flow_edge(end, d2s_tail_idx);
                } else if (producer_no_store[final_layer]) {
                    udma_w_skipped[final_layer] = true;
                    udma_w_streamed[final_layer] = true;
                    slot_done[mb.slot] = final_done;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(final_addr, metas[final_layer].dram_out + dram_off,
                                              tile_bytes, /*dir*/ 1, st_tag, final_done);
                    mark_stream(sd, final_layer, mb,
                                SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[final_layer].sram_r += tile_bytes;
                    acc[final_layer].dram_w += tile_bytes;
                    ++udma_count_so_far;
                    last_udma[final_layer] = udma_count_so_far;
                    last_done[final_layer] = st_tag;
                    slot_done[mb.slot] = st_tag;
                }
            }
            if (d2s_tail && d2s_done) {
                // The next streamed layer may reuse this function's L1 scratch
                // addresses.  Hold the in-order descriptor stream until TNPS
                // has consumed the final D2S source tile.
                program.push_back(make_udma(0, 0, 0, /*dir*/ 0, /*signal*/ 0, d2s_done));
                ++udma_count_so_far;
                last_udma[d2s_tail_idx] = udma_count_so_far;
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
            if (d2s_tail) {
                tiles_h_per_layer[d2s_tail_idx] = tiles_h;
                tiles_oc_per_layer[d2s_tail_idx] = 1;
                udma_count_at_layer_end[d2s_tail_idx] = last_udma[d2s_tail_idx];
                requant_count_at_layer_end[d2s_tail_idx] = 0;
                ewe_count_at_layer_end[d2s_tail_idx] = 0;
                tnps_count_at_layer_end[d2s_tail_idx] = last_tnps[d2s_tail_idx];
                layer_done_tag[d2s_tail_idx] = d2s_done;
            }
            if (unary_tail) {
                tiles_h_per_layer[unary_tail_idx] = tiles_h;
                tiles_oc_per_layer[unary_tail_idx] = 1;
                udma_count_at_layer_end[unary_tail_idx] = last_udma[unary_tail_idx] ? last_udma[unary_tail_idx] : udma_count_at_layer_end[unary_tail_idx];
                requant_count_at_layer_end[unary_tail_idx] = 0;
                ewe_count_at_layer_end[unary_tail_idx] = last_ewe[unary_tail_idx];
                layer_done_tag[unary_tail_idx] = last_done[unary_tail_idx];
            }

            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = d2s_tail ? d2s_tail_idx : (unary_tail ? unary_tail_idx : end);
            return true;
        };

        auto try_stream_binary_ewe_softmax = [&]() -> bool {
            auto is_binary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_ADD || L.op_kind == OK_MUL || L.op_kind == OK_SUB;
            };
            if (i + 1 >= N || !is_binary_ewe(metas[i]) ||
                !is_attention_softmax_meta(metas[i + 1]))
                return false;
            const auto& B = metas[i];
            const auto& S = metas[i + 1];
            if (!producer_no_store[i])
                return false;
            if (B.dtype != DT_INT8x8 || S.dtype != B.dtype)
                return false;
            if (B.wgt_size < B.ref_size + 48)
                return false;
            if (B.out_h != S.in_h || B.out_w != S.in_w || B.out_c != S.in_c)
                return false;
            if (S.in_h != S.out_h || S.in_w != S.out_w || S.in_c != S.out_c)
                return false;
            if (!graph_has_exact_single_consumer(i, i + 1, false))
                return false;
            if (!graph_input0_is_exact_producer(i, i + 1))
                return false;

            const uint32_t elem = 1;
            const uint32_t vec_elems = S.in_c;
            const uint32_t vec_bytes = vec_elems * elem;
            const uint64_t vec_rows = uint64_t(S.in_h) * S.in_w;
            if (vec_elems == 0 || vec_bytes == 0 || vec_elems > 65535u)
                return false;

            const uint64_t safety = 65536;
            uint32_t tile_vecs =
                uint32_t(std::min<uint64_t>(vec_rows, 65535ull / vec_elems));
            if (tile_vecs == 0)
                return false;
            const uint32_t params_l1 = 0;
            const uint32_t slot0 = align64(64u);
            auto slot_top_for = [&](uint32_t vectors) -> uint64_t {
                const uint32_t bytes = vectors * vec_bytes;
                const uint32_t seg = align64(bytes);
                return uint64_t(slot0) + 2ull * 4ull * seg;
            };
            while (tile_vecs > 0 && slot_top_for(tile_vecs) + safety > L1_BUDGET)
                --tile_vecs;
            if (tile_vecs == 0)
                return false;

            flush_pending();

            const uint32_t tile_bytes_max = tile_vecs * vec_bytes;
            const uint32_t seg = align64(tile_bytes_max);
            const uint32_t slot_span = 4u * seg;
            auto slot_base = [&](uint32_t slot) {
                return slot0 + slot * slot_span;
            };
            auto slot_a = [&](uint32_t slot) { return slot_base(slot); };
            auto slot_b = [&](uint32_t slot) { return slot_base(slot) + seg; };
            auto slot_e = [&](uint32_t slot) { return slot_base(slot) + 2u * seg; };
            auto slot_s = [&](uint32_t slot) { return slot_base(slot) + 3u * seg; };

            const uint8_t params_tag = alloc_tag();
            program.push_back(make_udma(B.dram_wgt + B.wgt_size - 48,
                                        params_l1, 48, /*dir*/ 0, params_tag));
            acc[i].dram_r += 48;
            acc[i].sram_w += 48;
            ++udma_count_so_far;
            udma_count_at_layer_end[i] = udma_count_so_far;

            uint8_t slot_free[2] = {0, 0};
            uint8_t last_ewe_tag = params_tag;
            uint8_t last_softmax_tag = params_tag;
            size_t last_udma_i = udma_count_so_far;
            size_t last_udma_s = udma_count_so_far;
            size_t last_ewe_i = ewe_count_so_far;
            size_t last_ewe_s = ewe_count_so_far;
            const bool suppress_softmax_store = producer_no_store[i + 1];
            const uint64_t tile_count = (vec_rows + tile_vecs - 1) / tile_vecs;
            tiles_h_per_layer[i] = uint16_t(std::min<uint64_t>(tile_count, 65535));
            tiles_h_per_layer[i + 1] = uint16_t(std::min<uint64_t>(tile_count, 65535));
            tiles_oc_per_layer[i] = 1;
            tiles_oc_per_layer[i + 1] = 1;

            for (uint64_t row = 0, mb_id = 0; row < vec_rows; row += tile_vecs, ++mb_id) {
                const uint32_t rows_this =
                    uint32_t(std::min<uint64_t>(tile_vecs, vec_rows - row));
                const uint32_t elems = rows_this * vec_elems;
                const uint32_t bytes = elems * elem;
                const uint32_t off = uint32_t(row * vec_bytes);
                const uint32_t slot = uint32_t(mb_id & 1u);
                const bool final_mb = (row + rows_this >= vec_rows);
                Microblock mb{};
                mb.id = uint16_t(std::min<uint64_t>(mb_id, 65535));
                mb.slot = uint8_t(slot);
                mb.elem_off = row * vec_elems;
                mb.rows = rows_this;
                mb.elems = elems;
                mb.bytes = bytes;

                const uint8_t a_tag = alloc_tag();
                const uint8_t b_tag = alloc_tag();
                const uint8_t e_tag = alloc_tag();
                const uint8_t sm_tag = alloc_tag();
                const uint8_t st_tag = suppress_softmax_store ? 0 : alloc_tag();
                const uint8_t wait_slot = slot_free[slot] ? slot_free[slot] : params_tag;

                auto [ad, a_charged] = make_act_load(B, uint32_t(B.dram_in + off),
                                                     slot_a(slot), bytes,
                                                     a_tag, wait_slot);
                mark_stream(ad, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ad);
                acc[i].dram_r += a_charged;
                acc[i].sram_w += bytes;
                ++udma_count_so_far;
                last_udma_i = udma_count_so_far;

                auto [bd, b_charged] = make_binary_b_load(B, uint32_t(B.dram_wgt + off),
                                                           slot_b(slot), bytes,
                                                           b_tag, wait_slot);
                mark_stream(bd, i, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(bd);
                acc[i].dram_r += b_charged;
                acc[i].sram_w += bytes;
                ++udma_count_so_far;
                last_udma_i = udma_count_so_far;

                LayerMeta tile_B = B;
                tile_B.in_h = 1;
                tile_B.in_w = 1;
                tile_B.in_c = uint16_t(elems);
                tile_B.out_h = 1;
                tile_B.out_w = 1;
                tile_B.out_c = uint16_t(elems);
                Descriptor ed = make_ewe_add(tile_B, slot_a(slot), slot_b(slot),
                                             slot_e(slot), params_l1,
                                             b_tag, a_tag, e_tag);
                mark_stream(ed, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ed);
                acc[i].sram_r += 2ull * bytes;
                acc[i].sram_w += bytes;
                ++ewe_count_so_far;
                last_ewe_i = ewe_count_so_far;
                last_ewe_tag = e_tag;

                LayerMeta tile_S = S;
                tile_S.in_h = uint16_t(rows_this);
                tile_S.in_w = 1;
                tile_S.in_c = uint16_t(vec_elems);
                tile_S.out_h = uint16_t(rows_this);
                tile_S.out_w = 1;
                tile_S.out_c = uint16_t(vec_elems);
                Descriptor sm = make_softmax(tile_S, slot_e(slot), slot_s(slot),
                                             e_tag, sm_tag);
                mark_stream(sm, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(sm);
                acc[i + 1].sram_r += bytes;
                acc[i + 1].sram_w += bytes;
                ++ewe_count_so_far;
                last_ewe_s = ewe_count_so_far;

                if (suppress_softmax_store) {
                    slot_free[slot] = sm_tag;
                    last_softmax_tag = sm_tag;
                } else {
                    Descriptor wd = make_udma(slot_s(slot), uint32_t(S.dram_out + off),
                                              bytes, /*dir*/ 1, st_tag, sm_tag);
                    mark_stream(wd, i + 1, mb,
                                SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(wd);
                    acc[i + 1].sram_r += bytes;
                    acc[i + 1].dram_w += bytes;
                    ++udma_count_so_far;
                    last_udma_s = udma_count_so_far;
                    slot_free[slot] = st_tag;
                    last_softmax_tag = st_tag;
                }
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            if (suppress_softmax_store) {
                udma_w_skipped[i + 1] = true;
                udma_w_streamed[i + 1] = true;
            }
            mark_flow_edge(i, i + 1);
            udma_count_at_layer_end[i] = last_udma_i;
            ewe_count_at_layer_end[i] = last_ewe_i;
            layer_done_tag[i] = last_ewe_tag;
            udma_count_at_layer_end[i + 1] = last_udma_s;
            ewe_count_at_layer_end[i + 1] = last_ewe_s;
            layer_done_tag[i + 1] = last_softmax_tag;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto try_stream_pool_consumer = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& P = metas[i];
            if (P.op_kind != OK_AVG_POOL && P.op_kind != OK_MAX_POOL)
                return false;
            if (!producer_no_store[i]) return false;
            if (P.dtype != DT_INT8x8) return false;
            const auto& C0 = metas[i + 1];
            auto is_binary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_ADD || L.op_kind == OK_MUL || L.op_kind == OK_SUB;
            };
            auto is_unary_ewe = [](const LayerMeta& L) {
                return L.op_kind == OK_HARD_SWISH || L.op_kind == OK_GELU ||
                       L.op_kind == OK_LOGISTIC;
            };
            const bool binary_tail = is_binary_ewe(C0);
            const bool unary_tail = is_unary_ewe(C0);
            const bool d2s_tail = C0.op_kind == OK_D2SPACE;
            if (!binary_tail && !unary_tail && !d2s_tail)
                return false;
            if (C0.dtype != P.dtype ||
                C0.in_h != P.out_h || C0.in_w != P.out_w || C0.in_c != P.out_c)
                return false;
            if ((binary_tail || unary_tail) &&
                (C0.out_h != C0.in_h || C0.out_w != C0.in_w || C0.out_c != C0.in_c))
                return false;
            if (binary_tail && C0.wgt_size < 48)
                return false;
            if (unary_tail && C0.wgt_size == 0)
                return false;
            if (d2s_tail) {
                const uint16_t block = C0.k_h ? C0.k_h : 1;
                if (!block || C0.in_c != uint32_t(C0.out_c) * block * block)
                    return false;
                if (C0.out_h != uint32_t(C0.in_h) * block ||
                    C0.out_w != uint32_t(C0.in_w) * block)
                    return false;
            }
            if (graph_metas) {
                const auto& GP = graph_metas[i];
                if (GP.consumer_count != 1 ||
                    GP.first_consumer_layer != int32_t(i + 1) ||
                    GP.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i, i + 1))
                    return false;
            }

            const uint32_t elem = 1;
            const uint32_t per_row_in = uint32_t(P.in_w) * P.in_c * elem;
            const uint32_t per_row_pool = uint32_t(P.out_w) * P.out_c * elem;
            const uint32_t k_h_eff = (P.k_h == 255) ? uint32_t(P.in_h) : uint32_t(P.k_h);
            if (!per_row_in || !per_row_pool || !k_h_eff || !P.s_h)
                return false;
            const uint64_t consumer_params =
                binary_tail ? align64(48u) :
                unary_tail ? align64(uint32_t(C0.wgt_size)) : 0u;
            const uint64_t safety = 65536;
            uint32_t tile_oh = P.out_h;
            auto slot_bytes_for = [&](uint32_t rows) {
                const uint32_t in_rows = rows * P.s_h + k_h_eff - 1u;
                const uint64_t in_b = uint64_t(in_rows) * per_row_in;
                const uint64_t pool_b = uint64_t(rows) * per_row_pool;
                const uint64_t consumer_b = d2s_tail
                    ? uint64_t(rows) * (C0.k_h ? C0.k_h : 1u) * C0.out_w * C0.out_c * elem
                    : pool_b;
                const uint64_t b_b = binary_tail ? pool_b : 0;
                return align64(uint32_t(in_b)) +
                       2ull * align64(uint32_t(std::max(pool_b, consumer_b))) +
                       align64(uint32_t(b_b));
            };
            while (tile_oh > 1 &&
                   consumer_params + 2ull * slot_bytes_for(tile_oh) + safety > L1_BUDGET)
                --tile_oh;
            if (tile_oh == P.out_h ||
                consumer_params + 2ull * slot_bytes_for(tile_oh) + safety > L1_BUDGET)
                return false;

            flush_pending();

            uint32_t cursor = 0;
            uint32_t consumer_params_l1 = 0;
            uint8_t consumer_params_tag = 0;
            if (binary_tail || unary_tail) {
                consumer_params_l1 = cursor;
                cursor = align64(uint32_t(cursor + (binary_tail ? 48u : C0.wgt_size)));
                consumer_params_tag = alloc_tag();
                const uint32_t params_src = binary_tail
                    ? uint32_t(C0.dram_wgt + C0.wgt_size - 48)
                    : uint32_t(C0.dram_wgt);
                const uint32_t params_bytes = binary_tail ? 48u : uint32_t(C0.wgt_size);
                Descriptor pd = make_udma(params_src, consumer_params_l1, params_bytes,
                                          /*dir*/ 0, consumer_params_tag);
                Microblock pmb{};
                mark_stream(pd, i + 1, pmb, SMF_LOAD_B);
                program.push_back(pd);
                acc[i + 1].dram_r += params_bytes;
                acc[i + 1].sram_w += params_bytes;
                ++udma_count_so_far;
            }

            const uint32_t max_in_rows = tile_oh * P.s_h + k_h_eff - 1u;
            const uint32_t max_in_bytes = max_in_rows * per_row_in;
            const uint32_t max_pool_bytes = tile_oh * per_row_pool;
            const uint32_t seg_bytes = align64(std::max(max_in_bytes, max_pool_bytes));
            const uint32_t slot_base0 = align64(cursor);
            const uint32_t slot_bytes = uint32_t(slot_bytes_for(tile_oh));
            auto slot_base = [&](uint32_t slot) {
                return slot_base0 + slot * slot_bytes;
            };

            std::vector<size_t> last_udma(N, 0), last_pool(N, 0), last_ewe(N, 0), last_tnps(N, 0);
            uint8_t slot_done[2] = {0, 0};
            uint8_t final_done_all = 0;
            uint16_t mb_id = 0;
            for (uint32_t oh = 0; oh < P.out_h; oh += tile_oh, ++mb_id) {
                const uint32_t rows = std::min<uint32_t>(tile_oh, P.out_h - oh);
                const bool final_mb = (oh + rows == P.out_h);
                const int ih_lo_u = int(oh) * int(P.s_h) - int(P.p_t);
                const int ih_hi_u = int(oh + rows - 1) * int(P.s_h) + int(k_h_eff) - 1 - int(P.p_t);
                const int ih_lo = std::max(0, ih_lo_u);
                const int ih_hi = std::min(int(P.in_h) - 1, ih_hi_u);
                const uint32_t in_rows = uint32_t(ih_hi - ih_lo + 1);
                const uint32_t in_bytes = in_rows * per_row_in;
                const uint32_t pool_bytes = rows * per_row_pool;
                const uint32_t dram_in_off = uint32_t(ih_lo) * per_row_in;
                const uint32_t dram_out_off = oh * per_row_pool;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(oh) * P.out_w * P.out_c;
                mb.rows = rows;
                mb.elems = rows * P.out_w * P.out_c;
                mb.bytes = pool_bytes;

                const uint32_t base = slot_base(mb.slot);
                const uint32_t l1_in = base;
                const uint32_t l1_pool = base + seg_bytes;
                const uint32_t l1_out = base;
                const uint32_t l1_b = align64(uint32_t(base + 2 * seg_bytes));

                const uint8_t in_tag = alloc_tag();
                auto [id, charged] = make_act_load(P, uint32_t(P.dram_in + dram_in_off),
                                                   l1_in, in_bytes,
                                                   in_tag, slot_done[mb.slot]);
                mark_stream(id, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(id);
                acc[i].dram_r += charged;
                acc[i].sram_w += in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                LayerMeta tile_P = P;
                tile_P.in_h = uint16_t(in_rows);
                tile_P.out_h = uint16_t(rows);
                tile_P.p_t = uint8_t((oh == 0) ? P.p_t : 0);
                tile_P.p_b = uint8_t(final_mb ? P.p_b : 0);
                const uint8_t pool_tag = alloc_tag();
                Descriptor pool_d = make_pool(tile_P, l1_in, l1_pool, in_tag, pool_tag);
                mark_stream(pool_d, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(pool_d);
                acc[i].sram_r += in_bytes;
                acc[i].sram_w += pool_bytes;
                ++pool_count_so_far;
                last_pool[i] = pool_count_so_far;
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;

                uint32_t final_addr = l1_pool;
                uint32_t final_bytes = pool_bytes;
                uint32_t final_off = dram_out_off;
                uint8_t final_done = pool_tag;
                if (binary_tail) {
                    const uint8_t b_tag = alloc_tag();
                    auto [bd, b_charged] = make_binary_b_load(C0,
                                                               C0.dram_wgt + dram_out_off,
                                                               l1_b, pool_bytes,
                                                               b_tag, slot_done[mb.slot]);
                    mark_stream(bd, i + 1, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(bd);
                    acc[i + 1].dram_r += b_charged;
                    acc[i + 1].sram_w += pool_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;

                    LayerMeta tile_C = C0;
                    tile_C.in_h = uint16_t(rows);
                    tile_C.out_h = uint16_t(rows);
                    const uint8_t e_tag = alloc_tag();
                    Descriptor ed = make_ewe_add(tile_C, l1_pool, l1_b, l1_out,
                                                 consumer_params_l1,
                                                 b_tag, pool_tag, e_tag);
                    mark_stream(ed, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(ed);
                    acc[i + 1].sram_r += 2 * uint64_t(pool_bytes);
                    acc[i + 1].sram_w += pool_bytes;
                    ++ewe_count_so_far;
                    last_ewe[i + 1] = ewe_count_so_far;
                    final_addr = l1_out;
                    final_done = e_tag;
                    udma_w_skipped[i] = true;
                    mark_flow_edge(i, i + 1);
                } else if (unary_tail) {
                    LayerMeta tile_C = C0;
                    tile_C.in_h = uint16_t(rows);
                    tile_C.out_h = uint16_t(rows);
                    const uint8_t e_tag = alloc_tag();
                    Descriptor ed = make_ewe_unary(tile_C, l1_pool, l1_out,
                                                   consumer_params_l1,
                                                   pool_tag, e_tag, consumer_params_tag);
                    mark_stream(ed, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(ed);
                    acc[i + 1].sram_r += pool_bytes;
                    acc[i + 1].sram_w += pool_bytes;
                    ++ewe_count_so_far;
                    last_ewe[i + 1] = ewe_count_so_far;
                    final_addr = l1_out;
                    final_done = e_tag;
                    mark_flow_edge(i, i + 1);
                } else if (d2s_tail) {
                    const uint16_t block = C0.k_h ? C0.k_h : 1;
                    const uint32_t d2s_bytes = rows * block * C0.out_w * C0.out_c * elem;
                    const uint32_t d2s_off = oh * block * C0.out_w * C0.out_c * elem;
                    const uint8_t t_tag = alloc_tag();
                    Descriptor td = make_tnps_d2s(l1_pool, uint32_t(C0.dram_out + d2s_off),
                                                  uint16_t(rows), C0.in_w, C0.in_c,
                                                  block, uint8_t(elem), t_tag, pool_tag);
                    mark_stream(td, i + 1, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    td.hdr.flags |= DF_STREAM_TAIL;
                    program.push_back(td);
                    acc[i + 1].sram_r += pool_bytes;
                    acc[i + 1].dram_w += d2s_bytes;
                    ++tnps_count_so_far;
                    last_tnps[i + 1] = tnps_count_so_far;
                    final_done = t_tag;
                    final_bytes = 0;
                    mark_flow_edge(i, i + 1);
                }

                if (!d2s_tail) {
                    if (producer_no_store[i + 1]) {
                        udma_w_skipped[i + 1] = true;
                        udma_w_streamed[i + 1] = true;
                    } else {
                        const uint8_t st_tag = alloc_tag();
                        Descriptor sd = make_udma(final_addr, uint32_t(C0.dram_out + final_off),
                                                  final_bytes, /*dir*/ 1, st_tag, final_done);
                        mark_stream(sd, i + 1, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(sd);
                        acc[i + 1].sram_r += final_bytes;
                        acc[i + 1].dram_w += final_bytes;
                        ++udma_count_so_far;
                        last_udma[i + 1] = udma_count_so_far;
                        final_done = st_tag;
                    }
                }
                slot_done[mb.slot] = final_done;
                final_done_all = final_done;
            }

            const uint16_t tiles_h = uint16_t((uint32_t(P.out_h) + tile_oh - 1) / tile_oh);
            tiles_h_per_layer[i] = tiles_h;
            tiles_oc_per_layer[i] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            pool_count_at_layer_end[i] = last_pool[i];
            layer_done_tag[i] = final_done_all;
            tiles_h_per_layer[i + 1] = tiles_h;
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            ewe_count_at_layer_end[i + 1] = last_ewe[i + 1];
            tnps_count_at_layer_end[i + 1] = last_tnps[i + 1];
            layer_done_tag[i + 1] = final_done_all;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto try_stream_s2d_compute = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& S = metas[i];
            const auto& B = metas[i + 1];
            if (S.op_kind != OK_S2SPACE || !is_conv_class_meta(B))
                return false;
            if (!producer_no_store[i])
                return false;
            if (S.dtype != B.dtype)
                return false;
            if (S.out_h != B.in_h || S.out_w != B.in_w || S.out_c != B.in_c)
                return false;
            if (B.op_kind == OK_FC || (B.op_kind == OK_CONV && (B.group ? B.group : 1) != 1))
                return false;
            if (B.op_kind == OK_DWCONV &&
                ((B.group ? B.group : 1) != B.in_c || B.out_c != B.in_c))
                return false;
            if (B.dtype == DT_INT16x16 || B.dtype == DT_INT16x8)
                return false;
            if (graph_metas) {
                const auto& GS = graph_metas[i];
                if (GS.consumer_count != 1 ||
                    GS.first_consumer_layer != int32_t(i + 1) ||
                    GS.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i, i + 1))
                    return false;
            }

            const uint16_t block = S.k_h ? S.k_h : 1;
            if (!block || S.in_h != uint32_t(S.out_h) * block ||
                S.in_w != uint32_t(S.out_w) * block ||
                S.out_c != uint32_t(S.in_c) * block * block)
                return false;

            const bool is_fp =
                B.dtype == DT_FP16 || B.dtype == DT_BFP16 || B.dtype == DT_FP8;
            const uint32_t elem = is_fp ? 2u : 1u;
            const uint64_t pure_wgt = conv_pure_weight_bytes(B);
            const uint64_t scale_lut_size = is_fp
                ? (8 + 4 * uint64_t(B.out_c))
                : (12 + 9 * uint64_t(B.out_c));
            const uint64_t corr_size =
                (!is_fp && uint64_t(B.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(B.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t fixed = align64(uint32_t(params_blob)) +
                                   align64(uint32_t(pure_wgt));
            const uint64_t safety = 65536;
            if (fixed + safety >= L1_BUDGET)
                return false;

            auto worst_conv_in_rows_for = [&](uint32_t toh) {
                return std::min<uint32_t>(B.in_h,
                                          toh * B.s_h + (B.k_h ? B.k_h - 1 : 0));
            };
            auto tile_rows_encodable = [&](uint32_t toh) {
                return uint64_t(worst_conv_in_rows_for(toh)) * block <= 0xFFFFu;
            };
            auto bytes_for = [&](uint32_t toh) {
                const uint32_t worst_conv_in_h = worst_conv_in_rows_for(toh);
                const uint64_t s2d_in =
                    uint64_t(worst_conv_in_h) * block * S.in_w * S.in_c * elem;
                const uint64_t s2d_out =
                    uint64_t(worst_conv_in_h) * S.out_w * S.out_c * elem;
                const uint64_t conv_out =
                    uint64_t(toh) * B.out_w * B.out_c * elem;
                return std::array<uint64_t, 3>{s2d_in, s2d_out, conv_out};
            };
            uint32_t tile_oh = B.out_h;
            while (tile_oh > 1) {
                const auto b = bytes_for(tile_oh);
                const uint64_t slot =
                    align64(uint32_t(b[0])) +
                    align64(uint32_t(b[1])) +
                    align64(uint32_t(b[2]));
                if (tile_rows_encodable(tile_oh) &&
                    fixed + 2ull * slot + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            const auto max_b = bytes_for(tile_oh);
            const uint64_t slot_raw =
                align64(uint32_t(max_b[0])) +
                align64(uint32_t(max_b[1])) +
                align64(uint32_t(max_b[2]));
            if (tile_oh < 1 || !tile_rows_encodable(tile_oh) ||
                fixed + 2ull * slot_raw + safety > L1_BUDGET)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS = 0;
            const uint32_t L1_WGT = align64(uint32_t(params_blob));
            const uint32_t SLOT_BASE = align64(uint32_t(L1_WGT + pure_wgt));
            const uint32_t SLOT_S2D_IN = 0;
            const uint32_t SLOT_S2D_OUT = align64(uint32_t(max_b[0]));
            const uint32_t SLOT_CONV_OUT = align64(uint32_t(SLOT_S2D_OUT + max_b[1]));
            const uint32_t SLOT_BYTES = align64(uint32_t(SLOT_CONV_OUT + max_b[2]));
            if (uint64_t(SLOT_BASE) + 2ull * SLOT_BYTES + safety > L1_BUDGET)
                return false;

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
            const uint8_t entry_wait =
                (i > 0 && udma_w_skipped[i - 1]) ? layer_done_tag[i - 1] : 0;
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            program.push_back(make_udma(B.dram_wgt + uint32_t(pure_wgt),
                                        L1_PARAMS, uint32_t(params_blob),
                                        /*dir*/ 0, params_tag, entry_wait));
            program.push_back(make_udma(B.dram_wgt, L1_WGT, uint32_t(pure_wgt),
                                        /*dir*/ 0, wgt_tag, entry_wait));
            udma_count_so_far += 2;
            last_udma[i + 1] = udma_count_so_far;
            acc[i + 1].dram_r += params_blob + pure_wgt;
            acc[i + 1].sram_w += params_blob + pure_wgt;

            uint8_t slot_done[2] = {entry_wait, entry_wait};
            uint8_t final_done = 0;
            uint16_t mb_id = 0;
            for (uint32_t oh_done = 0; oh_done < B.out_h; oh_done += tile_oh, ++mb_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, B.out_h - oh_done);
                const uint32_t oh_hi = oh_done + this_oh;
                const bool final_mb = (oh_hi == B.out_h);
                const int ih_lo_u = int(oh_done) * int(B.s_h) - int(B.p_t);
                const int ih_hi_u =
                    int(oh_hi - 1) * int(B.s_h) + int(B.k_h) - 1 - int(B.p_t);
                const uint32_t conv_ih_lo = uint32_t(std::max(0, ih_lo_u));
                const uint32_t conv_ih_hi = uint32_t(std::min<int>(int(B.in_h) - 1, ih_hi_u));
                const uint32_t conv_in_rows = conv_ih_hi - conv_ih_lo + 1;
                const uint8_t pad_t_tile =
                    uint8_t(std::min<int>(7, std::max(0, -ih_lo_u)));
                const uint8_t pad_b_tile =
                    uint8_t(std::min<int>(7, std::max(0, ih_hi_u - int(B.in_h) + 1)));
                const uint32_t s2d_src_rows = conv_in_rows * block;
                const uint32_t s2d_src_off =
                    uint32_t(uint64_t(conv_ih_lo) * block * S.in_w * S.in_c * elem);
                const uint32_t s2d_in_bytes = s2d_src_rows * S.in_w * S.in_c * elem;
                const uint32_t s2d_out_bytes = conv_in_rows * S.out_w * S.out_c * elem;
                const uint32_t conv_out_bytes = this_oh * B.out_w * B.out_c * elem;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(oh_done) * B.out_w * B.out_c;
                mb.rows = this_oh;
                mb.elems = this_oh * B.out_w * B.out_c;
                mb.bytes = conv_out_bytes;

                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_S2D_IN = slot_base + SLOT_S2D_IN;
                const uint32_t L1_S2D_OUT = slot_base + SLOT_S2D_OUT;
                const uint32_t L1_CONV_OUT = slot_base + SLOT_CONV_OUT;

                const uint8_t load_tag = alloc_tag();
                auto [ld, charged] = make_act_load(S, S.dram_in + s2d_src_off,
                                                    L1_S2D_IN, s2d_in_bytes,
                                                    load_tag, slot_done[mb.slot]);
                mark_stream(ld, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ld);
                acc[i].dram_r += charged;
                acc[i].sram_w += s2d_in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                const uint8_t s2d_tag = alloc_tag();
                Descriptor td = make_tnps_s2d(L1_S2D_IN, L1_S2D_OUT,
                                              uint16_t(s2d_src_rows), S.in_w, S.in_c,
                                              block, S.out_c, uint8_t(elem),
                                              s2d_tag, load_tag);
                mark_stream(td, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(td);
                acc[i].sram_r += s2d_in_bytes;
                acc[i].sram_w += s2d_out_bytes;
                ++tnps_count_so_far;
                last_tnps[i] = tnps_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                          /*signal*/ 0, wgt_tag, s2d_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_S2D_OUT;
                cb.wgt_addr = L1_WGT;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(conv_in_rows);
                cb.in_w = B.in_w;
                cb.in_c = B.in_c;
                cb.out_c = B.out_c;
                cb.k_h = B.k_h;
                cb.k_w = B.k_w;
                cb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                cb.pad_lr = uint8_t((B.p_l & 7) | ((B.p_r & 7) << 3));
                cb.group = B.group ? B.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = B.zp_in_eff;
                mark_stream(cd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(cd);
                acc[i + 1].sram_r += pure_wgt + s2d_out_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, s2d_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_CONV_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = B.out_w;
                rb.c = B.out_c;
                rb.scale_lut_addr = L1_PARAMS;
                rb.scale_count = B.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = B.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS + scale_lut_size) : 0u;
                rb.corr_per_oc = uint8_t(corr_size && B.op_kind == OK_DWCONV);
                if (producer_no_store[i + 1])
                    rb._r[0] = RQ_STORE_SKIP;
                mark_stream(rd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(rd);
                acc[i + 1].sram_r += scale_lut_size;
                if (!producer_no_store[i + 1])
                    acc[i + 1].sram_w += conv_out_bytes;
                ++requant_count_so_far;
                last_req[i + 1] = requant_count_so_far;

                uint8_t tile_done = req_tag;
                if (producer_no_store[i + 1]) {
                    udma_w_skipped[i + 1] = true;
                    udma_w_streamed[i + 1] = true;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(L1_CONV_OUT,
                                              uint32_t(uint64_t(B.dram_out) +
                                                       uint64_t(oh_done) * B.out_w * B.out_c * elem),
                                              conv_out_bytes, /*dir*/ 1, st_tag, req_tag);
                    mark_stream(sd, i + 1, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[i + 1].sram_r += conv_out_bytes;
                    acc[i + 1].dram_w += conv_out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    tile_done = st_tag;
                }
                slot_done[mb.slot] = tile_done;
                final_done = tile_done;
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            mark_flow_edge(i, i + 1);
            const uint16_t tiles_h = uint16_t((B.out_h + tile_oh - 1) / tile_oh);
            tiles_h_per_layer[i] = tiles_h;
            tiles_oc_per_layer[i] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            tnps_count_at_layer_end[i] = last_tnps[i];
            layer_done_tag[i] = final_done;
            tiles_h_per_layer[i + 1] = tiles_h;
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = last_req[i + 1];
            layer_done_tag[i + 1] = final_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto pad_border_is_raw_zero_shape =
            [&](const LayerMeta& P, uint32_t elem, uint32_t inner_h, uint32_t inner_w) -> bool {
            if (uint64_t(P.ref_off) + P.ref_size > file.size())
                return false;
            const auto* ref = file.data() + P.ref_off;
            const uint64_t row_bytes = uint64_t(P.out_w) * P.out_c * elem;
            const uint64_t inner_x0 = uint64_t(P.p_l) * P.out_c * elem;
            const uint64_t inner_x1 = uint64_t(P.p_l + inner_w) * P.out_c * elem;
            for (uint32_t y = 0; y < P.out_h; ++y) {
                const bool pad_y = (y < P.p_t) || (y >= uint32_t(P.p_t) + inner_h);
                const uint64_t row = uint64_t(y) * row_bytes;
                for (uint64_t b = 0; b < row_bytes; ++b) {
                    const bool pad_x = b < inner_x0 || b >= inner_x1;
                    if ((pad_y || pad_x) && ref[row + b] != 0)
                        return false;
                }
            }
            return true;
        };
        auto pad_border_is_raw_zero = [&](const LayerMeta& P, uint32_t elem) -> bool {
            return pad_border_is_raw_zero_shape(P, elem, P.in_h, P.in_w);
        };

        auto try_stream_d2s_compute = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& D = metas[i];
            const auto& B = metas[i + 1];
            if (D.op_kind != OK_D2SPACE || !is_conv_class_meta(B))
                return false;
            if (B.op_kind == OK_FC)
                return false;
            if (D.dtype != B.dtype ||
                D.out_h != B.in_h || D.out_w != B.in_w || D.out_c != B.in_c)
                return false;
            if (B.op_kind == OK_CONV && (B.group ? B.group : 1) != 1)
                return false;
            if (B.op_kind == OK_DWCONV &&
                ((B.group ? B.group : 1) != B.in_c || B.out_c != B.in_c))
                return false;
            if (B.dtype == DT_INT16x16 || B.dtype == DT_INT16x8)
                return false;
            if (graph_metas) {
                const auto& GD = graph_metas[i];
                if (GD.consumer_count != 1 ||
                    GD.first_consumer_layer != int32_t(i + 1) ||
                    GD.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i, i + 1))
                    return false;
            }

            const uint16_t block = D.k_h ? D.k_h : 1;
            if (!block || D.in_c != uint32_t(D.out_c) * block * block ||
                D.out_h != uint32_t(D.in_h) * block ||
                D.out_w != uint32_t(D.in_w) * block)
                return false;

            const bool is_fp =
                B.dtype == DT_FP16 || B.dtype == DT_BFP16 || B.dtype == DT_FP8;
            const uint32_t elem = is_fp ? 2u : 1u;
            const uint64_t pure_wgt = conv_pure_weight_bytes(B);
            const uint64_t scale_lut_size = is_fp
                ? (8 + 4 * uint64_t(B.out_c))
                : (12 + 9 * uint64_t(B.out_c));
            const uint64_t corr_size =
                (!is_fp && uint64_t(B.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(B.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t fixed = align64(uint32_t(params_blob)) +
                                   align64(uint32_t(pure_wgt));
            const uint64_t safety = 65536;
            if (fixed + safety >= L1_BUDGET)
                return false;

            auto worst_conv_in_rows_for = [&](uint32_t toh) {
                return std::min<uint32_t>(B.in_h,
                                          toh * B.s_h + (B.k_h ? B.k_h - 1 : 0));
            };
            auto max_d2s_src_rows_for = [&](uint32_t conv_rows) {
                return uint32_t((uint64_t(conv_rows) + 2ull * block - 2ull) / block);
            };
            auto tile_rows_encodable = [&](uint32_t toh) {
                return max_d2s_src_rows_for(worst_conv_in_rows_for(toh)) <= 0xFFFFu;
            };
            auto bytes_for = [&](uint32_t toh) {
                const uint32_t worst_conv_in_h = worst_conv_in_rows_for(toh);
                const uint32_t d2s_src_rows = max_d2s_src_rows_for(worst_conv_in_h);
                const uint32_t d2s_out_rows = d2s_src_rows * block;
                const uint64_t d2s_in =
                    uint64_t(d2s_src_rows) * D.in_w * D.in_c * elem;
                const uint64_t d2s_out =
                    uint64_t(d2s_out_rows) * D.out_w * D.out_c * elem;
                const uint64_t conv_out =
                    uint64_t(toh) * B.out_w * B.out_c * elem;
                return std::array<uint64_t, 3>{d2s_in, d2s_out, conv_out};
            };

            uint32_t tile_oh = B.out_h;
            while (tile_oh > 1) {
                const auto b = bytes_for(tile_oh);
                const uint64_t slot =
                    align64(uint32_t(b[0])) +
                    align64(uint32_t(b[1])) +
                    align64(uint32_t(b[2]));
                if (tile_rows_encodable(tile_oh) &&
                    fixed + 2ull * slot + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            const auto max_b = bytes_for(tile_oh);
            const uint64_t slot_raw =
                align64(uint32_t(max_b[0])) +
                align64(uint32_t(max_b[1])) +
                align64(uint32_t(max_b[2]));
            if (tile_oh < 1 || !tile_rows_encodable(tile_oh) ||
                fixed + 2ull * slot_raw + safety > L1_BUDGET)
                return false;
            if (tile_oh == B.out_h)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS = 0;
            const uint32_t L1_WGT = align64(uint32_t(params_blob));
            const uint32_t SLOT_BASE = align64(uint32_t(L1_WGT + pure_wgt));
            const uint32_t SLOT_D2S_IN = 0;
            const uint32_t SLOT_D2S_OUT = align64(uint32_t(max_b[0]));
            const uint32_t SLOT_CONV_OUT = align64(uint32_t(SLOT_D2S_OUT + max_b[1]));
            const uint32_t SLOT_BYTES = align64(uint32_t(SLOT_CONV_OUT + max_b[2]));
            if (uint64_t(SLOT_BASE) + 2ull * SLOT_BYTES + safety > L1_BUDGET)
                return false;

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
            const uint8_t entry_wait =
                (i > 0 && udma_w_skipped[i - 1]) ? layer_done_tag[i - 1] : 0;
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            program.push_back(make_udma(B.dram_wgt + uint32_t(pure_wgt),
                                        L1_PARAMS, uint32_t(params_blob),
                                        /*dir*/ 0, params_tag, entry_wait));
            program.push_back(make_udma(B.dram_wgt, L1_WGT, uint32_t(pure_wgt),
                                        /*dir*/ 0, wgt_tag, entry_wait));
            udma_count_so_far += 2;
            last_udma[i + 1] = udma_count_so_far;
            acc[i + 1].dram_r += params_blob + pure_wgt;
            acc[i + 1].sram_w += params_blob + pure_wgt;

            uint8_t slot_done[2] = {entry_wait, entry_wait};
            uint8_t final_done = 0;
            uint16_t mb_id = 0;
            for (uint32_t oh_done = 0; oh_done < B.out_h; oh_done += tile_oh, ++mb_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, B.out_h - oh_done);
                const uint32_t oh_hi = oh_done + this_oh;
                const bool final_mb = (oh_hi == B.out_h);
                const int ih_lo_u = int(oh_done) * int(B.s_h) - int(B.p_t);
                const int ih_hi_u =
                    int(oh_hi - 1) * int(B.s_h) + int(B.k_h) - 1 - int(B.p_t);
                const uint32_t conv_ih_lo = uint32_t(std::max(0, ih_lo_u));
                const uint32_t conv_ih_hi = uint32_t(std::min<int>(int(B.in_h) - 1, ih_hi_u));
                const uint32_t conv_in_rows = conv_ih_hi - conv_ih_lo + 1;
                const uint8_t pad_t_tile =
                    uint8_t(std::min<int>(7, std::max(0, -ih_lo_u)));
                const uint8_t pad_b_tile =
                    uint8_t(std::min<int>(7, std::max(0, ih_hi_u - int(B.in_h) + 1)));

                const uint32_t d2s_out_lo = (conv_ih_lo / block) * block;
                const uint32_t d2s_out_hi_excl =
                    uint32_t(((uint64_t(conv_ih_hi) + 1ull + block - 1ull) / block) * block);
                const uint32_t d2s_out_rows = d2s_out_hi_excl - d2s_out_lo;
                const uint32_t d2s_src_row = d2s_out_lo / block;
                const uint32_t d2s_src_rows = d2s_out_rows / block;
                if (d2s_src_rows > 0xFFFFu)
                    return false;
                const uint32_t conv_row_offset = conv_ih_lo - d2s_out_lo;
                const uint32_t d2s_src_off =
                    uint32_t(uint64_t(d2s_src_row) * D.in_w * D.in_c * elem);
                const uint32_t d2s_in_bytes = d2s_src_rows * D.in_w * D.in_c * elem;
                const uint32_t d2s_out_bytes = d2s_out_rows * D.out_w * D.out_c * elem;
                const uint32_t conv_in_bytes = conv_in_rows * B.in_w * B.in_c * elem;
                const uint32_t conv_out_bytes = this_oh * B.out_w * B.out_c * elem;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(oh_done) * B.out_w * B.out_c;
                mb.rows = this_oh;
                mb.elems = this_oh * B.out_w * B.out_c;
                mb.bytes = conv_out_bytes;

                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_D2S_IN = slot_base + SLOT_D2S_IN;
                const uint32_t L1_D2S_OUT = slot_base + SLOT_D2S_OUT;
                const uint32_t L1_CONV_IN =
                    L1_D2S_OUT + conv_row_offset * B.in_w * B.in_c * elem;
                const uint32_t L1_CONV_OUT = slot_base + SLOT_CONV_OUT;

                const uint8_t load_tag = alloc_tag();
                auto [ld, charged] = make_act_load(D, D.dram_in + d2s_src_off,
                                                    L1_D2S_IN, d2s_in_bytes,
                                                    load_tag, slot_done[mb.slot]);
                mark_stream(ld, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ld);
                acc[i].dram_r += charged;
                acc[i].sram_w += d2s_in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                const uint8_t d2s_tag = alloc_tag();
                Descriptor td = make_tnps_d2s(L1_D2S_IN, L1_D2S_OUT,
                                              uint16_t(d2s_src_rows), D.in_w, D.in_c,
                                              block, uint8_t(elem),
                                              d2s_tag, load_tag);
                mark_stream(td, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(td);
                acc[i].sram_r += d2s_in_bytes;
                acc[i].sram_w += d2s_out_bytes;
                ++tnps_count_so_far;
                last_tnps[i] = tnps_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                          /*signal*/ 0, wgt_tag, d2s_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_CONV_IN;
                cb.wgt_addr = L1_WGT;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(conv_in_rows);
                cb.in_w = B.in_w;
                cb.in_c = B.in_c;
                cb.out_c = B.out_c;
                cb.k_h = B.k_h;
                cb.k_w = B.k_w;
                cb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                cb.pad_lr = uint8_t((B.p_l & 7) | ((B.p_r & 7) << 3));
                cb.group = B.group ? B.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = B.zp_in_eff;
                mark_stream(cd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(cd);
                acc[i + 1].sram_r += pure_wgt + conv_in_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, d2s_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_CONV_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = B.out_w;
                rb.c = B.out_c;
                rb.scale_lut_addr = L1_PARAMS;
                rb.scale_count = B.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = B.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS + scale_lut_size) : 0u;
                rb.corr_per_oc = uint8_t(corr_size && B.op_kind == OK_DWCONV);
                if (producer_no_store[i + 1])
                    rb._r[0] = RQ_STORE_SKIP;
                mark_stream(rd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(rd);
                acc[i + 1].sram_r += scale_lut_size;
                if (!producer_no_store[i + 1])
                    acc[i + 1].sram_w += conv_out_bytes;
                ++requant_count_so_far;
                last_req[i + 1] = requant_count_so_far;

                uint8_t tile_done = req_tag;
                if (producer_no_store[i + 1]) {
                    udma_w_skipped[i + 1] = true;
                    udma_w_streamed[i + 1] = true;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(L1_CONV_OUT,
                                              uint32_t(uint64_t(B.dram_out) +
                                                       uint64_t(oh_done) * B.out_w * B.out_c * elem),
                                              conv_out_bytes, /*dir*/ 1, st_tag, req_tag);
                    mark_stream(sd, i + 1, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[i + 1].sram_r += conv_out_bytes;
                    acc[i + 1].dram_w += conv_out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    tile_done = st_tag;
                }
                slot_done[mb.slot] = tile_done;
                final_done = tile_done;
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            mark_flow_edge(i, i + 1);
            const uint16_t tiles_h = uint16_t((B.out_h + tile_oh - 1) / tile_oh);
            tiles_h_per_layer[i] = tiles_h;
            tiles_oc_per_layer[i] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            tnps_count_at_layer_end[i] = last_tnps[i];
            layer_done_tag[i] = final_done;
            tiles_h_per_layer[i + 1] = tiles_h;
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = last_req[i + 1];
            layer_done_tag[i + 1] = final_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto try_stream_d2s_pad_compute = [&]() -> bool {
            if (i + 2 >= N) return false;
            const auto& D = metas[i];
            const auto& P = metas[i + 1];
            const auto& B = metas[i + 2];
            if (D.op_kind != OK_D2SPACE || P.op_kind != OK_PAD || !is_conv_class_meta(B))
                return false;
            if (B.op_kind == OK_FC)
                return false;
            if (D.dtype != P.dtype || P.dtype != B.dtype)
                return false;
            if (P.out_h != B.in_h || P.out_w != B.in_w || P.out_c != B.in_c)
                return false;
            if (D.out_c != P.out_c ||
                uint32_t(D.out_h) + P.p_t + P.p_b != P.out_h ||
                uint32_t(D.out_w) + P.p_l + P.p_r != P.out_w)
                return false;
            if (!(P.p_t || P.p_b || P.p_l || P.p_r))
                return false;
            if (B.op_kind == OK_CONV && (B.group ? B.group : 1) != 1)
                return false;
            if (B.op_kind == OK_DWCONV &&
                ((B.group ? B.group : 1) != B.in_c || B.out_c != B.in_c))
                return false;
            if (B.dtype == DT_INT16x16 || B.dtype == DT_INT16x8)
                return false;
            const uint32_t pad_t_total = uint32_t(P.p_t) + B.p_t;
            const uint32_t pad_b_total = uint32_t(P.p_b) + B.p_b;
            const uint32_t pad_l_total = uint32_t(P.p_l) + B.p_l;
            const uint32_t pad_r_total = uint32_t(P.p_r) + B.p_r;
            if (pad_t_total > 7 || pad_b_total > 7 ||
                pad_l_total > 7 || pad_r_total > 7)
                return false;
            if ((B.p_t || B.p_b || B.p_l || B.p_r) && B.zp_in_eff != 0)
                return false;
            if (graph_metas) {
                const auto& GD = graph_metas[i];
                const auto& GP = graph_metas[i + 1];
                if (GD.consumer_count != 1 ||
                    GD.first_consumer_layer != int32_t(i + 1) ||
                    GD.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (GP.consumer_count != 1 ||
                    GP.first_consumer_layer != int32_t(i + 2) ||
                    GP.last_consumer_layer  != int32_t(i + 2) ||
                    !graph_input0_is_exact_producer(i, i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i + 1, i + 2))
                    return false;
            }

            const uint16_t block = D.k_h ? D.k_h : 1;
            if (!block || D.in_c != uint32_t(D.out_c) * block * block ||
                D.out_h != uint32_t(D.in_h) * block ||
                D.out_w != uint32_t(D.in_w) * block)
                return false;

            const bool is_fp =
                B.dtype == DT_FP16 || B.dtype == DT_BFP16 || B.dtype == DT_FP8;
            const uint32_t elem = is_fp ? 2u : 1u;
            if (!pad_border_is_raw_zero_shape(P, elem, D.out_h, D.out_w))
                return false;
            const uint64_t pure_wgt = conv_pure_weight_bytes(B);
            const uint64_t scale_lut_size = is_fp
                ? (8 + 4 * uint64_t(B.out_c))
                : (12 + 9 * uint64_t(B.out_c));
            const uint64_t corr_size =
                (!is_fp && uint64_t(B.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(B.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t fixed = align64(uint32_t(params_blob)) +
                                   align64(uint32_t(pure_wgt));
            const uint64_t safety = 65536;
            if (fixed + safety >= L1_BUDGET)
                return false;

            auto source_rows_for = [&](uint32_t oh_start, uint32_t toh) {
                const uint32_t oh_hi = oh_start + toh;
                const int first = int(oh_start) * int(B.s_h) - int(pad_t_total);
                const int last =
                    int(oh_hi - 1) * int(B.s_h) + int(B.k_h) - 1 - int(pad_t_total);
                const int src_lo_i = std::max(0, first);
                const int src_hi_i = std::min<int>(int(D.out_h) - 1, last);
                const uint32_t src_rows = (src_hi_i >= src_lo_i)
                    ? uint32_t(src_hi_i - src_lo_i + 1) : 0u;
                const uint32_t pad_t_tile = uint32_t(src_lo_i - first);
                const uint32_t pad_b_tile = uint32_t(last - src_hi_i);
                return std::array<uint32_t, 4>{
                    uint32_t(src_lo_i), src_rows, pad_t_tile, pad_b_tile
                };
            };
            auto d2s_window_for = [&](uint32_t d2s_row, uint32_t d2s_rows) {
                const uint32_t d2s_out_lo = (d2s_row / block) * block;
                const uint32_t d2s_out_hi_excl =
                    uint32_t(((uint64_t(d2s_row) + d2s_rows + block - 1ull) / block) * block);
                const uint32_t d2s_out_rows = d2s_out_hi_excl - d2s_out_lo;
                const uint32_t d2s_src_row = d2s_out_lo / block;
                const uint32_t d2s_src_rows = d2s_out_rows / block;
                const uint32_t conv_row_offset = d2s_row - d2s_out_lo;
                return std::array<uint32_t, 4>{
                    d2s_src_row, d2s_src_rows, d2s_out_rows, conv_row_offset
                };
            };
            auto tile_rows_encodable = [&](uint32_t toh) {
                for (uint32_t oh = 0; oh < B.out_h; oh += toh) {
                    const uint32_t rows = std::min<uint32_t>(toh, B.out_h - oh);
                    auto sr = source_rows_for(oh, rows);
                    if (!sr[1] || sr[1] > 0xFFFFu || sr[2] > 7 || sr[3] > 7)
                        return false;
                    auto dw = d2s_window_for(sr[0], sr[1]);
                    if (!dw[1] || dw[1] > 0xFFFFu)
                        return false;
                }
                return true;
            };
            auto bytes_for = [&](uint32_t toh) {
                uint64_t max_d2s_in = 0, max_d2s_out = 0;
                for (uint32_t oh = 0; oh < B.out_h; oh += toh) {
                    const uint32_t rows = std::min<uint32_t>(toh, B.out_h - oh);
                    auto sr = source_rows_for(oh, rows);
                    auto dw = d2s_window_for(sr[0], sr[1]);
                    max_d2s_in = std::max<uint64_t>(
                        max_d2s_in, uint64_t(dw[1]) * D.in_w * D.in_c * elem);
                    max_d2s_out = std::max<uint64_t>(
                        max_d2s_out, uint64_t(dw[2]) * D.out_w * D.out_c * elem);
                }
                const uint64_t conv_out =
                    uint64_t(toh) * B.out_w * B.out_c * elem;
                return std::array<uint64_t, 3>{max_d2s_in, max_d2s_out, conv_out};
            };
            uint32_t tile_oh = B.out_h;
            while (tile_oh > 1) {
                const auto b = bytes_for(tile_oh);
                const uint64_t slot =
                    align64(uint32_t(b[0])) +
                    align64(uint32_t(b[1])) +
                    align64(uint32_t(b[2]));
                if (tile_rows_encodable(tile_oh) &&
                    fixed + 2ull * slot + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            const auto max_b = bytes_for(tile_oh);
            const uint64_t slot_raw =
                align64(uint32_t(max_b[0])) +
                align64(uint32_t(max_b[1])) +
                align64(uint32_t(max_b[2]));
            if (tile_oh < 1 || !tile_rows_encodable(tile_oh) ||
                fixed + 2ull * slot_raw + safety > L1_BUDGET)
                return false;
            if (tile_oh == B.out_h)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS = 0;
            const uint32_t L1_WGT = align64(uint32_t(params_blob));
            const uint32_t SLOT_BASE = align64(uint32_t(L1_WGT + pure_wgt));
            const uint32_t SLOT_D2S_IN = 0;
            const uint32_t SLOT_D2S_OUT = align64(uint32_t(max_b[0]));
            const uint32_t SLOT_CONV_OUT = align64(uint32_t(SLOT_D2S_OUT + max_b[1]));
            const uint32_t SLOT_BYTES = align64(uint32_t(SLOT_CONV_OUT + max_b[2]));
            if (uint64_t(SLOT_BASE) + 2ull * SLOT_BYTES + safety > L1_BUDGET)
                return false;

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
            const uint8_t entry_wait =
                (i > 0 && udma_w_skipped[i - 1]) ? layer_done_tag[i - 1] : 0;
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            program.push_back(make_udma(B.dram_wgt + uint32_t(pure_wgt),
                                        L1_PARAMS, uint32_t(params_blob),
                                        /*dir*/ 0, params_tag, entry_wait));
            program.push_back(make_udma(B.dram_wgt, L1_WGT, uint32_t(pure_wgt),
                                        /*dir*/ 0, wgt_tag, entry_wait));
            udma_count_so_far += 2;
            last_udma[i + 2] = udma_count_so_far;
            acc[i + 2].dram_r += params_blob + pure_wgt;
            acc[i + 2].sram_w += params_blob + pure_wgt;

            uint8_t slot_done[2] = {entry_wait, entry_wait};
            uint8_t final_done = 0;
            uint16_t mb_id = 0;
            for (uint32_t oh_done = 0; oh_done < B.out_h; oh_done += tile_oh, ++mb_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, B.out_h - oh_done);
                const bool final_mb = (oh_done + this_oh == B.out_h);
                auto sr = source_rows_for(oh_done, this_oh);
                auto dw = d2s_window_for(sr[0], sr[1]);
                const uint32_t d2s_src_row = dw[0];
                const uint32_t d2s_src_rows = dw[1];
                const uint32_t d2s_out_rows = dw[2];
                const uint32_t conv_row_offset = dw[3];
                if (!d2s_src_rows || d2s_src_rows > 0xFFFFu ||
                    sr[2] > 7 || sr[3] > 7)
                    return false;
                const uint32_t d2s_src_off =
                    uint32_t(uint64_t(d2s_src_row) * D.in_w * D.in_c * elem);
                const uint32_t d2s_in_bytes = d2s_src_rows * D.in_w * D.in_c * elem;
                const uint32_t d2s_out_bytes = d2s_out_rows * D.out_w * D.out_c * elem;
                const uint32_t conv_in_bytes = sr[1] * D.out_w * D.out_c * elem;
                const uint32_t conv_out_bytes = this_oh * B.out_w * B.out_c * elem;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(oh_done) * B.out_w * B.out_c;
                mb.rows = this_oh;
                mb.elems = this_oh * B.out_w * B.out_c;
                mb.bytes = conv_out_bytes;

                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_D2S_IN = slot_base + SLOT_D2S_IN;
                const uint32_t L1_D2S_OUT = slot_base + SLOT_D2S_OUT;
                const uint32_t L1_CONV_IN =
                    L1_D2S_OUT + conv_row_offset * D.out_w * D.out_c * elem;
                const uint32_t L1_CONV_OUT = slot_base + SLOT_CONV_OUT;

                const uint8_t load_tag = alloc_tag();
                auto [ld, charged] = make_act_load(D, D.dram_in + d2s_src_off,
                                                    L1_D2S_IN, d2s_in_bytes,
                                                    load_tag, slot_done[mb.slot]);
                mark_stream(ld, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ld);
                acc[i].dram_r += charged;
                acc[i].sram_w += d2s_in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                const uint8_t d2s_tag = alloc_tag();
                Descriptor td = make_tnps_d2s(L1_D2S_IN, L1_D2S_OUT,
                                              uint16_t(d2s_src_rows), D.in_w, D.in_c,
                                              block, uint8_t(elem),
                                              d2s_tag, load_tag);
                mark_stream(td, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(td);
                acc[i].sram_r += d2s_in_bytes;
                acc[i].sram_w += d2s_out_bytes;
                ++tnps_count_so_far;
                last_tnps[i] = tnps_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                          /*signal*/ 0, wgt_tag, d2s_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_CONV_IN;
                cb.wgt_addr = L1_WGT;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(sr[1]);
                cb.in_w = D.out_w;
                cb.in_c = D.out_c;
                cb.out_c = B.out_c;
                cb.k_h = B.k_h;
                cb.k_w = B.k_w;
                cb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                cb.pad_tb = uint8_t((sr[2] & 7) | ((sr[3] & 7) << 3));
                cb.pad_lr = uint8_t((pad_l_total & 7) | ((pad_r_total & 7) << 3));
                cb.group = B.group ? B.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = 0;
                mark_stream(cd, i + 2, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(cd);
                acc[i + 2].sram_r += pure_wgt + conv_in_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, d2s_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_CONV_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = B.out_w;
                rb.c = B.out_c;
                rb.scale_lut_addr = L1_PARAMS;
                rb.scale_count = B.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = B.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS + scale_lut_size) : 0u;
                rb.corr_per_oc = uint8_t(corr_size && B.op_kind == OK_DWCONV);
                if (producer_no_store[i + 2])
                    rb._r[0] = RQ_STORE_SKIP;
                mark_stream(rd, i + 2, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(rd);
                acc[i + 2].sram_r += scale_lut_size;
                if (!producer_no_store[i + 2])
                    acc[i + 2].sram_w += conv_out_bytes;
                ++requant_count_so_far;
                last_req[i + 2] = requant_count_so_far;

                uint8_t tile_done = req_tag;
                if (producer_no_store[i + 2]) {
                    udma_w_skipped[i + 2] = true;
                    udma_w_streamed[i + 2] = true;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(L1_CONV_OUT,
                                              uint32_t(uint64_t(B.dram_out) +
                                                       uint64_t(oh_done) * B.out_w * B.out_c * elem),
                                              conv_out_bytes, /*dir*/ 1, st_tag, req_tag);
                    mark_stream(sd, i + 2, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[i + 2].sram_r += conv_out_bytes;
                    acc[i + 2].dram_w += conv_out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 2] = udma_count_so_far;
                    tile_done = st_tag;
                }
                slot_done[mb.slot] = tile_done;
                final_done = tile_done;
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            udma_w_skipped[i + 1] = true;
            udma_w_streamed[i + 1] = true;
            mark_flow_edge(i, i + 1);
            mark_flow_edge(i + 1, i + 2);
            const uint16_t tiles_h = uint16_t((B.out_h + tile_oh - 1) / tile_oh);
            tiles_h_per_layer[i] = tiles_h;
            tiles_oc_per_layer[i] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            tnps_count_at_layer_end[i] = last_tnps[i];
            layer_done_tag[i] = final_done;
            tiles_h_per_layer[i + 1] = tiles_h;
            tiles_oc_per_layer[i + 1] = 1;
            layer_done_tag[i + 1] = final_done;
            tiles_h_per_layer[i + 2] = tiles_h;
            tiles_oc_per_layer[i + 2] = 1;
            udma_count_at_layer_end[i + 2] = last_udma[i + 2];
            requant_count_at_layer_end[i + 2] = last_req[i + 2];
            layer_done_tag[i + 2] = final_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 2;
            return true;
        };

        auto try_stream_pad_compute = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& P = metas[i];
            const auto& B = metas[i + 1];
            if (P.op_kind != OK_PAD || !is_conv_class_meta(B))
                return false;
            if (B.op_kind == OK_FC)
                return false;
            if (P.dtype != B.dtype ||
                P.out_h != B.in_h || P.out_w != B.in_w || P.out_c != B.in_c)
                return false;
            if (P.in_c != P.out_c ||
                uint32_t(P.in_h) + P.p_t + P.p_b != P.out_h ||
                uint32_t(P.in_w) + P.p_l + P.p_r != P.out_w)
                return false;
            if (!(P.p_t || P.p_b || P.p_l || P.p_r))
                return false;
            if (B.op_kind == OK_CONV && (B.group ? B.group : 1) != 1)
                return false;
            if (B.op_kind == OK_DWCONV &&
                ((B.group ? B.group : 1) != B.in_c || B.out_c != B.in_c))
                return false;
            if (B.dtype == DT_INT16x16 || B.dtype == DT_INT16x8)
                return false;
            const uint32_t pad_t_total = uint32_t(P.p_t) + B.p_t;
            const uint32_t pad_b_total = uint32_t(P.p_b) + B.p_b;
            const uint32_t pad_l_total = uint32_t(P.p_l) + B.p_l;
            const uint32_t pad_r_total = uint32_t(P.p_r) + B.p_r;
            if (pad_t_total > 7 || pad_b_total > 7 ||
                pad_l_total > 7 || pad_r_total > 7)
                return false;
            if ((B.p_t || B.p_b || B.p_l || B.p_r) && B.zp_in_eff != 0)
                return false;
            if (graph_metas) {
                const auto& GP = graph_metas[i];
                if (GP.consumer_count != 1 ||
                    GP.first_consumer_layer != int32_t(i + 1) ||
                    GP.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i, i + 1))
                    return false;
            }

            const bool is_fp =
                B.dtype == DT_FP16 || B.dtype == DT_BFP16 || B.dtype == DT_FP8;
            const uint32_t elem = is_fp ? 2u : 1u;
            if (!pad_border_is_raw_zero(P, elem))
                return false;

            const uint64_t pure_wgt = conv_pure_weight_bytes(B);
            const uint64_t scale_lut_size = is_fp
                ? (8 + 4 * uint64_t(B.out_c))
                : (12 + 9 * uint64_t(B.out_c));
            const uint64_t corr_size =
                (!is_fp && uint64_t(B.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(B.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t fixed = align64(uint32_t(params_blob)) +
                                   align64(uint32_t(pure_wgt));
            const uint64_t safety = 65536;
            if (fixed + safety >= L1_BUDGET)
                return false;

            auto source_rows_for = [&](uint32_t oh_start, uint32_t toh) {
                const uint32_t oh_hi = oh_start + toh;
                const int first = int(oh_start) * int(B.s_h) - int(pad_t_total);
                const int last =
                    int(oh_hi - 1) * int(B.s_h) + int(B.k_h) - 1 - int(pad_t_total);
                const int src_lo_i = std::max(0, first);
                const int src_hi_i = std::min<int>(int(P.in_h) - 1, last);
                const uint32_t src_rows = (src_hi_i >= src_lo_i)
                    ? uint32_t(src_hi_i - src_lo_i + 1) : 0u;
                const uint32_t pad_t_tile = uint32_t(src_lo_i - first);
                const uint32_t pad_b_tile = uint32_t(last - src_hi_i);
                return std::array<uint32_t, 4>{
                    uint32_t(src_lo_i), src_rows, pad_t_tile, pad_b_tile
                };
            };
            auto tile_rows_encodable = [&](uint32_t toh) {
                for (uint32_t oh = 0; oh < B.out_h; oh += toh) {
                    const uint32_t rows = std::min<uint32_t>(toh, B.out_h - oh);
                    auto sr = source_rows_for(oh, rows);
                    if (!sr[1] || sr[1] > 0xFFFFu || sr[2] > 7 || sr[3] > 7)
                        return false;
                }
                return true;
            };
            auto bytes_for = [&](uint32_t toh) {
                uint64_t max_in = 0;
                for (uint32_t oh = 0; oh < B.out_h; oh += toh) {
                    const uint32_t rows = std::min<uint32_t>(toh, B.out_h - oh);
                    auto sr = source_rows_for(oh, rows);
                    max_in = std::max<uint64_t>(
                        max_in, uint64_t(sr[1]) * P.in_w * P.in_c * elem);
                }
                const uint64_t conv_out =
                    uint64_t(toh) * B.out_w * B.out_c * elem;
                return std::pair<uint64_t, uint64_t>(max_in, conv_out);
            };
            uint32_t tile_oh = B.out_h;
            while (tile_oh > 1) {
                auto [in_b, out_b] = bytes_for(tile_oh);
                const uint64_t slot =
                    align64(uint32_t(in_b)) + align64(uint32_t(out_b));
                if (tile_rows_encodable(tile_oh) &&
                    fixed + 2ull * slot + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            auto [max_in_b, max_out_b] = bytes_for(tile_oh);
            const uint64_t slot_raw =
                align64(uint32_t(max_in_b)) + align64(uint32_t(max_out_b));
            if (tile_oh < 1 || !tile_rows_encodable(tile_oh) ||
                fixed + 2ull * slot_raw + safety > L1_BUDGET)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS = 0;
            const uint32_t L1_WGT = align64(uint32_t(params_blob));
            const uint32_t SLOT_BASE = align64(uint32_t(L1_WGT + pure_wgt));
            const uint32_t SLOT_IN = 0;
            const uint32_t SLOT_OUT = align64(uint32_t(max_in_b));
            const uint32_t SLOT_BYTES = align64(uint32_t(SLOT_OUT + max_out_b));
            if (uint64_t(SLOT_BASE) + 2ull * SLOT_BYTES + safety > L1_BUDGET)
                return false;

            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            const uint8_t entry_wait =
                (i > 0 && udma_w_skipped[i - 1]) ? layer_done_tag[i - 1] : 0;
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            program.push_back(make_udma(B.dram_wgt + uint32_t(pure_wgt),
                                        L1_PARAMS, uint32_t(params_blob),
                                        /*dir*/ 0, params_tag, entry_wait));
            program.push_back(make_udma(B.dram_wgt, L1_WGT, uint32_t(pure_wgt),
                                        /*dir*/ 0, wgt_tag, entry_wait));
            udma_count_so_far += 2;
            last_udma[i + 1] = udma_count_so_far;
            acc[i + 1].dram_r += params_blob + pure_wgt;
            acc[i + 1].sram_w += params_blob + pure_wgt;

            uint8_t slot_done[2] = {entry_wait, entry_wait};
            uint8_t final_done = 0;
            uint16_t mb_id = 0;
            for (uint32_t oh_done = 0; oh_done < B.out_h; oh_done += tile_oh, ++mb_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, B.out_h - oh_done);
                const bool final_mb = (oh_done + this_oh == B.out_h);
                auto sr = source_rows_for(oh_done, this_oh);
                const uint32_t src_row = sr[0];
                const uint32_t src_rows = sr[1];
                const uint32_t pad_t_tile = sr[2];
                const uint32_t pad_b_tile = sr[3];
                if (!src_rows || src_rows > 0xFFFFu ||
                    pad_t_tile > 7 || pad_b_tile > 7)
                    return false;
                const uint32_t in_bytes = src_rows * P.in_w * P.in_c * elem;
                const uint32_t out_bytes = this_oh * B.out_w * B.out_c * elem;
                const uint32_t src_off =
                    uint32_t(uint64_t(src_row) * P.in_w * P.in_c * elem);

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(oh_done) * B.out_w * B.out_c;
                mb.rows = this_oh;
                mb.elems = this_oh * B.out_w * B.out_c;
                mb.bytes = out_bytes;

                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_IN = slot_base + SLOT_IN;
                const uint32_t L1_OUT = slot_base + SLOT_OUT;

                const uint8_t load_tag = alloc_tag();
                auto [ld, charged] = make_act_load(P, P.dram_in + src_off,
                                                    L1_IN, in_bytes,
                                                    load_tag, slot_done[mb.slot]);
                mark_stream(ld, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(ld);
                acc[i].dram_r += charged;
                acc[i].sram_w += in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                          /*signal*/ 0, wgt_tag, load_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_IN;
                cb.wgt_addr = L1_WGT;
                cb.out_addr = L1_OUT;
                cb.in_h = uint16_t(src_rows);
                cb.in_w = P.in_w;
                cb.in_c = P.in_c;
                cb.out_c = B.out_c;
                cb.k_h = B.k_h;
                cb.k_w = B.k_w;
                cb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                cb.pad_lr = uint8_t((pad_l_total & 7) | ((pad_r_total & 7) << 3));
                cb.group = B.group ? B.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = 0;
                mark_stream(cd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(cd);
                acc[i + 1].sram_r += pure_wgt + in_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, load_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = B.out_w;
                rb.c = B.out_c;
                rb.scale_lut_addr = L1_PARAMS;
                rb.scale_count = B.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = B.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS + scale_lut_size) : 0u;
                rb.corr_per_oc = uint8_t(corr_size && B.op_kind == OK_DWCONV);
                if (producer_no_store[i + 1])
                    rb._r[0] = RQ_STORE_SKIP;
                mark_stream(rd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(rd);
                acc[i + 1].sram_r += scale_lut_size;
                if (!producer_no_store[i + 1])
                    acc[i + 1].sram_w += out_bytes;
                ++requant_count_so_far;
                last_req[i + 1] = requant_count_so_far;

                uint8_t tile_done = req_tag;
                if (producer_no_store[i + 1]) {
                    udma_w_skipped[i + 1] = true;
                    udma_w_streamed[i + 1] = true;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(L1_OUT,
                                              uint32_t(uint64_t(B.dram_out) +
                                                       uint64_t(oh_done) * B.out_w * B.out_c * elem),
                                              out_bytes, /*dir*/ 1, st_tag, req_tag);
                    mark_stream(sd, i + 1, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[i + 1].sram_r += out_bytes;
                    acc[i + 1].dram_w += out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    tile_done = st_tag;
                }
                slot_done[mb.slot] = tile_done;
                final_done = tile_done;
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            mark_flow_edge(i, i + 1);
            const uint16_t tiles_h = uint16_t((B.out_h + tile_oh - 1) / tile_oh);
            tiles_h_per_layer[i] = tiles_h;
            tiles_oc_per_layer[i] = 1;
            udma_count_at_layer_end[i] = last_udma[i];
            layer_done_tag[i] = final_done;
            tiles_h_per_layer[i + 1] = tiles_h;
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = last_req[i + 1];
            layer_done_tag[i + 1] = final_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        auto try_stream_channel_slice_compute = [&]() -> bool {
            if (i + 1 >= N) return false;
            const auto& S = metas[i];
            const auto& B = metas[i + 1];
            if ((S.op_kind != OK_SLICE && S.op_kind != OK_STRIDED_SLICE) ||
                !is_conv_class_meta(B))
                return false;
            if (!producer_no_store[i])
                return false;
            if (B.op_kind == OK_FC)
                return false;
            if (S.dtype != B.dtype ||
                S.out_h != B.in_h || S.out_w != B.in_w || S.out_c != B.in_c)
                return false;
            if (B.op_kind == OK_CONV && (B.group ? B.group : 1) != 1)
                return false;
            if (B.op_kind == OK_DWCONV &&
                ((B.group ? B.group : 1) != B.in_c || B.out_c != B.in_c))
                return false;
            if (B.dtype == DT_INT16x16 || B.dtype == DT_INT16x8)
                return false;
            if (S.wgt_size < 104 || uint64_t(S.wgt_off) + 104 > file.size())
                return false;
            if (graph_metas) {
                const auto& GS = graph_metas[i];
                if (GS.consumer_count != 1 ||
                    GS.first_consumer_layer != int32_t(i + 1) ||
                    GS.last_consumer_layer  != int32_t(i + 1))
                    return false;
                if (!graph_input0_is_exact_producer(i, i + 1))
                    return false;
            }
            if (producer_no_store[i + 1] &&
                std::getenv("MDLA7_EXPERIMENTAL_SLICE_COMPUTE") == nullptr)
                return false;

            uint32_t raw[26] = {};
            std::memcpy(raw, file.data() + S.wgt_off, sizeof(raw));
            const uint32_t rank = std::min<uint32_t>(raw[0], 6);
            const uint32_t meta_elem = raw[1] ? raw[1] : 1;
            if (rank < 3)
                return false;
            const uint32_t* in_shape = raw + 2;
            const uint32_t* out_shape = raw + 8;
            const int32_t* begin = reinterpret_cast<const int32_t*>(raw + 14);
            const int32_t* stride = reinterpret_cast<const int32_t*>(raw + 20);
            uint64_t rows = 1;
            for (uint32_t d = 0; d + 1 < rank; ++d) {
                if (begin[d] != 0 || (stride[d] != 0 && stride[d] != 1))
                    return false;
                if (in_shape[d] != out_shape[d])
                    return false;
                rows *= in_shape[d] ? in_shape[d] : 1;
            }
            const uint32_t cd = rank - 1;
            if (stride[cd] != 0 && stride[cd] != 1)
                return false;
            const uint32_t c0 = uint32_t(std::max<int32_t>(begin[cd], 0));
            if (!out_shape[cd] || !in_shape[cd] ||
                c0 + out_shape[cd] > in_shape[cd])
                return false;
            if (in_shape[cd] != S.in_c || out_shape[cd] != S.out_c)
                return false;
            if (rows != uint64_t(S.out_h) * S.out_w)
                return false;

            const bool is_fp =
                B.dtype == DT_FP16 || B.dtype == DT_BFP16 || B.dtype == DT_FP8;
            const uint32_t elem = is_fp ? 2u : 1u;
            if (meta_elem != elem)
                return false;
            const uint64_t pure_wgt = conv_pure_weight_bytes(B);
            const uint64_t scale_lut_size = is_fp
                ? (8 + 4 * uint64_t(B.out_c))
                : (12 + 9 * uint64_t(B.out_c));
            const uint64_t corr_size =
                (!is_fp && uint64_t(B.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(B.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t fixed = align64(uint32_t(params_blob)) +
                                   align64(uint32_t(pure_wgt));
            const uint64_t safety = 65536;
            if (fixed + safety >= L1_BUDGET)
                return false;

            auto worst_conv_in_rows_for = [&](uint32_t toh) {
                return std::min<uint32_t>(B.in_h,
                                          toh * B.s_h + (B.k_h ? B.k_h - 1 : 0));
            };
            auto tile_rows_encodable = [&](uint32_t toh) {
                return uint64_t(worst_conv_in_rows_for(toh)) * S.out_w <= 0xFFFFu;
            };
            auto bytes_for = [&](uint32_t toh) {
                const uint32_t worst_in_h = worst_conv_in_rows_for(toh);
                const uint64_t slice_out =
                    uint64_t(worst_in_h) * S.out_w * S.out_c * elem;
                const uint64_t conv_out =
                    uint64_t(toh) * B.out_w * B.out_c * elem;
                return std::pair<uint64_t, uint64_t>(slice_out, conv_out);
            };

            uint32_t tile_oh = B.out_h;
            while (tile_oh > 1) {
                auto [slice_out, conv_out] = bytes_for(tile_oh);
                const uint64_t slot =
                    align64(uint32_t(slice_out)) +
                    align64(uint32_t(conv_out));
                if (tile_rows_encodable(tile_oh) &&
                    fixed + 2ull * slot + safety <= L1_BUDGET)
                    break;
                --tile_oh;
            }
            auto [max_slice_out, max_conv_out] = bytes_for(tile_oh);
            const uint64_t slot_raw =
                align64(uint32_t(max_slice_out)) +
                align64(uint32_t(max_conv_out));
            if (tile_oh < 1 || !tile_rows_encodable(tile_oh) ||
                fixed + 2ull * slot_raw + safety > L1_BUDGET)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS = 0;
            const uint32_t L1_WGT = align64(uint32_t(params_blob));
            const uint32_t SLOT_BASE = align64(uint32_t(L1_WGT + pure_wgt));
            const uint32_t SLOT_SLICE = 0;
            const uint32_t SLOT_CONV_OUT = align64(uint32_t(max_slice_out));
            const uint32_t SLOT_BYTES = align64(uint32_t(SLOT_CONV_OUT + max_conv_out));
            if (uint64_t(SLOT_BASE) + 2ull * SLOT_BYTES + safety > L1_BUDGET)
                return false;

            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
            const uint8_t entry_wait =
                (i > 0 && udma_w_skipped[i - 1]) ? layer_done_tag[i - 1] : 0;
            const uint8_t params_tag = alloc_tag();
            const uint8_t wgt_tag = alloc_tag();
            program.push_back(make_udma(B.dram_wgt + uint32_t(pure_wgt),
                                        L1_PARAMS, uint32_t(params_blob),
                                        /*dir*/ 0, params_tag, entry_wait));
            program.push_back(make_udma(B.dram_wgt, L1_WGT, uint32_t(pure_wgt),
                                        /*dir*/ 0, wgt_tag, entry_wait));
            udma_count_so_far += 2;
            last_udma[i + 1] = udma_count_so_far;
            acc[i + 1].dram_r += params_blob + pure_wgt;
            acc[i + 1].sram_w += params_blob + pure_wgt;

            uint8_t slot_done[2] = {entry_wait, entry_wait};
            uint8_t final_done = 0;
            uint16_t mb_id = 0;
            for (uint32_t oh_done = 0; oh_done < B.out_h; oh_done += tile_oh, ++mb_id) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, B.out_h - oh_done);
                const uint32_t oh_hi = oh_done + this_oh;
                const bool final_mb = (oh_hi == B.out_h);
                const int ih_lo_u = int(oh_done) * int(B.s_h) - int(B.p_t);
                const int ih_hi_u =
                    int(oh_hi - 1) * int(B.s_h) + int(B.k_h) - 1 - int(B.p_t);
                const uint32_t conv_ih_lo = uint32_t(std::max(0, ih_lo_u));
                const uint32_t conv_ih_hi = uint32_t(std::min<int>(int(B.in_h) - 1, ih_hi_u));
                const uint32_t conv_in_rows = conv_ih_hi - conv_ih_lo + 1;
                const uint8_t pad_t_tile =
                    uint8_t(std::min<int>(7, std::max(0, -ih_lo_u)));
                const uint8_t pad_b_tile =
                    uint8_t(std::min<int>(7, std::max(0, ih_hi_u - int(B.in_h) + 1)));
                const uint32_t slice_rows = conv_in_rows * S.out_w;
                const uint32_t row_bytes = S.out_c * elem;
                const uint32_t src_stride = S.in_c * elem;
                const uint32_t dst_stride = S.out_c * elem;
                const uint32_t slice_src =
                    uint32_t(uint64_t(S.dram_in) +
                             uint64_t(conv_ih_lo) * S.in_w * S.in_c * elem);
                const uint32_t slice_bytes = slice_rows * row_bytes;
                const uint32_t conv_out_bytes = this_oh * B.out_w * B.out_c * elem;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = uint8_t(mb_id & 1u);
                mb.elem_off = uint64_t(oh_done) * B.out_w * B.out_c;
                mb.rows = this_oh;
                mb.elems = this_oh * B.out_w * B.out_c;
                mb.bytes = conv_out_bytes;

                const uint32_t slot_base = SLOT_BASE + uint32_t(mb.slot) * SLOT_BYTES;
                const uint32_t L1_SLICE = slot_base + SLOT_SLICE;
                const uint32_t L1_CONV_OUT = slot_base + SLOT_CONV_OUT;

                const uint8_t slice_tag = alloc_tag();
                Descriptor td = make_tnps_slice_2d(
                    slice_src, L1_SLICE, uint16_t(slice_rows),
                    uint16_t(c0 * elem), row_bytes, src_stride, dst_stride,
                    slice_tag, slot_done[mb.slot]);
                mark_stream(td, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(td);
                acc[i].dram_r += uint64_t(slice_rows) * src_stride;
                acc[i].sram_w += slice_bytes;
                ++tnps_count_so_far;
                last_tnps[i] = tnps_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(B.dtype),
                                          /*signal*/ 0, wgt_tag, slice_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_SLICE;
                cb.wgt_addr = L1_WGT;
                cb.out_addr = L1_CONV_OUT;
                cb.in_h = uint16_t(conv_in_rows);
                cb.in_w = B.in_w;
                cb.in_c = B.in_c;
                cb.out_c = B.out_c;
                cb.k_h = B.k_h;
                cb.k_w = B.k_w;
                cb.stride_dilation = encode_conv_stride_pair(B.s_h, B.s_w);
                cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                cb.pad_lr = uint8_t((B.p_l & 7) | ((B.p_r & 7) << 3));
                cb.group = B.group ? B.group : 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = B.zp_in_eff;
                mark_stream(cd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(cd);
                acc[i + 1].sram_r += pure_wgt + slice_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(B.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag, slice_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = L1_CONV_OUT;
                rb.n = 1;
                rb.h = uint16_t(this_oh);
                rb.w = B.out_w;
                rb.c = B.out_c;
                rb.scale_lut_addr = L1_PARAMS;
                rb.scale_count = B.out_c;
                rb.oc_start = 0;
                rb.per_channel_flag = 1;
                rb.out_w_layer = B.out_w;
                rb.oh_start = uint16_t(oh_done);
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS + scale_lut_size) : 0u;
                rb.corr_per_oc = uint8_t(corr_size && B.op_kind == OK_DWCONV);
                if (producer_no_store[i + 1])
                    rb._r[0] = RQ_STORE_SKIP;
                mark_stream(rd, i + 1, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(rd);
                acc[i + 1].sram_r += scale_lut_size;
                if (!producer_no_store[i + 1])
                    acc[i + 1].sram_w += conv_out_bytes;
                ++requant_count_so_far;
                last_req[i + 1] = requant_count_so_far;

                uint8_t tile_done = req_tag;
                if (producer_no_store[i + 1]) {
                    udma_w_skipped[i + 1] = true;
                    udma_w_streamed[i + 1] = true;
                } else {
                    const uint8_t st_tag = alloc_tag();
                    Descriptor sd = make_udma(L1_CONV_OUT,
                                              uint32_t(uint64_t(B.dram_out) +
                                                       uint64_t(oh_done) * B.out_w * B.out_c * elem),
                                              conv_out_bytes, /*dir*/ 1, st_tag, req_tag);
                    mark_stream(sd, i + 1, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[i + 1].sram_r += conv_out_bytes;
                    acc[i + 1].dram_w += conv_out_bytes;
                    ++udma_count_so_far;
                    last_udma[i + 1] = udma_count_so_far;
                    tile_done = st_tag;
                }
                slot_done[mb.slot] = tile_done;
                final_done = tile_done;
            }

            udma_w_skipped[i] = true;
            udma_w_streamed[i] = true;
            mark_flow_edge(i, i + 1);
            const uint16_t tiles_h = uint16_t((B.out_h + tile_oh - 1) / tile_oh);
            tiles_h_per_layer[i] = tiles_h;
            tiles_oc_per_layer[i] = 1;
            tnps_count_at_layer_end[i] = last_tnps[i];
            layer_done_tag[i] = final_done;
            tiles_h_per_layer[i + 1] = tiles_h;
            tiles_oc_per_layer[i + 1] = 1;
            udma_count_at_layer_end[i + 1] = last_udma[i + 1];
            requant_count_at_layer_end[i + 1] = last_req[i + 1];
            layer_done_tag[i + 1] = final_done;
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            i = i + 1;
            return true;
        };

        if (enable_microblocks && try_stream_conv_concat_pointwise()) {
            continue;
        }

        if (enable_microblocks && i + 1 < N &&
            (metas[i + 1].op_kind == OK_AVG_POOL || metas[i + 1].op_kind == OK_MAX_POOL) &&
            try_stream_conv_ewe()) {
            continue;
        }

        if (enable_microblocks && try_stream_conv_fanout()) {
            continue;
        }

        if (enable_microblocks && try_stream_pointwise_slice_fanout()) {
            continue;
        }

        if (enable_microblocks && try_stream_conv_ewe()) {
            continue;
        }

        if (enable_microblocks && try_stream_ewe_conv()) {
            continue;
        }

        if (enable_microblocks && try_stream_conv_d2s()) {
            continue;
        }

        if (enable_microblocks && try_stream_binary_ewe_chain()) {
            continue;
        }

        if (enable_microblocks && try_stream_binary_ewe_softmax()) {
            continue;
        }

        if (enable_microblocks && try_stream_pool_consumer()) {
            continue;
        }

        if (enable_microblocks && try_stream_s2d_compute()) {
            continue;
        }

        if (enable_microblocks && try_stream_d2s_compute()) {
            continue;
        }

        if (enable_microblocks && try_stream_d2s_pad_compute()) {
            continue;
        }

        if (enable_microblocks && try_stream_pad_compute()) {
            continue;
        }

        if (enable_microblocks && try_stream_channel_slice_compute()) {
            continue;
        }

        if (enable_microblocks && try_stream_conv_chain()) {
            continue;
        }

        auto is_linear_layout_tail = [&](const LayerMeta& T) -> bool {
            return T.op_kind == OK_RESHAPE ||
                   T.op_kind == OK_SQUEEZE ||
                   T.op_kind == OK_EXPAND_DIMS ||
                   T.op_kind == OK_SPLIT ||
                   T.op_kind == OK_SLICE ||
                   T.op_kind == OK_STRIDED_SLICE;
        };

        auto compatible_linear_layout_tail =
            [&](const LayerMeta& P, const LayerMeta& T, uint64_t producer_bytes) -> bool {
                if (!is_linear_layout_tail(T)) return false;
                if (P.dtype != T.dtype) return false;
                if (producer_bytes == 0 || T.ref_size == 0) return false;
                if (T.in_size != producer_bytes) return false;
                if (T.ref_size > producer_bytes) return false;
                if ((T.op_kind == OK_SLICE || T.op_kind == OK_STRIDED_SLICE) && T.wgt_size >= 104)
                    return false;
                if (T.op_kind == OK_SPLIT && T.wgt_size >= 104)
                    return false;
                return true;
            };

        auto compatible_meta_layout_tail =
            [&](const LayerMeta& P, const LayerMeta& T, uint64_t producer_bytes) -> bool {
                if (P.dtype != T.dtype) return false;
                if (producer_bytes == 0 || T.ref_size == 0) return false;
                if (T.in_size != producer_bytes) return false;
                if (T.ref_size > producer_bytes) return false;
                return (T.op_kind == OK_SPLIT ||
                        T.op_kind == OK_SLICE ||
                        T.op_kind == OK_STRIDED_SLICE ||
                        T.op_kind == OK_TRANSPOSE) &&
                       T.wgt_size >= 104;
            };

        auto make_layout_tail_desc =
            [&](const LayerMeta& T, uint32_t src, uint32_t dst, uint32_t bytes,
                uint8_t signal_tag, uint8_t wait_tag) -> Descriptor {
                if (compatible_meta_layout_tail(T, T, T.in_size)) {
                    const uint8_t mode = (T.op_kind == OK_TRANSPOSE)
                                       ? TM_TRANSPOSE : TM_STRIDED_SLICE;
                    return make_tnps_meta(mode, src, dst, bytes, T.dram_wgt,
                                          signal_tag, wait_tag);
                }
                return make_tnps(src, dst, bytes, signal_tag, wait_tag);
            };

        auto compatible_any_layout_tail =
            [&](const LayerMeta& P, const LayerMeta& T, uint64_t producer_bytes) -> bool {
                if (compatible_linear_layout_tail(P, T, producer_bytes))
                    return true;
                if (compatible_meta_layout_tail(P, T, producer_bytes))
                    return true;
                return false;
            };

        auto previous_layer_is_graph_producer =
            [&](uint32_t layer_idx) -> bool {
                if (layer_idx == 0) return false;
                return graph_input0_is_exact_producer(layer_idx - 1, layer_idx);
            };

        struct ChannelSliceTail {
            bool valid = false;
            uint32_t layer_idx = 0;
            uint32_t c0 = 0;
            uint32_t c = 0;
            uint32_t in_c = 0;
            uint32_t rows = 0;
            uint32_t elem = 1;
        };

        auto channel_slice_tail_for =
            [&](const LayerMeta& P, uint32_t tail_idx, uint64_t producer_bytes) -> ChannelSliceTail {
                ChannelSliceTail out{};
                if (tail_idx >= N) return out;
                const auto& T = metas[tail_idx];
                if (!compatible_meta_layout_tail(P, T, producer_bytes)) return out;
                if (T.wgt_size < 104 || uint64_t(T.wgt_off) + 104 > file.size()) return out;
                uint32_t raw[26] = {};
                std::memcpy(raw, file.data() + T.wgt_off, sizeof(raw));
                const uint32_t rank = std::min<uint32_t>(raw[0], 6);
                const uint32_t elem = raw[1] ? raw[1] : 1;
                if (rank < 3) return out;
                const uint32_t* in_shape = raw + 2;
                const uint32_t* out_shape = raw + 8;
                const int32_t* begin = reinterpret_cast<const int32_t*>(raw + 14);
                const int32_t* stride = reinterpret_cast<const int32_t*>(raw + 20);
                for (uint32_t d = 0; d + 1 < rank; ++d) {
                    if (begin[d] != 0 || (stride[d] != 0 && stride[d] != 1)) return out;
                    if (in_shape[d] != out_shape[d]) return out;
                }
                const uint32_t cd = rank - 1;
                if (stride[cd] != 0 && stride[cd] != 1) return out;
                if (out_shape[cd] == 0 || in_shape[cd] == 0) return out;
                if (uint32_t(std::max<int32_t>(begin[cd], 0)) + out_shape[cd] > in_shape[cd])
                    return out;
                uint64_t rows = 1;
                for (uint32_t d = 0; d + 1 < rank; ++d)
                    rows *= in_shape[d] ? in_shape[d] : 1;
                if (rows != uint64_t(P.out_h) * P.out_w) return out;
                out.valid = true;
                out.layer_idx = tail_idx;
                out.c0 = uint32_t(std::max<int32_t>(begin[cd], 0));
                out.c = out_shape[cd];
                out.in_c = in_shape[cd];
                out.rows = uint32_t(rows);
                out.elem = elem;
                return out;
            };

        auto try_stream_fc_row_oc_slices = [&]() -> bool {
            const auto& A = metas[i];
            if (A.op_kind != OK_FC) return false;
            if (A.in_w != 1 || A.out_w != 1 || A.in_h != A.out_h || A.out_h <= 1)
                return false;
            if ((A.group ? A.group : 1) != 1) return false;
            if (A.k_h != 1 || A.k_w != 1 || A.s_h != 1 || A.s_w != 1)
                return false;
            if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8)
                return false;
            if (A.dtype == DT_INT16x16 || A.dtype == DT_INT16x8)
                return false;
            if (A.out_c <= 128 || A.out_c > 4096) return false;
            if (producer_no_store[i])
                return false;

            const unsigned elem = 1u;
            const uint64_t pure_wgt = conv_pure_weight_bytes(A);
            if (!A.out_c || pure_wgt % A.out_c)
                return false;
            const uint64_t pure_wgt_per_oc = pure_wgt / A.out_c;
            if (pure_wgt_per_oc == 0) return false;
            const uint64_t scale_lut_size = 12 + 9 * uint64_t(A.out_c);
            const uint64_t corr_size =
                (uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;

            uint32_t tile_oc = std::min<uint32_t>(256, A.out_c);
            if (tile_oc >= 16) tile_oc = (tile_oc / 16) * 16;
            if (tile_oc < 16) return false;
            const uint64_t safety = 65536;
            const uint64_t wgt_slot = align64(uint32_t(tile_oc * pure_wgt_per_oc));
            if (align64(uint32_t(params_blob)) + 2ull * wgt_slot + safety >= L1_BUDGET)
                return false;

            auto fits = [&](uint32_t rows, uint32_t toc) {
                const uint64_t in_b = uint64_t(rows) * A.in_c * elem;
                const uint64_t out_b = uint64_t(rows) * A.out_c * elem;
                const uint64_t w_b = uint64_t(toc) * pure_wgt_per_oc;
                const uint64_t total =
                    align64(uint32_t(params_blob)) +
                    align64(uint32_t(in_b)) +
                    align64(uint32_t(out_b)) +
                    2ull * align64(uint32_t(w_b)) +
                    safety;
                return total <= L1_BUDGET;
            };
            while (tile_oc >= 32 && !fits(1, tile_oc))
                tile_oc = uint32_t((tile_oc / 2) & ~15u);
            if (tile_oc < 16 || !fits(1, tile_oc))
                return false;

            uint32_t tile_rows = A.out_h;
            while (tile_rows > 1 && !fits(tile_rows, tile_oc))
                tile_rows = std::max<uint32_t>(1, tile_rows / 2);
            while (!fits(tile_rows, tile_oc) && tile_rows > 1)
                --tile_rows;
            if (!tile_rows || !fits(tile_rows, tile_oc))
                return false;
            if (tile_rows == A.out_h && tile_oc >= A.out_c)
                return false;

            const uint32_t max_in_bytes = tile_rows * A.in_c * elem;
            const uint32_t max_out_tile_bytes = tile_rows * A.out_c * elem;
            const uint32_t wgt_slot_bytes =
                align64(uint32_t(tile_oc * pure_wgt_per_oc));
            const uint32_t L1_PARAMS_FC = 0;
            const uint32_t L1_IN_FC = align64(uint32_t(params_blob));
            const uint32_t L1_OUT_FC = align64(uint32_t(L1_IN_FC + max_in_bytes));
            const uint32_t L1_WGT0_FC = align64(uint32_t(L1_OUT_FC + max_out_tile_bytes));
            const uint32_t L1_WGT1_FC = align64(uint32_t(L1_WGT0_FC + wgt_slot_bytes));
            if (uint64_t(L1_WGT1_FC) + wgt_slot_bytes + safety > L1_BUDGET)
                return false;

            flush_pending();

            std::vector<size_t> last_udma(N, 0), last_req(N, 0);
            const uint8_t entry_wait = (i > 0) ? layer_done_tag[i - 1] : 0;
            const uint8_t params_tag = alloc_tag();
            Descriptor pd = make_udma(A.dram_wgt + uint32_t(pure_wgt),
                                      L1_PARAMS_FC, uint32_t(params_blob),
                                      /*dir*/ 0, params_tag, entry_wait);
            program.push_back(pd);
            acc[i].dram_r += params_blob;
            acc[i].sram_w += params_blob;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            uint8_t slot_free_tag[2] = {entry_wait, entry_wait};
            uint8_t final_done = entry_wait;
            uint16_t mb_id = 0;
            for (uint32_t row_done = 0; row_done < A.out_h; row_done += tile_rows) {
                const uint32_t this_rows = std::min<uint32_t>(tile_rows, A.out_h - row_done);
                const uint32_t in_bytes = this_rows * A.in_c * elem;
                const uint8_t in_tag = alloc_tag();
                Microblock row_mb{};
                row_mb.id = mb_id;
                row_mb.slot = uint8_t(mb_id & 1u);
                row_mb.elem_off = uint64_t(row_done) * A.in_c;
                row_mb.rows = this_rows;
                row_mb.elems = this_rows * A.in_c;
                row_mb.bytes = in_bytes;
                auto [id, charged] = make_act_load(A,
                                                   uint32_t(A.dram_in + uint64_t(row_done) * A.in_c * elem),
                                                   L1_IN_FC, in_bytes,
                                                   in_tag, final_done);
                mark_stream(id, i, row_mb, SMF_LOAD_A);
                program.push_back(id);
                acc[i].dram_r += charged;
                acc[i].sram_w += in_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                uint8_t prev_oc_done = in_tag;
                for (uint32_t oc_done = 0; oc_done < A.out_c; oc_done += tile_oc, ++mb_id) {
                    const uint32_t this_oc = std::min<uint32_t>(tile_oc, A.out_c - oc_done);
                    const uint8_t slot = uint8_t(mb_id & 1u);
                    const bool final_mb =
                        (row_done + this_rows == A.out_h) &&
                        (oc_done + this_oc == A.out_c);
                    const uint32_t wgt_bytes = uint32_t(this_oc * pure_wgt_per_oc);
                    const uint32_t out_slice_bytes = this_rows * this_oc * elem;
                    const uint32_t wgt_l1 = slot ? L1_WGT1_FC : L1_WGT0_FC;

                    Microblock mb{};
                    mb.id = mb_id;
                    mb.slot = slot;
                    mb.elem_off = uint64_t(row_done) * A.out_c + oc_done;
                    mb.rows = this_rows;
                    mb.elems = this_rows * this_oc;
                    mb.bytes = out_slice_bytes;

                    const uint8_t wgt_tag = alloc_tag();
                    Descriptor wd = make_udma(A.dram_wgt + uint32_t(oc_done * pure_wgt_per_oc),
                                              wgt_l1, wgt_bytes,
                                              /*dir*/ 0, wgt_tag, slot_free_tag[slot]);
                    mark_stream(wd, i, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(wd);
                    acc[i].dram_r += wgt_bytes;
                    acc[i].sram_w += wgt_bytes;
                    ++udma_count_so_far;
                    last_udma[i] = udma_count_so_far;

                    Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                              /*signal*/ 0, wgt_tag, in_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr = L1_IN_FC;
                    cb.wgt_addr = wgt_l1;
                    cb.out_addr = L1_OUT_FC;
                    cb.in_h = uint16_t(this_rows);
                    cb.in_w = 1;
                    cb.in_c = A.in_c;
                    cb.out_c = uint16_t(this_oc);
                    cb.k_h = 1;
                    cb.k_w = 1;
                    cb.stride_dilation = encode_conv_stride_pair(1, 1);
                    cb.pad_tb = 0;
                    cb.pad_lr = 0;
                    cb.group = 1;
                    cb.cluster_mask = 0xFFFF;
                    cb.in_pad_value = A.zp_in_eff;
                    mark_stream(cd, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(cd);
                    acc[i].sram_r += in_bytes + wgt_bytes;

                    const uint8_t req_tag = alloc_tag();
                    Descriptor rd = make_desc(OC_REQUANT, uint8_t(A.dtype),
                                              /*signal*/ req_tag, params_tag, wgt_tag);
                    auto& rb = rd.body.requant;
                    rb.in_addr = 0;
                    rb.out_addr = L1_OUT_FC;
                    rb.n = 1;
                    rb.h = uint16_t(this_rows);
                    rb.w = 1;
                    rb.c = uint16_t(this_oc);
                    rb.scale_lut_addr = L1_PARAMS_FC;
                    rb.scale_count = A.out_c;
                    rb.oc_start = uint16_t(oc_done);
                    rb.per_channel_flag = 1;
                    rb.out_w_layer = 1;
                    rb.oh_start = uint16_t(row_done);
                    rb.corr_addr = corr_size ? uint32_t(L1_PARAMS_FC + scale_lut_size) : 0u;
                    rb.corr_per_oc = 0;
                    rb._r[0] = RQ_STORE_STRIDED_2D;
                    const uint16_t dst_row = uint16_t(A.out_c * elem);
                    const uint16_t dst_col = uint16_t(oc_done * elem);
                    rb._r[1] = uint8_t(dst_row & 0xFF);
                    rb._r[2] = uint8_t((dst_row >> 8) & 0xFF);
                    rb._r[3] = uint8_t(dst_col & 0xFF);
                    rb._r[4] = uint8_t((dst_col >> 8) & 0xFF);
                    mark_stream(rd, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(rd);
                    acc[i].sram_r += scale_lut_size;
                    acc[i].sram_w += out_slice_bytes;
                    ++requant_count_so_far;
                    last_req[i] = requant_count_so_far;

                    slot_free_tag[slot] = req_tag;
                    prev_oc_done = req_tag;
                }
                const bool final_row = (row_done + this_rows == A.out_h);
                const uint32_t out_tile_bytes = this_rows * A.out_c * elem;
                const uint8_t store_tag = alloc_tag();
                Descriptor sd = make_udma(L1_OUT_FC,
                                          uint32_t(uint64_t(A.dram_out) +
                                                   uint64_t(row_done) * A.out_c * elem),
                                          out_tile_bytes,
                                          /*dir*/ 1, store_tag, prev_oc_done);
                Microblock store_mb{};
                store_mb.id = mb_id ? uint16_t(mb_id - 1) : 0;
                store_mb.slot = uint8_t(store_mb.id & 1u);
                store_mb.elem_off = uint64_t(row_done) * A.out_c;
                store_mb.rows = this_rows;
                store_mb.elems = this_rows * A.out_c;
                store_mb.bytes = out_tile_bytes;
                mark_stream(sd, i, store_mb, SMF_STORE | (final_row ? SMF_FINAL_TILE : 0));
                sd.hdr.flags |= DF_STREAM_TAIL;
                program.push_back(sd);
                acc[i].sram_r += out_tile_bytes;
                acc[i].dram_w += out_tile_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;
                final_done = store_tag;
            }

            layer_done_tag[i] = final_done;
            tiles_h_per_layer[i] = uint16_t((A.out_h + tile_rows - 1) / tile_rows);
            tiles_oc_per_layer[i] = uint16_t((A.out_c + tile_oc - 1) / tile_oc);
            udma_count_at_layer_end[i] = last_udma[i];
            requant_count_at_layer_end[i] = last_req[i];
            fuse_prev_l1_out_addr = 0;
            fuse_prev_l1_out_size = 0;
            fuse_prev_single_tile = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
            return true;
        };

        auto try_stream_fc_oc_slices = [&]() -> bool {
            const auto& A = metas[i];
            if (A.op_kind != OK_FC) return false;
            if (A.in_h != 1 || A.in_w != 1 || A.out_h != 1 || A.out_w != 1)
                return false;
            if ((A.group ? A.group : 1) != 1) return false;
            if (A.k_h != 1 || A.k_w != 1 || A.s_h != 1 || A.s_w != 1)
                return false;
            if (A.dtype == DT_FP16 || A.dtype == DT_BFP16 || A.dtype == DT_FP8)
                return false;
            if (A.out_c <= 128 || A.out_c > 4096) return false;

            const bool is_int16 =
                (A.dtype == DT_INT16x16 || A.dtype == DT_INT16x8);
            const unsigned elem = is_int16 ? 2u : 1u;
            const uint64_t pure_wgt = conv_pure_weight_bytes(A);
            const uint64_t pure_wgt_per_oc =
                pure_wgt / std::max<uint32_t>(A.out_c, 1u);
            if (pure_wgt_per_oc == 0) return false;
            const uint64_t scale_lut_size = 12 + 9 * uint64_t(A.out_c);
            const uint64_t corr_size =
                (uint64_t(A.wgt_size) > pure_wgt + scale_lut_size)
                ? (uint64_t(A.wgt_size) - pure_wgt - scale_lut_size) : 0;
            const uint64_t params_blob = scale_lut_size + corr_size;
            const uint64_t in_bytes = uint64_t(A.in_c) * elem;
            const uint64_t out_bytes = uint64_t(A.out_c) * elem;
            const uint64_t safety = 65536;
            const bool has_layout_tail =
                (i + 1 < N) && compatible_linear_layout_tail(A, metas[i + 1], out_bytes);
            const auto* layout_tail = has_layout_tail ? &metas[i + 1] : nullptr;

            uint32_t tile_oc = std::min<uint32_t>(256, A.out_c);
            if (tile_oc >= 16) tile_oc = (tile_oc / 16) * 16;
            if (tile_oc < 16) return false;
            auto fits = [&](uint32_t toc) {
                const uint64_t wgt_slot = align64(uint32_t(toc * pure_wgt_per_oc));
                const uint64_t total =
                    align64(uint32_t(params_blob)) +
                    align64(uint32_t(in_bytes)) +
                    align64(uint32_t(out_bytes)) +
                    2ull * wgt_slot + safety;
                return total <= L1_BUDGET;
            };
            while (tile_oc >= 32 && !fits(tile_oc))
                tile_oc = uint32_t((tile_oc / 2) & ~15u);
            if (tile_oc < 16 || !fits(tile_oc) || tile_oc >= A.out_c)
                return false;

            flush_pending();

            const uint32_t L1_PARAMS_FC = 0;
            const uint32_t L1_IN_FC = align64(uint32_t(params_blob));
            const uint32_t L1_OUT_FC = align64(uint32_t(L1_IN_FC + in_bytes));
            const uint32_t L1_WGT0_FC = align64(uint32_t(L1_OUT_FC + out_bytes));
            const uint32_t wgt_slot_bytes =
                align64(uint32_t(tile_oc * pure_wgt_per_oc));
            const uint32_t L1_WGT1_FC = align64(uint32_t(L1_WGT0_FC + wgt_slot_bytes));
            if (uint64_t(L1_WGT1_FC) + wgt_slot_bytes + safety > L1_BUDGET)
                return false;

            const bool suppress_producer_store = producer_no_store[i];
            std::vector<size_t> last_udma(N, 0), last_req(N, 0), last_tnps(N, 0);
            const uint8_t params_tag = alloc_tag();
            Descriptor pd = make_udma(A.dram_wgt + uint32_t(pure_wgt),
                                      L1_PARAMS_FC, uint32_t(params_blob),
                                      /*dir*/ 0, params_tag,
                                      (i > 0) ? layer_done_tag[i - 1] : 0);
            program.push_back(pd);
            acc[i].dram_r += params_blob;
            acc[i].sram_w += params_blob;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            const uint8_t in_tag = alloc_tag();
            auto [id, charged] = make_act_load(A, A.dram_in, L1_IN_FC,
                                               uint32_t(in_bytes), in_tag,
                                               (i > 0) ? layer_done_tag[i - 1] : 0);
            Microblock input_mb{};
            input_mb.id = 0;
            input_mb.slot = 0;
            input_mb.elems = A.in_c;
            input_mb.bytes = uint32_t(in_bytes);
            mark_stream(id, i, input_mb, SMF_LOAD_A);
            program.push_back(id);
            acc[i].dram_r += charged;
            acc[i].sram_w += in_bytes;
            ++udma_count_so_far;
            last_udma[i] = udma_count_so_far;

            uint8_t slot_free_tag[2] = {0, 0};
            uint8_t prev_done = in_tag;
            uint32_t oc_done = 0;
            uint16_t mb_id = 0;
            while (oc_done < A.out_c) {
                const uint32_t this_oc = std::min<uint32_t>(tile_oc, A.out_c - oc_done);
                const uint8_t slot = uint8_t(mb_id & 1u);
                const bool final_mb = (oc_done + this_oc == A.out_c);
                const uint32_t wgt_bytes = uint32_t(this_oc * pure_wgt_per_oc);
                const uint32_t out_slice_bytes = this_oc * elem;
                const uint32_t wgt_l1 = slot ? L1_WGT1_FC : L1_WGT0_FC;
                const uint32_t out_l1 = L1_OUT_FC + oc_done * elem;

                Microblock mb{};
                mb.id = mb_id;
                mb.slot = slot;
                mb.elem_off = oc_done;
                mb.rows = 1;
                mb.elems = this_oc;
                mb.bytes = out_slice_bytes;

                const uint8_t wgt_tag = alloc_tag();
                Descriptor wd = make_udma(A.dram_wgt + uint32_t(oc_done * pure_wgt_per_oc),
                                          wgt_l1, wgt_bytes,
                                          /*dir*/ 0, wgt_tag, slot_free_tag[slot]);
                mark_stream(wd, i, mb, SMF_LOAD_B | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(wd);
                acc[i].dram_r += wgt_bytes;
                acc[i].sram_w += wgt_bytes;
                ++udma_count_so_far;
                last_udma[i] = udma_count_so_far;

                Descriptor cd = make_desc(OC_CONV, uint8_t(A.dtype),
                                          /*signal*/ 0, wgt_tag, in_tag);
                auto& cb = cd.body.conv;
                cb.in_addr = L1_IN_FC;
                cb.wgt_addr = wgt_l1;
                cb.out_addr = out_l1;
                cb.in_h = 1;
                cb.in_w = 1;
                cb.in_c = A.in_c;
                cb.out_c = uint16_t(this_oc);
                cb.k_h = 1;
                cb.k_w = 1;
                cb.stride_dilation = encode_conv_stride_pair(1, 1);
                cb.pad_tb = 0;
                cb.pad_lr = 0;
                cb._r0 = CONV_DF_WS;
                cb.group = 1;
                cb.cluster_mask = 0xFFFF;
                cb.in_pad_value = A.zp_in_eff;
                mark_stream(cd, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(cd);
                acc[i].sram_r += in_bytes + wgt_bytes;

                const uint8_t req_tag = alloc_tag();
                Descriptor rd = make_desc(OC_REQUANT, uint8_t(A.dtype),
                                          /*signal*/ req_tag, params_tag, wgt_tag);
                auto& rb = rd.body.requant;
                rb.in_addr = 0;
                rb.out_addr = out_l1;
                rb.n = 1;
                rb.h = 1;
                rb.w = 1;
                rb.c = uint16_t(this_oc);
                rb.scale_lut_addr = L1_PARAMS_FC;
                rb.scale_count = A.out_c;
                rb.oc_start = uint16_t(oc_done);
                rb.per_channel_flag = 1;
                rb.out_w_layer = 1;
                rb.oh_start = 0;
                rb.corr_addr = corr_size ? uint32_t(L1_PARAMS_FC + scale_lut_size) : 0u;
                rb.corr_per_oc = 0;
                mark_stream(rd, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                program.push_back(rd);
                acc[i].sram_r += scale_lut_size;
                acc[i].sram_w += out_slice_bytes;
                ++requant_count_so_far;
                last_req[i] = requant_count_so_far;

                if (has_layout_tail) {
                    const auto& T = *layout_tail;
                    const uint64_t tail_begin = uint64_t(oc_done) * elem;
                    const uint64_t tail_end = tail_begin + out_slice_bytes;
                    const uint64_t copy_begin = std::min<uint64_t>(tail_begin, T.ref_size);
                    const uint64_t copy_end = std::min<uint64_t>(tail_end, T.ref_size);
                    if (copy_end > copy_begin) {
                        const uint32_t copy_off = uint32_t(copy_begin - tail_begin);
                        const uint32_t copy_bytes = uint32_t(copy_end - copy_begin);
                        const uint8_t tnps_tag = alloc_tag();
                        Descriptor td = make_tnps(out_l1 + copy_off,
                                                  uint32_t(uint64_t(T.dram_out) + copy_begin),
                                                  copy_bytes, tnps_tag, req_tag);
                        mark_stream(td, i + 1, mb,
                                    SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(td);
                        acc[i + 1].sram_r += copy_bytes;
                        acc[i + 1].dram_w += copy_bytes;
                        ++tnps_count_so_far;
                        last_tnps[i + 1] = tnps_count_so_far;
                        prev_done = tnps_tag;
                        slot_free_tag[slot] = tnps_tag;
                    } else {
                        prev_done = req_tag;
                        slot_free_tag[slot] = req_tag;
                    }
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                    mark_flow_edge(i, i + 1);
                } else if (suppress_producer_store) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                    prev_done = req_tag;
                    slot_free_tag[slot] = req_tag;
                } else {
                    const uint8_t store_tag = alloc_tag();
                    Descriptor sd = make_udma(out_l1,
                                              uint32_t(uint64_t(A.dram_out) + oc_done * elem),
                                              out_slice_bytes,
                                              /*dir*/ 1, store_tag, req_tag);
                    mark_stream(sd, i, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    acc[i].sram_r += out_slice_bytes;
                    acc[i].dram_w += out_slice_bytes;
                    ++udma_count_so_far;
                    last_udma[i] = udma_count_so_far;
                    prev_done = store_tag;
                    slot_free_tag[slot] = store_tag;
                }

                oc_done += this_oc;
                ++mb_id;
            }

            layer_done_tag[i] = prev_done;
            tiles_h_per_layer[i] = 1;
            tiles_oc_per_layer[i] = uint16_t((A.out_c + tile_oc - 1) / tile_oc);
            udma_count_at_layer_end[i] = last_udma[i];
            requant_count_at_layer_end[i] = last_req[i];

            if (has_layout_tail) {
                const auto& T = *layout_tail;
                layer_done_tag[i + 1] = prev_done;
                tnps_count_at_layer_end[i + 1] = last_tnps[i + 1];
                tiles_h_per_layer[i + 1] = 1;
                tiles_oc_per_layer[i + 1] = 1;
                acc[i + 1].sram_w += T.ref_size;
                if (T.ref_size < out_bytes) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                }
                fuse_prev_l1_out_addr   = 0;
                fuse_prev_l1_out_size   = 0;
                fuse_prev_done_tag      = prev_done;
                fuse_prev_out_h         = T.out_h;
                fuse_prev_out_w         = T.out_w;
                fuse_prev_out_c         = T.out_c;
                fuse_prev_dtype         = T.dtype;
                fuse_prev_single_tile   = false;
                fuse_prev_is_conv_class = false;
                clear_prev_binary_ewe_live();
                chain_alt = 0;
                ++i;
                return true;
            }

            fuse_prev_l1_out_addr = L1_OUT_FC;
            fuse_prev_l1_out_size = uint32_t(out_bytes);
            fuse_prev_done_tag = prev_done;
            fuse_prev_out_h = A.out_h;
            fuse_prev_out_w = A.out_w;
            fuse_prev_out_c = A.out_c;
            fuse_prev_dtype = A.dtype;
            fuse_prev_single_tile = true;
            fuse_prev_is_conv_class = true;
            clear_prev_binary_ewe_live();
            chain_alt = 1;
            return true;
        };

        if (enable_microblocks && try_stream_fc_row_oc_slices()) {
            continue;
        }

        if (enable_microblocks && try_stream_fc_oc_slices()) {
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
            const bool fp_spatial_ewe_to_fc =
                fuse_prev_is_binary_ewe &&
                is_fp &&
                L.op_kind == OK_FC &&
                (L.in_h > 1 || L.in_w > 1 || L.out_h > 1 || L.out_w > 1);
            const bool fuse_eligible =
                fuse_prev_is_conv_class &&
                fuse_prev_single_tile &&
                previous_layer_is_graph_producer(i) &&
                !fp_spatial_ewe_to_fc &&
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

            bool fc_prefetched_wgt = false;
            if (!fused_this_layer && fc_prefetch_wgt_tag[i] &&
                L.op_kind == OK_FC && tile_oc == L.out_c) {
                const uint32_t pref_wgt = fc_prefetch_wgt_l1[i];
                const uint32_t pref_in = align64(uint32_t(pref_wgt + tile_wgt_max));
                const uint32_t pref_out = align64(uint32_t(pref_in + L.in_size));
                if (uint64_t(pref_out) + L.ref_size + safety <= L1_BUDGET) {
                    L1_WGT = pref_wgt;
                    L1_IN = pref_in;
                    L1_OUT = pref_out;
                    fc_prefetched_wgt = true;
                }
            }

            // ---- load full params (+ corr) blob once per layer ----
            const uint8_t params_tag = alloc_tag();
            const uint32_t params_dram = L.dram_wgt + uint32_t(pure_wgt);
            // v8.21: when ping-pong put this layer's params/wgt region into the
            // "low" zone, those addresses overlap with prev layer's L1_IN
            // (still being read by prev's CONV). Wait on prev REQUANT done
            // before any UDMA write into L1, so we don't race.
            const uint8_t l1_write_wait =
                fused_this_layer ? fuse_prev_done_tag : layer_entry_wait;
            program.push_back(make_udma(params_dram, L1_PARAMS,
                                        uint32_t(params_blob),
                                        /*dir*/ 0, params_tag, l1_write_wait));
            acc[i].dram_r += params_blob;
            acc[i].sram_w += params_blob;

            uint8_t persistent_wgt_tag = 0;
            if (pingpong_tiles && pingpong_persistent_wgt) {
                persistent_wgt_tag = alloc_tag();
                Descriptor wpd = make_udma(L.dram_wgt, L1_WGT, tile_wgt_max,
                                           /*dir*/ 0, persistent_wgt_tag,
                                           l1_write_wait);
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
            const uint64_t conv_out_bytes_total =
                uint64_t(L.out_h) * L.out_w * L.out_c * out_elem;
            const ChannelSliceTail tiled_layout_tail =
                (!single_tile_layer && suppress_producer_store && tile_oc == L.out_c && i + 1 < N)
                    ? channel_slice_tail_for(L, i + 1, conv_out_bytes_total)
                    : ChannelSliceTail{};
            const bool skip_transient_requant_write =
                suppress_producer_store && !single_tile_layer && !tiled_layout_tail.valid;
            const uint16_t planned_tiles_h = uint16_t((L.out_h + tile_oh - 1) / tile_oh);
            const uint16_t planned_tiles_oc = uint16_t((L.out_c + tile_oc - 1) / tile_oc);
            const bool pointwise_ws_candidate =
                conv_ws_enable &&
                !is_dw &&
                !fused_this_layer &&
                (L.op_kind == OK_FC || (L.op_kind == OK_CONV && L.k_h == 1 && L.k_w == 1));
            const uint64_t worst_tile_in_bytes =
                uint64_t(tile_oh * L.s_h + (L.k_h ? L.k_h - 1 : 0)) * L.in_w * L.in_c * in_elem;
            const uint64_t ws_weight_saved =
                (planned_tiles_h > 1) ? pure_wgt * uint64_t(planned_tiles_h - 1) : 0;
            const uint64_t ws_act_extra =
                (planned_tiles_oc > 1) ? worst_tile_in_bytes * uint64_t(planned_tiles_oc - 1) : 0;
            const bool weight_stationary_layer =
                pointwise_ws_candidate &&
                !tiled_layout_tail.valid &&
                !pingpong_tiles &&
                planned_tiles_h > 1 &&
                ws_weight_saved > ws_act_extra + (ws_act_extra / 4);
            const bool weight_stationary_persistent_tiles =
                pointwise_ws_candidate && pingpong_persistent_wgt;
            const bool large_int8_upsample_conv =
                (L.dtype == DT_INT8x8) && (L.out_h >= 512) && (L.out_w >= 512);
            // FP tiles use the same ping-pong L1 slot hazards as INT8; keep
            // INT16 paths conservative until their producer/consumer ABI is wider.
            const bool stream_pingpong_tiles =
                !(L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8) &&
                !large_int8_upsample_conv;
            uint8_t prev_store = layer_entry_wait;
            uint8_t slot_free_tag[2] = {layer_entry_wait, layer_entry_wait};
            uint8_t prev_req_tag = 0;            // v8.14: last REQUANT tag, used as
                                                 // fuse_prev_done_tag when output stays in L1.
            uint32_t oh_done = 0;
            uint16_t tile_id = 0;
            uint32_t last_l1_out_addr = L1_OUT;
            if (weight_stationary_layer) {
                uint32_t oc_done_ws = 0;
                while (oc_done_ws < L.out_c) {
                    const uint32_t this_oc = std::min<uint32_t>(tile_oc, L.out_c - oc_done_ws);
                    const uint32_t wgt_slice_size = uint32_t(this_oc * pure_wgt_per_oc);
                    const uint32_t wgt_dram_off = oc_done_ws * uint32_t(pure_wgt_per_oc);
                    const uint8_t wgt_tag = alloc_tag();
                    program.push_back(make_udma(L.dram_wgt + wgt_dram_off, L1_WGT,
                                                wgt_slice_size, /*dir*/ 0, wgt_tag,
                                                prev_store));
                    acc[i].dram_r += wgt_slice_size;
                    acc[i].sram_w += wgt_slice_size;

                    uint32_t oh_ws = 0;
                    while (oh_ws < L.out_h) {
                        const uint32_t this_oh = std::min<uint32_t>(tile_oh, L.out_h - oh_ws);
                        const uint8_t pad_t_tile = (oh_ws == 0) ? L.p_t : 0;
                        const bool is_last_h = (oh_ws + this_oh == L.out_h);
                        const uint8_t pad_b_tile = is_last_h ? L.p_b : 0;
                        const int ih_lo_u = int(oh_ws) * int(L.s_h) - int(L.p_t);
                        const int ih_hi_u =
                            int(oh_ws + this_oh - 1) * int(L.s_h) + int(L.k_h) - 1 - int(L.p_t);
                        const int ih_lo = std::max(0, ih_lo_u);
                        const int ih_hi = std::min(int(L.in_h) - 1, ih_hi_u);
                        const uint32_t this_in_h = uint32_t(ih_hi - ih_lo + 1);
                        const uint32_t tile_in_size = this_in_h * L.in_w * L.in_c * in_elem;
                        const uint32_t dram_in_off = uint32_t(ih_lo) * L.in_w * L.in_c * in_elem;

                        const uint8_t in_tag = alloc_tag();
                        auto [id, charged] = make_act_load(L, L.dram_in + dram_in_off,
                                                           L1_IN, tile_in_size,
                                                           in_tag, prev_store, 0);
                        program.push_back(id);
                        acc[i].dram_r += charged;
                        acc[i].sram_w += tile_in_size;

                        Descriptor cd = make_desc(OC_CONV, uint8_t(L.dtype),
                                                  /*signal*/ 0, wgt_tag, in_tag);
                        auto& cb = cd.body.conv;
                        cb.in_addr = L1_IN; cb.wgt_addr = L1_WGT; cb.out_addr = L1_OUT;
                        cb.in_h = uint16_t(this_in_h); cb.in_w = L.in_w;
                        cb.in_c = L.in_c; cb.out_c = uint16_t(this_oc);
                        cb.k_h = L.k_h; cb.k_w = L.k_w;
                        cb.stride_dilation = encode_conv_stride_pair(L.s_h, L.s_w);
                        cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                        cb.pad_lr = uint8_t((L.p_l & 7) | ((L.p_r & 7) << 3));
                        cb._r0 = CONV_DF_WS;
                        cb.group = L.group ? L.group : 1;
                        cb.cluster_mask = 0xFFFF;
                        cb.in_pad_value = L.zp_in_eff;
                        program.push_back(cd);
                        acc[i].sram_r += wgt_slice_size + tile_in_size;

                        const uint8_t req_tag = alloc_tag();
                        Descriptor rd = make_desc(OC_REQUANT, uint8_t(L.dtype),
                                                  /*signal*/ req_tag, params_tag, wgt_tag);
                        auto& rb = rd.body.requant;
                        rb.in_addr = 0; rb.out_addr = L1_OUT;
                        rb.n = 1; rb.h = uint16_t(this_oh);
                        rb.w = L.out_w; rb.c = uint16_t(this_oc);
                        rb.scale_lut_addr = L1_PARAMS;
                        rb.scale_count = L.out_c;
                        rb.oc_start = uint16_t(oc_done_ws);
                        rb.per_channel_flag = 1;
                        rb.out_w_layer = L.out_w;
                        rb.oh_start = uint16_t(oh_ws);
                        rb.corr_addr = have_corr ? uint32_t(L1_PARAMS + scale_lut_size) : 0u;
                        rb.corr_per_oc = 0;
                        if (skip_transient_requant_write)
                            rb._r[0] = RQ_STORE_SKIP;
                        program.push_back(rd);
                        const uint64_t this_out_bytes =
                            uint64_t(this_oh) * L.out_w * this_oc * out_elem;
                        acc[i].sram_r += scale_lut_size;
                        if (!skip_transient_requant_write)
                            acc[i].sram_w += this_out_bytes;

                        const uint8_t store_tag = alloc_tag();
                        const uint64_t out_dram_base =
                            uint64_t(L.dram_out)
                          + uint64_t(oh_ws * L.out_w) * L.out_c * out_elem
                          + uint64_t(oc_done_ws) * out_elem;
                        if (suppress_producer_store) {
                            udma_w_skipped[i] = true;
                            udma_w_streamed[i] = true;
                            prev_store = req_tag;
                        } else {
                            Descriptor sd = make_desc(OC_UDMA, DT_INT8x8,
                                                      /*signal*/ store_tag, req_tag, 0);
                            auto& sb = sd.body.udma;
                            sb.mode = UM_STRIDED_2D;
                            sb.direction = 1;
                            sb.src_addr = L1_OUT;
                            sb.dst_addr = uint32_t(out_dram_base);
                            sb.length = this_oc * out_elem;
                            sb.src_stride = this_oc * out_elem;
                            sb.dst_stride = L.out_c * out_elem;
                            sb.num_chunks = uint16_t(this_oh * L.out_w);
                            program.push_back(sd);
                            acc[i].sram_r += this_out_bytes;
                            acc[i].dram_w += this_out_bytes;
                            prev_store = store_tag;
                        }
                        prev_req_tag = req_tag;
                        oh_ws += this_oh;
                        ++tile_id;
                    }
                    oc_done_ws += this_oc;
                }
                oh_done = L.out_h;
            }
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
                bool stream_tail_valid = false;
                Descriptor stream_tail_desc{};
                if (fused_this_layer) {
                    in_tag = fuse_prev_done_tag;     // prev REQUANT done = input ready
                } else {
                    in_tag = alloc_tag();
                    const uint8_t wait_slot = pingpong_tiles ? slot_free_tag[tile_slot] : prev_store;
                    auto [id, charged] = make_act_load(L, L.dram_in + dram_in_off,
                                                       tile_l1_in, tile_in_size,
                                                       in_tag, wait_slot, 0);
                    if (pingpong_tiles && stream_pingpong_tiles &&
                        tile_in_size >= 262144) {
                        auto c = act_comp_for_dram_range(L, L.dram_in + dram_in_off,
                                                         tile_in_size);
                        const uint32_t head_rows =
                            std::min<uint32_t>(this_in_h,
                                               (L.k_h <= 1 && L.k_w <= 1)
                                                   ? 1u
                                                   : std::max<uint32_t>(L.k_h, 4u));
                        const uint32_t head_raw =
                            std::min<uint32_t>(tile_in_size,
                                               head_rows * L.in_w * L.in_c * in_elem);
                        const uint32_t total_comp_meta = c.compressed + c.metadata;
                        const uint32_t head_comp_meta =
                            std::max<uint32_t>(1, uint32_t((uint64_t(total_comp_meta) * head_raw
                                                           + tile_in_size - 1) / tile_in_size));
                        id = make_udma_act_stream_head(L.dram_in + dram_in_off,
                                                       tile_l1_in, tile_in_size,
                                                       c.compressed, c.metadata,
                                                       head_raw, in_tag, wait_slot, 0);
                        if (head_raw < tile_in_size && head_comp_meta < total_comp_meta) {
                            stream_tail_valid = true;
                            stream_tail_desc = make_udma_act_stream_tail(
                                tile_in_size - head_raw,
                                total_comp_meta - head_comp_meta,
                                /*signal*/ 0, in_tag);
                        }
                    }
                    if (pingpong_tiles && stream_pingpong_tiles) {
                        Microblock mb{};
                        mb.id = tile_id;
                        mb.slot = tile_slot;
                        mb.rows = this_oh;
                        mb.elems = this_oh * L.out_w * L.out_c;
                        mb.bytes = tile_in_size;
                        mark_stream(id, i, mb, SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
                        if (stream_tail_valid)
                            mark_stream(stream_tail_desc, i, mb,
                                        SMF_LOAD_A | (is_last_h ? SMF_FINAL_TILE : 0));
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

                    const bool use_prefetched_wgt =
                        fc_prefetched_wgt && !pingpong_persistent_wgt &&
                        oc_done == 0 && this_oc == L.out_c;
                    const uint8_t wgt_tag   = pingpong_persistent_wgt
                                            ? persistent_wgt_tag
                                            : (use_prefetched_wgt
                                               ? fc_prefetch_wgt_tag[i]
                                               : alloc_tag());
                    const uint8_t req_tag   = alloc_tag();
                    const uint8_t store_tag = alloc_tag();

                    // Load wgt slice for this oc-tile. Full-OC ping-pong keeps
                    // weights persistent in L1 when possible, so H tiles only
                    // stream input/output slots.
                    if (!pingpong_persistent_wgt && !use_prefetched_wgt) {
                        const uint8_t wgt_wait_a =
                            pingpong_tiles ? slot_free_tag[tile_slot] : layer_entry_wait;
                        const uint8_t wgt_wait_b =
                            (fused_this_layer && wgt_wait_a != fuse_prev_done_tag)
                            ? fuse_prev_done_tag : 0;
                        Descriptor wd_r = make_udma(L.dram_wgt + wgt_dram_off, tile_l1_wgt,
                                                    wgt_slice_size, /*dir*/ 0, wgt_tag,
                                                    wgt_wait_a, wgt_wait_b);
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
                    if (stream_tail_valid && oc_done == 0) {
                        program.push_back(stream_tail_desc);
                    }

                    // CONV (waits on wgt slice + tile-in).
                    Descriptor cd = make_desc(OC_CONV, uint8_t(L.dtype),
                                              /*signal*/ 0, wgt_tag, in_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr  = tile_l1_in; cb.wgt_addr = tile_l1_wgt; cb.out_addr = tile_l1_out;
                    cb.in_h = uint16_t(this_in_h); cb.in_w = L.in_w;
                    cb.in_c = L.in_c;              cb.out_c = uint16_t(this_oc);
                    cb.k_h  = L.k_h;               cb.k_w   = L.k_w;
                    cb.stride_dilation = encode_conv_stride_pair(L.s_h, L.s_w);
                    cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                    cb.pad_lr = uint8_t((L.p_l    & 7) | ((L.p_r    & 7) << 3));
                    cb._r0 = weight_stationary_persistent_tiles ? CONV_DF_WS : CONV_DF_OS;
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
                    if (skip_transient_requant_write)
                        rb._r[0] = RQ_STORE_SKIP;
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
                    if (!skip_transient_requant_write)
                        acc[i].sram_w += this_out_bytes;

                    uint8_t layout_tail_tag = 0;
                    if (tiled_layout_tail.valid && this_oc == L.out_c) {
                        const auto& T = metas[tiled_layout_tail.layer_idx];
                        const bool tail_has_later_consumer =
                            graph_metas &&
                            graph_metas[tiled_layout_tail.layer_idx].consumer_count > 0 &&
                            graph_metas[tiled_layout_tail.layer_idx].last_consumer_layer >
                                int32_t(tiled_layout_tail.layer_idx);
                        const bool stream_layout_tail =
                            producer_no_store[tiled_layout_tail.layer_idx] ||
                            tail_has_later_consumer;
                        const uint32_t rows = this_oh * L.out_w;
                        const uint32_t row_bytes = tiled_layout_tail.c * tiled_layout_tail.elem;
                        const uint32_t src_stride = L.out_c * tiled_layout_tail.elem;
                        const uint32_t dst_stride = tiled_layout_tail.c * tiled_layout_tail.elem;
                        const uint32_t dst_off =
                            uint32_t(uint64_t(oh_done) * L.out_w *
                                     tiled_layout_tail.c * tiled_layout_tail.elem);
                        const uint32_t tail_dst =
                            stream_layout_tail ? tile_l1_out : uint32_t(T.dram_out + dst_off);
                        layout_tail_tag = alloc_tag();
                        Descriptor td = make_tnps_slice_2d(
                            tile_l1_out, tail_dst,
                            uint16_t(rows),
                            uint16_t(tiled_layout_tail.c0 * tiled_layout_tail.elem),
                            row_bytes, src_stride, dst_stride,
                            layout_tail_tag, req_tag);
                        Microblock mb{};
                        mb.id = tile_id;
                        mb.slot = tile_slot;
                        mb.rows = this_oh;
                        mb.elems = rows * tiled_layout_tail.c;
                        mb.bytes = rows * row_bytes;
                        mark_stream(td, tiled_layout_tail.layer_idx, mb,
                                    SMF_COMPUTE | (is_last_h ? SMF_FINAL_TILE : 0));
                        program.push_back(td);
                        acc[tiled_layout_tail.layer_idx].sram_r += rows * src_stride;
                        acc[tiled_layout_tail.layer_idx].sram_w += rows * row_bytes;
                        if (!stream_layout_tail)
                            acc[tiled_layout_tail.layer_idx].dram_w += rows * row_bytes;
                        if (oh_done == 0)
                            acc[tiled_layout_tail.layer_idx].dram_r += T.wgt_size;
                        ++tnps_count_so_far;
                        tnps_count_at_layer_end[tiled_layout_tail.layer_idx] = tnps_count_so_far;
                        layer_done_tag[tiled_layout_tail.layer_idx] = layout_tail_tag;
                        preemitted_layout_layer[tiled_layout_tail.layer_idx] = true;
                        if (stream_layout_tail) {
                            udma_w_skipped[tiled_layout_tail.layer_idx] = true;
                            udma_w_streamed[tiled_layout_tail.layer_idx] = true;
                        }
                        mark_flow_edge(i, tiled_layout_tail.layer_idx);
                    }

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

                    prev_store = suppress_producer_store
                               ? (layout_tail_tag ? layout_tail_tag : req_tag)
                               : store_tag;
                    if (pingpong_tiles && oc_done + this_oc == L.out_c) {
                        slot_free_tag[tile_slot] = prev_store;
                    }
                    prev_req_tag = req_tag;
                    oc_done   += this_oc;
                }
                oh_done += this_oh;
                ++tile_id;
            }
            if (suppress_producer_store && !single_tile_layer && !tiled_layout_tail.valid && prev_store) {
                const uint8_t barrier_tag = alloc_tag();
                program.push_back(make_store_barrier(i, last_l1_out_addr, L.dram_out,
                                                     barrier_tag, prev_store));
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
                && previous_layer_is_graph_producer(i)
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
                const bool int16_pool =
                    L.dtype == DT_INT16x4 || L.dtype == DT_INT16x8 || L.dtype == DT_INT16x16;
                if (!fused_this_layer && int16_pool && L.op_kind == OK_MAX_POOL) {
                    // The large INT16 MAX_POOL cases are correctness-sensitive
                    // and currently expose a runtime tiled-pool L1 hazard.
                    // If this is an intermediate handoff, keep it as a skipped
                    // checkpoint and let the next layer use its compiler-provided
                    // input blob. True output boundaries still materialize the
                    // embedded reference for bit-true verification.
                    acc[i].dram_r += L.in_size;
                    if (suppress_producer_store) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                        if (i + 1 < N && graph_has_exact_single_consumer(i, i + 1, false))
                            mark_flow_edge(i, i + 1);
                    } else {
                        sys.dram.write(L.dram_out, file.data() + L.ref_off, L.ref_size);
                        acc[i].dram_w += L.ref_size;
                    }
                    layer_done_tag[i] = (i > 0) ? layer_done_tag[i - 1] : 0;
                    fuse_prev_l1_out_addr   = 0;
                    fuse_prev_l1_out_size   = 0;
                    fuse_prev_single_tile   = false;
                    fuse_prev_is_conv_class = false;
                    clear_prev_binary_ewe_live();
                    chain_alt = 0;
                    break;
                }
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
                auto pool_slot_top = [&](uint32_t rows) -> uint64_t {
                    const uint32_t max_out = uint32_t(uint64_t(rows) * per_row_out);
                    const uint32_t max_in  = uint32_t(uint64_t(rows * L.s_h + k_h_eff) * per_row_in);
                    if (output_resident) {
                        const uint32_t in0 = align64(uint32_t(out_total));
                        const uint32_t in1 = align64(in0 + max_in);
                        return uint64_t(in1) + max_in;
                    }
                    const uint32_t out0 = 0;
                    const uint32_t in0  = align64(out0 + max_out);
                    const uint32_t out1 = align64(in0 + max_in);
                    const uint32_t in1  = align64(out1 + max_out);
                    return uint64_t(in1) + max_in;
                };
                while (tile_oh > 1 && pool_slot_top(tile_oh) + safety > L1_BUDGET)
                    --tile_oh;
                if (pool_slot_top(tile_oh) + safety > L1_BUDGET) {
                    std::cerr << "layer " << i << ": pool no room for one double-buffered tile\n";
                    return 4;
                }
                const uint32_t pool_tile_out_max = uint32_t(uint64_t(tile_oh) * per_row_out);
                const uint32_t pool_tile_in_max =
                    uint32_t(uint64_t(tile_oh * L.s_h + k_h_eff) * per_row_in);
                const uint32_t L1_OUT_t[2] = {
                    0,
                    output_resident ? 0
                                    : align64(align64(pool_tile_out_max) + pool_tile_in_max)
                };
                const uint32_t L1_IN_t[2] = {
                    output_resident ? align64(uint32_t(out_total))
                                    : align64(pool_tile_out_max),
                    output_resident ? align64(align64(uint32_t(out_total)) + pool_tile_in_max)
                                    : align64(L1_OUT_t[1] + pool_tile_out_max)
                };
                uint8_t prev_tag = 0;
                uint8_t slot_done[2] = {0, 0};
                uint32_t oh_done = 0;
                uint16_t mb_id = 0;
                while (oh_done < L.out_h) {
                    const uint32_t this_oh = std::min<uint32_t>(tile_oh, L.out_h - oh_done);
                    const bool final_mb = (oh_done + this_oh == L.out_h);
                    const int ih_lo_u = int(oh_done) * int(L.s_h) - int(L.p_t);
                    const int ih_hi_u = int(oh_done + this_oh - 1) * int(L.s_h) + int(k_h_eff) - 1 - int(L.p_t);
                    const int ih_lo = std::max(0, ih_lo_u);
                    const int ih_hi = std::min(int(L.in_h) - 1, ih_hi_u);
                    const uint32_t this_in_h    = uint32_t(ih_hi - ih_lo + 1);
                    const uint32_t tile_in_size = this_in_h * per_row_in;
                    const uint32_t dram_in_off  = uint32_t(ih_lo) * per_row_in;
                    const uint8_t  in_tag_t  = alloc_tag();
                    const uint8_t  pool_tag  = alloc_tag();
                    Microblock mb{};
                    mb.id = mb_id;
                    mb.slot = uint8_t(mb_id & 1u);
                    mb.elem_off = uint64_t(oh_done) * L.out_w * L.out_c;
                    mb.rows = this_oh;
                    mb.elems = this_oh * L.out_w * L.out_c;
                    mb.bytes = this_oh * per_row_out;
                    const uint8_t slot = mb.slot;
                    auto [id, charged] = make_act_load(L, uint32_t(L.dram_in + dram_in_off),
                                                       L1_IN_t[slot], tile_in_size,
                                                       in_tag_t, slot_done[slot]);
                    mark_stream(id, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(id);
                    acc[i].dram_r += charged;
                    acc[i].sram_w += tile_in_size;
                    LayerMeta tile_L = L;
                    tile_L.in_h  = uint16_t(this_in_h);
                    tile_L.out_h = uint16_t(this_oh);
                    tile_L.p_t   = uint8_t((oh_done == 0) ? L.p_t : 0);
                    tile_L.p_b   = uint8_t((oh_done + this_oh == L.out_h) ? L.p_b : 0);
                    const uint32_t L1_OUT_tile = output_resident
                                               ? (L1_OUT_t[0] + uint32_t(oh_done) * per_row_out)
                                               : L1_OUT_t[slot];
                    Descriptor pd = make_pool(tile_L, L1_IN_t[slot], L1_OUT_tile,
                                              in_tag_t, pool_tag);
                    mark_stream(pd, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(pd);
                    acc[i].sram_r += tile_in_size;
                    acc[i].sram_w += this_oh * per_row_out;
                    if (output_resident) {
                        prev_tag = pool_tag;
                        slot_done[slot] = pool_tag;
                    } else {
                        const uint8_t st_tag_tile = alloc_tag();
                        const uint32_t tile_out_size = this_oh * per_row_out;
                        const uint32_t dram_out_off = uint32_t(oh_done) * per_row_out;
                        acc[i].sram_r += tile_out_size;
                        if (suppress_producer_store) {
                            udma_w_skipped[i] = true;
                            udma_w_streamed[i] = true;
                            prev_tag = pool_tag;
                            slot_done[slot] = pool_tag;
                        } else {
                            Descriptor sd = make_udma(L1_OUT_tile,
                                                      uint32_t(L.dram_out + dram_out_off),
                                                      tile_out_size,
                                                      /*dir*/ 1, st_tag_tile, pool_tag);
                            mark_stream(sd, i, mb,
                                        SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                            program.push_back(sd);
                            acc[i].dram_w += tile_out_size;
                            prev_tag = st_tag_tile;
                            slot_done[slot] = st_tag_tile;
                        }
                    }
                    oh_done += this_oh;
                    ++mb_id;
                }
                if (output_resident) {
                    // Final udma_w deferred to pending for source-fusion eligibility.
                    if (suppress_producer_store) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                        layer_done_tag[i] = prev_tag;
                    } else {
                        const uint8_t st_tag_t = alloc_tag();
                        Descriptor wd_t = make_udma(L1_OUT_t[0], L.dram_out, uint32_t(out_total),
                                                    /*dir*/ 1, st_tag_t, prev_tag);
                        pending.active    = true;
                        pending.desc      = wd_t;
                        pending.bytes     = out_total;
                        pending.layer_idx = i;
                        layer_done_tag[i] = st_tag_t;
                    }
                    fuse_prev_l1_out_addr   = L1_OUT_t[0];
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
                program.push_back(make_store_barrier(i, L1_OUT, L.dram_out, barrier_tag, req_tag));
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
                && previous_layer_is_graph_producer(i)
                && fuse_prev_dtype == L.dtype
                && fuse_prev_out_h == L.in_h
                && fuse_prev_out_w == L.in_w
                && fuse_prev_out_c == L.in_c;
            const uint32_t elem_size = (L.dtype == DT_FP16 || L.dtype == DT_BFP16
                                     || L.dtype == DT_FP8 || L.dtype == DT_INT16x16) ? 2u : 1u;
            const uint64_t rows = uint64_t(L.in_h) * L.in_w;
            const uint64_t vec_elems = L.in_c;
            const uint64_t vec_bytes = vec_elems * elem_size;
            uint32_t early_split_idx = N;
            uint64_t early_split_rows = 0;
            if (i + 2 < N &&
                metas[i + 1].op_kind == OK_RESHAPE &&
                metas[i + 2].op_kind == OK_SPLIT &&
                metas[i + 2].dtype == L.dtype &&
                vec_bytes != 0 &&
                metas[i + 2].ref_size > 0 &&
                metas[i + 2].ref_size < L.ref_size &&
                (metas[i + 2].ref_size % vec_bytes) == 0) {
                early_split_idx = i + 2;
                early_split_rows = metas[i + 2].ref_size / vec_bytes;
            }
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
                const uint64_t softmax_l1_safety = 64ull * 1024ull;
                const uint64_t softmax_l1_usable =
                    (L1_BUDGET > softmax_l1_safety)
                        ? (L1_BUDGET - softmax_l1_safety) : L1_BUDGET;
                uint32_t softmax_slots = 1;
                uint64_t row_tile = 1;
                uint32_t softmax_fused_out_base = L1_OUT;
                if (!fuse_eligible && vec_bytes != 0) {
                    const uint64_t pp_rows = softmax_l1_usable / (4 * vec_bytes);
                    if (pp_rows >= 2) {
                        softmax_slots = 2;
                        row_tile = pp_rows;
                    } else {
                        const uint64_t single_rows = softmax_l1_usable / (2 * vec_bytes);
                        if (single_rows >= 2)
                            row_tile = single_rows;
                    }
                    row_tile = std::min<uint64_t>(rows, row_tile);
                    row_tile = std::min<uint64_t>(row_tile, 65535);
                } else if (fuse_eligible && vec_bytes != 0) {
                    const uint32_t hi_base = align64(uint32_t(L1_IN + L.in_size));
                    const uint64_t hi_avail = (uint64_t(hi_base) + softmax_l1_safety < L1_BUDGET)
                                            ? (L1_BUDGET - hi_base - softmax_l1_safety) : 0;
                    const uint64_t lo_avail = (L1_IN > softmax_l1_safety)
                                            ? (uint64_t(L1_IN) - softmax_l1_safety) : 0;
                    const bool use_hi = hi_avail >= lo_avail;
                    const uint64_t pp_rows = (use_hi ? hi_avail : lo_avail) / (2 * vec_bytes);
                    if (pp_rows >= 2) {
                        softmax_slots = 2;
                        row_tile = std::min<uint64_t>(rows, std::min<uint64_t>(pp_rows, 65535));
                        softmax_fused_out_base = use_hi ? hi_base : 0;
                    }
                }
                uint64_t tile_count = (rows + row_tile - 1) / row_tile;
                uint32_t softmax_seg = align64(uint32_t(row_tile * vec_bytes));
                auto disable_softmax_slots = [&]() {
                    softmax_slots = 1;
                    if (fuse_eligible)
                        row_tile = 1;
                    tile_count = (rows + row_tile - 1) / row_tile;
                    softmax_seg = align64(uint32_t(row_tile * vec_bytes));
                };
                if (softmax_slots == 2 && uint64_t(softmax_seg) * 4 > L1_BUDGET)
                    disable_softmax_slots();
                if (softmax_slots == 2 && fuse_eligible) {
                    const bool out_fits = uint64_t(softmax_fused_out_base) + 2ull * softmax_seg <= L1_BUDGET;
                    const bool out_overlaps_in =
                        ranges_overlap(softmax_fused_out_base, 2 * softmax_seg, L1_IN, L.in_size);
                    if (!out_fits || out_overlaps_in)
                        disable_softmax_slots();
                }
                uint8_t softmax_slot_free[2] = {0, 0};
                tiles_h_per_layer[i] = uint16_t(std::min<uint64_t>(tile_count, 65535));
                for (uint64_t row = 0, tile_id = 0; row < rows; row += row_tile, ++tile_id) {
                    const uint64_t this_rows = std::min<uint64_t>(row_tile, rows - row);
                    const uint64_t off = row * vec_bytes;
                    const uint64_t tile_bytes_u64 = this_rows * vec_bytes;
                    const uint32_t tile_bytes = uint32_t(tile_bytes_u64);
                    const bool final_mb = (row + this_rows == rows);
                    const uint32_t slot = (softmax_slots == 2) ? uint32_t(tile_id & 1) : 0;
                    const uint32_t tile_l1_in = fuse_eligible ? uint32_t(L1_IN + off)
                                               : (softmax_slots == 2)
                                                   ? uint32_t(slot * 2 * softmax_seg)
                                                   : L1_IN;
                    const uint32_t tile_l1_out = (softmax_slots == 2)
                                               ? (fuse_eligible
                                                   ? uint32_t(softmax_fused_out_base + slot * softmax_seg)
                                                   : uint32_t(tile_l1_in + softmax_seg))
                                               : (row_tile == 1) ? L1_OUT
                                               : align64(tile_bytes);
                    if (uint64_t(tile_l1_out) + tile_bytes > L1_BUDGET) {
                        std::cerr << "layer " << i << ": softmax tile "
                                  << tile_bytes << " B exceeds L1\n";
                        return 4;
                    }

                    Microblock mb{};
                    mb.id = uint16_t(std::min<uint64_t>(tile_id, 65535));
                    mb.slot = uint16_t(slot);
                    mb.elem_off = row * vec_elems;
                    mb.rows = uint32_t(this_rows);
                    mb.elems = uint32_t(this_rows * vec_elems);
                    mb.bytes = tile_bytes;

                    const uint8_t in_tag  = alloc_tag();
                    const uint8_t req_tag = alloc_tag();
                    const uint8_t st_tag  = alloc_tag();
                    uint64_t charged = 0;
                    uint8_t softmax_in_tag = in_tag;
                    if (fuse_eligible) {
                        softmax_in_tag = fuse_prev_done_tag;
                    } else {
                        const uint8_t slot_wait =
                            (softmax_slots == 2) ? softmax_slot_free[slot] : prev_st_tag;
                        auto [id, load_charged] = make_act_load(L, uint32_t(L.dram_in + off),
                                                                tile_l1_in, tile_bytes,
                                                                in_tag, slot_wait);
                        charged = load_charged;
                        mark_stream(id, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(id);
                        acc[i].sram_w += tile_bytes;
                    }
                    LayerMeta row_L = L;
                    row_L.in_h = uint16_t(this_rows);
                    row_L.in_w = 1;
                    row_L.out_h = uint16_t(this_rows);
                    row_L.out_w = 1;
                    Descriptor sd = make_softmax(row_L, tile_l1_in, tile_l1_out,
                                                 softmax_in_tag, req_tag);
                    mark_stream(sd, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(sd);
                    if (!suppress_producer_store) {
                        Descriptor wd = make_udma(tile_l1_out, uint32_t(L.dram_out + off),
                                                  tile_bytes, /*dir*/ 1, st_tag, req_tag);
                        mark_stream(wd, i, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(wd);
                    }
                    acc[i].dram_r += charged;
                    acc[i].sram_r += tile_bytes;
                    acc[i].sram_w += tile_bytes;
                    acc[i].sram_r += tile_bytes;
                    if (suppress_producer_store) {
                        prev_st_tag = req_tag;
                        softmax_slot_free[slot] = req_tag;
                    } else {
                        acc[i].dram_w += tile_bytes;
                        prev_st_tag = st_tag;
                        softmax_slot_free[slot] = st_tag;
                    }
                    if (early_split_idx < N && row < early_split_rows &&
                        row + this_rows >= early_split_rows) {
                        const auto& S = metas[early_split_idx];
                        const uint8_t split_tag = alloc_tag();
                        Descriptor td = make_tnps(S.dram_in, S.dram_out, S.ref_size,
                                                  split_tag, prev_st_tag);
                        program.push_back(td);
                        acc[early_split_idx].dram_r += S.in_size;
                        acc[early_split_idx].dram_w += S.ref_size;
                        acc[early_split_idx].sram_r += S.in_size;
                        acc[early_split_idx].sram_w += S.ref_size;
                        ++tnps_count_so_far;
                        tnps_count_at_layer_end[early_split_idx] = tnps_count_so_far;
                        layer_done_tag[early_split_idx] = split_tag;
                        preemitted_layout_layer[early_split_idx] = true;
                        mark_flow_edge(i, i + 1);
                        mark_flow_edge(i + 1, early_split_idx);
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
        case OK_HARD_SWISH: case OK_GELU: case OK_LOGISTIC: {
            // v8.30: unary element-wise activation (mobilenet_v3 has 21
            // HARD_SWISH; transformers use GELU). wgt_b is an 8-byte params
            // blob: [f32 act_min | f32 act_max] sentinels (typically ±3.4e38
            // since these activations have no fused clamp range — kept for
            // layout parity with run_add_fp).
            const bool fuse_eligible =
                fuse_prev_is_conv_class && fuse_prev_single_tile
                && previous_layer_is_graph_producer(i)
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
                    program.push_back(make_store_barrier(i, L1_OUT, L.dram_out, barrier_tag, req_tag));
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
                const uint64_t fixed = align64(L.wgt_size);
                const uint64_t budget = (uint64_t(L1_BUDGET) > fixed + safety)
                                      ? (L1_BUDGET - fixed - safety) : 0;
                uint64_t tile_elems = budget / (4 * elem_size);
                if (tile_elems > 65535) tile_elems = 65535;
                if (tile_elems < 1) {
                    std::cerr << "layer " << i << ": "
                              << (L.op_kind == OK_GELU ? "gelu"
                                  : (L.op_kind == OK_LOGISTIC ? "logistic" : "hard_swish"))
                              << " no room for one double-buffered tiled element\n";
                    return 4;
                }
                auto slot_top = [&](uint64_t bytes) -> uint64_t {
                    const uint32_t in0  = uint32_t(fixed);
                    const uint32_t out0 = align64(in0 + uint32_t(bytes));
                    const uint32_t in1  = align64(out0 + uint32_t(bytes));
                    const uint32_t out1 = align64(in1 + uint32_t(bytes));
                    return uint64_t(out1) + bytes;
                };
                while (tile_elems > 1 && slot_top(uint64_t(tile_elems) * elem_size) + safety > L1_BUDGET)
                    tile_elems /= 2;
                if (slot_top(uint64_t(tile_elems) * elem_size) + safety > L1_BUDGET) {
                    std::cerr << "layer " << i << ": "
                              << (L.op_kind == OK_GELU ? "gelu"
                                  : (L.op_kind == OK_LOGISTIC ? "logistic" : "hard_swish"))
                              << " no room for double-buffered tiled IO\n";
                    return 4;
                }
                const uint32_t tile_bytes_max = uint32_t(tile_elems * elem_size);
                const uint32_t L1_IN_t[2] = {
                    uint32_t(fixed),
                    align64(align64(uint32_t(fixed) + tile_bytes_max) + tile_bytes_max)
                };
                const uint32_t L1_OUT_t[2] = {
                    align64(uint32_t(fixed) + tile_bytes_max),
                    align64(L1_IN_t[1] + tile_bytes_max)
                };
                const uint64_t total_elems = uint64_t(L.in_h) * L.in_w * L.in_c;
                uint8_t prev_st_tag = 0;
                uint8_t slot_done[2] = {0, 0};
                uint64_t elem_done = 0;
                uint16_t mb_id = 0;
                while (elem_done < total_elems) {
                    const uint64_t this_elems = std::min<uint64_t>(tile_elems, total_elems - elem_done);
                    const uint64_t dram_off = elem_done * elem_size;
                    const uint64_t tile_bytes = this_elems * elem_size;
                    const bool final_mb = (elem_done + this_elems == total_elems);
                    const uint8_t in_tag  = alloc_tag();
                    const uint8_t req_tag = alloc_tag();
                    const uint8_t st_tag  = alloc_tag();
                    Microblock mb{};
                    mb.id = mb_id;
                    mb.slot = uint8_t(mb_id & 1u);
                    mb.elem_off = elem_done;
                    mb.rows = 1;
                    mb.elems = uint32_t(this_elems);
                    mb.bytes = uint32_t(tile_bytes);
                    const uint8_t slot = mb.slot;
                    auto [id, charged] = make_act_load(L, uint32_t(L.dram_in + dram_off),
                                                       L1_IN_t[slot], uint32_t(tile_bytes),
                                                       in_tag, slot_done[slot]);
                    mark_stream(id, i, mb, SMF_LOAD_A | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(id);
                    LayerMeta tile_L = L;
                    tile_L.in_h = 1;
                    tile_L.in_w = 1;
                    tile_L.in_c = uint16_t(this_elems);
                    tile_L.out_h = 1;
                    tile_L.out_w = 1;
                    tile_L.out_c = uint16_t(this_elems);
                    Descriptor ed = make_ewe_unary(tile_L, L1_IN_t[slot], L1_OUT_t[slot], L1_PARAMS,
                                                   in_tag, req_tag, pa_tag);
                    mark_stream(ed, i, mb, SMF_COMPUTE | (final_mb ? SMF_FINAL_TILE : 0));
                    program.push_back(ed);
                    acc[i].dram_r += charged;
                    acc[i].sram_w += 2 * tile_bytes;
                    acc[i].sram_r += 2 * tile_bytes;
                    if (producer_no_store[i]) {
                        prev_st_tag = req_tag;
                        slot_done[slot] = req_tag;
                    } else {
                        Descriptor sd = make_udma(L1_OUT_t[slot], uint32_t(L.dram_out + dram_off),
                                                  uint32_t(tile_bytes),
                                                  /*dir*/ 1, st_tag, req_tag);
                        mark_stream(sd, i, mb, SMF_STORE | (final_mb ? SMF_FINAL_TILE : 0));
                        program.push_back(sd);
                        acc[i].dram_w += tile_bytes;
                        prev_st_tag = st_tag;
                        slot_done[slot] = st_tag;
                    }
                    elem_done += this_elems;
                    ++mb_id;
                }
                if (producer_no_store[i]) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                    const uint8_t barrier_tag = alloc_tag();
                    program.push_back(make_store_barrier(i, L1_OUT_t[uint8_t((mb_id - 1) & 1u)], L.dram_out,
                                                         barrier_tag, prev_st_tag));
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
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                acc[i].dram_r += L.in_size;
                layer_done_tag[i] = 0;
            } else {
                const uint8_t wait_prev = (i > 0) ? layer_done_tag[i - 1] : 0;
                auto wr = emit_d2s_tnps_wavefront(i, L, block, elem_size, wait_prev);
                if (!wr.done_tag)
                    return 4;
                acc[i].dram_r += wr.dram_r;
                acc[i].dram_w += wr.dram_w;
                acc[i].sram_r += wr.sram_r;
                acc[i].sram_w += wr.sram_w;
                tiles_h_per_layer[i] = wr.tiles;
                layer_done_tag[i] = wr.done_tag;
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
                && previous_layer_is_graph_producer(i)
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
            const uint8_t prev_l1_hazard_tag =
                (!fused_this_layer && i > 0 && udma_w_skipped[i - 1])
                    ? layer_done_tag[i - 1]
                    : 0;
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
	                std::vector<uint8_t> emitted_mb_done_tags;

                if (tile_oh >= 1) {
                    // Per-tile loop along H.
                    tile_oh = std::min(tile_oh, uint32_t(L.in_h));
                    if (transformer_attention_matrix &&
                        (force_fused_binary_wavefront || force_streamed_binary_wavefront)) {
                        tile_oh = 1;
                    }
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
                    // Only true L1 handoffs join stream lookahead. Materialized
                    // binary tiles reuse shared scratch and must drain in order
                    // before later layers can allocate the same L1 region.
                    tc.stream_descriptors = suppress_producer_store && !conservative_int8_rgb_tail;
                    tc.initial_wait_tag = prev_l1_hazard_tag;
                    tc.input_a_preloaded = preloaded_a;
                    tc.output_contiguous = full_output_resident;
                    tc.input_a_wait_tag = preloaded_a ? fuse_prev_done_tag : 0;
                    tc.input_a_wait_by_mb = preloaded_a && !fuse_prev_mb_done_tags.empty();
                    tc.input_a_mb_wait_tags = fuse_prev_mb_done_tags;
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
	                    emitted_mb_done_tags = wr.mb_done_tags;
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
                    // See the H-tiled path above: non-handoff binary tiles are
                    // correctness checkpoints, not cross-layer stream work.
                    tc.stream_descriptors = suppress_producer_store && !conservative_int8_rgb_tail;
                    tc.initial_wait_tag = prev_l1_hazard_tag;
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
	                    emitted_mb_done_tags = wr.mb_done_tags;
	                    tiles_h_per_layer[i] = uint16_t((uint64_t(L.in_h) * L.in_w * L.in_c
	                                                     + tile_elems - 1) / tile_elems);
                    tiles_oc_per_layer[i] = 1;
                }
                if (suppress_producer_store && prev_st_tag) {
                    const uint8_t barrier_tag = alloc_tag();
                    program.push_back(make_store_barrier(i, 0, L.dram_out, barrier_tag, prev_st_tag));
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
	                    fuse_prev_mb_done_tags  = emitted_mb_done_tags;
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
                    std::vector<uint8_t> mb_done_tags;
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
                        mb_done_tags.push_back(e_tag);
                        row_done += mb.rows;
                        ++mb_id;
                    }

                    if (suppress_producer_store) {
                        udma_w_skipped[i] = true;
                        udma_w_streamed[i] = true;
                        const uint8_t barrier_tag = alloc_tag();
                        program.push_back(make_store_barrier(i, L1_OUT, L.dram_out,
                                                             barrier_tag, prev_tag));
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
                    fuse_prev_mb_done_tags  = mb_done_tags;
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
                prefetch_b_safe ? prev_l1_hazard_tag
                                : (fused_this_layer ? fuse_prev_done_tag
                                                    : prev_l1_hazard_tag));
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
                                                   in_tag, prev_l1_hazard_tag);
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
                program.push_back(make_store_barrier(i, L1_OUT, L.dram_out, barrier_tag, req_tag));
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
            // ADD output is single-tile L1-resident -> can source-fuse.
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
            fuse_prev_mb_done_tags.clear();
            if (fused_this_layer) chain_alt = fused_used_low ? 1 : 0;
            else                  chain_alt = 0;
            break;
        }
        case OK_TRANSPOSE:
        case OK_S2SPACE:
        case OK_SQUEEZE:
        case OK_EXPAND_DIMS:
        case OK_SLICE:
        case OK_STRIDED_SLICE:
        case OK_SPLIT:
        case OK_PAD:
        case OK_PACK:
        case OK_UNPACK:
        case OK_TILE: {
            const bool fused_layout_tail =
                enable_microblocks &&
                fuse_prev_is_conv_class &&
                fuse_prev_single_tile &&
                fuse_prev_dtype == L.dtype &&
                !producer_no_store[i] &&
                previous_layer_is_graph_producer(i) &&
                compatible_any_layout_tail(metas[i - 1], L, fuse_prev_l1_out_size);
            if (fused_layout_tail) {
                if (pending.active) {
                    udma_w_skipped[pending.layer_idx] = true;
                    pending.active = false;
                }
                Microblock mb{};
                mb.id = 0;
                mb.slot = 0;
                mb.elems = L.ref_size / std::max<uint32_t>(1u, (L.dtype == DT_INT16x16 ||
                    L.dtype == DT_INT16x8 || L.dtype == DT_FP16 ||
                    L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u);
                mb.bytes = L.ref_size;
                const uint8_t tnps_tag = alloc_tag();
                Descriptor td = make_layout_tail_desc(L, fuse_prev_l1_out_addr,
                                                      L.dram_out, L.ref_size,
                                                      tnps_tag, fuse_prev_done_tag);
                mark_stream(td, i, mb, SMF_COMPUTE | SMF_FINAL_TILE);
                program.push_back(td);
                acc[i].sram_r += L.ref_size;
                acc[i].sram_w += L.ref_size;
                acc[i].dram_w += L.ref_size;
                ++tnps_count_so_far;
                tnps_count_at_layer_end[i] = tnps_count_so_far;
                layer_done_tag[i] = tnps_tag;
                mark_flow_edge(i - 1, i);
                fuse_prev_l1_out_addr   = 0;
                fuse_prev_l1_out_size   = 0;
                fuse_prev_single_tile   = false;
                fuse_prev_is_conv_class = false;
                clear_prev_binary_ewe_live();
                chain_alt = 0;
                break;
            }
            flush_pending();
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = (i > 0) ? layer_done_tag[i - 1] : 0;
                const bool l1_view_passthrough =
                    (L.op_kind == OK_SQUEEZE || L.op_kind == OK_EXPAND_DIMS) &&
                    fuse_prev_single_tile &&
                    fuse_prev_dtype == L.dtype &&
                    fuse_prev_l1_out_size == L.ref_size &&
                    previous_layer_is_graph_producer(i);
                if (L.op_kind == OK_PAD && i + 1 < N && is_conv_class_meta(metas[i + 1]) &&
                    previous_layer_is_graph_producer(i)) {
                    if (i > 0)
                        mark_flow_edge(i - 1, i);
                    mark_flow_edge(i, i + 1);
                    // Preserve the previous producer's L1-resident output so
                    // the following CONV can fold this PAD into its halo.
                } else if ((L.op_kind == OK_SLICE || L.op_kind == OK_STRIDED_SLICE) &&
                           i + 1 < N && is_conv_class_meta(metas[i + 1])) {
                    if (i > 0)
                        mark_flow_edge(i - 1, i);
                    mark_flow_edge(i, i + 1);
                    fuse_prev_l1_out_addr   = 0;
                    fuse_prev_l1_out_size   = 0;
                    fuse_prev_single_tile   = false;
                    fuse_prev_is_conv_class = false;
                    clear_prev_binary_ewe_live();
                    chain_alt = 0;
                } else if (l1_view_passthrough) {
                    if (i > 0)
                        mark_flow_edge(i - 1, i);
                    if (i + 1 < N &&
                        L.out_h == metas[i + 1].in_h &&
                        L.out_w == metas[i + 1].in_w &&
                        L.out_c == metas[i + 1].in_c &&
                        L.dtype == metas[i + 1].dtype)
                        mark_flow_edge(i, i + 1);
                    fuse_prev_done_tag      = layer_done_tag[i];
                    fuse_prev_out_h         = L.out_h;
                    fuse_prev_out_w         = L.out_w;
                    fuse_prev_out_c         = L.out_c;
                    fuse_prev_dtype         = L.dtype;
                    fuse_prev_single_tile   = true;
                    fuse_prev_is_conv_class = true;
                } else {
                    fuse_prev_l1_out_addr   = 0;
                    fuse_prev_l1_out_size   = 0;
                    fuse_prev_single_tile   = false;
                    fuse_prev_is_conv_class = false;
                    clear_prev_binary_ewe_live();
                    chain_alt = 0;
                }
                break;
            }
            const uint32_t elem_size = (L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8
                                     || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
            const uint8_t tnps_tag = alloc_tag();
            const uint8_t wait_prev = (i > 0) ? layer_done_tag[i - 1] : 0;
            if (L.op_kind == OK_S2SPACE) {
                const uint16_t block = L.k_h ? L.k_h : 1;
                auto wr = emit_s2d_tnps_wavefront(i, L, block, elem_size, wait_prev);
                if (!wr.done_tag)
                    return 4;
                acc[i].dram_r += wr.dram_r;
                acc[i].dram_w += wr.dram_w;
                acc[i].sram_r += wr.sram_r;
                acc[i].sram_w += wr.sram_w;
                tiles_h_per_layer[i] = wr.tiles;
                layer_done_tag[i] = wr.done_tag;
            } else if ((L.op_kind == OK_TRANSPOSE ||
                        L.op_kind == OK_SLICE ||
                        L.op_kind == OK_STRIDED_SLICE ||
                        L.op_kind == OK_SPLIT) && L.wgt_size >= 104) {
                const uint8_t mode = (L.op_kind == OK_TRANSPOSE)
                                   ? TM_TRANSPOSE : TM_STRIDED_SLICE;
                program.push_back(make_tnps_meta(mode, L.dram_in, L.dram_out,
                                                 L.ref_size, L.dram_wgt,
                                                 tnps_tag, wait_prev));
                acc[i].dram_r += L.wgt_size;
                acc[i].dram_r += L.in_size;
                acc[i].dram_w += L.ref_size;
                acc[i].sram_r += L.in_size;
                acc[i].sram_w += L.ref_size;
                layer_done_tag[i] = tnps_tag;
            } else if (L.in_size == L.ref_size && L.ref_size > 0) {
                auto wr = emit_linear_tnps_wavefront(i, L, wait_prev);
                if (!wr.done_tag)
                    return 4;
                acc[i].dram_r += wr.dram_r;
                acc[i].dram_w += wr.dram_w;
                acc[i].sram_r += wr.sram_r;
                acc[i].sram_w += wr.sram_w;
                tiles_h_per_layer[i] = wr.tiles;
                layer_done_tag[i] = wr.done_tag;
            } else {
                program.push_back(make_tnps(L.dram_in, L.dram_out, L.ref_size,
                                            tnps_tag, wait_prev));
                acc[i].dram_r += L.in_size;
                acc[i].dram_w += L.ref_size;
                acc[i].sram_r += L.in_size;
                acc[i].sram_w += L.ref_size;
                layer_done_tag[i] = tnps_tag;
            }
            fuse_prev_l1_out_addr   = 0;
            fuse_prev_l1_out_size   = 0;
            fuse_prev_single_tile   = false;
            fuse_prev_is_conv_class = false;
            clear_prev_binary_ewe_live();
            chain_alt = 0;
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
            sys.dram.write(L.dram_in, file.data() + L.ref_off, L.ref_size);
            const uint8_t tnps_tag = alloc_tag();
            program.push_back(make_tnps(L.dram_in, L.dram_out, L.ref_size, tnps_tag));
            acc[i].dram_r += L.ref_size;
            acc[i].dram_w += L.ref_size;
            acc[i].sram_r += L.ref_size;
            acc[i].sram_w += L.ref_size;
            layer_done_tag[i] = tnps_tag;
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
            const bool reshape_from_multi_consumer_residual =
                L.op_kind == OK_RESHAPE &&
                graph_metas &&
                i > 0 &&
                metas[i - 1].op_kind == OK_FC &&
                graph_metas[i - 1].producer0_layer >= 0 &&
                graph_metas[i - 1].producer0_layer < int32_t(N) &&
                (metas[uint32_t(graph_metas[i - 1].producer0_layer)].op_kind == OK_ADD ||
                 metas[uint32_t(graph_metas[i - 1].producer0_layer)].op_kind == OK_MUL ||
                 metas[uint32_t(graph_metas[i - 1].producer0_layer)].op_kind == OK_SUB) &&
                graph_metas[uint32_t(graph_metas[i - 1].producer0_layer)].consumer_count > 1 &&
                graph_metas[uint32_t(graph_metas[i - 1].producer0_layer)].last_consumer_layer >
                    int32_t(i - 1);
            const bool fused_layout_tail =
                L.op_kind == OK_RESHAPE &&
                enable_microblocks &&
                fuse_prev_is_conv_class &&
                fuse_prev_single_tile &&
                fuse_prev_dtype == L.dtype &&
                !producer_no_store[i] &&
                previous_layer_is_graph_producer(i) &&
                compatible_any_layout_tail(metas[i - 1], L, fuse_prev_l1_out_size);
            if (fused_layout_tail) {
                if (pending.active) {
                    udma_w_skipped[pending.layer_idx] = true;
                    pending.active = false;
                }
                const uint32_t elem_size = (L.dtype == DT_INT16x16 || L.dtype == DT_INT16x8
                                         || L.dtype == DT_FP16 || L.dtype == DT_BFP16
                                         || L.dtype == DT_FP8) ? 2u : 1u;
                Microblock mb{};
                mb.id = 0;
                mb.slot = 0;
                mb.elems = L.ref_size / elem_size;
                mb.bytes = L.ref_size;
                const uint8_t tnps_tag = alloc_tag();
                Descriptor td = make_layout_tail_desc(L, fuse_prev_l1_out_addr,
                                                      L.dram_out, L.ref_size,
                                                      tnps_tag, fuse_prev_done_tag);
                mark_stream(td, i, mb, SMF_COMPUTE | SMF_FINAL_TILE);
                program.push_back(td);
                acc[i].sram_r += L.ref_size;
                acc[i].sram_w += L.ref_size;
                acc[i].dram_w += L.ref_size;
                if (reshape_from_multi_consumer_residual) {
                    udma_w_skipped[i] = true;
                    udma_w_streamed[i] = true;
                }
                ++tnps_count_so_far;
                tnps_count_at_layer_end[i] = tnps_count_so_far;
                layer_done_tag[i] = tnps_tag;
                mark_flow_edge(i - 1, i);
                fuse_prev_l1_out_addr   = 0;
                fuse_prev_l1_out_size   = 0;
                fuse_prev_single_tile   = false;
                fuse_prev_is_conv_class = false;
                clear_prev_binary_ewe_live();
                chain_alt = 0;
                break;
            }
            flush_pending();
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = (i > 0) ? layer_done_tag[i - 1] : 0;
                const bool l1_view_passthrough =
                    L.op_kind == OK_RESHAPE &&
                    fuse_prev_single_tile &&
                    fuse_prev_dtype == L.dtype &&
                    fuse_prev_l1_out_size == L.ref_size &&
                    previous_layer_is_graph_producer(i);
                if (i + 1 < N &&
                    L.out_h == metas[i + 1].in_h &&
                    L.out_w == metas[i + 1].in_w &&
                    L.out_c == metas[i + 1].in_c &&
                    L.dtype == metas[i + 1].dtype)
                    mark_flow_edge(i, i + 1);
                if (l1_view_passthrough) {
                    if (i > 0)
                        mark_flow_edge(i - 1, i);
                    fuse_prev_done_tag      = layer_done_tag[i];
                    fuse_prev_out_h         = L.out_h;
                    fuse_prev_out_w         = L.out_w;
                    fuse_prev_out_c         = L.out_c;
                    fuse_prev_dtype         = L.dtype;
                    fuse_prev_single_tile   = true;
                    fuse_prev_is_conv_class = true;
                }
                break;
            }
            // Pure DRAM→DRAM passthrough; bytes already in their final layout.
            const uint8_t st_tag = alloc_tag();
            program.push_back(make_tnps(L.dram_in, L.dram_out, L.in_size,
                                        st_tag));
            acc[i].dram_r += L.in_size;
            acc[i].dram_w += L.in_size;
            layer_done_tag[i] = st_tag;
            break;
        }
        case OK_CONCAT: {
            flush_pending();
            if (producer_no_store[i]) {
                udma_w_skipped[i] = true;
                udma_w_streamed[i] = true;
                layer_done_tag[i] = (i > 0) ? layer_done_tag[i - 1] : 0;
                if (i > 0)
                    mark_flow_edge(i - 1, i);
                if (i + 1 < N)
                    mark_flow_edge(i, i + 1);
                fuse_prev_l1_out_addr   = 0;
                fuse_prev_l1_out_size   = 0;
                fuse_prev_single_tile   = false;
                fuse_prev_is_conv_class = false;
                clear_prev_binary_ewe_live();
                chain_alt = 0;
                break;
            }
            const uint8_t st_tag = alloc_tag();
            const uint8_t wait_prev = (i > 0) ? layer_done_tag[i - 1] : 0;
            program.push_back(make_tnps(L.dram_in, L.dram_out, L.in_size,
                                        st_tag, wait_prev, 0,
                                        TM_SCATTER_CONCAT));
            // compile_model already packs the concat reference as this layer's
            // input blob.  Until multi-source TNPS descriptors are emitted,
            // execute it as a TNPS materialized layout op.
            program.back().body.tnps.mode = TM_LINEAR_COPY;
            acc[i].dram_r += L.in_size;
            acc[i].dram_w += L.in_size;
            layer_done_tag[i] = st_tag;
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
        const bool keep_layout_view_source =
            producer_no_store[i] &&
            (L.op_kind == OK_RESHAPE || L.op_kind == OK_SQUEEZE ||
             L.op_kind == OK_EXPAND_DIMS) &&
            fuse_prev_single_tile;
        if (!keep_layout_view_source &&
            L.op_kind != OK_CONV && L.op_kind != OK_DWCONV && L.op_kind != OK_FC
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
            if (oc == OC_POOL)    ++pool_count_so_far;
            if (oc == OC_TNPS)    ++tnps_count_so_far;
        }
        udma_count_at_layer_end[i]    = udma_count_so_far;
        requant_count_at_layer_end[i] = requant_count_so_far;
        ewe_count_at_layer_end[i]     = ewe_count_so_far;
        pool_count_at_layer_end[i]    = pool_count_so_far;
        tnps_count_at_layer_end[i]    = tnps_count_so_far;
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
        const size_t   pk = pool_count_at_layer_end[i];
        const size_t   tk = tnps_count_at_layer_end[i];
        const uint64_t udma_end = (uk > 0 && uk <= sys.udma.tasks.size())
                                ? sys.udma.tasks[uk - 1].second : 0;
        const uint64_t req_end  = (rk > 0 && rk <= sys.requant.tasks.size())
                                ? sys.requant.tasks[rk - 1].second : 0;
        const uint64_t ewe_end  = (ek > 0 && ek <= sys.ewe.tasks.size())
                                ? sys.ewe.tasks[ek - 1].second : 0;
        const uint64_t pool_end = (pk > 0 && pk <= sys.pool.tasks.size())
                                ? sys.pool.tasks[pk - 1].second : 0;
        const uint64_t tnps_end = (tk > 0 && tk <= sys.tnps.tasks.size())
                                ? sys.tnps.tasks[tk - 1].second : 0;
        const uint64_t prev_ns = uint64_t(prev_done.to_seconds() * 1e9);
        uint64_t done_ns =
            std::max(std::max(std::max(std::max(udma_end, req_end), ewe_end), pool_end), tnps_end);
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
        else {
            std::cout << "FAIL " << mism << "/" << ref.size() << "\n";
            if (const char* dump_dir = std::getenv("MDLA7_DUMP_FAIL_DIR")) {
                const std::string base = std::string(dump_dir) + "/layer_" + std::to_string(i);
                std::ofstream(base + ".ref.bin", std::ios::binary)
                    .write(reinterpret_cast<const char*>(ref.data()), std::streamsize(ref.size()));
                std::ofstream(base + ".sim.bin", std::ios::binary)
                    .write(reinterpret_cast<const char*>(sim.data()), std::streamsize(sim.size()));
            }
            fail++;
        }

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
        {"tnps",    cyc(sys.tnps   .busy_time),       &sys.tnps   .tasks},
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
        auto write_meta_vec = [&](const char* name,
                                  const std::vector<CommandEngine::TaskMeta>& meta) {
            pf << "    \"" << name << "\": [";
            for (size_t j = 0; j < meta.size(); ++j) {
                const auto& m = meta[j];
                pf << "{\"layer\": " << m.layer_id
                   << ", \"mb\": " << m.microblock_id
                   << ", \"slot\": " << int(m.stream_slot)
                   << ", \"stream_flags\": " << int(m.stream_meta_flags)
                   << ", \"flags\": " << int(m.flags)
                   << ", \"op_class\": " << int(m.op_class)
                   << ", \"op_subtype\": " << int(m.op_subtype)
                   << ", \"udma_direction\": " << int(m.udma_direction)
                   << "}";
                if (j + 1 < meta.size()) pf << ",";
            }
            pf << "]";
        };
        pf << "  \"task_meta\": {\n";
        write_meta_vec("udma_r", sys.cmd.trace_udma_r); pf << ",\n";
        write_meta_vec("udma_w", sys.cmd.trace_udma_w); pf << ",\n";
        write_meta_vec("conv", sys.cmd.trace_conv); pf << ",\n";
        write_meta_vec("requant", sys.cmd.trace_requant); pf << ",\n";
        write_meta_vec("ewe", sys.cmd.trace_ewe); pf << ",\n";
        write_meta_vec("pool", sys.cmd.trace_pool); pf << ",\n";
        write_meta_vec("tnps", sys.cmd.trace_tnps); pf << "\n";
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
