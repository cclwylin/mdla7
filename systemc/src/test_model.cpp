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
#include <cstring>
#include "mdla7/system.h"
#include "mdla7/fp_utils.h"

using namespace mdla7;

namespace {

#pragma pack(push, 1)
struct ProgHeader {
    uint32_t magic;          // 'MDL7'
    uint32_t version;        // 2
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

enum OpKindEnum : uint16_t {
    OK_CONV     = 0,
    OK_DWCONV   = 1,
    OK_AVG_POOL = 2,
    OK_MAX_POOL = 3,
    OK_SOFTMAX  = 4,
    OK_RESHAPE  = 5,
    OK_FC       = 6,        // FC == 1×1 conv with H=W=1; same execution path
    OK_ADD      = 7,        // element-wise binary add (residual / SE)
    OK_CONCAT   = 8,        // channel-axis concat (DRAM→DRAM copy, no L1)
    OK_GATHER   = 9,        // indexed lookup (DRAM→DRAM copy, no L1)
};
inline const char* op_name(uint16_t k) {
    switch (k) {
        case OK_CONV:     return "   conv";
        case OK_DWCONV:   return " dwconv";
        case OK_AVG_POOL: return "avgpool";
        case OK_MAX_POOL: return "maxpool";
        case OK_SOFTMAX:  return "softmax";
        case OK_RESHAPE:  return "reshape";
        case OK_FC:       return "     fc";
        case OK_ADD:      return "    add";
        case OK_CONCAT:   return " concat";
        case OK_GATHER:   return " gather";
    }
    return "??unknown";
}
#pragma pack(pop)
static_assert(sizeof(ProgHeader) == 16);
static_assert(sizeof(LayerMeta)  == 64);

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
    p.k_h = L.k_h;  p.k_w = L.k_w;
    p.stride = uint8_t((L.s_h == 2 ? 1 : 0) | ((L.s_w == 2 ? 1 : 0) << 2));
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
    d.hdr.dtype  = DT_INT8x8;
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
    e.subtype = ES_ADD;
    return d;
}

// v1.2: pure-weight bytes for a conv layer (excludes the requant params blob
// that compile_model appended to wgt_size). v4.1: int16 = 2 byte. v8: FP = 4
// byte. v8.10: FP storage in DRAM/L1 is FP16 = 2 byte/elem (FP cluster has
// FP32 accumulator internally — see spec §3A.2).
static uint64_t conv_pure_weight_bytes(const LayerMeta& L) {
    const uint32_t group = L.group ? L.group : 1;
    const uint64_t elements = uint64_t(L.out_c) * L.k_h * L.k_w * (uint64_t(L.in_c) / group);
    const unsigned esize =
        (L.dtype == DT_INT16x16
         || L.dtype == DT_FP16 || L.dtype == DT_BFP16 || L.dtype == DT_FP8) ? 2u : 1u;
    return elements * esize;
}

} // anon

int sc_main(int argc, char* argv[]) {
    sc_core::sc_report_handler::set_actions(sc_core::SC_INFO, sc_core::SC_DO_NOTHING);

    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " program.bin [--quiet]\n";
        return 2;
    }
    bool quiet = (argc > 2 && std::string(argv[2]) == "--quiet");

    // --- Load program ----
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "open " << argv[1] << " failed\n"; return 2; }
    std::vector<uint8_t> file(static_cast<size_t>(f.tellg()));
    f.seekg(0); f.read(reinterpret_cast<char*>(file.data()), file.size());

    auto* hdr = reinterpret_cast<ProgHeader*>(file.data());
    if (hdr->magic != 0x374C444Du || hdr->version != 2u) {
        std::cerr << "bad magic/version\n"; return 2;
    }
    auto* metas = reinterpret_cast<LayerMeta*>(file.data() + sizeof(ProgHeader));
    const uint32_t N = hdr->num_layers;
    std::cout << "test_model: " << argv[1] << "  ("
              << N << " layers, " << file.size() / 1024 << " KB)\n";

    // --- Build sim, populate DRAM, build descriptor program ----
    Mdla7System sys("mdla7");

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
    constexpr uint32_t L1_BUDGET = L1MESH_BYTES;     // 2 MB (spec §3A.10)

    // Helper: build a descriptor with up to two waits.
    auto make_desc = [](OpClass cls, uint8_t dtype, uint8_t signal_tag,
                        uint8_t wait_a, uint8_t wait_b) -> Descriptor {
        Descriptor d{};
        d.hdr.op_class_subtype = cls;
        d.hdr.dtype       = dtype;
        d.hdr.signal_tag  = signal_tag;
        d.hdr.wait_count  = (wait_a ? 1 : 0) + (wait_b ? 1 : 0);
        d.hdr.wait_tags[0] = wait_a;
        d.hdr.wait_tags[1] = wait_b;
        return d;
    };

    // Per-layer accounting (v6.1 — moved from static to per-tile so tile-fill
    // halo redundancy and re-loaded params show up in the totals).
    struct LayerAcc { uint64_t dram_r = 0, dram_w = 0, sram_r = 0, sram_w = 0; };
    std::vector<LayerAcc> acc(N);

    // v7.1: per-layer tile counts (oh × oc) so the console / JSON / CSV / HTML
    // surface the tiling decision.  For non-conv layers both are 1.
    std::vector<uint16_t> tiles_h_per_layer (N, 1);
    std::vector<uint16_t> tiles_oc_per_layer(N, 1);

    std::vector<Descriptor> program;
    program.reserve(8 * N);

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
    int      fused_count = 0;          // for reporting

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

    for (uint32_t i = 0; i < N; ++i) {
        const auto& L = metas[i];
        // v8 / v8.10: per-dtype element width.  FP layers now store FP16 in
        // DRAM/L1 (2 B/elem); compute uses FP32 internally.
        const unsigned in_elem  =
            (L.dtype == DT_INT16x16
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
            uint32_t tile_oc = L.out_c;
            if (pure_wgt + scale_lut_size + safety > L1_BUDGET) {
                if (is_dw) {
                    std::cerr << "layer " << i
                              << ": dwconv with weights > 2 MB not supported "
                                 "(would need correlated OC+IC slicing)\n";
                    return 4;
                }
                const uint64_t half = (L1_BUDGET - scale_lut_size - safety) / 2;
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

            if (fuse_eligible) {
                L1_IN              = fuse_prev_l1_out_addr;
                const uint32_t in_end = fuse_prev_l1_out_addr + fuse_prev_l1_out_size;
                L1_PARAMS          = align64(in_end);
                L1_WGT             = align64(L1_PARAMS + uint32_t(params_blob));
                L1_OUT             = align64(L1_WGT + tile_wgt_max);
                const uint64_t worst_out =
                    uint64_t(L.out_h) * L.out_w * tile_oc * out_elem;
                if (uint64_t(L1_OUT) + worst_out + safety <= L1_BUDGET) {
                    fused_this_layer = true;
                    fused_count++;
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

            if (!fused_this_layer) {
                // Standard non-fused layout — same as before.
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
            }

            // ---- load full params (+ corr) blob once per layer ----
            const uint8_t params_tag = alloc_tag();
            const uint32_t params_dram = L.dram_wgt + uint32_t(pure_wgt);
            program.push_back(make_udma(params_dram, L1_PARAMS,
                                        uint32_t(params_blob),
                                        /*dir*/ 0, params_tag));
            acc[i].dram_r += params_blob;
            acc[i].sram_w += params_blob;

            // ---- emit per-tile descriptors (oh outer, oc inner) ----
            // v8.14: precompute single_tile here so the udma_w emission can
            // decide whether to defer the store to `pending` (skip path) or
            // push it inline (multi-tile, no fusion source).
            const bool single_tile_layer = (tile_oh == L.out_h) && (tile_oc == L.out_c);
            uint8_t prev_store = 0;
            uint8_t prev_req_tag = 0;            // v8.14: last REQUANT tag, used as
                                                 // fuse_prev_done_tag when output stays in L1.
            uint32_t oh_done = 0;
            while (oh_done < L.out_h) {
                const uint32_t this_oh = std::min<uint32_t>(tile_oh, L.out_h - oh_done);
                const uint8_t pad_t_tile = (oh_done == 0) ? L.p_t : 0;
                const bool    is_last_h  = (oh_done + this_oh == L.out_h);
                const uint8_t pad_b_tile = is_last_h ? L.p_b : 0;

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
                    program.push_back(make_udma(L.dram_in + dram_in_off, L1_IN, tile_in_size,
                                                /*dir*/ 0, in_tag, prev_store, 0));
                    acc[i].dram_r += tile_in_size;
                    acc[i].sram_w += tile_in_size;
                }

                uint32_t oc_done = 0;
                while (oc_done < L.out_c) {
                    const uint32_t this_oc = std::min<uint32_t>(tile_oc, L.out_c - oc_done);
                    const uint32_t wgt_slice_size = uint32_t(this_oc * pure_wgt_per_oc);
                    const uint32_t wgt_dram_off   = oc_done * uint32_t(pure_wgt_per_oc);

                    const uint8_t wgt_tag   = alloc_tag();
                    const uint8_t req_tag   = alloc_tag();
                    const uint8_t store_tag = alloc_tag();

                    // Load wgt slice for this oc-tile.
                    program.push_back(make_udma(L.dram_wgt + wgt_dram_off, L1_WGT,
                                                wgt_slice_size,
                                                /*dir*/ 0, wgt_tag));
                    acc[i].dram_r += wgt_slice_size;
                    acc[i].sram_w += wgt_slice_size;

                    // CONV (waits on wgt slice + tile-in).
                    Descriptor cd = make_desc(OC_CONV, uint8_t(L.dtype),
                                              /*signal*/ 0, wgt_tag, in_tag);
                    auto& cb = cd.body.conv;
                    cb.in_addr  = L1_IN; cb.wgt_addr = L1_WGT; cb.out_addr = L1_OUT;
                    cb.in_h = uint16_t(this_in_h); cb.in_w = L.in_w;
                    cb.in_c = L.in_c;              cb.out_c = uint16_t(this_oc);
                    cb.k_h  = L.k_h;               cb.k_w   = L.k_w;
                    cb.stride_dilation = encode_stride_pair(L.s_h, L.s_w);
                    cb.pad_tb = uint8_t((pad_t_tile & 7) | ((pad_b_tile & 7) << 3));
                    cb.pad_lr = uint8_t((L.p_l    & 7) | ((L.p_r    & 7) << 3));
                    cb.group  = L.group ? L.group : 1;
                    cb.cluster_mask = 0xFFFF;
                    cb.in_pad_value = L.zp_in_eff;            // v7: TFLite-correct boundary
                    program.push_back(cd);
                    acc[i].sram_r += wgt_slice_size + tile_in_size;

                    // REQUANT (params at L1_PARAMS; oc_start picks the slice).
                    Descriptor rd = make_desc(OC_REQUANT, uint8_t(L.dtype),
                                              /*signal*/ req_tag, params_tag, 0);
                    auto& rb = rd.body.requant;
                    rb.in_addr = 0; rb.out_addr = L1_OUT;
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
                    if (this_oc == L.out_c) {
                        Descriptor wd = make_udma(L1_OUT, uint32_t(out_dram_base),
                                                  uint32_t(this_out_bytes),
                                                  /*dir*/ 1, store_tag, req_tag, 0);
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
                        sb.src_addr   = L1_OUT;
                        sb.dst_addr   = uint32_t(out_dram_base);
                        sb.length     = this_oc * out_elem;
                        sb.src_stride = this_oc  * out_elem;
                        sb.dst_stride = L.out_c  * out_elem;
                        sb.num_chunks = uint16_t(this_oh * L.out_w);
                        program.push_back(sd);
                        acc[i].sram_r += this_out_bytes;
                        acc[i].dram_w += this_out_bytes;
                    }

                    prev_store = store_tag;
                    prev_req_tag = req_tag;
                    oc_done   += this_oc;
                }
                oh_done += this_oh;
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
            break;
        }
        case OK_AVG_POOL: case OK_MAX_POOL: {
            // v8.14: pool can't fuse with prev — flush any deferred udma_w.
            flush_pending();
            const uint32_t L1_IN  = 0;
            const uint32_t L1_OUT = align64(L.in_size);
            if (uint64_t(L1_OUT) + L.ref_size > L1_BUDGET) {
                std::cerr << "layer " << i << ": pool input+output exceed 2 MB L1\n";
                return 4;
            }
            const uint8_t in_tag  = alloc_tag();
            const uint8_t req_tag = alloc_tag();
            const uint8_t st_tag  = alloc_tag();
            program.push_back(make_udma(L.dram_in, L1_IN, L.in_size,
                                        /*dir*/ 0, in_tag));
            program.push_back(make_pool(L, L1_IN, L1_OUT, in_tag, req_tag));
            program.push_back(make_udma(L1_OUT, L.dram_out, L.ref_size,
                                        /*dir*/ 1, st_tag, req_tag));
            acc[i].dram_r += L.in_size;
            acc[i].sram_w += L.in_size;
            acc[i].sram_r += L.in_size;
            acc[i].sram_w += L.ref_size;
            acc[i].sram_r += L.ref_size;
            acc[i].dram_w += L.ref_size;
            layer_done_tag[i] = st_tag;
            break;
        }
        case OK_SOFTMAX: {
            flush_pending();
            const uint32_t L1_IN  = 0;
            const uint32_t L1_OUT = align64(L.in_size);
            if (uint64_t(L1_OUT) + L.ref_size > L1_BUDGET) {
                std::cerr << "layer " << i << ": softmax exceeds 2 MB L1\n";
                return 4;
            }
            const uint8_t in_tag  = alloc_tag();
            const uint8_t req_tag = alloc_tag();
            const uint8_t st_tag  = alloc_tag();
            program.push_back(make_udma(L.dram_in, L1_IN, L.in_size,
                                        /*dir*/ 0, in_tag));
            program.push_back(make_softmax(L, L1_IN, L1_OUT, in_tag, req_tag));
            program.push_back(make_udma(L1_OUT, L.dram_out, L.ref_size,
                                        /*dir*/ 1, st_tag, req_tag));
            acc[i].dram_r += L.in_size;
            acc[i].sram_w += L.in_size;
            acc[i].sram_r += L.in_size;
            acc[i].sram_w += L.ref_size;
            acc[i].sram_r += L.ref_size;
            acc[i].dram_w += L.ref_size;
            layer_done_tag[i] = st_tag;
            break;
        }
        case OK_ADD: {
            flush_pending();
            // wgt_b blob = input-B bytes followed by 48-byte ADD params blob.
            const uint32_t L1_WGT  = 0;          // input-B + params
            const uint32_t L1_IN   = align64(L.wgt_size);
            const uint32_t L1_OUT  = align64(L1_IN + L.in_size);
            if (uint64_t(L1_OUT) + L.ref_size > L1_BUDGET) {
                std::cerr << "layer " << i << ": ADD exceeds 2 MB L1\n";
                return 4;
            }
            const uint32_t params_l1 = L1_WGT + (L.wgt_size - 48);
            const uint8_t wgt_tag = alloc_tag();
            const uint8_t in_tag  = alloc_tag();
            const uint8_t req_tag = alloc_tag();
            const uint8_t st_tag  = alloc_tag();
            program.push_back(make_udma(L.dram_wgt, L1_WGT, L.wgt_size,
                                        /*dir*/ 0, wgt_tag));
            program.push_back(make_udma(L.dram_in,  L1_IN,  L.in_size,
                                        /*dir*/ 0, in_tag));
            program.push_back(make_ewe_add(L, L1_IN, L1_WGT, L1_OUT, params_l1,
                                           wgt_tag, in_tag, req_tag));
            program.push_back(make_udma(L1_OUT, L.dram_out, L.ref_size,
                                        /*dir*/ 1, st_tag, req_tag));
            acc[i].dram_r += L.in_size + L.wgt_size;
            acc[i].sram_w += L.in_size + L.wgt_size;
            acc[i].sram_r += L.in_size + L.wgt_size;
            acc[i].sram_w += L.ref_size;
            acc[i].sram_r += L.ref_size;
            acc[i].dram_w += L.ref_size;
            layer_done_tag[i] = st_tag;
            break;
        }
        case OK_RESHAPE:
        case OK_CONCAT:
        case OK_GATHER: {
            flush_pending();
            // Pure DRAM→DRAM passthrough; bytes already in their final layout.
            const uint8_t st_tag = alloc_tag();
            program.push_back(make_udma(L.dram_in, L.dram_out, L.in_size,
                                        /*dir*/ 1, st_tag));
            acc[i].dram_r += L.in_size;
            acc[i].dram_w += L.in_size;
            layer_done_tag[i] = st_tag;
            break;
        }
        default:
            std::cerr << "unknown op_kind " << L.op_kind << "\n";
            return 3;
        }
        // v8.13: reset fusion state for any op that wasn't CONV/DWCONV/FC.
        // CONV/DWCONV/FC branch sets these explicitly above.
        if (L.op_kind != OK_CONV && L.op_kind != OK_DWCONV && L.op_kind != OK_FC) {
            fuse_prev_is_conv_class = false;
            fuse_prev_single_tile   = false;
            fuse_prev_l1_out_size   = 0;
        }
        // Count UDMA + REQUANT descriptors emitted for this layer so we can
        // later look up the layer's "done time".  v8.14: REQUANT count covers
        // fusion-source layers whose udma_w was dropped — without UDMA marks,
        // the layer's true end is the REQUANT end.
        for (size_t pi = prog_start; pi < program.size(); ++pi) {
            const OpClass oc = program[pi].hdr.op_class();
            if (oc == OC_UDMA)    ++udma_count_so_far;
            if (oc == OC_REQUANT) ++requant_count_so_far;
        }
        udma_count_at_layer_end[i]    = udma_count_so_far;
        requant_count_at_layer_end[i] = requant_count_so_far;
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
    sc_core::sc_start(100.0, sc_core::SC_MS);

    if (quiet) std::cout.clear();

    // --- Verify each layer (byte-level so int8/int32 outputs use one path) ----
    int pass = 0, fail = 0;
    sc_core::sc_time prev_done = sc_core::SC_ZERO_TIME;
    uint64_t total_dram_r = 0, total_dram_w = 0;
    uint64_t total_sram_r = 0, total_sram_w = 0;

    // v4.3 profile output: per-layer + per-engine.
    struct LayerProfile {
        uint32_t id;
        std::string op;
        uint16_t in_h, in_w, in_c, out_h, out_w, out_c;
        uint8_t  k_h, k_w, s_h, s_w;
        uint16_t group;
        bool     pass;
        uint64_t cycles_layer, cycles_cum;
        uint64_t dram_r, dram_w, sram_r, sram_w;
        uint16_t tiles_h, tiles_oc;       // v7.1
        double   util_pct;                // v7.1: avg engine utilization within this layer's window
    };
    std::vector<LayerProfile> profile;
    profile.reserve(N);

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
        const uint64_t udma_end = (uk > 0 && uk <= sys.udma.tasks.size())
                                ? sys.udma.tasks[uk - 1].second : 0;
        const uint64_t req_end  = (rk > 0 && rk <= sys.requant.tasks.size())
                                ? sys.requant.tasks[rk - 1].second : 0;
        const uint64_t prev_ns = uint64_t(prev_done.to_seconds() * 1e9);
        uint64_t done_ns = std::max(udma_end, req_end);
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
            std::cout << "FUSED (output stays in L1)\n";
            ++pass;        // not verified, but not a failure either
        } else if (layer_pass) { std::cout << "PASS\n"; pass++; }
        else { std::cout << "FAIL " << mism << "/" << ref.size() << "\n"; fail++; }

        // accumulate the profile entry
        profile.push_back({
            i, op_name(L.op_kind),
            L.in_h, L.in_w, L.in_c, L.out_h, L.out_w, L.out_c,
            L.k_h, L.k_w, L.s_h, L.s_w, L.group,
            layer_pass, cyc_layer, cyc_total,
            dram_r, dram_w, sram_r, sram_w,
            th, toc, util_pct
        });
    }
    auto t = sys.cmd.last_activity;
    const uint64_t total_cycles = uint64_t(t.to_seconds() * 1e9);
    auto fmt_mb = [](uint64_t b){ return double(b) / (1024.0 * 1024.0); };
    std::cout << "\n  summary: " << pass << "/" << N
              << " layers PASS, " << fail << " FAIL"
              << " (" << fused_skipped << " fused — output stayed in L1, no per-layer verify)\n";
    std::cout << "  sim time: " << t
              << "  (= " << total_cycles << " cycles @ 1 GHz)\n";
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
        pf << "    \"util_peak_engine\": \"" << peak_eng << "\"\n";
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
        cf << "id,op,in_h,in_w,in_c,out_h,out_w,out_c,k_h,k_w,s_h,s_w,group,"
              "tiles_h,tiles_oc,pass,cycles_layer,cycles_cum,conv_util_pct,"
              "dram_r,dram_w,sram_r,sram_w\n";
        for (const auto& L : profile) {
            // op_name() pads with leading spaces for table alignment; strip for CSV.
            std::string op_clean = L.op;
            size_t a = op_clean.find_first_not_of(' ');
            if (a != std::string::npos) op_clean = op_clean.substr(a);
            cf << L.id << "," << op_clean << ","
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
