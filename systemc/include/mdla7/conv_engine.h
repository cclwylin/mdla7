#pragma once

// CONV Engine — see spec §3A.4 / §3A.5
// 16 cluster, hybrid INT+FP, INT48 acc, output via 16 INT32 chain to Requant.
//
// v1.3: real chain dataflow.
//   - INT8 path computes int32 partial sums per (oh, ow, oc) and pushes them
//     to chain[oc % 16] in NHWC scan order.
//   - The RequantEngine drains the chains, applies per-channel requant, and
//     writes INT8 results to L1Mesh.
//   - ConvEngine no longer writes anything to L1Mesh directly.
//   - Other dtypes remain cycle-model-only stubs.

#include <systemc>
#include <array>
#include <vector>
#include <iostream>
#include <cstring>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"
#include "mdla7/fp_utils.h"

namespace mdla7 {

inline uint32_t decode_stride(uint8_t enc) {
    // v9.3 CONV uses 4-bit direct stride encoding so ETHZ_V6 stride=3
    // downsamplers and stride=16 ViT patchify convs are representable.
    // Nibble 0 is reserved for stride 16.
    return enc ? uint32_t(enc) : 16u;
}

// Cycle count via bit-mult invariant (spec §5.3 v1):
//   cycle = ceil(MAC_total * a_bits * b_bits / 1,048,576) + tile_fill
inline uint64_t conv_cycles(const ConvBody& c, DType dtype, uint64_t out_count) {
    const uint32_t group = c.group ? c.group : 1;
    const uint32_t in_per_group = c.in_c / group;
    uint64_t mac_total = uint64_t(c.k_h) * c.k_w * in_per_group * out_count;
    uint64_t a, b;
    switch (dtype) {
        case DT_INT8x4:   a = 8;  b = 4;  break;
        case DT_INT8x8:   a = 8;  b = 8;  break;
        case DT_INT16x4:  a = 16; b = 4;  break;
        case DT_INT16x8:  a = 16; b = 8;  break;
        case DT_INT16x16: a = 16; b = 16; break;
        case DT_FP8:      a = 8;  b = 8;  break;
        case DT_FP16:
        case DT_BFP16:    a = 16; b = 16; break;
        default:          a = 8;  b = 8;  break;
    }
    uint64_t bit_mult = mac_total * a * b;
    // Per-CONV-dispatch fill latency. Each tile pays this cost (spec §3A.5
    // pipeline: weight broadcast + first-pixel cluster fill ≈ 64 cyc) so a
    // multi-tile layer naturally accumulates fills. With the 2 MB L1 in
    // test_model.cpp emitting per-tile CONV descriptors, this is now the
    // dominant overhead for layers that don't fit single-shot.
    return (bit_mult + 1048575) / 1048576 + 64;
}

SC_MODULE(ConvEngine) {
    sc_core::sc_fifo_in<DescriptorBody>            cfg_in;
    std::array<sc_core::sc_fifo<int32_t>*, 16>     chain_out;
    sc_core::sc_fifo_out<uint8_t>                  done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};   // v4.3 profiler
    std::vector<std::pair<uint64_t, uint64_t>> tasks;     // (start_ns, end_ns)

    SC_HAS_PROCESS(ConvEngine);
    ConvEngine(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) {
        for (auto& p : chain_out) p = nullptr;
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            const ConvBody& c = body.conv;
            DType dt = static_cast<DType>(last_dtype);

            uint32_t s_h   = decode_stride(c.stride_dilation & 0x0F);
            uint32_t s_w   = decode_stride((c.stride_dilation >> 4) & 0x0F);
            uint32_t pad_t = c.pad_tb & 7;
            uint32_t pad_b = (c.pad_tb >> 3) & 7;
            uint32_t pad_l = c.pad_lr & 7;
            uint32_t pad_r = (c.pad_lr >> 3) & 7;
            uint32_t out_h = (c.in_h + pad_t + pad_b - c.k_h) / s_h + 1;
            uint32_t out_w = (c.in_w + pad_l + pad_r - c.k_w) / s_w + 1;

            std::cout << "[CONV] in=" << c.in_h << "x" << c.in_w << "x" << c.in_c
                      << "  k=" << int(c.k_h) << "x" << int(c.k_w)
                      << "  s=" << s_h << "x" << s_w
                      << "  pad=" << pad_t << "/" << pad_l
                      << "  out=" << out_h << "x" << out_w << "x" << c.out_c
                      << "  dtype=" << int(dt) << "\n";

            if (dt == DT_INT8x8) {
                compute_int<int8_t, int8_t>(c, s_h, s_w, pad_t, pad_l, out_h, out_w);
            } else if (dt == DT_INT16x16) {
                compute_int<int16_t, int16_t>(c, s_h, s_w, pad_t, pad_l, out_h, out_w);
            } else if (dt == DT_INT16x8) {
                // v8.27: hybrid INT16x8 — int16 activations, int8 weights
                // (TFLite "16x8 quantization", e.g. esrgan_int16/unet_int16).
                compute_int<int16_t, int8_t>(c, s_h, s_w, pad_t, pad_l, out_h, out_w);
            } else if (is_fp_dtype(dt)) {
                // v8: FP16 / BFP16 / FP8 path.  Internally we accumulate in
                // FP32; weights and activations are stored as FP32 in DRAM/L1
                // (compile_model casts FP16→FP32 at compile time so the engine
                // stays dtype-agnostic).  Spec §3A.2 has FP16/BF16 at
                // 4096 MAC/cycle = same bit-mult as INT16×16 — cycle model
                // already encodes this; only functional compute is new here.
                compute_fp(c, s_h, s_w, pad_t, pad_l, out_h, out_w);
            }

            uint64_t out_count = uint64_t(out_h) * out_w * c.out_c;
            uint64_t cyc = conv_cycles(c, dt, out_count);
            std::cout << "[CONV] estimated " << cyc << " cycles\n";
            // v8.6: don't double-count L1 input/weight reads on top of the
            // compute pipeline.  In real HW the cluster pipeline runs WHILE
            // operands stream in; the engine's wall-clock is max(L1 read time,
            // compute throughput), not sum.  We model that by computing how
            // many cycles already elapsed (L1 reads via impose_bank_latency)
            // and only waiting the remainder.
            const sc_core::sc_time elapsed = sc_core::sc_time_stamp() - t_begin;
            const uint64_t elapsed_cyc = uint64_t(elapsed.to_seconds() * 1e9);
            if (cyc > elapsed_cyc) wait(cyc - elapsed_cyc, sc_core::SC_NS);

            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end  .to_seconds() * 1e9));
            done_tag_out.write(0);
        }
    }

    // v1.3 + v4.1: stream int32 partial sums into chain[oc % 16] in NHWC order.
    // v8.27: split activation and weight types so INT16x8 hybrid (int16 act,
    // int8 wgt) works alongside the existing INT8×8 / INT16×16 paths.
    template <typename T_a, typename T_w>
    void compute_int(const ConvBody& c,
                     uint32_t s_h, uint32_t s_w,
                     uint32_t pad_t, uint32_t pad_l,
                     uint32_t out_h, uint32_t out_w) {
        const uint32_t group         = c.group ? c.group : 1;
        const uint32_t in_per_group  = c.in_c  / group;
        const uint32_t out_per_group = c.out_c / group;
        const uint64_t weight_elems  = uint64_t(c.out_c) * c.k_h * c.k_w * in_per_group;
        // v7: TFLite-style padding = zp_in. With in_pad_value=0, behaviour
        // matches v6. With in_pad_value=zp_in, the bias_eff fold (which uses
        // the FULL kernel sum_w) becomes correct at boundaries.
        const int64_t pad_v = int64_t(c.in_pad_value);

        std::vector<T_a> in_buf (uint64_t(c.in_h) * c.in_w * c.in_c);
        std::vector<T_w> wgt_buf(weight_elems);
        l1mgr.read(c.in_addr,  in_buf .data(), in_buf .size() * sizeof(T_a));
        l1mgr.read(c.wgt_addr, wgt_buf.data(), wgt_buf.size() * sizeof(T_w));

        for (uint32_t oh = 0; oh < out_h; ++oh)
        for (uint32_t ow = 0; ow < out_w; ++ow)
        for (uint32_t oc = 0; oc < c.out_c; ++oc) {
            const uint32_t g       = oc / out_per_group;
            const uint32_t ic_base = g  * in_per_group;
            // INT64 accumulator (spec §3A.1: INT48 ample headroom for int16x16).
            int64_t sum = 0;
            for (uint32_t kh = 0; kh < c.k_h; ++kh)
            for (uint32_t kw = 0; kw < c.k_w; ++kw)
            for (uint32_t icr = 0; icr < in_per_group; ++icr) {
                int ih = int(oh) * int(s_h) + int(kh) - int(pad_t);
                int iw = int(ow) * int(s_w) + int(kw) - int(pad_l);
                int64_t a = (ih >= 0 && ih < int(c.in_h) &&
                             iw >= 0 && iw < int(c.in_w))
                          ? int64_t(in_buf[(ih * c.in_w + iw) * c.in_c + (ic_base + icr)])
                          : pad_v;
                int64_t w = wgt_buf[((oc * c.k_h + kh) * c.k_w + kw) * in_per_group + icr];
                sum += a * w;
            }
            // Narrow to int32 with saturation (chain payload is int32).
            const int32_t psum =
                (sum >  INT32_MAX) ? INT32_MAX :
                (sum <  INT32_MIN) ? INT32_MIN : int32_t(sum);
            const int lane = oc & 0xF;
            if (chain_out[lane]) chain_out[lane]->write(psum);
        }

        const char* tname =
            (sizeof(T_a) == 2 && sizeof(T_w) == 2) ? "INT16x16" :
            (sizeof(T_a) == 2 && sizeof(T_w) == 1) ? "INT16x8"  : "INT8x8";
        std::cout << "[CONV] pushed " << uint64_t(out_h) * out_w * c.out_c
                  << " psums to chain (" << tname << ")\n";
    }

    // v8 / v8.10: FP path. Storage is FP16 (2 byte/elem); compute runs in FP32
    // (matches spec §3A.2: FP cluster has FP32 accumulator).  Result is
    // bit-cast to int32 and shipped through the 16-lane chain to RequantEngine.
    void compute_fp(const ConvBody& c,
                    uint32_t s_h, uint32_t s_w,
                    uint32_t pad_t, uint32_t pad_l,
                    uint32_t out_h, uint32_t out_w) {
        const uint32_t group         = c.group ? c.group : 1;
        const uint32_t in_per_group  = c.in_c  / group;
        const uint32_t out_per_group = c.out_c / group;
        const uint64_t weight_elems  = uint64_t(c.out_c) * c.k_h * c.k_w * in_per_group;
        // c.in_pad_value holds an FP16 bit pattern (0 = +0.0).
        const float pad_f = fp16_to_fp32(uint16_t(c.in_pad_value));

        // Read FP16 storage from L1 then up-cast to FP32 once for compute.
        const uint64_t in_count = uint64_t(c.in_h) * c.in_w * c.in_c;
        std::vector<uint16_t> in_h16 (in_count);
        std::vector<uint16_t> wgt_h16(weight_elems);
        l1mgr.read(c.in_addr,  in_h16 .data(), in_h16 .size() * sizeof(uint16_t));
        l1mgr.read(c.wgt_addr, wgt_h16.data(), wgt_h16.size() * sizeof(uint16_t));
        std::vector<float> in_buf (in_count);
        std::vector<float> wgt_buf(weight_elems);
        for (size_t i = 0; i < in_count;     ++i) in_buf [i] = fp16_to_fp32(in_h16 [i]);
        for (size_t i = 0; i < weight_elems; ++i) wgt_buf[i] = fp16_to_fp32(wgt_h16[i]);

        for (uint32_t oh = 0; oh < out_h; ++oh)
        for (uint32_t ow = 0; ow < out_w; ++ow)
        for (uint32_t oc = 0; oc < c.out_c; ++oc) {
            const uint32_t g       = oc / out_per_group;
            const uint32_t ic_base = g  * in_per_group;
            float sum = 0.0f;
            for (uint32_t kh = 0; kh < c.k_h; ++kh)
            for (uint32_t kw = 0; kw < c.k_w; ++kw)
            for (uint32_t icr = 0; icr < in_per_group; ++icr) {
                int ih = int(oh) * int(s_h) + int(kh) - int(pad_t);
                int iw = int(ow) * int(s_w) + int(kw) - int(pad_l);
                float a = (ih >= 0 && ih < int(c.in_h) &&
                           iw >= 0 && iw < int(c.in_w))
                        ? in_buf[(ih * c.in_w + iw) * c.in_c + (ic_base + icr)]
                        : pad_f;
                float w = wgt_buf[((oc * c.k_h + kh) * c.k_w + kw) * in_per_group + icr];
                sum += a * w;
            }
            int32_t bits;
            std::memcpy(&bits, &sum, 4);                // bit-cast FP32 → int32 chain payload
            const int lane = oc & 0xF;
            if (chain_out[lane]) chain_out[lane]->write(bits);
        }
        std::cout << "[CONV] pushed " << uint64_t(out_h) * out_w * c.out_c
                  << " FP32 psums to chain (FP path)\n";
    }

    // v0 hack: dtype latched by CmdEng before pushing body to FIFO.
    uint8_t last_dtype = DT_INT8x8;
};

} // namespace mdla7
