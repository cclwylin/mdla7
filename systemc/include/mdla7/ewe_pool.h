#pragma once

// EWE + POOL Engine — see spec §3A.7
//
// v1: real INT8 compute (bit-exact vs reference).
// v2.2 cycle model:
//   POOL: pipelined; cycle = max(K_h*K_w, 1) per output element across 16 lanes
//         + AXI fill (length / 128b) + drain (16 cycle).
//   EWE softmax: 3-pass exp+reduce_sum+div, 16-lane;
//         cycle = ceil(elem/16) * 3 + AXI fill/drain.

#include <systemc>
#include <iostream>
#include <vector>
#include <cstring>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"
#include "mdla7/softmax_lut.h"
#include "mdla7/requant.h"
#include "mdla7/fp_utils.h"

namespace mdla7 {

SC_MODULE(EweEngine) {
    sc_core::sc_fifo_in<DescriptorBody> cfg_in;
    sc_core::sc_fifo_out<uint8_t>       done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    uint8_t last_dtype = DT_INT8x8;            // v8.17: latched by CmdEng per dispatch

    SC_HAS_PROCESS(EweEngine);
    EweEngine(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) { SC_THREAD(run); }

    // v6: TFLite int8 ADD reference. Params blob layout at e.lut_addr:
    //   [ i32 zp_a | i32 zp_b | i32 zp_out
    //   | i32 mult_a | i32 shift_a
    //   | i32 mult_b | i32 shift_b
    //   | i32 mult_out | i32 shift_out
    //   | i32 left_shift | i32 act_min | i32 act_max ]   = 48 bytes
    void run_add(const EweBody& e, uint64_t elems) {
        std::vector<int8_t> a_buf(elems), b_buf(elems), out_buf(elems);
        l1mgr.read(e.in_a_addr, a_buf.data(), elems);
        l1mgr.read(e.in_b_addr, b_buf.data(), elems);
        int32_t p[12];
        l1mgr.read(e.lut_addr, p, sizeof(p));
        const int32_t zp_a = p[0], zp_b = p[1], zp_out = p[2];
        const int32_t mult_a = p[3], shift_a = p[4];
        const int32_t mult_b = p[5], shift_b = p[6];
        const int32_t mult_o = p[7], shift_o = p[8];
        const int32_t left_shift = p[9];
        const int32_t act_min = p[10], act_max = p[11];
        for (uint64_t i = 0; i < elems; ++i) {
            int32_t a = (int32_t(a_buf[i]) - zp_a) << left_shift;
            int32_t b = (int32_t(b_buf[i]) - zp_b) << left_shift;
            int32_t sa = multiply_by_quantized_multiplier(a, mult_a, shift_a);
            int32_t sb = multiply_by_quantized_multiplier(b, mult_b, shift_b);
            int32_t s  = sa + sb;
            int32_t v  = multiply_by_quantized_multiplier(s, mult_o, shift_o) + zp_out;
            if (v < act_min) v = act_min;
            if (v > act_max) v = act_max;
            out_buf[i] = int8_t(v);
        }
        l1mgr.write(e.out_addr, out_buf.data(), elems);
    }

    // v8.17: FP element-wise ADD. Storage is FP16 (2 B/elem) in L1; compute
    // in FP32 (matches the FP CONV/REQUANT path). Params blob layout at
    // e.lut_addr is the same 48 bytes the INT path uses, but the first 8
    // bytes are now [ f32 act_min | f32 act_max ] (sentinels +/-3.4e38 stand
    // in for +/-inf so the clamp is unconditional, mirroring sim ≡ ref).
    void run_add_fp(const EweBody& e, uint64_t elems) {
        std::vector<uint16_t> a16(elems), b16(elems), out16(elems);
        l1mgr.read(e.in_a_addr, a16.data(), elems * sizeof(uint16_t));
        l1mgr.read(e.in_b_addr, b16.data(), elems * sizeof(uint16_t));
        float clip[2];
        l1mgr.read(e.lut_addr, clip, sizeof(clip));
        const float act_min = clip[0], act_max = clip[1];
        for (uint64_t i = 0; i < elems; ++i) {
            float v = fp16_to_fp32(a16[i]) + fp16_to_fp32(b16[i]);
            if (v < act_min) v = act_min;
            if (v > act_max) v = act_max;
            out16[i] = fp32_to_fp16(v);
        }
        l1mgr.write(e.out_addr, out16.data(), elems * sizeof(uint16_t));
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            const EweBody& e = body.ewe;
            uint64_t elems = uint64_t(e.h) * e.w * e.c;
            const bool fp = is_fp_dtype(last_dtype);
            if (e.subtype == ES_ADD) {
                std::cout << "[EWE] add " << e.h << "x" << e.w << "x" << e.c
                          << "  dtype=" << (fp ? "fp" : "int") << "\n";
                // Note: 0 is a valid L1Mesh address (e.g. L1_WGT slot used for
                // input-B), so don't gate on non-zero addr; rely on subtype.
                if (elems > 0) {
                    if (fp) run_add_fp(e, elems);
                    else    run_add   (e, elems);
                }
                // Per-element pipelined across 16 lanes.
                wait((elems + 15) / 16, sc_core::SC_NS);
            } else {
                std::cout << "[EWE] softmax " << e.h << "x" << e.w << "x" << e.c << "\n";
                // v1: real LUT-based INT8 softmax (matches softmax_int8_ref in
                // scripts/compile_model.py byte-for-byte). 0 is a valid L1Mesh
                // address (v7 layout puts the softmax input there), so don't
                // gate on non-zero addr — guard on element count instead.
                if (elems > 0) {
                    std::vector<int8_t> in_buf(elems), out_buf(elems);
                    l1mgr.read(e.in_a_addr, in_buf.data(), elems);
                    softmax_int8(in_buf.data(), out_buf.data(), elems);
                    l1mgr.write(e.out_addr, out_buf.data(), elems);
                }
                // v2.2 compute cycles only (memory time already accrued inside
                // L1Mesh/Dram via the read/write calls above).
                //   3-pass schedule: exp / reduce_sum / div, 16 lanes pipelined.
                const uint64_t per_pass = (elems + 15) / 16;
                wait(3 * per_pass, sc_core::SC_NS);
            }
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end  .to_seconds() * 1e9));
            done_tag_out.write(0);
        }
    }
};

inline uint32_t pool_decode_stride(uint8_t enc) {
    return (enc & 1) ? 2 : 1;
}

SC_MODULE(PoolEngine) {
    sc_core::sc_fifo_in<DescriptorBody> cfg_in;
    sc_core::sc_fifo_out<uint8_t>       done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    uint8_t last_dtype = DT_INT8x8;            // v8.17: latched by CmdEng per dispatch

    SC_HAS_PROCESS(PoolEngine);
    PoolEngine(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) { SC_THREAD(run); }

    // v8.17: FP avg/max pool. Storage FP16 in L1; compute FP32 internally.
    // AVG sums in (kh, kw) order with running-FP32 add and divides by the
    // window count — same reduction order pool_fp_ref uses, so sim ≡ ref.
    void run_pool_fp(const PoolBody& p) {
        const uint32_t s_h = pool_decode_stride(p.stride & 0x3);
        const uint32_t s_w = pool_decode_stride((p.stride >> 2) & 0x3);
        const uint32_t pT  = p.pad_tb & 7;
        const uint32_t pL  = p.pad_lr & 7;
        const uint64_t in_elems  = uint64_t(p.in_h)  * p.in_w  * p.in_c;
        const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
        std::vector<uint16_t> in16(in_elems), out16(out_elems);
        l1mgr.read(p.in_addr, in16.data(), in_elems * sizeof(uint16_t));

        for (uint32_t oh = 0; oh < p.out_h; ++oh)
        for (uint32_t ow = 0; ow < p.out_w; ++ow)
        for (uint32_t c  = 0; c  < p.out_c; ++c) {
            if (p.mode == PM_MAX) {
                float best = -3.4e38f;
                for (uint32_t kh = 0; kh < p.k_h; ++kh)
                for (uint32_t kw = 0; kw < p.k_w; ++kw) {
                    int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                    int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                    if (ih < 0 || ih >= int(p.in_h)) continue;
                    if (iw < 0 || iw >= int(p.in_w)) continue;
                    float v = fp16_to_fp32(in16[(ih * p.in_w + iw) * p.in_c + c]);
                    if (v > best) best = v;
                }
                out16[(oh * p.out_w + ow) * p.out_c + c] = fp32_to_fp16(best);
            } else {                            // AVG / GLOBAL via avg
                float s = 0.0f; uint32_t n = 0;
                for (uint32_t kh = 0; kh < p.k_h; ++kh)
                for (uint32_t kw = 0; kw < p.k_w; ++kw) {
                    int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                    int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                    if (ih < 0 || ih >= int(p.in_h)) continue;
                    if (iw < 0 || iw >= int(p.in_w)) continue;
                    s += fp16_to_fp32(in16[(ih * p.in_w + iw) * p.in_c + c]);
                    ++n;
                }
                const uint32_t div = p.count_include_pad
                                   ? (uint32_t(p.k_h) * uint32_t(p.k_w))
                                   : (n ? n : 1);
                float v = s / float(div);
                out16[(oh * p.out_w + ow) * p.out_c + c] = fp32_to_fp16(v);
            }
        }
        l1mgr.write(p.out_addr, out16.data(), out_elems * sizeof(uint16_t));
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            const PoolBody& p = body.pool;

            const uint32_t s_h = pool_decode_stride(p.stride & 0x3);
            const uint32_t s_w = pool_decode_stride((p.stride >> 2) & 0x3);
            const uint32_t pT  = p.pad_tb & 7;
            const uint32_t pL  = p.pad_lr & 7;
            const char*    mode = (p.mode == PM_MAX) ? "max"
                                : (p.mode == PM_AVG) ? "avg" : "global";
            const bool fp = is_fp_dtype(last_dtype);

            std::cout << "[POOL] mode=" << mode
                      << "  dtype=" << (fp ? "fp" : "int")
                      << "  in=" << p.in_h << "x" << p.in_w << "x" << p.in_c
                      << "  k=" << int(p.k_h) << "x" << int(p.k_w)
                      << "  out=" << p.out_h << "x" << p.out_w << "x" << p.out_c
                      << "\n";

            if (fp) {
                run_pool_fp(p);
                // Cycle model: same per-output K_h*K_w lane occupancy as INT.
                const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
                const uint64_t per_lane  = (out_elems + 15) / 16;
                wait(per_lane * std::max<uint32_t>(p.k_h * p.k_w, 1), sc_core::SC_NS);
                const sc_core::sc_time t_end = sc_core::sc_time_stamp();
                busy_time += t_end - t_begin;
                tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                                   uint64_t(t_end  .to_seconds() * 1e9));
                done_tag_out.write(0);
                continue;
            }

            std::vector<int8_t> in_buf (uint64_t(p.in_h)  * p.in_w  * p.in_c);
            std::vector<int8_t> out_buf(uint64_t(p.out_h) * p.out_w * p.out_c);
            l1mgr.read(p.in_addr, in_buf.data(), in_buf.size());

            for (uint32_t oh = 0; oh < p.out_h; ++oh)
            for (uint32_t ow = 0; ow < p.out_w; ++ow)
            for (uint32_t c  = 0; c  < p.out_c; ++c) {
                if (p.mode == PM_MAX) {
                    int32_t best = -128;
                    for (uint32_t kh = 0; kh < p.k_h; ++kh)
                    for (uint32_t kw = 0; kw < p.k_w; ++kw) {
                        int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                        int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                        if (ih < 0 || ih >= int(p.in_h)) continue;
                        if (iw < 0 || iw >= int(p.in_w)) continue;
                        int32_t v = in_buf[(ih * p.in_w + iw) * p.in_c + c];
                        if (v > best) best = v;
                    }
                    out_buf[(oh * p.out_w + ow) * p.out_c + c] = int8_t(best);
                } else {                            // AVG (or GLOBAL via avg)
                    int32_t s = 0; uint32_t n = 0;
                    for (uint32_t kh = 0; kh < p.k_h; ++kh)
                    for (uint32_t kw = 0; kw < p.k_w; ++kw) {
                        int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                        int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                        if (ih < 0 || ih >= int(p.in_h)) continue;
                        if (iw < 0 || iw >= int(p.in_w)) continue;
                        s += in_buf[(ih * p.in_w + iw) * p.in_c + c]; ++n;
                    }
                    uint32_t div = p.count_include_pad ? (p.k_h * p.k_w)
                                                       : (n ? n : 1);
                    int32_t q = (s >= 0)
                              ?  ( s + int32_t(div) / 2) / int32_t(div)
                              : -((-s + int32_t(div) / 2) / int32_t(div));
                    if (q >  127) q =  127;
                    if (q < -128) q = -128;
                    out_buf[(oh * p.out_w + ow) * p.out_c + c] = int8_t(q);
                }
            }

            l1mgr.write(p.out_addr, out_buf.data(), out_buf.size());
            // v2.2 compute only (memory accounted in L1Mesh).
            // per-output-element work = K_h × K_w compares pipelined / 16 lanes.
            const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
            const uint64_t per_lane  = (out_elems + 15) / 16;
            wait(per_lane * std::max<uint32_t>(p.k_h * p.k_w, 1), sc_core::SC_NS);
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end  .to_seconds() * 1e9));
            done_tag_out.write(0);
        }
    }
};

} // namespace mdla7
