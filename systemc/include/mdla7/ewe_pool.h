#pragma once

// EWE + POOL Engine — see spec §3A.7
//
// v1: real INT8 compute (bit-exact vs reference).
// v2.2 cycle model:
//   POOL: dtype-scaled pipelined engine:
//         INT8=64 lanes, INT16=32 lanes, FP=32 lanes.
//         cycle = ceil(out_elem/lanes) * max(K_h*K_w, 1)
//         + AXI fill (length / 128b) + drain (16 cycle).
//   EWE: dtype-scaled element-wise engine:
//        INT8=64 lanes, INT16=32 lanes, FP=32 lanes.
//   EWE softmax: 3-pass exp+reduce_sum+div;
//         cycle = ceil(elem/lanes) * 3 + AXI fill/drain.

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

static constexpr uint64_t EWE_INT8_LANES  = 64;
static constexpr uint64_t EWE_INT16_LANES = 32;
static constexpr uint64_t EWE_FP_LANES    = 32;

inline uint64_t ewe_lanes_for_dtype(uint8_t dtype) {
    if (is_fp_dtype(dtype)) return EWE_FP_LANES;
    switch (static_cast<DType>(dtype)) {
        case DT_INT16x4:
        case DT_INT16x8:
        case DT_INT16x16:
            return EWE_INT16_LANES;
        case DT_INT8x4:
        case DT_INT8x8:
        default:
            return EWE_INT8_LANES;
    }
}

inline uint64_t pool_lanes_for_dtype(uint8_t dtype) {
    return ewe_lanes_for_dtype(dtype);
}

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
    // v8.30: `op` selects ADD (0), MUL (1), or SUB (2) — same param layout
    // and reduction order. SUB is `a - b`; TFLite has no SUB activation so
    // the clamp is the same ±sentinel as the others.
    void run_binary_fp(const EweBody& e, uint64_t elems, uint8_t op) {
        std::vector<uint16_t> a16(elems), b16(elems), out16(elems);
        l1mgr.read(e.in_a_addr, a16.data(), elems * sizeof(uint16_t));
        l1mgr.read(e.in_b_addr, b16.data(), elems * sizeof(uint16_t));
        float clip[2];
        l1mgr.read(e.lut_addr, clip, sizeof(clip));
        const float act_min = clip[0], act_max = clip[1];
        for (uint64_t i = 0; i < elems; ++i) {
            const float a = fp16_to_fp32(a16[i]);
            const float b = fp16_to_fp32(b16[i]);
            float v = (op == 1) ? (a * b)
                    : (op == 2) ? (a - b)
                                : (a + b);
            if (v < act_min) v = act_min;
            if (v > act_max) v = act_max;
            out16[i] = fp32_to_fp16(v);
        }
        l1mgr.write(e.out_addr, out16.data(), elems * sizeof(uint16_t));
    }

    // v8.30: TFLite int8 MUL.
    //   raw   = (a - zp_a) * (b - zp_b)     // int32, fits int48
    //   out_q = MBQM(raw, mq, sh) + zp_o    // single multiplier
    //   out   = clip(out_q, [act_min, act_max])
    // Params blob layout (12 int32, mirrors run_add):
    //   [ zp_a | zp_b | zp_o | mq_a | sh_a (unused) | mq_b | sh_b (unused)
    //   | mq_o (= MUL multiplier) | sh_o | left_shift (unused) | act_min | act_max ]
    void run_mul(const EweBody& e, uint64_t elems) {
        std::vector<int8_t> a_buf(elems), b_buf(elems), out_buf(elems);
        l1mgr.read(e.in_a_addr, a_buf.data(), elems);
        l1mgr.read(e.in_b_addr, b_buf.data(), elems);
        int32_t p[12];
        l1mgr.read(e.lut_addr, p, sizeof(p));
        const int32_t zp_a = p[0], zp_b = p[1], zp_out = p[2];
        const int32_t mult_o = p[7], shift_o = p[8];
        const int32_t act_min = p[10], act_max = p[11];
        for (uint64_t i = 0; i < elems; ++i) {
            const int32_t a = int32_t(a_buf[i]) - zp_a;
            const int32_t b = int32_t(b_buf[i]) - zp_b;
            int32_t v = multiply_by_quantized_multiplier(a * b, mult_o, shift_o) + zp_out;
            if (v < act_min) v = act_min;
            if (v > act_max) v = act_max;
            out_buf[i] = int8_t(v);
        }
        l1mgr.write(e.out_addr, out_buf.data(), elems);
    }

    // v8.30: TFLite int8 SUB. Same shape as run_add but with sb negated.
    //   sa = MBQM((a - zp_a) << ls, mult_a, shift_a)
    //   sb = MBQM((b - zp_b) << ls, mult_b, shift_b)
    //   out = clip( MBQM(sa - sb, mult_o, shift_o) + zp_o, [act_min, act_max] )
    void run_sub(const EweBody& e, uint64_t elems) {
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
            int32_t s  = sa - sb;
            int32_t v  = multiply_by_quantized_multiplier(s, mult_o, shift_o) + zp_out;
            if (v < act_min) v = act_min;
            if (v > act_max) v = act_max;
            out_buf[i] = int8_t(v);
        }
        l1mgr.write(e.out_addr, out_buf.data(), elems);
    }

    // v8.30: FP unary activations (HARD_SWISH / GELU). Storage FP16; compute FP32.
    // HARD_SWISH: y = x * relu6(x + 3) / 6  (TFLite spec exactly).
    // GELU: tanh-approximation,
    //   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // numpy ref must mirror the same loop order + std::tanhf invocation to
    // stay byte-identical against numpy.tanh on FP32 input.
    void run_unary_fp(const EweBody& e, uint64_t elems, uint8_t subtype) {
        std::vector<uint16_t> in16(elems), out16(elems);
        l1mgr.read(e.in_a_addr, in16.data(), elems * sizeof(uint16_t));
        float clip[2];
        l1mgr.read(e.lut_addr, clip, sizeof(clip));
        const float act_min = clip[0], act_max = clip[1];
        constexpr float k = 0.7978845608028654f;        // sqrt(2/pi)
        constexpr float c = 0.044715f;
        for (uint64_t i = 0; i < elems; ++i) {
            const float x = fp16_to_fp32(in16[i]);
            float y;
            if (subtype == ES_GELU) {
                const float u = k * (x + c * x * x * x);
                y = 0.5f * x * (1.0f + std::tanh(u));
            } else {                                    // ES_HARD_SWISH
                float r = x + 3.0f;
                if (r < 0.0f) r = 0.0f;
                if (r > 6.0f) r = 6.0f;
                y = x * r / 6.0f;
            }
            if (y < act_min) y = act_min;
            if (y > act_max) y = act_max;
            out16[i] = fp32_to_fp16(y);
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
            const uint64_t lanes = ewe_lanes_for_dtype(last_dtype);
            // v8.30: binary ADD/MUL/SUB share the same dispatch shape (two
            // input tensors + 48-byte params) — only the math differs. Unary
            // HARD_SWISH/GELU mirror the softmax shape (single input + 8-byte
            // clamp params) but use compute_fp-style nonlinearities.
            if (e.subtype == ES_ADD || e.subtype == ES_MUL || e.subtype == ES_SUB) {
                const char* nm = (e.subtype == ES_MUL) ? "mul"
                               : (e.subtype == ES_SUB) ? "sub" : "add";
                std::cout << "[EWE] " << nm << " " << e.h << "x" << e.w << "x" << e.c
                          << "  dtype=" << (fp ? "fp" : "int") << "\n";
                if (elems > 0) {
                    if (fp) {
                        const uint8_t op = (e.subtype == ES_MUL) ? 1
                                         : (e.subtype == ES_SUB) ? 2 : 0;
                        run_binary_fp(e, elems, op);
                    } else {
                        if      (e.subtype == ES_MUL) run_mul(e, elems);
                        else if (e.subtype == ES_SUB) run_sub(e, elems);
                        else                          run_add(e, elems);
                    }
                }
                wait((elems + lanes - 1) / lanes, sc_core::SC_NS);
            } else if (e.subtype == ES_HARD_SWISH || e.subtype == ES_GELU) {
                const char* nm = (e.subtype == ES_GELU) ? "gelu" : "h_swsh";
                std::cout << "[EWE] " << nm << " " << e.h << "x" << e.w << "x" << e.c
                          << "  dtype=" << (fp ? "fp" : "int") << "\n";
                if (elems > 0 && fp) {
                    run_unary_fp(e, elems, e.subtype);
                }
                // GELU = exp + tanh + arith (~6 ops); HARD_SWISH ~ 4 ops.
                // Lump under 1 cycle/elem/lane (matches softmax exp pass).
                wait((elems + lanes - 1) / lanes, sc_core::SC_NS);
            } else {
                std::cout << "[EWE] softmax " << e.h << "x" << e.w << "x" << e.c
                          << "  dtype=" << (fp ? "fp" : "int") << "\n";
                if (elems > 0) {
                    if (fp) {
                        // v8.28: FP softmax. Numerically-stable 3-pass form:
                        // (1) max-reduce, (2) exp(x - max) → running FP32 sum,
                        // (3) divide by sum and cast back to FP16. Sum is a
                        // sequential running-add to match `softmax_fp_ref` in
                        // compile_model.py byte for byte.
                        std::vector<uint16_t> in16(elems), out16(elems);
                        l1mgr.read(e.in_a_addr, in16.data(), elems * sizeof(uint16_t));
                        float max_v = -3.4e38f;
                        for (uint64_t k = 0; k < elems; ++k) {
                            const float v = fp16_to_fp32(in16[k]);
                            if (v > max_v) max_v = v;
                        }
                        std::vector<float> exp_v(elems);
                        for (uint64_t k = 0; k < elems; ++k)
                            exp_v[k] = std::exp(fp16_to_fp32(in16[k]) - max_v);
                        float s = 0.0f;
                        for (uint64_t k = 0; k < elems; ++k) s += exp_v[k];
                        if (s == 0.0f) s = 1.0f;
                        for (uint64_t k = 0; k < elems; ++k)
                            out16[k] = fp32_to_fp16(exp_v[k] / s);
                        l1mgr.write(e.out_addr, out16.data(), elems * sizeof(uint16_t));
                    } else {
                        // v1: real LUT-based INT8 softmax (matches
                        // softmax_int8_ref in compile_model.py byte for byte).
                        std::vector<int8_t> in_buf(elems), out_buf(elems);
                        l1mgr.read(e.in_a_addr, in_buf.data(), elems);
                        softmax_int8(in_buf.data(), out_buf.data(), elems);
                        l1mgr.write(e.out_addr, out_buf.data(), elems);
                    }
                }
                // v2.2 compute cycles only (memory time already accrued inside
                // L1Mesh/Dram via the read/write calls above).
                //   3-pass schedule: exp / reduce_sum / div, pipelined lanes.
                const uint64_t per_pass = (elems + lanes - 1) / lanes;
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
    // v8.23: 2-bit log2 encoding (matches ConvEngine's decode_stride):
    // 0→1, 1→2, 2→4, 3→8. Old form returned only 1 or 2, so deeplab's 8x8
    // global avgpool with stride=8 silently degraded to stride=1.
    static const uint32_t lut[4] = {1, 2, 4, 8};
    return lut[enc & 0x3];
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
        const uint32_t k_h = (p.k_h == 255) ? uint32_t(p.in_h) : uint32_t(p.k_h);
        const uint32_t k_w = (p.k_w == 255) ? uint32_t(p.in_w) : uint32_t(p.k_w);
        const uint64_t in_elems  = uint64_t(p.in_h)  * p.in_w  * p.in_c;
        const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
        std::vector<uint16_t> in16(in_elems), out16(out_elems);
        l1mgr.read(p.in_addr, in16.data(), in_elems * sizeof(uint16_t));

        for (uint32_t oh = 0; oh < p.out_h; ++oh)
        for (uint32_t ow = 0; ow < p.out_w; ++ow)
        for (uint32_t c  = 0; c  < p.out_c; ++c) {
            if (p.mode == PM_MAX) {
                float best = -3.4e38f;
                for (uint32_t kh = 0; kh < k_h; ++kh)
                for (uint32_t kw = 0; kw < k_w; ++kw) {
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
                for (uint32_t kh = 0; kh < k_h; ++kh)
                for (uint32_t kw = 0; kw < k_w; ++kw) {
                    int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                    int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                    if (ih < 0 || ih >= int(p.in_h)) continue;
                    if (iw < 0 || iw >= int(p.in_w)) continue;
                    s += fp16_to_fp32(in16[(ih * p.in_w + iw) * p.in_c + c]);
                    ++n;
                }
                const uint32_t div = p.count_include_pad
                                   ? (k_h * k_w)
                                   : (n ? n : 1);
                float v = s / float(div);
                out16[(oh * p.out_w + ow) * p.out_c + c] = fp32_to_fp16(v);
            }
        }
        l1mgr.write(p.out_addr, out16.data(), out_elems * sizeof(uint16_t));
    }

    template <typename T>
    void run_pool_int(const PoolBody& p) {
        const uint32_t s_h = pool_decode_stride(p.stride & 0x3);
        const uint32_t s_w = pool_decode_stride((p.stride >> 2) & 0x3);
        const uint32_t pT  = p.pad_tb & 7;
        const uint32_t pL  = p.pad_lr & 7;
        const uint32_t k_h = (p.k_h == 255) ? uint32_t(p.in_h) : uint32_t(p.k_h);
        const uint32_t k_w = (p.k_w == 255) ? uint32_t(p.in_w) : uint32_t(p.k_w);
        const uint64_t in_elems  = uint64_t(p.in_h)  * p.in_w  * p.in_c;
        const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
        const int32_t min_v = (sizeof(T) == 2) ? -32768 : -128;
        const int32_t max_v = (sizeof(T) == 2) ?  32767 :  127;
        std::vector<T> in_buf(in_elems), out_buf(out_elems);
        l1mgr.read(p.in_addr, in_buf.data(), in_elems * sizeof(T));

        for (uint32_t oh = 0; oh < p.out_h; ++oh)
        for (uint32_t ow = 0; ow < p.out_w; ++ow)
        for (uint32_t c  = 0; c  < p.out_c; ++c) {
            if (p.mode == PM_MAX) {
                int32_t best = min_v;
                for (uint32_t kh = 0; kh < k_h; ++kh)
                for (uint32_t kw = 0; kw < k_w; ++kw) {
                    int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                    int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                    if (ih < 0 || ih >= int(p.in_h)) continue;
                    if (iw < 0 || iw >= int(p.in_w)) continue;
                    int32_t v = int32_t(in_buf[(ih * p.in_w + iw) * p.in_c + c]);
                    if (v > best) best = v;
                }
                out_buf[(oh * p.out_w + ow) * p.out_c + c] = T(best);
            } else {                            // AVG (or GLOBAL via avg)
                int32_t s = 0; uint32_t n = 0;
                for (uint32_t kh = 0; kh < k_h; ++kh)
                for (uint32_t kw = 0; kw < k_w; ++kw) {
                    int ih = int(oh)*int(s_h) + int(kh) - int(pT);
                    int iw = int(ow)*int(s_w) + int(kw) - int(pL);
                    if (ih < 0 || ih >= int(p.in_h)) continue;
                    if (iw < 0 || iw >= int(p.in_w)) continue;
                    s += int32_t(in_buf[(ih * p.in_w + iw) * p.in_c + c]); ++n;
                }
                uint32_t div = p.count_include_pad ? (k_h * k_w)
                                                   : (n ? n : 1);
                int32_t q = (s >= 0)
                          ?  ( s + int32_t(div) / 2) / int32_t(div)
                          : -((-s + int32_t(div) / 2) / int32_t(div));
                if (q > max_v) q = max_v;
                if (q < min_v) q = min_v;
                out_buf[(oh * p.out_w + ow) * p.out_c + c] = T(q);
            }
        }

        l1mgr.write(p.out_addr, out_buf.data(), out_elems * sizeof(T));
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            const PoolBody& p = body.pool;

            const uint32_t k_h = (p.k_h == 255) ? uint32_t(p.in_h) : uint32_t(p.k_h);
            const uint32_t k_w = (p.k_w == 255) ? uint32_t(p.in_w) : uint32_t(p.k_w);
            const char*    mode = (p.mode == PM_MAX) ? "max"
                                : (p.mode == PM_AVG) ? "avg" : "global";
            const bool fp = is_fp_dtype(last_dtype);
            const bool int16 = (last_dtype == DT_INT16x4
                             || last_dtype == DT_INT16x8
                             || last_dtype == DT_INT16x16);
            const uint64_t lanes = pool_lanes_for_dtype(last_dtype);

            std::cout << "[POOL] mode=" << mode
                      << "  dtype=" << (fp ? "fp" : "int")
                      << "  in=" << p.in_h << "x" << p.in_w << "x" << p.in_c
                      << "  k=" << k_h << "x" << k_w
                      << "  out=" << p.out_h << "x" << p.out_w << "x" << p.out_c
                      << "\n";

            if (fp) {
                run_pool_fp(p);
                // Cycle model: same per-output K_h*K_w lane occupancy as INT.
                const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
                const uint64_t per_lane  = (out_elems + lanes - 1) / lanes;
                wait(per_lane * std::max<uint32_t>(k_h * k_w, 1), sc_core::SC_NS);
                const sc_core::sc_time t_end = sc_core::sc_time_stamp();
                busy_time += t_end - t_begin;
                tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                                   uint64_t(t_end  .to_seconds() * 1e9));
                done_tag_out.write(0);
                continue;
            }

            if (int16) run_pool_int<int16_t>(p);
            else       run_pool_int<int8_t>(p);
            // v2.2 compute only (memory accounted in L1Mesh).
            // per-output-element work = K_h × K_w compares pipelined / lanes.
            const uint64_t out_elems = uint64_t(p.out_h) * p.out_w * p.out_c;
            const uint64_t per_lane  = (out_elems + lanes - 1) / lanes;
            wait(per_lane * std::max<uint32_t>(k_h * k_w, 1), sc_core::SC_NS);
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end  .to_seconds() * 1e9));
            done_tag_out.write(0);
        }
    }
};

} // namespace mdla7
