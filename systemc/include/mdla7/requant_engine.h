#pragma once

// Requant Engine — 16 per-cluster lanes (spec §3A.5).
//
// v1.2 / v1.3 + v3.5 fused activation + v6 bias/zp_in folding:
//   - Drains chain[oc % 16] in NHWC scan order (matching CONV's push order).
//   - Params table layout at scale_lut_addr (v6):
//        [ int32 zp_out | int32 act_min | int32 act_max
//        | int32 mult[OC] | int8 shift[OC] | int32 bias_eff[OC] ]
//   - Applies TFLite-style fixed-point requantization
//     (psum + bias_eff -> multiply_by_quantized_multiplier + zp + clamp).
//     bias_eff[oc] = bias[oc] - zp_in * sum_w[oc] is precomputed by
//     compile_model.py to absorb both real bias and the zp_in subtraction
//     so CONV cluster math stays a pure sum(in*w) -> int32 chain.
//   - act_min/act_max collapse FUSED_NONE/RELU/RELU6/RELU_N1_TO_1.
//   - Bit-exact against scripts/compile_model.py's numpy reference.

#include <systemc>
#include <array>
#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"
#include "mdla7/requant.h"
#include "mdla7/fp_utils.h"

namespace mdla7 {

SC_MODULE(RequantEngine) {
    sc_core::sc_fifo_in<DescriptorBody>           cfg_in;
    std::array<sc_core::sc_fifo<int32_t>*, 16>    chain_in;     // from CONV clusters
    sc_core::sc_fifo_out<uint8_t>                 done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    uint8_t last_dtype = DT_INT8x8;     // v4.1: latched by CmdEng each layer

    // v8.8: Requant lanes effectively fused into the CONV cluster output stage.
    // Spec §3A.5 has 16 clusters × 16 chain outputs each; once MBQM (or FP
    // bias+clamp) is integrated at every cluster's output, requant throughput
    // = 16 × 16 = 256 elem/cyc.  Bumping LANES from 16 to 256 captures this
    // fusion without restructuring the descriptor flow (CONV still pushes
    // through chain; engine just drains 16× faster).  Real HW would skip
    // chain entirely; the timing we measure is the same.
    static constexpr uint64_t LANES = 256;

    SC_HAS_PROCESS(RequantEngine);
    RequantEngine(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) {
        for (auto& p : chain_in) p = nullptr;
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            const RequantBody& r = body.requant;
            const uint32_t OH = r.h, OW = r.w, OC = r.c;
            const uint32_t OC_layer = r.scale_count ? r.scale_count : OC;   // params blob covers full layer
            const uint32_t oc_start = r.oc_start;
            const uint64_t total = uint64_t(OH) * OW * OC;
            std::cout << "[Requant] " << OH << "x" << OW << "x" << OC
                      << "  oc_start=" << oc_start << "/" << OC_layer
                      << "  scale_lut=0x" << std::hex << r.scale_lut_addr
                      << std::dec << "\n";

            // v8: FP path — no MBQM, just bit-cast chain → fp32, add bias, clip,
            // store as fp32. Params blob layout for FP layers:
            //   [ f32 act_min | f32 act_max | f32 bias[OC_layer] ] = 8 + 4*OC_layer bytes
            if (is_fp_dtype(last_dtype)) {
                float clip[2];
                std::vector<float> bias(OC);
                l1mgr.read(r.scale_lut_addr,                                       clip,        2 * sizeof(float));
                l1mgr.read(r.scale_lut_addr + 8 + 4 * oc_start,                    bias.data(), OC * sizeof(float));
                const float act_min = clip[0];
                const float act_max = clip[1];
                // v8.10: store FP16 (2 B/elem) — half the udma_w bandwidth of
                // the v8 FP32 path.  Internal compute stays FP32.
                std::vector<uint16_t> out_h16(total);
                uint64_t reads = 0;
                for (uint32_t oh = 0; oh < OH; ++oh)
                for (uint32_t ow = 0; ow < OW; ++ow)
                for (uint32_t oc = 0; oc < OC; ++oc) {
                    const int lane = oc & 0xF;
                    int32_t bits = chain_in[lane] ? chain_in[lane]->read() : 0;
                    float psum; std::memcpy(&psum, &bits, 4);
                    float v = psum + bias[oc];
                    if (v < act_min) v = act_min;
                    if (v > act_max) v = act_max;
                    out_h16[(oh * OW + ow) * OC + oc] = fp32_to_fp16(v);
                    if (++reads % LANES == 0) wait(1, sc_core::SC_NS);
                }
                l1mgr.write(r.out_addr, out_h16.data(), out_h16.size() * sizeof(uint16_t));
                // v8.6: pipeline overlap — see ConvEngine. The L1 write happens
                // concurrently with the LANES MBQM pipeline in real HW, so
                // total time = max(write_cyc, pipeline_cyc), not sum.
                {
                    const sc_core::sc_time elapsed = sc_core::sc_time_stamp() - t_begin;
                    const uint64_t elapsed_cyc = uint64_t(elapsed.to_seconds() * 1e9);
                    const uint64_t pipe = (total + LANES - 1) / LANES;
                    if (pipe > elapsed_cyc) wait(pipe - elapsed_cyc, sc_core::SC_NS);
                }
                const sc_core::sc_time t_end = sc_core::sc_time_stamp();
                busy_time += t_end - t_begin;
                tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                                   uint64_t(t_end  .to_seconds() * 1e9));
                done_tag_out.write(0);
                continue;
            }

            // Read per-channel requant params from L1Mesh. Params blob is
            // packed for the entire layer (scale_count = OC_layer); OC tiling
            // selects [oc_start .. oc_start+OC) without re-loading the blob.
            int32_t hdr[3] = {0, -128, 127};            // zp_out, act_min, act_max
            std::vector<int32_t> mult(OC);
            std::vector<int8_t>  shift(OC);
            std::vector<int32_t> bias_eff(OC);
            const uint32_t off_mult     = 12;
            const uint32_t off_shift    = off_mult + 4 * OC_layer;
            const uint32_t off_bias_eff = off_shift + OC_layer;
            l1mgr.read(r.scale_lut_addr,                                          hdr,             3 * sizeof(int32_t));
            l1mgr.read(r.scale_lut_addr + off_mult     + 4 * oc_start,            mult.data(),     OC * sizeof(int32_t));
            l1mgr.read(r.scale_lut_addr + off_shift    + 1 * oc_start,            shift.data(),    OC * sizeof(int8_t));
            l1mgr.read(r.scale_lut_addr + off_bias_eff + 4 * oc_start,            bias_eff.data(), OC * sizeof(int32_t));

            // v7: per-pixel correction (asymmetric uint8 weight zp). corr_addr=0
            // means none. Two layouts:
            //   per_oc=0: [OH_layer, OW_layer]            (non-DW conv; same value across OC)
            //   per_oc=1: [OH_layer, OW_layer, OC_layer]  (depthwise conv)
            // We read just the slice covering this dispatch (oh_start.. + this h, full OW_layer).
            const bool      have_corr   = (r.corr_addr != 0);
            const uint32_t  OW_layer    = r.out_w_layer ? r.out_w_layer : OW;
            const uint32_t  oh_start    = r.oh_start;
            const bool      corr_per_oc = (r.corr_per_oc != 0);
            std::vector<int32_t> corr;          // sliced for this tile
            if (have_corr) {
                const uint64_t row_elems = corr_per_oc
                    ? uint64_t(OW_layer) * OC_layer
                    : uint64_t(OW_layer);
                const uint64_t slice_elems = row_elems * OH;
                corr.resize(slice_elems);
                const uint32_t addr =
                    r.corr_addr + uint32_t(4 * row_elems * oh_start);
                l1mgr.read(addr, corr.data(), uint32_t(slice_elems * sizeof(int32_t)));
            }
            const int32_t zp_out  = hdr[0];
            const int32_t act_min = hdr[1];
            const int32_t act_max = hdr[2];

            // v4.1: int16 path widens the saturation range and writes 2 bytes/elem.
            const bool int16_out = (last_dtype == DT_INT16x16);
            const int32_t lo = int16_out ? -32768 : -128;
            const int32_t hi = int16_out ?  32767 :  127;
            int32_t a_min = std::max(act_min, lo);
            int32_t a_max = std::min(act_max, hi);

            // Drain chains in NHWC scan order; per-channel requant + activation clamp.
            std::vector<int16_t> out16(int16_out ? total : 0);
            std::vector<int8_t>  out8 (int16_out ? 0     : total);
            // v8.7 + v8.8: chain backpressure at LANES throughput.
            uint64_t reads = 0;
            for (uint32_t oh = 0; oh < OH; ++oh)
            for (uint32_t ow = 0; ow < OW; ++ow)
            for (uint32_t oc = 0; oc < OC; ++oc) {
                const int lane = oc & 0xF;
                int32_t psum   = chain_in[lane] ? chain_in[lane]->read() : 0;
                // v6: fold bias + zp_in subtraction into psum (precomputed by compile_model).
                int64_t with_bias = int64_t(psum) + int64_t(bias_eff[oc]);
                // v7: per-pixel asymmetric-uint8 correction (zp_w * Σ_window(in) term).
                if (have_corr) {
                    int64_t corr_val;
                    if (corr_per_oc) {
                        corr_val = corr[(oh * OW_layer + ow) * OC_layer
                                        + oc + oc_start];
                    } else {
                        corr_val = corr[oh * OW_layer + ow];
                    }
                    with_bias += corr_val;
                }
                int32_t psum_b = (with_bias >  INT32_MAX) ? INT32_MAX :
                                 (with_bias <  INT32_MIN) ? INT32_MIN : int32_t(with_bias);
                int32_t scaled = multiply_by_quantized_multiplier(
                                    psum_b, mult[oc], int(shift[oc]));
                int32_t v = scaled + zp_out;
                if (v < a_min) v = a_min;
                if (v > a_max) v = a_max;
                const size_t idx = (oh * OW + ow) * OC + oc;
                if (int16_out) out16[idx] = int16_t(v);
                else           out8 [idx] = int8_t (v);
                if (++reads % LANES == 0) wait(1, sc_core::SC_NS);
            }
            if (int16_out) l1mgr.write(r.out_addr, out16.data(), out16.size() * 2);
            else           l1mgr.write(r.out_addr, out8 .data(), out8 .size());

            // v8.6 + v8.8: pipeline overlaps with L1 write at LANES throughput.
            const sc_core::sc_time elapsed = sc_core::sc_time_stamp() - t_begin;
            const uint64_t elapsed_cyc = uint64_t(elapsed.to_seconds() * 1e9);
            const uint64_t pipe = (total + LANES - 1) / LANES;
            if (pipe > elapsed_cyc) wait(pipe - elapsed_cyc, sc_core::SC_NS);
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end  .to_seconds() * 1e9));
            done_tag_out.write(0);
        }
    }
};

} // namespace mdla7
