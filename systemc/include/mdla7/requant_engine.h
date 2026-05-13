#pragma once

// Requant Engine — shared CONV/EWE quantize-pack resource (spec §3A.5).
//
// v1.2 / v1.3 + v3.5 fused activation + v6 bias/zp_in folding:
//   - Drains chain[oc % CONV_REQUANT_CHAIN_LANES] in NHWC scan order
//     (matching CONV's push order).
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
    std::array<sc_core::sc_fifo<int32_t>*,
               CONV_REQUANT_CHAIN_LANES>         chain_in;     // from CONV clusters
    sc_core::sc_fifo_out<uint8_t>                 done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    std::vector<RtlPhaseTrace> last_rtl_phases;
    std::vector<std::vector<RtlPhaseTrace>> rtl_phase_tasks;
    uint8_t last_dtype = DT_INT8x8;     // v4.1: latched by CmdEng each layer
    EngineModel engine_model = EngineModel::Rtl;

    // v8.38: Requant is modeled as the shared CONV/EWE quantize-pack resource.
    // The CONV->Requant input chain is 128 INT32/cyc = 4096 bit/cyc, while the
    // downstream MBQM/clamp/pack resource remains 512 elem/cyc.
    static constexpr uint64_t CHAIN_LANES = CONV_REQUANT_CHAIN_LANES;
    static constexpr uint64_t PACK_LANES = 512;

    SC_HAS_PROCESS(RequantEngine);
    RequantEngine(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) {
        for (auto& p : chain_in) p = nullptr;
        SC_THREAD(run);
    }

    static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
        return b ? ((a + b - 1) / b) : 0;
    }

    uint64_t out_elem_bytes() const {
        return (is_fp_dtype(last_dtype) ||
                last_dtype == DT_INT16x8 ||
                last_dtype == DT_INT16x16) ? 2u : 1u;
    }

    void rtl_record_phase(const char* name, uint64_t cycles,
                          uint64_t read_bytes = 0, uint64_t write_bytes = 0,
                          uint64_t elems = 0, uint64_t lanes = 0,
                          const char* stall = "") {
        RtlPhaseTrace phase;
        phase.name = name;
        phase.cycles = cycles;
        phase.read_bytes = read_bytes;
        phase.write_bytes = write_bytes;
        phase.elems = elems;
        phase.lanes = lanes;
        phase.stall = stall ? stall : "";
        last_rtl_phases.push_back(phase);
    }

    void rtl_record_requant_transaction(uint64_t total, uint32_t oc,
                                        uint32_t oc_layer, bool fp,
                                        bool have_corr, bool skip_l1_write,
                                        uint64_t write_bytes) {
        const uint64_t param_bytes = fp
            ? (8 + uint64_t(oc) * sizeof(float))
            : (12 + uint64_t(oc) * (sizeof(int32_t) + sizeof(int8_t) + sizeof(int32_t)));
        const uint64_t corr_bytes = have_corr ? total * sizeof(int32_t) : 0;
        const uint64_t read_bytes = param_bytes + corr_bytes;
        last_rtl_phases.clear();
        rtl_record_phase("issue", 5);
        rtl_record_phase("param_read",
                         ceil_div_u64(read_bytes, PayloadPortCount::REQUANT_R * PAYLOAD_BYTES),
                         read_bytes, 0, oc_layer, PayloadPortCount::REQUANT_R,
                         have_corr ? "params_corr_read" : "params_read");
        rtl_record_phase("chain_read", ceil_div_u64(total, CHAIN_LANES),
                         0, 0, total, CHAIN_LANES, "conv_chain");
        rtl_record_phase("pack", ceil_div_u64(total, PACK_LANES),
                         0, 0, total, PACK_LANES,
                         fp ? "fp_clip_pack" : "mbqm_pack");
        if (!skip_l1_write) {
            rtl_record_phase("write",
                             ceil_div_u64(write_bytes, PayloadPortCount::REQUANT_W * PAYLOAD_BYTES),
                             0, write_bytes, total, PayloadPortCount::REQUANT_W,
                             "payload_write");
        }
        rtl_record_phase("done", 3);
    }

    void synth_record_requant_transaction(uint64_t total, uint32_t oc,
                                          uint32_t oc_layer, bool fp,
                                          bool have_corr, bool skip_l1_write,
                                          uint64_t write_bytes) {
        const uint64_t param_bytes = fp
            ? (8 + uint64_t(oc) * sizeof(float))
            : (12 + uint64_t(oc) * (sizeof(int32_t) + sizeof(int8_t) + sizeof(int32_t)));
        const uint64_t corr_bytes = have_corr ? total * sizeof(int32_t) : 0;
        const uint64_t read_bytes = param_bytes + corr_bytes;
        last_rtl_phases.clear();
        rtl_record_phase("cfg_decode", 2, 0, 0, 0, 0, "synth_cfg");
        rtl_record_phase("param_fetch",
                         ceil_div_u64(read_bytes, PayloadPortCount::REQUANT_R * PAYLOAD_BYTES) + 1,
                         read_bytes, 0, oc_layer, PayloadPortCount::REQUANT_R,
                         have_corr ? "synth_params_corr" : "synth_params");
        rtl_record_phase("chain_sync", ceil_div_u64(total, CHAIN_LANES) + 1,
                         0, 0, total, CHAIN_LANES, "synth_conv_chain");
        rtl_record_phase("quant_pipe", ceil_div_u64(total, PACK_LANES) + 2,
                         0, 0, total, PACK_LANES,
                         fp ? "synth_fp_clip_pack" : "synth_mbqm_pack");
        if (!skip_l1_write) {
            rtl_record_phase("payload_write",
                             ceil_div_u64(write_bytes, PayloadPortCount::REQUANT_W * PAYLOAD_BYTES) + 1,
                             0, write_bytes, total, PayloadPortCount::REQUANT_W,
                             "synth_payload_write");
        }
        rtl_record_phase("retire", 1, 0, 0, 0, 0, "synth_done");
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
            const bool skip_l1_write = (r._r[0] == RQ_STORE_SKIP);
            const bool d2s_store_req = (r._r[0] == RQ_STORE_D2SPACE);
            const bool strided_store_req = (r._r[0] == RQ_STORE_STRIDED_2D);
            const uint32_t d2s_block = d2s_store_req ? std::max<uint32_t>(1, r._r[1]) : 1;
            const uint32_t d2s_out_c = d2s_store_req
                ? uint32_t(r._r[2] | (uint16_t(r._r[3]) << 8))
                : OC;
            const uint32_t strided_dst_row = strided_store_req
                ? uint32_t(r._r[1] | (uint16_t(r._r[2]) << 8))
                : 0;
            const uint32_t strided_dst_col = strided_store_req
                ? uint32_t(r._r[3] | (uint16_t(r._r[4]) << 8))
                : 0;
            const bool d2s_store =
                d2s_store_req && d2s_out_c && OC == d2s_out_c * d2s_block * d2s_block;
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
                // v10: direct CONV->D2SPACE store uses the same output swizzle
                // as the integer path, only with FP16 payloads.
                const uint64_t dst_total = d2s_store
                    ? uint64_t(OH) * d2s_block * OW * d2s_block * d2s_out_c
                    : total;
                std::vector<uint16_t> out_h16(dst_total);
                uint64_t reads = 0;
                for (uint32_t oh = 0; oh < OH; ++oh)
                for (uint32_t ow = 0; ow < OW; ++ow)
                for (uint32_t oc = 0; oc < OC; ++oc) {
                    const std::size_t lane = oc % CONV_REQUANT_CHAIN_LANES;
                    int32_t bits = chain_in[lane] ? chain_in[lane]->read() : 0;
                    float psum; std::memcpy(&psum, &bits, 4);
                    float v = psum + bias[oc];
                    if (v < act_min) v = act_min;
                    if (v > act_max) v = act_max;
                    size_t idx = (oh * OW + ow) * OC + oc;
                    if (d2s_store) {
                        const uint32_t q = d2s_out_c ? (oc / d2s_out_c) : 0;
                        const uint32_t real_oc = d2s_out_c ? (oc % d2s_out_c) : oc;
                        const uint32_t bh = d2s_block ? (q / d2s_block) : 0;
                        const uint32_t bw = d2s_block ? (q % d2s_block) : 0;
                        const uint32_t out_h = oh * d2s_block + bh;
                        const uint32_t out_w = ow * d2s_block + bw;
                        idx = (uint64_t(out_h) * (OW * d2s_block) + out_w) *
                              d2s_out_c + real_oc;
                    }
                    out_h16[idx] = fp32_to_fp16(v);
                    if (++reads % CHAIN_LANES == 0) wait(1, sc_core::SC_NS);
                }
                l1mgr.write(r.out_addr, out_h16.data(), out_h16.size() * sizeof(uint16_t));
                // v8.6: pipeline overlap — see ConvEngine. The L1 write happens
                // concurrently with the PACK_LANES MBQM pipeline in real HW, so
                // total time = max(write_cyc, pipeline_cyc), not sum.
                {
                    const sc_core::sc_time elapsed = sc_core::sc_time_stamp() - t_begin;
                    const uint64_t elapsed_cyc = uint64_t(elapsed.to_seconds() * 1e9);
                    const uint64_t pipe = (total + PACK_LANES - 1) / PACK_LANES;
                    if (is_synth_style(engine_model))
                        synth_record_requant_transaction(total, OC, OC_layer, true,
                                                         false, false,
                                                         out_h16.size() * sizeof(uint16_t));
                    else if (is_rtl_style(engine_model))
                        rtl_record_requant_transaction(total, OC, OC_layer, true,
                                                       false, false,
                                                       out_h16.size() * sizeof(uint16_t));
                    if (pipe > elapsed_cyc) wait(pipe - elapsed_cyc, sc_core::SC_NS);
                }
                const sc_core::sc_time t_end = sc_core::sc_time_stamp();
                busy_time += t_end - t_begin;
                tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                                   uint64_t(t_end  .to_seconds() * 1e9));
                rtl_phase_tasks.push_back(is_rtl_style(engine_model)
                                          ? last_rtl_phases
                                          : std::vector<RtlPhaseTrace>{});
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
            // v8.27: INT16x8 hybrid also produces int16 output.
            const bool int16_out = (last_dtype == DT_INT16x16
                                    || last_dtype == DT_INT16x8);
            const int32_t lo = int16_out ? -32768 : -128;
            const int32_t hi = int16_out ?  32767 :  127;
            int32_t a_min = std::max(act_min, lo);
            int32_t a_max = std::min(act_max, hi);

            // Drain chains in NHWC scan order; per-channel requant + activation clamp.
            const uint64_t dst_total = d2s_store
                ? uint64_t(OH) * d2s_block * OW * d2s_block * d2s_out_c
                : total;
            std::vector<int16_t> out16((!skip_l1_write && int16_out) ? dst_total : 0);
            std::vector<int8_t>  out8 ((!skip_l1_write && !int16_out) ? dst_total : 0);
            // v8.7 + v8.8: chain backpressure at CONV_REQUANT_CHAIN_LANES throughput.
            uint64_t reads = 0;
            for (uint32_t oh = 0; oh < OH; ++oh)
            for (uint32_t ow = 0; ow < OW; ++ow)
            for (uint32_t oc = 0; oc < OC; ++oc) {
                const std::size_t lane = oc % CONV_REQUANT_CHAIN_LANES;
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
                if (!skip_l1_write) {
                    size_t idx = (oh * OW + ow) * OC + oc;
                    if (d2s_store) {
                        const uint32_t q = d2s_out_c ? (oc / d2s_out_c) : 0;
                        const uint32_t real_oc = d2s_out_c ? (oc % d2s_out_c) : oc;
                        const uint32_t bh = d2s_block ? (q / d2s_block) : 0;
                        const uint32_t bw = d2s_block ? (q % d2s_block) : 0;
                        const uint32_t out_h = oh * d2s_block + bh;
                        const uint32_t out_w = ow * d2s_block + bw;
                        idx = (uint64_t(out_h) * (OW * d2s_block) + out_w) * d2s_out_c + real_oc;
                    }
                    if (int16_out) out16[idx] = int16_t(v);
                    else           out8 [idx] = int8_t (v);
                }
                if (++reads % CHAIN_LANES == 0) wait(1, sc_core::SC_NS);
            }
            if (!skip_l1_write) {
                if (strided_store_req && !d2s_store) {
                    const uint32_t elem = int16_out ? 2u : 1u;
                    const uint32_t rows = OH * OW;
                    const uint32_t src_row = OC * elem;
                    if (strided_dst_row && strided_dst_col + src_row <= strided_dst_row) {
                        if (int16_out) {
                            l1mgr.wait_ticket(l1mgr.write_strided_rows(
                                r.out_addr, out16.data(), rows, src_row,
                                strided_dst_row, strided_dst_col));
                        } else {
                            l1mgr.wait_ticket(l1mgr.write_strided_rows(
                                r.out_addr, out8.data(), rows, src_row,
                                strided_dst_row, strided_dst_col));
                        }
                    } else {
                        SC_REPORT_ERROR("RequantEngine", "invalid strided store layout");
                    }
                } else if (int16_out) {
                    l1mgr.write(r.out_addr, out16.data(), out16.size() * 2);
                } else {
                    l1mgr.write(r.out_addr, out8 .data(), out8 .size());
                }
            }

            // v8.6 + v8.8: pipeline overlaps with L1 write at PACK_LANES throughput.
            const sc_core::sc_time elapsed = sc_core::sc_time_stamp() - t_begin;
            const uint64_t elapsed_cyc = uint64_t(elapsed.to_seconds() * 1e9);
            const uint64_t pipe = (total + PACK_LANES - 1) / PACK_LANES;
            const uint64_t write_bytes = skip_l1_write ? 0 : dst_total * out_elem_bytes();
            if (is_synth_style(engine_model))
                synth_record_requant_transaction(total, OC, OC_layer, false,
                                                 have_corr, skip_l1_write, write_bytes);
            else if (is_rtl_style(engine_model))
                rtl_record_requant_transaction(total, OC, OC_layer, false,
                                               have_corr, skip_l1_write, write_bytes);
            if (pipe > elapsed_cyc) wait(pipe - elapsed_cyc, sc_core::SC_NS);
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end  .to_seconds() * 1e9));
            rtl_phase_tasks.push_back(is_rtl_style(engine_model)
                                      ? last_rtl_phases
                                      : std::vector<RtlPhaseTrace>{});
            done_tag_out.write(0);
        }
    }
};

} // namespace mdla7
