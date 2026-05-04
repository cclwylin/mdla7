// Synthetic INT8 conv unit test:
//   - generate deterministic INT8 input + weights
//   - compute reference INT32 output in pure C++
//   - run the same conv via Mdla7System
//   - compare every output element

#include <systemc>
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include "mdla7/system.h"
#include "mdla7/reference.h"

using namespace mdla7;

namespace {

Descriptor udma_load(uint32_t src, uint32_t dst, uint32_t bytes, uint8_t sig) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_UDMA;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.signal_tag = sig;
    d.body.udma.mode = UM_LINEAR_COPY;
    d.body.udma.src_addr  = src;
    d.body.udma.dst_addr  = dst;
    d.body.udma.length    = bytes;
    return d;
}

Descriptor conv_op(uint32_t in_addr, uint32_t wgt_addr, uint32_t out_addr,
                   uint16_t in_h, uint16_t in_w, uint16_t in_c, uint16_t out_c,
                   uint8_t k_h, uint8_t k_w,
                   uint8_t pad, uint8_t wait_a, uint8_t wait_b, uint8_t sig) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_CONV;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.signal_tag = sig;
    d.hdr.wait_count = 2;
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    auto& c = d.body.conv;
    c.in_addr  = in_addr; c.wgt_addr = wgt_addr; c.out_addr = out_addr;
    c.in_h = in_h; c.in_w = in_w; c.in_c = in_c; c.out_c = out_c;
    c.k_h = k_h;   c.k_w = k_w;
    c.stride_dilation = 0;     // stride 1 / dilation 1
    c.pad_tb = uint8_t(pad | (pad << 3));
    c.pad_lr = uint8_t(pad | (pad << 3));
    c.group  = 1;
    c.cluster_mask = 0xFFFF;
    return d;
}

Descriptor dma_out(uint32_t src, uint32_t dst, uint32_t bytes, uint8_t wait_tag, uint8_t sig) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_UDMA;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.wait_count = 1;
    d.hdr.wait_tags[0] = wait_tag;
    d.hdr.signal_tag = sig;
    d.body.udma.mode = UM_LINEAR_COPY;
    d.body.udma.direction = 1;
    d.body.udma.src_addr  = src;
    d.body.udma.dst_addr  = dst;
    d.body.udma.length    = bytes;
    return d;
}

} // anon

int sc_main(int /*argc*/, char* /*argv*/[]) {
    sc_core::sc_report_handler::set_actions(sc_core::SC_INFO, sc_core::SC_DO_NOTHING);

    // Layer params — kept small so reference compute is fast.
    constexpr uint16_t H = 6, W = 6, Cin = 4, Cout = 8, K = 3;
    constexpr uint8_t  PAD = 1;                    // SAME-style for K=3, S=1
    constexpr uint16_t OH = (H + 2*PAD - K) + 1;   // = H
    constexpr uint16_t OW = (W + 2*PAD - K) + 1;
    constexpr uint32_t IN_BYTES  = uint32_t(H) * W * Cin;
    constexpr uint32_t WGT_BYTES = uint32_t(Cout) * K * K * Cin;
    constexpr uint32_t OUT_BYTES = uint32_t(OH) * OW * Cout * sizeof(int32_t);

    // Deterministic test data (small range so int32 sums are easy to read).
    std::vector<int8_t> in_t (IN_BYTES);
    std::vector<int8_t> wgt_t(WGT_BYTES);
    for (size_t i = 0; i < in_t .size(); ++i) in_t [i] = int8_t((i % 7)  - 3);   // -3..3
    for (size_t i = 0; i < wgt_t.size(); ++i) wgt_t[i] = int8_t((i % 5)  - 2);   // -2..2

    // Reference compute.
    std::vector<int32_t> ref = ref::conv_int8(
        in_t.data(), wgt_t.data(),
        H, W, Cin, Cout, K, K, /*s_h*/ 1, /*s_w*/ 1,
        PAD, PAD, PAD, PAD);

    // Build & run sim.
    Mdla7System sys("mdla7");
    constexpr uint32_t DRAM_WGT = 0x10000000;
    constexpr uint32_t DRAM_IN  = 0x10100000;
    constexpr uint32_t DRAM_OUT = 0x10200000;
    constexpr uint32_t L1_WGT = 0x00000000;
    constexpr uint32_t L1_IN  = 0x00010000;
    constexpr uint32_t L1_OUT = 0x00020000;

    sys.dram.write(DRAM_WGT, wgt_t.data(), WGT_BYTES);
    sys.dram.write(DRAM_IN,  in_t .data(), IN_BYTES);

    sys.host.program = {
        udma_load(DRAM_WGT, L1_WGT, WGT_BYTES, /*sig*/ 1),
        udma_load(DRAM_IN,  L1_IN,  IN_BYTES,  /*sig*/ 2),
        conv_op  (L1_IN, L1_WGT, L1_OUT, H, W, Cin, Cout, K, K, PAD,
                  /*wait*/ 1, 2, /*sig*/ 3),
        dma_out  (L1_OUT, DRAM_OUT, OUT_BYTES, /*wait*/ 3, /*sig*/ 4),
    };

    std::cout << "=== test_conv_synth: " << H << "x" << W << "x" << Cin
              << " -> " << OH << "x" << OW << "x" << Cout << " (K=" << int(K) << ") ===\n";
    sc_core::sc_start(100, sc_core::SC_US);

    // Read sim output back from DRAM and compare.
    std::vector<int32_t> sim(OH * OW * Cout, 0);
    sys.dram.read(DRAM_OUT, sim.data(), OUT_BYTES);

    int mismatches = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] != sim[i]) {
            if (mismatches < 5) {
                std::cout << "  mismatch [" << i << "]  ref=" << ref[i]
                          << "  sim=" << sim[i] << "\n";
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        std::cout << "PASS  (" << ref.size() << " elements verified)\n";
        return 0;
    }
    std::cout << "FAIL  " << mismatches << " / " << ref.size() << " mismatches\n";
    return 1;
}
