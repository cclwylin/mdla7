// v0 dry-run testbench: load a small INT8 conv via UDMA, run CONV → Requant.

#include <systemc>
#include <iostream>
#include "mdla7/system.h"

using namespace mdla7;

static Descriptor make_udma_load(uint32_t src, uint32_t dst, uint32_t bytes,
                                 uint8_t signal_tag) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_UDMA;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.layer_id = 0;
    d.body.udma.mode = UM_LINEAR_COPY;
    d.body.udma.direction = 0;
    d.body.udma.src_addr  = src;
    d.body.udma.dst_addr  = dst;
    d.body.udma.length    = bytes;
    return d;
}

static Descriptor make_conv(uint32_t in_addr, uint32_t wgt_addr, uint32_t out_addr,
                            uint16_t in_h, uint16_t in_w, uint16_t in_c, uint16_t out_c,
                            uint8_t k_h, uint8_t k_w,
                            uint8_t wait_a, uint8_t wait_b, uint8_t signal_tag) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_CONV;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = (wait_a ? 1 : 0) + (wait_b ? 1 : 0);
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    d.hdr.layer_id = 1;
    auto& c = d.body.conv;
    c.in_addr  = in_addr;  c.wgt_addr = wgt_addr; c.out_addr = out_addr;
    c.in_h = in_h; c.in_w = in_w; c.in_c = in_c; c.out_c = out_c;
    c.k_h = k_h;   c.k_w = k_w;
    c.stride_dilation = 0;     // s=1, d=1
    c.pad_tb = c.pad_lr = ((k_h - 1) / 2) | (((k_w - 1) / 2) << 3);
    c.group  = 1;
    c.cluster_mask = 0xFFFF;   // all 16 cluster
    return d;
}

static Descriptor make_requant(uint32_t in_addr, uint32_t out_addr,
                               uint16_t h, uint16_t w, uint16_t c,
                               uint8_t wait_tag, uint8_t signal_tag) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_REQUANT;
    d.hdr.dtype = DT_INT8x8;
    d.hdr.signal_tag = signal_tag;
    d.hdr.wait_count = wait_tag ? 1 : 0;
    d.hdr.wait_tags[0] = wait_tag;
    d.hdr.layer_id = 1;
    auto& r = d.body.requant;
    r.in_addr = in_addr; r.out_addr = out_addr;
    r.n = 1; r.h = h; r.w = w; r.c = c;
    r.per_channel_flag = 1;
    return d;
}

int sc_main(int /*argc*/, char* /*argv*/[]) {
    sc_core::sc_report_handler::set_actions(
        sc_core::SC_INFO, sc_core::SC_DO_NOTHING);

    Mdla7System sys("mdla7");

    // Address map: weights at DRAM, output goes back to DRAM.
    constexpr uint32_t DRAM_WGT  = 0x10000000;
    constexpr uint32_t DRAM_ACT  = 0x10100000;
    constexpr uint32_t DRAM_OUT  = 0x10200000;
    constexpr uint32_t L1_WGT    = 0x00000000;
    constexpr uint32_t L1_ACT    = 0x00010000;
    constexpr uint32_t L1_OUT    = 0x00020000;

    // Test layer: 8x8x16 input, 3x3 kernel, 32 out channels.
    constexpr uint16_t H = 8, W = 8, Cin = 16, Cout = 32, K = 3;
    constexpr uint32_t WGT_BYTES = K * K * Cin * Cout;     // INT8 weights
    constexpr uint32_t ACT_BYTES = H * W * Cin;            // INT8 act
    constexpr uint32_t OUT_BYTES = H * W * Cout;

    // Pre-fill DRAM with deterministic patterns so we can spot copies.
    std::vector<uint8_t> wgt(WGT_BYTES); for (size_t i = 0; i < wgt.size(); ++i) wgt[i] = uint8_t(i);
    std::vector<uint8_t> act(ACT_BYTES); for (size_t i = 0; i < act.size(); ++i) act[i] = uint8_t(i ^ 0x55);
    sys.dram.write(DRAM_WGT, wgt.data(), WGT_BYTES);
    sys.dram.write(DRAM_ACT, act.data(), ACT_BYTES);

    // Build descriptor program — model one conv layer with full chain.
    sys.host.program = {
        make_udma_load(DRAM_WGT, L1_WGT, WGT_BYTES, /*signal*/ 1),
        make_udma_load(DRAM_ACT, L1_ACT, ACT_BYTES, /*signal*/ 2),
        make_conv(L1_ACT, L1_WGT, L1_OUT,
                  H, W, Cin, Cout, K, K,
                  /*wait*/ 1, 2, /*signal*/ 3),
        make_requant(L1_OUT, L1_OUT, H, W, Cout, /*wait*/ 3, /*signal*/ 4),
        make_udma_load(L1_OUT, DRAM_OUT, OUT_BYTES, /*signal*/ 5),
    };
    // The last UDMA needs to wait on tag 4 (Requant done):
    sys.host.program.back().hdr.wait_count = 1;
    sys.host.program.back().hdr.wait_tags[0] = 4;
    sys.host.program.back().body.udma.direction = 1;     // L1 -> DRAM

    std::cout << "=== MDLA7 v0 dry run ===\n";
    sc_core::sc_start(50, sc_core::SC_US);
    std::cout << "=== sim done @ "
              << sc_core::sc_time_stamp() << " ===\n";
    return 0;
}
