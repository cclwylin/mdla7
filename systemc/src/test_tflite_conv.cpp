// Reads a conv_layer.bin written by scripts/extract_conv.py, runs the same
// conv on Mdla7System, and compares against the embedded INT32 reference.

#include <systemc>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include "mdla7/system.h"

using namespace mdla7;

namespace {

struct ConvBlob {
    uint16_t H, W, Cin, OH, OW, OC;
    uint8_t  Kh, Kw, sh, sw, pT, pB, pL, pR;
    std::vector<int8_t>  in_buf;
    std::vector<int8_t>  wgt_buf;
    std::vector<int32_t> ref;
};

bool load_blob(const std::string& path, ConvBlob& b) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "open " << path << " failed\n"; return false; }
    uint32_t magic, version;
    f.read(reinterpret_cast<char*>(&magic),   4);
    f.read(reinterpret_cast<char*>(&version), 4);
    if (magic != 0x374C444D || version != 1) {
        std::cerr << "bad magic/version  magic=0x" << std::hex << magic
                  << " ver=" << version << std::dec << "\n";
        return false;
    }
    f.read(reinterpret_cast<char*>(&b.H),   2);
    f.read(reinterpret_cast<char*>(&b.W),   2);
    f.read(reinterpret_cast<char*>(&b.Cin), 2);
    f.read(reinterpret_cast<char*>(&b.OH),  2);
    f.read(reinterpret_cast<char*>(&b.OW),  2);
    f.read(reinterpret_cast<char*>(&b.OC),  2);
    f.read(reinterpret_cast<char*>(&b.Kh),  1);
    f.read(reinterpret_cast<char*>(&b.Kw),  1);
    f.read(reinterpret_cast<char*>(&b.sh),  1);
    f.read(reinterpret_cast<char*>(&b.sw),  1);
    f.read(reinterpret_cast<char*>(&b.pT),  1);
    f.read(reinterpret_cast<char*>(&b.pB),  1);
    f.read(reinterpret_cast<char*>(&b.pL),  1);
    f.read(reinterpret_cast<char*>(&b.pR),  1);
    b.in_buf .resize(uint64_t(b.H)  * b.W  * b.Cin);
    b.wgt_buf.resize(uint64_t(b.OC) * b.Kh * b.Kw * b.Cin);
    b.ref    .resize(uint64_t(b.OH) * b.OW * b.OC);
    f.read(reinterpret_cast<char*>(b.in_buf .data()), b.in_buf .size());
    f.read(reinterpret_cast<char*>(b.wgt_buf.data()), b.wgt_buf.size());
    f.read(reinterpret_cast<char*>(b.ref    .data()), b.ref    .size() * 4);
    return f.good() || f.eof();
}

uint8_t encode_stride_pair(uint8_t s_h, uint8_t s_w) {
    auto enc = [](uint8_t s) -> uint8_t { return s == 4 ? 2 : s == 2 ? 1 : 0; };
    return uint8_t(enc(s_h) | (enc(s_w) << 2));
}

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

Descriptor conv_op(const ConvBlob& b,
                   uint32_t in_addr, uint32_t wgt_addr, uint32_t out_addr,
                   uint8_t wait_a, uint8_t wait_b, uint8_t sig) {
    Descriptor d{};
    d.hdr.op_class_subtype = OC_CONV;
    d.hdr.dtype  = DT_INT8x8;
    d.hdr.signal_tag = sig;
    d.hdr.wait_count = 2;
    d.hdr.wait_tags[0] = wait_a;
    d.hdr.wait_tags[1] = wait_b;
    auto& c = d.body.conv;
    c.in_addr = in_addr; c.wgt_addr = wgt_addr; c.out_addr = out_addr;
    c.in_h = b.H;  c.in_w = b.W;  c.in_c = b.Cin;  c.out_c = b.OC;
    c.k_h = b.Kh;  c.k_w = b.Kw;
    c.stride_dilation = encode_stride_pair(b.sh, b.sw);
    c.pad_tb = uint8_t((b.pT & 7) | ((b.pB & 7) << 3));
    c.pad_lr = uint8_t((b.pL & 7) | ((b.pR & 7) << 3));
    c.group  = 1;
    c.cluster_mask = 0xFFFF;
    return d;
}

Descriptor dma_out(uint32_t src, uint32_t dst, uint32_t bytes,
                   uint8_t wait_tag, uint8_t sig) {
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

int sc_main(int argc, char* argv[]) {
    sc_core::sc_report_handler::set_actions(sc_core::SC_INFO, sc_core::SC_DO_NOTHING);

    std::string blob = (argc >= 2) ? argv[1] : "build/conv_layer.bin";
    ConvBlob b;
    if (!load_blob(blob, b)) return 2;

    std::cout << "=== test_tflite_conv: " << blob << "\n"
              << "    in=" << b.H << "x" << b.W << "x" << b.Cin
              << "  wgt=" << b.OC << "x" << int(b.Kh) << "x" << int(b.Kw)
              << "x" << b.Cin
              << "  s=" << int(b.sh) << "x" << int(b.sw)
              << "  pad=" << int(b.pT) << "/" << int(b.pB) << "/"
                          << int(b.pL) << "/" << int(b.pR)
              << "  out=" << b.OH << "x" << b.OW << "x" << b.OC << "\n";

    Mdla7System sys("mdla7");
    constexpr uint32_t DRAM_WGT = 0x10000000;
    constexpr uint32_t DRAM_IN  = 0x10800000;
    constexpr uint32_t DRAM_OUT = 0x11000000;
    constexpr uint32_t L1_WGT = 0x00000000;
    const     uint32_t L1_IN  = uint32_t(b.wgt_buf.size() + 0x10000) & ~0xFFFu;
    const     uint32_t L1_OUT = (L1_IN + uint32_t(b.in_buf.size()) + 0x10000) & ~0xFFFu;

    const uint32_t IN_BYTES  = uint32_t(b.in_buf .size());
    const uint32_t WGT_BYTES = uint32_t(b.wgt_buf.size());
    const uint32_t OUT_BYTES = uint32_t(b.ref    .size() * sizeof(int32_t));

    sys.dram.write(DRAM_WGT, b.wgt_buf.data(), WGT_BYTES);
    sys.dram.write(DRAM_IN,  b.in_buf .data(), IN_BYTES);

    sys.host.program = {
        udma_load(DRAM_WGT, L1_WGT, WGT_BYTES, 1),
        udma_load(DRAM_IN,  L1_IN,  IN_BYTES,  2),
        conv_op  (b, L1_IN, L1_WGT, L1_OUT, /*wait*/ 1, 2, /*sig*/ 3),
        dma_out  (L1_OUT, DRAM_OUT, OUT_BYTES, /*wait*/ 3, /*sig*/ 4),
    };

    // Larger conv layers may need more sim time.
    const uint64_t budget_ns = std::max<uint64_t>(
        100'000, uint64_t(b.OH) * b.OW * b.OC * b.Kh * b.Kw * b.Cin / 100);
    sc_core::sc_start(double(budget_ns), sc_core::SC_NS);

    std::vector<int32_t> sim(b.ref.size(), 0);
    sys.dram.read(DRAM_OUT, sim.data(), OUT_BYTES);

    int mismatches = 0;
    for (size_t i = 0; i < b.ref.size(); ++i) {
        if (b.ref[i] != sim[i]) {
            if (mismatches < 5) {
                std::cout << "  mismatch[" << i << "]  ref=" << b.ref[i]
                          << "  sim=" << sim[i] << "\n";
            }
            mismatches++;
        }
    }
    if (mismatches == 0) {
        std::cout << "PASS  (" << b.ref.size() << " elements)\n";
        return 0;
    }
    std::cout << "FAIL  " << mismatches << " / " << b.ref.size() << "\n";
    return 1;
}
