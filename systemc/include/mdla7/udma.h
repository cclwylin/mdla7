#pragma once

// UDMA — DRAM <-> L1Mesh data movement; 5 op modes (see spec §3A.9).
//
//   LINEAR_COPY    : memcpy(dst, src, length).
//   STRIDED_2D     : copy num_chunks rows, each `length` bytes,
//                    stepping by src_stride / dst_stride between rows.
//   INDEXED_GATHER : dst[i*dst_stride .. +length] = src[idx[i]*src_stride .. +length].
//                    idx[] (uint32_t) lives at idx_table_addr.
//   SCATTER_CONCAT : append num_chunks sources sequentially to dst.
//                    each source = (uint32_t src_addr, uint32_t length) pair at idx_table_addr.
//   STRIDED_SLICE  : 2D slice — rows in [slice_begin[0], slice_end[0]),
//                    starting at column-byte slice_begin[1] within each row.
//                    Each output row is `length` bytes; src rows step src_stride,
//                    dst rows step dst_stride.
//   DEPTH_TO_SPACE : NHWC depth-to-space transform. Encoded as:
//                    src_stride=input row bytes, dst_stride=output row bytes,
//                    num_chunks=input rows, slice_begin={W,Cin,block,Cout},
//                    length=element bytes.
//   ACT_DECOMP_COPY: DRAM compressed activation -> L1 raw activation.
//                    length=raw bytes, src_stride=compressed bytes,
//                    dst_stride=metadata/header bytes.
//   ACT_COMP_COPY  : L1 raw activation -> DRAM raw activation.
//                    DRAM write compression is disabled for now, so timing
//                    charges raw bytes even if compressed metadata is present.

#include <systemc>
#include <cstring>
#include <iostream>
#include <vector>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"

namespace mdla7 {

SC_MODULE(Udma) {
    sc_core::sc_fifo_in<DescriptorBody>  cfg_in;
    sc_core::sc_fifo_out<uint8_t>        done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    // v8.4: split by direction so the Gantt can show DRAM→L1 (read) and
    // L1→DRAM (write) on separate lanes — the two are very different
    // workloads (input/weight loads vs output stores) and overlap is the
    // interesting bandwidth-utilization story.
    sc_core::sc_time busy_time_read {sc_core::SC_ZERO_TIME};
    sc_core::sc_time busy_time_write{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks_read;
    std::vector<std::pair<uint64_t, uint64_t>> tasks_write;
    std::vector<RtlPhaseTrace> last_rtl_phases;
    std::vector<std::vector<RtlPhaseTrace>> rtl_phase_tasks_read;
    std::vector<std::vector<RtlPhaseTrace>> rtl_phase_tasks_write;
    EngineModel engine_model = EngineModel::Analytical;
    static constexpr uint32_t UDMA_READ_OUTSTANDING = 2;
    static constexpr bool ACT_COMP_WRITE_ENABLE = false;

    SC_HAS_PROCESS(Udma);
    Udma(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) {
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            const UdmaBody& u = body.udma;

            switch (u.mode) {
            case UM_LINEAR_COPY:    do_linear(u);   break;
            case UM_STRIDED_2D:     do_strided(u);  break;
            case UM_INDEXED_GATHER: do_gather(u);   break;
            case UM_SCATTER_CONCAT: do_concat(u);   break;
            case UM_STRIDED_SLICE:  do_slice(u);    break;
            case UM_DEPTH_TO_SPACE: do_depth_to_space(u); break;
            case UM_ACT_DECOMP_COPY: do_act_decomp(u); break;
            case UM_ACT_COMP_COPY:   do_act_comp(u);   break;
            case UM_ACT_DECOMP_STREAM_HEAD: do_act_decomp_stream_head(u); break;
            case UM_ACT_DECOMP_STREAM_TAIL: do_act_decomp_stream_tail(u); break;
            default:
                std::cout << "[UDMA] unknown mode=" << int(u.mode) << "\n";
                wait(10, sc_core::SC_NS);
                break;
            }
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            const auto t_begin_ns = uint64_t(t_begin.to_seconds() * 1e9);
            const auto t_end_ns   = uint64_t(t_end  .to_seconds() * 1e9);
            if (is_rtl_style(engine_model))
                rtl_record_udma_transaction(u, t_end_ns - t_begin_ns);
            busy_time += t_end - t_begin;
            tasks.emplace_back(t_begin_ns, t_end_ns);          // legacy combined
            if (u.direction == 1) {                            // L1 → DRAM (store)
                busy_time_write += t_end - t_begin;
                tasks_write.emplace_back(t_begin_ns, t_end_ns);
                rtl_phase_tasks_write.push_back(is_rtl_style(engine_model)
                                                ? last_rtl_phases
                                                : std::vector<RtlPhaseTrace>{});
            } else {                                            // DRAM → L1 (load)
                busy_time_read  += t_end - t_begin;
                tasks_read.emplace_back(t_begin_ns, t_end_ns);
                rtl_phase_tasks_read.push_back(is_rtl_style(engine_model)
                                               ? last_rtl_phases
                                               : std::vector<RtlPhaseTrace>{});
            }
            done_tag_out.write(0);
        }
    }

    static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
        return b ? ((a + b - 1) / b) : 0;
    }

    uint64_t payload_bytes_for(const UdmaBody& u) const {
        switch (u.mode) {
        case UM_STRIDED_2D:
            return uint64_t(u.length) * u.num_chunks;
        case UM_INDEXED_GATHER:
            return uint64_t(u.length) * u.num_chunks + uint64_t(u.num_chunks) * sizeof(uint32_t);
        case UM_SCATTER_CONCAT:
            return uint64_t(u.num_chunks) * 8u;
        case UM_STRIDED_SLICE:
            return uint64_t(u.length) * (uint16_t(u.slice_end[0]) - uint16_t(u.slice_begin[0]));
        case UM_DEPTH_TO_SPACE:
            return uint64_t(u.num_chunks) * u.src_stride;
        case UM_ACT_DECOMP_COPY:
        case UM_ACT_COMP_COPY:
            return uint64_t(u.src_stride ? u.src_stride : u.length) + u.dst_stride;
        case UM_ACT_DECOMP_STREAM_HEAD:
            return u.idx_table_addr ? u.idx_table_addr : u.length;
        case UM_ACT_DECOMP_STREAM_TAIL:
            return u.length;
        case UM_LINEAR_COPY:
        default:
            return u.length;
        }
    }

    uint64_t codec_cycles_for(const UdmaBody& u) const {
        switch (u.mode) {
        case UM_ACT_DECOMP_COPY:
        case UM_ACT_COMP_COPY:
        case UM_ACT_DECOMP_STREAM_HEAD:
        case UM_ACT_DECOMP_STREAM_TAIL:
            return ceil_div_u64(u.length, 512);
        default:
            return 0;
        }
    }

    void rtl_add_phase(const char* name, uint64_t cycles,
                       uint64_t read_bytes = 0, uint64_t write_bytes = 0,
                       const char* stall = "") {
        RtlPhaseTrace phase;
        phase.name = name;
        phase.cycles = cycles;
        phase.read_bytes = read_bytes;
        phase.write_bytes = write_bytes;
        phase.stall = stall ? stall : "";
        last_rtl_phases.push_back(phase);
    }

    void rtl_record_udma_transaction(const UdmaBody& u, uint64_t observed_cycles) {
        const uint64_t bytes = payload_bytes_for(u);
        const uint64_t codec = codec_cycles_for(u);
        last_rtl_phases.clear();
        rtl_add_phase("issue", 4);
        if (u.direction == 1) {
            rtl_add_phase("l1_read", ceil_div_u64(bytes, 256),
                          bytes, 0, "l1_payload_read");
            if (codec)
                rtl_add_phase("codec", codec, 0, 0, "act_codec");
            rtl_add_phase("dram_write", ceil_div_u64(bytes, 48) + 50,
                          0, bytes, "dram_write");
        } else {
            const uint64_t dram_bytes = effective_read_bytes(uint32_t(bytes));
            rtl_add_phase("dram_read", ceil_div_u64(dram_bytes, 48) + 50,
                          dram_bytes, 0, "dram_read");
            if (codec)
                rtl_add_phase("codec", codec, 0, 0, "act_codec");
            rtl_add_phase("l1_write", ceil_div_u64(bytes, 256),
                          0, bytes, "l1_payload_write");
        }
        rtl_add_phase("done", observed_cycles ? 1 : 0);
    }

    // v2.1: bandwidth is now modeled inside L1Mesh / Dram (each call to
    //       l1mgr.read/write blocks the calling thread for the access time).
    //       UDMA only adds a per-descriptor 16-cycle decode startup.
    void wait_bytes(uint64_t /*bytes*/) {
        wait(16, sc_core::SC_NS);
    }

    uint32_t effective_read_bytes(uint32_t bytes) const {
        return (bytes + UDMA_READ_OUTSTANDING - 1) / UDMA_READ_OUTSTANDING;
    }

    void do_linear(const UdmaBody& u) {
        std::cout << "[UDMA] LINEAR_COPY  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr
                  << "  len=" << std::dec << u.length << " B\n";
        std::vector<uint8_t> buf(u.length);
        if (u.direction == 0 && addr_in_dram(u.src_addr)) {
            l1mgr.read_compressed(u.src_addr, buf.data(), u.length,
                                  effective_read_bytes(u.length));
        } else {
            l1mgr.read(u.src_addr, buf.data(), u.length);
        }
        l1mgr.write(u.dst_addr, buf.data(), u.length);
        wait_bytes(u.length);
    }

    void do_strided(const UdmaBody& u) {
        std::cout << "[UDMA] STRIDED_2D  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr
                  << "  rows=" << std::dec << u.num_chunks
                  << "  row_len=" << u.length << "\n";
        std::vector<uint8_t> buf(u.length);
        for (uint16_t r = 0; r < u.num_chunks; ++r) {
            l1mgr.read (u.src_addr + r * u.src_stride, buf.data(), u.length);
            l1mgr.write(u.dst_addr + r * u.dst_stride, buf.data(), u.length);
        }
        wait_bytes(uint64_t(u.length) * u.num_chunks);
    }

    void do_gather(const UdmaBody& u) {
        std::cout << "[UDMA] INDEXED_GATHER  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr << std::dec
                  << "  n=" << u.num_chunks
                  << "  elem=" << u.length << "\n";
        // Read index table (uint32_t each).
        std::vector<uint32_t> idx(u.num_chunks);
        l1mgr.read(u.idx_table_addr, idx.data(), idx.size() * sizeof(uint32_t));
        std::vector<uint8_t> buf(u.length);
        for (uint16_t i = 0; i < u.num_chunks; ++i) {
            uint32_t s = u.src_addr + uint32_t(idx[i]) * u.src_stride;
            uint32_t d = u.dst_addr + uint32_t(i)      * u.dst_stride;
            l1mgr.read (s, buf.data(), u.length);
            l1mgr.write(d, buf.data(), u.length);
        }
        wait_bytes(uint64_t(u.length) * u.num_chunks);
    }

    struct ConcatEntry { uint32_t src_addr; uint32_t length; };
    void do_concat(const UdmaBody& u) {
        std::cout << "[UDMA] SCATTER_CONCAT  dst=0x" << std::hex << u.dst_addr
                  << std::dec << "  sources=" << u.num_chunks << "\n";
        std::vector<ConcatEntry> srcs(u.num_chunks);
        l1mgr.read(u.idx_table_addr, srcs.data(), srcs.size() * sizeof(ConcatEntry));
        uint32_t cursor = u.dst_addr;
        uint64_t total = 0;
        for (auto& s : srcs) {
            std::vector<uint8_t> buf(s.length);
            l1mgr.read (s.src_addr, buf.data(), s.length);
            l1mgr.write(cursor,     buf.data(), s.length);
            cursor += s.length;
            total  += s.length;
        }
        wait_bytes(total);
    }

    void do_slice(const UdmaBody& u) {
        uint16_t r0 = u.slice_begin[0], r1 = u.slice_end[0];
        uint16_t col_off = u.slice_begin[1];
        std::cout << "[UDMA] STRIDED_SLICE  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr << std::dec
                  << "  rows=[" << r0 << "," << r1 << ")"
                  << "  col_off=" << col_off
                  << "  row_len=" << u.length << "\n";
        std::vector<uint8_t> buf(u.length);
        for (uint16_t r = r0; r < r1; ++r) {
            uint32_t s = u.src_addr + r * u.src_stride + col_off;
            uint32_t d = u.dst_addr + (r - r0) * u.dst_stride;
            l1mgr.read (s, buf.data(), u.length);
            l1mgr.write(d, buf.data(), u.length);
        }
        wait_bytes(uint64_t(u.length) * (r1 - r0));
    }

    void do_depth_to_space(const UdmaBody& u) {
        const uint32_t H = u.num_chunks;
        const uint32_t W = u.slice_begin[0];
        const uint32_t Cin = u.slice_begin[1];
        const uint32_t block = u.slice_begin[2];
        const uint32_t Cout = u.slice_begin[3];
        const uint32_t elem = u.length ? u.length : 1;
        std::cout << "[UDMA] DEPTH_TO_SPACE  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr << std::dec
                  << "  in=" << H << "x" << W << "x" << Cin
                  << "  block=" << block << "  cout=" << Cout << "\n";
        if (!H || !W || !Cin || !block || !Cout || Cin != Cout * block * block) {
            wait(10, sc_core::SC_NS);
            return;
        }
        const uint64_t src_bytes = uint64_t(H) * u.src_stride;
        const uint64_t dst_bytes = uint64_t(H) * block * u.dst_stride;
        std::vector<uint8_t> src(src_bytes);
        std::vector<uint8_t> dst(dst_bytes);
        l1mgr.read(u.src_addr, src.data(), src.size());
        for (uint32_t ih = 0; ih < H; ++ih) {
            for (uint32_t iw = 0; iw < W; ++iw) {
                for (uint32_t ic = 0; ic < Cin; ++ic) {
                    const uint32_t q = ic / Cout;
                    const uint32_t oc = ic % Cout;
                    const uint32_t bh = q / block;
                    const uint32_t bw = q % block;
                    const uint32_t oh = ih * block + bh;
                    const uint32_t ow = iw * block + bw;
                    const uint32_t src_off = (ih * W * Cin + iw * Cin + ic) * elem;
                    const uint32_t dst_off = oh * u.dst_stride
                                           + (ow * Cout + oc) * elem;
                    std::memcpy(dst.data() + dst_off, src.data() + src_off, elem);
                }
            }
        }
        l1mgr.write(u.dst_addr, dst.data(), dst.size());
        wait_bytes(src_bytes);
    }

    void wait_act_codec(uint64_t raw_bytes) {
        // v9 ACTC: 512 B/cycle codec pipe. This is intentionally
        // separate from DRAM/L1 bandwidth; it models the hardware resource
        // between UDMA and L1, after compressed bytes have been fetched.
        wait(double((raw_bytes + 511) / 512), sc_core::SC_NS);
    }

    void wait_stream_act_slice(uint64_t raw_bytes, uint64_t comp_meta_bytes) {
        // Row-streaming approximation: the descriptor below already copied the
        // full functional tile into L1, while these waits reserve the UDMA/ACTC
        // resource for the prologue or tail byte range.
        const uint64_t dram_cyc = (effective_read_bytes(uint32_t(comp_meta_bytes)) + 47) / 48 + 50;
        const uint64_t l1_cyc = (raw_bytes + 255) / 256;
        const uint64_t codec_cyc = (raw_bytes + 511) / 512;
        wait(double(std::max(dram_cyc, l1_cyc) + codec_cyc + 16), sc_core::SC_NS);
    }

    void do_act_decomp(const UdmaBody& u) {
        const uint32_t raw_bytes = u.length;
        const uint32_t comp_bytes = u.src_stride ? u.src_stride : raw_bytes;
        const uint32_t meta_bytes = u.dst_stride;
        std::cout << "[UDMA] ACT_DECOMP  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr << std::dec
                  << "  raw=" << raw_bytes
                  << "  comp=" << comp_bytes
                  << "  meta=" << meta_bytes << " B\n";
        std::vector<uint8_t> buf(raw_bytes);
        // Functional memory remains raw so existing references still verify.
        // Timing charges only compressed payload + metadata for the DRAM read.
        l1mgr.read_compressed(u.src_addr, buf.data(), raw_bytes,
                              effective_read_bytes(comp_bytes + meta_bytes));
        wait_act_codec(raw_bytes);
        l1mgr.write(u.dst_addr, buf.data(), raw_bytes);
        wait_bytes(comp_bytes + meta_bytes);
    }

    void do_act_decomp_stream_head(const UdmaBody& u) {
        const uint32_t raw_bytes = u.length;
        const uint32_t comp_bytes = u.src_stride ? u.src_stride : raw_bytes;
        const uint32_t meta_bytes = u.dst_stride;
        const uint32_t head_raw = std::min<uint32_t>(u.idx_table_addr ? u.idx_table_addr : raw_bytes,
                                                     raw_bytes);
        const uint32_t total_comp_meta = comp_bytes + meta_bytes;
        const uint32_t head_comp_meta = std::max<uint32_t>(
            1, uint32_t((uint64_t(total_comp_meta) * head_raw + raw_bytes - 1) / raw_bytes));
        std::cout << "[UDMA] ACT_STREAM_HEAD  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr << std::dec
                  << "  raw=" << raw_bytes
                  << "  head=" << head_raw
                  << "  comp=" << comp_bytes
                  << "  meta=" << meta_bytes << " B\n";
        std::vector<uint8_t> buf(raw_bytes);
        l1mgr.read_compressed_instant(u.src_addr, buf.data(), raw_bytes);
        l1mgr.write_instant(u.dst_addr, buf.data(), raw_bytes);
        wait_stream_act_slice(head_raw, head_comp_meta);
    }

    void do_act_decomp_stream_tail(const UdmaBody& u) {
        const uint32_t raw_bytes = u.length;
        const uint32_t comp_meta_bytes = u.src_stride ? u.src_stride : raw_bytes;
        std::cout << "[UDMA] ACT_STREAM_TAIL  raw=" << raw_bytes
                  << "  comp_meta=" << comp_meta_bytes << " B\n";
        wait_stream_act_slice(raw_bytes, comp_meta_bytes);
    }

    void do_act_comp(const UdmaBody& u) {
        const uint32_t raw_bytes = u.length;
        const uint32_t comp_bytes = u.src_stride ? u.src_stride : raw_bytes;
        const uint32_t meta_bytes = u.dst_stride;
        std::cout << "[UDMA] ACT_COMP  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr << std::dec
                  << "  raw=" << raw_bytes
                  << "  comp=" << comp_bytes
                  << "  meta=" << meta_bytes << " B\n";
        std::vector<uint8_t> buf(raw_bytes);
        l1mgr.read(u.src_addr, buf.data(), raw_bytes);
        wait_act_codec(raw_bytes);
        if constexpr (ACT_COMP_WRITE_ENABLE) {
            // Functional DRAM remains raw; timing charges compressed payload +
            // metadata so final output verification can keep reading raw bytes.
            l1mgr.write_compressed(u.dst_addr, buf.data(), raw_bytes,
                                   comp_bytes + meta_bytes);
            wait_bytes(comp_bytes + meta_bytes);
        } else {
            l1mgr.write(u.dst_addr, buf.data(), raw_bytes);
            wait_bytes(raw_bytes);
        }
    }
};

} // namespace mdla7
