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

#include <systemc>
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
            default:
                std::cout << "[UDMA] unknown mode=" << int(u.mode) << "\n";
                wait(10, sc_core::SC_NS);
                break;
            }
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            const auto t_begin_ns = uint64_t(t_begin.to_seconds() * 1e9);
            const auto t_end_ns   = uint64_t(t_end  .to_seconds() * 1e9);
            busy_time += t_end - t_begin;
            tasks.emplace_back(t_begin_ns, t_end_ns);          // legacy combined
            if (u.direction == 1) {                            // L1 → DRAM (store)
                busy_time_write += t_end - t_begin;
                tasks_write.emplace_back(t_begin_ns, t_end_ns);
            } else {                                            // DRAM → L1 (load)
                busy_time_read  += t_end - t_begin;
                tasks_read.emplace_back(t_begin_ns, t_end_ns);
            }
            done_tag_out.write(0);
        }
    }

    // v2.1: bandwidth is now modeled inside L1Mesh / Dram (each call to
    //       l1mgr.read/write blocks the calling thread for the access time).
    //       UDMA only adds a per-descriptor 16-cycle decode startup.
    void wait_bytes(uint64_t /*bytes*/) {
        wait(16, sc_core::SC_NS);
    }

    void do_linear(const UdmaBody& u) {
        std::cout << "[UDMA] LINEAR_COPY  src=0x" << std::hex << u.src_addr
                  << "  dst=0x" << u.dst_addr
                  << "  len=" << std::dec << u.length << " B\n";
        std::vector<uint8_t> buf(u.length);
        l1mgr.read (u.src_addr, buf.data(), u.length);
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
};

} // namespace mdla7
