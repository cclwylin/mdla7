#pragma once

// TNPS — tensor permutation / packing engine.  This is data-movement only:
// transpose-like layout transforms, gather/slice/concat style reshapes, and
// byte-preserving materialization are modeled here instead of UDMA so they can
// overlap with DRAM traffic in the schedule.

#include <systemc>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"

namespace mdla7 {

SC_MODULE(TnpsEngine) {
    sc_core::sc_fifo_in<DescriptorBody> cfg_in;
    sc_core::sc_fifo_out<uint8_t>       done_tag_out;

    L1Manager& l1mgr;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    std::vector<RtlPhaseTrace> last_rtl_phases;
    std::vector<std::vector<RtlPhaseTrace>> rtl_phase_tasks;
    EngineModel engine_model = EngineModel::Analytical;
    sc_core::sc_time task_begin{sc_core::SC_ZERO_TIME};

    SC_HAS_PROCESS(TnpsEngine);
    TnpsEngine(sc_core::sc_module_name nm, L1Manager& mgr)
      : sc_module(nm), l1mgr(mgr) {
        SC_THREAD(run);
    }

    void run() {
        while (true) {
            DescriptorBody body = cfg_in.read();
            const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
            task_begin = t_begin;
            const TnpsBody& t = body.tnps;
            switch (t.mode) {
            case TM_LINEAR_COPY:    do_linear(t); break;
            case TM_STRIDED_2D:     do_strided(t); break;
            case TM_INDEXED_GATHER: do_gather(t); break;
            case TM_SCATTER_CONCAT: do_concat(t); break;
            case TM_STRIDED_SLICE:  do_slice_meta(t); break;
            case TM_DEPTH_TO_SPACE: do_depth_to_space(t); break;
            case TM_SPACE_TO_DEPTH: do_space_to_depth(t); break;
            case TM_TRANSPOSE:      do_transpose_meta(t); break;
            case TM_CHANNEL_PACK:   do_channel_pack(t); break;
            default:
                std::cout << "[TNPS] unknown mode=" << int(t.mode) << "\n";
                wait(10, sc_core::SC_NS);
                break;
            }
            const sc_core::sc_time t_end = sc_core::sc_time_stamp();
            busy_time += t_end - t_begin;
            tasks.emplace_back(uint64_t(t_begin.to_seconds() * 1e9),
                               uint64_t(t_end.to_seconds() * 1e9));
            rtl_phase_tasks.push_back(is_rtl_style(engine_model)
                                      ? last_rtl_phases
                                      : std::vector<RtlPhaseTrace>{});
            done_tag_out.write(0);
        }
    }

    void wait_bytes(uint64_t bytes) {
        // 8 R + 8 W payload lanes = 128 B/cyc in each direction.
        const uint64_t model_cycles = (bytes + 127) / 128 + 8;
        if (!is_rtl_style(engine_model)) {
            wait(double(model_cycles), sc_core::SC_NS);
            return;
        }
        rtl_run_copy_transaction(bytes);
    }

    static uint64_t ceil_div_u64(uint64_t a, uint64_t b) {
        return b ? ((a + b - 1) / b) : 0;
    }

    void rtl_wait_phase(const char* name, uint64_t cycles,
                        uint64_t read_bytes = 0, uint64_t write_bytes = 0,
                        const char* stall = "") {
        RtlPhaseTrace phase;
        phase.name = name;
        phase.cycles = cycles;
        phase.read_bytes = read_bytes;
        phase.write_bytes = write_bytes;
        phase.stall = stall ? stall : "";
        last_rtl_phases.push_back(phase);
        if (cycles)
            wait(cycles, sc_core::SC_NS);
    }

    void rtl_run_copy_transaction(uint64_t bytes) {
        last_rtl_phases.clear();
        rtl_wait_phase("issue", 4);
        rtl_wait_phase("read", ceil_div_u64(
            bytes, PayloadPortCount::TNPS_R * PAYLOAD_BYTES),
            bytes, 0, "payload_read");
        rtl_wait_phase("write", ceil_div_u64(
            bytes, PayloadPortCount::TNPS_W * PAYLOAD_BYTES),
            0, bytes, "payload_write");
        rtl_wait_phase("done", 4);
    }

    void do_linear(const TnpsBody& t) {
        std::cout << "[TNPS] LINEAR_COPY  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  len=" << t.length << " B\n";
        std::vector<uint8_t> buf(t.length);
        l1mgr.read(t.src_addr, buf.data(), t.length);
        l1mgr.write(t.dst_addr, buf.data(), t.length);
        wait_bytes(t.length);
    }

    void do_strided(const TnpsBody& t) {
        std::cout << "[TNPS] STRIDED_2D  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  rows=" << t.num_chunks
                  << "  row_len=" << t.length << "\n";
        std::vector<uint8_t> buf(t.length);
        for (uint16_t r = 0; r < t.num_chunks; ++r) {
            l1mgr.read(t.src_addr + r * t.src_stride, buf.data(), t.length);
            l1mgr.write(t.dst_addr + r * t.dst_stride, buf.data(), t.length);
        }
        wait_bytes(uint64_t(t.length) * t.num_chunks);
    }

    void do_gather(const TnpsBody& t) {
        std::cout << "[TNPS] INDEXED_GATHER  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  n=" << t.num_chunks
                  << "  elem=" << t.length << "\n";
        std::vector<uint32_t> idx(t.num_chunks);
        l1mgr.read(t.idx_table_addr, idx.data(), idx.size() * sizeof(uint32_t));
        std::vector<uint8_t> buf(t.length);
        for (uint16_t i = 0; i < t.num_chunks; ++i) {
            const uint32_t s = t.src_addr + uint32_t(idx[i]) * t.src_stride;
            const uint32_t d = t.dst_addr + uint32_t(i) * t.dst_stride;
            l1mgr.read(s, buf.data(), t.length);
            l1mgr.write(d, buf.data(), t.length);
        }
        wait_bytes(uint64_t(t.length) * t.num_chunks);
    }

    void do_channel_pack(const TnpsBody& t) {
        const uint32_t src_row = t.length;
        const uint32_t dst_row = t.dst_stride;
        const uint32_t dst_col = t.slice_begin[0];
        const uint32_t rows = t.num_chunks;
        std::cout << "[TNPS] CHANNEL_PACK  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  rows=" << rows
                  << "  src_row=" << src_row
                  << "  dst_row=" << dst_row
                  << "  dst_col=" << dst_col << "\n";
        if (!rows || !src_row || !dst_row || dst_col + src_row > dst_row) {
            wait(8, sc_core::SC_NS);
            return;
        }
        auto ticket = l1mgr.channel_pack(t.src_addr, t.dst_addr, rows, src_row,
                                         t.src_stride, dst_row, dst_col);
        L1Manager::wait_ticket(ticket);
        wait_bytes(uint64_t(rows) * src_row);
    }

    struct ConcatEntry { uint32_t src_addr; uint32_t length; };
    void do_concat(const TnpsBody& t) {
        std::cout << "[TNPS] SCATTER_CONCAT  dst=0x" << std::hex << t.dst_addr
                  << std::dec << "  sources=" << t.num_chunks << "\n";
        std::vector<ConcatEntry> srcs(t.num_chunks);
        l1mgr.read(t.idx_table_addr, srcs.data(), srcs.size() * sizeof(ConcatEntry));
        uint32_t cursor = t.dst_addr;
        uint64_t total = 0;
        for (const auto& s : srcs) {
            std::vector<uint8_t> buf(s.length);
            l1mgr.read(s.src_addr, buf.data(), s.length);
            l1mgr.write(cursor, buf.data(), s.length);
            cursor += s.length;
            total += s.length;
        }
        wait_bytes(total);
    }

    void do_slice(const TnpsBody& t) {
        const uint16_t r0 = t.slice_begin[0], r1 = t.slice_end[0];
        const uint16_t col_off = t.slice_begin[1];
        std::cout << "[TNPS] STRIDED_SLICE  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  rows=[" << r0 << "," << r1 << ")"
                  << "  col_off=" << col_off
                  << "  row_len=" << t.length << "\n";
        const uint32_t rows = (r1 > r0) ? uint32_t(r1 - r0) : 0u;
        if (!rows || !t.length) return;
        if (t.src_stride && col_off + t.length <= t.src_stride) {
            const uint64_t span64 = uint64_t(rows - 1) * t.src_stride + col_off + t.length;
            const uint64_t dst64 = uint64_t(rows) * t.dst_stride;
            if (span64 <= (128ull << 20) && dst64 <= (128ull << 20)) {
                std::vector<uint8_t> src(span64);
                l1mgr.read(t.src_addr + uint32_t(r0) * t.src_stride, src.data(),
                           uint32_t(span64));
                if (t.dst_stride == t.length) {
                    std::vector<uint8_t> dst(uint64_t(rows) * t.length);
                    for (uint32_t r = 0; r < rows; ++r) {
                        std::memcpy(dst.data() + uint64_t(r) * t.length,
                                    src.data() + uint64_t(r) * t.src_stride + col_off,
                                    t.length);
                    }
                    l1mgr.write(t.dst_addr, dst.data(), uint32_t(dst.size()));
                } else {
                    std::vector<uint8_t> row(t.length);
                    for (uint32_t r = 0; r < rows; ++r) {
                        std::memcpy(row.data(),
                                    src.data() + uint64_t(r) * t.src_stride + col_off,
                                    t.length);
                        l1mgr.write(t.dst_addr + uint64_t(r) * t.dst_stride,
                                    row.data(), t.length);
                    }
                }
                wait_bytes(uint64_t(t.length) * rows);
                return;
            }
        }
        std::vector<uint8_t> buf(t.length);
        for (uint16_t r = r0; r < r1; ++r) {
            const uint32_t s = t.src_addr + r * t.src_stride + col_off;
            const uint32_t d = t.dst_addr + (r - r0) * t.dst_stride;
            l1mgr.read(s, buf.data(), t.length);
            l1mgr.write(d, buf.data(), t.length);
        }
        wait_bytes(uint64_t(t.length) * (r1 - r0));
    }

    struct TnpsMeta {
        uint32_t rank = 0;
        uint32_t elem = 1;
        uint32_t in_shape[6] = {};
        uint32_t out_shape[6] = {};
        int32_t  a[6] = {};
        int32_t  b[6] = {};
    };

    bool read_meta(const TnpsBody& t, TnpsMeta& m) {
        if (!t.idx_table_addr) return false;
        uint32_t raw[26] = {};
        l1mgr.read(t.idx_table_addr, raw, sizeof(raw));
        m.rank = std::min<uint32_t>(raw[0], 6);
        m.elem = raw[1] ? raw[1] : 1;
        if (!m.rank) return false;
        for (uint32_t i = 0; i < 6; ++i) {
            m.in_shape[i] = raw[2 + i];
            m.out_shape[i] = raw[8 + i];
            m.a[i] = int32_t(raw[14 + i]);
            m.b[i] = int32_t(raw[20 + i]);
        }
        return true;
    }

    static uint64_t product(const uint32_t* shape, uint32_t rank) {
        uint64_t p = 1;
        for (uint32_t i = 0; i < rank; ++i) p *= shape[i] ? shape[i] : 1;
        return p;
    }

    static void strides_for(const uint32_t* shape, uint32_t rank, uint64_t* strides) {
        uint64_t s = 1;
        for (int i = int(rank) - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i] ? shape[i] : 1;
        }
    }

    void do_transpose_meta(const TnpsBody& t) {
        TnpsMeta m{};
        if (!read_meta(t, m)) {
            do_linear(t);
            return;
        }
        std::cout << "[TNPS] TRANSPOSE  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  rank=" << m.rank << "  elem=" << m.elem << "\n";
        const uint64_t in_elems = product(m.in_shape, m.rank);
        const uint64_t out_elems = product(m.out_shape, m.rank);
        std::vector<uint8_t> src(in_elems * m.elem);
        std::vector<uint8_t> dst(out_elems * m.elem);
        l1mgr.read(t.src_addr, src.data(), src.size());
        uint64_t in_strides[6] = {}, out_strides[6] = {};
        strides_for(m.in_shape, m.rank, in_strides);
        strides_for(m.out_shape, m.rank, out_strides);
        for (uint64_t out_idx = 0; out_idx < out_elems; ++out_idx) {
            uint64_t rem = out_idx;
            uint64_t in_idx = 0;
            for (uint32_t od = 0; od < m.rank; ++od) {
                const uint64_t coord = rem / out_strides[od];
                rem %= out_strides[od];
                const uint32_t id = uint32_t(m.a[od]);
                if (id < m.rank) in_idx += coord * in_strides[id];
            }
            std::memcpy(dst.data() + out_idx * m.elem,
                        src.data() + in_idx * m.elem, m.elem);
        }
        l1mgr.write(t.dst_addr, dst.data(), dst.size());
        wait_bytes(src.size() + dst.size());
    }

    void do_slice_meta(const TnpsBody& t) {
        TnpsMeta m{};
        if (!read_meta(t, m)) {
            do_slice(t);
            return;
        }
        std::cout << "[TNPS] STRIDED_SLICE_META  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  rank=" << m.rank << "  elem=" << m.elem << "\n";
        const uint64_t in_elems = product(m.in_shape, m.rank);
        const uint64_t out_elems = product(m.out_shape, m.rank);
        std::vector<uint8_t> src(in_elems * m.elem);
        std::vector<uint8_t> dst(out_elems * m.elem);
        l1mgr.read(t.src_addr, src.data(), src.size());
        uint64_t in_strides[6] = {}, out_strides[6] = {};
        strides_for(m.in_shape, m.rank, in_strides);
        strides_for(m.out_shape, m.rank, out_strides);
        for (uint64_t out_idx = 0; out_idx < out_elems; ++out_idx) {
            uint64_t rem = out_idx;
            uint64_t in_idx = 0;
            for (uint32_t d = 0; d < m.rank; ++d) {
                const uint64_t coord = rem / out_strides[d];
                rem %= out_strides[d];
                const int32_t begin = m.a[d];
                const int32_t stride = m.b[d] ? m.b[d] : 1;
                in_idx += uint64_t(begin + int32_t(coord) * stride) * in_strides[d];
            }
            std::memcpy(dst.data() + out_idx * m.elem,
                        src.data() + in_idx * m.elem, m.elem);
        }
        l1mgr.write(t.dst_addr, dst.data(), dst.size());
        wait_bytes(src.size() + dst.size());
    }

    void do_depth_to_space(const TnpsBody& t) {
        const uint32_t H = t.num_chunks;
        const uint32_t W = t.slice_begin[0];
        const uint32_t Cin = t.slice_begin[1];
        const uint32_t block = t.slice_begin[2];
        const uint32_t Cout = t.slice_begin[3];
        const uint32_t elem = t.length ? t.length : 1;
        std::cout << "[TNPS] DEPTH_TO_SPACE  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  in=" << H << "x" << W << "x" << Cin
                  << "  block=" << block << "  cout=" << Cout << "\n";
        if (!H || !W || !Cin || !block || !Cout || Cin != Cout * block * block) {
            wait(10, sc_core::SC_NS);
            return;
        }
        const uint64_t src_bytes = uint64_t(H) * t.src_stride;
        const uint64_t dst_bytes = uint64_t(H) * block * t.dst_stride;
        std::vector<uint8_t> src(src_bytes);
        std::vector<uint8_t> dst(dst_bytes);
        l1mgr.read(t.src_addr, src.data(), src.size());
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
                    const uint32_t dst_off = oh * t.dst_stride
                                           + (ow * Cout + oc) * elem;
                    std::memcpy(dst.data() + dst_off, src.data() + src_off, elem);
                }
            }
        }
        l1mgr.write(t.dst_addr, dst.data(), dst.size());
        wait_bytes(src_bytes + dst_bytes);
    }

    void do_space_to_depth(const TnpsBody& t) {
        const uint32_t H = t.num_chunks;
        const uint32_t W = t.slice_begin[0];
        const uint32_t Cin = t.slice_begin[1];
        const uint32_t block = t.slice_begin[2];
        const uint32_t Cout = t.slice_begin[3];
        const uint32_t elem = t.length ? t.length : 1;
        std::cout << "[TNPS] SPACE_TO_DEPTH  src=0x" << std::hex << t.src_addr
                  << "  dst=0x" << t.dst_addr << std::dec
                  << "  in=" << H << "x" << W << "x" << Cin
                  << "  block=" << block << "  cout=" << Cout << "\n";
        if (!H || !W || !Cin || !block || H % block || W % block ||
            Cout != Cin * block * block) {
            wait(10, sc_core::SC_NS);
            return;
        }
        const uint32_t OH = H / block;
        const uint32_t OW = W / block;
        const uint64_t src_bytes = uint64_t(H) * W * Cin * elem;
        const uint64_t dst_bytes = uint64_t(OH) * OW * Cout * elem;
        std::vector<uint8_t> src(src_bytes);
        std::vector<uint8_t> dst(dst_bytes);
        l1mgr.read(t.src_addr, src.data(), src.size());
        for (uint32_t oh = 0; oh < OH; ++oh) {
            for (uint32_t ow = 0; ow < OW; ++ow) {
                for (uint32_t bh = 0; bh < block; ++bh) {
                    for (uint32_t bw = 0; bw < block; ++bw) {
                        for (uint32_t ic = 0; ic < Cin; ++ic) {
                            const uint32_t ih = oh * block + bh;
                            const uint32_t iw = ow * block + bw;
                            const uint32_t oc = (bh * block + bw) * Cin + ic;
                            const uint64_t src_off =
                                (uint64_t(ih) * W * Cin + uint64_t(iw) * Cin + ic) * elem;
                            const uint64_t dst_off =
                                (uint64_t(oh) * OW * Cout + uint64_t(ow) * Cout + oc) * elem;
                            std::memcpy(dst.data() + dst_off, src.data() + src_off, elem);
                        }
                    }
                }
            }
        }
        l1mgr.write(t.dst_addr, dst.data(), dst.size());
        wait_bytes(src_bytes + dst_bytes);
    }
};

} // namespace mdla7
