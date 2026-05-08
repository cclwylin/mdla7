#pragma once

// L1Mesh + DRAM + L1_Manager
//
// HW spec: CONV ACT/WGT AXI_R connects directly to L1Mesh, bypassing
// L1Manager, so CONV reads get the highest service priority. L1Manager
// arbitrates non-CONV Engine/UDMA ingress. The current SystemC L1Manager below
// is still a simplified pass-through router; full priority contention is a
// future refinement.
//
// v2.1 / v3.2: L1Mesh has 16 banks (256-byte interleave) — concurrent
//       accesses to different banks proceed in parallel, only same-bank
//       accesses serialize.  Each bank port = 16 byte / cycle (one AXI 128b
//       lane).  16 banks × 16 byte = 256 B / cycle peak per direction
//       (matches spec §3.2 L1_Manager↔L1Mesh = 16 R + 16 W lanes).
//       AXI bursts use 16 beats × 16 B = 256 B as the transaction granularity.
//       The NoC fabric is modeled as two parallel 4x4 mesh planes sharing this
//       same 16-bank SRAM backend.
//
// v2.3: DRAM models LPDDR-class row-hit / row-miss latency.
// v3.1: + DRAM refresh — periodic stall when crossing a refresh boundary.
// v8.25: spec frequency bump to 1.9 GHz + DRAM dual-channel LPDDR5X-10667.
//        Bandwidth math at 1.9 GHz, dual x32:
//          10.667 Gbps/pin × 32 pins × 2 ch / 8 = 85.3 GB/s
//          85.3 GB/s ÷ 1.9 G cycle/s ≈ 44.9 → BYTES_PER_CYCLE = 48
//        Row-miss tRP+tRCD ≈ 26 ns × 1.9 GHz ≈ 50 cycles (LPDDR5 tighter
//        than LPDDR6's 30-cyc baseline at 1 GHz).
//        Refresh: tREFI ≈ 3.9 µs × 1.9 GHz = 7410 cycles → keep 7800 (close
//        enough; sim cycle scale is abstract — what matters is the ratio
//        REFRESH_STALL/REFRESH_PERIOD ≈ 200/7800 = 2.6%, matching real
//        LPDDR5 tRFC/tREFI overhead).
// v8.32: L1Mesh SRAM runs at 1.3 GHz while the simulator cycle axis remains
//        core-clock based at 1.9 GHz. One SRAM beat therefore costs
//        1.9 / 1.3 core cycles.
// v8.33: selectable L1 timing:
//        FastEstimate = aggregate 16-bank bandwidth estimate (default);
//        PortConflict = per-bank finish-array SRAM port conflict model.
//        MeshConflict = dual-4x4 mesh edge/router/link + SRAM macro conflict model.

#include <systemc>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include "mdla7/descriptor.h"

namespace mdla7 {

enum class L1TimingMode {
    FastEstimate,
    PortConflict,
    MeshConflict,
};

class L1Mesh : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(L1Mesh);
    L1Mesh(sc_core::sc_module_name nm,
           std::size_t bytes = L1MESH_BYTES,
           L1TimingMode timing_mode = L1TimingMode::FastEstimate)
      : sc_module(nm), timing_mode_(timing_mode), mem(bytes, 0) {}

    void read(uint32_t offset, void* dst, uint32_t n) {
        std::memcpy(dst, &mem[offset], n);
        if (in_process()) impose_latency(read_bank_finish_, offset, n);
    }
    void write(uint32_t offset, const void* src, uint32_t n) {
        std::memcpy(&mem[offset], src, n);
        if (in_process()) impose_latency(write_bank_finish_, offset, n);
    }
    std::size_t size() const { return mem.size(); }
    void set_timing_mode(L1TimingMode mode) { timing_mode_ = mode; }
    L1TimingMode timing_mode() const { return timing_mode_; }

private:
    static constexpr unsigned N_BANKS         = 16;
    // 16-byte stripe = one AXI 128b beat. Sequential access fans out across
    // all 16 banks → 256 B/cycle peak (matches spec §3.2).
    static constexpr unsigned AXI_BEAT_BYTES  = 16;
    static constexpr unsigned AXI_BURST_BEATS = 16;
    static constexpr unsigned AXI_BURST_BYTES = AXI_BURST_BEATS * AXI_BEAT_BYTES;
    static constexpr unsigned BANK_STRIDE     = AXI_BEAT_BYTES;
    static constexpr unsigned BYTES_PER_CYCLE = 16;    // per-bank AXI 128b lane
    static constexpr double   CORE_CLOCK_GHZ  = 1.9;
    static constexpr double   SRAM_CLOCK_GHZ  = 1.3;

    static bool in_process() {
        return sc_core::sc_get_current_process_handle().valid();
    }

    void impose_latency(sc_core::sc_time bank_finish[N_BANKS],
                        uint32_t offset, uint32_t bytes) {
        if (timing_mode_ == L1TimingMode::MeshConflict)
            impose_mesh_latency(bank_finish, offset, bytes,
                                bank_finish == read_bank_finish_);
        else if (timing_mode_ == L1TimingMode::PortConflict)
            impose_bank_latency(bank_finish, offset, bytes);
        else
            impose_fast_latency(bytes);
    }

    // Fast mode: aggregate-bandwidth estimate. It preserves the 16-bank
    // sequential peak but skips per-stripe finish-array conflict accounting.
    void impose_fast_latency(uint32_t bytes) {
        const uint32_t aggregate_bpc = AXI_BURST_BYTES;
        const double beats = double((bytes + aggregate_bpc - 1) / aggregate_bpc);
        const sc_core::sc_time access(
            beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
            sc_core::SC_NS);
        if (access != sc_core::SC_ZERO_TIME) sc_core::wait(access);
    }

    // Conflict mode: model 16 banks. An access spanning multiple banks accumulates
    // their per-bank wait times. The longest bank latency wins (parallel),
    // since simultaneous bank accesses overlap.
    void impose_bank_latency(sc_core::sc_time bank_finish[N_BANKS],
                             uint32_t offset, uint32_t bytes) {
        const sc_core::sc_time now = sc_core::sc_time_stamp();
        sc_core::sc_time max_finish = now;
        uint32_t end = offset + bytes;
        // Walk every bank stripe touched.
        for (uint32_t a = offset; a < end; ) {
            uint32_t bank = (a / BANK_STRIDE) % N_BANKS;
            uint32_t next_stripe = ((a / BANK_STRIDE) + 1) * BANK_STRIDE;
            uint32_t chunk = std::min(end, next_stripe) - a;
            const double beats = double((chunk + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE);
            const sc_core::sc_time access(
                beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
                sc_core::SC_NS);
            const sc_core::sc_time start =
                (bank_finish[bank] > now) ? bank_finish[bank] : now;
            const sc_core::sc_time finish = start + access;
            bank_finish[bank] = finish;
            if (finish > max_finish) max_finish = finish;
            a += chunk;
        }
        if (max_finish > now) sc_core::wait(max_finish - now);
    }

    static sc_core::sc_time service_one_cycle(sc_core::sc_time& finish,
                                              sc_core::sc_time now) {
        const sc_core::sc_time access(1.0, sc_core::SC_NS);
        const sc_core::sc_time start = (finish > now) ? finish : now;
        finish = start + access;
        return finish;
    }

    static sc_core::sc_time service_sram_beat(sc_core::sc_time& finish,
                                              sc_core::sc_time now,
                                              uint32_t bytes) {
        const double beats = double((bytes + BYTES_PER_CYCLE - 1) /
                                    BYTES_PER_CYCLE);
        const sc_core::sc_time access(
            beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
            sc_core::SC_NS);
        const sc_core::sc_time start = (finish > now) ? finish : now;
        finish = start + access;
        return finish;
    }

    // Mesh mode: 16 SRAM banks sit behind two parallel 4x4 mesh planes. Each
    // 16B bank stripe is one flit. The request enters from a deterministic edge
    // port, takes XY routing through one-flit/cycle directed links on the less
    // busy plane, then arbitrates for the shared target bank's 16B/cycle SRAM
    // macro port.
    //
    // This is intentionally still an architectural timing model, not a
    // cycle-accurate packet network: current read/write calls do not carry the
    // requester class (CONV ACT, CONV WGT, UDMA, EWE, ...), so priority is
    // approximated by separate read/write edge ingress while all internal mesh
    // links are shared.
    void impose_mesh_latency(sc_core::sc_time bank_finish[N_BANKS],
                             uint32_t offset, uint32_t bytes,
                             bool is_read) {
        const sc_core::sc_time now = sc_core::sc_time_stamp();
        sc_core::sc_time max_finish = now;
        const uint32_t end = offset + bytes;
        for (uint32_t a = offset; a < end; ) {
            const uint32_t bank = (a / BANK_STRIDE) % N_BANKS;
            const uint32_t dst_x = bank % MESH_W;
            const uint32_t dst_y = bank / MESH_W;
            const uint32_t src_x = is_read ? 0 : (MESH_W - 1);
            const uint32_t src_y = dst_y;
            const uint32_t next_stripe = ((a / BANK_STRIDE) + 1) * BANK_STRIDE;
            const uint32_t chunk = std::min(end, next_stripe) - a;

            const unsigned plane = pick_mesh_plane(now, src_x, src_y,
                                                   dst_x, dst_y, is_read);
            sc_core::sc_time t = route_on_mesh_plane(plane, now, src_x, src_y,
                                                     dst_x, dst_y, is_read);
            t = service_sram_beat(bank_finish[bank], t, chunk);
            if (t > max_finish) max_finish = t;
            a += chunk;
        }
        if (max_finish > now) sc_core::wait(max_finish - now);
    }

    static constexpr unsigned MESH_PLANES = 2;
    static constexpr unsigned MESH_W = 4;
    static constexpr unsigned MESH_H = 4;
    enum MeshDir : unsigned { DIR_N = 0, DIR_E = 1, DIR_S = 2,
                              DIR_W = 3, DIR_LOCAL = 4 };
    static constexpr unsigned node_id(unsigned x, unsigned y) {
        return y * MESH_W + x;
    }

    sc_core::sc_time reserve_virtual_cycle(const sc_core::sc_time& finish,
                                           sc_core::sc_time now) const {
        const sc_core::sc_time access(1.0, sc_core::SC_NS);
        const sc_core::sc_time start = (finish > now) ? finish : now;
        return start + access;
    }

    sc_core::sc_time estimate_mesh_plane(unsigned plane, sc_core::sc_time now,
                                         uint32_t src_x, uint32_t src_y,
                                         uint32_t dst_x, uint32_t dst_y,
                                         bool is_read) const {
        sc_core::sc_time t = now;
        t = reserve_virtual_cycle(mesh_edge_finish_[plane][is_read ? 0 : 1][src_y], t);

        uint32_t x = src_x;
        uint32_t y = src_y;
        while (x != dst_x) {
            const bool east = x < dst_x;
            const uint32_t out_dir = east ? DIR_E : DIR_W;
            t = reserve_virtual_cycle(mesh_router_out_finish_[plane][node_id(x, y)][out_dir], t);
            if (east) {
                t = reserve_virtual_cycle(mesh_hlink_finish_[plane][y][x][0], t);
                ++x;
            } else {
                t = reserve_virtual_cycle(mesh_hlink_finish_[plane][y][x - 1][1], t);
                --x;
            }
        }
        while (y != dst_y) {
            const bool south = y < dst_y;
            const uint32_t out_dir = south ? DIR_S : DIR_N;
            t = reserve_virtual_cycle(mesh_router_out_finish_[plane][node_id(x, y)][out_dir], t);
            if (south) {
                t = reserve_virtual_cycle(mesh_vlink_finish_[plane][y][x][0], t);
                ++y;
            } else {
                t = reserve_virtual_cycle(mesh_vlink_finish_[plane][y - 1][x][1], t);
                --y;
            }
        }
        return reserve_virtual_cycle(mesh_router_out_finish_[plane][node_id(x, y)][DIR_LOCAL], t);
    }

    unsigned pick_mesh_plane(sc_core::sc_time now,
                             uint32_t src_x, uint32_t src_y,
                             uint32_t dst_x, uint32_t dst_y,
                             bool is_read) const {
        unsigned best = 0;
        sc_core::sc_time best_t = estimate_mesh_plane(0, now, src_x, src_y,
                                                      dst_x, dst_y, is_read);
        for (unsigned p = 1; p < MESH_PLANES; ++p) {
            const sc_core::sc_time t = estimate_mesh_plane(p, now, src_x, src_y,
                                                           dst_x, dst_y, is_read);
            if (t < best_t) {
                best = p;
                best_t = t;
            }
        }
        return best;
    }

    sc_core::sc_time route_on_mesh_plane(unsigned plane, sc_core::sc_time now,
                                         uint32_t src_x, uint32_t src_y,
                                         uint32_t dst_x, uint32_t dst_y,
                                         bool is_read) {
        sc_core::sc_time t = now;
        t = service_one_cycle(mesh_edge_finish_[plane][is_read ? 0 : 1][src_y], t);

        uint32_t x = src_x;
        uint32_t y = src_y;
        while (x != dst_x) {
            const bool east = x < dst_x;
            const uint32_t out_dir = east ? DIR_E : DIR_W;
            t = service_one_cycle(mesh_router_out_finish_[plane][node_id(x, y)][out_dir], t);
            if (east) {
                t = service_one_cycle(mesh_hlink_finish_[plane][y][x][0], t);
                ++x;
            } else {
                t = service_one_cycle(mesh_hlink_finish_[plane][y][x - 1][1], t);
                --x;
            }
        }
        while (y != dst_y) {
            const bool south = y < dst_y;
            const uint32_t out_dir = south ? DIR_S : DIR_N;
            t = service_one_cycle(mesh_router_out_finish_[plane][node_id(x, y)][out_dir], t);
            if (south) {
                t = service_one_cycle(mesh_vlink_finish_[plane][y][x][0], t);
                ++y;
            } else {
                t = service_one_cycle(mesh_vlink_finish_[plane][y - 1][x][1], t);
                --y;
            }
        }
        return service_one_cycle(mesh_router_out_finish_[plane][node_id(x, y)][DIR_LOCAL], t);
    }

    L1TimingMode timing_mode_;
    std::vector<uint8_t> mem;
    sc_core::sc_time read_bank_finish_ [N_BANKS];
    sc_core::sc_time write_bank_finish_[N_BANKS];
    sc_core::sc_time mesh_edge_finish_[MESH_PLANES][2][MESH_H];
    sc_core::sc_time mesh_router_out_finish_[MESH_PLANES][N_BANKS][5];
    sc_core::sc_time mesh_hlink_finish_[MESH_PLANES][MESH_H][MESH_W - 1][2];
    sc_core::sc_time mesh_vlink_finish_[MESH_PLANES][MESH_H - 1][MESH_W][2];
};

class Dram : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(Dram);
    Dram(sc_core::sc_module_name nm, std::size_t bytes = 256 * 1024 * 1024)
      : sc_module(nm), mem(bytes, 0) {
        for (auto& r : open_row_) r = -1;
    }

    void read(uint32_t addr, void* dst, uint32_t n) {
        std::memcpy(dst, &mem[addr - DRAM_BASE], n);
        if (in_process()) impose_latency(addr, n);
    }
    void read_compressed(uint32_t addr, void* dst, uint32_t raw_n,
                         uint32_t compressed_n) {
        std::memcpy(dst, &mem[addr - DRAM_BASE], raw_n);
        if (in_process()) impose_latency(addr, compressed_n);
    }
    void write(uint32_t addr, const void* src, uint32_t n) {
        std::memcpy(&mem[addr - DRAM_BASE], src, n);
        if (in_process()) impose_latency(addr, n);
    }
    void write_compressed(uint32_t addr, const void* src, uint32_t raw_n,
                          uint32_t compressed_n) {
        std::memcpy(&mem[addr - DRAM_BASE], src, raw_n);
        if (in_process()) impose_latency(addr, compressed_n);
    }
    std::size_t size() const { return mem.size(); }

private:
    static constexpr uint32_t ROW_BYTES        = 8 * 1024;   // 8 KB row
    static constexpr uint32_t N_BANKS          = 16;
    // v8.25: dual-channel LPDDR5X-10667 @ 1.9 GHz core clock.
    //   85.3 GB/s ÷ 1.9 G cyc/s = 44.9 B/cyc → BYTES_PER_CYCLE = 48.
    //   tRP+tRCD ≈ 26 ns at 1.9 GHz = ~50 cycles.
    //   Refresh tuned for LPDDR5 tRFC/tREFI ratio.
    //   AXI burst length = 16 beats. Since each beat is 128b/16B, DRAM timing
    //   charges whole 256B burst windows touched by an access.
    // (v8.11 LPDDR6 baseline at 1 GHz was 64 B/cyc + 30 cyc miss + 100 cyc
    //  refresh-stall; replaced here to reflect the new spec.)
    static constexpr unsigned AXI_BEAT_BYTES   = 16;
    static constexpr unsigned AXI_BURST_BEATS  = 16;
    static constexpr unsigned AXI_BURST_BYTES  = AXI_BURST_BEATS * AXI_BEAT_BYTES;
    static constexpr unsigned BYTES_PER_CYCLE  = 48;
    static constexpr unsigned ROW_MISS_PENALTY = 50;          // cycles
    static constexpr unsigned REFRESH_PERIOD   = 7800;        // cycles (~ tREFI / 8)
    static constexpr unsigned REFRESH_STALL    = 200;         // cycles per refresh

    static bool in_process() {
        return sc_core::sc_get_current_process_handle().valid();
    }

    void impose_latency(uint32_t addr, uint32_t bytes) {
        const uint32_t off  = addr - DRAM_BASE;
        const uint32_t bank = (off / ROW_BYTES) % N_BANKS;
        const int32_t  row  = int32_t((off / ROW_BYTES) / N_BANKS);
        const sc_core::sc_time penalty(
            (open_row_[bank] == row) ? 0.0 : double(ROW_MISS_PENALTY),
            sc_core::SC_NS);
        const uint32_t burst_bytes = charged_burst_bytes(off, bytes);
        const sc_core::sc_time access(
            double((burst_bytes + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE),
            sc_core::SC_NS);
        const sc_core::sc_time now = sc_core::sc_time_stamp();
        sc_core::sc_time start = (last_finish_ > now) ? last_finish_ : now;

        // v3.1: refresh — every REFRESH_PERIOD ns we lose REFRESH_STALL cycles.
        const uint64_t cur_period = uint64_t(start.to_seconds() * 1e9) / REFRESH_PERIOD;
        if (cur_period > last_refresh_period_) {
            const uint64_t missed = cur_period - last_refresh_period_;
            start += sc_core::sc_time(double(missed * REFRESH_STALL), sc_core::SC_NS);
            last_refresh_period_ = cur_period;
        }

        const sc_core::sc_time finish = start + penalty + access;
        last_finish_ = finish;
        open_row_[bank] = row;
        if (finish > now) sc_core::wait(finish - now);
    }

    static uint32_t charged_burst_bytes(uint32_t off, uint32_t bytes) {
        if (bytes == 0) return 0;
        const uint32_t first = off / AXI_BURST_BYTES;
        const uint32_t last  = (off + bytes - 1) / AXI_BURST_BYTES;
        return (last - first + 1) * AXI_BURST_BYTES;
    }

    std::vector<uint8_t> mem;
    sc_core::sc_time last_finish_ {sc_core::SC_ZERO_TIME};
    int32_t open_row_[N_BANKS];
    uint64_t last_refresh_period_ = 0;
};

// v0: simplified pass-through. HW L1_Manager arbitrates non-CONV Engine/UDMA
// ingress; CONV ACT/WGT reads bypass it through direct L1Mesh AXI_R paths.
class L1Manager : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(L1Manager);
    L1Manager(sc_core::sc_module_name nm, L1Mesh& mesh, Dram& dram)
      : sc_module(nm), mesh_(mesh), dram_(dram) {}

    void read(uint32_t addr, void* dst, uint32_t n) {
        if (addr_in_l1mesh(addr)) mesh_.read(addr, dst, n);
        else if (addr_in_dram(addr)) dram_.read(addr, dst, n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    void read_compressed(uint32_t addr, void* dst, uint32_t raw_n,
                         uint32_t compressed_n) {
        if (addr_in_l1mesh(addr)) mesh_.read(addr, dst, raw_n);
        else if (addr_in_dram(addr)) dram_.read_compressed(addr, dst, raw_n, compressed_n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    void write(uint32_t addr, const void* src, uint32_t n) {
        if (addr_in_l1mesh(addr)) mesh_.write(addr, src, n);
        else if (addr_in_dram(addr)) dram_.write(addr, src, n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    void write_compressed(uint32_t addr, const void* src, uint32_t raw_n,
                          uint32_t compressed_n) {
        if (addr_in_l1mesh(addr)) mesh_.write(addr, src, raw_n);
        else if (addr_in_dram(addr)) dram_.write_compressed(addr, src, raw_n, compressed_n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }

private:
    L1Mesh& mesh_;
    Dram& dram_;
};

} // namespace mdla7
