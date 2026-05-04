#pragma once

// L1Mesh + DRAM + L1_Manager
//
// v2.1 / v3.2: L1Mesh has 16 banks (256-byte interleave) — concurrent
//       accesses to different banks proceed in parallel, only same-bank
//       accesses serialize.  Each bank port = 16 byte / cycle (one AXI 128b
//       lane).  16 banks × 16 byte = 256 B / cycle peak per direction
//       (matches spec §3.2 L1_Manager↔L1Mesh = 16 R + 16 W lanes).
//
// v2.3: DRAM models LPDDR6 row-hit / row-miss latency.
//       Bandwidth: 32 byte / cycle (one LPDDR6 channel @ 1 GHz).
//       Page miss penalty: 30 cycles (precharge + activate).
// v3.1: + DRAM refresh — every 7800 ns inject 100 cycles of stall.
//       Implemented by snapping the scheduling cursor forward when crossing
//       a refresh boundary.

#include <systemc>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>
#include "mdla7/descriptor.h"

namespace mdla7 {

class L1Mesh : public sc_core::sc_module {
public:
    SC_HAS_PROCESS(L1Mesh);
    L1Mesh(sc_core::sc_module_name nm, std::size_t bytes = L1MESH_BYTES)
      : sc_module(nm), mem(bytes, 0) {}

    void read(uint32_t offset, void* dst, uint32_t n) {
        std::memcpy(dst, &mem[offset], n);
        if (in_process()) impose_bank_latency(read_bank_finish_, offset, n);
    }
    void write(uint32_t offset, const void* src, uint32_t n) {
        std::memcpy(&mem[offset], src, n);
        if (in_process()) impose_bank_latency(write_bank_finish_, offset, n);
    }
    std::size_t size() const { return mem.size(); }

private:
    static constexpr unsigned N_BANKS         = 16;
    // 16-byte stripe = one AXI 128b beat. Sequential access fans out across
    // all 16 banks → 256 B/cycle peak (matches spec §3.2).
    static constexpr unsigned BANK_STRIDE     = 16;
    static constexpr unsigned BYTES_PER_CYCLE = 16;    // per-bank AXI 128b lane

    static bool in_process() {
        return sc_core::sc_get_current_process_handle().valid();
    }

    // v3.2: model 16 banks. An access spanning multiple banks accumulates
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
            const sc_core::sc_time access(
                double((chunk + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE),
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

    std::vector<uint8_t> mem;
    sc_core::sc_time read_bank_finish_ [N_BANKS];
    sc_core::sc_time write_bank_finish_[N_BANKS];
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
    void write(uint32_t addr, const void* src, uint32_t n) {
        std::memcpy(&mem[addr - DRAM_BASE], src, n);
        if (in_process()) impose_latency(addr, n);
    }
    std::size_t size() const { return mem.size(); }

private:
    static constexpr uint32_t ROW_BYTES        = 8 * 1024;   // 8 KB row
    static constexpr uint32_t N_BANKS          = 16;
    // v8.11: spec bump from 1×LPDDR6 (32 B/cyc) to dual-channel LPDDR6
    // (64 B/cyc).  Halves DRAM-bound layer time without changing the row /
    // refresh model.  Single-channel was the v0 baseline and undersized for
    // the 65,536 MAC peak compute; 64 B/cyc better matches what a
    // mobile-class NPU actually ships with (e.g., Apple A-series, Tegra X2).
    static constexpr unsigned BYTES_PER_CYCLE  = 64;
    static constexpr unsigned ROW_MISS_PENALTY = 30;          // cycles
    static constexpr unsigned REFRESH_PERIOD   = 7800;        // ns (~ tREFI / 8)
    static constexpr unsigned REFRESH_STALL    = 100;         // cycles per refresh

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
        const sc_core::sc_time access(
            double((bytes + BYTES_PER_CYCLE - 1) / BYTES_PER_CYCLE),
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

    std::vector<uint8_t> mem;
    sc_core::sc_time last_finish_ {sc_core::SC_ZERO_TIME};
    int32_t open_row_[N_BANKS];
    uint64_t last_refresh_period_ = 0;
};

// v0: thin pass-through. Real L1_Manager will arbitrate among Engines + UDMA.
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
    void write(uint32_t addr, const void* src, uint32_t n) {
        if (addr_in_l1mesh(addr)) mesh_.write(addr, src, n);
        else if (addr_in_dram(addr)) dram_.write(addr, src, n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }

private:
    L1Mesh& mesh_;
    Dram& dram_;
};

} // namespace mdla7
