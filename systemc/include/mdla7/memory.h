#pragma once

// L1Mesh + DRAM + L1_Manager
//
// HW spec: CONV ACT_R and WGT_R Payload reads have separate dedicated links
// into L1Mesh, bypassing L1Manager, so CONV reads get the highest service
// priority. L1Manager arbitrates non-CONV Engine/UDMA ingress. The current
// SystemC L1Manager below is still a simplified pass-through router; full
// priority contention is a future refinement.
//
// v2.1 / v3.2: L1Mesh has 16 banks (256-byte interleave) — concurrent
//       accesses to different banks proceed in parallel, only same-bank
//       accesses serialize.  Each SRAM macro port = one 16-byte Payload beat
//       per SRAM cycle and is 1R/W, not independent 1R+1W.  16 banks × 16 byte =
//       256 B / cycle peak aggregate backend bandwidth
//       (matches spec §3.2 L1_Manager↔L1Mesh = 16R + 16W Payload lanes).
//       Payload is an internal per-beat protocol; tid + last groups beats into
//       a logical transaction and no burst metadata is carried.
//       The NoC fabric is modeled as two parallel 4x4 mesh planes sharing this
//       same 16-bank SRAM backend.
//       Router input FIFO depth is provisionally 2 flits; current timing only
//       models edge/router-output/link finish times, not input backpressure.
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
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include "mdla7/descriptor.h"

namespace mdla7 {

enum class L1TimingMode {
    FastEstimate,
    PortConflict,
    MeshConflict,
    MeshOptimistic,
};

enum class EngineModel {
    Analytical,
    RtlStyle,
};

inline bool is_rtl_style(EngineModel model) {
    return model == EngineModel::RtlStyle;
}

inline const char* engine_model_name(EngineModel model) {
    return is_rtl_style(model) ? "rtl" : "model";
}

class L1Mesh : public sc_core::sc_module {
public:
    struct Stats {
        struct Lane {
            uint64_t accesses = 0;
            uint64_t bytes = 0;
            double latency_ns = 0.0;
            double wait_ns = 0.0;
            double service_ns = 0.0;
            double max_latency_ns = 0.0;
            double max_wait_ns = 0.0;
            double max_service_ns = 0.0;
        };
        uint64_t accesses = 0;
        uint64_t bytes = 0;
        uint64_t stripes = 0;
        uint64_t chunks = 0;
        double edge_wait_ns = 0.0, edge_service_ns = 0.0;
        double router_wait_ns = 0.0, router_service_ns = 0.0;
        double link_wait_ns = 0.0, link_service_ns = 0.0;
        double local_wait_ns = 0.0, local_service_ns = 0.0;
        double sram_wait_ns = 0.0, sram_service_ns = 0.0;
        double imposed_wait_ns = 0.0;
        std::array<Lane, 16> read_lane{};
        std::array<Lane, 16> write_lane{};
    };

    struct AccessTicket {
        sc_core::sc_time done{sc_core::SC_ZERO_TIME};
    };

    SC_HAS_PROCESS(L1Mesh);
    L1Mesh(sc_core::sc_module_name nm,
           std::size_t bytes = L1MESH_BYTES,
           L1TimingMode timing_mode = L1TimingMode::FastEstimate)
      : sc_module(nm), timing_mode_(timing_mode), mem(bytes, 0) {
        if (const char* p = std::getenv("MDLA7_L1_PAYLOAD_PROBE")) {
            if (*p) open_payload_probe(p);
        } else if (const char* p = std::getenv("MDLA7_L1_AXI_PROBE")) {
            if (*p) open_payload_probe(p);
        }
    }
    ~L1Mesh() override { flush_payload_probe_row(); }

    void read(uint32_t offset, void* dst, uint32_t n) {
        AccessTicket t = read_async(offset, dst, n);
        wait_ticket(t);
    }
    void write(uint32_t offset, const void* src, uint32_t n) {
        AccessTicket t = write_async(offset, src, n);
        wait_ticket(t);
    }
    void write_instant(uint32_t offset, const void* src, uint32_t n) {
        std::memcpy(&mem[offset], src, n);
    }
    AccessTicket read_async(uint32_t offset, void* dst, uint32_t n) {
        std::memcpy(dst, &mem[offset], n);
        return in_process() ? AccessTicket{schedule_latency(offset, n, true)}
                            : AccessTicket{sc_core::sc_time_stamp()};
    }
    AccessTicket write_async(uint32_t offset, const void* src, uint32_t n) {
        std::memcpy(&mem[offset], src, n);
        return in_process() ? AccessTicket{schedule_latency(offset, n, false)}
                            : AccessTicket{sc_core::sc_time_stamp()};
    }
    AccessTicket write_strided_rows(uint32_t dst, const void* src,
                                    uint32_t rows, uint32_t src_row,
                                    uint32_t dst_row, uint32_t dst_col) {
        if (!rows || !src_row || !dst_row || dst_col + src_row > dst_row)
            return AccessTicket{sc_core::sc_time_stamp()};
        const auto* s = static_cast<const uint8_t*>(src);
        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(&mem[dst + uint64_t(r) * dst_row + dst_col],
                        s + uint64_t(r) * src_row,
                        src_row);
        }
        return in_process() ? AccessTicket{schedule_latency(dst + dst_col, rows * src_row, false)}
                            : AccessTicket{sc_core::sc_time_stamp()};
    }
    AccessTicket channel_pack(uint32_t src, uint32_t dst,
                              uint32_t rows, uint32_t src_row,
                              uint32_t src_stride, uint32_t dst_row,
                              uint32_t dst_col) {
        if (!rows || !src_row || !dst_row || dst_col + src_row > dst_row)
            return AccessTicket{sc_core::sc_time_stamp()};
        if (!src_stride) src_stride = src_row;
        for (uint32_t r = 0; r < rows; ++r) {
            std::memcpy(&mem[dst + uint64_t(r) * dst_row + dst_col],
                        &mem[src + uint64_t(r) * src_stride],
                        src_row);
        }
        if (!in_process()) return AccessTicket{sc_core::sc_time_stamp()};
        const sc_core::sc_time src_done =
            (src_stride == src_row)
                ? schedule_latency(src, rows * src_row, true)
                : schedule_latency(src, rows * src_stride, true);
        const sc_core::sc_time dst_done =
            schedule_latency(dst, rows * dst_row, false);
        return AccessTicket{std::max(src_done, dst_done)};
    }
    static void wait_ticket(const AccessTicket& t) {
        const sc_core::sc_time now = sc_core::sc_time_stamp();
        if (in_process() && t.done > now) sc_core::wait(t.done - now);
    }
    std::size_t size() const { return mem.size(); }
    void set_timing_mode(L1TimingMode mode) { timing_mode_ = mode; }
    L1TimingMode timing_mode() const { return timing_mode_; }
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }

private:
    static constexpr unsigned N_BANKS         = 16;
    // 16-byte stripe = one Payload beat. Sequential access fans out across
    // all 16 banks → 256 B/cycle peak (matches spec §3.2).
    static constexpr unsigned PAYLOAD_BEAT_BYTES = PAYLOAD_BYTES;
    static constexpr unsigned PAYLOAD_FAST_WINDOW_BYTES =
        PayloadPortCount::L1MESH_R * PAYLOAD_BEAT_BYTES;
    static constexpr unsigned PAYLOAD_SCHED_CHUNK_BEATS = 16;
    static constexpr unsigned PAYLOAD_SCHED_CHUNK_BYTES =
        PayloadPortCount::L1MESH_R * PAYLOAD_BEAT_BYTES *
        PAYLOAD_SCHED_CHUNK_BEATS;
    static constexpr unsigned BANK_STRIDE     = PAYLOAD_BEAT_BYTES;
    static constexpr unsigned BYTES_PER_CYCLE = PAYLOAD_BEAT_BYTES; // per-bank SRAM macro
    static constexpr double   CORE_CLOCK_GHZ  = 1.9;
    static constexpr double   SRAM_CLOCK_GHZ  = 1.3;

    static bool in_process() {
        return sc_core::sc_get_current_process_handle().valid();
    }

    sc_core::sc_time schedule_latency(uint32_t offset, uint32_t bytes,
                                      bool is_read) {
        if (timing_mode_ == L1TimingMode::MeshConflict ||
            timing_mode_ == L1TimingMode::MeshOptimistic)
            return schedule_mesh_latency(offset, bytes, is_read);
        else if (timing_mode_ == L1TimingMode::PortConflict)
            return schedule_bank_latency(offset, bytes, is_read);
        return schedule_fast_latency(bytes);
    }

    // Fast mode: aggregate-bandwidth estimate. It preserves the 16-bank
    // sequential peak but skips per-stripe finish-array conflict accounting.
    sc_core::sc_time schedule_fast_latency(uint32_t bytes) {
        const uint32_t aggregate_bpc = PAYLOAD_FAST_WINDOW_BYTES;
        const double beats = double((bytes + aggregate_bpc - 1) / aggregate_bpc);
        const sc_core::sc_time access(
            beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
            sc_core::SC_NS);
        return sc_core::sc_time_stamp() + access;
    }

    // Conflict mode: model 16 banks. An access spanning multiple banks accumulates
    // their per-bank wait times. The longest bank latency wins (parallel),
    // since simultaneous bank accesses overlap.
    sc_core::sc_time schedule_bank_latency(uint32_t offset, uint32_t bytes,
                                           bool is_read) {
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
                (sram_bank_finish_[bank] > now) ? sram_bank_finish_[bank] : now;
            probe_payload_input(start, is_read, bank, a);
            const sc_core::sc_time finish = start + access;
            sram_bank_finish_[bank] = finish;
            if (finish > max_finish) max_finish = finish;
            a += chunk;
        }
        return max_finish;
    }

    static sc_core::sc_time service_one_cycle(sc_core::sc_time& finish,
                                              sc_core::sc_time now) {
        const sc_core::sc_time access(1.0, sc_core::SC_NS);
        const sc_core::sc_time start = (finish > now) ? finish : now;
        finish = start + access;
        return finish;
    }

    static double ns(sc_core::sc_time t) {
        return t.to_seconds() * 1e9;
    }

    sc_core::sc_time reserve_resource(sc_core::sc_time& finish,
                                      sc_core::sc_time now,
                                      uint32_t beats,
                                      double& wait_ns,
                                      double& service_ns) {
        if (beats == 0) return now;
        const sc_core::sc_time access(double(beats), sc_core::SC_NS);
        const sc_core::sc_time start = (finish > now) ? finish : now;
        wait_ns += ns(start - now);
        service_ns += ns(access);
        finish = start + access;
        return start;
    }

    sc_core::sc_time service_sram_beat(sc_core::sc_time& finish,
                                       sc_core::sc_time now,
                                       uint32_t bytes) {
        const double beats = double((bytes + BYTES_PER_CYCLE - 1) /
                                    BYTES_PER_CYCLE);
        const sc_core::sc_time access(
            beats * (CORE_CLOCK_GHZ / SRAM_CLOCK_GHZ),
            sc_core::SC_NS);
        const sc_core::sc_time start = (finish > now) ? finish : now;
        stats_.sram_wait_ns += ns(start - now);
        stats_.sram_service_ns += ns(access);
        finish = start + access;
        return finish;
    }

    static uint32_t ceil_div(uint32_t a, uint32_t b) {
        return (a + b - 1) / b;
    }

    static uint32_t swizzled_bank(uint32_t stripe) {
        // Rotate each 256B super-stripe across the 4x4 physical mesh. Linear
        // 16B striping already uses all banks inside one Payload window; the XOR
        // term prevents repeated transformer slices with matching offsets from
        // hammering the same physical row/column every block.
        return (stripe ^ (stripe >> 4)) & (N_BANKS - 1);
    }

    static sc_core::sc_time max_time(sc_core::sc_time a, sc_core::sc_time b) {
        return (a > b) ? a : b;
    }

    void reserve_route(uint32_t src_x, uint32_t src_y,
                       uint32_t dst_x, uint32_t dst_y,
                       uint32_t beats,
                       sc_core::sc_time now,
                       unsigned plane,
                       sc_core::sc_time& ready) {
        uint32_t x = src_x;
        uint32_t y = src_y;
        while (x != dst_x) {
            const bool east = x < dst_x;
            const uint32_t out_dir = east ? DIR_E : DIR_W;
            sc_core::sc_time start = reserve_resource(
                mesh_router_out_finish_[plane][node_id(x, y)][out_dir], now, beats,
                stats_.router_wait_ns, stats_.router_service_ns);
            ready = max_time(ready, start);
            if (east) {
                start = reserve_resource(mesh_hlink_finish_[plane][y][x][0], now, beats,
                                         stats_.link_wait_ns, stats_.link_service_ns);
                ready = max_time(ready, start);
                ++x;
            } else {
                start = reserve_resource(mesh_hlink_finish_[plane][y][x - 1][1], now, beats,
                                         stats_.link_wait_ns, stats_.link_service_ns);
                ready = max_time(ready, start);
                --x;
            }
        }
        while (y != dst_y) {
            const bool south = y < dst_y;
            const uint32_t out_dir = south ? DIR_S : DIR_N;
            sc_core::sc_time start = reserve_resource(
                mesh_router_out_finish_[plane][node_id(x, y)][out_dir], now, beats,
                stats_.router_wait_ns, stats_.router_service_ns);
            ready = max_time(ready, start);
            if (south) {
                start = reserve_resource(mesh_vlink_finish_[plane][y][x][0], now, beats,
                                         stats_.link_wait_ns, stats_.link_service_ns);
                ready = max_time(ready, start);
                ++y;
            } else {
                start = reserve_resource(mesh_vlink_finish_[plane][y - 1][x][1], now, beats,
                                         stats_.link_wait_ns, stats_.link_service_ns);
                ready = max_time(ready, start);
                --y;
            }
        }
        sc_core::sc_time start = reserve_resource(
            mesh_router_out_finish_[plane][node_id(x, y)][DIR_LOCAL], now, beats,
            stats_.local_wait_ns, stats_.local_service_ns);
        ready = max_time(ready, start);
    }

    // Mesh mode: 16 SRAM banks sit behind two parallel 4x4 mesh planes. The
    // original model charged
    // every 16B stripe for edge + router + link + local latency in series.
    // That was intentionally conservative, but it made small transformer-like
    // slices pay packet startup cost over and over.
    //
    // Current model is a transparent Payload NoC:
    //   * long blocking API calls are internally chopped into fixed simulator
    //     chunks so timing can reserve ingress/router/link/SRAM resources;
    //     this is not protocol burst information and is not carried in Payload;
    //   * bank SRAM ports remain the throughput limiter;
    //   * edge/router/link/local resources are reserved for contention stats;
    //   * resource service time is pipelined and only prior contention delays
    //     the caller. MeshOptimistic skips those NoC reservations entirely.
    //
    // This is intentionally still an architectural timing model, not a
    // cycle-accurate packet network: current read/write calls do not carry the
    // requester class (CONV ACT, CONV WGT, UDMA, EWE, ...), so priority is
    // approximated by independent read/write multi-edge ingress while all
    // internal mesh links are shared.
    sc_core::sc_time schedule_mesh_latency(uint32_t offset, uint32_t bytes,
                                           bool is_read) {
        const sc_core::sc_time now = sc_core::sc_time_stamp();
        const uint32_t end = offset + bytes;
        sc_core::sc_time chunk_now = now;

        stats_.accesses += 1;
        stats_.bytes += bytes;
        const bool optimistic = timing_mode_ == L1TimingMode::MeshOptimistic;

        for (uint32_t chunk_off = offset; chunk_off < end; ) {
            const uint32_t chunk_end = std::min<uint32_t>(
                end, chunk_off + PAYLOAD_SCHED_CHUNK_BYTES);
            std::array<uint32_t, N_BANKS> bank_bytes{};
            std::array<uint32_t, N_BANKS> bank_addr{};
            std::array<bool, N_BANKS> bank_seen{};
            uint64_t chunk_stripes = 0;
            for (uint32_t a = chunk_off; a < chunk_end; ) {
                const uint32_t stripe = a / BANK_STRIDE;
                const uint32_t bank = swizzled_bank(stripe);
                const uint32_t next_stripe = ((a / BANK_STRIDE) + 1) * BANK_STRIDE;
                const uint32_t beat_chunk = std::min(chunk_end, next_stripe) - a;
                bank_bytes[bank] += beat_chunk;
                if (!bank_seen[bank]) {
                    bank_seen[bank] = true;
                    bank_addr[bank] = a;
                }
                ++chunk_stripes;
                a += beat_chunk;
            }

            stats_.stripes += chunk_stripes;
            stats_.chunks += 1;

            const uint32_t chunk_bytes = chunk_end - chunk_off;
            const uint32_t aggregate_beats =
                ceil_div(chunk_bytes, N_BANKS * BYTES_PER_CYCLE);

            sc_core::sc_time ingress_ready = chunk_now;
            if (!optimistic) {
                // Four row ingress lanes per side and direction. Pick the
                // earliest perimeter lane for this scheduling chunk.
                sc_core::sc_time best_start;
                bool have_best = false;
                unsigned best_plane = 0;
                unsigned best_side = 0;
                unsigned best_row = 0;
                const unsigned rw = is_read ? 0 : 1;
                for (unsigned plane = 0; plane < MESH_PLANES; ++plane) {
                    for (unsigned side = 0; side < 2; ++side) {
                        for (unsigned row = 0; row < MESH_H; ++row) {
                            const sc_core::sc_time finish = mesh_edge_finish_[plane][rw][side][row];
                            const sc_core::sc_time start =
                                (finish > chunk_now) ? finish : chunk_now;
                            if (!have_best || start < best_start) {
                                best_start = start;
                                best_plane = plane;
                                best_side = side;
                                best_row = row;
                                have_best = true;
                            }
                        }
                    }
                }
                ingress_ready = reserve_resource(
                    mesh_edge_finish_[best_plane][rw][best_side][best_row], chunk_now,
                    aggregate_beats, stats_.edge_wait_ns, stats_.edge_service_ns);
            }

            sc_core::sc_time chunk_finish = chunk_now;
            for (uint32_t bank = 0; bank < N_BANKS; ++bank) {
                const uint32_t bank_chunk = bank_bytes[bank];
                if (!bank_chunk) continue;
                const uint32_t beats = ceil_div(bank_chunk, BYTES_PER_CYCLE);
                const uint32_t dst_x = bank % MESH_W;
                const uint32_t dst_y = bank / MESH_W;
                sc_core::sc_time ready = ingress_ready;

                if (!optimistic) {
                    const uint32_t west_dist = dst_x;
                    const uint32_t east_dist = (MESH_W - 1) - dst_x;
                    const uint32_t src_x = (west_dist <= east_dist) ? 0 : (MESH_W - 1);
                    const uint32_t src_y = dst_y;
                    const unsigned plane = (bank + stats_.chunks) % MESH_PLANES;
                    reserve_route(src_x, src_y, dst_x, dst_y, beats,
                                  chunk_now, plane, ready);
                }

                const double sram_service_before = stats_.sram_service_ns;
                probe_payload_input(ready, is_read, bank, bank_addr[bank]);
                sc_core::sc_time t = service_sram_beat(sram_bank_finish_[bank], ready, bank_chunk);
                const double lane_service = stats_.sram_service_ns - sram_service_before;
                Stats::Lane& lane = is_read ? stats_.read_lane[bank]
                                            : stats_.write_lane[bank];
                lane.accesses += 1;
                lane.bytes += bank_chunk;
                const double latency = ns(t - chunk_now);
                const double lane_wait = std::max(0.0, latency - lane_service);
                lane.latency_ns += latency;
                lane.wait_ns += lane_wait;
                lane.service_ns += lane_service;
                if (latency > lane.max_latency_ns) lane.max_latency_ns = latency;
                if (lane_wait > lane.max_wait_ns) lane.max_wait_ns = lane_wait;
                if (lane_service > lane.max_service_ns) lane.max_service_ns = lane_service;
                if (t > chunk_finish) chunk_finish = t;
            }
            chunk_now = chunk_finish;
            chunk_off = chunk_end;
        }
        stats_.imposed_wait_ns += ns(chunk_now - now);
        return chunk_now;
    }

    static constexpr unsigned MESH_PLANES = 2;
    static constexpr unsigned MESH_W = 4;
    static constexpr unsigned MESH_H = 4;
    static constexpr unsigned ROUTER_INPUT_FIFO_DEPTH = 2; // flits, architectural knob
    enum MeshDir : unsigned { DIR_N = 0, DIR_E = 1, DIR_S = 2,
                              DIR_W = 3, DIR_LOCAL = 4 };
    static constexpr unsigned node_id(unsigned x, unsigned y) {
        return y * MESH_W + x;
    }

    static const char* engine_id_from_process() {
        const auto h = sc_core::sc_get_current_process_handle();
        if (!h.valid()) return "host";
        const char* n = h.name();
        if (!n) return "unknown";
        if (std::strstr(n, "conv")) return "conv";
        if (std::strstr(n, "requant")) return "requant";
        if (std::strstr(n, "ewe")) return "ewe";
        if (std::strstr(n, "pool")) return "pool";
        if (std::strstr(n, "udma")) return "udma";
        if (std::strstr(n, "host")) return "host";
        return n;
    }

    static const char* ordinal(unsigned i) {
        static thread_local char buf[8];
        const unsigned n = i + 1;
        const unsigned mod100 = n % 100;
        const char* suffix = "th";
        if (mod100 < 11 || mod100 > 13) {
            switch (n % 10) {
            case 1: suffix = "st"; break;
            case 2: suffix = "nd"; break;
            case 3: suffix = "rd"; break;
            default: break;
            }
        }
        std::snprintf(buf, sizeof(buf), "%u%s", n, suffix);
        return buf;
    }

    void open_payload_probe(const char* path) {
        payload_probe_.open(path);
        if (!payload_probe_) {
            std::cerr << "warning: cannot open MDLA7_L1_PAYLOAD_PROBE path: "
                      << path << "\n";
            return;
        }
        payload_probe_rows_.clear();
        payload_probe_ << "cycle";
        for (unsigned i = 0; i < PAYLOAD_INPUTS; ++i) {
            payload_probe_ << ',' << ordinal(i) << " payload (engineid) addr";
        }
        payload_probe_ << '\n';
    }

    void flush_payload_probe_row() {
        if (!payload_probe_) return;
        for (const auto& [cycle, cells] : payload_probe_rows_) {
            payload_probe_ << cycle;
            for (const auto& cell : cells) {
                payload_probe_ << ',' << cell;
            }
            payload_probe_ << '\n';
        }
        payload_probe_.flush();
        payload_probe_rows_.clear();
    }

    static uint64_t cycle_of(sc_core::sc_time t) {
        return static_cast<uint64_t>(t.to_seconds() * 1.0e9 + 0.5);
    }

    void probe_payload_input(sc_core::sc_time now, bool is_read,
                             uint32_t bank, uint32_t addr) {
        if (!payload_probe_) return;
        const uint64_t cyc = cycle_of(now);
        const unsigned lane = (is_read ? 0u : N_BANKS) + (bank % N_BANKS);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "(%s) 0x%08x",
                      engine_id_from_process(), addr);
        auto& cells = payload_probe_rows_[cyc];
        if (!payload_probe_rows_initialized_.count(cyc)) {
            cells.fill(std::string{});
            payload_probe_rows_initialized_.insert(cyc);
        }
        if (!cells[lane].empty())
            cells[lane] += "|";
        cells[lane] += buf;
    }

    L1TimingMode timing_mode_;
    Stats stats_;
    std::vector<uint8_t> mem;
    sc_core::sc_time sram_bank_finish_[N_BANKS];
    sc_core::sc_time mesh_edge_finish_[MESH_PLANES][2][2][MESH_H];
    sc_core::sc_time mesh_router_out_finish_[MESH_PLANES][N_BANKS][5];
    sc_core::sc_time mesh_hlink_finish_[MESH_PLANES][MESH_H][MESH_W - 1][2];
    sc_core::sc_time mesh_vlink_finish_[MESH_PLANES][MESH_H - 1][MESH_W][2];
    static constexpr unsigned PAYLOAD_INPUTS =
        PayloadPortCount::L1MESH_R + PayloadPortCount::L1MESH_W;
    std::ofstream payload_probe_;
    std::map<uint64_t, std::array<std::string, PAYLOAD_INPUTS>> payload_probe_rows_;
    std::set<uint64_t> payload_probe_rows_initialized_;
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
    void read_compressed_instant(uint32_t addr, void* dst, uint32_t raw_n) {
        std::memcpy(dst, &mem[addr - DRAM_BASE], raw_n);
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
// ingress; CONV ACT_R/WGT_R reads bypass it through two dedicated direct
// L1Mesh Payload paths.
class L1Manager : public sc_core::sc_module {
public:
    using AccessTicket = L1Mesh::AccessTicket;

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
    void read_compressed_instant(uint32_t addr, void* dst, uint32_t raw_n) {
        if (addr_in_l1mesh(addr)) mesh_.read(addr, dst, raw_n);
        else if (addr_in_dram(addr)) dram_.read_compressed_instant(addr, dst, raw_n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    void write(uint32_t addr, const void* src, uint32_t n) {
        if (addr_in_l1mesh(addr)) mesh_.write(addr, src, n);
        else if (addr_in_dram(addr)) dram_.write(addr, src, n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    void write_instant(uint32_t addr, const void* src, uint32_t n) {
        if (addr_in_l1mesh(addr)) mesh_.write_instant(addr, src, n);
        else if (addr_in_dram(addr)) dram_.write(addr, src, n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    void write_compressed(uint32_t addr, const void* src, uint32_t raw_n,
                          uint32_t compressed_n) {
        if (addr_in_l1mesh(addr)) mesh_.write(addr, src, raw_n);
        else if (addr_in_dram(addr)) dram_.write_compressed(addr, src, raw_n, compressed_n);
        else SC_REPORT_ERROR("L1Manager", "addr out of range");
    }
    AccessTicket read_async(uint32_t addr, void* dst, uint32_t n) {
        if (addr_in_l1mesh(addr)) return mesh_.read_async(addr, dst, n);
        read(addr, dst, n);
        return AccessTicket{sc_core::sc_time_stamp()};
    }
    AccessTicket write_async(uint32_t addr, const void* src, uint32_t n) {
        if (addr_in_l1mesh(addr)) return mesh_.write_async(addr, src, n);
        write(addr, src, n);
        return AccessTicket{sc_core::sc_time_stamp()};
    }
    AccessTicket write_strided_rows(uint32_t dst, const void* src,
                                    uint32_t rows, uint32_t src_row,
                                    uint32_t dst_row, uint32_t dst_col) {
        if (addr_in_l1mesh(dst))
            return mesh_.write_strided_rows(dst, src, rows, src_row, dst_row, dst_col);
        SC_REPORT_ERROR("L1Manager", "write_strided_rows requires L1Mesh dst address");
        return AccessTicket{sc_core::sc_time_stamp()};
    }
    AccessTicket channel_pack(uint32_t src, uint32_t dst,
                              uint32_t rows, uint32_t src_row,
                              uint32_t src_stride, uint32_t dst_row,
                              uint32_t dst_col) {
        if (addr_in_l1mesh(src) && addr_in_l1mesh(dst))
            return mesh_.channel_pack(src, dst, rows, src_row, src_stride, dst_row, dst_col);
        SC_REPORT_ERROR("L1Manager", "channel_pack requires L1Mesh addresses");
        return AccessTicket{sc_core::sc_time_stamp()};
    }
    static void wait_ticket(const AccessTicket& t) { L1Mesh::wait_ticket(t); }
    static void wait_all(std::initializer_list<AccessTicket> tickets) {
        sc_core::sc_time done = sc_core::sc_time_stamp();
        for (const auto& t : tickets)
            if (t.done > done) done = t.done;
        wait_ticket(AccessTicket{done});
    }

private:
    L1Mesh& mesh_;
    Dram& dram_;
};

} // namespace mdla7
