#pragma once

// Command Engine — descriptor decode + dependency tracker → 5 engine config FIFO
// (see spec §3A.8 / §3A.10)
// v0: in-order dispatch with optional wait_tags AND-check; signal_tag set on done.

#include <systemc>
#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"

namespace mdla7 {

SC_MODULE(CommandEngine) {
    // From host (DRAM ring buffer, modeled as sc_fifo for v0):
    sc_core::sc_fifo_in<Descriptor> desc_in;

    // Per-engine config FIFOs (1T payload, depth 4):
    sc_core::sc_fifo_out<DescriptorBody> conv_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> requant_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> ewe_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> pool_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> tnps_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> udma_cfg_out;

    // done_tag from each engine:
    sc_core::sc_fifo_in<uint8_t> conv_done, requant_done, ewe_done, pool_done, tnps_done, udma_done;

    // Side-channel for v0: latch dtype into CONV/Requant before dispatch.
    // v8.17: same wiring extended to EWE/POOL so the FP path picks up FP16
    // storage + FP32 compute when the dispatched descriptor's header dtype
    // is one of FP16/BFP16/FP8.
    uint8_t* conv_dtype_latch = nullptr;
    uint8_t* req_dtype_latch  = nullptr;
    uint8_t* ewe_dtype_latch  = nullptr;
    uint8_t* pool_dtype_latch = nullptr;

    struct TaskMeta {
        uint16_t layer_id = 0;
        uint16_t microblock_id = 0;
        uint8_t stream_slot = 0;
        uint8_t stream_meta_flags = 0;
        uint8_t flags = 0;
        uint8_t op_class = 0;
        uint8_t op_subtype = 0;
        uint8_t udma_direction = 0;
    };
    std::vector<TaskMeta> trace_udma_r, trace_udma_w;
    std::vector<TaskMeta> trace_conv, trace_requant, trace_ewe, trace_pool, trace_tnps;
    sc_core::sc_time busy_time{sc_core::SC_ZERO_TIME};
    std::vector<std::pair<uint64_t, uint64_t>> tasks;
    std::vector<std::vector<RtlPhaseTrace>> rtl_phase_tasks;
    EngineModel engine_model = EngineModel::Analytical;

    SC_HAS_PROCESS(CommandEngine);
    CommandEngine(sc_core::sc_module_name nm) : sc_module(nm) {
        SC_THREAD(dispatch);
        SC_THREAD(collect);
    }

    void dispatch() {
        std::deque<Descriptor> pending;
        constexpr size_t LOOKAHEAD_LIMIT = 16;   // stay well below 8-bit tag wrap distance
        while (true) {
            while (desc_in.num_available() > 0 && pending.size() < LOOKAHEAD_LIMIT) {
                Descriptor d = desc_in.read();
                pending.push_back(d);
            }

            auto best = pending.end();
            int best_prio = std::numeric_limits<int>::max();
            bool tail_waiting = false;
            for (auto it = pending.begin(); it != pending.end(); ++it) {
                // Only descriptors explicitly marked as stream-pipeline work
                // may be bypassed. Normal descriptors keep in-order issue
                // because many schedules reuse fixed L1 regions.
                if (!(it->hdr.flags & DF_STREAM)) {
                    if (it == pending.begin() && waits_ready(pending.begin(), it)) {
                        best = it;
                    }
                    break;
                }
                if (!waits_ready(pending.begin(), it)) {
                    if (stream_tail_priority(*it))
                        tail_waiting = true;
                    continue;
                }
                if (tail_waiting && !allowed_during_tail_wait(*it))
                    continue;
                const int prio = stream_issue_priority(*it);
                if (prio < best_prio) {
                    best = it;
                    best_prio = prio;
                    if (prio == 0) break;
                }
            }

            bool issued = false;
            if (best != pending.end()) {
                issue(*best);
                pending.erase(best);
                issued = true;
            }
            if (issued) continue;

            if (pending.empty())
                wait(desc_in.data_written_event());
            else {
                wait(desc_in.data_written_event() | tag_changed);
            }
        }
    }

    bool waits_ready(std::deque<Descriptor>::const_iterator begin,
                     std::deque<Descriptor>::const_iterator self) const {
        const Descriptor& d = *self;
        for (int w = 0; w < d.hdr.wait_count; ++w) {
            uint8_t tg = d.hdr.wait_tags[w];
            if (tag_pending[tg] != 0) return false;
            for (auto it = begin; it != self; ++it) {
                if (it->hdr.signal_tag == tg)
                    return false;
            }
        }
        return true;
    }

    int stream_issue_priority(const Descriptor& d) const {
        if (stream_tail_priority(d)) return 0;
        int base = 50;
        switch (d.hdr.op_class()) {
        case OC_EWE:
            // Launch compute as soon as its microblock is ready. Later UDMA_R
            // work can then issue while this engine is busy.
            base = 10; break;
        case OC_UDMA:
            // DRAM->L1 loads feed compute; stores drain in the background.
            base = (d.body.udma.direction == 0) ? 20 : 60;
            break;
        case OC_CONV:    base = 30; break;
        case OC_REQUANT: base = 40; break;
        case OC_POOL:    base = 40; break;
        case OC_TNPS:
            // Layout consumers often wait on a producer microblock prefix. Let
            // ready TNPS work launch before filling more same-engine compute
            // FIFO entries, so it can overlap with the producer's tail.
            base = 8; break;
        default:         base = 70; break;
        }
        // Microblock wavefront tie-breaker: keep work roughly in tile order
        // without letting an older store block younger loads/compute.
        return base * 4096 + int(d.hdr.microblock_id);
    }

    bool stream_tail_priority(const Descriptor& d) const {
        return (d.hdr.flags & DF_STREAM_TAIL) != 0;
    }

    bool allowed_during_tail_wait(const Descriptor& d) const {
        // Let later slots fetch their data while an older tile waits for its
        // D2S/EWE tail. Also allow a shallow front of the next slot's
        // CONV/REQUANT chain, leaving the deeper CONV work to overlap with
        // EWE once the tail becomes ready.
        if (d.hdr.op_class() == OC_UDMA && d.body.udma.direction == 0)
            return true;
        if ((d.hdr.op_class() == OC_CONV || d.hdr.op_class() == OC_REQUANT)
            && (d.hdr.stream_meta_flags & SMF_COMPUTE))
            return true;
        return false;
    }

    void issue(const Descriptor& d) {
        const sc_core::sc_time t_begin = sc_core::sc_time_stamp();
        std::cout << "[CmdEng] dispatch op_class=" << int(d.hdr.op_class())
                  << " layer_id=" << d.hdr.layer_id
                  << " signal_tag=" << int(d.hdr.signal_tag)
                  << " wait_count=" << int(d.hdr.wait_count);
        if (d.hdr.flags & DF_STREAM) {
            std::cout << " slot=" << int(d.hdr.stream_slot)
                      << " mb=" << d.hdr.microblock_id
                      << " smeta=0x" << std::hex << int(d.hdr.stream_meta_flags) << std::dec;
        }
        std::cout << "\n";

        // Reserve signal tags at issue time. waits_ready() also checks older
        // pending descriptors with the same signal tag, so stream consumers
        // cannot observe stale done state, while future wrapped tags do not
        // poison already-issued work.
        if (d.hdr.signal_tag)
            ++tag_pending[d.hdr.signal_tag];
        // queue this task's signal_tag for the engine (FIFO pairing).
        pending_tags[d.hdr.op_class()].push(d.hdr.signal_tag);

        switch (d.hdr.op_class()) {
        case OC_CONV:
            record_trace(d);
            if (conv_dtype_latch) *conv_dtype_latch = d.hdr.dtype;
            conv_cfg_out.write(d.body); break;
        case OC_REQUANT:
            record_trace(d);
            if (req_dtype_latch) *req_dtype_latch = d.hdr.dtype;
            requant_cfg_out.write(d.body); break;
        case OC_EWE:
            record_trace(d);
            if (ewe_dtype_latch) *ewe_dtype_latch = d.hdr.dtype;
            ewe_cfg_out.write(d.body); break;
        case OC_POOL:
            record_trace(d);
            if (pool_dtype_latch) *pool_dtype_latch = d.hdr.dtype;
            pool_cfg_out.write(d.body); break;
        case OC_TNPS:
            record_trace(d);
            tnps_cfg_out.write(d.body); break;
        case OC_UDMA:
            record_trace(d);
            udma_cfg_out.write(d.body); break;
        default:
            std::cout << "[CmdEng] unknown op_class\n"; break;
        }
        const sc_core::sc_time t_end = sc_core::sc_time_stamp();
        const uint64_t dispatch_cycles = uint64_t((t_end - t_begin).to_seconds() * 1e9);
        const uint64_t display_cycles = is_rtl_style(engine_model)
                                      ? std::max<uint64_t>(1, dispatch_cycles)
                                      : dispatch_cycles;
        const uint64_t t_begin_ns = uint64_t(t_begin.to_seconds() * 1e9);
        busy_time += t_end - t_begin;
        tasks.emplace_back(t_begin_ns, t_begin_ns + display_cycles);
        if (is_rtl_style(engine_model)) {
            RtlPhaseTrace phase;
            phase.name = "dispatch";
            phase.cycles = display_cycles;
            phase.elems = d.hdr.wait_count;
            phase.lanes = d.hdr.op_class();
            phase.stall = "cfg_fifo_write";
            rtl_phase_tasks.push_back({phase});
        } else {
            rtl_phase_tasks.push_back({});
        }
    }

    void record_trace(const Descriptor& d) {
        TaskMeta m{};
        m.layer_id = d.hdr.layer_id;
        m.microblock_id = d.hdr.microblock_id;
        m.stream_slot = d.hdr.stream_slot;
        m.stream_meta_flags = d.hdr.stream_meta_flags;
        m.flags = d.hdr.flags;
        m.op_class = uint8_t(d.hdr.op_class());
        m.op_subtype = d.hdr.op_subtype();
        m.udma_direction = (d.hdr.op_class() == OC_UDMA) ? d.body.udma.direction : 0;
        switch (d.hdr.op_class()) {
        case OC_UDMA:
            if (m.udma_direction == 1) trace_udma_w.push_back(m);
            else                       trace_udma_r.push_back(m);
            break;
        case OC_CONV:    trace_conv.push_back(m); break;
        case OC_REQUANT: trace_requant.push_back(m); break;
        case OC_EWE:     trace_ewe.push_back(m); break;
        case OC_POOL:    trace_pool.push_back(m); break;
        case OC_TNPS:    trace_tnps.push_back(m); break;
        default: break;
        }
    }

    void collect() {
        // Wait on any done fifo's data_written_event so sim idles when all
        // engines are quiet — keeps sc_time_stamp meaningful instead of
        // running to the sc_start budget.
        while (true) {
            wait(conv_done.data_written_event()
               | requant_done.data_written_event()
               | ewe_done.data_written_event()
               | pool_done.data_written_event()
               | tnps_done.data_written_event()
               | udma_done.data_written_event());
            check(conv_done,    OC_CONV);
            check(requant_done, OC_REQUANT);
            check(ewe_done,     OC_EWE);
            check(pool_done,    OC_POOL);
            check(tnps_done,    OC_TNPS);
            check(udma_done,    OC_UDMA);
        }
    }

    void check(sc_core::sc_fifo_in<uint8_t>& f, OpClass cls) {
        while (f.num_available() > 0) {
            (void)f.read();   // payload from engine — currently always 0
            uint8_t t = 0;
            if (!pending_tags[cls].empty()) {
                t = pending_tags[cls].front();
                pending_tags[cls].pop();
            }
            if (t) {
                if (tag_pending[t] > 0)
                    --tag_pending[t];
                tag_changed.notify();
                     last_activity = sc_core::sc_time_stamp();
                     tag_fire_time[t] = last_activity;
                     std::cout << "[CmdEng] engine " << int(cls)
                               << " done; tag " << int(t) << " set\n"; }
            else {  std::cout << "[CmdEng] engine " << int(cls)
                              << " done (no signal_tag)\n"; }
        }
    }

    uint16_t tag_pending[256]{};
    std::queue<uint8_t> pending_tags[OC_NUM];
    sc_core::sc_event tag_changed;
    // Time of the most recent tag completion. Used by test harness to report
    // "real" sim time even when sc_start ran for its full budget.
    sc_core::sc_time last_activity{sc_core::SC_ZERO_TIME};
    // Wall sim time when each tag fired — used for per-layer cycle reporting.
    sc_core::sc_time tag_fire_time[256] = {};
};

} // namespace mdla7
