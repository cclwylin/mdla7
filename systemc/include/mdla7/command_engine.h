#pragma once

// Command Engine — descriptor decode + dependency tracker → 5 engine config FIFO
// (see spec §3A.8 / §3A.10)
// v0: in-order dispatch with optional wait_tags AND-check; signal_tag set on done.

#include <systemc>
#include <array>
#include <iostream>
#include <queue>
#include "mdla7/descriptor.h"

namespace mdla7 {

SC_MODULE(CommandEngine) {
    // From host (DRAM ring buffer, modeled as sc_fifo for v0):
    sc_core::sc_fifo_in<Descriptor> desc_in;

    // Per-engine config FIFOs (1T payload, depth 4):
    sc_core::sc_fifo_out<DescriptorBody> conv_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> requant_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> ewe_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> pool_cfg_out;
    sc_core::sc_fifo_out<DescriptorBody> udma_cfg_out;

    // done_tag from each engine:
    sc_core::sc_fifo_in<uint8_t> conv_done, requant_done, ewe_done, pool_done, udma_done;

    // Side-channel for v0: latch dtype into CONV/Requant before dispatch.
    // v8.17: same wiring extended to EWE/POOL so the FP path picks up FP16
    // storage + FP32 compute when the dispatched descriptor's header dtype
    // is one of FP16/BFP16/FP8.
    uint8_t* conv_dtype_latch = nullptr;
    uint8_t* req_dtype_latch  = nullptr;
    uint8_t* ewe_dtype_latch  = nullptr;
    uint8_t* pool_dtype_latch = nullptr;

    SC_HAS_PROCESS(CommandEngine);
    CommandEngine(sc_core::sc_module_name nm) : sc_module(nm) {
        for (int i = 0; i < 256; ++i) tag_done[i] = true;   // start state
        SC_THREAD(dispatch);
        SC_THREAD(collect);
    }

    void dispatch() {
        while (true) {
            Descriptor d = desc_in.read();
            // wait on tags (AND-of-all). Use a notify event so we don't
            // burn cycles polling — keeps sc_time_stamp accurate.
            for (int w = 0; w < d.hdr.wait_count; ++w) {
                uint8_t tg = d.hdr.wait_tags[w];
                while (!tag_done[tg]) wait(tag_changed);
            }

            std::cout << "[CmdEng] dispatch op_class=" << int(d.hdr.op_class())
                      << " layer_id=" << d.hdr.layer_id
                      << " signal_tag=" << int(d.hdr.signal_tag)
                      << " wait_count=" << int(d.hdr.wait_count) << "\n";

            // mark signal_tag pending until done report comes back
            if (d.hdr.signal_tag) {
                tag_done[d.hdr.signal_tag] = false;
            }
            // queue this task's signal_tag for the engine (FIFO pairing).
            pending_tags[d.hdr.op_class()].push(d.hdr.signal_tag);

            switch (d.hdr.op_class()) {
            case OC_CONV:
                if (conv_dtype_latch) *conv_dtype_latch = d.hdr.dtype;
                conv_cfg_out.write(d.body); break;
            case OC_REQUANT:
                if (req_dtype_latch) *req_dtype_latch = d.hdr.dtype;
                requant_cfg_out.write(d.body); break;
            case OC_EWE:
                if (ewe_dtype_latch) *ewe_dtype_latch = d.hdr.dtype;
                ewe_cfg_out.write(d.body); break;
            case OC_POOL:
                if (pool_dtype_latch) *pool_dtype_latch = d.hdr.dtype;
                pool_cfg_out.write(d.body); break;
            case OC_UDMA:   udma_cfg_out.write(d.body); break;
            default:
                std::cout << "[CmdEng] unknown op_class\n"; break;
            }
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
               | udma_done.data_written_event());
            check(conv_done,    OC_CONV);
            check(requant_done, OC_REQUANT);
            check(ewe_done,     OC_EWE);
            check(pool_done,    OC_POOL);
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
            if (t) { tag_done[t] = true; tag_changed.notify();
                     last_activity = sc_core::sc_time_stamp();
                     tag_fire_time[t] = last_activity;
                     std::cout << "[CmdEng] engine " << int(cls)
                               << " done; tag " << int(t) << " set\n"; }
            else {  std::cout << "[CmdEng] engine " << int(cls)
                              << " done (no signal_tag)\n"; }
        }
    }

    bool tag_done[256];
    std::queue<uint8_t> pending_tags[OC_NUM];
    sc_core::sc_event tag_changed;
    // Time of the most recent tag completion. Used by test harness to report
    // "real" sim time even when sc_start ran for its full budget.
    sc_core::sc_time last_activity{sc_core::SC_ZERO_TIME};
    // Wall sim time when each tag fired — used for per-layer cycle reporting.
    sc_core::sc_time tag_fire_time[256] = {};
};

} // namespace mdla7
