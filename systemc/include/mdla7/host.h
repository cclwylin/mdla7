#pragma once

// Host (RISC-V) — v0 stub: pushes a hard-coded descriptor stream.
// v1+ will read from a JSON / TFLite-derived descriptor file or hook a RISC-V ISS.

#include <systemc>
#include <vector>
#include <iostream>
#include "mdla7/descriptor.h"

namespace mdla7 {

SC_MODULE(Host) {
    sc_core::sc_fifo_out<Descriptor> desc_out;

    std::vector<Descriptor> program;   // populated before sc_start

    SC_HAS_PROCESS(Host);
    Host(sc_core::sc_module_name nm) : sc_module(nm) { SC_THREAD(run); }

    void run() {
        std::cout << "[Host] uploading " << program.size()
                  << " descriptor(s) to ring buffer\n";
        for (auto& d : program) {
            desc_out.write(d);
            wait(1, sc_core::SC_NS);
        }
        std::cout << "[Host] all descriptors submitted; idle\n";
    }
};

} // namespace mdla7
