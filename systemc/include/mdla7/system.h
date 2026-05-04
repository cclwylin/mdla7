#pragma once

// MDLA7 top-level — instantiate everything, wire FIFOs.

#include <systemc>
#include <array>
#include <memory>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"
#include "mdla7/udma.h"
#include "mdla7/conv_engine.h"
#include "mdla7/requant_engine.h"
#include "mdla7/ewe_pool.h"
#include "mdla7/command_engine.h"
#include "mdla7/host.h"

namespace mdla7 {

class Mdla7System : public sc_core::sc_module {
public:
    // Channels (must be constructed before submodules so we can bind in init list).
    sc_core::sc_fifo<Descriptor>     desc_stream{"desc_stream", 256};
    sc_core::sc_fifo<DescriptorBody> conv_cfg{"conv_cfg", 4};
    sc_core::sc_fifo<DescriptorBody> requant_cfg{"requant_cfg", 4};
    sc_core::sc_fifo<DescriptorBody> ewe_cfg{"ewe_cfg", 4};
    sc_core::sc_fifo<DescriptorBody> pool_cfg{"pool_cfg", 4};
    sc_core::sc_fifo<DescriptorBody> udma_cfg{"udma_cfg", 4};

    // CONV → Requant chain: 16 lanes of INT32 partial sums (§3A.5).
    std::array<std::unique_ptr<sc_core::sc_fifo<int32_t>>, 16> chain;

    sc_core::sc_fifo<uint8_t> conv_done{"conv_done", 4};
    sc_core::sc_fifo<uint8_t> requant_done{"requant_done", 4};
    sc_core::sc_fifo<uint8_t> ewe_done{"ewe_done", 4};
    sc_core::sc_fifo<uint8_t> pool_done{"pool_done", 4};
    sc_core::sc_fifo<uint8_t> udma_done{"udma_done", 4};

    // Modules.
    L1Mesh        l1mesh;
    Dram          dram;
    L1Manager     l1mgr;
    Udma          udma;
    ConvEngine    conv;
    RequantEngine requant;
    EweEngine     ewe;
    PoolEngine    pool;
    CommandEngine cmd;
    Host          host;

    SC_HAS_PROCESS(Mdla7System);
    // v8.22: dram_bytes parameterised so test_model can size the DRAM model
    // to fit the program (deeplab_v3_plus + similar large segmentation models
    // need >256 MB; default 256 MB segfaults on `sys.dram.write` out-of-bounds).
    Mdla7System(sc_core::sc_module_name nm,
                std::size_t dram_bytes = 256 * 1024 * 1024)
      : sc_module(nm),
        l1mesh ("l1mesh"),
        dram   ("dram", dram_bytes),
        l1mgr  ("l1mgr",    l1mesh, dram),
        udma   ("udma",     l1mgr),
        conv   ("conv",     l1mgr),
        requant("requant",  l1mgr),
        ewe    ("ewe",      l1mgr),
        pool   ("pool",     l1mgr),
        cmd    ("cmd"),
        host   ("host")
    {
        for (int i = 0; i < 16; ++i) {
            chain[i] = std::make_unique<sc_core::sc_fifo<int32_t>>(
                ("chain_" + std::to_string(i)).c_str(), 2);
            conv.chain_out[i]    = chain[i].get();
            requant.chain_in[i]  = chain[i].get();
        }

        host.desc_out(desc_stream);
        cmd.desc_in (desc_stream);

        cmd.conv_cfg_out(conv_cfg);     conv.cfg_in(conv_cfg);
        cmd.requant_cfg_out(requant_cfg); requant.cfg_in(requant_cfg);
        cmd.ewe_cfg_out(ewe_cfg);       ewe.cfg_in(ewe_cfg);
        cmd.pool_cfg_out(pool_cfg);     pool.cfg_in(pool_cfg);
        cmd.udma_cfg_out(udma_cfg);     udma.cfg_in(udma_cfg);

        conv.done_tag_out(conv_done);       cmd.conv_done(conv_done);
        requant.done_tag_out(requant_done); cmd.requant_done(requant_done);
        ewe.done_tag_out(ewe_done);         cmd.ewe_done(ewe_done);
        pool.done_tag_out(pool_done);       cmd.pool_done(pool_done);
        udma.done_tag_out(udma_done);       cmd.udma_done(udma_done);

        cmd.conv_dtype_latch = &conv   .last_dtype;
        cmd.req_dtype_latch  = &requant.last_dtype;
        cmd.ewe_dtype_latch  = &ewe    .last_dtype;
        cmd.pool_dtype_latch = &pool   .last_dtype;
    }
};

} // namespace mdla7
