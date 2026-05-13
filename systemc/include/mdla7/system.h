#pragma once

// MDLA7 top-level — instantiate everything, wire FIFOs.

#include <systemc>
#include <array>
#include <memory>
#include <string>
#include "mdla7/descriptor.h"
#include "mdla7/memory.h"
#include "mdla7/udma.h"
#include "mdla7/tnps.h"
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
    sc_core::sc_fifo<DescriptorBody> tnps_cfg{"tnps_cfg", 4};
    sc_core::sc_fifo<DescriptorBody> udma_cfg{"udma_cfg", 4};

    // CONV → Requant chain: 128 lanes of INT32 partial sums = 4096 bit/cyc (§3A.5).
    std::array<std::unique_ptr<sc_core::sc_fifo<int32_t>>,
               CONV_REQUANT_CHAIN_LANES> chain;

    // Architectural Payload lane scaffold. The current functional engines
    // still call L1Manager::read/write; these channels pin down the agreed
    // Engine↔L1 and the two dedicated CONV↔L1Mesh ACT_R/WGT_R port counts
    // for the beat-level model.
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::REQUANT_R> requant_payload_r;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::REQUANT_W> requant_payload_w;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::EWE_R> ewe_payload_r;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::EWE_W> ewe_payload_w;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::POOL_R> pool_payload_r;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::POOL_W> pool_payload_w;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::TNPS_R> tnps_payload_r;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::TNPS_W> tnps_payload_w;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::L1MESH_R> l1mgr_l1mesh_payload_r;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::L1MESH_W> l1mgr_l1mesh_payload_w;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::CONV_ACT_R> conv_act_payload_r;
    std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, PayloadPortCount::CONV_WGT_R> conv_wgt_payload_r;

    sc_core::sc_fifo<uint8_t> conv_done{"conv_done", 4};
    sc_core::sc_fifo<uint8_t> requant_done{"requant_done", 4};
    sc_core::sc_fifo<uint8_t> ewe_done{"ewe_done", 4};
    sc_core::sc_fifo<uint8_t> pool_done{"pool_done", 4};
    sc_core::sc_fifo<uint8_t> tnps_done{"tnps_done", 4};
    sc_core::sc_fifo<uint8_t> udma_done{"udma_done", 4};

    // Modules.
    L1Mesh        l1mesh;
    Dram          dram;
    L1Manager     l1mgr;
    Udma          udma;
    TnpsEngine    tnps;
    ConvEngine    conv;
    RequantEngine requant;
    EweEngine     ewe;
    PoolEngine    pool;
    CommandEngine cmd;
    Host          host;

    SC_HAS_PROCESS(Mdla7System);
    // v8.22: dram_bytes parameterised so the model runner can size the DRAM model
    // to fit the program (deeplab_v3_plus + similar large segmentation models
    // need >256 MB; default 256 MB segfaults on `sys.dram.write` out-of-bounds).
    Mdla7System(sc_core::sc_module_name nm,
                std::size_t dram_bytes = 256 * 1024 * 1024,
                L1TimingMode l1_timing_mode = L1TimingMode::Rtl,
                EngineModel engine_model = EngineModel::Rtl)
      : sc_module(nm),
        l1mesh ("l1mesh", L1MESH_BYTES, l1_timing_mode),
        dram   ("dram", dram_bytes),
        l1mgr  ("l1mgr",    l1mesh, dram),
        udma   ("udma",     l1mgr),
        tnps   ("tnps",     l1mgr),
        conv   ("conv",     l1mgr),
        requant("requant",  l1mgr),
        ewe    ("ewe",      l1mgr),
        pool   ("pool",     l1mgr),
        cmd    ("cmd"),
        host   ("host")
    {
        init_payload_fifos(requant_payload_r, "requant_payload_r");
        init_payload_fifos(requant_payload_w, "requant_payload_w");
        init_payload_fifos(ewe_payload_r, "ewe_payload_r");
        init_payload_fifos(ewe_payload_w, "ewe_payload_w");
        init_payload_fifos(pool_payload_r, "pool_payload_r");
        init_payload_fifos(pool_payload_w, "pool_payload_w");
        init_payload_fifos(tnps_payload_r, "tnps_payload_r");
        init_payload_fifos(tnps_payload_w, "tnps_payload_w");
        init_payload_fifos(l1mgr_l1mesh_payload_r, "l1mgr_l1mesh_payload_r");
        init_payload_fifos(l1mgr_l1mesh_payload_w, "l1mgr_l1mesh_payload_w");
        init_payload_fifos(conv_act_payload_r, "conv_act_payload_r");
        init_payload_fifos(conv_wgt_payload_r, "conv_wgt_payload_r");

        for (std::size_t i = 0; i < CONV_REQUANT_CHAIN_LANES; ++i) {
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
        cmd.tnps_cfg_out(tnps_cfg);     tnps.cfg_in(tnps_cfg);
        cmd.udma_cfg_out(udma_cfg);     udma.cfg_in(udma_cfg);

        conv.done_tag_out(conv_done);       cmd.conv_done(conv_done);
        requant.done_tag_out(requant_done); cmd.requant_done(requant_done);
        ewe.done_tag_out(ewe_done);         cmd.ewe_done(ewe_done);
        pool.done_tag_out(pool_done);       cmd.pool_done(pool_done);
        tnps.done_tag_out(tnps_done);       cmd.tnps_done(tnps_done);
        udma.done_tag_out(udma_done);       cmd.udma_done(udma_done);

        cmd.conv_dtype_latch = &conv   .last_dtype;
        cmd.req_dtype_latch  = &requant.last_dtype;
        cmd.ewe_dtype_latch  = &ewe    .last_dtype;
        cmd.pool_dtype_latch = &pool   .last_dtype;

        l1mesh.engine_model = engine_model;
        l1mgr.engine_model = engine_model;
        udma.engine_model = engine_model;
        conv.engine_model = engine_model;
        requant.engine_model = engine_model;
        ewe.engine_model = engine_model;
        pool.engine_model = engine_model;
        tnps.engine_model = engine_model;
        cmd.engine_model = engine_model;
    }

private:
    template <std::size_t N>
    static void init_payload_fifos(
        std::array<std::unique_ptr<sc_core::sc_fifo<Payload>>, N>& fifos,
        const char* prefix) {
        for (std::size_t i = 0; i < N; ++i) {
            fifos[i] = std::make_unique<sc_core::sc_fifo<Payload>>(
                (std::string(prefix) + "_" + std::to_string(i)).c_str(), 2);
        }
    }
};

} // namespace mdla7
