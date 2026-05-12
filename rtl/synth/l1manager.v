`timescale 1ns/1ps

`ifndef MDLA7_L1MANAGER_V
`define MDLA7_L1MANAGER_V
`include "common.v"

`timescale 1ns/1ps

module l1manager #(
    parameter L1_BYTES_PER_CYCLE = 256,
    parameter DRAM_BYTES_PER_CYCLE = 48,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input             clk,
    input             rst_n,

    input             req_valid,
    output            req_ready,
    input             req_write,
    input             req_l1,
    input      [3:0]  req_source,
    input      [7:0]  req_tid,
    input      [31:0] req_bytes,
    input      [31:0] req_payload_cycles,
    input      [ADDR_WIDTH-1:0] req_addr,
    input      [DATA_WIDTH-1:0] req_wdata,
    input      [DATA_WIDTH/8-1:0] req_wstrb,

    input             udma_req_valid,
    output            udma_req_ready,
    input             udma_req_write,
    input      [7:0]  udma_req_tid,
    input      [31:0] udma_req_bytes,
    input      [31:0] udma_req_payload_cycles,
    input      [ADDR_WIDTH-1:0] udma_req_addr,
    input      [DATA_WIDTH-1:0] udma_req_wdata,
    input      [DATA_WIDTH/8-1:0] udma_req_wstrb,

    input             requant_req_valid,
    output            requant_req_ready,
    input             requant_req_write,
    input      [7:0]  requant_req_tid,
    input      [31:0] requant_req_bytes,
    input      [31:0] requant_req_payload_cycles,
    input      [ADDR_WIDTH-1:0] requant_req_addr,
    input      [DATA_WIDTH-1:0] requant_req_wdata,
    input      [DATA_WIDTH/8-1:0] requant_req_wstrb,

    input             ewe_req_valid,
    output            ewe_req_ready,
    input             ewe_req_write,
    input      [7:0]  ewe_req_tid,
    input      [31:0] ewe_req_bytes,
    input      [31:0] ewe_req_payload_cycles,
    input      [ADDR_WIDTH-1:0] ewe_req_addr,
    input      [DATA_WIDTH-1:0] ewe_req_wdata,
    input      [DATA_WIDTH/8-1:0] ewe_req_wstrb,

    input             pool_req_valid,
    output            pool_req_ready,
    input             pool_req_write,
    input      [7:0]  pool_req_tid,
    input      [31:0] pool_req_bytes,
    input      [31:0] pool_req_payload_cycles,
    input      [ADDR_WIDTH-1:0] pool_req_addr,
    input      [DATA_WIDTH-1:0] pool_req_wdata,
    input      [DATA_WIDTH/8-1:0] pool_req_wstrb,

    input             tnps_req_valid,
    output            tnps_req_ready,
    input             tnps_req_write,
    input      [7:0]  tnps_req_tid,
    input      [31:0] tnps_req_bytes,
    input      [31:0] tnps_req_payload_cycles,
    input      [ADDR_WIDTH-1:0] tnps_req_addr,
    input      [DATA_WIDTH-1:0] tnps_req_wdata,
    input      [DATA_WIDTH/8-1:0] tnps_req_wstrb,

    output reg                  mesh_req_write,
    output reg [ADDR_WIDTH-1:0] mesh_req_addr,
    output reg [31:0]           mesh_req_bytes,
    output reg [DATA_WIDTH-1:0] mesh_req_wdata,
    output reg [DATA_WIDTH/8-1:0] mesh_req_wstrb,
    output reg [3:0]            mesh_req_source,
    output reg [7:0]            mesh_req_tid,

    output            resp_valid,
    input             resp_ready,
    output            busy,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles,
    output     [3:0]  debug_source,
    output     [7:0]  debug_tid
);
    localparam [3:0] PH_REQ_FETCH       = 4'd1;
    localparam [3:0] PH_ARB             = 4'd2;
    localparam [3:0] PH_L1_PAYLOAD      = 4'd3;
    localparam [3:0] PH_DRAM_READ_DATA  = 4'd4;
    localparam [3:0] PH_DRAM_WRITE_DATA = 4'd5;
    localparam [3:0] PH_RESP            = 4'd6;

    localparam [2:0] SRC_LEGACY  = 3'd0;
    localparam [2:0] SRC_UDMA    = 3'd1;
    localparam [2:0] SRC_REQUANT = 3'd2;
    localparam [2:0] SRC_EWE     = 3'd3;
    localparam [2:0] SRC_POOL    = 3'd4;
    localparam [2:0] SRC_TNPS    = 3'd5;
    localparam [2:0] SRC_NONE    = 3'd7;

    localparam [3:0] DBG_LEGACY  = 4'd0;
    localparam [3:0] DBG_UDMA    = 4'd6;
    localparam [3:0] DBG_REQUANT = 4'd2;
    localparam [3:0] DBG_EWE     = 4'd3;
    localparam [3:0] DBG_POOL    = 4'd4;
    localparam [3:0] DBG_TNPS    = 4'd5;

    reg [5:0] valid0;
    reg [5:0] valid1;
    reg       q0_write [0:5];
    reg       q1_write [0:5];
    reg       q0_l1 [0:5];
    reg       q1_l1 [0:5];
    reg [3:0] q0_source [0:5];
    reg [3:0] q1_source [0:5];
    reg [7:0] q0_tid [0:5];
    reg [7:0] q1_tid [0:5];
    reg [31:0] q0_bytes [0:5];
    reg [31:0] q1_bytes [0:5];
    reg [31:0] q0_payload_cycles [0:5];
    reg [31:0] q1_payload_cycles [0:5];
    reg [ADDR_WIDTH-1:0] q0_addr [0:5];
    reg [ADDR_WIDTH-1:0] q1_addr [0:5];
    reg [DATA_WIDTH-1:0] q0_wdata [0:5];
    reg [DATA_WIDTH-1:0] q1_wdata [0:5];
    reg [DATA_WIDTH/8-1:0] q0_wstrb [0:5];
    reg [DATA_WIDTH/8-1:0] q1_wstrb [0:5];

    reg [2:0] arb_src;
    reg arb_valid;
    reg arb_write;
    reg arb_l1;
    reg [3:0] arb_source;
    reg [7:0] arb_tid;
    reg [31:0] arb_bytes;
    reg [31:0] arb_payload_cycles;
    reg [ADDR_WIDTH-1:0] arb_addr;
    reg [DATA_WIDTH-1:0] arb_wdata;
    reg [DATA_WIDTH/8-1:0] arb_wstrb;
    wire phase_busy;

    wire phase_start_ready;
    wire phase_start_fire = arb_valid && phase_start_ready;
    wire legacy_push = req_valid && req_ready;
    wire udma_push = udma_req_valid && udma_req_ready;
    wire requant_push = requant_req_valid && requant_req_ready;
    wire ewe_push = ewe_req_valid && ewe_req_ready;
    wire pool_push = pool_req_valid && pool_req_ready;
    wire tnps_push = tnps_req_valid && tnps_req_ready;

    assign req_ready = !valid1[SRC_LEGACY];
    assign udma_req_ready = !valid1[SRC_UDMA];
    assign requant_req_ready = !valid1[SRC_REQUANT];
    assign ewe_req_ready = !valid1[SRC_EWE];
    assign pool_req_ready = !valid1[SRC_POOL];
    assign tnps_req_ready = !valid1[SRC_TNPS];
    assign debug_source = arb_source;
    assign debug_tid = arb_tid;
    assign busy = phase_busy || arb_valid || (valid1 != 6'd0) || resp_valid;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    function [31:0] max1;
        input [31:0] value;
        begin
            max1 = (value == 32'd0) ? 32'd1 : value;
        end
    endfunction

    task enqueue0;
        input [2:0] src;
        input write;
        input l1;
        input [3:0] source;
        input [7:0] tid;
        input [31:0] nbytes;
        input [31:0] payload_cycles;
        input [ADDR_WIDTH-1:0] addr;
        input [DATA_WIDTH-1:0] wdata;
        input [DATA_WIDTH/8-1:0] wstrb;
        begin
            if (phase_start_fire && (arb_src == src)) begin
                if (valid1[src]) begin
                    valid1[src] <= 1'b1;
                    q1_write[src] <= write;
                    q1_l1[src] <= l1;
                    q1_source[src] <= source;
                    q1_tid[src] <= tid;
                    q1_bytes[src] <= nbytes;
                    q1_payload_cycles[src] <= payload_cycles;
                    q1_addr[src] <= addr;
                    q1_wdata[src] <= wdata;
                    q1_wstrb[src] <= wstrb;
                end else begin
                    valid0[src] <= 1'b1;
                    valid1[src] <= 1'b0;
                    q0_write[src] <= write;
                    q0_l1[src] <= l1;
                    q0_source[src] <= source;
                    q0_tid[src] <= tid;
                    q0_bytes[src] <= nbytes;
                    q0_payload_cycles[src] <= payload_cycles;
                    q0_addr[src] <= addr;
                    q0_wdata[src] <= wdata;
                    q0_wstrb[src] <= wstrb;
                end
            end else if (!valid0[src]) begin
                valid0[src] <= 1'b1;
                q0_write[src] <= write;
                q0_l1[src] <= l1;
                q0_source[src] <= source;
                q0_tid[src] <= tid;
                q0_bytes[src] <= nbytes;
                q0_payload_cycles[src] <= payload_cycles;
                q0_addr[src] <= addr;
                q0_wdata[src] <= wdata;
                q0_wstrb[src] <= wstrb;
            end else begin
                valid1[src] <= 1'b1;
                q1_write[src] <= write;
                q1_l1[src] <= l1;
                q1_source[src] <= source;
                q1_tid[src] <= tid;
                q1_bytes[src] <= nbytes;
                q1_payload_cycles[src] <= payload_cycles;
                q1_addr[src] <= addr;
                q1_wdata[src] <= wdata;
                q1_wstrb[src] <= wstrb;
            end
        end
    endtask

    always @* begin
        arb_src = SRC_NONE;
        arb_valid = 1'b0;
        if (valid0[SRC_UDMA]) begin
            arb_src = SRC_UDMA;
            arb_valid = 1'b1;
        end else if (valid0[SRC_REQUANT]) begin
            arb_src = SRC_REQUANT;
            arb_valid = 1'b1;
        end else if (valid0[SRC_EWE]) begin
            arb_src = SRC_EWE;
            arb_valid = 1'b1;
        end else if (valid0[SRC_POOL]) begin
            arb_src = SRC_POOL;
            arb_valid = 1'b1;
        end else if (valid0[SRC_TNPS]) begin
            arb_src = SRC_TNPS;
            arb_valid = 1'b1;
        end else if (valid0[SRC_LEGACY]) begin
            arb_src = SRC_LEGACY;
            arb_valid = 1'b1;
        end

        arb_write = 1'b0;
        arb_l1 = 1'b1;
        arb_source = 4'd0;
        arb_tid = 8'd0;
        arb_bytes = 32'd0;
        arb_payload_cycles = 32'd0;
        arb_addr = {ADDR_WIDTH{1'b0}};
        arb_wdata = {DATA_WIDTH{1'b0}};
        arb_wstrb = {DATA_WIDTH/8{1'b0}};
        if (arb_valid) begin
            arb_write = q0_write[arb_src];
            arb_l1 = q0_l1[arb_src];
            arb_source = q0_source[arb_src];
            arb_tid = q0_tid[arb_src];
            arb_bytes = q0_bytes[arb_src];
            arb_payload_cycles = q0_payload_cycles[arb_src];
            arb_addr = q0_addr[arb_src];
            arb_wdata = q0_wdata[arb_src];
            arb_wstrb = q0_wstrb[arb_src];
        end
    end

    wire [31:0] default_l1_cycles = max1(ceil_div(arb_bytes, L1_BYTES_PER_CYCLE));
    wire [31:0] default_dram_cycles = max1(ceil_div(arb_bytes, DRAM_BYTES_PER_CYCLE));
    wire [31:0] payload_cycles = (arb_payload_cycles != 32'd0)
        ? arb_payload_cycles
        : (arb_l1 ? default_l1_cycles : default_dram_cycles);
    wire [3:0] payload_phase = arb_l1
        ? PH_L1_PAYLOAD
        : (arb_write ? PH_DRAM_WRITE_DATA : PH_DRAM_READ_DATA);

    wire [4*32-1:0] phase_cycles = {
        32'd1,
        payload_cycles,
        32'd1,
        32'd1
    };

    wire [4*4-1:0] phase_ids = {
        PH_RESP,
        payload_phase,
        PH_ARB,
        PH_REQ_FETCH
    };

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid0 <= 6'd0;
            valid1 <= 6'd0;
            mesh_req_write <= 1'b0;
            mesh_req_addr <= {ADDR_WIDTH{1'b0}};
            mesh_req_bytes <= 32'd0;
            mesh_req_wdata <= {DATA_WIDTH{1'b0}};
            mesh_req_wstrb <= {DATA_WIDTH/8{1'b0}};
            mesh_req_source <= 4'd0;
            mesh_req_tid <= 8'd0;
            for (i = 0; i < 6; i = i + 1) begin
                q0_write[i] <= 1'b0;
                q1_write[i] <= 1'b0;
                q0_l1[i] <= 1'b1;
                q1_l1[i] <= 1'b1;
                q0_source[i] <= 4'd0;
                q1_source[i] <= 4'd0;
                q0_tid[i] <= 8'd0;
                q1_tid[i] <= 8'd0;
                q0_bytes[i] <= 32'd0;
                q1_bytes[i] <= 32'd0;
                q0_payload_cycles[i] <= 32'd0;
                q1_payload_cycles[i] <= 32'd0;
                q0_addr[i] <= {ADDR_WIDTH{1'b0}};
                q1_addr[i] <= {ADDR_WIDTH{1'b0}};
                q0_wdata[i] <= {DATA_WIDTH{1'b0}};
                q1_wdata[i] <= {DATA_WIDTH{1'b0}};
                q0_wstrb[i] <= {DATA_WIDTH/8{1'b0}};
                q1_wstrb[i] <= {DATA_WIDTH/8{1'b0}};
            end
        end else begin
            if (phase_start_fire) begin
                mesh_req_write <= arb_write;
                mesh_req_addr <= arb_addr;
                mesh_req_bytes <= arb_bytes;
                mesh_req_wdata <= arb_wdata;
                mesh_req_wstrb <= arb_wstrb;
                mesh_req_source <= arb_source;
                mesh_req_tid <= arb_tid;
                if (valid1[arb_src]) begin
                    valid1[arb_src] <= 1'b0;
                    q0_write[arb_src] <= q1_write[arb_src];
                    q0_l1[arb_src] <= q1_l1[arb_src];
                    q0_source[arb_src] <= q1_source[arb_src];
                    q0_tid[arb_src] <= q1_tid[arb_src];
                    q0_bytes[arb_src] <= q1_bytes[arb_src];
                    q0_payload_cycles[arb_src] <= q1_payload_cycles[arb_src];
                    q0_addr[arb_src] <= q1_addr[arb_src];
                    q0_wdata[arb_src] <= q1_wdata[arb_src];
                    q0_wstrb[arb_src] <= q1_wstrb[arb_src];
                end else begin
                    valid0[arb_src] <= 1'b0;
                end
            end

            if (legacy_push)
                enqueue0(SRC_LEGACY, req_write, req_l1,
                         (req_source == 4'd0) ? DBG_LEGACY : req_source,
                         req_tid, req_bytes, req_payload_cycles, req_addr, req_wdata, req_wstrb);
            if (udma_push)
                enqueue0(SRC_UDMA, udma_req_write, 1'b1,
                         DBG_UDMA, udma_req_tid, udma_req_bytes,
                         udma_req_payload_cycles, udma_req_addr, udma_req_wdata, udma_req_wstrb);
            if (requant_push)
                enqueue0(SRC_REQUANT, requant_req_write, 1'b1,
                         DBG_REQUANT, requant_req_tid,
                         requant_req_bytes, requant_req_payload_cycles, requant_req_addr,
                         requant_req_wdata, requant_req_wstrb);
            if (ewe_push)
                enqueue0(SRC_EWE, ewe_req_write, 1'b1,
                         DBG_EWE, ewe_req_tid, ewe_req_bytes,
                         ewe_req_payload_cycles, ewe_req_addr, ewe_req_wdata, ewe_req_wstrb);
            if (pool_push)
                enqueue0(SRC_POOL, pool_req_write, 1'b1,
                         DBG_POOL, pool_req_tid, pool_req_bytes,
                         pool_req_payload_cycles, pool_req_addr, pool_req_wdata, pool_req_wstrb);
            if (tnps_push)
                enqueue0(SRC_TNPS, tnps_req_write, 1'b1,
                         DBG_TNPS, tnps_req_tid, tnps_req_bytes,
                         tnps_req_payload_cycles, tnps_req_addr, tnps_req_wdata, tnps_req_wstrb);
        end
    end

    mdla7_synth_phase_engine #(
        .NUM_PHASES(4),
        .PHASE_W(4)
    ) u_phase (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(arb_valid),
        .start_ready(phase_start_ready),
        .phase_cycles(phase_cycles),
        .phase_ids(phase_ids),
        .phase_stall(1'b0),
        .busy(phase_busy),
        .done_valid(resp_valid),
        .done_ready(resp_ready),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );
endmodule

`endif
