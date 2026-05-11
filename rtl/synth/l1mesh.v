`timescale 1ns/1ps

`include "common.v"

`timescale 1ns/1ps

/* verilator lint_off DECLFILENAME */

module mdla7_l1mesh4x4_tile #(
    parameter DATA_WIDTH = 128,
    parameter TILE_WORDS = 49152
) (
    input                         clk,
    input                         start_fire,
    input                         req_write,
    input      [27:0]             local_word_addr,
    input      [3:0]              bank_id,
    input      [DATA_WIDTH-1:0]   req_wdata,
    input      [DATA_WIDTH/8-1:0] req_wstrb,
    output reg [DATA_WIDTH-1:0]   resp_rdata
);
    localparam STRB_WIDTH = DATA_WIDTH / 8;

    reg [DATA_WIDTH-1:0] mem [0:TILE_WORDS-1];
    wire [31:0] tile_word_addr = {local_word_addr, bank_id};

    integer i;
    always @(posedge clk) begin
        if (start_fire) begin
            if (req_write) begin
                if (tile_word_addr < TILE_WORDS) begin
                    for (i = 0; i < STRB_WIDTH; i = i + 1) begin
                        if (req_wstrb[i])
                            mem[tile_word_addr][i*8 +: 8] <= req_wdata[i*8 +: 8];
                    end
                end
            end else begin
                resp_rdata <= (tile_word_addr < TILE_WORDS)
                    ? mem[tile_word_addr]
                    : {DATA_WIDTH{1'b0}};
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */

module l1mesh #(
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128,
    parameter MEM_WORDS = 196608,
    parameter BYTES_PER_CYCLE = 16,
    parameter SYNTH_L1_PIPE_CYCLES = 3
) (
    input                         clk,
    input                         rst_n,

    input                         req_valid,
    output                        req_ready,
    input                         req_write,
    input      [ADDR_WIDTH-1:0]   req_addr,
    input      [31:0]             req_bytes,
    input      [31:0]             route_cycles,
    input      [DATA_WIDTH-1:0]   req_wdata,
    input      [DATA_WIDTH/8-1:0] req_wstrb,

    output                        resp_valid,
    input                         resp_ready,
    output reg [DATA_WIDTH-1:0]   resp_rdata,
    output                        busy,
    output     [3:0]              phase_id,
    output     [31:0]             remaining_cycles
);
    localparam [3:0] PH_ADDR_DECODE = 4'd1;
    localparam [3:0] PH_GLOBAL_MESH = 4'd2;
    localparam [3:0] PH_TILE_MESH   = 4'd3;
    localparam [3:0] PH_BANK_ARB    = 4'd4;
    localparam [3:0] PH_SRAM_MACRO  = 4'd5;
    localparam [3:0] PH_RESP        = 4'd6;
    localparam STRB_WIDTH = DATA_WIDTH / 8;
    localparam [31:0] TILE_WORDS = (MEM_WORDS + 3) / 4;
    localparam [31:0] SYNTH_L1_PIPE_CYCLES_32 = SYNTH_L1_PIPE_CYCLES;

    wire [31:0] word_addr = {10'd0, req_addr} / STRB_WIDTH;
    wire [5:0]  bank_global = word_addr[5:0];
    wire [1:0]  tile_id = bank_global[5:4];
    wire [3:0]  bank_id = bank_global[3:0];
    wire [1:0]  tile_x = {1'b0, tile_id[0]};
    wire [1:0]  tile_y = {1'b0, tile_id[1]};
    wire [1:0]  bank_x = bank_id[1:0];
    wire [1:0]  bank_y = bank_id[3:2];
    wire [27:0] local_word_addr = {2'd0, word_addr[31:6]};
    wire        start_fire = req_valid && req_ready;
    wire        phase_done;

    wire [DATA_WIDTH-1:0] tile_rdata0;
    wire [DATA_WIDTH-1:0] tile_rdata1;
    wire [DATA_WIDTH-1:0] tile_rdata2;
    wire [DATA_WIDTH-1:0] tile_rdata3;

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

    function [31:0] manhattan2;
        input [1:0] x;
        input [1:0] y;
        begin
            manhattan2 = {30'd0, x} + {30'd0, y};
        end
    endfunction

    wire [31:0] charged_bytes = (req_bytes == 32'd0) ? STRB_WIDTH : req_bytes;
    wire [31:0] global_mesh_cycles = max1(route_cycles);
    wire [31:0] inter_tile_hops = manhattan2(tile_x, tile_y);
    wire [31:0] intra_tile_hops = manhattan2(bank_x, bank_y);
    wire [31:0] tile_mesh_cycles = max1(SYNTH_L1_PIPE_CYCLES_32 + inter_tile_hops + intra_tile_hops);
    wire [31:0] bank_arb_cycles = max1({30'd0, bank_id[1:0]} + 32'd1);
    wire [31:0] sram_macro_cycles = max1(ceil_div(charged_bytes, BYTES_PER_CYCLE));

    wire [6*32-1:0] phase_cycles = {
        32'd1,
        sram_macro_cycles,
        bank_arb_cycles,
        tile_mesh_cycles,
        global_mesh_cycles,
        32'd1
    };

    wire [6*4-1:0] phase_ids = {
        PH_RESP,
        PH_SRAM_MACRO,
        PH_BANK_ARB,
        PH_TILE_MESH,
        PH_GLOBAL_MESH,
        PH_ADDR_DECODE
    };

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile0 (
        .clk(clk),
        .start_fire(start_fire && (tile_id == 2'd0)),
        .req_write(req_write),
        .local_word_addr(local_word_addr),
        .bank_id(bank_id),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata0)
    );

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile1 (
        .clk(clk),
        .start_fire(start_fire && (tile_id == 2'd1)),
        .req_write(req_write),
        .local_word_addr(local_word_addr),
        .bank_id(bank_id),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata1)
    );

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile2 (
        .clk(clk),
        .start_fire(start_fire && (tile_id == 2'd2)),
        .req_write(req_write),
        .local_word_addr(local_word_addr),
        .bank_id(bank_id),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata2)
    );

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile3 (
        .clk(clk),
        .start_fire(start_fire && (tile_id == 2'd3)),
        .req_write(req_write),
        .local_word_addr(local_word_addr),
        .bank_id(bank_id),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata3)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_rdata <= {DATA_WIDTH{1'b0}};
        end else if (start_fire && !req_write) begin
            case (tile_id)
                2'd0: resp_rdata <= tile_rdata0;
                2'd1: resp_rdata <= tile_rdata1;
                2'd2: resp_rdata <= tile_rdata2;
                2'd3: resp_rdata <= tile_rdata3;
                default: resp_rdata <= {DATA_WIDTH{1'b0}};
            endcase
        end
    end

    assign resp_valid = phase_done;

    mdla7_synth_phase_engine #(
        .NUM_PHASES(6),
        .PHASE_W(4)
    ) u_phase (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(req_valid),
        .start_ready(req_ready),
        .phase_cycles(phase_cycles),
        .phase_ids(phase_ids),
        .phase_stall(1'b0),
        .busy(busy),
        .done_valid(phase_done),
        .done_ready(resp_ready),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );
endmodule
