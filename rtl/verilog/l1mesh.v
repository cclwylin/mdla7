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
    input      [3:0]              req_source,
    input      [7:0]              req_tid,

    input                         debug_crc_start,
    input      [ADDR_WIDTH-1:0]   debug_crc_addr,
    input      [31:0]             debug_crc_count,
    output reg                    debug_crc_busy,
    output reg                    debug_crc_done,
    output reg [31:0]             debug_crc,
    output reg [31:0]             debug_crc_byte_count,

    output                        resp_valid,
    output                        resp_read,
    output reg [3:0]              resp_source,
    output reg [7:0]              resp_tid,
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
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;

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
    reg [ADDR_WIDTH-1:0] debug_crc_scan_addr;
    reg [31:0] debug_crc_remaining;
    reg [31:0] debug_crc_value;
    reg [31:0] debug_crc_count_value;
    reg resp_read_q;
    integer debug_crc_i;

    /* verilator lint_off UNUSEDSIGNAL */
    wire [DATA_WIDTH-1:0] tile_rdata0;
    wire [DATA_WIDTH-1:0] tile_rdata1;
    wire [DATA_WIDTH-1:0] tile_rdata2;
    wire [DATA_WIDTH-1:0] tile_rdata3;
    /* verilator lint_on UNUSEDSIGNAL */

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

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_rdata <= {DATA_WIDTH{1'b0}};
            resp_read_q <= 1'b0;
            resp_source <= 4'd0;
            resp_tid <= 8'd0;
            debug_crc_busy <= 1'b0;
            debug_crc_done <= 1'b0;
            debug_crc <= FNV_OFFSET;
            debug_crc_byte_count <= 32'd0;
            debug_crc_scan_addr <= {ADDR_WIDTH{1'b0}};
            debug_crc_remaining <= 32'd0;
        end else if (start_fire) begin
            resp_read_q <= !req_write;
            resp_source <= req_source;
            resp_tid <= req_tid;
            if (!req_write)
                resp_rdata <= debug_read_word(req_addr);
        end else begin
            if (resp_valid && resp_ready) begin
                resp_read_q <= 1'b0;
                resp_source <= 4'd0;
                resp_tid <= 8'd0;
            end
            debug_crc_done <= 1'b0;
            if (debug_crc_start && !debug_crc_busy) begin
                debug_crc_busy <= (debug_crc_count != 32'd0);
                debug_crc_done <= (debug_crc_count == 32'd0);
                debug_crc <= FNV_OFFSET;
                debug_crc_byte_count <= 32'd0;
                debug_crc_scan_addr <= debug_crc_addr;
                debug_crc_remaining <= debug_crc_count;
            end else if (debug_crc_busy) begin
                debug_crc_value = debug_crc;
                debug_crc_count_value = debug_crc_byte_count;
                for (debug_crc_i = 0; debug_crc_i < 16; debug_crc_i = debug_crc_i + 1) begin
                    if (debug_crc_i < debug_crc_remaining) begin
                        debug_crc_value = fnv_byte(
                            debug_crc_value,
                            debug_read_byte(debug_crc_scan_addr + debug_crc_i[ADDR_WIDTH-1:0])
                        );
                        debug_crc_count_value = debug_crc_count_value + 32'd1;
                    end
                end
                debug_crc <= debug_crc_value;
                debug_crc_byte_count <= debug_crc_count_value;
                if (debug_crc_remaining <= 32'd16) begin
                    debug_crc_busy <= 1'b0;
                    debug_crc_done <= 1'b1;
                    debug_crc_remaining <= 32'd0;
                end else begin
                    debug_crc_remaining <= debug_crc_remaining - 32'd16;
                    debug_crc_scan_addr <= debug_crc_scan_addr + {{(ADDR_WIDTH-5){1'b0}}, 5'd16};
                end
            end
        end
    end
    /* verilator lint_on BLKSEQ */

    function [31:0] fnv_byte;
        input [31:0] crc;
        input [7:0] byte_value;
        begin
            fnv_byte = (crc ^ {24'd0, byte_value}) * FNV_PRIME;
        end
    endfunction

    function [7:0] debug_read_byte;
        input [ADDR_WIDTH-1:0] byte_addr;
        reg [31:0] dbg_word_addr;
        reg [5:0] dbg_bank_global;
        reg [1:0] dbg_tile_id;
        reg [3:0] dbg_bank_id;
        reg [27:0] dbg_local_word_addr;
        reg [31:0] dbg_tile_word_addr;
        reg [3:0] dbg_lane;
        begin
            dbg_word_addr = {10'd0, byte_addr} / STRB_WIDTH;
            dbg_bank_global = dbg_word_addr[5:0];
            dbg_tile_id = dbg_bank_global[5:4];
            dbg_bank_id = dbg_bank_global[3:0];
            dbg_local_word_addr = {2'd0, dbg_word_addr[31:6]};
            dbg_tile_word_addr = {dbg_local_word_addr, dbg_bank_id};
            dbg_lane = byte_addr[3:0];
            if (dbg_tile_word_addr >= TILE_WORDS) begin
                debug_read_byte = 8'd0;
            end else begin
                case (dbg_tile_id)
                    2'd0: debug_read_byte = u_tile0.mem[dbg_tile_word_addr][dbg_lane*8 +: 8];
                    2'd1: debug_read_byte = u_tile1.mem[dbg_tile_word_addr][dbg_lane*8 +: 8];
                    2'd2: debug_read_byte = u_tile2.mem[dbg_tile_word_addr][dbg_lane*8 +: 8];
                    2'd3: debug_read_byte = u_tile3.mem[dbg_tile_word_addr][dbg_lane*8 +: 8];
                    default: debug_read_byte = 8'd0;
                endcase
            end
        end
    endfunction

    function [DATA_WIDTH-1:0] debug_read_word;
        input [ADDR_WIDTH-1:0] byte_addr;
        integer read_i;
        begin
            debug_read_word = {DATA_WIDTH{1'b0}};
            for (read_i = 0; read_i < STRB_WIDTH; read_i = read_i + 1)
                debug_read_word[read_i*8 +: 8] =
                    debug_read_byte(byte_addr + read_i[ADDR_WIDTH-1:0]);
        end
    endfunction

    assign resp_valid = phase_done && !start_fire;
    assign resp_read = resp_read_q;

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
