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
    input      [31:0]             tile_word_addr,
    input      [DATA_WIDTH-1:0]   req_wdata,
    input      [DATA_WIDTH/8-1:0] req_wstrb,
    output reg [DATA_WIDTH-1:0]   resp_rdata
);
    localparam STRB_WIDTH = DATA_WIDTH / 8;

    reg [DATA_WIDTH-1:0] mem [0:TILE_WORDS-1];

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

module mdla7_mesh4x4_router_node #(
    parameter NODE_X = 0,
    parameter NODE_Y = 0
) (
    input        packet_valid,
    input  [1:0] dst_x,
    input  [1:0] dst_y,
    output       north_valid,
    output       south_valid,
    output       west_valid,
    output       east_valid,
    output       local_valid
);
    assign east_valid = packet_valid && (NODE_X[1:0] < dst_x);
    assign west_valid = packet_valid && (NODE_X[1:0] > dst_x);
    assign south_valid = packet_valid && (NODE_X[1:0] == dst_x) &&
                         (NODE_Y[1:0] < dst_y);
    assign north_valid = packet_valid && (NODE_X[1:0] == dst_x) &&
                         (NODE_Y[1:0] > dst_y);
    assign local_valid = packet_valid && (NODE_X[1:0] == dst_x) &&
                         (NODE_Y[1:0] == dst_y);
endmodule

module mdla7_mesh4x4_edge_fabric (
    input        clk,
    input        rst_n,
    input        start,
    input  [1:0] src_x,
    input  [1:0] src_y,
    input  [1:0] dst_x,
    input  [1:0] dst_y,
    output reg   busy,
    output reg   done,
    output reg [3:0] hops,
    output       north_link_valid,
    output       south_link_valid,
    output       west_link_valid,
    output       east_link_valid,
    output       local_link_valid
);
    reg [1:0] cur_x;
    reg [1:0] cur_y;
    reg [1:0] target_x;
    reg [1:0] target_y;

    wire [15:0] node_packet_valid;
    wire [15:0] node_north_valid;
    wire [15:0] node_south_valid;
    wire [15:0] node_west_valid;
    wire [15:0] node_east_valid;
    wire [15:0] node_local_valid;
    wire [11:0] hlink_east_valid;
    wire [11:0] hlink_west_valid;
    wire [11:0] vlink_south_valid;
    wire [11:0] vlink_north_valid;

    genvar nx;
    genvar ny;
    generate
        for (ny = 0; ny < 4; ny = ny + 1) begin : gen_mesh_y
            for (nx = 0; nx < 4; nx = nx + 1) begin : gen_mesh_x
                localparam integer NODE_ID = ny * 4 + nx;
                assign node_packet_valid[NODE_ID] =
                    busy && (cur_x == nx[1:0]) && (cur_y == ny[1:0]);
                mdla7_mesh4x4_router_node #(
                    .NODE_X(nx),
                    .NODE_Y(ny)
                ) u_router (
                    .packet_valid(node_packet_valid[NODE_ID]),
                    .dst_x(target_x),
                    .dst_y(target_y),
                    .north_valid(node_north_valid[NODE_ID]),
                    .south_valid(node_south_valid[NODE_ID]),
                    .west_valid(node_west_valid[NODE_ID]),
                    .east_valid(node_east_valid[NODE_ID]),
                    .local_valid(node_local_valid[NODE_ID])
                );
            end
        end
    endgenerate

    genvar hx;
    genvar hy;
    generate
        for (hy = 0; hy < 4; hy = hy + 1) begin : gen_hlink_y
            for (hx = 0; hx < 3; hx = hx + 1) begin : gen_hlink_x
                localparam integer HLINK_ID = hy * 3 + hx;
                localparam integer WEST_NODE_ID = hy * 4 + hx;
                localparam integer EAST_NODE_ID = hy * 4 + hx + 1;
                assign hlink_east_valid[HLINK_ID] = node_east_valid[WEST_NODE_ID];
                assign hlink_west_valid[HLINK_ID] = node_west_valid[EAST_NODE_ID];
            end
        end
    endgenerate

    genvar vx;
    genvar vy;
    generate
        for (vy = 0; vy < 3; vy = vy + 1) begin : gen_vlink_y
            for (vx = 0; vx < 4; vx = vx + 1) begin : gen_vlink_x
                localparam integer VLINK_ID = vy * 4 + vx;
                localparam integer NORTH_NODE_ID = vy * 4 + vx;
                localparam integer SOUTH_NODE_ID = (vy + 1) * 4 + vx;
                assign vlink_south_valid[VLINK_ID] = node_south_valid[NORTH_NODE_ID];
                assign vlink_north_valid[VLINK_ID] = node_north_valid[SOUTH_NODE_ID];
            end
        end
    endgenerate

    assign north_link_valid = |vlink_north_valid;
    assign south_link_valid = |vlink_south_valid;
    assign west_link_valid = |hlink_west_valid;
    assign east_link_valid = |hlink_east_valid;
    assign local_link_valid = |node_local_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            hops <= 4'd0;
            cur_x <= 2'd0;
            cur_y <= 2'd0;
            target_x <= 2'd0;
            target_y <= 2'd0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                cur_x <= src_x;
                cur_y <= src_y;
                target_x <= dst_x;
                target_y <= dst_y;
                hops <= 4'd0;
            end else if (busy) begin
                if (east_link_valid) begin
                    cur_x <= cur_x + 2'd1;
                    hops <= hops + 4'd1;
                end else if (west_link_valid) begin
                    cur_x <= cur_x - 2'd1;
                    hops <= hops + 4'd1;
                end else if (south_link_valid) begin
                    cur_y <= cur_y + 2'd1;
                    hops <= hops + 4'd1;
                end else if (north_link_valid) begin
                    cur_y <= cur_y - 2'd1;
                    hops <= hops + 4'd1;
                end else if (local_link_valid) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
        end
    end
endmodule

module l1mesh #(
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128,
    parameter MEM_WORDS = 196608,
    parameter BYTES_PER_CYCLE = 16,
    parameter SYNTH_L1_PIPE_CYCLES = 3,
    parameter EDGE_MESH4X4_COUNT = 8,
    parameter STORAGE_MESH4X4_COUNT = 4,
    parameter QUAD_SRAM_PORTS = 4,
    parameter SRAM_MACRO_WORDS = 768
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
    localparam [3:0] PH_L1MESH_SELECT    = 4'd2;
    localparam [3:0] PH_MESH4X4_ROUTE    = 4'd3;
    localparam [3:0] PH_QUAD_SRAM_SELECT = 4'd4;
    localparam [3:0] PH_SRAM_MACRO       = 4'd5;
    localparam [3:0] PH_RESP             = 4'd6;
    localparam STRB_WIDTH = DATA_WIDTH / 8;
    localparam [31:0] TILE_WORDS = (MEM_WORDS + STORAGE_MESH4X4_COUNT - 1) /
                                    STORAGE_MESH4X4_COUNT;
    localparam [31:0] SYNTH_L1_PIPE_CYCLES_32 = SYNTH_L1_PIPE_CYCLES;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;

    wire [31:0] word_addr = {10'd0, req_addr} / STRB_WIDTH;
    wire [1:0]  storage_mesh_id = word_addr[1:0];
    wire [3:0]  quad_sram_id = word_addr[5:2];
    wire [1:0]  sram_macro_port = word_addr[7:6];
    wire        edge_mesh_half = word_addr[8];
    wire [2:0]  edge_mesh_id = {edge_mesh_half, storage_mesh_id};
    wire [31:0] sram_macro_word = word_addr >> 8;
    wire [31:0] storage_tile_word_addr = (((sram_macro_word << 4) +
                                           {28'd0, quad_sram_id}) << 2) +
                                           {30'd0, sram_macro_port};
    wire [1:0]  quad_x = quad_sram_id[1:0];
    wire [1:0]  quad_y = quad_sram_id[3:2];
    wire        start_fire = req_valid && req_ready;
    wire        phase_done;
    reg [ADDR_WIDTH-1:0] debug_crc_scan_addr;
    reg [31:0] debug_crc_remaining;
    reg [31:0] debug_crc_value;
    reg [31:0] debug_crc_count_value;
    reg resp_read_q;
    reg [1:0] route_src_x;
    reg [1:0] route_src_y;
    integer debug_crc_i;

    /* verilator lint_off UNUSEDSIGNAL */
    wire [DATA_WIDTH-1:0] tile_rdata0;
    wire [DATA_WIDTH-1:0] tile_rdata1;
    wire [DATA_WIDTH-1:0] tile_rdata2;
    wire [DATA_WIDTH-1:0] tile_rdata3;
    /* verilator lint_on UNUSEDSIGNAL */

    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_busy;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_done;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_north;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_south;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_west;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_east;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_local;
    wire [EDGE_MESH4X4_COUNT*4-1:0] edge_route_hops;

    always @* begin
        case (req_source)
            4'd1: begin route_src_x = 2'd0; route_src_y = 2'd0; end
            4'd2: begin route_src_x = 2'd1; route_src_y = 2'd0; end
            4'd3: begin route_src_x = 2'd0; route_src_y = 2'd1; end
            4'd4: begin route_src_x = 2'd1; route_src_y = 2'd1; end
            4'd5: begin route_src_x = 2'd0; route_src_y = 2'd1; end
            4'd6: begin route_src_x = 2'd1; route_src_y = 2'd0; end
            default: begin route_src_x = 2'd0; route_src_y = 2'd0; end
        endcase
    end

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
    wire [31:0] l1mesh_select_cycles = max1(route_cycles);
    wire [31:0] mesh4x4_route_cycles = max1(SYNTH_L1_PIPE_CYCLES_32 +
                                             manhattan2(quad_x, quad_y));
    wire [31:0] quad_sram_select_cycles = max1({30'd0, sram_macro_port} + 32'd1);
    wire [31:0] sram_macro_cycles = max1(ceil_div(charged_bytes, BYTES_PER_CYCLE));

    /* verilator lint_off UNUSEDSIGNAL */
    wire [31:0] l1mesh_shape_unused = EDGE_MESH4X4_COUNT +
                                      QUAD_SRAM_PORTS +
                                      SRAM_MACRO_WORDS +
                                      {29'd0, edge_mesh_id} +
                                      {31'd0, |edge_route_done} +
                                      {31'd0, |edge_route_north} +
                                      {31'd0, |edge_route_south} +
                                      {31'd0, |edge_route_west} +
                                      {31'd0, |edge_route_east} +
                                      {31'd0, |edge_route_local} +
                                      {28'd0, edge_route_hops[3:0]};
    /* verilator lint_on UNUSEDSIGNAL */

    wire [6*32-1:0] phase_cycles = {
        32'd1,
        sram_macro_cycles,
        quad_sram_select_cycles,
        mesh4x4_route_cycles,
        l1mesh_select_cycles,
        32'd1
    };

    wire [6*4-1:0] phase_ids = {
        PH_RESP,
        PH_SRAM_MACRO,
        PH_QUAD_SRAM_SELECT,
        PH_MESH4X4_ROUTE,
        PH_L1MESH_SELECT,
        PH_ADDR_DECODE
    };

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile0 (
        .clk(clk),
        .start_fire(start_fire && (storage_mesh_id == 2'd0)),
        .req_write(req_write),
        .tile_word_addr(storage_tile_word_addr),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata0)
    );

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile1 (
        .clk(clk),
        .start_fire(start_fire && (storage_mesh_id == 2'd1)),
        .req_write(req_write),
        .tile_word_addr(storage_tile_word_addr),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata1)
    );

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile2 (
        .clk(clk),
        .start_fire(start_fire && (storage_mesh_id == 2'd2)),
        .req_write(req_write),
        .tile_word_addr(storage_tile_word_addr),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata2)
    );

    mdla7_l1mesh4x4_tile #(
        .DATA_WIDTH(DATA_WIDTH),
        .TILE_WORDS(TILE_WORDS)
    ) u_tile3 (
        .clk(clk),
        .start_fire(start_fire && (storage_mesh_id == 2'd3)),
        .req_write(req_write),
        .tile_word_addr(storage_tile_word_addr),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .resp_rdata(tile_rdata3)
    );

    genvar edge_mesh_gen;
    generate
        for (edge_mesh_gen = 0;
             edge_mesh_gen < EDGE_MESH4X4_COUNT;
             edge_mesh_gen = edge_mesh_gen + 1) begin : gen_edge_mesh
            mdla7_mesh4x4_edge_fabric u_edge_mesh (
                .clk(clk),
                .rst_n(rst_n),
                .start(start_fire && (edge_mesh_id == edge_mesh_gen[2:0])),
                .src_x(route_src_x),
                .src_y(route_src_y),
                .dst_x(quad_x),
                .dst_y(quad_y),
                .busy(edge_route_busy[edge_mesh_gen]),
                .done(edge_route_done[edge_mesh_gen]),
                .hops(edge_route_hops[edge_mesh_gen*4 +: 4]),
                .north_link_valid(edge_route_north[edge_mesh_gen]),
                .south_link_valid(edge_route_south[edge_mesh_gen]),
                .west_link_valid(edge_route_west[edge_mesh_gen]),
                .east_link_valid(edge_route_east[edge_mesh_gen]),
                .local_link_valid(edge_route_local[edge_mesh_gen])
            );
        end
    endgenerate

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
        reg [1:0] dbg_storage_mesh_id;
        reg [3:0] dbg_quad_sram_id;
        reg [1:0] dbg_sram_macro_port;
        reg [31:0] dbg_sram_macro_word;
        reg [31:0] dbg_tile_word_addr;
        reg [3:0] dbg_lane;
        begin
            dbg_word_addr = {10'd0, byte_addr} / STRB_WIDTH;
            dbg_storage_mesh_id = dbg_word_addr[1:0];
            dbg_quad_sram_id = dbg_word_addr[5:2];
            dbg_sram_macro_port = dbg_word_addr[7:6];
            dbg_sram_macro_word = dbg_word_addr >> 8;
            dbg_tile_word_addr = (((dbg_sram_macro_word << 4) +
                                   {28'd0, dbg_quad_sram_id}) << 2) +
                                   {30'd0, dbg_sram_macro_port};
            dbg_lane = byte_addr[3:0];
            if (dbg_tile_word_addr >= TILE_WORDS) begin
                debug_read_byte = 8'd0;
            end else begin
                case (dbg_storage_mesh_id)
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
        reg [31:0] dbg_word_addr;
        reg [1:0] dbg_storage_mesh_id;
        reg [3:0] dbg_quad_sram_id;
        reg [1:0] dbg_sram_macro_port;
        reg [31:0] dbg_sram_macro_word;
        reg [31:0] dbg_tile_word_addr;
        begin
            dbg_word_addr = {10'd0, byte_addr} / STRB_WIDTH;
            dbg_storage_mesh_id = dbg_word_addr[1:0];
            dbg_quad_sram_id = dbg_word_addr[5:2];
            dbg_sram_macro_port = dbg_word_addr[7:6];
            dbg_sram_macro_word = dbg_word_addr >> 8;
            dbg_tile_word_addr = (((dbg_sram_macro_word << 4) +
                                   {28'd0, dbg_quad_sram_id}) << 2) +
                                   {30'd0, dbg_sram_macro_port};
            if (dbg_tile_word_addr >= TILE_WORDS) begin
                debug_read_word = {DATA_WIDTH{1'b0}};
            end else begin
                case (dbg_storage_mesh_id)
                    2'd0: debug_read_word = u_tile0.mem[dbg_tile_word_addr];
                    2'd1: debug_read_word = u_tile1.mem[dbg_tile_word_addr];
                    2'd2: debug_read_word = u_tile2.mem[dbg_tile_word_addr];
                    2'd3: debug_read_word = u_tile3.mem[dbg_tile_word_addr];
                    default: debug_read_word = {DATA_WIDTH{1'b0}};
                endcase
            end
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
