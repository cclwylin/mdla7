`timescale 1ns/1ps

`include "common.v"

`timescale 1ns/1ps

/* verilator lint_off DECLFILENAME */

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
    input        flit_valid,
    output       flit_ready,
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
    localparam FIFO_DEPTH = 4;
    localparam DIR_N = 0;
    localparam DIR_S = 1;
    localparam DIR_W = 2;
    localparam DIR_E = 3;
    localparam DIR_LOCAL = 4;

    reg [1:0] cur_x;
    reg [1:0] cur_y;
    reg [1:0] target_x;
    reg [1:0] target_y;
    reg [2:0] current_input_dir;
    reg [2:0] input_fifo_count [0:15][0:4];

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
    wire [3:0] inject_node_id = {src_y, src_x};
    wire [3:0] cur_node_id = {cur_y, cur_x};
    reg [3:0] next_node_id;
    reg [2:0] selected_dir;
    reg [2:0] next_input_dir;
    reg downstream_ready;
    integer fifo_node_i;
    integer fifo_dir_i;

    assign flit_ready = !busy &&
                        (input_fifo_count[inject_node_id][DIR_LOCAL] < FIFO_DEPTH);

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

    always @* begin
        next_node_id = cur_node_id;
        selected_dir = DIR_LOCAL;
        next_input_dir = DIR_LOCAL;
        if (busy && (cur_x < target_x)) begin
            next_node_id = {cur_y, cur_x + 2'd1};
            selected_dir = DIR_E;
            next_input_dir = DIR_W;
        end else if (busy && (cur_x > target_x)) begin
            next_node_id = {cur_y, cur_x - 2'd1};
            selected_dir = DIR_W;
            next_input_dir = DIR_E;
        end else if (busy && (cur_y < target_y)) begin
            next_node_id = {cur_y + 2'd1, cur_x};
            selected_dir = DIR_S;
            next_input_dir = DIR_N;
        end else if (busy && (cur_y > target_y)) begin
            next_node_id = {cur_y - 2'd1, cur_x};
            selected_dir = DIR_N;
            next_input_dir = DIR_S;
        end
        downstream_ready = (selected_dir == DIR_LOCAL) ||
                           (input_fifo_count[next_node_id][next_input_dir] < FIFO_DEPTH);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            hops <= 4'd0;
            cur_x <= 2'd0;
            cur_y <= 2'd0;
            target_x <= 2'd0;
            target_y <= 2'd0;
            current_input_dir <= DIR_LOCAL;
            for (fifo_node_i = 0; fifo_node_i < 16; fifo_node_i = fifo_node_i + 1) begin
                for (fifo_dir_i = 0; fifo_dir_i < 5; fifo_dir_i = fifo_dir_i + 1) begin
                    input_fifo_count[fifo_node_i][fifo_dir_i] <= 3'd0;
                end
            end
        end else begin
            done <= 1'b0;
            if (flit_valid && flit_ready) begin
                busy <= 1'b1;
                cur_x <= src_x;
                cur_y <= src_y;
                target_x <= dst_x;
                target_y <= dst_y;
                current_input_dir <= DIR_LOCAL;
                input_fifo_count[inject_node_id][DIR_LOCAL] <=
                    input_fifo_count[inject_node_id][DIR_LOCAL] + 3'd1;
                hops <= 4'd0;
            end else if (busy && downstream_ready) begin
                if (input_fifo_count[cur_node_id][current_input_dir] != 3'd0)
                    input_fifo_count[cur_node_id][current_input_dir] <=
                        input_fifo_count[cur_node_id][current_input_dir] - 3'd1;
                if (selected_dir == DIR_LOCAL) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    input_fifo_count[next_node_id][next_input_dir] <=
                        input_fifo_count[next_node_id][next_input_dir] + 3'd1;
                    cur_x <= next_node_id[1:0];
                    cur_y <= next_node_id[3:2];
                    current_input_dir <= next_input_dir;
                    hops <= hops + 4'd1;
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
    output     [3:0]              resp_source,
    output     [7:0]              resp_tid,
    input                         resp_ready,
    output     [DATA_WIDTH-1:0]   resp_rdata,
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
    localparam [31:0] SYNTH_L1_PIPE_CYCLES_32 = SYNTH_L1_PIPE_CYCLES;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam SRAM_MACRO_COUNT = STORAGE_MESH4X4_COUNT * 16 * QUAD_SRAM_PORTS;
    localparam PAYLOAD_LANE_COUNT = 104;
    localparam LANE_CONV_ACT_R_BASE = 0;
    localparam LANE_CONV_WGT_R_BASE = 32;
    localparam LANE_L1MGR_R_BASE = 64;
    localparam LANE_L1MGR_W_BASE = 80;
    localparam LANE_REQUANT_W_BASE = 96;
    localparam LANE_CONV_ACT_R_COUNT = 32;
    localparam LANE_CONV_WGT_R_COUNT = 32;
    localparam LANE_L1MGR_R_COUNT = 16;
    localparam LANE_L1MGR_W_COUNT = 16;
    localparam LANE_REQUANT_W_COUNT = 8;
    localparam EXPECTED_MEM_WORDS = STORAGE_MESH4X4_COUNT * 16 *
                                     QUAD_SRAM_PORTS * SRAM_MACRO_WORDS;

    initial begin
        if (MEM_WORDS != EXPECTED_MEM_WORDS)
            $error("l1mesh MEM_WORDS does not match SRAM macro hierarchy");
    end

    reg                         q0_valid;
    reg                         q1_valid;
    reg                         q0_write;
    reg                         q1_write;
    reg [ADDR_WIDTH-1:0]        q0_addr;
    reg [ADDR_WIDTH-1:0]        q1_addr;
    reg [31:0]                  q0_bytes;
    reg [31:0]                  q1_bytes;
    reg [31:0]                  q0_route_cycles;
    reg [31:0]                  q1_route_cycles;
    reg [DATA_WIDTH-1:0]        q0_wdata;
    reg [DATA_WIDTH-1:0]        q1_wdata;
    reg [DATA_WIDTH/8-1:0]      q0_wstrb;
    reg [DATA_WIDTH/8-1:0]      q1_wstrb;
    reg [3:0]                   q0_source;
    reg [3:0]                   q1_source;
    reg [7:0]                   q0_tid;
    reg [7:0]                   q1_tid;
    reg [31:0]                  sram_macro_busy [0:SRAM_MACRO_COUNT-1];
    reg [31:0]                  payload_lane_busy [0:PAYLOAD_LANE_COUNT-1];
    reg [DATA_WIDTH-1:0]        sram_macro_mem [0:SRAM_MACRO_COUNT-1]
                                               [0:SRAM_MACRO_WORDS-1];
    reg                         active_resp_read;
    reg [3:0]                   active_resp_source;
    reg [7:0]                   active_resp_tid;
    reg [DATA_WIDTH-1:0]        active_resp_rdata;
    reg                         resp0_valid;
    reg                         resp1_valid;
    reg                         resp0_read;
    reg                         resp1_read;
    reg [3:0]                   resp0_source;
    reg [3:0]                   resp1_source;
    reg [7:0]                   resp0_tid;
    reg [7:0]                   resp1_tid;
    reg [DATA_WIDTH-1:0]        resp0_rdata;
    reg [DATA_WIDTH-1:0]        resp1_rdata;

    wire phase_busy;
    wire phase_start_ready;
    wire selected_edge_busy;
    wire selected_sram_busy;
    wire selected_payload_lane_busy;
    wire resource_start_ready;
    wire phase_start_valid = q0_valid && resource_start_ready;
    wire phase_start_fire = phase_start_valid && phase_start_ready;
    wire phase_done_ready = !resp1_valid;
    wire phase_done_push = phase_done && phase_done_ready;
    wire req_push = req_valid && req_ready;
    wire resp_pop = resp_valid && resp_ready;

    assign req_ready = !q1_valid || phase_start_fire;
    assign busy = phase_busy || q0_valid || q1_valid || phase_done ||
                  resp0_valid || resp1_valid;

    wire [31:0] word_addr = {10'd0, q0_addr} / STRB_WIDTH;
    wire [1:0]  storage_mesh_id = word_addr[1:0];
    wire [3:0]  quad_sram_id = word_addr[5:2];
    wire [1:0]  sram_macro_port = word_addr[7:6];
    wire        edge_mesh_half = word_addr[8];
    wire [2:0]  edge_mesh_id = {edge_mesh_half, storage_mesh_id};
    wire [31:0] sram_macro_word = word_addr >> 8;
    wire [7:0]  sram_macro_index = {storage_mesh_id, quad_sram_id, sram_macro_port};
    wire [1:0]  quad_x = quad_sram_id[1:0];
    wire [1:0]  quad_y = quad_sram_id[3:2];
    wire        phase_done;
    reg [ADDR_WIDTH-1:0] debug_crc_scan_addr;
    reg [31:0] debug_crc_remaining;
    reg [31:0] debug_crc_value;
    reg [31:0] debug_crc_count_value;
    reg [1:0] route_src_x;
    reg [1:0] route_src_y;
    integer debug_crc_i;
    integer resource_i;
    integer lane_i;
    integer sram_byte_i;

    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_busy;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_done;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_north;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_south;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_west;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_east;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_route_local;
    wire [EDGE_MESH4X4_COUNT*4-1:0] edge_route_hops;
    wire [EDGE_MESH4X4_COUNT-1:0] edge_flit_ready;
    reg [31:0] payload_lane_start;
    reg [31:0] payload_lane_count;
    reg [6:0] selected_payload_lane;
    reg payload_lane_available;
    integer payload_lane_i;

    assign selected_edge_busy = !edge_flit_ready[edge_mesh_id];
    assign selected_sram_busy = (sram_macro_busy[sram_macro_index] != 32'd0);
    assign selected_payload_lane_busy = !payload_lane_available;
    assign resource_start_ready = !selected_edge_busy && !selected_sram_busy &&
                                  !selected_payload_lane_busy;

    always @* begin
        case (q0_source)
            4'd1: begin route_src_x = 2'd0; route_src_y = 2'd0; end
            4'd2: begin route_src_x = 2'd1; route_src_y = 2'd0; end
            4'd3: begin route_src_x = 2'd0; route_src_y = 2'd1; end
            4'd4: begin route_src_x = 2'd1; route_src_y = 2'd1; end
            4'd5: begin route_src_x = 2'd0; route_src_y = 2'd1; end
            4'd6: begin route_src_x = 2'd1; route_src_y = 2'd0; end
            default: begin route_src_x = 2'd0; route_src_y = 2'd0; end
        endcase
    end

    always @* begin
        if (q0_source == 4'd1 && !q0_write) begin
            payload_lane_start = LANE_CONV_ACT_R_BASE;
            payload_lane_count = LANE_CONV_ACT_R_COUNT;
        end else if (q0_source == 4'd7 && !q0_write) begin
            payload_lane_start = LANE_CONV_WGT_R_BASE;
            payload_lane_count = LANE_CONV_WGT_R_COUNT;
        end else if (q0_source == 4'd2 && q0_write) begin
            payload_lane_start = LANE_REQUANT_W_BASE;
            payload_lane_count = LANE_REQUANT_W_COUNT;
        end else if (q0_write) begin
            payload_lane_start = LANE_L1MGR_W_BASE;
            payload_lane_count = LANE_L1MGR_W_COUNT;
        end else begin
            payload_lane_start = LANE_L1MGR_R_BASE;
            payload_lane_count = LANE_L1MGR_R_COUNT;
        end
        selected_payload_lane = payload_lane_start[6:0];
        payload_lane_available = 1'b0;
        for (payload_lane_i = 0; payload_lane_i < PAYLOAD_LANE_COUNT; payload_lane_i = payload_lane_i + 1) begin
            if (!payload_lane_available &&
                (payload_lane_i >= payload_lane_start) &&
                (payload_lane_i < payload_lane_start + payload_lane_count) &&
                (payload_lane_busy[payload_lane_i] == 32'd0)) begin
                selected_payload_lane = payload_lane_i[6:0];
                payload_lane_available = 1'b1;
            end
        end
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

    wire [31:0] charged_bytes = (q0_bytes == 32'd0) ? STRB_WIDTH : q0_bytes;
    wire [31:0] l1mesh_select_cycles = max1(q0_route_cycles);
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

    genvar edge_mesh_gen;
    generate
        for (edge_mesh_gen = 0;
             edge_mesh_gen < EDGE_MESH4X4_COUNT;
             edge_mesh_gen = edge_mesh_gen + 1) begin : gen_edge_mesh
            mdla7_mesh4x4_edge_fabric u_edge_mesh (
                .clk(clk),
                .rst_n(rst_n),
                .flit_valid(phase_start_fire && (edge_mesh_id == edge_mesh_gen[2:0])),
                .flit_ready(edge_flit_ready[edge_mesh_gen]),
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

    task load_q0;
        input write;
        input [ADDR_WIDTH-1:0] addr;
        input [31:0] bytes;
        input [31:0] route;
        input [DATA_WIDTH-1:0] wdata;
        input [DATA_WIDTH/8-1:0] wstrb;
        input [3:0] source;
        input [7:0] tid;
        begin
            q0_write <= write;
            q0_addr <= addr;
            q0_bytes <= bytes;
            q0_route_cycles <= route;
            q0_wdata <= wdata;
            q0_wstrb <= wstrb;
            q0_source <= source;
            q0_tid <= tid;
        end
    endtask

    task load_q1;
        input write;
        input [ADDR_WIDTH-1:0] addr;
        input [31:0] bytes;
        input [31:0] route;
        input [DATA_WIDTH-1:0] wdata;
        input [DATA_WIDTH/8-1:0] wstrb;
        input [3:0] source;
        input [7:0] tid;
        begin
            q1_write <= write;
            q1_addr <= addr;
            q1_bytes <= bytes;
            q1_route_cycles <= route;
            q1_wdata <= wdata;
            q1_wstrb <= wstrb;
            q1_source <= source;
            q1_tid <= tid;
        end
    endtask

    task move_q1_to_q0;
        begin
            q0_write <= q1_write;
            q0_addr <= q1_addr;
            q0_bytes <= q1_bytes;
            q0_route_cycles <= q1_route_cycles;
            q0_wdata <= q1_wdata;
            q0_wstrb <= q1_wstrb;
            q0_source <= q1_source;
            q0_tid <= q1_tid;
        end
    endtask

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            q0_valid <= 1'b0;
            q1_valid <= 1'b0;
            q0_write <= 1'b0;
            q1_write <= 1'b0;
            q0_addr <= {ADDR_WIDTH{1'b0}};
            q1_addr <= {ADDR_WIDTH{1'b0}};
            q0_bytes <= 32'd0;
            q1_bytes <= 32'd0;
            q0_route_cycles <= 32'd0;
            q1_route_cycles <= 32'd0;
            q0_wdata <= {DATA_WIDTH{1'b0}};
            q1_wdata <= {DATA_WIDTH{1'b0}};
            q0_wstrb <= {DATA_WIDTH/8{1'b0}};
            q1_wstrb <= {DATA_WIDTH/8{1'b0}};
            q0_source <= 4'd0;
            q1_source <= 4'd0;
            q0_tid <= 8'd0;
            q1_tid <= 8'd0;
            for (resource_i = 0; resource_i < SRAM_MACRO_COUNT; resource_i = resource_i + 1)
                sram_macro_busy[resource_i] <= 32'd0;
            for (lane_i = 0; lane_i < PAYLOAD_LANE_COUNT; lane_i = lane_i + 1)
                payload_lane_busy[lane_i] <= 32'd0;
            active_resp_read <= 1'b0;
            active_resp_source <= 4'd0;
            active_resp_tid <= 8'd0;
            active_resp_rdata <= {DATA_WIDTH{1'b0}};
            resp0_valid <= 1'b0;
            resp1_valid <= 1'b0;
            resp0_read <= 1'b0;
            resp1_read <= 1'b0;
            resp0_source <= 4'd0;
            resp1_source <= 4'd0;
            resp0_tid <= 8'd0;
            resp1_tid <= 8'd0;
            resp0_rdata <= {DATA_WIDTH{1'b0}};
            resp1_rdata <= {DATA_WIDTH{1'b0}};
            debug_crc_busy <= 1'b0;
            debug_crc_done <= 1'b0;
            debug_crc <= FNV_OFFSET;
            debug_crc_byte_count <= 32'd0;
            debug_crc_scan_addr <= {ADDR_WIDTH{1'b0}};
            debug_crc_remaining <= 32'd0;
        end else begin
            for (resource_i = 0; resource_i < SRAM_MACRO_COUNT; resource_i = resource_i + 1) begin
                if (sram_macro_busy[resource_i] != 32'd0)
                    sram_macro_busy[resource_i] <= sram_macro_busy[resource_i] - 32'd1;
            end
            for (lane_i = 0; lane_i < PAYLOAD_LANE_COUNT; lane_i = lane_i + 1) begin
                if (payload_lane_busy[lane_i] != 32'd0)
                    payload_lane_busy[lane_i] <= payload_lane_busy[lane_i] - 32'd1;
            end

            if (phase_start_fire) begin
                sram_macro_busy[sram_macro_index] <= sram_macro_cycles;
                payload_lane_busy[selected_payload_lane] <=
                    max1(ceil_div(charged_bytes, STRB_WIDTH));
                active_resp_read <= !q0_write;
                active_resp_source <= q0_source;
                active_resp_tid <= q0_tid;
                if (q0_write) begin
                    active_resp_rdata <= {DATA_WIDTH{1'b0}};
                    if (sram_macro_word < SRAM_MACRO_WORDS) begin
                        for (sram_byte_i = 0; sram_byte_i < STRB_WIDTH; sram_byte_i = sram_byte_i + 1) begin
                            if (q0_wstrb[sram_byte_i])
                                sram_macro_mem[sram_macro_index][sram_macro_word][sram_byte_i*8 +: 8]
                                    <= q0_wdata[sram_byte_i*8 +: 8];
                        end
                    end
                end else begin
                    active_resp_rdata <= sram_read_word(q0_addr);
                end
            end

            if (phase_start_fire && req_push) begin
                if (q1_valid) begin
                    move_q1_to_q0();
                    load_q1(req_write, req_addr, req_bytes, route_cycles,
                            req_wdata, req_wstrb, req_source, req_tid);
                    q0_valid <= 1'b1;
                    q1_valid <= 1'b1;
                end else begin
                    load_q0(req_write, req_addr, req_bytes, route_cycles,
                            req_wdata, req_wstrb, req_source, req_tid);
                    q0_valid <= 1'b1;
                    q1_valid <= 1'b0;
                end
            end else if (phase_start_fire) begin
                if (q1_valid) begin
                    move_q1_to_q0();
                    q0_valid <= 1'b1;
                    q1_valid <= 1'b0;
                end else begin
                    q0_valid <= 1'b0;
                    q1_valid <= 1'b0;
                end
            end else if (req_push) begin
                if (!q0_valid) begin
                    load_q0(req_write, req_addr, req_bytes, route_cycles,
                            req_wdata, req_wstrb, req_source, req_tid);
                    q0_valid <= 1'b1;
                end else begin
                    load_q1(req_write, req_addr, req_bytes, route_cycles,
                            req_wdata, req_wstrb, req_source, req_tid);
                    q1_valid <= 1'b1;
                end
            end

            if (phase_done_push && resp_pop) begin
                if (resp1_valid) begin
                    resp0_valid <= 1'b1;
                    resp0_read <= resp1_read;
                    resp0_source <= resp1_source;
                    resp0_tid <= resp1_tid;
                    resp0_rdata <= resp1_rdata;
                    resp1_valid <= 1'b1;
                    resp1_read <= active_resp_read;
                    resp1_source <= active_resp_source;
                    resp1_tid <= active_resp_tid;
                    resp1_rdata <= active_resp_rdata;
                end else begin
                    resp0_valid <= 1'b1;
                    resp0_read <= active_resp_read;
                    resp0_source <= active_resp_source;
                    resp0_tid <= active_resp_tid;
                    resp0_rdata <= active_resp_rdata;
                    resp1_valid <= 1'b0;
                end
            end else if (phase_done_push) begin
                if (!resp0_valid) begin
                    resp0_valid <= 1'b1;
                    resp0_read <= active_resp_read;
                    resp0_source <= active_resp_source;
                    resp0_tid <= active_resp_tid;
                    resp0_rdata <= active_resp_rdata;
                end else begin
                    resp1_valid <= 1'b1;
                    resp1_read <= active_resp_read;
                    resp1_source <= active_resp_source;
                    resp1_tid <= active_resp_tid;
                    resp1_rdata <= active_resp_rdata;
                end
            end else if (resp_pop) begin
                if (resp1_valid) begin
                    resp0_valid <= 1'b1;
                    resp0_read <= resp1_read;
                    resp0_source <= resp1_source;
                    resp0_tid <= resp1_tid;
                    resp0_rdata <= resp1_rdata;
                    resp1_valid <= 1'b0;
                end else begin
                    resp0_valid <= 1'b0;
                end
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

    function [DATA_WIDTH-1:0] sram_read_word;
        input [ADDR_WIDTH-1:0] byte_addr;
        reg [31:0] dbg_word_addr;
        reg [7:0] dbg_sram_macro_index;
        reg [31:0] dbg_sram_macro_word;
        begin
            dbg_word_addr = {10'd0, byte_addr} / STRB_WIDTH;
            dbg_sram_macro_index = {dbg_word_addr[1:0],
                                    dbg_word_addr[5:2],
                                    dbg_word_addr[7:6]};
            dbg_sram_macro_word = dbg_word_addr >> 8;
            if (dbg_sram_macro_word >= SRAM_MACRO_WORDS) begin
                sram_read_word = {DATA_WIDTH{1'b0}};
            end else begin
                sram_read_word = sram_macro_mem[dbg_sram_macro_index][dbg_sram_macro_word];
            end
        end
    endfunction

    function [7:0] debug_read_byte;
        input [ADDR_WIDTH-1:0] byte_addr;
        reg [DATA_WIDTH-1:0] dbg_word_data;
        reg [3:0] dbg_lane;
        begin
            dbg_word_data = sram_read_word(byte_addr);
            dbg_lane = byte_addr[3:0];
            debug_read_byte = dbg_word_data[dbg_lane*8 +: 8];
        end
    endfunction

    assign resp_valid = resp0_valid;
    assign resp_read = resp0_read;
    assign resp_source = resp0_source;
    assign resp_tid = resp0_tid;
    assign resp_rdata = resp0_rdata;

    mdla7_synth_phase_engine #(
        .NUM_PHASES(6),
        .PHASE_W(4)
    ) u_phase (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(phase_start_valid),
        .start_ready(phase_start_ready),
        .phase_cycles(phase_cycles),
        .phase_ids(phase_ids),
        .phase_stall(1'b0),
        .busy(phase_busy),
        .done_valid(phase_done),
        .done_ready(phase_done_ready),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );
endmodule
