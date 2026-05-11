`timescale 1ns/1ps

`include "common.v"

module mdla7_top_final #(
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                       clk,
    input                       rst_n,

    input                       desc_valid,
    output                      desc_ready,
    input      [3:0]            desc_op_class,
    input      [31:0]           bytes,
    input      [31:0]           udma_dram_read_bytes,
    input      [31:0]           udma_codec_cycles,
    input                       udma_direction_write,
    input      [ADDR_WIDTH-1:0] l1mesh_addr,
    input      [DATA_WIDTH-1:0] l1mesh_wdata,
    input      [DATA_WIDTH/8-1:0] l1mesh_wstrb,

    input                       tnps_mode_space_to_depth,
    input      [15:0]           tnps_in_h,
    input      [15:0]           tnps_in_w,
    input      [15:0]           tnps_in_c,
    input      [15:0]           tnps_out_h,
    input      [15:0]           tnps_out_w,
    input      [15:0]           tnps_out_c,
    input      [15:0]           tnps_block,
    input      [1:0]            tnps_elem_bytes,
    input      [31:0]           tnps_sample_out_elem_index,
    input      [31:0]           tnps_sample_in_elem_index,

    output                      done_valid,
    input                       done_ready,
    output                      busy,
    output     [3:0]            active_op_class,
    output     [3:0]            active_phase_id,
    output     [31:0]           active_remaining_cycles,
    output     [31:0]           tnps_sample_src_byte_offset,
    output     [31:0]           tnps_sample_dst_byte_offset,
    output                      tnps_sample_valid,
    output     [31:0]           placement_route_cycles,
    output     [8:0]            block_busy,
    output     [8:0]            block_done_valid
);
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;

    localparam [2:0] ST_IDLE = 3'd0;
    localparam [2:0] ST_RUN  = 3'd1;
    localparam [2:0] ST_WAIT = 3'd2;
    localparam [2:0] ST_DONE = 3'd3;

    reg [2:0] state;
    reg done_valid_q;
    reg [3:0] op_class_q;
    reg start_pending;
    reg engine_done_seen;
    reg [31:0] bytes_q;
    reg [31:0] udma_dram_read_bytes_q;
    reg [31:0] udma_codec_cycles_q;
    reg udma_direction_write_q;
    reg [ADDR_WIDTH-1:0] l1mesh_addr_q;
    reg [DATA_WIDTH-1:0] l1mesh_wdata_q;
    reg [DATA_WIDTH/8-1:0] l1mesh_wstrb_q;
    reg tnps_mode_space_to_depth_q;
    reg [15:0] tnps_in_h_q;
    reg [15:0] tnps_in_w_q;
    reg [15:0] tnps_in_c_q;
    reg [15:0] tnps_out_h_q;
    reg [15:0] tnps_out_w_q;
    reg [15:0] tnps_out_c_q;
    reg [15:0] tnps_block_q;
    reg [1:0] tnps_elem_bytes_q;
    reg [31:0] tnps_sample_out_elem_index_q;
    reg [31:0] tnps_sample_in_elem_index_q;

    wire udma_start_ready;
    wire udma_busy;
    wire udma_done_valid;
    wire [3:0] udma_phase_id;
    wire [31:0] udma_remaining_cycles;
    wire udma_l1_req_valid;
    wire udma_l1_req_ready;
    wire udma_l1_req_write;
    wire [31:0] udma_l1_req_bytes;
    wire [31:0] udma_l1_req_payload_cycles;

    wire tnps_start_ready;
    wire tnps_busy;
    wire tnps_done_valid;
    wire [3:0] tnps_phase_id;
    wire [31:0] tnps_remaining_cycles;
    wire tnps_l1_req_valid;
    wire tnps_l1_req_ready;
    wire tnps_l1_req_write;
    wire [31:0] tnps_l1_req_bytes;
    wire [31:0] tnps_l1_req_payload_cycles;

    wire l1mgr_busy;
    wire l1mgr_resp_valid;
    wire l1mgr_resp_ready;
    wire [3:0] l1mgr_phase_id;
    wire [31:0] l1mgr_remaining_cycles;
    wire l1mgr_mesh_req_write;
    wire [ADDR_WIDTH-1:0] l1mgr_mesh_req_addr;
    wire [31:0] l1mgr_mesh_req_bytes;
    wire [DATA_WIDTH-1:0] l1mgr_mesh_req_wdata;
    wire [DATA_WIDTH/8-1:0] l1mgr_mesh_req_wstrb;

    wire l1mesh_req_ready;
    wire l1mesh_busy;
    wire l1mesh_resp_valid;
    wire [3:0] l1mesh_phase_id;
    wire [31:0] l1mesh_remaining_cycles;
    wire [DATA_WIDTH-1:0] l1mesh_rdata;
    wire [1:0] route_source_x;
    wire [1:0] route_source_y;
    wire [1:0] route_tile_x;
    wire [1:0] route_tile_y;
    wire [1:0] route_bank_x;
    wire [1:0] route_bank_y;
    wire legacy_req_ready;
    wire requant_req_ready_unused;
    wire ewe_req_ready_unused;
    wire pool_req_ready_unused;
    wire [3:0] l1mgr_debug_source_unused;
    wire [7:0] l1mgr_debug_tid_unused;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [155:0] final_debug_unused = {
        l1mesh_rdata,
        route_source_x,
        route_source_y,
        route_tile_x,
        route_tile_y,
        route_bank_x,
        route_bank_y,
        legacy_req_ready,
        requant_req_ready_unused,
        ewe_req_ready_unused,
        pool_req_ready_unused,
        l1mgr_debug_source_unused,
        l1mgr_debug_tid_unused
    };
    /* verilator lint_on UNUSEDSIGNAL */

    wire run_udma = (op_class_q == OP_UDMA);
    wire run_tnps = (op_class_q == OP_TNPS);
    wire selected_start_ready = run_udma ? udma_start_ready :
                                run_tnps ? tnps_start_ready : 1'b1;
    wire selected_done_valid = run_udma ? udma_done_valid :
                               run_tnps ? tnps_done_valid : 1'b1;
    wire selected_busy = run_udma ? udma_busy :
                         run_tnps ? tnps_busy : 1'b0;
    wire [3:0] selected_phase = run_udma ? udma_phase_id :
                                run_tnps ? tnps_phase_id : 4'd0;
    wire [31:0] selected_remaining = run_udma ? udma_remaining_cycles :
                                     run_tnps ? tnps_remaining_cycles : 32'd0;
    wire l1_drained = !l1mgr_busy && !l1mgr_resp_valid && !l1mesh_busy && !l1mesh_resp_valid;
    wire udma_start = start_pending && run_udma;
    wire tnps_start = start_pending && run_tnps;

    assign desc_ready = (state == ST_IDLE);
    assign done_valid = done_valid_q;
    assign busy = (state != ST_IDLE);
    assign active_op_class = op_class_q;
    assign active_phase_id = selected_busy ? selected_phase :
                             l1mgr_busy ? l1mgr_phase_id :
                             l1mesh_busy ? l1mesh_phase_id : 4'd0;
    assign active_remaining_cycles = selected_busy ? selected_remaining :
                                     l1mgr_busy ? l1mgr_remaining_cycles :
                                     l1mesh_busy ? l1mesh_remaining_cycles : 32'd0;
    assign l1mgr_resp_ready = l1mesh_req_ready;
    assign block_busy = {l1mesh_busy, l1mgr_busy, udma_busy, tnps_busy, 5'd0};
    assign block_done_valid = {l1mesh_resp_valid, l1mgr_resp_valid, udma_done_valid, tnps_done_valid, 5'd0};

    vf_l1mesh_route_estimator u_route (
        .source_id(op_class_q),
        .addr(l1mesh_addr_q),
        .route_cycles(placement_route_cycles),
        .source_x(route_source_x),
        .source_y(route_source_y),
        .tile_x(route_tile_x),
        .tile_y(route_tile_y),
        .bank_x(route_bank_x),
        .bank_y(route_bank_y)
    );

    vf_udma_engine u_udma (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(udma_start),
        .start_ready(udma_start_ready),
        .direction_write(udma_direction_write_q),
        .bytes(bytes_q),
        .dram_read_bytes(udma_dram_read_bytes_q),
        .codec_cycles(udma_codec_cycles_q),
        .l1_req_valid(udma_l1_req_valid),
        .l1_req_ready(udma_l1_req_ready),
        .l1_req_write(udma_l1_req_write),
        .l1_req_bytes(udma_l1_req_bytes),
        .l1_req_payload_cycles(udma_l1_req_payload_cycles),
        .busy(udma_busy),
        .done_valid(udma_done_valid),
        .done_ready(1'b1),
        .phase_id(udma_phase_id),
        .remaining_cycles(udma_remaining_cycles)
    );

    vf_tnps_engine u_tnps (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(tnps_start),
        .start_ready(tnps_start_ready),
        .bytes(bytes_q),
        .mode_space_to_depth(tnps_mode_space_to_depth_q),
        .in_h(tnps_in_h_q),
        .in_w(tnps_in_w_q),
        .in_c(tnps_in_c_q),
        .out_h(tnps_out_h_q),
        .out_w(tnps_out_w_q),
        .out_c(tnps_out_c_q),
        .block(tnps_block_q),
        .elem_bytes(tnps_elem_bytes_q),
        .sample_out_elem_index(tnps_sample_out_elem_index_q),
        .sample_in_elem_index(tnps_sample_in_elem_index_q),
        .l1_req_valid(tnps_l1_req_valid),
        .l1_req_ready(tnps_l1_req_ready),
        .l1_req_write(tnps_l1_req_write),
        .l1_req_bytes(tnps_l1_req_bytes),
        .l1_req_payload_cycles(tnps_l1_req_payload_cycles),
        .busy(tnps_busy),
        .done_valid(tnps_done_valid),
        .done_ready(1'b1),
        .phase_id(tnps_phase_id),
        .remaining_cycles(tnps_remaining_cycles),
        .sample_src_byte_offset(tnps_sample_src_byte_offset),
        .sample_dst_byte_offset(tnps_sample_dst_byte_offset),
        .sample_valid(tnps_sample_valid)
    );

    l1manager u_l1manager (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(1'b0),
        .req_ready(legacy_req_ready),
        .req_write(1'b0),
        .req_l1(1'b1),
        .req_source(4'd0),
        .req_tid(8'd0),
        .req_bytes(32'd0),
        .req_payload_cycles(32'd0),
        .req_addr({ADDR_WIDTH{1'b0}}),
        .req_wdata({DATA_WIDTH{1'b0}}),
        .req_wstrb({DATA_WIDTH/8{1'b0}}),
        .udma_req_valid(udma_l1_req_valid),
        .udma_req_ready(udma_l1_req_ready),
        .udma_req_write(udma_l1_req_write),
        .udma_req_tid(8'd1),
        .udma_req_bytes(udma_l1_req_bytes),
        .udma_req_payload_cycles(udma_l1_req_payload_cycles),
        .udma_req_addr(l1mesh_addr_q),
        .udma_req_wdata(l1mesh_wdata_q),
        .udma_req_wstrb(l1mesh_wstrb_q),
        .requant_req_valid(1'b0),
        .requant_req_ready(requant_req_ready_unused),
        .requant_req_write(1'b0),
        .requant_req_tid(8'd0),
        .requant_req_bytes(32'd0),
        .requant_req_payload_cycles(32'd0),
        .requant_req_addr({ADDR_WIDTH{1'b0}}),
        .requant_req_wdata({DATA_WIDTH{1'b0}}),
        .requant_req_wstrb({DATA_WIDTH/8{1'b0}}),
        .ewe_req_valid(1'b0),
        .ewe_req_ready(ewe_req_ready_unused),
        .ewe_req_write(1'b0),
        .ewe_req_tid(8'd0),
        .ewe_req_bytes(32'd0),
        .ewe_req_payload_cycles(32'd0),
        .ewe_req_addr({ADDR_WIDTH{1'b0}}),
        .ewe_req_wdata({DATA_WIDTH{1'b0}}),
        .ewe_req_wstrb({DATA_WIDTH/8{1'b0}}),
        .pool_req_valid(1'b0),
        .pool_req_ready(pool_req_ready_unused),
        .pool_req_write(1'b0),
        .pool_req_tid(8'd0),
        .pool_req_bytes(32'd0),
        .pool_req_payload_cycles(32'd0),
        .pool_req_addr({ADDR_WIDTH{1'b0}}),
        .pool_req_wdata({DATA_WIDTH{1'b0}}),
        .pool_req_wstrb({DATA_WIDTH/8{1'b0}}),
        .tnps_req_valid(tnps_l1_req_valid),
        .tnps_req_ready(tnps_l1_req_ready),
        .tnps_req_write(tnps_l1_req_write),
        .tnps_req_tid(8'd2),
        .tnps_req_bytes(tnps_l1_req_bytes),
        .tnps_req_payload_cycles(tnps_l1_req_payload_cycles),
        .tnps_req_addr(l1mesh_addr_q),
        .tnps_req_wdata(l1mesh_wdata_q),
        .tnps_req_wstrb(l1mesh_wstrb_q),
        .mesh_req_write(l1mgr_mesh_req_write),
        .mesh_req_addr(l1mgr_mesh_req_addr),
        .mesh_req_bytes(l1mgr_mesh_req_bytes),
        .mesh_req_wdata(l1mgr_mesh_req_wdata),
        .mesh_req_wstrb(l1mgr_mesh_req_wstrb),
        .resp_valid(l1mgr_resp_valid),
        .resp_ready(l1mgr_resp_ready),
        .busy(l1mgr_busy),
        .phase_id(l1mgr_phase_id),
        .remaining_cycles(l1mgr_remaining_cycles),
        .debug_source(l1mgr_debug_source_unused),
        .debug_tid(l1mgr_debug_tid_unused)
    );

    l1mesh u_l1mesh (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(l1mgr_resp_valid),
        .req_ready(l1mesh_req_ready),
        .req_write(l1mgr_mesh_req_write),
        .req_addr(l1mgr_mesh_req_addr),
        .req_bytes(l1mgr_mesh_req_bytes),
        .route_cycles(placement_route_cycles),
        .req_wdata(l1mgr_mesh_req_wdata),
        .req_wstrb(l1mgr_mesh_req_wstrb),
        .resp_valid(l1mesh_resp_valid),
        .resp_ready(1'b1),
        .resp_rdata(l1mesh_rdata),
        .busy(l1mesh_busy),
        .phase_id(l1mesh_phase_id),
        .remaining_cycles(l1mesh_remaining_cycles)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done_valid_q <= 1'b0;
            op_class_q <= 4'd0;
            start_pending <= 1'b0;
            engine_done_seen <= 1'b0;
            bytes_q <= 32'd0;
            udma_dram_read_bytes_q <= 32'd0;
            udma_codec_cycles_q <= 32'd0;
            udma_direction_write_q <= 1'b0;
            l1mesh_addr_q <= {ADDR_WIDTH{1'b0}};
            l1mesh_wdata_q <= {DATA_WIDTH{1'b0}};
            l1mesh_wstrb_q <= {DATA_WIDTH/8{1'b0}};
            tnps_mode_space_to_depth_q <= 1'b1;
            tnps_in_h_q <= 16'd0;
            tnps_in_w_q <= 16'd0;
            tnps_in_c_q <= 16'd0;
            tnps_out_h_q <= 16'd0;
            tnps_out_w_q <= 16'd0;
            tnps_out_c_q <= 16'd0;
            tnps_block_q <= 16'd0;
            tnps_elem_bytes_q <= 2'd1;
            tnps_sample_out_elem_index_q <= 32'd0;
            tnps_sample_in_elem_index_q <= 32'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done_valid_q <= 1'b0;
                    if (desc_valid && desc_ready) begin
                        op_class_q <= desc_op_class;
                        bytes_q <= bytes;
                        udma_dram_read_bytes_q <= udma_dram_read_bytes;
                        udma_codec_cycles_q <= udma_codec_cycles;
                        udma_direction_write_q <= udma_direction_write;
                        l1mesh_addr_q <= l1mesh_addr;
                        l1mesh_wdata_q <= l1mesh_wdata;
                        l1mesh_wstrb_q <= l1mesh_wstrb;
                        tnps_mode_space_to_depth_q <= tnps_mode_space_to_depth;
                        tnps_in_h_q <= tnps_in_h;
                        tnps_in_w_q <= tnps_in_w;
                        tnps_in_c_q <= tnps_in_c;
                        tnps_out_h_q <= tnps_out_h;
                        tnps_out_w_q <= tnps_out_w;
                        tnps_out_c_q <= tnps_out_c;
                        tnps_block_q <= tnps_block;
                        tnps_elem_bytes_q <= tnps_elem_bytes;
                        tnps_sample_out_elem_index_q <= tnps_sample_out_elem_index;
                        tnps_sample_in_elem_index_q <= tnps_sample_in_elem_index;
                        start_pending <= 1'b1;
                        engine_done_seen <= 1'b0;
                        state <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (start_pending && selected_start_ready)
                        start_pending <= 1'b0;
                    if (selected_done_valid)
                        engine_done_seen <= 1'b1;
                    if (!start_pending && (engine_done_seen || selected_done_valid))
                        state <= ST_WAIT;
                end
                ST_WAIT: begin
                    if (l1_drained) begin
                        done_valid_q <= 1'b1;
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (done_valid_q && done_ready) begin
                        done_valid_q <= 1'b0;
                        state <= ST_IDLE;
                    end
                end
                default: begin
                    state <= ST_IDLE;
                    done_valid_q <= 1'b0;
                    start_pending <= 1'b0;
                    engine_done_seen <= 1'b0;
                end
            endcase
        end
    end
endmodule
