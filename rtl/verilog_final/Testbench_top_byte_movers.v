`timescale 1ns/1ps

module Testbench_top_byte_movers;
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;

    reg clk;
    reg rst_n;
    reg desc_valid;
    wire desc_ready;
    reg [3:0] desc_op_class;
    reg [31:0] bytes;
    reg [31:0] udma_dram_read_bytes;
    reg [31:0] udma_codec_cycles;
    reg udma_direction_write;
    reg [21:0] l1mesh_addr;
    reg [127:0] l1mesh_wdata;
    reg [15:0] l1mesh_wstrb;
    reg tnps_mode_space_to_depth;
    reg [15:0] tnps_in_h;
    reg [15:0] tnps_in_w;
    reg [15:0] tnps_in_c;
    reg [15:0] tnps_out_h;
    reg [15:0] tnps_out_w;
    reg [15:0] tnps_out_c;
    reg [15:0] tnps_block;
    reg [1:0] tnps_elem_bytes;
    reg [31:0] tnps_sample_out_elem_index;
    reg [31:0] tnps_sample_in_elem_index;
    reg [127:0] conv_act_vec;
    reg [127:0] conv_wgt_vec;
    reg [7:0] conv_elem_count;
    reg signed [15:0] conv_zp_in;
    reg signed [31:0] conv_bias;
    reg signed [31:0] conv_multiplier;
    reg signed [7:0] conv_shift;
    reg signed [31:0] conv_zp_out;
    reg signed [31:0] conv_act_min;
    reg signed [31:0] conv_act_max;
    reg signed [31:0] requant_input_value;
    reg pool_avg_mode;
    reg [127:0] pool_sample_vec;
    reg [7:0] pool_elem_count;
    reg [1:0] ewe_op_mode;
    reg [127:0] ewe_a_vec;
    reg [127:0] ewe_b_vec;
    reg [7:0] ewe_elem_count;
    wire done_valid;
    wire busy;
    wire [3:0] active_op_class;
    wire [3:0] active_phase_id;
    wire [31:0] active_remaining_cycles;
    wire [31:0] tnps_sample_src_byte_offset;
    wire [31:0] tnps_sample_dst_byte_offset;
    wire tnps_sample_valid;
    wire signed [31:0] conv_acc_out;
    wire signed [31:0] conv_scaled_out;
    wire signed [7:0] conv_out_q;
    wire signed [31:0] requant_scaled_out;
    wire signed [7:0] requant_out_q;
    wire signed [31:0] pool_out;
    wire signed [7:0] pool_out_q;
    wire signed [31:0] ewe_out;
    wire signed [7:0] ewe_out_q;
    wire [31:0] placement_route_cycles;
    wire [8:0] block_busy;
    wire [8:0] block_done_valid;
    integer failures;
    integer watchdog;

    always #5 clk = ~clk;

    mdla7_top_final u_top (
        .clk(clk),
        .rst_n(rst_n),
        .desc_valid(desc_valid),
        .desc_ready(desc_ready),
        .desc_op_class(desc_op_class),
        .bytes(bytes),
        .udma_dram_read_bytes(udma_dram_read_bytes),
        .udma_codec_cycles(udma_codec_cycles),
        .udma_direction_write(udma_direction_write),
        .l1mesh_addr(l1mesh_addr),
        .l1mesh_wdata(l1mesh_wdata),
        .l1mesh_wstrb(l1mesh_wstrb),
        .tnps_mode_space_to_depth(tnps_mode_space_to_depth),
        .tnps_in_h(tnps_in_h),
        .tnps_in_w(tnps_in_w),
        .tnps_in_c(tnps_in_c),
        .tnps_out_h(tnps_out_h),
        .tnps_out_w(tnps_out_w),
        .tnps_out_c(tnps_out_c),
        .tnps_block(tnps_block),
        .tnps_elem_bytes(tnps_elem_bytes),
        .tnps_sample_out_elem_index(tnps_sample_out_elem_index),
        .tnps_sample_in_elem_index(tnps_sample_in_elem_index),
        .conv_act_vec(conv_act_vec),
        .conv_wgt_vec(conv_wgt_vec),
        .conv_elem_count(conv_elem_count),
        .conv_zp_in(conv_zp_in),
        .conv_bias(conv_bias),
        .conv_multiplier(conv_multiplier),
        .conv_shift(conv_shift),
        .conv_zp_out(conv_zp_out),
        .conv_act_min(conv_act_min),
        .conv_act_max(conv_act_max),
        .requant_input_value(requant_input_value),
        .pool_avg_mode(pool_avg_mode),
        .pool_sample_vec(pool_sample_vec),
        .pool_elem_count(pool_elem_count),
        .ewe_op_mode(ewe_op_mode),
        .ewe_a_vec(ewe_a_vec),
        .ewe_b_vec(ewe_b_vec),
        .ewe_elem_count(ewe_elem_count),
        .done_valid(done_valid),
        .done_ready(1'b1),
        .busy(busy),
        .active_op_class(active_op_class),
        .active_phase_id(active_phase_id),
        .active_remaining_cycles(active_remaining_cycles),
        .tnps_sample_src_byte_offset(tnps_sample_src_byte_offset),
        .tnps_sample_dst_byte_offset(tnps_sample_dst_byte_offset),
        .tnps_sample_valid(tnps_sample_valid),
        .placement_route_cycles(placement_route_cycles),
        .block_busy(block_busy),
        .block_done_valid(block_done_valid),
        .conv_acc_out(conv_acc_out),
        .conv_scaled_out(conv_scaled_out),
        .conv_out_q(conv_out_q),
        .requant_scaled_out(requant_scaled_out),
        .requant_out_q(requant_out_q),
        .pool_out(pool_out),
        .pool_out_q(pool_out_q),
        .ewe_out(ewe_out),
        .ewe_out_q(ewe_out_q)
    );

    task issue_desc;
        input [3:0] op_class;
        begin
            @(posedge clk);
            while (!desc_ready)
                @(posedge clk);
            desc_op_class = op_class;
            desc_valid = 1'b1;
            @(posedge clk);
            desc_valid = 1'b0;
        end
    endtask

    task wait_done;
        input [255:0] label;
        begin
            watchdog = 0;
            while (!done_valid && watchdog < 1000) begin
                watchdog = watchdog + 1;
                @(posedge clk);
            end
            if (!done_valid) begin
                $display("FAIL: timeout waiting for %0s active_op=%0d phase=%0d remain=%0d busy=%0d block_busy=%09b block_done=%09b",
                         label, active_op_class, active_phase_id,
                         active_remaining_cycles, busy, block_busy, block_done_valid);
                failures = failures + 1;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        desc_valid = 1'b0;
        desc_op_class = 4'd0;
        bytes = 32'd256;
        udma_dram_read_bytes = 32'd512;
        udma_codec_cycles = 32'd3;
        udma_direction_write = 1'b0;
        l1mesh_addr = 22'h0002a0;
        l1mesh_wdata = 128'h0123456789abcdef0011223344556677;
        l1mesh_wstrb = 16'hffff;
        tnps_mode_space_to_depth = 1'b1;
        tnps_in_h = 16'd4;
        tnps_in_w = 16'd4;
        tnps_in_c = 16'd1;
        tnps_out_h = 16'd2;
        tnps_out_w = 16'd2;
        tnps_out_c = 16'd4;
        tnps_block = 16'd2;
        tnps_elem_bytes = 2'd1;
        tnps_sample_out_elem_index = 32'd2;
        tnps_sample_in_elem_index = 32'd0;
        conv_act_vec = 128'd0;
        conv_wgt_vec = 128'd0;
        conv_elem_count = 8'd0;
        conv_zp_in = 16'sd0;
        conv_bias = 32'sd0;
        conv_multiplier = 32'sd1073741824;
        conv_shift = 8'sd1;
        conv_zp_out = 32'sd0;
        conv_act_min = -32'sd128;
        conv_act_max = 32'sd127;
        requant_input_value = 32'sd0;
        pool_avg_mode = 1'b0;
        pool_sample_vec = 128'd0;
        pool_elem_count = 8'd0;
        ewe_op_mode = 2'd0;
        ewe_a_vec = 128'd0;
        ewe_b_vec = 128'd0;
        ewe_elem_count = 8'd0;
        failures = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        issue_desc(OP_UDMA);
        wait_done("udma");
        if (placement_route_cycles == 32'd0) begin
            $display("FAIL: UDMA placement_route_cycles was zero");
            failures = failures + 1;
        end

        bytes = 32'd128;
        l1mesh_addr = 22'h0003f0;
        issue_desc(OP_TNPS);
        wait_done("tnps");
        if (!tnps_sample_valid ||
            (tnps_sample_src_byte_offset != 32'd4) ||
            (tnps_sample_dst_byte_offset != 32'd2)) begin
            $display("FAIL: TNPS sample mapping valid=%0d src=%0d dst=%0d",
                     tnps_sample_valid,
                     tnps_sample_src_byte_offset,
                     tnps_sample_dst_byte_offset);
            failures = failures + 1;
        end

        if (failures == 0)
            $display("PASS: verilog_final top byte movers UDMA/TNPS");
        else
            $display("FAIL: verilog_final top byte movers failures=%0d", failures);
        $finish;
    end
endmodule
