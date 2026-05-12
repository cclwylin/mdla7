`timescale 1ns/1ps

module Testbench_top_byte_movers;
    localparam [3:0] OP_CONV = 4'd1;
    localparam [3:0] OP_REQUANT = 4'd2;
    localparam [3:0] OP_EWE = 4'd3;
    localparam [3:0] OP_POOL = 4'd4;
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
    reg conv_fp_mode;
    reg conv_int16_mode;
    reg signed [15:0] conv_zp_in;
    reg signed [31:0] conv_bias;
    reg signed [31:0] conv_multiplier;
    reg signed [7:0] conv_shift;
    reg signed [31:0] conv_zp_out;
    reg signed [31:0] conv_act_min;
    reg signed [31:0] conv_act_max;
    reg [15:0] conv_in_h;
    reg [15:0] conv_in_w;
    reg [15:0] conv_in_c;
    reg [15:0] conv_out_h;
    reg [15:0] conv_out_w;
    reg [15:0] conv_out_c;
    reg [7:0] conv_k_h;
    reg [7:0] conv_k_w;
    reg [7:0] conv_stride_h;
    reg [7:0] conv_stride_w;
    reg [7:0] conv_dilation_h;
    reg [7:0] conv_dilation_w;
    reg signed [15:0] conv_pad_top;
    reg signed [15:0] conv_pad_left;
    reg [1:0] conv_elem_bytes;
    reg [31:0] conv_out_elem_index;
    reg [7:0] conv_tile_output_count;
    reg conv_partial_first;
    reg conv_partial_accumulate;
    reg conv_partial_final;
    reg [15:0] conv_sample_kh;
    reg [15:0] conv_sample_kw;
    reg [15:0] conv_sample_ic;
    reg signed [31:0] requant_input_value;
    reg pool_avg_mode;
    reg pool_fp_mode;
    reg pool_int16_mode;
    reg [127:0] pool_sample_vec;
    reg [7:0] pool_elem_count;
    reg [1:0] ewe_op_mode;
    reg ewe_fp_mode;
    reg ewe_int16_mode;
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
    wire [63:0] conv_fp_sum_bits;
    wire signed [31:0] conv_int16_acc_out;
    wire [31:0] conv_sample_input_byte_offset;
    wire [31:0] conv_sample_weight_byte_offset;
    wire [31:0] conv_sample_output_byte_offset;
    wire conv_sample_input_valid;
    wire [31:0] conv_first_input_byte_offset;
    wire [31:0] conv_first_weight_byte_offset;
    wire [7:0] conv_window_valid_count;
    wire [31:0] conv_tile_last_output_byte_offset;
    wire conv_tile_last_input_valid;
    wire [7:0] conv_tile_last_window_valid_count;
    wire [3:0] conv_tile_scoreboard_valid_mask;
    wire signed [31:0] conv_tile_scoreboard_q_sum;
    wire [127:0] conv_tile_result_out_elem_indices;
    wire [127:0] conv_tile_result_output_byte_offsets;
    wire [127:0] conv_tile_result_acc_values;
    wire [127:0] conv_tile_result_q_values;
    wire [3:0] conv_writeback_valid_mask;
    wire [127:0] conv_writeback_output_byte_offsets;
    wire [127:0] conv_writeback_q_values;
    wire [3:0] conv_shadow_valid_mask;
    wire [127:0] conv_shadow_output_byte_offsets;
    wire [127:0] conv_shadow_q_values;
    wire [15:0] conv_shadow_mem_valid_mask;
    wire [511:0] conv_shadow_mem_output_byte_offsets;
    wire [511:0] conv_shadow_mem_q_values;
    wire conv_shadow_read_valid;
    wire [31:0] conv_shadow_read_output_byte_offset;
    wire [31:0] conv_shadow_read_q_value;
    wire [31:0] conv_shadow_crc;
    wire [31:0] conv_shadow_byte_count;
    wire [3:0] conv_psum_valid_mask;
    wire [127:0] conv_psum_acc_values;
    wire signed [31:0] requant_scaled_out;
    wire signed [7:0] requant_out_q;
    wire signed [31:0] pool_out;
    wire signed [7:0] pool_out_q;
    wire [63:0] pool_fp_bits;
    wire signed [31:0] ewe_out;
    wire signed [7:0] ewe_out_q;
    wire [63:0] ewe_fp_bits;
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
        .conv_fp_mode(conv_fp_mode),
        .conv_int16_mode(conv_int16_mode),
        .conv_zp_in(conv_zp_in),
        .conv_bias(conv_bias),
        .conv_multiplier(conv_multiplier),
        .conv_shift(conv_shift),
        .conv_zp_out(conv_zp_out),
        .conv_act_min(conv_act_min),
        .conv_act_max(conv_act_max),
        .conv_in_h(conv_in_h),
        .conv_in_w(conv_in_w),
        .conv_in_c(conv_in_c),
        .conv_out_h(conv_out_h),
        .conv_out_w(conv_out_w),
        .conv_out_c(conv_out_c),
        .conv_k_h(conv_k_h),
        .conv_k_w(conv_k_w),
        .conv_stride_h(conv_stride_h),
        .conv_stride_w(conv_stride_w),
        .conv_dilation_h(conv_dilation_h),
        .conv_dilation_w(conv_dilation_w),
        .conv_pad_top(conv_pad_top),
        .conv_pad_left(conv_pad_left),
        .conv_elem_bytes(conv_elem_bytes),
        .conv_out_elem_index(conv_out_elem_index),
        .conv_tile_output_count(conv_tile_output_count),
        .conv_partial_first(conv_partial_first),
        .conv_partial_accumulate(conv_partial_accumulate),
        .conv_partial_final(conv_partial_final),
        .conv_refcrc_mode(1'b0),
        .conv_refcrc_expected_crc(32'd0),
        .conv_refcrc_expected_count(32'd0),
        .conv_refcrc_ref_off(32'd0),
        .conv_sample_kh(conv_sample_kh),
        .conv_sample_kw(conv_sample_kw),
        .conv_sample_ic(conv_sample_ic),
        .requant_input_value(requant_input_value),
        .pool_avg_mode(pool_avg_mode),
        .pool_fp_mode(pool_fp_mode),
        .pool_int16_mode(pool_int16_mode),
        .pool_sample_vec(pool_sample_vec),
        .pool_elem_count(pool_elem_count),
        .ewe_op_mode(ewe_op_mode),
        .ewe_fp_mode(ewe_fp_mode),
        .ewe_int16_mode(ewe_int16_mode),
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
        .conv_fp_sum_bits(conv_fp_sum_bits),
        .conv_int16_acc_out(conv_int16_acc_out),
        .conv_sample_input_byte_offset(conv_sample_input_byte_offset),
        .conv_sample_weight_byte_offset(conv_sample_weight_byte_offset),
        .conv_sample_output_byte_offset(conv_sample_output_byte_offset),
        .conv_sample_input_valid(conv_sample_input_valid),
        .conv_first_input_byte_offset(conv_first_input_byte_offset),
        .conv_first_weight_byte_offset(conv_first_weight_byte_offset),
        .conv_window_valid_count(conv_window_valid_count),
        .conv_tile_last_output_byte_offset(conv_tile_last_output_byte_offset),
        .conv_tile_last_input_valid(conv_tile_last_input_valid),
        .conv_tile_last_window_valid_count(conv_tile_last_window_valid_count),
        .conv_tile_scoreboard_valid_mask(conv_tile_scoreboard_valid_mask),
        .conv_tile_scoreboard_q_sum(conv_tile_scoreboard_q_sum),
        .conv_tile_result_out_elem_indices(conv_tile_result_out_elem_indices),
        .conv_tile_result_output_byte_offsets(conv_tile_result_output_byte_offsets),
        .conv_tile_result_acc_values(conv_tile_result_acc_values),
        .conv_tile_result_q_values(conv_tile_result_q_values),
        .conv_writeback_valid_mask(conv_writeback_valid_mask),
        .conv_writeback_output_byte_offsets(conv_writeback_output_byte_offsets),
        .conv_writeback_q_values(conv_writeback_q_values),
        .conv_shadow_valid_mask(conv_shadow_valid_mask),
        .conv_shadow_output_byte_offsets(conv_shadow_output_byte_offsets),
        .conv_shadow_q_values(conv_shadow_q_values),
        .conv_shadow_mem_valid_mask(conv_shadow_mem_valid_mask),
        .conv_shadow_mem_output_byte_offsets(conv_shadow_mem_output_byte_offsets),
        .conv_shadow_mem_q_values(conv_shadow_mem_q_values),
        .conv_shadow_read_valid(conv_shadow_read_valid),
        .conv_shadow_read_output_byte_offset(conv_shadow_read_output_byte_offset),
        .conv_shadow_read_q_value(conv_shadow_read_q_value),
        .conv_shadow_crc(conv_shadow_crc),
        .conv_shadow_byte_count(conv_shadow_byte_count),
        .conv_psum_valid_mask(conv_psum_valid_mask),
        .conv_psum_acc_values(conv_psum_acc_values),
        .requant_scaled_out(requant_scaled_out),
        .requant_out_q(requant_out_q),
        .pool_out(pool_out),
        .pool_out_q(pool_out_q),
        .pool_fp_bits(pool_fp_bits),
        .ewe_out(ewe_out),
        .ewe_out_q(ewe_out_q),
        .ewe_fp_bits(ewe_fp_bits)
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
        conv_fp_mode = 1'b0;
        conv_int16_mode = 1'b0;
        conv_zp_in = 16'sd0;
        conv_bias = 32'sd0;
        conv_multiplier = 32'sd1073741824;
        conv_shift = 8'sd1;
        conv_zp_out = 32'sd0;
        conv_act_min = -32'sd128;
        conv_act_max = 32'sd127;
        conv_in_h = 16'd1;
        conv_in_w = 16'd1;
        conv_in_c = 16'd1;
        conv_out_h = 16'd1;
        conv_out_w = 16'd3;
        conv_out_c = 16'd1;
        conv_k_h = 8'd1;
        conv_k_w = 8'd1;
        conv_stride_h = 8'd1;
        conv_stride_w = 8'd1;
        conv_dilation_h = 8'd1;
        conv_dilation_w = 8'd1;
        conv_pad_top = 16'sd0;
        conv_pad_left = 16'sd0;
        conv_elem_bytes = 2'd1;
        conv_out_elem_index = 32'd0;
        conv_tile_output_count = 8'd1;
        conv_partial_first = 1'b0;
        conv_partial_accumulate = 1'b0;
        conv_partial_final = 1'b0;
        conv_sample_kh = 16'd0;
        conv_sample_kw = 16'd0;
        conv_sample_ic = 16'd0;
        requant_input_value = 32'sd0;
        pool_avg_mode = 1'b0;
        pool_fp_mode = 1'b0;
        pool_int16_mode = 1'b0;
        pool_sample_vec = 128'd0;
        pool_elem_count = 8'd0;
        ewe_op_mode = 2'd0;
        ewe_fp_mode = 1'b0;
        ewe_int16_mode = 1'b0;
        ewe_a_vec = 128'd0;
        ewe_b_vec = 128'd0;
        ewe_elem_count = 8'd0;
        failures = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        bytes = 32'd16;
        l1mesh_addr = 22'h0002a0;
        conv_act_vec = 128'd0;
        conv_wgt_vec = 128'd0;
        conv_act_vec[7:0] = 8'd4;
        conv_act_vec[15:8] = 8'd3;
        conv_act_vec[23:16] = 8'd2;
        conv_act_vec[31:24] = 8'd1;
        conv_act_vec[39:32] = -8'sd4;
        conv_act_vec[47:40] = 8'd7;
        conv_wgt_vec[7:0] = 8'd3;
        conv_wgt_vec[15:8] = -8'sd1;
        conv_wgt_vec[23:16] = 8'd1;
        conv_wgt_vec[31:24] = 8'd2;
        conv_wgt_vec[39:32] = 8'd5;
        conv_wgt_vec[47:40] = 8'd6;
        conv_elem_count = 8'd6;
        conv_fp_mode = 1'b0;
        conv_zp_in = 16'sd0;
        conv_bias = 32'sd5;
        conv_multiplier = 32'sd1073741824;
        conv_shift = 8'sd1;
        conv_zp_out = 32'sd0;
        conv_act_min = -32'sd128;
        conv_act_max = 32'sd127;
        conv_in_h = 16'd1;
        conv_in_w = 16'd6;
        conv_in_c = 16'd1;
        conv_out_h = 16'd1;
        conv_out_w = 16'd3;
        conv_out_c = 16'd1;
        conv_k_h = 8'd1;
        conv_k_w = 8'd6;
        conv_stride_h = 8'd1;
        conv_stride_w = 8'd1;
        conv_dilation_h = 8'd1;
        conv_dilation_w = 8'd1;
        conv_pad_top = 16'sd0;
        conv_pad_left = 16'sd0;
        conv_elem_bytes = 2'd1;
        conv_out_elem_index = 32'd0;
        conv_tile_output_count = 8'd3;
        conv_partial_first = 1'b1;
        conv_partial_accumulate = 1'b0;
        conv_sample_kh = 16'd0;
        conv_sample_kw = 16'd5;
        conv_sample_ic = 16'd0;
        issue_desc(OP_CONV);
        wait_done("conv");
        if ((conv_acc_out != 32'sd40) || (conv_out_q != 8'sd40)) begin
            $display("FAIL: CONV top sample acc=%0d out=%0d", conv_acc_out, conv_out_q);
            failures = failures + 1;
        end
        if (!conv_sample_input_valid ||
            (conv_sample_input_byte_offset != 32'd5) ||
            (conv_sample_weight_byte_offset != 32'd5) ||
            (conv_sample_output_byte_offset != 32'd0) ||
            (conv_first_input_byte_offset != 32'd0) ||
            (conv_first_weight_byte_offset != 32'd0) ||
            (conv_window_valid_count != 8'd6) ||
            (conv_tile_last_output_byte_offset != 32'd2) ||
            !conv_tile_last_input_valid ||
            (conv_tile_last_window_valid_count != 8'd4) ||
            (conv_tile_scoreboard_valid_mask != 4'b0111) ||
            (conv_tile_scoreboard_q_sum != 32'sd120) ||
            (conv_tile_result_out_elem_indices[31:0] != 32'd0) ||
            (conv_tile_result_out_elem_indices[95:64] != 32'd2) ||
            (conv_tile_result_output_byte_offsets[95:64] != 32'd2) ||
            ($signed(conv_tile_result_acc_values[95:64]) != 32'sd40) ||
            ($signed(conv_tile_result_q_values[95:64]) != 32'sd40)) begin
            $display("FAIL: CONV top 2D sample valid=%0d in=%0d wgt=%0d out=%0d first_in=%0d first_wgt=%0d valid_count=%0d tile_last_out=%0d tile_valid=%0d tile_count=%0d tile_mask=%04b tile_q_sum=%0d entry0_idx=%0d entry2_idx=%0d entry2_off=%0d entry2_acc=%0d entry2_q=%0d",
                     conv_sample_input_valid,
                     conv_sample_input_byte_offset,
                     conv_sample_weight_byte_offset,
                     conv_sample_output_byte_offset,
                     conv_first_input_byte_offset,
                     conv_first_weight_byte_offset,
                     conv_window_valid_count,
                     conv_tile_last_output_byte_offset,
                     conv_tile_last_input_valid,
                     conv_tile_last_window_valid_count,
                     conv_tile_scoreboard_valid_mask,
                     conv_tile_scoreboard_q_sum,
                     conv_tile_result_out_elem_indices[31:0],
                     conv_tile_result_out_elem_indices[95:64],
                     conv_tile_result_output_byte_offsets[95:64],
                     $signed(conv_tile_result_acc_values[95:64]),
                     $signed(conv_tile_result_q_values[95:64]));
            failures = failures + 1;
        end
        if ((conv_psum_valid_mask != 4'b0111) ||
            ($signed(conv_psum_acc_values[31:0]) != 32'sd40) ||
            ($signed(conv_psum_acc_values[95:64]) != 32'sd40)) begin
            $display("FAIL: CONV psum first mask=%04b psum0=%0d psum2=%0d",
                     conv_psum_valid_mask,
                     $signed(conv_psum_acc_values[31:0]),
                     $signed(conv_psum_acc_values[95:64]));
            failures = failures + 1;
        end

        conv_partial_first = 1'b0;
        conv_partial_accumulate = 1'b1;
        conv_partial_final = 1'b1;
        issue_desc(OP_CONV);
        wait_done("conv_psum_accumulate");
        if ((conv_psum_valid_mask != 4'b0111) ||
            ($signed(conv_psum_acc_values[31:0]) != 32'sd80) ||
            ($signed(conv_psum_acc_values[95:64]) != 32'sd80) ||
            ($signed(conv_tile_result_acc_values[31:0]) != 32'sd80) ||
            ($signed(conv_tile_result_acc_values[95:64]) != 32'sd80) ||
            (conv_tile_scoreboard_q_sum != 32'sd240) ||
            ($signed(conv_tile_result_q_values[31:0]) != 32'sd80) ||
            ($signed(conv_tile_result_q_values[95:64]) != 32'sd80) ||
            (conv_writeback_valid_mask != 4'b0111) ||
            (conv_writeback_output_byte_offsets[31:0] != 32'd0) ||
            (conv_writeback_output_byte_offsets[95:64] != 32'd2) ||
            ($signed(conv_writeback_q_values[31:0]) != 32'sd80) ||
            ($signed(conv_writeback_q_values[95:64]) != 32'sd80) ||
            (conv_shadow_valid_mask != 4'b0111) ||
            (conv_shadow_output_byte_offsets[31:0] != 32'd0) ||
            (conv_shadow_output_byte_offsets[95:64] != 32'd2) ||
            ($signed(conv_shadow_q_values[31:0]) != 32'sd80) ||
            ($signed(conv_shadow_q_values[95:64]) != 32'sd80) ||
            (conv_shadow_mem_valid_mask[0] != 1'b1) ||
            (conv_shadow_mem_valid_mask[2] != 1'b1) ||
            (conv_shadow_mem_output_byte_offsets[31:0] != 32'd0) ||
            (conv_shadow_mem_output_byte_offsets[95:64] != 32'd2) ||
            ($signed(conv_shadow_mem_q_values[31:0]) != 32'sd80) ||
            ($signed(conv_shadow_mem_q_values[95:64]) != 32'sd80)) begin
            $display("FAIL: CONV psum accumulate mask=%04b psum0=%0d psum2=%0d tile_q_sum=%0d wb_mask=%04b wb_off2=%0d wb_q2=%0d shadow_mask=%04b shadow_q2=%0d mem_mask=%04x mem_q2=%0d",
                     conv_psum_valid_mask,
                     $signed(conv_psum_acc_values[31:0]),
                     $signed(conv_psum_acc_values[95:64]),
                     conv_tile_scoreboard_q_sum,
                     conv_writeback_valid_mask,
                     conv_writeback_output_byte_offsets[95:64],
                     $signed(conv_writeback_q_values[95:64]),
                     conv_shadow_valid_mask,
                     $signed(conv_shadow_q_values[95:64]),
                     conv_shadow_mem_valid_mask,
                     $signed(conv_shadow_mem_q_values[95:64]));
            failures = failures + 1;
        end
        conv_partial_accumulate = 1'b0;
        conv_partial_final = 1'b0;
        conv_out_elem_index = 32'd2;
        issue_desc(OP_CONV);
        wait_done("conv_shadow_readback");
        if (!conv_shadow_read_valid ||
            (conv_shadow_read_output_byte_offset != 32'd2) ||
            ($signed(conv_shadow_read_q_value) != 32'sd80)) begin
            $display("FAIL: CONV shadow readback valid=%0d off=%0d q=%0d",
                     conv_shadow_read_valid,
                     conv_shadow_read_output_byte_offset,
                     $signed(conv_shadow_read_q_value));
            failures = failures + 1;
        end
        conv_out_elem_index = 32'd0;

        bytes = 32'd4;
        l1mesh_addr = 22'h0002a8;
        conv_act_vec = 128'd0;
        conv_wgt_vec = 128'd0;
        conv_act_vec[15:0] = 16'h3c00;  // 1.0
        conv_act_vec[31:16] = 16'hc000; // -2.0
        conv_wgt_vec[15:0] = 16'h4000;  // 2.0
        conv_wgt_vec[31:16] = 16'h3800; // 0.5
        conv_elem_count = 8'd2;
        conv_fp_mode = 1'b1;
        conv_int16_mode = 1'b0;
        issue_desc(OP_CONV);
        wait_done("conv_fp");
        if (conv_fp_sum_bits != 64'h3ff0000000000000) begin
            $display("FAIL: CONV FP top sample sum_bits=%016x", conv_fp_sum_bits);
            failures = failures + 1;
        end
        conv_fp_mode = 1'b0;

        bytes = 32'd8;
        l1mesh_addr = 22'h0002ac;
        conv_act_vec = 128'd0;
        conv_wgt_vec = 128'd0;
        conv_act_vec[15:0] = 16'sd4;
        conv_act_vec[31:16] = 16'sd3;
        conv_act_vec[47:32] = -16'sd2;
        conv_act_vec[63:48] = 16'sd1;
        conv_wgt_vec[15:0] = 16'sd3;
        conv_wgt_vec[31:16] = -16'sd1;
        conv_wgt_vec[47:32] = 16'sd5;
        conv_wgt_vec[63:48] = 16'sd2;
        conv_elem_count = 8'd4;
        conv_int16_mode = 1'b1;
        issue_desc(OP_CONV);
        wait_done("conv_int16");
        if (conv_int16_acc_out != 32'sd1) begin
            $display("FAIL: CONV INT16 top sample acc=%0d", conv_int16_acc_out);
            failures = failures + 1;
        end
        conv_int16_mode = 1'b0;

        bytes = 32'd1;
        l1mesh_addr = 22'h0002b0;
        requant_input_value = 32'sd40;
        issue_desc(OP_REQUANT);
        wait_done("requant");
        if (requant_out_q != 8'sd40) begin
            $display("FAIL: REQUANT top sample scaled=%0d out=%0d",
                     requant_scaled_out, requant_out_q);
            failures = failures + 1;
        end

        bytes = 32'd7;
        l1mesh_addr = 22'h0002c0;
        pool_avg_mode = 1'b0;
        pool_fp_mode = 1'b0;
        pool_sample_vec = 128'd0;
        pool_sample_vec[7:0] = 8'd4;
        pool_sample_vec[15:8] = 8'd3;
        pool_sample_vec[23:16] = 8'd2;
        pool_sample_vec[31:24] = 8'd1;
        pool_sample_vec[39:32] = -8'sd4;
        pool_sample_vec[47:40] = 8'd7;
        pool_sample_vec[55:48] = 8'd0;
        pool_elem_count = 8'd7;
        issue_desc(OP_POOL);
        wait_done("pool");
        if (pool_out_q != 8'sd7) begin
            $display("FAIL: POOL top sample out=%0d", pool_out);
            failures = failures + 1;
        end

        bytes = 32'd8;
        l1mesh_addr = 22'h0002c8;
        pool_avg_mode = 1'b1;
        pool_fp_mode = 1'b1;
        pool_int16_mode = 1'b0;
        pool_sample_vec = 128'd0;
        pool_sample_vec[15:0] = 16'h3c00;  // 1.0
        pool_sample_vec[31:16] = 16'h4000; // 2.0
        pool_sample_vec[47:32] = 16'h4200; // 3.0
        pool_sample_vec[63:48] = 16'h4400; // 4.0
        pool_elem_count = 8'd4;
        issue_desc(OP_POOL);
        wait_done("pool_fp");
        if (pool_fp_bits != 64'h4004000000000000) begin
            $display("FAIL: POOL FP top sample bits=%016x", pool_fp_bits);
            failures = failures + 1;
        end
        pool_fp_mode = 1'b0;

        bytes = 32'd8;
        l1mesh_addr = 22'h0002cc;
        pool_avg_mode = 1'b1;
        pool_fp_mode = 1'b0;
        pool_int16_mode = 1'b1;
        pool_sample_vec = 128'd0;
        pool_sample_vec[15:0] = -16'sd9;
        pool_sample_vec[31:16] = -16'sd1;
        pool_sample_vec[47:32] = 16'sd1;
        pool_sample_vec[63:48] = 16'sd5;
        pool_elem_count = 8'd4;
        issue_desc(OP_POOL);
        wait_done("pool_int16");
        if (pool_out != -32'sd1) begin
            $display("FAIL: POOL INT16 top sample out=%0d", pool_out);
            failures = failures + 1;
        end
        pool_int16_mode = 1'b0;

        bytes = 32'd4;
        l1mesh_addr = 22'h0002d0;
        ewe_op_mode = 2'd0;
        ewe_fp_mode = 1'b0;
        ewe_a_vec = 128'd0;
        ewe_b_vec = 128'd0;
        ewe_a_vec[7:0] = 8'd4;
        ewe_a_vec[15:8] = 8'd3;
        ewe_a_vec[23:16] = 8'd2;
        ewe_a_vec[31:24] = 8'd1;
        ewe_b_vec[7:0] = 8'd3;
        ewe_b_vec[15:8] = -8'sd1;
        ewe_b_vec[23:16] = 8'd1;
        ewe_b_vec[31:24] = 8'd2;
        ewe_elem_count = 8'd4;
        issue_desc(OP_EWE);
        wait_done("ewe");
        if ((ewe_out != 32'sd15) || (ewe_out_q != 8'sd7)) begin
            $display("FAIL: EWE top sample sum=%0d first=%0d", ewe_out, ewe_out_q);
            failures = failures + 1;
        end

        bytes = 32'd4;
        l1mesh_addr = 22'h0002d8;
        ewe_op_mode = 2'd1;
        ewe_fp_mode = 1'b1;
        ewe_int16_mode = 1'b0;
        ewe_a_vec = 128'd0;
        ewe_b_vec = 128'd0;
        ewe_a_vec[15:0] = 16'h3c00;  // 1.0
        ewe_a_vec[31:16] = 16'h4000; // 2.0
        ewe_b_vec[15:0] = 16'h4000;  // 2.0
        ewe_b_vec[31:16] = 16'h4200; // 3.0
        ewe_elem_count = 8'd2;
        issue_desc(OP_EWE);
        wait_done("ewe_fp");
        if (ewe_fp_bits != 64'h4020000000000000) begin
            $display("FAIL: EWE FP top sample bits=%016x", ewe_fp_bits);
            failures = failures + 1;
        end
        ewe_fp_mode = 1'b0;

        bytes = 32'd8;
        l1mesh_addr = 22'h0002da;
        ewe_op_mode = 2'd0;
        ewe_fp_mode = 1'b0;
        ewe_int16_mode = 1'b1;
        ewe_a_vec = 128'd0;
        ewe_b_vec = 128'd0;
        ewe_a_vec[15:0] = 16'sd4;
        ewe_a_vec[31:16] = 16'sd3;
        ewe_a_vec[47:32] = -16'sd2;
        ewe_a_vec[63:48] = 16'sd1;
        ewe_b_vec[15:0] = 16'sd3;
        ewe_b_vec[31:16] = -16'sd1;
        ewe_b_vec[47:32] = 16'sd5;
        ewe_b_vec[63:48] = 16'sd2;
        ewe_elem_count = 8'd4;
        issue_desc(OP_EWE);
        wait_done("ewe_int16");
        if (ewe_out != 32'sd15) begin
            $display("FAIL: EWE INT16 top sample sum=%0d", ewe_out);
            failures = failures + 1;
        end
        ewe_int16_mode = 1'b0;

        bytes = 32'd4;
        l1mesh_addr = 22'h0002dc;
        ewe_op_mode = 2'd3;
        ewe_fp_mode = 1'b1;
        ewe_a_vec = 128'd0;
        ewe_b_vec = 128'd0;
        ewe_a_vec[15:0] = 16'h3c00;  // 1.0
        ewe_a_vec[31:16] = 16'h0000; // 0.0
        ewe_elem_count = 8'd2;
        issue_desc(OP_EWE);
        wait_done("ewe_logistic_fp");
        if (ewe_fp_bits != 64'h3ff3b26a7aead15e) begin
            $display("FAIL: EWE logistic FP top sample bits=%016x", ewe_fp_bits);
            failures = failures + 1;
        end
        ewe_fp_mode = 1'b0;

        bytes = 32'd256;
        l1mesh_addr = 22'h0002e0;
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
            $display("PASS: verilog_final top integration CONV/REQUANT/POOL/EWE/UDMA/TNPS");
        else
            $display("FAIL: verilog_final top integration failures=%0d", failures);
        $finish;
    end
endmodule
