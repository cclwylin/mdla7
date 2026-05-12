`timescale 1ns/1ps

module Testbench_host_program;
    reg clk;
    reg rst_n;

    wire desc_valid;
    wire desc_ready;
    wire [3:0] desc_op_class;
    wire [31:0] bytes;
    wire [31:0] udma_dram_read_bytes;
    wire [31:0] udma_codec_cycles;
    wire udma_direction_write;
    wire [21:0] l1mesh_addr;
    wire [127:0] l1mesh_wdata;
    wire [15:0] l1mesh_wstrb;
    wire tnps_mode_space_to_depth;
    wire [15:0] tnps_in_h;
    wire [15:0] tnps_in_w;
    wire [15:0] tnps_in_c;
    wire [15:0] tnps_out_h;
    wire [15:0] tnps_out_w;
    wire [15:0] tnps_out_c;
    wire [15:0] tnps_block;
    wire [1:0] tnps_elem_bytes;
    wire [31:0] tnps_sample_out_elem_index;
    wire [31:0] tnps_sample_in_elem_index;
    wire [127:0] conv_act_vec;
    wire [127:0] conv_wgt_vec;
    wire [7:0] conv_elem_count;
    wire conv_fp_mode;
    wire conv_int16_mode;
    wire signed [15:0] conv_zp_in;
    wire signed [31:0] conv_bias;
    wire signed [31:0] conv_multiplier;
    wire signed [7:0] conv_shift;
    wire signed [31:0] conv_zp_out;
    wire signed [31:0] conv_act_min;
    wire signed [31:0] conv_act_max;
    wire [15:0] conv_in_h;
    wire [15:0] conv_in_w;
    wire [15:0] conv_in_c;
    wire [15:0] conv_out_h;
    wire [15:0] conv_out_w;
    wire [15:0] conv_out_c;
    wire [7:0] conv_k_h;
    wire [7:0] conv_k_w;
    wire [7:0] conv_stride_h;
    wire [7:0] conv_stride_w;
    wire [7:0] conv_dilation_h;
    wire [7:0] conv_dilation_w;
    wire signed [15:0] conv_pad_top;
    wire signed [15:0] conv_pad_left;
    wire [1:0] conv_elem_bytes;
    wire [31:0] conv_out_elem_index;
    wire [7:0] conv_tile_output_count;
    wire conv_partial_first;
    wire conv_partial_accumulate;
    wire [15:0] conv_sample_kh;
    wire [15:0] conv_sample_kw;
    wire [15:0] conv_sample_ic;
    wire signed [31:0] requant_input_value;
    wire pool_avg_mode;
    wire pool_fp_mode;
    wire pool_int16_mode;
    wire [127:0] pool_sample_vec;
    wire [7:0] pool_elem_count;
    wire [1:0] ewe_op_mode;
    wire ewe_fp_mode;
    wire ewe_int16_mode;
    wire [127:0] ewe_a_vec;
    wire [127:0] ewe_b_vec;
    wire [7:0] ewe_elem_count;
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
    wire test_done;
    wire test_fail;
    wire [31:0] issued_count;
    wire [31:0] done_count;
    wire top_done_ready_unused;
    integer watchdog;

    always #5 clk = ~clk;

    host_final u_host (
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
        .top_done_valid(done_valid),
        .top_done_ready(top_done_ready_unused),
        .top_busy(busy),
        .active_op_class(active_op_class),
        .active_phase_id(active_phase_id),
        .active_remaining_cycles(active_remaining_cycles),
        .placement_route_cycles(placement_route_cycles),
        .tnps_sample_src_byte_offset(tnps_sample_src_byte_offset),
        .tnps_sample_dst_byte_offset(tnps_sample_dst_byte_offset),
        .tnps_sample_valid(tnps_sample_valid),
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
        .conv_psum_valid_mask(conv_psum_valid_mask),
        .conv_psum_acc_values(conv_psum_acc_values),
        .requant_scaled_out(requant_scaled_out),
        .requant_out_q(requant_out_q),
        .pool_out(pool_out),
        .pool_out_q(pool_out_q),
        .pool_fp_bits(pool_fp_bits),
        .ewe_out(ewe_out),
        .ewe_out_q(ewe_out_q),
        .ewe_fp_bits(ewe_fp_bits),
        .block_busy(block_busy),
        .block_done_valid(block_done_valid),
        .test_done(test_done),
        .test_fail(test_fail),
        .issued_count(issued_count),
        .done_count(done_count)
    );

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

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        watchdog = 0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        while (!test_done && watchdog < 6000000) begin
            watchdog = watchdog + 1;
            @(posedge clk);
        end

        if (!test_done) begin
            $display("FAIL: verilog_final host program timeout issued=%0d done=%0d busy=%0d",
                     issued_count, done_count, busy);
        end else if (test_fail) begin
            $display("FAIL: verilog_final host program host reported failure issued=%0d done=%0d",
                     issued_count, done_count);
        end else if ((issued_count == 32'd0) || (issued_count != done_count)) begin
            $display("FAIL: verilog_final host program counts issued=%0d done=%0d",
                     issued_count, done_count);
        end else begin
            $display("PASS: verilog_final host-driven CONV/REQUANT/POOL/EWE/UDMA/TNPS program issued=%0d done=%0d",
                     issued_count, done_count);
        end
        $finish;
    end
endmodule
