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
    wire signed [15:0] conv_zp_in;
    wire signed [31:0] conv_bias;
    wire signed [31:0] conv_multiplier;
    wire signed [7:0] conv_shift;
    wire signed [31:0] conv_zp_out;
    wire signed [31:0] conv_act_min;
    wire signed [31:0] conv_act_max;
    wire signed [31:0] requant_input_value;
    wire pool_avg_mode;
    wire [127:0] pool_sample_vec;
    wire [7:0] pool_elem_count;
    wire [1:0] ewe_op_mode;
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
    wire signed [31:0] requant_scaled_out;
    wire signed [7:0] requant_out_q;
    wire signed [31:0] pool_out;
    wire signed [7:0] pool_out_q;
    wire signed [31:0] ewe_out;
    wire signed [7:0] ewe_out_q;
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
        .requant_scaled_out(requant_scaled_out),
        .requant_out_q(requant_out_q),
        .pool_out(pool_out),
        .pool_out_q(pool_out_q),
        .ewe_out(ewe_out),
        .ewe_out_q(ewe_out_q),
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
