`timescale 1ns/1ps

module Testbench_conv_datapath;
    reg [16*8-1:0] act_vec;
    reg [16*8-1:0] wgt_vec;
    reg [7:0] elem_count;
    reg signed [15:0] zp_in;
    reg signed [31:0] bias;
    reg signed [31:0] multiplier;
    reg signed [7:0] shift;
    reg signed [31:0] zp_out;
    reg signed [31:0] act_min;
    reg signed [31:0] act_max;
    wire signed [31:0] acc_out;
    wire signed [31:0] scaled_out;
    wire signed [7:0] out_q;
    reg [15:0] in_h;
    reg [15:0] in_w;
    reg [15:0] in_c;
    reg [15:0] out_h;
    reg [15:0] out_w;
    reg [15:0] out_c;
    reg [7:0] k_h;
    reg [7:0] k_w;
    reg [7:0] stride_h;
    reg [7:0] stride_w;
    reg [7:0] dilation_h;
    reg [7:0] dilation_w;
    reg signed [15:0] pad_top;
    reg signed [15:0] pad_left;
    reg [1:0] elem_bytes;
    reg [31:0] out_elem_index;
    reg [7:0] tile_output_count;
    reg [15:0] sample_kh;
    reg [15:0] sample_kw;
    reg [15:0] sample_ic;
    wire [31:0] input_byte_offset;
    wire [31:0] weight_byte_offset;
    wire [31:0] output_byte_offset;
    wire input_valid;
    reg clk;
    reg rst_n;
    wire engine_start_ready;
    wire engine_l1_req_valid;
    wire engine_l1_req_write;
    wire [31:0] engine_l1_req_bytes;
    wire [31:0] engine_l1_req_payload_cycles;
    wire engine_busy;
    wire engine_done_valid;
    wire [3:0] engine_phase_id;
    wire [31:0] engine_remaining_cycles;
    wire signed [31:0] engine_acc_out;
    wire signed [31:0] engine_scaled_out;
    wire signed [7:0] engine_out_q;
    wire [63:0] engine_fp_sum_bits;
    wire signed [31:0] engine_int16_acc_out;
    wire [31:0] engine_input_byte_offset;
    wire [31:0] engine_weight_byte_offset;
    wire [31:0] engine_output_byte_offset;
    wire engine_input_valid;
    wire [31:0] engine_first_input_byte_offset;
    wire [31:0] engine_first_weight_byte_offset;
    wire [7:0] engine_window_valid_count;
    wire [31:0] engine_tile_last_output_byte_offset;
    wire engine_tile_last_input_valid;
    wire [7:0] engine_tile_last_window_valid_count;
    wire [3:0] engine_tile_scoreboard_valid_mask;
    wire signed [31:0] engine_tile_scoreboard_q_sum;
    wire [127:0] engine_tile_result_out_elem_indices;
    wire [127:0] engine_tile_result_output_byte_offsets;
    wire [127:0] engine_tile_result_acc_values;
    wire [127:0] engine_tile_result_q_values;
    integer failures;

    always #5 clk = ~clk;

    vf_conv_int8_mac #(
        .MAX_ELEMS(16)
    ) u_mac (
        .act_vec(act_vec),
        .wgt_vec(wgt_vec),
        .elem_count(elem_count),
        .zp_in(zp_in),
        .bias(bias),
        .multiplier(multiplier),
        .shift(shift),
        .zp_out(zp_out),
        .act_min(act_min),
        .act_max(act_max),
        .acc_out(acc_out),
        .scaled_out(scaled_out),
        .out_q(out_q)
    );

    vf_conv2d_addrgen u_addrgen (
        .in_h(in_h),
        .in_w(in_w),
        .in_c(in_c),
        .out_h(out_h),
        .out_w(out_w),
        .out_c(out_c),
        .k_h(k_h),
        .k_w(k_w),
        .stride_h(stride_h),
        .stride_w(stride_w),
        .dilation_h(dilation_h),
        .dilation_w(dilation_w),
        .pad_top(pad_top),
        .pad_left(pad_left),
        .elem_bytes(elem_bytes),
        .out_elem_index(out_elem_index),
        .sample_kh(sample_kh),
        .sample_kw(sample_kw),
        .sample_ic(sample_ic),
        .input_byte_offset(input_byte_offset),
        .weight_byte_offset(weight_byte_offset),
        .output_byte_offset(output_byte_offset),
        .input_valid(input_valid)
    );

    vf_conv_sample_engine u_sample_engine (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(1'b0),
        .start_ready(engine_start_ready),
        .act_vec(act_vec),
        .wgt_vec(wgt_vec),
        .elem_count(elem_count),
        .fp_mode(1'b0),
        .int16_mode(1'b0),
        .zp_in(zp_in),
        .bias(bias),
        .multiplier(multiplier),
        .shift(shift),
        .zp_out(zp_out),
        .act_min(act_min),
        .act_max(act_max),
        .conv_in_h(in_h),
        .conv_in_w(in_w),
        .conv_in_c(in_c),
        .conv_out_h(out_h),
        .conv_out_w(out_w),
        .conv_out_c(out_c),
        .conv_k_h(k_h),
        .conv_k_w(k_w),
        .conv_stride_h(stride_h),
        .conv_stride_w(stride_w),
        .conv_dilation_h(dilation_h),
        .conv_dilation_w(dilation_w),
        .conv_pad_top(pad_top),
        .conv_pad_left(pad_left),
        .conv_elem_bytes(elem_bytes),
        .conv_out_elem_index(out_elem_index),
        .conv_tile_output_count(tile_output_count),
        .conv_partial_first(1'b0),
        .conv_partial_accumulate(1'b0),
        .conv_partial_final(1'b0),
        .conv_sample_kh(sample_kh),
        .conv_sample_kw(sample_kw),
        .conv_sample_ic(sample_ic),
        .l1_req_valid(engine_l1_req_valid),
        .l1_req_ready(1'b1),
        .l1_req_write(engine_l1_req_write),
        .l1_req_bytes(engine_l1_req_bytes),
        .l1_req_payload_cycles(engine_l1_req_payload_cycles),
        .busy(engine_busy),
        .done_valid(engine_done_valid),
        .done_ready(1'b1),
        .phase_id(engine_phase_id),
        .remaining_cycles(engine_remaining_cycles),
        .acc_out(engine_acc_out),
        .scaled_out(engine_scaled_out),
        .out_q(engine_out_q),
        .fp_sum_bits(engine_fp_sum_bits),
        .int16_acc_out(engine_int16_acc_out),
        .conv_sample_input_byte_offset(engine_input_byte_offset),
        .conv_sample_weight_byte_offset(engine_weight_byte_offset),
        .conv_sample_output_byte_offset(engine_output_byte_offset),
        .conv_sample_input_valid(engine_input_valid),
        .conv_first_input_byte_offset(engine_first_input_byte_offset),
        .conv_first_weight_byte_offset(engine_first_weight_byte_offset),
        .conv_window_valid_count(engine_window_valid_count),
        .conv_tile_last_output_byte_offset(engine_tile_last_output_byte_offset),
        .conv_tile_last_input_valid(engine_tile_last_input_valid),
        .conv_tile_last_window_valid_count(engine_tile_last_window_valid_count),
        .conv_tile_scoreboard_valid_mask(engine_tile_scoreboard_valid_mask),
        .conv_tile_scoreboard_q_sum(engine_tile_scoreboard_q_sum),
        .conv_tile_result_out_elem_indices(engine_tile_result_out_elem_indices),
        .conv_tile_result_output_byte_offsets(engine_tile_result_output_byte_offsets),
        .conv_tile_result_acc_values(engine_tile_result_acc_values),
        .conv_tile_result_q_values(engine_tile_result_q_values),
        .conv_writeback_valid_mask(),
        .conv_writeback_output_byte_offsets(),
        .conv_writeback_q_values(),
        .conv_shadow_valid_mask(),
        .conv_shadow_output_byte_offsets(),
        .conv_shadow_q_values(),
        .conv_shadow_mem_valid_mask(),
        .conv_shadow_mem_output_byte_offsets(),
        .conv_shadow_mem_q_values(),
        .conv_psum_valid_mask(),
        .conv_psum_acc_values()
    );

    task clear_vecs;
        begin
            act_vec = {128{1'b0}};
            wgt_vec = {128{1'b0}};
        end
    endtask

    task set_pair;
        input integer idx;
        input signed [7:0] act;
        input signed [7:0] wgt;
        begin
            act_vec[idx*8 +: 8] = act[7:0];
            wgt_vec[idx*8 +: 8] = wgt[7:0];
        end
    endtask

    task expect_case;
        input [255:0] name;
        input signed [31:0] exp_acc;
        input signed [31:0] exp_scaled;
        input signed [7:0] exp_out;
        begin
            #1;
            if ((acc_out !== exp_acc) || (scaled_out !== exp_scaled) || (out_q !== exp_out)) begin
                $display("FAIL: %0s acc=%0d exp=%0d scaled=%0d exp=%0d out=%0d exp=%0d",
                         name, acc_out, exp_acc, scaled_out, exp_scaled, out_q, exp_out);
                failures = failures + 1;
            end
        end
    endtask

    task expect_addr;
        input [255:0] name;
        input exp_valid;
        input [31:0] exp_input;
        input [31:0] exp_weight;
        input [31:0] exp_output;
        begin
            #1;
            if ((input_valid !== exp_valid) ||
                (input_byte_offset !== exp_input) ||
                (weight_byte_offset !== exp_weight) ||
                (output_byte_offset !== exp_output)) begin
                $display("FAIL: %0s valid=%0d exp=%0d in=%0d exp=%0d wgt=%0d exp=%0d out=%0d exp=%0d",
                         name, input_valid, exp_valid,
                         input_byte_offset, exp_input,
                         weight_byte_offset, exp_weight,
                         output_byte_offset, exp_output);
                failures = failures + 1;
            end
        end
    endtask

    task expect_engine_sample;
        input [255:0] name;
        input exp_valid;
        input [31:0] exp_input;
        input [31:0] exp_weight;
        input [31:0] exp_output;
        input signed [31:0] exp_acc;
        input signed [7:0] exp_out;
        input [31:0] exp_first_input;
        input [31:0] exp_first_weight;
        input [7:0] exp_valid_count;
        input [31:0] exp_tile_last_output;
        input exp_tile_last_valid;
        input [7:0] exp_tile_last_valid_count;
        input [3:0] exp_tile_valid_mask;
        input signed [31:0] exp_tile_q_sum;
        begin
            #1;
            if ((engine_input_valid !== exp_valid) ||
                (engine_input_byte_offset !== exp_input) ||
                (engine_weight_byte_offset !== exp_weight) ||
                (engine_output_byte_offset !== exp_output) ||
                (engine_acc_out !== exp_acc) ||
                (engine_out_q !== exp_out) ||
                (engine_first_input_byte_offset !== exp_first_input) ||
                (engine_first_weight_byte_offset !== exp_first_weight) ||
                (engine_window_valid_count !== exp_valid_count) ||
                (engine_tile_last_output_byte_offset !== exp_tile_last_output) ||
                (engine_tile_last_input_valid !== exp_tile_last_valid) ||
                (engine_tile_last_window_valid_count !== exp_tile_last_valid_count) ||
                (engine_tile_scoreboard_valid_mask !== exp_tile_valid_mask) ||
                (engine_tile_scoreboard_q_sum !== exp_tile_q_sum) ||
                (engine_tile_result_out_elem_indices[31:0] !== 32'd0) ||
                (engine_tile_result_out_elem_indices[95:64] !== 32'd2) ||
                (engine_tile_result_output_byte_offsets[95:64] !== exp_tile_last_output) ||
                ($signed(engine_tile_result_acc_values[95:64]) !== exp_acc) ||
                ($signed(engine_tile_result_q_values[95:64]) !== exp_out)) begin
                $display("FAIL: %0s engine valid=%0d exp=%0d in=%0d exp=%0d wgt=%0d exp=%0d out_off=%0d exp=%0d acc=%0d exp=%0d q=%0d exp=%0d first_in=%0d exp=%0d first_wgt=%0d exp=%0d valid_count=%0d exp=%0d tile_last_out=%0d exp=%0d tile_valid=%0d exp=%0d tile_count=%0d exp=%0d tile_mask=%04b exp=%04b tile_q_sum=%0d exp=%0d entry0_idx=%0d entry2_idx=%0d entry2_off=%0d entry2_acc=%0d entry2_q=%0d",
                         name, engine_input_valid, exp_valid,
                         engine_input_byte_offset, exp_input,
                         engine_weight_byte_offset, exp_weight,
                         engine_output_byte_offset, exp_output,
                         engine_acc_out, exp_acc,
                         engine_out_q, exp_out,
                         engine_first_input_byte_offset, exp_first_input,
                         engine_first_weight_byte_offset, exp_first_weight,
                         engine_window_valid_count, exp_valid_count,
                         engine_tile_last_output_byte_offset, exp_tile_last_output,
                         engine_tile_last_input_valid, exp_tile_last_valid,
                         engine_tile_last_window_valid_count, exp_tile_last_valid_count,
                         engine_tile_scoreboard_valid_mask, exp_tile_valid_mask,
                         engine_tile_scoreboard_q_sum, exp_tile_q_sum,
                         engine_tile_result_out_elem_indices[31:0],
                         engine_tile_result_out_elem_indices[95:64],
                         engine_tile_result_output_byte_offsets[95:64],
                         $signed(engine_tile_result_acc_values[95:64]),
                         $signed(engine_tile_result_q_values[95:64]));
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        #12;
        rst_n = 1'b1;
        failures = 0;
        multiplier = 32'sd1073741824; // 0.5 in Q31; MBQM(x, mult, 1) returns x for small x.
        shift = 8'sd1;
        zp_out = 32'sd0;
        act_min = -32'sd128;
        act_max = 32'sd127;

        clear_vecs();
        elem_count = 8'd4;
        zp_in = 16'sd0;
        bias = 32'sd5;
        set_pair(0, 8'sd3,  -8'sd2);
        set_pair(1, -8'sd4,  8'sd5);
        set_pair(2, 8'sd7,   8'sd6);
        set_pair(3, 8'sd1,  -8'sd3);
        // 5 + 3*(-2) + (-4)*5 + 7*6 + 1*(-3) = 18
        expect_case("basic dot", 32'sd18, 32'sd18, 8'sd18);

        clear_vecs();
        elem_count = 8'd3;
        zp_in = 16'sd2;
        bias = -32'sd1;
        set_pair(0, 8'sd4,  8'sd3);
        set_pair(1, 8'sd2,  8'sd7);
        set_pair(2, -8'sd2, -8'sd5);
        // -1 + (4-2)*3 + (2-2)*7 + (-2-2)*(-5) = 25
        expect_case("zp and bias", 32'sd25, 32'sd25, 8'sd25);

        clear_vecs();
        elem_count = 8'd2;
        zp_in = 16'sd0;
        bias = 32'sd0;
        set_pair(0, 8'sd100, 8'sd3);
        set_pair(1, 8'sd80,  8'sd2);
        // raw 460, identity MBQM, clamp to int8 max.
        expect_case("clamp high", 32'sd460, 32'sd127, 8'sd127);

        clear_vecs();
        elem_count = 8'd1;
        zp_in = 16'sd0;
        bias = 32'sd0;
        set_pair(0, -8'sd100, 8'sd3);
        expect_case("clamp low", -32'sd300, -32'sd128, -8'sd128);

        clear_vecs();
        elem_count = 8'd1;
        zp_in = 16'sd0;
        bias = 32'sd0;
        multiplier = 32'sd2147483647;
        shift = -8'sd1;
        set_pair(0, 8'sd21, 8'sd1);
        // MBQM(21, ~1.0, -1) rounds 21/2 to 11.
        expect_case("round pot", 32'sd21, 32'sd11, 8'sd11);

        in_h = 16'd5;
        in_w = 16'd6;
        in_c = 16'd3;
        out_h = 16'd3;
        out_w = 16'd4;
        out_c = 16'd8;
        k_h = 8'd3;
        k_w = 8'd3;
        stride_h = 8'd1;
        stride_w = 8'd1;
        dilation_h = 8'd1;
        dilation_w = 8'd1;
        pad_top = 16'sd1;
        pad_left = 16'sd1;
        elem_bytes = 2'd1;
        tile_output_count = 8'd1;
        out_elem_index = 32'd53; // oh=1, ow=2, oc=5.
        sample_kh = 16'd2;
        sample_kw = 16'd1;
        sample_ic = 16'd2;
        expect_addr("nhwc valid window", 1'b1, 32'd44, 32'd189, 32'd53);

        clear_vecs();
        elem_count = 8'd8;
        zp_in = 16'sd0;
        bias = 32'sd5;
        multiplier = 32'sd1073741824;
        shift = 8'sd1;
        set_pair(0, 8'sd1,  8'sd1);
        set_pair(1, 8'sd2, -8'sd1);
        set_pair(2, 8'sd3,  8'sd2);
        set_pair(3, 8'sd4, -8'sd2);
        set_pair(4, 8'sd5,  8'sd3);
        set_pair(5, 8'sd6, -8'sd3);
        set_pair(6, 8'sd7,  8'sd4);
        set_pair(7, 8'sd8, -8'sd4);
        in_h = 16'd2;
        in_w = 16'd2;
        in_c = 16'd2;
        out_h = 16'd1;
        out_w = 16'd1;
        out_c = 16'd1;
        k_h = 8'd2;
        k_w = 8'd2;
        stride_h = 8'd1;
        stride_w = 8'd1;
        dilation_h = 8'd1;
        dilation_w = 8'd1;
        pad_top = 16'sd0;
        pad_left = 16'sd0;
        elem_bytes = 2'd1;
        out_elem_index = 32'd0;
        tile_output_count = 8'd3;
        sample_kh = 16'd1;
        sample_kw = 16'd1;
        sample_ic = 16'd1;
        // 5 + 1 - 2 + 6 - 8 + 15 - 18 + 28 - 32 = -5.
        expect_engine_sample("2d output pixel sample mac", 1'b1, 32'd7, 32'd7, 32'd0, -32'sd5, -8'sd6, 32'd0, 32'd0, 8'd8, 32'd2, 1'b0, 8'd0, 4'b0111, -32'sd18);

        in_h = 16'd5;
        in_w = 16'd6;
        in_c = 16'd3;
        out_h = 16'd3;
        out_w = 16'd4;
        out_c = 16'd8;
        k_h = 8'd3;
        k_w = 8'd3;
        stride_h = 8'd1;
        stride_w = 8'd1;
        dilation_h = 8'd1;
        dilation_w = 8'd1;
        pad_top = 16'sd1;
        pad_left = 16'sd1;
        elem_bytes = 2'd1;
        out_elem_index = 32'd0; // oh=0, ow=0, oc=0.
        sample_kh = 16'd0;
        sample_kw = 16'd0;
        sample_ic = 16'd1;
        expect_addr("padding invalid window", 1'b0, 32'd0, 32'd8, 32'd0);

        in_h = 16'd7;
        in_w = 16'd7;
        in_c = 16'd2;
        out_h = 16'd3;
        out_w = 16'd3;
        out_c = 16'd4;
        k_h = 8'd2;
        k_w = 8'd2;
        stride_h = 8'd2;
        stride_w = 8'd2;
        dilation_h = 8'd2;
        dilation_w = 8'd2;
        pad_top = 16'sd0;
        pad_left = 16'sd0;
        elem_bytes = 2'd2;
        out_elem_index = 32'd19; // oh=1, ow=1, oc=3.
        sample_kh = 16'd1;
        sample_kw = 16'd1;
        sample_ic = 16'd1;
        expect_addr("stride dilation int16 window", 1'b1, 32'd130, 32'd62, 32'd38);

        if (failures == 0)
            $display("PASS: verilog_final conv int8 MAC datapath and 2D address walk");
        else
            $display("FAIL: verilog_final conv datapath failures=%0d", failures);
        $finish;
    end
endmodule
