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
    integer failures;

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

    initial begin
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

        if (failures == 0)
            $display("PASS: verilog_final conv int8 MAC datapath");
        else
            $display("FAIL: verilog_final conv int8 MAC datapath failures=%0d", failures);
        $finish;
    end
endmodule
