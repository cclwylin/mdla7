`timescale 1ns/1ps

module Testbench_tnps_datapath;
    reg mode_s2d;
    reg [31:0] out_idx;
    reg [31:0] in_idx;
    reg [15:0] in_h;
    reg [15:0] in_w;
    reg [15:0] in_c;
    reg [15:0] out_h;
    reg [15:0] out_w;
    reg [15:0] out_c;
    wire [31:0] src_off;
    wire [31:0] dst_off;
    wire valid;
    integer failures;

    vf_tnps_addrgen u_addrgen (
        .mode_space_to_depth(mode_s2d),
        .in_h(in_h),
        .in_w(in_w),
        .in_c(in_c),
        .out_h(out_h),
        .out_w(out_w),
        .out_c(out_c),
        .block(16'd2),
        .elem_bytes(2'd1),
        .out_elem_index(out_idx),
        .in_elem_index(in_idx),
        .src_byte_offset(src_off),
        .dst_byte_offset(dst_off),
        .valid(valid)
    );

    task expect_s2d;
        input [31:0] idx;
        input [31:0] exp_src;
        begin
            mode_s2d = 1'b1;
            in_h = 16'd4;
            in_w = 16'd4;
            in_c = 16'd1;
            out_h = 16'd2;
            out_w = 16'd2;
            out_c = 16'd4;
            out_idx = idx;
            in_idx = 32'd0;
            #1;
            if (!valid || (src_off != exp_src) || (dst_off != idx)) begin
                $display("FAIL: S2D idx=%0d src=%0d exp=%0d dst=%0d valid=%0d",
                         idx, src_off, exp_src, dst_off, valid);
                failures = failures + 1;
            end
        end
    endtask

    task expect_d2s;
        input [31:0] idx;
        input [31:0] exp_dst;
        begin
            mode_s2d = 1'b0;
            in_h = 16'd2;
            in_w = 16'd2;
            in_c = 16'd4;
            out_h = 16'd4;
            out_w = 16'd4;
            out_c = 16'd1;
            out_idx = 32'd0;
            in_idx = idx;
            #1;
            if (!valid || (src_off != idx) || (dst_off != exp_dst)) begin
                $display("FAIL: D2S idx=%0d dst=%0d exp=%0d src=%0d valid=%0d",
                         idx, dst_off, exp_dst, src_off, valid);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        expect_s2d(32'd0, 32'd0);
        expect_s2d(32'd1, 32'd1);
        expect_s2d(32'd2, 32'd4);
        expect_s2d(32'd3, 32'd5);
        expect_s2d(32'd4, 32'd2);
        expect_s2d(32'd15, 32'd15);

        expect_d2s(32'd0, 32'd0);
        expect_d2s(32'd1, 32'd1);
        expect_d2s(32'd2, 32'd4);
        expect_d2s(32'd3, 32'd5);
        expect_d2s(32'd15, 32'd15);

        if (failures == 0)
            $display("PASS: verilog_final TNPS datapath address mapping");
        else
            $display("FAIL: verilog_final TNPS datapath address mapping failures=%0d", failures);
        $finish;
    end
endmodule
