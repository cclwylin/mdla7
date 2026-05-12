`timescale 1ns/1ps

module Testbench_ewe_datapath;
    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg [1:0] op_mode;
    reg fp_mode;
    reg int16_mode;
    reg [16*8-1:0] a_vec;
    reg [16*8-1:0] b_vec;
    reg [7:0] elem_count;
    wire l1_req_valid;
    wire l1_req_write;
    wire [31:0] l1_req_bytes;
    wire [31:0] l1_req_payload_cycles;
    wire busy;
    wire done_valid;
    wire [3:0] phase_id;
    wire [31:0] remaining_cycles;
    wire signed [31:0] ewe_out;
    wire signed [7:0] out_q;
    wire [63:0] fp_ewe_bits;
    integer failures;
    integer watchdog;

    always #5 clk = ~clk;

    vf_ewe_sample_engine #(
        .MAX_ELEMS(16)
    ) u_ewe (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .op_mode(op_mode),
        .fp_mode(fp_mode),
        .int16_mode(int16_mode),
        .final_q_mode(1'b0),
        .sramcrc_mode(1'b0),
        .sramcrc_expected_count(32'd0),
        .out_byte_offset(32'd0),
        .zp_a(32'sd0),
        .zp_b(32'sd0),
        .zp_out(32'sd0),
        .mult_a(32'sd1073741824),
        .shift_a(8'sd0),
        .mult_b(32'sd1073741824),
        .shift_b(8'sd0),
        .mult_out(32'sd1073741824),
        .shift_out(8'sd0),
        .left_shift(32'sd0),
        .act_min(-32'sd128),
        .act_max(32'sd127),
        .a_vec(a_vec),
        .b_vec(b_vec),
        .elem_count(elem_count),
        .l1_req_valid(l1_req_valid),
        .l1_req_ready(1'b1),
        .l1_req_write(l1_req_write),
        .l1_req_bytes(l1_req_bytes),
        .l1_req_payload_cycles(l1_req_payload_cycles),
        .busy(busy),
        .done_valid(done_valid),
        .done_ready(1'b1),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles),
        .ewe_out(ewe_out),
        .out_q(out_q),
        .sramcrc_crc(),
        .sramcrc_count(),
        .fp_ewe_bits(fp_ewe_bits)
    );

    task clear_vecs;
        begin
            a_vec = {128{1'b0}};
            b_vec = {128{1'b0}};
        end
    endtask

    task set_pair;
        input integer idx;
        input signed [7:0] a_val;
        input signed [7:0] b_val;
        begin
            a_vec[idx*8 +: 8] = a_val[7:0];
            b_vec[idx*8 +: 8] = b_val[7:0];
        end
    endtask

    task run_case;
        input [255:0] name;
        input signed [31:0] exp_sum;
        input signed [7:0] exp_first;
        input [31:0] exp_bytes;
        begin
            @(posedge clk);
            while (!start_ready)
                @(posedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            start_valid = 1'b0;

            watchdog = 0;
            while (!done_valid && watchdog < 200) begin
                watchdog = watchdog + 1;
                @(posedge clk);
            end

            if (!done_valid) begin
                $display("FAIL: %0s timeout busy=%0d phase=%0d remaining=%0d",
                         name, busy, phase_id, remaining_cycles);
                failures = failures + 1;
            end else begin
                if (!fp_mode && !int16_mode && ((ewe_out !== exp_sum) || (out_q !== exp_first))) begin
                    $display("FAIL: %0s sum=%0d exp=%0d first=%0d exp=%0d",
                             name, ewe_out, exp_sum, out_q, exp_first);
                    failures = failures + 1;
                end
                if (int16_mode && (ewe_out !== exp_sum)) begin
                    $display("FAIL: %0s int16_sum=%0d exp=%0d",
                             name, ewe_out, exp_sum);
                    failures = failures + 1;
                end
                if (l1_req_bytes !== exp_bytes) begin
                    $display("FAIL: %0s l1_req_bytes=%0d exp=%0d",
                             name, l1_req_bytes, exp_bytes);
                    failures = failures + 1;
                end
            end
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start_valid = 1'b0;
        op_mode = 2'd0;
        fp_mode = 1'b0;
        int16_mode = 1'b0;
        elem_count = 8'd0;
        clear_vecs();
        failures = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        clear_vecs();
        op_mode = 2'd0;
        elem_count = 8'd4;
        set_pair(0, 8'sd4,  8'sd3);
        set_pair(1, 8'sd3, -8'sd1);
        set_pair(2, 8'sd2,  8'sd1);
        set_pair(3, 8'sd1,  8'sd2);
        run_case("add vector", 32'sd15, 8'sd7, 32'd4);

        clear_vecs();
        op_mode = 2'd1;
        elem_count = 8'd4;
        set_pair(0, 8'sd64,  8'sd3);
        set_pair(1, -8'sd64, 8'sd3);
        set_pair(2, 8'sd8,  -8'sd9);
        set_pair(3, -8'sd7, -8'sd8);
        run_case("mul saturate vector", -32'sd17, 8'sd127, 32'd4);

        clear_vecs();
        op_mode = 2'd2;
        elem_count = 8'd3;
        set_pair(0, -8'sd120, 8'sd20);
        set_pair(1, 8'sd50,   8'sd10);
        set_pair(2, -8'sd5,  -8'sd8);
        run_case("sub vector", -32'sd85, -8'sd128, 32'd3);

        clear_vecs();
        op_mode = 2'd0;
        elem_count = 8'd20;
        set_pair(0, 8'sd1, 8'sd0);
        set_pair(1, 8'sd1, 8'sd0);
        set_pair(2, 8'sd1, 8'sd0);
        set_pair(3, 8'sd1, 8'sd0);
        set_pair(4, 8'sd1, 8'sd0);
        set_pair(5, 8'sd1, 8'sd0);
        set_pair(6, 8'sd1, 8'sd0);
        set_pair(7, 8'sd1, 8'sd0);
        set_pair(8, 8'sd1, 8'sd0);
        set_pair(9, 8'sd1, 8'sd0);
        set_pair(10, 8'sd1, 8'sd0);
        set_pair(11, 8'sd1, 8'sd0);
        set_pair(12, 8'sd1, 8'sd0);
        set_pair(13, 8'sd1, 8'sd0);
        set_pair(14, 8'sd1, 8'sd0);
        set_pair(15, 8'sd1, 8'sd0);
        run_case("count cap", 32'sd16, 8'sd1, 32'd16);

        clear_vecs();
        op_mode = 2'd1;
        fp_mode = 1'b1;
        elem_count = 8'd2;
        a_vec[15:0] = 16'h3c00;
        a_vec[31:16] = 16'h4000;
        b_vec[15:0] = 16'h4000;
        b_vec[31:16] = 16'h4200;
        run_case("fp mul vector", 32'sd0, 8'sd0, 32'd4);
        if (fp_ewe_bits != 64'h4020000000000000) begin
            $display("FAIL: fp mul vector bits=%016x exp=4020000000000000", fp_ewe_bits);
            failures = failures + 1;
        end
        fp_mode = 1'b0;

        clear_vecs();
        op_mode = 2'd3;
        fp_mode = 1'b1;
        elem_count = 8'd2;
        a_vec[15:0] = 16'h3c00;
        a_vec[31:16] = 16'h0000;
        run_case("fp logistic vector", 32'sd0, 8'sd0, 32'd4);
        if (fp_ewe_bits != 64'h3ff3b26a7aead15e) begin
            $display("FAIL: fp logistic vector bits=%016x exp=3ff3b26a7aead15e", fp_ewe_bits);
            failures = failures + 1;
        end
        fp_mode = 1'b0;

        clear_vecs();
        op_mode = 2'd0;
        int16_mode = 1'b1;
        elem_count = 8'd4;
        a_vec[15:0] = 16'sd4;
        a_vec[31:16] = 16'sd3;
        a_vec[47:32] = -16'sd2;
        a_vec[63:48] = 16'sd1;
        b_vec[15:0] = 16'sd3;
        b_vec[31:16] = -16'sd1;
        b_vec[47:32] = 16'sd5;
        b_vec[63:48] = 16'sd2;
        run_case("int16 add vector", 32'sd15, 8'sd0, 32'd8);
        int16_mode = 1'b0;

        if (failures == 0)
            $display("PASS: verilog_final EWE vector datapath");
        else
            $display("FAIL: verilog_final EWE vector datapath failures=%0d", failures);
        $finish;
    end
endmodule
