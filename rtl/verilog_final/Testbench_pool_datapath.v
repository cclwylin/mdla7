`timescale 1ns/1ps

module Testbench_pool_datapath;
    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg avg_mode;
    reg fp_mode;
    reg int16_mode;
    reg [16*8-1:0] sample_vec;
    reg [7:0] elem_count;
    wire l1_req_valid;
    wire l1_req_write;
    wire [31:0] l1_req_bytes;
    wire [31:0] l1_req_payload_cycles;
    wire busy;
    wire done_valid;
    wire [3:0] phase_id;
    wire [31:0] remaining_cycles;
    wire signed [31:0] pool_out;
    wire signed [7:0] out_q;
    wire [63:0] fp_pool_bits;
    integer failures;
    integer watchdog;

    always #5 clk = ~clk;

    vf_pool_sample_engine #(
        .MAX_ELEMS(16)
    ) u_pool (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .avg_mode(avg_mode),
        .fp_mode(fp_mode),
        .int16_mode(int16_mode),
        .read_sample_from_l1(1'b0),
        .refcrc_mode(1'b0),
        .sramcrc_mode(1'b0),
        .refcrc_expected_count(32'd0),
        .refcrc_ref_off(32'd0),
        .out_byte_offset(32'd0),
        .l1_req_base_addr(22'd0),
        .sample_vec(sample_vec),
        .elem_count(elem_count),
        .l1_resp_valid(1'b0),
        .l1_resp_rdata(128'd0),
        .l1_req_valid(l1_req_valid),
        .l1_req_ready(1'b1),
        .l1_req_write(l1_req_write),
        .l1_req_addr(),
        .l1_req_bytes(l1_req_bytes),
        .l1_req_payload_cycles(l1_req_payload_cycles),
        .l1_req_wdata(),
        .l1_req_wstrb(),
        .busy(busy),
        .done_valid(done_valid),
        .done_ready(1'b1),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles),
        .pool_out(pool_out),
        .out_q(out_q),
        .fp_pool_bits(fp_pool_bits),
        .refcrc_crc(),
        .refcrc_count()
    );

    task clear_vec;
        begin
            sample_vec = {128{1'b0}};
        end
    endtask

    task set_value;
        input integer idx;
        input signed [7:0] value;
        begin
            sample_vec[idx*8 +: 8] = value[7:0];
        end
    endtask

    task run_case;
        input [255:0] name;
        input signed [31:0] exp_out;
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
                if (!fp_mode && !int16_mode && ((pool_out !== exp_out) || (out_q !== exp_out[7:0]))) begin
                    $display("FAIL: %0s out=%0d exp=%0d q=%0d",
                             name, pool_out, exp_out, out_q);
                    failures = failures + 1;
                end
                if (int16_mode && (pool_out !== exp_out)) begin
                    $display("FAIL: %0s int16_out=%0d exp=%0d",
                             name, pool_out, exp_out);
                    failures = failures + 1;
                end
                if ((!fp_mode && !int16_mode && (l1_req_bytes !== 32'd1)) ||
                    (int16_mode && (l1_req_bytes !== 32'd4)) ||
                    (fp_mode && (l1_req_bytes !== 32'd8))) begin
                    $display("FAIL: %0s store bytes=%0d exp=%0d",
                             name, l1_req_bytes, fp_mode ? 32'd8 : (int16_mode ? 32'd4 : 32'd1));
                    failures = failures + 1;
                end
                if (l1_req_payload_cycles !== 32'd2) begin
                    $display("FAIL: %0s payload_cycles=%0d exp=2",
                             name, l1_req_payload_cycles);
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
        avg_mode = 1'b0;
        fp_mode = 1'b0;
        int16_mode = 1'b0;
        elem_count = 8'd0;
        clear_vec();
        failures = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        clear_vec();
        avg_mode = 1'b0;
        elem_count = 8'd5;
        set_value(0, -8'sd4);
        set_value(1, 8'sd9);
        set_value(2, 8'sd2);
        set_value(3, 8'sd7);
        set_value(4, -8'sd1);
        run_case("max pool", 32'sd9);

        clear_vec();
        avg_mode = 1'b1;
        elem_count = 8'd4;
        set_value(0, 8'sd4);
        set_value(1, 8'sd5);
        set_value(2, -8'sd1);
        set_value(3, 8'sd0);
        run_case("avg pool", 32'sd2);

        clear_vec();
        avg_mode = 1'b1;
        elem_count = 8'd3;
        set_value(0, -8'sd9);
        set_value(1, -8'sd1);
        set_value(2, 8'sd1);
        run_case("negative avg trunc", -32'sd3);

        clear_vec();
        avg_mode = 1'b0;
        elem_count = 8'd20;
        set_value(0, 8'sd1);
        set_value(1, 8'sd2);
        set_value(2, 8'sd3);
        set_value(3, 8'sd4);
        set_value(4, 8'sd5);
        set_value(5, 8'sd6);
        set_value(6, 8'sd7);
        set_value(7, 8'sd8);
        set_value(8, 8'sd9);
        set_value(9, 8'sd10);
        set_value(10, 8'sd11);
        set_value(11, 8'sd12);
        set_value(12, 8'sd13);
        set_value(13, 8'sd14);
        set_value(14, 8'sd15);
        set_value(15, 8'sd16);
        run_case("count cap", 32'sd16);

        clear_vec();
        avg_mode = 1'b1;
        fp_mode = 1'b1;
        elem_count = 8'd4;
        sample_vec[15:0] = 16'h3c00;
        sample_vec[31:16] = 16'h4000;
        sample_vec[47:32] = 16'h4200;
        sample_vec[63:48] = 16'h4400;
        run_case("fp avg pool", 32'sd0);
        if (fp_pool_bits != 64'h4004000000000000) begin
            $display("FAIL: fp avg pool bits=%016x exp=4004000000000000", fp_pool_bits);
            failures = failures + 1;
        end
        fp_mode = 1'b0;

        clear_vec();
        avg_mode = 1'b0;
        int16_mode = 1'b1;
        elem_count = 8'd4;
        sample_vec[15:0] = -16'sd9;
        sample_vec[31:16] = -16'sd1;
        sample_vec[47:32] = 16'sd1;
        sample_vec[63:48] = 16'sd5;
        run_case("int16 max pool", 32'sd5);

        clear_vec();
        avg_mode = 1'b1;
        int16_mode = 1'b1;
        elem_count = 8'd4;
        sample_vec[15:0] = -16'sd9;
        sample_vec[31:16] = -16'sd1;
        sample_vec[47:32] = 16'sd1;
        sample_vec[63:48] = 16'sd5;
        run_case("int16 avg pool", -32'sd1);
        int16_mode = 1'b0;

        if (failures == 0)
            $display("PASS: verilog_final pool datapath");
        else
            $display("FAIL: verilog_final pool datapath failures=%0d", failures);
        $finish;
    end
endmodule
