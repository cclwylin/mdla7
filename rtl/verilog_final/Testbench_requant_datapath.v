`timescale 1ns/1ps

module Testbench_requant_datapath;
    reg clk;
    reg rst_n;
    reg start_valid;
    wire start_ready;
    reg signed [31:0] input_value;
    reg signed [31:0] multiplier;
    reg signed [7:0] shift;
    reg signed [31:0] zp_out;
    reg signed [31:0] act_min;
    reg signed [31:0] act_max;
    wire l1_req_valid;
    wire l1_req_write;
    wire [31:0] l1_req_bytes;
    wire [31:0] l1_req_payload_cycles;
    wire busy;
    wire done_valid;
    wire [3:0] phase_id;
    wire [31:0] remaining_cycles;
    wire signed [31:0] scaled_out;
    wire signed [7:0] out_q;
    integer failures;
    integer watchdog;

    always #5 clk = ~clk;

    vf_requant_sample_engine u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .input_value(input_value),
        .multiplier(multiplier),
        .shift(shift),
        .zp_out(zp_out),
        .act_min(act_min),
        .act_max(act_max),
        .sramcrc_mode(1'b0),
        .sramcrc_expected_count(32'd0),
        .out_byte_offset(32'd0),
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
        .sramcrc_crc(),
        .sramcrc_count(),
        .scaled_out(scaled_out),
        .out_q(out_q)
    );

    task run_case;
        input [255:0] name;
        input signed [31:0] exp_scaled;
        input signed [7:0] exp_q;
        begin
            @(posedge clk);
            while (!start_ready)
                @(posedge clk);
            start_valid = 1'b1;
            @(posedge clk);
            start_valid = 1'b0;

            watchdog = 0;
            while (!done_valid && watchdog < 100) begin
                watchdog = watchdog + 1;
                @(posedge clk);
            end

            if (!done_valid) begin
                $display("FAIL: %0s timeout busy=%0d phase=%0d remaining=%0d",
                         name, busy, phase_id, remaining_cycles);
                failures = failures + 1;
            end else begin
                if ((scaled_out !== exp_scaled) || (out_q !== exp_q)) begin
                    $display("FAIL: %0s scaled=%0d exp=%0d out=%0d exp=%0d",
                             name, scaled_out, exp_scaled, out_q, exp_q);
                    failures = failures + 1;
                end
                if (!l1_req_write || (l1_req_bytes !== 32'd1)) begin
                    $display("FAIL: %0s store token write=%0d bytes=%0d",
                             name, l1_req_write, l1_req_bytes);
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
        input_value = 32'sd0;
        multiplier = 32'sd1073741824;
        shift = 8'sd1;
        zp_out = 32'sd0;
        act_min = -32'sd128;
        act_max = 32'sd127;
        failures = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        input_value = 32'sd25;
        multiplier = 32'sd1073741824;
        shift = 8'sd1;
        zp_out = 32'sd0;
        run_case("identity small", 32'sd25, 8'sd25);

        input_value = 32'sd10;
        zp_out = 32'sd3;
        run_case("output zero point", 32'sd13, 8'sd13);

        input_value = 32'sd460;
        zp_out = 32'sd0;
        run_case("clamp high", 32'sd127, 8'sd127);

        input_value = -32'sd300;
        run_case("clamp low", -32'sd128, -8'sd128);

        input_value = 32'sd21;
        multiplier = 32'sd2147483647;
        shift = -8'sd1;
        run_case("round pot", 32'sd11, 8'sd11);

        if (failures == 0)
            $display("PASS: verilog_final requant datapath");
        else
            $display("FAIL: verilog_final requant datapath failures=%0d", failures);
        $finish;
    end
endmodule
