`timescale 1ns/1ps

module Testbench_l1mesh_storage;
    reg clk;
    reg rst_n;
    reg req_valid;
    wire req_ready;
    reg req_write;
    reg [21:0] req_addr;
    reg [31:0] req_bytes;
    reg [31:0] route_cycles;
    reg [127:0] req_wdata;
    reg [15:0] req_wstrb;
    reg [3:0] req_source;
    reg [7:0] req_tid;
    wire resp_valid;
    wire resp_read;
    wire [3:0] resp_source;
    wire [7:0] resp_tid;
    reg resp_ready;
    wire [127:0] resp_rdata;
    wire busy;
    wire [3:0] phase_id;
    wire [31:0] remaining_cycles;

    integer errors;
    integer watchdog;

    always #5 clk = ~clk;

    l1mesh u_mesh (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(req_valid),
        .req_ready(req_ready),
        .req_write(req_write),
        .req_addr(req_addr),
        .req_bytes(req_bytes),
        .route_cycles(route_cycles),
        .req_wdata(req_wdata),
        .req_wstrb(req_wstrb),
        .req_source(req_source),
        .req_tid(req_tid),
        .debug_crc_start(1'b0),
        .debug_crc_addr(22'd0),
        .debug_crc_count(32'd0),
        .debug_crc_busy(),
        .debug_crc_done(),
        .debug_crc(),
        .debug_crc_byte_count(),
        .resp_valid(resp_valid),
        .resp_read(resp_read),
        .resp_source(resp_source),
        .resp_tid(resp_tid),
        .resp_ready(resp_ready),
        .resp_rdata(resp_rdata),
        .busy(busy),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );

    task issue_req;
        input write;
        input [21:0] addr;
        input [7:0] tid;
        input [127:0] wdata;
        begin
            req_write = write;
            req_addr = addr;
            req_tid = tid;
            req_wdata = wdata;
            req_valid = 1'b1;
            watchdog = 0;
            while (!req_ready && watchdog < 200) begin
                watchdog = watchdog + 1;
                @(posedge clk);
            end
            @(posedge clk);
            req_valid = 1'b0;
            req_write = 1'b0;
            req_wdata = 128'd0;
        end
    endtask

    task wait_resp;
        input expect_read;
        input [7:0] expect_tid;
        input [127:0] expect_data;
        begin
            watchdog = 0;
            while (!resp_valid && watchdog < 400) begin
                watchdog = watchdog + 1;
                @(posedge clk);
            end
            if (!resp_valid) begin
                $display("FAIL: response timeout tid=%0h", expect_tid);
                errors = errors + 1;
            end else begin
                if (resp_read !== expect_read) begin
                    $display("FAIL: resp_read tid=%0h got=%0b expected=%0b",
                             expect_tid, resp_read, expect_read);
                    errors = errors + 1;
                end
                if (resp_tid !== expect_tid) begin
                    $display("FAIL: resp_tid got=%0h expected=%0h",
                             resp_tid, expect_tid);
                    errors = errors + 1;
                end
                if (expect_read && (resp_rdata !== expect_data)) begin
                    $display("FAIL: resp_data tid=%0h got=%032x expected=%032x",
                             expect_tid, resp_rdata, expect_data);
                    errors = errors + 1;
                end
            end
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        req_valid = 1'b0;
        req_write = 1'b0;
        req_addr = 22'd0;
        req_bytes = 32'd16;
        route_cycles = 32'd1;
        req_wdata = 128'd0;
        req_wstrb = 16'hffff;
        req_source = 4'd2;
        req_tid = 8'd0;
        resp_ready = 1'b1;
        errors = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        issue_req(1'b1, 22'h000000, 8'h11, 128'h00112233445566778899aabbccddeeff);
        wait_resp(1'b0, 8'h11, 128'd0);
        issue_req(1'b1, 22'h0003f0, 8'h12, 128'hffeeddccbbaa99887766554433221100);
        wait_resp(1'b0, 8'h12, 128'd0);

        issue_req(1'b0, 22'h000000, 8'h21, 128'd0);
        wait_resp(1'b1, 8'h21, 128'h00112233445566778899aabbccddeeff);
        issue_req(1'b0, 22'h0003f0, 8'h22, 128'd0);
        wait_resp(1'b1, 8'h22, 128'hffeeddccbbaa99887766554433221100);

        resp_ready = 1'b0;
        issue_req(1'b0, 22'h000000, 8'h31, 128'd0);
        issue_req(1'b0, 22'h0003f0, 8'h32, 128'd0);
        repeat (40) @(posedge clk);
        if (!busy) begin
            $display("FAIL: L1Mesh should remain busy while response FIFO is backpressured");
            errors = errors + 1;
        end
        resp_ready = 1'b1;
        wait_resp(1'b1, 8'h31, 128'h00112233445566778899aabbccddeeff);
        wait_resp(1'b1, 8'h32, 128'hffeeddccbbaa99887766554433221100);

        if (errors == 0)
            $display("PASS: L1Mesh SRAM macro hierarchy storage/readback");
        else
            $display("FAIL: L1Mesh SRAM macro hierarchy errors=%0d", errors);
        $finish;
    end
endmodule
