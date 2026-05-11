`timescale 1ns/1ps

module Testbench_l1mesh_contention;
    reg clk;
    reg rst_n;
    reg udma_valid;
    reg requant_valid;
    reg ewe_valid;
    reg pool_valid;
    reg tnps_valid;
    wire udma_ready;
    wire requant_ready;
    wire ewe_ready;
    wire pool_ready;
    wire tnps_ready;
    wire mgr_resp_valid;
    wire mesh_req_ready;
    wire mesh_resp_valid;
    wire mgr_busy;
    wire mesh_busy;
    wire [3:0] mgr_phase;
    wire [31:0] mgr_remaining;
    wire [3:0] mesh_phase;
    wire [31:0] mesh_remaining;
    wire [3:0] debug_source;
    wire [7:0] debug_tid;
    wire legacy_ready;
    wire mesh_req_write;
    wire [21:0] mesh_req_addr;
    wire [31:0] mesh_req_bytes;
    wire [127:0] mesh_req_wdata;
    wire [15:0] mesh_req_wstrb;
    wire [127:0] mesh_rdata;
    integer udma_accepted;
    integer requant_accepted;
    integer ewe_accepted;
    integer pool_accepted;
    integer tnps_accepted;
    integer mesh_responses;
    reg [4:0] blocked_seen;
    wire [219:0] debug_unused = {
        legacy_ready,
        mgr_phase,
        mgr_remaining,
        mesh_phase,
        mesh_remaining,
        debug_source,
        debug_tid,
        mesh_rdata
    };

    always #5 clk = ~clk;

    l1manager u_mgr (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(1'b0),
        .req_ready(legacy_ready),
        .req_write(1'b0),
        .req_l1(1'b1),
        .req_source(4'd0),
        .req_tid(8'd0),
        .req_bytes(32'd0),
        .req_payload_cycles(32'd0),
        .req_addr(22'd0),
        .req_wdata(128'd0),
        .req_wstrb(16'd0),
        .udma_req_valid(udma_valid),
        .udma_req_ready(udma_ready),
        .udma_req_write(1'b1),
        .udma_req_tid(8'h10),
        .udma_req_bytes(32'd256),
        .udma_req_payload_cycles(32'd12),
        .udma_req_addr(22'h000000),
        .udma_req_wdata(128'h112233445566778899aabbccddeeff00),
        .udma_req_wstrb(16'hffff),
        .requant_req_valid(requant_valid),
        .requant_req_ready(requant_ready),
        .requant_req_write(1'b1),
        .requant_req_tid(8'h20),
        .requant_req_bytes(32'd1),
        .requant_req_payload_cycles(32'd2),
        .requant_req_addr(22'h000040),
        .requant_req_wdata(128'h0000000000000000000000000000007f),
        .requant_req_wstrb(16'h0001),
        .ewe_req_valid(ewe_valid),
        .ewe_req_ready(ewe_ready),
        .ewe_req_write(1'b0),
        .ewe_req_tid(8'h30),
        .ewe_req_bytes(32'd16),
        .ewe_req_payload_cycles(32'd2),
        .ewe_req_addr(22'h000080),
        .ewe_req_wdata(128'd0),
        .ewe_req_wstrb(16'd0),
        .pool_req_valid(pool_valid),
        .pool_req_ready(pool_ready),
        .pool_req_write(1'b0),
        .pool_req_tid(8'h40),
        .pool_req_bytes(32'd16),
        .pool_req_payload_cycles(32'd2),
        .pool_req_addr(22'h0000c0),
        .pool_req_wdata(128'd0),
        .pool_req_wstrb(16'd0),
        .tnps_req_valid(tnps_valid),
        .tnps_req_ready(tnps_ready),
        .tnps_req_write(1'b1),
        .tnps_req_tid(8'h50),
        .tnps_req_bytes(32'd128),
        .tnps_req_payload_cycles(32'd2),
        .tnps_req_addr(22'h000100),
        .tnps_req_wdata(128'hffeeddccbbaa99887766554433221100),
        .tnps_req_wstrb(16'hffff),
        .mesh_req_write(mesh_req_write),
        .mesh_req_addr(mesh_req_addr),
        .mesh_req_bytes(mesh_req_bytes),
        .mesh_req_wdata(mesh_req_wdata),
        .mesh_req_wstrb(mesh_req_wstrb),
        .resp_valid(mgr_resp_valid),
        .resp_ready(mesh_req_ready),
        .busy(mgr_busy),
        .phase_id(mgr_phase),
        .remaining_cycles(mgr_remaining),
        .debug_source(debug_source),
        .debug_tid(debug_tid)
    );

    l1mesh u_mesh (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(mgr_resp_valid),
        .req_ready(mesh_req_ready),
        .req_write(mesh_req_write),
        .req_addr(mesh_req_addr),
        .req_bytes(mesh_req_bytes),
        .route_cycles(32'd4),
        .req_wdata(mesh_req_wdata),
        .req_wstrb(mesh_req_wstrb),
        .resp_valid(mesh_resp_valid),
        .resp_ready(1'b1),
        .resp_rdata(mesh_rdata),
        .busy(mesh_busy),
        .phase_id(mesh_phase),
        .remaining_cycles(mesh_remaining)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            udma_accepted <= 0;
            requant_accepted <= 0;
            ewe_accepted <= 0;
            pool_accepted <= 0;
            tnps_accepted <= 0;
            mesh_responses <= 0;
            blocked_seen <= 5'd0;
        end else begin
            if (udma_valid && udma_ready)
                udma_accepted <= udma_accepted + 1;
            if (requant_valid && requant_ready)
                requant_accepted <= requant_accepted + 1;
            if (ewe_valid && ewe_ready)
                ewe_accepted <= ewe_accepted + 1;
            if (pool_valid && pool_ready)
                pool_accepted <= pool_accepted + 1;
            if (tnps_valid && tnps_ready)
                tnps_accepted <= tnps_accepted + 1;

            if (udma_valid && !udma_ready)
                blocked_seen[0] <= 1'b1;
            if (requant_valid && !requant_ready)
                blocked_seen[1] <= 1'b1;
            if (ewe_valid && !ewe_ready)
                blocked_seen[2] <= 1'b1;
            if (pool_valid && !pool_ready)
                blocked_seen[3] <= 1'b1;
            if (tnps_valid && !tnps_ready)
                blocked_seen[4] <= 1'b1;

            if (mesh_resp_valid)
                mesh_responses <= mesh_responses + 1;
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        udma_valid = 1'b0;
        requant_valid = 1'b0;
        ewe_valid = 1'b0;
        pool_valid = 1'b0;
        tnps_valid = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        udma_valid = 1'b1;
        requant_valid = 1'b1;
        ewe_valid = 1'b1;
        pool_valid = 1'b1;
        tnps_valid = 1'b1;
        repeat (8) @(posedge clk);
        udma_valid = 1'b0;
        requant_valid = 1'b0;
        ewe_valid = 1'b0;
        pool_valid = 1'b0;
        tnps_valid = 1'b0;
        repeat (420) @(posedge clk);

        if ((udma_accepted >= 2) &&
            (requant_accepted >= 2) &&
            (ewe_accepted >= 2) &&
            (pool_accepted >= 2) &&
            (tnps_accepted >= 2) &&
            (blocked_seen == 5'b11111) &&
            (mesh_responses >= 10)) begin
            $display("PASS: verilog_final L1Manager/L1Mesh multi-source contention udma=%0d requant=%0d ewe=%0d pool=%0d tnps=%0d responses=%0d",
                     udma_accepted, requant_accepted, ewe_accepted,
                     pool_accepted, tnps_accepted, mesh_responses);
        end else begin
            $display("FAIL: verilog_final L1Manager/L1Mesh multi-source contention udma=%0d requant=%0d ewe=%0d pool=%0d tnps=%0d blocked=%05b responses=%0d mgr_busy=%0d mesh_busy=%0d",
                     udma_accepted, requant_accepted, ewe_accepted,
                     pool_accepted, tnps_accepted, blocked_seen,
                     mesh_responses, mgr_busy, mesh_busy);
        end
        $finish;
    end
endmodule
