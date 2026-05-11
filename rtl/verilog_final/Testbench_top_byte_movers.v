`timescale 1ns/1ps

module Testbench_top_byte_movers;
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;

    reg clk;
    reg rst_n;
    reg desc_valid;
    wire desc_ready;
    reg [3:0] desc_op_class;
    reg [31:0] bytes;
    reg [31:0] udma_dram_read_bytes;
    reg [31:0] udma_codec_cycles;
    reg udma_direction_write;
    reg [21:0] l1mesh_addr;
    reg [127:0] l1mesh_wdata;
    reg [15:0] l1mesh_wstrb;
    reg tnps_mode_space_to_depth;
    reg [15:0] tnps_in_h;
    reg [15:0] tnps_in_w;
    reg [15:0] tnps_in_c;
    reg [15:0] tnps_out_h;
    reg [15:0] tnps_out_w;
    reg [15:0] tnps_out_c;
    reg [15:0] tnps_block;
    reg [1:0] tnps_elem_bytes;
    reg [31:0] tnps_sample_out_elem_index;
    reg [31:0] tnps_sample_in_elem_index;
    wire done_valid;
    wire busy;
    wire [3:0] active_op_class;
    wire [3:0] active_phase_id;
    wire [31:0] active_remaining_cycles;
    wire [31:0] tnps_sample_src_byte_offset;
    wire [31:0] tnps_sample_dst_byte_offset;
    wire tnps_sample_valid;
    wire [31:0] placement_route_cycles;
    wire [8:0] block_busy;
    wire [8:0] block_done_valid;
    integer failures;
    integer watchdog;

    always #5 clk = ~clk;

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
        .block_done_valid(block_done_valid)
    );

    task issue_desc;
        input [3:0] op_class;
        begin
            @(posedge clk);
            while (!desc_ready)
                @(posedge clk);
            desc_op_class = op_class;
            desc_valid = 1'b1;
            @(posedge clk);
            desc_valid = 1'b0;
        end
    endtask

    task wait_done;
        input [255:0] label;
        begin
            watchdog = 0;
            while (!done_valid && watchdog < 1000) begin
                watchdog = watchdog + 1;
                @(posedge clk);
            end
            if (!done_valid) begin
                $display("FAIL: timeout waiting for %0s active_op=%0d phase=%0d remain=%0d busy=%0d block_busy=%09b block_done=%09b",
                         label, active_op_class, active_phase_id,
                         active_remaining_cycles, busy, block_busy, block_done_valid);
                failures = failures + 1;
            end
            @(posedge clk);
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        desc_valid = 1'b0;
        desc_op_class = 4'd0;
        bytes = 32'd256;
        udma_dram_read_bytes = 32'd512;
        udma_codec_cycles = 32'd3;
        udma_direction_write = 1'b0;
        l1mesh_addr = 22'h0002a0;
        l1mesh_wdata = 128'h0123456789abcdef0011223344556677;
        l1mesh_wstrb = 16'hffff;
        tnps_mode_space_to_depth = 1'b1;
        tnps_in_h = 16'd4;
        tnps_in_w = 16'd4;
        tnps_in_c = 16'd1;
        tnps_out_h = 16'd2;
        tnps_out_w = 16'd2;
        tnps_out_c = 16'd4;
        tnps_block = 16'd2;
        tnps_elem_bytes = 2'd1;
        tnps_sample_out_elem_index = 32'd2;
        tnps_sample_in_elem_index = 32'd0;
        failures = 0;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;

        issue_desc(OP_UDMA);
        wait_done("udma");
        if (placement_route_cycles == 32'd0) begin
            $display("FAIL: UDMA placement_route_cycles was zero");
            failures = failures + 1;
        end

        bytes = 32'd128;
        l1mesh_addr = 22'h0003f0;
        issue_desc(OP_TNPS);
        wait_done("tnps");
        if (!tnps_sample_valid ||
            (tnps_sample_src_byte_offset != 32'd4) ||
            (tnps_sample_dst_byte_offset != 32'd2)) begin
            $display("FAIL: TNPS sample mapping valid=%0d src=%0d dst=%0d",
                     tnps_sample_valid,
                     tnps_sample_src_byte_offset,
                     tnps_sample_dst_byte_offset);
            failures = failures + 1;
        end

        if (failures == 0)
            $display("PASS: verilog_final top byte movers UDMA/TNPS");
        else
            $display("FAIL: verilog_final top byte movers failures=%0d", failures);
        $finish;
    end
endmodule
