`timescale 1ns/1ps

module Testbench;
    /* verilator lint_off UNUSEDSIGNAL */
    reg clk;
    reg rst_n;
    reg [63:0] cycle_count;

    wire        desc_valid;
    wire        desc_ready;
    wire [3:0]  desc_op_class;
    wire [7:0]  desc_wait_count;
    wire [15:0] desc_layer_id;
    wire [15:0] desc_microblock_id;
    wire [7:0]  desc_stream_slot;
    wire [7:0]  desc_stream_meta_flags;
    wire [31:0] cfg_write_cycles;

    wire [31:0] bytes;
    wire [31:0] read_bytes;
    wire [31:0] write_bytes;
    wire [31:0] act_bytes;
    wire [31:0] wgt_bytes;
    wire [31:0] in_elems;
    wire [31:0] out_elems;
    wire [31:0] total_elems;
    wire [31:0] lanes;
    wire [31:0] window;
    wire [31:0] mac_cycles;
    wire [31:0] compute_cycles;
    wire [31:0] fill_cycles;
    wire        skip_l1_write;
    wire [31:0] layer_index_desc;
    wire [31:0] layer_ref_off;
    wire [31:0] layer_ref_size;

    wire        udma_direction_write;
    wire [31:0] udma_dram_read_bytes;
    wire [31:0] udma_codec_cycles;

    wire        l1_req_write;
    wire        l1mgr_req_l1;
    wire [3:0]  l1mgr_req_source;
    wire [7:0]  l1mgr_req_tid;
    wire [31:0] l1mgr_payload_cycles;
    wire [21:0] l1mesh_addr;
    wire [31:0] l1mesh_route_cycles;
    wire [127:0] l1mesh_wdata;
    wire [15:0] l1mesh_wstrb;
    wire [127:0] l1mesh_rdata;

    wire        top_done_valid;
    wire        top_done_ready;
    wire        top_busy;
    wire [3:0]  active_op_class;
    wire [3:0]  active_phase_id;
    wire [31:0] active_remaining_cycles;
    wire [8:0]  block_busy;
    wire [8:0]  block_done_valid;
    wire [31:0] datapath_crc;
    wire        datapath_ok;
    /* verilator lint_on UNUSEDSIGNAL */

    wire        dram_req_valid;
    wire        dram_req_ready;
    wire        dram_req_write;
    wire [31:0] dram_req_addr;
    wire [31:0] dram_req_wdata;
    wire [3:0]  dram_req_wstrb;
    wire        dram_resp_valid;
    wire        dram_resp_ready;
    wire [31:0] dram_resp_rdata;

    wire        test_done;
    wire        test_fail;

    always #5 clk = ~clk;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            cycle_count <= 64'd0;
        else
            cycle_count <= cycle_count + 64'd1;
    end

    host u_host (
        .clk(clk),
        .rst_n(rst_n),
        .desc_valid(desc_valid),
        .desc_ready(desc_ready),
        .desc_op_class(desc_op_class),
        .desc_wait_count(desc_wait_count),
        .desc_layer_id(desc_layer_id),
        .desc_microblock_id(desc_microblock_id),
        .desc_stream_slot(desc_stream_slot),
        .desc_stream_meta_flags(desc_stream_meta_flags),
        .cfg_write_cycles(cfg_write_cycles),
        .bytes(bytes),
        .read_bytes(read_bytes),
        .write_bytes(write_bytes),
        .act_bytes(act_bytes),
        .wgt_bytes(wgt_bytes),
        .in_elems(in_elems),
        .out_elems(out_elems),
        .total_elems(total_elems),
        .lanes(lanes),
        .window(window),
        .mac_cycles(mac_cycles),
        .compute_cycles(compute_cycles),
        .fill_cycles(fill_cycles),
        .skip_l1_write(skip_l1_write),
        .layer_index_desc(layer_index_desc),
        .layer_ref_off(layer_ref_off),
        .layer_ref_size(layer_ref_size),
        .udma_direction_write(udma_direction_write),
        .udma_dram_read_bytes(udma_dram_read_bytes),
        .udma_codec_cycles(udma_codec_cycles),
        .l1_req_write(l1_req_write),
        .l1mgr_req_l1(l1mgr_req_l1),
        .l1mgr_req_source(l1mgr_req_source),
        .l1mgr_req_tid(l1mgr_req_tid),
        .l1mgr_payload_cycles(l1mgr_payload_cycles),
        .l1mesh_addr(l1mesh_addr),
        .l1mesh_route_cycles(l1mesh_route_cycles),
        .l1mesh_wdata(l1mesh_wdata),
        .l1mesh_wstrb(l1mesh_wstrb),
        .top_done_valid(top_done_valid),
        .top_done_ready(top_done_ready),
        .top_busy(top_busy),
        .active_op_class(active_op_class),
        .active_phase_id(active_phase_id),
        .active_remaining_cycles(active_remaining_cycles),
        .block_busy(block_busy),
        .block_done_valid(block_done_valid),
        .top_datapath_crc(datapath_crc),
        .top_datapath_ok(datapath_ok),
        .dram_req_valid(dram_req_valid),
        .dram_req_ready(dram_req_ready),
        .dram_req_write(dram_req_write),
        .dram_req_addr(dram_req_addr),
        .dram_req_wdata(dram_req_wdata),
        .dram_req_wstrb(dram_req_wstrb),
        .dram_resp_valid(dram_resp_valid),
        .dram_resp_ready(dram_resp_ready),
        .dram_resp_rdata(dram_resp_rdata),
        .test_done(test_done),
        .test_fail(test_fail)
    );

    dram u_dram (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(dram_req_valid),
        .req_ready(dram_req_ready),
        .req_write(dram_req_write),
        .req_addr(dram_req_addr),
        .req_wdata(dram_req_wdata),
        .req_wstrb(dram_req_wstrb),
        .resp_valid(dram_resp_valid),
        .resp_ready(dram_resp_ready),
        .resp_rdata(dram_resp_rdata)
    );

    mdla7_top u_mdla7_top (
        .clk(clk),
        .rst_n(rst_n),
        .desc_valid(desc_valid),
        .desc_ready(desc_ready),
        .desc_op_class(desc_op_class),
        .desc_wait_count(desc_wait_count),
        .desc_layer_id(desc_layer_id),
        .desc_microblock_id(desc_microblock_id),
        .desc_stream_slot(desc_stream_slot),
        .desc_stream_meta_flags(desc_stream_meta_flags),
        .cfg_write_cycles(cfg_write_cycles),
        .bytes(bytes),
        .read_bytes(read_bytes),
        .write_bytes(write_bytes),
        .act_bytes(act_bytes),
        .wgt_bytes(wgt_bytes),
        .in_elems(in_elems),
        .out_elems(out_elems),
        .total_elems(total_elems),
        .lanes(lanes),
        .window(window),
        .mac_cycles(mac_cycles),
        .compute_cycles(compute_cycles),
        .fill_cycles(fill_cycles),
        .skip_l1_write(skip_l1_write),
        .layer_index(layer_index_desc),
        .ref_off(layer_ref_off),
        .ref_size(layer_ref_size),
        .udma_direction_write(udma_direction_write),
        .udma_dram_read_bytes(udma_dram_read_bytes),
        .udma_codec_cycles(udma_codec_cycles),
        .l1_req_write(l1_req_write),
        .l1mgr_req_l1(l1mgr_req_l1),
        .l1mgr_req_source(l1mgr_req_source),
        .l1mgr_req_tid(l1mgr_req_tid),
        .l1mgr_payload_cycles(l1mgr_payload_cycles),
        .l1mesh_addr(l1mesh_addr),
        .l1mesh_route_cycles(l1mesh_route_cycles),
        .l1mesh_wdata(l1mesh_wdata),
        .l1mesh_wstrb(l1mesh_wstrb),
        .l1mesh_rdata(l1mesh_rdata),
        .done_valid(top_done_valid),
        .done_ready(top_done_ready),
        .busy(top_busy),
        .active_op_class(active_op_class),
        .active_phase_id(active_phase_id),
        .active_remaining_cycles(active_remaining_cycles),
        .block_busy(block_busy),
        .block_done_valid(block_done_valid),
        .datapath_crc(datapath_crc),
        .datapath_ok(datapath_ok)
    );

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
    end

    initial begin
        repeat (20000000) @(posedge clk);
        $display("FAIL: Testbench timeout");
        $display("VERILOG_CYCLES: %0d", cycle_count);
        $finish;
    end

    initial begin
        wait (test_done);
        if (test_fail) begin
            $display("FAIL: Testbench host/mdla7_top/dram control+datapath path");
        end else begin
            $display("PASS: Testbench host/mdla7_top/dram control+datapath path");
        end
        $display("VERILOG_CYCLES: %0d", cycle_count);
        $finish;
    end
endmodule
