`timescale 1ns/1ps

module tb_mdla7_top_control;
    reg clk;
    reg rst_n;

    reg         desc_valid;
    wire        desc_ready;
    reg  [3:0]  desc_op_class;
    reg  [7:0]  desc_wait_count;
    reg  [15:0] desc_layer_id;
    reg  [15:0] desc_microblock_id;
    reg  [7:0]  desc_stream_slot;
    reg  [7:0]  desc_stream_meta_flags;
    reg  [31:0] cfg_write_cycles;

    reg  [31:0] bytes;
    reg  [31:0] read_bytes;
    reg  [31:0] write_bytes;
    reg  [31:0] act_bytes;
    reg  [31:0] wgt_bytes;
    reg  [31:0] in_elems;
    reg  [31:0] out_elems;
    reg  [31:0] total_elems;
    reg  [31:0] lanes;
    reg  [31:0] window;
    reg  [31:0] mac_cycles;
    reg  [31:0] compute_cycles;
    reg  [31:0] fill_cycles;
    reg         skip_l1_write;
    reg  [31:0] layer_index;
    reg  [31:0] ref_off;
    reg  [31:0] ref_size;

    reg         udma_direction_write;
    reg  [31:0] udma_dram_read_bytes;
    reg  [31:0] udma_codec_cycles;

    reg         l1_req_write;
    reg         l1mgr_req_l1;
    reg  [3:0]  l1mgr_req_source;
    reg  [7:0]  l1mgr_req_tid;
    reg  [31:0] l1mgr_payload_cycles;
    reg  [21:0] l1mesh_addr;
    reg  [31:0] l1mesh_route_cycles;
    reg  [127:0] l1mesh_wdata;
    reg  [15:0] l1mesh_wstrb;

    /* verilator lint_off UNUSEDSIGNAL */
    wire [127:0] l1mesh_rdata;
    wire         done_valid;
    reg          done_ready;
    wire         busy;
    wire [3:0]   active_op_class;
    wire [3:0]   active_phase_id;
    wire [31:0]  active_remaining_cycles;
    wire [8:0]   block_busy;
    wire [8:0]   block_done_valid;
    wire [31:0]  datapath_crc;
    wire         datapath_ok;
    /* verilator lint_on UNUSEDSIGNAL */

    integer timeout;
    integer errors;

    always #5 clk = ~clk;

    mdla7_top dut (
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
        .layer_index(layer_index),
        .ref_off(ref_off),
        .ref_size(ref_size),
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
        .done_valid(done_valid),
        .done_ready(done_ready),
        .busy(busy),
        .active_op_class(active_op_class),
        .active_phase_id(active_phase_id),
        .active_remaining_cycles(active_remaining_cycles),
        .block_busy(block_busy),
        .block_done_valid(block_done_valid),
        .datapath_crc(datapath_crc),
        .datapath_ok(datapath_ok)
    );

    task fail;
        input [511:0] msg;
        begin
            $display("FAIL: %0s at t=%0t", msg, $time);
            errors = errors + 1;
        end
    endtask

    task init_inputs;
        begin
            desc_valid = 1'b0;
            desc_op_class = 4'd0;
            desc_wait_count = 8'd0;
            desc_layer_id = 16'd0;
            desc_microblock_id = 16'd0;
            desc_stream_slot = 8'd0;
            desc_stream_meta_flags = 8'd0;
            cfg_write_cycles = 32'd2;
            bytes = 32'd256;
            read_bytes = 32'd192;
            write_bytes = 32'd160;
            act_bytes = 32'd384;
            wgt_bytes = 32'd512;
            in_elems = 32'd128;
            out_elems = 32'd96;
            total_elems = 32'd96;
            lanes = 32'd16;
            window = 32'd4;
            mac_cycles = 32'd7;
            compute_cycles = 32'd5;
            fill_cycles = 32'd6;
            skip_l1_write = 1'b0;
            layer_index = 32'd0;
            ref_off = 32'd0;
            ref_size = 32'd0;
            udma_direction_write = 1'b0;
            udma_dram_read_bytes = 32'd288;
            udma_codec_cycles = 32'd3;
            l1_req_write = 1'b0;
            l1mgr_req_l1 = 1'b1;
            l1mgr_req_source = 4'd3;
            l1mgr_req_tid = 8'd9;
            l1mgr_payload_cycles = 32'd4;
            l1mesh_addr = 22'd16;
            l1mesh_route_cycles = 32'd3;
            l1mesh_wdata = 128'h0123_4567_89ab_cdef_0123_4567_89ab_cdef;
            l1mesh_wstrb = 16'hffff;
            done_ready = 1'b1;
        end
    endtask

    task issue_and_wait;
        input [3:0] op_class;
        input [8:0] expected_done_bit;
        reg saw_busy;
        reg saw_block_done;
        begin
            saw_busy = 1'b0;
            saw_block_done = 1'b0;
            @(posedge clk);
            if (!desc_ready)
                fail("desc_ready low before issue");
            desc_op_class = op_class;
            desc_valid = 1'b1;
            @(posedge clk);
            desc_valid = 1'b0;
            #1;
            if (!busy)
                fail("busy did not assert after descriptor issue");

            timeout = 0;
            while (!done_valid && timeout < 300) begin
                if (expected_done_bit != 9'd0 && (block_done_valid & expected_done_bit) != 9'd0)
                    saw_block_done = 1'b1;
                @(posedge clk);
                #1;
                timeout = timeout + 1;
                if (busy)
                    saw_busy = 1'b1;
            end
            if (expected_done_bit != 9'd0 && (block_done_valid & expected_done_bit) != 9'd0)
                saw_block_done = 1'b1;
            if (!done_valid)
                fail("top done_valid timeout");
            if (!saw_busy)
                fail("busy was never observed during transaction");
            if (active_op_class !== op_class)
                fail("active_op_class mismatch");
            if (expected_done_bit != 9'd0 && !saw_block_done)
                fail("selected block_done_valid bit was not high at completion");

            @(posedge clk);
            #1;
            if (!desc_ready)
                fail("desc_ready did not return after done");
        end
    endtask

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        errors = 0;
        init_inputs();
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        issue_and_wait(4'd0, 9'b000000001);
        issue_and_wait(4'd1, 9'b000000010);
        issue_and_wait(4'd2, 9'b000000100);
        issue_and_wait(4'd3, 9'b000001000);
        issue_and_wait(4'd4, 9'b000010000);
        issue_and_wait(4'd5, 9'b000100000);
        issue_and_wait(4'd6, 9'b001000000);
        issue_and_wait(4'd7, 9'b010000000);
        issue_and_wait(4'd8, 9'b100000000);

        if (errors == 0) begin
            $display("PASS: mdla7_top control path");
            $finish;
        end
        $display("FAIL: mdla7_top control path errors=%0d", errors);
        $finish;
    end
endmodule
