`timescale 1ns/1ps

module host_final #(
    parameter MAX_COMMANDS = 64
) (
    input              clk,
    input              rst_n,

    output reg         desc_valid,
    input              desc_ready,
    output reg [3:0]   desc_op_class,
    output reg [31:0]  bytes,
    output reg [31:0]  udma_dram_read_bytes,
    output reg [31:0]  udma_codec_cycles,
    output reg         udma_direction_write,
    output reg [21:0]  l1mesh_addr,
    output reg [127:0] l1mesh_wdata,
    output reg [15:0]  l1mesh_wstrb,

    output reg         tnps_mode_space_to_depth,
    output reg [15:0]  tnps_in_h,
    output reg [15:0]  tnps_in_w,
    output reg [15:0]  tnps_in_c,
    output reg [15:0]  tnps_out_h,
    output reg [15:0]  tnps_out_w,
    output reg [15:0]  tnps_out_c,
    output reg [15:0]  tnps_block,
    output reg [1:0]   tnps_elem_bytes,
    output reg [31:0]  tnps_sample_out_elem_index,
    output reg [31:0]  tnps_sample_in_elem_index,

    input              top_done_valid,
    output             top_done_ready,
    input              top_busy,
    input      [3:0]   active_op_class,
    input      [3:0]   active_phase_id,
    input      [31:0]  active_remaining_cycles,
    input      [31:0]  placement_route_cycles,
    input      [31:0]  tnps_sample_src_byte_offset,
    input      [31:0]  tnps_sample_dst_byte_offset,
    input              tnps_sample_valid,
    input      [8:0]   block_busy,
    input      [8:0]   block_done_valid,

    output reg         test_done,
    output reg         test_fail,
    output reg [31:0]  issued_count,
    output reg [31:0]  done_count
);
    localparam [3:0] OP_DONE = 4'd0;
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;

    localparam [2:0] ST_LOAD  = 3'd0;
    localparam [2:0] ST_ISSUE = 3'd1;
    localparam [2:0] ST_WAIT  = 3'd2;
    localparam [2:0] ST_NEXT  = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;

    localparam WORDS_PER_COMMAND = 20;

    reg [2:0] state;
    reg [31:0] command_index;
    reg [31:0] watchdog;
    reg [31:0] cmd_mem [0:MAX_COMMANDS*WORDS_PER_COMMAND-1];
    reg [1023:0] program_path;
    integer load_i;

    wire [31:0] base = command_index * WORDS_PER_COMMAND;
    wire [3:0] next_op = cmd_mem[base][3:0];

    assign top_done_ready = 1'b1;

    task load_command;
        begin
            desc_op_class <= cmd_mem[base][3:0];
            bytes <= cmd_mem[base + 1];
            l1mesh_addr <= cmd_mem[base + 2][21:0];
            udma_direction_write <= cmd_mem[base + 3][0];
            tnps_mode_space_to_depth <= cmd_mem[base + 3][1];
            udma_dram_read_bytes <= cmd_mem[base + 4];
            udma_codec_cycles <= cmd_mem[base + 5];
            tnps_in_h <= cmd_mem[base + 6][15:0];
            tnps_in_w <= cmd_mem[base + 7][15:0];
            tnps_in_c <= cmd_mem[base + 8][15:0];
            tnps_out_h <= cmd_mem[base + 9][15:0];
            tnps_out_w <= cmd_mem[base + 10][15:0];
            tnps_out_c <= cmd_mem[base + 11][15:0];
            tnps_block <= cmd_mem[base + 12][15:0];
            tnps_elem_bytes <= cmd_mem[base + 13][1:0];
            tnps_sample_out_elem_index <= cmd_mem[base + 14];
            tnps_sample_in_elem_index <= cmd_mem[base + 15];
            l1mesh_wdata <= {cmd_mem[base + 2], cmd_mem[base + 1],
                             cmd_mem[base + 14], cmd_mem[base + 15]};
            l1mesh_wstrb <= 16'hffff;
        end
    endtask

    initial begin
        for (load_i = 0; load_i < MAX_COMMANDS * WORDS_PER_COMMAND; load_i = load_i + 1)
            cmd_mem[load_i] = 32'd0;

        // Command 0: UDMA read path.
        cmd_mem[0] = {28'd0, OP_UDMA};
        cmd_mem[1] = 32'd256;
        cmd_mem[2] = 32'h0000_02a0;
        cmd_mem[3] = 32'd0;
        cmd_mem[4] = 32'd512;
        cmd_mem[5] = 32'd3;

        // Command 1: TNPS space-to-depth over a 4x4x1 tensor, block=2.
        cmd_mem[20] = {28'd0, OP_TNPS};
        cmd_mem[21] = 32'd128;
        cmd_mem[22] = 32'h0000_03f0;
        cmd_mem[23] = 32'd2;
        cmd_mem[26] = 32'd4;
        cmd_mem[27] = 32'd4;
        cmd_mem[28] = 32'd1;
        cmd_mem[29] = 32'd2;
        cmd_mem[30] = 32'd2;
        cmd_mem[31] = 32'd4;
        cmd_mem[32] = 32'd2;
        cmd_mem[33] = 32'd1;
        cmd_mem[34] = 32'd2;
        cmd_mem[35] = 32'd0;
        cmd_mem[36] = 32'd4;
        cmd_mem[37] = 32'd2;
        cmd_mem[38] = 32'd1;

        // Command 2: stop.
        cmd_mem[40] = {28'd0, OP_DONE};

        program_path = "";
        if ($value$plusargs("FINAL_PROGRAM=%s", program_path))
            $readmemh(program_path, cmd_mem);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_LOAD;
            command_index <= 32'd0;
            watchdog <= 32'd0;
            desc_valid <= 1'b0;
            desc_op_class <= 4'd0;
            bytes <= 32'd0;
            udma_dram_read_bytes <= 32'd0;
            udma_codec_cycles <= 32'd0;
            udma_direction_write <= 1'b0;
            l1mesh_addr <= 22'd0;
            l1mesh_wdata <= 128'd0;
            l1mesh_wstrb <= 16'd0;
            tnps_mode_space_to_depth <= 1'b1;
            tnps_in_h <= 16'd0;
            tnps_in_w <= 16'd0;
            tnps_in_c <= 16'd0;
            tnps_out_h <= 16'd0;
            tnps_out_w <= 16'd0;
            tnps_out_c <= 16'd0;
            tnps_block <= 16'd0;
            tnps_elem_bytes <= 2'd1;
            tnps_sample_out_elem_index <= 32'd0;
            tnps_sample_in_elem_index <= 32'd0;
            test_done <= 1'b0;
            test_fail <= 1'b0;
            issued_count <= 32'd0;
            done_count <= 32'd0;
        end else begin
            case (state)
                ST_LOAD: begin
                    desc_valid <= 1'b0;
                    watchdog <= 32'd0;
                    if (next_op == OP_DONE) begin
                        test_done <= 1'b1;
                        state <= ST_DONE;
                    end else begin
                        load_command();
                        state <= ST_ISSUE;
                    end
                end
                ST_ISSUE: begin
                    if (!desc_valid)
                        desc_valid <= 1'b1;
                    if (desc_valid && desc_ready) begin
                        desc_valid <= 1'b0;
                        issued_count <= issued_count + 32'd1;
                        state <= ST_WAIT;
                    end
                end
                ST_WAIT: begin
                    watchdog <= watchdog + 32'd1;
                    if (watchdog == 32'd5000000) begin
                        $display("HOST_FINAL_FAIL: timeout cmd=%0d op=%0d active=%0d phase=%0d remaining=%0d top_busy=%0d block_busy=%09b block_done=%09b",
                                 command_index, desc_op_class, active_op_class,
                                 active_phase_id, active_remaining_cycles, top_busy,
                                 block_busy, block_done_valid);
                        test_fail <= 1'b1;
                        test_done <= 1'b1;
                        state <= ST_DONE;
                    end else if (top_done_valid) begin
                        if (placement_route_cycles == 32'd0) begin
                            $display("HOST_FINAL_FAIL: zero route cycles cmd=%0d op=%0d",
                                     command_index, desc_op_class);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_TNPS) &&
                            ((tnps_sample_valid != (cmd_mem[base + 18] != 32'd0)) ||
                             ((cmd_mem[base + 18] != 32'd0) &&
                              ((tnps_sample_src_byte_offset != cmd_mem[base + 16]) ||
                               (tnps_sample_dst_byte_offset != cmd_mem[base + 17]))))) begin
                            $display("HOST_FINAL_FAIL: TNPS sample cmd=%0d valid=%0d src=%0d dst=%0d",
                                     command_index, tnps_sample_valid,
                                     tnps_sample_src_byte_offset,
                                     tnps_sample_dst_byte_offset);
                            test_fail <= 1'b1;
                        end
                        done_count <= done_count + 32'd1;
                        state <= ST_NEXT;
                    end
                end
                ST_NEXT: begin
                    command_index <= command_index + 32'd1;
                    state <= ST_LOAD;
                end
                ST_DONE: begin
                    desc_valid <= 1'b0;
                    test_done <= 1'b1;
                end
                default: begin
                    state <= ST_LOAD;
                    desc_valid <= 1'b0;
                end
            endcase
        end
    end
endmodule
