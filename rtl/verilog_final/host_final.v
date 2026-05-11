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
    output reg [127:0] conv_act_vec,
    output reg [127:0] conv_wgt_vec,
    output reg [7:0]   conv_elem_count,
    output reg         conv_fp_mode,
    output reg         conv_int16_mode,
    output reg signed [15:0] conv_zp_in,
    output reg signed [31:0] conv_bias,
    output reg signed [31:0] conv_multiplier,
    output reg signed [7:0]  conv_shift,
    output reg signed [31:0] conv_zp_out,
    output reg signed [31:0] conv_act_min,
    output reg signed [31:0] conv_act_max,
    output reg signed [31:0] requant_input_value,
    output reg         pool_avg_mode,
    output reg         pool_fp_mode,
    output reg         pool_int16_mode,
    output reg [127:0] pool_sample_vec,
    output reg [7:0]   pool_elem_count,
    output reg [1:0]   ewe_op_mode,
    output reg         ewe_fp_mode,
    output reg         ewe_int16_mode,
    output reg [127:0] ewe_a_vec,
    output reg [127:0] ewe_b_vec,
    output reg [7:0]   ewe_elem_count,

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
    input signed [31:0] conv_acc_out,
    input signed [31:0] conv_scaled_out,
    input signed [7:0]  conv_out_q,
    input      [63:0]   conv_fp_sum_bits,
    input signed [31:0] conv_int16_acc_out,
    input signed [31:0] requant_scaled_out,
    input signed [7:0]  requant_out_q,
    input signed [31:0] pool_out,
    input signed [7:0]  pool_out_q,
    input      [63:0]   pool_fp_bits,
    input signed [31:0] ewe_out,
    input signed [7:0]  ewe_out_q,
    input      [63:0]   ewe_fp_bits,
    input      [8:0]   block_busy,
    input      [8:0]   block_done_valid,

    output reg         test_done,
    output reg         test_fail,
    output reg [31:0]  issued_count,
    output reg [31:0]  done_count
);
    localparam [3:0] OP_DONE = 4'd0;
    localparam [3:0] OP_CONV = 4'd1;
    localparam [3:0] OP_REQUANT = 4'd2;
    localparam [3:0] OP_EWE = 4'd3;
    localparam [3:0] OP_POOL = 4'd4;
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
            conv_act_vec <= {cmd_mem[base + 7], cmd_mem[base + 6],
                             cmd_mem[base + 5], cmd_mem[base + 4]};
            conv_wgt_vec <= {cmd_mem[base + 11], cmd_mem[base + 10],
                             cmd_mem[base + 9], cmd_mem[base + 8]};
            conv_elem_count <= cmd_mem[base + 12][7:0];
            conv_fp_mode <= cmd_mem[base + 12][8];
            conv_int16_mode <= cmd_mem[base + 12][11];
            conv_zp_in <= cmd_mem[base + 12][31:16];
            conv_bias <= cmd_mem[base + 13];
            conv_multiplier <= cmd_mem[base + 14];
            conv_shift <= cmd_mem[base + 15][7:0];
            conv_zp_out <= {{24{cmd_mem[base + 15][15]}}, cmd_mem[base + 15][15:8]};
            conv_act_min <= cmd_mem[base + 16];
            conv_act_max <= cmd_mem[base + 17];
            requant_input_value <= cmd_mem[base + 4];
            pool_sample_vec <= {cmd_mem[base + 7], cmd_mem[base + 6],
                                cmd_mem[base + 5], cmd_mem[base + 4]};
            pool_elem_count <= cmd_mem[base + 12][7:0];
            pool_avg_mode <= cmd_mem[base + 12][8];
            pool_fp_mode <= cmd_mem[base + 12][9];
            pool_int16_mode <= cmd_mem[base + 12][11];
            ewe_a_vec <= {cmd_mem[base + 7], cmd_mem[base + 6],
                          cmd_mem[base + 5], cmd_mem[base + 4]};
            ewe_b_vec <= {cmd_mem[base + 11], cmd_mem[base + 10],
                          cmd_mem[base + 9], cmd_mem[base + 8]};
            ewe_elem_count <= cmd_mem[base + 12][7:0];
            ewe_op_mode <= cmd_mem[base + 12][9:8];
            ewe_fp_mode <= cmd_mem[base + 12][10];
            ewe_int16_mode <= cmd_mem[base + 12][11];
            l1mesh_wdata <= {cmd_mem[base + 2], cmd_mem[base + 1],
                             cmd_mem[base + 14], cmd_mem[base + 15]};
            l1mesh_wstrb <= 16'hffff;
        end
    endtask

    initial begin
        for (load_i = 0; load_i < MAX_COMMANDS * WORDS_PER_COMMAND; load_i = load_i + 1)
            cmd_mem[load_i] = 32'd0;

        // Command 0: CONV sample MAC.
        cmd_mem[0] = {28'd0, OP_CONV};
        cmd_mem[1] = 32'd16;
        cmd_mem[2] = 32'h0000_02a0;
        cmd_mem[4] = 32'h0102_0304;
        cmd_mem[5] = 32'hfc07_0000;
        cmd_mem[8] = 32'h0201_ff03;
        cmd_mem[9] = 32'h0506_0000;
        cmd_mem[12] = 32'd6;
        cmd_mem[13] = 32'd5;
        cmd_mem[14] = 32'sd1073741824;
        cmd_mem[15] = 32'd1;
        cmd_mem[16] = -32'sd128;
        cmd_mem[17] = 32'sd127;
        cmd_mem[18] = 32'd18;

        // Command 1: REQUANT sample using the CONV raw accumulator.
        cmd_mem[20] = {28'd0, OP_REQUANT};
        cmd_mem[21] = 32'd1;
        cmd_mem[22] = 32'h0000_02b0;
        cmd_mem[24] = 32'd18;
        cmd_mem[34] = 32'sd1073741824;
        cmd_mem[35] = 32'd1;
        cmd_mem[36] = -32'sd128;
        cmd_mem[37] = 32'sd127;
        cmd_mem[38] = 32'd18;

        // Command 2: POOL sample max over 7 INT8 elements.
        cmd_mem[40] = {28'd0, OP_POOL};
        cmd_mem[41] = 32'd7;
        cmd_mem[42] = 32'h0000_02c0;
        cmd_mem[44] = 32'h0102_0304;
        cmd_mem[45] = 32'hfc07_0000;
        cmd_mem[52] = 32'd7;
        cmd_mem[58] = 32'd7;

        // Command 3: EWE sample add over 4 INT8 elements.
        cmd_mem[60] = {28'd0, OP_EWE};
        cmd_mem[61] = 32'd4;
        cmd_mem[62] = 32'h0000_02d0;
        cmd_mem[64] = 32'h0102_0304;
        cmd_mem[68] = 32'h0201_ff03;
        cmd_mem[72] = 32'd4;
        cmd_mem[78] = 32'd15;

        // Command 4: UDMA read path.
        cmd_mem[80] = {28'd0, OP_UDMA};
        cmd_mem[81] = 32'd256;
        cmd_mem[82] = 32'h0000_02a0;
        cmd_mem[83] = 32'd0;
        cmd_mem[84] = 32'd512;
        cmd_mem[85] = 32'd3;

        // Command 5: TNPS space-to-depth over a 4x4x1 tensor, block=2.
        cmd_mem[100] = {28'd0, OP_TNPS};
        cmd_mem[101] = 32'd128;
        cmd_mem[102] = 32'h0000_03f0;
        cmd_mem[103] = 32'd2;
        cmd_mem[106] = 32'd4;
        cmd_mem[107] = 32'd4;
        cmd_mem[108] = 32'd1;
        cmd_mem[109] = 32'd2;
        cmd_mem[110] = 32'd2;
        cmd_mem[111] = 32'd4;
        cmd_mem[112] = 32'd2;
        cmd_mem[113] = 32'd1;
        cmd_mem[114] = 32'd2;
        cmd_mem[115] = 32'd0;
        cmd_mem[116] = 32'd4;
        cmd_mem[117] = 32'd2;
        cmd_mem[118] = 32'd1;

        // Command 6: stop.
        cmd_mem[120] = {28'd0, OP_DONE};

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
            conv_act_vec <= 128'd0;
            conv_wgt_vec <= 128'd0;
            conv_elem_count <= 8'd0;
            conv_fp_mode <= 1'b0;
            conv_int16_mode <= 1'b0;
            conv_zp_in <= 16'sd0;
            conv_bias <= 32'sd0;
            conv_multiplier <= 32'sd1073741824;
            conv_shift <= 8'sd1;
            conv_zp_out <= 32'sd0;
            conv_act_min <= -32'sd128;
            conv_act_max <= 32'sd127;
            requant_input_value <= 32'sd0;
            pool_avg_mode <= 1'b0;
            pool_fp_mode <= 1'b0;
            pool_int16_mode <= 1'b0;
            pool_sample_vec <= 128'd0;
            pool_elem_count <= 8'd0;
            ewe_op_mode <= 2'd0;
            ewe_fp_mode <= 1'b0;
            ewe_int16_mode <= 1'b0;
            ewe_a_vec <= 128'd0;
            ewe_b_vec <= 128'd0;
            ewe_elem_count <= 8'd0;
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
                        if ((desc_op_class == OP_CONV) && conv_fp_mode &&
                            (conv_fp_sum_bits !== {cmd_mem[base + 17], cmd_mem[base + 16]})) begin
                            $display("HOST_FINAL_FAIL: CONV FP sample cmd=%0d got=%016x expected=%016x",
                                     command_index, conv_fp_sum_bits,
                                     {cmd_mem[base + 17], cmd_mem[base + 16]});
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_CONV) && conv_int16_mode &&
                            (conv_int16_acc_out !== $signed(cmd_mem[base + 18]))) begin
                            $display("HOST_FINAL_FAIL: CONV INT16 sample cmd=%0d acc=%0d expected=%0d",
                                     command_index, conv_int16_acc_out,
                                     $signed(cmd_mem[base + 18]));
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            (conv_out_q !== cmd_mem[base + 18][7:0])) begin
                            $display("HOST_FINAL_FAIL: CONV sample cmd=%0d acc=%0d scaled=%0d out=%0d expected=%0d",
                                     command_index, conv_acc_out, conv_scaled_out,
                                     conv_out_q, $signed(cmd_mem[base + 18][7:0]));
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_REQUANT) &&
                            (requant_out_q !== cmd_mem[base + 18][7:0])) begin
                            $display("HOST_FINAL_FAIL: REQUANT sample cmd=%0d scaled=%0d out=%0d expected=%0d",
                                     command_index, requant_scaled_out,
                                     requant_out_q, $signed(cmd_mem[base + 18][7:0]));
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_POOL) && pool_fp_mode &&
                            (pool_fp_bits !== {cmd_mem[base + 17], cmd_mem[base + 16]})) begin
                            $display("HOST_FINAL_FAIL: POOL FP sample cmd=%0d got=%016x expected=%016x avg=%0d",
                                     command_index, pool_fp_bits,
                                     {cmd_mem[base + 17], cmd_mem[base + 16]},
                                     pool_avg_mode);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_POOL) && pool_int16_mode &&
                            (pool_out !== $signed(cmd_mem[base + 18]))) begin
                            $display("HOST_FINAL_FAIL: POOL INT16 sample cmd=%0d out=%0d expected=%0d avg=%0d",
                                     command_index, pool_out,
                                     $signed(cmd_mem[base + 18]),
                                     pool_avg_mode);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_POOL) && !pool_fp_mode && !pool_int16_mode &&
                            (pool_out_q !== cmd_mem[base + 18][7:0])) begin
                            $display("HOST_FINAL_FAIL: POOL sample cmd=%0d out=%0d expected=%0d avg=%0d",
                                     command_index, pool_out,
                                     $signed(cmd_mem[base + 18][7:0]),
                                     pool_avg_mode);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_EWE) && ewe_fp_mode &&
                            (ewe_fp_bits !== {cmd_mem[base + 17], cmd_mem[base + 16]})) begin
                            $display("HOST_FINAL_FAIL: EWE FP sample cmd=%0d got=%016x expected=%016x mode=%0d",
                                     command_index, ewe_fp_bits,
                                     {cmd_mem[base + 17], cmd_mem[base + 16]},
                                     ewe_op_mode);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_EWE) && !ewe_fp_mode &&
                            (ewe_out !== $signed(cmd_mem[base + 18]))) begin
                            $display("HOST_FINAL_FAIL: EWE vector sample cmd=%0d sum=%0d first=%0d expected_sum=%0d mode=%0d",
                                     command_index, ewe_out, ewe_out_q,
                                     $signed(cmd_mem[base + 18]),
                                     ewe_op_mode);
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
