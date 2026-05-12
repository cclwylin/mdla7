`timescale 1ns/1ps

module host_final #(
    parameter MAX_COMMANDS = 4096
) (
    input              clk,
    input              rst_n,

    output reg         desc_valid,
    input              desc_ready,
    output reg [3:0]   desc_op_class,
    output reg [15:0]  desc_layer_id,
    output reg [15:0]  desc_microblock_id,
    output reg [7:0]   desc_stream_slot,
    output reg [7:0]   desc_stream_meta_flags,
    output reg [31:0]  bytes,
    output reg [31:0]  udma_dram_read_bytes,
    output reg [31:0]  udma_codec_cycles,
    output reg         udma_direction_write,
    output reg         udma_final_write_mode,
    output reg         udma_sramcrc_mode,
    output reg [7:0]   udma_input_byte,
    output reg [31:0]  udma_out_byte_offset,
    output reg [31:0]  udma_sramcrc_expected_crc,
    output reg [31:0]  udma_sramcrc_expected_count,
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
    output reg         tnps_final_write_mode,
    output reg         tnps_sramcrc_mode,
    output reg [7:0]   tnps_input_byte,
    output reg [31:0]  tnps_out_byte_offset,
    output reg [31:0]  tnps_sramcrc_expected_crc,
    output reg [31:0]  tnps_sramcrc_expected_count,
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
    output reg [15:0]  conv_in_h,
    output reg [15:0]  conv_in_w,
    output reg [15:0]  conv_in_c,
    output reg [15:0]  conv_out_h,
    output reg [15:0]  conv_out_w,
    output reg [15:0]  conv_out_c,
    output reg [7:0]   conv_k_h,
    output reg [7:0]   conv_k_w,
    output reg [7:0]   conv_stride_h,
    output reg [7:0]   conv_stride_w,
    output reg [7:0]   conv_dilation_h,
    output reg [7:0]   conv_dilation_w,
    output reg signed [15:0] conv_pad_top,
    output reg signed [15:0] conv_pad_left,
    output reg [1:0]   conv_elem_bytes,
    output reg [31:0]  conv_out_elem_index,
    output reg [7:0]   conv_tile_output_count,
    output reg         conv_partial_first,
    output reg         conv_partial_accumulate,
    output reg         conv_partial_final,
    output reg         conv_refcrc_mode,
    output reg         conv_sramcrc_mode,
    output reg [31:0]  conv_refcrc_expected_crc,
    output reg [31:0]  conv_refcrc_expected_count,
    output reg [31:0]  conv_refcrc_ref_off,
    output reg [15:0]  conv_sample_kh,
    output reg [15:0]  conv_sample_kw,
    output reg [15:0]  conv_sample_ic,
    output reg signed [31:0] requant_input_value,
    output reg         requant_read_input_from_l1,
    output reg         requant_sramcrc_mode,
    output reg [31:0]  requant_sramcrc_expected_crc,
    output reg [31:0]  requant_sramcrc_expected_count,
    output reg [31:0]  requant_out_byte_offset,
    output reg         pool_avg_mode,
    output reg         pool_fp_mode,
    output reg         pool_int16_mode,
    output reg         pool_read_sample_from_l1,
    output reg         pool_refcrc_mode,
    output reg         pool_sramcrc_mode,
    output reg [31:0]  pool_refcrc_expected_crc,
    output reg [31:0]  pool_refcrc_expected_count,
    output reg [31:0]  pool_refcrc_ref_off,
    output reg [31:0]  pool_out_byte_offset,
    output reg [127:0] pool_sample_vec,
    output reg [7:0]   pool_elem_count,
    output reg [1:0]   ewe_op_mode,
    output reg         ewe_fp_mode,
    output reg         ewe_int16_mode,
    output reg         ewe_final_q_mode,
    output reg         ewe_read_a_from_l1,
    output reg         ewe_sramcrc_mode,
    output reg [31:0]  ewe_sramcrc_expected_crc,
    output reg [31:0]  ewe_sramcrc_expected_count,
    output reg [31:0]  ewe_out_byte_offset,
    output reg [127:0] ewe_a_vec,
    output reg [127:0] ewe_b_vec,
    output reg [7:0]   ewe_elem_count,
    output reg signed [31:0] ewe_zp_a,
    output reg signed [31:0] ewe_zp_b,
    output reg signed [31:0] ewe_zp_out,
    output reg signed [31:0] ewe_mult_a,
    output reg signed [7:0]  ewe_shift_a,
    output reg signed [31:0] ewe_mult_b,
    output reg signed [7:0]  ewe_shift_b,
    output reg signed [31:0] ewe_mult_out,
    output reg signed [7:0]  ewe_shift_out,
    output reg signed [31:0] ewe_left_shift,
    output reg signed [31:0] ewe_act_min,
    output reg signed [31:0] ewe_act_max,

    input              top_done_valid,
    output             top_done_ready,
    input              top_busy,
    input      [3:0]   active_op_class,
    input      [15:0]  active_layer_id,
    input      [15:0]  active_microblock_id,
    input      [7:0]   active_stream_slot,
    input      [7:0]   active_stream_meta_flags,
    input      [3:0]   active_phase_id,
    input      [31:0]  active_remaining_cycles,
    input      [31:0]  placement_route_cycles,
    input      [31:0]  l1mesh_crc,
    input      [31:0]  l1mesh_crc_count,
    input      [31:0]  udma_sramcrc_crc,
    input      [31:0]  udma_sramcrc_count,
    input      [31:0]  tnps_sample_src_byte_offset,
    input      [31:0]  tnps_sample_dst_byte_offset,
    input              tnps_sample_valid,
    input      [31:0]  tnps_sramcrc_crc,
    input      [31:0]  tnps_sramcrc_count,
    input signed [31:0] conv_acc_out,
    input signed [31:0] conv_scaled_out,
    input signed [7:0]  conv_out_q,
    input      [63:0]   conv_fp_sum_bits,
    input signed [31:0] conv_int16_acc_out,
    input      [31:0]  conv_sample_input_byte_offset,
    input      [31:0]  conv_sample_weight_byte_offset,
    input      [31:0]  conv_sample_output_byte_offset,
    input              conv_sample_input_valid,
    input      [31:0]  conv_first_input_byte_offset,
    input      [31:0]  conv_first_weight_byte_offset,
    input      [7:0]   conv_window_valid_count,
    input      [31:0]  conv_tile_last_output_byte_offset,
    input              conv_tile_last_input_valid,
    input      [7:0]   conv_tile_last_window_valid_count,
    input      [3:0]   conv_tile_scoreboard_valid_mask,
    input signed [31:0] conv_tile_scoreboard_q_sum,
    input      [127:0] conv_tile_result_out_elem_indices,
    input      [127:0] conv_tile_result_output_byte_offsets,
    input      [127:0] conv_tile_result_acc_values,
    input      [127:0] conv_tile_result_q_values,
    input      [3:0]   conv_writeback_valid_mask,
    input      [127:0] conv_writeback_output_byte_offsets,
    input      [127:0] conv_writeback_q_values,
    input      [3:0]   conv_shadow_valid_mask,
    input      [127:0] conv_shadow_output_byte_offsets,
    input      [127:0] conv_shadow_q_values,
    input      [15:0]  conv_shadow_mem_valid_mask,
    input      [511:0] conv_shadow_mem_output_byte_offsets,
    input      [511:0] conv_shadow_mem_q_values,
    input              conv_shadow_read_valid,
    input      [31:0]  conv_shadow_read_output_byte_offset,
    input      [31:0]  conv_shadow_read_q_value,
    input      [31:0]  conv_shadow_crc,
    input      [31:0]  conv_shadow_byte_count,
    input      [3:0]   conv_psum_valid_mask,
    input      [127:0] conv_psum_acc_values,
    input signed [31:0] requant_scaled_out,
    input signed [7:0]  requant_out_q,
    input      [31:0]   requant_sramcrc_crc,
    input      [31:0]   requant_sramcrc_count,
    input signed [31:0] pool_out,
    input signed [7:0]  pool_out_q,
    input      [63:0]   pool_fp_bits,
    input      [31:0]   pool_refcrc_crc,
    input      [31:0]   pool_refcrc_count,
    input signed [31:0] ewe_out,
    input signed [7:0]  ewe_out_q,
    input      [31:0]   ewe_sramcrc_crc,
    input      [31:0]   ewe_sramcrc_count,
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
    localparam [3:0] OP_L1CRC = 4'd7;

    localparam [2:0] ST_LOAD  = 3'd0;
    localparam [2:0] ST_ISSUE = 3'd1;
    localparam [2:0] ST_WAIT  = 3'd2;
    localparam [2:0] ST_NEXT  = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;

    localparam WORDS_PER_COMMAND = 32;

    reg [2:0] state;
    reg [31:0] command_index;
    reg [31:0] watchdog;
    reg [31:0] cmd_mem [0:MAX_COMMANDS*WORDS_PER_COMMAND-1];
    reg [1023:0] program_path;
    integer load_i;

    wire [31:0] base = command_index * WORDS_PER_COMMAND;
    wire [3:0] next_op = cmd_mem[base][3:0];
    wire [7:0] expected_conv_tile_count =
        (cmd_mem[base + 31][7:0] == 8'd0) ? 8'd1 :
        (cmd_mem[base + 31][7:0] > 8'd4) ? 8'd4 :
        cmd_mem[base + 31][7:0];
    wire [3:0] expected_conv_tile_valid_mask =
        (expected_conv_tile_count == 8'd1) ? 4'b0001 :
        (expected_conv_tile_count == 8'd2) ? 4'b0011 :
        (expected_conv_tile_count == 8'd3) ? 4'b0111 : 4'b1111;
    wire [31:0] expected_conv_elem_bytes =
        {30'd0, ((cmd_mem[base + 12][8] || cmd_mem[base + 12][11]) ? 2'd2 : 2'd1)};
    wire [31:0] expected_conv_base_out_elem_index =
        (expected_conv_elem_bytes == 32'd0) ? 32'd0 : (cmd_mem[base + 27] / expected_conv_elem_bytes);
    wire signed [31:0] expected_conv_tile_q_sum =
        $signed({{24{cmd_mem[base + 18][7]}}, cmd_mem[base + 18][7:0]}) *
        $signed({24'd0, expected_conv_tile_count});
    wire [31:0] expected_conv_tile_last_index =
        expected_conv_base_out_elem_index + {24'd0, expected_conv_tile_count} - 32'd1;
    wire [31:0] expected_conv_tile_last_byte_offset =
        cmd_mem[base + 27] + (({24'd0, expected_conv_tile_count} - 32'd1) * expected_conv_elem_bytes);
    wire [3:0] expected_conv_tile_last_shadow_slot = expected_conv_tile_last_byte_offset[3:0];
    wire [31:0] expected_conv_tile_q_value =
        {{24{cmd_mem[base + 18][7]}}, cmd_mem[base + 18][7:0]};
    wire [31:0] expected_conv_tile_acc_value =
        cmd_mem[base + 3][6] ? cmd_mem[base + 19] : conv_acc_out;
    wire [31:0] expected_conv_psum_acc_value =
        (cmd_mem[base + 3][4] || cmd_mem[base + 3][5]) ? cmd_mem[base + 19] : conv_acc_out;
    wire microblock_descriptor_mode = cmd_mem[base + 3][13];

    assign top_done_ready = 1'b1;

    function [7:0] final_write_byte;
        input [3:0] op;
        begin
            case (op)
                OP_TNPS: final_write_byte = cmd_mem[base + 4][7:0];
                OP_UDMA: final_write_byte = cmd_mem[base + 6][7:0];
                default: final_write_byte = cmd_mem[base + 18][7:0];
            endcase
        end
    endfunction

    function [127:0] byte_lane_wdata;
        input [7:0] value;
        input [3:0] lane;
        begin
            byte_lane_wdata = {120'd0, value} << ({lane, 3'd0});
        end
    endfunction

    task load_command;
        begin
            desc_op_class <= cmd_mem[base][3:0];
            desc_layer_id <= cmd_mem[base][19:4];
            desc_microblock_id <= {4'd0, cmd_mem[base][31:20]};
            desc_stream_slot <= cmd_mem[base + 3][23:16];
            desc_stream_meta_flags <= cmd_mem[base + 3][31:24];
            bytes <= cmd_mem[base + 1];
            l1mesh_addr <= cmd_mem[base + 2][21:0];
            udma_direction_write <= cmd_mem[base + 3][0];
            udma_final_write_mode <= cmd_mem[base + 3][6];
            udma_sramcrc_mode <= cmd_mem[base + 3][10];
            udma_input_byte <= cmd_mem[base + 6][7:0];
            udma_out_byte_offset <= cmd_mem[base + 27];
            udma_sramcrc_expected_crc <= cmd_mem[base + 28];
            udma_sramcrc_expected_count <= cmd_mem[base + 29];
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
            tnps_final_write_mode <= cmd_mem[base + 3][6];
            tnps_sramcrc_mode <= cmd_mem[base + 3][10];
            tnps_input_byte <= cmd_mem[base + 4][7:0];
            tnps_out_byte_offset <= cmd_mem[base + 27];
            tnps_sramcrc_expected_crc <= cmd_mem[base + 28];
            tnps_sramcrc_expected_count <= cmd_mem[base + 29];
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
            conv_in_h <= cmd_mem[base + 20][15:0];
            conv_in_w <= cmd_mem[base + 20][31:16];
            conv_in_c <= cmd_mem[base + 21][15:0];
            conv_out_h <= (cmd_mem[base + 30][31:16] == 16'd0) ? 16'd1 : cmd_mem[base + 30][31:16];
            conv_out_w <= (cmd_mem[base + 24][31:16] == 16'd0) ? 16'd1 : cmd_mem[base + 24][31:16];
            conv_out_c <= cmd_mem[base + 21][31:16];
            conv_k_h <= cmd_mem[base + 22][7:0];
            conv_k_w <= cmd_mem[base + 22][15:8];
            conv_stride_h <= cmd_mem[base + 22][23:16];
            conv_stride_w <= cmd_mem[base + 22][31:24];
            conv_dilation_h <= cmd_mem[base + 23][7:0];
            conv_dilation_w <= cmd_mem[base + 23][15:8];
            conv_pad_top <= 16'sd0;
            conv_pad_left <= 16'sd0;
            conv_elem_bytes <= expected_conv_elem_bytes[1:0];
            conv_out_elem_index <= expected_conv_base_out_elem_index;
            conv_tile_output_count <= (cmd_mem[base + 31][7:0] == 8'd0) ? 8'd1 : cmd_mem[base + 31][7:0];
            conv_partial_first <= cmd_mem[base + 3][4];
            conv_partial_accumulate <= cmd_mem[base + 3][5];
            conv_partial_final <= cmd_mem[base + 3][6];
            conv_refcrc_mode <= cmd_mem[base + 3][9];
            conv_sramcrc_mode <= cmd_mem[base + 3][10];
            conv_refcrc_expected_crc <= cmd_mem[base + 28];
            conv_refcrc_expected_count <= cmd_mem[base + 29];
            conv_refcrc_ref_off <= cmd_mem[base + 25];
            conv_sample_kh <= {8'd0, cmd_mem[base + 23][23:16]};
            conv_sample_kw <= {8'd0, cmd_mem[base + 23][31:24]};
            conv_sample_ic <= cmd_mem[base + 24][15:0];
            requant_input_value <= cmd_mem[base + 4];
            requant_read_input_from_l1 <= cmd_mem[base + 3][11];
            requant_sramcrc_mode <= cmd_mem[base + 3][10];
            requant_sramcrc_expected_crc <= cmd_mem[base + 28];
            requant_sramcrc_expected_count <= cmd_mem[base + 29];
            requant_out_byte_offset <= cmd_mem[base + 27];
            pool_sample_vec <= {cmd_mem[base + 7], cmd_mem[base + 6],
                                cmd_mem[base + 5], cmd_mem[base + 4]};
            pool_elem_count <= cmd_mem[base + 12][7:0];
            pool_avg_mode <= cmd_mem[base + 12][8];
            pool_fp_mode <= cmd_mem[base + 12][9];
            pool_int16_mode <= cmd_mem[base + 12][11];
            pool_read_sample_from_l1 <= cmd_mem[base + 3][11];
            pool_refcrc_mode <= cmd_mem[base + 3][9];
            pool_sramcrc_mode <= cmd_mem[base + 3][10];
            pool_refcrc_expected_crc <= cmd_mem[base + 28];
            pool_refcrc_expected_count <= cmd_mem[base + 29];
            pool_refcrc_ref_off <= cmd_mem[base + 25];
            pool_out_byte_offset <= cmd_mem[base + 27];
            ewe_a_vec <= {cmd_mem[base + 7], cmd_mem[base + 6],
                          cmd_mem[base + 5], cmd_mem[base + 4]};
            ewe_b_vec <= {cmd_mem[base + 11], cmd_mem[base + 10],
                          cmd_mem[base + 9], cmd_mem[base + 8]};
            ewe_elem_count <= cmd_mem[base + 12][7:0];
            ewe_op_mode <= cmd_mem[base + 12][9:8];
            ewe_fp_mode <= cmd_mem[base + 12][10];
            ewe_int16_mode <= cmd_mem[base + 12][11];
            ewe_final_q_mode <= cmd_mem[base + 3][6];
            ewe_read_a_from_l1 <= cmd_mem[base + 3][11];
            ewe_sramcrc_mode <= cmd_mem[base + 3][10];
            ewe_sramcrc_expected_crc <= cmd_mem[base + 28];
            ewe_sramcrc_expected_count <= cmd_mem[base + 29];
            ewe_out_byte_offset <= cmd_mem[base + 27];
            ewe_zp_a <= cmd_mem[base + 13];
            ewe_zp_b <= cmd_mem[base + 14];
            ewe_zp_out <= cmd_mem[base + 15];
            ewe_mult_a <= cmd_mem[base + 16];
            ewe_shift_a <= cmd_mem[base + 17][7:0];
            ewe_mult_b <= cmd_mem[base + 20];
            ewe_shift_b <= cmd_mem[base + 21][7:0];
            ewe_mult_out <= cmd_mem[base + 22];
            ewe_shift_out <= cmd_mem[base + 23][7:0];
            ewe_left_shift <= cmd_mem[base + 24];
            ewe_act_min <= cmd_mem[base + 25];
            ewe_act_max <= cmd_mem[base + 26];
            if (cmd_mem[base + 3][6]) begin
                l1mesh_addr <= cmd_mem[base + 27][21:0];
                l1mesh_wdata <= byte_lane_wdata(final_write_byte(cmd_mem[base][3:0]),
                                                cmd_mem[base + 27][3:0]);
                l1mesh_wstrb <= 16'h0001 << cmd_mem[base + 27][3:0];
            end else begin
                l1mesh_wdata <= {cmd_mem[base + 2], cmd_mem[base + 1],
                                 cmd_mem[base + 14], cmd_mem[base + 15]};
                l1mesh_wstrb <= 16'hffff;
            end
        end
    endtask

    initial begin
        for (load_i = 0; load_i < MAX_COMMANDS * WORDS_PER_COMMAND; load_i = load_i + 1)
            cmd_mem[load_i] = 32'd0;

        // Command 0: CONV sample MAC, first partial-K tile.
        cmd_mem[0] = {28'd0, OP_CONV};
        cmd_mem[1] = 32'd16;
        cmd_mem[2] = 32'h0000_02a0;
        cmd_mem[3] = 32'd28;
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
        cmd_mem[19] = 32'd18;
        cmd_mem[20] = 32'h0006_0001;
        cmd_mem[21] = 32'h0001_0001;
        cmd_mem[22] = 32'h0101_0601;
        cmd_mem[23] = 32'h0500_0101;
        cmd_mem[24] = 32'h0003_0000;
        cmd_mem[25] = 32'd5;
        cmd_mem[26] = 32'd5;
        cmd_mem[27] = 32'd0;
        cmd_mem[28] = 32'd0;
        cmd_mem[29] = 32'd0;
        cmd_mem[30] = 32'd6;
        cmd_mem[31] = (32'd1 << 16) | (32'd4 << 8) | 32'd3;

        // Command 1: CONV sample MAC, accumulate a second partial-K tile.
        cmd_mem[32] = {28'd0, OP_CONV};
        cmd_mem[33] = 32'd16;
        cmd_mem[34] = 32'h0000_02a0;
        cmd_mem[35] = 32'd108;
        cmd_mem[36] = 32'h0102_0304;
        cmd_mem[37] = 32'hfc07_0000;
        cmd_mem[40] = 32'h0201_ff03;
        cmd_mem[41] = 32'h0506_0000;
        cmd_mem[44] = 32'd6;
        cmd_mem[45] = 32'd5;
        cmd_mem[46] = 32'sd1073741824;
        cmd_mem[47] = 32'd1;
        cmd_mem[48] = -32'sd128;
        cmd_mem[49] = 32'sd127;
        cmd_mem[50] = 32'd36;
        cmd_mem[51] = 32'd36;
        cmd_mem[52] = 32'h0006_0001;
        cmd_mem[53] = 32'h0001_0001;
        cmd_mem[54] = 32'h0101_0601;
        cmd_mem[55] = 32'h0500_0101;
        cmd_mem[56] = 32'h0003_0000;
        cmd_mem[57] = 32'd5;
        cmd_mem[58] = 32'd5;
        cmd_mem[59] = 32'd0;
        cmd_mem[60] = 32'd0;
        cmd_mem[61] = 32'd0;
        cmd_mem[62] = 32'd6;
        cmd_mem[63] = (32'd1 << 16) | (32'd4 << 8) | 32'd3;

        // Command 2: REQUANT sample using the CONV raw accumulator.
        cmd_mem[64] = {28'd0, OP_REQUANT};
        cmd_mem[65] = 32'd1;
        cmd_mem[66] = 32'h0000_02b0;
        cmd_mem[68] = 32'd18;
        cmd_mem[78] = 32'sd1073741824;
        cmd_mem[79] = 32'd1;
        cmd_mem[80] = -32'sd128;
        cmd_mem[81] = 32'sd127;
        cmd_mem[82] = 32'd18;

        // Command 3: POOL sample max over 7 INT8 elements.
        cmd_mem[96] = {28'd0, OP_POOL};
        cmd_mem[97] = 32'd7;
        cmd_mem[98] = 32'h0000_02c0;
        cmd_mem[100] = 32'h0102_0304;
        cmd_mem[101] = 32'hfc07_0000;
        cmd_mem[108] = 32'd7;
        cmd_mem[114] = 32'd7;

        // Command 4: EWE sample add over 4 INT8 elements.
        cmd_mem[128] = {28'd0, OP_EWE};
        cmd_mem[129] = 32'd4;
        cmd_mem[130] = 32'h0000_02d0;
        cmd_mem[132] = 32'h0102_0304;
        cmd_mem[136] = 32'h0201_ff03;
        cmd_mem[140] = 32'd4;
        cmd_mem[146] = 32'd15;

        // Command 5: UDMA read path.
        cmd_mem[160] = {28'd0, OP_UDMA};
        cmd_mem[161] = 32'd256;
        cmd_mem[162] = 32'h0000_02a0;
        cmd_mem[163] = 32'd0;
        cmd_mem[164] = 32'd512;
        cmd_mem[165] = 32'd3;

        // Command 6: TNPS space-to-depth over a 4x4x1 tensor, block=2.
        cmd_mem[192] = {28'd0, OP_TNPS};
        cmd_mem[193] = 32'd128;
        cmd_mem[194] = 32'h0000_03f0;
        cmd_mem[195] = 32'd2;
        cmd_mem[198] = 32'd4;
        cmd_mem[199] = 32'd4;
        cmd_mem[200] = 32'd1;
        cmd_mem[201] = 32'd2;
        cmd_mem[202] = 32'd2;
        cmd_mem[203] = 32'd4;
        cmd_mem[204] = 32'd2;
        cmd_mem[205] = 32'd1;
        cmd_mem[206] = 32'd2;
        cmd_mem[207] = 32'd0;
        cmd_mem[208] = 32'd4;
        cmd_mem[209] = 32'd2;
        cmd_mem[210] = 32'd1;

        // Command 7: CONV nonzero output-index probe of prior shadow writeback.
        cmd_mem[224] = {28'd0, OP_CONV};
        cmd_mem[225] = 32'd16;
        cmd_mem[226] = 32'h0000_02a0;
        cmd_mem[227] = 32'd140;
        cmd_mem[228] = 32'h0102_0304;
        cmd_mem[229] = 32'hfc07_0000;
        cmd_mem[232] = 32'h0201_ff03;
        cmd_mem[233] = 32'h0506_0000;
        cmd_mem[236] = 32'd6;
        cmd_mem[237] = 32'd5;
        cmd_mem[238] = 32'sd1073741824;
        cmd_mem[239] = 32'd1;
        cmd_mem[240] = -32'sd128;
        cmd_mem[241] = 32'sd127;
        cmd_mem[242] = 32'd18;
        cmd_mem[243] = 32'd36;
        cmd_mem[244] = 32'h0006_0001;
        cmd_mem[245] = 32'h0001_0001;
        cmd_mem[246] = 32'h0101_0601;
        cmd_mem[247] = 32'h0300_0101;
        cmd_mem[248] = 32'h0003_0000;
        cmd_mem[249] = 32'd5;
        cmd_mem[250] = 32'd3;
        cmd_mem[251] = 32'd2;
        cmd_mem[252] = 32'd2;
        cmd_mem[253] = 32'd0;
        cmd_mem[254] = 32'd4;
        cmd_mem[255] = (32'd1 << 16) | (32'd4 << 8) | 32'd1;

        // Command 8: stop.
        cmd_mem[256] = {28'd0, OP_DONE};

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
            desc_layer_id <= 16'd0;
            desc_microblock_id <= 16'd0;
            desc_stream_slot <= 8'd0;
            desc_stream_meta_flags <= 8'd0;
            bytes <= 32'd0;
            udma_dram_read_bytes <= 32'd0;
            udma_codec_cycles <= 32'd0;
            udma_direction_write <= 1'b0;
            udma_final_write_mode <= 1'b0;
            udma_sramcrc_mode <= 1'b0;
            udma_input_byte <= 8'd0;
            udma_out_byte_offset <= 32'd0;
            udma_sramcrc_expected_crc <= 32'd0;
            udma_sramcrc_expected_count <= 32'd0;
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
            tnps_final_write_mode <= 1'b0;
            tnps_sramcrc_mode <= 1'b0;
            tnps_input_byte <= 8'd0;
            tnps_out_byte_offset <= 32'd0;
            tnps_sramcrc_expected_crc <= 32'd0;
            tnps_sramcrc_expected_count <= 32'd0;
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
            conv_in_h <= 16'd1;
            conv_in_w <= 16'd1;
            conv_in_c <= 16'd1;
            conv_out_h <= 16'd1;
            conv_out_w <= 16'd1;
            conv_out_c <= 16'd1;
            conv_k_h <= 8'd1;
            conv_k_w <= 8'd1;
            conv_stride_h <= 8'd1;
            conv_stride_w <= 8'd1;
            conv_dilation_h <= 8'd1;
            conv_dilation_w <= 8'd1;
            conv_pad_top <= 16'sd0;
            conv_pad_left <= 16'sd0;
            conv_elem_bytes <= 2'd1;
            conv_out_elem_index <= 32'd0;
            conv_tile_output_count <= 8'd1;
            conv_partial_first <= 1'b0;
            conv_partial_accumulate <= 1'b0;
            conv_partial_final <= 1'b0;
            conv_refcrc_mode <= 1'b0;
            conv_sramcrc_mode <= 1'b0;
            conv_refcrc_expected_crc <= 32'd0;
            conv_refcrc_expected_count <= 32'd0;
            conv_refcrc_ref_off <= 32'd0;
            conv_sample_kh <= 16'd0;
            conv_sample_kw <= 16'd0;
            conv_sample_ic <= 16'd0;
            requant_input_value <= 32'sd0;
            requant_read_input_from_l1 <= 1'b0;
            requant_sramcrc_mode <= 1'b0;
            requant_sramcrc_expected_crc <= 32'd0;
            requant_sramcrc_expected_count <= 32'd0;
            requant_out_byte_offset <= 32'd0;
            pool_avg_mode <= 1'b0;
            pool_fp_mode <= 1'b0;
            pool_int16_mode <= 1'b0;
            pool_read_sample_from_l1 <= 1'b0;
            pool_refcrc_mode <= 1'b0;
            pool_sramcrc_mode <= 1'b0;
            pool_refcrc_expected_crc <= 32'd0;
            pool_refcrc_expected_count <= 32'd0;
            pool_refcrc_ref_off <= 32'd0;
            pool_out_byte_offset <= 32'd0;
            pool_sample_vec <= 128'd0;
            pool_elem_count <= 8'd0;
            ewe_op_mode <= 2'd0;
            ewe_fp_mode <= 1'b0;
            ewe_int16_mode <= 1'b0;
            ewe_final_q_mode <= 1'b0;
            ewe_read_a_from_l1 <= 1'b0;
            ewe_sramcrc_mode <= 1'b0;
            ewe_sramcrc_expected_crc <= 32'd0;
            ewe_sramcrc_expected_count <= 32'd0;
            ewe_out_byte_offset <= 32'd0;
            ewe_a_vec <= 128'd0;
            ewe_b_vec <= 128'd0;
            ewe_elem_count <= 8'd0;
            ewe_zp_a <= 32'sd0;
            ewe_zp_b <= 32'sd0;
            ewe_zp_out <= 32'sd0;
            ewe_mult_a <= 32'sd1073741824;
            ewe_shift_a <= 8'sd0;
            ewe_mult_b <= 32'sd1073741824;
            ewe_shift_b <= 8'sd0;
            ewe_mult_out <= 32'sd1073741824;
            ewe_shift_out <= 8'sd0;
            ewe_left_shift <= 32'sd0;
            ewe_act_min <= -32'sd128;
            ewe_act_max <= 32'sd127;
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
                        if ((active_layer_id !== desc_layer_id) ||
                            (active_microblock_id !== desc_microblock_id) ||
                            (active_stream_slot !== desc_stream_slot) ||
                            (active_stream_meta_flags !== desc_stream_meta_flags)) begin
                            $display("HOST_FINAL_FAIL: metadata mismatch cmd=%0d layer=%0d/%0d mb=%0d/%0d slot=%0d/%0d flags=%02x/%02x",
                                     command_index,
                                     active_layer_id, desc_layer_id,
                                     active_microblock_id, desc_microblock_id,
                                     active_stream_slot, desc_stream_slot,
                                     active_stream_meta_flags, desc_stream_meta_flags);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_L1CRC) &&
                            ((l1mesh_crc !== cmd_mem[base + 28]) ||
                             (l1mesh_crc_count !== cmd_mem[base + 29]))) begin
                            $display("HOST_FINAL_FAIL: L1Mesh crc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d addr=%0d",
                                     command_index, l1mesh_crc,
                                     cmd_mem[base + 28],
                                     l1mesh_crc_count,
                                     cmd_mem[base + 29],
                                     cmd_mem[base + 2]);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && conv_fp_mode &&
                            (conv_fp_sum_bits !== {cmd_mem[base + 17], cmd_mem[base + 16]})) begin
                            $display("HOST_FINAL_FAIL: CONV FP sample cmd=%0d got=%016x expected=%016x",
                                     command_index, conv_fp_sum_bits,
                                     {cmd_mem[base + 17], cmd_mem[base + 16]});
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && conv_int16_mode &&
                            (conv_int16_acc_out !== $signed(cmd_mem[base + 18]))) begin
                            $display("HOST_FINAL_FAIL: CONV INT16 sample cmd=%0d acc=%0d expected=%0d",
                                     command_index, conv_int16_acc_out,
                                     $signed(cmd_mem[base + 18]));
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            !cmd_mem[base + 3][9] &&
                            !cmd_mem[base + 3][10] &&
                            !cmd_mem[base + 3][6] &&
                            (conv_out_q !== cmd_mem[base + 18][7:0])) begin
                            $display("HOST_FINAL_FAIL: CONV sample cmd=%0d acc=%0d scaled=%0d out=%0d expected=%0d",
                                     command_index, conv_acc_out, conv_scaled_out,
                                     conv_out_q, $signed(cmd_mem[base + 18][7:0]));
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            !cmd_mem[base + 3][9] &&
                            !cmd_mem[base + 3][10] &&
                            cmd_mem[base + 3][2] &&
                            ((conv_sample_input_valid !== cmd_mem[base + 3][3]) ||
                             (conv_sample_input_byte_offset !== cmd_mem[base + 25]) ||
                             (conv_sample_weight_byte_offset !== cmd_mem[base + 26]) ||
                             (conv_sample_output_byte_offset !== cmd_mem[base + 27]) ||
                             (conv_first_input_byte_offset !== cmd_mem[base + 28]) ||
                             (conv_first_weight_byte_offset !== cmd_mem[base + 29]) ||
                             (conv_window_valid_count !== cmd_mem[base + 30][7:0]) ||
                             (conv_tile_last_output_byte_offset !== expected_conv_tile_last_byte_offset) ||
                             (conv_tile_last_input_valid !== cmd_mem[base + 31][16]) ||
                             (conv_tile_last_window_valid_count !== cmd_mem[base + 31][15:8]) ||
                             (conv_tile_scoreboard_valid_mask !== expected_conv_tile_valid_mask) ||
                             (conv_tile_scoreboard_q_sum !== expected_conv_tile_q_sum) ||
                             (conv_tile_result_out_elem_indices[31:0] !== expected_conv_base_out_elem_index) ||
                             (conv_tile_result_output_byte_offsets[31:0] !== cmd_mem[base + 27]) ||
                             (conv_tile_result_acc_values[31:0] !== expected_conv_tile_acc_value) ||
                             (conv_tile_result_q_values[31:0] !== expected_conv_tile_q_value) ||
                             (conv_tile_result_out_elem_indices[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                             expected_conv_tile_last_index) ||
                             (conv_tile_result_output_byte_offsets[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_tile_last_byte_offset) ||
                             (conv_tile_result_acc_values[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_tile_acc_value) ||
                             (conv_tile_result_q_values[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_tile_q_value))) begin
                            $display("HOST_FINAL_FAIL: CONV 2D sample cmd=%0d valid=%0d expected=%0d in=%0d expected=%0d wgt=%0d expected=%0d out=%0d expected=%0d first_in=%0d expected=%0d first_wgt=%0d expected=%0d valid_count=%0d expected=%0d tile_last_out=%0d tile_last_valid=%0d tile_last_count=%0d tile_mask=%04b expected=%04b tile_q_sum=%0d expected=%0d tile_first_idx=%0d tile_last_idx=%0d expected=%0d tile_last_off=%0d expected=%0d tile_acc0=%0d tile_acc_last=%0d expected=%0d tile_q0=%0d tile_q_last=%0d expected=%0d tile_count=%0d",
                                     command_index,
                                     conv_sample_input_valid, cmd_mem[base + 3][3],
                                     conv_sample_input_byte_offset, cmd_mem[base + 25],
                                     conv_sample_weight_byte_offset, cmd_mem[base + 26],
                                     conv_sample_output_byte_offset, cmd_mem[base + 27],
                                     conv_first_input_byte_offset, cmd_mem[base + 28],
                                     conv_first_weight_byte_offset, cmd_mem[base + 29],
                                     conv_window_valid_count, cmd_mem[base + 30][7:0],
                                     conv_tile_last_output_byte_offset,
                                     conv_tile_last_input_valid,
                                     conv_tile_last_window_valid_count,
                                     conv_tile_scoreboard_valid_mask,
                                     expected_conv_tile_valid_mask,
                                     conv_tile_scoreboard_q_sum,
                                     expected_conv_tile_q_sum,
                                     conv_tile_result_out_elem_indices[31:0],
                                     conv_tile_result_out_elem_indices[(expected_conv_tile_count - 8'd1) * 32 +: 32],
                                     expected_conv_tile_last_index,
                                     conv_tile_result_output_byte_offsets[(expected_conv_tile_count - 8'd1) * 32 +: 32],
                                     expected_conv_tile_last_byte_offset,
                                     $signed(conv_tile_result_acc_values[31:0]),
                                     $signed(conv_tile_result_acc_values[(expected_conv_tile_count - 8'd1) * 32 +: 32]),
                                     $signed(expected_conv_tile_acc_value),
                                     $signed(conv_tile_result_q_values[31:0]),
                                     $signed(conv_tile_result_q_values[(expected_conv_tile_count - 8'd1) * 32 +: 32]),
                                     $signed(expected_conv_tile_q_value),
                                     expected_conv_tile_count);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            !cmd_mem[base + 3][9] &&
                            !cmd_mem[base + 3][10] &&
                            cmd_mem[base + 3][2] &&
                            (cmd_mem[base + 3][4] || cmd_mem[base + 3][5]) &&
                            ((conv_psum_valid_mask !== expected_conv_tile_valid_mask) ||
                             (conv_psum_acc_values[31:0] !== expected_conv_psum_acc_value) ||
                             (conv_psum_acc_values[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_psum_acc_value))) begin
                            $display("HOST_FINAL_FAIL: CONV psum cmd=%0d mask=%04b expected=%04b psum0=%0d psum_last=%0d expected=%0d first=%0d accum=%0d",
                                     command_index,
                                     conv_psum_valid_mask,
                                     expected_conv_tile_valid_mask,
                                     $signed(conv_psum_acc_values[31:0]),
                                     $signed(conv_psum_acc_values[(expected_conv_tile_count - 8'd1) * 32 +: 32]),
                                     $signed(expected_conv_psum_acc_value),
                                     cmd_mem[base + 3][4],
                                     cmd_mem[base + 3][5]);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            !cmd_mem[base + 3][9] &&
                            !cmd_mem[base + 3][10] &&
                            cmd_mem[base + 3][7] &&
                            (!conv_shadow_read_valid ||
                             (conv_shadow_read_output_byte_offset !== cmd_mem[base + 27]) ||
                             (conv_shadow_read_q_value !== cmd_mem[base + 19]))) begin
                            $display("HOST_FINAL_FAIL: CONV shadow read cmd=%0d valid=%0d off=%0d expected=%0d q=%0d expected=%0d",
                                     command_index,
                                     conv_shadow_read_valid,
                                     conv_shadow_read_output_byte_offset,
                                     cmd_mem[base + 27],
                                     $signed(conv_shadow_read_q_value),
                                     $signed(cmd_mem[base + 19]));
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            !cmd_mem[base + 3][9] &&
                            !cmd_mem[base + 3][10] &&
                            cmd_mem[base + 3][8] &&
                            ((conv_shadow_crc !== cmd_mem[base + 28]) ||
                             (conv_shadow_byte_count !== cmd_mem[base + 29]))) begin
                            $display("HOST_FINAL_FAIL: CONV shadow crc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d",
                                     command_index,
                                     conv_shadow_crc,
                                     cmd_mem[base + 28],
                                     conv_shadow_byte_count,
                                     cmd_mem[base + 29]);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            cmd_mem[base + 3][9] &&
                            ((conv_shadow_crc !== cmd_mem[base + 28]) ||
                             (conv_shadow_byte_count !== cmd_mem[base + 29]) ||
                             (cmd_mem[base + 1] !== cmd_mem[base + 29]) ||
                             (cmd_mem[base + 29] == 32'd0))) begin
                            $display("HOST_FINAL_FAIL: CONV compact refcrc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d desc_bytes=%0d",
                                     command_index,
                                     conv_shadow_crc,
                                     cmd_mem[base + 28],
                                     conv_shadow_byte_count,
                                     cmd_mem[base + 29],
                                     cmd_mem[base + 1]);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            cmd_mem[base + 3][10] &&
                            ((conv_shadow_crc !== cmd_mem[base + 28]) ||
                             (conv_shadow_byte_count !== cmd_mem[base + 29]) ||
                             (cmd_mem[base + 29] == 32'd0))) begin
                            $display("HOST_FINAL_FAIL: CONV output SRAM crc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d start=%0d",
                                     command_index,
                                     conv_shadow_crc,
                                     cmd_mem[base + 28],
                                     conv_shadow_byte_count,
                                     cmd_mem[base + 29],
                                     cmd_mem[base + 27]);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_CONV) && !conv_fp_mode && !conv_int16_mode &&
                            !cmd_mem[base + 3][9] &&
                            !cmd_mem[base + 3][10] &&
                            cmd_mem[base + 3][2] && cmd_mem[base + 3][6] &&
                            ((conv_writeback_valid_mask !== expected_conv_tile_valid_mask) ||
                             (conv_writeback_output_byte_offsets[31:0] !== cmd_mem[base + 27]) ||
                             (conv_writeback_output_byte_offsets[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_tile_last_byte_offset) ||
                             (conv_writeback_q_values[31:0] !== expected_conv_tile_q_value) ||
                             (conv_writeback_q_values[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_tile_q_value) ||
                             (conv_shadow_valid_mask !== expected_conv_tile_valid_mask) ||
                             (conv_shadow_output_byte_offsets[31:0] !== cmd_mem[base + 27]) ||
                             (conv_shadow_output_byte_offsets[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                             expected_conv_tile_last_byte_offset) ||
                             (conv_shadow_q_values[31:0] !== expected_conv_tile_q_value) ||
                             (conv_shadow_q_values[(expected_conv_tile_count - 8'd1) * 32 +: 32] !==
                              expected_conv_tile_q_value) ||
                             !conv_shadow_mem_valid_mask[cmd_mem[base + 27][3:0]] ||
                             !conv_shadow_mem_valid_mask[expected_conv_tile_last_shadow_slot] ||
                             (conv_shadow_mem_output_byte_offsets[cmd_mem[base + 27][3:0] * 32 +: 32] !==
                              cmd_mem[base + 27]) ||
                             (conv_shadow_mem_output_byte_offsets[expected_conv_tile_last_shadow_slot * 32 +: 32] !==
                              expected_conv_tile_last_byte_offset) ||
                             (conv_shadow_mem_q_values[cmd_mem[base + 27][3:0] * 32 +: 32] !==
                              expected_conv_tile_q_value) ||
                             (conv_shadow_mem_q_values[expected_conv_tile_last_shadow_slot * 32 +: 32] !==
                              expected_conv_tile_q_value))) begin
                            $display("HOST_FINAL_FAIL: CONV writeback cmd=%0d mask=%04b expected=%04b off0=%0d off_last=%0d expected=%0d q0=%0d q_last=%0d expected=%0d shadow_mask=%04b shadow_off_last=%0d shadow_q_last=%0d mem_mask=%04x mem_last_slot=%0d mem_off_last=%0d mem_q_last=%0d",
                                     command_index,
                                     conv_writeback_valid_mask,
                                     expected_conv_tile_valid_mask,
                                     conv_writeback_output_byte_offsets[31:0],
                                     conv_writeback_output_byte_offsets[(expected_conv_tile_count - 8'd1) * 32 +: 32],
                                     expected_conv_tile_last_byte_offset,
                                     $signed(conv_writeback_q_values[31:0]),
                                     $signed(conv_writeback_q_values[(expected_conv_tile_count - 8'd1) * 32 +: 32]),
                                     $signed(expected_conv_tile_q_value),
                                     conv_shadow_valid_mask,
                                     conv_shadow_output_byte_offsets[(expected_conv_tile_count - 8'd1) * 32 +: 32],
                                     $signed(conv_shadow_q_values[(expected_conv_tile_count - 8'd1) * 32 +: 32]),
                                     conv_shadow_mem_valid_mask,
                                     expected_conv_tile_last_shadow_slot,
                                     conv_shadow_mem_output_byte_offsets[expected_conv_tile_last_shadow_slot * 32 +: 32],
                                     $signed(conv_shadow_mem_q_values[expected_conv_tile_last_shadow_slot * 32 +: 32]));
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_REQUANT) && requant_sramcrc_mode &&
                            ((requant_sramcrc_crc !== requant_sramcrc_expected_crc) ||
                             (requant_sramcrc_count !== requant_sramcrc_expected_count))) begin
                            $display("HOST_FINAL_FAIL: REQUANT sramcrc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d",
                                     command_index, requant_sramcrc_crc,
                                     requant_sramcrc_expected_crc,
                                     requant_sramcrc_count,
                                     requant_sramcrc_expected_count);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_UDMA) && udma_sramcrc_mode &&
                            ((udma_sramcrc_crc !== udma_sramcrc_expected_crc) ||
                             (udma_sramcrc_count !== udma_sramcrc_expected_count))) begin
                            $display("HOST_FINAL_FAIL: UDMA sramcrc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d",
                                     command_index, udma_sramcrc_crc,
                                     udma_sramcrc_expected_crc,
                                     udma_sramcrc_count,
                                     udma_sramcrc_expected_count);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_REQUANT) && !requant_sramcrc_mode &&
                            (requant_out_q !== cmd_mem[base + 18][7:0])) begin
                            $display("HOST_FINAL_FAIL: REQUANT sample cmd=%0d scaled=%0d out=%0d expected=%0d",
                                     command_index, requant_scaled_out,
                                     requant_out_q, $signed(cmd_mem[base + 18][7:0]));
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_POOL) && (pool_refcrc_mode || pool_sramcrc_mode) &&
                            ((pool_refcrc_crc !== pool_refcrc_expected_crc) ||
                             (pool_refcrc_count !== pool_refcrc_expected_count))) begin
                            $display("HOST_FINAL_FAIL: POOL crc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d sram=%0d",
                                     command_index, pool_refcrc_crc,
                                     pool_refcrc_expected_crc,
                                     pool_refcrc_count, pool_refcrc_expected_count,
                                     pool_sramcrc_mode);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_POOL) && !pool_refcrc_mode && !pool_sramcrc_mode && pool_fp_mode &&
                            (pool_fp_bits !== {cmd_mem[base + 17], cmd_mem[base + 16]})) begin
                            $display("HOST_FINAL_FAIL: POOL FP sample cmd=%0d got=%016x expected=%016x avg=%0d",
                                     command_index, pool_fp_bits,
                                     {cmd_mem[base + 17], cmd_mem[base + 16]},
                                     pool_avg_mode);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_POOL) && !pool_refcrc_mode && !pool_sramcrc_mode && pool_int16_mode &&
                            (pool_out !== $signed(cmd_mem[base + 18]))) begin
                            $display("HOST_FINAL_FAIL: POOL INT16 sample cmd=%0d out=%0d expected=%0d avg=%0d",
                                     command_index, pool_out,
                                     $signed(cmd_mem[base + 18]),
                                     pool_avg_mode);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_POOL) && !pool_refcrc_mode && !pool_sramcrc_mode && !pool_fp_mode && !pool_int16_mode &&
                            (pool_out_q !== cmd_mem[base + 18][7:0])) begin
                            $display("HOST_FINAL_FAIL: POOL sample cmd=%0d out=%0d expected=%0d avg=%0d",
                                     command_index, pool_out,
                                     $signed(cmd_mem[base + 18][7:0]),
                                     pool_avg_mode);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_EWE) && ewe_sramcrc_mode &&
                            ((ewe_sramcrc_crc !== ewe_sramcrc_expected_crc) ||
                             (ewe_sramcrc_count !== ewe_sramcrc_expected_count))) begin
                            $display("HOST_FINAL_FAIL: EWE sramcrc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d",
                                     command_index, ewe_sramcrc_crc,
                                     ewe_sramcrc_expected_crc,
                                     ewe_sramcrc_count,
                                     ewe_sramcrc_expected_count);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_EWE) && !ewe_sramcrc_mode && !ewe_final_q_mode && ewe_fp_mode &&
                            (ewe_fp_bits !== {cmd_mem[base + 17], cmd_mem[base + 16]})) begin
                            $display("HOST_FINAL_FAIL: EWE FP sample cmd=%0d got=%016x expected=%016x mode=%0d",
                                     command_index, ewe_fp_bits,
                                     {cmd_mem[base + 17], cmd_mem[base + 16]},
                                     ewe_op_mode);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_EWE) && !ewe_sramcrc_mode && !ewe_final_q_mode && !ewe_fp_mode &&
                            (ewe_out !== $signed(cmd_mem[base + 18]))) begin
                            $display("HOST_FINAL_FAIL: EWE vector sample cmd=%0d sum=%0d first=%0d expected_sum=%0d mode=%0d",
                                     command_index, ewe_out, ewe_out_q,
                                     $signed(cmd_mem[base + 18]),
                                     ewe_op_mode);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class == OP_TNPS) && tnps_sramcrc_mode &&
                            ((tnps_sramcrc_crc !== tnps_sramcrc_expected_crc) ||
                             (tnps_sramcrc_count !== tnps_sramcrc_expected_count))) begin
                            $display("HOST_FINAL_FAIL: TNPS sramcrc cmd=%0d crc=%08x expected=%08x bytes=%0d expected=%0d",
                                     command_index, tnps_sramcrc_crc,
                                     tnps_sramcrc_expected_crc,
                                     tnps_sramcrc_count,
                                     tnps_sramcrc_expected_count);
                            test_fail <= 1'b1;
                        end
                        if (!microblock_descriptor_mode && (desc_op_class == OP_TNPS) && !tnps_sramcrc_mode && !tnps_final_write_mode &&
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
