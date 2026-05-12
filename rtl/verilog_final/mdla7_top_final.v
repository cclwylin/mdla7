`timescale 1ns/1ps

`include "common.v"

module mdla7_top_final #(
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                       clk,
    input                       rst_n,

    input                       desc_valid,
    output                      desc_ready,
    input      [3:0]            desc_op_class,
    input      [31:0]           bytes,
    input      [31:0]           udma_dram_read_bytes,
    input      [31:0]           udma_codec_cycles,
    input                       udma_direction_write,
    input      [ADDR_WIDTH-1:0] l1mesh_addr,
    input      [DATA_WIDTH-1:0] l1mesh_wdata,
    input      [DATA_WIDTH/8-1:0] l1mesh_wstrb,

    input                       tnps_mode_space_to_depth,
    input      [15:0]           tnps_in_h,
    input      [15:0]           tnps_in_w,
    input      [15:0]           tnps_in_c,
    input      [15:0]           tnps_out_h,
    input      [15:0]           tnps_out_w,
    input      [15:0]           tnps_out_c,
    input      [15:0]           tnps_block,
    input      [1:0]            tnps_elem_bytes,
    input      [31:0]           tnps_sample_out_elem_index,
    input      [31:0]           tnps_sample_in_elem_index,
    input      [127:0]          conv_act_vec,
    input      [127:0]          conv_wgt_vec,
    input      [7:0]            conv_elem_count,
    input                       conv_fp_mode,
    input                       conv_int16_mode,
    input signed [15:0]         conv_zp_in,
    input signed [31:0]         conv_bias,
    input signed [31:0]         conv_multiplier,
    input signed [7:0]          conv_shift,
    input signed [31:0]         conv_zp_out,
    input signed [31:0]         conv_act_min,
    input signed [31:0]         conv_act_max,
    input      [15:0]           conv_in_h,
    input      [15:0]           conv_in_w,
    input      [15:0]           conv_in_c,
    input      [15:0]           conv_out_h,
    input      [15:0]           conv_out_w,
    input      [15:0]           conv_out_c,
    input      [7:0]            conv_k_h,
    input      [7:0]            conv_k_w,
    input      [7:0]            conv_stride_h,
    input      [7:0]            conv_stride_w,
    input      [7:0]            conv_dilation_h,
    input      [7:0]            conv_dilation_w,
    input signed [15:0]         conv_pad_top,
    input signed [15:0]         conv_pad_left,
    input      [1:0]            conv_elem_bytes,
    input      [31:0]           conv_out_elem_index,
    input      [7:0]            conv_tile_output_count,
    input                       conv_partial_first,
    input                       conv_partial_accumulate,
    input                       conv_partial_final,
    input                       conv_refcrc_mode,
    input                       conv_sramcrc_mode,
    input      [31:0]           conv_refcrc_expected_crc,
    input      [31:0]           conv_refcrc_expected_count,
    input      [31:0]           conv_refcrc_ref_off,
    input      [15:0]           conv_sample_kh,
    input      [15:0]           conv_sample_kw,
    input      [15:0]           conv_sample_ic,
    input signed [31:0]         requant_input_value,
    input                       pool_avg_mode,
    input                       pool_fp_mode,
    input                       pool_int16_mode,
    input                       pool_refcrc_mode,
    input                       pool_sramcrc_mode,
    input      [31:0]           pool_refcrc_expected_count,
    input      [31:0]           pool_refcrc_ref_off,
    input      [31:0]           pool_out_byte_offset,
    input      [127:0]          pool_sample_vec,
    input      [7:0]            pool_elem_count,
    input      [1:0]            ewe_op_mode,
    input                       ewe_fp_mode,
    input                       ewe_int16_mode,
    input      [127:0]          ewe_a_vec,
    input      [127:0]          ewe_b_vec,
    input      [7:0]            ewe_elem_count,

    output                      done_valid,
    input                       done_ready,
    output                      busy,
    output     [3:0]            active_op_class,
    output     [3:0]            active_phase_id,
    output     [31:0]           active_remaining_cycles,
    output     [31:0]           tnps_sample_src_byte_offset,
    output     [31:0]           tnps_sample_dst_byte_offset,
    output                      tnps_sample_valid,
    output     [31:0]           placement_route_cycles,
    output     [8:0]            block_busy,
    output     [8:0]            block_done_valid,
    output signed [31:0]        conv_acc_out,
    output signed [31:0]        conv_scaled_out,
    output signed [7:0]         conv_out_q,
    output     [63:0]           conv_fp_sum_bits,
    output signed [31:0]        conv_int16_acc_out,
    output     [31:0]           conv_sample_input_byte_offset,
    output     [31:0]           conv_sample_weight_byte_offset,
    output     [31:0]           conv_sample_output_byte_offset,
    output                      conv_sample_input_valid,
    output     [31:0]           conv_first_input_byte_offset,
    output     [31:0]           conv_first_weight_byte_offset,
    output     [7:0]            conv_window_valid_count,
    output     [31:0]           conv_tile_last_output_byte_offset,
    output                      conv_tile_last_input_valid,
    output     [7:0]            conv_tile_last_window_valid_count,
    output     [3:0]            conv_tile_scoreboard_valid_mask,
    output signed [31:0]        conv_tile_scoreboard_q_sum,
    output     [127:0]          conv_tile_result_out_elem_indices,
    output     [127:0]          conv_tile_result_output_byte_offsets,
    output     [127:0]          conv_tile_result_acc_values,
    output     [127:0]          conv_tile_result_q_values,
    output     [3:0]            conv_writeback_valid_mask,
    output     [127:0]          conv_writeback_output_byte_offsets,
    output     [127:0]          conv_writeback_q_values,
    output     [3:0]            conv_shadow_valid_mask,
    output     [127:0]          conv_shadow_output_byte_offsets,
    output     [127:0]          conv_shadow_q_values,
    output     [15:0]           conv_shadow_mem_valid_mask,
    output     [511:0]          conv_shadow_mem_output_byte_offsets,
    output     [511:0]          conv_shadow_mem_q_values,
    output                      conv_shadow_read_valid,
    output     [31:0]           conv_shadow_read_output_byte_offset,
    output     [31:0]           conv_shadow_read_q_value,
    output     [31:0]           conv_shadow_crc,
    output     [31:0]           conv_shadow_byte_count,
    output     [3:0]            conv_psum_valid_mask,
    output     [127:0]          conv_psum_acc_values,
    output signed [31:0]        requant_scaled_out,
    output signed [7:0]         requant_out_q,
    output signed [31:0]        pool_out,
    output signed [7:0]         pool_out_q,
    output     [63:0]           pool_fp_bits,
    output     [31:0]           pool_refcrc_crc,
    output     [31:0]           pool_refcrc_count,
    output signed [31:0]        ewe_out,
    output signed [7:0]         ewe_out_q,
    output     [63:0]           ewe_fp_bits
);
    localparam [3:0] OP_CONV = 4'd1;
    localparam [3:0] OP_REQUANT = 4'd2;
    localparam [3:0] OP_EWE = 4'd3;
    localparam [3:0] OP_POOL = 4'd4;
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;

    localparam [2:0] ST_IDLE = 3'd0;
    localparam [2:0] ST_RUN  = 3'd1;
    localparam [2:0] ST_WAIT = 3'd2;
    localparam [2:0] ST_DONE = 3'd3;

    reg [2:0] state;
    reg done_valid_q;
    reg [3:0] op_class_q;
    reg start_pending;
    reg engine_done_seen;
    reg [31:0] bytes_q;
    reg [31:0] udma_dram_read_bytes_q;
    reg [31:0] udma_codec_cycles_q;
    reg udma_direction_write_q;
    reg [ADDR_WIDTH-1:0] l1mesh_addr_q;
    reg [DATA_WIDTH-1:0] l1mesh_wdata_q;
    reg [DATA_WIDTH/8-1:0] l1mesh_wstrb_q;
    reg tnps_mode_space_to_depth_q;
    reg [15:0] tnps_in_h_q;
    reg [15:0] tnps_in_w_q;
    reg [15:0] tnps_in_c_q;
    reg [15:0] tnps_out_h_q;
    reg [15:0] tnps_out_w_q;
    reg [15:0] tnps_out_c_q;
    reg [15:0] tnps_block_q;
    reg [1:0] tnps_elem_bytes_q;
    reg [31:0] tnps_sample_out_elem_index_q;
    reg [31:0] tnps_sample_in_elem_index_q;
    reg [127:0] conv_act_vec_q;
    reg [127:0] conv_wgt_vec_q;
    reg [7:0] conv_elem_count_q;
    reg conv_fp_mode_q;
    reg conv_int16_mode_q;
    reg signed [15:0] conv_zp_in_q;
    reg signed [31:0] conv_bias_q;
    reg signed [31:0] conv_multiplier_q;
    reg signed [7:0] conv_shift_q;
    reg signed [31:0] conv_zp_out_q;
    reg signed [31:0] conv_act_min_q;
    reg signed [31:0] conv_act_max_q;
    reg [15:0] conv_in_h_q;
    reg [15:0] conv_in_w_q;
    reg [15:0] conv_in_c_q;
    reg [15:0] conv_out_h_q;
    reg [15:0] conv_out_w_q;
    reg [15:0] conv_out_c_q;
    reg [7:0] conv_k_h_q;
    reg [7:0] conv_k_w_q;
    reg [7:0] conv_stride_h_q;
    reg [7:0] conv_stride_w_q;
    reg [7:0] conv_dilation_h_q;
    reg [7:0] conv_dilation_w_q;
    reg signed [15:0] conv_pad_top_q;
    reg signed [15:0] conv_pad_left_q;
    reg [1:0] conv_elem_bytes_q;
    reg [31:0] conv_out_elem_index_q;
    reg [7:0] conv_tile_output_count_q;
    reg conv_partial_first_q;
    reg conv_partial_accumulate_q;
    reg conv_partial_final_q;
    reg conv_refcrc_mode_q;
    reg conv_sramcrc_mode_q;
    reg [31:0] conv_refcrc_expected_crc_q;
    reg [31:0] conv_refcrc_expected_count_q;
    reg [31:0] conv_refcrc_ref_off_q;
    reg [15:0] conv_sample_kh_q;
    reg [15:0] conv_sample_kw_q;
    reg [15:0] conv_sample_ic_q;
    reg signed [31:0] requant_input_value_q;
    reg pool_avg_mode_q;
    reg pool_fp_mode_q;
    reg pool_int16_mode_q;
    reg pool_refcrc_mode_q;
    reg pool_sramcrc_mode_q;
    reg [31:0] pool_refcrc_expected_count_q;
    reg [31:0] pool_refcrc_ref_off_q;
    reg [31:0] pool_out_byte_offset_q;
    reg [127:0] pool_sample_vec_q;
    reg [7:0] pool_elem_count_q;
    reg [1:0] ewe_op_mode_q;
    reg ewe_fp_mode_q;
    reg ewe_int16_mode_q;
    reg [127:0] ewe_a_vec_q;
    reg [127:0] ewe_b_vec_q;
    reg [7:0] ewe_elem_count_q;

    wire conv_start_ready;
    wire conv_busy;
    wire conv_done_valid;
    wire [3:0] conv_phase_id;
    wire [31:0] conv_remaining_cycles;
    wire conv_l1_req_valid;
    wire conv_l1_req_ready;
    wire conv_l1_req_write;
    wire [31:0] conv_l1_req_bytes;
    wire [31:0] conv_l1_req_payload_cycles;
    wire requant_start_ready;
    wire requant_busy;
    wire requant_done_valid;
    wire [3:0] requant_phase_id;
    wire [31:0] requant_remaining_cycles;
    wire requant_l1_req_valid;
    wire requant_l1_req_ready;
    wire requant_l1_req_write;
    wire [31:0] requant_l1_req_bytes;
    wire [31:0] requant_l1_req_payload_cycles;
    wire pool_start_ready;
    wire pool_busy;
    wire pool_done_valid;
    wire [3:0] pool_phase_id;
    wire [31:0] pool_remaining_cycles;
    wire pool_l1_req_valid;
    wire pool_l1_req_ready;
    wire pool_l1_req_write;
    wire [31:0] pool_l1_req_bytes;
    wire [31:0] pool_l1_req_payload_cycles;
    wire ewe_start_ready;
    wire ewe_busy;
    wire ewe_done_valid;
    wire [3:0] ewe_phase_id;
    wire [31:0] ewe_remaining_cycles;
    wire ewe_l1_req_valid;
    wire ewe_l1_req_ready;
    wire ewe_l1_req_write;
    wire [31:0] ewe_l1_req_bytes;
    wire [31:0] ewe_l1_req_payload_cycles;

    wire udma_start_ready;
    wire udma_busy;
    wire udma_done_valid;
    wire [3:0] udma_phase_id;
    wire [31:0] udma_remaining_cycles;
    wire udma_l1_req_valid;
    wire udma_l1_req_ready;
    wire udma_l1_req_write;
    wire [31:0] udma_l1_req_bytes;
    wire [31:0] udma_l1_req_payload_cycles;

    wire tnps_start_ready;
    wire tnps_busy;
    wire tnps_done_valid;
    wire [3:0] tnps_phase_id;
    wire [31:0] tnps_remaining_cycles;
    wire tnps_l1_req_valid;
    wire tnps_l1_req_ready;
    wire tnps_l1_req_write;
    wire [31:0] tnps_l1_req_bytes;
    wire [31:0] tnps_l1_req_payload_cycles;

    wire l1mgr_busy;
    wire l1mgr_resp_valid;
    wire l1mgr_resp_ready;
    wire [3:0] l1mgr_phase_id;
    wire [31:0] l1mgr_remaining_cycles;
    wire l1mgr_mesh_req_write;
    wire [ADDR_WIDTH-1:0] l1mgr_mesh_req_addr;
    wire [31:0] l1mgr_mesh_req_bytes;
    wire [DATA_WIDTH-1:0] l1mgr_mesh_req_wdata;
    wire [DATA_WIDTH/8-1:0] l1mgr_mesh_req_wstrb;

    wire l1mesh_req_ready;
    wire l1mesh_busy;
    wire l1mesh_resp_valid;
    wire [3:0] l1mesh_phase_id;
    wire [31:0] l1mesh_remaining_cycles;
    wire [DATA_WIDTH-1:0] l1mesh_rdata;
    wire [1:0] route_source_x;
    wire [1:0] route_source_y;
    wire [1:0] route_tile_x;
    wire [1:0] route_tile_y;
    wire [1:0] route_bank_x;
    wire [1:0] route_bank_y;
    wire legacy_req_ready;
    wire [3:0] l1mgr_debug_source_unused;
    wire [7:0] l1mgr_debug_tid_unused;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [155:0] final_debug_unused = {
        l1mesh_rdata,
        route_source_x,
        route_source_y,
        route_tile_x,
        route_tile_y,
        route_bank_x,
        route_bank_y,
        legacy_req_ready,
        l1mgr_debug_source_unused,
        l1mgr_debug_tid_unused
    };
    /* verilator lint_on UNUSEDSIGNAL */

    wire run_conv = (op_class_q == OP_CONV);
    wire run_requant = (op_class_q == OP_REQUANT);
    wire run_ewe = (op_class_q == OP_EWE);
    wire run_pool = (op_class_q == OP_POOL);
    wire run_udma = (op_class_q == OP_UDMA);
    wire run_tnps = (op_class_q == OP_TNPS);
    wire selected_start_ready = run_conv ? conv_start_ready :
                                run_requant ? requant_start_ready :
                                run_ewe ? ewe_start_ready :
                                run_pool ? pool_start_ready :
                                run_udma ? udma_start_ready :
                                run_tnps ? tnps_start_ready : 1'b1;
    wire selected_done_valid = run_conv ? conv_done_valid :
                               run_requant ? requant_done_valid :
                               run_ewe ? ewe_done_valid :
                               run_pool ? pool_done_valid :
                               run_udma ? udma_done_valid :
                               run_tnps ? tnps_done_valid : 1'b1;
    wire selected_busy = run_conv ? conv_busy :
                         run_requant ? requant_busy :
                         run_ewe ? ewe_busy :
                         run_pool ? pool_busy :
                         run_udma ? udma_busy :
                         run_tnps ? tnps_busy : 1'b0;
    wire [3:0] selected_phase = run_conv ? conv_phase_id :
                                run_requant ? requant_phase_id :
                                run_ewe ? ewe_phase_id :
                                run_pool ? pool_phase_id :
                                run_udma ? udma_phase_id :
                                run_tnps ? tnps_phase_id : 4'd0;
    wire [31:0] selected_remaining = run_conv ? conv_remaining_cycles :
                                     run_requant ? requant_remaining_cycles :
                                     run_ewe ? ewe_remaining_cycles :
                                     run_pool ? pool_remaining_cycles :
                                     run_udma ? udma_remaining_cycles :
                                     run_tnps ? tnps_remaining_cycles : 32'd0;
    wire l1_drained = !l1mgr_busy && !l1mgr_resp_valid && !l1mesh_busy && !l1mesh_resp_valid;
    wire conv_start = start_pending && run_conv;
    wire requant_start = start_pending && run_requant;
    wire ewe_start = start_pending && run_ewe;
    wire pool_start = start_pending && run_pool;
    wire udma_start = start_pending && run_udma;
    wire tnps_start = start_pending && run_tnps;

    assign desc_ready = (state == ST_IDLE);
    assign done_valid = done_valid_q;
    assign busy = (state != ST_IDLE);
    assign active_op_class = op_class_q;
    assign active_phase_id = selected_busy ? selected_phase :
                             l1mgr_busy ? l1mgr_phase_id :
                             l1mesh_busy ? l1mesh_phase_id : 4'd0;
    assign active_remaining_cycles = selected_busy ? selected_remaining :
                                     l1mgr_busy ? l1mgr_remaining_cycles :
                                     l1mesh_busy ? l1mesh_remaining_cycles : 32'd0;
    assign l1mgr_resp_ready = l1mesh_req_ready;
    assign conv_l1_req_ready = run_conv && l1mesh_req_ready;
    assign block_busy = {l1mesh_busy, l1mgr_busy, udma_busy, tnps_busy, pool_busy, ewe_busy, requant_busy, conv_busy, 1'b0};
    assign block_done_valid = {l1mesh_resp_valid, l1mgr_resp_valid, udma_done_valid, tnps_done_valid, pool_done_valid, ewe_done_valid, requant_done_valid, conv_done_valid, 1'b0};

    vf_l1mesh_route_estimator u_route (
        .source_id(op_class_q),
        .addr(l1mesh_addr_q),
        .route_cycles(placement_route_cycles),
        .source_x(route_source_x),
        .source_y(route_source_y),
        .tile_x(route_tile_x),
        .tile_y(route_tile_y),
        .bank_x(route_bank_x),
        .bank_y(route_bank_y)
    );

    vf_conv_sample_engine u_conv (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(conv_start),
        .start_ready(conv_start_ready),
        .act_vec(conv_act_vec_q),
        .wgt_vec(conv_wgt_vec_q),
        .elem_count(conv_elem_count_q),
        .fp_mode(conv_fp_mode_q),
        .int16_mode(conv_int16_mode_q),
        .zp_in(conv_zp_in_q),
        .bias(conv_bias_q),
        .multiplier(conv_multiplier_q),
        .shift(conv_shift_q),
        .zp_out(conv_zp_out_q),
        .act_min(conv_act_min_q),
        .act_max(conv_act_max_q),
        .conv_in_h(conv_in_h_q),
        .conv_in_w(conv_in_w_q),
        .conv_in_c(conv_in_c_q),
        .conv_out_h(conv_out_h_q),
        .conv_out_w(conv_out_w_q),
        .conv_out_c(conv_out_c_q),
        .conv_k_h(conv_k_h_q),
        .conv_k_w(conv_k_w_q),
        .conv_stride_h(conv_stride_h_q),
        .conv_stride_w(conv_stride_w_q),
        .conv_dilation_h(conv_dilation_h_q),
        .conv_dilation_w(conv_dilation_w_q),
        .conv_pad_top(conv_pad_top_q),
        .conv_pad_left(conv_pad_left_q),
        .conv_elem_bytes(conv_elem_bytes_q),
        .conv_out_elem_index(conv_out_elem_index_q),
        .conv_tile_output_count(conv_tile_output_count_q),
        .conv_partial_first(conv_partial_first_q),
        .conv_partial_accumulate(conv_partial_accumulate_q),
        .conv_partial_final(conv_partial_final_q),
        .conv_refcrc_mode(conv_refcrc_mode_q),
        .conv_sramcrc_mode(conv_sramcrc_mode_q),
        .conv_refcrc_expected_crc(conv_refcrc_expected_crc_q),
        .conv_refcrc_expected_count(conv_refcrc_expected_count_q),
        .conv_refcrc_ref_off(conv_refcrc_ref_off_q),
        .conv_sample_kh(conv_sample_kh_q),
        .conv_sample_kw(conv_sample_kw_q),
        .conv_sample_ic(conv_sample_ic_q),
        .l1_req_valid(conv_l1_req_valid),
        .l1_req_ready(conv_l1_req_ready),
        .l1_req_write(conv_l1_req_write),
        .l1_req_bytes(conv_l1_req_bytes),
        .l1_req_payload_cycles(conv_l1_req_payload_cycles),
        .busy(conv_busy),
        .done_valid(conv_done_valid),
        .done_ready(1'b1),
        .phase_id(conv_phase_id),
        .remaining_cycles(conv_remaining_cycles),
        .acc_out(conv_acc_out),
        .scaled_out(conv_scaled_out),
        .out_q(conv_out_q),
        .fp_sum_bits(conv_fp_sum_bits),
        .int16_acc_out(conv_int16_acc_out),
        .conv_sample_input_byte_offset(conv_sample_input_byte_offset),
        .conv_sample_weight_byte_offset(conv_sample_weight_byte_offset),
        .conv_sample_output_byte_offset(conv_sample_output_byte_offset),
        .conv_sample_input_valid(conv_sample_input_valid),
        .conv_first_input_byte_offset(conv_first_input_byte_offset),
        .conv_first_weight_byte_offset(conv_first_weight_byte_offset),
        .conv_window_valid_count(conv_window_valid_count),
        .conv_tile_last_output_byte_offset(conv_tile_last_output_byte_offset),
        .conv_tile_last_input_valid(conv_tile_last_input_valid),
        .conv_tile_last_window_valid_count(conv_tile_last_window_valid_count),
        .conv_tile_scoreboard_valid_mask(conv_tile_scoreboard_valid_mask),
        .conv_tile_scoreboard_q_sum(conv_tile_scoreboard_q_sum),
        .conv_tile_result_out_elem_indices(conv_tile_result_out_elem_indices),
        .conv_tile_result_output_byte_offsets(conv_tile_result_output_byte_offsets),
        .conv_tile_result_acc_values(conv_tile_result_acc_values),
        .conv_tile_result_q_values(conv_tile_result_q_values),
        .conv_writeback_valid_mask(conv_writeback_valid_mask),
        .conv_writeback_output_byte_offsets(conv_writeback_output_byte_offsets),
        .conv_writeback_q_values(conv_writeback_q_values),
        .conv_shadow_valid_mask(conv_shadow_valid_mask),
        .conv_shadow_output_byte_offsets(conv_shadow_output_byte_offsets),
        .conv_shadow_q_values(conv_shadow_q_values),
        .conv_shadow_mem_valid_mask(conv_shadow_mem_valid_mask),
        .conv_shadow_mem_output_byte_offsets(conv_shadow_mem_output_byte_offsets),
        .conv_shadow_mem_q_values(conv_shadow_mem_q_values),
        .conv_shadow_read_valid(conv_shadow_read_valid),
        .conv_shadow_read_output_byte_offset(conv_shadow_read_output_byte_offset),
        .conv_shadow_read_q_value(conv_shadow_read_q_value),
        .conv_shadow_crc(conv_shadow_crc),
        .conv_shadow_byte_count(conv_shadow_byte_count),
        .conv_psum_valid_mask(conv_psum_valid_mask),
        .conv_psum_acc_values(conv_psum_acc_values)
    );

    vf_requant_sample_engine u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(requant_start),
        .start_ready(requant_start_ready),
        .input_value(requant_input_value_q),
        .multiplier(conv_multiplier_q),
        .shift(conv_shift_q),
        .zp_out(conv_zp_out_q),
        .act_min(conv_act_min_q),
        .act_max(conv_act_max_q),
        .l1_req_valid(requant_l1_req_valid),
        .l1_req_ready(requant_l1_req_ready),
        .l1_req_write(requant_l1_req_write),
        .l1_req_bytes(requant_l1_req_bytes),
        .l1_req_payload_cycles(requant_l1_req_payload_cycles),
        .busy(requant_busy),
        .done_valid(requant_done_valid),
        .done_ready(1'b1),
        .phase_id(requant_phase_id),
        .remaining_cycles(requant_remaining_cycles),
        .scaled_out(requant_scaled_out),
        .out_q(requant_out_q)
    );

    vf_pool_sample_engine u_pool (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(pool_start),
        .start_ready(pool_start_ready),
        .avg_mode(pool_avg_mode_q),
        .fp_mode(pool_fp_mode_q),
        .int16_mode(pool_int16_mode_q),
        .refcrc_mode(pool_refcrc_mode_q),
        .sramcrc_mode(pool_sramcrc_mode_q),
        .refcrc_expected_count(pool_refcrc_expected_count_q),
        .refcrc_ref_off(pool_refcrc_ref_off_q),
        .out_byte_offset(pool_out_byte_offset_q),
        .sample_vec(pool_sample_vec_q),
        .elem_count(pool_elem_count_q),
        .l1_req_valid(pool_l1_req_valid),
        .l1_req_ready(pool_l1_req_ready),
        .l1_req_write(pool_l1_req_write),
        .l1_req_bytes(pool_l1_req_bytes),
        .l1_req_payload_cycles(pool_l1_req_payload_cycles),
        .busy(pool_busy),
        .done_valid(pool_done_valid),
        .done_ready(1'b1),
        .phase_id(pool_phase_id),
        .remaining_cycles(pool_remaining_cycles),
        .pool_out(pool_out),
        .out_q(pool_out_q),
        .fp_pool_bits(pool_fp_bits),
        .refcrc_crc(pool_refcrc_crc),
        .refcrc_count(pool_refcrc_count)
    );

    vf_ewe_sample_engine u_ewe (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(ewe_start),
        .start_ready(ewe_start_ready),
        .op_mode(ewe_op_mode_q),
        .fp_mode(ewe_fp_mode_q),
        .int16_mode(ewe_int16_mode_q),
        .a_vec(ewe_a_vec_q),
        .b_vec(ewe_b_vec_q),
        .elem_count(ewe_elem_count_q),
        .l1_req_valid(ewe_l1_req_valid),
        .l1_req_ready(ewe_l1_req_ready),
        .l1_req_write(ewe_l1_req_write),
        .l1_req_bytes(ewe_l1_req_bytes),
        .l1_req_payload_cycles(ewe_l1_req_payload_cycles),
        .busy(ewe_busy),
        .done_valid(ewe_done_valid),
        .done_ready(1'b1),
        .phase_id(ewe_phase_id),
        .remaining_cycles(ewe_remaining_cycles),
        .ewe_out(ewe_out),
        .out_q(ewe_out_q),
        .fp_ewe_bits(ewe_fp_bits)
    );

    vf_udma_engine u_udma (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(udma_start),
        .start_ready(udma_start_ready),
        .direction_write(udma_direction_write_q),
        .bytes(bytes_q),
        .dram_read_bytes(udma_dram_read_bytes_q),
        .codec_cycles(udma_codec_cycles_q),
        .l1_req_valid(udma_l1_req_valid),
        .l1_req_ready(udma_l1_req_ready),
        .l1_req_write(udma_l1_req_write),
        .l1_req_bytes(udma_l1_req_bytes),
        .l1_req_payload_cycles(udma_l1_req_payload_cycles),
        .busy(udma_busy),
        .done_valid(udma_done_valid),
        .done_ready(1'b1),
        .phase_id(udma_phase_id),
        .remaining_cycles(udma_remaining_cycles)
    );

    vf_tnps_engine u_tnps (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(tnps_start),
        .start_ready(tnps_start_ready),
        .bytes(bytes_q),
        .mode_space_to_depth(tnps_mode_space_to_depth_q),
        .in_h(tnps_in_h_q),
        .in_w(tnps_in_w_q),
        .in_c(tnps_in_c_q),
        .out_h(tnps_out_h_q),
        .out_w(tnps_out_w_q),
        .out_c(tnps_out_c_q),
        .block(tnps_block_q),
        .elem_bytes(tnps_elem_bytes_q),
        .sample_out_elem_index(tnps_sample_out_elem_index_q),
        .sample_in_elem_index(tnps_sample_in_elem_index_q),
        .l1_req_valid(tnps_l1_req_valid),
        .l1_req_ready(tnps_l1_req_ready),
        .l1_req_write(tnps_l1_req_write),
        .l1_req_bytes(tnps_l1_req_bytes),
        .l1_req_payload_cycles(tnps_l1_req_payload_cycles),
        .busy(tnps_busy),
        .done_valid(tnps_done_valid),
        .done_ready(1'b1),
        .phase_id(tnps_phase_id),
        .remaining_cycles(tnps_remaining_cycles),
        .sample_src_byte_offset(tnps_sample_src_byte_offset),
        .sample_dst_byte_offset(tnps_sample_dst_byte_offset),
        .sample_valid(tnps_sample_valid)
    );

    l1manager u_l1manager (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(1'b0),
        .req_ready(legacy_req_ready),
        .req_write(1'b0),
        .req_l1(1'b1),
        .req_source(4'd0),
        .req_tid(8'd0),
        .req_bytes(32'd0),
        .req_payload_cycles(32'd0),
        .req_addr({ADDR_WIDTH{1'b0}}),
        .req_wdata({DATA_WIDTH{1'b0}}),
        .req_wstrb({DATA_WIDTH/8{1'b0}}),
        .udma_req_valid(udma_l1_req_valid),
        .udma_req_ready(udma_l1_req_ready),
        .udma_req_write(udma_l1_req_write),
        .udma_req_tid(8'd1),
        .udma_req_bytes(udma_l1_req_bytes),
        .udma_req_payload_cycles(udma_l1_req_payload_cycles),
        .udma_req_addr(l1mesh_addr_q),
        .udma_req_wdata(l1mesh_wdata_q),
        .udma_req_wstrb(l1mesh_wstrb_q),
        .requant_req_valid(requant_l1_req_valid),
        .requant_req_ready(requant_l1_req_ready),
        .requant_req_write(requant_l1_req_write),
        .requant_req_tid(8'd0),
        .requant_req_bytes(requant_l1_req_bytes),
        .requant_req_payload_cycles(requant_l1_req_payload_cycles),
        .requant_req_addr(l1mesh_addr_q),
        .requant_req_wdata(l1mesh_wdata_q),
        .requant_req_wstrb(l1mesh_wstrb_q),
        .ewe_req_valid(ewe_l1_req_valid),
        .ewe_req_ready(ewe_l1_req_ready),
        .ewe_req_write(ewe_l1_req_write),
        .ewe_req_tid(8'd0),
        .ewe_req_bytes(ewe_l1_req_bytes),
        .ewe_req_payload_cycles(ewe_l1_req_payload_cycles),
        .ewe_req_addr(l1mesh_addr_q),
        .ewe_req_wdata(l1mesh_wdata_q),
        .ewe_req_wstrb(l1mesh_wstrb_q),
        .pool_req_valid(pool_l1_req_valid),
        .pool_req_ready(pool_l1_req_ready),
        .pool_req_write(pool_l1_req_write),
        .pool_req_tid(8'd0),
        .pool_req_bytes(pool_l1_req_bytes),
        .pool_req_payload_cycles(pool_l1_req_payload_cycles),
        .pool_req_addr(l1mesh_addr_q),
        .pool_req_wdata(l1mesh_wdata_q),
        .pool_req_wstrb(l1mesh_wstrb_q),
        .tnps_req_valid(tnps_l1_req_valid),
        .tnps_req_ready(tnps_l1_req_ready),
        .tnps_req_write(tnps_l1_req_write),
        .tnps_req_tid(8'd2),
        .tnps_req_bytes(tnps_l1_req_bytes),
        .tnps_req_payload_cycles(tnps_l1_req_payload_cycles),
        .tnps_req_addr(l1mesh_addr_q),
        .tnps_req_wdata(l1mesh_wdata_q),
        .tnps_req_wstrb(l1mesh_wstrb_q),
        .mesh_req_write(l1mgr_mesh_req_write),
        .mesh_req_addr(l1mgr_mesh_req_addr),
        .mesh_req_bytes(l1mgr_mesh_req_bytes),
        .mesh_req_wdata(l1mgr_mesh_req_wdata),
        .mesh_req_wstrb(l1mgr_mesh_req_wstrb),
        .resp_valid(l1mgr_resp_valid),
        .resp_ready(l1mgr_resp_ready),
        .busy(l1mgr_busy),
        .phase_id(l1mgr_phase_id),
        .remaining_cycles(l1mgr_remaining_cycles),
        .debug_source(l1mgr_debug_source_unused),
        .debug_tid(l1mgr_debug_tid_unused)
    );

    l1mesh u_l1mesh (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(run_conv ? conv_l1_req_valid : l1mgr_resp_valid),
        .req_ready(l1mesh_req_ready),
        .req_write(run_conv ? conv_l1_req_write : l1mgr_mesh_req_write),
        .req_addr(run_conv ? l1mesh_addr_q : l1mgr_mesh_req_addr),
        .req_bytes(run_conv ? conv_l1_req_bytes : l1mgr_mesh_req_bytes),
        .route_cycles(placement_route_cycles),
        .req_wdata(run_conv ? l1mesh_wdata_q : l1mgr_mesh_req_wdata),
        .req_wstrb(run_conv ? l1mesh_wstrb_q : l1mgr_mesh_req_wstrb),
        .resp_valid(l1mesh_resp_valid),
        .resp_ready(1'b1),
        .resp_rdata(l1mesh_rdata),
        .busy(l1mesh_busy),
        .phase_id(l1mesh_phase_id),
        .remaining_cycles(l1mesh_remaining_cycles)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done_valid_q <= 1'b0;
            op_class_q <= 4'd0;
            start_pending <= 1'b0;
            engine_done_seen <= 1'b0;
            bytes_q <= 32'd0;
            udma_dram_read_bytes_q <= 32'd0;
            udma_codec_cycles_q <= 32'd0;
            udma_direction_write_q <= 1'b0;
            l1mesh_addr_q <= {ADDR_WIDTH{1'b0}};
            l1mesh_wdata_q <= {DATA_WIDTH{1'b0}};
            l1mesh_wstrb_q <= {DATA_WIDTH/8{1'b0}};
            tnps_mode_space_to_depth_q <= 1'b1;
            tnps_in_h_q <= 16'd0;
            tnps_in_w_q <= 16'd0;
            tnps_in_c_q <= 16'd0;
            tnps_out_h_q <= 16'd0;
            tnps_out_w_q <= 16'd0;
            tnps_out_c_q <= 16'd0;
            tnps_block_q <= 16'd0;
            tnps_elem_bytes_q <= 2'd1;
            tnps_sample_out_elem_index_q <= 32'd0;
            tnps_sample_in_elem_index_q <= 32'd0;
            conv_act_vec_q <= 128'd0;
            conv_wgt_vec_q <= 128'd0;
            conv_elem_count_q <= 8'd0;
            conv_fp_mode_q <= 1'b0;
            conv_int16_mode_q <= 1'b0;
            conv_zp_in_q <= 16'sd0;
            conv_bias_q <= 32'sd0;
            conv_multiplier_q <= 32'sd1073741824;
            conv_shift_q <= 8'sd1;
            conv_zp_out_q <= 32'sd0;
            conv_act_min_q <= -32'sd128;
            conv_act_max_q <= 32'sd127;
            conv_in_h_q <= 16'd1;
            conv_in_w_q <= 16'd1;
            conv_in_c_q <= 16'd1;
            conv_out_h_q <= 16'd1;
            conv_out_w_q <= 16'd1;
            conv_out_c_q <= 16'd1;
            conv_k_h_q <= 8'd1;
            conv_k_w_q <= 8'd1;
            conv_stride_h_q <= 8'd1;
            conv_stride_w_q <= 8'd1;
            conv_dilation_h_q <= 8'd1;
            conv_dilation_w_q <= 8'd1;
            conv_pad_top_q <= 16'sd0;
            conv_pad_left_q <= 16'sd0;
            conv_elem_bytes_q <= 2'd1;
            conv_out_elem_index_q <= 32'd0;
            conv_tile_output_count_q <= 8'd1;
            conv_partial_first_q <= 1'b0;
            conv_partial_accumulate_q <= 1'b0;
            conv_partial_final_q <= 1'b0;
            conv_refcrc_mode_q <= 1'b0;
            conv_sramcrc_mode_q <= 1'b0;
            conv_refcrc_expected_crc_q <= 32'd0;
            conv_refcrc_expected_count_q <= 32'd0;
            conv_refcrc_ref_off_q <= 32'd0;
            conv_sample_kh_q <= 16'd0;
            conv_sample_kw_q <= 16'd0;
            conv_sample_ic_q <= 16'd0;
            requant_input_value_q <= 32'sd0;
            pool_avg_mode_q <= 1'b0;
            pool_fp_mode_q <= 1'b0;
            pool_int16_mode_q <= 1'b0;
            pool_refcrc_mode_q <= 1'b0;
            pool_sramcrc_mode_q <= 1'b0;
            pool_refcrc_expected_count_q <= 32'd0;
            pool_refcrc_ref_off_q <= 32'd0;
            pool_out_byte_offset_q <= 32'd0;
            pool_sample_vec_q <= 128'd0;
            pool_elem_count_q <= 8'd0;
            ewe_op_mode_q <= 2'd0;
            ewe_fp_mode_q <= 1'b0;
            ewe_int16_mode_q <= 1'b0;
            ewe_a_vec_q <= 128'd0;
            ewe_b_vec_q <= 128'd0;
            ewe_elem_count_q <= 8'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    done_valid_q <= 1'b0;
                    if (desc_valid && desc_ready) begin
                        op_class_q <= desc_op_class;
                        bytes_q <= bytes;
                        udma_dram_read_bytes_q <= udma_dram_read_bytes;
                        udma_codec_cycles_q <= udma_codec_cycles;
                        udma_direction_write_q <= udma_direction_write;
                        l1mesh_addr_q <= l1mesh_addr;
                        l1mesh_wdata_q <= l1mesh_wdata;
                        l1mesh_wstrb_q <= l1mesh_wstrb;
                        tnps_mode_space_to_depth_q <= tnps_mode_space_to_depth;
                        tnps_in_h_q <= tnps_in_h;
                        tnps_in_w_q <= tnps_in_w;
                        tnps_in_c_q <= tnps_in_c;
                        tnps_out_h_q <= tnps_out_h;
                        tnps_out_w_q <= tnps_out_w;
                        tnps_out_c_q <= tnps_out_c;
                        tnps_block_q <= tnps_block;
                        tnps_elem_bytes_q <= tnps_elem_bytes;
                        tnps_sample_out_elem_index_q <= tnps_sample_out_elem_index;
                        tnps_sample_in_elem_index_q <= tnps_sample_in_elem_index;
                        conv_act_vec_q <= conv_act_vec;
                        conv_wgt_vec_q <= conv_wgt_vec;
                        conv_elem_count_q <= conv_elem_count;
                        conv_fp_mode_q <= conv_fp_mode;
                        conv_int16_mode_q <= conv_int16_mode;
                        conv_zp_in_q <= conv_zp_in;
                        conv_bias_q <= conv_bias;
                        conv_multiplier_q <= conv_multiplier;
                        conv_shift_q <= conv_shift;
                        conv_zp_out_q <= conv_zp_out;
                        conv_act_min_q <= conv_act_min;
                        conv_act_max_q <= conv_act_max;
                        conv_in_h_q <= conv_in_h;
                        conv_in_w_q <= conv_in_w;
                        conv_in_c_q <= conv_in_c;
                        conv_out_h_q <= conv_out_h;
                        conv_out_w_q <= conv_out_w;
                        conv_out_c_q <= conv_out_c;
                        conv_k_h_q <= conv_k_h;
                        conv_k_w_q <= conv_k_w;
                        conv_stride_h_q <= conv_stride_h;
                        conv_stride_w_q <= conv_stride_w;
                        conv_dilation_h_q <= conv_dilation_h;
                        conv_dilation_w_q <= conv_dilation_w;
                        conv_pad_top_q <= conv_pad_top;
                        conv_pad_left_q <= conv_pad_left;
                        conv_elem_bytes_q <= conv_elem_bytes;
                        conv_out_elem_index_q <= conv_out_elem_index;
                        conv_tile_output_count_q <= conv_tile_output_count;
                        conv_partial_first_q <= conv_partial_first;
                        conv_partial_accumulate_q <= conv_partial_accumulate;
                        conv_partial_final_q <= conv_partial_final;
                        conv_refcrc_mode_q <= conv_refcrc_mode;
                        conv_sramcrc_mode_q <= conv_sramcrc_mode;
                        conv_refcrc_expected_crc_q <= conv_refcrc_expected_crc;
                        conv_refcrc_expected_count_q <= conv_refcrc_expected_count;
                        conv_refcrc_ref_off_q <= conv_refcrc_ref_off;
                        conv_sample_kh_q <= conv_sample_kh;
                        conv_sample_kw_q <= conv_sample_kw;
                        conv_sample_ic_q <= conv_sample_ic;
                        requant_input_value_q <= requant_input_value;
                        pool_avg_mode_q <= pool_avg_mode;
                        pool_fp_mode_q <= pool_fp_mode;
                        pool_int16_mode_q <= pool_int16_mode;
                        pool_refcrc_mode_q <= pool_refcrc_mode;
                        pool_sramcrc_mode_q <= pool_sramcrc_mode;
                        pool_refcrc_expected_count_q <= pool_refcrc_expected_count;
                        pool_refcrc_ref_off_q <= pool_refcrc_ref_off;
                        pool_out_byte_offset_q <= pool_out_byte_offset;
                        pool_sample_vec_q <= pool_sample_vec;
                        pool_elem_count_q <= pool_elem_count;
                        ewe_op_mode_q <= ewe_op_mode;
                        ewe_fp_mode_q <= ewe_fp_mode;
                        ewe_int16_mode_q <= ewe_int16_mode;
                        ewe_a_vec_q <= ewe_a_vec;
                        ewe_b_vec_q <= ewe_b_vec;
                        ewe_elem_count_q <= ewe_elem_count;
                        start_pending <= 1'b1;
                        engine_done_seen <= 1'b0;
                        state <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (start_pending && selected_start_ready)
                        start_pending <= 1'b0;
                    if (selected_done_valid)
                        engine_done_seen <= 1'b1;
                    if (!start_pending && (engine_done_seen || selected_done_valid))
                        state <= ST_WAIT;
                end
                ST_WAIT: begin
                    if (l1_drained) begin
                        done_valid_q <= 1'b1;
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (done_valid_q && done_ready) begin
                        done_valid_q <= 1'b0;
                        state <= ST_IDLE;
                    end
                end
                default: begin
                    state <= ST_IDLE;
                    done_valid_q <= 1'b0;
                    start_pending <= 1'b0;
                    engine_done_seen <= 1'b0;
                end
            endcase
        end
    end
endmodule
