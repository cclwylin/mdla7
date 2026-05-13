`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_CONV_V
`define MDLA7_VERILOG_CONV_V

/* verilator lint_off DECLFILENAME */
module vf_conv_int8_mac #(
    parameter MAX_ELEMS = 16
) (
    input      [MAX_ELEMS*8-1:0] act_vec,
    input      [MAX_ELEMS*8-1:0] wgt_vec,
    input      [7:0]             elem_count,
    input signed [15:0]          zp_in,
    input signed [31:0]          bias,
    input signed [31:0]          multiplier,
    input signed [7:0]           shift,
    input signed [31:0]          zp_out,
    input signed [31:0]          act_min,
    input signed [31:0]          act_max,
    output reg signed [31:0]     acc_out,
    output reg signed [31:0]     scaled_out,
    output reg signed [7:0]      out_q
);
    integer i;
    reg signed [31:0] av;
    reg signed [31:0] wv;
    reg signed [63:0] acc64;
    reg signed [31:0] clamped_acc;
    reg signed [31:0] quantized;
    reg signed [31:0] clamped;
`ifdef MDLA7_DPI_DATAPATH
    import "DPI-C" function void mdla7_dpi_conv_int8_mac(
        input int act0, input int act1, input int act2, input int act3,
        input int wgt0, input int wgt1, input int wgt2, input int wgt3,
        input int elem_count, input int zp_in, input int bias,
        input int multiplier, input int shift, input int zp_out,
        input int act_min, input int act_max,
        output int dpi_acc_out, output int dpi_scaled_out, output int dpi_out_q
    );
    reg dpi_datapath_enabled;
    reg signed [31:0] dpi_acc_out;
    reg signed [31:0] dpi_scaled_out;
    reg signed [31:0] dpi_out_q;

    initial begin
        dpi_datapath_enabled = $test$plusargs("MDLA7_DATAPATH_DPI");
    end
`endif

    function signed [31:0] clamp_i32;
        input signed [63:0] value;
        begin
            if (value < -64'sd2147483648)
                clamp_i32 = -32'sd2147483648;
            else if (value > 64'sd2147483647)
                clamp_i32 = 32'sd2147483647;
            else
                clamp_i32 = value[31:0];
        end
    endfunction

    function signed [31:0] saturating_doubling_high_mul;
        input signed [31:0] a;
        input signed [31:0] b;
        reg signed [63:0] p;
        reg signed [63:0] nudge;
        reg signed [63:0] r;
        begin
            if ((a == -32'sd2147483648) && (b == -32'sd2147483648)) begin
                saturating_doubling_high_mul = 32'sd2147483647;
            end else begin
                p = $signed(a) * $signed(b);
                nudge = (p >= 0) ? 64'sd1073741824 : -64'sd1073741823;
                r = (p + nudge) >>> 31;
                saturating_doubling_high_mul = clamp_i32(r);
            end
        end
    endfunction

    function signed [31:0] rounding_divide_by_pot;
        input signed [31:0] x;
        input integer exponent;
        reg signed [63:0] x64;
        reg signed [63:0] mask;
        reg signed [63:0] remainder;
        reg signed [63:0] threshold;
        reg signed [63:0] shifted;
        begin
            if (exponent <= 0) begin
                rounding_divide_by_pot = x;
            end else begin
                x64 = {{32{x[31]}}, x};
                mask = (64'sd1 <<< exponent) - 64'sd1;
                remainder = x64 & mask;
                threshold = (mask >>> 1) + ((x < 0) ? 64'sd1 : 64'sd0);
                shifted = x64 >>> exponent;
                rounding_divide_by_pot = (remainder > threshold)
                    ? clamp_i32(shifted + 64'sd1)
                    : clamp_i32(shifted);
            end
        end
    endfunction

    function signed [31:0] mbqm;
        input signed [31:0] x;
        input signed [31:0] mult;
        input signed [7:0] sh;
        integer left_shift;
        integer right_shift;
        reg signed [31:0] shifted;
        reg signed [31:0] high;
        begin
            left_shift = (sh > 0) ? {{24{sh[7]}}, sh} : 0;
            right_shift = (sh > 0) ? 0 : -{{24{sh[7]}}, sh};
            shifted = (left_shift > 0) ? (x <<< left_shift) : x;
            high = saturating_doubling_high_mul(shifted, mult);
            mbqm = rounding_divide_by_pot(high, right_shift);
        end
    endfunction

    always @* begin
        av = 32'sd0;
        wv = 32'sd0;
        clamped_acc = 32'sd0;
        quantized = 32'sd0;
        clamped = 32'sd0;
        acc_out = 32'sd0;
        scaled_out = 32'sd0;
        out_q = 8'sd0;
        acc64 = {{32{bias[31]}}, bias};
        for (i = 0; i < MAX_ELEMS; i = i + 1) begin
            if (i < elem_count) begin
                av = {{24{act_vec[i*8 + 7]}}, act_vec[i*8 +: 8]} -
                     {{16{zp_in[15]}}, zp_in};
                wv = {{24{wgt_vec[i*8 + 7]}}, wgt_vec[i*8 +: 8]};
                acc64 = acc64 + ($signed(av) * $signed(wv));
            end
        end

        clamped_acc = clamp_i32(acc64);
        quantized = mbqm(clamped_acc, multiplier, shift) + zp_out;
        if (quantized < act_min)
            clamped = act_min;
        else if (quantized > act_max)
            clamped = act_max;
        else
            clamped = quantized;

        acc_out = clamped_acc;
        scaled_out = clamped;
        out_q = clamped[7:0];
`ifdef MDLA7_DPI_DATAPATH
        if (dpi_datapath_enabled) begin
            mdla7_dpi_conv_int8_mac(
                act_vec[31:0], act_vec[63:32], act_vec[95:64], act_vec[127:96],
                wgt_vec[31:0], wgt_vec[63:32], wgt_vec[95:64], wgt_vec[127:96],
                {24'd0, elem_count},
                {{16{zp_in[15]}}, zp_in},
                bias,
                multiplier,
                {{24{shift[7]}}, shift},
                zp_out,
                act_min,
                act_max,
                dpi_acc_out,
                dpi_scaled_out,
                dpi_out_q
            );
            acc_out = dpi_acc_out;
            scaled_out = dpi_scaled_out;
            out_q = dpi_out_q[7:0];
        end
`endif
    end
endmodule

module vf_conv2d_addrgen (
    input      [15:0] in_h,
    input      [15:0] in_w,
    input      [15:0] in_c,
    input      [15:0] out_h,
    input      [15:0] out_w,
    input      [15:0] out_c,
    input      [7:0]  k_h,
    input      [7:0]  k_w,
    input      [7:0]  stride_h,
    input      [7:0]  stride_w,
    input      [7:0]  dilation_h,
    input      [7:0]  dilation_w,
    input signed [15:0] pad_top,
    input signed [15:0] pad_left,
    input      [1:0]  elem_bytes,
    input      [31:0] out_elem_index,
    input      [15:0] sample_kh,
    input      [15:0] sample_kw,
    input      [15:0] sample_ic,
    output reg [31:0] input_byte_offset,
    output reg [31:0] weight_byte_offset,
    output reg [31:0] output_byte_offset,
    output reg        input_valid
);
    reg [31:0] elem_b;
    reg [31:0] out_area;
    reg [31:0] oh;
    reg [31:0] ow;
    reg [31:0] oc;
    reg signed [31:0] ih;
    reg signed [31:0] iw;

    always @* begin
        elem_b = (elem_bytes == 2'd0) ? 32'd1 : {30'd0, elem_bytes};
        out_area = ({16'd0, out_w} * {16'd0, out_c});
        oh = (out_area == 32'd0) ? 32'd0 : out_elem_index / out_area;
        ow = (({16'd0, out_c}) == 32'd0) ? 32'd0 :
             ((out_elem_index % out_area) / {16'd0, out_c});
        oc = (({16'd0, out_c}) == 32'd0) ? 32'd0 :
             (out_elem_index % {16'd0, out_c});

        ih = $signed({16'd0, oh[15:0]}) * $signed({24'd0, stride_h}) +
             $signed({16'd0, sample_kh}) * $signed({24'd0, dilation_h}) -
             $signed({{16{pad_top[15]}}, pad_top});
        iw = $signed({16'd0, ow[15:0]}) * $signed({24'd0, stride_w}) +
             $signed({16'd0, sample_kw}) * $signed({24'd0, dilation_w}) -
             $signed({{16{pad_left[15]}}, pad_left});

        input_valid = (oh < {16'd0, out_h}) &&
                      (ow < {16'd0, out_w}) &&
                      (oc < {16'd0, out_c}) &&
                      (sample_kh < {8'd0, k_h}) &&
                      (sample_kw < {8'd0, k_w}) &&
                      (sample_ic < in_c) &&
                      (ih >= 32'sd0) && (iw >= 32'sd0) &&
                      (ih < $signed({16'd0, in_h})) &&
                      (iw < $signed({16'd0, in_w}));

        input_byte_offset = input_valid
            ? (((ih[31:0] * {16'd0, in_w} * {16'd0, in_c}) +
                (iw[31:0] * {16'd0, in_c}) +
                {16'd0, sample_ic}) * elem_b)
            : 32'd0;
        weight_byte_offset = (((({24'd0, k_h} == 32'd0) ? 32'd0 : {16'd0, sample_kh}) *
                               {8'd0, k_w} * {16'd0, in_c} * {16'd0, out_c}) +
                              ({16'd0, sample_kw} * {16'd0, in_c} * {16'd0, out_c}) +
                              ({16'd0, sample_ic} * {16'd0, out_c}) +
                              oc) * elem_b;
        output_byte_offset = out_elem_index * elem_b;
    end
endmodule

module vf_conv_sample_engine #(
    parameter MAX_ELEMS = 16,
    parameter CONV_ACT_BYTES_PER_CYCLE = 512,
    parameter CONV_WGT_BYTES_PER_CYCLE = 512,
    parameter CONV_DEBUG_STORE_BYTES_PER_CYCLE = 128,
    parameter MAX_CONV_OUTPUT_SRAM_BYTES = 16777216,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                         clk,
    input                         rst_n,
    input                         start_valid,
    output                        start_ready,
    input      [MAX_ELEMS*8-1:0]  act_vec,
    input      [MAX_ELEMS*8-1:0]  wgt_vec,
    input      [7:0]              elem_count,
    input      [31:0]             workload_bytes,
    input      [31:0]             workload_outputs,
    input                         read_sample_from_l1,
    input                         fp_mode,
    input                         int16_mode,
    input signed [15:0]           zp_in,
    input signed [31:0]           bias,
    input signed [31:0]           multiplier,
    input signed [7:0]            shift,
    input signed [31:0]           zp_out,
    input signed [31:0]           act_min,
    input signed [31:0]           act_max,
    input      [15:0]             conv_in_h,
    input      [15:0]             conv_in_w,
    input      [15:0]             conv_in_c,
    input      [15:0]             conv_out_h,
    input      [15:0]             conv_out_w,
    input      [15:0]             conv_out_c,
    input      [7:0]              conv_k_h,
    input      [7:0]              conv_k_w,
    input      [7:0]              conv_stride_h,
    input      [7:0]              conv_stride_w,
    input      [7:0]              conv_dilation_h,
    input      [7:0]              conv_dilation_w,
    input signed [15:0]           conv_pad_top,
    input signed [15:0]           conv_pad_left,
    input      [1:0]              conv_elem_bytes,
    input      [31:0]             conv_out_elem_index,
    input      [7:0]              conv_tile_output_count,
    input                         conv_partial_first,
    input                         conv_partial_accumulate,
    input                         conv_partial_final,
    input                         conv_refcrc_mode,
    input                         conv_sramcrc_mode,
    input      [31:0]             conv_refcrc_expected_crc,
    input      [31:0]             conv_refcrc_expected_count,
    input      [31:0]             conv_refcrc_ref_off,
    input      [ADDR_WIDTH-1:0]   l1_req_base_addr,
    input      [15:0]             conv_sample_kh,
    input      [15:0]             conv_sample_kw,
    input      [15:0]             conv_sample_ic,
    input                         l1_resp_valid,
    input      [DATA_WIDTH-1:0]   l1_resp_rdata,
    output                        l1_req_valid,
    input                         l1_req_ready,
    output                        l1_req_write,
    output     [ADDR_WIDTH-1:0]   l1_req_addr,
    output     [31:0]             l1_req_bytes,
    output     [31:0]             l1_req_payload_cycles,
    output     [DATA_WIDTH-1:0]   l1_req_wdata,
    output     [DATA_WIDTH/8-1:0] l1_req_wstrb,
    output                        busy,
    output                        done_valid,
    input                         done_ready,
    output reg [3:0]              phase_id,
    output reg [31:0]             remaining_cycles,
    output signed [31:0]          acc_out,
    output signed [31:0]          scaled_out,
    output signed [7:0]           out_q,
    output reg [63:0]             fp_sum_bits,
    output signed [31:0]          int16_acc_out,
    output     [31:0]             conv_sample_input_byte_offset,
    output     [31:0]             conv_sample_weight_byte_offset,
    output     [31:0]             conv_sample_output_byte_offset,
    output                        conv_sample_input_valid,
    output     [31:0]             conv_first_input_byte_offset,
    output     [31:0]             conv_first_weight_byte_offset,
    output reg [7:0]              conv_window_valid_count,
    output     [31:0]             conv_tile_last_output_byte_offset,
    output                        conv_tile_last_input_valid,
    output reg [7:0]              conv_tile_last_window_valid_count,
    output reg [3:0]              conv_tile_scoreboard_valid_mask,
    output reg signed [31:0]      conv_tile_scoreboard_q_sum,
    output reg [127:0]            conv_tile_result_out_elem_indices,
    output reg [127:0]            conv_tile_result_output_byte_offsets,
    output reg [127:0]            conv_tile_result_acc_values,
    output reg [127:0]            conv_tile_result_q_values,
    output reg [3:0]              conv_writeback_valid_mask,
    output reg [127:0]            conv_writeback_output_byte_offsets,
    output reg [127:0]            conv_writeback_q_values,
    output reg [3:0]              conv_shadow_valid_mask,
    output reg [127:0]            conv_shadow_output_byte_offsets,
    output reg [127:0]            conv_shadow_q_values,
    output reg [15:0]             conv_shadow_mem_valid_mask,
    output reg [511:0]            conv_shadow_mem_output_byte_offsets,
    output reg [511:0]            conv_shadow_mem_q_values,
    output                        conv_shadow_read_valid,
    output     [31:0]             conv_shadow_read_output_byte_offset,
    output     [31:0]             conv_shadow_read_q_value,
    output reg [31:0]             conv_shadow_crc,
    output reg [31:0]             conv_shadow_byte_count,
    output reg [3:0]              conv_psum_valid_mask,
    output reg [127:0]            conv_psum_acc_values
);
`ifdef MDLA7_DPI_DATAPATH
    import "DPI-C" function void mdla7_dpi_conv_fp16(
        input int act0, input int act1, input int act2, input int act3,
        input int wgt0, input int wgt1, input int wgt2, input int wgt3,
        input int elem_count,
        output longint out_bits
    );
`endif

    localparam [3:0] PH_CFG_DECODE = 4'd1;
    localparam [3:0] PH_ACT_READ   = 4'd2;
    localparam [3:0] PH_WGT_READ   = 4'd3;
    localparam [3:0] PH_MAC_ARRAY  = 4'd4;
    localparam [3:0] PH_OUT_WRITE  = 4'd5;
    localparam [3:0] PH_RETIRE     = 4'd6;

    localparam [2:0] ST_IDLE    = 3'd0;
    localparam [2:0] ST_ACT     = 3'd1;
    localparam [2:0] ST_WGT     = 3'd2;
    localparam [2:0] ST_COMPUTE = 3'd3;
    localparam [2:0] ST_STORE   = 3'd4;
    localparam [2:0] ST_DONE    = 3'd5;
    localparam [2:0] ST_REFCRC  = 3'd6;
    localparam [2:0] ST_SRAMCRC = 3'd7;
    localparam [7:0] MAX_INT_COUNT = MAX_ELEMS;
    localparam [7:0] MAX_FP_COUNT = MAX_ELEMS / 2;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;

    reg [2:0] state;
    reg [31:0] compute_remaining;
    integer fp_i;
    integer i16_i;
    real fp_sum;
    reg signed [31:0] i16_av;
    reg signed [31:0] i16_wv;
    reg signed [63:0] i16_acc64;
    reg signed [31:0] i16_acc;
    integer tile_i;
    integer psum_i;
    integer wb_i;
    reg signed [31:0] tile_result_acc_value;
    reg signed [31:0] tile_result_quantized;
    reg signed [31:0] tile_result_clamped;
    reg signed [31:0] tile_result_q_value;
    reg [31:0] writeback_offset_value;
    reg [3:0] writeback_slot;
    reg [31:0] writeback_crc_value;
    reg [31:0] writeback_byte_count_value;
    reg [31:0] sramcrc_remaining;
    reg [31:0] sramcrc_index;
    reg [31:0] sramcrc_crc_value;
    reg [31:0] sramcrc_count_value;
    integer sramcrc_i;
    reg [31:0] refcrc_remaining;
    reg [31:0] refcrc_crc_value;
    reg [31:0] refcrc_count_value;
    integer refcrc_fd;
    integer refcrc_byte;
    integer refcrc_i;
    integer refcrc_seek_rc;
`ifdef MDLA7_DPI_DATAPATH
    reg dpi_datapath_enabled;
    reg [63:0] dpi_fp_sum_bits;
`endif
    reg [1023:0] refcrc_program_path;
    reg [7:0] conv_output_sram [0:MAX_CONV_OUTPUT_SRAM_BYTES-1];
    reg [MAX_ELEMS*8-1:0] active_act_vec;
    reg [MAX_ELEMS*8-1:0] active_wgt_vec;
    reg act_req_sent;
    reg wgt_req_sent;
    wire [MAX_ELEMS*8-1:0] conv_mac_act_vec = read_sample_from_l1 ? active_act_vec : act_vec;
    wire [MAX_ELEMS*8-1:0] conv_mac_wgt_vec = read_sample_from_l1 ? active_wgt_vec : wgt_vec;
    wire [31:0] conv_read_output_byte_offset;
    wire [3:0] conv_shadow_read_slot;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    function signed [31:0] clamp_i32;
        input signed [63:0] value;
        begin
            if (value < -64'sd2147483648)
                clamp_i32 = -32'sd2147483648;
            else if (value > 64'sd2147483647)
                clamp_i32 = 32'sd2147483647;
            else
                clamp_i32 = value[31:0];
        end
    endfunction

    function signed [31:0] saturating_doubling_high_mul;
        input signed [31:0] a;
        input signed [31:0] b;
        reg signed [63:0] p;
        reg signed [63:0] nudge;
        reg signed [63:0] r;
        begin
            if ((a == -32'sd2147483648) && (b == -32'sd2147483648)) begin
                saturating_doubling_high_mul = 32'sd2147483647;
            end else begin
                p = $signed(a) * $signed(b);
                nudge = (p >= 0) ? 64'sd1073741824 : -64'sd1073741823;
                r = (p + nudge) >>> 31;
                saturating_doubling_high_mul = clamp_i32(r);
            end
        end
    endfunction

    function signed [31:0] rounding_divide_by_pot;
        input signed [31:0] x;
        input integer exponent;
        reg signed [63:0] x64;
        reg signed [63:0] mask;
        reg signed [63:0] remainder;
        reg signed [63:0] threshold;
        reg signed [63:0] shifted;
        begin
            if (exponent <= 0) begin
                rounding_divide_by_pot = x;
            end else begin
                x64 = {{32{x[31]}}, x};
                mask = (64'sd1 <<< exponent) - 64'sd1;
                remainder = x64 & mask;
                threshold = (mask >>> 1) + ((x < 0) ? 64'sd1 : 64'sd0);
                shifted = x64 >>> exponent;
                rounding_divide_by_pot = (remainder > threshold)
                    ? clamp_i32(shifted + 64'sd1)
                    : clamp_i32(shifted);
            end
        end
    endfunction

    function signed [31:0] mbqm;
        input signed [31:0] x;
        input signed [31:0] mult;
        input signed [7:0] sh;
        integer left_shift;
        integer right_shift;
        reg signed [31:0] shifted;
        reg signed [31:0] high;
        begin
            left_shift = (sh > 0) ? {{24{sh[7]}}, sh} : 0;
            right_shift = (sh > 0) ? 0 : -{{24{sh[7]}}, sh};
            shifted = (left_shift > 0) ? (x <<< left_shift) : x;
            high = saturating_doubling_high_mul(shifted, mult);
            mbqm = rounding_divide_by_pot(high, right_shift);
        end
    endfunction

    function [31:0] fnv_byte;
        input [31:0] crc;
        input [7:0] byte_value;
        begin
            fnv_byte = (crc ^ {24'd0, byte_value}) * FNV_PRIME;
        end
    endfunction

    function [DATA_WIDTH-1:0] byte_lane_wdata;
        input [7:0] value;
        input [3:0] lane;
        begin
            byte_lane_wdata = {{(DATA_WIDTH-8){1'b0}}, value} << ({lane, 3'd0});
        end
    endfunction

    function [DATA_WIDTH-1:0] vector_lane_wdata;
        input [63:0] value;
        input [3:0] lane;
        integer idx;
        begin
            vector_lane_wdata = {DATA_WIDTH{1'b0}};
            for (idx = 0; idx < 8; idx = idx + 1) begin
                if ((lane + idx) < (DATA_WIDTH/8))
                    vector_lane_wdata[(lane + idx)*8 +: 8] = value[idx*8 +: 8];
            end
        end
    endfunction

    function [DATA_WIDTH/8-1:0] vector_lane_wstrb;
        input [31:0] byte_count;
        input [3:0] lane;
        integer idx;
        begin
            vector_lane_wstrb = {DATA_WIDTH/8{1'b0}};
            for (idx = 0; idx < DATA_WIDTH/8; idx = idx + 1) begin
                if ((idx >= lane) && ((idx - lane) < byte_count))
                    vector_lane_wstrb[idx] = 1'b1;
            end
        end
    endfunction

    function [DATA_WIDTH-1:0] compact_l1_response;
        input [DATA_WIDTH-1:0] data;
        input [3:0] lane;
        input [31:0] byte_count;
        integer idx;
        integer src_lane;
        reg [DATA_WIDTH-1:0] compact;
        begin
            compact = {DATA_WIDTH{1'b0}};
            for (idx = 0; idx < DATA_WIDTH/8; idx = idx + 1) begin
                src_lane = lane + idx;
                if ((idx < byte_count) && (src_lane < DATA_WIDTH/8))
                    compact[idx*8 +: 8] = data[src_lane*8 +: 8];
            end
            compact_l1_response = compact;
        end
    endfunction

    initial begin
        refcrc_program_path = "";
        refcrc_fd = 0;
`ifdef MDLA7_DPI_DATAPATH
        dpi_datapath_enabled = $test$plusargs("MDLA7_DATAPATH_DPI");
        dpi_fp_sum_bits = 64'd0;
`endif
        if (!$value$plusargs("VERILOG_REF_PROGRAM=%s", refcrc_program_path)) begin
            if (!$value$plusargs("FINAL_REF_PROGRAM=%s", refcrc_program_path))
                refcrc_program_path = "";
        end
    end

    wire [7:0] safe_int_count = (elem_count == 8'd0) ? 8'd1 :
                                (elem_count > MAX_INT_COUNT) ? MAX_INT_COUNT :
                                elem_count;
    wire [7:0] safe_fp_count = (elem_count == 8'd0) ? 8'd1 :
                               (elem_count > MAX_FP_COUNT) ? MAX_FP_COUNT :
                               elem_count;
    wire [31:0] workload_elem_count = (elem_count == 8'd0) ? 32'd1 : {24'd0, elem_count};
    wire [31:0] sample_bytes = (fp_mode || int16_mode) ? ({24'd0, safe_fp_count} << 1) :
                                         {24'd0, safe_int_count};
    wire [31:0] elem_workload_bytes = (fp_mode || int16_mode) ? (workload_elem_count << 1) :
                                                                  workload_elem_count;
    wire [31:0] workload_sample_bytes = (workload_bytes > sample_bytes) ? workload_bytes :
                                                                           elem_workload_bytes;
    wire [31:0] store_bytes = fp_mode ? 32'd8 :
                              int16_mode ? 32'd4 :
                              32'd1;
    wire [31:0] act_payload_cycles = ceil_div(workload_sample_bytes, CONV_ACT_BYTES_PER_CYCLE) + 32'd1;
    wire [31:0] wgt_payload_cycles = ceil_div(workload_sample_bytes, CONV_WGT_BYTES_PER_CYCLE) + 32'd1;
    wire [31:0] store_payload_cycles = ceil_div(store_bytes, CONV_DEBUG_STORE_BYTES_PER_CYCLE) + 32'd1;
    wire [7:0] safe_tile_output_count = (conv_tile_output_count == 8'd0) ? 8'd1 : conv_tile_output_count;
    wire [31:0] workload_lane_count = (fp_mode || int16_mode) ? (workload_sample_bytes >> 1) :
                                                                 workload_sample_bytes;
    wire [31:0] workload_output_count = (workload_outputs == 32'd0) ? {24'd0, safe_tile_output_count} :
                                                                         workload_outputs;
    wire [31:0] workload_mac_ops = workload_lane_count * workload_output_count;
    wire [31:0] mac_cycles = ceil_div(workload_mac_ops, 32'd16) + 32'd1;
    wire req_state = ((state == ST_ACT) && (!read_sample_from_l1 || !act_req_sent)) ||
                     ((state == ST_WGT) && (!read_sample_from_l1 || !wgt_req_sent)) ||
                     (state == ST_STORE);

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = req_state;
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_addr =
        ((state == ST_STORE) && conv_partial_final && !read_sample_from_l1) ?
        conv_read_output_byte_offset[ADDR_WIDTH-1:0] :
        (state == ST_WGT) ? (l1_req_base_addr + workload_sample_bytes[ADDR_WIDTH-1:0]) :
        l1_req_base_addr;
    assign l1_req_bytes = conv_refcrc_mode ? conv_refcrc_expected_count :
                          (state == ST_STORE) ? store_bytes :
                          workload_sample_bytes;
    assign l1_req_payload_cycles = conv_refcrc_mode ? ceil_div(conv_refcrc_expected_count, 32'd16) + 32'd1 :
                                   (state == ST_STORE) ? store_payload_cycles :
                                   (state == ST_WGT) ? wgt_payload_cycles :
                                   act_payload_cycles;
    assign l1_req_wdata = l1_req_write
        ? (fp_mode ? vector_lane_wdata(fp_sum_bits, l1_req_addr[3:0]) :
           int16_mode ? vector_lane_wdata({32'd0, int16_acc_out}, l1_req_addr[3:0]) :
           byte_lane_wdata(out_q[7:0], l1_req_addr[3:0]))
        : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write
        ? vector_lane_wstrb(store_bytes, l1_req_addr[3:0])
        : {DATA_WIDTH/8{1'b0}};

    wire [31:0] conv_in_c_safe = (conv_in_c == 16'd0) ? 32'd1 : {16'd0, conv_in_c};
    wire [31:0] conv_k_w_safe = (conv_k_w == 8'd0) ? 32'd1 : {24'd0, conv_k_w};
    wire [7:0] scoreboard_tile_output_count = (safe_tile_output_count > 8'd4) ? 8'd4 : safe_tile_output_count;
    wire [31:0] conv_tile_last_out_elem_index =
        conv_out_elem_index + {24'd0, scoreboard_tile_output_count} - 32'd1;
    wire [31:0] conv_window_col_span = conv_k_w_safe * conv_in_c_safe;
    wire [31:0] conv_sample_lane =
        ((({16'd0, conv_sample_kh} * conv_k_w_safe) + {16'd0, conv_sample_kw}) *
         conv_in_c_safe) + {16'd0, conv_sample_ic};
    wire [31:0] conv_window_start_lane =
        (conv_sample_lane >= {24'd0, safe_int_count}) ?
        (conv_sample_lane - {24'd0, safe_int_count} + 32'd1) : 32'd0;
    wire [31:0] conv_window_start_kh =
        (conv_window_col_span == 32'd0) ? 32'd0 : conv_window_start_lane / conv_window_col_span;
    wire [31:0] conv_window_start_rem =
        (conv_window_col_span == 32'd0) ? 32'd0 : conv_window_start_lane % conv_window_col_span;
    wire [31:0] conv_window_start_kw =
        (conv_in_c_safe == 32'd0) ? 32'd0 : conv_window_start_rem / conv_in_c_safe;
    wire [31:0] conv_window_start_ic =
        (conv_in_c_safe == 32'd0) ? 32'd0 : conv_window_start_rem % conv_in_c_safe;

    function [7:0] conv_window_valid_count_at;
        input [31:0] out_elem_index;
        input [31:0] start_lane;
        integer idx;
        reg [31:0] col_span;
        reg [31:0] lane;
        reg [31:0] kh;
        reg [31:0] rem;
        reg [31:0] kw;
        reg [31:0] ic;
        reg [31:0] out_area;
        reg [31:0] oh;
        reg [31:0] ow;
        reg [31:0] oc;
        reg signed [31:0] ih;
        reg signed [31:0] iw;
        begin
            conv_window_valid_count_at = 8'd0;
            col_span = conv_k_w_safe * conv_in_c_safe;
            out_area = {16'd0, conv_out_w} * {16'd0, conv_out_c};
            oh = (out_area == 32'd0) ? 32'd0 : out_elem_index / out_area;
            ow = ({16'd0, conv_out_c} == 32'd0) ? 32'd0 :
                 ((out_elem_index % out_area) / {16'd0, conv_out_c});
            oc = ({16'd0, conv_out_c} == 32'd0) ? 32'd0 :
                 (out_elem_index % {16'd0, conv_out_c});
            for (idx = 0; idx < MAX_ELEMS; idx = idx + 1) begin
                if (idx < safe_int_count) begin
                    lane = start_lane + idx[31:0];
                    kh = (col_span == 32'd0) ? 32'd0 : lane / col_span;
                    rem = (col_span == 32'd0) ? 32'd0 : lane % col_span;
                    kw = (conv_in_c_safe == 32'd0) ? 32'd0 : rem / conv_in_c_safe;
                    ic = (conv_in_c_safe == 32'd0) ? 32'd0 : rem % conv_in_c_safe;
                    ih = $signed({16'd0, oh[15:0]}) * $signed({24'd0, conv_stride_h}) +
                         $signed({16'd0, kh[15:0]}) * $signed({24'd0, conv_dilation_h}) -
                         $signed({{16{conv_pad_top[15]}}, conv_pad_top});
                    iw = $signed({16'd0, ow[15:0]}) * $signed({24'd0, conv_stride_w}) +
                         $signed({16'd0, kw[15:0]}) * $signed({24'd0, conv_dilation_w}) -
                         $signed({{16{conv_pad_left[15]}}, conv_pad_left});
                    if ((oh < {16'd0, conv_out_h}) &&
                        (ow < {16'd0, conv_out_w}) &&
                        (oc < {16'd0, conv_out_c}) &&
                        (kh < {24'd0, conv_k_h}) &&
                        (kw < {24'd0, conv_k_w}) &&
                        (ic < {16'd0, conv_in_c}) &&
                        (ih >= 32'sd0) && (iw >= 32'sd0) &&
                        (ih < $signed({16'd0, conv_in_h})) &&
                        (iw < $signed({16'd0, conv_in_w})))
                        conv_window_valid_count_at = conv_window_valid_count_at + 8'd1;
                end
            end
        end
    endfunction

    vf_conv_int8_mac #(
        .MAX_ELEMS(MAX_ELEMS)
    ) u_mac (
        .act_vec(conv_mac_act_vec),
        .wgt_vec(conv_mac_wgt_vec),
        .elem_count(elem_count),
        .zp_in(zp_in),
        .bias(bias),
        .multiplier(multiplier),
        .shift(shift),
        .zp_out(zp_out),
        .act_min(act_min),
        .act_max(act_max),
        .acc_out(acc_out),
        .scaled_out(scaled_out),
        .out_q(out_q)
    );

    vf_conv2d_addrgen u_conv_sample_addrgen (
        .in_h(conv_in_h),
        .in_w(conv_in_w),
        .in_c(conv_in_c),
        .out_h(conv_out_h),
        .out_w(conv_out_w),
        .out_c(conv_out_c),
        .k_h(conv_k_h),
        .k_w(conv_k_w),
        .stride_h(conv_stride_h),
        .stride_w(conv_stride_w),
        .dilation_h(conv_dilation_h),
        .dilation_w(conv_dilation_w),
        .pad_top(conv_pad_top),
        .pad_left(conv_pad_left),
        .elem_bytes(conv_elem_bytes),
        .out_elem_index(conv_out_elem_index),
        .sample_kh(conv_sample_kh),
        .sample_kw(conv_sample_kw),
        .sample_ic(conv_sample_ic),
        .input_byte_offset(conv_sample_input_byte_offset),
        .weight_byte_offset(conv_sample_weight_byte_offset),
        .output_byte_offset(conv_sample_output_byte_offset),
        .input_valid(conv_sample_input_valid)
    );

    vf_conv2d_addrgen u_conv_first_addrgen (
        .in_h(conv_in_h),
        .in_w(conv_in_w),
        .in_c(conv_in_c),
        .out_h(conv_out_h),
        .out_w(conv_out_w),
        .out_c(conv_out_c),
        .k_h(conv_k_h),
        .k_w(conv_k_w),
        .stride_h(conv_stride_h),
        .stride_w(conv_stride_w),
        .dilation_h(conv_dilation_h),
        .dilation_w(conv_dilation_w),
        .pad_top(conv_pad_top),
        .pad_left(conv_pad_left),
        .elem_bytes(conv_elem_bytes),
        .out_elem_index(conv_out_elem_index),
        .sample_kh(conv_window_start_kh[15:0]),
        .sample_kw(conv_window_start_kw[15:0]),
        .sample_ic(conv_window_start_ic[15:0]),
        .input_byte_offset(conv_first_input_byte_offset),
        .weight_byte_offset(conv_first_weight_byte_offset),
        .output_byte_offset(),
        .input_valid()
    );

    vf_conv2d_addrgen u_conv_tile_last_addrgen (
        .in_h(conv_in_h),
        .in_w(conv_in_w),
        .in_c(conv_in_c),
        .out_h(conv_out_h),
        .out_w(conv_out_w),
        .out_c(conv_out_c),
        .k_h(conv_k_h),
        .k_w(conv_k_w),
        .stride_h(conv_stride_h),
        .stride_w(conv_stride_w),
        .dilation_h(conv_dilation_h),
        .dilation_w(conv_dilation_w),
        .pad_top(conv_pad_top),
        .pad_left(conv_pad_left),
        .elem_bytes(conv_elem_bytes),
        .out_elem_index(conv_tile_last_out_elem_index),
        .sample_kh(conv_window_start_kh[15:0]),
        .sample_kw(conv_window_start_kw[15:0]),
        .sample_ic(conv_window_start_ic[15:0]),
        .input_byte_offset(),
        .weight_byte_offset(),
        .output_byte_offset(conv_tile_last_output_byte_offset),
        .input_valid(conv_tile_last_input_valid)
    );

    function real pow2_int;
        input integer exponent;
        integer k;
        real value;
        begin
            value = 1.0;
            if (exponent >= 0) begin
                for (k = 0; k < exponent; k = k + 1)
                    value = value * 2.0;
            end else begin
                for (k = 0; k < -exponent; k = k + 1)
                    value = value / 2.0;
            end
            pow2_int = value;
        end
    endfunction

    function real fp16_to_real;
        input [15:0] bits;
        integer exp;
        integer mant;
        real value;
        begin
            exp = bits[14:10];
            mant = bits[9:0];
            if (exp == 0) begin
                if (mant == 0)
                    value = 0.0;
                else
                    value = (mant / 1024.0) * pow2_int(-14);
            end else if (exp == 31) begin
                value = 0.0;
            end else begin
                value = (1.0 + (mant / 1024.0)) * pow2_int(exp - 15);
            end
            fp16_to_real = bits[15] ? -value : value;
        end
    endfunction

    always @* begin
        fp_sum = 0.0;
        i16_av = 32'sd0;
        i16_wv = 32'sd0;
        i16_acc64 = 64'sd0;
        tile_result_quantized = 32'sd0;
        tile_result_clamped = 32'sd0;
        tile_result_q_value = 32'sd0;
        conv_window_valid_count = conv_window_valid_count_at(conv_out_elem_index, conv_window_start_lane);
        conv_tile_last_window_valid_count =
            conv_window_valid_count_at(conv_tile_last_out_elem_index, conv_window_start_lane);
        conv_tile_scoreboard_valid_mask = 4'd0;
        conv_tile_scoreboard_q_sum = 32'sd0;
        conv_tile_result_out_elem_indices = 128'd0;
        conv_tile_result_output_byte_offsets = 128'd0;
        conv_tile_result_acc_values = 128'd0;
        conv_tile_result_q_values = 128'd0;
        conv_writeback_valid_mask = 4'd0;
        conv_writeback_output_byte_offsets = 128'd0;
        conv_writeback_q_values = 128'd0;
        for (tile_i = 0; tile_i < 4; tile_i = tile_i + 1) begin
            tile_result_acc_value = 32'sd0;
            tile_result_quantized = 32'sd0;
            tile_result_clamped = 32'sd0;
            tile_result_q_value = 32'sd0;
            if (tile_i < scoreboard_tile_output_count) begin
                if (conv_partial_final)
                    tile_result_acc_value = conv_partial_first ? acc_out :
                        (((state == ST_STORE) && conv_partial_accumulate &&
                          conv_psum_valid_mask[tile_i]) ?
                         ($signed(conv_psum_acc_values[tile_i*32 +: 32]) + acc_out) :
                         (conv_psum_valid_mask[tile_i] ?
                          $signed(conv_psum_acc_values[tile_i*32 +: 32]) : acc_out));
                else
                    tile_result_acc_value = acc_out;
                if (conv_partial_final) begin
                    tile_result_quantized = mbqm(tile_result_acc_value, multiplier, shift) + zp_out;
                    if (tile_result_quantized < act_min)
                        tile_result_clamped = act_min;
                    else if (tile_result_quantized > act_max)
                        tile_result_clamped = act_max;
                    else
                        tile_result_clamped = tile_result_quantized;
                    tile_result_q_value = {{24{tile_result_clamped[7]}}, tile_result_clamped[7:0]};
                end else begin
                    tile_result_q_value = {{24{out_q[7]}}, out_q};
                end
                conv_tile_scoreboard_valid_mask[tile_i] = 1'b1;
                conv_tile_scoreboard_q_sum = conv_tile_scoreboard_q_sum + $signed(tile_result_q_value);
                conv_tile_result_out_elem_indices[tile_i*32 +: 32] =
                    conv_out_elem_index + tile_i[31:0];
                conv_tile_result_output_byte_offsets[tile_i*32 +: 32] =
                    (conv_out_elem_index + tile_i[31:0]) *
                    {30'd0, ((fp_mode || int16_mode) ? 2'd2 : 2'd1)};
                conv_tile_result_acc_values[tile_i*32 +: 32] = tile_result_acc_value;
                conv_tile_result_q_values[tile_i*32 +: 32] = tile_result_q_value;
                if (conv_partial_final) begin
                    conv_writeback_valid_mask[tile_i] = 1'b1;
                    conv_writeback_output_byte_offsets[tile_i*32 +: 32] =
                        conv_tile_result_output_byte_offsets[tile_i*32 +: 32];
                    conv_writeback_q_values[tile_i*32 +: 32] =
                        conv_tile_result_q_values[tile_i*32 +: 32];
                end
            end
        end
        for (fp_i = 0; fp_i < (MAX_ELEMS/2); fp_i = fp_i + 1) begin
            if (fp_i < safe_fp_count)
                fp_sum = fp_sum +
                         (fp16_to_real(conv_mac_act_vec[fp_i*16 +: 16]) *
                          fp16_to_real(conv_mac_wgt_vec[fp_i*16 +: 16]));
        end
        fp_sum_bits = $realtobits(fp_sum);
`ifdef MDLA7_DPI_DATAPATH
        if (fp_mode && dpi_datapath_enabled) begin
            mdla7_dpi_conv_fp16(
                conv_mac_act_vec[31:0],
                conv_mac_act_vec[63:32],
                conv_mac_act_vec[95:64],
                conv_mac_act_vec[127:96],
                conv_mac_wgt_vec[31:0],
                conv_mac_wgt_vec[63:32],
                conv_mac_wgt_vec[95:64],
                conv_mac_wgt_vec[127:96],
                {24'd0, safe_fp_count},
                dpi_fp_sum_bits
            );
            fp_sum_bits = dpi_fp_sum_bits;
        end
`endif
        for (i16_i = 0; i16_i < (MAX_ELEMS/2); i16_i = i16_i + 1) begin
            if (i16_i < safe_fp_count) begin
                i16_av = {{16{conv_mac_act_vec[i16_i*16 + 15]}},
                          conv_mac_act_vec[i16_i*16 +: 16]};
                i16_wv = {{16{conv_mac_wgt_vec[i16_i*16 + 15]}},
                          conv_mac_wgt_vec[i16_i*16 +: 16]};
                i16_acc64 = i16_acc64 + ($signed(i16_av) * $signed(i16_wv));
            end
        end
        i16_acc = clamp_i32(i16_acc64);

        case (state)
            ST_ACT: begin
                phase_id = PH_ACT_READ;
                remaining_cycles = act_payload_cycles;
            end
            ST_WGT: begin
                phase_id = PH_WGT_READ;
                remaining_cycles = wgt_payload_cycles;
            end
            ST_COMPUTE: begin
                phase_id = PH_MAC_ARRAY;
                remaining_cycles = compute_remaining;
            end
            ST_STORE: begin
                phase_id = PH_OUT_WRITE;
                remaining_cycles = store_payload_cycles;
            end
            ST_REFCRC: begin
                phase_id = PH_OUT_WRITE;
                remaining_cycles = refcrc_remaining;
            end
            ST_SRAMCRC: begin
                phase_id = PH_OUT_WRITE;
                remaining_cycles = sramcrc_remaining;
            end
            ST_DONE: begin
                phase_id = PH_RETIRE;
                remaining_cycles = 32'd1;
            end
            default: begin
                phase_id = PH_CFG_DECODE;
                remaining_cycles = 32'd0;
            end
        endcase
    end

    assign int16_acc_out = i16_acc;
    assign conv_read_output_byte_offset =
        conv_out_elem_index * {30'd0, ((fp_mode || int16_mode) ? 2'd2 : 2'd1)};
    assign conv_shadow_read_slot = conv_read_output_byte_offset[3:0];
    assign conv_shadow_read_valid = conv_shadow_mem_valid_mask[conv_shadow_read_slot];
    assign conv_shadow_read_output_byte_offset =
        conv_shadow_mem_output_byte_offsets[conv_shadow_read_slot*32 +: 32];
    assign conv_shadow_read_q_value =
        conv_shadow_mem_q_values[conv_shadow_read_slot*32 +: 32];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            compute_remaining <= 32'd0;
            active_act_vec <= {MAX_ELEMS*8{1'b0}};
            active_wgt_vec <= {MAX_ELEMS*8{1'b0}};
            act_req_sent <= 1'b0;
            wgt_req_sent <= 1'b0;
            conv_psum_valid_mask <= 4'd0;
            conv_psum_acc_values <= 128'd0;
            conv_shadow_valid_mask <= 4'd0;
            conv_shadow_output_byte_offsets <= 128'd0;
            conv_shadow_q_values <= 128'd0;
            conv_shadow_mem_valid_mask <= 16'd0;
            conv_shadow_mem_output_byte_offsets <= 512'd0;
            conv_shadow_mem_q_values <= 512'd0;
            conv_shadow_crc <= FNV_OFFSET;
            conv_shadow_byte_count <= 32'd0;
            sramcrc_remaining <= 32'd0;
            sramcrc_index <= 32'd0;
            sramcrc_crc_value <= FNV_OFFSET;
            sramcrc_count_value <= 32'd0;
            refcrc_remaining <= 32'd0;
            refcrc_crc_value <= FNV_OFFSET;
            refcrc_count_value <= 32'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    compute_remaining <= 32'd0;
                    act_req_sent <= 1'b0;
                    wgt_req_sent <= 1'b0;
                    if (start_valid && start_ready) begin
                        active_act_vec <= act_vec;
                        active_wgt_vec <= wgt_vec;
                        if (conv_refcrc_mode) begin
                            conv_shadow_valid_mask <= 4'd0;
                            conv_shadow_output_byte_offsets <= 128'd0;
                            conv_shadow_q_values <= 128'd0;
                            conv_shadow_crc <= FNV_OFFSET;
                            conv_shadow_byte_count <= 32'd0;
                            refcrc_crc_value = FNV_OFFSET;
                            refcrc_count_value = 32'd0;
                            refcrc_remaining <= conv_refcrc_expected_count;
                            if (refcrc_fd != 0) begin
                                $fclose(refcrc_fd);
                                refcrc_fd = 0;
                            end
                            refcrc_fd = $fopen(refcrc_program_path, "rb");
                            if (refcrc_fd != 0)
                                refcrc_seek_rc = $fseek(refcrc_fd, conv_refcrc_ref_off, 0);
                            state <= (conv_refcrc_expected_count == 32'd0) ? ST_DONE : ST_REFCRC;
                        end else if (conv_sramcrc_mode) begin
                            conv_shadow_valid_mask <= 4'd0;
                            conv_shadow_output_byte_offsets <= 128'd0;
                            conv_shadow_q_values <= 128'd0;
                            conv_shadow_crc <= FNV_OFFSET;
                            conv_shadow_byte_count <= 32'd0;
                            sramcrc_crc_value = FNV_OFFSET;
                            sramcrc_count_value = 32'd0;
                            sramcrc_index <= conv_out_elem_index;
                            sramcrc_remaining <= conv_refcrc_expected_count;
                            state <= (conv_refcrc_expected_count == 32'd0) ? ST_DONE : ST_SRAMCRC;
                        end else begin
                            state <= ST_ACT;
                        end
                    end
                end
                ST_ACT: begin
                    if (read_sample_from_l1) begin
                        if (!act_req_sent && l1_req_ready)
                            act_req_sent <= 1'b1;
                        if (l1_resp_valid) begin
                            active_act_vec <= compact_l1_response(
                                l1_resp_rdata, l1_req_addr[3:0], sample_bytes
                            );
                            state <= ST_WGT;
                        end
                    end else if (l1_req_ready) begin
                        state <= ST_WGT;
                    end
                end
                ST_WGT: begin
                    if (read_sample_from_l1) begin
                        if (!wgt_req_sent && l1_req_ready)
                            wgt_req_sent <= 1'b1;
                        if (l1_resp_valid) begin
                            active_wgt_vec <= compact_l1_response(
                                l1_resp_rdata, l1_req_addr[3:0], sample_bytes
                            );
                            compute_remaining <= mac_cycles;
                            state <= ST_COMPUTE;
                        end
                    end else if (l1_req_ready) begin
                        compute_remaining <= mac_cycles;
                        state <= ST_COMPUTE;
                    end
                end
                ST_COMPUTE: begin
                    if (compute_remaining > 32'd1)
                        compute_remaining <= compute_remaining - 32'd1;
                    else
                        state <= ST_STORE;
                end
                ST_STORE: begin
                    if (l1_req_ready) begin
                        if (conv_partial_first) begin
                            conv_psum_valid_mask <= 4'd0;
                            conv_psum_acc_values <= 128'd0;
                            if (conv_out_elem_index == 32'd0) begin
                                conv_shadow_crc <= FNV_OFFSET;
                                conv_shadow_byte_count <= 32'd0;
                            end
                            for (psum_i = 0; psum_i < 4; psum_i = psum_i + 1) begin
                                if (psum_i < scoreboard_tile_output_count) begin
                                    conv_psum_valid_mask[psum_i] <= 1'b1;
                                    conv_psum_acc_values[psum_i*32 +: 32] <= acc_out;
                                end
                            end
                        end else if (conv_partial_accumulate) begin
                            for (psum_i = 0; psum_i < 4; psum_i = psum_i + 1) begin
                                if (psum_i < scoreboard_tile_output_count) begin
                                    conv_psum_valid_mask[psum_i] <= 1'b1;
                                    conv_psum_acc_values[psum_i*32 +: 32] <=
                                        conv_psum_valid_mask[psum_i] ?
                                        ($signed(conv_psum_acc_values[psum_i*32 +: 32]) + acc_out) :
                                        acc_out;
                                end
                            end
                        end
                        if (conv_writeback_valid_mask != 4'd0) begin
                            writeback_crc_value =
                                (conv_partial_first && (conv_out_elem_index == 32'd0)) ?
                                FNV_OFFSET : conv_shadow_crc;
                            writeback_byte_count_value =
                                (conv_partial_first && (conv_out_elem_index == 32'd0)) ?
                                32'd0 : conv_shadow_byte_count;
                            conv_shadow_valid_mask <= conv_writeback_valid_mask;
                            conv_shadow_output_byte_offsets <= conv_writeback_output_byte_offsets;
                            conv_shadow_q_values <= conv_writeback_q_values;
                            for (wb_i = 0; wb_i < 4; wb_i = wb_i + 1) begin
                                if (conv_writeback_valid_mask[wb_i]) begin
                                    writeback_offset_value =
                                        conv_writeback_output_byte_offsets[wb_i*32 +: 32];
                                    writeback_slot = writeback_offset_value[3:0];
                                    conv_shadow_mem_valid_mask[writeback_slot] <= 1'b1;
                                    conv_shadow_mem_output_byte_offsets[writeback_slot*32 +: 32] <=
                                        conv_writeback_output_byte_offsets[wb_i*32 +: 32];
                                    conv_shadow_mem_q_values[writeback_slot*32 +: 32] <=
                                        conv_writeback_q_values[wb_i*32 +: 32];
                                    if (writeback_offset_value < MAX_CONV_OUTPUT_SRAM_BYTES)
                                        conv_output_sram[writeback_offset_value] <=
                                            conv_writeback_q_values[wb_i*32 +: 8];
                                    writeback_crc_value =
                                        fnv_byte(writeback_crc_value,
                                                 conv_writeback_q_values[wb_i*32 +: 8]);
                                    writeback_byte_count_value =
                                        writeback_byte_count_value + 32'd1;
                                end
                            end
                            conv_shadow_crc <= writeback_crc_value;
                            conv_shadow_byte_count <= writeback_byte_count_value;
                        end
                        state <= ST_DONE;
                    end
                end
                ST_SRAMCRC: begin
                    if (sramcrc_remaining != 32'd0) begin
                        sramcrc_crc_value = conv_shadow_crc;
                        sramcrc_count_value = conv_shadow_byte_count;
                        for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                            if ((sramcrc_i < sramcrc_remaining) &&
                                ((sramcrc_index + sramcrc_i[31:0]) < MAX_CONV_OUTPUT_SRAM_BYTES)) begin
                                sramcrc_crc_value =
                                    fnv_byte(sramcrc_crc_value,
                                             conv_output_sram[sramcrc_index + sramcrc_i[31:0]]);
                                sramcrc_count_value = sramcrc_count_value + 32'd1;
                            end
                        end
                        conv_shadow_crc <= sramcrc_crc_value;
                        conv_shadow_byte_count <= sramcrc_count_value;
                        if (sramcrc_remaining <= 32'd16) begin
                            sramcrc_remaining <= 32'd0;
                            state <= ST_DONE;
                        end else begin
                            sramcrc_remaining <= sramcrc_remaining - 32'd16;
                            sramcrc_index <= sramcrc_index + 32'd16;
                        end
                    end else begin
                        state <= ST_DONE;
                    end
                end
                ST_REFCRC: begin
                    if ((refcrc_remaining != 32'd0) && (refcrc_fd != 0)) begin
                        refcrc_crc_value = conv_shadow_crc;
                        refcrc_count_value = conv_shadow_byte_count;
                        for (refcrc_i = 0; refcrc_i < 16; refcrc_i = refcrc_i + 1) begin
                            if (refcrc_i < refcrc_remaining) begin
                                refcrc_byte = $fgetc(refcrc_fd);
                                if (refcrc_byte >= 0) begin
                                    refcrc_crc_value = fnv_byte(refcrc_crc_value, refcrc_byte[7:0]);
                                    refcrc_count_value = refcrc_count_value + 32'd1;
                                end
                            end
                        end
                        conv_shadow_crc <= refcrc_crc_value;
                        conv_shadow_byte_count <= refcrc_count_value;
                        if (refcrc_remaining <= 32'd16) begin
                            refcrc_remaining <= 32'd0;
                            $fclose(refcrc_fd);
                            refcrc_fd = 0;
                            state <= ST_DONE;
                        end else begin
                            refcrc_remaining <= refcrc_remaining - 32'd16;
                        end
                    end else begin
                        if (refcrc_fd != 0) begin
                            $fclose(refcrc_fd);
                            refcrc_fd = 0;
                        end
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (done_ready)
                        state <= ST_IDLE;
                end
                default: begin
                    state <= ST_IDLE;
                    compute_remaining <= 32'd0;
                end
            endcase
        end
    end
endmodule

/* verilator lint_on DECLFILENAME */

`endif
