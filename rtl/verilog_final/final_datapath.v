`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_FINAL_DATAPATH_V
`define MDLA7_VERILOG_FINAL_DATAPATH_V

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

module vf_tnps_addrgen (
    input             mode_space_to_depth,
    input      [15:0] in_h,
    input      [15:0] in_w,
    input      [15:0] in_c,
    input      [15:0] out_h,
    input      [15:0] out_w,
    input      [15:0] out_c,
    input      [15:0] block,
    input      [1:0]  elem_bytes,
    input      [31:0] out_elem_index,
    input      [31:0] in_elem_index,
    output reg [31:0] src_byte_offset,
    output reg [31:0] dst_byte_offset,
    output reg        valid
);
    reg [31:0] elem;
    reg [31:0] oh;
    reg [31:0] ow;
    reg [31:0] oc;
    reg [31:0] ih;
    reg [31:0] iw;
    reg [31:0] ic;
    reg [31:0] bh;
    reg [31:0] bw;
    reg [31:0] q;
    reg [31:0] elem_b;

    always @* begin
        elem_b = (elem_bytes == 2'd0) ? 32'd1 : {30'd0, elem_bytes};
        src_byte_offset = 32'd0;
        dst_byte_offset = 32'd0;
        valid = 1'b0;
        elem = 32'd0;
        oh = 32'd0;
        ow = 32'd0;
        oc = 32'd0;
        ih = 32'd0;
        iw = 32'd0;
        ic = 32'd0;
        bh = 32'd0;
        bw = 32'd0;
        q = 32'd0;

        if (mode_space_to_depth) begin
            if ((in_h != 16'd0) && (in_w != 16'd0) && (in_c != 16'd0) &&
                (out_h != 16'd0) && (out_w != 16'd0) && (out_c != 16'd0) &&
                (block != 16'd0) &&
                (in_h == out_h * block) &&
                (in_w == out_w * block) &&
                (out_c == in_c * block * block) &&
                (out_elem_index < out_h * out_w * out_c)) begin
                oh = out_elem_index / ({16'd0, out_w} * {16'd0, out_c});
                elem = out_elem_index % ({16'd0, out_w} * {16'd0, out_c});
                ow = elem / {16'd0, out_c};
                oc = elem % {16'd0, out_c};
                q = oc / {16'd0, in_c};
                ic = oc % {16'd0, in_c};
                bh = q / {16'd0, block};
                bw = q % {16'd0, block};
                ih = oh * block + bh;
                iw = ow * block + bw;
                src_byte_offset = ((ih * in_w * in_c) + (iw * in_c) + ic) * elem_b;
                dst_byte_offset = out_elem_index * elem_b;
                valid = 1'b1;
            end
        end else begin
            if ((in_h != 16'd0) && (in_w != 16'd0) && (in_c != 16'd0) &&
                (out_h != 16'd0) && (out_w != 16'd0) && (out_c != 16'd0) &&
                (block != 16'd0) &&
                (out_h == in_h * block) &&
                (out_w == in_w * block) &&
                (in_c == out_c * block * block) &&
                (in_elem_index < in_h * in_w * in_c)) begin
                ih = in_elem_index / ({16'd0, in_w} * {16'd0, in_c});
                elem = in_elem_index % ({16'd0, in_w} * {16'd0, in_c});
                iw = elem / {16'd0, in_c};
                ic = elem % {16'd0, in_c};
                q = ic / {16'd0, out_c};
                oc = ic % {16'd0, out_c};
                bh = q / {16'd0, block};
                bw = q % {16'd0, block};
                oh = ih * block + bh;
                ow = iw * block + bw;
                src_byte_offset = in_elem_index * elem_b;
                dst_byte_offset = ((oh * out_w * out_c) + (ow * out_c) + oc) * elem_b;
                valid = 1'b1;
            end
        end
    end
endmodule

module vf_conv_sample_engine #(
    parameter MAX_ELEMS = 16,
    parameter L1_BYTES_PER_CYCLE = 256
) (
    input                         clk,
    input                         rst_n,
    input                         start_valid,
    output                        start_ready,
    input      [MAX_ELEMS*8-1:0]  act_vec,
    input      [MAX_ELEMS*8-1:0]  wgt_vec,
    input      [7:0]              elem_count,
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
    input      [15:0]             conv_sample_kh,
    input      [15:0]             conv_sample_kw,
    input      [15:0]             conv_sample_ic,
    output                        l1_req_valid,
    input                         l1_req_ready,
    output                        l1_req_write,
    output     [31:0]             l1_req_bytes,
    output     [31:0]             l1_req_payload_cycles,
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
    output reg [3:0]              conv_psum_valid_mask,
    output reg [127:0]            conv_psum_acc_values
);
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
    localparam [7:0] MAX_INT_COUNT = MAX_ELEMS;
    localparam [7:0] MAX_FP_COUNT = MAX_ELEMS / 2;

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
    reg signed [31:0] tile_result_acc_value;

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

    wire [7:0] safe_int_count = (elem_count == 8'd0) ? 8'd1 :
                                (elem_count > MAX_INT_COUNT) ? MAX_INT_COUNT :
                                elem_count;
    wire [7:0] safe_fp_count = (elem_count == 8'd0) ? 8'd1 :
                               (elem_count > MAX_FP_COUNT) ? MAX_FP_COUNT :
                               elem_count;
    wire [31:0] sample_bytes = (fp_mode || int16_mode) ? ({24'd0, safe_fp_count} << 1) :
                                         {24'd0, safe_int_count};
    wire [31:0] payload_cycles = ceil_div(sample_bytes, L1_BYTES_PER_CYCLE) + 32'd1;
    wire [31:0] mac_cycles = ceil_div(fp_mode ? {24'd0, safe_fp_count} :
                                      int16_mode ? {24'd0, safe_fp_count} :
                                                   {24'd0, safe_int_count}, 32'd16) + 32'd1;
    wire req_state = (state == ST_ACT) || (state == ST_WGT) || (state == ST_STORE);

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = req_state;
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_bytes = sample_bytes;
    assign l1_req_payload_cycles = payload_cycles;

    wire [31:0] conv_in_c_safe = (conv_in_c == 16'd0) ? 32'd1 : {16'd0, conv_in_c};
    wire [31:0] conv_k_w_safe = (conv_k_w == 8'd0) ? 32'd1 : {24'd0, conv_k_w};
    wire [7:0] safe_tile_output_count = (conv_tile_output_count == 8'd0) ? 8'd1 : conv_tile_output_count;
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
        .act_vec(act_vec),
        .wgt_vec(wgt_vec),
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
        conv_window_valid_count = conv_window_valid_count_at(conv_out_elem_index, conv_window_start_lane);
        conv_tile_last_window_valid_count =
            conv_window_valid_count_at(conv_tile_last_out_elem_index, conv_window_start_lane);
        conv_tile_scoreboard_valid_mask = 4'd0;
        conv_tile_scoreboard_q_sum = 32'sd0;
        conv_tile_result_out_elem_indices = 128'd0;
        conv_tile_result_output_byte_offsets = 128'd0;
        conv_tile_result_acc_values = 128'd0;
        conv_tile_result_q_values = 128'd0;
        for (tile_i = 0; tile_i < 4; tile_i = tile_i + 1) begin
            tile_result_acc_value = 32'sd0;
            if (tile_i < scoreboard_tile_output_count) begin
                if (conv_partial_final)
                    tile_result_acc_value = conv_psum_valid_mask[tile_i] ?
                        $signed(conv_psum_acc_values[tile_i*32 +: 32]) : acc_out;
                else
                    tile_result_acc_value = acc_out;
                conv_tile_scoreboard_valid_mask[tile_i] = 1'b1;
                conv_tile_scoreboard_q_sum = conv_tile_scoreboard_q_sum + $signed(out_q);
                conv_tile_result_out_elem_indices[tile_i*32 +: 32] =
                    conv_out_elem_index + tile_i[31:0];
                conv_tile_result_output_byte_offsets[tile_i*32 +: 32] =
                    (conv_out_elem_index + tile_i[31:0]) *
                    {30'd0, ((fp_mode || int16_mode) ? 2'd2 : 2'd1)};
                conv_tile_result_acc_values[tile_i*32 +: 32] = tile_result_acc_value;
                conv_tile_result_q_values[tile_i*32 +: 32] =
                    {{24{out_q[7]}}, out_q};
            end
        end
        for (fp_i = 0; fp_i < (MAX_ELEMS/2); fp_i = fp_i + 1) begin
            if (fp_i < safe_fp_count)
                fp_sum = fp_sum +
                         (fp16_to_real(act_vec[fp_i*16 +: 16]) *
                          fp16_to_real(wgt_vec[fp_i*16 +: 16]));
        end
        fp_sum_bits = $realtobits(fp_sum);
        for (i16_i = 0; i16_i < (MAX_ELEMS/2); i16_i = i16_i + 1) begin
            if (i16_i < safe_fp_count) begin
                i16_av = {{16{act_vec[i16_i*16 + 15]}}, act_vec[i16_i*16 +: 16]};
                i16_wv = {{16{wgt_vec[i16_i*16 + 15]}}, wgt_vec[i16_i*16 +: 16]};
                i16_acc64 = i16_acc64 + ($signed(i16_av) * $signed(i16_wv));
            end
        end
        i16_acc = clamp_i32(i16_acc64);

        case (state)
            ST_ACT: begin
                phase_id = PH_ACT_READ;
                remaining_cycles = payload_cycles;
            end
            ST_WGT: begin
                phase_id = PH_WGT_READ;
                remaining_cycles = payload_cycles;
            end
            ST_COMPUTE: begin
                phase_id = PH_MAC_ARRAY;
                remaining_cycles = compute_remaining;
            end
            ST_STORE: begin
                phase_id = PH_OUT_WRITE;
                remaining_cycles = payload_cycles;
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

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            compute_remaining <= 32'd0;
            conv_psum_valid_mask <= 4'd0;
            conv_psum_acc_values <= 128'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    compute_remaining <= 32'd0;
                    if (start_valid && start_ready)
                        state <= ST_ACT;
                end
                ST_ACT: begin
                    if (l1_req_ready)
                        state <= ST_WGT;
                end
                ST_WGT: begin
                    if (l1_req_ready) begin
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

module vf_requant_sample_engine #(
    parameter WRITE_BYTES_PER_CYCLE = 64
) (
    input                  clk,
    input                  rst_n,
    input                  start_valid,
    output                 start_ready,
    input signed [31:0]    input_value,
    input signed [31:0]    multiplier,
    input signed [7:0]     shift,
    input signed [31:0]    zp_out,
    input signed [31:0]    act_min,
    input signed [31:0]    act_max,
    output                 l1_req_valid,
    input                  l1_req_ready,
    output                 l1_req_write,
    output     [31:0]      l1_req_bytes,
    output     [31:0]      l1_req_payload_cycles,
    output                 busy,
    output                 done_valid,
    input                  done_ready,
    output reg [3:0]       phase_id,
    output reg [31:0]      remaining_cycles,
    output signed [31:0]   scaled_out,
    output signed [7:0]    out_q
);
    localparam [3:0] PH_CFG_DECODE  = 4'd1;
    localparam [3:0] PH_PARAM_FETCH = 4'd2;
    localparam [3:0] PH_QUANT_PIPE  = 4'd3;
    localparam [3:0] PH_OUT_WRITE   = 4'd4;
    localparam [3:0] PH_RETIRE      = 4'd5;

    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_PARAM = 3'd1;
    localparam [2:0] ST_PIPE  = 3'd2;
    localparam [2:0] ST_STORE = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;

    reg [2:0] state;
    reg [31:0] pipe_remaining;
    reg signed [31:0] quantized;
    reg signed [31:0] clamped;

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

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = (state == ST_STORE);
    assign l1_req_write = 1'b1;
    assign l1_req_bytes = 32'd1;
    assign l1_req_payload_cycles = 32'd2;
    assign scaled_out = clamped;
    assign out_q = clamped[7:0];

    always @* begin
        quantized = mbqm(input_value, multiplier, shift) + zp_out;
        if (quantized < act_min)
            clamped = act_min;
        else if (quantized > act_max)
            clamped = act_max;
        else
            clamped = quantized;

        case (state)
            ST_PARAM: begin phase_id = PH_PARAM_FETCH; remaining_cycles = 32'd2; end
            ST_PIPE: begin phase_id = PH_QUANT_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_DONE: begin phase_id = PH_RETIRE; remaining_cycles = 32'd1; end
            default: begin phase_id = PH_CFG_DECODE; remaining_cycles = 32'd0; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            pipe_remaining <= 32'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    if (start_valid && start_ready)
                        state <= ST_PARAM;
                end
                ST_PARAM: begin
                    pipe_remaining <= 32'd2;
                    state <= ST_PIPE;
                end
                ST_PIPE: begin
                    if (pipe_remaining > 32'd1)
                        pipe_remaining <= pipe_remaining - 32'd1;
                    else
                        state <= ST_STORE;
                end
                ST_STORE: begin
                    if (l1_req_ready)
                        state <= ST_DONE;
                end
                ST_DONE: begin
                    if (done_ready)
                        state <= ST_IDLE;
                end
                default: begin
                    state <= ST_IDLE;
                    pipe_remaining <= 32'd0;
                end
            endcase
        end
    end
endmodule

module vf_pool_sample_engine #(
    parameter MAX_ELEMS = 16
) (
    input                         clk,
    input                         rst_n,
    input                         start_valid,
    output                        start_ready,
    input                         avg_mode,
    input                         fp_mode,
    input                         int16_mode,
    input      [MAX_ELEMS*8-1:0]  sample_vec,
    input      [7:0]              elem_count,
    output                        l1_req_valid,
    input                         l1_req_ready,
    output                        l1_req_write,
    output     [31:0]             l1_req_bytes,
    output     [31:0]             l1_req_payload_cycles,
    output                        busy,
    output                        done_valid,
    input                         done_ready,
    output reg [3:0]              phase_id,
    output reg [31:0]             remaining_cycles,
    output reg signed [31:0]      pool_out,
    output signed [7:0]           out_q,
    output reg [63:0]             fp_pool_bits
);
    localparam [3:0] PH_CFG_DECODE   = 4'd1;
    localparam [3:0] PH_WINDOW_FETCH = 4'd2;
    localparam [3:0] PH_REDUCE_PIPE  = 4'd3;
    localparam [3:0] PH_OUT_WRITE    = 4'd4;
    localparam [3:0] PH_RETIRE       = 4'd5;

    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_FETCH = 3'd1;
    localparam [2:0] ST_PIPE  = 3'd2;
    localparam [2:0] ST_STORE = 3'd3;
    localparam [2:0] ST_DONE  = 3'd4;
    localparam [7:0] MAX_COUNT = MAX_ELEMS;
    localparam [7:0] MAX_FP_COUNT = MAX_ELEMS / 2;

    reg [2:0] state;
    reg [31:0] pipe_remaining;
    integer i;
    integer fp_i;
    integer i16_i;
    reg signed [31:0] value;
    reg signed [31:0] sum;
    reg signed [31:0] max_value;
    reg signed [31:0] avg_value;
    reg signed [31:0] signed_count;
    reg signed [31:0] i16_value;
    reg signed [31:0] i16_sum;
    reg signed [31:0] i16_max_value;
    reg signed [31:0] i16_avg_value;
    reg signed [31:0] signed_i16_count;
    real fp_sum;
    real fp_value;
    real fp_max_value;
    real fp_pool_value;
    wire [7:0] safe_count = (elem_count == 8'd0) ? 8'd1 :
                             (elem_count > MAX_COUNT) ? MAX_COUNT :
                             elem_count;
    wire [7:0] safe_fp_count = (elem_count == 8'd0) ? 8'd1 :
                               (elem_count > MAX_FP_COUNT) ? MAX_FP_COUNT :
                               elem_count;

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = (state == ST_FETCH) || (state == ST_STORE);
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_bytes = (state == ST_FETCH) ? ((fp_mode || int16_mode) ? ({24'd0, safe_fp_count} << 1) :
                                                                          {24'd0, safe_count}) :
                          (fp_mode ? 32'd8 : (int16_mode ? 32'd4 : 32'd1));
    assign l1_req_payload_cycles = 32'd2;
    assign out_q = pool_out[7:0];

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
        sum = 32'sd0;
        max_value = -32'sd128;
        signed_count = {24'd0, safe_count};
        i16_value = 32'sd0;
        i16_sum = 32'sd0;
        i16_max_value = -32'sd32768;
        i16_avg_value = 32'sd0;
        signed_i16_count = {24'd0, safe_fp_count};
        fp_sum = 0.0;
        fp_max_value = -1.0e300;
        for (i = 0; i < MAX_ELEMS; i = i + 1) begin
            if (i < safe_count) begin
                value = {{24{sample_vec[i*8 + 7]}}, sample_vec[i*8 +: 8]};
                sum = sum + value;
                if (value > max_value)
                    max_value = value;
            end
        end
        for (fp_i = 0; fp_i < (MAX_ELEMS/2); fp_i = fp_i + 1) begin
            if (fp_i < safe_fp_count) begin
                fp_value = fp16_to_real(sample_vec[fp_i*16 +: 16]);
                fp_sum = fp_sum + fp_value;
                if (fp_value > fp_max_value)
                    fp_max_value = fp_value;
            end
        end
        for (i16_i = 0; i16_i < (MAX_ELEMS/2); i16_i = i16_i + 1) begin
            if (i16_i < safe_fp_count) begin
                i16_value = {{16{sample_vec[i16_i*16 + 15]}}, sample_vec[i16_i*16 +: 16]};
                i16_sum = i16_sum + i16_value;
                if (i16_value > i16_max_value)
                    i16_max_value = i16_value;
            end
        end
        avg_value = sum / signed_count;
        pool_out = avg_mode ? avg_value : max_value;
        fp_pool_value = avg_mode ? (fp_sum / safe_fp_count) : fp_max_value;
        fp_pool_bits = $realtobits(fp_pool_value);
        i16_avg_value = i16_sum / signed_i16_count;
        if (int16_mode)
            pool_out = avg_mode ? i16_avg_value : i16_max_value;

        case (state)
            ST_FETCH: begin phase_id = PH_WINDOW_FETCH; remaining_cycles = l1_req_payload_cycles; end
            ST_PIPE: begin phase_id = PH_REDUCE_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_DONE: begin phase_id = PH_RETIRE; remaining_cycles = 32'd1; end
            default: begin phase_id = PH_CFG_DECODE; remaining_cycles = 32'd0; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            pipe_remaining <= 32'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    if (start_valid && start_ready)
                        state <= ST_FETCH;
                end
                ST_FETCH: begin
                    if (l1_req_ready) begin
                        pipe_remaining <= {24'd0, (fp_mode || int16_mode) ? safe_fp_count : safe_count} + 32'd1;
                        state <= ST_PIPE;
                    end
                end
                ST_PIPE: begin
                    if (pipe_remaining > 32'd1)
                        pipe_remaining <= pipe_remaining - 32'd1;
                    else
                        state <= ST_STORE;
                end
                ST_STORE: begin
                    if (l1_req_ready)
                        state <= ST_DONE;
                end
                ST_DONE: begin
                    if (done_ready)
                        state <= ST_IDLE;
                end
                default: begin
                    state <= ST_IDLE;
                    pipe_remaining <= 32'd0;
                end
            endcase
        end
    end
endmodule

module vf_ewe_sample_engine #(
    parameter MAX_ELEMS = 16
) (
    input                         clk,
    input                         rst_n,
    input                         start_valid,
    output                        start_ready,
    input      [1:0]              op_mode,
    input                         fp_mode,
    input                         int16_mode,
    input      [MAX_ELEMS*8-1:0]  a_vec,
    input      [MAX_ELEMS*8-1:0]  b_vec,
    input      [7:0]              elem_count,
    output                        l1_req_valid,
    input                         l1_req_ready,
    output                        l1_req_write,
    output     [31:0]             l1_req_bytes,
    output     [31:0]             l1_req_payload_cycles,
    output                        busy,
    output                        done_valid,
    input                         done_ready,
    output reg [3:0]              phase_id,
    output reg [31:0]             remaining_cycles,
    output reg signed [31:0]      ewe_out,
    output signed [7:0]           out_q,
    output reg [63:0]             fp_ewe_bits
);
    localparam [3:0] PH_CFG_DECODE = 4'd1;
    localparam [3:0] PH_A_READ     = 4'd2;
    localparam [3:0] PH_B_READ     = 4'd3;
    localparam [3:0] PH_LANE_PIPE  = 4'd4;
    localparam [3:0] PH_OUT_WRITE  = 4'd5;
    localparam [3:0] PH_RETIRE     = 4'd6;

    localparam [2:0] ST_IDLE  = 3'd0;
    localparam [2:0] ST_A     = 3'd1;
    localparam [2:0] ST_B     = 3'd2;
    localparam [2:0] ST_PIPE  = 3'd3;
    localparam [2:0] ST_STORE = 3'd4;
    localparam [2:0] ST_DONE  = 3'd5;
    localparam [7:0] MAX_COUNT = MAX_ELEMS;
    localparam [7:0] MAX_FP_COUNT = MAX_ELEMS / 2;

    reg [2:0] state;
    reg [31:0] pipe_remaining;
    reg signed [31:0] av;
    reg signed [31:0] bv;
    reg signed [31:0] raw;
    reg signed [31:0] lane_value;
    reg signed [31:0] first_lane_value;
    integer lane;
    integer fp_lane;
    integer i16_lane;
    real fp_av;
    real fp_bv;
    real fp_raw;
    real fp_sum;
    reg signed [31:0] i16_av;
    reg signed [31:0] i16_bv;
    reg signed [31:0] i16_raw;
    reg signed [31:0] i16_first_lane_value;
    reg signed [31:0] i16_sum;
    wire [7:0] safe_count = (elem_count == 8'd0) ? 8'd1 :
                             (elem_count > MAX_COUNT) ? MAX_COUNT :
                             elem_count;
    wire [7:0] safe_fp_count = (elem_count == 8'd0) ? 8'd1 :
                               (elem_count > MAX_FP_COUNT) ? MAX_FP_COUNT :
                               elem_count;

    function signed [31:0] clamp_i8;
        input signed [31:0] value;
        begin
            if (value < -32'sd128)
                clamp_i8 = -32'sd128;
            else if (value > 32'sd127)
                clamp_i8 = 32'sd127;
            else
                clamp_i8 = value;
        end
    endfunction

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

    function real logistic_real;
        input real value;
        begin
            logistic_real = 1.0 / (1.0 + $exp(-value));
        end
    endfunction

    task compute_lane;
        input integer lane_idx;
        begin
            av = {{24{a_vec[lane_idx*8 + 7]}}, a_vec[lane_idx*8 +: 8]};
            bv = {{24{b_vec[lane_idx*8 + 7]}}, b_vec[lane_idx*8 +: 8]};
            case (op_mode)
                2'd1: raw = av * bv;
                2'd2: raw = av - bv;
                default: raw = av + bv;
            endcase
            lane_value = clamp_i8(raw);
        end
    endtask

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = (state == ST_A) || (state == ST_B) || (state == ST_STORE);
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_bytes = (fp_mode || int16_mode) ? ({24'd0, safe_fp_count} << 1) :
                                                   {24'd0, safe_count};
    assign l1_req_payload_cycles = 32'd2;
    assign out_q = first_lane_value[7:0];

    always @* begin
        av = 32'sd0;
        bv = 32'sd0;
        raw = 32'sd0;
        lane_value = 32'sd0;
        first_lane_value = 32'sd0;
        ewe_out = 32'sd0;
        fp_sum = 0.0;
        fp_av = 0.0;
        fp_bv = 0.0;
        fp_raw = 0.0;
        i16_av = 32'sd0;
        i16_bv = 32'sd0;
        i16_raw = 32'sd0;
        i16_first_lane_value = 32'sd0;
        i16_sum = 32'sd0;
        for (lane = 0; lane < MAX_ELEMS; lane = lane + 1) begin
            if (lane < safe_count) begin
                compute_lane(lane);
                if (lane == 0)
                    first_lane_value = lane_value;
                ewe_out = ewe_out + lane_value;
            end
        end
        for (fp_lane = 0; fp_lane < (MAX_ELEMS/2); fp_lane = fp_lane + 1) begin
            if (fp_lane < safe_fp_count) begin
                fp_av = fp16_to_real(a_vec[fp_lane*16 +: 16]);
                fp_bv = fp16_to_real(b_vec[fp_lane*16 +: 16]);
                case (op_mode)
                    2'd1: fp_raw = fp_av * fp_bv;
                    2'd2: fp_raw = fp_av - fp_bv;
                    2'd3: fp_raw = logistic_real(fp_av);
                    default: fp_raw = fp_av + fp_bv;
                endcase
                fp_sum = fp_sum + fp_raw;
            end
        end
        fp_ewe_bits = $realtobits(fp_sum);
        for (i16_lane = 0; i16_lane < (MAX_ELEMS/2); i16_lane = i16_lane + 1) begin
            if (i16_lane < safe_fp_count) begin
                i16_av = {{16{a_vec[i16_lane*16 + 15]}}, a_vec[i16_lane*16 +: 16]};
                i16_bv = {{16{b_vec[i16_lane*16 + 15]}}, b_vec[i16_lane*16 +: 16]};
                case (op_mode)
                    2'd1: i16_raw = i16_av * i16_bv;
                    2'd2: i16_raw = i16_av - i16_bv;
                    default: i16_raw = i16_av + i16_bv;
                endcase
                if (i16_lane == 0)
                    i16_first_lane_value = i16_raw;
                i16_sum = i16_sum + i16_raw;
            end
        end
        if (int16_mode) begin
            first_lane_value = i16_first_lane_value;
            ewe_out = i16_sum;
        end

        case (state)
            ST_A: begin phase_id = PH_A_READ; remaining_cycles = l1_req_payload_cycles; end
            ST_B: begin phase_id = PH_B_READ; remaining_cycles = l1_req_payload_cycles; end
            ST_PIPE: begin phase_id = PH_LANE_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_DONE: begin phase_id = PH_RETIRE; remaining_cycles = 32'd1; end
            default: begin phase_id = PH_CFG_DECODE; remaining_cycles = 32'd0; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            pipe_remaining <= 32'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    if (start_valid && start_ready)
                        state <= ST_A;
                end
                ST_A: begin
                    if (l1_req_ready)
                        state <= ST_B;
                end
                ST_B: begin
                    if (l1_req_ready) begin
                        pipe_remaining <= {24'd0, (fp_mode || int16_mode) ? safe_fp_count : safe_count} + 32'd1;
                        state <= ST_PIPE;
                    end
                end
                ST_PIPE: begin
                    if (pipe_remaining > 32'd1)
                        pipe_remaining <= pipe_remaining - 32'd1;
                    else
                        state <= ST_STORE;
                end
                ST_STORE: begin
                    if (l1_req_ready)
                        state <= ST_DONE;
                end
                ST_DONE: begin
                    if (done_ready)
                        state <= ST_IDLE;
                end
                default: begin
                    state <= ST_IDLE;
                    pipe_remaining <= 32'd0;
                end
            endcase
        end
    end
endmodule

module vf_udma_engine #(
    parameter L1_BYTES_PER_CYCLE = 256,
    parameter DRAM_BYTES_PER_CYCLE = 48,
    parameter DRAM_STARTUP_CYCLES = 50,
    parameter DRAM_CMD_CYCLES = 8
) (
    input             clk,
    input             rst_n,
    input             start_valid,
    output            start_ready,
    input             direction_write,
    input      [31:0] bytes,
    input      [31:0] dram_read_bytes,
    input      [31:0] codec_cycles,
    output            l1_req_valid,
    input             l1_req_ready,
    output            l1_req_write,
    output     [31:0] l1_req_bytes,
    output     [31:0] l1_req_payload_cycles,
    output            busy,
    output            done_valid,
    input             done_ready,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles
);
    localparam [3:0] PH_CFG_DECODE       = 4'd1;
    localparam [3:0] PH_L1_PAYLOAD_READ  = 4'd2;
    localparam [3:0] PH_CODEC_PIPE       = 4'd3;
    localparam [3:0] PH_DRAM_CMD         = 4'd4;
    localparam [3:0] PH_DRAM_WRITE_DATA  = 4'd5;
    localparam [3:0] PH_DRAM_READ_DATA   = 4'd6;
    localparam [3:0] PH_L1_PAYLOAD_WRITE = 4'd7;
    localparam [3:0] PH_RETIRE           = 4'd8;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    wire [31:0] effective_dram_read_bytes =
        (dram_read_bytes == 32'd0) ? bytes : dram_read_bytes;
    wire [31:0] l1_payload_cycles = ceil_div(bytes, L1_BYTES_PER_CYCLE) + 32'd1;
    wire [31:0] codec_pipe_cycles = (codec_cycles == 32'd0) ? 32'd0 : (codec_cycles + 32'd1);
    wire [31:0] dram_write_cycles =
        ceil_div(bytes, DRAM_BYTES_PER_CYCLE) + DRAM_STARTUP_CYCLES;
    wire [31:0] dram_read_cycles =
        ceil_div(effective_dram_read_bytes, DRAM_BYTES_PER_CYCLE) + DRAM_STARTUP_CYCLES;

    wire [31:0] phase1_cycles = direction_write ? l1_payload_cycles : DRAM_CMD_CYCLES;
    wire [31:0] phase2_cycles = direction_write ? codec_pipe_cycles : dram_read_cycles;
    wire [31:0] phase3_cycles = direction_write ? DRAM_CMD_CYCLES : codec_pipe_cycles;
    wire [31:0] phase4_cycles = direction_write ? dram_write_cycles : l1_payload_cycles;

    wire [3:0] phase1_id = direction_write ? PH_L1_PAYLOAD_READ : PH_DRAM_CMD;
    wire [3:0] phase2_id = direction_write ? PH_CODEC_PIPE : PH_DRAM_READ_DATA;
    wire [3:0] phase3_id = direction_write ? PH_DRAM_CMD : PH_CODEC_PIPE;
    wire [3:0] phase4_id = direction_write ? PH_DRAM_WRITE_DATA : PH_L1_PAYLOAD_WRITE;

    wire [6*32-1:0] phase_cycles = {
        32'd1,
        phase4_cycles,
        phase3_cycles,
        phase2_cycles,
        phase1_cycles,
        32'd2
    };

    wire [6*4-1:0] phase_ids = {
        PH_RETIRE,
        phase4_id,
        phase3_id,
        phase2_id,
        phase1_id,
        PH_CFG_DECODE
    };

    wire payload_phase_active = busy &&
        ((phase_id == PH_L1_PAYLOAD_READ) || (phase_id == PH_L1_PAYLOAD_WRITE));
    reg payload_token_sent;
    wire payload_token_fire = l1_req_valid && l1_req_ready;
    wire start_fire = start_valid && start_ready;
    wire phase_stall = payload_phase_active && !payload_token_sent && !l1_req_ready;

    assign l1_req_valid = payload_phase_active && !payload_token_sent;
    assign l1_req_write = !direction_write;
    assign l1_req_bytes = bytes;
    assign l1_req_payload_cycles = l1_payload_cycles;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            payload_token_sent <= 1'b0;
        else if (start_fire)
            payload_token_sent <= 1'b0;
        else if (payload_token_fire)
            payload_token_sent <= 1'b1;
        else if (!payload_phase_active)
            payload_token_sent <= 1'b0;
    end

    mdla7_synth_phase_engine #(
        .NUM_PHASES(6),
        .PHASE_W(4)
    ) u_phase (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .phase_cycles(phase_cycles),
        .phase_ids(phase_ids),
        .phase_stall(phase_stall),
        .busy(busy),
        .done_valid(done_valid),
        .done_ready(done_ready),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );
endmodule

module vf_tnps_engine #(
    parameter READ_PORTS = 2,
    parameter WRITE_PORTS = 2,
    parameter PAYLOAD_BYTES = 32,
    parameter PERMUTE_BYTES_PER_CYCLE = 128
) (
    input             clk,
    input             rst_n,
    input             start_valid,
    output            start_ready,
    input      [31:0] bytes,
    input             mode_space_to_depth,
    input      [15:0] in_h,
    input      [15:0] in_w,
    input      [15:0] in_c,
    input      [15:0] out_h,
    input      [15:0] out_w,
    input      [15:0] out_c,
    input      [15:0] block,
    input      [1:0]  elem_bytes,
    input      [31:0] sample_out_elem_index,
    input      [31:0] sample_in_elem_index,
    output            l1_req_valid,
    input             l1_req_ready,
    output            l1_req_write,
    output     [31:0] l1_req_bytes,
    output     [31:0] l1_req_payload_cycles,
    output            busy,
    output            done_valid,
    input             done_ready,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles,
    output     [31:0] sample_src_byte_offset,
    output     [31:0] sample_dst_byte_offset,
    output            sample_valid
);
    localparam [3:0] PH_CFG_DECODE    = 4'd1;
    localparam [3:0] PH_PAYLOAD_READ  = 4'd2;
    localparam [3:0] PH_PERMUTE_PIPE  = 4'd3;
    localparam [3:0] PH_PAYLOAD_WRITE = 4'd4;
    localparam [3:0] PH_RETIRE        = 4'd5;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    vf_tnps_addrgen u_addrgen (
        .mode_space_to_depth(mode_space_to_depth),
        .in_h(in_h),
        .in_w(in_w),
        .in_c(in_c),
        .out_h(out_h),
        .out_w(out_w),
        .out_c(out_c),
        .block(block),
        .elem_bytes(elem_bytes),
        .out_elem_index(sample_out_elem_index),
        .in_elem_index(sample_in_elem_index),
        .src_byte_offset(sample_src_byte_offset),
        .dst_byte_offset(sample_dst_byte_offset),
        .valid(sample_valid)
    );

    wire [31:0] payload_read_cycles = ceil_div(bytes, READ_PORTS * PAYLOAD_BYTES) + 32'd1;
    wire [31:0] permute_pipe_cycles = ceil_div(bytes, PERMUTE_BYTES_PER_CYCLE) + 32'd2;
    wire [31:0] payload_write_cycles = ceil_div(bytes, WRITE_PORTS * PAYLOAD_BYTES) + 32'd1;
    wire start_fire = start_valid && start_ready;

    wire [5*32-1:0] phase_cycles = {
        32'd1,
        payload_write_cycles,
        permute_pipe_cycles,
        payload_read_cycles,
        32'd2
    };

    wire [5*4-1:0] phase_ids = {
        PH_RETIRE,
        PH_PAYLOAD_WRITE,
        PH_PERMUTE_PIPE,
        PH_PAYLOAD_READ,
        PH_CFG_DECODE
    };

    wire payload_phase_active = busy &&
        ((phase_id == PH_PAYLOAD_READ) || (phase_id == PH_PAYLOAD_WRITE));
    reg payload_token_sent;
    wire payload_token_fire = l1_req_valid && l1_req_ready;
    wire phase_stall = payload_phase_active && !payload_token_sent && !l1_req_ready;

    assign l1_req_valid = payload_phase_active && !payload_token_sent;
    assign l1_req_write = (phase_id == PH_PAYLOAD_WRITE);
    assign l1_req_bytes = bytes;
    assign l1_req_payload_cycles = (phase_id == PH_PAYLOAD_WRITE)
        ? payload_write_cycles
        : payload_read_cycles;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            payload_token_sent <= 1'b0;
        else if (start_fire)
            payload_token_sent <= 1'b0;
        else if (payload_token_fire)
            payload_token_sent <= 1'b1;
        else if (!payload_phase_active)
            payload_token_sent <= 1'b0;
    end

    mdla7_synth_phase_engine #(
        .NUM_PHASES(5),
        .PHASE_W(4)
    ) u_phase (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .phase_cycles(phase_cycles),
        .phase_ids(phase_ids),
        .phase_stall(phase_stall),
        .busy(busy),
        .done_valid(done_valid),
        .done_ready(done_ready),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );
endmodule

module vf_l1mesh_route_estimator #(
    parameter BASE_CYCLES = 1,
    parameter GLOBAL_HOP_CYCLES = 1,
    parameter LOCAL_HOP_CYCLES = 1
) (
    input      [3:0]  source_id,
    input      [21:0] addr,
    output reg [31:0] route_cycles,
    output reg [1:0]  source_x,
    output reg [1:0]  source_y,
    output     [1:0]  tile_x,
    output     [1:0]  tile_y,
    output     [1:0]  bank_x,
    output     [1:0]  bank_y
);
    /* verilator lint_off UNUSEDSIGNAL */
    wire [15:0] addr_unused = {addr[21:10], addr[3:0]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [5:0] bank_global = addr[9:4];
    wire [1:0] tile_id = bank_global[5:4];
    wire [3:0] bank_id = bank_global[3:0];
    wire [31:0] global_dx;
    wire [31:0] global_dy;
    wire [31:0] local_hops;

    assign tile_x = {1'b0, tile_id[0]};
    assign tile_y = {1'b0, tile_id[1]};
    assign bank_x = bank_id[1:0];
    assign bank_y = bank_id[3:2];
    assign global_dx = (source_x > tile_x)
        ? ({30'd0, source_x} - {30'd0, tile_x})
        : ({30'd0, tile_x} - {30'd0, source_x});
    assign global_dy = (source_y > tile_y)
        ? ({30'd0, source_y} - {30'd0, tile_y})
        : ({30'd0, tile_y} - {30'd0, source_y});
    assign local_hops = {30'd0, bank_x} + {30'd0, bank_y};

    always @* begin
        case (source_id)
            4'd1: begin source_x = 2'd0; source_y = 2'd0; end // CONV
            4'd2: begin source_x = 2'd1; source_y = 2'd0; end // REQUANT
            4'd3: begin source_x = 2'd0; source_y = 2'd1; end // EWE
            4'd4: begin source_x = 2'd1; source_y = 2'd1; end // POOL
            4'd5: begin source_x = 2'd0; source_y = 2'd1; end // TNPS
            4'd6: begin source_x = 2'd1; source_y = 2'd0; end // UDMA
            default: begin source_x = 2'd0; source_y = 2'd0; end
        endcase
        route_cycles = BASE_CYCLES +
                       (global_dx + global_dy) * GLOBAL_HOP_CYCLES +
                       local_hops * LOCAL_HOP_CYCLES;
    end
endmodule
/* verilator lint_on DECLFILENAME */

`endif
