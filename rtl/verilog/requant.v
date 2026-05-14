`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_REQUANT_V
`define MDLA7_VERILOG_REQUANT_V

/* verilator lint_off DECLFILENAME */
module vf_requant_sample_engine #(
    parameter READ_BYTES_PER_CYCLE = 128,
    parameter WRITE_BYTES_PER_CYCLE = 128,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                  clk,
    input                  rst_n,
    input                  start_valid,
    output                 start_ready,
    input signed [31:0]    input_value,
    // v12 Phase 1: CONV->REQUANT chain input. When use_chain_input=1, ignore
    // input_value/L1 and pull psum from the chain handshake instead. Sample
    // mode keeps the existing input_value path untouched.
    input                  use_chain_input,
    input                  chain_psum_valid,
    input signed [31:0]    chain_psum_data,
    output                 chain_psum_ready,
    // v12 Phase 2: FP mode. When fp_mode=1, REQUANT skips MBQM and instead
    // interprets active_input_value as FP32 bits, clamps against fp32 act_min/
    // act_max bit patterns, casts down to FP16, and writes 2 bytes to L1.
    input signed [31:0]    multiplier,
    input signed [7:0]     shift,
    input signed [31:0]    zp_out,
    input signed [31:0]    act_min,
    input signed [31:0]    act_max,
    // v12 Phase 2: FP mode. When fp_mode=1, REQUANT skips MBQM and instead
    // does FP32(active_input_value bit-cast) + fp_bias -> clamp -> FP16.
    // act_min/act_max are reinterpreted as FP32 bit patterns. Output is 2 B
    // (FP16 lane) instead of 1 B (INT8). Sample-mode INT path unchanged.
    input                  fp_mode,
    input      [31:0]      fp_bias,
    input                  read_input_from_l1,
    input                  sramcrc_mode,
    input      [31:0]      sramcrc_expected_count,
    input      [31:0]      out_byte_offset,
    input      [ADDR_WIDTH-1:0] l1_req_base_addr,
    input                  l1_resp_valid,
    input      [127:0]     l1_resp_rdata,
    output                 l1_req_valid,
    input                  l1_req_ready,
    output                 l1_req_write,
    output     [ADDR_WIDTH-1:0] l1_req_addr,
    output     [31:0]      l1_req_bytes,
    output     [31:0]      l1_req_payload_cycles,
    output     [DATA_WIDTH-1:0] l1_req_wdata,
    output     [DATA_WIDTH/8-1:0] l1_req_wstrb,
    output                 busy,
    output                 done_valid,
    input                  done_ready,
    output reg [3:0]       phase_id,
    output reg [31:0]      remaining_cycles,
    output reg [31:0]      sramcrc_crc,
    output reg [31:0]      sramcrc_count,
    output signed [31:0]   scaled_out,
    output signed [7:0]    out_q,
    // v12 Phase 2: FP16 result. 0 in INT mode.
    output     [15:0]      fp_q
);
    localparam [3:0] PH_CFG_DECODE  = 4'd1;
    localparam [3:0] PH_PARAM_FETCH = 4'd2;
    localparam [3:0] PH_QUANT_PIPE  = 4'd3;
    localparam [3:0] PH_OUT_WRITE   = 4'd4;
    localparam [3:0] PH_RETIRE      = 4'd5;

    localparam [3:0] ST_IDLE       = 4'd0;
    localparam [3:0] ST_PARAM      = 4'd1;
    localparam [3:0] ST_PIPE       = 4'd2;
    localparam [3:0] ST_STORE      = 4'd3;
    localparam [3:0] ST_DONE       = 4'd4;
    localparam [3:0] ST_SRAMCRC    = 4'd5;
    localparam [3:0] ST_CHAIN_WAIT = 4'd6;   // v12: wait for CONV chain psum
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam integer MAX_REQUANT_OUTPUT_SRAM_BYTES = 16777216;

    reg [3:0] state;
    reg [31:0] pipe_remaining;
    reg signed [31:0] quantized;
    reg signed [31:0] clamped;
    reg [31:0] sramcrc_remaining;
    reg [31:0] sramcrc_index;
    reg [31:0] sramcrc_crc_value;
    reg [31:0] sramcrc_count_value;
    reg [7:0] output_sram [0:MAX_REQUANT_OUTPUT_SRAM_BYTES-1];
    reg param_req_sent;
    reg signed [31:0] active_input_value;
    integer sramcrc_i;

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

    // v12 Phase 2: FP32 bit pattern compare (treats both as ordered floats with
    // sign-magnitude semantics; NaN handling not required for our clamp use).
    function fp32_lt;
        input [31:0] a;
        input [31:0] b;
        reg sa, sb;
        begin
            sa = a[31];
            sb = b[31];
            if (sa != sb)
                fp32_lt = sa;                                 // negative < positive
            else if (!sa)
                fp32_lt = (a[30:0] < b[30:0]);                // both pos: smaller mag wins
            else
                fp32_lt = (a[30:0] > b[30:0]);                // both neg: larger mag wins
        end
    endfunction

    // v12 Phase 2: IEEE-754 FP32 -> FP16 with round-to-nearest-even, infinity
    // saturation, and zero-flush for subnormals (matches SystemC fp_utils).
    function [15:0] fp32_to_fp16;
        input [31:0] f32;
        reg sign;
        reg [7:0] exp32;
        reg [22:0] mant32;
        reg [4:0] exp16;
        reg [9:0] mant16;
        reg [12:0] guard;
        integer e_unbiased;
        begin
            sign  = f32[31];
            exp32 = f32[30:23];
            mant32 = f32[22:0];
            if (exp32 == 8'd0) begin
                fp32_to_fp16 = {sign, 5'd0, 10'd0};            // zero / subnormal -> 0
            end else if (exp32 == 8'hff) begin
                fp32_to_fp16 = {sign, 5'h1f, mant32[22:13]};   // inf / nan
            end else begin
                e_unbiased = $signed({1'b0, exp32}) - 127;
                if (e_unbiased > 15) begin
                    fp32_to_fp16 = {sign, 5'h1f, 10'd0};       // overflow -> +/-inf
                end else if (e_unbiased < -14) begin
                    fp32_to_fp16 = {sign, 5'd0, 10'd0};        // underflow -> +/-0
                end else begin
                    exp16  = e_unbiased[4:0] + 5'd15;
                    mant16 = mant32[22:13];
                    guard  = mant32[12:0];
                    // Round-to-nearest, ties-to-even.
                    if (guard[12] && (|guard[11:0] || mant16[0]))
                        {exp16, mant16} = {exp16, mant16} + 1;
                    fp32_to_fp16 = {sign, exp16, mant16};
                end
            end
        end
    endfunction

    // v12 Phase 2: FP datapath registers. fp_clamped_bits is the clamped FP32
    // bit pattern; fp_q is the down-cast FP16 result. INT path keeps using
    // `clamped`/`out_q` exactly as before.
    reg [31:0] fp_clamped_bits;
    wire [15:0] fp_q_bits;
    wire [31:0] fp_in_with_bias;
    // Bias add in FP mode: re-interpret active_input_value as float, then we
    // simply OR/add via a real-valued helper. To keep this synth-friendly,
    // bias is applied OUTSIDE the engine in sample mode (compile_model folds
    // it into the chain payload, same as INT). fp_bias is plumbed for future
    // use in tile/streaming mode where per-OC bias add lives inside REQUANT.
    assign fp_in_with_bias = active_input_value;
    assign fp_q_bits = fp32_to_fp16(fp_clamped_bits);

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    // v12 Phase 1: in chain mode REQUANT asserts ready while waiting for the
    // psum from CONV. Sample mode never enters ST_CHAIN_WAIT, so chain side
    // back-pressures CONV (which currently silently drops if not consumed).
    assign chain_psum_ready = (state == ST_CHAIN_WAIT);
    assign l1_req_valid = ((state == ST_PARAM) && read_input_from_l1 && !param_req_sent) ||
                          (state == ST_STORE);
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_addr = (state == ST_STORE) ? out_byte_offset[ADDR_WIDTH-1:0] : l1_req_base_addr;
    // v12 Phase 2: FP store is 2 bytes (FP16); INT store stays 1 byte.
    assign l1_req_bytes = (state == ST_PARAM) ? 32'd4 :
                          (state == ST_STORE) ? (fp_mode ? 32'd2 : 32'd1) :
                          32'd0;
    assign l1_req_payload_cycles =
        (state == ST_PARAM) ? ((32'd4 + READ_BYTES_PER_CYCLE - 32'd1) /
                               READ_BYTES_PER_CYCLE + 32'd1) :
        (state == ST_STORE) ? ((l1_req_bytes + WRITE_BYTES_PER_CYCLE - 32'd1) /
                               WRITE_BYTES_PER_CYCLE + 32'd1) :
        32'd2;
    // FP mode: place fp_q_bits (16 bits) at the byte-aligned lane.
    // INT mode: place out_q (8 bits) at the byte lane.
    assign l1_req_wdata = l1_req_write
        ? (fp_mode ? ({{(DATA_WIDTH-16){1'b0}}, fp_q_bits} << ({l1_req_addr[3:0], 3'd0}))
                   : byte_lane_wdata(out_q[7:0], l1_req_addr[3:0]))
        : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write
        ? (fp_mode ? ({{(DATA_WIDTH/8-2){1'b0}}, 2'b11} << l1_req_addr[3:0])
                   : ({{(DATA_WIDTH/8-1){1'b0}}, 1'b1} << l1_req_addr[3:0]))
        : {DATA_WIDTH/8{1'b0}};
    assign scaled_out = clamped;
    assign out_q = clamped[7:0];
    assign fp_q = fp_q_bits;

    always @* begin
        quantized = mbqm(active_input_value, multiplier, shift) + zp_out;
        if (quantized < act_min)
            clamped = act_min;
        else if (quantized > act_max)
            clamped = act_max;
        else
            clamped = quantized;

        // v12 Phase 2: FP clamp. Treats active_input_value, act_min, act_max
        // as FP32 bit patterns and picks the closest in-range value. This
        // shadow path is computed unconditionally; it only drives output when
        // fp_mode=1 (selected by l1_req_wdata mux above).
        if (fp32_lt(fp_in_with_bias, act_min))
            fp_clamped_bits = act_min;
        else if (fp32_lt(act_max, fp_in_with_bias))
            fp_clamped_bits = act_max;
        else
            fp_clamped_bits = fp_in_with_bias;

        case (state)
            ST_PARAM: begin phase_id = PH_PARAM_FETCH; remaining_cycles = l1_req_payload_cycles; end
            ST_PIPE: begin phase_id = PH_QUANT_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_SRAMCRC: begin phase_id = PH_QUANT_PIPE; remaining_cycles = sramcrc_remaining; end
            ST_CHAIN_WAIT: begin phase_id = PH_PARAM_FETCH; remaining_cycles = 32'd1; end
            ST_DONE: begin phase_id = PH_RETIRE; remaining_cycles = 32'd1; end
            default: begin phase_id = PH_CFG_DECODE; remaining_cycles = 32'd0; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            pipe_remaining <= 32'd0;
            sramcrc_remaining <= 32'd0;
            sramcrc_index <= 32'd0;
            sramcrc_crc <= FNV_OFFSET;
            sramcrc_count <= 32'd0;
            param_req_sent <= 1'b0;
            active_input_value <= 32'sd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    param_req_sent <= 1'b0;
                    if (start_valid && start_ready) begin
                        if (sramcrc_mode) begin
                            sramcrc_crc <= FNV_OFFSET;
                            sramcrc_count <= 32'd0;
                            sramcrc_index <= out_byte_offset;
                            sramcrc_remaining <= sramcrc_expected_count;
                            state <= (sramcrc_expected_count == 32'd0) ? ST_DONE : ST_SRAMCRC;
                        end else if (use_chain_input) begin
                            // v12 Phase 1: pull psum from CONV chain.
                            state <= ST_CHAIN_WAIT;
                        end else begin
                            active_input_value <= input_value;
                            state <= ST_PARAM;
                        end
                    end
                end
                ST_CHAIN_WAIT: begin
                    if (chain_psum_valid) begin
                        active_input_value <= chain_psum_data;
                        pipe_remaining <= 32'd2;
                        state <= ST_PIPE;
                    end
                end
                ST_PARAM: begin
                    if (read_input_from_l1) begin
                        if (!param_req_sent && l1_req_ready)
                            param_req_sent <= 1'b1;
                        if (l1_resp_valid) begin
                            case (out_byte_offset[3:0])
                                4'h0: active_input_value <= l1_resp_rdata[31:0];
                                4'h1: active_input_value <= l1_resp_rdata[39:8];
                                4'h2: active_input_value <= l1_resp_rdata[47:16];
                                4'h3: active_input_value <= l1_resp_rdata[55:24];
                                4'h4: active_input_value <= l1_resp_rdata[63:32];
                                4'h5: active_input_value <= l1_resp_rdata[71:40];
                                4'h6: active_input_value <= l1_resp_rdata[79:48];
                                4'h7: active_input_value <= l1_resp_rdata[87:56];
                                4'h8: active_input_value <= l1_resp_rdata[95:64];
                                4'h9: active_input_value <= l1_resp_rdata[103:72];
                                4'ha: active_input_value <= l1_resp_rdata[111:80];
                                4'hb: active_input_value <= l1_resp_rdata[119:88];
                                4'hc: active_input_value <= l1_resp_rdata[127:96];
                                default: active_input_value <= 32'sd0;
                            endcase
                            pipe_remaining <= 32'd2;
                            state <= ST_PIPE;
                        end
                    end else begin
                        pipe_remaining <= 32'd2;
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
                    if (l1_req_ready) begin
                        if (out_byte_offset < MAX_REQUANT_OUTPUT_SRAM_BYTES) begin
                            if (fp_mode) begin
                                // v12 Phase 2: write FP16 (2 bytes, little-endian).
                                output_sram[out_byte_offset]      <= fp_q[7:0];
                                if (out_byte_offset + 32'd1 < MAX_REQUANT_OUTPUT_SRAM_BYTES)
                                    output_sram[out_byte_offset + 32'd1] <= fp_q[15:8];
                            end else begin
                                output_sram[out_byte_offset] <= out_q;
                            end
                        end
                        state <= ST_DONE;
                    end
                end
                ST_SRAMCRC: begin
                    if (sramcrc_remaining != 32'd0) begin
                        sramcrc_crc_value = sramcrc_crc;
                        sramcrc_count_value = sramcrc_count;
                        for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                            if ((sramcrc_i < sramcrc_remaining) &&
                                ((sramcrc_index + sramcrc_i[31:0]) < MAX_REQUANT_OUTPUT_SRAM_BYTES)) begin
                                sramcrc_crc_value =
                                    fnv_byte(sramcrc_crc_value,
                                             output_sram[sramcrc_index + sramcrc_i[31:0]]);
                                sramcrc_count_value = sramcrc_count_value + 32'd1;
                            end
                        end
                        sramcrc_crc <= sramcrc_crc_value;
                        sramcrc_count <= sramcrc_count_value;
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

/* verilator lint_on DECLFILENAME */

`endif
