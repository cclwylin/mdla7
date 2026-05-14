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
    // v12 Phase 3: per-OC params table loaded from L1. When param_load_mode=1
    // REQUANT enters ST_PARAM_LOAD_OC on dispatch and pulls oc_count entries
    // for mult/shift/bias_eff (SystemC layout: hdr 12B + mult[OC]*4B +
    // shift[OC]*1B + bias_eff[OC]*4B starting at param_l1_addr). PIPE then
    // reads mult_mem[oc_index]/shift_mem[oc_index]/bias_eff_mem[oc_index]
    // instead of the descriptor-direct multiplier/shift/bias ports. Sample
    // mode keeps param_load_mode=0 and the existing single-set inputs.
    input                  param_load_mode,
    input      [ADDR_WIDTH-1:0] param_l1_addr,
    input      [15:0]      oc_count,
    input      [15:0]      oc_index,
    // v12 Phase 6a: OC tile drain loop. When tile_drain_count > 1 REQUANT
    // repeats the quantise-store cycle for consecutive output bytes. Default 0
    // is treated as 1 (single-element sample mode, unchanged behaviour).
    input      [15:0]      tile_drain_count,
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
    localparam [3:0] ST_PARAM_LOAD = 4'd7;   // v12 Phase 3: load per-OC params
    // ST_PARAM_LOAD region selector. Sequenced through hdr -> mult -> shift
    // -> bias_eff in 16-byte beats; param_load_done indicates we've drained
    // the last region and can transition to ST_PARAM/ST_CHAIN_WAIT.
    localparam [1:0] PARAM_REGION_HDR  = 2'd0;
    localparam [1:0] PARAM_REGION_MULT = 2'd1;
    localparam [1:0] PARAM_REGION_SHFT = 2'd2;
    localparam [1:0] PARAM_REGION_BIAS = 2'd3;
    localparam integer MAX_OC_LAYER = 256;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam integer MAX_REQUANT_OUTPUT_SRAM_BYTES = 16777216;

    reg [3:0] state;
    reg [15:0] tile_drain_remaining;  // v12 Phase 6a: remaining tile iterations
    reg [31:0] active_out_byte_offset; // v12 Phase 6a: current write address (advances per drain)
    reg [31:0] pipe_remaining;
    reg signed [31:0] quantized;
    reg signed [31:0] clamped;
    // v12 Phase 3: per-OC param storage and ST_PARAM_LOAD bookkeeping.
    reg signed [31:0] mult_mem     [0:MAX_OC_LAYER-1];
    reg signed [7:0]  shift_mem    [0:MAX_OC_LAYER-1];
    reg signed [31:0] bias_eff_mem [0:MAX_OC_LAYER-1];
    reg signed [31:0] hdr_zp_out;
    reg signed [31:0] hdr_act_min;
    reg signed [31:0] hdr_act_max;
    reg [1:0]  param_region;          // current sub-region (HDR/MULT/SHFT/BIAS)
    reg [15:0] param_byte_offset;     // byte offset within current region
    reg        param_beat_req_sent;   // L1 handshake state
    reg signed [31:0] mbqm_mult_in;   // muxed multiplier feeding MBQM
    reg signed [7:0]  mbqm_shift_in;  // muxed shift feeding MBQM
    reg signed [31:0] mbqm_bias_in;   // muxed bias_eff (added pre-MBQM)
    reg signed [31:0] mbqm_zp_in;     // muxed zp_out
    reg signed [31:0] mbqm_actmin_in; // muxed act_min
    reg signed [31:0] mbqm_actmax_in; // muxed act_max
    integer     param_byte_i;
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
    reg  [31:0] fp_in_with_bias;
    // v12 Phase 2: FP bias add. Reinterpret active_input_value + fp_bias as
    // FP32 bit patterns, add, and feed into clamp. Uses simulation-only
    // $bitstoshortreal / $shortrealtobits; the engine targets behavioral
    // ArcSim, not synth. Sample mode descriptors leave fp_bias=0 so this
    // becomes a no-op for the legacy path.
    always @* begin
        if (fp_bias == 32'd0) begin
            fp_in_with_bias = active_input_value;
        end else begin
            fp_in_with_bias = $shortrealtobits(
                $bitstoshortreal(active_input_value) + $bitstoshortreal(fp_bias));
        end
    end
    assign fp_q_bits = fp32_to_fp16(fp_clamped_bits);

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    // v12 Phase 1: in chain mode REQUANT asserts ready while waiting for the
    // psum from CONV. Sample mode never enters ST_CHAIN_WAIT, so chain side
    // back-pressures CONV (which currently silently drops if not consumed).
    assign chain_psum_ready = (state == ST_CHAIN_WAIT);
    assign l1_req_valid = ((state == ST_PARAM) && read_input_from_l1 && !param_req_sent) ||
                          ((state == ST_PARAM_LOAD) && !param_beat_req_sent) ||
                          (state == ST_STORE);
    assign l1_req_write = (state == ST_STORE);
    // v12 Phase 6a: use active_out_byte_offset for write address (advances per tile drain).
    assign l1_req_addr =
        (state == ST_STORE)      ? active_out_byte_offset[ADDR_WIDTH-1:0] :
        (state == ST_PARAM_LOAD) ? (param_l1_addr +
                                    {{(ADDR_WIDTH-16){1'b0}}, param_byte_offset}) :
                                   l1_req_base_addr;
    // v12 Phase 2: FP store is 2 bytes (FP16); INT store stays 1 byte.
    // v12 Phase 3: PARAM_LOAD reads 16 bytes per beat (DRAM model cap).
    assign l1_req_bytes = (state == ST_PARAM)      ? 32'd4 :
                          (state == ST_PARAM_LOAD) ? 32'd16 :
                          (state == ST_STORE)      ? (fp_mode ? 32'd2 : 32'd1) :
                          32'd0;
    assign l1_req_payload_cycles =
        (state == ST_PARAM) ? ((32'd4 + READ_BYTES_PER_CYCLE - 32'd1) /
                               READ_BYTES_PER_CYCLE + 32'd1) :
        (state == ST_PARAM_LOAD) ? 32'd2 :
        (state == ST_STORE) ? ((l1_req_bytes + WRITE_BYTES_PER_CYCLE - 32'd1) /
                               WRITE_BYTES_PER_CYCLE + 32'd1) :
        32'd2;
    // FP mode: place fp_q_bits (16 bits) at the byte-aligned lane.
    // INT mode: place out_q (8 bits) at the byte lane.
    assign l1_req_wdata = l1_req_write
        ? (fp_mode ? ({{(DATA_WIDTH-16){1'b0}}, fp_q_bits} << ({active_out_byte_offset[3:0], 3'd0}))
                   : byte_lane_wdata(out_q[7:0], active_out_byte_offset[3:0]))
        : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write
        ? (fp_mode ? ({{(DATA_WIDTH/8-2){1'b0}}, 2'b11} << active_out_byte_offset[3:0])
                   : ({{(DATA_WIDTH/8-1){1'b0}}, 1'b1} << active_out_byte_offset[3:0]))
        : {DATA_WIDTH/8{1'b0}};
    assign scaled_out = clamped;
    assign out_q = clamped[7:0];
    assign fp_q = fp_q_bits;

    always @* begin
        // v12 Phase 3: pick params from per-OC table when param_load_mode=1,
        // otherwise fall through to the descriptor-direct ports (sample path).
        if (param_load_mode) begin
            mbqm_mult_in   = mult_mem    [oc_index[7:0]];
            mbqm_shift_in  = shift_mem   [oc_index[7:0]];
            mbqm_bias_in   = bias_eff_mem[oc_index[7:0]];
            mbqm_zp_in     = hdr_zp_out;
            mbqm_actmin_in = hdr_act_min;
            mbqm_actmax_in = hdr_act_max;
        end else begin
            mbqm_mult_in   = multiplier;
            mbqm_shift_in  = shift;
            mbqm_bias_in   = 32'sd0;
            mbqm_zp_in     = zp_out;
            mbqm_actmin_in = act_min;
            mbqm_actmax_in = act_max;
        end
        quantized = mbqm(active_input_value + mbqm_bias_in, mbqm_mult_in, mbqm_shift_in) + mbqm_zp_in;
        if (quantized < mbqm_actmin_in)
            clamped = mbqm_actmin_in;
        else if (quantized > mbqm_actmax_in)
            clamped = mbqm_actmax_in;
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
            ST_PARAM_LOAD: begin phase_id = PH_PARAM_FETCH; remaining_cycles = l1_req_payload_cycles; end
            ST_PIPE: begin phase_id = PH_QUANT_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_SRAMCRC: begin phase_id = PH_QUANT_PIPE; remaining_cycles = sramcrc_remaining; end
            ST_CHAIN_WAIT: begin phase_id = PH_PARAM_FETCH; remaining_cycles = 32'd1; end
            ST_DONE: begin phase_id = PH_RETIRE; remaining_cycles = 32'd1; end
            default: begin phase_id = PH_CFG_DECODE; remaining_cycles = 32'd0; end
        endcase
    end

    // v12 Phase 3: end-of-region offsets (in bytes from param_l1_addr).
    wire [15:0] param_off_mult_end  = 16'd12 + (oc_count <<< 2);
    wire [15:0] param_off_shift_end = param_off_mult_end + oc_count;
    wire [15:0] param_off_bias_end  = param_off_shift_end + (oc_count <<< 2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            tile_drain_remaining <= 16'd1;
            active_out_byte_offset <= 32'd0;
            pipe_remaining <= 32'd0;
            sramcrc_remaining <= 32'd0;
            sramcrc_index <= 32'd0;
            sramcrc_crc <= FNV_OFFSET;
            sramcrc_count <= 32'd0;
            param_req_sent <= 1'b0;
            active_input_value <= 32'sd0;
            param_region <= PARAM_REGION_HDR;
            param_byte_offset <= 16'd0;
            param_beat_req_sent <= 1'b0;
            hdr_zp_out <= 32'sd0;
            hdr_act_min <= -32'sd128;
            hdr_act_max <= 32'sd127;
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    param_req_sent <= 1'b0;
                    if (start_valid && start_ready) begin
                        // v12 Phase 6a: latch drain count; 0 treated as 1.
                        tile_drain_remaining <= (tile_drain_count == 16'd0) ? 16'd1 : tile_drain_count;
                        active_out_byte_offset <= out_byte_offset;
                        if (sramcrc_mode) begin
                            sramcrc_crc <= FNV_OFFSET;
                            sramcrc_count <= 32'd0;
                            sramcrc_index <= out_byte_offset;
                            sramcrc_remaining <= sramcrc_expected_count;
                            state <= (sramcrc_expected_count == 32'd0) ? ST_DONE : ST_SRAMCRC;
                        end else if (param_load_mode) begin
                            // v12 Phase 3: preload per-OC param table from L1.
                            param_region <= PARAM_REGION_HDR;
                            param_byte_offset <= 16'd0;
                            param_beat_req_sent <= 1'b0;
                            state <= ST_PARAM_LOAD;
                        end else if (use_chain_input) begin
                            // v12 Phase 1: pull psum from CONV chain.
                            state <= ST_CHAIN_WAIT;
                        end else begin
                            active_input_value <= input_value;
                            state <= ST_PARAM;
                        end
                    end
                end
                ST_PARAM_LOAD: begin
                    if (!param_beat_req_sent && l1_req_ready)
                        param_beat_req_sent <= 1'b1;
                    if (param_beat_req_sent && l1_resp_valid) begin
                        param_beat_req_sent <= 1'b0;
                        case (param_region)
                            PARAM_REGION_HDR: begin
                                hdr_zp_out  <= $signed(l1_resp_rdata[31:0]);
                                hdr_act_min <= $signed(l1_resp_rdata[63:32]);
                                hdr_act_max <= $signed(l1_resp_rdata[95:64]);
                                param_region <= PARAM_REGION_MULT;
                                param_byte_offset <= 16'd12;
                            end
                            PARAM_REGION_MULT: begin
                                // 4 mult entries per beat, indexed by (offset-12)/4.
                                for (param_byte_i = 0; param_byte_i < 4; param_byte_i = param_byte_i + 1) begin
                                    if ({{16{1'b0}}, param_byte_offset} + (param_byte_i * 4) < {{16{1'b0}}, param_off_mult_end}) begin
                                        mult_mem[((param_byte_offset - 16'd12) >> 2) + param_byte_i] <=
                                            $signed(l1_resp_rdata[(param_byte_i * 32) +: 32]);
                                    end
                                end
                                if ((param_byte_offset + 16'd16) >= param_off_mult_end) begin
                                    param_region <= PARAM_REGION_SHFT;
                                    param_byte_offset <= param_off_mult_end;
                                end else begin
                                    param_byte_offset <= param_byte_offset + 16'd16;
                                end
                            end
                            PARAM_REGION_SHFT: begin
                                // 16 shift entries (1 byte each) per beat.
                                for (param_byte_i = 0; param_byte_i < 16; param_byte_i = param_byte_i + 1) begin
                                    if ({{16{1'b0}}, param_byte_offset} + param_byte_i < {{16{1'b0}}, param_off_shift_end}) begin
                                        shift_mem[(param_byte_offset - param_off_mult_end) + param_byte_i] <=
                                            $signed(l1_resp_rdata[(param_byte_i * 8) +: 8]);
                                    end
                                end
                                if ((param_byte_offset + 16'd16) >= param_off_shift_end) begin
                                    param_region <= PARAM_REGION_BIAS;
                                    param_byte_offset <= param_off_shift_end;
                                end else begin
                                    param_byte_offset <= param_byte_offset + 16'd16;
                                end
                            end
                            PARAM_REGION_BIAS: begin
                                // 4 bias_eff entries per beat.
                                for (param_byte_i = 0; param_byte_i < 4; param_byte_i = param_byte_i + 1) begin
                                    if ({{16{1'b0}}, param_byte_offset} + (param_byte_i * 4) < {{16{1'b0}}, param_off_bias_end}) begin
                                        bias_eff_mem[((param_byte_offset - param_off_shift_end) >> 2) + param_byte_i] <=
                                            $signed(l1_resp_rdata[(param_byte_i * 32) +: 32]);
                                    end
                                end
                                if ((param_byte_offset + 16'd16) >= param_off_bias_end) begin
                                    // Done loading. Transition to chain or sample path.
                                    if (use_chain_input)
                                        state <= ST_CHAIN_WAIT;
                                    else begin
                                        active_input_value <= input_value;
                                        state <= ST_PARAM;
                                    end
                                end else begin
                                    param_byte_offset <= param_byte_offset + 16'd16;
                                end
                            end
                        endcase
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
                        if (active_out_byte_offset < MAX_REQUANT_OUTPUT_SRAM_BYTES) begin
                            if (fp_mode) begin
                                // v12 Phase 2: write FP16 (2 bytes, little-endian).
                                output_sram[active_out_byte_offset]      <= fp_q[7:0];
                                if (active_out_byte_offset + 32'd1 < MAX_REQUANT_OUTPUT_SRAM_BYTES)
                                    output_sram[active_out_byte_offset + 32'd1] <= fp_q[15:8];
                            end else begin
                                output_sram[active_out_byte_offset] <= out_q;
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
                    // v12 Phase 6a: tile drain loop — if more elements remain,
                    // advance the output offset by 1 (INT8) or 2 (FP16) and
                    // re-run the quantise/store cycle. Chain mode re-enters
                    // ST_CHAIN_WAIT; sample mode re-enters ST_PARAM (input_value
                    // is set from the next chain beat when we get there).
                    if (tile_drain_remaining > 16'd1) begin
                        tile_drain_remaining <= tile_drain_remaining - 16'd1;
                        active_out_byte_offset <= active_out_byte_offset +
                            (fp_mode ? 32'd2 : 32'd1);
                        if (use_chain_input)
                            state <= ST_CHAIN_WAIT;
                        else
                            state <= ST_PARAM;
                    end else if (done_ready) begin
                        state <= ST_IDLE;
                    end
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
