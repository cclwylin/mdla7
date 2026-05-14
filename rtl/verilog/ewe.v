`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_EWE_V
`define MDLA7_VERILOG_EWE_V

/* verilator lint_off DECLFILENAME */
module vf_ewe_sample_engine #(
    parameter MAX_ELEMS = 16,
    parameter READ_BYTES_PER_CYCLE = 256,
    parameter WRITE_BYTES_PER_CYCLE = 128,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                         clk,
    input                         rst_n,
    input                         start_valid,
    output                        start_ready,
    input      [1:0]              op_mode,
    input                         fp_mode,
    input                         int16_mode,
    // v11: when high, this descriptor is a unary INT8 LUT lookup. b_vec
    // carries the precomputed LUT[a_vec] bytes (one per active lane); the
    // engine's role is to validate dispatch and timing -- the actual LUT
    // memory + L1 load belong to a future full-vector engine block.
    // v11: when high, this descriptor is a unary INT8 LUT lookup. The 256-byte
    // LUT lives in L1 at lut_addr; the engine loads it on entry (16 beats x
    // 16B reads), then for each sampled input byte does lut_mem[uint8(av)]
    // and writes the result back. Generator no longer pre-stuffs b_vec.
    input                         lut_mode,
    input      [ADDR_WIDTH-1:0]   lut_addr,
    input                         final_q_mode,
    input                         read_a_from_l1,
    input                         sramcrc_mode,
    input      [31:0]             sramcrc_expected_count,
    input      [31:0]             out_byte_offset,
    input      [ADDR_WIDTH-1:0]   l1_req_base_addr,
    input signed [31:0]           zp_a,
    input signed [31:0]           zp_b,
    input signed [31:0]           zp_out,
    input signed [31:0]           mult_a,
    input signed [7:0]            shift_a,
    input signed [31:0]           mult_b,
    input signed [7:0]            shift_b,
    input signed [31:0]           mult_out,
    input signed [7:0]            shift_out,
    input signed [31:0]           left_shift,
    input signed [31:0]           act_min,
    input signed [31:0]           act_max,
    input      [MAX_ELEMS*8-1:0]  a_vec,
    input      [MAX_ELEMS*8-1:0]  b_vec,
    input      [7:0]              elem_count,
    input                         l1_resp_valid,
    input      [127:0]            l1_resp_rdata,
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
    output reg signed [31:0]      ewe_out,
    output signed [7:0]           out_q,
    output reg [31:0]             sramcrc_crc,
    output reg [31:0]             sramcrc_count,
    output reg [63:0]             fp_ewe_bits
);
    localparam [3:0] PH_CFG_DECODE = 4'd1;
    localparam [3:0] PH_A_READ     = 4'd2;
    localparam [3:0] PH_B_READ     = 4'd3;
    localparam [3:0] PH_LANE_PIPE  = 4'd4;
    localparam [3:0] PH_OUT_WRITE  = 4'd5;
    localparam [3:0] PH_RETIRE     = 4'd6;

    localparam [3:0] ST_IDLE     = 4'd0;
    localparam [3:0] ST_A        = 4'd1;
    localparam [3:0] ST_B        = 4'd2;
    localparam [3:0] ST_PIPE     = 4'd3;
    localparam [3:0] ST_STORE    = 4'd4;
    localparam [3:0] ST_DONE     = 4'd5;
    localparam [3:0] ST_SRAMCRC  = 4'd6;
    localparam [3:0] ST_LUT_LOAD = 4'd7;   // v11: pre-load 256B LUT from L1
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam integer MAX_EWE_OUTPUT_SRAM_BYTES = 16777216;
    localparam [7:0] MAX_COUNT = MAX_ELEMS;
    localparam [7:0] MAX_FP_COUNT = MAX_ELEMS / 2;

    reg [3:0] state;
    reg [31:0] pipe_remaining;
    // v11: 256-byte unary LUT preloaded from L1 in 16 beats of 16 B each.
    // Indexed by uint8(input_byte); compile_model precomputes the table from
    // TFLite quant params so a lookup is bit-exact against the reference.
    reg [7:0] lut_mem [0:255];
    reg [4:0] lut_load_beat;          // 0..16; 16 = load done
    reg       lut_beat_req_sent;      // request fired, awaiting response
    integer   lut_byte;
    reg signed [31:0] av;
    reg signed [31:0] bv;
    reg signed [31:0] raw;
    reg signed [31:0] lane_value;
    reg signed [31:0] first_lane_value;
    reg [31:0] sramcrc_remaining;
    reg [31:0] sramcrc_index;
    reg [31:0] sramcrc_crc_value;
    reg [31:0] sramcrc_count_value;
    reg [7:0] output_sram [0:MAX_EWE_OUTPUT_SRAM_BYTES-1];
    reg [DATA_WIDTH-1:0] store_wdata_value;
    reg [DATA_WIDTH/8-1:0] store_wstrb_value;
    reg [31:0] store_byte_count_value;
    integer lane;
    integer fp_lane;
    integer i16_lane;
    integer sramcrc_i;
    real fp_av;
    real fp_bv;
    real fp_raw;
    real fp_sum;
`ifdef MDLA7_DPI_DATAPATH
    import "DPI-C" function void mdla7_dpi_ewe_fp16(
        input int avec0, input int avec1, input int avec2, input int avec3,
        input int bvec0, input int bvec1, input int bvec2, input int bvec3,
        input int elem_count, input int op_mode,
        output longint out_bits
    );
    reg dpi_datapath_enabled;
    reg [63:0] dpi_fp_ewe_bits;

    initial begin
        dpi_datapath_enabled = $test$plusargs("MDLA7_DATAPATH_DPI");
    end
`endif
    reg signed [31:0] i16_av;
    reg signed [31:0] i16_bv;
    reg signed [31:0] i16_raw;
    reg signed [31:0] i16_first_lane_value;
    reg signed [31:0] i16_sum;
    reg a_req_sent;
    reg [MAX_ELEMS*8-1:0] active_a_vec;
    reg [7:0] l1_resp_byte;
    wire [7:0] safe_count = (elem_count == 8'd0) ? 8'd1 :
                             (elem_count > MAX_COUNT) ? MAX_COUNT :
                             elem_count;
    wire [7:0] safe_fp_count = (elem_count == 8'd0) ? 8'd1 :
                               (elem_count > MAX_FP_COUNT) ? MAX_FP_COUNT :
                               elem_count;
    wire [31:0] ewe_read_bytes = (fp_mode || int16_mode) ? ({24'd0, safe_fp_count} << 1) :
                                                            {24'd0, safe_count};

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
        input [DATA_WIDTH-1:0] value;
        input [3:0] lane;
        begin
            vector_lane_wdata = value << ({lane, 3'd0});
        end
    endfunction

    function [DATA_WIDTH/8-1:0] vector_lane_wstrb;
        input [31:0] byte_count;
        input [3:0] lane;
        integer idx;
        integer absolute_lane;
        reg [DATA_WIDTH/8-1:0] mask;
        begin
            mask = {DATA_WIDTH/8{1'b0}};
            for (idx = 0; idx < DATA_WIDTH/8; idx = idx + 1) begin
                absolute_lane = lane + idx;
                if ((idx < byte_count) && (absolute_lane < DATA_WIDTH/8))
                    mask[absolute_lane] = 1'b1;
            end
            vector_lane_wstrb = mask;
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
        integer ls;
        integer rs;
        reg signed [31:0] shifted;
        reg signed [31:0] high;
        begin
            ls = (sh > 0) ? {{24{sh[7]}}, sh} : 0;
            rs = (sh > 0) ? 0 : -{{24{sh[7]}}, sh};
            shifted = (ls > 0) ? clamp_i32({{32{x[31]}}, x} <<< ls) : x;
            high = saturating_doubling_high_mul(shifted, mult);
            mbqm = rounding_divide_by_pot(high, rs);
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
            av = {{24{active_a_vec[lane_idx*8 + 7]}}, active_a_vec[lane_idx*8 +: 8]};
            bv = {{24{b_vec[lane_idx*8 + 7]}}, b_vec[lane_idx*8 +: 8]};
            if (lut_mode) begin
                // INT8 unary LUT lookup. lut_mem was preloaded from L1 in
                // ST_LUT_LOAD; index by uint8(av) and sign-extend the result.
                lane_value = clamp_i8($signed({{24{lut_mem[av[7:0]][7]}},
                                                lut_mem[av[7:0]]}));
            end else begin
            case (op_mode)
                2'd1: raw = av * bv;
                2'd2: raw = av - bv;
                default: raw = av + bv;
            endcase
            if (final_q_mode) begin
                case (op_mode)
                    2'd1: begin
                        raw = mbqm(clamp_i32((av - zp_a) * (bv - zp_b)), mult_out, shift_out) + zp_out;
                    end
                    2'd2: begin
                        raw = mbqm(
                            clamp_i32(mbqm(clamp_i32((av - zp_a) <<< left_shift), mult_a, shift_a) -
                                      mbqm(clamp_i32((bv - zp_b) <<< left_shift), mult_b, shift_b)),
                            mult_out,
                            shift_out
                        ) + zp_out;
                    end
                    default: begin
                        raw = mbqm(
                            clamp_i32(mbqm(clamp_i32((av - zp_a) <<< left_shift), mult_a, shift_a) +
                                      mbqm(clamp_i32((bv - zp_b) <<< left_shift), mult_b, shift_b)),
                            mult_out,
                            shift_out
                        ) + zp_out;
                    end
                endcase
                if (raw < act_min)
                    lane_value = act_min;
                else if (raw > act_max)
                    lane_value = act_max;
                else
                    lane_value = raw;
            end else begin
                lane_value = clamp_i8(raw);
            end
            end  // !lut_mode
        end
    endtask

    always @* begin
        case (out_byte_offset[3:0])
            4'h0: l1_resp_byte = l1_resp_rdata[7:0];
            4'h1: l1_resp_byte = l1_resp_rdata[15:8];
            4'h2: l1_resp_byte = l1_resp_rdata[23:16];
            4'h3: l1_resp_byte = l1_resp_rdata[31:24];
            4'h4: l1_resp_byte = l1_resp_rdata[39:32];
            4'h5: l1_resp_byte = l1_resp_rdata[47:40];
            4'h6: l1_resp_byte = l1_resp_rdata[55:48];
            4'h7: l1_resp_byte = l1_resp_rdata[63:56];
            4'h8: l1_resp_byte = l1_resp_rdata[71:64];
            4'h9: l1_resp_byte = l1_resp_rdata[79:72];
            4'ha: l1_resp_byte = l1_resp_rdata[87:80];
            4'hb: l1_resp_byte = l1_resp_rdata[95:88];
            4'hc: l1_resp_byte = l1_resp_rdata[103:96];
            4'hd: l1_resp_byte = l1_resp_rdata[111:104];
            4'he: l1_resp_byte = l1_resp_rdata[119:112];
            default: l1_resp_byte = l1_resp_rdata[127:120];
        endcase
    end

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = ((state == ST_A) && (!read_a_from_l1 || !a_req_sent)) ||
                          (state == ST_B) || (state == ST_STORE) ||
                          ((state == ST_LUT_LOAD) && !lut_beat_req_sent);
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_addr =
        (state == ST_STORE)    ? out_byte_offset[ADDR_WIDTH-1:0] :
        (state == ST_LUT_LOAD) ? (lut_addr +
                                  {{(ADDR_WIDTH-9){1'b0}}, lut_load_beat[3:0], 4'd0}) :
                                 l1_req_base_addr;
    assign l1_req_bytes =
        (state == ST_STORE)    ? store_byte_count_value :
        (state == ST_LUT_LOAD) ? 32'd16 :
                                 ewe_read_bytes;
    assign l1_req_payload_cycles =
        (state == ST_STORE) ? ((store_byte_count_value + WRITE_BYTES_PER_CYCLE - 32'd1) /
                               WRITE_BYTES_PER_CYCLE + 32'd1) :
        (state == ST_LUT_LOAD) ? 32'd2 :
        ((ewe_read_bytes + READ_BYTES_PER_CYCLE - 32'd1) /
         READ_BYTES_PER_CYCLE + 32'd1);
    assign l1_req_wdata = l1_req_write ? store_wdata_value : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write ? store_wstrb_value : {DATA_WIDTH/8{1'b0}};
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
                fp_av = fp16_to_real(active_a_vec[fp_lane*16 +: 16]);
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
`ifdef MDLA7_DPI_DATAPATH
        if (dpi_datapath_enabled && fp_mode) begin
            mdla7_dpi_ewe_fp16(
                active_a_vec[31:0],
                active_a_vec[63:32],
                active_a_vec[95:64],
                active_a_vec[127:96],
                b_vec[31:0],
                b_vec[63:32],
                b_vec[95:64],
                b_vec[127:96],
                {24'd0, safe_fp_count},
                {30'd0, op_mode},
                dpi_fp_ewe_bits
            );
            fp_ewe_bits = dpi_fp_ewe_bits;
        end
`endif
        for (i16_lane = 0; i16_lane < (MAX_ELEMS/2); i16_lane = i16_lane + 1) begin
            if (i16_lane < safe_fp_count) begin
                i16_av = {{16{active_a_vec[i16_lane*16 + 15]}}, active_a_vec[i16_lane*16 +: 16]};
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
        store_byte_count_value = fp_mode ? 32'd8 : (int16_mode ? 32'd2 : 32'd1);
        if (fp_mode)
            store_wdata_value = vector_lane_wdata(fp_ewe_bits, out_byte_offset[3:0]);
        else if (int16_mode)
            store_wdata_value = vector_lane_wdata({112'd0, first_lane_value[15:0]}, out_byte_offset[3:0]);
        else
            store_wdata_value = byte_lane_wdata(out_q[7:0], out_byte_offset[3:0]);
        store_wstrb_value = vector_lane_wstrb(store_byte_count_value, out_byte_offset[3:0]);

        case (state)
            ST_A: begin phase_id = PH_A_READ; remaining_cycles = l1_req_payload_cycles; end
            ST_B: begin phase_id = PH_B_READ; remaining_cycles = l1_req_payload_cycles; end
            ST_PIPE: begin phase_id = PH_LANE_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_SRAMCRC: begin phase_id = PH_LANE_PIPE; remaining_cycles = sramcrc_remaining; end
            ST_LUT_LOAD: begin phase_id = PH_A_READ; remaining_cycles = {27'd0, 5'd16} - {27'd0, lut_load_beat}; end
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
            a_req_sent <= 1'b0;
            active_a_vec <= {MAX_ELEMS*8{1'b0}};
            lut_load_beat <= 5'd0;
            lut_beat_req_sent <= 1'b0;
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    a_req_sent <= 1'b0;
                    lut_load_beat <= 5'd0;
                    lut_beat_req_sent <= 1'b0;
                    if (start_valid && start_ready) begin
                        if (sramcrc_mode) begin
                            sramcrc_crc <= FNV_OFFSET;
                            sramcrc_count <= 32'd0;
                            sramcrc_index <= out_byte_offset;
                            sramcrc_remaining <= sramcrc_expected_count;
                            state <= (sramcrc_expected_count == 32'd0) ? ST_DONE : ST_SRAMCRC;
                        end else begin
                            active_a_vec <= a_vec;
                            // v11: lut_mode pre-loads the 256-byte table from
                            // L1 before reading the input; non-lut modes go
                            // straight to ST_A.
                            state <= lut_mode ? ST_LUT_LOAD : ST_A;
                        end
                    end
                end
                ST_LUT_LOAD: begin
                    if (!lut_beat_req_sent && l1_req_ready) begin
                        lut_beat_req_sent <= 1'b1;
`ifdef MDLA7_DEBUG_EWE_LUT
                        $display("[EWE_LUT] beat=%0d issue addr=0x%h", lut_load_beat, l1_req_addr);
`endif
                    end
                    if (lut_beat_req_sent && l1_resp_valid) begin
                        for (lut_byte = 0; lut_byte < 16; lut_byte = lut_byte + 1)
                            lut_mem[{lut_load_beat[3:0], 4'd0} + lut_byte[3:0]] <=
                                l1_resp_rdata[lut_byte*8 +: 8];
                        lut_beat_req_sent <= 1'b0;
`ifdef MDLA7_DEBUG_EWE_LUT
                        $display("[EWE_LUT] beat=%0d recv data=0x%h", lut_load_beat, l1_resp_rdata);
`endif
                        if (lut_load_beat == 5'd15) begin
                            lut_load_beat <= 5'd0;
                            state <= ST_A;
                        end else begin
                            lut_load_beat <= lut_load_beat + 5'd1;
                        end
                    end
                end
                ST_A: begin
                    if (read_a_from_l1) begin
                        if (!a_req_sent && l1_req_ready)
                            a_req_sent <= 1'b1;
                        if (l1_resp_valid) begin
                            active_a_vec <= {MAX_ELEMS*8{1'b0}};
                            if (fp_mode || int16_mode)
                                active_a_vec <= compact_l1_response(
                                    l1_resp_rdata, l1_req_base_addr[3:0], l1_req_bytes
                                );
                            else
                                active_a_vec[7:0] <= l1_resp_byte;
                            // lut_mode is unary -> skip B and go straight to PIPE.
                            if (lut_mode) begin
                                pipe_remaining <= {24'd0, safe_count} + 32'd1;
                                state <= ST_PIPE;
                            end else begin
                                state <= ST_B;
                            end
                        end
                    end else if (l1_req_ready) begin
                        if (lut_mode) begin
                            pipe_remaining <= {24'd0, safe_count} + 32'd1;
                            state <= ST_PIPE;
                        end else begin
                            state <= ST_B;
                        end
                    end
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
                    if (l1_req_ready) begin
                        for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                            if ((sramcrc_i < store_byte_count_value) &&
                                ((out_byte_offset + sramcrc_i[31:0]) < MAX_EWE_OUTPUT_SRAM_BYTES))
                                output_sram[out_byte_offset + sramcrc_i[31:0]] <=
                                    store_wdata_value[(out_byte_offset[3:0] + sramcrc_i[3:0]) * 8 +: 8];
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
                                ((sramcrc_index + sramcrc_i[31:0]) < MAX_EWE_OUTPUT_SRAM_BYTES)) begin
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
