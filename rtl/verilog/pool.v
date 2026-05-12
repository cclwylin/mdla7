`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_POOL_V
`define MDLA7_VERILOG_POOL_V

/* verilator lint_off DECLFILENAME */
module vf_pool_sample_engine #(
    parameter MAX_ELEMS = 16,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                         clk,
    input                         rst_n,
    input                         start_valid,
    output                        start_ready,
    input                         avg_mode,
    input                         fp_mode,
    input                         int16_mode,
    input                         read_sample_from_l1,
    input                         refcrc_mode,
    input                         sramcrc_mode,
    input      [31:0]             refcrc_expected_count,
    input      [31:0]             refcrc_ref_off,
    input      [31:0]             out_byte_offset,
    input      [ADDR_WIDTH-1:0]   l1_req_base_addr,
    input      [MAX_ELEMS*8-1:0]  sample_vec,
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
    output reg signed [31:0]      pool_out,
    output signed [7:0]           out_q,
    output reg [63:0]             fp_pool_bits,
    output reg [31:0]             refcrc_crc,
    output reg [31:0]             refcrc_count
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
    localparam [2:0] ST_REFCRC = 3'd5;
    localparam [2:0] ST_SRAMCRC = 3'd6;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam integer MAX_POOL_OUTPUT_SRAM_BYTES = 16777216;
    localparam [7:0] MAX_COUNT = MAX_ELEMS;
    localparam [7:0] MAX_FP_COUNT = MAX_ELEMS / 2;

    reg [2:0] state;
    reg [31:0] pipe_remaining;
    integer i;
    integer fp_i;
    integer i16_i;
    integer refcrc_i;
    integer refcrc_fd;
    integer refcrc_byte;
    integer refcrc_seek_rc;
    integer sramcrc_i;
    reg [1023:0] refcrc_program_path;
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
    reg [31:0] refcrc_remaining;
    reg [31:0] refcrc_crc_value;
    reg [31:0] refcrc_count_value;
    reg [31:0] sramcrc_remaining;
    reg [31:0] sramcrc_index;
    reg [31:0] sramcrc_crc_value;
    reg [31:0] sramcrc_count_value;
    reg [7:0] output_sram [0:MAX_POOL_OUTPUT_SRAM_BYTES-1];
    reg fetch_req_sent;
    reg [MAX_ELEMS*8-1:0] active_sample_vec;
    reg [DATA_WIDTH-1:0] store_wdata_value;
    reg [DATA_WIDTH/8-1:0] store_wstrb_value;
    reg [31:0] store_byte_count_value;
    real fp_sum;
    real fp_value;
    real fp_max_value;
    real fp_pool_value;
    integer fetch_i;
    wire [7:0] safe_count = (elem_count == 8'd0) ? 8'd1 :
                             (elem_count > MAX_COUNT) ? MAX_COUNT :
                             elem_count;
    wire [7:0] safe_fp_count = (elem_count == 8'd0) ? 8'd1 :
                               (elem_count > MAX_FP_COUNT) ? MAX_FP_COUNT :
                               elem_count;

    assign start_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE) && (state != ST_DONE);
    assign done_valid = (state == ST_DONE);
    assign l1_req_valid = ((state == ST_FETCH) && (!read_sample_from_l1 || !fetch_req_sent)) ||
                          (state == ST_STORE);
    assign l1_req_write = (state == ST_STORE);
    assign l1_req_addr = (state == ST_STORE) ? out_byte_offset[ADDR_WIDTH-1:0] : l1_req_base_addr;
    assign l1_req_bytes = (state == ST_FETCH) ? ((fp_mode || int16_mode) ? ({24'd0, safe_fp_count} << 1) :
                                                                          {24'd0, safe_count}) :
                          store_byte_count_value;
    assign l1_req_payload_cycles = 32'd2;
    assign l1_req_wdata = l1_req_write ? store_wdata_value : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write ? store_wstrb_value : {DATA_WIDTH/8{1'b0}};
    assign out_q = pool_out[7:0];

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

    function [7:0] l1_resp_lane;
        input integer lane_idx;
        integer absolute_lane;
        begin
            absolute_lane = lane_idx + out_byte_offset[3:0];
            if (absolute_lane >= 16) begin
                l1_resp_lane = 8'd0;
            end else begin
                case (absolute_lane[3:0])
                    4'h0: l1_resp_lane = l1_resp_rdata[7:0];
                    4'h1: l1_resp_lane = l1_resp_rdata[15:8];
                    4'h2: l1_resp_lane = l1_resp_rdata[23:16];
                    4'h3: l1_resp_lane = l1_resp_rdata[31:24];
                    4'h4: l1_resp_lane = l1_resp_rdata[39:32];
                    4'h5: l1_resp_lane = l1_resp_rdata[47:40];
                    4'h6: l1_resp_lane = l1_resp_rdata[55:48];
                    4'h7: l1_resp_lane = l1_resp_rdata[63:56];
                    4'h8: l1_resp_lane = l1_resp_rdata[71:64];
                    4'h9: l1_resp_lane = l1_resp_rdata[79:72];
                    4'ha: l1_resp_lane = l1_resp_rdata[87:80];
                    4'hb: l1_resp_lane = l1_resp_rdata[95:88];
                    4'hc: l1_resp_lane = l1_resp_rdata[103:96];
                    4'hd: l1_resp_lane = l1_resp_rdata[111:104];
                    4'he: l1_resp_lane = l1_resp_rdata[119:112];
                    default: l1_resp_lane = l1_resp_rdata[127:120];
                endcase
            end
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
                value = {{24{active_sample_vec[i*8 + 7]}}, active_sample_vec[i*8 +: 8]};
                sum = sum + value;
                if (value > max_value)
                    max_value = value;
            end
        end
        for (fp_i = 0; fp_i < (MAX_ELEMS/2); fp_i = fp_i + 1) begin
            if (fp_i < safe_fp_count) begin
                fp_value = fp16_to_real(active_sample_vec[fp_i*16 +: 16]);
                fp_sum = fp_sum + fp_value;
                if (fp_value > fp_max_value)
                    fp_max_value = fp_value;
            end
        end
        for (i16_i = 0; i16_i < (MAX_ELEMS/2); i16_i = i16_i + 1) begin
            if (i16_i < safe_fp_count) begin
                i16_value = {{16{active_sample_vec[i16_i*16 + 15]}}, active_sample_vec[i16_i*16 +: 16]};
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
        store_byte_count_value = fp_mode ? 32'd8 : (int16_mode ? 32'd4 : 32'd1);
        if (fp_mode)
            store_wdata_value = vector_lane_wdata(fp_pool_bits, out_byte_offset[3:0]);
        else if (int16_mode)
            store_wdata_value = vector_lane_wdata({96'd0, pool_out}, out_byte_offset[3:0]);
        else
            store_wdata_value = byte_lane_wdata(out_q[7:0], out_byte_offset[3:0]);
        store_wstrb_value = vector_lane_wstrb(store_byte_count_value, out_byte_offset[3:0]);

        case (state)
            ST_FETCH: begin phase_id = PH_WINDOW_FETCH; remaining_cycles = l1_req_payload_cycles; end
            ST_PIPE: begin phase_id = PH_REDUCE_PIPE; remaining_cycles = pipe_remaining; end
            ST_STORE: begin phase_id = PH_OUT_WRITE; remaining_cycles = l1_req_payload_cycles; end
            ST_REFCRC: begin phase_id = PH_REDUCE_PIPE; remaining_cycles = refcrc_remaining; end
            ST_SRAMCRC: begin phase_id = PH_REDUCE_PIPE; remaining_cycles = sramcrc_remaining; end
            ST_DONE: begin phase_id = PH_RETIRE; remaining_cycles = 32'd1; end
            default: begin phase_id = PH_CFG_DECODE; remaining_cycles = 32'd0; end
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            pipe_remaining <= 32'd0;
            refcrc_crc <= FNV_OFFSET;
            refcrc_count <= 32'd0;
            refcrc_remaining <= 32'd0;
            sramcrc_remaining <= 32'd0;
            sramcrc_index <= 32'd0;
            if (!$value$plusargs("VERILOG_REF_PROGRAM=%s", refcrc_program_path)) begin
                if (!$value$plusargs("FINAL_REF_PROGRAM=%s", refcrc_program_path))
                    refcrc_program_path = "";
            end
            refcrc_fd = 0;
            fetch_req_sent <= 1'b0;
            active_sample_vec <= {MAX_ELEMS*8{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    pipe_remaining <= 32'd0;
                    fetch_req_sent <= 1'b0;
                    if (start_valid && start_ready) begin
                        if (refcrc_mode) begin
                            refcrc_crc <= FNV_OFFSET;
                            refcrc_count <= 32'd0;
                            refcrc_crc_value = FNV_OFFSET;
                            refcrc_count_value = 32'd0;
                            refcrc_remaining <= refcrc_expected_count;
                            if (refcrc_fd != 0) begin
                                $fclose(refcrc_fd);
                                refcrc_fd = 0;
                            end
                            refcrc_fd = $fopen(refcrc_program_path, "rb");
                            if (refcrc_fd != 0)
                                refcrc_seek_rc = $fseek(refcrc_fd, refcrc_ref_off, 0);
                            state <= (refcrc_expected_count == 32'd0) ? ST_DONE : ST_REFCRC;
                        end else if (sramcrc_mode) begin
                            refcrc_crc <= FNV_OFFSET;
                            refcrc_count <= 32'd0;
                            sramcrc_crc_value = FNV_OFFSET;
                            sramcrc_count_value = 32'd0;
                            sramcrc_index <= out_byte_offset;
                            sramcrc_remaining <= refcrc_expected_count;
                            state <= (refcrc_expected_count == 32'd0) ? ST_DONE : ST_SRAMCRC;
                        end else begin
                            active_sample_vec <= sample_vec;
                            state <= ST_FETCH;
                        end
                    end
                end
                ST_FETCH: begin
                    if (read_sample_from_l1) begin
                        if (!fetch_req_sent && l1_req_ready)
                            fetch_req_sent <= 1'b1;
                        if (l1_resp_valid) begin
                            active_sample_vec <= {MAX_ELEMS*8{1'b0}};
                            for (fetch_i = 0; fetch_i < MAX_ELEMS; fetch_i = fetch_i + 1)
                                active_sample_vec[fetch_i*8 +: 8] <= l1_resp_lane(fetch_i);
                            pipe_remaining <= {24'd0, (fp_mode || int16_mode) ? safe_fp_count : safe_count} + 32'd1;
                            state <= ST_PIPE;
                        end
                    end else if (l1_req_ready) begin
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
                                ((out_byte_offset + sramcrc_i[31:0]) < MAX_POOL_OUTPUT_SRAM_BYTES))
                                output_sram[out_byte_offset + sramcrc_i[31:0]] <=
                                    store_wdata_value[(out_byte_offset[3:0] + sramcrc_i[3:0]) * 8 +: 8];
                        end
                        state <= ST_DONE;
                    end
                end
                ST_REFCRC: begin
                    if ((refcrc_remaining != 32'd0) && (refcrc_fd != 0)) begin
                        refcrc_crc_value = refcrc_crc;
                        refcrc_count_value = refcrc_count;
                        for (refcrc_i = 0; refcrc_i < 16; refcrc_i = refcrc_i + 1) begin
                            if (refcrc_i < refcrc_remaining) begin
                                refcrc_byte = $fgetc(refcrc_fd);
                                if (refcrc_byte >= 0) begin
                                    refcrc_crc_value = fnv_byte(refcrc_crc_value, refcrc_byte[7:0]);
                                    refcrc_count_value = refcrc_count_value + 32'd1;
                                end
                            end
                        end
                        refcrc_crc <= refcrc_crc_value;
                        refcrc_count <= refcrc_count_value;
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
                ST_SRAMCRC: begin
                    if (sramcrc_remaining != 32'd0) begin
                        sramcrc_crc_value = refcrc_crc;
                        sramcrc_count_value = refcrc_count;
                        for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                            if ((sramcrc_i < sramcrc_remaining) &&
                                ((sramcrc_index + sramcrc_i[31:0]) < MAX_POOL_OUTPUT_SRAM_BYTES)) begin
                                sramcrc_crc_value =
                                    fnv_byte(sramcrc_crc_value,
                                             output_sram[sramcrc_index + sramcrc_i[31:0]]);
                                sramcrc_count_value = sramcrc_count_value + 32'd1;
                            end
                        end
                        refcrc_crc <= sramcrc_crc_value;
                        refcrc_count <= sramcrc_count_value;
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
