`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_TNPS_V
`define MDLA7_VERILOG_TNPS_V

/* verilator lint_off DECLFILENAME */
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

module vf_tnps_engine #(
    parameter READ_PORTS = 8,
    parameter WRITE_PORTS = 8,
    parameter PAYLOAD_BYTES = 16,
    parameter PERMUTE_BYTES_PER_CYCLE = 128,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
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
    input             final_write_mode,
    input             sramcrc_mode,
    input      [7:0]  input_byte,
    input      [127:0] input_vec,
    input      [31:0] out_byte_offset,
    input      [31:0] sramcrc_expected_count,
    input      [ADDR_WIDTH-1:0] l1_req_base_addr,
    input             l1_resp_valid,
    input      [DATA_WIDTH-1:0] l1_resp_rdata,
    output            l1_req_valid,
    input             l1_req_ready,
    output            l1_req_write,
    output     [ADDR_WIDTH-1:0] l1_req_addr,
    output     [31:0] l1_req_bytes,
    output     [31:0] l1_req_payload_cycles,
    output     [DATA_WIDTH-1:0] l1_req_wdata,
    output     [DATA_WIDTH/8-1:0] l1_req_wstrb,
    output            busy,
    output            done_valid,
    input             done_ready,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles,
    output     [31:0] sample_src_byte_offset,
    output     [31:0] sample_dst_byte_offset,
    output            sample_valid,
    output reg [31:0] sramcrc_crc,
    output reg [31:0] sramcrc_count
);
    localparam [3:0] PH_CFG_DECODE    = 4'd1;
    localparam [3:0] PH_PAYLOAD_READ  = 4'd2;
    localparam [3:0] PH_PERMUTE_PIPE  = 4'd3;
    localparam [3:0] PH_PAYLOAD_WRITE = 4'd4;
    localparam [3:0] PH_RETIRE        = 4'd5;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam integer MAX_TNPS_OUTPUT_SRAM_BYTES = 16777216;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
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
        input [127:0] value;
        input [3:0] lane;
        begin
            vector_lane_wdata = value << ({lane, 3'd0});
        end
    endfunction

    function [DATA_WIDTH/8-1:0] vector_lane_wstrb;
        input [31:0] byte_count;
        input [3:0] lane;
        reg [DATA_WIDTH/8-1:0] mask;
        integer idx;
        integer absolute_lane;
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
    reg [31:0] sramcrc_remaining;
    reg [31:0] sramcrc_index;
    reg [31:0] sramcrc_crc_value;
    reg [31:0] sramcrc_count_value;
    reg [7:0] output_sram [0:MAX_TNPS_OUTPUT_SRAM_BYTES-1];
    reg [DATA_WIDTH-1:0] permute_vec;
    reg payload_read_req_sent;
    reg payload_read_resp_seen;
    integer sramcrc_i;
    wire payload_token_fire = l1_req_valid && l1_req_ready;
    wire phase_stall =
        (payload_phase_active && (phase_id == PH_PAYLOAD_READ) &&
         (!payload_read_req_sent || !payload_read_resp_seen)) ||
        (payload_phase_active && (phase_id == PH_PAYLOAD_WRITE) &&
         !payload_token_sent && !l1_req_ready);

    wire sramcrc_active = sramcrc_mode && busy && (phase_id == PH_PERMUTE_PIPE);
    wire final_write_active = final_write_mode && busy && (phase_id == PH_PAYLOAD_WRITE);
    wire [DATA_WIDTH-1:0] final_write_vec = final_write_mode ? permute_vec : input_vec;

    assign l1_req_valid = payload_phase_active &&
        ((phase_id == PH_PAYLOAD_READ) ? !payload_read_req_sent : !payload_token_sent);
    assign l1_req_write = (phase_id == PH_PAYLOAD_WRITE);
    assign l1_req_addr = (phase_id == PH_PAYLOAD_READ)
        ? (l1_req_base_addr + sample_src_byte_offset[ADDR_WIDTH-1:0])
        : out_byte_offset[ADDR_WIDTH-1:0];
    assign l1_req_bytes = bytes;
    assign l1_req_payload_cycles = (phase_id == PH_PAYLOAD_WRITE)
        ? payload_write_cycles
        : payload_read_cycles;
    assign l1_req_wdata = l1_req_write
        ? vector_lane_wdata(final_write_vec, l1_req_addr[3:0])
        : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write
        ? vector_lane_wstrb(bytes, l1_req_addr[3:0])
        : {DATA_WIDTH/8{1'b0}};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            payload_token_sent <= 1'b0;
            payload_read_req_sent <= 1'b0;
            payload_read_resp_seen <= 1'b0;
        end else if (start_fire) begin
            payload_token_sent <= 1'b0;
            payload_read_req_sent <= 1'b0;
            payload_read_resp_seen <= 1'b0;
        end else if (payload_token_fire) begin
            payload_token_sent <= 1'b1;
            if (phase_id == PH_PAYLOAD_READ)
                payload_read_req_sent <= 1'b1;
        end else if ((phase_id == PH_PAYLOAD_READ) && l1_resp_valid) begin
            payload_read_resp_seen <= 1'b1;
        end else if (!payload_phase_active) begin
            payload_token_sent <= 1'b0;
            if (phase_id != PH_PAYLOAD_READ) begin
                payload_read_req_sent <= 1'b0;
                payload_read_resp_seen <= 1'b0;
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sramcrc_crc <= FNV_OFFSET;
            sramcrc_count <= 32'd0;
            sramcrc_remaining <= 32'd0;
            sramcrc_index <= 32'd0;
            permute_vec <= {DATA_WIDTH{1'b0}};
        end else if (start_fire) begin
            permute_vec <= input_vec;
            if (sramcrc_mode) begin
                sramcrc_crc <= FNV_OFFSET;
                sramcrc_count <= 32'd0;
                sramcrc_remaining <= sramcrc_expected_count;
                sramcrc_index <= out_byte_offset;
            end
        end else if ((phase_id == PH_PAYLOAD_READ) && l1_resp_valid) begin
            permute_vec <= compact_l1_response(l1_resp_rdata, l1_req_addr[3:0], bytes);
        end else if (final_write_active && payload_token_fire) begin
            for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                if ((sramcrc_i < bytes) &&
                    ((out_byte_offset + sramcrc_i[31:0]) < MAX_TNPS_OUTPUT_SRAM_BYTES))
                    output_sram[out_byte_offset + sramcrc_i[31:0]] <=
                        final_write_vec[sramcrc_i*8 +: 8];
            end
        end else if (sramcrc_active && (sramcrc_remaining != 32'd0)) begin
            sramcrc_crc_value = sramcrc_crc;
            sramcrc_count_value = sramcrc_count;
            for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                if ((sramcrc_i < sramcrc_remaining) &&
                    ((sramcrc_index + sramcrc_i[31:0]) < MAX_TNPS_OUTPUT_SRAM_BYTES)) begin
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
            end else begin
                sramcrc_remaining <= sramcrc_remaining - 32'd16;
                sramcrc_index <= sramcrc_index + 32'd16;
            end
        end
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

/* verilator lint_on DECLFILENAME */

`endif
