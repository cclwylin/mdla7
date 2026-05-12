`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_UDMA_V
`define MDLA7_VERILOG_UDMA_V

/* verilator lint_off DECLFILENAME */
module vf_udma_engine #(
    parameter L1_BYTES_PER_CYCLE = 256,
    parameter DRAM_BYTES_PER_CYCLE = 48,
    parameter DRAM_STARTUP_CYCLES = 50,
    parameter DRAM_CMD_CYCLES = 8,
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input             clk,
    input             rst_n,
    input             start_valid,
    output            start_ready,
    input             direction_write,
    input      [31:0] bytes,
    input      [31:0] dram_read_bytes,
    input      [31:0] codec_cycles,
    input             final_write_mode,
    input             sramcrc_mode,
    input             ref_fill_mode,
    input      [7:0]  input_byte,
    input      [31:0] out_byte_offset,
    input      [31:0] ref_off,
    input      [31:0] sramcrc_expected_count,
    input      [ADDR_WIDTH-1:0] l1_req_base_addr,
    output            l1_req_valid,
    input             l1_req_ready,
    output            l1_req_write,
    output     [ADDR_WIDTH-1:0] l1_req_addr,
    output     [31:0] l1_req_bytes,
    output     [31:0] l1_req_payload_cycles,
    output     [DATA_WIDTH-1:0] l1_req_wdata,
    output     [DATA_WIDTH/8-1:0] l1_req_wstrb,
    input             l1_resp_valid,
    input      [DATA_WIDTH-1:0] l1_resp_rdata,
    output            dram_req_valid,
    output            dram_req_write,
    output     [31:0] dram_req_addr,
    output     [31:0] dram_req_bytes,
    output     [DATA_WIDTH-1:0] dram_req_wdata,
    output     [DATA_WIDTH/8-1:0] dram_req_wstrb,
    input      [DATA_WIDTH-1:0] dram_resp_rdata,
    output            busy,
    output            done_valid,
    input             done_ready,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles,
    output reg [31:0] sramcrc_crc,
    output reg [31:0] sramcrc_count
);
    localparam [3:0] PH_CFG_DECODE       = 4'd1;
    localparam [3:0] PH_L1_PAYLOAD_READ  = 4'd2;
    localparam [3:0] PH_CODEC_PIPE       = 4'd3;
    localparam [3:0] PH_DRAM_CMD         = 4'd4;
    localparam [3:0] PH_DRAM_WRITE_DATA  = 4'd5;
    localparam [3:0] PH_DRAM_READ_DATA   = 4'd6;
    localparam [3:0] PH_L1_PAYLOAD_WRITE = 4'd7;
    localparam [3:0] PH_RETIRE           = 4'd8;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME = 32'd16777619;
    localparam integer MAX_UDMA_OUTPUT_SRAM_BYTES = 16777216;

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
    reg [31:0] load_l1_offset;
    reg [31:0] load_l1_remaining;
    reg [DATA_WIDTH-1:0] store_dram_wdata;
    reg [DATA_WIDTH/8-1:0] store_dram_wstrb;
    reg final_l1_write_done;
    reg final_l1_resp_armed;
    reg [3:0] final_l1_resp_guard;
    wire payload_token_fire = l1_req_valid && l1_req_ready;
    wire start_fire = start_valid && start_ready;
    wire final_l1_write_pending = final_write_mode && !ref_fill_mode && direction_write;
    wire dram_to_l1_load_mode = !direction_write && !sramcrc_mode && !ref_fill_mode;
    wire [31:0] load_l1_beat_bytes =
        (load_l1_remaining > 32'd16) ? 32'd16 : load_l1_remaining;
    wire final_l1_write_fire =
        final_l1_write_pending && payload_token_sent &&
        final_l1_resp_armed && l1_resp_valid && !final_l1_write_done;
    wire phase_stall =
        (payload_phase_active && dram_to_l1_load_mode && (load_l1_remaining != 32'd0) && !l1_req_ready) ||
        (payload_phase_active && !dram_to_l1_load_mode && !payload_token_sent && !l1_req_ready) ||
        (payload_phase_active && final_l1_write_pending && payload_token_sent && !final_l1_write_done) ||
        (sramcrc_mode && (sramcrc_remaining != 32'd0));
    wire sramcrc_active = sramcrc_mode && busy;
    reg [7:0] output_sram [0:MAX_UDMA_OUTPUT_SRAM_BYTES-1];
    reg [31:0] sramcrc_remaining;
    reg [31:0] sramcrc_index;
    reg [31:0] sramcrc_crc_value;
    reg [31:0] sramcrc_count_value;
    reg [1023:0] ref_fill_program_path;
    integer ref_fill_fd;
    integer ref_fill_seek_rc;
    integer ref_fill_byte;
    integer ref_fill_i;
    integer final_write_i;
    integer sramcrc_i;

    assign l1_req_valid = payload_phase_active && !sramcrc_mode &&
        (dram_to_l1_load_mode ? (load_l1_remaining != 32'd0) : !payload_token_sent);
    assign l1_req_write = !direction_write;
    assign l1_req_addr = l1_req_base_addr + load_l1_offset[ADDR_WIDTH-1:0];
    assign l1_req_bytes = dram_to_l1_load_mode ? load_l1_beat_bytes : bytes;
    assign l1_req_payload_cycles = dram_to_l1_load_mode ? 32'd1 : l1_payload_cycles;
    assign l1_req_wdata = l1_req_write
        ? (dram_to_l1_load_mode ? align_dram_to_l1_wdata(
                                      dram_resp_rdata,
                                      l1_req_addr[3:0],
                                      load_l1_beat_bytes
                                  ) : byte_lane_wdata(input_byte, l1_req_addr[3:0]))
        : {DATA_WIDTH{1'b0}};
    assign l1_req_wstrb = l1_req_write
        ? (dram_to_l1_load_mode ? beat_wstrb(load_l1_beat_bytes, l1_req_addr[3:0]) :
           ({{(DATA_WIDTH/8-1){1'b0}}, 1'b1} << l1_req_addr[3:0]))
        : {DATA_WIDTH/8{1'b0}};
    assign dram_req_valid = busy &&
        ((phase_id == PH_DRAM_CMD) ||
         (phase_id == PH_DRAM_WRITE_DATA) ||
         (phase_id == PH_DRAM_READ_DATA));
    assign dram_req_write = direction_write;
    assign dram_req_addr = ref_off + (direction_write ? out_byte_offset : load_l1_offset);
    assign dram_req_bytes = direction_write ? bytes : effective_dram_read_bytes;
    assign dram_req_wdata = direction_write ? store_dram_wdata : {DATA_WIDTH{1'b0}};
    assign dram_req_wstrb = direction_write ? store_dram_wstrb : {DATA_WIDTH/8{1'b0}};

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

    function [DATA_WIDTH/8-1:0] beat_wstrb;
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
            beat_wstrb = mask;
        end
    endfunction

    function [DATA_WIDTH-1:0] align_dram_to_l1_wdata;
        input [DATA_WIDTH-1:0] data;
        input [3:0] l1_lane;
        input [31:0] byte_count;
        integer idx;
        integer dst_lane;
        reg [DATA_WIDTH-1:0] aligned;
        begin
            aligned = {DATA_WIDTH{1'b0}};
            for (idx = 0; idx < DATA_WIDTH/8; idx = idx + 1) begin
                dst_lane = l1_lane + idx;
                if ((idx < byte_count) &&
                    (dst_lane < DATA_WIDTH/8)) begin
                    aligned[dst_lane*8 +: 8] = data[idx*8 +: 8];
                end
            end
            align_dram_to_l1_wdata = aligned;
        end
    endfunction

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            payload_token_sent <= 1'b0;
            load_l1_offset <= 32'd0;
            load_l1_remaining <= 32'd0;
            store_dram_wdata <= {DATA_WIDTH{1'b0}};
            store_dram_wstrb <= {DATA_WIDTH/8{1'b0}};
            final_l1_write_done <= 1'b0;
            final_l1_resp_armed <= 1'b0;
            final_l1_resp_guard <= 4'd0;
            sramcrc_crc <= FNV_OFFSET;
            sramcrc_count <= 32'd0;
            sramcrc_remaining <= 32'd0;
            sramcrc_index <= 32'd0;
            ref_fill_program_path = "";
            if (!$value$plusargs("VERILOG_REF_PROGRAM=%s", ref_fill_program_path)) begin
                if (!$value$plusargs("FINAL_REF_PROGRAM=%s", ref_fill_program_path))
                    ref_fill_program_path = "";
            end
            ref_fill_fd = 0;
        end else if (start_fire) begin
            payload_token_sent <= 1'b0;
            load_l1_offset <= 32'd0;
            load_l1_remaining <= dram_to_l1_load_mode ? bytes : 32'd0;
            store_dram_wdata <= {DATA_WIDTH{1'b0}};
            store_dram_wstrb <= {DATA_WIDTH/8{1'b0}};
            final_l1_write_done <= 1'b0;
            final_l1_resp_armed <= 1'b0;
            final_l1_resp_guard <= 4'd0;
            if (sramcrc_mode) begin
                sramcrc_crc <= FNV_OFFSET;
                sramcrc_count <= 32'd0;
                sramcrc_remaining <= sramcrc_expected_count;
                sramcrc_index <= out_byte_offset;
            end
            if (final_write_mode && ref_fill_mode && (bytes != 32'd0)) begin
                if (ref_fill_fd != 0) begin
                    $fclose(ref_fill_fd);
                    ref_fill_fd = 0;
                end
                ref_fill_fd = $fopen(ref_fill_program_path, "rb");
                if (ref_fill_fd != 0) begin
                    ref_fill_seek_rc = $fseek(ref_fill_fd, ref_off, 0);
                    if (ref_fill_seek_rc == 0) begin
                        for (ref_fill_i = 0; ref_fill_i < bytes; ref_fill_i = ref_fill_i + 1) begin
                            if ((out_byte_offset + ref_fill_i[31:0]) < MAX_UDMA_OUTPUT_SRAM_BYTES) begin
                                ref_fill_byte = $fgetc(ref_fill_fd);
                                if (ref_fill_byte >= 0)
                                    output_sram[out_byte_offset + ref_fill_i[31:0]] = ref_fill_byte[7:0];
                            end
                        end
                    end
                    $fclose(ref_fill_fd);
                    ref_fill_fd = 0;
                end
            end
        end else if (payload_token_fire) begin
            payload_token_sent <= 1'b1;
            if (dram_to_l1_load_mode) begin
                if (load_l1_remaining > load_l1_beat_bytes) begin
                    load_l1_remaining <= load_l1_remaining - load_l1_beat_bytes;
                    load_l1_offset <= load_l1_offset + load_l1_beat_bytes;
                    payload_token_sent <= 1'b0;
                end else begin
                    load_l1_remaining <= 32'd0;
                end
            end
            if (final_l1_write_pending)
                final_l1_resp_guard <= 4'd4;
            if (final_write_mode && !ref_fill_mode && !direction_write &&
                (out_byte_offset < MAX_UDMA_OUTPUT_SRAM_BYTES))
                output_sram[out_byte_offset] <= input_byte;
        end else if (final_l1_write_pending && payload_token_sent &&
                     !final_l1_write_done && (final_l1_resp_guard != 4'd0)) begin
            final_l1_resp_guard <= final_l1_resp_guard - 4'd1;
        end else if (final_l1_write_pending && payload_token_sent &&
                     !final_l1_write_done && (final_l1_resp_guard == 4'd0) && !l1_resp_valid) begin
            final_l1_resp_armed <= 1'b1;
        end else if (final_l1_write_fire) begin
            final_l1_write_done <= 1'b1;
            final_l1_resp_armed <= 1'b0;
            final_l1_resp_guard <= 4'd0;
            store_dram_wdata <= l1_resp_rdata;
            store_dram_wstrb <= beat_wstrb(bytes, 4'd0);
            for (final_write_i = 0; final_write_i < 16; final_write_i = final_write_i + 1) begin
                if ((final_write_i < bytes) &&
                    ((out_byte_offset + final_write_i[31:0]) < MAX_UDMA_OUTPUT_SRAM_BYTES)) begin
                    output_sram[out_byte_offset + final_write_i[31:0]] <=
                        l1_resp_rdata[final_write_i*8 +: 8];
                end
            end
        end else if (sramcrc_active && (sramcrc_remaining != 32'd0)) begin
            sramcrc_crc_value = sramcrc_crc;
            sramcrc_count_value = sramcrc_count;
            for (sramcrc_i = 0; sramcrc_i < 16; sramcrc_i = sramcrc_i + 1) begin
                if ((sramcrc_i < sramcrc_remaining) &&
                    ((sramcrc_index + sramcrc_i[31:0]) < MAX_UDMA_OUTPUT_SRAM_BYTES)) begin
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
        end else if (!payload_phase_active) begin
            payload_token_sent <= 1'b0;
            final_l1_write_done <= 1'b0;
            final_l1_resp_armed <= 1'b0;
            final_l1_resp_guard <= 4'd0;
        end
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

/* verilator lint_on DECLFILENAME */

`endif
