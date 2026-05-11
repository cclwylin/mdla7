`timescale 1ns/1ps

`include "common.v"

`timescale 1ns/1ps

module udma #(
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
    input      [31:0] layer_index,
    input      [31:0] ref_off,
    input      [31:0] ref_size,

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
    output     [31:0] datapath_crc,
    output            datapath_ok
);
    localparam [3:0] PH_CFG_DECODE      = 4'd1;
    localparam [3:0] PH_L1_PAYLOAD_READ = 4'd2;
    localparam [3:0] PH_CODEC_PIPE      = 4'd3;
    localparam [3:0] PH_DRAM_CMD        = 4'd4;
    localparam [3:0] PH_DRAM_WRITE_DATA = 4'd5;
    localparam [3:0] PH_DRAM_READ_DATA  = 4'd6;
    localparam [3:0] PH_L1_PAYLOAD_WRITE = 4'd7;
    localparam [3:0] PH_RETIRE          = 4'd8;

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
    wire start_fire = start_valid && start_ready;

    mdla7_true_datapath u_datapath (
        .clk(clk),
        .rst_n(rst_n),
        .start_fire(start_fire),
        .layer_index(layer_index),
        .ref_off(ref_off),
        .ref_size(ref_size),
        .datapath_crc(datapath_crc),
        .datapath_ok(datapath_ok)
    );

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
    wire phase_stall = payload_phase_active && !payload_token_sent && !l1_req_ready;

    assign l1_req_valid = payload_phase_active && !payload_token_sent;
    assign l1_req_write = !direction_write;
    assign l1_req_bytes = bytes;
    assign l1_req_payload_cycles = l1_payload_cycles;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            payload_token_sent <= 1'b0;
        end else begin
            if (start_fire)
                payload_token_sent <= 1'b0;
            else if (payload_token_fire)
                payload_token_sent <= 1'b1;
            else if (!payload_phase_active)
                payload_token_sent <= 1'b0;
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
