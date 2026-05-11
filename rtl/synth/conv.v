`timescale 1ns/1ps

`include "common.v"

`timescale 1ns/1ps

module conv #(
    parameter ACT_PORTS = 4,
    parameter WGT_PORTS = 4,
    parameter PAYLOAD_BYTES = 32,
    parameter CHAIN_LANES = 128
) (
    input             clk,
    input             rst_n,

    input             start_valid,
    output            start_ready,
    input      [31:0] act_bytes,
    input      [31:0] wgt_bytes,
    input      [31:0] out_elems,
    input      [31:0] mac_cycles,
    input      [31:0] fill_cycles,
    input      [31:0] layer_index,
    input      [31:0] ref_off,
    input      [31:0] ref_size,

    output            busy,
    output            done_valid,
    input             done_ready,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles,
    output     [31:0] datapath_crc,
    output            datapath_ok
);
    localparam [3:0] PH_CFG_DECODE  = 4'd1;
    localparam [3:0] PH_ACT_STREAM  = 4'd2;
    localparam [3:0] PH_WGT_STREAM  = 4'd3;
    localparam [3:0] PH_CLUSTER_FILL = 4'd4;
    localparam [3:0] PH_MAC_PIPE    = 4'd5;
    localparam [3:0] PH_PSUM_CHAIN  = 4'd6;
    localparam [3:0] PH_RETIRE      = 4'd7;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    wire [31:0] act_stream_cycles = ceil_div(act_bytes, ACT_PORTS * PAYLOAD_BYTES) + 32'd1;
    wire [31:0] wgt_stream_cycles = ceil_div(wgt_bytes, WGT_PORTS * PAYLOAD_BYTES) + 32'd1;
    wire [31:0] psum_chain_cycles = ceil_div(out_elems, CHAIN_LANES) + 32'd1;
    wire [31:0] cluster_fill_cycles = (fill_cycles == 32'd0) ? 32'd64 : fill_cycles;
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

    wire [7*32-1:0] phase_cycles = {
        32'd1,
        psum_chain_cycles,
        mac_cycles,
        cluster_fill_cycles,
        wgt_stream_cycles,
        act_stream_cycles,
        32'd2
    };

    wire [7*4-1:0] phase_ids = {
        PH_RETIRE,
        PH_PSUM_CHAIN,
        PH_MAC_PIPE,
        PH_CLUSTER_FILL,
        PH_WGT_STREAM,
        PH_ACT_STREAM,
        PH_CFG_DECODE
    };

    mdla7_synth_phase_engine #(
        .NUM_PHASES(7),
        .PHASE_W(4)
    ) u_phase (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(start_valid),
        .start_ready(start_ready),
        .phase_cycles(phase_cycles),
        .phase_ids(phase_ids),
        .phase_stall(1'b0),
        .busy(busy),
        .done_valid(done_valid),
        .done_ready(done_ready),
        .phase_id(phase_id),
        .remaining_cycles(remaining_cycles)
    );
endmodule
