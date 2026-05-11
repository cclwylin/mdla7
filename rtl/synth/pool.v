`timescale 1ns/1ps

`include "common.v"

`timescale 1ns/1ps

module pool #(
    parameter READ_PORTS = 2,
    parameter WRITE_PORTS = 2,
    parameter PAYLOAD_BYTES = 32,
    parameter ELEM_BYTES = 1
) (
    input             clk,
    input             rst_n,

    input             start_valid,
    output            start_ready,
    input      [31:0] in_elems,
    input      [31:0] out_elems,
    input      [31:0] lanes,
    input      [31:0] window,
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
    localparam [3:0] PH_CFG_DECODE   = 4'd1;
    localparam [3:0] PH_WINDOW_FETCH = 4'd2;
    localparam [3:0] PH_REDUCE_PIPE  = 4'd3;
    localparam [3:0] PH_PAYLOAD_WRITE = 4'd4;
    localparam [3:0] PH_RETIRE       = 4'd5;

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    wire [31:0] safe_lanes = (lanes == 32'd0) ? 32'd1 : lanes;
    wire [31:0] safe_window = (window == 32'd0) ? 32'd1 : window;
    wire [31:0] bytes_in = in_elems * ELEM_BYTES;
    wire [31:0] bytes_out = out_elems * ELEM_BYTES;
    wire [31:0] compute_cycles = ceil_div(out_elems, safe_lanes) * safe_window;

    wire [31:0] window_fetch_cycles = ceil_div(bytes_in, READ_PORTS * PAYLOAD_BYTES) + 32'd1;
    wire [31:0] reduce_pipe_cycles = compute_cycles + 32'd2;
    wire [31:0] payload_write_cycles = ceil_div(bytes_out, WRITE_PORTS * PAYLOAD_BYTES) + 32'd1;
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

    wire [5*32-1:0] phase_cycles = {
        32'd1,
        payload_write_cycles,
        reduce_pipe_cycles,
        window_fetch_cycles,
        32'd2
    };

    wire [5*4-1:0] phase_ids = {
        PH_RETIRE,
        PH_PAYLOAD_WRITE,
        PH_REDUCE_PIPE,
        PH_WINDOW_FETCH,
        PH_CFG_DECODE
    };

    wire payload_phase_active = busy &&
        ((phase_id == PH_WINDOW_FETCH) || (phase_id == PH_PAYLOAD_WRITE));
    reg payload_token_sent;
    wire payload_token_fire = l1_req_valid && l1_req_ready;
    wire phase_stall = payload_phase_active && !payload_token_sent && !l1_req_ready;

    assign l1_req_valid = payload_phase_active && !payload_token_sent;
    assign l1_req_write = (phase_id == PH_PAYLOAD_WRITE);
    assign l1_req_bytes = (phase_id == PH_PAYLOAD_WRITE) ? bytes_out : bytes_in;
    assign l1_req_payload_cycles = (phase_id == PH_PAYLOAD_WRITE)
        ? payload_write_cycles
        : window_fetch_cycles;

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
