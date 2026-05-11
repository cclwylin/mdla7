`timescale 1ns/1ps

`include "common.v"

`timescale 1ns/1ps

module command (
    input             clk,
    input             rst_n,

    input             start_valid,
    output            start_ready,
    input      [31:0] cfg_write_cycles,
    input      [7:0]  wait_count,
    input      [7:0]  op_class,

    output            busy,
    output            done_valid,
    input             done_ready,
    output     [3:0]  phase_id,
    output     [31:0] remaining_cycles,
    output     [7:0]  debug_wait_count,
    output     [7:0]  debug_op_class
);
    localparam [3:0] PH_DECODE    = 4'd1;
    localparam [3:0] PH_CFG_WRITE = 4'd2;

    wire [31:0] safe_cfg_write_cycles =
        (cfg_write_cycles == 32'd0) ? 32'd1 : cfg_write_cycles;

    assign debug_wait_count = wait_count;
    assign debug_op_class = op_class;

    wire [2*32-1:0] phase_cycles = {
        safe_cfg_write_cycles,
        32'd1
    };

    wire [2*4-1:0] phase_ids = {
        PH_CFG_WRITE,
        PH_DECODE
    };

    mdla7_synth_phase_engine #(
        .NUM_PHASES(2),
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
