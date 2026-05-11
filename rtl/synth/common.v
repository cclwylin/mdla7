`timescale 1ns/1ps

`ifndef MDLA7_SYNTH_COMMON_V
`define MDLA7_SYNTH_COMMON_V

// Shared phase sequencer for MDLA7 synth-mode latency RTL.
// Cycles are packed LSB-first: phase_cycles[31:0] is phase 0.
/* verilator lint_off DECLFILENAME */
module mdla7_synth_phase_engine #(
    parameter NUM_PHASES = 8,
    parameter PHASE_W = 4
) (
    input                       clk,
    input                       rst_n,

    input                       start_valid,
    output                      start_ready,
    input      [NUM_PHASES*32-1:0] phase_cycles,
    input      [NUM_PHASES*PHASE_W-1:0] phase_ids,
    input                       phase_stall,

    output reg                  busy,
    output reg                  done_valid,
    input                       done_ready,
    output reg [PHASE_W-1:0]    phase_id,
    output reg [31:0]           remaining_cycles
);
    localparam [7:0] IDX_NONE = 8'hff;

    reg [7:0] phase_index;
    reg [7:0] next_idx;

    assign start_ready = !busy && !done_valid;

    function [31:0] phase_cycle_at;
        input [7:0] idx;
        begin
            phase_cycle_at = phase_cycles[idx*32 +: 32];
        end
    endfunction

    function [PHASE_W-1:0] phase_id_at;
        input [7:0] idx;
        begin
            phase_id_at = phase_ids[idx*PHASE_W +: PHASE_W];
        end
    endfunction

    function [7:0] find_index;
        input [7:0] start_idx;
        integer i;
        reg found;
        begin
            find_index = IDX_NONE;
            found = 1'b0;
            for (i = 0; i < NUM_PHASES; i = i + 1) begin
                if (!found && (i >= start_idx) && (phase_cycles[i*32 +: 32] != 32'd0)) begin
                    find_index = i[7:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done_valid <= 1'b0;
            phase_id <= {PHASE_W{1'b0}};
            phase_index <= 8'd0;
            remaining_cycles <= 32'd0;
        end else begin
            if (done_valid && done_ready)
                done_valid <= 1'b0;

            if (busy) begin
                if (phase_stall) begin
                    remaining_cycles <= remaining_cycles;
                end else if (remaining_cycles > 32'd1) begin
                    remaining_cycles <= remaining_cycles - 32'd1;
                end else begin
                    next_idx = find_index(phase_index + 8'd1);
                    if (next_idx == IDX_NONE) begin
                        busy <= 1'b0;
                        done_valid <= 1'b1;
                        phase_id <= {PHASE_W{1'b0}};
                        remaining_cycles <= 32'd0;
                    end else begin
                        phase_index <= next_idx;
                        phase_id <= phase_id_at(next_idx);
                        remaining_cycles <= phase_cycle_at(next_idx);
                    end
                end
            end else if (start_valid && start_ready) begin
                next_idx = find_index(8'd0);
                if (next_idx == IDX_NONE) begin
                    done_valid <= 1'b1;
                    phase_id <= {PHASE_W{1'b0}};
                    remaining_cycles <= 32'd0;
                end else begin
                    busy <= 1'b1;
                    phase_index <= next_idx;
                    phase_id <= phase_id_at(next_idx);
                    remaining_cycles <= phase_cycle_at(next_idx);
                end
            end
        end
    end
    /* verilator lint_on BLKSEQ */
endmodule

// Simulation datapath helper.
//
// Under Verilator, a DPI-C compute core parses the MDL7 program image and runs
// the selected layer's actual datapath from input/weight/metadata bytes.  The
// returned CRC is compared by host.v against the fast-model reference CRC.
// OP_MATERIALIZE remains the compiler-defined pre-materialized-output fallback.
module mdla7_true_datapath (
    input             clk,
    input             rst_n,
    input             start_fire,
    input      [31:0] layer_index,
    input      [31:0] ref_off,
    input      [31:0] ref_size,
    output reg [31:0] datapath_crc,
    output reg        datapath_ok
);
`ifdef SYNTHESIS
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            datapath_crc <= 32'd0;
            datapath_ok <= 1'b1;
        end else if (start_fire) begin
            datapath_crc <= 32'd0;
            datapath_ok <= 1'b1;
        end
    end
`else
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;

`ifdef VERILATOR
    import "DPI-C" context task mdla7_dpi_compute_layer_crc(
        input string program_path,
        input int layer_index,
        output int crc,
        output bit ok
    );

    string program_path;
    int dpi_crc;
    bit dpi_ok;

    initial begin
        program_path = "rtl/bin/Hotspot/gpt2_quant_L24_L63.bin";
        if (!$value$plusargs("PROGRAM=%s", program_path))
            program_path = "rtl/bin/Hotspot/gpt2_quant_L24_L63.bin";
    end

    /* verilator lint_off BLKSEQ */
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            datapath_crc <= FNV_OFFSET;
            datapath_ok <= 1'b1;
        end else if (start_fire) begin
            if (ref_size == 32'd0) begin
                datapath_crc <= FNV_OFFSET;
                datapath_ok <= 1'b1;
            end else begin
                dpi_crc = FNV_OFFSET;
                dpi_ok = 1'b0;
                mdla7_dpi_compute_layer_crc(program_path, layer_index, dpi_crc, dpi_ok);
                datapath_crc <= dpi_crc;
                datapath_ok <= dpi_ok;
                if (!dpi_ok) begin
                    $display("DATAPATH_FAIL: layer=%0d true datapath failed ref_off=%0d ref_size=%0d PROGRAM=%0s",
                             layer_index, ref_off, ref_size, program_path);
                end
            end
        end
    end
    /* verilator lint_on BLKSEQ */
`else
    initial begin
        datapath_crc = FNV_OFFSET;
        datapath_ok = 1'b0;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            datapath_crc <= FNV_OFFSET;
            datapath_ok <= 1'b0;
        end else if (start_fire) begin
            $display("DATAPATH_FAIL: true datapath requires Verilator DPI-C");
            datapath_crc <= FNV_OFFSET;
            datapath_ok <= 1'b0;
        end
    end
`endif
`endif
endmodule
/* verilator lint_on DECLFILENAME */

`endif
