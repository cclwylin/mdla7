`timescale 1ns/1ps

`ifndef MDLA7_VERILOG_ROUTE_V
`define MDLA7_VERILOG_ROUTE_V

/* verilator lint_off DECLFILENAME */
module vf_l1mesh_route_estimator #(
    parameter BASE_CYCLES = 1,
    parameter GLOBAL_HOP_CYCLES = 1,
    parameter LOCAL_HOP_CYCLES = 1
) (
    input      [3:0]  source_id,
    input      [21:0] addr,
    output reg [31:0] route_cycles,
    output reg [1:0]  source_x,
    output reg [1:0]  source_y,
    output     [1:0]  tile_x,
    output     [1:0]  tile_y,
    output     [1:0]  bank_x,
    output     [1:0]  bank_y
);
    /* verilator lint_off UNUSEDSIGNAL */
    wire [15:0] addr_unused = {addr[21:10], addr[3:0]};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [5:0] bank_global = addr[9:4];
    wire [1:0] tile_id = bank_global[5:4];
    wire [3:0] bank_id = bank_global[3:0];
    wire [31:0] global_dx;
    wire [31:0] global_dy;
    wire [31:0] local_hops;

    assign tile_x = {1'b0, tile_id[0]};
    assign tile_y = {1'b0, tile_id[1]};
    assign bank_x = bank_id[1:0];
    assign bank_y = bank_id[3:2];
    assign global_dx = (source_x > tile_x)
        ? ({30'd0, source_x} - {30'd0, tile_x})
        : ({30'd0, tile_x} - {30'd0, source_x});
    assign global_dy = (source_y > tile_y)
        ? ({30'd0, source_y} - {30'd0, tile_y})
        : ({30'd0, tile_y} - {30'd0, source_y});
    assign local_hops = {30'd0, bank_x} + {30'd0, bank_y};

    always @* begin
        case (source_id)
            4'd1: begin source_x = 2'd0; source_y = 2'd0; end // CONV
            4'd2: begin source_x = 2'd1; source_y = 2'd0; end // REQUANT
            4'd3: begin source_x = 2'd0; source_y = 2'd1; end // EWE
            4'd4: begin source_x = 2'd1; source_y = 2'd1; end // POOL
            4'd5: begin source_x = 2'd0; source_y = 2'd1; end // TNPS
            4'd6: begin source_x = 2'd1; source_y = 2'd0; end // UDMA
            default: begin source_x = 2'd0; source_y = 2'd0; end
        endcase
        route_cycles = BASE_CYCLES +
                       (global_dx + global_dy) * GLOBAL_HOP_CYCLES +
                       local_hops * LOCAL_HOP_CYCLES;
    end
endmodule

/* verilator lint_on DECLFILENAME */

`endif
