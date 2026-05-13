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
    wire [7:0] addr_unused = addr[3:0];
    /* verilator lint_on UNUSEDSIGNAL */
    wire [31:0] word_addr = {10'd0, addr} >> 4;
    wire [1:0] storage_mesh_id = word_addr[1:0];
    wire [3:0] quad_sram_id = word_addr[5:2];
    wire [1:0] sram_macro_port = word_addr[7:6];
    wire       edge_mesh_half = word_addr[8];
    wire [1:0] edge_mesh_x = {edge_mesh_half, storage_mesh_id[0]};
    wire [1:0] edge_mesh_y = {1'b0, storage_mesh_id[1]};
    wire [31:0] global_dx;
    wire [31:0] global_dy;
    wire [31:0] local_hops;

    assign tile_x = edge_mesh_x;
    assign tile_y = edge_mesh_y;
    assign bank_x = quad_sram_id[1:0];
    assign bank_y = quad_sram_id[3:2];
    assign global_dx = (source_x > edge_mesh_x)
        ? ({30'd0, source_x} - {30'd0, edge_mesh_x})
        : ({30'd0, edge_mesh_x} - {30'd0, source_x});
    assign global_dy = (source_y > edge_mesh_y)
        ? ({30'd0, source_y} - {30'd0, edge_mesh_y})
        : ({30'd0, edge_mesh_y} - {30'd0, source_y});
    assign local_hops = {30'd0, bank_x} + {30'd0, bank_y} +
                        {30'd0, sram_macro_port};

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
