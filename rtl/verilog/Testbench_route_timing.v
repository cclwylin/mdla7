`timescale 1ns/1ps

module Testbench_route_timing;
    reg [3:0] source_id;
    reg [21:0] addr;
    wire [31:0] route_cycles;
    wire [1:0] source_x;
    wire [1:0] source_y;
    wire [1:0] tile_x;
    wire [1:0] tile_y;
    wire [1:0] bank_x;
    wire [1:0] bank_y;
    integer failures;

    vf_l1mesh_route_estimator #(
        .BASE_CYCLES(1),
        .GLOBAL_HOP_CYCLES(2),
        .LOCAL_HOP_CYCLES(1)
    ) u_route (
        .source_id(source_id),
        .addr(addr),
        .route_cycles(route_cycles),
        .source_x(source_x),
        .source_y(source_y),
        .tile_x(tile_x),
        .tile_y(tile_y),
        .bank_x(bank_x),
        .bank_y(bank_y)
    );

    task expect_route;
        input [3:0] src;
        input [21:0] a;
        input [31:0] exp;
        begin
            source_id = src;
            addr = a;
            #1;
            if (route_cycles != exp) begin
                $display("FAIL: route src=%0d addr=0x%0x got=%0d exp=%0d srcxy=%0d,%0d tile=%0d,%0d bank=%0d,%0d",
                         src, a, route_cycles, exp,
                         source_x, source_y, tile_x, tile_y, bank_x, bank_y);
                failures = failures + 1;
            end
        end
    endtask

    initial begin
        failures = 0;
        expect_route(4'd1, 22'h000000, 32'd1);
        expect_route(4'd6, 22'h000000, 32'd3);
        expect_route(4'd3, 22'h0003f0, 32'd9);
        expect_route(4'd4, 22'h0002a0, 32'd7);

        if (failures == 0)
            $display("PASS: verilog placement-aware route timing");
        else
            $display("FAIL: verilog placement-aware route timing failures=%0d", failures);
        $finish;
    end
endmodule
