`timescale 1ns/1ps

module dram #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 32,
    parameter MEM_WORDS = 1024
) (
    input                       clk,
    input                       rst_n,

    input                       req_valid,
    output                      req_ready,
    input                       req_write,
    input      [31:0]           req_addr,
    input      [DATA_WIDTH-1:0] req_wdata,
    input      [DATA_WIDTH/8-1:0] req_wstrb,

    output reg                  resp_valid,
    input                       resp_ready,
    output reg [DATA_WIDTH-1:0] resp_rdata
);
    localparam STRB_WIDTH = DATA_WIDTH / 8;
    localparam WORD_ADDR_WIDTH = $clog2(MEM_WORDS);
    localparam [31:0] ADDR_MASK = (ADDR_WIDTH >= 32)
        ? 32'hffff_ffff
        : (32'hffff_ffff >> (32 - ADDR_WIDTH));

    reg [DATA_WIDTH-1:0] mem [0:MEM_WORDS-1];
    wire [31:0] req_addr_sized = req_addr & ADDR_MASK;
    wire req_addr_in_window = ((req_addr & ~ADDR_MASK) == 32'd0);
    wire [31:0] word_addr_full = req_addr_sized >> 2;
    wire [WORD_ADDR_WIDTH-1:0] word_addr = word_addr_full[WORD_ADDR_WIDTH-1:0];
    wire word_addr_in_range = req_addr_in_window && (word_addr_full < MEM_WORDS);
    wire fire = req_valid && req_ready;

    integer i;

    assign req_ready = !resp_valid || resp_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_valid <= 1'b0;
            resp_rdata <= {DATA_WIDTH{1'b0}};
        end else begin
            if (resp_valid && resp_ready)
                resp_valid <= 1'b0;

            if (fire) begin
                if (req_write) begin
                    if (word_addr_in_range) begin
                        for (i = 0; i < STRB_WIDTH; i = i + 1) begin
                            if (req_wstrb[i])
                                mem[word_addr][i*8 +: 8] <= req_wdata[i*8 +: 8];
                        end
                    end
                    resp_rdata <= req_wdata;
                end else begin
                    resp_rdata <= word_addr_in_range
                        ? mem[word_addr]
                        : {DATA_WIDTH{1'b0}};
                end
                resp_valid <= 1'b1;
            end
        end
    end
endmodule
