`timescale 1ns/1ps

module host #(
    parameter MAX_PROGRAM_BYTES = 65536,
    parameter MAX_LAYERS = 1024
) (
    input             clk,
    input             rst_n,

    output reg        desc_valid,
    input             desc_ready,
    output reg [3:0]  desc_op_class,
    output reg [7:0]  desc_wait_count,
    output reg [15:0] desc_layer_id,
    output reg [15:0] desc_microblock_id,
    output reg [7:0]  desc_stream_slot,
    output reg [7:0]  desc_stream_meta_flags,
    output reg [31:0] cfg_write_cycles,

    output reg [31:0] bytes,
    output reg [31:0] read_bytes,
    output reg [31:0] write_bytes,
    output reg [31:0] act_bytes,
    output reg [31:0] wgt_bytes,
    output reg [31:0] in_elems,
    output reg [31:0] out_elems,
    output reg [31:0] total_elems,
    output reg [31:0] lanes,
    output reg [31:0] window,
    output reg [31:0] mac_cycles,
    output reg [31:0] compute_cycles,
    output reg [31:0] fill_cycles,
    output reg        skip_l1_write,
    output reg [31:0] layer_index_desc,
    output reg [31:0] layer_ref_off,
    output reg [31:0] layer_ref_size,

    output reg        udma_direction_write,
    output reg [31:0] udma_dram_read_bytes,
    output reg [31:0] udma_codec_cycles,

    output reg        l1_req_write,
    output reg        l1mgr_req_l1,
    output reg [3:0]  l1mgr_req_source,
    output reg [7:0]  l1mgr_req_tid,
    output reg [31:0] l1mgr_payload_cycles,
    output reg [21:0] l1mesh_addr,
    output reg [31:0] l1mesh_route_cycles,
    output reg [127:0] l1mesh_wdata,
    output reg [15:0] l1mesh_wstrb,

    input             top_done_valid,
    output            top_done_ready,
    input             top_busy,
    input      [3:0]  active_op_class,
    input      [3:0]  active_phase_id,
    input      [31:0] active_remaining_cycles,
    input      [8:0]  block_busy,
    input      [8:0]  block_done_valid,
    input      [31:0] top_datapath_crc,
    input             top_datapath_ok,

    output reg        dram_req_valid,
    input             dram_req_ready,
    output            dram_req_write,
    output reg [31:0] dram_req_addr,
    output reg [31:0] dram_req_wdata,
    output     [3:0]  dram_req_wstrb,
    input             dram_resp_valid,
    output            dram_resp_ready,
    input      [31:0] dram_resp_rdata,

    output reg        test_done,
    output reg        test_fail
);
    localparam [31:0] MAGIC_MDL7 = 32'h374c444d;
    localparam [31:0] FNV_OFFSET = 32'h811c9dc5;
    localparam [31:0] FNV_PRIME  = 32'd16777619;

    localparam [2:0] ST_IDLE       = 3'd0;
    localparam [2:0] ST_ISSUE      = 3'd1;
    localparam [2:0] ST_WAIT_DONE  = 3'd2;
    localparam [2:0] ST_STATUS_REQ = 3'd3;
    localparam [2:0] ST_STATUS_RSP = 3'd4;
    localparam [2:0] ST_NEXT       = 3'd5;
    localparam [2:0] ST_DONE       = 3'd6;

    localparam [3:0] OP_COMMAND_ONLY = 4'd0;
    localparam [3:0] OP_CONV         = 4'd1;
    localparam [3:0] OP_REQUANT      = 4'd2;
    localparam [3:0] OP_EWE          = 4'd3;
    localparam [3:0] OP_POOL         = 4'd4;
    localparam [3:0] OP_TNPS         = 4'd5;
    localparam [3:0] OP_UDMA         = 4'd6;
    localparam [3:0] OP_L1MANAGER    = 4'd7;

    localparam [7:0] SMF_LOAD_A      = 8'h01;
    localparam [7:0] SMF_LOAD_B      = 8'h02;
    localparam [7:0] SMF_COMPUTE     = 8'h04;
    localparam [7:0] SMF_STORE       = 8'h08;
    localparam [7:0] SMF_FINAL_TILE  = 8'h10;

    localparam [31:0] MICRO_TILE_BYTES = 32'd1048576;

    reg [2:0] state;
    reg [31:0] layer_index;
    reg [31:0] micro_index;
    reg [2:0]  micro_step;
    reg [31:0] micro_total;
    reg [2:0]  micro_steps;
    reg [3:0]  layer_main_class;
    reg        final_micro_desc;
    reg [8:0] expected_done_bit;
    reg       saw_block_done;
    reg       saw_l1mgr_done;
    reg       saw_l1mesh_done;
    reg [31:0] watchdog;

    reg [7:0] program_mem [0:MAX_PROGRAM_BYTES-1];
    reg [31:0] timing_cycles [0:MAX_LAYERS-1];
    reg [1023:0] program_path;
    reg [1023:0] timing_path;
    integer program_fd;
    integer checksum_fd;
    integer checksum_seek_rc;
    integer checksum_ch;
    integer checksum_i;
    integer program_bytes;
    integer load_count;
    reg load_ok;
    reg timing_loaded;
    reg [31:0] program_magic;
    reg [31:0] program_version;
    reg [31:0] program_layers;
    reg [31:0] program_data_offset;

    reg [15:0] meta_in_h;
    reg [15:0] meta_in_w;
    reg [15:0] meta_in_c;
    reg [15:0] meta_out_h;
    reg [15:0] meta_out_w;
    reg [15:0] meta_out_c;
    reg [7:0]  meta_k_h;
    reg [7:0]  meta_k_w;
    reg [31:0] meta_in_size;
    reg [31:0] meta_wgt_size;
    reg [31:0] meta_ref_size;
    reg [31:0] meta_ref_off;
    reg [15:0] meta_op_kind;
    reg [15:0] meta_dtype;
    reg [31:0] meta_timing_cycles;
    reg [31:0] layer_base;
    reg [31:0] expected_crc;
    reg        expected_crc_ok;

    assign top_done_ready = 1'b1;
    assign dram_req_write = 1'b1;
    assign dram_req_wstrb = 4'hf;
    assign dram_resp_ready = 1'b1;

    function [7:0] rd8;
        input [31:0] off;
        begin
            rd8 = (off < MAX_PROGRAM_BYTES) ? program_mem[off] : 8'd0;
        end
    endfunction

    function [15:0] rd16;
        input [31:0] off;
        begin
            rd16 = {rd8(off + 32'd1), rd8(off)};
        end
    endfunction

    function [31:0] rd32;
        input [31:0] off;
        begin
            rd32 = {rd8(off + 32'd3), rd8(off + 32'd2), rd8(off + 32'd1), rd8(off)};
        end
    endfunction

    function [31:0] product3;
        input [15:0] a;
        input [15:0] b;
        input [15:0] c;
        reg [63:0] tmp;
        begin
            tmp = a * b * c;
            product3 = (tmp > 64'h0000_0000_ffff_ffff) ? 32'hffff_ffff : tmp[31:0];
        end
    endfunction

    function [31:0] bounded_bytes;
        input [31:0] value;
        begin
            if (value == 32'd0)
                bounded_bytes = 32'd64;
            else if (value > 32'd4096)
                bounded_bytes = 32'd4096;
            else
                bounded_bytes = value;
        end
    endfunction

    function [31:0] bounded_elems;
        input [31:0] value;
        begin
            if (value == 32'd0)
                bounded_elems = 32'd1;
            else if (value > 32'd4096)
                bounded_elems = 32'd4096;
            else
                bounded_elems = value;
        end
    endfunction

    function [31:0] short_cycles;
        input [31:0] value;
        begin
            short_cycles = 32'd1 + (value % 32'd32);
        end
    endfunction

    function [31:0] max32;
        input [31:0] a;
        input [31:0] b;
        begin
            max32 = (a > b) ? a : b;
        end
    endfunction

    function [31:0] min32;
        input [31:0] a;
        input [31:0] b;
        begin
            min32 = (a < b) ? a : b;
        end
    endfunction

    function [31:0] max1_32;
        input [31:0] value;
        begin
            max1_32 = (value == 32'd0) ? 32'd1 : value;
        end
    endfunction

    function [31:0] ceil_div32;
        input [31:0] value;
        input [31:0] denom;
        reg [63:0] tmp;
        reg [63:0] denom64;
        /* verilator lint_off UNUSEDSIGNAL */
        reg [63:0] quotient;
        /* verilator lint_on UNUSEDSIGNAL */
        begin
            if ((value == 32'd0) || (denom == 32'd0)) begin
                ceil_div32 = 32'd1;
            end else begin
                tmp = {32'd0, value} + {32'd0, denom} - 64'd1;
                denom64 = {32'd0, denom};
                quotient = tmp / denom64;
                ceil_div32 = quotient[31:0];
            end
        end
    endfunction

    function [31:0] slice_bytes;
        input [31:0] total;
        input [31:0] off;
        begin
            if (total <= off)
                slice_bytes = 32'd0;
            else
                slice_bytes = min32(MICRO_TILE_BYTES, total - off);
        end
    endfunction

    function [31:0] crc_byte;
        input [31:0] crc;
        input [7:0]  byte_value;
        begin
            crc_byte = (crc ^ {24'd0, byte_value}) * FNV_PRIME;
        end
    endfunction

    task checksum_file_region;
        input [31:0] off;
        input [31:0] size;
        output [31:0] crc;
        output ok;
        begin
            crc = FNV_OFFSET;
            ok = 1'b0;
            checksum_fd = $fopen(program_path, "rb");
            if (checksum_fd == 0) begin
                $display("HOST_FAIL: failed to open PROGRAM for checksum=%0s", program_path);
            end else begin
                checksum_seek_rc = $fseek(checksum_fd, off, 0);
                if (checksum_seek_rc != 0) begin
                    $display("HOST_FAIL: failed to seek ref_off=%0d", off);
                end else begin
                    ok = 1'b1;
                    for (checksum_i = 0; checksum_i < size; checksum_i = checksum_i + 1) begin
                        checksum_ch = $fgetc(checksum_fd);
                        if (checksum_ch < 0) begin
                            ok = 1'b0;
                        end else begin
                            crc = crc_byte(crc, checksum_ch[7:0]);
                        end
                    end
                end
                $fclose(checksum_fd);
            end
        end
    endtask

    function [3:0] op_kind_to_class;
        input [15:0] op_kind;
        begin
            case (op_kind)
                16'd0, 16'd1, 16'd6:
                    op_kind_to_class = OP_CONV;
                16'd2, 16'd3:
                    op_kind_to_class = OP_POOL;
                16'd4, 16'd7, 16'd10, 16'd11, 16'd12, 16'd13, 16'd27:
                    op_kind_to_class = OP_EWE;
                16'd5, 16'd8, 16'd14, 16'd16, 16'd17, 16'd18, 16'd19,
                16'd20, 16'd21, 16'd22, 16'd23, 16'd24, 16'd25, 16'd26:
                    op_kind_to_class = OP_TNPS;
                16'd9, 16'd15:
                    op_kind_to_class = OP_UDMA;
                default:
                    op_kind_to_class = OP_COMMAND_ONLY;
            endcase
        end
    endfunction

    function [2:0] micro_steps_for_class;
        input [3:0] op_class;
        begin
            case (op_class)
                OP_CONV:
                    micro_steps_for_class = 3'd5;
                OP_EWE:
                    micro_steps_for_class = 3'd4;
                OP_POOL, OP_TNPS:
                    micro_steps_for_class = 3'd3;
                default:
                    micro_steps_for_class = 3'd1;
            endcase
        end
    endfunction

    function [3:0] micro_step_op;
        input [3:0] op_class;
        input [2:0] step;
        begin
            case (op_class)
                OP_CONV: begin
                    case (step)
                        3'd2:
                            micro_step_op = OP_CONV;
                        3'd3:
                            micro_step_op = OP_REQUANT;
                        default:
                            micro_step_op = OP_UDMA;
                    endcase
                end
                OP_EWE: begin
                    micro_step_op = (step == 3'd2) ? OP_EWE : OP_UDMA;
                end
                OP_POOL: begin
                    micro_step_op = (step == 3'd1) ? OP_POOL : OP_UDMA;
                end
                OP_TNPS: begin
                    micro_step_op = (step == 3'd1) ? OP_TNPS : OP_UDMA;
                end
                OP_UDMA:
                    micro_step_op = OP_UDMA;
                default:
                    micro_step_op = OP_COMMAND_ONLY;
            endcase
        end
    endfunction

    function [7:0] micro_step_flags;
        input [3:0] op_class;
        input [2:0] step;
        begin
            case (op_class)
                OP_CONV: begin
                    case (step)
                        3'd0:
                            micro_step_flags = SMF_LOAD_B;
                        3'd1:
                            micro_step_flags = SMF_LOAD_A;
                        3'd2, 3'd3:
                            micro_step_flags = SMF_COMPUTE;
                        default:
                            micro_step_flags = SMF_STORE;
                    endcase
                end
                OP_EWE: begin
                    case (step)
                        3'd0:
                            micro_step_flags = SMF_LOAD_A;
                        3'd1:
                            micro_step_flags = SMF_LOAD_B;
                        3'd2:
                            micro_step_flags = SMF_COMPUTE;
                        default:
                            micro_step_flags = SMF_STORE;
                    endcase
                end
                OP_POOL, OP_TNPS: begin
                    case (step)
                        3'd0:
                            micro_step_flags = SMF_LOAD_A;
                        3'd1:
                            micro_step_flags = SMF_COMPUTE;
                        default:
                            micro_step_flags = SMF_STORE;
                    endcase
                end
                OP_UDMA:
                    micro_step_flags = SMF_STORE;
                default:
                    micro_step_flags = 8'd0;
            endcase
        end
    endfunction

    always @* begin
        expected_done_bit = (9'b000000001 << desc_op_class);
    end

    task read_layer_meta;
        input [31:0] layer;
        begin
            layer_base = 32'd16 + (layer * 32'd64);
            meta_in_h = rd16(layer_base + 32'd0);
            meta_in_w = rd16(layer_base + 32'd2);
            meta_in_c = rd16(layer_base + 32'd4);
            meta_out_h = rd16(layer_base + 32'd6);
            meta_out_w = rd16(layer_base + 32'd8);
            meta_out_c = rd16(layer_base + 32'd10);
            meta_k_h = rd8(layer_base + 32'd12);
            meta_k_w = rd8(layer_base + 32'd13);
            meta_in_size = rd32(layer_base + 32'd32);
            meta_wgt_size = rd32(layer_base + 32'd36);
            meta_ref_size = rd32(layer_base + 32'd40);
            meta_ref_off = rd32(layer_base + 32'd52);
            meta_op_kind = rd16(layer_base + 32'd58);
            meta_dtype = rd16(layer_base + 32'd60);
            meta_timing_cycles = (layer < MAX_LAYERS) ? timing_cycles[layer] : 32'd0;
        end
    endtask

    task prepare_layer;
        input [31:0] layer;
        reg [31:0] payload_max;
        begin
            read_layer_meta(layer);
            layer_main_class = op_kind_to_class(meta_op_kind);
            micro_steps = micro_steps_for_class(layer_main_class);
            payload_max = max32(max32(meta_in_size, meta_wgt_size), meta_ref_size);
            micro_total = ceil_div32(payload_max, MICRO_TILE_BYTES);
            checksum_file_region(meta_ref_off, meta_ref_size, expected_crc, expected_crc_ok);
        end
    endtask

    task drive_micro_descriptor;
        input [31:0] layer;
        input [31:0] mb;
        input [2:0]  step;
        reg [31:0] elems_in;
        reg [31:0] elems_out;
        reg [31:0] payload_off;
        reg [31:0] payload_in;
        reg [31:0] payload_wgt;
        reg [31:0] payload_out;
        reg [31:0] payload_any;
        reg [31:0] desc_budget;
        reg [31:0] timed_l1_cycles;
        reg [31:0] timed_engine_cycles;
        reg [31:0] timed_aux_cycles;
        reg [7:0]  stream_flags;
        begin
            elems_in = bounded_elems(product3(meta_in_h, meta_in_w, meta_in_c));
            elems_out = bounded_elems(product3(meta_out_h, meta_out_w, meta_out_c));
            payload_off = mb * MICRO_TILE_BYTES;
            payload_in = bounded_bytes(slice_bytes(meta_in_size, payload_off));
            payload_wgt = bounded_bytes(slice_bytes(meta_wgt_size, payload_off));
            payload_out = bounded_bytes(slice_bytes(meta_ref_size, payload_off));
            payload_any = bounded_bytes(slice_bytes(max32(max32(meta_in_size, meta_wgt_size), meta_ref_size),
                                                   payload_off));
            if (meta_timing_cycles != 32'd0) begin
                desc_budget = ceil_div32(meta_timing_cycles, micro_total * {29'd0, micro_steps});
                timed_l1_cycles = max1_32(desc_budget / 32'd3);
                timed_engine_cycles = max1_32(desc_budget - timed_l1_cycles);
                timed_aux_cycles = max1_32(desc_budget / 32'd8);
            end else if (timing_loaded) begin
                desc_budget = 32'd1;
                timed_l1_cycles = 32'd1;
                timed_engine_cycles = 32'd1;
                timed_aux_cycles = 32'd1;
            end else begin
                desc_budget = 32'd0;
                timed_l1_cycles = 32'd0;
                timed_engine_cycles = 32'd0;
                timed_aux_cycles = 32'd0;
            end
            stream_flags = micro_step_flags(layer_main_class, step);
            final_micro_desc = ((mb + 32'd1) >= micro_total) &&
                               ((step + 3'd1) >= micro_steps);
            if (final_micro_desc)
                stream_flags = stream_flags | SMF_FINAL_TILE;

            desc_op_class = micro_step_op(layer_main_class, step);
            desc_wait_count = {5'd0, step};
            desc_layer_id = layer[15:0];
            desc_microblock_id = mb[15:0];
            desc_stream_slot = mb[7:0];
            desc_stream_meta_flags = stream_flags;
            cfg_write_cycles = timing_loaded ? 32'd1 : 32'd2;
            bytes = payload_any;
            read_bytes = ((stream_flags & SMF_LOAD_B) != 8'd0) ? payload_wgt : payload_in;
            write_bytes = payload_out;
            act_bytes = payload_in;
            wgt_bytes = payload_wgt;
            in_elems = elems_in;
            out_elems = elems_out;
            total_elems = elems_out;
            lanes = (meta_dtype == 16'd0) ? 32'd64 : 32'd32;
            window = {16'd0, meta_k_h} * {16'd0, meta_k_w};
            if (window == 32'd0)
                window = 32'd1;
            mac_cycles = timing_loaded ? timed_engine_cycles : short_cycles(elems_out);
            compute_cycles = timing_loaded ? timed_engine_cycles : short_cycles(elems_out);
            fill_cycles = timing_loaded
                ? timed_aux_cycles
                : ((meta_op_kind == 16'd1) ? 32'd48 : 32'd64);
            skip_l1_write = 1'b0;
            layer_index_desc = layer;
            layer_ref_off = meta_ref_off;
            layer_ref_size = final_micro_desc ? meta_ref_size : 32'd0;
            udma_direction_write = ((stream_flags & SMF_STORE) != 8'd0);
            udma_dram_read_bytes = ((stream_flags & SMF_STORE) != 8'd0) ? payload_out :
                                   ((stream_flags & SMF_LOAD_B) != 8'd0) ? payload_wgt :
                                   payload_in;
            udma_codec_cycles = timing_loaded
                ? timed_engine_cycles
                : short_cycles(udma_dram_read_bytes);
            l1_req_write = ((stream_flags & SMF_STORE) != 8'd0);
            l1mgr_req_l1 = 1'b1;
            l1mgr_req_source = desc_op_class;
            l1mgr_req_tid = {mb[3:0], 1'b0, step};
            l1mgr_payload_cycles = timing_loaded
                ? timed_l1_cycles
                : short_cycles(payload_any);
            l1mesh_addr = {layer[9:0], mb[7:0], 1'b0, step};
            l1mesh_route_cycles = timing_loaded ? timed_l1_cycles : 32'd3;
            l1mesh_wdata = {8{12'h0a5, desc_op_class}};
            l1mesh_wstrb = 16'hffff;
        end
    endtask

    initial begin
        program_path = "rtl/bin/Hotspot/gpt2_quant_L24_L63.bin";
        if (!$value$plusargs("PROGRAM=%s", program_path))
            program_path = "rtl/bin/Hotspot/gpt2_quant_L24_L63.bin";

        load_ok = 1'b0;
        timing_loaded = 1'b0;
        timing_path = "";
        for (load_count = 0; load_count < MAX_PROGRAM_BYTES; load_count = load_count + 1)
            program_mem[load_count] = 8'd0;
        for (load_count = 0; load_count < MAX_LAYERS; load_count = load_count + 1)
            timing_cycles[load_count] = 32'd0;

        program_fd = $fopen(program_path, "rb");
        if (program_fd == 0) begin
            $display("HOST: failed to open PROGRAM=%0s", program_path);
        end else begin
            program_bytes = $fread(program_mem, program_fd, 0, MAX_PROGRAM_BYTES);
            $fclose(program_fd);
            program_magic = rd32(32'd0);
            program_version = rd32(32'd4);
            program_layers = rd32(32'd8);
            program_data_offset = rd32(32'd12);
            load_ok = (program_magic == MAGIC_MDL7) &&
                      (program_layers != 32'd0) &&
                      (program_layers <= MAX_LAYERS) &&
                      (32'd16 + program_layers * 32'd64 <= MAX_PROGRAM_BYTES);
            if (load_ok) begin
                $display("HOST: loaded %0s header_bytes=%0d version=%0d layers=%0d data_offset=%0d",
                         program_path, program_bytes, program_version,
                         program_layers, program_data_offset);
            end else begin
                $display("HOST: invalid program %0s magic=%08x layers=%0d bytes=%0d",
                         program_path, program_magic, program_layers, program_bytes);
            end
        end

        if ($value$plusargs("TIMING=%s", timing_path)) begin
            $readmemh(timing_path, timing_cycles);
            timing_loaded = 1'b1;
            $display("HOST: loaded timing %0s", timing_path);
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            layer_index <= 32'd0;
            micro_index <= 32'd0;
            micro_step <= 3'd0;
            micro_total <= 32'd1;
            micro_steps <= 3'd1;
            layer_main_class <= OP_COMMAND_ONLY;
            final_micro_desc <= 1'b0;
            desc_valid <= 1'b0;
            desc_op_class <= OP_COMMAND_ONLY;
            desc_wait_count <= 8'd0;
            desc_layer_id <= 16'd0;
            desc_microblock_id <= 16'd0;
            desc_stream_slot <= 8'd0;
            desc_stream_meta_flags <= 8'd0;
            cfg_write_cycles <= 32'd0;
            bytes <= 32'd0;
            read_bytes <= 32'd0;
            write_bytes <= 32'd0;
            act_bytes <= 32'd0;
            wgt_bytes <= 32'd0;
            in_elems <= 32'd0;
            out_elems <= 32'd0;
            total_elems <= 32'd0;
            lanes <= 32'd0;
            window <= 32'd1;
            mac_cycles <= 32'd1;
            compute_cycles <= 32'd1;
            fill_cycles <= 32'd1;
            skip_l1_write <= 1'b0;
            layer_index_desc <= 32'd0;
            layer_ref_off <= 32'd0;
            layer_ref_size <= 32'd0;
            udma_direction_write <= 1'b0;
            udma_dram_read_bytes <= 32'd0;
            udma_codec_cycles <= 32'd0;
            l1_req_write <= 1'b0;
            l1mgr_req_l1 <= 1'b1;
            l1mgr_req_source <= OP_COMMAND_ONLY;
            l1mgr_req_tid <= 8'd0;
            l1mgr_payload_cycles <= 32'd0;
            l1mesh_addr <= 22'd0;
            l1mesh_route_cycles <= 32'd0;
            l1mesh_wdata <= 128'd0;
            l1mesh_wstrb <= 16'd0;
            dram_req_valid <= 1'b0;
            dram_req_addr <= 32'd0;
            dram_req_wdata <= 32'd0;
            test_done <= 1'b0;
            test_fail <= 1'b0;
            saw_block_done <= 1'b0;
            saw_l1mgr_done <= 1'b0;
            saw_l1mesh_done <= 1'b0;
            watchdog <= 32'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    test_done <= 1'b0;
                    test_fail <= !load_ok;
                    layer_index <= 32'd0;
                    micro_index <= 32'd0;
                    micro_step <= 3'd0;
                    if (load_ok) begin
                        prepare_layer(32'd0);
                        state <= ST_ISSUE;
                    end else begin
                        state <= ST_DONE;
                    end
                end
                ST_ISSUE: begin
                    if (!desc_valid) begin
                        drive_micro_descriptor(layer_index, micro_index, micro_step);
                        desc_valid <= 1'b1;
                        saw_block_done <= 1'b0;
                        saw_l1mgr_done <= 1'b0;
                        saw_l1mesh_done <= 1'b0;
                        watchdog <= 32'd0;
                    end
                    if (desc_valid && desc_ready) begin
                        desc_valid <= 1'b0;
                        state <= ST_WAIT_DONE;
                    end
                end
                ST_WAIT_DONE: begin
                    watchdog <= watchdog + 32'd1;
                    if ((block_done_valid & expected_done_bit) != 9'd0)
                        saw_block_done <= 1'b1;
                    if (block_done_valid[7])
                        saw_l1mgr_done <= 1'b1;
                    if (block_done_valid[8])
                        saw_l1mesh_done <= 1'b1;
                    if (watchdog == 32'd2000000) begin
                        $display("HOST_FAIL: timeout layer=%0d op_class=%0d active_op=%0d phase=%0d remaining=%0d top_busy=%0d block_busy=%09b block_done=%09b",
                                 layer_index, desc_op_class, active_op_class,
                                 active_phase_id, active_remaining_cycles, top_busy,
                                 block_busy, block_done_valid);
                        test_fail <= 1'b1;
                        state <= ST_DONE;
                    end else if (top_done_valid) begin
                        if (!saw_block_done && ((block_done_valid & expected_done_bit) == 9'd0)) begin
                            $display("HOST_FAIL: missing block done layer=%0d op_kind=%0d op_class=%0d block_done=%09b expected=%09b",
                                     layer_index, meta_op_kind, desc_op_class,
                                     block_done_valid, expected_done_bit);
                            test_fail <= 1'b1;
                        end
                        if (active_op_class != desc_op_class) begin
                            $display("HOST_FAIL: active op mismatch layer=%0d expected=%0d got=%0d",
                                     layer_index, desc_op_class, active_op_class);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class != OP_COMMAND_ONLY) &&
                            (desc_op_class != OP_CONV) &&
                            !saw_l1mgr_done && !block_done_valid[7]) begin
                            $display("HOST_FAIL: missing L1Manager done layer=%0d micro=%0d step=%0d op_class=%0d block_done=%09b",
                                     layer_index, micro_index, micro_step,
                                     desc_op_class, block_done_valid);
                            test_fail <= 1'b1;
                        end
                        if ((desc_op_class != OP_COMMAND_ONLY) &&
                            (desc_op_class != OP_L1MANAGER) &&
                            !saw_l1mesh_done && !block_done_valid[8]) begin
                            $display("HOST_FAIL: missing L1Mesh done layer=%0d micro=%0d step=%0d op_class=%0d block_done=%09b",
                                     layer_index, micro_index, micro_step,
                                     desc_op_class, block_done_valid);
                            test_fail <= 1'b1;
                        end
                        if (final_micro_desc && (desc_op_class != OP_COMMAND_ONLY) &&
                            (meta_ref_size != 32'd0)) begin
                            if (!expected_crc_ok || !top_datapath_ok ||
                                (top_datapath_crc != expected_crc)) begin
                                $display("HOST_FAIL: datapath mismatch layer=%0d op_kind=%0d op_class=%0d expected_crc=%08x got_crc=%08x exp_ok=%0d got_ok=%0d ref_off=%0d ref_size=%0d",
                                         layer_index, meta_op_kind, desc_op_class,
                                         expected_crc, top_datapath_crc,
                                         expected_crc_ok, top_datapath_ok,
                                         meta_ref_off, meta_ref_size);
                                test_fail <= 1'b1;
                            end
                        end
                        dram_req_valid <= 1'b1;
                        dram_req_addr <= 32'h0000_0100 +
                                         {4'd0, layer_index[15:0], micro_index[7:0],
                                          micro_step[1:0], 2'd0};
                        dram_req_wdata <= {
                            16'h7a5a,
                            final_micro_desc,
                            desc_stream_meta_flags[4:0],
                            ((desc_op_class == OP_CONV) ||
                             saw_l1mgr_done || block_done_valid[7]) &&
                            (saw_l1mesh_done || block_done_valid[8]) &&
                            (saw_block_done || ((block_done_valid & expected_done_bit) != 9'd0)),
                            top_busy,
                            test_fail,
                            desc_op_class,
                            micro_step
                        };
                        state <= ST_STATUS_REQ;
                    end
                end
                ST_STATUS_REQ: begin
                    if (dram_req_valid && dram_req_ready) begin
                        dram_req_valid <= 1'b0;
                        state <= ST_STATUS_RSP;
                    end
                end
                ST_STATUS_RSP: begin
                    if (dram_resp_valid) begin
                        if (dram_resp_rdata[31:16] != 16'h7a5a) begin
                            $display("HOST_FAIL: bad DRAM status echo layer=%0d rdata=%08x",
                                     layer_index, dram_resp_rdata);
                            test_fail <= 1'b1;
                        end
                        state <= ST_NEXT;
                    end
                end
                ST_NEXT: begin
                    if (!final_micro_desc) begin
                        if ((micro_step + 3'd1) >= micro_steps) begin
                            micro_step <= 3'd0;
                            micro_index <= micro_index + 32'd1;
                        end else begin
                            micro_step <= micro_step + 3'd1;
                        end
                        state <= ST_ISSUE;
                    end else begin
                        if (layer_index + 32'd1 >= program_layers) begin
                            state <= ST_DONE;
                        end else begin
                            layer_index <= layer_index + 32'd1;
                            micro_index <= 32'd0;
                            micro_step <= 3'd0;
                            prepare_layer(layer_index + 32'd1);
                            state <= ST_ISSUE;
                        end
                    end
                end
                ST_DONE: begin
                    desc_valid <= 1'b0;
                    dram_req_valid <= 1'b0;
                    test_done <= 1'b1;
                end
                default: begin
                    state <= ST_DONE;
                    test_fail <= 1'b1;
                end
            endcase
        end
    end
endmodule
