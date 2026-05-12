`timescale 1ns/1ps

module mdla7_top #(
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                       clk,
    input                       rst_n,

    input                       desc_valid,
    output                      desc_ready,
    input      [3:0]            desc_op_class,
    input      [7:0]            desc_wait_count,
    input      [15:0]           desc_layer_id,
    input      [15:0]           desc_microblock_id,
    input      [7:0]            desc_stream_slot,
    input      [7:0]            desc_stream_meta_flags,
    input      [31:0]           cfg_write_cycles,

    input      [31:0]           bytes,
    input      [31:0]           read_bytes,
    input      [31:0]           write_bytes,
    input      [31:0]           act_bytes,
    input      [31:0]           wgt_bytes,
    input      [31:0]           in_elems,
    input      [31:0]           out_elems,
    input      [31:0]           total_elems,
    input      [31:0]           lanes,
    input      [31:0]           window,
    input      [31:0]           mac_cycles,
    input      [31:0]           compute_cycles,
    input      [31:0]           fill_cycles,
    input                       skip_l1_write,
    input      [31:0]           layer_index,
    input      [31:0]           ref_off,
    input      [31:0]           ref_size,

    input                       udma_direction_write,
    input      [31:0]           udma_dram_read_bytes,
    input      [31:0]           udma_codec_cycles,

    input                       l1_req_write,
    input                       l1mgr_req_l1,
    input      [3:0]            l1mgr_req_source,
    input      [7:0]            l1mgr_req_tid,
    input      [31:0]           l1mgr_payload_cycles,
    input      [ADDR_WIDTH-1:0] l1mesh_addr,
    input      [31:0]           l1mesh_route_cycles,
    input      [DATA_WIDTH-1:0] l1mesh_wdata,
    input      [DATA_WIDTH/8-1:0] l1mesh_wstrb,

    output     [DATA_WIDTH-1:0] l1mesh_rdata,
    output reg                  done_valid,
    input                       done_ready,
    output                      busy,
    output     [3:0]            active_op_class,
    output reg [3:0]            active_phase_id,
    output reg [31:0]           active_remaining_cycles,
    output     [8:0]            block_busy,
    output     [8:0]            block_done_valid,
    output reg [31:0]           datapath_crc,
    output reg                  datapath_ok
);
    localparam [3:0] OP_COMMAND_ONLY = 4'd0;
    localparam [3:0] OP_CONV         = 4'd1;
    localparam [3:0] OP_REQUANT      = 4'd2;
    localparam [3:0] OP_EWE          = 4'd3;
    localparam [3:0] OP_POOL         = 4'd4;
    localparam [3:0] OP_TNPS         = 4'd5;
    localparam [3:0] OP_UDMA         = 4'd6;
    localparam [3:0] OP_L1MANAGER    = 4'd7;
    localparam [3:0] OP_L1MESH       = 4'd8;

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_CMD    = 3'd1;
    localparam [2:0] ST_L1MGR  = 3'd2;
    localparam [2:0] ST_L1MESH = 3'd3;
    localparam [2:0] ST_ENGINE = 3'd4;
    localparam [2:0] ST_DONE   = 3'd5;

    reg [2:0] state;
    reg       cmd_start_pending;
    reg       l1mgr_start_pending;
    reg       l1mesh_start_pending;
    reg       engine_start_pending;
    reg       engine_done_seen;

    reg [3:0]  op_class_q;
    reg [7:0]  wait_count_q;
    reg [15:0] layer_id_q;
    reg [15:0] microblock_id_q;
    reg [7:0]  stream_slot_q;
    reg [7:0]  stream_meta_flags_q;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [47:0] stream_descriptor_debug =
        {layer_id_q, microblock_id_q, stream_slot_q, stream_meta_flags_q};
    /* verilator lint_on UNUSEDSIGNAL */
    reg [31:0] cfg_write_cycles_q;
    reg [31:0] bytes_q;
    reg [31:0] read_bytes_q;
    reg [31:0] write_bytes_q;
    reg [31:0] act_bytes_q;
    reg [31:0] wgt_bytes_q;
    reg [31:0] in_elems_q;
    reg [31:0] out_elems_q;
    reg [31:0] total_elems_q;
    reg [31:0] lanes_q;
    reg [31:0] window_q;
    reg [31:0] mac_cycles_q;
    reg [31:0] compute_cycles_q;
    reg [31:0] fill_cycles_q;
    reg        skip_l1_write_q;
    reg [31:0] layer_index_q;
    reg [31:0] ref_off_q;
    reg [31:0] ref_size_q;
    reg        udma_direction_write_q;
    reg [31:0] udma_dram_read_bytes_q;
    reg [31:0] udma_codec_cycles_q;
    reg        l1_req_write_q;
    reg        l1mgr_req_l1_q;
    reg [3:0]  l1mgr_req_source_q;
    reg [7:0]  l1mgr_req_tid_q;
    reg [31:0] l1mgr_payload_cycles_q;
    reg [ADDR_WIDTH-1:0] l1mesh_addr_q;
    reg [31:0] l1mesh_route_cycles_q;
    reg [DATA_WIDTH-1:0] l1mesh_wdata_q;
    reg [DATA_WIDTH/8-1:0] l1mesh_wstrb_q;

    wire cmd_start_ready;
    wire cmd_busy;
    wire cmd_done_valid;
    wire [3:0] cmd_phase_id;
    wire [31:0] cmd_remaining_cycles;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [7:0] cmd_debug_wait_count;
    wire [7:0] cmd_debug_op_class;
    /* verilator lint_on UNUSEDSIGNAL */

    wire conv_start_ready;
    wire conv_busy;
    wire conv_done_valid;
    wire [3:0] conv_phase_id;
    wire [31:0] conv_remaining_cycles;
    wire [31:0] conv_datapath_crc;
    wire conv_datapath_ok;

    wire requant_start_ready;
    wire requant_busy;
    wire requant_done_valid;
    wire [3:0] requant_phase_id;
    wire [31:0] requant_remaining_cycles;
    wire [31:0] requant_datapath_crc;
    wire requant_datapath_ok;
    wire requant_engine_l1_req_valid;
    wire requant_engine_l1_req_write;
    wire [31:0] requant_engine_l1_req_bytes;
    wire [31:0] requant_engine_l1_req_payload_cycles;

    wire ewe_start_ready;
    wire ewe_busy;
    wire ewe_done_valid;
    wire [3:0] ewe_phase_id;
    wire [31:0] ewe_remaining_cycles;
    wire [31:0] ewe_datapath_crc;
    wire ewe_datapath_ok;
    wire ewe_engine_l1_req_valid;
    wire ewe_engine_l1_req_write;
    wire [31:0] ewe_engine_l1_req_bytes;
    wire [31:0] ewe_engine_l1_req_payload_cycles;

    wire pool_start_ready;
    wire pool_busy;
    wire pool_done_valid;
    wire [3:0] pool_phase_id;
    wire [31:0] pool_remaining_cycles;
    wire [31:0] pool_datapath_crc;
    wire pool_datapath_ok;
    wire pool_engine_l1_req_valid;
    wire pool_engine_l1_req_write;
    wire [31:0] pool_engine_l1_req_bytes;
    wire [31:0] pool_engine_l1_req_payload_cycles;

    wire tnps_start_ready;
    wire tnps_busy;
    wire tnps_done_valid;
    wire [3:0] tnps_phase_id;
    wire [31:0] tnps_remaining_cycles;
    wire [31:0] tnps_datapath_crc;
    wire tnps_datapath_ok;
    wire tnps_engine_l1_req_valid;
    wire tnps_engine_l1_req_write;
    wire [31:0] tnps_engine_l1_req_bytes;
    wire [31:0] tnps_engine_l1_req_payload_cycles;

    wire udma_start_ready;
    wire udma_busy;
    wire udma_done_valid;
    wire [3:0] udma_phase_id;
    wire [31:0] udma_remaining_cycles;
    wire [31:0] udma_datapath_crc;
    wire udma_datapath_ok;
    wire udma_engine_l1_req_valid;
    wire udma_engine_l1_req_write;
    wire [31:0] udma_engine_l1_req_bytes;
    wire [31:0] udma_engine_l1_req_payload_cycles;

    wire l1mgr_req_ready;
    wire udma_l1_req_ready;
    wire requant_l1_req_ready;
    wire ewe_l1_req_ready;
    wire pool_l1_req_ready;
    wire tnps_l1_req_ready;
    wire l1mgr_busy;
    wire l1mgr_resp_valid;
    wire [3:0] l1mgr_phase_id;
    wire [31:0] l1mgr_remaining_cycles;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [3:0] l1mgr_debug_source;
    wire [7:0] l1mgr_debug_tid;
    /* verilator lint_on UNUSEDSIGNAL */
    wire l1mgr_mesh_req_write;
    wire [ADDR_WIDTH-1:0] l1mgr_mesh_req_addr;
    wire [31:0] l1mgr_mesh_req_bytes;
    wire [DATA_WIDTH-1:0] l1mgr_mesh_req_wdata;
    wire [DATA_WIDTH/8-1:0] l1mgr_mesh_req_wstrb;

    wire l1mesh_req_ready;
    wire l1mesh_busy;
    wire l1mesh_resp_valid;
    wire [3:0] l1mesh_phase_id;
    wire [31:0] l1mesh_remaining_cycles;
    /* verilator lint_off UNUSEDSIGNAL */
    wire l1mesh_debug_crc_busy;
    wire l1mesh_debug_crc_done;
    wire [31:0] l1mesh_debug_crc;
    wire [31:0] l1mesh_debug_crc_byte_count;
    /* verilator lint_on UNUSEDSIGNAL */
    wire engine_payload_op =
        (op_class_q == OP_UDMA) ||
        (op_class_q == OP_REQUANT) ||
        (op_class_q == OP_EWE) ||
        (op_class_q == OP_POOL) ||
        (op_class_q == OP_TNPS);
    wire engine_l1_drained =
        !engine_payload_op ||
        (!l1mgr_busy && !l1mgr_resp_valid && !l1mesh_busy && !l1mesh_resp_valid);
    wire l1mgr_resp_ready = engine_payload_op ? l1mesh_req_ready : 1'b1;
    wire l1mesh_auto_start = engine_payload_op && l1mgr_resp_valid;
    wire l1mesh_req_valid = l1mesh_start || l1mesh_auto_start;
    wire direct_l1mesh_req = (op_class_q == OP_CONV) || (op_class_q == OP_L1MESH);

    wire command_start = cmd_start_pending;
    wire conv_start = engine_start_pending && (op_class_q == OP_CONV);
    wire requant_start = engine_start_pending && (op_class_q == OP_REQUANT);
    wire ewe_start = engine_start_pending && (op_class_q == OP_EWE);
    wire pool_start = engine_start_pending && (op_class_q == OP_POOL);
    wire tnps_start = engine_start_pending && (op_class_q == OP_TNPS);
    wire udma_start = engine_start_pending && (op_class_q == OP_UDMA);
    wire l1mgr_start = l1mgr_start_pending;
    wire l1mesh_start = l1mesh_start_pending;
    wire udma_l1_req_valid = udma_engine_l1_req_valid;
    wire requant_l1_req_valid = requant_engine_l1_req_valid;
    wire ewe_l1_req_valid = ewe_engine_l1_req_valid;
    wire pool_l1_req_valid = pool_engine_l1_req_valid;
    wire tnps_l1_req_valid = tnps_engine_l1_req_valid;
    wire legacy_l1_req_valid = l1mgr_start &&
        (op_class_q != OP_UDMA) &&
        (op_class_q != OP_REQUANT) &&
        (op_class_q != OP_EWE) &&
        (op_class_q != OP_POOL) &&
        (op_class_q != OP_TNPS);

    reg selected_start_ready;
    reg selected_done_valid;
    reg selected_l1mgr_ready;

    assign desc_ready = (state == ST_IDLE);
    assign busy = (state != ST_IDLE);
    assign active_op_class = op_class_q;
    assign block_busy = {
        l1mesh_busy,
        l1mgr_busy,
        udma_busy,
        tnps_busy,
        pool_busy,
        ewe_busy,
        requant_busy,
        conv_busy,
        cmd_busy
    };
    assign block_done_valid = {
        l1mesh_resp_valid,
        l1mgr_resp_valid,
        udma_done_valid,
        tnps_done_valid,
        pool_done_valid,
        ewe_done_valid,
        requant_done_valid,
        conv_done_valid,
        cmd_done_valid
    };

    always @* begin
        selected_start_ready = 1'b1;
        selected_done_valid = 1'b1;
        selected_l1mgr_ready = l1mgr_req_ready;
        case (op_class_q)
            OP_CONV: begin
                selected_start_ready = conv_start_ready;
                selected_done_valid = conv_done_valid;
            end
            OP_REQUANT: begin
                selected_start_ready = requant_start_ready;
                selected_done_valid = requant_done_valid;
                selected_l1mgr_ready = requant_l1_req_ready;
            end
            OP_EWE: begin
                selected_start_ready = ewe_start_ready;
                selected_done_valid = ewe_done_valid;
                selected_l1mgr_ready = ewe_l1_req_ready;
            end
            OP_POOL: begin
                selected_start_ready = pool_start_ready;
                selected_done_valid = pool_done_valid;
                selected_l1mgr_ready = pool_l1_req_ready;
            end
            OP_TNPS: begin
                selected_start_ready = tnps_start_ready;
                selected_done_valid = tnps_done_valid;
                selected_l1mgr_ready = tnps_l1_req_ready;
            end
            OP_UDMA: begin
                selected_start_ready = udma_start_ready;
                selected_done_valid = udma_done_valid;
                selected_l1mgr_ready = udma_l1_req_ready;
            end
            OP_L1MANAGER: begin
                selected_start_ready = l1mgr_req_ready;
                selected_done_valid = l1mgr_resp_valid;
                selected_l1mgr_ready = l1mgr_req_ready;
            end
            OP_L1MESH: begin
                selected_start_ready = l1mesh_req_ready;
                selected_done_valid = l1mesh_resp_valid;
            end
            default: begin
                selected_start_ready = 1'b1;
                selected_done_valid = 1'b1;
                selected_l1mgr_ready = l1mgr_req_ready;
            end
        endcase
    end

    always @* begin
        active_phase_id = 4'd0;
        active_remaining_cycles = 32'd0;
        datapath_crc = 32'h811c9dc5;
        datapath_ok = 1'b1;
        case (op_class_q)
            OP_CONV: begin
                datapath_crc = conv_datapath_crc;
                datapath_ok = conv_datapath_ok;
            end
            OP_REQUANT: begin
                datapath_crc = requant_datapath_crc;
                datapath_ok = requant_datapath_ok;
            end
            OP_EWE: begin
                datapath_crc = ewe_datapath_crc;
                datapath_ok = ewe_datapath_ok;
            end
            OP_POOL: begin
                datapath_crc = pool_datapath_crc;
                datapath_ok = pool_datapath_ok;
            end
            OP_TNPS: begin
                datapath_crc = tnps_datapath_crc;
                datapath_ok = tnps_datapath_ok;
            end
            OP_UDMA: begin
                datapath_crc = udma_datapath_crc;
                datapath_ok = udma_datapath_ok;
            end
            default: begin
                datapath_crc = 32'h811c9dc5;
                datapath_ok = 1'b1;
            end
        endcase
        if (state == ST_CMD) begin
            active_phase_id = cmd_phase_id;
            active_remaining_cycles = cmd_remaining_cycles;
        end else if (state == ST_L1MGR) begin
            active_phase_id = l1mgr_phase_id;
            active_remaining_cycles = l1mgr_remaining_cycles;
        end else if (state == ST_L1MESH) begin
            active_phase_id = l1mesh_phase_id;
            active_remaining_cycles = l1mesh_remaining_cycles;
        end else if (state == ST_ENGINE) begin
            case (op_class_q)
                OP_CONV: begin
                    active_phase_id = conv_phase_id;
                    active_remaining_cycles = conv_remaining_cycles;
                    datapath_crc = conv_datapath_crc;
                    datapath_ok = conv_datapath_ok;
                end
                OP_REQUANT: begin
                    active_phase_id = requant_phase_id;
                    active_remaining_cycles = requant_remaining_cycles;
                    datapath_crc = requant_datapath_crc;
                    datapath_ok = requant_datapath_ok;
                end
                OP_EWE: begin
                    active_phase_id = ewe_phase_id;
                    active_remaining_cycles = ewe_remaining_cycles;
                    datapath_crc = ewe_datapath_crc;
                    datapath_ok = ewe_datapath_ok;
                end
                OP_POOL: begin
                    active_phase_id = pool_phase_id;
                    active_remaining_cycles = pool_remaining_cycles;
                    datapath_crc = pool_datapath_crc;
                    datapath_ok = pool_datapath_ok;
                end
                OP_TNPS: begin
                    active_phase_id = tnps_phase_id;
                    active_remaining_cycles = tnps_remaining_cycles;
                    datapath_crc = tnps_datapath_crc;
                    datapath_ok = tnps_datapath_ok;
                end
                OP_UDMA: begin
                    active_phase_id = udma_phase_id;
                    active_remaining_cycles = udma_remaining_cycles;
                    datapath_crc = udma_datapath_crc;
                    datapath_ok = udma_datapath_ok;
                end
                OP_L1MANAGER: begin
                    active_phase_id = l1mgr_phase_id;
                    active_remaining_cycles = l1mgr_remaining_cycles;
                end
                OP_L1MESH: begin
                    active_phase_id = l1mesh_phase_id;
                    active_remaining_cycles = l1mesh_remaining_cycles;
                end
                default: begin
                    active_phase_id = 4'd0;
                    active_remaining_cycles = 32'd0;
                end
            endcase
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done_valid <= 1'b0;
            cmd_start_pending <= 1'b0;
            l1mgr_start_pending <= 1'b0;
            l1mesh_start_pending <= 1'b0;
            engine_start_pending <= 1'b0;
            engine_done_seen <= 1'b0;
            op_class_q <= OP_COMMAND_ONLY;
            wait_count_q <= 8'd0;
            layer_id_q <= 16'd0;
            microblock_id_q <= 16'd0;
            stream_slot_q <= 8'd0;
            stream_meta_flags_q <= 8'd0;
            cfg_write_cycles_q <= 32'd0;
            bytes_q <= 32'd0;
            read_bytes_q <= 32'd0;
            write_bytes_q <= 32'd0;
            act_bytes_q <= 32'd0;
            wgt_bytes_q <= 32'd0;
            in_elems_q <= 32'd0;
            out_elems_q <= 32'd0;
            total_elems_q <= 32'd0;
            lanes_q <= 32'd0;
            window_q <= 32'd0;
            mac_cycles_q <= 32'd0;
            compute_cycles_q <= 32'd0;
            fill_cycles_q <= 32'd0;
            skip_l1_write_q <= 1'b0;
            layer_index_q <= 32'd0;
            ref_off_q <= 32'd0;
            ref_size_q <= 32'd0;
            udma_direction_write_q <= 1'b0;
            udma_dram_read_bytes_q <= 32'd0;
            udma_codec_cycles_q <= 32'd0;
            l1_req_write_q <= 1'b0;
            l1mgr_req_l1_q <= 1'b1;
            l1mgr_req_source_q <= 4'd0;
            l1mgr_req_tid_q <= 8'd0;
            l1mgr_payload_cycles_q <= 32'd0;
            l1mesh_addr_q <= {ADDR_WIDTH{1'b0}};
            l1mesh_route_cycles_q <= 32'd0;
            l1mesh_wdata_q <= {DATA_WIDTH{1'b0}};
            l1mesh_wstrb_q <= {DATA_WIDTH/8{1'b0}};
        end else begin
            case (state)
                ST_IDLE: begin
                    done_valid <= 1'b0;
                    if (desc_valid && desc_ready) begin
                        op_class_q <= desc_op_class;
                        wait_count_q <= desc_wait_count;
                        layer_id_q <= desc_layer_id;
                        microblock_id_q <= desc_microblock_id;
                        stream_slot_q <= desc_stream_slot;
                        stream_meta_flags_q <= desc_stream_meta_flags;
                        cfg_write_cycles_q <= cfg_write_cycles;
                        bytes_q <= bytes;
                        read_bytes_q <= read_bytes;
                        write_bytes_q <= write_bytes;
                        act_bytes_q <= act_bytes;
                        wgt_bytes_q <= wgt_bytes;
                        in_elems_q <= in_elems;
                        out_elems_q <= out_elems;
                        total_elems_q <= total_elems;
                        lanes_q <= lanes;
                        window_q <= window;
                        mac_cycles_q <= mac_cycles;
                        compute_cycles_q <= compute_cycles;
                        fill_cycles_q <= fill_cycles;
                        skip_l1_write_q <= skip_l1_write;
                        layer_index_q <= layer_index;
                        ref_off_q <= ref_off;
                        ref_size_q <= ref_size;
                        udma_direction_write_q <= udma_direction_write;
                        udma_dram_read_bytes_q <= udma_dram_read_bytes;
                        udma_codec_cycles_q <= udma_codec_cycles;
                        l1_req_write_q <= l1_req_write;
                        l1mgr_req_l1_q <= l1mgr_req_l1;
                        l1mgr_req_source_q <= l1mgr_req_source;
                        l1mgr_req_tid_q <= l1mgr_req_tid;
                        l1mgr_payload_cycles_q <= l1mgr_payload_cycles;
                        l1mesh_addr_q <= l1mesh_addr;
                        l1mesh_route_cycles_q <= l1mesh_route_cycles;
                        l1mesh_wdata_q <= l1mesh_wdata;
                        l1mesh_wstrb_q <= l1mesh_wstrb;
                        cmd_start_pending <= 1'b1;
                        l1mgr_start_pending <= 1'b0;
                        l1mesh_start_pending <= 1'b0;
                        engine_start_pending <= 1'b0;
                        engine_done_seen <= 1'b0;
                        state <= ST_CMD;
                    end
                end
                ST_CMD: begin
                    if (cmd_start_pending && cmd_start_ready)
                        cmd_start_pending <= 1'b0;
                    if (cmd_done_valid) begin
                        if (op_class_q == OP_COMMAND_ONLY) begin
                            done_valid <= 1'b1;
                            state <= ST_DONE;
                        end else if ((op_class_q == OP_CONV) || (op_class_q == OP_L1MESH)) begin
                            l1mesh_start_pending <= 1'b1;
                            state <= ST_L1MESH;
                        end else if (op_class_q == OP_L1MANAGER) begin
                            l1mgr_start_pending <= 1'b1;
                            state <= ST_L1MGR;
                        end else begin
                            engine_start_pending <= 1'b1;
                            state <= ST_ENGINE;
                        end
                    end
                end
                ST_L1MGR: begin
                    if (l1mgr_start_pending && selected_l1mgr_ready)
                        l1mgr_start_pending <= 1'b0;
                    if (!l1mgr_start_pending && l1mgr_resp_valid) begin
                        if (op_class_q == OP_L1MANAGER) begin
                            done_valid <= 1'b1;
                            state <= ST_DONE;
                        end else begin
                            l1mesh_start_pending <= 1'b1;
                            state <= ST_L1MESH;
                        end
                    end
                end
                ST_L1MESH: begin
                    if (l1mesh_start_pending && l1mesh_req_ready)
                        l1mesh_start_pending <= 1'b0;
                    if (!l1mesh_start_pending && l1mesh_resp_valid) begin
                        if (op_class_q == OP_L1MESH) begin
                            done_valid <= 1'b1;
                            state <= ST_DONE;
                        end else begin
                            engine_start_pending <= 1'b1;
                            state <= ST_ENGINE;
                        end
                    end
                end
                ST_ENGINE: begin
                    if (engine_start_pending && selected_start_ready)
                        engine_start_pending <= 1'b0;
                    if (selected_done_valid)
                        engine_done_seen <= 1'b1;
                    if (!engine_start_pending &&
                        (engine_done_seen || selected_done_valid) &&
                        engine_l1_drained) begin
                        done_valid <= 1'b1;
                        engine_done_seen <= 1'b0;
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (done_valid && done_ready) begin
                        done_valid <= 1'b0;
                        state <= ST_IDLE;
                    end
                end
                default: begin
                    state <= ST_IDLE;
                    done_valid <= 1'b0;
                    cmd_start_pending <= 1'b0;
                    l1mgr_start_pending <= 1'b0;
                    l1mesh_start_pending <= 1'b0;
                    engine_start_pending <= 1'b0;
                    engine_done_seen <= 1'b0;
                end
            endcase
        end
    end

    command u_command (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(command_start),
        .start_ready(cmd_start_ready),
        .cfg_write_cycles(cfg_write_cycles_q),
        .wait_count(wait_count_q),
        .op_class({4'd0, op_class_q}),
        .busy(cmd_busy),
        .done_valid(cmd_done_valid),
        .done_ready(1'b1),
        .phase_id(cmd_phase_id),
        .remaining_cycles(cmd_remaining_cycles),
        .debug_wait_count(cmd_debug_wait_count),
        .debug_op_class(cmd_debug_op_class)
    );

    conv u_conv (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(conv_start),
        .start_ready(conv_start_ready),
        .act_bytes(act_bytes_q),
        .wgt_bytes(wgt_bytes_q),
        .out_elems(out_elems_q),
        .mac_cycles(mac_cycles_q),
        .fill_cycles(fill_cycles_q),
        .layer_index(layer_index_q),
        .ref_off(ref_off_q),
        .ref_size(ref_size_q),
        .busy(conv_busy),
        .done_valid(conv_done_valid),
        .done_ready(1'b1),
        .phase_id(conv_phase_id),
        .remaining_cycles(conv_remaining_cycles),
        .datapath_crc(conv_datapath_crc),
        .datapath_ok(conv_datapath_ok)
    );

    requant u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(requant_start),
        .start_ready(requant_start_ready),
        .read_bytes(read_bytes_q),
        .total_elems(total_elems_q),
        .write_bytes(write_bytes_q),
        .skip_l1_write(skip_l1_write_q),
        .layer_index(layer_index_q),
        .ref_off(ref_off_q),
        .ref_size(ref_size_q),
        .l1_req_valid(requant_engine_l1_req_valid),
        .l1_req_ready(requant_l1_req_ready),
        .l1_req_write(requant_engine_l1_req_write),
        .l1_req_bytes(requant_engine_l1_req_bytes),
        .l1_req_payload_cycles(requant_engine_l1_req_payload_cycles),
        .busy(requant_busy),
        .done_valid(requant_done_valid),
        .done_ready(1'b1),
        .phase_id(requant_phase_id),
        .remaining_cycles(requant_remaining_cycles),
        .datapath_crc(requant_datapath_crc),
        .datapath_ok(requant_datapath_ok)
    );

    ewe u_ewe (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(ewe_start),
        .start_ready(ewe_start_ready),
        .read_bytes(read_bytes_q),
        .compute_cycles(compute_cycles_q),
        .write_bytes(write_bytes_q),
        .layer_index(layer_index_q),
        .ref_off(ref_off_q),
        .ref_size(ref_size_q),
        .l1_req_valid(ewe_engine_l1_req_valid),
        .l1_req_ready(ewe_l1_req_ready),
        .l1_req_write(ewe_engine_l1_req_write),
        .l1_req_bytes(ewe_engine_l1_req_bytes),
        .l1_req_payload_cycles(ewe_engine_l1_req_payload_cycles),
        .busy(ewe_busy),
        .done_valid(ewe_done_valid),
        .done_ready(1'b1),
        .phase_id(ewe_phase_id),
        .remaining_cycles(ewe_remaining_cycles),
        .datapath_crc(ewe_datapath_crc),
        .datapath_ok(ewe_datapath_ok)
    );

    pool u_pool (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(pool_start),
        .start_ready(pool_start_ready),
        .in_elems(in_elems_q),
        .out_elems(out_elems_q),
        .lanes(lanes_q),
        .window(window_q),
        .layer_index(layer_index_q),
        .ref_off(ref_off_q),
        .ref_size(ref_size_q),
        .l1_req_valid(pool_engine_l1_req_valid),
        .l1_req_ready(pool_l1_req_ready),
        .l1_req_write(pool_engine_l1_req_write),
        .l1_req_bytes(pool_engine_l1_req_bytes),
        .l1_req_payload_cycles(pool_engine_l1_req_payload_cycles),
        .busy(pool_busy),
        .done_valid(pool_done_valid),
        .done_ready(1'b1),
        .phase_id(pool_phase_id),
        .remaining_cycles(pool_remaining_cycles),
        .datapath_crc(pool_datapath_crc),
        .datapath_ok(pool_datapath_ok)
    );

    tnps u_tnps (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(tnps_start),
        .start_ready(tnps_start_ready),
        .bytes(bytes_q),
        .layer_index(layer_index_q),
        .ref_off(ref_off_q),
        .ref_size(ref_size_q),
        .l1_req_valid(tnps_engine_l1_req_valid),
        .l1_req_ready(tnps_l1_req_ready),
        .l1_req_write(tnps_engine_l1_req_write),
        .l1_req_bytes(tnps_engine_l1_req_bytes),
        .l1_req_payload_cycles(tnps_engine_l1_req_payload_cycles),
        .busy(tnps_busy),
        .done_valid(tnps_done_valid),
        .done_ready(1'b1),
        .phase_id(tnps_phase_id),
        .remaining_cycles(tnps_remaining_cycles),
        .datapath_crc(tnps_datapath_crc),
        .datapath_ok(tnps_datapath_ok)
    );

    udma u_udma (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(udma_start),
        .start_ready(udma_start_ready),
        .direction_write(udma_direction_write_q),
        .bytes(bytes_q),
        .dram_read_bytes(udma_dram_read_bytes_q),
        .codec_cycles(udma_codec_cycles_q),
        .layer_index(layer_index_q),
        .ref_off(ref_off_q),
        .ref_size(ref_size_q),
        .l1_req_valid(udma_engine_l1_req_valid),
        .l1_req_ready(udma_l1_req_ready),
        .l1_req_write(udma_engine_l1_req_write),
        .l1_req_bytes(udma_engine_l1_req_bytes),
        .l1_req_payload_cycles(udma_engine_l1_req_payload_cycles),
        .busy(udma_busy),
        .done_valid(udma_done_valid),
        .done_ready(1'b1),
        .phase_id(udma_phase_id),
        .remaining_cycles(udma_remaining_cycles),
        .datapath_crc(udma_datapath_crc),
        .datapath_ok(udma_datapath_ok)
    );

    l1manager u_l1manager (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(legacy_l1_req_valid),
        .req_ready(l1mgr_req_ready),
        .req_write(l1_req_write_q),
        .req_l1(l1mgr_req_l1_q),
        .req_source(l1mgr_req_source_q),
        .req_tid(l1mgr_req_tid_q),
        .req_bytes(bytes_q),
        .req_payload_cycles(l1mgr_payload_cycles_q),
        .req_addr(l1mesh_addr_q),
        .req_wdata(l1mesh_wdata_q),
        .req_wstrb(l1mesh_wstrb_q),
        .udma_req_valid(udma_l1_req_valid),
        .udma_req_ready(udma_l1_req_ready),
        .udma_req_write(udma_engine_l1_req_write),
        .udma_req_tid(l1mgr_req_tid_q),
        .udma_req_bytes(udma_engine_l1_req_bytes),
        .udma_req_payload_cycles(udma_engine_l1_req_payload_cycles),
        .udma_req_addr(l1mesh_addr_q),
        .udma_req_wdata(l1mesh_wdata_q),
        .udma_req_wstrb(l1mesh_wstrb_q),
        .requant_req_valid(requant_l1_req_valid),
        .requant_req_ready(requant_l1_req_ready),
        .requant_req_write(requant_engine_l1_req_write),
        .requant_req_tid(l1mgr_req_tid_q),
        .requant_req_bytes(requant_engine_l1_req_bytes),
        .requant_req_payload_cycles(requant_engine_l1_req_payload_cycles),
        .requant_req_addr(l1mesh_addr_q),
        .requant_req_wdata(l1mesh_wdata_q),
        .requant_req_wstrb(l1mesh_wstrb_q),
        .ewe_req_valid(ewe_l1_req_valid),
        .ewe_req_ready(ewe_l1_req_ready),
        .ewe_req_write(ewe_engine_l1_req_write),
        .ewe_req_tid(l1mgr_req_tid_q),
        .ewe_req_bytes(ewe_engine_l1_req_bytes),
        .ewe_req_payload_cycles(ewe_engine_l1_req_payload_cycles),
        .ewe_req_addr(l1mesh_addr_q),
        .ewe_req_wdata(l1mesh_wdata_q),
        .ewe_req_wstrb(l1mesh_wstrb_q),
        .pool_req_valid(pool_l1_req_valid),
        .pool_req_ready(pool_l1_req_ready),
        .pool_req_write(pool_engine_l1_req_write),
        .pool_req_tid(l1mgr_req_tid_q),
        .pool_req_bytes(pool_engine_l1_req_bytes),
        .pool_req_payload_cycles(pool_engine_l1_req_payload_cycles),
        .pool_req_addr(l1mesh_addr_q),
        .pool_req_wdata(l1mesh_wdata_q),
        .pool_req_wstrb(l1mesh_wstrb_q),
        .tnps_req_valid(tnps_l1_req_valid),
        .tnps_req_ready(tnps_l1_req_ready),
        .tnps_req_write(tnps_engine_l1_req_write),
        .tnps_req_tid(l1mgr_req_tid_q),
        .tnps_req_bytes(tnps_engine_l1_req_bytes),
        .tnps_req_payload_cycles(tnps_engine_l1_req_payload_cycles),
        .tnps_req_addr(l1mesh_addr_q),
        .tnps_req_wdata(l1mesh_wdata_q),
        .tnps_req_wstrb(l1mesh_wstrb_q),
        .mesh_req_write(l1mgr_mesh_req_write),
        .mesh_req_addr(l1mgr_mesh_req_addr),
        .mesh_req_bytes(l1mgr_mesh_req_bytes),
        .mesh_req_wdata(l1mgr_mesh_req_wdata),
        .mesh_req_wstrb(l1mgr_mesh_req_wstrb),
        .resp_valid(l1mgr_resp_valid),
        .resp_ready(l1mgr_resp_ready),
        .busy(l1mgr_busy),
        .phase_id(l1mgr_phase_id),
        .remaining_cycles(l1mgr_remaining_cycles),
        .debug_source(l1mgr_debug_source),
        .debug_tid(l1mgr_debug_tid)
    );

    l1mesh u_l1mesh (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(l1mesh_req_valid),
        .req_ready(l1mesh_req_ready),
        .req_write(direct_l1mesh_req ? l1_req_write_q : l1mgr_mesh_req_write),
        .req_addr(direct_l1mesh_req ? l1mesh_addr_q : l1mgr_mesh_req_addr),
        .req_bytes(direct_l1mesh_req ? bytes_q : l1mgr_mesh_req_bytes),
        .route_cycles(l1mesh_route_cycles_q),
        .req_wdata(direct_l1mesh_req ? l1mesh_wdata_q : l1mgr_mesh_req_wdata),
        .req_wstrb(direct_l1mesh_req ? l1mesh_wstrb_q : l1mgr_mesh_req_wstrb),
        .debug_crc_start(1'b0),
        .debug_crc_addr({ADDR_WIDTH{1'b0}}),
        .debug_crc_count(32'd0),
        .debug_crc_busy(l1mesh_debug_crc_busy),
        .debug_crc_done(l1mesh_debug_crc_done),
        .debug_crc(l1mesh_debug_crc),
        .debug_crc_byte_count(l1mesh_debug_crc_byte_count),
        .resp_valid(l1mesh_resp_valid),
        .resp_ready(1'b1),
        .resp_rdata(l1mesh_rdata),
        .busy(l1mesh_busy),
        .phase_id(l1mesh_phase_id),
        .remaining_cycles(l1mesh_remaining_cycles)
    );
endmodule
