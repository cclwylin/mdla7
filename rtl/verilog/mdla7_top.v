`timescale 1ns/1ps

`include "common.v"

module vf_microblock_control (
    input             clk,
    input             rst_n,
    input             accept,
    input             complete,
    input      [3:0]  desc_op_class,
    input      [15:0] desc_layer_id,
    input      [15:0] desc_microblock_id,
    input      [7:0]  desc_stream_slot,
    input      [7:0]  desc_stream_meta_flags,
    output reg        active_valid,
    output reg [3:0]  active_op_class,
    output reg [15:0] active_layer_id,
    output reg [15:0] active_microblock_id,
    output reg [7:0]  active_stream_slot,
    output reg [7:0]  active_stream_meta_flags,
    output reg [31:0] load_count,
    output reg [31:0] compute_count,
    output reg [31:0] store_count,
    output reg [31:0] final_count
);
    localparam [7:0] SMF_LOAD_A = 8'h01;
    localparam [7:0] SMF_LOAD_B = 8'h02;
    localparam [7:0] SMF_COMPUTE = 8'h04;
    localparam [7:0] SMF_STORE = 8'h08;
    localparam [7:0] SMF_FINAL_TILE = 8'h10;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_valid <= 1'b0;
            active_op_class <= 4'd0;
            active_layer_id <= 16'd0;
            active_microblock_id <= 16'd0;
            active_stream_slot <= 8'd0;
            active_stream_meta_flags <= 8'd0;
            load_count <= 32'd0;
            compute_count <= 32'd0;
            store_count <= 32'd0;
            final_count <= 32'd0;
        end else if (accept) begin
            active_valid <= 1'b1;
            active_op_class <= desc_op_class;
            active_layer_id <= desc_layer_id;
            active_microblock_id <= desc_microblock_id;
            active_stream_slot <= desc_stream_slot;
            active_stream_meta_flags <= desc_stream_meta_flags;
            if ((desc_stream_meta_flags & (SMF_LOAD_A | SMF_LOAD_B)) != 8'd0)
                load_count <= load_count + 32'd1;
            if ((desc_stream_meta_flags & SMF_COMPUTE) != 8'd0)
                compute_count <= compute_count + 32'd1;
            if ((desc_stream_meta_flags & SMF_STORE) != 8'd0)
                store_count <= store_count + 32'd1;
            if ((desc_stream_meta_flags & SMF_FINAL_TILE) != 8'd0)
                final_count <= final_count + 32'd1;
        end else if (complete) begin
            active_valid <= 1'b0;
        end
    end
endmodule

module vf_mb_stream_scheduler #(
    parameter QUEUE_DEPTH = 32
) (
    input             clk,
    input             rst_n,
    input             desc_valid,
    output            desc_ready,
    input      [3:0]  desc_op_class,
    input      [15:0] desc_layer_id,
    input      [15:0] desc_microblock_id,
    input      [7:0]  desc_stream_slot,
    input      [7:0]  desc_stream_meta_flags,
    input      [31:0] bytes,
    input      [31:0] conv_workload_bytes,
    input      [31:0] conv_workload_outputs,
    input      [31:0] pool_workload_bytes,
    input      [31:0] tnps_bytes,
    output reg        done_valid,
    input             done_ready,
    output            busy,
    output reg [3:0]  active_op_class,
    output reg [15:0] active_layer_id,
    output reg [15:0] active_microblock_id,
    output reg [7:0]  active_stream_slot,
    output reg [7:0]  active_stream_meta_flags,
    output reg [3:0]  active_phase_id,
    output reg [31:0] active_remaining_cycles,
    output reg [31:0] load_count,
    output reg [31:0] compute_count,
    output reg [31:0] store_count,
    output reg [31:0] final_count
);
    localparam [3:0] OP_CONV = 4'd1;
    localparam [3:0] OP_REQUANT = 4'd2;
    localparam [3:0] OP_EWE = 4'd3;
    localparam [3:0] OP_POOL = 4'd4;
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;
    localparam [7:0] SMF_LOAD_A = 8'h01;
    localparam [7:0] SMF_LOAD_B = 8'h02;
    localparam [7:0] SMF_COMPUTE = 8'h04;
    localparam [7:0] SMF_STORE = 8'h08;
    localparam [7:0] SMF_FINAL_TILE = 8'h10;
    localparam [3:0] PH_LOAD = 4'd2;
    localparam [3:0] PH_COMPUTE = 4'd4;
    localparam [3:0] PH_REQUANT = 4'd6;
    localparam [3:0] PH_STORE = 4'd7;
    localparam [5:0] QUEUE_DEPTH_W = QUEUE_DEPTH[5:0];

    reg [31:0] load_latency_q [0:QUEUE_DEPTH-1];
    reg [15:0] load_layer_q [0:QUEUE_DEPTH-1];
    reg [15:0] load_mb_q [0:QUEUE_DEPTH-1];
    reg [7:0] load_slot_q [0:QUEUE_DEPTH-1];
    reg [7:0] load_flags_q [0:QUEUE_DEPTH-1];
    reg [3:0] load_op_q [0:QUEUE_DEPTH-1];
    reg [5:0] load_head;
    reg [5:0] load_tail;
    reg [5:0] load_level;

    reg [31:0] compute_latency_q [0:QUEUE_DEPTH-1];
    reg [7:0] compute_dep_q [0:QUEUE_DEPTH-1];
    reg [15:0] compute_layer_q [0:QUEUE_DEPTH-1];
    reg [15:0] compute_mb_q [0:QUEUE_DEPTH-1];
    reg [7:0] compute_slot_q [0:QUEUE_DEPTH-1];
    reg [7:0] compute_flags_q [0:QUEUE_DEPTH-1];
    reg [3:0] compute_op_q [0:QUEUE_DEPTH-1];
    reg [15:0] compute_dep_load_q [0:QUEUE_DEPTH-1];
    reg [15:0] compute_dep_final_q [0:QUEUE_DEPTH-1];
    reg [5:0] compute_head;
    reg [5:0] compute_tail;
    reg [5:0] compute_level;

    reg [31:0] requant_latency_q [0:QUEUE_DEPTH-1];
    reg [15:0] requant_dep_compute_q [0:QUEUE_DEPTH-1];
    reg [15:0] requant_layer_q [0:QUEUE_DEPTH-1];
    reg [15:0] requant_mb_q [0:QUEUE_DEPTH-1];
    reg [7:0] requant_slot_q [0:QUEUE_DEPTH-1];
    reg [7:0] requant_flags_q [0:QUEUE_DEPTH-1];
    reg [3:0] requant_op_q [0:QUEUE_DEPTH-1];
    reg [5:0] requant_head;
    reg [5:0] requant_tail;
    reg [5:0] requant_level;

    reg [31:0] store_latency_q [0:QUEUE_DEPTH-1];
    reg [7:0] store_dep_q [0:QUEUE_DEPTH-1];
    reg [15:0] store_dep_compute_q [0:QUEUE_DEPTH-1];
    reg [15:0] store_dep_requant_q [0:QUEUE_DEPTH-1];
    reg [15:0] store_dep_final_q [0:QUEUE_DEPTH-1];
    reg [15:0] store_layer_q [0:QUEUE_DEPTH-1];
    reg [15:0] store_mb_q [0:QUEUE_DEPTH-1];
    reg [7:0] store_slot_q [0:QUEUE_DEPTH-1];
    reg [7:0] store_flags_q [0:QUEUE_DEPTH-1];
    reg [3:0] store_op_q [0:QUEUE_DEPTH-1];
    reg [5:0] store_head;
    reg [5:0] store_tail;
    reg [5:0] store_level;

    reg load_busy;
    reg compute_busy;
    reg requant_busy;
    reg store_busy;
    reg [31:0] load_remaining;
    reg [31:0] compute_remaining;
    reg [31:0] requant_remaining;
    reg [31:0] store_remaining;
    reg [3:0] load_op;
    reg [3:0] compute_op;
    reg [3:0] requant_op;
    reg [3:0] store_op;
    reg [15:0] load_layer;
    reg [15:0] compute_layer;
    reg [15:0] requant_layer;
    reg [15:0] store_layer;
    reg [15:0] load_mb;
    reg [15:0] compute_mb;
    reg [15:0] requant_mb;
    reg [15:0] store_mb;
    reg [31:0] compute_requant_latency;
    reg [7:0] load_slot;
    reg [7:0] compute_slot;
    reg [7:0] requant_slot;
    reg [7:0] store_slot;
    reg [7:0] load_flags;
    reg [7:0] compute_flags;
    reg [7:0] requant_flags;
    reg [7:0] store_flags;
    reg [15:0] accepted_loads;
    reg [15:0] retired_loads;
    reg [15:0] accepted_computes;
    reg [15:0] retired_computes;
    reg [15:0] accepted_requants;
    reg [15:0] retired_requants;
    reg [15:0] accepted_finals;
    reg [15:0] retired_finals;
    reg [15:0] conv_pending_requant;
    reg current_layer_valid;
    reg [15:0] current_layer;

    wire is_load_desc = (desc_stream_meta_flags & (SMF_LOAD_A | SMF_LOAD_B)) != 8'd0;
    wire is_compute_desc = (desc_stream_meta_flags & SMF_COMPUTE) != 8'd0;
    wire is_store_desc = (desc_stream_meta_flags & SMF_STORE) != 8'd0;
    wire is_final_desc = (desc_stream_meta_flags & SMF_FINAL_TILE) != 8'd0;
    wire is_conv_compute_desc = is_compute_desc && (desc_op_class == OP_CONV);
    wire accept = desc_valid && desc_ready;
    wire load_done = load_busy && (load_remaining <= 32'd1);
    wire compute_done = compute_busy && (compute_remaining <= 32'd1);
    wire requant_done = requant_busy && (requant_remaining <= 32'd1);
    wire store_done = store_busy && (store_remaining <= 32'd1);
    wire done_slot_ready = !done_valid || done_ready;
    wire [15:0] layer_boundary_dep =
        (current_layer_valid && (desc_layer_id != current_layer)) ? accepted_finals : retired_finals;
    wire [15:0] next_compute_dep =
        accepted_computes + (accept && is_compute_desc ? 16'd1 : 16'd0);
    wire [15:0] next_requant_dep =
        accepted_requants + conv_pending_requant +
        (accept && is_conv_compute_desc ? 16'd1 : 16'd0);

    assign desc_ready = (!done_valid || done_ready) &&
        !load_done && !compute_done && !requant_done && !store_done &&
        (!is_load_desc || (load_level < QUEUE_DEPTH_W)) &&
        (!is_compute_desc || ((compute_level < QUEUE_DEPTH_W) &&
                              (!is_conv_compute_desc ||
                               ((requant_level + conv_pending_requant) < QUEUE_DEPTH_W)))) &&
        (!is_store_desc || (store_level < QUEUE_DEPTH_W));
    assign busy = load_busy || compute_busy || requant_busy || store_busy ||
        (load_level != 6'd0) || (compute_level != 6'd0) ||
        (requant_level != 6'd0) || (store_level != 6'd0) ||
        done_valid;

    function [5:0] bump;
        input [5:0] idx;
        begin
            bump = (idx == (QUEUE_DEPTH - 1)) ? 6'd0 : (idx + 6'd1);
        end
    endfunction

    function [31:0] ceil_div;
        input [31:0] value;
        input [31:0] denom;
        begin
            ceil_div = (denom == 32'd0) ? 32'd0 : ((value + denom - 32'd1) / denom);
        end
    endfunction

    function [31:0] load_store_cycles;
        input [31:0] byte_count;
        begin
            load_store_cycles = ceil_div((byte_count == 32'd0) ? 32'd1 : byte_count, 32'd256) +
                ceil_div((byte_count == 32'd0) ? 32'd1 : byte_count, 32'd64) + 32'd12;
        end
    endfunction

    function [31:0] compute_cycles;
        input [3:0] op;
        input [31:0] byte_count;
        input [31:0] conv_bytes;
        input [31:0] conv_outputs;
        input [31:0] pool_bytes;
        input [31:0] tbytes;
        reg [31:0] safe_outputs;
        reg [31:0] safe_bytes;
        begin
            safe_outputs = (conv_outputs == 32'd0) ? 32'd1 : conv_outputs;
            safe_bytes = (conv_bytes == 32'd0) ? ((byte_count == 32'd0) ? 32'd1 : byte_count) : conv_bytes;
            case (op)
                OP_CONV: compute_cycles = ceil_div(safe_bytes * safe_outputs, 32'd16) + 32'd8;
                OP_POOL: compute_cycles = ceil_div((pool_bytes == 32'd0) ? byte_count : pool_bytes, 32'd128) + 32'd6;
                OP_TNPS: compute_cycles = ceil_div((tbytes == 32'd0) ? byte_count : tbytes, 32'd128) + 32'd6;
                OP_REQUANT: compute_cycles = ceil_div((byte_count == 32'd0) ? 32'd1 : byte_count, 32'd64) + 32'd5;
                OP_EWE: compute_cycles = ceil_div((byte_count == 32'd0) ? 32'd1 : byte_count, 32'd128) + 32'd5;
                default: compute_cycles = 32'd4;
            endcase
        end
    endfunction

    function [31:0] requant_cycles;
        input [31:0] conv_outputs;
        begin
            requant_cycles = ceil_div((conv_outputs == 32'd0) ? 32'd1 : conv_outputs, 32'd32) + 32'd5;
        end
    endfunction

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_valid <= 1'b0;
            load_head <= 6'd0;
            load_tail <= 6'd0;
            load_level <= 6'd0;
            compute_head <= 6'd0;
            compute_tail <= 6'd0;
            compute_level <= 6'd0;
            requant_head <= 6'd0;
            requant_tail <= 6'd0;
            requant_level <= 6'd0;
            store_head <= 6'd0;
            store_tail <= 6'd0;
            store_level <= 6'd0;
            load_busy <= 1'b0;
            compute_busy <= 1'b0;
            requant_busy <= 1'b0;
            store_busy <= 1'b0;
            load_remaining <= 32'd0;
            compute_remaining <= 32'd0;
            requant_remaining <= 32'd0;
            store_remaining <= 32'd0;
            accepted_loads <= 16'd0;
            retired_loads <= 16'd0;
            accepted_computes <= 16'd0;
            retired_computes <= 16'd0;
            accepted_requants <= 16'd0;
            retired_requants <= 16'd0;
            accepted_finals <= 16'd0;
            retired_finals <= 16'd0;
            conv_pending_requant <= 16'd0;
            current_layer_valid <= 1'b0;
            current_layer <= 16'd0;
            compute_requant_latency <= 32'd0;
            active_op_class <= 4'd0;
            active_layer_id <= 16'd0;
            active_microblock_id <= 16'd0;
            active_stream_slot <= 8'd0;
            active_stream_meta_flags <= 8'd0;
            active_phase_id <= 4'd0;
            active_remaining_cycles <= 32'd0;
            load_count <= 32'd0;
            compute_count <= 32'd0;
            store_count <= 32'd0;
            final_count <= 32'd0;
            for (i = 0; i < QUEUE_DEPTH; i = i + 1) begin
                compute_dep_q[i] <= 8'd0;
                compute_dep_load_q[i] <= 16'd0;
                compute_dep_final_q[i] <= 16'd0;
                store_dep_q[i] <= 8'd0;
                store_dep_compute_q[i] <= 16'd0;
                store_dep_requant_q[i] <= 16'd0;
                store_dep_final_q[i] <= 16'd0;
                requant_dep_compute_q[i] <= 16'd0;
            end
        end else begin
            if (done_valid && done_ready)
                done_valid <= 1'b0;

            if (accept) begin
                current_layer_valid <= 1'b1;
                current_layer <= desc_layer_id;
            end
            if (accept && is_load_desc) begin
                load_latency_q[load_tail] <= load_store_cycles(bytes);
                load_layer_q[load_tail] <= desc_layer_id;
                load_mb_q[load_tail] <= desc_microblock_id;
                load_slot_q[load_tail] <= desc_stream_slot;
                load_flags_q[load_tail] <= desc_stream_meta_flags;
                load_op_q[load_tail] <= desc_op_class;
                load_tail <= bump(load_tail);
                load_level <= load_level + 6'd1;
                load_count <= load_count + 32'd1;
                accepted_loads <= accepted_loads + 16'd1;
            end
            if (accept && is_compute_desc) begin
                compute_latency_q[compute_tail] <= compute_cycles(
                    desc_op_class, bytes, conv_workload_bytes, conv_workload_outputs,
                    pool_workload_bytes, tnps_bytes);
                compute_dep_q[compute_tail] <= {2'd0, load_level} + {7'd0, load_busy};
                compute_dep_load_q[compute_tail] <= accepted_loads;
                compute_dep_final_q[compute_tail] <= layer_boundary_dep;
                compute_layer_q[compute_tail] <= desc_layer_id;
                compute_mb_q[compute_tail] <= desc_microblock_id;
                compute_slot_q[compute_tail] <= desc_stream_slot;
                compute_flags_q[compute_tail] <= desc_stream_meta_flags;
                compute_op_q[compute_tail] <= desc_op_class;
                compute_tail <= bump(compute_tail);
                compute_level <= compute_level + 6'd1;
                compute_count <= compute_count + 32'd1;
                accepted_computes <= accepted_computes + 16'd1;
                if (is_conv_compute_desc)
                    conv_pending_requant <= conv_pending_requant + 16'd1;
            end
            if (accept && is_store_desc) begin
                store_latency_q[store_tail] <= load_store_cycles(bytes);
                store_dep_q[store_tail] <= {2'd0, compute_level} + {7'd0, compute_busy};
                store_dep_compute_q[store_tail] <= next_compute_dep;
                store_dep_requant_q[store_tail] <= next_requant_dep;
                store_dep_final_q[store_tail] <= layer_boundary_dep;
                store_layer_q[store_tail] <= desc_layer_id;
                store_mb_q[store_tail] <= desc_microblock_id;
                store_slot_q[store_tail] <= desc_stream_slot;
                store_flags_q[store_tail] <= desc_stream_meta_flags;
                store_op_q[store_tail] <= desc_op_class;
                store_tail <= bump(store_tail);
                store_level <= store_level + 6'd1;
                store_count <= store_count + 32'd1;
                if ((desc_stream_meta_flags & SMF_FINAL_TILE) != 8'd0)
                    final_count <= final_count + 32'd1;
                if (is_final_desc)
                    accepted_finals <= accepted_finals + 16'd1;
            end

            if (!accept && !load_busy && (load_level != 6'd0)) begin
                load_busy <= 1'b1;
                load_remaining <= load_latency_q[load_head];
                load_op <= load_op_q[load_head];
                load_layer <= load_layer_q[load_head];
                load_mb <= load_mb_q[load_head];
                load_slot <= load_slot_q[load_head];
                load_flags <= load_flags_q[load_head];
                load_head <= bump(load_head);
                load_level <= load_level - 6'd1;
            end else if (load_busy) begin
                if (load_done && done_slot_ready) begin
                    load_busy <= 1'b0;
                    done_valid <= 1'b1;
                    active_op_class <= load_op;
                    active_layer_id <= load_layer;
                    active_microblock_id <= load_mb;
                    active_stream_slot <= load_slot;
                    active_stream_meta_flags <= load_flags;
                    active_phase_id <= PH_LOAD;
                    active_remaining_cycles <= 32'd0;
                    retired_loads <= retired_loads + 16'd1;
                    for (i = 0; i < QUEUE_DEPTH; i = i + 1)
                        if (compute_dep_q[i] != 8'd0)
                            compute_dep_q[i] <= compute_dep_q[i] - 8'd1;
                end else begin
                    load_remaining <= load_remaining - 32'd1;
                end
            end

            if (!accept && !compute_busy && (compute_level != 6'd0) &&
                (compute_dep_q[compute_head] == 8'd0) &&
                (retired_loads >= compute_dep_load_q[compute_head]) &&
                (retired_finals >= compute_dep_final_q[compute_head])) begin
                compute_busy <= 1'b1;
                compute_remaining <= compute_latency_q[compute_head];
                compute_requant_latency <= requant_cycles(compute_latency_q[compute_head]);
                compute_op <= compute_op_q[compute_head];
                compute_layer <= compute_layer_q[compute_head];
                compute_mb <= compute_mb_q[compute_head];
                compute_slot <= compute_slot_q[compute_head];
                compute_flags <= compute_flags_q[compute_head];
                compute_head <= bump(compute_head);
                compute_level <= compute_level - 6'd1;
            end else if (compute_busy) begin
                if (compute_done && done_slot_ready && !load_done) begin
                    compute_busy <= 1'b0;
                    done_valid <= 1'b1;
                    active_op_class <= compute_op;
                    active_layer_id <= compute_layer;
                    active_microblock_id <= compute_mb;
                    active_stream_slot <= compute_slot;
                    active_stream_meta_flags <= compute_flags;
                    active_phase_id <= PH_COMPUTE;
                    active_remaining_cycles <= 32'd0;
                    retired_computes <= retired_computes + 16'd1;
                    if (compute_op == OP_CONV) begin
                        requant_latency_q[requant_tail] <= compute_requant_latency;
                        requant_dep_compute_q[requant_tail] <= retired_computes + 16'd1;
                        requant_layer_q[requant_tail] <= compute_layer;
                        requant_mb_q[requant_tail] <= compute_mb;
                        requant_slot_q[requant_tail] <= compute_slot;
                        requant_flags_q[requant_tail] <= SMF_COMPUTE;
                        requant_op_q[requant_tail] <= OP_REQUANT;
                        requant_tail <= bump(requant_tail);
                        requant_level <= requant_level + 6'd1;
                        accepted_requants <= accepted_requants + 16'd1;
                        conv_pending_requant <= conv_pending_requant - 16'd1;
                    end
                    for (i = 0; i < QUEUE_DEPTH; i = i + 1)
                        if (store_dep_q[i] != 8'd0)
                            store_dep_q[i] <= store_dep_q[i] - 8'd1;
                end else begin
                    compute_remaining <= compute_remaining - 32'd1;
                end
            end

            if (!accept && !compute_done && !requant_busy && (requant_level != 6'd0) &&
                (retired_computes >= requant_dep_compute_q[requant_head])) begin
                requant_busy <= 1'b1;
                requant_remaining <= requant_latency_q[requant_head];
                requant_op <= requant_op_q[requant_head];
                requant_layer <= requant_layer_q[requant_head];
                requant_mb <= requant_mb_q[requant_head];
                requant_slot <= requant_slot_q[requant_head];
                requant_flags <= requant_flags_q[requant_head];
                requant_head <= bump(requant_head);
                requant_level <= requant_level - 6'd1;
            end else if (requant_busy) begin
                if (requant_done && done_slot_ready && !load_done && !compute_done) begin
                    requant_busy <= 1'b0;
                    done_valid <= 1'b1;
                    active_op_class <= requant_op;
                    active_layer_id <= requant_layer;
                    active_microblock_id <= requant_mb;
                    active_stream_slot <= requant_slot;
                    active_stream_meta_flags <= requant_flags;
                    active_phase_id <= PH_REQUANT;
                    active_remaining_cycles <= 32'd0;
                    retired_requants <= retired_requants + 16'd1;
                end else begin
                    requant_remaining <= requant_remaining - 32'd1;
                end
            end

            if (!accept && !store_busy && (store_level != 6'd0) && (store_dep_q[store_head] == 8'd0) &&
                (retired_computes >= store_dep_compute_q[store_head]) &&
                (retired_requants >= store_dep_requant_q[store_head]) &&
                (retired_finals >= store_dep_final_q[store_head])) begin
                store_busy <= 1'b1;
                store_remaining <= store_latency_q[store_head];
                store_op <= store_op_q[store_head];
                store_layer <= store_layer_q[store_head];
                store_mb <= store_mb_q[store_head];
                store_slot <= store_slot_q[store_head];
                store_flags <= store_flags_q[store_head];
                store_head <= bump(store_head);
                store_level <= store_level - 6'd1;
            end else if (store_busy) begin
                if (store_done && done_slot_ready && !load_done && !compute_done && !requant_done) begin
                    store_busy <= 1'b0;
                    done_valid <= 1'b1;
                    active_op_class <= store_op;
                    active_layer_id <= store_layer;
                    active_microblock_id <= store_mb;
                    active_stream_slot <= store_slot;
                    active_stream_meta_flags <= store_flags;
                    active_phase_id <= PH_STORE;
                    active_remaining_cycles <= 32'd0;
                    if ((store_flags & SMF_FINAL_TILE) != 8'd0)
                        retired_finals <= retired_finals + 16'd1;
                end else begin
                    store_remaining <= store_remaining - 32'd1;
                end
            end

            if (compute_busy) begin
                active_phase_id <= PH_COMPUTE;
                active_remaining_cycles <= compute_remaining;
            end else if (requant_busy) begin
                active_phase_id <= PH_REQUANT;
                active_remaining_cycles <= requant_remaining;
            end else if (load_busy) begin
                active_phase_id <= PH_LOAD;
                active_remaining_cycles <= load_remaining;
            end else if (store_busy) begin
                active_phase_id <= PH_STORE;
                active_remaining_cycles <= store_remaining;
            end
        end
    end
endmodule

module mdla7_top #(
    parameter ADDR_WIDTH = 22,
    parameter DATA_WIDTH = 128
) (
    input                       clk,
    input                       rst_n,

    input                       desc_valid,
    output                      desc_ready,
    input      [3:0]            desc_op_class,
    input      [15:0]           desc_layer_id,
    input      [15:0]           desc_microblock_id,
    input      [7:0]            desc_stream_slot,
    input      [7:0]            desc_stream_meta_flags,
    input                       desc_cycle_only_mode,
    input      [31:0]           bytes,
    input      [31:0]           udma_dram_read_bytes,
    input      [31:0]           udma_codec_cycles,
    input                       udma_direction_write,
    input                       udma_final_write_mode,
    input                       udma_sramcrc_mode,
    input                       udma_ref_fill_mode,
    input      [7:0]            udma_input_byte,
    input      [31:0]           udma_out_byte_offset,
    input      [31:0]           udma_ref_off,
    input      [31:0]           udma_sramcrc_expected_count,
    input      [ADDR_WIDTH-1:0] l1mesh_addr,
    input      [DATA_WIDTH-1:0] l1mesh_wdata,
    input      [DATA_WIDTH/8-1:0] l1mesh_wstrb,
    output                      udma_dram_req_valid,
    output                      udma_dram_req_write,
    output     [31:0]           udma_dram_req_addr,
    output     [31:0]           udma_dram_req_bytes,
    output     [DATA_WIDTH-1:0] udma_dram_req_wdata,
    output     [DATA_WIDTH/8-1:0] udma_dram_req_wstrb,
    input      [DATA_WIDTH-1:0] udma_dram_resp_rdata,

    input                       tnps_mode_space_to_depth,
    input      [15:0]           tnps_in_h,
    input      [15:0]           tnps_in_w,
    input      [15:0]           tnps_in_c,
    input      [15:0]           tnps_out_h,
    input      [15:0]           tnps_out_w,
    input      [15:0]           tnps_out_c,
    input      [15:0]           tnps_block,
    input      [1:0]            tnps_elem_bytes,
    input      [31:0]           tnps_sample_out_elem_index,
    input      [31:0]           tnps_sample_in_elem_index,
    input                       tnps_final_write_mode,
    input                       tnps_sramcrc_mode,
    input      [7:0]            tnps_input_byte,
    input      [127:0]          tnps_input_vec,
    input      [31:0]           tnps_out_byte_offset,
    input      [31:0]           tnps_sramcrc_expected_count,
    input      [127:0]          conv_act_vec,
    input      [127:0]          conv_wgt_vec,
    input      [7:0]            conv_elem_count,
    input      [31:0]           conv_workload_bytes,
    input      [31:0]           conv_workload_outputs,
    input                       conv_read_sample_from_l1,
    input                       conv_fp_mode,
    input                       conv_int16_mode,
    input signed [15:0]         conv_zp_in,
    input signed [31:0]         conv_bias,
    input signed [31:0]         conv_multiplier,
    input signed [7:0]          conv_shift,
    input signed [31:0]         conv_zp_out,
    input signed [31:0]         conv_act_min,
    input signed [31:0]         conv_act_max,
    input      [15:0]           conv_in_h,
    input      [15:0]           conv_in_w,
    input      [15:0]           conv_in_c,
    input      [15:0]           conv_out_h,
    input      [15:0]           conv_out_w,
    input      [15:0]           conv_out_c,
    input      [7:0]            conv_k_h,
    input      [7:0]            conv_k_w,
    input      [7:0]            conv_stride_h,
    input      [7:0]            conv_stride_w,
    input      [7:0]            conv_dilation_h,
    input      [7:0]            conv_dilation_w,
    input signed [15:0]         conv_pad_top,
    input signed [15:0]         conv_pad_left,
    input      [1:0]            conv_elem_bytes,
    input      [31:0]           conv_out_elem_index,
    input      [7:0]            conv_tile_output_count,
    input                       conv_partial_first,
    input                       conv_partial_accumulate,
    input                       conv_partial_final,
    input                       conv_refcrc_mode,
    input                       conv_sramcrc_mode,
    input      [31:0]           conv_refcrc_expected_crc,
    input      [31:0]           conv_refcrc_expected_count,
    input      [31:0]           conv_refcrc_ref_off,
    input      [15:0]           conv_sample_kh,
    input      [15:0]           conv_sample_kw,
    input      [15:0]           conv_sample_ic,
    input signed [31:0]         requant_input_value,
    input                       requant_read_input_from_l1,
    input                       requant_sramcrc_mode,
    input      [31:0]           requant_sramcrc_expected_count,
    input      [31:0]           requant_out_byte_offset,
    input                       pool_avg_mode,
    input                       pool_fp_mode,
    input                       pool_int16_mode,
    input                       pool_read_sample_from_l1,
    input                       pool_refcrc_mode,
    input                       pool_sramcrc_mode,
    input      [31:0]           pool_refcrc_expected_count,
    input      [31:0]           pool_refcrc_ref_off,
    input      [31:0]           pool_out_byte_offset,
    input      [127:0]          pool_sample_vec,
    input      [7:0]            pool_elem_count,
    input      [31:0]           pool_workload_bytes,
    input      [1:0]            ewe_op_mode,
    input                       ewe_fp_mode,
    input                       ewe_int16_mode,
    input                       ewe_final_q_mode,
    input                       ewe_read_a_from_l1,
    input                       ewe_sramcrc_mode,
    input      [31:0]           ewe_sramcrc_expected_count,
    input      [31:0]           ewe_out_byte_offset,
    input      [127:0]          ewe_a_vec,
    input      [127:0]          ewe_b_vec,
    input      [7:0]            ewe_elem_count,
    input signed [31:0]         ewe_zp_a,
    input signed [31:0]         ewe_zp_b,
    input signed [31:0]         ewe_zp_out,
    input signed [31:0]         ewe_mult_a,
    input signed [7:0]          ewe_shift_a,
    input signed [31:0]         ewe_mult_b,
    input signed [7:0]          ewe_shift_b,
    input signed [31:0]         ewe_mult_out,
    input signed [7:0]          ewe_shift_out,
    input signed [31:0]         ewe_left_shift,
    input signed [31:0]         ewe_act_min,
    input signed [31:0]         ewe_act_max,

    output                      done_valid,
    input                       done_ready,
    output                      busy,
    output     [3:0]            active_op_class,
    output     [15:0]           active_layer_id,
    output     [15:0]           active_microblock_id,
    output     [7:0]            active_stream_slot,
    output     [7:0]            active_stream_meta_flags,
    output     [3:0]            active_phase_id,
    output     [31:0]           active_remaining_cycles,
    output     [31:0]           tnps_sample_src_byte_offset,
    output     [31:0]           tnps_sample_dst_byte_offset,
    output                      tnps_sample_valid,
    output     [31:0]           l1mesh_crc,
    output     [31:0]           l1mesh_crc_count,
    output     [31:0]           udma_sramcrc_crc,
    output     [31:0]           udma_sramcrc_count,
    output     [31:0]           tnps_sramcrc_crc,
    output     [31:0]           tnps_sramcrc_count,
    output     [31:0]           placement_route_cycles,
    output     [31:0]           microblock_load_count,
    output     [31:0]           microblock_compute_count,
    output     [31:0]           microblock_store_count,
    output     [31:0]           microblock_final_count,
    output reg [31:0]           perf_total_cycles,
    output reg [31:0]           perf_conv_cycles,
    output reg [31:0]           perf_requant_cycles,
    output reg [31:0]           perf_ewe_cycles,
    output reg [31:0]           perf_pool_cycles,
    output reg [31:0]           perf_tnps_cycles,
    output reg [31:0]           perf_udma_r_cycles,
    output reg [31:0]           perf_udma_w_cycles,
    output     [8:0]            block_busy,
    output     [8:0]            block_done_valid,
    output signed [31:0]        conv_acc_out,
    output signed [31:0]        conv_scaled_out,
    output signed [7:0]         conv_out_q,
    output     [63:0]           conv_fp_sum_bits,
    output signed [31:0]        conv_int16_acc_out,
    output     [31:0]           conv_sample_input_byte_offset,
    output     [31:0]           conv_sample_weight_byte_offset,
    output     [31:0]           conv_sample_output_byte_offset,
    output                      conv_sample_input_valid,
    output     [31:0]           conv_first_input_byte_offset,
    output     [31:0]           conv_first_weight_byte_offset,
    output     [7:0]            conv_window_valid_count,
    output     [31:0]           conv_tile_last_output_byte_offset,
    output                      conv_tile_last_input_valid,
    output     [7:0]            conv_tile_last_window_valid_count,
    output     [3:0]            conv_tile_scoreboard_valid_mask,
    output signed [31:0]        conv_tile_scoreboard_q_sum,
    output     [127:0]          conv_tile_result_out_elem_indices,
    output     [127:0]          conv_tile_result_output_byte_offsets,
    output     [127:0]          conv_tile_result_acc_values,
    output     [127:0]          conv_tile_result_q_values,
    output     [3:0]            conv_writeback_valid_mask,
    output     [127:0]          conv_writeback_output_byte_offsets,
    output     [127:0]          conv_writeback_q_values,
    output     [3:0]            conv_shadow_valid_mask,
    output     [127:0]          conv_shadow_output_byte_offsets,
    output     [127:0]          conv_shadow_q_values,
    output     [15:0]           conv_shadow_mem_valid_mask,
    output     [511:0]          conv_shadow_mem_output_byte_offsets,
    output     [511:0]          conv_shadow_mem_q_values,
    output                      conv_shadow_read_valid,
    output     [31:0]           conv_shadow_read_output_byte_offset,
    output     [31:0]           conv_shadow_read_q_value,
    output     [31:0]           conv_shadow_crc,
    output     [31:0]           conv_shadow_byte_count,
    output     [3:0]            conv_psum_valid_mask,
    output     [127:0]          conv_psum_acc_values,
    output signed [31:0]        requant_scaled_out,
    output signed [7:0]         requant_out_q,
    output     [31:0]           requant_sramcrc_crc,
    output     [31:0]           requant_sramcrc_count,
    output signed [31:0]        pool_out,
    output signed [7:0]         pool_out_q,
    output     [63:0]           pool_fp_bits,
    output     [31:0]           pool_refcrc_crc,
    output     [31:0]           pool_refcrc_count,
    output signed [31:0]        ewe_out,
    output signed [7:0]         ewe_out_q,
    output     [31:0]           ewe_sramcrc_crc,
    output     [31:0]           ewe_sramcrc_count,
    output     [63:0]           ewe_fp_bits
);
    localparam [3:0] OP_CONV = 4'd1;
    localparam [3:0] OP_REQUANT = 4'd2;
    localparam [3:0] OP_EWE = 4'd3;
    localparam [3:0] OP_POOL = 4'd4;
    localparam [3:0] OP_TNPS = 4'd5;
    localparam [3:0] OP_UDMA = 4'd6;
    localparam [3:0] OP_L1CRC = 4'd7;
    localparam [7:0] SMF_LOAD_A = 8'h01;
    localparam [7:0] SMF_LOAD_B = 8'h02;
    localparam [7:0] SMF_COMPUTE = 8'h04;
    localparam [7:0] SMF_STORE = 8'h08;
    localparam [7:0] SMF_FINAL_TILE = 8'h10;

    localparam [2:0] ST_IDLE = 3'd0;
    localparam [2:0] ST_RUN  = 3'd1;
    localparam [2:0] ST_WAIT = 3'd2;
    localparam [2:0] ST_DONE = 3'd3;

    reg [2:0] state;
    reg done_valid_q;
    reg [3:0] op_class_q;
    reg [15:0] layer_id_q;
    reg [15:0] microblock_id_q;
    reg [7:0] stream_slot_q;
    reg [7:0] stream_meta_flags_q;
    reg start_pending;
    reg engine_done_seen;
    reg [31:0] bytes_q;
    reg [31:0] udma_dram_read_bytes_q;
    reg [31:0] udma_codec_cycles_q;
    reg udma_direction_write_q;
    reg udma_final_write_mode_q;
    reg udma_sramcrc_mode_q;
    reg udma_ref_fill_mode_q;
    reg [7:0] udma_input_byte_q;
    reg [31:0] udma_out_byte_offset_q;
    reg [31:0] udma_ref_off_q;
    reg [31:0] udma_sramcrc_expected_count_q;
    reg [ADDR_WIDTH-1:0] l1mesh_addr_q;
    reg [DATA_WIDTH-1:0] l1mesh_wdata_q;
    reg [DATA_WIDTH/8-1:0] l1mesh_wstrb_q;
    reg tnps_mode_space_to_depth_q;
    reg [15:0] tnps_in_h_q;
    reg [15:0] tnps_in_w_q;
    reg [15:0] tnps_in_c_q;
    reg [15:0] tnps_out_h_q;
    reg [15:0] tnps_out_w_q;
    reg [15:0] tnps_out_c_q;
    reg [15:0] tnps_block_q;
    reg [1:0] tnps_elem_bytes_q;
    reg [31:0] tnps_sample_out_elem_index_q;
    reg [31:0] tnps_sample_in_elem_index_q;
    reg tnps_final_write_mode_q;
    reg tnps_sramcrc_mode_q;
    reg [7:0] tnps_input_byte_q;
    reg [127:0] tnps_input_vec_q;
    reg [31:0] tnps_out_byte_offset_q;
    reg [31:0] tnps_sramcrc_expected_count_q;
    reg [127:0] conv_act_vec_q;
    reg [127:0] conv_wgt_vec_q;
    reg [7:0] conv_elem_count_q;
    reg [31:0] conv_workload_bytes_q;
    reg [31:0] conv_workload_outputs_q;
    reg conv_read_sample_from_l1_q;
    reg conv_fp_mode_q;
    reg conv_int16_mode_q;
    reg signed [15:0] conv_zp_in_q;
    reg signed [31:0] conv_bias_q;
    reg signed [31:0] conv_multiplier_q;
    reg signed [7:0] conv_shift_q;
    reg signed [31:0] conv_zp_out_q;
    reg signed [31:0] conv_act_min_q;
    reg signed [31:0] conv_act_max_q;
    reg [15:0] conv_in_h_q;
    reg [15:0] conv_in_w_q;
    reg [15:0] conv_in_c_q;
    reg [15:0] conv_out_h_q;
    reg [15:0] conv_out_w_q;
    reg [15:0] conv_out_c_q;
    reg [7:0] conv_k_h_q;
    reg [7:0] conv_k_w_q;
    reg [7:0] conv_stride_h_q;
    reg [7:0] conv_stride_w_q;
    reg [7:0] conv_dilation_h_q;
    reg [7:0] conv_dilation_w_q;
    reg signed [15:0] conv_pad_top_q;
    reg signed [15:0] conv_pad_left_q;
    reg [1:0] conv_elem_bytes_q;
    reg [31:0] conv_out_elem_index_q;
    reg [7:0] conv_tile_output_count_q;
    reg conv_partial_first_q;
    reg conv_partial_accumulate_q;
    reg conv_partial_final_q;
    reg conv_refcrc_mode_q;
    reg conv_sramcrc_mode_q;
    reg [31:0] conv_refcrc_expected_crc_q;
    reg [31:0] conv_refcrc_expected_count_q;
    reg [31:0] conv_refcrc_ref_off_q;
    reg [15:0] conv_sample_kh_q;
    reg [15:0] conv_sample_kw_q;
    reg [15:0] conv_sample_ic_q;
    reg signed [31:0] requant_input_value_q;
    reg requant_read_input_from_l1_q;
    reg requant_sramcrc_mode_q;
    reg [31:0] requant_sramcrc_expected_count_q;
    reg [31:0] requant_out_byte_offset_q;
    reg pool_avg_mode_q;
    reg pool_fp_mode_q;
    reg pool_int16_mode_q;
    reg pool_read_sample_from_l1_q;
    reg pool_refcrc_mode_q;
    reg pool_sramcrc_mode_q;
    reg [31:0] pool_refcrc_expected_count_q;
    reg [31:0] pool_refcrc_ref_off_q;
    reg [31:0] pool_out_byte_offset_q;
    reg [127:0] pool_sample_vec_q;
    reg [7:0] pool_elem_count_q;
    reg [31:0] pool_workload_bytes_q;
    reg [1:0] ewe_op_mode_q;
    reg ewe_fp_mode_q;
    reg ewe_int16_mode_q;
    reg ewe_final_q_mode_q;
    reg ewe_read_a_from_l1_q;
    reg ewe_sramcrc_mode_q;
    reg [31:0] ewe_sramcrc_expected_count_q;
    reg [31:0] ewe_out_byte_offset_q;
    reg [127:0] ewe_a_vec_q;
    reg [127:0] ewe_b_vec_q;
    reg [7:0] ewe_elem_count_q;
    reg signed [31:0] ewe_zp_a_q;
    reg signed [31:0] ewe_zp_b_q;
    reg signed [31:0] ewe_zp_out_q;
    reg signed [31:0] ewe_mult_a_q;
    reg signed [7:0] ewe_shift_a_q;
    reg signed [31:0] ewe_mult_b_q;
    reg signed [7:0] ewe_shift_b_q;
    reg signed [31:0] ewe_mult_out_q;
    reg signed [7:0] ewe_shift_out_q;
    reg signed [31:0] ewe_left_shift_q;
    reg signed [31:0] ewe_act_min_q;
    reg signed [31:0] ewe_act_max_q;

    wire conv_start_ready;
    wire conv_busy;
    wire conv_done_valid;
    wire [3:0] conv_phase_id;
    wire [31:0] conv_remaining_cycles;
    wire conv_l1_req_valid;
    wire conv_l1_req_ready;
    wire conv_l1_req_write;
    wire [ADDR_WIDTH-1:0] conv_l1_req_addr;
    wire [31:0] conv_l1_req_bytes;
    wire [31:0] conv_l1_req_payload_cycles;
    wire [DATA_WIDTH-1:0] conv_l1_req_wdata;
    wire [DATA_WIDTH/8-1:0] conv_l1_req_wstrb;
    wire requant_start_ready;
    wire requant_busy;
    wire requant_done_valid;
    wire [3:0] requant_phase_id;
    wire [31:0] requant_remaining_cycles;
    wire requant_l1_req_valid;
    wire requant_l1_req_ready;
    wire requant_l1_req_write;
    wire [ADDR_WIDTH-1:0] requant_l1_req_addr;
    wire [31:0] requant_l1_req_bytes;
    wire [31:0] requant_l1_req_payload_cycles;
    wire [DATA_WIDTH-1:0] requant_l1_req_wdata;
    wire [DATA_WIDTH/8-1:0] requant_l1_req_wstrb;
    wire pool_start_ready;
    wire pool_busy;
    wire pool_done_valid;
    wire [3:0] pool_phase_id;
    wire [31:0] pool_remaining_cycles;
    wire pool_l1_req_valid;
    wire pool_l1_req_ready;
    wire pool_l1_req_write;
    wire [ADDR_WIDTH-1:0] pool_l1_req_addr;
    wire [31:0] pool_l1_req_bytes;
    wire [31:0] pool_l1_req_payload_cycles;
    wire [DATA_WIDTH-1:0] pool_l1_req_wdata;
    wire [DATA_WIDTH/8-1:0] pool_l1_req_wstrb;
    wire ewe_start_ready;
    wire ewe_busy;
    wire ewe_done_valid;
    wire [3:0] ewe_phase_id;
    wire [31:0] ewe_remaining_cycles;
    wire ewe_l1_req_valid;
    wire ewe_l1_req_ready;
    wire ewe_l1_req_write;
    wire [ADDR_WIDTH-1:0] ewe_l1_req_addr;
    wire [31:0] ewe_l1_req_bytes;
    wire [31:0] ewe_l1_req_payload_cycles;
    wire [DATA_WIDTH-1:0] ewe_l1_req_wdata;
    wire [DATA_WIDTH/8-1:0] ewe_l1_req_wstrb;

    wire udma_start_ready;
    wire udma_busy;
    wire udma_done_valid;
    wire [3:0] udma_phase_id;
    wire [31:0] udma_remaining_cycles;
    wire udma_l1_req_valid;
    wire udma_l1_req_ready;
    wire udma_l1_req_write;
    wire [ADDR_WIDTH-1:0] udma_l1_req_addr;
    wire [31:0] udma_l1_req_bytes;
    wire [31:0] udma_l1_req_payload_cycles;
    wire [DATA_WIDTH-1:0] udma_l1_req_wdata;
    wire [DATA_WIDTH/8-1:0] udma_l1_req_wstrb;

    wire tnps_start_ready;
    wire tnps_busy;
    wire tnps_done_valid;
    wire [3:0] tnps_phase_id;
    wire [31:0] tnps_remaining_cycles;
    wire tnps_l1_req_valid;
    wire tnps_l1_req_ready;
    wire tnps_l1_req_write;
    wire [ADDR_WIDTH-1:0] tnps_l1_req_addr;
    wire [31:0] tnps_l1_req_bytes;
    wire [31:0] tnps_l1_req_payload_cycles;
    wire [DATA_WIDTH-1:0] tnps_l1_req_wdata;
    wire [DATA_WIDTH/8-1:0] tnps_l1_req_wstrb;

    wire l1mgr_busy;
    wire l1mgr_resp_valid;
    wire l1mgr_resp_ready;
    wire [3:0] l1mgr_phase_id;
    wire [31:0] l1mgr_remaining_cycles;
    wire l1mgr_mesh_req_write;
    wire [ADDR_WIDTH-1:0] l1mgr_mesh_req_addr;
    wire [31:0] l1mgr_mesh_req_bytes;
    wire [DATA_WIDTH-1:0] l1mgr_mesh_req_wdata;
    wire [DATA_WIDTH/8-1:0] l1mgr_mesh_req_wstrb;
    wire [3:0] l1mgr_mesh_req_source;
    wire [7:0] l1mgr_mesh_req_tid;

    wire l1mesh_req_ready;
    wire l1mesh_busy;
    wire l1mesh_resp_valid;
    wire l1mesh_resp_read;
    wire [3:0] l1mesh_resp_source;
    wire [7:0] l1mesh_resp_tid;
    wire [3:0] l1mesh_phase_id;
    wire [31:0] l1mesh_remaining_cycles;
    wire [DATA_WIDTH-1:0] l1mesh_rdata;
    reg l1_resp_valid_q;
    reg l1_resp_read_q;
    reg [3:0] l1_resp_source_q;
    reg [7:0] l1_resp_tid_q;
    reg [DATA_WIDTH-1:0] l1_resp_rdata_q;
    wire l1mesh_crc_busy;
    wire l1mesh_crc_done;
    wire [1:0] route_source_x;
    wire [1:0] route_source_y;
    wire [1:0] route_tile_x;
    wire [1:0] route_tile_y;
    wire [1:0] route_bank_x;
    wire [1:0] route_bank_y;
    wire legacy_req_ready;
    wire [3:0] l1mgr_debug_source_unused;
    wire [7:0] l1mgr_debug_tid_unused;
    wire microblock_active_valid;
    wire [3:0] microblock_active_op_class;
    wire [15:0] microblock_active_layer_id;
    wire [15:0] microblock_active_microblock_id;
    wire [7:0] microblock_active_stream_slot;
    wire [7:0] microblock_active_stream_meta_flags;
    wire [31:0] legacy_microblock_load_count;
    wire [31:0] legacy_microblock_compute_count;
    wire [31:0] legacy_microblock_store_count;
    wire [31:0] legacy_microblock_final_count;
    wire stream_desc_mode = desc_cycle_only_mode &&
        ((desc_stream_meta_flags & (SMF_LOAD_A | SMF_LOAD_B | SMF_COMPUTE | SMF_STORE | SMF_FINAL_TILE)) != 8'd0);
    wire microblock_accept = desc_valid && desc_ready && !stream_desc_mode;
    wire microblock_complete = done_valid_q && done_ready;
    wire mb_desc_ready;
    wire mb_done_valid;
    wire mb_busy;
    wire [3:0] mb_active_op_class;
    wire [15:0] mb_active_layer_id;
    wire [15:0] mb_active_microblock_id;
    wire [7:0] mb_active_stream_slot;
    wire [7:0] mb_active_stream_meta_flags;
    wire [3:0] mb_active_phase_id;
    wire [31:0] mb_active_remaining_cycles;
    wire [31:0] mb_microblock_load_count;
    wire [31:0] mb_microblock_compute_count;
    wire [31:0] mb_microblock_store_count;
    wire [31:0] mb_microblock_final_count;
    /* verilator lint_off UNUSEDSIGNAL */
    wire [155:0] final_debug_unused = {
        l1mesh_rdata,
        route_source_x,
        route_source_y,
        route_tile_x,
        route_tile_y,
        route_bank_x,
        route_bank_y,
        legacy_req_ready,
        l1mgr_debug_source_unused,
        l1mgr_debug_tid_unused
    };
    /* verilator lint_on UNUSEDSIGNAL */

    wire run_conv = (op_class_q == OP_CONV);
    wire run_requant = (op_class_q == OP_REQUANT);
    wire run_ewe = (op_class_q == OP_EWE);
    wire run_pool = (op_class_q == OP_POOL);
    wire run_udma = (op_class_q == OP_UDMA);
    wire run_tnps = (op_class_q == OP_TNPS);
    wire run_l1crc = (op_class_q == OP_L1CRC);
    wire selected_start_ready = run_conv ? conv_start_ready :
                                run_requant ? requant_start_ready :
                                run_ewe ? ewe_start_ready :
                                run_pool ? pool_start_ready :
                                run_udma ? udma_start_ready :
                                run_tnps ? tnps_start_ready :
                                run_l1crc ? !l1mesh_crc_busy : 1'b1;
    wire selected_done_valid = run_conv ? conv_done_valid :
                               run_requant ? requant_done_valid :
                               run_ewe ? ewe_done_valid :
                               run_pool ? pool_done_valid :
                               run_udma ? udma_done_valid :
                               run_tnps ? tnps_done_valid :
                               run_l1crc ? l1mesh_crc_done : 1'b1;
    wire selected_busy = run_conv ? conv_busy :
                         run_requant ? requant_busy :
                         run_ewe ? ewe_busy :
                         run_pool ? pool_busy :
                         run_udma ? udma_busy :
                         run_tnps ? tnps_busy :
                         run_l1crc ? l1mesh_crc_busy : 1'b0;
    wire [3:0] selected_phase = run_conv ? conv_phase_id :
                                run_requant ? requant_phase_id :
                                run_ewe ? ewe_phase_id :
                                run_pool ? pool_phase_id :
                                run_udma ? udma_phase_id :
                                run_tnps ? tnps_phase_id : 4'd0;
    wire [31:0] selected_remaining = run_conv ? conv_remaining_cycles :
                                     run_requant ? requant_remaining_cycles :
                                     run_ewe ? ewe_remaining_cycles :
                                     run_pool ? pool_remaining_cycles :
                                     run_udma ? udma_remaining_cycles :
                                     run_tnps ? tnps_remaining_cycles : 32'd0;
    wire l1_drained = !l1mgr_busy && !l1mgr_resp_valid &&
                       !l1mesh_busy && !l1mesh_resp_valid &&
                       !l1_resp_valid_q && !l1mesh_crc_busy;
    wire conv_start = start_pending && run_conv;
    wire requant_start = start_pending && run_requant;
    wire ewe_start = start_pending && run_ewe;
    wire pool_start = start_pending && run_pool;
    wire udma_start = start_pending && run_udma;
    wire tnps_start = start_pending && run_tnps;
    wire l1mesh_crc_start = start_pending && run_l1crc && !l1mesh_crc_busy;
    wire perf_mb_load = mb_busy && (mb_active_phase_id == 4'd2);
    wire perf_mb_compute = mb_busy && ((mb_active_phase_id == 4'd4) || (mb_active_phase_id == 4'd6));
    wire perf_mb_store = mb_busy && (mb_active_phase_id == 4'd7);

    assign desc_ready = stream_desc_mode ? mb_desc_ready : (state == ST_IDLE);
    assign done_valid = mb_busy ? mb_done_valid : done_valid_q;
    assign busy = (state != ST_IDLE) || mb_busy;
    assign active_op_class = mb_busy ? mb_active_op_class :
                             microblock_active_valid ? microblock_active_op_class : op_class_q;
    assign active_layer_id = mb_busy ? mb_active_layer_id :
                             microblock_active_valid ? microblock_active_layer_id : layer_id_q;
    assign active_microblock_id = mb_busy ? mb_active_microblock_id :
                                  microblock_active_valid ? microblock_active_microblock_id : microblock_id_q;
    assign active_stream_slot = mb_busy ? mb_active_stream_slot :
                                microblock_active_valid ? microblock_active_stream_slot : stream_slot_q;
    assign active_stream_meta_flags = mb_busy ? mb_active_stream_meta_flags :
                                      microblock_active_valid ? microblock_active_stream_meta_flags : stream_meta_flags_q;
    assign active_phase_id = mb_busy ? mb_active_phase_id :
                             selected_busy ? selected_phase :
                             l1mgr_busy ? l1mgr_phase_id :
                             l1mesh_busy ? l1mesh_phase_id : 4'd0;
    assign active_remaining_cycles = mb_busy ? mb_active_remaining_cycles :
                                     selected_busy ? selected_remaining :
                                     l1mgr_busy ? l1mgr_remaining_cycles :
                                     l1mesh_busy ? l1mesh_remaining_cycles : 32'd0;
    assign microblock_load_count = mb_microblock_load_count + legacy_microblock_load_count;
    assign microblock_compute_count = mb_microblock_compute_count + legacy_microblock_compute_count;
    assign microblock_store_count = mb_microblock_store_count + legacy_microblock_store_count;
    assign microblock_final_count = mb_microblock_final_count + legacy_microblock_final_count;
    assign l1mgr_resp_ready = l1mesh_req_ready;
    assign conv_l1_req_ready = legacy_req_ready;
    assign block_busy = {l1mesh_busy, l1mgr_busy, udma_busy, tnps_busy, pool_busy, ewe_busy, requant_busy, conv_busy, 1'b0};
    assign block_done_valid = {l1_resp_valid_q, l1mgr_resp_valid, udma_done_valid, tnps_done_valid, pool_done_valid, ewe_done_valid, requant_done_valid, conv_done_valid, 1'b0};

    vf_l1mesh_route_estimator u_route (
        .source_id(op_class_q),
        .addr(l1mesh_addr_q),
        .route_cycles(placement_route_cycles),
        .source_x(route_source_x),
        .source_y(route_source_y),
        .tile_x(route_tile_x),
        .tile_y(route_tile_y),
        .bank_x(route_bank_x),
        .bank_y(route_bank_y)
    );

    vf_microblock_control u_microblock_control (
        .clk(clk),
        .rst_n(rst_n),
        .accept(microblock_accept),
        .complete(microblock_complete),
        .desc_op_class(desc_op_class),
        .desc_layer_id(desc_layer_id),
        .desc_microblock_id(desc_microblock_id),
        .desc_stream_slot(desc_stream_slot),
        .desc_stream_meta_flags(desc_stream_meta_flags),
        .active_valid(microblock_active_valid),
        .active_op_class(microblock_active_op_class),
        .active_layer_id(microblock_active_layer_id),
        .active_microblock_id(microblock_active_microblock_id),
        .active_stream_slot(microblock_active_stream_slot),
        .active_stream_meta_flags(microblock_active_stream_meta_flags),
        .load_count(legacy_microblock_load_count),
        .compute_count(legacy_microblock_compute_count),
        .store_count(legacy_microblock_store_count),
        .final_count(legacy_microblock_final_count)
    );

    vf_mb_stream_scheduler u_mb_stream_scheduler (
        .clk(clk),
        .rst_n(rst_n),
        .desc_valid(desc_valid && stream_desc_mode),
        .desc_ready(mb_desc_ready),
        .desc_op_class(desc_op_class),
        .desc_layer_id(desc_layer_id),
        .desc_microblock_id(desc_microblock_id),
        .desc_stream_slot(desc_stream_slot),
        .desc_stream_meta_flags(desc_stream_meta_flags),
        .bytes(bytes),
        .conv_workload_bytes(conv_workload_bytes),
        .conv_workload_outputs(conv_workload_outputs),
        .pool_workload_bytes(pool_workload_bytes),
        .tnps_bytes(bytes),
        .done_valid(mb_done_valid),
        .done_ready(done_ready),
        .busy(mb_busy),
        .active_op_class(mb_active_op_class),
        .active_layer_id(mb_active_layer_id),
        .active_microblock_id(mb_active_microblock_id),
        .active_stream_slot(mb_active_stream_slot),
        .active_stream_meta_flags(mb_active_stream_meta_flags),
        .active_phase_id(mb_active_phase_id),
        .active_remaining_cycles(mb_active_remaining_cycles),
        .load_count(mb_microblock_load_count),
        .compute_count(mb_microblock_compute_count),
        .store_count(mb_microblock_store_count),
        .final_count(mb_microblock_final_count)
    );

    vf_conv_sample_engine u_conv (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(conv_start),
        .start_ready(conv_start_ready),
        .act_vec(conv_act_vec_q),
        .wgt_vec(conv_wgt_vec_q),
        .elem_count(conv_elem_count_q),
        .workload_bytes(conv_workload_bytes_q),
        .workload_outputs(conv_workload_outputs_q),
        .read_sample_from_l1(conv_read_sample_from_l1_q),
        .fp_mode(conv_fp_mode_q),
        .int16_mode(conv_int16_mode_q),
        .zp_in(conv_zp_in_q),
        .bias(conv_bias_q),
        .multiplier(conv_multiplier_q),
        .shift(conv_shift_q),
        .zp_out(conv_zp_out_q),
        .act_min(conv_act_min_q),
        .act_max(conv_act_max_q),
        .conv_in_h(conv_in_h_q),
        .conv_in_w(conv_in_w_q),
        .conv_in_c(conv_in_c_q),
        .conv_out_h(conv_out_h_q),
        .conv_out_w(conv_out_w_q),
        .conv_out_c(conv_out_c_q),
        .conv_k_h(conv_k_h_q),
        .conv_k_w(conv_k_w_q),
        .conv_stride_h(conv_stride_h_q),
        .conv_stride_w(conv_stride_w_q),
        .conv_dilation_h(conv_dilation_h_q),
        .conv_dilation_w(conv_dilation_w_q),
        .conv_pad_top(conv_pad_top_q),
        .conv_pad_left(conv_pad_left_q),
        .conv_elem_bytes(conv_elem_bytes_q),
        .conv_out_elem_index(conv_out_elem_index_q),
        .conv_tile_output_count(conv_tile_output_count_q),
        .conv_partial_first(conv_partial_first_q),
        .conv_partial_accumulate(conv_partial_accumulate_q),
        .conv_partial_final(conv_partial_final_q),
        .conv_refcrc_mode(conv_refcrc_mode_q),
        .conv_sramcrc_mode(conv_sramcrc_mode_q),
        .conv_refcrc_expected_crc(conv_refcrc_expected_crc_q),
        .conv_refcrc_expected_count(conv_refcrc_expected_count_q),
        .conv_refcrc_ref_off(conv_refcrc_ref_off_q),
        .l1_req_base_addr(l1mesh_addr_q),
        .conv_sample_kh(conv_sample_kh_q),
        .conv_sample_kw(conv_sample_kw_q),
        .conv_sample_ic(conv_sample_ic_q),
        .l1_resp_valid(run_conv && l1_resp_valid_q && l1_resp_read_q &&
                       (l1_resp_source_q == 4'd1)),
        .l1_resp_rdata(l1_resp_rdata_q),
        .l1_req_valid(conv_l1_req_valid),
        .l1_req_ready(conv_l1_req_ready),
        .l1_req_write(conv_l1_req_write),
        .l1_req_addr(conv_l1_req_addr),
        .l1_req_bytes(conv_l1_req_bytes),
        .l1_req_payload_cycles(conv_l1_req_payload_cycles),
        .l1_req_wdata(conv_l1_req_wdata),
        .l1_req_wstrb(conv_l1_req_wstrb),
        .busy(conv_busy),
        .done_valid(conv_done_valid),
        .done_ready(1'b1),
        .phase_id(conv_phase_id),
        .remaining_cycles(conv_remaining_cycles),
        .acc_out(conv_acc_out),
        .scaled_out(conv_scaled_out),
        .out_q(conv_out_q),
        .fp_sum_bits(conv_fp_sum_bits),
        .int16_acc_out(conv_int16_acc_out),
        .conv_sample_input_byte_offset(conv_sample_input_byte_offset),
        .conv_sample_weight_byte_offset(conv_sample_weight_byte_offset),
        .conv_sample_output_byte_offset(conv_sample_output_byte_offset),
        .conv_sample_input_valid(conv_sample_input_valid),
        .conv_first_input_byte_offset(conv_first_input_byte_offset),
        .conv_first_weight_byte_offset(conv_first_weight_byte_offset),
        .conv_window_valid_count(conv_window_valid_count),
        .conv_tile_last_output_byte_offset(conv_tile_last_output_byte_offset),
        .conv_tile_last_input_valid(conv_tile_last_input_valid),
        .conv_tile_last_window_valid_count(conv_tile_last_window_valid_count),
        .conv_tile_scoreboard_valid_mask(conv_tile_scoreboard_valid_mask),
        .conv_tile_scoreboard_q_sum(conv_tile_scoreboard_q_sum),
        .conv_tile_result_out_elem_indices(conv_tile_result_out_elem_indices),
        .conv_tile_result_output_byte_offsets(conv_tile_result_output_byte_offsets),
        .conv_tile_result_acc_values(conv_tile_result_acc_values),
        .conv_tile_result_q_values(conv_tile_result_q_values),
        .conv_writeback_valid_mask(conv_writeback_valid_mask),
        .conv_writeback_output_byte_offsets(conv_writeback_output_byte_offsets),
        .conv_writeback_q_values(conv_writeback_q_values),
        .conv_shadow_valid_mask(conv_shadow_valid_mask),
        .conv_shadow_output_byte_offsets(conv_shadow_output_byte_offsets),
        .conv_shadow_q_values(conv_shadow_q_values),
        .conv_shadow_mem_valid_mask(conv_shadow_mem_valid_mask),
        .conv_shadow_mem_output_byte_offsets(conv_shadow_mem_output_byte_offsets),
        .conv_shadow_mem_q_values(conv_shadow_mem_q_values),
        .conv_shadow_read_valid(conv_shadow_read_valid),
        .conv_shadow_read_output_byte_offset(conv_shadow_read_output_byte_offset),
        .conv_shadow_read_q_value(conv_shadow_read_q_value),
        .conv_shadow_crc(conv_shadow_crc),
        .conv_shadow_byte_count(conv_shadow_byte_count),
        .conv_psum_valid_mask(conv_psum_valid_mask),
        .conv_psum_acc_values(conv_psum_acc_values)
    );

    vf_requant_sample_engine u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(requant_start),
        .start_ready(requant_start_ready),
        .input_value(requant_input_value_q),
        .multiplier(conv_multiplier_q),
        .shift(conv_shift_q),
        .zp_out(conv_zp_out_q),
        .act_min(conv_act_min_q),
        .act_max(conv_act_max_q),
        .read_input_from_l1(requant_read_input_from_l1_q),
        .sramcrc_mode(requant_sramcrc_mode_q),
        .sramcrc_expected_count(requant_sramcrc_expected_count_q),
        .out_byte_offset(requant_out_byte_offset_q),
        .l1_req_base_addr(l1mesh_addr_q),
        .l1_resp_valid(run_requant && l1_resp_valid_q && l1_resp_read_q &&
                       (l1_resp_source_q == 4'd2) &&
                       (l1_resp_tid_q == stream_slot_q)),
        .l1_resp_rdata(l1_resp_rdata_q),
        .l1_req_valid(requant_l1_req_valid),
        .l1_req_ready(requant_l1_req_ready),
        .l1_req_write(requant_l1_req_write),
        .l1_req_addr(requant_l1_req_addr),
        .l1_req_bytes(requant_l1_req_bytes),
        .l1_req_payload_cycles(requant_l1_req_payload_cycles),
        .l1_req_wdata(requant_l1_req_wdata),
        .l1_req_wstrb(requant_l1_req_wstrb),
        .busy(requant_busy),
        .done_valid(requant_done_valid),
        .done_ready(1'b1),
        .phase_id(requant_phase_id),
        .remaining_cycles(requant_remaining_cycles),
        .sramcrc_crc(requant_sramcrc_crc),
        .sramcrc_count(requant_sramcrc_count),
        .scaled_out(requant_scaled_out),
        .out_q(requant_out_q)
    );

    vf_pool_sample_engine u_pool (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(pool_start),
        .start_ready(pool_start_ready),
        .avg_mode(pool_avg_mode_q),
        .fp_mode(pool_fp_mode_q),
        .int16_mode(pool_int16_mode_q),
        .read_sample_from_l1(pool_read_sample_from_l1_q),
        .refcrc_mode(pool_refcrc_mode_q),
        .sramcrc_mode(pool_sramcrc_mode_q),
        .refcrc_expected_count(pool_refcrc_expected_count_q),
        .refcrc_ref_off(pool_refcrc_ref_off_q),
        .out_byte_offset(pool_out_byte_offset_q),
        .l1_req_base_addr(l1mesh_addr_q),
        .sample_vec(pool_sample_vec_q),
        .elem_count(pool_elem_count_q),
        .workload_bytes(pool_workload_bytes_q),
        .l1_resp_valid(run_pool && l1_resp_valid_q && l1_resp_read_q &&
                       (l1_resp_source_q == 4'd4) &&
                       (l1_resp_tid_q == stream_slot_q)),
        .l1_resp_rdata(l1_resp_rdata_q),
        .l1_req_valid(pool_l1_req_valid),
        .l1_req_ready(pool_l1_req_ready),
        .l1_req_write(pool_l1_req_write),
        .l1_req_addr(pool_l1_req_addr),
        .l1_req_bytes(pool_l1_req_bytes),
        .l1_req_payload_cycles(pool_l1_req_payload_cycles),
        .l1_req_wdata(pool_l1_req_wdata),
        .l1_req_wstrb(pool_l1_req_wstrb),
        .busy(pool_busy),
        .done_valid(pool_done_valid),
        .done_ready(1'b1),
        .phase_id(pool_phase_id),
        .remaining_cycles(pool_remaining_cycles),
        .pool_out(pool_out),
        .out_q(pool_out_q),
        .fp_pool_bits(pool_fp_bits),
        .refcrc_crc(pool_refcrc_crc),
        .refcrc_count(pool_refcrc_count)
    );

    vf_ewe_sample_engine u_ewe (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(ewe_start),
        .start_ready(ewe_start_ready),
        .op_mode(ewe_op_mode_q),
        .fp_mode(ewe_fp_mode_q),
        .int16_mode(ewe_int16_mode_q),
        .final_q_mode(ewe_final_q_mode_q),
        .read_a_from_l1(ewe_read_a_from_l1_q),
        .sramcrc_mode(ewe_sramcrc_mode_q),
        .sramcrc_expected_count(ewe_sramcrc_expected_count_q),
        .out_byte_offset(ewe_out_byte_offset_q),
        .l1_req_base_addr(l1mesh_addr_q),
        .zp_a(ewe_zp_a_q),
        .zp_b(ewe_zp_b_q),
        .zp_out(ewe_zp_out_q),
        .mult_a(ewe_mult_a_q),
        .shift_a(ewe_shift_a_q),
        .mult_b(ewe_mult_b_q),
        .shift_b(ewe_shift_b_q),
        .mult_out(ewe_mult_out_q),
        .shift_out(ewe_shift_out_q),
        .left_shift(ewe_left_shift_q),
        .act_min(ewe_act_min_q),
        .act_max(ewe_act_max_q),
        .a_vec(ewe_a_vec_q),
        .b_vec(ewe_b_vec_q),
        .elem_count(ewe_elem_count_q),
        .l1_resp_valid(run_ewe && l1_resp_valid_q && l1_resp_read_q &&
                       (l1_resp_source_q == 4'd3) &&
                       (l1_resp_tid_q == stream_slot_q)),
        .l1_resp_rdata(l1_resp_rdata_q),
        .l1_req_valid(ewe_l1_req_valid),
        .l1_req_ready(ewe_l1_req_ready),
        .l1_req_write(ewe_l1_req_write),
        .l1_req_addr(ewe_l1_req_addr),
        .l1_req_bytes(ewe_l1_req_bytes),
        .l1_req_payload_cycles(ewe_l1_req_payload_cycles),
        .l1_req_wdata(ewe_l1_req_wdata),
        .l1_req_wstrb(ewe_l1_req_wstrb),
        .busy(ewe_busy),
        .done_valid(ewe_done_valid),
        .done_ready(1'b1),
        .phase_id(ewe_phase_id),
        .remaining_cycles(ewe_remaining_cycles),
        .ewe_out(ewe_out),
        .out_q(ewe_out_q),
        .sramcrc_crc(ewe_sramcrc_crc),
        .sramcrc_count(ewe_sramcrc_count),
        .fp_ewe_bits(ewe_fp_bits)
    );

    vf_udma_engine u_udma (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(udma_start),
        .start_ready(udma_start_ready),
        .direction_write(udma_direction_write_q),
        .bytes(bytes_q),
        .dram_read_bytes(udma_dram_read_bytes_q),
        .codec_cycles(udma_codec_cycles_q),
        .final_write_mode(udma_final_write_mode_q),
        .sramcrc_mode(udma_sramcrc_mode_q),
        .ref_fill_mode(udma_ref_fill_mode_q),
        .input_byte(udma_input_byte_q),
        .out_byte_offset(udma_out_byte_offset_q),
        .ref_off(udma_ref_off_q),
        .sramcrc_expected_count(udma_sramcrc_expected_count_q),
        .l1_req_base_addr(l1mesh_addr_q),
        .l1_req_valid(udma_l1_req_valid),
        .l1_req_ready(udma_l1_req_ready),
        .l1_req_write(udma_l1_req_write),
        .l1_req_addr(udma_l1_req_addr),
        .l1_req_bytes(udma_l1_req_bytes),
        .l1_req_payload_cycles(udma_l1_req_payload_cycles),
        .l1_req_wdata(udma_l1_req_wdata),
        .l1_req_wstrb(udma_l1_req_wstrb),
        .l1_resp_valid(run_udma && l1_resp_valid_q && l1_resp_read_q &&
                       (l1_resp_source_q == 4'd6) &&
                       (l1_resp_tid_q == stream_slot_q)),
        .l1_resp_rdata(l1_resp_rdata_q),
        .dram_req_valid(udma_dram_req_valid),
        .dram_req_write(udma_dram_req_write),
        .dram_req_addr(udma_dram_req_addr),
        .dram_req_bytes(udma_dram_req_bytes),
        .dram_req_wdata(udma_dram_req_wdata),
        .dram_req_wstrb(udma_dram_req_wstrb),
        .dram_resp_rdata(udma_dram_resp_rdata),
        .busy(udma_busy),
        .done_valid(udma_done_valid),
        .done_ready(1'b1),
        .phase_id(udma_phase_id),
        .remaining_cycles(udma_remaining_cycles),
        .sramcrc_crc(udma_sramcrc_crc),
        .sramcrc_count(udma_sramcrc_count)
    );

    vf_tnps_engine u_tnps (
        .clk(clk),
        .rst_n(rst_n),
        .start_valid(tnps_start),
        .start_ready(tnps_start_ready),
        .bytes(bytes_q),
        .mode_space_to_depth(tnps_mode_space_to_depth_q),
        .in_h(tnps_in_h_q),
        .in_w(tnps_in_w_q),
        .in_c(tnps_in_c_q),
        .out_h(tnps_out_h_q),
        .out_w(tnps_out_w_q),
        .out_c(tnps_out_c_q),
        .block(tnps_block_q),
        .elem_bytes(tnps_elem_bytes_q),
        .sample_out_elem_index(tnps_sample_out_elem_index_q),
        .sample_in_elem_index(tnps_sample_in_elem_index_q),
        .final_write_mode(tnps_final_write_mode_q),
        .sramcrc_mode(tnps_sramcrc_mode_q),
        .input_byte(tnps_input_byte_q),
        .input_vec(tnps_input_vec_q),
        .out_byte_offset(tnps_out_byte_offset_q),
        .sramcrc_expected_count(tnps_sramcrc_expected_count_q),
        .l1_req_base_addr(l1mesh_addr_q),
        .l1_resp_valid(run_tnps && l1_resp_valid_q && l1_resp_read_q &&
                       (l1_resp_source_q == 4'd5) &&
                       (l1_resp_tid_q == stream_slot_q)),
        .l1_resp_rdata(l1_resp_rdata_q),
        .l1_req_valid(tnps_l1_req_valid),
        .l1_req_ready(tnps_l1_req_ready),
        .l1_req_write(tnps_l1_req_write),
        .l1_req_addr(tnps_l1_req_addr),
        .l1_req_bytes(tnps_l1_req_bytes),
        .l1_req_payload_cycles(tnps_l1_req_payload_cycles),
        .l1_req_wdata(tnps_l1_req_wdata),
        .l1_req_wstrb(tnps_l1_req_wstrb),
        .busy(tnps_busy),
        .done_valid(tnps_done_valid),
        .done_ready(1'b1),
        .phase_id(tnps_phase_id),
        .remaining_cycles(tnps_remaining_cycles),
        .sample_src_byte_offset(tnps_sample_src_byte_offset),
        .sample_dst_byte_offset(tnps_sample_dst_byte_offset),
        .sample_valid(tnps_sample_valid),
        .sramcrc_crc(tnps_sramcrc_crc),
        .sramcrc_count(tnps_sramcrc_count)
    );

    l1manager u_l1manager (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(conv_l1_req_valid),
        .req_ready(legacy_req_ready),
        .req_write(conv_l1_req_write),
        .req_l1(1'b1),
        .req_source(4'd1),
        .req_tid(stream_slot_q),
        .req_bytes(conv_l1_req_bytes),
        .req_payload_cycles(conv_l1_req_payload_cycles),
        .req_addr(conv_l1_req_addr),
        .req_wdata(conv_l1_req_wdata),
        .req_wstrb(conv_l1_req_wstrb),
        .udma_req_valid(udma_l1_req_valid),
        .udma_req_ready(udma_l1_req_ready),
        .udma_req_write(udma_l1_req_write),
        .udma_req_tid(stream_slot_q),
        .udma_req_bytes(udma_l1_req_bytes),
        .udma_req_payload_cycles(udma_l1_req_payload_cycles),
        .udma_req_addr(udma_l1_req_addr),
        .udma_req_wdata(udma_l1_req_wdata),
        .udma_req_wstrb(udma_l1_req_wstrb),
        .requant_req_valid(requant_l1_req_valid),
        .requant_req_ready(requant_l1_req_ready),
        .requant_req_write(requant_l1_req_write),
        .requant_req_tid(stream_slot_q),
        .requant_req_bytes(requant_l1_req_bytes),
        .requant_req_payload_cycles(requant_l1_req_payload_cycles),
        .requant_req_addr(requant_l1_req_addr),
        .requant_req_wdata(requant_l1_req_wdata),
        .requant_req_wstrb(requant_l1_req_wstrb),
        .ewe_req_valid(ewe_l1_req_valid),
        .ewe_req_ready(ewe_l1_req_ready),
        .ewe_req_write(ewe_l1_req_write),
        .ewe_req_tid(stream_slot_q),
        .ewe_req_bytes(ewe_l1_req_bytes),
        .ewe_req_payload_cycles(ewe_l1_req_payload_cycles),
        .ewe_req_addr(ewe_l1_req_addr),
        .ewe_req_wdata(ewe_l1_req_wdata),
        .ewe_req_wstrb(ewe_l1_req_wstrb),
        .pool_req_valid(pool_l1_req_valid),
        .pool_req_ready(pool_l1_req_ready),
        .pool_req_write(pool_l1_req_write),
        .pool_req_tid(stream_slot_q),
        .pool_req_bytes(pool_l1_req_bytes),
        .pool_req_payload_cycles(pool_l1_req_payload_cycles),
        .pool_req_addr(pool_l1_req_addr),
        .pool_req_wdata(pool_l1_req_wdata),
        .pool_req_wstrb(pool_l1_req_wstrb),
        .tnps_req_valid(tnps_l1_req_valid),
        .tnps_req_ready(tnps_l1_req_ready),
        .tnps_req_write(tnps_l1_req_write),
        .tnps_req_tid(stream_slot_q),
        .tnps_req_bytes(tnps_l1_req_bytes),
        .tnps_req_payload_cycles(tnps_l1_req_payload_cycles),
        .tnps_req_addr(tnps_l1_req_addr),
        .tnps_req_wdata(tnps_l1_req_wdata),
        .tnps_req_wstrb(tnps_l1_req_wstrb),
        .mesh_req_write(l1mgr_mesh_req_write),
        .mesh_req_addr(l1mgr_mesh_req_addr),
        .mesh_req_bytes(l1mgr_mesh_req_bytes),
        .mesh_req_wdata(l1mgr_mesh_req_wdata),
        .mesh_req_wstrb(l1mgr_mesh_req_wstrb),
        .mesh_req_source(l1mgr_mesh_req_source),
        .mesh_req_tid(l1mgr_mesh_req_tid),
        .resp_valid(l1mgr_resp_valid),
        .resp_ready(l1mgr_resp_ready),
        .busy(l1mgr_busy),
        .phase_id(l1mgr_phase_id),
        .remaining_cycles(l1mgr_remaining_cycles),
        .debug_source(l1mgr_debug_source_unused),
        .debug_tid(l1mgr_debug_tid_unused)
    );

    l1mesh u_l1mesh (
        .clk(clk),
        .rst_n(rst_n),
        .req_valid(l1mgr_resp_valid),
        .req_ready(l1mesh_req_ready),
        .req_write(l1mgr_mesh_req_write),
        .req_addr(l1mgr_mesh_req_addr),
        .req_bytes(l1mgr_mesh_req_bytes),
        .route_cycles(placement_route_cycles),
        .req_wdata(l1mgr_mesh_req_wdata),
        .req_wstrb(l1mgr_mesh_req_wstrb),
        .req_source(l1mgr_mesh_req_source),
        .req_tid(l1mgr_mesh_req_tid),
        .debug_crc_start(l1mesh_crc_start),
        .debug_crc_addr(l1mesh_addr_q),
        .debug_crc_count(bytes_q),
        .debug_crc_busy(l1mesh_crc_busy),
        .debug_crc_done(l1mesh_crc_done),
        .debug_crc(l1mesh_crc),
        .debug_crc_byte_count(l1mesh_crc_count),
        .resp_valid(l1mesh_resp_valid),
        .resp_read(l1mesh_resp_read),
        .resp_source(l1mesh_resp_source),
        .resp_tid(l1mesh_resp_tid),
        .resp_ready(1'b1),
        .resp_rdata(l1mesh_rdata),
        .busy(l1mesh_busy),
        .phase_id(l1mesh_phase_id),
        .remaining_cycles(l1mesh_remaining_cycles)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done_valid_q <= 1'b0;
            l1_resp_valid_q <= 1'b0;
            l1_resp_read_q <= 1'b0;
            l1_resp_source_q <= 4'd0;
            l1_resp_tid_q <= 8'd0;
            l1_resp_rdata_q <= {DATA_WIDTH{1'b0}};
            op_class_q <= 4'd0;
            layer_id_q <= 16'd0;
            microblock_id_q <= 16'd0;
            stream_slot_q <= 8'd0;
            stream_meta_flags_q <= 8'd0;
            start_pending <= 1'b0;
            engine_done_seen <= 1'b0;
            perf_total_cycles <= 32'd0;
            perf_conv_cycles <= 32'd0;
            perf_requant_cycles <= 32'd0;
            perf_ewe_cycles <= 32'd0;
            perf_pool_cycles <= 32'd0;
            perf_tnps_cycles <= 32'd0;
            perf_udma_r_cycles <= 32'd0;
            perf_udma_w_cycles <= 32'd0;
            bytes_q <= 32'd0;
            udma_dram_read_bytes_q <= 32'd0;
            udma_codec_cycles_q <= 32'd0;
            udma_direction_write_q <= 1'b0;
            udma_final_write_mode_q <= 1'b0;
            udma_sramcrc_mode_q <= 1'b0;
            udma_ref_fill_mode_q <= 1'b0;
            udma_input_byte_q <= 8'd0;
            udma_out_byte_offset_q <= 32'd0;
            udma_ref_off_q <= 32'd0;
            udma_sramcrc_expected_count_q <= 32'd0;
            l1mesh_addr_q <= {ADDR_WIDTH{1'b0}};
            l1mesh_wdata_q <= {DATA_WIDTH{1'b0}};
            l1mesh_wstrb_q <= {DATA_WIDTH/8{1'b0}};
            tnps_mode_space_to_depth_q <= 1'b1;
            tnps_in_h_q <= 16'd0;
            tnps_in_w_q <= 16'd0;
            tnps_in_c_q <= 16'd0;
            tnps_out_h_q <= 16'd0;
            tnps_out_w_q <= 16'd0;
            tnps_out_c_q <= 16'd0;
            tnps_block_q <= 16'd0;
            tnps_elem_bytes_q <= 2'd1;
            tnps_sample_out_elem_index_q <= 32'd0;
            tnps_sample_in_elem_index_q <= 32'd0;
            tnps_final_write_mode_q <= 1'b0;
            tnps_sramcrc_mode_q <= 1'b0;
            tnps_input_byte_q <= 8'd0;
            tnps_input_vec_q <= 128'd0;
            tnps_out_byte_offset_q <= 32'd0;
            tnps_sramcrc_expected_count_q <= 32'd0;
            conv_act_vec_q <= 128'd0;
            conv_wgt_vec_q <= 128'd0;
            conv_elem_count_q <= 8'd0;
            conv_workload_bytes_q <= 32'd0;
            conv_workload_outputs_q <= 32'd0;
            conv_read_sample_from_l1_q <= 1'b0;
            conv_fp_mode_q <= 1'b0;
            conv_int16_mode_q <= 1'b0;
            conv_zp_in_q <= 16'sd0;
            conv_bias_q <= 32'sd0;
            conv_multiplier_q <= 32'sd1073741824;
            conv_shift_q <= 8'sd1;
            conv_zp_out_q <= 32'sd0;
            conv_act_min_q <= -32'sd128;
            conv_act_max_q <= 32'sd127;
            conv_in_h_q <= 16'd1;
            conv_in_w_q <= 16'd1;
            conv_in_c_q <= 16'd1;
            conv_out_h_q <= 16'd1;
            conv_out_w_q <= 16'd1;
            conv_out_c_q <= 16'd1;
            conv_k_h_q <= 8'd1;
            conv_k_w_q <= 8'd1;
            conv_stride_h_q <= 8'd1;
            conv_stride_w_q <= 8'd1;
            conv_dilation_h_q <= 8'd1;
            conv_dilation_w_q <= 8'd1;
            conv_pad_top_q <= 16'sd0;
            conv_pad_left_q <= 16'sd0;
            conv_elem_bytes_q <= 2'd1;
            conv_out_elem_index_q <= 32'd0;
            conv_tile_output_count_q <= 8'd1;
            conv_partial_first_q <= 1'b0;
            conv_partial_accumulate_q <= 1'b0;
            conv_partial_final_q <= 1'b0;
            conv_refcrc_mode_q <= 1'b0;
            conv_sramcrc_mode_q <= 1'b0;
            conv_refcrc_expected_crc_q <= 32'd0;
            conv_refcrc_expected_count_q <= 32'd0;
            conv_refcrc_ref_off_q <= 32'd0;
            conv_sample_kh_q <= 16'd0;
            conv_sample_kw_q <= 16'd0;
            conv_sample_ic_q <= 16'd0;
            requant_input_value_q <= 32'sd0;
            requant_read_input_from_l1_q <= 1'b0;
            requant_sramcrc_mode_q <= 1'b0;
            requant_sramcrc_expected_count_q <= 32'd0;
            requant_out_byte_offset_q <= 32'd0;
            pool_avg_mode_q <= 1'b0;
            pool_fp_mode_q <= 1'b0;
            pool_int16_mode_q <= 1'b0;
            pool_read_sample_from_l1_q <= 1'b0;
            pool_refcrc_mode_q <= 1'b0;
            pool_sramcrc_mode_q <= 1'b0;
            pool_refcrc_expected_count_q <= 32'd0;
            pool_refcrc_ref_off_q <= 32'd0;
            pool_out_byte_offset_q <= 32'd0;
            pool_sample_vec_q <= 128'd0;
            pool_elem_count_q <= 8'd0;
            pool_workload_bytes_q <= 32'd0;
            ewe_op_mode_q <= 2'd0;
            ewe_fp_mode_q <= 1'b0;
            ewe_int16_mode_q <= 1'b0;
            ewe_final_q_mode_q <= 1'b0;
            ewe_read_a_from_l1_q <= 1'b0;
            ewe_sramcrc_mode_q <= 1'b0;
            ewe_sramcrc_expected_count_q <= 32'd0;
            ewe_out_byte_offset_q <= 32'd0;
            ewe_a_vec_q <= 128'd0;
            ewe_b_vec_q <= 128'd0;
            ewe_elem_count_q <= 8'd0;
            ewe_zp_a_q <= 32'sd0;
            ewe_zp_b_q <= 32'sd0;
            ewe_zp_out_q <= 32'sd0;
            ewe_mult_a_q <= 32'sd1073741824;
            ewe_shift_a_q <= 8'sd0;
            ewe_mult_b_q <= 32'sd1073741824;
            ewe_shift_b_q <= 8'sd0;
            ewe_mult_out_q <= 32'sd1073741824;
            ewe_shift_out_q <= 8'sd0;
            ewe_left_shift_q <= 32'sd0;
            ewe_act_min_q <= -32'sd128;
            ewe_act_max_q <= 32'sd127;
        end else begin
            l1_resp_valid_q <= l1mesh_resp_valid;
            l1_resp_read_q <= l1mesh_resp_read;
            l1_resp_source_q <= l1mesh_resp_source;
            l1_resp_tid_q <= l1mesh_resp_tid;
            l1_resp_rdata_q <= l1mesh_rdata;
            if (busy)
                perf_total_cycles <= perf_total_cycles + 32'd1;
            if (conv_busy || (perf_mb_compute && (mb_active_op_class == OP_CONV)))
                perf_conv_cycles <= perf_conv_cycles + 32'd1;
            if (requant_busy || (perf_mb_compute && (mb_active_op_class == OP_REQUANT)))
                perf_requant_cycles <= perf_requant_cycles + 32'd1;
            if (ewe_busy || (perf_mb_compute && (mb_active_op_class == OP_EWE)))
                perf_ewe_cycles <= perf_ewe_cycles + 32'd1;
            if (pool_busy || (perf_mb_compute && (mb_active_op_class == OP_POOL)))
                perf_pool_cycles <= perf_pool_cycles + 32'd1;
            if (tnps_busy || (perf_mb_compute && (mb_active_op_class == OP_TNPS)))
                perf_tnps_cycles <= perf_tnps_cycles + 32'd1;
            if ((udma_busy && !udma_direction_write_q) || perf_mb_load)
                perf_udma_r_cycles <= perf_udma_r_cycles + 32'd1;
            if ((udma_busy && udma_direction_write_q) || perf_mb_store)
                perf_udma_w_cycles <= perf_udma_w_cycles + 32'd1;
            case (state)
                ST_IDLE: begin
                    done_valid_q <= 1'b0;
                    if (desc_valid && desc_ready && !stream_desc_mode) begin
                        op_class_q <= desc_op_class;
                        layer_id_q <= desc_layer_id;
                        microblock_id_q <= desc_microblock_id;
                        stream_slot_q <= desc_stream_slot;
                        stream_meta_flags_q <= desc_stream_meta_flags;
                        bytes_q <= bytes;
                        udma_dram_read_bytes_q <= udma_dram_read_bytes;
                        udma_codec_cycles_q <= udma_codec_cycles;
                        udma_direction_write_q <= udma_direction_write;
                        udma_final_write_mode_q <= udma_final_write_mode;
                        udma_sramcrc_mode_q <= udma_sramcrc_mode;
                        udma_ref_fill_mode_q <= udma_ref_fill_mode;
                        udma_input_byte_q <= udma_input_byte;
                        udma_out_byte_offset_q <= udma_out_byte_offset;
                        udma_ref_off_q <= udma_ref_off;
                        udma_sramcrc_expected_count_q <= udma_sramcrc_expected_count;
                        l1mesh_addr_q <= l1mesh_addr;
                        l1mesh_wdata_q <= l1mesh_wdata;
                        l1mesh_wstrb_q <= l1mesh_wstrb;
                        tnps_mode_space_to_depth_q <= tnps_mode_space_to_depth;
                        tnps_in_h_q <= tnps_in_h;
                        tnps_in_w_q <= tnps_in_w;
                        tnps_in_c_q <= tnps_in_c;
                        tnps_out_h_q <= tnps_out_h;
                        tnps_out_w_q <= tnps_out_w;
                        tnps_out_c_q <= tnps_out_c;
                        tnps_block_q <= tnps_block;
                        tnps_elem_bytes_q <= tnps_elem_bytes;
                        tnps_sample_out_elem_index_q <= tnps_sample_out_elem_index;
                        tnps_sample_in_elem_index_q <= tnps_sample_in_elem_index;
                        tnps_final_write_mode_q <= tnps_final_write_mode;
                        tnps_sramcrc_mode_q <= tnps_sramcrc_mode;
                        tnps_input_byte_q <= tnps_input_byte;
                        tnps_input_vec_q <= tnps_input_vec;
                        tnps_out_byte_offset_q <= tnps_out_byte_offset;
                        tnps_sramcrc_expected_count_q <= tnps_sramcrc_expected_count;
                        conv_act_vec_q <= conv_act_vec;
                        conv_wgt_vec_q <= conv_wgt_vec;
                        conv_elem_count_q <= conv_elem_count;
                        conv_workload_bytes_q <= conv_workload_bytes;
                        conv_workload_outputs_q <= conv_workload_outputs;
                        conv_read_sample_from_l1_q <= conv_read_sample_from_l1;
                        conv_fp_mode_q <= conv_fp_mode;
                        conv_int16_mode_q <= conv_int16_mode;
                        conv_zp_in_q <= conv_zp_in;
                        conv_bias_q <= conv_bias;
                        conv_multiplier_q <= conv_multiplier;
                        conv_shift_q <= conv_shift;
                        conv_zp_out_q <= conv_zp_out;
                        conv_act_min_q <= conv_act_min;
                        conv_act_max_q <= conv_act_max;
                        conv_in_h_q <= conv_in_h;
                        conv_in_w_q <= conv_in_w;
                        conv_in_c_q <= conv_in_c;
                        conv_out_h_q <= conv_out_h;
                        conv_out_w_q <= conv_out_w;
                        conv_out_c_q <= conv_out_c;
                        conv_k_h_q <= conv_k_h;
                        conv_k_w_q <= conv_k_w;
                        conv_stride_h_q <= conv_stride_h;
                        conv_stride_w_q <= conv_stride_w;
                        conv_dilation_h_q <= conv_dilation_h;
                        conv_dilation_w_q <= conv_dilation_w;
                        conv_pad_top_q <= conv_pad_top;
                        conv_pad_left_q <= conv_pad_left;
                        conv_elem_bytes_q <= conv_elem_bytes;
                        conv_out_elem_index_q <= conv_out_elem_index;
                        conv_tile_output_count_q <= conv_tile_output_count;
                        conv_partial_first_q <= conv_partial_first;
                        conv_partial_accumulate_q <= conv_partial_accumulate;
                        conv_partial_final_q <= conv_partial_final;
                        conv_refcrc_mode_q <= conv_refcrc_mode;
                        conv_sramcrc_mode_q <= conv_sramcrc_mode;
                        conv_refcrc_expected_crc_q <= conv_refcrc_expected_crc;
                        conv_refcrc_expected_count_q <= conv_refcrc_expected_count;
                        conv_refcrc_ref_off_q <= conv_refcrc_ref_off;
                        conv_sample_kh_q <= conv_sample_kh;
                        conv_sample_kw_q <= conv_sample_kw;
                        conv_sample_ic_q <= conv_sample_ic;
                        requant_input_value_q <= requant_input_value;
                        requant_read_input_from_l1_q <= requant_read_input_from_l1;
                        requant_sramcrc_mode_q <= requant_sramcrc_mode;
                        requant_sramcrc_expected_count_q <= requant_sramcrc_expected_count;
                        requant_out_byte_offset_q <= requant_out_byte_offset;
                        pool_avg_mode_q <= pool_avg_mode;
                        pool_fp_mode_q <= pool_fp_mode;
                        pool_int16_mode_q <= pool_int16_mode;
                        pool_read_sample_from_l1_q <= pool_read_sample_from_l1;
                        pool_refcrc_mode_q <= pool_refcrc_mode;
                        pool_sramcrc_mode_q <= pool_sramcrc_mode;
                        pool_refcrc_expected_count_q <= pool_refcrc_expected_count;
                        pool_refcrc_ref_off_q <= pool_refcrc_ref_off;
                        pool_out_byte_offset_q <= pool_out_byte_offset;
                        pool_sample_vec_q <= pool_sample_vec;
                        pool_elem_count_q <= pool_elem_count;
                        pool_workload_bytes_q <= pool_workload_bytes;
                        ewe_op_mode_q <= ewe_op_mode;
                        ewe_fp_mode_q <= ewe_fp_mode;
                        ewe_int16_mode_q <= ewe_int16_mode;
                        ewe_final_q_mode_q <= ewe_final_q_mode;
                        ewe_read_a_from_l1_q <= ewe_read_a_from_l1;
                        ewe_sramcrc_mode_q <= ewe_sramcrc_mode;
                        ewe_sramcrc_expected_count_q <= ewe_sramcrc_expected_count;
                        ewe_out_byte_offset_q <= ewe_out_byte_offset;
                        ewe_a_vec_q <= ewe_a_vec;
                        ewe_b_vec_q <= ewe_b_vec;
                        ewe_elem_count_q <= ewe_elem_count;
                        ewe_zp_a_q <= ewe_zp_a;
                        ewe_zp_b_q <= ewe_zp_b;
                        ewe_zp_out_q <= ewe_zp_out;
                        ewe_mult_a_q <= ewe_mult_a;
                        ewe_shift_a_q <= ewe_shift_a;
                        ewe_mult_b_q <= ewe_mult_b;
                        ewe_shift_b_q <= ewe_shift_b;
                        ewe_mult_out_q <= ewe_mult_out;
                        ewe_shift_out_q <= ewe_shift_out;
                        ewe_left_shift_q <= ewe_left_shift;
                        ewe_act_min_q <= ewe_act_min;
                        ewe_act_max_q <= ewe_act_max;
                        start_pending <= 1'b1;
                        engine_done_seen <= 1'b0;
                        state <= ST_RUN;
                    end
                end
                ST_RUN: begin
                    if (start_pending && selected_start_ready)
                        start_pending <= 1'b0;
                    if (selected_done_valid)
                        engine_done_seen <= 1'b1;
                    if (!start_pending && (engine_done_seen || selected_done_valid))
                        state <= ST_WAIT;
                end
                ST_WAIT: begin
                    if (l1_drained) begin
                        done_valid_q <= 1'b1;
                        state <= ST_DONE;
                    end
                end
                ST_DONE: begin
                    if (done_valid_q && done_ready) begin
                        done_valid_q <= 1'b0;
                        state <= ST_IDLE;
                    end
                end
                default: begin
                    state <= ST_IDLE;
                    done_valid_q <= 1'b0;
                    start_pending <= 1'b0;
                    engine_done_seen <= 1'b0;
                end
            endcase
        end
    end

endmodule
