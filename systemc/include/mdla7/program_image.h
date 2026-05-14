#pragma once

#include <cstdint>

namespace mdla7 {

#pragma pack(push, 1)
struct ProgHeader {
    uint32_t magic;          // 'MDL7'
    uint32_t version;        // 2/3 use LayerMetaDiskV3, 4 uses LayerMetaDiskV4
    uint32_t num_layers;
    uint32_t data_offset;
};

struct LayerMeta {
    uint16_t in_h, in_w, in_c, out_h, out_w, out_c;
    uint8_t  k_h, k_w, s_h, s_w, p_t, p_b, p_l, p_r;
    uint32_t dram_in, dram_wgt, dram_out;
    uint32_t in_size, wgt_size, ref_size;
    uint64_t in_off, wgt_off, ref_off;
    uint16_t group;
    uint16_t op_kind;
    uint16_t dtype;
    int16_t  zp_in_eff;
};

struct LayerMetaDiskV3 {
    uint16_t in_h, in_w, in_c, out_h, out_w, out_c;
    uint8_t  k_h, k_w, s_h, s_w, p_t, p_b, p_l, p_r;
    uint32_t dram_in, dram_wgt, dram_out;
    uint32_t in_size, wgt_size, ref_size;
    uint32_t in_off, wgt_off, ref_off;
    uint16_t group;
    uint16_t op_kind;
    uint16_t dtype;
    int16_t  zp_in_eff;
};

struct LayerMetaDiskV4 {
    uint16_t in_h, in_w, in_c, out_h, out_w, out_c;
    uint8_t  k_h, k_w, s_h, s_w, p_t, p_b, p_l, p_r;
    uint32_t dram_in, dram_wgt, dram_out;
    uint32_t in_size, wgt_size, ref_size;
    uint64_t in_off, wgt_off, ref_off;
    uint16_t group;
    uint16_t op_kind;
    uint16_t dtype;
    int16_t  zp_in_eff;
};

struct GraphMeta {
    int32_t input0_tensor, input1_tensor, output_tensor;
    int32_t producer0_layer, producer1_layer;
    int32_t first_consumer_layer, last_consumer_layer;
    int32_t consumer_count;
};

enum OpKindEnum : uint16_t {
    OK_CONV        = 0,
    OK_DWCONV      = 1,
    OK_AVG_POOL    = 2,
    OK_MAX_POOL    = 3,
    OK_SOFTMAX     = 4,
    OK_RESHAPE     = 5,
    OK_FC          = 6,
    OK_ADD         = 7,
    OK_CONCAT      = 8,
    OK_GATHER      = 9,
    OK_MUL         = 10,
    OK_SUB         = 11,
    OK_HARD_SWISH  = 12,
    OK_GELU        = 13,
    OK_D2SPACE     = 14,
    OK_MATERIALIZE = 15,
    OK_TRANSPOSE   = 16,
    OK_S2SPACE     = 17,
    OK_SQUEEZE     = 18,
    OK_EXPAND_DIMS = 19,
    OK_SLICE       = 20,
    OK_STRIDED_SLICE = 21,
    OK_PAD         = 22,
    OK_PACK        = 23,
    OK_UNPACK      = 24,
    OK_TILE        = 25,
    OK_SPLIT       = 26,
    OK_LOGISTIC    = 27,
    OK_RSQRT       = 28,    // 1/sqrt(x), INT8 LUT-based EWE unary
    OK_TANH        = 29,    // tanh(x), INT8 LUT-based EWE unary
    OK_FC_BMM      = 30,    // BATCH_MATMUL lowered to 1x1 CONV; same engine as OK_FC
};

inline const char* op_name(uint16_t k) {
    switch (k) {
        case OK_CONV:       return "   conv";
        case OK_DWCONV:     return " dwconv";
        case OK_AVG_POOL:   return "avgpool";
        case OK_MAX_POOL:   return "maxpool";
        case OK_SOFTMAX:    return "softmax";
        case OK_RESHAPE:    return "reshape";
        case OK_FC:         return "     fc";
        case OK_ADD:        return "    add";
        case OK_CONCAT:     return " concat";
        case OK_GATHER:     return " gather";
        case OK_MUL:        return "    mul";
        case OK_SUB:        return "    sub";
        case OK_HARD_SWISH: return "h_swsh";
        case OK_GELU:       return "   gelu";
        case OK_D2SPACE:    return "d2spac";
        case OK_MATERIALIZE:return "matrlz";
        case OK_TRANSPOSE:  return " trnps";
        case OK_S2SPACE:    return "s2spac";
        case OK_SQUEEZE:    return "squeez";
        case OK_EXPAND_DIMS:return "expand";
        case OK_SLICE:      return " slice";
        case OK_STRIDED_SLICE:return "sslice";
        case OK_PAD:        return "   pad";
        case OK_PACK:       return "  pack";
        case OK_UNPACK:     return "unpack";
        case OK_TILE:       return "  tile";
        case OK_SPLIT:      return " split";
        case OK_LOGISTIC:   return "logist";
        case OK_RSQRT:      return " rsqrt";
        case OK_TANH:       return "  tanh";
        case OK_FC_BMM:     return "fc(bmm)";
    }
    return "??unknown";
}
// True for both native FC and BATCH_MATMUL lowered to CONV.
inline bool is_fc_kind(uint16_t k) {
    return k == OK_FC || k == OK_FC_BMM;
}
#pragma pack(pop)

static_assert(sizeof(ProgHeader) == 16);
static_assert(sizeof(LayerMetaDiskV3) == 64);
static_assert(sizeof(LayerMetaDiskV4) == 76);
static_assert(sizeof(GraphMeta)  == 32);

} // namespace mdla7
