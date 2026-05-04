#pragma once

// LUT for v1 SOFTMAX. Values: round(exp(-i / 32) * 16384) for i in [0, 256).
// MUST be byte-identical with the Python copy in scripts/compile_model.py
// so numpy reference and SystemC sim produce the same INT8 output.

#include <cstdint>

namespace mdla7 {

static const int32_t SOFTMAX_LUT[256] = {
    16384, 15880, 15391, 14918, 14459, 14014, 13583, 13165,
    12760, 12367, 11987, 11618, 11261, 10914, 10578, 10253,
     9937,  9632,  9335,  9048,  8770,  8500,  8238,  7985,
     7739,  7501,  7270,  7047,  6830,  6620,  6416,  6219,
     6027,  5842,  5662,  5488,  5319,  5155,  4997,  4843,
     4694,  4550,  4410,  4274,  4143,  4015,  3892,  3772,
     3656,  3543,  3434,  3329,  3226,  3127,  3031,  2937,
     2847,  2760,  2675,  2592,  2513,  2435,  2360,  2288,
     2217,  2149,  2083,  2019,  1957,  1897,  1838,  1782,
     1727,  1674,  1622,  1572,  1524,  1477,  1432,  1388,
     1345,  1304,  1263,  1225,  1187,  1150,  1115,  1081,
     1047,  1015,   984,   954,   924,   896,   868,   842,
      816,   791,   766,   743,   720,   698,   676,   655,
      635,   616,   597,   578,   561,   543,   527,   510,
      495,   480,   465,   450,   437,   423,   410,   398,
      385,   373,   362,   351,   340,   330,   319,   310,
      300,   291,   282,   273,   265,   257,   249,   241,
      234,   227,   220,   213,   206,   200,   194,   188,
      182,   176,   171,   166,   161,   156,   151,   146,
      142,   137,   133,   129,   125,   121,   118,   114,
      110,   107,   104,   101,    97,    94,    92,    89,
       86,    83,    81,    78,    76,    74,    71,    69,
       67,    65,    63,    61,    59,    57,    56,    54,
       52,    51,    49,    47,    46,    45,    43,    42,
       41,    39,    38,    37,    36,    35,    34,    33,
       32,    31,    30,    29,    28,    27,    26,    25,
       25,    24,    23,    22,    22,    21,    20,    20,
       19,    19,    18,    17,    17,    16,    16,    15,
       15,    14,    14,    14,    13,    13,    12,    12,
       12,    11,    11,    11,    10,    10,    10,     9,
        9,     9,     9,     8,     8,     8,     8,     7,
        7,     7,     7,     6,     6,     6,     6,     6,
};

// Deterministic INT8 softmax.
//   logits  : int8 input, length n  (along the softmax axis)
//   output  : int8, length n, in [0, 127]
// Algorithm:
//   diff[i]  = clamp(max - logits[i], 0..255)
//   exp_q[i] = SOFTMAX_LUT[diff[i]]                        (int32)
//   sum_q    = sum(exp_q)
//   out[i]   = sat_int8((exp_q[i] * 127) / max(sum_q, 1))   (round-to-zero)
inline void softmax_int8(const int8_t* logits, int8_t* out, std::size_t n) {
    int max_v = -128;
    for (std::size_t i = 0; i < n; ++i)
        if (logits[i] > max_v) max_v = logits[i];

    int64_t sum_q = 0;
    int32_t* exp_q = new int32_t[n];
    for (std::size_t i = 0; i < n; ++i) {
        int diff = max_v - int(logits[i]);
        if (diff < 0)   diff = 0;
        if (diff > 255) diff = 255;
        exp_q[i] = SOFTMAX_LUT[diff];
        sum_q   += exp_q[i];
    }
    if (sum_q == 0) sum_q = 1;
    for (std::size_t i = 0; i < n; ++i) {
        int64_t v = (int64_t(exp_q[i]) * 127) / sum_q;
        if (v > 127) v = 127;
        if (v < 0)   v = 0;
        out[i] = int8_t(v);
    }
    delete[] exp_q;
}

}  // namespace mdla7
