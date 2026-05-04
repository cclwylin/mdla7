#pragma once

// TFLite-style fixed-point requantization (gemmlowp primitives).
// Used by RequantEngine to turn int32 partial sums into int8 outputs that
// are bit-exact against the numpy reference in scripts/compile_model.py.

#include <cstdint>
#include <climits>

namespace mdla7 {

// (a * b * 2 + 2^30) >> 31, saturated to int32.
inline int32_t saturating_doubling_high_mul(int32_t a, int32_t b) {
    int64_t x = static_cast<int64_t>(a) * b;
    int64_t r = (x + (1LL << 30)) >> 31;
    if (r > INT32_MAX) r = INT32_MAX;
    if (r < INT32_MIN) r = INT32_MIN;
    return static_cast<int32_t>(r);
}

// Signed right-shift with TFLite's "round half away from zero".
inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    const int32_t mask      = (int32_t(1) << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

// Apply a (Q0.31 multiplier, shift) pair produced by QuantizeMultiplier().
//   shift > 0  → effective_scale > 1, do a left shift first
//   shift <= 0 → right shift after the doubling-high mul
inline int32_t multiply_by_quantized_multiplier(int32_t x,
                                                int32_t mult,
                                                int     shift) {
    int left_shift  = shift > 0 ?  shift : 0;
    int right_shift = shift > 0 ?  0     : -shift;
    int32_t shifted = (left_shift > 0) ? (x << left_shift) : x;
    return rounding_divide_by_pot(saturating_doubling_high_mul(shifted, mult),
                                  right_shift);
}

inline int8_t saturate_to_int8(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return static_cast<int8_t>(x);
}

}  // namespace mdla7
