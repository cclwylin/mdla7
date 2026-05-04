#pragma once

// Software-only IEEE 754 binary16 / brain-float-16 conversions.
// Used by compile_model.py and the FP CONV path so we don't depend on
// compiler-specific _Float16 / __bf16 extensions for portability.

#include <cstdint>
#include <cstring>

namespace mdla7 {

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
    const uint32_t exp  = (uint32_t(h) & 0x7C00u) >> 10;
    const uint32_t mant =  uint32_t(h) & 0x03FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            // Subnormal — normalize.
            uint32_t m = mant; int s = 0;
            while ((m & 0x0400u) == 0) { m <<= 1; ++s; }
            m &= 0x03FFu;
            bits = sign | (uint32_t(127 - 15 - s + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {                                  // inf / NaN
        bits = sign | 0x7F800000u | (mant << 13);
    } else {                                                    // normal
        bits = sign | (uint32_t(int(exp) + 127 - 15) << 23) | (mant << 13);
    }
    float f; std::memcpy(&f, &bits, 4); return f;
}

inline uint16_t fp32_to_fp16(float f) {
    // IEEE 754 binary32 → binary16 with round-to-nearest-even (matches numpy
    // `arr.astype(np.float16)`).  v8.15: previously truncated, which drifted
    // by 1 ulp for ~50% of outputs vs the numpy reference; alignment is
    // required for the fused-FP chain to stay bit-exact across layers.
    uint32_t bits; std::memcpy(&bits, &f, 4);
    const uint32_t sign      = (bits >> 16) & 0x8000u;
    const uint32_t exp_in    = (bits >> 23) & 0xFFu;
    const uint32_t mant_in   = bits & 0x007FFFFFu;

    // Inf / NaN: keep as inf (sign-preserving); for NaN, propagate a quiet NaN
    // payload (top mantissa bit set) so we never produce a signaling result.
    if (exp_in == 0xFFu) {
        return uint16_t(sign | 0x7C00u | (mant_in ? 0x0200u : 0u));
    }

    int32_t exp = int32_t(exp_in) - 127 + 15;

    if (exp >= 31) {
        // Magnitude too large for binary16 → +/-inf.
        return uint16_t(sign | 0x7C00u);
    }

    if (exp <= 0) {
        // Subnormal binary16 (or underflow to 0). Far-underflow short-circuit
        // saves the round-bit math.
        if (exp < -10) return uint16_t(sign);
        // 24-bit significand (implicit 1 + 23 mantissa bits).
        const uint32_t m_full   = mant_in | 0x00800000u;
        const uint32_t shift    = uint32_t(1 - exp + 13);   // bits to drop
        const uint32_t result   = m_full >> shift;
        const uint32_t round_lo = m_full & ((1u << shift) - 1u);
        const uint32_t halfway  = 1u << (shift - 1);
        uint32_t out = result;
        if (round_lo > halfway || (round_lo == halfway && (result & 1u))) {
            ++out;   // may carry into the implicit bit, encoding a normal — that's fine,
                     // the encoded bits just look like exp=1, mant=0 which IS the smallest
                     // normal binary16 (= smallest non-subnormal). Correct by construction.
        }
        return uint16_t(sign | out);
    }

    // Normal range: round 23-bit mantissa down to 10 bits.
    uint32_t mant_out      = mant_in >> 13;
    const uint32_t round_lo = mant_in & 0x1FFFu;          // 13 dropped bits
    if (round_lo > 0x1000u || (round_lo == 0x1000u && (mant_out & 1u))) {
        ++mant_out;
        if (mant_out == 0x400u) {
            // Mantissa rounded up past 0x3FF → carry into exponent.
            mant_out = 0u;
            ++exp;
            if (exp >= 31) return uint16_t(sign | 0x7C00u);    // overflow → inf
        }
    }
    return uint16_t(sign | (uint32_t(exp) << 10) | mant_out);
}

inline float bf16_to_fp32(uint16_t b) {
    uint32_t bits = uint32_t(b) << 16;
    float f; std::memcpy(&f, &bits, 4); return f;
}

inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4);
    return uint16_t(bits >> 16);
}

inline bool is_fp_dtype(uint8_t dt) {
    return dt == /*DT_FP8 */8 || dt == /*DT_FP16*/9 || dt == /*DT_BFP16*/10;
}

}   // namespace mdla7
