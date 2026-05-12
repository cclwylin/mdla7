#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

static int32_t clamp_i32(int64_t value) {
    if (value < INT32_MIN) return INT32_MIN;
    if (value > INT32_MAX) return INT32_MAX;
    return static_cast<int32_t>(value);
}

static int32_t saturating_doubling_high_mul(int32_t a, int32_t b) {
    if (a == INT32_MIN && b == INT32_MIN) return INT32_MAX;
    const int64_t p = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    const int64_t nudge = (p >= 0) ? (1LL << 30) : (1LL - (1LL << 30));
    return clamp_i32((p + nudge) >> 31);
}

static int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    const int64_t mask = (1LL << exponent) - 1LL;
    const int64_t remainder = static_cast<int64_t>(x) & mask;
    const int64_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    const int64_t shifted = static_cast<int64_t>(x) >> exponent;
    return clamp_i32(shifted + (remainder > threshold ? 1 : 0));
}

static int32_t mbqm(int32_t x, int32_t multiplier, int32_t shift) {
    const int left_shift = shift > 0 ? shift : 0;
    const int right_shift = shift > 0 ? 0 : -shift;
    const int32_t shifted = left_shift > 0 ? clamp_i32(static_cast<int64_t>(x) << left_shift) : x;
    return rounding_divide_by_pot(saturating_doubling_high_mul(shifted, multiplier), right_shift);
}

static int8_t lane_i8(int32_t word, int lane) {
    return static_cast<int8_t>((static_cast<uint32_t>(word) >> (lane * 8)) & 0xffU);
}

static uint16_t lane_u16(const int32_t words[4], int lane) {
    const int byte_lane = lane * 2;
    const uint16_t lo = (static_cast<uint32_t>(words[byte_lane / 4]) >> ((byte_lane % 4) * 8)) & 0xffU;
    const uint16_t hi = (static_cast<uint32_t>(words[(byte_lane + 1) / 4]) >> (((byte_lane + 1) % 4) * 8)) & 0xffU;
    return static_cast<uint16_t>(lo | (hi << 8));
}

static double fp16_to_double(uint16_t bits) {
    const int sign = (bits >> 15) & 1;
    const int exp = (bits >> 10) & 0x1f;
    const int mant = bits & 0x3ff;
    double value = 0.0;
    if (exp == 0) {
        value = mant == 0 ? 0.0 : std::ldexp(static_cast<double>(mant) / 1024.0, -14);
    } else if (exp == 31) {
        value = 0.0;
    } else {
        value = std::ldexp(1.0 + static_cast<double>(mant) / 1024.0, exp - 15);
    }
    return sign ? -value : value;
}

static int64_t double_bits(double value) {
    int64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace

extern "C" void mdla7_dpi_conv_int8_mac(
    int32_t act0, int32_t act1, int32_t act2, int32_t act3,
    int32_t wgt0, int32_t wgt1, int32_t wgt2, int32_t wgt3,
    int32_t elem_count, int32_t zp_in, int32_t bias,
    int32_t multiplier, int32_t shift, int32_t zp_out,
    int32_t act_min, int32_t act_max,
    int32_t* acc_out, int32_t* scaled_out, int32_t* out_q) {
    const int32_t act_words[4] = {act0, act1, act2, act3};
    const int32_t wgt_words[4] = {wgt0, wgt1, wgt2, wgt3};
    int64_t acc = bias;
    const int lanes = elem_count < 0 ? 0 : (elem_count > 16 ? 16 : elem_count);
    for (int i = 0; i < lanes; ++i) {
        const int32_t av = static_cast<int32_t>(lane_i8(act_words[i / 4], i % 4)) - zp_in;
        const int32_t wv = static_cast<int32_t>(lane_i8(wgt_words[i / 4], i % 4));
        acc += static_cast<int64_t>(av) * static_cast<int64_t>(wv);
    }
    const int32_t clamped_acc = clamp_i32(acc);
    int32_t q = mbqm(clamped_acc, multiplier, static_cast<int8_t>(shift & 0xff)) + zp_out;
    if (q < act_min) q = act_min;
    if (q > act_max) q = act_max;
    *acc_out = clamped_acc;
    *scaled_out = q;
    *out_q = static_cast<int8_t>(q & 0xff);
}

extern "C" void mdla7_dpi_pool_fp16(
    int32_t vec0, int32_t vec1, int32_t vec2, int32_t vec3,
    int32_t elem_count, int32_t avg_mode, int64_t* out_bits) {
    const int32_t words[4] = {vec0, vec1, vec2, vec3};
    const int lanes = elem_count < 0 ? 0 : (elem_count > 8 ? 8 : elem_count);
    double sum = 0.0;
    double max_value = -1.0e300;
    for (int i = 0; i < lanes; ++i) {
        const double value = fp16_to_double(lane_u16(words, i));
        sum += value;
        if (value > max_value) max_value = value;
    }
    const double result = avg_mode ? (lanes == 0 ? 0.0 : sum / lanes) : max_value;
    *out_bits = double_bits(result);
}

extern "C" void mdla7_dpi_ewe_fp16(
    int32_t avec0, int32_t avec1, int32_t avec2, int32_t avec3,
    int32_t bvec0, int32_t bvec1, int32_t bvec2, int32_t bvec3,
    int32_t elem_count, int32_t op_mode, int64_t* out_bits) {
    const int32_t aw[4] = {avec0, avec1, avec2, avec3};
    const int32_t bw[4] = {bvec0, bvec1, bvec2, bvec3};
    const int lanes = elem_count < 0 ? 0 : (elem_count > 8 ? 8 : elem_count);
    double sum = 0.0;
    for (int i = 0; i < lanes; ++i) {
        const double a = fp16_to_double(lane_u16(aw, i));
        const double b = fp16_to_double(lane_u16(bw, i));
        switch (op_mode & 3) {
            case 1: sum += a * b; break;
            case 2: sum += a - b; break;
            case 3: sum += 1.0 / (1.0 + std::exp(-a)); break;
            default: sum += a + b; break;
        }
    }
    *out_bits = double_bits(sum);
}
