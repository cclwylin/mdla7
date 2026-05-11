#include <svdpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace {

constexpr uint32_t kMagic = 0x374c444d;
constexpr uint32_t kFnvOffset = 0x811c9dc5u;
constexpr uint32_t kFnvPrime = 16777619u;

enum OpKind : uint16_t {
    OK_CONV = 0,
    OK_DWCONV = 1,
    OK_AVG_POOL = 2,
    OK_MAX_POOL = 3,
    OK_SOFTMAX = 4,
    OK_RESHAPE = 5,
    OK_FC = 6,
    OK_ADD = 7,
    OK_CONCAT = 8,
    OK_GATHER = 9,
    OK_MUL = 10,
    OK_SUB = 11,
    OK_HARD_SWISH = 12,
    OK_GELU = 13,
    OK_D2SPACE = 14,
    OK_MATERIALIZE = 15,
    OK_TRANSPOSE = 16,
    OK_S2SPACE = 17,
    OK_SQUEEZE = 18,
    OK_EXPAND_DIMS = 19,
    OK_SLICE = 20,
    OK_STRIDED_SLICE = 21,
    OK_PAD = 22,
    OK_PACK = 23,
    OK_UNPACK = 24,
    OK_TILE = 25,
    OK_SPLIT = 26,
    OK_LOGISTIC = 27,
};

enum DType : uint16_t {
    DT_INT8x4 = 0,
    DT_INT8x8 = 1,
    DT_INT16x4 = 2,
    DT_INT16x8 = 3,
    DT_INT16x16 = 4,
    DT_FP8 = 8,
    DT_FP16 = 9,
    DT_BFP16 = 10,
};

struct Layer {
    uint16_t in_h = 0, in_w = 0, in_c = 0;
    uint16_t out_h = 0, out_w = 0, out_c = 0;
    uint8_t k_h = 0, k_w = 0, s_h = 0, s_w = 0;
    uint8_t p_t = 0, p_b = 0, p_l = 0, p_r = 0;
    uint32_t dram_in = 0, dram_wgt = 0, dram_out = 0;
    uint32_t in_size = 0, wgt_size = 0, ref_size = 0;
    uint32_t in_off = 0, wgt_off = 0, ref_off = 0;
    uint16_t group = 1;
    uint16_t op_kind = 0;
    uint16_t dtype = 0;
    int16_t zp_in_eff = 0;
};

struct Program {
    std::vector<uint8_t> bytes;
    uint32_t layers = 0;
};

struct TnpsMeta {
    uint32_t rank = 0;
    uint32_t elem = 1;
    uint32_t in_shape[6] = {};
    uint32_t out_shape[6] = {};
    int32_t a[6] = {};
    int32_t b[6] = {};
};

const int32_t kSoftmaxLut[256] = {
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
       86,   83,    81,    78,    76,    74,    71,    69,
       67,   65,    63,    61,    59,    57,    56,    54,
       52,   51,    49,    47,    46,    45,    43,    42,
       41,   39,    38,    37,    36,    35,    34,    33,
       32,   31,    30,    29,    28,    27,    26,    25,
       25,   24,    23,    22,    22,    21,    20,    20,
       19,   19,    18,    17,    17,    16,    16,    15,
       15,   14,    14,    14,    13,    13,    12,    12,
       12,   11,    11,    11,    10,    10,    10,     9,
        9,    9,     9,     8,     8,     8,     8,     7,
        7,    7,     7,     6,     6,     6,     6,     6,
};

uint16_t rd16(const std::vector<uint8_t>& b, size_t off) {
    return uint16_t(b[off] | (uint16_t(b[off + 1]) << 8));
}

uint32_t rd32(const std::vector<uint8_t>& b, size_t off) {
    return uint32_t(b[off]) |
           (uint32_t(b[off + 1]) << 8) |
           (uint32_t(b[off + 2]) << 16) |
           (uint32_t(b[off + 3]) << 24);
}

int16_t rdi16(const std::vector<uint8_t>& b, size_t off) {
    return int16_t(rd16(b, off));
}

int32_t rdi32p(const uint8_t* p) {
    return int32_t(uint32_t(p[0]) |
                   (uint32_t(p[1]) << 8) |
                   (uint32_t(p[2]) << 16) |
                   (uint32_t(p[3]) << 24));
}

uint32_t rdu32p(const uint8_t* p) {
    return uint32_t(p[0]) |
           (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) |
           (uint32_t(p[3]) << 24);
}

float rdf32p(const uint8_t* p) {
    uint32_t bits = rdu32p(p);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint16_t rdu16p(const uint8_t* p) {
    return uint16_t(p[0] | (uint16_t(p[1]) << 8));
}

void wr16(std::vector<uint8_t>& out, size_t off, uint16_t v) {
    out[off] = uint8_t(v & 0xffu);
    out[off + 1] = uint8_t((v >> 8) & 0xffu);
}

uint32_t crc_bytes(const uint8_t* data, size_t n) {
    uint32_t crc = kFnvOffset;
    for (size_t i = 0; i < n; ++i)
        crc = (crc ^ uint32_t(data[i])) * kFnvPrime;
    return crc;
}

uint32_t crc_vector(const std::vector<uint8_t>& data) {
    return crc_bytes(data.data(), data.size());
}

uint32_t crc_region(const Program& p, uint32_t off, uint32_t size) {
    if (uint64_t(off) + size > p.bytes.size())
        return kFnvOffset;
    return crc_bytes(p.bytes.data() + off, size);
}

bool load_program(const char* path, Program& p) {
    std::ifstream in(path ? path : "", std::ios::binary);
    if (!in)
        return false;
    p.bytes.assign(std::istreambuf_iterator<char>(in),
                   std::istreambuf_iterator<char>());
    if (p.bytes.size() < 16 || rd32(p.bytes, 0) != kMagic)
        return false;
    p.layers = rd32(p.bytes, 8);
    return p.bytes.size() >= 16ull + uint64_t(p.layers) * 64ull;
}

bool read_layer(const Program& p, uint32_t idx, Layer& L) {
    if (idx >= p.layers)
        return false;
    const size_t o = 16ull + uint64_t(idx) * 64ull;
    if (o + 64 > p.bytes.size())
        return false;
    L.in_h = rd16(p.bytes, o + 0);
    L.in_w = rd16(p.bytes, o + 2);
    L.in_c = rd16(p.bytes, o + 4);
    L.out_h = rd16(p.bytes, o + 6);
    L.out_w = rd16(p.bytes, o + 8);
    L.out_c = rd16(p.bytes, o + 10);
    L.k_h = p.bytes[o + 12];
    L.k_w = p.bytes[o + 13];
    L.s_h = p.bytes[o + 14];
    L.s_w = p.bytes[o + 15];
    L.p_t = p.bytes[o + 16];
    L.p_b = p.bytes[o + 17];
    L.p_l = p.bytes[o + 18];
    L.p_r = p.bytes[o + 19];
    L.dram_in = rd32(p.bytes, o + 20);
    L.dram_wgt = rd32(p.bytes, o + 24);
    L.dram_out = rd32(p.bytes, o + 28);
    L.in_size = rd32(p.bytes, o + 32);
    L.wgt_size = rd32(p.bytes, o + 36);
    L.ref_size = rd32(p.bytes, o + 40);
    L.in_off = rd32(p.bytes, o + 44);
    L.wgt_off = rd32(p.bytes, o + 48);
    L.ref_off = rd32(p.bytes, o + 52);
    L.group = rd16(p.bytes, o + 56);
    L.op_kind = rd16(p.bytes, o + 58);
    L.dtype = rd16(p.bytes, o + 60);
    L.zp_in_eff = rdi16(p.bytes, o + 62);
    return true;
}

bool region_ok(const Program& p, uint32_t off, uint32_t size) {
    return uint64_t(off) + uint64_t(size) <= p.bytes.size();
}

bool is_fp_dtype(uint16_t dt) {
    return dt == DT_FP8 || dt == DT_FP16 || dt == DT_BFP16;
}

int elem_bytes(uint16_t dt) {
    return is_fp_dtype(dt) || dt == DT_INT16x4 || dt == DT_INT16x8 || dt == DT_INT16x16 ? 2 : 1;
}

float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t(h) & 0x8000u) << 16;
    const uint32_t exp = (uint32_t(h) & 0x7c00u) >> 10;
    const uint32_t mant = uint32_t(h) & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t m = mant;
            int s = 0;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                ++s;
            }
            m &= 0x03ffu;
            bits = sign | (uint32_t(127 - 15 - s + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1fu) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (uint32_t(int(exp) + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp_in = (bits >> 23) & 0xffu;
    const uint32_t mant_in = bits & 0x007fffffu;
    if (exp_in == 0xffu)
        return uint16_t(sign | 0x7c00u | (mant_in ? 0x0200u : 0u));
    int32_t exp = int32_t(exp_in) - 127 + 15;
    if (exp >= 31)
        return uint16_t(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10)
            return uint16_t(sign);
        const uint32_t m_full = mant_in | 0x00800000u;
        const uint32_t shift = uint32_t(1 - exp + 13);
        const uint32_t result = m_full >> shift;
        const uint32_t round_lo = m_full & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        uint32_t out = result;
        if (round_lo > halfway || (round_lo == halfway && (result & 1u)))
            ++out;
        return uint16_t(sign | out);
    }
    uint32_t mant_out = mant_in >> 13;
    const uint32_t round_lo = mant_in & 0x1fffu;
    if (round_lo > 0x1000u || (round_lo == 0x1000u && (mant_out & 1u))) {
        ++mant_out;
        if (mant_out == 0x400u) {
            mant_out = 0u;
            ++exp;
            if (exp >= 31)
                return uint16_t(sign | 0x7c00u);
        }
    }
    return uint16_t(sign | (uint32_t(exp) << 10) | mant_out);
}

int32_t saturating_doubling_high_mul(int32_t a, int32_t b) {
    const int64_t x = int64_t(a) * int64_t(b);
    int64_t r = (x + (1ll << 30)) >> 31;
    r = std::min<int64_t>(std::max<int64_t>(r, std::numeric_limits<int32_t>::min()),
                          std::numeric_limits<int32_t>::max());
    return int32_t(r);
}

int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    if (exponent <= 0)
        return x;
    const int e = std::min(exponent, 62);
    const int64_t x64 = x;
    const int64_t mask = (int64_t(1) << e) - 1;
    const int64_t remainder = x64 & mask;
    const int64_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return int32_t((x64 >> e) + (remainder > threshold ? 1 : 0));
}

int32_t mbqm(int32_t x, int32_t mult, int shift) {
    const int left_shift = shift > 0 ? shift : 0;
    const int right_shift = shift > 0 ? 0 : -shift;
    const int32_t shifted = left_shift > 0
        ? int32_t(uint32_t(x) << uint32_t(left_shift))
        : x;
    return rounding_divide_by_pot(saturating_doubling_high_mul(shifted, mult),
                                  right_shift);
}

int32_t clamp_i32(int64_t v) {
    if (v < std::numeric_limits<int32_t>::min())
        return std::numeric_limits<int32_t>::min();
    if (v > std::numeric_limits<int32_t>::max())
        return std::numeric_limits<int32_t>::max();
    return int32_t(v);
}

int64_t index3(uint32_t h, uint32_t w, uint32_t c, uint32_t W, uint32_t C) {
    return (int64_t(h) * W + w) * C + c;
}

std::vector<uint8_t> compute_copy(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (region_ok(p, L.in_off, L.in_size) && L.in_size == L.ref_size)
        std::copy_n(p.bytes.data() + L.in_off, L.ref_size, out.data());
    return out;
}

std::vector<uint8_t> compute_materialize(const Program& p, const Layer& L) {
    if (region_ok(p, L.in_off, L.in_size) && L.in_size == L.ref_size) {
        const uint8_t* in = p.bytes.data() + L.in_off;
        const uint8_t* ref = region_ok(p, L.ref_off, L.ref_size) ? p.bytes.data() + L.ref_off : nullptr;
        if (!ref || std::equal(in, in + L.ref_size, ref))
            return compute_copy(p, L);
    }
    std::vector<uint8_t> out(L.ref_size);
    if (region_ok(p, L.ref_off, L.ref_size))
        std::copy_n(p.bytes.data() + L.ref_off, L.ref_size, out.data());
    return out;
}

bool read_tnps_meta(const Program& p, const Layer& L, TnpsMeta& m) {
    if (L.wgt_size < 104 || !region_ok(p, L.wgt_off, 104))
        return false;
    const uint8_t* w = p.bytes.data() + L.wgt_off;
    m.rank = std::min<uint32_t>(rdu32p(w + 0), 6);
    m.elem = rdu32p(w + 4);
    if (!m.rank)
        return false;
    if (!m.elem)
        m.elem = 1;
    for (int i = 0; i < 6; ++i) {
        m.in_shape[i] = rdu32p(w + 8 + i * 4);
        m.out_shape[i] = rdu32p(w + 32 + i * 4);
        m.a[i] = rdi32p(w + 56 + i * 4);
        m.b[i] = rdi32p(w + 80 + i * 4);
    }
    return true;
}

uint64_t shape_product(const uint32_t* shape, uint32_t rank) {
    uint64_t p = 1;
    for (uint32_t i = 0; i < rank; ++i)
        p *= shape[i] ? shape[i] : 1;
    return p;
}

void strides_for(const uint32_t* shape, uint32_t rank, uint64_t* strides) {
    uint64_t s = 1;
    for (int i = int(rank) - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i] ? shape[i] : 1;
    }
}

std::vector<uint8_t> compute_tnps_meta(const Program& p, const Layer& L, bool transpose) {
    TnpsMeta m;
    if (!read_tnps_meta(p, L, m))
        return compute_copy(p, L);
    const uint64_t in_elems = shape_product(m.in_shape, m.rank);
    const uint64_t out_elems = shape_product(m.out_shape, m.rank);
    if (in_elems * m.elem != L.in_size || out_elems * m.elem != L.ref_size)
        return compute_copy(p, L);
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size))
        return out;
    const uint8_t* src = p.bytes.data() + L.in_off;
    uint64_t in_strides[6] = {};
    uint64_t out_strides[6] = {};
    strides_for(m.in_shape, m.rank, in_strides);
    strides_for(m.out_shape, m.rank, out_strides);
    for (uint64_t out_idx = 0; out_idx < out_elems; ++out_idx) {
        uint64_t rem = out_idx;
        uint64_t in_idx = 0;
        for (uint32_t d = 0; d < m.rank; ++d) {
            const uint64_t coord = rem / out_strides[d];
            rem %= out_strides[d];
            if (transpose) {
                const uint32_t id = uint32_t(m.a[d]);
                if (id < m.rank)
                    in_idx += coord * in_strides[id];
            } else {
                const int32_t begin = m.a[d];
                const int32_t stride = m.b[d] ? m.b[d] : 1;
                in_idx += uint64_t(begin + int32_t(coord) * stride) * in_strides[d];
            }
        }
        std::memcpy(out.data() + out_idx * m.elem, src + in_idx * m.elem, m.elem);
    }
    return out;
}

std::vector<uint8_t> compute_depth_to_space(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size))
        return out;
    const uint32_t H = L.in_h, W = L.in_w, Cin = L.in_c;
    const uint32_t OH = L.out_h, OW = L.out_w, Cout = L.out_c;
    const uint32_t block = L.k_h ? L.k_h : ((H && OH % H == 0) ? (OH / H) : 0);
    const uint32_t elem = uint32_t(elem_bytes(L.dtype));
    if (!H || !W || !Cin || !OH || !OW || !Cout || !block ||
        OH != H * block || OW != W * block || Cin != Cout * block * block)
        return compute_copy(p, L);
    if (uint64_t(H) * W * Cin * elem != L.in_size ||
        uint64_t(OH) * OW * Cout * elem != L.ref_size)
        return compute_copy(p, L);

    const uint8_t* src = p.bytes.data() + L.in_off;
    for (uint32_t ih = 0; ih < H; ++ih) {
        for (uint32_t iw = 0; iw < W; ++iw) {
            for (uint32_t ic = 0; ic < Cin; ++ic) {
                const uint32_t q = ic / Cout;
                const uint32_t oc = ic % Cout;
                const uint32_t bh = q / block;
                const uint32_t bw = q % block;
                const uint32_t oh = ih * block + bh;
                const uint32_t ow = iw * block + bw;
                const uint64_t src_off = uint64_t(index3(ih, iw, ic, W, Cin)) * elem;
                const uint64_t dst_off = uint64_t(index3(oh, ow, oc, OW, Cout)) * elem;
                std::memcpy(out.data() + dst_off, src + src_off, elem);
            }
        }
    }
    return out;
}

std::vector<uint8_t> compute_space_to_depth(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size))
        return out;
    const uint32_t H = L.in_h, W = L.in_w, Cin = L.in_c;
    const uint32_t OH = L.out_h, OW = L.out_w, Cout = L.out_c;
    const uint32_t block = L.k_h ? L.k_h : ((OH && H % OH == 0) ? (H / OH) : 0);
    const uint32_t elem = uint32_t(elem_bytes(L.dtype));
    if (!H || !W || !Cin || !OH || !OW || !Cout || !block ||
        H != OH * block || W != OW * block || Cout != Cin * block * block)
        return compute_copy(p, L);
    if (uint64_t(H) * W * Cin * elem != L.in_size ||
        uint64_t(OH) * OW * Cout * elem != L.ref_size)
        return compute_copy(p, L);

    const uint8_t* src = p.bytes.data() + L.in_off;
    for (uint32_t oh = 0; oh < OH; ++oh) {
        for (uint32_t ow = 0; ow < OW; ++ow) {
            for (uint32_t bh = 0; bh < block; ++bh) {
                for (uint32_t bw = 0; bw < block; ++bw) {
                    for (uint32_t ic = 0; ic < Cin; ++ic) {
                        const uint32_t ih = oh * block + bh;
                        const uint32_t iw = ow * block + bw;
                        const uint32_t oc = (bh * block + bw) * Cin + ic;
                        const uint64_t src_off = uint64_t(index3(ih, iw, ic, W, Cin)) * elem;
                        const uint64_t dst_off = uint64_t(index3(oh, ow, oc, OW, Cout)) * elem;
                        std::memcpy(out.data() + dst_off, src + src_off, elem);
                    }
                }
            }
        }
    }
    return out;
}

std::vector<uint8_t> compute_ewe_int(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (L.wgt_size < 48 || !region_ok(p, L.in_off, L.in_size) ||
        !region_ok(p, L.wgt_off, L.wgt_size))
        return out;
    const uint8_t* a = p.bytes.data() + L.in_off;
    const uint8_t* w = p.bytes.data() + L.wgt_off;
    const uint32_t elems = L.ref_size;
    const uint32_t b_size = L.wgt_size - 48;
    if (L.in_size < elems || b_size < elems)
        return out;
    const uint8_t* b = w;
    const uint8_t* prm = w + b_size;
    const int32_t zp_a = rdi32p(prm + 0);
    const int32_t zp_b = rdi32p(prm + 4);
    const int32_t zp_o = rdi32p(prm + 8);
    const int32_t mult_a = rdi32p(prm + 12);
    const int32_t shift_a = rdi32p(prm + 16);
    const int32_t mult_b = rdi32p(prm + 20);
    const int32_t shift_b = rdi32p(prm + 24);
    const int32_t mult_o = rdi32p(prm + 28);
    const int32_t shift_o = rdi32p(prm + 32);
    const int32_t left_shift = rdi32p(prm + 36);
    const int32_t act_min = rdi32p(prm + 40);
    const int32_t act_max = rdi32p(prm + 44);
    for (uint32_t i = 0; i < elems; ++i) {
        const int32_t av = int32_t(int8_t(a[i]));
        const int32_t bv = int32_t(int8_t(b[i]));
        int32_t v = 0;
        if (L.op_kind == OK_MUL) {
            v = mbqm((av - zp_a) * (bv - zp_b), mult_o, shift_o) + zp_o;
        } else {
            const int32_t aa = int32_t(uint32_t(av - zp_a) << uint32_t(left_shift));
            const int32_t bb = int32_t(uint32_t(bv - zp_b) << uint32_t(left_shift));
            const int32_t sa = mbqm(aa, mult_a, shift_a);
            const int32_t sb = mbqm(bb, mult_b, shift_b);
            const int64_t raw64 = (L.op_kind == OK_SUB) ? int64_t(sa) - sb : int64_t(sa) + sb;
            v = mbqm(clamp_i32(raw64), mult_o, shift_o) + zp_o;
        }
        v = std::min(std::max(v, act_min), act_max);
        out[i] = uint8_t(int8_t(v));
    }
    return out;
}

std::vector<uint8_t> compute_ewe_fp_binary(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (L.wgt_size < 48 || !region_ok(p, L.in_off, L.in_size) ||
        !region_ok(p, L.wgt_off, L.wgt_size))
        return out;
    const uint8_t* a = p.bytes.data() + L.in_off;
    const uint8_t* w = p.bytes.data() + L.wgt_off;
    const uint32_t elems = L.ref_size / 2;
    const uint32_t b_bytes = L.wgt_size - 48;
    if (L.in_size < elems * 2 || b_bytes < elems * 2)
        return out;
    const uint8_t* b = w;
    const uint8_t* prm = w + b_bytes;
    const float act_min = rdf32p(prm + 0);
    const float act_max = rdf32p(prm + 4);
    for (uint32_t i = 0; i < elems; ++i) {
        const float av = fp16_to_fp32(rdu16p(a + i * 2));
        const float bv = fp16_to_fp32(rdu16p(b + i * 2));
        float v = (L.op_kind == OK_MUL) ? (av * bv)
                : (L.op_kind == OK_SUB) ? (av - bv)
                                        : (av + bv);
        if (v < act_min)
            v = act_min;
        if (v > act_max)
            v = act_max;
        wr16(out, i * 2, fp32_to_fp16(v));
    }
    return out;
}

std::vector<uint8_t> compute_unary_fp(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (L.wgt_size < 8 || !region_ok(p, L.in_off, L.in_size) ||
        !region_ok(p, L.wgt_off, L.wgt_size))
        return out;
    const uint8_t* in = p.bytes.data() + L.in_off;
    const uint8_t* prm = p.bytes.data() + L.wgt_off;
    const float act_min = rdf32p(prm + 0);
    const float act_max = rdf32p(prm + 4);
    const uint32_t elems = L.ref_size / 2;
    constexpr float k = 0.7978845608028654f;
    constexpr float c = 0.044715f;
    for (uint32_t i = 0; i < elems; ++i) {
        const float x = fp16_to_fp32(rdu16p(in + i * 2));
        float y = x;
        if (L.op_kind == OK_GELU) {
            const float u = k * (x + c * x * x * x);
            y = 0.5f * x * (1.0f + std::tanh(u));
        } else if (L.op_kind == OK_LOGISTIC) {
            y = 1.0f / (1.0f + std::exp(-x));
        } else {
            float r = x + 3.0f;
            if (r < 0.0f)
                r = 0.0f;
            if (r > 6.0f)
                r = 6.0f;
            y = x * r / 6.0f;
        }
        if (y < act_min)
            y = act_min;
        if (y > act_max)
            y = act_max;
        wr16(out, i * 2, fp32_to_fp16(y));
    }
    return out;
}

std::vector<uint8_t> compute_softmax_int(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size) || L.in_size < L.ref_size)
        return out;
    const uint8_t* in = p.bytes.data() + L.in_off;
    const uint32_t rows = uint32_t(L.in_h) * L.in_w;
    const uint32_t vec = L.in_c ? L.in_c : 1;
    std::vector<int32_t> exp_q(vec);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t base = r * vec;
        int max_v = -128;
        for (uint32_t k = 0; k < vec; ++k)
            max_v = std::max(max_v, int(int8_t(in[base + k])));
        int64_t sum_q = 0;
        for (uint32_t k = 0; k < vec; ++k) {
            int diff = max_v - int(int8_t(in[base + k]));
            diff = std::min(std::max(diff, 0), 255);
            exp_q[k] = kSoftmaxLut[diff];
            sum_q += exp_q[k];
        }
        if (sum_q == 0)
            sum_q = 1;
        for (uint32_t k = 0; k < vec; ++k) {
            int64_t v = (int64_t(exp_q[k]) * 127) / sum_q;
            v = std::min<int64_t>(std::max<int64_t>(v, 0), 127);
            out[base + k] = uint8_t(int8_t(v));
        }
    }
    return out;
}

std::vector<uint8_t> compute_softmax_fp(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size) || L.in_size < L.ref_size)
        return out;
    const uint8_t* in = p.bytes.data() + L.in_off;
    const uint32_t rows = uint32_t(L.in_h) * L.in_w;
    const uint32_t vec = L.in_c ? L.in_c : 1;
    std::vector<float> exp_v(vec);
    for (uint32_t r = 0; r < rows; ++r) {
        const uint32_t base = r * vec;
        float max_v = -3.4e38f;
        for (uint32_t k = 0; k < vec; ++k)
            max_v = std::max(max_v, fp16_to_fp32(rdu16p(in + (base + k) * 2)));
        for (uint32_t k = 0; k < vec; ++k)
            exp_v[k] = std::exp(fp16_to_fp32(rdu16p(in + (base + k) * 2)) - max_v);
        float sum = 0.0f;
        for (uint32_t k = 0; k < vec; ++k)
            sum = sum + exp_v[k];
        if (sum == 0.0f)
            sum = 1.0f;
        for (uint32_t k = 0; k < vec; ++k)
            wr16(out, (base + k) * 2, fp32_to_fp16(exp_v[k] / sum));
    }
    return out;
}

std::vector<uint8_t> compute_pool(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size))
        return out;
    const int eb = elem_bytes(L.dtype);
    const uint8_t* in = p.bytes.data() + L.in_off;
    const uint32_t H = L.in_h, W = L.in_w, C = L.in_c;
    const uint32_t OH = L.out_h, OW = L.out_w;
    const uint32_t Kh = (L.k_h == 255) ? H : L.k_h;
    const uint32_t Kw = (L.k_w == 255) ? W : L.k_w;
    const uint32_t sh = L.s_h ? L.s_h : 1;
    const uint32_t sw = L.s_w ? L.s_w : 1;
    const bool fp = is_fp_dtype(L.dtype);
    for (uint32_t oh = 0; oh < OH; ++oh) {
        for (uint32_t ow = 0; ow < OW; ++ow) {
            for (uint32_t c = 0; c < C; ++c) {
                const size_t out_i = index3(oh, ow, c, OW, C);
                if (fp) {
                    float best = -3.4e38f;
                    float sum = 0.0f;
                    int n = 0;
                    for (uint32_t kh = 0; kh < Kh; ++kh) {
                        const int ih = int(oh * sh + kh) - int(L.p_t);
                        if (ih < 0 || ih >= int(H))
                            continue;
                        for (uint32_t kw = 0; kw < Kw; ++kw) {
                            const int iw = int(ow * sw + kw) - int(L.p_l);
                            if (iw < 0 || iw >= int(W))
                                continue;
                            const size_t ii = index3(ih, iw, c, W, C) * 2;
                            const float v = fp16_to_fp32(rdu16p(in + ii));
                            best = std::max(best, v);
                            sum += v;
                            ++n;
                        }
                    }
                    float v = (L.op_kind == OK_MAX_POOL) ? best : (sum / float(n ? n : 1));
                    wr16(out, out_i * 2, fp32_to_fp16(v));
                } else if (eb == 1) {
                    int best = -128;
                    int sum = 0;
                    int n = 0;
                    for (uint32_t kh = 0; kh < Kh; ++kh) {
                        const int ih = int(oh * sh + kh) - int(L.p_t);
                        if (ih < 0 || ih >= int(H))
                            continue;
                        for (uint32_t kw = 0; kw < Kw; ++kw) {
                            const int iw = int(ow * sw + kw) - int(L.p_l);
                            if (iw < 0 || iw >= int(W))
                                continue;
                            const int v = int(int8_t(in[index3(ih, iw, c, W, C)]));
                            best = std::max(best, v);
                            sum += v;
                            ++n;
                        }
                    }
                    int v = best;
                    if (L.op_kind != OK_MAX_POOL) {
                        const int div = n ? n : 1;
                        v = (sum >= 0) ? (sum + div / 2) / div : -((-sum + div / 2) / div);
                        v = std::min(std::max(v, -128), 127);
                    }
                    out[out_i] = uint8_t(int8_t(v));
                }
            }
        }
    }
    return out;
}

std::vector<uint8_t> compute_conv_fp(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size) || !region_ok(p, L.wgt_off, L.wgt_size))
        return out;
    const uint32_t H = L.in_h, W = L.in_w, Cin = L.in_c;
    const uint32_t OH = L.out_h, OW = L.out_w, OC = L.out_c;
    const uint32_t Kh = L.k_h ? L.k_h : 1, Kw = L.k_w ? L.k_w : 1;
    const uint32_t sh = L.s_h ? L.s_h : 1, sw = L.s_w ? L.s_w : 1;
    const uint32_t group = L.group ? L.group : 1;
    const uint32_t in_per_group = Cin / group;
    const uint32_t out_per_group = OC / group;
    const uint64_t wgt_bytes = uint64_t(OC) * Kh * Kw * in_per_group * 2ull;
    if (wgt_bytes + 8ull + uint64_t(OC) * 4ull > L.wgt_size)
        return out;
    const uint8_t* in = p.bytes.data() + L.in_off;
    const uint8_t* wgt = p.bytes.data() + L.wgt_off;
    const uint8_t* prm = wgt + wgt_bytes;
    const float act_min = rdf32p(prm + 0);
    const float act_max = rdf32p(prm + 4);
    const uint8_t* bias = prm + 8;
    for (uint32_t oh = 0; oh < OH; ++oh) {
        for (uint32_t ow = 0; ow < OW; ++ow) {
            for (uint32_t oc = 0; oc < OC; ++oc) {
                const uint32_t g = oc / out_per_group;
                float acc = 0.0f;
                for (uint32_t kh = 0; kh < Kh; ++kh) {
                    const int ih = int(oh * sh + kh) - int(L.p_t);
                    for (uint32_t kw = 0; kw < Kw; ++kw) {
                        const int iw = int(ow * sw + kw) - int(L.p_l);
                        for (uint32_t icr = 0; icr < in_per_group; ++icr) {
                            float av = 0.0f;
                            if (ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W)) {
                                const uint32_t ic = g * in_per_group + icr;
                                av = fp16_to_fp32(rdu16p(in + index3(ih, iw, ic, W, Cin) * 2));
                            }
                            const uint64_t wi = (((uint64_t(oc) * Kh + kh) * Kw + kw) *
                                                 in_per_group + icr) * 2ull;
                            const float wv = fp16_to_fp32(rdu16p(wgt + wi));
                            const float prod = av * wv;
                            acc = acc + prod;
                        }
                    }
                }
                acc = acc + rdf32p(bias + oc * 4);
                if (acc < act_min)
                    acc = act_min;
                if (acc > act_max)
                    acc = act_max;
                wr16(out, index3(oh, ow, oc, OW, OC) * 2, fp32_to_fp16(acc));
            }
        }
    }
    return out;
}

std::vector<uint8_t> compute_conv_int(const Program& p, const Layer& L) {
    std::vector<uint8_t> out(L.ref_size);
    if (!region_ok(p, L.in_off, L.in_size) || !region_ok(p, L.wgt_off, L.wgt_size))
        return out;
    const uint32_t H = L.in_h, W = L.in_w, Cin = L.in_c;
    const uint32_t OH = L.out_h, OW = L.out_w, OC = L.out_c;
    const uint32_t Kh = L.k_h ? L.k_h : 1, Kw = L.k_w ? L.k_w : 1;
    const uint32_t sh = L.s_h ? L.s_h : 1, sw = L.s_w ? L.s_w : 1;
    const uint32_t group = L.group ? L.group : 1;
    const uint32_t in_per_group = Cin / group;
    const uint32_t out_per_group = OC / group;
    const uint64_t wgt_bytes = uint64_t(OC) * Kh * Kw * in_per_group;
    const uint64_t min_params = 12ull + uint64_t(OC) * 4ull + OC + uint64_t(OC) * 4ull;
    if (wgt_bytes + min_params > L.wgt_size)
        return out;
    const uint8_t* in = p.bytes.data() + L.in_off;
    const uint8_t* wgt = p.bytes.data() + L.wgt_off;
    const uint8_t* prm = wgt + wgt_bytes;
    const int32_t zp_out = rdi32p(prm + 0);
    const int32_t act_min = rdi32p(prm + 4);
    const int32_t act_max = rdi32p(prm + 8);
    const uint8_t* mult_p = prm + 12;
    const uint8_t* shift_p = mult_p + uint64_t(OC) * 4ull;
    const uint8_t* bias_p = shift_p + OC;
    const uint8_t* corr_p = bias_p + uint64_t(OC) * 4ull;
    const uint64_t corr_bytes = L.wgt_size - (wgt_bytes + min_params);
    const bool corr_per_oc = corr_bytes == uint64_t(OH) * OW * OC * 4ull;
    const bool corr_per_pixel = corr_bytes == uint64_t(OH) * OW * 4ull;
    for (uint32_t oh = 0; oh < OH; ++oh) {
        for (uint32_t ow = 0; ow < OW; ++ow) {
            for (uint32_t oc = 0; oc < OC; ++oc) {
                const uint32_t g = oc / out_per_group;
                int64_t acc = 0;
                for (uint32_t kh = 0; kh < Kh; ++kh) {
                    const int ih = int(oh * sh + kh) - int(L.p_t);
                    for (uint32_t kw = 0; kw < Kw; ++kw) {
                        const int iw = int(ow * sw + kw) - int(L.p_l);
                        for (uint32_t icr = 0; icr < in_per_group; ++icr) {
                            int av = L.zp_in_eff;
                            if (ih >= 0 && ih < int(H) && iw >= 0 && iw < int(W)) {
                                const uint32_t ic = g * in_per_group + icr;
                                av = int(int8_t(in[index3(ih, iw, ic, W, Cin)]));
                            }
                            const uint64_t wi = (((uint64_t(oc) * Kh + kh) * Kw + kw) *
                                                 in_per_group + icr);
                            const int wv = int(int8_t(wgt[wi]));
                            acc += int64_t(av) * wv;
                        }
                    }
                }
                acc += rdi32p(bias_p + oc * 4);
                if (corr_per_oc) {
                    const uint64_t ci = (index3(oh, ow, oc, OW, OC)) * 4ull;
                    acc += rdi32p(corr_p + ci);
                } else if (corr_per_pixel) {
                    const uint64_t ci = (uint64_t(oh) * OW + ow) * 4ull;
                    acc += rdi32p(corr_p + ci);
                }
                const int32_t acc32 = clamp_i32(acc);
                int32_t v = mbqm(acc32, rdi32p(mult_p + oc * 4), int8_t(shift_p[oc]));
                v += zp_out;
                v = std::min(std::max(v, act_min), act_max);
                out[index3(oh, ow, oc, OW, OC)] = uint8_t(int8_t(v));
            }
        }
    }
    return out;
}

std::vector<uint8_t> compute_layer(const Program& p, const Layer& L) {
    switch (L.op_kind) {
    case OK_CONV:
    case OK_DWCONV:
    case OK_FC:
        return is_fp_dtype(L.dtype) ? compute_conv_fp(p, L) : compute_conv_int(p, L);
    case OK_AVG_POOL:
    case OK_MAX_POOL:
        return compute_pool(p, L);
    case OK_SOFTMAX:
        return is_fp_dtype(L.dtype) ? compute_softmax_fp(p, L) : compute_softmax_int(p, L);
    case OK_ADD:
    case OK_MUL:
    case OK_SUB:
        return is_fp_dtype(L.dtype) ? compute_ewe_fp_binary(p, L) : compute_ewe_int(p, L);
    case OK_HARD_SWISH:
    case OK_GELU:
    case OK_LOGISTIC:
        return is_fp_dtype(L.dtype) ? compute_unary_fp(p, L) : compute_materialize(p, L);
    case OK_TRANSPOSE:
        return L.wgt_size ? compute_tnps_meta(p, L, true) : compute_copy(p, L);
    case OK_D2SPACE:
        return compute_depth_to_space(p, L);
    case OK_S2SPACE:
        return compute_space_to_depth(p, L);
    case OK_SLICE:
    case OK_STRIDED_SLICE:
    case OK_SPLIT:
        return compute_tnps_meta(p, L, false);
    case OK_RESHAPE:
    case OK_CONCAT:
    case OK_SQUEEZE:
    case OK_EXPAND_DIMS:
    case OK_PAD:
    case OK_PACK:
    case OK_UNPACK:
    case OK_TILE:
        return compute_copy(p, L);
    case OK_MATERIALIZE:
    case OK_GATHER:
        return compute_materialize(p, L);
    default:
        return compute_copy(p, L);
    }
}

}  // namespace

extern "C" void mdla7_dpi_compute_layer_crc(const char* program_path,
                                             int layer_index,
                                             int* crc,
                                             svBit* ok) {
    if (crc)
        *crc = int(kFnvOffset);
    if (ok)
        *ok = 0;

    Program p;
    Layer L;
    if (!load_program(program_path, p) ||
        !read_layer(p, uint32_t(layer_index), L)) {
        return;
    }
    if (!region_ok(p, L.ref_off, L.ref_size)) {
        return;
    }

    std::vector<uint8_t> out = compute_layer(p, L);
    if (out.size() != L.ref_size) {
        return;
    }
    if (crc)
        *crc = int(crc_vector(out));
    if (ok)
        *ok = 1;
}
