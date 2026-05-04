#pragma once

// Reference (pure C++) NN op implementations for unit test comparison.
// Layout matches ConvEngine::compute_int8:
//   act NHWC, wgt [OC, k_h, k_w, in_per_group], out INT32 NHWC.
// group=1 is standard conv; group=in_c is depthwise.

#include <cstdint>
#include <vector>

namespace mdla7 {
namespace ref {

inline std::vector<int32_t> conv_int8(
    const int8_t* in, const int8_t* wgt,
    uint32_t in_h, uint32_t in_w, uint32_t in_c, uint32_t out_c,
    uint32_t k_h, uint32_t k_w,
    uint32_t s_h, uint32_t s_w,
    uint32_t pad_t, uint32_t pad_l, uint32_t pad_b, uint32_t pad_r,
    uint32_t group = 1)
{
    const uint32_t in_per_group  = in_c  / group;
    const uint32_t out_per_group = out_c / group;
    const uint32_t out_h = (in_h + pad_t + pad_b - k_h) / s_h + 1;
    const uint32_t out_w = (in_w + pad_l + pad_r - k_w) / s_w + 1;
    std::vector<int32_t> out(uint64_t(out_h) * out_w * out_c, 0);

    for (uint32_t oh = 0; oh < out_h; ++oh)
    for (uint32_t ow = 0; ow < out_w; ++ow)
    for (uint32_t oc = 0; oc < out_c; ++oc) {
        const uint32_t g       = oc / out_per_group;
        const uint32_t ic_base = g  * in_per_group;
        int32_t s = 0;
        for (uint32_t kh = 0; kh < k_h; ++kh)
        for (uint32_t kw = 0; kw < k_w; ++kw)
        for (uint32_t icr = 0; icr < in_per_group; ++icr) {
            int ih = int(oh) * int(s_h) + int(kh) - int(pad_t);
            int iw = int(ow) * int(s_w) + int(kw) - int(pad_l);
            if (ih >= 0 && ih < int(in_h) && iw >= 0 && iw < int(in_w)) {
                s += int32_t(in [(ih * in_w + iw) * in_c + (ic_base + icr)])
                   * int32_t(wgt[((oc * k_h + kh) * k_w + kw) * in_per_group + icr]);
            }
        }
        out[(oh * out_w + ow) * out_c + oc] = s;
    }
    return out;
}

} // namespace ref
} // namespace mdla7
