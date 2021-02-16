/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx2_convolution_with_dw_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
// using namespace dnnl::impl::memory_format;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace nstl;

#define src_blk_off(f, n, c, d, h, w) \
(pd()->ndims() == 3) \
? (f).blk_off(n, c, w) \
: (pd()->ndims() == 4) \
? (f).blk_off(n, c, h, w) \
: (f).blk_off(n, c, d, h, w)

#define wht_blk_off_(f, g, ...) \
pd()->with_groups() ? (f).blk_off(g, __VA_ARGS__) : (f).blk_off(__VA_ARGS__)
#define wht_blk_off(f, g, oc, ic, kd, kh, kw) \
(pd()->ndims() == 3) \
? wht_blk_off_(f, g, oc, ic, kw) \
: (pd()->ndims() == 4) \
? wht_blk_off_(f, g, oc, ic, kh, kw) \
: wht_blk_off_(f, g, oc, ic, kd, kh, kw)

void jit_avx2_convolution_with_dw_conv_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    auto scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = kernel_old_->jcp;
    const auto &jcp_dw = kernel_dw_->jcp;
    const int MB = pd()->MB();

    auto dw_bias = jcp_dw.conv_biases;

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = MB * jcp.ngroups * ocb_work * jcp.oh;

    auto ker = [&](const int ithr, const int nthr) {
        auto compute_row_gen = [&](float *ws_p, int n, int g, int ocb, int ocb_num, int oh, int num_rows) {
            for (int h = 0; h < num_rows; h++) {
                if ((oh + h) < 0 || (oh + h) >= jcp.oh) {
                    for (int chb = ocb; chb < ocb + ocb_num; chb++) {
                        memset(ws_p + (((oh + h) + 1) % jcp_dw.kh) * jcp.ow * jcp.oc_block +
                               (chb - ocb) * jcp_dw.kh * jcp.ow * jcp.oc_block, 0, jcp.ow * jcp.oc_block * sizeof(float));
                    }
                } else {
                    for (int icb = 0; icb < jcp.nb_ic; ++icb) {
                        auto par_conv = jit_conv_call_s();

                        const int ij = (oh + h) * jcp.stride_h;
                        const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                        const int i_b_overflow = nstl::max(jcp.ih, ij
                                                                   + (jcp.kh - 1) * (jcp.dilate_h + 1) - jcp.t_pad +
                                                                   1) - jcp.ih;

                        const size_t _oc = g * jcp.nb_oc + ocb;
                        const size_t _ic = g * jcp.nb_ic + icb;

                        const int ih = nstl::max(ij - jcp.t_pad
                                                 + div_up(i_t_overflow,
                                                          (jcp.dilate_h + 1)) * (jcp.dilate_h + 1), 0);
                        par_conv.src = &src[src_d.blk_off(n,
                                                          jcp.ic == 3 ? 0 : _ic, ih, 0)];

                        par_conv.dst = &ws_p[(((oh + h) + 1) % jcp_dw.kh) * jcp.ow *
                                             jcp.oc_block];

                        const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                        par_conv.filt = &weights[pd()->with_groups()
                                                 ? weights_d.blk_off(g, ocb,
                                                                     jcp.ic == 3 ? 0 : icb, wh, 0)
                                                 : weights_d.blk_off(ocb,
                                                                     jcp.ic == 3 ? 0 : icb, wh, 0)];

                        if (icb == 0) {
                            if (bias)
                                par_conv.bias =
                                        &bias[bias_d.blk_off(_oc * jcp.oc_block)];
                            par_conv.flags |= FLAG_IC_FIRST;
                        }

                        if (icb + 1 == jcp.nb_ic) {
                            par_conv.flags |= FLAG_IC_LAST;
                        }

                        par_conv.oc_off = _oc * jcp.oc_block * sizeof(float);

                        par_conv.oc_blocks =
                                nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                        par_conv.kw_padding = 0;
                        const int kh_padding = jcp.kh
                                               - div_up(i_t_overflow, (jcp.dilate_h + 1))
                                               - div_up(i_b_overflow, (jcp.dilate_h + 1));
                        par_conv.kh_padding = nstl::max(0, kh_padding);
                        (*kernel_old_)(&par_conv);
                    }
                }
            }
        };

        auto compute_row_dw = [&](const float *ws_p, int n, int ocb, int ocb_num,
                                  int dst_idx) {
            for (int chb = ocb; chb < nstl::min(ocb + ocb_num, jcp.nb_oc); chb++) {
                auto par_conv_dw = jit_conv_call_s();

                par_conv_dw.src_row0 = &ws_p[(((dst_idx + 1) - 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row1 = &ws_p[(((dst_idx + 1) - 0) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];
                par_conv_dw.src_row2 = &ws_p[(((dst_idx + 1) + 1) % jcp_dw.kh) * jcp_dw.iw * jcp_dw.ch_block +
                                             (chb - ocb) * jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block];

                par_conv_dw.dst = &dst[n * jcp_dw.oc * jcp_dw.oh * jcp_dw.ow + chb * jcp_dw.ch_block * jcp_dw.oh * jcp_dw.ow +
                                       dst_idx / jcp_dw.stride_h * jcp_dw.ow * jcp_dw.ch_block];

                par_conv_dw.kh_padding = jcp_dw.kh;
                par_conv_dw.filt = &jcp_dw.conv_weights[chb * jcp_dw.kh * jcp_dw.kw * jcp_dw.ch_block];
                par_conv_dw.bias = &dw_bias[chb * jcp_dw.ch_block];
                par_conv_dw.ur_w = (size_t) (jcp_dw.ow);
                par_conv_dw.oc_work = nstl::min((chb + 1) * jcp_dw.ch_block, (int) jcp_dw.oc) - chb * jcp_dw.ch_block;
                par_conv_dw.oc_off = chb * jcp_dw.ch_block * sizeof(float);

                (*kernel_dw_)(&par_conv_dw);
            }
        };

        size_t start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto dw_conv_buffer = scratchpad.get<data_t>(key_dw_conv_buffer);
        size_t dw_conv_buffer_size_ = (size_t) jcp_dw.kh * jcp_dw.iw * jcp_dw.ch_block * jcp.nb_oc_blocking;
        auto pbuf = dw_conv_buffer + ithr * dw_conv_buffer_size_;

        size_t n{0}, g{0}, ocbb{0}, oh{0};
        nd_iterator_init(start, n, MB, g, jcp.ngroups, ocbb, ocb_work,
                         oh, jcp.oh);
        for (size_t iwork = start; iwork < end; ++iwork) {
            int ocb = ocbb * jcp.nb_oc_blocking;
            int ocb_num = jcp.nb_oc_blocking;

            if (iwork == start || oh == 0) {
                compute_row_gen(pbuf, n, g, ocb, ocb_num, oh - 1, 2);
            } else {
                compute_row_gen(pbuf, n, g, ocb, ocb_num, oh, 1);
            }

            if (iwork > start && ((oh - 1) % jcp_dw.stride_h == 0) && oh > 0) {
                compute_row_dw(pbuf, n, ocb, ocb_num, oh - 1);
            }

            if ((iwork == end - 1 || (int) oh == jcp.oh - 1) && ((oh) % jcp_dw.stride_h == 0)) {
                compute_row_gen(pbuf, n, g, ocb, ocb_num, oh + 1, 1);
                compute_row_dw(pbuf, n, ocb, ocb_num, oh);
            }

            nd_iterator_step(n, MB, g, jcp.ngroups, ocbb, ocb_work,
                             oh, jcp.oh);
        }
    };

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.get<data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;

        auto dw_padded_bias = scratchpad.get<data_t>(key_dw_conv_padded_bias);
        utils::array_copy(dw_padded_bias, dw_bias, jcp.oc_without_padding);
        utils::array_set(dw_padded_bias + jcp.oc_without_padding, 0.f,
                         jcp.oc - jcp.oc_without_padding);
        dw_bias = dw_padded_bias;
    }

    parallel(0, ker);

    if (pd()->wants_zero_pad_dst())
        ctx.memory(DNNL_ARG_DST)->zero_pad(ctx.stream());
}

}
}
}
}
