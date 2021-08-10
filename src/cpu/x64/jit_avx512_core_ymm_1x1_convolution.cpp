/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_core_ymm_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

#define data_blk_off(f, n, c, d, h, w) \
    ((ndims == 3) ? (f).blk_off(n, c, w) \
                  : ((ndims == 4) ? (f).blk_off(n, c, h, w) \
                                  : (f).blk_off(n, c, d, h, w)))
/* convolution forward */

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_ymm_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward(const exec_ctx_t &ctx) const {
    const auto &jcp = kernel_->jcp;
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();

    if (pd()->wants_padded_bias()) {
        auto padded_bias
                = scratchpad.template get<dst_data_t>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bias = padded_bias;
    }

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });

    if (pd()->wants_zero_pad_dst())
        ctx.memory(DNNL_ARG_DST)->zero_pad(ctx.stream());
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_core_ymm_1x1_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_thr(const int ithr, const int nthr,
        const src_data_t *src, const wei_data_t *weights,
        const dst_data_t *bias, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = kernel_->jcp;
    auto rtus_space = pd()->rtus_.reduce_src_
            ? scratchpad.get<src_data_t>(key_conv_rtus_space)
            : nullptr;

    const int ndims = src_d.ndims();
    const int stride_d = (ndims == 5) ? pd()->desc()->strides[0] : 1;
    const int stride_h = (ndims == 3) ? 1 : pd()->desc()->strides[ndims - 4];
    const int stride_w = pd()->desc()->strides[ndims - 3];

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx2>::call_params_t();

    const int nb_oc = jcp.nb_load;
    const int nb_ic = jcp.nb_reduce;
    const int nb_ic_blocking = jcp.nb_reduce_blocking;

    // override some constants for fused dw_conv

    auto init_bcast = [&](int iwork, int bcast_end, int &n, int &g,
                              int &bcast_step, int &od, int &oh, int &ow,
                              int &id, int &ih, int &iw) {
        int osb {0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb, jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * jcp.bcast_block;
        od = os / (jcp.oh * jcp.ow);
        int os_2d = os % (jcp.oh * jcp.ow);
        oh = os_2d / jcp.ow;
        ow = os_2d % jcp.ow;

        id = od * stride_d;
        ih = oh * stride_h;
        iw = ow * stride_w;
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os, bcast_step * jcp.bcast_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int ocb_end, int &load_step) {
        load_step = step(
                jcp.nb_load_blocking, ocb_end - ocb, jcp.nb_load_blocking_max);
        const auto max_oc
                = nstl::min(ocb_end * jcp.oc_block, jcp.oc_without_padding);
        p.load_dim = this_block_size(
                ocb * jcp.oc_block, max_oc, load_step * jcp.oc_block);
    };

    auto init_reduce = [&](int icb) {
        const int nb_ic_blocking_step
                = nstl::min(icb + nb_ic_blocking, nb_ic) - icb;
        p.first_last_flag = 0 | (icb == 0 ? FLAG_REDUCE_FIRST : 0)
                | (icb + nb_ic_blocking_step >= nb_ic ? FLAG_REDUCE_LAST : 0);

        p.reduce_dim = this_block_size(
                icb * jcp.ic_block, jcp.ic, nb_ic_blocking_step * jcp.ic_block);
        rp.icb = p.reduce_dim;
    };

    auto ker_1x1 = [&](int ocb, int ocb_start, int icb, int n, int g, int od,
                           int oh, int ow, int id, int ih, int iw) {
        const bool is_dst_layout_nxc = utils::one_of(jcp.dst_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int oc_off_idx = is_dst_layout_nxc
                ? g * jcp.oc + ocb * jcp.oc_block
                : g * nb_oc + ocb;
        const size_t dst_off = data_blk_off(dst_d, n, oc_off_idx, od, oh, ow);

        p.output_data = &dst[dst_off];
        p.bias_data
                = &bias[oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block)];

        p.load_data
                = &weights[pd()->with_groups() ? weights_d.blk_off(g, ocb, icb)
                                               : weights_d.blk_off(ocb, icb)];
        const bool is_src_layout_nxc = utils::one_of(jcp.src_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        const int ic_off_idx = is_src_layout_nxc
                ? g * jcp.ic + icb * jcp.ic_block
                : g * nb_ic + icb;
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                    + (is_src_layout_nxc ? ic_off_idx
                                         : jcp.is * ic_off_idx * jcp.ic_block);
            if (ocb == ocb_start) {
                rp.src = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);
                (*rtus_driver_)(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + data_blk_off(src_d, n, ic_off_idx, id, ih, iw);

        p.oc_off = oc_off_idx * (is_dst_layout_nxc ? 1 : jcp.oc_block) * sizeof(float);

        (*kernel_)(&p);
    };

    auto conv_1x1 = [&](int bcast_start, int bcast_end, int ocb_start,
                            int ocb_end) {
        if (bcast_start >= bcast_end || ocb_start >= ocb_end) return;

        if (jcp.loop_order == loop_rlb) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    int iwork = bcast_start;
                    while (iwork < bcast_end) {
                        int n {0}, g {0}, bcast_step {0}, od {0}, oh {0},
                                ow {0}, id {0}, ih {0}, iw {0};
                        init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh,
                                ow, id, ih, iw);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        iwork += bcast_step;
                    }
                    ocb += load_step;
                }
            }
        } else if (jcp.loop_order == loop_lbr) {
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, ocb_end, load_step);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                            id {0}, ih {0}, iw {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    iwork += bcast_step;
                }
                ocb += load_step;
            }
        } else if (jcp.loop_order == loop_rbl) {
            for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                init_reduce(icb);
                int iwork = bcast_start;
                while (iwork < bcast_end) {
                    int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                            id {0}, ih {0}, iw {0};
                    init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow,
                            id, ih, iw);
                    int ocb = ocb_start;
                    while (ocb < ocb_end) {
                        int load_step;
                        init_load(ocb, ocb_end, load_step);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                        ocb += load_step;
                    }
                    iwork += bcast_step;
                }
            }
        } else if (jcp.loop_order == loop_blr) {
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n {0}, g {0}, bcast_step {0}, od {0}, oh {0}, ow {0},
                        id {0}, ih {0}, iw {0};
                init_bcast(iwork, bcast_end, n, g, bcast_step, od, oh, ow, id,
                        ih, iw);
                int ocb = ocb_start;
                while (ocb < ocb_end) {
                    int load_step;
                    init_load(ocb, ocb_end, load_step);
                    for (int icb = 0; icb < nb_ic; icb += nb_ic_blocking) {
                        init_reduce(icb);
                        ker_1x1(ocb, ocb_start, icb, n, g, od, oh, ow, id, ih,
                                iw);
                    }
                    ocb += load_step;
                }
                iwork += bcast_step;
            }
        } else {
            assert(!"unsupported loop order");
        }
    };

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;
    int bcast_start {0}, bcast_end {0}, ocb_start {0}, ocb_end {0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end, jcp.nb_load,
            ocb_start, ocb_end, jcp.load_grp_count);

    conv_1x1(bcast_start, bcast_end, ocb_start, ocb_end);
}

template struct jit_avx512_core_ymm_1x1_convolution_fwd_t<data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
