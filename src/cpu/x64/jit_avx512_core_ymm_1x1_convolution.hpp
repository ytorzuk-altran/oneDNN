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

#ifndef CPU_X64_JIT_AVX512_CORE_YMM_1X1_CONVOLUTION_HPP
#define CPU_X64_JIT_AVX512_CORE_YMM_1X1_CONVOLUTION_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/primitive_hashing.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/dw_convolution_utils.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_avx512_core_ymm_1x1_conv_kernel.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <impl::data_type_t src_type, impl::data_type_t wei_type = src_type,
        impl::data_type_t dst_type = src_type>
struct jit_avx512_core_ymm_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , jcp_()
            , rtus_() {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_core, ""),
                jit_avx512_core_ymm_1x1_convolution_fwd_t);

        status_t init(engine_t *engine) {
            using namespace utils;
            bool ok = true && is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(src_type, wei_type, dst_type, dst_type,
                            data_type::undef)
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, dst_type)
                    && !has_zero_dim_memory() && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md(), weights_md());

            status_t status = jit_avx512_core_ymm_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(), *attr(),
                    dnnl_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_ymm_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_);

            rtus_prepare_space_info(this, scratchpad, jcp_.nthr);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(), OIw8i8o,
                    gOIw8i8o, OIhw8i8o, gOIhw8i8o, OIdhw8i8o, gOIdhw8i8o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend status_t init_rtus_driver(conv_t *self);

    jit_avx512_core_ymm_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd) {}

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_,
                new jit_avx512_core_ymm_1x1_conv_kernel(
                        pd()->jcp_, *pd()->attr(), *pd()->dst_md(0))));
        CHECK(kernel_->create_kernel());

        CHECK(init_rtus_driver<avx2>(this));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const dst_data_t *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<jit_avx512_core_ymm_1x1_conv_kernel> kernel_;
    std::unique_ptr<rtus_driver_t<avx2>> rtus_driver_;
};

using jit_avx512_core_ymm_1x1_convolution_fwd_f32_t
        = jit_avx512_core_ymm_1x1_convolution_fwd_t<data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
