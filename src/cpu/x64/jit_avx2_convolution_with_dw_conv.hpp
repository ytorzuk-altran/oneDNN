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

#ifndef CPU_X64_JIT_AVX2_CONVOLUTION_WITH_DW_CONV_HPP
#define CPU_X64_JIT_AVX2_CONVOLUTION_WITH_DW_CONV_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/x64/cpu_reducer.hpp"

#include "cpu/x64/jit_avx2_conv_kernel_f32_old.hpp"
#include "cpu/x64/jit_uni_dw_conv_row_f32.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx2_convolution_with_dw_conv_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc,
             const primitive_attr_t *attr,
             const typename pd_t::base_class *hint_fwd_pd)
                : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd), jcp_(), jcp_dw_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx2, ""),
                jit_avx2_convolution_with_dw_conv_fwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            bool ok = true
                      && this->set_default_formats()
                      && utils::one_of(this->desc()->prop_kind, forward_training,
                                       forward_inference)
                      && utils::one_of(this->desc()->alg_kind,
                                       alg_kind::convolution_auto,
                                       alg_kind::convolution_direct)
                      && !this->has_zero_dim_memory()
                      && utils::everyone_is(data_type::f32,
                                            this->desc()->src_desc.data_type,
                                            this->desc()->weights_desc.data_type,
                                            this->desc()->dst_desc.data_type)
                      && IMPLICATION(this->with_bias(),
                                     data_type::f32 == this->desc()->bias_desc.data_type);
            if (!ok) return status::unimplemented;

            status_t sts = jit_avx2_conv_kernel_f32_old::init_conf(
                    jcp_, *desc(), *src_md(), *weights_md(), *dst_md(), *attr());
            if (sts != status::success) return sts;

            if (jcp_.with_dw_conv) {
                status_t sts_dw = jit_uni_dw_conv_row_f32<avx2>::init_conf(jcp_, jcp_dw_, *this->attr());
                if (sts_dw != status::success) return sts_dw;
            } else {
                return status::unimplemented;
            }

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_kernel_f32_old::init_scratchpad(scratchpad, jcp_, jcp_dw_);

            return status::success;
        }

        jit_conv_conf_t jcp_;
        jit_conv_conf_t jcp_dw_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const bool flat = IC() < 8;
            auto src_tag = flat
                           ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                           : utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto dst_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                           ? utils::pick(2 * ndims() - 6 + flat, gOIw8i8o, gOwi8o,
                                         gOIhw8i8o, gOhwi8o, gOIdhw8i8o, gOdhwi8o)
                           : utils::pick(2 * ndims() - 6 + flat, OIw8i8o, Owi8o,
                                         OIhw8i8o, Ohwi8o, OIdhw8i8o, Odhwi8o);

            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    };

    jit_avx2_convolution_with_dw_conv_fwd_t(const pd_t *apd) : primitive_t(apd),
        kernel_old_(nullptr) { // todo: [antonvor]
        kernel_old_ = new jit_avx2_conv_kernel_f32_old(pd()->jcp_, pd()->jcp_dw_, *pd()->attr());

        if (pd()->jcp_.with_dw_conv) {
            kernel_dw_ = new jit_uni_dw_conv_row_f32<avx2>(pd()->jcp_dw_, *pd()->attr(), pd()->jcp_dw_.ch_block);
        }
    }

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_old_,
                              new jit_avx2_conv_kernel_f32_old(pd()->jcp_, pd()->jcp_dw_, *pd()->attr())));
        CHECK(kernel_old_->create_kernel());

        CHECK(safe_ptr_assign(kernel_dw_,
                              new jit_uni_dw_conv_row_f32<avx2>((pd()->jcp_dw_), *pd()->attr(), pd()->jcp_dw_.ch_block)));
        CHECK(kernel_dw_->create_kernel());

        return status::success;
    }

    ~jit_avx2_convolution_with_dw_conv_fwd_t() {
        delete kernel_old_;

        if (pd()->jcp_.with_dw_conv) {
            delete kernel_dw_;
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    jit_avx2_conv_kernel_f32_old *kernel_old_;
    jit_uni_dw_conv_row_f32<avx2> *kernel_dw_;

};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif