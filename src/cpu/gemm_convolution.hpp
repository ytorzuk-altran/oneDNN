/*******************************************************************************
* Copyright 2016-2020 Intel Corporation
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

#ifndef CPU_JIT_GEMM_CONVOLUTION_HPP
#define CPU_JIT_GEMM_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "gemm_convolution_utils.hpp"
#include "gemm/gemm.hpp"
#include "ref_eltwise.hpp"
#include "ref_depthwise.hpp"

#include "jit_generator.hpp"
#include "jit_uni_eltwise.hpp"
#include "jit_uni_depthwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct ker_args {
    float *dst;
    const float *bias;
    size_t len;
    size_t oc_offset;
};

class uni_pp_kernel_t {
public:
    virtual ~uni_pp_kernel_t() = default;

    void (*ker_)(const ker_args *args);

    virtual void operator()(float *dst, const float *bias, const int len, const int oc_start, const int oc_work, const int oc_stride) = 0;
};

template <cpu_isa_t isa>
class jit_pp_kernel_t : public uni_pp_kernel_t, jit_generator
{
public:
    DECLARE_CPU_JIT_AUX_FUNCTIONS(gemm_convolution_fwd_t::jit_pp_kernel_);
    jit_pp_kernel_t(const cpu_convolution_fwd_pd_t *pd);
    ~jit_pp_kernel_t() {
        for (auto inj : eltwise_injectors_)
            delete inj;
        eltwise_injectors_.clear();
        for (auto inj : depthwise_injectors_)
            delete inj;
        depthwise_injectors_.clear();
    }

    void operator()(float *dst, const float *bias, const int len, const int oc_start, const int oc_work, const int oc_stride) override;

private:
    void generate();

    void (*ker_)(const ker_args *args);

    nstl::vector<jit_uni_eltwise_injector_f32<isa> *> eltwise_injectors_;
    nstl::vector<jit_uni_depthwise_injector_f32<isa> *> depthwise_injectors_;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    static const size_t vlen = cpu_isa_traits<isa>::vlen / sizeof(float);

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_dst = rdx;
    Xbyak::Reg64 reg_bias = rbx;

    Xbyak::Reg64 reg_len = r8;
    Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Xbyak::Reg64 reg_oc_offset = r9;
    Xbyak::Reg64 reg_rem_mask = r10;
    Xbyak::Opmask kreg_rem_mask = k1;

    Xbyak::Reg64 eltwise_reserved_1_ = r11;
    Xbyak::Opmask eltwise_reserved_2_ = k2;
    Xbyak::Opmask depthwise_reserved_2_ = k2;

    //  sse42/avx2
    Xbyak::Reg64 reg_ptr_maskmovdqu_dst = rdi; // sse42: store destination - must be rdi
    Xbyak::Label l_table;
    Xbyak::Reg64 reg_table = r12;
    Xbyak::Reg64 reg_shift_table = r13;
    Vmm vreg_mask = Vmm(0); //  sse42: mask for blendvps must be in xmm0
    Vmm vreg_zero;

    //  post_ops
    Xbyak::Reg64 reg_d_weights = r14;
    Xbyak::Reg64 reg_d_bias = r15;
    Vmm vreg_d_weights, vreg_d_bias;
    post_ops_t post_ops_;

    bool do_bias_ = false;
    int idx_compute_vreg_start_;
    int idx_compute_vreg_max_;

    int idx_vreg_dst(int iter) {
        int idx = idx_compute_vreg_start_ + 0;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + 1;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Vmm vreg_dst(int iter) { return Vmm(idx_vreg_dst(iter)); };
    Vmm vreg_bias(int iter) { return Vmm(idx_vreg_bias(iter)); };
};

class ref_pp_kernel_t : public uni_pp_kernel_t {
public:
    ref_pp_kernel_t(const cpu_convolution_fwd_pd_t *pd);
    ~ref_pp_kernel_t() {
        for (auto impl : ref_eltwise_impls_)
            delete impl;
        ref_eltwise_impls_.clear();
        for (auto impl : ref_depthwise_impls_)
            delete impl;
        ref_depthwise_impls_.clear();
    }

    void operator()(float *dst, const float *bias, const int len, const int oc_start, const int oc_work, const int oc_stride) override;

private:
    nstl::vector<ref_eltwise_scalar_fwd_t*> ref_eltwise_impls_;
    nstl::vector<ref_depthwise_scalar_fwd_t*> ref_depthwise_impls_;

    post_ops_t post_ops_;
    bool do_bias_ = false;
    bool use_fast_relu = false;
    float fast_relu_ns;
};

struct gemm_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_fwd_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
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
                && IMPLICATION(this->with_bias(), data_type::f32
                                   == this->desc()->bias_desc.data_type)
                && this->src_pd_.desc()->format == src_format()
                && this->dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format()
                && this->is_gemm_conv_format()
                && !this->attr()->has_asymmetric_quantization();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_pd(), weights_pd(0), dst_pd(), *attr(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(src_format()));
            if (this->dst_pd_.desc()->format == any)
                CHECK(this->dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            if (this->bias_pd_.desc()->format == any)
                CHECK(this->bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

        virtual bool is_gemm_conv_format() const {
            const auto &p = this->attr()->post_ops_;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len_; i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::sum, primitive_kind::eltwise, primitive_kind::depthwise,
                            primitive_kind::quantization);
                }
                return ok;
            };
            auto contain = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind) != -1; };
            auto position = [&](mkldnn::impl::primitive_kind_t kind) { return p.find(kind); };
            auto count = [&](mkldnn::impl::primitive_kind_t kind) { return p.count(kind); };

            return all_post_ops_supported() &&
                   count(primitive_kind::sum) <= 1 &&
                   IMPLICATION(contain(primitive_kind::sum), position(primitive_kind::sum) == 0);
        }
    };

    gemm_convolution_fwd_t(const pd_t *apd, const input_vector &inputs,
           const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true)
    {
        const auto &post_ops = pd()->attr()->post_ops_;
        const data_t one = 1.0, zero = 0.0;
        beta_ = post_ops.find(primitive_kind::sum) >= 0 ? one : zero;

        bool has_bias = pd()->with_bias(),
                has_post_ops = pd()->attr()->post_ops_.len_ > 0;

        postops_in_ip_ = has_bias || has_post_ops;
        if (postops_in_ip_) {
            if (mayiuse(avx512_common)) {
                pp_kernel_ = new jit_pp_kernel_t<avx512_common>(apd);
            } else if (mayiuse(avx2)) {
                pp_kernel_ = new jit_pp_kernel_t<avx2>(apd);
            } else if (mayiuse(sse42)) {
                pp_kernel_ = new jit_pp_kernel_t<sse42>(apd);
            } else {
                pp_kernel_ = new ref_pp_kernel_t(apd);
            }
        }
    }

    ~gemm_convolution_fwd_t() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    data_t beta_;

    nstl::vector<ref_eltwise_scalar_fwd_t*> eltwise_injectors;
    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;

    uni_pp_kernel_t *pp_kernel_ = nullptr;
    bool postops_in_ip_ = false;
};

struct gemm_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_data_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
                && this->set_default_params() == status::success
                && this->desc()->prop_kind == backward_data
                && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                           alg_kind::convolution_direct)
                && !this->has_zero_dim_memory()
                && utils::everyone_is(data_type::f32,
                        this->desc()->diff_src_desc.data_type,
                        this->desc()->weights_desc.data_type,
                        this->desc()->diff_dst_desc.data_type)
                && this->diff_src_pd_.desc()->format == src_format()
                && this->diff_dst_pd_.desc()->format == src_format()
                && this->weights_pd_.desc()->format == wei_format()
                && this->is_supported_post_ops();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_pd(), weights_pd(0), diff_dst_pd(), *attr(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->diff_src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->diff_src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->diff_src_pd_.desc()->format == any)
                CHECK(this->diff_src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->weights_pd_.desc()->format == any)
                CHECK(this->weights_pd_.set_format(wei_format()));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }

        virtual bool is_supported_post_ops() const {
            const auto &p = this->attr()->post_ops_;
            if (p.len_ > 1)
                return false;

            auto all_post_ops_supported = [&]() {
                bool ok = true;

                for (int i = 0; i < p.len_; i++) {
                    ok = ok && utils::one_of(p.entry_[i].kind, primitive_kind::depthwise);
                }
                return ok;
            };

            return all_post_ops_supported();
        }
    };

    gemm_convolution_bwd_data_t(const pd_t *apd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {
        const auto &post_ops = pd()->attr()->post_ops_;
        for (int i = 0; i < post_ops.len_; i++) {
            auto &post_op = post_ops.entry_[i];
            if (post_op.is_depthwise()) {
                depthwise_injectors.push_back(new ref_depthwise_scalar_fwd_t(post_op.depthwise.alg));
            }
        }
    }
    ~gemm_convolution_bwd_data_t() {
        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_data:
            execute_backward_data();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    nstl::vector<ref_depthwise_scalar_fwd_t*> depthwise_injectors;

    void execute_backward_data() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

struct gemm_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_weights_t);

        virtual status_t init() override {
            using namespace prop_kind;
            using namespace memory_format;

            assert(this->engine()->kind() == engine_kind::cpu);

            bool ok = true
            && this->set_default_params() == status::success
            && this->desc()->prop_kind == backward_weights
            && utils::one_of(this->desc()->alg_kind, alg_kind::convolution_auto,
                       alg_kind::convolution_direct)
            && !this->has_zero_dim_memory()
            && utils::everyone_is(data_type::f32,
                    this->desc()->src_desc.data_type,
                    this->desc()->diff_weights_desc.data_type,
                    this->desc()->diff_dst_desc.data_type)
            && IMPLICATION(this->with_bias(),
                    data_type::f32 == this->desc()->diff_bias_desc.data_type)
            && this->src_pd_.desc()->format == src_format()
            && this->diff_dst_pd_.desc()->format == src_format()
            && this->diff_weights_pd_.desc()->format == wei_format();
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_pd(), diff_weights_pd(0), diff_dst_pd(), *attr(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        memory_format_t src_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (utils::pick(ndims_sp - 1, ncw, nchw, ncdhw));
        }

        memory_format_t wei_format() const {
            using namespace memory_format;
            const int ndims_sp = this->desc()->src_desc.ndims - 2;
            return (this->with_groups()
                ? utils::pick(ndims_sp - 1, goiw, goihw, goidhw)
                : utils::pick(ndims_sp - 1, oiw, oihw, oidhw));
        }

        virtual status_t set_default_params() override {
            using namespace memory_format;
            if (this->src_pd_.desc()->format == any)
                CHECK(this->src_pd_.set_format(src_format()));
            if (this->diff_dst_pd_.desc()->format == any)
                CHECK(this->diff_dst_pd_.set_format(src_format()));
            if (this->diff_weights_pd_.desc()->format == any)
                CHECK(this->diff_weights_pd_.set_format(wei_format()));
            if (this->diff_bias_pd_.desc()->format == any)
                CHECK(this->diff_bias_pd_.set_format(x));
            if (this->desc()->alg_kind == alg_kind::convolution_auto)
                CHECK(this->set_alg_kind(alg_kind::convolution_direct));
            return status::success;
        }
    };

    gemm_convolution_bwd_weights_t(const pd_t *apd, const input_vector &inputs,
              const output_vector &outputs)
        : cpu_primitive_t(apd, inputs, outputs, true) {}
    ~gemm_convolution_bwd_weights_t() {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const {
        switch (pd()->desc()->prop_kind) {
        case prop_kind::backward_weights:
            execute_backward_weights();
            break;
        default:
            assert(!"invalid prop_kind");
        }
        e->set_state(event_t::ready);
    }

private:
    void execute_backward_weights() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
