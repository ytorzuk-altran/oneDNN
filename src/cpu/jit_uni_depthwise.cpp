/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <mkldnn_types.h>
#include "mkldnn_types.h"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"
#include "jit_generator.hpp"

#include "jit_uni_depthwise.hpp"

#define GET_OFF(field) offsetof(jit_args, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

struct jit_args {
    const float *from;
    const float *to;
    const float *weights;
    const float *bias;
    size_t work_amount;
};

struct jit_uni_depthwise_kernel_f32 : public c_compatible {
    const depthwise_desc_t &desc_;
    void (*ker_)(const jit_args *);
    bool with_bias_;

    void operator()(const jit_args *args) { assert(ker_); ker_(args); }

    jit_uni_depthwise_kernel_f32(const depthwise_desc_t &desc, bool with_bias)
        : desc_(desc), ker_(nullptr), with_bias_(with_bias) {}
    virtual ~jit_uni_depthwise_kernel_f32() {}


};

/* jit kernels */
namespace {

template <cpu_isa_t isa>
struct jit_uni_scale_shift_kernel_f32 : public jit_uni_depthwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_scale_shift_kernel_f32)
    jit_uni_scale_shift_kernel_f32(const depthwise_desc_t &desc, bool with_bias)
        : jit_uni_depthwise_kernel_f32(desc, with_bias), jit_generator() {
        assert(desc.alg_kind == alg_kind::depthwise_scale_shift);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        bool isFlat = (desc.src_desc.format == nchw && desc.dst_desc.format == nchw) ||
                      (desc.src_desc.format == ncdhw && desc.dst_desc.format == ncdhw);

        Reg64 param = abi_param1;

        const int block_size = isa == avx512_common ? 16 : 8;
        const int main_loop_step = (isFlat || desc.src_desc.format == nc) ? block_size : 1;

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_scale, ptr[param + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);
        if (with_bias_)
            mov(reg_shift, ptr[param + GET_OFF(bias)]);

        Label main_loop_label;
        Label tail_loop_label;
        Label tail_loop_flat_label;
        Label exit_label;

        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            if (isFlat) {
                uni_vbroadcastss(get_scale_reg(i), ptr[reg_scale]);
                if (with_bias_)
                    uni_vbroadcastss(get_shift_reg(i), ptr[reg_shift]);
                else
                    uni_vpxor(get_shift_reg(i), get_shift_reg(i), get_shift_reg(i));
            } else {
                uni_vmovups(get_scale_reg(i), ptr[reg_scale + i*4*sizeof(float)]);
                if (with_bias_)
                    uni_vmovups(get_shift_reg(i), ptr[reg_shift + i*4*sizeof(float)]);
                else
                    uni_vpxor(get_shift_reg(i), get_shift_reg(i), get_shift_reg(i));
            }
        }

        if (isFlat) {
            uni_vbroadcastss(xmm_scale, ptr[reg_scale]);
            if (with_bias_)
                uni_vbroadcastss(xmm_shift, ptr[reg_shift]);
            else
                uni_vpxor(xmm_shift, xmm_shift, xmm_shift);
        }

        L(main_loop_label); {
            cmp(reg_work_amount, main_loop_step-1);
            jle(isFlat ? tail_loop_flat_label : tail_loop_label, T_NEAR);

            int repeats = isa == sse42 ? 2 : 1;
            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src, ptr[reg_from + i*4*sizeof(float)]);
                uni_vmovups(vmm_dst, get_shift_reg(i));
                uni_vfmadd231ps(vmm_dst, vmm_src, get_scale_reg(i));
                uni_vmovups(ptr[reg_to + i*4*sizeof(float)], vmm_dst);
            }

            add(reg_from, block_size*sizeof(float));
            add(reg_to, block_size*sizeof(float));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);
            movss(xmm_shift, ptr[reg_shift]);
            movss(xmm_scale, ptr[reg_scale]);
            uni_vmovups(xmm_dst, xmm_shift);
            uni_vfmadd231ps(xmm_dst, xmm_src, xmm_scale);
            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            add(reg_shift, 1*sizeof(float));
            add(reg_scale, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_flat_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);
            uni_vmovups(xmm_dst, xmm_shift);
            uni_vfmadd231ps(xmm_dst, xmm_src, xmm_scale);
            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_flat_label, T_NEAR);
        }

        L(exit_label);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    inline Vmm get_scale_reg(int idx) { return Vmm(idx + 2); }
    inline Vmm get_shift_reg(int idx) { return Vmm(idx + 4); }

    Reg64 reg_from = r8;
    Reg64 reg_to = r9;
    Reg64 reg_work_amount = r10;
    Reg64 reg_scale = r11;
    Reg64 reg_shift = r12;

    Vmm vmm_src = Vmm(0);
    Vmm vmm_dst = Vmm(1);

    Xmm xmm_src = Xmm(0);
    Xmm xmm_dst = Xmm(1);
    Xmm xmm_scale = Xmm(6);
    Xmm xmm_shift = Xmm(7);
};

template <cpu_isa_t isa>
struct jit_uni_prelu_kernel_f32 : public jit_uni_depthwise_kernel_f32,
    public jit_generator
{
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_prelu_kernel_f32)
    jit_uni_prelu_kernel_f32(const depthwise_desc_t &desc, bool with_bias)
        : jit_uni_depthwise_kernel_f32(desc, with_bias), jit_generator() {
        assert(desc.alg_kind == alg_kind::depthwise_prelu);
        assert(isa == sse42 || isa == avx2 || isa == avx512_common);

        bool isFlat = (desc.src_desc.format == nchw && desc.dst_desc.format == nchw) ||
                      (desc.src_desc.format == ncdhw && desc.dst_desc.format == ncdhw);

        Reg64 param = abi_param1;

        const int block_size = isa == avx512_common ? 16 : 8;
        const int main_loop_step = (isFlat || desc.src_desc.format == nc) ? block_size : 1;

        this->preamble();

        mov(reg_from, ptr[param + GET_OFF(from)]);
        mov(reg_to, ptr[param + GET_OFF(to)]);
        mov(reg_scale, ptr[param + GET_OFF(weights)]);
        mov(reg_work_amount, ptr[param + GET_OFF(work_amount)]);

        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        int repeats = isa == sse42 ? 2 : 1;
        for (int i = 0; i < repeats; i++) {
            if (isFlat) {
                uni_vbroadcastss(get_scale_reg(i), ptr[reg_scale]);
            } else {
                uni_vmovups(get_scale_reg(i), ptr[reg_scale + i*4*sizeof(float)]);
            }
        }

        if (isFlat) {
            uni_vbroadcastss(xmm_scale, ptr[reg_scale]);
        }

        Label main_loop_label;
        Label tail_loop_label;
        Label tail_loop_flat_label;
        Label exit_label;

        L(main_loop_label); {
            cmp(reg_work_amount, main_loop_step-1);
            jle(isFlat ? tail_loop_flat_label :tail_loop_label, T_NEAR);

            for (int i = 0; i < repeats; i++) {
                uni_vmovups(vmm_src, ptr[reg_from + i*4*sizeof(float)]);

                if (isa == sse42) {
                    pxor(vmm_mask, vmm_mask);
                    cmpps(vmm_mask, vmm_src, _cmp_gt_os);
                    movups(vmm_dst, vmm_src);
                    mulps(vmm_src, get_scale_reg(i));
                    blendvps(vmm_dst, vmm_src);
                } else if (isa == avx2) {
                    vcmpgtps(vmm_mask, vmm_src, vmm_zero);
                    vmulps(vmm_dst, vmm_src, get_scale_reg(i));
                    vblendvps(vmm_dst, vmm_dst, vmm_src, vmm_mask);
                } else if (isa == avx512_common) {
                    Opmask kmask = Opmask(7);
                    vmovups(vmm_dst, vmm_src);
                    vcmpps(kmask, vmm_src, vmm_zero, _cmp_lt_os);
                    vmulps(vmm_dst | kmask, vmm_src, get_scale_reg(i));
                }

                uni_vmovups(ptr[reg_to + i*4*sizeof(float)], vmm_dst);
            }

            add(reg_from, block_size*sizeof(float));
            add(reg_to, block_size*sizeof(float));
            sub(reg_work_amount, main_loop_step);

            jmp(main_loop_label, T_NEAR);
        }

        L(tail_loop_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);
            movss(xmm_scale, ptr[reg_scale]);

            pxor(xmm_mask, xmm_mask);
            cmpps(xmm_mask, xmm_src, _cmp_gt_os);
            movups(xmm_dst, xmm_src);
            mulps(xmm_src, xmm_scale);
            blendvps(xmm_dst, xmm_src);

            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            add(reg_scale, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_label, T_NEAR);
        }

        L(tail_loop_flat_label); {
            cmp(reg_work_amount, 0);
            jle(exit_label, T_NEAR);

            movss(xmm_src, ptr[reg_from]);

            pxor(xmm_mask, xmm_mask);
            cmpps(xmm_mask, xmm_src, _cmp_gt_os);
            movups(xmm_dst, xmm_src);
            mulps(xmm_src, xmm_scale);
            blendvps(xmm_dst, xmm_src);

            movss(ptr[reg_to], xmm_dst);

            add(reg_from, 1*sizeof(float));
            add(reg_to, 1*sizeof(float));
            dec(reg_work_amount);

            jmp(tail_loop_flat_label, T_NEAR);
        }

        L(exit_label);

        this->postamble();

        ker_ = (decltype(ker_))this->getCode();
    }

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xmm,
                                             isa == avx2, Ymm, Zmm>::type;

    inline Vmm get_scale_reg(int idx) { return Vmm(idx + 4); }

    Reg64 reg_from = r8;
    Reg64 reg_to = r9;
    Reg64 reg_work_amount = r10;
    Reg64 reg_scale = r11;

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_src = Vmm(1);
    Vmm vmm_zero = Vmm(2);
    Vmm vmm_dst = Vmm(3);

    Xmm xmm_mask = Xmm(0);
    Xmm xmm_src = Xmm(1);
    Xmm xmm_dst = Xmm(3);
    Xmm xmm_scale = Xmm(4);

    const unsigned char _cmp_gt_os = 6;
    const unsigned char _cmp_lt_os = 1;
};

} /* namespace */

template <cpu_isa_t isa>
status_t jit_uni_depthwise_fwd_t<isa>::pd_t::init() {
    using namespace alg_kind;

    memory_format_t desired_blk_fmt, desired_pln_fmt;
    if (desc()->src_desc.ndims == 5) {
        desired_blk_fmt = isa == avx512_common ? nCdhw16c : nCdhw8c;
        desired_pln_fmt = ncdhw;
    } else if (desc()->src_desc.ndims == 4) {
        desired_blk_fmt = isa == avx512_common ? nChw16c : nChw8c;
        desired_pln_fmt = nchw;
    } else {
        desired_blk_fmt = nc;
        desired_pln_fmt = nc;
    }

    assert(engine()->kind() == engine_kind::cpu);
    bool ok = true && mayiuse(isa)
        && utils::one_of(desc()->prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference)
        && utils::everyone_is(data_type::f32, desc()->src_desc.data_type, desc()->dst_desc.data_type)
        && desc()->src_desc.format == desc()->dst_desc.format
        && utils::one_of(desc()->src_desc.format, desired_blk_fmt, desired_pln_fmt)
        && utils::one_of(desc()->dst_desc.format, desired_blk_fmt, desired_pln_fmt)
        && utils::one_of(desc()->weights_desc.format, x)
        && IMPLICATION(this->with_bias(), x == desc()->bias_desc.format)
        && attr()->has_default_values();

    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa>
jit_uni_depthwise_fwd_t<isa>::jit_uni_depthwise_fwd_t(const pd_t *apd,
        const input_vector &inputs, const output_vector &outputs)
    : cpu_primitive_t(apd, inputs, outputs), kernel_(nullptr),
      padded_weights_(nullptr), padded_bias_(nullptr) {
    const auto &desc = *pd()->desc();
    switch (desc.alg_kind) {
        case alg_kind::depthwise_scale_shift:
            kernel_ = new jit_uni_scale_shift_kernel_f32<isa>(desc, pd()->with_bias()); break;
        case alg_kind::depthwise_prelu:
            kernel_ = new jit_uni_prelu_kernel_f32<isa>(desc, pd()->with_bias()); break;
        default: assert(!"unknown depthwise alg_kind");
    }

    const int simd_w = isa == avx512_common ? 16 : 8;
    const memory_desc_wrapper data_d(pd()->src_pd());
    const int c_without_padding = data_d.dims()[1];
    const int c_padded = rnd_up(c_without_padding, simd_w);

    if (pd()->want_padded_weights()) {
        padded_weights_ = (data_t *)malloc(sizeof(data_t) * c_padded, 64);
        for (int oc = c_without_padding; oc < c_padded; ++oc)
            padded_weights_[oc] = 0;

        if (pd()->with_bias()) {
            padded_bias_ = (data_t *)malloc(sizeof(data_t) * c_padded, 64);
            for (int oc = c_without_padding; oc < c_padded; ++oc)
                padded_bias_[oc] = 0;
        }
    }
}

template <cpu_isa_t isa>
jit_uni_depthwise_fwd_t<isa>::~jit_uni_depthwise_fwd_t() {
    delete kernel_;
    free(padded_weights_);
    free(padded_bias_);
}

template <cpu_isa_t isa>
void jit_uni_depthwise_fwd_t<isa>::execute_forward() const {
    auto src = reinterpret_cast<const data_t *>(this->input_memory(0));
    auto weights = reinterpret_cast<const data_t *>(this->input_memory(1));
    auto bias = reinterpret_cast<const data_t *>(this->input_memory(2));
    auto dst = reinterpret_cast<data_t *>(this->memory());

    const memory_desc_wrapper data_d(pd()->src_pd());
    const memory_desc_wrapper weights_d(pd()->weights_pd(0));
    const memory_desc_wrapper bias_d(pd()->weights_pd(1));

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();

    const int simd_w = isa == avx512_common ? 16 : 8;
    const int ch_block_size = (data_d.format() == nchw) || (data_d.format() == ncdhw) ? 1 : simd_w;
    const int CB = div_up(C, ch_block_size);

    if (pd()->want_padded_weights()) {
        for (int oc = 0; oc < C; ++oc)
            padded_weights_[oc] = weights[oc];
        weights = padded_weights_;

        if (pd()->with_bias()) {
            for (int oc = 0; oc < C; ++oc)
                padded_bias_[oc] = bias[oc];
            bias = padded_bias_;
        }
    }

    parallel_nd(MB, CB, D, H,
        [&](int mb, int cb, int d, int h) {
        auto arg = jit_args();

        size_t data_off = data_d.ndims() == 4
                          ? data_d.blk_off(mb, cb, h)
                          : data_d.ndims() == 5
                            ? data_d.blk_off(mb, cb, d, h)
                            : data_d.blk_off(mb, cb * ch_block_size);

        arg.from    = &src[data_off];
        arg.to      = &dst[data_off];
        arg.weights = &weights[weights_d.blk_off(cb * ch_block_size)];
        if (bias)
            arg.bias = &bias[bias_d.blk_off(cb * ch_block_size)];
        arg.work_amount = data_d.format() == nc ? nstl::min(ch_block_size, C - cb * ch_block_size) : (size_t)W;

        (*kernel_)(&arg);
    });
}

template struct jit_uni_depthwise_fwd_t<sse42>;
template struct jit_uni_depthwise_fwd_t<avx2>;
template struct jit_uni_depthwise_fwd_t<avx512_common>;

}
}
}
