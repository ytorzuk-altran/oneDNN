/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <cstdlib>
#include <functional>

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_conv_zp_src_pad_comp.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace gemm_x8s8s32x_convolution_utils {
using namespace dnnl::impl::cpu::gemm_x8s8s32x_convolution_utils;

template <cpu_isa_t isa>
struct jit_pp_ker_t : pp_ker_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            gemm_x8s8s32x_convolution_utils::jit_pp_ker_t);

    jit_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
            : pp_ker_t(pd, jcp)
            , do_eltwise_(false)
            , do_sum_(false)
            , sum_scale_(0)
            , sum_data_type_(dnnl_f32)
            , default_OC_loop_unroll_(4)
            , max_OC_loop_unroll_(isa == avx512_common ? 12 : 6)
            , idx_compute_vreg_start_(0)
            , idx_compute_vreg_max_(isa == avx512_common ? 31 : 15)
            , compute_vregs_per_iter_(1)
    {
        if (utils::one_of(isa, avx2, sse41)) {
            idx_compute_vreg_start_ += 2;   //  Vmm(0), Vmm(1) - for masks
        }
        if (do_scale_) {
            vreg_scale = Vmm(idx_compute_vreg_start_++);
        }
        dst_data_type_size_ = types::data_type_size(dst_data_type_);
        if (dst_data_type_ == data_type::u8 || utils::one_of(isa, avx2, sse41)) {
            vreg_zero = Vmm(idx_compute_vreg_start_++);
        }
        bool only_eltwise_or_sum = true;
        for (int idx = 0; idx < post_ops_.len(); ++idx) {
            const auto &e = post_ops_.entry_[idx];
            if (e.is_eltwise(true)) {
                do_eltwise_ = true;
            } else if (e.is_sum()) {
                do_sum_ = true;
                sum_scale_ = e.sum.scale;
                sum_data_type_ = e.sum.dt;
            } else {
                only_eltwise_or_sum = false;
            }
        }
        if (post_ops_.len() > 0 && !only_eltwise_or_sum) {
            vreg_d_weights = Vmm(idx_compute_vreg_max_--);
            vreg_d_bias = Vmm(idx_compute_vreg_max_--);
        }

        do_signed_scaling_ = jcp_.signed_input;
        if (do_signed_scaling_)
            vreg_signed_scale = Vmm(idx_compute_vreg_start_++);

        if (do_bias_) {
            bias_data_type_size_ = types::data_type_size(bias_data_type_);
            compute_vregs_per_iter_++;
        }
        if (do_sum_) {
            vreg_sum_scale = Vmm(idx_compute_vreg_start_++);
            compute_vregs_per_iter_++;
        }

        for (int i = 0; i < post_ops_.len(); i++) {
            auto &post_op = post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                jit_eltwise_injectors_.push_back(new jit_uni_eltwise_injector_f32<isa>(
                        this, post_op.eltwise, true, eltwise_reserved, mask_post_op_reserved));
            } else if (post_op.is_depthwise()) {
                jit_depthwise_injectors_.push_back(new jit_uni_depthwise_injector_f32<isa>(
                        this, post_op.depthwise.alg, mask_post_op_reserved));
            }
        }

        int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1) / compute_vregs_per_iter_;
        max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);
        default_OC_loop_unroll_ = nstl::min(default_OC_loop_unroll_, max_unroll);
    }
    ~jit_pp_ker_t() {
        for (auto inj : jit_eltwise_injectors_)
            delete inj;
        jit_eltwise_injectors_.clear();
        for (auto inj : jit_depthwise_injectors_)
            delete inj;
        jit_depthwise_injectors_.clear();
    }

    status_t create_kernel() override { return jit_generator::create_kernel(); }

    void operator()(void *void_dst, acc_data_t *acc, const char *bias, const float *scales, float sum_scale, float signed_scale,
                    int g, size_t start, size_t end,
                    const zero_point_call_params_t &zp,
                    const void * /* post_ops_binary_rhs_arg_vec */,
                    const void * /* dst_orig */, const exec_ctx_t &ctx,
                    const memory_desc_t &dst_md,
        const single_gemm_conv_chunk_desc_t &chunk_desc) const override {

        if (end <= start) return;

        char *dst = (char *)void_dst;

        ker_args_t args;
        size_t oc_offset = start % OC_;
        size_t os_offset = start / OC_;
        args.acc = acc + start;
        args.dst = dst
                   + (os_offset * dst_os_stride_ + oc_offset)
                     * dst_data_type_size_;
        args.bias = bias + (g * jcp_.oc + oc_offset) * bias_data_type_size_;
        args.scales = scales + scale_idx_mult_ * (g * jcp_.oc + oc_offset);
        args.sum_scale = sum_scale_;
        args.signed_scale = signed_scale;
        args.len = end - start;
        args.oc_offset = oc_offset;
        args.g_offset = g * jcp_.oc;
        jit_generator::operator()(&args);
    }

private:
    void generate() override;

    struct ker_args_t {
        char *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float sum_scale;
        float signed_scale;
        size_t len;
        size_t oc_offset;
        size_t g_offset;
    };

    nstl::vector<jit_uni_eltwise_injector_f32<isa> *> jit_eltwise_injectors_;
    nstl::vector<jit_uni_depthwise_injector_f32<isa> *> jit_depthwise_injectors_;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    static const size_t vlen = cpu_isa_traits<isa>::vlen / sizeof(float);

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_dst = rdx;
    Xbyak::Reg64 reg_acc = rax;
    Xbyak::Reg64 reg_bias = rbx;
    Xbyak::Reg64 reg_scales = rsi;
    Xbyak::Reg64 reg_g_offset = rbp;

    Xbyak::Reg64 reg_len = r8;
    Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Xbyak::Reg64 reg_oc_offset = r9;
    Xbyak::Reg64 reg_rem_mask_short = r10;
    Xbyak::Opmask kreg_rem_mask_short = k1;

    Vmm vreg_zero, vreg_scale, vreg_sum_scale, vreg_signed_scale, vreg_comp;

    //  sse41/avx2
    Xbyak::Reg64 reg_ptr_maskmovdqu_dst = rdi; // sse41: store destination - must be rdi
    Xbyak::Label l_table;
    Xbyak::Reg64 reg_table = r12;
    Xbyak::Reg64 reg_shift_table = r13;
    Vmm vreg_mask = Vmm(0); //  sse41: mask for blendvps must be in xmm0
    Vmm vreg_store_mask = Vmm(1);

    //  post_ops
    Xbyak::Opmask mask_post_op_reserved = k2;
    Xbyak::Reg64 eltwise_reserved = rax;
    Xbyak::Reg64 reg_d_weights = r14;
    Xbyak::Reg64 reg_d_bias = r15;
    Vmm vreg_d_weights, vreg_d_bias;

    size_t dst_data_type_size_ = 0;
    size_t bias_data_type_size_ = 0;

    bool do_eltwise_;
    bool do_sum_;
    float sum_scale_;
    data_type_t sum_data_type_;
    bool do_signed_scaling_;

    int default_OC_loop_unroll_;
    int max_OC_loop_unroll_;
    int idx_compute_vreg_start_;
    int idx_compute_vreg_max_;
    int compute_vregs_per_iter_;

    int idx_vreg_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 0;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 1;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_prev_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 2;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Vmm vreg_dst(int idx) { return Vmm(idx_vreg_dst(idx)); };
    Xbyak::Ymm ymm_dst(int idx) { return Xbyak::Ymm(idx_vreg_dst(idx)); };
    Xbyak::Xmm xmm_dst(int idx) { return Xbyak::Xmm(idx_vreg_dst(idx)); };
    Vmm vreg_bias(int idx) { return Vmm(idx_vreg_bias(idx)); };
    Vmm vreg_prev_dst(int idx) { return Vmm(idx_vreg_prev_dst(idx)); };
};

template <cpu_isa_t isa>
void jit_pp_ker_t<isa>::generate() {
    using namespace Xbyak;
    using namespace utils;

    preamble();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    mov(reg_g_offset, ptr[reg_param + PARAM_OFF(g_offset)]);
    if (do_sum_)
        uni_vbroadcastss(vreg_sum_scale, ptr[reg_param + PARAM_OFF(sum_scale)]);
    if (do_signed_scaling_)
        uni_vbroadcastss(vreg_signed_scale, ptr[reg_param + PARAM_OFF(signed_scale)]);
    if (do_scale_ && scale_idx_mult_ == 0)
        uni_vbroadcastss(vreg_scale, dword[reg_scales]);
#undef PARAM_OFF

    if (do_eltwise_ || dst_data_type_ == data_type::u8 || utils::one_of(isa, avx2, sse41))
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);

    if (utils::one_of(isa, avx2, sse41))
        mov(reg_table, l_table);

    auto apply_post_ops = [&](size_t offset, int idx) {
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < post_ops_.len(); i++) {
            auto& post_op = post_ops_.entry_[i];
            if (post_op.is_sum()) {
                auto dst_addr = ptr[reg_dst + offset * dst_data_type_size_];
                auto vreg_prev_dst_ = vreg_prev_dst(idx);
                switch (sum_data_type_) {
                    case data_type::f32:
                    case data_type::s32: uni_vmovups(vreg_prev_dst_, dst_addr); break;
                    case data_type::s8: uni_vpmovsxbd(vreg_prev_dst_, dst_addr); break;
                    case data_type::u8: uni_vpmovzxbd(vreg_prev_dst_, dst_addr); break;
                    default: assert(!"unsupported data type");
                }
                if (sum_data_type_ != data_type::f32)
                    uni_vcvtdq2ps(vreg_prev_dst(idx), vreg_prev_dst(idx));

                uni_vfmadd231ps(vreg_dst(idx), vreg_prev_dst(idx), vreg_sum_scale);
            } else if (post_op.is_eltwise()) {
                jit_eltwise_injectors_[eltwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1);
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                add(reg_oc_offset, reg_g_offset);
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data + offset));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data + offset));
                lea(reg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                lea(reg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                jit_depthwise_injectors_[depthwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1, reg_d_weights, reg_d_bias);
                depthwise_inj_idx++;
                sub(reg_oc_offset, reg_g_offset);
            } else if (post_op.is_quantization()) {
                add(reg_oc_offset, reg_g_offset);
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_data_type_ == dnnl_f32 || i != post_ops_.len() - 1;

                if (post_op.quantization.crop_low_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_ + offset));
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_));
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.crop_high_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_ + offset));
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_));
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_d_weights);
                uni_vminps(vreg_dst(idx), vreg_dst(idx), vreg_d_bias);

                if (post_op.quantization.input_scale_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_ + offset));
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_));
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.input_shift_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_ + offset));
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_));
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);

                if (do_rounding)
                    uni_vroundps(vreg_dst(idx), vreg_dst(idx), 0);

                if (do_dequantization) {
                    if (post_op.quantization.output_scale_data->count_ != 1) {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_ + offset));
                        uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_));
                        uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                    }

                    if (post_op.quantization.output_shift_data->count_ != 1) {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_ + offset));
                        uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_));
                        uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                    }

                    uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);
                }
                sub(reg_oc_offset, reg_g_offset);
            }
        }
    };

    // Load accumulated value, convert to float,
    // bias (if any), scaling, and simple operations (if any);
    // then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        if (do_scale_ && scale_idx_mult_ > 0) {
            assert(scale_idx_mult_ == 1);
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_ = vreg_scale;
            if (isa == avx512_common) {
                if (apply_mask)
                    vreg_scale_ = vreg_scale_ | kreg_rem_mask_short;
                uni_vmovups(vreg_scale_, scale_addr);
            } else {
                if (apply_mask)
                    if (isa != sse41) {
                        uni_vblendvps(vreg_scale, vreg_zero, scale_addr, vreg_mask);
                    } else {
                        uni_vmovups(vreg_scale, vreg_zero);
                        uni_vblendvps(vreg_scale, vreg_scale, scale_addr, vreg_mask);
                    }
                else
                    uni_vmovups(vreg_scale, scale_addr);
            }
        }

        auto vreg_dst_ = vreg_dst(idx);
        if (isa == avx512_common) {
            if (apply_mask)
                vreg_dst_ = vreg_dst_ | kreg_rem_mask_short;
            uni_vcvtdq2ps(vreg_dst_, acc_addr);
        } else {
            if (apply_mask) {
                if (isa != sse41) {
                    uni_vblendvps(vreg_dst_, vreg_zero, acc_addr, vreg_mask);
                } else {
                    uni_vmovups(vreg_dst_, acc_addr);
                }
                uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
            } else {
                if (isa == sse41) {
                    uni_vmovups(vreg_dst_, acc_addr);
                    uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
                } else {
                    uni_vcvtdq2ps(vreg_dst_, acc_addr);
                }
            }
        }

        if (do_signed_scaling_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_signed_scale);

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (isa == avx512_common && apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_short;

            switch (bias_data_type_) {
                case data_type::s8: uni_vpmovsxbd(vreg_bias_, bias_addr); break;
                case data_type::u8: uni_vpmovzxbd(vreg_bias_, bias_addr); break;
                case data_type::s32:
                case data_type::f32: uni_vmovups(vreg_bias_, bias_addr); break;
                default: assert(!"unimplemented");
            }
            if (bias_data_type_ != data_type::f32)
                uni_vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            uni_vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        if (do_scale_)
            uni_vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        apply_post_ops(offset, idx);

        if (dst_data_type_ != data_type::f32) {
            if (isa == avx512_common) {
                auto rmode_control = T_rn_sae;
                vcvtps2dq(vreg_dst(idx) | rmode_control, vreg_dst(idx));
            } else {
                uni_vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
            }
        }

        if (dst_data_type_ == data_type::u8)
            uni_vpmaxsd(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        auto dst_addr = ptr[reg_dst + offset * dst_data_type_size_];
        switch (dst_data_type_) {
            case data_type::s8:
                if (isa == avx512_common) {
                    vpmovsdb(dst_addr, vreg_dst_);
                } else {
                    uni_vpackssdw(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (isa != sse41)
                        vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                    uni_vpacksswb(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        if (isa != sse41) {
                            vmovq(dst_addr, xmm_dst(idx));
                        } else {
                            movd(dst_addr, xmm_dst(idx));
                        }
                    }
                }
                break;
            case data_type::u8:
                if (isa == avx512_common) {
                    vpmovusdb(dst_addr, vreg_dst_);
                } else {
                    uni_vpackusdw(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (isa != sse41)
                        vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                    uni_vpackuswb(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        if (isa != sse41) {
                            vmovq(dst_addr, xmm_dst(idx));
                        } else {
                            movd(dst_addr, xmm_dst(idx));
                        }
                    }
                }
                break;
            case data_type::f32:
            case data_type::s32:
                if (isa == avx512_common) {
                    uni_vmovups(dst_addr, vreg_dst_);
                } else {
                    if (apply_mask) {
                        if (isa != sse41) {
                            vmaskmovps(dst_addr, vreg_mask, vreg_dst_);
                        } else {
                            lea(reg_ptr_maskmovdqu_dst, dst_addr);
                            maskmovdqu(vreg_dst_, vreg_mask);
                        }
                    } else {
                        uni_vmovups(dst_addr, vreg_dst_);
                    }
                }
                break;
            default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * dst_data_type_size_);
        add(reg_acc, offset * sizeof(acc_data_t));
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            add(reg_scales, offset * sizeof(float));
        }
        if (do_bias_)
            add(reg_bias, offset * bias_data_type_size_);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * dst_data_type_size_]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        }
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_)
            sub(reg_bias, OC_ * bias_data_type_size_);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            sub(reg_scales, OC_ * sizeof(float));
        }
        add(reg_dst, (dst_os_stride_ - OC_) * dst_data_type_size_);
    };

    //                    <--------- OC --------------->
    //
    // ^  ................+..............+-------------+.......................
    // |  .               : not accessed |Prologue loop|                      .
    // |  .               +--------------+-------------+                      .
    //    .               |                            |                      .
    // O  .               |  Main loop (unrolled)      |                      .
    // S  .               |                            |                      .
    //    .               +--------------+-------------+                      .
    // |  .               | Epilogue loop|not accessed :                      .
    // v  ................+--------------+.............+.......................

    bool do_post_ops = post_ops_.len() != 0;

    Label prologue_end;
    cmp(reg_oc_offset, 0);
    je(prologue_end, T_NEAR);

    // Prologue loop
    {
        mov(reg_tmp, OC_);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);

        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
        cmp(reg_tmp, vlen);
        jl(prologue_loop_tail, T_NEAR);
        L(prologue_loop);
        {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            if (do_post_ops)
                add(reg_oc_offset, vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        if (isa == avx512_common) {
            mov(reg_rem_mask_short, 1);
            // cl == reg_tmp because reg_tmp <= vlen here
            shl(reg_rem_mask_short, cl);
            sub(reg_rem_mask_short, 1);
            jz(prologue_loop_end, T_NEAR);

            kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        } else {
            mov(reg_shift_table, vlen);
            sub(reg_shift_table, reg_tmp);
            uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
            if (dst_data_type_ == data_type::s8 || dst_data_type_ == data_type::u8) {
                mov(reg_shift_table, vlen * sizeof(float));
                sub(reg_shift_table, reg_tmp);
                uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
            }
        }
        compute(0, 0, true);
        advance_ptrs_reg(reg_tmp);

        L(prologue_loop_end);
        rewind_ptrs();
    }
    L(prologue_end);

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len, OC_);
        jl(main_loop_end, T_NEAR);

        size_t OC_loop, OC_tail;
        if (OC_ < max_OC_loop_unroll_ * vlen) {
            // Fully unroll small loops
            OC_loop = 0;
            OC_tail = OC_;
        } else {
            OC_loop = vlen * default_OC_loop_unroll_;
            OC_tail = OC_ % OC_loop;
        }

        assert(!!OC_loop || !!OC_tail);

        if (OC_tail % vlen) {
            int vlen_tail = OC_tail % vlen;
            if (isa == avx512_common) {
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask_short, reg_tmp);
            } else {
                mov(reg_shift_table, vlen - vlen_tail);
                uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
                if (dst_data_type_ == data_type::s8 || dst_data_type_ == data_type::u8) {
                    mov(reg_shift_table, vlen * sizeof(float));
                    sub(reg_shift_table, vlen_tail);
                    uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
                }
            }
        }

        Label main_loop;
        L(main_loop);
        {
            if (do_post_ops)
                mov(reg_oc_offset, 0);

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(OC_, OC_loop));
                Label oc_loop;
                L(oc_loop);
                {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop);
                    if (do_post_ops)
                        add(reg_oc_offset, OC_loop);
                    sub(reg_tmp, OC_loop);
                    jnz(oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, use_mask);
                }
                advance_ptrs_imm(OC_tail);
            }

            rewind_ptrs();
            sub(reg_len, OC_);
            cmp(reg_len, OC_);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        if (do_post_ops)
            mov(reg_oc_offset, 0);
        cmp(reg_len, vlen);
        jl(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop);
        {
            compute(0, 0, false);
            sub(reg_len, vlen);
            advance_ptrs_imm(vlen);
            if (do_post_ops)
                add(reg_oc_offset, vlen);
            cmp(reg_len, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        if (isa == avx512_common) {
            mov(reg_rem_mask_short, 1);
            shl(reg_rem_mask_short, cl); // reg_tmp == rcx and reg_tail < vlen
            sub(reg_rem_mask_short, 1);
            jz(epilogue_end, T_NEAR);
            kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        } else {
            mov(reg_shift_table, vlen);
            sub(reg_shift_table, reg_tmp);
            uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
            if (dst_data_type_ == data_type::s8 || dst_data_type_ == data_type::u8) {
                mov(reg_shift_table, vlen * sizeof(float));
                sub(reg_shift_table, reg_tmp);
                uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
            }
        }
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    for (auto& inj : jit_eltwise_injectors_)
        inj->prepare_table();

    if (utils::one_of(isa, avx2, sse41)) {
        align(64);
        L(l_table);
        for (size_t i = 0; i < vlen; i++) dd(0xFFFFFFFF);
        for (size_t i = 0; i < vlen; i++) dd(0x00000000);
    }
}

pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
    if (mayiuse(avx512_common)) {
        return new jit_pp_ker_t<avx512_common>(pd, jcp);
    } else if (mayiuse(avx2)) {
        return new jit_pp_ker_t<avx2>(pd, jcp);
    } else if (mayiuse(sse41)) {
        return new jit_pp_ker_t<sse41>(pd, jcp);
    }
    return nullptr;
}

//struct jit_pp_ker_t : pp_ker_t, public jit_generator {
//    DECLARE_CPU_JIT_AUX_FUNCTIONS(
//            gemm_x8s8s32x_convolution_utils::jit_pp_ker_t);
//
//    jit_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp);
//
//    status_t create_kernel() override { return jit_generator::create_kernel(); }
//    void operator()(void *void_dst, const acc_data_t *acc, const char *bias,
//            const float *scales, float sum_scale, float signed_scale, int g,
//            size_t start, size_t end, const zero_point_call_params_t &zp,
//            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
//            const exec_ctx_t & /* ctx */, const memory_desc_t & /* dst_md */,
//            const single_gemm_conv_chunk_desc_t &) const override;
//
//private:
//    void apply_postops(const Xbyak::Reg64 &reg_dst, const int idx);
//    void generate() override;
//    void append_zp_src_comp(size_t offset, int idx, bool apply_mask);
//    void load_as_f32(const Xbyak::Zmm &dst, const Xbyak::Opmask &mask,
//            const Xbyak::Address &src_addr, const data_type_t &src_dt);
//
//    int vreg_dst_idx(const int idx) const noexcept;
//    Xbyak::Zmm get_vreg_dst(int idx) const;
//    Xbyak::Zmm get_vreg_bias(int idx) const;
//    Xbyak::Zmm get_vreg_prev_dst(int idx) const;
//    Xbyak::Zmm get_vreg_zp_comp_src(int idx) const;
//    Xbyak::Zmm get_masked_vreg_dst(int idx, bool apply_mask) const;
//    Xbyak::Zmm reserve_zmm();
//
//    template <typename T>
//    void advance_binary_postops_off(const T &offset);
//    void zero_binary_postops_off();
//    void set_binary_postops_off(const Xbyak::Reg64 &reg);
//    const Xbyak::Opmask &opmask_binary = k2;
//
//    struct ker_args_t {
//        char *dst;
//        const acc_data_t *acc;
//        const char *bias;
//        const float *scales;
//        float sum_scale;
//        float signed_scale;
//        size_t len;
//        size_t oc_offset;
//        const int32_t *zp_src;
//        const int32_t *zp_dst;
//        const int32_t *zp_src_comp;
//        const int32_t *zp_src_pad_comp;
//        size_t g_oc_offset_prologue;
//        size_t g_oc_offset;
//        const void *post_ops_binary_rhs_arg_vec;
//        const void *dst_orig;
//        dim_t h;
//        dim_t w;
//        dim_t w_size;
//        dim_t w_off;
//        dim_t zp_src_pad_com_d_offset;
//        bool should_apply_zp_src_pad_comp_d;
//    };
//
//    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
//            postops_injector_;
//
//    size_t number_of_reserved_zmm_regs_;
//    const size_t bias_data_type_size_;
//    const size_t dst_data_type_size_;
//    const bool saturation_needed_;
//
//    const Xbyak::Reg64 &reg_param_ = rdi;
//    const Xbyak::Reg64 &reg_tmp_ = rcx; // intentional for shifting purposes
//
//    const Xbyak::Reg64 &reg_dst_ = rdx;
//    const Xbyak::Reg64 &reg_acc_ = rax;
//    const Xbyak::Reg64 &reg_bias_ = rbx;
//    const Xbyak::Reg64 &reg_scales_ = rsi;
//    const Xbyak::Reg64 &reg_len_ = r8;
//    const Xbyak::Reg64 &reg_oc_offset_ = r9;
//    const Xbyak::Reg64 &reg_rem_mask_short_ = r10;
//    const Xbyak::Reg64 &reg_rem_mask_vlen_ = reg_rem_mask_short_;
//    const Xbyak::Reg64 &reg_zp_pad_comp_temp_ = r10;
//    const Xbyak::Reg64 &reg_zp_pad_comp_ = r11;
//    const Xbyak::Reg8 &reg_should_apply_src_pad_comp_ = r13b;
//
//    const Xbyak::Reg64 &reg_tmp_comp_
//            = r12; // used to broadcast scalar values to vreg
//    const Xbyak::Reg64 &reg_g_oc_off_ = reg_tmp_comp_;
//    const Xbyak::Reg64 &reg_zp_src_comp_ = r14;
//
//    const Xbyak::Zmm vreg_zero_;
//    const Xbyak::Zmm vreg_scale_;
//    const Xbyak::Zmm vreg_sum_scale_;
//    const Xbyak::Zmm vreg_signed_scale_;
//    const Xbyak::Zmm vreg_saturation_ubound_;
//    const Xbyak::Zmm vreg_zp_dst_common_;
//
//    const Xbyak::Opmask &kreg_rem_mask_short_ = k3;
//    const Xbyak::Opmask &kreg_rem_mask_vlen_ = k4;
//
//    static constexpr size_t def_unroll_ = 4u;
//    size_t zmm_step_;
//    const size_t bias_step_factor_;
//    const size_t sum_step_factor_;
//    const size_t max_unroll_;
//    int dst_l_offset_ = 0;
//
//    std::unique_ptr<jit_gemm_x8s8s32x_zp_pad_comp_helper> zp_pad_comp_helper_;
//};

//jit_pp_ker_t::jit_pp_ker_t(
//        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
//    : pp_ker_t(pd, jcp)
//    , number_of_reserved_zmm_regs_(0)
//    , bias_data_type_size_(jcp.bias_data_type != data_type::undef
//                      ? types::data_type_size(jcp.bias_data_type)
//                      : 0u)
//    , dst_data_type_size_(types::data_type_size(jcp.dst_data_type))
//    , saturation_needed_(utils::one_of(
//              jcp_.dst_data_type, data_type::u8, data_type::s8, data_type::s32))
//    , vreg_zero_((jcp_.with_eltwise || saturation_needed_) ? reserve_zmm()
//                                                           : Xbyak::Zmm(0))
//    , vreg_scale_(reserve_zmm())
//    , vreg_sum_scale_(jcp_.with_sum ? reserve_zmm() : Xbyak::Zmm(0))
//    , vreg_signed_scale_(jcp_.signed_input ? reserve_zmm() : Xbyak::Zmm(0))
//    , vreg_saturation_ubound_(
//              saturation_needed_ ? reserve_zmm() : Xbyak::Zmm(0))
//    , vreg_zp_dst_common_(jcp_.zp.dst_exists ? reserve_zmm() : Xbyak::Zmm(0))
//    , zmm_step_(1u)
//    , bias_step_factor_(jcp_.with_bias ? zmm_step_++ : 0u)
//    , sum_step_factor_(jcp_.with_sum ? zmm_step_++ : 0)
//    , max_unroll_((cpu_isa_traits<avx512_core>::n_vregs
//                          - number_of_reserved_zmm_regs_)
//              / zmm_step_)
//    , zp_pad_comp_helper_(jit_gemm_convolution_utils::padding_exists(jcp)
//                              && jcp.zp.src_exists
//                      ? utils::make_unique<
//                              jit_gemm_x8s8s32x_zp_pad_comp_helper>(this, jcp_,
//                              reg_zp_pad_comp_, reg_zp_pad_comp_temp_,
//                              reg_should_apply_src_pad_comp_,
//                              pd->src_md()->ndims)
//                      : nullptr)
//
//{
//
//    if (jcp.with_eltwise || jcp.with_binary) {
//        using namespace binary_injector;
//        static constexpr bool preserve_gpr = true;
//        static constexpr bool preserve_vmm = true;
//        static constexpr size_t helper_vmm_idx = 31;
//        static constexpr size_t prelu_helper_vmm_idx = 30; // todo: [antonvor]
//        // tail_size = 1 just indicates that tailing is to be performed
//        // actual tail value is held in opmask passed to injector
//        static constexpr size_t tail_size = 1;
//        static constexpr bool use_exact_tail_scalar_bcast = false;
//
//#define PARAM_OFF(x) offsetof(ker_args_t, x)
//        const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
//                r13, r14, preserve_gpr, preserve_vmm,
//                PARAM_OFF(post_ops_binary_rhs_arg_vec),
//                memory_desc_wrapper(pd->dst_md()), tail_size, opmask_binary,
//                use_exact_tail_scalar_bcast, prelu_helper_vmm_idx};
//#undef PARAM_OFF
//
//        const static_params_t static_params {reg_param_, rhs_arg_static_params};
//
//        postops_injector_ = utils::make_unique<
//                injector::jit_uni_postops_injector_t<avx512_core>>(
//                this, jcp_.post_ops, static_params);
//    }
//}
//
//void jit_pp_ker_t::operator()(void *void_dst, const acc_data_t *acc,
//        const char *bias, const float *scales, float sum_scale,
//        float signed_scale, int g, size_t start, size_t end,
//        const zero_point_call_params_t &zp,
//        const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
//        const exec_ctx_t & /* ctx */, const memory_desc_t & /* dst_md */,
//        const single_gemm_conv_chunk_desc_t &chunk_desc) const {
//
//    if (end <= start) return;
//
//    char *dst = (char *)void_dst;
//
//    ker_args_t args;
//    const auto dv = std::div(start, jcp_.oc);
//    const size_t oc_offset = dv.rem;
//    const size_t os_offset = dv.quot;
//    args.acc = acc + start;
//    args.dst = dst
//            + (os_offset * jcp_.dst_os_stride + oc_offset)
//                    * dst_data_type_size_;
//
//    const ptrdiff_t g_oc_offset = g * jcp_.oc;
//    const ptrdiff_t g_oc_offset_prologue = g_oc_offset + oc_offset;
//    args.bias = bias + g_oc_offset_prologue * bias_data_type_size_;
//    args.zp_src = zp.src + (jcp_.zp.src_is_common ? 0 : g_oc_offset_prologue);
//    args.zp_src_comp
//            = zp.src_comp ? zp.src_comp + g_oc_offset_prologue : nullptr;
//    args.zp_dst = zp.dst;
//    args.scales = scales + jcp_.scale_idx_mult * g_oc_offset_prologue;
//    args.sum_scale = sum_scale;
//    args.signed_scale = signed_scale;
//    args.len = end - start;
//    args.oc_offset = oc_offset;
//
//    args.g_oc_offset = g_oc_offset;
//    args.g_oc_offset_prologue = g_oc_offset_prologue;
//
//    args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
//    args.dst_orig = dst_orig;
//
//    if (zp_pad_comp_helper_) {
//        const auto hw
//                = std::div(static_cast<dim_t>(os_offset), chunk_desc.w_size_);
//        args.h = hw.quot + chunk_desc.h_off_;
//        args.w = hw.rem + chunk_desc.w_off_;
//        args.w_size = chunk_desc.w_size_ + chunk_desc.w_off_;
//        args.w_off = chunk_desc.w_off_;
//        args.zp_src_pad_comp = zp.src_pad_comp;
//        const auto zp_src_pad_com_d
//                = zp_pad_comp_helper_->calculate_zp_src_pad_com_d(
//                        chunk_desc.d_off_);
//        args.zp_src_pad_com_d_offset = zp_src_pad_com_d.offset;
//        args.should_apply_zp_src_pad_comp_d
//                = zp_src_pad_com_d.should_apply_pad_comp_d;
//    }
//
//    jit_generator::operator()(&args);
//}
//
//template <typename T>
//void jit_pp_ker_t::advance_binary_postops_off(const T &offset) {
//    add(reg_g_oc_off_, offset);
//
//    Xbyak::Label end;
//    cmp(reg_g_oc_off_, jcp_.oc);
//    jl(end, T_NEAR);
//    xor_(reg_g_oc_off_, reg_g_oc_off_);
//
//    L(end);
//}
//void jit_pp_ker_t::zero_binary_postops_off() {
//    xor_(reg_g_oc_off_, reg_g_oc_off_);
//    dst_l_offset_ = 0;
//}
//void jit_pp_ker_t::set_binary_postops_off(const Xbyak::Reg64 &reg) {
//    mov(reg_g_oc_off_, reg);
//    dst_l_offset_ = 0;
//}
//
//Xbyak::Zmm jit_pp_ker_t::reserve_zmm() {
//    return Xbyak::Zmm(number_of_reserved_zmm_regs_++);
//}
//
//int jit_pp_ker_t::vreg_dst_idx(const int idx) const noexcept {
//    return (number_of_reserved_zmm_regs_ + idx * zmm_step_);
//}
//
//Xbyak::Zmm jit_pp_ker_t::get_vreg_dst(int idx) const {
//    return Xbyak::Zmm(vreg_dst_idx(idx));
//}
//
//Xbyak::Zmm jit_pp_ker_t::get_vreg_bias(int idx) const {
//    return Xbyak::Zmm(vreg_dst_idx(idx) + bias_step_factor_);
//}
//
//Xbyak::Zmm jit_pp_ker_t::get_vreg_prev_dst(int idx) const {
//    return Xbyak::Zmm(vreg_dst_idx(idx) + sum_step_factor_);
//}
//
//Xbyak::Zmm jit_pp_ker_t::get_masked_vreg_dst(int idx, bool apply_mask) const {
//    auto vreg_dst = this->get_vreg_dst(idx);
//    if (apply_mask)
//        vreg_dst = vreg_dst | kreg_rem_mask_short_;
//    else
//        vreg_dst = vreg_dst | kreg_rem_mask_vlen_;
//    return vreg_dst;
//}
//
//void jit_pp_ker_t::append_zp_src_comp(size_t offset, int idx, bool apply_mask) {
//    const auto vreg_dst_masked = get_masked_vreg_dst(idx, apply_mask);
//    const auto vreg_dst = get_vreg_dst(idx);
//    const auto zp_src_comp_offset = offset * sizeof(int32_t);
//    const auto zp_src_comp_addr = ptr[reg_zp_src_comp_ + zp_src_comp_offset];
//
//    vpaddd(vreg_dst_masked, vreg_dst, zp_src_comp_addr);
//
//    if (zp_pad_comp_helper_)
//        zp_pad_comp_helper_->zp_src_comp_pad_operation(
//                [&](const Xbyak::Reg64 &reg_zp_pad_comp) {
//                    vpaddd(vreg_dst_masked, vreg_dst,
//                            ptr[reg_zp_pad_comp + zp_src_comp_offset]);
//                });
//}
//
//void jit_pp_ker_t::apply_postops(const Xbyak::Reg64 &reg_dst, const int idx) {
//#define PARAM_OFF(x) offsetof(ker_args_t, x)
//    if (jcp_.with_eltwise || jcp_.with_binary) {
//        if (jcp_.with_binary) {
//            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
//            const auto dst_offset_reg = reg_dst;
//            const auto vmm_idx = vreg_dst_idx(idx);
//            rhs_arg_params.vmm_idx_to_oc_elem_off_addr.emplace(
//                    vmm_idx, ptr[reg_param_ + PARAM_OFF(g_oc_offset)]);
//            rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
//                    vmm_idx, reg_g_oc_off_);
//            rhs_arg_params.vmm_idx_to_out_off_oprnd.emplace(
//                    vmm_idx, dst_offset_reg);
//            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
//                    vmm_idx, dst_l_offset_);
//            rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
//
//            const injector_utils::register_preserve_guard_t register_guard(
//                    this, {dst_offset_reg});
//            sub(dst_offset_reg, ptr[reg_param_ + PARAM_OFF(dst_orig)]);
//            const auto size = sizeof(jcp_.dst_data_type);
//            if (size) shr(dst_offset_reg, std::log2(size));
//
//            postops_injector_->compute_vector(
//                    vreg_dst_idx(idx), rhs_arg_params);
//        } else
//            postops_injector_->compute_vector(vreg_dst_idx(idx));
//    }
//#undef PARAM_OFF
//}
//
//void jit_pp_ker_t::load_as_f32(const Xbyak::Zmm &dst,
//        const Xbyak::Opmask &mask_reg, const Xbyak::Address &src_addr,
//        const data_type_t &src_dt) {
//
//    const auto dst_masked = dst | mask_reg;
//
//    switch (src_dt) {
//        case data_type::s8: vpmovsxbd(dst_masked, src_addr); break;
//        case data_type::u8: vpmovzxbd(dst_masked, src_addr); break;
//        case data_type::s32: vcvtdq2ps(dst_masked, src_addr); break;
//        case data_type::f32: vmovups(dst_masked, src_addr); break;
//        default: assert(!"unimplemented");
//    }
//
//    if (utils::one_of(src_dt, data_type::s8, data_type::u8))
//        vcvtdq2ps(dst_masked, dst);
//}
//
//void jit_pp_ker_t::generate() {
//    using namespace Xbyak;
//    using namespace utils;
//
//    size_t vlen = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
//    for (; vlen >= 1 && (jcp_.oc % vlen != 0); --vlen) {}
//
//    preamble();
//
//#ifdef _WIN32
//    mov(reg_param_, rcx);
//#endif
//
//#define PARAM_OFF(x) offsetof(ker_args_t, x)
//    mov(reg_dst_, ptr[reg_param_ + PARAM_OFF(dst)]);
//    mov(reg_acc_, ptr[reg_param_ + PARAM_OFF(acc)]);
//    mov(reg_bias_, ptr[reg_param_ + PARAM_OFF(bias)]);
//    mov(reg_scales_, ptr[reg_param_ + PARAM_OFF(scales)]);
//    mov(reg_len_, ptr[reg_param_ + PARAM_OFF(len)]);
//    mov(reg_oc_offset_, ptr[reg_param_ + PARAM_OFF(oc_offset)]);
//
//    if (jcp_.zp.src_exists) {
//        mov(reg_zp_src_comp_, ptr[reg_param_ + PARAM_OFF(zp_src_comp)]);
//        if (zp_pad_comp_helper_)
//            zp_pad_comp_helper_->init(PARAM_OFF(w), PARAM_OFF(h),
//                    PARAM_OFF(w_size), PARAM_OFF(w_off),
//                    PARAM_OFF(zp_src_pad_comp), PARAM_OFF(g_oc_offset_prologue),
//                    PARAM_OFF(g_oc_offset), PARAM_OFF(zp_src_pad_com_d_offset),
//                    PARAM_OFF(should_apply_zp_src_pad_comp_d));
//    }
//
//    if (jcp_.zp.dst_exists) {
//        mov(reg_tmp_, ptr[reg_param_ + PARAM_OFF(zp_dst)]);
//        vcvtdq2ps(vreg_zp_dst_common_, ptr_b[reg_tmp_]);
//    }
//
//    if (jcp_.with_sum)
//        vbroadcastss(vreg_sum_scale_, ptr[reg_param_ + PARAM_OFF(sum_scale)]);
//    if (jcp_.signed_input)
//        vbroadcastss(
//                vreg_signed_scale_, ptr[reg_param_ + PARAM_OFF(signed_scale)]);
//    if (jcp_.scale_idx_mult == 0) vbroadcastss(vreg_scale_, dword[reg_scales_]);
//#undef PARAM_OFF
//
//    mov(reg_rem_mask_vlen_, 1);
//    shl(reg_rem_mask_vlen_, vlen);
//    sub(reg_rem_mask_vlen_, 1);
//    kmovq(kreg_rem_mask_vlen_, reg_rem_mask_vlen_);
//
//    if (jcp_.with_eltwise) vxorps(vreg_zero_, vreg_zero_, vreg_zero_);
//    if (saturation_needed_)
//        init_saturate_f32(vreg_zero_, vreg_saturation_ubound_, reg_tmp_comp_,
//                data_type::f32, jcp_.dst_data_type);
//
//    if (jcp_.with_binary) set_binary_postops_off(reg_oc_offset_);
//
//    // Load accumulated value, convert to float, apply sum (if any),
//    // bias (if any), scaling, and relu (if any);
//    // then convert to destination type and store
//    const auto compute = [&](size_t offset, int idx, bool apply_mask) {
//        auto acc_addr = ptr[reg_acc_ + offset * sizeof(acc_data_t)];
//
//        const auto &mask_reg
//                = apply_mask ? kreg_rem_mask_short_ : kreg_rem_mask_vlen_;
//
//        if (jcp_.scale_idx_mult > 0) {
//            assert(jcp_.scale_idx_mult == 1);
//            const auto scale_addr = ptr[reg_scales_ + offset * sizeof(float)];
//            auto vreg_scale = vreg_scale_;
//            vreg_scale = vreg_scale | mask_reg;
//            vmovups(vreg_scale, scale_addr);
//        }
//
//        if (jcp_.with_binary) {
//            if (offset) {
//                advance_binary_postops_off(vlen);
//                dst_l_offset_ += offset;
//            }
//            kmovq(opmask_binary, mask_reg);
//        }
//        const auto vreg_dst_masked = get_masked_vreg_dst(idx, apply_mask);
//        const auto vreg_dst = get_vreg_dst(idx);
//        if (jcp_.zp.src_exists) {
//            vmovups(vreg_dst_masked, acc_addr);
//            append_zp_src_comp(offset, idx, apply_mask);
//            vcvtdq2ps(vreg_dst_masked, vreg_dst);
//        } else {
//            vcvtdq2ps(vreg_dst_masked, acc_addr);
//        }
//
//        if (jcp_.signed_input)
//            vmulps(vreg_dst_masked, vreg_dst, vreg_signed_scale_);
//
//        if (jcp_.with_bias) {
//            const auto bias_addr
//                    = ptr[reg_bias_ + offset * bias_data_type_size_];
//            const auto vreg_bias = get_vreg_bias(idx);
//            load_as_f32(vreg_bias, mask_reg, bias_addr, jcp_.bias_data_type);
//            vaddps(vreg_dst_masked, vreg_dst, vreg_bias);
//        }
//
//        vmulps(vreg_dst_masked, vreg_dst, vreg_scale_);
//
//        const auto dst_addr = ptr[reg_dst_ + offset * dst_data_type_size_];
//
//        if (jcp_.with_sum) {
//            const auto vreg_prev_dst = get_vreg_prev_dst(idx);
//            load_as_f32(vreg_prev_dst, mask_reg, dst_addr, jcp_.dst_data_type);
//            vfmadd231ps(vreg_dst_masked, vreg_prev_dst, vreg_sum_scale_);
//        }
//
//        apply_postops(reg_dst_, idx);
//
//        if (jcp_.zp.dst_exists) {
//            vaddps(vreg_dst_masked, vreg_dst, vreg_zp_dst_common_);
//        }
//
//        if (saturation_needed_) {
//            saturate_f32(get_vreg_dst(idx), vreg_zero_, vreg_saturation_ubound_,
//                    jcp_.dst_data_type);
//            vcvtps2dq(vreg_dst_masked, vreg_dst);
//        }
//
//        switch (jcp_.dst_data_type) {
//            case data_type::s8: vpmovsdb(dst_addr, vreg_dst_masked); break;
//            case data_type::u8: vpmovusdb(dst_addr, vreg_dst_masked); break;
//            case data_type::f32:
//            case data_type::s32: vmovups(dst_addr, vreg_dst_masked); break;
//            default: assert(!"unimplemented");
//        }
//    };
//
//    // Advance all pointers by an immediate
//    const auto advance_ptrs_imm = [&](const size_t offset,
//                                          const size_t binary_offset) {
//        add(reg_dst_, offset * dst_data_type_size_);
//        add(reg_acc_, offset * sizeof(acc_data_t));
//        if (jcp_.with_binary) { advance_binary_postops_off(binary_offset); }
//        if (jcp_.scale_idx_mult) {
//            assert(jcp_.scale_idx_mult == 1);
//            add(reg_scales_, offset * sizeof(float));
//        }
//        if (jcp_.with_bias) add(reg_bias_, offset * bias_data_type_size_);
//        if (jcp_.zp.src_exists) {
//            add(reg_zp_src_comp_, offset * sizeof(int32_t));
//
//            if (zp_pad_comp_helper_) {
//                zp_pad_comp_helper_->zp_src_comp_pad_operation(
//                        [&](const Xbyak::Reg64 &reg_zp_pad_comp) {
//                            add(reg_zp_pad_comp, offset * sizeof(int32_t));
//                        });
//            }
//        }
//    };
//
//    // Advance all pointers by a value stored in a register
//    const auto advance_ptrs_reg = [&](const Reg64 offset,
//                                          const Reg64 binary_offset) {
//        lea(reg_dst_, ptr[reg_dst_ + offset * dst_data_type_size_]);
//        lea(reg_acc_, ptr[reg_acc_ + offset * sizeof(acc_data_t)]);
//        if (jcp_.with_binary) { advance_binary_postops_off(binary_offset); }
//        if (jcp_.scale_idx_mult) {
//            assert(jcp_.scale_idx_mult == 1);
//            lea(reg_scales_, ptr[reg_scales_ + offset * sizeof(float)]);
//        }
//        if (jcp_.with_bias)
//            lea(reg_bias_, ptr[reg_bias_ + offset * bias_data_type_size_]);
//
//        if (jcp_.zp.src_exists) {
//            lea(reg_zp_src_comp_,
//                    ptr[reg_zp_src_comp_ + offset * sizeof(int32_t)]);
//
//            if (zp_pad_comp_helper_)
//                zp_pad_comp_helper_->zp_src_comp_pad_operation(
//                        [&](const Xbyak::Reg64 &reg_zp_pad_comp) {
//                            lea(reg_zp_pad_comp,
//                                    ptr[reg_zp_pad_comp
//                                            + offset * sizeof(int32_t)]);
//                        });
//        }
//    };
//
//    // Rewind pointers that point to data that is indexed by output channel
//    // (bias or per-oc scaling factors)
//    const auto rewind_ptrs = [&]() {
//        if (jcp_.with_bias) sub(reg_bias_, jcp_.oc * bias_data_type_size_);
//        if (jcp_.with_binary) {
//            zero_binary_postops_off();
//            dst_l_offset_ = 0;
//        }
//        if (jcp_.zp.src_exists) {
//            const auto offset = jcp_.oc * sizeof(int32_t);
//            sub(reg_zp_src_comp_, offset);
//            if (zp_pad_comp_helper_)
//                zp_pad_comp_helper_->load_next_point_zp_src_comp_pad_addr();
//        }
//        if (jcp_.scale_idx_mult) {
//            assert(jcp_.scale_idx_mult == 1);
//            sub(reg_scales_, jcp_.oc * sizeof(float));
//        }
//        add(reg_dst_, (jcp_.dst_os_stride - jcp_.oc) * dst_data_type_size_);
//    };
//
//    //                    <--------- OC --------------->
//    //
//    // ^  ................+..............+-------------+.......................
//    // |  .               : not accessed |Prologue loop|                      .
//    // |  .               +--------------+-------------+                      .
//    //    .               |                            |                      .
//    // O  .               |  Main loop (unrolled)      |                      .
//    // S  .               |                            |                      .
//    //    .               +--------------+-------------+                      .
//    // |  .               | Epilogue loop|not accessed :                      .
//    // v  ................+--------------+.............+.......................
//
//    Label prologue_end;
//    cmp(reg_oc_offset_, 0);
//    je(prologue_end, T_NEAR);
//
//    // Prologue loop
//    {
//        mov(reg_tmp_, jcp_.oc);
//        sub(reg_tmp_, reg_oc_offset_);
//        cmp(reg_tmp_, reg_len_);
//        cmovg(reg_tmp_, reg_len_);
//        sub(reg_len_, reg_tmp_);
//
//        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
//        cmp(reg_tmp_, vlen);
//        jle(prologue_loop_tail, T_NEAR);
//        L(prologue_loop);
//        {
//            compute(0, max_unroll_ - 1, false);
//            advance_ptrs_imm(vlen, vlen);
//            sub(reg_tmp_, vlen);
//            cmp(reg_tmp_, vlen);
//            jge(prologue_loop, T_NEAR);
//        }
//
//        L(prologue_loop_tail);
//        mov(reg_rem_mask_short_, 1);
//        // cl == reg_tmp_ because reg_tmp_ <= vlen here
//        shl(reg_rem_mask_short_, cl);
//        sub(reg_rem_mask_short_, 1);
//        jz(prologue_loop_end, T_NEAR);
//
//        kmovq(kreg_rem_mask_short_, reg_rem_mask_short_);
//        compute(0, max_unroll_ - 1, true);
//        advance_ptrs_reg(reg_tmp_, reg_tmp_);
//
//        L(prologue_loop_end);
//        rewind_ptrs();
//    }
//    L(prologue_end);
//
//    // Main loop
//    Label main_loop_end;
//    {
//        cmp(reg_len_, jcp_.oc);
//        jle(main_loop_end, T_NEAR);
//
//        Label main_loop;
//        L(main_loop);
//        {
//            size_t OC_loop, OC_tail;
//            if (static_cast<size_t>(jcp_.oc) < max_unroll_ * vlen) {
//                // Fully unroll small loops
//                OC_loop = 0;
//                OC_tail = jcp_.oc;
//            } else {
//                OC_loop = vlen * def_unroll_;
//                OC_tail = jcp_.oc % OC_loop;
//            }
//
//            assert(!!OC_loop || !!OC_tail);
//
//            const int vlen_tail = OC_tail % vlen;
//            if (vlen_tail) {
//                unsigned tail_mask = (1 << vlen_tail) - 1;
//                mov(reg_tmp_, tail_mask);
//                kmovq(kreg_rem_mask_short_, reg_tmp_);
//            }
//
//            if (OC_loop) {
//                mov(reg_tmp_, rnd_dn(jcp_.oc, OC_loop));
//                Label oc_loop;
//                L(oc_loop);
//                {
//                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
//                        compute(offset, offset / vlen, false);
//                    advance_ptrs_imm(OC_loop, vlen);
//                    sub(reg_tmp_, OC_loop);
//                    jnz(oc_loop);
//                }
//            }
//
//            if (OC_tail) {
//                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
//                    bool use_mask = (offset + vlen) > OC_tail;
//                    compute(offset, offset / vlen, use_mask);
//                }
//                const size_t oc_tail_rem = OC_tail % vlen;
//                const size_t binary_offset = oc_tail_rem ? oc_tail_rem : vlen;
//                advance_ptrs_imm(OC_tail, binary_offset);
//            }
//
//            rewind_ptrs();
//            sub(reg_len_, jcp_.oc);
//            cmp(reg_len_, jcp_.oc);
//            jge(main_loop, T_NEAR);
//        }
//    }
//    L(main_loop_end);
//
//    // Epilogue loop
//    Label epilogue_end;
//    {
//        cmp(reg_len_, 0);
//        je(epilogue_end, T_NEAR);
//
//        Label epilogue_loop, epilogue_loop_tail;
//        cmp(reg_len_, vlen);
//        jle(epilogue_loop_tail, T_NEAR);
//        L(epilogue_loop);
//        {
//            compute(0, 0, false);
//            sub(reg_len_, vlen);
//            advance_ptrs_imm(vlen, vlen);
//            cmp(reg_len_, vlen);
//            jge(epilogue_loop, T_NEAR);
//        }
//
//        L(epilogue_loop_tail);
//        mov(reg_tmp_,
//                reg_len_); // reg_tmp_ is rcx, and we need cl for the shift
//        mov(reg_rem_mask_short_, 1);
//        shl(reg_rem_mask_short_, cl); // reg_tmp_ == rcx and reg_tail < vlen
//        sub(reg_rem_mask_short_, 1);
//        jz(epilogue_end, T_NEAR);
//        kmovq(kreg_rem_mask_short_, reg_rem_mask_short_);
//        compute(0, 0, true);
//    }
//
//    L(epilogue_end);
//
//    if (zp_pad_comp_helper_) zp_pad_comp_helper_->fin();
//
//    postamble();
//
//    if (jcp_.with_eltwise) postops_injector_->prepare_table();
//}
//
//bool mayiuse_jit_pp_kernel() noexcept {
//    return mayiuse(avx512_core);
//}
//
//pp_ker_t *jit_pp_ker_create(
//        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
//    return mayiuse_jit_pp_kernel() ? new jit_pp_ker_t(pd, jcp) : nullptr;
//}
//
//bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d) {
//    using namespace x64::injector;
//    static constexpr bool sum_at_pos_0_only = true;
//    static constexpr bool sum_requires_scale_one = false;
//    return mayiuse_jit_pp_kernel()
//            && dnnl::impl::cpu::x64::injector::post_ops_ok(
//                    {avx512_core, {binary, eltwise, sum}, post_ops, dst_d,
//                            sum_at_pos_0_only, sum_requires_scale_one});
//}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
