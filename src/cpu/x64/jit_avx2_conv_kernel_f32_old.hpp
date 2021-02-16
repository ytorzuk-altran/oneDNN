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

#ifndef CPU_X64_JIT_AVX2_CONV_KERNEL_F32_OLD_HPP
#define CPU_X64_JIT_AVX2_CONV_KERNEL_F32_OLD_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_uni_depthwise_injector.hpp"
#include "cpu/x64/jit_uni_quantization_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx2_conv_kernel_f32_old : public jit_generator {
    jit_avx2_conv_kernel_f32_old(jit_conv_conf_t ajcp, jit_conv_conf_t ajcp_dw,
                                 const primitive_attr_t &attr)
            : jcp(ajcp), jcp_dw(ajcp_dw), attr_(attr) {}

    ~jit_avx2_conv_kernel_f32_old() {
        for (auto inj : eltwise_injectors)
            delete inj;
        eltwise_injectors.clear();

        for (auto inj : depthwise_injectors)
            delete inj;
        depthwise_injectors.clear();

        for (auto inj : quantization_injectors)
            delete inj;
        quantization_injectors.clear();
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_kernel_f32_old)

    static bool post_ops_ok(jit_conv_conf_t &jcp,
                            const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
                              const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
                              const memory_desc_wrapper &weights_d,
                              const memory_desc_wrapper &dst_d,
                              const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
                                const jit_conv_conf_t &jcp, const jit_conv_conf_t &jcp_dw = jit_conv_conf_t());

    jit_conv_conf_t jcp;
    jit_conv_conf_t jcp_dw;
    const primitive_attr_t &attr_;

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t aux_reg_inp_d = r11;
    reg64_t aux_reg_ker_d = abi_not_param1;

    reg64_t reg_ki = rsi;
    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_oc_blocks = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_long_offt = r15;
    Xbyak::Reg32 reg_ci_flag = r13d;

    Xbyak::Ymm ytmp = Xbyak::Ymm(14);

    reg64_t reg_d_weights = imm_addr64;
    reg64_t reg_d_bias = ki_iter;

    Xbyak::Ymm ymm_d_weights = Xbyak::Ymm(14);
    Xbyak::Ymm ymm_d_bias = Xbyak::Ymm(15);

    Xbyak::Ymm ymm_sum = Xbyak::Ymm(15);

    nstl::vector<jit_uni_eltwise_injector_f32<avx2> *> eltwise_injectors;
    nstl::vector<jit_uni_depthwise_injector_f32<avx2> *> depthwise_injectors;
    nstl::vector<jit_uni_quantization_injector_f32<avx2> *> quantization_injectors;

    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r,
                                  int oc_blocks);

    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r,
                              char pad_label, int oc_blocks, char oc_blocks_label);

    inline void width_blk_step(int ur_w, int pad_l, int pad_r,
                               char pad_label, int oc_blocks, char oc_blocks_label);

    inline void solve_common(int oc_blocks, char oc_blocks_label);

    void generate() override;
};

}
}
}
}
#endif
