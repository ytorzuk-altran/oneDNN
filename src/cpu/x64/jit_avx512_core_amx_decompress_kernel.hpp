/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef CPU_X64_JIT_AVX512_CORE_AMX_DECOMPRESS_KERNEL_HPP
#define CPU_X64_JIT_AVX512_CORE_AMX_DECOMPRESS_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_avx512_core_amx_decompress_kernel_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_amx_decompress_kernel_t)
    jit_avx512_core_amx_decompress_kernel_t(jit_conv_conf_t ajsp)
        : jcp(ajsp) {}

    ~jit_avx512_core_amx_decompress_kernel_t() {}

    status_t create_kernel() override {
        CHECK(jit_generator::create_kernel());
        return status::success;
    }
    static status_t init_conf();

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);
            
    jit_conv_conf_t jcp;

private:
    /* compression */
    Xbyak::Reg64 wei_ptr = r14;
    Xbyak::Reg64 dst_ptr = r13;
        
    Xbyak::Zmm zmm_comp2 = Xbyak::Zmm(28);
    Xbyak::Zmm zmm_comp1 = Xbyak::Zmm(27);
    Xbyak::Zmm zmm_comp4 = Xbyak::Zmm(26);
    Xbyak::Zmm zmm_comp3 = Xbyak::Zmm(25);

    const Xbyak::Reg64 reg_ptr_decomp_src = r9;
    const Xbyak::Reg64 reg_ptr_decomp_dst = r8; //r10;
    const Xbyak::Reg64 reg_ptr_decomp_mask = rax; //rsi;
    const Xbyak::Reg64 reg_popcnt = rsi;

    const Xbyak::Reg64 reg_comp_mask_tmp1 = r10;
    const Xbyak::Reg64 reg_comp_mask_tmp2 = r12;
    const Xbyak::Reg64 reg_comp_mask_tmp3 = rbx;
    const Xbyak::Reg64 reg_comp_mask_tmp4 = rdx;

    const Xbyak::Opmask reg_comp_mask1 = k1;
    const Xbyak::Opmask reg_comp_mask2 = k2;
    const Xbyak::Opmask reg_comp_mask3 = k3;
    const Xbyak::Opmask reg_comp_mask4 = k4;

    size_t get_wei_icb_step() const;
    size_t get_wei_h_step() const;
    size_t get_wei_offset(int ocb, int kw) const;

    int get_wei_tensor(int i) const;

    void generate() override;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
