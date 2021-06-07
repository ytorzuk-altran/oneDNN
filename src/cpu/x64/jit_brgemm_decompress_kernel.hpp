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

#ifndef CPU_X64_JIT_BRGEMM_DECOMPRESS_KERNEL_HPP
#define CPU_X64_JIT_BRGEMM_DECOMPRESS_KERNEL_HPP

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_brgemm_decompress_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_decompress_kernel)

    jit_brgemm_decompress_kernel(const jit_brgemm_primitive_conf_t *jbgp)
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx) {
        blocks_ = jbgp->ic * 64 / 4096;
        
        create_kernel();
        // printf("jit_brgemm_decompress_kernel created\n");
    }


    void tile_configure(const char *palette) const { (*this)(palette); }

private:
    int blocks_;

    Xbyak::Reg64 reg_blocks = r12;
    Xbyak::Reg64 wei_ptr = r14;
    Xbyak::Reg64 dst_ptr = r13;

    const Xbyak::Zmm zmm_comp = Xbyak::Zmm(28);

    const Xbyak::Reg64 reg_bias = r11;
    const Xbyak::Reg64 reg_ptr_decomp_src = r9;
    const Xbyak::Reg64 reg_ptr_decomp_dst = r8; //r10;
    const Xbyak::Reg64 reg_ptr_decomp_mask = rax; //rsi;
    const Xbyak::Reg64 reg_popcnt = reg_bias;
    const Xbyak::Reg64 reg_comp_mask_tmp = reg_bias;
    const Xbyak::Opmask reg_comp_mask = k1;
    
    void generate() override;
};


} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
