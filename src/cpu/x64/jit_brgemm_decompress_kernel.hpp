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

struct jit_brgemm_inner_product_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_inner_product_kernel)
    jit_brgemm_inner_product_kernel() {}

    ~jit_brgemm_inner_product_kernel() {}

    status_t create_kernel() override {
        CHECK(jit_generator::create_kernel());
        return status::success;
    }
           
private:

    /* compression */
    Xbyak::Reg64 wei_ptr = r14;
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
