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
#include <float.h>

#include "common/memory_tracking.hpp"
#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_decompress_kernel.hpp"

#define GET_OFF(field) offsetof(brgemm_decomp_kernel_params_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

void jit_brgemm_decompress_kernel::generate() {
    preamble();
    mov(wei_ptr, ptr[param1 + GET_OFF(ptr_B)]);
    mov(reg_ptr_decomp_mask, ptr[param1 + GET_OFF(bitmask_ptr)]);
    mov(reg_ptr_decomp_dst, ptr[param1 + GET_OFF(scratch_buf)]);
    mov(reg_blocks, ptr[param1 + GET_OFF(blocks)]);

    for(int block = 0; block < blocks_; block++){
        int wei_offset =  block * 4096;
        lea(reg_ptr_decomp_src, ptr[wei_ptr +wei_offset]);
        int bitmask_off = wei_offset / (1 * 8);
        for (int cl = 0; cl < 64; cl++) {           
            vmovdqu8(zmm_comp, ptr[reg_ptr_decomp_src]);
            mov(reg_comp_mask_tmp, ptr[reg_ptr_decomp_mask + cl * 8 + bitmask_off]);
            kmovq(reg_comp_mask, reg_comp_mask_tmp);
            vpexpandb(zmm_comp | reg_comp_mask | T_z, zmm_comp);
            vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + cl * 64], zmm_comp);
            popcnt(reg_popcnt, reg_comp_mask_tmp);
            add(reg_ptr_decomp_src, reg_popcnt);
        }
    }
    postamble();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
