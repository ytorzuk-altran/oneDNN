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

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::utils;
using namespace Xbyak;

void jit_brgemm_inner_product_kernel::generate() {
    preamble();
    postamble();
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
