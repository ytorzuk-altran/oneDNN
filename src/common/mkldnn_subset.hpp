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

#pragma once
#include "mkldnn_macros.hpp"

#ifdef MKLDNN_SUBSET_FIND           // MKLDNN analysis
#include "mkldnn_itt.hpp"

namespace mkldnn {
namespace impl {
namespace domains {

MKLDNN_ITT_DOMAIN(CC0_MKLDNN);
MKLDNN_ITT_DOMAIN(CC1_MKLDNN);
MKLDNN_ITT_DOMAIN(CC2_MKLDNN);

} // namespace domains

#define MKLDNN_CSCOPE(region, ...)                                                           \
    MKLDNN_ITT_SCOPED_TASK(mkldnn::impl::domains::CC0_MKLDNN, MKLDNN_MACRO_TOSTRING(region));   \
    __VA_ARGS__

} // namespace impl
} // namespace mkldnn

#elif defined(MKLDNN_SUBSET)        // MKLDNN subset is used

// Scope is disabled
#define MKLDNN_CSCOPE_0(...)

// Scope is enabled
#define MKLDNN_CSCOPE_1(...) __VA_ARGS__

#define MKLDNN_CSCOPE(region, ...)      \
    MKLDNN_MACRO_EXPAND(MKLDNN_MACRO_CAT(MKLDNN_CSCOPE_, MKLDNN_MACRO_IS_ENABLED(MKLDNN_MACRO_CAT(MKLDNN_, region)))(__VA_ARGS__))

#else

#define MKLDNN_CSCOPE(region, ...) __VA_ARGS__

#endif
