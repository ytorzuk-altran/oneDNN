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

#include "mkldnn_itt.hpp"
#include <cstdlib>

#ifdef ENABLE_PROFILING_ITT
#include <ittnotify.h>
#endif

namespace mkldnn {
namespace impl {
namespace internal {

#ifdef ENABLE_PROFILING_ITT

domain_t domain(char const* name)
{
    return reinterpret_cast<domain_t>(__itt_domain_create(name));
}

handle_t handle(char const* name)
{
    return reinterpret_cast<handle_t>(__itt_string_handle_create(name));
}

void taskBegin(domain_t d, handle_t t)
{
    __itt_task_begin(reinterpret_cast<__itt_domain*>(d),
                    __itt_null,
                    __itt_null,
                    reinterpret_cast<__itt_string_handle*>(t));
}

void taskEnd(domain_t d)
{
    __itt_task_end(reinterpret_cast<__itt_domain*>(d));
}

void threadName(const char* name)
{
    __itt_thread_set_name(name);
}

#else

domain_t domain(char const *) { return nullptr; }

handle_t handle(char const *) { return nullptr; }

void taskBegin(domain_t, handle_t) { }

void taskEnd(domain_t) { }

void threadName(const char *) { }

#endif

}
}
}
