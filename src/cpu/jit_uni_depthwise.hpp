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

#ifndef CPU_JIT_UNI_DEPTHWISE_HPP
#define CPU_JIT_UNI_DEPTHWISE_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "cpu_depthwise_pd.hpp"
#include "cpu_engine.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_uni_depthwise_kernel_f32;

template <cpu_isa_t isa>
struct jit_uni_depthwise_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_depthwise_fwd_pd_t {
        pd_t(engine_t *engine, const depthwise_desc_t *adesc,
                const primitive_attr_t *attr,
                const depthwise_fwd_pd_t *hint_fwd_pd)
            : cpu_depthwise_fwd_pd_t(engine, adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_depthwise_fwd_t<isa>);

        virtual status_t init() override;
    };

    jit_uni_depthwise_fwd_t(const pd_t *apd, const input_vector &inputs,
                       const output_vector &outputs);
    ~jit_uni_depthwise_fwd_t();

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual void execute(event_t *e) const
    {
        execute_forward();
        e->set_state(event_t::ready);
    }

private:
    void execute_forward() const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_uni_depthwise_kernel_f32 *kernel_;
    data_t *padded_weights_;
    data_t *padded_bias_;
};

}
}
}

#endif
