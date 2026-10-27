#pragma once

#include "welvet/status.hpp"

namespace welvet::cuda {

// h[i] = float(double(float(SiLU(double(gate_pre[i])))) * double(up_post[i]))
// Matches Go/cpp SwiGLU product (exact_bit). Host f32 buffers of length n.
Status launch_swiglu_product(const float* gate_pre_host, const float* up_post_host, float* h_host,
                             int n);

} // namespace welvet::cuda
