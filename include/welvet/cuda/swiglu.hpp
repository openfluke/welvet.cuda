#pragma once

#include "welvet/status.hpp"

namespace welvet::cuda {

// h[i] = float(double(float(SiLU(double(gate_pre[i])))) * double(up_post[i]))
// Matches Go/cpp SwiGLU product (exact_bit). Host f32 buffers of length n.
Status launch_swiglu_product(const float* gate_pre_host, const float* up_post_host, float* h_host,
                             int n);

// SwiGLU NormalBP SiLU′ product (exact_bit vs Go/cpp):
//   silu_f = float(SiLU(double(gate_pre)))
//   dsilu_f = float(SiLU′(double(gate_pre)))
//   d_up = float(dh * double(silu_f))
//   d_gate = float(dh * double(up_post) * double(dsilu_f))
Status launch_swiglu_bwd_product(const float* gate_pre_host, const float* up_post_host,
                                 const float* grad_h_host, float* d_up_host, float* d_gate_host,
                                 int n);

} // namespace welvet::cuda
