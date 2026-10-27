#pragma once

#include "welvet/status.hpp"

namespace welvet::cuda {

// Shared RMSNorm / LayerNorm forward on CUDA (HW f64).
// center=0: RMSNorm  pre=x/rms, post=pre*γ
// center=1: LayerNorm pre=(x-μ)/σ, post=pre*γ+β
// gamma/beta: host f64 [dim]; beta may be null when center=0.
// x/pre/post: host f32 [n_tok * dim]
Status launch_norm_forward(const float* x_host, const double* gamma_f64, const double* beta_f64,
                           float* pre_host, float* post_host, int n_tok, int dim, double eps,
                           int center);

} // namespace welvet::cuda
