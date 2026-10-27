#pragma once

#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::layernorm {

struct Config {
  int dim = 0;
  double eps = 0.0; // <=0 → 1e-5
};

Status validate(Config* cfg);

struct Layer {
  Config cfg{};
  WeightStore gamma; // 1×Dim
  WeightStore beta;  // 1×Dim
};

Status new_configured(const Config& cfg, DType dt, const float* gamma_init, const float* beta_init,
                      Layer* out);

// Forward on CUDA: pre = (x−μ)/σ; post = γ⊙x̂ + β.
Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// NormalBP: dγ|dβ and dx; pre must be x̂ from forward.
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

} // namespace welvet::layernorm
