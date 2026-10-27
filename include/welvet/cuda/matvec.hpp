#pragma once

#include <cstddef>
#include <cstdint>

#include "welvet/activate.hpp"
#include "welvet/status.hpp"

namespace welvet::cuda {

// Resident Dense forward on CUDA: FormatNone decoded to f64 W on mount,
// then warm execute is upload-x → f64 MatVec → bias+act → download.
// Native HW float64 matches Go's f64-accumulate MatVec contract (exact_bit).
struct ResidentDense {
  void* W = nullptr;     // device f64[out*in]
  void* bias = nullptr;  // device f64[out] or null
  void* x = nullptr;     // device f32[batch*in]
  void* pre = nullptr;   // device f32[batch*out]
  void* post = nullptr;  // device f32[batch*out]
  int batch = 0;
  int out = 0;
  int in = 0;
  int bias_len = 0;
  Activation act = Activation::Linear;
};

Status resident_mount(ResidentDense* r, const double* W_f64, const double* bias_f64, int bias_len,
                      int batch, int out, int in, Activation act);
Status resident_execute(ResidentDense* r, const float* x_host, float* pre_host, float* post_host);
// Dense NormalBP: act′ → MatVecᵀ → outer (HW f64 accumulate, Go FormatNone contract).
Status resident_backward(ResidentDense* r, const float* grad_out, const float* input,
                         const float* pre, float* grad_in, float* grad_w, bool linear = false);
void resident_release(ResidentDense* r);

} // namespace welvet::cuda

