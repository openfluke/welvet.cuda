#pragma once

#include "welvet/status.hpp"

#include <cstdint>

namespace welvet::cuda {

// Softmax forward on CUDA (one thread per row). Matches Go/cpp float stable path:
// temp scale in f64, exp in f64→f32, normalize in f32; sparsemax/entmax/gumbel/mask.
// x/out: host f32 [n]; mask may be null when mask_len==0 (int32 0/1).
Status launch_softmax_forward(const float* x_host, float* out_host, int rows, int cols, int n,
                              int kind, double temperature, double entmax_alpha,
                              uint32_t gumbel_seed, const int32_t* mask_host, int mask_len);

} // namespace welvet::cuda
