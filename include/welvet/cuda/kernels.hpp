#pragma once

#include <cstddef>
#include <cstdint>

#include "welvet/status.hpp"

namespace welvet::cuda {

// M0 smoke: y[i] = a * x[i] + b  (device pointers).
Status launch_smoke_saxpy(const float* x_dev, float* y_dev, float a, float b, std::size_t n);

// Placeholder Dense GEMV (row-major W[out][in]) — not wired from host Dense yet.
Status launch_dense_matvec_f32(const float* W_dev, const float* x_dev, float* y_dev,
                               std::uint32_t in_dim, std::uint32_t out_dim);

} // namespace welvet::cuda
