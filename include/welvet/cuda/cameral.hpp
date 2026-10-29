#pragma once

#include "welvet/status.hpp"

#include <cstdint>

namespace welvet::cuda {

// Cameral GPU ops — host packs/stages; device does the arithmetic.
Status launch_cam_shadow(float* gy, const float* student, const float* teacher,
                         std::int32_t n, double coef);
Status launch_cam_gate_gap(const float* gy, std::int32_t n, float* out,
                           std::int32_t out_n);
Status launch_cam_blend(const float* packed_host, const double* alphas_host,
                        float* blended_host, std::int32_t n, std::int32_t nb);
Status launch_cam_bias_blend(const double* biases_host, const double* alphas_host,
                             double* blended_host, std::int32_t n,
                             std::int32_t nb);

} // namespace welvet::cuda
