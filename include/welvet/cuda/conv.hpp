#pragma once

#include "welvet/activate.hpp"
#include "welvet/status.hpp"

namespace welvet::cuda {

// ConvT1/2/3 forward on CUDA. W_f32 is FormatNone-decoded [filters * patch].
// Accumulators use HW float64 in Go scatter loop order (exact_bit vs Go/cpp).
// pre/post are host loom buffers of length batch*filters*out_spatial.

Status launch_convt1_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int seq, int out_len,
                             int kernel, int stride, int padding, Activation act);

Status launch_convt2_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int in_h, int in_w,
                             int out_h, int out_w, int kernel, int stride, int padding,
                             Activation act);

Status launch_convt3_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int in_d, int in_h,
                             int in_w, int out_d, int out_h, int out_w, int kernel, int stride,
                             int padding, Activation act);

} // namespace welvet::cuda
