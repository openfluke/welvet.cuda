#pragma once

#include "welvet/activate.hpp"
#include "welvet/cuda/matvec.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"
#include "welvet/weights.hpp"

namespace welvet::dense {

// CUDA-only Dense. FormatNone ×34 decode → resident f64 MatVec + bias/act on GPU.
struct Layer {
  int input_height = 0;
  int output_height = 0;
  Activation activation = Activation::Linear;
  WeightStore weights;

  mutable cuda::ResidentDense gpu{};
  mutable bool gpu_valid = false;
  mutable int gpu_batch = -1;

  Layer() = default;
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;
  Layer(Layer&& o) noexcept
      : input_height(o.input_height), output_height(o.output_height), activation(o.activation),
        weights(std::move(o.weights)), gpu(o.gpu), gpu_valid(o.gpu_valid), gpu_batch(o.gpu_batch) {
    o.gpu = {};
    o.gpu_valid = false;
    o.gpu_batch = -1;
  }
  Layer& operator=(Layer&& o) noexcept {
    if (this == &o) return *this;
    cuda::resident_release(&gpu);
    input_height = o.input_height;
    output_height = o.output_height;
    activation = o.activation;
    weights = std::move(o.weights);
    gpu = o.gpu;
    gpu_valid = o.gpu_valid;
    gpu_batch = o.gpu_batch;
    o.gpu = {};
    o.gpu_valid = false;
    o.gpu_batch = -1;
    return *this;
  }
  ~Layer() { cuda::resident_release(&gpu); }

  void invalidate_gpu() const { gpu_valid = false; }
};

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer);

bool dtype_supported_cuda_forward(DType dt);

// FormatNone Dense forward on CUDA with resident W.
Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);

// Dense NormalBP backward: dX / dW via CUDA f64 MatVecᵀ + outer + act′.
Status backward_f32(const Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);

// FormatNone SGD: W ← W − lr·dW, then invalidate resident GPU mirror.
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);

Status sync_gpu_from_weights(const Layer& layer, int batch);

} // namespace welvet::dense
