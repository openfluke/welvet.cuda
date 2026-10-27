#include "welvet/cuda/kernels.hpp"

#include <cuda_runtime.h>

namespace {

__global__ void welvet_smoke_saxpy_kernel(const float* x, float* y, float a, float b,
                                          unsigned long long n) {
  const unsigned long long gid =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] = a * x[gid] + b;
}

__global__ void welvet_dense_matvec_f32_kernel(const float* W, const float* x, float* y,
                                              unsigned in_dim, unsigned out_dim) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= out_dim) return;
  float acc = 0.0f;
  const unsigned row = gid * in_dim;
  for (unsigned k = 0; k < in_dim; ++k) {
    acc += W[row + k] * x[k];
  }
  y[gid] = acc;
}

welvet::Status cuda_status(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + cudaGetErrorString(err));
}

} // namespace

namespace welvet::cuda {

Status launch_smoke_saxpy(const float* x_dev, float* y_dev, float a, float b, std::size_t n) {
  if (!x_dev || !y_dev) return Status::Err(StatusCode::InvalidArgument, "null device ptr");
  if (n == 0) return Status::Err(StatusCode::InvalidArgument, "empty saxpy");

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  welvet_smoke_saxpy_kernel<<<blocks, kThreads>>>(x_dev, y_dev, a, b,
                                                  static_cast<unsigned long long>(n));
  auto st = cuda_status(cudaGetLastError(), "welvet_smoke_saxpy launch");
  if (!st) return st;
  return cuda_status(cudaDeviceSynchronize(), "welvet_smoke_saxpy sync");
}

Status launch_dense_matvec_f32(const float* W_dev, const float* x_dev, float* y_dev,
                               std::uint32_t in_dim, std::uint32_t out_dim) {
  if (!W_dev || !x_dev || !y_dev) {
    return Status::Err(StatusCode::InvalidArgument, "null device ptr");
  }
  if (in_dim == 0 || out_dim == 0) {
    return Status::Err(StatusCode::InvalidArgument, "bad dims");
  }

  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((out_dim + kThreads - 1) / kThreads);
  welvet_dense_matvec_f32_kernel<<<blocks, kThreads>>>(W_dev, x_dev, y_dev, in_dim, out_dim);
  auto st = cuda_status(cudaGetLastError(), "welvet_dense_matvec_f32 launch");
  if (!st) return st;
  return cuda_status(cudaDeviceSynchronize(), "welvet_dense_matvec_f32 sync");
}

} // namespace welvet::cuda
