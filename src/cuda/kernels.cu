#include "welvet/cuda/kernels.hpp"
#include "welvet/cuda/matvec.hpp"
#include "welvet/activate.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>

namespace {

__device__ double activate_dev(int act, double x) {
  switch (act) {
  case -1: // Linear
    return x;
  case 0: // ReLU
    return x < 0.0 ? 0.0 : x;
  case 6: // ReLU2
    return x < 0.0 ? 0.0 : x * x;
  case 5: // LeakyReLU
    return x < 0.0 ? 0.01 * x : x;
  case 1: { // Silu
    return x / (1.0 + exp(-x));
  }
  case 2: { // GELU approx (matches Go / welvet.cpp)
    constexpr double k = 0.7978845608028654;
    return 0.5 * x * (1.0 + tanh(k * (x + 0.044715 * x * x * x)));
  }
  case 3: // Tanh
    return tanh(x);
  case 4: // Sigmoid
    return 1.0 / (1.0 + exp(-x));
  default:
    return x;
  }
}

// y[b,o] = float(sum_k W[o,k] * double(x[b,k]))  — Go FormatNone MatVec contract.
__global__ void dense_matvec_f64_batch(const double* __restrict__ W, const float* __restrict__ x,
                                       float* __restrict__ y, int batch, int out, int in) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = batch * out;
  if (idx >= n) return;
  const int b = idx / out;
  const int o = idx - b * out;
  const double* wrow = W + static_cast<size_t>(o) * static_cast<size_t>(in);
  const float* xb = x + static_cast<size_t>(b) * static_cast<size_t>(in);
  double sum = 0.0;
  for (int k = 0; k < in; ++k) {
    sum += wrow[k] * static_cast<double>(xb[k]);
  }
  y[idx] = static_cast<float>(sum);
}

// pre += bias (f64), write back f32; post = Activate(FromFloat64(v)).
__global__ void dense_bias_act(float* __restrict__ pre, float* __restrict__ post,
                               const double* __restrict__ bias, int bias_len, int act, int batch,
                               int out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = batch * out;
  if (idx >= n) return;
  const int o = idx % out;
  double v = static_cast<double>(pre[idx]);
  if (bias && o < bias_len) {
    v += bias[o];
    pre[idx] = static_cast<float>(v);
  }
  const double a = activate_dev(act, v);
  post[idx] = static_cast<float>(a);
}

__global__ void welvet_smoke_saxpy_kernel(const float* x, float* y, float a, float b,
                                          unsigned long long n) {
  const unsigned long long gid =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] = a * x[gid] + b;
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

Status launch_dense_matvec_f32(const float*, const float*, float*, std::uint32_t, std::uint32_t) {
  return Status::Gap("use resident f64 Dense path (welvet::cuda::resident_*)");
}

Status resident_mount(ResidentDense* r, const double* W_f64, const double* bias_f64, int bias_len,
                      int batch, int out, int in, Activation act) {
  if (!r || !W_f64) return Status::Err(StatusCode::InvalidArgument, "null mount");
  if (batch <= 0 || out <= 0 || in <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "bad dims");
  }
  resident_release(r);
  r->batch = batch;
  r->out = out;
  r->in = in;
  r->bias_len = bias_len;
  r->act = act;

  const std::size_t w_bytes = static_cast<std::size_t>(out) * static_cast<std::size_t>(in) * sizeof(double);
  const std::size_t x_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(in) * sizeof(float);
  const std::size_t y_bytes = static_cast<std::size_t>(batch) * static_cast<std::size_t>(out) * sizeof(float);

  auto st = cuda_status(cudaMalloc(&r->W, w_bytes), "cudaMalloc W");
  if (!st) return st;
  st = cuda_status(cudaMemcpy(r->W, W_f64, w_bytes, cudaMemcpyHostToDevice), "upload W");
  if (!st) {
    resident_release(r);
    return st;
  }
  if (bias_f64 && bias_len > 0) {
    st = cuda_status(cudaMalloc(&r->bias, static_cast<std::size_t>(bias_len) * sizeof(double)),
                     "cudaMalloc bias");
    if (!st) {
      resident_release(r);
      return st;
    }
    st = cuda_status(cudaMemcpy(r->bias, bias_f64, static_cast<std::size_t>(bias_len) * sizeof(double),
                                cudaMemcpyHostToDevice),
                     "upload bias");
    if (!st) {
      resident_release(r);
      return st;
    }
  }
  st = cuda_status(cudaMalloc(&r->x, x_bytes), "cudaMalloc x");
  if (!st) {
    resident_release(r);
    return st;
  }
  st = cuda_status(cudaMalloc(&r->pre, y_bytes), "cudaMalloc pre");
  if (!st) {
    resident_release(r);
    return st;
  }
  st = cuda_status(cudaMalloc(&r->post, y_bytes), "cudaMalloc post");
  if (!st) {
    resident_release(r);
    return st;
  }
  return Status::Ok();
}

Status resident_execute(ResidentDense* r, const float* x_host, float* pre_host, float* post_host) {
  if (!r || !r->W || !x_host || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "null execute");
  }
  const std::size_t x_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->in) * sizeof(float);
  const std::size_t y_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->out) * sizeof(float);

  auto st = cuda_status(cudaMemcpy(r->x, x_host, x_bytes, cudaMemcpyHostToDevice), "upload x");
  if (!st) return st;

  const int n = r->batch * r->out;
  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;

  dense_matvec_f64_batch<<<blocks, kThreads>>>(static_cast<const double*>(r->W),
                                               static_cast<const float*>(r->x),
                                               static_cast<float*>(r->pre), r->batch, r->out, r->in);
  st = cuda_status(cudaGetLastError(), "matvec launch");
  if (!st) return st;

  const bool need_bias_act =
      r->act != Activation::Linear || (r->bias != nullptr && r->bias_len > 0);
  if (need_bias_act) {
    dense_bias_act<<<blocks, kThreads>>>(static_cast<float*>(r->pre), static_cast<float*>(r->post),
                                         static_cast<const double*>(r->bias), r->bias_len,
                                         static_cast<int>(r->act), r->batch, r->out);
    st = cuda_status(cudaGetLastError(), "bias_act launch");
    if (!st) return st;
  } else {
    st = cuda_status(cudaMemcpy(r->post, r->pre, y_bytes, cudaMemcpyDeviceToDevice), "post=pre");
    if (!st) return st;
  }

  st = cuda_status(cudaDeviceSynchronize(), "dense sync");
  if (!st) return st;

  st = cuda_status(cudaMemcpy(pre_host, r->pre, y_bytes, cudaMemcpyDeviceToHost), "download pre");
  if (!st) return st;
  return cuda_status(cudaMemcpy(post_host, r->post, y_bytes, cudaMemcpyDeviceToHost),
                     "download post");
}

void resident_release(ResidentDense* r) {
  if (!r) return;
  if (r->W) cudaFree(r->W);
  if (r->bias) cudaFree(r->bias);
  if (r->x) cudaFree(r->x);
  if (r->pre) cudaFree(r->pre);
  if (r->post) cudaFree(r->post);
  *r = {};
}

} // namespace welvet::cuda
