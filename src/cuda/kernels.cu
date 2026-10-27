#include "welvet/cuda/kernels.hpp"
#include "welvet/cuda/matvec.hpp"
#include "welvet/cuda/mha.hpp"
#include "welvet/cuda/norm.hpp"
#include "welvet/cuda/softmax.hpp"
#include "welvet/cuda/swiglu.hpp"
#include "welvet/cuda/embedding.hpp"
#include "welvet/cuda/conv.hpp"
#include "welvet/activate.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

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

// One token per thread. center=0 RMSNorm; center=1 LayerNorm. Matches Go/cpp f64 path.
__global__ void norm_forward_f64(const float* __restrict__ x, const double* __restrict__ gamma,
                                 const double* __restrict__ beta, float* __restrict__ pre,
                                 float* __restrict__ post, int n_tok, int dim, double eps,
                                 int center) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_tok) return;
  const int base = t * dim;
  double sum = 0.0;
  double sum_sq = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double v = static_cast<double>(x[base + i]);
    sum += v;
    sum_sq += v * v;
  }
  double inv;
  double mean = 0.0;
  if (center) {
    mean = sum / static_cast<double>(dim);
    const double var_ = sum_sq / static_cast<double>(dim) - mean * mean;
    inv = 1.0 / sqrt(var_ + eps);
  } else {
    inv = 1.0 / sqrt(sum_sq / static_cast<double>(dim) + eps);
  }
  for (int i = 0; i < dim; ++i) {
    const double xv = static_cast<double>(x[base + i]);
    const double xh = center ? (xv - mean) * inv : xv * inv;
    pre[base + i] = static_cast<float>(xh);
    double y = xh * gamma[i];
    if (center && beta) y += beta[i];
    post[base + i] = static_cast<float>(y);
  }
}

// Go/cpp SwiGLU product: float(double(float(SiLU(double(g)))) * double(up)).
__global__ void swiglu_product_f64(const float* __restrict__ gate_pre,
                                   const float* __restrict__ up_post, float* __restrict__ h,
                                   int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double silu = activate_dev(1 /* Silu */, static_cast<double>(gate_pre[i]));
  const float g_f = static_cast<float>(silu);
  h[i] = static_cast<float>(static_cast<double>(g_f) * static_cast<double>(up_post[i]));
}

// Embedding gather: id = abs(int(tokens[t*in_last])) % vocab; out[t,j] = Activate(W[id*emb+j]).
__global__ void embedding_gather_f32(const float* __restrict__ tokens, const float* __restrict__ W,
                                     float* __restrict__ out, int n_tok, int emb, int vocab,
                                     int in_last, int act) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = n_tok * emb;
  if (tid >= n) return;
  const int tok = tid / emb;
  const int col = tid - tok * emb;
  int id = static_cast<int>(tokens[tok * in_last]);
  if (id < 0) id = -id;
  if (vocab > 0) id = id % vocab;
  const float v = W[static_cast<size_t>(id) * static_cast<size_t>(emb) + static_cast<size_t>(col)];
  if (act == -1) {
    out[tid] = v;
  } else {
    out[tid] = static_cast<float>(activate_dev(act, static_cast<double>(v)));
  }
}

// One row per thread. Matches Go/cpp float stable Softmax (f64 exp→f32) + Metal kinds.
__global__ void softmax_forward_row(const float* __restrict__ x, float* __restrict__ out,
                                    float* __restrict__ logits, float* __restrict__ sorted,
                                    const int* __restrict__ mask, int rows, int cols, int kind,
                                    int mask_len, unsigned gumbel_seed, double temperature,
                                    float entmax_alpha) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= rows) return;

  const int start = tid * cols;
  float* row_logits = logits + start;
  float* row_sorted = sorted + start;
  float* result = out + start;

  const double inv_t = 1.0 / (temperature == 0.0 ? 1.0 : temperature);
  unsigned state = gumbel_seed;
  if (!state) state = 0xC0FFEEu;
  if (kind == 2) {
    for (int i = 0; i < start; ++i) {
      state = state * 1664525u + 1013904223u;
    }
  }

  float mx = -INFINITY;
  for (int i = 0; i < cols; ++i) {
    float v = static_cast<float>(static_cast<double>(x[start + i]) * inv_t);
    if (kind == 3 && start + i < mask_len && mask && !mask[start + i]) {
      v = -1e9f;
    }
    if (kind == 2) {
      state = state * 1664525u + 1013904223u;
      float u = static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
      if (u < 1e-10f) u = 1e-10f;
      v += -static_cast<float>(log(-log(static_cast<double>(u))));
    }
    row_logits[i] = v;
    row_sorted[i] = v;
    if (v > mx) mx = v;
  }

  float se = 0.0f;
  for (int i = 0; i < cols; ++i) {
    // Go: exp in float64, then round each term to Float32.
    result[i] = static_cast<float>(exp(static_cast<double>(row_logits[i] - mx)));
    se += result[i];
  }
  for (int i = 0; i < cols; ++i) {
    result[i] /= se;
  }

  float alpha = entmax_alpha;
  if (alpha == 0.0f) alpha = 1.5f;
  if (kind == 4 || (kind == 5 && alpha > 1.0f)) {
    for (int i = 0; i < cols; ++i) {
      for (int k = i + 1; k < cols; ++k) {
        if (row_sorted[i] < row_sorted[k]) {
          const float tmp = row_sorted[i];
          row_sorted[i] = row_sorted[k];
          row_sorted[k] = tmp;
        }
      }
    }
    float cum = 0.0f;
    int count = 0;
    for (int i = 0; i < cols; ++i) {
      cum += row_sorted[i];
      if (row_sorted[i] - (cum - 1.0f) / static_cast<float>(i + 1) > 0.0f) {
        count = i + 1;
      } else {
        break;
      }
    }
    float tau = 0.0f;
    if (count) {
      cum = 0.0f;
      for (int i = 0; i < count; ++i) {
        cum += row_sorted[i];
      }
      tau = (cum - 1.0f) / static_cast<float>(count);
    }
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
      const float sp = fmaxf(0.0f, row_logits[i] - tau);
      if (kind == 4 || alpha >= 2.0f) {
        result[i] = sp;
      } else {
        result[i] = (2.0f - alpha) * result[i] + (alpha - 1.0f) * sp;
      }
      sum += result[i];
    }
    if (kind == 5 && alpha < 2.0f) {
      for (int i = 0; i < cols; ++i) {
        result[i] /= sum;
      }
    }
  }
}

welvet::Status cuda_status(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + cudaGetErrorString(err));
}

// --- MHA ModeSelf attention (native f64; matches welvet.cpp / Go) ---

__device__ bool mha_attn_allow(int q_pos, int k_pos, int mask, int window, int msl, int prefix_len,
                               int window_causal) {
  if (mask == 2) return true; // Bidirectional
  if (mask == 3) {            // SlidingWindow
    const int w = window > 0 ? window : msl;
    if (window_causal) return k_pos <= q_pos && k_pos >= q_pos - w + 1;
    int d = q_pos - k_pos;
    if (d < 0) d = -d;
    return d < w;
  }
  if (mask == 4) { // PrefixLM
    if (q_pos < prefix_len && k_pos < prefix_len) return true;
    return k_pos <= q_pos;
  }
  return k_pos <= q_pos; // Causal / default
}

__device__ void mha_apply_rope(double* vec, int pos, int num_heads, int head_dim, double theta) {
  const int half = head_dim / 2;
  if (half <= 0 || theta <= 0.0) return;
  for (int h = 0; h < num_heads; ++h) {
    const int base = h * head_dim;
    for (int d = 0; d < half; ++d) {
      const double angle =
          static_cast<double>(pos) /
          pow(theta, static_cast<double>(2 * d) / static_cast<double>(head_dim));
      const double c = cos(angle);
      const double s = sin(angle);
      const double v0 = vec[base + d];
      const double v1 = vec[base + d + half];
      vec[base + d] = v0 * c - v1 * s;
      vec[base + d + half] = v0 * s + v1 * c;
    }
  }
}

__device__ double mha_alibi_bias(int head, int num_heads, int q_pos, int k_pos, int use_alibi) {
  if (!use_alibi) return 0.0;
  int n = num_heads;
  if (n <= 0) n = 1;
  const double slope = pow(2.0, -static_cast<double>(head + 1) * 8.0 / static_cast<double>(n));
  const int diff = q_pos - k_pos;
  return -slope * static_cast<double>(diff < 0 ? -diff : diff);
}

// Single-thread kernel: serial f64 attention matching welvet.cpp attn_forward.
__global__ void mha_attention_forward_f64(
    const float* __restrict__ q_proj, const float* __restrict__ k_proj,
    const float* __restrict__ v_proj, float* __restrict__ attn_out, double* __restrict__ cache_k,
    double* __restrict__ cache_v, double* __restrict__ Q, double* __restrict__ scores,
    int* __restrict__ k_positions, int batch, int seq, int num_heads, int num_kv, int head_dim,
    int msl, int mask, int pos_kind, int softmax_kind, int window, int prefix_len,
    int window_causal, double rope_theta, double scale_override) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv * head_dim;
  const int heads_per_kv = num_heads / num_kv;
  double scale = scale_override;
  if (scale <= 0.0) scale = 1.0 / sqrt(static_cast<double>(head_dim));
  const bool use_rope = (pos_kind == 0 || pos_kind == 3);
  const bool use_alibi = (pos_kind == 2 || pos_kind == 3);
  const bool use_sigmoid = (softmax_kind == 1);

  for (int i = 0; i < msl * kv_dim; ++i) {
    cache_k[i] = 0.0;
    cache_v[i] = 0.0;
  }

  for (int b = 0; b < batch; ++b) {
    const int seq_base = b * seq;
    for (int s = 0; s < seq; ++s) {
      const int pos = seq_base + s;
      const int tok = b * seq + s;
      for (int i = 0; i < q_dim; ++i) {
        Q[s * q_dim + i] = static_cast<double>(q_proj[tok * q_dim + i]);
      }
      double* k_row = cache_k + (pos % msl) * kv_dim;
      double* v_row = cache_v + (pos % msl) * kv_dim;
      for (int i = 0; i < kv_dim; ++i) {
        k_row[i] = static_cast<double>(k_proj[tok * kv_dim + i]);
        v_row[i] = static_cast<double>(v_proj[tok * kv_dim + i]);
      }
      if (use_rope) {
        mha_apply_rope(Q + s * q_dim, pos, num_heads, head_dim, rope_theta);
        mha_apply_rope(k_row, pos, num_kv, head_dim, rope_theta);
      }
    }

    const int kv_len = seq_base + seq;
    for (int s = 0; s < seq; ++s) {
      const int q_pos = seq_base + s;
      for (int h = 0; h < num_heads; ++h) {
        const int kv_head = h / heads_per_kv;
        const int q_off = s * q_dim + h * head_dim;
        const int a_off = (b * seq + s) * q_dim + h * head_dim;
        int n = 0;
        double max_score = -1e9;
        for (int k_pos = 0; k_pos < kv_len; ++k_pos) {
          if (!mha_attn_allow(q_pos, k_pos, mask, window, msl, prefix_len, window_causal)) continue;
          const int k_idx = k_pos % msl;
          double dot = 0.0;
          const int k_base = k_idx * kv_dim + kv_head * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            dot += Q[q_off + d] * cache_k[k_base + d];
          }
          const double score =
              dot * scale + mha_alibi_bias(h, num_heads, q_pos, k_pos, use_alibi ? 1 : 0);
          scores[n] = score;
          k_positions[n] = k_pos;
          if (score > max_score) max_score = score;
          ++n;
        }
        if (n == 0) {
          for (int d = 0; d < head_dim; ++d) attn_out[a_off + d] = 0.0f;
          continue;
        }
        if (use_sigmoid) {
          for (int i = 0; i < n; ++i) {
            scores[i] = 1.0 / (1.0 + exp(-scores[i]));
          }
        } else {
          double exp_sum = 0.0;
          for (int i = 0; i < n; ++i) {
            scores[i] = exp(scores[i] - max_score);
            exp_sum += scores[i];
          }
          const double inv = 1.0 / exp_sum;
          for (int i = 0; i < n; ++i) scores[i] *= inv;
        }
        for (int d = 0; d < head_dim; ++d) {
          double sum = 0.0;
          for (int i = 0; i < n; ++i) {
            const int v_idx = k_positions[i] % msl;
            sum += scores[i] * cache_v[v_idx * kv_dim + kv_head * head_dim + d];
          }
          attn_out[a_off + d] = static_cast<float>(sum);
        }
      }
    }
  }
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

Status launch_norm_forward(const float* x_host, const double* gamma_f64, const double* beta_f64,
                           float* pre_host, float* post_host, int n_tok, int dim, double eps,
                           int center) {
  if (!x_host || !gamma_f64 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "norm: null ptr");
  }
  if (n_tok <= 0 || dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "norm: bad dims");
  }
  if (center && !beta_f64) {
    return Status::Err(StatusCode::InvalidArgument, "norm: LayerNorm needs beta");
  }

  const std::size_t x_bytes =
      static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(dim) * sizeof(float);
  const std::size_t g_bytes = static_cast<std::size_t>(dim) * sizeof(double);

  float *x_dev = nullptr, *pre_dev = nullptr, *post_dev = nullptr;
  double *g_dev = nullptr, *b_dev = nullptr;

  auto cleanup = [&]() {
    if (x_dev) cudaFree(x_dev);
    if (pre_dev) cudaFree(pre_dev);
    if (post_dev) cudaFree(post_dev);
    if (g_dev) cudaFree(g_dev);
    if (b_dev) cudaFree(b_dev);
  };

  auto st = cuda_status(cudaMalloc(&x_dev, x_bytes), "norm cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&pre_dev, x_bytes), "norm cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&post_dev, x_bytes), "norm cudaMalloc post");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&g_dev, g_bytes), "norm cudaMalloc gamma");
  if (!st) {
    cleanup();
    return st;
  }
  if (center) {
    st = cuda_status(cudaMalloc(&b_dev, g_bytes), "norm cudaMalloc beta");
    if (!st) {
      cleanup();
      return st;
    }
  }

  st = cuda_status(cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice), "norm upload x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(g_dev, gamma_f64, g_bytes, cudaMemcpyHostToDevice), "norm upload γ");
  if (!st) {
    cleanup();
    return st;
  }
  if (center) {
    st = cuda_status(cudaMemcpy(b_dev, beta_f64, g_bytes, cudaMemcpyHostToDevice), "norm upload β");
    if (!st) {
      cleanup();
      return st;
    }
  }

  constexpr int kThreads = 256;
  const int blocks = (n_tok + kThreads - 1) / kThreads;
  norm_forward_f64<<<blocks, kThreads>>>(x_dev, g_dev, b_dev, pre_dev, post_dev, n_tok, dim, eps,
                                         center);
  st = cuda_status(cudaGetLastError(), "norm launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "norm sync");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(pre_host, pre_dev, x_bytes, cudaMemcpyDeviceToHost), "norm download pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(post_host, post_dev, x_bytes, cudaMemcpyDeviceToHost),
                   "norm download post");
  cleanup();
  return st;
}

Status launch_softmax_forward(const float* x_host, float* out_host, int rows, int cols, int n,
                              int kind, double temperature, double entmax_alpha,
                              uint32_t gumbel_seed, const int32_t* mask_host, int mask_len) {
  if (!x_host || !out_host) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: null ptr");
  }
  if (rows <= 0 || cols <= 0 || n <= 0 || rows * cols > n) {
    return Status::Err(StatusCode::InvalidArgument, "softmax: bad dims");
  }

  const std::size_t x_bytes = static_cast<std::size_t>(n) * sizeof(float);
  const std::size_t work_bytes =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(float);

  float *x_dev = nullptr, *out_dev = nullptr, *logits_dev = nullptr, *sorted_dev = nullptr;
  int* mask_dev = nullptr;

  auto cleanup = [&]() {
    if (x_dev) cudaFree(x_dev);
    if (out_dev) cudaFree(out_dev);
    if (logits_dev) cudaFree(logits_dev);
    if (sorted_dev) cudaFree(sorted_dev);
    if (mask_dev) cudaFree(mask_dev);
  };

  auto st = cuda_status(cudaMalloc(&x_dev, x_bytes), "softmax cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&out_dev, x_bytes), "softmax cudaMalloc out");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&logits_dev, work_bytes), "softmax cudaMalloc logits");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&sorted_dev, work_bytes), "softmax cudaMalloc sorted");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice), "softmax upload x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemset(out_dev, 0, x_bytes), "softmax zero out");
  if (!st) {
    cleanup();
    return st;
  }

  if (mask_host && mask_len > 0) {
    const std::size_t m_bytes = static_cast<std::size_t>(mask_len) * sizeof(int);
    st = cuda_status(cudaMalloc(&mask_dev, m_bytes), "softmax cudaMalloc mask");
    if (!st) {
      cleanup();
      return st;
    }
    st = cuda_status(cudaMemcpy(mask_dev, mask_host, m_bytes, cudaMemcpyHostToDevice),
                     "softmax upload mask");
    if (!st) {
      cleanup();
      return st;
    }
  }

  constexpr int kThreads = 256;
  const int blocks = (rows + kThreads - 1) / kThreads;
  softmax_forward_row<<<blocks, kThreads>>>(
      x_dev, out_dev, logits_dev, sorted_dev, mask_dev, rows, cols, kind, mask_len, gumbel_seed,
      temperature, static_cast<float>(entmax_alpha));
  st = cuda_status(cudaGetLastError(), "softmax launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "softmax sync");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(out_host, out_dev, x_bytes, cudaMemcpyDeviceToHost),
                   "softmax download");
  cleanup();
  return st;
}

Status launch_swiglu_product(const float* gate_pre_host, const float* up_post_host, float* h_host,
                             int n) {
  if (!gate_pre_host || !up_post_host || !h_host) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: null ptr");
  }
  if (n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu: bad n");
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  float *g_dev = nullptr, *u_dev = nullptr, *h_dev = nullptr;
  auto cleanup = [&]() {
    if (g_dev) cudaFree(g_dev);
    if (u_dev) cudaFree(u_dev);
    if (h_dev) cudaFree(h_dev);
  };

  auto st = cuda_status(cudaMalloc(&g_dev, bytes), "swiglu cudaMalloc gate");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&u_dev, bytes), "swiglu cudaMalloc up");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&h_dev, bytes), "swiglu cudaMalloc h");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(g_dev, gate_pre_host, bytes, cudaMemcpyHostToDevice),
                   "swiglu upload gate");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(u_dev, up_post_host, bytes, cudaMemcpyHostToDevice),
                   "swiglu upload up");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  swiglu_product_f64<<<blocks, kThreads>>>(g_dev, u_dev, h_dev, n);
  st = cuda_status(cudaGetLastError(), "swiglu launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "swiglu sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(h_host, h_dev, bytes, cudaMemcpyDeviceToHost), "swiglu download");
  cleanup();
  return st;
}

Status launch_embedding_gather(const float* tokens_host, const float* W_f32, float* out_host,
                               int n_tok, int emb, int vocab, int in_last, Activation act) {
  if (!tokens_host || !W_f32 || !out_host) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: null ptr");
  }
  if (n_tok <= 0 || emb <= 0 || vocab <= 0 || in_last <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "embedding: bad dims");
  }

  const int out_n = n_tok * emb;
  const std::size_t tok_bytes =
      static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(in_last) * sizeof(float);
  const std::size_t w_bytes =
      static_cast<std::size_t>(vocab) * static_cast<std::size_t>(emb) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(out_n) * sizeof(float);

  float *tok_dev = nullptr, *w_dev = nullptr, *out_dev = nullptr;
  auto cleanup = [&]() {
    if (tok_dev) cudaFree(tok_dev);
    if (w_dev) cudaFree(w_dev);
    if (out_dev) cudaFree(out_dev);
  };

  auto st = cuda_status(cudaMalloc(&tok_dev, tok_bytes), "embedding cudaMalloc tokens");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&w_dev, w_bytes), "embedding cudaMalloc W");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&out_dev, out_bytes), "embedding cudaMalloc out");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(tok_dev, tokens_host, tok_bytes, cudaMemcpyHostToDevice),
                   "embedding upload tokens");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(w_dev, W_f32, w_bytes, cudaMemcpyHostToDevice), "embedding upload W");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (out_n + kThreads - 1) / kThreads;
  embedding_gather_f32<<<blocks, kThreads>>>(tok_dev, w_dev, out_dev, n_tok, emb, vocab, in_last,
                                             static_cast<int>(act));
  st = cuda_status(cudaGetLastError(), "embedding launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "embedding sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(out_host, out_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "embedding download");
  cleanup();
  return st;
}

Status launch_mha_attention_forward(const float* q_host, const float* k_host, const float* v_host,
                                    float* attn_host, int batch, int seq, int num_heads,
                                    int num_kv_heads, int head_dim, int max_seq_len, int mask,
                                    int pos, int softmax, int window, int prefix_len,
                                    int window_causal, double rope_theta, double scale_override) {
  if (!q_host || !k_host || !v_host || !attn_host) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn: null ptr");
  }
  if (batch <= 0 || seq <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      max_seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn: bad dims");
  }
  if (num_heads % num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn: heads%kv!=0");
  }

  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int bs = batch * seq;
  const int kv_len_max = batch * seq; // ModeSelf seq_base+seq peak

  const std::size_t q_bytes = static_cast<std::size_t>(bs) * static_cast<std::size_t>(q_dim) * sizeof(float);
  const std::size_t k_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(kv_dim) * sizeof(float);
  const std::size_t cache_bytes =
      static_cast<std::size_t>(max_seq_len) * static_cast<std::size_t>(kv_dim) * sizeof(double);
  const std::size_t Q_bytes =
      static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim) * sizeof(double);
  const std::size_t scores_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(double);
  const std::size_t kpos_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(int);

  float *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *attn_dev = nullptr;
  double *cache_k = nullptr, *cache_v = nullptr, *Q = nullptr, *scores = nullptr;
  int* k_positions = nullptr;

  auto cleanup = [&]() {
    if (q_dev) cudaFree(q_dev);
    if (k_dev) cudaFree(k_dev);
    if (v_dev) cudaFree(v_dev);
    if (attn_dev) cudaFree(attn_dev);
    if (cache_k) cudaFree(cache_k);
    if (cache_v) cudaFree(cache_v);
    if (Q) cudaFree(Q);
    if (scores) cudaFree(scores);
    if (k_positions) cudaFree(k_positions);
  };

  auto st = cuda_status(cudaMalloc(&q_dev, q_bytes), "mha cudaMalloc q");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&k_dev, k_bytes), "mha cudaMalloc k");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&v_dev, k_bytes), "mha cudaMalloc v");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&attn_dev, q_bytes), "mha cudaMalloc attn");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&cache_k, cache_bytes), "mha cudaMalloc cache_k");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&cache_v, cache_bytes), "mha cudaMalloc cache_v");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&Q, Q_bytes), "mha cudaMalloc Q");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&scores, scores_bytes), "mha cudaMalloc scores");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&k_positions, kpos_bytes), "mha cudaMalloc kpos");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice), "mha upload q");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(k_dev, k_host, k_bytes, cudaMemcpyHostToDevice), "mha upload k");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(v_dev, v_host, k_bytes, cudaMemcpyHostToDevice), "mha upload v");
  if (!st) {
    cleanup();
    return st;
  }

  mha_attention_forward_f64<<<1, 1>>>(
      q_dev, k_dev, v_dev, attn_dev, cache_k, cache_v, Q, scores, k_positions, batch, seq,
      num_heads, num_kv_heads, head_dim, max_seq_len, mask, pos, softmax, window, prefix_len,
      window_causal, rope_theta, scale_override);
  st = cuda_status(cudaGetLastError(), "mha attn launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "mha attn sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(attn_host, attn_dev, q_bytes, cudaMemcpyDeviceToHost),
                   "mha download attn");
  cleanup();
  return st;
}

// ConvT: one thread per loom output. Gather in Go scatter contribution order
// (ic → spatial_in → kernel) so f64 sum matches exact_bit vs Go/cpp.
__global__ void convt1_gather_fwd(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ pre, float* __restrict__ post, int batch,
                                  int in_c, int filters, int seq, int out_len, int k, int stride,
                                  int padding, int act) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_out = batch * filters * out_len;
  if (tid >= n_out) return;
  const int b = tid / (filters * out_len);
  const int rem = tid - b * filters * out_len;
  const int f = rem / out_len;
  const int ow = rem - f * out_len;
  double sum = 0.0;
  for (int ic = 0; ic < in_c; ++ic) {
    for (int iw = 0; iw < seq; ++iw) {
      const double xv =
          static_cast<double>(x[static_cast<size_t>(b * in_c * seq + ic * seq + iw)]);
      for (int kk = 0; kk < k; ++kk) {
        if (iw * stride - padding + kk != ow) continue;
        const int kw_idx = f * in_c * k + ic * k + kk;
        sum += xv * static_cast<double>(w[kw_idx]);
      }
    }
  }
  const float p = static_cast<float>(sum);
  pre[tid] = p;
  post[tid] = static_cast<float>(activate_dev(act, static_cast<double>(p)));
}

__global__ void convt2_gather_fwd(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ pre, float* __restrict__ post, int batch,
                                  int in_c, int filters, int in_h, int in_w, int out_h, int out_w,
                                  int k, int stride, int padding, int act) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sp_out = out_h * out_w;
  const int n_out = batch * filters * sp_out;
  if (tid >= n_out) return;
  const int sp_in = in_h * in_w;
  const int kk = k * k;
  const int b = tid / (filters * sp_out);
  const int rem = tid - b * filters * sp_out;
  const int f = rem / sp_out;
  const int o = rem - f * sp_out;
  const int oh = o / out_w;
  const int ow = o - oh * out_w;
  double sum = 0.0;
  for (int ic = 0; ic < in_c; ++ic) {
    for (int ih = 0; ih < in_h; ++ih) {
      for (int iw = 0; iw < in_w; ++iw) {
        const double xv = static_cast<double>(
            x[static_cast<size_t>(b * in_c * sp_in + ic * sp_in + ih * in_w + iw)]);
        for (int kh = 0; kh < k; ++kh) {
          for (int kw = 0; kw < k; ++kw) {
            if (ih * stride - padding + kh != oh || iw * stride - padding + kw != ow) continue;
            const int kw_idx = f * in_c * kk + ic * kk + kh * k + kw;
            sum += xv * static_cast<double>(w[kw_idx]);
          }
        }
      }
    }
  }
  const float p = static_cast<float>(sum);
  pre[tid] = p;
  post[tid] = static_cast<float>(activate_dev(act, static_cast<double>(p)));
}

__global__ void convt3_gather_fwd(const float* __restrict__ x, const float* __restrict__ w,
                                  float* __restrict__ pre, float* __restrict__ post, int batch,
                                  int in_c, int filters, int in_d, int in_h, int in_w, int out_d,
                                  int out_h, int out_w, int k, int stride, int padding, int act) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int sp_out = out_d * out_h * out_w;
  const int n_out = batch * filters * sp_out;
  if (tid >= n_out) return;
  const int sp_in = in_d * in_h * in_w;
  const int kkk = k * k * k;
  const int b = tid / (filters * sp_out);
  const int rem = tid - b * filters * sp_out;
  const int f = rem / sp_out;
  const int o = rem - f * sp_out;
  const int od = o / (out_h * out_w);
  const int o2 = o - od * out_h * out_w;
  const int oh = o2 / out_w;
  const int ow = o2 - oh * out_w;
  double sum = 0.0;
  for (int ic = 0; ic < in_c; ++ic) {
    for (int id = 0; id < in_d; ++id) {
      for (int ih = 0; ih < in_h; ++ih) {
        for (int iw = 0; iw < in_w; ++iw) {
          const double xv = static_cast<double>(
              x[static_cast<size_t>(b * in_c * sp_in + ic * sp_in + id * in_h * in_w + ih * in_w +
                                    iw)]);
          for (int kd = 0; kd < k; ++kd) {
            for (int kh = 0; kh < k; ++kh) {
              for (int kw = 0; kw < k; ++kw) {
                if (id * stride - padding + kd != od || ih * stride - padding + kh != oh ||
                    iw * stride - padding + kw != ow) {
                  continue;
                }
                const int kw_idx = f * in_c * kkk + ic * kkk + kd * k * k + kh * k + kw;
                sum += xv * static_cast<double>(w[kw_idx]);
              }
            }
          }
        }
      }
    }
  }
  const float p = static_cast<float>(sum);
  pre[tid] = p;
  post[tid] = static_cast<float>(activate_dev(act, static_cast<double>(p)));
}

Status launch_convt1_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int seq,
                             int out_len, int kernel, int stride, int padding, Activation act) {
  if (!x_host || !W_f32 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: null ptr");
  }
  if (batch <= 0 || in_c <= 0 || filters <= 0 || seq <= 0 || out_len <= 0 || kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt1: bad dims");
  }
  const int n_out = batch * filters * out_len;
  const int n_in = batch * in_c * seq;
  const int n_w = filters * in_c * kernel;
  const std::size_t x_bytes = static_cast<std::size_t>(n_in) * sizeof(float);
  const std::size_t w_bytes = static_cast<std::size_t>(n_w) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(n_out) * sizeof(float);

  float *x_dev = nullptr, *w_dev = nullptr, *pre_dev = nullptr, *post_dev = nullptr;
  auto cleanup = [&]() {
    if (x_dev) cudaFree(x_dev);
    if (w_dev) cudaFree(w_dev);
    if (pre_dev) cudaFree(pre_dev);
    if (post_dev) cudaFree(post_dev);
  };

  auto st = cuda_status(cudaMalloc(&x_dev, x_bytes), "convt1 cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&w_dev, w_bytes), "convt1 cudaMalloc w");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&pre_dev, out_bytes), "convt1 cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&post_dev, out_bytes), "convt1 cudaMalloc post");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice), "convt1 upload x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(w_dev, W_f32, w_bytes, cudaMemcpyHostToDevice), "convt1 upload w");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (n_out + kThreads - 1) / kThreads;
  convt1_gather_fwd<<<blocks, kThreads>>>(x_dev, w_dev, pre_dev, post_dev, batch, in_c, filters,
                                          seq, out_len, kernel, stride, padding,
                                          static_cast<int>(act));
  st = cuda_status(cudaGetLastError(), "convt1 gather launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "convt1 sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(pre_host, pre_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "convt1 download pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(post_host, post_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "convt1 download post");
  cleanup();
  return st;
}

Status launch_convt2_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int in_h, int in_w,
                             int out_h, int out_w, int kernel, int stride, int padding,
                             Activation act) {
  if (!x_host || !W_f32 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "convt2: null ptr");
  }
  if (batch <= 0 || in_c <= 0 || filters <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 ||
      out_w <= 0 || kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt2: bad dims");
  }
  const int sp_in = in_h * in_w;
  const int sp_out = out_h * out_w;
  const int n_out = batch * filters * sp_out;
  const int n_in = batch * in_c * sp_in;
  const int n_w = filters * in_c * kernel * kernel;
  const std::size_t x_bytes = static_cast<std::size_t>(n_in) * sizeof(float);
  const std::size_t w_bytes = static_cast<std::size_t>(n_w) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(n_out) * sizeof(float);

  float *x_dev = nullptr, *w_dev = nullptr, *pre_dev = nullptr, *post_dev = nullptr;
  auto cleanup = [&]() {
    if (x_dev) cudaFree(x_dev);
    if (w_dev) cudaFree(w_dev);
    if (pre_dev) cudaFree(pre_dev);
    if (post_dev) cudaFree(post_dev);
  };

  auto st = cuda_status(cudaMalloc(&x_dev, x_bytes), "convt2 cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&w_dev, w_bytes), "convt2 cudaMalloc w");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&pre_dev, out_bytes), "convt2 cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&post_dev, out_bytes), "convt2 cudaMalloc post");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice), "convt2 upload x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(w_dev, W_f32, w_bytes, cudaMemcpyHostToDevice), "convt2 upload w");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (n_out + kThreads - 1) / kThreads;
  convt2_gather_fwd<<<blocks, kThreads>>>(x_dev, w_dev, pre_dev, post_dev, batch, in_c, filters,
                                          in_h, in_w, out_h, out_w, kernel, stride, padding,
                                          static_cast<int>(act));
  st = cuda_status(cudaGetLastError(), "convt2 gather launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "convt2 sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(pre_host, pre_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "convt2 download pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(post_host, post_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "convt2 download post");
  cleanup();
  return st;
}

Status launch_convt3_forward(const float* x_host, const float* W_f32, float* pre_host,
                             float* post_host, int batch, int in_c, int filters, int in_d, int in_h,
                             int in_w, int out_d, int out_h, int out_w, int kernel, int stride,
                             int padding, Activation act) {
  if (!x_host || !W_f32 || !pre_host || !post_host) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: null ptr");
  }
  if (batch <= 0 || in_c <= 0 || filters <= 0 || in_d <= 0 || in_h <= 0 || in_w <= 0 || out_d <= 0 ||
      out_h <= 0 || out_w <= 0 || kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: bad dims");
  }
  const int sp_in = in_d * in_h * in_w;
  const int sp_out = out_d * out_h * out_w;
  const int n_out = batch * filters * sp_out;
  const int n_in = batch * in_c * sp_in;
  const int n_w = filters * in_c * kernel * kernel * kernel;
  const std::size_t x_bytes = static_cast<std::size_t>(n_in) * sizeof(float);
  const std::size_t w_bytes = static_cast<std::size_t>(n_w) * sizeof(float);
  const std::size_t out_bytes = static_cast<std::size_t>(n_out) * sizeof(float);

  float *x_dev = nullptr, *w_dev = nullptr, *pre_dev = nullptr, *post_dev = nullptr;
  auto cleanup = [&]() {
    if (x_dev) cudaFree(x_dev);
    if (w_dev) cudaFree(w_dev);
    if (pre_dev) cudaFree(pre_dev);
    if (post_dev) cudaFree(post_dev);
  };

  auto st = cuda_status(cudaMalloc(&x_dev, x_bytes), "convt3 cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&w_dev, w_bytes), "convt3 cudaMalloc w");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&pre_dev, out_bytes), "convt3 cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&post_dev, out_bytes), "convt3 cudaMalloc post");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice), "convt3 upload x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(w_dev, W_f32, w_bytes, cudaMemcpyHostToDevice), "convt3 upload w");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (n_out + kThreads - 1) / kThreads;
  convt3_gather_fwd<<<blocks, kThreads>>>(x_dev, w_dev, pre_dev, post_dev, batch, in_c, filters,
                                          in_d, in_h, in_w, out_d, out_h, out_w, kernel, stride,
                                          padding, static_cast<int>(act));
  st = cuda_status(cudaGetLastError(), "convt3 gather launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "convt3 sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(pre_host, pre_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "convt3 download pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(post_host, post_dev, out_bytes, cudaMemcpyDeviceToHost),
                   "convt3 download post");
  cleanup();
  return st;
}

} // namespace welvet::cuda
