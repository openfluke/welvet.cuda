#include "welvet/cuda/kernels.hpp"
#include "welvet/cuda/matvec.hpp"
#include "welvet/cuda/mha.hpp"
#include "welvet/cuda/norm.hpp"
#include "welvet/cuda/softmax.hpp"
#include "welvet/cuda/swiglu.hpp"
#include "welvet/cuda/embedding.hpp"
#include "welvet/cuda/conv.hpp"
#include "welvet/cuda/tween.hpp"
#include "welvet/cuda/cameral.hpp"
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

__device__ double activate_deriv_dev(int act, double x) {
  switch (act) {
  case -1: // Linear
    return 1.0;
  case 0: // ReLU
    return x > 0.0 ? 1.0 : 0.0;
  case 6: // ReLU2
    return x > 0.0 ? 2.0 * x : 0.0;
  case 5: // LeakyReLU
    return x > 0.0 ? 1.0 : 0.01;
  case 1: { // Silu
    const double sig = 1.0 / (1.0 + exp(-x));
    return sig * (1.0 + x * (1.0 - sig));
  }
  case 2: { // GELU approx
    constexpr double k = 0.7978845608028654;
    const double u = k * (x + 0.044715 * x * x * x);
    const double th = tanh(u);
    const double du = k * (1.0 + 3.0 * 0.044715 * x * x);
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * du;
  }
  case 3: { // Tanh
    const double t = tanh(x);
    return 1.0 - t * t;
  }
  case 4: { // Sigmoid
    const double s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 - s);
  }
  default:
    return 1.0;
  }
}

// d_pre[i] = float(gy[i] * act′(pre[i]))
__global__ void dense_act_deriv(const float* __restrict__ gy, const float* __restrict__ pre,
                                float* __restrict__ d_pre, int act, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const double g = static_cast<double>(gy[idx]) *
                   activate_deriv_dev(act, static_cast<double>(pre[idx]));
  d_pre[idx] = static_cast<float>(g);
}

// gx[b,i] = float(sum_o W[o,i] * double(d_pre[b,o]))  — MatVecᵀ
__global__ void dense_matvect_f64_batch(const double* __restrict__ W, const float* __restrict__ d_pre,
                                        float* __restrict__ gx, int batch, int out, int in) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = batch * in;
  if (idx >= n) return;
  const int b = idx / in;
  const int i = idx - b * in;
  const float* gyb = d_pre + static_cast<size_t>(b) * static_cast<size_t>(out);
  double sum = 0.0;
  for (int o = 0; o < out; ++o) {
    sum += W[static_cast<size_t>(o) * static_cast<size_t>(in) + static_cast<size_t>(i)] *
           static_cast<double>(gyb[o]);
  }
  gx[idx] = static_cast<float>(sum);
}

// dW[o,i] — Go BackwardCPUTiled: per-batch add with float cast after each
// (FromFloat64(AsFloat64(dw) + g*x)), not a single f64 sum then cast.
__global__ void dense_outer_f64_batch(const float* __restrict__ d_pre, const float* __restrict__ x,
                                      float* __restrict__ dW, int batch, int out, int in) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = out * in;
  if (idx >= n) return;
  const int o = idx / in;
  const int i = idx - o * in;
  float acc = 0.0f;
  for (int b = 0; b < batch; ++b) {
    const double g = static_cast<double>(
        d_pre[static_cast<size_t>(b) * static_cast<size_t>(out) + static_cast<size_t>(o)]);
    const double xv = static_cast<double>(
        x[static_cast<size_t>(b) * static_cast<size_t>(in) + static_cast<size_t>(i)]);
    acc = static_cast<float>(static_cast<double>(acc) + g * xv);
  }
  dW[idx] = acc;
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

// Go/cpp RMSNorm / LayerNorm backward (native f64). One thread per token.
__global__ void norm_backward_f64(const float* __restrict__ x, const double* __restrict__ gamma,
                                  const float* __restrict__ pre, const float* __restrict__ gy,
                                  float* __restrict__ gx, int n_tok, int dim, double eps,
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
  if (center) {
    const double mean = sum / static_cast<double>(dim);
    const double var_ = sum_sq / static_cast<double>(dim) - mean * mean;
    inv = 1.0 / sqrt(var_ + eps);
  } else {
    inv = 1.0 / sqrt(sum_sq / static_cast<double>(dim) + eps);
  }

  // Keep γ*dy in registers / per-thread scratch — never round-trip through f32
  // (Go/cpp exact_bit requires full f64 intermediates).
  double sum_dxh = 0.0;
  double sum_dxh_xh = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double u = gamma[i] * static_cast<double>(gy[base + i]);
    const double xh = static_cast<double>(pre[base + i]);
    sum_dxh += u;
    sum_dxh_xh += u * xh;
  }
  const double n = static_cast<double>(dim);
  if (center) {
    for (int i = 0; i < dim; ++i) {
      const double xh = static_cast<double>(pre[base + i]);
      const double dxh = gamma[i] * static_cast<double>(gy[base + i]);
      gx[base + i] = static_cast<float>(inv / n * (n * dxh - sum_dxh - xh * sum_dxh_xh));
    }
  } else {
    const double mean = sum_dxh_xh / n;
    for (int i = 0; i < dim; ++i) {
      const double xh = static_cast<double>(pre[base + i]);
      const double u = gamma[i] * static_cast<double>(gy[base + i]);
      gx[base + i] = static_cast<float>(inv * (u - xh * mean));
    }
  }
}

// One thread per feature dim: dγ[i]=Σ_t dy*xh ; if center also dβ[i]=Σ_t dy.
__global__ void norm_param_grad_f64(const float* __restrict__ pre, const float* __restrict__ gy,
                                    float* __restrict__ dW, int n_tok, int dim, int center) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dim) return;
  double dg = 0.0;
  double db = 0.0;
  for (int t = 0; t < n_tok; ++t) {
    const int idx = t * dim + i;
    const double d = static_cast<double>(gy[idx]);
    dg += d * static_cast<double>(pre[idx]);
    db += d;
  }
  dW[i] = static_cast<float>(dg);
  if (center) dW[dim + i] = static_cast<float>(db);
}

// Softmax Jacobian: y⊙(g−⟨g,y⟩)/T. One thread per row. Matches Go/cpp float path.
__global__ void softmax_backward_row(const float* __restrict__ pre, const float* __restrict__ gy,
                                     float* __restrict__ gx, const int* __restrict__ mask, int rows,
                                     int cols, int kind, int mask_len, double temperature) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= rows) return;
  const int start = tid * cols;
  const double inv_t = 1.0 / (temperature == 0.0 ? 1.0 : temperature);
  float dot = 0.0f;
  for (int i = 0; i < cols; ++i) {
    float g = gy[start + i];
    if (kind == 3 && start + i < mask_len && mask && !mask[start + i]) g = 0.0f;
    dot += g * pre[start + i];
  }
  for (int j = 0; j < cols; ++j) {
    float g = gy[start + j];
    if (kind == 3 && start + j < mask_len && mask && !mask[start + j]) g = 0.0f;
    const float gl = pre[start + j] * (g - dot);
    gx[start + j] = static_cast<float>(static_cast<double>(gl) * inv_t);
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

// SwiGLU NormalBP SiLU′ product — matches Go Activate/ActivateDeriv → T then AsFloat64.
__global__ void swiglu_bwd_product_f64(const float* __restrict__ gate_pre,
                                       const float* __restrict__ up_post,
                                       const float* __restrict__ grad_h, float* __restrict__ d_up,
                                       float* __restrict__ d_gate, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double gp = static_cast<double>(gate_pre[i]);
  const float silu_f = static_cast<float>(activate_dev(1 /* Silu */, gp));
  const float dsilu_f = static_cast<float>(activate_deriv_dev(1 /* Silu */, gp));
  const double dh = static_cast<double>(grad_h[i]);
  const double up = static_cast<double>(up_post[i]);
  d_up[i] = static_cast<float>(dh * static_cast<double>(silu_f));
  d_gate[i] = static_cast<float>(dh * up * static_cast<double>(dsilu_f));
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

// Embedding scatter dW: one thread per (vocab_row, emb_col); sum over tokens in order.
__global__ void embedding_scatter_f32(const float* __restrict__ tokens,
                                      const float* __restrict__ grad_out, float* __restrict__ dW,
                                      int n_tok, int emb, int vocab, int in_last) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = vocab * emb;
  if (tid >= n) return;
  const int row = tid / emb;
  const int col = tid - row * emb;
  float sum = 0.0f;
  for (int t = 0; t < n_tok; ++t) {
    int id = static_cast<int>(tokens[t * in_last]);
    if (id < 0) id = -id;
    if (vocab > 0) id = id % vocab;
    if (id == row) {
      sum += grad_out[static_cast<size_t>(t) * static_cast<size_t>(emb) +
                      static_cast<size_t>(col)];
    }
  }
  dW[tid] = sum;
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

__device__ void mha_apply_rope_backward(double* grad, int pos, int num_heads, int head_dim,
                                        double theta) {
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
      const double v0 = grad[base + d];
      const double v1 = grad[base + d + half];
      grad[base + d] = v0 * c + v1 * s;
      grad[base + d + half] = -v0 * s + v1 * c;
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

// ModeSelf attention backward — matches welvet.cpp attn_backward + RoPEᵀ.
__global__ void mha_attention_backward_f64(
    const float* __restrict__ q_proj, const float* __restrict__ k_proj,
    const float* __restrict__ v_proj, const float* __restrict__ grad_attn,
    float* __restrict__ gQ_out, float* __restrict__ gK_out, float* __restrict__ gV_out,
    double* __restrict__ cache_k, double* __restrict__ cache_v, double* __restrict__ Q,
    double* __restrict__ gQ, double* __restrict__ gK, double* __restrict__ gV,
    double* __restrict__ scores, double* __restrict__ dW, double* __restrict__ d_score,
    int* __restrict__ k_positions, int batch, int seq, int num_heads, int num_kv, int head_dim,
    int msl, int mask, int pos_kind, int softmax_kind, int window, int prefix_len,
    int window_causal, double rope_theta, double scale_override) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv * head_dim;
  const int heads_per_kv = num_heads / num_kv;
  const int bs = batch * seq;
  double scale = scale_override;
  if (scale <= 0.0) scale = 1.0 / sqrt(static_cast<double>(head_dim));
  const bool use_rope = (pos_kind == 0 || pos_kind == 3);
  const bool use_alibi = (pos_kind == 2 || pos_kind == 3);
  const bool use_sigmoid = (softmax_kind == 1);

  for (int i = 0; i < msl * kv_dim; ++i) {
    gK[i] = 0.0;
    gV[i] = 0.0;
  }
  for (int i = 0; i < bs * q_dim; ++i) gQ_out[i] = 0.0f;
  for (int i = 0; i < bs * kv_dim; ++i) {
    gK_out[i] = 0.0f;
    gV_out[i] = 0.0f;
  }

  for (int b = 0; b < batch; ++b) {
    const int seq_base = b * seq;
    for (int i = 0; i < msl * kv_dim; ++i) {
      cache_k[i] = 0.0;
      cache_v[i] = 0.0;
    }
    for (int i = 0; i < seq * q_dim; ++i) gQ[i] = 0.0;

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
    for (int h = 0; h < num_heads; ++h) {
      const int kv_head = h / heads_per_kv;
      for (int q_pos_rel = 0; q_pos_rel < seq; ++q_pos_rel) {
        const int q_pos = seq_base + q_pos_rel;
        const int q_off = q_pos_rel * q_dim + h * head_dim;
        const int g_off_tok = (b * seq + q_pos_rel) * q_dim + h * head_dim;
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
        if (n == 0) continue;

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

        for (int i = 0; i < n; ++i) dW[i] = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          const double dy = static_cast<double>(grad_attn[g_off_tok + d]);
          for (int i = 0; i < n; ++i) {
            const int v_idx = k_positions[i] % msl;
            const double v = cache_v[v_idx * kv_dim + kv_head * head_dim + d];
            gV[v_idx * kv_dim + kv_head * head_dim + d] += scores[i] * dy;
            dW[i] += v * dy;
          }
        }

        if (use_sigmoid) {
          for (int i = 0; i < n; ++i) {
            const double sig = scores[i];
            d_score[i] = dW[i] * sig * (1.0 - sig) * scale;
          }
        } else {
          double sum = 0.0;
          for (int i = 0; i < n; ++i) sum += dW[i] * scores[i];
          for (int i = 0; i < n; ++i) {
            d_score[i] = scores[i] * (dW[i] - sum) * scale;
          }
        }

        for (int i = 0; i < n; ++i) {
          const int k_idx = k_positions[i] % msl;
          const int k_base = k_idx * kv_dim + kv_head * head_dim;
          const double ds = d_score[i];
          for (int d = 0; d < head_dim; ++d) {
            gQ[q_off + d] += ds * cache_k[k_base + d];
            gK[k_base + d] += ds * Q[q_off + d];
          }
        }
      }
    }

    for (int s = 0; s < seq; ++s) {
      const int pos = seq_base + s;
      if (use_rope) {
        mha_apply_rope_backward(gQ + s * q_dim, pos, num_heads, head_dim, rope_theta);
      }
      const int tok = b * seq + s;
      for (int i = 0; i < q_dim; ++i) {
        gQ_out[tok * q_dim + i] = static_cast<float>(gQ[s * q_dim + i]);
      }
    }
    if (use_rope) {
      for (int s = 0; s < seq; ++s) {
        const int pos = seq_base + s;
        const int k_idx = pos % msl;
        mha_apply_rope_backward(gK + k_idx * kv_dim, pos, num_kv, head_dim, rope_theta);
      }
    }
  }

  for (int tok = 0; tok < bs; ++tok) {
    const int k_idx = tok % msl;
    for (int i = 0; i < kv_dim; ++i) {
      gK_out[tok * kv_dim + i] = static_cast<float>(gK[k_idx * kv_dim + i]);
      gV_out[tok * kv_dim + i] = static_cast<float>(gV[k_idx * kv_dim + i]);
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

Status resident_backward(ResidentDense* r, const float* grad_out, const float* input,
                         const float* pre, float* grad_in, float* grad_w, bool linear) {
  if (!r || !r->W || !grad_out || !input || !pre || !grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "null resident backward");
  }
  const std::size_t y_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->out) * sizeof(float);
  const std::size_t x_bytes =
      static_cast<std::size_t>(r->batch) * static_cast<std::size_t>(r->in) * sizeof(float);
  const std::size_t dw_bytes =
      static_cast<std::size_t>(r->out) * static_cast<std::size_t>(r->in) * sizeof(float);

  float* d_gy = nullptr;
  float* d_pre = nullptr;
  float* d_dpre = nullptr;
  float* d_x = nullptr;
  float* d_gx = nullptr;
  float* d_dw = nullptr;
  auto cleanup = [&]() {
    if (d_gy) cudaFree(d_gy);
    if (d_pre) cudaFree(d_pre);
    if (d_dpre) cudaFree(d_dpre);
    if (d_x) cudaFree(d_x);
    if (d_gx) cudaFree(d_gx);
    if (d_dw) cudaFree(d_dw);
  };

  auto st = cuda_status(cudaMalloc(&d_gy, y_bytes), "bwd cudaMalloc gy");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&d_pre, y_bytes), "bwd cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&d_dpre, y_bytes), "bwd cudaMalloc dpre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&d_x, x_bytes), "bwd cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&d_gx, x_bytes), "bwd cudaMalloc gx");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&d_dw, dw_bytes), "bwd cudaMalloc dw");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(d_gy, grad_out, y_bytes, cudaMemcpyHostToDevice), "upload gy");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(d_pre, pre, y_bytes, cudaMemcpyHostToDevice), "upload pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(d_x, input, x_bytes, cudaMemcpyHostToDevice), "upload x");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int n_y = r->batch * r->out;
  const int n_x = r->batch * r->in;
  const int n_w = r->out * r->in;
  const int act_i = static_cast<int>(linear ? Activation::Linear : r->act);

  dense_act_deriv<<<(n_y + kThreads - 1) / kThreads, kThreads>>>(d_gy, d_pre, d_dpre, act_i, n_y);
  st = cuda_status(cudaGetLastError(), "act_deriv launch");
  if (!st) {
    cleanup();
    return st;
  }

  dense_matvect_f64_batch<<<(n_x + kThreads - 1) / kThreads, kThreads>>>(
      static_cast<const double*>(r->W), d_dpre, d_gx, r->batch, r->out, r->in);
  st = cuda_status(cudaGetLastError(), "matvect launch");
  if (!st) {
    cleanup();
    return st;
  }

  dense_outer_f64_batch<<<(n_w + kThreads - 1) / kThreads, kThreads>>>(d_dpre, d_x, d_dw, r->batch,
                                                                       r->out, r->in);
  st = cuda_status(cudaGetLastError(), "outer launch");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaDeviceSynchronize(), "bwd sync");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(grad_in, d_gx, x_bytes, cudaMemcpyDeviceToHost), "download gx");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(grad_w, d_dw, dw_bytes, cudaMemcpyDeviceToHost), "download dw");
  cleanup();
  return st;
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

Status launch_norm_backward(const float* x_host, const double* gamma_f64, const float* pre_host,
                            const float* grad_out_host, float* grad_in_host, float* dW_host,
                            int n_tok, int dim, double eps, int center) {
  if (!x_host || !gamma_f64 || !pre_host || !grad_out_host || !grad_in_host || !dW_host) {
    return Status::Err(StatusCode::InvalidArgument, "norm bwd: null ptr");
  }
  if (n_tok <= 0 || dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "norm bwd: bad dims");
  }

  const std::size_t x_bytes =
      static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(dim) * sizeof(float);
  const std::size_t g_bytes = static_cast<std::size_t>(dim) * sizeof(double);
  const int dW_n = center ? 2 * dim : dim;
  const std::size_t dw_bytes = static_cast<std::size_t>(dW_n) * sizeof(float);

  float *x_dev = nullptr, *pre_dev = nullptr, *gy_dev = nullptr, *gx_dev = nullptr, *dw_dev = nullptr;
  double* g_dev = nullptr;

  auto cleanup = [&]() {
    if (x_dev) cudaFree(x_dev);
    if (pre_dev) cudaFree(pre_dev);
    if (gy_dev) cudaFree(gy_dev);
    if (gx_dev) cudaFree(gx_dev);
    if (dw_dev) cudaFree(dw_dev);
    if (g_dev) cudaFree(g_dev);
  };

  auto st = cuda_status(cudaMalloc(&x_dev, x_bytes), "norm bwd cudaMalloc x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&pre_dev, x_bytes), "norm bwd cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gy_dev, x_bytes), "norm bwd cudaMalloc gy");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gx_dev, x_bytes), "norm bwd cudaMalloc gx");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&dw_dev, dw_bytes), "norm bwd cudaMalloc dW");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&g_dev, g_bytes), "norm bwd cudaMalloc gamma");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(x_dev, x_host, x_bytes, cudaMemcpyHostToDevice), "norm bwd upload x");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(pre_dev, pre_host, x_bytes, cudaMemcpyHostToDevice),
                   "norm bwd upload pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(gy_dev, grad_out_host, x_bytes, cudaMemcpyHostToDevice),
                   "norm bwd upload gy");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(g_dev, gamma_f64, g_bytes, cudaMemcpyHostToDevice),
                   "norm bwd upload γ");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemset(dw_dev, 0, dw_bytes), "norm bwd zero dW");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int tok_blocks = (n_tok + kThreads - 1) / kThreads;
  norm_backward_f64<<<tok_blocks, kThreads>>>(x_dev, g_dev, pre_dev, gy_dev, gx_dev, n_tok, dim, eps,
                                              center);
  st = cuda_status(cudaGetLastError(), "norm bwd launch");
  if (!st) {
    cleanup();
    return st;
  }

  const int dim_blocks = (dim + kThreads - 1) / kThreads;
  norm_param_grad_f64<<<dim_blocks, kThreads>>>(pre_dev, gy_dev, dw_dev, n_tok, dim, center);
  st = cuda_status(cudaGetLastError(), "norm param launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "norm bwd sync");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(grad_in_host, gx_dev, x_bytes, cudaMemcpyDeviceToHost),
                   "norm bwd download gx");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(dW_host, dw_dev, dw_bytes, cudaMemcpyDeviceToHost),
                   "norm bwd download dW");
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

Status launch_softmax_backward(const float* pre_host, const float* grad_out_host, float* grad_in_host,
                               int rows, int cols, int n, int kind, double temperature,
                               const int32_t* mask_host, int mask_len) {
  if (!pre_host || !grad_out_host || !grad_in_host) {
    return Status::Err(StatusCode::InvalidArgument, "softmax bwd: null ptr");
  }
  if (rows <= 0 || cols <= 0 || n <= 0 || rows * cols > n) {
    return Status::Err(StatusCode::InvalidArgument, "softmax bwd: bad dims");
  }

  const std::size_t x_bytes = static_cast<std::size_t>(n) * sizeof(float);
  float *pre_dev = nullptr, *gy_dev = nullptr, *gx_dev = nullptr;
  int* mask_dev = nullptr;

  auto cleanup = [&]() {
    if (pre_dev) cudaFree(pre_dev);
    if (gy_dev) cudaFree(gy_dev);
    if (gx_dev) cudaFree(gx_dev);
    if (mask_dev) cudaFree(mask_dev);
  };

  auto st = cuda_status(cudaMalloc(&pre_dev, x_bytes), "softmax bwd cudaMalloc pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gy_dev, x_bytes), "softmax bwd cudaMalloc gy");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gx_dev, x_bytes), "softmax bwd cudaMalloc gx");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(pre_dev, pre_host, x_bytes, cudaMemcpyHostToDevice),
                   "softmax bwd upload pre");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(gy_dev, grad_out_host, x_bytes, cudaMemcpyHostToDevice),
                   "softmax bwd upload gy");
  if (!st) {
    cleanup();
    return st;
  }

  if (mask_host && mask_len > 0) {
    const std::size_t m_bytes = static_cast<std::size_t>(mask_len) * sizeof(int);
    st = cuda_status(cudaMalloc(&mask_dev, m_bytes), "softmax bwd cudaMalloc mask");
    if (!st) {
      cleanup();
      return st;
    }
    st = cuda_status(cudaMemcpy(mask_dev, mask_host, m_bytes, cudaMemcpyHostToDevice),
                     "softmax bwd upload mask");
    if (!st) {
      cleanup();
      return st;
    }
  }

  constexpr int kThreads = 256;
  const int blocks = (rows + kThreads - 1) / kThreads;
  softmax_backward_row<<<blocks, kThreads>>>(pre_dev, gy_dev, gx_dev, mask_dev, rows, cols, kind,
                                             mask_len, temperature);
  st = cuda_status(cudaGetLastError(), "softmax bwd launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "softmax bwd sync");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(grad_in_host, gx_dev, x_bytes, cudaMemcpyDeviceToHost),
                   "softmax bwd download");
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

Status launch_swiglu_bwd_product(const float* gate_pre_host, const float* up_post_host,
                                 const float* grad_h_host, float* d_up_host, float* d_gate_host,
                                 int n) {
  if (!gate_pre_host || !up_post_host || !grad_h_host || !d_up_host || !d_gate_host) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu bwd: null ptr");
  }
  if (n <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "swiglu bwd: bad n");
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  float *g_dev = nullptr, *u_dev = nullptr, *h_dev = nullptr, *dup_dev = nullptr,
        *dg_dev = nullptr;
  auto cleanup = [&]() {
    if (g_dev) cudaFree(g_dev);
    if (u_dev) cudaFree(u_dev);
    if (h_dev) cudaFree(h_dev);
    if (dup_dev) cudaFree(dup_dev);
    if (dg_dev) cudaFree(dg_dev);
  };

  auto st = cuda_status(cudaMalloc(&g_dev, bytes), "swiglu bwd cudaMalloc gate");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&u_dev, bytes), "swiglu bwd cudaMalloc up");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&h_dev, bytes), "swiglu bwd cudaMalloc dh");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&dup_dev, bytes), "swiglu bwd cudaMalloc d_up");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&dg_dev, bytes), "swiglu bwd cudaMalloc d_gate");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(g_dev, gate_pre_host, bytes, cudaMemcpyHostToDevice),
                   "swiglu bwd upload gate");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(u_dev, up_post_host, bytes, cudaMemcpyHostToDevice),
                   "swiglu bwd upload up");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(h_dev, grad_h_host, bytes, cudaMemcpyHostToDevice),
                   "swiglu bwd upload dh");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  swiglu_bwd_product_f64<<<blocks, kThreads>>>(g_dev, u_dev, h_dev, dup_dev, dg_dev, n);
  st = cuda_status(cudaGetLastError(), "swiglu bwd launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "swiglu bwd sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(d_up_host, dup_dev, bytes, cudaMemcpyDeviceToHost),
                   "swiglu bwd download d_up");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(d_gate_host, dg_dev, bytes, cudaMemcpyDeviceToHost),
                   "swiglu bwd download d_gate");
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

Status launch_embedding_scatter(const float* tokens_host, const float* grad_out_host,
                                float* dW_host, int n_tok, int emb, int vocab, int in_last) {
  if (!tokens_host || !grad_out_host || !dW_host) {
    return Status::Err(StatusCode::InvalidArgument, "embedding scatter: null ptr");
  }
  if (n_tok <= 0 || emb <= 0 || vocab <= 0 || in_last <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "embedding scatter: bad dims");
  }

  const int nw = vocab * emb;
  const std::size_t tok_bytes =
      static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(in_last) * sizeof(float);
  const std::size_t go_bytes =
      static_cast<std::size_t>(n_tok) * static_cast<std::size_t>(emb) * sizeof(float);
  const std::size_t dw_bytes = static_cast<std::size_t>(nw) * sizeof(float);

  float *tok_dev = nullptr, *go_dev = nullptr, *dw_dev = nullptr;
  auto cleanup = [&]() {
    if (tok_dev) cudaFree(tok_dev);
    if (go_dev) cudaFree(go_dev);
    if (dw_dev) cudaFree(dw_dev);
  };

  auto st = cuda_status(cudaMalloc(&tok_dev, tok_bytes), "embedding scatter cudaMalloc tokens");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&go_dev, go_bytes), "embedding scatter cudaMalloc grad");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&dw_dev, dw_bytes), "embedding scatter cudaMalloc dW");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(tok_dev, tokens_host, tok_bytes, cudaMemcpyHostToDevice),
                   "embedding scatter upload tokens");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(go_dev, grad_out_host, go_bytes, cudaMemcpyHostToDevice),
                   "embedding scatter upload grad");
  if (!st) {
    cleanup();
    return st;
  }

  constexpr int kThreads = 256;
  const int blocks = (nw + kThreads - 1) / kThreads;
  embedding_scatter_f32<<<blocks, kThreads>>>(tok_dev, go_dev, dw_dev, n_tok, emb, vocab, in_last);
  st = cuda_status(cudaGetLastError(), "embedding scatter launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "embedding scatter sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(dW_host, dw_dev, dw_bytes, cudaMemcpyDeviceToHost),
                   "embedding scatter download");
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

Status launch_mha_attention_backward(const float* q_host, const float* k_host, const float* v_host,
                                     const float* grad_attn_host, float* gQ_host, float* gK_host,
                                     float* gV_host, int batch, int seq, int num_heads,
                                     int num_kv_heads, int head_dim, int max_seq_len, int mask,
                                     int pos, int softmax, int window, int prefix_len,
                                     int window_causal, double rope_theta, double scale_override) {
  if (!q_host || !k_host || !v_host || !grad_attn_host || !gQ_host || !gK_host || !gV_host) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn bwd: null ptr");
  }
  if (batch <= 0 || seq <= 0 || num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      max_seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn bwd: bad dims");
  }
  if (num_heads % num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha attn bwd: heads%kv!=0");
  }

  const int q_dim = num_heads * head_dim;
  const int kv_dim = num_kv_heads * head_dim;
  const int bs = batch * seq;
  const int kv_len_max = batch * seq;

  const std::size_t q_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(q_dim) * sizeof(float);
  const std::size_t k_bytes =
      static_cast<std::size_t>(bs) * static_cast<std::size_t>(kv_dim) * sizeof(float);
  const std::size_t cache_bytes =
      static_cast<std::size_t>(max_seq_len) * static_cast<std::size_t>(kv_dim) * sizeof(double);
  const std::size_t Q_bytes =
      static_cast<std::size_t>(seq) * static_cast<std::size_t>(q_dim) * sizeof(double);
  const std::size_t gQ_work_bytes = Q_bytes;
  const std::size_t scores_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(double);
  const std::size_t kpos_bytes = static_cast<std::size_t>(kv_len_max) * sizeof(int);

  float *q_dev = nullptr, *k_dev = nullptr, *v_dev = nullptr, *gy_dev = nullptr;
  float *gq_dev = nullptr, *gk_dev = nullptr, *gv_dev = nullptr;
  double *cache_k = nullptr, *cache_v = nullptr, *Q = nullptr, *gQ = nullptr, *gK = nullptr,
         *gV = nullptr, *scores = nullptr, *dW = nullptr, *d_score = nullptr;
  int* k_positions = nullptr;

  auto cleanup = [&]() {
    if (q_dev) cudaFree(q_dev);
    if (k_dev) cudaFree(k_dev);
    if (v_dev) cudaFree(v_dev);
    if (gy_dev) cudaFree(gy_dev);
    if (gq_dev) cudaFree(gq_dev);
    if (gk_dev) cudaFree(gk_dev);
    if (gv_dev) cudaFree(gv_dev);
    if (cache_k) cudaFree(cache_k);
    if (cache_v) cudaFree(cache_v);
    if (Q) cudaFree(Q);
    if (gQ) cudaFree(gQ);
    if (gK) cudaFree(gK);
    if (gV) cudaFree(gV);
    if (scores) cudaFree(scores);
    if (dW) cudaFree(dW);
    if (d_score) cudaFree(d_score);
    if (k_positions) cudaFree(k_positions);
  };

  auto st = cuda_status(cudaMalloc(&q_dev, q_bytes), "mha bwd cudaMalloc q");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&k_dev, k_bytes), "mha bwd cudaMalloc k");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&v_dev, k_bytes), "mha bwd cudaMalloc v");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gy_dev, q_bytes), "mha bwd cudaMalloc gy");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gq_dev, q_bytes), "mha bwd cudaMalloc gq");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gk_dev, k_bytes), "mha bwd cudaMalloc gk");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gv_dev, k_bytes), "mha bwd cudaMalloc gv");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&cache_k, cache_bytes), "mha bwd cudaMalloc cache_k");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&cache_v, cache_bytes), "mha bwd cudaMalloc cache_v");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&Q, Q_bytes), "mha bwd cudaMalloc Q");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gQ, gQ_work_bytes), "mha bwd cudaMalloc gQ work");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gK, cache_bytes), "mha bwd cudaMalloc gK");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&gV, cache_bytes), "mha bwd cudaMalloc gV");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&scores, scores_bytes), "mha bwd cudaMalloc scores");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&dW, scores_bytes), "mha bwd cudaMalloc dW");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&d_score, scores_bytes), "mha bwd cudaMalloc d_score");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMalloc(&k_positions, kpos_bytes), "mha bwd cudaMalloc kpos");
  if (!st) {
    cleanup();
    return st;
  }

  st = cuda_status(cudaMemcpy(q_dev, q_host, q_bytes, cudaMemcpyHostToDevice), "mha bwd upload q");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(k_dev, k_host, k_bytes, cudaMemcpyHostToDevice), "mha bwd upload k");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(v_dev, v_host, k_bytes, cudaMemcpyHostToDevice), "mha bwd upload v");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(gy_dev, grad_attn_host, q_bytes, cudaMemcpyHostToDevice),
                   "mha bwd upload gy");
  if (!st) {
    cleanup();
    return st;
  }

  mha_attention_backward_f64<<<1, 1>>>(
      q_dev, k_dev, v_dev, gy_dev, gq_dev, gk_dev, gv_dev, cache_k, cache_v, Q, gQ, gK, gV, scores,
      dW, d_score, k_positions, batch, seq, num_heads, num_kv_heads, head_dim, max_seq_len, mask,
      pos, softmax, window, prefix_len, window_causal, rope_theta, scale_override);
  st = cuda_status(cudaGetLastError(), "mha attn bwd launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "mha attn bwd sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(gQ_host, gq_dev, q_bytes, cudaMemcpyDeviceToHost), "mha bwd download gQ");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(gK_host, gk_dev, k_bytes, cudaMemcpyDeviceToHost), "mha bwd download gK");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(gV_host, gv_dev, k_bytes, cudaMemcpyDeviceToHost), "mha bwd download gV");
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

// Go parallel.projectGap — block average with f64 sums (matches Go float64 path).
__global__ void project_gap_f64(const float* __restrict__ src, float* __restrict__ dst,
                                unsigned long long sn, unsigned long long dn) {
  const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= dn) return;
  unsigned long long lo = i * sn / dn;
  unsigned long long hi = (i + 1) * sn / dn;
  if (hi <= lo) hi = lo + 1;
  if (hi > sn) hi = sn;
  double sum = 0.0;
  for (unsigned long long j = lo; j < hi; ++j) {
    sum += static_cast<double>(src[j]);
  }
  dst[i] = static_cast<float>(sum / static_cast<double>(hi - lo));
}

Status launch_project_gap(const float* src_host, float* dst_host, std::int32_t sn, std::int32_t dn) {
  if (!src_host || !dst_host) {
    return Status::Err(StatusCode::InvalidArgument, "project_gap: null host ptr");
  }
  if (sn <= 0 || dn <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "project_gap: empty");
  }
  if (sn == dn) {
    std::memcpy(dst_host, src_host, static_cast<std::size_t>(dn) * sizeof(float));
    return Status::Ok();
  }
  float *src_dev = nullptr, *dst_dev = nullptr;
  auto cleanup = [&]() {
    if (src_dev) cudaFree(src_dev);
    if (dst_dev) cudaFree(dst_dev);
  };
  auto st = cuda_status(cudaMalloc(&src_dev, static_cast<std::size_t>(sn) * sizeof(float)),
                        "project_gap malloc src");
  if (!st) return st;
  st = cuda_status(cudaMalloc(&dst_dev, static_cast<std::size_t>(dn) * sizeof(float)),
                   "project_gap malloc dst");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(src_dev, src_host, static_cast<std::size_t>(sn) * sizeof(float),
                              cudaMemcpyHostToDevice),
                   "project_gap upload");
  if (!st) {
    cleanup();
    return st;
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((static_cast<std::size_t>(dn) + kThreads - 1) / kThreads);
  project_gap_f64<<<blocks, kThreads>>>(src_dev, dst_dev, static_cast<unsigned long long>(sn),
                                        static_cast<unsigned long long>(dn));
  st = cuda_status(cudaGetLastError(), "project_gap launch");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaDeviceSynchronize(), "project_gap sync");
  if (!st) {
    cleanup();
    return st;
  }
  st = cuda_status(cudaMemcpy(dst_host, dst_dev, static_cast<std::size_t>(dn) * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "project_gap download");
  cleanup();
  return st;
}

// --- Cameral (Freeze/Shadow/CamSync) ---
__global__ void cam_shadow_f64(float* gy, const float* student, const float* teacher,
                               unsigned long long n, double coef) {
  unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double term = coef * 2.0 * (static_cast<double>(student[i]) - static_cast<double>(teacher[i])) /
                static_cast<double>(n);
  gy[i] = static_cast<float>(static_cast<double>(gy[i]) + term);
}
__global__ void cam_gate_gap_f64(const float* gy, float* out, unsigned long long n,
                                 unsigned long long out_n) {
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= out_n) return;
  double sum = 0;
  for (unsigned long long i = 0; i < n; ++i) sum += static_cast<double>(gy[i]);
  out[tid] = static_cast<float>((sum / static_cast<double>(n)) * 0.1);
}
__global__ void cam_blend_f64(const float* pack, const double* alphas, float* out,
                              unsigned long long n, unsigned long long nb) {
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  double mean = 0;
  for (unsigned long long k = 0; k < nb; ++k)
    mean += static_cast<double>(pack[k * n + tid]) / static_cast<double>(nb);
  for (unsigned long long k = 0; k < nb; ++k) {
    double a = alphas[k];
    float old = pack[k * n + tid];
    if (a <= 0) {
      out[k * n + tid] = old;
      continue;
    }
    if (a > 1) a = 1;
    out[k * n + tid] =
        static_cast<float>((1.0 - a) * static_cast<double>(old) + a * mean);
  }
}
__global__ void cam_bias_blend_f64(const double* bias, const double* alphas, double* out,
                                   unsigned long long n, unsigned long long nb) {
  unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  double mean = 0;
  for (unsigned long long k = 0; k < nb; ++k) mean += bias[k * n + tid] / static_cast<double>(nb);
  for (unsigned long long k = 0; k < nb; ++k) {
    double a = alphas[k];
    double old = bias[k * n + tid];
    if (a <= 0) {
      out[k * n + tid] = old;
      continue;
    }
    if (a > 1) a = 1;
    out[k * n + tid] = (1.0 - a) * old + a * mean;
  }
}

Status launch_cam_shadow(float* gy, const float* student, const float* teacher, std::int32_t n,
                         double coef) {
  if (!gy || !student || !teacher || n <= 0) return Status::Err(StatusCode::InvalidArgument, "cam_shadow args");
  float *d_gy = nullptr, *d_st = nullptr, *d_te = nullptr;
  size_t bytes = size_t(n) * sizeof(float);
  auto cleanup = [&] {
    if (d_gy) cudaFree(d_gy);
    if (d_st) cudaFree(d_st);
    if (d_te) cudaFree(d_te);
  };
  auto ok = cuda_status(cudaMalloc(&d_gy, bytes), "cam_shadow gy");
  if (!ok) return ok;
  ok = cuda_status(cudaMalloc(&d_st, bytes), "cam_shadow st");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMalloc(&d_te, bytes), "cam_shadow te");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(d_gy, gy, bytes, cudaMemcpyHostToDevice), "cam_shadow up gy");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(d_st, student, bytes, cudaMemcpyHostToDevice), "cam_shadow up st");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(d_te, teacher, bytes, cudaMemcpyHostToDevice), "cam_shadow up te");
  if (!ok) { cleanup(); return ok; }
  constexpr int kT = 256;
  int blocks = int((size_t(n) + kT - 1) / kT);
  cam_shadow_f64<<<blocks, kT>>>(d_gy, d_st, d_te, (unsigned long long)n, coef);
  ok = cuda_status(cudaGetLastError(), "cam_shadow launch");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaDeviceSynchronize(), "cam_shadow sync");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(gy, d_gy, bytes, cudaMemcpyDeviceToHost), "cam_shadow down");
  cleanup();
  return ok;
}

Status launch_cam_gate_gap(const float* gy, std::int32_t n, float* out, std::int32_t out_n) {
  if (!gy || !out || n <= 0 || out_n <= 0)
    return Status::Err(StatusCode::InvalidArgument, "cam_gate_gap args");
  float *d_gy = nullptr, *d_out = nullptr;
  auto cleanup = [&] {
    if (d_gy) cudaFree(d_gy);
    if (d_out) cudaFree(d_out);
  };
  auto ok = cuda_status(cudaMalloc(&d_gy, size_t(n) * sizeof(float)), "cam_gate gy");
  if (!ok) return ok;
  ok = cuda_status(cudaMalloc(&d_out, size_t(out_n) * sizeof(float)), "cam_gate out");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(d_gy, gy, size_t(n) * sizeof(float), cudaMemcpyHostToDevice), "cam_gate up");
  if (!ok) { cleanup(); return ok; }
  constexpr int kT = 256;
  int blocks = int((size_t(out_n) + kT - 1) / kT);
  cam_gate_gap_f64<<<blocks, kT>>>(d_gy, d_out, (unsigned long long)n, (unsigned long long)out_n);
  ok = cuda_status(cudaGetLastError(), "cam_gate launch");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaDeviceSynchronize(), "cam_gate sync");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(out, d_out, size_t(out_n) * sizeof(float), cudaMemcpyDeviceToHost),
                   "cam_gate down");
  cleanup();
  return ok;
}

Status launch_cam_blend(const float* packed_host, const double* alphas_host, float* blended_host,
                        std::int32_t n, std::int32_t nb) {
  if (!packed_host || !alphas_host || !blended_host || n <= 0 || nb < 2)
    return Status::Err(StatusCode::InvalidArgument, "cam_blend args");
  float *pack = nullptr, *out = nullptr;
  double* alphas = nullptr;
  size_t pbytes = size_t(n) * size_t(nb) * sizeof(float);
  size_t abytes = size_t(nb) * sizeof(double);
  auto cleanup = [&] {
    if (pack) cudaFree(pack);
    if (out) cudaFree(out);
    if (alphas) cudaFree(alphas);
  };
  auto ok = cuda_status(cudaMalloc(&pack, pbytes), "cam_blend pack");
  if (!ok) return ok;
  ok = cuda_status(cudaMalloc(&out, pbytes), "cam_blend out");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMalloc(&alphas, abytes), "cam_blend alpha");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(pack, packed_host, pbytes, cudaMemcpyHostToDevice), "cam_blend up p");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(alphas, alphas_host, abytes, cudaMemcpyHostToDevice), "cam_blend up a");
  if (!ok) { cleanup(); return ok; }
  constexpr int kT = 256;
  int blocks = int((size_t(n) + kT - 1) / kT);
  cam_blend_f64<<<blocks, kT>>>(pack, alphas, out, (unsigned long long)n, (unsigned long long)nb);
  ok = cuda_status(cudaGetLastError(), "cam_blend launch");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaDeviceSynchronize(), "cam_blend sync");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(blended_host, out, pbytes, cudaMemcpyDeviceToHost), "cam_blend down");
  cleanup();
  return ok;
}

Status launch_cam_bias_blend(const double* biases_host, const double* alphas_host,
                             double* blended_host, std::int32_t n, std::int32_t nb) {
  if (!biases_host || !alphas_host || !blended_host || n <= 0 || nb < 2)
    return Status::Err(StatusCode::InvalidArgument, "cam_bias_blend args");
  double *bias = nullptr, *out = nullptr, *alphas = nullptr;
  size_t bbytes = size_t(n) * size_t(nb) * sizeof(double);
  size_t abytes = size_t(nb) * sizeof(double);
  auto cleanup = [&] {
    if (bias) cudaFree(bias);
    if (out) cudaFree(out);
    if (alphas) cudaFree(alphas);
  };
  auto ok = cuda_status(cudaMalloc(&bias, bbytes), "cam_bias bias");
  if (!ok) return ok;
  ok = cuda_status(cudaMalloc(&out, bbytes), "cam_bias out");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMalloc(&alphas, abytes), "cam_bias alpha");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(bias, biases_host, bbytes, cudaMemcpyHostToDevice), "cam_bias up b");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(alphas, alphas_host, abytes, cudaMemcpyHostToDevice), "cam_bias up a");
  if (!ok) { cleanup(); return ok; }
  constexpr int kT = 256;
  int blocks = int((size_t(n) + kT - 1) / kT);
  cam_bias_blend_f64<<<blocks, kT>>>(bias, alphas, out, (unsigned long long)n,
                                     (unsigned long long)nb);
  ok = cuda_status(cudaGetLastError(), "cam_bias launch");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaDeviceSynchronize(), "cam_bias sync");
  if (!ok) { cleanup(); return ok; }
  ok = cuda_status(cudaMemcpy(blended_host, out, bbytes, cudaMemcpyDeviceToHost), "cam_bias down");
  cleanup();
  return ok;
}

} // namespace welvet::cuda
