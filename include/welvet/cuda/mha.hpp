#pragma once

#include "welvet/status.hpp"

namespace welvet::cuda {

// ModeSelf attention forward on CUDA (native f64 ALU).
// q/k/v_host: dense projection outputs [batch*seq, q_dim|kv_dim] as f32.
// attn_host:  [batch*seq, q_dim] f32 out.
// Config ints mirror metal op_attention / welvet.cpp attn_forward.
Status launch_mha_attention_forward(const float* q_host, const float* k_host, const float* v_host,
                                    float* attn_host, int batch, int seq, int num_heads,
                                    int num_kv_heads, int head_dim, int max_seq_len, int mask,
                                    int pos, int softmax, int window, int prefix_len,
                                    int window_causal, double rope_theta, double scale_override);

// ModeSelf attention backward (native f64 + RoPEᵀ). Matches welvet.cpp attn_backward.
// Inputs: q/k/v_host projections + grad_attn [batch*seq, q_dim].
// Outputs: gQ [bs,q_dim], gK/gV [bs,kv_dim] as f32 (per-token layout).
Status launch_mha_attention_backward(const float* q_host, const float* k_host, const float* v_host,
                                     const float* grad_attn_host, float* gQ_host, float* gK_host,
                                     float* gV_host, int batch, int seq, int num_heads,
                                     int num_kv_heads, int head_dim, int max_seq_len, int mask,
                                     int pos, int softmax, int window, int prefix_len,
                                     int window_causal, double rope_theta, double scale_override);

} // namespace welvet::cuda
