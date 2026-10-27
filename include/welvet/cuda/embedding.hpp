#pragma once

#include "welvet/activate.hpp"
#include "welvet/status.hpp"

namespace welvet::cuda {

// Embedding gather on CUDA. W_f32 is FormatNone-decoded table [vocab*emb].
// tokens: host f32 length n_tok*in_last; out: host f32 [n_tok*emb].
// id = abs(int(tokens[t*in_last])) % vocab; out[t,j] = Activate(W[id*emb+j], act).
Status launch_embedding_gather(const float* tokens_host, const float* W_f32, float* out_host,
                               int n_tok, int emb, int vocab, int in_last, Activation act);

// Embedding NormalBP scatter dW: for each (row,col), sum grad_out[t,col] over tokens
// whose id == row (token order 0..n_tok-1). Matches Go/cpp/Metal exact_bit.
Status launch_embedding_scatter(const float* tokens_host, const float* grad_out_host,
                                float* dW_host, int n_tok, int emb, int vocab, int in_last);

} // namespace welvet::cuda
