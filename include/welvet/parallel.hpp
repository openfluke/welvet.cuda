#pragma once

#include "welvet/dense.hpp"
#include "welvet/dtype.hpp"
#include "welvet/layer.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <optional>
#include <vector>

namespace welvet::parallel {

enum class Combine : int32_t {
  Concat = 0,
  Add = 1,
  Avg = 2,
  Filter = 3,
  Max = 4,
  SparseK = 5,
  Disagree = 6,
};

struct Config {
  int dim = 0;
  int out_feat = 0;
  int branches = 0;
  Combine combine = Combine::Concat;
  int seq_len = 0;
  int sparse_k = 0;
  double disagree_beta = 0;
};

int out_dim(const Config& cfg);
int weight_count(const Config& cfg);
int sparse_k_eff(const Config& cfg);
double disagree_beta_eff(const Config& cfg);
Status validate(Config* cfg);

using Branch = stack::Child;
using BranchKind = stack::Kind;

struct Layer {
  Config cfg{};
  std::vector<Branch> branches;
  std::optional<dense::Layer> gate;
  LayerMeta meta{};
};

Status new_configured(const Config& cfg, DType dt, const float* packed_or_null, Layer* out);
Status new_from_branches(const Config& cfg, std::vector<Branch> branches,
                         std::optional<dense::Layer> gate, Layer* out);

dense::Layer* dense_branch(Layer& l, int i);
const dense::Layer* dense_branch(const Layer& l, int i);

Status branch_forward(Branch& br, const TensorF32& input, const TensorF32* flat, TensorF32* pre,
                      TensorF32* post);
Status branch_backward(Branch& br, const TensorF32& grad_out, const TensorF32& input,
                       const TensorF32* flat, const TensorF32& pre, TensorF32* grad_in,
                       TensorF32* grad_w);
Status branch_apply_grad_sgd(Branch* br, const TensorF32& dW, double lr);
int branch_grad_w_size(const Branch& br);
Status branch_export_f32(const Branch& br, float* dst);
bool branch_needs_dense_flat(const Branch& br);

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post);
int grad_w_size(const Layer& layer);
Status split_combine_grad(const Layer& layer, const TensorF32& grad_out,
                          const std::vector<TensorF32>& branch_outs,
                          std::vector<TensorF32>* branch_grads);
Status approx_gate_grad(const Layer& layer, const TensorF32& grad_out, TensorF32* gate_gy);
Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w);
Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr);
Status export_packed_f32(const Layer& layer, float* dst);

} // namespace welvet::parallel
