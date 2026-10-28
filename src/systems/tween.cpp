#include "welvet/systems/tween.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/training.hpp"

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/mha.hpp"
#include "welvet/residual.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/stack.hpp"
#include "welvet/swiglu.hpp"

#include "welvet/cuda/tween.hpp"
#include "welvet/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace welvet::systems::tween {
namespace {

Status apply_one(architecture::Cell* cell, const TensorF32& dW, double lr) {
  if (!cell || dW.data.empty() || lr == 0.0) {
    return Status::Ok();
  }
  switch (cell->op_kind) {
  case architecture::OpKind::Dense:
    return dense::apply_grad_sgd(&cell->dense_op, dW, lr);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&cell->rmsnorm_op, dW, lr);
  case architecture::OpKind::LayerNorm:
    return layernorm::apply_grad_sgd(&cell->layernorm_op, dW, lr);
  case architecture::OpKind::SwiGLU:
    return swiglu::apply_grad_sgd(&cell->swiglu_op, dW, lr);
  case architecture::OpKind::Softmax:
    return softmax::apply_grad_sgd(&cell->softmax_op, dW, lr);
  case architecture::OpKind::Embedding:
    return embedding::apply_grad_sgd(&cell->embedding_op, dW, lr);
  case architecture::OpKind::MHA:
    return mha::apply_grad_sgd(&cell->mha_op, dW, lr);
  case architecture::OpKind::CNN1:
    return cnn1::apply_grad_sgd(&cell->cnn1_op, dW, lr);
  case architecture::OpKind::CNN2:
    return cnn2::apply_grad_sgd(&cell->cnn2_op, dW, lr);
  case architecture::OpKind::CNN3:
    return cnn3::apply_grad_sgd(&cell->cnn3_op, dW, lr);
  case architecture::OpKind::ConvT1:
    return convt1::apply_grad_sgd(&cell->convt1_op, dW, lr);
  case architecture::OpKind::ConvT2:
    return convt2::apply_grad_sgd(&cell->convt2_op, dW, lr);
  case architecture::OpKind::ConvT3:
    return convt3::apply_grad_sgd(&cell->convt3_op, dW, lr);
  case architecture::OpKind::RNN:
    return rnn::apply_grad_sgd(&cell->rnn_op, dW, lr);
  case architecture::OpKind::LSTM:
    return lstm::apply_grad_sgd(&cell->lstm_op, dW, lr);
  case architecture::OpKind::Sequential:
    return sequential::apply_grad_sgd(&cell->sequential_op, dW, lr);
  case architecture::OpKind::Residual:
    return residual::apply_grad_sgd(&cell->residual_op, dW, lr);
  case architecture::OpKind::Mamba:
    return mamba::apply_grad_sgd(&cell->mamba_op, dW, lr);
  case architecture::OpKind::Stack:
    return stack::apply_grad_sgd(&cell->stack_op, dW, lr);
  case architecture::OpKind::Metacognition:
    return metacognition::apply_grad_sgd(&cell->metacognition_op, dW, lr);
  case architecture::OpKind::GDN:
    return gdn::apply_grad_sgd(&cell->gdn_op, dW, lr);
  case architecture::OpKind::KMeans:
    return kmeans::apply_grad_sgd(&cell->kmeans_op, dW, lr);
  case architecture::OpKind::Parallel:
    return parallel::apply_grad_sgd(&cell->parallel_op, dW, lr);
  default:
    return Status::Ok();
  }
}

Status apply_dense_projected(dense::Layer& d, const TensorF32& grad_out, const TensorF32& input,
                             const TensorF32& pre, const TensorF32& post, double soft_lr) {
  if (grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  TensorF32 proj;
  if (auto st = project_gap(grad_out, post.shape, &proj); !st) return st;
  if (proj.data.empty()) {
    return Status::Ok();
  }
  TensorF32 g_in, dW;
  if (auto st = dense::backward_f32(d, proj, input, pre, &g_in, &dW); !st) {
    return st;
  }
  if (soft_lr != 0.0 && !dW.data.empty()) {
    if (auto st = dense::apply_grad_sgd(&d, dW, soft_lr); !st) {
      return st;
    }
  }
  return Status::Ok();
}

// Flatten [B,D] or [B,S,D] → [B*S,D] for Dense children (matches sequential::forward).
TensorF32 flatten_seq_like(const TensorF32& t) {
  if (t.shape.size() == 3) {
    TensorF32 out;
    const int b = static_cast<int>(t.shape[0]);
    const int s = static_cast<int>(t.shape[1]);
    const int d = static_cast<int>(t.shape[2]);
    out.shape = {static_cast<int64_t>(b) * s, d};
    out.data = t.data;
    return out;
  }
  return t;
}

// Go collectTweenLeaves + applyProjectedTween on Sequential Dense children.
Status apply_projected_seq(sequential::Layer& layer, const TensorF32& grad_out,
                           const TensorF32& input, double soft_lr) {
  if (layer.children.empty()) {
    return Status::Ok();
  }
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 flat_gy = flatten_seq_like(grad_out);
  std::vector<TensorF32> ins(layer.children.size());
  std::vector<TensorF32> pres(layer.children.size());
  std::vector<TensorF32> posts(layer.children.size());
  TensorF32 current = flat_in;
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    auto& ch = layer.children[i];
    ins[i] = current;
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween seq fwd: ") + st.message);
    }
    pres[i] = std::move(p);
    posts[i] = o;
    current = std::move(o);
  }
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    if (auto st = apply_dense_projected(layer.children[i], flat_gy, ins[i], pres[i], posts[i],
                                        soft_lr);
        !st) {
      return st;
    }
  }
  return Status::Ok();
}

// Go residual ChildOps leaf walk — same global gap on each F Dense (no chain rule).
Status apply_projected_res(residual::Layer& layer, const TensorF32& grad_out,
                           const TensorF32& input, double soft_lr) {
  if (layer.children.empty()) {
    return Status::Ok();
  }
  TensorF32 flat_in = flatten_seq_like(input);
  TensorF32 flat_gy = flatten_seq_like(grad_out);
  std::vector<TensorF32> ins(layer.children.size());
  std::vector<TensorF32> pres(layer.children.size());
  std::vector<TensorF32> posts(layer.children.size());
  TensorF32 current = flat_in;
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    auto& ch = layer.children[i];
    ins[i] = current;
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) {
      return Status::Err(st.code, std::string("tween res fwd: ") + st.message);
    }
    pres[i] = std::move(p);
    posts[i] = o;
    current = std::move(o);
  }
  for (std::size_t i = 0; i < layer.children.size(); ++i) {
    if (auto st = apply_dense_projected(layer.children[i], flat_gy, ins[i], pres[i], posts[i],
                                        soft_lr);
        !st) {
      return st;
    }
  }
  return Status::Ok();
}

// Capture every branch leaf before updating any weights. Filter's extra
// forward can advance stateful rule counters, exactly as Go collectTweenLeaves.
Status collect_projected(stack::Child& child, const TensorF32& input,
                         std::vector<ProjectedLeaf>* leaves, TensorF32* output);

Status collect_dense(dense::Layer& dense, const TensorF32& input,
                     std::vector<ProjectedLeaf>* leaves, TensorF32* output) {
  ProjectedLeaf leaf;
  leaf.dense = &dense;
  leaf.input = input;
  if (auto st = dense::forward_f32(dense, input, &leaf.pre, &leaf.post); !st) return st;
  *output = leaf.post;
  leaves->push_back(std::move(leaf));
  return Status::Ok();
}

// Rebuild Parallel output from already-collected branch posts (Go combineCollected /
// Metal combine_collected). Filter keeps a second child_forward.
Status combine_collected(const stack::Child& child, const std::vector<TensorF32>& posts,
                         TensorF32* output) {
  if (!output || posts.empty())
    return Status::Err(StatusCode::InvalidArgument, "tween: empty Parallel combine");
  const auto& nest = child.parallel_nest;
  const auto combine = static_cast<parallel::Combine>(nest.combine);
  const int nb = static_cast<int>(posts.size());
  if (combine == parallel::Combine::Concat) {
    const int rows = posts[0].shape.empty() ? 1 : static_cast<int>(posts[0].shape[0]);
    std::vector<int> feats(static_cast<std::size_t>(nb));
    int od = 0;
    for (int i = 0; i < nb; ++i) {
      if (rows <= 0 || posts[static_cast<std::size_t>(i)].data.size() %
                            static_cast<std::size_t>(rows))
        return Status::Err(StatusCode::InvalidArgument, "tween: Parallel concat shape");
      feats[static_cast<std::size_t>(i)] =
          static_cast<int>(posts[static_cast<std::size_t>(i)].data.size()) / rows;
      od += feats[static_cast<std::size_t>(i)];
    }
    output->shape = {rows, od};
    output->data.assign(static_cast<std::size_t>(rows * od), 0.0f);
    for (int r = 0; r < rows; ++r) {
      int off = r * od;
      for (int i = 0; i < nb; ++i) {
        const int f = feats[static_cast<std::size_t>(i)];
        const int src = r * f;
        const auto& o = posts[static_cast<std::size_t>(i)];
        for (int j = 0; j < f; ++j) {
          output->data[static_cast<std::size_t>(off + j)] =
              o.data[static_cast<std::size_t>(src + j)];
        }
        off += f;
      }
    }
    return Status::Ok();
  }
  const int n = static_cast<int>(posts[0].data.size());
  for (const auto& o : posts) {
    if (static_cast<int>(o.data.size()) != n)
      return Status::Err(StatusCode::InvalidArgument, "tween: unequal Parallel widths");
  }
  output->shape = posts[0].shape;
  output->data.assign(static_cast<std::size_t>(n), 0.0f);
  switch (combine) {
  case parallel::Combine::Add:
    for (const auto& o : posts)
      for (int j = 0; j < n; ++j)
        output->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
    return Status::Ok();
  case parallel::Combine::Avg: {
    const float inv = static_cast<float>(1.0 / static_cast<double>(nb));
    for (const auto& o : posts)
      for (int j = 0; j < n; ++j)
        output->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
    for (int j = 0; j < n; ++j) output->data[static_cast<std::size_t>(j)] *= inv;
    return Status::Ok();
  }
  case parallel::Combine::Max:
    for (int j = 0; j < n; ++j) {
      float best = posts[0].data[static_cast<std::size_t>(j)];
      for (int i = 1; i < nb; ++i) {
        const float v = posts[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)];
        if (v > best) best = v;
      }
      output->data[static_cast<std::size_t>(j)] = best;
    }
    return Status::Ok();
  case parallel::Combine::SparseK: {
    parallel::Config cfg;
    cfg.dim = nest.dim;
    cfg.out_feat = nest.out_feat;
    cfg.branches = nb;
    cfg.combine = combine;
    cfg.seq_len = nest.seq_len;
    cfg.sparse_k = nest.sparse_k;
    cfg.disagree_beta = nest.disagree_beta;
    int k = parallel::sparse_k_eff(cfg);
    if (k > nb) k = nb;
    if (k <= 0) k = 1;
    struct Scored {
      int i;
      double nrm;
    };
    std::vector<Scored> scores(static_cast<std::size_t>(nb));
    for (int i = 0; i < nb; ++i) {
      double s = 0.0;
      for (int j = 0; j < n; ++j) {
        const double v =
            static_cast<double>(posts[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)]);
        s += v * v;
      }
      scores[static_cast<std::size_t>(i)] = {i, std::sqrt(s)};
    }
    std::sort(scores.begin(), scores.end(), [](const Scored& a, const Scored& b) {
      if (a.nrm != b.nrm) return a.nrm > b.nrm;
      return a.i < b.i;
    });
    const float inv = static_cast<float>(1.0 / static_cast<double>(k));
    for (int t = 0; t < k; ++t) {
      const auto& o = posts[static_cast<std::size_t>(scores[static_cast<std::size_t>(t)].i)];
      for (int j = 0; j < n; ++j)
        output->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < n; ++j) output->data[static_cast<std::size_t>(j)] *= inv;
    return Status::Ok();
  }
  case parallel::Combine::Disagree: {
    parallel::Config cfg;
    cfg.dim = nest.dim;
    cfg.out_feat = nest.out_feat;
    cfg.branches = nb;
    cfg.combine = combine;
    cfg.seq_len = nest.seq_len;
    cfg.sparse_k = nest.sparse_k;
    cfg.disagree_beta = nest.disagree_beta;
    const double beta = parallel::disagree_beta_eff(cfg);
    const double inv = 1.0 / static_cast<double>(nb);
    std::vector<double> mean(static_cast<std::size_t>(n), 0.0);
    for (const auto& o : posts)
      for (int j = 0; j < n; ++j)
        mean[static_cast<std::size_t>(j)] +=
            static_cast<double>(o.data[static_cast<std::size_t>(j)]) * inv;
    if (nb == 2) {
      for (int j = 0; j < n; ++j) {
        const double a = static_cast<double>(posts[0].data[static_cast<std::size_t>(j)]);
        const double b = static_cast<double>(posts[1].data[static_cast<std::size_t>(j)]);
        output->data[static_cast<std::size_t>(j)] =
            static_cast<float>(mean[static_cast<std::size_t>(j)] + beta * (a - b));
      }
    } else {
      for (int j = 0; j < n; ++j) {
        const double a0 = static_cast<double>(posts[0].data[static_cast<std::size_t>(j)]);
        output->data[static_cast<std::size_t>(j)] = static_cast<float>(
            mean[static_cast<std::size_t>(j)] + beta * (a0 - mean[static_cast<std::size_t>(j)]));
      }
    }
    return Status::Ok();
  }
  default:
    return Status::Err(StatusCode::InvalidArgument, "tween: unsupported Parallel combine");
  }
}

Status collect_projected(stack::Child& child, const TensorF32& input,
                         std::vector<ProjectedLeaf>* leaves, TensorF32* output) {
  if (child.kind == stack::Kind::NestedStack) {
    TensorF32 current = input;
    for (auto& nested : child.nested) {
      TensorF32 next;
      if (auto st = collect_projected(nested, current, leaves, &next); !st) return st;
      current = std::move(next);
    }
    *output = std::move(current);
    return Status::Ok();
  }
  if (child.kind == stack::Kind::Sequential || child.kind == stack::Kind::Residual) {
    auto& children = child.kind == stack::Kind::Sequential
                         ? child.sequential_op.children : child.residual_op.children;
    TensorF32 flat = flatten_seq_like(input), current = flat;
    for (auto& dense : children) {
      TensorF32 next;
      if (auto st = collect_dense(dense, current, leaves, &next); !st) return st;
      current = std::move(next);
    }
    if (child.kind == stack::Kind::Residual && current.data.size() == flat.data.size()) {
      for (std::size_t i = 0; i < current.data.size(); ++i) {
        current.data[i] += flat.data[i];
      }
    }
    *output = std::move(current);
    return Status::Ok();
  }
  if (child.kind == stack::Kind::Parallel) {
    // Collect every branch / gate leaf before any weight update. Non-Filter combines
    // reuse those posts (Go combineCollected); Filter keeps the extra forward.
    std::vector<TensorF32> posts(child.nested.size());
    for (size_t i = 0; i < child.nested.size(); ++i) {
      if (auto st = collect_projected(child.nested[i], input, leaves, &posts[i]); !st) return st;
    }
    if (child.parallel_nest.has_gate) {
      TensorF32 gate_out;
      if (auto st = collect_dense(child.dense_op, input, leaves, &gate_out); !st) return st;
    }
    if (child.parallel_nest.combine == static_cast<int32_t>(parallel::Combine::Filter)) {
      TensorF32 pre;
      return stack::child_forward(child, input, &pre, output);
    }
    return combine_collected(child, posts, output);
  }
  ProjectedLeaf leaf;
  leaf.child = &child;
  leaf.input = input;
  if (auto st = stack::child_forward(child, input, &leaf.pre, &leaf.post); !st) return st;
  *output = leaf.post;
  if (stack::child_grad_w_size(child) > 0) leaves->push_back(std::move(leaf));
  return Status::Ok();
}

Status apply_projected_parallel(stack::Child& child, const TensorF32& gap,
                                const TensorF32& input, double soft_lr) {
  std::vector<ProjectedLeaf> leaves;
  TensorF32 output;
  if (auto st = collect_projected(child, input, &leaves, &output); !st) return st;
  for (auto& leaf : leaves) {
    TensorF32 projected, gx;
    if (auto st = project_gap(gap, leaf.post.shape, &projected); !st) return st;
    if (leaf.dense) {
      if (auto st = apply_dense_projected(*leaf.dense, projected, leaf.input, leaf.pre,
                                          leaf.post, soft_lr); !st) return st;
    } else if (auto st = stack::child_family_bp(*leaf.child, projected, leaf.input,
                                               leaf.pre, soft_lr, &gx); !st) return st;
  }
  return Status::Ok();
}

} // namespace

Status collect_leaves(stack::Child& child, const TensorF32& input,
                      std::vector<ProjectedLeaf>* leaves, TensorF32* output) {
  if (!leaves || !output)
    return Status::Err(StatusCode::InvalidArgument, "tween: nil collect output");
  return collect_projected(child, input, leaves, output);
}

Status apply_collected(stack::Child& child,const TensorF32& gy,const TensorF32& input,double lr) {
 return apply_projected_parallel(child,gy,input,lr);
}
Status apply_collected(stack::Layer& ops,const TensorF32& gy,const TensorF32& input,double lr) {
 stack::Child root;root.kind=stack::Kind::NestedStack;root.nested=std::move(ops.children);
 auto st=apply_projected_parallel(root,gy,input,lr);ops.children=std::move(root.nested);return st;
}

Status project_gap(const TensorF32& src, const std::vector<int64_t>& dst_shape, TensorF32* out) {
  if (!out || dst_shape.empty() || src.data.empty())
    return Status::Err(StatusCode::InvalidArgument, "tween: empty projection");
  size_t count = 1;
  for (auto dim : dst_shape) {
    if (dim <= 0 || size_t(dim) > size_t(INT32_MAX) / count)
      return Status::Err(StatusCode::InvalidArgument, "tween: invalid projection shape");
    count *= size_t(dim);
  }
  if (src.data.size() > size_t(INT32_MAX))
    return Status::Err(StatusCode::InvalidArgument, "tween: projection source too large");
  TensorF32 dst;
  dst.shape = dst_shape;
  dst.data.resize(count);
  if (static_cast<std::size_t>(src.data.size()) == count) {
    dst.data = src.data;
    *out = std::move(dst);
    return Status::Ok();
  }
  if (auto st = cuda::launch_project_gap(src.data.data(), dst.data.data(),
                                         int32_t(src.data.size()), int32_t(count));
      !st) {
    return st;
  }
  *out = std::move(dst);
  return Status::Ok();
}

Status apply_projected_ops(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input,
                           double soft_lr) {
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  TensorF32 current = input;
  for (stack::Child* ch : flat) {
    TensorF32 pre, post;
    if (auto st = stack::child_forward(*ch, current, &pre, &post); !st) return st;
    if (ch->kind == stack::Kind::Sequential || ch->kind == stack::Kind::Residual ||
        ch->kind == stack::Kind::Parallel) {
      if (auto st = apply_projected(*ch, grad_out, current, soft_lr); !st) return st;
    } else {
      TensorF32 proj, g_in;
      if (auto st = project_gap(grad_out, post.shape, &proj); !st) return st;
      if (auto st = stack::child_family_bp(*ch, proj, current, pre, soft_lr, &g_in); !st)
        return st;
    }
    // Feed the pre-update output to the next leaf, matching Go's collect tape.
    current = std::move(post);
  }
  return Status::Ok();
}

Status apply_projected(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                       double soft_lr) {
  if (grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  switch (c.kind) {
  case stack::Kind::Sequential:
    return apply_projected_seq(c.sequential_op, grad_out, input, soft_lr);
  case stack::Kind::Residual:
    return apply_projected_res(c.residual_op, grad_out, input, soft_lr);
  case stack::Kind::Parallel:
    return apply_projected_parallel(c, grad_out, input, soft_lr);
  case stack::Kind::NestedStack: {
    stack::Layer ops;
    ops.children = std::move(c.nested);
    auto st = apply_projected_ops(ops, grad_out, input, soft_lr);
    c.nested = std::move(ops.children);
    return st;
  }
  default: {
    TensorF32 pre, post, g_in;
    if (auto st = stack::child_forward(c, input, &pre, &post); !st) {
      return st;
    }
    TensorF32 proj;
    if (auto st = project_gap(grad_out, post.shape, &proj); !st) return st;
    return stack::child_family_bp(c, proj, input, pre, soft_lr, &g_in);
  }
  }
}

Status apply_cell_projected(architecture::Cell& cell, const TensorF32& grad_out,
                            const TensorF32& input, const TensorF32& pre, const TensorF32& post,
                            double soft_lr) {
  if (cell.meta.is_disabled || grad_out.data.empty() || input.data.empty()) {
    return Status::Ok();
  }
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return apply_projected_seq(cell.sequential_op, grad_out, input, soft_lr);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return apply_projected_res(cell.residual_op, grad_out, input, soft_lr);
  }
  TensorF32 proj;
  if (auto st = project_gap(grad_out, post.shape, &proj); !st) return st;
  if (proj.data.empty()) {
    return Status::Ok();
  }
  TensorF32 g_in, dW;
  if (auto st = runtime::backward::cell(cell, proj, input, pre, &g_in, &dW); !st) {
    return st;
  }
  return apply_one(&cell, dW, soft_lr);
}

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss) {
  if (!loss) {
    return Status::Err(StatusCode::InvalidArgument, "tween: nil loss");
  }
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween: empty forward tape");
  }
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) {
    return st;
  }
  TensorF32 gy;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) {
    return st;
  }
  const double soft_lr = lr * 0.5;
  // Reverse hop — Go trainStackMixed + trainTweenLocal per cell.
  for (int i = static_cast<int>(fwd.steps.size()) - 1; i >= 0; --i) {
    const auto& step = fwd.steps[static_cast<std::size_t>(i)];
    architecture::Cell* cell =
        architecture::at(g, step.coord.z, step.coord.y, step.coord.x, step.coord.l);
    if (!cell || cell->meta.is_disabled) {
      continue;
    }
    if (auto st = apply_cell_projected(*cell, gy, step.input, step.pre, step.post, soft_lr); !st) {
      return st;
    }
    // Go trainTweenLocal: pass gap forward only when lengths match input.
    TensorF32 next;
    next.shape = step.input.shape;
    next.data.assign(step.input.data.size(), 0.0f);
    if (gy.data.size() == next.data.size()) {
      next.data = gy.data;
    }
    gy = std::move(next);
  }
  return Status::Ok();
}

} // namespace welvet::systems::tween
