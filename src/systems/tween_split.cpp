#include "welvet/systems/tween_split.hpp"

#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/training.hpp"
#include "welvet/systems/tween.hpp"

#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/gdn.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/layernorm.hpp"
#include "welvet/lstm.hpp"
#include "welvet/mamba.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mha.hpp"
#include "welvet/parallel.hpp"
#include "welvet/residual.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/softmax.hpp"
#include "welvet/stack.hpp"
#include "welvet/swiglu.hpp"

#include <string>
#include <utility>
#include <vector>

namespace welvet::systems::tween_split {
namespace {

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

void scale_gap(TensorF32* g, double scale) {
  if (!g) return;
  for (float& v : g->data) {
    v = static_cast<float>(static_cast<double>(v) * scale);
  }
}

struct Leaf {
  bool dw_only = false;
  stack::Child* child = nullptr;
  dense::Layer* dense = nullptr;
  architecture::Cell* cell = nullptr;
  TensorF32 input;
  TensorF32 pre;
  TensorF32 post;
};

Status apply_dense_dw(dense::Layer& d, const TensorF32& gy, const TensorF32& input,
                      const TensorF32& pre, double lr, bool dw_only) {
  if (gy.data.empty() || input.data.empty()) return Status::Ok();
  TensorF32 g_in, dW;
  // GradWOnly ≡ Backward dW with discarded gin (same outer product).
  if (auto st = dense::backward_f32(d, gy, input, pre, &g_in, &dW); !st) return st;
  (void)dw_only;
  if (lr != 0.0 && !dW.data.empty()) return dense::apply_grad_sgd(&d, dW, lr);
  return Status::Ok();
}

Status apply_cell_dw(architecture::Cell& cell, const TensorF32& dW, double lr) {
  if (dW.data.empty() || lr == 0.0) return Status::Ok();
  switch (cell.op_kind) {
  case architecture::OpKind::Dense:
    return dense::apply_grad_sgd(&cell.dense_op, dW, lr);
  case architecture::OpKind::RMSNorm:
    return rmsnorm::apply_grad_sgd(&cell.rmsnorm_op, dW, lr);
  case architecture::OpKind::LayerNorm:
    return layernorm::apply_grad_sgd(&cell.layernorm_op, dW, lr);
  case architecture::OpKind::SwiGLU:
    return swiglu::apply_grad_sgd(&cell.swiglu_op, dW, lr);
  case architecture::OpKind::Softmax:
    return Status::Ok();
  case architecture::OpKind::Embedding:
    return embedding::apply_grad_sgd(&cell.embedding_op, dW, lr);
  case architecture::OpKind::MHA:
    return mha::apply_grad_sgd(&cell.mha_op, dW, lr);
  case architecture::OpKind::CNN1:
    return cnn1::apply_grad_sgd(&cell.cnn1_op, dW, lr);
  case architecture::OpKind::CNN2:
    return cnn2::apply_grad_sgd(&cell.cnn2_op, dW, lr);
  case architecture::OpKind::CNN3:
    return cnn3::apply_grad_sgd(&cell.cnn3_op, dW, lr);
  case architecture::OpKind::ConvT1:
    return convt1::apply_grad_sgd(&cell.convt1_op, dW, lr);
  case architecture::OpKind::ConvT2:
    return convt2::apply_grad_sgd(&cell.convt2_op, dW, lr);
  case architecture::OpKind::ConvT3:
    return convt3::apply_grad_sgd(&cell.convt3_op, dW, lr);
  case architecture::OpKind::RNN:
    return rnn::apply_grad_sgd(&cell.rnn_op, dW, lr);
  case architecture::OpKind::LSTM:
    return lstm::apply_grad_sgd(&cell.lstm_op, dW, lr);
  case architecture::OpKind::Sequential:
    return sequential::apply_grad_sgd(&cell.sequential_op, dW, lr);
  case architecture::OpKind::Residual:
    return residual::apply_grad_sgd(&cell.residual_op, dW, lr);
  case architecture::OpKind::Mamba:
    return mamba::apply_grad_sgd(&cell.mamba_op, dW, lr);
  case architecture::OpKind::Stack:
    return stack::apply_grad_sgd(&cell.stack_op, dW, lr);
  case architecture::OpKind::Metacognition:
    return metacognition::apply_grad_sgd(&cell.metacognition_op, dW, lr);
  case architecture::OpKind::GDN:
    return gdn::apply_grad_sgd(&cell.gdn_op, dW, lr);
  case architecture::OpKind::KMeans:
    return kmeans::apply_grad_sgd(&cell.kmeans_op, dW, lr);
  case architecture::OpKind::Parallel:
    return parallel::apply_grad_sgd(&cell.parallel_op, dW, lr);
  default:
    return Status::Ok();
  }
}

Status apply_one_leaf(Leaf& leaf, const TensorF32& gy, double lr) {
  if (leaf.dense) {
    return apply_dense_dw(*leaf.dense, gy, leaf.input, leaf.pre, lr, leaf.dw_only);
  }
  if (leaf.child) {
    if (leaf.dw_only && leaf.child->kind == stack::Kind::Dense) {
      return apply_dense_dw(leaf.child->dense_op, gy, leaf.input, leaf.pre, lr, true);
    }
    TensorF32 g_in;
    return stack::child_family_bp(*leaf.child, gy, leaf.input, leaf.pre, lr, &g_in);
  }
  if (leaf.cell) {
    TensorF32 g_in, dW;
    if (auto st = runtime::backward::cell(*leaf.cell, gy, leaf.input, leaf.pre, &g_in, &dW); !st) {
      return st;
    }
    return apply_cell_dw(*leaf.cell, dW, lr);
  }
  return Status::Ok();
}

Status collect_seq_dense(std::vector<dense::Layer>& children, const TensorF32& input,
                         std::vector<Leaf>* out) {
  TensorF32 current = flatten_seq_like(input);
  for (auto& ch : children) {
    Leaf leaf;
    leaf.dense = &ch;
    leaf.dw_only = true;
    leaf.input = current;
    TensorF32 p, o;
    if (auto st = dense::forward_f32(ch, current, &p, &o); !st) return st;
    leaf.pre = std::move(p);
    leaf.post = o;
    out->push_back(std::move(leaf));
    current = std::move(o);
  }
  return Status::Ok();
}

Status collect_child(stack::Child& c, const TensorF32& input, std::vector<Leaf>* out,
                     TensorF32* output = nullptr) {
  if (c.kind == stack::Kind::Parallel) {
    std::vector<systems::tween::ProjectedLeaf> captured;
    TensorF32 post;
    if (auto st = systems::tween::collect_leaves(c, input, &captured, &post); !st) return st;
    for (auto& source : captured) {
      Leaf leaf;
      leaf.child = source.child;
      leaf.dense = source.dense;
      leaf.dw_only = true;
      leaf.input = std::move(source.input);
      leaf.pre = std::move(source.pre);
      leaf.post = std::move(source.post);
      out->push_back(std::move(leaf));
    }
    if (output) *output = std::move(post);
    return Status::Ok();
  }
  if (c.kind == stack::Kind::Sequential || c.kind == stack::Kind::Residual) {
    const TensorF32 original = input;
    const size_t start = out->size();
    auto& children = c.kind == stack::Kind::Sequential ? c.sequential_op.children
                                                       : c.residual_op.children;
    if (auto st = collect_seq_dense(children, original, out); !st) return st;
    for (size_t i = start; i < out->size(); ++i) (*out)[i].dw_only = true;
    if (output) {
      *output = out->size() > start ? out->back().post : original;
      if (c.kind == stack::Kind::Residual && output->data.size() == original.data.size()) {
        output->shape = original.shape;
        for (size_t i = 0; i < output->data.size(); ++i) {
          output->data[i] += original.data[i];
        }
      }
    }
    return Status::Ok();
  }
  if (c.kind == stack::Kind::NestedStack) {
    TensorF32 current = input;
    for (auto& ch : c.nested) {
      if (auto st = collect_child(ch, current, out, &current); !st) return st;
    }
    if (output) *output = std::move(current);
    return Status::Ok();
  }
  Leaf leaf;
  leaf.dw_only = true;
  leaf.child = &c;
  leaf.input = input;
  if (auto st = stack::child_forward(c, input, &leaf.pre, &leaf.post); !st) return st;
  if (output) *output = leaf.post;
  if (stack::child_grad_w_size(c) > 0) out->push_back(std::move(leaf));
  return Status::Ok();
}

Status collect_cell(architecture::Cell& cell, const TensorF32& input, const TensorF32& pre,
                    const TensorF32& post, std::vector<Leaf>* out) {
  if (cell.meta.is_disabled) return Status::Ok();
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return collect_seq_dense(cell.sequential_op.children, input, out);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return collect_seq_dense(cell.residual_op.children, input, out);
  }
  if (cell.op_kind == architecture::OpKind::Softmax || cell.op_kind == architecture::OpKind::None) {
    return Status::Ok();
  }
  Leaf leaf;
  leaf.cell = &cell;
  leaf.input = input;
  leaf.pre = pre;
  leaf.post = post;
  out->push_back(std::move(leaf));
  return Status::Ok();
}

Status apply_leaves(std::vector<Leaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  const double scale = 1.0 / static_cast<double>(n);
  for (int i = 0; i < n; ++i) {
    Leaf& leaf = leaves[static_cast<std::size_t>(i)];
    if (leaf.post.data.empty()) continue;
    TensorF32 gy;
    if (auto st = systems::tween::project_gap(grad_out, leaf.post.shape, &gy); !st) return st;
    scale_gap(&gy, scale);
    if (auto st = apply_one_leaf(leaf, gy, lr); !st) {
      return Status::Err(st.code, std::string("tween_split leaf: ") + st.message);
    }
  }
  return Status::Ok();
}

// Head full Backward+SGD; returns true input gradient (Go trainTweenSplitHeadProxy).
Status apply_leaf_gx(Leaf& leaf, const TensorF32& gy, double lr, TensorF32* gx_out) {
  if (!gx_out) return Status::Err(StatusCode::InvalidArgument, "tween_split: nil gx");
  *gx_out = TensorF32{};
  if (gy.data.empty() || leaf.input.data.empty()) return Status::Ok();
  if (leaf.dense) {
    TensorF32 dW;
    if (auto st = dense::backward_f32(*leaf.dense, gy, leaf.input, leaf.pre, gx_out, &dW); !st)
      return st;
    if (lr != 0.0 && !dW.data.empty()) return dense::apply_grad_sgd(leaf.dense, dW, lr);
    return Status::Ok();
  }
  if (leaf.child) {
    return stack::child_family_bp(*leaf.child, gy, leaf.input, leaf.pre, lr, gx_out);
  }
  if (leaf.cell) {
    TensorF32 dW;
    if (auto st = runtime::backward::cell(*leaf.cell, gy, leaf.input, leaf.pre, gx_out, &dW); !st)
      return st;
    return apply_cell_dw(*leaf.cell, dW, lr);
  }
  return Status::Ok();
}

Status apply_hidden_dw(std::vector<Leaf>& leaves, int begin, int end, const TensorF32& src,
                       double scale, double lr) {
  for (int i = begin; i < end; ++i) {
    Leaf& leaf = leaves[static_cast<std::size_t>(i)];
    if (leaf.post.data.empty()) continue;
    TensorF32 gi;
    if (auto st = systems::tween::project_gap(src, leaf.post.shape, &gi); !st) return st;
    scale_gap(&gi, scale);
    if (auto st = apply_one_leaf(leaf, gi, lr); !st) {
      return Status::Err(st.code, std::string("tween_split hidden: ") + st.message);
    }
  }
  return Status::Ok();
}

Status apply_head_proxy_leaves(std::vector<Leaf>& leaves, const TensorF32& grad_out, double lr) {
  const int n = static_cast<int>(leaves.size());
  if (n == 0) return Status::Ok();
  Leaf& head = leaves[static_cast<std::size_t>(n - 1)];
  if (head.post.data.empty()) return apply_leaves(leaves, grad_out, lr);
  TensorF32 g_head;
  if (auto st = systems::tween::project_gap(grad_out, head.post.shape, &g_head); !st) return st;
  TensorF32 gx;
  if (auto st = apply_leaf_gx(head, g_head, lr, &gx); !st) {
    return Status::Err(st.code, std::string("tween_split head: ") + st.message);
  }
  if (n == 1) return Status::Ok();
  const TensorF32& src = gx.data.empty() ? grad_out : gx;
  const double scale = 1.0 / static_cast<double>(n - 1);
  return apply_hidden_dw(leaves, 0, n - 1, src, scale, lr);
}

Status apply_variant_leaves(std::vector<Leaf>& leaves, const TensorF32& gy, double lr,
                            Variant variant) {
  if (static_cast<int>(variant) < 0 || static_cast<int>(variant) > 6) {
    return Status::Gap("split: unsupported variant");
  }
  if (variant == Variant::Split) return apply_leaves(leaves, gy, lr);
  if (variant == Variant::HeadProxy) return apply_head_proxy_leaves(leaves, gy, lr);
  return Status::Gap(std::string("tween_split variant ") +
                     std::to_string(static_cast<int>(variant)) + " not wired on CUDA yet");
}

Status collect_stack_leaves(stack::Layer& ops, const TensorF32& input, std::vector<Leaf>* leaves,
                            TensorF32* output = nullptr) {
  std::vector<stack::Child*> flat;
  stack::flatten_ops(ops, &flat);
  if (flat.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: empty stack");
  }
  TensorF32 current = input;
  for (auto* child : flat) {
    if (auto st = collect_child(*child, current, leaves, &current); !st) return st;
  }
  if (output) *output = std::move(current);
  return Status::Ok();
}

Status collect_volume_leaves(architecture::Grid& g, const runtime::forward::Result& fwd,
                             std::vector<Leaf>* leaves) {
  for (const auto& step : fwd.steps) {
    architecture::Cell* cell =
        architecture::at(g, step.coord.z, step.coord.y, step.coord.x, step.coord.l);
    if (!cell) continue;
    if (auto st = collect_cell(*cell, step.input, step.pre, step.post, leaves); !st) return st;
  }
  return Status::Ok();
}

Status collect_cell_leaves(architecture::Cell& cell, const TensorF32& input, const TensorF32& pre,
                           const TensorF32& post, std::vector<Leaf>* leaves) {
  if (!leaves) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: nil leaves");
  }
  leaves->clear();
  if (cell.op_kind == architecture::OpKind::Sequential) {
    return collect_seq_dense(cell.sequential_op.children, input, leaves);
  }
  if (cell.op_kind == architecture::OpKind::Residual) {
    return collect_seq_dense(cell.residual_op.children, input, leaves);
  }
  Leaf lf;
  lf.cell = &cell;
  lf.input = input;
  lf.pre = pre;
  lf.post = post;
  leaves->push_back(lf);
  return Status::Ok();
}

} // namespace

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                       double lr, double* loss) {
  if (!loss) return Status::Err(StatusCode::InvalidArgument, "tween_split: nil loss");
  std::vector<Leaf> leaves;
  TensorF32 output, gy;
  if (auto st = collect_stack_leaves(ops, input, &leaves, &output); !st) return st;
  if (auto st = runtime::training::mse(output, target, loss); !st) return st;
  if (auto st = runtime::training::mse_grad(output, target, &gy); !st) return st;
  return apply_leaves(leaves, gy, lr);
}

Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input, double lr,
                   bool ordinary_backward) {
  std::vector<Leaf> leaves;
  if (auto st = collect_stack_leaves(ops, input, &leaves); !st) return st;
  if (ordinary_backward) {
    for (auto& leaf : leaves) leaf.dw_only = false;
  }
  return apply_leaves(leaves, grad_out, lr);
}

Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr, bool ordinary_backward) {
  std::vector<Leaf> leaves;
  if (auto st = collect_child(c, input, &leaves); !st) return st;
  if (ordinary_backward) {
    for (auto& leaf : leaves) leaf.dw_only = false;
  }
  return apply_leaves(leaves, grad_out, lr);
}

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss) {
  if (!loss) return Status::Err(StatusCode::InvalidArgument, "tween_split: nil loss");
  if (fwd.steps.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "tween_split: empty forward tape");
  }
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) return st;
  TensorF32 gy;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) return st;
  std::vector<Leaf> leaves;
  if (auto st = collect_volume_leaves(g, fwd, &leaves); !st) return st;
  return apply_leaves(leaves, gy, lr);
}

Status apply_cell_split(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
                        const TensorF32& pre, const TensorF32& post, double lr) {
  if (cell.meta.is_disabled || grad_out.data.empty() || input.data.empty()) return Status::Ok();
  std::vector<Leaf> leaves;
  if (auto st = collect_cell_leaves(cell, input, pre, post, &leaves); !st) return st;
  return apply_leaves(leaves, grad_out, lr);
}

Status train_stack_variant(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                           double lr, double* loss, Variant variant) {
  if (!loss) return Status::Err(StatusCode::InvalidArgument, "proxy: nil loss");
  std::vector<Leaf> leaves;
  TensorF32 post, gy;
  if (auto st = collect_stack_leaves(ops, input, &leaves, &post); !st) return st;
  if (auto st = runtime::training::mse(post, target, loss); !st) return st;
  if (auto st = runtime::training::mse_grad(post, target, &gy); !st) return st;
  return apply_variant_leaves(leaves, gy, lr, variant);
}

Status apply_variant_child(stack::Child& child, const TensorF32& gy, const TensorF32& input,
                           double lr, Variant variant) {
  std::vector<Leaf> leaves;
  if (auto st = collect_child(child, input, &leaves); !st) return st;
  return apply_variant_leaves(leaves, gy, lr, variant);
}

Status train_volume_variant(architecture::Grid& g, const runtime::forward::Result& fwd,
                            const TensorF32& target, double lr, double* loss, Variant variant) {
  if (!loss) return Status::Err(StatusCode::InvalidArgument, "proxy: nil loss");
  TensorF32 gy;
  std::vector<Leaf> leaves;
  if (auto st = runtime::training::mse(fwd.output, target, loss); !st) return st;
  if (auto st = runtime::training::mse_grad(fwd.output, target, &gy); !st) return st;
  if (auto st = collect_volume_leaves(g, fwd, &leaves); !st) return st;
  return apply_variant_leaves(leaves, gy, lr, variant);
}

Status apply_cell_variant(architecture::Cell& cell, const TensorF32& gy, const TensorF32& input,
                          const TensorF32& pre, const TensorF32& post, double lr, Variant variant) {
  if (cell.meta.is_disabled || input.data.empty() || gy.data.empty()) return Status::Ok();
  std::vector<Leaf> leaves;
  if (auto st = collect_cell_leaves(cell, input, pre, post, &leaves); !st) return st;
  return apply_variant_leaves(leaves, gy, lr, variant);
}

} // namespace welvet::systems::tween_split
