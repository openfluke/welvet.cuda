#pragma once

#include "welvet/architecture.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/stack.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

namespace welvet::systems::tween_split {

// ModeTweenSplit: one global gap, split 1/N across every trainable leaf
// (Go applySplitEven / OpenSplitTape). Full lr; no chain rule.
// Dense leaves use GradWOnly-equivalent dW; other Ops use local Backward+SGD.

// Proxy variants (metal/Go enum). … FastProxy=3 §16; LinearCache=4 §17; others later.
enum class Variant : int {
  Split = 0,
  HeadProxy = 1,
  Linear = 2,
  FastProxy = 3,
  LinearCache = 4,
  HeadProxyAsync = 5,
  Sparse = 6,
};

Status train_stack_mse(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                       double lr, double* loss);

Status apply_split(stack::Layer& ops, const TensorF32& grad_out, const TensorF32& input, double lr,
                   bool ordinary_backward = false);

Status apply_split_child(stack::Child& c, const TensorF32& grad_out, const TensorF32& input,
                         double lr, bool ordinary_backward = false);

Status train_volume_mse(architecture::Grid& g, const runtime::forward::Result& fwd,
                        const TensorF32& target, double lr, double* loss);

Status apply_cell_split(architecture::Cell& cell, const TensorF32& grad_out, const TensorF32& input,
                        const TensorF32& pre, const TensorF32& post, double lr);

// ModeTweenSplitHeadProxy: head full Backward+SGD @ full lr; hidden 1/(N−1) GradWOnly on head gx.
Status train_stack_variant(stack::Layer& ops, const TensorF32& input, const TensorF32& target,
                           double lr, double* loss, Variant variant);
Status apply_variant_child(stack::Child& child, const TensorF32& gy, const TensorF32& input,
                           double lr, Variant variant);
Status train_volume_variant(architecture::Grid& g, const runtime::forward::Result& fwd,
                            const TensorF32& target, double lr, double* loss, Variant variant);
Status apply_cell_variant(architecture::Cell& cell, const TensorF32& gy, const TensorF32& input,
                          const TensorF32& pre, const TensorF32& post, double lr, Variant variant);

} // namespace welvet::systems::tween_split
