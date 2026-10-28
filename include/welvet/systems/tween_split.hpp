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

} // namespace welvet::systems::tween_split
