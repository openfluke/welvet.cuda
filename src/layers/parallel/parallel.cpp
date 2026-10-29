#include "welvet/parallel.hpp"

#include "welvet/activate.hpp"
#include "welvet/cuda/cameral.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

namespace welvet::parallel {
namespace {

struct Layout {
  int batch = 0;
  int rows = 0; // seq or 1
  int dim = 0;
  bool poly_shape = false; // CNN/… non-[…,Dim] layout
  std::vector<int64_t> shape;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: empty input");
  }
  lay->dim = cfg.dim;
  lay->shape = input.shape;
  lay->poly_shape = false;
  if (cfg.seq_len > 0) {
    if (input.shape.size() != 3 || static_cast<int>(input.shape[1]) != cfg.seq_len ||
        static_cast<int>(input.shape[2]) != cfg.dim) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: need [batch,seq,dim]");
    }
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = cfg.seq_len;
  } else if (input.shape.size() == 2 && static_cast<int>(input.shape[1]) == cfg.dim) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = 1;
  } else if (input.shape.size() == 3 && static_cast<int>(input.shape[2]) == cfg.dim) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = static_cast<int>(input.shape[1]);
  } else {
    // Polymorphic ops (CNN/RNN/…) may use non-[…,Dim] layouts.
    if (input.shape.empty() || input.shape[0] <= 0) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: unsupported shape");
    }
    lay->batch = static_cast<int>(input.shape[0]);
    lay->rows = 1;
    lay->poly_shape = true;
  }
  if (lay->batch <= 0 || lay->rows <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: invalid batch/rows");
  }
  if (!lay->poly_shape) {
    const int want = lay->batch * lay->rows * lay->dim;
    if (static_cast<int>(input.data.size()) < want) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: data short");
    }
  }
  return Status::Ok();
}

TensorF32 flatten(const TensorF32& input, const Layout& lay) {
  TensorF32 out;
  const int n = lay.batch * lay.rows;
  out.shape = {n, lay.dim};
  out.data.assign(input.data.begin(),
                  input.data.begin() + static_cast<std::ptrdiff_t>(n * lay.dim));
  return out;
}

TensorF32 unflatten(const TensorF32& flat, const Layout& lay, int out_dim) {
  TensorF32 out;
  if (lay.rows == 1 && lay.shape.size() == 2) {
    out.shape = {lay.batch, out_dim};
  } else if (lay.poly_shape) {
    out.shape = {lay.batch, out_dim};
  } else {
    out.shape = {lay.batch, lay.rows, out_dim};
  }
  const int n = lay.batch * lay.rows * out_dim;
  out.data.assign(flat.data.begin(), flat.data.begin() + static_cast<std::ptrdiff_t>(n));
  return out;
}

std::vector<double> softmax_f64(const std::vector<double>& logits) {
  double max_z = logits[0];
  for (double z : logits) {
    if (z > max_z) max_z = z;
  }
  std::vector<double> out(logits.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double e = std::exp(logits[i] - max_z);
    out[i] = e;
    sum += e;
  }
  if (sum == 0.0) {
    const double inv = 1.0 / static_cast<double>(out.size());
    for (double& v : out) v = inv;
    return out;
  }
  for (double& v : out) v /= sum;
  return out;
}

Status combine_equal_width(const Config& cfg, const std::vector<TensorF32>& branch_out,
                           TensorF32* out) {
  if (!out || branch_out.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: combineEqualWidth empty");
  }
  const int nb = static_cast<int>(branch_out.size());
  const int n = static_cast<int>(branch_out[0].data.size());
  for (const auto& o : branch_out) {
    if (static_cast<int>(o.data.size()) != n) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: unequal branch widths");
    }
  }
  out->shape = branch_out[0].shape;
  out->data.assign(static_cast<std::size_t>(n), 0.0f);

  switch (cfg.combine) {
    case Combine::Add:
      for (const auto& o : branch_out) {
        for (int j = 0; j < n; ++j) {
          out->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
        }
      }
      break;
    case Combine::Avg: {
      const float inv = static_cast<float>(1.0 / static_cast<double>(nb));
      for (const auto& o : branch_out) {
        for (int j = 0; j < n; ++j) {
          out->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
        }
      }
      for (int j = 0; j < n; ++j) {
        out->data[static_cast<std::size_t>(j)] *= inv;
      }
      break;
    }
    case Combine::Max:
      for (int j = 0; j < n; ++j) {
        double best = static_cast<double>(branch_out[0].data[static_cast<std::size_t>(j)]);
        for (int i = 1; i < nb; ++i) {
          const double v =
              static_cast<double>(branch_out[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)]);
          if (v > best) best = v;
        }
        out->data[static_cast<std::size_t>(j)] = static_cast<float>(best);
      }
      break;
    case Combine::SparseK: {
      int k = sparse_k_eff(cfg);
      if (k > nb) k = nb;
      struct Scored {
        int i;
        double nrm;
      };
      std::vector<Scored> scores(static_cast<std::size_t>(nb));
      for (int i = 0; i < nb; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
          const double v =
              static_cast<double>(branch_out[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)]);
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
        const auto& o = branch_out[static_cast<std::size_t>(scores[static_cast<std::size_t>(t)].i)];
        for (int j = 0; j < n; ++j) {
          out->data[static_cast<std::size_t>(j)] += o.data[static_cast<std::size_t>(j)];
        }
      }
      for (int j = 0; j < n; ++j) {
        out->data[static_cast<std::size_t>(j)] *= inv;
      }
      break;
    }
    case Combine::Disagree: {
      const double beta = disagree_beta_eff(cfg);
      const double inv = 1.0 / static_cast<double>(nb);
      std::vector<double> mean(static_cast<std::size_t>(n), 0.0);
      for (const auto& o : branch_out) {
        for (int j = 0; j < n; ++j) {
          mean[static_cast<std::size_t>(j)] +=
              static_cast<double>(o.data[static_cast<std::size_t>(j)]) * inv;
        }
      }
      if (nb == 2) {
        for (int j = 0; j < n; ++j) {
          const double a = static_cast<double>(branch_out[0].data[static_cast<std::size_t>(j)]);
          const double b = static_cast<double>(branch_out[1].data[static_cast<std::size_t>(j)]);
          out->data[static_cast<std::size_t>(j)] =
              static_cast<float>(mean[static_cast<std::size_t>(j)] + beta * (a - b));
        }
      } else {
        for (int j = 0; j < n; ++j) {
          const double a0 = static_cast<double>(branch_out[0].data[static_cast<std::size_t>(j)]);
          out->data[static_cast<std::size_t>(j)] = static_cast<float>(
              mean[static_cast<std::size_t>(j)] + beta * (a0 - mean[static_cast<std::size_t>(j)]));
        }
      }
      break;
    }
    default:
      return Status::Err(StatusCode::InvalidArgument, "parallel: combineEqualWidth unsupported");
  }
  return Status::Ok();
}

bool layer_needs_flat(const Layer& layer) {
  if (layer.gate) return true;
  for (const auto& br : layer.branches) {
    if (branch_needs_dense_flat(br)) return true;
  }
  return false;
}

} // namespace

int out_dim(const Config& cfg) {
  if (cfg.out_feat <= 0) return 0;
  if (cfg.combine == Combine::Concat) return cfg.branches * cfg.out_feat;
  return cfg.out_feat;
}

int weight_count(const Config& cfg) {
  const int branch_w = cfg.branches * cfg.out_feat * cfg.dim;
  const int gate_w = cfg.combine == Combine::Filter ? cfg.branches * cfg.dim : 0;
  return branch_w + gate_w;
}

int sparse_k_eff(const Config& cfg) {
  int k = cfg.sparse_k;
  if (k <= 0) k = 2;
  if (k > cfg.branches) k = cfg.branches;
  if (k < 1) k = 1;
  return k;
}

double disagree_beta_eff(const Config& cfg) {
  if (cfg.disagree_beta == 0.0) return 1.0;
  return cfg.disagree_beta;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "parallel: nil config");
  if (cfg->dim <= 0 || cfg->branches <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: need positive Dim/Branches");
  }
  if (cfg->out_feat < 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: OutFeat < 0");
  }
  switch (cfg->combine) {
    case Combine::Concat:
    case Combine::Add:
    case Combine::Avg:
    case Combine::Filter:
    case Combine::Max:
    case Combine::SparseK:
    case Combine::Disagree:
      break;
    default:
      return Status::Err(StatusCode::InvalidArgument, "parallel: unknown Combine");
  }
  if (cfg->seq_len < 0 || cfg->sparse_k < 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: SeqLen/SparseK < 0");
  }
  return Status::Ok();
}

bool branch_needs_dense_flat(const Branch& br) {
  return br.kind == BranchKind::Dense;
}

Status branch_forward(Branch& br, const TensorF32& input, const TensorF32* flat, TensorF32* pre,
                      TensorF32* post) {
  if (br.kind == BranchKind::Dense && flat) {
    return stack::child_forward(br, *flat, pre, post);
  }
  return stack::child_forward(br, input, pre, post);
}

Status branch_backward(Branch& br, const TensorF32& grad_out, const TensorF32& input,
                       const TensorF32* flat, const TensorF32& pre, TensorF32* grad_in,
                       TensorF32* grad_w) {
  if (br.kind == BranchKind::Dense && flat) {
    return stack::child_backward(br, grad_out, *flat, pre, grad_in, grad_w);
  }
  return stack::child_backward(br, grad_out, input, pre, grad_in, grad_w);
}

Status branch_apply_grad_sgd(Branch* br, const TensorF32& dW, double lr) {
  return stack::child_apply_sgd(br, dW, lr);
}

int branch_grad_w_size(const Branch& br) { return stack::child_grad_w_size(br); }

Status branch_export_f32(const Branch& br, float* dst) {
  return stack::child_export_f32(br, dst);
}

dense::Layer* dense_branch(Layer& l, int i) {
  if (i < 0 || i >= static_cast<int>(l.branches.size())) return nullptr;
  auto& br = l.branches[static_cast<std::size_t>(i)];
  return br.kind == BranchKind::Dense ? &br.dense_op : nullptr;
}

const dense::Layer* dense_branch(const Layer& l, int i) {
  if (i < 0 || i >= static_cast<int>(l.branches.size())) return nullptr;
  const auto& br = l.branches[static_cast<std::size_t>(i)];
  return br.kind == BranchKind::Dense ? &br.dense_op : nullptr;
}

TrainMode effective_branch_mode(const Layer& l, int i, TrainMode parent) {
  return resolve_train_mode(i >= 0 && i < static_cast<int>(l.controls.modes.size())
                                ? l.controls.modes[static_cast<std::size_t>(i)]
                                : TrainMode::Inherit,
                            parent);
}

bool has_mixed_branch_modes(const Layer& l) {
  for (auto mode : l.controls.modes) {
    if (mode != TrainMode::Inherit) return true;
  }
  return false;
}

Status new_configured(const Config& cfg_in, DType dt, const float* packed_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "parallel: nil out");
  Config cfg = cfg_in;
  if (cfg.out_feat <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: NewConfigured requires OutFeat > 0");
  }
  if (auto st = validate(&cfg); !st) return st;

  const int child_n = cfg.out_feat * cfg.dim;
  Layer layer;
  layer.cfg = cfg;
  layer.branches.resize(static_cast<std::size_t>(cfg.branches));
  for (int i = 0; i < cfg.branches; ++i) {
    auto& br = layer.branches[static_cast<std::size_t>(i)];
    br.kind = BranchKind::Dense;
    const float* init = nullptr;
    if (packed_or_null) init = packed_or_null + i * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.out_feat, Activation::Linear, dt, init,
                                   &br.dense_op);
        !st) {
      return Status::Err(st.code, std::string("parallel branch ") + std::to_string(i) + ": " +
                                      st.message);
    }
  }
  if (cfg.combine == Combine::Filter) {
    dense::Layer gate;
    const float* ginit = nullptr;
    if (packed_or_null) ginit = packed_or_null + cfg.branches * child_n;
    if (auto st = dense::new_layer(cfg.dim, cfg.branches, Activation::Linear, dt, ginit, &gate);
        !st) {
      return Status::Err(st.code, std::string("parallel gate: ") + st.message);
    }
    layer.gate = std::move(gate);
  }

  layer.meta.type = LayerType::Parallel;
  layer.meta.dtype = dt;
  layer.meta.activation = Activation::Linear;
  layer.meta.input_height = cfg.dim;
  layer.meta.output_height = out_dim(cfg);
  *out = std::move(layer);
  return Status::Ok();
}

Status new_from_branches(const Config& cfg_in, std::vector<Branch> branches,
                         std::optional<dense::Layer> gate, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "parallel: nil out");
  if (branches.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: NewFromBranches needs ≥1 branch");
  }
  Config cfg = cfg_in;
  if (cfg.branches == 0) cfg.branches = static_cast<int>(branches.size());
  if (cfg.branches != static_cast<int>(branches.size())) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: cfg.Branches != len(branches)");
  }
  if (cfg.combine == Combine::Filter && !gate) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: filter mode requires Gate");
  }
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  layer.branches = std::move(branches);
  layer.gate = std::move(gate);
  layer.meta.type = LayerType::Parallel;
  layer.meta.dtype = DType::Float32;
  layer.meta.activation = Activation::Linear;
  layer.meta.input_height = cfg.dim;
  layer.meta.output_height = out_dim(cfg);
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  if (layer.branches.empty()) {
    *pre = input;
    *post = input;
    return Status::Ok();
  }

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;
  TensorF32 flat_storage;
  const TensorF32* flat = nullptr;
  if (layer_needs_flat(layer)) {
    if (lay.poly_shape) {
      return Status::Err(StatusCode::InvalidArgument,
                         "parallel: Dense/gate need standard […,Dim] layout");
    }
    flat_storage = flatten(input, lay);
    flat = &flat_storage;
  }
  const int n = lay.batch * lay.rows;
  const int nb = static_cast<int>(layer.branches.size());

  std::vector<TensorF32> branch_out(static_cast<std::size_t>(nb));
  std::vector<int> feats(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    auto& ch = layer.branches[static_cast<std::size_t>(i)];
    TensorF32 p, o;
    if (auto st = branch_forward(ch, input, flat, &p, &o); !st) {
      return Status::Err(st.code, std::string("parallel fwd branch ") + std::to_string(i) + ": " +
                                      st.message);
    }
    if (static_cast<int>(o.data.size()) % n != 0) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: branch out not divisible by n");
    }
    feats[static_cast<std::size_t>(i)] = static_cast<int>(o.data.size()) / n;
    branch_out[static_cast<std::size_t>(i)] = std::move(o);
  }

  int od = 0;
  if (cfg.combine == Combine::Concat) {
    od = std::accumulate(feats.begin(), feats.end(), 0);
  } else {
    od = feats[0];
    for (int i = 1; i < nb; ++i) {
      if (feats[static_cast<std::size_t>(i)] != od) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: unequal branch widths");
      }
    }
  }
  if (cfg.out_feat > 0) {
    const int expected = out_dim(cfg);
    if (expected > 0 && expected != od) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: measured out != cfg.OutDim");
    }
  } else {
    layer.cfg.out_feat = feats[0];
    if (cfg.combine == Combine::Concat) {
      bool equal = true;
      for (int i = 1; i < nb; ++i) {
        if (feats[static_cast<std::size_t>(i)] != feats[0]) {
          equal = false;
          break;
        }
      }
      if (!equal) layer.cfg.out_feat = 0;
    }
    layer.meta.output_height = od;
  }

  TensorF32 combined;
  combined.shape = {n, od};
  combined.data.assign(static_cast<std::size_t>(n * od), 0.0f);

  switch (cfg.combine) {
    case Combine::Add:
    case Combine::Avg:
    case Combine::Max:
    case Combine::SparseK:
    case Combine::Disagree: {
      TensorF32 merged;
      if (auto st = combine_equal_width(cfg, branch_out, &merged); !st) return st;
      combined.data = std::move(merged.data);
      break;
    }
    case Combine::Filter: {
      if (!layer.gate || !flat) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: filter mode requires Gate");
      }
      TensorF32 gp, go;
      if (auto st = dense::forward_f32(*layer.gate, *flat, &gp, &go); !st) {
        return Status::Err(st.code, std::string("parallel gate fwd: ") + st.message);
      }
      const int feat = feats[0];
      for (int r = 0; r < n; ++r) {
        std::vector<double> logits(static_cast<std::size_t>(nb));
        for (int i = 0; i < nb; ++i) {
          logits[static_cast<std::size_t>(i)] =
              static_cast<double>(go.data[static_cast<std::size_t>(r * nb + i)]);
        }
        const auto w = softmax_f64(logits);
        const int base = r * feat;
        for (int i = 0; i < nb; ++i) {
          const double wi = w[static_cast<std::size_t>(i)];
          const auto& o = branch_out[static_cast<std::size_t>(i)];
          for (int j = 0; j < feat; ++j) {
            combined.data[static_cast<std::size_t>(base + j)] += static_cast<float>(
                static_cast<double>(o.data[static_cast<std::size_t>(base + j)]) * wi);
          }
        }
      }
      break;
    }
    case Combine::Concat:
    default:
      for (int r = 0; r < n; ++r) {
        int off = r * od;
        for (int i = 0; i < nb; ++i) {
          const int f = feats[static_cast<std::size_t>(i)];
          const int src = r * f;
          const auto& o = branch_out[static_cast<std::size_t>(i)];
          for (int j = 0; j < f; ++j) {
            combined.data[static_cast<std::size_t>(off + j)] =
                o.data[static_cast<std::size_t>(src + j)];
          }
          off += f;
        }
      }
      break;
  }

  *pre = input;
  *post = unflatten(combined, lay, od);
  return Status::Ok();
}

int grad_w_size(const Layer& layer) {
  int n = 0;
  for (const auto& br : layer.branches) {
    n += branch_grad_w_size(br);
  }
  if (layer.gate) {
    n += layer.gate->input_height * layer.gate->output_height;
  }
  return n;
}

namespace {

std::vector<int> sparse_k_selected(const std::vector<TensorF32>& branch_out, int k) {
  const int nb = static_cast<int>(branch_out.size());
  if (k > nb) k = nb;
  if (k < 1) k = 1;
  struct Scored {
    int i;
    double nrm;
  };
  std::vector<Scored> scores(static_cast<std::size_t>(nb));
  for (int i = 0; i < nb; ++i) {
    double s = 0.0;
    for (float v : branch_out[static_cast<std::size_t>(i)].data) {
      const double f = static_cast<double>(v);
      s += f * f;
    }
    scores[static_cast<std::size_t>(i)] = {i, std::sqrt(s)};
  }
  std::sort(scores.begin(), scores.end(), [](const Scored& a, const Scored& b) {
    if (a.nrm != b.nrm) return a.nrm > b.nrm;
    return a.i < b.i;
  });
  std::vector<int> out(static_cast<std::size_t>(k));
  for (int t = 0; t < k; ++t) {
    out[static_cast<std::size_t>(t)] = scores[static_cast<std::size_t>(t)].i;
  }
  return out;
}

} // namespace

Status split_combine_grad(const Layer& layer, const TensorF32& grad_out,
                          const std::vector<TensorF32>& branch_outs,
                          std::vector<TensorF32>* branch_grads) {
  if (!branch_grads) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: nil branch_grads");
  }
  const int nb = static_cast<int>(branch_outs.size());
  branch_grads->assign(static_cast<std::size_t>(nb), TensorF32{});
  if (grad_out.data.empty()) {
    for (int i = 0; i < nb; ++i) {
      (*branch_grads)[static_cast<std::size_t>(i)].shape = branch_outs[static_cast<std::size_t>(i)].shape;
      (*branch_grads)[static_cast<std::size_t>(i)].data.assign(
          branch_outs[static_cast<std::size_t>(i)].data.size(), 0.0f);
    }
    return Status::Ok();
  }
  switch (layer.cfg.combine) {
    case Combine::Add:
      for (int i = 0; i < nb; ++i) {
        (*branch_grads)[static_cast<std::size_t>(i)] = grad_out;
      }
      return Status::Ok();
    case Combine::Avg: {
      const float inv = static_cast<float>(1.0 / static_cast<double>(nb));
      for (int i = 0; i < nb; ++i) {
        TensorF32 g = grad_out;
        for (float& v : g.data) v *= inv;
        (*branch_grads)[static_cast<std::size_t>(i)] = std::move(g);
      }
      return Status::Ok();
    }
    case Combine::Max: {
      for (int i = 0; i < nb; ++i) {
        (*branch_grads)[static_cast<std::size_t>(i)].shape = grad_out.shape;
        (*branch_grads)[static_cast<std::size_t>(i)].data.assign(grad_out.data.size(), 0.0f);
      }
      const int n = static_cast<int>(grad_out.data.size());
      for (int j = 0; j < n; ++j) {
        int best_i = 0;
        double best = static_cast<double>(branch_outs[0].data[static_cast<std::size_t>(j)]);
        for (int i = 1; i < nb; ++i) {
          const double v =
              static_cast<double>(branch_outs[static_cast<std::size_t>(i)].data[static_cast<std::size_t>(j)]);
          if (v > best) {
            best = v;
            best_i = i;
          }
        }
        (*branch_grads)[static_cast<std::size_t>(best_i)].data[static_cast<std::size_t>(j)] =
            grad_out.data[static_cast<std::size_t>(j)];
      }
      return Status::Ok();
    }
    case Combine::SparseK: {
      const auto sel = sparse_k_selected(branch_outs, sparse_k_eff(layer.cfg));
      const float inv = static_cast<float>(1.0 / static_cast<double>(sel.size()));
      for (int i = 0; i < nb; ++i) {
        (*branch_grads)[static_cast<std::size_t>(i)].shape = grad_out.shape;
        (*branch_grads)[static_cast<std::size_t>(i)].data.assign(grad_out.data.size(), 0.0f);
      }
      for (int i : sel) {
        for (std::size_t j = 0; j < grad_out.data.size(); ++j) {
          (*branch_grads)[static_cast<std::size_t>(i)].data[j] = grad_out.data[j] * inv;
        }
      }
      return Status::Ok();
    }
    case Combine::Disagree: {
      const double beta = disagree_beta_eff(layer.cfg);
      const double inv = 1.0 / static_cast<double>(nb);
      for (int i = 0; i < nb; ++i) {
        (*branch_grads)[static_cast<std::size_t>(i)].shape = grad_out.shape;
        (*branch_grads)[static_cast<std::size_t>(i)].data.assign(grad_out.data.size(), 0.0f);
      }
      if (nb == 2) {
        const float ca = static_cast<float>(inv + beta);
        const float cb = static_cast<float>(inv - beta);
        for (std::size_t j = 0; j < grad_out.data.size(); ++j) {
          (*branch_grads)[0].data[j] = grad_out.data[j] * ca;
          (*branch_grads)[1].data[j] = grad_out.data[j] * cb;
        }
      } else {
        const float c0 = static_cast<float>(beta + (1.0 - beta) * inv);
        const float co = static_cast<float>((1.0 - beta) * inv);
        for (std::size_t j = 0; j < grad_out.data.size(); ++j) {
          (*branch_grads)[0].data[j] = grad_out.data[j] * c0;
          for (int i = 1; i < nb; ++i) {
            (*branch_grads)[static_cast<std::size_t>(i)].data[j] = grad_out.data[j] * co;
          }
        }
      }
      return Status::Ok();
    }
    case Combine::Concat: {
      if (grad_out.shape.size() < 2) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: concat grad needs rank≥2");
      }
      const int batch = static_cast<int>(grad_out.shape[0]);
      const int total_feat = static_cast<int>(grad_out.shape[1]);
      int off = 0;
      for (int i = 0; i < nb; ++i) {
        const auto& bo = branch_outs[static_cast<std::size_t>(i)];
        if (bo.shape.empty()) {
          return Status::Err(StatusCode::InvalidArgument, "parallel: concat empty branch out");
        }
        const int feat = static_cast<int>(bo.shape.back());
        TensorF32 g;
        g.shape = {batch, feat};
        g.data.assign(static_cast<std::size_t>(batch * feat), 0.0f);
        for (int b = 0; b < batch; ++b) {
          const int src = b * total_feat + off;
          const int dst = b * feat;
          for (int j = 0; j < feat; ++j) {
            g.data[static_cast<std::size_t>(dst + j)] =
                grad_out.data[static_cast<std::size_t>(src + j)];
          }
        }
        (*branch_grads)[static_cast<std::size_t>(i)] = std::move(g);
        off += feat;
      }
      return Status::Ok();
    }
    default: // Filter / unknown — treat like Add (Go splitCombineGrad)
      for (int i = 0; i < nb; ++i) {
        (*branch_grads)[static_cast<std::size_t>(i)] = grad_out;
      }
      return Status::Ok();
  }
}

Status approx_gate_grad(const Layer& layer, const TensorF32& grad_out, TensorF32* gate_gy) {
  if (!gate_gy || grad_out.data.empty() || grad_out.shape.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "cameral gate gap shape");
  }
  const int nb = static_cast<int>(layer.branches.size());
  gate_gy->shape = {grad_out.shape[0], static_cast<std::int64_t>(nb)};
  gate_gy->data.resize(static_cast<std::size_t>(gate_gy->shape[0]) * static_cast<std::size_t>(nb));
  return cuda::launch_cam_gate_gap(grad_out.data.data(), static_cast<std::int32_t>(grad_out.data.size()),
                                   gate_gy->data.data(), static_cast<std::int32_t>(gate_gy->data.size()));
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& /*pre*/, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: nil grad buffers");
  }
  const int nb = static_cast<int>(layer.branches.size());
  if (nb == 0) {
    *grad_in = input;
    for (float& v : grad_in->data) v = 0.0f;
    if (!grad_out.data.empty() && grad_out.data.size() == grad_in->data.size()) {
      grad_in->data = grad_out.data;
    }
    grad_w->shape = {0};
    grad_w->data.clear();
    return Status::Ok();
  }

  Layout lay;
  if (auto st = parse_layout(layer.cfg, input, &lay); !st) return st;
  TensorF32 flat_storage;
  const TensorF32* flat = nullptr;
  if (layer_needs_flat(layer)) {
    if (lay.poly_shape) {
      return Status::Err(StatusCode::InvalidArgument,
                         "parallel: Dense/gate need standard […,Dim] layout");
    }
    flat_storage = flatten(input, lay);
    flat = &flat_storage;
  }

  std::vector<TensorF32> branch_pres(static_cast<std::size_t>(nb));
  std::vector<TensorF32> branch_outs(static_cast<std::size_t>(nb));
  std::vector<int> feats(static_cast<std::size_t>(nb));
  const int n = lay.batch * lay.rows;
  for (int i = 0; i < nb; ++i) {
    if (auto st = branch_forward(layer.branches[static_cast<std::size_t>(i)], input, flat,
                                 &branch_pres[static_cast<std::size_t>(i)],
                                 &branch_outs[static_cast<std::size_t>(i)]);
        !st) {
      return Status::Err(st.code, std::string("parallel bwd recompute: ") + st.message);
    }
    if (static_cast<int>(branch_outs[static_cast<std::size_t>(i)].data.size()) % n != 0) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: branch out not divisible by n");
    }
    feats[static_cast<std::size_t>(i)] =
        static_cast<int>(branch_outs[static_cast<std::size_t>(i)].data.size()) / n;
  }

  grad_in->shape = input.shape;
  grad_in->data.assign(input.data.size(), 0.0f);
  // Match Go: accumulate ∂L/∂x in float64, cast once (f32+= loses Parallel multi-branch bits).
  std::vector<double> gin_acc(input.data.size(), 0.0);
  grad_w->data.clear();
  grad_w->data.reserve(static_cast<std::size_t>(grad_w_size(layer)));

  auto accum_in = [&](const TensorF32& gin) {
    const std::size_t lim = std::min(gin_acc.size(), gin.data.size());
    for (std::size_t j = 0; j < lim; ++j) {
      gin_acc[j] += static_cast<double>(gin.data[j]);
    }
  };

  auto bwd_one = [&](int i, const TensorF32& branch_gy) -> Status {
    TensorF32 gy = branch_gy;
    const auto& bout = branch_outs[static_cast<std::size_t>(i)];
    if (gy.data.size() == bout.data.size() && gy.shape != bout.shape && !bout.shape.empty()) {
      gy.shape = bout.shape;
    }
    TensorF32 gin, dW;
    if (auto st = branch_backward(layer.branches[static_cast<std::size_t>(i)], gy, input, flat,
                                  branch_pres[static_cast<std::size_t>(i)], &gin, &dW);
        !st) {
      return Status::Err(st.code, std::string("parallel bwd branch: ") + st.message);
    }
    accum_in(gin);
    grad_w->data.insert(grad_w->data.end(), dW.data.begin(), dW.data.end());
    return Status::Ok();
  };

  // Flatten grad to [n, outDim] for combine routing (matches Go flattenFeat).
  int od = 0;
  if (layer.cfg.combine == Combine::Concat) {
    od = std::accumulate(feats.begin(), feats.end(), 0);
  } else {
    od = feats[0];
  }
  TensorF32 gy_flat;
  gy_flat.shape = {n, od};
  gy_flat.data.assign(static_cast<std::size_t>(n * od), 0.0f);
  if (static_cast<int>(grad_out.data.size()) >= n * od) {
    gy_flat.data.assign(grad_out.data.begin(),
                        grad_out.data.begin() + static_cast<std::ptrdiff_t>(n * od));
  } else {
    gy_flat.data = grad_out.data;
    gy_flat.data.resize(static_cast<std::size_t>(n * od), 0.0f);
  }

  switch (layer.cfg.combine) {
    case Combine::Add:
    case Combine::Avg: {
      // Match Go cpu.go: scale gy in f64 then cast (not f32 inv via split_combine_grad).
      const double scale =
          layer.cfg.combine == Combine::Avg ? (1.0 / static_cast<double>(nb)) : 1.0;
      const int feat = feats[0];
      TensorF32 scaled;
      scaled.shape = {n, feat};
      scaled.data.resize(static_cast<std::size_t>(n * feat));
      for (std::size_t j = 0; j < scaled.data.size(); ++j) {
        scaled.data[j] =
            static_cast<float>(static_cast<double>(gy_flat.data[j]) * scale);
      }
      for (int i = 0; i < nb; ++i) {
        if (auto st = bwd_one(i, scaled); !st) return st;
      }
      break;
    }
    case Combine::Max:
    case Combine::SparseK:
    case Combine::Disagree: {
      // Rebuild equal-width branch outs as [n, feat] views for Max/SparseK/Disagree.
      std::vector<TensorF32> flat_outs(static_cast<std::size_t>(nb));
      for (int i = 0; i < nb; ++i) {
        flat_outs[static_cast<std::size_t>(i)].shape = {n, feats[static_cast<std::size_t>(i)]};
        flat_outs[static_cast<std::size_t>(i)].data = branch_outs[static_cast<std::size_t>(i)].data;
      }
      std::vector<TensorF32> branch_gys;
      if (auto st = split_combine_grad(layer, gy_flat, flat_outs, &branch_gys); !st) return st;
      for (int i = 0; i < nb; ++i) {
        if (auto st = bwd_one(i, branch_gys[static_cast<std::size_t>(i)]); !st) return st;
      }
      break;
    }
    case Combine::Filter: {
      if (!layer.gate || !flat) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: filter bwd requires Gate");
      }
      TensorF32 gate_pre, gate_out;
      if (auto st = dense::forward_f32(*layer.gate, *flat, &gate_pre, &gate_out); !st) return st;
      const int feat = feats[0];
      TensorF32 gate_logits_grad;
      gate_logits_grad.shape = {n, nb};
      gate_logits_grad.data.assign(static_cast<std::size_t>(n * nb), 0.0f);
      for (int r = 0; r < n; ++r) {
        std::vector<double> logits(static_cast<std::size_t>(nb));
        for (int i = 0; i < nb; ++i) {
          logits[static_cast<std::size_t>(i)] =
              static_cast<double>(gate_out.data[static_cast<std::size_t>(r * nb + i)]);
        }
        const auto w = softmax_f64(logits);
        const int base = r * feat;
        std::vector<double> gy(static_cast<std::size_t>(nb), 0.0);
        for (int i = 0; i < nb; ++i) {
          double dot = 0.0;
          const auto& o = branch_outs[static_cast<std::size_t>(i)];
          for (int j = 0; j < feat; ++j) {
            dot += static_cast<double>(gy_flat.data[static_cast<std::size_t>(base + j)]) *
                   static_cast<double>(o.data[static_cast<std::size_t>(base + j)]);
          }
          gy[static_cast<std::size_t>(i)] = dot;
        }
        // Softmax backward: g_logits_i = w_i * (gy_i - sum_k w_k gy_k)
        double sum = 0.0;
        for (int i = 0; i < nb; ++i) {
          sum += w[static_cast<std::size_t>(i)] * gy[static_cast<std::size_t>(i)];
        }
        for (int i = 0; i < nb; ++i) {
          gate_logits_grad.data[static_cast<std::size_t>(r * nb + i)] = static_cast<float>(
              w[static_cast<std::size_t>(i)] * (gy[static_cast<std::size_t>(i)] - sum));
        }
      }
      for (int i = 0; i < nb; ++i) {
        TensorF32 scaled;
        scaled.shape = {n, feat};
        scaled.data.assign(static_cast<std::size_t>(n * feat), 0.0f);
        for (int r = 0; r < n; ++r) {
          std::vector<double> logits(static_cast<std::size_t>(nb));
          for (int k = 0; k < nb; ++k) {
            logits[static_cast<std::size_t>(k)] =
                static_cast<double>(gate_out.data[static_cast<std::size_t>(r * nb + k)]);
          }
          const auto w = softmax_f64(logits);
          const int base = r * feat;
          const double wi = w[static_cast<std::size_t>(i)];
          for (int j = 0; j < feat; ++j) {
            scaled.data[static_cast<std::size_t>(base + j)] = static_cast<float>(
                static_cast<double>(gy_flat.data[static_cast<std::size_t>(base + j)]) * wi);
          }
        }
        if (auto st = bwd_one(i, scaled); !st) return st;
      }
      TensorF32 gx_g, dw_g;
      if (auto st = dense::backward_f32(*layer.gate, gate_logits_grad, *flat, gate_pre, &gx_g, &dw_g);
          !st) {
        return Status::Err(st.code, std::string("parallel gate bwd: ") + st.message);
      }
      accum_in(gx_g);
      grad_w->data.insert(grad_w->data.end(), dw_g.data.begin(), dw_g.data.end());
      break;
    }
    case Combine::Concat:
    default: {
      int offset = 0;
      for (int i = 0; i < nb; ++i) {
        const int f = feats[static_cast<std::size_t>(i)];
        TensorF32 branch_gy;
        branch_gy.shape = {n, f};
        branch_gy.data.assign(static_cast<std::size_t>(n * f), 0.0f);
        for (int r = 0; r < n; ++r) {
          const int src = r * od + offset;
          const int dst = r * f;
          for (int j = 0; j < f; ++j) {
            branch_gy.data[static_cast<std::size_t>(dst + j)] =
                gy_flat.data[static_cast<std::size_t>(src + j)];
          }
        }
        offset += f;
        if (auto st = bwd_one(i, branch_gy); !st) return st;
      }
      break;
    }
  }
  for (std::size_t j = 0; j < gin_acc.size(); ++j) {
    grad_in->data[j] = static_cast<float>(gin_acc[j]);
  }
  grad_w->shape = {static_cast<int64_t>(grad_w->data.size())};
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer || dW.data.empty() || lr == 0.0) {
    return Status::Ok();
  }
  int off = 0;
  for (std::size_t i = 0; i < layer->branches.size(); ++i) {
    auto& br = layer->branches[i];
    const int n = branch_grad_w_size(br);
    if (n <= 0) continue;
    if (off + n > static_cast<int>(dW.data.size())) {
      return Status::Err(StatusCode::InvalidArgument, "parallel: dW short at branch");
    }
    TensorF32 slice;
    slice.shape = {n};
    slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + n);
    if (auto st = branch_apply_grad_sgd(&br, slice, lr); !st) {
      return Status::Err(st.code, std::string("parallel branch SGD: ") + st.message);
    }
    off += n;
  }
  if (layer->gate) {
    const int n = layer->gate->input_height * layer->gate->output_height;
    if (n > 0) {
      if (off + n > static_cast<int>(dW.data.size())) {
        return Status::Err(StatusCode::InvalidArgument, "parallel: dW short at gate");
      }
      TensorF32 slice;
      slice.shape = {n};
      slice.data.assign(dW.data.begin() + off, dW.data.begin() + off + n);
      if (auto st = dense::apply_grad_sgd(&(*layer->gate), slice, lr); !st) {
        return st;
      }
    }
  }
  return Status::Ok();
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "parallel: nil export dst");
  }
  int off = 0;
  for (const auto& br : layer.branches) {
    const int n = branch_grad_w_size(br);
    if (n > 0) {
      if (auto st = branch_export_f32(br, dst + off); !st) return st;
    }
    off += n;
  }
  if (layer.gate) {
    if (auto st = layer.gate->weights.export_f32(dst + off); !st) return st;
  }
  return Status::Ok();
}

} // namespace welvet::parallel
