// WDB1 / WVN1 — Dense NormalBP leaf + volumetric (Dense-only Grid path).

#include "volume_fwd.hpp"
#include "welvet/systems/tween.hpp"
#include "welvet/runtime/tween_train.hpp"

#include "welvet/architecture.hpp"
#include "welvet/cnn1.hpp"
#include "welvet/cnn2.hpp"
#include "welvet/cnn3.hpp"
#include "welvet/convt1.hpp"
#include "welvet/convt2.hpp"
#include "welvet/convt3.hpp"
#include "welvet/lstm.hpp"
#include "welvet/residual.hpp"
#include "welvet/rnn.hpp"
#include "welvet/sequential.hpp"
#include "welvet/stack.hpp"
#include "welvet/interop/child_wire.hpp"
#include "welvet/parallel.hpp"
#include "welvet/kmeans.hpp"
#include "welvet/gdn.hpp"
#include "welvet/metacognition.hpp"
#include "welvet/mamba.hpp"

#include "welvet/dense.hpp"
#include "welvet/embedding.hpp"
#include "welvet/mha.hpp"
#include "welvet/runtime/backward.hpp"
#include "welvet/runtime/forward.hpp"
#include "welvet/runtime/training.hpp"
#include "welvet/swiglu.hpp"
#include "welvet_c.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool read_exact(void* p, size_t n) { return std::fread(p, 1, n, stdin) == n; }
template <typename T>
bool read_exact(T* p, size_t n) {
  return read_exact(static_cast<void*>(p), n * sizeof(T));
}
bool write_exact(const void* p, size_t n) { return std::fwrite(p, 1, n, stdout) == n; }
template <typename T>
bool write_exact(const T* p, size_t n) {
  return write_exact(static_cast<const void*>(p), n * sizeof(T));
}

} // namespace

int run_dense_normalbp(int bench_iters) {
  int32_t batch = 0, in = 0, out = 0, act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WDB1 header\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(out * in));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len > 0 && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * out));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(batch * out));
  std::vector<float> post(static_cast<std::size_t>(batch * out));
  std::vector<float> dX(static_cast<std::size_t>(batch * in));
  std::vector<float> dW(static_cast<std::size_t>(out * in));
  std::vector<float> W_after(static_cast<std::size_t>(out * in));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_dense_normalbp_f32(batch, in, out, act, dtype, backend, W.data(),
                                     bias.empty() ? nullptr : bias.data(), bias.size(), x.data(),
                                     gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                     dW.data(), W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WDBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

int run_volume_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WVN1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, out = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t in = 0, out = 0, act = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&out, 1) || !read_exact(&act, 1) ||
        !read_exact(&dtype, 1)) {
      return 2;
    }
    c.in = in;
    c.out = out;
    c.act = act;
    c.dtype = dtype;
    c.W.resize(static_cast<std::size_t>(out * in));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len > 0 && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in < 0) first_in = in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_out = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_out = it->out;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_out};
  target.data.resize(static_cast<std::size_t>(batch * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::dense::Layer layer;
      auto st = welvet::dense::new_layer(c.in, c.out, static_cast<welvet::Activation>(c.act),
                                         static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) return st;
      if (!c.bias.empty()) layer.weights.bias = c.bias;
      st = welvet::architecture::bind_dense(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  const bool tween = mode == VolumeTrainMode::Tween;
  if (!write_exact(tween ? "TWVO" : "WVNO", 4) || !write_exact(&loss, 1)) return 2;
  if (!tween) {
    const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
    if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
      return 2;
    }
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t o = c.out, i = c.in;
    std::vector<float> Wa(static_cast<std::size_t>(o * i));
    auto est = cell->dense_op.weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&o, 1) || !write_exact(&i, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  if (!tween) {
    const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
    if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
      return 2;
    }
  }
  return 0;
}

#include "welvet/layernorm.hpp"
#include "welvet/rmsnorm.hpp"
#include "welvet/softmax.hpp"

int run_rmsnorm_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) || !read_exact(&eps, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRB1 header\n");
    return 2;
  }
  (void)backend;
  std::vector<float> gamma(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(x.size()), post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(dim)),
      gamma_after(static_cast<std::size_t>(dim));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_rmsnorm_normalbp_f32(batch, seq, dim, eps, dtype, backend, gamma.data(), x.data(),
                                       gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                       dW.data(), gamma_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t g_n = static_cast<int32_t>(gamma_after.size());
  if (!write_exact("WRBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&g_n, 1) ||
      !write_exact(gamma_after.data(), gamma_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

int run_volume_rmsnorm_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WRV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0;
    int dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!read_exact(&dim, 1) || !read_exact(&dtype, 1) || !read_exact(&eps, 1)) return 2;
    c.dim = dim;
    c.dtype = dtype;
    c.eps = eps;
    c.gamma.resize(static_cast<std::size_t>(dim));
    if (!read_exact(c.gamma.data(), c.gamma.size())) return 2;
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::rmsnorm::Config cfg;
      cfg.dim = c.dim;
      cfg.eps = c.eps;
      welvet::rmsnorm::Layer layer;
      auto st = welvet::rmsnorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                c.gamma.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_rmsnorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_exact(mode == VolumeTrainMode::Tween ? "TRVO" : "WRVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t d = c.dim;
    std::vector<float> ga(static_cast<std::size_t>(d));
    auto est = cell->rmsnorm_op.gamma.export_f32(ga.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&d, 1) || !write_exact(ga.data(), ga.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

int run_layernorm_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  double eps = 0.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) || !read_exact(&eps, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLB1 header\n");
    return 2;
  }
  (void)backend;
  std::vector<float> gamma(static_cast<std::size_t>(dim));
  std::vector<float> beta(static_cast<std::size_t>(dim));
  if (!read_exact(gamma.data(), gamma.size()) || !read_exact(beta.data(), beta.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(x.size()), post(x.size()), dX(x.size()),
      dW(static_cast<std::size_t>(2 * dim)), gamma_after(static_cast<std::size_t>(dim)),
      beta_after(static_cast<std::size_t>(dim));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_layernorm_normalbp_f32(batch, seq, dim, eps, dtype, backend, gamma.data(),
                                         beta.data(), x.data(), gyt.data(), use_target, lr,
                                         pre.data(), post.data(), dX.data(), dW.data(),
                                         gamma_after.data(), beta_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t g_n = static_cast<int32_t>(gamma_after.size());
  const int32_t b_n = static_cast<int32_t>(beta_after.size());
  if (!write_exact("WLBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&g_n, 1) ||
      !write_exact(gamma_after.data(), gamma_after.size()) || !write_exact(&b_n, 1) ||
      !write_exact(beta_after.data(), beta_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

int run_volume_layernorm_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WLV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0;
    int dtype = 1;
    double eps = 0.0;
    welvet::architecture::Coord coord{};
    std::vector<float> gamma;
    std::vector<float> beta;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t dim = 0, dtype = 1;
    double eps = 0.0;
    if (!read_exact(&dim, 1) || !read_exact(&dtype, 1) || !read_exact(&eps, 1)) return 2;
    c.dim = dim;
    c.dtype = dtype;
    c.eps = eps;
    c.gamma.resize(static_cast<std::size_t>(dim));
    c.beta.resize(static_cast<std::size_t>(dim));
    if (!read_exact(c.gamma.data(), c.gamma.size()) || !read_exact(c.beta.data(), c.beta.size())) {
      return 2;
    }
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::layernorm::Config cfg;
      cfg.dim = c.dim;
      cfg.eps = c.eps;
      welvet::layernorm::Layer layer;
      auto st = welvet::layernorm::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                  c.gamma.data(), c.beta.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_layernorm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_exact(mode == VolumeTrainMode::Tween ? "TLVO" : "WLVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t d = c.dim;
    std::vector<float> ga(static_cast<std::size_t>(d));
    std::vector<float> ba(static_cast<std::size_t>(d));
    auto eg = cell->layernorm_op.gamma.export_f32(ga.data());
    if (!eg) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", eg.message.c_str());
      return 1;
    }
    auto eb = cell->layernorm_op.beta.export_f32(ba.data());
    if (!eb) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", eb.message.c_str());
      return 1;
    }
    if (!write_exact(&d, 1) || !write_exact(ga.data(), ga.size()) ||
        !write_exact(ba.data(), ba.size())) {
      return 2;
    }
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

int run_softmax_normalbp(int bench_iters) {
  int32_t n = 0, dim = 0, seq = 0, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1,
          backend = 0, rank = 0, mask_len = 0, hier_len = 0;
  double temperature = 1.0, entmax_alpha = 0.0;
  if (!read_exact(&n, 1) || !read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
      !read_exact(&rows, 1) || !read_exact(&cols, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1) || !read_exact(&rank, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WXB1 header\n");
    return 2;
  }
  if (rank < 1 || rank > 8) {
    std::fprintf(stderr, "welvet_dense_fwd: bad softmax nbp rank\n");
    return 2;
  }
  std::vector<int32_t> shape(static_cast<std::size_t>(rank));
  if (!read_exact(shape.data(), shape.size())) return 2;
  if (!read_exact(&mask_len, 1)) return 2;
  std::vector<uint8_t> mask(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
  if (mask_len > 0 && !read_exact(mask.data(), mask.size())) return 2;
  if (!read_exact(&hier_len, 1)) return 2;
  std::vector<int32_t> hier(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
  if (hier_len > 0 && !read_exact(hier.data(), hier.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(n));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(n));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(n));
  std::vector<float> post(pre.size()), dX(pre.size());
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_softmax_normalbp_f32(
        n, dim, seq, kind, rows, cols, temperature, entmax_alpha, gumbel_seed, dtype, backend, rank,
        shape.data(), mask_len > 0 ? mask.data() : nullptr, mask_len,
        hier_len > 0 ? hier.data() : nullptr, hier_len, x.data(), gyt.data(), use_target, lr,
        pre.data(), post.data(), dX.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = 0;
  const int32_t w_n = 0;
  if (!write_exact("WXBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(&w_n, 1) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

int run_volume_softmax_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WXV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq_len = 1, kind = 0, rows = 0, cols = 0, gumbel_seed = 0, dtype = 1;
    double temperature = 1.0, entmax_alpha = 0.0;
    std::vector<uint8_t> mask;
    std::vector<int32_t> hier;
    welvet::architecture::Coord coord{};
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t dim = 0, seq = 0, kind = 0, crow = 0, ccol = 0, gumbel_seed = 0, dtype = 1;
    double temperature = 1.0, entmax_alpha = 0.0;
    int32_t mask_len = 0, hier_len = 0;
    if (!read_exact(&dim, 1) || !read_exact(&seq, 1) || !read_exact(&kind, 1) ||
        !read_exact(&crow, 1) || !read_exact(&ccol, 1) || !read_exact(&temperature, 1) ||
        !read_exact(&entmax_alpha, 1) || !read_exact(&gumbel_seed, 1) || !read_exact(&dtype, 1) ||
        !read_exact(&mask_len, 1)) {
      return 2;
    }
    c.dim = dim;
    c.seq_len = seq;
    c.kind = kind;
    c.rows = crow;
    c.cols = ccol;
    c.temperature = temperature;
    c.entmax_alpha = entmax_alpha;
    c.gumbel_seed = gumbel_seed;
    c.dtype = dtype;
    c.mask.resize(static_cast<std::size_t>(mask_len > 0 ? mask_len : 0));
    if (mask_len > 0 && !read_exact(c.mask.data(), c.mask.size())) return 2;
    if (!read_exact(&hier_len, 1)) return 2;
    c.hier.resize(static_cast<std::size_t>(hier_len > 0 ? hier_len : 0));
    if (hier_len > 0 && !read_exact(c.hier.data(), c.hier.size())) return 2;
    if (first_dim < 0) first_dim = dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_dim = it->dim;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::softmax::Config cfg;
      cfg.dim = c.dim;
      cfg.seq_len = c.seq_len <= 0 ? 1 : c.seq_len;
      cfg.rows = c.rows;
      cfg.cols = c.cols;
      cfg.temperature = c.temperature;
      cfg.kind = static_cast<welvet::softmax::Kind>(c.kind);
      cfg.entmax_alpha = c.entmax_alpha;
      cfg.gumbel_seed = static_cast<uint32_t>(c.gumbel_seed);
      if (!c.mask.empty()) {
        cfg.mask.resize(c.mask.size());
        for (std::size_t i = 0; i < c.mask.size(); ++i) {
          cfg.mask[i] = c.mask[i] != 0;
        }
      }
      if (!c.hier.empty()) {
        cfg.hierarchy_levels.assign(c.hier.begin(), c.hier.end());
      }
      welvet::softmax::Layer layer;
      auto st = welvet::softmax::new_configured(cfg, static_cast<welvet::DType>(c.dtype), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_softmax(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                              std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_exact(mode == VolumeTrainMode::Tween ? "TXVO" : "WXVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    const int32_t nw = 0;
    if (!write_exact(&nw, 1)) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WSB1 — SwiGLU NormalBP leaf
int run_swiglu_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, in = 0, inter = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&in, 1) || !read_exact(&inter, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSB1 header\n");
    return 2;
  }
  std::vector<float> Gate(static_cast<std::size_t>(inter * in));
  std::vector<float> Up(static_cast<std::size_t>(inter * in));
  std::vector<float> Down(static_cast<std::size_t>(in * inter));
  if (!read_exact(Gate.data(), Gate.size()) || !read_exact(Up.data(), Up.size()) ||
      !read_exact(Down.data(), Down.size())) {
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                       static_cast<std::size_t>(in));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  const int nw = 2 * inter * in + in * inter;
  std::vector<float> pre(static_cast<std::size_t>(batch) * static_cast<std::size_t>(seq) *
                         static_cast<std::size_t>(inter));
  std::vector<float> post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(nw)),
      W_after(static_cast<std::size_t>(nw));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_swiglu_normalbp_f32(batch, seq, in, inter, dtype, backend, Gate.data(), Up.data(),
                                      Down.data(), x.data(), gyt.data(), use_target, lr, pre.data(),
                                      post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WSBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WSV1 — Volume SwiGLU NormalBP
int run_volume_swiglu_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WSV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, inter = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Gate, Up, Down;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t in = 0, inter = 0, dtype = 1;
    if (!read_exact(&in, 1) || !read_exact(&inter, 1) || !read_exact(&dtype, 1)) return 2;
    c.in = in;
    c.inter = inter;
    c.dtype = dtype;
    c.Gate.resize(static_cast<std::size_t>(inter * in));
    c.Up.resize(static_cast<std::size_t>(inter * in));
    c.Down.resize(static_cast<std::size_t>(in * inter));
    if (!read_exact(c.Gate.data(), c.Gate.size()) || !read_exact(c.Up.data(), c.Up.size()) ||
        !read_exact(c.Down.data(), c.Down.size())) {
      return 2;
    }
    if (first_in < 0) first_in = in;
    cells.push_back(std::move(c));
  }
  if (first_in < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_in = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_in = it->in;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_in};
  target.data.resize(static_cast<std::size_t>(batch * final_in));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::swiglu::Config cfg;
      cfg.input_dim = c.in;
      cfg.intermediate_dim = c.inter;
      welvet::swiglu::Layer layer;
      auto st = welvet::swiglu::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                               c.Gate.data(), c.Up.data(), c.Down.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_swiglu(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_exact(mode == VolumeTrainMode::Tween ? "TSVO" : "WSVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = 2 * c.inter * c.in + c.in * c.inter;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::swiglu::export_weights_f32(cell->swiglu_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WEB1 — Embedding NormalBP leaf
int run_embedding_normalbp(int bench_iters) {
  int32_t batch = 0, vocab = 0, emb = 0, seq = 0, in_rank = 2, in_last = 1, act = -1, dtype = 1,
          backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&vocab, 1) || !read_exact(&emb, 1) ||
      !read_exact(&seq, 1) || !read_exact(&in_rank, 1) || !read_exact(&in_last, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEB1 header\n");
    return 2;
  }
  std::vector<float> W(static_cast<std::size_t>(vocab * emb));
  if (!read_exact(W.data(), W.size())) return 2;
  const int x_n = batch * seq * in_last;
  std::vector<float> x(static_cast<std::size_t>(x_n));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  const int out_n = (in_rank == 1 ? seq : batch * seq) * emb;
  std::vector<float> gyt(static_cast<std::size_t>(out_n));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(static_cast<std::size_t>(out_n));
  std::vector<float> post(pre.size()), dX(static_cast<std::size_t>(x_n)),
      dW(static_cast<std::size_t>(vocab * emb)), W_after(dW.size());
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_embedding_normalbp_f32(batch, vocab, emb, seq, in_rank, in_last, act, dtype,
                                         backend, W.data(), x.data(), gyt.data(), use_target, lr,
                                         pre.data(), post.data(), dX.data(), dW.data(),
                                         W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WEBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WEV1 — Volume Embedding NormalBP
int run_volume_embedding_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WEV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t vocab = 0, emb = 0, seq = 0, act = -1, dtype = 1;
    if (!read_exact(&vocab, 1) || !read_exact(&emb, 1) || !read_exact(&seq, 1) ||
        !read_exact(&act, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.vocab = vocab;
    c.emb = emb;
    c.seq = seq;
    c.act = act;
    c.dtype = dtype;
    c.W.resize(static_cast<std::size_t>(vocab * emb));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_seq < 0) first_seq = seq;
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_emb = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_emb = it->emb;
      final_seq = it->seq;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_emb};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_emb));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::embedding::Config cfg;
      cfg.vocab_size = c.vocab;
      cfg.embedding_dim = c.emb;
      cfg.seq_len = c.seq;
      welvet::embedding::Layer layer;
      auto st = welvet::embedding::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                  static_cast<welvet::Activation>(c.act), c.W.data(),
                                                  &layer);
      if (!st) return st;
      st = welvet::architecture::bind_embedding(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_exact(mode == VolumeTrainMode::Tween ? "TEVO" : "WEVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = c.vocab * c.emb;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = cell->embedding_op.weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// WHB1 — MHA NormalBP leaf (ModeSelf)
int run_mha_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
  int32_t mask = 0, pos = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&num_heads, 1) || !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) ||
      !read_exact(&msl, 1) || !read_exact(&mask, 1) || !read_exact(&pos, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WHB1 header\n");
    return 2;
  }
  if (num_heads <= 0 || d_model <= 0 || d_model % num_heads != 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad mha dims\n");
    return 2;
  }
  const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
  const int kv_heads = num_kv > 0 ? num_kv : num_heads;
  const int q_dim = num_heads * hd;
  const int kv_dim = kv_heads * hd;
  const int nw = q_dim * d_model + kv_dim * d_model + kv_dim * d_model + d_model * q_dim;

  std::vector<float> Q(static_cast<std::size_t>(q_dim * d_model));
  std::vector<float> K(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> V(static_cast<std::size_t>(kv_dim * d_model));
  std::vector<float> O(static_cast<std::size_t>(d_model * q_dim));
  if (!read_exact(Q.data(), Q.size()) || !read_exact(K.data(), K.size()) ||
      !read_exact(V.data(), V.size()) || !read_exact(O.data(), O.size())) {
    return 2;
  }
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(x.size());
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  (void)backend;
  std::vector<float> pre(static_cast<std::size_t>(batch * seq * q_dim));
  std::vector<float> post(x.size()), dX(x.size()), dW(static_cast<std::size_t>(nw)),
      W_after(static_cast<std::size_t>(nw));
  double loss = 0.0;

  auto once = [&]() -> welvet_status {
    return welvet_mha_normalbp_f32(batch, seq, d_model, num_heads, num_kv, head_dim, msl, mask, pos,
                                   dtype, backend, Q.data(), K.data(), V.data(), O.data(), x.data(),
                                   gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                   dW.data(), W_after.data(), &loss);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }

  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact("WHBO", 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

// WHV1 — Volume MHA NormalBP
int run_volume_mha_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short WHV1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 2;
  }
  (void)backend;

  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int mask = 0, pos = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> Q, K, V, O;
  };
  auto hops = welvet::architecture::hop_order(
      welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_seq = -1, first_d = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    int32_t seq = 0, d_model = 0, num_heads = 0, num_kv = 0, head_dim = 0, msl = 0;
    int32_t mask = 0, pos = 0, dtype = 1;
    if (!read_exact(&seq, 1) || !read_exact(&d_model, 1) || !read_exact(&num_heads, 1) ||
        !read_exact(&num_kv, 1) || !read_exact(&head_dim, 1) || !read_exact(&msl, 1) ||
        !read_exact(&mask, 1) || !read_exact(&pos, 1) || !read_exact(&dtype, 1)) {
      return 2;
    }
    c.seq = seq;
    c.d_model = d_model;
    c.num_heads = num_heads;
    c.num_kv = num_kv;
    c.head_dim = head_dim;
    c.msl = msl;
    c.mask = mask;
    c.pos = pos;
    c.dtype = dtype;
    const int hd = head_dim > 0 ? head_dim : d_model / num_heads;
    const int kv_heads = num_kv > 0 ? num_kv : num_heads;
    const int q_dim = num_heads * hd;
    const int kv_dim = kv_heads * hd;
    c.Q.resize(static_cast<std::size_t>(q_dim * d_model));
    c.K.resize(static_cast<std::size_t>(kv_dim * d_model));
    c.V.resize(static_cast<std::size_t>(kv_dim * d_model));
    c.O.resize(static_cast<std::size_t>(d_model * q_dim));
    if (!read_exact(c.Q.data(), c.Q.size()) || !read_exact(c.K.data(), c.K.size()) ||
        !read_exact(c.V.data(), c.V.size()) || !read_exact(c.O.data(), c.O.size())) {
      return 2;
    }
    if (first_seq < 0) {
      first_seq = seq;
      first_d = d_model;
    }
    cells.push_back(std::move(c));
  }
  if (first_seq < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_d};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_seq = 0, final_d = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (it->enabled) {
      final_seq = it->seq;
      final_d = it->d_model;
      break;
    }
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_d};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_d));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) {
        if (auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l)) {
          cell->meta.is_disabled = true;
        }
        continue;
      }
      welvet::mha::Config cfg;
      cfg.d_model = c.d_model;
      cfg.num_heads = c.num_heads;
      cfg.num_kv_heads = c.num_kv;
      cfg.head_dim = c.head_dim;
      cfg.max_seq_len = c.msl;
      cfg.mask = static_cast<welvet::mha::MaskKind>(c.mask);
      cfg.pos = static_cast<welvet::mha::PosKind>(c.pos);
      cfg.mode = welvet::mha::AttnMode::Self;
      cfg.causal = (cfg.mask == welvet::mha::MaskKind::Causal);
      welvet::mha::Layer layer;
      auto st = welvet::mha::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.Q.data(),
                                            c.K.data(), c.V.data(), c.O.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_mha(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                          std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) {
      return st;
    }
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  if (!write_exact(mode == VolumeTrainMode::Tween ? "THVO" : "WHVO", 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  int32_t n_cells = 0;
  for (const auto& c : cells) {
    if (c.enabled) ++n_cells;
  }
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int32_t nw = welvet::mha::grad_w_size(cell->mha_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::mha::export_weights_f32(cell->mha_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

// C1B1 / C2B1 / C3B1 — CNN NormalBP leaf
// C1V1 / C2V1 / C3V1 — Volume CNN NormalBP
// ---------------------------------------------------------------------------

int write_cnn_nbp_leaf_out(const char* magic, const std::vector<float>& pre,
                           const std::vector<float>& post, const std::vector<float>& dX,
                           const std::vector<float>& dW, const std::vector<float>& W_after,
                           double loss) {
  const int32_t pre_n = static_cast<int32_t>(pre.size());
  const int32_t post_n = static_cast<int32_t>(post.size());
  const int32_t dx_n = static_cast<int32_t>(dX.size());
  const int32_t dw_n = static_cast<int32_t>(dW.size());
  const int32_t w_n = static_cast<int32_t>(W_after.size());
  if (!write_exact(magic, 4) || !write_exact(&pre_n, 1) || !write_exact(pre.data(), pre.size()) ||
      !write_exact(&post_n, 1) || !write_exact(post.data(), post.size()) ||
      !write_exact(&dx_n, 1) || !write_exact(dX.data(), dX.size()) || !write_exact(&dw_n, 1) ||
      !write_exact(dW.data(), dW.size()) || !write_exact(&w_n, 1) ||
      !write_exact(W_after.data(), W_after.size()) || !write_exact(&loss, 1)) {
    return 2;
  }
  return 0;
}

template <typename Fn>
int bench_cnn_nbp_leaf(int bench_iters, Fn&& once, const char* out_magic,
                       std::vector<float>& pre, std::vector<float>& post, std::vector<float>& dX,
                       std::vector<float>& dW, std::vector<float>& W_after, double& loss) {
  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (st.code != 0) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) {
      return 2;
    }
  } else {
    auto st = once();
    if (st.code != 0) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message);
      return 1;
    }
  }
  return write_cnn_nbp_leaf_out(out_magic, pre, post, dX, dW, W_after, loss);
}

int run_cnn1_normalbp(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&seq, 1) || !read_exact(&kernel, 1) || !read_exact(&stride, 1) ||
      !read_exact(&padding, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short C1B1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_len = (seq + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * seq));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * filters * out_len));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_cnn1_normalbp_f32(batch, in_c, filters, seq, kernel, stride, padding, act, dtype,
                                    backend, W.data(), bias.data(), bias.size(), x.data(),
                                    gyt.data(), use_target, lr, pre.data(), post.data(), dX.data(),
                                    dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "C1BO", pre, post, dX, dW, W_after, loss);
}

int run_cnn2_normalbp(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0,
          padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&height, 1) || !read_exact(&width, 1) || !read_exact(&kernel, 1) ||
      !read_exact(&stride, 1) || !read_exact(&padding, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short C2B1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_h = (height + 2 * padding - kernel) / s + 1;
  const int out_w = (width + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * filters * out_h * out_w));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_cnn2_normalbp_f32(batch, in_c, filters, height, width, kernel, stride, padding,
                                    act, dtype, backend, W.data(), bias.data(), bias.size(),
                                    x.data(), gyt.data(), use_target, lr, pre.data(), post.data(),
                                    dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "C2BO", pre, post, dX, dW, W_after, loss);
}

int run_cnn3_normalbp(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, depth = 0, height = 0, width = 0, kernel = 0,
          stride = 0, padding = 0;
  int32_t act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&depth, 1) || !read_exact(&height, 1) || !read_exact(&width, 1) ||
      !read_exact(&kernel, 1) || !read_exact(&stride, 1) || !read_exact(&padding, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short C3B1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_d = (depth + 2 * padding - kernel) / s + 1;
  const int out_h = (height + 2 * padding - kernel) / s + 1;
  const int out_w = (width + 2 * padding - kernel) / s + 1;
  const int patch = in_c * kernel * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  uint32_t bias_len = 0;
  if (!read_exact(&bias_len, 1)) return 2;
  std::vector<double> bias(bias_len);
  if (bias_len && !read_exact(bias.data(), bias.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * depth * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * filters * out_d * out_h * out_w));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_cnn3_normalbp_f32(batch, in_c, filters, depth, height, width, kernel, stride,
                                    padding, act, dtype, backend, W.data(), bias.data(),
                                    bias.size(), x.data(), gyt.data(), use_target, lr, pre.data(),
                                    post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "C3BO", pre, post, dX, dW, W_after, loss);
}

int write_cnn_vol_nbp_out(const char* magic, double loss, const welvet::runtime::backward::Result& bwd,
                          const welvet::runtime::forward::Result& fwd,
                          const std::vector<std::vector<float>>& Ws) {
  if (!write_exact(magic, 4) || !write_exact(&loss, 1)) return 2;
  const int32_t gin_n = static_cast<int32_t>(bwd.grad_in.data.size());
  if (!write_exact(&gin_n, 1) || !write_exact(bwd.grad_in.data.data(), bwd.grad_in.data.size())) {
    return 2;
  }
  const int32_t n_cells = static_cast<int32_t>(Ws.size());
  if (!write_exact(&n_cells, 1)) return 2;
  for (const auto& Wa : Ws) {
    const int32_t nw = static_cast<int32_t>(Wa.size());
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  const int32_t out_n = static_cast<int32_t>(fwd.output.data.size());
  if (!write_exact(&out_n, 1) || !write_exact(fwd.output.data.data(), fwd.output.data.size())) {
    return 2;
  }
  return 0;
}

int run_volume_cnn1_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short C1V1 header\n");
    return 2;
  }
  if (batch <= 0) {
    std::fprintf(stderr, "welvet_dense_fwd: bad batch\n");
    return 1;
  }
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in_c = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) ||
        !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int patch = c.in_c * c.kernel;
    c.W.resize(static_cast<std::size_t>(c.filters * patch));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in_c < 0) {
      first_in_c = c.in_c;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) {
    std::fprintf(stderr, "welvet_dense_fwd: no enabled cells\n");
    return 1;
  }

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_filters = 0, final_out = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_filters = it->filters;
    const int s = it->stride <= 0 ? 1 : it->stride;
    final_out = (it->seq + 2 * it->padding - it->kernel) / s + 1;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_filters, final_out};
  target.data.resize(static_cast<std::size_t>(batch * final_filters * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::cnn1::Config cfg;
      cfg.in_channels = c.in_c;
      cfg.filters = c.filters;
      cfg.seq_len = c.seq;
      cfg.kernel = c.kernel;
      cfg.stride = c.stride;
      cfg.padding = c.padding;
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::cnn1::Layer layer;
      auto st = welvet::cnn1::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                             &layer);
      if (!st) return st;
      if (!c.bias.empty()) layer.proj.weights.bias = c.bias;
            st = welvet::architecture::bind_cnn1(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                           std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::cnn1::grad_w_size(cell->cnn1_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::cnn1::export_weights_f32(cell->cnn1_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "C1TO" : "C1VO", loss, bwd, fwd, Ws);
}

int run_volume_cnn2_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short C2V1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0, act = 0,
        dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in_c = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.height, 1) ||
        !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) ||
        !read_exact(&c.padding, 1) || !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int patch = c.in_c * c.kernel * c.kernel;
    c.W.resize(static_cast<std::size_t>(c.filters * patch));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in_c < 0) {
      first_in_c = c.in_c;
      first_h = c.height;
      first_w = c.width;
    }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_filters = 0, final_oh = 0, final_ow = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_filters = it->filters;
    const int s = it->stride <= 0 ? 1 : it->stride;
    final_oh = (it->height + 2 * it->padding - it->kernel) / s + 1;
    final_ow = (it->width + 2 * it->padding - it->kernel) / s + 1;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_filters, final_oh, final_ow};
  target.data.resize(static_cast<std::size_t>(batch * final_filters * final_oh * final_ow));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::cnn2::Config cfg;
      cfg.in_channels = c.in_c;
      cfg.filters = c.filters;
      cfg.height = c.height;
      cfg.width = c.width;
      cfg.kernel = c.kernel;
      cfg.stride = c.stride;
      cfg.padding = c.padding;
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::cnn2::Layer layer;
      auto st = welvet::cnn2::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                             &layer);
      if (!st) return st;
      if (!c.bias.empty()) layer.proj.weights.bias = c.bias;
            st = welvet::architecture::bind_cnn2(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                           std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::cnn2::grad_w_size(cell->cnn2_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::cnn2::export_weights_f32(cell->cnn2_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "C2TO" : "C2VO", loss, bwd, fwd, Ws);
}

int run_volume_cnn3_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short C3V1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in_c = 0, filters = 0, d = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0,
        act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
    std::vector<double> bias;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in_c = -1, first_d = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.d, 1) ||
        !read_exact(&c.height, 1) || !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) ||
        !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) || !read_exact(&c.act, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int patch = c.in_c * c.kernel * c.kernel * c.kernel;
    c.W.resize(static_cast<std::size_t>(c.filters * patch));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    uint32_t bias_len = 0;
    if (!read_exact(&bias_len, 1)) return 2;
    c.bias.resize(bias_len);
    if (bias_len && !read_exact(c.bias.data(), c.bias.size())) return 2;
    if (first_in_c < 0) {
      first_in_c = c.in_c;
      first_d = c.d;
      first_h = c.height;
      first_w = c.width;
    }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_d, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_d * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_filters = 0, final_od = 0, final_oh = 0, final_ow = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_filters = it->filters;
    const int s = it->stride <= 0 ? 1 : it->stride;
    final_od = (it->d + 2 * it->padding - it->kernel) / s + 1;
    final_oh = (it->height + 2 * it->padding - it->kernel) / s + 1;
    final_ow = (it->width + 2 * it->padding - it->kernel) / s + 1;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_filters, final_od, final_oh, final_ow};
  target.data.resize(
      static_cast<std::size_t>(batch * final_filters * final_od * final_oh * final_ow));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::cnn3::Config cfg;
      cfg.in_channels = c.in_c;
      cfg.filters = c.filters;
      cfg.depth = c.d;
      cfg.height = c.height;
      cfg.width = c.width;
      cfg.kernel = c.kernel;
      cfg.stride = c.stride;
      cfg.padding = c.padding;
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::cnn3::Layer layer;
      auto st = welvet::cnn3::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                             &layer);
      if (!st) return st;
      if (!c.bias.empty()) layer.proj.weights.bias = c.bias;
            st = welvet::architecture::bind_cnn3(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                           std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::cnn3::grad_w_size(cell->cnn3_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::cnn3::export_weights_f32(cell->cnn3_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "C3TO" : "C3VO", loss, bwd, fwd, Ws);
}




int run_convt1_normalbp(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0,
          out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&seq, 1) || !read_exact(&kernel, 1) || !read_exact(&stride, 1) ||
      !read_exact(&padding, 1) || !read_exact(&out_pad, 1) || !read_exact(&act, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short T1B1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_len = (seq - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * seq));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * filters * out_len));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_convt1_normalbp_f32(batch, in_c, filters, seq, kernel, stride, padding, out_pad,
                                      act, dtype, backend, W.data(), x.data(), gyt.data(),
                                      use_target, lr, pre.data(), post.data(), dX.data(),
                                      dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "T1BO", pre, post, dX, dW, W_after, loss);
}

int run_convt2_normalbp(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0,
          padding = 0, out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&height, 1) || !read_exact(&width, 1) || !read_exact(&kernel, 1) ||
      !read_exact(&stride, 1) || !read_exact(&padding, 1) || !read_exact(&out_pad, 1) ||
      !read_exact(&act, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short T2B1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_h = (height - 1) * s - 2 * padding + kernel + out_pad;
  const int out_w = (width - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * filters * out_h * out_w));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_convt2_normalbp_f32(batch, in_c, filters, height, width, kernel, stride, padding,
                                      out_pad, act, dtype, backend, W.data(), x.data(), gyt.data(),
                                      use_target, lr, pre.data(), post.data(), dX.data(),
                                      dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "T2BO", pre, post, dX, dW, W_after, loss);
}

int run_convt3_normalbp(int bench_iters) {
  int32_t batch = 0, in_c = 0, filters = 0, depth = 0, height = 0, width = 0, kernel = 0,
          stride = 0, padding = 0, out_pad = 0, act = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&in_c, 1) || !read_exact(&filters, 1) ||
      !read_exact(&depth, 1) || !read_exact(&height, 1) || !read_exact(&width, 1) ||
      !read_exact(&kernel, 1) || !read_exact(&stride, 1) || !read_exact(&padding, 1) ||
      !read_exact(&out_pad, 1) || !read_exact(&act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short T3B1 header\n");
    return 2;
  }
  const int s = stride <= 0 ? 1 : stride;
  const int out_d = (depth - 1) * s - 2 * padding + kernel + out_pad;
  const int out_h = (height - 1) * s - 2 * padding + kernel + out_pad;
  const int out_w = (width - 1) * s - 2 * padding + kernel + out_pad;
  const int patch = in_c * kernel * kernel * kernel;
  std::vector<float> W(static_cast<std::size_t>(filters * patch));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * in_c * depth * height * width));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * filters * out_d * out_h * out_w));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_convt3_normalbp_f32(batch, in_c, filters, depth, height, width, kernel, stride,
                                      padding, out_pad, act, dtype, backend, W.data(), x.data(),
                                      gyt.data(), use_target, lr, pre.data(), post.data(),
                                      dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "T3BO", pre, post, dX, dW, W_after, loss);
}


int run_volume_convt1_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short T1V1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in_c = 0, filters = 0, seq = 0, kernel = 0, stride = 0, padding = 0, out_pad = 0, act = 0,
        dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in_c = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) ||
        !read_exact(&c.out_pad, 1) || !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int patch = c.in_c * c.kernel;
    c.W.resize(static_cast<std::size_t>(c.filters * patch));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in_c < 0) {
      first_in_c = c.in_c;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_seq};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_seq));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_filters = 0, final_out = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_filters = it->filters;
    const int s = it->stride <= 0 ? 1 : it->stride;
    final_out = (it->seq - 1) * s - 2 * it->padding + it->kernel + it->out_pad;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_filters, final_out};
  target.data.resize(static_cast<std::size_t>(batch * final_filters * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::convt1::Config cfg;
      cfg.in_channels = c.in_c;
      cfg.filters = c.filters;
      cfg.seq_len = c.seq;
      cfg.kernel = c.kernel;
      cfg.stride = c.stride;
      cfg.padding = c.padding;
      cfg.output_padding = c.out_pad;
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::convt1::Layer layer;
      auto st = welvet::convt1::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                               &layer);
      if (!st) return st;
            st = welvet::architecture::bind_convt1(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::convt1::grad_w_size(cell->convt1_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::convt1::export_weights_f32(cell->convt1_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "T1TO" : "T1VO", loss, bwd, fwd, Ws);
}

int run_volume_convt2_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short T2V1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in_c = 0, filters = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0,
        out_pad = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in_c = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.height, 1) ||
        !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) || !read_exact(&c.stride, 1) ||
        !read_exact(&c.padding, 1) || !read_exact(&c.out_pad, 1) || !read_exact(&c.act, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int patch = c.in_c * c.kernel * c.kernel;
    c.W.resize(static_cast<std::size_t>(c.filters * patch));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in_c < 0) {
      first_in_c = c.in_c;
      first_h = c.height;
      first_w = c.width;
    }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_filters = 0, final_oh = 0, final_ow = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_filters = it->filters;
    const int s = it->stride <= 0 ? 1 : it->stride;
    final_oh = (it->height - 1) * s - 2 * it->padding + it->kernel + it->out_pad;
    final_ow = (it->width - 1) * s - 2 * it->padding + it->kernel + it->out_pad;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_filters, final_oh, final_ow};
  target.data.resize(static_cast<std::size_t>(batch * final_filters * final_oh * final_ow));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::convt2::Config cfg;
      cfg.in_channels = c.in_c;
      cfg.filters = c.filters;
      cfg.height = c.height;
      cfg.width = c.width;
      cfg.kernel = c.kernel;
      cfg.stride = c.stride;
      cfg.padding = c.padding;
      cfg.output_padding = c.out_pad;
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::convt2::Layer layer;
      auto st = welvet::convt2::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                               &layer);
      if (!st) return st;
            st = welvet::architecture::bind_convt2(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::convt2::grad_w_size(cell->convt2_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::convt2::export_weights_f32(cell->convt2_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "T2TO" : "T2VO", loss, bwd, fwd, Ws);
}

int run_volume_convt3_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short T3V1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in_c = 0, filters = 0, d = 0, height = 0, width = 0, kernel = 0, stride = 0, padding = 0,
        out_pad = 0, act = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in_c = -1, first_d = -1, first_h = -1, first_w = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in_c, 1) || !read_exact(&c.filters, 1) || !read_exact(&c.d, 1) ||
        !read_exact(&c.height, 1) || !read_exact(&c.width, 1) || !read_exact(&c.kernel, 1) ||
        !read_exact(&c.stride, 1) || !read_exact(&c.padding, 1) || !read_exact(&c.out_pad, 1) ||
        !read_exact(&c.act, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int patch = c.in_c * c.kernel * c.kernel * c.kernel;
    c.W.resize(static_cast<std::size_t>(c.filters * patch));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in_c < 0) {
      first_in_c = c.in_c;
      first_d = c.d;
      first_h = c.height;
      first_w = c.width;
    }
    cells.push_back(std::move(c));
  }
  if (first_in_c < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_in_c, first_d, first_h, first_w};
  input.data.resize(static_cast<std::size_t>(batch * first_in_c * first_d * first_h * first_w));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_filters = 0, final_od = 0, final_oh = 0, final_ow = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_filters = it->filters;
    const int s = it->stride <= 0 ? 1 : it->stride;
    final_od = (it->d - 1) * s - 2 * it->padding + it->kernel + it->out_pad;
    final_oh = (it->height - 1) * s - 2 * it->padding + it->kernel + it->out_pad;
    final_ow = (it->width - 1) * s - 2 * it->padding + it->kernel + it->out_pad;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_filters, final_od, final_oh, final_ow};
  target.data.resize(
      static_cast<std::size_t>(batch * final_filters * final_od * final_oh * final_ow));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::convt3::Config cfg;
      cfg.in_channels = c.in_c;
      cfg.filters = c.filters;
      cfg.depth = c.d;
      cfg.height = c.height;
      cfg.width = c.width;
      cfg.kernel = c.kernel;
      cfg.stride = c.stride;
      cfg.padding = c.padding;
      cfg.output_padding = c.out_pad;
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::convt3::Layer layer;
      auto st = welvet::convt3::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                               &layer);
      if (!st) return st;
            st = welvet::architecture::bind_convt3(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::convt3::grad_w_size(cell->convt3_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::convt3::export_weights_f32(cell->convt3_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "T3TO" : "T3VO", loss, bwd, fwd, Ws);
}



// ---------------------------------------------------------------------------
// RNB1 — RNN NormalBP leaf
// RNV1 — Volume RNN NormalBP
// LSB1 — LSTM NormalBP leaf
// LSV1 — Volume LSTM NormalBP
// ---------------------------------------------------------------------------


int run_rnn_normalbp(int bench_iters) {
  int32_t batch = 0, input_size = 0, hidden_size = 0, seq_len = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&input_size, 1) || !read_exact(&hidden_size, 1) ||
      !read_exact(&seq_len, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short RNB1 header\n");
    return 2;
  }
  const int nw = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq_len * input_size));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq_len * hidden_size));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_rnn_normalbp_f32(batch, input_size, hidden_size, seq_len, dtype, backend, W.data(),
                                   x.data(), gyt.data(), use_target, lr, pre.data(), post.data(),
                                   dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "RNBO", pre, post, dX, dW, W_after, loss);
}

int run_lstm_normalbp(int bench_iters) {
  int32_t batch = 0, input_size = 0, hidden_size = 0, seq_len = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&input_size, 1) || !read_exact(&hidden_size, 1) ||
      !read_exact(&seq_len, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short LSB1 header\n");
    return 2;
  }
  const int gate = hidden_size * input_size + hidden_size * hidden_size + hidden_size;
  const int nw = 4 * gate;
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq_len * input_size));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq_len * hidden_size));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  const int npre = batch * seq_len * 5 * hidden_size;
  std::vector<float> pre(static_cast<std::size_t>(npre)), post(gyt.size()), dX(x.size()), dW(W.size()),
      W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_lstm_normalbp_f32(batch, input_size, hidden_size, seq_len, dtype, backend, W.data(),
                                    x.data(), gyt.data(), use_target, lr, pre.data(), post.data(),
                                    dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "LSBO", pre, post, dX, dW, W_after, loss);
}

int run_volume_rnn_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short RNV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, hid = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in, 1) || !read_exact(&c.hid, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = c.hid * c.in + c.hid * c.hid + c.hid;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in < 0) {
      first_in = c.in;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_in < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_hid = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_hid = it->hid;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_hid};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_hid));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::rnn::Config cfg;
      cfg.input_size = c.in;
      cfg.hidden_size = c.hid;
      cfg.seq_len = c.seq;
      welvet::rnn::Layer layer;
      auto st = welvet::rnn::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                            &layer);
      if (!st) return st;
      st = welvet::architecture::bind_rnn(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                          std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::rnn::grad_w_size(cell->rnn_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::rnn::export_packed_f32(cell->rnn_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "RNTO" : "RNVO", loss, bwd, fwd, Ws);
}

int run_volume_lstm_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short LSV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int in = 0, hid = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_in = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.in, 1) || !read_exact(&c.hid, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = 4 * (c.hid * c.in + c.hid * c.hid + c.hid);
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_in < 0) {
      first_in = c.in;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_in < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_in};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_in));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_hid = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_hid = it->hid;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_hid};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_hid));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::lstm::Config cfg;
      cfg.input_size = c.in;
      cfg.hidden_size = c.hid;
      cfg.seq_len = c.seq;
      welvet::lstm::Layer layer;
      auto st = welvet::lstm::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                             &layer);
      if (!st) return st;
      st = welvet::architecture::bind_lstm(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                           std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::lstm::grad_w_size(cell->lstm_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::lstm::export_packed_f32(cell->lstm_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "LSTO" : "LSVO", loss, bwd, fwd, Ws);
}

// ---------------------------------------------------------------------------
// SQB1 — Sequential NormalBP leaf
// SQV1 — Volume Sequential NormalBP
// REB1 — Residual NormalBP leaf
// REV1 — Volume Residual NormalBP
// STKB — Stack NestedStack-of-Dense NormalBP leaf
// STKV — Volume Stack NestedStack-of-Dense NormalBP
// ---------------------------------------------------------------------------

int run_sequential_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, depth = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&depth, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short SQB1 header\n");
    return 2;
  }
  const int nw = depth * dim * dim;
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_sequential_normalbp_f32(batch, seq, dim, depth, child_act, dtype, backend,
                                          W.data(), x.data(), gyt.data(), use_target, lr, pre.data(),
                                          post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "SQBO", pre, post, dX, dW, W_after, loss);
}

int run_residual_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, depth = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&depth, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short REB1 header\n");
    return 2;
  }
  const int nw = depth * dim * dim;
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_residual_normalbp_f32(batch, seq, dim, depth, child_act, dtype, backend, W.data(),
                                        x.data(), gyt.data(), use_target, lr, pre.data(),
                                        post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "REBO", pre, post, dX, dW, W_after, loss);
}

int run_stack_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, n_children = 0, child_act = -1, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&n_children, 1) || !read_exact(&child_act, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short STKB header\n");
    return 2;
  }
  const int nw = n_children * dim * dim;
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_stack_normalbp_f32(batch, seq, dim, n_children, child_act, dtype, backend,
                                     W.data(), x.data(), gyt.data(), use_target, lr, pre.data(),
                                     post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "STKO", pre, post, dX, dW, W_after, loss);
}

int run_volume_sequential_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short SQV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq = 0, chain = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.chain, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = c.chain * c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) {
      first_dim = c.dim;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) {
    input.shape = {batch, first_dim};
  } else {
    input.shape = {batch, first_seq, first_dim};
  }
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_dim = it->dim;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  if (final_seq == 1) {
    target.shape = {batch, final_dim};
  } else {
    target.shape = {batch, final_seq, final_dim};
  }
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::sequential::Config cfg;
      cfg.dim = c.dim;
      cfg.seq_len = c.seq;
      cfg.depth = c.chain;
      welvet::sequential::Layer layer;
      auto st = welvet::sequential::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                   welvet::Activation::Linear, c.W.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_sequential(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                 std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::sequential::grad_w_size(cell->sequential_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::sequential::export_packed_f32(cell->sequential_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "SQTO" : "SQVO", loss, bwd, fwd, Ws);
}

int run_volume_residual_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short REV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq = 0, chain = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.chain, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = c.chain * c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) {
      first_dim = c.dim;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) {
    input.shape = {batch, first_dim};
  } else {
    input.shape = {batch, first_seq, first_dim};
  }
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_dim = it->dim;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  if (final_seq == 1) {
    target.shape = {batch, final_dim};
  } else {
    target.shape = {batch, final_seq, final_dim};
  }
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::residual::Config cfg;
      cfg.dim = c.dim;
      cfg.seq_len = c.seq;
      cfg.depth = c.chain;
      welvet::residual::Layer layer;
      auto st = welvet::residual::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                 welvet::Activation::Linear, c.W.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_residual(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                               std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::residual::grad_w_size(cell->residual_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::residual::export_packed_f32(cell->residual_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "RETO" : "REVO", loss, bwd, fwd, Ws);
}

namespace {

welvet::Status make_vol_nested_stack_dense(int dim, int n_children, welvet::DType dtype,
                                           const float* packed, welvet::stack::Layer* out) {
  if (!out || !packed || dim <= 0 || n_children <= 0) {
    return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "stack vol: bad args");
  }
  welvet::stack::Child nest;
  nest.kind = welvet::stack::Kind::NestedStack;
  int off = 0;
  for (int i = 0; i < n_children; ++i) {
    welvet::stack::Child d;
    d.kind = welvet::stack::Kind::Dense;
    auto st = welvet::dense::new_layer(dim, dim, welvet::Activation::Linear, dtype, packed + off,
                                       &d.dense_op);
    if (!st) return st;
    off += dim * dim;
    nest.nested.push_back(std::move(d));
  }
  out->children.clear();
  out->children.push_back(std::move(nest));
  out->meta.type = welvet::LayerType::Stack;
  out->meta.dtype = dtype;
  out->meta.input_height = dim;
  out->meta.output_height = dim;
  return welvet::Status::Ok();
}

} // namespace

int run_volume_stack_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short STKV header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq = 0, chain = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.chain, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = c.chain * c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) {
      first_dim = c.dim;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch * first_seq, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_dim = it->dim;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch * final_seq, final_dim};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::stack::Layer layer;
      auto st = make_vol_nested_stack_dense(c.dim, c.chain, static_cast<welvet::DType>(c.dtype),
                                            c.W.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_stack(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::stack::grad_w_size(cell->stack_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::stack::export_packed_f32(cell->stack_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "SKTO" : "STVO", loss, bwd, fwd, Ws);
}

// MBB1 — Mamba NormalBP leaf
// MBV1 — Volume Mamba NormalBP
// ---------------------------------------------------------------------------

int run_mamba_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, d_model = 0, d_state = 0, expand = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&d_model, 1) ||
      !read_exact(&d_state, 1) || !read_exact(&expand, 1) || !read_exact(&dtype, 1) ||
      !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short MBB1 header\n");
    return 2;
  }
  int expand_eff = expand;
  if (expand_eff <= 0) expand_eff = 2;
  const int inner = expand_eff * d_model;
  const int nw = inner * (3 * d_model + 2);
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * d_model));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_mamba_normalbp_f32(batch, seq, d_model, d_state, expand, dtype, backend, W.data(),
                                     x.data(), gyt.data(), use_target, lr, pre.data(), post.data(),
                                     dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "MBBO", pre, post, dX, dW, W_after, loss);
}

int run_volume_mamba_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short MBV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int d_model = 0, d_state = 0, expand = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_d = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.d_model, 1) || !read_exact(&c.d_state, 1) || !read_exact(&c.expand, 1) ||
        !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    int exp = c.expand;
    if (exp <= 0) exp = 2;
    const int inner = exp * c.d_model;
    const int nw = inner * (3 * c.d_model + 2);
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_d < 0) {
      first_d = c.d_model;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_d < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_d};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_d = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_d = it->d_model;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_d};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_d));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::mamba::Config cfg;
      cfg.d_model = c.d_model;
      cfg.d_state = c.d_state;
      cfg.expand = c.expand;
      cfg.seq_len = c.seq;
      welvet::mamba::Layer layer;
      auto st = welvet::mamba::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(),
                                              &layer);
      if (!st) return st;
      st = welvet::architecture::bind_mamba(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                            std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::mamba::grad_w_size(cell->mamba_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::mamba::export_packed_f32(cell->mamba_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "MBTO" : "MBVO", loss, bwd, fwd, Ws);
}

// ---------------------------------------------------------------------------
// MCB1 — Metacognition NormalBP leaf
// MCV1 — Volume Metacognition NormalBP
// ---------------------------------------------------------------------------

int run_metacognition_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, dim = 0, dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&dim, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short MCB1 header\n");
    return 2;
  }
  const int nw = dim * dim;
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * dim));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_metacognition_normalbp_f32(batch, seq, dim, dtype, backend, W.data(), x.data(),
                                             gyt.data(), use_target, lr, pre.data(), post.data(),
                                             dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "MCBO", pre, post, dX, dW, W_after, loss);
}

int run_volume_metacognition_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short MCV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int dim = 0, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = c.dim * c.dim;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_dim < 0) {
      first_dim = c.dim;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) {
    input.shape = {batch, first_dim};
  } else {
    input.shape = {batch, first_seq, first_dim};
  }
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_dim = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_dim = it->dim;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  if (final_seq == 1) {
    target.shape = {batch, final_dim};
  } else {
    target.shape = {batch, final_seq, final_dim};
  }
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_dim));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::metacognition::Config cfg;
      cfg.dim = c.dim;
      cfg.seq_len = c.seq == 1 ? 0 : c.seq;
      welvet::metacognition::Layer layer;
      auto st = welvet::metacognition::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                      c.W.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_metacognition(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                                    std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::metacognition::grad_w_size(cell->metacognition_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::metacognition::export_packed_f32(cell->metacognition_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "MCTO" : "MCVO", loss, bwd, fwd, Ws);
}

// ---------------------------------------------------------------------------
// KMB1 — KMeans NormalBP leaf
// KMV1 — Volume KMeans NormalBP
// ---------------------------------------------------------------------------

int run_kmeans_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, k = 0, d = 0, output_mode = 0, act = -1, dtype = 1, backend = 0;
  double temperature = 1.0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&k, 1) || !read_exact(&d, 1) ||
      !read_exact(&output_mode, 1) || !read_exact(&act, 1) || !read_exact(&temperature, 1) ||
      !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short KMB1 header\n");
    return 2;
  }
  const int nw = k * d;
  const int out_dim =
      output_mode == 1 ? d : k; // Features=1 → D, Probabilities=0 → K
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * d));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * out_dim));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_kmeans_normalbp_f32(batch, seq, k, d, output_mode, act, temperature, dtype,
                                      backend, W.data(), x.data(), gyt.data(), use_target, lr,
                                      pre.data(), post.data(), dX.data(), dW.data(), W_after.data(),
                                      &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "KMBO", pre, post, dX, dW, W_after, loss);
}

int run_volume_kmeans_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short KMV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int k = 0, d = 0, seq = 0, output_mode = 0, act = -1, dtype = 1;
    double temperature = 1.0;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_d = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.k, 1) || !read_exact(&c.d, 1) || !read_exact(&c.seq, 1) ||
        !read_exact(&c.output_mode, 1) || !read_exact(&c.act, 1) || !read_exact(&c.temperature, 1) ||
        !read_exact(&c.dtype, 1)) {
      return 2;
    }
    const int nw = c.k * c.d;
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_d < 0) {
      first_d = c.d;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_d < 0) return 1;

  welvet::TensorF32 input;
  if (first_seq == 1) {
    input.shape = {batch, first_d};
  } else {
    input.shape = {batch, first_seq, first_d};
  }
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_d));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_out = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_out = it->output_mode == 1 ? it->d : it->k;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  if (final_seq == 1) {
    target.shape = {batch, final_out};
  } else {
    target.shape = {batch, final_seq, final_out};
  }
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::kmeans::Config cfg;
      cfg.num_clusters = c.k;
      cfg.feature_dim = c.d;
      cfg.temperature = c.temperature;
      cfg.output_mode = static_cast<welvet::kmeans::OutputMode>(c.output_mode);
      cfg.activation = static_cast<welvet::Activation>(c.act);
      welvet::kmeans::Layer layer;
      auto st =
          welvet::kmeans::new_configured(cfg, static_cast<welvet::DType>(c.dtype), c.W.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_kmeans(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                             std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::kmeans::grad_w_size(cell->kmeans_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::kmeans::export_packed_f32(cell->kmeans_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "KMTO" : "KMVO", loss, bwd, fwd, Ws);
}

// ---------------------------------------------------------------------------
// GDB1 — GDN NormalBP leaf (f32 truncated-BPTT)
// GDV1 — Volume GDN NormalBP
// Magics match MCB1/MBB1 style (plan said WGDB/WGDO).
// ---------------------------------------------------------------------------

int run_gdn_normalbp(int bench_iters) {
  int32_t batch = 0, seq = 0, hidden = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 0;
  int32_t dtype = 1, backend = 0;
  if (!read_exact(&batch, 1) || !read_exact(&seq, 1) || !read_exact(&hidden, 1) ||
      !read_exact(&nk, 1) || !read_exact(&nv, 1) || !read_exact(&hd_k, 1) || !read_exact(&hd_v, 1) ||
      !read_exact(&conv_k, 1) || !read_exact(&dtype, 1) || !read_exact(&backend, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short GDB1 header\n");
    return 2;
  }
  welvet::gdn::Config cfg;
  cfg.hidden_size = hidden;
  cfg.num_key_heads = nk;
  cfg.num_value_heads = nv;
  cfg.key_head_dim = hd_k;
  cfg.value_head_dim = hd_v;
  cfg.conv_kernel = conv_k;
  if (auto st = welvet::gdn::validate(&cfg); !st) {
    std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
    return 1;
  }
  const int nw = welvet::gdn::weight_count(cfg);
  std::vector<float> W(static_cast<std::size_t>(nw));
  if (!read_exact(W.data(), W.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * seq * hidden));
  if (!read_exact(x.data(), x.size())) return 2;
  int32_t use_target = 0;
  if (!read_exact(&use_target, 1)) return 2;
  std::vector<float> gyt(static_cast<std::size_t>(batch * seq * hidden));
  if (!read_exact(gyt.data(), gyt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<float> pre(gyt.size()), post(gyt.size()), dX(x.size()), dW(W.size()), W_after(W.size());
  double loss = 0.0;
  auto once = [&]() -> welvet_status {
    return welvet_gdn_normalbp_f32(batch, seq, hidden, nk, nv, hd_k, hd_v, conv_k, dtype, backend,
                                   W.data(), x.data(), gyt.data(), use_target, lr, pre.data(),
                                   post.data(), dX.data(), dW.data(), W_after.data(), &loss);
  };
  return bench_cnn_nbp_leaf(bench_iters, once, "GDBO", pre, post, dX, dW, W_after, loss);
}

int run_volume_gdn_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short GDV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;

  struct CellInit {
    bool enabled = false;
    int hidden = 0, nk = 0, nv = 0, hd_k = 0, hd_v = 0, conv_k = 4, seq = 0, dtype = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> W;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_h = -1, first_seq = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.hidden, 1) || !read_exact(&c.nk, 1) || !read_exact(&c.nv, 1) ||
        !read_exact(&c.hd_k, 1) || !read_exact(&c.hd_v, 1) || !read_exact(&c.conv_k, 1) ||
        !read_exact(&c.seq, 1) || !read_exact(&c.dtype, 1)) {
      return 2;
    }
    welvet::gdn::Config cfg;
    cfg.hidden_size = c.hidden;
    cfg.num_key_heads = c.nk;
    cfg.num_value_heads = c.nv;
    cfg.key_head_dim = c.hd_k;
    cfg.value_head_dim = c.hd_v;
    cfg.conv_kernel = c.conv_k;
    if (auto st = welvet::gdn::validate(&cfg); !st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
    const int nw = welvet::gdn::weight_count(cfg);
    c.W.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.W.data(), c.W.size())) return 2;
    if (first_h < 0) {
      first_h = c.hidden;
      first_seq = c.seq;
    }
    cells.push_back(std::move(c));
  }
  if (first_h < 0) return 1;

  welvet::TensorF32 input;
  input.shape = {batch, first_seq, first_h};
  input.data.resize(static_cast<std::size_t>(batch * first_seq * first_h));
  if (!read_exact(input.data.data(), input.data.size())) return 2;

  int final_h = 0, final_seq = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_h = it->hidden;
    final_seq = it->seq;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_seq, final_h};
  target.data.resize(static_cast<std::size_t>(batch * final_seq * final_h));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;

  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;

  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::gdn::Config cfg;
      cfg.hidden_size = c.hidden;
      cfg.num_key_heads = c.nk;
      cfg.num_value_heads = c.nv;
      cfg.key_head_dim = c.hd_k;
      cfg.value_head_dim = c.hd_v;
      cfg.conv_kernel = c.conv_k;
      welvet::gdn::Layer layer;
      auto st = welvet::gdn::new_configured(cfg, c.W.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_gdn(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                          std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };

  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }

  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::gdn::grad_w_size(cell->gdn_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::gdn::export_packed_f32(cell->gdn_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "GDTO" : "GDVO", loss, bwd, fwd, Ws);
}

// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// PNB1 / PNBO — Parallel Dense NormalBP leaf (all combines)
// PNV1 / PNVO — Volume Parallel Dense Add/Avg NormalBP
// PPB1 / PPBO — Parallel poly branch NormalBP
// NBL1 / NBLO — NormalBPEdges child wire
// ---------------------------------------------------------------------------

namespace {

bool write_magic_local(const char* m) { return write_exact(m, 4); }

class ChildReader {
public:
  int depth = 0;
  bool allow_count(size_t n) const {
    return n <= (1 << 26) && (!replay_ || n <= (replay_->size() - offset_) / 4);
  }
  std::vector<unsigned char> bytes;
  explicit ChildReader(const std::vector<unsigned char>* replay = nullptr) : replay_(replay) {}
  template <typename T>
  bool read(T* p, size_t n) {
    const size_t size = n * sizeof(T);
    if (replay_) {
      if (offset_ > replay_->size() || size > replay_->size() - offset_) return false;
      if (size) std::memcpy(p, replay_->data() + offset_, size);
      offset_ += size;
      return true;
    }
    if (!read_exact(p, n)) return false;
    if (size) {
      const auto* begin = reinterpret_cast<const unsigned char*>(p);
      bytes.insert(bytes.end(), begin, begin + size);
    }
    return true;
  }

private:
  const std::vector<unsigned char>* replay_;
  size_t offset_ = 0;
};

bool read_child(ChildReader& reader, welvet::stack::Child* ch) {
  try {
    return welvet::interop::wire::read_child(reader, ch);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "model wire: %s\n", e.what());
    return false;
  }
}

welvet::Status train_parallel(welvet::parallel::Layer& l, const welvet::TensorF32& x,
                              const welvet::TensorF32& target, double lr, double* loss) {
  welvet::TensorF32 p, o, gy, dx, dw;
  if (auto st = welvet::parallel::forward(l, x, &p, &o); !st) return st;
  welvet::TensorF32 shaped_target = target;
  if (shaped_target.data.size() == o.data.size()) shaped_target.shape = o.shape;
  if (auto st = welvet::runtime::training::mse(o, shaped_target, loss); !st) return st;
  if (auto st = welvet::runtime::training::mse_grad(o, shaped_target, &gy); !st) return st;
  if (auto st = welvet::parallel::backward_f32(l, gy, x, p, &dx, &dw); !st) return st;
  return welvet::parallel::apply_grad_sgd(&l, dw, lr);
}

} // namespace

int run_parallel_normalbp(int bench_iters) {
  const char* out_magic = "PNBO";
  int32_t batch = 0, dim = 0, out_feat = 0, branches = 0, dtype = 1;
  if (!read_exact(&batch, 1) || !read_exact(&dim, 1) || !read_exact(&out_feat, 1) ||
      !read_exact(&branches, 1) || !read_exact(&dtype, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short PNB1 header\n");
    return 2;
  }
  if (batch <= 0 || dim <= 0 || out_feat <= 0 || branches <= 0) return 1;
  int32_t combine_i = 1;
  if (!read_exact(&combine_i, 1)) return 2;
  auto combine = static_cast<welvet::parallel::Combine>(combine_i);
  switch (combine) {
  case welvet::parallel::Combine::Concat:
  case welvet::parallel::Combine::Add:
  case welvet::parallel::Combine::Avg:
  case welvet::parallel::Combine::Filter:
  case welvet::parallel::Combine::Max:
  case welvet::parallel::Combine::SparseK:
  case welvet::parallel::Combine::Disagree:
    break;
  default:
    std::fprintf(stderr, "welvet_dense_fwd: PNB1 unknown combine %d\n", combine_i);
    return 1;
  }
  int wn = branches * out_feat * dim;
  if (combine == welvet::parallel::Combine::Filter) wn += branches * dim;
  std::vector<float> packed(static_cast<std::size_t>(wn));
  if (!read_exact(packed.data(), packed.size())) return 2;
  std::vector<float> x(static_cast<std::size_t>(batch * dim));
  const int out_d = (combine == welvet::parallel::Combine::Concat) ? branches * out_feat : out_feat;
  std::vector<float> tgt(static_cast<std::size_t>(batch * out_d));
  if (!read_exact(x.data(), x.size()) || !read_exact(tgt.data(), tgt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  std::vector<int32_t> modes(static_cast<std::size_t>(branches));
  if (!read_exact(modes.data(), modes.size())) return 2;
  double shadow_coef = 0.0, surprise_thresh = 0.0;
  if (!read_exact(&shadow_coef, 1) || !read_exact(&surprise_thresh, 1)) return 2;
  for (int m : modes) {
    if (m != 0 && m != 1) {
      std::fprintf(stderr, "PNB1: only Inherit/NormalBP supported\n");
      return 1;
    }
  }
  welvet::parallel::Config cfg;
  cfg.dim = dim;
  cfg.out_feat = out_feat;
  cfg.branches = branches;
  cfg.combine = combine;
  welvet::TensorF32 xin, target;
  xin.shape = {batch, dim};
  xin.data = x;
  target.shape = {batch, out_d};
  target.data = tgt;
  double loss = 0;
  welvet::parallel::Layer layer;
  auto once = [&]() -> welvet::Status {
    if (auto st = welvet::parallel::new_configured(cfg, static_cast<welvet::DType>(dtype),
                                                   packed.data(), &layer);
        !st) {
      return st;
    }
    return train_parallel(layer, xin, target, lr, &loss);
  };
  if (bench_iters > 0) {
    for (int i = 0; i < 2; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  if (!write_magic_local(out_magic) || !write_exact(&loss, 1) || !write_exact(&branches, 1)) return 2;
  for (int i = 0; i < branches; ++i) {
    auto* d = welvet::parallel::dense_branch(layer, i);
    if (!d) {
      std::fprintf(stderr, "welvet_dense_fwd: PNB1 expects Dense branches\n");
      return 1;
    }
    const int32_t nw = out_feat * dim;
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = d->weights.export_f32(Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    if (!write_exact(&nw, 1) || !write_exact(Wa.data(), Wa.size())) return 2;
  }
  return 0;
}

int run_volume_parallel_normalbp(int bench_iters, VolumeTrainMode mode) {
  int32_t depth = 0, rows = 0, cols = 0, lpc = 0, batch = 0, backend = 0, n_links = 0;
  if (!read_exact(&depth, 1) || !read_exact(&rows, 1) || !read_exact(&cols, 1) ||
      !read_exact(&lpc, 1) || !read_exact(&batch, 1) || !read_exact(&backend, 1) ||
      !read_exact(&n_links, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short PNV1 header\n");
    return 2;
  }
  if (batch <= 0) return 1;
  (void)backend;
  std::vector<int32_t> links(static_cast<std::size_t>(n_links) * 8);
  if (n_links > 0 && !read_exact(links.data(), links.size())) return 2;
  struct CellInit {
    bool enabled = false;
    int dim = 0, out_feat = 0, branches = 0, dtype = 1, combine = 1;
    welvet::architecture::Coord coord{};
    std::vector<float> packed;
  };
  auto hops = welvet::architecture::hop_order(welvet::architecture::new_grid(depth, rows, cols, lpc));
  std::vector<CellInit> cells;
  cells.reserve(hops.size());
  int first_dim = -1;
  for (const auto& coord : hops) {
    CellInit c;
    c.coord = coord;
    int32_t en = 0;
    if (!read_exact(&en, 1)) return 2;
    c.enabled = en != 0;
    if (!c.enabled) {
      cells.push_back(std::move(c));
      continue;
    }
    if (!read_exact(&c.dim, 1) || !read_exact(&c.out_feat, 1) || !read_exact(&c.branches, 1) ||
        !read_exact(&c.dtype, 1) || !read_exact(&c.combine, 1)) {
      return 2;
    }
    if (c.dim <= 0 || c.out_feat <= 0 || c.branches <= 0) return 1;
    if (c.combine != static_cast<int>(welvet::parallel::Combine::Add) &&
        c.combine != static_cast<int>(welvet::parallel::Combine::Avg)) {
      std::fprintf(stderr, "welvet_dense_fwd: parallel vol combine must be Add/Avg\n");
      return 1;
    }
    const int nw = c.branches * c.out_feat * c.dim;
    c.packed.resize(static_cast<std::size_t>(nw));
    if (!read_exact(c.packed.data(), c.packed.size())) return 2;
    if (first_dim < 0) first_dim = c.dim;
    cells.push_back(std::move(c));
  }
  if (first_dim < 0) return 1;
  welvet::TensorF32 input;
  input.shape = {batch, first_dim};
  input.data.resize(static_cast<std::size_t>(batch * first_dim));
  if (!read_exact(input.data.data(), input.data.size())) return 2;
  int final_out = 0;
  for (auto it = cells.rbegin(); it != cells.rend(); ++it) {
    if (!it->enabled) continue;
    final_out = it->out_feat;
    break;
  }
  welvet::TensorF32 target;
  target.shape = {batch, final_out};
  target.data.resize(static_cast<std::size_t>(batch * final_out));
  if (!read_exact(target.data.data(), target.data.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  welvet::architecture::Grid g;
  welvet::runtime::forward::Result fwd;
  welvet::runtime::backward::Result bwd;
  double loss = 0.0;
  auto rebuild_and_step = [&]() -> welvet::Status {
    g = welvet::architecture::new_grid(depth, rows, cols, lpc);
    for (int32_t i = 0; i < n_links; ++i) {
      const int32_t* L = links.data() + static_cast<std::size_t>(i) * 8;
      if (auto st = welvet::architecture::set_remote_link(g, L[0], L[1], L[2], L[3], L[4], L[5],
                                                          L[6], L[7]);
          !st) {
        return st;
      }
    }
    for (const auto& c : cells) {
      if (!c.enabled) continue;
      welvet::parallel::Config cfg;
      cfg.dim = c.dim;
      cfg.out_feat = c.out_feat;
      cfg.branches = c.branches;
      cfg.combine = static_cast<welvet::parallel::Combine>(c.combine);
      welvet::parallel::Layer layer;
      auto st = welvet::parallel::new_configured(cfg, static_cast<welvet::DType>(c.dtype),
                                                 c.packed.data(), &layer);
      if (!st) return st;
      st = welvet::architecture::bind_parallel(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l,
                                               std::move(layer));
      if (!st) return st;
    }
    if (auto st = welvet::runtime::forward::volume_dense(g, input, &fwd); !st) return st;
    if (mode == VolumeTrainMode::Tween) {
      bwd = {};
      return welvet::systems::tween::train_volume_mse(g, fwd, target, lr, &loss);
    }
    return welvet::runtime::training::step_dense(g, fwd, target, lr, &loss, &bwd);
  };
  if (bench_iters > 0) {
    for (int i = 0; i < 3; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = rebuild_and_step();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = rebuild_and_step();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  std::vector<std::vector<float>> Ws;
  for (const auto& c : cells) {
    if (!c.enabled) continue;
    auto* cell = welvet::architecture::at(g, c.coord.z, c.coord.y, c.coord.x, c.coord.l);
    if (!cell) return 1;
    const int nw = welvet::parallel::grad_w_size(cell->parallel_op);
    std::vector<float> Wa(static_cast<std::size_t>(nw));
    auto est = welvet::parallel::export_packed_f32(cell->parallel_op, Wa.data());
    if (!est) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
      return 1;
    }
    Ws.push_back(std::move(Wa));
  }
  return write_cnn_vol_nbp_out(mode == VolumeTrainMode::Tween ? "PNTO" : "PNVO", loss, bwd, fwd, Ws);
}

int run_parallel_poly_train(int bench_iters) {
  int32_t combine_i = 1, cfg_dim = 0, cfg_out_feat = 0, cfg_seq = 0, branches = 0;
  if (!read_exact(&combine_i, 1) || !read_exact(&cfg_dim, 1) || !read_exact(&cfg_out_feat, 1) ||
      !read_exact(&cfg_seq, 1) || !read_exact(&branches, 1)) {
    std::fprintf(stderr, "welvet_dense_fwd: short PPB1 header\n");
    return 2;
  }
  if (branches <= 0) return 1;
  if (combine_i != static_cast<int32_t>(welvet::parallel::Combine::Add) &&
      combine_i != static_cast<int32_t>(welvet::parallel::Combine::Avg)) {
    std::fprintf(stderr, "welvet_dense_fwd: PPB1 combine must be Add/Avg\n");
    return 1;
  }
  ChildReader initial;
  std::vector<welvet::parallel::Branch> brs;
  brs.reserve(static_cast<std::size_t>(branches));
  for (int i = 0; i < branches; ++i) {
    welvet::parallel::Branch br;
    if (!read_child(initial, &br)) {
      std::fprintf(stderr, "welvet_dense_fwd: PPB1 bad branch %d\n", i);
      return 2;
    }
    brs.push_back(std::move(br));
  }
  int32_t x_rank = 0, t_rank = 0;
  if (!read_exact(&x_rank, 1) || x_rank <= 0) return 2;
  std::vector<int32_t> x_shape(static_cast<std::size_t>(x_rank));
  if (!read_exact(x_shape.data(), x_shape.size())) return 2;
  int x_n = 1;
  for (int32_t d : x_shape) x_n *= static_cast<int>(d);
  std::vector<float> x(static_cast<std::size_t>(x_n));
  if (!read_exact(x.data(), x.size())) return 2;
  if (!read_exact(&t_rank, 1) || t_rank <= 0) return 2;
  std::vector<int32_t> t_shape(static_cast<std::size_t>(t_rank));
  if (!read_exact(t_shape.data(), t_shape.size())) return 2;
  int t_n = 1;
  for (int32_t d : t_shape) t_n *= static_cast<int>(d);
  std::vector<float> tgt(static_cast<std::size_t>(t_n));
  if (!read_exact(tgt.data(), tgt.size())) return 2;
  double lr = 0.0;
  if (!read_exact(&lr, 1)) return 2;
  welvet::parallel::Config cfg;
  cfg.dim = cfg_dim;
  cfg.out_feat = cfg_out_feat;
  cfg.branches = branches;
  cfg.combine = static_cast<welvet::parallel::Combine>(combine_i);
  cfg.seq_len = cfg_seq;
  welvet::TensorF32 xin, target;
  xin.shape.assign(x_shape.begin(), x_shape.end());
  xin.data = x;
  target.shape.assign(t_shape.begin(), t_shape.end());
  target.data = tgt;
  double loss = 0.0;
  welvet::parallel::Layer layer;
  auto once = [&]() -> welvet::Status {
    std::vector<welvet::parallel::Branch> copy;
    copy.reserve(brs.size());
    ChildReader replay(&initial.bytes);
    for (size_t i = 0; i < brs.size(); ++i) {
      welvet::parallel::Branch b;
      if (!read_child(replay, &b)) {
        return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "parallel: replay");
      }
      copy.push_back(std::move(b));
    }
    if (auto st = welvet::parallel::new_from_branches(cfg, std::move(copy), std::nullopt, &layer);
        !st) {
      return st;
    }
    return train_parallel(layer, xin, target, lr, &loss);
  };
  if (bench_iters > 0) {
    for (int i = 0; i < 2; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
      auto st = once();
      if (!st) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
        return 1;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const int32_t iters32 = static_cast<int32_t>(bench_iters);
    if (!write_exact("WVB1", 4) || !write_exact(&ns, 1) || !write_exact(&iters32, 1)) return 2;
  } else {
    auto st = once();
    if (!st) {
      std::fprintf(stderr, "welvet_dense_fwd: %s\n", st.message.c_str());
      return 1;
    }
  }
  if (!write_magic_local("PPBO") || !write_exact(&loss, 1) || !write_exact(&branches, 1)) return 2;
  for (int i = 0; i < branches; ++i) {
    const int32_t nw = welvet::stack::child_grad_w_size(layer.branches[static_cast<std::size_t>(i)]);
    std::vector<float> Wa(static_cast<std::size_t>(nw > 0 ? nw : 0));
    if (nw > 0) {
      auto est = welvet::stack::child_export_f32(layer.branches[static_cast<std::size_t>(i)],
                                                Wa.data());
      if (!est) {
        std::fprintf(stderr, "welvet_dense_fwd: %s\n", est.message.c_str());
        return 1;
      }
    }
    if (!write_exact(&nw, 1)) return 2;
    if (nw > 0 && !write_exact(Wa.data(), Wa.size())) return 2;
  }
  return 0;
}

int run_normalbp_child(int bench_iters) {
  ChildReader initial;
  welvet::stack::Child child;
  if (!read_child(initial, &child)) return 2;
  auto read_tensor = [](welvet::TensorF32* out) {
    int32_t rank = 0;
    if (!read_exact(&rank, 1) || rank <= 0 || rank > 8) return false;
    std::vector<int32_t> shape(rank);
    if (!read_exact(shape.data(), shape.size())) return false;
    size_t count = 1;
    for (int d : shape) {
      if (d <= 0 || count > size_t(1 << 26) / d) return false;
      count *= static_cast<size_t>(d);
    }
    out->shape.assign(shape.begin(), shape.end());
    out->data.resize(count);
    return read_exact(out->data.data(), count);
  };
  welvet::TensorF32 input, target, pre, post, dx, dw;
  if (!read_tensor(&input) || !read_tensor(&target)) return 2;
  double lr = 0, loss = 0;
  if (!read_exact(&lr, 1)) return 2;
  auto once = [&]() -> welvet::Status {
    ChildReader replay(&initial.bytes);
    child = welvet::stack::Child{};
    if (!read_child(replay, &child)) {
      return welvet::Status::Err(welvet::StatusCode::InvalidArgument, "NormalBP child replay");
    }
    if (auto st = welvet::stack::child_forward(child, input, &pre, &post); !st) return st;
    if (auto st = welvet::runtime::training::mse(post, target, &loss); !st) return st;
    welvet::TensorF32 gy;
    if (auto st = welvet::runtime::training::mse_grad(post, target, &gy); !st) return st;
    if (auto st = welvet::stack::child_backward(child, gy, input, pre, &dx, &dw); !st) return st;
    return welvet::stack::child_apply_sgd(&child, dw, lr);
  };
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i < (bench_iters > 0 ? bench_iters : 1); ++i) {
    if (auto st = once(); !st) {
      std::fprintf(stderr, "NormalBP child: %s\n", st.message.c_str());
      return 1;
    }
  }
  if (bench_iters > 0) {
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - begin)
                     .count();
    int32_t iters = bench_iters;
    if (!write_magic_local("WVB1") || !write_exact(&ns, 1) || !write_exact(&iters, 1)) return 2;
  }
  std::vector<float> weights(static_cast<std::size_t>(
      std::max(0, welvet::stack::child_grad_w_size(child))));
  if (!weights.empty()) {
    if (auto st = welvet::stack::child_export_f32(child, weights.data()); !st) return 1;
  }
  return write_cnn_nbp_leaf_out("NBLO", pre.data, post.data, dx.data, dw.data, weights, loss);
}

