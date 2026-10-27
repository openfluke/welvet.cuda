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

int run_volume_mamba_normalbp(int bench_iters) {
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
  return write_cnn_vol_nbp_out("MBVO", loss, bwd, fwd, Ws);
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

int run_volume_metacognition_normalbp(int bench_iters) {
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
  return write_cnn_vol_nbp_out("MCVO", loss, bwd, fwd, Ws);
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

int run_volume_kmeans_normalbp(int bench_iters) {
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
  return write_cnn_vol_nbp_out("KMVO", loss, bwd, fwd, Ws);
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

int run_volume_gdn_normalbp(int bench_iters) {
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
  return write_cnn_vol_nbp_out("GDVO", loss, bwd, fwd, Ws);
}

// ---------------------------------------------------------------------------
