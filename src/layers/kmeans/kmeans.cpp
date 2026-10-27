#include "welvet/kmeans.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::kmeans {
namespace {

struct LayInfo {
  int batch = 0;
  int d = 0;
  int k = 0;
  std::vector<int64_t> shape;
};

Status parse_in(const Config& cfg, const TensorF32& input, LayInfo* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: empty input");
  }
  const int d = cfg.feature_dim;
  if (input.shape.empty() || static_cast<int>(input.shape.back()) != d) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: last dim want FeatureDim");
  }
  int batch = 1;
  for (std::size_t i = 0; i + 1 < input.shape.size(); ++i) {
    batch *= static_cast<int>(input.shape[i]);
  }
  if (batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: invalid batch");
  }
  const int want = batch * d;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: data short");
  }
  lay->batch = batch;
  lay->d = d;
  lay->k = cfg.num_clusters;
  lay->shape = input.shape;
  return Status::Ok();
}

Status flatten_w_f32(const WeightStore& ws, std::vector<float>* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil w");
  out->resize(static_cast<std::size_t>(ws.rows * ws.cols));
  std::vector<double> row(static_cast<std::size_t>(ws.cols));
  for (int r = 0; r < ws.rows; ++r) {
    if (auto st = ws.decode_row_f64(r, row.data()); !st) return st;
    for (int c = 0; c < ws.cols; ++c) {
      (*out)[static_cast<std::size_t>(r * ws.cols + c)] =
          static_cast<float>(row[static_cast<std::size_t>(c)]);
    }
  }
  return Status::Ok();
}

std::vector<double> soft_max(const std::vector<double>& logits) {
  double max_z = logits[0];
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > max_z) max_z = logits[i];
  }
  std::vector<double> out(logits.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const double e = std::exp(logits[i] - max_z);
    out[i] = e;
    sum += e;
  }
  for (double& v : out) v /= sum;
  return out;
}

// Go softMaxBwd: y_i * (gy_i - ⟨gy, y⟩).
std::vector<double> soft_max_bwd(const std::vector<double>& gy, const std::vector<double>& y) {
  double dot = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    dot += gy[i] * y[i];
  }
  std::vector<double> out(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    out[i] = y[i] * (gy[i] - dot);
  }
  return out;
}

double temp_of(const Config& cfg) { return cfg.temperature == 0.0 ? 1.0 : cfg.temperature; }

} // namespace

int out_dim(const Config& cfg) {
  return cfg.output_mode == OutputMode::Features ? cfg.feature_dim : cfg.num_clusters;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil config");
  if (cfg->num_clusters <= 0 || cfg->feature_dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: need positive NumClusters/FeatureDim");
  }
  if (cfg->temperature < 0.0) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: Temperature < 0");
  }
  if (cfg->output_mode != OutputMode::Probabilities && cfg->output_mode != OutputMode::Features) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: unknown OutputMode");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* init_or_null, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "kmeans: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(cfg.feature_dim, cfg.num_clusters, Activation::Linear, dt,
                                 init_or_null, &layer.centers);
      !st) {
    return Status::Err(st.code, std::string("kmeans centers: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  LayInfo lay;
  if (auto st = parse_in(cfg, input, &lay); !st) return st;

  // Go materializes native centers as Float32 before distances.
  std::vector<float> w(static_cast<std::size_t>(lay.k * lay.d));
  if (auto st = layer.centers.weights.export_f32(w.data()); !st) {
    return Status::Err(st.code, std::string("kmeans weights: ") + st.message);
  }

  const double temp = temp_of(cfg);
  const double temp_sq = temp * temp;
  const int od = out_dim(cfg);

  pre->shape = lay.shape;
  if (!pre->shape.empty()) pre->shape.back() = od;
  const int out_n = lay.batch * od;
  pre->data.assign(static_cast<std::size_t>(out_n), 0.0f);
  post->shape = pre->shape;
  post->data.assign(static_cast<std::size_t>(out_n), 0.0f);

  for (int b = 0; b < lay.batch; ++b) {
    const int x_off = b * lay.d;
    std::vector<double> logits(static_cast<std::size_t>(lay.k));
    for (int k = 0; k < lay.k; ++k) {
      double sq = 0.0;
      const int c_off = k * lay.d;
      for (int d = 0; d < lay.d; ++d) {
        const double diff =
            static_cast<double>(input.data[static_cast<std::size_t>(x_off + d)]) -
            static_cast<double>(w[static_cast<std::size_t>(c_off + d)]);
        sq += diff * diff;
      }
      logits[static_cast<std::size_t>(k)] = -sq / (2.0 * temp_sq);
    }
    const auto a = soft_max(logits);
    const int o_off = b * od;
    if (cfg.output_mode == OutputMode::Features) {
      for (int d = 0; d < lay.d; ++d) {
        double v = 0.0;
        for (int k = 0; k < lay.k; ++k) {
          v += a[static_cast<std::size_t>(k)] *
               static_cast<double>(w[static_cast<std::size_t>(k * lay.d + d)]);
        }
        // Match Go: store pre as f32, then Activate(pre) — not activate(full f64).
        const float pre_f = static_cast<float>(v);
        pre->data[static_cast<std::size_t>(o_off + d)] = pre_f;
        post->data[static_cast<std::size_t>(o_off + d)] =
            static_cast<float>(activate_f64(static_cast<double>(pre_f), cfg.activation));
      }
    } else {
      for (int k = 0; k < lay.k; ++k) {
        const double v = a[static_cast<std::size_t>(k)];
        const float pre_f = static_cast<float>(v);
        pre->data[static_cast<std::size_t>(o_off + k)] = pre_f;
        post->data[static_cast<std::size_t>(o_off + k)] =
            static_cast<float>(activate_f64(static_cast<double>(pre_f), cfg.activation));
      }
    }
  }
  return Status::Ok();
}

int grad_w_size(const Layer& layer) {
  return layer.cfg.num_clusters * layer.cfg.feature_dim;
}

Status backward_f32(Layer& layer, const TensorF32& grad_out, const TensorF32& input,
                    const TensorF32& pre, TensorF32* grad_in, TensorF32* grad_w) {
  if (!grad_in || !grad_w) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: nil grad out");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;

  LayInfo lay;
  if (auto st = parse_in(cfg, input, &lay); !st) return st;
  if (grad_out.data.empty() || pre.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: nil gradOut/pre");
  }

  // Same export path as forward (Go FlattenF32).
  std::vector<float> w(static_cast<std::size_t>(lay.k * lay.d));
  if (auto st = layer.centers.weights.export_f32(w.data()); !st) {
    return Status::Err(st.code, std::string("kmeans weights: ") + st.message);
  }

  const double temp = temp_of(cfg);
  const double temp_sq = temp * temp;
  const int od = out_dim(cfg);
  const Activation act = cfg.activation;

  std::vector<double> g_in_acc(static_cast<std::size_t>(lay.batch * lay.d), 0.0);
  std::vector<double> g_w_acc(static_cast<std::size_t>(lay.k * lay.d), 0.0);

  for (int b = 0; b < lay.batch; ++b) {
    const int x_off = b * lay.d;
    const int o_off = b * od;

    std::vector<double> logits(static_cast<std::size_t>(lay.k));
    for (int k = 0; k < lay.k; ++k) {
      double sq = 0.0;
      const int c_off = k * lay.d;
      for (int d = 0; d < lay.d; ++d) {
        const double diff =
            static_cast<double>(input.data[static_cast<std::size_t>(x_off + d)]) -
            static_cast<double>(w[static_cast<std::size_t>(c_off + d)]);
        sq += diff * diff;
      }
      logits[static_cast<std::size_t>(k)] = -sq / (2.0 * temp_sq);
    }
    const auto a = soft_max(logits);

    // Go: AsFloat64(gradOut) * AsFloat64(ActivateDeriv(pre, act)) —
    // ActivateDeriv returns f32, so round the derivative through float32.
    std::vector<double> gy(static_cast<std::size_t>(od));
    for (int i = 0; i < od; ++i) {
      const double g =
          static_cast<double>(grad_out.data[static_cast<std::size_t>(o_off + i)]);
      const float deriv_f = static_cast<float>(activate_deriv_f64(
          static_cast<double>(pre.data[static_cast<std::size_t>(o_off + i)]), act));
      gy[static_cast<std::size_t>(i)] = g * static_cast<double>(deriv_f);
    }

    std::vector<double> grad_logits;
    if (cfg.output_mode == OutputMode::Features) {
      std::vector<double> gy_a(static_cast<std::size_t>(lay.k));
      for (int k = 0; k < lay.k; ++k) {
        double dot = 0.0;
        for (int d = 0; d < lay.d; ++d) {
          dot += gy[static_cast<std::size_t>(d)] *
                 static_cast<double>(w[static_cast<std::size_t>(k * lay.d + d)]);
        }
        gy_a[static_cast<std::size_t>(k)] = dot;
      }
      grad_logits = soft_max_bwd(gy_a, a);
      for (int k = 0; k < lay.k; ++k) {
        for (int d = 0; d < lay.d; ++d) {
          g_w_acc[static_cast<std::size_t>(k * lay.d + d)] +=
              a[static_cast<std::size_t>(k)] * gy[static_cast<std::size_t>(d)];
        }
      }
    } else {
      grad_logits = soft_max_bwd(gy, a);
    }

    for (int k = 0; k < lay.k; ++k) {
      const int c_off = k * lay.d;
      for (int d = 0; d < lay.d; ++d) {
        const double diff =
            static_cast<double>(input.data[static_cast<std::size_t>(x_off + d)]) -
            static_cast<double>(w[static_cast<std::size_t>(c_off + d)]);
        // Match Go: (gradLogits[k] * diff) / tempSq (-ffp-contract=off).
        const double g =
            (grad_logits[static_cast<std::size_t>(k)] * diff) / temp_sq;
        g_w_acc[static_cast<std::size_t>(c_off + d)] += g;
        g_in_acc[static_cast<std::size_t>(x_off + d)] -= g;
      }
    }
  }

  grad_in->shape = lay.shape;
  grad_in->data.resize(static_cast<std::size_t>(lay.batch * lay.d));
  for (std::size_t i = 0; i < g_in_acc.size(); ++i) {
    grad_in->data[i] = static_cast<float>(g_in_acc[i]);
  }
  grad_w->shape = {lay.k, lay.d};
  grad_w->data.resize(static_cast<std::size_t>(lay.k * lay.d));
  for (std::size_t i = 0; i < g_w_acc.size(); ++i) {
    grad_w->data[i] = static_cast<float>(g_w_acc[i]);
  }
  return Status::Ok();
}

Status apply_grad_sgd(Layer* layer, const TensorF32& dW, double lr) {
  if (!layer) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: ApplyGradSGD nil");
  }
  return dense::apply_grad_sgd(&layer->centers, dW, lr);
}

Status export_packed_f32(const Layer& layer, float* dst) {
  if (!dst) {
    return Status::Err(StatusCode::InvalidArgument, "kmeans: nil dst");
  }
  return layer.centers.weights.export_f32(dst);
}

} // namespace welvet::kmeans
