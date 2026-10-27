#include "welvet/layernorm.hpp"
#include "welvet/cuda/device.hpp"
#include "welvet/cuda/norm.hpp"

#include <string>
#include <vector>

namespace welvet::layernorm {
namespace {

struct Layout {
  int batch = 1;
  int seq = 1;
  int dim = 0;
};

Status parse_layout(int dim, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: empty input");
  }
  lay->dim = dim;
  if (input.shape.size() == 2) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = 1;
    if (static_cast<int>(input.shape[1]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "layernorm: width != Dim");
    }
  } else if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != dim) {
      return Status::Err(StatusCode::InvalidArgument, "layernorm: last dim != Dim");
    }
  } else {
    return Status::Err(StatusCode::InvalidArgument,
                       "layernorm: need [batch,dim] or [batch,seq,dim]");
  }
  if (lay->batch <= 0 || lay->seq <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: bad batch/seq");
  }
  const int want = lay->batch * lay->seq * dim;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: short data");
  }
  return Status::Ok();
}

} // namespace

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "layernorm: nil config");
  if (cfg->dim <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: need Dim>0");
  }
  if (cfg->eps <= 0.0) {
    cfg->eps = 1e-5;
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* gamma_init,
                      const float* beta_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "layernorm: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;
  if (!dtype_implemented_forward(dt)) {
    return Status::Err(StatusCode::Unsupported, std::string("layernorm CUDA unsupported dtype ") +
                                                    std::string(dtype_name(dt)));
  }

  Layer layer;
  layer.cfg = cfg;
  if (auto st = WeightStore::make(1, cfg.dim, dt, gamma_init, &layer.gamma); !st) {
    return Status::Err(st.code, std::string("layernorm gamma: ") + st.message);
  }
  if (auto st = WeightStore::make(1, cfg.dim, dt, beta_init, &layer.beta); !st) {
    return Status::Err(st.code, std::string("layernorm beta: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "layernorm: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg.dim, input, &lay); !st) return st;

  const int dim = cfg.dim;
  const int n_tok = lay.batch * lay.seq;

  std::vector<double> g(static_cast<std::size_t>(dim));
  std::vector<double> b(static_cast<std::size_t>(dim));
  if (auto st = layer.gamma.decode_row_f64(0, g.data()); !st) {
    return Status::Err(st.code, std::string("layernorm gamma: ") + st.message);
  }
  if (auto st = layer.beta.decode_row_f64(0, b.data()); !st) {
    return Status::Err(st.code, std::string("layernorm beta: ") + st.message);
  }

  if (lay.seq == 1) {
    pre->shape = {lay.batch, dim};
    post->shape = {lay.batch, dim};
  } else {
    pre->shape = {lay.batch, lay.seq, dim};
    post->shape = {lay.batch, lay.seq, dim};
  }
  pre->data.resize(static_cast<std::size_t>(n_tok * dim));
  post->data.resize(static_cast<std::size_t>(n_tok * dim));

  if (auto st = cuda::init_default_device(); !st) return st;
  return cuda::launch_norm_forward(input.data.data(), g.data(), b.data(), pre->data.data(),
                                   post->data.data(), n_tok, dim, cfg.eps, /*center=*/1);
}

} // namespace welvet::layernorm
