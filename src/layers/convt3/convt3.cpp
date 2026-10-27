#include "welvet/convt3.hpp"

#include "welvet/cuda/conv.hpp"

#include <string>
#include <vector>

namespace welvet::convt3 {
namespace {

Status flatten_w_f32(const WeightStore& ws, std::vector<float>* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "convt3: nil w");
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

struct Layout {
  int batch = 0, in_c = 0, in_d = 0, in_h = 0, in_w = 0, filters = 0, out_d = 0, out_h = 0,
      out_w = 0, k = 0, stride = 0, padding = 0;
};

Status parse_layout(const Config& cfg, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: empty input");
  }
  if (input.shape.size() != 5) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: need [batch,inC,D,H,W]");
  }
  lay->batch = static_cast<int>(input.shape[0]);
  lay->in_c = cfg.in_channels;
  lay->in_d = cfg.depth;
  lay->in_h = cfg.height;
  lay->in_w = cfg.width;
  lay->filters = cfg.filters;
  lay->out_d = out_d(cfg);
  lay->out_h = out_h(cfg);
  lay->out_w = out_w(cfg);
  lay->k = cfg.kernel;
  lay->stride = cfg.stride;
  lay->padding = cfg.padding;
  if (static_cast<int>(input.shape[1]) != lay->in_c ||
      static_cast<int>(input.shape[2]) != lay->in_d ||
      static_cast<int>(input.shape[3]) != lay->in_h ||
      static_cast<int>(input.shape[4]) != lay->in_w) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: shape mismatch");
  }
  if (lay->batch <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: bad batch");
  }
  const int want = lay->batch * lay->in_c * lay->in_d * lay->in_h * lay->in_w;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: short data");
  }
  return Status::Ok();
}

} // namespace

int output_spatial(int spatial, int kernel, int stride, int padding, int output_padding) {
  if (stride <= 0) stride = 1;
  return (spatial - 1) * stride - 2 * padding + kernel + output_padding;
}

int out_d(const Config& cfg) {
  return output_spatial(cfg.depth, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}
int out_h(const Config& cfg) {
  return output_spatial(cfg.height, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}
int out_w(const Config& cfg) {
  return output_spatial(cfg.width, cfg.kernel, cfg.stride, cfg.padding, cfg.output_padding);
}

int patch_dim(const Config& cfg) {
  return cfg.in_channels * cfg.kernel * cfg.kernel * cfg.kernel;
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "convt3: nil config");
  if (cfg->in_channels <= 0 || cfg->filters <= 0 || cfg->depth <= 0 || cfg->height <= 0 ||
      cfg->width <= 0 || cfg->kernel <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: need positive dims");
  }
  if (cfg->stride <= 0) cfg->stride = 1;
  if (cfg->padding < 0 || cfg->output_padding < 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: Padding/OutputPadding < 0");
  }
  if (cfg->output_padding >= cfg->stride) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: OutputPadding must be < Stride");
  }
  if (out_d(*cfg) <= 0 || out_h(*cfg) <= 0 || out_w(*cfg) <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: bad output spatial");
  }
  return Status::Ok();
}

Status new_configured(const Config& cfg_in, DType dt, const float* w_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "convt3: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(patch_dim(cfg), cfg.filters, cfg.activation, dt, w_init,
                                 &layer.proj);
      !st) {
    return Status::Err(st.code, std::string("convt3 proj: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  layer.cfg = cfg;

  Layout lay;
  if (auto st = parse_layout(cfg, input, &lay); !st) return st;

  std::vector<float> w;
  if (auto st = flatten_w_f32(layer.proj.weights, &w); !st) {
    return Status::Err(st.code, std::string("convt3 weights: ") + st.message);
  }
  const int kkk = lay.k * lay.k * lay.k;
  const int want_w = lay.filters * lay.in_c * kkk;
  if (static_cast<int>(w.size()) < want_w) {
    return Status::Err(StatusCode::InvalidArgument, "convt3: weight short");
  }

  const int sp_out = lay.out_d * lay.out_h * lay.out_w;
  const int n_out = lay.batch * lay.filters * sp_out;
  pre->shape = {lay.batch, lay.filters, lay.out_d, lay.out_h, lay.out_w};
  post->shape = {lay.batch, lay.filters, lay.out_d, lay.out_h, lay.out_w};
  pre->data.resize(static_cast<std::size_t>(n_out));
  post->data.resize(static_cast<std::size_t>(n_out));

  return cuda::launch_convt3_forward(input.data.data(), w.data(), pre->data.data(),
                                     post->data.data(), lay.batch, lay.in_c, lay.filters, lay.in_d,
                                     lay.in_h, lay.in_w, lay.out_d, lay.out_h, lay.out_w, lay.k,
                                     lay.stride, lay.padding, cfg.activation);
}

} // namespace welvet::convt3
