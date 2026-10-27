#include "welvet/mha.hpp"

#include "welvet/cuda/mha.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace welvet::mha {
namespace {

struct Layout {
  int batch = 1;
  int seq_len = 0;
  int d_model = 0;
};

Status parse_layout(int d_model, const TensorF32& input, Layout* lay) {
  if (!lay || input.data.empty()) {
    return Status::Err(StatusCode::InvalidArgument, "mha: empty input");
  }
  lay->d_model = d_model;
  if (input.shape.size() == 3) {
    lay->batch = static_cast<int>(input.shape[0]);
    lay->seq_len = static_cast<int>(input.shape[1]);
    if (static_cast<int>(input.shape[2]) != d_model) {
      return Status::Err(StatusCode::InvalidArgument, "mha: last dim != d_model");
    }
  } else if (input.shape.size() == 2) {
    lay->batch = 1;
    if (static_cast<int>(input.shape[1]) != d_model) {
      return Status::Err(StatusCode::InvalidArgument, "mha: width != d_model");
    }
    lay->seq_len = static_cast<int>(input.shape[0]);
  } else {
    return Status::Err(StatusCode::InvalidArgument, "mha: need [batch,seq,d] or [seq,d]");
  }
  if (lay->batch <= 0 || lay->seq_len <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: bad batch/seq");
  }
  const int want = lay->batch * lay->seq_len * lay->d_model;
  if (static_cast<int>(input.data.size()) < want) {
    return Status::Err(StatusCode::InvalidArgument, "mha: short data");
  }
  return Status::Ok();
}

} // namespace

double Config::scale() const {
  if (scale_override > 0) return scale_override;
  return 1.0 / std::sqrt(static_cast<double>(head_dim));
}

Status validate(Config* cfg) {
  if (!cfg) return Status::Err(StatusCode::InvalidArgument, "mha: nil config");
  if (cfg->d_model <= 0 || cfg->num_heads <= 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: need DModel>0 and NumHeads>0");
  }
  if (cfg->head_dim <= 0) {
    if (cfg->d_model % cfg->num_heads != 0) {
      return Status::Err(StatusCode::InvalidArgument, "mha: HeadDim unset and DModel%NumHeads!=0");
    }
    cfg->head_dim = cfg->d_model / cfg->num_heads;
  }
  if (cfg->num_kv_heads <= 0) cfg->num_kv_heads = cfg->num_heads;
  if (cfg->num_heads % cfg->num_kv_heads != 0) {
    return Status::Err(StatusCode::InvalidArgument, "mha: NumHeads not divisible by NumKVHeads");
  }
  if (cfg->max_seq_len <= 0) cfg->max_seq_len = 512;
  if (cfg->mask == MaskKind::Unspecified) {
    cfg->mask = MaskKind::Causal;
    cfg->causal = true;
  }
  if (cfg->mask == MaskKind::Causal) cfg->causal = true;
  if (cfg->rope_theta <= 0) cfg->rope_theta = 10000;
  if (cfg->softmax != SoftmaxKind::Standard && cfg->softmax != SoftmaxKind::Sigmoid) {
    return Status::Err(StatusCode::InvalidArgument, "mha: unknown Softmax");
  }
  return Status::Ok();
}

Config decoder_causal(int d_model, int num_heads, int num_kv_heads) {
  Config c;
  c.d_model = d_model;
  c.num_heads = num_heads;
  c.num_kv_heads = num_kv_heads;
  c.mask = MaskKind::Causal;
  c.pos = PosKind::RoPE;
  c.mode = AttnMode::Self;
  c.causal = true;
  return c;
}

Status new_configured(const Config& cfg_in, DType dt, const float* q_init, const float* k_init,
                      const float* v_init, const float* o_init, Layer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "mha: nil out");
  Config cfg = cfg_in;
  if (auto st = validate(&cfg); !st) return st;

  const int d = cfg.d_model;
  const int q_dim = cfg.q_dim();
  const int kv_dim = cfg.kv_dim();

  Layer layer;
  layer.cfg = cfg;
  if (auto st = dense::new_layer(d, q_dim, Activation::Linear, dt, q_init, &layer.Q); !st) {
    return Status::Err(st.code, std::string("mha Q: ") + st.message);
  }
  if (auto st = dense::new_layer(d, kv_dim, Activation::Linear, dt, k_init, &layer.K); !st) {
    return Status::Err(st.code, std::string("mha K: ") + st.message);
  }
  if (auto st = dense::new_layer(d, kv_dim, Activation::Linear, dt, v_init, &layer.V); !st) {
    return Status::Err(st.code, std::string("mha V: ") + st.message);
  }
  if (auto st = dense::new_layer(q_dim, d, Activation::Linear, dt, o_init, &layer.O); !st) {
    return Status::Err(st.code, std::string("mha O: ") + st.message);
  }
  *out = std::move(layer);
  return Status::Ok();
}

Status forward(Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "mha: nil pre/post");
  }
  Config cfg = layer.cfg;
  if (auto st = validate(&cfg); !st) return st;
  if (cfg.mode != AttnMode::Self) {
    return Status::Gap("mha forward: ModeCross not in baseline");
  }

  Layout lay;
  if (auto st = parse_layout(cfg.d_model, input, &lay); !st) return st;

  const int q_dim = cfg.q_dim();
  const int d = cfg.d_model;
  const int bs = lay.batch * lay.seq_len;

  if (lay.seq_len > cfg.max_seq_len) {
    return Status::Err(StatusCode::InvalidArgument, "mha: seq > MaxSeqLen");
  }

  TensorF32 flat_in;
  flat_in.shape = {bs, d};
  flat_in.data.assign(input.data.begin(), input.data.begin() + static_cast<std::size_t>(bs * d));

  TensorF32 q_pre, q_post, k_pre, k_post, v_pre, v_post;
  if (auto st = dense::forward_f32(layer.Q, flat_in, &q_pre, &q_post); !st) {
    return Status::Err(st.code, std::string("mha Q: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.K, flat_in, &k_pre, &k_post); !st) {
    return Status::Err(st.code, std::string("mha K: ") + st.message);
  }
  if (auto st = dense::forward_f32(layer.V, flat_in, &v_pre, &v_post); !st) {
    return Status::Err(st.code, std::string("mha V: ") + st.message);
  }

  TensorF32 attn_flat;
  attn_flat.shape = {bs, q_dim};
  attn_flat.data.resize(static_cast<std::size_t>(bs * q_dim));

  if (auto st = cuda::launch_mha_attention_forward(
          q_post.data.data(), k_post.data.data(), v_post.data.data(), attn_flat.data.data(),
          lay.batch, lay.seq_len, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.max_seq_len,
          static_cast<int>(cfg.mask), static_cast<int>(cfg.pos), static_cast<int>(cfg.softmax),
          cfg.window, cfg.prefix_len, int(cfg.window_causal || cfg.causal), cfg.rope_theta,
          cfg.scale_override);
      !st) {
    return st;
  }

  TensorF32 o_pre, o_post;
  if (auto st = dense::forward_f32(layer.O, attn_flat, &o_pre, &o_post); !st) {
    return Status::Err(st.code, std::string("mha O: ") + st.message);
  }

  pre->shape = {lay.batch, lay.seq_len, q_dim};
  pre->data = std::move(attn_flat.data);
  post->shape = {lay.batch, lay.seq_len, d};
  post->data = std::move(o_post.data);
  return Status::Ok();
}

} // namespace welvet::mha
