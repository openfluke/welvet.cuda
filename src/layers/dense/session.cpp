#include "welvet/dense_session.hpp"

#include "welvet/cuda/matvec.hpp"
#include "welvet/dense.hpp"
#include "welvet/weights.hpp"

#include <string>
#include <vector>

namespace welvet::dense {

struct Session {
  cuda::ResidentDense gpu;
  WeightStore weights;
  Activation act = Activation::Linear;
  int32_t batch = 0;
  int32_t in = 0;
  int32_t out = 0;
};

static Status decode_W_f64(const WeightStore& W, std::vector<double>* out) {
  out->resize(static_cast<std::size_t>(W.rows) * static_cast<std::size_t>(W.cols));
  std::vector<double> row(static_cast<std::size_t>(W.cols));
  for (int r = 0; r < W.rows; ++r) {
    if (auto st = W.decode_row_f64(r, row.data()); !st) return st;
    for (int c = 0; c < W.cols; ++c) {
      (*out)[static_cast<std::size_t>(r) * static_cast<std::size_t>(W.cols) +
             static_cast<std::size_t>(c)] = row[static_cast<std::size_t>(c)];
    }
  }
  return Status::Ok();
}

Status session_mount(int32_t batch, int32_t in, int32_t out, Activation act, DType dt,
                     const float* W_init, const double* bias, size_t bias_len, Session** sess) {
  if (!sess || !W_init) return Status::Err(StatusCode::InvalidArgument, "null mount args");
  if (!dtype_supported_cuda_forward(dt)) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("dense CUDA unsupported dtype ") + std::string(dtype_name(dt)));
  }

  auto* s = new Session();
  s->batch = batch;
  s->in = in;
  s->out = out;
  s->act = act;
  if (auto st = WeightStore::make(out, in, dt, W_init, &s->weights); !st) {
    delete s;
    return st;
  }
  if (bias && bias_len > 0) {
    s->weights.bias.assign(bias, bias + bias_len);
  }

  std::vector<double> W64;
  if (auto st = decode_W_f64(s->weights, &W64); !st) {
    delete s;
    return st;
  }
  const int bl = static_cast<int>(s->weights.bias.size());
  if (auto st = cuda::resident_mount(&s->gpu, W64.data(), bl ? s->weights.bias.data() : nullptr, bl,
                                     batch, out, in, act);
      !st) {
    delete s;
    return st;
  }
  *sess = s;
  return Status::Ok();
}

Status session_execute(Session* s, const float* x, float* pre, float* post) {
  if (!s || !x || !pre || !post) {
    return Status::Err(StatusCode::InvalidArgument, "null execute args");
  }
  return cuda::resident_execute(&s->gpu, x, pre, post);
}

void session_release(Session* s) {
  if (!s) return;
  cuda::resident_release(&s->gpu);
  delete s;
}

} // namespace welvet::dense
