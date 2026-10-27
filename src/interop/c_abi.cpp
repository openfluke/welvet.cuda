#include "welvet_c.h"

#include "welvet/cuda/device.hpp"
#include "welvet/dense.hpp"
#include "welvet/dense_session.hpp"
#include "welvet/gaps.hpp"
#include "welvet/status.hpp"
#include "welvet/tensor.hpp"

#include <cstring>
#include <string>

namespace {

welvet_status to_c(const welvet::Status& st) {
  welvet_status out{};
  out.code = static_cast<int32_t>(st.code);
  std::strncpy(out.message, st.message.c_str(), sizeof(out.message) - 1);
  out.message[sizeof(out.message) - 1] = '\0';
  return out;
}

} // namespace

extern "C" {

const char* welvet_version(void) {
  // M1 Dense FormatNone×34 forward signed (w2a exact_bit + speed) → board 13/100.
  return "0.13";
}

const char* welvet_runtime_identity(void) {
  static thread_local std::string id;
  id = welvet::cuda::runtime_identity();
  return id.c_str();
}

welvet_status welvet_cuda_init(void) { return to_c(welvet::cuda::init_default_device()); }

int32_t welvet_cuda_available(void) { return welvet::cuda::available() ? 1 : 0; }

int32_t welvet_gaps_count(void) { return static_cast<int32_t>(welvet::gaps::count_gaps()); }

int32_t welvet_gaps_done(void) { return static_cast<int32_t>(welvet::gaps::count_done()); }

welvet_status welvet_dense_mount(int32_t batch, int32_t in, int32_t out, int32_t act, int32_t dtype,
                                 const float* W_init, const double* bias, size_t bias_len,
                                 welvet_dense_session** sess_out) {
  if (auto st = welvet::cuda::init_default_device(); !st) return to_c(st);
  welvet::dense::Session* sess = nullptr;
  auto st = welvet::dense::session_mount(batch, in, out, static_cast<welvet::Activation>(act),
                                         static_cast<welvet::DType>(dtype), W_init, bias, bias_len,
                                         &sess);
  if (!st) return to_c(st);
  *sess_out = reinterpret_cast<welvet_dense_session*>(sess);
  return to_c(welvet::Status::Ok());
}

welvet_status welvet_dense_execute(welvet_dense_session* sess, const float* x, float* pre,
                                   float* post) {
  return to_c(welvet::dense::session_execute(reinterpret_cast<welvet::dense::Session*>(sess), x, pre,
                                             post));
}

void welvet_dense_release(welvet_dense_session* sess) {
  welvet::dense::session_release(reinterpret_cast<welvet::dense::Session*>(sess));
}

welvet_status welvet_dense_forward_f32(int32_t batch, int32_t in, int32_t out, int32_t act,
                                       int32_t dtype, int32_t /*backend*/, const float* W_init,
                                       const double* bias, size_t bias_len, const float* x,
                                       float* pre, float* post) {
  if (auto st = welvet::cuda::init_default_device(); !st) return to_c(st);

  welvet::dense::Layer layer;
  auto st = welvet::dense::new_layer(in, out, static_cast<welvet::Activation>(act),
                                     static_cast<welvet::DType>(dtype), W_init, &layer);
  if (!st) return to_c(st);
  if (bias && bias_len > 0) {
    layer.weights.bias.assign(bias, bias + bias_len);
  }

  welvet::TensorF32 xin = welvet::TensorF32::zeros(batch, in);
  std::memcpy(xin.data.data(), x, static_cast<std::size_t>(batch * in) * sizeof(float));
  welvet::TensorF32 pre_t, post_t;
  st = welvet::dense::forward_f32(layer, xin, &pre_t, &post_t);
  if (!st) return to_c(st);
  std::memcpy(pre, pre_t.data.data(), pre_t.data.size() * sizeof(float));
  std::memcpy(post, post_t.data.data(), post_t.data.size() * sizeof(float));
  return to_c(welvet::Status::Ok());
}

} // extern "C"
