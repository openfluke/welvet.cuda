#pragma once

#include <cstddef>
#include <cstdint>

#include "welvet/activate.hpp"
#include "welvet/dtype.hpp"
#include "welvet/status.hpp"

namespace welvet::dense {

struct Session;

Status session_mount(int32_t batch, int32_t in, int32_t out, Activation act, DType dt,
                     const float* W_init, const double* bias, size_t bias_len, Session** sess);

Status session_execute(Session* s, const float* x, float* pre, float* post);

void session_release(Session* s);

} // namespace welvet::dense
