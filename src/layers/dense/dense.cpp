#include "welvet/dense.hpp"

#include <string>
#include <vector>

namespace welvet::dense {

bool dtype_supported_cuda_forward(DType dt) { return dtype_implemented_forward(dt); }

Status new_layer(int in_h, int out_h, Activation act, DType dt, const float* init_or_null,
                 Layer* layer) {
  if (!layer) return Status::Err(StatusCode::InvalidArgument, "null Layer");
  if (in_h <= 0 || out_h <= 0) return Status::Err(StatusCode::InvalidArgument, "bad dims");
  if (!dtype_supported_cuda_forward(dt)) {
    return Status::Err(StatusCode::Unsupported,
                       std::string("dense CUDA unsupported dtype ") + std::string(dtype_name(dt)));
  }

  Layer L;
  L.input_height = in_h;
  L.output_height = out_h;
  L.activation = act;
  if (auto st = WeightStore::make(out_h, in_h, dt, init_or_null, &L.weights); !st) return st;
  *layer = std::move(L);
  return Status::Ok();
}

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

Status sync_gpu_from_weights(const Layer& layer, int batch) {
  if (batch <= 0) return Status::Err(StatusCode::InvalidArgument, "bad batch");
  std::vector<double> W64;
  if (auto st = decode_W_f64(layer.weights, &W64); !st) return st;
  const int bl = static_cast<int>(layer.weights.bias.size());
  const double* bias = bl > 0 ? layer.weights.bias.data() : nullptr;
  if (auto st = cuda::resident_mount(const_cast<cuda::ResidentDense*>(&layer.gpu), W64.data(), bias,
                                     bl, batch, layer.output_height, layer.input_height,
                                     layer.activation);
      !st) {
    return st;
  }
  layer.gpu_valid = true;
  layer.gpu_batch = batch;
  return Status::Ok();
}

Status forward_f32(const Layer& layer, const TensorF32& input, TensorF32* pre, TensorF32* post) {
  if (!pre || !post) return Status::Err(StatusCode::InvalidArgument, "dense: nil out");
  if (!dtype_supported_cuda_forward(layer.weights.dtype)) {
    return Status::Err(StatusCode::Unsupported, std::string("dense CUDA unsupported dtype ") +
                                                    std::string(dtype_name(layer.weights.dtype)));
  }
  if (input.shape.size() != 2) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input shape need [batch,in]");
  }
  const int batch = static_cast<int>(input.shape[0]);
  const int in = layer.input_height;
  const int out = layer.output_height;
  if (batch <= 0 || input.data.size() != static_cast<std::size_t>(batch) * static_cast<std::size_t>(in) ||
      static_cast<int>(input.shape[1]) != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: input width mismatch");
  }
  if (layer.weights.rows != out || layer.weights.cols != in) {
    return Status::Err(StatusCode::InvalidArgument, "dense: weight shape mismatch");
  }

  if (!layer.gpu_valid || layer.gpu_batch != batch) {
    if (auto st = sync_gpu_from_weights(layer, batch); !st) return st;
  }

  *pre = TensorF32::zeros(batch, out);
  *post = TensorF32::zeros(batch, out);
  return cuda::resident_execute(const_cast<cuda::ResidentDense*>(&layer.gpu), input.data.data(),
                                pre->data.data(), post->data.data());
}

} // namespace welvet::dense
