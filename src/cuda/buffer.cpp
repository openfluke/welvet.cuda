#include "welvet/cuda/buffer.hpp"

#include "welvet/cuda/device.hpp"

#include <cuda_runtime.h>

namespace {

welvet::Status cuda_status(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + cudaGetErrorString(err));
}

} // namespace

namespace welvet::cuda {

Status buffer_alloc(std::size_t bytes, Buffer* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null Buffer");
  if (bytes == 0) return Status::Err(StatusCode::InvalidArgument, "zero-size buffer");

  auto st = init_default_device();
  if (!st) return st;

  void* ptr = nullptr;
  st = cuda_status(cudaMalloc(&ptr, bytes), "cudaMalloc");
  if (!st) return st;

  out->device = ptr;
  out->bytes = bytes;
  out->device_owned = true;
  return Status::Ok();
}

Status buffer_upload(Buffer* buf, const void* host, std::size_t bytes) {
  if (!buf || !buf->device) return Status::Err(StatusCode::InvalidArgument, "null buffer");
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf->bytes) return Status::Err(StatusCode::InvalidArgument, "upload overflow");
  return cuda_status(cudaMemcpy(buf->device, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

Status buffer_download(const Buffer& buf, void* host, std::size_t bytes) {
  if (!buf.device) return Status::Err(StatusCode::InvalidArgument, "null buffer");
  if (!host) return Status::Err(StatusCode::InvalidArgument, "null host");
  if (bytes > buf.bytes) return Status::Err(StatusCode::InvalidArgument, "download overflow");
  return cuda_status(cudaMemcpy(host, buf.device, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

void buffer_release(Buffer* buf) {
  if (!buf || !buf->device) return;
  if (buf->device_owned) {
    (void)cudaFree(buf->device);
  }
  buf->device = nullptr;
  buf->bytes = 0;
}

} // namespace welvet::cuda
