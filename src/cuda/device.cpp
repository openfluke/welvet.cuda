#include "welvet/cuda/device.hpp"

#include <cuda_runtime.h>

#include <mutex>
#include <sstream>

namespace {

std::mutex g_device_mu;
int g_device = -1;
std::string g_identity;

welvet::Status cuda_status(cudaError_t err, const char* what) {
  if (err == cudaSuccess) return welvet::Status::Ok();
  return welvet::Status::Err(welvet::StatusCode::Internal,
                             std::string(what) + ": " + cudaGetErrorString(err));
}

welvet::cuda::DeviceInfo info_from(int index) {
  welvet::cuda::DeviceInfo i;
  i.index = index;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) {
    i.name = "(unavailable)";
    return i;
  }
  i.name = prop.name;
  i.compute_major = prop.major;
  i.compute_minor = prop.minor;
  i.total_global_mem = prop.totalGlobalMem;
  i.multi_processor_count = prop.multiProcessorCount;
  return i;
}

} // namespace

namespace welvet::cuda {

Status init_default_device() {
  std::lock_guard<std::mutex> lock(g_device_mu);
  if (g_device >= 0) return Status::Ok();

  int count = 0;
  auto st = cuda_status(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (!st) return st;
  if (count <= 0) {
    return Status::Err(StatusCode::Unsupported, "no CUDA device available");
  }

  st = cuda_status(cudaSetDevice(0), "cudaSetDevice");
  if (!st) return st;

  g_device = 0;
  auto info = info_from(0);
  std::ostringstream os;
  os << "cuda:" << info.name << ":idx=" << info.index << ":sm=" << info.compute_major << "."
     << info.compute_minor << ":mem=" << info.total_global_mem;
  g_identity = os.str();
  return Status::Ok();
}

bool available() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
  return count > 0;
}

Status default_device_info(DeviceInfo* out) {
  if (!out) return Status::Err(StatusCode::InvalidArgument, "null DeviceInfo");
  auto st = init_default_device();
  if (!st) return st;
  *out = info_from(g_device);
  return Status::Ok();
}

std::vector<DeviceInfo> list_devices() {
  std::vector<DeviceInfo> out;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) return out;
  out.reserve(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    out.push_back(info_from(i));
  }
  return out;
}

std::string runtime_identity() {
  (void)init_default_device();
  std::lock_guard<std::mutex> lock(g_device_mu);
  return g_identity.empty() ? std::string("cuda:unavailable") : g_identity;
}

} // namespace welvet::cuda
