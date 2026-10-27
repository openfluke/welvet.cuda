#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "welvet/status.hpp"

namespace welvet::cuda {

struct DeviceInfo {
  int index = -1;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t total_global_mem = 0;
  int multi_processor_count = 0;
};

// Process-wide default device (cudaSetDevice(0) unless overridden).
Status init_default_device();
bool available();
Status default_device_info(DeviceInfo* out);
std::vector<DeviceInfo> list_devices();

// Device + driver/runtime identity for provenance in w2a logs.
std::string runtime_identity();

} // namespace welvet::cuda
