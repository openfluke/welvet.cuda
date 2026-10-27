#pragma once

#include <cstddef>
#include <cstdint>

#include "welvet/status.hpp"

namespace welvet::cuda {

// Go parallel.projectGap — block-average src[sn] onto dst[dn] with f64 sums on GPU.
Status launch_project_gap(const float* src_host, float* dst_host, std::int32_t sn, std::int32_t dn);

} // namespace welvet::cuda
