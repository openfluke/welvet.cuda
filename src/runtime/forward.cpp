#include "welvet/runtime/forward.hpp"

namespace welvet::runtime {

Status volume_forward_gap() {
  return Status::Gap("volume forward on CUDA (Grid + resident cells)");
}

} // namespace welvet::runtime
