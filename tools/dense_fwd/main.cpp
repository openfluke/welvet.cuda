// Binary wire tool for future w2a go_v_cuda exact parity + timing.
// M0: version / gaps / CUDA identity only. Magics land with Dense gates.

#include "welvet/gaps.hpp"
#include "welvet/cuda/device.hpp"
#include "welvet_c.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

static void usage() {
  std::fprintf(stderr,
               "welvet_dense_fwd (welvet.cuda)\n"
               "  --version     print engine version\n"
               "  --identity    print CUDA runtime identity\n"
               "  --gaps        print gap census\n"
               "  --help\n");
}

int main(int argc, char** argv) {
  std::string cmd = argc > 1 ? argv[1] : "--help";

  if (cmd == "--version" || cmd == "-v") {
    std::printf("%s\n", welvet_version());
    return 0;
  }
  if (cmd == "--identity") {
    auto st = welvet_cuda_init();
    if (st.code != 0) {
      std::fprintf(stderr, "%s\n", st.message);
      return 1;
    }
    std::printf("%s\n", welvet_runtime_identity());
    return 0;
  }
  if (cmd == "--gaps") {
    welvet::gaps::print_census(std::cout);
    return 0;
  }
  if (cmd == "--help" || cmd == "-h") {
    usage();
    return 0;
  }

  std::fprintf(stderr, "unknown args (Dense wire magics are gap? until M1)\n");
  usage();
  return 2;
}
