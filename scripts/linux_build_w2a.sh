#!/usr/bin/env bash
# Build welvet.cuda then run sibling w2a Go↔CUDA suites.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="${W2A_DIR:-$ROOT/../w2a}"
BUILD="${BUILD_DIR:-$ROOT/build}"

usage() {
  cat <<EOF
usage: $0 [flags]

  (default)           build + TestGoVCudaDenseForward (parity + speed)
  --smoke             build + example_cuda_smoke
  --gaps              build + welvet_dense_fwd --gaps
  --dense-fwd         same as default
  --help
EOF
}

MODE=dense-fwd
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --dense-fwd) MODE=dense-fwd; shift ;;
    *)
      echo "unknown flag '$1'" >&2
      usage
      exit 2
      ;;
  esac
fi

"$ROOT/scripts/build_linux.sh"

export WELVET_CUDA_DENSE_FWD="$BUILD/welvet_dense_fwd"
# Optional C++ column in speed table
if [[ -z "${WELVET_DENSE_FWD:-}" ]]; then
  for cand in "$ROOT/../welvet.cpp/build/welvet_dense_fwd" "$ROOT/../welvet.cpp/build/Release/welvet_dense_fwd"; do
    if [[ -x "$cand" ]]; then
      export WELVET_DENSE_FWD="$cand"
      break
    fi
  done
fi

case "$MODE" in
  smoke)
    "$BUILD/example_cuda_smoke"
    ;;
  gaps)
    "$BUILD/welvet_dense_fwd" --gaps
    ;;
  dense-fwd)
    if [[ ! -d "$W2A" ]]; then
      echo "w2a sibling not found at $W2A" >&2
      exit 1
    fi
    # Go must link with the system toolchain. Conda GCC 14 is host-only for
    # nvcc; leaving CC/CXX set breaks linking prebuilt libwelvet.so (GLIBC).
    export LD_LIBRARY_PATH="${ROOT}/.deps/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
    (
      cd "$W2A"
      env -u CC -u CXX -u CMAKE_CXX_COMPILER -u CMAKE_C_COMPILER \
        PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
        WELVET_CUDA_DENSE_FWD="$WELVET_CUDA_DENSE_FWD" \
        WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-}" \
        LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        go test ./tests/go_v_cuda -v -count=1 -run '^TestGoVCudaDenseForward$'
    )
    ;;
esac
