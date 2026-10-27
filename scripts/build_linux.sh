#!/usr/bin/env bash
# Linux NVIDIA CUDA host build for welvet.cuda — normal C++ + nvcc.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${BUILD_DIR:-$ROOT/build}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "welvet.cuda build_linux.sh expects Linux (got $(uname -s))" >&2
  exit 1
fi

# Prefer an explicit CUDA toolkit (system install or local .deps extract).
if [[ -z "${CUDAToolkit_ROOT:-}" ]]; then
  for cand in \
    "${ROOT}/.deps/cuda/cuda" \
    "${CUDA_HOME:-}" \
    "${CUDA_PATH:-}" \
    /usr/local/cuda
  do
    if [[ -n "$cand" && -x "${cand}/bin/nvcc" ]]; then
      export CUDAToolkit_ROOT="$cand"
      break
    fi
  done
fi

if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
  export PATH="${CUDAToolkit_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDAToolkit_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found. Install the NVIDIA CUDA Toolkit, or set CUDAToolkit_ROOT." >&2
  echo "  (this tree can use a local extract under .deps/cuda/cuda)" >&2
  exit 1
fi

# nvcc needs a supported host compiler (GCC <= 14 for CUDA 12.9).
# Fedora 44 ships GCC 16 — use conda env welvet-cuda-build (GCC 14) when present.
CMAKE_ARGS=()
pick_host() {
  local gxx="$1" gcc="$2"
  [[ -x "$gxx" && -x "$gcc" ]] || return 1
  CMAKE_ARGS+=(
    -DCMAKE_CXX_COMPILER="$gxx"
    -DCMAKE_CUDA_HOST_COMPILER="$gxx"
    -DCMAKE_C_COMPILER="$gcc"
  )
  echo "host compiler: $gxx ($("$gxx" --version | head -1))"
  return 0
}

if [[ -n "${CMAKE_CXX_COMPILER:-}" ]]; then
  :
elif pick_host \
  "${CONDA_PREFIX:-}/bin/g++" \
  "${CONDA_PREFIX:-}/bin/gcc"
then
  :
elif pick_host \
  "${HOME}/miniconda3/envs/welvet-cuda-build/bin/g++" \
  "${HOME}/miniconda3/envs/welvet-cuda-build/bin/gcc"
then
  :
elif command -v g++-14 >/dev/null 2>&1 && command -v gcc-14 >/dev/null 2>&1; then
  pick_host "$(command -v g++-14)" "$(command -v gcc-14)" || true
else
  major="$(g++ -dumpfullversion 2>/dev/null | cut -d. -f1 || echo 0)"
  if [[ "$major" -gt 14 ]]; then
    echo "system g++ is $major; CUDA 12.x nvcc wants <= 14." >&2
    echo "  conda activate welvet-cuda-build   # GCC 14 (conda-forge)" >&2
    echo "  or: dnf/apt install gcc-14 g++-14 and re-run" >&2
    exit 1
  fi
fi

CMAKE_ARGS+=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-75}"
)
if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
  CMAKE_ARGS+=(-DCUDAToolkit_ROOT="$CUDAToolkit_ROOT")
fi

cmake -S "$ROOT" -B "$BUILD" "${CMAKE_ARGS[@]}" "$@"
cmake --build "$BUILD" -j"$(nproc 2>/dev/null || echo 4)"

echo "built: $BUILD"
echo "  lib:  $BUILD/libwelvet_cuda.so"
echo "  tool: $BUILD/welvet_dense_fwd"
echo "  ex:   $BUILD/example_cuda_smoke"
echo
echo "smoke:  $BUILD/example_cuda_smoke"
echo "gaps:   $BUILD/welvet_dense_fwd --gaps"
