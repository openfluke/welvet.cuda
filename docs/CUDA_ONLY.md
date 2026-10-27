# CUDA execution contract

`welvet.cuda` is a standalone native NVIDIA GPU backend. `welvet.cpp` is the
separate CPU backend; `welvet.metal` is the separate Apple GPU backend. Each
ships through its own native artifact and C ABI. There is no dependency from
one backend to the other, and **no host numerical fallback** labeled as CUDA.

## Supported computation

**M0** (host scaffolding — not engine-board credit alone):

- CUDA device probe + runtime identity string
- Device buffer alloc / H2D / D2H
- Compiled smoke kernel `welvet_smoke_saxpy`
- C ABI: version / init / available / gaps

**M1+** (engine credit — leaf FormatNone forwards):

- FormatNone×34 decode → resident **HW float64** MatVec / layer kernels on CUDA
- Dense + 21 other layers + seqmix leaf forward (Stack deferred to NormalBP)
- Wire tool `welvet_dense_fwd` multiplexes leaf magics — w2a `go_v_cuda` exact_bit×34

Everything else (NormalBP, volume, train modes, …) is gap until ticked in
[TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md) and the [README runway](../README.md).

## Non-negotiable

- Supported numerical forward/backward/training paths run on **CUDA**.
- Host code may schedule, validate, allocate and stage storage.
- Unsupported dtype/layer/mode → hard error (no silent CPU math).
- FormatNone + DType (or packed quant) is storage truth — not an undeclared f32 master.
- Parity vs Go is best-effort **60–100%** on w2a's spectrum; report the band.

## Build and delivery

```sh
./scripts/build_linux.sh
./build/example_cuda_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --smoke
./scripts/linux_build_w2a.sh --leaf-fwd-all
```


Tests and CPU parity references live in sibling `w2a`, not this repository.
Shared artifact: `libwelvet_cuda.so`. Requires CMake + NVIDIA CUDA Toolkit (`nvcc`)
and a **nvcc-supported host compiler** (GCC ≤14 for CUDA 12.9; MSVC on Windows).

On Fedora 44+ (system GCC 16), use the conda host toolchain:

```sh
conda activate welvet-cuda-build   # gcc/g++ 14 from conda-forge
./scripts/build_linux.sh
```

`build_linux.sh` auto-picks that env’s `g++` when present. This is normal
C++ + nvcc — not a Clang-CUDA special path.
