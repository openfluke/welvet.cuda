# welvet.cuda gap census

**Authoritative checklist:** [README.md](README.md) — full Go Welvet inventory as CUDA rebuild gates (3D volumetric engine). Tick there.  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.13`** — engine board **13 / 100**. This file is the machine-oriented dump (`welvet_dense_fwd --gaps`).

```bash
./scripts/build_linux.sh
./build/example_cuda_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --dense-fwd
```

## Done

| ID | Detail |
|---|---|
| CUDA device | `cudaGetDevice` / props, runtime identity |
| Kernels | nvcc compile `src/cuda/kernels.cu` (smoke + f64 MatVec) |
| Device buffers | `cudaMalloc` / upload / download |
| Smoke launch | `welvet_smoke_saxpy` |
| C ABI | version / init / gaps counters |
| Linux build | `scripts/build_linux.sh` |
| FormatNone×34 | pack/decode + Dense forward gates |
| Dense forward CUDA | HW f64 MatVec + bias/act; w2a exact_bit×34 |
| w2a go_v_cuda | Dense forward parity + Go/C++/CUDA speed |

## Open

Dense NormalBP, other layers, volume/runtime, train modes, systems, model/IO, lucy, CamSync, peak fused, NAC, Windows — see [README.md](README.md).
