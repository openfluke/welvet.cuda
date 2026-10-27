# welvet.cuda gap census

**Authoritative checklist:** [README.md](README.md) — full Go Welvet inventory as CUDA rebuild gates (3D volumetric engine). Tick there.  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.00`** — engine board **0 / 100**. This file is the machine-oriented dump (`welvet_dense_fwd --gaps`).

```bash
./scripts/build_linux.sh
./build/example_cuda_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
```

## Done (M0 host — not engine-board credit)

| ID | Detail |
|---|---|
| CUDA device | `cudaGetDevice` / props, runtime identity |
| Kernels | nvcc compile `src/cuda/kernels.cu` (smoke + matvec stub) |
| Device buffers | `cudaMalloc` / upload / download |
| Smoke launch | `welvet_smoke_saxpy` |
| C ABI | version / init / gaps counters |
| Linux build | `scripts/build_linux.sh` |

## Open

Everything else in [README.md](README.md): foundation, Dense, 23 layers, 34 dtypes, 20 quants, 29 train modes, Grid/Line/Mesh, systems, model/IO, lucy, CamSync, peak fused, w2a `Go_v_Cuda`.
