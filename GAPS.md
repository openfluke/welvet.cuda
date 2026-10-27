# welvet.cuda gap census

**Authoritative checklist:** [README.md](README.md) — full Go Welvet inventory as CUDA rebuild gates (3D volumetric engine). Tick there.  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.27`** — engine board **27 / 100**. This file is the machine-oriented dump (`welvet_dense_fwd --gaps`).

```bash
./scripts/build_linux.sh
./build/example_cuda_smoke
./build/welvet_dense_fwd --version
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
```

## Done

| ID | Detail |
|---|---|
| M0 CUDA host | device, buffers, smoke, C ABI, Linux build |
| FormatNone×34 | pack/decode + Dense MatVec path |
| Leaf FormatNone forward | Dense, MHA, SwiGLU, RMSNorm, LayerNorm, Softmax, Embedding, CNN1–3, ConvT1–3, RNN, LSTM, Sequential, Residual, Mamba, Metacognition, GDN(f32), KMeans, Parallel, seqmix — w2a exact_bit |
| w2a go_v_cuda | leaf forward parity + Go/C++/CUDA speed |

## Open

| ID | Detail |
|---|---|
| Stack leaf | no wire magic — lands with NormalBP `STKB` |
| NormalBP / train | all layers + 29 TrainModes |
| Volume forward | `*G1` / `WRF1`… mount-all |
| Runtime / systems / model / lucy / CamSync / peak fused / NAC / Windows | see README |

Live counters: `./build/welvet_dense_fwd --gaps` (done=32 open=30 at last census).
