# welvet.cuda gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.53`** — engine board **53 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --stepbp-train
./scripts/linux_build_w2a.sh --meshbp-train
```

## Done

| Slice | Detail |
|---|---|
| M0 CUDA host | device, buffers, smoke, C ABI, Linux |
| FormatNone×34 | pack/decode + Dense MatVec/SGD |
| Leaf FormatNone forward | all 23 layers + seqmix (w2a exact_bit) |
| **NormalBP Mode 1** | leaf+volume for all 23 layers — `TestGoVCudaNormalBP` PASS |
| **StepBP Line + Dense MeshBP** | STB1 + MBP1 + ModeMeshBP vols — `TestGoVCudaStepBPTrain` PASS |
| **Heterogeneous MeshBP** | MGB1/MGBO all 23 families, 103 groups — `TestGoVCudaMeshBPTrain` PASS |

## Open

Tween* / Split / Alt / CamSync / Freeze/Shadow / systems / model / lucy / peak fused / NAC / Windows / packed quants. See README.

Live counters: `./build/welvet_dense_fwd --gaps`.
