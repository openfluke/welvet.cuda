# welvet.cuda gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.51`** — engine board **51 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
```

## Done

| Slice | Detail |
|---|---|
| M0 CUDA host | device, buffers, smoke, C ABI, Linux |
| FormatNone×34 | pack/decode + Dense MatVec/SGD |
| Leaf FormatNone forward | all 23 layers + seqmix (w2a exact_bit) |
| **NormalBP Mode 1** | leaf+volume for all 23 layers — `TestGoVCudaNormalBP` PASS |

## Open

StepBP / MeshBP / Tween* / Split / Alt / CamSync / Freeze/Shadow / systems / model / lucy / peak fused / NAC / Windows / packed quants. See README.

Live counters: `./build/welvet_dense_fwd --gaps`.
