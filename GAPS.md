# welvet.cuda gap census

**Authoritative checklist:** [README.md](README.md).  
**Training detail:** [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).

**Version `0.69`** — engine board **69 / 100**. Dump: `welvet_dense_fwd --gaps`.

```bash
./scripts/build_linux.sh
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --stepbp-train
./scripts/linux_build_w2a.sh --meshbp-train
./scripts/linux_build_w2a.sh --tween-train
./scripts/linux_build_w2a.sh --tweenchain-train
./scripts/linux_build_w2a.sh --tweensplit-train
./scripts/linux_build_w2a.sh --steptweensplit-train
./scripts/linux_build_w2a.sh --tweenalt-train
./scripts/linux_build_w2a.sh --steptweenalt-train
./scripts/linux_build_w2a.sh --meshtween-train
./scripts/linux_build_w2a.sh --meshtweenchain-train
./scripts/linux_build_w2a.sh --steptween-train
./scripts/linux_build_w2a.sh --steptweenchain-train
./scripts/linux_build_w2a.sh --proxy-train
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
| **ModeTween** | TWB1/TWV1/per-kind/KMTV/PNTV — 130 groups — `TestGoVCudaTweenTrain` PASS |
| **ModeTweenChain** | TCB1/TCV1/per-kind/KMTC/PNTC — 130 groups — `TestGoVCudaTweenChainTrain` PASS |
| **ModeTweenSplit** | TSB1/TSP1/per-kind/KMSP/PNSP — 131 groups — `TestGoVCudaTweenSplitTrain` PASS |
| **ModeStepTweenSplit** | STS1/STSO + TweenSplit volumes — 130 groups — `TestGoVCudaStepTweenSplitTrain` PASS |
| **ModeTweenAlt** | TAB1/TAV1/per-kind/KMAL/PNAL — 130 groups — `TestGoVCudaTweenAltTrain` PASS |
| **ModeStepTweenAlt** | STA1/STAO + TweenAlt volumes — 130 groups — `TestGoVCudaStepTweenAltTrain` PASS |
| **MeshTween** | MTW1/MTWO + MGW1/MGWO + ModeTween volumes — 159 groups — `TestGoVCudaMeshTweenTrain` PASS |
| **MeshTweenChain** | MTC1/MTCO + MGC1/MGCO ≡ MeshBP FamilyBP — 160 groups — `TestGoVCudaMeshTweenChainTrain` PASS |
| **ModeStepTween** | STW1/STWO Line FamilyTween @ lr×0.5 + Tween volumes — 130 groups — `TestGoVCudaStepTweenTrain` PASS |
| **ModeStepTweenChain** | STC1/STCO Line FamilyBP alias + TweenChain volumes — 130 groups — `TestGoVCudaStepTweenChainTrain` PASS |
| **ModeTweenSplitHeadProxy** | GPB1/GPBO + PGB1/PGVO variant 1 — 65 leaf + 69 volume — `TestGoVCudaProxyTrain`/`ProxyVolumeTrain` PASS |
| **ModeTweenSplitLinear** | GPB1/GPBO + PGB1/PGVO variant 2 — 65 leaf + 69 volume — `TestGoVCudaProxyTrain`/`ProxyVolumeTrain` PASS |
| **ModeTweenSplitFastProxy** | GPB1/GPBO + PGB1/PGVO variant 3 — 65 leaf + 69 volume — `TestGoVCudaProxyTrain`/`ProxyVolumeTrain` PASS |
| **ModeTweenSplitLinearCache** | GPB1/GPBO + PGB1/PGVO variant 4 — 88 leaf + 92 volume (incl. epoch-22 refresh) — PASS |
| **ModeTweenSplitHeadProxyAsync** | GPB1/GPBO + PGB1/PGVO variant 5 — 65 leaf + 69 volume (4-epoch stale reuse) — PASS |
| **ModeTweenSplitSparse** | GPB1/GPBO + PGB1/PGVO variant 6 — 65 leaf + 69 volume (4-epoch rotate) — PASS |

## Open

Remaining Tween* (Mesh/Step proxy…) / CamSync / Freeze/Shadow / other systems / model / lucy / peak fused / NAC / Windows / packed quants. See README.

Live counters: `./build/welvet_dense_fwd --gaps`.
