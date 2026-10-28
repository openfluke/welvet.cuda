# welvet.cuda versioning

**One scoreboard:** same ten categories / 100 points as Go Welvet.  
**Formula:** `version = 0.{round(earned)}` until **100 → 1.0.0**.

| Tag | Meaning |
|-----|---------|
| **0.00** | Bedrock — CUDA host scaffold; **0 / 100** on the engine board |
| **0.13** | M1 Dense FormatNone×34 forward signed (Foundation 9 + Dense fwd 4) |
| **0.27** | Leaf FormatNone forwards ×22 layers + seqmix (Stack→NormalBP) |
| **0.51** | NormalBP Mode 1 signed for all 23 layers (`TestGoVCudaNormalBP`) |
| **0.52** | StepBP Line + Dense MeshBP (`TestGoVCudaStepBPTrain`) |
| **0.53** | Heterogeneous MeshBP MGB1 all 23 families (`TestGoVCudaMeshBPTrain`, 103 groups) |
| **0.54** | ModeTween projected local @ lr×0.5 (`TestGoVCudaTweenTrain`, 130 groups) |
| **0.55** | ModeTweenChain FamilyBP @ full lr (`TestGoVCudaTweenChainTrain`, 130 groups) |
| **0.56** | ModeTweenSplit OpenSplitTape 1/N @ full lr (`TestGoVCudaTweenSplitTrain`, 131 groups) |
| **0.57** | ModeStepTweenSplit Line FamilyTweenSplit (`TestGoVCudaStepTweenSplitTrain`, 130 groups) |
| **0.58** | ModeTweenAlt Split→Tween (`TestGoVCudaTweenAltTrain`, 130 groups) |
| **0.59** | ModeStepTweenAlt Line FamilyTweenAlt (`TestGoVCudaStepTweenAltTrain`, 130 groups) |
| **0.60** | MeshTween mesh-clock FamilyTween (`TestGoVCudaMeshTweenTrain`, 159 groups) |
| **0.61** | MeshTweenChain ≡ MeshBP FamilyBP (`TestGoVCudaMeshTweenChainTrain`, 160 groups) |
| **0.62** | ModeStepTween Line FamilyTween (`TestGoVCudaStepTweenTrain`, 130 groups) |
| **0.63** | ModeStepTweenChain Line FamilyBP alias (`TestGoVCudaStepTweenChainTrain`, 130 groups) |
| **0.64** | ModeTweenSplitHeadProxy (`TestGoVCudaProxyTrain` 65 + `TestGoVCudaProxyVolumeTrain` 69) |
| **0.65** | ModeTweenSplitLinear (`TestGoVCudaProxyTrain` 65 + `TestGoVCudaProxyVolumeTrain` 69) |
| **0.66** | ModeTweenSplitFastProxy (`TestGoVCudaProxyTrain` 65 + `TestGoVCudaProxyVolumeTrain` 69) |
| **0.67** | ModeTweenSplitLinearCache (`TestGoVCudaProxyTrain` 88 + `TestGoVCudaProxyVolumeTrain` 92) |
| **0.68** | ModeTweenSplitHeadProxyAsync (`TestGoVCudaProxyTrain` 65 + `TestGoVCudaProxyVolumeTrain` 69) |
| **0.69** | ModeTweenSplitSparse (`TestGoVCudaProxyTrain` 65 + `TestGoVCudaProxyVolumeTrain` 69) |
| **0.70** | MeshTweenSplit (`TestGoVCudaMeshSplitModes` 154; MGS1/MGSO FamilyTweenSplit @ full lr) |
| **0.71** | MeshTweenAlt (`TestGoVCudaMeshSplitModes` 154; MGA1/MGAO Split→Tween@lr×0.5) |
| **0.72** | MeshTweenSplitFastProxy (`TestGoVCudaMeshSplitModes` 172; MGF1/MGFO) |
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on NVIDIA GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on CUDA** | Other vendors (Metal/ROCm/QNN) — sibling repos |
| Evidence via w2a `Go_v_Cuda` | welvet.cpp / welvet.metal / Go scores (not inherited) |
| Determinism + speed on named NVIDIA GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU) · welvet.metal (Apple GPU) · **welvet.cuda** (this) · welvet.qnn · w2a.
