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
| **0.x** | `round(earned)` from the engine table in [README.md](README.md) |
| **1.0.0** | Engine board **100/100** — full Go parity on NVIDIA GPU (FormatNone path + Model/IO + Systems + CamSync + peak fused, or explicit gate rewrite) |

| In (moves `0.x`) | Out |
|------------------|-----|
| Layers, FormatNone×34, quants, train modes, Grid/Line/Mesh, Tween*, systems, model/IO — **on CUDA** | Other vendors (Metal/ROCm/QNN) — sibling repos |
| Evidence via w2a `Go_v_Cuda` | welvet.cpp / welvet.metal / Go scores (not inherited) |
| Determinism + speed on named NVIDIA GPU | Packaging polish alone |

**Sibling map:** welvet (Go ref) · welvet.cpp (CPU) · welvet.metal (Apple GPU) · **welvet.cuda** (this) · welvet.qnn · w2a.
