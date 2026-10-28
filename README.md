# welvet.cuda — Welvet (3D volumetric AI engine) on NVIDIA CUDA

**welvet.cuda is a full rebuild of Welvet — the 3D volumetric AI engine — in C++ / NVIDIA CUDA.** Same engine shape as Go Welvet: numerical types, layers, volumetric architecture (cells in a 3D grid), forward/backward, training modes, systems, model loading, inference, checkpoints. Execution is CUDA compute + resident device buffers. Host orchestration is C++ (CUDA Runtime / driver API at the device boundary).

A `1 × 1 × 1` volume is the ordinary single-cell neural-network case (layers inside that cell). Larger volumes connect cells that each hold their own layer stacks. This CUDA rebuild must own and execute that same topology — not “a Dense GEMV demo with a GPU sticker.”

It will run directly from C++ and through a stable C ABI. Go Welvet can later delegate via **NAC** (Native Acceleration Compute). The CUDA engine must also build, load models, execute volumes, and train **without** a Go process.

| | |
|--|--|
| **Version** | **`0.73`** |
| **Engine board** | **67 / 100** (same 10 categories as Go Welvet) |
| **Compute** | NVIDIA GPU (CUDA) — primary · CPU / Metal / QNN siblings remain separate repos — never implied |
| **Parity reference** | [welvet](https://github.com/openfluke/welvet) Go **v1.2.0** (100/100) |
| **CPU sibling** | [welvet.cpp](https://github.com/openfluke/welvet.cpp) **0.67** (semantics/wire to mirror — **score not inherited**) |
| **Validation** | [w2a](https://github.com/openfluke/w2a) `Go_v_Cuda` leaf FormatNone×34 + NormalBP + StepBP + MeshBP + ModeTween* |

**This README is the runway.** Every unchecked cell is planned CUDA work. Tick ⬜ → 🚧 → ✅ here as gates land. **Training port detail** (per-mode × layer evidence, wire names, version log) lives in [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md) — tick there as NormalBP / TrainModes land; this README keeps the mode summary table. [GAPS.md](GAPS.md) is the machine-oriented census mirror; [VERSIONING.md](VERSIONING.md) is the `0.x` formula. Pure-CUDA contract: [docs/CUDA_ONLY.md](docs/CUDA_ONLY.md).

### What exists today (M0–M1 + leaf FormatNone forwards)

```bash
./scripts/build_linux.sh
./build/example_cuda_smoke
./build/welvet_dense_fwd --version   # 0.73
./build/welvet_dense_fwd --identity
./build/welvet_dense_fwd --gaps
./scripts/linux_build_w2a.sh --leaf-fwd-all
./scripts/linux_build_w2a.sh --normalbp-all
./scripts/linux_build_w2a.sh --stepbp-train
./scripts/linux_build_w2a.sh --meshbp-train
./scripts/linux_build_w2a.sh --tween-train
./scripts/linux_build_w2a.sh --tweenchain-train
./scripts/linux_build_w2a.sh --tweensplit-train
./scripts/linux_build_w2a.sh --steptweensplit-train
./scripts/linux_build_w2a.sh --meshtweenchain-train
./scripts/linux_build_w2a.sh --steptween-train
./scripts/linux_build_w2a.sh --steptweenchain-train
./scripts/linux_build_w2a.sh --proxy-train
```

| Slice | Status |
|---|:---:|
| Linux x86_64 CMake → `libwelvet_cuda.so` | ✅ |
| Leaf FormatNone forward ×23 + seqmix | ✅ |
| NormalBP leaf+volume ×23 layers | ✅ `TestGoVCudaNormalBP` |
| StepBP Line + Dense MeshBP | ✅ `TestGoVCudaStepBPTrain` |
| Heterogeneous MeshBP (MGB1) | ✅ `TestGoVCudaMeshBPTrain` (103) |
| ModeTween / ModeTweenChain / ModeTweenSplit / ModeStepTweenSplit / ModeTweenAlt / ModeStepTweenAlt / MeshTween / MeshTweenChain / StepTween / StepTweenChain | ✅ 130 / 130 / 131 / 130 / 130 / 130 / 159 / 160 / 130 / 130 |
| ModeTweenSplitHeadProxy / Linear / FastProxy / LinearCache / HeadProxyAsync / Sparse | ✅ ProxyTrain + ProxyVolume (sparse: 65/69) |
| MeshTweenSplit / Alt / FastProxy / Sparse | ✅ `TestGoVCudaMeshSplitModes` 154+154+172+172 |
| Remaining Tween* (Step proxy) / CamSync / other systems / model | ⬜ |

---

## Contents

- [Relationship](#relationship)
- [3D volumetric shape](#3d-volumetric-shape)
- [Non-negotiable rules](#non-negotiable-rules)
- [Scoreboard and credit rules](#scoreboard-and-credit-rules)
- [Engine rebuild scoreboard — 100 points](#engine-rebuild-scoreboard--100-points)
- [Full engine subsystem runway](#full-engine-subsystem-runway)
- [Coverage axes](#coverage-axes)
- [DTypes — 34 numerical types](#dtypes-coredtype--34-numerical-types)
- [Quant formats — 20](#quant-formats-quantformat--20)
- [Layer runway — 23 kinds + seqmix](#layer-runway--23-kinds--seqmix)
- [Kernel and training runway](#kernel-and-training-runway)
- [Training modes — all 29 + cameral](#training-modes--all-29--cameral)
- [Volumetric execution and placement](#volumetric-execution-and-placement)
- [CUDA platform runway](#cuda-platform-runway)
- [Systems — dna / evolution / tween / tanhi / telemetry](#systems--engine-7)
- [Model / IO](#model--io--engine-8)
- [Lucy / stubs / apps](#lucy--stubs--apps)
- [FFI and C ABI](#ffi-and-c-abi)
- [Resident weights and memory](#resident-weights-and-memory)
- [Validation and performance (w2a)](#validation-and-performance-w2a)
- [Implementation sequence](#implementation-sequence)
- [Source layout](#source-layout)

---

## Relationship

| Repository | Responsibility |
|---|---|
| [welvet](https://github.com/openfluke/welvet) | Go engine — volumetric architecture, layers, dtypes, quants, train modes, ENTITY, CPU/SIMD/WebGPU |
| [welvet.cpp](https://github.com/openfluke/welvet.cpp) | Portable C++ CPU engine + C ABI (port **semantics**, not CPU kernels) |
| [welvet.metal](https://github.com/openfluke/welvet.metal) | Apple Metal sibling (separate tree; score not inherited) |
| **welvet.cuda** | **This repo** — full Welvet engine on NVIDIA CUDA |
| [welvet.qnn](https://github.com/openfluke/welvet.qnn) | Qualcomm Hexagon HTP sibling pattern |
| [w2a](https://github.com/openfluke/w2a) | Parity / speed / determinism (`Go_v_Cuda` planned like `Go_v_Cpp`) |

Go is the parity reference. CUDA owns its own topology, model state, numerical storage, forward/backward, and training on GPU. A native CUDA run does not need Go to orchestrate each layer.

### Architecture sketch — intended flow

```text
C++ application / CLI                 Go Welvet or w2a / future NAC
          |                                       |
     C++ engine API                          stable C ABI
          |                                       |
          +-------- welvet.cuda engine ----------+
                              |
          model / ENTITY / tokenizer / transformer / sampling
                              |
             3D architecture: cells, stacks, branches, links
                              |
          forward / backward / training / Step / Mesh / systems
                              |
                cudaDevice + module/kernels + streams + capability queries
                              |
             resident device buffers (weights + tensors + training state)
                              |
                    NVIDIA GPU (CUDA compute)
                    other vendors = sibling repos (never implied)
```

---

## 3D volumetric shape

Welvet is **not** “a list of layers.” It is a **3D volumetric AI engine**:

| Concept | Meaning |
|---|---|
| **Volume / Grid** | `Depth × Rows × Cols` cells; each cell holds a layer stack |
| **`1×1×1`** | Ordinary single-network case (still a volume of one cell) |
| **`2×2×2`, `3×3×3`, non-cubic** | Multi-cell topologies with hops and remote links |
| **Line / Step\*** | 1D systolic pipe through a stack (fill/drain ticks) |
| **Mesh\*** | Credit and scheduling across volumetric links |
| **Cameral / Parallel** | Branch modes, hemispheres, combine — still volumetric-aware |

CUDA must keep weights, activations, tapes, and credit **resident** across cells where the schedule requires it. Host orchestration without device residency is not “volumetric CUDA done.”

---

## Non-negotiable rules

Same spirit as Go Welvet:

1. **No engine tests in this tree** — validation lives in w2a (`Go_v_Cuda`).
2. **No silent fallbacks** — unsupported dtype/layer/mode → hard error (no fake “CUDA” that ran on host ALU unmarked).
3. **Nothing hardcoded to float32 as storage truth** — FormatNone + DType (or packed quant) is authoritative; scratch f32 at the CUDA ALU boundary must be explicit and accounted.
4. **No QAT** — `DType` + `QuantFormat` are storage truth.
5. **Train keeps storage truth** — updates preserve declared encoding (or unpack→update→re-pack with documented scratch).
6. **One Go feature folder → one CUDA runway row** (tick in this README).
7. **v1.0.0 = engine board 100/100** on CUDA. Apps/stubs/non-CUDA are tracked but do not invent board points unless a gate is written first.
8. **Best-effort determinism** — aim for 100% (`exact_bit`) agreement with Go; the accepted project parity range is **60–100% on w2a's determinism spectrum**. Report the measured band and numerical deviations before ✅.

### Numerical parity target

Deterministic exactness against Go is **best effort, aiming for 100%, with an
accepted range of 60–100%**. These percentages are w2a spectrum scores:
`industry` = 60%, `strict` = 80%, and `exact_bit` = 100%. They are not model
accuracy or the percentage of tensor elements allowed to be wrong.

CPU and CUDA GPU arithmetic can differ because of reduction order, rounding
and parallel execution. Those differences can accumulate in larger models and
longer training runs. Long-horizon training does not require bit-identical
weights or trajectories against Go: target industry-band agreement or better,
preserve the intended training semantics, and report the achieved band and
deviations. Pursue exactness where practical without making long-horizon bit
perfection a prerequisite for shipping.

Small regression fixtures may retain stronger exact checks. Their documented
gates record what has been verified; they do not impose universal exactness on
every model or training duration. The engine completion scoreboard below is
separate from this numerical parity spectrum.

---

## Scoreboard and credit rules

| Mark | Meaning | Credit |
|---|---|---:|
| ⬜ | Planned; no validated CUDA implementation | 0% |
| 🚧 | Implemented subset with reproducible evidence + listed gaps | 50% of gate |
| ✅ | Declared acceptance gate passes (w2a + device identity) | 100% of gate |
| N/A | Outside CUDA scope (reason required) | No credit |
| BLOCKED | Missing device / SDK / unresolved semantics | 0% |

**Version:** `welvet_version() = 0.{round(earned)}` until **100 → 1.0.0**. Only this engine board moves the tag. M0 host scaffolding does **not** earn engine points.

---

## Engine rebuild scoreboard — 100 points

Identical ten categories / weights as Go Welvet. Credit only from **this** CUDA tree’s evidence.

| # | Category (Go board) | Wt | CUDA today | Status | Earned |
|---:|---|---:|---|:---:|---:|
| 1 | **Foundation** — `core`, `weights`, quant, CUDA compute base, tiling policy | 15 | dtype/act 3 + FormatNone×34 store 4 + CUDA MatVec path 2; quant/tiling open | 🚧 | **9** |
| 2 | **Dense MatVec** — FormatNone×34 + quants; fwd/bwd/update on CUDA | 15 | FormatNone fwd+bwd+SGD 12; packed quant open | 🚧 | **12** |
| 3 | **Transformer stack** — MHA, SwiGLU, norms, Softmax, Emb, Sequential, Residual, seqmix | 14 | leaf FormatNone + NormalBP leaf/vol signed | ✅ | **14** |
| 4 | **CNN / RNN / LSTM** | 6 | CNN1–3 + RNN + LSTM leaf fwd + NormalBP | ✅ | **6** |
| 5 | **Extended layers** — GDN, ConvT1–3, Mamba, KMeans, Parallel, Metacognition | 7 | leaf fwd + NormalBP (incl. Stack); seqmix metadata | ✅ | **7** |
| 6 | **Runtime + architecture** — volumetric grid, forward, backward, training, step/mesh | 8 | Volume NormalBP + Step/Mesh/Tween* Line clocks + proxy volumes | ✅ | **8** |
| 7 | **Systems** — dna, evolution, tween, tanhi, telemetry | 5 | systems/tween* (+ HeadProxy…Sparse leaf/volume) + tween_alt | 🚧 | **5** |
| 8 | **Model / IO** — tokenizer, entity, transformer, sampling, hf | 8 | — | ⬜ | **0** |
| 9 | **Training credit** — 29 modes + cameral BranchModes / CamSync | 8 | … + all MeshTween* Split/Alt/proxy; Step proxy + CamSync open | 🚧 | **8** |
| 10 | **Peak fused / no host ALU** — fused CUDA kernels, decoder fuse, provenance | 14 | — | ⬜ | **0** |
| | **Total → version** | **100** | | **🚧** | **73 / 100 → `0.73`** |

**Path to 1.0:** same order Go used / welvet.cpp is following — Dense FormatNone + NormalBP → layers → Step/Mesh/Tween* → Systems + Model/IO → CamSync + peak fused CUDA. welvet.cpp’s ✅ list is the **port order**, not transferred credit.

---

## Full engine subsystem runway

### Foundation, weights and CUDA compute — §1 / §2

| Go reference | CUDA rebuild target | Status |
|---|---|:---:|
| `core/` | Tensor representation, all 34 dtypes, activations, layer IDs, execution config | 🚧 dtype/act + Dense gates |
| `weights/` | Dtype-native parameter storage, MatVec/transpose, grad/update, snapshots — **cudaMalloc buffer resident** | 🚧 FormatNone×34 + resident f64 MatVec fwd |
| `quant/` | All 20 packed formats Pack/Unpack/MatVec on CUDA (or honest unsupported) | ⬜ |
| `tiling/` | Threadgroup / grid dispatch policy, tails, workspace budgets, deterministic scheduling | ⬜ |
| `simd/` | N/A as Plan 9 — replaced by CUDA kernels (do not ship host AVX as “CUDA”) | N/A |
| `webgpu/` | Reference for device semantics only — implement as CUDA, not WGSL/Metal | N/A |
| `fusedgpu/` | Decoder/op fusion with resident intermediates + execution provenance on CUDA | ⬜ |
| CUDA runtime | Device, library, pipeline, command buffer, sync, error surface | 🚧 M0 host + Dense MatVec |

### Architecture and runtime — §6

| Go reference | CUDA rebuild target | Status |
|---|---|:---:|
| `architecture/` | Own the 3D volume, coordinates, cell layers, graph links, topology validation | 🚧 dims stub |
| `runtime/dispatch/` | Dispatch every supported op/variant with explicit CUDA capability checks | ⬜ |
| `runtime/forward/` | Full model/volume execution without Go orchestration | ⬜ |
| `runtime/backward/` | Reverse dependency traversal, tapes, gradients, joins, shared params on device | ⬜ |
| `runtime/training/` | Loss/credit/update, batches, state, training-mode selection on CUDA | ⬜ |
| `runtime/step/` | Systolic pipes, Line/Stack ticks, fill/drain, weight generations, RoleTrain/RoleAction | ⬜ |
| Mesh / volumetric scheduling | Volume dependencies, branch joins, cross-cell credit on CUDA-resident state | ⬜ |

### Systems — engine §7

| Go reference | Wt | CUDA rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `systems/dna/` | 1 | Encode/decode topology + params; native dtype preserved | ⬜ | 0 |
| `systems/evolution/` | 1 | DNA splice, mutation, cloning, NEAT-compatible behavior | ⬜ | 0 |
| `systems/tween/` | 1 | ModeTween project_gap + apply_projected + train_volume_mse (+ ModeTweenChain / ModeTweenSplit siblings) | ✅ | 1 |
| `systems/tanhi/` | 1 | TANHI structural/activation reporting + telemetry transport | ⬜ | 0 |
| `systems/telemetry/` | 1 | Blueprint, counts, estimates, execution/learning observability | ⬜ | 0 |
| **Subtotal** | **5** | | **⬜** | **0 / 5** |

### Model / IO — engine §8

| Go reference | Wt | CUDA rebuild target | Status | Earned |
|---|---:|---|:---:|---:|
| `model/tokenizer/` | 1.5 | BPE/HF tokenization, vocab/specials, encode/decode parity | ⬜ | 0 |
| `model/entity/` | 2 | ENTITY inspect/load/write, dtype-native blobs, cameral/Stack round trips | ⬜ | 0 |
| `model/transformer/` | 2.5 | Decoder construction, generation, KV cache, LM head, runtime integration | ⬜ | 0 |
| `model/sampling/` | 1 | ArgMax, temperature/TopK, penalties, banned IDs, deterministic RNG | ⬜ | 0 |
| `model/hf/` | 1 | Snapshot inspect, architecture detect, declared tensor-loader conversions | ⬜ | 0 |
| **Subtotal** | **8** | | **⬜** | **0 / 8** |

### Lucy / stubs / apps

| Surface | CUDA runway | Board | Status |
|---|---|---|:---:|
| `lucy/` | LPD / TrainStackCE / Lean density / Score | §8/§9 gates before credit | ⬜ |
| `model/wav2vec2/` | Inventory + port applicable speech paths | model scope | ⬜ |
| CamSync / BlendStores | Soft/hard blend across resident stores | §9 | ⬜ |
| `stub/` seed · serialization · hardware · memory · fountain · donate · … | Inventory; stubs ≠ done | off-board until gated | ⬜ |
| `apps/` (octo, …) | Native CLI shells on CUDA engine | off-board | ⬜ |

**Standalone acceptance:** a native CUDA-backed C++ app can construct or load a model, create its **volume**, execute forward, run the selected training rule, save/reload supported state, and continue — **no Go process**. FFI-only layer parity is insufficient.

---

## Coverage axes

A support claim is a tuple:

```text
(layer + variant + shape + numerical dtype + activation type
 + gradient type + accumulator type + pass + training mode
 + CUDA device/path + OS/arch + layout + residency + execution mode)
```

| Axis | Runway |
|---|---|
| Topology | `1×1×1`, `2×2×2`, `3×3×3`, non-cubic, cell stacks, connected/remote boundaries |
| Layer inventory | 23 kinds in Go `core/layer.go` + `layers/seqmix` |
| Storage dtype | All 34 `core.DType` |
| Quant | FormatNone + all 20 `quant.Format` (or explicit unsupported) |
| Activation host type | Go `core.Numeric` kinds; ABI uses explicit representations |
| Pass | Forward; dX; dW/db; state gradients when applicable |
| Training | Full backward + independently callable credit/update primitives |
| Execution | CUDA compute queues; documented host epilogue if any |
| Device | Named NVIDIA GPU (+ non-CUDA only if separately validated) |
| Platform | Linux x86_64 first; Windows later (optional) |
| Shape/layout | Batch, sequence, rank, stride, alignment, block tails |
| Precision | Stored / transferred / decoded / accumulated / output / grad / update |
| State | Stateless, recurrent, KV cache, saved activations, versioned weights, CamSync |

---

## DTypes (`core.DType`) — 34 numerical types

FormatNone storage. CUDA column = validated on-device path (not host decode pretending to be CUDA).

| # | DType | Pack/RT | CUDA F | CUDA B | CUDA update | Notes |
|--:|-------|:------:|:------:|:------:|:------:|-------|
| 0 | Float64 | ✅ | ✅ | ⬜ | ⬜ | |
| 1 | Float32 | ✅ | ✅ | ⬜ | ⬜ | likely first ALU |
| 2 | Float16 | ✅ | ✅ | ⬜ | ⬜ | native CUDA half |
| 3 | BFloat16 | ✅ | ✅ | ⬜ | ⬜ | |
| 4 | FP8E4M3 | ✅ | ✅ | ⬜ | ⬜ | |
| 5 | FP8E5M2 | ✅ | ✅ | ⬜ | ⬜ | |
| 6 | Int64 | ✅ | ✅ | ⬜ | ⬜ | |
| 7 | Int32 | ✅ | ✅ | ⬜ | ⬜ | |
| 8 | Int16 | ✅ | ✅ | ⬜ | ⬜ | |
| 9 | Int8 | ✅ | ✅ | ⬜ | ⬜ | |
| 10 | Uint64 | ✅ | ✅ | ⬜ | ⬜ | |
| 11 | Uint32 | ✅ | ✅ | ⬜ | ⬜ | |
| 12 | Uint16 | ✅ | ✅ | ⬜ | ⬜ | |
| 13 | Uint8 | ✅ | ✅ | ⬜ | ⬜ | |
| 14 | Int4 | ✅ | ✅ | ⬜ | ⬜ | |
| 15 | Uint4 | ✅ | ✅ | ⬜ | ⬜ | |
| 16 | FP4 | ✅ | ✅ | ⬜ | ⬜ | |
| 17 | Int2 | ✅ | ✅ | ⬜ | ⬜ | |
| 18 | Uint2 | ✅ | ✅ | ⬜ | ⬜ | |
| 19 | Ternary | ✅ | ✅ | ⬜ | ⬜ | |
| 20 | Binary | ✅ | ✅ | ⬜ | ⬜ | |
| 21 | Int | ✅ | ✅ | ⬜ | ⬜ | ABI width explicit |
| 22 | Uint | ✅ | ✅ | ⬜ | ⬜ | |
| 23 | Uintptr | ✅ | ✅ | ⬜ | ⬜ | never serialize as tensor id |
| 24 | Complex64 | ✅ | ✅ | ⬜ | ⬜ | |
| 25 | Complex128 | ✅ | ✅ | ⬜ | ⬜ | |
| 26 | NF4 | ✅ | ✅ | ⬜ | ⬜ | |
| 27 | FP6 | ✅ | ✅ | ⬜ | ⬜ | |
| 28 | Int6 | ✅ | ✅ | ⬜ | ⬜ | |
| 29 | Uint6 | ✅ | ✅ | ⬜ | ⬜ | |
| 30 | Int5 | ✅ | ✅ | ⬜ | ⬜ | |
| 31 | Uint5 | ✅ | ✅ | ⬜ | ⬜ | |
| 32 | Int3 | ✅ | ✅ | ⬜ | ⬜ | |
| 33 | Uint3 | ✅ | ✅ | ⬜ | ⬜ | |

### Numerical semantics gates

| Concern | Status |
|---|:---:|
| Integer/low-bit learning rules match Go | 🚧 Dense forward exact_bit×34 |
| Binary/ternary bit order, codes, masks | 🚧 Dense forward exact_bit |
| Complex real/imag layout | 🚧 Dense forward exact_bit |
| FP8/FP6/FP4/NF4 exact code tables | 🚧 Dense forward exact_bit |
| Accumulator dtype + reduction order + reproducibility policy | 🚧 HW f64 MatVec = Go contract |
| Explicit lossy conversion permission + scratch lifetime | 🚧 decode→f64 W on mount |
| Unsupported arithmetic → hard error (no silent widen) | 🚧 |

---

## Quant formats (`quant.Format`) — 20

Packed layouts. Row **None** = FormatNone + DType (not “FP32 master”).

| Format | CUDA pack+MatVec | CUDA Dense F | CUDA Dense B/update |
|--------|:-----------------:|:-------------:|:--------------------:|
| None | ✅ (via weights + DType) | ✅ | ⬜ |
| Q8_0 | ⬜ | ⬜ | ⬜ |
| Q4_0 | ⬜ | ⬜ | ⬜ |
| Q4_1 | ⬜ | ⬜ | ⬜ |
| Q5_0 | ⬜ | ⬜ | ⬜ |
| Q5_1 | ⬜ | ⬜ | ⬜ |
| Q2_K | ⬜ | ⬜ | ⬜ |
| Q3_K | ⬜ | ⬜ | ⬜ |
| Q4_K | ⬜ | ⬜ | ⬜ |
| Q5_K | ⬜ | ⬜ | ⬜ |
| Q6_K | ⬜ | ⬜ | ⬜ |
| IQ1_S | ⬜ | ⬜ | ⬜ |
| IQ2_XXS | ⬜ | ⬜ | ⬜ |
| IQ2_XS | ⬜ | ⬜ | ⬜ |
| IQ3_XXS | ⬜ | ⬜ | ⬜ |
| IQ3_S | ⬜ | ⬜ | ⬜ |
| IQ4_NL | ⬜ | ⬜ | ⬜ |
| IQ4_XS | ⬜ | ⬜ | ⬜ |
| TernaryPacked | ⬜ | ⬜ | ⬜ |
| BinaryPacked | ⬜ | ⬜ | ⬜ |
| AffinePacked | ⬜ | ⬜ | ⬜ |

---

## Layer runway — 23 kinds + seqmix

`dX` = input-gradient contract; `dW/db` = every trainable parameter. CUDA F/B = on-device (host epilogue must be labeled).

| ID | Layer | Forward target | Backward target | CUDA F | CUDA B |
|---:|---|---|---|:---:|:---:|
| 0 | **Dense** | GEMV/GEMM, bias, activation | dX, dW, act'; ApplyGradSGD | ✅ | ✅ |
| 1 | **MultiHeadAttention** | Q/K/V/O, attention, masks, RoPE, cache | Proj + attn + RoPE grads (ModeSelf) | ✅ | ✅ |
| 2 | **SwiGLU** | Gate/up/down, SiLU product | All projs + product/SiLU′ | ✅ | ✅ |
| 3 | **RMSNorm** | RMS + scale | dX + scale grad | ✅ | ✅ |
| 4 | **CNN1** | 1D conv | dX, kernel grads | ✅ | ✅ |
| 5 | **CNN2** | 2D conv | dX, kernel grads | ✅ | ✅ |
| 6 | **CNN3** | 3D conv | dX, kernel grads | ✅ | ✅ |
| 7 | **RNN** | Recurrent + state | BPTT + state grads | ✅ | ✅ |
| 8 | **LSTM** | Gates + h/c | Gate/param/h/c grads | ✅ | ✅ |
| 9 | **LayerNorm** | Mean/var, scale, bias | dX, scale/bias | ✅ | ✅ |
| 10 | **ConvTransposed1D** | 1D transposed | dX, kernel | ✅ | ✅ |
| 11 | **ConvTransposed2D** | 2D transposed | dX, kernel | ✅ | ✅ |
| 12 | **ConvTransposed3D** | 3D transposed | dX, kernel | ✅ | ✅ |
| 13 | **Embedding** | Gather | Scatter/reduction; defined token dX | ✅ | ✅ |
| 14 | **KMeans** | Soft assignment + centroids | Declared differentiable paths | ✅ | ✅ |
| 15 | **Softmax** | Standard/grid/temp/variants | JVPs (dX; weightless) | ✅ | ✅ |
| 16 | **Parallel** | Branches, routing, combine, cameral | Branch credit + BranchModes | ✅ | ✅ |
| 17 | **Sequential** | Ordered child chain | Reverse child credit | ✅ | ✅ |
| 18 | **Residual** | F(x) + skip | F reverse + skip grad | ✅ | ✅ |
| 19 | **Metacognition** | Observed compute + rules | lastGate-scale + Observed bwd | ✅ | ✅ |
| 20 | **Mamba** | Selective SSM scan | Declared BPTT + projs | ✅ | ✅ |
| 21 | **GDN** | Gated DeltaNet | Truncated-BPTT contract (f32-first) | ✅ | ✅ |
| 22 | **Stack** | Heterogeneous nested chain | Nested child credit | ✅ | ✅ |
| helper | **seqmix** | Sequence-mixing helpers | Helper grads | ✅ | ✅ |

### Layer acceptance details

| Family | Required before ✅ | Status |
|---|---|:---:|
| Dense | Transpose, batch, tails, strides, bias, act, dX-only, dW-only, update-only, dtype boundaries | 🚧 fwd×34 |
| Attention | Causal/cross/GQA; masks; ALiBi; sliding; dropout/RNG; KV invalidation | 🚧 ModeSelf fwd×34 |
| Norms | Epsilon, axes, small variance, saved stats, param grads | 🚧 RMS/LN fwd×34 |
| Softmax | Stable reductions; masks; temperature; Sparsemax/Entmax/Gumbel where present | 🚧 all Kind fwd×34 |
| Activations | Linear, ReLU, SiLU, GELU, Tanh, Sigmoid, LeakyReLU, ReLU2 + derivatives | 🚧 Dense/CNN acts |
| Convolution | Non-square/cubic, dilation, groups, overlap accum, transpose shapes | 🚧 CNN/ConvT fwd×34 |
| Recurrence/SSM | Initial/final state, sequence bounds, reset, tape, truncation | 🚧 RNN/LSTM/Mamba/GDN fwd |
| Composition | Nested non-Dense; shared stores; branch order; no duplicate updates | 🚧 Seq/Res/Parallel fwd |
| Discrete/index/rule | Explicit differentiability matching Go | 🚧 Emb/KMeans/Meta fwd |

### Dense first slice (M1)

| # | Case | Status |
|---|---|:---:|
| 1 | HW float64 MatVec on CUDA (Go FormatNone accumulate contract) | ✅ |
| 2 | Bias + activation on device | ✅ |
| 3 | Leaf forward exact_bit vs Go Float32 | ✅ |
| 4 | Leaf forward ×34 FormatNone exact_bit vs Go | ✅ `go_v_cuda` |
| 5 | NormalBP dX/dW/db + SGD Float32 resident | ⬜ |
| 6 | NormalBP ×34 + volume `1×1×1` / `2×2×2` | ⬜ |
| 7 | Speed table Go / C++ / CUDA + device identity | ✅ CUDA warm-exec wins ×34 |

---

## Kernel and training runway

### Shared compute primitives

| Primitive | Forward/use | Backward/update | Status |
|---|---|---|:---:|
| GEMV/GEMM | Dense + shared projections | GEMVᵀ, dX, outer/dW, db | 🚧 f64 GEMV fwd (Dense/SwiGLU/MHA/…) |
| Reduction | Sum, mean, var, max, norm, LSE | Reduction grads + broadcast | 🚧 norms + Softmax fwd |
| Elementwise | Bias, mul, add, acts, gates | Derivatives / JVPs | 🚧 Dense/SwiGLU/Residual fwd |
| Layout/shape | View, transpose, reshape, concat, split | Inverse mapping | ⬜ |
| Gather/scatter | Embedding, indexing, routing | Duplicate-index policy | ⬜ |
| Convolution | im2col or native CUDA conv | Overlap accum + filter grads | ⬜ |
| Attention/position | QK, scale, mask, AV, RoPE | Attn + positional grads | ⬜ |
| Scan/recurrence | RNN/LSTM/Mamba/GDN | Declared BPTT/truncation | ⬜ |
| Optimizer/update | Storage-preserving SGD first | Masks, LR, rounding | ⬜ |

---

## Training modes — all 29 + cameral

Canonical equations: [Welvet `training_modes.md`](https://github.com/openfluke/welvet/blob/main/training_modes.md).  
Tick **Inside node** (`1×1×1`) and **3D volumetric** separately. **Signed** = all supported FormatNone dtypes for that layer under the mode.

| # | TrainMode | Family | Inside node | 3D volumetric | Signed | Ver |
|---|---|---|:---:|:---:|:---:|---|
| 1 | NormalBP | BP | ✅ | ✅ | ✅ | 0.51 |
| 2 | StepBP | BP | ✅ | ✅ | ✅ | 0.52 |
| 3 | MeshBP | BP | ✅ | ✅ | ✅ | 0.53 |
| 4 | Tween | Tween | ✅ | ✅ | ✅ | 0.54 |
| 5 | StepTween | Tween | ✅ | ✅ | ✅ | 0.62 |
| 6 | MeshTween | Tween | ✅ | ✅ | ✅ | 0.60 |
| 7 | TweenChain | TweenChain | ✅ | ✅ | ✅ | 0.55 |
| 8 | StepTweenChain | TweenChain | ✅ | ✅ | ✅ | 0.63 |
| 9 | MeshTweenChain | TweenChain | ✅ | ✅ | ✅ | 0.61 |
| 10 | TweenSplit | TweenSplit | ✅ | ✅ | ✅ | 0.56 |
| 11 | StepTweenSplit | TweenSplit | ✅ | ✅ | ✅ | 0.57 |
| 12 | MeshTweenSplit | TweenSplit | ✅ | ✅ | ✅ | 0.70 |
| 13 | TweenAlt | TweenAlt | ✅ | ✅ | ✅ | 0.58 |
| 14 | StepTweenAlt | TweenAlt | ✅ | ✅ | ✅ | 0.59 |
| 15 | MeshTweenAlt | TweenAlt | ✅ | ✅ | ✅ | 0.71 |
| 16 | TweenSplitHeadProxy | TweenSplit | ✅ | ✅ | ✅ | 0.64 |
| 17 | TweenSplitLinear | TweenSplit | ✅ | ✅ | ✅ | 0.65 |
| 18 | TweenSplitFastProxy | TweenSplit | ✅ | ✅ | ✅ | 0.66 |
| 19 | TweenSplitLinearCache | TweenSplit | ✅ | ✅ | ✅ | 0.67 |
| 20 | TweenSplitHeadProxyAsync | TweenSplit | ✅ | ✅ | ✅ | 0.68 |
| 21 | TweenSplitSparse | TweenSplit | ✅ | ✅ | ✅ | 0.69 |
| 22 | MeshTweenSplitFastProxy | TweenSplit | ✅ | ✅ | ✅ | 0.72 |
| 23 | MeshTweenSplitSparse | TweenSplit | ✅ | ✅ | ✅ | 0.73 |
| 24 | StepTweenSplitHeadProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 25 | StepTweenSplitLinear | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 26 | StepTweenSplitFastProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 27 | StepTweenSplitLinearCache | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 28 | StepTweenSplitHeadProxyAsync | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 29 | StepTweenSplitSparse | TweenSplit | ⬜ | ⬜ | ⬜ | |
| — | Freeze | Freeze | ⬜ | ⬜ | ⬜ | |
| — | Shadow | Freeze | ⬜ | ⬜ | ⬜ | |
| — | Adversarial | BP-gated | ⬜ | ⬜ | ⬜ | |
| — | Memory | BP-gated | ⬜ | ⬜ | ⬜ | |
| — | CamSync / BlendStores | — | ⬜ | ⬜ | ⬜ | |
| — | Cameral BranchModes (Parallel) | — | ⬜ | ⬜ | ⬜ | |

Per-layer evidence tables (leaf ×34, volume shapes, speed, wires): **[docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md)**. Mode summary table above stays the high-level board; tick detailed rows in the checklist as w2a `go_v_cuda` suites land.

### Sign-off template (copy per mode × layer)

| # | Case | Status |
|---|---|:---:|
| 1 | Leaf Float32 (grads/weights/loss as applicable) | ⬜ |
| 2 | Leaf × all 34 FormatNone dtypes | ⬜ |
| 3 | Volume `1×1×1×1` Float32 | ⬜ |
| 4 | Volume `1×1×1×2` chain Float32 | ⬜ |
| 5 | Volume `1×1×1×1` ×34 | ⬜ |
| 6 | Volume `2×2×2×1` Float32 | ⬜ |
| 7 | Volume `2×1×1×1` ×34 | ⬜ |
| 8 | Speed: goµs / cudaµs / spd%Δ + `welvet_runtime_identity` | ⬜ |

---

## Volumetric execution and placement

| Capability | Acceptance | Status |
|---|---|:---:|
| Single cell | Layer stack in `1×1×1` on CUDA with Go parity | ⬜ |
| Multi-cell volume | Coords, ordering, links, boundary tensors intact | ⬜ |
| Non-cubic volumes | No hidden cubic-only assumption | ⬜ |
| Per-layer placement | Cell/layer/child → queryable CUDA path | ⬜ |
| Partition fusion | Fuse consecutive ops without changing visible semantics | ⬜ |
| Heterogeneous forward | Explicit host↔CUDA handoff | ⬜ |
| Heterogeneous backward | Gradient edges + tape ownership across boundary | ⬜ |
| Step systolic | Fill/drain, ticks, Line/Stack, RoleTrain/RoleAction | ⬜ |
| Mesh execution | Dependency order, joins, credit over links | ⬜ |
| Placement cost | Upload/download, queue latency, compile, memory pressure | ⬜ |
| Plan invalidation | Rebuild on shape/topology/weights/device/precision/mode change | ⬜ |

**Placement:** discover → query exact support → estimate cost → select → record → execute. Tiny `1×1×1` may prefer host; that must be an explicit plan, not a silent lie labeled CUDA.

---

## CUDA platform runway

| Gate | Status |
|---|:---:|
| Linux x86_64 CMake + nvcc build | ✅ |
| Device enumerate + runtime identity string | ✅ |
| Compile-time CUDA kernels (`kernels/*.cu`) | ✅ |
| Device buffer alloc / upload / download | ✅ |
| Smoke kernel `welvet_smoke_saxpy` launch | ✅ |
| Stream / sync / error surface helpers | ⬜ |
| Fatbin / multi-arch release packaging | ⬜ |
| Determinism policy doc (reduction order, warps, bands) | ⬜ |
| Windows CUDA artifact | ⬜ |
| Other vendors (Metal/ROCm/QNN) | N/A sibling repos |

| Platform | Artifact | Status |
|---|---|:---:|
| Linux x86_64 | `.so` / headers / kernels / wire tool | 🚧 M1 Dense fwd |
| Linux aarch64 | Explicitly bounded later | ⬜ |
| Windows x86_64 | Optional later | ⬜ |


## FFI and C ABI

| Surface | Status |
|---|:---:|
| ABI negotiation / version | ✅ `welvet_version` = `0.27` |
| CUDA init / available / identity | ✅ |
| Gaps census counters | ✅ |
| Tensor / weight / layer descriptors | 🚧 leaf FormatNone descriptors |
| Engine/model handles (volume create/load/train/checkpoint) | ⬜ |
| Dense forward / NormalBP wire magics (w2a) | 🚧 all leaf FormatNone fwd magics; NormalBP ⬜ |
| All layer NormalBP magics (mirror welvet.cpp names) | ⬜ |
| Volume / Step / Mesh magics + ModeTween TWB1/TWV1… + ModeTweenChain TCB1/TCV1… + ModeTweenSplit TSB1/TSP1… + ModeStepTweenSplit STS1/STSO + ModeTweenAlt TAB1/TAV1… + ModeStepTweenAlt STA1/STAO + MeshTween MTW1/MGW1 + MeshTweenChain MTC1/MGC1 + ModeStepTween STW1/STWO + ModeStepTweenChain STC1/STCO | ✅ StepTweenChain |
| Go NAC dispatcher selects CUDA | ⬜ |

Boundary rules match welvet.cpp: versioned descriptors, opaque handles, no C++ exceptions through C, explicit ownership, async inputs owned/copied, provenance in traces.

---

## Resident weights and memory

| Capability | Status |
|---|:---:|
| Authoritative owner (host vs CUDA generation) | 🚧 FormatNone host + f64 GPU mirror |
| Upload once / reuse across fwd/bwd/update | 🚧 upload-once W; warm exec reuses |
| Preserve storage truth (no undeclared f32 master) | ✅ FormatNone WeightStore |
| Derived device layout cache + invalidation | 🚧 Dense mount decode→f64 |
| Resident activations / tapes across adjacent layers | ⬜ |
| Resident gradients + clear policy | ⬜ |
| Resident update in declared storage semantics | ⬜ |
| Host-update mode (export grads → refresh device) | ⬜ |
| CamSync on resident stores | ⬜ |
| Checkpoint/export for ENTITY flows | ⬜ |
| Migration before device/plan change | ⬜ |
| Memory budget accounting | ⬜ |
| Unified/shared memory only under explicit sync contract | ⬜ |
| Concurrency / device-loss honesty | ⬜ |

---

## Validation and performance (w2a)

| Gate | Status |
|---|:---:|
| Pin Welvet + welvet.cpp revisions in fixtures | ⬜ |
| `Go_v_Cuda` harness (parity + speed) | ⬜ |
| Optional C++↔CUDA cross-check vs welvet.cpp on same Linux box | ⬜ |
| Standalone construct→volume→train→save/reload | ⬜ |
| Numerical / forward / backward / train / volume / modes matrices | ⬜ |
| Capability honesty (unsupported fails clearly) | ⬜ |
| Device identity on every evidence log | 🚧 helper ready |
| Distribution smoke on released artifact | ⬜ |

### Performance template

| Workload | dtype | pass/mode | Go µs | CUDA µs | spd%Δ | exact? | device | evidence |
|---|---|---|---:|---:|---:|:---:|---|---|
| Dense GEMV batch=8 in/out=256 | float32 | FormatNone fwd | 919 | 68 | −93% | exact_bit | GTX 1650 SUPER | `TestGoVCudaDenseForward` |
| Dense GEMV ×34 | all 34 | FormatNone fwd | 0.9–9.4ms | 68–94µs | −92…−99% | exact_bit | GTX 1650 SUPER | CUDA warm-exec wins 34/34 |

Measure cold compile, initial upload (`cuda_mount`), warm resident (`cuda_exec`), sync, export separately. Winner column uses warm exec.

---

## Implementation sequence

| Milestone | Concrete outcome | Status |
|---|---|:---:|
| **M0** — CUDA host + this README runway | Device/library/buffer/smoke + full Go inventory checklist | ✅ |
| **M1** — Dense FormatNone forward on CUDA | w2a `go_v_cuda` exact_bit×34 + Go/C++/CUDA speed | ✅ |
| **M2** — Dense NormalBP resident | Multi-step train parity leaf+vol | ✅ |
| **M3** — Dtype / quant breadth | ×34 (+ quants or honest gaps) | 🚧 FormatNone×34; quants open |
| **M4** — Layer breadth | All 23 layers leaf fwd + NormalBP | ✅ |
| **M5** — Full volumetric runtime | Grid + Step/Mesh + credit families | ⬜ |
| **M6** — Systems + Model/IO + Lucy | DNA…telemetry, ENTITY/HF/…, LPD | ⬜ |
| **M7** — Peak fused CUDA | Minimal host ALU + decoder fuse | ⬜ |
| **M8** — Windows + non-CUDA inventory | Device artifact; non-CUDA separate | ⬜ |
| **M9** — NAC + release review | Go delegate + clean-env smoke; board → 1.0.0 | ⬜ |

---

## Source layout

```text
include/welvet/              public C++ engine API
include/welvet/cuda/         device, buffer, kernels
include/welvet_c.h           stable C ABI
src/core/                    dtype, activate, tensor, status
src/cuda/                    CUDA Runtime host (device/buffer/launch)
src/weights/                 FormatNone store → resident device buffers
src/layers/…                 layer packages (Dense first)
src/architecture/            3D Grid / volume
src/runtime/                 forward, backward, training, step, mesh
src/systems/                 dna, evolution, tween, tanhi, telemetry
src/gaps/                    census (mirrors this README)
src/interop/                 C ABI
kernels/                     .cu kernels (compiled by nvcc)
tools/dense_fwd/             w2a wire tool
examples/cuda_smoke/         M0 smoke
scripts/build_linux.sh
scripts/linux_build_w2a.sh   build + invoke sibling w2a go_v_cuda
README.md                    ← you are here (authoritative runway)
docs/TRAINING_CHECKLIST.md   training port tracker (tick as w2a suites land)
docs/CUDA_ONLY.md            CUDA-only execution contract
GAPS.md                      machine-oriented gap dump
VERSIONING.md                0.x formula
```

---

## How to use this README

1. Pick the next ⬜ (after M0: Dense MatVec dispatch → M1).
2. Implement CUDA kernels + host wiring; preserve Go / welvet.cpp semantics.
3. Prove in w2a (`Go_v_Cuda`): determinism + speed on named device.
4. Flip the mark in **this file**, sync `GAPS.md`, recompute earned points, bump `welvet_version` / CMake `VERSION`.
5. Repeat until the engine board is 100 and the volume trains end-to-end on CUDA.

**Destination:** a complete standalone **3D volumetric** Welvet engine on NVIDIA GPU — same model contract as Go Welvet — with resident state, native API, optional FFI/NAC, and w2a evidence for every ✅.
