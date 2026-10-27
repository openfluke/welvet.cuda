# Training checklist — welvet.cuda

One **TrainMode** at a time. Finish that mode’s table (or the rows you care about), bump version, move on.

Canonical enum: Go `layers/parallel/train_mode.go` (`TrainMode`).  
`inherit` (0) = resolve-only — not a ship target.

**Port mirror of** [`welvet.metal/docs/TRAINING_CHECKLIST.md`](../../welvet.metal/docs/TRAINING_CHECKLIST.md)
and [`welvet.cpp/docs/TRAINING_CHECKLIST.md`](../../welvet.cpp/docs/TRAINING_CHECKLIST.md).
Same cases / wires / column meanings; status here is **CUDA** only. metal/cpp `[x]` does **not** transfer credit.

**Current engine:** `0.00` (M0 host scaffold — **0 / 100** on the engine board). No training rows signed yet.

---

## CUDA-only mandate (non-negotiable)

**welvet.cuda = GPU CUDA compute only.** Host/CPU math belongs in **welvet.cpp** (and Go).  
Host numerical training loops are forbidden. Host scheduling, validation, allocation and storage staging are permitted; residency and performance work remain separate gates.

Current executable contract: [CUDA_ONLY.md](CUDA_ONLY.md).

Validation lives in sibling **w2a** (`go_v_cuda`), not in this tree — same pattern as `go_v_metal` / `go_v_cpp`.

```sh
./scripts/build_linux.sh
./scripts/linux_build_w2a.sh --smoke          # M0
# later, as suites land in w2a:
# ./scripts/linux_build_w2a.sh --dense-train
# ./scripts/linux_build_w2a.sh --normalbp-all
# ./scripts/linux_build_w2a.sh --training-all
```

**Project parity target:** best-effort deterministic agreement with Go, aiming
for 100% and accepting 60–100% on w2a's spectrum (`industry` through `exact_bit`).
Long-horizon training need not reproduce an identical CPU trajectory. See
[the README policy](../README.md#numerical-parity-target).

---

### Columns (every table)

| Column | Means |
|---|---|
| **Inside node** | Layer trains correctly in a `1×1×1` cell (single-node / ordinary NN case). go_v_cuda / exact_bit (or declared band). |
| **3D volumetric** | Same mode works when that layer sits in a multi-cell Grid (volume Forward/Backward + credit across cells). |
| **Fully signed off** | **All** supported FormatNone dtypes for that layer under this mode (not just f32). Softmax = dX only. GDN = f32-only in Go → sign off when f32 path is done. |

Tick `[x]` when validated. **Ver** records a release tag when one exists, otherwise a dated unreleased validation. **Fully signed off** requires every supported dtype, not only Float32 smoke coverage.

---

## Mode summary (tick as suites land)

High-level board also lives in [README.md — Training modes](../README.md#training-modes--all-29--cameral). Detail evidence sections below grow as w2a wires land.

| # | TrainMode | Family | Inside node | 3D volumetric | Signed | Ver |
|---|---|---|:---:|:---:|:---:|---|
| 1 | NormalBP | BP | ⬜ | ⬜ | ⬜ | |
| 2 | StepBP | BP | ⬜ | ⬜ | ⬜ | |
| 3 | MeshBP | BP | ⬜ | ⬜ | ⬜ | |
| 4 | Tween | Tween | ⬜ | ⬜ | ⬜ | |
| 5 | StepTween | Tween | ⬜ | ⬜ | ⬜ | |
| 6 | MeshTween | Tween | ⬜ | ⬜ | ⬜ | |
| 7 | TweenChain | TweenChain | ⬜ | ⬜ | ⬜ | |
| 8 | StepTweenChain | TweenChain | ⬜ | ⬜ | ⬜ | |
| 9 | MeshTweenChain | TweenChain | ⬜ | ⬜ | ⬜ | |
| 10 | TweenSplit | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 11 | StepTweenSplit | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 12 | MeshTweenSplit | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 13 | TweenAlt | TweenAlt | ⬜ | ⬜ | ⬜ | |
| 14 | StepTweenAlt | TweenAlt | ⬜ | ⬜ | ⬜ | |
| 15 | MeshTweenAlt | TweenAlt | ⬜ | ⬜ | ⬜ | |
| 16 | TweenSplitHeadProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 17 | TweenSplitLinear | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 18 | TweenSplitFastProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 19 | TweenSplitLinearCache | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 20 | TweenSplitHeadProxyAsync | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 21 | TweenSplitSparse | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 22 | MeshTweenSplitFastProxy | TweenSplit | ⬜ | ⬜ | ⬜ | |
| 23 | MeshTweenSplitSparse | TweenSplit | ⬜ | ⬜ | ⬜ | |
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

---

## Layer × NormalBP (section 1) — evidence stubs

Wire names mirror welvet.cpp / welvet.metal where practical so w2a suites stay parallel. All unchecked at M0.

### Dense NormalBP

Run (future): `./scripts/linux_build_w2a.sh --dense-train`  
Wire: `WDB1`/`WDBO` (leaf), `WVN1`/`WVNO` (volume).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 ReLU (dX/dW/W/loss) | [ ] | |
| 2 | Leaf × all 34 dtypes | [ ] | |
| 3 | Volume 1×1×1×1 Float32 | [ ] | |
| 4 | Volume 1×1×1×2 chain Float32 | [ ] | |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [ ] | |
| 6 | Volume 2×2×2×1 Float32 | [ ] | |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [ ] | |

### Remaining NormalBP families

Tick a dedicated evidence section (same 7-case shape unless noted) when each lands:

| Family | Wire leaf/vol | Status |
|---|---|:---:|
| RMSNorm | `WRB1`/`WRBO`, `WRV1`/`WRVO` | ⬜ |
| LayerNorm | `WLB1`/`WLBO`, `WLV1`/`WLVO` | ⬜ |
| SwiGLU | `WSB1`/`WSBO`, `WSV1`/`WSVO` | ⬜ |
| Softmax | `WXB1`/`WXBO`, `WXV1`/`WXVO` | ⬜ |
| Embedding | `WEB1`/`WEBO`, `WEV1`/`WEVO` | ⬜ |
| MHA | `WHB1`/`WHBO`, `WHV1`/`WHVO` | ⬜ |
| CNN1/2/3 | (cpp/metal twin wires) | ⬜ |
| ConvT1/2/3 | (cpp/metal twin wires) | ⬜ |
| RNN / LSTM | (cpp/metal twin wires) | ⬜ |
| Sequential / Residual / Stack | (cpp/metal twin wires) | ⬜ |
| KMeans / Parallel | (cpp/metal twin wires) | ⬜ |
| Mamba / Metacognition / GDN | (cpp/metal twin wires) | ⬜ |

---

## Sign-off template (copy per mode × layer)

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

## How we work this file

1. Implement the next CUDA gate (Dense forward → NormalBP → next layer / mode).
2. Add or extend a `go_v_cuda` suite in sibling **w2a**; run via `linux_build_w2a.sh`.
3. Tick rows here + the README mode summary; sync `GAPS.md`; bump `welvet_version` / CMake when the engine board moves.
4. Never claim CUDA credit from welvet.cpp or welvet.metal checkmarks.
