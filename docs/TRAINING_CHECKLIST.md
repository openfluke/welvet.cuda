# Training checklist — welvet.cuda

One **TrainMode** at a time. Finish that mode’s table (or the rows you care about), bump version, move on.

Canonical enum: Go `layers/parallel/train_mode.go` (`TrainMode`).  
`inherit` (0) = resolve-only — not a ship target.

**Port mirror of** [`welvet.metal/docs/TRAINING_CHECKLIST.md`](../../welvet.metal/docs/TRAINING_CHECKLIST.md)
and [`welvet.cpp/docs/TRAINING_CHECKLIST.md`](../../welvet.cpp/docs/TRAINING_CHECKLIST.md).
Same cases / wires / column meanings; status here is **CUDA** only. metal/cpp checkmarks do **not** transfer credit.

**Current engine:** `0.57` (leaf FormatNone forwards + NormalBP + StepBP + MeshBP + ModeTween + ModeTweenChain + ModeTweenSplit + **ModeStepTweenSplit signed** — **57 / 100** on the engine board).
Leaf forward + NormalBP leaf/volume signed for all 23 layer kinds (+ seqmix metadata). `TestGoVCudaNormalBP` PASS. `TestGoVCudaStepBPTrain` PASS. `TestGoVCudaMeshBPTrain` PASS. `TestGoVCudaTweenTrain` PASS. `TestGoVCudaTweenChainTrain` PASS. `TestGoVCudaTweenSplitTrain` PASS. `TestGoVCudaStepTweenSplitTrain` PASS (130 groups). Remaining Tween* variants / CamSync remain open.

---

## CUDA-only mandate (non-negotiable)

**welvet.cuda = GPU CUDA compute only.** Host/CPU math belongs in **welvet.cpp** (and Go).  
Host numerical training loops are forbidden. Host scheduling, validation, allocation and storage staging are permitted; residency and performance work remain separate gates.

There is **no** silent host/CPU numerical fallback labeled as CUDA. Unsupported dtype/layer/mode → hard error.

Current executable contract: [CUDA_ONLY.md](CUDA_ONLY.md).

Validation lives in sibling **w2a** (`go_v_cuda`), not in this tree — same pattern as `go_v_metal` / `go_v_cpp`.

```sh
./scripts/build_linux.sh
./scripts/linux_build_w2a.sh --smoke
./scripts/linux_build_w2a.sh --leaf-fwd-all   # all leaf FormatNone forwards (signed)
# as suites land:
# ./scripts/linux_build_w2a.sh --dense-train
# ./scripts/linux_build_w2a.sh --normalbp-all
# ./scripts/linux_build_w2a.sh --training-all
```

## w2a Go↔CUDA (leaf FormatNone forward) — signed

Run: `./scripts/linux_build_w2a.sh --leaf-fwd-all` (or per-layer `--*-fwd`).  
Exact tensor bits vs Go (+ cpp speed column when present). Stack has no leaf forward magic.

| Layer | Test | Wire | Status |
|---|---|---|:---:|
| Dense | `TestGoVCudaDenseForward` | `WVD3`→`WVO1` | [x] |
| RMSNorm | `TestGoVCudaRMSNorm` | `WVR1`→`WVRO` | [x] |
| LayerNorm | `TestGoVCudaLayerNorm` | `WVL1`→`WVLO` | [x] |
| Softmax | `TestGoVCudaSoftmax` | `WSM1`→`WSMO` | [x] |
| SwiGLU | `TestGoVCudaSwiGLU` | `WVS1`→`WVSO` | [x] |
| Embedding | `TestGoVCudaEmbedding` | `WEM1`→`WEMO` | [x] |
| MHA | `TestGoVCudaMHA` | `WVM1`→`WVMO` | [x] |
| CNN1/2/3 | `TestGoVCudaCNN` | `WVC*`→`WVCO` | [x] |
| ConvT1/2/3 | `TestGoVCudaConvT` | `WVT*`→`WVTO` | [x] |
| RNN | `TestGoVCudaRNN` | `WRN1`→`WRNO` | [x] |
| LSTM | `TestGoVCudaLSTM` | `WLS1`→`WLSO` | [x] |
| Sequential | `TestGoVCudaSequential` | `WSQ1`→`WSQO` | [x] |
| Residual | `TestGoVCudaResidual` | `WRE1`→`WREO` | [x] |
| Mamba | `TestGoVCudaMamba` | `WMB1`→`WMBO` | [x] |
| Metacognition | `TestGoVCudaMetacognition` | `WMC1`→`WMCO` | [x] |
| GDN | `TestGoVCudaGDN` | `WGD1`→`WGDO` | [x] f32 |
| KMeans | `TestGoVCudaKMeans` | `WKM1`→`WKMO` | [x] |
| Parallel | `TestGoVCudaParallel` | `WPL1`→`WPLO` | [x] |
| seqmix | `TestGoVCudaSeqmix` | `WSX1`→`WSXO` | [x] |
| Stack | `TestGoVCudaStackTrain` | `STKB`→`STKO` / `STKV`→`STVO` | [x] NormalBP |

## w2a Go↔CUDA (Freeze / Shadow / Adversarial / Memory / Cameral / CamSync) — sign-off evidence

Not signed yet. Same scope as welvet.metal when the CUDA path lands: Freeze/Shadow, Adversarial/Memory, Cameral BranchModes, CamSync/BlendStores — leaf + placed volume + native dtype sweep — via w2a. No welvet.cpp or welvet.metal credit transfers.

| Scope | Inside node | Placed volume | Native dtype sweep | Evidence |
|---|:---:|:---:|:---:|---|
| Freeze / Shadow | ⬜ | ⬜ | ⬜ | |
| Adversarial / Memory | ⬜ | ⬜ | ⬜ | |
| Shadow teacher credit | ⬜ | ⬜ | ⬜ | |
| Cameral BranchModes | ⬜ | ⬜ | ⬜ | |
| CamSync / BlendStores | ⬜ | ⬜ | ⬜ | |
| Native replacement / coherence | ⬜ | ⬜ | ⬜ | |

### Numerical execution inventory

| Path | Execution today |
|---|---|
| NormalBP layers | ✅ Dense…Stack + Mamba + Metacognition + GDN(f32) + KMeans + Parallel leaf+vol signed |
| MSE / MSEGrad | ✅ host f64 match Go (Dense NormalBP) |
| SGD and native storage updates | ✅ FormatNone Dense SGD |
| GDN | ✅ f32 truncated-BPTT NormalBP leaf+vol |
| Tween / TweenChain / Split / Alt / proxy / mesh variants | 🚧 ModeTween + ModeTweenChain + ModeTweenSplit + ModeStepTweenSplit signed; remaining Tween* open |
| Scheduling and data movement | Host validation, scheduling, allocation, copies and storage staging (permitted) |
| GPU-only enforcement | M0 host scaffold only; numerical CUDA-only gate lands with first training suites |

**Project parity target:** best-effort deterministic agreement with Go, aiming
for 100% and accepting 60–100% on w2a's spectrum (`industry` through `exact_bit`).
Long-horizon training need not reproduce an identical CPU trajectory. See
[the README policy](../README.md#numerical-parity-target).

**Fixture precision policy (when suites land):** exact tensor bits where required by each case; finite loss
uses `abs(go-cuda) <= 1e-12 * max(1, abs(go))` unless a row declares another band. Document
measured bands before signing ✅. GDN / Softmax / MHA may use the same spectrum floors as welvet.metal once evidenced on CUDA.

Checkmarks below indicate validated CUDA correctness under these bands, not a speed
win or completion of GPU residency/performance work. Unsupported and unverified
modes remain unchecked. **Nothing below is signed at M0.**

---

### Columns (every table)

| Column | Means |
|---|---|
| **Inside node** | Layer trains correctly in a `1×1×1` cell (single-node / ordinary NN case). go_v_cuda / exact_bit (or declared band). |
| **3D volumetric** | Same mode works when that layer sits in a multi-cell Grid (volume Forward/Backward + credit across cells). |
| **Fully signed off** | **All** supported FormatNone dtypes for that layer under this mode (not just f32). Softmax = dX only. GDN = f32-only in Go → sign off when f32 path is done. |

Tick `[ ]` when validated. **Ver** records a release tag when one exists, otherwise a dated unreleased validation. **Fully signed off** requires every supported dtype, not only Float32 smoke coverage.

**All 29 training modes:** run `./scripts/linux_build_w2a.sh --training-all`
to build and run the full training matrix plus CUDA-only regressions.
Run `./scripts/linux_build_w2a.sh --training-25-29-all` for the final five Line
modes, shared proxy volumes and CUDA-only guard.

**NormalBP subset:** run `./scripts/linux_build_w2a.sh --normalbp-all` to build
CUDA and run both `TestGoVCudaNormalBP` and `TestGoVCudaOnly` in sibling `w2a`.
Run `./scripts/linux_build_w2a.sh --step-tween-all` for NormalBP plus all five
sections 2–6 and the CUDA-only guard.
The combined runners default to one measured benchmark iteration per fixture;
set `WELVET_CUDA_BENCH_ITERS=2000` for longer performance measurements.

---

## w2a Go↔CUDA (Dense NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --dense-train`  

Wire: `WDB1`/`WDBO` (leaf), `WVN1`/`WVNO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: warm-up + `--bench=N` (in-process; no spawn) → **goµs / cudaµs / spd%Δ** (`(cuda−go)/go×100`, neg ⇒ CUDA faster). Resident soft-double W (no remount each op); microbench spd%Δ is GPU-launch-bound vs Go host.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 ReLU (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔CUDA (RMSNorm NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --rmsnorm-train`  
Wire: `WRB1`/`WRBO` (leaf), `WRV1`/`WRVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention as Dense (leaf mount-once + γ reset; no remount each iter).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγ/γ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔CUDA (LayerNorm NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --layernorm-train`  
Wire: `WLB1`/`WLBO` (leaf), `WLV1`/`WLVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + γ/β reset).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dγβ/γβ/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔CUDA (SwiGLU NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --swiglu-train`  
Wire: `WSB1`/`WSBO` (leaf), `WSV1`/`WSVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + speed table |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 + speed table |

## w2a Go↔CUDA (Softmax NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --softmax-train`  
Wire: `WXB1`/`WXBO` (leaf), `WXV1`/`WXVO` (volume). Softmax = dX only (weightless).  
Timing: leaf mount-once (reuse layer; no W). Band: **≥loose** floor (Softmax hard to bit-match); leaf×34 / volume hit **strict (~80%)**; Kind×34 worst **loose**; Gumbel = smoke.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Standard Float32 (dX/loss) | [x] | strict + speed |
| 2 | Leaf all Kind Float32 (+ gumbel smoke) | [x] | ≥loose; gumbel non-det |
| 3 | Leaf Standard × all 34 dtypes | [x] | strict ×34 + speed |
| 4 | Leaf all Kind × all 34 dtypes | [x] | ≥loose (worst=loose) |
| 5 | Volume 1×1×1×1 Float32 | [x] | inside-node strict |
| 6 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer strict |
| 7 | Volume 1×1×1×1 × all 34 dtypes | [x] | strict ×34 |
| 8 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric strict |
| 9 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell strict ×34 |

## w2a Go↔CUDA (Embedding NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --embedding-train`  
Wire: `WEB1`/`WEBO` (leaf), `WEV1`/`WEVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + W reset). dX zeros (discrete IDs); dW scatter.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf × all 34 dtypes | [x] | exact_bit + speed table |
| 3 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 4 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 5 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 6 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed |
| 7 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔CUDA (MHA NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --mha-train`  
Wire: `WHB1`/`WHBO` (leaf), `WHV1`/`WHVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (resident Dense Q/K/V/O; microbench spd%Δ is launch-bound). ModeSelf baseline (no dropout/QKNorm/cross).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf DecoderCausal Float32 (dX/dW/W/loss) | [x] | exact_bit + speed |
| 2 | Leaf GQA + Encoder Float32 | [x] | exact_bit |
| 3 | Leaf DecoderCausal × all 34 dtypes | [x] | W exact_bit; dX/dW ≥strict (uint2 dW ~1 ULP) |
| 4 | Volume 1×1×1×1 Float32 | [x] | inside-node + speed |
| 5 | Volume 1×1×1×2 chain Float32 | [x] | multi-layer cell + speed |
| 6 | Volume 1×1×1×1 × all 34 dtypes | [x] | + spectrum |
| 7 | Volume 2×2×2×1 Float32 | [x] | 3D volumetric + speed; subnormal weight differences allowed |
| 8 | Volume 2×1×1×1 × all 34 dtypes | [x] | multi-cell ×34 |

## w2a Go↔CUDA (CNN1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --cnn-train`  
Wire: `C1B1`/`C1BO`, `C2B1`/`C2BO`, `C3B1`/`C3BO` (leaf); `C*V1`/`C*VO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). CUDA convolution gather forward + input/filter gradient kernels.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 CNN1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 CNN1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (ConvT1/2/3 NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --convt-train`  
Wire: `T1B1`/`T1BO`, `T2B1`/`T2BO`, `T3B1`/`T3BO` (leaf); `T*V1`/`T*VO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention (leaf mount-once + proj W reset). Scatter forward + act Jacobian + scatter adjoint (no bias).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1–3 | Leaf Float32 ConvT1/2/3 | [x] | exact_bit + speed |
| 4–6 | Leaf ×34 ConvT1/2/3 | [x] | exact_bit spectrum |
| 7–9 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving pad |
| 10–12 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (RNN NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --rnn-train`  
Wire: `RNB1`/`RNBO` (leaf); `RNV1`/`RNVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. BPTT via Dense IH/HH + tanh′; loom `[dIH|dHH|dBias]`.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (LSTM NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --lstm-train`  
Wire: `LSB1`/`LSBO` (leaf); `LSV1`/`LSVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. BPTT 4-gate Dense; loom `[i|f|g|o]` packs.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | in=hid chain |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (Sequential NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --sequential-train`  
Wire: `SQB1`/`SQBO` (leaf); `SQV1`/`SQVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. Reverse Dense Dim→Dim chain; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving Dim |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (Residual NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --residual-train`  
Wire: `REB1`/`REBO` (leaf); `REV1`/`REVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. F reverse Dense + skip grad; loom Depth×[Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | y=F(x)+x |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (Mamba NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --mamba-train`  
Wire: `MBB1`/`MBBO` (leaf); `MBV1`/`MBVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. Scan BPTT + In/Out Dense; loom In|Out|ALog|DSkip.

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | shape-preserving D |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (Metacognition NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --metacognition-train`  
Wire: `MCB1`/`MCBO` (leaf); `MCV1`/`MCVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. lastGate-scale + Observed Dense; loom [Dim×Dim].

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | default cool rules |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | scaled init for 8-hop |

## w2a Go↔CUDA (GDN NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --gdn-train`  
Wire: `GDB1`/`GDBO` (leaf); `GDV1`/`GDVO` (volume). ≥loose leaf / ≥industry volume vs Go (CUDA f32 scan; declared GPU precision floors).
Timing: same `--bench=N` / spd%Δ convention. Truncated-BPTT (state/conv stop-grad); f32-only (no ×34).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | ≥loose (dW floor) + speed |
| 2 | Volume 1×1×1×1 | [x] | ≥industry |
| 3 | Volume 1×1×1×2 chain | [x] | ≥industry |
| 4 | Volume 2×2×2×1 | [x] | ≥industry; scaled init |

## w2a Go↔CUDA (Stack NestedStack-of-Dense NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --stack-train`  
Wire: `STKB`/`STKO` (leaf); `STKV`/`STVO` (volume). Exact tensor bits vs Go, subject to the precision policy above.
Timing: same `--bench=N` / spd%Δ convention. NestedStack of Dense Dim→Dim Linear; loom n_children×[Dim×Dim]. Avoids `STB1` (StepBP).

| # | Case | Status | Notes |
|---|---|:---:|---|
| 1 | Leaf Float32 | [x] | exact_bit + speed |
| 2 | Leaf ×34 | [x] | exact_bit spectrum |
| 3–5 | Volume 1×1×1×1 / chain / ×34 | [x] | NestedStack Dense |
| 6–7 | Volume 2×2×2×1 / 2×1×1×1×34 | [x] | binary 211 post denorm ≤1e-18 OK |

---

## w2a Go↔CUDA (KMeans NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --normalbp-all` (`KMeans` and KMeans
`NormalBPEdges` groups in `TestGoVCudaNormalBP`).
Wires: `KMB1`/`KMBO`, `KMV1`/`KMVO`; exported `welvet_kmeans_normalbp_f32` ABI.

| Evidence | Status | Scope |
|---|:---:|---|
| Probabilities and Features Float32 | [x] | Forward, stored pre, dX, dW, native SGD, finite MSE loss |
| Probabilities and Features ×34 | [x] | FormatNone native centers; both output modes |
| All eight activations / temperatures | [x] | Probabilities and Features; temperature 0→1, 0.25, 2.5; rank-3 inputs |
| Volume 111 and two-layer chain Float32 | [x] | Features, complete reverse credit and updated centers |
| Volume 111 / 211 ×34 | [x] | Output, input gradient, loss and updated native weights |
| Volume 222 Float32 | [x] | Multi-cell joins and reverse-hop credit |

**Status:** ✅ signed (CUDA).
reconstruction, activation, assignment/center/input gradients and SGD. Go's
Float32 activation/derivative materialization points are preserved. Dynamic GPU
scratch has no fixed cluster-count cap. The narrow input-gradient cancellation
band above is explicit; larger errors and non-finite values fail. Performance
and complete graph residency remain separate work.

---

## w2a Go↔CUDA (Parallel NormalBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --normalbp-all` (`Parallel`, `ParallelPoly`,
Parallel `NormalBPEdges`, and `ParallelVolumes` groups).
Wires: `PNB1`/`PNBO`, `PPB1`/`PPBO`, `PNV1`/`PNVO`, `NBL1`/`NBLO`.
Exported Dense-branch ABI: `welvet_parallel_normalbp_f32`.

| Evidence | Status | Scope |
|---|:---:|---|
| All seven combines Float32 | [x] | Concat, Add, Avg, Filter, Max, SparseK, Disagree |
| Direct gradients for all seven combines ×34 | [x] | Rank-3 pre/post, dX, dW, native SGD; Filter gate weights included |
| Polymorphic branches | [x] | All 23 branch kinds Float32; 15 dtype-generic branch fixtures ×34 |
| Nested Parallel / Stack | [x] | Recursive GPU forward/backward and packed updates |
| Routing edges | [x] | First-branch Max ties, index-ordered SparseK ties, SparseK=1, zero lr, unequal-width Concat, nested Filter gate |
| Volume 111 / chain / 211 ×34 | [x] | Add and Avg; loss, output, input gradients, native weights |
| Volume 222 Float32 | [x] | Add branches through multi-cell graph joins |
| ABI / host-API guard | [x] | Exported ABI gradients and SGD match C++ CUDA API; legacy CPU matvec still rejects |

**Status:** ✅ signed (CUDA).
kernels and stage buffers. CUDA performs every combine, routing operation,
softmax gate derivative and input-gradient reduction. Input gradients cast only
after the promoted branch sum; Avg backward uses the promoted reciprocal.
Integer-backed 53-bit arithmetic preserves cancellation and Go arm64 fused
multiply/add rounding in Disagree. Native FormatNone weights remain native.
Other TrainModes, full residency and throughput tuning retain separate gates.

---

## w2a Go↔CUDA (KMeans + Parallel, sections 2–6) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --step-tween-all`.
Cases: [`kmeans_parallel_modes.go`](../../w2a/suites/go_v_cuda/kmeans_parallel_modes.go).
Each mode adds the same 50 groups below, with fresh Go fixtures and independent
benchmark replay. The combined suite also reruns the 219 NormalBP groups and
`TestGoVCudaOnly`.

| New evidence per mode | Groups | Status | Scope |
|---|---:|:---:|---|
| KMeans Probabilities / Features | 4 | [ ] | Float32 and ×34; SiLU, temperature 0.7, shape-changing KMeans→Dense chain |
| All seven Parallel combines | 14 | [ ] | Float32 and ×34; three branches, Filter gate export, Concat→Dense width change |
| Polymorphic Parallel branches | 23 | [ ] | Every branch kind Float32; nested Parallel/Stack, recurrent and convolutional branches |
| Stateful Filter collection | 1 | [ ] | Metacognition rules and gate, capture-before-update semantics |
| KMeans / Parallel volume credit | 8 | [ ] | 111 / two-layer chain / 211 ×34, plus 222 Float32 for each kind |

**Status:** ⬜ not signed yet (CUDA).
Line ticks in the multi-hop fixtures. Tween and StepTween project the global
gap onto each trainable Parallel leaf and its gate at half lr; the collect tape
is complete before SGD starts. Filter performs its Go-compatible extra forward,
including stateful rule counters. TweenChain uses full-rate BP. Volume credit
uses the same shared Forward+Step paths as the existing mode fixtures; Parallel
volume Tween treats the cell as a local backward/update unit.

Leaf wires: `STB1`/`STBO`, `TWB1`/`TWBO`, `TCB1`/`TCBO`, `STW1`/`STWO`,
`STC1`/`STCO`. Volume wires: `KMV1`/`KMVO`, `KMTV`/`KMTO`, `KMTC`/`KMCO`,
`PNV1`/`PNVO`, `PNTV`/`PNTO`, `PNTC`/`PNCO`. Step variants reuse their named
parent volume credit as specified in each mode section.

Native weight checks remain exact or within the existing FP32 subnormal policy;
GDN branches retain the declared ≥strict weight floor, exact packed lengths,
and finite loss band. Benchmark outputs must match independent fresh outputs.
KMeans volume input gradients retain the narrow cancellation band declared
above. GPU-only probes also exercise KMeans/Parallel mesh SGD. Full mesh-clock
Go parity is validated separately in section 7. Full residency and throughput
remain separate work.

---

## w2a Go↔CUDA (StepBP Line + MeshBP) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --stepbp-train`  
Wires: `STB1`/`STBO` (Line FamilyBP); `MBP1`/`MBPO` (Dense mesh-clock FamilyBP).
`TestGoVCudaStepBPTrain` contains **134 case groups**, including the dtype loops below.
The runner also executes `TestGoVCudaOnly` and defaults to one measured iteration.
Every forward, backward, loss, gradient join and SGD calculation uses CUDA kernels;
host code schedules ticks, stores tapes and stages buffers.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Line Float32 | [x] | Per-tick tapes and FamilyBP on pop; exact weights, finite loss band |
| Dense Line ×34 | [x] | All supported FormatNone dtypes |
| RMSNorm / LayerNorm Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| Softmax Line Float32 + ×34 | [x] | Weightless loss check uses the finite loss band |
| SwiGLU / Sequential / Residual Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| Mamba / Metacognition Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| GDN Line Float32 | [x] | ≥strict weights; finite loss band; GPU float2 reductions/transcendentals |
| NestedStack Line Float32 + ×34 | [x] | Recursive flattening and child updates |
| Embedding / MHA Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| CNN1 / ConvT1 Line Float32 + ×34 | [x] | Explicit Line dispatch |
| CNN2/3 / ConvT2/3 Line ×34 | [x] | Explicit rank-4/rank-5 inputs, including Float32 |
| RNN / LSTM Line Float32 + ×34 | [x] | Exact weights; finite loss band |
| Dense mesh-clock 111 / 211 / 222 Float32 | [x] | Double-buffer forward ticks, reverse-time credit and SGD |
| Dense mesh-clock 111 / 222 ×34 | [x] | Repeated ticks and native storage updates |
| Dense mesh-clock disabled cell + remote link | [x] | Disabled passthrough and remote gradient routing |
| Volume 222 Float32: all 23 supported families | [x] | Shared volume Forward+Step path used by Go ModeMeshBP |
| Volume 211 ×34: all 22 dtype-generic families | [x] | GDN is Float32-only; retains its volume ≥industry floor |
| Benchmark fixture reset | [x] | Line dtype sweeps and GDN compare benchmark results with fresh-run results; Dense mesh dtype/routing fixtures do likewise |

**Scope:** volume rows deliberately reuse the NormalBP volume fixtures because
Go's grid ModeMeshBP uses that same Forward+Step path. They do not establish
mesh-clock BPTT parity for non-Dense cells. The separate `MGB1` all-family
clock fixtures and section 7 provide that evidence.
KMeans and Parallel are signed off for Line/shared-volume credit above. No CPU fallback is supplied.

**Status:** ✅ signed (CUDA) — `TestGoVCudaStepBPTrain` PASS (134 groups, ~838s @ BENCH_ITERS=1).
repetitions rebuild the initial child configuration and weights; timing includes
that reset, matching the Go fixture setup. These small correctness fixtures do
not establish a performance win.

## w2a Go↔CUDA (Tween ModeTween) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tween-train`  
Wires: `TWB1`/`TWBO` (Stack ModeTween leaf), `TWV1`/`TWVO` (Dense volume),
and per-kind volume wires `TRV1`…`MBTV`, `MCTV`, `GDTV`, `SKTV`.
`TestGoVCudaTweenTrain` contains **130 case groups**, including **29 leaf ×34**
and **26 volume ×34** dtype sweeps. GDN is Float32-only.

All numerical projection, forward, backward, loss and SGD work runs on CUDA.
The global gap is block-averaged onto each local output shape and applied at
`soft_lr = lr × 0.5`; Sequential/Residual expand their Dense leaves without
chain-rule credit. Host code schedules the walk and stages buffers/tapes.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Float32 + 1-hop ×34 | [x] | Width-changing projected credit; native weight updates |
| RMSNorm / LayerNorm Float32 + ×34 | [x] | Exact weights; finite loss band |
| Softmax Float32 + ×34 | [x] | Weightless loss checks; GPU exp uses promoted evaluation before Float32 rounding |
| SwiGLU Float32 + ×34 | [x] | Exact weights; finite loss band |
| Sequential / Residual Float32 + ×34 | [x] | Leaf-projected credit, original forward inputs retained |
| Mamba / Metacognition Float32 + ×34 | [x] | Exact weights; finite loss band |
| GDN Float32 | [x] | ≥strict updated weights; finite loss band (one Float32-step weight difference remains) |
| NestedStack Float32 + ×34 | [x] | Flattened leaf walk; exact weights and finite loss band |
| Embedding / MHA Float32 + ×34 | [x] | Exact weights; finite loss band |
| CNN1/2/3 / ConvT1/2/3 ×34 | [x] | Explicit per-dimension leaf fixtures, including Float32 and rank-5 inputs |
| RNN / LSTM Float32 + ×34 | [x] | Exact weights; finite loss band |
| Dense volume 111 / 211 / 222 Float32 | [x] | Reverse-hop projected local updates |
| Volume 222 Float32: all 23 supported families | [x] | Per-kind local update dispatch; finite loss band |
| Volume 211 ×34: all 22 dtype-generic families | [x] | Native storage updates across all supported dtypes |
| GPU projection and nested-credit primitives | [x] | Downsample, upsample, cancellation, invalid shapes, and original-input local updates |
| Benchmark fixture reset | [x] | All leaf dtype sweeps and Dense volume dtype cases compare measured-run output against an independent run |

**Status:** ✅ ModeTween signed (CUDA).
Loss uses the finite float2 band above. GDN leaf weights require ≥strict, rather
than exact bits. Volume weight subnormal flushing follows the existing CUDA
policy. KMeans and Parallel have independent ModeTween evidence above. No host
numerical fallback is supplied.

**Validated:** 2026-10-28. `TestGoVCudaTweenTrain` — **130 case groups** PASS in
717.464 seconds (`WELVET_CUDA_BENCH_ITERS=1`, `-timeout 90m`). Projection uses
CUDA `launch_project_gap` (f64 block averages). This sign-off is ModeTween only.
TweenChain / Split / Alt and Step/Mesh variants have separate sign-offs below.
Graph residency and performance tuning remain open; benchmark timings include
resetting the fixture.


## w2a Go↔CUDA (TweenChain ModeTweenChain) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweenchain-train`  
Wires: `TCB1`/`TCBO` (Stack FamilyBP at full lr), `TCV1`/`TCVO` (Dense volume),
and per-kind volume wires `TRC1`…`MBTC`, `MCTC`, `GDTC`, `SKTC`.
`TestGoVCudaTweenChainTrain` contains **130 case groups**, including **29 leaf ×34**
and **26 volume ×34** dtype sweeps. GDN is Float32-only.

Forward, backward, MSE/MSEGrad and native SGD use CUDA kernels. TweenChain
propagates the true chain-rule input gradient at **full lr**, including through
Sequential, Residual and NestedStack. Volume credit is the same Forward+Step
operation as NormalBP; the named TweenChain entry point and every wire are
exercised here. Host code handles scheduling, allocation and tape/buffer staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Case group | Status | Evidence |
|---|:---:|---|
| Dense multi-hop Float32 + 1-hop ×34 | [x] | Full-lr chain-rule credit; exact updated weights and finite loss band |
| RMSNorm / LayerNorm Float32 + ×34 | [x] | Exact weights; finite loss band |
| Softmax Float32 + ×34 | [x] | Weightless loss check uses the finite CUDA loss band |
| SwiGLU Float32 + ×34 | [x] | Exact weights; finite loss band |
| Sequential / Residual Float32 + ×34 | [x] | Full chain rule through Dense children |
| Mamba / Metacognition Float32 + ×34 | [x] | Exact weights; finite loss band |
| GDN Float32 | [x] | ≥strict updated weights; finite loss band; Float32 rounding difference retained |
| NestedStack Float32 + ×34 | [x] | Recursive flattening with reverse chain-rule credit |
| Embedding / MHA Float32 + ×34 | [x] | Exact weights; finite loss band |
| CNN1/2/3 / ConvT1/2/3 ×34 | [x] | Explicit per-dimension fixtures, including Float32 and rank-5 inputs |
| RNN / LSTM Float32 + ×34 | [x] | Exact weights; finite loss band |
| Dense volume 111 / 211 / 222 Float32 | [x] | NormalBP-equivalent volume credit at full lr |
| Volume 222 Float32: all 23 supported families | [x] | Per-kind TweenChain protocol; Softmax volume uses NormalBP Softmax floor |
| Volume 211 ×34: all 22 dtype-generic families | [x] | Native weight updates and chain-rule input gradients |
| GPU chain-rule/full-lr primitives | [x] | Named TCB1/TCV1 entry points; FamilyBP @ full lr via systems/tween_chain |
| Benchmark fixture reset | [x] | Leaf dtype sweeps and Dense volume dtype fixtures compare measured-run output with an independent run |

**Status:** ✅ ModeTweenChain signed (CUDA).
loss band apply where specified. Softmax volume gin retains the NormalBP Softmax
DetLoose+/~strict floor. GDN leaf weights retain ≥strict; its volume retains
≥industry. The existing subnormal and KMeans cancellation policies remain
explicit above. KMeans/Parallel have independent evidence for this mode.
Remaining variants retain their own gates. Residency and performance tuning
remain open; benchmark timings include fixture reset.

## w2a Go↔CUDA (TweenSplit ModeTweenSplit) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweensplit-train`.
`TestGoVCudaTweenSplitTrain` contains **131 case groups**: the existing 80 plus
51 KMeans/Parallel groups. Wires: `TSB1`/`TSBO`; existing per-kind Split volumes
plus `KMSP`/`KMSO` and `PNSP`/`PNSO`.

Single collect tape; global 1/N share at full lr. All numerical forward, loss,
projection, 1/N scaling, local backward and native SGD work uses CUDA kernels.
Host code collects tapes, schedules leaves and stages buffers. No chain-rule
input gradient is claimed for this mode.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 supported families | [x] | Native dtype sweeps for the 22 generic families; GDN Float32 only |
| Both KMeans outputs | [x] | Probabilities / Features, Float32 and ×34, including shape-changing KMeans→Dense |
| Seven Parallel combines | [x] | Float32 and ×34; Filter gate counts as a trainable leaf; Concat→Dense width changes |
| Polymorphic Parallel | [x] | All 23 branch kinds Float32, including nested Stack/Parallel and rank-4/rank-5 convolutional branches |
| Stateful Filter | [x] | Metacognition capture and gate update; complete tape before SGD |
| Mixed weightless/trainable leaves | [x] | Dense + Softmax Parallel followed by Dense ×34; Softmax excluded from N |
| Shared volume credit | [x] | All 23 families; KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Benchmark reset | [x] | Fresh loss/weights match measured runs; all new leaf and volume fixtures require exact CUDA replay across at least two iterations |
| Collect primitives | [x] | Existing hand-calculated 1/N sharing and tape preservation guards |

Parallel leaf-mode Split recursively collects trainable branches, nested
containers, Sequential/Residual Dense leaves and the Filter gate. It reuses
Tween's collector, including Filter's extra forward and state transitions;
updates begin only after collection. Split loss is scored from the collected
output, whose native branch axes can differ from regular Parallel Forward.
The Go fixture discovers target shapes through Go's public `OpenSplitTape`.
Volume Split retains its existing grid contract: each placed KMeans/Parallel
cell is one local backward/SGD unit; Sequential/Residual expand Dense children.

Existing precision gates apply: finite loss band, exact native weights or the
documented FP32 subnormal policy. GDN leaf/Parallel branch weights retain
≥strict and GDN shared volumes retain ≥industry. No precision gates were
weakened. Other Split/Alt modes retain their separate sign-off sections. Full
residency and throughput tuning remain separate work.

**Status:** ✅ ModeTweenSplit signed (CUDA). All 131 TweenSplit groups passed in
775.850 seconds under the existing precision gates.

## w2a Go↔CUDA (StepTweenSplit ModeStepTweenSplit) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweensplit-train`.

Wires: `STS1`/`STSO`; shared volume `TSP1` and per-kind Split variants. `TestGoVCudaStepTweenSplitTrain`: **130 case groups**.

Line per-hop Split at full lr; N counts trainable leaves under that hop. The popped global gap stays frozen. Numerical work uses CUDA kernels; host code handles scheduling, tapes and staging. Parallel collect matches Go `combineCollected` (Filter keeps the extra forward). KMeans/Parallel Line multi-tick fixtures allow ≥strict weights where CUDA MatMul ulps accumulate across five pops (single-pop remains exact_bit).

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [x] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [x] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [x] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [x] | All 23 supported families (TweenSplit volume wires) |
| Volume 211 ×34 | [x] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [x] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [x] | FamilyTweenSplit on pop via `apply_split_child`; frozen gap passthrough |

**Status:** ✅ ModeStepTweenSplit signed (CUDA). All 130 groups passed in 760.015s.

## w2a Go↔CUDA (TweenAlt ModeTweenAlt) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --tweenalt-train`  
Batch: `./scripts/linux_build_w2a.sh --tween-next-all` (all eight requested sections + CUDA-only checks).

Wires: `TAB1`/`TABO`; volume `TAV1` and per-kind Alt variants. `TestGoVCudaTweenAltTrain`: **130 case groups**.

Split at full lr, refresh prediction and gap, then Tween at lr×0.5. AltTimes=1 is validated. Numerical work uses CUDA kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Collect/phase primitives | [ ] | Hand-calculated 1/N sharing and refreshed Alt loss/weights; input tape retained |

**Status:** ⬜ not signed yet (CUDA).

## w2a Go↔CUDA (StepTweenAlt ModeStepTweenAlt) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweenalt-train`  
Batch: `./scripts/linux_build_w2a.sh --tween-next-all` (all eight requested sections + CUDA-only checks).

Wires: `STA1`/`STAO`; shared volume `TAV1` and per-kind Alt variants. `TestGoVCudaStepTweenAltTrain`: **130 case groups**.

Line per-hop Split→Tween at full/half lr with a frozen popped gap. Its shared volume path refreshes the gap between phases. Numerical work uses CUDA kernels; host code handles scheduling, tapes and staging.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 ×34 dtype sweeps, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Status:** ⬜ not signed yet (CUDA).

## w2a Go↔CUDA (MeshTween mesh-clock) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --meshtween-train`.
The runner includes MeshTween, MeshBP, ModeTween and CUDA-only regression gates.
`TestGoVCudaMeshTweenTrain` contains **159 case groups**: 48 existing Dense/shared
volume groups, eight KMeans/Parallel volume groups, and 103 all-family clock groups.
Wires: `MGW1`/`MGWO` for actual heterogeneous clock nodes, existing `MTW1`/`MTWO`
for Dense, and the existing per-kind ModeTween volume wires.

Last-tick captured mesh credit applies local projected updates at lr×0.5.
`step::train_mse(FamilyTween)` and the generic clock wire share
`step::apply_tween`; warmup/remote fixtures supply the captured state directly.
All numerical forward, gap projection, backward and native SGD work uses CUDA.
Host code schedules ticks, stores tapes, rebuilds fixtures and stages buffers.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 actual node kinds | [ ] | 111 / 211 ×34, plus 222 Float32; GDN Float32 only |
| Recurrent remote clock | [ ] | 22 shape-preserving families; detached warmup then captured ticks |
| KMeans output modes | [ ] | Probabilities and Features ×34 in three-node chains |
| Parallel combines | [ ] | All seven ×34, including Filter gate updates and Concat width changes |
| Shared volume credit | [ ] | All 23 families; supported dtype loops, including KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Zero learning rate | [ ] | Dense/KMeans/Parallel preserve native weights ×34 |
| Captured tensors and updates | [ ] | Every tick/node input, pre and post; final loss/output and all updated weights |
| Exact replay | [ ] | Independent fresh CUDA result matches at least two measured iterations exactly |
| Credit primitive | [ ] | Existing hand-calculated two-cell update distinguishes local Tween from BPTT/full-lr credit |

**Scope:** this is last-tick local mesh credit; no temporal backward pass or
chain-rule input/accumulated parameter gradient is exported. Sequential and
Residual project onto their Dense leaves. Stack and Parallel nodes use their
concrete backward/SGD as local units, matching the existing grid ModeTween
volume contract. Gap copying continues only when the next input length matches;
otherwise the upstream gap is zero. Go references combine Go step capture,
concrete Go backward/SGD and its public local-Tween collector in w2a.
Clock precision policy: finite loss band, exact/native-subnormal weight
checks, GDN tensors ≥loose and weights ≥strict. Shared-volume checks retain
their existing gates, including GDN ≥industry. Full graph residency and
performance tuning remain separate work. MeshTweenChain retains its own gate.

**Status:** ⬜ not signed yet (CUDA).
combined runner, including 103 MeshBP groups, 130 ModeTween groups and CUDA-only
guards, passed in 299.198 seconds. Shared/static builds passed.

## w2a Go↔CUDA (MeshTweenChain mesh-clock) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --meshtweenchain-train`.
The runner covers MeshTweenChain, MeshTween, ModeTweenChain and CUDA-only guards.
`TestGoVCudaMeshTweenChainTrain` contains **160 case groups**: 48 existing
Dense/shared-volume groups, eight KMeans/Parallel volume groups, 103 all-family
clock groups, and one direct 23-family MeshBP equivalence group.

Wires: `MGC1`/`MGCO` binds actual heterogeneous clock nodes; existing Dense
`MTC1`/`MTCO` and per-kind TweenChain volume wires retain their coverage.
MeshTweenChain uses the existing `runtime/step` FamilyBP path: captured forward
ticks → reverse-time BPTT → accumulated parameter gradients → native SGD at full
lr. Every numerical operation uses CUDA kernels; host code schedules ticks,
stores tapes, stages buffers and rebuilds replay fixtures. The wire alias adds
no separate numerical implementation or CPU fallback.

| Evidence | Status | Scope |
|---|:---:|---|
| All 23 actual node kinds | [ ] | 111 / 211 ×34 for 22 dtype-generic families; GDN Float32 only; all 23 at 222 Float32 |
| Recurrent remote clock | [ ] | 22 shape-preserving families; detached warmup then captured ticks |
| KMeans output modes | [ ] | Probabilities / Features ×34 with real Dense adapter nodes |
| Parallel combines | [ ] | All seven ×34, including Filter gate gradients and Concat shape changes |
| Captured tensors and gradients | [ ] | Every tick/node input, pre and post; final output/loss, oldest input gradient and packed accumulated parameter gradients |
| Full-rate native SGD | [ ] | Updated weights for every enabled node; Dense/KMeans/Parallel zero-lr preservation ×34 |
| Exact replay | [ ] | Fresh CUDA result equals at least two measured repetitions across all exported tensors, weights, tapes and loss |
| MeshBP equivalence | [ ] | Independent MeshBP and MeshTweenChain wire calls agree exactly for all 23 families at 211 Float32 |
| Shared volume credit | [ ] | All 23 families, including KMeans/Parallel 111/chain/211 ×34 and 222 Float32 |
| Credit primitive | [ ] | Existing hand-calculated test distinguishes full-rate BPTT from last-tick half-rate MeshTween |

The existing precision policy applies: finite loss band; exact/subnormal tensor
and weight gates where specified; MHA gradients ≥strict, Softmax gradients ≥loose;
GDN clock tensors ≥loose and weights ≥strict. GDN shared-volume checks retain
≥industry. Only KMeans input gradients use the narrow cancellation allowance.
Parameter gradient axes are flattened for packed comparison; activation and
input-gradient shapes remain strict. Embedding uses one 111 tick and disabled
ingress plus RMSNorm adapter nodes in grids. SwiGLU uses the bounded recurrent
trajectory from section 7. These are correctness fixtures; full residency and
throughput tuning retain separate gates.

**Status:** ⬜ not signed yet (CUDA).
The combined runner also passed 159 MeshTween groups, 130 ModeTweenChain groups
and CUDA-only guards in 321.915 seconds. Shared/static builds passed.

## w2a Go↔CUDA (StepTween ModeStepTween) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptween-train`  
Batch: `./scripts/linux_build_w2a.sh --step-tween-all` (sections 2–6 + NormalBP + CUDA-only checks).

Wires: `STW1`/`STWO`; shared volume `TWV1` and per-kind Tween variants. `TestGoVCudaStepTweenTrain`: **130 case groups**.

Line per-hop projected local credit at lr×0.5, applied when a training flight pops. Numerical work uses CUDA kernels; host code handles scheduling, tapes and staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 dtype-generic families ×34, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Status:** ⬜ not signed yet (CUDA).
CUDA loss band apply where specified; GDN branches retain ≥strict. The
existing subnormal policy and KMeans volume cancellation band remain explicit.
KMeans/Parallel evidence is included above. Full graph residency and
performance tuning remain open.

## w2a Go↔CUDA (StepTweenChain ModeStepTweenChain) — sign-off evidence

Run: `./scripts/linux_build_w2a.sh --steptweenchain-train`  
Batch: `./scripts/linux_build_w2a.sh --step-tween-all` (sections 2–6 + NormalBP + CUDA-only checks).

Wires: `STC1`/`STCO`; shared volume `TCV1` and per-kind TweenChain variants. `TestGoVCudaStepTweenChainTrain`: **130 case groups**.

Line per-hop true chain-rule BP at full lr, applied when a training flight pops. Numerical work uses CUDA kernels; host code handles scheduling, tapes and staging.

KMeans / Parallel: the 50 additional groups per mode are detailed in the
sections 2–6 extension evidence above.

| Evidence | Status | Scope |
|---|:---:|---|
| Dense multi-hop Float32 | [ ] | Loss and updated weights; frozen Line gap where applicable |
| All supported leaf families | [ ] | 23 families; 22 dtype-generic families ×34, GDN Float32 only |
| CNN2/3 and ConvT2/3 | [ ] | Explicit rank-4/rank-5 ×34 fixtures |
| Volume 222 Float32 | [ ] | All 23 supported families |
| Volume 211 ×34 | [ ] | All 22 dtype-generic families; shared parent volume credit for Line variants |
| Benchmark reset | [ ] | Every leaf dtype sweep compares measured-run output with an independent fixture |
| Line state/credit primitives | [ ] | Warmup, per-hop credit, frozen gap, action flights without training, unsupported-mode rejection |

**Status:** ⬜ not signed yet (CUDA).
CUDA loss band apply where specified; GDN branches retain ≥strict. The
existing subnormal policy and KMeans volume cancellation band remain explicit.
KMeans/Parallel evidence is included above. Full graph residency and
performance tuning remain open.

## 1 — NormalBP

Chain-rule Backward + ApplyGradSGD. Start here. Dense first.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.51 |
| RMSNorm | [x] | [x] | [x] | 0.51 |
| LayerNorm | [x] | [x] | [x] | 0.51 |
| SwiGLU | [x] | [x] | [x] | 0.51 |
| Softmax | [x] | [x] | [x] | 0.51 |
| Embedding | [x] | [x] | [x] | 0.51 |
| MHA | [x] | [x] | [x] | 0.51 |
| CNN1 | [x] | [x] | [x] | 0.51 |
| CNN2 | [x] | [x] | [x] | 0.51 |
| CNN3 | [x] | [x] | [x] | 0.51 |
| ConvT1 | [x] | [x] | [x] | 0.51 |
| ConvT2 | [x] | [x] | [x] | 0.51 |
| ConvT3 | [x] | [x] | [x] | 0.51 |
| RNN | [x] | [x] | [x] | 0.51 |
| LSTM | [x] | [x] | [x] | 0.51 |
| Sequential | [x] | [x] | [x] | 0.51 |
| Residual | [x] | [x] | [x] | 0.51 |
| KMeans | [x] | [x] | [x] | 0.51 |
| Mamba | [x] | [x] | [x] | 0.51 |
| Metacognition | [x] | [x] | [x] | 0.51 |
| GDN | [x] | [x] | [x] | 0.51 |
| Parallel | [x] | [x] | [x] | 0.51 |
| Stack | [x] | [x] | [x] | 0.51 |

**Checkpoint:** ✅ NormalBP signed (CUDA) — `TestGoVCudaNormalBP` PASS; leaf+vol ×34 (GDN f32). Next: StepBP / MeshBP.

---

## 2 — StepBP

Needs Step `Line` pipe + FamilyBP (NormalBP credit) on pop.  
Ship: thin Stack host + `runtime/line` TickLine/TrainLineMSE; wire `STB1`/`STBO`; w2a `--stepbp-train`.  
**3D volumetric** for StepBP = MeshBP (`runtime/step` mesh-clock FamilyBP + ModeMeshBP volume Step). All CNN/ConvT dimensions have explicit Line ×34 fixtures, including rank-5 inputs.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.52 |
| RMSNorm | [x] | [x] | [x] | 0.52 |
| LayerNorm | [x] | [x] | [x] | 0.52 |
| Softmax | [x] | [x] | [x] | 0.52 |
| SwiGLU | [x] | [x] | [x] | 0.52 |
| Embedding | [x] | [x] | [x] | 0.52 |
| MHA | [x] | [x] | [x] | 0.52 |
| CNN1 | [x] | [x] | [x] | 0.52 |
| CNN2 | [x] | [x] | [x] | 0.52 |
| CNN3 | [x] | [x] | [x] | 0.52 |
| ConvT1 | [x] | [x] | [x] | 0.52 |
| ConvT2 | [x] | [x] | [x] | 0.52 |
| ConvT3 | [x] | [x] | [x] | 0.52 |
| RNN | [x] | [x] | [x] | 0.52 |
| LSTM | [x] | [x] | [x] | 0.52 |
| Sequential | [x] | [x] | [x] | 0.52 |
| Residual | [x] | [x] | [x] | 0.52 |
| KMeans | [x] | [x] | [x] | 0.52 |
| Mamba | [x] | [x] | [x] | 0.52 |
| Metacognition | [x] | [x] | [x] | 0.52 |
| GDN | [x] | [x] | [x] | 0.52 |
| Parallel | [x] | [x] | [x] | 0.52 |
| Stack | [x] | [x] | [x] | 0.52 |

**Checkpoint:** ✅ StepBP Line + Dense MeshBP signed (CUDA) — STB1 FamilyBP + MBP1 mesh-clock + ModeMeshBP vols; KMeans/Parallel Line included.

---

## 3 — Tween

ModeTween: projected local credit @ `soft_lr = lr × 0.5` (Go `applyProjectedTween` — same gap on every leaf, no chain rule).  
Ship: `systems/tween` + `runtime/tween_train`; wires `TWB1`/`TWBO` + `TWV1`/`TWVO` + per-kind `TRV1`…`MBTV`; w2a `--tween-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. 3D = reverse-hop volume 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.54 |
| RMSNorm | [x] | [x] | [x] | 0.54 |
| LayerNorm | [x] | [x] | [x] | 0.54 |
| SwiGLU | [x] | [x] | [x] | 0.54 |
| Softmax | [x] | [x] | [x] | 0.54 |
| Embedding | [x] | [x] | [x] | 0.54 |
| MHA | [x] | [x] | [x] | 0.54 |
| CNN1 | [x] | [x] | [x] | 0.54 |
| CNN2 | [x] | [x] | [x] | 0.54 |
| CNN3 | [x] | [x] | [x] | 0.54 |
| ConvT1 | [x] | [x] | [x] | 0.54 |
| ConvT2 | [x] | [x] | [x] | 0.54 |
| ConvT3 | [x] | [x] | [x] | 0.54 |
| RNN | [x] | [x] | [x] | 0.54 |
| LSTM | [x] | [x] | [x] | 0.54 |
| Sequential | [x] | [x] | [x] | 0.54 |
| Residual | [x] | [x] | [x] | 0.54 |
| KMeans | [x] | [x] | [x] | 0.54 |
| Mamba | [x] | [x] | [x] | 0.54 |
| Metacognition | [x] | [x] | [x] | 0.54 |
| GDN | [x] | [x] | [x] | 0.54 |
| Parallel | [x] | [x] | [x] | 0.54 |
| Stack | [x] | [x] | [x] | 0.54 |

**Checkpoint:** ✅ ModeTween signed (CUDA) — projected local @ `lr×0.5`; TWB1/TWV1 + per-kind volumes + KMeans/Parallel; 130 groups.

---

## 4 — TweenChain

ModeTweenChain: named Tween, credit = BP (Go `familyTweenChain` ≡ `familyBP` @ full lr).  
Ship: `systems/tween_chain` + `runtime/tween_chain_train`; wires `TCB1`/`TCBO` + `TCV1`/`TCVO` + per-kind `TRC1`…`MBTC`; w2a `--tweenchain-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. 3D = volume Step 2×2×2×1 for all NormalBP-signed kinds.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.55 |
| RMSNorm | [x] | [x] | [x] | 0.55 |
| LayerNorm | [x] | [x] | [x] | 0.55 |
| SwiGLU | [x] | [x] | [x] | 0.55 |
| Softmax | [x] | [x] | [x] | 0.55 |
| Embedding | [x] | [x] | [x] | 0.55 |
| MHA | [x] | [x] | [x] | 0.55 |
| CNN1 | [x] | [x] | [x] | 0.55 |
| CNN2 | [x] | [x] | [x] | 0.55 |
| CNN3 | [x] | [x] | [x] | 0.55 |
| ConvT1 | [x] | [x] | [x] | 0.55 |
| ConvT2 | [x] | [x] | [x] | 0.55 |
| ConvT3 | [x] | [x] | [x] | 0.55 |
| RNN | [x] | [x] | [x] | 0.55 |
| LSTM | [x] | [x] | [x] | 0.55 |
| Sequential | [x] | [x] | [x] | 0.55 |
| Residual | [x] | [x] | [x] | 0.55 |
| KMeans | [x] | [x] | [x] | 0.55 |
| Mamba | [x] | [x] | [x] | 0.55 |
| Metacognition | [x] | [x] | [x] | 0.55 |
| GDN | [x] | [x] | [x] | 0.55 |
| Parallel | [x] | [x] | [x] | 0.55 |
| Stack | [x] | [x] | [x] | 0.55 |

**Checkpoint:** ✅ ModeTweenChain signed (CUDA) — FamilyBP @ full lr; TCB1/TCV1 + per-kind volumes + KMeans/Parallel; 130 groups.

---

## 5 — StepTween

Needs Line + Tween.  
Ship: `runtime/line` Credit::FamilyTween on pop (`apply_projected` @ `lr×0.5`); wire `STW1`/`STWO`; w2a `--steptween-train`.  
**3D volumetric** for StepTween = MeshTween volume (= ModeTween reverse-hop projected local; existing `TWV1`/`TRV1`… wires). CNN2/3 and ConvT2/3 share CNN1/ConvT1 path.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 6 — StepTweenChain

Needs Line + TweenChain.  
Ship: `runtime/line` Credit::FamilyBP on pop (same credit as StepBP / TweenChain); wire `STC1`/`STCO`; w2a `--steptweenchain-train`.  
**3D volumetric** = MeshTweenChain volume (= ModeTweenChain volume Step wires `TCV1`/`TRC1`…).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 7 — MeshBP

Grid tick FamilyBP: `runtime/step` Forward×ticks → reverse-time Backward BPTT →
ApplyGradSGD. `MGB1`/`MGBO` binds actual heterogeneous node kinds; existing
`MBP1`/`MBPO` Dense fixtures remain covered. Numerical forward, loss, backward,
gradient joins and native SGD use CUDA kernels. Host code schedules ticks,
stores tapes, stages buffers and rebuilds benchmark fixtures.

Run: `./scripts/linux_build_w2a.sh --meshbp-train`.
The runner includes `TestGoVCudaMeshBPTrain` (**103 case groups**), the existing StepBP/volume suite,
and NormalBP. Go ModeMeshBP on grid also uses the shared
volume Forward+Step path validated at 222 Float32 and 211 ×34.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.53 |
| RMSNorm | [x] | [x] | [x] | 0.53 |
| LayerNorm | [x] | [x] | [x] | 0.53 |
| SwiGLU | [x] | [x] | [x] | 0.53 |
| Softmax | [x] | [x] | [x] | 0.53 |
| Embedding | [x] | [x] | [x] | 0.53 |
| MHA | [x] | [x] | [x] | 0.53 |
| CNN1 | [x] | [x] | [x] | 0.53 |
| CNN2 | [x] | [x] | [x] | 0.53 |
| CNN3 | [x] | [x] | [x] | 0.53 |
| ConvT1 | [x] | [x] | [x] | 0.53 |
| ConvT2 | [x] | [x] | [x] | 0.53 |
| ConvT3 | [x] | [x] | [x] | 0.53 |
| RNN | [x] | [x] | [x] | 0.53 |
| LSTM | [x] | [x] | [x] | 0.53 |
| Sequential | [x] | [x] | [x] | 0.53 |
| Residual | [x] | [x] | [x] | 0.53 |
| KMeans | [x] | [x] | [x] | 0.53 |
| Mamba | [x] | [x] | [x] | 0.53 |
| Metacognition | [x] | [x] | [x] | 0.53 |
| GDN | [x] | [x] | [x] | 0.53 |
| Parallel | [x] | [x] | [x] | 0.53 |
| Stack | [x] | [x] | [x] | 0.53 |

**Validated:** 2026-10-28. The 103 MeshBP case groups passed in
1356.030 seconds (`WELVET_CUDA_BENCH_ITERS=1`) with at least two measured
replay iterations per fixture. Shared build passed.

**Checkpoint:** ✅ MeshBP signed (CUDA) — all 23 supported families have actual
mesh-clock node and grid BPTT evidence. All 22 dtype-generic families cover
111 and 211 ×34; GDN is Float32-only. Every family also covers 222 Float32.
Shape-changing Embedding uses a single-tick 111 node and disabled ingress plus
RMSNorm adapter nodes in grids; its integer indices are not fed back as
embedding outputs. Remote self-links cover the 22 shape-preserving families
with detached warmup followed by captured ticks. SwiGLU uses a shorter bounded
recurrent trajectory to avoid vanishing intermediate subnormals outside the
declared GPU precision contract.

Both KMeans outputs and all seven Parallel combines additionally cover ×34
three-node clock chains, including shape-changing Concat/probabilities→Dense
and Filter gate gradients/updates. Dense/KMeans/Parallel zero-lr cases preserve
native weights across ×34; `step::apply_sgd(0)` is a no-op.

Checks include every captured node input/pre/post, final output/loss, oldest
input gradient, packed accumulated parameter gradients, and post-SGD native
weights. Parameter gradients use a flat packed axis for comparison; activation
and input-gradient shapes remain strict. Existing precision policy applies:
GDN tensors ≥loose and updated weights ≥strict, MHA gradients ≥strict, Softmax
gradients ≥loose, other gates exact or the documented FP32 subnormal policy.
Only KMeans input gradients use its narrow near-zero cancellation allowance.
Fresh and benchmark CUDA runs must match exactly across all exported tensors,
loss, weights and tapes. MeshBP measures at least two iterations to exercise
reset between repetitions. Replay timing includes rebuilding/staging; residency
and performance tuning retain separate gates. MeshTween/other modes retain
their own clock sign-off below.

## 8 — MeshTween

Needs Mesh + Tween.  
Ship: `runtime/step` `Credit::FamilyTween` — Forward×ticks → last-tick `apply_tween` / `apply_cell_projected` @ `lr×0.5` (`MGW1`/`MGWO`, existing Dense `MTW1`/`MTWO`); w2a `--meshtween-train`.

**3D volumetric** = ModeTween volume credit (`TWV1`/`TRV1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Status:** ⬜ not signed yet (CUDA).
and CUDA-only runner passed in 299.198 seconds. Shared and static builds passed.

**Checkpoint:** ⬜ not signed yet (CUDA).
sign-off. The 159 MeshTween groups passed in 122.91 seconds. Clock fixtures
cover 111/211 ×34 for the 22 dtype-generic families and 222 Float32 for all 23;
GDN remains Float32-only. Every captured node input/pre/post, final output/loss
and updated weight is checked. Clock GDN tensors require ≥loose and weights
≥strict; other weights remain exact or use the existing subnormal policy.
Shared volume gates retain their existing floors.

The 103 generic clock groups include recurrent remote links with detached
warmup for 22 shape-preserving families, both KMeans outputs and seven Parallel
combines ×34, and Dense/KMeans/Parallel zero-lr native preservation ×34.
Embedding uses one 111 tick, and disabled ingress plus RMSNorm adapters in grids;
integer indices are not fed back from embedding outputs. SwiGLU uses the bounded
recurrent trajectory documented in section 7. Fresh and at least two measured
CUDA replays match exactly across outputs, weights, loss and all captured tapes.

This mode applies local credit only on the last tick. Sequential/Residual
project onto Dense leaves; Stack/Parallel cells are concrete local backward/SGD
units, as in the shared grid ModeTween contract. No temporal BPTT, chain-rule
input gradient or accumulated parameter gradient is claimed. Full residency
and throughput tuning retain separate gates. MeshTweenChain remains section 9.

---

## 9 — MeshTweenChain

Needs Mesh + TweenChain.  
Ship: `runtime/step` `Credit::FamilyBP` under mesh clock (`MGC1`/`MGCO`, existing Dense `MTC1`/`MTCO` ≡ MeshBP); w2a `--meshtweenchain-train`.

**3D volumetric** = ModeTweenChain volume credit (`TCV1`/`TRC1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Status:** ⬜ not signed yet (CUDA).
ModeTweenChain/CUDA-only runner passed in 321.915 seconds. Shared and static
builds passed.

**Checkpoint:** ⬜ not signed yet (CUDA).
including KMeans and Parallel. All 160 MeshTweenChain groups passed in 134.60
seconds. The 22 dtype-generic families cover 111/211 ×34; GDN is Float32-only.
All 23 cover 222 Float32. Existing shared-volume gates retain their floors.

Clock evidence includes every tick/node input, pre and post; final loss/output;
oldest input gradients; packed parameter gradients accumulated across ticks;
and full-lr native SGD weights. Recurrent remote links cover 22 shape-preserving
families after detached warmup. Both KMeans output modes and all seven Parallel
combines additionally cover ×34 three-node chains. Dense/KMeans/Parallel zero-lr
cases preserve native weights ×34. Fresh and at least two measured CUDA replays
match exactly. A direct 211 Float32 comparison verifies exact MeshBP/MeshTweenChain
wire equivalence for every supported family.

Embedding uses one 111 tick and disabled ingress plus RMSNorm adapters in grids;
its integer indices are not fed back from embedding outputs. SwiGLU uses the
bounded recurrent trajectory from section 7. Existing clock precision gates
apply: finite loss band, exact/subnormal gates, MHA gradients ≥strict, Softmax
gradients ≥loose, GDN tensors ≥loose and weights ≥strict; only KMeans input
gradients use its narrow cancellation allowance. Gradient packing normalizes
only parameter axes; activation/input-gradient shapes remain strict. The named
wire reuses CUDA FamilyBP BPTT/SGD; it adds no numerical fallback. Residency
and performance tuning remain separate work.

---

## 10 — TweenSplit

Ship: `systems::tween_split` + `runtime/tween_split_train` — collect trainable leaves, apply \(g_i=\frac1N P(g_y)\) @ full lr (`TSB1`/`TSP1`/`TRP1`…); w2a `--tweensplit-train`.  
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. Softmax weightless (N may be 0).

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.56 |
| RMSNorm | [x] | [x] | [x] | 0.56 |
| LayerNorm | [x] | [x] | [x] | 0.56 |
| SwiGLU | [x] | [x] | [x] | 0.56 |
| Softmax | [x] | [x] | [x] | 0.56 |
| Embedding | [x] | [x] | [x] | 0.56 |
| MHA | [x] | [x] | [x] | 0.56 |
| CNN1 | [x] | [x] | [x] | 0.56 |
| CNN2 | [x] | [x] | [x] | 0.56 |
| CNN3 | [x] | [x] | [x] | 0.56 |
| ConvT1 | [x] | [x] | [x] | 0.56 |
| ConvT2 | [x] | [x] | [x] | 0.56 |
| ConvT3 | [x] | [x] | [x] | 0.56 |
| RNN | [x] | [x] | [x] | 0.56 |
| LSTM | [x] | [x] | [x] | 0.56 |
| Sequential | [x] | [x] | [x] | 0.56 |
| Residual | [x] | [x] | [x] | 0.56 |
| KMeans | [x] | [x] | [x] | 0.56 |
| Mamba | [x] | [x] | [x] | 0.56 |
| Metacognition | [x] | [x] | [x] | 0.56 |
| GDN | [x] | [x] | [x] | 0.56 |
| Parallel | [x] | [x] | [x] | 0.56 |
| Stack | [x] | [x] | [x] | 0.56 |

**Checkpoint:** ✅ ModeTweenSplit signed (CUDA) — OpenSplitTape 1/N @ full lr; TSB1/TSP1 + per-kind volumes + KMeans/Parallel; 131 groups in 775.850s.

Parallel leaf-mode Split counts all trainable branch leaves, nested Stack/Parallel
leaves and its Filter gate before applying projected 1/N gaps at full lr. Softmax
is excluded from N, including the mixed Dense/Softmax ×34 fixture. Capture and
stateful Filter's extra forward finish before any update. Split loss and targets
use the collected output's native axes, matching Go `OpenSplitTape`.
Placed KMeans/Parallel volume cells retain the existing local-unit grid contract.

All 51 KMeans/Parallel groups passed, including 111/chain/211 ×34 and 222 Float32
volumes, all 23 polymorphic Parallel branch kinds, stateful Filter, native SGD and
exact fresh-versus-measured replay with at least two iterations. Numerical work
uses CUDA kernels; Go references stay in w2a. Other Split/Alt mode sign-offs remain
separate.

---

## 11 — StepTweenSplit

Needs Line + TweenSplit.  
Ship: `runtime/line` Credit::FamilyTweenSplit on pop (`apply_split_child` 1/N @ full lr); wire `STS1`/`STSO`; w2a `--steptweensplit-train`.  
**3D volumetric** = MeshTweenSplit volume (= ModeTweenSplit / TweenSplit volume wires `TSP1`/`TRP1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [x] | [x] | [x] | 0.57 |
| RMSNorm | [x] | [x] | [x] | 0.57 |
| LayerNorm | [x] | [x] | [x] | 0.57 |
| SwiGLU | [x] | [x] | [x] | 0.57 |
| Softmax | [x] | [x] | [x] | 0.57 |
| Embedding | [x] | [x] | [x] | 0.57 |
| MHA | [x] | [x] | [x] | 0.57 |
| CNN1 | [x] | [x] | [x] | 0.57 |
| CNN2 | [x] | [x] | [x] | 0.57 |
| CNN3 | [x] | [x] | [x] | 0.57 |
| ConvT1 | [x] | [x] | [x] | 0.57 |
| ConvT2 | [x] | [x] | [x] | 0.57 |
| ConvT3 | [x] | [x] | [x] | 0.57 |
| RNN | [x] | [x] | [x] | 0.57 |
| LSTM | [x] | [x] | [x] | 0.57 |
| Sequential | [x] | [x] | [x] | 0.57 |
| Residual | [x] | [x] | [x] | 0.57 |
| KMeans | [x] | [x] | [x] | 0.57 |
| Mamba | [x] | [x] | [x] | 0.57 |
| Metacognition | [x] | [x] | [x] | 0.57 |
| GDN | [x] | [x] | [x] | 0.57 |
| Parallel | [x] | [x] | [x] | 0.57 |
| Stack | [x] | [x] | [x] | 0.57 |

**Checkpoint:** ✅ ModeStepTweenSplit signed (CUDA) — Line FamilyTweenSplit 1/N @ full lr; STS1/STSO + TweenSplit volumes; 130 groups in 760.015s.

---

## 12 — TweenAlt

ModeTweenAlt: Split at full lr, refresh the prediction/gap, then Tween at lr×0.5.
Ship: `systems/tween_alt` + `runtime/tween_alt_train`; `TAB1`/`TABO` and `TAV1`/per-kind volume wires.
CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures. Default AltTimes=1.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 13 — StepTweenAlt

Needs Line + TweenAlt.  
Ship: `runtime/line` Credit::FamilyTweenAlt on pop (`apply_alt_child` Split→Tween frozen gap); wire `STA1`/`STAO`; w2a `--steptweenalt-train`.  
**3D volumetric** = MeshTweenAlt volume (= ModeTweenAlt / TweenAlt volume wires `TAV1`/`TRA1`…). CNN2/3 and ConvT2/3 have explicit ×34 leaf fixtures.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 14 — TweenSplitHeadProxy

Head backward/SGD uses full lr; its input gradient is projected to hidden leaves with 1/(N−1) sharing. Dense hidden updates use GradWOnly.

Wires: `GPB1`/`GPBO` (variant 1); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVCudaProxyTrain` and `TestGoVCudaProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 15 — TweenSplitLinear

Reverse linear-credit traversal updates each leaf at 1/N and propagates through the updated weights. Nested Stack, Residual and all Parallel combine routes retain their Go ordering.

Wires: `GPB1`/`GPBO` (variant 2); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVCudaProxyTrain` and `TestGoVCudaProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 16 — TweenSplitFastProxy

Compute a linear head proxy before updating the head, then share it across hidden leaves at 1/(N−1). Dense local updates use GradWOnly.

Wires: `GPB1`/`GPBO` (variant 3); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVCudaProxyTrain` and `TestGoVCudaProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 17 — TweenSplitLinearCache

Linear credit refreshes every 20 epochs. Intervening updates reuse cached projections rescaled by live/cached gap norms; GPU kernels compute norms and rescaling. The tests cross the refresh boundary with 22 epochs.

Wires: `GPB1`/`GPBO` (variant 4); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVCudaProxyTrain` and `TestGoVCudaProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 18 — TweenSplitHeadProxyAsync

Apply the previous proxy to hidden leaves, update the head, then capture the next proxy through updated head weights. The first epoch uses the fresh proxy; four-epoch tests exercise stale reuse.

Wires: `GPB1`/`GPBO` (variant 5); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVCudaProxyTrain` and `TestGoVCudaProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 19 — TweenSplitSparse

Update the head and one rotating hidden leaf per epoch, using the pre-update linear head proxy. Four-epoch tests exercise rotation and state reset.

Wires: `GPB1`/`GPBO` (variant 6); `PGB1`/`PGVO` for actual-type volumes. Tests: `TestGoVCudaProxyTrain` and `TestGoVCudaProxyVolumeTrain`. The leaf matrix covers all 23 families and KMeans/Parallel variants; volume matrices cover 111/211 supported native dtypes and 222 Float32. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 20 — MeshTweenSplit

Mesh-clock Split applies full-lr 1/N local credit to each enabled cell using the final captured tick. The gap remains frozen and passes to earlier cells only when input lengths agree.

Wires: `MGS1`/`MGSO`. `TestGoVCudaMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 21 — MeshTweenAlt

Mesh-clock Alt applies Split then half-lr Tween to the final captured local tape with the frozen popped gap. Shared volume Alt separately refreshes prediction/gap between phases.

Wires: `MGA1`/`MGAO`. `TestGoVCudaMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 22 — MeshTweenSplitFastProxy

Mesh-clock FastProxy applies per-cell pre-update head proxy credit on the final captured tick. Shared volume FastProxy uses the global collected volume tape.

Wires: `MGF1`/`MGFO`. `TestGoVCudaMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 23 — MeshTweenSplitSparse

Mesh-clock Sparse updates the head and one hidden leaf per eligible local unit. Stack/Parallel nodes own rotation state; shared volume Sparse owns grid state.

Wires: `MGQ1`/`MGQO`. `TestGoVCudaMeshSplitModes` covers 103 clock groups per mode plus shared volume groups: all 23 actual kinds, 111/211 supported native dtypes, 222 Float32, remote/recurrent links, both KMeans outputs, seven Parallel combines, zero-lr preservation and exact repeated replay.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 24 — StepTweenSplitHeadProxy

Line HeadProxy applies full-lr head backward/SGD on pop, then projects head input credit across hidden leaves. The popped global gap stays frozen. Shared volume coverage uses the section 14 HeadProxy path.

Wire: `STH1`/`STHO`. `TestGoVCudaProxyTrain/StepTweenSplitHeadProxy` covers 65 leaf/Line groups, including native CNN2/3 and ConvT2/3. Section 14 supplies its shared volume evidence. GDN is Float32 only.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 25 — StepTweenSplitLinear

Line per-hop credit on pop uses the shared CUDA `systems::tween_split::apply_variant_child` implementation (variant 2). The popped gap stays frozen. Reverse local linear credit uses updated weights and 1/N sharing.

Wire: `STL1`/`STLO`. Tests: `TestGoVCudaStepProxyTrain/StepTweenSplitLinear`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 15 uses `PGB1`/`PGVO` variant 2 and `TestGoVCudaProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 26 — StepTweenSplitFastProxy

Line per-hop credit on pop uses the shared CUDA `systems::tween_split::apply_variant_child` implementation (variant 3). The popped gap stays frozen. The local head proxy is captured before the head update, then shared across hidden leaves.

Wire: `STF1`/`STFO`. Tests: `TestGoVCudaStepProxyTrain/StepTweenSplitFastProxy`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 16 uses `PGB1`/`PGVO` variant 3 and `TestGoVCudaProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 27 — StepTweenSplitLinearCache

Line per-hop credit on pop uses the shared CUDA `systems::tween_split::apply_variant_child` implementation (variant 4). The popped gap stays frozen. Eligible Parallel hops retain cached projections and GPU-produced norms between pops. Dense hops have no accelerator state. Seven 26-tick fixtures cross the 20-pop refresh boundary for every Parallel combine.

Wire: `SLC1`/`SLCO`. Tests: `TestGoVCudaStepProxyTrain/StepTweenSplitLinearCache`; 76 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 17 uses `PGB1`/`PGVO` variant 4 and `TestGoVCudaProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 28 — StepTweenSplitHeadProxyAsync

Line per-hop credit on pop uses the shared CUDA `systems::tween_split::apply_variant_child` implementation (variant 5). The popped gap stays frozen. Eligible Parallel hops retain their previous hidden proxy between pops and capture the next proxy after updating the head. Dense hops have no accelerator state.

Wire: `SHA1`/`SHAO`. Tests: `TestGoVCudaStepProxyTrain/StepTweenSplitHeadProxyAsync`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 18 uses `PGB1`/`PGVO` variant 5 and `TestGoVCudaProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## 29 — StepTweenSplitSparse

Line per-hop credit on pop uses the shared CUDA `systems::tween_split::apply_variant_child` implementation (variant 6). The popped gap stays frozen. Eligible Parallel hops rotate the selected hidden leaf between pops. Dense hops have no accelerator state; the no-state hidden selection follows Go.

Wire: `SSP1`/`SSPO`. Tests: `TestGoVCudaStepProxyTrain/StepTweenSplitSparse`; 69 groups. Includes all 23 actual families, ×34 native dtype sweeps where supported, both KMeans outputs, seven Parallel combines, polymorphic branches and Filter state. CNN2/3 and ConvT2/3 use explicit rank-4/rank-5 fixtures.

**3D volumetric:** the corresponding parent proxy mode from section 19 uses `PGB1`/`PGVO` variant 6 and `TestGoVCudaProxyVolumeTrain`: 111/211 supported native dtypes and 222 Float32. This is shared volume credit, separate from the Line clock.

Run: `./scripts/linux_build_w2a.sh --training-25-29-all`.

| Layer | Inside node | 3D volumetric | Fully signed off | Ver |
|---|:---:|:---:|:---:|---|
| Dense | [ ] | [ ] | [ ] | |
| RMSNorm | [ ] | [ ] | [ ] | |
| LayerNorm | [ ] | [ ] | [ ] | |
| SwiGLU | [ ] | [ ] | [ ] | |
| Softmax | [ ] | [ ] | [ ] | |
| Embedding | [ ] | [ ] | [ ] | |
| MHA | [ ] | [ ] | [ ] | |
| CNN1 | [ ] | [ ] | [ ] | |
| CNN2 | [ ] | [ ] | [ ] | |
| CNN3 | [ ] | [ ] | [ ] | |
| ConvT1 | [ ] | [ ] | [ ] | |
| ConvT2 | [ ] | [ ] | [ ] | |
| ConvT3 | [ ] | [ ] | [ ] | |
| RNN | [ ] | [ ] | [ ] | |
| LSTM | [ ] | [ ] | [ ] | |
| Sequential | [ ] | [ ] | [ ] | |
| Residual | [ ] | [ ] | [ ] | |
| KMeans | [ ] | [ ] | [ ] | |
| Mamba | [ ] | [ ] | [ ] | |
| Metacognition | [ ] | [ ] | [ ] | |
| GDN | [ ] | [ ] | [ ] | |
| Parallel | [ ] | [ ] | [ ] | |
| Stack | [ ] | [ ] | [ ] | |

**Checkpoint:** ⬜ not signed yet (CUDA).

---

## Version log

| Tag / date | What landed |
|---|---|
| **0.00** (M0) | CUDA host scaffold (device / buffers / smoke saxpy / C ABI / gaps). Engine board **0 / 100**. Training checklist mirrored from welvet.metal — **all rows unchecked**. |
| **0.32** | NormalBP RNN + LSTM leaf+vol (`RNB1`/`RNV1`, `LSB1`/`LSV1`) via Dense IH/HH / 4-gate BPTT. Engine board **32 / 100**. |
| **0.51** | NormalBP Mode 1 fully signed (23 families leaf+vol). Engine board **51 / 100**. |
| **0.52** | StepBP Line FamilyBP (`STB1`) + Dense MeshBP (`MBP1`) + ModeMeshBP shared vols; `TestGoVCudaStepBPTrain` PASS. Engine board **52 / 100**. |
| **0.53** | Heterogeneous MeshBP (`MGB1`/`MGBO`) all 23 families — 103 clock groups; `TestGoVCudaMeshBPTrain` PASS (~1356s @ BENCH_ITERS=1). Engine board **53 / 100**. |
| **0.54** | ModeTween (`TWB1`/`TWV1`/per-kind/`KMTV`/`PNTV`) — 130 groups; `TestGoVCudaTweenTrain` PASS (~717s @ BENCH_ITERS=1). Engine board **54 / 100**. |
| **0.55** | ModeTweenChain (`TCB1`/`TCV1`/per-kind/`KMTC`/`PNTC`) — 130 groups; `TestGoVCudaTweenChainTrain` PASS (~742s @ BENCH_ITERS=1). Engine board **55 / 100**. |
| | Tick rows + bump version as w2a `go_v_cuda` suites pass. |

## Mixed Go / C++ / CUDA routing — sign-off evidence

- [ ] JSON `cuda` resolves to a real CUDA backend; no silent Go fallback.
- [ ] Explicit per-child placements survive Stack/Parallel inheritance.
- [ ] Incoming global gradients, dInput/dWeights and native SGD cross segment boundaries.
- [ ] Dense proxy primitives use CUDA when their layer is assigned CUDA.
- [ ] 13-layer Go 0–3 → C++ 4–8 → CUDA 9–12 forward/backward/update comparison.
- [ ] Grid (0,0,0), (1,1,1), (2,2,2) placement and existing remote-link semantics.
- [ ] Routed backward/SGD for all 23 baseline families over supported native dtypes (GDN Float32).
- [ ] Persistent homogeneous MSE mode delegation and mixed Go/C++/CUDA local training modes.
- [ ] Routed weights return automatically for Go execution/saving; explicit sessions keep explicit SyncToGo.
- [ ] Router lifecycle, missing-library errors and pending C++ parameter handoff.

Run `./scripts/linux_build_w2a.sh --go-routing`; see [GO_INTEGRATION.md](GO_INTEGRATION.md).
