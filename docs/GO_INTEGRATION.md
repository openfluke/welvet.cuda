# Go integration: mounted models and mixed backend routing

The native artifact is `libwelvet_cuda.so`. Go Welvet's `nativecuda`
package loads it through a versioned C ABI. No Go pointers are retained by C++.
The backend owns copies of model parameters, configuration and its training
state until `Close`. It never calls welvet.cpp or silently trains on the CPU.
Packaging is outside the engine completion score. The model-session ABI starts
at version 1.

## Build and install

From the CUDA checkout, with the Go sibling at `../welvet`:

```sh
# Linux x86_64
./scripts/linux_install_go.sh

# Windows amd64 (PowerShell; MSVC Build Tools + CUDA Toolkit)
./scripts/windows_install_go.ps1
```

Linux installs `libwelvet_cuda.so` into Go's `native/cuda/linux_x86_64/`.
Windows installs `libwelvet_cuda.dll` into `native/cuda/windows_amd64/`.
Both write the C header and SHA-256 manifest. Override `WELVET_GO_ROOT` for
another Go checkout. Building the backend requires CMake, a CUDA Toolkit
(`nvcc`), and a host compiler supported by that toolkit (GCC ≤ 14 for CUDA
12.x on Linux; MSVC on Windows). Go applications using the prebuilt library
need Go/cgo and a compatible NVIDIA driver/runtime; they do not rebuild the
C++ engine or require welvet.cpp.

Verified hosts: Linux x86_64 and Windows amd64 with an NVIDIA GPU. CUDA
architectures can be selected with `CMAKE_CUDA_ARCHITECTURES` (Windows default
`75;86`).

## Explicit weight ownership

In a Go application:

```go
import "github.com/openfluke/welvet/nativecuda"

session, err := nativecuda.Mount(model) // layer, Stack, Parallel or Grid
if err != nil { return err }
defer session.Close()

loss, err := session.TrainMSE(input, target, parallel.ModeNormalBP, 0.001)
if err != nil { return err }
_ = loss

output, err := session.Forward(input)
if err != nil { return err }
_ = output

// Go stores still contain their original weights until this explicit call.
if err := session.SyncToGo(); err != nil { return err }
return entity.WriteCameralFile("trained.entity", model, parallel.ModeNormalBP)
```

The save example assumes `model` is a `*parallel.Stack` supported by Go's
Cameral ENTITY writer. Other model shapes use their existing Go save APIs.
`SyncToGo` restores native bytes, scale, Float32 payloads and Float64 biases.
Mamba's ALog/DSkip and GDN's projection/parameter arrays are included. Derived
Go weight caches are replaced along with the stores. Export preparation and
metadata validation finish before Go weights are committed.

`Weights()` returns an owned snapshot without changing Go. `SyncCams(when)`
applies CamSync inside the native session and is separate from `SyncToGo()`.
`Close()` is idempotent and discards any unpulled updates; it never saves or
implicitly changes Go weights. Mount reads the current Go Store values; pull any
pending updates from another backend into Go first. The native session continues
from its current weights after a pull. Mounting the same Go model again creates
a new independent session from the Go values at that point.

Do not replace the source model's topology/stores, or mutate/read/save its
weights concurrently with `SyncToGo`. Calls on one native Session serialize;
separate sessions own independent parameter copies. A failure during a native
training operation is reported; training is not a transactional rollback API.

## Build and deployment selection

Build a CUDA-only Go application with:

```sh
go build -tags welvet_cuda ./cmd/your-app
```

The tag disables the independent `nativecpp` cgo binding while retaining Go's
explicit CPU implementations. Without it, the existing Go CPU-native binding
retains its original build behavior and may link the CPU sibling. Enable
`nativecuda.EnableRouting()` to connect `Exec.Backend` and the existing JSON
configuration to persistent CUDA mounts. Explicit `Mount` sessions remain
available independently. Automatic hardware choice/performance planning remains
separate from explicit JSON placement.

Library resolution order is an explicit `Options{LibraryPath: ...}`, then
`WELVET_CUDA_LIBRARY`, then `libwelvet_cuda.so` beside the executable, then the
installed artifact in the Go checkout. For distribution, place the `.so` beside
the executable or supply an explicit path/environment variable. A compiled
source-checkout path is only a development fallback, not a deployment strategy.
Loading an incompatible library/platform returns an error.

## Model descriptor (WMN1) and weight snapshot (NWS1)

Persistent mounts use little-endian binary descriptors:

- Magic `WMN1` (`0x314e4d57`)
- `root` ∈ {0=Stack, 1=Parallel, 2=Grid}
- child count, optional grid dims `[D,R,C,L]`
- recursive children via the same child wire as dense_fwd / CMB1
- Cameral controls for the root and each nested child
- store count + FormatNone payloads (dtype, rows, cols, scale, f32, native, bias)
- for Grid roots: per-cell enable/remote-link targets

`welvet_model_export_weights` returns snapshot magic `NWS1` (`0x3153574e`) with
store payloads and optional Mamba/GDN extras. Callers free blobs with
`welvet_cuda_blob_free`.

## Segment / credit APIs

ABI version 1 includes optional segment helpers that never invent a local loss:

- `welvet_model_forward_pair` — Stack/Parallel pre+post
- `welvet_model_backward` — Stack/Parallel dInput+dWeights from an external gy
- `welvet_model_apply_sgd` — apply packed weight gradients
- `welvet_model_train_credit` — Cameral-aware credit step
- `welvet_model_dense_proxy` — LinearGradIn / GradWOnly for a single Dense Stack
- `welvet_model_sync` — CamSync now / AfterPulse / AfterStep / AfterSample

Grid sessions train through `welvet_model_train_mse` with Mesh* / NormalBP /
Tween* modes; cell CamSync is configured at mount.

## Development smoke

```sh
./scripts/build_linux.sh
./scripts/linux_install_go.sh
export WELVET_CUDA_LIBRARY="$PWD/../welvet/native/cuda/linux_x86_64/libwelvet_cuda.so"
export WELVET_CUDA_DENSE_FWD="$PWD/build/welvet_dense_fwd"
# Mixed JSON routing (requires C++ dense_fwd when available):
./scripts/linux_build_w2a.sh --go-routing
```

Do not confuse wire CLI suites (`welvet_dense_fwd` magics such as CMB1/PNB1)
with the cgo model-session path; both share Cameral controls and CamSync
semantics, but the session ABI is the supported Go Mount surface.
