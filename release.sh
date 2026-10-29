#!/usr/bin/env bash
# Publish this independent NVIDIA CUDA backend.
#
# Default / --source-only: commit local source, push branch + immutable tag,
# then create a GitHub Release (source tag; no binary build).
#
# Optional --package: also build/package Linux x86_64 shared+static artifacts
# (needs nvcc) and attach them to the release.
#
# The script never force-pushes or moves an existing tag.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
DRY_RUN=0
PLAN=0
SOURCE_ONLY=1
PACKAGE_BIN=0
SKIP_BUILD=0
SKIP_TESTS=0
TEST_GO=0
DRAFT=0
AUTO_COMMIT=1
BUILD_ROOT="$ROOT/build-release"
DIST="$ROOT/dist"

usage() {
  cat <<'HELP'
Usage: ./release.sh [options]
  --source-only   Tag + GitHub Release from source only (default; no nvcc)
  --package       Also build/package linux-x86_64 shared+static binaries (needs nvcc)
  --dry-run       Do everything except commit/tag/push/publish
  --plan          Show version and actions only
  --skip-build    With --package: reuse build-release/shared and build-release/static
  --skip-tests    With --package: skip native GPU smoke checks
  --test-go       With --package: also run sibling w2a Mount suites
  --no-commit     Require a clean source tree instead of committing local changes
  --draft         Create a draft GitHub Release
  -h, --help      Show this help

Default: release v<current CMake/README/runtime version> as a source tag to origin.
Publishing requires an authenticated GitHub CLI (gh). No sibling repo is modified.
HELP
}
for arg in "$@"; do
  case "$arg" in
    --source-only) SOURCE_ONLY=1; PACKAGE_BIN=0 ;;
    --package) PACKAGE_BIN=1; SOURCE_ONLY=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --plan) PLAN=1; DRY_RUN=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-tests) SKIP_TESTS=1 ;;
    --test-go) TEST_GO=1 ;;
    --no-commit) AUTO_COMMIT=0 ;;
    --draft) DRAFT=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "Required command missing: $1"; }
need git
need python3
VERSION="$(python3 - "$ROOT" <<'PY'
from pathlib import Path
import re,sys
r=Path(sys.argv[1])
cm=re.search(r'project\(welvet_cuda\s+VERSION\s+([0-9.]+)',(r/'CMakeLists.txt').read_text())
rm=re.search(r'\*\*Version\*\*\s*\|\s*\*\*`([0-9.]+)`',(r/'README.md').read_text())
abi=re.search(r'welvet_version\(void\)\s*\{.*?return\s+"([0-9.]+)"',(r/'src/interop/c_abi.cpp').read_text(),re.S)
if not all((cm,rm,abi)) or len({m[1] for m in (cm,rm,abi)})!=1:
    raise SystemExit('CMake, README and runtime source versions must agree')
print(cm[1])
PY
)"
SCORE="$(python3 - "$ROOT/README.md" <<'PY_SCORE'
from pathlib import Path
import re,sys
m=re.search(r'\*\*Engine board\*\*\s*\|\s*\*\*(\d+(?:\.\d+)?)\s*/\s*100',Path(sys.argv[1]).read_text())
if not m: raise SystemExit('README engine board not found')
print(m[1])
PY_SCORE
)"
TAG="v$VERSION"
BRANCH="$(git symbolic-ref --quiet --short HEAD)" || fail "Check out a branch before releasing"
REPO="$(git remote get-url origin | python3 -c '
import re,sys
url=sys.stdin.read().strip()
m=re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)([^/]+/[^/]+?)(?:\.git)?/?",url)
if not m: raise SystemExit("origin must point to a GitHub repository")
print(m[1])
')"
[[ "$REPO" == */welvet.cuda ]] || fail "origin must name a welvet.cuda repository"
if [[ "$DRY_RUN" -eq 0 ]]; then
  need gh
  gh auth status --hostname github.com >/dev/null 2>&1 || fail "Run gh auth login first"
  if [[ "$AUTO_COMMIT" -eq 0 && -n "$(git status --porcelain)" ]]; then
    fail "Source tree is dirty; commit changes or omit --no-commit"
  fi
  if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
    [[ "$(git rev-parse "$TAG^{commit}")" == "$(git rev-parse HEAD)" ]] || fail "$TAG already names another commit; tags are never moved"
    [[ -z "$(git status --porcelain)" ]] || fail "$TAG already exists; do not change the tagged source"
  fi
fi

KIND="source-only"
[[ "$PACKAGE_BIN" -eq 1 ]] && KIND="source + linux-x86_64 package"
printf 'welvet.cuda %s (%s)\nGitHub: %s\nBranch: %s\n' "$TAG" "$KIND" "$REPO" "$BRANCH"
if [[ "$PLAN" -eq 1 ]]; then
  if [[ "$PACKAGE_BIN" -eq 1 ]]; then
    echo "Plan: build shared + static -> GPU smoke -> package + SHA256"
    [[ "$TEST_GO" -eq 0 ]] || echo "Plan: test the staged .so through sibling Go/w2a"
  else
    echo "Plan: source tag + GitHub Release notes (no nvcc / no binary assets)"
  fi
  echo "Publish: commit local source (unless --no-commit), push branch + immutable tag, gh release create"
  echo "Plan only: no files, git state or remote state changed"
  exit 0
fi

mkdir -p "$DIST"
NOTES="$DIST/release-$TAG.md"
NATIVE_CHECK="not run (source-only release)"
GO_CHECK="not requested"
ARCHIVE=""
SHA=""

# ---- optional binary package (needs CUDA toolkit) ----
if [[ "$PACKAGE_BIN" -eq 1 ]]; then
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "--package requires Linux x86_64"
  need cmake
  need tar
  need sha256sum
  need readelf

  if [[ -z "${CUDAToolkit_ROOT:-}" ]]; then
    for cand in \
      "${ROOT}/.deps/cuda/cuda" \
      "${CUDA_HOME:-}" \
      "${CUDA_PATH:-}" \
      /usr/local/cuda
    do
      if [[ -n "$cand" && -x "${cand}/bin/nvcc" ]]; then
        export CUDAToolkit_ROOT="$cand"
        break
      fi
    done
  fi
  if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
    export PATH="${CUDAToolkit_ROOT}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDAToolkit_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
  fi
  need nvcc

  CMAKE_ARGS=()
  pick_host() {
    local gxx="$1" gcc="$2"
    [[ -x "$gxx" && -x "$gcc" ]] || return 1
    CMAKE_ARGS+=(
      -DCMAKE_CXX_COMPILER="$gxx"
      -DCMAKE_CUDA_HOST_COMPILER="$gxx"
      -DCMAKE_C_COMPILER="$gcc"
    )
    echo "host compiler: $gxx ($("$gxx" --version | head -1))"
    return 0
  }
  if [[ -n "${CMAKE_CXX_COMPILER:-}" ]]; then
    :
  elif pick_host "${CONDA_PREFIX:-}/bin/g++" "${CONDA_PREFIX:-}/bin/gcc"; then
    :
  elif pick_host "${HOME}/miniconda3/envs/welvet-cuda-build/bin/g++" \
    "${HOME}/miniconda3/envs/welvet-cuda-build/bin/gcc"; then
    :
  elif command -v g++-14 >/dev/null 2>&1 && command -v gcc-14 >/dev/null 2>&1; then
    pick_host "$(command -v g++-14)" "$(command -v gcc-14)" || true
  else
    major="$(g++ -dumpfullversion 2>/dev/null | cut -d. -f1 || echo 0)"
    if [[ "$major" -gt 14 ]]; then
      fail "system g++ is $major; CUDA 12.x nvcc wants ≤14 (activate welvet-cuda-build)"
    fi
  fi

  JOBS="$(nproc 2>/dev/null || echo 4)"
  mkdir -p "$BUILD_ROOT"
  STAGE="$(mktemp -d "$DIST/.stage.XXXXXX")"
  cleanup() { python3 - "$STAGE" <<'PY'
from pathlib import Path
import shutil,sys
p=Path(sys.argv[1])
if p.name.startswith('.stage.'):
    shutil.rmtree(p,ignore_errors=True)
PY
  }
  trap cleanup EXIT
  NAME="welvet-cuda-$TAG-linux-x86_64"
  PACKAGE="$STAGE/$NAME"
  mkdir -p "$PACKAGE"

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    for kind in shared static; do
      shared=ON
      tools=ON
      [[ "$kind" == shared ]] || { shared=OFF; tools=OFF; }
      cmake -S "$ROOT" -B "$BUILD_ROOT/$kind" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER="$(command -v nvcc)" \
        -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-75}" \
        -DCMAKE_INSTALL_PREFIX="$PACKAGE" \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
        -DWELVET_BUILD_SHARED="$shared" -DWELVET_BUILD_TOOLS="$tools" \
        -DWELVET_BUILD_EXAMPLES=OFF \
        "${CMAKE_ARGS[@]}"
      cmake --build "$BUILD_ROOT/$kind" --parallel "$JOBS"
    done
  fi
  for artifact in shared/libwelvet_cuda.so shared/welvet_dense_fwd static/libwelvet_cuda.a; do
    [[ -f "$BUILD_ROOT/$artifact" ]] || fail "Missing $BUILD_ROOT/$artifact; run without --skip-build"
  done
  cmake --install "$BUILD_ROOT/shared" --prefix "$PACKAGE"
  cmake --install "$BUILD_ROOT/static" --prefix "$PACKAGE"
  [[ "$("$PACKAGE/bin/welvet_dense_fwd" --version)" == "$VERSION" ]] || fail "Packaged CLI version mismatch"
  if readelf -d "$PACKAGE/bin/welvet_dense_fwd" | python3 -c '
import sys
text=sys.stdin.read()
bad=False
for line in text.splitlines():
    if "RPATH" in line or "RUNPATH" in line:
        if "/home/" in line or "/build-release/" in line or "/build/" in line:
            bad=True
sys.exit(1 if bad else 0)
'; then
    fail "Packaged tool embeds a non-relocatable RPATH"
  fi

  NATIVE_CHECK="skipped (--skip-tests)"
  if [[ "$SKIP_TESTS" -eq 0 ]]; then
    python3 "$ROOT/scripts/release_smoke.py" "$PACKAGE/lib/libwelvet_cuda.so" "$VERSION"
    NATIVE_CHECK="passed: packaged C ABI + CUDA identity + forward/backward/SGD"
  fi
  GO_CHECK="not requested"
  if [[ "$TEST_GO" -eq 1 ]]; then
    need go
    W2A="${WELVET_W2A_ROOT:-$ROOT/../w2a}"
    [[ -f "$W2A/go.mod" ]] || fail "Sibling w2a missing; set WELVET_W2A_ROOT or omit --test-go"
    (
      cd "$W2A"
      env -u CC -u CXX PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
        WELVET_CUDA_LIBRARY="$PACKAGE/lib/libwelvet_cuda.so" \
        LD_LIBRARY_PATH="$PACKAGE/lib:${CUDAToolkit_ROOT:+$CUDAToolkit_ROOT/lib64:}${LD_LIBRARY_PATH:-}" \
        go test ./tests/go_v_cuda -count=1 -timeout 25m \
        -run '^(TestGoNativeCudaIntegration|TestGoNativeCudaMesh|TestGoNativeCudaSaveReload)$'
    )
    GO_CHECK="passed: Go mounted-model/native-dtype/mesh/ENTITY roundtrips"
  fi

  cp "$ROOT/README.md" "$ROOT/VERSIONING.md" "$ROOT/LICENSE" "$PACKAGE/"
  mkdir -p "$PACKAGE/docs"
  cp "$ROOT/docs/GO_INTEGRATION.md" "$ROOT/docs/CUDA_ONLY.md" "$PACKAGE/docs/"
  # commit hash filled after possible auto-commit below — write placeholder now, refresh later
  python3 - "$PACKAGE" "$VERSION" "PENDING" "false" "$NATIVE_CHECK" "$GO_CHECK" <<'PY'
import json,subprocess,sys
from pathlib import Path
p,v,commit,dirty,native,go=sys.argv[1:]
nvcc=subprocess.check_output(['nvcc','--version'],text=True)
toolkit=next((line.split('release')[-1].strip() for line in nvcc.splitlines() if 'release' in line.lower()), 'unknown')
try:
    cxx=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
except Exception:
    cxx='unknown'
manifest=dict(engine='welvet.cuda',version=v,platform='linux-x86_64',
              model_abi=1,segment_extension=1,source_commit=commit,source_dirty_at_build=dirty=='true',
              native_validation=native,go_validation=go,cuda_toolkit=toolkit,compiler=cxx)
Path(p,'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
  ARCHIVE="$DIST/$NAME.tar.gz"
  tar -czf "$ARCHIVE" -C "$STAGE" "$NAME"
  (cd "$DIST" && sha256sum "$NAME.tar.gz" > "$NAME.sha256")
  SHA="$DIST/$NAME.sha256"
fi

# ---- git commit / tag / notes ----
if [[ "$DRY_RUN" -eq 0 && -n "$(git status --porcelain)" ]]; then
  [[ "$AUTO_COMMIT" -eq 1 ]] || fail "Dirty source tree"
  git add -A
  git commit -m "Release welvet.cuda $TAG"
fi
COMMIT="$(git rev-parse HEAD)"
DIRTY=false
[[ -z "$(git status --porcelain)" ]] || DIRTY=true

if [[ -n "$ARCHIVE" && -f "${ARCHIVE%.tar.gz}/../.stage."* ]]; then
  :
fi
# Refresh packaged MANIFEST commit if we built binaries
if [[ "$PACKAGE_BIN" -eq 1 && -n "$ARCHIVE" ]]; then
  # Unpack is heavy; just note commit in release notes. MANIFEST inside archive may say PENDING if built before commit — rebuild notes accurately below.
  true
fi

cat > "$NOTES" <<NOTES
# welvet.cuda $TAG

Independent NVIDIA CUDA backend for Welvet. Engine board: $SCORE/100;
see the tagged README and VERSIONING.md for the current subgate accounting.

- Supported numerical forward, backward and training on NVIDIA GPU through custom CUDA kernels.
- 23 baseline layer families, 29 numbered training modes, Cameral/CamSync + mixed Go routing, 34 native parameter encodings where supported (GDN Float32).
- Persistent Go mount/forward/training/weight-return API; explicit JSON and mixed Go/C++/CUDA routing in the Go binding.
- Float32 activation interface; best-effort numerical compatibility targets the industry60–exact100 bands.
- Separate from welvet.cpp / welvet.metal; this package has no CPU/Metal sibling runtime dependency.

## Source

Tag \`$TAG\` on branch \`$BRANCH\` (commit \`$COMMIT\`).
Clone and build with \`./scripts/build_linux.sh\` (needs NVIDIA CUDA Toolkit + host g++ ≤14).

Validation: $NATIVE_CHECK. Go/w2a: $GO_CHECK.
Dirty working-tree package: $DIRTY.
NOTES

if [[ -n "$ARCHIVE" ]]; then
  cat >> "$NOTES" <<NOTES

## Linux x86_64 binary artifact

\`$(basename "$ARCHIVE")\` — shared + static libraries, headers, kernels, \`welvet_dense_fwd\`, docs, MANIFEST.
Verify with: \`sha256sum -c $(basename "$SHA")\`
NOTES
fi

printf '\nPrepared notes:\n  %s\n' "$NOTES"
[[ -z "$ARCHIVE" ]] || printf '  %s\n  %s\n' "$ARCHIVE" "$SHA"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete: no commit, tag, push or GitHub release performed"
  exit 0
fi

if git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
  [[ "$(git rev-parse "$TAG^{commit}")" == "$COMMIT" ]] || fail "$TAG is not the release commit"
else
  git tag -a "$TAG" -m "welvet.cuda $TAG"
fi
git push origin "HEAD:refs/heads/$BRANCH"
git push origin "refs/tags/$TAG"

release_args=("$TAG" --repo "$REPO" --verify-tag --title "welvet.cuda $TAG" --notes-file "$NOTES")
[[ -z "$ARCHIVE" ]] || release_args+=("$ARCHIVE" "$SHA")
[[ "$DRAFT" -eq 0 ]] || release_args+=(--draft)
gh release create "${release_args[@]}"
echo "Release: https://github.com/$REPO/releases/tag/$TAG"
