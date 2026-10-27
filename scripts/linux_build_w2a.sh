#!/usr/bin/env bash
# Build welvet.cuda then run sibling w2a Go↔CUDA suites (as they land).
# M0: engine build + smoke only. Training flags mirror welvet.metal's macos_build_w2a.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="${W2A_DIR:-$ROOT/../w2a}"
BUILD="${BUILD_DIR:-$ROOT/build}"

usage() {
  cat <<EOF
usage: $0 [flags]

  (default)           build welvet.cuda + run example_cuda_smoke
  --smoke             same as default
  --gaps              build + welvet_dense_fwd --gaps
  --help

Future (land with M1+ / go_v_cuda):
  --dense-train       Dense NormalBP leaf/volume vs Go
  --normalbp-all      all NormalBP layer suites
  --training-all      all 29 modes + CUDA-only guards
EOF
}

MODE=smoke
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --dense-train|--normalbp-all|--training-all|*)
      echo "w2a go_v_cuda suite '$1' is not wired yet (M0 baseline)." >&2
      echo "Engine builds and smoke live here; parity suites land in sibling w2a." >&2
      exit 2
      ;;
  esac
fi

"$ROOT/scripts/build_linux.sh"

case "$MODE" in
  smoke)
    "$BUILD/example_cuda_smoke"
    ;;
  gaps)
    "$BUILD/welvet_dense_fwd" --gaps
    ;;
esac

if [[ -d "$W2A" ]]; then
  echo "w2a sibling: $W2A (go_v_cuda suites: gap? until M1)"
else
  echo "w2a sibling not found at $W2A (ok for M0 host-only)"
fi
