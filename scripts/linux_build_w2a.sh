#!/usr/bin/env bash
# Build welvet.cuda then run sibling w2a Go↔CUDA suites.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W2A="${W2A_DIR:-$ROOT/../w2a}"
BUILD="${BUILD_DIR:-$ROOT/build}"

usage() {
  cat <<EOF
usage: $0 [flags]

  (default)           build + TestGoVCudaDenseForward
  --smoke             build + example_cuda_smoke
  --gaps              build + welvet_dense_fwd --gaps
  --dense-fwd         Dense FormatNone×34 forward
  --rmsnorm-fwd       RMSNorm FormatNone×34 forward
  --layernorm-fwd     LayerNorm FormatNone×34 forward
  --softmax-fwd       Softmax forward
  --swiglu-fwd        SwiGLU forward
  --embedding-fwd     Embedding forward
  --mha-fwd           MHA forward
  --cnn-fwd           CNN1/2/3 forward
  --convt-fwd         ConvT1/2/3 forward
  --rnn-fwd           RNN forward
  --lstm-fwd          LSTM forward
  --sequential-fwd    Sequential forward
  --residual-fwd      Residual forward
  --mamba-fwd         Mamba forward
  --metacognition-fwd Metacognition forward
  --gdn-fwd           GDN forward
  --kmeans-fwd        KMeans forward
  --parallel-fwd      Parallel forward
  --seqmix-fwd        seqmix smoke
  --leaf-fwd-all      all leaf FormatNone forward suites
  --help
EOF
}

MODE=dense-fwd
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --dense-fwd|--rmsnorm-fwd|--layernorm-fwd|--softmax-fwd|--swiglu-fwd|--embedding-fwd|--mha-fwd|--cnn-fwd|--convt-fwd|--rnn-fwd|--lstm-fwd|--sequential-fwd|--residual-fwd|--mamba-fwd|--metacognition-fwd|--gdn-fwd|--kmeans-fwd|--parallel-fwd|--seqmix-fwd|--leaf-fwd-all)
      MODE="${1#--}"; shift ;;
    *)
      echo "unknown flag '$1'" >&2
      usage
      exit 2
      ;;
  esac
fi

"$ROOT/scripts/build_linux.sh"

export WELVET_CUDA_DENSE_FWD="$BUILD/welvet_dense_fwd"
if [[ -z "${WELVET_DENSE_FWD:-}" ]]; then
  for cand in "$ROOT/../welvet.cpp/build/welvet_dense_fwd" "$ROOT/../welvet.cpp/build/Release/welvet_dense_fwd"; do
    if [[ -x "$cand" ]]; then
      export WELVET_DENSE_FWD="$cand"
      break
    fi
  done
fi

run_go_test() {
  local pattern="$1"
  if [[ ! -d "$W2A" ]]; then
    echo "w2a sibling not found at $W2A" >&2
    exit 1
  fi
  export LD_LIBRARY_PATH="${ROOT}/.deps/cuda/cuda/lib64:${LD_LIBRARY_PATH:-}"
  (
    cd "$W2A"
    env -u CC -u CXX -u CMAKE_CXX_COMPILER -u CMAKE_C_COMPILER \
      PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
      WELVET_CUDA_DENSE_FWD="$WELVET_CUDA_DENSE_FWD" \
      WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-}" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      go test ./tests/go_v_cuda -v -count=1 -run "$pattern"
  )
}

case "$MODE" in
  smoke) "$BUILD/example_cuda_smoke" ;;
  gaps) "$BUILD/welvet_dense_fwd" --gaps ;;
  dense-fwd) run_go_test '^TestGoVCudaDenseForward$' ;;
  rmsnorm-fwd) run_go_test '^TestGoVCudaRMSNorm$' ;;
  layernorm-fwd) run_go_test '^TestGoVCudaLayerNorm$' ;;
  softmax-fwd) run_go_test '^TestGoVCudaSoftmax$' ;;
  swiglu-fwd) run_go_test '^TestGoVCudaSwiGLU$' ;;
  embedding-fwd) run_go_test '^TestGoVCudaEmbedding$' ;;
  mha-fwd) run_go_test '^TestGoVCudaMHA$' ;;
  cnn-fwd) run_go_test '^TestGoVCudaCNN$' ;;
  convt-fwd) run_go_test '^TestGoVCudaConvT$' ;;
  rnn-fwd) run_go_test '^TestGoVCudaRNN$' ;;
  lstm-fwd) run_go_test '^TestGoVCudaLSTM$' ;;
  sequential-fwd) run_go_test '^TestGoVCudaSequential$' ;;
  residual-fwd) run_go_test '^TestGoVCudaResidual$' ;;
  mamba-fwd) run_go_test '^TestGoVCudaMamba$' ;;
  metacognition-fwd) run_go_test '^TestGoVCudaMetacognition$' ;;
  gdn-fwd) run_go_test '^TestGoVCudaGDN$' ;;
  kmeans-fwd) run_go_test '^TestGoVCudaKMeans$' ;;
  parallel-fwd) run_go_test '^TestGoVCudaParallel$' ;;
  seqmix-fwd) run_go_test '^TestGoVCudaSeqmix$' ;;
  leaf-fwd-all) run_go_test '^TestGoVCuda(DenseForward|RMSNorm|LayerNorm|Softmax|SwiGLU|Embedding|MHA|CNN|ConvT|RNN|LSTM|Sequential|Residual|Mamba|Metacognition|GDN|KMeans|Parallel|Seqmix)$' ;;
esac
