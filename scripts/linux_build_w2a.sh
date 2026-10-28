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
  --dense-train       Dense NormalBP leaf + volume (7 cases)
  --rmsnorm-train     RMSNorm NormalBP leaf + volume (7 cases)
  --layernorm-train   LayerNorm NormalBP leaf + volume (7 cases)
  --softmax-train     Softmax NormalBP leaf + volume (9 cases)
  --swiglu-train      SwiGLU NormalBP leaf + volume (7 cases)
  --embedding-train   Embedding NormalBP leaf + volume (7 cases)
  --mha-train         MHA NormalBP leaf + volume (8 cases)
  --cnn-train         CNN1/2/3 NormalBP leaf + volume
  --convt-train       ConvT1/2/3 NormalBP leaf + volume
  --rnn-train         RNN NormalBP leaf + volume
  --lstm-train        LSTM NormalBP leaf + volume
  --sequential-train  Sequential NormalBP leaf + volume
  --residual-train    Residual NormalBP leaf + volume
  --stack-train       Stack NestedStack-of-Dense NormalBP leaf + volume
  --mamba-train       Mamba NormalBP leaf + volume
  --metacognition-train Metacognition NormalBP leaf + volume
  --gdn-train         GDN NormalBP leaf + volume (f32)
  --kmeans-train      KMeans NormalBP leaf + volume
  --parallel-train    Parallel + ParallelPoly + NormalBPEdges
  --normalbp-all      Full TestGoVCudaNormalBP aggregator
  --stepbp-train      StepBP Line STB1 + Dense MeshBP MBP1 (+ ModeMeshBP vols)
  --meshbp-train      Heterogeneous MeshBP MGB1 (+ StepBP/volume + NormalBP + CUDA-only)
  --tween-train       ModeTween TWB1/TWV1 + per-kind volumes (TestGoVCudaTweenTrain)
  --tweenchain-train  ModeTweenChain TCB1/TCV1 + per-kind volumes (TestGoVCudaTweenChainTrain)
  --tweensplit-train  ModeTweenSplit TSB1/TSP1 + per-kind volumes (TestGoVCudaTweenSplitTrain)
  --steptweensplit-train  ModeStepTweenSplit STS1/STSO + TweenSplit volumes (TestGoVCudaStepTweenSplitTrain)
  --tweenalt-train    ModeTweenAlt TAB1/TAV1 + per-kind volumes (TestGoVCudaTweenAltTrain)
  --steptweenalt-train  ModeStepTweenAlt STA1/STAO + TweenAlt volumes (TestGoVCudaStepTweenAltTrain)
  --meshtween-train   MeshTween MTW1/MGW1 + ModeTween volumes (TestGoVCudaMeshTweenTrain)
  --meshtweenchain-train  MeshTweenChain MTC1/MGC1 + TweenChain volumes (TestGoVCudaMeshTweenChainTrain)
  --help
EOF
}

MODE=dense-fwd
if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --smoke) MODE=smoke; shift ;;
    --gaps) MODE=gaps; shift ;;
    --dense-fwd|--rmsnorm-fwd|--layernorm-fwd|--softmax-fwd|--swiglu-fwd|--embedding-fwd|--mha-fwd|--cnn-fwd|--convt-fwd|--rnn-fwd|--lstm-fwd|--sequential-fwd|--residual-fwd|--mamba-fwd|--metacognition-fwd|--gdn-fwd|--kmeans-fwd|--parallel-fwd|--seqmix-fwd|--leaf-fwd-all|--dense-train|--rmsnorm-train|--layernorm-train|--softmax-train|--swiglu-train|--embedding-train|--mha-train|--cnn-train|--convt-train|--rnn-train|--lstm-train|--sequential-train|--residual-train|--stack-train|--mamba-train|--metacognition-train|--gdn-train|--kmeans-train|--parallel-train|--normalbp-all|--stepbp-train|--meshbp-train|--tween-train|--tweenchain-train|--tweensplit-train|--steptweensplit-train|--tweenalt-train|--steptweenalt-train|--meshtween-train|--meshtweenchain-train)
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
  # Default 1 measured bench iter (match metal); set WELVET_CUDA_BENCH_ITERS=2000 for long perf.
  export WELVET_CUDA_BENCH_ITERS="${WELVET_CUDA_BENCH_ITERS:-1}"
  (
    cd "$W2A"
    env -u CC -u CXX -u CMAKE_CXX_COMPILER -u CMAKE_C_COMPILER \
      PATH="/usr/bin:/usr/local/go/bin:${PATH}" \
      WELVET_CUDA_DENSE_FWD="$WELVET_CUDA_DENSE_FWD" \
      WELVET_DENSE_FWD="${WELVET_DENSE_FWD:-}" \
      WELVET_CUDA_BENCH_ITERS="$WELVET_CUDA_BENCH_ITERS" \
      LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
      go test ./tests/go_v_cuda -v -count=1 -timeout 90m -run "$pattern"
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
  dense-train) run_go_test '^TestGoVCudaDenseTrain$' ;;
  rmsnorm-train) run_go_test '^TestGoVCudaRMSNormTrain$' ;;
  layernorm-train) run_go_test '^TestGoVCudaLayerNormTrain$' ;;
  softmax-train) run_go_test '^TestGoVCudaSoftmaxTrain$' ;;
  swiglu-train) run_go_test '^TestGoVCudaSwiGLUTrain$' ;;
  embedding-train) run_go_test '^TestGoVCudaEmbeddingTrain$' ;;
  mha-train) run_go_test '^TestGoVCudaMHATrain$' ;;
  cnn-train) run_go_test '^TestGoVCudaCNNTrain$' ;;
  convt-train) run_go_test '^TestGoVCudaConvTTrain$' ;;
  rnn-train) run_go_test '^TestGoVCudaRNNTrain$' ;;
  lstm-train) run_go_test '^TestGoVCudaLSTMTrain$' ;;
  sequential-train) run_go_test '^TestGoVCudaSequentialTrain$' ;;
  residual-train) run_go_test '^TestGoVCudaResidualTrain$' ;;
  stack-train) run_go_test '^TestGoVCudaStackTrain$' ;;
  mamba-train) run_go_test '^TestGoVCudaMambaTrain$' ;;
  metacognition-train) run_go_test '^TestGoVCudaMetacognitionTrain$' ;;
  gdn-train) run_go_test '^TestGoVCudaGDNTrain$' ;;
  kmeans-train) run_go_test '^TestGoVCudaKMeansTrain$' ;;
  parallel-train) run_go_test '^TestGoVCuda(ParallelTrain|ParallelPolyTrain|NormalBPEdges)$' ;;
  normalbp-all) run_go_test '^TestGoVCudaNormalBP$' ;;
  stepbp-train) run_go_test '^TestGoVCudaStepBPTrain$' ;;
  meshbp-train) run_go_test '^(TestGoVCudaMeshBPTrain|TestGoVCudaStepBPTrain|TestGoVCudaNormalBP)$' ;;
  tween-train) run_go_test '^TestGoVCudaTweenTrain$' ;;
  tweenchain-train) run_go_test '^TestGoVCudaTweenChainTrain$' ;;
  tweensplit-train) run_go_test '^TestGoVCudaTweenSplitTrain$' ;;
  steptweensplit-train) run_go_test '^TestGoVCudaStepTweenSplitTrain$' ;;
  tweenalt-train) run_go_test '^TestGoVCudaTweenAltTrain$' ;;
  steptweenalt-train) run_go_test '^TestGoVCudaStepTweenAltTrain$' ;;
  meshtween-train) run_go_test '^(TestGoVCudaMeshTweenTrain|TestGoVCudaMeshBPTrain|TestGoVCudaTweenTrain)$' ;;
  meshtweenchain-train) run_go_test '^(TestGoVCudaMeshTweenChainTrain|TestGoVCudaMeshTweenTrain|TestGoVCudaTweenChainTrain)$' ;;
esac
