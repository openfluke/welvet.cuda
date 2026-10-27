#include "welvet/gaps.hpp"

#include <iostream>

namespace welvet::gaps {
namespace {

const std::vector<Entry>& build() {
  static const std::vector<Entry> k = {
      // --- M0 CUDA host (this scaffold) ---
      {"cuda.device.init", "cuda", "cudaDevice probe + runtime identity", true},
      {"cuda.kernels.compile", "cuda", "nvcc compile src/cuda/kernels.cu (smoke + matvec stub)", true},
      {"cuda.buffer.device", "cuda", "cudaMalloc alloc/upload/download", true},
      {"cuda.kernel.smoke", "cuda", "Launch welvet_smoke_saxpy", true},
      {"abi.version", "interop", "welvet_version + welvet_cuda_init C ABI", true},
      {"platform.linux", "platform", "Linux x86_64 CUDA host build", true},

      // --- Core / weights (port from welvet.cpp / welvet.metal semantics) ---
      {"core.dtype.all34", "core", "all 34 DType names + FormatNone Dense forward gates", true},
      {"core.activate", "core", "Activate/ActivateDeriv host helpers (GPU act in Dense kernel)", true},
      {"core.fp_codecs", "core", "FP8/FP4/NF4/FP6 pack-unpack for FormatNone storage", true},
      {"weights.formatnone.x34", "weights", "FormatNone pack/decode ×34 for Dense forward", true},
      {"weights.matvec.cuda", "weights", "resident f64 Dense MatVec + bias/act on CUDA", true},
      {"weights.resident", "weights", "upload-once W/bias; warm execute reuses device buffers", true},

      // --- Layers (same 23 kinds as welvet.cpp / Go / welvet.metal) ---
      {"dense.forward.cuda", "layers", "Dense CUDA FormatNone×34 forward (exact_bit vs Go)", true},
      {"dense.normalbp.cuda", "layers", "gap? Dense NormalBP dX/dW/db + SGD CUDA", false},
      {"mha.forward.cuda", "layers", "gap? MHA ModeSelf CUDA", false},
      {"mha.normalbp.cuda", "layers", "gap? MHA NormalBP CUDA", false},
      {"swiglu.cuda", "layers", "gap? SwiGLU fwd+NormalBP CUDA", false},
      {"rmsnorm.cuda", "layers", "gap? RMSNorm fwd+NormalBP CUDA", false},
      {"layernorm.cuda", "layers", "gap? LayerNorm fwd+NormalBP CUDA", false},
      {"softmax.cuda", "layers", "gap? Softmax all Kind + dX CUDA", false},
      {"embedding.cuda", "layers", "gap? Embedding gather/scatter CUDA", false},
      {"cnn.cuda", "layers", "gap? CNN1/2/3 fwd+NormalBP CUDA", false},
      {"convt.cuda", "layers", "gap? ConvT1/2/3 fwd+NormalBP CUDA", false},
      {"rnn.cuda", "layers", "gap? RNN BPTT CUDA", false},
      {"lstm.cuda", "layers", "gap? LSTM BPTT CUDA", false},
      {"sequential.cuda", "layers", "gap? Sequential Dense chain CUDA", false},
      {"residual.cuda", "layers", "gap? Residual F(x)+x CUDA", false},
      {"kmeans.cuda", "layers", "gap? KMeans soft assignment CUDA", false},
      {"mamba.cuda", "layers", "gap? Mamba scan CUDA", false},
      {"metacognition.cuda", "layers", "gap? Metacognition CUDA", false},
      {"gdn.cuda", "layers", "gap? GDN f32 CUDA", false},
      {"stack.cuda", "layers", "gap? Stack NestedStack-of-Dense CUDA", false},
      {"parallel.cuda", "layers", "gap? Parallel branches + BranchModes CUDA", false},
      {"seqmix.cuda", "layers", "gap? seqmix helper CUDA", false},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture", "gap? Grid topology + remote links CUDA residency", false},
      {"runtime.forward", "runtime", "gap? volume forward CUDA", false},
      {"runtime.backward", "runtime", "gap? volume backward CUDA", false},
      {"runtime.step", "runtime", "gap? Line StepBP/StepTween* CUDA", false},
      {"runtime.mesh", "runtime", "gap? MeshBP/MeshTween* CUDA", false},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.camsync", "training", "gap? CamSync / BlendStores CUDA", false},

      // --- Systems / model ---
      {"systems.tween", "systems", "gap? Tween* projected credit CUDA", false},
      {"systems.dna", "systems", "gap? DNA", false},
      {"systems.evolution", "systems", "gap? evolution", false},
      {"systems.tanhi", "systems", "gap? TANHI", false},
      {"systems.telemetry", "systems", "gap? telemetry", false},
      {"model.tokenizer", "model", "gap? tokenizer", false},
      {"model.entity", "model", "gap? ENTITY", false},
      {"model.transformer", "model", "gap? transformer", false},
      {"model.sampling", "model", "gap? sampling", false},
      {"model.hf", "model", "gap? HF load", false},
      {"lucy.lpd", "lucy", "gap? Lucy/LPD", false},

      // --- Delivery / validation ---
      {"platform.windows", "platform", "gap? Windows CUDA artifact", false},
      {"w2a.go_v_cuda", "interop", "w2a Go_v_Cuda Dense forward parity + speed", true},
      {"nac.go_delegate", "interop", "gap? Go Welvet NAC → CUDA dispatch", false},
      {"determinism.policy", "cuda", "gap? documented reduction order + exact_bit bands", false},
  };
  return k;
}

} // namespace

const std::vector<Entry>& all() { return build(); }

std::size_t count_gaps() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (!e.implemented) ++n;
  }
  return n;
}

std::size_t count_done() {
  std::size_t n = 0;
  for (const auto& e : all()) {
    if (e.implemented) ++n;
  }
  return n;
}

void print_census(std::ostream& os) {
  os << "welvet.cuda gaps: done=" << count_done() << " open=" << count_gaps() << "\n";
  for (const auto& e : all()) {
    os << (e.implemented ? "  [done] " : "  [gap?] ") << e.id << " — " << e.detail << "\n";
  }
}

std::string mark(std::string_view what) {
  return std::string("gap? ") + std::string(what);
}

} // namespace welvet::gaps
