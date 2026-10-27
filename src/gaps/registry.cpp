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
      {"mha.forward.cuda", "layers", "MHA ModeSelf FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"mha.normalbp.cuda", "layers", "gap? MHA NormalBP CUDA", false},
      {"swiglu.forward.cuda", "layers", "SwiGLU FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"swiglu.normalbp.cuda", "layers", "gap? SwiGLU NormalBP CUDA", false},
      {"rmsnorm.cuda", "layers", "RMSNorm FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"layernorm.cuda", "layers", "LayerNorm FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"softmax.cuda", "layers", "Softmax all Kind FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"embedding.forward.cuda", "layers", "Embedding FormatNone×34 CUDA gather (exact_bit vs Go)", true},
      {"embedding.normalbp.cuda", "layers", "gap? Embedding scatter NormalBP CUDA", false},
      {"cnn.cuda", "layers", "CNN1/2/3 FormatNone×34 CUDA forward (im2col→Dense exact_bit)", true},
      {"cnn.normalbp.cuda", "layers", "gap? CNN1/2/3 NormalBP CUDA", false},
      {"convt.cuda", "layers", "ConvT1/2/3 FormatNone×34 CUDA forward (f64 scatter exact_bit)", true},
      {"convt.normalbp.cuda", "layers", "gap? ConvT1/2/3 NormalBP CUDA", false},
      {"rnn.cuda", "layers", "RNN FormatNone×34 CUDA forward (Dense IH/HH exact_bit)", true},
      {"rnn.normalbp.cuda", "layers", "gap? RNN BPTT CUDA", false},
      {"lstm.cuda", "layers", "LSTM FormatNone×34 CUDA forward (4-gate Dense exact_bit)", true},
      {"lstm.normalbp.cuda", "layers", "gap? LSTM BPTT CUDA", false},
      {"sequential.cuda", "layers", "Sequential Dense chain FormatNone×34 CUDA forward (exact_bit)",
       true},
      {"residual.cuda", "layers", "Residual F(x)+x FormatNone×34 CUDA forward (exact_bit)", true},
      {"kmeans.cuda", "layers", "KMeans soft assignment FormatNone×34 CUDA (exact_bit)", true},
      {"mamba.cuda", "layers", "Mamba Dense+scan FormatNone×34 CUDA forward (exact_bit)", true},
      {"metacognition.cuda", "layers", "Metacognition Observed+rules FormatNone×34 CUDA (exact_bit)",
       true},
      {"gdn.cuda", "layers", "GDN Float32-only f32 forward (exact_bit vs Go)", true},
      {"stack.cuda", "layers", "gap? Stack NestedStack-of-Dense CUDA", false},
      {"parallel.cuda", "layers", "Parallel Dense branches + combine modes CUDA (exact_bit)", true},
      {"seqmix.cuda", "layers", "seqmix Kind name smoke WSX1 (metadata, no tensors)", true},

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
