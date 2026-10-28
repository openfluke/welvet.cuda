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
      {"dense.normalbp.cuda", "layers", "Dense NormalBP dX/dW + FormatNone SGD CUDA (leaf+vol)",
       true},
      {"mha.forward.cuda", "layers", "MHA ModeSelf FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"mha.normalbp.cuda", "layers", "MHA NormalBP ModeSelf + RoPEᵀ CUDA (leaf+vol)", true},
      {"swiglu.forward.cuda", "layers", "SwiGLU FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"swiglu.normalbp.cuda", "layers", "SwiGLU NormalBP projs+SiLU′ CUDA (leaf+vol)", true},
      {"rmsnorm.cuda", "layers", "RMSNorm FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"rmsnorm.normalbp.cuda", "layers", "RMSNorm NormalBP dX/dγ + SGD CUDA (leaf+vol)", true},
      {"layernorm.cuda", "layers", "LayerNorm FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"layernorm.normalbp.cuda", "layers", "LayerNorm NormalBP dX/dγβ + SGD CUDA (leaf+vol)", true},
      {"softmax.cuda", "layers", "Softmax all Kind FormatNone×34 CUDA forward (exact_bit vs Go)", true},
      {"softmax.normalbp.cuda", "layers", "Softmax NormalBP dX-only CUDA (leaf+vol, weightless)",
       true},
      {"embedding.forward.cuda", "layers", "Embedding FormatNone×34 CUDA gather (exact_bit vs Go)", true},
      {"embedding.normalbp.cuda", "layers", "Embedding scatter NormalBP CUDA (leaf+vol)", true},
      {"cnn.cuda", "layers", "CNN1/2/3 FormatNone×34 CUDA forward (im2col→Dense exact_bit)", true},
      {"cnn.normalbp.cuda", "layers", "CNN1/2/3 im2col→Dense NormalBP CUDA (leaf+vol)", true},
      {"convt.cuda", "layers", "ConvT1/2/3 FormatNone×34 CUDA forward (f64 scatter exact_bit)", true},
      {"convt.normalbp.cuda", "layers", "ConvT1/2/3 scatter adjoint NormalBP CUDA (leaf+vol)", true},
      {"rnn.cuda", "layers", "RNN FormatNone×34 CUDA forward (Dense IH/HH exact_bit)", true},
      {"rnn.normalbp.cuda", "layers", "RNN BPTT Dense IH/HH + tanh′ + SGD FormatNone×34; RNB1/RNV1",
       true},
      {"lstm.cuda", "layers", "LSTM FormatNone×34 CUDA forward (4-gate Dense exact_bit)", true},
      {"lstm.normalbp.cuda", "layers", "LSTM BPTT 4-gate Dense + SGD FormatNone×34; LSB1/LSV1",
       true},
      {"sequential.cuda", "layers", "Sequential Dense chain FormatNone×34 CUDA forward (exact_bit)",
       true},
      {"sequential.normalbp.cuda", "layers",
       "Sequential NormalBP reverse Dense chain + ApplyGradSGD FormatNone×34; SQB1/SQV1", true},
      {"residual.cuda", "layers", "Residual F(x)+x FormatNone×34 CUDA forward (exact_bit)", true},
      {"residual.normalbp.cuda", "layers",
       "Residual NormalBP F reverse Dense + skip grad + ApplyGradSGD FormatNone×34; REB1/REV1",
       true},
      {"kmeans.cuda", "layers", "KMeans soft assignment FormatNone×34 CUDA (exact_bit)", true},
      {"kmeans.normalbp.cuda", "layers",
       "KMeans NormalBP soft-assignment + SGD FormatNone×34; KMB1/KMV1", true},
      {"mamba.cuda", "layers", "Mamba Dense+scan FormatNone×34 CUDA forward (exact_bit)", true},
      {"mamba.normalbp.cuda", "layers",
       "Mamba NormalBP scan BPTT + In/Out Dense FormatNone×34; MBB1/MBV1", true},
      {"metacognition.cuda", "layers", "Metacognition Observed+rules FormatNone×34 CUDA (exact_bit)",
       true},
      {"metacognition.normalbp.cuda", "layers",
       "Metacognition NormalBP lastGate + Observed Dense FormatNone×34; MCB1/MCV1", true},
      {"gdn.cuda", "layers", "GDN Float32-only f32 forward (exact_bit vs Go)", true},
      {"gdn.normalbp.cuda", "layers",
       "GDN truncated-BPTT NormalBP f32-only; GDB1/GDV1", true},
      {"stack.cuda", "layers", "Stack NestedStack-of-Dense CUDA (NormalBP STKB/STKV)", true},
      {"stack.normalbp.cuda", "layers",
       "C ABI welvet_stack_normalbp_f32 + STKB/STKV tool", true},
      {"parallel.cuda", "layers", "Parallel Dense branches + combine modes CUDA (exact_bit)", true},
      {"parallel.normalbp.cuda", "layers",
       "Parallel NormalBP Dense/poly + edges + volume; PNB1/PNV1/PPB1/NBL1", true},
      {"seqmix.cuda", "layers", "seqmix Kind name smoke WSX1 (metadata, no tensors)", true},

      // --- Runtime / architecture / training ---
      {"arch.grid", "architecture",
       "Grid topology + binds Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel", true},
      {"runtime.forward", "runtime",
       "volume forward Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel CUDA", true},
      {"runtime.backward", "runtime",
       "volume backward Dense…Stack/Mamba/Meta/GDN/KMeans/Parallel CUDA", true},
      {"runtime.step", "runtime", "Line StepBP FamilyBP (STB1) + runtime/line", true},
      {"runtime.mesh", "runtime",
       "Heterogeneous MeshBP FamilyBP (MGB1) + Dense MBP1 + runtime/step", true},
      {"train.modes.x29", "training", "gap? all 29 TrainModes + Freeze/Shadow/Adv/Memory", false},
      {"train.camsync", "training", "gap? CamSync / BlendStores CUDA", false},

      // --- Systems / model ---
      {"systems.tween", "systems", "ModeTween projected credit CUDA", true},
      {"systems.tween_chain", "systems",
       "ModeTweenChain FamilyBP @ full lr; TCB1/TCBO + TCV1/TCVO + TRC1…MBTC", true},
      {"runtime.tween_chain_train", "runtime",
       "TrainStackMSE ModeTweenChain (FamilyBP @ full lr)", true},
      {"systems.tween_split", "systems",
       "ModeTweenSplit 1/N projected gap @ full lr; TSB1/TSP1 + TRP1…", true},
      {"runtime.tween_split_train", "runtime",
       "TrainStackMSE ModeTweenSplit (OpenSplitTape 1/N)", true},
      {"runtime.line.family_tween", "runtime",
       "Line FamilyTween on pop (STW1/STWO; apply_projected @ lr×0.5)", true},
      {"runtime.line.family_tween_chain", "runtime",
       "Line StepTweenChain FamilyBP alias (STC1/STCO ≡ StepBP on pop @ full lr)", true},
      {"systems.tween_split_head_proxy", "systems",
       "ModeTweenSplitHeadProxy head Backward+SGD @ full lr; hidden 1/(N−1) GradWOnly; GPB1/PGB1 variant 1",
       true},
      {"systems.tween_split_linear", "systems",
       "ModeTweenSplitLinear reverse walk 1/N + LinearGradIn; GPB1/PGB1 variant 2", true},
      {"systems.tween_split_fast_proxy", "systems",
       "ModeTweenSplitFastProxy linear head proxy then hidden 1/(N−1) GradWOnly; GPB1/PGB1 variant 3",
       true},
      {"runtime.line.family_tween_split", "runtime",
       "Line FamilyTweenSplit on pop (STS1/STSO; apply_split_child 1/N @ full lr)", true},
      {"systems.tween_alt", "systems",
       "ModeTweenAlt Split@lr → Tween@lr×0.5; TAB1/TAV1 + TRA1…KMAL/PNAL", true},
      {"runtime.tween_alt_train", "runtime",
       "TrainStackMSE ModeTweenAlt (AltTimes × Split then Tween soft lr)", true},
      {"runtime.line.family_tween_alt", "runtime",
       "Line FamilyTweenAlt on pop (STA1/STAO; apply_alt_child Split→Tween frozen gap)", true},
      {"runtime.step.family_tween", "runtime",
       "MeshTween last-tick local credit (MTW1/MTWO, MGW1/MGWO; apply_cell_projected @ lr×0.5)",
       true},
      {"runtime.step.meshtweenchain", "runtime",
       "MeshTweenChain ≡ MeshBP FamilyBP alias (MTC1/MTCO, MGC1/MGCO)", true},
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
