#pragma once

enum class VolumeTrainMode {
  NormalBP = 0,
  Tween = 1,
  TweenChain = 2,
  TweenSplit = 3,
  TweenAlt = 4,
  // GPB1 variants: main passes 4+variant (HeadProxy=1 → ProxyHead=5).
  ProxyHead = 5,
  ProxyLinear = 6,
  ProxyFast = 7,
  ProxyCache = 8,
  ProxyAsync = 9,
  ProxySparse = 10,
};

// Dense NormalBP leaf (WDB1) + volume (WVN1) wires for welvet_dense_fwd.
int run_dense_normalbp(int bench_iters);
int run_volume_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// RMSNorm NormalBP leaf (WRB1) + volume (WRV1).
int run_rmsnorm_normalbp(int bench_iters);
int run_volume_rmsnorm_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// LayerNorm NormalBP leaf (WLB1) + volume (WLV1).
int run_layernorm_normalbp(int bench_iters);
int run_volume_layernorm_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// Softmax NormalBP leaf (WXB1) + volume (WXV1).
int run_softmax_normalbp(int bench_iters);
int run_volume_softmax_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// SwiGLU NormalBP leaf (WSB1) + volume (WSV1).
int run_swiglu_normalbp(int bench_iters);
int run_volume_swiglu_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// Embedding NormalBP leaf (WEB1) + volume (WEV1).
int run_embedding_normalbp(int bench_iters);
int run_volume_embedding_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// MHA NormalBP leaf (WHB1) + volume (WHV1).
int run_mha_normalbp(int bench_iters);
int run_volume_mha_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// CNN NormalBP leaf (C*B1) + volume (C*V1).
int run_cnn1_normalbp(int bench_iters);
int run_cnn2_normalbp(int bench_iters);
int run_cnn3_normalbp(int bench_iters);
int run_volume_cnn1_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_volume_cnn2_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_volume_cnn3_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
// ConvT NormalBP leaf (T*B1) + volume (T*V1).
int run_convt1_normalbp(int bench_iters);
int run_convt2_normalbp(int bench_iters);
int run_convt3_normalbp(int bench_iters);
int run_volume_convt1_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_volume_convt2_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_volume_convt3_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// RNN NormalBP leaf (RNB1) + volume (RNV1).
int run_rnn_normalbp(int bench_iters);
int run_volume_rnn_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// LSTM NormalBP leaf (LSB1) + volume (LSV1).
int run_lstm_normalbp(int bench_iters);
int run_volume_lstm_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// Sequential NormalBP leaf (SQB1) + volume (SQV1).
int run_sequential_normalbp(int bench_iters);
int run_volume_sequential_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// Residual NormalBP leaf (REB1) + volume (REV1).
int run_residual_normalbp(int bench_iters);
int run_volume_residual_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// Stack NestedStack-of-Dense NormalBP leaf (STKB) + volume (STKV).
int run_stack_normalbp(int bench_iters);
int run_volume_stack_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);

// Line StepBP (STB1→STBO) + Dense mesh-clock MeshBP (MBP1→MBPO)
// + heterogeneous mesh-clock FamilyBP (MGB1→MGBO).
#include "welvet/runtime/line.hpp"
int run_tween_leaf(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::Tween);
int run_stepbp(int bench_iters,
               welvet::runtime::line::Credit credit = welvet::runtime::line::Credit::FamilyBP,
               const char* response = "STBO");
int run_meshbp(int bench_iters);
int run_meshtween(int bench_iters);
int run_meshtweenchain(int bench_iters);
int run_mesh_generic_bp(int bench_iters, bool tween = false, const char* out_magic = nullptr,
                        int proxy_volume = 0, int mesh_family = 0);

int run_mamba_normalbp(int bench_iters);
int run_volume_mamba_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_metacognition_normalbp(int bench_iters);
int run_volume_metacognition_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_gdn_normalbp(int bench_iters);
int run_volume_gdn_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_kmeans_normalbp(int bench_iters);
int run_volume_kmeans_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_parallel_normalbp(int bench_iters);
int run_volume_parallel_normalbp(int bench_iters, VolumeTrainMode mode = VolumeTrainMode::NormalBP);
int run_parallel_poly_train(int bench_iters);
int run_normalbp_child(int bench_iters);
int run_cameral(int bench_iters); // CMB1 -> CMBO
