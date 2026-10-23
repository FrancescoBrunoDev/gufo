#include "src/models/qwen/forward.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

void TestEmbeddingLookup() {
  const std::size_t hidden_size = 4;
  const std::vector<float> table = {
      1.0F, 2.0F, 3.0F, 4.0F,  // token 0
      5.0F, 6.0F, 7.0F, 8.0F,  // token 1
  };
  std::vector<float> hidden(hidden_size, 0.0F);

  const gufo::models::QwenTensorRef ref = {.data = table.data(),
                                           .type = gufo::core::GgmlType::kF32,
                                           .num_elements = table.size()};
  gufo::models::ForwardEmbedding(1, ref, hidden_size, hidden);
  assert(hidden[0] == 5.0F);
  assert(hidden[1] == 6.0F);
  assert(hidden[2] == 7.0F);
  assert(hidden[3] == 8.0F);
}

// The recurrent cache mirrors the HIP arena: one slot per DeltaNet layer. Trunk
// layer 4 of a four-interval stack is the second DeltaNet layer, so it must
// write slot 1 and leave slot 4 (and every other slot) untouched.
void TestSsmCacheUsesTheSsmSlot() {
  gufo::core::ModelConfig config;
  config.num_layers = 8;
  config.hidden_size = 8;
  config.intermediate_size = 16;
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 4;
  config.vocab_size = 16;
  config.full_attention_interval = 4;
  config.ssm_conv_kernel = 4;
  config.ssm_state_size = 4;
  config.ssm_group_count = 2;
  config.ssm_time_step_rank = 2;
  config.ssm_inner_size = 8;
  config.rotary_dim = 2;

  const auto scratch = [](std::size_t size) {
    return std::vector<float>(size, 0.5F);
  };
  std::vector<float> qkv_weights(config.SsmQkvSize() * config.hidden_size,
                                 0.1F);
  std::vector<float> gate_weights(config.ssm_inner_size * config.hidden_size,
                                  0.1F);
  std::vector<float> alpha_weights(
      config.ssm_time_step_rank * config.hidden_size, 0.1F);
  std::vector<float> beta_weights(alpha_weights.size(), 0.1F);
  std::vector<float> out_weights(config.hidden_size * config.ssm_inner_size,
                                 0.1F);
  std::vector<float> conv_weights(config.SsmQkvSize() * config.ssm_conv_kernel,
                                  0.1F);
  std::vector<float> decay(config.ssm_time_step_rank, -0.05F);
  std::vector<float> dt(config.ssm_time_step_rank, 0.1F);
  std::vector<float> norm(config.SsmValueSize(), 1.0F);
  const auto ref = [](const std::vector<float>& values) {
    return gufo::models::QwenTensorRef{.data = values.data(),
                                       .type = gufo::core::GgmlType::kF32,
                                       .num_elements = values.size()};
  };

  gufo::models::QwenLayerWeights layer;
  layer.attn_qkv = ref(qkv_weights);
  layer.attn_gate = ref(gate_weights);
  layer.ssm_alpha = ref(alpha_weights);
  layer.ssm_beta = ref(beta_weights);
  layer.ssm_out = ref(out_weights);
  layer.ssm_conv1d = ref(conv_weights);
  layer.ssm_a = ref(decay);
  layer.ssm_dt = ref(dt);
  layer.ssm_norm = ref(norm);

  gufo::models::QwenSsmCache cache(
      config.SsmLayerCount(), config.SsmQkvSize(), config.ssm_conv_kernel,
      config.ssm_time_step_rank, config.ssm_state_size, config.SsmValueSize());
  std::vector<float> x(config.hidden_size, 1.0F);
  std::vector<float> qkv_scratch(config.SsmQkvSize());
  std::vector<float> gate_scratch(config.ssm_inner_size);
  std::vector<float> out_buf(config.ssm_inner_size);
  std::vector<float> out(config.hidden_size);

  gufo::models::ForwardSSM(x, layer, config, cache, 4, qkv_scratch,
                           gate_scratch, out_buf, out);

  const auto conv_of = [&](std::uint32_t slot) {
    const auto span = cache.GetConvState(slot);
    return std::accumulate(span.begin(), span.end(), 0.0F);
  };
  assert(std::abs(conv_of(config.SsmLayerIndex(4)) - 0.0F) > 1e-6F);
  assert(conv_of(4) == 0.0F);
  assert(conv_of(0) == 0.0F);
}

void TestQ8_0EmbeddingLookup() {
  constexpr std::size_t hidden_size = 32;
  std::vector<gufo::quant::block_q8_0> table(2);
  for (std::size_t token = 0; token < table.size(); ++token) {
    table[token].d = 0x3C00;  // fp16 1.0
    for (std::size_t i = 0; i < hidden_size; ++i) {
      table[token].qs[i] = static_cast<std::int8_t>(token * 32 + i);
    }
  }

  std::vector<float> hidden(hidden_size, 0.0F);
  const gufo::models::QwenTensorRef ref = {
      .data = table.data(),
      .type = gufo::core::GgmlType::kQ8_0,
      .num_elements = table.size() * hidden_size};
  gufo::models::ForwardEmbedding(1, ref, hidden_size, hidden);

  for (std::size_t i = 0; i < hidden_size; ++i) {
    assert(hidden[i] == static_cast<float>(32 + i));
  }
}

void TestRoPEPreservation() {
  const std::uint32_t num_heads = 2;
  const std::uint32_t num_kv_heads = 1;
  const std::uint32_t head_dim = 4;
  std::vector<float> q = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F};
  std::vector<float> k = {1.0F, 1.0F, 1.0F, 1.0F};

  const float norm_before = std::sqrt(q[0] * q[0] + q[2] * q[2]);
  gufo::models::ForwardRoPE(q, k, num_heads, num_kv_heads, head_dim, head_dim,
                            5, 10000.0F);
  const float norm_after = std::sqrt(q[0] * q[0] + q[2] * q[2]);

  assert(std::abs(norm_before - norm_after) < 1e-5F);
}

void TestSwiGLUFFN() {
  const std::size_t hidden_size = 2;
  const std::size_t intermediate_size = 2;
  const std::vector<float> x = {1.0F, 2.0F};
  const std::vector<float> gate_w = {1.0F, 0.0F, 0.0F, 1.0F};  // identity
  const std::vector<float> up_w = {1.0F, 0.0F, 0.0F, 1.0F};    // identity
  const std::vector<float> down_w = {1.0F, 0.0F, 0.0F, 1.0F};  // identity

  const gufo::models::QwenTensorRef gate_ref = {
      .data = gate_w.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = gate_w.size()};
  const gufo::models::QwenTensorRef up_ref = {
      .data = up_w.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = up_w.size()};
  const gufo::models::QwenTensorRef down_ref = {
      .data = down_w.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = down_w.size()};

  std::vector<float> gate_sc(intermediate_size);
  std::vector<float> up_sc(intermediate_size);
  std::vector<float> act_sc(intermediate_size);
  std::vector<float> ffn_out(hidden_size);

  gufo::models::ForwardFFN(x, gate_ref, up_ref, down_ref, hidden_size,
                           intermediate_size, gate_sc, up_sc, act_sc, ffn_out);

  // gate = [1.0, 2.0], up = [1.0, 2.0]
  // silu(1.0) * 1.0 = (1 / (1 + exp(-1))) * 1.0 ~= 0.731058
  assert(ffn_out[0] > 0.73F && ffn_out[0] < 0.74F);
}

void TestGreedyArgmax() {
  const std::vector<float> logits = {0.1F, -2.5F, 14.8F, 3.2F, 12.0F};
  const auto best = gufo::models::GreedyArgmax(logits);
  assert(best == 2);
}

void TestAttentionWithKvCache() {
  const std::uint32_t num_heads = 2;
  const std::uint32_t num_kv_heads = 2;
  const std::uint32_t head_dim = 4;
  gufo::models::QwenKvCache kv_cache(1, num_kv_heads, 16, head_dim);

  std::vector<float> q = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
  std::vector<float> k = {1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F};
  std::vector<float> v = {2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F};
  std::vector<float> scores(16, 0.0F);
  std::vector<float> out(8, 0.0F);

  gufo::models::ForwardAttention(q, k, v, {}, {}, kv_cache, 0, 0, num_heads,
                                 num_kv_heads, head_dim, 8, scores, out);

  // Since pos=0, softmax weight is 1.0, out should equal value vectors
  assert(std::abs(out[0] - 2.0F) < 1e-5F);
  assert(std::abs(out[1] - 3.0F) < 1e-5F);
  assert(std::abs(out[4] - 6.0F) < 1e-5F);
}

int main() {
  TestEmbeddingLookup();
  TestQ8_0EmbeddingLookup();
  TestRoPEPreservation();
  TestSwiGLUFFN();
  TestGreedyArgmax();
  TestAttentionWithKvCache();
  TestSsmCacheUsesTheSsmSlot();
  std::cout << "All Qwen forward tests passed.\n";
  return 0;
}
