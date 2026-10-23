#include "src/models/qwen/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

void ExpectNear(float actual, float expected, float tolerance,
                std::string_view msg) {
  if (!(std::fabs(actual - expected) <= tolerance)) {
    std::cerr << "Assertion failed: " << msg << " (actual " << actual
              << ", expected " << expected << ")\n";
    std::exit(1);
  }
}

/// Independent reference: y = A * x with A row-major [rows, cols].
std::vector<float> MatVec(const std::vector<float>& a,
                          const std::vector<float>& x, std::size_t rows,
                          std::size_t cols) {
  std::vector<float> y(rows, 0.0F);
  for (std::size_t r = 0; r < rows; ++r) {
    float sum = 0.0F;
    for (std::size_t c = 0; c < cols; ++c) {
      sum += a[r * cols + c] * x[c];
    }
    y[r] = sum;
  }
  return y;
}

std::vector<float> SiluMul(const std::vector<float>& gate,
                           const std::vector<float>& up) {
  std::vector<float> out(gate.size());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    out[i] = (gate[i] / (1.0F + std::exp(-gate[i]))) * up[i];
  }
  return out;
}

/// Independent routing reference: full softmax, top-k, renormalized.
void ReferenceRoute(const std::vector<float>& router,
                    const std::vector<float>& x, std::size_t experts,
                    std::size_t hidden, std::size_t used,
                    std::vector<std::size_t>* indices,
                    std::vector<float>* weights) {
  const auto logits = MatVec(router, x, experts, hidden);
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(experts);
  float total = 0.0F;
  for (std::size_t e = 0; e < experts; ++e) {
    probs[e] = std::exp(logits[e] - max_logit);
    total += probs[e];
  }
  for (auto& p : probs) {
    p /= total;
  }
  indices->clear();
  weights->clear();
  std::vector<bool> taken(experts, false);
  for (std::size_t slot = 0; slot < used; ++slot) {
    std::size_t best = 0;
    float best_value = -1.0F;
    for (std::size_t e = 0; e < experts; ++e) {
      if (!taken[e] && probs[e] > best_value) {
        best_value = probs[e];
        best = e;
      }
    }
    taken[best] = true;
    indices->push_back(best);
    weights->push_back(best_value);
  }
  float selected_total = 0.0F;
  for (const float w : *weights) {
    selected_total += w;
  }
  for (auto& w : *weights) {
    w /= selected_total;
  }
}

/// Deterministic pseudo-random weights, so the fixture is readable and stable.
std::vector<float> Fill(std::size_t count, std::uint32_t seed) {
  std::vector<float> values(count);
  std::uint32_t state = seed;
  for (auto& v : values) {
    state = state * 1664525U + 1013904223U;
    v = (static_cast<float>(state >> 8 & 0xFFFFU) / 65535.0F - 0.5F) * 2.0F;
  }
  return values;
}

gufo::models::QwenTensorRef MakeRef(std::vector<float>& data) {
  return gufo::models::QwenTensorRef{
      .data = data.data(),
      .type = gufo::core::GgmlType::kF32,
      .num_elements = data.size(),
      .available_bytes = data.size() * sizeof(float)};
}

void TestMoeRouteMatchesReference() {
  constexpr std::size_t kHidden = 6;
  constexpr std::size_t kExperts = 5;
  constexpr std::size_t kUsed = 3;

  auto router_data = Fill(kExperts * kHidden, 7);
  auto x_data = Fill(kHidden, 11);
  auto router = MakeRef(router_data);
  const std::span<const float> x(x_data);

  std::vector<float> logits(kExperts);
  gufo::models::qwen::MoeSelection selection;
  gufo::models::qwen::MoeRoute(router, x, static_cast<std::uint32_t>(kExperts),
                               static_cast<std::uint32_t>(kUsed),
                               std::span<float>(logits), &selection);

  std::vector<std::size_t> ref_indices;
  std::vector<float> ref_weights;
  ReferenceRoute(router_data, x_data, kExperts, kHidden, kUsed, &ref_indices,
                 &ref_weights);

  Expect(selection.count == kUsed, "route selects used_count experts");
  for (std::size_t slot = 0; slot < kUsed; ++slot) {
    Expect(selection.experts[slot] == ref_indices[slot],
           "route selects the same expert order as the reference");
    ExpectNear(selection.weights[slot], ref_weights[slot], 1e-6F,
               "renormalized top-k weight matches the reference");
  }
  float weight_sum = 0.0F;
  for (std::uint32_t slot = 0; slot < selection.count; ++slot) {
    weight_sum += selection.weights[slot];
  }
  ExpectNear(weight_sum, 1.0F, 1e-6F, "norm_topk_prob weights sum to one");
}

void TestMoeRouteCountsAfterSoftmax() {
  // A masked-out (equal) logits field must still yield `used` experts, and the
  // selection must never repeat an expert.
  constexpr std::size_t kHidden = 4;
  constexpr std::size_t kExperts = 6;
  constexpr std::size_t kUsed = 4;
  std::vector<float> router_data(kExperts * kHidden, 0.0F);
  std::vector<float> x_data(kHidden, 1.0F);
  auto router = MakeRef(router_data);
  std::vector<float> logits(kExperts);
  gufo::models::qwen::MoeSelection selection;
  gufo::models::qwen::MoeRoute(router, std::span<const float>(x_data),
                               static_cast<std::uint32_t>(kExperts),
                               static_cast<std::uint32_t>(kUsed),
                               std::span<float>(logits), &selection);

  Expect(selection.count == kUsed, "ties still select used_count experts");
  for (std::size_t a = 0; a < kUsed; ++a) {
    for (std::size_t b = a + 1; b < kUsed; ++b) {
      Expect(selection.experts[a] != selection.experts[b],
             "a selected expert is never repeated");
    }
  }
  ExpectNear(selection.weights[0], 1.0F / static_cast<float>(kUsed), 1e-6F,
             "ties split the renormalized weight evenly");
}

void TestMoeFfnMatchesReference() {
  constexpr std::size_t kHidden = 5;
  constexpr std::size_t kExperts = 4;
  constexpr std::size_t kUsed = 2;
  constexpr std::size_t kExpertFfn = 3;

  gufo::core::ModelConfig config;
  config.hidden_size = kHidden;
  config.expert_count = kExperts;
  config.expert_used_count = kUsed;
  config.expert_feed_forward_length = kExpertFfn;
  config.shared_expert_feed_forward_length = kExpertFfn;
  config.intermediate_size = kExpertFfn;

  auto router_data = Fill(kExperts * kHidden, 3);
  auto gate_exps = Fill(kExperts * kExpertFfn * kHidden, 5);
  auto up_exps = Fill(kExperts * kExpertFfn * kHidden, 9);
  auto down_exps = Fill(kExperts * kHidden * kExpertFfn, 13);
  auto shexp_gate_in = Fill(kHidden, 17);
  auto shexp_gate = Fill(kExpertFfn * kHidden, 19);
  auto shexp_up = Fill(kExpertFfn * kHidden, 23);
  auto shexp_down = Fill(kHidden * kExpertFfn, 29);
  auto x_data = Fill(kHidden, 31);

  gufo::models::QwenLayerWeights layer;
  layer.ffn_gate_inp = MakeRef(router_data);
  layer.ffn_gate_exps = MakeRef(gate_exps);
  layer.ffn_up_exps = MakeRef(up_exps);
  layer.ffn_down_exps = MakeRef(down_exps);
  layer.ffn_gate_inp_shexp = MakeRef(shexp_gate_in);
  layer.ffn_gate_shexp = MakeRef(shexp_gate);
  layer.ffn_up_shexp = MakeRef(shexp_up);
  layer.ffn_down_shexp = MakeRef(shexp_down);

  std::vector<float> logits(kExperts);
  std::vector<float> ffn_gate(kExpertFfn);
  std::vector<float> ffn_up(kExpertFfn);
  std::vector<float> act(kExpertFfn);
  std::vector<float> expert_out(kHidden);
  std::vector<float> out(kHidden, 0.0F);

  gufo::models::qwen::MoeFfnForward(
      gufo::models::qwen::CpuModuleContext{}, layer, config,
      std::span<const float>(x_data), std::span<float>(logits),
      std::span<float>(ffn_gate), std::span<float>(ffn_up),
      std::span<float>(act), std::span<float>(expert_out),
      std::span<float>(out));

  // Independent reference assembled from plain loops.
  std::vector<std::size_t> indices;
  std::vector<float> weights;
  ReferenceRoute(router_data, x_data, kExperts, kHidden, kUsed, &indices,
                 &weights);
  std::vector<float> expected(kHidden, 0.0F);
  for (std::size_t slot = 0; slot < kUsed; ++slot) {
    const std::size_t e = indices[slot];
    const std::vector<float> gate_w(
        gate_exps.begin() + e * kExpertFfn * kHidden,
        gate_exps.begin() + (e + 1) * kExpertFfn * kHidden);
    const std::vector<float> up_w(
        up_exps.begin() + e * kExpertFfn * kHidden,
        up_exps.begin() + (e + 1) * kExpertFfn * kHidden);
    const std::vector<float> down_w(
        down_exps.begin() + e * kHidden * kExpertFfn,
        down_exps.begin() + (e + 1) * kHidden * kExpertFfn);
    const auto expert =
        MatVec(down_w,
               SiluMul(MatVec(gate_w, x_data, kExpertFfn, kHidden),
                       MatVec(up_w, x_data, kExpertFfn, kHidden)),
               kHidden, kExpertFfn);
    for (std::size_t i = 0; i < kHidden; ++i) {
      expected[i] += weights[slot] * expert[i];
    }
  }
  const auto shared =
      MatVec(shexp_down,
             SiluMul(MatVec(shexp_gate, x_data, kExpertFfn, kHidden),
                     MatVec(shexp_up, x_data, kExpertFfn, kHidden)),
             kHidden, kExpertFfn);
  const float shared_scale =
      1.0F / (1.0F + std::exp(-MatVec(shexp_gate_in, x_data, 1, kHidden)[0]));
  for (std::size_t i = 0; i < kHidden; ++i) {
    expected[i] += shared_scale * shared[i];
  }

  for (std::size_t i = 0; i < kHidden; ++i) {
    ExpectNear(out[i], expected[i], 1e-5F,
               "MoE FFN output matches the reference for every hidden index");
  }
}

}  // namespace

int main() {
  TestMoeRouteMatchesReference();
  TestMoeRouteCountsAfterSoftmax();
  TestMoeFfnMatchesReference();
  std::cout << "All Qwen MoE reference tests passed successfully!\n";
  return 0;
}
