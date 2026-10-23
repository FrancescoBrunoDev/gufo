// CPU-vs-GPU draft parity for the Ornith 1.5 122B artifact: the last block of
// the GGUF is a mixture-of-experts `nextn` block, so this exercises the packed
// expert GEMVs, the MoE router and the shared expert inside the MTP graph.
//
// GUFO_QWEN35MOE_MTP_MODEL (or argv[1]) points at the same shard set that
// serves the trunk; shard discovery picks up the siblings.

#include "src/models/qwen/hip/mtp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/executor.hpp"
#include "src/models/qwen/mtp_reference.hpp"

namespace {

constexpr int kSkipped = 77;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::vector<float> MakeTargetHidden(std::size_t size) {
  std::vector<float> hidden(size);
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    const auto centered = static_cast<int>(index % 257U) - 128;
    hidden[index] = static_cast<float>(centered) / 256.0F;
  }
  return hidden;
}

struct Comparison {
  double rmse{0.0};
  double cosine{0.0};
  float max_abs{0.0F};
};

Comparison Compare(std::span<const float> actual,
                   std::span<const float> expected) {
  Expect(actual.size() == expected.size(), "comparison size mismatch");
  double squared_error = 0.0;
  double actual_squared = 0.0;
  double expected_squared = 0.0;
  double dot = 0.0;
  float max_abs = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double lhs = actual[index];
    const double rhs = expected[index];
    const double difference = lhs - rhs;
    squared_error += difference * difference;
    actual_squared += lhs * lhs;
    expected_squared += rhs * rhs;
    dot += lhs * rhs;
    max_abs = std::max(max_abs, static_cast<float>(std::abs(difference)));
  }
  const double count = static_cast<double>(actual.size());
  return {
      .rmse = std::sqrt(squared_error / count),
      .cosine = dot / std::sqrt(actual_squared * expected_squared),
      .max_abs = max_abs,
  };
}

/// Bytes the H2D copy and the device residency avoid by keeping the packed
/// expert tensors quantized instead of expanding them to BF16.
std::size_t ExpertBf16BytesAvoided(
    const gufo::speculative::QwenMtpWeights& weights) {
  if (!weights.config.IsMoe()) {
    return 0;
  }
  const auto& layer = weights.layer;
  std::size_t saved = 0;
  for (const auto* tensor :
       {&layer.ffn_gate_exps, &layer.ffn_up_exps, &layer.ffn_down_exps}) {
    const std::size_t bf16_bytes = tensor->num_elements * sizeof(std::uint16_t);
    saved += bf16_bytes -
             gufo::quant::EncodedSizeBytes(tensor->type, tensor->num_elements);
  }
  return saved;
}

}  // namespace

int main(int argc, const char* const* argv) {
  try {
    const char* model_path =
        argc > 1 ? argv[1] : std::getenv("GUFO_QWEN35MOE_MTP_MODEL");
    if (model_path == nullptr) {
      std::cout << "qwen_mtp_moe_gpu_test: skipped "
                   "(pass the GGUF path or set GUFO_QWEN35MOE_MTP_MODEL)\n";
      return kSkipped;
    }

    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> reader(
        gufo::core::GgufReader::OpenFile(model_path, &error));
    Expect(reader != nullptr, error);

    auto reference =
        gufo::speculative::QwenMtpReference::Create(reader, 8, &error);
    Expect(reference != nullptr, "CPU MTP reference: " + error);
    Expect(reference->GetWeights().config.IsMoe(),
           "the artifact declares a mixture-of-experts draft block");

    auto target_model = gufo::hip::QwenGpuModel::CreateFromGguf(reader, &error);
    Expect(target_model != nullptr, "target GPU model: " + error);

    const auto setup_start = std::chrono::steady_clock::now();
    auto mtp_model =
        gufo::hip::QwenMtpGpuModel::Create(reader, target_model, &error);
    Expect(mtp_model != nullptr, "GPU MTP model: " + error);
    auto executor = gufo::hip::QwenMtpGpuExecutor::Create(mtp_model, 8, &error);
    Expect(executor != nullptr, "GPU MTP executor: " + error);
    const double setup_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      setup_start)
            .count();
    const std::size_t expert_bytes_avoided =
        ExpertBf16BytesAvoided(mtp_model->GetWeights());

    const auto target_hidden =
        MakeTargetHidden(reference->GetWeights().config.hidden_size);
    const auto reference_hidden =
        reference->ForwardHidden(17, target_hidden, 0);
    Expect(!reference_hidden.empty(), "CPU draft step");
    const std::vector<float> reference_copy(reference_hidden.begin(),
                                            reference_hidden.end());

    const auto token =
        executor->ForwardTargetHidden(17, target_hidden, 0, true);
    const auto gpu_hidden = executor->CopyLastHidden();
    const auto hidden_comparison = Compare(gpu_hidden, reference_copy);
    const auto logits = executor->CopyLastLogits();
    const std::size_t vocab = reference->GetWeights().config.vocab_size;
    const std::array<std::uint32_t, 6> kSelectedTokens{
        0, 1, 17, 42, 198, static_cast<std::uint32_t>(vocab - 1)};
    float max_logit_error = 0.0F;
    for (const auto selected : kSelectedTokens) {
      const float expected = reference->ComputeLogit(selected);
      max_logit_error =
          std::max(max_logit_error, std::abs(logits[selected] - expected));
    }
    Expect(max_logit_error < 2.0F, "selected logit error");
    Expect(std::isfinite(hidden_comparison.rmse), "hidden RMSE is finite");
    // The draft block routes 256 experts, so one near-tie can flip an expert
    // and land the copied-hidden cosine near 0.998 (the dense 27B sits above
    // 0.999). Acceptance is still decided by the target's own logits, so the
    // bound keeps the draft honest without demanding dense-level agreement.
    Expect(hidden_comparison.cosine > 0.995, "hidden cosine");
    Expect(hidden_comparison.rmse < 0.15, "hidden RMSE");
    Expect(hidden_comparison.max_abs < 1.0F, "hidden max absolute error");

    const std::vector<float> first_gpu_hidden(gpu_hidden.begin(),
                                              gpu_hidden.end());
    executor->Reset();
    const auto repeated_token =
        executor->ForwardTargetHidden(17, target_hidden, 0, true);
    const auto repeated_hidden = executor->CopyLastHidden();
    Expect(repeated_token == token, "reset token determinism");
    Expect(Compare(repeated_hidden, first_gpu_hidden).rmse == 0.0,
           "reset hidden determinism");

    gufo::hip::QwenMtpGpuDraftConfig draft_config{
        .max_context = 8,
        .max_draft_tokens = 2,
    };
    auto draft_backend = gufo::hip::QwenMtpGpuDraftBackend::Create(
        mtp_model, draft_config, &error);
    Expect(draft_backend != nullptr, "GPU MTP draft backend: " + error);
    const std::array<gufo::tokenization::TokenId, 2> prompt_tokens{5, 17};
    std::vector<float> prompt_hidden(target_hidden.size() * 2);
    std::ranges::copy(target_hidden, prompt_hidden.begin());
    for (std::size_t index = 0; index < target_hidden.size(); ++index) {
      prompt_hidden[target_hidden.size() + index] =
          target_hidden[index] * 0.75F;
    }
    Expect(draft_backend->PrimeTargetContext({
               .prompt_tokens = prompt_tokens,
               .prompt_hidden_states = prompt_hidden,
               .hidden_size = target_hidden.size(),
               .first_token = token,
           }),
           draft_backend->GetLastError());
    std::vector<gufo::tokenization::TokenId> sequence{prompt_tokens[0],
                                                      prompt_tokens[1], token};
    const auto proposal = draft_backend->Propose(sequence, 2, 2);
    Expect(proposal.tokens.size() == 2, "GPU MTP proposal size");
    for (const auto proposed : proposal.tokens) {
      Expect(proposed < vocab, "proposed draft token is inside the vocabulary");
    }

    std::cout << "qwen_mtp_moe_gpu_test: token=" << token
              << " packed_bytes=" << mtp_model->GetPackedWeightBytes()
              << " expert_bf16_bytes_avoided=" << expert_bytes_avoided
              << " pack_seconds=" << mtp_model->GetPackTimeSeconds()
              << " setup_seconds=" << setup_seconds
              << " hidden_rmse=" << hidden_comparison.rmse
              << " hidden_cosine=" << hidden_comparison.cosine
              << " hidden_max_abs=" << hidden_comparison.max_abs
              << " selected_logit_max_abs=" << max_logit_error << '\n';
    return 0;
  } catch (const std::exception& exception) {
    std::cerr << "qwen_mtp_moe_gpu_test failed: " << exception.what() << '\n';
    return 1;
  }
}
