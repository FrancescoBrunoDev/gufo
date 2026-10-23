// The Ornith 1.5 122B artifact puts a full mixture-of-experts block behind the
// trunk: block 48 of 49 is the `nextn` draft block. These tests pin the MTP
// loader to the trunk layer count (not the 27B's literal 64), to the MoE FFN
// contract, and to the promise that a draft block never touches the trunk.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/mtp_reference.hpp"
#include "src/models/qwen/state.hpp"

namespace {

using gufo::core::GgmlType;

void Expect(bool condition, std::string_view msg) {
  if (!condition) {
    std::cerr << "Assertion failed: " << msg << "\n";
    std::exit(1);
  }
}

constexpr std::size_t kHidden = 64;
constexpr std::uint32_t kTrunkLayers = 48;
constexpr std::size_t kExperts = 8;
constexpr std::uint32_t kExpertUsed = 2;
constexpr std::size_t kExpertFfn = 32;
constexpr std::uint32_t kHeads = 2;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::size_t kHeadDim = 32;
constexpr std::size_t kAttention = kHeads * kHeadDim;
constexpr std::size_t kKv = kKvHeads * kHeadDim;
constexpr std::size_t kSsmInner = 128;
constexpr std::size_t kSsmGroup = 1;
constexpr std::size_t kSsmState = 128;
constexpr std::size_t kSsmRank = 1;
constexpr std::size_t kSsmQkv = (2 * kSsmGroup * kSsmState) + kSsmInner;
constexpr std::size_t kVocab = 128;
constexpr std::uint32_t kContext = 32;

/// Builds a GGUF image with real F32 payload, enough for the model loaders to
/// bind every tensor of a synthetic 48-trunk-layer + MTP-block MoE artifact.
class GgufWriter final {
public:
  GgufWriter() {
    AppendBytes("GGUF", 4);
    AppendPod<std::uint32_t>(3);
    tensor_count_pos_ = buffer_.size();
    AppendPod<std::uint64_t>(0);
    metadata_count_pos_ = buffer_.size();
    AppendPod<std::uint64_t>(0);
  }

  void AddMetadataString(std::string_view key, std::string_view value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kString));
    AppendString(value);
    ++metadata_count_;
  }

  void AddMetadataUint32(std::string_view key, std::uint32_t value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kUint32));
    AppendPod(value);
    ++metadata_count_;
  }

  void AddMetadataFloat32(std::string_view key, float value) {
    AppendString(key);
    AppendPod(static_cast<std::uint32_t>(gufo::core::GgufValueType::kFloat32));
    AppendPod(value);
    ++metadata_count_;
  }

  /// Random-weight matrix, `scale` bounds the magnitude.
  void AddMatrix(std::string_view name,
                 const std::vector<std::uint64_t>& dimensions, float scale) {
    std::size_t elements = 1;
    for (const auto dimension : dimensions) {
      elements *= dimension;
    }
    std::vector<float> values(elements);
    for (auto& value : values) {
      value = NextRandom() * scale;
    }
    AddF32Tensor(name, dimensions, values);
  }

  /// Norm and SSM parameter tensor filled with `value` everywhere.
  void AddFilled(std::string_view name,
                 const std::vector<std::uint64_t>& dimensions, float value) {
    std::size_t elements = 1;
    for (const auto dimension : dimensions) {
      elements *= dimension;
    }
    AddF32Tensor(name, dimensions, std::vector<float>(elements, value));
  }

  std::vector<std::uint8_t> Finish() {
    std::memcpy(buffer_.data() + metadata_count_pos_, &metadata_count_,
                sizeof(metadata_count_));
    const std::uint64_t tensor_count = tensors_.size();
    std::memcpy(buffer_.data() + tensor_count_pos_, &tensor_count,
                sizeof(tensor_count));
    for (const auto& tensor : tensors_) {
      AppendString(tensor.name);
      AppendPod(static_cast<std::uint32_t>(tensor.dimensions.size()));
      for (const auto dimension : tensor.dimensions) {
        AppendPod(dimension);
      }
      AppendPod(static_cast<std::uint32_t>(GgmlType::kF32));
      AppendPod(tensor.offset);
    }
    PadToAlignment();
    buffer_.insert(buffer_.end(), payload_.begin(), payload_.end());
    return std::move(buffer_);
  }

private:
  struct TensorRecord {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    std::uint64_t offset{0};
  };

  void AddF32Tensor(std::string_view name,
                    const std::vector<std::uint64_t>& dimensions,
                    const std::vector<float>& values) {
    PadPayloadToAlignment();
    TensorRecord record{.name = std::string(name),
                        .dimensions = dimensions,
                        .offset = static_cast<std::uint64_t>(payload_.size())};
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
    payload_.insert(payload_.end(), bytes, bytes + (values.size() * 4));
    tensors_.push_back(std::move(record));
  }

  void PadPayloadToAlignment() {
    const std::size_t remainder = payload_.size() % 32;
    if (remainder != 0) {
      payload_.resize(payload_.size() + (32 - remainder), 0);
    }
  }

  void PadToAlignment() {
    const std::size_t remainder = buffer_.size() % 32;
    if (remainder != 0) {
      buffer_.resize(buffer_.size() + (32 - remainder), 0);
    }
  }

  float NextRandom() {
    state_ = (state_ * 1664525U) + 1013904223U;
    return (static_cast<float>(state_ >> 8U) / 16777216.0F) - 0.5F;
  }

  void AppendBytes(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    buffer_.insert(buffer_.end(), bytes, bytes + size);
  }

  template<typename T>
  void AppendPod(T value) {
    AppendBytes(&value, sizeof(T));
  }

  void AppendString(std::string_view text) {
    AppendPod(static_cast<std::uint64_t>(text.size()));
    AppendBytes(text.data(), text.size());
  }

  std::vector<std::uint8_t> buffer_;
  std::vector<std::uint8_t> payload_;
  std::vector<TensorRecord> tensors_;
  std::size_t tensor_count_pos_{0};
  std::size_t metadata_count_pos_{0};
  std::uint64_t metadata_count_{0};
  std::uint32_t state_{0xC0FFEEU};
};

void AddTrunkLayer(GgufWriter& writer, std::uint32_t index) {
  const std::string prefix = "blk." + std::to_string(index) + ".";
  writer.AddFilled(prefix + "attn_norm.weight", {kHidden}, 1.0F);
  writer.AddFilled(prefix + "post_attention_norm.weight", {kHidden}, 1.0F);
  if ((index + 1) % 4 == 0) {
    writer.AddMatrix(prefix + "attn_q.weight", {kHidden, 2 * kAttention}, 0.1F);
    writer.AddMatrix(prefix + "attn_k.weight", {kHidden, kKv}, 0.1F);
    writer.AddMatrix(prefix + "attn_v.weight", {kHidden, kKv}, 0.1F);
    writer.AddMatrix(prefix + "attn_output.weight", {kAttention, kHidden},
                     0.1F);
    writer.AddFilled(prefix + "attn_q_norm.weight", {kHeadDim}, 1.0F);
    writer.AddFilled(prefix + "attn_k_norm.weight", {kHeadDim}, 1.0F);
  } else {
    writer.AddMatrix(prefix + "attn_qkv.weight", {kHidden, kSsmQkv}, 0.1F);
    writer.AddMatrix(prefix + "attn_gate.weight", {kHidden, kSsmInner}, 0.1F);
    writer.AddFilled(prefix + "ssm_a", {kSsmRank}, -1.0F);
    writer.AddMatrix(prefix + "ssm_conv1d.weight", {4, kSsmQkv}, 0.1F);
    writer.AddFilled(prefix + "ssm_dt.bias", {kSsmRank}, 0.01F);
    writer.AddMatrix(prefix + "ssm_alpha.weight", {kHidden, kSsmRank}, 0.1F);
    writer.AddMatrix(prefix + "ssm_beta.weight", {kHidden, kSsmRank}, 0.1F);
    writer.AddFilled(prefix + "ssm_norm.weight", {kSsmInner / kSsmRank}, 1.0F);
    writer.AddMatrix(prefix + "ssm_out.weight", {kSsmInner, kHidden}, 0.1F);
  }
  writer.AddMatrix(prefix + "ffn_gate_inp.weight", {kHidden, kExperts}, 0.1F);
  writer.AddMatrix(prefix + "ffn_gate_exps.weight",
                   {kHidden, kExpertFfn, kExperts}, 0.1F);
  writer.AddMatrix(prefix + "ffn_up_exps.weight",
                   {kHidden, kExpertFfn, kExperts}, 0.1F);
  writer.AddMatrix(prefix + "ffn_down_exps.weight",
                   {kExpertFfn, kHidden, kExperts}, 0.1F);
  writer.AddFilled(prefix + "ffn_gate_inp_shexp.weight", {kHidden}, 0.1F);
  writer.AddMatrix(prefix + "ffn_gate_shexp.weight", {kHidden, kExpertFfn},
                   0.1F);
  writer.AddMatrix(prefix + "ffn_up_shexp.weight", {kHidden, kExpertFfn}, 0.1F);
  writer.AddMatrix(prefix + "ffn_down_shexp.weight", {kExpertFfn, kHidden},
                   0.1F);
}

void AddMtpBlock(GgufWriter& writer) {
  const std::string prefix = "blk." + std::to_string(kTrunkLayers) + ".";
  writer.AddFilled(prefix + "nextn.enorm.weight", {kHidden}, 1.0F);
  writer.AddFilled(prefix + "nextn.hnorm.weight", {kHidden}, 1.0F);
  writer.AddMatrix(prefix + "nextn.eh_proj.weight", {2 * kHidden, kHidden},
                   0.1F);
  writer.AddFilled(prefix + "nextn.shared_head_norm.weight", {kHidden}, 1.0F);
  writer.AddFilled(prefix + "attn_norm.weight", {kHidden}, 1.0F);
  writer.AddFilled(prefix + "post_attention_norm.weight", {kHidden}, 1.0F);
  writer.AddMatrix(prefix + "attn_q.weight", {kHidden, 2 * kAttention}, 0.1F);
  writer.AddMatrix(prefix + "attn_k.weight", {kHidden, kKv}, 0.1F);
  writer.AddMatrix(prefix + "attn_v.weight", {kHidden, kKv}, 0.1F);
  writer.AddMatrix(prefix + "attn_output.weight", {kAttention, kHidden}, 0.1F);
  writer.AddFilled(prefix + "attn_q_norm.weight", {kHeadDim}, 1.0F);
  writer.AddFilled(prefix + "attn_k_norm.weight", {kHeadDim}, 1.0F);
  writer.AddMatrix(prefix + "ffn_gate_inp.weight", {kHidden, kExperts}, 0.1F);
  writer.AddMatrix(prefix + "ffn_gate_exps.weight",
                   {kHidden, kExpertFfn, kExperts}, 0.1F);
  writer.AddMatrix(prefix + "ffn_up_exps.weight",
                   {kHidden, kExpertFfn, kExperts}, 0.1F);
  writer.AddMatrix(prefix + "ffn_down_exps.weight",
                   {kExpertFfn, kHidden, kExperts}, 0.1F);
  writer.AddFilled(prefix + "ffn_gate_inp_shexp.weight", {kHidden}, 0.1F);
  writer.AddMatrix(prefix + "ffn_gate_shexp.weight", {kHidden, kExpertFfn},
                   0.1F);
  writer.AddMatrix(prefix + "ffn_up_shexp.weight", {kHidden, kExpertFfn}, 0.1F);
  writer.AddMatrix(prefix + "ffn_down_shexp.weight", {kExpertFfn, kHidden},
                   0.1F);
}

/// `declared_experts` is the metadata value; the tensors always carry
/// `kExperts` experts, so a different value must fail the MTP shape checks.
std::vector<std::uint8_t> BuildSyntheticMoeGguf(
    std::uint32_t declared_experts) {
  GgufWriter writer;
  writer.AddMetadataString("general.architecture", "qwen35moe");
  writer.AddMetadataString("general.name", "synthetic-ornith-122b-mtp");
  writer.AddMetadataUint32("qwen35moe.block_count", kTrunkLayers + 1);
  writer.AddMetadataUint32("qwen35moe.embedding_length", kHidden);
  writer.AddMetadataUint32("qwen35moe.attention.head_count", kHeads);
  writer.AddMetadataUint32("qwen35moe.attention.head_count_kv", kKvHeads);
  writer.AddMetadataUint32("qwen35moe.attention.key_length", kHeadDim);
  writer.AddMetadataUint32("qwen35moe.context_length", kContext);
  writer.AddMetadataUint32("qwen35moe.full_attention_interval", 4);
  writer.AddMetadataUint32("qwen35moe.nextn_predict_layers", 1);
  writer.AddMetadataUint32("qwen35moe.expert_count", declared_experts);
  writer.AddMetadataUint32("qwen35moe.expert_used_count", kExpertUsed);
  writer.AddMetadataUint32("qwen35moe.expert_feed_forward_length", kExpertFfn);
  writer.AddMetadataUint32("qwen35moe.expert_shared_feed_forward_length",
                           kExpertFfn);
  writer.AddMetadataUint32("qwen35moe.ssm.conv_kernel", 4);
  writer.AddMetadataUint32("qwen35moe.ssm.state_size", kSsmState);
  writer.AddMetadataUint32("qwen35moe.ssm.group_count", kSsmGroup);
  writer.AddMetadataUint32("qwen35moe.ssm.time_step_rank", kSsmRank);
  writer.AddMetadataUint32("qwen35moe.ssm.inner_size", kSsmInner);
  writer.AddMetadataUint32("qwen35moe.rope.dimension_count", kHeadDim);
  writer.AddMetadataFloat32("qwen35moe.rope.freq_base", 10000.0F);

  writer.AddMatrix("token_embd.weight", {kHidden, kVocab}, 0.1F);
  writer.AddFilled("output_norm.weight", {kHidden}, 1.0F);
  writer.AddMatrix("output.weight", {kHidden, kVocab}, 0.1F);
  for (std::uint32_t index = 0; index < kTrunkLayers; ++index) {
    AddTrunkLayer(writer, index);
  }
  AddMtpBlock(writer);
  return writer.Finish();
}

gufo::core::ModelConfig LoadConfig(const gufo::core::GgufReader& reader) {
  std::string error;
  const auto config = reader.ExtractModelConfig(&error);
  Expect(config.has_value(), "synthetic MoE config extracts: " + error);
  return *config;
}

void TestMtpMoeBindingBindsEveryNextnTensor() {
  const auto bytes = BuildSyntheticMoeGguf(kExperts);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Expect(reader != nullptr, "synthetic MoE reader opens: " + error);

  const auto config = LoadConfig(*reader);
  Expect(config.num_layers == kTrunkLayers,
         "the nextn block is excluded from the trunk layers");
  Expect(config.mtp_num_layers == 1, "one MTP layer");
  Expect(config.IsMoe(), "the synthetic artifact is a mixture of experts");
  Expect(config.intermediate_size == kExpertFfn,
         "the FFN width falls back to one expert's width");

  const auto weights =
      gufo::speculative::QwenMtpWeights::LoadFromGguf(*reader, &error);
  Expect(weights.has_value(), "MTP weights bind from blk." +
                                  std::to_string(kTrunkLayers) + ": " + error);

  const auto& layer = weights->layer;
  const std::vector<std::pair<const gufo::models::QwenTensorRef*, const char*>>
      tensors = {
          {&weights->embedding_norm, "nextn.enorm"},
          {&weights->hidden_norm, "nextn.hnorm"},
          {&weights->fusion_projection, "nextn.eh_proj"},
          {&weights->shared_head_norm, "nextn.shared_head_norm"},
          {&layer.attn_norm, "attn_norm"},
          {&layer.attn_q, "attn_q"},
          {&layer.attn_k, "attn_k"},
          {&layer.attn_v, "attn_v"},
          {&layer.attn_output, "attn_output"},
          {&layer.attn_q_norm, "attn_q_norm"},
          {&layer.attn_k_norm, "attn_k_norm"},
          {&layer.ffn_norm, "post_attention_norm"},
          {&layer.ffn_gate_inp, "ffn_gate_inp"},
          {&layer.ffn_gate_exps, "ffn_gate_exps"},
          {&layer.ffn_up_exps, "ffn_up_exps"},
          {&layer.ffn_down_exps, "ffn_down_exps"},
          {&layer.ffn_gate_inp_shexp, "ffn_gate_inp_shexp"},
          {&layer.ffn_gate_shexp, "ffn_gate_shexp"},
          {&layer.ffn_up_shexp, "ffn_up_shexp"},
          {&layer.ffn_down_shexp, "ffn_down_shexp"},
      };
  for (const auto& [tensor, name] : tensors) {
    Expect(!tensor->empty(), std::string("MTP tensor is bound: ") + name);
  }
  Expect(layer.is_full_attention, "the MTP block is a full-attention layer");
  Expect(layer.ffn_gate_exps.num_elements == kExperts * kExpertFfn * kHidden,
         "the packed expert width matches the declared experts");
  Expect(layer.ffn_down_exps.num_elements == kExperts * kExpertFfn * kHidden,
         "the packed down-projection width matches the declared experts");
}

void TestWrongExpertCountIsRejected() {
  const auto bytes = BuildSyntheticMoeGguf(kExperts / 2);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Expect(reader != nullptr, "wrong-expert-count reader opens: " + error);

  const auto weights =
      gufo::speculative::QwenMtpWeights::LoadFromGguf(*reader, &error);
  Expect(!weights.has_value(),
         "MTP loading rejects expert tensors whose width contradicts "
         "expert_count");
  Expect(error.find("shape mismatch") != std::string::npos,
         "the rejection names the shape mismatch: " + error);
}

void TestTrunkIsInvariantToTheDraftBlock() {
  const auto bytes = BuildSyntheticMoeGguf(kExperts);
  std::string error;
  const auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Expect(reader != nullptr, "trunk-invariance reader opens: " + error);

  const auto loaded =
      gufo::models::QwenModelWeights::LoadFromGguf(*reader, &error);
  Expect(loaded.has_value(),
         "trunk weights load with the MTP block present: " + error);
  Expect(loaded->layers.size() == kTrunkLayers,
         "the trunk keeps 48 layers with the draft block in the file");
  const auto* mtp_norm = reader->FindTensor(
      "blk." + std::to_string(kTrunkLayers) + ".post_attention_norm.weight");
  Expect(mtp_norm != nullptr, "the draft block's norm is in the file");
  Expect(loaded->layers.back().ffn_norm.data != mtp_norm->data,
         "the last trunk layer is blk.47, not the draft block");

  const auto run = [&](const gufo::models::QwenModelWeights& weights) {
    gufo::models::QwenKvCache kv_cache(weights.config.num_layers,
                                       weights.config.num_key_value_heads,
                                       kContext, weights.config.head_dim);
    gufo::models::QwenSsmCache ssm_cache(
        weights.config.num_layers, weights.config.SsmQkvSize(),
        weights.config.ssm_conv_kernel, weights.config.ssm_time_step_rank,
        weights.config.ssm_state_size, weights.config.SsmValueSize());
    gufo::models::QwenScratchArena arena(weights.config);
    std::vector<float> logits(weights.config.vocab_size);
    gufo::models::ForwardModel(7, 0, weights, kv_cache, ssm_cache, arena,
                               logits);
    return logits;
  };

  gufo::models::QwenModelWeights with_mtp = *loaded;
  with_mtp.config.mtp_num_layers = 1;
  gufo::models::QwenModelWeights without_mtp = *loaded;
  without_mtp.config.mtp_num_layers = 0;
  const auto with_logits = run(with_mtp);
  const auto without_logits = run(without_mtp);
  Expect(with_logits.size() == without_logits.size(),
         "both trunk runs produce the same logit width");
  Expect(std::memcmp(with_logits.data(), without_logits.data(),
                     with_logits.size() * sizeof(float)) == 0,
         "the draft block does not change a single trunk logit");
}

void TestDraftStepRunsOnTheMoeBlock() {
  const auto bytes = BuildSyntheticMoeGguf(kExperts);
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Expect(reader != nullptr, "draft-step reader opens: " + error);
  std::shared_ptr<const gufo::core::GgufReader> shared(std::move(reader));

  auto reference = gufo::speculative::QwenMtpReference::Create(
      std::move(shared), kContext, &error);
  Expect(reference != nullptr, "MTP reference is created: " + error);

  std::vector<float> target_hidden(kHidden);
  for (std::size_t index = 0; index < target_hidden.size(); ++index) {
    target_hidden[index] = (static_cast<float>(index % 17U) - 8.0F) / 16.0F;
  }
  const auto hidden = reference->ForwardHidden(7, target_hidden, 0);
  Expect(hidden.size() == kHidden, "the MTP draft step returns the hidden row");
  for (std::size_t index = 0; index < hidden.size(); ++index) {
    Expect(std::isfinite(hidden[index]),
           "the MoE draft step produces a finite hidden value");
  }
  Expect(std::isfinite(reference->ComputeLogit(3)),
         "the shared head produces a finite draft logit");
}

}  // namespace

int main() {
  TestMtpMoeBindingBindsEveryNextnTensor();
  TestWrongExpertCountIsRejected();
  TestTrunkIsInvariantToTheDraftBlock();
  TestDraftStepRunsOnTheMoeBlock();
  std::cout << "All Qwen MoE MTP reference tests passed successfully!\n";
  return 0;
}
