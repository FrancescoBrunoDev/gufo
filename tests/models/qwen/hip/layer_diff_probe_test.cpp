#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/forward.hpp"
#include "src/models/qwen/modules/embed.hpp"
#include "src/models/qwen/modules/unembed.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen/tokenizer.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/executor.hpp"

namespace {

constexpr const char* kPrompt = "The capital of Italy is";

/// The arena keeps five feature taps, so the probe compares five layers spread
/// across the stack; narrow the list once a span diverges.
std::vector<std::uint32_t> ParseTaps(const char* text) {
  std::vector<std::uint32_t> taps;
  std::string value = text != nullptr ? text : "0,4,12,24,47";
  std::size_t start = 0;
  while (start <= value.size()) {
    const auto comma = value.find(',', start);
    const auto piece = value.substr(start, comma - start);
    if (!piece.empty()) {
      taps.push_back(static_cast<std::uint32_t>(std::stoul(piece)));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return taps;
}

void WriteDump(const std::string& path, const std::vector<float>& values) {
  std::ofstream out(path, std::ios::binary);
  const auto count = static_cast<std::uint64_t>(values.size());
  out.write(reinterpret_cast<const char*>(&count), sizeof(count));
  out.write(reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

/// CPU reference: one token at a time, capturing the last token's hidden state
/// after every layer.
/// Overwrites the reservoir with the first `hidden` floats of `path`, which is
/// a layer dump in the same format this probe writes. Used to feed one engine's
/// layer output into the other so the next layer can be compared in isolation.
bool InjectHidden(const std::string& path, std::span<float> hidden_state) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }
  std::uint64_t count = 0;
  in.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!in || count < hidden_state.size()) {
    return false;
  }
  in.read(reinterpret_cast<char*>(hidden_state.data()),
          static_cast<std::streamsize>(hidden_state.size() * sizeof(float)));
  return static_cast<bool>(in);
}

int RunCpu(const std::string& model_path, const std::string& out_path,
           std::span<const std::uint32_t> taps, std::size_t token_count,
           const std::string& inject_path) {
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &err);
  if (reader == nullptr) {
    std::cerr << "open: " << err << "\n";
    return 1;
  }
  auto weights = gufo::models::QwenModelWeights::LoadFromGguf(*reader, &err);
  if (!weights.has_value()) {
    std::cerr << "weights: " << err << "\n";
    return 1;
  }
  auto tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*reader, &err);
  if (!tokenizer) {
    std::cerr << "tokenizer: " << err << "\n";
    return 1;
  }
  const auto& config = weights->config;
  const std::size_t hidden = config.hidden_size;
  auto tokens = tokenizer->Encode(kPrompt);
  tokens.resize(std::min(tokens.size(), token_count));
  std::cout << "cpu tokens=" << tokens.size() << " layers=" << config.num_layers
            << "\n";

  gufo::models::QwenKvCache kv(config.FullAttentionLayerCount(),
                               config.num_key_value_heads, 4096,
                               config.head_dim);
  gufo::models::QwenSsmCache ssm(
      config.num_layers, config.SsmQkvSize(), config.ssm_conv_kernel,
      config.ssm_time_step_rank, config.ssm_state_size, config.SsmValueSize());
  gufo::models::QwenScratchArena arena(config);
  std::vector<float> logits(config.vocab_size);
  std::vector<float> layers(taps.size() * hidden);
  const gufo::models::qwen::CpuModuleContext module_ctx;

  for (std::size_t pos = 0; pos < tokens.size(); ++pos) {
    gufo::models::qwen::EmbedForward(module_ctx, tokens[pos],
                                     weights->token_embd, hidden, arena.hidden);
    for (std::uint32_t layer = 0; layer < config.num_layers; ++layer) {
      gufo::models::ForwardLayer(arena.hidden, weights->layers[layer], config,
                                 kv, ssm, layer,
                                 static_cast<std::uint32_t>(pos), arena);
      if (pos + 1 == tokens.size()) {
        for (std::size_t tap = 0; tap < taps.size(); ++tap) {
          if (taps[tap] == layer) {
            std::memcpy(layers.data() + (tap * hidden), arena.hidden.data(),
                        hidden * sizeof(float));
          }
        }
        if (layer == 0 && !inject_path.empty()) {
          if (!InjectHidden(inject_path, arena.hidden)) {
            std::cerr << "cannot inject " << inject_path << "\n";
            return 2;
          }
        }
      }
    }
  }
  const gufo::models::qwen::CpuLayerContext layer_ctx(config, arena);
  gufo::models::qwen::UnembedForward(layer_ctx, weights->output_norm,
                                     weights->output, arena.hidden, logits);
  WriteDump(out_path + ".logits", logits);
  WriteDump(out_path + ".layers", layers);
  std::cout << "cpu wrote " << out_path << " logits[0]=" << logits[0] << "\n";
  return 0;
}

/// HIP reference: one prefill pass with a capture on every layer.
int RunGpu(const std::string& model_path, const std::string& out_path,
           const std::string& mode, std::uint32_t max_context,
           std::span<const std::uint32_t> taps, std::size_t token_count) {
  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(model_path, &err);
  if (reader == nullptr) {
    std::cerr << "open: " << err << "\n";
    return 1;
  }
  auto executor = gufo::hip::QwenGpuExecutor::CreateFromGguf(
      std::shared_ptr<const gufo::core::GgufReader>(std::move(reader)), &err,
      max_context);
  if (executor == nullptr) {
    std::cerr << "executor: " << err << "\n";
    return 1;
  }
  const auto& config = executor->GetConfig();
  const std::size_t hidden = config.hidden_size;
  auto tokens = executor->GetTokenizer().Encode(kPrompt);
  tokens.resize(std::min(tokens.size(), token_count));
  std::cout << "gpu tokens=" << tokens.size() << " layers=" << config.num_layers
            << "\n";
  executor->SetPromptHiddenCapture(true, taps);
  // `gputok` walks the prompt with the decode path so the probe can separate
  // the batched prefill chain from steady-state decode.
  const bool token_by_token = mode == "gputok";
  std::uint32_t first = 0;
  if (token_by_token) {
    for (std::size_t pos = 0; pos < tokens.size(); ++pos) {
      first =
          executor->ForwardToken(tokens[pos], static_cast<std::uint32_t>(pos),
                                 pos + 1 == tokens.size());
    }
  } else {
    first = executor->ForwardPromptBatch(tokens);
  }
  const auto captured = executor->GetPromptHiddenStates();
  const auto logits = executor->CopyLastLogits();
  std::cout << "gpu first token " << first << " captured " << captured.size()
            << " values\n";
  // The capture holds every prompt row; keep the last row per layer.
  std::vector<float> last(hidden * taps.size(), 0.0F);
  const std::size_t row_stride = hidden * taps.size();
  const std::size_t rows = captured.size() / row_stride;
  if (rows != 0) {
    std::memcpy(last.data(), captured.data() + (rows - 1) * row_stride,
                row_stride * sizeof(float));
  }
  WriteDump(out_path + ".logits",
            std::vector<float>(logits.begin(), logits.end()));
  WriteDump(out_path + ".layers", last);
  std::cout << "gpu wrote " << out_path << "\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage: " << argv[0]
              << " <model.gguf> <cpu|gpu|gputok> <out-prefix> [max_context]"
                 " [taps] [tokens]\n";
    return 77;  // ctest skip: this probe is driven by hand
  }
  const auto taps = ParseTaps(argc > 5 ? argv[5] : nullptr);
  const std::size_t token_count =
      argc > 6 ? static_cast<std::size_t>(std::stoul(argv[6])) : 1024U;
  const std::string mode = argv[2];
  const std::uint32_t context =
      argc > 4 ? static_cast<std::uint32_t>(std::stoul(argv[4])) : 4096U;
  try {
    if (mode == "cpu") {
      return RunCpu(argv[1], argv[3], taps, token_count,
                    argc > 7 ? argv[7] : "");
    }
    if (mode == "gpu" || mode == "gputok") {
      return RunGpu(argv[1], argv[3], mode, context, taps, token_count);
    }
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
  std::cerr << "unknown mode\n";
  return 2;
}
#else
int main() {
  std::cout << "HIP disabled\n";
  return 77;
}
#endif
