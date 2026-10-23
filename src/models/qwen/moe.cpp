#include "src/models/qwen/moe.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

#include "src/models/qwen/forward.hpp"  // TensorGEMV

#if defined(ENGINE_ENABLE_HIP)
#include "src/models/qwen/hip/kernels/moe.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "src/models/qwen/hip/ops/swiglu.hpp"
#endif

namespace gufo::models::qwen {
namespace {

/// One packed expert of a 3D expert tensor. `experts` are the last dimension of
/// the GGUF shape, i.e. the outermost, so each expert is a contiguous run of
/// `hidden * ffn` elements.
QwenTensorRef ExpertSlice(const QwenTensorRef& packed, std::size_t expert,
                          std::size_t elements) noexcept {
  return packed.Slice(expert * elements, elements);
}

void SiluMul(std::span<const float> gate, std::span<const float> up,
             std::span<float> out) noexcept {
  for (std::size_t i = 0; i < out.size(); ++i) {
    const float g = gate[i];
    out[i] = (g / (1.0F + std::exp(-g))) * up[i];
  }
}

}  // namespace

void MoeRoute(const QwenTensorRef& router, std::span<const float> x,
              std::uint32_t expert_count, std::uint32_t used_count,
              std::span<float> logits, MoeSelection* selection) noexcept {
  assert(selection != nullptr);
  assert(used_count <= kMaxSelectedExperts);
  assert(logits.size() >= expert_count);
  selection->count = 0;
  if (expert_count == 0 || used_count == 0 || x.empty()) {
    return;
  }

  TensorGEMV(router, x, expert_count, x.size(), logits);

  // Softmax over every expert in float, like the reference implementation.
  float max_logit = -std::numeric_limits<float>::infinity();
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    max_logit = std::max(max_logit, logits[e]);
  }
  float total = 0.0F;
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    logits[e] = std::exp(logits[e] - max_logit);
    total += logits[e];
  }
  if (!(total > 0.0F)) {
    return;
  }
  for (std::uint32_t e = 0; e < expert_count; ++e) {
    logits[e] /= total;
  }

  const std::uint32_t selected = std::min(used_count, expert_count);
  for (std::uint32_t slot = 0; slot < selected; ++slot) {
    std::uint32_t best = 0;
    float best_value = -1.0F;
    for (std::uint32_t e = 0; e < expert_count; ++e) {
      if (logits[e] > best_value) {
        best_value = logits[e];
        best = e;
      }
    }
    // Absorb the picked expert so the next slot scans the remaining ones.
    logits[best] = -1.0F;
    selection->experts[slot] = best;
    selection->weights[slot] = best_value;
  }

  // norm_topk_prob: the selected probabilities are renormalized.
  float selected_total = 0.0F;
  for (std::uint32_t slot = 0; slot < selected; ++slot) {
    selected_total += selection->weights[slot];
  }
  if (selected_total > 0.0F) {
    for (std::uint32_t slot = 0; slot < selected; ++slot) {
      selection->weights[slot] /= selected_total;
    }
  }
  selection->count = selected;
}

void MoeFfnForward(const QwenLayerWeights& layer,
                   const core::ModelConfig& config, std::span<const float> x,
                   std::span<float> router_logits,
                   std::span<float> gate_scratch, std::span<float> up_scratch,
                   std::span<float> act_scratch, std::span<float> expert_out,
                   std::span<float> out) noexcept {
  const std::size_t hidden = config.hidden_size;
  const std::size_t expert_ffn = config.expert_feed_forward_length;
  const std::size_t shared_ffn = config.shared_expert_feed_forward_length != 0
                                     ? config.shared_expert_feed_forward_length
                                     : expert_ffn;
  assert(x.size() >= hidden);
  assert(out.size() >= hidden);
  assert(router_logits.size() >= config.expert_count);
  assert(gate_scratch.size() >= expert_ffn);
  assert(up_scratch.size() >= expert_ffn);
  assert(act_scratch.size() >= expert_ffn);
  assert(expert_out.size() >= hidden);

  std::fill_n(out.begin(), hidden, 0.0F);

  MoeSelection selection;
  MoeRoute(layer.ffn_gate_inp, x.first(hidden), config.expert_count,
           config.expert_used_count, router_logits, &selection);
  if (std::getenv("GUFO_MOE_DEBUG") != nullptr) {
    std::printf("DEBUG cpu route:");
    for (std::uint32_t slot = 0; slot < selection.count; ++slot)
      std::printf(" %u:%.6f", selection.experts[slot], selection.weights[slot]);
    std::printf("\n");
  }

  const std::span<const float> input = x.first(hidden);
  for (std::uint32_t slot = 0; slot < selection.count; ++slot) {
    const auto expert = selection.experts[slot];
    const auto gate_w =
        ExpertSlice(layer.ffn_gate_exps, expert, expert_ffn * hidden);
    const auto up_w =
        ExpertSlice(layer.ffn_up_exps, expert, expert_ffn * hidden);
    const auto down_w =
        ExpertSlice(layer.ffn_down_exps, expert, hidden * expert_ffn);
    TensorGEMV(gate_w, input, expert_ffn, hidden, gate_scratch);
    TensorGEMV(up_w, input, expert_ffn, hidden, up_scratch);
    SiluMul(gate_scratch, up_scratch, act_scratch);
    TensorGEMV(down_w, act_scratch.first(expert_ffn), hidden, expert_ffn,
               expert_out);
    const float weight = selection.weights[slot];
    for (std::size_t i = 0; i < hidden; ++i) {
      out[i] += weight * expert_out[i];
    }
  }

  if (!layer.ffn_gate_shexp.empty()) {
    // The shared expert is a plain SwiGLU FFN scaled by sigmoid(w . x).
    TensorGEMV(layer.ffn_gate_shexp, input, shared_ffn, hidden, gate_scratch);
    TensorGEMV(layer.ffn_up_shexp, input, shared_ffn, hidden, up_scratch);
    SiluMul(gate_scratch.first(shared_ffn), up_scratch.first(shared_ffn),
            act_scratch.first(shared_ffn));
    TensorGEMV(layer.ffn_down_shexp, act_scratch.first(shared_ffn), hidden,
               shared_ffn, expert_out);
    float gate_logit = 0.0F;
    TensorGEMV(layer.ffn_gate_inp_shexp, input, 1, hidden,
               std::span<float>(&gate_logit, 1));
    const float scale = 1.0F / (1.0F + std::exp(-gate_logit));
    for (std::size_t i = 0; i < hidden; ++i) {
      out[i] += scale * expert_out[i];
    }
  }
}

#if defined(ENGINE_ENABLE_HIP)
namespace {

void LaunchRoutedExperts(const QwenLayerWeights& layer,
                         const core::ModelConfig& config, std::size_t row,
                         std::span<const float> x, std::span<float> out,
                         std::span<float> gate_scratch,
                         std::span<float> up_scratch,
                         std::span<float> act_scratch,
                         std::span<float> expert_out,
                         std::span<const std::uint32_t> selected,
                         const float* row_weights_device, hipStream_t stream) {
  const std::size_t hidden = config.hidden_size;
  const std::size_t expert_ffn = config.expert_feed_forward_length;
  const std::span<const float> input = x.subspan(row * hidden, hidden);
  std::span<float> row_out = out.subspan(row * hidden, hidden);
  (void)hipMemsetAsync(row_out.data(), 0, hidden * sizeof(float), stream);
  for (std::uint32_t slot = 0; slot < config.expert_used_count; ++slot) {
    const auto gate_w =
        ExpertSlice(layer.ffn_gate_exps, selected[slot], expert_ffn * hidden);
    const auto up_w =
        ExpertSlice(layer.ffn_up_exps, selected[slot], expert_ffn * hidden);
    const auto down_w =
        ExpertSlice(layer.ffn_down_exps, selected[slot], hidden * expert_ffn);
    if (gate_w.empty() || up_w.empty() || down_w.empty()) {
      return;
    }
    ::gufo::hip::LaunchGEMV(gate_w.data, gate_w.type, input.data(),
                            gate_scratch.data(), expert_ffn, hidden, stream);
    ::gufo::hip::LaunchGEMV(up_w.data, up_w.type, input.data(),
                            up_scratch.data(), expert_ffn, hidden, stream);
    ::gufo::hip::LaunchBatchedSwiGLUActivation(
        gate_scratch.data(), up_scratch.data(), act_scratch.data(), nullptr,
        expert_ffn, stream);
    ::gufo::hip::LaunchGEMV(down_w.data, down_w.type, act_scratch.data(),
                            expert_out.data(), hidden, expert_ffn, stream);
    // The routed weight stays in device memory: the accumulate kernel reads it.
    ::gufo::hip::LaunchMoeAccumulate(expert_out.data(),
                                     row_weights_device + slot, row_out.data(),
                                     hidden, stream);
  }
}

/// Shared expert: a dense SwiGLU FFN whose output is scaled by
/// sigmoid(gate_inp_shexp . x). It never enters the routed softmax.
void LaunchSharedExpert(const QwenLayerWeights& layer,
                        const core::ModelConfig& config, std::size_t row,
                        std::span<const float> x, std::span<float> out,
                        std::span<float> gate_scratch,
                        std::span<float> up_scratch,
                        std::span<float> act_scratch,
                        std::span<float> expert_out,
                        std::span<float> gate_logit, hipStream_t stream) {
  const std::size_t hidden = config.hidden_size;
  const std::size_t expert_ffn = config.expert_feed_forward_length;
  const std::size_t shared_ffn = config.shared_expert_feed_forward_length != 0
                                     ? config.shared_expert_feed_forward_length
                                     : expert_ffn;
  const std::span<const float> input = x.subspan(row * hidden, hidden);
  std::span<float> row_out = out.subspan(row * hidden, hidden);
  ::gufo::hip::LaunchGEMV(layer.ffn_gate_shexp.data, layer.ffn_gate_shexp.type,
                          input.data(), gate_scratch.data(), shared_ffn, hidden,
                          stream);
  ::gufo::hip::LaunchGEMV(layer.ffn_up_shexp.data, layer.ffn_up_shexp.type,
                          input.data(), up_scratch.data(), shared_ffn, hidden,
                          stream);
  ::gufo::hip::LaunchBatchedSwiGLUActivation(
      gate_scratch.data(), up_scratch.data(), act_scratch.data(), nullptr,
      shared_ffn, stream);
  ::gufo::hip::LaunchGEMV(layer.ffn_down_shexp.data, layer.ffn_down_shexp.type,
                          act_scratch.data(), expert_out.data(), hidden,
                          shared_ffn, stream);
  ::gufo::hip::LaunchGEMV(layer.ffn_gate_inp_shexp.data,
                          layer.ffn_gate_inp_shexp.type, input.data(),
                          gate_logit.data(), 1, hidden, stream);
  ::gufo::hip::LaunchSigmoidScaledAdd(gate_logit.data(), expert_out.data(),
                                      row_out.data(), hidden, stream);
}

}  // namespace

void MoeFfnForwardRows(
    const HipModuleContext& ctx, const QwenLayerWeights& layer,
    const core::ModelConfig& config, std::size_t rows, std::span<const float> x,
    std::span<float> router_logits, std::span<std::uint32_t> expert_ids,
    std::span<float> expert_weights, std::span<float> gate_scratch,
    std::span<float> up_scratch, std::span<float> act_scratch,
    std::span<float> expert_out, std::span<float> out) noexcept {
  const std::size_t hidden = config.hidden_size;
  const std::size_t experts = config.expert_count;
  const std::size_t selected_per_row = config.expert_used_count;
  if (rows == 0 || experts == 0) {
    return;
  }
  const auto stream = static_cast<hipStream_t>(ctx.Stream());

  // Routing for every row stays on device: the router projection, the softmax
  // and the top-k run as GPU work, and exactly one read brings back the
  // per-row expert ids plus their weights.
  for (std::size_t row = 0; row < rows; ++row) {
    ::gufo::hip::LaunchGEMV(layer.ffn_gate_inp.data, layer.ffn_gate_inp.type,
                            x.data() + (row * hidden),
                            router_logits.data() + (row * experts), experts,
                            hidden, stream);
    ::gufo::hip::LaunchMoeRoute(
        router_logits.data() + (row * experts),
        expert_ids.data() + (row * kMaxSelectedExperts),
        expert_weights.data() + (row * kMaxSelectedExperts),
        static_cast<std::uint32_t>(experts),
        static_cast<std::uint32_t>(selected_per_row), stream);
  }

  // shortcut: the routed ids and weights cross to the host once per layer so
  // each expert reuses the existing GEMV launchers on its packed slice. Move
  // the expert GEMV on device when this read shows up in a prefill profile.
  const std::size_t per_row = rows * kMaxSelectedExperts;
  std::vector<std::uint32_t> host_ids(per_row);
  // The arena stream is created with hipStreamNonBlocking, so a blocking copy
  // on the default stream is not ordered against the routing kernels above and
  // would read the previous layer's ids. Drain the stream first.
  if (hipStreamSynchronize(stream) != hipSuccess ||
      hipMemcpy(host_ids.data(), expert_ids.data(),
                per_row * sizeof(std::uint32_t),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return;
  }

  if (std::getenv("GUFO_MOE_DEBUG") != nullptr) {
    for (std::size_t row = 0; row < rows; ++row) {
      std::printf("DEBUG hip route:");
      for (std::size_t slot = 0; slot < selected_per_row; ++slot)
        std::printf(" %u:%.6f", host_ids[row * kMaxSelectedExperts + slot],
                    expert_weights[row * kMaxSelectedExperts + slot]);
      std::printf("\n");
    }
  }
  const bool has_shared_expert = !layer.ffn_gate_shexp.empty();
  for (std::size_t row = 0; row < rows; ++row) {
    const auto selected = std::span<const std::uint32_t>(
        host_ids.data() + (row * kMaxSelectedExperts), selected_per_row);
    LaunchRoutedExperts(layer, config, row, x, out, gate_scratch, up_scratch,
                        act_scratch, expert_out, selected,
                        expert_weights.data() + (row * kMaxSelectedExperts),
                        stream);
    if (has_shared_expert) {
      // The shared-expert scalar gate reuses the tail of this row's logits.
      LaunchSharedExpert(layer, config, row, x, out, gate_scratch, up_scratch,
                         act_scratch, expert_out,
                         router_logits.subspan(row * experts, 1), stream);
    }
  }
}

void MoeFfnForward(const HipModuleContext& ctx, const QwenLayerWeights& layer,
                   const core::ModelConfig& config, std::span<const float> x,
                   std::span<float> router_logits,
                   std::span<std::uint32_t> expert_ids,
                   std::span<float> expert_weights,
                   std::span<float> gate_scratch, std::span<float> up_scratch,
                   std::span<float> act_scratch, std::span<float> expert_out,
                   std::span<float> out) noexcept {
  MoeFfnForwardRows(ctx, layer, config, 1, x, router_logits, expert_ids,
                    expert_weights, gate_scratch, up_scratch, act_scratch,
                    expert_out, out);
}
#endif  // defined(ENGINE_ENABLE_HIP)

}  // namespace gufo::models::qwen
