#ifndef GUFO_MODELS_QWEN_MOE_HPP_
#define GUFO_MODELS_QWEN_MOE_HPP_

#include <array>
#include <cstdint>
#include <span>

#include "src/models/qwen/modules/layer_view.hpp"
#include "src/models/qwen/modules/module_ctx.hpp"
#include "src/models/qwen/state.hpp"

namespace gufo::models::qwen {

/// Routed experts per token that the selection and kernel scratch support.
inline constexpr std::uint32_t kMaxSelectedExperts = 8;

/// Top-k routing decision for one token.
struct MoeSelection {
  std::uint32_t count{0};
  std::array<std::uint32_t, kMaxSelectedExperts> experts{};
  std::array<float, kMaxSelectedExperts> weights{};
};

/// Qwen3.5-MoE routing: softmax over every expert, top-k by probability, then
/// renormalized so the selected weights sum to one. There is no routed scale
/// (the family's GGUF carries no expert scale key) and the shared expert is
/// gated separately, so it never enters this softmax.
/// `logits` is scratch for `expert_count` values.
void MoeRoute(const QwenTensorRef& router, std::span<const float> x,
              std::uint32_t expert_count, std::uint32_t used_count,
              std::span<float> logits, MoeSelection* selection) noexcept;

/// MoE feed-forward for one token: the weighted sum of the selected SwiGLU
/// experts plus the sigmoid-gated shared expert.
/// `router_logits` holds `expert_count` values, `gate_scratch`/`up_scratch`/
/// `act_scratch` hold one expert's intermediate width and `expert_out` holds
/// `hidden_size` values. `out` is overwritten.
void MoeFfnForward(const QwenLayerWeights& layer,
                   const core::ModelConfig& config, std::span<const float> x,
                   std::span<float> router_logits,
                   std::span<float> gate_scratch, std::span<float> up_scratch,
                   std::span<float> act_scratch, std::span<float> expert_out,
                   std::span<float> out) noexcept;

/// HIP overload for one token. Routing stays on device; `expert_ids` and
/// `expert_weights` hold `kMaxSelectedExperts` entries each.
void MoeFfnForward(const HipModuleContext& ctx, const QwenLayerWeights& layer,
                   const core::ModelConfig& config, std::span<const float> x,
                   std::span<float> router_logits,
                   std::span<std::uint32_t> expert_ids,
                   std::span<float> expert_weights,
                   std::span<float> gate_scratch, std::span<float> up_scratch,
                   std::span<float> act_scratch, std::span<float> expert_out,
                   std::span<float> out) noexcept;

/// HIP overload for `rows` tokens sharing one synchronous routing read:
/// `x`/`out` hold `rows * hidden`, `router_logits` `rows * expert_count`, and
/// `expert_ids`/`expert_weights` `rows * kMaxSelectedExperts`. The per-expert
/// scratch holds one row's width and is reused across rows.
void MoeFfnForwardRows(
    const HipModuleContext& ctx, const QwenLayerWeights& layer,
    const core::ModelConfig& config, std::size_t rows, std::span<const float> x,
    std::span<float> router_logits, std::span<std::uint32_t> expert_ids,
    std::span<float> expert_weights, std::span<float> gate_scratch,
    std::span<float> up_scratch, std::span<float> act_scratch,
    std::span<float> expert_out, std::span<float> out) noexcept;

/// Expert-bucketed variant for a prefill batch. Same routing and numerics as
/// `MoeFfnForwardRows`, but the tokens that chose one expert are packed with
/// `LaunchMoeGatherRows` and served by one batched GEMM per expert, so the
/// launch count per layer drops from `rows * used * 3` to `experts * 3`.
/// `gathered` holds `rows * hidden`, `perm_rows`/`perm_slots`
/// `rows * kMaxSelectedExperts`, and `expert_offsets` `expert_count + 1`.
void MoeFfnForwardRowsBucketed(
    const HipModuleContext& ctx, const QwenLayerWeights& layer,
    const core::ModelConfig& config, std::size_t rows, std::span<const float> x,
    std::span<float> router_logits, std::span<std::uint32_t> expert_ids,
    std::span<float> expert_weights, std::span<float> gate_scratch,
    std::span<float> up_scratch, std::span<float> act_scratch,
    std::span<float> expert_out, std::span<float> out,
    std::span<float> gathered, std::span<std::uint32_t> perm_rows,
    std::span<std::uint32_t> perm_slots,
    std::span<std::uint32_t> expert_offsets, std::span<std::byte> q8_activation,
    std::span<float> moe_gate, void* hipblas) noexcept;

}  // namespace gufo::models::qwen

#endif  // GUFO_MODELS_QWEN_MOE_HPP_
