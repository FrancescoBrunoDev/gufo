#ifndef GUFO_MODELS_QWEN_HIP_KERNELS_MOE_HPP_
#define GUFO_MODELS_QWEN_HIP_KERNELS_MOE_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gufo::hip {

/// Softmax over every expert followed by the top-`used_count` selection, both
/// on device: `logits` holds `expert_count` router values, and the selected
/// probabilities are renormalized so their sum is one (Qwen3.5-MoE routing).
/// `expert_ids` and `expert_weights` hold `used_count` entries each.
void LaunchMoeRoute(const float* logits, std::uint32_t* expert_ids,
                    float* expert_weights, std::uint32_t expert_count,
                    std::uint32_t used_count, hipStream_t stream = nullptr);

/// The same routing for a whole chunk: one block per row. `slot_stride` is the
/// row stride of the id and weight arrays, which stay `kMaxSelectedExperts`
/// wide.
void LaunchMoeRouteRows(const float* logits, std::uint32_t* expert_ids,
                        float* expert_weights, std::uint32_t rows,
                        std::uint32_t expert_count, std::uint32_t used_count,
                        std::uint32_t slot_stride,
                        hipStream_t stream = nullptr);

/// out += weight * expert_out for one selected expert. `weight` is a device
/// scalar, so the weighted sum never round-trips through the host.
void LaunchMoeAccumulate(const float* expert_out, const float* weight,
                         float* out, std::size_t elements,
                         hipStream_t stream = nullptr);

/// Copies `count` rows of `x` picked by `rows` into a packed buffer, so one
/// batched GEMM can serve every token that routed to the same expert.
void LaunchMoeGatherRows(const float* x, const std::uint32_t* rows, float* dst,
                         std::uint32_t count, std::size_t hidden,
                         hipStream_t stream = nullptr);

/// out[row] += sum over slots of weights[row * slot_stride + slot] *
/// src[slot_to_packed[row * slot_stride + slot]], one block per row. A row's
/// selections are all combined by the same block, so no atomics are needed and
/// the sum runs in slot order. `slot_to_packed` uses 0xFFFFFFFF for a selection
/// the bucketing skipped.
void LaunchMoeCombineSlots(const float* src,
                           const std::uint32_t* slot_to_packed,
                           const float* weights, float* out, std::uint32_t rows,
                           std::size_t hidden, std::uint32_t slot_stride,
                           std::uint32_t used_slots,
                           hipStream_t stream = nullptr);

/// out[rows[i]] += weights[rows[i] * slot_stride + slots[i]] * src[i] for every
/// packed row `i`. Entries of one token can reach this kernel from different
/// blocks, so the accumulation is atomic per element.
void LaunchMoeScatterAccumulate(const float* src, const std::uint32_t* rows,
                                const std::uint32_t* slots,
                                const float* weights, float* out,
                                std::uint32_t count, std::size_t hidden,
                                std::uint32_t slot_stride,
                                hipStream_t stream = nullptr);

/// out += sigmoid(gate[0]) * expert_out, with the gate logit read from device
/// memory. Used by the shared expert of a mixture-of-experts FFN.
void LaunchSigmoidScaledAdd(const float* gate_logit, const float* expert_out,
                            float* out, std::size_t elements,
                            hipStream_t stream = nullptr);

/// The same for a chunk: `gate_logits` holds one scalar per row, and every row
/// contributes `sigmoid(gate) * expert_out` to `out`.
void LaunchSigmoidScaledAddRows(const float* gate_logits,
                                const float* expert_out, float* out,
                                std::uint32_t rows, std::size_t hidden,
                                hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_MOE_HPP_
