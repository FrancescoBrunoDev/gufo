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

/// out += weight * expert_out for one selected expert. `weight` is a device
/// scalar, so the weighted sum never round-trips through the host.
void LaunchMoeAccumulate(const float* expert_out, const float* weight,
                         float* out, std::size_t elements,
                         hipStream_t stream = nullptr);

/// out += sigmoid(gate[0]) * expert_out, with the gate logit read from device
/// memory. Used by the shared expert of a mixture-of-experts FFN.
void LaunchSigmoidScaledAdd(const float* gate_logit, const float* expert_out,
                            float* out, std::size_t elements,
                            hipStream_t stream = nullptr);

}  // namespace gufo::hip

#endif  // GUFO_MODELS_QWEN_HIP_KERNELS_MOE_HPP_
