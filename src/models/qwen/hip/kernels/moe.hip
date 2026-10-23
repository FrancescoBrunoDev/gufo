#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "src/models/qwen/hip/kernels/moe.hpp"

namespace gufo::hip {
namespace {

constexpr std::uint32_t kRoutingLanes = 32;
/// Marks a (row, slot) the bucketing skipped, so the combine leaves it out.
constexpr std::uint32_t kUnsetPackedSlot = 0xFFFFFFFFU;
constexpr std::uint64_t kRoutingMask = 0xFFFFFFFFFFFFFFFFULL;

/// One warp selects the experts. Every reduction is a warp reduction, and the
/// candidates live in shared memory so a picked expert can be masked out
/// between rounds. Ties resolve to the lowest expert index, matching the CPU
/// reference's first-maximum scan.
__device__ void RouteRow(const float* __restrict__ logits,
                         std::uint32_t* __restrict__ expert_ids,
                         float* __restrict__ expert_weights,
                         std::uint32_t expert_count, std::uint32_t used_count) {
  extern __shared__ float probabilities[];
  const std::uint32_t lane = threadIdx.x;

  float local_max = -INFINITY;
  for (std::uint32_t e = lane; e < expert_count; e += kRoutingLanes) {
    local_max = fmaxf(local_max, logits[e]);
  }
  for (int offset = kRoutingLanes / 2; offset > 0; offset >>= 1) {
    local_max =
        fmaxf(local_max, __shfl_down_sync(kRoutingMask, local_max, offset));
  }
  const float max_logit = __shfl_sync(kRoutingMask, local_max, 0);

  float local_sum = 0.0F;
  for (std::uint32_t e = lane; e < expert_count; e += kRoutingLanes) {
    local_sum += expf(logits[e] - max_logit);
  }
  for (int offset = kRoutingLanes / 2; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(kRoutingMask, local_sum, offset);
  }
  const float total = __shfl_sync(kRoutingMask, local_sum, 0);
  const float inverse_total = total > 0.0F ? 1.0F / total : 0.0F;

  for (std::uint32_t e = lane; e < expert_count; e += kRoutingLanes) {
    probabilities[e] = expf(logits[e] - max_logit) * inverse_total;
  }
  __syncwarp(kRoutingMask);

  for (std::uint32_t slot = 0; slot < used_count; ++slot) {
    float best = -INFINITY;
    std::uint32_t best_index = 0;
    for (std::uint32_t e = lane; e < expert_count; e += kRoutingLanes) {
      if (probabilities[e] > best) {
        best = probabilities[e];
        best_index = e;
      }
    }
    for (int offset = kRoutingLanes / 2; offset > 0; offset >>= 1) {
      const float other = __shfl_down_sync(kRoutingMask, best, offset);
      const auto other_index =
          __shfl_down_sync(kRoutingMask, best_index, offset);
      if (other > best || (other == best && other_index < best_index)) {
        best = other;
        best_index = other_index;
      }
    }
    best = __shfl_sync(kRoutingMask, best, 0);
    best_index = __shfl_sync(kRoutingMask, best_index, 0);
    if (lane == 0) {
      expert_ids[slot] = best_index;
      expert_weights[slot] = best;
    }
    // A negative sentinel keeps an already picked expert out of later rounds.
    probabilities[best_index] = -1.0F;
    __syncwarp(kRoutingMask);
  }

  // norm_topk_prob: renormalize the selected probabilities.
  if (lane == 0) {
    float selected_total = 0.0F;
    for (std::uint32_t slot = 0; slot < used_count; ++slot) {
      selected_total += expert_weights[slot];
    }
    if (selected_total > 0.0F) {
      for (std::uint32_t slot = 0; slot < used_count; ++slot) {
        expert_weights[slot] /= selected_total;
      }
    }
  }
}

/// out += weight * expert_out; the scalar weight stays in device memory.
__global__ void MoeAccumulateKernel(const float* __restrict__ expert_out,
                                    const float* __restrict__ weight,
                                    float* __restrict__ out,
                                    std::size_t elements) {
  const float scale = *weight;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    out[i] += scale * expert_out[i];
  }
}

/// Packs the rows that chose one expert so a batched GEMM sees them
/// contiguously.
__global__ void MoeGatherRowsKernel(const float* __restrict__ x,
                                    const std::uint32_t* __restrict__ rows,
                                    float* __restrict__ dst,
                                    std::uint32_t count, std::size_t hidden) {
  const std::size_t total = static_cast<std::size_t>(count) * hidden;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t row = i / hidden;
    const std::size_t column = i - (row * hidden);
    dst[i] = x[(static_cast<std::size_t>(rows[row]) * hidden) + column];
  }
}

/// Sums a row's routed expert results, one block per row. `slot_to_packed` maps
/// (row, slot) to the position the bucketing gave that selection, so a block
/// owns its whole output row: no atomics, and the accumulation happens in slot
/// order, which keeps the result independent of the expert schedule.
__global__ void MoeCombineSlotsKernel(
    const float* __restrict__ src,
    const std::uint32_t* __restrict__ slot_to_packed,
    const float* __restrict__ weights, float* __restrict__ out,
    std::uint32_t rows, std::size_t hidden, std::uint32_t slot_stride,
    std::uint32_t used_slots) {
  const std::uint32_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const float* row_weights =
      weights + (static_cast<std::size_t>(row) * slot_stride);
  const std::uint32_t* row_slots =
      slot_to_packed + (static_cast<std::size_t>(row) * slot_stride);
  float* dst = out + (static_cast<std::size_t>(row) * hidden);
  for (std::size_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    float acc = 0.0F;
    for (std::uint32_t slot = 0; slot < used_slots; ++slot) {
      const std::uint32_t packed = row_slots[slot];
      if (packed == kUnsetPackedSlot) {
        continue;
      }
      acc += row_weights[slot] *
             src[(static_cast<std::size_t>(packed) * hidden) + col];
    }
    dst[col] += acc;
  }
}

/// Weighted accumulation of a packed expert result back onto its token. The
/// selected weight stays in device memory.
__global__ void MoeScatterAccumulateKernel(
    const float* __restrict__ src, const std::uint32_t* __restrict__ rows,
    const std::uint32_t* __restrict__ slots, const float* __restrict__ weights,
    float* __restrict__ out, std::uint32_t count, std::size_t hidden,
    std::uint32_t slot_stride) {
  const std::size_t total = static_cast<std::size_t>(count) * hidden;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t row = i / hidden;
    const std::size_t column = i - (row * hidden);
    const std::uint32_t token = rows[row];
    const float scale =
        weights[(static_cast<std::size_t>(token) * slot_stride) + slots[row]];
    atomicAdd(&out[(static_cast<std::size_t>(token) * hidden) + column],
              scale * src[i]);
  }
}

/// out += sigmoid(gate_logit) * expert_out for the shared expert.
__global__ void SigmoidScaledAddKernel(const float* __restrict__ gate_logit,
                                       const float* __restrict__ expert_out,
                                       float* __restrict__ out,
                                       std::size_t elements) {
  const float scale = 1.0F / (1.0F + expf(-gate_logit[0]));
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    out[i] += scale * expert_out[i];
  }
}

__global__ void MoeRouteKernel(const float* __restrict__ logits,
                               std::uint32_t* __restrict__ expert_ids,
                               float* __restrict__ expert_weights,
                               std::uint32_t expert_count,
                               std::uint32_t used_count) {
  RouteRow(logits, expert_ids, expert_weights, expert_count, used_count);
}

/// One routing block per row: the per-row launch was 2048 launches per layer at
/// prefill width, and routing is cheap enough to batch.
__global__ void MoeRouteRowsKernel(const float* __restrict__ logits,
                                   std::uint32_t* __restrict__ expert_ids,
                                   float* __restrict__ expert_weights,
                                   std::uint32_t expert_count,
                                   std::uint32_t used_count,
                                   std::uint32_t slot_stride) {
  const std::uint32_t row = blockIdx.x;
  RouteRow(logits + (static_cast<std::size_t>(row) * expert_count),
           expert_ids + (static_cast<std::size_t>(row) * slot_stride),
           expert_weights + (static_cast<std::size_t>(row) * slot_stride),
           expert_count, used_count);
}

__global__ void SigmoidScaledAddRowsKernel(
    const float* __restrict__ gate_logits, const float* __restrict__ expert_out,
    float* __restrict__ out, std::uint32_t rows, std::size_t hidden) {
  const std::uint32_t row = blockIdx.y;
  const float scale = 1.0F / (1.0F + expf(-gate_logits[row]));
  const float* source = expert_out + (static_cast<std::size_t>(row) * hidden);
  float* target = out + (static_cast<std::size_t>(row) * hidden);
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < hidden;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    target[i] += scale * source[i];
  }
}

}  // namespace

void LaunchMoeRouteRows(const float* logits, std::uint32_t* expert_ids,
                        float* expert_weights, std::uint32_t rows,
                        std::uint32_t expert_count, std::uint32_t used_count,
                        std::uint32_t slot_stride, hipStream_t stream) {
  constexpr std::uint32_t kMaxExperts = 8192;
  if (logits == nullptr || expert_ids == nullptr || expert_weights == nullptr ||
      rows == 0 || expert_count == 0 || expert_count > kMaxExperts ||
      used_count == 0) {
    return;
  }
  const std::size_t shared_bytes =
      static_cast<std::size_t>(expert_count) * sizeof(float);
  MoeRouteRowsKernel<<<rows, kRoutingLanes, shared_bytes, stream>>>(
      logits, expert_ids, expert_weights, expert_count, used_count,
      slot_stride);
}

void LaunchSigmoidScaledAddRows(const float* gate_logits,
                                const float* expert_out, float* out,
                                std::uint32_t rows, std::size_t hidden,
                                hipStream_t stream) {
  if (gate_logits == nullptr || expert_out == nullptr || out == nullptr ||
      rows == 0 || hidden == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 128;
  const std::uint32_t blocks =
      static_cast<std::uint32_t>((hidden + kThreads - 1) / kThreads < 32
                                     ? (hidden + kThreads - 1) / kThreads
                                     : 32);
  const dim3 grid(blocks, rows);
  SigmoidScaledAddRowsKernel<<<grid, kThreads, 0, stream>>>(
      gate_logits, expert_out, out, rows, hidden);
}

void LaunchMoeRoute(const float* logits, std::uint32_t* expert_ids,
                    float* expert_weights, std::uint32_t expert_count,
                    std::uint32_t used_count, hipStream_t stream) {
  // The expert field must fit the shared-memory candidate array; every model in
  // this family is far below it.
  constexpr std::uint32_t kMaxExperts = 8192;
  if (logits == nullptr || expert_ids == nullptr || expert_weights == nullptr ||
      expert_count == 0 || expert_count > kMaxExperts || used_count == 0) {
    return;
  }
  const std::size_t shared_bytes =
      static_cast<std::size_t>(expert_count) * sizeof(float);
  MoeRouteKernel<<<1, kRoutingLanes, shared_bytes, stream>>>(
      logits, expert_ids, expert_weights, expert_count, used_count);
}

void LaunchSigmoidScaledAdd(const float* gate_logit, const float* expert_out,
                            float* out, std::size_t elements,
                            hipStream_t stream) {
  if (gate_logit == nullptr || expert_out == nullptr || out == nullptr ||
      elements == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 256;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  SigmoidScaledAddKernel<<<static_cast<std::uint32_t>(blocks < 1024 ? blocks
                                                                    : 1024),
                           kThreads, 0, stream>>>(gate_logit, expert_out, out,
                                                  elements);
}

void LaunchMoeAccumulate(const float* expert_out, const float* weight,
                         float* out, std::size_t elements, hipStream_t stream) {
  if (expert_out == nullptr || weight == nullptr || out == nullptr ||
      elements == 0) {
    return;
  }
  constexpr std::uint32_t kThreads = 256;
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  MoeAccumulateKernel<<<static_cast<std::uint32_t>(blocks < 1024 ? blocks
                                                                 : 1024),
                        kThreads, 0, stream>>>(expert_out, weight, out,
                                               elements);
}

void LaunchMoeGatherRows(const float* x, const std::uint32_t* rows, float* dst,
                         std::uint32_t count, std::size_t hidden,
                         hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (count == 0 || hidden == 0) {
    return;
  }
  const std::size_t total = static_cast<std::size_t>(count) * hidden;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + kThreads - 1) / kThreads, 1024));
  MoeGatherRowsKernel<<<blocks, kThreads, 0, stream>>>(x, rows, dst, count,
                                                       hidden);
}

void LaunchMoeCombineSlots(const float* src,
                           const std::uint32_t* slot_to_packed,
                           const float* weights, float* out, std::uint32_t rows,
                           std::size_t hidden, std::uint32_t slot_stride,
                           std::uint32_t used_slots, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (rows == 0 || hidden == 0 || used_slots == 0) {
    return;
  }
  MoeCombineSlotsKernel<<<rows, kThreads, 0, stream>>>(
      src, slot_to_packed, weights, out, rows, hidden, slot_stride, used_slots);
}

void LaunchMoeScatterAccumulate(const float* src, const std::uint32_t* rows,
                                const std::uint32_t* slots,
                                const float* weights, float* out,
                                std::uint32_t count, std::size_t hidden,
                                std::uint32_t slot_stride, hipStream_t stream) {
  constexpr std::uint32_t kThreads = 256;
  if (count == 0 || hidden == 0) {
    return;
  }
  const std::size_t total = static_cast<std::size_t>(count) * hidden;
  const std::uint32_t blocks = static_cast<std::uint32_t>(
      std::min<std::size_t>((total + kThreads - 1) / kThreads, 1024));
  MoeScatterAccumulateKernel<<<blocks, kThreads, 0, stream>>>(
      src, rows, slots, weights, out, count, hidden, slot_stride);
}

}  // namespace gufo::hip
