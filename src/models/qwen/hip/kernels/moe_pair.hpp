#ifndef GUFO_SRC_MODELS_QWEN_HIP_KERNELS_MOE_PAIR_HPP_
#define GUFO_SRC_MODELS_QWEN_HIP_KERNELS_MOE_PAIR_HPP_
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/core/gguf_reader.hpp"

namespace gufo::hip {

/// One launch for a mixture-of-experts bucket's gate and up projection: both
/// weight matrices read the bucket's Q8_1 activation once and the SwiGLU result
/// lands directly in the tiled Q8_1 layout the down projection consumes, so
/// neither FP32 intermediate is stored. `gate_out` may be null; when it is not,
/// the FP32 gate tile is also written in the [token, expert_ffn] layout of the
/// single-matrix route.
///
/// Returns false when the shape is outside this kernel's contract -- `m` and
/// `k` must be multiples of 32 -- in which case nothing is launched.
[[nodiscard]] bool LaunchMoePairGateUpSwiGluQ8(
    core::GgmlType type, const void* w_gate, const void* w_up,
    const void* q8_1_x, void* q8_1_out, float* gate_out, std::size_t batch,
    std::size_t m, std::size_t k, hipStream_t stream);

}  // namespace gufo::hip
#endif
#endif
