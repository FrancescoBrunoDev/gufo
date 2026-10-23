#ifndef GUFO_TESTS_MODELS_QWEN_HIP_MOE_PAIR_GEMM_HPP_
#define GUFO_TESTS_MODELS_QWEN_HIP_MOE_PAIR_GEMM_HPP_
#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include <cstddef>

#include "src/models/qwen/hip/kernels/moe_pair.hpp"

namespace gufo::test {

/// Difference between two tiled Q8_1 activation buffers written for the same
/// [batch, k] tensor: the largest int8 code difference, the largest block-scale
/// difference and the largest activation-sum difference, over every slot of
/// every tile a partial bucket leaves behind. Both pointers are host copies.
struct Q8TileComparison {
  int max_abs_byte = 0;
  float max_abs_scale = 0.0F;
  float max_abs_sum = 0.0F;
};

[[nodiscard]] Q8TileComparison CompareTiledQ8_1(const void* lhs,
                                                const void* rhs,
                                                std::size_t batch,
                                                std::size_t k);

}  // namespace gufo::test
#endif
#endif
