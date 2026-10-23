#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "src/models/qwen/hip/kernels/moe_pair.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"
#include "tests/models/qwen/hip/moe_pair_gemm.hpp"

namespace gufo::test {

using gufo::hip::LaunchMoePairGateUpSwiGluQ8;

Q8TileComparison CompareTiledQ8_1(const void* lhs, const void* rhs,
                                  std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const auto* a = static_cast<const std::uint8_t*>(lhs);
  const auto* b = static_cast<const std::uint8_t*>(rhs);
  const auto* a_sums =
      reinterpret_cast<const float*>(a + gufo::hip::Q8ActTiledBytes(batch, k));
  const auto* b_sums =
      reinterpret_cast<const float*>(b + gufo::hip::Q8ActTiledBytes(batch, k));
  Q8TileComparison out;
  // Every slot in every existing tile, not just the live tokens: the down
  // projection reads whole 16-token tiles, so the tail a partial bucket leaves
  // has to match the separate route's zero fill too.
  const std::size_t slots = ((batch + gufo::hip::kQ8ActTileTokens - 1) /
                             gufo::hip::kQ8ActTileTokens) *
                            gufo::hip::kQ8ActTileTokens;
  for (std::size_t tok = 0; tok < slots; ++tok) {
    const std::size_t tt = tok / gufo::hip::kQ8ActTileTokens;
    const std::size_t tl = tok % gufo::hip::kQ8ActTileTokens;
    for (std::size_t blk = 0; blk < num_blocks; ++blk) {
      const std::size_t base =
          ((tt * num_blocks) + blk) * gufo::hip::kQ8ActTileBytes;
      for (std::size_t lane = 0; lane < 32; ++lane) {
        const std::size_t at =
            base + ((lane >> 4u) * 256) + (tl * 16) + (lane & 15u);
        const int qa = static_cast<int>(a[at]);
        const int qb = static_cast<int>(b[at]);
        const int dq = (qa > qb) ? (qa - qb) : (qb - qa);
        out.max_abs_byte = (dq > out.max_abs_byte) ? dq : out.max_abs_byte;
      }
      const std::size_t scale_at =
          base + gufo::hip::kQ8ActScaleOffset + (tl * 4);
      float sa = 0.0F;
      float sb = 0.0F;
      std::memcpy(&sa, a + scale_at, sizeof(sa));
      std::memcpy(&sb, b + scale_at, sizeof(sb));
      out.max_abs_scale = std::max(out.max_abs_scale, std::fabs(sa - sb));
      const std::size_t sum_at =
          (((tt * num_blocks) + blk) * gufo::hip::kQ8ActTileTokens) + tl;
      out.max_abs_sum =
          std::max(out.max_abs_sum, std::fabs(a_sums[sum_at] - b_sums[sum_at]));
    }
  }
  return out;
}

}  // namespace gufo::test
