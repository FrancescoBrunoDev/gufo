// Paired MoE gate/up bucket GEMM.
//
// Production runs three launches per expert bucket: the gate projection, the
// up projection, and a SwiGLU + Q8_1 pass over the two FP32 results. Both
// projections multiply the same activation tile, so the first two launches
// read that tile twice and the third reads 2*count*expert_ffn FP32 bytes back
// that a GEMM had in a register moments earlier.
//
// This kernel keeps the tile geometry the single-matrix route uses (BM 32 rows
// weights tile, adaptive BN token tile) and computes both projections from one
// activation read, then writes SiLU(gate) * up straight into the tiled Q8_1
// layout the down projection consumes. Neither FP32 gate nor FP32 up is
// stored; the second instantiation writes the FP32 gate tile as well for the
// case the caller needs it.
//
// Two shape facts make the fused epilogue free of cross-block traffic:
//
//   * The 32-row macro tile is exactly one 32-element quantization block of
//     the next GEMM's K dimension, so a block owns every value its block
//     scales depend on.
//   * After the K loop the activation staging array is dead and is exactly
//     BN*32 floats, the size of the epilogue's [token][32 rows] scratch, so the
//     transpose and the reduction cost no extra LDS.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "src/models/qwen/hip/kernels/moe_pair.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hpp"

namespace gufo::hip {
namespace {

using core::GgmlType;
using hip::DecodeQuantSub16;
using hip::kQ8ActScaleOffset;
using hip::kQ8ActTileBytes;
using hip::kQ8ActTileTokens;
using hip::Q8ActSumSidecar;
using hip::QuantRowBytes;
using hip::QuantSub16;
using hip::StoreQ8ActLane;
using hip::WmmaQuant;
using int32x4_t = hip::int32x4_t;
using int32x8_t = hip::int32x8_t;

// Same K-block stage depth and wave shape as the single-matrix bucket kernel;
// only the row tile is pinned, because one macro tile has to be one
// quantization block.
constexpr int kPairMatrices = 2;

template<int BM, int BN, int BK, int WM, int WN, GgmlType WType, bool StoreGate>
__launch_bounds__(WM* WN * 32, 1) __global__
    void MoePairGateUpSwiGluQ8Kernel(const void* __restrict__ w_gate,
                                     const void* __restrict__ w_up,
                                     const void* __restrict__ x_blocks,
                                     void* __restrict__ q8_out,
                                     float* __restrict__ gate_out,
                                     std::size_t batch, std::size_t m,
                                     std::size_t k) {
  constexpr bool PerHalfScale =
      WType == GgmlType::kQ6_K || WType == GgmlType::kQ3_K;
  constexpr bool HasOffset =
      WType == GgmlType::kQ4_K || WType == GgmlType::kQ5_K;

  static_assert(BM == 32, "one macro tile must be one Q8_1 quantization block");
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kThreads = WM * WN * 32;
  constexpr int kStageWaves = kThreads / 32;
  constexpr int kAccumulatorElements = 8;
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  constexpr int kScaleHalves = PerHalfScale ? 2 : 1;
  constexpr int kOffRowTiles = HasOffset ? kRowTiles : 1;
  constexpr int kStageTiles = (kTokTiles + kStageWaves - 1) / kStageWaves;

  __shared__ int32x4_t s_a[kPairMatrices][BK][kRowTiles][32];
  __shared__ float s_dw[kPairMatrices][BK][kScaleHalves][kRowTiles][16];
  __shared__ float s_off[kPairMatrices][BK][kOffRowTiles][16];
  __shared__ int32x4_t s_b[BK][kTokTiles][32];
  __shared__ float s_dx[BK][kTokTiles][16];
  __shared__ float s_sx[HasOffset ? BK : 1][kTokTiles][16];

  // The activation's own block count (its K) and the count the quantized
  // output is indexed by (this tile's row block) are different numbers here.
  const std::size_t num_blocks = k / 32;
  const std::size_t out_blocks = m / 32;
  const std::size_t row_bytes = QuantRowBytes(WType, k);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid / 32;
  const int stage_wave = tid >> 5;
  const int lane_id = tid % 32;
  const int stage_lane = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kPairMatrices][kWaveRowTiles][kWaveTokTiles][kAccumulatorElements];
#pragma unroll
  for (int mm = 0; mm < kPairMatrices; ++mm) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
        for (int l = 0; l < kAccumulatorElements; ++l) {
          acc[mm][i][j][l] = 0.0F;
        }
      }
    }
  }

  // Both weight matrices share one fetch stream: the 2*BM*BK units of a stage
  // divide evenly across the block, unlike the single-matrix kernel where half
  // the threads have nothing to fetch at BM 32.
  constexpr int kUnitsPerMatrix = BM * BK;
  constexpr int kFetchUnits = kPairMatrices * kUnitsPerMatrix;
  constexpr int kPrefetch = (kFetchUnits + kThreads - 1) / kThreads;
  constexpr bool kPartialFetch = (kPrefetch * kThreads) != kFetchUnits;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw0[kPrefetch];
  float r_dw1[kPrefetch];
  float r_off[kPrefetch];
  int32x4_t r_b[kStageTiles][BK];
  float r_dx[kStageTiles][BK];
  float r_sx[kStageTiles][HasOffset ? BK : 1];

  const int num_kb = static_cast<int>(num_blocks);
  // Unit layout inside one prefetch step: matrix, then row of the macro tile,
  // then the K block the thread stages.
  const std::uint8_t* w_row[kPrefetch];
  int r_matrix[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int idx = (p * kThreads) + tid;
    const bool unit_live = !kPartialFetch || idx < kFetchUnits;
    const int idx_clamped = unit_live ? idx : 0;
    r_matrix[p] = idx_clamped / kUnitsPerMatrix;
    const int within = idx_clamped % kUnitsPerMatrix;
    const int r = static_cast<int>(r_block) + (within / BK);
    const void* const matrix = (r_matrix[p] == 0) ? w_gate : w_up;
    w_row[p] = static_cast<const std::uint8_t*>(matrix) +
               (static_cast<std::size_t>(r) * row_bytes);
  }

  const std::size_t num_act_tiles =
      (batch + kQ8ActTileTokens - 1) / kQ8ActTileTokens;
  int b_tile[kStageTiles];
  const std::int8_t* b_base[kStageTiles];
  std::size_t act_tile[kStageTiles];
  float tile_scale[kStageTiles];
  bool b_live[kStageTiles];
#pragma unroll
  for (int t = 0; t < kStageTiles; ++t) {
    b_tile[t] = stage_wave + (t * kStageWaves);
    const std::size_t wanted = tt_block + static_cast<std::size_t>(b_tile[t]);
    const bool live = (b_tile[t] < kTokTiles) && (wanted < num_act_tiles);
    act_tile[t] = live ? wanted : 0;
    tile_scale[t] = live ? 1.0F : 0.0F;
    b_live[t] = b_tile[t] < kTokTiles;
    b_base[t] = reinterpret_cast<const std::int8_t*>(x_blocks) +
                (act_tile[t] * num_blocks * kQ8ActTileBytes);
  }
  const float* activation_sums = Q8ActSumSidecar(x_blocks, batch, k);

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * kThreads) + tid;
      const int idx_clamped = (!kPartialFetch || idx < kFetchUnits) ? idx : 0;
      const int kb = kb0 + (idx_clamped % kUnitsPerMatrix) % BK;
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const float live = (kb < num_kb) ? 1.0F : 0.0F;
      QuantSub16 lo;
      QuantSub16 hi;
      DecodeQuantSub16(WType, w_row[p],
                       static_cast<std::size_t>(kb_clamped) * 2, lo);
      DecodeQuantSub16(WType, w_row[p],
                       (static_cast<std::size_t>(kb_clamped) * 2) + 1, hi);
      __builtin_memcpy(&r_q0[p], lo.q, 16);
      __builtin_memcpy(&r_q1[p], hi.q, 16);
      r_dw0[p] = lo.scale * live;
      r_dw1[p] = hi.scale * live;
      if constexpr (HasOffset) {
        r_off[p] = lo.offset * live;
      }
    }
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
      if (!b_live[t]) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile = b_base[t] + (static_cast<std::size_t>(kb_clamped) *
                                        kQ8ActTileBytes);
        r_b[t][i] = reinterpret_cast<const int32x4_t*>(tile)[stage_lane];
        r_dx[t][i] = ((kb < num_kb) ? tile_scale[t] : 0.0F) *
                     reinterpret_cast<const float*>(
                         tile + kQ8ActScaleOffset)[lane_id & 15];
        if constexpr (HasOffset) {
          r_sx[t][i] = ((kb < num_kb) ? tile_scale[t] : 0.0F) *
                       activation_sums[(((act_tile[t] * num_blocks) +
                                         static_cast<std::size_t>(kb_clamped)) *
                                        kQ8ActTileTokens) +
                                       static_cast<std::size_t>(lane_id & 15)];
        }
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * kThreads) + tid;
      if constexpr (kPartialFetch) {
        if (idx >= kFetchUnits) {
          continue;
        }
      }
      const int within = idx % kUnitsPerMatrix;
      const int rr = within / BK;
      const int kk = within % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      const int slot = (rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2));
      s_a[r_matrix[p]][kk][rs][rl] = r_q0[p];
      s_a[r_matrix[p]][kk][rs][16 + rl] = r_q1[p];
      s_dw[r_matrix[p]][kk][0][rs][slot] = r_dw0[p];
      if constexpr (PerHalfScale) {
        s_dw[r_matrix[p]][kk][1][rs][slot] = r_dw1[p];
      }
      if constexpr (HasOffset) {
        s_off[r_matrix[p]][kk][rs][slot] = r_off[p];
      }
    }
#pragma unroll
    for (int t = 0; t < kStageTiles; ++t) {
      if (!b_live[t]) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile[t]][stage_lane] = r_b[t][i];
        if (stage_lane < 16) {
          s_dx[i][b_tile[t]][stage_lane] = r_dx[t][i];
          if constexpr (HasOffset) {
            s_sx[i][b_tile[t]][stage_lane] = r_sx[t][i];
          }
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll(BK <= 2 ? BK : 1)
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kPairMatrices][kWaveRowTiles];
      int32x4_t a1[kPairMatrices][kWaveRowTiles];
      float dw0[kPairMatrices][kWaveRowTiles][kAccumulatorElements];
      float dw1[PerHalfScale ? kPairMatrices : 1]
               [PerHalfScale ? kWaveRowTiles : 1][kAccumulatorElements];
      float off[HasOffset ? kPairMatrices : 1][HasOffset ? kWaveRowTiles : 1]
               [kAccumulatorElements];
#pragma unroll
      for (int mm = 0; mm < kPairMatrices; ++mm) {
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          const int rs = (wave_row * kWaveRowTiles) + i;
          a0[mm][i] = s_a[mm][kb][rs][sub_lane];
          a1[mm][i] = s_a[mm][kb][rs][16 + sub_lane];
          const float4 lo = *reinterpret_cast<const float4*>(
              &s_dw[mm][kb][0][rs][half_id * kAccumulatorElements]);
          dw0[mm][i][0] = lo.x;
          dw0[mm][i][1] = lo.y;
          dw0[mm][i][2] = lo.z;
          dw0[mm][i][3] = lo.w;
          const float4 up = *reinterpret_cast<const float4*>(
              &s_dw[mm][kb][0][rs][(half_id * kAccumulatorElements) + 4]);
          dw0[mm][i][4] = up.x;
          dw0[mm][i][5] = up.y;
          dw0[mm][i][6] = up.z;
          dw0[mm][i][7] = up.w;
          if constexpr (PerHalfScale) {
            const float4 hlo = *reinterpret_cast<const float4*>(
                &s_dw[mm][kb][1][rs][half_id * kAccumulatorElements]);
            dw1[mm][i][0] = hlo.x;
            dw1[mm][i][1] = hlo.y;
            dw1[mm][i][2] = hlo.z;
            dw1[mm][i][3] = hlo.w;
            const float4 hup = *reinterpret_cast<const float4*>(
                &s_dw[mm][kb][1][rs][(half_id * kAccumulatorElements) + 4]);
            dw1[mm][i][4] = hup.x;
            dw1[mm][i][5] = hup.y;
            dw1[mm][i][6] = hup.z;
            dw1[mm][i][7] = hup.w;
          }
          if constexpr (HasOffset) {
            const float4 olo = *reinterpret_cast<const float4*>(
                &s_off[mm][kb][rs][half_id * kAccumulatorElements]);
            off[mm][i][0] = olo.x;
            off[mm][i][1] = olo.y;
            off[mm][i][2] = olo.z;
            off[mm][i][3] = olo.w;
            const float4 oup = *reinterpret_cast<const float4*>(
                &s_off[mm][kb][rs][(half_id * kAccumulatorElements) + 4]);
            off[mm][i][4] = oup.x;
            off[mm][i][5] = oup.y;
            off[mm][i][6] = oup.z;
            off[mm][i][7] = oup.w;
          }
        }
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
        float sx = 0.0F;
        if constexpr (HasOffset) {
          sx = s_sx[kb][ts][sub_lane];
        }
#pragma unroll
        for (int mm = 0; mm < kPairMatrices; ++mm) {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            if constexpr (PerHalfScale) {
              int32x8_t c0{};
              int32x8_t c1{};
              c0 = WmmaQuant<32>(a0[mm][i], b0, c0);
              c1 = WmmaQuant<32>(a1[mm][i], b1, c1);
#pragma unroll
              for (int l = 0; l < kAccumulatorElements; ++l) {
                acc[mm][i][j][l] +=
                    (dw0[mm][i][l] * dx) * static_cast<float>(c0[l]) +
                    (dw1[mm][i][l] * dx) * static_cast<float>(c1[l]);
              }
            } else {
              int32x8_t c{};
              c = WmmaQuant<32>(a0[mm][i], b0, c);
              c = WmmaQuant<32>(a1[mm][i], b1, c);
#pragma unroll
              for (int l = 0; l < kAccumulatorElements; ++l) {
                acc[mm][i][j][l] +=
                    (dw0[mm][i][l] * dx) * static_cast<float>(c[l]);
              }
            }
            if constexpr (HasOffset) {
#pragma unroll
              for (int l = 0; l < kAccumulatorElements; ++l) {
                acc[mm][i][j][l] -= off[mm][i][l] * sx;
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Epilogue. The activation staging array is dead and is exactly BN*32
  // floats, so it becomes [token in the macro tile][row in the 32-row
  // quantization block] scratch and no wave keeps its own slot.
  constexpr int kEpilogueScratchBytes =
      BN * 32 * static_cast<int>(sizeof(float));
  static_assert(sizeof(s_b) == kEpilogueScratchBytes,
                "the activation stage must be exactly the epilogue scratch");
  float* const s_out = reinterpret_cast<float*>(&s_b[0][0][0]);
  const std::size_t blk = r_block / 32;

  const auto store_tile = [&](int row_tile, int tok_tile, bool gate_values) {
    const std::size_t r0 =
        r_block +
        static_cast<std::size_t>(((wave_row * kWaveRowTiles) + row_tile) * 16);
    const std::size_t t0 =
        t_block +
        static_cast<std::size_t>(((wave_tok * kWaveTokTiles) + tok_tile) * 16);
#pragma unroll
    for (int l = 0; l < kAccumulatorElements; ++l) {
      const int row_local = (2 * l) + half_id;
      const std::size_t tok = t0 + static_cast<std::size_t>(sub_lane);
      if (tok >= batch) {
        continue;
      }
      if (gate_values) {
        s_out[((tok - t_block) * 32) + (r0 - r_block) +
              static_cast<std::size_t>(row_local)] =
            acc[0][row_tile][tok_tile][l];
      } else {
        const float g = acc[0][row_tile][tok_tile][l];
        const float u = acc[1][row_tile][tok_tile][l];
        s_out[((tok - t_block) * 32) + (r0 - r_block) +
              static_cast<std::size_t>(row_local)] =
            (g / (1.0F + expf(-g))) * u;
      }
    }
  };

  __syncthreads();
  if constexpr (StoreGate) {
    // The caller's FP32 gate tile: one extra transpose through the same
    // scratch, then a coalesced [token, m] store with one lane per row.
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        store_tile(i, j, true);
      }
    }
    __syncthreads();
    for (int tok_local = stage_wave; tok_local < BN; tok_local += kStageWaves) {
      const std::size_t tok = t_block + static_cast<std::size_t>(tok_local);
      if (tok < batch) {
        gate_out[(tok * m) + r_block + static_cast<std::size_t>(lane_id)] =
            s_out[(tok_local * 32) + lane_id];
      }
    }
    __syncthreads();
  }

  // Unrolled so the accumulator indices are compile-time constants; a runtime
  // index would spill the whole tile to private memory.
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      store_tile(i, j, false);
    }
  }
  __syncthreads();

  // One 32-lane group per token: lane == row of the quantization block, so the
  // scale and the offset term come from one full-wave reduction each, and the
  // store is the same tiled Q8_1 helper the quantize kernels use. Tokens past
  // the batch quantize to zeros, which is what the separate route's tail
  // zeroing pass writes for them.
  for (int tok_local = stage_wave; tok_local < BN; tok_local += kStageWaves) {
    const std::size_t tok = t_block + static_cast<std::size_t>(tok_local);
    if ((tok / kQ8ActTileTokens) >= num_act_tiles) {
      continue;
    }
    const float val = (tok < batch) ? s_out[(tok_local * 32) + lane_id] : 0.0F;
    float max_abs = fabsf(val);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
    }
    const float d = max_abs / 127.0F;
    const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
    const auto q = static_cast<std::int8_t>(roundf(val * id));
    StoreQ8ActLane<true>(q8_out, batch, out_blocks, tok, blk,
                         static_cast<std::size_t>(lane_id), d, q);
  }
}

template<int BM, int BN, GgmlType WType, bool StoreGate>
void LaunchPairTile(const void* w_gate, const void* w_up, const void* x,
                    void* q8_out, float* gate_out, std::size_t batch,
                    std::size_t m, std::size_t k, hipStream_t stream) {
  const dim3 grid(static_cast<unsigned int>((batch + BN - 1) / BN),
                  static_cast<unsigned int>(m / BM));
  MoePairGateUpSwiGluQ8Kernel<BM, BN, 4, 2, 4, WType, StoreGate>
      <<<grid, 256, 0, stream>>>(w_gate, w_up, x, q8_out, gate_out, batch, m,
                                 k);
}

template<GgmlType WType, bool StoreGate>
void LaunchPairTyped(const void* w_gate, const void* w_up, const void* x,
                     void* q8_out, float* gate_out, std::size_t batch,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  // The 64-token tile matches the single-matrix route's choice for narrow
  // buckets. Above it this stops at 128: the 256-token tile holds four token
  // subtiles per wave, so the two live matrices need 64 accumulator registers,
  // and that instantiation spilled 72 VGPRs per lane into a 164-byte private
  // segment. It measured no faster than two 128-token blocks (0.97-1.01x the
  // chain at bucket 256); with the 128 cap that width reaches 1.23-1.29x.
  if (batch <= 64) {
    LaunchPairTile<32, 64, WType, StoreGate>(w_gate, w_up, x, q8_out, gate_out,
                                             batch, m, k, stream);
    return;
  }
  LaunchPairTile<32, 128, WType, StoreGate>(w_gate, w_up, x, q8_out, gate_out,
                                            batch, m, k, stream);
}

template<bool StoreGate>
bool LaunchPairDispatch(GgmlType type, const void* w_gate, const void* w_up,
                        const void* q8_1_x, void* q8_1_out, float* gate_out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  switch (type) {
    case GgmlType::kQ4_K:
      LaunchPairTyped<GgmlType::kQ4_K, StoreGate>(
          w_gate, w_up, q8_1_x, q8_1_out, gate_out, batch, m, k, stream);
      return true;
    case GgmlType::kQ5_K:
      LaunchPairTyped<GgmlType::kQ5_K, StoreGate>(
          w_gate, w_up, q8_1_x, q8_1_out, gate_out, batch, m, k, stream);
      return true;
    default:
      return false;
  }
}

}  // namespace

bool LaunchMoePairGateUpSwiGluQ8(GgmlType type, const void* w_gate,
                                 const void* w_up, const void* q8_1_x,
                                 void* q8_1_out, float* gate_out,
                                 std::size_t batch, std::size_t m,
                                 std::size_t k, hipStream_t stream) {
  if (batch == 0 || m == 0 || k == 0 || (m % 32) != 0 || (k % 32) != 0) {
    return false;
  }
  if (gate_out != nullptr) {
    return LaunchPairDispatch<true>(type, w_gate, w_up, q8_1_x, q8_1_out,
                                    gate_out, batch, m, k, stream);
  }
  return LaunchPairDispatch<false>(type, w_gate, w_up, q8_1_x, q8_1_out,
                                   nullptr, batch, m, k, stream);
}

}  // namespace gufo::hip
