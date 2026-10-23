// Benchmark for the one GEMM every qwen35moe expert bucket runs: N rows of
// weights against a handful of token rows. It compares the two routes the
// backend already has -- the exact FP32 batched small-batch kernel and the
// tiled Q8_1 activation route the dense FFN prefill uses -- at the expert shape
// this artifact ships (1024 x 3072). Timing only; every case also checks that
// the tiled route stays within the tolerance the model path accepts.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/hip/hip_utils.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
#include "tests/models/qwen/hip/moe_pair_gemm.hpp"

namespace {

// The two expert shapes: gate/up are [expert_ffn, hidden] and down is
// [hidden, expert_ffn].
struct Shape {
  const char* name;
  std::size_t m;
  std::size_t k;
};
constexpr Shape kShapes[] = {{"gate_up", 1024, 3072}, {"down", 3072, 1024}};

float ElapsedUs(hipEvent_t start, hipEvent_t stop) {
  float ms = 0.0F;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  return ms * 1000.0F;
}

// Fill a quantized matrix with a pattern that dequantizes to finite, ordinary
// magnitudes, so the timing measures the kernel and not denormal handling.
// `base` distinguishes two matrices and the payload cycles per block, so a
// weight row, a K block and the row/block mapping of the output all differ.
void FillFinite(gufo::core::GgmlType type, std::vector<std::byte>& packed,
                std::uint8_t base = 0x01) {
  const std::size_t block_elems = ::gufo::quant::QuantizedBlockElements(type);
  // The element count has to be a whole block: EncodedSizeBytes(type, 1) is
  // zero for every 256-element K-quant block, which used to leave these
  // matrices untouched and compared uninitialized bytes with zeros.
  const std::size_t block_bytes =
      ::gufo::quant::EncodedSizeBytes(type, block_elems);
  if (block_bytes == 0 || block_elems == 0) {
    return;
  }
  std::size_t index = 0;
  for (std::size_t b = 0; b + block_bytes <= packed.size();
       b += block_bytes, ++index) {
    std::byte* block = packed.data() + b;
    // Q4_K/Q5_K/Q8_0 all start with a float16 scale, so a small fill gives a
    // tiny offset and a scale of one: finite, ordinary magnitudes.
    const std::uint8_t payload = static_cast<std::uint8_t>(base + (index % 7));
    std::memset(block, payload, block_bytes);
    const std::uint16_t one = 0x3C00;  // fp16 1.0
    std::memcpy(block, &one, sizeof(one));
  }
}

struct Case {
  const char* name;
  gufo::core::GgmlType type;
};

}  // namespace

namespace {

// The expert gate/up shape: both matrices are [expert_ffn, hidden].
constexpr std::size_t kPairFfn = 1024;
constexpr std::size_t kPairHidden = 3072;
constexpr int kPairIterations = 20;

// Today's per-bucket chain against the paired kernel. The input quantization
// is in both routes because both need the bucket's Q8_1 activation; the two
// launches the pair removes are the second projection and the separate SwiGLU
// + Q8_1 pass. Two interleaved runs of each so drift is visible.
void RunPairGateUpBench() {
  const std::vector<Case> cases = {
      {"q4_k", gufo::core::GgmlType::kQ4_K},
      {"q5_k", gufo::core::GgmlType::kQ5_K},
  };
  // 24 is the ragged width: it exercises the zero-filled tail of the last
  // 16-token tile, which the down projection reads as part of the tile.
  const std::vector<std::size_t> widths = {16, 24, 32, 64, 128, 256};

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  hipStream_t stream = nullptr;

  for (const Case& bench_case : cases) {
    const std::size_t row_bytes =
        ::gufo::quant::QuantizedRowBytes(bench_case.type, kPairHidden);
    std::vector<std::byte> host_gate_w(row_bytes * kPairFfn);
    std::vector<std::byte> host_up_w(row_bytes * kPairFfn);
    FillFinite(bench_case.type, host_gate_w);
    // Distinct payload so a gate/up weight mix-up cannot pass the comparison.
    FillFinite(bench_case.type, host_up_w, 0x02);
    void* dev_gate_w = nullptr;
    void* dev_up_w = nullptr;
    HIP_CHECK(hipMalloc(&dev_gate_w, host_gate_w.size()));
    HIP_CHECK(hipMalloc(&dev_up_w, host_up_w.size()));
    HIP_CHECK(hipMemcpy(dev_gate_w, host_gate_w.data(), host_gate_w.size(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_up_w, host_up_w.data(), host_up_w.size(),
                        hipMemcpyHostToDevice));
    const double bucket_gb = 2.0 * static_cast<double>(host_gate_w.size()) /
                             (1024.0 * 1024.0 * 1024.0);

    std::printf(
        "\npair gate/up bucket  type=%s N=%zu K=%zu  unique weight bytes="
        "%.2f MiB per bucket\n",
        bench_case.name, kPairFfn, kPairHidden,
        2.0 * static_cast<double>(host_gate_w.size()) / (1024.0 * 1024.0));
    std::printf("%6s %5s %10s %11s %10s %11s %9s %8s %12s %11s %11s\n", "width",
                "run", "chain_us", "chain GB/s", "pair_us", "pair GB/s",
                "speedup", "max|dq|", "max|d scale|", "max|d sum|",
                "max|d gate|");

    for (const std::size_t batch : widths) {
      const std::size_t x_elems = batch * kPairHidden;
      const std::size_t y_elems = batch * kPairFfn;
      std::vector<float> host_x(x_elems);
      for (std::size_t i = 0; i < x_elems; ++i) {
        host_x[i] = 0.05F * std::sin(0.017F * static_cast<float>(i));
      }
      float* dev_x = nullptr;
      void* dev_q8_x = nullptr;
      float* dev_gate = nullptr;
      float* dev_up = nullptr;
      float* dev_gate_pair = nullptr;
      void* dev_q8_chain = nullptr;
      void* dev_q8_pair = nullptr;
      HIP_CHECK(hipMalloc(&dev_x, x_elems * sizeof(float)));
      HIP_CHECK(hipMalloc(&dev_gate, y_elems * sizeof(float)));
      HIP_CHECK(hipMalloc(&dev_up, y_elems * sizeof(float)));
      HIP_CHECK(hipMalloc(&dev_gate_pair, y_elems * sizeof(float)));
      const std::size_t q8_x_bytes =
          gufo::hip::QuantizedActivationBytes(batch, kPairHidden);
      const std::size_t q8_y_bytes =
          gufo::hip::QuantizedActivationBytes(batch, kPairFfn);
      HIP_CHECK(hipMalloc(&dev_q8_x, q8_x_bytes));
      HIP_CHECK(hipMalloc(&dev_q8_chain, q8_y_bytes));
      HIP_CHECK(hipMalloc(&dev_q8_pair, q8_y_bytes));
      HIP_CHECK(hipMemcpy(dev_x, host_x.data(), x_elems * sizeof(float),
                          hipMemcpyHostToDevice));

      const auto run_chain = [&] {
        gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dev_x, dev_q8_x, batch,
                                                        kPairHidden, stream);
        gufo::hip::LaunchMoeBucketQuantGEMM(bench_case.type, dev_gate_w,
                                            dev_q8_x, dev_gate, batch, kPairFfn,
                                            kPairHidden, stream);
        gufo::hip::LaunchMoeBucketQuantGEMM(bench_case.type, dev_up_w, dev_q8_x,
                                            dev_up, batch, kPairFfn,
                                            kPairHidden, stream);
        gufo::hip::LaunchBatchedFusedSwiGLUQuantizeQ8_1(
            dev_gate, dev_up, dev_q8_chain, batch, kPairFfn, stream);
      };
      const auto run_pair = [&] {
        gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dev_x, dev_q8_x, batch,
                                                        kPairHidden, stream);
        if (!gufo::hip::LaunchMoePairGateUpSwiGluQ8(
                bench_case.type, dev_gate_w, dev_up_w, dev_q8_x, dev_q8_pair,
                dev_gate_pair, batch, kPairFfn, kPairHidden, stream)) {
          std::printf("paired kernel: unsupported shape\n");
        }
      };
      const auto time = [&](const auto& body) {
        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < kPairIterations; ++i) {
          body();
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        return ElapsedUs(start, stop) / kPairIterations;
      };

      for (int warm = 0; warm < 3; ++warm) {
        run_chain();
        run_pair();
      }
      HIP_CHECK(hipStreamSynchronize(stream));

      // Correctness once per width, on the same buffers the timings used.
      run_chain();
      run_pair();
      HIP_CHECK(hipStreamSynchronize(stream));
      std::vector<std::byte> host_q8_chain(q8_y_bytes);
      std::vector<std::byte> host_q8_pair(q8_y_bytes);
      HIP_CHECK(hipMemcpy(host_q8_chain.data(), dev_q8_chain, q8_y_bytes,
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(host_q8_pair.data(), dev_q8_pair, q8_y_bytes,
                          hipMemcpyDeviceToHost));
      std::vector<float> host_gate(y_elems);
      std::vector<float> host_gate_pair(y_elems);
      HIP_CHECK(hipMemcpy(host_gate.data(), dev_gate, y_elems * sizeof(float),
                          hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(host_gate_pair.data(), dev_gate_pair,
                          y_elems * sizeof(float), hipMemcpyDeviceToHost));
      const auto diff = gufo::test::CompareTiledQ8_1(
          host_q8_chain.data(), host_q8_pair.data(), batch, kPairFfn);
      float max_gate = 0.0F;
      for (std::size_t i = 0; i < y_elems; ++i) {
        max_gate =
            std::max(max_gate, std::fabs(host_gate[i] - host_gate_pair[i]));
      }

      for (int run = 0; run < 2; ++run) {
        const float chain_us = time(run_chain);
        const float pair_us = time(run_pair);
        std::printf(
            "%6zu %5d %10.2f %11.1f %10.2f %11.1f %8.2fx %8d %12.3e %11.3e "
            "%11.3e\n",
            batch, run + 1, chain_us, bucket_gb / (chain_us * 1e-6), pair_us,
            bucket_gb / (pair_us * 1e-6), chain_us / pair_us, diff.max_abs_byte,
            static_cast<double>(diff.max_abs_scale),
            static_cast<double>(diff.max_abs_sum),
            static_cast<double>(max_gate));
      }

      HIP_CHECK(hipFree(dev_x));
      HIP_CHECK(hipFree(dev_q8_x));
      HIP_CHECK(hipFree(dev_gate));
      HIP_CHECK(hipFree(dev_up));
      HIP_CHECK(hipFree(dev_gate_pair));
      HIP_CHECK(hipFree(dev_q8_chain));
      HIP_CHECK(hipFree(dev_q8_pair));
    }
    HIP_CHECK(hipFree(dev_gate_w));
    HIP_CHECK(hipFree(dev_up_w));
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "pair") == 0) {
    RunPairGateUpBench();
    return 0;
  }
  const std::vector<Case> cases = {
      {"q4_k", gufo::core::GgmlType::kQ4_K},
      {"q5_k", gufo::core::GgmlType::kQ5_K},
      {"q8_0", gufo::core::GgmlType::kQ8_0},
  };
  const std::vector<std::size_t> batches = {16, 24, 32, 64, 96, 128, 192, 256};
  constexpr int kIterations = 20;

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  for (const Shape& shape : kShapes) {
    const std::size_t kExpertsWidth = shape.m;
    const std::size_t kHidden = shape.k;
    for (const Case& bench_case : cases) {
      const std::size_t row_bytes =
          ::gufo::quant::QuantizedRowBytes(bench_case.type, kHidden);
      if (row_bytes == 0) {
        std::printf("%s: unsupported type\n", bench_case.name);
        continue;
      }
      std::vector<std::byte> host_weight(row_bytes * kExpertsWidth);
      FillFinite(bench_case.type, host_weight);
      void* dev_weight = nullptr;
      HIP_CHECK(hipMalloc(&dev_weight, host_weight.size()));
      HIP_CHECK(hipMemcpy(dev_weight, host_weight.data(), host_weight.size(),
                          hipMemcpyHostToDevice));
      const double weight_gb =
          static_cast<double>(host_weight.size()) / (1024.0 * 1024.0 * 1024.0);

      std::printf("\n%s %s  N=%zu K=%zu  weight=%.1f MiB\n", bench_case.name,
                  shape.name, kExpertsWidth, kHidden,
                  static_cast<double>(host_weight.size()) / (1024.0 * 1024.0));
      std::printf("%6s %11s %11s %11s %10s %10s %10s %8s\n", "batch",
                  "exact_fp32", "q8_dense", "q8_moe", "exact GB/s",
                  "dense GB/s", "moe GB/s", "cos");

      for (const std::size_t batch : batches) {
        const std::size_t x_elems = batch * kHidden;
        const std::size_t y_elems = batch * kExpertsWidth;
        std::vector<float> host_x(x_elems);
        for (std::size_t i = 0; i < x_elems; ++i) {
          host_x[i] = 0.05F * std::sin(0.017F * static_cast<float>(i));
        }
        float* dev_x = nullptr;
        void* dev_q8 = nullptr;
        float* dev_y = nullptr;
        float* dev_y_ref = nullptr;
        HIP_CHECK(hipMalloc(&dev_x, x_elems * sizeof(float)));
        HIP_CHECK(hipMalloc(&dev_y, y_elems * sizeof(float)));
        HIP_CHECK(hipMalloc(&dev_y_ref, y_elems * sizeof(float)));
        HIP_CHECK(hipMalloc(
            &dev_q8, gufo::hip::QuantizedActivationBytes(batch, kHidden)));
        HIP_CHECK(hipMemcpy(dev_x, host_x.data(), x_elems * sizeof(float),
                            hipMemcpyHostToDevice));

        hipStream_t stream = nullptr;
        for (int warm = 0; warm < 3; ++warm) {
          gufo::hip::LaunchBatchedQuantGEMMFp32(bench_case.type, dev_weight,
                                                dev_x, dev_y_ref, batch,
                                                kExpertsWidth, kHidden, stream);
        }
        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < kIterations; ++i) {
          gufo::hip::LaunchBatchedQuantGEMMFp32(bench_case.type, dev_weight,
                                                dev_x, dev_y_ref, batch,
                                                kExpertsWidth, kHidden, stream);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const float exact_us = ElapsedUs(start, stop) / kIterations;

        for (int warm = 0; warm < 3; ++warm) {
          gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dev_x, dev_q8, batch,
                                                          kHidden, stream);
          gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
              bench_case.type, dev_weight, dev_q8, dev_y, batch, kExpertsWidth,
              kHidden, stream);
        }
        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < kIterations; ++i) {
          gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dev_x, dev_q8, batch,
                                                          kHidden, stream);
          gufo::hip::LaunchBatchedQuantGEMMPreQuantized(
              bench_case.type, dev_weight, dev_q8, dev_y, batch, kExpertsWidth,
              kHidden, stream);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const float q8_us = ElapsedUs(start, stop) / kIterations;

        float* dev_y_moe = nullptr;
        HIP_CHECK(hipMalloc(&dev_y_moe, y_elems * sizeof(float)));
        for (int warm = 0; warm < 3; ++warm) {
          gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dev_x, dev_q8, batch,
                                                          kHidden, stream);
          gufo::hip::LaunchMoeBucketQuantGEMM(bench_case.type, dev_weight,
                                              dev_q8, dev_y_moe, batch,
                                              kExpertsWidth, kHidden, stream);
        }
        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < kIterations; ++i) {
          gufo::hip::LaunchQuantizeActivationQ8_1FromFp32(dev_x, dev_q8, batch,
                                                          kHidden, stream);
          gufo::hip::LaunchMoeBucketQuantGEMM(bench_case.type, dev_weight,
                                              dev_q8, dev_y_moe, batch,
                                              kExpertsWidth, kHidden, stream);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const float moe_us = ElapsedUs(start, stop) / kIterations;

        // The production pattern is one launch per expert, a few hundred per
        // layer, so a launch only ever has that bucket's blocks in flight. This
        // measures whether that concurrency ceiling is what holds the weight
        // stream: the same 256 launches over 1..8 streams, disjoint outputs.
        constexpr int kBucketCount = 256;
        float* dev_y_ms = nullptr;
        HIP_CHECK(hipMalloc(&dev_y_ms, 8 * y_elems * sizeof(float)));
        hipStream_t side[7] = {};
        for (auto& s : side) {
          HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
        }
        std::printf("%6zu buckets256 GB/s:", batch);
        for (int streams = 1; streams <= 8; streams *= 2) {
          const int per = kBucketCount / streams;
          const auto run = [&] {
            for (int s = 0; s < streams; ++s) {
              hipStream_t st = (s == 0) ? stream : side[s - 1];
              float* y = dev_y_ms + (static_cast<std::size_t>(s) * y_elems);
              for (int i = 0; i < per; ++i) {
                gufo::hip::LaunchMoeBucketQuantGEMM(bench_case.type, dev_weight,
                                                    dev_q8, y, batch,
                                                    kExpertsWidth, kHidden, st);
              }
            }
          };
          run();
          HIP_CHECK(hipDeviceSynchronize());
          const auto t0 = std::chrono::steady_clock::now();
          run();
          HIP_CHECK(hipDeviceSynchronize());
          const double us = std::chrono::duration<double, std::micro>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
          const double gb = kBucketCount * weight_gb;
          std::printf(" %dx=%.1f", streams, gb / (us * 1e-6));
        }
        std::printf("\n");
        for (auto& s : side) {
          HIP_CHECK(hipStreamDestroy(s));
        }
        HIP_CHECK(hipFree(dev_y_ms));

        std::vector<float> host_y(y_elems);
        std::vector<float> host_ref(y_elems);
        HIP_CHECK(hipMemcpy(host_y.data(), dev_y, y_elems * sizeof(float),
                            hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(host_ref.data(), dev_y_ref, y_elems * sizeof(float),
                            hipMemcpyDeviceToHost));
        double dot = 0.0;
        double ref_norm = 0.0;
        double q8_norm = 0.0;
        for (std::size_t i = 0; i < y_elems; ++i) {
          dot += static_cast<double>(host_y[i]) * host_ref[i];
          ref_norm += static_cast<double>(host_ref[i]) * host_ref[i];
          q8_norm += static_cast<double>(host_y[i]) * host_y[i];
        }
        const double cosine =
            dot / std::sqrt(std::max(ref_norm * q8_norm, 1e-30));

        std::printf("%6zu %9.1fus %9.1fus %9.1fus %10.1f %10.1f %10.1f %8.5f\n",
                    batch, exact_us, q8_us, moe_us,
                    weight_gb / (exact_us * 1e-6), weight_gb / (q8_us * 1e-6),
                    weight_gb / (moe_us * 1e-6), cosine);

        HIP_CHECK(hipFree(dev_x));
        HIP_CHECK(hipFree(dev_q8));
        HIP_CHECK(hipFree(dev_y));
        HIP_CHECK(hipFree(dev_y_ref));
        HIP_CHECK(hipFree(dev_y_moe));
      }
      HIP_CHECK(hipFree(dev_weight));
    }
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
