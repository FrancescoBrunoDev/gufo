// MoE expert projection throughput at the Ornith 1.5 122B shape.
//
// The bucketed prefill calls LaunchBatchedQuantGEMMFp32 once per expert bucket,
// so its cost is set by this entry point at N=1024, K=3072 and the bucket sizes
// an actual chunk produces. This harness reports the achievable weight
// bandwidth and arithmetic throughput for those exact shapes.
//
//   nix develop -c tools/bench/build.sh tools/qwen35moe/moe_expert_gemm_bench.hip
//   /tmp/moe_expert_gemm_bench iq4 1024 3072 16 [iterations=20]
#define ENGINE_ENABLE_HIP 1
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/kernels/prefill_quant_gemm.hip"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::test::DeviceBuffer;

__device__ unsigned Hash(unsigned value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void FillBytes(unsigned char* data, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = static_cast<unsigned char>(Hash(static_cast<unsigned>(i)));
  }
}

__global__ void FillFloats(float* data, std::size_t count) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = (static_cast<int>(Hash(static_cast<unsigned>(i)) % 20001) - 10000) *
              0.00027F;
  }
}

struct Event {
  Event() { HIP_CHECK(hipEventCreate(&event)); }
  ~Event() { (void)hipEventDestroy(event); }
  hipEvent_t event{};
};

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      throw std::invalid_argument(
          "usage: moe_expert_gemm_bench iq4|q4|q5|q8 N K batch [iterations=20]");
    }
    const std::string name = argv[1];
    const std::size_t n = std::stoul(argv[2]);
    const std::size_t k = std::stoul(argv[3]);
    const std::size_t batch = std::stoul(argv[4]);
    const unsigned iterations = argc > 5 ? std::stoul(argv[5]) : 20;

    GgmlType type = GgmlType::kQ8_0;
    if (name == "iq4") {
      type = GgmlType::kIQ4_XS;
    } else if (name == "q4") {
      type = GgmlType::kQ4_K;
    } else if (name == "q5") {
      type = GgmlType::kQ5_K;
    } else if (name != "q8") {
      throw std::invalid_argument("expected iq4, q4, q5 or q8");
    }

    // One expert slice: the packed byte size of an N x K matrix.
    const std::size_t raw_bytes =
        static_cast<std::size_t>(n) * k * 4 + 64 * 1024 * 1024;
    DeviceBuffer<unsigned char> weights(raw_bytes);
    DeviceBuffer<float> input(batch * k), output(batch * n);
    FillBytes<<<(raw_bytes + 255) / 256, 256>>>(weights.data(), raw_bytes);
    FillFloats<<<(input.size() + 255) / 256, 256>>>(input.data(),
                                                    input.size());
    HIP_CHECK(hipDeviceSynchronize());

    const std::size_t weight_bytes = raw_bytes;
    const auto launch = [&] {
      gufo::hip::LaunchBatchedQuantGEMMFp32(type, weights.data(), input.data(),
                                            output.data(), batch, n, k, nullptr);
    };
    for (unsigned i = 0; i < 3; ++i) {
      launch();
    }
    HIP_CHECK(hipDeviceSynchronize());
    Event begin, end;
    HIP_CHECK(hipEventRecord(begin.event));
    for (unsigned i = 0; i < iterations; ++i) {
      launch();
    }
    HIP_CHECK(hipEventRecord(end.event));
    HIP_CHECK(hipEventSynchronize(end.event));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, begin.event, end.event));
    ms /= static_cast<float>(iterations);

    // Traffic of one weight pass; the ratio to the measured time says whether
    // the kernel keeps the weight in flight once per bucket.
    const double gbs = static_cast<double>(weight_bytes) / (ms * 1.0e6);
    const double gflops =
        2.0 * static_cast<double>(batch) * n * k / (ms * 1.0e6);
    std::printf(
        "type=%s N=%zu K=%zu batch=%zu weight_bytes=%zu ms=%.4f "
        "weight_GB/s=%.1f TFLOP/s=%.3f\n",
        name.c_str(), n, k, batch, weight_bytes, ms, gbs, gflops / 1000.0);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
