#include "src/models/qwen/moe.hpp"

#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/models/qwen/hip/execution_policy.hpp"
#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

using gufo::models::QwenLayerWeights;
using gufo::models::QwenTensorRef;
using gufo::test::DeviceBuffer;

/// Deterministic fixture shared by the CPU and GPU runs.
std::vector<float> Fill(std::size_t count, std::uint32_t seed) {
  std::vector<float> values(count);
  std::uint32_t state = seed;
  for (auto& v : values) {
    state = state * 1664525U + 1013904223U;
    v = (static_cast<float>((state >> 8) & 0xFFFFU) / 65535.0F - 0.5F) * 2.0F;
  }
  return values;
}

QwenTensorRef HostRef(const std::vector<float>& data) {
  return QwenTensorRef{.data = data.data(),
                       .type = gufo::core::GgmlType::kF32,
                       .num_elements = data.size(),
                       .available_bytes = data.size() * sizeof(float)};
}

QwenTensorRef DeviceRef(const DeviceBuffer<float>& buffer) {
  return QwenTensorRef{.data = buffer.data(),
                       .type = gufo::core::GgmlType::kF32,
                       .num_elements = buffer.size(),
                       .available_bytes = buffer.size() * sizeof(float)};
}

}  // namespace

int main() {
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen MoE GPU ops test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  // The real 122B shapes per layer, scaled down in expert count so the fixture
  // fits in a test allocation, plus the exact top-k of the family.
  constexpr std::size_t kHidden = 64;
  constexpr std::size_t kExperts = 32;
  constexpr std::size_t kUsed = 8;
  constexpr std::size_t kExpertFfn = 48;

  gufo::core::ModelConfig config;
  config.hidden_size = kHidden;
  config.expert_count = kExperts;
  config.expert_used_count = kUsed;
  config.expert_feed_forward_length = kExpertFfn;
  config.shared_expert_feed_forward_length = kExpertFfn;
  config.intermediate_size = kExpertFfn;

  auto router_data = Fill(kExperts * kHidden, 3);
  auto gate_exps = Fill(kExperts * kExpertFfn * kHidden, 5);
  auto up_exps = Fill(kExperts * kExpertFfn * kHidden, 9);
  auto down_exps = Fill(kExperts * kHidden * kExpertFfn, 13);
  auto shexp_gate_in = Fill(kHidden, 17);
  auto shexp_gate = Fill(kExpertFfn * kHidden, 19);
  auto shexp_up = Fill(kExpertFfn * kHidden, 23);
  auto shexp_down = Fill(kHidden * kExpertFfn, 29);
  auto x_data = Fill(kHidden, 31);

  QwenLayerWeights host_layer;
  host_layer.ffn_gate_inp = HostRef(router_data);
  host_layer.ffn_gate_exps = HostRef(gate_exps);
  host_layer.ffn_up_exps = HostRef(up_exps);
  host_layer.ffn_down_exps = HostRef(down_exps);
  host_layer.ffn_gate_inp_shexp = HostRef(shexp_gate_in);
  host_layer.ffn_gate_shexp = HostRef(shexp_gate);
  host_layer.ffn_up_shexp = HostRef(shexp_up);
  host_layer.ffn_down_shexp = HostRef(shexp_down);

  // CPU reference.
  std::vector<float> host_logits(kExperts);
  std::vector<float> host_gate(kExpertFfn);
  std::vector<float> host_up(kExpertFfn);
  std::vector<float> host_act(kExpertFfn);
  std::vector<float> host_expert_out(kHidden);
  std::vector<float> expected(kHidden, 0.0F);
  gufo::models::qwen::MoeFfnForward(
      host_layer, config, std::span<const float>(x_data),
      std::span<float>(host_logits), std::span<float>(host_gate),
      std::span<float>(host_up), std::span<float>(host_act),
      std::span<float>(host_expert_out), std::span<float>(expected));

  // Same weights and inputs on device.
  DeviceBuffer<float> device_router(router_data);
  DeviceBuffer<float> device_gate_exps(gate_exps);
  DeviceBuffer<float> device_up_exps(up_exps);
  DeviceBuffer<float> device_down_exps(down_exps);
  DeviceBuffer<float> device_shexp_gate_in(shexp_gate_in);
  DeviceBuffer<float> device_shexp_gate(shexp_gate);
  DeviceBuffer<float> device_shexp_up(shexp_up);
  DeviceBuffer<float> device_shexp_down(shexp_down);
  DeviceBuffer<float> device_x(x_data);
  DeviceBuffer<float> device_logits(kExperts);
  DeviceBuffer<std::uint32_t> device_ids(
      gufo::models::qwen::kMaxSelectedExperts);
  DeviceBuffer<float> device_weights(gufo::models::qwen::kMaxSelectedExperts);
  DeviceBuffer<float> device_ffn_gate(kExpertFfn);
  DeviceBuffer<float> device_ffn_up(kExpertFfn);
  DeviceBuffer<float> device_act(kExpertFfn);
  DeviceBuffer<float> device_expert_out(kHidden);
  DeviceBuffer<float> device_out(kHidden);

  QwenLayerWeights device_layer;
  device_layer.ffn_gate_inp = DeviceRef(device_router);
  device_layer.ffn_gate_exps = DeviceRef(device_gate_exps);
  device_layer.ffn_up_exps = DeviceRef(device_up_exps);
  device_layer.ffn_down_exps = DeviceRef(device_down_exps);
  device_layer.ffn_gate_inp_shexp = DeviceRef(device_shexp_gate_in);
  device_layer.ffn_gate_shexp = DeviceRef(device_shexp_gate);
  device_layer.ffn_up_shexp = DeviceRef(device_shexp_up);
  device_layer.ffn_down_shexp = DeviceRef(device_shexp_down);

  gufo::models::qwen::MoeFfnForward(
      gufo::models::qwen::HipModuleContext{}, device_layer, config,
      std::span<const float>(device_x.data(), kHidden),
      std::span<float>(device_logits.data(), kExperts),
      std::span<std::uint32_t>(device_ids.data(),
                               gufo::models::qwen::kMaxSelectedExperts),
      std::span<float>(device_weights.data(),
                       gufo::models::qwen::kMaxSelectedExperts),
      std::span<float>(device_ffn_gate.data(), kExpertFfn),
      std::span<float>(device_ffn_up.data(), kExpertFfn),
      std::span<float>(device_act.data(), kExpertFfn),
      std::span<float>(device_expert_out.data(), kHidden),
      std::span<float>(device_out.data(), kHidden));

  const auto sync_status = hipDeviceSynchronize();
  if (sync_status != hipSuccess) {
    std::cerr << "MoE HIP sync failed: " << hipGetErrorString(sync_status)
              << " (last: " << hipGetErrorString(hipGetLastError()) << ")\n";
  }
  gufo::test::Expect(sync_status == hipSuccess, "MoE HIP launch completed");

  const auto actual = device_out.CopyToHost();
  gufo::test::ExpectSpanNear(std::span<const float>(expected),
                             std::span<const float>(actual), 5e-4F,
                             "HIP MoE FFN matches the CPU reference");

  // The routing decision itself must match too, not just the final sum.
  gufo::models::qwen::MoeSelection cpu_selection;
  gufo::models::qwen::MoeRoute(host_layer.ffn_gate_inp,
                               std::span<const float>(x_data), kExperts, kUsed,
                               std::span<float>(host_logits), &cpu_selection);
  const auto device_selected = device_ids.CopyToHost();
  const auto device_selected_weights = device_weights.CopyToHost();
  for (std::uint32_t slot = 0; slot < kUsed; ++slot) {
    gufo::test::Expect(device_selected[slot] == cpu_selection.experts[slot],
                       "HIP router selects the CPU expert order");
    gufo::test::ExpectNear(cpu_selection.weights[slot],
                           device_selected_weights[slot], 1e-5F,
                           "HIP router weight matches the CPU weight");
  }

  std::cout << "Qwen MoE GPU ops test passed on gfx1151.\n";
  return 0;
}
#else
int main() {
  std::cout << "HIP disabled, skipping Qwen MoE GPU ops test.\n";
  return 77;
}
#endif
