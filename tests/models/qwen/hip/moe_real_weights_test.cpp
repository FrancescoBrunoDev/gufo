#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/moe.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "tests/models/qwen/hip/support/comparisons.hpp"
#include "tests/models/qwen/hip/support/device.hpp"
#include "tests/models/qwen/hip/support/device_buffer.hpp"

namespace {

using gufo::models::QwenLayerWeights;
using gufo::models::QwenTensorRef;

QwenTensorRef HostRef(const gufo::core::GgufReader& reader,
                      const std::string& name) {
  const auto* tensor = reader.FindTensor(name);
  gufo::test::Expect(tensor != nullptr, ("missing tensor " + name).c_str());
  std::size_t available = 0;
  const auto address = reinterpret_cast<std::uintptr_t>(tensor->data);
  for (const auto& region : reader.GetMappedRegions()) {
    const auto region_address = reinterpret_cast<std::uintptr_t>(region.data);
    if (address >= region_address && address - region_address < region.size) {
      available = region.size - (address - region_address);
      break;
    }
  }
  return QwenTensorRef{.data = tensor->data,
                       .type = tensor->type,
                       .num_elements = tensor->ElementCount(),
                       .available_bytes = available};
}

/// Deterministic input vector: every layer sees the same activation, so a
/// divergence between the CPU and GPU paths cannot come from the fixture.
std::vector<float> InputFixture(std::size_t hidden) {
  std::vector<float> x(hidden);
  for (std::size_t i = 0; i < hidden; ++i) {
    x[i] = (static_cast<float>((i * 37) % 101) - 50.0F) / 50.0F;
  }
  return x;
}

}  // namespace

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : std::getenv("GUFO_ORNITH_MODEL");
  if (path == nullptr) {
    std::cout << "GUFO_ORNITH_MODEL is unset; skipping MoE real-weight test.\n";
    return 77;
  }
  const int device_status = gufo::test::GateHipDevice(
      gufo::test::HipDeviceRequirement::kOptional, "Qwen MoE real-weight test");
  if (device_status != gufo::test::kHipTestSuccess) {
    return device_status;
  }

  std::string err;
  auto reader = gufo::core::GgufReader::OpenFile(path, &err);
  gufo::test::Expect(reader != nullptr, ("open failed: " + err).c_str());
  auto config_opt = reader->ExtractModelConfig(&err);
  gufo::test::Expect(config_opt.has_value(), ("config failed: " + err).c_str());
  const auto& config = *config_opt;
  gufo::test::Expect(config.IsMoe(), "artifact declares experts");
  const std::size_t hidden = config.hidden_size;
  const std::size_t experts = config.expert_count;
  const std::size_t used = config.expert_used_count;
  const std::size_t expert_ffn = config.expert_feed_forward_length;

  const std::uint32_t layer =
      argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 0U;
  const std::string prefix = "blk." + std::to_string(layer) + ".";
  QwenLayerWeights host_layer;
  host_layer.ffn_gate_inp = HostRef(*reader, prefix + "ffn_gate_inp.weight");
  host_layer.ffn_gate_exps = HostRef(*reader, prefix + "ffn_gate_exps.weight");
  host_layer.ffn_up_exps = HostRef(*reader, prefix + "ffn_up_exps.weight");
  host_layer.ffn_down_exps = HostRef(*reader, prefix + "ffn_down_exps.weight");
  host_layer.ffn_gate_inp_shexp =
      HostRef(*reader, prefix + "ffn_gate_inp_shexp.weight");
  host_layer.ffn_gate_shexp =
      HostRef(*reader, prefix + "ffn_gate_shexp.weight");
  host_layer.ffn_up_shexp = HostRef(*reader, prefix + "ffn_up_shexp.weight");
  host_layer.ffn_down_shexp =
      HostRef(*reader, prefix + "ffn_down_shexp.weight");

  const auto x = InputFixture(hidden);
  std::vector<float> cpu_logits(experts);
  std::vector<float> cpu_gate(expert_ffn);
  std::vector<float> cpu_up(expert_ffn);
  std::vector<float> cpu_act(expert_ffn);
  std::vector<float> cpu_expert_out(hidden);
  std::vector<float> cpu_out(hidden, 0.0F);
  gufo::models::qwen::MoeFfnForward(
      gufo::models::qwen::CpuModuleContext{}, host_layer, config,
      std::span<const float>(x), std::span<float>(cpu_logits),
      std::span<float>(cpu_gate), std::span<float>(cpu_up),
      std::span<float>(cpu_act), std::span<float>(cpu_expert_out),
      std::span<float>(cpu_out));

  // Upload the real quantized tensors and run the HIP path on the same input.
  auto upload = [&](const QwenTensorRef& ref) {
    const auto bytes = ref.EncodedSizeBytes();
    gufo::test::DeviceBuffer<std::uint8_t> buffer(bytes);
    HIP_CHECK(hipMemcpy(buffer.data(), ref.data, bytes, hipMemcpyHostToDevice));
    return buffer;
  };
  auto device_router = upload(host_layer.ffn_gate_inp);
  auto device_gate_exps = upload(host_layer.ffn_gate_exps);
  auto device_up_exps = upload(host_layer.ffn_up_exps);
  auto device_down_exps = upload(host_layer.ffn_down_exps);
  auto device_shexp_gate_in = upload(host_layer.ffn_gate_inp_shexp);
  auto device_shexp_gate = upload(host_layer.ffn_gate_shexp);
  auto device_shexp_up = upload(host_layer.ffn_up_shexp);
  auto device_shexp_down = upload(host_layer.ffn_down_shexp);

  auto device_ref = [](auto& buffer, gufo::core::GgmlType type,
                       std::size_t elements) {
    return QwenTensorRef{.data = buffer.data(),
                         .type = type,
                         .num_elements = elements,
                         .available_bytes = buffer.size()};
  };
  QwenLayerWeights device_layer;
  device_layer.ffn_gate_inp =
      device_ref(device_router, host_layer.ffn_gate_inp.type,
                 host_layer.ffn_gate_inp.num_elements);
  device_layer.ffn_gate_exps =
      device_ref(device_gate_exps, host_layer.ffn_gate_exps.type,
                 host_layer.ffn_gate_exps.num_elements);
  device_layer.ffn_up_exps =
      device_ref(device_up_exps, host_layer.ffn_up_exps.type,
                 host_layer.ffn_up_exps.num_elements);
  device_layer.ffn_down_exps =
      device_ref(device_down_exps, host_layer.ffn_down_exps.type,
                 host_layer.ffn_down_exps.num_elements);
  device_layer.ffn_gate_inp_shexp =
      device_ref(device_shexp_gate_in, host_layer.ffn_gate_inp_shexp.type,
                 host_layer.ffn_gate_inp_shexp.num_elements);
  device_layer.ffn_gate_shexp =
      device_ref(device_shexp_gate, host_layer.ffn_gate_shexp.type,
                 host_layer.ffn_gate_shexp.num_elements);
  device_layer.ffn_up_shexp =
      device_ref(device_shexp_up, host_layer.ffn_up_shexp.type,
                 host_layer.ffn_up_shexp.num_elements);
  device_layer.ffn_down_shexp =
      device_ref(device_shexp_down, host_layer.ffn_down_shexp.type,
                 host_layer.ffn_down_shexp.num_elements);

  gufo::test::DeviceBuffer<float> device_x(x);
  gufo::test::DeviceBuffer<float> device_logits(experts);
  gufo::test::DeviceBuffer<std::uint32_t> device_ids(
      gufo::models::qwen::kMaxSelectedExperts);
  gufo::test::DeviceBuffer<float> device_weights(
      gufo::models::qwen::kMaxSelectedExperts);
  gufo::test::DeviceBuffer<float> device_gate(expert_ffn);
  gufo::test::DeviceBuffer<float> device_up(expert_ffn);
  gufo::test::DeviceBuffer<float> device_act(expert_ffn);
  gufo::test::DeviceBuffer<float> device_expert_out(hidden);
  gufo::test::DeviceBuffer<float> device_out(hidden);

  gufo::models::qwen::MoeFfnForward(
      gufo::models::qwen::HipModuleContext{}, device_layer, config,
      std::span<const float>(device_x.data(), hidden),
      std::span<float>(device_logits.data(), experts),
      std::span<std::uint32_t>(device_ids.data(),
                               gufo::models::qwen::kMaxSelectedExperts),
      std::span<float>(device_weights.data(),
                       gufo::models::qwen::kMaxSelectedExperts),
      std::span<float>(device_gate.data(), expert_ffn),
      std::span<float>(device_up.data(), expert_ffn),
      std::span<float>(device_act.data(), expert_ffn),
      std::span<float>(device_expert_out.data(), hidden),
      std::span<float>(device_out.data(), hidden));
  HIP_CHECK(hipDeviceSynchronize());

  const auto gpu_out = device_out.CopyToHost();
  const auto gpu_ids = device_ids.CopyToHost();
  const auto gpu_weights = device_weights.CopyToHost();

  gufo::models::qwen::MoeSelection cpu_selection;
  gufo::models::qwen::MoeRoute(
      host_layer.ffn_gate_inp, std::span<const float>(x),
      static_cast<std::uint32_t>(experts), static_cast<std::uint32_t>(used),
      std::span<float>(cpu_logits), &cpu_selection);

  std::cout << "layer " << layer << " experts=" << experts << " used=" << used
            << " e_ffn=" << expert_ffn
            << " gate=" << gufo::core::ToString(host_layer.ffn_gate_exps.type)
            << "\n";
  std::cout << "cpu ids: ";
  for (std::uint32_t slot = 0; slot < used; ++slot) {
    std::cout << cpu_selection.experts[slot] << "("
              << cpu_selection.weights[slot] << ") ";
  }
  std::cout << "\ngpu ids: ";
  for (std::uint32_t slot = 0; slot < used; ++slot) {
    std::cout << gpu_ids[slot] << "(" << gpu_weights[slot] << ") ";
  }
  std::cout << "\n";

  float max_diff = 0.0F;
  float cpu_norm = 0.0F;
  for (std::size_t i = 0; i < hidden; ++i) {
    max_diff = std::max(max_diff, std::fabs(cpu_out[i] - gpu_out[i]));
    cpu_norm += cpu_out[i] * cpu_out[i];
  }
  std::cout << "cpu[0..5]: ";
  for (std::size_t i = 0; i < 6; ++i) {
    std::cout << cpu_out[i] << " ";
  }
  std::cout << "\ngpu[0..5]: ";
  for (std::size_t i = 0; i < 6; ++i) {
    std::cout << gpu_out[i] << " ";
  }
  std::cout << "\nmax |cpu-gpu| = " << max_diff
            << "  cpu l2 = " << std::sqrt(cpu_norm) << "\n";
  return max_diff <= 2e-3F * std::sqrt(cpu_norm) ? 0 : 1;
}
#else
int main() {
  std::cout << "HIP disabled; skipping MoE real-weight test.\n";
  return 77;
}
#endif
