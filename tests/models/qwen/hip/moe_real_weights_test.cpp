#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/hip/ops/gemm.hpp"
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
      host_layer, config, std::span<const float>(x),
      std::span<float>(cpu_logits), std::span<float>(cpu_gate),
      std::span<float>(cpu_up), std::span<float>(cpu_act),
      std::span<float>(cpu_expert_out), std::span<float>(cpu_out));

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
            << " e_ffn=" << expert_ffn << " inter=" << config.intermediate_size
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
  // The bucketed prefill must agree with the per-row path on identical input.
  // Wide chunks take the tiled Q8_1 shared-expert route, which quantizes
  // activations, so that width is held to a looser bound than the exact small
  // one.
  for (const auto [kRows, min_cosine] :
       {std::pair<std::size_t, float>{5, 0.999F},
        std::pair<std::size_t, float>{32, 0.9995F},
        std::pair<std::size_t, float>{1024, 0.9995F}}) {
    using gufo::models::qwen::kMaxSelectedExperts;
    std::vector<float> rows_x(kRows * hidden);
    for (std::size_t r = 0; r < kRows; ++r) {
      for (std::size_t i = 0; i < hidden; ++i) {
        // Distinct rows route to different experts, which keeps most buckets
        // small; a shared scaling would give every row the same eight.
        rows_x[(r * hidden) + i] =
            x[i] + (0.35F * static_cast<float>(r) *
                    std::cos(0.01F * static_cast<float>(i) *
                             (1.0F + static_cast<float>(r))));
      }
    }
    gufo::test::DeviceBuffer<float> dx(rows_x);
    gufo::test::DeviceBuffer<float> dlogits(kRows * experts);
    gufo::test::DeviceBuffer<std::uint32_t> dids(kRows * kMaxSelectedExperts);
    gufo::test::DeviceBuffer<float> dweights(kRows * kMaxSelectedExperts);
    // Sized for the expert-parallel path, which gives each of its streams a
    // private slice of every per-expert buffer.
    constexpr std::size_t kSlices = gufo::hip::kMoePrefillSlices;
    gufo::test::DeviceBuffer<float> dgate(kSlices * kRows * expert_ffn);
    gufo::test::DeviceBuffer<float> dup(kSlices * kRows * expert_ffn);
    gufo::test::DeviceBuffer<float> dact(kSlices * kRows * expert_ffn);
    // One slot per packed expert row, as the arena allocates it.
    gufo::test::DeviceBuffer<float> dexpert_out(kRows * kMaxSelectedExperts *
                                                hidden);
    gufo::test::DeviceBuffer<float> dout_rows(kRows * hidden);
    gufo::test::DeviceBuffer<float> dout_bucket(kRows * hidden);
    gufo::test::DeviceBuffer<float> dgathered(kSlices * kRows * hidden);
    gufo::test::DeviceBuffer<std::uint32_t> dperm_rows(kRows *
                                                       kMaxSelectedExperts);
    gufo::test::DeviceBuffer<std::uint32_t> dperm_slots(kRows *
                                                        kMaxSelectedExperts);
    gufo::test::DeviceBuffer<std::uint32_t> doffsets(experts + 1);
    gufo::test::DeviceBuffer<float> dgate_logits(kRows);
    const std::size_t shared_ffn =
        config.shared_expert_feed_forward_length != 0
            ? config.shared_expert_feed_forward_length
            : expert_ffn;
    const std::size_t q8_bytes = gufo::hip::QuantizedActivationBytes(
        std::max<std::size_t>(kRows, 16), std::max(hidden, shared_ffn));
    gufo::test::DeviceBuffer<std::byte> dq8(kSlices * q8_bytes);

    // Two schedules, same arithmetic: everything on the main stream, then the
    // expert-parallel one the arena uses (expert e on stream e % slices against
    // slice e % slices, scatters still sequential). Both must land on the
    // per-row reference.
    // Production uses a non-blocking stream for the main work; the legacy null
    // stream has implicit-synchronization rules that the side streams do not
    // take part in, so mirror production here.
    hipStream_t main_stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&main_stream, hipStreamNonBlocking));
    std::vector<hipStream_t> ring{main_stream};
    for (std::size_t i = 1; i < kSlices; ++i) {
      hipStream_t side = nullptr;
      HIP_CHECK(hipStreamCreateWithFlags(&side, hipStreamNonBlocking));
      ring.push_back(side);
    }
    std::vector<void*> ring_void;
    for (hipStream_t side : ring) {
      ring_void.push_back(reinterpret_cast<void*>(side));
    }
    std::vector<void*> events;
    for (std::size_t i = 0; i < kSlices; ++i) {
      hipEvent_t event = nullptr;
      HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
      events.push_back(reinterpret_cast<void*>(event));
    }
    std::vector<float> bucket_unsplit;
    for (const bool split : {false, true}) {
      gufo::models::qwen::MoeFfnForwardRowsBucketed(
          gufo::models::qwen::HipModuleContext{main_stream}, device_layer,
          config, kRows, std::span<const float>(dx.data(), kRows * hidden),
          std::span<float>(dlogits.data(), kRows * experts),
          std::span<std::uint32_t>(dids.data(), kRows * kMaxSelectedExperts),
          std::span<float>(dweights.data(), kRows * kMaxSelectedExperts),
          std::span<float>(dgate.data(), dgate.size()),
          std::span<float>(dup.data(), dup.size()),
          std::span<float>(dact.data(), dact.size()),
          std::span<float>(dexpert_out.data(), dexpert_out.size()),
          std::span<float>(dout_bucket.data(), kRows * hidden),
          std::span<float>(dgathered.data(), dgathered.size()),
          std::span<std::uint32_t>(dperm_rows.data(),
                                   kRows * kMaxSelectedExperts),
          std::span<std::uint32_t>(dperm_slots.data(),
                                   kRows * kMaxSelectedExperts),
          std::span<std::uint32_t>(doffsets.data(), experts + 1),
          std::span<std::byte>(dq8.data(), dq8.size()),
          std::span<float>(dgate_logits.data(), kRows), nullptr,
          split ? std::span<void*>(ring_void) : std::span<void*>{},
          split ? std::span<void*>(events) : std::span<void*>{});
      HIP_CHECK(hipStreamSynchronize(main_stream));
      HIP_CHECK(hipDeviceSynchronize());
      if (split) {
        const auto split_host = dout_bucket.CopyToHost();
        float schedule_max = 0.0F;
        for (std::size_t i = 0; i < split_host.size(); ++i) {
          schedule_max = std::max(schedule_max,
                                  std::fabs(split_host[i] - bucket_unsplit[i]));
        }
        std::cout << "expert-parallel schedule: rows=" << kRows
                  << " max|diff|=" << schedule_max << "\n";
        if (schedule_max > 1e-6F) {
          std::cout << "expert-parallel schedule changed the result\n";
          return 1;
        }
      } else {
        bucket_unsplit = dout_bucket.CopyToHost();
      }
    }
    for (void* event : events) {
      HIP_CHECK(hipEventDestroy(static_cast<hipEvent_t>(event)));
    }
    for (std::size_t i = 1; i < ring.size(); ++i) {
      HIP_CHECK(hipStreamDestroy(ring[i]));
    }
    HIP_CHECK(hipStreamDestroy(main_stream));
    gufo::models::qwen::MoeFfnForwardRows(
        gufo::models::qwen::HipModuleContext{}, device_layer, config, kRows,
        std::span<const float>(dx.data(), kRows * hidden),
        std::span<float>(dlogits.data(), kRows * experts),
        std::span<std::uint32_t>(dids.data(), kRows * kMaxSelectedExperts),
        std::span<float>(dweights.data(), kRows * kMaxSelectedExperts),
        std::span<float>(dgate.data(), kRows * expert_ffn),
        std::span<float>(dup.data(), kRows * expert_ffn),
        std::span<float>(dact.data(), kRows * expert_ffn),
        std::span<float>(dexpert_out.data(), kRows * hidden),
        std::span<float>(dout_rows.data(), kRows * hidden));
    HIP_CHECK(hipDeviceSynchronize());
    const auto rows_host = dout_rows.CopyToHost();
    const auto bucket_host = dout_bucket.CopyToHost();
    float bucket_max = 0.0F;
    float rows_norm = 0.0F;
    float dot = 0.0F;
    float bucket_norm = 0.0F;
    for (std::size_t i = 0; i < rows_host.size(); ++i) {
      bucket_max =
          std::max(bucket_max, std::fabs(rows_host[i] - bucket_host[i]));
      rows_norm += rows_host[i] * rows_host[i];
      bucket_norm += bucket_host[i] * bucket_host[i];
      dot += rows_host[i] * bucket_host[i];
    }
    const float cosine =
        dot / std::sqrt(std::max(rows_norm * bucket_norm, 1e-30F));
    std::cout << "bucketed vs per-row: rows=" << kRows
              << " max|diff|=" << bucket_max << " cosine=" << cosine << "\n";
    if (cosine < min_cosine) {
      std::cout << "bucketed prefill disagrees with the per-row path\n";
      return 1;
    }
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
