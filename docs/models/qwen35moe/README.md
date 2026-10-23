# Ornith 1.5 122B (`qwen35moe`)

Hybrid SSM + Mixture-of-Experts Qwen3.5-family model. Gufo loads it and the CPU
reference path answers correctly; the HIP path still emits wrong tokens, so the
artifact is refused there until that is fixed.

## Status

| Piece | State |
| --- | --- |
| GGUF reading (3 shards, metadata-only shard, packed 3D experts) | done |
| `ModelConfig` extraction (`expert_*` keys) | done |
| Weight binding and validation (router, packed experts, shared expert) | done |
| CPU reference FFN (`moe.cpp`) + numerical test | done |
| HIP routing, weighted accumulate, gated shared expert | done, tested on gfx1151 |
| HIP executor wiring (decode, batched decode, prefill) | done |
| HIP end-to-end correctness for this artifact | **broken, see below** |
| MTP (`nextn`) draft head, vision (`mmproj-BF16`) | **not implemented** |

`hip::QwenGpuModel::CreateFromGguf` still refuses an MoE artifact, now because of
the unresolved numeric disagreement described under "Open bug" rather than a
missing FFN.

## Artifact

`unsloth/Ornith-1.5-122B-GGUF`, `UD-Q4_K_XL` + `mmproj-BF16.gguf`
(revision `745bd742473a6cdddb1c6f78fb9a3b3a9293bd41`). Measured from the GGUF:

- `general.architecture = qwen35moe`, 49 blocks = 48 trunk + 1 `nextn` (MTP)
- hidden 3072, 32 heads / 2 KV heads, head_dim 256, rotary 64, vocab 248320
- 36 Gated DeltaNet layers + 12 full-attention layers (`full_attention_interval 4`)
- `ssm.inner_size 8192`, `ssm.time_step_rank 64`, state 128, groups 16
- 256 experts, top-8, expert FFN width 1024, shared expert width 1024
- weights 74.23 GiB (Q4_K 29.11 · Q5_K 27.84 · Q8_0 8.54 · IQ4_XS 7.97 ·
  Q6_K 0.62 · F32 0.15) spread over three shards; the first shard holds
  metadata only and no tensors

Attention, SSM and `nextn` tensor names match Qwen3.8-27B, so only the FFN
differs.

## FFN semantics

For one token, with `x` the post-attention-norm input and `P` the expert count:

```
logits  = W_router · x                     # ffn_gate_inp, [hidden, P], F32
probs   = softmax(logits)                  # over ALL experts, fp32
top     = topk(probs, k = expert_used_count)
w       = top / sum(top)                   # norm_topk_prob, no routed scale
routed  = Σ w_i · down_i(silu(gate_i(x)) · up_i(x))
shared  = down_s(silu(gate_s(x)) · up_s(x))
out     = routed + sigmoid(gate_inp_shexp · x) · shared
```

The shared expert is gated by its own scalar sigmoid and never enters the routed
softmax. Expert `i` of `ffn_{gate,up}_exps [hidden, e_ffn, P]` and
`ffn_down_exps [e_ffn, hidden, P]` is the contiguous element run
`[i · hidden · e_ffn, (i + 1) · hidden · e_ffn)`, because the expert axis is the
outermost GGUF dimension. Every expert run is block aligned for the quantized
formats present here (Q4_K, Q5_K, Q6_K, Q8_0, IQ4_XS).

Sources: llama.cpp `qwen35moe.cpp` (`build_layer_ffn`) and the reference
`Qwen3NextSparseMoeBlock` in `modeling_qwen3_next.py`.

## Verified

- `tools`-free probes against the real 74 GiB artifact: all three shards open,
  config extracts as `layers=48 hidden=3072 ffn=1024 experts=256 used=8`,
  `IsValidQwen()` holds, and every per-layer MoE tensor validates, including the
  last expert slice of a packed 3D tensor.
- `tests/models/qwen/cpu/moe_test.cpp` (`ctest -R qwen_moe_test`) checks routing
  and the full FFN against an independently written reference.
- `tests/models/qwen/cpu/forward_test.cpp` still passes, so dense Qwen3.8-27B
  behaviour is unchanged.
- `ctest -R qwen_moe_gpu_test` on gfx1151 (ROCm 7.2.3 container): the HIP
  overload matches the CPU reference within 5e-4 and selects the same experts
  with the same renormalized weights.
- `tests/models/qwen/hip/moe_real_weights_test.cpp` compares HIP against the CPU
  reference on the artifact's own experts, layer by layer (see below).
- Hosted CPU suite 125/125 and the gfx1151 `gpu-fast` subset 59/59 stay green.

## Open bug: HIP output is wrong for this artifact

Measured, in order:

- `gufo prompt --cpu` on the artifact answers correctly (`The capital of Italy is`
  → ` Rome.`), so the parser, config, weight binding, layer wiring, SSM,
  attention and MoE math are all correct as written.
- The same prompt on the HIP backend emits digits (`1000.00`). It also fails for
  a one-token prompt, so the state is already wrong before steady-state decode.
- The dense Qwen3.8-27B run on the same container and GPU still answers
  correctly, so the engine and toolchain are sound.
- `ctest -R qwen_moe_real_weights_test` compares the MoE FFN against the CPU
  reference on the artifact's own weights, layer by layer: identical to 1e-7 for
  IQ4_XS, Q4_K, Q5_K and Q8_0, with the same routed expert ids and weights.
- Disabling the row-split recurrence, the fused SSM input projections, the
  fused QK-norm/RoPE/KV write, and the fp16 KV cache each left the output wrong.

A layer-by-layer CPU-versus-HIP comparison is now available:
`tests/models/qwen/hip/layer_diff_probe_test.cpp` dumps the last prompt token's
hidden state after each layer from either engine. Calibrated on the dense 27B
(CPU versus HIP stays at `cos ≈ 0.999997` for layers 0-3, so a dense model does
not amplify reference noise):

| layer | 122B `cpu·hip` | 27B `cpu·hip` |
| --- | --- | --- |
| 0 | 0.999996 | 0.999997 |
| 1 | 0.982247 | 0.999997 |
| 2 | 0.989638 | 0.999998 |
| 3 | 0.989154 | 0.999997 |

Layer 0 is inside the dense-noise band and the divergence starts at layer 1,
which is a second Gated DeltaNet layer running the same kernels as layer 0. The
hip output magnitudes also run low (layer 3: 1.901 versus 2.161). Forcing
`kv_cache_storage = kFp32` changes layer 1 by nothing, so this is not KV
precision. Since the MoE FFN and its router are exact on layer 1's own weights,
the remaining suspect is the SSM chain at
`hidden=3072 / head_count_kv=2 / ssm.time_step_rank=64`, most likely its
recurrent-state addressing for a second layer (layer 0 starts from zeroed state,
so an aliased or mis-strided slot would only show from layer 1 onward).
Enabling the SSM replay log and comparing the per-layer conv/DeltaNet state
bytes between the two engines is the next check.

## Remaining work

1. Fix the numeric disagreement above, then drop the `CreateFromGguf` guard.
2. **HIP MoE prefill**: expert bucketing plus a grouped GEMM; the dense prefill
   route table in `hip/kernels/prefill_quant_gemm.hip` is specialized for the
   27B shapes (5120/17408) and would fall back to generic paths here.
3. **Weight residency**: `hip/model_loader.cpp` copies every GGUF region into
   anonymous memory before `hipHostRegister`. The source pages are dropped as
   they are copied, so residency is about 74 GiB rather than 2×; reuse
   `src/core/hip/committed_backing` (the DeepSeek path) only to skip that copy.
4. **MTP** and **vision**, once the dense backbone runs.
