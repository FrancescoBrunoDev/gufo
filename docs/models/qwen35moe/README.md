# Ornith 1.5 122B (`qwen35moe`)

Hybrid SSM + Mixture-of-Experts Qwen3.5-family model. Gufo loads it and both the
CPU reference path and the HIP path answer correctly.

## Status

| Piece | State |
| --- | --- |
| GGUF reading (3 shards, metadata-only shard, packed 3D experts) | done |
| `ModelConfig` extraction (`expert_*` keys) | done |
| Weight binding and validation (router, packed experts, shared expert) | done |
| CPU reference FFN (`moe.cpp`) + numerical test | done |
| HIP routing, weighted accumulate, gated shared expert | done, tested on gfx1151 |
| HIP executor wiring (decode, batched decode, prefill) | done |
| HIP end-to-end correctness for this artifact | done (stream-ordering fix) |
| MTP (`nextn`) draft head, CPU and HIP | done |
| Vision (`mmproj-BF16`) | done |

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

## Resolved bug: the HIP moe routing read was not stream ordered

The HIP run first answered `1000.00` instead of ` Rome.`, and the failure survived
every fused kernel being switched off. The cause was in the new MoE path, not in
the SSM or attention kernels:

`MoeFfnForwardRows` brought the routed expert ids back to the host with a
blocking `hipMemcpy` on the default stream. The arena stream is created with
`hipStreamNonBlocking` (`hip/arena.cpp`), so that copy was **not** ordered
against `LaunchMoeRoute` on the arena stream: each layer read the ids that the
*previous* layer had left in the buffer. Layer 0 came out right only because its
queue was deep enough to finish the routing first, which is why the damage began
exactly at layer 1 and then rode the residual stream. The same copy is on the
decode path, so both prefill and steady-state decode were wrong.

The fix is a `hipStreamSynchronize` before the copy. It does not add a dependency
the shortcut in that comment did not already have; the real removal is to stop
bringing expert selection to the host at all, which the comment records as the
follow-up.

Evidence, one token through `tests/models/qwen/hip/layer_diff_probe_test.cpp`
(CPU versus HIP, last prompt token):

| layer | before | after |
| --- | --- | --- |
| 0 | 0.99999595 | 0.99999595 |
| 1 | 0.98224720 | 0.99999702 |
| 2 | 0.98963810 | 0.99999815 |
| 3 | 0.98915430 | 0.99999666 |
| logits | -0.60555600 | 0.99992100 |

Two identical HIP runs now produce bit-identical layer dumps, where the routed
ids used to be a race. `gufo prompt --raw` on the artifact answers
`The capital of Italy is` → ` Rome.` and continues `The capital of France is Paris.`

Rules that made the search cheap, worth repeating:

- Calibrate before believing a difference: the dense 27B holds `cos ≈ 0.999997`
  between the engines, so a dense model does not amplify reference noise, and the
  122B only started diverging at layer 1.
- A cross-matrix of every CPU layer against every HIP layer rules out a
  one-layer offset in the tap capture.
- Feeding one engine's layer output into the other isolates a single layer: the
  CPU reproduced its own layer-1 result from the HIP layer-0 state to within
  2e-6, proving the HIP layer itself was wrong rather than an amplified input.
- Independent audits of the SSM and prefill routes found no shape bug (the
  quantized projections match the CPU reference to 1.8e-6 at every 122B shape,
  including the 64-row `ssm_alpha`/`ssm_beta`), which is what redirected the
  search to host/device ordering in the new code.

## MTP draft head

The draft machinery was already generic; four things in it assumed the dense
27B, and each is now derived instead of hardcoded:

- the draft block index was the literal `blk.64`, now `blk.<config.num_layers>`
  (48 here, and still 64 for the 27B, which stays byte-identical);
- the draft FFN was dense, so `mtp_reference.cpp` and `hip/mtp/executor.cpp` now
  branch on `IsMoe()` and call `MoeFfnForward` with router/expert scratch;
- `IsMatrixType` rejected this artifact's projection formats, so it now accepts
  the same set the target loader binds (IQ4_XS, Q4_K, Q5_K, Q8_0, ...);
- the HIP side repacked every MTP matrix to BF16. The expert tensors are copied
  quantized instead: 2.78 GB packed, 2.26 GB of BF16 conversions avoided, 0.56 s
  to pack.

Measured on the artifact (one draft step, CPU versus HIP): hidden cosine
0.998126, RMSE 0.139, max absolute error 0.570, selected logit error 0.348.
The cosine sits below the dense 27B's 0.999 because a 256-expert router can flip
one expert on a near tie; that does not weaken the feature, since the verifier
accepts a draft only when it matches the target's own argmax
(`speculative_verifier.cpp`). End to end with `--speculative mtp`:

```
[Speculative]: acceptance=0.368932 drafted=103 accepted=38 verification_steps=25
Generated 64 tokens on GPU in 5.01578s (12.7597 tok/s)
```

The greedy 64-token completion is **token-for-token identical** with and without
MTP, which is the property that matters.

## Vision

The sidecar is geometrically the 27B's ViT: 27 blocks, `image_size 768`,
`patch_size 16`, `embedding_length 1152`, `ffn 4304`, 16 heads, merge 2,
`qwen3vl_merger`, no deepstack. Only the projection width differs (5120 -> 3072),
so no kernel changed. Three gates hardwired that width and now derive it from the
GGUF:

| site | before | after |
| --- | --- | --- |
| `vision/encoder.hip` | `width` allowlist `5120`/`2560` | `clip.vision.projection_dim` must equal the model width, cross-checked against `mm.2.weight` |
| `serve/inference_backend.cpp` | `if (embedding_length == 5120)` | width from `<architecture>.embedding_length` |
| `cli/prompt/prompt.cpp` | same | same |

A mismatched sidecar is now refused while opening, before any upload, instead of
failing in the prefill splice. Encoder evidence on the real mmproj: 256x256 ->
`64x3072` rows (8x8 placeholders), 371x193 resized to 384x192 -> `72x3072`,
projector identity `sha256 a5ff1592...`, and `<|image_pad|>` 248056 matches
`kImageToken` for this vocabulary. End to end on the artifact, a red circle with
a blue square answers "a large red circle, and inside it a smaller blue square,
centered".

## Remaining work

1. **HIP MoE prefill**: expert bucketing plus a grouped GEMM; the dense prefill
   route table in `hip/kernels/prefill_quant_gemm.hip` is specialized for the
   27B shapes (5120/17408) and would fall back to generic paths here.
3. **Weight residency**: `hip/model_loader.cpp` copies every GGUF region into
   anonymous memory before `hipHostRegister`. The source pages are dropped as
   they are copied, so residency is about 74 GiB rather than 2×; reuse
   `src/core/hip/committed_backing` (the DeepSeek path) only to skip that copy.
4. **Vision numeric reference**: `tools/qwen27b/vision_reference.py` still
   allowlists projector widths 2560/5120, so the ViT/merger values at 3072 are
   not compared against a HF reference; the equivalence rests on geometric
   identity and the row/width/identity checks above.
