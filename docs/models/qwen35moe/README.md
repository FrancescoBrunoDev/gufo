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

## Prefill throughput: expert bucketing and weight sharing

The first working prefill routed per token: for every token and every selected
expert it issued three single-row GEMVs. Prompt processing was therefore flat in
the prompt length (44.1 / 44.4 / 43.4 tok/s at 128 / 512 / 2048 tokens), the
signature of a launch- and traffic-bound path rather than a compute-bound one.
Each of those GEMVs also streamed a whole 1-3 MiB expert matrix for a single
token.

Three mechanisms, each measured on its own:

1. **Bucketing** (`MoeFfnForwardRowsBucketed`). Tokens that picked the same
   expert are packed (`LaunchMoeGatherRows`) and each projection becomes one
   batched GEMM (`LaunchBatchedQuantGEMMFp32`), scattered back with the routed
   weight still on device (`LaunchMoeScatterAccumulate`). One host read of the
   ids per layer, not per row.
2. **Sixteen-row weight sharing.** `LaunchBatchedQuantGEMMFp32` split any bucket
   wider than eight rows into `ceil(count / 8)` passes, and every pass re-streamed
   the whole expert. An expert bucket has no 27B shape, so it fell through every
   specialized branch. Q4_K/Q5_K buckets of exactly sixteen rows now take
   `LaunchKQuantSixteenWave64`, one stream per bucket. Measured in isolation at
   N=1024/K=3072: 29.7 us against 41.0 us for two eight-row passes (Q4_K), 29.8
   against 33.0 (Q5_K). IQ4_XS was measured at 76.6 us there, slower than its two
   passes, so it keeps the eight-row split; this artifact's experts are all
   Q4_K/Q5_K, so nothing is lost.
3. **Batched shared expert.** The shared expert is a dense FFN, and it was still
   run one row at a time, which re-read its three Q8_0 matrices for every token.
   From sixteen rows up it now takes the tiled Q8_1 prefill route the dense FFN
   already uses (`LaunchQuantizeActivationQ8_1FromFp32`, then the fused
   gate/up pair, the SwiGLU-quantize epilogue and the down projection), so each
   matrix is streamed once for the whole chunk. Narrower chunks keep the FP32
   batched path, which stays bit-identical to the per-row path. Only the scalar
   `sigmoid(gate)` remains per row: that weight is fp32, and the batched entry
   points return early for fp32.
   The small-row path is exact: `bucketed vs per-row rows=5 cosine=1`. The wide
   path quantizes the activations, so it is measured against the exact path at
   `rows=32 cosine=0.999921`, held to 0.9995 by
   `qwen_moe_real_weights_test`; this is the same numerics class the dense FFN
   prefill and llama.cpp use.

Measured on the artifact (`gufo bench`, gfx1151, RelWithDebInfo with assertions,
same run repeated twice):

| test | per-token routing | bucketed | + 16-row sharing | + batched shared expert | + matrix core | + expert-parallel | llama.cpp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pp128 | 44.1 | 131.3 | 135.9 | 157.4 | 217.3 | **321.4** | 157.4 |
| pp512 | 44.4 | 160.9 | 170.5 | 195.5 | 369.1 | **530.6** | 530.9 |
| pp2048 | 43.4 | 162.6 | 175.4 | 198.9 | 584.5 | **735.0** | - |
| pp4096 | - | - | - | - | 612.8 | **713.8** | - |
| tg8 | 19.5 | 19.6 | 19.6 | 19.6 | 19.6 | 19.6 | 21.6 |

The shared-expert step alone is worth 15% of prefill: skipping it entirely
measured pp512 224.7 and pp2048 230.4, so about half of what it still costs is
the per-row scalar gate and the Q8_1 quantize passes.

## Prefill: what the stairstep above is made of

Each column that moved is one measured decision, not a guess.

**Wide chunks take the matrix-core route.** `moe_expert_gemm_bench` times both
backends at the real expert shape (`N=1024, K=3072`). The exact FP32 kernel is
best at exactly one 16-row pass (57 GB/s) and then degrades linearly, because a
second 16 rows means a second read of the whole matrix: 14 GB/s at 64 rows. The
tiled Q8_1 matrix-core route holds 43 GB/s at every width, so buckets of 24 rows
or more use it, and buckets above 64 rows widen the token tile to 128 and 256 so
the weight is still streamed once. Narrow buckets keep the exact kernel, which
wins there (29.2 µs against 38.3 at 16 rows) and keeps the bit-exact activation
contract the operator test compares against.

**The router is read once per chunk, not once per row.** It is a `[256, 3072]`
fp32 matrix, 3.1 MB. Forming it row by row cost 1.6 GB per layer at pp512 --
more than every expert matrix in the layer combined -- so it now goes through one
fp32 `LaunchHipblasGEMM`. The same reasoning applies to the shared expert's
scalar gate, which is now one `[rows, 1]` GEMM.

**Routing, the gate scale and the bucketing are batched.** One routing kernel
per row plus one sigmoid-scale kernel per row was 2048 launches per layer at
pp2048 width; `LaunchMoeRouteRows` and `LaunchSigmoidScaledAddRows` make it one
each, worth 28% of pp512 and 35% of pp2048 on its own.

**Where the remaining time is.** Ablations with the model still running, so
every shape is unchanged. Skipping the routed expert GEMMs measures pp512 1455
tok/s and pp2048 892; skipping the whole mixture-of-experts FFN measures pp2048
1921, which puts 70% of the chunk inside that FFN (routed GEMMs, shared expert
and the gather/quantize/scatter plumbing). On the non-expert side, skipping the
attention measures pp2048 623.6 and skipping the DeltaNet recurrence 602.6
against 584.8, so after the tiled attention fix those two are 6% and 3%: nearly
all of the remaining non-expert time is the dense projections and the norms.

**The tiled attention kernel now covers this model's head layout.** The twelve
full-attention layers were running `prefill_baseline` above a 1024-token
context, because `IsTiledAttentionSupported` compared the head layout against
the Flash-Next constants (24 query heads, 4 KV heads) while this model has 32
and 2. The kernel only used those constants for head indexing, so the shape is
now a template argument and both layouts are instantiated; the WMMA route keeps
its own predicate, and the pack kernel keeps its plain name because the WMMA
route declares it. All twelve layers report `prefill_tiled` in the dispatch
telemetry, and a 1600-token prompt with forty greedy tokens produces a
continuation byte-identical to the forced-baseline run. pp2048 455.4 -> 582.7
and pp4096 612.8; pp512 is unchanged because the tiled path only applies above
1024 tokens of visible context.

Measure each width in its own run: the first prompt of a combined sweep reads
about 3x low here (60.96 against 217.4 for pp128), and it repeats as a clean
217.3-217.6 three times on its own.

**The ablation ledger at pp2048.** Each row below is one build with that phase
skipped and everything else untouched, so the shapes are identical. Total is
2.79 s for 2048 tokens.

| phase skipped | pp2048 | phase costs |
| --- | --- | --- |
| nothing | 735 | - |
| the three routed expert GEMMs | 1110 | **1.40 s** (51%) |
| the scatter back to the tokens | 782 | 0.17 s (6%), since replaced by one atomic-free combine |
| the shared expert | 770 | 0.13 s (5%) |
| the gather | 764 | 0.11 s (4%) |
| everything outside the MoE FFN | 1921 | 0.85 s (31%) |

The expert phase is no longer 1.40 s of weight streaming at 50 GB/s: at pp2048
the 256 buckets read 1.45 GB of expert weights per layer, and the gather,
the gate/up scratch, the down projection's Q8 activation and the output panel
add 0.79 GB more. 2.24 GB per layer in 27 ms is 83 GB/s, which is exactly what
`moe_expert_gemm_bench` measures for the four-stream pattern. The weight read is
therefore already at the rate the kernel can hold, and the remaining lever on
this phase is the scratch traffic, not the dequant: fusing the SwiGLU epilogue
into the up projection (-0.26 GB/layer) and the down projection's epilogue into
the accumulation (-0.39 GB/layer) are worth about 0.2 s and 0.3 s.

**The scatter became one combine.** Every expert launched its own scatter, 256
small kernels per layer that accumulated with an atomic per element, so a chunk
reached 12288 launches and 2.4 G atomics. The bucketing already knows where each
selection landed, so it also builds the reverse map from (row, slot) to packed
position and one kernel per layer sums a row's eight selections in slot order:
one block owns a whole output row, there is nothing to synchronise, and the
result no longer depends on the order the experts ran in. It is still
deterministic across the split and unsplit schedules
(`qwen_moe_real_weights_test` asserts `max|diff| = 0` at 5, 32 and 1024 rows) and
its order now matches the per-row path's, which moved `rows=5` from 5.96e-08 to
2.98e-08. Worth pp128 321 -> 340, pp2048 746 -> 779, pp4096 729 -> 747.

**The non-expert remainder is at the dense rate, not below it.** The old note
here claimed it ran at a third of the dense FFN's per-FLOP rate; that was an
arithmetic error from counting 50 M parameters per layer instead of the ~150 M
this model has (the DeltaNet q/k/v projections are 8192 wide). The 0.85 s covers
~29 TFLOP of Q8_0 GEMM at ~35 TFLOPS, the same rate the dense 27B path reaches
on this part, and a scout audit confirms every non-expert projection in this
artifact is Q8_0 and already takes the tiled Q8_1 route. There is no lever
there.

**Experts run four at a time.** A bucket GEMM is one launch of about 32 blocks,
and the part has 40 compute units, so a single stream leaves most of the part
idle no matter how many experts a layer has: 256 launches in a row still only
ever have one bucket's blocks resident. `moe_expert_gemm_bench` measures the
production pattern (256 buckets of the reported width, the same work spread over
1, 2, 4 and 8 streams):

| Q4_K gate/up | 16 rows | 32 | 64 | 128 |
| --- | --- | --- | --- | --- |
| one stream | 48.9 | 48.6 | 46.6 | 30.5 GB/s |
| four streams | 84.4 | 83.7 | 83.2 | 51.5 GB/s |
| eight streams | 84.7 | 83.8 | 83.2 | 51.5 GB/s |

Four streams is the knee; eight adds nothing. Benchmarking one bucket at a time
had hidden this, because the launch tail looked like the whole cost. Expert `e`
now runs its gather, quantize, gate/up, SwiGLU and down on stream `e % 4`
against slice `e % 4` of every per-expert scratch buffer, so no two experts that
run at the same time share memory, and consecutive experts that share a stream
also share a slice and therefore serialize. The scatters stay on the main stream
after a fence, in expert order, which keeps the accumulation order and the
result bit-identical to the unsplit path
(`qwen_moe_real_weights_test` asserts `max|diff| = 0` at 5, 32 and 1024 rows).

The expert output needed its own slot per packed row for this: with a shared
buffer, a later expert on the same stream overwrote the output of an earlier one
before the deferred scatter read it. The slots are the same packed layout the
gather already uses, so this costs no extra traffic.

**The shared expert moved in front of the experts, for a correctness reason.**
It reused slice zero and the expert-output buffer after the join, so its Q8_1
route was reading scratch that the expert pass had rewritten. The symptom was
exact for five rows, a 4% difference on one row of a 32-row chunk, and a few
unrelated rows at 1024. Running it first -- before any expert touches a slice,
writing its own result through `gathered` -- removes the aliasing and makes both
schedules bit-identical. It costs about 3% of pp2048 compared with the aliased
order, which is the price of the result no longer depending on what was left in
scratch. The ablation puts the whole shared expert at 0.13 s (5%), so
overlapping it with the expert phase on its own slice is worth at most that.

That split explains the pp curve, and the expert path is no longer the place to
look for a large win at short prompts: `moe_expert_gemm_bench` measures the
three projections of a sixteen-token bucket at 65 GB/s per byte read (gate and
up at 57, down at 69), against about 68 GB/s for llama.cpp. What is left is the
scratch traffic above (0.79 GB of the 2.24 GB a layer moves) and, behind it,
wider dequant: 83 GB/s against Flash-Next's reported 130.

`kernel_resources_test` guards the new instantiations: the same sweep briefly
moved the dense path's 128-token tile to 64 and that instantiation spilled 348
bytes of scratch, which the check caught.

Prefill is 7.3x (pp128) to 16.9x (pp2048) faster than the per-token routing and
scales with the prompt length, because a longer chunk gives every expert more
rows per weight stream. Generation is untouched, as it must be.

### What is left in prompt processing

The same expert matrix must be streamed once per sixteen-token group, so the
traffic floor at 512 tokens is about one pass over all 256 experts per layer
(~1.4 GB/layer, ~65 GB/chunk). The kernels do not reach it: the isolated
measurement above is 29.7 us for 1.77 MB, about 60 GB/s against the roughly
240 GB/s this part reaches on large buffers. The landing kernels
(`SmallBatchKQuantExactFp32GEMMKernel`, `SmallBatchQ8_0ExactFp32VecGEMMKernel`)
dequantize into scalar fp32 FMAs with no matrix core, stage a 40 KiB activation
tile and run one workgroup per 48 output rows, so they are bound by occupancy and
dequantization ALU, not by HBM. Closing the rest of the gap to llama.cpp needs an
MMA grouped GEMM over a token-sorted expert tile, which is a kernel project of
its own (`W8A8BlockedWmmaGEMMKernel` is the closest existing structure, but it
consumes Q8_1 activations and its 128-row x 16-token tile gives only eight
workgroups at M=1024).

`LaunchMoeRowDot`, the fp32 scalar-gate kernel added for the first shared-expert
attempt, was removed with it: the per-row GEMV is cheaper and still needed.

### Why the routing read is not the next thing to remove

Keeping the routing on device would remove one `hipStreamSynchronize` and one
ids copy per layer, 48 per chunk. At the measured 2.6 s for a 512-token chunk
that is roughly 1.4 ms, or 0.05%: not where the time goes. It also does not
explain the draft head losing 18%: a verification step spreads eight rows over
about fifty distinct experts, so it must stream about fifty expert matrices
(~290 MB/layer) to check eight candidates, against ~42 MB for one decode token.
Speculation on this MoE is bandwidth-bound and needs either a large acceptance
rate or expert weight residency; neither is a scheduling fix.

## Remaining work

1. **HIP MoE prefill**: bucketing, sixteen-row weight sharing and the batched
   shared expert landed. What is left is the MMA grouped GEMM described above,
   which is what keeps the kernels at about a quarter of both the arithmetic and
   the bandwidth ceiling.
3. **Weight residency**: `hip/model_loader.cpp` copies every GGUF region into
   anonymous memory before `hipHostRegister`. The source pages are dropped as
   they are copied, so residency is about 74 GiB rather than 2×; reuse
   `src/core/hip/committed_backing` (the DeepSeek path) only to skip that copy.
4. **Vision numeric reference**: `tools/qwen27b/vision_reference.py` still
   allowlists projector widths 2560/5120, so the ViT/merger values at 3072 are
   not compared against a HF reference; the equivalence rests on geometric
   identity and the row/width/identity checks above.
