# Architecture

How Kimi K3 maps onto this codebase. The model's own technical report is in
[kimi-k3-tech-report.pdf](kimi-k3-tech-report.pdf); section numbers below refer to it.

## The model

| | |
|---|---|
| Parameters | 2.78T total, ~104B active per token |
| Layers | 93: 69 KDA + 24 Gated MLA. Layer 0 has a dense FFN instead of a MoE block, so 92 layers route. |
| Hidden | 7168 |
| Attention heads | 96 |
| Routed experts | 896 per layer, top-16 selected |
| Shared experts | 2, full width |
| Latent MoE width | 3584 |
| Expert intermediate | 3072 |
| Vocabulary | 163,840 |
| Activation | SiTU-GLU (β₁ = 4, β₂ = 25) |
| Expert weights | native MXFP4 |

Every one of these is **read from the checkpoint's own `config.json`** at startup, not
hardcoded. The reader (`include/k3/k3_cfg.h`) refuses to substitute a default for a
missing field, because a config it half-understands produces a model that runs and is
architecturally wrong.

The MLA layers sit at one-based positions 4, 8, 12, … 92, **and 93**, the last two
layers are both MLA, so the final layer always performs global attention (§2.1).

## Where the memory goes

| component | size | residency |
|---|---:|---|
| Routed experts | 1.45 TB | never resident, streamed, MXFP4 |
| Dense trunk | 108.81 GB | resident *or* streamed |
| Embed + final norm + lm_head | 4.70 GB | always resident |
| Recurrent state (93 layers) | 626 MB | always resident |
| MLA KV cache | 2.37 MB/position | only with `--incremental` |

This table is the whole design. The 1.45 TB never lands in RAM; the 108.81 GB is the
dial; the 5.3 GB is the floor.

## Module map

```
include/k3/
  k3.h            public types, config, and the three invariants that must hold
  k3_cfg.h        config reader, never defaults a missing field

src/core/
  k3_ops.c        every kernel: RMSNorm, SiTU-GLU, ShortConv, KDA recurrence,
                  Gated MLA, AttnRes, MoE, MXFP4 matmul, bf16 matmul

src/io/
  k3_st.c         safetensors reader, hand-written, no dependency
  k3_load.c       one coalesced pread per expert
  k3_trunk.c      trunk streaming: ring buffer plus a pinned prefix

src/cache/
  k3_cache.c      routed-expert LRU with batch prefetch

src/model/
  k3_bind.c       binds tensors by name onto layer structures

src/tokenizer/
  k3_tok.h        loads the released tiktoken.model directly

src/chat/
  k3_chat.c       strict JSONL transcripts and official XTML segment rendering
  k3_sampler.c    deterministic PCG32 temperature/top-p sampling

src/cli/
  k3_run.c        batch CLI plus the text-only chat REPL
```

## Attention: two mechanisms

**KDA** (69 layers, §2.1.1) is a linear-attention recurrence with a channel-wise forget
gate. Per layer: separate q/k/v projections → ShortConv with fused SiLU → L2Norm on q and
k only → per-head β from a sigmoid → decay `g = g_min · sigmoid(e^A · z)` with
`g_min = −5` and **A indexed per head** → the recurrence → head-wise RMSNorm → a
full-rank output gate.

The recurrence order is load-bearing: decay the state, read from it, write the delta,
*then* read the output from the already-updated state.

**Gated MLA** (24 layers, §2.1.2) compresses keys and values into a low-rank latent. It
uses **NoPE**, no positional encoding at all, yet the 64 rope dimensions still exist
and are still cached. Only the rotation is absent. Dropping the slots would change the
head width from 192 to 128 and silently produce a different model.

## Attention Residuals (§2.2)

Instead of accumulating one residual through depth, each layer attends over the outputs
of preceding *blocks*. Layers are partitioned into blocks of 12; at a block boundary the
running residual is snapshotted and cleared. The token embedding is always the first
source, because layer 0 is itself a boundary.

## Stable LatentMoE (§2.3)

The routed path projects down to a 3584-wide latent, dispatches to 16 of 896 experts,
RMSNorms the **aggregate** (not each expert), then projects back to full width. Two
shared experts process the input at full width and are added unweighted.

The router computes independent sigmoid scores; they do not sum to 1, and a frozen
per-expert bias steers **selection only**. Combining weights come from the *unbiased*
scores. Using the biased scores for the weights still routes to the same experts and only
perturbs the mixture, which is exactly why it is easy to get wrong and hard to notice.

## MXFP4 experts

Expert weights ship in OCP MX FP4 and are **never dequantised**. `k3_matmul_mxfp4`
consumes packed nibbles directly:

```
value = E2M1[nibble] · 2^(E8M0_scale − 127)
```

with one shared 8-bit exponent per 32 elements. One expert is 33,030,144 parameters in
17,547,264 bytes, 0.53125 bytes per weight.

Dequantising would turn a 17.5 MB expert into 132 MB, and a token touches 1,472 of them
(92 routing layers × top-16): 194 GB per token of pure widening. The whole streaming
design depends on not doing that.

Nibble order is a convention, not a rule: the low nibble is the even element. Reversing
it yields a matrix with the right values in the wrong places: every statistic looks
correct and the model is wrong. There is a fixture for exactly this.

## Trunk streaming

Each layer's trunk tensors form one contiguous run inside its shard, so
`tools/pack_trunk.py` copies them into a single file as 93 sequential range copies, and
loading a layer later is one `pread` from a known offset.

The trunk is walked in the same fixed order every token, which makes prefetch perfect and
lets the read overlap compute. Given enough budget, layers are *pinned* rather than
cycled. A cyclic scan defeats LRU, so a pinned prefix is used instead.

## Correctness

- **op level**: every kernel against reference values at a declared tolerance
- **layer level**: all 93 layers of the released checkpoint individually
- **model level**: teacher forcing, greedy decode, and incremental decode must all match
  a reference exactly on a tiny model with the same tensor graph
- **numeric level**: logits from the released 93-layer checkpoint compared elementwise
  against a torch reference

See [TESTING.md](TESTING.md).
