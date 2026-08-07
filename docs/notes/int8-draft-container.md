# Int8 draft container: the piece that makes the hybrid fast

## Where it fits

Hybrid decode (`--draft-trunk`) is implemented and proven correct: an int8-derived draft
proposes tokens, the exact bf16 model verifies them, and the emitted tokens are exactly the
exact model's greedy output. Measured teacher-forced agreement of the int8 derivation is
94.2% against a 96.2% ceiling, and a first flight on the real checkpoint accepted 66.7% of
drafts with byte-identical output.

What is missing is the SPEED half. The current draft trunk is stored in bf16 (109 GB): it is
a fidelity simulator, not a size-reduced container, so on a 64 GB machine it cannot be
resident and streams about half of itself per draft step. Measured, that made the hybrid
slower than the exact run at that budget. The draft only pays off when it is RESIDENT, and a
true int8 container is 56.7 GB, which fits the 64-110 GB band.

## The build

Three parts.

1. **Format.** `tools/pack_trunk.py --int8`: for every 2D bf16 trunk tensor, per-row symmetric
   absmax int8 (or per-group, group 128, if per-row quality is short). Store scales INLINE,
   one fp32 per row prepended to that row's int8 bytes, so a weight matrix stays a single
   tagged pointer and the kernel derives the scale from `W`. This avoids threading a parallel
   scale array through K3MlaW / K3KdaW / K3MoeW / K3LayerW. Manifest carries `dtype: "I8R"`.

2. **Kernel.** `k3_matmul_q8(y, x, W, in, out)` where each row is `[f32 scale][int8[in]]`:
   widen int8 to int32, multiply by fp32 activation, accumulate, scale once at the end. It
   does NOT need to be bit-identical to anything: the draft is only a proposal source, and
   the exact bf16 model is the sole authority on emitted tokens. So it can use the fastest
   AVX2 form (maddubs-style or cvt+fma) without the four-accumulator determinism contract.

3. **Dispatch.** Add `K3_WI8` to the wdt enum, one branch in `k3_mmw`, `k3_wsz` returns the
   per-row stride, and `k3_bind_layer_mem` recognises the `I8R` dtype from the trunk manifest.
   Only the DRAFT weights ever carry this tag; the exact model stays bf16, so no oracle gate
   and no exactness claim is touched.

## Gate

Pack the int8 container, confirm the draft's teacher-forced agreement is still ~94% (the
inline per-row scheme should match the qdq measurement), run the hybrid resident under a
64-68 GB cgroup cap, and require its s/token to beat the exact run at the same budget (19.8 on
the proof NVMe). Output identity against the exact greedy run is the correctness gate and is
already structural.

## Built, and what it measured

The container is built and works: `k3_matmul_q8` (unit-tested at rel-L2 0.37% vs fp32),
`tools/int8_trunk.py` (produces a 54.47 GB container, half the bf16 trunk), the K3_DT_I8R
dtype and the bind wiring (matmul weights pointed at directly and tagged K3_WI8; tensors read
elementwise as fp32, like the AttnRes projection, dequantised into the widen buffer). The
draft trunk loads, is resident (66 GB RSS on a 64 GB cap), and produces byte-identical output.
Its teacher-forced agreement measured 90.9% on a 22-token sample, in the same band as the qdq
simulator, and on one prompt the exact model accepted 100% of drafts (mean run 4.0).

But the resident hybrid measured 53 s/token against 19.8 for exact decode at the same 64 GB.
The reason is a real one and it is the important finding: **making the TRUNK resident does not
make a draft step cheap, because the draft still streams the routed EXPERTS.** The experts are
MXFP4 and 1.45 TB; they can never be resident, so every drafted token still reads ~25.8 GB of
experts, the same as a real decode step. A draft that costs as much as the thing it drafts for
cannot amortise anything. Splitting a 64 GB machine 56/2.5 between draft and exact also starved
the exact model's trunk budget, so its verify sweeps streamed the full bf16 trunk.

## Cache-only routing: built, and the economics measured

The fix was implemented: `K3ExpertSrc::resident` plus a `cache_only` MoE mode make the draft
route only among experts already in the cache and renormalise over them, so a drafted token
reads zero new expert bytes. Output stays bit-exact (the exact model verifies), and the
weightless gates are untouched.

Measured, it revealed the real tension. With cache-only routing and a small (4 GB) expert
cache, draft acceptance collapsed from 66-100% to 12.5%, because the cache held under 1% of
the experts so the draft routed among far too few and proposed badly. The draft became cheap
but useless. So the two dead ends are:

- full-routing draft: accurate, but streams all experts, so a draft step costs as much as a
  real one;
- cache-only draft: cheap, but accurate only when the cache holds most routed experts.

Both are resolved only in the same place: LARGE RAM. The hybrid pays when the 55 GB int8 draft
trunk is resident AND the expert cache is large enough (tens of GB) that cache-only routing
stays accurate AND the exact model still has a trunk pin. That is a 100 GB-plus machine, not a
laptop, and at 128 GB plain exact decode is already 5.59 s/token, so the hybrid's marginal gain
there is uncertain and unproven. It is not the laptop lever it was hoped to be.

## Status

The container, the q8 kernel, the I8R format, the bind wiring, and cache-only routing are all
built, correct, and bit-exact, and are a reusable foundation. The honest conclusion is that the
quantized-self-draft hybrid is a large-memory technique with narrow, unproven economics, not a
consumer-laptop speedup. The laptop wins are elsewhere: the fused kernels, RAM-first pinning,
chunk-union prefill, and conversation resume, all measured.

## What was hoped, before the measurements

The missing piece is a CHEAP draft on the expert side, so a proposal costs far less than a real
step. Two routes, both identity-safe because the draft only proposes:

- **cache-only routing:** the draft routes only among experts already resident in the cache,
  reading zero new expert bytes per draft token. The exact verify still uses true routing.
- **top-k reduction:** the draft uses top-4 of 896 instead of top-16, cutting draft expert
  reads 4x.

With either, a draft token costs a fraction of a real one, the resident int8 trunk carries the
draft's dense compute, and the batched bf16 verify amortises across accepted tokens. That is the
configuration where the 90.9% agreement and 100% acceptance measured here turn into a real
speedup. The container and kernel built here are the reusable half; the draft-expert reduction is
the next step, and it is a small change to the draft's routing call, not another format.
