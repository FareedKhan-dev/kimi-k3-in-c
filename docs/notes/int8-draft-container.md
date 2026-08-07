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

## Expected

56.7 GB resident draft, decode drafting at RAM speed while the streamed bf16 verify amortises
across accepted tokens. On the 64 GB band this is the difference between the hybrid being a
proven idea and a measured 1.4-1.6x, bit-exact. It is a contained kernel-plus-format job, not
a research risk; it is deferred here only to keep this change set shippable and reviewed.
