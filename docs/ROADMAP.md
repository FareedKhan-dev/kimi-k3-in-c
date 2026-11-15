# Roadmap

Ordered by value, with the reasoning stated so the order can be argued with.

## 1. Long prompts: what is left after 2.0.0

The part that made a long prompt impractical is done: MLA attention, quadratic in sequence
length and the one serial O(T^2) loop in the engine, runs across the 96 heads in parallel
since 2.0.0, bit-identical at any thread count. Two costs remain, and both are now the
things worth measuring.

Prefill is still one forward pass over the whole prompt, so its activations grow with it,
about 14 GB at 32k positions next to a 77.7 GB KV cache. Chunked prefill would bound that,
reusing the tail-prefill path GATE 3b already proves exact.

Every prompt token still re-reads the trunk from memory for its projections, about 109 GB
of bf16 per token, which a batched projection kernel would amortise across a chunk. The
batched bf16 kernel from #67 is the starting point, once a wall-clock number on a long
prefill shows it winning.

## 2. Re-run the published campaign under the replicating harness

The measured noise floor is 33%, and almost every published figure is a single sample.
The harness itself is done, `benchmarks/memory-ladder.sh` and `benchmarks/split-sweep.sh`
already default to 3 repeats and report mean, sd and spread. What remains is re-running
the 12 ladder rungs and the 12 splits under it and replacing the single-sample tables in
docs/data/ with replicated ones.

## 3. Thread scaling

`OMP_NUM_THREADS` has never been swept on this engine. The workload is I/O bound at low
memory budgets, so the useful thread count is probably well below the core count, and
on memory-bound workloads throughput often *declines* past a point. Unknown here.

## 4. SIMD in the KDA recurrence

The bf16 trunk matmul and the MXFP4 expert matmul already have hand-written AVX2 paths
(`src/core/k3_ops.c`), each written to reproduce the scalar reduction order exactly. The
KDA recurrence does not: it is still plain scalar C, and it is the largest remaining
un-vectorised kernel on the non-I/O path.

## 5. Sampling

Batch generation is greedy only. Chat has opt-in temperature, top-p and seed
(`--temperature`, `--top-p`, `--seed`), and is greedy until one of them is given. Keep
it that way: greedy decoding is what makes output identical across memory budgets, which
is a property the test suite depends on.

## 6. Chat retained-state equivalence

Text chat implements K3's official XTML formatting, JSONL restart, and, under
`--incremental`, retains the KDA/MLA state across REPL turns: the ids are recorded where
they are fed (`src/chat/k3_prefix.h`), and a turn whose rendered transcript begins with
exactly that record prefills only its tail. The equivalence gate that had to exist first
is GATE 3b of `tests/unit/k3_model.c`: turn 2 on top of turn 1's state must be
bit-identical -- logits, every KV row, the KDA state -- to a full prefill of the same
transcript. Any divergence (`/reset`, an edited history) starts over. Restart from a
`--history` file still re-prefills: no opaque state is serialised.

## 7. Vision

K3 is natively multimodal. The vision tower is 27 layers and ~0.4B parameters, small,
and its weights are ~0.9 GB, a fraction of a percent of the checkpoint. Self-contained
enough to be tractable.

## 8. Tools and serving

The message model leaves room for tool calls, but there is no tool execution or HTTP API.
Both remain deliberately late product surface.

## Explicitly not planned

**A precision dial for the trunk.** The trunk is streamed losslessly rather than
quantised, and that is a design decision, not an omission. Post-hoc int4 measures ~17%
mean relative weight error on K3 attention tensors against ~1% for int8; streaming
costs time, which more memory buys back, while rounding costs accuracy, which nothing
buys back.

**GPU support.** Out of scope for this project.
