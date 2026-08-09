# Roadmap

Ordered by value, with the reasoning stated so the order can be argued with.

## 1. Chunked prefill, the highest-value missing piece

Prefill currently runs as a single forward pass over the whole prompt, and attention in
the 24 MLA layers is quadratic in sequence length. The practical effect: the context
ceiling is 32k tokens, but a 21k-token prompt does not complete in reasonable time.

Raising the ceiling was necessary and is done. This is the part that makes it usable.

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

## 5. Chat retained-state equivalence

Text chat now implements K3's official XTML formatting, JSONL restart, and safe complete
history re-prefill. The next optimization is retaining in-process KDA/MLA state across
REPL turns only after an equivalence gate proves that its token stream matches a full
re-prefill of the same transcript.

## 6. Vision

K3 is natively multimodal. The vision tower is 27 layers and ~0.4B parameters, small,
and its weights are ~0.9 GB, a fraction of a percent of the checkpoint. Self-contained
enough to be tractable.

## 7. Tools and serving

The message model leaves room for tool calls, but there is no tool execution or HTTP API.
Both remain deliberately late product surface.

## Explicitly not planned

**A precision dial for the trunk.** The trunk is streamed losslessly rather than
quantised, and that is a design decision, not an omission. Post-hoc int4 measures ~17%
mean relative weight error on K3 attention tensors against ~1% for int8; streaming
costs time, which more memory buys back, while rounding costs accuracy, which nothing
buys back.

**GPU support.** Out of scope for this project.
