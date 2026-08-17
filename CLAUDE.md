# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Kimi K3 (2.78T param MoE) inference engine in portable C99. No BLAS, no PyTorch/GPU
runtime — six C files plus headers, compiled to a ~176KB binary. Core trick: the 1.45TB
of routed experts stay on disk in packed 4-bit form and are streamed/cached on demand
(LRU), while a much smaller dense trunk is optionally pinned in RAM to a configurable
depth. Same checkpoint runs identically (byte-for-byte output) from 8GB RAM up to
fully-resident on 128GB+; only speed changes.

## Build & test

Build uses [Task](https://taskfile.dev) (`brew install go-task`), not Make — see
`Taskfile.yml`. `task --list` shows every task.

```bash
task build          # build bin/k3 (7 C files + OpenMP, seconds)
task test           # full weightless test suite — no checkpoint, no network needed. THE gate.
task asan           # AddressSanitizer + UBSan build
task ubsan
task portable       # generic AVX2, no -march=native (for distributable binaries)
task debug          # -O0 -g with assertions
task format         # clang-format the tree
```

CMake is an alternative (same output binary, used for IDE integration):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`task test` must stay green with no model weights — if it's red on `main`, that's the bug
to fix first before anything else. CI additionally runs a GCC+Clang matrix under
`-Werror`, sanitizers over the parsers and cache, and ruff/shellcheck as blocking checks.

To run a single unit test binary directly, check `tests/unit/` — each `test_*.c` builds
its own binary under `task test`; check the Taskfile's `test` task for how they're invoked
if isolating one.

Cache-behavior spot check (no model needed):
```bash
python3 tools/sim_cache.py tests/fixtures/expert_trace.bin
```

## Architecture

```
include/k3/
  k3.h              public header — config, weights, every kernel prototype
  k3_cfg.h          config reader, header-only, refuses to substitute defaults
src/
  core/k3_ops.c     numeric kernels: RMSNorm, KDA, MLA, MoE, MXFP4 matmul
  io/k3_st.c        safetensors reader — hand-written JSON scan, O_DIRECT reads
  io/k3_load.c      locating one expert's bytes inside a shard
  io/k3_trunk.c     streaming the dense trunk: pinned prefix + a ring slot
  cache/k3_cache.c  routed-expert LRU cache + batch prefetch
  model/k3_bind.c   binds checkpoint tensor names to kernel arguments
  tokenizer/k3_tok.h byte-level BPE loaded from tiktoken.model
  cli/k3_run.c      the k3 binary itself: memory plan, decode loop, reporting
tools/              python: pack the trunk, replay the cache, verify vs. torch reference
benchmarks/         cgroup memory ladder, split sweep
tests/              fixtures, tiny oracle, 93-layer conformance run
```

Real weights are never required for development — `tests/fixtures/` (committed) carries a
13-layer oracle model built with the same tensor graph as the released model, checked
against a PyTorch reference (`tools/k3_ref.py`).

### The four size reductions (why this fits in 8GB at all)

1. Experts already ship at 4-bit (MXFP4); multiplied straight out of packed form, never
   dequantized to a dense tensor.
2. KDA (Kimi Delta Attention) — recurrent state of fixed size, doesn't grow with sequence
   length.
3. MLA — one shared latent instead of per-head KV, collapses attention memory.
4. The dense trunk streams from a packed single file (`k3_trunk.c`) instead of requiring
   full residency — this is what turns the memory floor into a dial (`--trunk-gb`).

### Three invariants (get these wrong and it still "works", just wrong)

Documented at the top of `k3.h` and worth re-reading before touching attention/MoE code:

1. `A_log` is indexed **per head**, not per channel — checkpoint ships `head_dim` floats
   but only the first `num_heads` are meaningful.
2. MLA uses NoPE, but the 64 rope dimensions still exist and are still cached — only the
   rotation itself is absent; dropping the slots changes head width.
3. MoE routing bias steers expert **selection** only — the combining weights come from
   the unbiased sigmoid scores, not the biased ones.

Each is gated by an adversarial fixture chosen so a plausible-but-wrong implementation
fails it (see README "Three invariants" section for the specific fixtures).

### Floating point contract

`-ffp-contract=off` is deliberate and load-bearing, not cosmetic: without it a compiler
may fuse multiply+add into FMA, changing rounding. Scalar, OpenMP, and AVX2 paths must
produce **bit-identical** output — a performance change must never become an accuracy
change. Never remove this flag.

### Exit codes (used by scripts, keep stable)

| code | meaning |
|---:|---|
| 0 | success |
| 1 | a tensor failed to bind, or a forward pass failed |
| 2 | usage error, or config unreadable with confidence (engine refuses to guess) |
| 4 | run finished but ≥1 routed expert failed to load — output is unsound. Distinct from 1 because this is silent numerical corruption, not a hard failure |

## Project conventions (from CONTRIBUTING.md)

- **A wrong answer that looks right is the worst failure mode here.** This engine can
  load the wrong architecture, stream a corrupt expert, or mis-tokenize and still emit
  fluent, plausible text with no crash and no NaN. Defensive checks in the code exist
  specifically because that failure class is invisible without them.
- **Fail loudly, never silently.** Missing config field → refuse, never substitute a
  default. Expert fails to load → count it and fail the run; never `continue` past it.
- **A test that cannot fail is not a test.** Fixtures must be adversarial — if a
  plausible wrong implementation would still pass, change the input until it wouldn't.
- **Comments explain why, not what** — especially anywhere a plausible-looking
  implementation would be wrong.
- Performance claims need ≥3 runs reported (run-to-run spread is ~33% on identical
  config) — prefer counts (bytes read/token, cache evictions, pinned layers) over
  wall-clock seconds since counts are immune to scheduling noise. See
  `docs/BENCHMARKING.md`.
- C99, 4-space indent, 90 columns. Warnings are errors in CI; `-Wpointer-arith` is
  deliberate (weight pointers are `const void *`, and GNU void-pointer arithmetic
  strides by 1 byte silently on GCC).

### Adding a kernel

1. Generate a fixture with `tools/emit_fixtures.py`, including its tolerance.
2. Add the case to `tests/unit/test_ops.c`.
3. Verify against `tools/k3_ref.py` (PyTorch reference).
4. Check it still holds on a real layer: `task test-all SHARD_DIR=...`.

## Platform notes

Linux/x86-64 is the reference platform (uses `O_DIRECT`, `posix_memalign`,
`getrusage`), needs AVX2+FMA. macOS/arm64 builds with plain `task build` but needs
`brew install libomp` (Apple Clang ships no OpenMP runtime) — `Taskfile.yml`
autodetects and wires this up via `UNAME_S`/`UNAME_M`.

Real checkpoint work (1.56TB download, trunk packing) is a separate, slow path
(`scripts/download-model.sh`, `scripts/pack-trunk.sh`) — not needed for engine
development; `task test` covers all normal development without it.
