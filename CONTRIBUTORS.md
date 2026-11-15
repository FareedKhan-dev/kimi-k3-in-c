# Contributors

People outside the project who have landed real fixes or features in this codebase.

GitHub's contributor list is built from commit authors. Most of the pull requests below
were rebased and squash-merged by hand to clear conflicts with other work landing at the
same time, and those commits went in under the maintainer's name, so that list misses
most of the people here. This file is the correct record. Pull requests merged after
v1.1.0 are committed under their author's name.

- **[douglasmun](https://github.com/douglasmun)** -- got the engine building and
  passing the test suite on macOS / Apple Silicon.
- **[cwwjacobs](https://github.com/cwwjacobs)** -- verified checkpoint downloads
  against Hub checksums; added the synthetic trunk streaming regression test covering
  the one-slot guard and two-slot async prefetch.
- **[mahavak](https://github.com/mahavak)** -- regenerated the tiny checkpoint fixture
  used for end-to-end test runs.
- **[sulfierry](https://github.com/sulfierry)** -- fixed the Hugging Face checkpoint
  download for the current `hf` CLI.
- **[Barba2k2](https://github.com/Barba2k2)** -- overlapped trunk reads with layer
  compute; made CI install the Python tool versions `pyproject.toml` already pins,
  instead of whatever was newest that day.
- **[ShaalanMarwan](https://github.com/ShaalanMarwan)** -- fixed the CMake build on
  ARM64/AArch64, where the x86-specific `-mavx2`/`-mfma` flags were applied
  unconditionally.
- **[TROY665](https://github.com/TROY665)** -- the in-register MXFP4 nibble decode
  that took the expert matmul kernel about 1.6x faster, bit-identical; found and fixed
  the bf16 benchmark weights that made the documented bit-identity hash check
  unwinnable.
- **[ysgao](https://github.com/ysgao)** -- macOS/Apple Silicon download-script
  portability, and the Darwin `pread()` 2 GiB ceiling that failed on the embedding
  table and a packed trunk layer.
- **[openchat-ai](https://github.com/openchat-ai)** -- the flat-row AVX2 fast path for
  the MXFP4 kernel, about 1.56x over the nibble-decode kernel it built on.
- **[arafatsolok](https://github.com/arafatsolok)** -- NEON ports of the bf16, MXFP4
  and q8 matmul kernels, so Apple Silicon and ARM server builds are no longer scalar
  only.
- **[genesisrevelationinc-debug](https://github.com/genesisrevelationinc-debug)** --
  native Windows support via MinGW-w64, including the heap-corruption bug from pairing
  `_aligned_malloc` with plain `free()`.
- **[AuricTW](https://github.com/AuricTW)** -- the opt-in `ultra` preset that keeps the
  complete model within an 8 GB class memory budget, verified with four full runs on a
  Jetson Orin Nano Super against the real checkpoint.
- **[biokraft](https://github.com/biokraft)** -- proposed the CMake CI regression job
  that actually builds and tests the CMake path (previously documented as
  interchangeable with Make but never verified in CI), and pinned Python tool
  dependencies in `pyproject.toml`.
- **[cablepull](https://github.com/cablepull)** -- the parallel-chunked trunk reader
  that takes the streamed trunk off queue depth 1; the `--stop-id` flag, with the
  emit-time check that truncates a speculative sweep exactly like serial decode; and
  running the incremental prefill on `--gen 0` so a fixed prefix can be warmed once and
  resumed from a byte-exact state.
- **[Avicennasis](https://github.com/Avicennasis)** -- hardened `--stop-id` parsing
  against silent typos, added its `stopped_at` line to `k3_run.json` and a weightless
  contract gate, and kept `k3_run.json` valid when `--gen 0` generates nothing; carried
  the XTML chat REPL forward with the tokenizer bundle dropped, `--no-think` and
  `--thinking-effort`, and the Windows portability fixes it needed (`getline`, `fsync`,
  and a `rename` that actually replaces an existing file); found and fixed the missing
  OpenMP link on the CMake Apple/libomp path; carried Deobot2's trunk pinning and
  prefetcher work through review, with the synthetic trunk test adapted to it; Ctrl-C
  that stops at a safe point and still writes everything; a
  checksummed state file published by atomic rename; chat turns that reuse the previous
  turn's state behind a bit-exact gate; available memory that is real on macOS and
  capped by commit headroom on Windows; missing-shard diagnostics that name the file;
  safetensors span checks validated against every shard of the released checkpoint;
  memory admission that respects the cgroup limit a container or `systemd-run` scope
  imposes, measured down to the dirty pages that cannot count as room; and refusing an
  expert whose E8M0 scales carry the NaN code.
- **[Deobot2](https://github.com/Deobot2)** -- largest-first trunk pinning, the
  `--trunk-ring` flag, and two prefetcher re-read bugs found by instrumentation rather
  than by inspection.
- **[Blake Evans](https://github.com/BlakeEvans22)** -- the original K3 XTML chat core
  and REPL, reproducing the checkpoint's own chat format so the engine answers a
  chat-shaped prompt instead of completing it.
- **[FermiHart](https://github.com/FermiHart)** -- a set of parser hardening fixes
  across the safetensors reader, the trunk reader, and the vendored JSON library:
  integer-overflow saturation, offset and size checks that run before the arithmetic
  they protect rather than after, a library that no longer calls `exit()` on an
  allocation failure, and a new weightless fault-injection test; a batched MoE prefill
  path that read uninitialized memory on an expert load failure instead of contributing
  zero, exactly as the per-token path already does; and CLI refusals for a malformed
  `--ids` list, an out-of-range `--layers`, and a lost `--out` file; a strict JSON parser
  that refuses truncated input instead of reading past it; a trunk.json leak on the
  refusal path; sampler tests that run without the tokenizer; and `--top-k` for chat.
- **[santhoshsathish94](https://github.com/santhoshsathish94)** -- found that
  `--preset auto` could never start on the machines it was written for and fixed it,
  with the measurements that made the thread-count default change a clear call; split
  expert reads so a batch no longer waits on its slowest read.
- **[Irfanwani](https://github.com/Irfanwani)** -- the `--layers N` download mode, so the
  pipeline can be exercised on a disk too small for the checkpoint.
- **[Rudy Celekli](https://github.com/rudycelekli)**: most of the 2.0.0 hardening round.
  In the engine, a tokenizer loader that refuses a corrupt or truncated vocabulary, state
  headers checked against the config before any read, a prompt past the ceiling refused
  instead of cut, experts held to the config's shape, split reads with a short middle
  chunk refused, output writes that fail the run when they fail, an opt-in request trace,
  strict numeric flags, a chat sampler that refuses invalid logits and measures the
  nucleus after top-k, and a Makefile that rebuilds when flags or headers change. In the
  tools, conformance, logit, benchmark, bandwidth, quantization, packing, budget,
  tokenizer-path, cache-replay and doctor scripts that refuse to report what they did not
  measure.
- **[WaleedHU](https://github.com/WaleedHU)**: made the documented `--layers 1`
  development path actually run, with an empty expert cache when the shard set holds no
  routed expert.
- **[mechmind-dwv](https://github.com/mechmind-dwv)**: the Termux guide for Android ARM64,
  and the report that showed two tests assumed a writable `/tmp`.

Thank you, all of you.
