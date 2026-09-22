# Contributors

People outside the project who have landed real fixes or features in this codebase.
GitHub's own contributor graph misses a few of these: a handful of pull requests hit a
merge conflict against other work landing the same day and had to be rebased by hand
through a new pull request to get a clean CI run, which is why their code shows up in
the history under a different name than the one that wrote it. This file is the correct
record.

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
  OpenMP link on the CMake Apple/libomp path; largest-first trunk pinning, the
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
  `--ids` list, an out-of-range `--layers`, and a lost `--out` file.

Thank you, all of you.
