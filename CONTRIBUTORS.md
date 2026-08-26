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
  compute.
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

Thank you, all of you.
