# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`--chat`**: a terminal REPL implementing K3's official XTML chat format, message
  envelopes, the `thinking_effort` system preamble, and `reasoning_content` on prior
  assistant turns, rendered exactly as the checkpoint's own encoder does, with JSONL
  history and a parser for one assistant completion. Greedy by default; `--temperature`,
  `--top-p`, and `--seed` opt in to sampling. `--no-think` and `--thinking-effort low`,
  `high`, or `max` skip or size the think channel, which on a short answer is most of
  the cost. Without a chat template the engine had completed a chat-shaped prompt as if
  it were mid document rather than answering it.
- **`--trunk-ring N`**: the streaming trunk's prefetch queue depth is now a flag
  (default 2) instead of fixed at two slots.
- **`--stop-id N`** (repeatable, up to 8): generation halts as soon as the model emits
  a listed token id. Off by default, so `--gen N` still means exactly N tokens for
  every benchmark and oracle gate. The stop id stays in the sequence, so `--save-state`
  and a later `--load-state` continue from what the model actually produced, and the
  check runs at emit time so a `--spec` sweep is truncated at the stop exactly like
  serial decode. `k3_run.json` gains `"stopped_at"` (the id, or -1). Parsing is with
  `strtol` and refuses a non-integer, a negative, or an id past the vocabulary, since a
  stop the model can never emit is indistinguishable from a model that never emitted
  one.
- **Prefill on `--gen 0`**: `--gen 0 --incremental --save-state` now runs the prompt's
  prefill and saves its exact KV and recurrent state with zero generated tokens, so a
  shared prefix (a system prompt, a long document) can be warmed once and resumed many
  times with `--load-state`. Previously `--gen 0` skipped the decode loop and saved
  nothing useful.
- **Windows support**: builds natively via MSYS2's MinGW-w64 GCC, no WSL required.
  `make`, `make test`, and `make test-all` pass every gate unmodified, including the
  full-model oracle and tokenizer parity (45/45) against real Kimi K3 weights.
  `src/io/k3_portable_io.h` gained a Windows branch alongside the existing Darwin one,
  porting `O_DIRECT` (via `FILE_FLAG_NO_BUFFERING`, intercepted at `open()` since
  Windows -- unlike Darwin -- cannot add it to an already-open handle), `pread` (via
  `ReadFile`'s `OVERLAPPED` offset fields, chosen specifically because it does not
  share mutable file-pointer state across threads the way `SetFilePointerEx` +
  `ReadFile` would), `posix_memalign` (via `_aligned_malloc`), and `getrusage`/
  `MemAvailable` (via `GetProcessMemoryInfo`/`GlobalMemoryStatusEx`). `make asan`/
  `make ubsan` switch to Clang on Windows (MinGW-w64's GCC package ships no sanitizer
  runtime at all, confirmed directly rather than assumed).

### Changed

- **Trunk layers are read in parallel chunks.** `load_run()` streamed each layer with
  one sequential `pread` loop, so the device saw queue depth 1. It now splits the layer
  into 64 MiB chunks (a multiple of `K3_TRUNK_ALIGN`, so every chunk stays aligned for
  `O_DIRECT` and `F_NOCACHE`) issued under an OpenMP parallel for, matching what the
  expert path already does. Without OpenMP the loop still runs one chunk at a time. A
  short read in any chunk fails the whole layer, as before, and the decoded output and
  `trunk_bytes_read` are unchanged.
- **Trunk layers are pinned largest-first instead of as a prefix.** Prefix pinning
  always pinned the 2.34 GB dense layer first, which sized the streaming ring's slot to
  it even after it stopped needing the ring at all. `K3_PIN_PREFIX=1` restores the old
  order on the same binary.

### Fixed

- **Hostile or corrupt safetensors, trunk, and config input could reach undefined
  behavior instead of a refusal.** Integer overflow in shape and offset arithmetic
  (a hostile shape or `data_offsets` pair near `INT64_MAX` could wrap and defeat the
  bounds checks meant to catch it), an out-of-bounds read on an empty trunk layer
  array, a batched MoE prefill path that summed uninitialized memory when an expert
  failed to load instead of contributing zero as the per-token path already does, and
  unchecked allocations in the JSON parser and the shard directory listing that could
  crash through a null pointer rather than fail cleanly. A new weightless test,
  `test_st_faults`, builds mutated copies of the fixture shards and asserts each
  failure mode is refused loudly. No input that was already valid changes.
- **`--ids` with a non-numeric piece silently became token id 0** instead of being
  refused, because a failed `strtol` call and a real id of 0 are indistinguishable
  without checking where parsing stopped.
- **`--layers 0` or a value past the real layer count silently ran the complete
  model**, as if `--layers` had not been given, instead of being refused as the typo
  it almost always is.
- **An `--out` path that could not be written still exited 0**, indistinguishable from
  a run whose result was actually saved. It now exits 3.
- **`k3_run.json` was not valid JSON after a run that generated nothing.** With
  `nout == 0` the `seconds_per_token` field computed `t_total / nout` and emitted a
  bare `inf`, so a harness driving `--gen 0 --save-state` failed on the one run it
  needed to parse. It now reports `0`.
- **Heap corruption on Windows** (`STATUS_HEAP_CORRUPTION`) in the trunk and expert-
  cache arena allocators: `_aligned_malloc`, which backs the Windows `posix_memalign`
  shim, must be freed with `_aligned_free`, not plain `free`. POSIX's `posix_memalign`
  carries no such restriction, so this compiled cleanly and only crashed once the
  corrupted allocator metadata was actually used, well after the allocation itself.
  Three call sites needed the fix: `k3_cache.c`'s cache arena, and `k3_trunk.c`'s
  trunk arena and per-layer pinned buffers.
- **`SHARD_DIR`/`TOK_FILES` unquoted in the Makefile**: a path containing a space
  (routine on Windows, e.g. an "AI LOCAL MODELS" folder) silently split into extra
  argv entries instead of failing loudly, and `test_expert`/`test_real_layer`/
  `test_tok`/`test_cfg` read whichever truncated token happened to resolve to a path,
  rather than refusing outright.

## [1.0.0] - 2026-08-07

Verified end to end on the full released checkpoint, and made substantially faster, with
byte-identical output preserved at every step. The first-run experience, which was broken
on a clean clone, now works.

### Added

- **`--preset auto`**: sizes the trunk and expert-cache budgets from the machine's own free
  RAM, trunk-first, so a user need not pick a preset by hand. A gigabyte given to the trunk
  is worth far more than a gigabyte of expert cache, and auto pins accordingly, capping the
  pin below the RAM ceiling after a heavy-pin regression was measured.
- **Chunk-union prefill**: a batched-prefill MoE that fetches each unique routed expert once
  per chunk instead of once per token, measured to read about half the expert bytes on a
  prompt, with the generated token bit-identical to the per-token path.
- **Conversation resume** (`--save-state` / `--load-state`): carries the recurrent state and
  KV cache to disk so a second turn resumes instead of re-reading the whole prompt, measured
  3.9x faster on turn two with identical output. Refuses to restore state from a different
  architecture.
- **`--spec N`**: speculative decode by n-gram drafting with batched greedy verification;
  output is exactly the serial greedy decode by construction.
- `--tf-check`, teacher-forced agreement over an id sequence in one sweep, for measuring
  draft quality; `tools/qdq_trunk.py` and `tools/int8_trunk.py` for deriving quantized
  trunks.

### Changed

- **Fused matmul kernels** (fp32, bf16, MXFP4): sixteen partitioned accumulators with
  explicitly fused products, taking the trunk matmul to its memory floor (about eight times
  less per-token compute) while keeping the scalar and AVX2 paths bitwise identical.
- **KDA recurrence parallelised over heads**, bit-identical to the serial form.
- `scripts/k3-doctor.sh` per-preset speed expectations refreshed to the v1.0.0 numbers, with
  the streaming presets noted as disk-bound and the resident tier as compute-bound.

### Fixed

- All shell scripts are committed executable; the first documented command no longer fails
  with Permission denied on a clean clone.
- `scripts/download-model.sh` uses the current `hf` CLI and pins an immutable revision with
  checksum verification; it no longer attempts a pip install that cannot succeed on the
  target OS, and refuses to start without free space for the checkpoint.
- `scripts/k3-doctor.sh` no longer fails a machine that can build and test the engine; the
  memory floor is a warning about running the checkpoint, not a hard stop.
- The config-refusal fixtures the docs describe now exist and are gated in `make test`,
  ctest and CI; the tokenizer leg reports NOT RUN rather than passing silently; CI runs
  `make test` rather than a hand-picked subset.
- A silent-corruption path in the MLA KV overflow and one in the single-slot trunk reader,
  both of which could emit a plausible wrong token, now abort or are prevented.
- The MXFP4 packer alignment and the tiny-checkpoint scale rule.

### Research notes, not shipped as features

- Lossless trunk compression and a quantized-self-draft hybrid were both built and measured,
  and both turned out to help only narrow regimes. The findings and prototypes are kept in
  [`docs/notes/`](docs/notes/).

## [0.1.0] - 2026-07-31

First public release.

### Added

- Full 93-layer Kimi K3 inference: 69 KDA + 24 Gated MLA layers, 896 routed experts with
  top-16 selection, SiTU-GLU, Attention Residuals, native MXFP4 expert weights.
- **Trunk streaming**, which turns the memory budget into a dial rather than a floor. The
  model runs in 8 GB and in 224 GB and produces byte-identical output at every budget
  measured in between.
- MXFP4 matmul that consumes packed nibbles directly, never materialising a dequantised
  expert.
- BPE tokenizer in C, reading the released `tiktoken.model` directly, text in, text out
  with no external step.
- Config reader that loads the checkpoint's own `config.json` and **refuses** a config it
  cannot fully understand rather than defaulting missing fields.
- Incremental decode with a KV cache and carried recurrent state, verified to produce the
  same tokens as full recompute.
- Named memory presets (`--preset laptop|desktop|workstation|server|max`) derived from
  the measured memory ladder.
- `scripts/k3-doctor.sh`, reports whether a machine can run the model, which preset
  fits, and how fast its storage is.
- `scripts/download-model.sh`, fetches the checkpoint and verifies it byte-exactly
  against the published total, because a partial download produces wrong output silently.
- Test suite that runs entirely without model weights: op fixtures, expert cache,
  safetensors reader, config reader, and end-to-end oracle gates (teacher forcing,
  greedy decode, and incremental decode).
- CI: build matrix across GCC and Clang, warnings-as-errors, ASan and UBSan, Python and
  shell lint. Tokenizer parity is built and reported but CANNOT gate on a clean
  checkout, because it needs the vocabulary that ships with the model weights; run
  `make tok` locally against a downloaded checkpoint.

### Known limitations

- No chunked prefill, so long prompts are impractical despite a 32k context ceiling.
- Greedy decoding only; no chat template; no serving layer; no vision; CPU only.

See [docs/ROADMAP.md](docs/ROADMAP.md).

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
