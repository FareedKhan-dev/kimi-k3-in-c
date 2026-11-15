# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **An expert whose E8M0 scales carried the NaN code loaded silently.** Scale byte 255 is
  NaN by the OCP MX spec and no quantiser writes it; the kernels mapped it to zero and
  skipped the 32-weight group, so a damaged shard ran with pieces of experts missing and
  nothing said so. The streaming cache now scans the scale bytes of every expert as they
  land in memory, 1 byte per 32 weights, and refuses the expert naming the shard file, the
  tensor and the count; the run then fails the way a missing tensor fails it. The kernel
  mapping stays as defence in depth. The 7,168 real scale bytes in `tests/fixtures/mxfp4.json`
  range 120 to 122; the 96 released shards were not re-checked for this change.

- **The trunk report counted a prefetched layer as a hit** whenever its read finished
  before the walk bound it. On a fully streamed run of the tiny checkpoint, every one of
  the 156 layer binds was a real read ("reads 156 against 156 the walk owes"), and the
  line above it still said `hits 144 (92.3%), reads 12`. A layer the reader fetched
  ahead is now charged as a read to the bind that reaches it, whichever thread gets there
  first, so the counts no longer depend on timing and the hit rate again tracks the
  pinned fraction. Counting only: output is unchanged. This was also the cause of an
  intermittent `test_trunk` failure on CI, which now tests both interleavings.

## [1.1.0] - 2026-10-02

Chat, native Windows builds alongside Linux and macOS, an 8 GB class mode for the
complete model, NEON kernels for ARM, and a round of hardening on everything that parses
checkpoint files. Output is unchanged wherever it was already defined: every oracle gate
still matches the reference exactly.

One compatibility note: `--save-state` files move to version 2, with a checksum. A
version 1 file is refused with a clear message rather than read; regenerate it with
`--gen 0 --incremental --save-state`.

### Added

- **`--preset ultra` and `--ultra-low-memory`**: the complete 93 layer model inside an
  8 GB class memory budget, by streaming exact embedding rows and bounded lm_head chunks
  and reusing one recurrent state slot. Same weights, same top-16 routing, same kernels.
  Verified with four full runs on a Jetson Orin Nano Super against the released
  checkpoint, identical ids and identical logits each time. Slow, by design.
- **NEON kernels on aarch64** for the bf16, MXFP4 and q8 matmuls, so Apple Silicon and
  ARM servers are no longer scalar only. Bit identical to the scalar path.
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
- **Ctrl-C stops at a safe point.** The first Ctrl-C lets the step in flight finish,
  then writes the state file, the `--out` JSON and the reports exactly as a finished run
  would, and exits 5. A second Ctrl-C kills. Before, SIGINT discarded everything.
- **Chat reuses the previous turn's state.** When the next turn's rendered prompt begins
  with exactly the ids the carried state was built from, only the new tail is prefilled.
  GATE 3b in the oracle requires that to be bit identical to a full prefill: logits, every
  KV row, and the KDA state.
- **`--top-k K`** for chat sampling, on the existing sampler. Off by default.
- **`--threads N`**, and a default of the physical core count on Linux instead of one
  thread per logical CPU, which measured about 23% slower per token on a 16 core SMT part.
- **`download-model.sh <dest> --layers N`** fetches only the shards a `--layers N` run
  needs, resolved from the Hub's own index, with a banner saying the result is not the
  model.
- **A missing tensor names the missing file.** The shard filenames declare the shard
  count, so a partial download now says which file is absent instead of only which
  tensor.
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

- **The MXFP4 expert matmul is about 1.6x faster**, decoding nibbles in register, plus
  another 1.56x from a flat-row AVX2 path when the group is a multiple of 16. Bit
  identical, checked against the kernel hashes before and after.
- **Expert reads are split into 1 MiB chunks**, so a batch of reads no longer waits on its
  slowest expert. Output is unchanged.
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

- **macOS could not read the embedding table or a packed trunk layer.** Darwin's pread
  refuses a single request of 2 GiB or more, and both exceed it; reads are now chunked.
  The download script also works on macOS and with current huggingface_hub releases.
- **CMake on Apple Silicon and on macOS.** The ARM64 build was handed x86 flags, and the
  macOS build silently came out single threaded because CMake could not find Homebrew's
  libomp. Both are fixed.
- **`k3_matmul_mxfp4` now enforces its documented preconditions.** An odd input width or a
  group above 64 aborts with a message instead of writing past the caller's buffer.
- **`--preset auto` refused itself on large machines.** It planned to 98% of available
  memory while admission allowed 95%, so from about 100 GB of RAM upward the recommended
  preset could never start, and the refusal printed a negative shortfall. One constant
  now drives both.
- **On macOS the available memory read as 0**, which silently switched off both memory
  refusals. Darwin and Windows now have their own probes, and Windows also caps by the
  remaining commit limit.
- **A state file could be wrong without the loader noticing.** One flipped byte restored
  cleanly, and a save that failed partway destroyed the previous good file. The payload
  is now checksummed and the file published by an atomic rename. State files move to
  version 2; a version 1 file is refused rather than guessed at.
- **Safetensors headers whose tensor spans overlap or leave a gap** were read without
  complaint, shifting every tensor after them. They are refused now, the same rule the
  reference loader applies; all 96 shards of the released checkpoint pass it.
- **The JSON parser read past the end of a truncated string** and accepted incomplete
  documents. Both are refusals now. Every JSON file the engine reads from the released
  checkpoint parses to the same tree as before.
- **A refused trunk.json leaked its whole parse tree**, and `--ids` values past the int
  range wrapped to a real token id.
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

[Unreleased]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/FareedKhan-dev/kimi-k3-in-c/releases/tag/v0.1.0
