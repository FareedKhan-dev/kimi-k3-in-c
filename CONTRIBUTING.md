# Contributing

Real fixes and features from outside the project are credited in
[CONTRIBUTORS.md](CONTRIBUTORS.md).

## Before anything

```bash
make -j && make test
```

`make test` needs no model weights and must stay green. If it is red on `main`, that is
the bug worth fixing first.

`tools/*.py` (fixture generation, reference conformance, cache replay) needs `numpy`,
`torch`, and, for lint and the tokenizer parity check, `ruff` and `tiktoken`. Exact
versions are pinned in `pyproject.toml`; install them however you normally manage Python
dependencies, for example `pip install numpy==2.5.2 torch==2.13.0` and, for the dev
group, `pip install ruff==0.16.3 tiktoken==0.13.0`.

## The standard this codebase holds itself to

**A wrong answer that looks right is the worst failure mode here.** This engine can load
the wrong architecture, stream a corrupt expert, or mis-tokenize a prompt and still emit
fluent, plausible text. Nothing crashes. Several of the defensive checks in the code
exist because that class of failure is invisible without them.

So:

- **Fail loudly, never silently.** If a config field is missing, refuse, do not
  substitute a default. If an expert fails to load, count it and make the run fail; do
  not `continue` past it and produce a token that is missing part of its routed sum.
- **A test that cannot fail is not a test.** Make fixtures adversarial: if a plausible
  wrong implementation would still pass, change the input until it would not.
- **Comments explain why, not what.** Anywhere a plausible-looking implementation would
  be wrong, say so at that spot.

## Performance changes

The measured run-to-run spread on an identical configuration is **33%**. A single-sample
comparison proves nothing.

Report at least three runs per arm, and report all of them:

| arm | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|

Where you can, measure **counts instead of seconds**, bytes read per token, cache
evictions, pinned layers. They are immune to scheduling noise and make a much stronger
claim. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Style

- C99, 4-space indent, 90 columns. `make format` if you have clang-format.
- Warnings are errors in CI. In particular `-Wpointer-arith` is deliberate: weight
  pointers are `const void *`, and arithmetic on void strides by one byte under GCC
  silently returning the wrong tensor.
- Keep `-ffp-contract=off`. The op tests compare against a reference at a fixed
  tolerance; letting the compiler fuse multiply-adds moves results past it.

## Adding a kernel

1. Generate a fixture with `tools/emit_fixtures.py`, including its tolerance.
2. Add the case to `tests/unit/test_ops.c`.
3. Verify against the reference implementation in `tools/k3_ref.py`.
4. Check it still holds on a real layer: `make test-all SHARD_DIR=...`.

## Commits and PRs

Present tense, imperative: `add chunked prefill`, not `added` or `adds`. Explain *why* in
the body; the diff already shows what. Fill in the PR template's verification section;
it is a checklist, not a formality.

## Scope

Read [docs/ROADMAP.md](docs/ROADMAP.md) first. It also lists what is deliberately *not*
planned and why, which may save you writing something that will be declined.
