## What this changes

<!-- One or two sentences. What behaviour is different after this PR? -->

## Why

<!-- The problem being solved. Link an issue if there is one. -->

## Verification

- [ ] `task test` passes (all weightless gates)
- [ ] `task portable` builds with no new warnings
- [ ] If kernels changed: `./bin/test_ops tests/fixtures/ops` still passes at declared tolerance
- [ ] If the config or tokenizer path changed: `task cfg` and `task tok` pass
- [ ] If output could change: the oracle gates (`./bin/k3_model tests/fixtures`) still match exactly

## Numbers, if this is a performance change

<!-- Timing claims need a noise floor. Report at least 3 runs of each arm, or say
     explicitly that the effect was not measured against variance. See docs/BENCHMARKING.md. -->

| arm | run 1 | run 2 | run 3 | mean |
|-----|-------|-------|-------|------|
|     |       |       |       |      |

## Risk

<!-- What could this break that the tests would not catch? -->
