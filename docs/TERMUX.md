# Building and testing on Termux (Android ARM64)

Tested on Android 14, Termux, clang, ARM64.

## Setup

```bash
pkg install -y git clang make cmake python binutils libomp
git clone https://github.com/FareedKhan-dev/kimi-k3-in-c.git
cd kimi-k3-in-c

make portable -j$(nproc)
make test
```

Termux has no writable `/tmp`. It sets `TMPDIR` to `$PREFIX/tmp` instead, and the tests
honour `TMPDIR` as of 2.0.0. On 1.1.0 and earlier, two tests (`test_trunk` and
`test_chat`) wrote to a literal `/tmp`; run `make test` under
`proot -b "$PREFIX/tmp:/tmp"` there (`pkg install proot`).

Result

```
VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
ALL WEIGHTLESS TESTS PASSED
```

## Tokenizer parity

This needs the released vocabulary, which ships with the checkpoint rather than with
this repository. Four small files are enough; `tokenization_kimi.py` is required because
`tools/tok_parity.py` reads the split regex out of it rather than restating it:

```bash
HF_HUB_DISABLE_XET=1 hf download moonshotai/Kimi-K3 \
    tiktoken.model tokenizer_config.json config.json tokenization_kimi.py \
    --local-dir ~/k3model
make test TOK_FILES=~/k3model     # the tokenizer and chat legs now run
make tok  TOK_FILES=~/k3model     # token-for-token parity, needs `pip install tiktoken`
```

## Limitations

The full model (1.56 TB) cannot run on a phone. `make test` is the full extent of what
Termux can verify.
