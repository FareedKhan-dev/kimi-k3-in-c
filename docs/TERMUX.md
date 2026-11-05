# Building and testing on Termux (Android ARM64)

Tested on Android 14, Termux, clang, ARM64.

## Setup

```bash
pkg install -y git clang make cmake python binutils libomp proot
git clone https://github.com/FareedKhan-dev/kimi-k3-in-c.git
cd kimi-k3-in-c

export TMPDIR=$PREFIX/tmp
mkdir -p "$TMPDIR"

make portable -j$(nproc)
proot -b "$PREFIX/tmp:/tmp" make test
```

Result

```
VERDICT: ENGINE MATCHES THE REFERENCE EXACTLY
ALL WEIGHTLESS TESTS PASSED
```

Tokenizer parity

Requires the checkpoint's tiktoken.model:

```bash
HF_HUB_DISABLE_XET=1 hf download moonshotai/Kimi-K3 tiktoken.model tokenizer_config.json config.json --local-dir ~/k3model
make tok TOK_FILES=~/k3model
```

Limitations

The full model (1.56 TB) cannot run on a phone. make test is the full extent of what Termux can verify.
