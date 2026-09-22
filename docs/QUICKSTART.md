# Quickstart

From nothing to generated text.

## 1. Check the machine

```bash
./scripts/k3-doctor.sh
```

Reports whether this machine can run the model, which preset fits its memory, and how
fast its storage is. Storage matters more than you would expect, the engine moves
~135 GB per token at small budgets.

## 2. Build

```bash
make -j
```

No dependencies beyond a C compiler and OpenMP. If `-march=native` is a problem (you are
building for a different machine), use `make portable`.

## 3. Verify

```bash
make test
```

Seconds, and **no model weights required**. Op kernels, the streaming cache, the
safetensors reader, the config reader, the tokenizer, and the end-to-end oracle gates.

## 4. Get the weights

```bash
export HF_TOKEN=...            # a HuggingFace token with access to the model
./scripts/download-model.sh ~/k3model
```

1.56 TB across 96 shards; roughly 30 minutes at 1 GB/s. The script verifies the byte
total against the published figure afterwards, because a partial download does not fail
loudly, it produces wrong tokens.

## 5. Pack the trunk

```bash
./scripts/pack-trunk.sh ~/k3model ~/k3trunk
```

A few minutes. This is what allows the trunk to stream, and therefore what makes the
memory budget adjustable rather than fixed at ~115 GB.

## 6. Run

```bash
./bin/k3 ~/k3model \
    --trunk ~/k3trunk --preset server \
    --tok ~/k3model \
    --prompt "The capital of France is" --gen 16 --incremental
```

```
--- generated text ---
 Paris.
----------------------
```

## Where to go next

- [TUNING.md](TUNING.md), pick a budget and a split; one decision dominates
- [PERFORMANCE.md](PERFORMANCE.md), the full memory ladder and its noise floor
- `examples/`, runnable scripts, including one that proves output is identical across
  memory budgets

For a text conversation rather than a raw continuation, use the official K3 XTML REPL:

```bash
./bin/k3 ~/k3model --trunk ~/k3trunk --preset desktop --tok ~/k3model \
  --chat --system "You are a helpful assistant." \
  --history my-session.jsonl --incremental
```

The JSONL transcript is portable but sensitive: it retains assistant `reasoning_content`
as well as visible replies, because K3 needs that complete record for the next turn.

## Troubleshooting

**"REFUSING: the KV cache for N positions needs X"**, the context you asked for does not
fit. Shorten it, or drop `--incremental` (full recompute carries no KV cache, but is
O(T²) in time).

**Very slow with a long prompt**, expected. Prefill is not chunked and MLA attention is
quadratic in sequence length. See [ROADMAP.md](ROADMAP.md).

**"config could not be read with confidence"**, the reader found a config it does not
fully understand and refused rather than guessing. The message lists every missing field.

**Output is a continuation, not an answer**, use `--chat`; ordinary `--prompt` deliberately
keeps raw-completion behavior for compatibility.
