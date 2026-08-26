# Tuning

## The short version

1. Run `./scripts/k3-doctor.sh` and use the preset it names.
2. If you tune by hand: **fill the trunk before you feed the expert cache.**
3. Use `--incremental` unless you are validating against full recompute.

Everything below is why.

## One decision matters more than the rest

The engine splits your memory budget between two caches:

- `--trunk-gb`, the ring buffer and pinned layers for the **dense trunk**
- `--cache-gb`, the arena for **routed experts**

These are not interchangeable, and the asymmetry is large.

Per token the engine re-reads the **entire 108.81 GB trunk**, all 93 layers, in a fixed
order, every single token. It reads only **~25.8 GB of routed experts**, because just 16
of 896 are selected per layer.

So a gigabyte given to the trunk removes about **1.17 GB/token of guaranteed traffic**
(one pinned layer, never read again). A gigabyte given to the expert cache removes,
below roughly 36 GB of arena, **nothing measurable**.

Measured at a fixed 128 GB budget, endpoints of a six-point sweep:

| trunk | cache | s/token |
|---:|---:|---:|
| 12.3 | 110.7 | 28.38 |
| **110.0** | **13.0** | **16.80** |

**1.69× from allocation alone.** The faster configuration has a *smaller* expert cache
and 0.0% expert retention.

These are single samples against a 33% noise floor, so read the *direction*, not the
exact factor. The direction is supported by twelve points across two independent budgets
(Spearman ρ = −0.886 at 128 GB, −0.714 at 32 GB) and by the mechanism above; the
magnitude is not replicated. All twelve rows and the caveats are in
[PERFORMANCE.md](PERFORMANCE.md#allocation-beats-capacity), raw data in
[data/trunk-cache-split.tsv](data/trunk-cache-split.tsv), and
`benchmarks/split-sweep.sh` re-runs the experiment with repetitions.

### If the trunk and the checkpoint are on separate physical devices

The asymmetry above assumes trunk and expert reads compete for one drive's bandwidth.
On a machine where the packed trunk lives on one device and the checkpoint (routed
experts) lives on another, that coupling does not hold: trunk reads then run on an
otherwise idle device and overlap compute almost for free, so pinning more trunk layers
buys little once the ring is already overlapping.

The threshold that actually matters on this layout is the one the engine reports at
startup, the point where a second ring slot becomes affordable:

```
ring held at 1 slot: a second slot needs X.XX GB and the trunk budget is Y.YY GB,
so reads are NOT overlapped with compute. Raise --trunk-gb above X.XX GB to enable it.
```

Below that figure, trunk I/O sits on the critical path and costs measurably more; above
it, the curve flattens and further pinning stops paying for itself. Set `--trunk-gb`
just above that reported figure and give any remaining budget to `--cache-gb` instead,
since on split storage the expert device is the sole bottleneck. Reported and measured
by a user running the packed trunk on NVMe and the checkpoint on a separate SATA drive;
see [PERFORMANCE.md](PERFORMANCE.md) for the full writeup.

## Why the expert cache is so weak

This is the model's design, not a shortcoming of the implementation.

K3's router is trained with **Quantile Balancing**, which deliberately flattens expert
usage so that no expert is favoured. Flat usage is precisely what defeats a
least-recently-used cache: with no hot subset, a few gigabytes retain nothing worth
keeping. Measurements bear this out, retention stays at exactly 0.0% from 28 slots all
the way to 1,344, and the bytes read per token do not move by a single decimal.

It does eventually engage. Somewhere around 36 GB of arena, retention jumps to ~30% and
bytes/token fall from 25.83 to 18.11. But by then you could have pinned most of the trunk
for the same memory and gone faster.

## Presets

```console
$ ./bin/k3 --list-presets
```

| preset | trunk | cache | total | expect |
|---|---:|---:|---:|---|
| `ultra` | 2.5 | 0.31 | ~3 GB | proof-of-life only; streams embed/lm_head and reuses one state slot |
| `laptop` | 3 | 1 | ~10 GB | ~32 s/token |
| `desktop` | 16 | 10 | ~32 GB | ~31 s/token |
| `workstation` | 60 | 30 | ~96 GB | ~24 s/token |
| `server` | 110 | 13 | ~128 GB | ~17 s/token |
| `max` | 110 | 109 | ~224 GB | ~19 s/token |

`server` is the best value: the trunk is fully pinned at 110 GB and everything beyond
that is spent on a cache that contributes little. Note `max` is not faster than `server`
in these measurements, the extra 96 GB buys nothing outside the noise floor.

Flags after `--preset` override it, so `--preset server --cache-gb 40` works.

`ultra` is deliberately a separate execution path, not a smaller ordinary preset. It
requires a packed trunk, streams exact BF16 embedding/lm_head bytes, and with full
recompute clears and reuses one layer-state slot. It trades I/O for RAM and is intended
for short, deterministic proof runs; speculative/draft decoding is rejected in this
mode. The arithmetic, Top-K routing and weight precision are unchanged.

## Other options

**`--incremental`** carries a KV cache and the recurrent state between tokens instead of
recomputing the prefix. Verified to produce identical tokens to full recompute. Use it
unless you are specifically testing that equivalence.

Note it allocates ~2.37 MB of KV cache **per position**, so long contexts cost
memory: 16k positions is ~39 GB. The engine computes the requirement up front and refuses
with both numbers rather than being OOM-killed an hour in.

**`--trunk DIR`** enables trunk streaming and is what makes memory a dial. Without it the
trunk is fully resident and the floor is ~115 GB. Pack it once with
`tools/pack_trunk.py`.

**`--layers N`** binds only the first N layers. Useful for testing the machinery on a
partial download; the output is not the full model and the engine says so.

## Storage matters more than you expect

The engine moves ~135 GB per token at low budgets. Storage bandwidth is usually the
ceiling, not the CPU.

- Put the checkpoint and the packed trunk on the **fastest local NVMe** available.
- Network or spinning storage will dominate everything else. A 6× difference in device
  bandwidth is a 6× difference in throughput on this workload, which is larger than any
  tuning decision in this document.
- `./scripts/k3-doctor.sh` measures your device so you know which regime you are in.

## Threads

`OMP_NUM_THREADS` defaults to your core count. The workload is I/O bound at low memory
budgets, so more threads help less than you would expect once the trunk is streaming.

This has not been swept systematically on this engine see [ROADMAP.md](ROADMAP.md).

## Before you conclude a change helped

The measured run-to-run spread on an identical configuration is **33%**. Differences
smaller than that are not effects. Run each arm at least three times.
[BENCHMARKING.md](BENCHMARKING.md) has the procedure.
