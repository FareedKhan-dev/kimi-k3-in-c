# Documentation

## Start here

| | |
|---|---|
| [QUICKSTART.md](QUICKSTART.md) | from nothing to generated text |
| [TUNING.md](TUNING.md) | choosing a memory budget, one decision dominates |

## Reference

| | |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | how the model maps onto the code |
| [API.md](API.md) | the C interface, for embedding the engine |
| [PERFORMANCE.md](PERFORMANCE.md) | the memory ladder, measured, with its noise floor |
| [TESTING.md](TESTING.md) | what each gate proves |
| [BENCHMARKING.md](BENCHMARKING.md) | how to measure without fooling yourself |
| [ROADMAP.md](ROADMAP.md) | what is missing, in priority order |
| [data/](data/) | the raw measurement output every table is transcribed from |

## Upstream material

| | |
|---|---|
| [kimi-k3-tech-report.pdf](kimi-k3-tech-report.pdf) | the model's technical report, as published |

Reproduced for reference; it remains the property of its authors. Every architectural
claim in [ARCHITECTURE.md](ARCHITECTURE.md) cites the section of this report it comes
from, so the two can be checked against each other.

## The three claims worth checking first

If you are evaluating whether this project is worth your time, these are the load-bearing
claims and where each is substantiated:

1. **A 2.78T-parameter model runs in 8 GB of RAM.**
   [PERFORMANCE.md](PERFORMANCE.md), twelve budgets, measured peak RSS at each.
   Raw: [data/memory-ladder.tsv](data/memory-ladder.tsv).

2. **Output is byte-identical across memory budgets.** Memory buys speed, not
   capability. [PERFORMANCE.md](PERFORMANCE.md), and `examples/02-memory-budgets.sh`
   reproduces it in three runs. The `ids` column of
   [data/memory-ladder.tsv](data/memory-ladder.tsv) is the claim in raw form: twelve
   rows, one identical token sequence.

3. **The engine matches a reference implementation exactly.** Teacher forcing, greedy
   decode, and incremental decode, on a model with the released tensor graph.
   [TESTING.md](TESTING.md); `task test` runs it in seconds with no weights.

Claims about *speed* carry a caveat that [PERFORMANCE.md](PERFORMANCE.md) states up
front: run-to-run variance on identical configurations is 33%
([data/replication.tsv](data/replication.tsv)), so single-sample differences below that
are not effects. Claims about *output* carry no such caveat, those are exact, and they
are counts rather than stopwatch readings.
