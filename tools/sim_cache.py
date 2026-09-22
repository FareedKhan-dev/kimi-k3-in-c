#!/usr/bin/env python
"""
sim_cache.py - how much RAM does Kimi K3 actually need?

THE IDEA
    Hit rate versus cache size is the question the whole streaming design turns on, and
    measuring it directly would mean re-running the model once per cache size on a
    machine big enough to hold the whole expert pool. That is expensive and unnecessary:
    routing does not depend on the cache, so ONE run produces the entire curve. k3_cache
    records every (layer, expert) request in order, and this replays that trace at any
    capacity under any policy.

WHAT IT REPORTS
    LRU      what the engine used to implement, kept as the baseline the change is
             measured against.
    S3-FIFO  what the engine implements now (src/cache/k3_cache.c). Small FIFO, main
             FIFO, ghost queue; see the note on the function.
    Belady   the optimal offline policy, which no real cache can beat. The gap between
             LRU and Belady is what said the lever was the policy rather than the size:
             25.5 points at 64 GB on the published trace. S3-FIFO is the attempt to
             close some of it, and this is where that attempt is checked.
    Pinned   the hottest N experts held permanently, LRU for the rest. This is what the
             usage histogram is for.

usage: sim_cache.py expert_trace.bin [--expert-bytes 17547264] [--disk-mbs 1234]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, OrderedDict, deque

import numpy as np

EXPERT_BYTES = 17_547_264          # measured, not assumed: see README
TOTAL_EXPERTS = 92 * 896


def lru(trace, cap):
    """Returns hits. trace is a list of integer keys."""
    seen = OrderedDict()
    hits = 0
    for k in trace:
        if k in seen:
            seen.move_to_end(k)
            hits += 1
        else:
            if len(seen) >= cap:
                seen.popitem(last=False)
            seen[k] = 1
    return hits


def belady(trace, cap):
    """Optimal: evict whatever is needed furthest in the future. Needs the whole trace,
    which is exactly why no online cache can implement it."""
    n = len(trace)
    nxt = [n] * n
    last = {}
    for i in range(n - 1, -1, -1):
        k = trace[i]
        nxt[i] = last.get(k, n)
        last[k] = i

    resident = {}                    # key -> next use index
    hits = 0
    for i, k in enumerate(trace):
        if k in resident:
            hits += 1
            resident[k] = nxt[i]
            continue
        if len(resident) >= cap:
            victim = max(resident, key=resident.get)
            if resident[victim] < nxt[i]:
                # Everything resident is needed sooner than this one; skip caching it.
                continue
            del resident[victim]
        resident[k] = nxt[i]
    return hits


def s3fifo(trace, cap, small_frac=0.1):
    """S3-FIFO, the policy src/cache/k3_cache.c actually implements.

    Kept here as well as in C for one reason: this is where the decision to replace LRU
    was made, and a claim about a policy that can only be checked by re-running the model
    is not a claim anybody can check. Replaying the same trace through both is cheap, and
    the gap it reports is the gap the engine should show.

    Three structures, and the split is the whole idea:
      S  a small FIFO taking every new admission, so one-hit wonders leave quickly
         without ever reaching the main cache. K3's routing produces a lot of them.
      M  the main FIFO, holding objects that earned a place by being touched again.
      G  a ghost FIFO of KEYS evicted from S, so a prompt return skips the probation.

    Frequency is a saturating 2-bit counter, not a recency order -- which is what makes
    it cheaper than LRU as well as better on this trace.
    """
    small = max(1, int(cap * small_frac))
    S, M = deque(), deque()
    G, Gset = deque(), set()
    freq = {}
    resident = set()
    hits = 0

    def evict():
        while True:
            # Take from S while it is over its share, and also when M has nothing to
            # give -- otherwise a small S with an empty M spins here forever, which the
            # C side hits as a failed admission rather than a hang.
            if S and (len(S) >= small or not M):
                v = S.popleft()
                if freq.get(v, 0) > 0:
                    freq[v] = 0
                    M.append(v)
                    continue
                resident.discard(v)
                freq.pop(v, None)
                if len(G) >= cap:
                    Gset.discard(G.popleft())
                G.append(v)
                Gset.add(v)
                return
            if M:
                v = M.popleft()
                if freq.get(v, 0) > 0:
                    freq[v] -= 1
                    M.append(v)
                    continue
                resident.discard(v)
                freq.pop(v, None)
                return
            return                # nothing resident at all

    for k in trace:
        if k in resident:
            hits += 1
            freq[k] = min(freq.get(k, 0) + 1, 3)
            continue
        if len(resident) >= cap:
            evict()
        freq[k] = 0
        resident.add(k)
        if k in Gset:
            Gset.discard(k)
            M.append(k)
        else:
            S.append(k)
    return hits


def pinned_lru(trace, cap, npin):
    """Hold the npin hottest experts permanently, LRU the remaining capacity.

    Two honesty constraints:
      - A pinned expert still has to be READ once. Counting its first touch as a hit
        would make pinning look like it conjures weights out of nothing.
      - The hot set here is chosen from this very trace, which is future knowledge. Real
        pinning uses a profile from a previous workload, so this is an UPPER BOUND on
        what histogram-driven pinning can deliver, not a prediction.
    """
    if npin >= cap:
        npin = cap - 1
    hot = {k for k, _ in Counter(trace).most_common(npin)}
    loaded = set()
    seen = OrderedDict()
    room = max(cap - len(hot), 1)
    hits = 0
    for k in trace:
        if k in hot:
            if k in loaded:
                hits += 1
            else:
                loaded.add(k)          # compulsory miss, paid once
            continue
        if k in seen:
            seen.move_to_end(k)
            hits += 1
        else:
            if len(seen) >= room:
                seen.popitem(last=False)
            seen[k] = 1
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--expert-bytes", type=int, default=EXPERT_BYTES)
    ap.add_argument("--disk-mbs", type=float, default=1234.0,
                    help="measured random-cold read rate, for the time column")
    a = ap.parse_args()

    raw = np.fromfile(a.trace, dtype=np.int32)
    if raw.size % 2:
        sys.exit("trace is not an even number of int32")
    pairs = raw.reshape(-1, 2)
    layers, experts = pairs[:, 0], pairs[:, 1]
    keys = (layers.astype(np.int64) << 20) | experts.astype(np.int64)
    trace = keys.tolist()

    n = len(trace)
    uniq = len(set(trace))
    per_tok = 1472                      # top-16 x 92 MoE layers
    ntok = max(n // per_tok, 1)
    print("trace: %d requests, %d distinct experts, about %d token(s)" % (n, uniq, ntok))
    print("distinct experts touched: %d of %d (%.2f%% of the pool)"
          % (uniq, TOTAL_EXPERTS, 100.0 * uniq / TOTAL_EXPERTS))
    print("if nothing were cached: %.2f GB per token\n"
          % (per_tok * a.expert_bytes / 1e9))

    c = Counter(trace)
    top = c.most_common(10)
    print("hottest experts: " + ", ".join("L%d/e%d x%d" % (k >> 20, k & 0xFFFFF, v)
                                          for k, v in top[:6]))
    reuse = sum(v - 1 for v in c.values())
    print("total reuse: %d of %d requests are repeats (%.1f%%)\n"
          % (reuse, n, 100.0 * reuse / n))

    caps_gb = [8, 16, 32, 64, 128, 192, 256, 384, 512, 768, 1024, 1450]
    print("%-9s %8s %9s %9s %9s %9s %12s %11s" %
          ("CACHE", "SLOTS", "LRU", "S3-FIFO", "BELADY", "PIN+LRU",
           "GB READ/TOK", "SEC/TOK"))
    print("-" * 86)
    for gb in caps_gb:
        cap = int(gb * 1e9 // a.expert_bytes)
        if cap < 17:
            continue
        h = lru(trace, cap)
        s3 = s3fifo(trace, cap)
        b = belady(trace, cap)
        p = pinned_lru(trace, cap, cap // 2)
        # The read column follows the policy the engine RUNS, which is S3-FIFO. Quoting
        # LRU's miss count beside a table whose point is that LRU was replaced would
        # describe a configuration nobody uses.
        miss = n - s3
        gb_tok = miss * a.expert_bytes / 1e9 / ntok
        sec = gb_tok * 1000.0 / a.disk_mbs
        print("%-9s %8d %8.2f%% %8.2f%% %8.2f%% %8.2f%% %12.2f %11.2f"
              % ("%d GB" % gb, cap, 100.0 * h / n, 100.0 * s3 / n, 100.0 * b / n,
                 100.0 * p / n, gb_tok, sec))
    print("-" * 86)
    print("SEC/TOK is expert I/O only, at the measured %.0f MB/s random-cold rate, and\n"
          "follows the S3-FIFO column because that is the policy the engine runs.\n"
          "It excludes compute, which overlaps with it in a real serving loop."
          % a.disk_mbs)
    print("BELADY and PIN+LRU both use future knowledge, so they are ceilings rather\n"
          "than achievable policies. S3-FIFO is what the engine does; LRU is what it did.")
    comp = uniq
    print("compulsory misses: %d (every expert must be read at least once), a ceiling of\n"
          "%.2f%% hit rate for ANY policy at ANY size on this trace."
          % (comp, 100.0 * (n - comp) / n))

    full = uniq * a.expert_bytes / 1e9
    print("\nholding every expert this trace touched needs %.2f GB." % full)
    print("holding the whole pool needs %.2f GB." % (TOTAL_EXPERTS * a.expert_bytes / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
