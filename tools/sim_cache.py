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
    LRU      what the engine actually implements.
    Belady   the optimal offline policy, which no real cache can beat. The gap between
             the two is the most a cleverer replacement policy could ever be worth. If
             the gap is small, effort belongs in prefetching or pinning, not in
             replacement.
    Pinned   the hottest N experts held permanently, LRU for the rest. This is what the
             usage histogram is for.

usage: sim_cache.py expert_trace.bin [--expert-bytes 17547264] [--disk-mbs 1234]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict, Counter

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

    if a.expert_bytes <= 0:
        ap.error("--expert-bytes must be positive")
    if not math.isfinite(a.disk_mbs) or a.disk_mbs <= 0:
        ap.error("--disk-mbs must be finite and positive")
    try:
        if os.path.getsize(a.trace) % (2 * np.dtype(np.int32).itemsize):
            ap.error("trace must contain complete pairs of int32 values")
        raw = np.fromfile(a.trace, dtype=np.int32)
    except OSError as error:
        ap.error(str(error))
    if raw.size == 0:
        ap.error("trace must contain at least one request")
    pairs = raw.reshape(-1, 2)
    layers, experts = pairs[:, 0], pairs[:, 1]
    if np.any(layers < 0) or np.any(experts < 0) or np.any(experts >= (1 << 20)):
        ap.error("trace layer/expert ids must be non-negative; expert ids must fit 20 bits")
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
    print("%-9s %8s %9s %9s %9s %12s %11s" %
          ("CACHE", "SLOTS", "LRU", "BELADY", "PIN+LRU", "GB READ/TOK", "SEC/TOK"))
    print("-" * 76)
    for gb in caps_gb:
        cap = int(gb * 1e9 // a.expert_bytes)
        if cap < 17:
            continue
        h = lru(trace, cap)
        b = belady(trace, cap)
        p = pinned_lru(trace, cap, cap // 2)
        miss = n - h
        gb_tok = miss * a.expert_bytes / 1e9 / ntok
        sec = gb_tok * 1000.0 / a.disk_mbs
        print("%-9s %8d %8.2f%% %8.2f%% %8.2f%% %12.2f %11.2f"
              % ("%d GB" % gb, cap, 100.0 * h / n, 100.0 * b / n, 100.0 * p / n,
                 gb_tok, sec))
    print("-" * 76)
    print("SEC/TOK is expert I/O only, at the supplied %.0f MB/s random-cold rate.\n"
          "It excludes compute, which overlaps with it in a real serving loop."
          % a.disk_mbs)
    print("BELADY and PIN+LRU both use future knowledge, so they are ceilings rather\n"
          "than achievable policies. LRU is what the engine actually does.")
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
