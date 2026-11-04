#!/usr/bin/env python
"""
cmp_logits.py - compare the C engine's final-position logits against the torch reference.

This is the definitive conformance check for the real model. Everything else compares
either tokens (argmax, which hides near-ties) or single layers in isolation. Two engines
can emit identical tokens for a whole generation while their logit vectors differ by a
large systematic amount, because argmax only cares which entry is biggest. Comparing the
full 163,840-wide vector elementwise is what actually catches an error in the final norm,
the lm_head, or the model-level AttnRes aggregator.

Inputs:
  c_logits.bin   raw float32 from k3_run --dump-logits
  ref.json       from tools/ref_forward.py

The tolerance is the same yardstick tools/verify_real_layer.py uses: float32 resolution
scaled by the square root of the accumulation width, times a slack factor. The lm_head
is a dot product of length hidden_size, and the hidden state arriving at it has already
accumulated error through 93 layers, so the budget is generous by design. What it is
looking for is a STRUCTURAL error, not the last bit.

usage: cmp_logits.py <c_logits.bin> <ref.json> [hidden]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cbin, refp = sys.argv[1], sys.argv[2]
    hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 7168

    # The first argument is normally k3_run's raw float32 dump, but accepting a
    # ref_forward JSON too lets this same comparator score one REFERENCE variant against
    # another -- for example an int8 simulation against the bf16 golden vector -- without
    # a second, subtly different comparison script.
    if cbin.endswith(".json"):
        C = json.load(open(cbin))
        c = np.array(C["logits_bits"], dtype=np.uint32).view(np.float32)
    else:
        C = {}
        if os.path.getsize(cbin) % np.dtype(np.float32).itemsize:
            print("INVALID INPUT: logit dump is not a whole number of float32 values")
            return 1
        c = np.fromfile(cbin, dtype=np.float32)
    R = json.load(open(refp))
    r = np.array(R["logits_bits"], dtype=np.uint32).view(np.float32)

    if any(v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() for v in (c, r)):
        print("INVALID INPUT: logits must be nonempty, finite, one-dimensional vectors")
        return 1

    if c.shape != r.shape:
        print("SHAPE MISMATCH: C has %d logits, reference has %d" % (c.size, r.size))
        return 1

    # Cross-check that both sides ran the SAME prompt. The C dump is a bare headerless
    # float32 blob, so nothing in the two files otherwise ties them together: comparing a
    # C run on one prompt against a reference on another would produce a confident,
    # meaningless mismatch. k3_run writes its prompt to the --out JSON; if that file is
    # alongside the dump, require agreement rather than assuming it.
    ref_ids = R.get("prompt_ids")
    if C.get("prompt_ids") is not None:
        if list(C["prompt_ids"]) != list(ref_ids or []):
            print("PROMPT MISMATCH: %s vs %s" % (C["prompt_ids"], ref_ids)); return 1
        print("prompt cross-check     : both sides ran %s" % (list(ref_ids),))
        crun = None
    else:
        crun = os.path.join(os.path.dirname(os.path.abspath(cbin)), "c_run.json")
    if crun is not None and ref_ids is not None and os.path.exists(crun):
        try:
            c_ids = json.load(open(crun)).get("prompt_ids")
        except Exception:
            c_ids = None
        if c_ids is not None:
            if list(c_ids) != list(ref_ids):
                print("PROMPT MISMATCH: C ran %s, reference ran %s" % (c_ids, ref_ids))
                print("These two runs are not comparable. Aborting rather than reporting "
                      "a meaningless difference.")
                return 1
            print("prompt cross-check     : both sides ran %s" % (list(c_ids),))
    elif crun is not None:
        print("prompt cross-check     : SKIPPED (no c_run.json beside the dump)")

    d = np.abs(c.astype(np.float64) - r.astype(np.float64))
    scale = max(float(np.abs(r).max()), 1e-30)
    rel = float(d.max()) / scale
    budget = 1.2e-7 * np.sqrt(hidden) * 50

    ca, ra = int(np.argmax(c)), int(np.argmax(r))
    # Rank correlation on the top of the distribution is the property that actually
    # decides generated text, so report it separately from the raw elementwise diff.
    ctop = set(np.argsort(-c)[:10].tolist())
    rtop = set(np.argsort(-r)[:10].tolist())

    print("vocab                 : %d" % c.size)
    print("C argmax              : %d  (logit %.6f)" % (ca, c[ca]))
    print("reference argmax      : %d  (logit %.6f)" % (ra, r[ra]))
    print("top-10 overlap        : %d/10" % len(ctop & rtop))
    print("max |diff|            : %.6e" % float(d.max()))
    print("relative to max|ref|  : %.6e   (budget %.1e)" % (rel, budget))
    print("mean |diff|           : %.6e" % float(d.mean()))
    print("correlation           : %.9f"
          % float(np.corrcoef(c.astype(np.float64), r.astype(np.float64))[0, 1]))

    ok_tok = ca == ra
    ok_num = rel <= budget
    # A genuine near-tie can flip argmax between two numerically-agreeing engines, so an
    # argmax difference is only damning when the logits themselves also disagree. Report
    # the gap so a tie is distinguishable from a real divergence rather than silently
    # failing a correct engine.
    if not ok_tok:
        gap = abs(float(r[ra]) - float(r[ca])) / max(abs(float(r[ra])), 1e-30)
        print("argmax differs; reference gap between the two candidates: %.3e relative"
              % gap)
    print()
    if ok_tok and ok_num:
        print("VERIFIED: the supplied final-position logit vectors agree elementwise "
              "within the reported tolerance, and their argmax tokens agree.")
        return 0
    if not ok_tok:
        print("TOKEN MISMATCH: C would emit %d, the reference would emit %d" % (ca, ra))
    if not ok_num:
        print("NUMERIC MISMATCH: relative error %.3e exceeds the %.1e budget" % (rel, budget))
        i = int(np.argmax(d))
        print("  worst at index %d: C=%.9g reference=%.9g" % (i, c[i], r[i]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
