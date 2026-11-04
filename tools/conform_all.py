#!/usr/bin/env python
"""
conform_all.py - run the real-weights layer conformance check across ALL 93 layers.

WHY THIS EXISTS
    The rest of the test suite covers two extremes and leaves a gap between them. The op
    fixtures and the full-model oracle use a TINY model (hidden 128, 13 layers) built by
    this project: they prove the arithmetic and the composition, but say nothing about
    the real checkpoint. test_real_layer proves the BINDING for ONE layer of the real
    model. Neither covers whether the binding is right for all 93 layers, whether the
    layer map places KDA and MLA correctly across the whole stack, or whether any single
    layer's weights are transposed, swapped, or off by one shard.

    That gap matters because the failure mode is invisible. A layer bound to the wrong
    tensor of the right shape still produces finite, plausible numbers. Only comparison
    against an independent implementation reading the same bytes can see it.

WHAT IT DOES
    For each layer L in 0..92:
      1. c/test_real_layer dumps that layer's staged outputs from the real shards
         (attention -> router -> MoE), computed by the C engine.
      2. tools/verify_real_layer.py recomputes every stage in torch from the same
         shards and compares, at a tolerance of 1.2e-7 * sqrt(width) * 50, which is
         float32 rounding for a dot product of that accumulation width.
      3. The per-stage relative errors are recorded.

    A layer counts as PASSED only with a fresh matching dump, a successful verifier,
    and recorded finite passing stage evidence. Unique dump paths are retained in
    conform_all.json; an old real_layer.json is never reused as this run's input.

usage: conform_all.py <shard_dir> <c_dir> <python> [first] [last] [T] [cache_gb]
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

STAGE_RE = re.compile(r"^\s{2}(\S+)\s+max\|diff\|\s+(\S+)\s+rel\s+(\S+)\s+budget\s+(\S+)\s+(OK|MISMATCH)")


def dump_kind(path, layer, tokens):
    with open(path, encoding="utf-8") as stream:
        dump = json.load(stream)
    if not isinstance(dump, dict):
        raise ValueError("dump must be a JSON object")
    if type(dump.get("layer")) is not int or dump["layer"] != layer:
        raise ValueError("dump layer does not match the request")
    if type(dump.get("T")) is not int or dump["T"] != tokens:
        raise ValueError("dump token count does not match the request")
    if type(dump.get("is_mla")) is not int or dump["is_mla"] not in (0, 1):
        raise ValueError("dump attention kind is invalid")
    return "MLA" if dump["is_mla"] else "KDA"


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    shards, cdir, py = sys.argv[1], sys.argv[2], sys.argv[3]
    first = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    last = int(sys.argv[5]) if len(sys.argv) > 5 else 92
    T = sys.argv[6] if len(sys.argv) > 6 else "4"
    cache = sys.argv[7] if len(sys.argv) > 7 else "2"
    try:
        tokens = int(T)
        if tokens < 1:
            raise ValueError
    except ValueError:
        print("T must be a positive integer", file=sys.stderr)
        return 2

    rows = []
    t_start = time.time()
    for L in range(first, last + 1):
        t0 = time.time()
        fd, dump_path = tempfile.mkstemp(prefix="real_layer_%02d_" % L, suffix=".json", dir=cdir)
        os.close(fd)
        dump_path = os.path.abspath(dump_path)
        r = subprocess.run([os.path.join(cdir, "test_real_layer"), shards, str(L), T, cache, dump_path],
                           cwd=cdir, capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            rows.append({"layer": L, "kind": "?", "ok": False,
                         "why": "test_real_layer exit %d" % r.returncode,
                         "tail": r.stdout[-400:] + r.stderr[-400:], "stages": {}, "dump_path": dump_path})
            print("L%-3d  DUMP FAILED (exit %d)" % (L, r.returncode), flush=True)
            continue
        try:
            kind = dump_kind(dump_path, L, tokens)
        except (OSError, ValueError, TypeError) as error:
            rows.append({"layer": L, "kind": "?", "ok": False,
                         "why": "invalid fresh dump: " + str(error), "stages": {}, "dump_path": dump_path})
            print("L%-3d  INVALID FRESH DUMP: %s" % (L, error), flush=True)
            continue

        v = subprocess.run([py, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "verify_real_layer.py"),
                            shards, dump_path],
                           cwd=cdir, capture_output=True, text=True, timeout=7200)
        stages = {}
        invalid_stage = False
        for line in v.stdout.splitlines():
            m = STAGE_RE.match(line)
            if m:
                try:
                    rel, budget = float(m.group(3)), float(m.group(4))
                    valid = (math.isfinite(rel) and rel >= 0 and math.isfinite(budget) and
                             budget > 0 and math.isfinite(rel / budget) and m.group(5) == "OK")
                except ValueError:
                    invalid_stage = True
                    continue
                invalid_stage = invalid_stage or not valid
                stages[m.group(1)] = {"rel": rel if math.isfinite(rel) else None,
                                      "budget": budget if math.isfinite(budget) else None, "ok": valid}
        evidence = bool(stages) and not invalid_stage
        ok = v.returncode == 0 and evidence
        worst = max((s["rel"] / s["budget"] for s in stages.values())) if evidence else None
        rows.append({"layer": L, "kind": kind, "ok": ok, "stages": stages,
                     "worst_ratio": worst, "seconds": time.time() - t0, "dump_path": dump_path,
                     "tail": "" if ok else (v.stdout[-1200:] + v.stderr[-600:])})
        print("L%-3d %-3s  %-4s  %2d stages  worst %.2fx budget   %.0fs" %
              (L, kind, "PASS" if ok else "FAIL", len(stages), worst if worst is not None else float("nan"), time.time() - t0),
              flush=True)
        if not ok:
            for line in v.stdout.splitlines():
                if "MISMATCH" in line or line.startswith("FAILED") or line.startswith("   "):
                    print("        " + line.strip(), flush=True)

    npass = sum(1 for r in rows if r["ok"])
    nfail = len(rows) - npass
    kda = [r for r in rows if r["kind"] == "KDA"]
    mla = [r for r in rows if r["kind"] == "MLA"]
    print("\n" + "=" * 78)
    print("LAYERS %d..%d   %d passed, %d failed   (%d KDA, %d MLA)   %.0f s total"
          % (first, last, npass, nfail, len(kda), len(mla), time.time() - t_start))
    good = [r["worst_ratio"] for r in rows if r["ok"] and r["worst_ratio"] == r["worst_ratio"]]
    if good:
        print("worst stage across all passing layers: %.2fx of its rounding budget"
              % max(good))
    if nfail:
        print("\nFAILING LAYERS:")
        for r in rows:
            if not r["ok"]:
                print("  L%d (%s): %s" % (r["layer"], r["kind"], r.get("why", "stage mismatch")))
    print("=" * 78)
    print("VERDICT: %s" % ("ALL LAYERS CONFORM" if nfail == 0 else "%d LAYER(S) DISAGREE" % nfail))

    with open(os.path.join(cdir, "conform_all.json"), "w") as f:
        json.dump({"first": first, "last": last, "passed": npass, "failed": nfail,
                   "rows": rows}, f, indent=1, allow_nan=False)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
