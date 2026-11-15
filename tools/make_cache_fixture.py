#!/usr/bin/env python
"""
make_cache_fixture.py - a small safetensors shard holding COMPLETE routed experts, so the
streaming expert cache can be tested without the 1.56 TB checkpoint.

WHY THIS EXISTS
    The streaming cache is the part of this project everything else rests on, and no
    other test reaches it. Every op fixture and all three oracle gates exercise the
    RESIDENT expert bank (K3MoeW.w1/w3/w2); the cache path (K3MoeW.src) is only taken
    when running the real model against real shards.

    That gap is not theoretical. A batch-prefetch defect handing the same slot to several
    experts at once satisfies every op fixture and all three oracle gates, because none
    of them ever calls the cache, the only symptom is a different token from the real
    model, which is indistinguishable from any other difference without a reference to
    compare against. These fixtures make the cache testable on its own.

    The fixtures here are tiny -- six U8 tensors per expert, a few KB each -- but they are
    STRUCTURALLY real: the same six-tensor naming, the same U8 dtypes, laid out
    back-to-back so k3_expert_ref sees one contiguous run exactly as it does in the real
    checkpoint.

WHAT MAKES THE CONTENT USEFUL
    Every byte of expert e is derived from e, so if the cache ever returns expert a's
    bytes for expert b -- which is precisely what slot aliasing does -- the test sees it
    immediately rather than seeing plausible-looking noise.

usage: make_cache_fixture.py [out_dir] [n_experts]
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIX_CACHE

# A real expert's geometry, scaled down: w1 and w3 are [INTER][LATENT], w2 is
# [LATENT][INTER], the shapes k3_expert_check_shape holds every expert to. test_cache
# sets the same two numbers in its config. 3,264 bytes per expert, not a multiple of
# 4096, so O_DIRECT slot alignment is gated rather than coincidental.
LATENT, INTER, GROUP = 64, 32, 32
SHAPES = {0: (INTER, LATENT), 1: (LATENT, INTER), 2: (INTER, LATENT)}   # w1, w2, w3


def dims(which: int, packed: bool) -> tuple[int, int]:
    rows, logical = SHAPES[which]
    return rows, (logical // 2 if packed else logical // GROUP)
FMT = "language_model.model.layers.%d.block_sparse_moe.experts.%d.%s.%s"


def expert_bytes(e: int, which: int, packed: bool) -> np.ndarray:
    """Deterministic and expert-specific, so a mix-up cannot look like valid data."""
    rows, cols = dims(which, packed)
    n = rows * cols
    base = (e * 977 + which * 31 + (0 if packed else 7)) & 0xFF
    return ((np.arange(n, dtype=np.uint32) + base) % 251).astype(np.uint8)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else FIX_CACHE
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    os.makedirs(out, exist_ok=True)

    tensors, blob, off = {}, bytearray(), 0
    for e in range(n):
        # w1, w2, w3 each contribute packed then scale, IN THAT ORDER and adjacent, so
        # the six tensors of one expert form a single contiguous run on disk. That is the
        # property k3_expert_ref checks and the reason one expert is one pread.
        for wi, wname in enumerate(("w1", "w2", "w3")):
            for packed in (True, False):
                a = expert_bytes(e, wi, packed)
                shape = list(dims(wi, packed))
                name = FMT % (0, e, wname, "weight_packed" if packed else "weight_scale")
                tensors[name] = {"dtype": "U8", "shape": shape,
                                 "data_offsets": [off, off + a.nbytes]}
                blob += a.tobytes()
                off += a.nbytes

    hdr = json.dumps(tensors).encode("utf-8")
    pad = (-len(hdr)) % 8                      # safetensors wants an 8-byte aligned body
    hdr += b" " * pad
    path = os.path.join(out, "model-00001-of-00001.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(bytes(blob))

    per = sum(r * c for w in range(3) for r, c in (dims(w, True), dims(w, False)))
    print("wrote %s" % path)
    print("  %d experts, %d tensors, %d bytes each, %d bytes total"
          % (n, len(tensors), per, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
