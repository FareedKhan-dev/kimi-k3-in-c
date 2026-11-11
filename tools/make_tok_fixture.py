#!/usr/bin/env python
"""
make_tok_fixture.py - tiny tiktoken.model / tokenizer_config.json pairs, so the K3
tokenizer loader (src/tokenizer/k3_tok.h) is tested on a clean checkout.

WHY THIS EXISTS
    The released vocabulary ships with the checkpoint, so every tokenizer gate in
    `make test` reports NOT RUN in CI. The loader is then never exercised at all, and its
    failure modes are the silent kind: a corrupt or truncated tiktoken.model loads, emits
    ids, and is wrong. These fixtures are structurally real (base64 token bytes, a rank
    per line, added tokens keyed by id) and small enough to check in.

WHAT EACH DIRECTORY IS FOR
    good/            256 single bytes, four merges, two added tokens. "hello hello" must
                     encode to [259, 258, 32, 259, 258]: without rank BPE (k3_tok.h trap
                     #2) it degrades to single bytes, and without the byte-level keys
                     (trap #1) the space, stored as U+0120, is never found.
    dup_rank/        rank 7 on two lines: what a download cut inside the last rank field
                     looks like ("... 259" -> "... 2").
    missing_rank/    rank 100 absent from the middle.
    junk_rank/       a rank field "12x", which atoi read as 12.
    dup_bytes/       two ranks for the same bytes; the encoder can reach only one.
    added_collides/  an added token on a rank's id.
    truncated/       the last merge lines cut at a line boundary, so the added tokens no
                     longer start right after the last rank. Every line is valid on its
                     own; only the gap shows it.

usage: make_tok_fixture.py [out_dir]
"""
from __future__ import annotations

import base64
import json
import os
import sys

from _paths import FIX_TOK

MERGES = [b"ll", b"lo", b"llo", b"he"]
ADDED = [("[BOS]", True), ("<|im_end|>", True)]


def ranks() -> list[tuple[bytes, int]]:
    rows = [(bytes([b]), b) for b in range(256)]
    rows += [(m, 256 + i) for i, m in enumerate(MERGES)]
    return rows


def write(out: str, name: str, rows: list[tuple[bytes, str]], added: dict[int, tuple]) -> None:
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "tiktoken.model"), "wb") as f:
        for tok, rank in rows:
            f.write(base64.b64encode(tok) + b" " + str(rank).encode() + b"\n")
    cfg = {"added_tokens_decoder": {
        str(i): {"content": c, "special": s} for i, (c, s) in sorted(added.items())}}
    with open(os.path.join(d, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else FIX_TOK
    base = [(t, str(r)) for t, r in ranks()]
    n = len(base)
    added = {n + i: a for i, a in enumerate(ADDED)}

    write(out, "good", base, added)
    write(out, "dup_rank", base + [(b"\xfe\xfe", "7")], added)
    write(out, "missing_rank", base[:100] + base[101:], added)
    write(out, "junk_rank", base[:12] + [(base[12][0], "12x")] + base[13:], added)
    write(out, "dup_bytes", base + [(b"ll", str(n))], {n + 1 + i: a for i, a in enumerate(ADDED)})
    write(out, "added_collides", base, {100: ADDED[0], n: ADDED[1]})
    write(out, "truncated", base[:-2], added)
    return 0


if __name__ == "__main__":
    sys.exit(main())
