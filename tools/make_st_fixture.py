#!/usr/bin/env python
"""
make_st_fixture.py - synthetic safetensors shards that exercise the awkward cases.

WHY SYNTHETIC FILES WHEN A REAL CHECKPOINT EXISTS
    The real shards are 17 GB each and contain only the three dtypes Moonshot happened
    to use, all well behaved. They cannot tell me whether the reader handles a scalar
    shape, a zero-element tensor, an F16 subnormal, a NaN, a name containing an escape,
    or a header with __metadata__ in it. Those are exactly the inputs where a
    hand-written scanner goes wrong, and every one of them fails SILENTLY: a misparsed
    offset still reads bytes, just the wrong ones.

    So: synthetic files for coverage, the real checkpoint for authenticity. Both.

FORMAT WRITTEN HERE
    [8 bytes little-endian N][N bytes JSON][data], data_offsets relative to end of header.
    Identical to what tools/fetch_mxfp4.py verified against the released shards.
"""
from __future__ import annotations

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import FIX_ST

OUT = sys.argv[1] if len(sys.argv) > 1 else FIX_ST


def bf16_bits(f32: np.ndarray) -> np.ndarray:
    """Top 16 bits of each float32. Truncation, not rounding, so the round trip
    bf16 -> f32 is exact and the expected values are unambiguous."""
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def write_shard(path: str, tensors: dict, metadata: dict | None = None) -> dict:
    """tensors: name -> (dtype_string, numpy array of the RAW stored bytes)."""
    header, blobs, off = {}, [], 0
    if metadata is not None:
        header["__metadata__"] = metadata
    for name, (dt, arr, shape) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hj = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    return header


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)

    # ---- shard A: the awkward cases -------------------------------------------------
    f32 = (rng.standard_normal(256) * 3.0).astype(np.float32)

    # bf16 generated from the STORED side so the expectation is exact, and seeded with
    # specials: +inf, -inf, NaN, +0, -0, and the smallest normal.
    bits = rng.integers(0, 65536, size=128, dtype=np.uint16)
    bits[0], bits[1], bits[2] = 0x7F80, 0xFF80, 0x7FC0      # +inf, -inf, NaN
    bits[3], bits[4], bits[5] = 0x0000, 0x8000, 0x0080      # +0, -0, min normal
    bf16 = bits

    # f16 with subnormals, which need renormalisation on widening and are the single
    # most likely thing to be wrong in a hand-written converter.
    f16 = np.zeros(64, dtype=np.float16)
    f16[0] = np.float16(np.inf); f16[1] = np.float16(-np.inf); f16[2] = np.float16(np.nan)
    f16[3] = np.float16(0.0);    f16[4] = np.float16(-0.0)
    f16[5] = np.frombuffer(np.uint16(0x0001).tobytes(), dtype=np.float16)[0]   # min subnormal
    f16[6] = np.frombuffer(np.uint16(0x03FF).tobytes(), dtype=np.float16)[0]   # max subnormal
    f16[7] = np.frombuffer(np.uint16(0x0400).tobytes(), dtype=np.float16)[0]   # min normal
    f16[8:] = rng.standard_normal(56).astype(np.float16)

    u8 = rng.integers(0, 256, size=(7, 32), dtype=np.uint8)

    # Small, deliberately non-vector-width model matrices for the streamed
    # embedding/lm_head parity gate. Seven columns exercises the scalar tail after the
    # four-accumulator loop; eleven rows makes chunk/order mistakes visible.
    stream_rng = np.random.default_rng(29)
    stream_embed = bf16_bits(
        (stream_rng.standard_normal((11, 7)) * 0.25).astype(np.float32)
    )
    stream_head = bf16_bits(
        (stream_rng.standard_normal((11, 7)) * 0.5).astype(np.float32)
    )

    A = {
        "plain.f32.2d":        ("F32",  f32, (16, 16)),
        "plain.bf16.1d":       ("BF16", bf16, (128,)),
        "tricky.f16.1d":       ("F16",  f16, (64,)),
        "packed.u8.2d":        ("U8",   u8, (7, 32)),
        "language_model.model.embed_tokens.weight":
                               ("BF16", stream_embed, (11, 7)),
        "language_model.lm_head.weight":
                               ("BF16", stream_head, (11, 7)),
        # A scalar has shape []. numel must be 1, not 0, or the size check rejects it.
        "scalar.f32":          ("F32",  np.float32([3.5]), ()),
        # A zero-element tensor is legal and spans zero bytes. A reader that treats
        # nbytes == 0 as an error, or that divides by numel, dies here.
        "empty.f32":           ("F32",  np.zeros(0, np.float32), (0, 8)),
        # Deep dotted names sharing a long prefix, which is what the real checkpoint
        # looks like and what stresses the hash.
        "language_model.model.layers.0.block_sparse_moe.experts.0.w1.weight_packed":
                               ("U8", rng.integers(0, 256, 64, dtype=np.uint8), (2, 32)),
        "language_model.model.layers.0.block_sparse_moe.experts.1.w1.weight_packed":
                               ("U8", rng.integers(0, 256, 64, dtype=np.uint8), (2, 32)),
        # 3D and 4D, to prove ndim is not assumed to be 2.
        "rank3.f32":           ("F32", rng.standard_normal(24).astype(np.float32), (2, 3, 4)),
        "rank4.u8":            ("U8",  rng.integers(0, 256, 16, dtype=np.uint8), (2, 2, 2, 2)),
    }
    ha = write_shard(os.path.join(OUT, "model-00001-of-00002.safetensors"), A,
                     metadata={"format": "pt", "note": "__metadata__ must be skipped, "
                                                       "not indexed as a tensor"})

    # ---- shard B: a second file, to prove sharding and stable shard ordering ---------
    B = {
        "second.shard.f32": ("F32", rng.standard_normal(100).astype(np.float32), (10, 10)),
        "second.shard.u8":  ("U8",  rng.integers(0, 256, 300, dtype=np.uint8), (300,)),
        # A name with a quote and a backslash. Legal JSON, and it breaks any scanner
        # that finds the closing quote by searching for the next '"'.
        'weird\\name."with"quotes': ("F32", np.float32([1, 2, 3]), (3,)),
    }
    hb = write_shard(os.path.join(OUT, "model-00002-of-00002.safetensors"), B)

    n = len(ha) - 1 + len(hb)      # minus __metadata__
    print("wrote %d shards, %d tensors to %s" % (2, n, OUT))
    for k, v in list(ha.items()) + list(hb.items()):
        if k == "__metadata__":
            continue
        print("  %-72s %-5s %s" % (k, v["dtype"], v["shape"]))


if __name__ == "__main__":
    main()
