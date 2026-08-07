#!/usr/bin/env python3
"""qdq_trunk.py: quantize-dequantize a packed trunk in place, for draft-quality tests.

Reads a packed trunk (trunk.bin + trunk.json), simulates storing every 2D bf16 weight
tensor at reduced precision (int8 or int4, per-row or per-group symmetric absmax), and
writes the dequantized result back as ORDINARY bf16 in an identical container. The
output trunk is byte-compatible with the engine: same manifest schema, same offsets,
same dtypes. Running the unmodified engine against it answers, with no new C code, the
question every quantized-draft design stands on: how often does the quantized model's
greedy token agree with the exact model's?

    python3 tools/qdq_trunk.py <src_trunk_dir> <dst_trunk_dir> --bits 8 [--group 0]

--group 0 means per-row scales; a positive value means per-group along the row.
Norms, biases and anything 1D pass through untouched, as do non-bf16 tensors.
Deterministic: same input bytes always produce the same output bytes.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(f32):
    u = f32.view(np.uint32)
    # round-to-nearest-even on the dropped mantissa bits, matching torch's cast
    rounding = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounding) >> 16).astype(np.uint16)


def qdq(f32, bits, group):
    rows, cols = f32.shape
    qmax = (1 << (bits - 1)) - 1          # 127 for int8, 7 for int4
    if group and group > 0 and cols % group == 0:
        g = f32.reshape(rows, cols // group, group)
        scale = np.abs(g).max(axis=2, keepdims=True) / qmax
        scale[scale == 0] = 1.0
        q = np.clip(np.rint(g / scale), -qmax, qmax)
        return (q * scale).reshape(rows, cols).astype(np.float32)
    scale = np.abs(f32).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(f32 / scale), -qmax, qmax)
    return (q * scale).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8))
    ap.add_argument("--group", type=int, default=0)
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.src, "trunk.json")))
    os.makedirs(args.dst, exist_ok=True)
    shutil.copyfile(os.path.join(args.src, "trunk.json"),
                    os.path.join(args.dst, "trunk.json"))

    src = open(os.path.join(args.src, "trunk.bin"), "rb")
    total = os.path.getsize(os.path.join(args.src, "trunk.bin"))
    dst = open(os.path.join(args.dst, "trunk.bin"), "wb")
    # start from a byte-identical copy so padding and non-quantized regions carry over
    print("copying %.1f GB container..." % (total / 1e9))
    shutil.copyfileobj(src, dst, 64 << 20)
    dst.flush()
    src.close()
    dst.close()

    dst = open(os.path.join(args.dst, "trunk.bin"), "r+b")
    done_t = 0
    done_b = 0
    for lay in man["layers"]:
        base = lay["file_off"]
        for _name, t in lay["tensors"].items():
            if t.get("dtype") not in ("BF16", "bf16"):
                continue
            shape = t.get("shape") or []
            if len(shape) != 2:
                continue                      # norms, biases: pass through exact
            rows, cols = int(shape[0]), int(shape[1])
            nb = int(t["nbytes"])
            if nb != rows * cols * 2:
                continue                      # unexpected layout: leave untouched
            off = base + int(t["off"])
            dst.seek(off)
            raw = np.frombuffer(dst.read(nb), dtype=np.uint16).copy()
            f32 = bf16_to_f32(raw).reshape(rows, cols)
            out = f32_to_bf16(qdq(f32, args.bits, args.group).ravel())
            dst.seek(off)
            dst.write(out.tobytes())
            done_t += 1
            done_b += nb
        print("  layer %3d done (%d tensors, %.1f GB so far)"
              % (lay.get("layer", -1), done_t, done_b / 1e9), flush=True)
    dst.close()
    print("qdq complete: %d tensors, %.1f GB re-quantized at int%d group=%s"
          % (done_t, done_b / 1e9, args.bits, args.group or "row"))


if __name__ == "__main__":
    sys.exit(main())
