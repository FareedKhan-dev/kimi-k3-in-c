#!/usr/bin/env python
"""
pack_trunk.py - extract the resident trunk into one compact file on fast local storage.

WHY
    The engine holds all 110 GB of trunk in RAM because reading it per token from the
    network volume (measured 772 MB/s) would cost 142 s/token. Local NVMe measures
    4.5 GB/s, which turns the same read into 24 s/token, and the trunk is walked in a
    fixed order (layer 0..92, every token) so it prefetches perfectly and overlaps
    compute. That makes streaming the trunk viable, and streaming is LOSSLESS, unlike
    every quantisation route.

    Kimi K3's own technical report is the reason lossless matters here. Section 4.1.4:
    the MoE experts are MXFP4 with quantisation-aware training, "while all non-expert
    components (attention projections, latent MoE projections, shared experts, and MoE
    routers) remain in higher precision". That list is exactly this trunk. Moonshot
    deliberately did not quantise it, and post-hoc 4-bit measures 12% output error on
    real K3 weights against 0.54% for int8. So the trunk is copied here byte for byte.

WHAT MAKES THIS CHEAP
    Every layer's trunk tensors form ONE contiguous run inside its shard, verified for
    every layer before copying. So packing is 93 sequential range copies, and loading a
    layer later is a single pread from a known offset.

Writes <out>/trunk.bin and <out>/trunk.json.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time

CHUNK = 64 << 20
ALIGN = 4096        # O_DIRECT needs offset, length and buffer all aligned


def shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n


def main():
    if len(sys.argv) < 3:
        print("usage: pack_trunk.py <shard_dir> <out_dir> [n_layers]")
        return 2
    src, out = sys.argv[1], sys.argv[2]
    nlayers = int(sys.argv[3]) if len(sys.argv) > 3 else 93
    os.makedirs(out, exist_ok=True)

    shard_names = sorted(fn for fn in os.listdir(src) if fn.endswith(".safetensors"))
    print("indexing %d safetensors shards..." % len(shard_names))
    # name -> (shard_path, absolute_offset, nbytes, dtype, shape)
    where = {}
    skipped_routed = 0
    for fn in shard_names:
        p = os.path.join(src, fn)
        hdr, base = shard_header(p)
        for name, e in hdr.items():
            if name == "__metadata__":
                continue
            # Routed experts remain in the official shards and are streamed from there;
            # they are never copied into trunk.bin. Avoid retaining their hundreds of
            # thousands of Python dict entries just to reject them again per layer.
            # Routers and shared experts do not match this path and remain in the exact
            # dense trunk.
            if ".block_sparse_moe.experts." in name:
                skipped_routed += 1
                continue
            a, b = e["data_offsets"]
            where[name] = (p, base + a, b - a, e["dtype"], e["shape"])

    print("indexed %d trunk tensors; skipped %d routed-expert metadata entries"
          % (len(where), skipped_routed))

    manifest = {"n_layers": nlayers, "align": ALIGN, "layers": []}
    total = 0
    t0 = time.time()
    outp = os.path.join(out, "trunk.bin")
    with open(outp, "wb") as dst:
        for L in range(nlayers):
            pre = "language_model.model.layers.%d." % L
            mine = {k: v for k, v in where.items()
                    if k.startswith(pre) and ".block_sparse_moe.experts." not in k}
            if not mine:
                print("layer %d: NO TENSORS FOUND" % L)
                return 1
            shards = {v[0] for v in mine.values()}
            if len(shards) != 1:
                print("layer %d spans %d shards, cannot pack as one run" % (L, len(shards)))
                return 1
            sp = shards.pop()
            lo = min(v[1] for v in mine.values())
            hi = max(v[1] + v[2] for v in mine.values())
            own = sum(v[2] for v in mine.values())
            if hi - lo != own:
                # A gap would mean copying expert bytes too; refuse rather than bloat.
                print("layer %d trunk is NOT contiguous (span %d, own %d)" % (L, hi - lo, own))
                return 1

            # Start every run on a 4096 boundary so the engine can read it with
            # O_DIRECT. Under a tight cgroup the buffered path collapses (measured
            # 1,878 MB/s against 6,553 MB/s unconstrained) because the page cache has no
            # room and thrashes; O_DIRECT moves bytes straight into the engine's own
            # aligned buffer with no double-buffering. It costs 93 x under 4 KB of
            # padding in a 108 GB file.
            pad = (-dst.tell()) % ALIGN
            if pad:
                dst.write(b"\0" * pad)
            file_off = dst.tell()
            assert file_off % ALIGN == 0
            with open(sp, "rb") as f:
                f.seek(lo)
                left = own
                while left:
                    b = f.read(min(CHUNK, left))
                    if not b:
                        print("layer %d: short read" % L)
                        return 1
                    dst.write(b)
                    left -= len(b)
            # Pad the tail too: an O_DIRECT read is rounded up to ALIGN, so the last
            # layer must not run off the end of the file.
            pad = (-dst.tell()) % ALIGN
            if pad:
                dst.write(b"\0" * pad)

            manifest["layers"].append({
                "layer": L,
                "shard": os.path.basename(sp),
                "run_start": lo,          # absolute offset in the shard
                # The engine reads this many bytes with O_DIRECT, which requires the
                # LENGTH to be aligned as well as the offset. Real-checkpoint runs are
                # huge and align by luck; fixture-sized runs are not, which is why this
                # only ever showed up on small models as "short read on layer N".
                #
                # Rounded to ALIGN (4096) rather than to a 512-byte sector. 512 is not
                # enough on a 4Kn device, and it disagrees with the rest of this file:
                # runs are padded to ALIGN, file_off is asserted to be a multiple of
                # ALIGN, the manifest advertises ALIGN, and k3_trunk.c refuses O_DIRECT
                # outright unless that advertised value is 4096. The tail is already
                # zero-padded to ALIGN below, so this still reads only padding.
                "nbytes": (own + ALIGN - 1) & ~(ALIGN - 1),
                "file_off": file_off,     # offset in trunk.bin
                "tensors": {k: {"off": v[1] - lo, "nbytes": v[2],
                                "dtype": v[3], "shape": v[4]}
                            for k, v in sorted(mine.items())},
            })
            total += own
            if (L + 1) % 10 == 0 or L + 1 == nlayers:
                el = time.time() - t0
                print("  packed %d/%d layers, %.2f GB, %.0f s (%.0f MB/s)"
                      % (L + 1, nlayers, total / 1e9, el, total / 1e6 / max(el, 1e-9)))
                sys.stdout.flush()

    with open(os.path.join(out, "trunk.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    big = max(x["nbytes"] for x in manifest["layers"])
    print("\nwrote %s: %.2f GB across %d layers" % (outp, total / 1e9, nlayers))
    print("largest layer run: %.3f GB  <- the streaming slot size" % (big / 1e9))
    print("wrote %s" % os.path.join(out, "trunk.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
