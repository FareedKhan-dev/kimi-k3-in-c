#!/usr/bin/env python3
"""int8_trunk.py - derive a per-row int8 draft trunk from a packed bf16 trunk.

The hybrid decoder (--draft-trunk) uses a quantized copy of the trunk as a DRAFT that
proposes tokens; the exact bf16 model then verifies every one, so the draft's precision
affects only speed, never output. A bf16-format draft is 109 GB and cannot be resident on
a 64 GB machine, which is where the hybrid pays off. This writes a true int8 container at
~56 GB.

Input is a packed trunk (trunk.bin + trunk.json from pack_trunk.py). For every 2D BF16
tensor, each row becomes [f32 scale][int8 * cols] with a symmetric per-row absmax scale,
which is exactly the layout k3_matmul_q8 reads. Everything else (norms, 1D tensors,
anything not BF16) is copied verbatim. Run alignment and the manifest structure are
preserved so the engine's trunk reader and bind path handle it unchanged apart from the
I8R dtype.

    python3 tools/int8_trunk.py <bf16_trunk_dir> <int8_out_dir>
"""
import json
import os
import sys

import numpy as np

ALIGN = 4096
CHUNK = 64 << 20


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def quant_row_int8(f32_2d):
    """[rows, cols] f32 -> bytes: per row [f32 scale][int8 * cols]."""
    rows, cols = f32_2d.shape
    amax = np.abs(f32_2d).max(axis=1)
    scale = (amax / 127.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.rint(f32_2d / scale[:, None]).clip(-128, 127).astype(np.int8)
    # each row is [4 scale bytes][cols int8]; build [rows, 4+cols] uint8 in one shot
    row = np.empty((rows, 4 + cols), dtype=np.uint8)
    row[:, :4] = scale.view(np.uint8).reshape(rows, 4)
    row[:, 4:] = q.view(np.uint8)
    return row.tobytes(), rows * (4 + cols)


def main():
    if len(sys.argv) < 3:
        print("usage: int8_trunk.py <bf16_trunk_dir> <int8_out_dir>")
        return 2
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    man = json.load(open(os.path.join(src, "trunk.json")))
    binp = open(os.path.join(src, "trunk.bin"), "rb")
    outp = open(os.path.join(out, "trunk.bin"), "wb")

    newman = {k: v for k, v in man.items() if k != "layers"}
    newman["layers"] = []
    nq = 0
    for li, lay in enumerate(man["layers"]):
        base = lay["file_off"]
        # read the whole run once
        binp.seek(base)
        run = binp.read(lay["nbytes"])

        # rebuild the run tensor by tensor, in ascending source offset so the packed
        # order is preserved; new offsets are assigned as we append.
        items = sorted(lay["tensors"].items(), key=lambda kv: kv[1]["off"])
        pad = (-outp.tell()) % ALIGN
        if pad:
            outp.write(b"\0" * pad)
        file_off = outp.tell()
        assert file_off % ALIGN == 0

        new_tensors = {}
        run_pos = 0
        for name, t in items:
            o, nb, dt, shape = t["off"], t["nbytes"], t["dtype"], t.get("shape", [])
            raw = run[o:o + nb]
            if dt == "BF16" and len(shape) == 2:
                f = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape[0], shape[1])
                enc, enb = quant_row_int8(f)
                outp.write(enc)
                new_tensors[name] = {"off": run_pos, "nbytes": enb, "dtype": "I8R",
                                     "shape": shape}
                run_pos += enb
                nq += 1
            else:
                outp.write(raw)
                new_tensors[name] = {"off": run_pos, "nbytes": nb, "dtype": dt,
                                     "shape": shape}
                run_pos += nb

        pad = (-outp.tell()) % ALIGN
        if pad:
            outp.write(b"\0" * pad)
        run_bytes = (run_pos + ALIGN - 1) & ~(ALIGN - 1)
        nl = {k: v for k, v in lay.items() if k not in ("file_off", "nbytes", "tensors")}
        nl.update({"file_off": file_off, "nbytes": run_bytes, "tensors": new_tensors})
        newman["layers"].append(nl)
        if (li + 1) % 10 == 0 or li + 1 == len(man["layers"]):
            print("  int8 %d/%d layers, %.1f GB out" % (li + 1, len(man["layers"]),
                                                        outp.tell() / 1e9), flush=True)

    binp.close()
    outp.close()
    json.dump(newman, open(os.path.join(out, "trunk.json"), "w"))
    sz = os.path.getsize(os.path.join(out, "trunk.bin"))
    print("wrote %s: %.2f GB, %d tensors int8-quantised" % (out, sz / 1e9, nq))
    return 0


if __name__ == "__main__":
    sys.exit(main())
