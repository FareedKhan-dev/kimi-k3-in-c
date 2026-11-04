#!/usr/bin/env python
"""
budget.py - the resident-versus-streamable budget, measured from the real shard headers.

WHY MEASURE RATHER THAN DERIVE
    The obvious way to size the resident set is to multiply config values by hand. It
    gives the wrong answer, and the way it goes wrong is instructive: the shared experts
    are 6144 wide (num_shared_experts 2 x moe_intermediate 3072, fused into one KimiMLP
    at modeling_kimi_linear.py:797-799) and they ship BF16, not MXFP4. They are dense and
    active on every token, so they belong in the resident set, but they look like
    experts, so hand arithmetic files them under "streamable" and understates the floor.

    This script sums the actual bytes of every tensor in the checkpoint, classified by
    what the engine must do with each one, and assumes nothing about the layer count or
    the expert count.

    Classification rule: a tensor is STREAMABLE only if it is a ROUTED expert, because
    only routed experts are selected per token. Everything else runs on every token and
    must be resident, or paged in and out on every token, which is not streaming but
    thrashing.

HOW IT READS THE CHECKPOINT
    Only the header block at the front of each .safetensors shard, which is a JSON
    dictionary of tensor names, dtypes and shapes. That is a few MB in total against
    1.56 TB of shards, so this runs in seconds and needs no GPU, no torch, and no
    complete download, a partial checkout still gives an exact answer for the shards
    that are present.

usage: budget.py <model_dir>
       budget.py            (reads $K3_MODEL_DIR)
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re
import struct
import sys

WIDTH = {"U8": 1, "BF16": 2, "F16": 2, "F32": 4}
# MXFP4 stores half a byte per weight plus one E8M0 byte per 32-element group.
MXFP4_BYTES_PER_PARAM = 0.5 + 1.0 / 32.0


def classify(name: str) -> str:
    if ".block_sparse_moe.experts." in name:
        return "routed experts (STREAMABLE)"
    if ".block_sparse_moe.shared_experts." in name:
        return "shared experts (resident, dense)"
    if ".block_sparse_moe.routed_expert_" in name:
        return "MoE latent up/down/norm (resident)"
    if ".block_sparse_moe.gate" in name:
        return "MoE router (resident)"
    if ".self_attn." in name:
        return "attention (resident)"
    if "_res_norm" in name or "_res_proj" in name:
        return "attn-res aggregators (resident)"
    if "layernorm" in name or name.endswith(".norm.weight"):
        return "layer norms (resident)"
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding / lm_head (resident)"
    if name.startswith("language_model."):
        return "other text tower (resident)"
    return "vision / other (not used by the text path)"


def human(b: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1000.0:
            return "%7.2f %s" % (b, u)
        b /= 1000.0
    return "%7.2f PB" % b


def read_header(path: str) -> dict:
    """Return one shard's tensor index.

    A safetensors file starts with an 8-byte little-endian length followed by that many
    bytes of JSON. Only those bytes are read, so this touches a few hundred KB of a
    17 GB shard.

    The length is bounds-checked before it is used as a read size: it comes from a file
    that may have been fetched from a mirror, and an absurd value would otherwise turn
    into an absurd allocation.
    """
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("%s: too short to be a safetensors file" % path)
        n = struct.unpack("<Q", raw)[0]
        if not 0 < n < (1 << 30):
            raise ValueError("%s: implausible header length %d" % (path, n))
        blob = f.read(n)
        if len(blob) != n:
            raise ValueError("%s: header truncated (%d of %d bytes)" % (path, len(blob), n))
    return json.loads(blob.decode("utf-8"))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("K3_MODEL_DIR", "")
    if not root:
        print("usage: budget.py <model_dir>")
        print("       or set K3_MODEL_DIR")
        return 2

    files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
    if not files:
        print("no .safetensors shards in %s" % root)
        print("Point this at the directory holding the released checkpoint. A partial")
        print("download is fine: the totals below will simply cover the shards present.")
        return 1
    print("reading %d shard headers from %s\n" % (len(files), root))

    by_class = collections.Counter()
    cnt_class = collections.Counter()
    by_dtype = collections.Counter()
    ntensor = 0
    total = 0
    # routed expert params, counted from the packed shapes rather than assumed
    routed_params = 0
    routed_bytes = 0
    per_layer_expert_bytes = collections.Counter()
    expert_parts = collections.defaultdict(set)
    expert_sizes = collections.Counter()
    observed_layers = set()
    kda_layers, mla_layers = set(), set()

    for fp in files:
        h = read_header(fp)
        # safetensors puts tensor entries at the top level, alongside an optional
        # "__metadata__" key. Older extracts nested them under "tensors"; accept both so
        # a previously-extracted header still works.
        entries = h.get("tensors", h)
        for name, e in entries.items():
            if name == "__metadata__" or not isinstance(e, dict):
                continue
            a, b = e["data_offsets"]
            nb = b - a
            ntensor += 1
            total += nb
            c = classify(name)
            by_class[c] += nb
            cnt_class[c] += 1
            by_dtype[e["dtype"]] += nb

            layer = re.search(r"layers\.(\d+)\.", name)
            if layer:
                observed_layers.add(int(layer.group(1)))
            m = re.search(r"layers\.(\d+)\.self_attn\.(\w+)", name)
            if m:
                # A KDA layer has A_log and conv1d weights; an MLA layer has kv_a_proj.
                if m.group(2) in ("A_log",):
                    kda_layers.add(int(m.group(1)))
                if "kv_a_proj" in m.group(2) or "kv_b_proj" in m.group(2):
                    mla_layers.add(int(m.group(1)))

            if ".block_sparse_moe.experts." in name:
                routed_bytes += nb
                if name.endswith("weight_packed"):
                    routed_params += e["shape"][0] * e["shape"][1] * 2
                lm = re.search(r"layers\.(\d+)\.", name)
                if lm:
                    per_layer_expert_bytes[int(lm.group(1))] += nb
                expert = re.search(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)$", name)
                if expert:
                    key = (int(expert.group(1)), int(expert.group(2)))
                    expert_parts[key].add(expert.group(3))
                    expert_sizes[key] += nb

    if not total:
        print("no tensor bytes present; no hardware estimate is available")
        return 1

    print("%-42s %8s  %12s   %s" % ("CLASS", "TENSORS", "BYTES", "SHARE"))
    print("-" * 82)
    for c, b in by_class.most_common():
        print("%-42s %8d  %12s   %5.2f%%" % (c, cnt_class[c], human(b), 100.0 * b / total))
    print("-" * 82)
    print("%-42s %8d  %12s   100.00%%" % ("CHECKPOINT BYTES PRESENT", ntensor, human(total)))

    print("\nby dtype:")
    for d, b in by_dtype.most_common():
        print("   %-5s %12s   %5.2f%%" % (d, human(b), 100.0 * b / total))

    stream = by_class["routed experts (STREAMABLE)"]
    resident = total - stream - by_class.get("vision / other (not used by the text path)", 0)
    vision = by_class.get("vision / other (not used by the text path)", 0)

    print("\n" + "=" * 82)
    print("THE NUMBER THAT MATTERS")
    print("=" * 82)
    print("  resident (text path, must stay in RAM) : %s" % human(resident))
    print("  streamable (routed experts, from NVMe) : %s" % human(stream))
    print("  vision tower (text-only inference)     : %s" % human(vision))
    print("  checkpoint bytes in shards present     : %s" % human(total))

    print("\nrouted experts, measured:")
    print("  parameters            : %s" % format(routed_params, ","))
    print("  bytes                 : %s" % human(routed_bytes))
    if routed_params:
        print("  bytes per parameter   : %.6f  (MXFP4 predicts %.6f)"
              % (routed_bytes / routed_params, MXFP4_BYTES_PER_PARAM))
    else:
        print("  bytes per parameter   : unavailable (no packed routed weights present)")
    nlayers_moe = len(per_layer_expert_bytes)
    if nlayers_moe:
        per = sorted(set(per_layer_expert_bytes.values()))
        print("  MoE layers            : %d" % nlayers_moe)
        print("  bytes per MoE layer   : %s%s"
              % (human(per[0]), "" if len(per) == 1 else "  (VARIES: %d distinct)" % len(per)))
        print("  observed expert count : %d observed experts" % len(expert_parts))

    print("\nlayer map from the checkpoint itself:")
    print("  layers with A_log (KDA)      : %d" % len(kda_layers))
    print("  layers with kv_a/kv_b (MLA)  : %d" % len(mla_layers))
    both = kda_layers & mla_layers
    print("  layers claiming both         : %d%s" % (len(both), "  <-- WRONG" if both else ""))
    if observed_layers:
        missing = sorted(observed_layers - set(per_layer_expert_bytes))
        print("  observed layers without routed tensors: %s" % missing)

    required = {f"{matrix}.{kind}" for matrix in ("w1", "w2", "w3")
                for kind in ("weight_packed", "weight_scale")}
    if not expert_parts or any(not required.issubset(parts) for parts in expert_parts.values()):
        print("\nno complete routed expert set: incomplete/absent expert coverage; "
              "cache capacity cannot be projected")
        return 0
    if len(set(expert_sizes.values())) != 1:
        print("\nobserved experts have different sizes; cache capacity cannot be projected")
        return 0
    expert_count = len(expert_parts)
    expert_bytes = next(iter(expert_sizes.values()))
    print("\nwhat this means for hardware (shards present only):")
    for ram in (32, 48, 64, 96, 128, 192, 256):
        fits = ram * 1e9 - resident
        n = min(expert_count, int(fits / expert_bytes)) if fits > 0 else 0
        pct = 100.0 * n / expert_count
        print("  %3d GB RAM -> %s for experts, about %6d of %d cached (%.1f%%)"
              % (ram, human(fits) if fits > 0 else "  DOES NOT FIT",
                 max(n, 0), expert_count, pct))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
