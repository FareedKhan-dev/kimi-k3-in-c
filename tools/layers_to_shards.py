#!/usr/bin/env python
"""
layers_to_shards.py - which shards does a `--layers N` run actually need?

WHY THIS EXISTS
    `--layers N` binds only the first N decoder layers, so it runs against a partial
    shard set. But which shards hold layers 0..N-1 is not something you can guess from
    the filenames: tensors are packed bin-first, so one layer spans several shards and
    one shard holds slices of several layers (layer 0 alone touches shards 1 AND 2,
    while the embedding table and the output head both live in shard 94). Guessing
    "the first K shards" downloads the wrong set and the engine refuses it with
    "missing tensors in this shard set" after the gigabytes have already moved.

    The Hub publishes the exact tensor-to-shard map as model.safetensors.index.json
    (59 MB, fetched by download-model.sh before this runs). This script reads that map
    and prints the minimal shard set for the requested prefix: every shard holding any
    tensor of layers 0..N-1, plus the shards holding the embedding table and the
    output head, which every run needs regardless of depth.

    usage: layers_to_shards.py <model.safetensors.index.json> --layers N

    Prints one shard filename per line, sorted, on stdout. Anything else is an error:
    exit 2, following the engine's own convention that a usage or config problem is
    exit 2, never a silent default.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")
EMBED_KEY = "language_model.model.embed_tokens.weight"
HEAD_KEY = "language_model.lm_head.weight"


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="map a --layers N prefix to shard files")
    p.add_argument("index", help="model.safetensors.index.json from the Hub")
    p.add_argument("--layers", required=True, help="layer-prefix depth N (>= 1)")
    args = p.parse_args(argv)
    try:
        layers = int(args.layers)
    except ValueError:
        p.error("--layers must be a positive integer, got %r" % args.layers)
    if layers < 1:
        p.error("--layers must be a positive integer, got %r" % args.layers)
    args.layers = layers
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        with open(args.index, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        print("layers_to_shards: cannot read %s: %s" % (args.index, e), file=sys.stderr)
        return 2
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        print(
            "layers_to_shards: %s has no usable weight_map; refusing to guess." % args.index,
            file=sys.stderr,
        )
        return 2

    max_layer = -1
    for name in weight_map:
        m = LAYER_RE.match(name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    if max_layer < 0:
        print(
            "layers_to_shards: no language_model.model.layers.* tensors in %s." % args.index,
            file=sys.stderr,
        )
        return 2
    if args.layers > max_layer + 1:
        print(
            "layers_to_shards: --layers %d exceeds the %d layers in %s."
            % (args.layers, max_layer + 1, args.index),
            file=sys.stderr,
        )
        return 2

    for key in (EMBED_KEY, HEAD_KEY):
        if key not in weight_map:
            print(
                "layers_to_shards: %s is absent from %s; refusing to guess where it lives."
                % (key, args.index),
                file=sys.stderr,
            )
            return 2

    need: set[str] = {weight_map[EMBED_KEY], weight_map[HEAD_KEY]}
    for name, shard in weight_map.items():
        m = LAYER_RE.match(name)
        if m and int(m.group(1)) < args.layers:
            need.add(shard)
    for shard in sorted(need):
        print(shard)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
