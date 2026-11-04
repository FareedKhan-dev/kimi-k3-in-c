"""Validate one fresh engine receipt and emit its benchmark TSV fields."""
import json
import math
import sys


def fields(result, prompt, generated):
    if result["prompt_ids"] != prompt:
        raise ValueError("prompt ids do not match the benchmark request")
    ids = result["generated_ids"]
    if not isinstance(ids, list) or len(ids) != generated or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("generated ids must contain the requested number of non-negative integers")
    if result["interrupted"] is not False or type(result["expert_drops"]) is not int or result["expert_drops"] != 0:
        raise ValueError("interrupted or numerically invalid run")
    layers, completed = result["layers_requested"], result["layers_completed"]
    if type(layers) is not int or type(completed) is not int or not 0 < layers == completed:
        raise ValueError("not all requested layers completed")
    seconds = result["seconds_per_token"]
    rss = result["peak_rss_bytes"]
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds_per_token must be finite and positive")
    if isinstance(rss, bool) or not math.isfinite(rss) or rss <= 0:
        raise ValueError("peak_rss_bytes must be finite and positive")
    read = 0
    for name in ("expert_bytes_read", "trunk_bytes_read", "embedding_bytes_read", "lm_head_bytes_read"):
        value = result[name]
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a non-negative integer")
        read += value
    return f"{seconds:.6f}\t{rss / 1e9:.6f}\t{read / 1e9:.6f}\t" + ",".join(map(str, ids))


def main():
    if len(sys.argv) != 4:
        print("usage: read_result.py <result.json> <prompt ids> <generation count>", file=sys.stderr)
        return 2
    try:
        with open(sys.argv[1], encoding="utf-8") as stream:
            result = json.load(stream)
        prompt = [int(value) for value in sys.argv[2].split(",")]
        print(fields(result, prompt, int(sys.argv[3])))
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        print("invalid benchmark result: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
