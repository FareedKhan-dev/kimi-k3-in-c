#!/usr/bin/env bash
# One-token, full-checkpoint acceptance run for an 8 GB-class Jetson.
set -euo pipefail

MODEL="${1:?usage: jetson-proof-of-life.sh <model_dir> <trunk_dir> [output_dir]}"
TRUNK="${2:?usage: jetson-proof-of-life.sh <model_dir> <trunk_dir> [output_dir]}"
OUT="${3:-$PWD/k3-proof-$(date +%Y%m%d-%H%M%S)}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPECT_SHARDS=96
EXPECT_BYTES=1560936091448
EXPECT_TRUNK_BYTES=108811952128
EXPECT_TOKEN=17374
MAX_RSS_BYTES=$((6 * 1024 * 1024 * 1024))
VERIFY_MARKER="$MODEL/.k3_hf_verified"

case "$(uname -m)" in
    aarch64|arm64) ;;
    *) echo "FAIL: this runner is for ARM64/Jetson, got $(uname -m)" >&2; exit 2 ;;
esac
[ -d "$MODEL" ] || { echo "FAIL: no model directory $MODEL" >&2; exit 2; }
[ -f "$TRUNK/trunk.bin" ] && [ -f "$TRUNK/trunk.json" ] || {
    echo "FAIL: $TRUNK is not a packed trunk directory" >&2; exit 2;
}
[ -x "$ROOT/bin/k3" ] || {
    echo "FAIL: $ROOT/bin/k3 is not built; run make first" >&2; exit 2;
}

# Count and byte total catch partial transfers. The marker below is stronger: it is
# created only after `hf cache verify` checks the official immutable Hub snapshot.
N=$(find "$MODEL" -maxdepth 1 -name '*.safetensors' | wc -l)
B=$(find "$MODEL" -maxdepth 1 -name '*.safetensors' -printf '%s\n' \
    | awk '{s+=$1} END{printf "%.0f\n", s}')
[ "$N" -eq "$EXPECT_SHARDS" ] || {
    echo "FAIL: checkpoint has $N shards, expected $EXPECT_SHARDS" >&2; exit 2;
}
[ "$B" -eq "$EXPECT_BYTES" ] || {
    echo "FAIL: checkpoint has $B safetensors bytes, expected $EXPECT_BYTES" >&2; exit 2;
}
[ -f "$VERIFY_MARKER" ] || {
    echo "FAIL: no checksum-verification marker $VERIFY_MARKER" >&2
    echo "      run scripts/download-model.sh $MODEL first" >&2
    exit 2
}
if find "$MODEL" -maxdepth 1 -name '*.safetensors' -newer "$VERIFY_MARKER" -print -quit \
    | grep -q .; then
    echo "FAIL: checkpoint shard changed after checksum verification" >&2
    exit 2
fi
grep -qx 'repo=moonshotai/Kimi-K3' "$VERIFY_MARKER" || {
    echo "FAIL: verification marker names a different repository" >&2; exit 2;
}
REVISION=$(sed -n 's/^revision=//p' "$VERIFY_MARKER")
if ! printf '%s' "$REVISION" | grep -Eq '^[0-9a-fA-F]{40}$'; then
    echo "FAIL: verification marker has no immutable 40-hex revision" >&2
    exit 2
fi

read -r NL TB < <(python3 - "$TRUNK/trunk.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    j = json.load(f)
layers = j.get("layers") or []
ids = [x.get("layer", i) for i, x in enumerate(layers)]
if ids != list(range(93)):
    raise SystemExit("trunk layer ids are not exactly 0..92")
print(len(layers), sum(int(x["nbytes"]) for x in layers))
PY
)
[ "$NL" -eq 93 ] || { echo "FAIL: packed trunk has $NL layers, expected 93" >&2; exit 2; }
[ "$TB" -eq "$EXPECT_TRUNK_BYTES" ] || {
    echo "FAIL: packed trunk metadata has $TB bytes, expected $EXPECT_TRUNK_BYTES" >&2
    exit 2
}
[ "$(stat -c%s "$TRUNK/trunk.bin")" -eq "$EXPECT_TRUNK_BYTES" ] || {
    echo "FAIL: trunk.bin size does not match its complete metadata" >&2; exit 2;
}

mkdir -p "$OUT"
cp "$VERIFY_MARKER" "$OUT/checkpoint-verification.txt"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    GIT_COMMIT=$(git -C "$ROOT" rev-parse HEAD)
    git -C "$ROOT" status --porcelain=v1 >"$OUT/source-status.txt"
    git -C "$ROOT" diff --binary >"$OUT/source.patch"
    (cd "$ROOT" && git ls-files --cached --others --exclude-standard -z \
        | sort -z | xargs -0 sha256sum) >"$OUT/source-files.sha256"
else
    GIT_COMMIT=unavailable-source-bundle
    echo "source was deployed as a bundle without .git metadata" >"$OUT/source-status.txt"
    : >"$OUT/source.patch"
    (cd "$ROOT" && find . -type f \
        -not -path './.git/*' -not -path './build*/*' -not -path './bin*/*' \
        -not -path './cmake-build*/*' -print0 | sort -z | xargs -0 sha256sum) \
        >"$OUT/source-files.sha256"
fi
{
    echo "timestamp=$(date --iso-8601=seconds)"
    echo "git_commit=$GIT_COMMIT"
    echo "uname=$(uname -a)"
    echo "machine=$(uname -m)"
    echo "nproc=$(nproc)"
    echo "model_dir=$MODEL"
    echo "model_revision=$REVISION"
    echo "trunk_dir=$TRUNK"
    echo "model_mount=$(findmnt -T "$MODEL" -no SOURCE,FSTYPE,TARGET,OPTIONS)"
    echo "trunk_mount=$(findmnt -T "$TRUNK" -no SOURCE,FSTYPE,TARGET,OPTIONS)"
    echo "--- os-release ---"
    cat /etc/os-release
    if [ -r /etc/nv_tegra_release ]; then
        echo "--- Jetson Linux ---"
        cat /etc/nv_tegra_release
    fi
    echo "--- CPU ---"
    lscpu
    echo "--- memory ---"
    cat /proc/meminfo
    swapon --show --bytes 2>/dev/null || true
    zramctl 2>/dev/null || true
    echo "--- storage ---"
    df -hT "$MODEL" "$TRUNK" /
    lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,MODEL,ROTA
    echo "--- compiler ---"
    cc --version
    echo "--- binary ---"
    file "$ROOT/bin/k3"
    ldd "$ROOT/bin/k3" || true
} >"$OUT/environment.txt"

# Serial expert reads avoid seeking sixteen ways across a mechanical HDD. This changes
# only I/O scheduling; Top-K routing, official weights and arithmetic are untouched.
export K3_NOPREFETCH=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "starting full Kimi K3 proof-of-life"
echo "  model : $MODEL"
echo "  trunk : $TRUNK"
echo "  output: $OUT"
echo "  required first token: $EXPECT_TOKEN (' Paris'; exact gate)"

K3_CMD=(
    "$ROOT/bin/k3" "$MODEL"
    --trunk "$TRUNK" --preset ultra
    --tok "$MODEL" --prompt "The capital of France is"
    --gen 1
    --dump-logits "$OUT/logits.f32" --out "$OUT/result.json"
)
if [ -x /usr/bin/time ]; then
    /usr/bin/time -v -o "$OUT/time.txt" "${K3_CMD[@]}" \
        > >(tee "$OUT/run.log") 2> >(tee -a "$OUT/run.log" >&2)
else
    echo "/usr/bin/time unavailable; engine getrusage remains authoritative" >"$OUT/time.txt"
    "${K3_CMD[@]}" > >(tee "$OUT/run.log") 2> >(tee -a "$OUT/run.log" >&2)
fi

python3 - "$OUT/result.json" "$MAX_RSS_BYTES" "$OUT/acceptance.json" <<'PY'
import json, sys
path, limit, acceptance_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(path, encoding="utf-8") as f:
    r = json.load(f)
hard_checks = {
    "prompt tokenization": r.get("prompt_ids") == [1008, 10484, 318, 15383, 387],
    "official 93-layer walk": r.get("layers_requested") == 93 and r.get("layers_completed") == 93,
    "no expert drops": r.get("expert_drops") == 0,
    "exactly one generated token": len(r.get("generated_ids") or []) == 1,
    "exact token ID 17374": r.get("generated_ids") == [17374],
    "exact decoded token ' Paris'": r.get("generated_text") == " Paris",
    "ultra path active": r.get("ultra_low_memory") is True,
    "peak RSS <= 6 GiB": 0 < r.get("peak_rss_bytes", 0) <= limit,
    "trunk was read": r.get("trunk_bytes_read", 0) > 0,
    "experts were read": r.get("expert_bytes_read", 0) > 0,
    "embedding was streamed": r.get("embedding_bytes_read", 0) > 0,
    "lm_head was streamed": r.get("lm_head_bytes_read", 0) > 0,
}
ids = r.get("generated_ids") or []
text = r.get("generated_text") or ""
reference_match = ids == [17374] and text == " Paris"
semantic_reasonable = "paris" in text.casefold()
report = {
    "hard_checks": hard_checks,
    "hard_pass": all(hard_checks.values()),
    "reference_token_id": 17374,
    "reference_decoded": " Paris",
    "reference_match": reference_match,
    "generated_ids": ids,
    "generated_text": text,
    "semantic_reasonable": semantic_reasonable,
    "semantic_rule": "the single decoded token contains Paris, case-insensitive",
}
with open(acceptance_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")
for name, ok in hard_checks.items():
    print(("PASS" if ok else "FAIL") + ": " + name)
print(("MATCH" if reference_match else "MISMATCH") + ": exact token 17374 (' Paris')")
print("DIAGNOSTIC: semantic token " + repr(text) +
      (" contains Paris" if semantic_reasonable else " does not contain Paris"))
if not all(hard_checks.values()):
    raise SystemExit(1)
PY

echo "PROOF-OF-LIFE PASSED: exact token 17374 (' Paris'), 93/93 layers, no drops."
