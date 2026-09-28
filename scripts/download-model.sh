#!/usr/bin/env bash
# download-model.sh, fetch the Kimi K3 checkpoint and verify it byte-exactly.
#
#   scripts/download-model.sh <dest_dir> [--layers N]
#
# The checkpoint is 1.56 TB across 96 safetensors shards. A partial or corrupt download
# does not fail loudly, it produces wrong tokens, so the byte total is checked against
# the published figure before anything else is allowed to use it.
#
# With --layers N, only the shards holding layers 0..N-1 (plus the embedding table and
# the output head, which every run needs) are fetched, as resolved from the Hub's own
# tensor-to-shard index rather than guessed from filenames. The result runs ONLY with
# `k3 ... --layers M` for M <= N; it is a test instrument for exercising the pipeline,
# measuring per-layer cost and checking the I/O path, not the model. The engine prints
# that plainly on every such run, and so does this script.
#
# The repository is public, so no token is needed. If you have one, the hf CLI picks it
# up from $HF_TOKEN or ~/.cache/huggingface/token on its own; this script never reads,
# echoes or forwards it.

set -euo pipefail

# `stat` and `find -printf` are spelled differently on BSD/macOS than on GNU/Linux.
# Resolve the stat flavour once here rather than shelling out per file.
case "$(uname -s)" in
    Darwin) STAT_FMT='-f%z' ;;
    *)      STAT_FMT='-c%s' ;;
esac
filesize() { stat $STAT_FMT "$1"; }

DEST="${1:?usage: download-model.sh <dest_dir> [--layers N]}"
LAYERS=""
if [ "${2:-}" != "" ]; then
    [ "${2:-}" = "--layers" ] || {
        echo "usage: download-model.sh <dest_dir> [--layers N]" >&2
        exit 1
    }
    LAYERS="${3:?usage: download-model.sh <dest_dir> [--layers N]}"
    [ "${4:-}" = "" ] || {
        echo "usage: download-model.sh <dest_dir> [--layers N]" >&2
        exit 1
    }
    # atoi-style parsing turned "1junk" into 1 in the CLI once; refuse that shape
    # here rather than downloading a prefix nobody asked for.
    case "$LAYERS" in
        ''|*[!0-9]*|0*)
            echo "download-model.sh: --layers must be a positive integer, got '$LAYERS'." >&2
            exit 1 ;;
    esac
fi
REPO="moonshotai/Kimi-K3"

# Published totals for the released checkpoint. Verified, not assumed.
EXPECT_SHARDS=96
EXPECT_BYTES=1560936091448

# The downloader is the `hf` CLI from huggingface_hub 1.x. The older entry points are
# gone: `python3 -m huggingface_hub.commands.huggingface_cli` was removed when the CLI
# was renamed, and the `[cli]` extra no longer exists.
#
# We deliberately do NOT attempt a pip install here. On the platform this project
# targets, a system pip install cannot succeed: Ubuntu 24.04 ships no pip module at
# all, and once pip is present PEP 668 marks the interpreter externally-managed and
# refuses. Guessing wrong in an unattended script that is about to move 1.56 TB is
# worse than stopping with an instruction.
if ! command -v hf >/dev/null 2>&1; then
    cat >&2 <<'EOF'
the `hf` CLI is required and was not found on PATH.

Install it one of these ways, then re-run:

  pipx install huggingface_hub            # recommended; what PEP 668 points you to
  uv tool install huggingface_hub
  python3 -m venv ~/.venvs/hf && ~/.venvs/hf/bin/pip install huggingface_hub
    then add ~/.venvs/hf/bin to PATH

On Debian/Ubuntu, `pipx` is `sudo apt install pipx`.
EOF
    exit 1
fi

# An older installation can provide `hf` without `hf cache verify`, which is what turns
# the size check below into a real integrity check.
hf cache verify --help >/dev/null 2>&1 || {
    echo "this hf CLI has no 'cache verify'; upgrade huggingface_hub and re-run." >&2
    exit 1
}

# Resolve the branch to an immutable commit ONCE, and use it for both the download and
# the verification. Otherwise `main` can move between the two steps and the checksums are
# compared against a different snapshot than the one on disk.
#
# Resolved through the hf CLI rather than `python3 -c "import huggingface_hub"`: the
# recommended installs above put the library in an isolated environment and expose only
# the `hf` executable, so importing it from the system interpreter fails on exactly the
# setups this script just told the user to create.
#
# The `hf models info` JSON is parsed with the stdlib `json` module, not a text scan --
# huggingface_hub 1.28 prints it as one compact line rather than the pretty-printed,
# one-key-per-line form an earlier `tr | awk '/^sha:/'` scan assumed, and that scan
# silently found nothing against compact output. `json` needs no import of
# huggingface_hub itself, so it does not hit the isolated-environment problem above.
# NOTE: json.load reads its input to EOF, so `hf` still runs to completion rather than
# taking SIGPIPE from an early-exiting consumer under `set -o pipefail`.
REVISION="${K3_REVISION:-}"
if [ -z "$REVISION" ]; then
    REVISION="$(hf models info "$REPO" 2>/dev/null \
                | python3 -c 'import json, sys
try:
    print(json.load(sys.stdin).get("sha", ""))
except Exception:
    print("")')"
fi
case "$REVISION" in
    ????????????????????????????????????????) ;;   # 40 hex characters
    *) echo "could not resolve a commit for $REPO (got '${REVISION:-empty}')." >&2
       echo "  Set K3_REVISION to a commit sha to skip this lookup." >&2
       exit 1 ;;
esac

mkdir -p "$DEST"

# Partial mode: fetch only the shards a --layers N run binds. The shard set comes
# from tools/layers_to_shards.py over the Hub's own tensor index, never from a
# filename guess, and the per-shard sizes below are checked against the same
# published figures as the full download. The full byte total and the Hub checksum
# verification are defined over all 96 shards, so they cannot pass here and are
# skipped loudly rather than weakened into something that looks like proof.
if [ -n "$LAYERS" ]; then
    MAPPER="$(dirname "$0")/../tools/layers_to_shards.py"
    SIZES="$(dirname "$0")/shard_sizes.txt"
    [ -f "$SIZES" ] || {
        echo "FAIL: $SIZES is missing; the subset cannot be sized or verified." >&2
        exit 1
    }
    echo "resolving layer prefix --layers $LAYERS to shards…"
    echo "  (fetching the 59 MB tensor index first; the shards follow)"
    HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
    hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" --max-workers 16 \
        --include "model.safetensors.index.json" \
        --include "config.json" \
        --include "tokenizer_config.json" \
        --include "tiktoken.model"
    [ -f "$DEST/config.json" ] || {
        echo "FAIL: config.json did not download; without it the engine refuses to start." >&2
        exit 1
    }
    [ -f "$DEST/tiktoken.model" ] || {
        echo "WARNING: tiktoken.model did not download; --ids runs work without it," >&2
        echo "  but --prompt/--prompt-file need --tok pointing at a tokenizer." >&2
    }
    SHARD_LIST="$DEST/.k3_partial_shards.txt"
    python3 "$MAPPER" "$DEST/model.safetensors.index.json" --layers "$LAYERS" >"$SHARD_LIST" || {
        echo "FAIL: could not resolve --layers $LAYERS to shards; nothing fetched." >&2
        exit 1
    }
    # Size the subset BEFORE moving gigabytes: a --layers prefix still pulls whole
    # expert pools per layer (each MoE layer holds 896 routed experts), so N=8 is
    # ~125 GB, not megabytes. Failing here reports a number instead of filling the disk.
    NEED=0
    while read -r name; do
        [ -n "$name" ] || continue
        want=$(awk -v n="$name" '$1 == n {print $2}' "$SIZES")
        [ -n "$want" ] || {
            echo "FAIL: $name is not in the published shard sizes; refusing to guess." >&2
            exit 1
        }
        NEED=$((NEED + want))
    done <"$SHARD_LIST"
    NSHARD=$(wc -l <"$SHARD_LIST" | tr -d ' ')
    AVAIL=$(df -P -k "$DEST" | awk 'NR==2 {print $4 * 1024}')
    if [ "$AVAIL" -lt "$NEED" ]; then
        printf 'FAIL: %s has %s bytes free, --layers %s needs %s across %s shard(s).\n' \
            "$DEST" "$AVAIL" "$LAYERS" "$NEED" "$NSHARD" >&2
        exit 1
    fi
    echo "downloading $NSHARD shard(s), $NEED bytes, for --layers $LAYERS"
    INC=()
    while read -r name; do
        [ -n "$name" ] || continue
        INC+=(--include "$name")
    done <"$SHARD_LIST"
    # Every element is an exact shard filename from the index (no spaces, no
    # globs), so the quoted expansion passes one --include per shard.
    HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
    hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" --max-workers 16 \
        "${INC[@]}"
    echo
    echo "verifying…"
    bad=0
    while read -r name; do
        [ -n "$name" ] || continue
        want=$(awk -v n="$name" '$1 == n {print $2}' "$SIZES")
        got=$(filesize "$DEST/$name" 2>/dev/null || echo 0)
        if [ "$got" != "$want" ]; then
            printf '  BAD  %s: %s bytes, expected %s\n' "$name" "$got" "$want"
            bad=$((bad + 1))
        fi
    done <"$SHARD_LIST"
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $bad shard(s) do not match their published sizes."
        echo "      Delete just those files and re-run; the download resumes."
        exit 1
    fi
    printf '  shards : all %s match their published sizes individually\n' "$NSHARD"
    echo
    echo "PARTIAL CHECKPOINT: layers 0..$((LAYERS - 1)) plus embed/head, $NSHARD of $EXPECT_SHARDS shards."
    echo "  This is NOT the full model. Run only as:"
    echo "      ./bin/k3 $DEST --layers $LAYERS --ids ...   (or any M <= $LAYERS)"
    echo "  The engine prints the same warning on every such run. Full runs need the"
    echo "  whole 1.56 TB: re-run without --layers."
    exit 0
fi

# Free-space preflight. Without this the transfer runs until the filesystem fills, which
# takes the machine's logging and package manager with it, and the byte-total check below
# then reports a mismatch -- a true statement that misdiagnoses the actual failure.
AVAIL=$(df -P -k "$DEST" | awk 'NR==2 {print $4 * 1024}')
if [ "$AVAIL" -lt "$EXPECT_BYTES" ]; then
    printf 'FAIL: %s has %s bytes free, the checkpoint needs %s.\n' \
        "$DEST" "$AVAIL" "$EXPECT_BYTES" >&2
    printf '      Short by %s bytes (%.2f TB). Point <dest_dir> at a larger filesystem.\n' \
        "$((EXPECT_BYTES - AVAIL))" \
        "$(awk -v d="$((EXPECT_BYTES - AVAIL))" 'BEGIN{print d/1e12}')" >&2
    printf '      Packing the trunk afterwards needs a further ~109 GB.\n' >&2
    exit 1
fi

echo "downloading $REPO@$REVISION -> $DEST"
echo "  1.56 TB across $EXPECT_SHARDS shards; expect ~30 min at 1 GB/s"
echo

# Xet is the transfer backend in huggingface_hub 1.x. HF_HUB_ENABLE_HF_TRANSFER, which
# this script used to set, is ignored there and was a hard error on 0.x whenever the
# hf_transfer package was absent -- which it always was, since nothing installed it.
#
# A token is not needed: the repository is public. If one is present in the environment
# or in a saved login the CLI uses it for higher rate limits; this script never touches it.
HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" --max-workers 16

echo
echo "verifying…"
# find, not `ls | wc -l`: under `set -euo pipefail` a glob that matches nothing makes
# ls exit non-zero and the script dies HERE, so the "download is incomplete" message
# below -- the entire point of this verification block -- would never be reached.
#
# Summed with a loop rather than `find -printf '%s\n'`: -printf is a GNU find
# extension that BSD/macOS find does not support. -print0 is POSIX-portable.
N=$(find "$DEST" -maxdepth 1 -name '*.safetensors' | wc -l)
B=0
while IFS= read -r -d '' f; do
    B=$((B + $(filesize "$f")))
done < <(find "$DEST" -maxdepth 1 -name '*.safetensors' -print0)

printf '  shards : %s (expect %s)\n' "$N" "$EXPECT_SHARDS"
printf '  bytes  : %s (expect %s)\n' "$B" "$EXPECT_BYTES"

if [ "$N" -ne "$EXPECT_SHARDS" ]; then
    echo "FAIL: wrong shard count, the download is incomplete."
    exit 1
fi
if [ "$B" -ne "$EXPECT_BYTES" ]; then
    echo "FAIL: byte total mismatch (delta $((B - EXPECT_BYTES)))."
    echo "      A partial checkpoint yields wrong output silently. Re-run to resume."
    exit 1
fi

# Per-shard sizes. The total above already proves the download is complete; this proves
# WHICH shard is wrong when it is not, turning "re-download 1.56 TB" into "re-download
# 17 GB". It also catches the one case a total cannot: two shards wrong in opposite
# directions by the same amount.
SIZES="$(dirname "$0")/shard_sizes.txt"
if [ -f "$SIZES" ]; then
    bad=0
    while read -r name want; do
        [ -n "$name" ] || continue
        got=$(filesize "$DEST/$name" 2>/dev/null || echo 0)
        if [ "$got" != "$want" ]; then
            printf '  BAD  %s: %s bytes, expected %s\n' "$name" "$got" "$want"
            bad=$((bad + 1))
        fi
    done < "$SIZES"
    if [ "$bad" -ne 0 ]; then
        echo "FAIL: $bad shard(s) do not match their published sizes."
        echo "      Delete just those files and re-run; the download resumes."
        exit 1
    fi
    printf '  shards : all %s match their published sizes individually\n' "$EXPECT_SHARDS"
fi

# Sizes prove the download is COMPLETE. They cannot prove it is the RIGHT bytes: a
# substituted shard of identical length passes everything above. SECURITY.md names that
# gap; this closes it by comparing against the hashes the Hub published for the exact
# commit resolved earlier. It re-reads all 1.56 TB, so it is not free -- on a machine
# without SHA-NI it can take longer than the download did.
echo
echo "verifying checksums against Hub metadata for ${REVISION}…"
echo "  (re-reads the full 1.56 TB; set K3_SKIP_CHECKSUM=1 to skip)"
if [ "${K3_SKIP_CHECKSUM:-0}" = "1" ]; then
    echo "  SKIPPED by K3_SKIP_CHECKSUM=1; sizes were still checked above."
else
    hf cache verify "$REPO" --revision "$REVISION" --local-dir "$DEST" \
        --fail-on-missing-files
    echo "  RESULT : checksum-verified snapshot at $REVISION"
    # The proof runner must be able to prove which immutable Hub snapshot these bytes
    # came from without re-hashing 1.56 TB immediately before every experiment. Write a
    # marker only after the Hub checksum verification succeeds; size-only or explicitly
    # skipped verification never creates it.
    printf 'repo=%s\nrevision=%s\nverified_at=%s\n' \
        "$REPO" "$REVISION" "$(date --iso-8601=seconds)" \
        >"$DEST/.k3_hf_verified"
fi
echo
echo "next: scripts/pack-trunk.sh $DEST <trunk_dir>"
echo "      packing the trunk is what lets the engine stream it, which is what makes"
echo "      the memory budget a dial instead of a floor."
