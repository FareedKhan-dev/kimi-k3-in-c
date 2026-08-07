#!/usr/bin/env bash
# k3-doctor, check whether this machine can run Kimi K3, and how fast.
#
# Answers three questions before you spend an hour finding out the hard way:
#   1. Is the toolchain present and does the engine build?
#   2. How much memory is there, and which preset does that imply?
#   3. How fast is the storage the weights will stream from?
#
# Exits non-zero if the machine cannot run the model at all.
#
# LINUX ONLY, like every script under scripts/ and benchmarks/. This one reads
# /proc/meminfo and uses GNU `stat` and `df` options; the engine itself additionally
# needs O_DIRECT and getrusage. The tokenizer and config reader are portable C99 and
# build anywhere, but the streaming engine does not.

set -u

case "$(uname -s)" in
    Linux) ;;
    *)  echo "k3-doctor: this script and the streaming engine are Linux-only."
        echo "  (/proc/meminfo, O_DIRECT, GNU stat/df). Detected: $(uname -s)"
        exit 1 ;;
esac

RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
[ -t 1 ] || { RED=""; GRN=""; YLW=""; DIM=""; RST=""; }

ok()   { printf '  %sok%s    %s\n'   "$GRN" "$RST" "$*"; }
warn() { printf '  %swarn%s  %s\n'   "$YLW" "$RST" "$*"; }
bad()  { printf '  %sFAIL%s  %s\n'   "$RED" "$RST" "$*"; FAILED=1; }
hdr()  { printf '\n%s\n' "$*"; }

FAILED=0
MODEL_DIR="${1:-}"

printf '%s\n' "Kimi K3, environment check"

# ------------------------------------------------------------------ toolchain --
hdr "toolchain"
if command -v cc >/dev/null 2>&1 || command -v gcc >/dev/null 2>&1; then
    CCBIN=$(command -v cc || command -v gcc)
    ok "C compiler: $($CCBIN --version 2>&1 | head -1)"
else
    bad "no C compiler found (install build-essential or equivalent)"
fi
command -v make >/dev/null 2>&1 && ok "make: $(make --version | head -1)" || bad "make not found"
if command -v python3 >/dev/null 2>&1; then
    ok "python3: $(python3 --version 2>&1)"
else
    warn "python3 not found, the trunk packer and reference tools need it"
fi

# ------------------------------------------------------------------------ cpu --
hdr "cpu"
NCPU=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
ok "cores: $NCPU"
if grep -qm1 avx2 /proc/cpuinfo 2>/dev/null; then
    ok "AVX2: present"
else
    warn "AVX2 not detected, the engine will run but the expert matmuls lose their fast path"
fi
grep -qm1 avx512f /proc/cpuinfo 2>/dev/null && ok "AVX-512: present" \
    || printf '  %sinfo  AVX-512 absent (not required)%s\n' "$DIM" "$RST"

# --------------------------------------------------------------------- memory --
hdr "memory"
MEM_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
MEM_GB=$(( MEM_KB / 1024 / 1024 ))
AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo 2>/dev/null || echo 0)
AVAIL_GB=$(( AVAIL_KB / 1024 / 1024 ))
ok "total: ${MEM_GB} GB, available: ${AVAIL_GB} GB"

# Boundaries follow the measured memory ladder (docs/PERFORMANCE.md). Expectations are v1.0.0
# anchors from docs/data/speed-2026-08.md on the reference box (124 cores, fast NVMe): treat
# them as order-of-magnitude, not a promise. Streaming presets (laptop/desktop/workstation)
# scale with your disk; the trunk becomes resident above ~128 GB, where decode is compute-bound
# and scales with core count instead.
if   [ "$AVAIL_GB" -ge 192 ]; then PRESET=server;      EXPECT="~6 s/token"
elif [ "$AVAIL_GB" -ge  96 ]; then PRESET=workstation; EXPECT="~6-20 s/token"
elif [ "$AVAIL_GB" -ge  32 ]; then PRESET=desktop;     EXPECT="~24 s/token"
elif [ "$AVAIL_GB" -ge  10 ]; then PRESET=laptop;      EXPECT="~27 s/token"
else PRESET=""; fi

if [ -n "$PRESET" ]; then
    ok "recommended preset: --preset $PRESET   (expect $EXPECT)"
else
    # A warning, not a failure. This threshold is about RUNNING the checkpoint; it says
    # nothing about building the engine or running the test suite, both of which need no
    # weights and pass comfortably here. `make test`'s own ceiling is the ~1.7 GB single
    # allocation in tests/unit/scale_test.c. Failing the whole check on this number told
    # people their machine was broken when the only thing they could not do was the one
    # thing that needs 1.56 TB of disk they also did not have.
    warn "under 10 GB available, below the floor for running the checkpoint (~8.2 GB peak RSS)"
    printf '  %s      the build and "make test" need no weights and work fine here%s\n' \
        "$DIM" "$RST"
fi

# -------------------------------------------------------------------- storage --
hdr "storage"
TARGET="${MODEL_DIR:-$PWD}"
if [ -d "$TARGET" ]; then
    AVAIL_DISK=$(df -BG "$TARGET" 2>/dev/null | awk 'NR==2{gsub("G","",$4); print $4}')
    ok "free space at $TARGET: ${AVAIL_DISK:-?} GB"
    # 1.56 TB checkpoint + ~109 GB packed trunk.
    if [ -n "${AVAIL_DISK:-}" ] && [ "$AVAIL_DISK" -lt 1700 ]; then
        warn "the full checkpoint needs ~1.56 TB plus ~109 GB for the packed trunk"
    fi

    printf '  %smeasuring sequential read (2 GB, this takes a moment)…%s\n' "$DIM" "$RST"
    TMPF="$TARGET/.k3_doctor_probe"
    if dd if=/dev/zero of="$TMPF" bs=1M count=2048 conv=fsync 2>/dev/null; then
        sync
        RATE=$(dd if="$TMPF" of=/dev/null bs=4M 2>&1 | awk '/copied/{print $(NF-1)" "$NF}')
        rm -f "$TMPF"
        ok "sequential read: ${RATE:-unknown}"
        printf '  %sthe engine streams ~135 GB per token; storage is usually the ceiling%s\n' "$DIM" "$RST"
    else
        warn "could not write a probe file to $TARGET"
        rm -f "$TMPF" 2>/dev/null
    fi
else
    warn "no model directory given; pass one as the first argument to check its storage"
fi

# ---------------------------------------------------------------- model files --
if [ -n "$MODEL_DIR" ]; then
    hdr "model"
    # find, not `ls | wc -l`: a glob matching nothing would abort the script under
    # `set -euo pipefail` instead of reporting zero shards.
    N=$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)
    if [ "$N" -eq 0 ]; then
        warn "no .safetensors shards in $MODEL_DIR, run scripts/download-model.sh"
    else
        ok "shards present: $N"
        [ "$N" -eq 96 ] || warn "expected 96 shards for the full checkpoint"
    fi
    for f in config.json tiktoken.model tokenizer_config.json; do
        [ -f "$MODEL_DIR/$f" ] && ok "$f" || warn "$f missing (needed for text in/out)"
    done
fi

hdr "result"
if [ "$FAILED" -eq 0 ]; then
    if [ -n "$PRESET" ]; then
        printf '  %sthis machine can run Kimi K3%s\n' "$GRN" "$RST"
    else
        printf '  %sthis machine can build and test the engine, but not run the checkpoint%s\n' \
            "$YLW" "$RST"
    fi

    # The command printed here is the one a user is most likely to copy, so every part
    # of it has to work as written:
    #   --tok    is REQUIRED by --prompt. Without it the engine exits 2 rather than
    #            guessing where the tokenizer lives.
    #   --trunk  is what makes the preset mean anything. Omit it and the trunk loads
    #            fully resident at ~110 GB, which is the opposite of the budget just
    #            recommended.
    # MODEL_DIR is substituted when one was given, so the line can be pasted verbatim.
    if [ -n "$PRESET" ]; then
        M="${MODEL_DIR:-<model_dir>}"
        printf '\n  next:\n'
        printf '    %smake -j%s\n' "$DIM" "$RST"
        printf '    %s./scripts/pack-trunk.sh %s ~/k3trunk%s\n' "$DIM" "$M" "$RST"
        printf '    %s./bin/k3 %s --trunk ~/k3trunk --preset %s \\%s\n' "$DIM" "$M" "$PRESET" "$RST"
        printf '    %s         --tok %s --prompt "Hello" --gen 8 --incremental%s\n' "$DIM" "$M" "$RST"
    else
        # No preset fits, but the weightless path always does, and it is the whole of
        # the README's Quick start. Printing nothing here was what made the old FAIL
        # read as "give up".
        printf '\n  next:\n'
        printf '    %smake -j%s\n' "$DIM" "$RST"
        printf '    %smake test%s\n' "$DIM" "$RST"
        printf '  %sboth need no checkpoint. Running the model needs ~10 GB of RAM and%s\n' \
            "$DIM" "$RST"
        printf '  %s~1.7 TB of disk; come back with those and re-run this check.%s\n' \
            "$DIM" "$RST"
    fi
    exit 0
else
    printf '  %sblocking problems above%s\n' "$RED" "$RST"
    exit 1
fi
