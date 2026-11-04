#!/usr/bin/env bash
# Sweep the memory budget and check that output is identical at every rung.
#
#   benchmarks/memory-ladder.sh <model_dir> <trunk_dir> <out_dir> [reps]
#
# Defaults to 3 repetitions per rung, because the measured run-to-run spread on an
# identical configuration is 33%, a single sample per rung cannot distinguish an effect
# from noise. See docs/BENCHMARKING.md.
#
# LINUX ONLY. Imposing a memory ceiling needs cgroups via systemd-run. There is no
# portable equivalent, and without a real ceiling every rung simply uses as much memory
# as it likes, so the ladder measures nothing while still producing a full table.
set -u
MODEL="${1:?usage: memory-ladder.sh <model_dir> <trunk_dir> <out_dir> [reps]}"
TRUNK="${2:?}"
OUT="${3:?}"
REPS="${4:-3}"
IDS=1008,10484,318,15383,387
GEN=8

# Preflight. Each of these, left unchecked, yields a COMPLETE and plausible-looking
# ladder composed entirely of failures, because the runner below cannot otherwise tell
# "killed for exceeding its memory ceiling" apart from "the command does not exist".
command -v systemd-run >/dev/null 2>&1 || {
    echo "systemd-run not found. This harness needs it to impose a genuine memory ceiling;"
    echo "without one every rung would run unconstrained and the ladder would be"
    echo "meaningless rather than merely wrong. Linux with systemd is required."
    exit 1
}
systemd-run --scope --user -q true 2>/dev/null || {
    echo "systemd-run --user does not work here (no user session bus?)."
    echo "Try: loginctl enable-linger \$USER, or run under a full login session."
    exit 1
}
[ -x ./bin/k3 ] || { echo "./bin/k3 not built, run 'make -j' first"; exit 1; }
[ -d "$MODEL" ] || { echo "no such model dir: $MODEL"; exit 1; }
[ -d "$TRUNK" ] || { echo "no such trunk dir: $TRUNK, run scripts/pack-trunk.sh"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for the summary"; exit 1; }

mkdir -p "$OUT"
TSV="$OUT/ladder.tsv"
printf 'total_gb\ttrunk_gb\tcache_gb\trep\ts_per_tok\tpeak_rss_gb\tgb_read\tids\n' > "$TSV"

# Machine provenance. A ladder without it cannot be compared against another machine's,
# and these are the first numbers anyone asks for when two ladders disagree.
{
    echo "date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host        : $(uname -srm)"
    echo "cpu         : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "cores       : $(nproc 2>/dev/null || echo '?')"
    echo "memtotal    : $(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo 2>/dev/null)"
    echo "swaptotal   : $(awk '/SwapTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo 2>/dev/null)"
    echo "trunk_fs    : $(df -PT "$TRUNK" 2>/dev/null | awk 'NR==2{print $2, $1}')"
    echo "model_fs    : $(df -PT "$MODEL" 2>/dev/null | awk 'NR==2{print $2, $1}')"
    echo "k3          : $(./bin/k3 --version 2>&1 | head -1)"
    echo "reps/rung   : $REPS"
    echo "ids         : $IDS"
    echo "gen         : $GEN"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"
echo

# Budget -> split. The trunk is filled first: it is re-read in full every token, while
# only ~25.8 GB of experts are touched. docs/TUNING.md has the measurements.
rungs="8:3:1 16:7:4 32:16:10 64:35:24 96:60:30 128:110:13"

REF=""
REF_RUNG=""
DIVERGED=0
for rung in $rungs; do
    TOT=${rung%%:*}; rest=${rung#*:}; TR=${rest%%:*}; CA=${rest#*:}
    for r in $(seq 1 "$REPS"); do
        tag="${TOT}gb_r${r}"
        echo "== ${TOT} GB (trunk ${TR} / cache ${CA}) rep ${r}/${REPS} =="

        # MemorySwapMax=0 alongside MemoryMax. Without it, a rung that exceeds its
        # ceiling swaps instead of dying: the run "passes" and its s/token measures swap
        # bandwidth rather than the configuration under test. That is a wrong number
        # wearing the shape of a right one, which is worse than a missing row.
        # A private fresh path cannot reuse a previous campaign's successful JSON.
        RESULT=$(mktemp "$OUT/$tag.json.XXXXXX") || exit 1
        systemd-run --scope --user -q \
            -p MemoryMax=${TOT}G -p MemorySwapMax=0 \
            ./bin/k3 "$MODEL" --ids "$IDS" --gen "$GEN" \
            --trunk "$TRUNK" --trunk-gb "$TR" --cache-gb "$CA" --incremental \
            --out "$RESULT" > "$OUT/$tag.log" 2>&1
        rc=$?

        if [ $rc -ne 0 ]; then
            # Distinguish "did not fit" from "broke". Only SIGKILL (137), which is how
            # the kernel terminates a cgroup over MemoryMax, is a legitimate rung result.
            # Recording any other failure as OOM fabricates a data point.
            if [ $rc -eq 137 ] || grep -qi 'out of memory\|oom-kill' "$OUT/$tag.log"; then
                printf '%s\t%s\t%s\t%s\tOOM\t-\t-\t-\n' "$TOT" "$TR" "$CA" "$r" >> "$TSV"
                echo "   did not fit, that is a result, not an error"
                continue
            fi
            echo "   *** FAILED with exit $rc, not an OOM, and not a data point ***"
            tail -5 "$OUT/$tag.log" | sed 's/^/   | /'
            exit 1
        fi

        METRICS=$(python3 "$(dirname "$0")/read_result.py" "$RESULT" "$IDS" "$GEN") || {
            echo "   *** invalid or missing fresh result: $RESULT ***"
            exit 1
        }
        IFS=$'\t' read -r SPT RSS GBR IDSO <<< "$METRICS"
        mv -f "$RESULT" "$OUT/$tag.json" || exit 1

        if [ -z "$REF" ]; then
            REF="$IDSO"; REF_RUNG="${TOT} GB rep $r"
        elif [ "$IDSO" != "$REF" ]; then
            echo "   *** OUTPUT DIFFERS from $REF_RUNG, this is a bug ***"
            echo "       expected: $REF"
            echo "       got     : $IDSO"
            DIVERGED=1
        fi

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$TOT" "$TR" "$CA" "$r" "${SPT:--}" "${RSS:--}" "${GBR:--}" "$IDSO" >> "$TSV"
        echo "   ${SPT} s/token, peak RSS ${RSS} GB"
    done
done

echo
column -t "$TSV"
echo
python3 - "$TSV" <<'PY'
import sys, csv, statistics, collections
rows = list(csv.DictReader(open(sys.argv[1]), delimiter='\t'))
by = collections.defaultdict(list)
for r in rows:
    try: by[r['total_gb']].append(float(r['s_per_tok']))
    except (ValueError, KeyError): pass
print(f"{'budget':>8}  {'mean':>7}  {'sd':>6}  {'spread':>7}  n")
for k, v in sorted(by.items(), key=lambda kv: int(kv[0])):
    if not v: continue
    m = statistics.mean(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    sp = (max(v) - min(v)) / m * 100 if len(v) > 1 else 0.0
    print(f"{k+' GB':>8}  {m:7.2f}  {sd:6.2f}  {sp:6.1f}%  {len(v)}")
    if len(v) == 1:
        print(f"{'':>8}  ^ one sample: no spread, so no conclusion from this rung")
print("\nA difference between budgets that is smaller than the spread within a budget"
      "\nis not an effect.")
PY

if [ "$DIVERGED" -ne 0 ]; then
    echo
    echo "FAIL: output was not identical across memory budgets."
    exit 1
fi
echo
echo "Output identical at every rung. Machine details: $OUT/machine.txt"
