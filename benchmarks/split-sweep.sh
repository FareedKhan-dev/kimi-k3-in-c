#!/usr/bin/env bash
# Hold total memory FIXED and sweep the trunk/cache split.
#
#   benchmarks/split-sweep.sh <model_dir> <trunk_dir> <out_dir> [total_gb] [reps]
#
# This is the experiment behind the single most useful tuning result in the project:
# at a fixed budget, memory given to the trunk is worth more than memory given to the
# expert cache. memory-ladder.sh varies the total and cannot separate "more memory" from
# "better-allocated memory"; this script varies only the allocation, so any difference it
# finds is attributable to the split alone.
#
# Defaults to 3 repetitions per split. docs/PERFORMANCE.md records a 33% run-to-run
# spread on identical configurations, which is large enough that a one-sample-per-split
# sweep cannot distinguish an allocation effect from dispersion.
#
# LINUX ONLY see the note in memory-ladder.sh.
set -u
MODEL="${1:?usage: split-sweep.sh <model_dir> <trunk_dir> <out_dir> [total_gb] [reps]}"
TRUNK="${2:?}"
OUT="${3:?}"
TOTAL="${4:-128}"
REPS="${5:-3}"
IDS=1008,10484,318,15383,387
GEN=8

command -v systemd-run >/dev/null 2>&1 || {
    echo "systemd-run not found; a genuine memory ceiling is required. See memory-ladder.sh."
    exit 1
}
systemd-run --scope --user -q true 2>/dev/null || {
    echo "systemd-run --user does not work here (no user session bus?)."
    exit 1
}
[ -x ./bin/k3 ] || { echo "./bin/k3 not built, run 'make -j' first"; exit 1; }
[ -d "$MODEL" ] || { echo "no such model dir: $MODEL"; exit 1; }
[ -d "$TRUNK" ] || { echo "no such trunk dir: $TRUNK, run scripts/pack-trunk.sh"; exit 1; }
command -v python3 >/dev/null || { echo "python3 required for the summary"; exit 1; }

mkdir -p "$OUT"
TSV="$OUT/split.tsv"
printf 'total_gb\ttrunk_gb\tcache_gb\trep\ts_per_tok\tpeak_rss_gb\tgb_read\tids\n' > "$TSV"

{
    echo "date        : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host        : $(uname -srm)"
    echo "cpu         : $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "cores       : $(nproc 2>/dev/null || echo '?')"
    echo "memtotal    : $(awk '/MemTotal/{printf "%.1f GB", $2/1048576}' /proc/meminfo 2>/dev/null)"
    echo "trunk_fs    : $(df -PT "$TRUNK" 2>/dev/null | awk 'NR==2{print $2, $1}')"
    echo "k3          : $(./bin/k3 --version 2>&1 | head -1)"
    echo "total_gb    : $TOTAL"
    echo "reps/split  : $REPS"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"
echo

# Fractions of the budget given to the trunk, cache-heavy first so the sweep runs against
# the hypothesis rather than with it. The engine needs a few GB outside both budgets, so
# the two never sum to the full total.
FRACS="0.10 0.25 0.60 0.86"

REF=""
DIVERGED=0
for f in $FRACS; do
    TR=$(python3 -c "print(round($TOTAL * $f, 1))")
    CA=$(python3 -c "print(round($TOTAL * (1 - $f) - 4, 1))")
    for r in $(seq 1 "$REPS"); do
        tag="tr${TR}_r${r}"
        echo "== ${TOTAL} GB total: trunk ${TR} / cache ${CA}, rep ${r}/${REPS} =="
        # A private fresh path cannot reuse a previous campaign's successful JSON.
        RESULT=$(mktemp "$OUT/$tag.json.XXXXXX") || exit 1
        systemd-run --scope --user -q \
            -p MemoryMax=${TOTAL}G -p MemorySwapMax=0 \
            ./bin/k3 "$MODEL" --ids "$IDS" --gen "$GEN" \
            --trunk "$TRUNK" --trunk-gb "$TR" --cache-gb "$CA" --incremental \
            --out "$RESULT" > "$OUT/$tag.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            # Unlike the ladder, an OOM here is a HARNESS defect, not a result: every
            # split in this sweep is sized to fit the same fixed total. Recording it as
            # a data point would hide a mis-sized split.
            echo "   *** FAILED with exit $rc at a split that should fit ***"
            tail -5 "$OUT/$tag.log" | sed 's/^/   | /'
            exit 1
        fi
        METRICS=$(python3 "$(dirname "$0")/read_result.py" "$RESULT" "$IDS" "$GEN") || {
            echo "   *** invalid or missing fresh result: $RESULT ***"
            exit 1
        }
        IFS=$'\t' read -r SPT RSS GBR IDSO <<< "$METRICS"
        mv -f "$RESULT" "$OUT/$tag.json" || exit 1

        if [ -z "$REF" ]; then REF="$IDSO"
        elif [ "$IDSO" != "$REF" ]; then
            echo "   *** OUTPUT DIFFERS across splits, this is a bug ***"
            DIVERGED=1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$TOTAL" "$TR" "$CA" "$r" "${SPT:--}" "${RSS:--}" "${GBR:--}" "$IDSO" >> "$TSV"
        echo "   ${SPT} s/token, peak RSS ${RSS} GB, ${GBR} GB read"
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
    try: by[float(r['trunk_gb'])].append(float(r['s_per_tok']))
    except (ValueError, KeyError): pass
if not by:
    print("no usable rows"); raise SystemExit(1)

print(f"{'trunk GB':>9}  {'mean':>7}  {'sd':>6}  {'spread':>7}  n")
means = {}
for k in sorted(by):
    v = by[k]
    m = statistics.mean(v)
    sd = statistics.stdev(v) if len(v) > 1 else 0.0
    sp = (max(v) - min(v)) / m * 100 if len(v) > 1 else 0.0
    means[k] = m
    print(f"{k:9.1f}  {m:7.2f}  {sd:6.2f}  {sp:6.1f}%  {len(v)}")

lo, hi = min(means), max(means)
factor = means[lo] / means[hi]
worst_spread = max(
    ((max(v) - min(v)) / statistics.mean(v) * 100) for v in by.values() if len(v) > 1
) if any(len(v) > 1 for v in by.values()) else None

print(f"\ntrunk {hi:.1f} GB is {factor:.2f}x faster than trunk {lo:.1f} GB "
      f"at the same total budget.")
ordered = [means[k] for k in sorted(means)]
print("monotonic in trunk size:", "yes" if ordered == sorted(ordered, reverse=True) else "NO")
if worst_spread is not None:
    print(f"worst within-split spread: {worst_spread:.1f}%")
    print("The effect is credible only if the factor above clearly exceeds this spread.")
else:
    print("Only one sample per split, no spread, so no conclusion. Re-run with reps > 1.")
PY

if [ "$DIVERGED" -ne 0 ]; then
    echo
    echo "FAIL: output was not identical across splits."
    exit 1
fi
echo
echo "Output identical across successful runs. Machine details: $OUT/machine.txt"
