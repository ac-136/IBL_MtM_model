#!/bin/bash
# Submit learnability sessions from a PINNED worktree, one named job each.
#
#   script/learnability_submit.sh [--only EID] [--dry-run]
#
# Must be run from a clean detached worktree; EXPECT_SHA is set to its HEAD so no
# later commit can reach a queued job. --time per session = cost estimate x 1.5
# plus 30 min for eval, rounded up to 10 min (over-requests kill backfill).
# Sessions whose own_session_summary.json already exists are skipped.
set -u
cd "$(dirname "$0")/.." || exit 2
ONLY=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --only) ONLY="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
[ -z "$(git status --porcelain)" ] || { echo "ERROR: tree is dirty" >&2; exit 2; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "HEAD" ] || {
    echo "ERROR: not a detached worktree; submit from the pinned campaign worktree" >&2; exit 2; }
SHA=$(git rev-parse HEAD)
LOG=/projects/beml/ac136/IBL_MtM_model/results_parity
EVAL_ROOT=/work/nvme/beml/ac136/IBL_MTM/results/learnability/own_session

python3 - "$ONLY" <<'PY' > /tmp/learn_submit_$$.tsv
import json, math, sys
only = sys.argv[1]
m = json.load(open("script/learnability_manifest.json"))
for s in m["sessions"]:
    if only and s["eid"] != only:
        continue
    h = s["est_pretrain_gpu_h"] * 1.5 + 0.5
    mins = max(30, int(math.ceil(h * 60 / 10.0)) * 10)
    print(f"{s['eid']}\t{mins // 60:02d}:{mins % 60:02d}:00")
PY
n=0
while IFS=$'\t' read -r EID T; do
    if [ -f "$EVAL_ROOT/$EID/own_session_summary.json" ]; then
        echo "skip $EID (summary exists)"; continue
    fi
    CMD=(sbatch --export=ALL,EXPECT_SHA=$SHA --time=$T -J "learn-${EID:0:8}"
         -o "$LOG/learn-%j.out" -e "$LOG/learn-%j.err" script/learnability_session.sh "$EID")
    if [ $DRY -eq 1 ]; then echo "${CMD[*]}"; else "${CMD[@]}"; fi
    n=$((n+1))
done < /tmp/learn_submit_$$.tsv
rm -f /tmp/learn_submit_$$.tsv
echo "$([ $DRY -eq 1 ] && echo 'would submit' || echo submitted) $n job(s) at $SHA"
