#!/bin/bash
# Learnability report over the manifest's 52 sessions. Reads the per-session
# own_session_summary.json files only; no model is loaded.
#
#   cd /projects/beml/ac136/ndt-learnability        # detached, pinned
#   EXPECT_SHA=$(git rev-parse HEAD) sbatch script/learnability_report.sh
#
# Writes ${RESULTS_ROOT}/learnability/report/report_<sha12>.txt and refuses to
# overwrite it, so a report is always attributable to one commit.

#SBATCH --job-name=learn-report
#SBATCH --output=../results_parity/learn-report-%j.out
#SBATCH --error=../results_parity/learn-report-%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

find_repo_root() {
    local c
    for c in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
             "$PWD" "$PWD/.." \
             "$(cd "$(dirname "$0")" 2>/dev/null && pwd)" \
             "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; do
        [ -n "$c" ] || continue
        if [ -d "$c/ndt" ] && [ -d "$c/src" ] && [ -d "$c/script" ]; then
            (cd "$c" && pwd); return 0
        fi
    done
    return 1
}
REPO_ROOT="$(find_repo_root)" || { echo "cannot locate the repo root" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2
source script/parity_common.sh

SHA=$(git rev-parse HEAD)
OUT_DIR="${RESULTS_ROOT}/learnability/report"
OUT="${OUT_DIR}/report_${SHA:0:12}.txt"

echo "=================================================="
echo "learnability report"
echo "git sha              : ${SHA}"
echo "repo root            : ${REPO_ROOT}"
echo "slurm job            : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "output               : ${OUT}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; the report must be attributable to a commit." >&2
    git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$SHA" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $SHA." >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi
if [ -e "$OUT" ]; then
    echo "ERROR: $OUT exists. Remove it or report from another commit." >&2; exit 2
fi
mkdir -p "$OUT_DIR" || exit 2

{
    echo "# learnability_report.py at ${SHA}"
    echo "# slurm job ${SLURM_JOB_ID:-local} on ${SLURM_JOB_NODELIST:-local}, $(date -Is)"
    echo
    python script/learnability_report.py
} > "$OUT" 2>&1
rc=$?
cat "$OUT"
echo
echo "report exit=${rc} (1 = some session has failed checks; see the header lines)"
exit $rc
