#!/bin/bash
# Stage 3-diag: own-session learnability for one Stage-2 source.
#
#   sbatch own_session_eval.sh EID
#
# Evaluates the pretrained source on its OWN untouched test split with nothing
# re-initialised, plus three untrained references (init seeds 0, 1, 2). A
# diagnostic, not a gate. Audit asserts run first and refuse anything that is
# not a Stage-2 source -- including every official held-out session.

#SBATCH --job-name=own-sess
#SBATCH --output=../results_parity/own-sess-%j.out
#SBATCH --error=../results_parity/own-sess-%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

EID=${1:?usage: own_session_eval.sh EID}

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

OUT="${RESULTS_ROOT}/pilot/own_session/${EID}"

echo "=================================================="
echo "Stage 3-diag: own-session learnability"
echo "eid            : ${EID}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git describe   : $(git describe --tags --always 2>/dev/null || echo unknown)"
echo "slurm job      : ${SLURM_JOB_ID:-local}"
echo "slurm nodelist : ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; a diagnostic run must be attributable to a commit." >&2
    git status --porcelain >&2
    exit 2
fi
# 2c item 7: pin the exact commit this job was submitted against.
if [ -n "${EXPECT_SHA:-}" ]; then
    ACTUAL_SHA="$(git rev-parse HEAD)"
    if [ "$ACTUAL_SHA" != "$EXPECT_SHA" ]; then
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $ACTUAL_SHA." >&2
        exit 2
    fi
    echo "commit verified: $ACTUAL_SHA"
fi
if [ -e "$OUT" ]; then
    echo "ERROR: $OUT exists. Remove it or choose another root." >&2
    exit 2
fi
mkdir -p "$OUT" || exit 2

python script/write_capture_meta.py "$OUT" "own_session_${EID}" "$EID" 0 \
    --cmd "own_session_eval.sh $EID"

python script/own_session_eval.py "$EID"
rc=$?

echo
echo "own-session ${EID} finished, exit=${rc}"
[ $rc -eq 0 ] && find "$OUT" -type f \( -name '*.json' -o -name '*.npy' -o -name '*.npz' \) \
    | sed "s|^${OUT}/|  |" | sort
conda deactivate
exit $rc
