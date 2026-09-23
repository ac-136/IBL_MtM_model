#!/bin/bash
# Single-session learnability: pretrain one manifest session (unless reused),
# then evaluate it on its own test split.
#
#   sbatch [--time=...] learnability_session.sh EID
#
# Submitted by learnability_submit.sh, which sets --time per session from the
# cost model. Train and eval share one job so no afterok chain can strand work.
# Resubmitting is safe: a COMPLETE campaign pretrain is kept and only the eval
# runs; a PARTIAL one is refused rather than overwritten.

#SBATCH --job-name=learn
#SBATCH --output=../results_parity/learn-%j.out
#SBATCH --error=../results_parity/learn-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

EID=${1:?usage: learnability_session.sh EID}

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

MANIFEST=script/learnability_manifest.json
PRE_SUB=learnability/pretrain
PRE_ROOT="${RESULTS_ROOT}/${PRE_SUB}"
EVAL_ROOT="${RESULTS_ROOT}/learnability/own_session"
RUN_DIR="${PRE_ROOT}/src_${EID}/train/${EID}"
OUT="${EVAL_ROOT}/${EID}"

echo "=================================================="
echo "learnability session : ${EID}"
echo "git sha              : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "slurm job            : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; runs must be attributable to a commit." >&2
    git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $(git rev-parse HEAD)." >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi

REUSE=$(python -c "
import json,sys
m={s['eid']:s for s in json.load(open('$MANIFEST'))['sessions']}
if '$EID' not in m: sys.exit(4)
print(m['$EID'].get('reuse_pilot_pretrain') or '')")
[ $? -eq 0 ] || { echo "ERROR: $EID is not in $MANIFEST" >&2; exit 2; }

complete() {   # every artifact the eval and the report read
    local d="$1"
    for f in model_best.pt model_last.pt model_best.state_dict.pt model_last.state_dict.pt \
             run_summary.json loss/loss_history.npz; do
        [ -f "$d/$f" ] || return 1
    done
}

# ------------------------------------------------------------ 1. pretrain
if [ -n "$REUSE" ]; then
    echo "pretrain: REUSING pilot run $REUSE"
    complete "$REUSE" || { echo "ERROR: reused pilot run is incomplete" >&2; exit 2; }
elif complete "$RUN_DIR"; then
    echo "pretrain: already complete at $RUN_DIR (resubmission) -- not retraining"
elif [ -e "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR exists but is INCOMPLETE. Inspect and remove it; not overwriting." >&2
    exit 2
else
    echo "---------- [1/2] pretrain (300 epochs, recipe_mtm_frozen) ----------"
    PRETRAIN_SUBDIR="$PRE_SUB" bash script/pilot_pretrain.sh "$EID" 300
    rc=$?
    [ $rc -eq 0 ] || { echo "pretrain failed (exit=$rc); not evaluating" >&2; exit $rc; }
    complete "$RUN_DIR" || { echo "ERROR: pretrain reported success but artifacts are missing" >&2; exit 3; }
fi

# ------------------------------------------------------------ 2. own-session eval
echo
echo "---------- [2/2] own-session eval (best + last) ----------"
if [ -e "$OUT" ]; then
    echo "ERROR: $OUT exists. Remove it or choose another root." >&2; exit 2
fi
mkdir -p "$OUT" || exit 2
python script/write_capture_meta.py "$OUT" "learnability_${EID}" "$EID" 300 \
    --cmd "learnability_session.sh $EID"
python script/own_session_eval.py "$EID" --manifest "$MANIFEST" \
    --pretrain-root "$PRE_ROOT" --out-root "$EVAL_ROOT"
rc=$?
echo; echo "learnability ${EID} finished, exit=${rc}"
conda deactivate
exit $rc
