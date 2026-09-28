#!/bin/bash
# M4 V2: own-session evaluation of every member of one multi-session checkpoint.
#
#   sbatch ms_member_eval.sh LABEL CKPT EID [EID ...]
#
# Each member is evaluated by script/member_eval.py with the FULL saved model and
# vocabulary; only the evaluation data is restricted to that member's test split.
# One job, members in sequence. Writes to $RESULTS_ROOT/multisession/$LABEL/<eid>/.

#SBATCH --job-name=ms-member
#SBATCH --output=../results_parity/ms-member-%j.out
#SBATCH --error=../results_parity/ms-member-%j.err
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

LABEL=${1:?usage: ms_member_eval.sh LABEL CKPT EID [EID ...]}
CKPT=${2:?usage: ms_member_eval.sh LABEL CKPT EID [EID ...]}
shift 2
[ $# -ge 1 ] || { echo "ERROR: give at least one member EID" >&2; exit 2; }

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

OUT="${RESULTS_ROOT}/multisession/${LABEL}"
echo "M4 V2 member eval: ${LABEL}  ckpt ${CKPT}  members $*"
echo "git sha: $(git rev-parse HEAD)   slurm job: ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty" >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $(git rev-parse HEAD)" >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi
if [ -e "$OUT" ]; then
    echo "ERROR: $OUT exists. Choose another LABEL or remove it." >&2; exit 2
fi
mkdir -p "$OUT" || exit 2

rc_all=0
for EID in "$@"; do
    echo; echo "---------- member ${EID} ----------"
    python script/member_eval.py "$CKPT" "$EID" --out-root "$OUT"
    rc=$?
    echo "member ${EID} exit=${rc}"
    [ $rc -ne 0 ] && rc_all=$rc
done
echo; echo "M4 V2 member eval ${LABEL} finished, exit=${rc_all}"
conda deactivate
exit $rc_all
