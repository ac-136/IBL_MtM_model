#!/bin/bash
# Stage 2: one 300-epoch single-session pretrain, at the restored MtM recipe.
#
#   sbatch pilot_pretrain.sh EID [EPOCHS]
#
# Five of these run independently and in parallel, one GPU each. This is the
# first stage whose question is scientific rather than infrastructural: does the
# restored MtM recipe train stably across representative rs and bwm sessions?
#
# Sources are HELD-IN sessions. Development targets and the official held-out
# sessions are both untouched here.

#SBATCH --job-name=pilot-pre
#SBATCH --output=../results_parity/pilot-pre-%j.out
#SBATCH --error=../results_parity/pilot-pre-%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

EID=${1:?usage: pilot_pretrain.sh EID [EPOCHS]}
EPOCHS=${2:-300}
DATA_ROOT=/work/nvme/beml/ac136/LFP_FM/data/datasets

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

OUT_ROOT="${RESULTS_ROOT}/pilot/pretrain"
LOG_DIR="${OUT_ROOT}/src_${EID}/train/${EID}"

echo "=================================================="
echo "Stage 2: single-session pretrain at restored MtM defaults"
echo "eid            : ${EID}"
echo "epochs         : ${EPOCHS}"
echo "data root      : ${DATA_ROOT}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git describe   : $(git describe --tags --always 2>/dev/null || echo unknown)"
echo "slurm job      : ${SLURM_JOB_ID:-local}"
echo "slurm nodelist : ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; a pilot run must be attributable to a commit." >&2
    git status --porcelain >&2
    exit 2
fi
[ -d "$DATA_ROOT/$EID/data" ] || { echo "ERROR: no data at $DATA_ROOT/$EID/data" >&2; exit 2; }
if [ -e "$LOG_DIR" ]; then
    echo "ERROR: $LOG_DIR exists. Remove it or choose another root." >&2
    exit 2
fi

python -m ndt.cli.train_single \
    --eid "$EID" \
    --data-root "$DATA_ROOT" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "src_${EID}" \
    --num-epochs "$EPOCHS"
rc=$?

echo
echo "pretrain ${EID} finished, exit=${rc}"
[ $rc -eq 0 ] && ls -1 "$LOG_DIR" | sed 's/^/  /'
conda deactivate
exit $rc
