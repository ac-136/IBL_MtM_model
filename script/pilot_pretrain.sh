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

# Output subdirectory under RESULTS_ROOT. Defaults to the pilot location so every
# existing invocation is unchanged; the learnability campaign sets
# PRETRAIN_SUBDIR=learnability/pretrain.
OUT_ROOT="${RESULTS_ROOT}/${PRETRAIN_SUBDIR:-pilot/pretrain}"
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
# 2c item 7: Slurm runs against the LIVE working tree, so a commit landing while
# this job sat in the queue would silently change what it measures. A clean tree
# does not catch that -- the tree is clean at both commits. Pass EXPECT_SHA to
# pin the exact commit this job was submitted against.
if [ -n "${EXPECT_SHA:-}" ]; then
    ACTUAL_SHA="$(git rev-parse HEAD)"
    if [ "$ACTUAL_SHA" != "$EXPECT_SHA" ]; then
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $ACTUAL_SHA." >&2
        echo "       Something was committed while this job was queued." >&2
        exit 2
    fi
    echo "commit verified: $ACTUAL_SHA"
fi
[ -d "$DATA_ROOT/$EID/data" ] || { echo "ERROR: no data at $DATA_ROOT/$EID/data" >&2; exit 2; }
if [ -e "$LOG_DIR" ]; then
    echo "ERROR: $LOG_DIR exists. Remove it or choose another root." >&2
    exit 2
fi
mkdir -p "$LOG_DIR" || exit 2

# 2c item 1: the same provenance record every other run gets.
CMD="pilot_pretrain.sh $EID $EPOCHS"
python script/write_capture_meta.py "$LOG_DIR" "src_${EID}" "$EID" "$EPOCHS" \
    --cmd "$CMD" --config "$LOG_DIR/resolved_config.yaml"

# 2c item 3: 20 rather than the config's 1. At 300 epochs that was ~685 PNGs and
# ~140 MB per run. Overridden here, not in the config, so the MtM recipe file
# stays at its published values and the choice appears in the recorded args.
python -m ndt.cli.train_single \
    --eid "$EID" \
    --data-root "$DATA_ROOT" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "src_${EID}" \
    --save-plot-every-n-epochs "${PLOT_EVERY:-20}" \
    --num-epochs "$EPOCHS"
rc=$?

echo
echo "pretrain ${EID} finished, exit=${rc}"
[ $rc -eq 0 ] && ls -1 "$LOG_DIR" | sed 's/^/  /'
conda deactivate
exit $rc
