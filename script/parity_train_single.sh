#!/bin/bash
# Stage-1b parity: single-session SSL training, 5 epochs.
#
#   sbatch parity_train_single.sh TREE EID
#     TREE  src | ndt | src_rerun
#     EID   an entry from parity_eids.txt, with or without the _aligned suffix
#
# Writes to  $RESULTS_ROOT/parity_$TREE/ss_$EID/train/$EID/

#SBATCH --job-name=par-ss
#SBATCH --output=../results_parity/par-ss-%j.out
#SBATCH --error=../results_parity/par-ss-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

TREE=${1:?usage: parity_train_single.sh TREE EID}
EID=${2:?usage: parity_train_single.sh TREE EID}
KIND=single

# Slurm copies the submitted script to a spool directory, so "$0" is
# /var/spool/slurmd/jobNNN/slurm_script and dirname "$0" is useless here.
# SLURM_SUBMIT_DIR is where sbatch was invoked, which may be the repo root or
# script/, so search for the directory that actually holds the repo.
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
parity_tree_setup "$TREE" "$KIND"
parity_banner

BARE="${EID%_aligned}"
if [ ! -d "$BASE_PATH/$DATA_TYPE/$BARE/data" ] && [ ! -d "$BASE_PATH/$DATA_TYPE/${BARE}_aligned/data" ]; then
    echo "Dataset not found: $BASE_PATH/$DATA_TYPE/$BARE(_aligned)/data" >&2
    exit 2
fi

"${PY_CMD[@]}" \
    --eid "$EID" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "ss_${BARE}" \
    --num-epochs "$NUM_EPOCHS"
rc=$?

echo "exit=$rc"
conda deactivate
exit $rc
