#!/bin/bash
# Stage-1b parity: multi-session SSL training, 5 epochs.
#
#   sbatch parity_train_multi.sh TREE RUN_NAME EID [EID ...]
#     TREE      src | ndt | src_rerun
#     RUN_NAME  e.g. ms_first2, ms_all5
#     EID ...   entries from parity_eids.txt, with or without the _aligned suffix
#
# Writes to  $RESULTS_ROOT/parity_$TREE/train/$RUN_NAME/
#
# eids pass through as given; session_data_dir resolves either directory layout.

#SBATCH --job-name=par-ms
#SBATCH --output=../results_parity/par-ms-%j.out
#SBATCH --error=../results_parity/par-ms-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

TREE=${1:?usage: parity_train_multi.sh TREE RUN_NAME EID [EID ...]}
RUN_NAME=${2:?usage: parity_train_multi.sh TREE RUN_NAME EID [EID ...]}
shift 2
EIDS=("$@")
KIND=multi

if [ ${#EIDS[@]} -eq 0 ]; then
    echo "need at least one eid" >&2; exit 2
fi

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
echo "run name    : $RUN_NAME"
echo "sessions    : ${EIDS[*]}"

ALIGNED=()
for e in "${EIDS[@]}"; do
    b="${e%_aligned}"
    if [ ! -d "$BASE_PATH/$DATA_TYPE/$b/data" ] && [ ! -d "$BASE_PATH/$DATA_TYPE/${b}_aligned/data" ]; then
        echo "Dataset not found: $BASE_PATH/$DATA_TYPE/$b(_aligned)/data" >&2
        exit 2
    fi
    ALIGNED+=("$b")
done

"${PY_CMD[@]}" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --num-sessions "${#ALIGNED[@]}" \
    --train-session-eid "${ALIGNED[@]}" \
    --model_name "$RUN_NAME" \
    --num-epochs "$NUM_EPOCHS"
rc=$?

echo "exit=$rc"
conda deactivate
exit $rc
