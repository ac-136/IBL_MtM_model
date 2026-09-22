#!/bin/bash
# Stage-1b parity: finetune (5 epochs) then evaluate, on one test session.
#
#   sbatch parity_finetune_eval.sh TREE MODEL_TAG MODEL_PATH TEST_EID
#     TREE        src | ndt | src_rerun
#     MODEL_TAG   short name for the pretrained source, e.g. ss_0cad7ea8 or ms_all5
#     MODEL_PATH  path to that model's model_best.pt
#     TEST_EID    an entry from data/test_re_eids.txt, with or without _aligned
#
# Writes to  $RESULTS_ROOT/parity_$TREE/ft_${MODEL_TAG}/{finetune,eval}/$TEST_EID/
#
# --use_dummy is deliberately omitted; see the note in parity_common.sh.

#SBATCH --job-name=par-fte
#SBATCH --output=../results_parity/par-fte-%j.out
#SBATCH --error=../results_parity/par-fte-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

TREE=${1:?usage: parity_finetune_eval.sh TREE MODEL_TAG MODEL_PATH TEST_EID}
MODEL_TAG=${2:?usage: parity_finetune_eval.sh TREE MODEL_TAG MODEL_PATH TEST_EID}
MODEL_PATH=${3:?usage: parity_finetune_eval.sh TREE MODEL_TAG MODEL_PATH TEST_EID}
TEST_EID=${4:?usage: parity_finetune_eval.sh TREE MODEL_TAG MODEL_PATH TEST_EID}
KIND=finetune

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
echo "model tag   : $MODEL_TAG"
echo "model path  : $MODEL_PATH"
echo "test eid    : $TEST_EID"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Pretrained checkpoint not found: $MODEL_PATH" >&2
    exit 2
fi
BARE="${TEST_EID%_aligned}"
if [ ! -d "$BASE_PATH/$DATA_TYPE/$BARE/data" ] && [ ! -d "$BASE_PATH/$DATA_TYPE/${BARE}_aligned/data" ]; then
    echo "Dataset not found: $BASE_PATH/$DATA_TYPE/$BARE(_aligned)/data" >&2
    exit 2
fi

"${PY_CMD[@]}" \
    --mask_ratio 0.3 \
    --mask_mode temporal \
    --model_name NDT1 \
    --train True \
    --eval True \
    --base_path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "ft_${MODEL_TAG}" \
    --num_train_sessions 1 \
    --test_eid "$TEST_EID" \
    --model_path "$MODEL_PATH" \
    --num-epochs "$NUM_EPOCHS"
rc=$?

echo "exit=$rc"
conda deactivate
exit $rc
