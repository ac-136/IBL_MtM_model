#!/bin/bash

#SBATCH --job-name=batch-train-eval
#SBATCH --output=batch-train-eval-%j.out
#SBATCH --error=batch-train-eval-%j.err

#SBATCH --time=00:05:00

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

EID_LIST="${1:-test_eids.txt}"

if [ ! -f "$EID_LIST" ]; then
    echo "EID list not found: $EID_LIST"
    exit 1
fi

MODEL_NAME="NDT1"
MASK_MODE="temporal"
NUM_TRAIN_SESSIONS=34
MODE="train-eval"
MODEL_PATH="/work/hdd/beml/ac136/results_og_multi_session/train/num_session_10/model_best.pt"


while IFS= read -r line || [ -n "$line" ]
do
    TEST_EID="$line"

    if [ -z "$TEST_EID" ]; then
        continue
    fi

    echo "Submitting job for eid $line"

    sbatch finetune_eval_multi_session.sh \
        "$MODEL_NAME" \
        "$MASK_MODE" \
        "$NUM_TRAIN_SESSIONS" \
        "$TEST_EID" \
        "$MODE" \
        "$MODEL_PATH"

done < "$EID_LIST"
