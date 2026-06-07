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

BASE_MODEL_DIR="/work/hdd/beml/ac136/training_og/train/num_session_1/model_NDT1/method_ssl/mask_temporal/stitch_True"
MODEL_LIST="ss_model_eids.txt"


MODEL_NAME="NDT1"
MASK_MODE="temporal"
NUM_TRAIN_SESSIONS=34
TEST_EID="recording_1"
MODE="train-eval"

while IFS= read -r line || [ -n "$line" ]
do
    MODEL_PATH="$BASE_MODEL_DIR/$line/model_best.pt"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "Skipping $line (no model_best.pt)"
        continue
    fi

    echo "Submitting job for model $line"

    sbatch finetune_eval_multi_session.sh \
        "$MODEL_NAME" \
        "$MASK_MODE" \
        "$NUM_TRAIN_SESSIONS" \
        "$TEST_EID" \
        "$MODE" \
        "$MODEL_PATH"

done < "$MODEL_LIST"