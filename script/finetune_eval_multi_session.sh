#!/bin/bash

# Usage:
#   sbatch finetune_eval_multi_session.sh MODEL MASK NUM_SESSIONS EID MODE \
#       MODEL_PATH [OUTPUT_MODEL_NAME] [DATA_ROOT]

#SBATCH --job-name=single-session-finetune-eval
#SBATCH --output=single-session-finetune-eval-%j.out
#SBATCH --error=single-session-finetune-eval-%j.err

#SBATCH --time=08:00:00

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

START_TIME=$(date +%s)

. ~/.bashrc
conda activate mtm

# Check python path
echo "Conda prefix: $CONDA_PREFIX"
which python

MODEL_NAME=${1}
MASK_MODE=${2}
NUM_TRAIN_SESSIONS=${3}
TEST_EID=${4}
MODE=${5}
MODEL_PATH=${6}
OUTPUT_MODEL_NAME=${7:-mtm-baseline-34-sessions_test}
DATA_ROOT=${8:-}

TRAIN=False
EVAL=FALSE

if [ $MASK_MODE == "all" ]; then
    PROMPTING=True
else
    PROMPTING=False
fi

# if train in MODE
if [[ $MODE == *"train"* ]]; then
    # echo "Training"
    TRAIN=True
fi

if [[ $MODE == *"eval"* ]]; then
    # echo "Evaluating"
    EVAL=True
fi

BASE_PATH="/work/nvme/beml/ac136/IBL_MTM/data"
DATA_TYPE="processed_mtm"
RESULTS_PATH="/work/nvme/beml/ac136/IBL_MTM/results/mtm"

DATA_ROOT_ARGS=()
if [[ -n "$DATA_ROOT" ]]; then
    if [[ ! -d "$DATA_ROOT/$TEST_EID/data" ]]; then
        echo "Dataset directory not found: $DATA_ROOT/$TEST_EID/data" >&2
        exit 2
    fi
    DATA_ROOT_ARGS=(--data-root "$DATA_ROOT")
fi

echo "Mask mode: $MASK_MODE"
echo "Model name: $MODEL_NAME"
echo "Prompting: $PROMPTING"
echo "Train: $TRAIN"
echo "Eval: $EVAL"
echo "Base path: $BASE_PATH"
echo "Data type: $DATA_TYPE"
echo "Data root: ${DATA_ROOT:-$BASE_PATH/$DATA_TYPE}"
echo "Results path: $RESULTS_PATH"
echo "Output model name: $OUTPUT_MODEL_NAME"
echo "Num train sessions: $NUM_TRAIN_SESSIONS"
echo "Test eid: $TEST_EID"
echo "Single session model name: $MODEL_PATH"

cd ../

#  --base_path $SCRATCH/IBL_foundation_model \
python src/finetune_eval_multi_session_just_spikes.py --mask_ratio 0.3 \
                         --mask_mode $MASK_MODE \
                         --model_name $MODEL_NAME \
                         --prompting $PROMPTING \
                         --train $TRAIN \
                         --eval $EVAL \
                         --base_path $BASE_PATH \
                         --data-type $DATA_TYPE \
                         "${DATA_ROOT_ARGS[@]}" \
                         --results-path $RESULTS_PATH \
                         --output-model-name $OUTPUT_MODEL_NAME \
                         --num_train_sessions $NUM_TRAIN_SESSIONS \
                         --test_eid $TEST_EID \
                         --model_path $MODEL_PATH \
                         --use_dummy


cd script

conda deactivate

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "Runtime (seconds): $RUNTIME"
