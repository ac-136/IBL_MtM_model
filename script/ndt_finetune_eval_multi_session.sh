#!/bin/bash

# Stage-1 parity twin of finetune_eval_multi_session.sh, running the vendored
# ndt/ package.
#
# Usage:
#   sbatch ndt_finetune_eval_multi_session.sh MODEL MASK NUM_SESSIONS EID MODE \
#       MODEL_PATH [OUTPUT_MODEL_NAME] [DATA_ROOT]
#
# Dataset and output paths are settable without editing this file:
#   BASE_PATH=/my/data DATA_TYPE=my_sessions \
#     sbatch ndt_finetune_eval_multi_session.sh NDT1 all 1 <eid> train-eval <ckpt>
# Data is read from  $BASE_PATH/$DATA_TYPE/<eid>/data/{train,val,test}.parquet
# unless DATA_ROOT (arg 8) is given, which overrides the input path only.
#
# RESULTS_PATH defaults to .../results/mtm_ndtport so this lands beside the
# original run rather than overwriting it. To compare, diff the .npy files under
#   <results>/mtm/<name>/eval/<eid>/   vs   <results>/mtm_ndtport/<name>/eval/<eid>/

#SBATCH --job-name=ndt-single-session-finetune-eval
#SBATCH --output=ndt-single-session-finetune-eval-%j.out
#SBATCH --error=ndt-single-session-finetune-eval-%j.err

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

# if train in MODE
if [[ $MODE == *"train"* ]]; then
    TRAIN=True
fi

if [[ $MODE == *"eval"* ]]; then
    EVAL=True
fi

# ---- settable dataset / output paths -------------------------------------
BASE_PATH="${BASE_PATH:-/work/nvme/beml/ac136/IBL_MTM/data}"
DATA_TYPE="${DATA_TYPE:-processed_mtm}"
RESULTS_PATH="${RESULTS_PATH:-/work/nvme/beml/ac136/IBL_MTM/results/mtm_ndtport}"
# --------------------------------------------------------------------------

DATA_ROOT_ARGS=()
if [[ -n "$DATA_ROOT" ]]; then
    if [[ ! -d "$DATA_ROOT/$TEST_EID/data" ]]; then
        echo "Dataset directory not found: $DATA_ROOT/$TEST_EID/data" >&2
        exit 2
    fi
    DATA_ROOT_ARGS=(--data-root "$DATA_ROOT")
elif [[ ! -d "$BASE_PATH/$DATA_TYPE/$TEST_EID/data" ]]; then
    echo "Dataset directory not found: $BASE_PATH/$DATA_TYPE/$TEST_EID/data" >&2
    exit 2
fi

echo "Tree: ndt (vendored package)"
echo "Mask mode: $MASK_MODE"
echo "Model name: $MODEL_NAME"
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

python -m ndt.cli.finetune_eval --mask_ratio 0.3 \
                         --mask_mode $MASK_MODE \
                         --model_name $MODEL_NAME \
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

EXIT_CODE=$?

cd script

conda deactivate

END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
echo "Runtime (seconds): $RUNTIME"
exit $EXIT_CODE
