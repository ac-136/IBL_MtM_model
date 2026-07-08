#!/bin/bash

#SBATCH --job-name=finetune-mg
#SBATCH --output=../results/%x-%j.out
#SBATCH --error=../results/%x-%j.err

#SBATCH --time=04:00:00

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=4

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

TRAIN=False
EVAL=FALSE

if [ $MASK_MODE == "all" ]; then
    PROMPTING=True
else
    PROMPTING=False
fi

# if train in MODE
if [[ $MODE == *"train"* ]]; then
    TRAIN=True
fi

if [[ $MODE == *"eval"* ]]; then
    EVAL=True
fi

BASE_PATH="/work/hdd/beml/ac136"
DATA_TYPE="just_spikes"
RESULTS_PATH="results_og_multi_session"

cd ..

gpu_count=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
if ! [ "${gpu_count}" -ge 1 ] 2>/dev/null; then
    gpu_count=1
fi
export GPU_BENCHMARK_GPUS=$gpu_count

HF_CACHE_DIR="/tmp/hf_datasets_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE="$HF_CACHE_DIR/datasets"

echo "Mask mode: $MASK_MODE"
echo "Model name: $MODEL_NAME"
echo "Prompting: $PROMPTING"
echo "Train: $TRAIN"
echo "Eval: $EVAL"
echo "Base path: $BASE_PATH"
echo "Num train sessions: $NUM_TRAIN_SESSIONS"
echo "Test eid: $TEST_EID"
echo "Single session model name: $MODEL_PATH"
echo "GPUs: $gpu_count"

accelerate launch --num_processes "$gpu_count" --num_machines 1 \
    src/finetune_eval_multi_session_just_spikes.py --mask_ratio 0.3 \
                         --mask_mode $MASK_MODE \
                         --model_name $MODEL_NAME \
                         --prompting $PROMPTING \
                         --train $TRAIN \
                         --eval $EVAL \
                         --base_path $BASE_PATH \
                         --data-type $DATA_TYPE \
                         --results-path $RESULTS_PATH \
                         --num_train_sessions $NUM_TRAIN_SESSIONS \
                         --test_eid $TEST_EID \
                         --model_path $MODEL_PATH \
                         --use_dummy
exit_code=$?

rm -rf "$HF_CACHE_DIR"

cd script

conda deactivate
exit "${exit_code}"
