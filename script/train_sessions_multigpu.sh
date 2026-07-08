#!/bin/bash

#SBATCH --job-name=ss-mg
#SBATCH --output=../results/%x-%j.out
#SBATCH --error=../results/%x-%j.err

#SBATCH -t 01:00:00

#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=4

. ~/.bashrc
conda activate mtm

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

BASE_PATH="/work/nvme/beml/ac136/IBL_MTM/data"
DATA_TYPE="processed_mtm"
RESULTS_PATH="/work/nvme/beml/ac136/IBL_MTM/results/mtm"

EID=${1}
TRAIN_BATCH_SIZE=${2:-}
EVAL_BATCH_SIZE=${3:-}

cmd=(accelerate launch --num_processes "$gpu_count" --num_machines 1 src/train_sessions.py \
    --eid "$EID" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$RESULTS_PATH")
if [ -n "${TRAIN_BATCH_SIZE}" ]; then
    cmd+=(--train-batch-size "${TRAIN_BATCH_SIZE}")
fi
if [ -n "${EVAL_BATCH_SIZE}" ]; then
    cmd+=(--eval-batch-size "${EVAL_BATCH_SIZE}")
fi

echo "EID: $EID"
echo "GPUs: $gpu_count"

"${cmd[@]}"
exit_code=$?

rm -rf "$HF_CACHE_DIR"

cd script

conda deactivate
exit "${exit_code}"
