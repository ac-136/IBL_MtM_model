#!/bin/bash

#SBATCH --job-name=ms-mg
#SBATCH --output=../results/%x-%j.out
#SBATCH --error=../results/%x-%j.err

#SBATCH -t 10:00:00

#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=4

. ~/.bashrc
conda activate mtm

MODEL_NAME=""
POSITIONAL_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --model_name|--model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            sqqhift
            ;;
    esac
done

NUM_SESSIONS="${POSITIONAL_ARGS[0]:-}"
TRAIN_SESSION_EIDS=("${POSITIONAL_ARGS[@]:1}")

if [ -n "$NUM_SESSIONS" ] && [ ${#TRAIN_SESSION_EIDS[@]} -gt 0 ] && [ "$NUM_SESSIONS" -ne ${#TRAIN_SESSION_EIDS[@]} ]; then
    echo "Error: num_sessions ($NUM_SESSIONS) does not match the number of provided train_session_eid values (${#TRAIN_SESSION_EIDS[@]})." >&2
    exit 1
fi

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

BASE_PATH="/work/hdd/beml/ac136"
DATA_TYPE="processed_mtm"
RESULTS_PATH="benchmark_results/ms"

cmd=(accelerate launch --num_processes "$gpu_count" --num_machines 1 src/multi_train_session.py \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$RESULTS_PATH")

if [ -n "$NUM_SESSIONS" ]; then
    cmd+=(--num-sessions "$NUM_SESSIONS")
fi

if [ -n "$MODEL_NAME" ]; then
    cmd+=(--model_name "$MODEL_NAME")
fi

if [ ${#TRAIN_SESSION_EIDS[@]} -gt 0 ]; then
    cmd+=(--train-session-eid "${TRAIN_SESSION_EIDS[@]}")
fi

printf 'Launching multi-session multi-GPU training with command:\n%s\n' "$(printf '%q ' "${cmd[@]}")"
echo "GPUs: $gpu_count"

"${cmd[@]}"
exit_code=$?

rm -rf "$HF_CACHE_DIR"

cd script

conda deactivate
exit "${exit_code}"
