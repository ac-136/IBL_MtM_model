#!/bin/bash

#SBATCH --job-name=benchmark-ms
#SBATCH --output=benchmark-ms-%j.out
#SBATCH --error=benchmark-ms-%j.err

#SBATCH -t 10:00:00

#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

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
            shift
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

RUN_ID=${SLURM_JOB_ID:-local}
RUN_NAME="${MODEL_NAME:-multi_train}"
benchmark_file="gpu_benchmark_${RUN_NAME}_${RUN_ID}.txt"
start_time=$(date +%s)
status="running"
exit_code=0

cmd=(srun python src/multi_train_session.py)

if [ -n "$NUM_SESSIONS" ]; then
    cmd+=(--num-sessions "$NUM_SESSIONS")
fi

if [ -n "$MODEL_NAME" ]; then
    cmd+=(--model_name "$MODEL_NAME")
fi

if [ ${#TRAIN_SESSION_EIDS[@]} -gt 0 ]; then
    cmd+=(--train-session-eid "${TRAIN_SESSION_EIDS[@]}")
fi

iso_time() {
    date -d "@$1" --iso-8601=seconds 2>/dev/null || date -u -r "$1" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u
}

command_string() {
    printf "%q " "$@"
}

gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd "|" -)
if [ -z "${gpu_name}" ]; then
    gpu_name="unavailable"
fi
nvidia_driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
if [ -z "${nvidia_driver_version}" ]; then
    nvidia_driver_version="unavailable"
fi
python_version=$(python -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo "unavailable")
torch_version=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "unavailable")
torch_cuda_version=$(python -c 'import torch; print(torch.version.cuda or "unavailable")' 2>/dev/null || echo "unavailable")
git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unavailable")
git_status=$(git status --short 2>/dev/null | wc -l | tr -d ' ' || echo "unavailable")
hostname_value=$(hostname 2>/dev/null || echo "unavailable")
repo_dir=$(pwd)
train_session_eids_string="${TRAIN_SESSION_EIDS[*]}"

write_benchmark() {
    local end_time duration gpu_hours
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    gpu_hours=$(python - <<PYTHON
print(${duration} * ${gpu_count} / 3600.0)
PYTHON
)

    cat > "${benchmark_file}" <<EOF
run_name: ${RUN_NAME}
run_id: ${RUN_ID}
num_sessions: ${NUM_SESSIONS}
train_session_eids: ${train_session_eids_string}
status: ${status}
exit_code: ${exit_code}
command: $(command_string "${cmd[@]}")
repo_dir: ${repo_dir}
git_commit: ${git_commit}
git_dirty_files: ${git_status}
hostname: ${hostname_value}
python_version: ${python_version}
torch_version: ${torch_version}
torch_cuda_version: ${torch_cuda_version}
slurm_job_id: ${SLURM_JOB_ID:-}
slurm_job_name: ${SLURM_JOB_NAME:-}
slurm_partition: ${SLURM_JOB_PARTITION:-}
slurm_nodelist: ${SLURM_JOB_NODELIST:-}
start_time: $(iso_time "${start_time}")
end_time: $(iso_time "${end_time}")
duration_s: ${duration}
gpus: ${gpu_count}
gpu_name: ${gpu_name}
nvidia_driver_version: ${nvidia_driver_version}
gpu_hours_run: ${gpu_hours}
EOF
    echo "GPU benchmark written to ${benchmark_file}"
}

record_interrupt() {
    exit_code=$1
    status="interrupted"
    write_benchmark
    cd script 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    exit "${exit_code}"
}

trap 'record_interrupt 130' INT
trap 'record_interrupt 143' TERM

printf 'Launching multi-session training with command:\n%s\n' "$(command_string "${cmd[@]}")"

"${cmd[@]}"
exit_code=$?
if [ "${exit_code}" -eq 0 ]; then
    status="success"
else
    status="failed"
fi
write_benchmark

cd script

conda deactivate
exit "${exit_code}"
