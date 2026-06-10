#!/bin/bash

#SBATCH --job-name=benchmark-ss-mg
#SBATCH --output=benchmark-ss-mg-%j.out
#SBATCH --error=benchmark-ss-mg-%j.err

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
export TRAIN_SESSIONS_RESULTS_PATH="benchmark_results/gpu_throughput/max_batch"

EID=${1}
TRAIN_BATCH_SIZE=${2:-}
EVAL_BATCH_SIZE=${3:-}
BENCHMARK_REPEAT_FACTOR=${4:-}
RUN_ID=${SLURM_JOB_ID:-local}
benchmark_file="gpu_benchmark_${EID}_${RUN_ID}.txt"
gpu_monitor_file="gpu_util_${EID}_${RUN_ID}.csv"
gpu_monitor_pid=""
start_time=$(date +%s)
status="running"
exit_code=0
cmd=(accelerate launch --num_processes "$gpu_count" --num_machines 1 src/train_sessions.py --eid "$EID")
if [ -n "${TRAIN_BATCH_SIZE}" ]; then
    cmd+=(--train-batch-size "${TRAIN_BATCH_SIZE}")
fi
if [ -n "${EVAL_BATCH_SIZE}" ]; then
    cmd+=(--eval-batch-size "${EVAL_BATCH_SIZE}")
fi
if [ -n "${BENCHMARK_REPEAT_FACTOR}" ]; then
    cmd+=(--benchmark-repeat-factor "${BENCHMARK_REPEAT_FACTOR}")
fi

iso_time() {
    date -d "@$1" --iso-8601=seconds 2>/dev/null || date -u -r "$1" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -u
}

command_string() {
    printf "%q " "$@"
}

start_gpu_monitor() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi \
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw \
            --format=csv \
            -l 1 > "${gpu_monitor_file}" &
        gpu_monitor_pid=$!
        echo "GPU utilization monitor writing to ${gpu_monitor_file}"
    else
        gpu_monitor_file="unavailable"
    fi
}

stop_gpu_monitor() {
    if [ -n "${gpu_monitor_pid}" ]; then
        kill "${gpu_monitor_pid}" 2>/dev/null || true
        wait "${gpu_monitor_pid}" 2>/dev/null || true
        gpu_monitor_pid=""
    fi
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

write_benchmark() {
    local end_time duration gpu_hours
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    gpu_hours=$(python - <<PYTHON
print(${duration} * ${gpu_count} / 3600.0)
PYTHON
)

    cat > "${benchmark_file}" <<EOF
eid: ${EID}
train_batch_size: ${TRAIN_BATCH_SIZE:-config_default}
eval_batch_size: ${EVAL_BATCH_SIZE:-config_default}
benchmark_repeat_factor: ${BENCHMARK_REPEAT_FACTOR:-1}
run_id: ${RUN_ID}
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
gpu_monitor_file: ${gpu_monitor_file}
EOF
    echo "GPU benchmark written to ${benchmark_file}"
}

record_interrupt() {
    exit_code=$1
    status="interrupted"
    stop_gpu_monitor
    write_benchmark
    cd script 2>/dev/null || true
    conda deactivate 2>/dev/null || true
    exit "${exit_code}"
}

trap 'record_interrupt 130' INT
trap 'record_interrupt 143' TERM

echo "EID: $EID"
echo "GPUs: $gpu_count"

start_gpu_monitor
"${cmd[@]}"
exit_code=$?
stop_gpu_monitor
if [ "${exit_code}" -eq 0 ]; then
    status="success"
else
    status="failed"
fi
write_benchmark

cd script

conda deactivate
exit "${exit_code}"
