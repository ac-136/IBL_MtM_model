#!/bin/bash

#SBATCH --job-name=benchmark-ss
#SBATCH --output=benchmark-ss-%j.out
#SBATCH --error=benchmark-ss-%j.err

#SBATCH -t 05:00:00

#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

cd ..

gpu_count=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}
if ! [ "${gpu_count}" -ge 1 ] 2>/dev/null; then
    gpu_count=1
fi
export GPU_BENCHMARK_GPUS=$gpu_count

EID=${1}
OUTPUT_MODEL_NAME=${2:-mtm-single-session_test}
BASE_PATH="/work/nvme/beml/ac136/IBL_MTM/data"
DATA_TYPE="processed_mtm"
RESULTS_PATH="/work/nvme/beml/ac136/IBL_MTM/results/mtm"
RUN_ID=${SLURM_JOB_ID:-local}
benchmark_file="gpu_benchmark_${EID}_${RUN_ID}.txt"
start_time=$(date +%s)
status="running"
exit_code=0
cmd=(python src/train_sessions.py \
    --eid "$EID" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$RESULTS_PATH" \
    --output-model-name "$OUTPUT_MODEL_NAME")

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

echo "EID: $EID"
echo "Base path: $BASE_PATH"
echo "Data type: $DATA_TYPE"
echo "Results path: $RESULTS_PATH"
echo "Output model name: $OUTPUT_MODEL_NAME"

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
