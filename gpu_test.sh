#!/bin/bash
# SBATCH --job-name=gpu-test
# SBATCH --output=gpu-test-%j.out
# SBATCH --error=gpu-test-%j.err

#SBATCH --time=00:20:00
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc

# unload modules (get rid of all possible conflicts)
module purge

# load desired modules
module load gcc-native/12.3
module load PrgEnv-nvidia/8.5.0
module load cuda/11.8

# verify CUDA and env
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
which nvcc
nvcc --version

conda activate ibl-fm

cd /projects/beml/ac136/IBL_MtM_model

echo "===== nvidia-smi output ====="
nvidia-smi

echo "===== PyTorch CUDA check ====="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('Number of GPUs:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))
    # Run a simple CUDA tensor operation
    x = torch.rand(3,3).cuda()
    y = torch.rand(3,3).cuda()
    z = x + y
    print('CUDA tensor operation result:', z)
    print('Allocated GPU memory (MB):', torch.cuda.memory_allocated() / 1024**2)
    print('Cached GPU memory (MB):', torch.cuda.memory_reserved() / 1024**2)
else:
    print('CUDA not available')
"

echo "===== CUDA driver version ====="
cat /proc/driver/nvidia/version || echo 'No NVIDIA driver info found'

echo "===== GPU device properties ====="
python -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_properties(0))
else:
    print('No CUDA device available')
"

# run python code
python gpu_test.py

conda deactivate
