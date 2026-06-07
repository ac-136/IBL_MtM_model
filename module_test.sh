#!/bin/bash
# SBATCH --job-name=module-test
# SBATCH --output=module-test-%j.out
# SBATCH --error=module-test-%j.err

#SBATCH --time=00:20:00
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate ibl

# Confirm GPU visibility
echo "===== nvidia-smi output ====="
nvidia-smi

# Check PyTorch CUDA availability
echo "===== PyTorch CUDA check ====="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA version:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('Current GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    print('CUDA tensor result:', (x + y))
"

conda deactivate