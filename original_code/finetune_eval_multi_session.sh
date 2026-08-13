#!/bin/bash

#SBATCH --job-name=multi-session
#SBATCH --output=multi-session-%j.out
#SBATCH --error=multi-session-%j.err

#SBATCH --time=04:00:00
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate og_base

# Check python path
echo "Conda prefix: $CONDA_PREFIX"
which python

MODEL_NAME=${1}
MASK_MODE=${2}
NUM_TRAIN_SESSIONS=${3}
TEST_EID=${4}
MODE=${5}

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

BASE_PATH="/work/hdd/beml/ac136"

echo "Mask mode: $MASK_MODE"
echo "Model name: $MODEL_NAME"
echo "Prompting: $PROMPTING"
echo "Train: $TRAIN"
echo "Eval: $EVAL"
echo "Base path: $BASE_PATH"
echo "Num train sessions: $NUM_TRAIN_SESSIONS"
echo "Test eid: $TEST_EID"

cd ../

#  --base_path $SCRATCH/IBL_foundation_model \
python src/finetune_eval_multi_session_just_spikes.py --mask_ratio 0.3 \
                         --mask_mode $MASK_MODE \
                         --model_name $MODEL_NAME \
                         --prompting $PROMPTING \
                         --train $TRAIN \
                         --eval $EVAL \
                         --base_path $BASE_PATH \
                         --num_train_sessions $NUM_TRAIN_SESSIONS \
                         --test_eid $TEST_EID \
                         --use_dummy

cd script

conda deactivate