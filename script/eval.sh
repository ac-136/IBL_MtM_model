#!/bin/bash

#SBATCH --job-name=eval
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err

#SBATCH --time=04:00:00
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

MODEL_NAME=${1} # Model name e.g. NDT1, NDT2
MASK_MODE=${2} # Mask mode e.g. all,temporal
NUM_TRAIN_SESSIONS=${3} # Number of training sessions, e.g. 1, 10, 34
MODE=${4} # train,eval,train-eval

while IFS= read -r line
do
    echo "Evaluate on test eid: $line"
    sbatch finetune_eval_multi_session.sh $MODEL_NAME $MASK_MODE $NUM_TRAIN_SESSIONS $line $MODE
done < "../data/test_re_eids.txt"