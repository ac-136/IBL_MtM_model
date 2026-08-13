#!/bin/bash

#SBATCH --job-name=batch-finetune-mg
#SBATCH --output=../results/%x-%j.out
#SBATCH --error=../results/%x-%j.err

#SBATCH --time=00:05:00

#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

RESULTS_PATH="/work/nvme/beml/ac136/IBL_MTM/results/mtm"
TARGET_EID_LIST="${1:-../data/test_re_eids.txt}"

if [ ! -f "$TARGET_EID_LIST" ]; then
    echo "Target EID list not found: $TARGET_EID_LIST" >&2
    exit 1
fi

for model_path in "$RESULTS_PATH"/*/train/model_best.pt; do
    [ -f "$model_path" ] || continue
    source_eid="$(basename "$(dirname "$(dirname "$model_path")")")"

    while IFS= read -r target_eid || [ -n "$target_eid" ]; do
        if [ -z "$target_eid" ]; then
            continue
        fi

        echo "Submitting finetune+eval: source=$source_eid target=$target_eid"

        sbatch finetune_multigpu.sh NDT1 temporal 1 "${target_eid}_aligned" train-eval "$model_path"
    done < "$TARGET_EID_LIST"
done
