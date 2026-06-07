#!/bin/bash

#SBATCH --job-name=batch-train
#SBATCH --output=batch-train-%j.out
#SBATCH --error=batch-train-%j.err

#SBATCH --time=00:05:00

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

BASE_EID_DIR="/work/hdd/beml/ac136/processed_kimia_8_22"

for eid_dir in "$BASE_EID_DIR"/*/; do
    if [ ! -d "$eid_dir" ]; then
        continue
    fi

    eid_dir="${eid_dir%/}"
    eid_name="$(basename "$eid_dir")"

    echo "Submitting job for EID directory $eid_name"

    sbatch "train_sessions.sh" "$eid_name"
done
