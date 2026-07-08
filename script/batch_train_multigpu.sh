#!/bin/bash

#SBATCH --job-name=batch-train-mg
#SBATCH --output=../results/%x-%j.out
#SBATCH --error=../results/%x-%j.err

#SBATCH --time=00:05:00

#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --account=beml-dtai-gh

EID_LIST="${1:-../data/mtm_overlap_eids.txt}"

if [ ! -f "$EID_LIST" ]; then
    echo "EID list not found: $EID_LIST" >&2
    exit 1
fi

while IFS= read -r eid || [ -n "$eid" ]; do
    if [ -z "$eid" ]; then
        continue
    fi

    echo "Submitting job for eid ${eid}_aligned"

    sbatch train_sessions_multigpu.sh "${eid}_aligned"
done < "$EID_LIST"
