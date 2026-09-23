#!/bin/bash
# Step 0: capture one single-session reference run (train -> finetune -> eval).
#
#   sbatch step0_capture.sh LABEL [EPOCHS] [EID]
#     LABEL   names the capture directory, e.g. repro_a, repro_b, post3a, reference
#     EPOCHS  default 2 (gates B and D); use 5 for the canonical reference (E)
#     EID     default 0cad7ea8... (280 neurons, 207 trials: the smallest session)
#
# Writes to  $RESULTS_ROOT/step0/$LABEL/
#     ss_$BARE/train/$BARE/           model_best.pt, model_last.pt, loss/loss_history.npz
#     ft_$BARE/finetune/$BARE/        finetuned checkpoints
#     ft_$BARE/eval/$BARE/            co_smooth/bps.npy, forward_pred/bps.npy, r2.npy
#     capture_meta.json               environment and provenance
#
# Train, finetune and eval run in ONE job on purpose. Chaining them with
# --dependency=afterok previously stranded 28 downstream jobs as
# DependencyNeverSatisfied when an upstream job failed, and a partial capture is
# worse than none: the finetune leg is what exercises Trainer a second time
# under a different config.
#
# --use_dummy is deliberately omitted; see the note in parity_common.sh.

#SBATCH --job-name=step0-cap
#SBATCH --output=../results_parity/step0-cap-%j.out
#SBATCH --error=../results_parity/step0-cap-%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

LABEL=${1:?usage: step0_capture.sh LABEL [EPOCHS] [EID]}
EPOCHS=${2:-2}
EID=${3:-0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63_aligned}

# Slurm copies the submitted script to a spool directory, so "$0" is
# /var/spool/slurmd/jobNNN/slurm_script and dirname "$0" is useless here.
find_repo_root() {
    local c
    for c in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
             "$PWD" "$PWD/.." \
             "$(cd "$(dirname "$0")" 2>/dev/null && pwd)" \
             "$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; do
        [ -n "$c" ] || continue
        if [ -d "$c/ndt" ] && [ -d "$c/src" ] && [ -d "$c/script" ]; then
            (cd "$c" && pwd); return 0
        fi
    done
    return 1
}
REPO_ROOT="$(find_repo_root)" || { echo "cannot locate the repo root" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2
source script/parity_common.sh

BARE="${EID%_aligned}"
OUT_ROOT="${RESULTS_ROOT}/step0/${LABEL}"
TRAIN_DIR="${OUT_ROOT}/ss_${BARE}/train/${BARE}"
CKPT="${TRAIN_DIR}/model_best.pt"

echo "=================================================="
echo "step 0 capture : ${LABEL}"
echo "eid            : ${EID}"
echo "epochs         : ${EPOCHS}"
echo "out root       : ${OUT_ROOT}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git dirty      : $(test -n "$(git status --porcelain 2>/dev/null)" && echo YES || echo no)"
echo "slurm job      : ${SLURM_JOB_ID:-local}"
echo "slurm nodelist : ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

# A capture taken from a dirty tree cannot be attributed to a commit, which is
# the whole point of recording the SHA. Refuse rather than produce a reference
# that claims a provenance it does not have.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; commit before capturing." >&2
    git status --porcelain >&2
    exit 2
fi

# Re-running into a populated directory would silently mix two captures.
if [ -e "$OUT_ROOT" ]; then
    echo "ERROR: $OUT_ROOT already exists. Choose another LABEL or remove it." >&2
    exit 2
fi
mkdir -p "$OUT_ROOT" || exit 2

if [ ! -d "$BASE_PATH/$DATA_TYPE/$BARE/data" ] && [ ! -d "$BASE_PATH/$DATA_TYPE/${BARE}_aligned/data" ]; then
    echo "Dataset not found: $BASE_PATH/$DATA_TYPE/$BARE(_aligned)/data" >&2
    exit 2
fi

# ---------------------------------------------------------------- metadata
# Shared with pilot_*.sh so every run gets the same record, not just captures.
python script/write_capture_meta.py "$OUT_ROOT" "$LABEL" "$EID" "$EPOCHS" \
    --cmd "step0_capture.sh $LABEL $EPOCHS $EID"

# ---------------------------------------------------------------- 1. train
echo
echo "---------- [1/2] single-session training ----------"
python -m ndt.cli.train_single \
    --eid "$EID" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "ss_${BARE}" \
    --num-epochs "$EPOCHS"
rc=$?
if [ $rc -ne 0 ]; then
    echo "training failed (exit=$rc); not attempting finetune" >&2
    exit $rc
fi

if [ ! -f "$CKPT" ]; then
    echo "ERROR: training reported success but $CKPT is missing" >&2
    exit 3
fi

# ------------------------------------------------- 2. finetune + evaluate
echo
echo "---------- [2/2] finetune + eval ----------"
python -m ndt.cli.finetune_eval \
    --mask_ratio 0.3 \
    --mask_mode temporal \
    --model_name NDT1 \
    --train True \
    --eval True \
    --base_path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "ft_${BARE}" \
    --num_train_sessions 1 \
    --test_eid "$EID" \
    --model_path "$CKPT" \
    --num-epochs "$EPOCHS"
rc=$?

echo
echo "=================================================="
echo "capture ${LABEL} finished, exit=${rc}"
echo "artifacts under ${OUT_ROOT}:"
find "$OUT_ROOT" -type f \( -name '*.npz' -o -name '*.npy' -o -name '*.pt' -o -name '*.json' \) \
    | sed "s|^${OUT_ROOT}/|  |" | sort
echo "=================================================="

conda deactivate
exit $rc
