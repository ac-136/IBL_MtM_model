#!/bin/bash
# Stage 3: one source -> unseen development-target transfer, the measurement
# the whole pilot exists to make.
#
#   sbatch pilot_transfer.sh SRC_EID TGT_EID [EPOCHS]
#
# Takes a Stage-2 pretrained source and finetunes + evaluates it on a session it
# has never seen, producing the co-smoothing bits per spike that hard criterion 5
# is judged on. Ten of these run across five sources x two development targets.
#
# The target is always a DEVELOPMENT target. The four official held-out sessions
# stay sealed until the recipe is frozen; script/stage3_preflight.py enforces
# that against the manifests before anything is submitted.

#SBATCH --job-name=pilot-xfer
#SBATCH --output=../results_parity/pilot-xfer-%j.out
#SBATCH --error=../results_parity/pilot-xfer-%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

SRC_EID=${1:?usage: pilot_transfer.sh SRC_EID TGT_EID [EPOCHS]}
TGT_EID=${2:?usage: pilot_transfer.sh SRC_EID TGT_EID [EPOCHS]}
EPOCHS=${3:-1000}
DATA_ROOT=/work/nvme/beml/ac136/LFP_FM/data/datasets

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

SRC_CKPT="${RESULTS_ROOT}/pilot/pretrain/src_${SRC_EID}/train/${SRC_EID}/model_best.pt"
OUT_ROOT="${RESULTS_ROOT}/pilot/transfer"
RUN_DIR="${OUT_ROOT}/from_${SRC_EID}"
LOG_DIR="${RUN_DIR}/finetune/${TGT_EID}"

echo "=================================================="
echo "Stage 3: transfer S -> H, S != H"
echo "source S       : ${SRC_EID}"
echo "target H       : ${TGT_EID}  (development target)"
echo "source ckpt    : ${SRC_CKPT}"
echo "epochs         : ${EPOCHS}"
echo "data root      : ${DATA_ROOT}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git describe   : $(git describe --tags --always 2>/dev/null || echo unknown)"
echo "slurm job      : ${SLURM_JOB_ID:-local}"
echo "slurm nodelist : ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; a pilot run must be attributable to a commit." >&2
    git status --porcelain >&2
    exit 2
fi
# 2c item 7: Slurm runs against the LIVE working tree, so a commit landing while
# this job sat in the queue would silently change what it measures. A clean tree
# does not catch that -- the tree is clean at both commits. Pass EXPECT_SHA to
# pin the exact commit this job was submitted against.
if [ -n "${EXPECT_SHA:-}" ]; then
    ACTUAL_SHA="$(git rev-parse HEAD)"
    if [ "$ACTUAL_SHA" != "$EXPECT_SHA" ]; then
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $ACTUAL_SHA." >&2
        echo "       Something was committed while this job was queued." >&2
        exit 2
    fi
    echo "commit verified: $ACTUAL_SHA"
fi

[ -f "$SRC_CKPT" ] || { echo "ERROR: source checkpoint missing: $SRC_CKPT" >&2; exit 2; }
[ -d "$DATA_ROOT/$TGT_EID/data" ] || { echo "ERROR: no data for $TGT_EID" >&2; exit 2; }
if [ "$SRC_EID" = "$TGT_EID" ]; then
    echo "ERROR: source and target are the same session; this measures nothing." >&2
    exit 2
fi
if [ -e "$LOG_DIR" ]; then
    echo "ERROR: $LOG_DIR exists. Remove it or choose another root." >&2
    exit 2
fi
mkdir -p "$LOG_DIR" || exit 2

# ---------------------------------------------------------------- metadata
CMD="pilot_transfer.sh $SRC_EID $TGT_EID $EPOCHS"
python script/write_capture_meta.py "$LOG_DIR" "from_${SRC_EID}_to_${TGT_EID}" \
    "$TGT_EID" "$EPOCHS" --cmd "$CMD" --config "$LOG_DIR/resolved_config.yaml"

# ------------------------------------------- [1/2] the 89/5 transfer contract
# Checked before any optimizer step, on THIS pair rather than on the one pair
# Stage 1 happened to verify. Costs one extra dataset load against a multi-hour
# job, and it is the difference between "the contract is tested somewhere" and
# "this run obeyed it".
echo
echo "---------- [1/2] transfer contract, pre-optimizer-step ----------"
python script/check_transfer_contract.py "$SRC_CKPT" "$TGT_EID" "$DATA_ROOT"
rc=$?
if [ $rc -ne 0 ]; then
    echo "transfer contract FAILED; not spending GPU-hours on this pair" >&2
    exit $rc
fi

# ------------------------------------------------ [2/2] finetune + evaluate
# No --save-plot-every-n-epochs here, and deliberately so. 2c item 3 added that
# flag to train_single because ssl_session_trainer.yaml plots every epoch;
# finetune_sessions_trainer.yaml already plots every 10, which over 1000 epochs
# is ~230 PNGs and ~150 MB -- acceptable. Adding the flag to finetune_eval would
# mean a code change after baseline_2c_provenance was gated and tagged, buying
# nothing.
echo
echo "---------- [2/2] finetune + eval, S -> H ----------"
python -m ndt.cli.finetune_eval \
    --mask_ratio 0.3 \
    --mask_mode temporal \
    --model_name NDT1 \
    --train True \
    --eval True \
    --base_path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --data-root "$DATA_ROOT" \
    --results-path "$OUT_ROOT" \
    --output-model-name "from_${SRC_EID}" \
    --num_train_sessions 1 \
    --test_eid "$TGT_EID" \
    --model_path "$SRC_CKPT" \
    --num-epochs "$EPOCHS"
rc=$?

echo
echo "=================================================="
echo "transfer ${SRC_EID} -> ${TGT_EID} finished, exit=${rc}"
if [ $rc -eq 0 ]; then
    find "$RUN_DIR" -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.json' \) \
        | sed "s|^${RUN_DIR}/|  |" | sort
fi
echo "=================================================="

conda deactivate
exit $rc
