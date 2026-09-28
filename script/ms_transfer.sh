#!/bin/bash
# M4 V3: transfer a MULTI-session source checkpoint onto one development target.
#
#   sbatch ms_transfer.sh LABEL SRC_CKPT TGT_EID [EPOCHS]
#
# The transfer contract is checked on THIS pair before any optimizer step:
#   * source classification: 89 shared + (1 + 4N) source-session-specific;
#   * the source-specific tensors are excluded from the load;
#   * the 89 shared tensors land in the target exactly;
#   * the target's own 5 session-specific tensors stay at their pre-load state.
# If the contract fails, the finetune never runs.
#
# The target must be a development target, never an official held-out session.
#
# Writes to $RESULTS_ROOT/multisession/$LABEL/.

#SBATCH --job-name=ms-xfer
#SBATCH --output=../results_parity/ms-xfer-%j.out
#SBATCH --error=../results_parity/ms-xfer-%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

LABEL=${1:?usage: ms_transfer.sh LABEL SRC_CKPT TGT_EID [EPOCHS]}
SRC_CKPT=${2:?usage: ms_transfer.sh LABEL SRC_CKPT TGT_EID [EPOCHS]}
TGT_EID=${3:?usage: ms_transfer.sh LABEL SRC_CKPT TGT_EID [EPOCHS]}
EPOCHS=${4:-2}
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

OUT="${RESULTS_ROOT}/multisession/${LABEL}"

echo "=================================================="
echo "M4 V3 transfer : ${LABEL}"
echo "source ckpt    : ${SRC_CKPT}"
echo "target         : ${TGT_EID}"
echo "epochs         : ${EPOCHS}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "slurm job      : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty" >&2; git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $(git rev-parse HEAD)" >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi
[ -f "$SRC_CKPT" ] || { echo "ERROR: source checkpoint missing: $SRC_CKPT" >&2; exit 2; }

# Sealed-set guard: the target must be a development target and never an
# official held-out session. Manifest sizes are asserted so the guard can't pass
# vacuously on an empty read.
python - "$TGT_EID" <<'PY' || exit 2
import os, sys
sys.path.insert(0, "script")
import stage3_preflight as pf
t = sys.argv[1].removesuffix("_aligned")
ho = set()
for c in ("rs", "bwm"):
    ho |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldout_eids.txt")))
assert len(ho) == 80, f"held-out manifest size {len(ho)}; guard untrustworthy"
dev = set(pf.DEV_TARGETS) | {"41872d7f-75cb-4445-bb1a-132b354c44f0",
                             "837b4e6a-ccfd-49b0-a1dd-3aa53fbf2ecb"}
if t in ho:
    sys.exit(f"ERROR: {t} is an official held-out session")
if t not in dev:
    sys.exit(f"ERROR: {t} is not a development target")
print(f"sealed-set guard: target {t[:8]} is a development target, not held-out")
PY

if [ -e "$OUT" ]; then
    echo "ERROR: $OUT exists. Choose another LABEL or remove it." >&2; exit 2
fi
mkdir -p "$OUT" || exit 2
python script/write_capture_meta.py "$OUT" "$LABEL" "$TGT_EID" "$EPOCHS" \
    --cmd "ms_transfer.sh $LABEL $SRC_CKPT $TGT_EID $EPOCHS"

echo
echo "---------- [1/2] transfer contract, pre-optimizer-step ----------"
python script/check_transfer_contract.py "$SRC_CKPT" "$TGT_EID" "$DATA_ROOT" \
    | tee "$OUT/transfer_contract.txt"
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ]; then
    echo "transfer contract FAILED; not finetuning" >&2
    exit $rc
fi

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
    --data-root "$DATA_ROOT" \
    --results-path "$OUT" \
    --output-model-name xfer \
    --num_train_sessions 1 \
    --test_eid "$TGT_EID" \
    --model_path "$SRC_CKPT" \
    --num-epochs "$EPOCHS"
rc=$?

echo
echo "M4 V3 transfer ${LABEL} finished, exit=${rc}"
[ $rc -eq 0 ] && find "$OUT" -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.json' -o -name '*.pt' -o -name '*.txt' \) \
    | sed "s|^${OUT}/|  |" | sort
conda deactivate
exit $rc
