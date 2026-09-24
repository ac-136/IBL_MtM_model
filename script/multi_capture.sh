#!/bin/bash
# Multi-session capture: one reproducible train run on an explicit session list.
#
#   sbatch multi_capture.sh LABEL EPOCHS --sessions EID [EID ...]
#   sbatch multi_capture.sh LABEL EPOCHS --manifest PATH
#
# Writes to $RESULTS_ROOT/multisession/$LABEL/ :
#   ms/train/...            checkpoints, loss_history.npz, run_summary.json
#   capture_meta.json       environment and provenance
#
# Held-in sessions only: every eid is checked against the official held-out
# manifests before anything runs. Own-session evaluation of each member is added
# in M3, once the audits accept a multi-session vocabulary (defect M-g).

#SBATCH --job-name=ms-cap
#SBATCH --output=../results_parity/ms-cap-%j.out
#SBATCH --error=../results_parity/ms-cap-%j.err
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

LABEL=${1:?usage: multi_capture.sh LABEL EPOCHS (--sessions EID... | --manifest PATH)}
EPOCHS=${2:?usage: multi_capture.sh LABEL EPOCHS (--sessions EID... | --manifest PATH)}
shift 2
SESSION_ARGS=("$@")
[ ${#SESSION_ARGS[@]} -ge 2 ] || { echo "ERROR: give --sessions EID... or --manifest PATH" >&2; exit 2; }
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
echo "multi-session capture : ${LABEL}"
echo "epochs                : ${EPOCHS}"
echo "sessions              : ${SESSION_ARGS[*]}"
echo "git sha               : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "git describe          : $(git describe --tags --always 2>/dev/null || echo unknown)"
echo "slurm job             : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; a capture must be attributable to a commit." >&2
    git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    ACTUAL_SHA="$(git rev-parse HEAD)"
    [ "$ACTUAL_SHA" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $ACTUAL_SHA." >&2; exit 2; }
    echo "commit verified: $ACTUAL_SHA"
fi

# Sealed-set guard: resolve the list exactly as train_multi will, then refuse any
# official held-out eid. Manifest sizes are asserted so an empty read cannot make
# the guard vacuous.
python - "${SESSION_ARGS[@]}" <<'PY' || exit 2
import os, sys
sys.path.insert(0, "script"); sys.path.insert(0, ".")
import stage3_preflight as pf
from ndt.cli.train_multi import build_parser, resolve_sessions
eids, src, _ = resolve_sessions(build_parser().parse_args(sys.argv[1:]))
ho, hi = set(), set()
for c in ("rs", "bwm"):
    ho |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldout_eids.txt")))
    hi |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldin_eids.txt")))
assert len(ho) == 80 and len(hi) == 321, f"manifest sizes {len(ho)}/{len(hi)}; guard untrustworthy"
bad = [e for e in eids if e in ho]
if bad:
    sys.exit(f"ERROR: official held-out session(s) requested for training: {bad}")
missing = [e for e in eids if e not in hi]
if missing:
    sys.exit(f"ERROR: session(s) not in the held-in pool: {missing}")
print(f"sealed-set guard: {len(eids)} session(s), all held-in (source: {src})")
PY

if [ -e "$OUT" ]; then
    echo "ERROR: $OUT already exists. Choose another LABEL or remove it." >&2; exit 2
fi
mkdir -p "$OUT" || exit 2
python script/write_capture_meta.py "$OUT" "$LABEL" "multi-session" "$EPOCHS" \
    --cmd "multi_capture.sh $LABEL $EPOCHS ${SESSION_ARGS[*]}"

python -m ndt.cli.train_multi \
    "${SESSION_ARGS[@]}" \
    --data-root "$DATA_ROOT" \
    --data-type datasets \
    --results-path "$OUT" \
    --output-model-name ms \
    --num-epochs "$EPOCHS"
rc=$?

echo
echo "multi-session capture ${LABEL} finished, exit=${rc}"
[ $rc -eq 0 ] && find "$OUT" -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.json' -o -name '*.pt' \) \
    | sed "s|^${OUT}/|  |" | sort
conda deactivate
exit $rc
