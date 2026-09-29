#!/bin/bash
# Behaviour decoding for one held-in session: the frozen-NDT1 probe (GPU)
# followed by the raw-spike linear baseline and chance/null (CPU).
#
#   cd /projects/beml/ac136/ndt-decoding              # detached, pinned
#   L=/projects/beml/ac136/IBL_MtM_model/results_parity
#   sbatch --export=ALL,EXPECT_SHA=$(git rev-parse HEAD) [--time=...] \
#          -J dec-${EID:0:8} -o $L/dec-%j.out -e $L/dec-%j.err \
#          script/decode_session.sh EID
#
# Outputs: ${RESULTS_ROOT}/${DECODE_DIR:-decoding}/{probe,baseline}/<eid>/ --
# both refuse to overwrite; set DECODE_DIR to write a new version beside an
# old one. --time comes from the canary; the default is a canary ceiling.

#SBATCH --job-name=decode
#SBATCH --output=../results_parity/dec-%j.out
#SBATCH --error=../results_parity/dec-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

EID=${1:?usage: decode_session.sh EID}

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

DECODE_DIR="${DECODE_DIR:-decoding}"
PROBE_OUT="${RESULTS_ROOT}/${DECODE_DIR}/probe/${EID}"
BASE_OUT="${RESULTS_ROOT}/${DECODE_DIR}/baseline/${EID}"

echo "=================================================="
echo "behaviour decoding   : ${EID}"
echo "git sha              : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "slurm job            : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; runs must be attributable to a commit." >&2
    git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $(git rev-parse HEAD)." >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi
for d in "$PROBE_OUT" "$BASE_OUT"; do
    if [ -e "$d" ]; then
        echo "ERROR: $d exists. Remove it or choose another root." >&2; exit 2
    fi
done

# ------------------------------------------------------------ 1. probe (GPU)
echo; echo "---------- [1/2] frozen-NDT1 probe ----------"
mkdir -p "$PROBE_OUT" || exit 2
python script/write_capture_meta.py "$PROBE_OUT" "decode_probe_${EID}" "$EID" 0 \
    --cmd "decode_session.sh $EID" || exit 2
python script/behavior_decode.py "$EID" --out-root "$(dirname "$PROBE_OUT")"
rc_probe=$?
echo "probe exit=${rc_probe}"

# ------------------------------------------------------------ 2. baseline (CPU)
echo; echo "---------- [2/2] raw-spike baseline + chance ----------"
mkdir -p "$BASE_OUT" || exit 2
python script/write_capture_meta.py "$BASE_OUT" "decode_baseline_${EID}" "$EID" 0 \
    --cmd "decode_session.sh $EID" || exit 2
python script/decode_baseline.py "$EID" --out-root "$(dirname "$BASE_OUT")"
rc_base=$?
echo "baseline exit=${rc_base}"

echo; echo "decoding ${EID} finished: probe=${rc_probe} baseline=${rc_base}"
conda deactivate
[ $rc_probe -eq 0 ] && [ $rc_base -eq 0 ]
