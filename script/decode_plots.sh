#!/bin/bash
# Diagnostic plots for the behaviour-decoding sessions (CPU: refits the
# raw-spike baseline, audits it against the saved run, then plots).
#
#   cd /projects/beml/ac136/ndt-decoding              # detached, pinned
#   L=/projects/beml/ac136/IBL_MtM_model/results_parity
#   sbatch --export=ALL,EXPECT_SHA=$(git rev-parse HEAD) \
#          -o $L/decplot-%j.out -e $L/decplot-%j.err script/decode_plots.sh [DIR] [--only static]
#
# DIR defaults to decoding_v2. Writes ${RESULTS_ROOT}/DIR/plots/<eid>/; each
# session refuses to overwrite, except with --only static, which regenerates
# just choice_confusion.png / block_recall.png. Exit 1 if any session fails.

#SBATCH --job-name=decode-plots
#SBATCH --output=../results_parity/decplot-%j.out
#SBATCH --error=../results_parity/decplot-%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=96g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=beml-dtai-gh
#SBATCH --gpus-per-node=1

. ~/.bashrc
conda activate mtm

set -u

DIR=${1:-decoding_v2}
shift $(( $# > 0 ? 1 : 0 ))
EXTRA=("$@")
SESSIONS=(71e55bfe-5a3a-4cba-bdc7-f085140d798e 862ade13-53cd-4221-a3fa-dda8643641f2
          4aa1d525-5c7d-4c50-a147-ec53a9014812 b9c205c3-feac-485b-a89d-afc96d9cb280
          5b49aca6-a6f4-4075-931a-617ad64c219c)

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

echo "=================================================="
echo "decoding diagnostic plots : ${DIR}"
echo "git sha                   : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "slurm job                 : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; plots must be attributable to a commit." >&2
    git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$(git rev-parse HEAD)" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $(git rev-parse HEAD)." >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi

nfail=0
for EID in "${SESSIONS[@]}"; do
    echo; echo "---------- ${EID} ----------"
    python script/decode_plots.py "$EID" --dir "$DIR" ${EXTRA[@]+"${EXTRA[@]}"}
    rc=$?
    echo "${EID:0:8} exit=${rc}"
    [ $rc -eq 0 ] || nfail=$((nfail + 1))
done

echo; echo "decode plots: $(( ${#SESSIONS[@]} - nfail ))/${#SESSIONS[@]} sessions ok"
conda deactivate
[ $nfail -eq 0 ]
