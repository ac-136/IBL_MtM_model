#!/bin/bash
# Post-merge gate: does the single-session tooling still reproduce the campaign
# on the merged tree (ms_validated merged into the single-session line)?
#
#   cd /projects/beml/ac136/ndt-merge-gate          # detached, pinned
#   L=/projects/beml/ac136/IBL_MtM_model/results_parity
#   sbatch --export=ALL,EXPECT_SHA=$(git rev-parse HEAD) \
#          -o $L/merge-gate-%j.out -e $L/merge-gate-%j.err script/merge_gate.sh
#
# G1  offline suites: test_multisession, test_vocab_contract, test_learnability
# G2  own_session_eval --dry-run: 5 pilot sources (pilot mode), 2 manifest
#     sessions (one reused pilot pretrain, one fresh), and negative controls
#     that must exit 3
# G3  own_session_eval --manifest on v1 learnability checkpoints, written to a
#     new root and compared exactly with the campaign output (bps.npy, r2.npy
#     byte for byte; trial_terms.npz array for array; summary variants equal)
#
# Writes only under ${RESULTS_ROOT}/merge_gate/<sha12>; refuses if it exists.

#SBATCH --job-name=merge-gate
#SBATCH --output=../results_parity/merge-gate-%j.out
#SBATCH --error=../results_parity/merge-gate-%j.err
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

SHA=$(git rev-parse HEAD)
MANIFEST=script/learnability_manifest.json
CAMPAIGN="${RESULTS_ROOT}/learnability/own_session"
PRE_ROOT="${RESULTS_ROOT}/learnability/pretrain"
GATE_ROOT="${RESULTS_ROOT}/merge_gate/${SHA:0:12}"

PILOT=(862ade13-53cd-4221-a3fa-dda8643641f2 71e55bfe-5a3a-4cba-bdc7-f085140d798e
       5b49aca6-a6f4-4075-931a-617ad64c219c b182b754-3c3e-4942-8144-6ee790926b58
       f56194bc-8215-4ae8-bc6a-89781ad8e050)
DEV_TARGET=572a95d1-39ca-42e1-8424-5c9ffcb2df87
HELD_OUT=0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63
# G3 sessions: the canary (fresh, rs), a fresh bwm session whose neuron-mean is
# dominated by sparse neurons, and a reused pilot pretrain.
G3=(f140a2ec-fd49-4814-994a-fe3476f14e66 0b7ee1b6-42db-46cd-a465-08f531366187
    862ade13-53cd-4221-a3fa-dda8643641f2)

echo "=================================================="
echo "merge gate"
echo "git sha              : ${SHA}"
echo "git describe         : $(git describe --tags --always 2>/dev/null)"
echo "slurm job            : ${SLURM_JOB_ID:-local}   node: ${SLURM_JOB_NODELIST:-local}"
echo "gate root            : ${GATE_ROOT}"
echo "=================================================="

if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "ERROR: working tree is dirty; the gate must be attributable to a commit." >&2
    git status --porcelain >&2; exit 2
fi
if [ -n "${EXPECT_SHA:-}" ]; then
    [ "$SHA" = "$EXPECT_SHA" ] || {
        echo "ERROR: expected commit $EXPECT_SHA but the tree is at $SHA." >&2; exit 2; }
    echo "commit verified: $EXPECT_SHA"
fi
[ -z "$(git diff ms_validated HEAD -- ndt/)" ] || {
    echo "ERROR: ndt/ differs from ms_validated; this gate assumes the validated ndt/." >&2; exit 2; }
echo "ndt/ identical to ms_validated"
if [ -e "$GATE_ROOT" ]; then
    echo "ERROR: $GATE_ROOT exists. Remove it or gate another commit." >&2; exit 2
fi
mkdir -p "$GATE_ROOT" || exit 2
python script/write_capture_meta.py "$GATE_ROOT" "merge_gate" "none" 0 \
    --cmd "merge_gate.sh" || exit 2

declare -a RESULTS=()
record() {   # name, pass(0/1), detail
    local tag; [ "$2" -eq 0 ] && tag=PASS || tag=FAIL
    RESULTS+=("[$tag] $1${3:+   $3}")
    echo ">>> [$tag] $1${3:+   $3}"
}
expect_rc() {   # name, expected rc, command...
    local name=$1 want=$2; shift 2
    "$@"; local rc=$?
    [ $rc -eq "$want" ]; record "$name" $? "exit $rc (want $want)"
}

# ------------------------------------------------------------ G1 offline
echo; echo "---------- G1: offline suites ----------"
for t in test_multisession test_vocab_contract test_learnability; do
    expect_rc "G1 $t" 0 python script/$t.py
done

# ------------------------------------------------------------ G2 dry-runs
echo; echo "---------- G2: dry-runs ----------"
for e in "${PILOT[@]}"; do
    expect_rc "G2 pilot dry-run ${e:0:8}" 0 python script/own_session_eval.py "$e" --dry-run
done
for e in 862ade13-53cd-4221-a3fa-dda8643641f2 f140a2ec-fd49-4814-994a-fe3476f14e66; do
    expect_rc "G2 manifest dry-run ${e:0:8}" 0 python script/own_session_eval.py "$e" --dry-run \
        --manifest "$MANIFEST" --pretrain-root "$PRE_ROOT" --out-root "$GATE_ROOT/own_session"
done
expect_rc "G2 neg: pilot dev target ${DEV_TARGET:0:8}" 3 \
    python script/own_session_eval.py "$DEV_TARGET" --dry-run
expect_rc "G2 neg: pilot held-out ${HELD_OUT:0:8}" 3 \
    python script/own_session_eval.py "$HELD_OUT" --dry-run
expect_rc "G2 neg: manifest held-out ${HELD_OUT:0:8}" 3 \
    python script/own_session_eval.py "$HELD_OUT" --dry-run \
        --manifest "$MANIFEST" --pretrain-root "$PRE_ROOT" --out-root "$GATE_ROOT/own_session"

# ------------------------------------------------------------ G3 reproduce
echo; echo "---------- G3: reproduce campaign eval from v1 checkpoints ----------"
for e in "${G3[@]}"; do
    expect_rc "G3 eval ${e:0:8}" 0 python script/own_session_eval.py "$e" \
        --manifest "$MANIFEST" --pretrain-root "$PRE_ROOT" --out-root "$GATE_ROOT/own_session"
    python - "$CAMPAIGN/$e" "$GATE_ROOT/own_session/$e" <<'PY'
import filecmp, json, sys
from pathlib import Path
import numpy as np
ref, new = map(Path, sys.argv[1:3])
bad, n = [], 0
for role in ("best", "last"):
    for task in ("co_smooth", "forward_pred"):
        for f in ("bps.npy", "r2.npy"):
            n += 1
            a, b = ref / role / task / f, new / role / task / f
            if not (b.is_file() and filecmp.cmp(a, b, shallow=False)):
                bad.append(f"{role}/{task}/{f}")
        n += 1
        a, b = ref / role / task / "trial_terms.npz", new / role / task / "trial_terms.npz"
        if not b.is_file():
            bad.append(f"{role}/{task}/trial_terms.npz missing"); continue
        za, zb = np.load(a), np.load(b)
        if sorted(za.files) != sorted(zb.files) or any(
                za[k].dtype != zb[k].dtype or za[k].shape != zb[k].shape
                or za[k].tobytes() != zb[k].tobytes() for k in za.files):
            bad.append(f"{role}/{task}/trial_terms.npz")
n += 1
sa = json.loads((ref / "own_session_summary.json").read_text())
sb_path = new / "own_session_summary.json"
if not sb_path.is_file() or json.loads(sb_path.read_text())["variants"] != sa["variants"]:
    bad.append("own_session_summary.json variants")
print(f"compare {ref.name[:8]}: {n - len(bad)}/{n} exact" + (f"; differ: {bad}" if bad else ""))
sys.exit(1 if bad else 0)
PY
    rc=$?; [ $rc -eq 0 ]; record "G3 exact vs campaign ${e:0:8}" $? "see compare line above"
done

# ------------------------------------------------------------ summary
echo; echo "================ merge gate summary ================"
nfail=0
for r in "${RESULTS[@]}"; do echo "$r"; [[ "$r" == "[FAIL]"* ]] && nfail=$((nfail+1)); done
echo "merge gate: $(( ${#RESULTS[@]} - nfail ))/${#RESULTS[@]} passed at ${SHA}"
conda deactivate
exit $(( nfail > 0 ))
