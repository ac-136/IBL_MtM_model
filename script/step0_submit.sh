#!/bin/bash
# Step 0: submit the reference-establishing captures, one gate at a time.
#
#   ./step0_submit.sh b     gate B: two identical 2-epoch captures (DO THIS FIRST)
#   ./step0_submit.sh d     gate D: one 2-epoch capture after 3a has landed
#   ./step0_submit.sh e     the canonical 5-epoch reference
#   ./step0_submit.sh --dry b    print the sbatch lines without submitting
#
# The gates are deliberately not chained. Each one has to be *read* before the
# next is submitted: a failure at B or D is stop-the-line, not a retry, because
# it would mean the machinery is not deterministic and every later comparison
# is meaningless.

set -u
cd "$(dirname "$0")"

DRY=0
if [ "${1:-}" = "--dry" ]; then DRY=1; shift; fi
GATE="${1:-}"

RESULTS_ROOT="${RESULTS_ROOT:-/work/nvme/beml/ac136/IBL_MTM/results}"
STEP0="${RESULTS_ROOT}/step0"
mkdir -p ../results_parity
LOGDIR="$(cd .. && pwd)/results_parity"

sub() {  # sub <label> <args...>
    local label="$1"; shift
    local args=(--parsable
                --job-name="$label"
                --output="${LOGDIR}/${label}-%j.out"
                --error="${LOGDIR}/${label}-%j.err")
    if [ "$DRY" = "1" ]; then
        echo "      sbatch ${args[*]} step0_capture.sh $*" >&2
        echo "dry"
    else
        sbatch "${args[@]}" step0_capture.sh "$@"
    fi
}

echo "results root : $RESULTS_ROOT"
echo "step 0 root  : $STEP0"
echo "git sha      : $(git -C .. rev-parse HEAD 2>/dev/null || echo unknown)"
echo

case "$GATE" in
  b)
    echo "-- gate B: pre-logging reproducibility, two identical 2-epoch captures"
    a=$(sub step0-repro_a repro_a 2); echo "     repro_a -> job $a"
    b=$(sub step0-repro_b repro_b 2); echo "     repro_b -> job $b"
    cat <<EOF

Both captures run the same eid, seed and epoch count. They may land on
different nodes, which is the more honest test; each capture records its own
SLURM_JOB_NODELIST in capture_meta.json.

When both finish:

  python script/compare_runs.py $STEP0/repro_a $STEP0/repro_b

Required result: every artifact at exactly 0.0, and 0 unreadable.
Anything else is stop-the-line. Do NOT submit gate D until this is clean.
EOF
    ;;
  d)
    echo "-- gate D: logging parity, one 2-epoch capture after 3a"
    a=$(sub step0-post3a post3a 2); echo "     post3a -> job $a"
    cat <<EOF

When it finishes, compare it against a gate-B capture. The 3a keys are new, so
they are allowed on the B side only; everything that existed before must still
be bit-identical:

  python script/compare_runs.py $STEP0/repro_a $STEP0/post3a \\
      --allow-new-keys lr eval_metric best_epoch

Required result: every pre-existing artifact at exactly 0.0. A key that
DISAPPEARED is still a failure, and --allow-new-keys will not mask it.
EOF
    ;;
  e)
    echo "-- gate E: the canonical 5-epoch reference"
    a=$(sub step0-reference reference 5); echo "     reference -> job $a"
    cat <<EOF

This is the artifact everything downstream is compared against. It is not
itself gated; its trustworthiness comes from B and D having passed.

Optionally submit a second copy to confirm determinism at 5 epochs too:

  ./step0_submit.sh e2

Then freeze:  git tag -a baseline_pre_tier1
EOF
    ;;
  e2)
    echo "-- optional: a second copy of the 5-epoch reference"
    a=$(sub step0-reference_b reference_b 5); echo "     reference_b -> job $a"
    echo
    echo "  python script/compare_runs.py $STEP0/reference $STEP0/reference_b"
    ;;
  *)
    echo "usage: ./step0_submit.sh [--dry] {b|d|e|e2}" >&2
    exit 2 ;;
esac

[ "$DRY" = "1" ] && echo && echo "(dry run, nothing was actually submitted)"
exit 0
