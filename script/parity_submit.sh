#!/bin/bash
# Stage-1b parity: submit the whole 45-job matrix.
#
#   ./parity_submit.sh control    3 control reruns only (DO THIS FIRST)
#   ./parity_submit.sh train      training jobs for src and ndt
#   ./parity_submit.sh finetune   finetune+eval, assumes checkpoints exist
#   ./parity_submit.sh all        train then finetune, chained with afterok
#   ./parity_submit.sh --dry all  print the sbatch lines without submitting
#
# Run the control phase first and compare parity_src against parity_src_rerun.
# That delta is the GPU noise floor, and it is the threshold every later
# comparison is judged against. Without it an src-vs-ndt difference cannot be
# attributed to the port.

set -u
cd "$(dirname "$0")"

DRY=0
if [ "${1:-}" = "--dry" ]; then DRY=1; shift; fi
PHASE="${1:-all}"

RESULTS_ROOT="${RESULTS_ROOT:-/work/nvme/beml/ac136/IBL_MTM/results}"
mkdir -p ../results_parity

mapfile -t PARITY_EIDS < <(grep -v '^[[:space:]]*$' ../parity_eids.txt)
mapfile -t ALL_TEST    < <(grep -v '^[[:space:]]*$' ../data/test_re_eids.txt)
TEST_EIDS=("${ALL_TEST[0]}" "${ALL_TEST[1]}")   # two is enough; see the plan

FIRST2=("${PARITY_EIDS[0]}" "${PARITY_EIDS[1]}")

declare -a TRAIN_IDS=()
declare -a FT_IDS=()
declare -a CTRL_IDS=()

# `sub` is always called as $(sub ...), i.e. in a subshell, so it cannot keep a
# counter for the parent. Job totals are derived from the arrays below, which
# the callers append to in the parent shell.
LOGDIR="$(cd .. && pwd)/results_parity"

sub() {  # sub <depend-csv-or-empty> <label> <script> <args...>
    local dep="$1"; shift
    local label="$1"; shift
    # #SBATCH directives are parsed before the script body runs, so the tree
    # cannot reach them. Overriding on the command line makes every log file
    # name say which tree produced it.
    local args=(--parsable
                --job-name="$label"
                --output="${LOGDIR}/${label}-%j.out"
                --error="${LOGDIR}/${label}-%j.err")
    [ -n "$dep" ] && args+=(--dependency=afterok:"$dep")
    if [ "$DRY" = "1" ]; then
        # the sbatch line goes to stderr so it does not pollute the job id that
        # the caller captures with $(...)
        echo "      sbatch ${args[*]} $*" >&2
        echo "dry"
    else
        sbatch "${args[@]}" "$@"
    fi
}

submit_training() {  # submit_training <tree>
    local tree="$1" id
    echo "-- training, tree=$tree"
    for e in "${PARITY_EIDS[@]}"; do
        id=$(sub "" "par-${tree}-ss-${e:0:8}" parity_train_single.sh "$tree" "$e"); TRAIN_IDS+=("$id")
        echo "     single $e -> job $id"
    done
    id=$(sub "" "par-${tree}-ms_first2" parity_train_multi.sh "$tree" ms_first2 "${FIRST2[@]}"); TRAIN_IDS+=("$id")
    echo "     multi ms_first2 -> job $id"
    id=$(sub "" "par-${tree}-ms_all5" parity_train_multi.sh "$tree" ms_all5 "${PARITY_EIDS[@]}"); TRAIN_IDS+=("$id")
    echo "     multi ms_all5 -> job $id"
}

submit_finetune() {  # submit_finetune <tree> <dep-csv>
    # Separate statements on purpose: bash expands every word of a `local`
    # command before performing any of its assignments, so `local a=$1 b=$a`
    # sees an unbound `a` and dies under `set -u`.
    local tree="$1"
    local dep="$2"
    local root="$RESULTS_ROOT/parity_$tree"
    local id
    echo "-- finetune+eval, tree=$tree"
    for e in "${PARITY_EIDS[@]}"; do
        for t in "${TEST_EIDS[@]}"; do
            id=$(sub "$dep" "par-${tree}-fte-ss${e:0:8}-on-${t:0:8}" \
                     parity_finetune_eval.sh "$tree" "ss_${e}" \
                     "$root/ss_${e}/train/${e}/model_best.pt" "$t")
            FT_IDS+=("$id")
            echo "     ss_${e} on $t -> job $id"
        done
    done
    for m in ms_first2 ms_all5; do
        for t in "${TEST_EIDS[@]}"; do
            id=$(sub "$dep" "par-${tree}-fte-${m}-on-${t:0:8}" \
                     parity_finetune_eval.sh "$tree" "$m" \
                     "$root/train/$m/model_best.pt" "$t")
            FT_IDS+=("$id")
            echo "     $m on $t -> job $id"
        done
    done
}

echo "phase        : $PHASE"
echo "parity eids  : ${PARITY_EIDS[*]}"
echo "test eids    : ${TEST_EIDS[*]}"
echo "results root : $RESULTS_ROOT"
echo

case "$PHASE" in
  control)
    # One rerun per job type, all against tree=src_rerun, to measure the GPU
    # noise floor. The finetune control deliberately consumes the checkpoint
    # produced by tree=src, not by src_rerun, so that it isolates
    # nondeterminism in the finetune stage rather than compounding it with
    # nondeterminism inherited from a different pretrained model.
    echo "-- control reruns (src_rerun)"
    a=$(sub "" "par-src_rerun-ss-${PARITY_EIDS[0]:0:8}" parity_train_single.sh src_rerun "${PARITY_EIDS[0]}")
    CTRL_IDS+=("$a"); echo "     single ${PARITY_EIDS[0]} -> job $a"
    b=$(sub "" "par-src_rerun-ms_first2" parity_train_multi.sh src_rerun ms_first2 "${FIRST2[@]}")
    CTRL_IDS+=("$b"); echo "     multi ms_first2 -> job $b"
    c=$(sub "" "par-src_rerun-fte-ss${PARITY_EIDS[0]:0:8}-on-${TEST_EIDS[0]:0:8}" \
             parity_finetune_eval.sh src_rerun "ss_${PARITY_EIDS[0]}" \
             "$RESULTS_ROOT/parity_src/ss_${PARITY_EIDS[0]}/train/${PARITY_EIDS[0]}/model_best.pt" \
             "${TEST_EIDS[0]}")
    CTRL_IDS+=("$c"); echo "     finetune ss_${PARITY_EIDS[0]} on ${TEST_EIDS[0]} -> job $c"
    echo
    echo "NOTE: the finetune control needs parity_src's checkpoint to exist first."
    echo "      Run './parity_submit.sh train' before this if it does not."
    ;;
  train)
    submit_training src
    submit_training ndt
    ;;
  finetune)
    submit_finetune src ""
    submit_finetune ndt ""
    ;;
  all)
    submit_training src
    submit_training ndt
    dep=$(IFS=:; echo "${TRAIN_IDS[*]}")
    echo
    submit_finetune src "$dep"
    submit_finetune ndt "$dep"
    ;;
  *)
    echo "unknown phase: $PHASE" >&2; exit 2 ;;
esac

echo
TOTAL=$(( ${#TRAIN_IDS[@]} + ${#FT_IDS[@]} + ${#CTRL_IDS[@]} ))
echo "submitted $TOTAL job(s)  "\
     "(train ${#TRAIN_IDS[@]}, finetune ${#FT_IDS[@]}, control ${#CTRL_IDS[@]})"
[ "$DRY" = "1" ] && echo "(dry run, nothing was actually submitted)"
echo
echo "when they finish:"
echo "  python script/compare_runs.py $RESULTS_ROOT/parity_src $RESULTS_ROOT/parity_src_rerun   # noise floor"
echo "  python script/compare_runs.py $RESULTS_ROOT/parity_src $RESULTS_ROOT/parity_ndt         # the port"
