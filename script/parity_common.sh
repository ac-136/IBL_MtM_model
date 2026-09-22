#!/bin/bash
# Shared settings for the Stage-1b parity runs. Sourced by the parity_*.sh jobs.
#
# One parameterised script per job type, dispatching on TREE, so that the two
# sides of the comparison differ ONLY in which module is invoked. Any other
# difference between them would be a confound.

set -u

BASE_PATH="${BASE_PATH:-/work/nvme/beml/ac136/IBL_MTM/data}"
DATA_TYPE="${DATA_TYPE:-processed_mtm}"
RESULTS_ROOT="${RESULTS_ROOT:-/work/nvme/beml/ac136/IBL_MTM/results}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"

# Deliberately NOT passing --use_dummy. utils.utils.dummy_load calls
# torch.rand(80000, 80000) on a background thread, which advances the global RNG
# at an unpredictable point relative to training and allocates 25.6 GB. It only
# exists to keep the GPU busy for accounting, and it would make these runs
# irreproducible.

# parity_tree_setup TREE  ->  sets PY_CMD (array) and OUT_ROOT
# kind is one of: single, multi, finetune
parity_tree_setup() {
    local tree="$1" kind="$2"
    case "$tree" in
        src|src_rerun)
            case "$kind" in
                single)   PY_CMD=(python src/train_sessions.py) ;;
                multi)    PY_CMD=(python src/multi_train_session.py) ;;
                finetune) PY_CMD=(python src/finetune_eval_multi_session_just_spikes.py) ;;
                *) echo "unknown kind: $kind" >&2; exit 2 ;;
            esac
            ;;
        ndt)
            case "$kind" in
                single)   PY_CMD=(python -m ndt.cli.train_single) ;;
                multi)    PY_CMD=(python -m ndt.cli.train_multi) ;;
                finetune) PY_CMD=(python -m ndt.cli.finetune_eval) ;;
                *) echo "unknown kind: $kind" >&2; exit 2 ;;
            esac
            ;;
        *) echo "unknown tree: $tree (expected src, ndt or src_rerun)" >&2; exit 2 ;;
    esac
    OUT_ROOT="${RESULTS_ROOT}/parity_${tree}"
}

parity_banner() {
    echo "=================================================="
    echo "tree        : ${TREE}"
    echo "kind        : ${KIND}"
    echo "command     : ${PY_CMD[*]}"
    echo "out root    : ${OUT_ROOT}"
    echo "base path   : ${BASE_PATH}"
    echo "data type   : ${DATA_TYPE}"
    echo "num epochs  : ${NUM_EPOCHS}"
    echo "slurm job   : ${SLURM_JOB_ID:-local}"
    echo "=================================================="
}
