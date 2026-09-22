#!/bin/bash
# Stage 1b: one real source -> unseen development-target transfer.
#
#   sbatch step1_transfer_smoke.sh [EPOCHS]
#
# Everything proven so far trains and evaluates on ONE session. This is the
# first run that takes a model pretrained on session S and finetunes it onto a
# different, unseen session H -- the core operation of the whole scaling study.
#
#   S = 032452e9  (406 clusters, outside the shipped 40, already on disk from 1c-ii)
#   H = 572a95d1  (602 clusters, rs DEVELOPMENT target)
#
# H is a development target, never one of the four official held-out sessions:
# those stay sealed until the recipe is frozen.
#
# Mechanics test. The source's hyperparameters do not matter.

#SBATCH --job-name=step1-transfer
#SBATCH --output=../results_parity/step1-transfer-%j.out
#SBATCH --error=../results_parity/step1-transfer-%j.err
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

EPOCHS=${1:-2}
SRC_EID=032452e9-1886-449d-9c13-0f192572e19f
TGT_EID=572a95d1-39ca-42e1-8424-5c9ffcb2df87
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

SRC_CKPT="${RESULTS_ROOT}/step1_smoke/${SRC_EID}/ss_${SRC_EID}/train/${SRC_EID}/model_best.pt"
OUT_ROOT="${RESULTS_ROOT}/step1_transfer"

echo "=================================================="
echo "Stage 1b: transfer S -> H, S != H"
echo "source S       : ${SRC_EID}"
echo "target H       : ${TGT_EID}  (development target)"
echo "epochs         : ${EPOCHS}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "slurm nodelist : ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

[ -f "$SRC_CKPT" ] || { echo "ERROR: source checkpoint missing: $SRC_CKPT" >&2; exit 2; }
[ -d "$DATA_ROOT/$TGT_EID/data" ] || { echo "ERROR: no data for $TGT_EID" >&2; exit 2; }
if [ "$SRC_EID" = "$TGT_EID" ]; then
    echo "ERROR: source and target are the same session; this proves nothing." >&2
    exit 2
fi

rm -rf "$OUT_ROOT"; mkdir -p "$OUT_ROOT" || exit 2

# ---------------------------------------------------------------------------
# [1/2] The load contract, checked before any optimizer step.
# ---------------------------------------------------------------------------
# Reconstructs exactly what finetune_eval does -- builds the target model with
# vocabulary [H], then loads the filtered source -- using the same production
# functions, so what is asserted is the real logic rather than a paraphrase.
# Snapshotting H before the load is the only way to assert "untouched"; once
# training starts, that state is gone.
echo
echo "---------- [1/2] transfer contract, pre-optimizer-step ----------"
python - "$SRC_CKPT" "$TGT_EID" "$DATA_ROOT" <<'PY'
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from ndt.cli.finetune_eval import build_parser
from ndt.utils.config_utils import config_from_kwargs, update_config
from ndt.utils.dataset_utils import load_ibl_dataset_locally
from ndt.utils.utils import set_seed
from ndt.models.ndt1 import NDT1
from ndt.paths import config_path
from ndt.vocab import (vocab_for_single_session, filter_transfer_state_dict,
                       classify_state_dict_keys)

src_path, tgt_eid, data_root = sys.argv[1:4]
results = []
def check(name, ok, detail=""):
    results.append((bool(ok), name, detail))

# --- source
src_ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
src_sd = {k: v.clone() for k, v in src_ckpt["model"].state_dict().items()}
src_vocab = src_ckpt.get("session_vocab")
print(f"source vocabulary: {src_vocab}")

# --- target model, built the way finetune_eval builds it
cfg = config_from_kwargs({"model": "include:" + config_path("ndt1_stitching")})
cfg = update_config(config_path("finetune_sessions_trainer"), cfg)
set_seed(cfg.seed)
_, _, _, meta = load_ibl_dataset_locally(
    eid=None, num_sessions=cfg.data.num_sessions, train_session_eid=[tgt_eid],
    test_session_eid=cfg.data.test_session_eid, data_type="processed_mtm",
    base_path="/work/nvme/beml/ac136/IBL_MTM/data", data_root=data_root)
cfg = update_config(cfg, meta)
tgt_vocab, rule = vocab_for_single_session(tgt_eid)
model = NDT1(cfg.model, **cfg.method.model_kwargs, **meta, session_vocab=tgt_vocab)

# state_dict() hands back live references; clone or the load mutates the snapshot
# too and "untouched" becomes trivially true.
pre = {k: v.clone() for k, v in model.state_dict().items()}

shared, specific = classify_state_dict_keys(pre.keys())

check("2. target vocabulary is [H] and resolves to row 0",
      list(model.encoder.embedder.eid_lookup) == [tgt_eid.removesuffix("_aligned")]
      and model.encoder.embedder.eid_to_indx[tgt_eid.removesuffix("_aligned")] == 0,
      f"vocab={model.encoder.embedder.eid_lookup}")

check("5. the two prefix sets exhaustively partition the checkpoint",
      len(shared) + len(specific) == len(pre),
      f"{len(shared)} shared + {len(specific)} specific == {len(pre)}")

# S and H have different cluster counts here, so the session-specific tensors
# differ in SHAPE; comparable-and-equal is the thing to rule out.
comparable = [k for k in specific if k in src_sd and src_sd[k].shape == pre[k].shape]
nonconst = [k for k in comparable if not torch.equal(src_sd[k], pre[k])]
check("6. non-vacuity: S and H are distinguishable on the session-specific side",
      len(comparable) == 0 or len(nonconst) > 0,
      f"{len(comparable)} shape-comparable, {len(nonconst)} of them differ pre-load"
      + ("  (differing cluster counts -> no key overlap, so the load cannot "
         "confuse them by name)" if not comparable else ""))

filtered = filter_transfer_state_dict(src_sd)
missing, unexpected = model.load_state_dict(filtered, strict=False)
check("7a. no unexpected keys from the filtered source", not unexpected, str(sorted(unexpected)[:4]))
check("7b. exactly the session-specific keys left un-loaded",
      set(missing) == set(specific), f"missing={len(missing)} specific={len(specific)}")

post = model.state_dict()
bad_shared = [k for k in shared if not torch.equal(post[k], src_sd[k])]
check(f"4. SHARED: all {len(shared)} tensors post_load == source exactly",
      not bad_shared, f"{len(bad_shared)} failed: {sorted(bad_shared)[:4]}")

bad_spec = [k for k in specific if not torch.equal(post[k], pre[k])]
check(f"3. SESSION-SPECIFIC: all {len(specific)} tensors post_load == H's pre-load "
      f"state exactly", not bad_spec, f"{len(bad_spec)} modified: {sorted(bad_spec)[:4]}")

print("=" * 70)
npass = sum(1 for ok, _, _ in results if ok)
for ok, name, detail in sorted(results, key=lambda r: r[1]):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
print("=" * 70)
print(f"{npass}/{len(results)} contract checks passed")
sys.exit(0 if npass == len(results) else 1)
PY
rc=$?
[ $rc -ne 0 ] && { echo "transfer contract FAILED; not running the end-to-end leg" >&2; exit $rc; }

# ---------------------------------------------------------------------------
# [2/2] The real thing: finetune + eval S -> H end to end.
# ---------------------------------------------------------------------------
echo
echo "---------- [2/2] end-to-end finetune + eval, S -> H ----------"
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
echo "stage 1b finished, exit=${rc}"
[ $rc -eq 0 ] && find "$OUT_ROOT" -name '*.npy' -o -name '*.npz' -o -name 'run_summary.json' \
    | sed "s|^${OUT_ROOT}/|  |" | sort
echo "=================================================="

conda deactivate
exit $rc
