#!/bin/bash
# 1c-ii Test 2: train on an eid OUTSIDE the shipped 40-entry vocabulary.
#
#   sbatch step1_smoke_outside40.sh [EID] [EPOCHS]
#
# Before 1c this was a hard KeyError at ndt/models/ndt1.py:209 on the first
# forward pass, because the session embedding was indexed against a 40-entry
# dict built at module import. Re-running the old canonical eid does not
# exercise that at all, which is the point of this test.
#
# It isolates one thing:
#     arbitrary EID -> explicit vocabulary -> train -> save
#                   -> reload pickle -> reload state_dict -> eval
#
# Geometry-safe eid on purpose. The first candidate in the pool,
# 004d8fd5-..., has 1239 clusters against max_space_length: 668, so it could
# fail for a reason with nothing to do with the vocabulary and muddy the result.
# 032452e9 has 406 neurons and 287 trials: under the cap, close to the Step 0
# session's 207 trials. The 1239-cluster case belongs to the single-session
# track, where the question is what the max_space_length policy should be.

#SBATCH --job-name=step1-smoke
#SBATCH --output=../results_parity/step1-smoke-%j.out
#SBATCH --error=../results_parity/step1-smoke-%j.err
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

EID=${1:-032452e9-1886-449d-9c13-0f192572e19f}
EPOCHS=${2:-2}
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

OUT_ROOT="${RESULTS_ROOT}/step1_smoke/${EID}"
TRAIN_DIR="${OUT_ROOT}/ss_${EID}/train/${EID}"

echo "=================================================="
echo "1c-ii smoke test: eid outside the shipped 40"
echo "eid            : ${EID}"
echo "data root      : ${DATA_ROOT}"
echo "epochs         : ${EPOCHS}"
echo "git sha        : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "slurm nodelist : ${SLURM_JOB_NODELIST:-local}"
echo "=================================================="

# The whole point is that this eid is NOT in the shipped list.
if grep -qx "$EID" ndt/data/target_eids.txt; then
    echo "ERROR: $EID IS in target_eids.txt, so this proves nothing." >&2
    exit 2
fi
echo "confirmed: $EID is not in target_eids.txt"

if [ ! -d "$DATA_ROOT/$EID/data" ]; then
    echo "ERROR: no data at $DATA_ROOT/$EID/data" >&2
    exit 2
fi

rm -rf "$OUT_ROOT"; mkdir -p "$OUT_ROOT" || exit 2

# ---------------------------------------------------------------- 1. train
echo
echo "---------- [1/2] training on an outside-the-40 eid ----------"
python -m ndt.cli.train_single \
    --eid "$EID" \
    --data-root "$DATA_ROOT" \
    --base-path "$BASE_PATH" \
    --data-type "$DATA_TYPE" \
    --results-path "$OUT_ROOT" \
    --output-model-name "ss_${EID}" \
    --num-epochs "$EPOCHS"
rc=$?
if [ $rc -ne 0 ]; then
    echo "training failed (exit=$rc)" >&2
    exit $rc
fi

# ------------------------------------ 2. save / reload / eval the contract
echo
echo "---------- [2/2] reload both checkpoint forms + eval ----------"
python - "$TRAIN_DIR" "$EID" <<'PY'
import sys, json
from pathlib import Path
import torch

train_dir, eid = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(Path.cwd()))

from ndt.cli.train_single import build_config
from ndt.utils.config_utils import update_config
from ndt.compat import (resave_as_state_dict, load_pickled_model,
                        load_model_from_state_dict)
from ndt.vocab import VocabularyError

results = []
def check(name, fn, expect_error=None):
    try:
        out = fn()
    except Exception as e:
        ok = expect_error is not None and isinstance(e, expect_error)
        results.append((ok, name, f"{type(e).__name__}: {str(e)[:100]}"))
        return None
    ok = expect_error is None
    results.append((ok, name, "" if ok else f"expected {expect_error.__name__}"))
    return out

pkl = train_dir / "model_best.pt"
assert pkl.is_file(), f"missing {pkl}"

# The vocabulary the run actually used, straight off the trained checkpoint.
raw = torch.load(pkl, map_location="cpu", weights_only=False)
print("recorded in checkpoint:",
      json.dumps({k: raw.get(k) for k in
                  ("session_vocab", "vocabulary_ordering_rule",
                   "checkpoint_format_version")}))

check("pickle reload, checkpoint authoritative", lambda: load_pickled_model(pkl))
check("pickle reload, matching explicit vocab succeeds",
      lambda: load_pickled_model(pkl, session_vocab=[eid]))
check("pickle reload, wrong eid fails clearly",
      lambda: load_pickled_model(pkl, session_vocab=["deadbeef-0000-0000-0000-000000000000"]),
      expect_error=VocabularyError)

sdp = train_dir / "model_best.state_dict.pt"
resave_as_state_dict(pkl, sdp)
saved = torch.load(sdp, map_location="cpu", weights_only=False)
check("state_dict carries the vocabulary through resave",
      lambda: None if saved.get("session_vocab") == [eid] else
      (_ for _ in ()).throw(AssertionError(f"got {saved.get('session_vocab')}")))

cfg = build_config()
model = raw["model"]
meta = {"num_neurons": list(model.encoder.stitcher.stitcher_dict.keys()),
        "num_sessions": 1, "eids": {eid}}
n_neurons = [int(k) for k in meta["num_neurons"]]
cfg = update_config(cfg, {"num_neurons": n_neurons, "num_sessions": 1, "eids": {eid}})
kw = dict(cfg.method.model_kwargs)

m = check("state_dict reload: metadata FIRST, then construct, then load",
          lambda: load_model_from_state_dict(
              sdp, cfg.model, num_neurons=n_neurons, num_sessions=1,
              eids={eid}, **kw))
check("state_dict reload, wrong eid fails clearly",
      lambda: load_model_from_state_dict(
          sdp, cfg.model, session_vocab=["deadbeef-0000-0000-0000-000000000000"],
          num_neurons=n_neurons, num_sessions=1, eids={eid}, **kw),
      expect_error=VocabularyError)

# The reloaded weights must actually equal the originals, not merely load.
if m is not None:
    rl = m[0].state_dict()
    orig = model.state_dict()
    worst = max((rl[k].float() - orig[k].float()).abs().max().item() for k in orig)
    check(f"reloaded state_dict is numerically identical (max|d|={worst:.2e})",
          lambda: None if worst == 0.0 else
          (_ for _ in ()).throw(AssertionError(f"max|d|={worst}")))
    # And it must run a forward pass on this eid without a KeyError.
    emb = m[0].encoder.embedder
    check("forward-path lookup resolves this eid to a row",
          lambda: emb.eid_to_indx[eid])

print("=" * 66)
npass = sum(1 for ok, _, _ in results if ok)
for ok, name, detail in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok and detail:
        print(f"         {detail}")
print("=" * 66)
print(f"{npass}/{len(results)} checks passed")
sys.exit(0 if npass == len(results) else 1)
PY
rc=$?

echo
echo "smoke test finished, exit=${rc}"
conda deactivate
exit $rc
