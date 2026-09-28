#!/usr/bin/env python
"""Raw-spike linear decoders and the chance/null for one held-in session.

    python script/decode_baseline.py EID

Same data access as the frozen-NDT1 probe (``behavior_decode.py``): the same
MtM splits, the same ``decode_components.prepare_targets`` output (targets,
masks, train-fitted scaling, train-decided drops). Inputs are the raw binned
spikes, flattened to (trials, bins x neurons), unnormalised as the encoder
sees them.

* choice, block: L2 LogisticRegression, C grid of ndt's behavior_decoding;
* wheel speed, whisker ME: Ridge to all 100 bins, alpha grid of the same;
* each grid value is fit on TRAIN only and chosen on VAL (val cross-entropy,
  or val MSE on scaled targets), with no refit on train+val and no CV folds;
  the chosen model is evaluated once on TEST. (ndt's behavior_decoding fits
  GridSearchCV on train+val instead -- a recorded departure.)
* chance/null from train only: the majority class, the per-bin mean trace.

Ridge cannot mask single entries, so regressors use the trials whose 100 bins
are all finite (the same rows every metric uses).
Exit codes: 0 ok, 2 output exists, 3 audit failed.
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

import decode_components as dc  # noqa: E402
from behavior_decode import AuditError, DATA_ROOT, RESULTS, audit, read_behaviour  # noqa: E402

OUT_ROOT = RESULTS / "decoding" / "baseline"
# ndt/utils/eval_utils.py behavior_decoding, L760 (C) and L770 (alpha).
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
ALPHA_GRID = [0, 1e1, 1e2, 1e3, 1e4]


def binned_spikes(eid, data_root, split):
    """(n_trials, T * N) dense float32 counts from the sparse parquet columns."""
    t = pq.read_table(os.path.join(data_root, eid, "data", f"{split}.parquet"),
                      columns=["spikes_sparse_data", "spikes_sparse_indices",
                               "spikes_sparse_indptr", "spikes_sparse_shape"]).to_pydict()
    rows = []
    for d, i, p, s in zip(t["spikes_sparse_data"], t["spikes_sparse_indices"],
                          t["spikes_sparse_indptr"], t["spikes_sparse_shape"]):
        rows.append(sp.csr_matrix((d, i, p), shape=tuple(s)).toarray().reshape(-1))
    return np.asarray(rows, dtype=np.float32)


def fit_static(name, X, tg, seed):
    rows = {s: tg[s][name]["mask"] for s in dc.SPLITS}
    y = {s: tg[s][name]["y"] for s in dc.SPLITS}
    classes = np.arange(dc.N_CLASSES[name])
    grid = []
    for C in C_GRID:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = LogisticRegression(C=C, max_iter=1000, random_state=seed)
            m.fit(X["train"][rows["train"]], y["train"][rows["train"]])
        proba = np.zeros((rows["val"].sum(), len(classes)))
        proba[:, m.classes_] = m.predict_proba(X["val"][rows["val"]])
        grid.append({"C": C, "val_ce": float(log_loss(y["val"][rows["val"]], proba, labels=classes)),
                     "n_warnings": len(w), "model": m})
    best = min(grid, key=lambda g: g["val_ce"])
    pred = best["model"].predict(X["test"])
    return pred, {"grid": [{k: v for k, v in g.items() if k != "model"} for g in grid],
                  "chosen": {"C": best["C"], "val_ce": best["val_ce"]}}


def fit_dynamic(name, X, tg):
    rows = {s: dc.evaluable(name, tg[s][name]) for s in dc.SPLITS}
    y = {s: tg[s][name]["y"] for s in dc.SPLITS}
    grid = []
    for alpha in ALPHA_GRID:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = Ridge(alpha=alpha)
            m.fit(X["train"][rows["train"]], y["train"][rows["train"]])
        pv = m.predict(X["val"][rows["val"]])
        grid.append({"alpha": alpha,
                     "val_mse": float(np.mean((pv - y["val"][rows["val"]]) ** 2)),
                     "n_warnings": len(w), "model": m})
    best = min(grid, key=lambda g: g["val_mse"])
    pred = best["model"].predict(X["test"])
    return pred, {"grid": [{k: v for k, v in g.items() if k != "model"} for g in grid],
                  "chosen": {"alpha": best["alpha"], "val_mse": best["val_mse"]},
                  "n_train_rows": int(rows["train"].sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    eid = args.eid.removesuffix("_aligned")

    log = []
    print(f"=== raw-spike baseline + chance: {eid} ===")
    try:
        audit(eid, args.data_root, log)
    except AuditError as e:
        print(f"AUDIT FAILED: {e}"); return 3

    out = Path(args.out_root) / eid
    if out.exists() and any(p.name != "capture_meta.json" for p in out.iterdir()):
        print(f"ERROR: {out} already holds results"); return 2
    out.mkdir(parents=True, exist_ok=True)

    raw = read_behaviour(eid, args.data_root)
    prepared = dc.prepare_targets(raw)
    tg = prepared["targets"]
    X = {s: binned_spikes(eid, args.data_root, s) for s in dc.SPLITS}
    for s in dc.SPLITS:
        assert X[s].shape[0] == len(raw[s]["choice"]), f"{s}: spikes/behaviour row mismatch"
    cond = np.stack([raw["test"][c] for c in dc.PSTH_CONDITIONS], axis=1)

    results = {}
    for name in dc.active_targets(prepared):
        if name in dc.STATIC:
            pred, sel = fit_static(name, X, tg, args.seed)
        else:
            pred, sel = fit_dynamic(name, X, tg)
        chance_pred, chance_info = dc.chance_predictions(name, tg["train"][name], len(pred))
        results[name] = {
            "linear": {**dc.metrics_for(name, tg["test"][name], pred, cond),
                       "ci95": dc.bootstrap_ci(name, tg["test"][name], pred, cond),
                       "selection": sel},
            "chance": {**dc.metrics_for(name, tg["test"][name], chance_pred, cond),
                       **({"majority_class": chance_info["majority_class"]}
                          if "majority_class" in chance_info else {})},
        }
        lin, ch = results[name]["linear"], results[name]["chance"]
        keys = [k for k, v in lin.items() if isinstance(v, float)]
        print(f"  {name:22s} linear " + " ".join(f"{k} {lin[k]:+.4f}" for k in keys)
              + " | chance " + " ".join(f"{k} {ch.get(k, float('nan')):+.4f}" for k in keys)
              + f"  [{sel['chosen']}]")

    (out / "metrics.json").write_text(json.dumps({
        "eid": eid, "data_root": args.data_root, "seed": args.seed,
        "inputs": "raw binned spikes, flattened (T x N), unnormalised",
        "split_roles": {"train": "fit, scaling, drops, chance", "val": "C/alpha selection",
                        "test": "metrics, once"},
        "targets": prepared["info"], "results": results}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
