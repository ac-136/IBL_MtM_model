#!/usr/bin/env python
"""Draw the single-session learnability sample, once, before any training.

    python script/select_learnability_sessions.py [--out script/learnability_manifest.json]

bwm: a 3x3 grid on tertiles of train-trial count and cluster count, both computed
over ALL 303 bwm held-in sessions; 4 sessions drawn per cell with
numpy.random.default_rng(42). No outlier exclusion.

rs: all measurable held-in rs sessions, i.e. val >= 30 and test >= 30 trials --
a predeclared MEASURABILITY rule (own-session bps needs test trials), the same
one the pilot used. Exclusions are recorded by eid with the reason.

Deterministic: candidates are sorted by eid before drawing and cells are visited
in a fixed order, so the sample does not depend on manifest file order. Held-in
only; the official held-out set is read solely to assert it is disjoint.

Pilot sources that happen to be drawn reuse their pilot pretrains; pilot sources
that are NOT drawn are listed separately as reference-only and never enter the
primary analysis population.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
MANIFESTS = Path("/projects/beml/ac136/LFP_FM/data/scaling_manifests")
DATA_ROOT = Path("/work/nvme/beml/ac136/LFP_FM/data/datasets")
PILOT_PRETRAIN = Path("/work/nvme/beml/ac136/IBL_MTM/results/pilot/pretrain")
PILOT_SOURCES = ["5b49aca6-a6f4-4075-931a-617ad64c219c", "71e55bfe-5a3a-4cba-bdc7-f085140d798e",
                 "862ade13-53cd-4221-a3fa-dda8643641f2", "b182b754-3c3e-4942-8144-6ee790926b58",
                 "f56194bc-8215-4ae8-bc6a-89781ad8e050"]
SEED = 42
PER_CELL = 4
MIN_EVAL_TRIALS = 30
LEVELS = ("low", "mid", "high")
# Stage-2 measured pretraining rate (s/epoch/trial) against cluster count.
RATE_CLUSTERS = [258, 736, 1069, 1255, 1899]
RATE_S = [0.0285, 0.0526, 0.0551, 0.0575, 0.0587]
EPOCHS = 300


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def session_facts(eid):
    d = DATA_ROOT / eid / "data"
    n = {s: pq.ParquetFile(d / f"{s}.parquet").metadata.num_rows for s in ("train", "val", "test")}
    shp = (pq.ParquetFile(d / "train.parquet")
           .read_row_group(0, columns=["spikes_sparse_shape"]).slice(0, 1)
           .to_pydict()["spikes_sparse_shape"][0])
    return {"clusters": int(shp[1]), "n_train": n["train"], "n_val": n["val"], "n_test": n["test"]}


def est_gpu_h(clusters, n_train):
    return float(np.interp(clusters, RATE_CLUSTERS, RATE_S)) * EPOCHS * n_train / 3600


def pilot_reuse(eid):
    d = PILOT_PRETRAIN / f"src_{eid}" / "train" / eid
    need = ["model_best.pt", "model_last.pt", "model_best.state_dict.pt",
            "model_last.state_dict.pt", "run_summary.json", "loss/loss_history.npz"]
    return str(d) if all((d / f).is_file() for f in need) else None


def tertile(x, edges):
    # side="right": a value equal to an edge goes to the HIGHER tertile.
    return int(np.searchsorted(edges, x, side="right"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "script" / "learnability_manifest.json"))
    args = ap.parse_args()

    heldin = {c: read_list(MANIFESTS / c / "heldin_eids.txt") for c in ("bwm", "rs")}
    heldout = {c: read_list(MANIFESTS / c / "heldout_eids.txt") for c in ("bwm", "rs")}
    n_in = sum(len(v) for v in heldin.values()); n_out = sum(len(v) for v in heldout.values())
    # Non-vacuity: an empty or wrong read would make the disjointness check pass
    # without testing anything.
    assert n_in == 321 and n_out == 80, f"expected 321 held-in / 80 held-out, read {n_in}/{n_out}"
    sealed = set(heldout["bwm"]) | set(heldout["rs"])

    facts = {e: session_facts(e) for c in ("bwm", "rs") for e in heldin[c]}

    # ---------------- bwm: 3x3 grid, 4 per cell
    bwm = sorted(heldin["bwm"])
    tr = np.array([facts[e]["n_train"] for e in bwm])
    cl = np.array([facts[e]["clusters"] for e in bwm])
    tr_edges = np.quantile(tr, [1 / 3, 2 / 3])
    cl_edges = np.quantile(cl, [1 / 3, 2 / 3])
    cells = {}
    for e, t, c in zip(bwm, tr, cl):
        cells.setdefault((tertile(t, tr_edges), tertile(c, cl_edges)), []).append(e)

    rng = np.random.default_rng(SEED)
    sessions, cell_sizes = [], {}
    for ti in range(3):                      # fixed visiting order: trials outer,
        for ci in range(3):                  # clusters inner
            cand = sorted(cells.get((ti, ci), []))
            key = f"trials_{LEVELS[ti]}/clusters_{LEVELS[ci]}"
            cell_sizes[key] = len(cand)
            assert len(cand) >= PER_CELL, f"cell {key} has only {len(cand)} candidates"
            for j in sorted(rng.choice(len(cand), PER_CELL, replace=False)):
                sessions.append({"eid": cand[j], "cohort": "bwm", "cell": key})

    # ---------------- rs: every measurable session
    rs_excluded = []
    for e in sorted(heldin["rs"]):
        f = facts[e]
        if f["n_val"] >= MIN_EVAL_TRIALS and f["n_test"] >= MIN_EVAL_TRIALS:
            sessions.append({"eid": e, "cohort": "rs", "cell": "census"})
        else:
            rs_excluded.append({"eid": e, **f,
                                "reason": f"measurability: needs val>={MIN_EVAL_TRIALS} and "
                                          f"test>={MIN_EVAL_TRIALS}"})

    for s in sessions:
        s.update(facts[s["eid"]])
        s["reuse_pilot_pretrain"] = pilot_reuse(s["eid"])
        s["est_pretrain_gpu_h"] = (0.0 if s["reuse_pilot_pretrain"]
                                   else round(est_gpu_h(s["clusters"], s["n_train"]), 3))

    eids = [s["eid"] for s in sessions]
    assert len(eids) == len(set(eids)), "duplicate eid in sample"
    assert not (set(eids) & sealed), "an official held-out eid is in the sample"
    n_bwm = sum(s["cohort"] == "bwm" for s in sessions)
    n_rs = sum(s["cohort"] == "rs" for s in sessions)
    assert n_bwm == 9 * PER_CELL, n_bwm

    # Pilot sources not drawn: reference only, never in the primary population.
    reference = []
    for e in PILOT_SOURCES:
        if e in eids:
            continue
        coh = "bwm" if e in heldin["bwm"] else "rs"
        reference.append({"eid": e, "cohort": coh, **facts[e],
                          "pilot_pretrain": pilot_reuse(e),
                          "role": "pilot reference only; excluded from every statistic"})

    manifest = {
        "purpose": "single-session learnability characterisation (descriptive; held-in only)",
        "primary_population": "the 'sessions' list only (36 bwm + measurable rs)",
        "rule": {
            "bwm": f"3x3 tertile grid on n_train x clusters over all {len(bwm)} bwm held-in; "
                   f"{PER_CELL}/cell via numpy.random.default_rng({SEED}); candidates sorted by eid; "
                   "cells visited trials-outer, clusters-inner; value == edge -> higher tertile",
            "rs": f"all measurable held-in rs: val>={MIN_EVAL_TRIALS} and test>={MIN_EVAL_TRIALS}",
        },
        "seed": SEED,
        "source_manifests_sha256": {f"{c}/{k}": sha256_file(MANIFESTS / c / f"{k}_eids.txt")
                                    for c in ("bwm", "rs") for k in ("heldin", "heldout")},
        "bwm_tertile_edges": {"n_train": [float(x) for x in tr_edges],
                              "clusters": [float(x) for x in cl_edges]},
        "bwm_cell_sizes": cell_sizes,
        "rs_excluded": rs_excluded,
        "n_sessions": len(sessions), "n_bwm": n_bwm, "n_rs": n_rs,
        "n_reused": sum(1 for s in sessions if s["reuse_pilot_pretrain"]),
        "est_total_pretrain_gpu_h": round(sum(s["est_pretrain_gpu_h"] for s in sessions), 1),
        "sessions": sessions,
        "pilot_reference": reference,
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {args.out}")
    print(f"bwm tertile edges: n_train {tr_edges.round(1).tolist()}, clusters {cl_edges.round(1).tolist()}")
    print("bwm cell sizes (candidates):", cell_sizes)
    print(f"sample: {n_bwm} bwm + {n_rs} rs = {len(sessions)}; "
          f"reusing {manifest['n_reused']} pilot pretrains: "
          f"{[s['eid'][:8] for s in sessions if s['reuse_pilot_pretrain']]}")
    print(f"pilot reference only: {[r['eid'][:8] for r in reference]}")
    print(f"rs excluded: {[(x['eid'][:8], x['n_train'], x['n_val'], x['n_test']) for x in rs_excluded]}")
    print(f"estimated pretraining: {manifest['est_total_pretrain_gpu_h']} GPU-h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
