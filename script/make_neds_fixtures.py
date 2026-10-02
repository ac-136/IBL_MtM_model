#!/usr/bin/env python
"""Freeze NEDS component-parity references into committed fixtures.

    python script/make_neds_fixtures.py [--neds-dir neds_scripts] [--neds-commit SHA]

Run once while the NEDS reference copy is present. Writes
script/fixtures/neds_parity.json (provenance, literals, golden outputs) and
script/fixtures/neds_parity.npz (the synthetic inputs those outputs were
computed on). test_behavior_decode.py checks decode_components against these
fixtures, so the parity tests no longer need neds_scripts/; when the tree is
present the tests also re-derive every value from its source and check the
recorded file hashes, catching a changed reference copy.

Everything taken from NEDS is executed or read verbatim with ast: the lookup
literals (base.py), var_value2label (eval_utils.py), align_data
(ibl_data_utils.py) and plot_psth with its compute_* helpers (eval_utils.py).
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score as r2_score_sklearn

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "script" / "fixtures"
HASHED = ("base.py", "eval_utils.py", "ibl_data_utils.py", "mm.py")
PSTH_FUNCS = ["compute_PSTH", "compute_all_psth", "compute_R2_psth", "compute_R2_main", "plot_psth"]
NAME2IDX = {"block": [2], "choice": [0], "reward": [1], "wheel": [3]}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neds_functions(path, names, extra_globals=None):
    """Exec the named top-level functions from a NEDS file, verbatim."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    src = [ast.get_source_segment(text, n) for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(src) == len(names), f"missing NEDS functions in {Path(path).name}"
    g = {"np": np, "r2_score_sklearn": r2_score_sklearn, "r2_score": r2_score_sklearn,
         **(extra_globals or {})}
    exec("\n\n".join(src), g)
    return g


def neds_literal(path, target):
    """The literal assigned to ``target`` anywhere in a NEDS file."""
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == target for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(target)


def synthetic_inputs():
    """Fixed inputs, stored in the fixture so no RNG stream is relied on later."""
    rng = np.random.default_rng(20260930)
    out = {"align_wheel": rng.gamma(2.0, 1.0, (30, 100))}
    layouts = {
        "random_80": np.stack([rng.choice([0.2, 0.5, 0.8], 80), rng.choice([-1.0, 1.0], 80),
                               rng.choice([0.0, 1.0], 80)], axis=1),
        # cells of size 1, 2 and 3 alongside larger ones
        "small_cells": np.array([[0.5, -1.0, 0.0]] + [[0.5, 1.0, 1.0]] * 2 + [[0.2, 1.0, 0.0]] * 3
                                + [[0.2, -1.0, 1.0]] * 20 + [[0.8, 1.0, 1.0]] * 24),
        "one_cell": np.tile([[0.8, 1.0, 1.0]], (40, 1)),
    }
    for name, cond in layouts.items():
        k = len(cond)
        y = rng.normal(size=(k, 100))
        out[f"psth_{name}_cond"] = cond
        out[f"psth_{name}_y"] = y
        out[f"psth_{name}_pred"] = 0.6 * y + 0.4 * rng.normal(size=(k, 100))
    return out, list(layouts)


def neds_outputs(neds_dir, inputs, layouts):
    """Every golden value, computed by NEDS's own code on ``inputs``."""
    neds_dir = Path(neds_dir)
    lit = {"choice_lookup": neds_literal(neds_dir / "base.py", "choice_lookup"),
           "block_lookup": neds_literal(neds_dir / "base.py", "block_lookup"),
           # tuple keys -> [[value...], label] pairs for JSON
           "var_value2label": {var: [[list(k), v] for k, v in d.items()]
                               for var, d in neds_literal(neds_dir / "eval_utils.py",
                                                          "var_value2label").items()}}
    ad = neds_functions(neds_dir / "ibl_data_utils.py", ["align_data"],
                        {"DYNAMIC_VARS": ["wheel-speed", "whisker-motion-energy"],
                         "standardize_lfp_data": None})
    wheel = inputs["align_wheel"]
    _, beh, _, _, _ = ad["align_data"](np.zeros((len(wheel), 100, 4)), {"wheel-speed": list(wheel)},
                                       None, beh_names=["wheel-speed"], trials_mask=None)
    ev = neds_functions(neds_dir / "eval_utils.py", PSTH_FUNCS)
    psth = {}
    for name in layouts:
        cond, y, pred = (inputs[f"psth_{name}_{k}"] for k in ("cond", "y", "pred"))
        X = np.zeros((len(cond), 100, 4))
        X[:, :, 0], X[:, :, 1], X[:, :, 2] = cond[:, 1:2], cond[:, 2:3], cond[:, 0:1]
        r2_psth, r2_trial = ev["plot_psth"](X, y, pred, ["block", "choice", "reward"], NAME2IDX, {},
                                            save_plot=False)
        psth[name] = {"r2_psth": float(r2_psth), "r2_trial": float(r2_trial)}
    return lit, beh["wheel-speed"], psth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neds-dir", default=str(REPO / "neds_scripts"))
    ap.add_argument("--neds-commit", default=None,
                    help="NEDS upstream commit, if known (the local copy is not a git checkout)")
    args = ap.parse_args()
    neds_dir = Path(args.neds_dir)

    inputs, layouts = synthetic_inputs()
    lit, aligned, psth = neds_outputs(neds_dir, inputs, layouts)
    inputs["align_wheel_neds_output"] = aligned

    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURES / "neds_parity.npz", **inputs)
    (FIXTURES / "neds_parity.json").write_text(json.dumps({
        "provenance": {
            "source": "user-supplied NEDS reference copy (neds_scripts/), not a git checkout",
            "neds_commit": args.neds_commit or "unknown",
            "sha256": {f: sha256(neds_dir / f) for f in HASHED},
            "generator": "script/make_neds_fixtures.py",
        },
        "literals": lit,
        "psth_layouts": layouts,
        "psth": psth,
        "npz_sha256": sha256(FIXTURES / "neds_parity.npz"),
    }, indent=2))
    print(f"wrote {FIXTURES}/neds_parity.{{json,npz}}")
    for name, v in psth.items():
        print(f"  {name}: r2_psth {v['r2_psth']:+.6f} r2_trial {v['r2_trial']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
