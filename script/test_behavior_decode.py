#!/usr/bin/env python
"""Offline checks for the behaviour-decoding tooling. CPU, seconds to minutes.

    python script/test_behavior_decode.py

Parity checks run NEDS's OWN code (functions and literals extracted from the
reference copy in neds_scripts/ with ast, not restated), against
decode_components. Also: loss semantics, strict split isolation, selection
bookkeeping, and the held-in guard.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import r2_score as r2_score_sklearn  # noqa: E402

import decode_components as dc  # noqa: E402

NEDS = REPO / "neds_scripts"
results = []


def check(name, ok, detail=""):
    results.append((bool(ok), name, detail))


def neds_functions(path, names, extra_globals=None):
    """Exec the named top-level functions from a NEDS file, verbatim."""
    tree = ast.parse(path.read_text())
    src = [ast.get_source_segment(path.read_text(), n) for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(src) == len(names), f"missing NEDS functions in {path.name}"
    g = {"np": np, "r2_score_sklearn": r2_score_sklearn, "r2_score": r2_score_sklearn,
         **(extra_globals or {})}
    exec("\n\n".join(src), g)
    return g


def neds_literal(path, target):
    """The dict literal assigned to ``target`` anywhere in a NEDS file."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == target for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(target)


def synthetic_raw(rng, n=(60, 12, 20), T=100, nan_trials=(3, 7)):
    raw = {}
    for split, k in zip(dc.SPLITS, n):
        wheel = rng.gamma(2.0, 1.0, (k, T))
        wme = rng.normal(5, 3, (k, T))
        for i in nan_trials:
            if i < k:
                wheel[i] = np.nan
        wme[0, :5] = np.nan
        choice = rng.choice([-1.0, 1.0], k)
        choice[1] = 0.0
        raw[split] = {"choice": choice, "block": rng.choice([0.2, 0.5, 0.8], k),
                      "reward": rng.choice([0.0, 1.0], k),
                      "wheel-speed": wheel, "whisker-motion-energy": wme}
    return raw


def main():
    rng = np.random.default_rng(0)

    # ---------------------------------------------------------- component parity
    for name, lit in (("choice", "choice_lookup"), ("block", "block_lookup")):
        lookup = neds_literal(NEDS / "base.py", lit)
        vals = np.array([float(k) for k in lookup] * 3, dtype=np.float32)
        neds_labels = np.array([lookup[str(x)] for x in vals])   # base.py:527-528
        ours, mask = dc.encode_static(name, vals)
        check(f"parity: {name} encoding == NEDS {lit}", mask.all() and np.array_equal(ours, neds_labels),
              str(lookup))
    labels, mask = dc.encode_static("choice", np.array([0.0, np.nan, 1.0]))
    check("choice 0 and NaN are masked, +1 kept", list(mask) == [False, False, True])

    ad_globals = neds_functions(NEDS / "ibl_data_utils.py", ["align_data"],
                                {"DYNAMIC_VARS": ["wheel-speed", "whisker-motion-energy"],
                                 "standardize_lfp_data": None})
    wheel = rng.gamma(2.0, 1.0, (30, 100))
    spikes = np.zeros((30, 100, 4))
    _, neds_beh, _, _, _ = ad_globals["align_data"](
        spikes, {"wheel-speed": list(wheel)}, None, beh_names=["wheel-speed"], trials_mask=None)
    ours = dc.minmax_scale(wheel, wheel.min(), wheel.max())
    check("parity: min-max formula == NEDS align_data (same statistics)",
          np.allclose(ours, neds_beh["wheel-speed"], rtol=0, atol=1e-12))

    torch.manual_seed(0)
    x = torch.randn(5, 100, 16)
    sh, dh = dc.StaticHead(100, 16, 3), dc.DynamicHead(16)
    check("parity: static head (B,T,H)->(B,C), token weights init in [0,1)",
          sh(x).shape == (5, 3) and bool(((sh.token_weight >= 0) & (sh.token_weight < 1)).all()))
    check("parity: dynamic head (B,T,H)->(B,T)", dh(x).shape == (5, 100))

    ev = neds_functions(NEDS / "eval_utils.py",
                        ["compute_PSTH", "compute_all_psth", "compute_R2_psth",
                         "compute_R2_main", "plot_psth"])
    K = 80
    y, pred = rng.normal(size=(K, 100)), rng.normal(size=(K, 100))
    pred = 0.6 * y + 0.4 * pred
    cond = np.stack([rng.choice([0.2, 0.5, 0.8], K), rng.choice([-1.0, 1.0], K),
                     rng.choice([0.0, 1.0], K)], axis=1)       # (block, choice, reward)
    X = np.zeros((K, 100, 4))
    X[:, :, 0], X[:, :, 1], X[:, :, 2] = cond[:, 1:2], cond[:, 2:3], cond[:, 0:1]
    name2idx = {"block": [2], "choice": [0], "reward": [1], "wheel": [3]}
    r2_psth, r2_trial = ev["plot_psth"](X, y, pred, ["block", "choice", "reward"], name2idx, {},
                                        save_plot=False)
    ours = dc.dynamic_metrics(y, pred, cond)
    check("parity: PSTH R2 == NEDS plot_psth", np.isclose(ours["r2_psth"], r2_psth, rtol=0, atol=1e-12),
          f"{ours['r2_psth']:.6f} vs {r2_psth:.6f}")
    check("parity: trial R2 == NEDS plot_psth (unsubtracted)",
          np.isclose(ours["r2_trial"], r2_trial, rtol=0, atol=1e-12))
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    a, b = rng.integers(0, 3, 50), rng.integers(0, 3, 50)
    m = dc.static_metrics(a, b)
    check("parity: accuracy and balanced accuracy == sklearn (NEDS eval_utils.py:464-465)",
          m["acc"] == accuracy_score(a, b) and m["balanced_acc"] == balanced_accuracy_score(a, b))

    # ---------------------------------------------------------- loss semantics
    pred = torch.randn(6, 100, requires_grad=True)
    yv = torch.randn(6, 100)
    mask = torch.rand(6, 100) > 0.3
    y_nan = yv.clone()
    y_nan[~mask] = float("nan")
    loss, n = dc.component_loss("wheel-speed", pred, y_nan, mask)
    manual = ((pred - yv) ** 2)[mask].mean()
    loss.backward()
    check("loss: dynamic masked mean ignores masked/NaN entries",
          torch.allclose(loss, manual) and n == int(mask.sum()))
    check("loss: no NaN gradient from NaN masked targets", bool(torch.isfinite(pred.grad).all()))
    logits = torch.randn(6, 3, requires_grad=True)
    lab = torch.tensor([0, 1, 2, 0, 1, 2])
    smask = torch.tensor([True, False, True, True, False, True])
    loss, n = dc.component_loss("block", logits, lab, smask)
    manual = torch.nn.functional.cross_entropy(logits[smask], lab[smask])
    check("loss: static CE is the mean over valid trials", torch.allclose(loss, manual) and n == 4)
    loss, n = dc.component_loss("block", logits, lab, torch.zeros(6, dtype=torch.bool))
    loss.backward()
    check("loss: all-masked component is exactly 0 with n_valid 0 and finite grad",
          float(loss) == 0.0 and n == 0 and bool(torch.isfinite(logits.grad).all()))

    raw = synthetic_raw(rng)
    prep = dc.prepare_targets(raw)
    torch.manual_seed(1)
    heads = dc.build_heads(dc.active_targets(prep), 100, 8)
    feat = torch.randn(60, 100, 8)
    tg = {n: {"y": torch.as_tensor(t["y"], dtype=torch.long if n in dc.STATIC else torch.float32),
              "mask": torch.as_tensor(t["mask"])} for n, t in prep["targets"]["train"].items()}
    total, comps = dc.total_loss(heads, feat, tg)
    check("loss: total == plain sum of logged components",
          torch.allclose(total, sum(c[0] for c in comps.values())))
    total2, _ = dc.total_loss(heads, feat, tg)
    check("loss: deterministic", torch.equal(total, total2))

    # ---------------------------------------------------------- split isolation
    info = prep["info"]
    check("targets: choice 0 masked and counted", info["choice"]["n_masked"]["train"] == 1)
    check("targets: invalid wheel trials counted", info["wheel-speed"]["n_invalid_trials"]["train"] == 2)
    tr = raw["train"]["wheel-speed"]
    check("targets: wheel scaled with TRAIN min/max",
          np.isclose(info["wheel-speed"]["train_min"], np.nanmin(tr))
          and np.isclose(info["wheel-speed"]["train_max"], np.nanmax(tr)))

    def perturbed(split):
        r = {s: {k: np.array(v, copy=True) for k, v in raw[s].items()} for s in dc.SPLITS}
        r[split]["wheel-speed"] = r[split]["wheel-speed"] * 50 + 7
        r[split]["whisker-motion-energy"] = -r[split]["whisker-motion-energy"]
        r[split]["choice"] = -r[split]["choice"]
        r[split]["block"] = r[split]["block"][::-1].copy()
        return r

    import behavior_decode as bd
    import decode_baseline as db

    def train(prep_, epochs=6):
        torch.manual_seed(0)
        feats = {s: torch.randn(len(raw[s]["choice"]), 100, 8,
                                generator=torch.Generator().manual_seed(i))
                 for i, s in enumerate(dc.SPLITS)}
        log = []
        heads, hist, sel = bd.train_heads(feats, prep_, epochs, 42, torch.device("cpu"),
                                          torch.nn.Linear(2, 2), log)
        return heads, hist, sel, log, feats

    heads0, hist0, sel0, log0, feats = train(prep)
    for split in ("val", "test"):
        p = dc.prepare_targets(perturbed(split))
        same_info = all(p["info"][n].get(k) == info[n].get(k)
                        for n in dc.TARGETS for k in ("dropped", "train_min", "train_max",
                                                     "train_invalid_frac", "train_classes_present"))
        same_train = all(np.array_equal(p["targets"]["train"][n]["y"], prep["targets"]["train"][n]["y"])
                         and np.array_equal(p["targets"]["train"][n]["mask"],
                                            prep["targets"]["train"][n]["mask"]) for n in dc.TARGETS)
        chance_same = all(np.array_equal(dc.chance_predictions(n, p["targets"]["train"][n], 5)[0],
                                         dc.chance_predictions(n, prep["targets"]["train"][n], 5)[0])
                          for n in dc.active_targets(prep))
        check(f"isolation: perturbing {split} leaves scaling, drops, train targets, chance unchanged",
              same_info and same_train and chance_same)
        heads1, hist1, sel1, _, _ = train(p)
        same_traj = all(hist1[k] == hist0[k] for k in hist0 if k.startswith("train_"))
        check(f"isolation: perturbing {split} leaves the head training trajectory unchanged",
              same_traj)
        if split == "test":
            same_sel = (sel1["joint_epoch"] == sel0["joint_epoch"] and
                        all(sel1["per_target_diagnostic"][n]["own_best_epoch"]
                            == sel0["per_target_diagnostic"][n]["own_best_epoch"]
                            for n in sel0["per_target_diagnostic"]))
            same_w = all(torch.equal(a, b) for a, b in zip(heads0.state_dict().values(),
                                                          heads1.state_dict().values()))
            check("isolation: perturbing test leaves joint and per-target epochs and weights unchanged",
                  same_sel and same_w)

        # baseline: grid models fit on train only; test never reaches selection
        Xs = {s: rng.poisson(0.3, (len(raw[s]["choice"]), 200)).astype(np.float32) for s in dc.SPLITS}
        Xs = {s: v for s, v in Xs.items()}
        _, s0 = db.fit_static("choice", Xs, prep["targets"], 42)
        _, s1 = db.fit_static("choice", Xs, p["targets"], 42)
        _, r0 = db.fit_dynamic("wheel-speed", Xs, prep["targets"])
        _, r1 = db.fit_dynamic("wheel-speed", Xs, p["targets"])
        if split == "test":
            check("isolation: perturbing test leaves baseline val losses and chosen C/alpha unchanged",
                  s0 == s1 and r0 == r1)
        else:
            check("isolation: baseline C/alpha chosen on val (val perturbation can move it; train fit fixed)",
                  [g["C"] for g in s0["grid"]] == [g["C"] for g in s1["grid"]])

    # ---------------------------------------------------------- selection bookkeeping
    names = dc.active_targets(prep)
    argmins = {n: int(np.argmin(hist0[f"val_{n}"])) for n in names}
    check("selection: per-target epochs == argmin of logged val component loss",
          all(sel0["per_target_diagnostic"][n]["own_best_epoch"] == argmins[n] for n in names))
    check("selection: joint epoch == argmin of logged val total", sel0["joint_epoch"]
          == int(np.argmin(hist0["val_total"])))
    tv = {n: {"y": torch.as_tensor(t["y"], dtype=torch.long if n in dc.STATIC else torch.float32),
              "mask": torch.as_tensor(t["mask"])} for n, t in prep["targets"]["val"].items()}
    heads0.eval()
    with torch.no_grad():
        vt, _ = dc.total_loss(heads0, feats["val"], tv)
    check("selection: returned heads ARE the joint-epoch heads (val total reproduces)",
          np.isclose(float(vt), sel0["joint_val_total"], rtol=0, atol=1e-6))
    check("optimizer holds only head parameters",
          any(ok and name.startswith("n.") for ok, name, _ in log0))

    # ---------------------------------------------------------- shared preprocessing
    src = (REPO / "script" / "decode_baseline.py").read_text()
    check("baseline uses the probe's read_behaviour and dc.prepare_targets",
          db.read_behaviour is bd.read_behaviour and "dc.prepare_targets(raw)" in src)

    # ---------------------------------------------------------- held-in guard
    for eid, label in (("0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63", "held-out"),
                       ("00000000-0000-0000-0000-000000000000", "unknown")):
        r = subprocess.run([sys.executable, str(REPO / "script" / "behavior_decode.py"), eid,
                            "--dry-run"], capture_output=True, text=True)
        check(f"guard: {label} eid exits 3", r.returncode == 3, f"exit {r.returncode}")

    n = sum(ok for ok, _, _ in results)
    for ok, name, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    print(f"\ntest_behavior_decode: {n}/{len(results)} passed")
    return 0 if n == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
