#!/usr/bin/env python
"""Offline checks for the behaviour-decoding tooling. CPU, seconds to minutes.

    python script/test_behavior_decode.py

Parity checks compare decode_components against committed golden fixtures
(script/fixtures/neds_parity.{json,npz}) holding NEDS's own literals and the
outputs of NEDS's own functions, made by make_neds_fixtures.py. They need no
NEDS checkout. When the reference copy neds_scripts/ is present (and
DECODE_TESTS_NO_NEDS is unset), the suite also re-derives every fixture value
from that source and checks the recorded file hashes. Also: loss semantics,
strict split isolation, selection bookkeeping, and the held-in guard.
"""

import hashlib
import json
import os
import subprocess
import sys
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


FIXTURES = REPO / "script" / "fixtures"
FIX = json.loads((FIXTURES / "neds_parity.json").read_text())
FIXNPZ = np.load(FIXTURES / "neds_parity.npz")
NEDS_ON = NEDS.is_dir() and not os.environ.get("DECODE_TESTS_NO_NEDS")


def fixture_value2label():
    """var_value2label from the fixture, with NEDS's tuple keys restored."""
    return {var: {tuple(k): v for k, v in pairs}
            for var, pairs in FIX["literals"]["var_value2label"].items()}


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
    check("fixture: npz matches the sha256 recorded in neds_parity.json",
          hashlib.sha256((FIXTURES / "neds_parity.npz").read_bytes()).hexdigest() == FIX["npz_sha256"])
    for name, lit in (("choice", "choice_lookup"), ("block", "block_lookup")):
        lookup = FIX["literals"][lit]
        vals = np.array([float(k) for k in lookup] * 3, dtype=np.float32)
        neds_labels = np.array([lookup[str(x)] for x in vals])   # NEDS base.py:527-528
        ours, mask = dc.encode_static(name, vals)
        check(f"parity: {name} encoding == NEDS {lit} (fixture)",
              mask.all() and np.array_equal(ours, neds_labels), str(lookup))
    labels, mask = dc.encode_static("choice", np.array([0.0, np.nan, 1.0]))
    check("choice 0 and NaN are masked, +1 kept", list(mask) == [False, False, True])

    wheel = FIXNPZ["align_wheel"]
    ours = dc.minmax_scale(wheel, wheel.min(), wheel.max())
    check("parity: min-max formula == NEDS align_data output (fixture, same statistics)",
          np.allclose(ours, FIXNPZ["align_wheel_neds_output"], rtol=0, atol=1e-12))

    torch.manual_seed(0)
    x = torch.randn(5, 100, 16)
    sh, dh = dc.StaticHead(100, 16, 3), dc.DynamicHead(16)
    check("parity: static head (B,T,H)->(B,C), token weights init in [0,1)",
          sh(x).shape == (5, 3) and bool(((sh.token_weight >= 0) & (sh.token_weight < 1)).all()))
    check("parity: dynamic head (B,T,H)->(B,T)", dh(x).shape == (5, 100))

    for layout in FIX["psth_layouts"]:
        cond, y, pred = (FIXNPZ[f"psth_{layout}_{k}"] for k in ("cond", "y", "pred"))
        ours = dc.dynamic_metrics(y, pred, cond)
        ref = FIX["psth"][layout]
        check(f"parity: PSTH and trial R2 == NEDS plot_psth (fixture, {layout})",
              np.isclose(ours["r2_psth"], ref["r2_psth"], rtol=0, atol=1e-12)
              and np.isclose(ours["r2_trial"], ref["r2_trial"], rtol=0, atol=1e-12),
              f"{ours['r2_psth']:.6f}/{ours['r2_trial']:.6f} vs {ref['r2_psth']:.6f}/{ref['r2_trial']:.6f}")

    if NEDS_ON:
        import make_neds_fixtures as mnf
        hashes = {f: mnf.sha256(NEDS / f) for f in FIX["provenance"]["sha256"]}
        check("source: neds_scripts/ files hash to the values recorded in the fixture",
              hashes == FIX["provenance"]["sha256"],
              ", ".join(f for f in hashes if hashes[f] != FIX["provenance"]["sha256"][f]))
        inputs = {k: FIXNPZ[k] for k in FIXNPZ.files}
        lit, aligned, psth = mnf.neds_outputs(NEDS, inputs, FIX["psth_layouts"])
        check("source: NEDS literals re-extracted == fixture", lit == FIX["literals"])
        check("source: NEDS align_data re-run == fixture output",
              np.array_equal(aligned, FIXNPZ["align_wheel_neds_output"]))
        check("source: NEDS plot_psth re-run == fixture outputs", psth == FIX["psth"])
    else:
        print("note: neds_scripts/ absent or DECODE_TESTS_NO_NEDS set; parity checked from fixtures only")
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
    null = dc.null_val_losses(prep["targets"]["train"], prep["targets"]["val"])
    scores = [dc.selection_score({n: hist0[f"val_{n}"][e] for n in names}, null)
              for e in range(len(hist0["val_total"]))]
    check("selection: joint epoch == argmin of the logged null-normalized val score",
          sel0["joint_epoch"] == int(np.argmin(scores))
          and np.allclose(scores, hist0["val_selection_score"], rtol=0, atol=1e-12))
    check("selection: score is the equally weighted mean of val_loss/null (not the val total)",
          np.isclose(sel0["joint_score"],
                     np.mean([hist0[f"val_{n}"][sel0["joint_epoch"]] / null[n] for n in names])))
    tv = {n: {"y": torch.as_tensor(t["y"], dtype=torch.long if n in dc.STATIC else torch.float32),
              "mask": torch.as_tensor(t["mask"])} for n, t in prep["targets"]["val"].items()}
    heads0.eval()
    with torch.no_grad():
        _, comps = dc.total_loss(heads0, feats["val"], tv)
    rescored = dc.selection_score({n: float(comps[n][0]) for n in names}, null)
    check("selection: returned heads ARE the joint-epoch heads (val score reproduces)",
          np.isclose(rescored, sel0["joint_score"], rtol=0, atol=1e-6))

    # ---------------------------------------------------------- null val losses
    tr, va = prep["targets"]["train"], prep["targets"]["val"]
    counts = np.bincount(tr["block"]["y"][tr["block"]["mask"]], minlength=3) + 1.0
    manual_ce = float(-np.log((counts / counts.sum())[va["block"]["y"][va["block"]["mask"]]]).mean())
    m = tr["wheel-speed"]["mask"]
    trace = (tr["wheel-speed"]["y"] * m).sum(0) / m.sum(0)
    manual_mse = float(((trace[None] - va["wheel-speed"]["y"]) ** 2)[va["wheel-speed"]["mask"]].mean())
    check("null: static = CE of add-one-smoothed train class prior on val",
          np.isclose(null["block"], manual_ce, rtol=0, atol=1e-12))
    check("null: dynamic = MSE of the per-bin train mean trace on val",
          np.isclose(null["wheel-speed"], manual_mse, rtol=0, atol=1e-12))
    p_test = dc.prepare_targets(perturbed("test"))
    check("null: independent of the test split",
          dc.null_val_losses(p_test["targets"]["train"], p_test["targets"]["val"]) == null)
    vl = {n: 2.0 * null[n] for n in null}
    scaled_null = dict(null, **{"wheel-speed": null["wheel-speed"] * 1e3})
    vl_scaled = dict(vl, **{"wheel-speed": vl["wheel-speed"] * 1e3})
    check("null: score is invariant to a target's loss scale (equal weighting after normalization)",
          np.isclose(dc.selection_score(vl, null), dc.selection_score(vl_scaled, scaled_null))
          and np.isclose(dc.selection_score(vl, null), 2.0))

    # ---------------------------------------------------------- PSTH R2: point only
    pt = prep["targets"]["test"]["wheel-speed"]
    cond = np.stack([raw["test"][c] for c in dc.PSTH_CONDITIONS], axis=1)
    pr = np.where(pt["mask"], pt["y"], 0.0) + 0.1
    mets = dc.metrics_for("wheel-speed", pt, pr, cond)
    ci = dc.bootstrap_ci("wheel-speed", pt, pr, cond, n_boot=50)
    check("psth: r2_psth reported as a point estimate with cell sizes, no CI; r2_trial has a CI",
          "r2_psth" in mets and "psth_cells" in mets and "r2_psth" not in ci and "r2_trial" in ci)
    check("primary: trial R2 for dynamic targets, acc / balanced acc for static",
          mets["primary"] == "r2_trial"
          and dc.metrics_for("block", prep["targets"]["test"]["block"],
                             prep["targets"]["test"]["block"]["y"], cond)["primary"] == "balanced_acc")
    check("optimizer holds only head parameters",
          any(ok and name.startswith("n.") for ok, name, _ in log0))

    # ---------------------------------------------------------- diagnostics: paired deltas
    r2 = np.random.default_rng(5)
    yb = r2.integers(0, 3, 90)
    tgt_b = {"y": yb, "mask": np.ones(90, bool)}
    pb, lb = np.where(r2.random(90) < 0.7, yb, r2.integers(0, 3, 90)), r2.integers(0, 3, 90)
    nb = np.zeros(90, int)
    fixed = [np.arange(90), r2.integers(0, 90, 90), r2.integers(0, 90, 90)]
    d = dc.paired_delta_ci("block", tgt_b, pb, lb, nb, draws=fixed)
    from sklearn.metrics import balanced_accuracy_score as bas
    manual = [(bas(yb[i], pb[i]) - bas(yb[i], nb[i]), bas(yb[i], pb[i]) - bas(yb[i], lb[i])) for i in fixed]
    check("delta: each fixed draw's paired deltas == direct computation on those indices",
          np.allclose(d["draw_deltas"]["null"], [m[0] for m in manual], rtol=0, atol=1e-12)
          and np.allclose(d["draw_deltas"]["linear"], [m[1] for m in manual], rtol=0, atol=1e-12))
    check("delta: identity draw reproduces the point deltas",
          np.isclose(d["draw_deltas"]["null"][0], d["delta_null"])
          and np.isclose(d["draw_deltas"]["linear"][0], d["delta_linear"]))
    yd = r2.normal(size=(40, 100)); tgt_d = {"y": yd, "mask": np.ones((40, 100), bool)}
    pdn, ldn, ndn = yd + 0.3 * r2.normal(size=yd.shape), yd + 0.5 * r2.normal(size=yd.shape), np.zeros_like(yd)
    fx = [r2.integers(0, 40, 40) for _ in range(3)]
    dd = dc.paired_delta_ci("wheel-speed", tgt_d, pdn, ldn, ndn, draws=fx)
    r2t = lambda a, b: float(r2_score_sklearn(a.reshape(-1), b.reshape(-1)))
    check("delta: trial-R2 deltas use the same indices for all three models",
          np.allclose(dd["draw_deltas"]["linear"], [r2t(yd[i], pdn[i]) - r2t(yd[i], ldn[i]) for i in fx],
                      rtol=0, atol=1e-10))
    draws_b = dc.delta_draws("block", yb, 20, np.random.default_rng(0))
    check("delta: block draws are class-stratified (every class count preserved)",
          all(np.array_equal(np.bincount(yb[i], minlength=3), np.bincount(yb, minlength=3)) for i in draws_b))
    yc = r2.integers(0, 2, 90)
    draws_c = dc.delta_draws("choice", yc, 50, np.random.default_rng(0))
    check("delta: choice / dynamic draws are ordinary n-with-replacement",
          all(len(i) == 90 for i in draws_c)
          and any(not np.array_equal(np.bincount(yc[i], minlength=2), np.bincount(yc, minlength=2))
                  for i in draws_c))
    check("delta: block scheme labelled class-stratified, choice ordinary",
          d["scheme"] == "class-stratified"
          and dc.paired_delta_ci("choice", {"y": yc, "mask": np.ones(90, bool)}, yc, yc, yc,
                                 n_boot=5)["scheme"] == "ordinary")

    # ---------------------------------------------------------- diagnostics: trial selection
    ys = r2.normal(size=(60, 100)) + np.linspace(0, 5, 60)[:, None]
    ms = np.ones((60, 100), bool)
    ms[3] = False
    ms[10, :50] = False
    ch, full = dc.select_trials(ys, ms)
    act = {i: ys[i][ms[i]].mean() for i in range(60) if ms[i].any()}
    cand = sorted([i for i in range(60) if ms[i].all()], key=lambda i: act[i])
    bins = np.array_split(np.array(cand), 8)
    check("select: 8 distinct fully valid trials, median-ranked one per mean-activity bin",
          len(set(ch.tolist())) == 8 and full.all()
          and ch.tolist() == [int(b[(len(b) - 1) // 2]) for b in bins])
    check("select: deterministic and a function of GT/mask only",
          np.array_equal(ch, dc.select_trials(ys.copy(), ms.copy())[0]))
    ms2 = np.zeros((10, 100), bool); ms2[:5] = True; ms2[5:, :30] = True
    ch2, full2 = dc.select_trials(r2.normal(size=(10, 100)), ms2)
    check("select: partially valid trials used only when fewer than 8 fully valid",
          len(ch2) == 8 and full2.sum() == 5 and set(ch2[full2].tolist()) == set(range(5)))
    yp = np.array([[1.0, 5.0, np.nan], [2.0, 2.0, 2.0]])
    mp = np.isfinite(yp)
    mp2 = np.zeros((2, 3), bool); mp2[0, :2] = True
    check("select: mean activity is over valid bins only",
          dc.select_trials(yp, mp2, k=2)[0].tolist() == [0] and
          dc.select_trials(np.nan_to_num(yp, nan=100.0), mp, k=2)[0].tolist() == [1, 0])

    # ---------------------------------------------------------- diagnostics: confusion semantics
    yy = r2.integers(0, 3, 200); pp = np.where(r2.random(200) < 0.6, yy, r2.integers(0, 3, 200))
    counts, norm = dc.row_normalised_confusion(yy, pp, 3)
    check("confusion: rows are true classes; diagonal mean == balanced accuracy",
          np.array_equal(counts.sum(1), np.bincount(yy, minlength=3))
          and np.isclose(np.diag(norm).mean(), bas(yy, pp), rtol=0, atol=1e-12))
    neds_labels = fixture_value2label()

    def labels_match(class_names):
        ok = True
        for name in ("choice", "block"):
            for raw_v, cls in dc.LOOKUP[name].items():
                neds = neds_labels[name][(raw_v,)]
                ours = class_names[name][cls]
                ok &= (ours.split(" ")[0] == neds)
        return ok
    check("confusion: class labels map through LOOKUP and match NEDS var_value2label (fixture)",
          labels_match(dc.CLASS_NAMES)
          and dc.CLASS_NAMES["choice"] == {0: "right (-1)", 1: "left (+1)"}
          and dc.CLASS_NAMES["block"] == {0: "p(left)=0.2", 1: "p(left)=0.5", 2: "p(left)=0.8"})
    swapped = {"choice": {0: dc.CLASS_NAMES["choice"][1], 1: dc.CLASS_NAMES["choice"][0]},
               "block": dc.CLASS_NAMES["block"]}
    check("confusion: a swapped choice mapping is caught", not labels_match(swapped))

    # ---------------------------------------------------------- diagnostics: baseline refit
    Xr = {s: r2.poisson(0.3, (len(raw[s]["choice"]), 200)).astype(np.float32) for s in dc.SPLITS}
    pa, sa = db.fit_static("block", Xr, prep["targets"], 42)
    pb2, sb = db.fit_static("block", Xr, prep["targets"], 42)
    da, ra = db.fit_dynamic("wheel-speed", Xr, prep["targets"])
    db2, rb = db.fit_dynamic("wheel-speed", Xr, prep["targets"])
    check("refit: categorical predictions and chosen C identical across refits",
          np.array_equal(pa, pb2) and sa["chosen"] == sb["chosen"])
    check("refit: continuous predictions within audit tolerance, same alpha",
          np.allclose(da, db2, rtol=1e-7, atol=0) and ra["chosen"]["alpha"] == rb["chosen"]["alpha"])

    # ---------------------------------------------------------- aggregate: states and triggers
    import decode_aggregate as da
    check("aggregate: CI states are exclusive and exhaustive at the boundaries",
          da.ci_state([0.01, 0.2]) == "above" and da.ci_state([0.0, 0.2]) == "uncertain"
          and da.ci_state([-0.1, 0.0]) == "at/below" and da.ci_state([0.0, 0.0]) == "at/below"
          and da.ci_state([-0.1, 0.05]) == "uncertain" and da.ci_state([-0.3, -0.1]) == "at/below")

    def d_entry(lo, hi, point=0.1):
        return {"point": {"probe": 0.5, "linear": 0.4, "null": 0.3},
                "delta_null": point, "delta_linear": point,
                "delta_null_ci": [lo, hi], "delta_linear_ci": [lo, hi]}

    def mk(eid, ci_by_target, dropped=(), band=">0", block=None, audit=True):
        return {"eid": eid, "dropped": {t: "dropped on train" for t in dropped},
                "deltas": {t: d_entry(*ci) for t, ci in ci_by_target.items()},
                "cosmooth_band": band, "block_control": block, "audit_ok": audit}

    below, unc = (-0.2, -0.01), (-0.1, 0.1)
    four_below = {t: below for t in dc.TARGETS}
    rows = [mk("a" * 8, four_below), mk("b" * 8, {t: unc for t in dc.TARGETS})]
    t3 = da.trigger_contradiction(rows)
    check("trigger 3: exactly 50% at/below does not fire", t3["eligible"] == 2 and not t3["fires"])
    rows.append(mk("c" * 8, {**{t: below for t in dc.TARGETS[:3]}}, dropped=(dc.TARGETS[3],)))
    t3 = da.trigger_contradiction(rows)
    check("trigger 3: >50% fires; a dropped target leaves 3 available and still counts",
          t3["eligible"] == 3 and t3["hits"] == ["a" * 8, "c" * 8] and t3["fires"])
    rows2 = [mk("d" * 8, {t: below for t in dc.TARGETS[:2]}, dropped=dc.TARGETS[2:]),
             mk("e" * 8, four_below, band="~0")]
    check("trigger 3: < 3 available targets or band not >0 -> not eligible",
          da.trigger_contradiction(rows2)["eligible"] == 0
          and not da.trigger_contradiction(rows2)["fires"])
    check("trigger 3: 'uncertain' never counts toward firing",
          not da.trigger_contradiction([mk("f" * 8, {t: unc for t in dc.TARGETS})] * 3)["fires"])

    blk = lambda n, pos, probe: {"n_half_test": n, "recall_half_position": pos,  # noqa: E731
                                 "recall_half_probe": probe}
    r1 = [mk("g" * 8, {}, block=blk(10, 0.9, 0.8)), mk("h" * 8, {}, block=blk(9, 1.0, 0.1)),
          mk("i" * 8, {}, block=blk(15, 0.5, 0.9))]
    t1 = da.trigger_block(r1)
    check("trigger 1: 9 half-block trials excluded, 10 eligible; ties count; 1/2 = 50% does not fire",
          t1["eligible"] == 2 and t1["excluded"] == ["h" * 8] and t1["hits"] == ["g" * 8]
          and not t1["fires"])
    r1.append(mk("j" * 8, {}, block=blk(12, 0.7, 0.7)))
    r1.append(mk("k" * 8, {}, dropped=("block",), block=blk(20, 1.0, 0.0)))
    t1 = da.trigger_block(r1)
    check("trigger 1: equal recall is a hit; dropped block excluded; 2/3 fires",
          t1["eligible"] == 3 and "k" * 8 in t1["excluded"] and t1["fires"])

    ok_row = mk("l" * 8, {t: unc for t in dc.TARGETS[:3]}, dropped=(dc.TARGETS[3],))
    check("trigger 2: a dropped target never counts as a failure",
          not da.trigger_tooling([ok_row])["fires"])
    nan_row = mk("m" * 8, {t: unc for t in dc.TARGETS})
    nan_row["deltas"][dc.TARGETS[1]]["point"]["probe"] = float("nan")
    check("trigger 2: non-finite primary metric on a non-dropped target fires; failed audit fires",
          da.trigger_tooling([nan_row])["fires"]
          and da.trigger_tooling([mk("n" * 8, {t: unc for t in dc.TARGETS}, audit=False)])["fires"])

    tt = da.target_table([ok_row, nan_row, mk("o" * 8, {t: below for t in dc.TARGETS})],
                         dc.TARGETS[3])
    check("aggregate: eligible n = N - dropped; status counts sum to eligible n",
          tt["N"] == 3 and tt["dropped"] == 1 and tt["eligible"] == 2
          and sum(tt["null_states"].values()) == tt["eligible"]
          and sum(tt["linear_states"].values()) == tt["eligible"])

    # trial position (splits.npz) is in parquet row order: the parquet choice of
    # every row equals the IBL trial table's choice at that row's trial index
    import pandas as pd, pyarrow.parquet as pq
    E = "71e55bfe-5a3a-4cba-bdc7-f085140d798e"
    sp = np.load(f"{da.DATA_ROOT}/{E}/splits.npz")
    trials = pd.read_pickle(f"/work/nvme/beml/ac136/LFP_FM/data/processed_wheel/{E}/trials.pkl")
    aligned = True
    for s in dc.SPLITS:
        ch = np.array([v[0] for v in pq.read_table(f"{da.DATA_ROOT}/{E}/data/{s}.parquet",
                                                   columns=["choice"]).column(0).to_pylist()])
        aligned &= np.array_equal(ch, trials["choice"].to_numpy()[sp[f"{s}_trial_indices"]])
    check("aggregate: splits.npz trial indices follow parquet row order (real session)", aligned)
    yb = np.array([0, 1, 1, 2, 1, 0]); mb = np.array([1, 1, 1, 1, 0, 1], bool)
    check("aggregate: half_recall uses valid p(left)=0.5 test trials only",
          da.half_recall(yb, np.array([0, 1, 0, 2, 1, 0]), mb) == 0.5)

    # ---------------------------------------------------------- block 0.2-vs-0.8 check
    import decode_block_check as dbc
    tb = {"y": np.array([0, 2, 1, 0, 2, 2]), "mask": np.array([1, 1, 1, 1, 1, 0], bool)}
    sub = dbc.biased_target(tb)
    yv, pv = tb["y"][sub["mask"]], np.array([0, 1, 1, 0, 2, 2])[sub["mask"]]
    check("block check: only true 0.2/0.8 trials kept; a 0.5 prediction counts as an error",
          sub["mask"].tolist() == [True, True, False, True, True, False]
          and np.isclose(dc.static_metrics(yv, pv)["balanced_acc"], (1.0 + 0.5) / 2))

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
