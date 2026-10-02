#!/usr/bin/env python
"""Cross-session behaviour-decoding aggregate, relation to neural prediction,
and the predeclared Tier 4 closure triggers.

    python script/decode_aggregate.py [--dir decoding_v2]

Populations (enforced in code):
  PRIMARY       the 48 untouched learnability sessions (34 bwm + 14 rs); every
                claim and every trigger is computed here, per cohort
  COMPLETENESS  all 52 (36 bwm + 16 rs), printed for reference; fires nothing
The four development sessions shaped the protocol and never enter PRIMARY;
they are marked "dev" wherever they appear. The stress test 5b49aca6 is in
neither population.

Probe-vs-null (and probe-vs-linear) status comes only from the paired 95% CI
recorded per session in plot_audit.json (authoritative, never recomputed):
  above: lo > 0      uncertain: lo <= 0 < hi      at/below: hi <= 0
Every target table prints population N / eligible n / dropped n; status
counts and fractions use eligible n. Descriptive only.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import numpy as np  # noqa: E402

import decode_components as dc  # noqa: E402

RESULTS = Path("/work/nvme/beml/ac136/IBL_MTM/results")
DATA_ROOT = "/work/nvme/beml/ac136/LFP_FM/data/datasets_wheel"
MANIFEST = REPO / "script" / "learnability_manifest.json"
DEV = ("71e55bfe-5a3a-4cba-bdc7-f085140d798e", "862ade13-53cd-4221-a3fa-dda8643641f2",
       "4aa1d525-5c7d-4c50-a147-ec53a9014812", "b9c205c3-feac-485b-a89d-afc96d9cb280")
STRESS = "5b49aca6-a6f4-4075-931a-617ad64c219c"
STATES = ("above", "uncertain", "at/below")
BLOCK_HALF = dc.LOOKUP["block"][0.5]          # class index of p(left)=0.5
MIN_HALF_TRIALS = 10                          # trigger 1 eligibility (fixed in the plan)
TIME_FLAT = 0.005                             # descriptive t_var grouping from the record
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]


# --------------------------------------------------------------------------
# Pure functions (unit-tested in test_behavior_decode.py)
# --------------------------------------------------------------------------

def ci_state(ci):
    """Three mutually exclusive, exhaustive states from a [lo, hi] interval."""
    lo, hi = float(ci[0]), float(ci[1])
    if lo > 0:
        return "above"
    if hi <= 0:
        return "at/below"
    return "uncertain"


def target_table(rows, target):
    """rows: per-session dicts with 'dropped' (set of targets) and 'deltas'
    {target: {...}}. Returns N / eligible n / dropped n and the state counts."""
    N = len(rows)
    elig = [r for r in rows if target not in r["dropped"] and target in r["deltas"]]
    dropped = [r for r in rows if target in r["dropped"]]
    out = {"N": N, "eligible": len(elig), "dropped": len(dropped),
           "dropped_sessions": [(r["eid"][:8], r["dropped"][target]) for r in dropped],
           "missing": N - len(elig) - len(dropped)}
    for which, key in (("null", "delta_null_ci"), ("linear", "delta_linear_ci")):
        states = [ci_state(r["deltas"][target][key]) for r in elig]
        out[f"{which}_states"] = {s: states.count(s) for s in STATES}
        out[f"{which}_frac"] = {s: (states.count(s) / len(elig) if elig else float("nan"))
                                for s in STATES}
    for k in ("probe", "linear", "null"):
        v = np.array([r["deltas"][target]["point"][k] for r in elig], dtype=float)
        out[k] = summary(v)
    for k in ("delta_null", "delta_linear"):
        v = np.array([r["deltas"][target][k] for r in elig], dtype=float)
        out[k] = summary(v)
    return out


def summary(v):
    if len(v) == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    return {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75))}


def trigger_block(rows):
    """Trigger 1. Eligible: block not dropped and >= MIN_HALF_TRIALS p(left)=0.5
    test trials. Fires if > 50% of eligible have position recall(0.5) >=
    probe recall(0.5)."""
    elig, excluded, hits = [], [], []
    for r in rows:
        b = r.get("block_control")
        if "block" in r["dropped"] or b is None or b["n_half_test"] < MIN_HALF_TRIALS:
            excluded.append(r["eid"][:8])
            continue
        elig.append(r["eid"][:8])
        if b["recall_half_position"] >= b["recall_half_probe"]:
            hits.append(r["eid"][:8])
    frac = len(hits) / len(elig) if elig else float("nan")
    return {"eligible": len(elig), "excluded": excluded, "hits": hits, "frac": frac,
            "fires": bool(elig) and frac > 0.5}


def trigger_tooling(rows):
    """Trigger 2. Fires if any audit failed or any non-dropped target has a
    non-finite primary metric (probe, linear or null). Dropped targets never count."""
    bad = []
    for r in rows:
        if not r["audit_ok"]:
            bad.append((r["eid"][:8], "audit"))
        for t in dc.TARGETS:
            if t in r["dropped"]:
                continue
            d = r["deltas"].get(t)
            if d is None:
                bad.append((r["eid"][:8], f"{t}: no primary metric"))
                continue
            for k in ("probe", "linear", "null"):
                if not math.isfinite(d["point"][k]):
                    bad.append((r["eid"][:8], f"{t}: non-finite {k}"))
    return {"failures": bad, "fires": bool(bad)}


def trigger_contradiction(rows):
    """Trigger 3. Eligible: co-smoothing band '>0' and >= 3 non-dropped targets.
    Fires if > 50% of eligible are at/below null (Delta_null CI hi <= 0) on >= 3
    of their non-dropped targets. 'Uncertain' never counts."""
    elig, hits = [], []
    for r in rows:
        avail = [t for t in dc.TARGETS if t not in r["dropped"] and t in r["deltas"]]
        if r["cosmooth_band"] != ">0" or len(avail) < 3:
            continue
        elig.append(r["eid"][:8])
        below = sum(ci_state(r["deltas"][t]["delta_null_ci"]) == "at/below" for t in avail)
        if below >= 3:
            hits.append(r["eid"][:8])
    frac = len(hits) / len(elig) if elig else float("nan")
    return {"eligible": len(elig), "hits": hits, "frac": frac,
            "fires": bool(elig) and frac > 0.5}


def position_baseline(pos, tg_block, seed=42):
    """Decode block from trial position alone, with the protocol's split roles:
    standardise and fit on train, choose C on val (cross-entropy), test once.
    pos: {split: trial index per row (parquet row order)}. Returns test labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    rows = {s: tg_block[s]["mask"] for s in dc.SPLITS}
    mu, sd = pos["train"][rows["train"]].mean(), pos["train"][rows["train"]].std() or 1.0
    X = {s: ((pos[s] - mu) / sd).reshape(-1, 1) for s in dc.SPLITS}
    y = {s: tg_block[s]["y"] for s in dc.SPLITS}
    classes = np.arange(dc.N_CLASSES["block"])
    best = None
    for C in C_GRID:
        m = LogisticRegression(C=C, max_iter=1000, random_state=seed)
        m.fit(X["train"][rows["train"]], y["train"][rows["train"]])
        proba = np.zeros((rows["val"].sum(), len(classes)))
        proba[:, m.classes_] = m.predict_proba(X["val"][rows["val"]])
        ce = log_loss(y["val"][rows["val"]], proba, labels=classes)
        if best is None or ce < best[0]:
            best = (ce, C, m)
    return best[2].predict(X["test"]), {"C": best[1], "val_ce": float(best[0])}


def half_recall(y, pred, mask):
    rows = mask & (y == BLOCK_HALF)
    return float((np.asarray(pred)[rows] == BLOCK_HALF).mean()) if rows.any() else float("nan")


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------

def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or len(set(x[ok])) < 2 or len(set(y[ok])) < 2:
        return None, int(ok.sum())
    r, p = spearmanr(x[ok], y[ok])
    return (float(r), float(p)), int(ok.sum())


def load_session(eid, cohort, root, data_root):
    from behavior_decode import read_behaviour
    probe_dir, plot_dir = root / "probe" / eid, root / "plots" / eid
    audit_p = plot_dir / "plot_audit.json"
    row = {"eid": eid, "cohort": cohort, "dev": eid in DEV, "dropped": {}, "deltas": {},
           "audit_ok": False}
    if not audit_p.is_file():
        return row
    a = json.loads(audit_p.read_text())
    row["audit_ok"] = True       # plot_audit.json is written only after every audit passed
    row["dropped"] = a["dropped"]
    row["deltas"] = a["deltas"]

    s = json.loads((RESULTS / "learnability" / "own_session" / eid / "own_session_summary.json")
                   .read_text())["variants"]["best"]
    row["cosmooth_band"] = s["co_smooth"]["band"]
    row["co_bps"] = s["co_smooth"]["mean_bps"]
    row["fp_bps"] = s["forward_pred"]["mean_bps"]
    row["t_var"] = s["co_smooth"]["median_temporal_variation_ratio"]
    row["fp_band"] = s["forward_pred"]["band"]

    if "block" not in row["dropped"]:
        raw = read_behaviour(eid, data_root)
        tg = dc.prepare_targets(raw)["targets"]
        tb = {sp: tg[sp]["block"] for sp in dc.SPLITS}
        splits = np.load(Path(data_root) / eid / "splits.npz")
        pos = {sp: splits[f"{sp}_trial_indices"].astype(float) for sp in dc.SPLITS}
        for sp in dc.SPLITS:
            assert len(pos[sp]) == len(raw[sp]["block"]), f"{eid[:8]} {sp}: positions vs rows"
        pp = np.load(probe_dir / "preds.npz")
        pred_pos, sel = position_baseline(pos, tb)
        yt, mt = tb["test"]["y"], tb["test"]["mask"]
        row["block_control"] = {
            "n_half_test": int((mt & (yt == BLOCK_HALF)).sum()),
            "recall_half_probe": half_recall(yt, pp["block_pred"], mt),
            "recall_half_position": half_recall(yt, pred_pos, mt),
            "balanced_acc_position": dc.static_metrics(yt[mt], pred_pos[mt])["balanced_acc"],
            "position_selection": sel,
        }
    return row


def fmt_frac(states, n):
    return "  ".join(f"{s} {states[s]}/{n}" for s in STATES)


def report_population(rows, label, lines):
    lines.append(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    for cohort in ("bwm", "rs", "pooled (exploratory)"):
        rs = rows if cohort.startswith("pooled") else [r for r in rows if r["cohort"] == cohort]
        lines.append(f"\n--- {cohort}: N = {len(rs)} "
                     f"({sum(r['dev'] for r in rs)} development) ---")
        for t in dc.TARGETS:
            tt = target_table(rs, t)
            lines.append(f"  {t} [{dc.PRIMARY[t]}]  population N {tt['N']} / eligible n "
                         f"{tt['eligible']} / dropped n {tt['dropped']}"
                         + (f" / missing {tt['missing']}" if tt["missing"] else ""))
            for k in ("null", "probe", "linear", "delta_null", "delta_linear"):
                v = tt[k]
                lines.append(f"      {k:13s} median {v['median']:+.3f}  IQR [{v['q1']:+.3f}, {v['q3']:+.3f}]")
            lines.append(f"      probe vs null:   {fmt_frac(tt['null_states'], tt['eligible'])}")
            lines.append(f"      probe vs linear: {fmt_frac(tt['linear_states'], tt['eligible'])}")
            if tt["dropped_sessions"]:
                lines.append(f"      dropped: {tt['dropped_sessions']}")
        bc = [r["block_control"] for r in rs if r.get("block_control")]
        if bc:
            lines.append(f"  block control (trial position only), n {len(bc)}: balanced acc median "
                         f"{np.median([b['balanced_acc_position'] for b in bc]):.3f}; "
                         f"recall(0.5) position median {np.nanmedian([b['recall_half_position'] for b in bc]):.3f} "
                         f"vs probe {np.nanmedian([b['recall_half_probe'] for b in bc]):.3f}")


def relation(rows, label, lines):
    lines.append(f"\n--- relation to neural prediction: {label} ---")
    for cohort in ("bwm", "rs"):
        rs = [r for r in rows if r["cohort"] == cohort]
        lines.append(f"  {cohort} (N {len(rs)}): Spearman r (p, n)")
        for t in dc.TARGETS:
            el = [r for r in rs if t not in r["dropped"] and t in r["deltas"]]
            cells = []
            for dk in ("delta_null", "delta_linear"):
                for cov in ("co_bps", "fp_bps", "t_var"):
                    res, n = spearman([r[cov] for r in el], [r["deltas"][t][dk] for r in el])
                    cells.append(f"{dk[6:]}~{cov} " + (f"{res[0]:+.2f} ({res[1]:.2f}, {n})" if res else f"n/a ({n})"))
            lines.append(f"    {t[:12]:12s} " + "; ".join(cells))
        for t in dc.TARGETS:
            el = [r for r in rs if t not in r["dropped"] and t in r["deltas"]]
            tab = {}
            for r in el:
                key = (r["cosmooth_band"], ci_state(r["deltas"][t]["delta_null_ci"]))
                tab[key] = tab.get(key, 0) + 1
            flat = {}
            for r in el:
                key = ("time-flat" if r["t_var"] <= TIME_FLAT else "not flat",
                       ci_state(r["deltas"][t]["delta_null_ci"]))
                flat[key] = flat.get(key, 0) + 1
            lines.append(f"    {t[:12]:12s} co-band x probe-vs-null: "
                         + ", ".join(f"{b}|{s}:{tab.get((b, s), 0)}" for b in (">0", "~0", "<0") for s in STATES))
            lines.append(f"    {'':12s} time-flat x probe-vs-null: "
                         + ", ".join(f"{b}|{s}:{flat.get((b, s), 0)}" for b in ("time-flat", "not flat") for s in STATES))


def triggers(rows, lines, fire=True):
    out = {}
    for cohort in ("bwm", "rs"):
        rs = [r for r in rows if r["cohort"] == cohort]
        t1, t2, t3 = trigger_block(rs), trigger_tooling(rs), trigger_contradiction(rs)
        out[cohort] = {"block_confound": t1, "tooling": t2, "contradiction": t3}
        tag = "" if fire else "  (reference only, fires nothing)"
        lines.append(f"  {cohort}: T1 block confound {'FIRES' if t1['fires'] else 'no'} "
                     f"({len(t1['hits'])}/{t1['eligible']} eligible, excluded {t1['excluded']}){tag}")
        lines.append(f"  {cohort}: T2 tooling {'FIRES' if t2['fires'] else 'no'} {t2['failures'] or ''}{tag}")
        lines.append(f"  {cohort}: T3 contradiction {'FIRES' if t3['fires'] else 'no'} "
                     f"({len(t3['hits'])}/{t3['eligible']} eligible) {t3['hits'] or ''}{tag}")
    return out


def strip_figure(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(dc.TARGETS), 2, figsize=(12, 2.4 * len(dc.TARGETS)),
                             constrained_layout=True)
    for i, t in enumerate(dc.TARGETS):
        for j, (dk, ck, lab) in enumerate((("delta_null", "delta_null_ci", "ΔNull"),
                                           ("delta_linear", "delta_linear_ci", "ΔLinear"))):
            ax = axes[i, j]
            el = [r for r in rows if t not in r["dropped"] and t in r["deltas"]]
            el.sort(key=lambda r: (r["cohort"], r["deltas"][t][dk]))
            for k, r in enumerate(el):
                d = r["deltas"][t]
                col = {"bwm": "#1f5fbf", "rs": "#e07b00"}[r["cohort"]]
                ax.plot(d[ck], [k, k], color=col, lw=1, alpha=0.8)
                ax.plot(d[dk], k, "D" if r["dev"] else "o", ms=4, color=col,
                        mfc="white" if r["dev"] else col)
            ax.axvline(0, color="black", lw=0.8)
            ax.set_yticks([])
            ax.set_title(f"{t}  {lab}  (n={len(el)}; blue bwm, orange rs; open = dev)", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="decoding_v2")
    ap.add_argument("--data-root", default=DATA_ROOT)
    args = ap.parse_args()
    root = RESULTS / args.dir

    m = json.loads(MANIFEST.read_text())
    sessions = {s["eid"]: s["cohort"] for s in m["sessions"]}
    assert len(sessions) == 52 and STRESS not in sessions
    assert all(e in sessions for e in DEV)
    complete = [load_session(e, c, root, args.data_root) for e, c in sorted(sessions.items())]
    primary = [r for r in complete if not r["dev"]]
    cnt = lambda rs, c: sum(r["cohort"] == c for r in rs)  # noqa: E731
    assert len(primary) == 48 and cnt(primary, "bwm") == 34 and cnt(primary, "rs") == 14
    assert len(complete) == 52 and cnt(complete, "bwm") == 36 and cnt(complete, "rs") == 16
    assert not any(r["eid"] in DEV for r in primary)
    assert not any(r["eid"] == STRESS for r in complete)

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    lines = [f"Behaviour decoding aggregate ({args.dir}) at {sha}",
             "Status from paired 95% CIs (plot_audit.json): above lo>0 | uncertain lo<=0<hi | at/below hi<=0",
             f"Sessions without plot_audit.json (counted as missing / audit failure): "
             f"{[r['eid'][:8] for r in complete if not r['audit_ok']]}"]
    report_population(primary, "PRIMARY: 48 untouched sessions (34 bwm + 14 rs)", lines)
    relation(primary, "PRIMARY 48", lines)
    lines.append(f"\n{'=' * 78}\nCLOSURE TRIGGERS (primary 48, per cohort; fires if either cohort fires)\n{'=' * 78}")
    trig = triggers(primary, lines)
    fired = sorted({k for c in trig.values() for k, v in c.items() if v["fires"]})
    lines.append(f"  => {'FIRED: ' + ', '.join(fired) if fired else 'no trigger fired: Tier 4 closes'}")

    report_population(complete, "COMPLETENESS: all 52 (36 bwm + 16 rs); dev sessions marked", lines)
    relation(complete, "COMPLETENESS 52", lines)
    lines.append("\nTriggers on the 52 (reference only):")
    trig52 = triggers(complete, lines, fire=False)

    out = root / "aggregate"
    out.mkdir(exist_ok=True)
    rep = out / f"report_{sha}.txt"
    if rep.exists():
        print(f"ERROR: {rep} exists"); return 2
    rep.write_text("\n".join(lines) + "\n")
    (out / f"aggregate_{sha}.json").write_text(json.dumps(
        {"sha": sha, "triggers_primary": trig, "triggers_complete_reference": trig52, "fired": fired,
         "sessions": [{k: v for k, v in r.items() if k != "deltas"} for r in complete]},
        indent=2, default=str))
    strip_figure(primary, out / f"deltas_primary_{sha}.png", "Primary 48: paired Δ with 95% CI")
    strip_figure(complete, out / f"deltas_complete_{sha}.png", "All 52 (open markers = development)")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
