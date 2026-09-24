#!/usr/bin/env python
"""Learnability characterisation report.

    python script/learnability_report.py

Population rule, enforced in code rather than by layout:
  * PRIMARY population = the manifest's 52 'sessions' (36 bwm + 16 rs). Every
    count, percentage and correlation is computed over these only.
  * Pilot sources NOT in the sample (manifest 'pilot_reference') appear in a
    separate table and never enter a statistic. Asserted below.
  * Cohorts first, separately. bwm is a balanced stratified sample (4 per 3x3
    cell), so "X/36" describes the stratified characterisation sample, not the
    natural frequency among all 303 held-in bwm sessions. rs is all measurable
    held-in rs. A pooled analysis follows, labelled EXPLORATORY.
Descriptive only: no threshold is derived from anything printed here.

The band rule is unchanged (learnability.BAND_RULE). Next to every band the
report prints the CI WIDTH (hi - lo), because '~0' means only "the 95% CI
contains 0". A narrow CI around 0 is close to the mean-rate null; a wide one is
"too uncertain to tell". The learnability canary f140a2ec [best] was '~0' with a
CI of [-642.8, +0.02], its neuron-mean dragged down by near-silent neurons.
No threshold is applied to the width. It is printed so a '~0' cannot be read as
"mean-like" without it.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "script" / "learnability_manifest.json"
EVAL_ROOT = Path("/work/nvme/beml/ac136/IBL_MTM/results/learnability/own_session")
PILOT_EVAL = Path("/work/nvme/beml/ac136/IBL_MTM/results/pilot/own_session")
BANDS = (">0", "~0", "<0")
COVARS = [("clusters", lambda s: s["clusters"]), ("n_train", lambda s: s["n_train"]),
          ("mean_rate", lambda s: s["covariates"]["mean_rate_per_bin"]),
          ("frac_zero", lambda s: s["covariates"]["frac_zero_bins"]),
          ("silent", lambda s: s["covariates"]["silent_neurons"]),
          ("rate_cv", lambda s: s["covariates"]["rate_cv_across_neurons"]),
          ("loss_drop", lambda s: s["curve"]["loss_drop_frac"]),
          ("drop_by_ep50", lambda s: s["curve"]["frac_of_drop_by_epoch_50"]),
          ("final_slope", lambda s: s["curve"]["final_50_loss_slope"]),
          ("best_epoch", lambda s: s["curve"]["best_epoch"]),
          ("best_metric", lambda s: s["curve"]["best_metric"])]
OUTCOMES = [("bps", lambda s: s["variants"]["best"]["co_smooth"]["mean_bps"]),
            ("t_var", lambda s: s["variants"]["best"]["co_smooth"]["median_temporal_variation_ratio"])]


def co(s, role="best"):
    return s["variants"][role]["co_smooth"]


def ci_width(b):
    return b["ci95_hi"] - b["ci95_lo"]


LEGEND = ("Band = own-session co-smoothing bps vs the mean-rate null, from the fixed rule "
          "(learnability.BAND_RULE).\n'~0' = the 95% CI contains 0: NOT distinguishable from "
          "the null. It does NOT mean 'mean-like'; read it\nwith ci_w (CI width, hi-lo) and "
          "t_var. No threshold is applied to ci_w; it is descriptive.")


def band_counts(rows, role):
    c = {b: 0 for b in BANDS}
    for s in rows:
        c[co(s, role)["band"]] = c.get(co(s, role)["band"], 0) + 1
    return c


def correlations(rows, label):
    print(f"\n  Spearman correlations ({label}, n={len(rows)}; descriptive only):")
    print(f"    {'covariate':13s} " + "  ".join(f"{o:>16s}" for o, _ in OUTCOMES))
    for name, f in COVARS:
        cells = []
        for _, g in OUTCOMES:
            x = np.array([f(s) for s in rows], dtype=float)
            y = np.array([g(s) for s in rows], dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 4 and len(set(x[ok])) > 1:
                r, p = spearmanr(x[ok], y[ok])
                cells.append(f"{r:+.2f} (p={p:.2f})")
            else:
                cells.append("n/a")
        print(f"    {name:13s} " + "  ".join(f"{c:>16s}" for c in cells))


def cohort_block(rows, cohort, n_expected, phrase):
    print(f"\n=== {cohort.upper()} — {phrase} ===")
    print(f"  sessions reported: {len(rows)}/{n_expected}")
    for role in ("best", "last"):
        c = band_counts(rows, role)
        print(f"  [{role}] co-smoothing band: " + ", ".join(
            f"{c[b]}/{n_expected} {b}" for b in BANDS) + f"  ({phrase})")
        near = sorted((ci_width(co(s, role)), s["eid"][:8], co(s, role)["mean_bps"])
                      for s in rows if co(s, role)["band"] == "~0")
        if near:
            print(f"    '~0' sessions [{role}], by CI width: " + ", ".join(
                f"{e} (bps {m:+.3f}, ci_w {w:.3f})" for w, e, m in near))
    moved = [s["eid"][:8] for s in rows if co(s, "best")["band"] != co(s, "last")["band"]]
    print(f"  band differs best vs last: {len(moved)}/{len(rows)} {moved if moved else ''}")
    if cohort == "bwm":
        print("  per 3x3 cell (best):")
        cells = {}
        for s in rows:
            cells.setdefault(s["cell"], []).append(co(s)["band"])
        for k in sorted(cells):
            b = cells[k]
            print(f"    {k:30s} " + "  ".join(f"{x}:{b.count(x)}" for x in BANDS) + f"  (of {len(b)})")
    if len(rows) >= 4:
        correlations(rows, cohort)


def main():
    m = json.loads(MANIFEST.read_text())
    primary = {s["eid"]: s for s in m["sessions"]}
    reference = {r["eid"] for r in m["pilot_reference"]}
    assert not (set(primary) & reference), "a reference source is in the primary population"

    rows, missing, failing = [], [], []
    for eid, entry in primary.items():
        p = EVAL_ROOT / eid / "own_session_summary.json"
        if not p.is_file():
            missing.append(eid); continue
        s = json.loads(p.read_text())
        assert s["eid"] == eid and s.get("mode") == "manifest", f"unexpected summary at {p}"
        s["cell"] = entry["cell"]
        rows.append(s)
        if s["n_checks_failed"]:
            failing.append(eid)
    stray = {p.parent.name for p in EVAL_ROOT.glob("*/own_session_summary.json")} - set(primary)
    # Enforce the population rule: nothing outside the 52 enters a statistic.
    assert all(s["eid"] in primary for s in rows)
    assert not any(s["eid"] in reference for s in rows)

    print("Learnability characterisation — primary population: the manifest's "
          f"{len(primary)} sessions ({m['n_bwm']} bwm + {m['n_rs']} rs)")
    print(f"reported {len(rows)}/{len(primary)}; missing {len(missing)}; "
          f"with failed checks {len(failing)} {[e[:8] for e in failing]}")
    if stray:
        print(f"NOTE: {len(stray)} summaries outside the manifest ignored: {sorted(stray)}")

    print("\n" + LEGEND)
    print("\n=== per-session table (primary population) ===")
    print(f"{'eid':9s} {'coh':4s} {'cell':28s} {'clust':>5s} {'train':>5s} {'band':>4s} "
          f"{'bps':>8s} {'ci95':>19s} {'ci_w':>8s} {'med':>7s} {'t_var':>6s} {'last':>4s} {'reuse':>5s}")
    for s in sorted(rows, key=lambda r: (r["cohort"], r["cell"], r["eid"])):
        b = co(s)
        print(f"{s['eid'][:8]:9s} {s['cohort']:4s} {s['cell']:28s} {s['clusters']:>5d} "
              f"{s['n_train']:>5d} {b['band']:>4s} {b['mean_bps']:>+8.4f} "
              f"[{b['ci95_lo']:+.4f},{b['ci95_hi']:+.4f}] {ci_width(b):>8.3f} {b['median_bps']:>+7.4f} "
              f"{b['median_temporal_variation_ratio']:>6.3f} {co(s, 'last')['band']:>4s} "
              f"{'yes' if s['reused_pilot_pretrain'] else '':>5s}")

    cohort_block([s for s in rows if s["cohort"] == "bwm"], "bwm", m["n_bwm"],
                 "bwm sessions in the stratified characterisation sample")
    cohort_block([s for s in rows if s["cohort"] == "rs"], "rs", m["n_rs"],
                 "all measurable held-in rs sessions")

    print("\n=== POOLED 52 — EXPLORATORY (secondary; mixes a stratified and a census sample) ===")
    if len(rows) >= 4:
        correlations(rows, "pooled, exploratory")

    print("\n=== PILOT REFERENCE — context only, excluded from every statistic above ===")
    for r in m["pilot_reference"]:
        p = PILOT_EVAL / r["eid"] / "own_session_summary.json"
        if p.is_file():
            t = json.loads(p.read_text())["variants"]["trained"]["co_smooth"]
            print(f"  {r['eid'][:8]} {r['cohort']:4s} band {t['band']:>3s} bps {t['mean_bps']:+.4f} "
                  f"CI [{t['ci95_lo']:+.4f},{t['ci95_hi']:+.4f}] ci_w {ci_width(t):.3f} t_var "
                  f"{t['median_temporal_variation_ratio']:.3f}  (pilot Stage 3-diag, best only)")
        else:
            print(f"  {r['eid'][:8]} — no pilot own-session summary found")
    if missing:
        print(f"\nINCOMPLETE: {len(missing)} of {len(primary)} sessions have no summary yet")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
