#!/usr/bin/env python
"""Per-session diagnostic plots for the behaviour-decoding exploratory run.

    python script/decode_plots.py EID [--dir decoding_v2]

Writes to ${RESULTS}/<dir>/plots/<eid>/:
  decoding_summary.png   null / probe / linear per primary metric (four panels;
                         header: eid and cohort). The paired deltas and their CIs
                         are in plot_audit.json and the GT/pred titles
  wheel_gt_pred.png      8 GT-stratified test trials: GT, probe, raw-spike linear
  wme_gt_pred.png
  choice_confusion.png   row-normalised 2x2 confusion, probe | linear
  block_recall.png       per-class recall, probe vs linear (mean = balanced acc)
  plot_audit.json        audits, deltas and CIs, selected trials

Nothing is retrained. Probe predictions come from the run's preds.npz. The
raw-spike linear baseline saved metrics only in v2, so it is REFIT with its own
code (train fit, val selection) and audited against the saved run first:
chosen C/alpha and every categorical metric must match exactly; continuous
quantities must match to 1e-9 (the saved run's BLAS threading was not pinned,
so bit equality is not guaranteed). Any audit failure: exit 3, no plots.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import decode_components as dc  # noqa: E402
from behavior_decode import DATA_ROOT, RESULTS, read_behaviour  # noqa: E402
import decode_baseline as db  # noqa: E402

ATOL = 1e-9
PRED_RTOL = 1e-7
LABEL = {"choice": "Choice accuracy", "block": "Block balanced accuracy",
         "wheel-speed": "Wheel trial R²", "whisker-motion-energy": "WME trial R²"}
SHORT = {"wheel-speed": "wheel", "whisker-motion-energy": "wme"}
COLORS = {"null": "#8c8c8c", "probe": "#1f5fbf", "linear": "#e07b00", "gt": "black"}
# Bin centres as bin_behavior samples them: right edges of 20 ms bins from -0.5 s.
TIME = -0.5 + 0.02 * np.arange(1, 101)


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "[n/a]"


def audit_close(a, b):
    diff = abs(float(a) - float(b))
    return {"exact": float(a) == float(b), "within_tol": diff <= ATOL, "max_diff": diff}


def run_audits(eid, probe_m, probe_p, base_m, tg, cond, X):
    """Recompute everything plotted; raise on any mismatch. Returns (audit, linear preds)."""
    audit = {"probe": {}, "linear": {}, "chance": {}}
    bad = []
    names = [n for n in dc.TARGETS if n in probe_m["metrics"]]
    linear = {}
    for n in names:
        # probe: preds.npz reproduces metrics.json, and its y/mask are today's targets
        same_tg = (np.array_equal(probe_p[f"{n}_y"], tg["test"][n]["y"])
                   and np.array_equal(probe_p[f"{n}_mask"], tg["test"][n]["mask"]))
        rec = dc.metrics_for(n, tg["test"][n], probe_p[f"{n}_pred"], cond)
        saved = probe_m["metrics"][n]
        keys = [k for k, v in saved.items() if isinstance(v, float)]
        exact = same_tg and all(rec[k] == saved[k] for k in keys)
        audit["probe"][n] = {"targets_identical": same_tg, "metrics_exact": exact}
        if not exact:
            bad.append(f"probe {n}: recomputed metrics differ")

        # linear: refit, then compare with the saved baseline run
        if n in dc.STATIC:
            pred, sel = db.fit_static(n, X, tg, seed=42)
            hyper = "C"
        else:
            pred, sel = db.fit_dynamic(n, X, tg)
            hyper = "alpha"
        linear[n] = pred
        saved_lin = base_m["results"][n]["linear"]
        rec_lin = dc.metrics_for(n, tg["test"][n], pred, cond)
        a = {"chosen_" + hyper: {"saved": saved_lin["selection"]["chosen"][hyper],
                                 "refit": sel["chosen"][hyper]}}
        if sel["chosen"][hyper] != saved_lin["selection"]["chosen"][hyper]:
            bad.append(f"linear {n}: chosen {hyper} differs")
        for k in [k for k, v in saved_lin.items() if isinstance(v, float)]:
            a[k] = audit_close(rec_lin[k], saved_lin[k])
            categorical = n in dc.STATIC            # accuracies are counts over labels
            if (categorical and not a[k]["exact"]) or not a[k]["within_tol"]:
                bad.append(f"linear {n} {k}: refit {rec_lin[k]} vs saved {saved_lin[k]}")
        vkey = "val_ce" if n in dc.STATIC else "val_mse"
        vd = [audit_close(r[vkey], s[vkey]) for r, s in
              zip(sel["grid"], saved_lin["selection"]["grid"])]
        a["val_grid"] = {"exact": all(d["exact"] for d in vd),
                         "max_diff": max(d["max_diff"] for d in vd)}
        if not all(d["max_diff"] <= 1e-7 * max(1.0, abs(s[vkey])) for d, s in
                   zip(vd, saved_lin["selection"]["grid"])):
            bad.append(f"linear {n}: val grid losses differ beyond tolerance")
        audit["linear"][n] = a

        # chance: train-only null reproduces the saved chance metrics exactly
        null_pred, _ = dc.chance_predictions(n, tg["train"][n], len(pred))
        rec_ch = dc.metrics_for(n, tg["test"][n], null_pred, cond)
        saved_ch = base_m["results"][n]["chance"]
        ch_ok = all(rec_ch[k] == saved_ch[k] for k, v in saved_ch.items() if isinstance(v, float))
        audit["chance"][n] = {"metrics_exact": ch_ok}
        if not ch_ok:
            bad.append(f"chance {n}: recomputed metrics differ")
    if bad:
        raise SystemExit("AUDIT FAILED:\n  " + "\n  ".join(bad))
    return audit, linear


def session_cohort(eid):
    """'BWM' or 'RS', from the learnability manifest, else the pilot sources."""
    import stage3_preflight as pf
    manifest = json.loads((REPO / "script" / "learnability_manifest.json").read_text())
    cohort = {s["eid"]: s["cohort"] for s in manifest["sessions"]}.get(eid)
    if cohort is None and eid in pf.SOURCES:
        cohort = pf.SOURCES[eid][0]
    return (cohort or "unknown").upper()


def signed(x):
    return f"{x:+.3f}".replace("-", "−")


def short_header(eid, d):
    """Three lines: session and cohort; primary metric for probe, linear and
    null; the two paired deltas (point estimates)."""
    p = d["point"]
    return (f"{eid[:8]} — {session_cohort(eid)}\n"
            f"Probe {p['probe']:.3f} | Linear {p['linear']:.3f} | Null {p['null']:.3f}\n"
            f"ΔNull {signed(d['delta_null'])} | ΔLinear {signed(d['delta_linear'])}")


def plot_summary(path, eid, deltas, dropped):
    """Four panels, one per primary metric: Null, Probe and Linear as points.
    Header: the session eid and its cohort only."""
    rows = list(dc.TARGETS)
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.5, 2.1 * len(rows)))
    for ax_v, n in zip(axes, rows):
        if n in dropped:
            ax_v.set_facecolor("#eeeeee"); ax_v.set_xticks([]); ax_v.set_yticks([])
            ax_v.text(0.5, 0.5, f"{LABEL[n]}: DROPPED on train\n{dropped[n]}",
                      ha="center", va="center", transform=ax_v.transAxes, color="#555555")
            continue
        d = deltas[n]
        pts = [("Null", d["point"]["null"], "null"), ("Probe", d["point"]["probe"], "probe"),
               ("Linear", d["point"]["linear"], "linear")]
        for i, (lab, v, key) in enumerate(pts):
            ax_v.plot(v, 2 - i, "o", ms=9, color=COLORS[key])
            ax_v.annotate(f"{v:.3f}", (v, 2 - i), textcoords="offset points", xytext=(8, -4),
                          fontsize=9, color=COLORS[key])
        vals = [p[1] for p in pts]
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        pad = 0.15 * (hi - lo if hi > lo else 1.0)
        ax_v.set_xlim(lo - pad, hi + 2.5 * pad)
        ax_v.axvline(0, color="#cccccc", lw=0.8, zorder=0)
        ax_v.set_yticks([2, 1, 0]); ax_v.set_yticklabels([p[0] for p in pts])
        ax_v.set_ylim(-0.6, 2.6)
        ax_v.set_title(LABEL[n], fontsize=10, loc="left")
    fig.suptitle(f"{eid}  ({session_cohort(eid)})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gt_pred(path, eid, n, tg, probe_pred, linear_pred, d):
    y, mask = tg["test"][n]["y"], tg["test"][n]["mask"]
    chosen, full = dc.select_trials(y, mask, k=8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes.ravel()):
        if j >= len(chosen):
            ax.axis("off"); continue
        i = chosen[j]
        m = mask[i]
        gt = np.where(m, y[i], np.nan)
        ax.plot(TIME, gt, color=COLORS["gt"], lw=1.6, label="GT")
        ax.plot(TIME, probe_pred[i], color=COLORS["probe"], lw=1.2, label="probe")
        ax.plot(TIME, linear_pred[i], color=COLORS["linear"], lw=1.2, label="linear")
        ax.axvline(0, color="#999999", lw=0.8, ls="--")
        ax.set_title(f"test trial {i}  (GT-activity bin {j + 1}/8)"
                     + ("" if full[j] else "  [partially valid]"), fontsize=9)
        if j % 4 == 0:
            ax.set_ylabel("Behavior (scaled using train min/max)", fontsize=8)
        if j >= 4:
            ax.set_xlabel("time from stimOn (s)", fontsize=8)
        if j == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(short_header(eid, d), fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return chosen.tolist(), full.tolist()


SIMPLE_CLASS = {"choice": {0: "Right", 1: "Left"},
                "block": {0: "p(left)=0.2", 1: "p(left)=0.5", 2: "p(left)=0.8"}}


def plot_choice_confusion(path, eid, tg, probe_pred, linear_pred, d):
    """Two row-normalised 2x2 matrices, Probe | Linear. Rows: true choice."""
    n = "choice"
    rows = dc.evaluable(n, tg["test"][n])
    y = tg["test"][n]["y"][rows]
    names = [SIMPLE_CLASS[n][c] for c in range(2)]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2), constrained_layout=True)
    for j, (ax, (lab, pred)) in enumerate(zip(axes, (("Probe", probe_pred), ("Linear", linear_pred)))):
        pr = np.asarray(pred)[rows]
        counts, norm = dc.row_normalised_confusion(y, pr, 2)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{norm[a, b]:.0%}\n({counts[a, b]})", ha="center", va="center",
                        fontsize=12, color="white" if norm[a, b] > 0.6 else "black")
        ax.set_title(lab, fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=11)
        ax.set_yticks([0, 1]); ax.set_yticklabels(names if j == 0 else ["", ""], fontsize=11)
        ax.set_xlabel("Predicted choice", fontsize=11)
        if j == 0:
            ax.set_ylabel("True choice", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.75, label="share of true choice")
    fig.suptitle(short_header(eid, d), fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_block_recall(path, eid, tg, probe_pred, linear_pred, d):
    """Per true block class: Probe vs Linear recall side by side. Balanced
    accuracy is exactly the mean of each model's three bars."""
    n = "block"
    rows = dc.evaluable(n, tg["test"][n])
    y = tg["test"][n]["y"][rows]
    counts = np.bincount(y, minlength=3)
    x = np.arange(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    for j, (lab, pred, key) in enumerate((("Probe", probe_pred, "probe"), ("Linear", linear_pred, "linear"))):
        pr = np.asarray(pred)[rows]
        rec = np.diag(dc.row_normalised_confusion(y, pr, 3)[1])
        xs = x + (j - 0.5) * w
        ax.bar(xs, rec, width=w, color=COLORS[key], label=lab)
        for xv, v in zip(xs, rec):
            ax.text(xv, v + 0.015, f"{v:.0%}", ha="center", fontsize=10)
    ax.axhline(1 / 3, color="black", ls="--", lw=1, label="Chance (1/3)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{SIMPLE_CLASS[n][c]}\nn={counts[c]}" for c in range(3)], fontsize=11)
    ax.set_xlabel("True block", fontsize=11)
    ax.set_ylabel("Recall (share of class decoded correctly)", fontsize=10)
    ax.set_ylim(0, 1.28)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(short_header(eid, d), fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_static(out, eid, tg, probe_p, linear, deltas):
    """choice_confusion.png and block_recall.png (replace any earlier versions)."""
    if "choice" in deltas:
        plot_choice_confusion(out / "choice_confusion.png", eid, tg, probe_p["choice_pred"],
                              linear["choice"], deltas["choice"])
    if "block" in deltas:
        plot_block_recall(out / "block_recall.png", eid, tg, probe_p["block_pred"],
                          linear["block"], deltas["block"])
    stale = out / "block_confusion.png"
    if stale.exists():
        stale.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--dir", default="decoding_v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--only", choices=["static", "summary", "panels"], default=None,
                    help="regenerate only the class plots (static: choice_confusion.png, "
                         "block_recall.png), decoding_summary.png, or panels (GT/pred plus "
                         "class plots) in an existing plot dir "
                         "(audits still run first)")
    args = ap.parse_args()
    eid = args.eid.removesuffix("_aligned")
    root = RESULTS / args.dir
    out = Path(args.out) if args.out else root / "plots" / eid
    if args.only is None and out.exists() and any(out.iterdir()):
        print(f"ERROR: {out} already holds plots"); return 2
    if args.only is not None and not out.is_dir():
        print(f"ERROR: --only needs an existing plot dir: {out}"); return 2

    probe_dir, base_dir = root / "probe" / eid, root / "baseline" / eid
    probe_m = json.loads((probe_dir / "metrics.json").read_text())
    selection = json.loads((probe_dir / "selection.json").read_text())
    probe_p = np.load(probe_dir / "preds.npz")
    base_m = json.loads((base_dir / "metrics.json").read_text())

    raw = read_behaviour(eid, args.data_root)
    prepared = dc.prepare_targets(raw)
    tg = prepared["targets"]
    cond = np.stack([raw["test"][c] for c in dc.PSTH_CONDITIONS], axis=1)
    X = {s: db.binned_spikes(eid, args.data_root, s) for s in dc.SPLITS}

    try:
        audit, linear = run_audits(eid, probe_m, probe_p, base_m, tg, cond, X)
    except SystemExit as e:
        print(e); return 3

    dropped = {n: prepared["info"][n]["dropped"] for n in dc.TARGETS if prepared["info"][n]["dropped"]}
    deltas = {}
    for n in dc.active_targets(prepared):
        null_pred, _ = dc.chance_predictions(n, tg["train"][n], len(linear[n]))
        deltas[n] = dc.paired_delta_ci(n, tg["test"][n], probe_p[f"{n}_pred"], linear[n], null_pred)
        # the plotted points are the saved numbers
        k = dc.PRIMARY[n]
        assert deltas[n]["point"]["probe"] == probe_m["metrics"][n][k]
        assert deltas[n]["point"]["null"] == base_m["results"][n]["chance"][k]
        assert abs(deltas[n]["point"]["linear"] - base_m["results"][n]["linear"][k]) <= ATOL

    if args.only == "static":
        plot_static(out, eid, tg, probe_p, linear, deltas)
        print(f"wrote class plots to {out}")
        return 0
    if args.only == "panels":
        for n in dc.DYNAMIC:
            if n in deltas:
                plot_gt_pred(out / f"{SHORT[n]}_gt_pred.png", eid, n, tg, probe_p[f"{n}_pred"],
                             linear[n], deltas[n])
        plot_static(out, eid, tg, probe_p, linear, deltas)
        print(f"wrote GT/pred and class plots to {out}")
        return 0
    if args.only == "summary":
        plot_summary(out / "decoding_summary.png", eid, deltas, dropped)
        print(f"wrote summary to {out}")
        return 0

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True, exist_ok=True)

    plot_summary(out / "decoding_summary.png", eid, deltas, dropped)
    selected = {}
    for n in dc.DYNAMIC:
        if n in deltas:
            selected[n] = plot_gt_pred(out / f"{SHORT[n]}_gt_pred.png", eid, n, tg,
                                       probe_p[f"{n}_pred"], linear[n], deltas[n])
    plot_static(out, eid, tg, probe_p, linear, deltas)

    (out / "plot_audit.json").write_text(json.dumps({
        "eid": eid, "dir": args.dir, "code": sha, "audit": audit,
        "deltas": {n: {k: v for k, v in d.items() if k != "draw_deltas"} for n, d in deltas.items()},
        "delta_ci_includes_zero": {
            n: {"null": bool(d["delta_null_ci"][0] <= 0 <= d["delta_null_ci"][1]),
                "linear": bool(d["delta_linear_ci"][0] <= 0 <= d["delta_linear_ci"][1])}
            for n, d in deltas.items() if "delta_null_ci" in d},
        "selected_trials": {n: {"indices": s[0], "fully_valid": s[1]} for n, s in selected.items()},
        "dropped": dropped}, indent=2))

    for n, d in deltas.items():
        print(f"  {n:22s} {d['metric']:13s} null {d['point']['null']:+.3f} probe {d['point']['probe']:+.3f} "
              f"linear {d['point']['linear']:+.3f} | Δnull {d['delta_null']:+.3f} {fmt_ci(d.get('delta_null_ci'))}"
              f" Δlinear {d['delta_linear']:+.3f} {fmt_ci(d.get('delta_linear_ci'))}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
