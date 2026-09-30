#!/usr/bin/env python
"""Per-session diagnostic plots for the behaviour-decoding exploratory run.

    python script/decode_plots.py EID [--dir decoding_v2]

Writes to ${RESULTS}/<dir>/plots/<eid>/:
  decoding_summary.png   null / probe / linear per primary metric, and the two
                         paired deltas (probe - null, probe - linear) with 95% CIs
                         -- the only uncertainty shown in any plot
  wheel_gt_pred.png      8 GT-stratified test trials: GT, probe, raw-spike linear
  wme_gt_pred.png
  choice_confusion.png   row-normalised confusion, probe | linear
  block_confusion.png
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


def plot_summary(path, eid, deltas, dropped, title):
    rows = list(dc.TARGETS)
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.1 * len(rows)),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    for r, n in enumerate(rows):
        ax_v, ax_d = axes[r]
        if n in dropped:
            for ax in (ax_v, ax_d):
                ax.set_facecolor("#eeeeee"); ax.set_xticks([]); ax.set_yticks([])
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

        items = [("Δ_null", d["delta_null"], d.get("delta_null_ci")),
                 ("Δ_linear", d["delta_linear"], d.get("delta_linear_ci"))]
        span = [0.0] + [v for _, v, _ in items] + [c for _, _, ci in items if ci for c in ci]
        for i, (lab, v, ci) in enumerate(items):
            y = 1 - i
            if ci:
                ax_d.plot(ci, [y, y], color="black", lw=2)
            ax_d.plot(v, y, "s", ms=7, color="black")
            ax_d.annotate(f"{lab} = {v:+.3f} {fmt_ci(ci)}", (1.0, y), xycoords=("axes fraction", "data"),
                          xytext=(-4, 7), textcoords="offset points", ha="right", fontsize=9)
        ax_d.axvline(0, color="red", lw=1)
        w = max(abs(min(span)), abs(max(span))) * 1.25 or 0.1
        ax_d.set_xlim(-w, w)
        ax_d.set_yticks([1, 0]); ax_d.set_yticklabels(["probe − null", "probe − linear"])
        ax_d.set_ylim(-0.7, 1.7)
        ax_d.set_title(f"paired 95% CI ({d['scheme']} bootstrap, n={d['n_eval']})",
                       fontsize=9, loc="left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gt_pred(path, n, tg, probe_pred, linear_pred, d, title):
    y, mask = tg["test"][n]["y"], tg["test"][n]["mask"]
    chosen, full = dc.select_trials(y, mask, k=8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), sharex=True)
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
    p = d["point"]
    fig.suptitle(
        f"{title}\n{LABEL[n]} on the whole test set (n={d['n_eval']}): "
        f"Probe = {p['probe']:.3f}   Linear = {p['linear']:.3f}   Null = {p['null']:.3f}\n"
        f"Probe − null = {d['delta_null']:+.3f} {fmt_ci(d.get('delta_null_ci'))}   "
        f"Probe − linear = {d['delta_linear']:+.3f} {fmt_ci(d.get('delta_linear_ci'))}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return chosen.tolist(), full.tolist()


def plot_confusion(path, n, tg, probe_pred, linear_pred, d, chance_metrics, title):
    rows = dc.evaluable(n, tg["test"][n])
    y = tg["test"][n]["y"][rows]
    k = dc.N_CLASSES[n]
    names = [dc.CLASS_NAMES[n][c] for c in range(k)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    im = None
    for ax, (lab, pred) in zip(axes, (("Frozen probe", probe_pred), ("Raw-spike linear", linear_pred))):
        counts, norm = dc.row_normalised_confusion(y, np.asarray(pred)[rows], k)
        im = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        for a in range(k):
            for b in range(k):
                ax.text(b, a, f"{norm[a, b]:.2f}\n({counts[a, b]})", ha="center", va="center",
                        fontsize=9, color="white" if norm[a, b] > 0.6 else "black")
        m = dc.static_metrics(y, np.asarray(pred)[rows])
        ax.set_title(f"{lab}\nacc {m['acc']:.3f} | balanced acc {m['balanced_acc']:.3f}", fontsize=9)
        ax.set_xticks(range(k)); ax.set_xticklabels(names, fontsize=8)
        ax.set_yticks(range(k)); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true (rows sum to 1)")
    fig.colorbar(im, ax=axes, shrink=0.8, label="fraction of true class")
    p = d["point"]
    per_class = np.bincount(y, minlength=k)
    masked = int((~tg["test"][n]["mask"]).sum())
    if n == "choice":
        head = (f"Choice accuracy (primary): Probe = {p['probe']:.3f}   Linear = {p['linear']:.3f}   "
                f"Chance = {p['null']:.3f} (train majority; balanced chance 0.500)")
    else:
        head = (f"BLOCK BALANCED ACCURACY: Probe = {p['probe']:.3f}   Linear = {p['linear']:.3f}   "
                f"Chance = {p['null']:.3f}")
    fig.suptitle(
        f"{title}\n{head}\n"
        f"Probe − chance = {d['delta_null']:+.3f} {fmt_ci(d.get('delta_null_ci'))}   "
        f"Probe − linear = {d['delta_linear']:+.3f} {fmt_ci(d.get('delta_linear_ci'))}\n"
        f"test n={len(y)}; per true class {dict(zip(names, per_class.tolist()))}"
        + (f"; {masked} masked (choice = 0)" if n == "choice" and masked else ""),
        fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--dir", default="decoding_v2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=DATA_ROOT)
    args = ap.parse_args()
    eid = args.eid.removesuffix("_aligned")
    root = RESULTS / args.dir
    out = Path(args.out) if args.out else root / "plots" / eid
    if out.exists() and any(out.iterdir()):
        print(f"ERROR: {out} already holds plots"); return 2

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

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    src = probe_m["source"]
    title = (f"{eid[:8]}  |  joint epoch {probe_m['joint_epoch']} (null-normalised val score)  |  "
             f"source {src['origin']} best v{src['format_version']}  |  code {sha}")
    out.mkdir(parents=True, exist_ok=True)

    plot_summary(out / "decoding_summary.png", eid, deltas, dropped, title)
    selected = {}
    for n in dc.DYNAMIC:
        if n in deltas:
            selected[n] = plot_gt_pred(out / f"{SHORT[n]}_gt_pred.png", n, tg,
                                       probe_p[f"{n}_pred"], linear[n], deltas[n], title)
    for n in dc.STATIC:
        if n in deltas:
            plot_confusion(out / f"{n}_confusion.png", n, tg, probe_p[f"{n}_pred"], linear[n],
                           deltas[n], base_m["results"][n]["chance"], title)

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
