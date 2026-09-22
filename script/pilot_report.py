#!/usr/bin/env python
"""Pilot report: one compact summary row per run, hard criteria kept apart from
diagnostics.

    python script/pilot_report.py [results_root]

A diagnostic warning must not read as a failure, and a hard failure must not
hide among warnings, so the two are reported in separate blocks and the verdict
is computed from the hard criteria alone.

Stage 2 gates on hard criteria 1-3 only, all of which are properties of the
training curve. Criteria 4-6 need eval artifacts and apply from stage 3.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEFAULT_ROOT = "/work/nvme/beml/ac136/IBL_MTM/results/pilot"

# Session facts, so the report can show what each run was up against without
# re-reading 466 parquet files.
SESSIONS = {
    "862ade13-53cd-4221-a3fa-dda8643641f2": ("rs", 258, 378),
    "71e55bfe-5a3a-4cba-bdc7-f085140d798e": ("rs", 1069, 800),
    "5b49aca6-a6f4-4075-931a-617ad64c219c": ("bwm", 736, 539),
    "b182b754-3c3e-4942-8144-6ee790926b58": ("bwm", 1255, 303),
    "f56194bc-8215-4ae8-bc6a-89781ad8e050": ("bwm", 1899, 486),
}


def finite(xs):
    a = np.asarray(xs, dtype=float)
    return bool(np.all(np.isfinite(a))) if a.size else False


def assess(summary, bench):
    """Return (row, hard, diag). `hard` values are True/False; `diag` are notes."""
    eid = (summary.get("eids") or ["?"])[0]
    cohort, clusters, trials = SESSIONS.get(eid, ("?", None, None))
    tl = summary.get("train_loss") or []
    em = summary.get("eval_metric") or []
    n = summary.get("num_epochs_recorded") or len(tl)

    hard = {}
    # 1. train_loss decreases
    hard["1 train_loss decreases"] = bool(tl) and tl[-1] < tl[0]
    # 2. eval_metric finite throughout
    hard["2 eval_metric finite"] = finite(em)
    # 3. no divergence / blow-up
    diverged = (not finite(tl)) or (bool(tl) and max(tl) > 10 * tl[0])
    hard["3 no divergence"] = not diverged

    diag = []
    if summary.get("best_epoch") == 0:
        diag.append("best_epoch==0 (noisy selection metric; see criterion 7)")
    if n and summary.get("best_epoch") == n - 1:
        diag.append("best_epoch==final (may still have been improving; criterion 8)")
    if bool(tl) and tl[-1] > 0.9 * tl[0]:
        diag.append(f"train_loss fell only {100*(1-tl[-1]/tl[0]):.1f}%")

    gpu_h = bench.get("total_gpu_hours") if bench else None
    row = {
        "eid": eid[:8], "cohort": cohort, "clusters": clusters, "trials": trials,
        "epochs": n,
        "loss_first": tl[0] if tl else None, "loss_final": tl[-1] if tl else None,
        "best_epoch": summary.get("best_epoch"), "best_metric": summary.get("best_metric"),
        "gpu_h": gpu_h,
    }
    return row, hard, diag


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    runs = sorted(root.glob("**/run_summary.json"))
    if not runs:
        sys.exit(f"no run_summary.json under {root}")

    rows, all_hard, all_diag = [], [], []
    for rs in runs:
        summary = json.loads(rs.read_text())
        bench_path = rs.parent / "benchmark.json"
        bench = json.loads(bench_path.read_text()) if bench_path.is_file() else {}
        row, hard, diag = assess(summary, bench)
        rows.append(row); all_hard.append(hard); all_diag.append((row["eid"], diag))

    hdr = (f"{'eid':9s} {'coh':4s} {'clust':>6s} {'trials':>6s} {'ep':>4s} "
           f"{'loss_first':>10s} {'loss_final':>10s} {'drop':>6s} "
           f"{'best_ep':>7s} {'best_metric':>12s} {'gpu_h':>6s}")
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for r in rows:
        drop = (f"{100*(1-r['loss_final']/r['loss_first']):.0f}%"
                if r["loss_first"] and r["loss_final"] is not None else "?")
        gpu = f"{r['gpu_h']:.2f}" if r["gpu_h"] else "?"
        metric = f"{r['best_metric']:.6f}" if r["best_metric"] is not None else "?"
        print(f"{r['eid']:9s} {r['cohort']:4s} {str(r['clusters']):>6s} "
              f"{str(r['trials']):>6s} {str(r['epochs']):>4s} "
              f"{r['loss_first']:>10.5f} {r['loss_final']:>10.5f} {drop:>6s} "
              f"{str(r['best_epoch']):>7s} {metric:>12s} {gpu:>6s}")

    print("\n--- HARD criteria (stage 2 gates on 1-3; these decide go/no-go) ---")
    names = sorted({k for h in all_hard for k in h})
    for name in names:
        bad = [rows[i]["eid"] for i, h in enumerate(all_hard) if not h.get(name)]
        print(f"  [{'PASS' if not bad else 'FAIL'}] {name}"
              + (f"   failing: {bad}" if bad else f"   {len(rows)}/{len(rows)}"))

    print("\n--- DIAGNOSTIC (investigate; NOT a stop condition) ---")
    any_diag = False
    for eid, diag in all_diag:
        for d in diag:
            print(f"  [warn] {eid}  {d}"); any_diag = True

    # Cross-run diagnostics (criteria 10 and 11). These are properties of the SET
    # of runs, not of any one run, so they have to be computed here; an earlier
    # version reported "none" while never evaluating them at all.
    ms = [r["best_metric"] for r in rows if r["best_metric"] is not None]
    if len(ms) >= 2:
        rng = max(ms) - min(ms)
        if rng > 0.1:
            print(f"  [warn] criterion 11: best_metric spread {rng:.4f} "
                  f"({min(ms):+.4f} to {max(ms):+.4f}) across {len(ms)} sources")
            any_diag = True
        tr = [r["trials"] for r in rows if r["best_metric"] is not None and r["trials"]]
        cl = [r["clusters"] for r in rows if r["best_metric"] is not None and r["clusters"]]
        if len(tr) == len(ms) and len(set(tr)) > 1:
            ct = float(np.corrcoef(ms, tr)[0, 1])
            cc = float(np.corrcoef(ms, cl)[0, 1]) if len(cl) == len(ms) else float("nan")
            print(f"  [info] best_metric vs trials r={ct:+.3f}, vs clusters r={cc:+.3f}"
                  f"   (n={len(ms)}; indicative only)")
    coh = {}
    for r in rows:
        if r["best_metric"] is not None and r["cohort"] in ("rs", "bwm"):
            coh.setdefault(r["cohort"], []).append(r["best_metric"])
    if len(coh) == 2:
        means = {k: sum(v) / len(v) for k, v in coh.items()}
        print(f"  [info] criterion 10 cohort means: "
              + ", ".join(f"{k} {means[k]:+.4f} (n={len(coh[k])})" for k in sorted(coh))
              + "   — too few per cohort to conclude a cohort effect")

    if not any_diag:
        print("  [ok] none")

    n_fail = sum(1 for h in all_hard for v in h.values() if not v)
    print("\n" + "-" * len(hdr))
    if n_fail:
        print(f"VERDICT: {n_fail} hard-criterion failure(s) across {len(rows)} run(s)")
    else:
        print(f"VERDICT: all hard criteria pass across {len(rows)} run(s)")
    print("Diagnostics above are informational and do not affect this verdict.")
    print("-" * len(hdr))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
