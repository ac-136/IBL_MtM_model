#!/usr/bin/env python
"""Stage-3 report: hard criteria 4-6 over the ten source -> target transfers.

    python script/stage3_report.py [transfer_root]

Stage 2 gated on criteria 1-3, all properties of a training curve. Criteria 4-6
need eval artifacts and are what actually decide recipe health, because they are
the only ones measured on a session the model never saw.

Hard criteria and diagnostics are printed in separate blocks and the verdict is
computed from the hard criteria alone: a diagnostic warning must not read as a
failure, and a hard failure must not hide among warnings.

THRESHOLDS ARE FIXED HERE, IN CODE, rather than chosen while reading results.
Criterion 5's bands come from the plan and predate every result. Criterion 4's
numeric floor is set in this file; see FINITE_FLOOR for exactly what was known
when it was chosen.
"""

import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = "/work/nvme/beml/ac136/IBL_MTM/results/pilot/transfer"

SOURCES = {
    "862ade13-53cd-4221-a3fa-dda8643641f2": ("rs", 258, 378),
    "71e55bfe-5a3a-4cba-bdc7-f085140d798e": ("rs", 1069, 800),
    "5b49aca6-a6f4-4075-931a-617ad64c219c": ("bwm", 736, 539),
    "b182b754-3c3e-4942-8144-6ee790926b58": ("bwm", 1255, 303),
    "f56194bc-8215-4ae8-bc6a-89781ad8e050": ("bwm", 1899, 486),
}
DEV_TARGETS = {
    "572a95d1-39ca-42e1-8424-5c9ffcb2df87": ("rs", 602, 233),
    "8c552ddc-813e-4035-81cc-3971b57efe65": ("bwm", 1775, 319),
    "41872d7f-75cb-4445-bb1a-132b354c44f0": ("rs", 308, 354),
    "837b4e6a-ccfd-49b0-a1dd-3aa53fbf2ecb": ("bwm", 860, 392),
}
EXPECTED_PAIRS = 10

# Criterion 4 gates on CO_SMOOTH only. That is the plan's wording -- "co_smooth/
# bps.npy and r2.npy are finite for a large majority of clusters" -- and it is
# also the array criterion 5 is judged on. forward_pred finiteness is reported
# as a diagnostic instead, because forward prediction sees strictly less of each
# trial and so has more genuinely undefined clusters for data reasons.
#
# FINITE_FLOOR is the numeric reading of "a large majority". Fixed when exactly
# ONE of the ten pairs had been measured (the canary: co_smooth 590/602 = 98.0%,
# forward_pred 524/602 = 87.0%), so it is informed by a single observation and
# is deliberately set well below it rather than just under it. A session falling
# below SESSION_DEAD is treated as a whole session going non-finite, which the
# plan names as the real signal.
FINITE_FLOOR = 0.80
SESSION_DEAD = 0.50

# Criterion 5 bands, fixed in the plan before any result existed.
HEALTHY, INCONCLUSIVE = 0.80, 0.50


def mean_finite(a):
    f = np.asarray(a, dtype=float)
    m = np.isfinite(f)
    return (float(f[m].mean()) if m.any() else float("nan"),
            int(m.sum()), int(f.size))


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    if not root.is_dir():
        sys.exit(f"no transfer root at {root}")

    rows = []
    for summ in sorted(root.glob("from_*/finetune/*/run_summary.json")):
        s = json.loads(summ.read_text())
        tgt = summ.parent.name
        src = s.get("source_eid") or summ.parent.parent.parent.name[len("from_"):]
        ev = root / f"from_{src}" / "eval" / tgt
        r = {"src": src, "tgt": tgt,
             "src_coh": SOURCES.get(src, ("?",))[0],
             "tgt_coh": DEV_TARGETS.get(tgt, ("?",))[0],
             "tgt_clusters": DEV_TARGETS.get(tgt, ("?", None))[1],
             "best_epoch": s.get("best_epoch"), "best_metric": s.get("best_metric"),
             "sha": (s.get("source_checkpoint_sha256") or "")[:12],
             "role": s.get("source_checkpoint_role")}
        bench = summ.parent / "benchmark.json"
        r["gpu_h"] = (json.loads(bench.read_text()).get("total_gpu_hours")
                      if bench.is_file() else None)
        for task in ("co_smooth", "forward_pred"):
            p = ev / task / "bps.npy"
            if p.is_file():
                m, nf, n = mean_finite(np.load(p))
                r[f"{task}_bps"], r[f"{task}_fin"], r[f"{task}_n"] = m, nf, n
            else:
                r[f"{task}_bps"], r[f"{task}_fin"], r[f"{task}_n"] = None, 0, 0
        p = ev / "co_smooth" / "r2.npy"
        if p.is_file():
            _, nf, n = mean_finite(np.load(p))
            r["co_smooth_r2_fin"], r["co_smooth_r2_n"] = nf, n
        else:
            r["co_smooth_r2_fin"] = r["co_smooth_r2_n"] = 0
        rows.append(r)

    if not rows:
        sys.exit(f"no transfer run_summary.json under {root}")

    hdr = (f"{'source':9s} {'target':9s} {'coh':8s} {'tclust':>6s} "
           f"{'co_smooth':>10s} {'fwd_pred':>9s} {'cs_finite':>10s} "
           f"{'fp_finite':>9s} {'best_ep':>7s} {'gpu_h':>6s}")
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))
    for r in rows:
        cs = f"{r['co_smooth_bps']:+.4f}" if r["co_smooth_bps"] is not None else "?"
        fp = f"{r['forward_pred_bps']:+.4f}" if r["forward_pred_bps"] is not None else "?"
        csf = "{}/{}".format(r["co_smooth_fin"], r["co_smooth_n"])
        fpf = "{}/{}".format(r["forward_pred_fin"], r["forward_pred_n"])
        gph = "{:.2f}".format(r["gpu_h"]) if r["gpu_h"] else "?"
        coh = r["src_coh"] + "->" + r["tgt_coh"]
        print(f"{r['src'][:8]:9s} {r['tgt'][:8]:9s} "
              f"{coh:8s} {str(r['tgt_clusters']):>6s} "
              f"{cs:>10s} {fp:>9s} {csf:>10s} {fpf:>9s} "
              f"{str(r['best_epoch']):>7s} {gph:>6s}")

    print(f"\n--- HARD criteria 4-6 (these decide recipe health) ---")
    fails = []

    # -- criterion 4
    bad4 = []
    for r in rows:
        for label, nf, n in (("bps", r["co_smooth_fin"], r["co_smooth_n"]),
                             ("r2", r["co_smooth_r2_fin"], r["co_smooth_r2_n"])):
            frac = nf / n if n else 0.0
            if frac < FINITE_FLOOR:
                bad4.append(f"{r['src'][:8]}->{r['tgt'][:8]} co_smooth/{label} "
                            f"{100*frac:.1f}%"
                            + ("  <-- WHOLE SESSION" if frac < SESSION_DEAD else ""))
    ok4 = not bad4
    print(f"  [{'PASS' if ok4 else 'FAIL'}] 4. co_smooth bps and r2 finite for "
          f">= {100*FINITE_FLOOR:.0f}% of clusters"
          + (f"   {len(rows)}/{len(rows)} pairs" if ok4 else ""))
    for b in bad4:
        print(f"         {b}")
    if not ok4:
        fails.append("4")

    # -- criterion 5
    scored = [r for r in rows if r["co_smooth_bps"] is not None
              and np.isfinite(r["co_smooth_bps"])]
    npos = sum(1 for r in scored if r["co_smooth_bps"] > 0)
    frac = npos / len(scored) if scored else 0.0
    band = ("HEALTHY" if frac >= HEALTHY else
            "INCONCLUSIVE" if frac >= INCONCLUSIVE else "NOT HEALTHY")
    ok5 = frac >= HEALTHY
    print(f"  [{'PASS' if ok5 else 'FAIL'}] 5. co-smoothing bps positive in aggregate: "
          f"{npos}/{len(scored)} pairs ({100*frac:.0f}%) -> {band}")
    print(f"         bands fixed in advance: >=80% healthy, 50-79% inconclusive, "
          f"<50% not healthy")
    if len(scored) < EXPECTED_PAIRS:
        print(f"         NOTE: only {len(scored)} of {EXPECTED_PAIRS} pairs measured -- "
              f"this is a partial verdict, not the Stage-3 result")
    if not ok5:
        fails.append("5")

    # -- criterion 6
    missing = EXPECTED_PAIRS - len(rows)
    by_src, by_tgt = {}, {}
    for r in rows:
        frac_i = r["co_smooth_fin"] / r["co_smooth_n"] if r["co_smooth_n"] else 0.0
        by_src.setdefault(r["src"], []).append(frac_i)
        by_tgt.setdefault(r["tgt"], []).append(frac_i)
    sys_bad = ([f"source {k[:8]}" for k, v in by_src.items()
                if len(v) > 1 and max(v) < FINITE_FLOOR]
               + [f"target {k[:8]}" for k, v in by_tgt.items()
                  if len(v) > 1 and max(v) < FINITE_FLOOR])
    # "not finished yet" and "finished badly" are different states and must not
    # share a label. Missing pairs report INCOMPLETE and do NOT count as a
    # failure; only a genuinely systematic non-finite source or target does.
    ok6 = missing == 0 and not sys_bad
    tag = "PASS" if ok6 else ("FAIL" if sys_bad else "INCOMPLETE")
    print(f"  [{tag}] 6. all {EXPECTED_PAIRS} pairs complete, "
          f"no source or target systematically non-finite")
    if missing > 0:
        print(f"         {missing} pair(s) not yet present -- incomplete, not failed")
    for b in sys_bad:
        print(f"         {b} non-finite across all its pairs")
    if sys_bad:
        fails.append("6")

    print("\n--- DIAGNOSTIC (investigate; NOT a stop condition) ---")
    any_diag = False
    for r in rows:
        cs, fp = r["co_smooth_bps"], r["forward_pred_bps"]
        if cs is not None and fp is not None and np.isfinite(cs) and np.isfinite(fp) \
                and fp > cs:
            print(f"  [warn] 9. {r['src'][:8]}->{r['tgt'][:8]}: forward_pred "
                  f"({fp:+.4f}) above co_smooth ({cs:+.4f}) -- inversion, prompt to look")
            any_diag = True
        if r["forward_pred_n"]:
            ffp = r["forward_pred_fin"] / r["forward_pred_n"]
            if ffp < FINITE_FLOOR:
                print(f"  [info] {r['src'][:8]}->{r['tgt'][:8]}: forward_pred finite "
                      f"{100*ffp:.1f}% (not gated; forward pred sees less of each trial)")
                any_diag = True
    # criterion 10: cohort split. A property of the SET, computed here rather
    # than reported as "none" without being evaluated.
    coh = {}
    for r in scored:
        coh.setdefault(r["tgt_coh"], []).append(r["co_smooth_bps"])
    if len(coh) >= 2:
        print("  [info] 10. co_smooth bps by TARGET cohort: "
              + ", ".join(f"{k} {np.mean(v):+.4f} (n={len(v)})" for k, v in sorted(coh.items()))
              + "\n         two dev targets per cohort is too few to conclude a cohort "
                "effect; it would justify widening the pilot, not rejecting the recipe")
        any_diag = True
    if not any_diag:
        print("  [ok] no diagnostic warnings fired (all diagnostics were evaluated)")

    print("\n" + "-" * len(hdr))
    if len(rows) < EXPECTED_PAIRS:
        print(f"VERDICT: INCOMPLETE -- {len(rows)} of {EXPECTED_PAIRS} pairs measured."
              f"{' Hard criteria failing so far: ' + ', '.join(fails) if fails else ''}")
    elif fails:
        print(f"VERDICT: hard criteria {', '.join(fails)} FAIL -- the recipe is not "
              f"healthy enough to freeze")
    else:
        print(f"VERDICT: hard criteria 4-6 all pass across {len(rows)} pairs")
    print("Diagnostics above are informational and do not affect this verdict.")
    print("-" * len(hdr))
    return 1 if (fails and len(rows) == EXPECTED_PAIRS) else 0


if __name__ == "__main__":
    sys.exit(main())
