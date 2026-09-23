#!/usr/bin/env python
"""Stage 3-diag report: own-session learnability for the five Stage-2 sources.

    python script/own_session_report.py [own_session_root]

The BAND is the primary result; every other column is a diagnostic explaining
how that BPS arose. The band rule is printed verbatim so the table cannot be
read against a different one. No thresholds are applied to the diagnostics.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "script"))

from learnability import BAND_RULE  # noqa: E402

DEFAULT_ROOT = "/work/nvme/beml/ac136/IBL_MTM/results/pilot/own_session"
EXPECTED = 5


def rng(vals):
    vals = [v for v in vals if v == v]
    return f"[{min(vals):+.4f},{max(vals):+.4f}]" if vals else "?"


def block(rows, task):
    hdr = (f"{'source':9s} {'coh':4s} {'clust':>5s} {'band':>5s} {'mean_bps':>9s} "
           f"{'ci95':>19s} {'median':>8s} {'%pos':>5s} {'fin':>9s} {'r2_psth':>8s} "
           f"{'r2_trial':>8s} {'t_var':>7s} {'tr_var':>7s} {'untrained bps range':>20s}")
    print(f"\n=== {task} ===")
    print(hdr); print("-" * len(hdr))
    for s in rows:
        v = s["variants"]; t = v["trained"][task]
        un = [v[k][task]["mean_bps"] for k in v if k.startswith("untrained")]
        ci = f"[{t['ci95_lo']:+.4f},{t['ci95_hi']:+.4f}]"
        fin = f"{t['n_finite_bps']}/{t['n_neurons']}"
        print(f"{s['eid'][:8]:9s} {s['cohort']:4s} {s['clusters']:>5d} {t['band']:>5s} "
              f"{t['mean_bps']:>+9.4f} {ci:>19s} {t['median_bps']:>+8.4f} "
              f"{100*t['frac_neurons_bps_pos']:>4.0f}% {fin:>9s} "
              f"{t['mean_r2_psth']:>+8.4f} {t['mean_r2_trial']:>+8.4f} "
              f"{t['median_temporal_variation_ratio']:>7.3f} "
              f"{t['median_trial_variation_ratio']:>7.3f} {rng(un):>20s}")
    print("\nuntrained references (init seeds 0,1,2) -- range across seeds:")
    for s in rows:
        v = s["variants"]
        u = [v[k][task] for k in v if k.startswith("untrained")]
        print(f"  {s['eid'][:8]}  bands {sorted({x['band'] for x in u})}  "
              f"r2_psth {rng([x['mean_r2_psth'] for x in u])}  "
              f"t_var {rng([x['median_temporal_variation_ratio'] for x in u])}")


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT)
    rows = [json.loads(p.read_text()) for p in sorted(root.glob("*/own_session_summary.json"))]
    if not rows:
        sys.exit(f"no own_session_summary.json under {root}")
    print("BAND RULE (primary):", BAND_RULE)
    print("Columns after 'band' are diagnostics. t_var / tr_var = median ratio of "
          "prediction variance\nover time / over trials to the same for the observed "
          "spikes; ~0 = flat prediction.")
    bad = [r["eid"][:8] for r in rows if r["n_checks_failed"]]
    print(f"\nchecks: {len(rows) - len(bad)}/{len(rows)} sources with every audit and "
          f"non-vacuity check passing" + (f"  FAILING: {bad}" if bad else ""))
    block(rows, "co_smooth")
    block(rows, "forward_pred")
    if len(rows) < EXPECTED:
        print(f"\nINCOMPLETE: {len(rows)} of {EXPECTED} sources")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
