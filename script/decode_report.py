#!/usr/bin/env python
"""Exploratory behaviour-decoding report: frozen-NDT1 probe vs raw-spike
linear baseline vs train-only chance, on the development sessions, with the
stress-test session reported separately and never pooled.

    python script/decode_report.py

Every number is a TEST metric of a model selected on VAL. The val-only
selection table is a diagnostic: it compares the joint epoch with each
target's own best val epoch; those per-target epochs were never test-evaluated.
Descriptive only; one head-training seed.
"""

import json
import sys
from pathlib import Path

RESULTS = Path("/work/nvme/beml/ac136/IBL_MTM/results/decoding")
DEV = [
    "71e55bfe-5a3a-4cba-bdc7-f085140d798e",   # rs, reused pilot pretrain
    "862ade13-53cd-4221-a3fa-dda8643641f2",   # rs, reused pilot pretrain
    "4aa1d525-5c7d-4c50-a147-ec53a9014812",   # bwm, >0, t_var 0.14
    "b9c205c3-feac-485b-a89d-afc96d9cb280",   # bwm, >0, t_var 0.63
]
STRESS = ["5b49aca6-a6f4-4075-931a-617ad64c219c"]   # pilot source, time-flat predictions
TARGETS = ("choice", "block", "wheel-speed", "whisker-motion-energy")
KEYS = {"choice": ("acc", "balanced_acc"), "block": ("acc", "balanced_acc"),
        "wheel-speed": ("r2_trial", "r2_psth"), "whisker-motion-energy": ("r2_trial", "r2_psth")}


def load(kind, eid):
    p = RESULTS / kind / eid / "metrics.json"
    return json.loads(p.read_text()) if p.is_file() else None


def fmt(v, ci=None):
    if v is None:
        return f"{'—':>22s}"
    s = f"{v:+.3f}"
    if ci:
        s += f" [{ci[0]:+.2f},{ci[1]:+.2f}]"
    return f"{s:>22s}"


def session_block(eid, label=""):
    probe, base = load("probe", eid), load("baseline", eid)
    print(f"\n{eid[:8]} {label}")
    if probe is None or base is None:
        print(f"  MISSING: probe={'ok' if probe else 'none'} baseline={'ok' if base else 'none'}")
        return False
    print(f"  probe checks {len(probe['checks']) - probe['n_checks_failed']}/{len(probe['checks'])}; "
          f"joint epoch {probe['joint_epoch']}; source {probe['source']['origin']} "
          f"v{probe['source']['format_version']}")
    print(f"  {'target':22s} {'metric':13s} {'probe (test, CI)':>22s} {'linear (test, CI)':>22s} "
          f"{'chance':>22s}  n_eval  baseline choice")
    for t in TARGETS:
        info = base["targets"][t]
        if info["dropped"]:
            print(f"  {t:22s} DROPPED on train: {info['dropped']}")
            continue
        pm, br = probe["metrics"][t], base["results"][t]
        lin, ch = br["linear"], br["chance"]
        for k in KEYS[t]:
            print(f"  {t:22s} {k:13s} {fmt(pm.get(k), pm['ci95'].get(k))} "
                  f"{fmt(lin.get(k), lin['ci95'].get(k))} {fmt(ch.get(k))}  "
                  f"{pm['n_eval']:>6d}  {lin['selection']['chosen']}")
        masked = (info.get("n_masked") or info.get("n_invalid_trials"))
        print(f"  {'':22s} masked/invalid trials per split: {masked}")
    return True


def selection_table(eids):
    print("\n=== VAL-ONLY selection diagnostic (per-target epochs never test-evaluated) ===")
    print(f"  {'eid':9s} {'joint':>6s} " + " ".join(f"{t[:12]:>26s}" for t in TARGETS))
    for eid in eids:
        p = RESULTS / "probe" / eid / "selection.json"
        if not p.is_file():
            continue
        s = json.loads(p.read_text())
        cells = []
        for t in TARGETS:
            d = s["per_target_diagnostic"].get(t)
            if d is None:
                cells.append(f"{'dropped':>26s}")
                continue
            gap = d["val_loss_at_joint_epoch"] - d["own_best_val_loss"]
            cells.append(f"{'ep ' + str(d['own_best_epoch']) + ' gap ' + format(gap, '.2e'):>26s}")
        print(f"  {eid[:8]:9s} {s['joint_epoch']:>6d} " + " ".join(cells))
    print("  gap = val loss at the joint epoch minus the target's own best val loss (>= 0)")


def main():
    print("Behaviour decoding — exploratory (frozen NDT1 best checkpoint; heads only)")
    print("split roles: train = fit, scaling, drops, chance | val = selection | test = metrics, once")
    print("\n=== DEVELOPMENT SESSIONS ===")
    ok = [session_block(e) for e in DEV]
    selection_table(DEV)
    print("\n=== STRESS TEST — not pooled, read as a tooling stress result ===")
    ok += [session_block(e, "(stress test: >0 co-smoothing but time-flat predictions)") for e in STRESS]
    selection_table(STRESS)
    missing = len(ok) - sum(ok)
    if missing:
        print(f"\nINCOMPLETE: {missing} session(s) missing outputs")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
