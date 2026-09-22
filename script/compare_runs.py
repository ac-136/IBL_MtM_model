#!/usr/bin/env python
"""Compare two Stage-1b parity results trees artifact by artifact.

    python script/compare_runs.py <root_A> <root_B>

Typical use:

    # 1. the GPU noise floor, from the control reruns
    python script/compare_runs.py .../parity_src .../parity_src_rerun

    # 2. the port, judged against that floor
    python script/compare_runs.py .../parity_src .../parity_ndt

Compared, by relative path under each root:

    loss/loss_history.npz         train_loss and eval_loss per epoch
    model_best.pt, model_last.pt  every tensor in the saved state dict
    co_smooth/bps.npy             per-neuron bits per spike
    forward_pred/bps.npy          per-neuron bits per spike, forward prediction
    r2.npy                        all NaN under the current spikes-only config

benchmark.json is skipped: it is wall-clock timing and will always differ.

NaN handling matters here. r2.npy is entirely NaN today, and bps arrays carry
NaN for neurons whose bits-per-spike was non-finite. torch.equal and a plain
subtraction both get this wrong in opposite directions, so positions that are
NaN on both sides are treated as equal and a NaN on only one side is treated as
an infinite difference.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Checkpoints are pickled nn.Modules, so each one embeds the import path of its
# class: `models.ndt1.NDT1` for the src tree, `ndt.models.ndt1.NDT1` for the
# vendored one. Both trees must therefore be importable to read both sides.
_REPO = Path(__file__).resolve().parent.parent
for _p in (str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARTIFACTS = ("loss_history.npz", "model_best.pt", "model_last.pt", "bps.npy", "r2.npy")
SKIP_NAMES = {"benchmark.json"}


def nan_aware_max_diff(a, b):
    """Worst deviation, with NaN-on-both treated as equal and NaN-on-one as inf."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf"), f"shape {a.shape} vs {b.shape}"
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    if one_nan.any():
        return float("inf"), f"{int(one_nan.sum())} positions NaN on one side only"
    d = np.abs(a - b)
    d[both_nan] = 0.0
    if d.size == 0:
        return 0.0, "empty"
    note = "" if not both_nan.any() else f"{int(both_nan.sum())}/{d.size} NaN both sides"
    return float(d.max()), note


def load_arrays(path):
    """Return {name: ndarray} for one artifact file, or None if unreadable."""
    p = Path(path)
    try:
        if p.suffix == ".npy":
            return {p.name: np.load(p, allow_pickle=False)}
        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=False)
            return {f"{p.name}:{k}": z[k] for k in z.files}
        if p.suffix == ".pt":
            import torch
            obj = torch.load(p, map_location="cpu", weights_only=False)
            model = obj.get("model") if isinstance(obj, dict) else obj
            sd = model.state_dict() if hasattr(model, "state_dict") else model
            return {f"{p.name}:{k}": v.detach().cpu().numpy()
                    for k, v in sd.items() if hasattr(v, "detach")}
    except Exception as e:
        print(f"    could not read {p}: {type(e).__name__}: {e}")
        return None
    return None


COMPARE_NAMES = {"model_best.pt", "model_last.pt", "loss_history.npz",
                 "bps.npy", "r2.npy"}


def interesting(p):
    return p.name in COMPARE_NAMES and p.name not in SKIP_NAMES


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max abs difference treated as a pass (default 0.0, exact)")
    ap.add_argument("--quiet", action="store_true", help="only show non-matching artifacts")
    args = ap.parse_args()

    A, B = Path(args.a).resolve(), Path(args.b).resolve()
    for r in (A, B):
        if not r.is_dir():
            sys.exit(f"not a directory: {r}")

    rel_a = {p.relative_to(A) for p in A.rglob("*") if p.is_file() and interesting(p)}
    rel_b = {p.relative_to(B) for p in B.rglob("*") if p.is_file() and interesting(p)}

    only_a, only_b, shared = sorted(rel_a - rel_b), sorted(rel_b - rel_a), sorted(rel_a & rel_b)

    print(f"\nA = {A}\nB = {B}\n")
    if only_a:
        print(f"only in A ({len(only_a)}):")
        for r in only_a[:20]:
            print(f"  {r}")
    if only_b:
        print(f"only in B ({len(only_b)}):")
        for r in only_b[:20]:
            print(f"  {r}")
    if not shared:
        sys.exit("\nno artifacts in common; did the runs finish and write to these roots?")

    print(f"{len(shared)} artifact(s) in common\n")
    worst_overall, worst_where = 0.0, None
    failed, unreadable = [], []

    for rel in shared:
        da, db = load_arrays(A / rel), load_arrays(B / rel)
        if da is None or db is None:
            unreadable.append(str(rel))
            continue
        keys = sorted(set(da) & set(db))
        missing = sorted(set(da) ^ set(db))
        worst, worst_key, note = 0.0, None, ""
        for k in keys:
            d, n = nan_aware_max_diff(da[k], db[k])
            if d > worst:
                worst, worst_key, note = d, k, n
        ok = (not missing) and worst <= args.tol
        if worst > worst_overall:
            worst_overall, worst_where = worst, f"{rel} [{worst_key}]"
        if not ok:
            failed.append(str(rel))
        if ok and args.quiet:
            continue
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {rel}")
        print(f"         {len(keys)} array(s), max|A-B| = {worst:.6e}"
              + (f"  worst: {worst_key}" if worst_key and worst > 0 else "")
              + (f"  ({note})" if note else ""))
        if missing:
            print(f"         keys present on only one side: {missing[:6]}")

    print("\n" + "-" * 64)
    if unreadable:
        print(f"{len(unreadable)} artifact(s) unreadable:")
        for u in unreadable[:10]:
            print(f"  {u}")
    print(f"largest difference anywhere: {worst_overall:.6e}"
          + (f"   at {worst_where}" if worst_where else ""))
    if failed:
        print(f"{len(failed)} artifact(s) above tolerance {args.tol:g}:")
        for f in failed[:20]:
            print(f"  FAIL {f}")
        print("\nIf this is the src-vs-ndt comparison, compare that number against the\n"
              "src-vs-src_rerun noise floor before concluding the port is at fault.")
    else:
        print(f"all {len(shared)} artifacts agree within tolerance {args.tol:g}")
    print("-" * 64)
    sys.exit(1 if (failed or unreadable) else 0)


if __name__ == "__main__":
    main()
