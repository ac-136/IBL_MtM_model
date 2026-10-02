#!/usr/bin/env python
"""Evaluation-only check after the block-confound trigger fired (T1).

    python script/decode_block_check.py [--dir decoding_v2]

Question: how much block information is there once the p(left)=0.5 opening
segment, which trial position alone identifies, is set aside?

On each session's EXISTING test predictions, block is scored on the true
p(left)=0.2 and 0.8 trials only. The metric is balanced accuracy over those two
true classes (mean of their recalls); a prediction of 0.5 counts as an error.
Nothing is retrained: the heads were trained with three classes, which is
recorded as a caveat. Compared on identical trials:
  probe     frozen-NDT1 heads (probe preds.npz)
  linear    raw-spike baseline (baseline preds.npz; absent for the 4
            development sessions, which predate saving it)
  position  block from trial position alone (decode_aggregate.position_baseline,
            train fit, C on val; then scored on the same 0.2/0.8 trials)
  null      train majority among 0.2/0.8 trials (balanced accuracy 0.5)
Deltas use the class-stratified paired bootstrap of dc.paired_delta_ci
(B = 1000, seed 0); states as in decode_aggregate.ci_state. Populations as in
decode_aggregate: primary 48, completeness 52 (reference). Descriptive only.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import numpy as np  # noqa: E402

import decode_aggregate as da  # noqa: E402
import decode_components as dc  # noqa: E402

BIASED = (dc.LOOKUP["block"][0.2], dc.LOOKUP["block"][0.8])


def biased_target(t):
    """Block target restricted to true 0.2 / 0.8 trials (labels unchanged)."""
    return {"y": t["y"], "mask": t["mask"] & np.isin(t["y"], BIASED)}


def session(eid, cohort, root, data_root):
    from behavior_decode import read_behaviour
    raw = read_behaviour(eid, data_root)
    tg = dc.prepare_targets(raw)["targets"]
    if "block" not in dc.active_targets(dc.prepare_targets(raw)):
        return {"eid": eid, "cohort": cohort, "dev": eid in da.DEV, "dropped": True}
    tb = {s: tg[s]["block"] for s in dc.SPLITS}
    splits = np.load(Path(data_root) / eid / "splits.npz")
    pos = {s: splits[f"{s}_trial_indices"].astype(float) for s in dc.SPLITS}
    pred_pos, _ = da.position_baseline(pos, tb)
    probe = np.load(root / "probe" / eid / "preds.npz")
    assert np.array_equal(probe["block_y"], tb["test"]["y"])
    bp = root / "baseline" / eid / "preds.npz"
    linear = np.load(bp)["block_pred"] if bp.is_file() else None

    test = biased_target(tb["test"])
    train = biased_target(tb["train"])
    counts = np.bincount(train["y"][train["mask"]], minlength=3)
    null = np.full(len(test["y"]), BIASED[int(np.argmax(counts[list(BIASED)]))])
    n_cls = {f"{v}": int((test["mask"] & (test["y"] == c)).sum())
             for v, c in zip((0.2, 0.8), BIASED)}

    out = {"eid": eid, "cohort": cohort, "dev": eid in da.DEV, "dropped": False,
           "n_test": n_cls,
           "pred_half_share_probe": float((probe["block_pred"][test["mask"]] == dc.LOOKUP["block"][0.5]).mean())}
    vs_pos = dc.paired_delta_ci("block", test, probe["block_pred"], pred_pos, null)
    out["probe"] = vs_pos["point"]["probe"]
    out["position"] = vs_pos["point"]["linear"]
    out["null"] = vs_pos["point"]["null"]
    out["d_null"], out["d_null_ci"] = vs_pos["delta_null"], vs_pos.get("delta_null_ci")
    out["d_position"], out["d_position_ci"] = vs_pos["delta_linear"], vs_pos.get("delta_linear_ci")
    if linear is not None:
        vs_lin = dc.paired_delta_ci("block", test, probe["block_pred"], linear, null)
        assert vs_lin["point"]["probe"] == out["probe"]
        out["linear"] = vs_lin["point"]["linear"]
        out["d_linear"], out["d_linear_ci"] = vs_lin["delta_linear"], vs_lin.get("delta_linear_ci")
    return out


def table(rows, label, lines):
    lines.append(f"\n--- {label}: N = {len(rows)} ({sum(r['dev'] for r in rows)} development) ---")
    el = [r for r in rows if not r["dropped"]]
    lines.append(f"  population N {len(rows)} / eligible n {len(el)} / dropped n {len(rows) - len(el)}")
    for k in ("null", "probe", "linear", "position"):
        v = [r[k] for r in el if k in r]
        if v:
            lines.append(f"  {k:9s} balanced acc (0.2 vs 0.8)  median {np.median(v):.3f}  "
                         f"IQR [{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}]  (n {len(v)})")
    for k, lab in (("d_null", "probe vs null"), ("d_linear", "probe vs linear"),
                   ("d_position", "probe vs position")):
        have = [r for r in el if r.get(f"{k}_ci")]
        st = [da.ci_state(r[f"{k}_ci"]) for r in have]
        lines.append(f"  {lab:18s} median {np.median([r[k] for r in have]):+.3f}  "
                     + "  ".join(f"{s} {st.count(s)}/{len(have)}" for s in da.STATES))
    share = [r["pred_half_share_probe"] for r in el]
    lines.append(f"  probe predicts 0.5 on true 0.2/0.8 trials: median {np.median(share):.1%} of them")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="decoding_v2")
    ap.add_argument("--data-root", default=da.DATA_ROOT)
    args = ap.parse_args()
    root = da.RESULTS / args.dir
    m = json.loads(da.MANIFEST.read_text())
    sessions = {s["eid"]: s["cohort"] for s in m["sessions"]}
    rows = [session(e, c, root, args.data_root) for e, c in sorted(sessions.items())]
    primary = [r for r in rows if not r["dev"]]
    assert len(primary) == 48 and len(rows) == 52

    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    lines = [f"Block 0.2-vs-0.8 evaluation-only check ({args.dir}) at {sha}",
             "Existing predictions; scored on true p(left)=0.2/0.8 test trials; a 0.5 prediction is an error.",
             "Heads were trained with three classes (caveat). Paired class-stratified bootstrap CIs."]
    lines.append("\nPRIMARY 48")
    for c in ("bwm", "rs"):
        table([r for r in primary if r["cohort"] == c], f"primary {c}", lines)
    lines.append("\nCOMPLETENESS 52 (reference; linear absent for the 4 development sessions)")
    for c in ("bwm", "rs"):
        table([r for r in rows if r["cohort"] == c], f"all {c}", lines)

    out = root / "aggregate"
    rep = out / f"block_check_{sha}.txt"
    if rep.exists():
        print(f"ERROR: {rep} exists"); return 2
    rep.write_text("\n".join(lines) + "\n")
    (out / f"block_check_{sha}.json").write_text(json.dumps(rows, indent=1, default=str))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
