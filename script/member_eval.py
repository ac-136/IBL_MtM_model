#!/usr/bin/env python
"""Own-session evaluation of ONE member of a multi-session checkpoint (M3, M-g; M4 V2).

    python script/member_eval.py CKPT EID --out-root DIR

The FULL saved model and vocabulary are loaded, and only the evaluation DATA is
restricted to EID's own test split. A singleton model is never reconstructed:
that would replace the session embedding and projections actually trained and
would not test EID routing at all. Asserted: the loaded vocabulary is the full
vocabulary the run recorded, and EID is a member of it.

Reuses own_session_eval.py's capture, bootstrap and summary code, so the band
rule and every exactness check are identical to the single-session diagnostic.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import torch  # noqa: E402

import own_session_eval as ose  # noqa: E402
import stage3_preflight as pf  # noqa: E402
from learnability import BAND_RULE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("eid")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ckpt_p = Path(args.ckpt).resolve()
    eid = args.eid.removesuffix("_aligned")
    log = []

    def need(ok, name, detail=""):
        log.append((bool(ok), name, detail))
        if not ok:
            raise ose.AuditError(f"{name}: {detail}")

    def show():
        for ok, n, d in log:
            print(f"  [{'PASS' if ok else 'FAIL'}] {n}" + (f"   {d}" if d else ""))

    from ndt.compat import load_pickled_model
    from ndt.vocab import stitcher_session_sizes
    try:
        heldout = set()
        for c in ("rs", "bwm"):
            heldout |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldout_eids.txt")))
        need(len(heldout) == 80, "a. held-out manifest read at expected size", str(len(heldout)))
        need(eid not in heldout, "b. eid is NOT an official held-out session", eid[:8])
        summ_p = ckpt_p.parent / "run_summary.json"
        need(ckpt_p.is_file() and summ_p.is_file(), "c. checkpoint and run_summary.json present")
        summ = json.loads(summ_p.read_text())
        run_vocab = summ.get("session_vocab")
        model, vocab, _r, ver = load_pickled_model(str(ckpt_p))
        need(list(vocab) == list(run_vocab) and len(vocab) >= 1,
             "d. FULL vocabulary loaded (== the run's recorded vocabulary)",
             f"{len(vocab)} sessions, format v{ver}")
        need(eid in vocab, "e. eid is a member of that vocabulary", eid[:8])
        clusters = pf.clusters_of(eid)
        sizes = stitcher_session_sizes(model, list(vocab), summ.get("session_num_neurons"))
        need(sizes is not None and sizes.get(eid) == clusters,
             "f. this member's projection is sized for its cluster count",
             f"{sizes.get(eid) if sizes else None} vs parquet {clusters}")
        sd_p = ckpt_p.with_name(ckpt_p.stem + ".state_dict.pt")
        ok, n, detail = ose.state_equal(
            model.state_dict(), torch.load(sd_p, map_location="cpu", weights_only=False)["state_dict"])
        need(ok, f"g. loaded module == .state_dict.pt ({n} tensors)", detail)
        sha = ose.sha256(ckpt_p)
    except ose.AuditError as e:
        show(); print(f"\nAUDIT FAILED: {e}"); return 3
    show()
    if args.dry_run:
        return 0

    out = Path(args.out_root) / eid
    if out.exists():
        print(f"ERROR: {out} exists"); return 2
    out.mkdir(parents=True)
    from ndt.utils.eval_utils import load_model_data_local
    from ndt.paths import config_path
    model, acc, dataset, dataloader = load_model_data_local(
        model_config=config_path("ndt1_stitching_eval"), model_path=str(ckpt_p),
        trainer_config=config_path("trainer_ndt1"), seed=42, mask_name="mask_temporal",
        eid=eid, data_type=ose.DATA_TYPE, base_path=ose.BASE_PATH, data_root=ose.DATA_ROOT)
    live_vocab = list(model.encoder.embedder.eid_lookup)
    log.append((live_vocab == list(run_vocab), "d'. eval model carries the full vocabulary",
                f"{len(live_vocab)} sessions"))
    log.append((len(dataset) == ose.parquet_rows(eid, "test"), "h. eval data = this member's test split",
                f"{len(dataset)} vs test.parquet {ose.parquet_rows(eid, 'test')}"))
    if not all(ok for ok, _, _ in log):
        show(); return 3
    res = ose.run_eval(model, acc, dataloader, dataset, out / "trained", log, "trained")
    n_fail = sum(1 for ok, _, _ in log if not ok)
    summary = {"eid": eid, "checkpoint": str(ckpt_p), "checkpoint_sha256": sha,
               "session_vocab": live_vocab, "clusters": clusters, "band_rule": BAND_RULE,
               "variants": {"trained": res},
               "checks": [{"ok": ok, "name": n, "detail": d} for ok, n, d in log],
               "n_checks_failed": n_fail}
    (out / "member_eval_summary.json").write_text(json.dumps(summary, indent=2))
    show()
    t = res["co_smooth"]
    print(f"\n{eid[:8]} member of {len(live_vocab)}-session model: co_smooth mean "
          f"{t['mean_bps']:+.4f} CI [{t['ci95_lo']:+.4f}, {t['ci95_hi']:+.4f}] -> {t['band']}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
