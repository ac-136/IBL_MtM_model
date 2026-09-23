#!/usr/bin/env python
"""Stage 3-diag: evaluate a pretrained source on its OWN untouched test split.

    python script/own_session_eval.py EID [--dry-run]

A diagnostic, not a gate. It answers whether a Stage-2 source learned its own
session, measured the way Stage 3 measured transfer, with nothing re-initialised.

Reuses the Stage-3 eval path rather than re-implementing it:
``load_model_data_local`` (full pickled checkpoint via ``load_pickled_model`` --
no transfer filter, which lives only in finetune_eval's TRAINING leg) and
``co_smoothing_eval`` with finetune_eval's exact configs. ndt/ is untouched.

Predictions are captured by wrapping ``eval_utils.bits_per_spike``, which
``co_smoothing_eval`` calls once per neuron, in neuron order, with exactly the
arrays it scores. The saved bps.npy is still produced by the unmodified function,
and the capture is checked against it bit for bit.

--dry-run runs the audit asserts only: CPU, no eval.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

import stage3_preflight as pf  # noqa: E402  (SOURCES, DEV_TARGETS, manifests)
from learnability import (per_trial_terms, bps_from_terms,  # noqa: E402
                          bootstrap_session_mean, band, variation_ratios, BAND_RULE)

RESULTS = Path("/work/nvme/beml/ac136/IBL_MTM/results")
PRETRAIN = RESULTS / "pilot" / "pretrain"
TRANSFER = RESULTS / "pilot" / "transfer"
OUT_ROOT = RESULTS / "pilot" / "own_session"
DATA_ROOT = pf.DATA_ROOT
BASE_PATH = "/work/nvme/beml/ac136/IBL_MTM/data"
DATA_TYPE = "processed_mtm"
# Absolute, like RESULTS: Slurm logs are gitignored and exist only in the main
# checkout, so a REPO-relative path finds nothing when this runs from a pinned
# worktree -- which is how it is submitted. Audit 5a refused on exactly that.
STAGE2_LOGS = Path("/projects/beml/ac136/IBL_MtM_model/results_parity")
UNTRAINED_SEEDS = (0, 1, 2)
N_STATE_TENSORS = 94


class AuditError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parquet_rows(eid, split):
    return pq.ParquetFile(Path(DATA_ROOT) / eid / "data" / f"{split}.parquet").metadata.num_rows


def paths(eid):
    d = PRETRAIN / f"src_{eid}" / "train" / eid
    return d / "model_best.pt", d / "model_best.state_dict.pt", d / "run_summary.json"


def state_equal(a, b):
    """Exact equality over two state dicts; returns (ok, n_keys, detail)."""
    if set(a) != set(b):
        return False, len(a), f"key sets differ: {sorted(set(a) ^ set(b))[:4]}"
    bad = [k for k in a if not torch.equal(a[k].detach().cpu(), b[k].detach().cpu())]
    return not bad, len(a), (f"{len(bad)} differ: {bad[:4]}" if bad else "all equal")


def audit(eid, log):
    """Asserts 1-6 from the plan. Uniform across sources; raises on failure."""
    from ndt.compat import load_pickled_model

    def need(ok, name, detail=""):
        log.append((bool(ok), name, detail))
        if not ok:
            raise AuditError(f"{name}: {detail}")

    ckpt_p, sd_p, summ_p = paths(eid)

    # 6 first: never evaluate anything that is not a Stage-2 source, and never
    # touch an official held-out session. Manifest sizes asserted so an empty
    # read cannot make the guard vacuous.
    heldout, heldin = set(), set()
    for c in ("rs", "bwm"):
        heldout |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldout_eids.txt")))
        heldin |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldin_eids.txt")))
    need(len(heldout) == 80 and len(heldin) == 321, "6a. manifests read at expected size",
         f"{len(heldout)} held-out / {len(heldin)} held-in")
    need(eid in pf.SOURCES, "6b. eid is a Stage-2 source", eid[:8])
    need(eid not in heldout, "6c. eid is NOT an official held-out session", eid[:8])

    for p in (ckpt_p, sd_p, summ_p):
        need(p.is_file(), f"artifact present: {p.name}", str(p))

    # 1. the checkpoint evaluated is byte-identical to the one Stage 3 transferred
    h = sha256(ckpt_p)
    recs = sorted(TRANSFER.glob(f"from_{eid}/finetune/*/run_summary.json"))
    need(len(recs) == 2, "1a. both Stage-3 transfer records exist", f"found {len(recs)}")
    for r in recs:
        rec = json.loads(r.read_text())
        need(rec.get("source_checkpoint_sha256") == h,
             f"1b. sha256 matches Stage-3 record ({r.parent.name[:8]})",
             f"{h[:12]} vs {str(rec.get('source_checkpoint_sha256'))[:12]}")
        need(rec.get("source_checkpoint_role") == "best", "1c. Stage 3 used role 'best'",
             str(rec.get("source_checkpoint_role")))

    # 2. vocabulary, provenance, and the selected epoch
    ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    summ = json.loads(summ_p.read_text())
    need(ck.get("session_vocab") == [eid], "2a. session_vocab == [eid]",
         str(ck.get("session_vocab")))
    need(ck.get("vocabulary_ordering_rule") == "single_eid", "2b. ordering rule single_eid",
         str(ck.get("vocabulary_ordering_rule")))
    need(ck.get("checkpoint_format_version") == 1, "2c. checkpoint_format_version 1",
         str(ck.get("checkpoint_format_version")))
    need(ck.get("epoch") == summ.get("best_epoch"), "2d. checkpoint epoch == Stage-2 best_epoch",
         f"{ck.get('epoch')} vs {summ.get('best_epoch')}")

    # 3. geometry: the stitcher is keyed on THIS session's cluster count
    clusters = pf.clusters_of(eid)
    keys = list(ck["model"].state_dict())
    stitch = sorted({re.search(r"stitcher_dict\.(\d+)\.", k).group(1)
                     for k in keys if "stitcher_dict." in k})
    need(stitch == [str(clusters)], "3a. stitcher keyed on the session's cluster count",
         f"keys {stitch} vs parquet {clusters}")
    need(pf.SOURCES[eid][1] == clusters, "3b. cluster count matches the plan",
         f"{pf.SOURCES[eid][1]} vs {clusters}")

    # 4. the loaded module is exactly the durable state_dict -- nothing re-initialised
    model, vocab, _rule, _ver = load_pickled_model(str(ckpt_p))
    sd = torch.load(sd_p, map_location="cpu", weights_only=False)["state_dict"]
    ok, n, detail = state_equal(model.state_dict(), sd)
    need(ok and n == N_STATE_TENSORS, f"4. loaded module == .state_dict.pt ({n} tensors)", detail)

    # 5. splits: eval on test; pretraining saw train only
    n_train, n_test = parquet_rows(eid, "train"), parquet_rows(eid, "test")
    logs = sorted(STAGE2_LOGS.glob(f"pilot-pre-{eid[:8]}-*.out"))
    need(len(logs) == 1, "5a. exactly one Stage-2 log", f"{[l.name for l in logs]}")
    m = re.search(r"Train dataset size:\s+(\d+)", logs[0].read_text(errors="replace"))
    need(m and int(m.group(1)) == n_train,
         "5b. Stage-2 trained on the train split only",
         f"log {m.group(1) if m else '?'} vs train.parquet {n_train}")
    return {"sha256": h, "clusters": clusters, "n_train": n_train, "n_test": n_test,
            "best_epoch": summ.get("best_epoch"), "trained_state": sd}


class Capture:
    """Record (pred, gt) for every bits_per_spike call inside co_smoothing_eval."""

    def __init__(self, eu):
        self.eu, self.calls = eu, []

    def __enter__(self):
        self.orig = self.eu.bits_per_spike

        def rec(rates, spikes):
            # copy BEFORE the call: neg_log_likelihood mutates zero rates in place
            self.calls.append((np.array(rates, copy=True), np.array(spikes, copy=True)))
            return self.orig(rates, spikes)

        self.eu.bits_per_spike = rec
        return self

    def __exit__(self, *exc):
        self.eu.bits_per_spike = self.orig
        return False


def summarise(calls, save_path, log, tag):
    """Band + diagnostics for one (variant, task) from captured arrays."""
    import ndt.utils.eval_utils as eu
    saved = np.load(save_path / "bps.npy")
    r2 = np.load(save_path / "r2.npy")
    need = len(calls) == len(saved)
    log.append((need, f"{tag}: one capture per neuron", f"{len(calls)} vs {len(saved)}"))
    if not need:
        raise AuditError(f"{tag}: capture count {len(calls)} != neurons {len(saved)}")

    # capture fidelity: the unmodified function on the captured arrays reproduces
    # bps.npy exactly -- proves both the arrays and their ORDER
    redo = np.array([eu.bits_per_spike(p.copy(), g.copy()) for p, g in calls])
    redo = np.where(np.isinf(redo), np.nan, redo)
    exact = np.array_equal(redo, saved, equal_nan=True)
    log.append((exact, f"{tag}: captured arrays reproduce bps.npy exactly", ""))
    if not exact:
        raise AuditError(f"{tag}: capture does not reproduce bps.npy")

    s_cols, d_cols, vt, vr = [], [], [], []
    for p, g in calls:
        s, d, _ = per_trial_terms(p, g)
        s_cols.append(s); d_cols.append(d)
        a, b = variation_ratios(p, g)
        vt.append(a); vr.append(b)
    S, D = np.stack(s_cols, 1), np.stack(d_cols, 1)

    # the float64 decomposition vs ndt's float32 computation: report, don't gate
    dec = np.array([bps_from_terms(S[:, i], D[:, i]) for i in range(S.shape[1])])
    f32_gap = float(np.nanmax(np.abs(dec - saved)))
    lo, hi, _ = bootstrap_session_mean(S, D)
    fin = np.isfinite(saved)
    np.savez_compressed(save_path / "trial_terms.npz", s=S, d=D,
                        cells_per_trial=calls[0][0].reshape(calls[0][0].shape[0], -1).shape[1])
    return {
        "mean_bps": float(np.nanmean(saved)), "ci95_lo": lo, "ci95_hi": hi,
        "band": band(lo, hi),
        "median_bps": float(np.nanmedian(saved)),
        "frac_neurons_bps_pos": float(np.mean(saved[fin] > 0)) if fin.any() else float("nan"),
        "n_neurons": int(saved.size), "n_finite_bps": int(fin.sum()),
        "n_finite_r2_psth": int(np.isfinite(r2[:, 0]).sum()),
        "n_finite_r2_trial": int(np.isfinite(r2[:, 1]).sum()),
        "mean_r2_psth": float(np.nanmean(r2[:, 0])),
        "mean_r2_trial": float(np.nanmean(r2[:, 1])),
        "median_temporal_variation_ratio": float(np.nanmedian(vt)),
        "median_trial_variation_ratio": float(np.nanmedian(vr)),
        "n_test_trials": int(S.shape[0]),
        "float32_vs_float64_max_abs_bps_gap": f32_gap,
    }


def run_eval(model, accelerator, dataloader, dataset, out, log, tag):
    import matplotlib.pyplot as plt
    import ndt.utils.eval_utils as eu
    # finetune_eval.py's exact eval configs (co_smooth and forward_pred blocks)
    common = {"method_name": "mask_temporal", "n_time_steps": 100, "is_aligned": True,
              "subtract": "task", "n_jobs": 8, "data_type": DATA_TYPE}
    tasks = {
        "co_smooth": {**common, "mode": "per_neuron", "onset_alignment": [40]},
        "forward_pred": {**common, "mode": "forward_pred",
                         "held_out_list": list(range(90, 100)), "onset_alignment": []},
    }
    res = {}
    for task, cfg in tasks.items():
        save = out / task
        with Capture(eu) as cap:
            eu.co_smoothing_eval(model, accelerator, dataloader, dataset,
                                 save_path=str(save), **cfg)
        # co_smoothing_eval draws its histogram on the current figure; without a
        # close, later bps.png files accumulate earlier histograms. Plot only.
        plt.close("all")
        res[task] = summarise(cap.calls, save, log, f"{tag}/{task}")
    return res


def build_untrained(eid, seed, accelerator):
    """Fresh NDT1 via load_model_data_local's exact config sequence."""
    from ndt.utils.config_utils import config_from_kwargs, update_config
    from ndt.utils.dataset_utils import load_ibl_dataset_locally
    from ndt.utils.utils import set_seed
    from ndt.models.ndt1 import NDT1
    from ndt.paths import config_path
    model_config = config_path("ndt1_stitching_eval")
    config = config_from_kwargs({"model": f"include:{model_config}"})
    config = update_config(model_config, config)
    config = update_config(config_path("trainer_ndt1"), config)
    config.model.encoder.masker.mode = "temporal"
    _, _, _, meta = load_ibl_dataset_locally(num_sessions=1, test_session_eid=[], eid=eid,
                                             data_type=DATA_TYPE, base_path=BASE_PATH,
                                             data_root=DATA_ROOT)
    set_seed(seed)
    model = NDT1(config.model, **config.method.model_kwargs, **meta, session_vocab=[eid])
    model.encoder.masker.mode = "temporal"
    model.encoder.masker.force_active = False
    return accelerator.prepare(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--dry-run", action="store_true", help="audit asserts only; no eval")
    args = ap.parse_args()
    eid = args.eid.removesuffix("_aligned")
    log = []

    def show():
        for ok, name, detail in log:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    print(f"=== own-session learnability: {eid} ===\n--- audit ---")
    try:
        a = audit(eid, log)
    except AuditError as e:
        show(); print(f"\nAUDIT FAILED: {e}"); return 3
    show()
    n_audit = len(log)
    print(f"audit: {n_audit}/{n_audit} passed")
    if args.dry_run:
        return 0

    out = OUT_ROOT / eid
    if out.exists() and any(p.name != "capture_meta.json" for p in out.iterdir()):
        print(f"ERROR: {out} already holds results; remove it or choose another root")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    from ndt.utils.eval_utils import load_model_data_local
    from ndt.paths import config_path
    trained, accelerator, dataset, dataloader = load_model_data_local(
        model_config=config_path("ndt1_stitching_eval"),
        model_path=str(paths(eid)[0]), trainer_config=config_path("trainer_ndt1"),
        seed=42, mask_name="mask_temporal", eid=eid, data_type=DATA_TYPE,
        base_path=BASE_PATH, data_root=DATA_ROOT)

    # 4 again, on the model the eval will actually use (post accelerator.prepare)
    ok, n, detail = state_equal(trained.state_dict(), a["trained_state"])
    log.append((ok, f"4'. eval model == .state_dict.pt after prepare ({n})", detail))
    ok5 = len(dataset) == a["n_test"]
    log.append((ok5, "5c. eval dataset is the test split",
                f"{len(dataset)} vs test.parquet {a['n_test']}"))
    if not (ok and ok5):
        show(); return 3

    variants = {"trained": run_eval(trained, accelerator, dataloader, dataset,
                                    out / "trained", log, "trained")}
    trained_sd = {k: v.detach().cpu() for k, v in trained.state_dict().items()}
    untrained_sds = {}
    for s in UNTRAINED_SEEDS:
        m = build_untrained(eid, s, accelerator)
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        # non-vacuity: same architecture, genuinely different weights
        same_keys = set(sd) == set(trained_sd)
        nonconst = [k for k in sd if sd[k].numel() > 1 and sd[k].float().std() > 0]
        clash = [k for k in nonconst if torch.equal(sd[k], trained_sd[k])]
        log.append((same_keys and not clash,
                    f"untrained_s{s}: same keys as trained, differs on all "
                    f"{len(nonconst)} non-constant tensors", f"{len(clash)} identical"))
        for s0, sd0 in untrained_sds.items():
            clash2 = [k for k in nonconst if torch.equal(sd[k], sd0[k])]
            log.append((not clash2, f"untrained_s{s} differs from untrained_s{s0}",
                        f"{len(clash2)} identical non-constant tensors"))
        untrained_sds[s] = sd
        variants[f"untrained_s{s}"] = run_eval(m, accelerator, dataloader, dataset,
                                               out / f"untrained_s{s}", log, f"untrained_s{s}")
        del m
        torch.cuda.empty_cache()

    n_fail = sum(1 for ok, _, _ in log if not ok)
    summary = {"eid": eid, "cohort": pf.SOURCES[eid][0], "clusters": a["clusters"],
               "n_train": a["n_train"], "n_test": a["n_test"],
               "source_checkpoint_sha256": a["sha256"], "source_checkpoint_role": "best",
               "source_best_epoch": a["best_epoch"], "band_rule": BAND_RULE,
               "untrained_seeds": list(UNTRAINED_SEEDS), "variants": variants,
               "checks": [{"ok": ok, "name": n, "detail": d} for ok, n, d in log],
               "n_checks_failed": n_fail}
    (out / "own_session_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- all checks ---"); show()
    t = variants["trained"]["co_smooth"]
    print(f"\n{eid[:8]} own-session co_smooth: mean {t['mean_bps']:+.4f} "
          f"CI [{t['ci95_lo']:+.4f}, {t['ci95_hi']:+.4f}] -> band {t['band']}")
    print(f"checks: {len(log) - n_fail}/{len(log)} passed; wrote {out/'own_session_summary.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
