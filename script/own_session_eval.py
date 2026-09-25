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

Two modes. Default (pilot) mode is the Stage 3-diag run exactly as it was: the
five Stage-2 sources, the best checkpoint, three untrained references.
``--manifest PATH`` is the learnability campaign: any session in that manifest,
the best AND last checkpoints, no untrained references, and the audits that
depended on pilot-only artifacts generalised (see ``audit``).
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
                          bootstrap_session_mean, band, variation_ratios, BAND_RULE,
                          session_covariates, curve_features)

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


class Ctx:
    """What differs between pilot mode and manifest mode."""

    def __init__(self, manifest=None, pretrain_root=None, out_root=None):
        self.manifest = None
        if manifest:
            m = json.loads(Path(manifest).read_text())
            self.manifest = {s["eid"]: s for s in m["sessions"]}
            self.manifest_path = str(manifest)
        self.pretrain_root = Path(pretrain_root) if pretrain_root else PRETRAIN
        self.out_root = Path(out_root) if out_root else OUT_ROOT
        self.roles = ("best", "last") if self.manifest else ("best",)
        self.untrained_seeds = () if self.manifest else UNTRAINED_SEEDS

    def run_dir(self, eid):
        # A manifest session may reuse its pilot pretrain; the manifest says where.
        # .get: an eid outside the manifest must reach audit 6b and be REFUSED
        # there with a named failure, not crash here first.
        reuse = (self.manifest or {}).get(eid, {}).get("reuse_pilot_pretrain")
        if reuse:
            return Path(reuse)
        return self.pretrain_root / f"src_{eid}" / "train" / eid

    def variant(self, role):
        # pilot mode keeps its original output name so its layout is unchanged
        return "trained" if not self.manifest else role


PILOT = Ctx()


def paths(eid, role="best", ctx=PILOT):
    d = ctx.run_dir(eid)
    return d / f"model_{role}.pt", d / f"model_{role}.state_dict.pt", d / "run_summary.json"


def state_equal(a, b):
    """Exact equality over two state dicts; returns (ok, n_keys, detail)."""
    if set(a) != set(b):
        return False, len(a), f"key sets differ: {sorted(set(a) ^ set(b))[:4]}"
    bad = [k for k in a if not torch.equal(a[k].detach().cpu(), b[k].detach().cpu())]
    return not bad, len(a), (f"{len(bad)} differ: {bad[:4]}" if bad else "all equal")


def audit(eid, log, ctx=PILOT):
    """Asserts 1-6 from the plan. Uniform across sessions; raises on failure.

    Manifest mode differs only where pilot-only artifacts were assumed:
    6b checks manifest membership; 1 compares against Stage-3 transfer records
    only where they exist (otherwise the hash is recorded); 3b checks the
    manifest's cluster count; 5 reads n_train from run_summary.json when it is
    recorded there (since Stage 2c) and falls back to the Stage-2 log; 2d and 4
    run for every evaluated checkpoint role.
    """
    from ndt.compat import load_pickled_model

    def need(ok, name, detail=""):
        log.append((bool(ok), name, detail))
        if not ok:
            raise AuditError(f"{name}: {detail}")

    ckpt_p, sd_p, summ_p = paths(eid, "best", ctx)

    # 6 first: never evaluate anything that is not a Stage-2 source, and never
    # touch an official held-out session. Manifest sizes asserted so an empty
    # read cannot make the guard vacuous.
    heldout, heldin = set(), set()
    for c in ("rs", "bwm"):
        heldout |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldout_eids.txt")))
        heldin |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldin_eids.txt")))
    need(len(heldout) == 80 and len(heldin) == 321, "6a. manifests read at expected size",
         f"{len(heldout)} held-out / {len(heldin)} held-in")
    if ctx.manifest is None:
        need(eid in pf.SOURCES, "6b. eid is a Stage-2 source", eid[:8])
    else:
        need(eid in ctx.manifest, "6b. eid is in the learnability manifest", eid[:8])
    need(eid not in heldout, "6c. eid is NOT an official held-out session", eid[:8])

    for role in ctx.roles:
        for p in paths(eid, role, ctx):
            need(p.is_file(), f"artifact present: {p.name}", str(p))

    # 1. the checkpoint evaluated is byte-identical to the one Stage 3 transferred
    h = sha256(ckpt_p)
    hashes = {role: (h if role == "best" else sha256(paths(eid, role, ctx)[0]))
              for role in ctx.roles}
    recs = sorted(TRANSFER.glob(f"from_{eid}/finetune/*/run_summary.json"))
    if ctx.manifest is None:
        need(len(recs) == 2, "1a. both Stage-3 transfer records exist", f"found {len(recs)}")
    elif not recs:
        log.append((True, "1. no Stage-3 transfer records: sha256 recorded, not compared",
                    h[:12]))
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
    epochs = {"best": ck.get("epoch")}
    if "last" in ctx.roles:
        ck_last = torch.load(paths(eid, "last", ctx)[0], map_location="cpu", weights_only=False)
        final = summ.get("num_epochs_recorded", 0) - 1
        need(ck_last.get("epoch") == final, "2d'. last checkpoint epoch == final recorded epoch",
             f"{ck_last.get('epoch')} vs {final}")
        need(ck_last.get("session_vocab") == [eid], "2a'. last checkpoint session_vocab == [eid]",
             str(ck_last.get("session_vocab")))
        epochs["last"] = ck_last.get("epoch")

    # 3. geometry: this session's input projection is sized for its cluster count.
    # M3 (M-g): read from module shapes, not by parsing a numeric key, so v1
    # (count-keyed) and v2 (EID-keyed) checkpoints are audited identically.
    from ndt.vocab import stitcher_session_sizes
    clusters = pf.clusters_of(eid)
    sizes = stitcher_session_sizes(ck["model"], [eid])
    need(sizes == {eid: clusters}, "3a. stitcher sized for the session's cluster count",
         f"projections {sizes} vs parquet {clusters}")
    planned = pf.SOURCES[eid][1] if ctx.manifest is None else ctx.manifest[eid]["clusters"]
    need(planned == clusters, "3b. cluster count matches the plan",
         f"{planned} vs {clusters}")

    # 4. the loaded module is exactly the durable state_dict -- nothing re-initialised
    states = {}
    for role in ctx.roles:
        c_p, s_p, _ = paths(eid, role, ctx)
        model, vocab, _rule, _ver = load_pickled_model(str(c_p))
        sd = torch.load(s_p, map_location="cpu", weights_only=False)["state_dict"]
        ok, n, detail = state_equal(model.state_dict(), sd)
        # pilot mode keeps its original check label
        tag = "4. loaded module" if ctx.manifest is None else f"4. {role}: loaded module"
        need(ok and n == N_STATE_TENSORS, f"{tag} == .state_dict.pt ({n} tensors)", detail)
        states[role] = sd

    # 5. splits: eval on test; pretraining saw train only
    n_train, n_test = parquet_rows(eid, "train"), parquet_rows(eid, "test")
    if ctx.manifest is not None and "n_train" in summ:
        # recorded by the pretraining run itself since Stage 2c (item 4)
        need(summ["n_train"] == n_train and summ.get("pretrain_split_policy") == "train_only",
             "5b. pretraining used the train split only (run_summary)",
             f"run_summary {summ['n_train']} / {summ.get('pretrain_split_policy')} "
             f"vs train.parquet {n_train}")
    else:
        logs = sorted(STAGE2_LOGS.glob(f"pilot-pre-{eid[:8]}-*.out"))
        need(len(logs) == 1, "5a. exactly one Stage-2 log", f"{[l.name for l in logs]}")
        m = re.search(r"Train dataset size:\s+(\d+)", logs[0].read_text(errors="replace"))
        need(m and int(m.group(1)) == n_train,
             "5b. Stage-2 trained on the train split only",
             f"log {m.group(1) if m else '?'} vs train.parquet {n_train}")
    return {"sha256": h, "hashes": hashes, "epochs": epochs, "clusters": clusters,
            "n_train": n_train, "n_test": n_test, "best_epoch": summ.get("best_epoch"),
            "trained_state": states["best"], "states": states,
            "run_dir": str(ckpt_p.parent)}


class Capture:
    """Record (pred, gt) for every bits_per_spike call inside co_smoothing_eval."""

    def __init__(self, eu):
        self.eu, self.calls = eu, []

    def __enter__(self):
        self.orig = self.eu.bits_per_spike

        def rec(rates, spikes):
            # copy BEFORE the call: neg_log_likelihood mutates zero rates in place.
            # order="K" preserves the caller's memory layout. forward_pred passes
            # non-contiguous views (strides (4, 436, 4360)), and float32 np.sum's
            # pairwise summation order depends on layout: a C-order copy
            # reproduces bps.npy only to ~4e-6, a layout-preserving one exactly.
            self.calls.append((rates.copy(order="K"), spikes.copy(order="K")))
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
    redo = np.array([eu.bits_per_spike(p.copy(order="K"), g.copy(order="K"))
                     for p, g in calls])
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
    ap.add_argument("--manifest", default=None,
                    help="learnability manifest; switches to manifest mode")
    ap.add_argument("--pretrain-root", default=None,
                    help="manifest mode: root holding src_<eid>/train/<eid>")
    ap.add_argument("--out-root", default=None, help="where results are written")
    args = ap.parse_args()
    eid = args.eid.removesuffix("_aligned")
    ctx = Ctx(args.manifest, args.pretrain_root, args.out_root)
    log = []

    def show():
        for ok, name, detail in log:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    mode = "manifest" if ctx.manifest else "pilot"
    print(f"=== own-session learnability ({mode} mode): {eid} ===\n--- audit ---")
    try:
        a = audit(eid, log, ctx)
    except AuditError as e:
        show(); print(f"\nAUDIT FAILED: {e}"); return 3
    show()
    n_audit = len(log)
    print(f"audit: {n_audit}/{n_audit} passed")
    if args.dry_run:
        return 0

    out = ctx.out_root / eid
    if out.exists() and any(p.name != "capture_meta.json" for p in out.iterdir()):
        print(f"ERROR: {out} already holds results; remove it or choose another root")
        return 2
    out.mkdir(parents=True, exist_ok=True)

    from ndt.utils.eval_utils import load_model_data_local
    from ndt.paths import config_path
    variants, trained_sd = {}, None
    for role in ctx.roles:
        model, accelerator, dataset, dataloader = load_model_data_local(
            model_config=config_path("ndt1_stitching_eval"),
            model_path=str(paths(eid, role, ctx)[0]), trainer_config=config_path("trainer_ndt1"),
            seed=42, mask_name="mask_temporal", eid=eid, data_type=DATA_TYPE,
            base_path=BASE_PATH, data_root=DATA_ROOT)

        # 4 again, on the model the eval will actually use (post accelerator.prepare)
        ok, n, detail = state_equal(model.state_dict(), a["states"][role])
        label = "4'. eval model" if not ctx.manifest else f"4'. {role}: eval model"
        log.append((ok, f"{label} == .state_dict.pt after prepare ({n})", detail))
        ok5 = len(dataset) == a["n_test"]
        log.append((ok5, "5c. eval dataset is the test split" + ("" if role == "best" else f" ({role})"),
                    f"{len(dataset)} vs test.parquet {a['n_test']}"))
        if not (ok and ok5):
            show(); return 3
        v = ctx.variant(role)
        variants[v] = run_eval(model, accelerator, dataloader, dataset, out / v, log, v)
        if role == "best":
            trained_sd = {k: t_.detach().cpu() for k, t_ in model.state_dict().items()}
        if ctx.untrained_seeds:
            trained_model = model        # the untrained loop below reuses its accelerator
        else:
            del model
            torch.cuda.empty_cache()

    # M3: untrained references are built by current code, i.e. EID-keyed (v2). A
    # trained model loaded from a pre-M3 pickle is count-keyed (v1), so bring it
    # into v2 key space before comparing. Single session, so the map is
    # unambiguous; its size was already audited (3a).
    if ctx.untrained_seeds:
        from ndt.compat import _V1_KEY, remap_v1_stitch_keys
        if any(_V1_KEY.match(k) for k in trained_sd):
            trained_sd = remap_v1_stitch_keys(trained_sd, [[eid, a["clusters"]]])

    untrained_sds = {}
    for s in ctx.untrained_seeds:
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
    cohort = pf.SOURCES[eid][0] if not ctx.manifest else ctx.manifest[eid]["cohort"]
    summary = {"eid": eid, "cohort": cohort, "clusters": a["clusters"],
               "n_train": a["n_train"], "n_test": a["n_test"],
               "source_checkpoint_sha256": a["sha256"], "source_checkpoint_role": "best",
               "source_best_epoch": a["best_epoch"], "band_rule": BAND_RULE,
               "untrained_seeds": list(ctx.untrained_seeds), "variants": variants,
               "checks": [{"ok": ok, "name": n, "detail": d} for ok, n, d in log],
               "n_checks_failed": n_fail}
    if ctx.manifest:
        entry = ctx.manifest[eid]
        run_summary = json.loads((Path(a["run_dir"]) / "run_summary.json").read_text())
        summary.update({
            "mode": "manifest", "manifest": ctx.manifest_path,
            "cell": entry["cell"], "n_val": entry["n_val"],
            "checkpoints": {r: {"sha256": a["hashes"][r], "epoch": a["epochs"][r]}
                            for r in ctx.roles},
            "pretrain_run_dir": a["run_dir"],
            "reused_pilot_pretrain": bool(entry.get("reuse_pilot_pretrain")),
            "covariates": session_covariates(eid, DATA_ROOT),
            "curve": curve_features(Path(a["run_dir"]) / "loss" / "loss_history.npz", run_summary),
        })
    (out / "own_session_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- all checks ---"); show()
    for v, r in variants.items():
        if v.startswith("untrained"):
            continue
        c = r["co_smooth"]
        print(f"\n{eid[:8]} [{v}] own-session co_smooth: mean {c['mean_bps']:+.4f} "
              f"CI [{c['ci95_lo']:+.4f}, {c['ci95_hi']:+.4f}] -> band {c['band']}")
    print(f"checks: {len(log) - n_fail}/{len(log)} passed; wrote {out/'own_session_summary.json'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
