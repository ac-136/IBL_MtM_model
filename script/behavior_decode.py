#!/usr/bin/env python
"""Frozen-NDT1 behaviour decoding probe for one held-in session.

    python script/behavior_decode.py EID [--dry-run] [--epochs 2000]

The session's pretrained NDT1 (its BEST checkpoint) is loaded and never
updated: every pretrained parameter must equal the durable ``.state_dict.pt``
before feature extraction, after it, and after head training. Only four small
behaviour heads (choice, block, wheel speed, whisker motion energy) are
trained, on encoder features cached once per split with masking off.

Split roles are strict: train fits the target preprocessing, the heads and
the chance/null; val selects the joint epoch (lowest val total loss); test is
evaluated once, with the joint-epoch heads only. Each target's own best val
epoch is recorded as a diagnostic and never evaluated on test.

Components, parity choices and departures: ``decode_components.py``.
--dry-run runs the audits only: CPU, no feature extraction, no training.
Exit codes: 0 ok, 1 a post-run check failed, 2 output exists, 3 audit failed.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402

import decode_components as dc  # noqa: E402
import stage3_preflight as pf  # noqa: E402

RESULTS = Path("/work/nvme/beml/ac136/IBL_MTM/results")
LEARN_MANIFEST = REPO / "script" / "learnability_manifest.json"
LEARN_PRETRAIN = RESULTS / "learnability" / "pretrain"
DATA_ROOT = "/work/nvme/beml/ac136/LFP_FM/data/datasets_wheel"
OUT_ROOT = RESULTS / "decoding" / "probe"
BASE_PATH = "/work/nvme/beml/ac136/IBL_MTM/data"
DATA_TYPE = "processed_mtm"
N_STATE_TENSORS = 94
BEHAVIOUR_COLUMNS = ("choice", "block", "reward", "wheel-speed", "whisker-motion-energy")

# Optimisation defaults from NEDS's single-session config (trainer_mm.yaml,
# train.py:296-320): a starting point, not a parity claim.
LR, WD, EPS, BATCH, WARMUP, DIV_FACTOR = 2e-4, 0.01, 1e-8, 16, 0.15, 10


class AuditError(Exception):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def state_digest(sd):
    """Order-independent sha256 over every tensor's name, dtype, shape and bytes."""
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    return h.hexdigest()


def state_equal(a, b):
    if set(a) != set(b):
        return False, f"key sets differ: {sorted(set(a) ^ set(b))[:4]}"
    bad = [k for k in a if not torch.equal(a[k].detach().cpu(), b[k].detach().cpu())]
    return not bad, (f"{len(bad)} differ: {bad[:4]}" if bad else f"all {len(a)} equal")


def source_run(eid):
    """Where this session's own pretrain lives: learnability manifest first
    (reused pilot runs included), then the five pilot sources."""
    manifest = {s["eid"]: s for s in json.loads(LEARN_MANIFEST.read_text())["sessions"]}
    if eid in manifest:
        reuse = manifest[eid].get("reuse_pilot_pretrain")
        return (Path(reuse) if reuse else LEARN_PRETRAIN / f"src_{eid}" / "train" / eid), "learnability"
    if eid in pf.SOURCES:
        return Path(pf.PRETRAIN_ROOT) / f"src_{eid}" / "train" / eid, "pilot"
    return None, None


def read_behaviour(eid, data_root):
    raw = {}
    for split in dc.SPLITS:
        t = pq.read_table(os.path.join(data_root, eid, "data", f"{split}.parquet"),
                          columns=list(BEHAVIOUR_COLUMNS)).to_pydict()
        raw[split] = {
            "choice": np.array([v[0] for v in t["choice"]], dtype=float),
            "block": np.array([v[0] for v in t["block"]], dtype=float),
            "reward": np.array([v[0] for v in t["reward"]], dtype=float),
            "wheel-speed": np.array(t["wheel-speed"], dtype=float),
            "whisker-motion-energy": np.array(t["whisker-motion-energy"], dtype=float),
        }
    return raw


def audit(eid, data_root, log):
    def need(ok, name, detail=""):
        log.append((bool(ok), name, detail))
        if not ok:
            raise AuditError(f"{name}: {detail}")

    heldout, heldin = set(), set()
    for c in ("rs", "bwm"):
        heldout |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldout_eids.txt")))
        heldin |= set(pf.read_list(os.path.join(pf.MANIFESTS, c, "heldin_eids.txt")))
    need(len(heldout) == 80 and len(heldin) == 321, "a. manifests read at expected size",
         f"{len(heldout)} held-out / {len(heldin)} held-in")
    need(eid not in heldout, "b. eid is NOT an official held-out session", eid[:8])
    need(eid in heldin, "c. eid is held-in", eid[:8])

    run_dir, origin = source_run(eid)
    need(run_dir is not None, "d. eid has an own-session pretrain (learnability or pilot)", eid[:8])
    best, best_sd = run_dir / "model_best.pt", run_dir / "model_best.state_dict.pt"
    need(best.is_file() and best_sd.is_file(), "e. best checkpoint and state_dict exist", str(run_dir))

    cols = pq.ParquetFile(os.path.join(data_root, eid, "data", "train.parquet")).schema_arrow.names
    need(all(c in cols for c in BEHAVIOUR_COLUMNS), "f. behaviour columns present (incl. wheel-speed)",
         f"missing {[c for c in BEHAVIOUR_COLUMNS if c not in cols]}")

    from ndt.compat import load_pickled_model
    from ndt.vocab import stitcher_session_sizes
    ck_model, _vocab, _rule, version = load_pickled_model(str(best))
    sd = torch.load(best_sd, map_location="cpu", weights_only=False)["state_dict"]
    ok, detail = state_equal(ck_model.state_dict(), sd)
    need(ok and len(sd) == N_STATE_TENSORS, "g. pickled best == .state_dict.pt", detail)
    t = pq.ParquetFile(os.path.join(data_root, eid, "data", "train.parquet")).read_row_group(
        0, columns=["spikes_sparse_shape"]).slice(0, 1).to_pydict()
    clusters = t["spikes_sparse_shape"][0][1]
    sizes = stitcher_session_sizes(ck_model, [eid])
    need(sizes == {eid: clusters}, "h. stitcher sized for the session's cluster count",
         f"{sizes} vs parquet {clusters}")

    return {"run_dir": str(run_dir), "origin": origin, "best": str(best),
            "best_sha256": sha256_file(best), "state_sha256": sha256_file(best_sd),
            "state_digest": state_digest(sd), "format_version": version,
            "clusters": clusters, "state": sd}


def extract_features(eid, a, data_root, log):
    """Frozen-encoder features (n, T, H) for every split, masking off."""
    from datasets import load_dataset
    from ndt.loader.make_loader import make_loader
    from ndt.paths import config_path
    from ndt.utils.eval_utils import load_model_data_local
    from ndt.utils.utils import move_batch_to_device

    model, accelerator, _, _ = load_model_data_local(
        model_config=config_path("ndt1_stitching_eval"), model_path=a["best"],
        trainer_config=config_path("trainer_ndt1"), seed=42, mask_name="mask_temporal",
        eid=eid, data_type=DATA_TYPE, base_path=BASE_PATH, data_root=data_root)
    model.eval()
    model.encoder.mask = False
    for p in model.parameters():
        p.requires_grad_(False)

    ok, detail = state_equal(model.state_dict(), a["state"])
    log.append((ok, "i. eval model == .state_dict.pt before extraction", detail))
    log.append((not model.encoder.masker.force_active and not model.training,
                "j. masking off (eval mode, force_active False)", ""))

    d = os.path.join(data_root, eid, "data")
    ds = load_dataset("parquet", data_files={s: os.path.join(d, f"{s}.parquet") for s in dc.SPLITS})
    feats, checked = {}, False
    with torch.no_grad():
        for split in dc.SPLITS:
            loader = make_loader(ds[split], target=None, batch_size=256, pad_to_right=True,
                                 pad_value=-1., max_time_length=model.encoder.max_F,
                                 max_space_length=a["clusters"], dataset_name="ibl",
                                 load_meta=False, shuffle=False)
            out = []
            for batch in loader:
                batch = move_batch_to_device(batch, accelerator.device)
                spikes = batch["spikes_data"]
                n_neuron = spikes.shape[2]
                x, _ = model.encoder(spikes, batch["time_attn_mask"], batch["spikes_timestamps"],
                                     None, None, None, model.encoder.masker.mode, None,
                                     n_neuron, batch["eid"][0])
                if model.use_session:
                    x = x[:, x.shape[1] - spikes.shape[1]:]
                if not checked:
                    # The features are exactly what NDT1 decodes its rates from.
                    full = model(spikes, time_attn_mask=batch["time_attn_mask"],
                                 space_attn_mask=batch["space_attn_mask"],
                                 spikes_timestamps=batch["spikes_timestamps"],
                                 spikes_spacestamps=batch["spikes_spacestamps"],
                                 masking_mode=model.encoder.masker.mode,
                                 num_neuron=n_neuron, eid=batch["eid"][0])
                    rates = model.stitch_decoder(x, eid=batch["eid"][0], num_neuron=n_neuron)
                    log.append((torch.equal(rates, full.preds),
                                "k. cached features reproduce NDT1's own predictions exactly",
                                split))
                    checked = True
                out.append(x.float().cpu())
            feats[split] = torch.cat(out)
            log.append((feats[split].shape[0] == len(ds[split]),
                        f"l. {split} features cover every trial", f"{tuple(feats[split].shape)}"))

    ok, detail = state_equal(model.state_dict(), a["state"])
    log.append((ok, "m. eval model == .state_dict.pt after extraction", detail))
    return model, accelerator, feats


def to_device(prepared, split, device):
    return {n: {"y": torch.as_tensor(t["y"], device=device,
                                     dtype=torch.long if n in dc.STATIC else torch.float32),
                "mask": torch.as_tensor(t["mask"], device=device)}
            for n, t in prepared["targets"][split].items()}


def slice_targets(targets, idx):
    return {n: {"y": t["y"][idx], "mask": t["mask"][idx]} for n, t in targets.items()}


def train_heads(feats, prepared, epochs, seed, device, model, log):
    names = dc.active_targets(prepared)
    torch.manual_seed(seed)
    n_tokens, hidden = feats["train"].shape[1], feats["train"].shape[2]
    heads = dc.build_heads(names, n_tokens, hidden).to(device)

    head_ids = {id(p) for p in heads.parameters()}
    model_ids = {id(p) for p in model.parameters()}
    opt = torch.optim.AdamW(heads.parameters(), lr=LR, weight_decay=WD, eps=EPS)
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    log.append((opt_ids == head_ids and not (opt_ids & model_ids),
                "n. optimizer holds only head parameters", f"{len(opt_ids)} tensors"))

    x = {s: feats[s].to(device) for s in dc.SPLITS}
    tg = {s: to_device(prepared, s, device) for s in dc.SPLITS}
    n_train = x["train"].shape[0]
    steps = math.ceil(n_train / BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, total_steps=epochs * steps, max_lr=LR, pct_start=WARMUP,
        div_factor=DIV_FACTOR, anneal_strategy="cos")
    gen = torch.Generator().manual_seed(seed)

    # Fixed before training: train-derived null predictor scored on val.
    null = dc.null_val_losses(prepared["targets"]["train"], prepared["targets"]["val"])
    log.append((set(null) == set(names) and all(v > 0 and math.isfinite(v) for v in null.values()),
                "r. null val loss finite and positive for every active target",
                ", ".join(f"{n} {v:.4g}" for n, v in null.items())))

    hist = {f"{s}_{k}": [] for s in ("train", "val") for k in ["total", *names]}
    hist["val_selection_score"] = []
    best = {"epoch": -1, "score": math.inf, "val_total": math.inf, "state": None}
    per_target = {n: {"epoch": -1, "val_loss": math.inf} for n in names}

    for epoch in range(epochs):
        heads.train()
        perm = torch.randperm(n_train, generator=gen).to(device)
        for i in range(steps):
            idx = perm[i * BATCH:(i + 1) * BATCH]
            loss, _ = dc.total_loss(heads, x["train"][idx], slice_targets(tg["train"], idx))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        heads.eval()
        with torch.no_grad():
            for s in ("train", "val"):
                total, comps = dc.total_loss(heads, x[s], tg[s])
                hist[f"{s}_total"].append(float(total))
                for n in names:
                    hist[f"{s}_{n}"].append(float(comps[n][0]))
        score = dc.selection_score({n: hist[f"val_{n}"][-1] for n in names}, null)
        hist["val_selection_score"].append(score)
        if score < best["score"]:
            best = {"epoch": epoch, "score": score, "val_total": hist["val_total"][-1],
                    "state": {k: t.detach().clone() for k, t in heads.state_dict().items()}}
        for n in names:
            if hist[f"val_{n}"][-1] < per_target[n]["val_loss"]:
                per_target[n] = {"epoch": epoch, "val_loss": hist[f"val_{n}"][-1]}

    heads.load_state_dict(best["state"])
    selection = {
        "rule": ("joint epoch = argmin over epochs of mean_t(val_loss_t / null_val_loss_t), "
                 "equally weighted over active targets (first occurrence); only it is "
                 "test-evaluated. Training minimises the unweighted sum of the train losses."),
        "null_val_loss": null,
        "epochs": epochs, "joint_epoch": best["epoch"], "joint_score": best["score"],
        "joint_val_total": best["val_total"],
        "per_target_diagnostic": {
            n: {"own_best_epoch": per_target[n]["epoch"],
                "own_best_val_loss": per_target[n]["val_loss"],
                "val_loss_at_joint_epoch": hist[f"val_{n}"][best["epoch"]],
                "normalized_val_loss_at_joint_epoch": hist[f"val_{n}"][best["epoch"]] / null[n]}
            for n in names},
        "note": "per-target epochs are diagnostic only; no weights kept, never test-evaluated",
    }
    return heads, hist, selection


def predict(heads, feat, device):
    heads.eval()
    out = {}
    with torch.no_grad():
        f = feat.to(device)
        for n, head in heads.items():
            p = head(f)
            out[n] = (p.argmax(-1) if n in dc.STATIC else p).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eid")
    ap.add_argument("--dry-run", action="store_true", help="audits only; no GPU work")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    eid = args.eid.removesuffix("_aligned")

    log = []

    def show():
        for ok, name, detail in log:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    print(f"=== behaviour decoding probe: {eid} ===\n--- audit ---")
    try:
        a = audit(eid, args.data_root, log)
    except AuditError as e:
        show(); print(f"\nAUDIT FAILED: {e}"); return 3
    raw = read_behaviour(eid, args.data_root)
    prepared = dc.prepare_targets(raw)
    show()
    print(f"source: {a['origin']} {a['run_dir']} (format v{a['format_version']})")
    for n in dc.TARGETS:
        i = prepared["info"][n]
        print(f"  target {n:22s} {'DROPPED: ' + i['dropped'] if i['dropped'] else 'active'}")
    if args.dry_run:
        return 0

    out = Path(args.out_root) / eid
    if out.exists() and any(p.name != "capture_meta.json" for p in out.iterdir()):
        print(f"ERROR: {out} already holds results"); return 2
    out.mkdir(parents=True, exist_ok=True)

    model, accelerator, feats = extract_features(eid, a, args.data_root, log)
    device = accelerator.device
    for s in dc.SPLITS:
        n_rows = len(raw[s]["choice"])
        log.append((feats[s].shape[0] == n_rows, f"o. {s} features align with behaviour rows",
                    f"{feats[s].shape[0]} vs {n_rows}"))

    heads, hist, selection = train_heads(feats, prepared, args.epochs, args.seed, device, model, log)
    ok, detail = state_equal(model.state_dict(), a["state"])
    log.append((ok, "p. pretrained NDT parameters unchanged after head training", detail))
    log.append((state_digest(model.state_dict()) == a["state_digest"],
                "q. pretrained NDT state digest unchanged", a["state_digest"][:16]))

    # Test, once, with the joint-epoch heads only.
    preds = predict(heads, feats["test"], device)
    test = prepared["targets"]["test"]
    cond = np.stack([raw["test"][c] for c in dc.PSTH_CONDITIONS], axis=1)
    metrics = {}
    for n in dc.active_targets(prepared):
        metrics[n] = dc.metrics_for(n, test[n], preds[n], cond)
        metrics[n]["ci95"] = dc.bootstrap_ci(n, test[n], preds[n], cond)

    n_fail = sum(not ok for ok, _, _ in log)
    print("\n--- all checks ---"); show()
    print(f"\njoint epoch {selection['joint_epoch']} (selection score {selection['joint_score']:.5f}, "
          f"val total {selection['joint_val_total']:.5f})")
    for n, m in metrics.items():
        own = selection["per_target_diagnostic"][n]
        print(f"  {n:22s} " + "  ".join(f"{k} {v:+.4f}" for k, v in m.items() if isinstance(v, float))
              + f"  [n_eval {m['n_eval']}; own best val epoch {own['own_best_epoch']}]")

    torch.save({k: v.cpu() for k, v in heads.state_dict().items()}, out / "heads.pt")
    np.savez(out / "loss_history.npz", **{k: np.asarray(v) for k, v in hist.items()})
    np.savez(out / "preds.npz", **{f"{n}_{k}": v for n in preds
                                   for k, v in (("pred", preds[n]), ("y", test[n]["y"]),
                                                ("mask", test[n]["mask"]))},
             conditions=cond)
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    (out / "scaling.json").write_text(json.dumps(prepared["info"], indent=2))
    (out / "metrics.json").write_text(json.dumps({
        "eid": eid, "source": {k: a[k] for k in ("origin", "run_dir", "best", "best_sha256",
                                                 "state_sha256", "state_digest", "format_version",
                                                 "clusters")},
        "data_root": args.data_root, "seed": args.seed,
        "optim": {"lr": LR, "wd": WD, "eps": EPS, "batch": BATCH, "warmup": WARMUP,
                  "div_factor": DIV_FACTOR, "epochs": args.epochs},
        "primary_metric": dc.PRIMARY,
        "psth_r2": "secondary, point estimate only (no CI), reported with its cell sizes",
        "split_roles": {"train": "scaling, drops, head fit, null predictor",
                        "val": "joint epoch selection (null-normalized score)",
                        "test": "metrics, once, joint epoch only"},
        "metrics": metrics, "joint_epoch": selection["joint_epoch"],
        "checks": [{"ok": ok, "name": nm, "detail": d} for ok, nm, d in log],
        "n_checks_failed": n_fail}, indent=2))
    print(f"checks: {len(log) - n_fail}/{len(log)} passed; wrote {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
