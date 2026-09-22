#!/usr/bin/env python
"""Capture a ground-truth fingerprint of one NDT1 tree, then diff two captures.

Port artifact, not part of the vendored package. Everything here runs on CPU in
seconds; no GPU and no Slurm.

Establish ground truth from the current code:

    cd /projects/beml/ac136/IBL_MtM_model
    python script/port_baseline.py capture --tree src --out baseline/src \
        --eid 03cf52f6-fba6-4743-a42e-dd1ac3072343_aligned

Then fingerprint the port, reusing the ground-truth batch and weights so that
"same weights + same inputs" is literally true:

    python script/port_baseline.py capture --tree ndt --out baseline/ndt \
        --eid 03cf52f6-fba6-4743-a42e-dd1ac3072343_aligned \
        --batch baseline/src/batch.pt --load-weights baseline/src/init_state.pt

    python script/port_baseline.py compare baseline/src baseline/ndt

Covers checks 1-4:
  1 config        fully resolved config for all four load paths
  2 architecture  state_dict keys, shapes, param count, session embedding, stitcher keys
  3 init          seeded parameter tensors
  4 forward       one real batch through the model in eval() mode

Determinism note: ndt1_stitching.yaml sets masker.force_active=true, which makes
the Masker fire even under eval() and draw from torch.bernoulli. Capture sets
encoder.mask=False before the forward pass, the same thing
eval_utils.load_model_data_local does, so the forward output is reproducible.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SEED = 42

# Synthetic session sizes for the multi-session architecture check. Three
# distinct neuron counts so the stitcher has to build three keys.
MULTI_NEURONS = [748, 512, 300]


# ---------------------------------------------------------------- tree adapters

def load_tree(tree):
    """Import one tree and return a uniform handle onto it."""
    os.chdir(REPO)  # the src/ tree resolves configs and data against the cwd
    if tree == "src":
        sys.path.insert(0, str(REPO / "src"))
        import utils.config_utils as cfg
        import utils.utils as u
        from utils.dataset_utils import load_ibl_dataset_locally as load
        from loader.make_loader import make_loader as mk
        from models.ndt1 import NDT1 as M
        import models.ndt1 as ndt1_mod

        def cp(name):
            return f"src/configs/{name}.yaml"
    elif tree == "ndt":
        sys.path.insert(0, str(REPO))
        import ndt.utils.config_utils as cfg
        import ndt.utils.utils as u
        from ndt.utils.dataset_utils import load_ibl_dataset_locally as load
        from ndt.loader.make_loader import make_loader as mk
        from ndt.models.ndt1 import NDT1 as M
        import ndt.models.ndt1 as ndt1_mod
        from ndt.paths import config_path as cp
    else:
        raise SystemExit(f"unknown tree: {tree}")

    return dict(cfg=cfg, u=u, load=load, mk=mk, M=M, cp=cp, mod=ndt1_mod)


def resolve_configs(t):
    """The four config load paths, resolved exactly as each entry point does."""
    cfg, cp = t["cfg"], t["cp"]
    from_kw, upd = cfg.config_from_kwargs, cfg.update_config
    out = {}

    c = from_kw({"model": "include:" + cp("ndt1_stitching")})
    c = upd(cp("ndt1_stitching"), c)
    out["single"] = upd(cp("ssl_session_trainer"), c)

    c = from_kw({"model": "include:" + cp("ndt1_stitching")})
    c = upd(cp("ndt1_stitching"), c)
    out["multi"] = upd(cp("ssl_sessions_trainer"), c)

    c = from_kw({"model": "include:" + cp("ndt1_stitching")})
    out["finetune"] = upd(cp("finetune_sessions_trainer"), c)

    mc = cp("ndt1_stitching_eval")
    c = from_kw({"model": "include:" + mc})
    c = upd(mc, c)
    out["eval"] = upd(cp("trainer_ndt1"), c)
    return out


# ---------------------------------------------------------------- helpers

def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, set):
        return sorted(str(v) for v in o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return repr(o)


def write_json(path, obj):
    Path(path).write_text(json.dumps(plain(obj), sort_keys=True, indent=2) + "\n")


def tensors_equal(a, b):
    """Equality that treats NaN as equal to NaN.

    torch.equal compares elementwise, and NaN != NaN, so it reports two
    identical all-NaN placeholders as different. The loader emits exactly such
    placeholders for neuron_depths, neuron_regions and target when
    load_meta=False and target=None, so a naive check produces false failures.
    Conversely, plain subtraction hides a real difference when one side is NaN,
    because NaN > 0 is False.
    """
    import torch
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape:
            return False
        if a.is_floating_point() and b.is_floating_point():
            return bool(((a == b) | (a.isnan() & b.isnan())).all())
        return bool(torch.equal(a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(tensors_equal(a[k], b[k]) for k in a)
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


def max_abs_diff(a, b):
    """Worst elementwise deviation, ignoring positions that are NaN in both."""
    import torch
    if a.shape != b.shape:
        return float("inf")
    x, y = a.double(), b.double()
    both_nan = x.isnan() & y.isnan()
    d = (x - y).abs()
    d = torch.where(both_nan, torch.zeros_like(d), d)
    d = torch.where(d.isnan(), torch.full_like(d, float("inf")), d)
    return float(d.max().item()) if d.numel() else 0.0


# ---------------------------------------------------------------- capture

def capture(args):
    import torch

    t = load_tree(args.tree)
    out = Path(args.out).resolve()
    (out / "configs").mkdir(parents=True, exist_ok=True)

    # ---- check 1: resolved configs -------------------------------------
    configs = resolve_configs(t)
    for name, c in configs.items():
        write_json(out / "configs" / f"{name}.json", c)
    print(f"[1] wrote {len(configs)} resolved configs")

    base = configs["single"]

    # ---- check 5: this tree's own loader output -------------------------
    # Always built, even when --batch supplies a reference batch for the
    # forward pass. Otherwise the port's loader is never exercised and
    # loader/base.py goes untested.
    if not args.eid:
        raise SystemExit("--eid is required")
    t["u"].set_seed(SEED)
    # The two trees no longer share a signature: the vendored one has been
    # pruned of split_method, batch_size and seed, which the original still
    # accepts. Offer every argument and let each tree take what it declares,
    # so this keeps working as ndt continues to shrink.
    import inspect
    offered = dict(
        eid=args.eid, num_sessions=1, split_method=base.data.split_method,
        train_session_eid=[args.eid], test_session_eid=base.data.test_session_eid,
        batch_size=base.training.train_batch_size, seed=base.seed,
        just_spikes=True, data_type=args.data_type, base_path=args.base_path)
    accepted = set(inspect.signature(t["load"]).parameters)
    skipped = sorted(k for k in offered if k not in accepted)
    if skipped:
        print(f"[.] {args.tree} tree does not accept {skipped}; omitted")
    tr, _, _, meta = t["load"](**{k: v for k, v in offered.items() if k in accepted})
    t["u"].set_seed(SEED)
    dl = t["mk"](tr, target=None, load_meta=False,
                 batch_size=base.training.train_batch_size, pad_to_right=True,
                 pad_value=-1., max_time_length=base.data.max_time_length,
                 max_space_length=base.data.max_space_length,
                 dataset_name=base.data.dataset_name,
                 sort_by_depth=base.data.sort_by_depth,
                 sort_by_region=base.data.sort_by_region,
                 stitching=base.encoder.stitching, shuffle=False)
    own_batch = next(iter(dl))
    torch.save(own_batch, out / "batch.pt")
    write_json(out / "meta_data.json", meta)
    print(f"[5] built own batch from this tree's loader, "
          f"spikes {tuple(own_batch['spikes_data'].shape)}")

    # the batch actually fed to the model: the reference one when given
    if args.batch:
        batch = torch.load(args.batch, weights_only=False)
        print(f"[.] forward pass will use the reference batch {args.batch}")
    else:
        batch = own_batch

    # ---- checks 2 and 3: architecture and seeded init -------------------
    c = t["cfg"].update_config(base, meta)
    t["u"].set_seed(SEED)
    model = t["M"](c.model, **c.method.model_kwargs, **meta)
    model.eval()

    sd = model.state_dict()
    emb = sd.get("encoder.embedder.embed_session.weight")
    arch = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "n_state_dict_tensors": len(sd),
        "session_embedding_shape": list(emb.shape) if emb is not None else None,
        # The ndt tree dropped the module-level `include_eids` in 1c-i: the session
        # vocabulary is passed into the model now. Fall back to the embedding's row
        # count, which is the same number (one row per vocabulary entry), so captures
        # taken before and after that change stay comparable.
        "target_eids_entries": (len(t["mod"].include_eids)
                                if hasattr(t["mod"], "include_eids")
                                else (emb.shape[0] if emb is not None else None)),
        "stitcher_keys": sorted(model.encoder.stitcher.stitcher_dict.keys())
        if hasattr(model.encoder, "stitcher") else None,
        "stitch_decoder_keys": sorted(model.stitch_decoder.stitch_decoder_dict.keys())
        if hasattr(model, "stitch_decoder") else None,
        "state_dict": {k: list(v.shape) for k, v in sd.items()},
    }
    write_json(out / "architecture.json", arch)
    torch.save(sd, out / "init_state.pt")
    print(f"[2] {arch['total_parameters']:,} params, {arch['n_state_dict_tensors']} tensors, "
          f"stitcher keys {arch['stitcher_keys']}")
    print(f"[3] saved seeded init_state.pt")

    # ---- check 6: multi-session architecture ----------------------------
    # The single-session model above has one stitcher key, so it never
    # exercises multi-session stitching. Build a synthetic three-session model
    # to compare stitcher and stitch-decoder key sets and their weights.
    multi_cfg = resolve_configs(t)["multi"]
    multi_meta = {"num_neurons": MULTI_NEURONS, "num_sessions": len(MULTI_NEURONS),
                  "eids": {args.eid.replace("_aligned", "")}, "just_spikes": True}
    mc = t["cfg"].update_config(multi_cfg, multi_meta)
    t["u"].set_seed(SEED)
    mmodel = t["M"](mc.model, **mc.method.model_kwargs, **multi_meta)
    msd = mmodel.state_dict()
    write_json(out / "architecture_multi.json", {
        "num_neurons": MULTI_NEURONS,
        "total_parameters": sum(p.numel() for p in mmodel.parameters()),
        "n_state_dict_tensors": len(msd),
        "stitcher_keys": sorted(mmodel.encoder.stitcher.stitcher_dict.keys()),
        "stitch_decoder_keys": sorted(mmodel.stitch_decoder.stitch_decoder_dict.keys()),
        "state_dict": {k: list(v.shape) for k, v in msd.items()},
    })
    torch.save(msd, out / "init_state_multi.pt")
    print(f"[6] multi-session model {MULTI_NEURONS}: "
          f"stitcher keys {sorted(mmodel.encoder.stitcher.stitcher_dict.keys())}")
    del mmodel, msd

    # ---- optional: adopt the reference weights --------------------------
    if args.load_weights:
        ref = torch.load(args.load_weights, weights_only=False)
        missing, unexpected = model.load_state_dict(ref, strict=False)
        print(f"[.] loaded reference weights from {args.load_weights} "
              f"(missing {len(missing)}, unexpected {len(unexpected)})")
        if missing or unexpected:
            write_json(out / "weight_load_mismatch.json",
                       {"missing": list(missing), "unexpected": list(unexpected)})

    # ---- check 4: forward pass ------------------------------------------
    model.eval()
    model.encoder.mask = False  # force_active=true would otherwise randomise eval
    t["u"].set_seed(SEED)
    with torch.no_grad():
        o = model(batch["spikes_data"],
                  time_attn_mask=batch["time_attn_mask"],
                  space_attn_mask=batch["space_attn_mask"],
                  spikes_timestamps=batch["spikes_timestamps"],
                  spikes_spacestamps=batch["spikes_spacestamps"],
                  targets=batch["target"],
                  neuron_regions=batch["neuron_regions"],
                  masking_mode=c.encoder.masker.mode,
                  num_neuron=batch["spikes_data"].shape[2],
                  eid=batch["eid"][0])
    torch.save({"preds": o.preds, "targets": o.targets,
                "loss": o.loss, "n_examples": o.n_examples}, out / "forward.pt")
    write_json(out / "forward_summary.json", {
        "loss": float(o.loss.item()),
        "n_examples": int(o.n_examples),
        "preds_shape": list(o.preds.shape),
        "preds_sum": float(o.preds.double().sum().item()),
        "preds_mean": float(o.preds.double().mean().item()),
        "preds_min": float(o.preds.min().item()),
        "preds_max": float(o.preds.max().item()),
    })
    print(f"[4] forward loss {float(o.loss.item()):.6f}, preds {tuple(o.preds.shape)}")

    write_json(out / "manifest.json", {
        "tree": args.tree, "eid": args.eid, "seed": SEED,
        "encoder_mask_disabled": True,
        "masking_mode": c.encoder.masker.mode,
        "base_path": args.base_path, "data_type": args.data_type,
        "torch": torch.__version__, "python": sys.version.split()[0],
        "reused_batch": args.batch, "loaded_weights": args.load_weights,
    })
    print(f"\nwrote capture to {out}")


# ---------------------------------------------------------------- compare

def compare(args):
    import torch
    a, b = Path(args.a).resolve(), Path(args.b).resolve()
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if detail:
            for line in str(detail).rstrip().split("\n")[:20]:
                print(f"         {line}")
        if not ok:
            fails.append(label)

    print(f"\n{a.name}  vs  {b.name}")

    print("\nCheck 1: resolved config")
    for f in sorted((a / "configs").glob("*.json")):
        g = b / "configs" / f.name
        if not g.exists():
            check(f"{f.stem} path present", False, "missing in second capture")
            continue
        ta, tb = f.read_text(), g.read_text()
        d = [f"line {i+1}:\n  A: {x.strip()}\n  B: {y.strip()}"
             for i, (x, y) in enumerate(zip(ta.split("\n"), tb.split("\n"))) if x != y]
        check(f"{f.stem} config identical", not d, "\n".join(d[:5]))

    print("\nCheck 2: architecture")
    ja = json.loads((a / "architecture.json").read_text())
    jb = json.loads((b / "architecture.json").read_text())
    for k in ["total_parameters", "n_state_dict_tensors", "session_embedding_shape",
              "target_eids_entries", "stitcher_keys", "stitch_decoder_keys"]:
        check(f"{k} == {ja[k]}", ja[k] == jb[k], "" if ja[k] == jb[k] else f"A={ja[k]}  B={jb[k]}")
    sa, sb = ja["state_dict"], jb["state_dict"]
    only_a, only_b = sorted(set(sa) - set(sb)), sorted(set(sb) - set(sa))
    mism = [f"{k}: A{sa[k]} B{sb[k]}" for k in sorted(set(sa) & set(sb)) if sa[k] != sb[k]]
    check("state_dict keys and shapes match", not (only_a or only_b or mism),
          "\n".join(([f"only in A: {only_a[:8]}"] if only_a else [])
                    + ([f"only in B: {only_b[:8]}"] if only_b else []) + mism[:8]))

    print("\nCheck 3: initialization")
    pa = torch.load(a / "init_state.pt", weights_only=False)
    pb = torch.load(b / "init_state.pt", weights_only=False)
    shared = sorted(set(pa) & set(pb))
    worst, worst_k = 0.0, None
    nz = []
    for k in shared:
        if not tensors_equal(pa[k], pb[k]):
            nz.append(k)
            d = max_abs_diff(pa[k], pb[k])
            if d > worst:
                worst, worst_k = d, k
    check(f"all {len(shared)} init tensors equal", not nz,
          "" if not nz else f"{len(nz)} differ; worst {worst_k} max|d|={worst:.3e}")

    print("\nCheck 4: forward pass")
    fa = torch.load(a / "forward.pt", weights_only=False)
    fb = torch.load(b / "forward.pt", weights_only=False)
    same_shape = fa["preds"].shape == fb["preds"].shape
    check(f"preds shape {tuple(fa['preds'].shape)}", same_shape,
          "" if same_shape else f"A={tuple(fa['preds'].shape)} B={tuple(fb['preds'].shape)}")
    if same_shape:
        md = max_abs_diff(fa["preds"], fb["preds"])
        ld = abs(float(fa["loss"]) - float(fb["loss"]))
        print(f"         max|preds_A - preds_B| = {md:.6e}")
        print(f"         |loss_A - loss_B|      = {ld:.6e}   "
              f"(A={float(fa['loss']):.6f}  B={float(fb['loss']):.6f})")
        check("max abs preds difference is zero", md == 0.0)
        if md != 0.0:
            check("max abs preds difference below 1e-5", md < 1e-5)

    print("\nCheck 5: loader output, each tree building its own batch")
    ba = torch.load(a / "batch.pt", weights_only=False)
    bb = torch.load(b / "batch.pt", weights_only=False)
    keys_ok = set(ba) == set(bb)
    check("batch keys match", keys_ok,
          "" if keys_ok else f"only A: {sorted(set(ba)-set(bb))}  only B: {sorted(set(bb)-set(ba))}")
    diff_fields = [k for k in sorted(set(ba) & set(bb)) if not tensors_equal(ba[k], bb[k])]
    check(f"all {len(set(ba) & set(bb))} batch fields identical", not diff_fields,
          "" if not diff_fields else "differ: " + ", ".join(diff_fields))

    print("\nCheck 6: multi-session architecture")
    fa6, fb6 = a / "architecture_multi.json", b / "architecture_multi.json"
    if not (fa6.exists() and fb6.exists()):
        print("  [SKIP] one capture predates the multi-session check; recapture both")
    else:
        ma, mb = json.loads(fa6.read_text()), json.loads(fb6.read_text())
        for k in ["num_neurons", "total_parameters", "n_state_dict_tensors",
                  "stitcher_keys", "stitch_decoder_keys"]:
            check(f"multi {k} == {ma[k]}", ma[k] == mb[k],
                  "" if ma[k] == mb[k] else f"A={ma[k]}  B={mb[k]}")
        sa6, sb6 = ma["state_dict"], mb["state_dict"]
        mism6 = [f"{k}: A{sa6[k]} B{sb6[k]}" for k in sorted(set(sa6) & set(sb6))
                 if sa6[k] != sb6[k]]
        check("multi state_dict keys and shapes match",
              set(sa6) == set(sb6) and not mism6, "\n".join(mism6[:8]))
        qa = torch.load(a / "init_state_multi.pt", weights_only=False)
        qb = torch.load(b / "init_state_multi.pt", weights_only=False)
        nz6 = [k for k in sorted(set(qa) & set(qb)) if not tensors_equal(qa[k], qb[k])]
        check(f"all {len(set(qa) & set(qb))} multi-session weights equal", not nz6,
              "" if not nz6 else f"{len(nz6)} differ, first: {nz6[:6]}")

    print("\n" + "-" * 60)
    if fails:
        print(f"{len(fails)} check(s) failed:")
        for f in fails:
            print(f"  FAIL {f}")
    else:
        print("all checks passed; the two trees are numerically identical")
    print("-" * 60)
    sys.exit(1 if fails else 0)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="fingerprint one tree")
    c.add_argument("--tree", choices=["src", "ndt"], required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--eid", default=None)
    c.add_argument("--base-path", default="/work/nvme/beml/ac136/IBL_MTM/data")
    c.add_argument("--data-type", default="processed_mtm")
    c.add_argument("--batch", default=None, help="reuse a batch.pt from another capture")
    c.add_argument("--load-weights", default=None,
                   help="load an init_state.pt from another capture before the forward pass")
    c.set_defaults(fn=capture)

    d = sub.add_parser("compare", help="diff two captures")
    d.add_argument("a")
    d.add_argument("b")
    d.set_defaults(fn=compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
