#!/usr/bin/env python
"""M-pre: did the multi-session defects affect the original MtM experiments?

    python script/mpre_audit.py

Read-only. Evidence comes from three explicitly separated levels:
  upstream code        -- what 652c61c says (what the code permits or forces)
  released checkpoint  -- what multi-NDT1-baseline-34-sessions.pt proves
  paper training run   -- what can be established actually happened

The Q3/Q4/Q5a replay runs the UPSTREAM loader (extracted from 652c61c with git
archive into a temp dir -- no worktree, no repo writes) on CPU. A reproduced
defect proves it is EXECUTABLE in the historical code, not that the paper run
encountered it. Deterministic: no timestamps, fixed seeds.
"""

import collections
import hashlib
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parent.parent
UPSTREAM = "652c61c"
CKPT = Path("/work/nvme/beml/ac136/IBL_MTM/results/mtm/multi-NDT1-baseline-34-sessions.pt")
PROCESSED = Path("/work/nvme/beml/ac136/IBL_MTM/data/processed_mtm")
CANDIDATES = REPO / "script" / "mtm_train_eids_pasted.txt"
EID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
REPLAY = ["51e53aff-1d5d-4182-a684-aba783d50ae5",   # 862 clusters
          "c4432264-e1ae-446f-8a07-6280abade813",   # 862 clusters
          "862ade13-53cd-4221-a3fa-dda8643641f2"]   # 258 clusters
BATCH = 16                                          # ssl_sessions_trainer.yaml @652c61c


def git(*a):
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True,
                          check=True).stdout


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def local_dir(eid):
    for x in (eid, eid + "_aligned"):
        if (PROCESSED / x / "data").is_dir():
            return PROCESSED / x / "data"
    return None


def count(eid):
    d = local_dir(eid)
    if d is None:
        return None
    t = pq.ParquetFile(d / "train.parquet").read_row_group(
        0, columns=["spikes_sparse_shape"]).slice(0, 1).to_pydict()
    return t["spikes_sparse_shape"][0][1]


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    section("INPUT")
    cand = CANDIDATES.read_text().split()
    assert len(cand) == len(set(cand)), "duplicate EIDs in the pasted list"
    assert all(EID_RE.match(e) for e in cand), "malformed EID in the pasted list"
    target = git("show", f"{UPSTREAM}:data/target_eids.txt").split()
    test_re = git("show", f"{UPSTREAM}:data/test_re_eids.txt").split()
    print(f"pasted candidate list: {len(cand)} unique EIDs, sha256 {sha256(CANDIDATES)[:16]}")
    print(f"checkpoint filename says '34-sessions'; list has {len(cand)}: size UNRESOLVED")
    print(f"list == target_eids({len(target)}) - test_re_eids({len(test_re)}): "
          f"{set(cand) == set(target) - set(test_re)}")

    section("RELEASED CHECKPOINT")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = ck["model"]
    sd = m.state_dict()
    keys = sorted({int(re.search(r"stitcher_dict\.(\d+)\.", k).group(1))
                   for k in sd if "stitcher_dict." in k})
    dkeys = sorted({int(re.search(r"stitch_decoder_dict\.(\d+)\.", k).group(1))
                    for k in sd if "stitch_decoder_dict." in k})
    print(f"sha256 {sha256(CKPT)[:16]}  top-level {sorted(ck)}  epoch {ck['epoch']}")
    print(f"class {type(m).__module__}.{type(m).__name__}  tensors {len(sd)}")
    print(f"stitcher keys {len(keys)}, decoder keys {len(dkeys)}, identical sets: {keys == dkeys}")
    print(f"use_session {m.encoder.embedder.use_session}  vocab {len(m.encoder.embedder.eid_lookup)}  "
          f"embed_session {tuple(sd['encoder.embedder.embed_session.weight'].shape)}")

    section("Q1  candidate list -> 32 checkpoint count keys")
    c = {e: count(e) for e in cand}
    missing = [e for e, n in c.items() if n is None]
    have = {n for n in c.values() if n is not None}
    groups = collections.defaultdict(list)
    for e, n in c.items():
        if n is not None:
            groups[n].append(e)
    coll = {n: es for n, es in sorted(groups.items()) if len(es) > 1}
    print(f"counts available locally: {len(cand) - len(missing)}/{len(cand)}; "
          f"missing: {missing}")
    print(f"[observation, available counts only] unique {len(have)}; "
          f"== checkpoint keys: {have == set(keys)}; "
          f"extra {sorted(have - set(keys))}; absent {sorted(set(keys) - have)}")
    print("[observation] candidate count-collision groups among available counts:")
    for n, es in coll.items():
        print(f"   {n}: {es}")
    if missing:
        verdict = "UNDETERMINED / incomplete local data"
    elif have == set(keys):
        verdict = "EXACT MATCH"
    else:
        verdict = "INCONSISTENT"
    print(f"Q1 VERDICT: {verdict}")

    section("Q2  historical finetune: what is inherited / reset (upstream code)")
    ft = git("show", f"{UPSTREAM}:src/finetune_eval_multi_session.py").splitlines()
    for i, line in enumerate(ft, 1):
        if "load_state_dict" in line or "pop(" in line or "model = model_class(" in line:
            print(f"   L{i}: {line.strip()}")
    pops = [l for l in ft if "pop(" in l and not l.strip().startswith("#")]
    print(f"keys removed before load: {'none' if not pops else pops}")
    print("per component (upstream code): session embedding -> INHERITED in full "
          "(both models use the 40-EID target_eids vocab, shapes match);")
    print("   stitcher / stitch decoder -> INHERITED iff the target's cluster count is a pretrained "
          "key, else fresh (strict=False)")
    for e in test_re:
        n = count(e)
        print(f"   finetune target {e[:8]}: count {n}  in pretrained keys: "
              f"{(n in keys) if n is not None else 'count unavailable'}")

    section("Q3/Q4/Q5a  replay of the upstream loader path (CPU)")
    tmp = Path(tempfile.mkdtemp(prefix="mpre_652c61c_"))
    tar = subprocess.run(["git", "-C", str(REPO), "archive", UPSTREAM, "src/loader"],
                         capture_output=True, check=True).stdout
    tarfile.open(fileobj=io.BytesIO(tar)).extractall(tmp)
    print("upstream src/loader extracted to a temp dir (git archive, no repo writes)")
    sys.path.insert(0, str(tmp / "src"))
    from loader.make_loader import make_loader  # the 652c61c version
    from datasets import load_dataset, concatenate_datasets

    def load(eid, trim):
        d = load_dataset("parquet", data_files={s: str(local_dir(eid) / f"{s}.parquet")
                                                 for s in ("train", "val")})
        out = {}
        for s in ("train", "val"):
            n = len(d[s])
            # verbatim upstream rule, dataset_utils.load_ibl_dataset @652c61c
            k = n - n % BATCH if trim else n
            out[s] = d[s].select(list(range(k)))
        return out

    def replay(trim):
        np.random.seed(42); torch.manual_seed(42)
        parts = [load(e, trim) for e in REPLAY]
        res = {}
        for split in ("train", "val"):
            ds = concatenate_datasets([p[split] for p in parts])
            eids = [str(x).removesuffix("_aligned") for x in ds["eid"]]
            dl = make_loader(ds, target=None, load_meta=True, batch_size=BATCH,
                             pad_to_right=True, pad_value=-1., max_time_length=100,
                             max_space_length=668, dataset_name="ibl", sort_by_depth=False,
                             sort_by_region=False, stitching=True, shuffle=(split == "train"))
            np.random.seed(42)
            order = list(iter(dl.sampler))
            batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
            comp = [tuple(sorted({eids[j][:8] for j in b})) for b in batches]
            np.random.seed(42)
            crash = None
            try:
                for bi, b in enumerate(dl):
                    pass
            except RuntimeError as ex:
                crash = (bi if "bi" in dir() else 0, str(ex).splitlines()[0][:110])
            res[split] = (len(ds), comp, crash)
        return res, [len(p["train"]) for p in parts]

    for trim in (False, True):
        res, sizes = replay(trim)
        label = "WITH upstream trim" if trim else "WITHOUT trim (the fork's local loader)"
        print(f"\n-- {label}: per-session train sizes {sizes}")
        for split, (n, comp, crash) in res.items():
            mixed = [i for i, s in enumerate(comp) if len(s) > 1]
            pairs = collections.Counter(s for s in comp if len(s) > 1)
            print(f"   {split}: {n} trials, {len(comp)} batches; multi-EID batches {len(mixed)}"
                  f" {dict(pairs) if pairs else ''}")
            print(f"   {split}: collate {'FAILED at batch ' + str(crash[0]) + ': ' + crash[1] if crash else 'succeeded for every batch'}")
        if trim:
            same = [s for s in res["train"][1] if set(s) == {"51e53aff", "c4432264"}]
            print(f"   Q4: train batches mixing the SAME-COUNT pair (862): {len(same)}; "
                  f"forward uses batch['eid'][0] for the whole batch (upstream trainer)")
            vbuckets = len({count(e) for e in REPLAY})
            print(f"   Q5a: validation buckets keyed by count -> {vbuckets} buckets for "
                  f"{len(REPLAY)} sessions (upstream trainer: session_results[num_neuron])")

    section("Q5b  is the released .pt the selected-best checkpoint?")
    tr = git("show", f"{UPSTREAM}:src/trainer/base.py").splitlines()
    for i, line in enumerate(tr, 1):
        if "save_model(name=" in line:
            print(f"   L{i}: {line.strip()}")
    ep = ck["epoch"]
    print(f"epoch {ep}: multiple of save_every(50)? {ep % 50 == 0}; "
          f"== last epoch of a 1000-epoch run (999)? {ep == 999}")
    print("the file itself records no role; 'best' is CONSISTENT with the epoch but NOT PROVEN")

    section("Q6  evidence of a silent session-load skip in the paper run")
    searched = [CKPT.parent, REPO / "results_parity"]
    hits = []
    for root in searched:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in {".out", ".log", ".txt"} and p.stat().st_size < 5e7:
                try:
                    if "Error loading dataset" in p.read_text(errors="ignore"):
                        hits.append(str(p))
                except Exception:
                    pass
    print(f"searched: {[str(s) for s in searched]} (+ the checkpoint: no logs or metadata)")
    print(f"'Error loading dataset' occurrences: {len(hits)} {hits[:5]}")
    print("upstream code contains the silent skip (except ...: continue) -- executable, "
          "impact on the paper run UNDETERMINED without its training log")


if __name__ == "__main__":
    main()
