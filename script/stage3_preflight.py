#!/usr/bin/env python
"""Stage-3 preflight: check all ten source->target pairs before spending ~41 GPU-h.

    python script/stage3_preflight.py [--pair-index N]

Seconds of checking against ten multi-hour jobs. The load-bearing check is that
no target is an official held-out session: the sealed set is what the whole
dev/held-out separation exists to protect, and a single mistyped eid would
destroy it silently, with the runs completing and looking fine.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import pyarrow.parquet as pq  # noqa: E402

DATA_ROOT = "/work/nvme/beml/ac136/LFP_FM/data/datasets"
PRETRAIN_ROOT = "/work/nvme/beml/ac136/IBL_MTM/results/pilot/pretrain"
MANIFESTS = "/projects/beml/ac136/LFP_FM/data/scaling_manifests"

SOURCES = {
    "862ade13-53cd-4221-a3fa-dda8643641f2": ("rs", 258),
    "71e55bfe-5a3a-4cba-bdc7-f085140d798e": ("rs", 1069),
    "5b49aca6-a6f4-4075-931a-617ad64c219c": ("bwm", 736),
    "b182b754-3c3e-4942-8144-6ee790926b58": ("bwm", 1255),
    "f56194bc-8215-4ae8-bc6a-89781ad8e050": ("bwm", 1899),
}
# Stage 3 uses the first development pair. The canary is the first row.
DEV_TARGETS = {
    "572a95d1-39ca-42e1-8424-5c9ffcb2df87": ("rs", 602),
    "8c552ddc-813e-4035-81cc-3971b57efe65": ("bwm", 1775),
}
CANARY = ("862ade13-53cd-4221-a3fa-dda8643641f2",
          "572a95d1-39ca-42e1-8424-5c9ffcb2df87")

results = []


def check(ok, name, detail=""):
    results.append((bool(ok), name, detail))
    return bool(ok)


def read_list(p):
    return [l.strip() for l in open(p) if l.strip()]


def clusters_of(eid):
    f = pq.ParquetFile(os.path.join(DATA_ROOT, eid, "data", "train.parquet"))
    t = f.read_row_group(0, columns=["spikes_sparse_shape"]).slice(0, 1).to_pydict()
    return t["spikes_sparse_shape"][0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-index", type=int, default=None,
                    help="check only one pair, 0-based over sorted(source) x sorted(target)")
    args = ap.parse_args()

    # The sealed set, read from the manifests rather than hardcoded -- a
    # hardcoded copy could drift from the real held-out list, and drift in this
    # particular list is exactly the failure this guards against.
    heldout = set()
    for c in ("rs", "bwm"):
        heldout |= set(read_list(os.path.join(MANIFESTS, c, "heldout_eids.txt")))
    heldin = set()
    for c in ("rs", "bwm"):
        heldin |= set(read_list(os.path.join(MANIFESTS, c, "heldin_eids.txt")))
    # Non-vacuity. If the manifest path were wrong or the files empty, `heldout`
    # would be an empty set and every "target is not held-out" check below would
    # pass without testing anything -- the guard would report success precisely
    # when it had stopped working. Assert the set is the size the design says.
    if len(heldout) != 80 or len(heldin) != 321:
        print(f"PREFLIGHT ABORTED: expected 80 held-out and 321 held-in sessions, "
              f"read {len(heldout)} and {len(heldin)} from {MANIFESTS}. "
              f"The sealed-set guard cannot be trusted until this is resolved.",
              file=sys.stderr)
        return 2
    print(f"sealed held-out set: {len(heldout)} sessions, held-in {len(heldin)} "
          f"(read from manifests, sizes asserted)\n")

    from ndt.vocab import source_checkpoint_identity, VocabularyError

    pairs = [(s, t) for s in sorted(SOURCES) for t in sorted(DEV_TARGETS)]
    if args.pair_index is not None:
        pairs = [pairs[args.pair_index]]

    for src, tgt in pairs:
        tag = f"{src[:8]}->{tgt[:8]}"
        ck = os.path.join(PRETRAIN_ROOT, f"src_{src}", "train", src, "model_best.pt")
        sd = ck.replace(".pt", ".state_dict.pt")

        check(os.path.isfile(ck), f"{tag}: source checkpoint exists", ck)
        check(os.path.isfile(sd), f"{tag}: source .state_dict.pt exists", sd)
        check(src != tgt, f"{tag}: source and target differ")
        check(tgt in DEV_TARGETS, f"{tag}: target IS a development target")
        check(tgt not in heldout,
              f"{tag}: target is NOT an official held-out eid  <-- the sealed-set guard")
        check(tgt in heldin, f"{tag}: target is held-in (dev targets are drawn from held-in)")

        try:
            n = clusters_of(tgt)
            check(n == DEV_TARGETS[tgt][1],
                  f"{tag}: target cluster count matches the plan",
                  f"data={n} plan={DEV_TARGETS[tgt][1]}")
        except Exception as e:
            check(False, f"{tag}: target data readable", f"{type(e).__name__}: {e}")

        if os.path.isfile(ck):
            try:
                ident = source_checkpoint_identity(ck, expected_eid=src,
                                                   expected_clusters=SOURCES[src][1],
                                                   role="best")
                check(True, f"{tag}: source identity asserts pass",
                      f"sha={ident['source_checkpoint_sha256'][:12]} "
                      f"epoch={ident['source_checkpoint_epoch']}")
            except VocabularyError as e:
                check(False, f"{tag}: source identity asserts pass", str(e)[:100])

    print("=" * 78)
    npass = sum(1 for ok, _, _ in results if ok)
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail and not ok:
            print(f"         {detail}")
        elif detail and "sha=" in detail:
            print(f"         {detail}")
    print("=" * 78)
    print(f"{npass}/{len(results)} checks passed over {len(pairs)} pair(s)")
    print(f"canary pair: {CANARY[0][:8]} -> {CANARY[1][:8]}")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
