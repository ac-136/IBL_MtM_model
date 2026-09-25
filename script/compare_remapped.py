#!/usr/bin/env python
"""Three-part gate for M3: a v1 (count-keyed) reference vs a v2 (EID-keyed) run.

    python script/compare_remapped.py REF_DIR NEW_DIR

compare_runs.py matches checkpoint tensors by key name, so M3's intended renames
would read as failures there. For every checkpoint and state_dict common to both
trees, this renames REF's count keys to EIDs with the map NEW recorded in its
run_summary.json (under remap_v1_stitch_keys's injectivity rule), then requires:
  1. values exact after the remap;
  2. key changes confined to stitcher_dict / stitch_decoder_dict;
  3. everything else -- every other tensor, loss_history.npz, bps.npy, r2.npy --
     bit-identical.
run_summary.json differences are reported; any key outside the expected set
(the format version and source-provenance fields M3 changes) is a failure.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ndt.compat import load_checkpoint, remap_v1_stitch_keys  # noqa: E402

PROJ = (".stitcher_dict.", ".stitch_decoder_dict.")
EXPECTED_SUMMARY_CHANGES = {"checkpoint_format_version", "source_checkpoint_path",
                            "source_checkpoint_sha256", "source_checkpoint_format_version"}


def maybe_remap(sd, smap):
    """Remap only a count-keyed (v1) state_dict; a v2 one has no count keys."""
    from ndt.compat import _V1_KEY
    return remap_v1_stitch_keys(sd, smap) if any(_V1_KEY.match(k) for k in sd) else sd


def sd_of(path):
    obj = load_checkpoint(path)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    m = obj.get("model", obj) if isinstance(obj, dict) else obj
    return m.state_dict()


def main():
    ref, new = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    rel = lambda root, pat: {p.relative_to(root) for p in root.rglob(pat)}
    fails, n = [], 0

    for r in sorted(rel(ref, "*.npz") | rel(ref, "*.npy")):
        if not (new / r).exists():
            continue
        n += 1
        if r.suffix == ".npz":
            a, b = np.load(ref / r), np.load(new / r)
            ok = set(a.files) <= set(b.files) and all(
                np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)
        else:
            ok = np.array_equal(np.load(ref / r), np.load(new / r), equal_nan=True)
        print(f"  [{'PASS' if ok else 'FAIL'}] {r}")
        if not ok:
            fails.append(str(r))

    for r in sorted(rel(ref, "model_*.pt")):
        if not (new / r).exists():
            continue
        n += 1
        summ = json.loads((new / r.parent / "run_summary.json").read_text())
        smap = summ.get("session_num_neurons")
        a, b = sd_of(ref / r), sd_of(new / r)
        # remap whichever side is count-keyed (v1); a v2 side has no count keys and
        # is returned unchanged, so v1-vs-v1 and v1-vs-v2 are handled alike
        try:
            ra = maybe_remap(a, smap)
            rb = maybe_remap(b, smap)
        except Exception as e:
            print(f"  [FAIL] {r}: remap refused: {e}"); fails.append(str(r)); continue
        renamed = set(a) ^ set(ra)
        part1 = set(ra) == set(rb) and all(torch.equal(ra[k], rb[k]) for k in ra)
        part2 = all(any(p in k for p in PROJ) for k in renamed)
        part3 = all(torch.equal(a[k], b[k]) for k in a if not any(p in k for p in PROJ))
        ok = part1 and part2 and part3
        print(f"  [{'PASS' if ok else 'FAIL'}] {r}: {len(b)} tensors, {len(renamed) // 2} renamed; "
              f"values exact {part1}, renames confined {part2}, rest bit-identical {part3}")
        if not ok:
            fails.append(str(r))

    for r in sorted(rel(ref, "run_summary.json")):
        if not (new / r).exists():
            continue
        a = json.loads((ref / r).read_text()); b = json.loads((new / r).read_text())
        lost = sorted(set(a) - set(b))
        changed = sorted(k for k in a if k in b and a[k] != b[k])
        added = sorted(set(b) - set(a))
        bad = lost + [k for k in changed if k not in EXPECTED_SUMMARY_CHANGES]
        ok = not bad
        print(f"  [{'PASS' if ok else 'FAIL'}] {r}: changed {changed}; added {added}"
              + (f"; UNEXPECTED {bad}" if bad else ""))
        if not ok:
            fails.append(str(r))

    print("-" * 64)
    print(f"{n} numerical artifacts compared; "
          + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
    return 1 if fails or n == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
