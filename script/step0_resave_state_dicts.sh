#!/bin/bash
# Step 0 E: write a portable state_dict beside every checkpoint in a capture.
#
#   ./step0_resave_state_dicts.sh /path/to/capture [...]
#
# Checkpoints are pickled nn.Modules, so each one embeds the import path of its
# class. The Tier 1b fix re-keys the stitcher and changes that class, which can
# leave these checkpoints unloadable by the very code they exist to be compared
# against. resave_as_state_dict drops the module pickle and keeps only tensors
# plus the epoch, which survives arbitrary edits to the model class.
#
# Additive: the original model_*.pt files are left untouched. Outputs are named
# model_<name>.state_dict.pt next to them.

set -u
cd "$(dirname "$0")/.." || exit 2

[ $# -ge 1 ] || { echo "usage: $0 <capture-dir> [...]" >&2; exit 2; }

python - "$@" <<'PY'
import sys, glob, os
from ndt.compat import resave_as_state_dict

total = 0
for root in sys.argv[1:]:
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")
    ckpts = sorted(glob.glob(os.path.join(root, "**", "model_*.pt"), recursive=True))
    # Skip anything already converted, so the script is safe to re-run.
    ckpts = [c for c in ckpts if not c.endswith(".state_dict.pt")]
    if not ckpts:
        print(f"no checkpoints under {root}")
        continue
    print(f"=== {root} ({len(ckpts)} checkpoint(s)) ===")
    for src in ckpts:
        dst = src[:-len(".pt")] + ".state_dict.pt"
        try:
            resave_as_state_dict(src, dst)
            n = os.path.getsize(dst)
            print(f"  ok  {os.path.relpath(dst, root)}  ({n/1e6:.1f} MB)")
            total += 1
        except Exception as e:
            print(f"  FAILED {os.path.relpath(src, root)}: {type(e).__name__}: {e}")
            sys.exit(1)
print(f"\nwrote {total} state_dict file(s)")
PY
