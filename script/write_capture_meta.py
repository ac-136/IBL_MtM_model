#!/usr/bin/env python
"""Write capture_meta.json: the environment and provenance of one run.

    python script/write_capture_meta.py OUT_DIR LABEL EID EPOCHS [--cmd "..."] [--config PATH]

Factored out of step0_capture.sh so every run gets the same record, not just the
reference captures. The five Stage-2 pretrains predate this and have none, which
is the gap it closes.

Records the command line as well as the environment: knowing a run used torch
2.5 on a GH200 does not tell you which flags produced it, and the flags are
where --data-root, --num-epochs and --save-plot-every-n-epochs live.
"""

import argparse
import json
import os
import platform
import subprocess
import sys


def sh(*c):
    try:
        return subprocess.check_output(c, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("label")
    ap.add_argument("eid")
    ap.add_argument("epochs", type=int)
    ap.add_argument("--cmd", default=None, help="the command line this run was launched with")
    ap.add_argument("--config", default=None, help="path to the resolved config, if known")
    args = ap.parse_args()

    meta = {
        "label": args.label,
        "eid": args.eid,
        "num_epochs": args.epochs,
        "command": args.cmd,
        "resolved_config_path": args.config,
        "git_sha": sh("git", "rev-parse", "HEAD"),
        "git_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_describe": sh("git", "describe", "--tags", "--always"),
        "git_dirty": bool(sh("git", "status", "--porcelain")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    try:
        import torch
        meta.update({
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        })
    except Exception as e:
        meta["torch_error"] = f"{type(e).__name__}: {e}"
    try:
        import numpy
        meta["numpy"] = numpy.__version__
    except Exception:
        pass

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "capture_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"[meta] wrote {path}")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
