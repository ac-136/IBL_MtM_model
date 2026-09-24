#!/usr/bin/env python
"""Offline gates for the multi-session track. No GPU, no data beyond parquet metadata.

    python script/test_multisession.py

Grows stage by stage (M0, M1, M2, ...). Each check pins a property the plan's
gate depends on, against production code, not a restatement of it.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

results = []


def check(name, ok, detail=""):
    results.append((bool(ok), name, detail))


def git_blob(rev, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:{path}"],
                          capture_output=True, check=True).stdout


# ---------------------------------------------------------------- M0
def m0():
    from ndt.cli.train_multi import build_config
    cfg = build_config()
    # The MtM original multi-session recipe, 652c61c src/configs/ssl_sessions_trainer.yaml.
    want = {
        ("training", "num_epochs"): 1000,
        ("training", "train_batch_size"): 16,
        ("training", "test_batch_size"): 16,
        ("training", "save_every"): 50,
        ("optimizer", "lr"): 1e-4,
        ("optimizer", "wd"): 0.01,
        ("optimizer", "eps"): 1e-8,
        ("optimizer", "warmup_pct"): 0.15,
        ("optimizer", "gamma"): 0.95,
        ("optimizer", "div_factor"): 10,
        ("optimizer", "scheduler"): "cosine",
    }
    def get(a, b):
        try:
            return cfg[a][b]
        except (KeyError, AttributeError):
            return "<missing>"
    bad = {f"{a}.{b}": (get(a, b), v) for (a, b), v in want.items() if get(a, b) != v}
    check("M0a. train_multi.build_config() resolves to the 652c61c recipe",
          not bad, f"{len(want)} fields" if not bad else f"mismatch (got, want): {bad}")
    # The single-session recipe must be untouched by M0.
    same = (REPO / "ndt/configs/ssl_session_trainer.yaml").read_bytes() == \
        git_blob("recipe_mtm_frozen", "ndt/configs/ssl_session_trainer.yaml")
    check("M0b. ssl_session_trainer.yaml byte-identical to recipe_mtm_frozen", same)


if __name__ == "__main__":
    m0()
    print("=" * 72)
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    print("=" * 72)
    n = sum(ok for ok, _, _ in results)
    print(f"{n}/{len(results)} passed")
    sys.exit(0 if n == len(results) else 1)
