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


# ---------------------------------------------------------------- M1
PROCESSED = "/work/nvme/beml/ac136/IBL_MTM/data/processed_mtm"
SMALL = ["f140a2ec-fd49-4814-994a-fe3476f14e66_aligned",   # 308 clusters
         "57fd2325-67f4-4d45-9907-29e77d3043d7_aligned"]   # 318 clusters


def m1():
    import argparse, hashlib, io, contextlib, tempfile
    import torch
    from ndt.utils.dataset_utils import load_ibl_dataset_locally
    from ndt.cli import train_multi as tm
    from ndt.trainer.base import RUN_PROVENANCE_KEYS

    quiet = lambda: contextlib.redirect_stdout(io.StringIO())

    # a. a failed session raises instead of being skipped
    try:
        with quiet():
            load_ibl_dataset_locally(eid=None, num_sessions=2,
                                     train_session_eid=[SMALL[0], "00000000-0000-0000-0000-000000000000"],
                                     data_type="processed_mtm", base_path="/nonexistent",
                                     data_root=PROCESSED)
        check("M1a. a session that fails to load raises", False, "no exception")
    except RuntimeError as e:
        check("M1a. a session that fails to load raises", "00000000" in str(e), str(e)[:80])

    # b. an explicit list longer than num_sessions is refused, not truncated
    try:
        with quiet():
            load_ibl_dataset_locally(eid=None, num_sessions=1, train_session_eid=SMALL,
                                     data_type="processed_mtm", base_path="/nonexistent",
                                     data_root=PROCESSED)
        check("M1b. over-long explicit list refused", False, "no exception")
    except ValueError as e:
        check("M1b. over-long explicit list refused", "refusing to truncate" in str(e))

    # c. ordered per-session map, in load order; num_neurons untouched
    with quiet():
        _, _, _, meta = load_ibl_dataset_locally(eid=None, num_sessions=2,
                                                 train_session_eid=list(reversed(SMALL)),
                                                 data_type="processed_mtm",
                                                 base_path="/nonexistent", data_root=PROCESSED)
    want = [[SMALL[1][:36], 318], [SMALL[0][:36], 308]]
    check("M1c. session_num_neurons is ordered [eid, n] in load order",
          meta.get("session_num_neurons") == want, str(meta.get("session_num_neurons", "<missing>")))
    check("M1d. num_neurons still the set-derived list (unchanged semantics)",
          sorted(meta["num_neurons"]) == [308, 318] and isinstance(meta["num_neurons"], list))

    # e. session resolution refuses anything implicit or inconsistent
    ap = tm.build_parser()
    def refuses(argv):
        try:
            tm.resolve_sessions(ap.parse_args(argv)); return False
        except SystemExit:
            return True
    e0, e1 = SMALL[0][:36], SMALL[1][:36]
    check("M1e. no session source -> refused", refuses([]))
    check("M1f. duplicate eid -> refused", refuses(["--sessions", e0, e0]))
    check("M1g. malformed eid -> refused", refuses(["--sessions", "not-an-eid"]))
    check("M1h. --num-sessions disagreeing with the list -> refused",
          refuses(["--sessions", e0, e1, "--num-sessions", "1"]))
    eids, src, sha = tm.resolve_sessions(ap.parse_args(["--train-session-eid", SMALL[0], e1]))
    check("M1i. --train-session-eid alias works; _aligned stripped; order kept",
          eids == [e0, e1] and src == "cli" and sha is None, str(eids))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"{e1}\n{e0}\n")
    eids, src, sha = tm.resolve_sessions(ap.parse_args(["--manifest", f.name]))
    check("M1j. --manifest: file order, path and sha256 recorded",
          eids == [e1, e0] and src == f.name and
          sha == hashlib.sha256(open(f.name, "rb").read()).hexdigest())
    try:
        ap.parse_args(["--sessions", e0, "--manifest", f.name])
        check("M1k. --sessions and --manifest are mutually exclusive", False)
    except SystemExit:
        check("M1k. --sessions and --manifest are mutually exclusive", True)

    # l. only allowlisted run provenance reaches run_summary
    check("M1l. provenance allowlist is exactly the three run keys, no trainer-owned names",
          set(RUN_PROVENANCE_KEYS) == {"session_num_neurons", "session_list_source",
                                        "session_list_sha256"}
          and "session_vocab" not in RUN_PROVENANCE_KEYS)

    # m. the extra meta_data key is RNG-neutral for model construction
    from ndt.models.ndt1 import NDT1
    from ndt.utils.config_utils import config_from_kwargs, update_config
    from ndt.paths import config_path
    def build(extra):
        cfg = config_from_kwargs({"model": "include:" + config_path("ndt1_stitching")})
        cfg = update_config(config_path("ndt1_stitching"), cfg)
        cfg = update_config(config_path("ssl_session_trainer"), cfg)
        md = {"num_neurons": [308], "num_sessions": 1, "eids": {e0}}
        if extra:
            md["session_num_neurons"] = [[e0, 308]]
        torch.manual_seed(0)
        m = NDT1(cfg.model, **cfg.method.model_kwargs, **md, session_vocab=[e0])
        return m.state_dict(), torch.randn(4)
    a, ra = build(False); b, rb = build(True)
    same = set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a) and torch.equal(ra, rb)
    check("M1m. extra meta_data key: identical state_dict and RNG position", same,
          f"{len(a)} tensors")


if __name__ == "__main__":
    m0()
    m1()
    print("=" * 72)
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    print("=" * 72)
    n = sum(ok for ok, _, _ in results)
    print(f"{n}/{len(results)} passed")
    sys.exit(0 if n == len(results) else 1)
