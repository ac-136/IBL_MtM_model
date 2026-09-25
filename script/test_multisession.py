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


# ---------------------------------------------------------------- M2
def _synthetic_eids():
    # three sessions, non-divisible sizes at B=16; A and B stand in for same-count
    return ["A"] * 37 + ["B"] * 21 + ["C"] * 16


def m2():
    import numpy as np
    import torch
    from ndt.loader.base import SessionBatchSampler
    B = 16
    eids = _synthetic_eids()
    sizes = {"A": 37, "B": 21, "C": 16}

    def epochs(sampler, n):
        return [list(map(list, iter(sampler))) for _ in range(n)]

    s = SessionBatchSampler(eids, batch_size=B, shuffle=True, seed=42)
    e1, e2, e3 = epochs(s, 3)

    # 1. invariants
    flat = sorted(i for b in e1 for i in b)
    check("M2a. every index exactly once per epoch", flat == list(range(len(eids))))
    check("M2b. every batch single-EID", all(len({eids[i] for i in b}) == 1 for b in e1))
    short = {}
    for b in e1:
        if len(b) < B:
            short.setdefault(eids[b[0]], 0); short[eids[b[0]]] += 1
    want_short = {k: 1 for k, n in sizes.items() if n % B}
    check("M2c. only each session's final batch is short (one per non-divisible session)",
          short == want_short, f"short batches {short}")
    n_want = sum(-(-n // B) for n in sizes.values())
    check("M2d. #batches == sum ceil(n_i/B) == len(sampler)",
          len(e1) == n_want == len(s), f"{len(e1)} / {n_want} / {len(s)}")

    # 2. across epochs (user specification)
    member = lambda ep: [tuple(sorted(b)) for b in ep]
    check("M2e. epoch 1 and epoch 2 differ in batch membership and order",
          set(member(e1)) != set(member(e2)) and e1 != e2)
    s2 = SessionBatchSampler(eids, batch_size=B, shuffle=True, seed=42)
    check("M2f. same seed reproduces the epoch-by-epoch sequence (epochs 1-3)",
          epochs(s2, 3) == [e1, e2, e3])
    s3 = SessionBatchSampler(eids, batch_size=B, shuffle=True, seed=43)
    check("M2g. a different seed gives a different sequence", epochs(s3, 1)[0] != e1)

    # 3. shuffle=False: dataset order, session order, identical every epoch
    v = SessionBatchSampler(eids, batch_size=B, shuffle=False, seed=42)
    v1, v2 = epochs(v, 2)
    check("M2h. shuffle=False: dataset order within sessions, sessions in order, every epoch",
          [i for b in v1 for i in b] == list(range(len(eids))) and v1 == v2
          and v.session_order == ["A", "B", "C"])

    # 4. global RNG untouched
    np.random.seed(123); torch.manual_seed(123)
    np_state, t_state = np.random.get_state()[1].copy(), torch.get_rng_state().clone()
    epochs(SessionBatchSampler(eids, batch_size=B, shuffle=True, seed=42), 3)
    check("M2i. global NumPy and torch RNG state unchanged by iterating 3 epochs",
          np.array_equal(np.random.get_state()[1], np_state)
          and torch.equal(torch.get_rng_state(), t_state))

    # 5. make_loader routing, on real parquet data
    import io, contextlib
    from datasets import load_dataset, concatenate_datasets
    from ndt.loader.make_loader import make_loader
    from ndt.loader.base import LengthStitchGroupedSampler
    def ds(e):
        d = f"{PROCESSED}/{e}/data"
        return load_dataset("parquet", data_files={"val": f"{d}/val.parquet"})["val"]
    one = ds(SMALL[0]); two = concatenate_datasets([ds(SMALL[0]), ds(SMALL[1])])
    kw = dict(target=None, load_meta=False, batch_size=B, pad_to_right=True, pad_value=-1.,
              max_time_length=100, max_space_length=668, dataset_name="ibl",
              sort_by_depth=False, sort_by_region=False, stitching=True, shuffle=True)
    with contextlib.redirect_stdout(io.StringIO()):
        dl1 = make_loader(one, seed=42, **kw)
        dl2 = make_loader(two, seed=42, **kw)
    check("M2j. one-EID dataset keeps the old sampler",
          isinstance(dl1.sampler, LengthStitchGroupedSampler) and dl1.batch_size == B)
    check("M2k. multi-EID dataset gets SessionBatchSampler",
          isinstance(dl2.batch_sampler, SessionBatchSampler))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            make_loader(two, **kw)
        check("M2l. multi-EID without a seed raises", False)
    except ValueError:
        check("M2l. multi-EID without a seed raises", True)
    with contextlib.redirect_stdout(io.StringIO()):
        got = [b["eid"] for b in dl2]
    check("M2m. real multi-session loader: every collated batch single-EID, no crash",
          all(len(set(e)) == 1 for e in got), f"{len(got)} batches")

    # 6/7. eval bucketing and the homogeneity assert, on the real Trainer class
    from types import SimpleNamespace
    from ndt.trainer.base import Trainer
    tr = Trainer.__new__(Trainer)
    tr.model = SimpleNamespace(eval=lambda: None,
                               encoder=SimpleNamespace(embedder=SimpleNamespace(eid_lookup=["A", "B"])))
    tr.masking_mode = "temporal"
    tr.metric = "r2"
    tr.session_active_neurons = []
    tr.eval_metrics_per_session = {}
    tr.accelerator = SimpleNamespace(device="cpu")
    tr.config = SimpleNamespace(method=SimpleNamespace(model_kwargs=SimpleNamespace(
        method_name="ssl", loss="poisson_nll", clf=False, reg=False)))
    g = torch.Generator().manual_seed(0)
    def batch(e, n=4, N=6):
        return {"eid": [e] * n, "spikes_data": torch.zeros(n, 100, N)}
    def fake_forward(b, mm):
        n, N = b["spikes_data"].shape[0], b["spikes_data"].shape[2]
        tgt = torch.rand(n, 100, N, generator=g)
        return SimpleNamespace(loss=torch.tensor(1.0), n_examples=n, targets=tgt,
                               preds=torch.log(tgt + 0.1))
    tr._forward_model_outputs = fake_forward
    tr.eval_dataloader = [batch("A"), batch("B"), batch("A")]   # same count (6) for A and B
    import ndt.trainer.base as tb
    orig = tb.metrics_list
    tb.metrics_list = lambda gt, pred, metrics, device: {"r2": float(gt.shape[1])}
    try:
        res = Trainer.eval_epoch(tr)
    finally:
        tb.metrics_list = orig
    check("M2n. same-count sessions -> two eval buckets, two per-session series",
          len(res["eval_gt"]) == 2 and set(tr.eval_metrics_per_session) == {"A", "B"},
          f"buckets {len(res['eval_gt'])}, series {sorted(tr.eval_metrics_per_session)}")
    tr2 = Trainer.__new__(Trainer)
    tr2.accelerator = SimpleNamespace(device="cpu")
    try:
        Trainer._forward_model_outputs(tr2, {"eid": ["A", "B"]}, "temporal")
        check("M2o. a mixed-EID batch raises in the forward", False)
    except RuntimeError as e:
        check("M2o. a mixed-EID batch raises in the forward", "mixed-session" in str(e))


# ---------------------------------------------------------------- M3
SETA = [["862ade13-53cd-4221-a3fa-dda8643641f2", 258], ["5437ec1f-6535-470c-80b8-4c8806ee085d", 549],
        ["d9f0c293-df4c-410a-846d-842e47c6b502", 556]]
SETA_REF = "/work/nvme/beml/ac136/IBL_MTM/results/multisession/ms_setA_a/ms/train"


def _build(smap, seed=0, multi=True):
    import torch
    from ndt.models.ndt1 import NDT1
    from ndt.utils.config_utils import config_from_kwargs, update_config
    from ndt.paths import config_path
    cfg = config_from_kwargs({"model": "include:" + config_path("ndt1_stitching")})
    cfg = update_config(config_path("ndt1_stitching"), cfg)
    cfg = update_config(config_path("ssl_sessions_trainer" if multi else "ssl_session_trainer"), cfg)
    uniq = []
    for _, n in smap:
        if n not in uniq:
            uniq.append(n)
    meta = {"num_neurons": uniq, "num_sessions": len(smap), "eids": {e for e, _ in smap},
            "session_num_neurons": smap}
    torch.manual_seed(seed)
    return NDT1(cfg.model, **cfg.method.model_kwargs, **meta, session_vocab=[e for e, _ in smap]), cfg


def m3():
    import io, contextlib, json, re, subprocess, tempfile
    import torch
    from ndt.vocab import (CHECKPOINT_FORMAT_VERSION, SESSION_MAP_KEY, vocab_metadata,
                           model_session_num_neurons, assert_pickle_consistent, VocabularyError,
                           classify_state_dict_keys, filter_transfer_state_dict)
    from ndt.compat import (remap_v1_stitch_keys, resave_as_state_dict, load_model_from_state_dict,
                            load_pickled_model)
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())
    A, B = "aaaaaaaa-0000-0000-0000-000000000000", "bbbbbbbb-0000-0000-0000-000000000000"

    # a. same-count fixture: distinct EID-keyed projections, gradient isolation
    m, _ = _build([[A, 6], [B, 6]])
    st, dec = m.encoder.stitcher, m.stitch_decoder
    check("M3a. same-count sessions get distinct EID-keyed stitchers and decoders",
          list(st.stitcher_dict) == [A, B] and list(dec.stitch_decoder_dict) == [A, B]
          and st.stitcher_dict[A] is not st.stitcher_dict[B] and st.key_by == "eid")
    m.zero_grad()
    x = torch.randn(3, 100, 6)
    # stitcher and decoder sit on either side of the transformer (668 channels in,
    # 512 hidden out), so each gets its own input
    st(x, eid=A).sum().backward()
    dec(torch.randn(3, 100, dec.stitch_decoder_dict[A].in_features), eid=A).sum().backward()
    g = lambda mod: [p.grad for p in mod.parameters()]
    a_grads = all(gr is not None and gr.abs().sum() > 0 for gr in g(st.stitcher_dict[A]) + g(dec.stitch_decoder_dict[A]))
    b_clean = all(gr is None or gr.abs().sum() == 0 for gr in g(st.stitcher_dict[B]) + g(dec.stitch_decoder_dict[B]))
    check("M3b. a batch of session A trains A's projections and never touches B's", a_grads and b_clean)
    with torch.no_grad():
        y0 = st(x, eid=A).clone(); st.stitcher_dict[B].weight.add_(1.0); y1 = st(x, eid=A)
    check("M3c. routing: A's output is independent of B's weights", torch.equal(y0, y1))

    # b. v1 pickles still route by count
    with quiet():
        v1, vocab, _, ver = load_pickled_model(f"{SETA_REF}/model_best.pt")
    s1 = v1.encoder.stitcher
    xx = torch.randn(2, 100, 549)
    check("M3d. v1 pickle (no key_by) routes by count under the new code",
          getattr(s1, "key_by", "count") == "count" and ver == 1 and
          torch.equal(s1(xx, eid=SETA[1][0], num_neuron=549), s1.stitcher_dict["549"](xx)))

    # c. remap rules
    v1sd = torch.load(f"{SETA_REF}/model_best.state_dict.pt", map_location="cpu", weights_only=False)["state_dict"]
    r = remap_v1_stitch_keys(v1sd, SETA)
    changed = set(v1sd) ^ set(r)
    proj = lambda k: ".stitcher_dict." in k or ".stitch_decoder_dict." in k
    check("M3e. injective remap: only projection keys renamed, every value identical",
          all(proj(k) for k in changed) and len(changed) == 24 and
          all(torch.equal(v1sd[k], r[k]) for k in v1sd if k in r))
    ren = {re.sub(r"(stitcher_dict|stitch_decoder_dict)\.(\d+)\.",
                  lambda g: f"{g.group(1)}.{dict((str(n), e) for e, n in SETA)[g.group(2)]}.", k): v
           for k, v in v1sd.items()}
    check("M3f. remapped values equal the v1 values under the expected names",
          set(ren) == set(r) and all(torch.equal(ren[k], r[k]) for k in r))
    def refuses(smap):
        try:
            remap_v1_stitch_keys(v1sd, smap); return False
        except VocabularyError:
            return True
    check("M3g. two sessions sharing a count -> refused as ambiguous",
          refuses([[SETA[0][0], 258], [SETA[1][0], 549], ["x" * 8 + SETA[2][0][8:], 549]]))
    check("M3h. a count key with no session in the map -> refused", refuses(SETA[:2]))
    check("M3i. a mapped session with no projection -> refused", refuses(SETA + [[A, 999]]))

    # d. v2 metadata, resave, and state_dict loading
    check("M3j. CHECKPOINT_FORMAT_VERSION == 2", CHECKPOINT_FORMAT_VERSION == 2)
    m2_, cfg = _build(SETA, seed=1)
    tmp = Path(tempfile.mkdtemp())
    pkl = tmp / "model_best.pt"
    torch.save({"model": m2_, "epoch": 0, **vocab_metadata([e for e, _ in SETA], "manifest",
                session_num_neurons=model_session_num_neurons(m2_))}, pkl)
    with quiet():
        resave_as_state_dict(pkl, tmp / "model_best.state_dict.pt")
    saved = torch.load(tmp / "model_best.state_dict.pt", weights_only=False)
    check("M3k. v2 checkpoint and state_dict carry the ordered session map",
          saved.get(SESSION_MAP_KEY) == SETA and saved.get("checkpoint_format_version") == 2)
    kw = dict(num_neurons=[258, 549, 556], num_sessions=3, eids={e for e, _ in SETA},
              **cfg.method.model_kwargs)
    with quiet():
        rb, *_ = load_model_from_state_dict(tmp / "model_best.state_dict.pt", cfg.model, **kw)
    check("M3l. v2 state_dict rebuilds from its metadata alone, exactly",
          all(torch.equal(rb.state_dict()[k], m2_.state_dict()[k]) for k in m2_.state_dict()))
    with quiet():
        rv1, *_ = load_model_from_state_dict(f"{SETA_REF}/model_best.state_dict.pt", cfg.model,
                                             session_num_neurons=SETA, **kw)
    check("M3m. v1 multi-session state_dict + supplied map -> loads into a v2 model exactly",
          all(torch.equal(rv1.state_dict()[k], r[k]) for k in r))
    try:
        with quiet():
            load_model_from_state_dict(f"{SETA_REF}/model_best.state_dict.pt", cfg.model, **kw)
        check("M3n. v1 multi-session state_dict WITHOUT a map -> refused", False)
    except VocabularyError:
        check("M3n. v1 multi-session state_dict WITHOUT a map -> refused", True)
    bad = {"model": m2_, SESSION_MAP_KEY: [[e, n + 1] for e, n in SETA],
           **vocab_metadata([e for e, _ in SETA], "manifest")}
    try:
        assert_pickle_consistent(bad, m2_)
        check("M3o. a pickle whose recorded map disagrees with its stitcher -> refused", False)
    except VocabularyError:
        check("M3o. a pickle whose recorded map disagrees with its stitcher -> refused", True)

    # e. multi-source transfer accounting (M4 V3's contract, offline)
    src = m2_.state_dict()
    tgt, _ = _build([[A, 300]], seed=7, multi=False)
    pre = {k: v.clone() for k, v in tgt.state_dict().items()}
    s_sh, s_sp = classify_state_dict_keys(src.keys())
    t_sh, t_sp = classify_state_dict_keys(pre.keys())
    with quiet():
        missing, unexpected = tgt.load_state_dict(filter_transfer_state_dict(src, verbose=False), strict=False)
    post = tgt.state_dict()
    check("M3p. 3-session source -> 1-session target: 89 shared + 13 source-specific; "
          "89 inherited exactly; target's 5 specific untouched",
          len(s_sh) == 89 and len(s_sp) == 13 and len(t_sp) == 5 and not unexpected
          and set(missing) == set(t_sp)
          and all(torch.equal(post[k], src[k]) for k in s_sh)
          and all(torch.equal(post[k], pre[k]) for k in t_sp))

    # f. RNG equivalence against the M2 (count-keyed) construction
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(f"git -C {REPO} archive ms_reference_pre1b ndt | tar -x -C {d}", shell=True, check=True)
        # Importing test_multisession puts THIS repo first on sys.path; re-insert the
        # tree under test afterwards, or both builds silently use the new code and
        # the comparison is M3 against itself. Assert which module actually loaded.
        prog = ("import sys,torch; sys.path.insert(0, sys.argv[2]+'/script');"
                "import test_multisession as T; sys.path.insert(0, sys.argv[1]);"
                "m,_=T._build(T.SETA, seed=42); import ndt.models.ndt1 as N;"
                "assert N.__file__.startswith(sys.argv[1]), N.__file__;"
                "torch.save({'sd': m.state_dict(), 'after': torch.randn(8), 'src': N.__file__}, sys.argv[3])")
        # the M2 tree's NDT1 ignores session_num_neurons and keys by count
        subprocess.run([sys.executable, "-c", prog, d, str(REPO), f"{d}/old.pt"], check=True,
                       capture_output=True)
        subprocess.run([sys.executable, "-c", prog, str(REPO), str(REPO), f"{d}/new.pt"], check=True,
                       capture_output=True)
        o, n = torch.load(f"{d}/old.pt"), torch.load(f"{d}/new.pt")
    check("M3q0. the 'old' build really came from the M2 tree (non-vacuity)",
          not o["src"].startswith(str(REPO)) and any(re.search(r"_dict\.\d+\.", k) for k in o["sd"]),
          o["src"])
    ro = remap_v1_stitch_keys(o["sd"], SETA)
    check("M3q. RNG equivalence: M3 set-A construction == M2's after remap, same RNG position",
          set(ro) == set(n["sd"]) and all(torch.equal(ro[k], n["sd"][k]) for k in ro)
          and torch.equal(o["after"], n["after"]))


if __name__ == "__main__":
    m0()
    m1()
    m2()
    m3()
    print("=" * 72)
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    print("=" * 72)
    n = sum(ok for ok, _, _ in results)
    print(f"{n}/{len(results)} passed")
    sys.exit(0 if n == len(results) else 1)
