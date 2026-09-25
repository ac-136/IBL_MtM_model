#!/usr/bin/env python
"""Assert the 89/5 transfer contract for one source -> target pair.

    python script/check_transfer_contract.py SRC_CKPT TGT_EID DATA_ROOT

Factored out of ``step1_transfer_smoke.sh`` so every Stage-3 job can make the
same assertions its own source and target, rather than relying on the one pair
Stage 1 happened to check. It reconstructs exactly what ``finetune_eval`` does --
builds the target model with vocabulary [H], then loads the filtered source --
using the same production functions, so what is asserted is the real logic
rather than a paraphrase of it.

Runs BEFORE any optimizer step, because "session-specific tensors untouched" is
only checkable against a snapshot of H taken before the load; once training
starts that state is gone.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from ndt.utils.config_utils import config_from_kwargs, update_config  # noqa: E402
from ndt.utils.dataset_utils import load_ibl_dataset_locally  # noqa: E402
from ndt.utils.utils import set_seed  # noqa: E402
from ndt.models.ndt1 import NDT1  # noqa: E402
from ndt.paths import config_path  # noqa: E402
from ndt.vocab import (vocab_for_single_session, filter_transfer_state_dict,  # noqa: E402
                       classify_state_dict_keys)

BASE_PATH = "/work/nvme/beml/ac136/IBL_MTM/data"


def main(argv):
    src_path, tgt_eid, data_root = argv[1:4]
    results = []

    def check(name, ok, detail=""):
        results.append((bool(ok), name, detail))

    src_ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    # state_dict() hands back live references; clone or load_state_dict mutates
    # the snapshot too and every "untouched" assertion becomes trivially true.
    src_sd = {k: v.clone() for k, v in src_ckpt["model"].state_dict().items()}
    src_vocab = src_ckpt.get("session_vocab")
    print(f"source     : {src_path}")
    print(f"source vocab: {src_vocab}")
    print(f"target     : {tgt_eid}")

    cfg = config_from_kwargs({"model": "include:" + config_path("ndt1_stitching")})
    cfg = update_config(config_path("finetune_sessions_trainer"), cfg)
    set_seed(cfg.seed)
    _, _, _, meta = load_ibl_dataset_locally(
        eid=None, num_sessions=cfg.data.num_sessions, train_session_eid=[tgt_eid],
        test_session_eid=cfg.data.test_session_eid, data_type="processed_mtm",
        base_path=BASE_PATH, data_root=data_root)
    cfg = update_config(cfg, meta)
    tgt_vocab, _rule = vocab_for_single_session(tgt_eid)
    model = NDT1(cfg.model, **cfg.method.model_kwargs, **meta, session_vocab=tgt_vocab)

    pre = {k: v.clone() for k, v in model.state_dict().items()}
    shared, specific = classify_state_dict_keys(pre.keys())
    bare = tgt_eid.removesuffix("_aligned")

    # M3 (M-g): a source may be multi-session. The target must be in none of its
    # sessions, and the source's own classification is reported: 89 shared plus
    # 1 + 4N source-specific (embedding + a stitcher and decoder weight/bias per
    # session), all of which the filter excludes.
    src_eids = [str(e).removesuffix("_aligned") for e in (src_vocab or [])]
    check("1. target is not one of the source's sessions",
          bare not in src_eids, f"target {bare[:8]} vs source {[e[:8] for e in src_eids]}")
    src_shared, src_specific = classify_state_dict_keys(src_sd.keys())
    n_src = len(src_eids) or 1
    check(f"1b. source classification: {len(src_shared)} shared + {len(src_specific)} "
          f"source-specific (1 + 4x{n_src})",
          len(src_shared) == len(shared) and len(src_specific) == 1 + 4 * n_src,
          f"shared {len(src_shared)} (target {len(shared)}), specific {len(src_specific)}")
    check("2. target vocabulary is [H] and resolves to row 0",
          list(model.encoder.embedder.eid_lookup) == [bare]
          and model.encoder.embedder.eid_to_indx[bare] == 0,
          f"vocab={model.encoder.embedder.eid_lookup}")
    check("5. the two prefix sets exhaustively partition the checkpoint",
          len(shared) + len(specific) == len(pre),
          f"{len(shared)} shared + {len(specific)} specific == {len(pre)}")

    # Where S and H differ in cluster count the session-specific tensors differ
    # in SHAPE, so no key can collide; comparable-and-equal is what to rule out.
    comparable = [k for k in specific if k in src_sd and src_sd[k].shape == pre[k].shape]
    nonconst = [k for k in comparable if not torch.equal(src_sd[k], pre[k])]
    check("6. non-vacuity: S and H are distinguishable on the session-specific side",
          len(comparable) == 0 or len(nonconst) > 0,
          f"{len(comparable)} shape-comparable, {len(nonconst)} of them differ pre-load"
          + ("  (differing cluster counts -> no key overlap, so the load cannot "
             "confuse them by name)" if not comparable else ""))

    filtered = filter_transfer_state_dict(src_sd)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    check("7a. no unexpected keys from the filtered source",
          not unexpected, str(sorted(unexpected)[:4]))
    check("7b. exactly the session-specific keys left un-loaded",
          set(missing) == set(specific), f"missing={len(missing)} specific={len(specific)}")

    post = model.state_dict()
    bad_shared = [k for k in shared if not torch.equal(post[k], src_sd[k])]
    check(f"4. SHARED: all {len(shared)} tensors post_load == source exactly",
          not bad_shared, f"{len(bad_shared)} failed: {sorted(bad_shared)[:4]}")

    bad_spec = [k for k in specific if not torch.equal(post[k], pre[k])]
    check(f"3. SESSION-SPECIFIC: all {len(specific)} tensors post_load == H's "
          f"pre-load state exactly", not bad_spec,
          f"{len(bad_spec)} modified: {sorted(bad_spec)[:4]}")

    print("=" * 70)
    npass = sum(1 for ok, _, _ in results if ok)
    for ok, name, detail in sorted(results, key=lambda r: r[1]):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
    print("=" * 70)
    print(f"{npass}/{len(results)} contract checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
