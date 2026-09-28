#!/usr/bin/env python
"""1c-ii Test 1: the session-vocabulary load contract.

    python script/test_vocab_contract.py

Construction and load only -- no training, no GPU, no data. Runs in seconds on a
login node.

Why a two-eid fixture rather than folding this into the single-session smoke
test: a one-entry vocabulary cannot be meaningfully reordered. Equal-length
reorderings are exactly the failure that loads *cleanly* and is silently wrong,
attributing every session's learned embedding to a different session. A test
that only ever mismatches on length would pass while leaving that uncovered.
"""

import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ndt.cli.train_single import build_config                     # noqa: E402
from ndt.models.ndt1 import NDT1                                  # noqa: E402
from ndt.utils.config_utils import update_config                  # noqa: E402
from ndt.utils.utils import set_seed                              # noqa: E402
from ndt.compat import (resave_as_state_dict, load_pickled_model,  # noqa: E402
                        load_model_from_state_dict, legacy_session_vocab)
from ndt.vocab import (VocabularyError, vocab_metadata, RULE_MANIFEST,  # noqa: E402
                       RULE_LEGACY, RULE_SINGLE_EID, RULE_SORTED,
                       CHECKPOINT_FORMAT_VERSION, LEGACY_FORMAT_VERSION,
                       vocab_for_sessions, vocab_for_single_session,
                       assert_eid_in_vocab, classify_state_dict_keys,
                       filter_transfer_state_dict, SESSION_SPECIFIC_PREFIXES,
                       SHARED_PREFIXES)

A = "0cad7ea8-8e6c-4ad1-a5c5-53fbb2df1a63"
B = "5dcee0eb-b34d-4652-acc3-d10afc6eae68"
OUTSIDE = "032452e9-1886-449d-9c13-0f192572e19f"

results = []


def check(name, fn, expect_error=None):
    try:
        fn()
    except Exception as e:
        ok = expect_error is not None and isinstance(e, expect_error)
        results.append((ok, name, f"{type(e).__name__}: {str(e)[:110]}"))
        return
    ok = expect_error is None
    results.append((ok, name, "no error" if ok else f"expected {expect_error.__name__}"))


def build(vocab, num_neurons, seed=None):
    """Build an NDT1. `seed` overrides the config seed.

    The transfer fixture below MUST build its two models with different seeds.
    With the same seed the target's freshly-initialised weights are bit-identical
    to the source's, which makes "inherited" and "fresh" indistinguishable and
    both transfer assertions vacuous -- the shared one passes without anything
    having loaded, and the session-specific one fails without anything having
    been inherited.
    """
    cfg = build_config()
    # M3: EID-keyed projections need the [eid, n_neurons] map. Paired one-to-one
    # when the lists match; a legacy 40-entry vocabulary trained its first session.
    smap = ([[e, n] for e, n in zip(vocab, num_neurons)] if len(vocab) == len(num_neurons)
            else [[vocab[0], num_neurons[0]]])
    meta = {"num_neurons": num_neurons, "num_sessions": len(vocab), "eids": set(vocab),
            "session_num_neurons": smap}
    cfg = update_config(cfg, meta)
    set_seed(cfg.seed if seed is None else seed)
    return NDT1(cfg.model, **cfg.method.model_kwargs, **meta,
                session_vocab=vocab), cfg.model, meta


def main():
    tmp = Path(tempfile.mkdtemp(prefix="vocabtest_"))
    print(f"scratch: {tmp}\n")

    # ---------------------------------------------------------- vocab builders
    v, r = vocab_for_single_session(A + "_aligned")
    check("single-session vocab strips _aligned and is single_eid",
          lambda: (_ for _ in ()).throw(AssertionError("x")) if (v != [A] or r != RULE_SINGLE_EID) else None)

    v2, r2 = vocab_for_sessions({B, A}, ordered_eids=[A, B])
    check("manifest order wins over set order",
          lambda: None if (v2 == [A, B] and r2 == RULE_MANIFEST) else
          (_ for _ in ()).throw(AssertionError(f"got {v2} {r2}")))

    v3, r3 = vocab_for_sessions({B, A})
    check("no manifest falls back to sorted, and says so",
          lambda: None if (v3 == sorted([A, B]) and r3 == RULE_SORTED) else
          (_ for _ in ()).throw(AssertionError(f"got {v3} {r3}")))

    check("a loaded session missing from the manifest is refused",
          lambda: vocab_for_sessions({A, B}, ordered_eids=[A]),
          expect_error=VocabularyError)

    # ------------------------------------------------- build + save a 2-eid model
    model, model_cfg, meta = build([A, B], [280, 300])
    assert list(model.encoder.embedder.eid_lookup) == [A, B]
    assert model.encoder.embedder.embed_session.weight.shape[0] == 2

    pkl = tmp / "model_best.pt"
    torch.save({"model": model, "epoch": 0,
                **vocab_metadata([A, B], RULE_MANIFEST)}, pkl)
    sdp = tmp / "model_best.state_dict.pt"
    resave_as_state_dict(pkl, sdp)

    saved = torch.load(sdp, map_location="cpu", weights_only=False)
    check("resave carries session_vocab into .state_dict.pt",
          lambda: None if saved.get("session_vocab") == [A, B] else
          (_ for _ in ()).throw(AssertionError(f"got {saved.get('session_vocab')}")))
    check("resave carries the ordering rule and format version",
          lambda: None if (saved.get("vocabulary_ordering_rule") == RULE_MANIFEST
                           and saved.get("checkpoint_format_version") == CHECKPOINT_FORMAT_VERSION)
          else (_ for _ in ()).throw(AssertionError(f"got {saved}")))

    # ------------------------------------------------------- the load contract
    check("pickle, no caller vocab -> checkpoint is authoritative",
          lambda: load_pickled_model(pkl))
    check("pickle, matching caller vocab -> proceeds",
          lambda: load_pickled_model(pkl, session_vocab=[A, B]))
    check("pickle, REORDERED caller vocab -> fails",
          lambda: load_pickled_model(pkl, session_vocab=[B, A]),
          expect_error=VocabularyError)
    check("pickle, different-length caller vocab -> fails",
          lambda: load_pickled_model(pkl, session_vocab=[A]),
          expect_error=VocabularyError)

    check("state_dict, no caller vocab -> builds from checkpoint vocab",
          lambda: load_model_from_state_dict(sdp, model_cfg, num_neurons=[280, 300],
                                             num_sessions=2, eids={A, B},
                                             **build_config().method.model_kwargs))
    check("state_dict, matching caller vocab -> proceeds",
          lambda: load_model_from_state_dict(sdp, model_cfg, session_vocab=[A, B],
                                             num_neurons=[280, 300], num_sessions=2,
                                             eids={A, B},
                                             **build_config().method.model_kwargs),)
    check("state_dict, REORDERED caller vocab -> fails",
          lambda: load_model_from_state_dict(sdp, model_cfg, session_vocab=[B, A],
                                             num_neurons=[280, 300], num_sessions=2,
                                             eids={A, B},
                                             **build_config().method.model_kwargs),
          expect_error=VocabularyError)

    check("an eid absent from the vocabulary is refused, not extended",
          lambda: assert_eid_in_vocab([A, B], OUTSIDE),
          expect_error=VocabularyError)

    # ------------------------------------------------------- legacy inference
    legacy_pkl = tmp / "legacy_best.pt"
    lmodel, _, _ = build(legacy_session_vocab(), [280])
    torch.save({"model": lmodel, "epoch": 0}, legacy_pkl)   # no vocab metadata
    check("legacy pickle with no metadata infers the shipped 40 and loads",
          lambda: load_pickled_model(legacy_pkl))
    _, lv, lr_, lver = load_pickled_model(legacy_pkl)
    check("legacy inference reports rule=legacy_target_eids and version 0",
          lambda: None if (lr_ == RULE_LEGACY and lver == LEGACY_FORMAT_VERSION
                           and len(lv) == 40) else
          (_ for _ in ()).throw(AssertionError(f"got {lr_} v{lver} n={len(lv)}")))

    legacy_sd = tmp / "legacy_best.state_dict.pt"
    resave_as_state_dict(legacy_pkl, legacy_sd)
    ls = torch.load(legacy_sd, map_location="cpu", weights_only=False)
    check("legacy resave records the inferred vocabulary as v0, not as v1",
          lambda: None if (len(ls.get("session_vocab", [])) == 40
                           and ls.get("checkpoint_format_version") == LEGACY_FORMAT_VERSION)
          else (_ for _ in ()).throw(AssertionError(f"got v{ls.get('checkpoint_format_version')}")))

    # --------------------------------------------- pickle metadata/weight drift
    drifted = tmp / "drifted.pt"
    torch.save({"model": model, "epoch": 0,
                **vocab_metadata([A, B, OUTSIDE], RULE_MANIFEST)}, drifted)
    check("pickle whose metadata disagrees with its live module -> fails",
          lambda: load_pickled_model(drifted), expect_error=VocabularyError)

    # ================================================================
    # Transfer contract: the equal-cluster-count case
    # ================================================================
    # A natural same-count pair does not have to exist in the data -- the fixture
    # constructs one. Relying on finding two real sessions with identical cluster
    # counts would make this test's coverage an accident of the pool.
    #
    # The contract is stated as what the LOAD did, not as what the tensors
    # resemble:
    #
    #   shared           post_load == source            (inherited)
    #   session-specific post_load == H's pre-load state (untouched)
    #
    # An earlier version asserted "session-specific differs from source", which is
    # only a proxy: it fails for any constant-initialised tensor, since a zero bias
    # in H matches a zero bias in S whether or not anything loaded.
    N = 300
    src, src_cfg, _ = build([A], [N], seed=42)
    dst, _, _ = build([B], [N], seed=7)   # same cluster count, different session AND seed
    src_sd = {k: v.clone() for k, v in src.state_dict().items()}
    # state_dict() returns live references, so the pre-load snapshot must be cloned
    # or load_state_dict mutates it too and the comparison becomes trivially true.
    pre_sd = {k: v.clone() for k, v in dst.state_dict().items()}

    # M3 (v2): before 1b, same-count S and H had IDENTICAL key sets -- their count-
    # keyed projections shared names, the dangerous case. With EID keys the only
    # difference is exactly those projections; embed_session.weight still shares a
    # name, so the transfer filter is still load-bearing for the embedding.
    diff = set(src_sd) ^ set(pre_sd)
    proj = lambda k: ".stitcher_dict." in k or ".stitch_decoder_dict." in k
    check("v2: S and H differ ONLY in their EID-keyed projections (1b: no name collision)",
          lambda: None if diff and all(proj(k) for k in diff) and len(diff) == 8 else
          (_ for _ in ()).throw(AssertionError(f"unexpected key difference: {sorted(diff)[:6]}")))
    check("v2: embed_session.weight still shares a name, so the filter is still needed",
          lambda: None if "encoder.embedder.embed_session.weight" in set(src_sd) & set(pre_sd) else
          (_ for _ in ()).throw(AssertionError("embed_session name no longer shared")))

    shared, specific = classify_state_dict_keys(src_sd.keys())
    _, specific_dst = classify_state_dict_keys(pre_sd.keys())
    check(f"every key classified: {len(shared)} shared + {len(specific)} session-specific"
          f" == {len(src_sd)}",
          lambda: None if len(shared) + len(specific) == len(src_sd) else
          (_ for _ in ()).throw(AssertionError("partition does not cover the checkpoint")))

    # Non-vacuity: if S and H were identical going in, every assertion below would
    # pass no matter what the load did. Constant-initialised tensors are allowed to
    # match and are not evidence either way, so require at least one NON-constant
    # session-specific tensor to differ.
    spec_differ = [k for k in specific if k in pre_sd and not torch.equal(src_sd[k], pre_sd[k])]
    spec_differ += [k for k in specific if k not in pre_sd]   # EID-keyed: distinct by name
    check(f"non-vacuity: {len(spec_differ)}/{len(specific)} session-specific tensors "
          f"differ between S and H before loading",
          lambda: None if spec_differ else
          (_ for _ in ()).throw(AssertionError(
              "S and H are identical on every session-specific tensor pre-load; "
              "the freshness assertion below would pass vacuously")))

    check("an unrecognised key is refused, not silently transferred",
          lambda: classify_state_dict_keys(list(src_sd) + ["encoder.brand_new_module.weight"]),
          expect_error=VocabularyError)

    filtered = filter_transfer_state_dict(src_sd, verbose=False)
    check("filter keeps exactly the shared set",
          lambda: None if set(filtered) == set(shared) else
          (_ for _ in ()).throw(AssertionError("filtered set != shared set")))

    missing, unexpected = dst.load_state_dict(filtered, strict=False)
    check("no unexpected keys when loading the filtered source",
          lambda: None if not unexpected else
          (_ for _ in ()).throw(AssertionError(f"unexpected: {unexpected}")))
    check("exactly the session-specific keys are left un-loaded",
          lambda: None if set(missing) == set(specific_dst) else
          (_ for _ in ()).throw(AssertionError(f"missing != specific: {set(missing) ^ set(specific_dst)}")))

    post_sd = dst.state_dict()

    bad_shared = [k for k in shared if not torch.equal(post_sd[k], src_sd[k])]
    check(f"SHARED: all {len(shared)} tensors post_load == source exactly",
          lambda: None if not bad_shared else
          (_ for _ in ()).throw(AssertionError(
              f"{len(bad_shared)} shared tensor(s) did not transfer: {sorted(bad_shared)[:5]}")))

    bad_spec = [k for k in specific_dst if not torch.equal(post_sd[k], pre_sd[k])]
    check(f"SESSION-SPECIFIC: all {len(specific_dst)} tensors post_load == H's pre-load "
          f"state exactly (untouched by the load)",
          lambda: None if not bad_spec else
          (_ for _ in ()).throw(AssertionError(
              f"{len(bad_spec)} session-specific tensor(s) were modified by the load: "
              f"{sorted(bad_spec)[:5]}")))

    check("decoder.* is on the SHARED side, not misfiled with stitch_decoder.*",
          lambda: None if any(k.startswith("decoder.") for k in shared)
          and not any(k.startswith("decoder.") for k in specific) else
          (_ for _ in ()).throw(AssertionError("decoder.* misclassified")))

    # ------------------------------------------------------------------ report
    print("=" * 72)
    npass = sum(1 for ok, _, _ in results if ok)
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            print(f"         {detail}")
    print("=" * 72)
    print(f"{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
