"""Load checkpoints produced by the original, un-vendored tree.

`Trainer.save_model` pickles the live `nn.Module`, not a `state_dict`:

    torch.save({"model": self.model, "epoch": epoch}, ...)

Pickle records each object's class by import path, so a checkpoint written by
the original repository names `models.ndt1.NDT1` and `models.masker.Masker`.
Those top-level modules do not exist once the code is vendored under `ndt`, so
`torch.load` raises `ModuleNotFoundError: No module named 'models'`.

The classes themselves are unchanged, so registering module aliases is enough.

    from ndt.compat import load_checkpoint
    ckpt = load_checkpoint("/path/to/model_best.pt")
    model = ckpt["model"]

Prefer re-saving as a plain `state_dict` once loaded. A pickled module is
fragile in ways an alias cannot fix: it also binds the checkpoint to the class
definition, so any later edit to `NDT1.__init__` can break old files.
"""

import sys

# Legacy top-level name -> module in this package.
_ALIASES = {
    "models": "ndt.models",
    "models.ndt1": "ndt.models.ndt1",
    "models.masker": "ndt.models.masker",
    "models.model_output": "ndt.models.model_output",
    "utils": "ndt.utils",
    "utils.config_utils": "ndt.utils.config_utils",
    "loader": "ndt.loader",
    "loader.base": "ndt.loader.base",
    "trainer": "ndt.trainer",
    "trainer.base": "ndt.trainer.base",
}


def install_legacy_aliases(force=False):
    """Make legacy module paths resolve into this package.

    Existing entries in sys.modules are left alone unless force=True, so this is
    safe to call inside the original repo where a real `models` package exists.
    Returns the aliases that were installed.
    """
    import importlib
    installed = []
    for legacy, target in _ALIASES.items():
        if legacy in sys.modules and not force:
            continue
        try:
            sys.modules[legacy] = importlib.import_module(target)
            installed.append(legacy)
        except ImportError:
            pass
    return installed


def legacy_session_vocab():
    """The shipped 40-entry session vocabulary, in its original file order.

    `ndt/models/ndt1.py` used to read this at module import and size the session
    embedding from it. The model now takes `session_vocab` explicitly, so this
    is what a caller passes to reproduce the original behaviour exactly, and
    what a legacy checkpoint's row indices are relative to.

    Order is load-bearing: row *i* of a legacy `embed_session.weight` is this
    list's entry *i*, so re-ordering it silently reinterprets every existing
    checkpoint.
    """
    from .paths import data_path
    with open(data_path("target_eids.txt")) as f:
        return [line.rstrip() for line in f]


def load_checkpoint(path, map_location="cpu", **kwargs):
    """torch.load a checkpoint from either tree, old or new."""
    import torch
    install_legacy_aliases()
    kwargs.setdefault("weights_only", False)
    return torch.load(path, map_location=map_location, **kwargs)


def load_state_dict(path, map_location="cpu"):
    """Just the weights, whether the file holds a module or a state_dict."""
    obj = load_checkpoint(path, map_location=map_location)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    return model.state_dict() if hasattr(model, "state_dict") else model


def resave_as_state_dict(src_path, dst_path, map_location="cpu"):
    """Convert a pickled-module checkpoint into a portable state_dict file.

    The result depends on no local module paths and survives edits to the model
    class, so this is the durable fix for the 92 legacy checkpoints.
    """
    import torch
    from .vocab import VOCAB_KEY, RULE_KEY, VERSION_KEY, read_checkpoint_vocab

    obj = load_checkpoint(src_path, map_location=map_location)
    out = {"state_dict": load_state_dict(src_path, map_location=map_location)}
    if isinstance(obj, dict) and "epoch" in obj:
        out["epoch"] = obj["epoch"]

    # 1c-ii: the state_dict form is the durable one -- it is what survives edits
    # to the model class -- so it has to carry the vocabulary too. Without this
    # the conversion silently discards the only record of what the embedding
    # rows mean, and a same-length reordering would then load without error.
    # A legacy source has no recorded vocabulary; read_checkpoint_vocab infers
    # the shipped ordering, which is what such a checkpoint was in fact built
    # with, and says so.
    model = obj.get("model") if isinstance(obj, dict) else None
    embedder = getattr(getattr(model, "encoder", None), "embedder", None)
    if getattr(embedder, "eid_lookup", None) is not None:
        vocab, rule, version = read_checkpoint_vocab(obj, source=str(src_path))
        out[VOCAB_KEY] = list(vocab)
        out[RULE_KEY] = rule
        out[VERSION_KEY] = version
        # M3: a v2 state_dict must carry its [eid, n_neurons] map, or it cannot be
        # rebuilt from metadata alone. Prefer the recorded map; else read it from
        # the module. A v1 source has none (and needs none: count keys).
        from .vocab import SESSION_MAP_KEY, model_session_num_neurons
        smap = obj.get(SESSION_MAP_KEY) if isinstance(obj, dict) else None
        smap = smap if smap is not None else model_session_num_neurons(model)
        if smap is not None:
            out[SESSION_MAP_KEY] = [[str(e), int(n)] for e, n in smap]

    torch.save(out, dst_path)
    return dst_path


_V1_KEY = __import__("re").compile(r"^(.*\.(?:stitcher_dict|stitch_decoder_dict))\.(\d+)\.(.*)$")


def remap_v1_stitch_keys(state_dict, session_num_neurons):
    """(M3) Rename a v1 count-keyed state_dict to v2 EID keys. Values untouched.

    Allowed ONLY when the eid -> n_neurons mapping is injective. If two sessions
    share a count, a v1 checkpoint holds ONE projection that both used, and there
    is no correct way to split it: conversion fails loudly as ambiguous rather
    than guessing. (The released MtM 34-session checkpoint is refused for exactly
    this reason, which is correct.) Also refused: a count key with no session in
    the map, and a session in the map with no projection in the checkpoint.
    """
    from .vocab import VocabularyError, bare
    pairs = [(bare(e), int(n)) for e, n in session_num_neurons]
    by_count = {}
    for e, n in pairs:
        by_count.setdefault(n, []).append(e)
    shared = {n: es for n, es in by_count.items() if len(es) > 1}
    if shared:
        raise VocabularyError(
            f"v1 -> v2 remap is ambiguous: sessions share a neuron count and so share "
            f"one v1 projection that cannot be split between them: {shared}")
    out, seen = {}, set()
    for k, v in state_dict.items():
        m = _V1_KEY.match(k)
        if m is None:
            out[k] = v
            continue
        n = int(m.group(2))
        if n not in by_count:
            raise VocabularyError(f"v1 key {k} has count {n}, which no session in the map has")
        out[f"{m.group(1)}.{by_count[n][0]}.{m.group(3)}"] = v
        seen.add(n)
    missing = sorted(set(by_count) - seen)
    if missing:
        raise VocabularyError(f"sessions with counts {missing} have no projection in the v1 checkpoint")
    return out


def load_pickled_model(path, session_vocab=None, map_location="cpu"):
    """Load a pickled-module checkpoint under the 1c-ii load contract.

    The checkpoint is authoritative: its vocabulary defines what the embedding
    rows mean, because those rows *are* its weights. Passing ``session_vocab``
    asserts what the caller believes that vocabulary to be, and any disagreement
    fails rather than being silently overridden.

    Returns ``(model, vocab, rule, version)``.
    """
    from .vocab import (read_checkpoint_vocab, assert_vocab_matches,
                        assert_pickle_consistent)

    ckpt = load_checkpoint(path, map_location=map_location)
    model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    embedder = getattr(getattr(model, "encoder", None), "embedder", None)
    if getattr(embedder, "eid_lookup", None) is None:
        return model, None, None, None

    # A pickle carries a live module *and* recorded metadata, so the two can
    # drift apart; if they do, the recorded provenance is a lie.
    assert_pickle_consistent(ckpt, model)

    vocab, rule, version = read_checkpoint_vocab(ckpt, source=str(path))
    assert_vocab_matches(vocab, session_vocab, source=str(path))

    # A legacy pickle predates the metadata but still carries a live lookup; the
    # inferred vocabulary must agree with it or the inference was wrong.
    live = list(embedder.eid_lookup)
    if list(vocab) != live:
        from .vocab import VocabularyError
        raise VocabularyError(
            f"Inferred a {len(vocab)}-entry vocabulary for {path} but the pickled "
            f"module carries {len(live)} entries. Its row meanings cannot be "
            f"established, so it cannot be loaded safely."
        )
    return model, vocab, rule, version


def load_model_from_state_dict(path, config, session_vocab=None,
                               map_location="cpu", **model_kwargs):
    """Build a model from a portable .state_dict.pt, metadata first.

    The ordering here is required, not tidy. ``NDT1.__init__`` sizes
    ``nn.Embedding(len(vocab), hidden_size)`` from the vocabulary, so the
    vocabulary has to be known *before* the model is constructed. Constructing
    first and reconciling afterwards gives either a shape error or -- far worse
    -- a table of the right shape in the wrong order, which loads cleanly and
    silently attributes every session's learned embedding to a different
    session.

        load checkpoint metadata
            -> get session_vocab
            -> construct NDT1 using that vocab
            -> load_state_dict

    Returns ``(model, vocab, rule, version)``.
    """
    from .models.ndt1 import NDT1
    from .vocab import read_checkpoint_vocab, assert_vocab_matches, VocabularyError

    ckpt = load_checkpoint(path, map_location=map_location)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"{path} is not a portable state_dict checkpoint; use "
            f"resave_as_state_dict() to produce one."
        )
    sd = ckpt["state_dict"]

    vocab, rule, version = read_checkpoint_vocab(ckpt, source=str(path))
    vocab = assert_vocab_matches(vocab, session_vocab, source=str(path))

    # M3: an EID-keyed (v2) model needs the [eid, n_neurons] map to be built. A v2
    # checkpoint records it. A v1 checkpoint does not, so it is supplied by the
    # caller (e.g. from run_summary.json) or, for one session with one key,
    # inferred -- and the count keys are renamed under remap_v1_stitch_keys's
    # injectivity rule.
    from .vocab import SESSION_MAP_KEY
    smap = ckpt.get(SESSION_MAP_KEY) or model_kwargs.pop("session_num_neurons", None)
    counts = sorted({int(m.group(2)) for k in sd for m in [_V1_KEY.match(k)] if m})
    if counts:
        if smap is None and len(vocab) == 1 and len(counts) == 1:
            smap = [[vocab[0], counts[0]]]
        if smap is None:
            raise VocabularyError(
                f"{path} is count-keyed (v1) with {len(vocab)} sessions; pass "
                f"session_num_neurons to map its projections to sessions")
        sd = remap_v1_stitch_keys(sd, smap)
    model_kwargs.pop("session_num_neurons", None)
    model = NDT1(config, session_vocab=vocab, session_num_neurons=smap, **model_kwargs)
    try:
        model.load_state_dict(sd)
    except RuntimeError as e:
        emb = sd.get("encoder.embedder.embed_session.weight")
        rows = tuple(emb.shape)[0] if emb is not None else None
        raise VocabularyError(
            f"state_dict from {path} does not fit a model built with its own "
            f"{len(vocab)}-entry vocabulary"
            + (f" (embed_session has {rows} rows)" if rows is not None else "")
            + f". Underlying error: {e}"
        ) from e
    return model, vocab, rule, version
