"""Session vocabulary: construction, provenance, and the checkpoint load contract.

After 1c the session-embedding row numbers carry semantic meaning: row *i* is
the session at index *i* of the vocabulary the model was built with. A
checkpoint that does not record its vocabulary therefore cannot be interpreted,
and two vocabularies of the same length in different orders produce a
`state_dict` that loads without error and silently attributes every session's
learned embedding to a different session.

This module holds the three things that keeps honest: how a vocabulary is
derived, how it travels with a checkpoint, and what a reader is allowed to
assume.
"""

# Bumped when the checkpoint layout changes in a way a reader must notice.
#   missing -> v0, legacy, written before vocabularies were recorded
#   1       -> explicit-vocabulary checkpoint
#   2       -> reserved for 1b, when the stitcher/decoder state_dict is re-keyed
CHECKPOINT_FORMAT_VERSION = 1
LEGACY_FORMAT_VERSION = 0

# How the order was derived. This describes derivation only; it is not a place
# to record load-time events.
RULE_MANIFEST = "manifest"            # position in the ordered training manifest
RULE_SINGLE_EID = "single_eid"        # a one-session ad-hoc run
RULE_LEGACY = "legacy_target_eids"    # the shipped 40, in shipped order
RULE_SORTED = "sorted_fallback"       # no ordering supplied; sorted for determinism

VOCAB_KEY = "session_vocab"
RULE_KEY = "vocabulary_ordering_rule"
VERSION_KEY = "checkpoint_format_version"


class VocabularyError(ValueError):
    """Raised when a vocabulary cannot be trusted to mean what a caller assumes."""


def bare(eid):
    """Session eids appear both bare and with an _aligned suffix; the bare form
    is canonical, and is what the parquet `eid` column holds."""
    return eid.removesuffix("_aligned")


def vocab_for_single_session(eid):
    """Vocabulary for an ad-hoc single-session run."""
    return [bare(eid)], RULE_SINGLE_EID


def vocab_for_sessions(loaded_eids, ordered_eids=None):
    """Vocabulary for a multi-session run.

    ``loaded_eids`` is what actually loaded, typically ``meta_data["eids"]``,
    which is a **set**: iteration order over strings varies between processes
    unless PYTHONHASHSEED is fixed, so it can never define row order on its own.

    ``ordered_eids`` is the training manifest. When given it is authoritative,
    filtered to what actually loaded so a session that failed to load does not
    reserve a row. Without it the order falls back to sorted, which is
    deterministic but unstable under set growth: adding sessions re-indexes the
    ones already there. That is why the manifest is the normal scaling path and
    sorting is only a fallback.
    """
    loaded = {bare(e) for e in loaded_eids}
    if ordered_eids:
        seen, vocab = set(), []
        for e in ordered_eids:
            b = bare(e)
            if b in loaded and b not in seen:
                seen.add(b)
                vocab.append(b)
        missing = loaded - seen
        if missing:
            raise VocabularyError(
                f"{len(missing)} session(s) loaded but absent from the supplied "
                f"ordering, so they would have no embedding row: {sorted(missing)[:5]}"
            )
        return vocab, RULE_MANIFEST
    return sorted(loaded), RULE_SORTED


def vocab_metadata(vocab, rule, version=CHECKPOINT_FORMAT_VERSION):
    """The three provenance fields, as written into every artifact."""
    return {VOCAB_KEY: list(vocab), RULE_KEY: rule, VERSION_KEY: version}


def read_checkpoint_vocab(ckpt, legacy_vocab=None, source="checkpoint"):
    """Recover (vocab, rule, version) from a loaded checkpoint dict.

    A checkpoint with no vocabulary predates 1c and was necessarily built with
    the shipped 40-entry list in shipped order, so that is inferred rather than
    guessed. The inference is logged: a silent assumption about row meaning is
    exactly what this module exists to prevent.
    """
    if not isinstance(ckpt, dict):
        ckpt = {}
    vocab = ckpt.get(VOCAB_KEY)
    if vocab is not None:
        version = ckpt.get(VERSION_KEY, CHECKPOINT_FORMAT_VERSION)
        rule = ckpt.get(RULE_KEY)
        print(f"[vocab] {source}: {len(vocab)} entries, rule={rule}, format v{version}")
        return list(vocab), rule, version

    if legacy_vocab is None:
        from .compat import legacy_session_vocab
        legacy_vocab = legacy_session_vocab()
    print(f"[vocab] {source}: no vocabulary recorded; inferring the shipped "
          f"{len(legacy_vocab)}-entry target_eids.txt ordering (format v0 legacy)")
    return list(legacy_vocab), RULE_LEGACY, LEGACY_FORMAT_VERSION


def assert_vocab_matches(checkpoint_vocab, caller_vocab, source="checkpoint"):
    """The caller-supplied case of the load contract.

    Passing nothing means "use whatever the checkpoint says" and is handled by
    the caller. Passing a vocabulary is an assertion of intent, so it must match
    exactly; quietly overriding it would discard the very thing the caller was
    trying to pin down.
    """
    if caller_vocab is None:
        return list(checkpoint_vocab)
    caller = [bare(e) for e in caller_vocab]
    ckpt = [bare(e) for e in checkpoint_vocab]
    if caller == ckpt:
        return list(checkpoint_vocab)

    if len(caller) != len(ckpt):
        detail = f"length {len(caller)} vs {len(ckpt)} in the {source}"
    else:
        i = next(j for j, (a, b) in enumerate(zip(caller, ckpt)) if a != b)
        detail = (f"same length ({len(caller)}) but different order; first "
                  f"divergence at row {i}: caller has {caller[i]!r}, "
                  f"{source} has {ckpt[i]!r}")
    raise VocabularyError(
        "Supplied session vocabulary does not match the checkpoint's: " + detail +
        ". Row indices are the checkpoint's weights, so loading anyway would "
        "attribute learned embeddings to the wrong sessions. Pass no vocabulary "
        "to accept the checkpoint's, or pass one that matches."
    )


def assert_eid_in_vocab(vocab, eid, source="checkpoint"):
    """No extension: a session the checkpoint never saw has no row to use."""
    b = bare(eid)
    if b not in {bare(e) for e in vocab}:
        raise VocabularyError(
            f"Session {b!r} is not in the {source} vocabulary ({len(vocab)} "
            f"entries), so it has no session-embedding row. Extending the table "
            f"is a transfer-learning feature and is deliberately not supported."
        )
    return b


def assert_pickle_consistent(ckpt, model):
    """A pickled checkpoint carries a live module *and* recorded metadata, so the
    two can drift apart. They must agree, or the recorded provenance is a lie."""
    recorded = ckpt.get(VOCAB_KEY) if isinstance(ckpt, dict) else None
    if recorded is None:
        return
    embedder = getattr(getattr(model, "encoder", None), "embedder", None)
    live = getattr(embedder, "eid_lookup", None)
    if live is None:
        return
    if list(recorded) != list(live):
        raise VocabularyError(
            f"Checkpoint metadata records a {len(recorded)}-entry vocabulary but "
            f"the pickled module carries a {len(live)}-entry one. The checkpoint "
            f"cannot be interpreted; its provenance and its weights disagree."
        )
