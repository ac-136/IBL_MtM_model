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


# --------------------------------------------------------------------------
# Session-specific parameters, and the transfer contract
# --------------------------------------------------------------------------
#
# Transferring a model pretrained on session S onto a different session H must
# keep the shared backbone and re-initialise everything that belongs to S.
#
# This used to happen only half deliberately. `embed_session.weight` was popped
# explicitly, but the stitcher and decoder stayed fresh purely because their
# parameter names embed the *cluster count* --
# `encoder.stitcher.stitcher_dict.406.weight` -- so with differing counts they
# found no match and `strict=False` dropped them. Two sessions with the same
# cluster count would have loaded S's input and output projections straight
# into H: matching shapes, no error, wrong session.
#
# Both lists are kept, not just the exclusions, so that a session-specific
# module added later cannot inherit by default merely for being absent from a
# denylist. `classify_state_dict_keys` refuses to classify anything it does not
# recognise, so a new module breaks loudly instead of transferring silently.

SESSION_SPECIFIC_PREFIXES = (
    "encoder.embedder.embed_session.",
    "encoder.stitcher.",
    "stitch_decoder.",
)

# The embedder's shared children are listed individually rather than as a
# blanket `encoder.embedder.`, so no prefix-ordering rule is needed to stop it
# swallowing `embed_session`.
#
# `decoder.` is shared and is the entry most easily misfiled, sitting one
# underscore away from the session-specific `stitch_decoder.`. It is
# [n_channels, hidden] regardless of cluster count, so it belongs to the
# backbone; excluding it would throw away most of what transfer is for.
SHARED_PREFIXES = (
    "encoder.embedder.embed_pos.",
    "encoder.embedder.embed_spikes.",
    "encoder.embedder.projection.",
    "encoder.layers.",
    "encoder.out_norm.",
    "decoder.",
)


def classify_state_dict_keys(keys):
    """Partition checkpoint keys into (shared, session_specific).

    Raises VocabularyError if any key matches neither list. That is the point:
    an unrecognised parameter is exactly the case where a silent default would
    be wrong, and there is no safe guess about whether it belongs to a session.
    """
    shared, specific, unknown = [], [], []
    for k in keys:
        if k.startswith(SESSION_SPECIFIC_PREFIXES):
            specific.append(k)
        elif k.startswith(SHARED_PREFIXES):
            shared.append(k)
        else:
            unknown.append(k)
    if unknown:
        raise VocabularyError(
            f"{len(unknown)} checkpoint key(s) match neither SHARED_PREFIXES nor "
            f"SESSION_SPECIFIC_PREFIXES, so it cannot be decided whether they "
            f"belong to a session: {sorted(unknown)[:8]}. Add them to one list in "
            f"ndt/vocab.py -- do not let them transfer by default."
        )
    return shared, specific


def filter_transfer_state_dict(state_dict, verbose=True):
    """Keep only the shared backbone from a pretrained checkpoint.

    Everything session-specific is dropped so the target initialises it fresh.
    Key names are logged, not just counts, so the split is auditable from the
    job log rather than taken on trust.
    """
    shared, specific = classify_state_dict_keys(state_dict.keys())
    if verbose:
        print(f"[transfer] loading {len(shared)} shared tensor(s) from the source checkpoint")
        print(f"[transfer] dropping {len(specific)} session-specific tensor(s), "
              f"re-initialised for the target:")
        for k in sorted(specific):
            print(f"[transfer]   drop {k}")
    return {k: state_dict[k] for k in shared}


def source_checkpoint_identity(path, expected_eid=None, expected_clusters=None,
                               role="best"):
    """Pin down exactly which model a transfer run started from.

    Records sha256, eid, path, best epoch and which of the three saved
    checkpoints was used -- `best`, `last` and `epoch` all sit in the same
    directory and only the filename tells them apart, so the role has to be
    recorded or "which model was this?" stays inferable rather than known.

    Two checks are assertions, not notes, because both would otherwise surface
    as a confusing Stage-3 result rather than a clear failure here:
    the checkpoint's vocabulary must be exactly the source session, and its
    stitcher must be sized for that session's actual cluster count.
    """
    import hashlib
    from .compat import load_checkpoint

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()

    ckpt = load_checkpoint(path)
    vocab = ckpt.get(VOCAB_KEY) if isinstance(ckpt, dict) else None
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None

    if expected_eid is not None:
        want = [bare(expected_eid)]
        got = [bare(e) for e in (vocab or [])]
        if got != want:
            raise VocabularyError(
                f"Source checkpoint {path} has vocabulary {got}, expected {want}. "
                f"Either the wrong checkpoint was passed or its provenance is wrong; "
                f"a transfer from it could not be attributed to a source session."
            )

    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    stitcher = getattr(getattr(model, "encoder", None), "stitcher", None)
    keys = sorted(stitcher.stitcher_dict.keys()) if stitcher is not None else []
    ckpt_clusters = int(keys[0]) if len(keys) == 1 and keys[0].isdigit() else None
    if expected_clusters is not None and ckpt_clusters is not None:
        if ckpt_clusters != int(expected_clusters):
            raise VocabularyError(
                f"Source checkpoint {path} has a stitcher for {ckpt_clusters} clusters "
                f"but the source session's data has {expected_clusters}. The checkpoint "
                f"does not match the session it claims to come from."
            )

    return {
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": digest,
        "source_checkpoint_role": role,
        "source_checkpoint_epoch": epoch,
        "source_eid": (vocab or [None])[0],
        "source_clusters": ckpt_clusters,
        "source_" + VERSION_KEY: ckpt.get(VERSION_KEY) if isinstance(ckpt, dict) else None,
        "source_" + RULE_KEY: ckpt.get(RULE_KEY) if isinstance(ckpt, dict) else None,
    }
