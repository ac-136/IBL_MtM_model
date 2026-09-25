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
# v2 (M3): stitcher and stitch-decoder modules are keyed by session EID, and the
# checkpoint records its ordered [eid, n_neurons] map. v1 was keyed by count.
CHECKPOINT_FORMAT_VERSION = 2
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
SESSION_MAP_KEY = "session_num_neurons"   # v2: ordered [eid, n_neurons]


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


def vocab_metadata(vocab, rule, version=CHECKPOINT_FORMAT_VERSION, session_num_neurons=None):
    """The provenance fields written into every artifact. v2 adds the ordered
    [eid, n_neurons] map, which is what makes an EID-keyed checkpoint
    reconstructable from its metadata alone."""
    out = {VOCAB_KEY: list(vocab), RULE_KEY: rule, VERSION_KEY: version}
    if session_num_neurons is not None:
        out[SESSION_MAP_KEY] = [[str(e), int(n)] for e, n in session_num_neurons]
    return out


def model_session_num_neurons(model):
    """Ordered [eid, n_neurons] read from an EID-keyed stitcher, or None.

    Read from the weights, not tracked separately, so the recorded map cannot
    drift from the model it describes. None for a v1 (count-keyed) module."""
    enc = getattr(model, "encoder", model)
    stitcher = getattr(enc, "stitcher", None)
    if stitcher is None or getattr(stitcher, "key_by", "count") != "eid":
        return None
    return [[e, int(m.in_features)] for e, m in stitcher.stitcher_dict.items()]


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
    # M3 (v2): the recorded [eid, n_neurons] map must match the live stitcher.
    smap = ckpt.get(SESSION_MAP_KEY) if isinstance(ckpt, dict) else None
    if smap is not None:
        live_map = model_session_num_neurons(model)
        if [[str(e), int(n)] for e, n in smap] != live_map:
            raise VocabularyError(
                f"Checkpoint metadata records session map {smap} but the pickled "
                f"stitcher is {live_map}; provenance and weights disagree.")


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


def stitcher_session_sizes(model, vocab=None, session_num_neurons=None):
    """(M3, M-g) {eid: n_neurons} for each session's input projection, or None.

    v2: read from the EID keys. v1 (count keys): only unambiguous for a single
    session, where the one key belongs to the one vocabulary entry; otherwise None,
    since a count-keyed module cannot say which session a projection belongs to.
    """
    enc = getattr(model, "encoder", None)
    stitcher = getattr(enc, "stitcher", None)
    if stitcher is None:
        return None
    mods = stitcher.stitcher_dict
    if getattr(stitcher, "key_by", "count") == "eid":
        return {e: int(m.in_features) for e, m in mods.items()}
    if vocab is not None and len(vocab) == 1 and len(mods) == 1:
        (m,) = mods.values()
        return {bare(vocab[0]): int(m.in_features)}
    # v1 multi-session with an externally recorded map (e.g. run_summary.json):
    # attributable only under the same rule as remapping -- the map must be
    # injective and its counts must be exactly the projections present.
    if session_num_neurons:
        smap = {bare(e): int(n) for e, n in session_num_neurons}
        counts = list(smap.values())
        if len(set(counts)) == len(counts) and \
                sorted(counts) == sorted(int(m.in_features) for m in mods.values()) and \
                all(str(n) in mods for n in counts):
            return smap
    return None


def source_checkpoint_identity(path, expected_eid=None, expected_clusters=None,
                               role="best", expected_vocab=None,
                               expected_session_num_neurons=None):
    """Pin down exactly which model a transfer run started from.

    Records sha256, path, role, epoch and the source's sessions. The three saved
    checkpoints (`best`, `last`, `epoch`) differ only by filename, so the role
    has to be recorded or "which model was this?" stays inferable.

    Assertions, not notes, because each would otherwise surface as a confusing
    downstream result:
      * single-session source: vocabulary == [expected_eid], and its projection is
        sized for expected_clusters (unchanged from Stage 2c);
      * multi-session source (M3, M-g): vocabulary == expected_vocab, and each
        session's projection is sized for its entry in expected_session_num_neurons.
    Per-session sizes come from module shapes, so v1 and v2 checkpoints both work.
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
    got_vocab = [bare(e) for e in (vocab or [])]

    if expected_eid is not None:
        want = [bare(expected_eid)]
        if got_vocab != want:
            raise VocabularyError(
                f"Source checkpoint {path} has vocabulary {got_vocab}, expected {want}. "
                f"Either the wrong checkpoint was passed or its provenance is wrong; "
                f"a transfer from it could not be attributed to a source session."
            )
    if expected_vocab is not None and got_vocab != [bare(e) for e in expected_vocab]:
        raise VocabularyError(
            f"Source checkpoint {path} has vocabulary {got_vocab}, expected "
            f"{[bare(e) for e in expected_vocab]} (order matters: it defines embedding rows).")

    model = ckpt.get("model") if isinstance(ckpt, dict) else None
    sizes = stitcher_session_sizes(model, got_vocab)
    single = sizes[got_vocab[0]] if sizes and len(got_vocab) == 1 else None
    if expected_clusters is not None and single is not None and single != int(expected_clusters):
        raise VocabularyError(
            f"Source checkpoint {path} has a stitcher for {single} clusters "
            f"but the source session's data has {expected_clusters}. The checkpoint "
            f"does not match the session it claims to come from."
        )
    if expected_session_num_neurons is not None:
        want = {bare(e): int(n) for e, n in expected_session_num_neurons}
        if sizes != want:
            raise VocabularyError(
                f"Source checkpoint {path} has per-session projection sizes {sizes}, "
                f"expected {want}.")

    return {
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": digest,
        "source_checkpoint_role": role,
        "source_checkpoint_epoch": epoch,
        "source_eid": got_vocab[0] if len(got_vocab) == 1 else None,
        "source_clusters": single,
        "source_session_vocab": got_vocab or None,
        "source_session_num_neurons": ([[e, sizes[e]] for e in got_vocab if e in sizes]
                                       if sizes else None),
        "source_" + VERSION_KEY: ckpt.get(VERSION_KEY) if isinstance(ckpt, dict) else None,
        "source_" + RULE_KEY: ckpt.get(RULE_KEY) if isinstance(ckpt, dict) else None,
    }
