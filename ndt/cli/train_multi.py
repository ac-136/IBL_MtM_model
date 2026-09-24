"""Multi-session SSL training.

Stage-1 port note: originally a line-for-line copy of ``src/multi_train_session.py``
wrapped in ``run(args)``.

M1 (multi-session validation plan): the session list is now EXPLICIT and ordered,
from exactly one of ``--sessions EID...`` or ``--manifest PATH`` (an ordered eid
file such as ``scaling_manifests/<cohort>/seed_<S>/train_<N>_eids.txt``). Nothing
is truncated: ``num_sessions`` is set to the list length, and a disagreeing
``--num-sessions`` is refused. The flags train_single gained are added
(``--data-root``, ``--output-model-name``, ``--save-plot-every-n-epochs``), and
split counts plus the session list's provenance go into run_summary.json. The old
flags ``--train-session-eid`` and ``--model_name`` remain as aliases.

Batching is NOT changed here; that is M2. Until then a genuine multi-session run
still fails at collate (defect 1a).
"""

from ..utils.dataset_utils import load_ibl_dataset_locally
from accelerate import Accelerator
from ..loader.make_loader import make_loader
from ..utils.utils import set_seed
from ..utils.config_utils import config_from_kwargs, update_config
from ..models.ndt1 import NDT1
from ..vocab import vocab_for_sessions, bare
import hashlib
import re
import torch
import os
from ..trainer.make import make_trainer
import argparse
from ..utils.optimizer_utils import build_lr_scheduler
from ..paths import config_path

BASE_PATH = '/work/hdd/beml/ac136'
# DATA_TYPE = "just_spikes"
# RESULTS_PATH = "results_og_multi_session"

DATA_TYPE = "benchmark_datasets"
RESULTS_PATH = "benchmark_results"

_EID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def build_parser():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--sessions", nargs="+", default=None,
                     help="ordered training session eids; order defines embedding rows")
    src.add_argument("--manifest", type=str, default=None,
                     help="file of ordered training eids, one per line")
    # Pre-M1 flag, kept as an alias of --sessions.
    src.add_argument("--train-session-eid", nargs="+", default=None,
                     dest="train_session_eid")
    # Must equal the list length if given; never truncates.
    ap.add_argument("--num-sessions", type=int, default=None)
    ap.add_argument("--output-model-name", "--model_name", type=str, default=None,
                    dest="output_model_name")
    ap.add_argument("--base-path", type=str, default=BASE_PATH)
    ap.add_argument("--data-type", type=str, default=DATA_TYPE)
    # As in train_single: points the loader at a directory whose basename differs
    # from data_type. None means "use base_path/data_type".
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--results-path", type=str, default=RESULTS_PATH)
    # None means "use the config" for both.
    ap.add_argument("--num-epochs", type=int, default=None)
    ap.add_argument("--save-plot-every-n-epochs", type=int, default=None)
    return ap


def resolve_sessions(args):
    """Return (ordered bare eids, source, sha256 or None). Refuses anything implicit."""
    if args.manifest:
        raw = open(args.manifest).read().split()
        source = os.path.abspath(args.manifest)
        digest = hashlib.sha256(open(args.manifest, "rb").read()).hexdigest()
    elif args.sessions or args.train_session_eid:
        raw = list(args.sessions or args.train_session_eid)
        source, digest = "cli", None
    else:
        raise SystemExit("train_multi: give the training sessions explicitly with "
                         "--sessions EID... or --manifest PATH")
    eids = [bare(e) for e in raw]
    bad = [e for e in eids if not _EID.match(e)]
    if bad:
        raise SystemExit(f"train_multi: malformed eid(s): {bad[:3]}")
    dup = sorted({e for e in eids if eids.count(e) > 1})
    if dup:
        raise SystemExit(f"train_multi: duplicate eid(s) in the session list: {dup[:3]}")
    if args.num_sessions is not None and args.num_sessions != len(eids):
        raise SystemExit(f"train_multi: --num-sessions {args.num_sessions} disagrees with "
                         f"the {len(eids)}-session list; refusing to truncate or pad")
    return eids, source, digest


def build_config(args=None):
    """Resolve the config as the original script did, plus explicit overrides.

    ``args`` is optional so that parity checks can diff the base config without
    constructing an argparse namespace.
    """
    kwargs = {
        # "model": "include:src/configs/ndt1_stitching_prompting.yaml"
        "model": "include:" + config_path("ndt1_stitching")

    }

    config = config_from_kwargs(kwargs)
    config = update_config(config_path("ndt1_stitching"), config)
    config = update_config(config_path("ssl_sessions_trainer"), config)  # multi session

    if args is not None:
        eids, _, _ = resolve_sessions(args)
        config["data"]["train_session_eid"] = eids
        config["data"]["num_sessions"] = len(eids)
        # Item assignment, not attribute assignment: DictConfig.__getattr__
        # returns a fresh DictConfig, so attribute writes are silently lost.
        if getattr(args, "num_epochs", None) is not None:
            config["training"]["num_epochs"] = args.num_epochs
            print(f"num_epochs overridden to {args.num_epochs}")
        if getattr(args, "save_plot_every_n_epochs", None) is not None:
            config["training"]["save_plot_every_n_epochs"] = args.save_plot_every_n_epochs
            print(f"save_plot_every_n_epochs overridden to {args.save_plot_every_n_epochs}")
    return config


def run(args):
    base_path = getattr(args, "base_path", BASE_PATH)
    data_type = getattr(args, "data_type", DATA_TYPE)
    results_path = getattr(args, "results_path", RESULTS_PATH)

    print()
    print("Args to multi_train_session:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    # load config
    config = build_config(args)
    requested, list_source, list_sha = resolve_sessions(args)

    # set seed for reproducibility
    set_seed(config.seed)

    # load dataset from the multi-session config rather than forcing a single eid
    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset_locally(
                                eid=None,
                                num_sessions=config.data.num_sessions,
                                train_session_eid=config.data.train_session_eid,
                                test_session_eid=config.data.test_session_eid,  # []
                                data_type=data_type,
                                base_path=base_path,
                                data_root=getattr(args, "data_root", None))

    # M1: what loaded must be exactly what was asked for, in the asked order. The
    # loader now raises on a failed session, so this is belt and braces -- but it
    # is also the check that the per-session neuron map is aligned with the list.
    loaded_order = [e for e, _ in meta_data["session_num_neurons"]]
    if loaded_order != requested:
        raise RuntimeError(f"loaded sessions {loaded_order} != requested {requested}")

    # M2: stitcher/decoder construction order = vocabulary order. NeuralStitcher
    # and StitchDecoder register one module per entry of num_neurons, in list
    # order, so this fixes which RNG draws each session's projections receive --
    # the precondition for M3's remap gate. (Upstream used set order.) Unique
    # counts only: until M3, same-count sessions still share one module (1b).
    ordered_counts = []
    for _, n in meta_data["session_num_neurons"]:
        if n not in ordered_counts:
            ordered_counts.append(n)
    meta_data["num_neurons"] = ordered_counts

    num_sessions = len(meta_data["eids"])
    run_name = (args.output_model_name if getattr(args, "output_model_name", None)
                else "num_session_{}".format(num_sessions))

    # Same layout as train_single: <results>/<output-model-name>/train. With
    # --data-root, results_path is used as given rather than joined to base_path.
    if getattr(args, "data_root", None):
        log_dir = os.path.join(results_path, run_name, "train")
    else:
        log_dir = os.path.join(base_path, results_path, "train", run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # M1: what this run trained on, as train_single records it. `test` is loaded
    # and not used for pretraining -- the MtM policy.
    split_counts = {"n_train": len(train_dataset), "n_val": len(val_dataset),
                    "n_test": len(test_dataset)}
    print(f"split counts: {split_counts}  (policy: train_only)")

    print("Meta data: ")
    print(meta_data)
    print()

    # make the dataloader (made target None and load_meta False)
    train_dataloader = make_loader(train_dataset,
                             target=None,
                             load_meta=False,
                             batch_size=config.training.train_batch_size,
                             pad_to_right=True,
                             pad_value=-1.,
                             max_time_length=config.data.max_time_length,
                             max_space_length=config.data.max_space_length,
                             dataset_name=config.data.dataset_name,
                             sort_by_depth=config.data.sort_by_depth,
                             sort_by_region=config.data.sort_by_region,
                             stitching=config.encoder.stitching,
                             shuffle=True,
                             seed=config.seed)

    val_dataloader = make_loader(val_dataset,
                             target=None,
                             load_meta=False,
                             batch_size=config.training.test_batch_size,
                             pad_to_right=True,
                             pad_value=-1.,
                             max_time_length=config.data.max_time_length,
                             max_space_length=config.data.max_space_length,
                             dataset_name=config.data.dataset_name,
                             sort_by_depth=config.data.sort_by_depth,
                             sort_by_region=config.data.sort_by_region,
                             stitching=config.encoder.stitching,
                             shuffle=False,
                             seed=config.seed)

    # Initialize the accelerator
    accelerator = Accelerator()

    # load model
    NAME2MODEL = {"NDT1": NDT1}

    config = update_config(config, meta_data)
    model_class = NAME2MODEL[config.model.model_class]
    # 1c-ii: the vocabulary is this run's own training sessions, ordered by the
    # manifest when one was supplied. meta_data["eids"] is a set, so it can never
    # define row order on its own; vocab_for_sessions falls back to sorted only
    # when nothing ordered was given, and says so in the recorded rule.
    session_vocab, vocabulary_ordering_rule = vocab_for_sessions(
        meta_data["eids"], ordered_eids=config.data.train_session_eid)
    print(f"session vocab ({vocabulary_ordering_rule}): {len(session_vocab)} entries")
    # M2: the sampler orders sessions by first appearance in the concatenated
    # dataset; that must be the vocabulary order, or per-session bookkeeping and
    # embedding rows would disagree about which session is which.
    for name, dl in (("train", train_dataloader), ("val", val_dataloader)):
        order = getattr(dl.batch_sampler, "session_order", None)
        if order is not None and order != list(session_vocab):
            raise RuntimeError(f"{name} sampler session order {order} != vocab {session_vocab}")
    model = model_class(config.model, **config.method.model_kwargs, **meta_data,
                        session_vocab=session_vocab)
    model = accelerator.prepare(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        config=config,
        steps_per_epoch=len(train_dataloader),
    )

    trainer_kwargs = {
        "log_dir": log_dir,
        "accelerator": accelerator,
        "lr_scheduler": lr_scheduler,
        "config": config,
        "stitching": config.encoder.stitching,
        "vocabulary_ordering_rule": vocabulary_ordering_rule,
        "split_counts": split_counts,
        "pretrain_split_policy": "train_only",
        "session_list_source": list_source,
        "session_list_sha256": list_sha,
    }
    trainer = make_trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        optimizer=optimizer,
        **trainer_kwargs,
        **meta_data
    )

    # train loop
    trainer.train()
    return trainer


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
