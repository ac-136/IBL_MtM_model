"""Multi-session SSL training.

Stage-1 port note: the body below is a line-for-line copy of
``src/multi_train_session.py`` from the source repo, wrapped in ``run(args)`` so
that importing this module no longer starts a training run. Behavior is
unchanged, including the argparse surface.
"""

from ..utils.dataset_utils import load_ibl_dataset_locally
from accelerate import Accelerator
from ..loader.make_loader import make_loader
from ..utils.utils import set_seed
from ..utils.config_utils import config_from_kwargs, update_config
from ..models.ndt1 import NDT1
from ..compat import legacy_session_vocab
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


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-sessions", type=int, default=None)
    ap.add_argument("--train-session-eid", nargs="+", default=None)
    ap.add_argument("--model_name", type=str, default=None)
    # Added during the Stage-1 port. The source script hardcoded these as module
    # constants, which left no way to point the run at a different dataset. The
    # defaults reproduce the original values exactly, so behavior is unchanged
    # unless a flag is passed. Note the repo's own multi_train_sessions_multigpu.sh
    # already passed these three flags, against an argparse that rejected them.
    ap.add_argument("--base-path", type=str, default=BASE_PATH)
    ap.add_argument("--data-type", type=str, default=DATA_TYPE)
    ap.add_argument("--results-path", type=str, default=RESULTS_PATH)
    # Added for the Stage-1b parity runs. None means "use the config".
    ap.add_argument("--num-epochs", type=int, default=None)
    return ap


def build_config(args=None):
    """Resolve the config exactly as the original script did.

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
        if args.train_session_eid is not None:
            config["data"]["train_session_eid"] = args.train_session_eid
        if args.num_sessions is not None:
            config["data"]["num_sessions"] = args.num_sessions
        # Item assignment, not attribute assignment: DictConfig.__getattr__
        # returns a fresh DictConfig, so attribute writes are silently lost.
        if getattr(args, "num_epochs", None) is not None:
            config["training"]["num_epochs"] = args.num_epochs
            print(f"num_epochs overridden to {args.num_epochs}")
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

    # set seed for reproducibility
    set_seed(config.seed)

    # load dataset from the multi-session config rather than forcing a single eid
    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset_locally(
                                eid=None,
                                num_sessions=config.data.num_sessions,
                                train_session_eid=config.data.train_session_eid,
                                test_session_eid=config.data.test_session_eid,  # []
                                data_type=data_type,
                                base_path=base_path)

    num_sessions = len(meta_data["eids"])
    run_name = args.model_name if args.model_name is not None else "num_session_{}".format(num_sessions)

    log_dir = os.path.join(base_path, results_path,
                                "train",
                                run_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

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
                             shuffle=True)

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
                             shuffle=False)

    # Initialize the accelerator
    accelerator = Accelerator()

    # load model
    NAME2MODEL = {"NDT1": NDT1}

    config = update_config(config, meta_data)
    model_class = NAME2MODEL[config.model.model_class]
    # 1c-i: the session vocabulary is now passed explicitly instead of being read
    # from data/target_eids.txt at model-import time. legacy_session_vocab() is the
    # original 40-entry list in its original order, so behaviour is unchanged.
    model = model_class(config.model, **config.method.model_kwargs, **meta_data,
                        session_vocab=legacy_session_vocab())
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
