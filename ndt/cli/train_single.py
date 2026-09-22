"""Single-session SSL training.

Stage-1 port note: the body below is a line-for-line copy of
``src/train_sessions.py`` from the source repo, wrapped in ``run(args)`` so that
importing this module no longer starts a training run. Behavior is unchanged.
"""

from ..utils.dataset_utils import load_ibl_dataset_locally
from accelerate import Accelerator
from ..loader.make_loader import make_loader
from ..utils.utils import set_seed
from ..utils.config_utils import config_from_kwargs, update_config
from ..models.ndt1 import NDT1
from ..vocab import vocab_for_single_session
import torch
import os
from ..trainer.make import make_trainer
import argparse
from ..utils.optimizer_utils import build_lr_scheduler
from ..paths import config_path

BASE_PATH = "/work/nvme/beml/ac136/IBL_MTM/data"
DATA_TYPE = "processed_mtm"
RESULTS_PATH = "/work/nvme/beml/ac136/IBL_MTM/results/mtm"
OUTPUT_MODEL_NAME = "mtm-single-session_test"
# DATA_TYPE = "just_spikes"
# RESULTS_PATH = "training_og"

# DATA_TYPE = "processed_qixian_data_old"
# RESULTS_PATH = "qixian_model/early_stopping/param_8"

# DATA_TYPE = "processed_kimia_020923_concat/"
# RESULTS_PATH = "results_kimia_020923/"

# DATA_TYPE = "benchmark_datasets"
# RESULTS_PATH = "benchmark_results"

# DATA_TYPE = "processed_recording1_folds"
# RESULTS_PATH = "results_recording1_folds"


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eid", type=str, default='c7248e09-8c0d-40f2-9eb4-700a8973d8c8_aligned')
    ap.add_argument("--base-path", type=str, default=BASE_PATH)
    ap.add_argument("--data-type", type=str, default=DATA_TYPE)
    # Points the loader at a directory whose basename differs from data_type.
    # data_type still selects the dataset *format* and cannot be inferred from
    # the path, so the two stay separate. None means "use base_path/data_type",
    # so this is inert unless passed.
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--results-path", type=str, default=RESULTS_PATH)
    ap.add_argument("--output-model-name", type=str, default=OUTPUT_MODEL_NAME)
    # Added for the Stage-1b parity runs. None means "use the config", so the
    # flag is inert unless passed. Applied before the trainer and the LR
    # scheduler are built, since build_lr_scheduler derives total_steps from it.
    ap.add_argument("--num-epochs", type=int, default=None)
    # 2c item 3. None means "use the config", so this is inert unless passed and
    # the MtM recipe file stays at its published values. Overriding here puts the
    # choice in the recorded args instead of silently editing the recipe.
    # Plotting touches no RNG, so this cannot move training.
    ap.add_argument("--save-plot-every-n-epochs", type=int, default=None)
    return ap


def build_config():
    """Resolve the config exactly as the original script did.

    Split out so that parity checks can diff the resolved config without
    loading data or building a model.
    """
    kwargs = {
        # "model": "include:src/configs/ndt1_stitching_prompting.yaml"
        "model": "include:" + config_path("ndt1_stitching")

    }

    config = config_from_kwargs(kwargs)
    config = update_config(config_path("ndt1_stitching"), config)
    config = update_config(config_path("ssl_session_trainer"), config)  # single session
    return config


def run(args):
    eid = args.eid
    eid_name = eid[:-len("_aligned")] if eid.endswith("_aligned") else eid
    base_path = args.base_path
    data_type = args.data_type
    data_root = args.data_root
    results_path = args.results_path
    output_model_name = args.output_model_name

    print()
    print("Args to train_sessions:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print()

    # load config
    config = build_config()

    # Item assignment, not attribute assignment: DictConfig.__getattr__ returns
    # a fresh DictConfig wrapping the inner dict, so config.training.num_epochs = x
    # would write to a throwaway object and silently do nothing.
    if getattr(args, "num_epochs", None) is not None:
        config["training"]["num_epochs"] = args.num_epochs
        print(f"num_epochs overridden to {args.num_epochs}")
    if getattr(args, "save_plot_every_n_epochs", None) is not None:
        config["training"]["save_plot_every_n_epochs"] = args.save_plot_every_n_epochs
        print(f"save_plot_every_n_epochs overridden to {args.save_plot_every_n_epochs}")

    # set seed for reproducibility
    set_seed(config.seed)

    # load dataset
    train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset_locally(
                                eid=eid,
                                num_sessions=1,  # 1
                                train_session_eid=[eid],
                                test_session_eid=config.data.test_session_eid,  # []
                                data_type=data_type,
                                base_path=base_path,
                                data_root=data_root
                                )

    # 2c item 4: what this run actually trained on. `test` is loaded and not used
    # for pretraining -- the deliberate MtM policy -- so recording all three makes
    # that auditable from the checkpoint rather than inferred from the code.
    split_counts = {"n_train": len(train_dataset), "n_val": len(val_dataset),
                    "n_test": len(test_dataset)}
    print(f"split counts: {split_counts}  (policy: train_only)")

    log_dir = os.path.join(results_path,
                                output_model_name,
                                "train",
                                "{}".format(eid_name))
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
    # 1c-ii: the vocabulary is this run's own session, not the shipped 40. A
    # one-entry table is what lets an arbitrary eid train at all, and it changes
    # the embedding row from 10 to 0 for the old canonical session -- a
    # legitimate difference that retires baseline_pre_tier1.
    session_vocab, vocabulary_ordering_rule = vocab_for_single_session(eid)
    print(f"session vocab ({vocabulary_ordering_rule}): {session_vocab}")
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
