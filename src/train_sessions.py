from datasets import load_dataset, load_from_disk, concatenate_datasets, load_dataset_builder
from utils.dataset_utils import get_user_datasets, load_ibl_dataset_locally, split_both_dataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import get_data_from_h5
from models.ndt1 import NDT1
from models.stpatch import STPatch
import torch
import numpy as np
import subprocess
import os
from trainer.make import make_trainer
import threading
import argparse
from utils.optimizer_utils import build_lr_scheduler

JUST_SPIKES = True
BASE_PATH = '/work/hdd/beml/ac136'
# DATA_TYPE = "just_spikes"
# RESULTS_PATH = "training_og"

# DATA_TYPE = "processed_qixian_data_old"
# RESULTS_PATH = "qixian_model/early_stopping/param_8"

# DATA_TYPE = "processed_kimia_020923_concat/"
# RESULTS_PATH = "results_kimia_020923/"

DATA_TYPE = "benchmark_datasets"
RESULTS_PATH = os.environ.get(
    "TRAIN_SESSIONS_RESULTS_PATH",
    "benchmark_results/gpu_throughput/local",
)

# Optionally copy and use datasets from a fast tmpfs location.
# Set `USE_TMP_DATA=1` and optionally `TMP_DATA_DIR` to enable.
# Optionally set USE_TMP_DATA later if needed (removed tmp-copy logic)

# DATA_TYPE = "processed_recording1_folds"
# RESULTS_PATH = "results_recording1_folds"

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='c7248e09-8c0d-40f2-9eb4-700a8973d8c8_aligned')
ap.add_argument("--train-batch-size", type=int, default=None)
ap.add_argument("--eval-batch-size", type=int, default=None)
ap.add_argument("--benchmark-repeat-factor", type=int, default=1)
args = ap.parse_args()


eid = args.eid

# load config
kwargs = {
    # "model": "include:src/configs/ndt1_stitching_prompting.yaml"
    "model": "include:src/configs/ndt1_stitching.yaml"

}


config = config_from_kwargs(kwargs)
config = update_config("src/configs/ndt1_stitching.yaml", config)
config = update_config("src/configs/ssl_session_trainer.yaml", config) # single session
if args.train_batch_size is not None:
    config["training"]["train_batch_size"] = args.train_batch_size
if args.eval_batch_size is not None:
    config["training"]["test_batch_size"] = args.eval_batch_size
print(
    "Effective batch sizes: "
    f"train={config.training.train_batch_size}, eval={config.training.test_batch_size}"
)
print(f"Benchmark repeat factor: {args.benchmark_repeat_factor}")

# config = update_config("src/configs/ssl_sessions_trainer.yaml", config)

# set seed for reproducibility
set_seed(config.seed)

# load dataset
# eid = 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8_aligned'
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset_locally(
                            eid=eid,
                            num_sessions=1, # 1
                            split_method=config.data.split_method, # predefined
                            train_session_eid=[eid],
                            test_session_eid=config.data.test_session_eid, # []
                            batch_size=config.training.train_batch_size, # 8
                            eval_batch_size=config.training.test_batch_size,
                            seed=config.seed,
                            just_spikes=JUST_SPIKES,
                            data_type=DATA_TYPE,
                            base_path=BASE_PATH
                            )

def repeat_for_benchmark(dataset, repeat_factor, split_name):
    if repeat_factor <= 1 or len(dataset) == 0:
        return dataset
    repeated_dataset = concatenate_datasets([dataset] * repeat_factor)
    print(
        f"Repeated {split_name} dataset for benchmarking: "
        f"{len(dataset)} -> {len(repeated_dataset)} rows "
        f"(factor {repeat_factor})."
    )
    return repeated_dataset

if args.benchmark_repeat_factor < 1:
    raise ValueError("--benchmark-repeat-factor must be >= 1")

train_dataset = repeat_for_benchmark(
    train_dataset, args.benchmark_repeat_factor, "train"
)
val_dataset = repeat_for_benchmark(
    val_dataset, args.benchmark_repeat_factor, "val"
)

# # download dataset from huggingface
# eid = None
# train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset(config.dirs.dataset_cache_dir, 
#                            config.dirs.huggingface_org,
#                            eid='5dcee0eb-b34d-4652-acc3-d10afc6eae68',
#                            num_sessions=config.data.num_sessions,
#                            split_method=config.data.split_method,
#                            test_session_eid=config.data.test_session_eid,
#                            batch_size=config.training.train_batch_size,
#                            use_re=config.data.use_re,
#                            seed=config.seed)
# if config.data.use_aligned_test:
#     # aligned dataset
#     if eid is None:
#         test_dataset = load_from_disk(os.path.join('data', config.dirs.behav_dir))
#         data_columns = ['spikes_sparse_data', 'spikes_sparse_indices', 'spikes_sparse_indptr', 'spikes_sparse_shape']
#         test_dataset = concatenate_datasets([test_dataset["train"], test_dataset["val"], test_dataset["test"]])
#         test_dataset = test_dataset.select_columns(data_columns)
#     else:
#         aligned_dataset = load_from_disk(os.path.join('data', config.dirs.behav_dir))
#         aligned_dataset = concatenate_datasets([aligned_dataset["train"], aligned_dataset["val"], aligned_dataset["test"]])
#         train_dataset, test_dataset = split_both_dataset(aligned_dataset=aligned_dataset,
#                                                          unaligned_dataset=train_dataset,
#                                                          seed=config.seed)

num_sessions = len(meta_data["eids"])
gpu_count = int(os.environ.get("GPU_BENCHMARK_GPUS", 1))
global_train_batch_size = config.training.train_batch_size * gpu_count
run_name = (
    f"{eid}_{config.training.num_epochs}epochs"
    f"_trainbs{config.training.train_batch_size}"
    f"_evalbs{config.training.test_batch_size}"
    f"_gpus{gpu_count}"
    f"_globalbs{global_train_batch_size}"
    f"_repeat{args.benchmark_repeat_factor}"
)

log_dir = os.path.join(BASE_PATH, RESULTS_PATH, 
                            "train", 
                            # "num_session_{}".format(num_sessions), 
                            # "model_{}".format(config.model.model_class), 
                            # "method_{}".format(config.method.model_kwargs.method_name), 
                            # "mask_{}".format(config.encoder.masker.mode),
                            # "stitch_{}".format(config.encoder.stitching), 
                            run_name)
os.makedirs(log_dir, exist_ok=True)


# # make log dir
# log_dir = os.path.join(config.dirs.log_dir, 
#                        "train", 
#                        "num_session_{}".format(num_sessions), 
#                        "model_{}".format(config.model.model_class), 
#                        "method_{}".format(config.method.model_kwargs.method_name), 
#                        "mask_{}".format(config.encoder.masker.mode),
#                        "stitch_{}".format(config.encoder.stitching))
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# # wandb
# if config.wandb.use:
#     import wandb
#     wandb.init(project=config.wandb.project, entity=config.wandb.entity, config=config, name="train_model_{}_num_session_{}_method_{}_mask_{}_stitch_{}".format(config.model.model_class, num_sessions,config.method.model_kwargs.method_name,config.encoder.masker.mode, config.encoder.stitching))

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
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

if accelerator.is_main_process:
    print("Meta data: ")
    print(meta_data)
    print()

# load model
NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch}

config = update_config(config, meta_data)
model_class = NAME2MODEL[config.model.model_class]
model = model_class(config.model, **config.method.model_kwargs, **meta_data)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)

model, optimizer, train_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
)
lr_scheduler = build_lr_scheduler(
    optimizer=optimizer,
    config=config,
    steps_per_epoch=len(train_dataloader),
)
lr_scheduler = accelerator.prepare(lr_scheduler)

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
# # Shared variable to signal the dummy load to stop
# stop_dummy_load = threading.Event()
# if config.training.dummy:
#     # This is for HPC GPU usage, to avoid the GPU being idle
#     print("Running dummy load")
#     # Run dummy load in a separate thread
#     dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load,))
#     dummy_thread.start()
#     try:
#         # train loop
#         trainer.train()
#     finally:
#         # Signal the dummy load to stop and wait for the thread to finish
#         stop_dummy_load.set()
#         dummy_thread.join()
# else:

# train loop
trainer.train()
