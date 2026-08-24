from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from loader.make_loader import make_loader
from utils.utils import set_seed
from utils.config_utils import config_from_kwargs, update_config
from utils.dataset_utils import load_ibl_dataset_locally
from models.ndt1 import NDT1
from models.stpatch import STPatch
import torch
import os
from trainer.make import make_trainer
import argparse
from utils.optimizer_utils import build_lr_scheduler

JUST_SPIKES = True

ap = argparse.ArgumentParser()
ap.add_argument("--eid", type=str, default='c7248e09-8c0d-40f2-9eb4-700a8973d8c8_aligned')
ap.add_argument("--train-batch-size", type=int, default=None)
ap.add_argument("--eval-batch-size", type=int, default=None)
ap.add_argument("--base-path", type=str, default='/work/hdd/beml/ac136')
ap.add_argument("--data-type", type=str, default="benchmark_datasets")
ap.add_argument("--results-path", type=str, default="benchmark_results/gpu_throughput/local")
args = ap.parse_args()

eid = args.eid
BASE_PATH = args.base_path
DATA_TYPE = args.data_type
RESULTS_PATH = args.results_path

# load config
kwargs = {
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

# set seed for reproducibility
set_seed(config.seed)

# load dataset
train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset_locally(
                            eid=eid,
                            num_sessions=1,
                            split_method=config.data.split_method,
                            train_session_eid=[eid],
                            test_session_eid=config.data.test_session_eid,
                            batch_size=config.training.train_batch_size,
                            seed=config.seed,
                            just_spikes=JUST_SPIKES,
                            data_type=DATA_TYPE,
                            base_path=BASE_PATH
                            )

eid_name = eid[:-len("_aligned")] if eid.endswith("_aligned") else eid

log_dir = os.path.join(BASE_PATH, RESULTS_PATH, eid_name, "train")
os.makedirs(log_dir, exist_ok=True)

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

# train loop
trainer.train()
