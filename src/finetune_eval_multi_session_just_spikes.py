import argparse
from math import ceil
from utils.dataset_utils import load_ibl_dataset_locally
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from loader.make_loader import make_loader
from utils.utils import set_seed, dummy_load
from utils.config_utils import config_from_kwargs, update_config
from models.ndt1 import NDT1
from models.stpatch import STPatch
from models.itransformer import iTransformer
import torch
import os
from pathlib import Path
from trainer.make import make_trainer
from utils.eval_utils import load_model_data_local, co_smoothing_eval, behavior_decoding
from torch.optim.lr_scheduler import OneCycleLR
import threading
import warnings
warnings.simplefilter("ignore")

JUST_SPIKES = True
TRAINED = True

ap = argparse.ArgumentParser()
ap.add_argument("--test_eid", type=str, default='51e53aff-1d5d-4182-a684-aba783d50ae5')
ap.add_argument("--mask_ratio", type=float, default=0.3)
ap.add_argument("--mask_mode", type=str, default="all")
ap.add_argument("--model_name", type=str, default="NDT1")
ap.add_argument("--prompting", type=str, default="False")
ap.add_argument("--train", type=str, default="True")
ap.add_argument("--eval", type=str, default="True")
ap.add_argument("--base_path", type=str, default='/work/hdd/beml/ac136')
ap.add_argument("--data-type", type=str, default="just_spikes")
ap.add_argument("--results-path", type=str, default="results_og_multi_session")
ap.add_argument("--output-model-name", type=str, default=None)
ap.add_argument("--num_train_sessions", type=int, default=1)
ap.add_argument('--use_dummy', action='store_true')
ap.add_argument('--model_path', type=str, default='/work/hdd/beml/ac136/training_og/train/num_session_1/model_NDT1/method_ssl/mask_temporal/stitch_True/5dcee0eb-b34d-4652-acc3-d10afc6eae68/model_best.pt')
args = ap.parse_args()

eid = args.test_eid
eid_name = eid[:-len("_aligned")] if eid.endswith("_aligned") else eid
source_model_name = Path(args.model_path).parent.parent.name
output_model_name = args.output_model_name or source_model_name
base_path = args.base_path
DATA_TYPE = args.data_type
RESULTS_PATH = args.results_path
model_acroynm = args.model_name.lower()
num_train_sessions = args.num_train_sessions
assert num_train_sessions > 0, 'num_train_sessions should be greater than 0.'

print()
print("Args to finetune_eval_multi_session: ")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")
print()

if args.prompting == "True":
    if args.model_name == 'NDT1':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_stitching_prompting.yaml"
        }
    elif args.model_name == 'NDT2':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_prompting.yaml"
        }
else:
    if args.model_name == 'NDT1':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}_stitching.yaml"
        }
    elif args.model_name == 'NDT2':
        kwargs = {
            "model": f"include:src/configs/{model_acroynm}.yaml"
        }

config = config_from_kwargs(kwargs)
config = update_config("src/configs/finetune_sessions_trainer.yaml", config)

set_seed(config.seed)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    split_batches=True,
    even_batches=False,
    kwargs_handlers=[ddp_kwargs],
)

# Shared variable to signal the dummy load to stop
stop_dummy_load = threading.Event()
if args.use_dummy:
    print("Running dummy load")
    # Run dummy load in a separate thread
    dummy_thread = threading.Thread(target=dummy_load, args=(stop_dummy_load,80000))
    dummy_thread.start()
try:
    if args.train == "True":
        print('Start model training.')
        print('=====================')
        train_dataset, val_dataset, test_dataset, meta_data = load_ibl_dataset_locally(
                            eid=None,
                            num_sessions=config.data.num_sessions,
                            split_method=config.data.split_method,
                            train_session_eid=[eid],
                            test_session_eid=config.data.test_session_eid,
                            batch_size=config.training.train_batch_size,
                            seed=config.seed,
                            just_spikes=JUST_SPIKES,
                            data_type=DATA_TYPE,
                            base_path=base_path)

        if TRAINED:
            log_dir = os.path.join(base_path, RESULTS_PATH, output_model_name, "finetune", eid_name)
        else:
            log_dir = os.path.join(base_path, RESULTS_PATH,
                                "finetune",
                                "num_session_{}".format(num_train_sessions),
                                "model_{}".format(config.model.model_class),
                                "method_{}".format(config.method.model_kwargs.method_name),
                                "mask_{}".format(args.mask_mode),
                                "stitch_{}".format(config.model.encoder.stitching),
                                "{}".format(eid)
                                )
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("Meta data: ")
        print(meta_data)
        print()

        if args.model_name in ["NDT1", "iTransformer"]:
            max_space_length = config.data.max_space_length
        elif args.model_name in ["NDT2", "STPatch"]:
            max_space_F = config.model.encoder.embedder.max_space_F
            max_num_neurons = max(meta_data['num_neurons'])
            max_space_length = ceil(max_num_neurons/max_space_F) * max_space_F
        else:
            max_space_length = config.data.max_space_length

        meta_data['max_space_length'] = max_space_length

        print('encoder max space length:', max_space_length)

        train_dataloader = make_loader(train_dataset,
                                target=config.data.target,
                                load_meta=False,
                                batch_size=config.training.train_batch_size,
                                pad_to_right=True,
                                pad_value=-1.,
                                max_time_length=config.data.max_time_length,
                                max_space_length=max_space_length,
                                dataset_name=config.data.dataset_name,
                                sort_by_depth=config.data.sort_by_depth,
                                sort_by_region=config.data.sort_by_region,
                                stitching=config.model.encoder.stitching,
                                shuffle=True)

        val_dataloader = make_loader(val_dataset,
                                target=config.data.target,
                                load_meta=False,
                                batch_size=config.training.test_batch_size,
                                pad_to_right=True,
                                pad_value=-1.,
                                max_time_length=config.data.max_time_length,
                                max_space_length=max_space_length,
                                dataset_name=config.data.dataset_name,
                                sort_by_depth=config.data.sort_by_depth,
                                sort_by_region=config.data.sort_by_region,
                                stitching=config.model.encoder.stitching,
                                shuffle=False)

        # load model
        NAME2MODEL = {"NDT1": NDT1, "STPatch": STPatch}

        config = update_config(config, meta_data)
        model_class = NAME2MODEL[config.model.model_class]
        model = model_class(config.model, **config.method.model_kwargs, **meta_data)

        # load pretrain model
        if args.mask_mode == 'temporal':
            mask_path = 'baseline'
        elif args.mask_mode == 'all':
            mask_path = 'MtM'

        if TRAINED:
            pretrain_model_path = args.model_path
        else:
            pretrain_model_path = f'{base_path}/models/ibl-foundation-model__multi-{args.model_name}-{mask_path}-{num_train_sessions}-sessions/model_best.pt'


        if TRAINED or num_train_sessions > 1:
            print('\nLoad pretrain model from:', pretrain_model_path)
            print()
            # load weights that can be found in the pretrain model
            if DATA_TYPE != "datasets":
                ckpt = torch.load(pretrain_model_path, map_location="cpu")['model'].state_dict()
                # Remove session embedding weights from checkpoint
                ckpt.pop("encoder.embedder.embed_session.weight", None)

                model.load_state_dict(ckpt, strict=False)
            else:
                model.load_state_dict(torch.load(pretrain_model_path, map_location="cpu")['model'].state_dict(), strict=False)
        else:
            print('Train from scratch.')

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd, eps=config.optimizer.eps)
        lr_scheduler = OneCycleLR(
            optimizer=optimizer,
            total_steps=config.training.num_epochs * len(train_dataloader) // config.optimizer.gradient_accumulation_steps,
            max_lr=config.optimizer.lr,
            pct_start=config.optimizer.warmup_pct,
            div_factor=config.optimizer.div_factor,
        )
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

        print(config)
        print()

        trainer_kwargs = {
            "log_dir": log_dir,
            "accelerator": accelerator,
            "lr_scheduler": lr_scheduler,
            "config": config,
            "stitching": config.model.encoder.stitching,
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

    #########################

    if args.eval == "True":
        accelerator.wait_for_everyone()

    if args.eval == "True":
        print('Start model evaluation.')
        print('=======================')

        mask_name = f"mask_{args.mask_mode}"
        if args.model_name == "NDT2":
            model_name = "STPatch"
        else:
            model_name = args.model_name

        n_time_steps = 100

        co_smooth = True
        forward_pred = True

        if JUST_SPIKES:
            inter_region = False
            intra_region = False
        else:
            inter_region = True
            intra_region = True

        choice_decoding = False
        continuous_decoding = False

        print("Mask name: ", mask_name)
        print()

        if args.prompting == "True":
            model_config = f"src/configs/{model_acroynm}_stitching_prompting_eval.yaml"
        else:
            model_config = f"src/configs/{model_acroynm}_stitching_eval.yaml"

        # load pretrain model
        if args.mask_mode == 'temporal':
            mask_path = 'baseline'
        elif args.mask_mode == 'all':
            mask_path = 'MtM'

        if TRAINED:
            finetune_model_path = os.path.join(base_path, RESULTS_PATH, output_model_name, "finetune", eid_name, "model_best.pt")
        else:
            finetune_model_path = f'{base_path}/{RESULTS_PATH}/finetune/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/model_best.pt'

        configs = {
            'model_config': model_config,
            'model_path': finetune_model_path,
            'trainer_config': f'src/configs/trainer_{model_acroynm}.yaml',
            'dataset_path': None,
            'test_size': 0.2,
            'seed': 42,
            'mask_name': mask_name,
            'eid': eid,
            'stitching': True,
            'num_sessions': 1,
            'just_spikes': JUST_SPIKES,
            'data_type': DATA_TYPE,
            'base_path': base_path
        }

        # load your model and dataloader
        model, accelerator, dataset, dataloader = load_model_data_local(**configs)

        is_aligned = False

        if TRAINED:
            save_path = os.path.join(base_path, RESULTS_PATH, output_model_name, "eval", eid_name)
        else:
            save_path = f'{base_path}/{RESULTS_PATH}/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}'

        # co-smoothing
        if co_smooth:
            print('Start co-smoothing:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name,
                'save_path': f'{save_path}/co_smooth',
                'mode': 'per_neuron',
                'n_time_steps': n_time_steps,
                'is_aligned': is_aligned,
                'target_regions': None,
                'n_jobs': 8,
                'data_type': DATA_TYPE
            }

            results = co_smoothing_eval(model,
                            accelerator,
                            dataloader,
                            dataset,
                            **co_smoothing_configs)
            print(results)

        # forward prediction
        if forward_pred:
            print('Start forward prediction:')
            results = co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [],
                'method_name': mask_name,
                'save_path': f'{save_path}/forward_pred',
                'mode': 'forward_pred',
                'n_time_steps': n_time_steps,
                'held_out_list': list(range(90, 100)), # NLB uses 200 ms for fp
                'is_aligned': is_aligned,
                'target_regions': None,
                'n_jobs': 8,
                'data_type': DATA_TYPE
            }

            results = co_smoothing_eval(model,
                            accelerator,
                            dataloader,
                            dataset,
                            **co_smoothing_configs)
            print(results)


        # inter-region
        if inter_region:
            print('Start inter-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name,
                'save_path': save_path,
                'mode': 'inter_region',
                'n_time_steps': n_time_steps,
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 8
            }

            results = co_smoothing_eval(model,
                            accelerator,
                            dataloader,
                            dataset,
                            **co_smoothing_configs)
            print(results)


        # intra-region
        if intra_region:
            print('Start intra-region:')
            co_smoothing_configs = {
                'subtract': 'task',
                'onset_alignment': [40],
                'method_name': mask_name,
                'save_path': save_path,
                'mode': 'intra_region',
                'n_time_steps': n_time_steps,
                'held_out_list': None,
                'is_aligned': True,
                'target_regions': ['all'],
                'n_jobs': 8
            }

            results = co_smoothing_eval(model,
                            accelerator,
                            dataloader,
                            dataset,
                            **co_smoothing_configs)
            print(results)


        if choice_decoding:
            print('Start choice_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/finetune/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/model_best.pt',
                'trainer_config': f'src/configs/ppwang/trainer_sl_choice_{model_acroynm}.yaml',
                'dataset_path': None,
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/choice_decoding',
                'test_size': 0.2,
                'seed': 42,
                'mask_name': mask_name,
                'metric': 'acc',
                'from_scratch': False,
                'freeze_encoder': True,
                'mask_ratio': args.mask_ratio,
                'eid': eid,
                'num_sessions': 1 ,
                'num_train_sessions': num_train_sessions,
                'use_logreg': True,
                'use_trial_filter': True,
            }
            results = behavior_decoding(**configs)
            print(results)


        if continuous_decoding:
            print('Start continuous_decoding:')
            configs = {
                'model_config': model_config,
                'model_path': f'{base_path}/results/finetune/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/model_best.pt',
                'trainer_config': f'src/configs/ppwang/trainer_sl_continuous_{model_acroynm}.yaml',
                'dataset_path': None,
                'save_path': f'{base_path}/results/eval/num_session_{num_train_sessions}/model_NDT1/method_ssl/{mask_name}/stitch_True/{eid}/continuous_decoding',
                'test_size': 0.2,
                'seed': 42,
                'mask_name': mask_name,
                'metric': 'rsquared',
                'from_scratch': False,
                'freeze_encoder': True,
                'mask_ratio': args.mask_ratio,
                'eid': eid,
                'num_sessions': 1,
                'num_train_sessions': num_train_sessions,
                'use_logreg': True,
                'use_trial_filter': True
            }
            results = behavior_decoding(**configs)
            print(results)

finally:
    if args.use_dummy:
        stop_dummy_load.set()
        dummy_thread.join()
    print('Finish model evaluation.')
    print('=======================')
    print('Finish model training.')
    print('=====================')
