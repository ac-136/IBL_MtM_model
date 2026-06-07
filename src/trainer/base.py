import torch
import numpy as np
# import wandb
import os
import time
import json
from utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm
import random

class Trainer():
    def __init__(
            self,
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            **kwargs
    ):
        ### Add method to save loss ###
        self.train_losses = []
        self.eval_losses = []

        # get all the arguments
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer

        # get arguments from kwargs if they exist
        self.log_dir = kwargs.get("log_dir", None)
        self.accelerator = kwargs.get("accelerator", None)
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.config = kwargs.get("config", None)
        self.stitching = kwargs.get("stitching", None)
        self.num_neurons = kwargs.get("num_neurons", None)
        
        self.just_spikes = kwargs.get("just_spikes", False)

        self.model_class = self.config.model.model_class

        if self.config.method.model_kwargs.clf:
            self.metric = 'acc'
        elif self.config.method.model_kwargs.reg:
            self.metric = 'rsquared'
        else:
            self.metric = 'r2'
                
        self.session_active_neurons = []

        self.masking_ratio = model.encoder.masker.ratio
        self.masking_mode = model.encoder.masker.mode
        self.masking_schemes = ['neuron', 'causal']
        if self.masking_mode == "all":
            if not self.just_spikes:
                self.masking_schemes += ['intra-region', 'inter-region']

        early_stopping_cfg = getattr(self.config.training, "early_stopping", None)
        self.early_stopping_enabled = bool(getattr(early_stopping_cfg, "enabled", False))
        self.early_stopping_patience = int(getattr(early_stopping_cfg, "patience", 0))
        self.early_stopping_min_delta = float(getattr(early_stopping_cfg, "min_delta", 0.0))
        self.early_stopping_monitor = getattr(
            early_stopping_cfg,
            "monitor",
            f"eval_trial_avg_{self.metric}",
        )
        self.early_stopping_mode = getattr(early_stopping_cfg, "mode", "max")

        if self.masking_mode in ["combined", "all"]:
            print("(train) switch between masking modes: ", self.masking_schemes)

    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
        best_monitored_value = None
        epochs_without_improvement = 0
        epoch_durations = []
        gpu_count = int(os.environ.get("GPU_BENCHMARK_GPUS", 1))
        train_start = time.time()
        # train loop
        for epoch in range(self.config.training.num_epochs):
            epoch_start = time.time()
            train_epoch_results = self.train_epoch(epoch)
            eval_epoch_results = self.eval_epoch()
            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)
            epoch_gpu_hours = epoch_duration * gpu_count / 3600.0
            print(f"epoch: {epoch} train loss: {train_epoch_results['train_loss']} duration: {epoch_duration:.2f}s gpu_hours: {epoch_gpu_hours:.4f}")

            ### Save loss ###
            self.train_losses.append(
                train_epoch_results["train_loss"].item() 
                if isinstance(train_epoch_results["train_loss"], torch.Tensor) 
                else train_epoch_results["train_loss"]
            )

            self.eval_losses.append(
                eval_epoch_results["eval_loss"].item() 
                if isinstance(eval_epoch_results["eval_loss"], torch.Tensor) 
                else eval_epoch_results["eval_loss"]
            )

            if eval_epoch_results:
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                # if eval_epoch_results[f'eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    print(f"epoch: {epoch} best eval loss: {best_eval_loss}")
                    print(f"epoch: {epoch} best eval trial avg {self.metric}: {best_eval_trial_avg_metric}")
                    # save model
                    self.save_model(name="best", epoch=epoch)
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        gt_pred_fig = self.plot_epoch(
                            gt=eval_epoch_results['eval_gt'][0], 
                            preds=eval_epoch_results['eval_preds'][0], epoch=epoch,
                            active_neurons=self.session_active_neurons[0][:5]
                        )

                        # if self.config.wandb.use:
                        #     wandb.log({"best_epoch": epoch,
                        #             "best_gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                        #             "best_r2_fig": wandb.Image(gt_pred_fig['plot_r2'])})

                        # else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"best_gt_pred_fig_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"best_r2_fig_{epoch}.png")
                        )

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

                if self.early_stopping_enabled:
                    monitored_value = eval_epoch_results[self.early_stopping_monitor]
                    improved = self._is_improvement(monitored_value, best_monitored_value)

                    if improved:
                        best_monitored_value = monitored_value
                        epochs_without_improvement = 0
                        print(
                            f"epoch: {epoch} early stopping monitor improved "
                            f"({self.early_stopping_monitor}={monitored_value})"
                        )
                    else:
                        epochs_without_improvement += 1
                        print(
                            f"epoch: {epoch} no early stopping improvement for "
                            f"{epochs_without_improvement} epoch(s)"
                        )

            # save model by epoch
            if epoch % self.config.training.save_every == 0:
                self.save_model(name="epoch", epoch=epoch)

            # plot epoch
            if epoch % self.config.training.save_plot_every_n_epochs == 0:
                if self.config.method.model_kwargs.method_name == 'ssl':

                    gt_pred_fig = self.plot_epoch(
                        gt=eval_epoch_results['eval_gt'][0], 
                        preds=eval_epoch_results['eval_preds'][0], 
                        epoch=epoch,
                        active_neurons=self.session_active_neurons[0][:5]
                    )
                    # if self.config.wandb.use:
                    #     wandb.log({
                    #         "gt_pred_fig": wandb.Image(gt_pred_fig['plot_gt_pred']),
                    #         "r2_fig": wandb.Image(gt_pred_fig['plot_r2'])
                    #     })
                    # else:
                    gt_pred_fig['plot_gt_pred'].savefig(
                        os.path.join(self.log_dir, f"gt_pred_fig_{epoch}.png")
                    )
                    gt_pred_fig['plot_r2'].savefig(
                        os.path.join(self.log_dir, f"r2_fig_{epoch}.png")
                    )

            if (
                self.early_stopping_enabled
                and eval_epoch_results
                and epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.early_stopping_monitor}: {best_monitored_value}"
                )
                break

            # # wandb log
            # if self.config.wandb.use:
            #     wandb.log({
            #         "train_loss": train_epoch_results['train_loss'],
            #         "eval_loss": eval_epoch_results['eval_loss'],
            #         f"eval_trial_avg_{self.metric}": eval_epoch_results[f'eval_trial_avg_{self.metric}']
            #     })
                
        # save last model
        self.save_model(name="last", epoch=epoch)
        total_duration = time.time() - train_start
        total_gpu_hours = total_duration * gpu_count / 3600.0
        benchmark = {
            "num_epochs_completed": len(epoch_durations),
            "total_duration_s": total_duration,
            "total_gpu_count": gpu_count,
            "total_gpu_hours": total_gpu_hours,
            "avg_gpu_hours_per_epoch": total_gpu_hours / len(epoch_durations) if len(epoch_durations) > 0 else 0.0,
            "epoch_durations_s": epoch_durations,
            "epoch_gpu_hours": [d * gpu_count / 3600.0 for d in epoch_durations],
        }
        benchmark_path = os.path.join(self.log_dir, "benchmark.json")
        with open(benchmark_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        print(f"Saved GPU benchmark to {benchmark_path}")
        
        # if self.config.wandb.use:
        #     wandb.log({"best_eval_loss": best_eval_loss,
        #                f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric})

        ### Save loss as npz ###
        loss_save_path = os.path.join(self.log_dir, "loss")
        os.makedirs(loss_save_path, exist_ok=True)
        np.savez(
            os.path.join(loss_save_path, "loss_history.npz"),
            train_loss=np.array(self.train_losses),
            eval_loss=np.array(self.eval_losses),
        )

    def _is_improvement(self, current_value, best_value):
        if best_value is None:
            return True

        if self.early_stopping_mode == "min":
            return current_value < (best_value - self.early_stopping_min_delta)

        return current_value > (best_value + self.early_stopping_min_delta)

            
    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in tqdm(self.train_dataloader):
            if self.masking_mode in ["combined", "all"]:
                masking_mode = random.sample(self.masking_schemes, 1)[0]
                if masking_mode == 'temporal':
                    self.model.encoder.masker.ratio = 0.3
                elif masking_mode == 'causal':
                    self.model.encoder.masker.ratio = 0.6
                else:
                    self.model.encoder.masker.ratio = self.masking_ratio
            else:
                masking_mode = self.masking_mode
            # print(f"masking: {masking_mode}")
            outputs = self._forward_model_outputs(batch, masking_mode)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            train_loss += loss.item()
            train_examples += outputs.n_examples
        return{
            "train_loss": train_loss/train_examples
        }
    
    def _forward_model_outputs(self, batch, masking_mode):
        batch = move_batch_to_device(batch, self.accelerator.device)
        return self.model(
            batch['spikes_data'], 
            time_attn_mask=batch['time_attn_mask'],
            space_attn_mask=batch['space_attn_mask'],
            spikes_timestamps=batch['spikes_timestamps'], 
            spikes_spacestamps=batch['spikes_spacestamps'], 
            targets = batch['target'],
            neuron_regions=batch['neuron_regions'],
            masking_mode=masking_mode, 
            spike_augmentation=self.config.data.spike_augmentation,
            num_neuron=batch['spikes_data'].shape[2],
            eid=batch['eid'][0]  # each batch consists of data from the same eid
        ) 
    
    def eval_epoch(self):
        self.model.eval()
        eval_loss = 0.
        eval_examples = 0
        session_results = {}
        for num_neuron in self.num_neurons:
            session_results[num_neuron] = {
                "gt": [],
                "preds": []
            }
        if self.eval_dataloader:
            gt, preds = [], []
            with torch.no_grad():  
                for batch in self.eval_dataloader:
                    if self.masking_mode in ["combined", "all"]:
                        masking_mode = random.sample(self.masking_schemes, 1)[0]
                        if masking_mode == 'temporal':
                            self.model.encoder.masker.ratio = 0.3
                        elif masking_mode == 'causal':
                            self.model.encoder.masker.ratio = 0.6
                        else:
                            self.model.encoder.masker.ratio = self.masking_ratio
                    else:
                        masking_mode = self.masking_mode
                    outputs = self._forward_model_outputs(batch, masking_mode)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_examples += outputs.n_examples
                    if self.model_class in ['NDT1', 'iTransformer']:
                        num_neuron = batch['spikes_data'].shape[2]
                    elif self.model_class in ['NDT2', 'STPatch']:
                        num_neuron = outputs.num_neuron
                    if self.config.method.model_kwargs.method_name == 'ssl':
                        session_results[num_neuron]["gt"].append(outputs.targets.clone()[:,:,:num_neuron])
                        session_results[num_neuron]["preds"].append(outputs.preds.clone()[:,:,:num_neuron])
                    else:
                        session_results[num_neuron]["gt"].append(outputs.targets.clone())
                        session_results[num_neuron]["preds"].append(outputs.preds.clone())
                    
            results_list = []
            for idx, num_neuron in enumerate(self.num_neurons):
                if len(session_results[num_neuron]["gt"]) == 0:
                    print(
                        f"Skipping eval metrics for num_neuron={num_neuron} "
                        "because no validation batches were collected for this bucket."
                    )
                    continue
                _gt = torch.cat(session_results[num_neuron]["gt"], dim=0)
                _preds = torch.cat(session_results[num_neuron]["preds"], dim=0)

                if self.config.method.model_kwargs.loss == "poisson_nll":
                    _preds = torch.exp(_preds)
                elif self.config.method.model_kwargs.loss == "cross_entropy" :
                    _preds = torch.nn.functional.softmax(_preds, dim=1)
                gt.append(_gt)
                preds.append(_preds)
                result_idx = len(gt) - 1

                if len(self.session_active_neurons) < len(self.num_neurons):
                    active_neurons = np.argsort(gt[result_idx].cpu().numpy().sum((0,1)))[::-1][:50].tolist()
                    self.session_active_neurons.append(active_neurons)
                if self.config.method.model_kwargs.method_name == 'ssl':
                    results = metrics_list(gt = gt[result_idx][:,:,self.session_active_neurons[result_idx]].transpose(-1,0),
                                        pred = preds[result_idx][:,:,self.session_active_neurons[result_idx]].transpose(-1,0), 
                                        metrics=["r2"], 
                                        device=self.accelerator.device)
                    
                elif self.config.method.model_kwargs.method_name == 'sl':
                    if self.config.method.model_kwargs.clf:
                        results = metrics_list(gt = gt[result_idx].argmax(1),
                                            pred = preds[result_idx].argmax(1), 
                                            metrics=[self.metric], 
                                            device=self.accelerator.device)
                    elif self.config.method.model_kwargs.reg:
                        results = metrics_list(gt = gt[result_idx],
                                            pred = preds[result_idx],
                                            metrics=[self.metric],
                                            device=self.accelerator.device)
                results_list.append(results[self.metric])

        return {
            "eval_loss": eval_loss/eval_examples,
            f"eval_trial_avg_{self.metric}": np.mean(results_list) if len(results_list) > 0 else np.nan,
            "eval_gt": gt,
            "eval_preds": preds,
        }
    
    def plot_epoch(self, gt, preds, epoch, active_neurons):
        gt_pred_fig = plot_gt_pred(gt = gt.mean(0).T.cpu().numpy(),
                    pred = preds.mean(0).T.detach().cpu().numpy(),
                    epoch = epoch)
        
        r2_fig = plot_neurons_r2(gt = gt.mean(0),
                pred = preds.mean(0),
                neuron_idx=active_neurons,
                epoch = epoch)
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }
        

    def save_model(self, name="last", epoch=0):
        # save model
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))
        
