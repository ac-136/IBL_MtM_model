import torch
import numpy as np
import os
import time
import json
from ..utils.utils import move_batch_to_device, metrics_list, plot_gt_pred, plot_neurons_r2
from tqdm import tqdm

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
        # Tier 3a run recording. Strictly observational: every value below is
        # already computed by the training loop and was being printed to stdout
        # and discarded, so the number to plot against scale survived only in
        # the Slurm logs. Nothing here changes model selection or scheduling.
        self.eval_metrics = []
        self.learning_rates = []
        self.best_epoch = None
        self.best_metric = None

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
        # 3a: already passed in via **meta_data by every CLI, just never kept.
        # Recording which sessions a run actually used is what makes a scaling
        # point reproducible; `eids` is a set, so it is sorted at write time.
        self.eids = kwargs.get("eids", None)
        self.num_sessions = kwargs.get("num_sessions", None)
        
        if self.config.method.model_kwargs.clf:
            self.metric = 'acc'
        elif self.config.method.model_kwargs.reg:
            self.metric = 'rsquared'
        else:
            self.metric = 'r2'
                
        self.session_active_neurons = []

        self.masking_mode = model.encoder.masker.mode

    def train(self):
        best_eval_loss = torch.tensor(float('inf'))
        best_eval_trial_avg_metric = -torch.tensor(float('inf'))
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

            # 3a: the quality metric and the LR, both already computed above.
            # Reading them cannot perturb the RNG stream; the scheduler is
            # stepped per batch inside train_epoch and is only *read* here.
            _metric = eval_epoch_results[f"eval_trial_avg_{self.metric}"]
            self.eval_metrics.append(
                _metric.item() if isinstance(_metric, torch.Tensor) else _metric
            )
            self.learning_rates.append(
                self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None
                else self.optimizer.param_groups[0]["lr"]
            )

            if eval_epoch_results:
                if eval_epoch_results[f'eval_trial_avg_{self.metric}'] > best_eval_trial_avg_metric:
                # if eval_epoch_results[f'eval_loss'] < best_eval_loss:
                    best_eval_loss = eval_epoch_results[f'eval_loss']
                    best_eval_trial_avg_metric = eval_epoch_results[f'eval_trial_avg_{self.metric}']
                    # 3a: which epoch model_best.pt actually came from, which was
                    # otherwise recoverable only from stdout. The selection
                    # condition above is deliberately untouched.
                    self.best_epoch = epoch
                    self.best_metric = (
                        best_eval_trial_avg_metric.item()
                        if isinstance(best_eval_trial_avg_metric, torch.Tensor)
                        else best_eval_trial_avg_metric
                    )
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


                        # else:
                        gt_pred_fig['plot_gt_pred'].savefig(
                            os.path.join(self.log_dir, f"best_gt_pred_fig_{epoch}.png")
                        )
                        gt_pred_fig['plot_r2'].savefig(
                            os.path.join(self.log_dir, f"best_r2_fig_{epoch}.png")
                        )

                print(f"epoch: {epoch} eval loss: {eval_epoch_results['eval_loss']} {self.metric}: {eval_epoch_results[f'eval_trial_avg_{self.metric}']}")

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
                    # else:
                    gt_pred_fig['plot_gt_pred'].savefig(
                        os.path.join(self.log_dir, f"gt_pred_fig_{epoch}.png")
                    )
                    gt_pred_fig['plot_r2'].savefig(
                        os.path.join(self.log_dir, f"r2_fig_{epoch}.png")
                    )

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
        
        #                f"best_eval_trial_avg_{self.metric}": best_eval_trial_avg_metric})

        ### Save loss as npz ###
        loss_save_path = os.path.join(self.log_dir, "loss")
        os.makedirs(loss_save_path, exist_ok=True)
        np.savez(
            os.path.join(loss_save_path, "loss_history.npz"),
            train_loss=np.array(self.train_losses),
            eval_loss=np.array(self.eval_losses),
            # 3a: additive only. The two arrays above keep their exact previous
            # values, which is what the Step 0 logging-parity gate checks.
            eval_metric=np.array(self.eval_metrics),
            lr=np.array(self.learning_rates),
        )

        ### 3a: per-run facts, so the number to plot does not live in Slurm logs ###
        self.save_run_summary()


    def train_epoch(self, epoch):
        train_loss = 0.
        train_examples = 0
        self.model.train()
        for batch in tqdm(self.train_dataloader):
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
                    masking_mode = self.masking_mode
                    outputs = self._forward_model_outputs(batch, masking_mode)
                    loss = outputs.loss
                    eval_loss += loss.item()
                    eval_examples += outputs.n_examples
                    num_neuron = batch['spikes_data'].shape[2]
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
        

    def save_run_summary(self):
        """Write the per-run facts needed to plot a scaling point.

        Tier 3a, and strictly observational: every value is either already
        computed by the training loop, already passed in via **meta_data, or a
        pure read of model/optimizer state. Nothing here is computed for the
        first time and nothing is fed back into training, so this cannot move
        the RNG stream. The Step 0 logging-parity gate is what verifies that.
        """
        try:
            n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.model.parameters())
        except Exception:
            n_trainable = n_total = None

        # meta_data["eids"] is a set, and set iteration order over strings varies
        # between processes unless PYTHONHASHSEED is fixed. Sorting makes the
        # recorded session list comparable across runs.
        eids = sorted(self.eids) if self.eids is not None else None

        summary = {
            "num_epochs_configured": self.config.training.num_epochs,
            "num_epochs_recorded": len(self.train_losses),
            "metric_name": self.metric,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "final_train_loss": self.train_losses[-1] if self.train_losses else None,
            "final_eval_loss": self.eval_losses[-1] if self.eval_losses else None,
            "train_loss": self.train_losses,
            "eval_loss": self.eval_losses,
            "eval_metric": self.eval_metrics,
            "lr": self.learning_rates,
            "num_trainable_params": n_trainable,
            "num_total_params": n_total,
            "num_sessions": self.num_sessions,
            "num_neurons": self.num_neurons,
            "eids": eids,
            "stitching": self.stitching,
            "masking_mode": self.masking_mode,
            "seed": self.config.get("seed", None),
        }
        path = os.path.join(self.log_dir, "run_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved run summary to {path}")

        # The resolved config, so a run can be reconstructed without guessing
        # which YAML defaults were in force when it was launched.
        try:
            from omegaconf import OmegaConf
            cfg_path = os.path.join(self.log_dir, "resolved_config.yaml")
            with open(cfg_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.config))
            print(f"Saved resolved config to {cfg_path}")
        except Exception as e:
            print(f"could not save resolved config: {type(e).__name__}: {e}")

    def save_model(self, name="last", epoch=0):
        # save model
        print(f"saving model: {name} to {self.log_dir}")
        dict_config = {
            "model": self.model,
            "epoch": epoch,
        }
        torch.save(dict_config, os.path.join(self.log_dir, f"model_{name}.pt"))
        
