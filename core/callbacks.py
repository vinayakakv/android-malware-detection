import numbers
from typing import Tuple, List, Union

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch.nn
import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.metrics.metric import Metric

from core.model import MalwareDetector
from core.utils import plot_confusion_matrix, plot_curve


class InputMonitor(Callback):
    """
    Plots the histogram of input labels
    """

    def __init__(self):
        pass

    def on_train_batch_start(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            batch: Tuple[dgl.DGLHeteroGraph, torch.Tensor],
            batch_idx: int,
            dataloader_idx: int
    ):
        samples, labels = batch
        trainer.logger.experiment.log({
            'train_data_histogram': wandb.Histogram(labels.detach().cpu().numpy())
        }, commit=False)


class BestModelTagger(Callback):
    """
    Logs the "best_epoch" and the metric value corresponding to that to the logger
    Inspired from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/callbacks/early_stopping.py
    """

    def __init__(self, monitor: str = 'val_loss', mode: str = 'min'):
        self.monitor = monitor
        if mode not in ['min', 'max']:
            raise ValueError(f"Invalid mode {mode}. Must be one of 'min' or 'max'")
        self.mode = mode
        self.monitor_op = torch.lt if mode == 'min' else torch.gt
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logs = trainer.callback_metrics
        monitor_val = logs.get(self.monitor)
        if monitor_val is None:
            raise RuntimeError(f"{self.monitor} was supposed to be logged from model. Could not find that")
        if monitor_val is not None:
            if isinstance(monitor_val, Metric):
                monitor_val = monitor_val.compute()
            elif isinstance(monitor_val, numbers.Number):
                monitor_val = torch.tensor(monitor_val, device=pl_module.device, dtype=torch.float)
        if self.monitor_op(monitor_val, self.best_score):
            self.best_score = monitor_val
            trainer.logger.experiment.log({
                f'{self.mode}_{self.monitor}': monitor_val.cpu().numpy(),
                'best_epoch': trainer.current_epoch
            }, commit=False)


class MetricsLogger(Callback):

    def __init__(self, stages: Union[List[str], str]):
        valid_stages = {'train', 'val', 'test'}
        if stages == 'all':
            self.stages = valid_stages
        else:
            for stage in stages:
                if stage not in valid_stages:
                    raise ValueError(f"Stage {stage} is not valid. Must be one of {valid_stages}")
            self.stages = set(stages) & valid_stages

    @staticmethod
    def _plot_metrics(trainer: pl.Trainer, pl_module: MalwareDetector, stage: str):
        confusion_matrix = pl_module.test_outputs['confusion_matrix'].compute().cpu().numpy()
        plot_confusion_matrix(
            confusion_matrix,
            group_names=['TN', 'FP', 'FN', 'TP'],
            categories=['Benign', 'Malware'],
            cmap='binary'
        )
        trainer.logger.experiment.log({
            f'{stage}_confusion_matrix': wandb.Image(plt)
        }, commit=False)
        if stage != 'test':
            return
        roc = pl_module.test_outputs['roc'].compute()
        figure = plot_curve(roc[0].cpu(), roc[1].cpu(), 'roc')
        trainer.logger.experiment.log({
            f'ROC': figure
        }, commit=False)
        prc = pl_module.test_outputs['prc'].compute()
        figure = plot_curve(prc[1].cpu(), prc[0].cpu(), 'prc')
        trainer.logger.experiment.log({
            f'PRC': figure
        }, commit=False)

    @staticmethod
    def compute_metrics(pl_module: MalwareDetector, stage: str):
        metrics = {}
        if stage == 'train':
            metric_dict = pl_module.train_metrics
        elif stage == 'val':
            metric_dict = pl_module.val_metrics
        elif stage == 'test':
            metric_dict = pl_module.test_metrics
        else:
            raise ValueError(f"Invalid stage: {stage}")
        for metric_name, metric in metric_dict.items():
            metrics[metric_name] = metric.compute()
        return metrics

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: MalwareDetector, outputs):
        if 'train' not in self.stages:
            return
        trainer.logger.experiment.log(self.compute_metrics(pl_module, 'train'), commit=False)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: MalwareDetector):
        if 'val' not in self.stages or trainer.running_sanity_check:
            return
        trainer.logger.experiment.log(self.compute_metrics(pl_module, 'val'), commit=False)

    def on_test_end(self, trainer: pl.Trainer, pl_module: MalwareDetector):
        if 'test' not in self.stages:
            return
        trainer.logger.experiment.log(self.compute_metrics(pl_module, 'test'), commit=False)
        self._plot_metrics(trainer, pl_module, 'test')
