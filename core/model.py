from typing import Mapping
from typing import Tuple, Optional, Dict

import dgl
import dgl.nn.pytorch as graph_nn
import pytorch_lightning as pl
import pytorch_lightning.metrics as metrics
import torch
import torch.nn.functional as F
from dgl.nn import Sequential
from pytorch_lightning.metrics import Metric
from torch import nn


class MalwareDetector(pl.LightningModule):
    def __init__(
            self,
            input_dimension: int,
            convolution_algorithm: str,
            convolution_count: int,
    ):
        super().__init__()
        supported_algorithms = ['GraphConv', 'SAGEConv', 'TAGConv', 'DotGatConv']
        if convolution_algorithm not in supported_algorithms:
            raise ValueError(
                f"{convolution_algorithm} is not supported. Supported algorithms are {supported_algorithms}")
        self.save_hyperparameters()
        self.convolution_layers = []
        convolution_dimensions = [64, 32, 16]
        for dimension in convolution_dimensions[:convolution_count]:
            self.convolution_layers.append(self._get_convolution_layer(
                name=convolution_algorithm,
                input_dimension=input_dimension,
                output_dimension=dimension
            ))
            input_dimension = dimension
        self.convolution_layers = Sequential(*self.convolution_layers)
        self.last_dimension = input_dimension
        self.classify = nn.Linear(input_dimension, 1)
        # Metrics
        self.loss_func = nn.BCEWithLogitsLoss()
        self.train_metrics = self._get_metric_dict('train')
        self.val_metrics = self._get_metric_dict('val')
        self.test_metrics = self._get_metric_dict('test')
        self.test_outputs = nn.ModuleDict({
            'confusion_matrix': metrics.ConfusionMatrix(num_classes=2),
            'prc': metrics.PrecisionRecallCurve(compute_on_step=False),
            'roc': metrics.ROC(compute_on_step=False)
        })

    @staticmethod
    def _get_convolution_layer(
            name: str,
            input_dimension: int,
            output_dimension: int
    ) -> Optional[nn.Module]:
        return {
            "GraphConv": graph_nn.GraphConv(
                input_dimension,
                output_dimension,
                activation=F.relu
            ),
            "SAGEConv": graph_nn.SAGEConv(
                input_dimension,
                output_dimension,
                activation=F.relu,
                aggregator_type='mean',
                norm=F.normalize
            ),
            "DotGatConv": graph_nn.DotGatConv(
                input_dimension,
                output_dimension,
                num_heads=1
            ),
            "TAGConv": graph_nn.TAGConv(
                input_dimension,
                output_dimension,
                k=4
            )
        }.get(name, None)

    @staticmethod
    def _get_metric_dict(stage: str) -> Mapping[str, Metric]:
        return nn.ModuleDict({
            f'{stage}_accuracy': metrics.Accuracy(),
            f'{stage}_precision': metrics.Precision(num_classes=1),
            f'{stage}_recall': metrics.Recall(num_classes=1),
            f'{stage}_f1': metrics.FBeta(num_classes=1)
        })

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        with g.local_scope():
            h = g.ndata['features']
            h = self.convolution_layers(g, h)
            g.ndata['h'] = h if len(self.convolution_layers) > 0 else h[0]
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg).squeeze()

    def training_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int) -> torch.Tensor:
        bg, label = batch
        logits = self.forward(bg)
        loss = self.loss_func(logits, label)
        prediction = torch.sigmoid(logits)
        for metric_name, metric in self.train_metrics.items():
            metric.update(prediction, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int):
        bg, label = batch
        logits = self.forward(bg)
        loss = self.loss_func(logits, label)
        prediction = torch.sigmoid(logits)
        for metric_name, metric in self.val_metrics.items():
            metric.update(prediction, label)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], batch_idx: int):
        bg, label = batch
        logits = self.forward(bg)
        prediction = torch.sigmoid(logits)
        loss = self.loss_func(logits, label)
        for metric_name, metric in self.test_metrics.items():
            metric.update(prediction, label)
        for metric_name, metric in self.test_outputs.items():
            metric.update(prediction, label)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
