from pathlib import Path
from typing import List, Dict, Tuple, Union

import dgl
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from core.dataset import MalwareDataset


def stratified_split_dataset(samples: List[str],
                             labels: Dict[str, int],
                             ratios: Tuple[float, float]) -> Tuple[List[str], List[str]]:
    """
    Split the dataset into train and validation datasets based on the given ratio
    :param samples: List of file names
    :param labels: Mapping from file name to label
    :param ratios: Training ratio, validation ratio
    :return: List of file names in training and validation split
    """
    if sum(ratios) != 1:
        raise Exception("Invalid ratios provided")
    train_ratio, val_ratio = ratios
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=0)
    train_idx, val_idx = list(sss.split(samples, [labels[x] for x in samples]))[0]
    train_list = [samples[x] for x in train_idx]
    val_list = [samples[x] for x in val_idx]
    return train_list, val_list


@torch.no_grad()
def collate(samples: List[Tuple[dgl.DGLGraph, int]]) -> (dgl.DGLGraph, torch.Tensor):
    """
    Batches several graphs into one
    :param samples: Tuple containing graph and its label
    :return: Batched graph, and labels concatenated into a tensor
    """
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    return batched_graph, labels.float()


class MalwareDataModule(pl.LightningDataModule):
    """
    Handler class for data loading, splitting and initializing datasets and dataloaders.
    """

    def __init__(
            self,
            train_dir: Union[str, Path],
            test_dir: Union[str, Path],
            batch_size: int,
            split_ratios: Tuple[float, float],
            consider_features: List[str],
            num_workers: int,
            pin_memory: bool,
            split_train_val: bool,
    ):
        """
        Creates the MalwareDataModule
        :param train_dir: The directory containing the training samples
        :param test_dir: The directory containing the testing samples
        :param batch_size: Number of graphs in a batch
        :param split_ratios: Tuple containing training and validation split
        :param consider_features: Features types to consider
        :param num_workers: Number of processes to
        :param pin_memory: If True, said to be speeding up GPU data transfer
        :param split_train_val: If true, split the train dataset into train and validation,
                                else use test dataset for validation
        """
        super().__init__()
        self.train_dir = Path(train_dir)
        if not self.train_dir.exists():
            raise FileNotFoundError(f"Train directory {train_dir} does not exist. Could not read from it.")
        self.test_dir = Path(test_dir)
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Test directory {test_dir} does not exist. Could not read from it.")
        self.dataloader_kwargs = {
            'num_workers': num_workers,
            'batch_size': batch_size,
            'pin_memory': pin_memory,
            'collate_fn': collate,
            'drop_last': True
        }
        self.split_ratios = split_ratios
        self.split = split_train_val
        self.splitter = stratified_split_dataset
        self.consider_features = consider_features

    @staticmethod
    def get_samples(path: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """
        Get samples and labels from the given path
        :param path: The directory containing graphs
        :return: The file list, and their label mapping
        """
        base_path = Path(path)
        if not base_path.exists():
            raise FileNotFoundError(f'{base_path} does not exist')
        apk_list = sorted([x for x in base_path.iterdir()])
        samples = []
        labels = {}
        for apk in apk_list:
            samples.append(apk.name)
            labels[apk.name] = int("Benig" not in apk.name)
        return samples, labels

    def setup(self, stage=None):
        samples, labels = self.get_samples(self.train_dir)
        test_samples, test_labels = self.get_samples(self.test_dir)
        if self.split:
            train_samples, val_samples = self.splitter(samples, labels, self.split_ratios)
            val_dir = self.train_dir
            val_labels = labels
        else:
            train_samples = samples
            val_dir = self.test_dir
            val_samples, val_labels = test_samples, test_labels
        self.train_dataset = MalwareDataset(
            source_dir=self.train_dir,
            samples=train_samples,
            labels=labels,
            consider_features=self.consider_features
        )
        self.val_dataset = MalwareDataset(
            source_dir=val_dir,
            samples=val_samples,
            labels=val_labels,
            consider_features=self.consider_features
        )
        self.test_dataset = MalwareDataset(
            source_dir=self.test_dir,
            samples=test_samples,
            labels=test_labels,
            consider_features=self.consider_features
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.dataloader_kwargs)
