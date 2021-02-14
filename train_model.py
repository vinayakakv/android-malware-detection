import os
from pathlib import Path

import hydra
import wandb
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from core.callbacks import InputMonitor, BestModelTagger, MetricsLogger
from core.data_module import MalwareDataModule
from core.model import MalwareDetector


@hydra.main(config_path="config", config_name="conf")
def train_model(cfg: DictConfig) -> None:
    data_module = MalwareDataModule(**cfg['data'])

    model = MalwareDetector(**cfg['model'])

    callbacks = [ModelCheckpoint(
        dirpath=os.getcwd(),
        filename=str('{epoch:02d}-{val_loss:.2f}.pt'),
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_top_k=-1
    )]

    trainer_kwargs = dict(cfg['trainer'])
    force_retrain = cfg.get('force_retrain', False)
    if Path('last.ckpt').exists() and not force_retrain:
        trainer_kwargs['resume_from_checkpoint'] = 'last.ckpt'

    if 'logger' in cfg:
        # We use WandB logger
        logger = WandbLogger(
            **cfg['logger']['args'],
            tags=[f'testing' if "testing" in cfg else "training"]
        )
        if "testing" in cfg:
            logger.experiment.summary["test_type"] = cfg["testing"]
        logger.watch(model)
        logger.log_hyperparams(cfg['logger']['hparams'])
        if logger:
            trainer_kwargs['logger'] = logger
            callbacks.append(InputMonitor())
            callbacks.append(BestModelTagger(monitor='val_loss', mode='min'))
            callbacks.append(MetricsLogger(stages='all'))

    trainer = Trainer(
        callbacks=callbacks,
        **trainer_kwargs
    )
    testing = cfg.get('testing', '')
    if not testing:
        trainer.fit(model, datamodule=data_module)
    else:
        if testing not in ['last', 'best'] and 'epoch' not in testing:
            raise ValueError(f"testing must be one of 'best' or 'last' or 'epoch=N'. It is {testing}")
        elif 'epoch' in testing:
            # epoch in testing
            epoch = testing.split('@')[1]
            checkpoints = list(Path(os.getcwd()).glob(f"epoch={epoch}*.ckpt"))
            if len(checkpoints) < 0:
                print(f"Checkpoint at epoch = {epoch} not found.")
            assert len(checkpoints) == 1, f"Multiple checkpoints corresponding to epoch = {epoch} found."
            ckpt_path = checkpoints[0]
        else:
            if not Path('last.ckpt').exists():
                raise FileNotFoundError("No last.ckpt exists. Could not do any testing.")
            if testing == 'last':
                ckpt_path = 'last.ckpt'
            else:
                # best
                last_checkpoint = torch.load('last.ckpt')
                ckpt_path = last_checkpoint['callbacks'][ModelCheckpoint]['best_model_path']
        print(f"Using checkpoint {ckpt_path} for testing.")
        model = MalwareDetector.load_from_checkpoint(ckpt_path, **cfg['model'])
        trainer.test(model, datamodule=data_module, verbose=True)
    wandb.finish()


if __name__ == '__main__':
    train_model()
