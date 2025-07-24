import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class SegmentationTrainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.logger = TensorBoardLogger(
            save_dir=config.log_dir,
            name=config.project_name,
            version=config.run_name
        )

        self.checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            save_top_k=1,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            filename="{epoch}-{val_loss:.4f}",
            save_last=True
        )

        self.early_stopping = EarlyStopping(
            monitor=config.monitor_metric,
            patience=config.early_stopping_patience,
            mode=config.monitor_mode,
            verbose=True
        )

        self.trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            accelerator="gpu" if config.gpus > 0 else "cpu",
            devices=config.gpus if config.gpus > 0 else 1,
            precision=config.precision,
            logger=self.logger,
            callbacks=[self.checkpoint_callback, self.early_stopping],
            log_every_n_steps=config.log_every_n_steps,
            deterministic=False,
            enable_progress_bar=True
        )

    def fit(self):
        if self.config.resume_from_checkpoint:
            self.trainer.fit(self.model, self.train_loader, self.val_loader, ckpt_path=self.config.resume_from_checkpoint)
        else:
            self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def test(self):
        if self.test_loader is not None:
            if self.config.resume_from_checkpoint:
                self.trainer.test(self.model, self.test_loader, ckpt_path=self.config.resume_from_checkpoint)
            else:
                self.trainer.test(self.model, self.test_loader) 