import pytorch_lightning as pl
import torch
import torchio as tio
from model_training.losses import SegmentationCrossEntropyLoss
from model_training.optimizers import get_optimizer
from model_training.metrics import dice_coefficient, accuracy, jaccard_index, boundary_f1_score

class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = SegmentationCrossEntropyLoss()
        self.save_hyperparameters(config.__dict__)

    def forward(self, x):
        return self.model(x)

    def _prepare_binary(self, logits, y):
        # logits: (B, C, ...), y: (B, ...)
        # Convert to binary: 1 for target_label, 0 for background
        # Use only the background and target_label channels
        # If model outputs 14 channels, select [0, target_label]
        if logits.shape[1] > 2:
            idxs = [0, self.config.target_label]
            logits = logits[:, idxs, ...]
        y_bin = (y == self.config.target_label).long()
        return logits, y_bin

    def training_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.config.target_label is not None:
            logits, y = self._prepare_binary(logits, y)
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=logits.shape[1])
        iou = jaccard_index(logits, y, num_classes=logits.shape[1])
        bf1 = boundary_f1_score(logits, y, num_classes=logits.shape[1])
        acc = accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_bf1', bf1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.config.target_label is not None:
            logits, y = self._prepare_binary(logits, y)
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=logits.shape[1])
        iou = jaccard_index(logits, y, num_classes=logits.shape[1])
        bf1 = boundary_f1_score(logits, y, num_classes=logits.shape[1])
        acc = accuracy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_bf1', bf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if self.config.target_label is not None:
            logits, y = self._prepare_binary(logits, y)
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=logits.shape[1])
        iou = jaccard_index(logits, y, num_classes=logits.shape[1])
        bf1 = boundary_f1_score(logits, y, num_classes=logits.shape[1])
        acc = accuracy(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_bf1', bf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(self.config, self)
        if scheduler:
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": self.config.monitor_metric}
        return optimizer 