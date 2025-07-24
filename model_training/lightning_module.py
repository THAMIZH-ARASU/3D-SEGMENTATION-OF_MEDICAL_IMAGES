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

    def training_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA].squeeze(1).long()
        logits = self(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=self.config.num_classes)
        iou = jaccard_index(logits, y, num_classes=self.config.num_classes)
        bf1 = boundary_f1_score(logits, y, num_classes=self.config.num_classes)
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
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=self.config.num_classes)
        iou = jaccard_index(logits, y, num_classes=self.config.num_classes)
        bf1 = boundary_f1_score(logits, y, num_classes=self.config.num_classes)
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
        loss = self.loss_fn(logits, y)
        dice = dice_coefficient(logits, y, num_classes=self.config.num_classes)
        iou = jaccard_index(logits, y, num_classes=self.config.num_classes)
        bf1 = boundary_f1_score(logits, y, num_classes=self.config.num_classes)
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