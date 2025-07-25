import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from model_training.metrics import dice_coefficient, jaccard_index, boundary_f1_score, accuracy
import torchio as tio

class SegFormerLightningModule(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-4, model_name="nvidia/segformer-b0-finetuned-ade-512-512", class_weights=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        config = SegformerConfig.from_pretrained(model_name)
        config.num_labels = num_classes
        config.semantic_loss_ignore_index = -100
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, x):
        # SegFormer expects 3-channel input, but we have 1-channel CT
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        outputs = self.model(pixel_values=x)
        return outputs.logits

    def compute_loss(self, logits, targets):
        # Resize logits to match target size if needed
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode='bilinear', align_corners=False)
        if self.class_weights is not None:
            weights = self.class_weights.to(self.device)
            loss = F.cross_entropy(logits, targets.long(), weight=weights)
        else:
            loss = F.cross_entropy(logits, targets.long())
        return loss

    def training_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA]
        if x.dim() == 5:
            x = x[..., x.shape[-1] // 2]
            y = y[..., y.shape[-1] // 2]
        y = y.squeeze(1)
        logits = self(x)
        loss = self.compute_loss(logits, y)
        dice = dice_coefficient(logits, y, num_classes=self.num_classes)
        iou = jaccard_index(logits, y, num_classes=self.num_classes)
        bf1 = boundary_f1_score(logits, y, num_classes=self.num_classes)
        acc = accuracy(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', iou, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_bf1', bf1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA]
        if x.dim() == 5:
            x = x[..., x.shape[-1] // 2]
            y = y[..., y.shape[-1] // 2]
        y = y.squeeze(1)
        logits = self(x)
        loss = self.compute_loss(logits, y)
        dice = dice_coefficient(logits, y, num_classes=self.num_classes)
        iou = jaccard_index(logits, y, num_classes=self.num_classes)
        bf1 = boundary_f1_score(logits, y, num_classes=self.num_classes)
        acc = accuracy(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_bf1', bf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['image'][tio.DATA]
        y = batch['label'][tio.DATA]
        if x.dim() == 5:
            x = x[..., x.shape[-1] // 2]
            y = y[..., y.shape[-1] // 2]
        y = y.squeeze(1)
        logits = self(x)
        loss = self.compute_loss(logits, y)
        dice = dice_coefficient(logits, y, num_classes=self.num_classes)
        iou = jaccard_index(logits, y, num_classes=self.num_classes)
        bf1 = boundary_f1_score(logits, y, num_classes=self.num_classes)
        acc = accuracy(logits, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_dice', dice, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_iou', iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_bf1', bf1, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        } 