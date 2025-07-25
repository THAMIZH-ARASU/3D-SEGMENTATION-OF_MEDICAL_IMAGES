import os
import torch
import torchio as tio
import pytorch_lightning as pl
import numpy as np
from typing import List
import nibabel as nib

class Predictor:
    def __init__(self, lightning_module_class, config, model_class):
        self.config = config
        if model_class is not None:
            self.model = model_class(num_classes=config.num_classes, in_chan=config.input_channels, **config.model_params)
            self.lightning_module = lightning_module_class.load_from_checkpoint(
                config.checkpoint_path, model=self.model, config=config, map_location=config.device)
        else:
            # For SegFormer, instantiate LightningModule directly
            self.lightning_module = lightning_module_class.load_from_checkpoint(
                config.checkpoint_path, config=config, map_location=config.device)
        self.lightning_module.eval()
        self.lightning_module.to(config.device)

    def predict(self, dataloader: torch.utils.data.DataLoader, subject_list: List[dict]):
        os.makedirs(self.config.output_dir, exist_ok=True)
        with torch.no_grad():
            for i, (batch, subject) in enumerate(zip(dataloader, subject_list)):
                x = batch['image'][tio.DATA].to(self.config.device)
                shape = x.shape
                # If x is 5D (B, C, H, W, D), predict slice by slice along D
                if x.dim() == 5:
                    preds = []
                    for d in range(x.shape[-1]):
                        x_slice = x[..., d]  # (B, C, H, W)
                        if x_slice.dim() == 3:
                            x_slice = x_slice.unsqueeze(0)
                        if x_slice.shape[1] == 1:
                            x_slice = x_slice.repeat(1, 3, 1, 1)
                        logits = self.lightning_module(x_slice)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                            logits = logits[:, [0, self.config.target_label], ...]
                        pred_slice = torch.argmax(logits, dim=1).cpu().numpy()
                        if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                            pred_slice = np.where(pred_slice == 1, self.config.target_label, 0)
                        preds.append(pred_slice[0])  # [0] for batch size 1
                    pred = np.stack(preds, axis=-1)  # (H, W, D)
                else:
                    # Existing logic for 2D/3D
                    if x.dim() == 2:
                        x = x.unsqueeze(0).unsqueeze(0)
                    elif x.dim() == 3:
                        if x.shape[0] in [1, 3]:
                            x = x.unsqueeze(0)
                        else:
                            x = x.unsqueeze(0).unsqueeze(0)
                    elif x.dim() == 4:
                        if x.shape[1] != 1 and x.shape[1] != 3:
                            if x.shape[-3] in [1, 3]:
                                x = x.permute(0, 3, 1, 2)
                            else:
                                x = x.unsqueeze(1)
                    if x.dim() != 4:
                        raise RuntimeError(f"Input to model must be 4D (B, C, H, W), got {x.shape}")
                    if x.shape[1] == 1:
                        x = x.repeat(1, 3, 1, 1)
                    logits = self.lightning_module(x)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                        logits = logits[:, [0, self.config.target_label], ...]
                    pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
                    if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                        pred = np.where(pred == 1, self.config.target_label, 0)
                # Save prediction as NIfTI
                affine = batch['image'][tio.AFFINE][0]
                subject_id = subject['subject_id']
                out_path = os.path.join(self.config.output_dir, f"{subject_id}_pred.nii.gz")
                nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), out_path) 