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
        self.model = model_class(num_classes=config.num_classes, in_chan=config.input_channels, **config.model_params)
        self.lightning_module = lightning_module_class.load_from_checkpoint(
            config.checkpoint_path, model=self.model, config=config, map_location=config.device)
        self.lightning_module.eval()
        self.lightning_module.to(config.device)

    def predict(self, dataloader: torch.utils.data.DataLoader, subject_list: List[dict]):
        os.makedirs(self.config.output_dir, exist_ok=True)
        with torch.no_grad():
            for i, (batch, subject) in enumerate(zip(dataloader, subject_list)):
                x = batch['image'][tio.DATA].to(self.config.device)
                logits = self.lightning_module(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                # If binary (tumor only), select only background and target_label channels
                if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                    logits = logits[:, [0, self.config.target_label], ...]
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                # If binary, map 1 -> target_label, 0 -> 0
                if hasattr(self.config, 'target_label') and self.config.target_label is not None:
                    pred = np.where(pred == 1, self.config.target_label, 0)
                # Save prediction as NIfTI
                affine = batch['image'][tio.AFFINE][0]
                subject_id = subject['subject_id']
                out_path = os.path.join(self.config.output_dir, f"{subject_id}_pred.nii.gz")
                nib.save(nib.Nifti1Image(pred[0].astype(np.uint8), affine), out_path) 