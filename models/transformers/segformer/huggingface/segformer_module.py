import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormerModule(nn.Module):
    def __init__(self, model_name_or_path="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=3, **kwargs):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name_or_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        # x: (B, C, H, W) - expects 2D images, but for 3D, you may need to adapt or slice
        # For 3D, you may need to loop over slices or adapt the model
        # Here, we assume 2D for compatibility with HuggingFace SegFormer
        outputs = self.model(pixel_values=x)
        # outputs.logits: (B, num_classes, H, W)
        return outputs.logits

# Optional: function to get a SegFormerModule with custom weights

def get_pretrained_segformer(model_name_or_path, num_classes):
    return SegFormerModule(model_name_or_path=model_name_or_path, num_classes=num_classes) 