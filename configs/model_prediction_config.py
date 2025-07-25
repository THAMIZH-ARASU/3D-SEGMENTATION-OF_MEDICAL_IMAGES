from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelPredictionConfig:
    # Model
    model_name: str = "dformer3d"
    model_params: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str = "checkpoints/last.ckpt"

    # Data
    input_dir: str = "data_preprocessed/test"
    output_dir: str = "predictions"
    batch_size: int = 1
    num_workers: int = 2
    input_channels: int = 1
    num_classes: int = 14

    # Device
    gpus: int = 1
    device: str = "cuda"

    # Misc
    save_probabilities: bool = False
    save_logits: bool = False
    debug: bool = False 