from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelTrainingConfig:
    # Model
    model_name: str = "dformer3d"  # e.g., 'dformer3d', 'unet3d', etc.
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Data
    data_dir: str = "data_preprocessed"
    train_batch_size: int = 1
    val_batch_size: int = 1
    num_workers: int = 4
    input_channels: int = 1
    num_classes: int = 14

    # Training
    max_epochs: int = 100
    gpus: int = 1
    precision: str = "16-mixed"  # for mixed precision
    seed: int = 42
    log_every_n_steps: int = 10
    checkpoint_dir: str = "checkpoints"
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    early_stopping_patience: int = 30

    # Optimizer
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: Optional[str] = None
    scheduler_params: Optional[Dict[str, Any]] = field(default_factory=dict)

    # Logging
    log_dir: str = "logs"
    project_name: str = "ct-segmentation"
    run_name: Optional[str] = None

    # Misc
    resume_from_checkpoint: Optional[str] = None
    debug: bool = False
