"""
model_training_config.py

Configuration class for model training parameters. Defines model architecture, data settings, training hyperparameters, optimizer settings, and logging configuration used in the training pipeline.

This configuration is used by the SegmentationTrainer and SegmentationLightningModule to control:
- Model architecture and parameters
- Data loading and batch processing
- Training hyperparameters (epochs, learning rate, etc.)
- Optimizer and scheduler settings
- Logging and checkpointing behavior
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelTrainingConfig:
    """
    Configuration class for model training parameters used in segmentation model training.

    This class defines all parameters needed for training segmentation models,
    including model settings, data configuration, training hyperparameters,
    optimizer settings, and logging configuration.

    Attributes:
        model_name (str): Name of the model architecture to use.
            Options: "dformer3d", "segformer", etc. Default: "dformer3d".
        model_params (Dict[str, Any]): Additional model-specific parameters.
            Default: empty dict.
        data_dir (str): Directory containing preprocessed training data.
            Default: "data_preprocessed".
        train_batch_size (int): Batch size for training data.
            Default: 1.
        val_batch_size (int): Batch size for validation data.
            Default: 1.
        num_workers (int): Number of workers for data loading.
            Default: 4.
        input_channels (int): Number of input channels for the model.
            Default: 1 (grayscale CT images).
        num_classes (int): Number of output classes for segmentation.
            Default: 14 (Medical Decathlon format).
        max_epochs (int): Maximum number of training epochs.
            Default: 100.
        gpus (int): Number of GPUs to use for training.
            Default: 1.
        precision (str): Training precision ("16-mixed", "32", etc.).
            Default: "16-mixed" for mixed precision training.
        seed (int): Random seed for reproducibility.
            Default: 42.
        log_every_n_steps (int): Log metrics every N training steps.
            Default: 10.
        checkpoint_dir (str): Directory to save model checkpoints.
            Default: "checkpoints".
        monitor_metric (str): Metric to monitor for checkpointing.
            Default: "val_loss".
        monitor_mode (str): Mode for monitoring ("min" or "max").
            Default: "min" (minimize validation loss).
        early_stopping_patience (int): Number of epochs to wait before early stopping.
            Default: 30.
        optimizer (str): Optimizer to use ("adam", "adamw", "sgd").
            Default: "adam".
        learning_rate (float): Learning rate for training.
            Default: 1e-4.
        weight_decay (float): Weight decay for regularization.
            Default: 1e-5.
        scheduler (Optional[str]): Learning rate scheduler ("step", "plateau").
            Default: None.
        scheduler_params (Optional[Dict[str, Any]]): Scheduler-specific parameters.
            Default: empty dict.
        log_dir (str): Directory to save training logs.
            Default: "logs".
        project_name (str): Name of the training project.
            Default: "ct-segmentation".
        run_name (Optional[str]): Specific run name for logging.
            Default: None.
        resume_from_checkpoint (Optional[str]): Path to checkpoint to resume from.
            Default: None.
        debug (bool): Enable debug mode for additional logging.
            Default: False.
    """
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
    target_label: Optional[int] = None  # For single-class training (e.g., tumor=2)

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
