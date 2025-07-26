"""
model_prediction_config.py

Configuration class for model prediction parameters. Defines model settings, data paths, device configuration, and output settings used in the prediction pipeline.

This configuration is used by the Predictor class to control:
- Model architecture and checkpoint loading
- Input/output data paths and processing
- Device settings for inference
- Output format and saving options
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelPredictionConfig:
    """
    Configuration class for model prediction parameters used in segmentation inference.

    This class defines all parameters needed for running inference with trained
    segmentation models, including model settings, data paths, device configuration,
    and output options.

    Attributes:
        model_name (str): Name of the model architecture to use.
            Options: "dformer3d", "segformer", etc. Default: "dformer3d".
        model_params (Dict[str, Any]): Additional model-specific parameters.
            Default: empty dict.
        checkpoint_path (str): Path to the trained model checkpoint.
            Default: "checkpoints/last.ckpt".
        input_dir (str): Directory containing input images for prediction.
            Default: "data_preprocessed/test".
        output_dir (str): Directory to save prediction outputs.
            Default: "predictions".
        batch_size (int): Batch size for prediction processing.
            Default: 1.
        num_workers (int): Number of workers for data loading.
            Default: 2.
        input_channels (int): Number of input channels for the model.
            Default: 1 (grayscale CT images).
        num_classes (int): Number of output classes for segmentation.
            Default: 14 (Medical Decathlon format).
        target_label (Optional[int]): If set, maps binary output to this specific label.
            Useful for binary segmentation tasks. Default: None.
        gpus (int): Number of GPUs to use for prediction.
            Default: 1.
        device (str): Device to use for prediction ("cpu" or "cuda").
            Default: "cuda".
        save_probabilities (bool): Whether to save probability maps.
            Default: False.
        save_logits (bool): Whether to save raw logits.
            Default: False.
        debug (bool): Enable debug mode for additional logging.
            Default: False.
    """
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
    target_label: Optional[int] = None  # If set, map binary output to this label

    # Device
    gpus: int = 1
    device: str = "cuda"

    # Misc
    save_probabilities: bool = False
    save_logits: bool = False
    debug: bool = False 