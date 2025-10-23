"""
Configuration class for model evaluation parameters. Defines paths, evaluation metrics, label mappings, and output settings used in the evaluation pipeline.

This configuration is used by the Evaluator class to control:
- Data paths for predictions and ground truth
- Label mappings for different tissue types
- Evaluation metrics to compute
- Output settings and file paths
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelEvaluationConfig:
    """
    Configuration class for model evaluation parameters used in segmentation evaluation.

    This class defines all parameters needed for evaluating segmentation model
    performance, including data paths, label mappings, evaluation metrics,
    and output settings.

    Attributes:
        pred_dir (str): Directory containing prediction files.
            Default: "predictions".
        gt_dir (str): Directory containing ground truth label files.
            Default: "data_preprocessed/test".
        pred_suffix (str): Suffix for prediction files.
            Default: "_pred.nii.gz".
        gt_suffix (str): Suffix for ground truth files.
            Default: "_label.nii.gz".
        subject_ids (Optional[List[str]]): List of specific subject IDs to evaluate.
            If None, evaluates all subjects in pred_dir. Default: None.
        liver_label (int): Label value for liver tissue in ground truth.
            Default: 1.
        tumor_label (int): Label value for tumor tissue in ground truth.
            Default: 2.
        background_label (int): Label value for background in ground truth.
            Default: 0.
        num_classes (int): Total number of classes in the segmentation task.
            Default: 3 (background, liver, tumor).
        metrics (List[str]): List of metrics to compute during evaluation.
            Options: "dice", "iou", "bf1", "accuracy". Default: all four.
        batch_size (int): Batch size for evaluation processing.
            Default: 1.
        device (str): Device to use for evaluation ("cpu" or "cuda").
            Default: "cpu".
        save_csv (bool): Whether to save evaluation results to CSV file.
            Default: True.
        csv_path (str): Path for saving evaluation results CSV.
            Default: "evaluation_results.csv".
        print_summary (bool): Whether to print evaluation summary to console.
            Default: True.
        debug (bool): Enable debug mode for additional logging.
            Default: False.
    """
    # Data
    pred_dir: str = "predictions"
    gt_dir: str = "data_preprocessed/test"
    pred_suffix: str = "_pred.nii.gz"
    gt_suffix: str = "_label.nii.gz"
    subject_ids: Optional[List[str]] = None  # If None, evaluate all in pred_dir

    # Evaluation
    liver_label: int = 1
    tumor_label: int = 2
    background_label: int = 0
    num_classes: int = 3
    metrics: List[str] = field(default_factory=lambda: ["dice", "iou", "bf1", "accuracy"])
    batch_size: int = 1
    device: str = "cpu"

    # Output
    save_csv: bool = True
    csv_path: str = "evaluation_results.csv"
    print_summary: bool = True
    debug: bool = False
