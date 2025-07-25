from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelEvaluationConfig:
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
