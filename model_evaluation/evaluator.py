"""
Model evaluation module for medical image segmentation. Provides comprehensive evaluation of segmentation model performance using multiple metrics including Dice coefficient, Jaccard index, boundary F1 score, and accuracy.

This module handles loading of predictions and ground truth, computing evaluation
metrics for different tissue types (liver, tumor, whole), and saving results
to CSV format for further analysis.
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import logging
from model_evaluation.metrics import dice_coefficient, jaccard_index, boundary_f1_score, accuracy

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Comprehensive evaluator for medical image segmentation models.

    This class provides a complete evaluation pipeline for segmentation models,
    computing multiple metrics (Dice, IoU, Boundary F1, Accuracy) for different
    tissue types and saving results for analysis.

    Attributes:
        config: Evaluation configuration object containing paths and parameters.
        results: List of evaluation results for each subject.
    """

    def __init__(self, config):
        """
        Initialize the Evaluator with configuration.

        Args:
            config: Evaluation configuration object containing:
                - pred_dir: Directory with prediction files
                - gt_dir: Directory with ground truth files
                - pred_suffix, gt_suffix: File naming conventions
                - liver_label, tumor_label: Label values for different tissues
                - Other evaluation parameters
        """
        self.config = config
        self.results = []

    def evaluate_subject(self, pred_path, gt_path, subject_id):
        """
        Evaluate a single subject by computing metrics for different tissue types.

        This method loads prediction and ground truth files, computes metrics
        for liver, tumor, and whole (liver + tumor) regions, and stores the
        results for later analysis.

        Args:
            pred_path (str): Path to the prediction NIfTI file.
            gt_path (str): Path to the ground truth NIfTI file.
            subject_id (str): Unique identifier for the subject.

        Returns:
            dict: Dictionary containing computed metrics for the subject.

        Note:
            Metrics are computed for:
            - Liver tissue (config.liver_label)
            - Tumor tissue (config.tumor_label)
            - Whole region (liver + tumor combined)
        """
        pred = np.asarray(nib.load(pred_path).dataobj)
        gt = np.asarray(nib.load(gt_path).dataobj)
        metrics = {}
        for label, name in zip([self.config.liver_label, self.config.tumor_label], ["liver", "tumor"]):
            metrics[f"dice_{name}"] = dice_coefficient(pred, gt, label)
            metrics[f"iou_{name}"] = jaccard_index(pred, gt, label)
            metrics[f"bf1_{name}"] = boundary_f1_score(pred, gt, label)
            metrics[f"accuracy_{name}"] = accuracy(pred, gt, label)
        # Whole (liver or tumor)
        mask_pred = (pred == self.config.liver_label) | (pred == self.config.tumor_label)
        mask_gt = (gt == self.config.liver_label) | (gt == self.config.tumor_label)
        metrics["dice_whole"] = dice_coefficient(mask_pred, mask_gt, 1)
        metrics["iou_whole"] = jaccard_index(mask_pred, mask_gt, 1)
        metrics["bf1_whole"] = boundary_f1_score(mask_pred, mask_gt, 1)
        metrics["accuracy_whole"] = accuracy(mask_pred, mask_gt, 1)
        metrics["subject_id"] = subject_id
        logger.debug(f"Metrics for {subject_id}: {metrics}")
        self.results.append(metrics)
        return metrics

    def evaluate(self):
        """
        Evaluate all subjects in the prediction directory.

        This method scans the prediction directory for files matching the
        configured suffix, finds corresponding ground truth files, and
        evaluates each subject. It handles missing ground truth files
        gracefully with warnings.

        Note:
            The method processes all prediction files unless specific
            subject_ids are provided in the configuration.
        """
        logger.info(f"Evaluating predictions in {self.config.pred_dir}")
        pred_files = [f for f in os.listdir(self.config.pred_dir) if f.endswith(self.config.pred_suffix)]
        if self.config.subject_ids:
            pred_files = [f for f in pred_files if any(sid in f for sid in self.config.subject_ids)]
        for pred_file in pred_files:
            subject_id = pred_file.replace(self.config.pred_suffix, "")
            pred_path = os.path.join(self.config.pred_dir, pred_file)
            gt_path = os.path.join(self.config.gt_dir, f"{subject_id}{self.config.gt_suffix}")
            if not os.path.exists(gt_path):
                logger.warning(f"GT not found for {subject_id}")
                continue
            self.evaluate_subject(pred_path, gt_path, subject_id)
        logger.info("Evaluation complete.")

    def save_results(self):
        """
        Save evaluation results to CSV file.

        This method converts the collected evaluation results to a pandas
        DataFrame and saves it to the configured CSV path for further
        analysis and reporting.

        Note:
            Results are only saved if save_csv is enabled in the configuration.
        """
        if self.config.save_csv:
            df = pd.DataFrame(self.results)
            df.to_csv(self.config.csv_path, index=False)
            logger.info(f"Saved evaluation results to {self.config.csv_path}")

    def print_summary(self):
        """
        Print a formatted summary of evaluation results to console.

        This method computes and displays average metrics across all subjects
        for each tissue type (liver, tumor, whole) and each metric type
        (Dice, IoU, Boundary F1, Accuracy).

        Note:
            If no results are available, a warning message is displayed.
        """
        if not self.results:
            logger.warning("No results to summarize.")
            print("No results to summarize.")
            return
        df = pd.DataFrame(self.results)
        print("\n===== Evaluation Summary =====")
        for region in ["liver", "tumor", "whole"]:
            print(f"\n--- {region.capitalize()} ---")
            for metric in ["dice", "iou", "bf1", "accuracy"]:
                col = f"{metric}_{region}"
                if col in df:
                    avg = df[col].mean()
                    print(f"{metric.capitalize()}: {avg:.4f}")
        print("\n=============================") 