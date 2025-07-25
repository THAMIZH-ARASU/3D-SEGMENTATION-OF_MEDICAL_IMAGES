import argparse
import json
from configs.model_evaluation_config import ModelEvaluationConfig
from pipelines.model_evaluation_pipeline import run_evaluation_pipeline
from utils.logging_utils import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CT Segmentation Evaluation")
    parser.add_argument('--config', type=str, help='Path to evaluation config JSON (optional)')
    parser.add_argument('--pred_dir', type=str, help='Directory with predicted segmentations')
    parser.add_argument('--gt_dir', type=str, help='Directory with ground truth segmentations')
    parser.add_argument('--csv_path', type=str, help='Path to save CSV results')
    parser.add_argument('--save_csv', action='store_true', help='Save results to CSV')
    parser.add_argument('--print_summary', action='store_true', help='Print summary to console')
    parser.add_argument('--subject_ids', type=str, nargs='+', help='List of subject IDs to evaluate')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ModelEvaluationConfig(**config_dict)
    else:
        config = ModelEvaluationConfig()
    # CLI overrides
    if args.pred_dir:
        config.pred_dir = args.pred_dir
    if args.gt_dir:
        config.gt_dir = args.gt_dir
    if args.csv_path:
        config.csv_path = args.csv_path
    if args.save_csv:
        config.save_csv = True
    if args.print_summary:
        config.print_summary = True
    if args.subject_ids:
        config.subject_ids = args.subject_ids
    run_evaluation_pipeline(config)

if __name__ == "__main__":
    main()

# Example Usage
# python run_evaluation.py --pred_dir predictions --gt_dir data_preprocessed/test --print_summary --save_csv