import argparse
import json
from configs.model_training_config import ModelTrainingConfig
from pipelines.model_training_pipeline import run_training_pipeline
import torch
torch.set_float32_matmul_precision('high')
from utils.logging_utils import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CT Segmentation Training")
    parser.add_argument('--config', type=str, help='Path to training config JSON (optional)')
    parser.add_argument('--model_name', type=str, help='Model name to use (overrides config)')
    parser.add_argument('--run_name', type=str, help='Run name for logging')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use')
    parser.add_argument('--max_epochs', type=int, help='Max epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--target_label', type=int, help='Unique value for the target class (e.g., tumor=2)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ModelTrainingConfig(**config_dict)
    else:
        config = ModelTrainingConfig()
    # CLI overrides
    if args.model_name:
        config.model_name = args.model_name
    if args.run_name:
        config.run_name = args.run_name
    if args.gpus is not None:
        config.gpus = args.gpus
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.resume_from_checkpoint:
        config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.target_label is not None:
        config.target_label = args.target_label
    run_training_pipeline(config)

if __name__ == "__main__":
    main()

# Example usage: (For both liver and tumor)
# python run_training.py --model_name dformer3d --gpus 1 --run_name my_run

# Example usage: (For tumor only)
# python run_training.py --model_name dformer3d --gpus 1 --run_name tumor_only --target_label 2

# python run_training.py --model_name segformer --gpus 1 --run_name segformer_run --target_label 2