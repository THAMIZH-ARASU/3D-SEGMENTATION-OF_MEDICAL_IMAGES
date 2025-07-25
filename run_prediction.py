import argparse
import json
from configs.model_prediction_config import ModelPredictionConfig
from pipelines.model_prediction_pipeline import run_prediction_pipeline
from utils.logging_utils import setup_logging
import logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="CT Segmentation Prediction")
    parser.add_argument('--config', type=str, help='Path to prediction config JSON (optional)')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, help='Input directory with images')
    parser.add_argument('--output_dir', type=str, help='Output directory for predictions')
    parser.add_argument('--batch_size', type=int, help='Batch size for prediction')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to use')
    parser.add_argument('--device', type=str, help='Device to use (cuda or cpu)')
    parser.add_argument('--model_name', type=str, help='Model name to use')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ModelPredictionConfig(**config_dict)
    else:
        config = ModelPredictionConfig()
    # CLI overrides
    if args.checkpoint_path:
        config.checkpoint_path = args.checkpoint_path
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.gpus is not None:
        config.gpus = args.gpus
    if args.device:
        config.device = args.device
    if args.model_name:
        config.model_name = args.model_name
    run_prediction_pipeline(config)

if __name__ == "__main__":
    main()

#Example usage:
# python run_prediction.py --checkpoint_path checkpoints/last.ckpt --input_dir data_preprocessed/test --output_dir predictions --device cuda --model_name dformer3d