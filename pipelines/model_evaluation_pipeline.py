from configs.model_evaluation_config import ModelEvaluationConfig
from model_evaluation.evaluator import Evaluator

def run_evaluation_pipeline(config: ModelEvaluationConfig):
    evaluator = Evaluator(config)
    evaluator.evaluate()
    evaluator.save_results()
    if config.print_summary:
        evaluator.print_summary()
