import importlib
from configs.model_prediction_config import ModelPredictionConfig
from model_prediction.data import get_prediction_loader
from model_prediction.predictor import Predictor
from model_training.lightning_module import SegmentationLightningModule
from models.transformers.segformer.segformer_module import SegFormerLightningModule

# Model registry for extensibility
def get_model_class(model_name: str):
    registry = {
        'dformer3d': 'models.transformers.d_former.network.SegNetwork',
        'segformer': 'models.transformers.segformer.segformer_module.SegFormerLightningModule',
        # 'unet3d': 'models.cnns.unet_3d.UNet3D',
    }
    if model_name not in registry:
        raise ValueError(f"Model {model_name} not registered.")
    module_path, class_name = registry[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_prediction_pipeline(config: ModelPredictionConfig):
    loader, subject_list = get_prediction_loader(
        config.input_dir, batch_size=config.batch_size, num_workers=config.num_workers)
    ModelClass = get_model_class(config.model_name)
    if config.model_name == 'segformer':
        predictor = Predictor(SegFormerLightningModule, config, None)
    else:
        predictor = Predictor(SegmentationLightningModule, config, ModelClass)
    predictor.predict(loader, subject_list)
