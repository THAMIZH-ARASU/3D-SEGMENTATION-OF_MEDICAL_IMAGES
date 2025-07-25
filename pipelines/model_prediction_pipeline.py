import importlib
from configs.model_prediction_config import ModelPredictionConfig
from model_prediction.data import get_prediction_loader
from model_prediction.predictor import Predictor
from model_training.lightning_module import SegmentationLightningModule
from models.transformers.segformer.segformer_module import SegFormerModule

# Model registry for extensibility
def get_model_class(model_name: str):
    registry = {
        'dformer3d': 'models.transformers.d_former.network.SegNetwork',
        # 'unet3d': 'models.cnns.unet_3d.UNet3D',
        'segformer': SegFormerModule,
    }
    if model_name not in registry:
        raise ValueError(f"Model {model_name} not registered.")
    model_entry = registry[model_name]
    if isinstance(model_entry, str):
        module_path, class_name = model_entry.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        return model_entry

def run_prediction_pipeline(config: ModelPredictionConfig):
    # Data loader
    loader, subject_list = get_prediction_loader(
        config.input_dir, batch_size=config.batch_size, num_workers=config.num_workers)
    # Model
    ModelClass = get_model_class(config.model_name)
    if config.model_name == 'segformer':
        predictor = Predictor(SegmentationLightningModule, config, lambda **kwargs: ModelClass(num_classes=config.num_classes, **config.model_params))
    else:
        predictor = Predictor(SegmentationLightningModule, config, lambda **kwargs: ModelClass(num_classes=config.num_classes, in_chan=config.input_channels, **config.model_params))
    predictor.predict(loader, subject_list)
