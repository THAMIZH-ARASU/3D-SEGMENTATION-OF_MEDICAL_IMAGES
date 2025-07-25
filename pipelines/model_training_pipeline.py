import importlib
from configs.model_training_config import ModelTrainingConfig
from pipelines.data_loading_pipeline import get_dataloader
from model_training.trainer import SegmentationTrainer
from model_training.lightning_module import SegmentationLightningModule
from models.transformers.segformer.segformer_module import SegFormerLightningModule

# Model registry for extensibility
def get_model_class(model_name: str):
    registry = {
        'dformer3d': 'models.transformers.d_former.network.SegNetwork',
        'segformer': 'models.transformers.segformer.segformer_module.SegFormerLightningModule',
        # 'unet3d': 'models.cnns.unet_3d.UNet3D',  # Example for future
    }
    if model_name not in registry:
        raise ValueError(f"Model {model_name} not registered.")
    module_path, class_name = registry[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def run_training_pipeline(config: ModelTrainingConfig):
    # Set seed for reproducibility
    import pytorch_lightning as pl
    pl.seed_everything(config.seed)

    # Instantiate model
    ModelClass = get_model_class(config.model_name)
    if config.model_name == 'segformer':
        lightning_model = ModelClass(num_classes=config.num_classes, learning_rate=config.learning_rate, **config.model_params)
    else:
        base_model = ModelClass(num_classes=config.num_classes, in_chan=config.input_channels, **config.model_params)
        lightning_model = SegmentationLightningModule(base_model, config)

    # Data loaders
    train_loader = get_dataloader(config.data_dir, 'train', config.train_batch_size, config.num_workers, shuffle=True)
    val_loader = get_dataloader(config.data_dir, 'val', config.val_batch_size, config.num_workers, shuffle=False)
    test_loader = get_dataloader(config.data_dir, 'test', config.val_batch_size, config.num_workers, shuffle=False)

    # Trainer
    trainer = SegmentationTrainer(config, lightning_model, train_loader, val_loader, test_loader)
    trainer.fit()
    trainer.test()
